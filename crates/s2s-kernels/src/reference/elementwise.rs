//! Element-wise activations and gated linear units, forward only.
//!
//! These cover every activation a Kyutai (Moshi / DSM) transformer config can
//! name, with the exact scalar formulas of candle's CPU backend so that ports
//! of candle models agree to rounding:
//!
//! | [`Activation`] | `f(x)` |
//! |---|---|
//! | `Silu` | `x / (1 + e^-x)` |
//! | `GeluErf` | `x/2 · (1 + erf(x/√2))`, evaluated in `f64` |
//! | `GeluTanh` | `x/2 · (1 + tanh(√(2/π) · x · (1 + 0.044715 x²)))` |
//! | `Relu`, `Relu2`, `Relu6` | `max(x, 0)`, its square, `clamp(x, 0, 6)` |
//! | `Sigmoid`, `HardSigmoid` | `1 / (1 + e^-x)`, `clamp((x + 3) / 6, 0, 1)` |
//! | `Swish`, `HardSwish` | `x · sigmoid(x)`, `x · hard_sigmoid(x)` |
//! | `Mish` | `x · tanh(ln(1 + e^x))` |
//! | `Elu(α)`, `LeakyRelu(s)` | `x ≥ 0 ? x : α(e^x − 1)`, `max(x, 0) + s · min(x, 0)` |
//!
//! [`glu_fwd`] is the gated MLP core `h = act(gate) * up` (SwiGLU when `act`
//! is SiLU). Backward passes for the activations the trainer uses live in
//! [`activation`](super::activation).

use s2s_core::{cast, Float};

/// An element-wise activation function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    /// `x * sigmoid(x)`, written `x / (1 + exp(-x))`.
    Silu,
    /// Exact GELU with `erf`.
    GeluErf,
    /// Tanh approximation of GELU (`gelu_new` / `gelu_pytorch_tanh`).
    GeluTanh,
    /// `max(x, 0)`.
    Relu,
    /// `max(x, 0)^2`.
    Relu2,
    /// `clamp(x, 0, 6)`.
    Relu6,
    /// Logistic sigmoid.
    Sigmoid,
    /// `clamp((x + 3) / 6, 0, 1)`.
    HardSigmoid,
    /// `x * sigmoid(x)`, written as a product (same value as `Silu` up to rounding).
    Swish,
    /// `x * hard_sigmoid(x)`.
    HardSwish,
    /// `x * tanh(softplus(x))`.
    Mish,
    /// Exponential linear unit with the given `alpha`.
    Elu(f64),
    /// Leaky ReLU with the given negative slope.
    LeakyRelu(f64),
}

fn sigmoid<T: Float>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

fn hard_sigmoid<T: Float>(x: T) -> T {
    let (three, six): (T, T) = (cast(3.0), cast(6.0));
    ((x + three) / six).max(T::zero()).min(T::one())
}

/// Apply `act` to one value.
pub fn activation<T: Float>(act: Activation, x: T) -> T {
    let zero = T::zero();
    match act {
        Activation::Silu => x / (T::one() + (-x).exp()),
        Activation::GeluErf => {
            let v = x.to_f64().unwrap_or(f64::NAN);
            cast((erf(v / std::f64::consts::SQRT_2) + 1.0) * 0.5 * v)
        }
        Activation::GeluTanh => {
            let half: T = cast(0.5);
            let c: T = cast((2.0 / std::f64::consts::PI).sqrt());
            let k: T = cast(0.044715);
            half * x * (T::one() + (c * x * (T::one() + k * x * x)).tanh())
        }
        Activation::Relu => x.max(zero),
        Activation::Relu2 => {
            let r = x.max(zero);
            r * r
        }
        Activation::Relu6 => x.max(zero).min(cast(6.0)),
        Activation::Sigmoid => sigmoid(x),
        Activation::HardSigmoid => hard_sigmoid(x),
        Activation::Swish => x * sigmoid(x),
        Activation::HardSwish => x * hard_sigmoid(x),
        Activation::Mish => x * (x.exp() + T::one()).ln().tanh(),
        Activation::Elu(alpha) => {
            if x >= zero {
                x
            } else {
                cast::<T>(alpha) * (x.exp() - T::one())
            }
        }
        Activation::LeakyRelu(slope) => x.max(zero) + x.min(zero) * cast(slope),
    }
}

/// Apply `act` in place to every element of `x`.
pub fn activation_fwd<T: Float>(act: Activation, x: &mut [T]) {
    for v in x.iter_mut() {
        *v = activation(act, *v);
    }
}

/// Gated linear unit forward: `h = act(gate) * up`, element-wise over
/// equal-length buffers.
pub fn glu_fwd<T: Float>(act: Activation, gate: &[T], up: &[T], h: &mut [T]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), h.len());
    for i in 0..h.len() {
        h[i] = activation(act, gate[i]) * up[i];
    }
}

/// The error function (absolute error below `1e-10`).
///
/// Rational approximations from Boost, as ported by the `statrs` crate (MIT)
/// and vendored by candle (MIT/Apache-2.0); same coefficients and evaluation
/// order, so results match candle bit-for-bit (including its ~5e-11 error
/// around `0.5 <= x < 1`, far below `f32` resolution).
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x.is_infinite() {
        x.signum()
    } else if x == 0.0 {
        0.0
    } else if x < 0.0 {
        -erf_pos(-x)
    } else {
        erf_pos(x)
    }
}

/// Polynomial `coeff[0] + coeff[1] z + ...` by Horner's rule.
fn polynomial(z: f64, coeff: &[f64]) -> f64 {
    let n = coeff.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum = coeff[n - 1];
    for c in coeff[..n - 1].iter().rev() {
        sum = *c + z * sum;
    }
    sum
}

/// `erf(z)` for `z > 0` (statrs' `erf_impl(z, false)`).
fn erf_pos(z: f64) -> f64 {
    let ratio = |off: f64, n: &[f64], d: &[f64]| polynomial(z - off, n) / polynomial(z - off, d);
    if z < 0.5 {
        return if z < 1e-10 {
            z * 1.125 + z * 0.0033791670955125737
        } else {
            z * 1.125 + z * polynomial(z, ERF_IMPL_AN) / polynomial(z, ERF_IMPL_AD)
        };
    }
    if z >= 110.0 {
        return 1.0;
    }
    let (r, b) = if z < 0.75 {
        (ratio(0.5, ERF_IMPL_BN, ERF_IMPL_BD), 0.3440242112)
    } else if z < 1.25 {
        (ratio(0.75, ERF_IMPL_CN, ERF_IMPL_CD), 0.419990927)
    } else if z < 2.25 {
        (ratio(1.25, ERF_IMPL_DN, ERF_IMPL_DD), 0.4898625016)
    } else if z < 3.5 {
        (ratio(2.25, ERF_IMPL_EN, ERF_IMPL_ED), 0.5317370892)
    } else if z < 5.25 {
        (ratio(3.5, ERF_IMPL_FN, ERF_IMPL_FD), 0.5489973426)
    } else if z < 8.0 {
        (ratio(5.25, ERF_IMPL_GN, ERF_IMPL_GD), 0.5571740866)
    } else if z < 11.5 {
        (ratio(8.0, ERF_IMPL_HN, ERF_IMPL_HD), 0.5609807968)
    } else if z < 17.0 {
        (ratio(11.5, ERF_IMPL_IN, ERF_IMPL_ID), 0.5626493692)
    } else if z < 24.0 {
        (ratio(17.0, ERF_IMPL_JN, ERF_IMPL_JD), 0.5634598136)
    } else if z < 38.0 {
        (ratio(24.0, ERF_IMPL_KN, ERF_IMPL_KD), 0.5638477802)
    } else if z < 60.0 {
        (ratio(38.0, ERF_IMPL_LN, ERF_IMPL_LD), 0.5640528202)
    } else if z < 85.0 {
        (ratio(60.0, ERF_IMPL_MN, ERF_IMPL_MD), 0.5641309023)
    } else {
        (ratio(85.0, ERF_IMPL_NN, ERF_IMPL_ND), 0.5641584396)
    };
    let g = (-z * z).exp() / z;
    1.0 - (g * b + g * r)
}

// Coefficients (statrs `erf.rs`, MIT; via candle-core `cpu/erf.rs`).
const ERF_IMPL_AN: &[f64] = &[
    0.0033791670955125737,
    -0.0007369565304816795,
    -0.3747323373929196,
    0.08174424487335873,
    -0.04210893199365486,
    0.007016570951209575,
    -0.004950912559824351,
    0.0008716465990379225,
];
const ERF_IMPL_AD: &[f64] = &[
    1.0,
    -0.21808821808792464,
    0.4125429727254421,
    -0.08418911478731067,
    0.06553388564002416,
    -0.012001960445494177,
    0.00408165558926174,
    -0.0006159007215577697,
];
const ERF_IMPL_BN: &[f64] = &[
    -0.03617903907182625,
    0.2922518834448827,
    0.2814470417976045,
    0.12561020886276694,
    0.027413502826893053,
    0.0025083967216806575,
];
const ERF_IMPL_BD: &[f64] = &[
    1.0,
    1.8545005897903486,
    1.4357580303783142,
    0.5828276587530365,
    0.12481047693294975,
    0.011372417654635328,
];
const ERF_IMPL_CN: &[f64] = &[
    -0.03978768926111369,
    0.1531652124678783,
    0.19126029560093624,
    0.10276327061989304,
    0.029637090615738836,
    0.004609348678027549,
    0.0003076078203486802,
];
const ERF_IMPL_CD: &[f64] = &[
    1.0,
    1.955200729876277,
    1.6476231719938486,
    0.7682386070221262,
    0.20979318593650978,
    0.031956931689991336,
    0.0021336316089578537,
];
const ERF_IMPL_DN: &[f64] = &[
    -0.030083856055794972,
    0.05385788298444545,
    0.07262115416519142,
    0.036762846988804936,
    0.009646290155725275,
    0.0013345348007529107,
    7.780875997825043e-05,
];
const ERF_IMPL_DD: &[f64] = &[
    1.0,
    1.7596709814716753,
    1.3288357143796112,
    0.5525285965087576,
    0.13379305694133287,
    0.017950964517628076,
    0.0010471244001993736,
    -1.0664038182035734e-08,
];
const ERF_IMPL_EN: &[f64] = &[
    -0.011790757013722784,
    0.01426213209053881,
    0.020223443590296084,
    0.009306682999904321,
    0.00213357802422066,
    0.00025022987386460105,
    1.2053491221958819e-05,
];
const ERF_IMPL_ED: &[f64] = &[
    1.0,
    1.5037622520362048,
    0.9653977862044629,
    0.3392652304767967,
    0.06897406495415698,
    0.007710602624917683,
    0.0003714211015310693,
];
const ERF_IMPL_FN: &[f64] = &[
    -0.005469547955387293,
    0.004041902787317071,
    0.005496336955316117,
    0.002126164726039454,
    0.0003949840144950839,
    3.655654770644424e-05,
    1.3548589710993232e-06,
];
const ERF_IMPL_FD: &[f64] = &[
    1.0,
    1.2101969777363077,
    0.6209146682211439,
    0.17303843066114277,
    0.027655081377343203,
    0.0024062597442430973,
    8.918118172513365e-05,
    -4.655288362833827e-12,
];
const ERF_IMPL_GN: &[f64] = &[
    -0.0027072253590577837,
    0.00131875634250294,
    0.0011992593326100233,
    0.00027849619811344664,
    2.6782298821833186e-05,
    9.230436723150282e-07,
];
const ERF_IMPL_GD: &[f64] = &[
    1.0,
    0.8146328085431416,
    0.26890166585629954,
    0.044987721610304114,
    0.0038175966332024847,
    0.00013157189788859692,
    4.048153596757641e-12,
];
const ERF_IMPL_HN: &[f64] = &[
    -0.001099467206917422,
    0.00040642544275042267,
    0.0002744994894169007,
    4.652937706466594e-05,
    3.2095542539576746e-06,
    7.782860181450209e-08,
];
const ERF_IMPL_HD: &[f64] = &[
    1.0,
    0.5881737106118461,
    0.13936333128940975,
    0.016632934041708368,
    0.0010002392131023491,
    2.4254837521587224e-05,
];
const ERF_IMPL_IN: &[f64] = &[
    -0.0005690799360109496,
    0.00016949854037376225,
    5.184723545811009e-05,
    3.8281931223192885e-06,
    8.249899312818944e-08,
];
const ERF_IMPL_ID: &[f64] = &[
    1.0,
    0.33963725005113937,
    0.04347264787031066,
    0.002485493352246371,
    5.356333053371529e-05,
    -1.1749094440545958e-13,
];
const ERF_IMPL_JN: &[f64] = &[
    -0.00024131359948399134,
    5.742249752025015e-05,
    1.1599896292738377e-05,
    5.817621344025938e-07,
    8.539715550856736e-09,
];
const ERF_IMPL_JD: &[f64] = &[
    1.0,
    0.23304413829968784,
    0.02041869405464403,
    0.0007971856475643983,
    1.1701928167017232e-05,
];
const ERF_IMPL_KN: &[f64] = &[
    -0.00014667469927776036,
    1.6266655211228053e-05,
    2.6911624850916523e-06,
    9.79584479468092e-08,
    1.0199464762572346e-09,
];
const ERF_IMPL_KD: &[f64] = &[
    1.0,
    0.16590781294484722,
    0.010336171619150588,
    0.0002865930263738684,
    2.9840157084090034e-06,
];
const ERF_IMPL_LN: &[f64] = &[
    -5.839057976297718e-05,
    4.125103251054962e-06,
    4.3179092242025094e-07,
    9.933651555900132e-09,
    6.534805100201047e-11,
];
const ERF_IMPL_LD: &[f64] = &[
    1.0,
    0.10507708607203992,
    0.004142784286754756,
    7.263387546445238e-05,
    4.778184710473988e-07,
];
const ERF_IMPL_MN: &[f64] = &[
    -1.9645779760922958e-05,
    1.572438876668007e-06,
    5.439025111927009e-08,
    3.174724923691177e-10,
];
const ERF_IMPL_MD: &[f64] = &[
    1.0,
    0.05280398924095763,
    0.0009268760691517533,
    5.410117232266303e-06,
    5.350938458036424e-16,
];
const ERF_IMPL_NN: &[f64] = &[
    -7.892247039787227e-06,
    6.22088451660987e-07,
    1.457284456768824e-08,
    6.037155055427153e-11,
];
const ERF_IMPL_ND: &[f64] = &[
    1.0,
    0.03753288463562937,
    0.0004679195359746253,
    1.9384703927584565e-06,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known_values() {
        // Reference values (mpmath, 50 digits, rounded to f64).
        let cases = [
            (0.0, 0.0),
            (1e-12, 1.1283791670955126e-12),
            (0.1, 0.1124629160182849),
            (0.5, 0.5204998778130465),
            (1.0, 0.8427007929497149),
            (2.0, 0.9953222650189527),
            (3.0, 0.9999779095030014),
            (6.0, 1.0),
        ];
        for (x, want) in cases {
            assert!((erf(x) - want).abs() < 1e-10, "erf({x}) = {}", erf(x));
            assert!((erf(-x) + want).abs() < 1e-10);
        }
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert_eq!(erf(f64::NEG_INFINITY), -1.0);
        assert!(erf(f64::NAN).is_nan());
    }

    #[test]
    fn gelu_variants_agree_roughly() {
        for i in -40..=40 {
            let x = i as f64 * 0.1;
            let a = activation(Activation::GeluErf, x);
            let b = activation(Activation::GeluTanh, x);
            assert!((a - b).abs() < 1e-3, "x = {x}: {a} vs {b}");
        }
        assert!((activation(Activation::GeluErf, 1.0f64) - 0.8413447460685429).abs() < 1e-10);
    }

    #[test]
    fn spot_values() {
        use Activation::*;
        let s1 = 1.0 / (1.0 + (-1.0f64).exp());
        assert!((activation(Silu, 1.0f64) - s1).abs() < 1e-15);
        assert!((activation(Swish, 1.0f64) - s1).abs() < 1e-15);
        assert_eq!(activation(Relu, -2.0f64), 0.0);
        assert_eq!(activation(Relu2, 3.0f64), 9.0);
        assert_eq!(activation(Relu6, 7.0f64), 6.0);
        assert_eq!(activation(HardSigmoid, 0.0f64), 0.5);
        assert_eq!(activation(HardSwish, 4.0f64), 4.0);
        assert!((activation(Elu(1.0), -1.0f64) - ((-1.0f64).exp() - 1.0)).abs() < 1e-15);
        assert_eq!(activation(LeakyRelu(0.1), -2.0f64), -0.2);
        let m = (1.0f64.exp() + 1.0).ln().tanh();
        assert!((activation(Mish, 1.0f64) - m).abs() < 1e-15);
    }

    #[test]
    fn glu_silu_matches_swiglu() {
        let gate = [-3.0f64, -0.5, 0.0, 0.7, 2.5];
        let up = [1.0, 2.0, -1.0, 0.5, -2.0];
        let mut a = [0.0; 5];
        let mut b = [0.0; 5];
        glu_fwd(Activation::Silu, &gate, &up, &mut a);
        super::super::activation::swiglu_fwd(&gate, &up, &mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-14);
        }
        let mut c = gate;
        activation_fwd(Activation::Relu, &mut c);
        assert_eq!(c, [0.0, 0.0, 0.0, 0.7, 2.5]);
    }
}
