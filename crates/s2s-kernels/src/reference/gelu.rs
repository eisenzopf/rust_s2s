//! Exact (erf-based) GELU, as used by the Mimi transformer MLPs.
//!
//! `gelu(x) = x / 2 * (1 + erf(x / sqrt 2))`, evaluated in `f64` and rounded
//! once to `T`, exactly like candle's `gelu_erf` CPU op. The `erf` below is
//! the rational-approximation implementation from
//! [statrs](https://github.com/statrs-dev/statrs) (MIT), as vendored in
//! candle (MIT/Apache-2.0). Coefficients are written as the shortest decimal
//! that round-trips to the same `f64`, so results match Kyutai's candle-based
//! Mimi bit for bit.

use s2s_core::{cast, Float};

/// Exact GELU of one value, computed in `f64`.
#[inline]
pub fn gelu_erf(v: f64) -> f64 {
    (erf(v / std::f64::consts::SQRT_2) + 1.0) * 0.5 * v
}

/// Exact GELU forward, element-wise: `y[i] = gelu(x[i])`.
pub fn gelu_erf_fwd<T: Float>(x: &[T], y: &mut [T]) {
    assert_eq!(x.len(), y.len(), "gelu_erf_fwd: length mismatch");
    for (d, &s) in y.iter_mut().zip(x) {
        *d = cast(gelu_erf(s.to_f64().unwrap_or(f64::NAN)));
    }
}

/// In-place [`gelu_erf_fwd`].
pub fn gelu_erf_inplace<T: Float>(x: &mut [T]) {
    for s in x.iter_mut() {
        *s = cast(gelu_erf(s.to_f64().unwrap_or(f64::NAN)));
    }
}

/// The error function.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x >= 0.0 && x.is_infinite() {
        1.0
    } else if x <= 0.0 && x.is_infinite() {
        -1.0
    } else if x == 0. {
        0.0
    } else {
        erf_impl(x, false)
    }
}

/// Evaluates the polynomial with coefficients `coeff` (constant term first) at `z`.
fn polynomial(z: f64, coeff: &[f64]) -> f64 {
    let n = coeff.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum = coeff[n - 1];
    for c in coeff[0..n - 1].iter().rev() {
        sum = *c + z * sum;
    }
    sum
}

/// Polynomial coefficients for a numerator of `erf_impl`
/// in the interval [1e-10, 0.5].
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

/// Polynomial coefficients for a denominator of `erf_impl`
/// in the interval [1e-10, 0.5]
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

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [0.5, 0.75].
const ERF_IMPL_BN: &[f64] = &[
    -0.03617903907182625,
    0.2922518834448827,
    0.2814470417976045,
    0.12561020886276694,
    0.027413502826893053,
    0.0025083967216806575,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [0.5, 0.75].
const ERF_IMPL_BD: &[f64] = &[
    1.0,
    1.8545005897903486,
    1.4357580303783142,
    0.5828276587530365,
    0.12481047693294975,
    0.011372417654635328,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [0.75, 1.25].
const ERF_IMPL_CN: &[f64] = &[
    -0.03978768926111369,
    0.1531652124678783,
    0.19126029560093624,
    0.10276327061989304,
    0.029637090615738836,
    0.004609348678027549,
    0.0003076078203486802,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [0.75, 1.25].
const ERF_IMPL_CD: &[f64] = &[
    1.0,
    1.955200729876277,
    1.6476231719938486,
    0.7682386070221262,
    0.20979318593650978,
    0.031956931689991336,
    0.0021336316089578537,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [1.25, 2.25].
const ERF_IMPL_DN: &[f64] = &[
    -0.030083856055794972,
    0.05385788298444545,
    0.07262115416519142,
    0.036762846988804936,
    0.009646290155725275,
    0.0013345348007529107,
    7.780875997825043e-05,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [1.25, 2.25].
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

///  Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [2.25, 3.5].
const ERF_IMPL_EN: &[f64] = &[
    -0.011790757013722784,
    0.01426213209053881,
    0.020223443590296084,
    0.009306682999904321,
    0.00213357802422066,
    0.00025022987386460105,
    1.2053491221958819e-05,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [2.25, 3.5].
const ERF_IMPL_ED: &[f64] = &[
    1.0,
    1.5037622520362048,
    0.9653977862044629,
    0.3392652304767967,
    0.06897406495415698,
    0.007710602624917683,
    0.0003714211015310693,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [3.5, 5.25].
const ERF_IMPL_FN: &[f64] = &[
    -0.005469547955387293,
    0.004041902787317071,
    0.005496336955316117,
    0.002126164726039454,
    0.0003949840144950839,
    3.655654770644424e-05,
    1.3548589710993232e-06,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [3.5, 5.25].
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

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [5.25, 8].
const ERF_IMPL_GN: &[f64] = &[
    -0.0027072253590577837,
    0.00131875634250294,
    0.0011992593326100233,
    0.00027849619811344664,
    2.6782298821833186e-05,
    9.230436723150282e-07,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [5.25, 8].
const ERF_IMPL_GD: &[f64] = &[
    1.0,
    0.8146328085431416,
    0.26890166585629954,
    0.044987721610304114,
    0.0038175966332024847,
    0.00013157189788859692,
    4.048153596757641e-12,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [8, 11.5].
const ERF_IMPL_HN: &[f64] = &[
    -0.001099467206917422,
    0.00040642544275042267,
    0.0002744994894169007,
    4.652937706466594e-05,
    3.2095542539576746e-06,
    7.782860181450209e-08,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [8, 11.5].
const ERF_IMPL_HD: &[f64] = &[
    1.0,
    0.5881737106118461,
    0.13936333128940975,
    0.016632934041708368,
    0.0010002392131023491,
    2.4254837521587224e-05,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [11.5, 17].
const ERF_IMPL_IN: &[f64] = &[
    -0.0005690799360109496,
    0.00016949854037376225,
    5.184723545811009e-05,
    3.8281931223192885e-06,
    8.249899312818944e-08,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [11.5, 17].
const ERF_IMPL_ID: &[f64] = &[
    1.0,
    0.33963725005113937,
    0.04347264787031066,
    0.002485493352246371,
    5.356333053371529e-05,
    -1.1749094440545958e-13,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [17, 24].
const ERF_IMPL_JN: &[f64] = &[
    -0.00024131359948399134,
    5.742249752025015e-05,
    1.1599896292738377e-05,
    5.817621344025938e-07,
    8.539715550856736e-09,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [17, 24].
const ERF_IMPL_JD: &[f64] = &[
    1.0,
    0.23304413829968784,
    0.02041869405464403,
    0.0007971856475643983,
    1.1701928167017232e-05,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [24, 38].
const ERF_IMPL_KN: &[f64] = &[
    -0.00014667469927776036,
    1.6266655211228053e-05,
    2.6911624850916523e-06,
    9.79584479468092e-08,
    1.0199464762572346e-09,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [24, 38].
const ERF_IMPL_KD: &[f64] = &[
    1.0,
    0.16590781294484722,
    0.010336171619150588,
    0.0002865930263738684,
    2.9840157084090034e-06,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [38, 60].
const ERF_IMPL_LN: &[f64] = &[
    -5.839057976297718e-05,
    4.125103251054962e-06,
    4.3179092242025094e-07,
    9.933651555900132e-09,
    6.534805100201047e-11,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [38, 60].
const ERF_IMPL_LD: &[f64] = &[
    1.0,
    0.10507708607203992,
    0.004142784286754756,
    7.263387546445238e-05,
    4.778184710473988e-07,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [60, 85].
const ERF_IMPL_MN: &[f64] = &[
    -1.9645779760922958e-05,
    1.572438876668007e-06,
    5.439025111927009e-08,
    3.174724923691177e-10,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [60, 85].
const ERF_IMPL_MD: &[f64] = &[
    1.0,
    0.05280398924095763,
    0.0009268760691517533,
    5.410117232266303e-06,
    5.350938458036424e-16,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [85, 110].
const ERF_IMPL_NN: &[f64] = &[
    -7.892247039787227e-06,
    6.22088451660987e-07,
    1.457284456768824e-08,
    6.037155055427153e-11,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [85, 110].
const ERF_IMPL_ND: &[f64] = &[
    1.0,
    0.03753288463562937,
    0.0004679195359746253,
    1.9384703927584565e-06,
];

/// `erf_impl` computes the error function at `z`.
/// If `inv` is true, `1 - erf` is calculated as opposed to `erf`
fn erf_impl(z: f64, inv: bool) -> f64 {
    if z < 0.0 {
        if !inv {
            return -erf_impl(-z, false);
        }
        if z < -0.5 {
            return 2.0 - erf_impl(-z, true);
        }
        return 1.0 + erf_impl(-z, false);
    }

    let result = if z < 0.5 {
        if z < 1e-10 {
            z * 1.125 + z * 0.0033791670955125737
        } else {
            z * 1.125 + z * polynomial(z, ERF_IMPL_AN) / polynomial(z, ERF_IMPL_AD)
        }
    } else if z < 110.0 {
        let (r, b) = if z < 0.75 {
            (
                polynomial(z - 0.5, ERF_IMPL_BN) / polynomial(z - 0.5, ERF_IMPL_BD),
                0.3440242112,
            )
        } else if z < 1.25 {
            (
                polynomial(z - 0.75, ERF_IMPL_CN) / polynomial(z - 0.75, ERF_IMPL_CD),
                0.419990927,
            )
        } else if z < 2.25 {
            (
                polynomial(z - 1.25, ERF_IMPL_DN) / polynomial(z - 1.25, ERF_IMPL_DD),
                0.4898625016,
            )
        } else if z < 3.5 {
            (
                polynomial(z - 2.25, ERF_IMPL_EN) / polynomial(z - 2.25, ERF_IMPL_ED),
                0.5317370892,
            )
        } else if z < 5.25 {
            (
                polynomial(z - 3.5, ERF_IMPL_FN) / polynomial(z - 3.5, ERF_IMPL_FD),
                0.5489973426,
            )
        } else if z < 8.0 {
            (
                polynomial(z - 5.25, ERF_IMPL_GN) / polynomial(z - 5.25, ERF_IMPL_GD),
                0.5571740866,
            )
        } else if z < 11.5 {
            (
                polynomial(z - 8.0, ERF_IMPL_HN) / polynomial(z - 8.0, ERF_IMPL_HD),
                0.5609807968,
            )
        } else if z < 17.0 {
            (
                polynomial(z - 11.5, ERF_IMPL_IN) / polynomial(z - 11.5, ERF_IMPL_ID),
                0.5626493692,
            )
        } else if z < 24.0 {
            (
                polynomial(z - 17.0, ERF_IMPL_JN) / polynomial(z - 17.0, ERF_IMPL_JD),
                0.5634598136,
            )
        } else if z < 38.0 {
            (
                polynomial(z - 24.0, ERF_IMPL_KN) / polynomial(z - 24.0, ERF_IMPL_KD),
                0.5638477802,
            )
        } else if z < 60.0 {
            (
                polynomial(z - 38.0, ERF_IMPL_LN) / polynomial(z - 38.0, ERF_IMPL_LD),
                0.5640528202,
            )
        } else if z < 85.0 {
            (
                polynomial(z - 60.0, ERF_IMPL_MN) / polynomial(z - 60.0, ERF_IMPL_MD),
                0.5641309023,
            )
        } else {
            (
                polynomial(z - 85.0, ERF_IMPL_NN) / polynomial(z - 85.0, ERF_IMPL_ND),
                0.5641584396,
            )
        };
        let g = (-z * z).exp() / z;
        g * b + g * r
    } else {
        0.0
    };

    if inv && z >= 0.5 {
        result
    } else if z >= 0.5 || inv {
        1.0 - result
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known_values() {
        // High-precision reference values. The approximation, as vendored in
        // candle, is accurate to ~1e-10 (e.g. erf(0.5) is off by 5e-11): far
        // below f32 resolution, and what Kyutai's Mimi computes.
        for &(x, e) in &[
            (0.1, 0.112_462_916_018_284_9),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.0, 0.999_977_909_503_001_4),
        ] {
            assert!((erf(x) - e).abs() < 1e-10, "erf({x})");
            assert_eq!(erf(-x), -erf(x));
        }
        // Bit-exact values of candle 0.9.1's `candle::cpu::erf::erf`.
        assert_eq!(erf(0.5), 0.520_499_877_763_653_8);
        assert_eq!(erf(1.0), 0.842_700_792_942_714_9);
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert_eq!(erf(200.0), 1.0);
    }

    #[test]
    fn gelu_values() {
        let x = [-3.0f64, -1.0, 0.0, 0.5, 2.0];
        let mut y = [0.0; 5];
        gelu_erf_fwd(&x, &mut y);
        let e = [
            -0.004_049_694_094_890_31,
            -0.158_655_253_931_457_07,
            0.0,
            0.345_731_230_637_006_56,
            1.954_499_736_103_641_6,
        ];
        for (a, b) in y.iter().zip(&e) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
        let mut z = x;
        gelu_erf_inplace(&mut z);
        assert_eq!(z, y);
    }
}
