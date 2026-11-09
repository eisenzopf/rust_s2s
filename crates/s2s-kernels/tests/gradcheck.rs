//! Finite-difference gradient checks for every differentiable reference op.
//!
//! For an op `y = f(x)` and a random upstream gradient `dy`, the scalar
//! `L(x) = sum(dy * f(x))` has gradient `dL/dx`, which the op's backward pass
//! must reproduce. We compare against central differences in `f64`.

use s2s_core::SplitMix64;
use s2s_kernels::reference::{activation, attention, embedding, gemm, loss, norm, rope};

const H: f64 = 1e-5;

fn assert_close(name: &str, analytic: &[f64], numeric: &[f64]) {
    assert_eq!(analytic.len(), numeric.len());
    for (i, (&a, &n)) in analytic.iter().zip(numeric).enumerate() {
        let tol = 1e-6 + 1e-5 * a.abs().max(n.abs());
        assert!(
            (a - n).abs() <= tol,
            "{name}[{i}]: analytic {a:.9e} vs numeric {n:.9e} (diff {:.3e})",
            (a - n).abs()
        );
    }
}

/// Central-difference gradient of `loss` with respect to `x`.
fn numeric_grad(x: &[f64], loss: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    let mut xp = x.to_vec();
    (0..x.len())
        .map(|i| {
            let orig = xp[i];
            xp[i] = orig + H;
            let lp = loss(&xp);
            xp[i] = orig - H;
            let lm = loss(&xp);
            xp[i] = orig;
            (lp - lm) / (2.0 * H)
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn linear() {
    let mut r = SplitMix64::new(1);
    let (rows, din, dout) = (3, 4, 5);
    let x = r.normal_vec(rows * din, 1.0);
    let w = r.normal_vec(dout * din, 0.5);
    let b = r.normal_vec(dout, 0.5);
    let dy = r.normal_vec(rows * dout, 1.0);
    let f = |x: &[f64], w: &[f64], b: &[f64]| {
        let mut y = vec![0.0; rows * dout];
        gemm::linear_fwd(x, w, Some(b), &mut y, rows, din, dout);
        dot(&y, &dy)
    };
    let (mut dx, mut dw, mut db) = (vec![0.0; x.len()], vec![0.0; w.len()], vec![0.0; b.len()]);
    gemm::linear_bwd(
        &x,
        &w,
        &dy,
        Some(&mut dx),
        Some(&mut dw),
        Some(&mut db),
        rows,
        din,
        dout,
    );
    assert_close("dx", &dx, &numeric_grad(&x, |x| f(x, &w, &b)));
    assert_close("dw", &dw, &numeric_grad(&w, |w| f(&x, w, &b)));
    assert_close("db", &db, &numeric_grad(&b, |b| f(&x, &w, b)));
}

#[test]
fn rmsnorm() {
    let mut r = SplitMix64::new(2);
    let (rows, d) = (3, 7);
    let x = r.normal_vec(rows * d, 1.0);
    let w: Vec<f64> = (0..d).map(|_| 1.0 + 0.3 * r.normal()).collect();
    let dy = r.normal_vec(rows * d, 1.0);
    let eps = 1e-5;
    let f = |x: &[f64], w: &[f64]| {
        let (mut y, mut rs) = (vec![0.0; rows * d], vec![0.0; rows]);
        norm::rmsnorm_fwd(x, w, eps, &mut y, &mut rs, rows, d);
        dot(&y, &dy)
    };
    let (mut y, mut rstd) = (vec![0.0; rows * d], vec![0.0; rows]);
    norm::rmsnorm_fwd(&x, &w, eps, &mut y, &mut rstd, rows, d);
    let (mut dx, mut dw) = (vec![0.0; x.len()], vec![0.0; d]);
    norm::rmsnorm_bwd(&x, &w, &rstd, &dy, &mut dx, &mut dw, rows, d);
    assert_close("dx", &dx, &numeric_grad(&x, |x| f(x, &w)));
    assert_close("dw", &dw, &numeric_grad(&w, |w| f(&x, w)));
}

#[test]
fn rope_both_layouts() {
    let mut r = SplitMix64::new(3);
    let (tokens, heads, d) = (3, 2, 8);
    let positions = [0i64, 17, 4096];
    for layout in [rope::RopeLayout::Half, rope::RopeLayout::Interleaved] {
        let cfg = rope::RopeConfig {
            theta: 10_000.0,
            layout,
        };
        let x = r.normal_vec(tokens * heads * d, 1.0);
        let dy = r.normal_vec(x.len(), 1.0);
        let f = |x: &[f64]| {
            let mut y = x.to_vec();
            rope::rope_fwd(&mut y, &positions, tokens, heads, d, cfg);
            dot(&y, &dy)
        };
        let mut dx = dy.clone();
        rope::rope_bwd(&mut dx, &positions, tokens, heads, d, cfg);
        assert_close("dx", &dx, &numeric_grad(&x, f));
    }
}

#[test]
fn swiglu() {
    let mut r = SplitMix64::new(4);
    let n = 16;
    let g = r.normal_vec(n, 2.0);
    let u = r.normal_vec(n, 1.0);
    let dh = r.normal_vec(n, 1.0);
    let f = |g: &[f64], u: &[f64]| {
        let mut h = vec![0.0; n];
        activation::swiglu_fwd(g, u, &mut h);
        dot(&h, &dh)
    };
    let (mut dg, mut du) = (vec![0.0; n], vec![0.0; n]);
    activation::swiglu_bwd(&g, &u, &dh, &mut dg, &mut du);
    assert_close("dgate", &dg, &numeric_grad(&g, |g| f(g, &u)));
    assert_close("dup", &du, &numeric_grad(&u, |u| f(&g, u)));
}

fn check_attention(p: attention::AttnParams, seed: u64) {
    let mut r = SplitMix64::new(seed);
    let qn = p.batch * p.q_heads * p.t * p.d;
    let kn = p.batch * p.kv_heads * p.s * p.d;
    let q = r.normal_vec(qn, 1.0);
    let k = r.normal_vec(kn, 1.0);
    let v = r.normal_vec(kn, 1.0);
    let d_o = r.normal_vec(qn, 1.0);
    let lse_n = p.batch * p.q_heads * p.t;
    let f = |q: &[f64], k: &[f64], v: &[f64]| {
        let (mut o, mut lse) = (vec![0.0; qn], vec![0.0; lse_n]);
        attention::attention_fwd(&p, q, k, v, &mut o, &mut lse);
        dot(&o, &d_o)
    };
    let (mut o, mut lse) = (vec![0.0; qn], vec![0.0; lse_n]);
    attention::attention_fwd(&p, &q, &k, &v, &mut o, &mut lse);
    let (mut dq, mut dk, mut dv) = (vec![0.0; qn], vec![0.0; kn], vec![0.0; kn]);
    attention::attention_bwd(&p, &q, &k, &v, &o, &lse, &d_o, &mut dq, &mut dk, &mut dv);
    assert_close("dq", &dq, &numeric_grad(&q, |q| f(q, &k, &v)));
    assert_close("dk", &dk, &numeric_grad(&k, |k| f(&q, k, &v)));
    assert_close("dv", &dv, &numeric_grad(&v, |v| f(&q, &k, v)));
}

#[test]
fn attention_causal_mha() {
    check_attention(
        attention::AttnParams {
            batch: 2,
            q_heads: 2,
            kv_heads: 2,
            t: 4,
            s: 4,
            d: 4,
            scale: 0.5,
            q_offset: 0,
            mask: attention::AttnMask {
                causal: true,
                ..Default::default()
            },
        },
        5,
    );
}

#[test]
fn attention_gqa_window_pinned_offset() {
    // 4 query heads sharing 2 KV heads; 3 new queries at the end of a 7-key
    // sequence; sliding window of 3 with a 2-key pinned prefix.
    check_attention(
        attention::AttnParams {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            t: 3,
            s: 7,
            d: 4,
            scale: 0.5,
            q_offset: 4,
            mask: attention::AttnMask {
                causal: true,
                window: Some(3),
                pinned_prefix: 2,
            },
        },
        6,
    );
}

#[test]
fn attention_bidirectional() {
    check_attention(
        attention::AttnParams {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            t: 3,
            s: 5,
            d: 3,
            scale: 0.7,
            q_offset: 0,
            mask: attention::AttnMask::default(),
        },
        7,
    );
}

#[test]
fn embedding_sum() {
    let mut r = SplitMix64::new(8);
    let (n, d) = (4, 3);
    let vocab = [5usize, 3];
    let t0 = r.normal_vec(vocab[0] * d, 1.0);
    let t1 = r.normal_vec(vocab[1] * d, 1.0);
    // Repeated tokens and a NO_TOKEN slot exercise accumulation and skipping.
    let tokens = [0, 2, 4, 1, 0, embedding::NO_TOKEN, 3, 2];
    let dout = r.normal_vec(n * d, 1.0);
    let f = |t0: &[f64], t1: &[f64]| {
        let mut out = vec![0.0; n * d];
        embedding::embedding_sum_fwd(&[t0, t1], &vocab, &tokens, n, d, &mut out);
        dot(&out, &dout)
    };
    let (mut d0, mut d1) = (vec![0.0; t0.len()], vec![0.0; t1.len()]);
    embedding::embedding_sum_bwd(&mut [&mut d0, &mut d1], &vocab, &tokens, n, d, &dout);
    assert_close("dtable0", &d0, &numeric_grad(&t0, |t| f(t, &t1)));
    assert_close("dtable1", &d1, &numeric_grad(&t1, |t| f(&t0, t)));
}

#[test]
fn cross_entropy_weighted_with_ignore() {
    let mut r = SplitMix64::new(9);
    let (rows, v) = (4, 6);
    let logits = r.normal_vec(rows * v, 2.0);
    let targets = [3i64, loss::IGNORE_INDEX, 0, 5];
    let weights = [1.0, 7.0, 0.5, 2.0];
    let mean_loss = |l: &[f64]| {
        let mut lse = vec![0.0; rows];
        let (s, t) = loss::cross_entropy_fwd(l, &targets, Some(&weights), &mut lse, rows, v);
        s / t
    };
    let mut lse = vec![0.0; rows];
    let (_, total) = loss::cross_entropy_fwd(&logits, &targets, Some(&weights), &mut lse, rows, v);
    let mut dl = vec![0.0; logits.len()];
    loss::cross_entropy_bwd(
        &logits,
        &targets,
        Some(&weights),
        &lse,
        1.0 / total,
        &mut dl,
        rows,
        v,
    );
    assert_close("dlogits", &dl, &numeric_grad(&logits, mean_loss));
    // The ignored row gets no gradient at all.
    assert!(dl[v..2 * v].iter().all(|&g| g == 0.0));
}

#[test]
fn softmax() {
    let mut r = SplitMix64::new(10);
    let (rows, v) = (3, 5);
    let x = r.normal_vec(rows * v, 1.5);
    let dy = r.normal_vec(rows * v, 1.0);
    let f = |x: &[f64]| {
        let mut y = vec![0.0; rows * v];
        loss::softmax_fwd(x, &mut y, rows, v);
        dot(&y, &dy)
    };
    let mut y = vec![0.0; rows * v];
    loss::softmax_fwd(&x, &mut y, rows, v);
    let mut dx = vec![0.0; x.len()];
    loss::softmax_bwd(&y, &dy, &mut dx, rows, v);
    assert_close("dx", &dx, &numeric_grad(&x, f));
}

#[test]
fn attention_segmented_packed_sequences() {
    // Two packed segments per row: [1,1,1,2,2,2] and [5,5,7,7,7,7].
    let p = attention::AttnParams {
        batch: 2,
        q_heads: 2,
        kv_heads: 1,
        t: 6,
        s: 6,
        d: 4,
        scale: 0.5,
        q_offset: 0,
        mask: attention::AttnMask {
            causal: true,
            ..Default::default()
        },
    };
    let segs: Vec<u32> = vec![1, 1, 1, 2, 2, 2, 5, 5, 7, 7, 7, 7];
    let mut r = SplitMix64::new(42);
    let qn = 2 * 2 * 6 * 4;
    let kn = 2 * 6 * 4;
    let q = r.normal_vec(qn, 1.0);
    let k = r.normal_vec(kn, 1.0);
    let v = r.normal_vec(kn, 1.0);
    let d_o = r.normal_vec(qn, 1.0);
    let f = |q: &[f64], k: &[f64], v: &[f64]| {
        let (mut o, mut lse) = (vec![0.0; qn], vec![0.0; 24]);
        attention::attention_fwd_segmented(&p, Some(&segs), q, k, v, &mut o, &mut lse);
        dot(&o, &d_o)
    };
    let (mut o, mut lse) = (vec![0.0; qn], vec![0.0; 24]);
    attention::attention_fwd_segmented(&p, Some(&segs), &q, &k, &v, &mut o, &mut lse);
    let (mut dq, mut dk, mut dv) = (vec![0.0; qn], vec![0.0; kn], vec![0.0; kn]);
    attention::attention_bwd_segmented(
        &p,
        Some(&segs),
        &q,
        &k,
        &v,
        &o,
        &lse,
        &d_o,
        &mut dq,
        &mut dk,
        &mut dv,
    );
    assert_close("dq", &dq, &numeric_grad(&q, |q| f(q, &k, &v)));
    assert_close("dk", &dk, &numeric_grad(&k, |k| f(&q, k, &v)));
    assert_close("dv", &dv, &numeric_grad(&v, |v| f(&q, &k, v)));

    // Isolation: the second segment of row 0 equals running it alone.
    let sub = attention::AttnParams {
        batch: 1,
        t: 3,
        s: 3,
        ..p
    };
    let take_q = |src: &[f64], h: usize| src[(h * 6 + 3) * 4..(h * 6 + 6) * 4].to_vec();
    let q2: Vec<f64> = [take_q(&q, 0), take_q(&q, 1)].concat();
    let k2 = k[3 * 4..6 * 4].to_vec();
    let v2 = v[3 * 4..6 * 4].to_vec();
    let (mut o2, mut lse2) = (vec![0.0; 2 * 3 * 4], vec![0.0; 6]);
    attention::attention_fwd(&sub, &q2, &k2, &v2, &mut o2, &mut lse2);
    for h in 0..2 {
        for x in 0..12 {
            let a = o[(h * 6 + 3) * 4 + x];
            let b = o2[h * 12 + x];
            assert!((a - b).abs() < 1e-12, "head {h} elem {x}: {a} vs {b}");
        }
    }
}
