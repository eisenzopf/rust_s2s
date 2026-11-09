//! GPU vs. CPU-reference parity for the cuTile kernels (feature `cuda`).
//!
//! These need an sm_80+ GPU with a CUDA 13 driver, so they are `#[ignore]`d
//! by default. On a GPU machine:
//!
//! ```text
//! CUDA_TOOLKIT_PATH=/usr/local/cuda cargo test -p s2s-kernels --features cuda --test cuda_parity -- --ignored
//! ```

#![cfg(feature = "cuda")]

use s2s_core::SplitMix64;
use s2s_kernels::cuda::ops;
use s2s_kernels::reference::{activation, loss, norm};

/// cuTile's JIT recurses deeply; run on a big stack like its own tests.
fn with_big_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("GPU test thread panicked");
}

fn assert_close(name: &str, gpu: &[f32], cpu: &[f32], rtol: f32, atol: f32) {
    assert_eq!(gpu.len(), cpu.len(), "{name}: length");
    for (i, (&g, &c)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            (g - c).abs() <= atol + rtol * c.abs(),
            "{name}[{i}]: gpu {g} vs cpu {c}"
        );
    }
}

#[test]
#[ignore = "needs an sm_80+ GPU"]
fn vec_add_matches() {
    with_big_stack(|| {
        let mut r = SplitMix64::new(1);
        let x: Vec<f32> = r.normal_vec(4096, 1.0);
        let y: Vec<f32> = r.normal_vec(4096, 1.0);
        let gpu = ops::vec_add(&x, &y, 128).expect("launch");
        let cpu: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
        assert_close("vec_add", &gpu, &cpu, 0.0, 0.0);
    });
}

#[test]
#[ignore = "needs an sm_80+ GPU"]
fn rms_norm_fwd_matches() {
    with_big_stack(|| {
        let mut r = SplitMix64::new(2);
        let (rows, d, block) = (37, 1024, 256);
        let x: Vec<f32> = r.normal_vec(rows * d, 1.0);
        let w: Vec<f32> = r.normal_vec(d, 0.2).iter().map(|v: &f32| 1.0 + v).collect();
        let eps = 1e-5;
        let (y_gpu, rstd_gpu) = ops::rms_norm_fwd(&x, &w, eps, rows, d, block).expect("launch");
        let (mut y, mut rstd) = (vec![0.0f32; rows * d], vec![0.0f32; rows]);
        norm::rmsnorm_fwd(&x, &w, eps as f64, &mut y, &mut rstd, rows, d);
        assert_close("rstd", &rstd_gpu, &rstd, 1e-5, 1e-6);
        assert_close("y", &y_gpu, &y, 1e-5, 1e-5);
    });
}

#[test]
#[ignore = "needs an sm_80+ GPU"]
fn swiglu_fwd_matches() {
    with_big_stack(|| {
        let mut r = SplitMix64::new(3);
        let n = 8192;
        let g: Vec<f32> = r.normal_vec(n, 3.0);
        let u: Vec<f32> = r.normal_vec(n, 1.0);
        let gpu = ops::swiglu_fwd(&g, &u, 256).expect("launch");
        let mut cpu = vec![0.0f32; n];
        activation::swiglu_fwd(&g, &u, &mut cpu);
        assert_close("swiglu", &gpu, &cpu, 1e-5, 1e-6);
    });
}

#[test]
#[ignore = "needs an sm_80+ GPU"]
fn rms_norm_bwd_matches() {
    with_big_stack(|| {
        let mut r = SplitMix64::new(4);
        let (rows, d, block) = (19, 512, 128);
        let x: Vec<f32> = r.normal_vec(rows * d, 1.0);
        let w: Vec<f32> = r.normal_vec(d, 0.2).iter().map(|v: &f32| 1.0 + v).collect();
        let dy: Vec<f32> = r.normal_vec(rows * d, 1.0);
        let (mut y, mut rstd) = (vec![0.0f32; rows * d], vec![0.0f32; rows]);
        norm::rmsnorm_fwd(&x, &w, 1e-5, &mut y, &mut rstd, rows, d);
        let (dx_gpu, dw_gpu) =
            ops::rms_norm_bwd(&x, &w, &rstd, &dy, rows, d, block).expect("launch");
        let (mut dx, mut dw) = (vec![0.0f32; rows * d], vec![0.0f32; d]);
        norm::rmsnorm_bwd(&x, &w, &rstd, &dy, &mut dx, &mut dw, rows, d);
        assert_close("dx", &dx_gpu, &dx, 1e-4, 1e-5);
        assert_close("dw", &dw_gpu, &dw, 1e-4, 1e-4);
    });
}

#[test]
#[ignore = "needs an sm_80+ GPU"]
fn softmax_matches() {
    with_big_stack(|| {
        let mut r = SplitMix64::new(5);
        let (rows, v) = (13, 2048);
        let x: Vec<f32> = r.normal_vec(rows * v, 4.0);
        let gpu = ops::softmax(&x, rows, v, 256).expect("launch");
        let mut cpu = vec![0.0f32; rows * v];
        loss::softmax_fwd(&x, &mut cpu, rows, v);
        assert_close("softmax", &gpu, &cpu, 1e-4, 1e-7);
    });
}

#[test]
#[ignore = "needs an sm_80+ GPU"]
fn cross_entropy_matches() {
    with_big_stack(|| {
        let mut r = SplitMix64::new(6);
        let (rows, v) = (11, 2048);
        let logits: Vec<f32> = r.normal_vec(rows * v, 3.0);
        let mut targets: Vec<i64> = (0..rows).map(|_| r.below(v as u64) as i64).collect();
        targets[4] = loss::IGNORE_INDEX;
        let weights: Vec<f32> = (0..rows)
            .map(|i| if i % 3 == 0 { 0.5 } else { 1.0 })
            .collect();
        let mut lse = vec![0.0f32; rows];
        let (sum, total) =
            loss::cross_entropy_fwd(&logits, &targets, Some(&weights), &mut lse, rows, v);
        let mut dl = vec![0.0f32; rows * v];
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
        let (gsum, gtotal, gdl) =
            ops::cross_entropy(&logits, &targets, Some(&weights), 1.0 / total, rows, v, 256)
                .expect("launch");
        assert!(
            (gsum - sum).abs() <= 1e-3 * sum.abs(),
            "loss {gsum} vs {sum}"
        );
        assert_eq!(gtotal, total);
        assert_close("dlogits", &gdl, &dl, 1e-3, 1e-7);
    });
}

#[test]
#[ignore = "needs an sm_80+ GPU"]
fn adamw_matches() {
    with_big_stack(|| {
        use s2s_kernels::reference::optim::{adamw_step, AdamWConfig};
        let mut r = SplitMix64::new(7);
        let n = 4096;
        let p: Vec<f32> = r.normal_vec(n, 1.0);
        let g: Vec<f32> = r.normal_vec(n, 0.1);
        let m: Vec<f32> = r.normal_vec(n, 0.01);
        let v: Vec<f32> = r
            .normal_vec(n, 0.01)
            .iter()
            .map(|x: &f32| x.abs())
            .collect();
        let cfg = AdamWConfig {
            lr: 3e-4,
            ..Default::default()
        };
        let (gp, gm, gv) = ops::adamw_step(&p, &g, &m, &v, 7, &cfg, 0.5, 256).expect("launch");
        let (mut cp, mut cm, mut cv) = (p.clone(), m.clone(), v.clone());
        adamw_step(&mut cp, &g, &mut cm, &mut cv, 7, &cfg, 0.5);
        assert_close("param", &gp, &cp, 1e-5, 1e-7);
        assert_close("m", &gm, &cm, 1e-5, 1e-9);
        assert_close("v", &gv, &cv, 1e-5, 1e-9);
    });
}
