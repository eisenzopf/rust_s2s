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
use s2s_kernels::reference::{activation, norm};

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
