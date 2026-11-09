//! GPU tests for the cudarc wrappers (feature `cuda`). They need an sm_80+ GPU,
//! driver R580+ and libcublasLt / libnccl, so they are `#[ignore]`d here:
//!
//! ```text
//! cargo test -p s2s-cuda --features cuda --test gpu -- --ignored
//! ```

#![cfg(feature = "cuda")]

use s2s_core::SplitMix64;
use s2s_cuda::cudarc::driver::CudaContext;
use s2s_cuda::cutile::prelude::DeviceOp;
use s2s_cuda::cutile::tensor::ToHostVec;
use s2s_cuda::gemm::{F32Math, Gemm, Trans};
use s2s_cuda::half::bf16;
use s2s_cuda::interop::TileContext;
use s2s_cuda::nccl::NcclComm;
use s2s_kernels::reference::gemm::gemm as reference_gemm;
use s2s_train::comm::Communicator;

fn assert_close(what: &str, got: &[f32], want: &[f32], rtol: f32, atol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= atol + rtol * w.abs(),
            "{what}[{i}]: gpu {g} vs reference {w}"
        );
    }
}

/// Reference row-major GEMM in f64 with a bias added to every row.
#[allow(clippy::too_many_arguments)]
fn reference(
    ta: Trans,
    tb: Trans,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &[f32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let f = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
    let mut out = f(c);
    reference_gemm(
        ta,
        tb,
        m,
        n,
        k,
        alpha as f64,
        &f(a),
        &f(b),
        beta as f64,
        &mut out,
    );
    if let Some(bias) = bias {
        for r in 0..m {
            for j in 0..n {
                out[r * n + j] += bias[j] as f64;
            }
        }
    }
    out.iter().map(|&x| x as f32).collect()
}

#[test]
#[ignore = "needs a GPU"]
fn cublaslt_f32_matches_reference_gemm() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.default_stream();
    let mut g = Gemm::new(stream.clone()).expect("cuBLASLt");
    let mut rng = SplitMix64::new(1);
    let (m, n, k) = (37, 53, 70);
    for ta in [Trans::No, Trans::Yes] {
        for tb in [Trans::No, Trans::Yes] {
            for (alpha, beta, with_bias) in [(1.0, 0.0, false), (0.5, 1.5, true)] {
                let a: Vec<f32> = rng.normal_vec(m * k, 1.0);
                let b: Vec<f32> = rng.normal_vec(k * n, 1.0);
                let c: Vec<f32> = rng.normal_vec(m * n, 1.0);
                let bias: Vec<f32> = rng.normal_vec(n, 1.0);
                let want = reference(
                    ta,
                    tb,
                    m,
                    n,
                    k,
                    alpha,
                    &a,
                    &b,
                    beta,
                    &c,
                    with_bias.then_some(&bias[..]),
                );
                let da = stream.clone_htod(&a).unwrap();
                let db = stream.clone_htod(&b).unwrap();
                let mut dc = stream.clone_htod(&c).unwrap();
                let dbias = stream.clone_htod(&bias).unwrap();
                g.gemm(
                    ta,
                    tb,
                    m,
                    n,
                    k,
                    alpha,
                    &da,
                    &db,
                    beta,
                    &mut dc,
                    with_bias.then_some(&dbias),
                )
                .unwrap();
                let got = stream.clone_dtoh(&dc).unwrap();
                assert_close(&format!("f32 {ta:?}{tb:?}"), &got, &want, 1e-5, 1e-4);
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn cublaslt_tf32_and_bf16_match_reference_gemm() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.default_stream();
    let mut rng = SplitMix64::new(2);
    let (rows, d_in, d_out) = (64, 96, 80);
    // Inputs rounded to bf16 so the reference sees exactly what the GPU does.
    let round =
        |v: Vec<f32>| -> Vec<f32> { v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect() };
    let x = round(rng.normal_vec(rows * d_in, 1.0));
    let w = round(rng.normal_vec(d_out * d_in, 0.1));
    let bias: Vec<f32> = rng.normal_vec(d_out, 1.0);
    let zeros = vec![0.0f32; rows * d_out];
    let want = reference(
        Trans::No,
        Trans::Yes,
        rows,
        d_out,
        d_in,
        1.0,
        &x,
        &w,
        0.0,
        &zeros,
        Some(&bias),
    );

    // TF32: ~1e-3 relative products.
    let mut g = Gemm::new(stream.clone())
        .unwrap()
        .with_f32_math(F32Math::Tf32);
    let (dx, dw) = (
        stream.clone_htod(&x).unwrap(),
        stream.clone_htod(&w).unwrap(),
    );
    let dbias = stream.clone_htod(&bias).unwrap();
    let mut dy = stream.alloc_zeros::<f32>(rows * d_out).unwrap();
    g.linear_fwd(&dx, &dw, Some(&dbias), &mut dy, rows, d_in, d_out)
        .unwrap();
    assert_close(
        "tf32 linear",
        &stream.clone_dtoh(&dy).unwrap(),
        &want,
        5e-3,
        5e-3,
    );

    // bf16 in, f32 out: exact products, f32 accumulation.
    let to_bf = |v: &[f32]| v.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>();
    let (bx, bw) = (
        stream.clone_htod(&to_bf(&x)).unwrap(),
        stream.clone_htod(&to_bf(&w)).unwrap(),
    );
    let mut dy = stream.alloc_zeros::<f32>(rows * d_out).unwrap();
    g.linear_fwd(&bx, &bw, Some(&dbias), &mut dy, rows, d_in, d_out)
        .unwrap();
    assert_close(
        "bf16->f32 linear",
        &stream.clone_dtoh(&dy).unwrap(),
        &want,
        1e-5,
        1e-4,
    );

    // bf16 in, bf16 out: one bf16 rounding of the result (~4e-3 relative).
    let bbias = stream.clone_htod(&to_bf(&bias)).unwrap();
    let mut by = stream.alloc_zeros::<bf16>(rows * d_out).unwrap();
    g.linear_fwd(&bx, &bw, Some(&bbias), &mut by, rows, d_in, d_out)
        .unwrap();
    let got: Vec<f32> = stream
        .clone_dtoh(&by)
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect();
    assert_close("bf16->bf16 linear", &got, &want, 1e-2, 2e-2);
}

#[test]
#[ignore = "needs a GPU"]
fn nccl_single_rank_collectives() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let comm = NcclComm::single(ctx.default_stream()).expect("NCCL");
    assert_eq!((comm.rank(), comm.world_size()), (0, 1));
    let mut buf = vec![1.5f32, -2.0, 3.25];
    comm.all_reduce_sum(&mut buf).unwrap();
    assert_eq!(buf, vec![1.5, -2.0, 3.25]);
    let mut f = vec![0.5f64; 4];
    comm.all_reduce_sum_f64(&mut f).unwrap();
    assert_eq!(f, vec![0.5; 4]);
    let mut out = vec![0.0f32; 3];
    comm.reduce_scatter(&buf, &mut out).unwrap();
    assert_eq!(out, buf);
    comm.all_gather(&buf, &mut out).unwrap();
    assert_eq!(out, buf);
    comm.broadcast(&mut out, 0).unwrap();
    comm.barrier().unwrap();
    let mut bytes = b"manifest".to_vec();
    comm.broadcast_bytes(&mut bytes, 0).unwrap();
    assert_eq!(bytes, b"manifest");
}

#[test]
#[ignore = "needs a GPU"]
fn cudarc_buffer_round_trips_through_cutile_without_copy() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.default_stream();
    let tile = TileContext::new(&ctx).expect("primary context");
    let host: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
    let slice = stream.clone_htod(&host).unwrap();
    let (tensor, buffer) = tile.adopt(slice, &[32, 32]).unwrap();
    assert!(buffer.in_use());
    // cuTile reads the cudarc memory directly.
    let back = tensor.to_host_vec().sync().unwrap();
    assert_eq!(back, host);
    // The tensor is gone, so cudarc gets its buffer back.
    let slice = buffer.try_into_slice().expect("no cuTile holders left");
    assert_eq!(stream.clone_dtoh(&slice).unwrap(), host);
}
