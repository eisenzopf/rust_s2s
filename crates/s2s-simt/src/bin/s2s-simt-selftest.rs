//! GPU self-test: runs every cuda-oxide kernel and compares it against the
//! CPU reference in `s2s-kernels`. Needs an sm_80+ GPU:
//!
//! ```text
//! cd crates/s2s-simt && cargo oxide run
//! ```
//!
//! Exits non-zero on the first mismatch.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::DeviceAtomicF32;
use s2s_core::SplitMix64;
use s2s_kernels::reference::{delay, embedding, sampling};
use s2s_simt::{BLOCK, NO_TOKEN, blocks_for, kernels};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn check_close(name: &str, gpu: &[f32], cpu: &[f32], tol: f32) -> Res<()> {
    if gpu.len() != cpu.len() {
        return Err(format!("{name}: length {} vs {}", gpu.len(), cpu.len()).into());
    }
    for (i, (&g, &c)) in gpu.iter().zip(cpu).enumerate() {
        let same = (g == c) || (g - c).abs() <= tol * (1.0 + c.abs());
        if !same {
            return Err(format!("{name}[{i}]: gpu {g} vs cpu {c}").into());
        }
    }
    println!("PASSED {name} ({} values)", gpu.len());
    Ok(())
}

fn cfg(n: usize) -> LaunchConfig1D {
    LaunchConfig1D::new(blocks_for(n), BLOCK, 0)
}

fn main() -> Res<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: this package owns the embedded device bundle for `kernels`.
    let module = unsafe { kernels::load(&ctx)? };
    let mut rng = SplitMix64::new(0x5eed);

    // --- apply_token_bitmask -------------------------------------------
    {
        let v = 50_257; // not a multiple of 32, to exercise the tail word
        let logits: Vec<f32> = rng.normal_vec(v, 3.0);
        let allowed: Vec<u32> = (0..v.div_ceil(32)).map(|_| rng.next_u64() as u32).collect();
        let mut cpu = logits.clone();
        sampling::apply_token_bitmask(&mut cpu, &allowed);

        let allowed_dev = DeviceBuffer::from_host(&stream, &allowed)?;
        let mut logits_dev = DeviceBuffer::from_host(&stream, &logits)?;
        let prepared = module.prepare_apply_token_bitmask(cfg(v))?;
        module.apply_token_bitmask(&stream, &prepared, &allowed_dev, &mut logits_dev)?;
        check_close(
            "apply_token_bitmask",
            &logits_dev.to_host_vec(&stream)?,
            &cpu,
            0.0,
        )?;
    }

    // --- embedding_sum_fwd / bwd ---------------------------------------
    {
        // Three streams with different vocab sizes, like text + 2 codebooks.
        let vocab = [1000usize, 2048, 2048];
        let (n, d) = (300usize, 64usize);
        let tables: Vec<Vec<f32>> = vocab.iter().map(|&v| rng.normal_vec(v * d, 1.0)).collect();
        let mut tokens = vec![0u32; n * vocab.len()];
        for (k, t) in tokens.iter_mut().enumerate() {
            let s = k % vocab.len();
            // ~5% absent slots; small token range so rows repeat (atomics matter).
            *t = if rng.below(20) == 0 {
                NO_TOKEN
            } else {
                rng.below(vocab[s].min(50) as u64) as u32
            };
        }

        let refs: Vec<&[f32]> = tables.iter().map(|t| t.as_slice()).collect();
        let mut cpu_out = vec![0.0f32; n * d];
        embedding::embedding_sum_fwd(&refs, &vocab, &tokens, n, d, &mut cpu_out);

        let concat: Vec<f32> = tables.concat();
        let mut offsets = Vec::new();
        let mut acc = 0u32;
        for &v in &vocab {
            offsets.push(acc);
            acc += v as u32;
        }
        let tables_dev = DeviceBuffer::from_host(&stream, &concat)?;
        let offsets_dev = DeviceBuffer::from_host(&stream, &offsets)?;
        let tokens_dev = DeviceBuffer::from_host(&stream, &tokens)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, n * d)?;
        let prepared = module.prepare_embedding_sum_fwd(cfg(n * d))?;
        module.embedding_sum_fwd(
            &stream,
            &prepared,
            &tables_dev,
            &offsets_dev,
            &tokens_dev,
            vocab.len() as u32,
            d as u32,
            &mut out_dev,
        )?;
        check_close(
            "embedding_sum_fwd",
            &out_dev.to_host_vec(&stream)?,
            &cpu_out,
            1e-6,
        )?;

        let dout: Vec<f32> = rng.normal_vec(n * d, 1.0);
        let mut cpu_d: Vec<Vec<f32>> = vocab.iter().map(|&v| vec![0.0f32; v * d]).collect();
        {
            let mut dref: Vec<&mut [f32]> = cpu_d.iter_mut().map(|t| t.as_mut_slice()).collect();
            embedding::embedding_sum_bwd(&mut dref, &vocab, &tokens, n, d, &dout);
        }
        let dout_dev = DeviceBuffer::from_host(&stream, &dout)?;
        let dtables_dev =
            DeviceBuffer::<f32>::zeroed(&stream, concat.len())?.wrap_elem::<DeviceAtomicF32>();
        let mut done_dev = DeviceBuffer::<u32>::zeroed(&stream, n * d)?;
        let prepared = module.prepare_embedding_sum_bwd(cfg(n * d))?;
        module.embedding_sum_bwd(
            &stream,
            &prepared,
            &dout_dev,
            &tokens_dev,
            &offsets_dev,
            vocab.len() as u32,
            d as u32,
            (n * d) as u32,
            &dtables_dev,
            &mut done_dev,
        )?;
        let gpu_d = dtables_dev.peel_elem::<f32>().to_host_vec(&stream)?;
        // Atomic summation order differs from the CPU's, so allow rounding.
        check_close("embedding_sum_bwd", &gpu_d, &cpu_d.concat(), 1e-5)?;
    }

    // --- apply_delays ---------------------------------------------------
    {
        // Moshi-style layout: text, semantic (delay 0), 7 acoustic (delay 1),
        // user text (delay 6).
        let mut delays = vec![0u32, 0];
        delays.extend([1u32; 7]);
        delays.push(6);
        let streams = delays.len();
        let frames = 250usize;
        let pad: Vec<u32> = (0..streams as u32).map(|s| 60_000 + s).collect();
        let tokens: Vec<u32> = (0..frames * streams)
            .map(|_| rng.below(2048) as u32)
            .collect();
        let delays_usize: Vec<usize> = delays.iter().map(|&x| x as usize).collect();
        let cpu = delay::apply_delays(&tokens, frames, &delays_usize, &pad);

        let tokens_dev = DeviceBuffer::from_host(&stream, &tokens)?;
        let delays_dev = DeviceBuffer::from_host(&stream, &delays)?;
        let pad_dev = DeviceBuffer::from_host(&stream, &pad)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, cpu.len())?;
        let prepared = module.prepare_apply_delays(cfg(cpu.len()))?;
        module.apply_delays(
            &stream,
            &prepared,
            &tokens_dev,
            &delays_dev,
            &pad_dev,
            frames as u32,
            streams as u32,
            &mut out_dev,
        )?;
        let gpu = out_dev.to_host_vec(&stream)?;
        if gpu != cpu {
            return Err("apply_delays: mismatch".into());
        }
        println!("PASSED apply_delays ({} tokens)", gpu.len());
    }

    println!("ALL PASSED");
    Ok(())
}
