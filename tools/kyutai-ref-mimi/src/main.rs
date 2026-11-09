//! Generates Mimi parity fixtures with Kyutai's own Rust implementation.
//!
//! Usage (from this directory):
//!
//! ```text
//! cargo run -- reduced   ../../fixtures/mimi   # small model + outputs (committed)
//! cargo run -- pointwise ../../fixtures/mimi   # candle erf / gelu_erf / elu values (committed)
//! cargo run -- full      <scratch_dir> [frames] # full-size v0.1 model, random weights
//!                                               # (~350 MB); only *_names.json and
//!                                               # *_config.json are committed
//! ```
//!
//! `reduced` and `full`:
//! 1. build Kyutai's `moshi::mimi::Mimi` once on a `VarMap` to discover every
//!    parameter name and shape exactly as their code requests them,
//! 2. fill those parameters with deterministic pseudo-random values (seeded
//!    per tensor name, so values don't depend on iteration order),
//! 3. write them as safetensors with Kyutai's names, reload them through the
//!    normal safetensors `VarBuilder` path,
//! 4. run offline encode/decode, the individual stages (for intermediates)
//!    and the streaming `encode_step` / `decode_step` APIs on a deterministic
//!    test signal, and write everything to `<prefix>_io.safetensors`.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use moshi::mimi::{Config, Mimi, ResampleMethod};
use moshi::streaming::{StreamMask, StreamTensor};
use moshi::{conv, quantization, seanet, transformer};

const FRAME: usize = 1920;
const SEED: u64 = 0x5eed_2026_0001;

/// SplitMix64, seeded per tensor name.
struct Rng(u64);

impl Rng {
    fn for_name(name: &str) -> Self {
        // FNV-1a of the name, mixed with the global seed.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in name.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Rng(h ^ SEED)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.uniform() + 1.0) * 0.5 * (hi - lo)
    }
}

/// Deterministic values for one parameter, chosen by name so activations
/// stay in a sane range through the deep conv stacks.
fn init_values(name: &str, dims: &[usize]) -> Vec<f32> {
    let n: usize = dims.iter().product();
    let mut rng = Rng::for_name(name);
    let fan_in = if dims.len() >= 2 {
        dims[1..].iter().product::<usize>().max(1)
    } else {
        1
    };
    let leaf = name.rsplit('.').next().unwrap_or(name);
    let is_norm = name.contains(".norm1.") || name.contains(".norm2.");
    (0..n)
        .map(|_| match leaf {
            "_initialized" => 1.0,
            "cluster_usage" => rng.range(0.5, 2.0),
            // Codebook sums: entries of the embedding are sum / usage.
            "embedding_sum" => rng.uniform(),
            "scale" => rng.range(0.2, 0.6), // layer scale
            "weight_g" => rng.range(0.8, 1.2),
            "bias" if is_norm => rng.range(-0.1, 0.1),
            "bias" => rng.range(-0.1, 0.1),
            "weight" | "alpha" if is_norm => 1.0 + rng.range(-0.1, 0.1),
            // conv / linear weights (incl. weight-norm direction `weight_v`).
            _ => rng.uniform() / (fan_in as f32).sqrt(),
        })
        .collect()
}

fn reduced_config() -> Config {
    let seanet_cfg = seanet::Config {
        dimension: 32,
        channels: 1,
        causal: true,
        n_filters: 4,
        n_residual_layers: 1,
        activation: candle_nn::Activation::Elu(1.),
        compress: 2,
        dilation_base: 2,
        disable_norm_outer_blocks: 0,
        final_activation: None,
        kernel_size: 7,
        residual_kernel_size: 3,
        last_kernel_size: 3,
        lstm: 0,
        norm: conv::Norm::WeightNorm,
        pad_mode: conv::PadMode::Constant,
        ratios: vec![8, 6, 5, 4],
        true_skip: true,
    };
    let transformer_cfg = transformer::Config {
        d_model: seanet_cfg.dimension,
        num_heads: 4,
        num_layers: 2,
        causal: true,
        norm_first: true,
        bias_ff: false,
        bias_attn: false,
        layer_scale: Some(0.01),
        // Small so the 32-step test signal exercises the sliding window.
        context: 16,
        conv_kernel_size: 5,
        use_conv_bias: true,
        use_conv_block: false,
        cross_attention: None,
        max_period: 10000,
        gating: None,
        norm: moshi::NormType::LayerNorm,
        positional_embedding: transformer::PositionalEmbedding::Rope,
        dim_feedforward: 64,
        kv_repeat: 1,
        conv_layout: true,
        max_seq_len: 8192,
        shared_cross_attn: false,
    };
    Config {
        channels: 1,
        sample_rate: 24_000.,
        frame_rate: 12.5,
        renormalize: true,
        resample_method: ResampleMethod::Conv,
        seanet: seanet_cfg,
        transformer: transformer_cfg,
        quantizer_n_q: 4,
        quantizer_bins: 64,
        quantizer_dim: 16,
    }
}

/// Config as JSON in the field layout of `s2s_mimi::MimiConfig`.
fn config_json(cfg: &Config) -> String {
    let s = &cfg.seanet;
    let t = &cfg.transformer;
    let ratios: Vec<String> = s.ratios.iter().map(|r| r.to_string()).collect();
    let elu = match s.activation {
        candle_nn::Activation::Elu(a) => a,
        a => panic!("Mimi uses ELU, got {a:?}"),
    };
    format!(
        r#"{{
  "channels": {channels},
  "sample_rate": {sr},
  "frame_rate": {fr},
  "dimension": {dim},
  "n_filters": {nf},
  "n_residual_layers": {nres},
  "ratios": [{ratios}],
  "kernel_size": {ks},
  "residual_kernel_size": {rks},
  "last_kernel_size": {lks},
  "dilation_base": {db},
  "compress": {compress},
  "true_skip": {true_skip},
  "elu_alpha": {elu},
  "transformer_d_model": {dm},
  "transformer_num_heads": {nh},
  "transformer_num_layers": {nl},
  "transformer_dim_feedforward": {ff},
  "transformer_context": {ctx},
  "transformer_max_period": {mp},
  "transformer_layer_scale": {ls},
  "quantizer_n_q": {nq},
  "quantizer_bins": {bins},
  "quantizer_dim": {qd}
}}
"#,
        channels = cfg.channels,
        sr = cfg.sample_rate,
        fr = cfg.frame_rate,
        dim = s.dimension,
        nf = s.n_filters,
        nres = s.n_residual_layers,
        ratios = ratios.join(", "),
        ks = s.kernel_size,
        rks = s.residual_kernel_size,
        lks = s.last_kernel_size,
        db = s.dilation_base,
        compress = s.compress,
        true_skip = s.true_skip,
        elu = format!("{elu:?}"),
        ls = t.layer_scale.is_some(),
        dm = t.d_model,
        nh = t.num_heads,
        nl = t.num_layers,
        ff = t.dim_feedforward,
        ctx = t.context,
        mp = t.max_period,
        nq = cfg.quantizer_n_q,
        bins = cfg.quantizer_bins,
        qd = cfg.quantizer_dim,
    )
}

/// Every parameter name and shape Kyutai's code requests for `cfg`.
fn discover_params(cfg: &Config, dev: &Device) -> Result<BTreeMap<String, Vec<usize>>> {
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let _ = Mimi::new(cfg.clone(), vb)?;
    let data = vm.data().lock().unwrap();
    Ok(data
        .iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().dims().to_vec()))
        .collect())
}

/// Deterministic test signal: two tones, a chirp, light noise, fade-in.
fn test_signal(n: usize) -> Vec<f32> {
    let mut rng = Rng::for_name("test-signal");
    (0..n)
        .map(|i| {
            let t = i as f64 / 24_000.0;
            let env = (t / 0.05).min(1.0);
            let am = 0.75 + 0.25 * (2.0 * std::f64::consts::PI * 3.0 * t).sin();
            let v = 0.35 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                + 0.15 * (2.0 * std::f64::consts::PI * 1330.0 * t).sin()
                + 0.2 * (2.0 * std::f64::consts::PI * (300.0 + 1500.0 * t) * t).sin();
            (env * am * v) as f32 + 0.02 * rng.uniform()
        })
        .collect()
}

fn write_json(path: &Path, s: &str) -> Result<()> {
    std::fs::write(path, s).with_context(|| format!("writing {}", path.display()))
}

fn run(
    cfg: Config,
    run_cfg: Option<Config>,
    frames: usize,
    out: &Path,
    prefix: &str,
    write_names: bool,
) -> Result<()> {
    let dev = Device::Cpu;
    std::fs::create_dir_all(out)?;

    // 1-2. Names/shapes from Kyutai's builder, deterministic values.
    let params = discover_params(&cfg, &dev)?;
    let manifest: Vec<String> = params
        .iter()
        .map(|(k, v)| {
            let dims: Vec<String> = v.iter().map(|d| d.to_string()).collect();
            format!("  \"{k}\": [{}]", dims.join(", "))
        })
        .collect();
    if write_names {
        write_json(
            &out.join(format!("{prefix}_names.json")),
            &format!("{{\n{}\n}}\n", manifest.join(",\n")),
        )?;
    }
    let mut tensors = HashMap::new();
    for (name, dims) in &params {
        let vals = init_values(name, dims);
        tensors.insert(name.clone(), Tensor::from_vec(vals, dims.as_slice(), &dev)?);
    }
    let weights_path = out.join(format!("{prefix}.safetensors"));
    candle::safetensors::save(&tensors, &weights_path)?;
    let n_params: usize = params.values().map(|d| d.iter().product::<usize>()).sum();
    println!("{prefix}: {} tensors, {n_params} parameters", params.len());

    // 3. Reload through the regular safetensors path.
    let cfg = run_cfg.unwrap_or(cfg);
    write_json(
        &out.join(format!("{prefix}_config.json")),
        &config_json(&cfg),
    )?;
    let bytes = std::fs::read(&weights_path)?;
    let vb = VarBuilder::from_buffered_safetensors(bytes, DType::F32, &dev)?;
    let mut mimi = Mimi::new(cfg.clone(), vb.clone())?;

    // 4. Offline.
    let n = frames * FRAME;
    let pcm = test_signal(n);
    let pcm_in = Tensor::from_vec(pcm, (1, 1, n), &dev)?;
    let latent = mimi.encode_pre_quantize(&pcm_in)?;
    let codes = mimi.encode(&pcm_in)?;
    let pcm_out = mimi.decode(&codes)?;

    // Stage by stage, with the same weights, for intermediates.
    let dim = cfg.seanet.dimension;
    let enc = seanet::SeaNetEncoder::new(&cfg.seanet, vb.pp("encoder"))?;
    let enc_seanet = enc.forward(&pcm_in)?;
    let mut enc_tr =
        transformer::Transformer::new(None, dim, &cfg.transformer, vb.pp("encoder_transformer"))?;
    let enc_transformer = enc_tr.forward(&enc_seanet)?[0].clone();
    let stride = 2;
    let down = conv::ConvDownsample1d::new(stride, dim, true, true, vb.pp("downsample"))?;
    let latent2 = down.forward(&enc_transformer)?;
    let quantizer = quantization::SplitResidualVectorQuantizer::new(
        cfg.quantizer_dim,
        Some(dim),
        Some(dim),
        cfg.quantizer_n_q,
        cfg.quantizer_bins,
        vb.pp("quantizer"),
    )?;
    let quantized = quantizer.decode(&codes)?;
    let up = conv::ConvTrUpsample1d::new(stride, dim, true, true, vb.pp("upsample"))?;
    let dec_upsample = up.forward(&quantized)?;
    let mut dec_tr =
        transformer::Transformer::new(None, dim, &cfg.transformer, vb.pp("decoder_transformer"))?;
    let dec_transformer = dec_tr.forward(&dec_upsample)?[0].clone();
    let dec = seanet::SeaNetDecoder::new(&cfg.seanet, vb.pp("decoder"))?;
    let pcm_out2 = dec.forward(&dec_transformer)?;
    let d1 = max_abs_diff(&latent, &latent2)?;
    let d2 = max_abs_diff(&pcm_out, &pcm_out2)?;
    println!("stage-by-stage vs Mimi: latent {d1:e}, pcm {d2:e}");

    // Streaming: one 1920-sample chunk / one frame of codes per step.
    mimi.reset_state();
    let mut stream_codes = Vec::new();
    for f in 0..frames {
        let chunk = pcm_in.i((.., .., f * FRAME..(f + 1) * FRAME))?;
        let c = mimi.encode_step(&StreamTensor::from_tensor(chunk), &StreamMask::empty())?;
        if let Some(c) = c.as_option() {
            stream_codes.push(c.clone());
        }
    }
    let stream_codes = Tensor::cat(&stream_codes, D::Minus1)?;
    mimi.reset_state();
    let mut stream_pcm = Vec::new();
    for f in 0..frames {
        let c = codes.i((.., .., f..f + 1))?;
        let y = mimi.decode_step(&StreamTensor::from_tensor(c), &StreamMask::empty())?;
        if let Some(y) = y.as_option() {
            stream_pcm.push(y.clone());
        }
    }
    let stream_pcm = Tensor::cat(&stream_pcm, D::Minus1)?;
    report_stream(
        &codes,
        &stream_codes,
        &pcm_out,
        &stream_pcm,
        cfg.transformer.context,
    )?;

    let mut io = HashMap::new();
    io.insert("pcm_in".to_string(), pcm_in);
    io.insert("enc_seanet".to_string(), enc_seanet);
    io.insert("enc_transformer".to_string(), enc_transformer);
    io.insert("latent".to_string(), latent);
    io.insert("codes".to_string(), codes);
    io.insert("quantized".to_string(), quantized);
    io.insert("dec_upsample".to_string(), dec_upsample);
    io.insert("dec_transformer".to_string(), dec_transformer);
    io.insert("pcm_out".to_string(), pcm_out);
    io.insert("stream_codes".to_string(), stream_codes);
    io.insert("stream_pcm_out".to_string(), stream_pcm);
    candle::safetensors::save(&io, out.join(format!("{prefix}_io.safetensors")))?;
    Ok(())
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?)
}

/// Print how Kyutai's streaming output compares with their offline output.
fn report_stream(
    codes: &Tensor,
    s_codes: &Tensor,
    pcm: &Tensor,
    s_pcm: &Tensor,
    context: usize,
) -> Result<()> {
    let a = codes.flatten_all()?.to_vec1::<u32>()?;
    let b = s_codes.flatten_all()?.to_vec1::<u32>()?;
    let same = a.iter().zip(&b).filter(|(x, y)| x == y).count();
    println!(
        "kyutai streaming vs offline codes: {same}/{} equal",
        a.len()
    );
    let frames = codes.dim(D::Minus1)?;
    // The transformers run at 2 steps per frame.
    let safe = (context / 2).min(frames);
    let n = safe * FRAME;
    let d_all = max_abs_diff(pcm, s_pcm)?;
    let d_safe = max_abs_diff(&pcm.i((.., .., ..n))?, &s_pcm.i((.., .., ..n))?)?;
    println!("kyutai streaming vs offline pcm: max diff {d_all:e} overall, {d_safe:e} over the first {safe} frames");
    Ok(())
}

/// candle's CPU `erf` (f64), `gelu_erf` and `elu` (f32) on a grid, to pin the
/// pointwise ops bit for bit.
fn pointwise(out: &Path) -> Result<()> {
    let dev = Device::Cpu;
    std::fs::create_dir_all(out)?;
    let mut x: Vec<f64> = (0..=1000)
        .map(|i| -10.0 + 20.0 * i as f64 / 1000.0)
        .collect();
    x.extend([
        0.0, -0.0, 1e-12, -1e-12, 0.4999999, 0.5, 0.75, 1.25, 2.25, 3.5, 5.25, 8.0, 11.5, 17.0,
        30.0,
    ]);
    let erf: Vec<f64> = x.iter().map(|&v| candle::cpu::erf::erf(v)).collect();
    let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let n = x32.len();
    let xt = Tensor::from_vec(x32, n, &dev)?;
    let gelu = xt.gelu_erf()?;
    let elu = xt.elu(1.0)?;
    let mut io = HashMap::new();
    io.insert("x".to_string(), Tensor::from_vec(x.clone(), n, &dev)?);
    io.insert("erf".to_string(), Tensor::from_vec(erf, n, &dev)?);
    io.insert("x32".to_string(), xt);
    io.insert("gelu_erf".to_string(), gelu);
    io.insert("elu".to_string(), elu);
    candle::safetensors::save(&io, out.join("pointwise_ref.safetensors"))?;
    for v in [0.1f64, 0.5, 1.0, 2.0, 3.0] {
        println!("candle erf({v}) = {:?}", candle::cpu::erf::erf(v));
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("reduced");
    let out = PathBuf::from(args.get(2).map(String::as_str).unwrap_or("out"));
    // Optional third argument: number of 1920-sample frames for `full`.
    let frames: Option<usize> = args.get(3).map(|s| s.parse().expect("frames"));
    match mode {
        "reduced" => run(reduced_config(), None, 16, &out, "reduced", false),
        // Weights for Kyutai's default config (16 codebooks), run at 8 codebooks as Moshi/STT do.
        // Only `full_v0_1_names.json` and `full_v0_1_config.json` are committed.
        "full" => run(
            Config::v0_1(None),
            Some(Config::v0_1(Some(8))),
            frames.unwrap_or(6),
            &out,
            "full_v0_1",
            true,
        ),
        "pointwise" => pointwise(&out),
        m => anyhow::bail!("unknown mode {m}, expected `reduced`, `full` or `pointwise`"),
    }
}
