//! Parity-fixture generator for `s2s-model`'s port of Kyutai's multistream LM.
//!
//! Runs Kyutai's own Rust implementation (the `moshi` crate, candle on CPU,
//! f32) on reduced configs of the real architectures with seeded random
//! weights, and records everything `s2s-model` must reproduce.
//!
//! ```text
//! cargo run -- fixtures <out_dir>    # writes <out_dir>/{moshi,stt,variants}/...
//! cargo run -- manifests <out.json>  # names + shapes of full-size configs
//! ```
//!
//! Each fixture directory holds:
//! - `config.json`: the LM config in Kyutai's serde format (`moshi::lm::Config`),
//!   plus the driver settings under `"driver"`;
//! - `model.safetensors`: f32 weights under Kyutai's (candle) tensor names;
//! - `trace.json`: per-frame model inputs, text logits, depformer logits and
//!   sampled tokens, plus the outputs of Kyutai's own drivers
//!   (`lm_generate_multistream::State` / `asr::State`).
//!
//! The per-frame loop below is a line-by-line replica of Kyutai's drivers so
//! that intermediate logits can be recorded (their drivers only return
//! tokens). Every fixture cross-checks that replica against Kyutai's real
//! driver and real `LmModel::depformer_sample` and aborts on any mismatch.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{bail, ensure, Context, Result};
use candle::{DType, Device, IndexOp, Module, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use moshi::nn::{MaybeQuantizedEmbedding, MaybeQuantizedLinear, MaybeQuantizedVarBuilder};
use moshi::StreamingModule;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Deterministic weights
// ---------------------------------------------------------------------------

/// SplitMix64 + Box-Muller, same as `s2s_core::SplitMix64` (only used to make
/// the random weights reproducible; the weights themselves are stored).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn below(&mut self, n: u32) -> u32 {
        ((self.next_u64() as u128 * n as u128) >> 64) as u32
    }
}

/// A candle var-builder backend that records every requested tensor name and
/// shape and hands back a zero-strided (allocation-free) zero tensor. Used to
/// enumerate the parameters of a config without materializing them.
#[derive(Default)]
struct Recorder {
    seen: Mutex<BTreeMap<String, Vec<usize>>>,
}

impl SimpleBackend for Recorder {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle::Result<Tensor> {
        let mut seen = self.seen.lock().unwrap();
        if let Some(prev) = seen.get(name) {
            if prev.as_slice() != s.dims() {
                candle::bail!("{name} requested with shapes {prev:?} and {:?}", s.dims());
            }
        }
        seen.insert(name.to_string(), s.dims().to_vec());
        Tensor::zeros((), dtype, dev)?.broadcast_as(s)
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        false
    }
}

/// Struct holding a shared recorder so it can be read back after the
/// var-builder (which owns a boxed backend) is dropped.
struct SharedRecorder(std::sync::Arc<Recorder>);

impl SimpleBackend for SharedRecorder {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle::Result<Tensor> {
        self.0.get(s, name, h, dtype, dev)
    }
    fn contains_tensor(&self, name: &str) -> bool {
        self.0.contains_tensor(name)
    }
}

/// All parameter names and shapes `LmModel::new` asks for with this config.
fn param_shapes(cfg: &moshi::lm::Config) -> Result<BTreeMap<String, Vec<usize>>> {
    let rec = std::sync::Arc::new(Recorder::default());
    let vb = VarBuilder::from_backend(
        Box::new(SharedRecorder(rec.clone())),
        DType::F32,
        Device::Cpu,
    );
    let _model = moshi::lm::LmModel::new(cfg, MaybeQuantizedVarBuilder::Real(vb))?;
    let seen = rec.seen.lock().unwrap().clone();
    Ok(seen)
}

/// Initial standard deviation for a parameter, by name and shape.
fn init_for(name: &str, shape: &[usize]) -> (f64, f64) {
    let leaf = name.rsplit('.').next().unwrap_or(name);
    let is_emb = (name.contains("emb.") && !name.contains("low_rank") && leaf == "weight")
        || name.ends_with("embed.weight");
    match leaf {
        "alpha" => (1.0, 0.2),
        "scale" => (0.6, 0.2),
        "learnt_padding" => (0.0, 0.5),
        "bias" | "in_proj_bias" => (0.0, 0.1),
        _ if shape.len() == 1 => (1.0, 0.2), // LayerNorm weight
        _ if is_emb => (0.0, 1.0),
        _ => {
            let fan_in = *shape.last().unwrap() as f64;
            // Spread the text head so that greedy decoding has clear winners.
            let gain = if name.starts_with("text_linear") || name.contains("linear_out") {
                3.0
            } else {
                1.0
            };
            (0.0, gain / fan_in.sqrt())
        }
    }
}

fn random_weights(
    shapes: &BTreeMap<String, Vec<usize>>,
    seed: u64,
) -> Result<HashMap<String, Tensor>> {
    let mut rng = Rng(seed);
    let mut out = HashMap::new();
    for (name, shape) in shapes {
        let n: usize = shape.iter().product();
        let (mean, std) = init_for(name, shape);
        let data: Vec<f32> = (0..n).map(|_| (mean + std * rng.normal()) as f32).collect();
        out.insert(
            name.clone(),
            Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Config (de)serialization in Kyutai's serde format
// ---------------------------------------------------------------------------

fn transformer_json(c: &moshi::transformer::Config) -> Value {
    json!({
        "d_model": c.d_model,
        "num_heads": c.num_heads,
        "num_layers": c.num_layers,
        "causal": c.causal,
        "norm_first": c.norm_first,
        "bias_ff": c.bias_ff,
        "bias_attn": c.bias_attn,
        "layer_scale": c.layer_scale,
        "positional_embedding": c.positional_embedding,
        "use_conv_block": c.use_conv_block,
        "cross_attention": c.cross_attention.map(|(g, n, d)| json!([g, n, d])),
        "conv_kernel_size": c.conv_kernel_size,
        "use_conv_bias": c.use_conv_bias,
        "gating": c.gating,
        "norm": c.norm,
        "context": c.context,
        "max_period": c.max_period,
        "max_seq_len": c.max_seq_len,
        "kv_repeat": c.kv_repeat,
        "dim_feedforward": c.dim_feedforward,
        "conv_layout": c.conv_layout,
        "shared_cross_attn": c.shared_cross_attn,
    })
}

fn lm_config_json(c: &moshi::lm::Config) -> Value {
    use moshi::conditioner::ConditionerConfig;
    let conditioners = c.conditioners.as_ref().map(|m| {
        let mut o = serde_json::Map::new();
        let mut keys: Vec<_> = m.keys().collect();
        keys.sort();
        for k in keys {
            let v = match &m[k] {
                ConditionerConfig::Lut(l) => json!({
                    "type": "Lut", "n_bins": l.n_bins, "dim": l.dim,
                    "possible_values": l.possible_values,
                }),
                ConditionerConfig::ContinuousAttribute(a) => json!({
                    "type": "ContinuousAttribute", "dim": a.dim,
                    "scale_factor": a.scale_factor, "max_period": a.max_period,
                }),
            };
            o.insert(k.clone(), v);
        }
        Value::Object(o)
    });
    json!({
        "transformer": transformer_json(&c.transformer),
        "depformer": c.depformer.as_ref().map(|d| json!({
            "transformer": transformer_json(&d.transformer),
            "num_slices": d.num_slices,
            "low_rank_embeddings": d.low_rank_embeddings,
        })),
        "text_in_vocab_size": c.text_in_vocab_size,
        "text_out_vocab_size": c.text_out_vocab_size,
        "audio_vocab_size": c.audio_vocab_size,
        "audio_codebooks": c.audio_codebooks,
        "conditioners": conditioners,
        "extra_heads": c.extra_heads.as_ref().map(|e| json!({"num_heads": e.num_heads, "dim": e.dim})),
    })
}

/// Parse with Kyutai's serde impl and check our serializer round-trips.
fn parse_lm_config(v: &Value) -> Result<moshi::lm::Config> {
    let cfg: moshi::lm::Config = serde_json::from_value(v.clone()).context("parse lm config")?;
    let again = lm_config_json(&cfg);
    ensure!(&again == v, "config does not round-trip:\n{v}\nvs\n{again}");
    Ok(cfg)
}

fn tcfg(over: Value) -> Value {
    let mut base = json!({
        "d_model": 48, "num_heads": 4, "num_layers": 2, "causal": true, "norm_first": true,
        "bias_ff": false, "bias_attn": false, "layer_scale": null,
        "positional_embedding": "Rope", "use_conv_block": false, "cross_attention": null,
        "conv_kernel_size": 3, "use_conv_bias": true, "gating": "silu", "norm": "RmsNorm",
        "context": 5, "max_period": 10000, "max_seq_len": 4096, "kv_repeat": 1,
        "dim_feedforward": 192, "conv_layout": false, "shared_cross_attn": false,
    });
    for (k, v) in over.as_object().unwrap() {
        base[k] = v.clone();
    }
    base
}

// ---------------------------------------------------------------------------
// Probe depformer: Kyutai's `DepFormer::sample`, built from their public
// pieces, with the logits recorded.
// ---------------------------------------------------------------------------

struct ProbeSlice {
    transformer: moshi::transformer::StreamingTransformer,
    emb: MaybeQuantizedEmbedding,
    emb_low_rank: Option<MaybeQuantizedLinear>,
    linear_in: MaybeQuantizedLinear,
    linear_out: MaybeQuantizedLinear,
}

struct ProbeDepformer {
    slices: Vec<ProbeSlice>,
}

impl ProbeDepformer {
    fn new(cfg: &moshi::lm::Config, vb: &MaybeQuantizedVarBuilder) -> Result<Option<Self>> {
        let Some(dcfg) = cfg.depformer.as_ref() else {
            return Ok(None);
        };
        let dim = dcfg.transformer.d_model;
        let vb = vb.pp("depformer");
        let mut slices = vec![];
        for i in 0..dcfg.num_slices {
            let vb = vb.pp(i);
            let in_vs = if i == 0 {
                cfg.text_in_vocab_size
            } else {
                cfg.audio_vocab_size
            };
            let (emb, emb_low_rank) = match dcfg.low_rank_embeddings {
                None => (
                    MaybeQuantizedEmbedding::new(in_vs, dim, vb.pp("emb"))?,
                    None,
                ),
                Some(r) => (
                    MaybeQuantizedEmbedding::new(in_vs, r, vb.pp("emb"))?,
                    Some(moshi::nn::linear(
                        r,
                        dim,
                        false,
                        vb.pp("emb").pp("low_rank"),
                    )?),
                ),
            };
            slices.push(ProbeSlice {
                transformer: moshi::transformer::StreamingTransformer::new(
                    &dcfg.transformer,
                    vb.pp("transformer"),
                )?,
                emb,
                emb_low_rank,
                linear_in: moshi::nn::linear(
                    cfg.transformer.d_model,
                    dim,
                    false,
                    vb.pp("linear_in"),
                )?,
                linear_out: moshi::nn::linear(
                    dim,
                    cfg.audio_vocab_size - 1,
                    false,
                    vb.pp("linear_out"),
                )?,
            });
        }
        Ok(Some(Self { slices }))
    }

    /// Replica of `moshi::lm::DepFormer::sample` returning (tokens, logits).
    fn sample(
        &mut self,
        xs: &Tensor,
        text_token: Option<u32>,
        forced: &[Option<u32>],
        lp: &mut LogitsProcessor,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
        let mut tokens = vec![];
        let mut all_logits = vec![];
        let mut last_token = text_token;
        for idx in 0..self.slices.len() {
            if idx == 0 {
                self.slices[idx].transformer.reset_state();
            } else {
                let (lhs, rhs) = self.slices.split_at_mut(idx);
                rhs[0].transformer.copy_state(&lhs[idx - 1].transformer)?
            }
            let slice = &mut self.slices[idx];
            let xs = slice.linear_in.forward(xs)?;
            let xs = match last_token {
                Some(t) => {
                    let id = Tensor::from_vec(vec![t], (1, 1), xs.device())?;
                    let mut e = slice.emb.forward(&id)?;
                    if let Some(lr) = slice.emb_low_rank.as_ref() {
                        e = e.apply(lr)?;
                    }
                    xs.broadcast_add(&e)?
                }
                None => xs,
            };
            let xs = slice.transformer.forward(&xs)?;
            let logits = xs.apply(&slice.linear_out)?.i((0, 0))?;
            let token = lp.sample(&logits)?;
            all_logits.push(logits.to_vec1::<f32>()?);
            tokens.push(token);
            last_token = Some(forced.get(idx).copied().flatten().unwrap_or(token));
        }
        Ok((tokens, all_logits))
    }
}

fn argmax_lp() -> LogitsProcessor {
    LogitsProcessor::from_sampling(0, Sampling::ArgMax)
}

/// Smallest gap between the best and second-best logit (how safe greedy
/// decoding is against tiny numeric differences).
fn top2_gap(logits: &[f32]) -> f32 {
    let mut v = logits.to_vec();
    v.sort_by(|a, b| b.total_cmp(a));
    v[0] - v[1]
}

fn load_model(
    cfg: &moshi::lm::Config,
    st_bytes: &[u8],
) -> Result<(moshi::lm::LmModel, MaybeQuantizedVarBuilder<'static>)> {
    let vb = VarBuilder::from_buffered_safetensors(st_bytes.to_vec(), DType::F32, &Device::Cpu)?;
    let vb = MaybeQuantizedVarBuilder::Real(vb);
    let model = moshi::lm::LmModel::new(cfg, vb.clone())?;
    Ok((model, vb))
}

fn write_weights(dir: &Path, cfg: &moshi::lm::Config, seed: u64) -> Result<Vec<u8>> {
    let shapes = param_shapes(cfg)?;
    let weights = random_weights(&shapes, seed)?;
    let path = dir.join("model.safetensors");
    candle::safetensors::save(&weights, &path)?;
    Ok(std::fs::read(&path)?)
}

fn tok(t: u32) -> Tensor {
    Tensor::from_vec(vec![t], (1, 1), &Device::Cpu).unwrap()
}

// ---------------------------------------------------------------------------
// Multistream (Moshi-style) fixture
// ---------------------------------------------------------------------------

struct MultistreamSpec {
    name: &'static str,
    lm: Value,
    gen: moshi::lm_generate_multistream::Config,
    /// Conditioner to apply (`(name, lut value)`), if any.
    lut_condition: Option<(&'static str, &'static str)>,
    frames: usize,
    seed: u64,
}

fn multistream_fixture(out: &Path, spec: &MultistreamSpec) -> Result<()> {
    let dir = out.join(spec.name);
    std::fs::create_dir_all(&dir)?;
    let cfg = parse_lm_config(&spec.lm)?;
    let gen = &spec.gen;
    ensure!(gen.total_audio_codebooks() == cfg.audio_codebooks);
    let st = write_weights(&dir, &cfg, spec.seed)?;
    let (mut model, vb) = load_model(&cfg, &st)?;
    let mut probe = ProbeDepformer::new(&cfg, &vb)?.context("multistream needs a depformer")?;
    let conditions = match spec.lut_condition {
        None => None,
        Some((n, v)) => Some(
            model
                .condition_provider()
                .context("no conditioners")?
                .condition_lut(n, v)?,
        ),
    };
    // Same forced tokens as `lm_generate_multistream::State::new` (which
    // hardcodes two streams of 8 codebooks).
    let forced =
        moshi::lm::ForcedAudioTokens::new(gen.acoustic_delay, gen.audio_pad_token(), &[8, 8]);

    // User audio codes (input stream), one row per frame.
    let mut rng = Rng(spec.seed ^ 0xA5A5);
    let card = (gen.audio_vocab_size - 1) as u32;
    let user_audio: Vec<Vec<u32>> = (0..spec.frames)
        .map(|_| {
            (0..gen.input_audio_codebooks)
                .map(|_| rng.below(card))
                .collect()
        })
        .collect();

    // --- Replica of State::step_ (cfg_alpha None, no penalties, no forcing).
    let ungenerated = moshi::lm_generate_multistream::UNGENERATED;
    let total = gen.total_audio_codebooks();
    let mut grid = vec![vec![ungenerated; total]; spec.frames + gen.acoustic_delay];
    let mut text_lp = argmax_lp();
    let mut audio_lp = argmax_lp();
    let mut audio_lp_real = argmax_lp();
    let mut prev_text = gen.text_start_token;
    let mut frames = vec![];
    let mut min_gap = f32::INFINITY;
    for step in 0..spec.frames {
        for (c, &t) in user_audio[step].iter().enumerate() {
            grid[step][c + gen.generated_audio_codebooks] = t;
        }
        let mut codes = vec![];
        for cb in 0..total {
            let t = if cb == 0 || cb == gen.generated_audio_codebooks {
                if step == 0 {
                    gen.audio_pad_token()
                } else {
                    grid[step - 1][cb]
                }
            } else if step <= gen.acoustic_delay {
                gen.audio_pad_token()
            } else {
                grid[step - gen.acoustic_delay - 1][cb]
            };
            ensure!(
                t != ungenerated,
                "ungenerated input at step {step} codebook {cb}"
            );
            codes.push(t);
        }
        let text_in = prev_text;
        let (logits, ys) = model.forward_cond(
            Some(tok(text_in)),
            codes.iter().map(|&c| Some(tok(c))).collect(),
            conditions.as_ref(),
            &().into(),
        )?;
        let text_logits = logits.i((0, 0))?;
        let text_token = text_lp.sample(&text_logits)?;
        let text_logits = text_logits.to_vec1::<f32>()?;
        min_gap = min_gap.min(top2_gap(&text_logits));
        let forced_now = forced.forced_tokens(step);
        let (dep_tokens, dep_logits) =
            probe.sample(&ys, Some(text_token), forced_now, &mut audio_lp)?;
        let real = model
            .depformer_sample(&ys, Some(text_token), forced_now, &mut audio_lp_real)?
            .context("no depformer")?;
        ensure!(real == dep_tokens, "probe depformer disagrees with LmModel::depformer_sample at step {step}: {dep_tokens:?} vs {real:?}");
        for l in &dep_logits {
            min_gap = min_gap.min(top2_gap(l));
        }
        for c in 0..gen.generated_audio_codebooks {
            let delay = if c == 0 { 0 } else { gen.acoustic_delay };
            grid[step.saturating_sub(delay)][c] = dep_tokens[c];
        }
        frames.push(json!({
            "text_in": text_in,
            "audio_in": codes,
            "text_logits": text_logits,
            "text_token": text_token,
            "forced": forced_now,
            "depformer_logits": dep_logits,
            "audio_tokens": dep_tokens,
        }));
        prev_text = text_token;
    }

    // --- Kyutai's real driver on a fresh model.
    let (model2, _) = load_model(&cfg, &st)?;
    let mut state = moshi::lm_generate_multistream::State::new(
        model2,
        spec.frames + 1,
        argmax_lp(),
        argmax_lp(),
        None,
        None,
        None,
        gen.clone(),
    );
    let mut real_text = vec![];
    let mut prev = gen.text_start_token;
    for step in 0..spec.frames {
        let t = state.step_(
            Some(prev),
            &user_audio[step],
            None,
            None,
            conditions.as_ref(),
        )?;
        real_text.push(t);
        prev = t;
    }
    let ours_text: Vec<u32> = frames
        .iter()
        .map(|f| f["text_token"].as_u64().unwrap() as u32)
        .collect();
    ensure!(
        ours_text == real_text,
        "text tokens differ from State: {ours_text:?} vs {real_text:?}"
    );
    let real_grid: Vec<Vec<u32>> = state.audio_tokens(false).to_vec();
    ensure!(
        real_grid[..] == grid[..spec.frames],
        "audio grid differs from State"
    );

    let config = json!({
        "lm": spec.lm,
        "driver": {
            "kind": "multistream",
            "gen": serde_json::to_value(gen)?,
            "lut_condition": spec.lut_condition.map(|(n, v)| json!([n, v])),
        },
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config)? + "\n",
    )?;
    let trace = json!({
        "generator": "tools/kyutai-ref-lm (moshi 0.6.4 = kyutai-labs/moshi@e6a55d27, candle 0.9.1, CPU f32)",
        "seed": spec.seed,
        "user_audio": user_audio,
        "frames": frames,
        "state_text_tokens": real_text,
        "state_audio_tokens": real_grid,
        "state_last_audio_tokens": state.last_audio_tokens(),
    });
    std::fs::write(
        dir.join("trace.json"),
        serde_json::to_string(&trace)? + "\n",
    )?;
    println!(
        "{}: {} frames, min top-2 logit gap {min_gap:.4}",
        spec.name, spec.frames
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// STT (DSM ASR) fixture
// ---------------------------------------------------------------------------

struct SttSpec {
    name: &'static str,
    lm: Value,
    asr_delay_in_tokens: usize,
    /// Value for the `delay` continuous conditioner (`condition_cont("delay", -x)`).
    conditioning_delay: Option<f32>,
    frames: usize,
    seed: u64,
}

/// Returns whether the run produced both `Word` and `EndWord` messages.
fn stt_fixture(out: &Path, spec: &SttSpec) -> Result<bool> {
    let dir = out.join(spec.name);
    std::fs::create_dir_all(&dir)?;
    let cfg = parse_lm_config(&spec.lm)?;
    let st = write_weights(&dir, &cfg, spec.seed)?;
    let (mut model, _) = load_model(&cfg, &st)?;
    let conditions = match spec.conditioning_delay {
        None => None,
        Some(d) => Some(
            model
                .condition_provider()
                .context("no conditioners")?
                .condition_cont("delay", -d)?,
        ),
    };
    let n_q = cfg.audio_codebooks;
    let pad = (cfg.audio_vocab_size - 1) as u32;
    let start = (cfg.text_in_vocab_size - 1) as u32;
    let mut rng = Rng(spec.seed ^ 0x5A5A);
    let codes: Vec<Vec<u32>> = (0..spec.frames)
        .map(|_| (0..n_q).map(|_| rng.below(pad)).collect())
        .collect();

    // --- Replica of asr::State::step_tokens for one batch item.
    let mut next_codebooks = vec![pad; n_q];
    let mut step_idx = 0usize;
    let mut text_token = 0u32;
    let mut frames = vec![];
    let mut min_gap = f32::INFINITY;
    for frame in &codes {
        let audio_in: Vec<u32> = (0..n_q)
            .map(|c| {
                let v = next_codebooks[c];
                next_codebooks[c] = frame[c];
                if step_idx == 0 {
                    pad
                } else {
                    v
                }
            })
            .collect();
        let text_in = if step_idx == 0 { start } else { text_token };
        let (logits, ys) = model.forward_cond(
            Some(tok(text_in)),
            audio_in.iter().map(|&c| Some(tok(c))).collect(),
            conditions.as_ref(),
            &().into(),
        )?;
        let mut prs = vec![];
        for h in model.extra_heads(&ys)? {
            prs.push(
                candle_nn::ops::softmax_last_dim(&h)?
                    .i((0, 0))?
                    .to_vec1::<f32>()?,
            );
        }
        let text_logits = logits.i((0, 0))?;
        let t = text_logits.argmax(0)?.to_scalar::<u32>()?;
        let text_logits = text_logits.to_vec1::<f32>()?;
        min_gap = min_gap.min(top2_gap(&text_logits));
        text_token = t;
        step_idx += 1;
        frames.push(json!({
            "text_in": text_in,
            "audio_in": audio_in,
            "text_logits": text_logits,
            "text_token": t,
            "extra_heads": prs,
        }));
    }

    // --- Kyutai's real driver (with an all-zero Mimi, never called).
    let (model2, _) = load_model(&cfg, &st)?;
    let mimi = moshi::mimi::Mimi::new(
        moshi::mimi::Config::v0_1(Some(n_q)),
        VarBuilder::zeros(DType::F32, &Device::Cpu),
    )?;
    let mut state = moshi::asr::State::new(1, spec.asr_delay_in_tokens, 0.0, mimi, model2)?;
    let seen_text = RefCell::new(vec![]);
    let mut msgs = vec![];
    // One extra frame so the last output token shows up as an input.
    for (i, frame) in codes.iter().chain(std::iter::once(&codes[0])).enumerate() {
        let t = Tensor::from_vec(frame.clone(), (1, n_q, 1), &Device::Cpu)?;
        let out = state.step_tokens(&t, conditions.as_ref(), &().into(), |_, text, _| {
            seen_text
                .borrow_mut()
                .push(text.flatten_all().unwrap().to_vec1::<u32>().unwrap()[0]);
        })?;
        if i < codes.len() {
            msgs.extend(out);
        }
    }
    let seen = seen_text.into_inner();
    for (i, f) in frames.iter().enumerate() {
        ensure!(
            seen[i] == f["text_in"].as_u64().unwrap() as u32,
            "text input differs at {i}"
        );
    }
    let last = frames.last().unwrap()["text_token"].as_u64().unwrap() as u32;
    ensure!(seen[frames.len()] == last, "last text token differs");

    let config = json!({
        "lm": spec.lm,
        "driver": {
            "kind": "asr",
            "asr_delay_in_tokens": spec.asr_delay_in_tokens,
            "conditioning_delay": spec.conditioning_delay,
            "temperature": 0.0,
        },
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config)? + "\n",
    )?;
    let trace = json!({
        "generator": "tools/kyutai-ref-lm (moshi 0.6.4 = kyutai-labs/moshi@e6a55d27, candle 0.9.1, CPU f32)",
        "seed": spec.seed,
        "codes": codes,
        "frames": frames,
        "asr_msgs": serde_json::to_value(&msgs)?,
    });
    std::fs::write(
        dir.join("trace.json"),
        serde_json::to_string(&trace)? + "\n",
    )?;
    let has = |k: &str| {
        trace["asr_msgs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m.get(k).is_some())
    };
    let words = has("Word") && has("EndWord");
    println!(
        "{}: seed {}, {} frames, {} asr messages (words: {words}), min top-2 logit gap {min_gap:.4}",
        spec.name,
        spec.seed,
        spec.frames,
        msgs.len()
    );
    Ok(words)
}

// ---------------------------------------------------------------------------
// Fixture specs
// ---------------------------------------------------------------------------

fn fixtures(out: &Path) -> Result<()> {
    // Moshi v0.1 layout, shrunk: text + 8 generated + 8 input audio codebooks,
    // acoustic delay 2, RmsNorm + SiLU gating (dim_ff = 4 d, so the hidden
    // size takes the 11 d / 4 branch), RoPE, depformer with 8 slices.
    // context = 5 < frames, so the rotating KV cache wraps.
    multistream_fixture(
        out,
        &MultistreamSpec {
            name: "moshi",
            lm: json!({
                "transformer": tcfg(json!({})),
                "depformer": {
                    "transformer": tcfg(json!({
                        "d_model": 16, "num_heads": 2, "num_layers": 2, "dim_feedforward": 64,
                        "context": 8, "positional_embedding": "None",
                    })),
                    "num_slices": 8,
                    "low_rank_embeddings": null,
                },
                "text_in_vocab_size": 33,
                "text_out_vocab_size": 32,
                "audio_vocab_size": 17,
                "audio_codebooks": 16,
                "conditioners": null,
                "extra_heads": null,
            }),
            gen: moshi::lm_generate_multistream::Config {
                generated_audio_codebooks: 8,
                input_audio_codebooks: 8,
                audio_vocab_size: 17,
                acoustic_delay: 2,
                text_pad_token: 3,
                text_eop_token: 0,
                text_start_token: 32,
            },
            lut_condition: None,
            frames: 12,
            seed: 1,
        },
    )?;

    // DSM STT layout (asr_v0_1_1b / stt-1b-en_fr style): no depformer, text
    // + audio input only, extra heads, a continuous `delay` conditioner,
    // max_period 1e5, small context so the cache wraps.
    // Search for the first seed whose greedy transcript contains words
    // (pad = 3 / end-of-padding = 0 tokens around other tokens), so the word
    // segmentation logic of the ASR driver is covered.
    let mut seed = 2;
    while !stt_fixture(
        out,
        &SttSpec {
            name: "stt",
            lm: json!({
                "transformer": tcfg(json!({"context": 6, "max_period": 100000})),
                "depformer": null,
                "text_in_vocab_size": 33,
                "text_out_vocab_size": 32,
                "audio_vocab_size": 17,
                "audio_codebooks": 8,
                "conditioners": {
                    "delay": {"type": "ContinuousAttribute", "dim": 16, "scale_factor": 4.0, "max_period": 100.0},
                },
                "extra_heads": {"num_heads": 2, "dim": 6},
            }),
            asr_delay_in_tokens: 3,
            conditioning_delay: Some(0.5),
            frames: 24,
            seed,
        },
    )? {
        seed += 1;
        ensure!(seed < 1000, "no seed produced ASR words");
    }

    // Every other architectural switch Kyutai's Rust LM supports: LayerNorm,
    // ungated GELU MLP, biases, layer scale, sinusoidal positions on the
    // temporal transformer, RoPE + GELU gating (2/3 hidden branch) + low-rank
    // embeddings in the depformer, text_out == text_in, and a LUT conditioner.
    multistream_fixture(
        out,
        &MultistreamSpec {
            name: "variants",
            lm: json!({
                "transformer": tcfg(json!({
                    "d_model": 32, "num_heads": 2, "dim_feedforward": 80, "gating": null,
                    "norm": "LayerNorm", "layer_scale": 0.1, "bias_ff": true, "bias_attn": true,
                    "positional_embedding": "Sin", "context": 4,
                })),
                "depformer": {
                    "transformer": tcfg(json!({
                        "d_model": 16, "num_heads": 2, "num_layers": 1, "dim_feedforward": 48,
                        "context": 4, "gating": "gelu", "norm": "LayerNorm",
                        "positional_embedding": "Rope",
                    })),
                    "num_slices": 4,
                    "low_rank_embeddings": 8,
                },
                "text_in_vocab_size": 25,
                "text_out_vocab_size": 25,
                "audio_vocab_size": 13,
                "audio_codebooks": 8,
                "conditioners": {
                    "description": {"type": "Lut", "n_bins": 5, "dim": 4, "possible_values": ["bad", "neutral", "good"]},
                },
                "extra_heads": null,
            }),
            gen: moshi::lm_generate_multistream::Config {
                generated_audio_codebooks: 4,
                input_audio_codebooks: 4,
                audio_vocab_size: 13,
                acoustic_delay: 1,
                text_pad_token: 3,
                text_eop_token: 0,
                text_start_token: 24,
            },
            lut_condition: Some(("description", "good")),
            frames: 10,
            seed: 3,
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Full-size manifests
// ---------------------------------------------------------------------------

/// stt-1b-en_fr-like config: asr_v0_1_1b with the 8k tokenizer, 32 codebooks
/// and 4 extra heads (the published candle config also carries those).
fn stt_1b_like() -> moshi::lm::Config {
    let mut c = moshi::lm::Config::asr_v0_1_1b();
    c.text_in_vocab_size = 8001;
    c.text_out_vocab_size = 8000;
    c.audio_codebooks = 32;
    c.extra_heads = Some(moshi::lm::ExtraHeadsConfig {
        num_heads: 4,
        dim: 6,
    });
    c
}

fn manifests(out: &Path) -> Result<()> {
    use moshi::lm::Config as C;
    let presets: Vec<(&str, C)> = vec![
        ("v0_1", C::v0_1()),
        ("v0_1_streaming_8", C::v0_1_streaming(8)),
        ("v0_1_asr", C::v0_1_asr()),
        ("s2s_v0_1", C::s2s_v0_1()),
        ("s2s_2b_16rvq_202501", C::s2s_2b_16rvq_202501()),
        ("asr_v0_1_1b", C::asr_v0_1_1b()),
        ("asr_300m_202501", C::asr_300m_202501()),
        ("stt_1b_like", stt_1b_like()),
        // Cross-attention models (TTS / vision): enumerated so the loader can
        // report them as unsupported rather than mis-load them.
        ("tts_202501", C::tts_202501()),
        ("v0_1_vision", C::v0_1_vision()),
    ];
    let mut all = serde_json::Map::new();
    for (name, cfg) in presets {
        let shapes = param_shapes(&cfg)?;
        let n: usize = shapes.values().map(|s| s.iter().product::<usize>()).sum();
        println!(
            "{name}: {} tensors, {:.3} B params",
            shapes.len(),
            n as f64 / 1e9
        );
        let cfg_json = lm_config_json(&cfg);
        parse_lm_config(&cfg_json)?;
        all.insert(
            name.to_string(),
            json!({"config": cfg_json, "tensors": shapes}),
        );
    }
    std::fs::write(
        out,
        serde_json::to_string_pretty(&Value::Object(all))? + "\n",
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("fixtures") => fixtures(Path::new(args.get(2).context("missing out dir")?)),
        Some("manifests") => manifests(Path::new(args.get(2).context("missing out file")?)),
        _ => bail!("usage: kyutai-ref-lm (fixtures <out_dir> | manifests <out.json>)"),
    }
}
