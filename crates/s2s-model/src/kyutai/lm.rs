//! The multistream LM (`moshi::lm::LmModel`): embeddings, temporal
//! transformer, text head, depformer, extra heads and conditioners.
//!
//! # Tensor names (Kyutai's Rust / candle checkpoints)
//!
//! | Tensor | Shape |
//! |---|---|
//! | `text_emb.weight` | `[text_in_vocab, d]` |
//! | `emb.{c}.weight` | `[audio_vocab, d]` per input codebook |
//! | `transformer.layers.{l}.self_attn.in_proj_weight` (+ `in_proj_bias`) | `[3 d, d]` |
//! | `transformer.layers.{l}.self_attn.out_proj.{weight,bias}` | `[d, d]` |
//! | `transformer.layers.{l}.norm{1,2}.alpha` (RMS) / `.{alpha\|weight,bias}` (LN) | `[1, 1, d]` / `[d]` |
//! | `transformer.layers.{l}.gating.linear_{in,out}.weight` | `[2 h, d]`, `[d, h]` |
//! | `transformer.layers.{l}.linear{1,2}.weight` (ungated) | `[ff, d]`, `[d, ff]` |
//! | `transformer.layers.{l}.layer_scale_{1,2}.scale` | `[d]` |
//! | `out_norm.*`, `text_linear.weight` | norm, `[text_out_vocab, d]` |
//! | `depformer.{s}.linear_in.weight` | `[d_dep, d]` |
//! | `depformer.{s}.emb.weight` (+ `emb.low_rank.weight`) | `[vocab_s, d_dep \| r]` (`[d_dep, r]`) |
//! | `depformer.{s}.transformer.layers.{l}.*` | as above, width `d_dep` |
//! | `depformer.{s}.linear_out.weight` | `[audio_vocab - 1, d_dep]` |
//! | `extra_heads.{i}.weight` | `[dim, d]` |
//! | `condition_provider.conditioners.{name}.*` | see [`conditioner`](super::conditioner) |
//!
//! `vocab_0` is the text input vocabulary (slice 0 embeds the text token just
//! sampled), the other slices embed audio tokens.

use std::collections::BTreeMap;
use std::path::Path;

use s2s_kernels::reference::embedding::{embedding_sum_fwd, NO_TOKEN};

use super::conditioner::{Condition, ConditionProvider};
use super::config::{DepFormerConfig, LmConfig};
use super::error::{Error, Result};
use super::layers::{
    load_layers, new_caches, run_layers, KvCache, Layer, Linear, Norm, PosEnc, StreamingTransformer,
};
use super::sampling::Sampler;
use super::weights::{join, Loader, SafeTensors, ShapeManifest, WeightSource};

/// Output of one temporal-transformer step.
#[derive(Clone, Debug, PartialEq)]
pub struct StepOutput {
    /// Text logits `[text_out_vocab_size]`.
    pub text_logits: Vec<f32>,
    /// Transformer output after `out_norm` `[d_model]`: the depformer and
    /// extra-head input (Kyutai's `ys`).
    pub hidden: Vec<f32>,
}

/// Output of one depformer pass.
#[derive(Clone, Debug, PartialEq)]
pub struct DepformerOutput {
    /// Sampled token per slice (codebook).
    pub tokens: Vec<u32>,
    /// Logits per slice `[num_slices][audio_vocab_size - 1]`.
    pub logits: Vec<Vec<f32>>,
}

/// Result of [`LmModel::check_tensor_names`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameCheck {
    /// Tensors the config needs (all present with the right shapes).
    pub used: usize,
    /// Tensors in the checkpoint that the config does not use.
    pub unused: Vec<String>,
}

#[derive(Clone, Debug)]
struct DepSlice {
    linear_in: Linear,
    emb: Vec<f32>,
    emb_dim: usize,
    emb_vocab: usize,
    low_rank: Option<Linear>,
    layers: Vec<Layer>,
    linear_out: Linear,
}

/// The depth transformer. Kyutai gives each slice its own transformer and
/// copies the previous slice's KV cache into it before stepping; that is the
/// same as one cache per layer shared by all slices of a frame, which is what
/// this keeps (reset at slice 0).
#[derive(Clone, Debug)]
struct DepFormer {
    slices: Vec<DepSlice>,
    caches: Vec<KvCache>,
    pos_enc: PosEnc,
}

impl DepFormer {
    fn load(l: &mut Loader, cfg: &LmConfig, dcfg: &DepFormerConfig) -> Result<Self> {
        let dim = dcfg.transformer.d_model;
        let mut slices = Vec::with_capacity(dcfg.num_slices);
        for s in 0..dcfg.num_slices {
            let p = join("depformer", s);
            let emb_vocab = if s == 0 {
                cfg.text_in_vocab_size
            } else {
                cfg.audio_vocab_size
            };
            let emb_dim = dcfg.low_rank_embeddings.unwrap_or(dim);
            let emb = l.get(&join(&p, "emb.weight"), &[emb_vocab, emb_dim])?;
            let low_rank = match dcfg.low_rank_embeddings {
                None => None,
                Some(r) => Some(Linear::load(l, &join(&p, "emb.low_rank"), r, dim, false)?),
            };
            slices.push(DepSlice {
                linear_in: Linear::load(
                    l,
                    &join(&p, "linear_in"),
                    cfg.transformer.d_model,
                    dim,
                    false,
                )?,
                emb,
                emb_dim,
                emb_vocab,
                low_rank,
                layers: load_layers(l, &join(&p, "transformer"), &dcfg.transformer)?,
                linear_out: Linear::load(
                    l,
                    &join(&p, "linear_out"),
                    dim,
                    cfg.audio_vocab_size - 1,
                    false,
                )?,
            });
        }
        Ok(Self {
            slices,
            caches: new_caches(&dcfg.transformer),
            pos_enc: PosEnc::new(&dcfg.transformer),
        })
    }

    /// Kyutai's `DepFormer::sample`.
    fn sample(
        &mut self,
        hidden: &[f32],
        text_token: Option<u32>,
        forced: &[Option<u32>],
        sampler: &mut Sampler,
    ) -> Result<DepformerOutput> {
        self.caches.iter_mut().for_each(KvCache::reset);
        let mut tokens = Vec::with_capacity(self.slices.len());
        let mut logits = Vec::with_capacity(self.slices.len());
        let mut last_token = text_token;
        for (s, slice) in self.slices.iter().enumerate() {
            let mut x = slice.linear_in.forward(hidden);
            if let Some(t) = last_token {
                let t = t as usize;
                if t >= slice.emb_vocab {
                    return Err(Error::Input(format!(
                        "depformer slice {s}: token {t} out of range (vocab {})",
                        slice.emb_vocab
                    )));
                }
                let row = &slice.emb[t * slice.emb_dim..(t + 1) * slice.emb_dim];
                let e = match &slice.low_rank {
                    None => row.to_vec(),
                    Some(lr) => lr.forward(row),
                };
                x.iter_mut().zip(&e).for_each(|(x, e)| *x += e);
            }
            let y = run_layers(&slice.layers, &mut self.caches, &self.pos_enc, &x);
            let l = slice.linear_out.forward(&y);
            let token = sampler.sample(&l)?;
            logits.push(l);
            tokens.push(token);
            last_token = Some(forced.get(s).copied().flatten().unwrap_or(token));
        }
        Ok(DepformerOutput { tokens, logits })
    }
}

/// Kyutai's multistream LM, batch 1, streaming one frame per [`LmModel::step`].
#[derive(Clone, Debug)]
pub struct LmModel {
    cfg: LmConfig,
    text_emb: Vec<f32>,
    audio_embs: Vec<Vec<f32>>,
    transformer: StreamingTransformer,
    out_norm: Norm,
    text_linear: Linear,
    depformer: Option<DepFormer>,
    extra_heads: Vec<Linear>,
    condition_provider: Option<ConditionProvider>,
}

impl LmModel {
    fn load_with(cfg: &LmConfig, l: &mut Loader) -> Result<Self> {
        cfg.validate()?;
        let d = cfg.transformer.d_model;
        // A depformer with 0 slices (`v0_1_asr`) still exists, as in Kyutai:
        // `depformer_sample` then returns `Some(empty)`.
        let depformer = match &cfg.depformer {
            Some(dc) => Some(DepFormer::load(l, cfg, dc)?),
            None => None,
        };
        let text_emb = l.get("text_emb.weight", &[cfg.text_in_vocab_size, d])?;
        let out_norm = Norm::load(l, "out_norm", d, cfg.transformer.norm)?;
        let text_linear = Linear::load(l, "text_linear", d, cfg.text_out_vocab_size, false)?;
        let transformer = StreamingTransformer::load(l, "transformer", &cfg.transformer)?;
        let audio_embs = (0..cfg.audio_codebooks)
            .map(|c| l.get(&format!("emb.{c}.weight"), &[cfg.audio_vocab_size, d]))
            .collect::<Result<_>>()?;
        let condition_provider = match &cfg.conditioners {
            None => None,
            Some(c) => Some(ConditionProvider::load(l, "condition_provider", d, c)?),
        };
        let extra_heads = match &cfg.extra_heads {
            None => vec![],
            Some(e) => (0..e.num_heads)
                .map(|i| Linear::load(l, &join("extra_heads", i), d, e.dim, false))
                .collect::<Result<_>>()?,
        };
        Ok(Self {
            cfg: cfg.clone(),
            text_emb,
            audio_embs,
            transformer,
            out_norm,
            text_linear,
            depformer,
            extra_heads,
            condition_provider,
        })
    }

    /// Build the model from any weight source, checking every tensor's name
    /// and shape. Extra tensors in the source are ignored (as in Kyutai).
    pub fn load(cfg: &LmConfig, src: &mut dyn WeightSource) -> Result<Self> {
        let mut l = Loader::new(src, true);
        Self::load_with(cfg, &mut l)
    }

    /// Load from a safetensors file with Kyutai's tensor names (f32, bf16 or
    /// f16; converted to f32).
    pub fn load_safetensors(cfg: &LmConfig, path: impl AsRef<Path>) -> Result<Self> {
        let mut st = SafeTensors::open(path)?;
        Self::load(cfg, &mut st)
    }

    /// Check that a checkpoint with these tensor names and shapes has
    /// everything `cfg` needs, without loading any data. Fails like
    /// [`LmModel::load`] would (missing tensor, wrong shape, unsupported
    /// config); on success reports the unused tensors.
    pub fn check_tensor_names(
        cfg: &LmConfig,
        shapes: &BTreeMap<String, Vec<usize>>,
    ) -> Result<NameCheck> {
        let mut src = ShapeManifest(shapes);
        let mut l = Loader::new(&mut src, false);
        Self::load_with(cfg, &mut l)?;
        Ok(NameCheck {
            used: l.used_count(),
            unused: l.unused(),
        })
    }

    /// The configuration.
    pub fn config(&self) -> &LmConfig {
        &self.cfg
    }

    /// Clear the temporal transformer's KV cache (Kyutai's `reset_state`).
    pub fn reset_state(&mut self) {
        self.transformer.reset();
    }

    /// Frames processed since the last reset (the next RoPE position).
    pub fn current_seq_len(&self) -> usize {
        self.transformer.current_seq_len()
    }

    /// Audio codebooks the temporal transformer reads.
    pub fn in_audio_codebooks(&self) -> usize {
        self.audio_embs.len()
    }

    /// Audio codebooks the depformer generates.
    pub fn generated_audio_codebooks(&self) -> usize {
        self.depformer.as_ref().map_or(0, |d| d.slices.len())
    }

    /// Audio padding token.
    pub fn audio_pad_token(&self) -> u32 {
        self.cfg.audio_pad_token()
    }

    /// Text start token.
    pub fn text_start_token(&self) -> u32 {
        self.cfg.text_start_token()
    }

    /// The conditioners, if the model has any.
    pub fn condition_provider(&self) -> Option<&ConditionProvider> {
        self.condition_provider.as_ref()
    }

    /// One temporal-transformer step (Kyutai's `forward_cond` with `t = 1`).
    ///
    /// The input embedding is `text_emb[text]` (zeros when `None`) plus
    /// `emb[c][audio[c]]` for each codebook given (`None` = no contribution;
    /// `audio` may be shorter than [`LmModel::in_audio_codebooks`], like
    /// Kyutai's zip), plus the condition. Then the transformer, `out_norm`,
    /// and the text head.
    pub fn step(
        &mut self,
        text: Option<u32>,
        audio: &[Option<u32>],
        condition: Option<&Condition>,
    ) -> Result<StepOutput> {
        let d = self.cfg.transformer.d_model;
        if audio.len() > self.audio_embs.len() {
            return Err(Error::Input(format!(
                "{} audio tokens for {} codebooks",
                audio.len(),
                self.audio_embs.len()
            )));
        }
        let mut tables: Vec<&[f32]> = vec![&self.text_emb];
        let mut vocab = vec![self.cfg.text_in_vocab_size];
        let mut tokens = vec![text.unwrap_or(NO_TOKEN)];
        for (c, t) in audio.iter().enumerate() {
            tables.push(&self.audio_embs[c]);
            vocab.push(self.cfg.audio_vocab_size);
            tokens.push(t.unwrap_or(NO_TOKEN));
        }
        for (s, &t) in tokens.iter().enumerate() {
            if t != NO_TOKEN && t as usize >= vocab[s] {
                return Err(Error::Input(format!(
                    "token {t} out of range for stream {s} (vocab {})",
                    vocab[s]
                )));
            }
        }
        let mut x = vec![0.0f32; d];
        embedding_sum_fwd(&tables, &vocab, &tokens, 1, d, &mut x);
        if let Some(Condition(c)) = condition {
            if c.len() != d {
                return Err(Error::Input(format!(
                    "condition has width {}, expected {d}",
                    c.len()
                )));
            }
            x.iter_mut().zip(c).for_each(|(x, c)| *x += c);
        }
        let y = self.transformer.step(&x);
        let hidden = self.out_norm.forward(&y);
        let text_logits = self.text_linear.forward(&hidden);
        Ok(StepOutput {
            text_logits,
            hidden,
        })
    }

    /// Extra heads applied to the step's `hidden` (raw outputs, no softmax).
    pub fn extra_heads(&self, hidden: &[f32]) -> Vec<Vec<f32>> {
        self.extra_heads.iter().map(|h| h.forward(hidden)).collect()
    }

    /// Run the depformer on `hidden` (Kyutai's `depformer_sample`): slice 0
    /// embeds `text_token`, slice `s > 0` embeds the token of slice `s - 1`,
    /// or `forced[s - 1]` when that is `Some`. `None` without a depformer.
    pub fn depformer_sample(
        &mut self,
        hidden: &[f32],
        text_token: Option<u32>,
        forced: &[Option<u32>],
        sampler: &mut Sampler,
    ) -> Result<Option<DepformerOutput>> {
        match self.depformer.as_mut() {
            None => Ok(None),
            Some(d) => d.sample(hidden, text_token, forced, sampler).map(Some),
        }
    }
}
