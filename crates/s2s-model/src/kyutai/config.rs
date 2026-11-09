//! Configuration, mirroring Kyutai's `moshi::lm::Config` and friends.
//!
//! The structs deserialize Kyutai's own config files unchanged: the `model`
//! tables of `moshi-server` TOML configs and the JSON/TOML LM configs of the
//! Rust implementation (e.g. `rust/s2st-1b.toml`). Field names, enum spellings
//! and defaults follow `rust/moshi-core/src/{lm,transformer,conditioner,
//! lm_generate_multistream}.rs` at commit e6a55d27. The presets
//! ([`LmConfig::v0_1`], [`LmConfig::asr_v0_1_1b`], ...) are transcriptions of
//! the constructors in their `lm.rs`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::error::{Error, Result};

/// Activation of a gated MLP. Same variants and serde spellings as
/// `candle_nn::Activation` (lower-case names, plus the `gelu_new` /
/// `gelu_pytorch_tanh` aliases).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    /// Exact (erf) GELU.
    #[serde(alias = "gelu")]
    Gelu,
    /// Tanh-approximated GELU.
    #[serde(alias = "gelu_new")]
    NewGelu,
    /// `max(x, 0)`.
    Relu,
    /// `max(x, 0)^2`.
    Relu2,
    /// `clamp(x, 0, 6)`.
    Relu6,
    /// `x * sigmoid(x)`.
    Silu,
    /// Logistic sigmoid.
    Sigmoid,
    /// `clamp((x + 3) / 6, 0, 1)`.
    HardSigmoid,
    /// SwiGLU on a doubled input (halves the width). Not valid as an MLP
    /// gating activation; rejected at load time.
    Swiglu,
    /// `x * sigmoid(x)`.
    Swish,
    /// `x * tanh(softplus(x))`.
    Mish,
    /// `x * hard_sigmoid(x)`.
    HardSwish,
    /// ELU with the given alpha.
    Elu(f64),
    /// Leaky ReLU with the given slope.
    LeakyRelu(f64),
    /// Tanh-approximated GELU.
    #[serde(alias = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
}

/// Normalization layer type (`moshi::NormType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormType {
    /// RMSNorm, weight `alpha` of shape `[1, 1, d]`, `eps = 1e-8`.
    RmsNorm,
    /// LayerNorm, weight `alpha` `[1, 1, d]` or `weight` `[d]`, plus `bias`, `eps = 1e-5`.
    LayerNorm,
}

/// Positional embedding of a transformer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionalEmbedding {
    /// Interleaved rotary embedding on q and k, `theta = max_period`.
    Rope,
    /// Sinusoidal embedding added to the input.
    Sin,
    /// No positional information.
    None,
}

/// Cross-attention output gating (vision / TTS models). Parsed so configs
/// load; models using cross-attention are rejected at load time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossAttentionGating {
    /// No gating.
    Normal,
    /// Learned scalar through tanh.
    ConstantGatedTanh,
    /// Learned scalar through sigmoid.
    ConstantGatedSigmoid,
    /// Input-conditional MLP gate through tanh.
    ConditionalGatedTanh,
    /// Input-conditional MLP gate through sigmoid.
    ConditionalGatedSigmoid,
    /// Conditional sigmoid gate with a learnable bias.
    ConditionalGatedSigmoidLearnableBias,
    /// Conditional tanh gate with a learnable bias.
    ConditionalGatedTanhLearnableBias,
}

/// One transformer stack (`moshi::transformer::Config`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Model width.
    pub d_model: usize,
    /// Attention heads.
    pub num_heads: usize,
    /// Layers.
    pub num_layers: usize,
    /// Must be true (Kyutai only supports causal streaming).
    pub causal: bool,
    /// Must be true (pre-norm).
    pub norm_first: bool,
    /// Biases on the MLP linears.
    pub bias_ff: bool,
    /// Biases on the attention projections.
    pub bias_attn: bool,
    /// When set, residual branches are scaled by learned `layer_scale_{1,2}.scale`.
    pub layer_scale: Option<f64>,
    /// Positional embedding.
    pub positional_embedding: PositionalEmbedding,
    /// Must be false (unsupported in Kyutai's Rust code too).
    pub use_conv_block: bool,
    /// Cross-attention (vision / TTS). Unsupported by this port.
    pub cross_attention: Option<(CrossAttentionGating, NormType, Option<usize>)>,
    /// Unused (conv block).
    pub conv_kernel_size: usize,
    /// Unused (conv block).
    pub use_conv_bias: bool,
    /// Gated MLP activation; `None` means an ungated `linear1 → GELU → linear2` MLP.
    pub gating: Option<Activation>,
    /// Norm type.
    pub norm: NormType,
    /// Attention context in frames: the KV cache keeps the last `context`
    /// positions (sliding window, including the current one).
    pub context: usize,
    /// RoPE / sinusoidal frequency base.
    pub max_period: usize,
    /// Unused by streaming inference (kept for config compatibility).
    pub max_seq_len: usize,
    /// `num_heads / kv_heads`. Kyutai's forward only supports 1.
    pub kv_repeat: usize,
    /// MLP width (see [`TransformerConfig::gated_hidden`] for gated MLPs).
    pub dim_feedforward: usize,
    /// Unused (conv layout for Mimi).
    pub conv_layout: bool,
    /// Cross-attention weights shared across layers.
    #[serde(default)]
    pub shared_cross_attn: bool,
}

impl TransformerConfig {
    /// Per-head dimension.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    /// Hidden width of a gated MLP: `11 d / 4` when `dim_feedforward == 4 d`
    /// (Moshi's `hidden_scale = 4.125` convention), else `2 dim_feedforward / 3`.
    pub fn gated_hidden(&self) -> usize {
        if self.dim_feedforward == 4 * self.d_model {
            11 * self.d_model / 4
        } else {
            2 * self.dim_feedforward / 3
        }
    }

    /// Output rows of the fused `in_proj` (`q`, `k`, `v`).
    pub fn in_proj_rows(&self) -> usize {
        let num_kv = self.num_heads / self.kv_repeat.max(1);
        self.d_model + 2 * num_kv * self.head_dim()
    }

    /// Check the settings this port (and Kyutai's forward pass) supports.
    pub fn validate(&self, what: &str) -> Result<()> {
        let fail = |m: &str| Err(Error::Unsupported(format!("{what}: {m}")));
        if self.num_heads == 0 || !self.d_model.is_multiple_of(self.num_heads) {
            return Err(Error::Config(format!(
                "{what}: d_model {} not divisible by num_heads {}",
                self.d_model, self.num_heads
            )));
        }
        if !self.causal {
            return fail("only causal transformers are supported");
        }
        if !self.norm_first {
            return fail("only norm_first = true is supported");
        }
        if self.use_conv_block {
            return fail("conv blocks are not supported");
        }
        if self.cross_attention.is_some() {
            return fail("cross-attention (vision / TTS models) is not supported yet");
        }
        if self.kv_repeat != 1 {
            return fail("only kv_repeat = 1 is supported");
        }
        if self.context == 0 {
            return Err(Error::Config(format!("{what}: context must be > 0")));
        }
        if self.positional_embedding == PositionalEmbedding::Rope
            && !self.head_dim().is_multiple_of(2)
        {
            return Err(Error::Config(format!(
                "{what}: RoPE needs an even head dim"
            )));
        }
        if matches!(self.gating, Some(Activation::Swiglu)) {
            return fail("gating = swiglu halves the width and is not a valid MLP gate");
        }
        Ok(())
    }
}

/// Depth transformer (`moshi::lm::DepFormerConfig`).
///
/// In Kyutai's Rust (candle) format every depformer step ("slice") has its own
/// full set of weights: input projection `linear_in`, token embedding `emb`,
/// transformer layers and output head `linear_out`. This is what the PyTorch
/// configs call `depformer_weights_per_step = true` +
/// `depformer_multi_linear = true` (with norms duplicated per step).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DepFormerConfig {
    /// The per-slice transformer. `context` is normally `num_slices`.
    pub transformer: TransformerConfig,
    /// Number of codebooks the depformer generates per frame.
    pub num_slices: usize,
    /// When set, token embeddings are `[vocab, r]` followed by a `[d, r]` linear.
    pub low_rank_embeddings: Option<usize>,
}

/// Extra linear heads on the temporal transformer output (DSM STT models use
/// them for voice-activity / end-of-turn probabilities).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtraHeadsConfig {
    /// Number of heads.
    pub num_heads: usize,
    /// Output width of each head.
    pub dim: usize,
}

/// A lookup-table conditioner (`moshi::conditioner::LutConfig`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LutConfig {
    /// Table rows minus one (the table has `n_bins + 1` rows).
    pub n_bins: usize,
    /// Embedding width before the output projection.
    pub dim: usize,
    /// Allowed values, mapped to rows `0..`.
    pub possible_values: Vec<String>,
}

/// A continuous-attribute conditioner (sinusoidal embedding of a scalar).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContinuousAttributeConfig {
    /// Sinusoidal embedding width.
    pub dim: usize,
    /// The value is multiplied by this before embedding.
    pub scale_factor: f32,
    /// Frequency base.
    pub max_period: f32,
}

/// One conditioner (`moshi::conditioner::ConditionerConfig`, tagged by `type`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ConditionerConfig {
    /// Lookup table.
    Lut(LutConfig),
    /// Continuous scalar.
    ContinuousAttribute(ContinuousAttributeConfig),
}

/// Kyutai multistream LM configuration (`moshi::lm::Config`).
///
/// Stream conventions (from `lm.rs`): the text input vocabulary includes a
/// start token `text_in_vocab_size - 1`; each audio vocabulary includes a
/// padding token `audio_vocab_size - 1`, which the depformer cannot emit (its
/// heads have `audio_vocab_size - 1` outputs).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LmConfig {
    /// Temporal transformer.
    pub transformer: TransformerConfig,
    /// Depth transformer; `None` for STT models that only read audio.
    pub depformer: Option<DepFormerConfig>,
    /// Text input vocabulary (includes the start token).
    pub text_in_vocab_size: usize,
    /// Text output vocabulary (text head width).
    pub text_out_vocab_size: usize,
    /// Audio vocabulary per codebook (includes the padding token).
    pub audio_vocab_size: usize,
    /// Audio codebooks fed to the temporal transformer (all streams).
    pub audio_codebooks: usize,
    /// Conditioners (summed into the input embedding), by name.
    #[serde(default)]
    pub conditioners: Option<BTreeMap<String, ConditionerConfig>>,
    /// Extra output heads.
    #[serde(default)]
    pub extra_heads: Option<ExtraHeadsConfig>,
}

fn tc(
    d_model: usize,
    num_heads: usize,
    num_layers: usize,
    dim_feedforward: usize,
    context: usize,
    max_period: usize,
    gating: Option<Activation>,
    norm: NormType,
    positional_embedding: PositionalEmbedding,
) -> TransformerConfig {
    TransformerConfig {
        d_model,
        num_heads,
        num_layers,
        dim_feedforward,
        causal: true,
        norm_first: true,
        bias_ff: false,
        bias_attn: false,
        layer_scale: None,
        context,
        max_period,
        use_conv_block: false,
        use_conv_bias: true,
        cross_attention: None,
        gating,
        norm,
        positional_embedding,
        conv_layout: false,
        conv_kernel_size: 3,
        kv_repeat: 1,
        max_seq_len: 4096,
        shared_cross_attn: false,
    }
}

const SILU: Option<Activation> = Some(Activation::Silu);
use NormType::{LayerNorm, RmsNorm};
use PositionalEmbedding::Rope;

impl LmConfig {
    /// Kyutai's standard depformer (`Config::depformer_cfg`): 6 layers,
    /// d = 1024, 16 heads, SiLU gating, no positional embedding.
    pub fn depformer_cfg(num_slices: usize) -> DepFormerConfig {
        DepFormerConfig {
            transformer: tc(
                1024,
                16,
                6,
                1024 * 4,
                num_slices,
                10000,
                SILU,
                RmsNorm,
                PositionalEmbedding::None,
            ),
            num_slices,
            low_rank_embeddings: None,
        }
    }

    fn lm(
        transformer: TransformerConfig,
        depformer: Option<DepFormerConfig>,
        text: (usize, usize),
        audio_vocab_size: usize,
        audio_codebooks: usize,
    ) -> Self {
        Self {
            transformer,
            depformer,
            text_in_vocab_size: text.0,
            text_out_vocab_size: text.1,
            audio_vocab_size,
            audio_codebooks,
            conditioners: None,
            extra_heads: None,
        }
    }

    /// Moshi v0.1 (7B, `kyutai/moshiko-*` / `moshika-*`) with 8 input
    /// codebooks. Use [`LmConfig::v0_1_streaming`] for the full-duplex
    /// layout with 16 input codebooks.
    pub fn v0_1() -> Self {
        Self::lm(
            tc(4096, 32, 32, 4096 * 4, 3000, 10000, SILU, RmsNorm, Rope),
            Some(Self::depformer_cfg(8)),
            (32001, 32000),
            2049,
            8,
        )
    }

    /// Moshi v0.1 full duplex: 16 input codebooks (8 Moshi + 8 user) and a
    /// depformer with `num_slices` slices (8 for the released models).
    pub fn v0_1_streaming(num_slices: usize) -> Self {
        let mut s = Self::v0_1();
        s.audio_codebooks = 16;
        if let Some(d) = s.depformer.as_mut() {
            d.num_slices = num_slices;
            d.transformer.context = num_slices;
        }
        s
    }

    /// Moshi v0.1 configured as ASR (depformer with zero slices).
    pub fn v0_1_asr() -> Self {
        let mut s = Self::v0_1();
        s.audio_codebooks = 8;
        if let Some(d) = s.depformer.as_mut() {
            d.num_slices = 0;
            d.transformer.context = 0;
        }
        s
    }

    /// Kyutai's `tts_v0_1` (cross-attention; parse-only in this port).
    pub fn tts_v0_1() -> Self {
        let mut t = tc(2048, 32, 48, 4096 * 2, 4096, 10000, None, LayerNorm, Rope);
        t.cross_attention = Some((CrossAttentionGating::Normal, LayerNorm, None));
        Self::lm(t, Some(Self::depformer_cfg(16)), (32001, 32001), 2050, 16)
    }

    /// Kyutai's `s2s_v0_1` (speech-to-speech translation prototype).
    pub fn s2s_v0_1() -> Self {
        Self::lm(
            tc(2048, 16, 16, 4096 * 2, 3000, 10000, SILU, RmsNorm, Rope),
            Some(Self::depformer_cfg(16)),
            (48001, 48000),
            2049,
            16,
        )
    }

    /// Kyutai's `s2s_v0_1_streaming`.
    pub fn s2s_v0_1_streaming(num_slices: usize) -> Self {
        let mut s = Self::s2s_v0_1();
        s.audio_codebooks = 16;
        if let Some(d) = s.depformer.as_mut() {
            d.num_slices = num_slices;
            d.transformer.context = num_slices;
        }
        s
    }

    /// Kyutai's `asr_v0_1_1b`: the 1B DSM STT backbone (no depformer).
    pub fn asr_v0_1_1b() -> Self {
        Self::lm(
            tc(2048, 16, 16, 2048 * 4, 750, 100_000, SILU, RmsNorm, Rope),
            None,
            (48001, 48000),
            2049,
            8,
        )
    }

    /// Kyutai's `asr_300m_202501`.
    pub fn asr_300m_202501() -> Self {
        Self::lm(
            tc(1024, 8, 16, 1024 * 4, 750, 100_000, SILU, RmsNorm, Rope),
            None,
            (48001, 48000),
            2049,
            32,
        )
    }

    /// Kyutai's `tts_202501` (cross-attention; parse-only in this port).
    pub fn tts_202501() -> Self {
        let mut t = tc(2048, 32, 48, 2048 * 4, 500, 10000, SILU, RmsNorm, Rope);
        t.cross_attention = Some((CrossAttentionGating::Normal, LayerNorm, None));
        Self::lm(t, Some(Self::depformer_cfg(32)), (8001, 8000), 2049, 32)
    }

    /// Kyutai's `s2s_2b_16rvq_202501` (Hibiki-style 2B, 16 codebooks per stream).
    pub fn s2s_2b_16rvq_202501() -> Self {
        Self::lm(
            tc(2560, 20, 24, 2560 * 4, 3000, 100000, SILU, RmsNorm, Rope),
            Some(Self::depformer_cfg(16)),
            (48001, 48000),
            2049,
            32,
        )
    }

    /// Codebooks generated by the depformer per frame (0 without depformer).
    pub fn generated_audio_codebooks(&self) -> usize {
        self.depformer.as_ref().map_or(0, |d| d.num_slices)
    }

    /// Audio padding token (`audio_vocab_size - 1`).
    pub fn audio_pad_token(&self) -> u32 {
        self.audio_vocab_size as u32 - 1
    }

    /// Text start token (`text_in_vocab_size - 1`).
    pub fn text_start_token(&self) -> u32 {
        self.text_in_vocab_size as u32 - 1
    }

    /// Check everything this port supports. Called by the loader.
    pub fn validate(&self) -> Result<()> {
        self.transformer.validate("transformer")?;
        if self.text_in_vocab_size == 0 || self.audio_vocab_size < 2 {
            return Err(Error::Config("empty text or audio vocabulary".into()));
        }
        if let Some(d) = &self.depformer {
            if d.num_slices > 0 {
                d.transformer.validate("depformer.transformer")?;
            }
        }
        Ok(())
    }

    /// Parse a JSON LM config.
    pub fn from_json_str(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| Error::Format(format!("LM config JSON: {e}")))
    }

    /// Parse a TOML LM config (e.g. Kyutai's `rust/s2st-1b.toml`).
    pub fn from_toml_str(s: &str) -> Result<Self> {
        toml::from_str(s).map_err(|e| Error::Format(format!("LM config TOML: {e}")))
    }

    /// Load an LM config from a file. Accepts JSON or TOML (by extension),
    /// either a bare LM config, a `{"lm": ...}` wrapper (our fixtures), or a
    /// `moshi-server` config, in which case the `model` table of the first
    /// module that has one is used.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)?;
        let value = parse_any(path, &text)?;
        let lm = find_lm_value(&value)
            .ok_or_else(|| Error::Config(format!("{}: no LM config found", path.display())))?;
        serde_json::from_value(lm.clone())
            .map_err(|e| Error::Format(format!("{}: {e}", path.display())))
    }
}

/// Parse JSON or TOML (by extension, falling back to trying both) into a
/// JSON value.
pub(crate) fn parse_any(path: &Path, text: &str) -> Result<serde_json::Value> {
    let is_toml = path.extension().is_some_and(|e| e == "toml");
    let as_toml = || -> Result<serde_json::Value> {
        let v: toml::Value =
            toml::from_str(text).map_err(|e| Error::Format(format!("TOML: {e}")))?;
        serde_json::to_value(v).map_err(|e| Error::Format(format!("TOML: {e}")))
    };
    if is_toml {
        as_toml()
    } else {
        serde_json::from_str(text)
            .or_else(|e| as_toml().map_err(|_| Error::Format(format!("{}: {e}", path.display()))))
    }
}

fn find_lm_value(v: &serde_json::Value) -> Option<&serde_json::Value> {
    if v.get("transformer").is_some() && v.get("audio_vocab_size").is_some() {
        return Some(v);
    }
    if let Some(lm) = v.get("lm") {
        return Some(lm);
    }
    v.get("modules")?
        .as_object()?
        .values()
        .find_map(|m| m.get("model"))
}

/// Settings of Kyutai's multistream generation driver
/// (`moshi::lm_generate_multistream::Config`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultistreamConfig {
    /// Codebooks the depformer generates per frame (the model's own voice).
    pub generated_audio_codebooks: usize,
    /// Codebooks of the input (user) audio stream.
    pub input_audio_codebooks: usize,
    /// Audio vocabulary including the padding token.
    pub audio_vocab_size: usize,
    /// Delay of the acoustic codebooks (1..) relative to the semantic one, in frames.
    pub acoustic_delay: usize,
    /// Text padding token (PyTorch configs: `existing_text_padding_id`).
    pub text_pad_token: u32,
    /// End-of-padding token (`existing_text_end_padding_id`).
    pub text_eop_token: u32,
    /// Text start token (`text_in_vocab_size - 1`).
    pub text_start_token: u32,
}

impl MultistreamConfig {
    /// Moshi v0.1 full duplex (8 generated + 8 input codebooks, delay 2).
    pub fn v0_1() -> Self {
        Self {
            generated_audio_codebooks: 8,
            input_audio_codebooks: 8,
            audio_vocab_size: 2049,
            acoustic_delay: 2,
            text_eop_token: 0,
            text_pad_token: 3,
            text_start_token: 32000,
        }
    }

    /// Two generated streams of 8 codebooks, no input stream.
    pub fn v0_1_two_ways() -> Self {
        Self {
            generated_audio_codebooks: 16,
            input_audio_codebooks: 0,
            ..Self::v0_1()
        }
    }

    /// One generated stream, no input.
    pub fn v0_1_one_way() -> Self {
        Self {
            generated_audio_codebooks: 8,
            input_audio_codebooks: 0,
            ..Self::v0_1()
        }
    }

    /// Audio padding token.
    pub fn audio_pad_token(&self) -> u32 {
        self.audio_vocab_size as u32 - 1
    }

    /// All audio codebooks fed to the model.
    pub fn total_audio_codebooks(&self) -> usize {
        self.generated_audio_codebooks + self.input_audio_codebooks
    }

    /// Per-stream delays in Kyutai's PyTorch order (`[text, generated...,
    /// input...]`, the `delays` field of their `config.json`): 0 for text and
    /// for each stream's first (semantic) codebook, `acoustic_delay` for the
    /// rest.
    pub fn delays(&self) -> Vec<usize> {
        let mut d = vec![0];
        for n in [self.generated_audio_codebooks, self.input_audio_codebooks] {
            d.extend((0..n).map(|c| if c == 0 { 0 } else { self.acoustic_delay }));
        }
        d
    }
}

/// Settings of Kyutai's ASR driver (`moshi::asr::State` plus the
/// `moshi-server` `AsrConfig` fields that affect it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AsrConfig {
    /// Frames the transcript lags the audio (6 for `stt-1b-en_fr`, 0.5 s).
    pub asr_delay_in_tokens: usize,
    /// When set, condition on `condition_cont("delay", -conditioning_delay)`.
    #[serde(default)]
    pub conditioning_delay: Option<f32>,
    /// When set, condition on the `delay` conditioner's learnt padding.
    #[serde(default)]
    pub conditioning_learnt_padding: bool,
    /// Sampling temperature for the text stream (0 = greedy).
    #[serde(default)]
    pub temperature: Option<f64>,
}

impl AsrConfig {
    /// Read the ASR settings from a `moshi-server` TOML config: the first
    /// module of type `Asr` or `BatchedAsr`. Also returns its LM config.
    pub fn from_moshi_server_toml(text: &str) -> Result<(Self, LmConfig)> {
        let v: toml::Value =
            toml::from_str(text).map_err(|e| Error::Format(format!("TOML: {e}")))?;
        let v = serde_json::to_value(v).map_err(|e| Error::Format(e.to_string()))?;
        let module = v
            .get("modules")
            .and_then(|m| m.as_object())
            .and_then(|m| {
                m.values().find(|m| {
                    matches!(
                        m.get("type").and_then(|t| t.as_str()),
                        Some("Asr" | "BatchedAsr")
                    )
                })
            })
            .ok_or_else(|| Error::Config("no Asr/BatchedAsr module".into()))?;
        let asr: AsrConfig = serde_json::from_value(module.clone())
            .map_err(|e| Error::Format(format!("ASR module: {e}")))?;
        let lm = module
            .get("model")
            .ok_or_else(|| Error::Config("ASR module has no `model` table".into()))?;
        let lm: LmConfig = serde_json::from_value(lm.clone())
            .map_err(|e| Error::Format(format!("ASR model: {e}")))?;
        Ok((asr, lm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gated_hidden_rules() {
        let v = LmConfig::v0_1();
        assert_eq!(v.transformer.gated_hidden(), 11264);
        let d = LmConfig::depformer_cfg(8);
        assert_eq!(d.transformer.gated_hidden(), 2816);
        let mut t = v.transformer.clone();
        t.dim_feedforward = 48;
        assert_eq!(t.gated_hidden(), 32);
        assert_eq!(v.transformer.in_proj_rows(), 3 * 4096);
    }

    #[test]
    fn activation_spellings() {
        for (s, a) in [
            ("\"silu\"", Activation::Silu),
            ("\"gelu\"", Activation::Gelu),
            ("\"gelu_new\"", Activation::NewGelu),
            ("\"gelu_pytorch_tanh\"", Activation::GeluPytorchTanh),
            ("\"gelupytorchtanh\"", Activation::GeluPytorchTanh),
            ("{\"elu\": 1.0}", Activation::Elu(1.0)),
        ] {
            assert_eq!(serde_json::from_str::<Activation>(s).unwrap(), a);
        }
        assert_eq!(
            serde_json::to_string(&Activation::Silu).unwrap(),
            "\"silu\""
        );
    }

    /// Kyutai's own `rust/s2st-1b.toml` parses.
    #[test]
    fn parses_kyutai_s2st_toml() {
        let text = r#"
text_in_vocab_size = 48001
text_out_vocab_size = 48000
audio_vocab_size = 2049
audio_codebooks = 16

[transformer]
d_model = 2048
num_heads = 16
num_layers = 16
dim_feedforward = 8192
causal = true
norm_first = true
bias_ff = false
bias_attn = false
context = 3000
max_period = 100000
use_conv_block = false
use_conv_bias = true
gating = "silu"
norm = "RmsNorm"
positional_embedding = "Rope"
conv_layout = false
conv_kernel_size = 3
kv_repeat = 1
max_seq_len = 4096

[depformer]
num_slices = 8

[depformer.transformer]
d_model = 1024
num_heads = 16
num_layers = 6
dim_feedforward = 4096
causal = true
norm_first = true
bias_ff = false
bias_attn = false
context = 32
max_period = 10000
use_conv_block = false
use_conv_bias = true
gating = "silu"
norm = "RmsNorm"
positional_embedding = "None"
conv_layout = false
conv_kernel_size = 3
kv_repeat = 1
max_seq_len = 4096

[conditioners.description]
type = "Lut"
n_bins = 31
dim = 16
possible_values = ["very_bad", "bad", "neutral", "good", "very_good"]
"#;
        let c = LmConfig::from_toml_str(text).unwrap();
        assert_eq!(c.audio_codebooks, 16);
        let d = c.depformer.as_ref().unwrap();
        assert_eq!(d.num_slices, 8);
        assert_eq!(d.low_rank_embeddings, None);
        assert_eq!(
            d.transformer.positional_embedding,
            PositionalEmbedding::None
        );
        match &c.conditioners.as_ref().unwrap()["description"] {
            ConditionerConfig::Lut(l) => assert_eq!(l.possible_values.len(), 5),
            other => panic!("unexpected {other:?}"),
        }
        c.validate().unwrap();
        // Round trip.
        let j = serde_json::to_string(&c).unwrap();
        assert_eq!(LmConfig::from_json_str(&j).unwrap(), c);
    }

    #[test]
    fn moshi_server_asr_toml() {
        let text = r#"
[modules.asr]
path = "/api/asr-streaming"
type = "BatchedAsr"
lm_model_file = "hf://kyutai/stt-1b-en_fr-candle/model.safetensors"
asr_delay_in_tokens = 6
batch_size = 64
conditioning_learnt_padding = true
temperature = 0.0

[modules.asr.model]
audio_vocab_size = 2049
text_in_vocab_size = 8001
text_out_vocab_size = 8000
audio_codebooks = 32

[modules.asr.model.transformer]
d_model = 2048
num_heads = 16
num_layers = 16
dim_feedforward = 8192
causal = true
norm_first = true
bias_ff = false
bias_attn = false
context = 750
max_period = 100000
use_conv_block = false
use_conv_bias = true
gating = "silu"
norm = "RmsNorm"
positional_embedding = "Rope"
conv_layout = false
conv_kernel_size = 3
kv_repeat = 1
max_seq_len = 40960

[modules.asr.model.extra_heads]
num_heads = 4
dim = 6
"#;
        let (asr, lm) = AsrConfig::from_moshi_server_toml(text).unwrap();
        assert_eq!(asr.asr_delay_in_tokens, 6);
        assert!(asr.conditioning_learnt_padding);
        assert_eq!(asr.temperature, Some(0.0));
        assert_eq!(lm.audio_codebooks, 32);
        assert!(lm.depformer.is_none());
        assert_eq!(
            lm.extra_heads,
            Some(ExtraHeadsConfig {
                num_heads: 4,
                dim: 6
            })
        );
        lm.validate().unwrap();
    }

    #[test]
    fn unsupported_configs_are_rejected() {
        assert!(matches!(
            LmConfig::tts_202501().validate(),
            Err(Error::Unsupported(_))
        ));
        let mut c = LmConfig::asr_v0_1_1b();
        c.transformer.kv_repeat = 2;
        assert!(matches!(c.validate(), Err(Error::Unsupported(_))));
        for c in [
            LmConfig::v0_1(),
            LmConfig::v0_1_streaming(8),
            LmConfig::v0_1_asr(),
            LmConfig::s2s_v0_1(),
            LmConfig::s2s_v0_1_streaming(16),
            LmConfig::asr_v0_1_1b(),
            LmConfig::asr_300m_202501(),
            LmConfig::s2s_2b_16rvq_202501(),
        ] {
            c.validate().unwrap();
        }
    }

    #[test]
    fn multistream_delays() {
        let m = MultistreamConfig::v0_1();
        let d = m.delays();
        assert_eq!(d.len(), 17);
        assert_eq!(&d[..3], &[0, 0, 2]);
        assert_eq!(d[9], 0);
        assert_eq!(m.audio_pad_token(), 2048);
    }
}
