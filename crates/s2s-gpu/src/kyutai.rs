//! Kyutai's multistream LM on the GPU: [`GpuLm`], an
//! [`LmAccelerator`] for [`LmModel`].
//!
//! Supported configurations (Moshi v0.1, DSM STT 1B/2.6B): RMSNorm, SiLU
//! gating, RoPE or no positional embedding, optional layer scale, no
//! biases, `kv_repeat = 1`, head dim at most 256. Anything else is rejected
//! with [`Error::Unsupported`] at load time.
//!
//! Numerics follow the CPU port: the KV cache is the same rotating cache
//! (slot order and all), RoPE tables are computed on the host with
//! `rope_tables_f32` (f32 angles, like candle), and RMSNorm uses Kyutai's
//! `eps = 1e-8`. With [`WeightDtype::F32`] the GEMMs are strict f32, for
//! parity checks against the CPU; [`WeightDtype::Bf16`] (the default) keeps
//! the checkpoint's bf16 weights and casts activations to bf16 for each GEMM,
//! with f32 accumulation and f32 activations everywhere else, as Kyutai's
//! own bf16 inference does.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::bf16;
use s2s_cuda::gemm::Gemm;
use s2s_kernels::reference::rope::rope_tables_f32;
use s2s_model::kyutai::config::{Activation, NormType, PositionalEmbedding, TransformerConfig};
use s2s_model::kyutai::{
    Error, LmAccelerator, LmConfig, LmModel, Result, SafeTensors, StoredDtype,
};

use crate::simt::{Simt, ATTN_CHUNK, MAX_HEAD_DIM};

/// Kyutai's RMSNorm epsilon.
const RMS_EPS: f32 = 1e-8;

/// Storage type of the weight matrices on the GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightDtype {
    /// bf16 weights, bf16 GEMM inputs, f32 accumulation and outputs.
    #[default]
    Bf16,
    /// f32 weights and strict f32 GEMMs (parity checks; twice the memory).
    F32,
}

/// Options for [`GpuLm::load`].
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuLmOptions {
    /// CUDA device ordinal.
    pub device: usize,
    /// Weight storage type.
    pub dtype: WeightDtype,
}

fn backend<E: std::fmt::Debug>(e: E) -> Error {
    Error::Backend(format!("{e:?}"))
}

/// Load an [`LmModel`] whose compute runs on the GPU: the GPU side reads
/// its weights from `path`, the CPU side only the embeddings, conditioners
/// and extra heads.
pub fn load_lm(cfg: &LmConfig, path: impl AsRef<Path>, opts: GpuLmOptions) -> Result<LmModel> {
    let path = path.as_ref();
    let gpu = GpuLm::load(cfg, path, opts)?;
    let mut st = SafeTensors::open(path)?;
    LmModel::load_accelerated(cfg, &mut st, Box::new(gpu))
}

// ---------------------------------------------------------------------------
// Device handles and weights

struct Dev {
    stream: Arc<CudaStream>,
    gemm: Gemm,
    k: Simt,
}

enum Mat {
    F32(CudaSlice<f32>),
    Bf16(CudaSlice<bf16>),
}

/// `y = x W^T`, `W: [d_out, d_in]`, no bias.
struct Lin {
    w: Mat,
    d_in: usize,
    d_out: usize,
}

impl Dev {
    /// `y = lin(x)`. `xb` is bf16 scratch of exactly `d_in` elements (used
    /// for bf16 weights).
    fn linear(
        &mut self,
        lin: &Lin,
        x: &CudaSlice<f32>,
        xb: &mut CudaSlice<bf16>,
        y: &mut CudaSlice<f32>,
    ) -> Result<()> {
        match &lin.w {
            Mat::F32(w) => self
                .gemm
                .linear_fwd(x, w, None::<&CudaSlice<f32>>, y, 1, lin.d_in, lin.d_out)
                .map_err(backend),
            Mat::Bf16(w) => {
                self.k.f32_to_bf16(&self.stream, x, xb).map_err(backend)?;
                self.gemm
                    .linear_fwd(&*xb, w, None::<&CudaSlice<f32>>, y, 1, lin.d_in, lin.d_out)
                    .map_err(backend)
            }
        }
    }

    fn upload(&self, v: &[f32]) -> Result<CudaSlice<f32>> {
        self.stream.clone_htod(v).map_err(backend)
    }

    fn zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        n: usize,
    ) -> Result<CudaSlice<T>> {
        self.stream.alloc_zeros::<T>(n.max(1)).map_err(backend)
    }
}

/// Reads tensors from a safetensors file and uploads them.
struct Weights<'a> {
    st: SafeTensors,
    dev: &'a Dev,
    dtype: WeightDtype,
}

impl Weights<'_> {
    fn raw(&mut self, name: &str, shape: &[usize]) -> Result<(StoredDtype, Vec<u8>)> {
        let got = self
            .st
            .info(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?
            .shape
            .clone();
        if got != shape {
            return Err(Error::ShapeMismatch {
                name: name.to_string(),
                expected: shape.to_vec(),
                got,
            });
        }
        self.st.read_raw(name)
    }

    fn f32s(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (dt, raw) = self.raw(name, shape)?;
        Ok(match dt {
            StoredDtype::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            StoredDtype::BF16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
                .collect(),
            StoredDtype::F16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::f16::from_bits(u16::from_le_bytes(*b)).to_f32())
                .collect(),
            StoredDtype::F64 => raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b) as f32)
                .collect(),
        })
    }

    fn vec(&mut self, name: &str, shape: &[usize]) -> Result<CudaSlice<f32>> {
        let v = self.f32s(name, shape)?;
        self.dev.upload(&v)
    }

    fn lin(&mut self, name: &str, d_in: usize, d_out: usize) -> Result<Lin> {
        let w = match self.dtype {
            WeightDtype::F32 => Mat::F32(self.vec(name, &[d_out, d_in])?),
            WeightDtype::Bf16 => {
                let (dt, raw) = self.raw(name, &[d_out, d_in])?;
                let v: Vec<bf16> = if dt == StoredDtype::BF16 {
                    raw.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
                        .collect()
                } else {
                    drop(raw);
                    self.f32s(name, &[d_out, d_in])?
                        .into_iter()
                        .map(bf16::from_f32)
                        .collect()
                };
                Mat::Bf16(self.dev.stream.clone_htod(&v).map_err(backend)?)
            }
        };
        Ok(Lin { w, d_in, d_out })
    }
}

// ---------------------------------------------------------------------------
// Transformer stacks

struct Layer {
    norm1: CudaSlice<f32>,
    in_proj: Lin,
    out_proj: Lin,
    norm2: CudaSlice<f32>,
    lin_in: Lin,
    lin_out: Lin,
    ls1: Option<CudaSlice<f32>>,
    ls2: Option<CudaSlice<f32>>,
}

/// Shape of one transformer (shared by every layer and slice using it).
#[derive(Clone, Copy, Debug)]
struct Dims {
    d: usize,
    heads: usize,
    head_dim: usize,
    hidden: usize,
    cap: usize,
    rope: bool,
    max_period: usize,
}

impl Dims {
    fn new(cfg: &TransformerConfig, what: &str) -> Result<Self> {
        let fail = |m: &str| Err(Error::Unsupported(format!("GPU {what}: {m}")));
        if cfg.norm != NormType::RmsNorm {
            return fail("only RMSNorm is supported");
        }
        if cfg.gating != Some(Activation::Silu) {
            return fail("only SiLU-gated MLPs are supported");
        }
        if cfg.bias_ff || cfg.bias_attn {
            return fail("biases are not supported");
        }
        if cfg.kv_repeat != 1 || cfg.cross_attention.is_some() || cfg.use_conv_block {
            return fail("kv_repeat, cross-attention and conv blocks are not supported");
        }
        let rope = match cfg.positional_embedding {
            PositionalEmbedding::Rope => true,
            PositionalEmbedding::None => false,
            PositionalEmbedding::Sin => return fail("sinusoidal positions are not supported"),
        };
        let head_dim = cfg.head_dim();
        if head_dim > MAX_HEAD_DIM || head_dim * cfg.num_heads != cfg.d_model {
            return fail("head_dim must divide d_model and be at most 256");
        }
        Ok(Self {
            d: cfg.d_model,
            heads: cfg.num_heads,
            head_dim,
            hidden: cfg.gated_hidden(),
            cap: cfg.context,
            rope,
            max_period: cfg.max_period,
        })
    }
}

fn load_layer(w: &mut Weights, p: &str, dims: &Dims, layer_scale: bool) -> Result<Layer> {
    let (d, h) = (dims.d, dims.hidden);
    let ls = |w: &mut Weights, n: &str| -> Result<Option<CudaSlice<f32>>> {
        if layer_scale {
            Ok(Some(w.vec(&format!("{p}.{n}.scale"), &[d])?))
        } else {
            Ok(None)
        }
    };
    Ok(Layer {
        norm1: w.vec(&format!("{p}.norm1.alpha"), &[1, 1, d])?,
        in_proj: w.lin(&format!("{p}.self_attn.in_proj_weight"), d, 3 * d)?,
        out_proj: w.lin(&format!("{p}.self_attn.out_proj.weight"), d, d)?,
        norm2: w.vec(&format!("{p}.norm2.alpha"), &[1, 1, d])?,
        lin_in: w.lin(&format!("{p}.gating.linear_in.weight"), d, 2 * h)?,
        lin_out: w.lin(&format!("{p}.gating.linear_out.weight"), h, d)?,
        ls1: ls(w, "layer_scale_1")?,
        ls2: ls(w, "layer_scale_2")?,
    })
}

/// Per-stack KV caches (one per layer, `[heads, cap, head_dim]`) and the
/// rotating-cache counters shared by all layers.
struct Kv {
    k: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
    /// Filled slots (`<= cap`).
    len: usize,
    /// Slot written next.
    offset: usize,
    /// Positions since reset.
    total: usize,
}

impl Kv {
    fn new(dev: &Dev, dims: &Dims, layers: usize) -> Result<Self> {
        let n = dims.d * dims.cap;
        Ok(Self {
            k: (0..layers).map(|_| dev.zeros(n)).collect::<Result<_>>()?,
            v: (0..layers).map(|_| dev.zeros(n)).collect::<Result<_>>()?,
            len: 0,
            offset: 0,
            total: 0,
        })
    }

    fn reset(&mut self) {
        (self.len, self.offset, self.total) = (0, 0, 0);
    }
}

/// Activation buffers of one stack, each exactly the size its kernel or
/// GEMM expects.
struct Scratch {
    xn: CudaSlice<f32>,
    xn_bf: CudaSlice<bf16>,
    qkv: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    mlp_y: CudaSlice<f32>,
    mlp_h: CudaSlice<f32>,
    mlp_h_bf: CudaSlice<bf16>,
    po: CudaSlice<f32>,
    pml: CudaSlice<f32>,
    cos: CudaSlice<f32>,
    sin: CudaSlice<f32>,
}

impl Scratch {
    fn new(dev: &Dev, dims: &Dims) -> Result<Self> {
        let splits = dims.heads * dims.cap.div_ceil(ATTN_CHUNK);
        Ok(Self {
            xn: dev.zeros(dims.d)?,
            xn_bf: dev.zeros(dims.d)?,
            qkv: dev.zeros(3 * dims.d)?,
            attn: dev.zeros(dims.d)?,
            proj: dev.zeros(dims.d)?,
            mlp_y: dev.zeros(2 * dims.hidden)?,
            mlp_h: dev.zeros(dims.hidden)?,
            mlp_h_bf: dev.zeros(dims.hidden)?,
            po: dev.zeros(splits * dims.head_dim)?,
            pml: dev.zeros(splits * 2)?,
            cos: dev.zeros(dims.head_dim / 2)?,
            sin: dev.zeros(dims.head_dim / 2)?,
        })
    }
}

/// One position through `layers`, in place on `x` (Kyutai's
/// `StreamingTransformer::forward` with `t = 1`).
fn run_layers(
    dev: &mut Dev,
    layers: &[Layer],
    kv: &mut Kv,
    sc: &mut Scratch,
    dims: &Dims,
    x: &mut CudaSlice<f32>,
) -> Result<()> {
    let d = dims.d;
    if dims.rope {
        let (cos, sin) = rope_tables_f32(&[kv.total as i64], dims.head_dim, dims.max_period as f32);
        dev.stream.memcpy_htod(&cos, &mut sc.cos).map_err(backend)?;
        dev.stream.memcpy_htod(&sin, &mut sc.sin).map_err(backend)?;
    }
    let slot = kv.offset;
    let n = (kv.len + 1).min(dims.cap);
    for (l, layer) in layers.iter().enumerate() {
        let s = dev.stream.clone();
        dev.k
            .rms_norm(&s, x, &layer.norm1, &mut sc.xn, RMS_EPS)
            .map_err(backend)?;
        dev.linear(&layer.in_proj, &sc.xn, &mut sc.xn_bf, &mut sc.qkv)?;
        if dims.rope {
            dev.k
                .rope_interleaved(&s, &mut sc.qkv, &sc.cos, &sc.sin, d)
                .map_err(backend)?;
        }
        dev.k
            .kv_append(
                &s,
                &sc.qkv,
                &mut kv.k[l],
                &mut kv.v[l],
                d,
                dims.head_dim,
                dims.cap,
                slot,
            )
            .map_err(backend)?;
        dev.k
            .attn_decode(
                &s,
                &sc.qkv,
                &kv.k[l],
                &kv.v[l],
                &mut sc.po,
                &mut sc.pml,
                &mut sc.attn,
                dims.heads,
                dims.head_dim,
                dims.cap,
                n,
            )
            .map_err(backend)?;
        dev.linear(&layer.out_proj, &sc.attn, &mut sc.xn_bf, &mut sc.proj)?;
        dev.k
            .add_inplace(&s, x, &sc.proj, layer.ls1.as_ref())
            .map_err(backend)?;
        dev.k
            .rms_norm(&s, x, &layer.norm2, &mut sc.xn, RMS_EPS)
            .map_err(backend)?;
        dev.linear(&layer.lin_in, &sc.xn, &mut sc.xn_bf, &mut sc.mlp_y)?;
        dev.k
            .glu_silu(&s, &sc.mlp_y, &mut sc.mlp_h)
            .map_err(backend)?;
        dev.linear(&layer.lin_out, &sc.mlp_h, &mut sc.mlp_h_bf, &mut sc.proj)?;
        dev.k
            .add_inplace(&s, x, &sc.proj, layer.ls2.as_ref())
            .map_err(backend)?;
    }
    kv.offset = (kv.offset + 1) % dims.cap;
    kv.len = n;
    kv.total += 1;
    Ok(())
}

// ---------------------------------------------------------------------------
// The model

struct DepSlice {
    linear_in: Lin,
    layers: Vec<Layer>,
    linear_out: Lin,
}

struct Depformer {
    dims: Dims,
    slices: Vec<DepSlice>,
    kv: Kv,
    sc: Scratch,
    /// The temporal hidden state of the current frame `[d_temporal]`.
    hidden: CudaSlice<f32>,
    hidden_bf: CudaSlice<bf16>,
    x: CudaSlice<f32>,
    emb: CudaSlice<f32>,
    logits: CudaSlice<f32>,
}

/// Kyutai's multistream LM compute on one GPU; see the module docs.
pub struct GpuLm {
    dev: Dev,
    dims: Dims,
    layers: Vec<Layer>,
    kv: Kv,
    sc: Scratch,
    out_norm: CudaSlice<f32>,
    text_linear: Lin,
    dep: Option<Depformer>,
    x: CudaSlice<f32>,
    hidden: CudaSlice<f32>,
    logits: CudaSlice<f32>,
}

impl std::fmt::Debug for GpuLm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuLm")
            .field("dims", &self.dims)
            .field("layers", &self.layers.len())
            .field(
                "depformer_slices",
                &self.dep.as_ref().map_or(0, |d| d.slices.len()),
            )
            .field("seq_len", &self.kv.total)
            .finish_non_exhaustive()
    }
}

impl GpuLm {
    /// Load the GPU half of `cfg` from a safetensors checkpoint with Kyutai's
    /// tensor names.
    pub fn load(cfg: &LmConfig, path: &Path, opts: GpuLmOptions) -> Result<Self> {
        cfg.validate()?;
        let dims = Dims::new(&cfg.transformer, "transformer")?;
        let ctx = CudaContext::new(opts.device).map_err(backend)?;
        // One stream, used in order: cudarc's cross-stream event tracking
        // would only add overhead.
        // SAFETY: every buffer here is used on `stream` only.
        #[allow(unsafe_code)]
        unsafe {
            ctx.disable_event_tracking()
        };
        let stream = ctx.default_stream();
        let mut gemm = Gemm::new(stream.clone()).map_err(backend)?;
        if opts.dtype == WeightDtype::F32 {
            gemm = gemm.with_f32_math(s2s_cuda::gemm::F32Math::Strict);
        }
        let dev = Dev {
            k: Simt::load(&ctx).map_err(backend)?,
            gemm,
            stream,
        };
        let mut w = Weights {
            st: SafeTensors::open(path)?,
            dev: &dev,
            dtype: opts.dtype,
        };
        let d = dims.d;
        let ls = cfg.transformer.layer_scale.is_some();
        let layers = (0..cfg.transformer.num_layers)
            .map(|i| load_layer(&mut w, &format!("transformer.layers.{i}"), &dims, ls))
            .collect::<Result<Vec<_>>>()?;
        let out_norm = w.vec("out_norm.alpha", &[1, 1, d])?;
        let text_linear = w.lin("text_linear.weight", d, cfg.text_out_vocab_size)?;
        let dep = match &cfg.depformer {
            Some(dc) if dc.num_slices > 0 => {
                let dd = Dims::new(&dc.transformer, "depformer")?;
                let dls = dc.transformer.layer_scale.is_some();
                let vocab = cfg.audio_vocab_size - 1;
                let slices = (0..dc.num_slices)
                    .map(|s| {
                        let p = format!("depformer.{s}");
                        Ok(DepSlice {
                            linear_in: w.lin(&format!("{p}.linear_in.weight"), d, dd.d)?,
                            layers: (0..dc.transformer.num_layers)
                                .map(|i| {
                                    load_layer(
                                        &mut w,
                                        &format!("{p}.transformer.layers.{i}"),
                                        &dd,
                                        dls,
                                    )
                                })
                                .collect::<Result<_>>()?,
                            linear_out: w.lin(&format!("{p}.linear_out.weight"), dd.d, vocab)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Some(Depformer {
                    kv: Kv::new(&dev, &dd, dc.transformer.num_layers)?,
                    sc: Scratch::new(&dev, &dd)?,
                    hidden: dev.zeros(d)?,
                    hidden_bf: dev.zeros(d)?,
                    x: dev.zeros(dd.d)?,
                    emb: dev.zeros(dd.d)?,
                    logits: dev.zeros(vocab)?,
                    dims: dd,
                    slices,
                })
            }
            _ => None,
        };
        drop(w);
        Ok(Self {
            kv: Kv::new(&dev, &dims, layers.len())?,
            sc: Scratch::new(&dev, &dims)?,
            x: dev.zeros(d)?,
            hidden: dev.zeros(d)?,
            logits: dev.zeros(cfg.text_out_vocab_size)?,
            dev,
            dims,
            layers,
            out_norm,
            text_linear,
            dep,
        })
    }
}

impl LmAccelerator for GpuLm {
    fn step(&mut self, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if x.len() != self.dims.d {
            return Err(Error::Input(format!(
                "input embedding has width {}, expected {}",
                x.len(),
                self.dims.d
            )));
        }
        let s = self.dev.stream.clone();
        s.memcpy_htod(x, &mut self.x).map_err(backend)?;
        run_layers(
            &mut self.dev,
            &self.layers,
            &mut self.kv,
            &mut self.sc,
            &self.dims,
            &mut self.x,
        )?;
        self.dev
            .k
            .rms_norm(&s, &self.x, &self.out_norm, &mut self.hidden, RMS_EPS)
            .map_err(backend)?;
        self.dev.linear(
            &self.text_linear,
            &self.hidden,
            &mut self.sc.xn_bf,
            &mut self.logits,
        )?;
        let hidden = s.clone_dtoh(&self.hidden).map_err(backend)?;
        let logits = s.clone_dtoh(&self.logits).map_err(backend)?;
        Ok((hidden, logits))
    }

    fn reset(&mut self) {
        self.kv.reset();
    }

    fn current_seq_len(&self) -> usize {
        self.kv.total
    }

    fn depformer_begin(&mut self, hidden: &[f32]) -> Result<()> {
        let dep = self
            .dep
            .as_mut()
            .ok_or_else(|| Error::Config("model has no depformer".into()))?;
        if hidden.len() != self.dims.d {
            return Err(Error::Input(format!(
                "hidden has width {}, expected {}",
                hidden.len(),
                self.dims.d
            )));
        }
        self.dev
            .stream
            .memcpy_htod(hidden, &mut dep.hidden)
            .map_err(backend)?;
        dep.kv.reset();
        Ok(())
    }

    fn depformer_slice(&mut self, s: usize, emb: Option<&[f32]>) -> Result<Vec<f32>> {
        let dep = self
            .dep
            .as_mut()
            .ok_or_else(|| Error::Config("model has no depformer".into()))?;
        let slice = dep
            .slices
            .get(s)
            .ok_or_else(|| Error::Input(format!("depformer slice {s} out of range")))?;
        let stream = self.dev.stream.clone();
        self.dev.linear(
            &slice.linear_in,
            &dep.hidden,
            &mut dep.hidden_bf,
            &mut dep.x,
        )?;
        if let Some(e) = emb {
            if e.len() != dep.dims.d {
                return Err(Error::Input(format!(
                    "depformer embedding has width {}, expected {}",
                    e.len(),
                    dep.dims.d
                )));
            }
            stream.memcpy_htod(e, &mut dep.emb).map_err(backend)?;
            self.dev
                .k
                .add_inplace(&stream, &mut dep.x, &dep.emb, None)
                .map_err(backend)?;
        }
        run_layers(
            &mut self.dev,
            &slice.layers,
            &mut dep.kv,
            &mut dep.sc,
            &dep.dims,
            &mut dep.x,
        )?;
        self.dev.linear(
            &slice.linear_out,
            &dep.x,
            &mut dep.sc.xn_bf,
            &mut dep.logits,
        )?;
        stream.clone_dtoh(&dep.logits).map_err(backend)
    }
}
