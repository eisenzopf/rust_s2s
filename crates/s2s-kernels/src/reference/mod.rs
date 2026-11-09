//! CPU reference implementations (forward and backward).
//!
//! | Module | Ops | Plan §7.2 kernel # |
//! |---|---|---|
//! | [`gemm`] | GEMM with transposes, `Linear` fwd/bwd | 1 |
//! | [`norm`] | RMSNorm fwd/bwd (optional fused residual) | 2 |
//! | [`rope`] | Rotary position embedding fwd/bwd, explicit positions | 3 |
//! | [`activation`] | SwiGLU fwd/bwd | 4 |
//! | [`attention`] | Causal / sliding-window / pinned-prefix GQA attention fwd/bwd, single-query decode | 5, 6 |
//! | [`embedding`] | Multi-stream embedding gather-and-sum fwd / scatter-add bwd | 7 |
//! | [`loss`] | Softmax, log-softmax, weighted cross-entropy with ignore index | 8, 9 |
//! | [`optim`] | AdamW, global grad-norm, clipping | 10 |
//! | [`sampling`] | Temperature / top-k / top-p sampling with additive mask | 13, 14 |
//! | [`rvq`] | Residual vector quantizer encode/decode | 17, 18 |
//! | [`delay`] | Per-stream delay / undelay of token grids | 20 |
//! | [`rvq_ip`] | RVQ encode with inner-product nearest-code search (Mimi's form) | 17 |
//! | [`conv1d`] | Conv1d (stride, dilation, groups), causal padding, weight-norm folding, streaming state | 15 |
//! | [`conv_transpose1d`] | ConvTranspose1d (stride, groups), causal trim, streaming overlap-add state | 16 |
//! | [`elu`] | ELU fwd (Mimi SEANet activation) | 15 |
//! | [`gelu`] | Exact erf-based GELU fwd (Mimi transformer MLP) | — |
//! | [`layer_norm`] | LayerNorm fwd (Mimi transformers) | — |

pub mod activation;
pub mod attention;
pub mod conv1d;
pub mod conv_transpose1d;
pub mod delay;
pub mod elu;
pub mod embedding;
pub mod gelu;
pub mod gemm;
pub mod layer_norm;
pub mod loss;
pub mod norm;
pub mod optim;
pub mod rope;
pub mod rvq;
pub mod rvq_ip;
pub mod sampling;
