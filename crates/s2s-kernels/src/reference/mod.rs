//! CPU reference implementations (forward and backward).
//!
//! | Module | Ops | Plan §7.2 kernel # |
//! |---|---|---|
//! | [`gemm`] | GEMM with transposes, `Linear` fwd/bwd | 1 |
//! | [`norm`] | RMSNorm fwd/bwd (optional fused residual) | 2 |
//! | [`layernorm`] | LayerNorm fwd (inference) | 2 |
//! | [`rope`] | Rotary position embedding fwd/bwd, explicit positions; precomputed f32-angle tables | 3 |
//! | [`activation`] | SwiGLU fwd/bwd | 4 |
//! | [`elementwise`] | SiLU / GELU (erf, tanh) / ReLU / sigmoid families, `erf`, gated linear unit fwd (inference) | 4 |
//! | [`attention`] | Causal / sliding-window / pinned-prefix GQA attention fwd/bwd, single-query decode | 5, 6 |
//! | [`embedding`] | Multi-stream embedding gather-and-sum fwd / scatter-add bwd | 7 |
//! | [`loss`] | Softmax, log-softmax, weighted cross-entropy with ignore index | 8, 9 |
//! | [`optim`] | AdamW, global grad-norm, clipping | 10 |
//! | [`sampling`] | Temperature / top-k / top-p sampling with additive mask | 13, 14 |
//! | [`rvq`] | Residual vector quantizer encode/decode | 17, 18 |
//! | [`delay`] | Per-stream delay / undelay of token grids | 20 |

pub mod activation;
pub mod attention;
pub mod delay;
pub mod elementwise;
pub mod embedding;
pub mod gemm;
pub mod layernorm;
pub mod loss;
pub mod norm;
pub mod optim;
pub mod rope;
pub mod rvq;
pub mod sampling;
