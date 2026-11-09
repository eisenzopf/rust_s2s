# rust_s2s

A full-duplex speech-to-speech model (STT + LLM + TTS + tool calling fused into one network, in the style of Kyutai's Moshi / Delayed Streams Modeling), trained and served on CUDA from Rust using NVIDIA's CUDA Rust libraries: **cuTile Rust** and **cuda-oxide**.

- 📐 [Implementation plan](docs/IMPLEMENTATION_PLAN.md): design, kernels, training recipe, streaming runtime, milestones
- 🧭 [ADR 0001: GPU toolchain](docs/decisions/0001-gpu-toolchain.md): how cuTile and cuda-oxide are used, and what has been verified

**Status:** everything that can be built and verified without a GPU is in place:
- the full model;
- training (single-process and data-parallel logic) and streaming inference behind a WebSocket server;
- tool calling with grammar-constrained decoding;
- data pipeline, tokenizer and evaluation;
- Mimi and Kyutai-LM ports that match Kyutai's own code;
- GPU kernels in both CUDA Rust models.

**Nothing has run on a GPU yet**, and no model has been trained beyond tiny CPU runs. See the plan's milestone status notes.

## Layout

| Path | What | Toolchain |
|---|---|---|
| `crates/s2s-core` | dtypes, shapes, host tensors, deterministic RNG | stable (1.89+) |
| `crates/s2s-kernels` | CPU reference kernels, forward + backward, with finite-difference gradient checks; dense GPU kernels in **cuTile Rust** behind feature `cuda` | stable (1.89+) |
| `crates/s2s-nn` | layers with explicit forward/backward (Linear, RMSNorm, GQA attention + RoPE, SwiGLU, multi-stream embedding, transformer block with activation checkpointing, transformer LM); whole-model gradient checks | stable (1.89+) |
| `crates/s2s-train` | AdamW with parameter groups and clipping, LR schedules, safetensors checkpoints with bit-exact resume, gradient accumulation; data parallelism (DDP, ZeRO-1, sharded checkpoints) over a `Communicator` (threads on CPU, NCCL on GPU); `tiny-lm` CPU demo | stable (1.89+) |
| `crates/s2s-tools` | tool-calling runtime: registry and prompt rendering, tool-call span parser, JSON-schema-constrained decoding masks (llguidance), async executor with client-tool bridge, result injection | stable (1.89+) |
| `crates/s2s-server` | streaming server: WebSocket protocol, Opus/PCM audio, jitter buffer, lockstep 80 ms frame clock over an `Engine` trait, `FakeEngine`, web client; `s2s-server` binary | stable (1.89+) |
| `crates/s2s-data` | training-data pipeline (CPU): memory-mapped shard format, example assembly (delays, injection segments, targets, loss weights), deterministic resumable loader, synthetic tool-dialogue timeline assembly; `s2s-data` CLI (`synth`, `inspect`, `validate`) | stable (1.89+) |
| `crates/s2s-duplex` | **our model**: Temporal Transformer over the 21-stream layout + Depth Transformer, training forward/backward (packed windows with segment masks), streaming inference (`feed_row` / `feed_injection` / `sample_next`); `s2s-duplex-train` CLI | stable (1.89+) |
| `crates/s2s-infer` | streaming engine: the duplex model behind the server's `Engine` trait, with grammar-masked tool calls, tool executor, result injection, S2S/STT/TTS modes, turn events; `KyutaiEngine` serves Kyutai Moshi / STT checkpoints through the same trait; `MimiCodec` / `ToneCodec` | stable (1.89+) |
| `crates/s2s-mimi` | Kyutai's Mimi codec on CPU (offline and streaming; matches Kyutai's implementation, 100% of codes); `mimi-tokenize` CLI (WAV ⇄ codes) | stable (1.89+) |
| `crates/s2s-model` | Kyutai's multistream LM (Moshi / DSM STT) on CPU, loading their checkpoints; matches their logits to within 2.2e-5 | stable (1.89+) |
| `crates/s2s-cuda` | `cudarc` wrappers behind feature `cuda`: cuBLASLt GEMM, NCCL communicator, cudarc↔cuTile zero-copy interop (compile-checked) | stable (1.89+) |
| `crates/s2s-tokenizer` | text tokenizer: model-agnostic Hugging Face `tokenizer.json` wrapper that appends the stream special tokens, fixes the text-stream encoding conventions and implements the tokenizer traits of `s2s-tools` (grammar masks) and `s2s-data` (training data); small trained test tokenizers | stable (1.89+) |
| `crates/s2s-eval` | evaluation (CPU): WER/CER with S/D/I alignment and text normalization, tool-calling accuracy, latency percentiles and turn-taking latency, SI-SDR and log-mel distance (codec proxies); `s2s-eval` CLI (`wer`, `tools`, `latency`, `audio`) | stable (1.89+) |
| `crates/s2s-simt` | sparse/irregular GPU kernels in **cuda-oxide** (bitmask, embedding gather/scatter-add, delays, warp argmax, KV-cache ring write) plus a GPU self-test | Rust 1.99.1 + `cargo oxide` (standalone workspace) |
| `scripts/fetch-cuda-wheels.sh` | assembles a CUDA 13.5 toolkit tree (headers, `tileiras`, `ptxas`, libNVVM) from NVIDIA's PyPI wheels; no full CUDA install needed to compile | — |
| `scripts/check-simt.sh` | cuda-oxide kernels → PTX → cubins for sm_80/89/90/100 | — |

## Build and test

CPU only (no CUDA):

```bash
cargo test --workspace
```

cuTile kernels, compiled to cubins without a GPU:

```bash
scripts/fetch-cuda-wheels.sh .cuda-wheels
export CUDA_TOOLKIT_PATH=$PWD/.cuda-wheels/nvidia/cu13
export PATH=$CUDA_TOOLKIT_PATH/bin:$PATH LD_LIBRARY_PATH=$CUDA_TOOLKIT_PATH/lib
S2S_REQUIRE_TILEIRAS=1 cargo test -p s2s-kernels --features cuda
```

cuda-oxide kernels, compiled to cubins without a GPU (same environment as above):

```bash
rustup toolchain install 1.99.1 -c rust-src -c rustc-dev -c llvm-tools
cargo +1.99.1 install --locked --git https://github.com/NVIDIA/cuda-rust \
  --rev 367bc0f8e671070e00eebc7e5498f3e8eebefc72 cargo-oxide
scripts/check-simt.sh
```

On a machine with an sm_80+ GPU and a CUDA 13 driver, run the GPU parity checks against the CPU references:

```bash
cargo test -p s2s-kernels --features cuda --test cuda_parity -- --ignored   # cuTile
(cd crates/s2s-simt && cargo oxide run)                                     # cuda-oxide
```

### Train and serve (CPU today)

```bash
# Synthetic tool dialogues -> shard -> train our model
cargo run --release -p s2s-data -- synth --out dialogues.s2s crates/s2s-data/tests/data/weather.json
cargo run --release -p s2s-duplex --bin s2s-duplex-train -- --shards dialogues.s2s --steps 200 --checkpoint run.safetensors

# Tokenize real audio with Mimi (needs Kyutai's Mimi checkpoint)
cargo run --release -p s2s-mimi --bin mimi-tokenize -- encode --weights tokenizer.safetensors speech.wav speech.codes.json
```

### GPU tests on AWS

`scripts/aws/run_gpu_tests.py` runs all of the above on a short-lived EC2 GPU instance (default `g6.xlarge`, NVIDIA L4) that terminates itself when done, with a hard 2-hour deadline. It needs AWS credentials: `S2S_AWS_ACCESS_KEY_ID` / `S2S_AWS_SECRET_ACCESS_KEY` if set, otherwise the normal AWS credential chain (an SSO login or `AWS_PROFILE` on your machine works), and `pip install boto3`. Run it from a machine with your AWS login; the cloud dev environment has none. `scripts/aws/iam-policy.json` is a minimal IAM policy for it: it may launch only Ampere-or-newer GPU instances and write only to its own log bucket.

```bash
python3 scripts/aws/run_gpu_tests.py --dry-run   # show the plan, no AWS calls
python3 scripts/aws/run_gpu_tests.py             # launch, wait, print the log, terminate
```

