# rust_s2s

A full-duplex speech-to-speech model (STT + LLM + TTS + tool calling fused into one network, in the style of Kyutai's Moshi / Delayed Streams Modeling), trained and served on CUDA from Rust.

- 📐 [Implementation plan](docs/IMPLEMENTATION_PLAN.md): design, kernels, training recipe, streaming runtime, milestones
