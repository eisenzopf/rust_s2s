#!/usr/bin/env bash
# End-to-end speech check against a running s2s-kyutai-server (Moshi):
#   1. synthesize a spoken question (espeak-ng + sox),
#   2. stream it in real time with scripts/s2s-client.py and record the reply,
#   3. transcribe the reply with Kyutai STT through our own stack
#      (s2s-infer example `transcribe`) to see what the model actually said.
#
# Usage: scripts/e2e-speech.sh [QUESTION]
#
# Environment (defaults match the README's layout):
#   S2S_URL         ws://127.0.0.1:8080/ws
#   S2S_STT_DIR     dir with kyutai/stt-1b-en_fr-candle (model.safetensors)
#   S2S_STT_CONFIG  config-stt-en_fr-hf.toml (kyutai-labs/delayed-streams-modeling)
#   S2S_STT_TOK     the STT tokenizer as tokenizer.json (scripts/spm-to-tokenizer-json.py)
#   S2S_MIMI        tokenizer-e351c8d8-checkpoint125.safetensors
#   S2S_OUT         output directory (a fresh temp dir by default)
#   CARGO_FEATURES  e.g. "cuda" to transcribe on the GPU
# Needs: espeak-ng, sox, python3 with websockets + numpy.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
QUESTION="${1:-Hi there! How are you doing today? Can you tell me a fun fact about the ocean?}"
URL="${S2S_URL:-ws://127.0.0.1:8080/ws}"
: "${S2S_STT_DIR:?set S2S_STT_DIR}" "${S2S_STT_CONFIG:?set S2S_STT_CONFIG}"
: "${S2S_STT_TOK:?set S2S_STT_TOK}" "${S2S_MIMI:?set S2S_MIMI}"
OUT="${S2S_OUT:-$(mktemp -d)}"
mkdir -p "$OUT"

espeak-ng -v en-us -s 140 -w "$OUT/question_raw.wav" "$QUESTION"
sox "$OUT/question_raw.wav" -r 24000 -c 1 -b 16 "$OUT/question.wav" pad 0.5 0

echo "== question: $QUESTION"
status=0
python3 "$HERE/scripts/s2s-client.py" "$OUT/question.wav" "$OUT/reply.wav" --url "$URL" || status=$?

echo "== what the model said (STT of the reply audio)"
features=()
[ -n "${CARGO_FEATURES:-}" ] && features=(--features "$CARGO_FEATURES")
(cd "$HERE" && cargo run -q --release -p s2s-infer "${features[@]}" --example transcribe -- \
  --lm "$S2S_STT_DIR/model.safetensors" --config "$S2S_STT_CONFIG" \
  --mimi "$S2S_MIMI" --tokenizer "$S2S_STT_TOK" "$OUT/reply.wav") | tee "$OUT/reply.stt.txt" | grep '^TRANSCRIPT:'
if ! grep -qE '^TRANSCRIPT: .*[[:alpha:]]{2,}.*[[:alpha:]]{2,}' "$OUT/reply.stt.txt"; then
  echo "FAIL: the reply contains no recognizable speech"
  status=1
fi
echo "== files in $OUT"
exit "$status"
