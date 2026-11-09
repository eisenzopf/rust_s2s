#!/usr/bin/env bash
# Regenerate crates/s2s-gpu/src/s2s_simt.ptx from the cuda-oxide kernels in
# crates/s2s-simt. The PTX is checked in so that s2s-gpu builds on stable
# Rust without the cuda-oxide toolchain; rerun this after changing any
# kernel. `--check` fails if the checked-in file is stale (for CI).
#
# Needs: Rust 1.99.1 (rust-src, rustc-dev, llvm-tools) and cargo-oxide.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$HERE/crates/s2s-gpu/src/s2s_simt.ptx"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

(cd "$HERE/crates/s2s-simt" && cargo oxide inspect) > "$TMP/inspect.txt"
# The PTX follows the "PTX (<name>.ptx)" banner and its separator line.
awk 'f>=2{print} /^PTX \(/{f=1;next} f==1&&/^=+$/{f=2}' "$TMP/inspect.txt" > "$TMP/s2s_simt.ptx"
for kernel in rms_norm rope_interleaved kv_append attn_decode_partial attn_decode_combine glu_silu add_inplace f32_to_bf16; do
  grep -q "\.entry ${kernel}(" "$TMP/s2s_simt.ptx" || { echo "missing kernel: $kernel" >&2; exit 1; }
done

if [ "${1:-}" = "--check" ]; then
  diff -q "$TMP/s2s_simt.ptx" "$OUT" >/dev/null || { echo "$OUT is stale; run scripts/build-simt-ptx.sh" >&2; exit 1; }
  echo "$OUT is up to date"
else
  cp "$TMP/s2s_simt.ptx" "$OUT"
  echo "wrote $OUT ($(wc -l < "$OUT") lines)"
fi
