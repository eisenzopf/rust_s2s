#!/usr/bin/env python3
"""Convert a SentencePiece unigram model (Kyutai's text tokenizers) to a
Hugging Face tokenizer.json, then check the two agree token for token.

`tokenizers`' own SentencePieceUnigramTokenizer.from_spm always installs an
NMT + NFKC normalizer that collapses runs of spaces. Kyutai's models use the
`identity` normalizer, so that converter changes the ids of text with leading
or repeated spaces and of characters NFKC rewrites (e.g. "½"). This one keeps
the model's normalizer and splits words the way sentencepiece does.

Known differences from sentencepiece, rare in practice:
- Control, <unk> and byte pieces get a prohibitive score so text that spells
  them out ("<pad>", "<0x00>") does not encode to them. Where the vocabulary
  cannot spell such text without byte fallback (e.g. "<pad>" in the 8k STT
  tokenizer), Hugging Face still picks the piece.
- Exact score ties (long runs of spaces) may pick the same pieces in another
  order.

Usage: scripts/spm-to-tokenizer-json.py IN.model OUT.json [CHECK.txt ...]
Needs: pip install sentencepiece protobuf tokenizers
"""
import sys

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import Unigram

SAMPLES = [
    "Hello, I would like to check the weather in Paris tomorrow.",
    "  leading spaces and Ünïcödé — 42 ½ émoji 🎉",
    'tool_call{"name":"get_weather","args":{"city":"Paris"}}',
    "Line one.\nLine two.\tTabbed.   trailing   ",
    "Bonjour, je voudrais réserver une table pour deux à dix-neuf heures.",
    "ﬁligature ① Ｆｕｌｌｗｉｄｔｈ 日本語のテキスト",
]


def convert(src: str) -> Tokenizer:
    m = pb.ModelProto()
    m.ParseFromString(open(src, "rb").read())
    ts, ns = m.trainer_spec, m.normalizer_spec
    if ts.model_type != pb.TrainerSpec.UNIGRAM:
        sys.exit(f"{src}: only unigram models are supported")
    if ns.name not in ("identity", "nmt_nfkc"):
        sys.exit(f"{src}: unsupported normalizer {ns.name!r}")

    # sentencepiece never matches control, <unk> or byte pieces against text
    # (byte pieces are only a fallback, which Hugging Face looks up by name).
    # Hugging Face's Unigram has no unreachable pieces; a prohibitive score is
    # the closest equivalent.
    T = pb.ModelProto.SentencePiece
    unreachable = (T.CONTROL, T.UNKNOWN, T.BYTE)
    vocab = [(p.piece, -1e9 if p.type in unreachable else p.score) for p in m.pieces]
    tok = Tokenizer(Unigram(vocab, unk_id=ts.unk_id, byte_fallback=ts.byte_fallback))

    norms = []
    if ns.name == "nmt_nfkc":
        norms += [normalizers.Nmt(), normalizers.NFKC()]
    if ns.remove_extra_whitespaces:
        norms += [normalizers.Strip(), normalizers.Replace(Regex(" {2,}"), " ")]
    # The dummy prefix goes in the normalizer: Metaspace's own prepend skips
    # text that already starts with a space, sentencepiece's does not.
    if ns.add_dummy_prefix:
        norms.append(normalizers.Prepend("▁"))
    norms.append(normalizers.Replace(" ", "▁"))
    tok.normalizer = normalizers.Sequence(norms)
    if ts.treat_whitespace_as_suffix:
        sys.exit(f"{src}: treat_whitespace_as_suffix is not supported")
    if ts.split_by_whitespace and ts.allow_whitespace_only_pieces:
        # sentencepiece's SplitIntoWords: a run of spaces stays with the word
        # that follows it, so "a   b" -> "a", "▁▁▁b".
        tok.pre_tokenizer = pre_tokenizers.Split(Regex("▁*[^▁]+|▁+"), behavior="isolated")
    else:
        tok.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme="never", split=ts.split_by_whitespace
        )
    dec = [decoders.Replace("▁", " ")]
    if ts.byte_fallback:
        dec += [decoders.ByteFallback(), decoders.Fuse()]
    if ns.add_dummy_prefix:
        dec += [decoders.Strip(" ", 1, 0)]
    tok.decoder = decoders.Sequence(dec)
    return tok


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, out, *checks = sys.argv[1:]
    tok = convert(src)
    tok.save(out)

    sp = spm.SentencePieceProcessor(model_file=src)
    lines = list(SAMPLES)
    for path in checks:
        lines += open(path, encoding="utf-8").read().splitlines()
    bad = 0
    for line in lines:
        want = sp.encode(line)
        got = tok.encode(line, add_special_tokens=False).ids
        if want != got:
            bad += 1
            print(f"MISMATCH {line[:60]!r}\n  spm {want[:20]}\n  hf  {got[:20]}")
        elif tok.decode(got) != sp.decode(want):
            bad += 1
            print(f"DECODE MISMATCH {line[:60]!r}")
    print(f"{out}: vocab {tok.get_vocab_size()}, {len(lines) - bad}/{len(lines)} lines identical to sentencepiece")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
