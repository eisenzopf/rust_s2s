//! Parity of `s2s-mimi` with Kyutai's Rust Mimi (`moshi` 0.6.4, commit
//! e6a55d27), using fixtures written by `tools/kyutai-ref-mimi`:
//!
//! - `reduced*`: a Mimi with the published architecture but small widths
//!   (4 base filters, 32-dim latent, 2 transformer layers with a 16-step
//!   context, 4 codebooks of 64 entries), deterministic weights in Kyutai's
//!   names and weight-norm format, and their offline/streaming outputs on a
//!   16-frame test signal.
//! - `full_v0_1_*`: the parameter names/shapes and the config of Kyutai's
//!   default (full-size) Mimi.
//! - `pointwise_ref`: candle's `erf`, `gelu_erf` and `elu`.
//!
//! The real checkpoint is exercised by the `#[ignore]`d
//! `real_checkpoint_round_trip` (see its docs).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use s2s_kernels::reference::elu::elu;
use s2s_kernels::reference::gelu::{erf, gelu_erf_fwd};
use s2s_mimi::{Codes, Mimi, MimiConfig, SafetensorsCheckpoint, Signal, TensorMap, TensorSource};
use safetensors::tensor::Dtype;
use safetensors::SafeTensors;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/mimi")
}

/// Reference inputs/outputs; float tensors are `[1, C, T]`.
struct RefIo {
    bytes: Vec<u8>,
}

impl RefIo {
    fn open(path: &Path) -> Self {
        let bytes =
            std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        Self { bytes }
    }

    fn raw(&self, name: &str, dtype: Dtype) -> (Vec<usize>, Vec<u8>) {
        let st = SafeTensors::deserialize(&self.bytes).expect("valid safetensors");
        let t = st
            .tensor(name)
            .unwrap_or_else(|_| panic!("missing `{name}`"));
        assert_eq!(t.dtype(), dtype, "{name}");
        (t.shape().to_vec(), t.data().to_vec())
    }

    fn f32s(&self, name: &str) -> Vec<f32> {
        let (_, raw) = self.raw(name, Dtype::F32);
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn f64s(&self, name: &str) -> Vec<f64> {
        let (_, raw) = self.raw(name, Dtype::F64);
        raw.chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }

    fn signal(&self, name: &str) -> Signal {
        let (shape, _) = self.raw(name, Dtype::F32);
        assert!(shape.len() == 3 && shape[0] == 1, "{name}: {shape:?}");
        Signal::new(shape[1], shape[2], self.f32s(name))
    }

    fn codes(&self, name: &str) -> Codes {
        let (shape, raw) = self.raw(name, Dtype::U32);
        let v = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Codes::new(shape[1], shape[2], v).unwrap()
    }
}

/// `max |a - b|` and `max |b|`.
fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let d = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    (d, b.iter().map(|x| x.abs()).fold(0.0, f32::max))
}

fn code_match(a: &Codes, b: &Codes) -> f64 {
    assert_eq!((a.n_q(), a.frames()), (b.n_q(), b.frames()), "code shapes");
    let same = a
        .data()
        .iter()
        .zip(b.data())
        .filter(|(x, y)| x == y)
        .count();
    same as f64 / a.data().len() as f64
}

/// Assert `max |ours - theirs| <= rel * max |theirs|` and print the numbers.
fn check_close(what: &str, ours: &Signal, theirs: &Signal, rel: f32) {
    assert_eq!(
        (ours.channels, ours.len),
        (theirs.channels, theirs.len),
        "{what}: shape"
    );
    let (d, m) = max_abs_diff(&ours.data, &theirs.data);
    println!(
        "{what:>16}: max abs err {d:.3e} (max |ref| {m:.3e}, rel {:.3e})",
        d / m
    );
    assert!(d <= rel * m, "{what}: max abs err {d} > {rel} * {m}");
}

fn reduced() -> (Mimi, RefIo) {
    let cfg = MimiConfig::from_json(
        &std::fs::read_to_string(fixtures().join("reduced_config.json")).unwrap(),
    )
    .unwrap();
    let mimi = Mimi::load(fixtures().join("reduced.safetensors"), cfg).unwrap();
    (
        mimi,
        RefIo::open(&fixtures().join("reduced_io.safetensors")),
    )
}

/// Offline encode/decode against Kyutai's outputs, stage by stage.
fn offline_parity(mimi: &Mimi, io: &RefIo, rel: f32, min_code_match: f64) {
    let pcm = io.signal("pcm_in");
    let t0 = Instant::now();
    let enc = mimi.encode_stages(&pcm.data).unwrap();
    let t_enc = t0.elapsed();
    check_close("enc seanet", &enc.seanet, &io.signal("enc_seanet"), rel);
    check_close(
        "enc transformer",
        &enc.transformer,
        &io.signal("enc_transformer"),
        rel,
    );
    check_close("latent", &enc.latent, &io.signal("latent"), rel);
    let ref_codes = io.codes("codes");
    let m = code_match(&enc.codes, &ref_codes);
    println!(
        "           codes: {:.2}% equal ({} codes)",
        100.0 * m,
        ref_codes.data().len()
    );
    assert!(m >= min_code_match, "code match {m}");
    // Decode Kyutai's codes, so decoder parity does not depend on code flips.
    let t0 = Instant::now();
    let dec = mimi.decode_stages(&ref_codes).unwrap();
    let t_dec = t0.elapsed();
    check_close("quantized", &dec.quantized, &io.signal("quantized"), rel);
    check_close(
        "dec upsample",
        &dec.upsampled,
        &io.signal("dec_upsample"),
        rel,
    );
    check_close(
        "dec transformer",
        &dec.transformer,
        &io.signal("dec_transformer"),
        rel,
    );
    check_close("pcm out", &dec.pcm, &io.signal("pcm_out"), rel);
    let audio_s = pcm.len as f64 / mimi.config().sample_rate;
    println!(
        "encode {:.1} ms, decode {:.1} ms for {:.2} s of audio (RTF {:.3})",
        t_enc.as_secs_f64() * 1e3,
        t_dec.as_secs_f64() * 1e3,
        audio_s,
        (t_enc + t_dec).as_secs_f64() / audio_s
    );
}

#[test]
fn reduced_offline_matches_kyutai() {
    let (mimi, io) = reduced();
    // Activations agree to float rounding; all codes agree.
    offline_parity(&mimi, &io, 1e-5, 1.0);
}

#[test]
fn reduced_streaming_matches_offline_and_kyutai() {
    let (mimi, io) = reduced();
    let pcm = io.signal("pcm_in");
    let fs = mimi.config().frame_size();
    let frames = pcm.len / fs;
    let offline = mimi.encode(&pcm.data).unwrap();

    // Encoder: one 1920-sample push -> one frame of codes.
    let mut enc = mimi.streaming_encoder();
    let stream: Vec<Vec<u32>> = (0..frames)
        .map(|f| enc.push_frame(&pcm.data[f * fs..(f + 1) * fs]).unwrap())
        .collect();
    let stream = Codes::from_frames(&stream).unwrap();
    assert_eq!(stream, offline, "streaming codes != offline codes");
    assert_eq!(
        stream,
        io.codes("stream_codes"),
        "streaming codes != Kyutai's"
    );

    // Irregular chunks give the same codes.
    let mut st = mimi.encoder_state();
    let mut parts = Vec::new();
    let mut pos = 0;
    for (i, n) in [1000usize, 1, 2919, 3840, 777].iter().cycle().enumerate() {
        if pos >= pcm.len {
            break;
        }
        let end = (pos + n + i).min(pcm.len);
        parts.push(mimi.encode_step(&mut st, &pcm.data[pos..end]).unwrap());
        pos = end;
    }
    assert_eq!(Codes::concat(&parts).unwrap(), offline);

    // Decoder: one frame of codes -> 1920 samples, bit-identical to offline.
    let codes = io.codes("codes");
    let off_pcm = mimi.decode(&codes).unwrap();
    let mut dec = mimi.streaming_decoder();
    let mut s_pcm = Vec::new();
    for f in 0..frames {
        let out = dec.push_frame(&codes.frame(f)).unwrap();
        assert_eq!(out.len(), fs);
        s_pcm.extend(out);
    }
    assert_eq!(s_pcm, off_pcm, "streaming decode != offline decode");

    // Kyutai's own streaming decoder agrees with its offline output only
    // while the stream is shorter than the attention context (here 16
    // steps = 8 frames; see the crate docs).
    let theirs = io.f32s("stream_pcm_out");
    let safe = mimi.config().transformer_context / mimi.config().downsample_stride() * fs;
    let (d_safe, m) = max_abs_diff(&s_pcm[..safe], &theirs[..safe]);
    let (d_all, _) = max_abs_diff(&s_pcm, &theirs);
    println!(
        "vs Kyutai streaming decode: max abs err {d_safe:.3e} within context, {d_all:.3e} overall (max |ref| {m:.3e})"
    );
    assert!(d_safe <= 1e-5 * m);
}

#[test]
fn weight_norm_folded_and_unfolded_load_identically() {
    let ck = SafetensorsCheckpoint::open(fixtures().join("reduced.safetensors")).unwrap();
    let mut folded = TensorMap::from_source(&ck).unwrap();
    folded.fold_weight_norm();
    assert!(folded.names().iter().all(|n| !n.ends_with("weight_g")));
    let (mimi, io) = reduced();
    let mimi2 = Mimi::from_source(&folded, mimi.config().clone()).unwrap();
    let pcm = io.f32s("pcm_in");
    let codes = mimi.encode(&pcm).unwrap();
    assert_eq!(mimi2.encode(&pcm).unwrap(), codes);
    assert_eq!(mimi2.decode(&codes).unwrap(), mimi.decode(&codes).unwrap());
}

#[test]
fn fewer_codebooks_and_partial_frames() {
    let (mimi, io) = reduced();
    let pcm = io.f32s("pcm_in");
    let codes = mimi.encode(&pcm).unwrap();
    // Lower-bitrate decode uses the first k codebooks.
    for k in 1..=codes.n_q() {
        let out = mimi.decode(&codes.truncate_codebooks(k)).unwrap();
        assert_eq!(out.len(), pcm.len());
        assert!(out.iter().all(|v| v.is_finite()));
    }
    assert!(mimi.decode(&Codes::new(5, 1, vec![0; 5]).unwrap()).is_err());
    assert!(mimi.decode(&Codes::new(1, 1, vec![64]).unwrap()).is_err());
    // A partial trailing frame is padded (Kyutai's causal extra padding);
    // whole frames are unaffected by it.
    let n = pcm.len() - 700;
    let c = mimi.encode(&pcm[..n]).unwrap();
    assert_eq!(c.frames(), codes.frames());
    assert_eq!(
        c.slice(0, c.frames() - 1),
        codes.slice(0, codes.frames() - 1)
    );
    assert!(mimi.streaming_encoder().push_frame(&pcm[..100]).is_err());
}

#[test]
fn pointwise_ops_match_candle_bit_for_bit() {
    let io = RefIo::open(&fixtures().join("pointwise_ref.safetensors"));
    let x = io.f64s("x");
    for (v, e) in x.iter().zip(io.f64s("erf")) {
        assert_eq!(erf(*v).to_bits(), e.to_bits(), "erf({v})");
    }
    let x32 = io.f32s("x32");
    let mut g = vec![0.0f32; x32.len()];
    gelu_erf_fwd(&x32, &mut g);
    for ((a, b), v) in g.iter().zip(io.f32s("gelu_erf")).zip(&x32) {
        assert_eq!(a.to_bits(), b.to_bits(), "gelu_erf({v})");
    }
    for (v, e) in x32.iter().zip(io.f32s("elu")) {
        assert_eq!(elu(*v, 1.0).to_bits(), e.to_bits(), "elu({v})");
    }
}

#[test]
fn default_config_is_kyutai_v0_1() {
    let s = std::fs::read_to_string(fixtures().join("full_v0_1_config.json")).unwrap();
    assert_eq!(MimiConfig::from_json(&s).unwrap(), MimiConfig::v0_1());
}

/// Zero tensors with the names and shapes of Kyutai's full-size Mimi
/// (`cluster_usage` is 1). Values are allocated zeroed on demand, so pages
/// the loader never writes are never touched and the test stays light.
struct ZeroSource {
    shapes: BTreeMap<String, Vec<usize>>,
}

impl TensorSource for ZeroSource {
    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.shapes.get(name).cloned()
    }

    fn load_f32(&self, name: &str) -> s2s_mimi::Result<Vec<f32>> {
        let shape = self
            .shapes
            .get(name)
            .ok_or_else(|| s2s_mimi::MimiError::MissingTensor(name.to_string()))?;
        let n = shape.iter().product();
        Ok(if name.ends_with("cluster_usage") {
            vec![1.0; n]
        } else {
            vec![0.0; n]
        })
    }

    fn names(&self) -> Vec<String> {
        self.shapes.keys().cloned().collect()
    }
}

/// Kyutai's full-size names/shapes. `fold` replaces each
/// `weight_g`/`weight_v` pair by a folded `weight` of the `weight_v` shape
/// (both forms are accepted; the folded one is lighter to load).
fn full_size_manifest(fold: bool) -> ZeroSource {
    let s = std::fs::read_to_string(fixtures().join("full_v0_1_names.json")).unwrap();
    let names: BTreeMap<String, Vec<usize>> = serde_json::from_str(&s).unwrap();
    let mut shapes = BTreeMap::new();
    for (name, shape) in names {
        if fold && name.ends_with(".weight_g") {
            continue;
        }
        let name = match name.strip_suffix(".weight_v") {
            Some(p) if fold => format!("{p}.weight"),
            _ => name,
        };
        shapes.insert(name, shape);
    }
    ZeroSource { shapes }
}

#[test]
fn full_size_kyutai_names_and_shapes_are_accepted() {
    let m = full_size_manifest(true);
    // The manifest holds 16 codebooks (Kyutai's `v0_1(None)`); load all of them.
    let (mimi, report) =
        Mimi::from_source_with_report(&m, MimiConfig::v0_1().with_n_q(16)).unwrap();
    assert!(
        report.unused.iter().all(|n| n.ends_with("._initialized")),
        "unused tensors: {:?}",
        report.unused
    );
    assert_eq!(report.unused.len(), 16);
    // At the inference setting (8 codebooks) the extra codebooks are unused.
    drop(mimi);
    let (_, report) = Mimi::from_source_with_report(&m, MimiConfig::v0_1()).unwrap();
    let extra: Vec<_> = report
        .unused
        .iter()
        .filter(|n| !n.ends_with("._initialized"))
        .collect();
    assert_eq!(extra.len(), 8 * 2, "{extra:?}"); // 8 x (embedding_sum, cluster_usage)
    assert!(extra
        .iter()
        .all(|n| n.starts_with("quantizer.rvq_rest.vq.layers.")));
}

#[test]
fn full_size_weight_norm_names_are_accepted() {
    // The unfolded (weight_g / weight_v) form, as Kyutai's VarMap writes it.
    let m = full_size_manifest(false);
    let (_, report) = Mimi::from_source_with_report(&m, MimiConfig::v0_1().with_n_q(16)).unwrap();
    assert!(report.unused.iter().all(|n| n.ends_with("._initialized")));
}

/// Real-checkpoint round trip, informational (`--ignored`). Set
/// `S2S_MIMI_WEIGHTS=/path/to/tokenizer-e351c8d8-checkpoint125.safetensors`
/// (see the crate docs for where to download it), optionally
/// `S2S_MIMI_N_Q` (default 8), and optionally `S2S_MIMI_REF_IO` pointing at
/// a `*_io.safetensors` written by `tools/kyutai-ref-mimi` for the same
/// weights to check parity with Kyutai's implementation.
#[test]
#[ignore = "needs the published Mimi checkpoint (set S2S_MIMI_WEIGHTS)"]
fn real_checkpoint_round_trip() {
    let Ok(path) = std::env::var("S2S_MIMI_WEIGHTS") else {
        panic!("set S2S_MIMI_WEIGHTS to a Kyutai-format Mimi .safetensors file");
    };
    let n_q = std::env::var("S2S_MIMI_N_Q").map_or(8, |s| s.parse().expect("S2S_MIMI_N_Q"));
    let cfg = MimiConfig::v0_1().with_n_q(n_q);
    let t0 = Instant::now();
    let ck = SafetensorsCheckpoint::open(&path).unwrap();
    let (mimi, report) = Mimi::from_source_with_report(&ck, cfg).unwrap();
    println!(
        "loaded {path} in {:.2} s; unused tensors: {}",
        t0.elapsed().as_secs_f64(),
        report.unused.len()
    );

    if let Ok(io_path) = std::env::var("S2S_MIMI_REF_IO") {
        let io = RefIo::open(Path::new(&io_path));
        offline_parity(&mimi, &io, 1e-4, 0.995);
    }

    // 4 s test signal: a gliding tone with harmonics.
    let fs = mimi.config().frame_size();
    let n = 50 * fs;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / 24_000.0;
            let f0 = 150.0 + 100.0 * (2.0 * std::f64::consts::PI * 0.5 * t).sin();
            let ph = 2.0 * std::f64::consts::PI * f0 * t;
            (0.3 * ph.sin() + 0.1 * (2.0 * ph).sin() + 0.05 * (3.0 * ph).sin()) as f32
        })
        .collect();
    let t0 = Instant::now();
    let codes = mimi.encode(&pcm).unwrap();
    let t_enc = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let out = mimi.decode(&codes).unwrap();
    let t_dec = t0.elapsed().as_secs_f64();
    assert_eq!((codes.n_q(), codes.frames()), (n_q, 50));
    assert!(codes.data().iter().all(|&c| c < 2048));
    let err: f64 = pcm
        .iter()
        .zip(&out)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum();
    let sig: f64 = pcm.iter().map(|a| (*a as f64).powi(2)).sum();
    println!(
        "4 s: encode {t_enc:.2} s, decode {t_dec:.2} s, RTF {:.3}; reconstruction SNR {:.1} dB",
        (t_enc + t_dec) / 4.0,
        10.0 * (sig / err).log10()
    );

    // Streaming over the first 2 s equals offline.
    let mut enc = mimi.streaming_encoder();
    let mut dec = mimi.streaming_decoder();
    for f in 0..25 {
        let c = enc.push_frame(&pcm[f * fs..(f + 1) * fs]).unwrap();
        assert_eq!(c, codes.frame(f), "frame {f}");
        let y = dec.push_frame(&c).unwrap();
        assert_eq!(&y[..], &out[f * fs..(f + 1) * fs], "frame {f}");
    }
}

#[test]
fn tokenize_round_trip_matches_streaming_codec() {
    use s2s_mimi::{audio, tokenize};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/mimi");
    let cfg = s2s_mimi::MimiConfig::from_json(
        &std::fs::read_to_string(root.join("reduced_config.json")).unwrap(),
    )
    .unwrap();
    let mimi = s2s_mimi::Mimi::load(root.join("reduced.safetensors"), cfg).unwrap();
    let dir = std::env::temp_dir().join(format!("s2s-mimi-tok-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // 2.5 frames of audio: the partial frame is zero-padded to 3 frames.
    let pcm: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.02).sin() * 0.3).collect();
    let wav = dir.join("in.wav");
    audio::write_wav_24k(&wav, &pcm).unwrap();
    let codes_path = dir.join("in.codes.json");
    let file = tokenize::encode_wav(&mimi, &wav, &codes_path).unwrap();
    assert_eq!(file.frames.len(), 3);
    assert_eq!(file.n_q, 4);
    // Same codes as pushing the (quantized-to-16-bit, padded) samples through
    // the streaming encoder directly.
    let mut padded = audio::read_wav_mono_24k(&wav).unwrap();
    padded.resize(3 * 1920, 0.0);
    let mut enc = mimi.streaming_encoder();
    for (f, row) in file.frames.iter().enumerate() {
        assert_eq!(
            row,
            &enc.push_frame(&padded[f * 1920..(f + 1) * 1920]).unwrap()
        );
    }
    let out = dir.join("out.wav");
    assert_eq!(
        tokenize::decode_to_wav(&mimi, &codes_path, &out).unwrap(),
        3
    );
    assert_eq!(audio::read_wav_mono_24k(&out).unwrap().len(), 3 * 1920);
    std::fs::remove_dir_all(&dir).ok();
}
