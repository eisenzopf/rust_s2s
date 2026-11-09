use s2s_core::streams::StreamLayout;
use s2s_core::SplitMix64;
use s2s_data::shard::{decode_record, encode_record, index_path};
use s2s_data::{Error, ShardReader, ShardWriter};

use crate::common::{random_layout, random_sample, write_shard};

fn shard_bytes(n: usize, seed: u64) -> (Vec<s2s_data::Sample>, Vec<u8>, Vec<u8>) {
    let dir = tempfile::tempdir().unwrap();
    let mut rng = SplitMix64::new(seed);
    let samples: Vec<_> = (0..n)
        .map(|_| {
            let l = random_layout(&mut rng);
            random_sample(&mut rng, l)
        })
        .collect();
    let p = write_shard(&dir.path().join("x.s2s"), &samples);
    let data = std::fs::read(&p).unwrap();
    let index = std::fs::read(index_path(&p)).unwrap();
    (samples, data, index)
}

#[test]
fn write_read_round_trip_mmap_and_memory() {
    let dir = tempfile::tempdir().unwrap();
    let mut rng = SplitMix64::new(1);
    let samples: Vec<_> = (0..7)
        .map(|_| random_sample(&mut rng, StreamLayout::default()))
        .collect();
    let path = dir.path().join("a.s2s");
    let mut w = ShardWriter::create(&path).unwrap();
    assert!(w.is_empty());
    for (i, s) in samples.iter().enumerate() {
        assert_eq!(w.append(s).unwrap(), i);
    }
    assert_eq!(w.len(), 7);
    let info = w.finish().unwrap();
    assert_eq!(info.num_samples, 7);
    assert_eq!(
        info.total_frames,
        samples.iter().map(|s| s.frames as u64).sum::<u64>()
    );
    assert_eq!(info.index_path, index_path(&path));
    assert_eq!(std::fs::metadata(&path).unwrap().len(), info.data_len);

    for r in [
        ShardReader::open(&path).unwrap(),
        ShardReader::open_in_memory(&path).unwrap(),
    ] {
        assert_eq!(r.len(), 7);
        assert_eq!(r.path(), Some(path.as_path()));
        r.verify().unwrap();
        for (i, s) in samples.iter().enumerate() {
            assert_eq!(&r.get(i).unwrap(), s);
            assert_eq!(r.meta(i).unwrap(), s.meta);
            assert_eq!(r.entries()[i].frames as usize, s.frames);
            assert_eq!(r.entries()[i].offset % 8, 0);
        }
        assert!(matches!(r.get(7), Err(Error::Invalid(_))));
    }
}

#[test]
fn empty_shard_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_shard(&dir.path().join("empty.s2s"), &[]);
    let r = ShardReader::open(&p).unwrap();
    assert!(r.is_empty());
    r.verify().unwrap();
}

#[test]
fn randomized_record_and_shard_round_trips() {
    let mut rng = SplitMix64::new(0xDA7A);
    for _ in 0..200 {
        let l = random_layout(&mut rng);
        let s = random_sample(&mut rng, l);
        let rec = encode_record(&s).unwrap();
        assert_eq!(decode_record(&rec).unwrap(), s);
    }
    for seed in 0..10 {
        let (samples, data, index) = shard_bytes(1 + seed as usize * 3, seed);
        let r = ShardReader::from_bytes(data, &index).unwrap();
        assert_eq!(r.len(), samples.len());
        for (i, s) in samples.iter().enumerate() {
            assert_eq!(&r.get(i).unwrap(), s);
        }
    }
}

#[test]
fn every_single_byte_flip_is_detected() {
    let (_, data, index) = shard_bytes(3, 42);
    for pos in 0..data.len() {
        let mut d = data.clone();
        d[pos] ^= 0x5A;
        let res = ShardReader::from_bytes(d, &index).and_then(|r| r.verify());
        assert!(
            matches!(res, Err(Error::Format(_))),
            "flip at data byte {pos} went undetected"
        );
    }
    for pos in 0..index.len() {
        let mut ix = index.clone();
        ix[pos] ^= 0x01;
        let res = ShardReader::from_bytes(data.clone(), &ix);
        assert!(
            matches!(res, Err(Error::Format(_))),
            "flip at index byte {pos} went undetected"
        );
    }
}

#[test]
fn truncation_and_extension_are_rejected() {
    let (_, data, index) = shard_bytes(4, 3);
    for cut in (0..data.len()).step_by(7) {
        let res = ShardReader::from_bytes(data[..cut].to_vec(), &index);
        assert!(matches!(res, Err(Error::Format(_))), "cut at {cut}");
    }
    let mut longer = data.clone();
    longer.extend_from_slice(&[0; 8]);
    assert!(ShardReader::from_bytes(longer, &index).is_err());
    for cut in 0..index.len() {
        assert!(ShardReader::from_bytes(data.clone(), &index[..cut]).is_err());
    }
}

#[test]
fn truncated_file_on_disk_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let mut rng = SplitMix64::new(9);
    let samples: Vec<_> = (0..3)
        .map(|_| random_sample(&mut rng, StreamLayout::default()))
        .collect();
    let p = write_shard(&dir.path().join("t.s2s"), &samples);
    let len = std::fs::metadata(&p).unwrap().len();
    let f = std::fs::OpenOptions::new().write(true).open(&p).unwrap();
    f.set_len(len - 5).unwrap();
    drop(f);
    assert!(matches!(ShardReader::open(&p), Err(Error::Format(_))));
    // Shorter than the header, and empty.
    std::fs::write(&p, b"S2SSHARD").unwrap();
    assert!(matches!(ShardReader::open(&p), Err(Error::Format(_))));
    std::fs::write(&p, b"").unwrap();
    assert!(matches!(ShardReader::open(&p), Err(Error::Format(_))));
}

#[test]
fn unfinished_writer_and_foreign_index_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let mut rng = SplitMix64::new(5);
    let s = random_sample(&mut rng, StreamLayout::default());

    let p = dir.path().join("unfinished.s2s");
    let mut w = ShardWriter::create(&p).unwrap();
    w.append(&s).unwrap();
    drop(w);
    // No index at all.
    assert!(matches!(ShardReader::open(&p), Err(Error::Io { .. })));

    // An index from another (finished) shard does not fit.
    let a = write_shard(&dir.path().join("a.s2s"), std::slice::from_ref(&s));
    let b = write_shard(&dir.path().join("b.s2s"), &[s.clone(), s.clone()]);
    let a_data = std::fs::read(&a).unwrap();
    let b_index = std::fs::read(index_path(&b)).unwrap();
    assert!(matches!(
        ShardReader::from_bytes(a_data.clone(), &b_index),
        Err(Error::Format(_))
    ));

    // An unfinalized data file next to a valid index of the same bytes is
    // still rejected (flag unset, header CRC differs).
    let unfinished = std::fs::read(&p).unwrap();
    let a_index = std::fs::read(index_path(&a)).unwrap();
    assert!(matches!(
        ShardReader::from_bytes(unfinished, &a_index),
        Err(Error::Format(_))
    ));
    ShardReader::from_bytes(a_data, &a_index).unwrap();
}

#[test]
fn decode_never_panics_on_garbage() {
    let mut rng = SplitMix64::new(77);
    let s = random_sample(&mut rng, StreamLayout::default());
    let rec = encode_record(&s).unwrap();
    for cut in 0..rec.len() {
        assert!(decode_record(&rec[..cut]).is_err());
    }
    for _ in 0..2000 {
        let mut r = rec.clone();
        let n = 1 + rng.below(4);
        for _ in 0..n {
            let i = rng.below(r.len() as u64) as usize;
            r[i] = rng.next_u64() as u8;
        }
        // Must not panic; may or may not decode (CRC is checked by the reader).
        let _ = decode_record(&r);
    }
    for len in [0usize, 3, 64, 100, 1000] {
        let junk: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
        assert!(decode_record(&junk).is_err());
    }
    // Huge counts in the header must be rejected before allocating.
    let mut big = rec.clone();
    big[40..44].copy_from_slice(&u32::MAX.to_le_bytes()); // num_words
    assert!(matches!(decode_record(&big), Err(Error::Format(_))));
}

#[test]
fn writer_rejects_invalid_samples() {
    let dir = tempfile::tempdir().unwrap();
    let mut rng = SplitMix64::new(11);
    let mut w = ShardWriter::create(dir.path().join("bad.s2s")).unwrap();
    let good = random_sample(&mut rng, StreamLayout::default());

    let mut s = good.clone();
    s.tokens[0] = s.text_vocab; // Tu token out of range
    assert!(matches!(w.append(&s), Err(Error::Invalid(_))));

    let mut s = good.clone();
    s.tokens.pop();
    assert!(matches!(w.append(&s), Err(Error::Invalid(_))));

    let mut s = good.clone();
    s.tokens[s.layout.num_streams() - 1] = 5000; // audio out of range
    assert!(w.append(&s).is_err());

    assert!(w.is_empty());
    w.append(&good).unwrap();
    w.finish().unwrap();
}
