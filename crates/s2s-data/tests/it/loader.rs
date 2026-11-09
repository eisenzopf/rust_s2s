use std::sync::Arc;

use s2s_core::streams::StreamLayout;
use s2s_core::SplitMix64;
use s2s_data::sampler::epoch_order;
use s2s_data::{
    assemble, AssemblyConfig, Batch, Cursor, Dataset, Loader, LoaderConfig, Prefetcher, Sample,
    ShardReader, WindowMode, IGNORE_INDEX,
};

use crate::common::{random_sample, write_shard};

const LAYOUT: StreamLayout = StreamLayout {
    text_slots: 2,
    codebooks: 2,
    user_text_delay: 2,
    acoustic_delay: 1,
};

/// Two shards of random samples sharing one text pad id, plus the samples.
fn dataset(dir: &std::path::Path) -> (Arc<Dataset>, Vec<Sample>) {
    let mut rng = SplitMix64::new(99);
    let samples: Vec<Sample> = (0..11)
        .map(|i| {
            let mut s = random_sample(&mut rng, LAYOUT);
            // Unify the pad id; keep tokens in range of the smallest vocab.
            s.text_vocab = 300;
            s.text_pad = 3;
            s.meta.license = if i % 3 == 0 { "LDC" } else { "CC0" }.into();
            s
        })
        .collect();
    let a = write_shard(&dir.join("a.s2s"), &samples[..5]);
    let b = write_shard(&dir.join("b.s2s"), &samples[5..]);
    (Arc::new(Dataset::open(&[a, b]).unwrap()), samples)
}

fn config(mode: WindowMode, seed: u64) -> LoaderConfig {
    let mut c = LoaderConfig::new(AssemblyConfig::new(LAYOUT, 3), 3, 16);
    c.mode = mode;
    c.seed = seed;
    c
}

fn run(loader: &mut Loader, n: usize) -> Vec<Batch> {
    (0..n).map(|_| loader.next_batch().unwrap()).collect()
}

#[test]
fn epoch_order_is_a_seeded_permutation() {
    let a = epoch_order(5, 0, 100, true);
    let mut sorted = a.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..100).collect::<Vec<u32>>());
    assert_eq!(a, epoch_order(5, 0, 100, true));
    assert_ne!(a, epoch_order(5, 1, 100, true));
    assert_ne!(a, epoch_order(6, 0, 100, true));
    assert_eq!(epoch_order(5, 3, 10, false), (0..10).collect::<Vec<u32>>());
}

#[test]
fn loader_is_deterministic_and_seed_dependent() {
    let dir = tempfile::tempdir().unwrap();
    let (ds, _) = dataset(dir.path());
    for mode in [WindowMode::Pack, WindowMode::Crop] {
        let a = run(&mut Loader::new(ds.clone(), config(mode, 1)).unwrap(), 12);
        let b = run(&mut Loader::new(ds.clone(), config(mode, 1)).unwrap(), 12);
        let c = run(&mut Loader::new(ds.clone(), config(mode, 2)).unwrap(), 12);
        assert_eq!(a, b);
        assert_ne!(
            a.iter().map(|x| &x.tokens).collect::<Vec<_>>(),
            c.iter().map(|x| &x.tokens).collect::<Vec<_>>()
        );
        for batch in &a {
            assert_eq!(batch.tokens.len(), 3 * 16 * LAYOUT.num_streams());
            assert_eq!(batch.targets.len(), 3 * 16 * LAYOUT.num_generated());
            assert_eq!(batch.weights.len(), batch.targets.len());
            assert_eq!(batch.is_injection.len(), 3 * 16);
            assert_eq!(batch.segment_ids.len(), 3 * 16);
            assert_eq!(batch.positions.len(), 3 * 16);
        }
    }
}

#[test]
fn exact_resume_from_saved_cursor() {
    let dir = tempfile::tempdir().unwrap();
    let (ds, _) = dataset(dir.path());
    for mode in [WindowMode::Pack, WindowMode::Crop] {
        // Many batches: crosses several epochs (11 samples).
        let all = run(&mut Loader::new(ds.clone(), config(mode, 7)).unwrap(), 40);
        for cut in [1usize, 5, 13, 27] {
            let saved = all[cut - 1].cursor.to_json();
            let cursor = Cursor::from_json(&saved).unwrap();
            let mut resumed = Loader::with_cursor(ds.clone(), config(mode, 7), cursor).unwrap();
            assert_eq!(resumed.cursor(), cursor);
            let rest = run(&mut resumed, 40 - cut);
            assert_eq!(&rest[..], &all[cut..], "mode {mode:?}, resume after {cut}");
        }
        assert!(all.last().unwrap().cursor.epoch >= 2);
    }
}

#[test]
fn cursor_must_match_seed_and_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let (ds, _) = dataset(dir.path());
    let c = Cursor::start(1, ds.len() as u64);
    assert!(Loader::with_cursor(ds.clone(), config(WindowMode::Pack, 2), c).is_err());
    let c = Cursor::start(1, ds.len() as u64 + 1);
    assert!(Loader::with_cursor(ds.clone(), config(WindowMode::Pack, 1), c).is_err());
    let mut c = Cursor::start(1, ds.len() as u64);
    c.index = ds.len() as u64;
    assert!(Loader::with_cursor(ds.clone(), config(WindowMode::Pack, 1), c).is_err());
    assert!(Cursor::from_json("{\"seed\":1}").is_err());
}

#[test]
fn pack_mode_covers_every_position_exactly_once_per_epoch() {
    let dir = tempfile::tempdir().unwrap();
    let (ds, samples) = dataset(dir.path());
    let mut cfg = config(WindowMode::Pack, 0);
    cfg.shuffle = false;
    let cfg_asm = cfg.assembly;
    let examples: Vec<_> = samples
        .iter()
        .map(|s| assemble(s, &cfg_asm).unwrap())
        .collect();
    let total: usize = examples.iter().map(|e| e.positions).sum();
    let ns = LAYOUT.num_streams();
    let k = LAYOUT.num_generated();
    let mut expect_tokens = Vec::new();
    let mut expect_targets = Vec::new();
    for e in &examples {
        expect_tokens.extend_from_slice(&e.tokens);
        expect_targets.extend_from_slice(&e.targets);
    }

    let mut loader = Loader::new(ds, cfg).unwrap();
    let mut got_tokens = Vec::new();
    let mut got_targets = Vec::new();
    while got_tokens.len() < total * ns {
        let w = loader.next_window().unwrap();
        assert_eq!(w.len, 16);
        assert!(
            w.segment_ids.iter().all(|&s| s >= 1),
            "no filler in pack mode"
        );
        // Segment ids are non-decreasing; positions restart at segment starts.
        for p in 1..w.len {
            assert!(w.segment_ids[p] >= w.segment_ids[p - 1]);
            if w.segment_ids[p] == w.segment_ids[p - 1] {
                assert_eq!(w.positions[p], w.positions[p - 1] + 1);
            } else {
                assert_eq!(w.positions[p], 0);
            }
        }
        got_tokens.extend_from_slice(&w.tokens);
        got_targets.extend_from_slice(&w.targets);
    }
    assert_eq!(&got_tokens[..total * ns], &expect_tokens[..]);
    assert_eq!(&got_targets[..total * k], &expect_targets[..]);
}

#[test]
fn crop_mode_windows_are_crops_or_padded() {
    let dir = tempfile::tempdir().unwrap();
    let (ds, samples) = dataset(dir.path());
    let mut cfg = config(WindowMode::Crop, 3);
    cfg.shuffle = false;
    let mut loader = Loader::new(ds, cfg).unwrap();
    let ns = LAYOUT.num_streams();
    for s in &samples {
        let ex = assemble(s, &cfg.assembly).unwrap();
        let w = loader.next_window().unwrap();
        let n = ex.positions.min(16);
        let start = w.positions[0] as usize;
        assert!(start + n <= ex.positions);
        assert_eq!(
            &w.tokens[..n * ns],
            &ex.tokens[start * ns..(start + n) * ns]
        );
        for p in n..16 {
            assert_eq!(w.segment_ids[p], 0);
            assert_eq!(&w.tokens[p * ns..(p + 1) * ns], &LAYOUT.pad_tokens(3)[..]);
            let k = LAYOUT.num_generated();
            assert!(w.targets[p * k..(p + 1) * k]
                .iter()
                .all(|&t| t == IGNORE_INDEX));
        }
    }
}

#[test]
fn prefetcher_matches_plain_loader() {
    let dir = tempfile::tempdir().unwrap();
    let (ds, _) = dataset(dir.path());
    let plain = run(
        &mut Loader::new(ds.clone(), config(WindowMode::Pack, 4)).unwrap(),
        15,
    );
    let mut pf = Prefetcher::spawn(Loader::new(ds, config(WindowMode::Pack, 4)).unwrap(), 4);
    for b in &plain {
        assert_eq!(&pf.next_batch().unwrap(), b);
    }
    drop(pf); // stops the thread
}

#[test]
fn dataset_filter_and_errors() {
    let dir = tempfile::tempdir().unwrap();
    let (ds, samples) = dataset(dir.path());
    assert_eq!(ds.len(), 11);
    assert_eq!(ds.shards().len(), 2);
    assert_eq!(ds.get(7).unwrap(), samples[7]);
    let a = ShardReader::open(dir.path().join("a.s2s")).unwrap();
    let b = ShardReader::open(dir.path().join("b.s2s")).unwrap();
    let commercial = Dataset::new(vec![a, b])
        .filter(|m| m.license != "LDC")
        .unwrap();
    assert_eq!(commercial.len(), 7);
    for i in 0..commercial.len() {
        assert_ne!(commercial.meta(i).unwrap().license, "LDC");
    }
    assert!(commercial.get(7).is_err());

    // Empty dataset and zero batch size are rejected.
    let empty = Arc::new(Dataset::new(vec![]));
    assert!(Loader::new(empty, config(WindowMode::Pack, 0)).is_err());
    let mut cfg = config(WindowMode::Pack, 0);
    cfg.batch_size = 0;
    assert!(Loader::new(ds.clone(), cfg).is_err());

    // A layout mismatch surfaces as an error from the loader, not a panic.
    let mut cfg = config(WindowMode::Pack, 0);
    cfg.assembly.layout.text_slots = 3;
    let mut l = Loader::new(ds, cfg).unwrap();
    assert!(l.next_batch().is_err());
}
