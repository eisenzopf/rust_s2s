//! Shards → batches of fixed-length windows.
//!
//! [`Loader`] is single-threaded and fully deterministic given the seed and a
//! [`Cursor`]. [`Prefetcher`] runs a loader on a background thread and hands
//! batches over a bounded channel; since one thread produces them in order,
//! the batch stream is identical to the plain loader's.

use std::path::Path;
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::assemble::{assemble, AssemblyConfig, Example};
use crate::sampler::{Cursor, Sampler, Window, WindowMode};
use crate::{Error, Result, Sample, SampleMeta, ShardReader};

/// A list of shards seen as one indexed collection of samples, optionally
/// filtered.
#[derive(Debug)]
pub struct Dataset {
    shards: Vec<ShardReader>,
    /// `(shard, sample within shard)` for every included sample.
    items: Vec<(u32, u32)>,
}

impl Dataset {
    /// All samples of `shards`, in shard order.
    pub fn new(shards: Vec<ShardReader>) -> Self {
        let items = shards
            .iter()
            .enumerate()
            .flat_map(|(si, s)| (0..s.len()).map(move |i| (si as u32, i as u32)))
            .collect();
        Self { shards, items }
    }

    /// Open (memory-map) every shard in `paths`.
    pub fn open<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let shards = paths
            .iter()
            .map(ShardReader::open)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::new(shards))
    }

    /// Keep only samples whose metadata satisfies `keep` (e.g. a license
    /// filter). Reads every sample's metadata once.
    pub fn filter(mut self, mut keep: impl FnMut(&SampleMeta) -> bool) -> Result<Self> {
        let mut items = Vec::with_capacity(self.items.len());
        for &(s, i) in &self.items {
            if keep(&self.shards[s as usize].meta(i as usize)?) {
                items.push((s, i));
            }
        }
        self.items = items;
        Ok(self)
    }

    /// Number of (included) samples.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether there are no samples.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The shards.
    pub fn shards(&self) -> &[ShardReader] {
        &self.shards
    }

    fn locate(&self, i: usize) -> Result<(usize, usize)> {
        self.items
            .get(i)
            .map(|&(s, j)| (s as usize, j as usize))
            .ok_or_else(|| Error::Invalid(format!("sample {i} out of range ({})", self.len())))
    }

    /// Read sample `i`.
    pub fn get(&self, i: usize) -> Result<Sample> {
        let (s, j) = self.locate(i)?;
        self.shards[s].get(j)
    }

    /// Read sample `i`'s metadata.
    pub fn meta(&self, i: usize) -> Result<SampleMeta> {
        let (s, j) = self.locate(i)?;
        self.shards[s].meta(j)
    }
}

/// Loader settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoaderConfig {
    /// Windows per batch (`B`).
    pub batch_size: usize,
    /// Positions per window (`T`).
    pub window: usize,
    /// How samples become windows.
    pub mode: WindowMode,
    /// Shuffle the sample order each epoch.
    pub shuffle: bool,
    /// Sampler seed.
    pub seed: u64,
    /// Example assembly settings (layout, pad id, weights).
    pub assembly: AssemblyConfig,
}

impl LoaderConfig {
    /// Defaults: [`WindowMode::Pack`], shuffled, seed 0.
    pub fn new(assembly: AssemblyConfig, batch_size: usize, window: usize) -> Self {
        Self {
            batch_size,
            window,
            mode: WindowMode::Pack,
            shuffle: true,
            seed: 0,
            assembly,
        }
    }
}

/// A batch of `B` windows of `T` positions, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Batch {
    /// `B`.
    pub batch_size: usize,
    /// `T`.
    pub window: usize,
    /// `S`.
    pub num_streams: usize,
    /// `K`.
    pub num_generated: usize,
    /// `[B, T, S]` input tokens.
    pub tokens: Vec<u32>,
    /// `[B, T, K]` targets (`IGNORE_INDEX` = no loss).
    pub targets: Vec<i64>,
    /// `[B, T, K]` loss weights.
    pub weights: Vec<f32>,
    /// `[B, T]` injection mask.
    pub is_injection: Vec<u8>,
    /// `[B, T]` per-window segment ids (0 = filler).
    pub segment_ids: Vec<u32>,
    /// `[B, T]` positions within each segment's example.
    pub positions: Vec<u32>,
    /// The cursor just after this batch: restore it to continue with the
    /// next batch.
    pub cursor: Cursor,
}

/// Deterministic, resumable batch iterator over a [`Dataset`]. It cycles
/// through epochs forever; [`Iterator::next`] never returns `None`.
#[derive(Debug)]
pub struct Loader {
    dataset: Arc<Dataset>,
    cfg: LoaderConfig,
    sampler: Sampler,
    cursor: Cursor,
    pad_row: Vec<u32>,
    /// The example under the cursor, keyed by `(epoch, index)`.
    cache: Option<(u64, u64, Example)>,
}

impl Loader {
    /// Start at the beginning of epoch 0.
    pub fn new(dataset: Arc<Dataset>, cfg: LoaderConfig) -> Result<Self> {
        let cursor = Cursor::start(cfg.seed, dataset.len() as u64);
        Self::with_cursor(dataset, cfg, cursor)
    }

    /// Resume from a saved cursor. Errors if it was made with a different
    /// seed or dataset size, or points outside the dataset.
    pub fn with_cursor(dataset: Arc<Dataset>, cfg: LoaderConfig, cursor: Cursor) -> Result<Self> {
        if cfg.batch_size == 0 || cfg.window == 0 {
            return Err(Error::Invalid(
                "batch size and window must be positive".into(),
            ));
        }
        let n = dataset.len();
        let sampler = Sampler::new(cfg.seed, n, cfg.shuffle)?;
        if cursor.seed != cfg.seed || cursor.num_samples != n as u64 || cursor.index >= n as u64 {
            return Err(Error::Invalid(format!(
                "cursor {cursor:?} does not fit seed {} and {n} samples",
                cfg.seed
            )));
        }
        let pad_row = cfg.assembly.layout.pad_tokens(cfg.assembly.text_pad);
        Ok(Self {
            dataset,
            cfg,
            sampler,
            cursor,
            pad_row,
            cache: None,
        })
    }

    /// The cursor of the next window.
    pub fn cursor(&self) -> Cursor {
        self.cursor
    }

    /// The configuration.
    pub fn config(&self) -> &LoaderConfig {
        &self.cfg
    }

    /// Make sure the example under the cursor is cached; return its sample id.
    fn load_current(&mut self) -> Result<usize> {
        let (e, i) = (self.cursor.epoch, self.cursor.index);
        let id = self.sampler.sample_at(e, i);
        if !matches!(&self.cache, Some((ce, ci, _)) if *ce == e && *ci == i) {
            let sample = self.dataset.get(id)?;
            let ex = assemble(&sample, &self.cfg.assembly)?;
            self.cache = Some((e, i, ex));
        }
        Ok(id)
    }

    /// Produce the next window and advance the cursor.
    pub fn next_window(&mut self) -> Result<Window> {
        let t = self.cfg.window;
        let layout = self.cfg.assembly.layout;
        let mut w = Window::with_capacity(t, layout.num_streams(), layout.num_generated());
        match self.cfg.mode {
            WindowMode::Crop => {
                let id = self.load_current()?;
                let Some((_, _, ex)) = &self.cache else {
                    return Err(Error::Invalid("loader cache empty".into()));
                };
                let start = self
                    .sampler
                    .crop_offset(self.cursor.epoch, id, ex.positions, t);
                w.push_slice(ex, start, ex.positions.min(t), 1);
                w.fill_to(t, &self.pad_row);
                self.cursor.next_sample();
            }
            WindowMode::Pack => {
                let mut segment = 0u32;
                while w.len < t {
                    self.load_current()?;
                    let Some((_, _, ex)) = &self.cache else {
                        return Err(Error::Invalid("loader cache empty".into()));
                    };
                    let offset = self.cursor.offset as usize;
                    let total = ex.positions;
                    if offset >= total {
                        return Err(Error::Invalid(format!(
                            "cursor offset {offset} is past the sample's {total} positions"
                        )));
                    }
                    let n = (t - w.len).min(total - offset);
                    segment += 1;
                    w.push_slice(ex, offset, n, segment);
                    if offset + n == total {
                        self.cursor.next_sample();
                    } else {
                        self.cursor.offset = (offset + n) as u64;
                    }
                }
            }
        }
        Ok(w)
    }

    /// Produce the next batch and advance the cursor.
    pub fn next_batch(&mut self) -> Result<Batch> {
        let b = self.cfg.batch_size;
        let t = self.cfg.window;
        let s = self.cfg.assembly.layout.num_streams();
        let k = self.cfg.assembly.layout.num_generated();
        let mut batch = Batch {
            batch_size: b,
            window: t,
            num_streams: s,
            num_generated: k,
            tokens: Vec::with_capacity(b * t * s),
            targets: Vec::with_capacity(b * t * k),
            weights: Vec::with_capacity(b * t * k),
            is_injection: Vec::with_capacity(b * t),
            segment_ids: Vec::with_capacity(b * t),
            positions: Vec::with_capacity(b * t),
            cursor: self.cursor,
        };
        for _ in 0..b {
            let w = self.next_window()?;
            batch.tokens.extend_from_slice(&w.tokens);
            batch.targets.extend_from_slice(&w.targets);
            batch.weights.extend_from_slice(&w.weights);
            batch.is_injection.extend_from_slice(&w.is_injection);
            batch.segment_ids.extend_from_slice(&w.segment_ids);
            batch.positions.extend_from_slice(&w.positions);
        }
        batch.cursor = self.cursor;
        Ok(batch)
    }
}

impl Iterator for Loader {
    type Item = Result<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.next_batch())
    }
}

/// Runs a [`Loader`] on a background thread, keeping up to `depth` batches
/// ready. Dropping it stops the thread.
#[derive(Debug)]
pub struct Prefetcher {
    rx: Option<Receiver<Result<Batch>>>,
    handle: Option<JoinHandle<()>>,
}

impl Prefetcher {
    /// Start prefetching from `loader`'s current cursor.
    pub fn spawn(mut loader: Loader, depth: usize) -> Self {
        let (tx, rx) = sync_channel(depth.max(1));
        let handle = std::thread::spawn(move || loop {
            let item = loader.next_batch();
            let failed = item.is_err();
            if tx.send(item).is_err() || failed {
                break;
            }
        });
        Self {
            rx: Some(rx),
            handle: Some(handle),
        }
    }

    /// The next batch, in exactly the order the wrapped loader produces them.
    /// After an error the producer stops and later calls return an error too.
    pub fn next_batch(&mut self) -> Result<Batch> {
        self.rx
            .as_ref()
            .and_then(|rx| rx.recv().ok())
            .unwrap_or_else(|| Err(Error::Invalid("prefetch thread has stopped".into())))
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        // Closing the channel makes the producer's next send fail.
        drop(self.rx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
