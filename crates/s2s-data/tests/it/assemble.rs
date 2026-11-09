use s2s_core::streams::{StreamKind, StreamLayout, AUDIO_PAD as AP};
use s2s_core::SplitMix64;
use s2s_data::{
    assemble, AssemblyConfig, Injection, InjectionKind, LossWeights, Sample, SampleMeta,
    IGNORE_INDEX as IG,
};

use crate::common::{random_layout, random_sample};

/// 1 text slot, 2 codebooks, d_u = 1, tau = 1. Streams:
/// 0 Tu (d1), 1 Tm0 (d0), 2 Am0 (d0), 3 Am1 (d1), 4 Au0 (d0), 5 Au1 (d1).
fn tiny() -> Sample {
    let layout = StreamLayout {
        text_slots: 1,
        codebooks: 2,
        user_text_delay: 1,
        acoustic_delay: 1,
    };
    #[rustfmt::skip]
    let tokens = vec![
        5, 0, 100, 200, 300, 400,
        6, 7, 101, 201, 301, 401,
        0, 8, 102, 202, 302, 402,
    ];
    Sample {
        layout,
        frames: 3,
        text_vocab: 20,
        text_pad: 0,
        tokens,
        injections: vec![
            Injection {
                frame: 0,
                kind: InjectionKind::System,
                tokens: vec![11, 12],
            },
            Injection {
                frame: 2,
                kind: InjectionKind::ToolResult,
                tokens: vec![13],
            },
        ],
        words: vec![],
        meta: SampleMeta::default(),
    }
}

fn weights() -> LossWeights {
    LossWeights {
        user_text: 2.0,
        assistant_text: 1.0,
        semantic: 3.0,
        acoustic: 0.5,
        text_pad_factor: 0.25,
    }
}

#[test]
fn hand_built_example_matches_expected_arrays() {
    let s = tiny();
    let mut cfg = AssemblyConfig::for_sample(&s);
    cfg.weights = weights();
    let ex = assemble(&s, &cfg).unwrap();
    assert_eq!((ex.num_streams, ex.num_generated, ex.positions), (6, 4, 8));

    // Positions: start, inj 11, inj 12, row 0, row 1, inj 13, row 2, row 3.
    #[rustfmt::skip]
    let tokens = vec![
        0, 0,  AP,  AP,  AP,  AP,   // start ("frame -1")
        0, 11, AP,  AP,  AP,  AP,   // <sys> token 11
        0, 12, AP,  AP,  AP,  AP,   // token 12
        0, 0,  100, AP,  300, AP,   // row 0 (Tu, Am1, Au1 delay-padded)
        5, 7,  101, 200, 301, 400,  // row 1
        0, 13, AP,  AP,  AP,  AP,   // tool result token 13
        6, 8,  102, 201, 302, 401,  // row 2
        0, 0,  AP,  202, AP,  402,  // row 3 (Tu = frame 2's real <pad>)
    ];
    assert_eq!(ex.tokens, tokens);
    assert_eq!(ex.is_injection, vec![0, 1, 1, 0, 0, 1, 0, 0]);
    assert_eq!(ex.frame_index, vec![-1, -1, -1, 0, 1, -1, 2, 3]);

    // Targets over [Tu, Tm0, Am0, Am1], shifted by one position.
    #[rustfmt::skip]
    let targets: Vec<i64> = vec![
        IG, IG, IG,  IG,   // p0 -> injection
        IG, IG, IG,  IG,   // p1 -> injection
        IG, 0,  100, IG,   // p2 -> row 0: Tu, Am1 are delay padding
        5,  7,  101, 200,  // p3 -> row 1
        IG, IG, IG,  IG,   // p4 -> injection
        6,  8,  102, 201,  // p5 (last injected token) -> row 2, trained
        0,  IG, IG,  202,  // p6 -> row 3: Tm0, Am0 past the end
        IG, IG, IG,  IG,   // p7: no successor
    ];
    assert_eq!(ex.targets, targets);
    #[rustfmt::skip]
    let w: Vec<f32> = vec![
        0.0, 0.0,  0.0, 0.0,
        0.0, 0.0,  0.0, 0.0,
        0.0, 0.25, 3.0, 0.0,   // Tm0 target is <pad>: 1.0 * 0.25
        2.0, 1.0,  3.0, 0.5,
        0.0, 0.0,  0.0, 0.0,
        2.0, 1.0,  3.0, 0.5,
        0.5, 0.0,  0.0, 0.5,   // Tu target is <pad>: 2.0 * 0.25
        0.0, 0.0,  0.0, 0.0,
    ];
    assert_eq!(ex.weights, w);

    // The frame positions are exactly the delayed grid.
    assert_eq!(ex.aligned_frames(&s.layout, s.frames), s.tokens);
}

#[test]
fn without_start_frame_and_with_other_injection_stream() {
    let mut s = tiny();
    s.injections.remove(0); // no injection before frame 0
    let mut cfg = AssemblyConfig::for_sample(&s);
    cfg.start_frame = false;
    cfg.injection_stream = StreamKind::UserText;
    let ex = assemble(&s, &cfg).unwrap();
    // row 0, row 1, inj 13, row 2, row 3
    assert_eq!(ex.positions, 5);
    assert_eq!(ex.frame_index, vec![0, 1, -1, 2, 3]);
    assert_eq!(&ex.tokens[2 * 6..3 * 6], &[13, 0, AP, AP, AP, AP]);
    // Nothing predicts row 0 without a start frame.
    assert_eq!(&ex.targets[0..4], &[5, 7, 101, 200]);

    cfg.injection_stream = StreamKind::AssistantAudio(0);
    assert!(assemble(&s, &cfg).is_err());
}

#[test]
fn injection_at_end_and_without_delays() {
    let mut s = tiny();
    s.layout.user_text_delay = 0;
    s.layout.acoustic_delay = 0;
    s.injections = vec![Injection {
        frame: 3,
        kind: InjectionKind::Text,
        tokens: vec![9, 10],
    }];
    let ex = assemble(&s, &AssemblyConfig::for_sample(&s)).unwrap();
    // start, rows 0..3, then the trailing segment.
    assert_eq!(ex.frame_index, vec![-1, 0, 1, 2, -1, -1]);
    assert_eq!(ex.is_injection, vec![0, 0, 0, 0, 1, 1]);
    assert!(ex.targets[3 * 4..].iter().all(|&t| t == IG));
    assert!(ex.weights[3 * 4..].iter().all(|&w| w == 0.0));
}

#[test]
fn config_mismatches_are_errors() {
    let s = tiny();
    let mut cfg = AssemblyConfig::for_sample(&s);
    cfg.text_pad = 1;
    assert!(assemble(&s, &cfg).is_err());
    let mut cfg = AssemblyConfig::for_sample(&s);
    cfg.layout.codebooks = 3;
    assert!(assemble(&s, &cfg).is_err());
    // Delays may differ from the sample's recorded ones.
    let mut cfg = AssemblyConfig::for_sample(&s);
    cfg.layout.user_text_delay = 3;
    let ex = assemble(&s, &cfg).unwrap();
    assert_eq!(ex.aligned_frames(&cfg.layout, s.frames), s.tokens);
    let mut bad = s.clone();
    bad.injections.swap(0, 1);
    assert!(assemble(&bad, &AssemblyConfig::for_sample(&bad)).is_err());
}

#[test]
fn randomized_assembly_invariants() {
    let mut rng = SplitMix64::new(2024);
    for _ in 0..300 {
        let layout = random_layout(&mut rng);
        let s = random_sample(&mut rng, layout);
        let mut cfg = AssemblyConfig::for_sample(&s);
        cfg.start_frame = rng.below(2) == 0;
        let ex = assemble(&s, &cfg).unwrap();
        let (ns, k) = (ex.num_streams, ex.num_generated);
        let rows = s.frames + layout.max_delay();
        assert_eq!(
            ex.positions,
            rows + s.injection_tokens() + usize::from(cfg.start_frame)
        );
        assert_eq!(ex.tokens.len(), ex.positions * ns);
        assert_eq!(ex.targets.len(), ex.positions * k);
        assert_eq!(ex.aligned_frames(&layout, s.frames), s.tokens);
        // Injected tokens appear in order on Tm slot 0.
        let injected: Vec<u32> = (0..ex.positions)
            .filter(|&p| ex.is_injection[p] == 1)
            .map(|p| ex.tokens[p * ns + 1])
            .collect();
        let expected: Vec<u32> = s.injections.iter().flat_map(|i| i.tokens.clone()).collect();
        assert_eq!(injected, expected);
        for p in 0..ex.positions {
            let next_is_frame = p + 1 < ex.positions && ex.frame_index[p + 1] >= 0;
            for j in 0..k {
                let (t, w) = (ex.targets[p * k + j], ex.weights[p * k + j]);
                if t == IG {
                    assert_eq!(w, 0.0);
                } else {
                    assert!(next_is_frame, "target at {p} must predict a frame");
                    assert_eq!(t, ex.tokens[(p + 1) * ns + j] as i64);
                }
                if !next_is_frame {
                    assert_eq!(t, IG);
                }
            }
        }
    }
}
