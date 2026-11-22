//! Strict pair scorer over [`ScoreEvent`]s: a faithful port of the
//! reference scorer's `score_pair` (pitch-sequence LCS, interval-contour
//! LCS, strict onset/pitch/duration LCS with identical backtrack
//! tie-breaks, staff/spelling accuracy over strict pairs, per-measure
//! exact rate). Tolerances and rounding match the reference exactly.

use assembly_serialize::musicxml_import::ScoreEvent;
use serde::{Deserialize, Serialize};

/// Tolerances in beats, matching the reference scorer.
const ONSET_TOL: f64 = 0.0625;
const DUR_TOL: f64 = 0.0625;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairReport {
    pub gt_notes: usize,
    pub omr_notes: usize,
    pub matched: usize,
    pub note_precision: f64,
    pub note_recall: f64,
    pub note_f1: f64,
    pub contour_f1: f64,
    pub strict_matched: usize,
    pub strict_precision: f64,
    pub strict_recall: f64,
    pub strict_f1: f64,
    pub staff_acc: Option<f64>,
    pub spell_acc: Option<f64>,
    pub measure_exact: Option<f64>,
    /// Reference OMR-NED for the same pair (lower is better), attached by
    /// the run driver (`strict-xml --omr-ned`, scored by pinned musicdiff).
    /// Never computed here: this scorer stays strict-F1-only, so its
    /// semantics are unchanged whether the field is set or not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub omr_ned: Option<f64>,
    /// SHA-256 (hex) of the staged reference / hypothesis input bytes,
    /// logged by the run driver for byte-identical restaging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hypothesis_sha256: Option<String>,
}

fn round3(x: f64) -> f64 {
    // Match the reference scorer's Python round(x, 3): round-half-to-even.
    // Rust's f64::round() is half-away-from-zero and diverges on exact
    // halves (true strict-F1 0.0625 scored 0.063 instead of 0.062).
    let scaled = x * 1000.0;
    let floor = scaled.floor();
    if (scaled - floor - 0.5).abs() < 1e-9 {
        // Exact half up to float error: round to even.
        if (floor as i64) % 2 == 0 {
            floor / 1000.0
        } else {
            (floor + 1.0) / 1000.0
        }
    } else {
        scaled.round() / 1000.0
    }
}

fn strict_eq(a: &ScoreEvent, b: &ScoreEvent) -> bool {
    a.midi == b.midi && (a.onset - b.onset).abs() <= ONSET_TOL && (a.dur - b.dur).abs() <= DUR_TOL
}

/// LCS index pairs under the strict predicate, with the reference
/// backtrack (`>=` prefers up, match requires the diagonal step).
fn lcs_match(a: &[ScoreEvent], b: &[ScoreEvent]) -> Vec<(usize, usize)> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            dp[i + 1][j + 1] = if strict_eq(ai, bj) {
                dp[i][j] + 1
            } else if dp[i][j + 1] >= dp[i + 1][j] {
                dp[i][j + 1]
            } else {
                dp[i + 1][j]
            };
        }
    }
    let mut pairs = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 && j > 0 {
        if strict_eq(&a[i - 1], &b[j - 1]) && dp[i][j] == dp[i - 1][j - 1] + 1 {
            pairs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] >= dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    pairs.reverse();
    pairs
}

/// LCS length over plain MIDI sequences (pitch-only headline).
fn lcs_length(a: &[i32], b: &[i32]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let mut prev = vec![0usize; b.len() + 1];
    for x in a {
        let mut cur = vec![0usize; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = if x == y {
                prev[j] + 1
            } else {
                prev[j + 1].max(cur[j])
            };
        }
        prev = cur;
    }
    prev[b.len()]
}

fn f1(precision: f64, recall: f64) -> f64 {
    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

/// Outcome of [`score_pair`] with the EMPTY rule frozen at the type
/// level, mirroring [`crate::strict::StrictOutcome`]: an empty ground
/// truth NEVER yields a bare strict-F1 float (both-empty scored 0.0 here
/// â€” a deflationary pooler rather than the note-list path's 1.0, but the
/// same wrong column). Empty ground truth routes to the exact-match
/// column and stays out of every F1/SER aggregate; a non-empty ground
/// truth against empty OMR stays an included `Scored` strict-F1 = 0.0.
#[derive(Debug, Clone)]
pub enum PairOutcome {
    Scored(PairReport),
    EmptyReference {
        omr_len: usize,
        hyp_empty: bool,
        exact_match: f64,
    },
}

/// Score a hypothesis event list against ground truth, mirroring the
/// reference `score_pair` field for field.
pub fn score_pair(gt: &[ScoreEvent], omr: &[ScoreEvent]) -> PairOutcome {
    if gt.is_empty() {
        let hyp_empty = omr.is_empty();
        return PairOutcome::EmptyReference {
            omr_len: omr.len(),
            hyp_empty,
            exact_match: if hyp_empty { 1.0 } else { 0.0 },
        };
    }
    // Pitch-sequence LCS: the historic headline and regression guard.
    let gt_pitches: Vec<i32> = gt.iter().map(|e| e.midi).collect();
    let omr_pitches: Vec<i32> = omr.iter().map(|e| e.midi).collect();
    let matched = lcs_length(&gt_pitches, &omr_pitches);
    let note_precision = if omr_pitches.is_empty() {
        0.0
    } else {
        matched as f64 / omr_pitches.len() as f64
    };
    let note_recall = if gt_pitches.is_empty() {
        0.0
    } else {
        matched as f64 / gt_pitches.len() as f64
    };
    // Interval LCS is transposition-invariant: a contour F1 far above
    // note F1 diagnoses a systematic pitch-reference error.
    let gt_iv: Vec<i32> = gt_pitches.windows(2).map(|w| w[1] - w[0]).collect();
    let omr_iv: Vec<i32> = omr_pitches.windows(2).map(|w| w[1] - w[0]).collect();
    let contour_f1 = if gt_iv.is_empty() || omr_iv.is_empty() {
        0.0
    } else {
        let common = lcs_length(&gt_iv, &omr_iv) as f64;
        2.0 * common / (gt_iv.len() + omr_iv.len()) as f64
    };
    // Strict pass over (onset, midi, duration) with tolerance.
    let pairs = lcs_match(gt, omr);
    let strict = pairs.len();
    let strict_precision = if omr.is_empty() {
        0.0
    } else {
        strict as f64 / omr.len() as f64
    };
    let strict_recall = if gt.is_empty() {
        0.0
    } else {
        strict as f64 / gt.len() as f64
    };
    // Staff and spelling accuracy over strict-matched pairs only. With
    // MIDI already equal, equal step forces equal octave and alter.
    let (staff_acc, spell_acc) = if strict == 0 {
        (None, None)
    } else {
        let staff_ok = pairs
            .iter()
            .filter(|(gi, oj)| gt[*gi].staff == omr[*oj].staff)
            .count();
        let spell_ok = pairs
            .iter()
            .filter(|(gi, oj)| gt[*gi].step == omr[*oj].step)
            .count();
        (
            Some(round3(staff_ok as f64 / strict as f64)),
            Some(round3(spell_ok as f64 / strict as f64)),
        )
    };
    PairOutcome::Scored(PairReport {
        gt_notes: gt.len(),
        omr_notes: omr.len(),
        matched,
        note_precision: round3(note_precision),
        note_recall: round3(note_recall),
        note_f1: round3(f1(note_precision, note_recall)),
        contour_f1: round3(contour_f1),
        strict_matched: strict,
        strict_precision: round3(strict_precision),
        strict_recall: round3(strict_recall),
        strict_f1: round3(f1(strict_precision, strict_recall)),
        staff_acc,
        spell_acc,
        measure_exact: measure_exact_rate(gt, omr),
        omr_ned: None,
        reference_sha256: None,
        hypothesis_sha256: None,
    })
}

/// Fraction of measures whose full (onset, midi, duration) content
/// matches exactly, over every measure with notes on either side.
fn measure_exact_rate(gt: &[ScoreEvent], omr: &[ScoreEvent]) -> Option<f64> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut gt_by: BTreeMap<u32, Vec<(f64, i32, f64)>> = BTreeMap::new();
    let mut omr_by: BTreeMap<u32, Vec<(f64, i32, f64)>> = BTreeMap::new();
    for e in gt {
        gt_by
            .entry(e.measure)
            .or_default()
            .push((e.onset, e.midi, e.dur));
    }
    for e in omr {
        omr_by
            .entry(e.measure)
            .or_default()
            .push((e.onset, e.midi, e.dur));
    }
    let ordinals: BTreeSet<u32> = gt_by.keys().chain(omr_by.keys()).copied().collect();
    if ordinals.is_empty() {
        return None;
    }
    let mut ok = 0;
    for m in &ordinals {
        let mut g = gt_by.get(m).cloned().unwrap_or_default();
        let mut o = omr_by.get(m).cloned().unwrap_or_default();
        g.sort_by(|a, b| a.partial_cmp(b).unwrap());
        o.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if g == o {
            ok += 1;
        }
    }
    Some(round3(ok as f64 / ordinals.len() as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(measure: u32, onset: f64, midi: i32, dur: f64) -> ScoreEvent {
        ev_staff(measure, onset, midi, dur, 0)
    }

    fn ev_staff(measure: u32, onset: f64, midi: i32, dur: f64, staff: u32) -> ScoreEvent {
        ScoreEvent {
            measure,
            onset,
            midi,
            dur,
            staff,
            step: 'C',
            alter: 0,
        }
    }

    fn three() -> Vec<ScoreEvent> {
        vec![
            ev(1, 0.0, 60, 1.0),
            ev(1, 1.0, 64, 1.0),
            ev(1, 2.0, 67, 2.0),
        ]
    }

    fn scored(gt: &[ScoreEvent], omr: &[ScoreEvent]) -> PairReport {
        match score_pair(gt, omr) {
            PairOutcome::Scored(r) => r,
            PairOutcome::EmptyReference { .. } => panic!("expected a scored pair"),
        }
    }

    #[test]
    fn identical_lists_score_perfect() {
        let r = scored(&three(), &three());
        assert_eq!(
            r,
            PairReport {
                gt_notes: 3,
                omr_notes: 3,
                matched: 3,
                note_precision: 1.0,
                note_recall: 1.0,
                note_f1: 1.0,
                contour_f1: 1.0,
                strict_matched: 3,
                strict_precision: 1.0,
                strict_recall: 1.0,
                strict_f1: 1.0,
                staff_acc: Some(1.0),
                spell_acc: Some(1.0),
                measure_exact: Some(1.0),
                omr_ned: None,
                reference_sha256: None,
                hypothesis_sha256: None,
            }
        );
    }

    #[test]
    fn scorer_leaves_attached_fields_empty() {
        // The pure scorer never fills run-driver metadata: OMR-NED and
        // input hashes are attached by the CLI layer only, so scorer
        // semantics cannot depend on them.
        let r = scored(&three(), &three());
        assert_eq!(r.omr_ned, None);
        assert_eq!(r.reference_sha256, None);
        assert_eq!(r.hypothesis_sha256, None);
    }

    #[test]
    fn round3_matches_python_half_even_on_exact_halves() {
        // photo31/audiveris/luca-gloria: true strict-F1 is exactly 0.0625.
        assert_eq!(round3(0.0625), 0.062);
        // Odd half rounds up to even, matching Python round(0.1875, 3).
        assert_eq!(round3(0.1875), 0.188);
        // Non-halves and whole values are untouched.
        assert_eq!(round3(2.0 / 3.0), 0.667);
        assert_eq!(round3(1.0), 1.0);
        assert_eq!(round3(0.0), 0.0);
    }

    #[test]
    fn single_pitch_error_costs_both_sides() {
        let mut hyp = three();
        hyp[1].midi = 65;
        let r = scored(&three(), &hyp);
        assert_eq!((r.strict_matched, r.gt_notes, r.omr_notes), (2, 3, 3));
        assert_eq!(r.strict_f1, 0.667);
        assert_eq!(r.note_f1, 0.667);
    }

    #[test]
    fn staff_mismatch_counts_only_over_strict_pairs() {
        let mut hyp = three();
        hyp[0].staff = 1;
        let r = scored(&three(), &hyp);
        assert_eq!(r.strict_matched, 3);
        assert_eq!(r.staff_acc, Some(0.667));
        assert_eq!(r.spell_acc, Some(1.0));
    }

    #[test]
    fn transposition_keeps_contour_but_not_notes() {
        let hyp: Vec<ScoreEvent> = three()
            .into_iter()
            .map(|mut e| {
                e.midi += 2;
                e
            })
            .collect();
        let r = scored(&three(), &hyp);
        assert_eq!(r.note_f1, 0.0);
        assert_eq!(r.strict_f1, 0.0);
        assert_eq!(r.contour_f1, 1.0);
    }

    #[test]
    fn partial_measure_breaks_exact_rate() {
        let mut hyp = three();
        hyp[2].dur = 1.0;
        let r = scored(&three(), &hyp);
        assert_eq!(r.measure_exact, Some(0.0));
    }

    #[test]
    fn empty_ground_truth_routes_to_exact_match() {
        // Both-empty and gt-empty/hyp-nonempty pool no F1 float: the
        // exact-match column carries them instead.
        match score_pair(&[], &[]) {
            PairOutcome::EmptyReference {
                omr_len,
                hyp_empty,
                exact_match,
            } => assert_eq!((omr_len, hyp_empty, exact_match), (0, true, 1.0)),
            PairOutcome::Scored(_) => panic!("empty gt must not score"),
        }
        match score_pair(&[], &three()) {
            PairOutcome::EmptyReference { exact_match, .. } => {
                assert_eq!(exact_match, 0.0);
            }
            PairOutcome::Scored(_) => panic!("empty gt must not score"),
        }
    }

    #[test]
    fn empty_hypothesis_against_gt_stays_included_as_zero() {
        let r = scored(&three(), &[]);
        assert_eq!(
            (r.gt_notes, r.omr_notes, r.matched, r.strict_matched),
            (3, 0, 0, 0)
        );
        assert_eq!(r.note_f1, 0.0);
        assert_eq!(r.strict_f1, 0.0);
        assert_eq!(r.staff_acc, None);
    }
}
