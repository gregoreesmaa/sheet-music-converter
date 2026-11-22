//! Strict note-F1 port of the Sheetly benchmark definition:
//! a transcribed note counts iff pitch, onset-in-bar, and duration match.
//! Rests are not scored. Ground truth and hypothesis are each flattened to
//! pitched notes ordered by onset and aligned by LCS under the match
//! predicate (same pitch, same onset-in-bar within a 64th, same duration
//! within a 64th). Extra notes cost precision, missing notes cost recall.
//!
//! Differences from the reference scorer, recorded honestly: onset-in-bar
//! is computed against a uniform bar grid from the score time signature
//! (no anacrusis/pickup handling); enharmonic spelling is pitch-class
//! exact via MIDI (spelling unscored, as in the reference).

use assembly_serialize::Score;
use serde::{Deserialize, Serialize};

/// Tolerance in beats: one 64th note.
const TOL_BEATS: f64 = 1.0 / 16.0;
/// Float slop for exact tick-grid comparisons.
const EPS: f64 = 1e-9;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F1Report {
    pub matches: usize,
    pub ref_len: usize,
    pub hyp_len: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    /// Reference OMR-NED for the same pair (lower is better), attached by
    /// the run driver (`strict-f1 --omr-ned`, scored by pinned musicdiff).
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

#[derive(Debug, Clone, PartialEq)]
struct Note {
    midi: i32,
    onset_bar: f64,
    dur: f64,
}

/// Flatten pitched notes pooled across parts, ordered by onset.
/// Onset-in-bar and duration are in beats (tick-grid exact).
fn flatten(score: &Score) -> Vec<Note> {
    let divisions = score.divisions as f64;
    let bar = score.measure_ticks() as f64 / divisions;
    let mut notes: Vec<(f64, Note)> = Vec::new();
    for part in &score.parts {
        for group in &part.groups {
            for m in &group.members {
                if m.rest {
                    continue;
                }
                let onset = m.start_ticks as f64 / divisions;
                notes.push((
                    onset,
                    Note {
                        midi: m.midi,
                        onset_bar: onset % bar,
                        dur: m.dur_ticks as f64 / divisions,
                    },
                ));
            }
        }
    }
    notes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.midi.cmp(&b.1.midi)));
    notes.into_iter().map(|(_, n)| n).collect()
}

fn is_match(a: &Note, b: &Note) -> bool {
    a.midi == b.midi
        && (a.onset_bar - b.onset_bar).abs() <= TOL_BEATS + EPS
        && (a.dur - b.dur).abs() <= TOL_BEATS + EPS
}

/// LCS match count between reference and hypothesis note sequences.
fn lcs_matches(reference: &[Note], hypothesis: &[Note]) -> usize {
    let (short, long, swapped) = if hypothesis.len() < reference.len() {
        (hypothesis, reference, true)
    } else {
        (reference, hypothesis, false)
    };
    let mut prev = vec![0usize; short.len() + 1];
    let mut curr = vec![0usize; short.len() + 1];
    for l in long {
        for (j, s) in short.iter().enumerate() {
            let hit = if swapped {
                is_match(l, s)
            } else {
                is_match(s, l)
            };
            curr[j + 1] = if hit {
                prev[j] + 1
            } else {
                curr[j].max(prev[j + 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[short.len()]
}

/// Outcome of strict-F1 scoring with the EMPTY rule frozen at the type
/// level: an empty reference NEVER yields a bare F1 float (the old
/// `f1 = 1.0` for both-empty silently pooled perfect scores into suite
/// medians). Empty references route to the exact-match column instead â€”
/// 1 when the hypothesis is also empty, 0 otherwise â€” and stay out of
/// every F1/SER aggregate. A non-empty reference against an empty
/// hypothesis stays an included `Scored` F1 = 0.0.
#[derive(Debug, Clone)]
pub enum StrictOutcome {
    Scored(F1Report),
    EmptyReference {
        hyp_len: usize,
        hyp_empty: bool,
        exact_match: f64,
    },
}

/// Strict-F1 of a hypothesis score against a reference score.
pub fn strict_f1(reference: &Score, hypothesis: &Score) -> StrictOutcome {
    let ref_notes = flatten(reference);
    if ref_notes.is_empty() {
        let hyp_notes = flatten(hypothesis);
        let hyp_empty = hyp_notes.is_empty();
        return StrictOutcome::EmptyReference {
            hyp_len: hyp_notes.len(),
            hyp_empty,
            exact_match: if hyp_empty { 1.0 } else { 0.0 },
        };
    }
    let hyp_notes = flatten(hypothesis);
    let matches = lcs_matches(&ref_notes, &hyp_notes);
    let (ref_len, hyp_len) = (ref_notes.len(), hyp_notes.len());
    let (precision, recall, f1) = if matches == 0 {
        (0.0, 0.0, 0.0)
    } else {
        let p = matches as f64 / hyp_len as f64;
        let r = matches as f64 / ref_len as f64;
        (p, r, 2.0 * p * r / (p + r))
    };
    StrictOutcome::Scored(F1Report {
        matches,
        ref_len,
        hyp_len,
        precision,
        recall,
        f1,
        omr_ned: None,
        reference_sha256: None,
        hypothesis_sha256: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(json: serde_json::Value) -> Score {
        let input: assembly_serialize::ScoreInput = serde_json::from_value(json).unwrap();
        assembly_serialize::parse_score(&input).unwrap().0
    }

    fn empty_score() -> Score {
        Score {
            title: "empty".to_string(),
            composer: None,
            divisions: 480,
            beats: 4,
            beat_type: 4,
            key_fifths: 0,
            parts: Vec::new(),
            measure_attrs: Vec::new(),
        }
    }

    /// Unwrap the scored branch; empty-reference pairs must use the
    /// `StrictOutcome::EmptyReference` assertions below, never this.
    fn scored(reference: &Score, hypothesis: &Score) -> F1Report {
        match strict_f1(reference, hypothesis) {
            StrictOutcome::Scored(r) => r,
            StrictOutcome::EmptyReference { .. } => panic!("expected a scored pair"),
        }
    }

    fn three_notes() -> serde_json::Value {
        serde_json::json!({
            "divisions": 480,
            "time": {"beats": 4, "beat_type": 4},
            "parts": [{"name": "P", "notes": [
                {"pitch": "C4", "onset": 0.0, "dur": 1.0},
                {"pitch": "E4", "onset": 1.0, "dur": 1.0},
                {"pitch": "G4", "onset": 2.0, "dur": 2.0}
            ]}]
        })
    }

    #[test]
    fn identical_lists_score_one() {
        let r = scored(&score(three_notes()), &score(three_notes()));
        assert_eq!((r.matches, r.ref_len, r.hyp_len), (3, 3, 3));
        assert_eq!((r.precision, r.recall, r.f1), (1.0, 1.0, 1.0));
    }

    #[test]
    fn single_pitch_error_costs_both() {
        let mut hyp = three_notes();
        hyp["parts"][0]["notes"][1]["pitch"] = serde_json::json!("F4");
        let r = scored(&score(three_notes()), &score(hyp));
        assert_eq!((r.matches, r.ref_len, r.hyp_len), (2, 3, 3));
        assert!((r.precision - 2.0 / 3.0).abs() < 1e-12);
        assert!((r.recall - 2.0 / 3.0).abs() < 1e-12);
        assert!((r.f1 - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn extra_hyp_note_costs_precision_only() {
        let mut hyp = three_notes();
        // Non-overlapping extra note: a fully overlapped same-voice note
        // would be dropped by parse_score (`note-dropped-overlap`), so the
        // extra sits after the last note ends.
        hyp["parts"][0]["notes"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"pitch": "A4", "onset": 4.0, "dur": 0.5}));
        let r = scored(&score(three_notes()), &score(hyp));
        assert_eq!((r.matches, r.ref_len, r.hyp_len), (3, 3, 4));
        assert!((r.precision - 0.75).abs() < 1e-12);
        assert_eq!(r.recall, 1.0);
        assert!((r.f1 - 2.0 * 0.75 / 1.75).abs() < 1e-12);
    }

    #[test]
    fn missing_note_costs_recall_only() {
        let mut hyp = three_notes();
        hyp["parts"][0]["notes"].as_array_mut().unwrap().pop();
        let r = scored(&score(three_notes()), &score(hyp));
        assert_eq!((r.matches, r.ref_len, r.hyp_len), (2, 3, 2));
        assert_eq!(r.precision, 1.0);
        assert!((r.recall - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn onset_outside_tolerance_mismatches() {
        // Local fixture with slack after E4 so the shift cannot cascade
        // into G4 through overlap clamping.
        let mk = |e_onset: f64| {
            score(serde_json::json!({
                "divisions": 480,
                "time": {"beats": 4, "beat_type": 4},
                "parts": [{"name": "P", "notes": [
                    {"pitch": "C4", "onset": 0.0, "dur": 1.0},
                    {"pitch": "E4", "onset": e_onset, "dur": 1.0},
                    {"pitch": "G4", "onset": 3.0, "dur": 1.0}
                ]}]
            }))
        };
        // Shift E4 by a 32nd note (1/8 beat, over the 64th tolerance).
        let r = scored(&mk(1.0), &mk(1.125));
        assert_eq!((r.matches, r.ref_len, r.hyp_len), (2, 3, 3));
    }

    #[test]
    fn onset_inside_tolerance_matches() {
        let mut hyp = three_notes();
        // Shift E4 by a 128th (within the 64th tolerance).
        hyp["parts"][0]["notes"][1]["onset"] = serde_json::json!(1.0078125);
        let r = scored(&score(three_notes()), &score(hyp));
        assert_eq!((r.matches, r.f1), (3, 1.0));
    }

    #[test]
    fn duration_mismatch_does_not_match() {
        let mut hyp = three_notes();
        hyp["parts"][0]["notes"][0]["dur"] = serde_json::json!(0.5);
        let r = scored(&score(three_notes()), &score(hyp));
        assert_eq!(r.matches, 2);
    }

    #[test]
    fn rests_are_not_scored() {
        let mut with_rest = three_notes();
        with_rest["parts"][0]["notes"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"rest": true, "onset": 3.0, "dur": 1.0}));
        let r = scored(&score(with_rest.clone()), &score(with_rest));
        assert_eq!((r.matches, r.ref_len, r.hyp_len), (3, 3, 3));
        assert_eq!(r.f1, 1.0);
    }

    #[test]
    fn bar_index_is_not_part_of_the_match() {
        let mut hyp = three_notes();
        // Move every note one full bar later: same in-bar positions.
        for n in hyp["parts"][0]["notes"].as_array_mut().unwrap() {
            let onset = n["onset"].as_f64().unwrap();
            n["onset"] = serde_json::json!(onset + 4.0);
        }
        let r = scored(&score(three_notes()), &score(hyp));
        assert_eq!((r.matches, r.f1), (3, 1.0));
    }

    #[test]
    fn both_empty_routes_to_exact_match_not_bare_f1() {
        // The silent-pooling hazard, frozen out at the type level: no
        // F1Report (and no f1 = 1.0 float) exists on this path.
        match strict_f1(&empty_score(), &empty_score()) {
            StrictOutcome::EmptyReference {
                hyp_len,
                hyp_empty,
                exact_match,
            } => {
                assert_eq!((hyp_len, hyp_empty, exact_match), (0, true, 1.0));
            }
            StrictOutcome::Scored(_) => panic!("both-empty must not score"),
        }
    }

    #[test]
    fn empty_reference_with_notes_scores_exact_zero() {
        match strict_f1(&empty_score(), &score(three_notes())) {
            StrictOutcome::EmptyReference { exact_match, .. } => {
                assert_eq!(exact_match, 0.0);
            }
            StrictOutcome::Scored(_) => panic!("empty ref must not score"),
        }
    }

    #[test]
    fn nonempty_reference_with_empty_hypothesis_stays_included_as_zero() {
        let r = scored(&score(three_notes()), &empty_score());
        assert_eq!((r.matches, r.ref_len, r.hyp_len), (0, 3, 0));
        assert_eq!((r.precision, r.recall, r.f1), (0.0, 0.0, 0.0));
    }

    #[test]
    fn rest_only_reference_counts_as_empty() {
        // Rests are unscored, so a rest-only reference holds zero pitched
        // notes and routes to the exact-match column like any empty ref.
        let mut ref_only_rest = three_notes();
        for n in ref_only_rest["parts"][0]["notes"].as_array_mut().unwrap() {
            *n = serde_json::json!({"rest": true, "onset": 0.0, "dur": 4.0});
        }
        match strict_f1(&score(ref_only_rest), &empty_score()) {
            StrictOutcome::EmptyReference { exact_match, .. } => {
                assert_eq!(exact_match, 1.0);
            }
            StrictOutcome::Scored(_) => panic!("rest-only ref must not score"),
        }
    }
}
