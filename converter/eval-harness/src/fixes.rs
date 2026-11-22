//! Measure-atomic gesture scoring for the top-2 lattice (Phase 4, Task T4.2).
//!
//! A correction gesture is 1 click when the ground-truth measure equals
//! lattice candidate #2 (accept the alternative), else the note-level
//! token edits inside that measure. Lattice types are shared with
//! `assembly-serialize` (no schema duplication). Pure functions; no I/O.

use assembly_serialize::lattice::LatticeDoc;
use serde::Serialize;

/// Token-level Levenshtein distance (note-level edits within a measure).
pub fn token_edit_distance(a: &[String], b: &[String]) -> usize {
    let (n, m) = (a.len(), b.len());
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            cur[j] = (prev[j] + 1)
                .min(cur[j - 1] + 1)
                .min(prev[j - 1] + usize::from(a[i - 1] != b[j - 1]));
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// Leading header run (`clef-*`, `key*`, `time-*`): carried staff
/// context, not notes to fix. Stripped from both sides before
/// comparing so a note-identical candidate with and without the
/// copied header still matches. Mid-measure attribute changes are
/// content and are kept.
pub fn strip_leading_header(tokens: &[String]) -> &[String] {
    let mut i = 0;
    while i < tokens.len()
        && (tokens[i].starts_with("clef-")
            || tokens[i].starts_with("key")
            || tokens[i].starts_with("time-"))
    {
        i += 1;
    }
    &tokens[i..]
}

/// Gestures for one measure: 0 when the top-1 candidate transcribes it
/// exactly, 1 when candidate #2 matches (one accept click), else the
/// token edit distance between the top-1 candidate and ground truth.
/// Leading header runs are stripped from both sides first.
pub fn measure_gestures(
    gt_tokens: &[String],
    candidates: &[String],
) -> usize {
    let top1: Vec<String> =
        candidates.first().map(|c| split_tokens(c)).unwrap_or_default();
    let gt = strip_leading_header(gt_tokens);
    let top1 = strip_leading_header(&top1);
    if top1 == gt {
        return 0;
    }
    let cand2: Option<Vec<String>> =
        candidates.get(1).map(|c| split_tokens(c));
    if cand2.as_deref().map(strip_leading_header) == Some(gt) {
        return 1;
    }
    token_edit_distance(top1, gt)
}

/// Split a bekern+ measure token string on ASCII whitespace.
pub fn split_tokens(s: &str) -> Vec<String> {
    s.split_whitespace().map(|t| t.to_string()).collect()
}

/// Per-measure + total gesture estimate for one piece.
///
/// `gt_measures` and `lattice.measures` must cover the same measures
/// in the same order; a length mismatch is a hard error (silent
/// misalignment would attribute edits to the wrong measure).
#[derive(Debug, Clone, Serialize)]
pub struct FixesEstimate {
    pub total: usize,
    pub per_measure: Vec<usize>,
}

pub fn fixes_estimate(
    gt_measures: &[String],
    lattice: &LatticeDoc,
) -> Result<FixesEstimate, String> {
    if gt_measures.len() != lattice.measures.len() {
        return Err(format!(
            "ground truth has {} measures but lattice has {}",
            gt_measures.len(),
            lattice.measures.len()
        ));
    }
    let mut per_measure = Vec::with_capacity(gt_measures.len());
    for (gt, m) in gt_measures.iter().zip(lattice.measures.iter()) {
        per_measure.push(measure_gestures(&split_tokens(gt), &m.candidates));
    }
    Ok(FixesEstimate { total: per_measure.iter().sum(), per_measure })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_basics() {
        let t = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(token_edit_distance(&t(&[]), &t(&[])), 0);
        assert_eq!(token_edit_distance(&t(&["a"]), &t(&["a"])), 0);
        assert_eq!(token_edit_distance(&t(&["a"]), &t(&["b"])), 1);
        assert_eq!(
            token_edit_distance(&t(&["a", "b", "c"]), &t(&["a", "x", "c"])),
            1
        );
        assert_eq!(
            token_edit_distance(&t(&["a", "b"]), &t(&["a", "b", "c", "d"])),
            2
        );
    }

    #[test]
    fn gestures_zero_one_or_edits() {
        let t = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        // Exact top-1: no gesture.
        assert_eq!(measure_gestures(&t(&["a", "b"]), &t(&["a b", "a c"])), 0);
        // Top-1 wrong, candidate #2 exact: one click.
        assert_eq!(measure_gestures(&t(&["a", "c"]), &t(&["a b", "a c"])), 1);
        // Neither matches: token edits vs top-1.
        assert_eq!(
            measure_gestures(&t(&["a", "x", "c"]), &t(&["a b c", "a y c"])),
            1
        );
        assert_eq!(measure_gestures(&t(&["q", "q", "q"]), &t(&["a", "b"])), 3);
    }

    #[test]
    fn leading_headers_do_not_count_as_edits() {
        let t = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        // GT carries the copied header, candidate starts at voice-1.
        assert_eq!(
            measure_gestures(
                &t(&["clef-G-2", "key+0", "time-4-4", "voice-1", "quarter", "C", "4"]),
                &t(&["voice-1 quarter C 4", "voice-1 quarter D 4"]),
            ),
            0
        );
        // Mid-measure attribute changes are content, not context:
        // one deletion (the clef change) vs top-1.
        assert_eq!(
            measure_gestures(
                &t(&["clef-G-2", "voice-1", "clef-F-4", "quarter", "C", "4"]),
                &t(&["voice-1 quarter C 4", "voice-1 quarter C 4"]),
            ),
            1
        );
    }

    #[test]
    fn fixes_estimate_sums_and_rejects_mismatch() {
        let lat: LatticeDoc = serde_json::from_str(
            r#"{"stem": "s",
                "measures": [
                  {"measure": 0, "confidence": 0.9, "cluster_size": 2,
                   "candidates": ["a b", "a c"]},
                  {"measure": 1, "confidence": 0.4, "cluster_size": 1,
                   "candidates": ["d", "e"]}]}"#,
        )
        .unwrap();
        let est =
            fixes_estimate(&["a b".to_string(), "e".to_string()], &lat).unwrap();
        // Measure 0 exact (0) + measure 1 via candidate #2 (1).
        assert_eq!(est.per_measure, vec![0, 1]);
        assert_eq!(est.total, 1);
        assert!(fixes_estimate(&["only-one".to_string()], &lat).is_err());
    }
}
