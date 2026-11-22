//! Notation-as-written scoring over [`NotationEvent`]s: the page must
//! look identical to what the model saw, so enharmonic equivalence is an
//! error (B# is NOT C), and spelling, staff, voice, rests, tuplets and
//! attachment marks all count. The frozen reference ports
//! ([`strict`](crate::strict), [`strict_xml`](crate::strict_xml)) stay
//! untouched; this module is the replacement ruler.

use assembly_serialize::musicxml_import::NotationEvent;
use serde::{Deserialize, Serialize};

/// Tolerances in beats, mirroring the reference scorer.
const ONSET_TOL: f64 = 0.0625;
const DUR_TOL: f64 = 0.0625;

fn round3(x: f64) -> f64 {
    let scaled = x * 1000.0;
    let floor = scaled.floor();
    if (scaled - floor - 0.5).abs() < 1e-9 {
        if (floor as i64) % 2 == 0 {
            floor / 1000.0
        } else {
            (floor + 1.0) / 1000.0
        }
    } else {
        scaled.round() / 1000.0
    }
}

fn f1(precision: f64, recall: f64) -> f64 {
    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

fn prf(tp: usize, hyp_len: usize, ref_len: usize) -> (f64, f64, f64) {
    let precision = if hyp_len == 0 {
        1.0
    } else {
        tp as f64 / hyp_len as f64
    };
    let recall = if ref_len == 0 {
        1.0
    } else {
        tp as f64 / ref_len as f64
    };
    (precision, recall, f1(precision, recall))
}

/// Drift-immune order predicate: the full visible symbol identity with
/// the clock factored out (no onset, duration or measure). LCS under
/// this predicate is the bar-reset upper bound: symbols right in the
/// right order regardless of clock. The gap between `drift_f1` and the
/// timing-bound `notation_f1` IS the clock/drift share of the error.
fn order_eq(a: &NotationEvent, b: &NotationEvent) -> bool {
    a.staff == b.staff
        && a.voice == b.voice
        && a.rest == b.rest
        && a.tuplet == b.tuplet
        && a.grace == b.grace
        && (a.rest
            || (a.midi == b.midi
                && a.step == b.step
                && a.alter == b.alter
                && a.octave == b.octave))
}

/// Full-identity predicate: timing + MIDI + spelling + staff + voice +
/// rest/tuplet/grace class. B# is NOT C (step/alter/octave are identity);
/// attachments ride on notes and are priced separately.
fn notation_eq(a: &NotationEvent, b: &NotationEvent) -> bool {
    (a.onset - b.onset).abs() <= ONSET_TOL
        && (a.dur - b.dur).abs() <= DUR_TOL
        && a.staff == b.staff
        && a.voice == b.voice
        && a.rest == b.rest
        && a.tuplet == b.tuplet
        && a.grace == b.grace
        && (a.rest
            || (a.midi == b.midi
                && a.step == b.step
                && a.alter == b.alter
                && a.octave == b.octave))
}

fn timing_eq(a: &NotationEvent, b: &NotationEvent) -> bool {
    (a.onset - b.onset).abs() <= ONSET_TOL && (a.dur - b.dur).abs() <= DUR_TOL
}

/// LCS index pairs under an arbitrary predicate, with the reference
/// backtrack (`>=` prefers up, match requires the diagonal step).
fn lcs_pairs(
    gt: &[NotationEvent],
    omr: &[NotationEvent],
    eq: &dyn Fn(&NotationEvent, &NotationEvent) -> bool,
) -> Vec<(usize, usize)> {
    if gt.is_empty() || omr.is_empty() {
        return Vec::new();
    }
    let (n, m) = (gt.len(), omr.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for (i, a) in gt.iter().enumerate() {
        for (j, b) in omr.iter().enumerate() {
            dp[i + 1][j + 1] = if eq(a, b) {
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
        if eq(&gt[i - 1], &omr[j - 1]) && dp[i][j] == dp[i - 1][j - 1] + 1 {
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

fn pitched(it: &[NotationEvent]) -> Vec<&NotationEvent> {
    it.iter().filter(|e| !e.rest).collect()
}

/// Canonical per-event key for measure-exact comparison: the full
/// visible identity, attachments (articulations, slurs, dynamics,
/// hairpins) included.
fn event_key(e: &NotationEvent) -> String {
    let mut arts = e.articulations.clone();
    arts.sort();
    let mut dyns = e.dynamics.clone();
    dyns.sort();
    let mut hairs = e.hairpins.clone();
    hairs.sort();
    format!(
        "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
        e.onset,
        e.midi,
        e.dur,
        e.staff,
        e.step,
        e.alter,
        e.octave,
        e.voice,
        e.rest as u8,
        e.tuplet as u8,
        e.grace as u8,
        arts.join("+"),
        e.slur_start as u8,
        e.slur_stop as u8,
        dyns.join("+"),
        hairs.join("+"),
        e.measure,
    )
}

fn measure_exact_rate(gt: &[NotationEvent], omr: &[NotationEvent]) -> Option<f64> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut gt_by: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    let mut omr_by: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for e in gt {
        gt_by.entry(e.measure).or_default().push(event_key(e));
    }
    for e in omr {
        omr_by.entry(e.measure).or_default().push(event_key(e));
    }
    let ordinals: BTreeSet<u32> = gt_by.keys().chain(omr_by.keys()).copied().collect();
    if ordinals.is_empty() {
        return None;
    }
    let mut ok = 0;
    for m in &ordinals {
        let mut g = gt_by.get(m).cloned().unwrap_or_default();
        let mut o = omr_by.get(m).cloned().unwrap_or_default();
        g.sort();
        o.sort();
        if g == o {
            ok += 1;
        }
    }
    Some(round3(ok as f64 / ordinals.len() as f64))
}

/// Attachment marks on one event as a multiset of tokens: closed-subset
/// articulations, slur endpoints, dynamics, and hairpin wedges.
/// Global-multiset precision/recall over these tokens is the spray meter:
/// extra marks on any event cost precision, missing marks cost recall,
/// without moving alignment.
fn attach_tokens(e: &NotationEvent) -> Vec<String> {
    let mut toks: Vec<String> = e.articulations.clone();
    if e.slur_start {
        toks.push("slur-start".to_string());
    }
    if e.slur_stop {
        toks.push("slur-stop".to_string());
    }
    toks.extend(e.dynamics.iter().cloned());
    toks.extend(e.hairpins.iter().cloned());
    toks.sort();
    toks
}

fn attach_counts(it: &[NotationEvent]) -> std::collections::BTreeMap<String, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for e in it {
        for t in attach_tokens(e) {
            *counts.entry(t).or_insert(0) += 1;
        }
    }
    counts
}

fn dynamic_counts(it: &[NotationEvent]) -> std::collections::BTreeMap<String, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for e in it {
        for t in &e.dynamics {
            *counts.entry(t.clone()).or_insert(0) += 1;
        }
    }
    counts
}

fn hairpin_counts(it: &[NotationEvent]) -> std::collections::BTreeMap<String, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for e in it {
        for t in &e.hairpins {
            *counts.entry(t.clone()).or_insert(0) += 1;
        }
    }
    counts
}

fn multiset_prf(
    g_counts: &std::collections::BTreeMap<String, usize>,
    o_counts: &std::collections::BTreeMap<String, usize>,
) -> (f64, f64, f64) {
    let mut tp = 0usize;
    let mut g_total = 0usize;
    let mut o_total = 0usize;
    let keys: std::collections::BTreeSet<&String> =
        g_counts.keys().chain(o_counts.keys()).collect();
    for k in keys {
        let g = g_counts.get(k).copied().unwrap_or(0);
        let o = o_counts.get(k).copied().unwrap_or(0);
        tp += g.min(o);
        g_total += g;
        o_total += o;
    }
    prf(tp, o_total, g_total)
}

/// Per-staff header exact-match rate between two [`StaffHeader`] lists
/// (clef sign/line, key fifths, beats/beat-type must all agree; a staff
/// present on only one side is a mismatch). `None` when both are empty.
#[allow(dead_code)]
pub fn header_match_rate(
    a: &[assembly_serialize::musicxml_import::StaffHeader],
    b: &[assembly_serialize::musicxml_import::StaffHeader],
) -> Option<f64> {
    use std::collections::BTreeMap;
    if a.is_empty() && b.is_empty() {
        return None;
    }
    let key = |h: &assembly_serialize::musicxml_import::StaffHeader| {
        (h.clef_sign, h.clef_line, h.key_fifths, h.beats, h.beat_type)
    };
    let mut am: BTreeMap<u32, _> = BTreeMap::new();
    let mut bm: BTreeMap<u32, _> = BTreeMap::new();
    for h in a {
        am.insert(h.staff, key(h));
    }
    for h in b {
        bm.insert(h.staff, key(h));
    }
    let mut staves: Vec<u32> = am.keys().chain(bm.keys()).copied().collect();
    staves.sort();
    staves.dedup();
    if staves.is_empty() {
        return None;
    }
    let ok = staves
        .iter()
        .filter(|s| am.get(s) == bm.get(s))
        .count();
    Some(round3(ok as f64 / staves.len() as f64))
}

/// Per-measure, per-staff header exact-match rate between two
/// [`MeasureStaffHeader`] transition lists (initial measure-1 headers
/// plus every mid-stream clef, key, or time-signature change). A
/// `(measure, staff)` transition present on only one side, or with any
/// differing field, is a mismatch. `None` when both are empty.
pub fn measure_header_match_rate(
    a: &[assembly_serialize::musicxml_import::MeasureStaffHeader],
    b: &[assembly_serialize::musicxml_import::MeasureStaffHeader],
) -> Option<f64> {
    use std::collections::BTreeMap;
    if a.is_empty() && b.is_empty() {
        return None;
    }
    let val = |h: &assembly_serialize::musicxml_import::MeasureStaffHeader| {
        (h.clef_sign, h.clef_line, h.key_fifths, h.beats, h.beat_type)
    };
    let mut am: BTreeMap<(u32, u32), _> = BTreeMap::new();
    let mut bm: BTreeMap<(u32, u32), _> = BTreeMap::new();
    for h in a {
        am.insert((h.measure, h.staff), val(h));
    }
    for h in b {
        bm.insert((h.measure, h.staff), val(h));
    }
    let mut keys: Vec<(u32, u32)> = am.keys().chain(bm.keys()).copied().collect();
    keys.sort();
    keys.dedup();
    if keys.is_empty() {
        return None;
    }
    let ok = keys
        .iter()
        .filter(|k| am.get(k) == bm.get(k))
        .count();
    Some(round3(ok as f64 / keys.len() as f64))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NotationReport {
    pub notation_matched: usize,
    pub gt_events: usize,
    pub omr_events: usize,
    pub notation_precision: f64,
    pub notation_recall: f64,
    pub notation_f1: f64,
    pub drift_f1: f64,
    pub pitch_f1: f64,
    pub spell_f1: f64,
    pub staff_f1: f64,
    pub voice_f1: f64,
    pub rest_f1: f64,
    pub tuplet_f1: f64,
    pub grace_f1: f64,
    pub attach_precision: f64,
    pub attach_recall: f64,
    pub attach_agreement: f64,
    pub dynamics_precision: f64,
    pub dynamics_recall: f64,
    pub dynamics_f1: f64,
    pub hairpin_precision: f64,
    pub hairpin_recall: f64,
    pub hairpin_f1: f64,
    pub measure_exact: Option<f64>,
    pub header_match: Option<f64>,
    pub omr_ned: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hypothesis_sha256: Option<String>,
}

#[derive(Debug, Clone)]
pub enum NotationOutcome {
    Scored(NotationReport),
    EmptyReference {
        omr_len: usize,
        hyp_empty: bool,
        exact_match: f64,
    },
}

/// Score a hypothesis event list against ground truth under the
/// as-written rule. Facet F1s share one timing-anchored alignment each;
/// attachments are priced globally (spray meter) and on aligned pairs
/// (agreement). Headers ride separately via [`header_match_rate`].
pub fn score_notation_pair(gt: &[NotationEvent], omr: &[NotationEvent]) -> NotationOutcome {
    if gt.is_empty() {
        let hyp_empty = omr.is_empty();
        return NotationOutcome::EmptyReference {
            omr_len: omr.len(),
            hyp_empty,
            exact_match: if hyp_empty { 1.0 } else { 0.0 },
        };
    }
    let pairs = lcs_pairs(gt, omr, &notation_eq);
    let (notation_precision, notation_recall, notation_f1) =
        prf(pairs.len(), omr.len(), gt.len());
    let drift_tp = lcs_pairs(gt, omr, &order_eq).len();
    let (_, _, drift_f1) = prf(drift_tp, omr.len(), gt.len());

    let pitched_gt = pitched(gt);
    let pitched_omr = pitched(omr);
    let facet = |eq: fn(&NotationEvent, &NotationEvent) -> bool,
                 g: &[&NotationEvent],
                 o: &[&NotationEvent]|
     -> f64 {
        let tp = lcs_pairs(gt, omr, &|a, b| timing_eq(a, b) && eq(a, b)).len();
        let (_, _, f) = prf(tp, o.len(), g.len());
        f
    };
    let pitch_f1 = facet(|a, b| !a.rest && !b.rest && a.midi == b.midi, &pitched_gt, &pitched_omr);
    let spell_f1 = facet(
        |a, b| {
            !a.rest && !b.rest && a.step == b.step && a.alter == b.alter && a.octave == b.octave
        },
        &pitched_gt,
        &pitched_omr,
    );
    let staff_f1 = facet(|a, b| a.staff == b.staff, &gt.iter().collect::<Vec<_>>(), &omr.iter().collect::<Vec<_>>());
    let voice_f1 = facet(|a, b| a.voice == b.voice, &gt.iter().collect::<Vec<_>>(), &omr.iter().collect::<Vec<_>>());
    let rest_tp = lcs_pairs(gt, omr, &|a, b| {
        timing_eq(a, b) && a.rest && b.rest && a.staff == b.staff && a.voice == b.voice
    })
    .len();
    let n_rest = |it: &[NotationEvent]| it.iter().filter(|e| e.rest).count();
    let (_, _, rest_f1) = prf(rest_tp, n_rest(omr), n_rest(gt));
    let tup_tp = lcs_pairs(gt, omr, &|a, b| timing_eq(a, b) && !a.rest && a.tuplet == b.tuplet).len();
    let (_, _, tuplet_f1) = prf(tup_tp, pitched_omr.len(), pitched_gt.len());
    let grace_tp = lcs_pairs(gt, omr, &|a, b| {
        timing_eq(a, b)
            && a.grace
            && b.grace
            && a.staff == b.staff
            && a.voice == b.voice
            && a.step == b.step
            && a.alter == b.alter
            && a.octave == b.octave
    })
    .len();
    let n_grace = |it: &[NotationEvent]| it.iter().filter(|e| e.grace).count();
    let (_, _, grace_f1) = prf(grace_tp, n_grace(omr), n_grace(gt));

    let g_counts = attach_counts(gt);
    let o_counts = attach_counts(omr);
    let (attach_precision, attach_recall, _) = multiset_prf(&g_counts, &o_counts);
    let (dynamics_precision, dynamics_recall, dynamics_f1) =
        multiset_prf(&dynamic_counts(gt), &dynamic_counts(omr));
    let (hairpin_precision, hairpin_recall, hairpin_f1) =
        multiset_prf(&hairpin_counts(gt), &hairpin_counts(omr));
    let attach_agreement = if pairs.is_empty() {
        1.0
    } else {
        let ok = pairs
            .iter()
            .filter(|(gi, oi)| attach_tokens(&gt[*gi]) == attach_tokens(&omr[*oi]))
            .count();
        round3(ok as f64 / pairs.len() as f64)
    };

    NotationOutcome::Scored(NotationReport {
        notation_matched: pairs.len(),
        gt_events: gt.len(),
        omr_events: omr.len(),
        notation_precision: round3(notation_precision),
        notation_recall: round3(notation_recall),
        notation_f1: round3(notation_f1),
        drift_f1: round3(drift_f1),
        pitch_f1: round3(pitch_f1),
        spell_f1: round3(spell_f1),
        staff_f1: round3(staff_f1),
        voice_f1: round3(voice_f1),
        rest_f1: round3(rest_f1),
        tuplet_f1: round3(tuplet_f1),
        grace_f1: round3(grace_f1),
        attach_precision: round3(attach_precision),
        attach_recall: round3(attach_recall),
        attach_agreement: round3(attach_agreement),
        dynamics_precision: round3(dynamics_precision),
        dynamics_recall: round3(dynamics_recall),
        dynamics_f1: round3(dynamics_f1),
        hairpin_precision: round3(hairpin_precision),
        hairpin_recall: round3(hairpin_recall),
        hairpin_f1: round3(hairpin_f1),
        measure_exact: measure_exact_rate(gt, omr),
        header_match: None,
        omr_ned: None,
        reference_sha256: None,
        hypothesis_sha256: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(
        measure: u32,
        onset: f64,
        step: char,
        alter: i32,
        octave: i32,
        dur: f64,
        staff: u32,
        voice: u32,
    ) -> NotationEvent {
        let midi = (octave + 1) * 12
            + match step {
                'C' => 0,
                'D' => 2,
                'E' => 4,
                'F' => 5,
                'G' => 7,
                'A' => 9,
                'B' => 11,
                _ => 0,
            }
            + alter;
        NotationEvent {
            measure,
            onset,
            midi,
            dur,
            staff,
            step,
            alter,
            octave,
            voice,
            rest: false,
            tuplet: false,
            grace: false,
            articulations: Vec::new(),
            slur_start: false,
            slur_stop: false,
            dynamics: Vec::new(),
            hairpins: Vec::new(),
        }
    }

    fn rest(measure: u32, onset: f64, dur: f64, staff: u32, voice: u32) -> NotationEvent {
        NotationEvent {
            measure,
            onset,
            midi: -1,
            dur,
            staff,
            step: 'R',
            alter: 0,
            octave: -1,
            voice,
            rest: true,
            tuplet: false,
            grace: false,
            articulations: Vec::new(),
            slur_start: false,
            slur_stop: false,
            dynamics: Vec::new(),
            hairpins: Vec::new(),
        }
    }

    fn scored(gt: &[NotationEvent], omr: &[NotationEvent]) -> NotationReport {
        match score_notation_pair(gt, omr) {
            NotationOutcome::Scored(r) => r,
            NotationOutcome::EmptyReference { .. } => panic!("expected a scored pair"),
        }
    }

    #[test]
    fn identical_streams_score_one() {
        let gt = vec![
            note(1, 0.0, 'C', 0, 4, 1.0, 0, 1),
            note(1, 1.0, 'D', 0, 4, 1.0, 0, 1),
        ];
        let r = scored(&gt, &gt);
        assert_eq!(r.notation_f1, 1.0);
        assert_eq!(r.spell_f1, 1.0);
        assert_eq!(r.voice_f1, 1.0);
        assert_eq!(r.rest_f1, 1.0);
    }

    #[test]
    fn bsharp_is_not_c() {
        // MIDI 60 both sides; as-written they differ. The old ruler
        // matches; the notation ruler must not.
        let gt = vec![note(1, 0.0, 'B', 1, 3, 1.0, 0, 1)];
        let omr = vec![note(1, 0.0, 'C', 0, 4, 1.0, 0, 1)];
        let r = scored(&gt, &omr);
        assert_eq!(r.notation_f1, 0.0);
        assert_eq!(r.spell_f1, 0.0);
        // Pitch alone still matches: the gap is spelling, not hearing.
        assert_eq!(r.pitch_f1, 1.0);
    }

    #[test]
    fn voice_mismatch_is_an_error() {
        let gt = vec![note(1, 0.0, 'C', 0, 4, 1.0, 0, 1)];
        let omr = vec![note(1, 0.0, 'C', 0, 4, 1.0, 0, 2)];
        let r = scored(&gt, &omr);
        assert_eq!(r.notation_f1, 0.0);
        assert_eq!(r.voice_f1, 0.0);
    }

    #[test]
    fn staff_mismatch_is_an_error() {
        let gt = vec![note(1, 0.0, 'C', 0, 4, 1.0, 0, 1)];
        let omr = vec![note(1, 0.0, 'C', 0, 4, 1.0, 1, 1)];
        let r = scored(&gt, &omr);
        assert_eq!(r.notation_f1, 0.0);
        assert_eq!(r.staff_f1, 0.0);
    }

    #[test]
    fn rest_matches_rest_not_note() {
        let gt = vec![rest(1, 0.0, 1.0, 0, 1)];
        let same = scored(&gt, &gt);
        assert_eq!(same.rest_f1, 1.0);
        let note_as_rest = vec![note(1, 0.0, 'C', 0, 4, 1.0, 0, 1)];
        let r = scored(&gt, &note_as_rest);
        assert_eq!(r.notation_f1, 0.0);
        assert_eq!(r.rest_f1, 0.0);
    }

    #[test]
    fn tuplet_bracket_mismatch_is_an_error() {
        let mut gt = vec![note(1, 0.0, 'D', 0, 4, 0.5, 0, 1)];
        gt[0].tuplet = true;
        let omr = vec![note(1, 0.0, 'D', 0, 4, 0.5, 0, 1)];
        let r = scored(&gt, &omr);
        assert_eq!(r.notation_f1, 0.0);
    }

    #[test]
    fn spray_is_priced_without_breaking_alignment() {
        // Extra staccato on the same note: the note still aligns
        // (attachments ride on notes), but attachment precision drops.
        let gt = vec![note(1, 0.0, 'C', 0, 4, 1.0, 0, 1)];
        let mut omr = vec![note(1, 0.0, 'C', 0, 4, 1.0, 0, 1)];
        omr[0].articulations = vec!["stacc".to_string()];
        let r = scored(&gt, &omr);
        assert_eq!(r.notation_matched, 1);
        assert_eq!(r.attach_precision, 0.0);
        assert_eq!(r.attach_recall, 1.0);
    }

    #[test]
    fn missing_attachment_costs_recall_not_alignment() {
        let mut gt = vec![note(1, 0.0, 'C', 0, 4, 1.0, 0, 1)];
        gt[0].articulations = vec!["stacc".to_string()];
        let omr = vec![note(1, 0.0, 'C', 0, 4, 1.0, 0, 1)];
        let r = scored(&gt, &omr);
        assert_eq!(r.notation_matched, 1);
        assert_eq!(r.attach_precision, 1.0);
        assert_eq!(r.attach_recall, 0.0);
    }

    #[test]
    fn slur_endpoints_are_priced() {
        let mut gt = vec![
            note(1, 0.0, 'D', 0, 4, 1.0, 0, 1),
            note(1, 1.0, 'E', 0, 4, 1.0, 0, 1),
        ];
        gt[0].slur_start = true;
        gt[1].slur_stop = true;
        let omr = vec![
            note(1, 0.0, 'D', 0, 4, 1.0, 0, 1),
            note(1, 1.0, 'E', 0, 4, 1.0, 0, 1),
        ];
        let r = scored(&gt, &omr);
        assert_eq!(r.notation_matched, 2);
        assert_eq!(r.attach_recall, 0.0);
    }

    #[test]
    fn measure_exact_requires_spelling() {
        // Same MIDI in both measures; measure 2 respells D as C#... no:
        // measure 2 respells E as F-flat (midi 64 both).
        let gt = vec![
            note(1, 0.0, 'D', 0, 4, 1.0, 0, 1),
            note(2, 0.0, 'E', 0, 4, 1.0, 0, 1),
        ];
        let mut omr = gt.clone();
        omr[1].step = 'F';
        omr[1].alter = -1;
        let r = scored(&gt, &omr);
        assert_eq!(r.measure_exact, Some(0.5));
    }

    #[test]
    fn drift_immune_order_scores_symbols_despite_clock_shift() {
        // Same symbols, compressed clock: notation F1 craters (timing is
        // identity) while drift F1 stays perfect (order is identity).
        // The gap between the two IS the clock/drift share.
        let gt = vec![
            note(1, 0.0, 'C', 0, 4, 1.0, 0, 1),
            note(1, 1.0, 'D', 0, 4, 1.0, 0, 1),
            note(1, 2.0, 'E', 0, 4, 1.0, 0, 1),
        ];
        let omr = vec![
            note(1, 0.0, 'C', 0, 4, 0.5, 0, 1),
            note(1, 0.5, 'D', 0, 4, 0.5, 0, 1),
            note(1, 1.0, 'E', 0, 4, 0.5, 0, 1),
        ];
        let r = scored(&gt, &omr);
        assert_eq!(r.drift_f1, 1.0);
        assert!(r.notation_f1 < 1.0);
    }

    #[test]
    fn empty_reference_routes_to_exact_match() {
        let gt: Vec<NotationEvent> = Vec::new();
        match score_notation_pair(&gt, &[]) {
            NotationOutcome::EmptyReference { exact_match, .. } => assert_eq!(exact_match, 1.0),
            NotationOutcome::Scored(_) => panic!("expected empty routing"),
        }
        let omr = vec![note(1, 0.0, 'C', 0, 4, 1.0, 0, 1)];
        match score_notation_pair(&gt, &omr) {
            NotationOutcome::EmptyReference { exact_match, .. } => assert_eq!(exact_match, 0.0),
            NotationOutcome::Scored(_) => panic!("expected empty routing"),
        }
    }

    #[test]
    fn missing_dynamic_drops_attach_and_dynamics_recall() {
        let mut gt = vec![note(1, 0.0, 'C', 0, 4, 1.0, 0, 1)];
        gt[0].dynamics = vec!["dyn-mf".to_string()];
        let omr = vec![note(1, 0.0, 'C', 0, 4, 1.0, 0, 1)];
        let r = scored(&gt, &omr);
        assert_eq!(r.notation_matched, 1);
        assert_eq!(r.attach_precision, 1.0);
        assert_eq!(r.attach_recall, 0.0);
        assert_eq!(r.dynamics_precision, 1.0);
        assert_eq!(r.dynamics_recall, 0.0);
        assert_eq!(r.dynamics_f1, 0.0);
        assert_eq!(r.measure_exact, Some(0.0));
    }

    #[test]
    fn extra_hairpin_drops_attach_and_hairpin_precision() {
        let gt = vec![
            note(1, 0.0, 'C', 0, 4, 1.0, 0, 1),
            note(1, 1.0, 'D', 0, 4, 1.0, 0, 1),
        ];
        let mut omr = gt.clone();
        omr[0].hairpins = vec!["hairpin-cresc-start".to_string()];
        omr[1].hairpins = vec!["hairpin-stop".to_string()];
        let r = scored(&gt, &omr);
        assert_eq!(r.notation_matched, 2);
        assert_eq!(r.attach_precision, 0.0);
        assert_eq!(r.attach_recall, 1.0);
        assert_eq!(r.hairpin_precision, 0.0);
        assert_eq!(r.hairpin_recall, 1.0);
        assert_eq!(r.hairpin_f1, 0.0);
    }

    #[test]
    fn missing_hairpin_on_note_and_rest_drops_recall() {
        let mut gt = vec![
            note(1, 0.0, 'C', 0, 4, 1.0, 0, 1),
            rest(1, 1.0, 1.0, 0, 1),
        ];
        gt[0].hairpins = vec!["hairpin-dim-start".to_string()];
        gt[1].hairpins = vec!["hairpin-stop".to_string()];
        let omr = vec![
            note(1, 0.0, 'C', 0, 4, 1.0, 0, 1),
            rest(1, 1.0, 1.0, 0, 1),
        ];
        let r = scored(&gt, &omr);
        assert_eq!(r.notation_matched, 2);
        assert_eq!(r.attach_recall, 0.0);
        assert_eq!(r.hairpin_recall, 0.0);
        assert_eq!(r.hairpin_f1, 0.0);
    }

    #[test]
    fn end_to_end_xml_dynamics_and_hairpins_priced() {
        use assembly_serialize::musicxml_import::notation_events;
        let gt_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<direction><direction-type><dynamics><f/></dynamics></direction-type></direction>
<direction><direction-type><wedge type="crescendo"/></direction-type></direction>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
<direction><direction-type><wedge type="stop"/></direction-type></direction>
</measure>
</part>
</score-partwise>"#;
        let hyp_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;
        let gt = notation_events(gt_xml).unwrap();
        let hyp = notation_events(hyp_xml).unwrap();
        let exact = scored(&gt, &gt);
        assert_eq!(exact.attach_precision, 1.0);
        assert_eq!(exact.attach_recall, 1.0);
        assert_eq!(exact.dynamics_f1, 1.0);
        assert_eq!(exact.hairpin_f1, 1.0);
        let dropped = scored(&gt, &hyp);
        assert_eq!(dropped.attach_recall, 0.0);
        assert_eq!(dropped.dynamics_recall, 0.0);
        assert_eq!(dropped.hairpin_recall, 0.0);
    }

    #[test]
    fn midstream_clef_key_time_changes_priced_in_measure_header_match_rate() {
        use assembly_serialize::musicxml_import::measure_staff_headers;
        let gt_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>4</divisions><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
</measure>
<measure number="2">
<attributes><clef><sign>F</sign><line>4</line></clef><key><fifths>2</fifths></key><time><beats>3</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>D</step><octave>3</octave></pitch><duration>4</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;
        // Hypothesis matches m1, but completely misses the m2 clef/key/time change.
        let hyp_miss_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>4</divisions><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
</measure>
<measure number="2">
<note><pitch><step>D</step><octave>3</octave></pitch><duration>4</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;
        // Hypothesis gets clef & time in m2 right, but wrong key (fifths=1 instead of 2).
        let hyp_wrong_key = gt_xml.replace("<fifths>2</fifths>", "<fifths>1</fifths>");
        let gt_h = measure_staff_headers(gt_xml).unwrap();
        let miss_h = measure_staff_headers(hyp_miss_xml).unwrap();
        let wrong_h = measure_staff_headers(&hyp_wrong_key).unwrap();
        assert_eq!(measure_header_match_rate(&gt_h, &gt_h), Some(1.0));
        assert_eq!(measure_header_match_rate(&gt_h, &miss_h), Some(0.5));
        assert_eq!(measure_header_match_rate(&gt_h, &wrong_h), Some(0.5));
    }

    #[test]
    fn missing_and_extra_grace_notes_drop_grace_f1_and_notation_f1() {
        let mut grace_d5 = note(1, 0.0, 'D', 0, 5, 0.0, 0, 1);
        grace_d5.grace = true;
        let main_c5 = note(1, 0.0, 'C', 0, 5, 1.0, 0, 1);

        let gt_with = vec![grace_d5.clone(), main_c5.clone()];
        let omr_without = vec![main_c5.clone()];

        // Missing grace note drops grace_f1 to 0.0 and lowers notation_f1.
        let miss = scored(&gt_with, &omr_without);
        assert_eq!(miss.grace_f1, 0.0);
        assert!(miss.notation_f1 < 1.0);
        assert_eq!(miss.notation_matched, 1);

        // Hallucinated grace note drops grace_f1 to 0.0 and lowers notation_f1.
        let extra = scored(&omr_without, &gt_with);
        assert_eq!(extra.grace_f1, 0.0);
        assert!(extra.notation_f1 < 1.0);
        assert_eq!(extra.notation_matched, 1);

        // Exact match scores 1.0 on both grace_f1 and notation_f1.
        let exact = scored(&gt_with, &gt_with);
        assert_eq!(exact.grace_f1, 1.0);
        assert_eq!(exact.notation_f1, 1.0);
    }

    #[test]
    fn grace_note_confused_with_normal_note_fails_grace_f1_and_lcs() {
        let mut gt_grace = note(1, 0.0, 'D', 0, 5, 0.0, 0, 1);
        gt_grace.grace = true;
        // Identical pitch, onset, dur, staff, voice — differs ONLY in `grace`.
        let omr_normal = note(1, 0.0, 'D', 0, 5, 0.0, 0, 1);
        let r = scored(&[gt_grace], &[omr_normal]);
        assert_eq!(r.grace_f1, 0.0);
        assert_eq!(r.notation_f1, 0.0);
        assert_eq!(r.drift_f1, 0.0);
    }
}
