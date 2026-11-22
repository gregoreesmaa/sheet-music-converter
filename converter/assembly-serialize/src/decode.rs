//! bekern token stream -> [`ScoreInput`](crate::ScoreInput) (MVD scoring).
//!
//! Inverse of [`to_bekern`](crate::bekern::to_bekern): reconstructs
//! absolute performed onsets by accumulating written durations per
//! `voice-N` segment, scaling contributors inside open
//! `tuplet-start-A-N` spans by N/A. Mid-stream `clef`/`key`/`time`
//! tokens, measure dynamics/hairpins, barline variants, `grace` notes
//! and slur/articulation/fermata modifiers all parse into
//! [`ScoreInput`](crate::ScoreInput) (notes plus `measure_attrs`).
//! `bar-rest` jumps its voice cursor to the bar end. Voice assignment
//! is exact (markers are normative); a stream with missing markers
//! degrades to single-segment accumulation, never a crash.
//!
//! Tie handling differs by consumer (see [`decode_bekern`] vs
//! [`decode_bekern_unmerged`]): the note-list path merges chains
//! (its reference adapter merges too), while the MusicXML path
//! preserves every link (the XML scorer compares tie segments).

use crate::{MeasureAttr, NoteInput, PartInput, ScoreInput, TimeSig, SCORE_DIVISIONS};

fn written_dur(token: &str) -> Option<f64> {
    match token {
        "breve" => Some(8.0),
        "whole" => Some(4.0),
        "half" => Some(2.0),
        "quarter" => Some(1.0),
        "eighth" => Some(0.5),
        "16th" => Some(0.25),
        "32nd" => Some(0.125),
        "64th" => Some(0.0625),
        "128th" => Some(0.03125),
        _ => None,
    }
}

fn alter_suffix(alter: i32) -> Result<&'static str, String> {
    match alter {
        0 => Ok(""),
        1 => Ok("#"),
        2 => Ok("x"),
        -1 => Ok("b"),
        -2 => Ok("bb"),
        _ => Err(format!("unsupported alter {alter}")),
    }
}

/// Slur + modifier marks carried on one note (shared by the merged
/// and unmerged decode paths).
#[derive(Debug, Clone, Default)]
struct NoteMods {
    slur_start: bool,
    slur_stop: bool,
    articulations: Vec<String>,
}

impl NoteMods {
    /// Union of two links' marks (merged tie chains keep every mark).
    fn union(&self, other: &NoteMods) -> NoteMods {
        let mut articulations = self.articulations.clone();
        for a in &other.articulations {
            if !articulations.contains(a) {
                articulations.push(a.clone());
            }
        }
        NoteMods {
            slur_start: self.slur_start || other.slur_start,
            slur_stop: self.slur_stop || other.slur_stop,
            articulations,
        }
    }
}

fn mk_note(
    pitch: String,
    onset: f64,
    dur: f64,
    voice: u32,
    mods: &NoteMods,
) -> NoteInput {
    NoteInput {
        pitch: Some(pitch),
        rest: false,
        onset,
        dur,
        voice,
        lyric: None,
        grace: false,
        slur_start: mods.slur_start,
        slur_stop: mods.slur_stop,
        articulations: mods.articulations.clone(),
        tie_start: false,
        tie_stop: false,
    }
}

/// Consume one slur/modifier tail mark, if the head token is one.
fn take_mod(toks: &[&str], i: &mut usize, mods: &mut NoteMods) -> bool {
    match toks.get(*i).copied() {
        Some("slur-start") => {
            mods.slur_start = true;
            *i += 1;
            true
        }
        Some("slur-stop") => {
            mods.slur_stop = true;
            *i += 1;
            true
        }
        Some(s) if ["stacc", "accent", "tenuto", "fermata"].contains(&s) => {
            if !mods.articulations.iter().any(|a| a == s) {
                mods.articulations.push(s.to_string());
            }
            *i += 1;
            true
        }
        _ => false,
    }
}

/// Consume a pitched note's tail: slur/modifier and tie marks in any
/// interleave (encoder order is slurs, modifiers, then ties).
fn take_tail(
    toks: &[&str],
    i: &mut usize,
    mods: &mut NoteMods,
    tie_start: &mut bool,
    tie_stop: &mut bool,
) {
    loop {
        if take_mod(toks, i, mods) {
            continue;
        }
        match toks.get(*i).copied() {
            Some("tie-start") => {
                *tie_start = true;
                *i += 1;
            }
            Some("tie-stop") => {
                *tie_stop = true;
                *i += 1;
            }
            _ => break,
        }
    }
}

/// Consume a rest's tail: slur/modifier marks only (a tie mark ends
/// the tail and errors downstream as an unexpected token).
fn take_tail_rest(toks: &[&str], i: &mut usize, mods: &mut NoteMods) {
    while take_mod(toks, i, mods) {}
}

/// Mutable `measure_attrs` slot for one 1-based measure (created on
/// first use; callers sort by measure at the end).
fn attr_for(attrs: &mut Vec<MeasureAttr>, measure_1based: u32) -> &mut MeasureAttr {
    if let Some(i) = attrs.iter().position(|a| a.measure == measure_1based) {
        return &mut attrs[i];
    }
    attrs.push(MeasureAttr {
        measure: measure_1based,
        ..Default::default()
    });
    attrs.last_mut().expect("just pushed")
}

/// Closing-barline short style for a `bar-*` variant token.
fn barline_right_of(t: &str) -> Option<&'static str> {
    match t {
        "bar-light-heavy" => Some("light-heavy"),
        "bar-repeat-right" => Some("repeat-right"),
        "bar-double" => Some("double"),
        _ => None,
    }
}

/// Apply tie open/close to one pitched note. Returns the note to emit,
/// or `None` when the note is swallowed into an open chain (emitted on
/// close or end-of-stream flush). Same-note start+stop with no open
/// chain emits once (degenerate input, never produced by the encoder).
/// Apply tie open/close to one pitched note. Returns the note to emit,
/// or `None` when the note is swallowed into an open chain (emitted on
/// close or end-of-stream flush). Same-note start+stop with no open
/// chain emits once (degenerate input, never produced by the encoder).
/// Chains are keyed by pitch so parallel tied voices/chord tones merge
/// independently (a single slot would drop all but the last chain).
/// Per-pitch open-chain stack of `(seg, onset, dur_acc)`: parallel
/// tied voices merge independently, including unison doublings where
/// two voices tie the same pitch at once (ds0462 m19 — pitch-only
/// keying let the second tie-start overwrite the first). A tie_stop
/// takes the topmost same-segment entry (exact within a bar: segments
/// are stable per measure). Stops with no same-seg entry take the
/// single open entry when exactly one exists (the ordinary churned
/// cross-bar chain). Empty stack (or several opens with no same-seg
/// match) stays literal: page-cut stops, or adversarial overlaps
/// resolved LIFO-deterministically by taking the stack top. A
/// tie_start on a segment that already has an open entry overwrites it
/// (recency — mirrors the oracle's single-slot merge per (staff,
/// voice, pitch), which drops the superseded head). Stack order is
/// push order (oldest first), so the end-of-stream flush and every
/// fallback are deterministic.
#[allow(clippy::too_many_arguments)]
fn tie_note(
    pending: &mut std::collections::BTreeMap<
        (char, i32, i32),
        Vec<(usize, f64, f64, NoteMods)>,
    >,
    key: (char, i32, i32),
    pitch: String,
    onset: f64,
    performed: f64,
    voice: u32,
    seg: usize,
    tie_start: bool,
    tie_stop: bool,
    mods: &NoteMods,
) -> Option<NoteInput> {
    if tie_stop {
        // Move the stack out so the merge below never holds a borrow
        // across a map mutation.
        let mut stack = pending.remove(&key).unwrap_or_default();
        let i = stack
            .iter()
            .rposition(|(s, _, _, _)| *s == seg)
            .or(if stack.len() == 1 {
                Some(0)
            } else if stack.len() > 1 {
                // Ambiguous remainder (several opens, no same-seg
                // match): LIFO top wins, deterministically.
                // Documented residual.
                Some(stack.len() - 1)
            } else {
                None
            });
        match i {
            Some(i) => {
                let (_, o, acc, head_mods) = stack.remove(i);
                if !stack.is_empty() {
                    pending.insert(key, stack);
                }
                let total = acc + performed;
                let merged = head_mods.union(mods);
                if tie_start {
                    pending
                        .entry(key)
                        .or_default()
                        .push((seg, o, total, merged));
                    return None;
                }
                return Some(mk_note(pitch, o, total, voice, &merged));
            }
            None => {
                debug_assert!(stack.is_empty());
                return Some(mk_note(pitch, onset, performed, voice, mods));
            }
        }
    }
    if tie_start {
        // Recency: a restart on a segment that already has an open
        // entry supersedes it (mirrors the oracle dropping the old
        // head); a start on a fresh segment opens a parallel chain.
        let mut stack = pending.remove(&key).unwrap_or_default();
        stack.retain(|(s, _, _, _)| *s != seg);
        stack.push((seg, onset, performed, mods.clone()));
        pending.insert(key, stack);
        return None;
    }
    Some(mk_note(pitch, onset, performed, voice, mods))
}

/// Decode a bekern page to a single-part [`ScoreInput`], merging tie
/// chains into single notes (link-faithful input stays link-faithful:
/// each chain keeps every link's slur/modifier marks by union).
///
/// This is the MVD note-list path: its reference adapter
/// ([`score_input_from_musicxml`](crate::musicxml_import::score_input_from_musicxml))
/// merges ties too, so both sides agree (grace notes are dropped on
/// both sides as well). For link-faithful MusicXML output — the
/// strict-xml oracle compares tie segments — use
/// [`decode_bekern_unmerged`].
///
/// Voice assignment follows the normative `voice-N` markers (appearance
/// order per measure); a stream with missing markers degrades to
/// single-segment accumulation.
pub fn decode_bekern(tokens: &str) -> Result<ScoreInput, String> {
    decode_inner(tokens, true)
}

/// Decode a bekern page to a single-part [`ScoreInput`] preserving
/// every tie link as its own note (with `tie_start`/`tie_stop` flags
/// for the renderer). Grace notes are kept (zero-width, display
/// duration only). Used by `bekern2musicxml`, whose strict-xml oracle
/// compares tie segments rather than merged notes.
pub fn decode_bekern_unmerged(tokens: &str) -> Result<ScoreInput, String> {
    decode_inner(tokens, false)
}

/// Shared decode core: `merge_ties` selects the note-list semantics
/// ([`decode_bekern`]) vs the notation-faithful semantics
/// ([`decode_bekern_unmerged`]). Token parsing, cursors, spans,
/// measure attributes and error behavior are identical.
fn decode_inner(tokens: &str, merge_ties: bool) -> Result<ScoreInput, String> {
    let toks: Vec<&str> = tokens.split_whitespace().collect();
    let mut key_fifths = 0i32;
    let mut time = TimeSig {
        beats: 4,
        beat_type: 4,
    };
    let mut time_set = false;
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        if t.starts_with("clef-") {
            i += 1;
        } else if let Some(rest) = t.strip_prefix("key") {
            if !time_set {
                key_fifths = rest
                    .parse::<i32>()
                    .map_err(|_| format!("bad key token {t:?}"))?;
            }
            i += 1;
        } else if let Some(rest) = t.strip_prefix("time-") {
            if !time_set {
                let mut parts = rest.split('-');
                let beats: u32 = parts
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| format!("bad time token {t:?}"))?;
                let beat_type: u32 = parts
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| format!("bad time token {t:?}"))?;
                time = TimeSig { beats, beat_type };
                time_set = true;
            }
            i += 1;
        } else {
            break;
        }
    }

    let mut notes: Vec<NoteInput> = Vec::new();
    let mut measure_attrs: Vec<MeasureAttr> = Vec::new();
    let mut measure = 0u32;
    // Per-measure voice cursors (offsets within the measure); segment 0
    // is created lazily by the first event.
    let mut voices: Vec<f64> = vec![0.0];
    let mut seg = 0usize;
    // Last placed note onset per segment (for `chord` tones).
    let mut last_onset: Vec<Option<f64>> = vec![None];
    // Open tuplet ratios per segment (mirrors to_bekern's per-key
    // stacks: within a measure, segment <-> (staff, voice) is bijective,
    // and spans never cross `bar` by encoder construction). Pending
    // starts queue up (nested reopens) and drain into the next event.
    // Invariant: a `tuplet-start` immediately precedes its member event
    // (the encoder never emits `start … voice-X … event`); the grammar
    // mask is documented wider-than-encoder, so a generator that
    // separates a start from its event by a voice marker would misroute
    // the drain — latent, no behavior change.
    let mut spans: Vec<Vec<(u32, u32)>> = vec![Vec::new()];
    let mut pending_spans: Vec<(u32, u32)> = Vec::new();
    // Open tie chain: ((step, octave, alter), seg, onset, dur_acc,
    // marks). Only used when `merge_ties` is set.
    let mut pending_tie: std::collections::BTreeMap<
        (char, i32, i32),
        Vec<(usize, f64, f64, NoteMods)>,
    > = std::collections::BTreeMap::new();
    // Grace state: `grace_pending` is set by a `grace` marker and
    // consumed by the next duration token; `grace_run` marks chord
    // tones inheriting their root's grace.
    let mut grace_pending = false;
    let mut grace_run = false;
    // Whether any note or rest has been placed: mid-stream headers
    // before the first event are effectively leading headers.
    let mut seen_event = false;

    let scale_of = |spans: &[Vec<(u32, u32)>], s: usize| -> f64 {
        spans[s]
            .iter()
            .fold(1.0, |acc, &(a, n)| acc * n as f64 / a as f64)
    };
    let push_rest = |notes: &mut Vec<NoteInput>,
                     onset: f64,
                     dur: f64,
                     voice: u32,
                     mods: &NoteMods,
                     grace: bool| {
        notes.push(NoteInput {
            pitch: None,
            rest: true,
            onset,
            dur,
            voice,
            lyric: None,
            grace,
            slur_start: mods.slur_start,
            slur_stop: mods.slur_stop,
            articulations: mods.articulations.clone(),
            tie_start: false,
            tie_stop: false,
        });
    };

    let mut barlen = time.beats as f64 * 4.0 / time.beat_type.max(1) as f64;
    while i < toks.len() {
        let t = toks[i];
        if t == "bar" || barline_right_of(t).is_some() {
            if grace_pending {
                return Err(format!("dangling grace marker in measure {measure}"));
            }
            if let Some(style) = barline_right_of(t) {
                attr_for(&mut measure_attrs, measure + 1).barline_right =
                    Some(style.to_string());
            }
            measure += 1;
            voices = vec![0.0];
            seg = 0;
            last_onset = vec![None];
            spans = vec![Vec::new()];
            grace_run = false;
            // Pending starts survive the bar: the encoder emits them
            // immediately before their member event, so a start is only
            // ever separated from its event by whole-bar rests, which
            // carry no scale.
            i += 1;
            continue;
        }
        if t == "bar-repeat-left" {
            attr_for(&mut measure_attrs, measure + 1).barline_left =
                Some("repeat-left".to_string());
            i += 1;
            continue;
        }
        if t == "grace" {
            if grace_pending {
                return Err(format!("double grace marker in measure {measure}"));
            }
            grace_pending = true;
            i += 1;
            continue;
        }
        if let Some(body) = t.strip_prefix("clef-") {
            let mut parts = body.split('-');
            let sign = parts
                .next()
                .and_then(|s| s.chars().next())
                .ok_or_else(|| format!("bad clef token {t:?}"))?;
            let line: u32 = parts
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("bad clef token {t:?}"))?;
            if !seen_event && measure == 0 {
                // Effectively leading: the renderer always opens on
                // G/2, so there is no score-level clef to update.
            } else {
                let a = attr_for(&mut measure_attrs, measure + 1);
                a.clef_sign = Some(sign);
                a.clef_line = Some(line);
            }
            i += 1;
            continue;
        }
        if let Some(rest) = t.strip_prefix("key") {
            let fifths = rest
                .parse::<i32>()
                .map_err(|_| format!("bad key token {t:?}"))?;
            if !seen_event && measure == 0 {
                key_fifths = fifths;
            } else {
                attr_for(&mut measure_attrs, measure + 1).key_fifths = Some(fifths);
            }
            i += 1;
            continue;
        }
        if let Some(rest) = t.strip_prefix("time-") {
            let mut parts = rest.split('-');
            let beats: u32 = parts
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("bad time token {t:?}"))?;
            let beat_type: u32 = parts
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("bad time token {t:?}"))?;
            if beats == 0 || beat_type == 0 {
                return Err(format!("bad time token {t:?}"));
            }
            if !seen_event && measure == 0 {
                // Effectively leading (headerless stream): the onset
                // grid follows it.
                time = TimeSig { beats, beat_type };
                barlen = beats as f64 * 4.0 / beat_type.max(1) as f64;
            } else {
                attr_for(&mut measure_attrs, measure + 1).time =
                    Some(TimeSig { beats, beat_type });
            }
            i += 1;
            continue;
        }
        if let Some(short) = t.strip_prefix("dyn-") {
            match short {
                "pp" | "p" | "mp" | "mf" | "f" | "ff" | "sfz" => {
                    attr_for(&mut measure_attrs, measure + 1).dynamic =
                        Some(short.to_string());
                }
                _ => return Err(format!("unexpected token {t:?} in measure {measure}")),
            }
            i += 1;
            continue;
        }
        if let Some(short) = t.strip_prefix("hairpin-") {
            match short {
                "cresc-start" => {
                    attr_for(&mut measure_attrs, measure + 1).hairpin =
                        Some("cresc".to_string())
                }
                "dim-start" => {
                    attr_for(&mut measure_attrs, measure + 1).hairpin =
                        Some("dim".to_string())
                }
                "stop" => {
                    attr_for(&mut measure_attrs, measure + 1).hairpin =
                        Some("stop".to_string())
                }
                _ => return Err(format!("unexpected token {t:?} in measure {measure}")),
            }
            i += 1;
            continue;
        }
        if let Some(rest) = t.strip_prefix("voice-") {
            let n: usize = rest.parse().map_err(|_| format!("bad voice token {t:?}"))?;
            if n == 0 {
                return Err(format!("bad voice token {t:?}"));
            }
            while voices.len() < n {
                voices.push(0.0);
                last_onset.push(None);
                spans.push(Vec::new());
            }
            seg = n - 1;
            i += 1;
            continue;
        }
        if let Some(rest) = t.strip_prefix("tuplet-start-") {
            let mut parts = rest.split('-');
            let a: u32 = parts
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("bad tuplet token {t:?}"))?;
            let n: u32 = parts
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("bad tuplet token {t:?}"))?;
            if a == 0 || n == 0 {
                return Err(format!("bad tuplet token {t:?}"));
            }
            pending_spans.push((a, n));
            i += 1;
            continue;
        }
        if t == "tuplet-stop" {
            // seg is always valid (grown with voices); only an empty
            // stack errors — no indexing panic possible.
            spans[seg]
                .pop()
                .ok_or_else(|| format!("unmatched tuplet-stop in measure {measure}"))?;
            i += 1;
            continue;
        }
        // Note or rest event: [grace] [chord] (dur | rest [dur]) ...
        let mut chord = false;
        if t == "chord" {
            if grace_pending {
                return Err(format!("grace without duration in measure {measure}"));
            }
            chord = true;
            i += 1;
        }
        seen_event = true;
        // Event-level grace: the `grace` marker is consumed by the
        // duration token (chord tones inherit the run's grace).
        let is_grace = if chord {
            grace_run
        } else {
            let g = grace_pending;
            grace_pending = false;
            grace_run = g;
            g
        };
        let t = toks
            .get(i)
            .ok_or_else(|| format!("dangling chord marker in measure {measure}"))?;
        let (pitched, written) = if *t == "rest" {
            i += 1;
            (false, 0.0)
        } else if let Some(w) = written_dur(t) {
            i += 1;
            (true, w)
        } else if *t == "bar-rest" {
            if is_grace {
                return Err(format!("grace on bar-rest in measure {measure}"));
            }
            // Whole-bar rest: emit with full-bar duration (unscored),
            // jump this voice to the bar end.
            let onset = measure as f64 * barlen + voices[seg];
            push_rest(
                &mut notes,
                onset,
                barlen,
                seg as u32 + 1,
                &NoteMods::default(),
                false,
            );
            voices[seg] = barlen;
            last_onset[seg] = None;
            spans[seg].extend(pending_spans.drain(..));
            i += 1;
            continue;
        } else {
            return Err(format!("unexpected token {t:?} in measure {measure}"));
        };
        // Step + octave follow the duration token (alter, dots, ties
        // after — see below).
        let (step, octave) = if pitched {
            let s = toks
                .get(i)
                .ok_or_else(|| format!("missing step in measure {measure}"))?;
            let step = s
                .chars()
                .next()
                .ok_or_else(|| format!("bad step token {s:?} in measure {measure}"))?;
            if !"ABCDEFG".contains(step) {
                return Err(format!("bad step token {s:?} in measure {measure}"));
            }
            i += 1;
            let o = toks
                .get(i)
                .ok_or_else(|| format!("missing octave in measure {measure}"))?;
            let octave: i32 = o
                .parse()
                .map_err(|_| format!("bad octave token {o:?} in measure {measure}"))?;
            i += 1;
            (Some(step), Some(octave))
        } else {
            (None, None)
        };
        // Encoder order: step octave [alter] [dots] [slur-start]
        // [slur-stop] [modifiers] [tie-start] [tie-stop]; plain rests:
        // `rest` duration [dots] [modifiers]. The loops below accept any
        // interleave a generator might emit (the grammar mask agrees).
        let mut alter = 0i32;
        if pitched {
            // Alter tokens (`a-1`, `a1`, …) share their leading `a`
            // with the `accent` modifier: only consume the token here
            // when the suffix parses as an accidental offset, so a
            // following modifier falls through to the tail loop.
            if let Some(a) = toks
                .get(i)
                .filter(|s| s.starts_with('a') && s[1..].parse::<i32>().is_ok())
            {
                alter = a[1..]
                    .parse()
                    .map_err(|_| format!("bad alter token {a:?} in measure {measure}"))?;
                i += 1;
            }
        }
        let mut dots = 0u32;
        while toks.get(i) == Some(&"dot") {
            dots += 1;
            i += 1;
        }
        let mut dur_written = written * (2.0 - 2f64.powi(-(dots as i32)));
        // Slur/modifier marks and tie marks in any interleave (encoder
        // order is slurs, modifiers, then ties; the loop agrees with
        // the grammar mask, which admits the same set).
        let mut mods = NoteMods::default();
        let mut tie_start = false;
        let mut tie_stop = false;
        take_tail(&toks, &mut i, &mut mods, &mut tie_start, &mut tie_stop);
        if !pitched {
            // Plain-rest duration token follows the `rest` marker.
            let d = toks
                .get(i)
                .ok_or_else(|| format!("missing rest duration in measure {measure}"))?;
            dur_written = written_dur(d)
                .ok_or_else(|| format!("bad rest duration {d:?} in measure {measure}"))?;
            i += 1;
            while toks.get(i) == Some(&"dot") {
                dots += 1;
                i += 1;
            }
            dur_written *= 2.0 - 2f64.powi(-(dots as i32));
            // Rests take no ties (a tie mark here ends the tail and
            // errors as an unexpected bound token, like the encoder).
            take_tail_rest(&toks, &mut i, &mut mods);
        }
        // Pending spans open on this event's segment (`voice-N`
        // precedes it in the stream, so `seg` is already current).
        spans[seg].extend(pending_spans.drain(..));
        let scale = scale_of(&spans, seg);
        let performed = dur_written * scale;
        let base = measure as f64 * barlen;
        if chord {
            let onset = last_onset[seg]
                .ok_or_else(|| format!("dangling chord marker in measure {measure}"))?;
            let pitch = format!(
                "{}{}{}",
                step.ok_or_else(|| format!("chord on rest in measure {measure}"))?,
                alter_suffix(alter)?,
                octave.unwrap_or(4)
            );
            let key = (step.unwrap(), octave.unwrap_or(4), alter);
            if merge_ties {
                // Merged path drops grace (the reference adapter has
                // none); the onset is still recorded for chord runs.
                if !is_grace {
                    if let Some(note) = tie_note(
                        &mut pending_tie,
                        key,
                        pitch,
                        onset,
                        performed,
                        seg as u32 + 1,
                        seg,
                        tie_start,
                        tie_stop,
                        &mods,
                    ) {
                        notes.push(note);
                    }
                }
            } else {
                let mut note = mk_note(pitch, onset, performed, seg as u32 + 1, &mods);
                note.tie_start = tie_start;
                note.tie_stop = tie_stop;
                if is_grace {
                    note.grace = true;
                    note.dur = dur_written;
                }
                notes.push(note);
            }
            // Chord tones never advance the cursor.
        } else {
            let onset = base + voices[seg];
            if pitched {
                let pitch = format!(
                    "{}{}{}",
                    step.unwrap(),
                    alter_suffix(alter)?,
                    octave.unwrap_or(4)
                );
                let key = (step.unwrap(), octave.unwrap_or(4), alter);
                if merge_ties {
                    // Merged path drops grace (the reference adapter
                    // has none); the onset is recorded like the
                    // scorer so following chord tones still resolve.
                    if is_grace {
                        last_onset[seg] = Some(onset);
                    } else {
                        if let Some(note) = tie_note(
                            &mut pending_tie,
                            key,
                            pitch,
                            onset,
                            performed,
                            seg as u32 + 1,
                            seg,
                            tie_start,
                            tie_stop,
                            &mods,
                        ) {
                            notes.push(note);
                        }
                        last_onset[seg] = Some(onset);
                    }
                } else {
                    let mut note = mk_note(pitch, onset, performed, seg as u32 + 1, &mods);
                    note.tie_start = tie_start;
                    note.tie_stop = tie_stop;
                    if is_grace {
                        note.grace = true;
                        note.dur = dur_written;
                    }
                    notes.push(note);
                    last_onset[seg] = Some(onset);
                }
            } else if merge_ties && is_grace {
                // Dropped like pitched grace; the scorer records the
                // onset all the same.
                last_onset[seg] = Some(onset);
            } else {
                push_rest(
                    &mut notes,
                    onset,
                    if is_grace { dur_written } else { performed },
                    seg as u32 + 1,
                    &mods,
                    is_grace,
                );
                last_onset[seg] = if is_grace { Some(onset) } else { None };
            }
            // Grace notes steal no time: the cursor holds.
            if !is_grace {
                voices[seg] += performed;
            }
        }
    }
    if grace_pending {
        return Err("dangling grace marker at end of stream".to_string());
    }
    // Page-cut tie starts never see their stop: flush deterministically
    // (BTreeMap pitch order, oldest-first within each stack), keeping
    // each link's own voice (a hardcoded voice-1 flush used to shove
    // them into another voice's span and drop real notes on overlap).
    for ((s, o, a), stack) in pending_tie {
        for (fseg, onset, acc, fmods) in stack {
            let pitch = format!("{s}{}{}", alter_suffix(a)?, o);
            let note = mk_note(pitch, onset, acc, fseg as u32 + 1, &fmods);
            notes.push(note);
        }
    }
    measure_attrs.sort_by_key(|a| a.measure);
    Ok(ScoreInput {
        title: None,
        composer: None,
        divisions: SCORE_DIVISIONS,
        time: Some(time),
        key_fifths,
        parts: vec![PartInput {
            name: "decoded".to_string(),
            notes,
        }],
        notes: Vec::new(),
        part_name: None,
        measure_attrs,
    })
}

#[cfg(test)]
mod tests {
    use super::decode_bekern;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn roundtrip_ds0001_page_one_is_note_exact() {
        // The load-bearing invariant: encode->decode inverts exactly on
        // real multi-voice piano texture (chords, ties, bar rests,
        // 6 voice segments). Pitched notes compared as (pitch,
        // onset, dur) quantized to ticks; voice numbering differs by
        // design (appearance order vs MusicXML numbers) and rests are
        // unscored, so neither is compared.
        use super::super::bekern::to_bekern;
        use super::super::musicxml_import::{
            bekern_events, page_breaks, score_inputs_by_page, staff_headers,
        };
        let xml =
            std::fs::read_to_string("../../data-gen/score-corpus/big500-renders/ds0001.musicxml")
                .expect("corpus fixture ds0001.musicxml");
        let pages = score_inputs_by_page(&xml).unwrap();
        let gt = &pages[0].input;
        let breaks = page_breaks(&xml);
        let start = breaks[0].1;
        let end = breaks.get(1).map(|(_, m)| m - 1).unwrap_or(u32::MAX);
        let events: Vec<_> = bekern_events(&xml)
            .unwrap()
            .into_iter()
            .filter(|e| e.measure >= start && e.measure <= end)
            .collect();
        let headers = staff_headers(&xml).unwrap();
        let (toks, _) = to_bekern(&events, &headers).unwrap();
        let hyp = decode_bekern(&toks.join(" ")).unwrap();
        fn pitched(input: &super::super::ScoreInput) -> Vec<(String, i64, i64)> {
            let mut out = Vec::new();
            for part in &input.parts {
                for n in &part.notes {
                    if n.rest || n.pitch.is_none() {
                        continue;
                    }
                    out.push((
                        n.pitch.clone().unwrap(),
                        (n.onset * 480.0).round() as i64,
                        (n.dur * 480.0).round() as i64,
                    ));
                }
            }
            out.sort();
            out
        }
        let g = pitched(gt);
        let h = pitched(&hyp);
        assert_eq!(g.len(), h.len(), "note count survives the round trip");
        assert_eq!(g, h, "every pitched note survives the round trip");
    }

    #[test]
    fn roundtrip_ds0409_cross_bar_tuplets_are_note_exact() {
        // Regression: to_bekern used to emit tuplet spans crossing
        // `bar`, which decode's per-measure reset could not invert
        // (341/2529 corpus pages failed with "unmatched tuplet-stop"
        // or an index panic). Spans are now split at barlines on encode
        // and tracked per segment on decode: every ds0409 page (the
        // cross-bar case) must round-trip exactly.
        use super::super::bekern::to_bekern;
        use super::super::musicxml_import::{
            bekern_events, page_breaks, score_inputs_by_page, staff_headers,
        };
        let xml =
            std::fs::read_to_string("../../data-gen/score-corpus/big500-renders/ds0409.musicxml")
                .expect("corpus fixture ds0409.musicxml");
        let pages = score_inputs_by_page(&xml).unwrap();
        let breaks = page_breaks(&xml);
        assert_eq!(pages.len(), breaks.len(), "pages align with breaks");
        let headers = staff_headers(&xml).unwrap();
        let all = bekern_events(&xml).unwrap();
        // Tie chains severed by pagination cannot round-trip exactly:
        // the whole-score native path merges the chain, per-page tokens
        // sever it (decode flushes/partials deterministically). A page
        // is tie-clean iff every in-page tie_start's chain ends in-page
        // and every in-page tie_stop's chain starts in-page.
        fn tie_clean(
            all: &[super::super::musicxml_import::BekernNote],
            start: u32,
            end: u32,
        ) -> bool {
            let key = |e: &super::super::musicxml_import::BekernNote| {
                e.pitch.map(|(s, o, a)| (e.staff, e.voice, s, o, a))
            };
            let in_page = |m: u32| m >= start && m <= end;
            for (i, e) in all.iter().enumerate() {
                if !in_page(e.measure) {
                    continue;
                }
                let k = match key(e) {
                    Some(k) => k,
                    None => continue,
                };
                if e.tie_start {
                    // Walk the chain forward; every link must be in-page.
                    let mut j = i;
                    loop {
                        let nxt = all[j + 1..]
                            .iter()
                            .position(|o| key(o) == Some(k))
                            .map(|p| j + 1 + p);
                        match nxt {
                            Some(n) if in_page(all[n].measure) => {
                                if !all[n].tie_start {
                                    break;
                                }
                                j = n;
                            }
                            _ => return false,
                        }
                    }
                }
                if e.tie_stop {
                    // Walk the chain backward; every link must be in-page.
                    let mut j = i;
                    loop {
                        let prv = all[..j].iter().rposition(|o| key(o) == Some(k));
                        match prv {
                            Some(n) if in_page(all[n].measure) => {
                                if !all[n].tie_stop {
                                    break;
                                }
                                j = n;
                            }
                            _ => return false,
                        }
                    }
                }
            }
            true
        }
        // (pitch, global-measure-0idx, onset-in-bar, dur): measure is
        // comparable across sides when bar counting is right; a bar bug
        // scatters diffs everywhere, a local cause clusters them.
        fn pitched(
            input: &super::super::ScoreInput,
            barlen_ticks: i64,
            base_ticks: i64,
        ) -> Vec<(String, i64, i64, i64)> {
            let mut out = Vec::new();
            for part in &input.parts {
                for n in &part.notes {
                    if n.rest || n.pitch.is_none() {
                        continue;
                    }
                    // Both spellings are input-legal (README) and the
                    // scorer is MIDI-based; unify decode's "x" with the
                    // import path's "##" before comparing strings.
                    let abs = (n.onset * 480.0).round() as i64 + base_ticks;
                    out.push((
                        n.pitch.clone().unwrap().replace('x', "##"),
                        abs / barlen_ticks,
                        abs % barlen_ticks,
                        (n.dur * 480.0).round() as i64,
                    ));
                }
            }
            out.sort();
            out
        }
        let mut exact_pages = 0u32;
        for (pi, page) in pages.iter().enumerate() {
            let start = breaks[pi].1;
            let end = breaks.get(pi + 1).map(|(_, m)| m - 1).unwrap_or(u32::MAX);
            let events: Vec<_> = all
                .iter()
                .filter(|e| e.measure >= start && e.measure <= end)
                .collect();
            // to_bekern takes &[BekernNote]; re-collect owned refs.
            let owned: Vec<_> = events.into_iter().cloned().collect();
            let (toks, _) = to_bekern(&owned, &headers).unwrap();
            let hyp = decode_bekern(&toks.join(" "))
                .unwrap_or_else(|e| panic!("ds0409 page {pi} decodes: {e}"));
            if !tie_clean(&all, start, end) {
                // Severed tie chains: decode parses (pinned above) but
                // whole-score native merging cannot match per-page
                // tokens by construction. Exactness is pinned on
                // tie-clean pages below + the ds0001 page test.
                continue;
            }
            // Page 1+ starts mid-score: GT onsets are absolute beats
            // while decode counts measures from the page start, so the
            // comparison below is onset-in-bar (the strict scorer's own
            // notion) + duration.
            let barlen_ticks = {
                let t = page.input.time.clone().unwrap_or(super::super::TimeSig {
                    beats: 4,
                    beat_type: 4,
                });
                (t.beats as f64 * 4.0 / t.beat_type.max(1) as f64 * 480.0).round() as i64
            };
            let want = pitched(&page.input, barlen_ticks, 0);
            let got = pitched(&hyp, barlen_ticks, (start as i64 - 1) * barlen_ticks);
            if want != got {
                // Per-measure diff histogram (global 1-indexed
                // measures): clusters = local cause, scatter = bar bug.
                use std::collections::BTreeMap;
                let mut wmap: BTreeMap<i64, Vec<_>> = BTreeMap::new();
                for n in want.iter() {
                    wmap.entry(n.1 + 1).or_default().push(n.clone());
                }
                let mut gmap: BTreeMap<i64, Vec<_>> = BTreeMap::new();
                for n in got.iter() {
                    gmap.entry(n.1 + 1).or_default().push(n.clone());
                }
                for (m, w) in wmap.iter() {
                    let g = gmap.get(m).cloned().unwrap_or_default();
                    let (mut w, mut g) = (w.clone(), g);
                    w.sort();
                    g.sort();
                    if w != g {
                        eprintln!(
                            "page {pi} measure {m}: want {} notes, got {}",
                            w.len(),
                            g.len()
                        );
                    }
                }
            }
            assert_eq!(want, got, "ds0409 page {pi} survives the round trip");
            exact_pages += 1;
        }
        assert!(
            exact_pages > 0,
            "at least one ds0409 page must round-trip exactly"
        );
    }

    #[test]
    fn roundtrip_ds0037_nested_tuplets_are_note_exact() {
        // Diagnostic for nested tuplet brackets sharing one note: ds0037
        // measure 27 opens brackets 1+2 on one A and closes both on one
        // Bb (last-wins keeps number 2). If the dropped number-1 pair is
        // truly redundant, every tie-clean page round-trips exactly; if
        // not, the diff histogram names the damaged measures.
        use super::super::bekern::to_bekern;
        use super::super::musicxml_import::{
            bekern_events, page_breaks, score_inputs_by_page, staff_headers,
        };
        let xml =
            std::fs::read_to_string("../../data-gen/score-corpus/big500-renders/ds0037.musicxml")
                .expect("corpus fixture ds0037.musicxml");
        let pages = score_inputs_by_page(&xml).unwrap();
        let breaks = page_breaks(&xml);
        assert_eq!(pages.len(), breaks.len(), "pages align with breaks");
        let headers = staff_headers(&xml).unwrap();
        let all = bekern_events(&xml).unwrap();
        // Tie chains severed by pagination cannot round-trip exactly:
        // the whole-score native path merges the chain, per-page tokens
        // sever it (decode flushes/partials deterministically). A page
        // is tie-clean iff every in-page tie_start's chain ends in-page
        // and every in-page tie_stop's chain starts in-page.
        fn tie_clean(
            all: &[super::super::musicxml_import::BekernNote],
            start: u32,
            end: u32,
        ) -> bool {
            let key = |e: &super::super::musicxml_import::BekernNote| {
                e.pitch.map(|(s, o, a)| (e.staff, e.voice, s, o, a))
            };
            let in_page = |m: u32| m >= start && m <= end;
            for (i, e) in all.iter().enumerate() {
                if !in_page(e.measure) {
                    continue;
                }
                let k = match key(e) {
                    Some(k) => k,
                    None => continue,
                };
                if e.tie_start {
                    // Walk the chain forward; every link must be in-page.
                    let mut j = i;
                    loop {
                        let nxt = all[j + 1..]
                            .iter()
                            .position(|o| key(o) == Some(k))
                            .map(|p| j + 1 + p);
                        match nxt {
                            Some(n) if in_page(all[n].measure) => {
                                if !all[n].tie_start {
                                    break;
                                }
                                j = n;
                            }
                            _ => return false,
                        }
                    }
                }
                if e.tie_stop {
                    // Walk the chain backward; every link must be in-page.
                    let mut j = i;
                    loop {
                        let prv = all[..j].iter().rposition(|o| key(o) == Some(k));
                        match prv {
                            Some(n) if in_page(all[n].measure) => {
                                if !all[n].tie_stop {
                                    break;
                                }
                                j = n;
                            }
                            _ => return false,
                        }
                    }
                }
            }
            true
        }
        // (pitch, global-measure-0idx, onset-in-bar, dur): measure is
        // comparable across sides when bar counting is right; a bar bug
        // scatters diffs everywhere, a local cause clusters them.
        fn pitched(
            input: &super::super::ScoreInput,
            barlen_ticks: i64,
            base_ticks: i64,
        ) -> Vec<(String, i64, i64, i64)> {
            let mut out = Vec::new();
            for part in &input.parts {
                for n in &part.notes {
                    if n.rest || n.pitch.is_none() {
                        continue;
                    }
                    // Both spellings are input-legal (README) and the
                    // scorer is MIDI-based; unify decode's "x" with the
                    // import path's "##" before comparing strings.
                    let abs = (n.onset * 480.0).round() as i64 + base_ticks;
                    out.push((
                        n.pitch.clone().unwrap().replace('x', "##"),
                        abs / barlen_ticks,
                        abs % barlen_ticks,
                        (n.dur * 480.0).round() as i64,
                    ));
                }
            }
            out.sort();
            out
        }
        let mut exact_pages = 0u32;
        for (pi, page) in pages.iter().enumerate() {
            let start = breaks[pi].1;
            let end = breaks.get(pi + 1).map(|(_, m)| m - 1).unwrap_or(u32::MAX);
            let events: Vec<_> = all
                .iter()
                .filter(|e| e.measure >= start && e.measure <= end)
                .collect();
            // to_bekern takes &[BekernNote]; re-collect owned refs.
            let owned: Vec<_> = events.into_iter().cloned().collect();
            let (toks, _) = to_bekern(&owned, &headers).unwrap();
            let hyp = decode_bekern(&toks.join(" "))
                .unwrap_or_else(|e| panic!("ds0037 page {pi} decodes: {e}"));
            if !tie_clean(&all, start, end) {
                // Severed tie chains: decode parses (pinned above) but
                // whole-score native merging cannot match per-page
                // tokens by construction. Exactness is pinned on
                // tie-clean pages below + the ds0001 page test.
                continue;
            }
            // Page 1+ starts mid-score: GT onsets are absolute beats
            // while decode counts measures from the page start, so the
            // comparison below is onset-in-bar (the strict scorer's own
            // notion) + duration.
            let barlen_ticks = {
                let t = page.input.time.clone().unwrap_or(super::super::TimeSig {
                    beats: 4,
                    beat_type: 4,
                });
                (t.beats as f64 * 4.0 / t.beat_type.max(1) as f64 * 480.0).round() as i64
            };
            let want = pitched(&page.input, barlen_ticks, 0);
            let got = pitched(&hyp, barlen_ticks, (start as i64 - 1) * barlen_ticks);
            if want != got {
                // Per-measure diff histogram (global 1-indexed
                // measures): clusters = local cause, scatter = bar bug.
                use std::collections::BTreeMap;
                let mut wmap: BTreeMap<i64, Vec<_>> = BTreeMap::new();
                for n in want.iter() {
                    wmap.entry(n.1 + 1).or_default().push(n.clone());
                }
                let mut gmap: BTreeMap<i64, Vec<_>> = BTreeMap::new();
                for n in got.iter() {
                    gmap.entry(n.1 + 1).or_default().push(n.clone());
                }
                for (m, w) in wmap.iter() {
                    let g = gmap.get(m).cloned().unwrap_or_default();
                    let (mut w, mut g) = (w.clone(), g);
                    w.sort();
                    g.sort();
                    if w != g {
                        eprintln!(
                            "page {pi} measure {m}: want {} notes, got {}",
                            w.len(),
                            g.len()
                        );
                        // Multiset diff, capped: the actual damaged notes.
                        let mut gw = g.clone();
                        let mut missing = Vec::new();
                        for n in w.iter() {
                            if let Some(p) = gw.iter().position(|x| x == n) {
                                gw.remove(p);
                            } else {
                                missing.push(n.clone());
                            }
                        }
                        let mut ww = w.clone();
                        let mut extra = Vec::new();
                        for n in g.iter() {
                            if let Some(p) = ww.iter().position(|x| x == n) {
                                ww.remove(p);
                            } else {
                                extra.push(n.clone());
                            }
                        }
                        for n in missing.iter().take(8) {
                            eprintln!("  want-not-got {n:?}");
                        }
                        for n in extra.iter().take(8) {
                            eprintln!("  got-not-want {n:?}");
                        }
                    }
                }
            }
            assert_eq!(want, got, "ds0037 page {pi} survives the round trip");
            exact_pages += 1;
        }
        assert!(
            exact_pages > 0,
            "at least one ds0037 page must round-trip exactly"
        );
    }

    #[test]
    #[ignore = "campaign tool, not a unit test: run explicitly with -- --ignored for single-page diffs"]
    fn tmp_page_diag() {
        // TEMPORARY: single-page round-trip diff. SCORE_XML path +
        // PAGE_IDX (0-based) via env. Prints multiset diffs.
        use super::super::bekern::to_bekern;
        use super::super::musicxml_import::{
            bekern_events, page_breaks, score_inputs_by_page, staff_headers,
        };
        let (Ok(path), Ok(pi_str)) = (std::env::var("SCORE_XML"), std::env::var("PAGE_IDX")) else {
            eprintln!("tmp_page_diag: set SCORE_XML + PAGE_IDX; skipping");
            return;
        };
        let pi: usize = pi_str.parse().unwrap();
        let xml = std::fs::read_to_string(&path).expect("read score");
        let pages = score_inputs_by_page(&xml).unwrap();
        let breaks = page_breaks(&xml);
        let headers = staff_headers(&xml).unwrap();
        let all = bekern_events(&xml).unwrap();
        let page = &pages[pi];
        let start = breaks[pi].1;
        let end = breaks.get(pi + 1).map(|(_, m)| m - 1).unwrap_or(u32::MAX);
        eprintln!("page {pi} measures {start}..={end}");
        let owned: Vec<_> = all
            .iter()
            .filter(|e| e.measure >= start && e.measure <= end)
            .cloned()
            .collect();
        if let Ok(em) = std::env::var("DIAG_EV") {
            let em: u32 = em.parse().unwrap();
            for e in owned.iter().filter(|e| e.measure == em) {
                eprintln!(
                    "EV m{} staff{} v{} on{:.4} dur{:.6} {:?} ratio{:?} start{:?} stop{:?} rest{}",
                    e.measure,
                    e.staff,
                    e.voice,
                    e.onset,
                    e.dur_beats,
                    e.pitch,
                    e.tuplet_ratio,
                    e.tuplet_start,
                    e.tuplet_stop,
                    e.pitch.is_none()
                );
            }
        }
        let (toks, _) = to_bekern(&owned, &headers).unwrap();
        let hyp = decode_bekern(&toks.join(" ")).unwrap();
        fn pitched(
            input: &super::super::ScoreInput,
            barlen_ticks: i64,
            base_ticks: i64,
        ) -> Vec<(String, i64, i64, i64)> {
            let mut out = Vec::new();
            for part in &input.parts {
                for n in &part.notes {
                    if n.rest || n.pitch.is_none() {
                        continue;
                    }
                    let abs = (n.onset * 480.0).round() as i64 + base_ticks;
                    out.push((
                        n.pitch.clone().unwrap().replace('x', "##"),
                        abs / barlen_ticks,
                        abs % barlen_ticks,
                        (n.dur * 480.0).round() as i64,
                    ));
                }
            }
            out.sort();
            out
        }
        let barlen_ticks = {
            let t = page.input.time.clone().unwrap_or(super::super::TimeSig {
                beats: 4,
                beat_type: 4,
            });
            (t.beats as f64 * 4.0 / t.beat_type.max(1) as f64 * 480.0).round() as i64
        };
        let want = pitched(&page.input, barlen_ticks, 0);
        let got = pitched(&hyp, barlen_ticks, (start as i64 - 1) * barlen_ticks);
        use std::collections::BTreeMap;
        let mut wmap: BTreeMap<i64, Vec<_>> = BTreeMap::new();
        for n in want.iter() {
            wmap.entry(n.1 + 1).or_default().push(n.clone());
        }
        let mut gmap: BTreeMap<i64, Vec<_>> = BTreeMap::new();
        for n in got.iter() {
            gmap.entry(n.1 + 1).or_default().push(n.clone());
        }
        if let Ok(dm) = std::env::var("DIAG_M") {
            let dm: i64 = dm.parse().unwrap();
            if let Some(w) = wmap.get(&dm) {
                let mut w = w.clone();
                w.sort();
                eprintln!("ALL-WANT m{dm}: {w:?}");
            }
            if let Some(g) = gmap.get(&dm) {
                let mut g = g.clone();
                g.sort();
                eprintln!("ALL-GOT m{dm}: {g:?}");
            }
        }
        for (m, w) in wmap.iter() {
            let g = gmap.get(m).cloned().unwrap_or_default();
            let (mut w, mut g) = (w.clone(), g);
            w.sort();
            g.sort();
            if w != g {
                eprintln!("measure {m}: want {} got {}", w.len(), g.len());
                let mut gw = g.clone();
                let mut missing = Vec::new();
                for n in w.iter() {
                    if let Some(p) = gw.iter().position(|x| x == n) {
                        gw.remove(p);
                    } else {
                        missing.push(n.clone());
                    }
                }
                let mut ww = w.clone();
                let mut extra = Vec::new();
                for n in g.iter() {
                    if let Some(p) = ww.iter().position(|x| x == n) {
                        ww.remove(p);
                    } else {
                        extra.push(n.clone());
                    }
                }
                for n in missing.iter().take(6) {
                    eprintln!("  want-not-got {n:?}");
                }
                for n in extra.iter().take(6) {
                    eprintln!("  got-not-want {n:?}");
                }
            }
        }
    }

    #[test]
    #[ignore = "campaign tool, not a unit test: run explicitly with -- --ignored for the full-corpus gate"]
    fn tmp_corpus_roundtrip_audit() {
        // TEMPORARY baseline: per-score round-trip exactness over the big500
        // corpus (tie-severed pages skipped, like the ds0409 harness).
        // Reports counts only; delete after scoping the gap-fill/stale-span
        // fixes.
        use super::super::bekern::to_bekern;
        use super::super::musicxml_import::{
            bekern_events, page_breaks, score_inputs_by_page, staff_headers,
        };
        fn tie_clean(
            all: &[super::super::musicxml_import::BekernNote],
            start: u32,
            end: u32,
        ) -> bool {
            let key = |e: &super::super::musicxml_import::BekernNote| {
                e.pitch.map(|(s, o, a)| (e.staff, e.voice, s, o, a))
            };
            let in_page = |m: u32| m >= start && m <= end;
            for (i, e) in all.iter().enumerate() {
                if !in_page(e.measure) {
                    continue;
                }
                let k = match key(e) {
                    Some(k) => k,
                    None => continue,
                };
                if e.tie_start {
                    let mut j = i;
                    loop {
                        let nxt = all[j + 1..]
                            .iter()
                            .position(|o| key(o) == Some(k))
                            .map(|p| j + 1 + p);
                        match nxt {
                            Some(n) if in_page(all[n].measure) => {
                                if !all[n].tie_start {
                                    break;
                                }
                                j = n;
                            }
                            _ => return false,
                        }
                    }
                }
                if e.tie_stop {
                    let mut j = i;
                    loop {
                        let prv = all[..j].iter().rposition(|o| key(o) == Some(k));
                        match prv {
                            Some(n) if in_page(all[n].measure) => {
                                if !all[n].tie_stop {
                                    break;
                                }
                                j = n;
                            }
                            _ => return false,
                        }
                    }
                }
            }
            true
        }
        fn pitched(
            input: &super::super::ScoreInput,
            barlen_ticks: i64,
            base_ticks: i64,
        ) -> Vec<(String, i64, i64, i64)> {
            let mut out = Vec::new();
            for part in &input.parts {
                for n in &part.notes {
                    if n.rest || n.pitch.is_none() {
                        continue;
                    }
                    let abs = (n.onset * 480.0).round() as i64 + base_ticks;
                    out.push((
                        n.pitch.clone().unwrap().replace('x', "##"),
                        abs / barlen_ticks,
                        abs % barlen_ticks,
                        (n.dur * 480.0).round() as i64,
                    ));
                }
            }
            out.sort();
            out
        }
        let mut entries: Vec<_> = std::fs::read_dir("../../data-gen/score-corpus/big500-renders")
            .expect("renders dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".musicxml"))
            .collect();
        entries.sort();
        let (mut pg_exact, mut pg_skip, mut pg_fail) = (0u32, 0u32, 0u32);
        let mut bad_scores: Vec<String> = Vec::new();
        for name in &entries {
            let path = format!("../../data-gen/score-corpus/big500-renders/{name}");
            let xml = std::fs::read_to_string(&path).expect("read score");
            let pages = match score_inputs_by_page(&xml) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{name}: score inputs error {e}");
                    continue;
                }
            };
            let breaks = page_breaks(&xml);
            let headers = staff_headers(&xml).unwrap();
            let all = bekern_events(&xml).unwrap();
            let mut score_bad = 0u32;
            for (pi, page) in pages.iter().enumerate() {
                let start = breaks[pi].1;
                let end = breaks.get(pi + 1).map(|(_, m)| m - 1).unwrap_or(u32::MAX);
                let owned: Vec<_> = all
                    .iter()
                    .filter(|e| e.measure >= start && e.measure <= end)
                    .cloned()
                    .collect();
                let (toks, _) = match to_bekern(&owned, &headers) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("{name} p{pi}: encode error {e}");
                        pg_fail += 1;
                        score_bad += 1;
                        continue;
                    }
                };
                let hyp = match decode_bekern(&toks.join(" ")) {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!("{name} p{pi}: decode error {e}");
                        pg_fail += 1;
                        score_bad += 1;
                        continue;
                    }
                };
                if !tie_clean(&all, start, end) {
                    pg_skip += 1;
                    continue;
                }
                let barlen_ticks = {
                    let t = page.input.time.clone().unwrap_or(super::super::TimeSig {
                        beats: 4,
                        beat_type: 4,
                    });
                    (t.beats as f64 * 4.0 / t.beat_type.max(1) as f64 * 480.0).round() as i64
                };
                let want = pitched(&page.input, barlen_ticks, 0);
                let got = pitched(&hyp, barlen_ticks, (start as i64 - 1) * barlen_ticks);
                if want == got {
                    pg_exact += 1;
                } else {
                    pg_fail += 1;
                    score_bad += 1;
                    // Mechanism classification: counts, pitch sets,
                    // onset-multiset, duration-multiset.
                    let mut w = want.clone();
                    let mut g = got.clone();
                    w.sort();
                    g.sort();
                    let w_pitch: Vec<_> = w.iter().map(|n| n.0.clone()).collect();
                    let mut g_pitch: Vec<_> = g.iter().map(|n| n.0.clone()).collect();
                    g_pitch.sort();
                    let mut w_pitch_s = w_pitch.clone();
                    w_pitch_s.sort();
                    let w_on: Vec<_> = w.iter().map(|n| (n.0.clone(), n.2)).collect();
                    let mut g_on: Vec<_> = g.iter().map(|n| (n.0.clone(), n.2)).collect();
                    let mut w_on_s = w_on.clone();
                    w_on_s.sort();
                    g_on.sort();
                    let w_du: Vec<_> = w.iter().map(|n| (n.0.clone(), n.3)).collect();
                    let mut g_du: Vec<_> = g.iter().map(|n| (n.0.clone(), n.3)).collect();
                    let mut w_du_s = w_du.clone();
                    w_du_s.sort();
                    g_du.sort();
                    let cat = if w.len() != g.len() {
                        "COUNT"
                    } else if w_pitch_s != g_pitch {
                        "PITCH"
                    } else if w_on_s != g_on {
                        "ONSET"
                    } else if w_du_s != g_du {
                        "DUR"
                    } else {
                        "MIX"
                    };
                    // For ONSET: is the shift uniform (single missed gap)
                    // or scattered? Distinct onset-deltas (positional
                    // pairing; uniform shifts reorder little).
                    let mut deltas = std::collections::BTreeSet::new();
                    if cat == "ONSET" {
                        for (a, b) in w.iter().zip(g.iter()) {
                            deltas.insert((a.1 * 100000 + a.2) - (b.1 * 100000 + b.2));
                        }
                    }
                    eprintln!(
                        "ROW {name} p{pi} {cat} Ndeltas={} w={} g={}",
                        deltas.len(),
                        w.len(),
                        g.len()
                    );
                }
            }
            if score_bad > 0 {
                bad_scores.push(format!("{name}:{score_bad}"));
            }
        }
        eprintln!("AUDIT pages exact={pg_exact} skipped-tie={pg_skip} failed={pg_fail}");
        eprintln!(
            "AUDIT bad scores ({}): {}",
            bad_scores.len(),
            bad_scores.join(" ")
        );
        assert!(pg_exact > 0, "audit ran over pages");
    }

    // Parallel tied chord tones (both members tied across the same
    // boundary): each chain merges at its own start. A single pending
    // slot would drop the first chain and split the second.
    const PARALLEL_TIES: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>4</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><notations><tied type="start"/></notations></note>
<note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><notations><tied type="start"/></notations></note>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><notations><tied type="stop"/></notations></note>
<note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><notations><tied type="stop"/></notations></note>
<note><rest/><duration>8</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn parallel_tied_chord_tones_merge_per_chain() {
        use super::super::bekern::to_bekern;
        use super::super::musicxml_import::{bekern_events, staff_headers};
        let headers = staff_headers(PARALLEL_TIES).unwrap();
        let events = bekern_events(PARALLEL_TIES).unwrap();
        let (toks, _) = to_bekern(&events, &headers).unwrap();
        let hyp = decode_bekern(&toks.join(" ")).unwrap();
        let mut got: Vec<(String, i64, i64)> = hyp.parts[0]
            .notes
            .iter()
            .filter(|n| !n.rest)
            .map(|n| {
                (
                    n.pitch.clone().unwrap(),
                    (n.onset * 480.0).round() as i64,
                    (n.dur * 480.0).round() as i64,
                )
            })
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![("C4".to_string(), 0, 960), ("E4".to_string(), 0, 960),]
        );
    }

    // ds0026 m51->m52: a triplet left open at the barline (its stop
    // printed in m51 on another staff) meets new notes carrying a
    // different ratio with no start marker. The old bracket ended; the
    // new notes belong to a new unmarked bracket.
    const STALE_REOPEN_FIXTURE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>12</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="start"/></notations></note>
<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>F</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>A</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>B</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>F</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>G</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
</measure>
<measure number="2">
<note><pitch><step>G</step><octave>4</octave></pitch><duration>12</duration><voice>1</voice><type>quarter</type><time-modification><actual-notes>2</actual-notes><normal-notes>1</normal-notes></time-modification></note>
<note><pitch><step>A</step><octave>4</octave></pitch><duration>12</duration><voice>1</voice><type>quarter</type></note>
<note><rest/><duration>24</duration><voice>1</voice><type>half</type></note>
</measure>
</part>
</score-partwise>"#;

    // Genuine nesting: an explicitly marked (2,1) run stays open
    // while a triplet of 16ths starts inside it after an event-less
    // silence (`<forward>`, so the cursor truly lags and the
    // sequential rule does not fire). Both levels must survive, and
    // the triplet 16ths must sound at a sixth of a beat (40 ticks).
    const NESTED_TUPLETS_FIXTURE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>12</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>A</step><octave>4</octave></pitch><duration>12</duration><voice>1</voice><type>half</type><time-modification><actual-notes>2</actual-notes><normal-notes>1</normal-notes></time-modification><notations><tuplet type="start"/></notations></note>
<forward><duration>12</duration><voice>1</voice></forward>
<note><pitch><step>B</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice><type>16th</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="start"/></notations></note>
<note><pitch><step>C</step><octave>5</octave></pitch><duration>2</duration><voice>1</voice><type>16th</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>D</step><octave>5</octave></pitch><duration>2</duration><voice>1</voice><type>16th</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="stop"/></notations></note>
</measure>
</part>
</score-partwise>"#;

    fn pitched_ticks(score: &super::super::ScoreInput) -> Vec<(String, i64, i64)> {
        let mut got: Vec<(String, i64, i64)> = score.parts[0]
            .notes
            .iter()
            .filter(|n| !n.rest)
            .map(|n| {
                (
                    n.pitch.clone().unwrap(),
                    (n.onset * 480.0).round() as i64,
                    (n.dur * 480.0).round() as i64,
                )
            })
            .collect();
        got.sort();
        got
    }

    #[test]
    fn ratio_change_across_barline_drops_stale_reopen() {
        use super::super::bekern::to_bekern;
        use super::super::musicxml_import::{bekern_events, staff_headers};
        let headers = staff_headers(STALE_REOPEN_FIXTURE).unwrap();
        let events = bekern_events(STALE_REOPEN_FIXTURE).unwrap();
        let (toks, _) = to_bekern(&events, &headers).unwrap();
        // The m1 triplet opens once; it must not reopen in m2 under the
        // (2,1) duplets. The duplet opens its own implicit span instead.
        assert_eq!(
            toks.iter().filter(|t| *t == "tuplet-start-3-2").count(),
            1,
            "stale (3,2) reattached in m2: {toks:?}"
        );
        assert_eq!(toks.iter().filter(|t| *t == "tuplet-start-2-1").count(), 1);
        // The duplet quarter is written as a half (2.0 beats) and sounds
        // half of that (1.0 = 480 ticks); the plain quarter that follows
        // sounds whole (480).
        let hyp = decode_bekern(&toks.join(" ")).unwrap();
        assert_eq!(
            pitched_ticks(&hyp),
            vec![
                ("A4".to_string(), 800, 160),
                ("A4".to_string(), 2400, 480),
                ("B4".to_string(), 960, 160),
                ("C4".to_string(), 0, 160),
                ("C5".to_string(), 1120, 160),
                ("D4".to_string(), 160, 160),
                ("D5".to_string(), 1280, 160),
                ("E4".to_string(), 320, 160),
                ("E5".to_string(), 1440, 160),
                ("F4".to_string(), 480, 160),
                ("F5".to_string(), 1600, 160),
                ("G4".to_string(), 640, 160),
                ("G4".to_string(), 1920, 480),
                ("G5".to_string(), 1760, 160),
            ]
        );
    }

    #[test]
    fn explicit_start_after_gap_nests_under_open_span() {
        use super::super::bekern::to_bekern;
        use super::super::musicxml_import::{bekern_events, staff_headers};
        let headers = staff_headers(NESTED_TUPLETS_FIXTURE).unwrap();
        let events = bekern_events(NESTED_TUPLETS_FIXTURE).unwrap();
        let (toks, _) = to_bekern(&events, &headers).unwrap();
        // The inner triplet starts after a real silence gap, so the
        // sequential rule does not fire: both levels stay open and the
        // triplet 16ths sound at a twelfth of a beat (40 ticks).
        assert_eq!(
            toks.iter().filter(|t| *t == "tuplet-start-2-1").count(),
            1,
            "tokens: {toks:?}"
        );
        assert_eq!(toks.iter().filter(|t| *t == "tuplet-start-3-2").count(), 1);
        let hyp = decode_bekern(&toks.join(" ")).unwrap();
        assert_eq!(
            pitched_ticks(&hyp),
            vec![
                ("A4".to_string(), 0, 480),
                ("B4".to_string(), 960, 40),
                ("C5".to_string(), 1040, 40),
                ("D5".to_string(), 1120, 40),
            ],
            "tokens: {toks:?}"
        );
    }

    // ds0226 m35: an unmarked (2,1) run ends exactly at beat 2.0 and
    // an explicitly marked (3,2) triplet starts there (sequential
    // brackets, not nesting). Nesting the triplet under the still-open
    // duplet triple-scales below and dust-fails the gap fill.
    const SEQUENTIAL_TUPLETS_FIXTURE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>12</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>A</step><octave>4</octave></pitch><duration>3</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>2</actual-notes><normal-notes>1</normal-notes></time-modification></note>
<note><pitch><step>B</step><octave>4</octave></pitch><duration>3</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>2</actual-notes><normal-notes>1</normal-notes></time-modification></note>
<note><pitch><step>C</step><octave>5</octave></pitch><duration>3</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>2</actual-notes><normal-notes>1</normal-notes></time-modification></note>
<note><pitch><step>D</step><octave>5</octave></pitch><duration>3</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>2</actual-notes><normal-notes>1</normal-notes></time-modification></note>
<note><pitch><step>E</step><octave>5</octave></pitch><duration>3</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>2</actual-notes><normal-notes>1</normal-notes></time-modification></note>
<note><pitch><step>F</step><octave>5</octave></pitch><duration>3</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>2</actual-notes><normal-notes>1</normal-notes></time-modification></note>
<note><pitch><step>G</step><octave>5</octave></pitch><duration>3</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>2</actual-notes><normal-notes>1</normal-notes></time-modification></note>
<note><pitch><step>A</step><octave>5</octave></pitch><duration>3</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>2</actual-notes><normal-notes>1</normal-notes></time-modification></note>
<note><pitch><step>B</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="start"/></notations></note>
<note><pitch><step>C</step><octave>6</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>D</step><octave>6</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="stop"/></notations></note>
<note><rest/><duration>12</duration><voice>1</voice><type>quarter</type></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn explicit_start_after_abutting_run_closes_stale_span() {
        use super::super::bekern::to_bekern;
        use super::super::musicxml_import::{bekern_events, staff_headers};
        let headers = staff_headers(SEQUENTIAL_TUPLETS_FIXTURE).unwrap();
        let events = bekern_events(SEQUENTIAL_TUPLETS_FIXTURE).unwrap();
        // Pre-fix this errors (nested scale-3 dust: untypeable 0.498).
        let (toks, _) = to_bekern(&events, &headers).unwrap();
        // The duplet opens once and closes before the triplet; the
        // triplet opens once at true scale.
        assert_eq!(toks.iter().filter(|t| *t == "tuplet-start-2-1").count(), 1);
        assert_eq!(toks.iter().filter(|t| *t == "tuplet-start-3-2").count(), 1);
        // Duplet eighths sound as 16ths (120 ticks); triplet eighths
        // sound at a true third (160 ticks).
        let hyp = decode_bekern(&toks.join(" ")).unwrap();
        assert_eq!(
            pitched_ticks(&hyp),
            vec![
                ("A4".to_string(), 0, 120),
                ("A5".to_string(), 840, 120),
                ("B4".to_string(), 120, 120),
                ("B5".to_string(), 960, 160),
                ("C5".to_string(), 240, 120),
                ("C6".to_string(), 1120, 160),
                ("D5".to_string(), 360, 120),
                ("D6".to_string(), 1280, 160),
                ("E5".to_string(), 480, 120),
                ("F5".to_string(), 600, 120),
                ("G5".to_string(), 720, 120),
            ]
        );
    }

    // Accepted S2 residual (arch-review-gap-encoding §5): an
    // explicitly starting triplet lands exactly on the unmarked duplet
    // gridline (abutting) and duplet members resume after it closes.
    // If the duplet genuinely contains the triplet (aligned nesting),
    // S2 flattens it today: the duplet closes, the triplet decodes at
    // top level, the resumed duplet re-opens implicitly. The §2 source
    // query (explicit start strictly inside another bracket's interval
    // on the same key) is the instrument that keeps this bounded; if
    // it ever finds genuine aligned nesting in the wild, this test
    // names the culprit mechanism and becomes the lookahead fix's
    // acceptance criterion.
    const S2_RESIDUAL_FIXTURE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>12</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>A</step><octave>4</octave></pitch><duration>12</duration><voice>1</voice><type>quarter</type><time-modification><actual-notes>2</actual-notes><normal-notes>1</normal-notes></time-modification></note>
<note><pitch><step>B</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="start"/></notations></note>
<note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="stop"/></notations></note>
<note><pitch><step>G</step><octave>4</octave></pitch><duration>12</duration><voice>1</voice><type>quarter</type><time-modification><actual-notes>2</actual-notes><normal-notes>1</normal-notes></time-modification></note>
<note><rest/><duration>12</duration><voice>1</voice><type>quarter</type></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn abutting_explicit_start_flattens_outer_span() {
        use super::super::bekern::to_bekern;
        use super::super::musicxml_import::{bekern_events, staff_headers};
        let headers = staff_headers(S2_RESIDUAL_FIXTURE).unwrap();
        let events = bekern_events(S2_RESIDUAL_FIXTURE).unwrap();
        let (toks, _) = to_bekern(&events, &headers).unwrap();
        // The duplet opens, closes at the abutting triplet start, and
        // re-opens implicitly for the resumed member: two 2-1 spans,
        // one top-level 3-2 triplet.
        assert_eq!(
            toks.iter().filter(|t| *t == "tuplet-start-2-1").count(),
            2,
            "tokens: {toks:?}"
        );
        assert_eq!(
            toks.iter().filter(|t| *t == "tuplet-start-3-2").count(),
            1,
            "tokens: {toks:?}"
        );
        let hyp = decode_bekern(&toks.join(" ")).unwrap();
        assert_eq!(
            pitched_ticks(&hyp),
            vec![
                ("A4".to_string(), 0, 480),
                ("B4".to_string(), 480, 160),
                ("C5".to_string(), 640, 160),
                ("D5".to_string(), 800, 160),
                ("G4".to_string(), 960, 480),
            ],
            "tokens: {toks:?}"
        );
    }

    #[test]
    fn unison_ties_across_voices_merge_independently() {
        // ds0462 m19: two voices tie the same pitch (unison C5) with
        // independent chains. Pitch-only keying lets the second
        // tie-start overwrite the first, emitting a [0..600] fragment
        // plus a literal [720..1200]; both chains must merge instead.
        let score = decode_bekern(
            "clef-G-2 key+0 time-4-4 voice-1 quarter C 5 dot tie-start \
             voice-2 quarter E 4 chord quarter C 5 tie-start \
             voice-2 16th C 5 tie-stop voice-1 quarter C 5 tie-stop bar",
        )
        .unwrap();
        let mut got: Vec<(String, i64, i64)> = score.parts[0]
            .notes
            .iter()
            .filter(|n| !n.rest)
            .map(|n| {
                (
                    n.pitch.clone().unwrap(),
                    (n.onset * 480.0).round() as i64,
                    (n.dur * 480.0).round() as i64,
                )
            })
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                ("C5".to_string(), 0, 600),
                ("C5".to_string(), 0, 1200),
                ("E4".to_string(), 0, 480),
            ]
        );
    }

    #[test]
    fn cross_bar_tie_merges_when_seg_churns() {
        // Same voice tied across a barline whose segment number changes
        // (voice entry order churn): with exactly one open chain the
        // stop falls back to it. Pin: pitch-only keying already merged
        // this; the stack must not regress it.
        let score = decode_bekern(
            "clef-G-2 key+0 time-4-4 voice-1 quarter C 5 tie-start bar \
             voice-2 half C 5 tie-stop bar",
        )
        .unwrap();
        let mut got: Vec<(String, i64, i64)> = score.parts[0]
            .notes
            .iter()
            .filter(|n| !n.rest)
            .map(|n| {
                (
                    n.pitch.clone().unwrap(),
                    (n.onset * 480.0).round() as i64,
                    (n.dur * 480.0).round() as i64,
                )
            })
            .collect();
        got.sort();
        assert_eq!(got, vec![("C5".to_string(), 0, 1440)]);
    }

    #[test]
    fn ambiguous_overlap_resolves_lifo_deterministically() {
        // Two open same-pitch chains, stop on a third segment: the
        // stack top wins, and repeated decodes agree. The surviving
        // chain flushes literally at end of stream.
        let toks = "clef-G-2 key+0 time-4-4 voice-1 quarter C 5 tie-start \
             voice-2 half C 5 tie-start voice-3 whole C 5 tie-stop bar";
        let oto = |s: &str| {
            let mut got: Vec<(String, i64, i64)> = decode_bekern(s).unwrap().parts[0]
                .notes
                .iter()
                .filter(|n| !n.rest)
                .map(|n| {
                    (
                        n.pitch.clone().unwrap(),
                        (n.onset * 480.0).round() as i64,
                        (n.dur * 480.0).round() as i64,
                    )
                })
                .collect();
            got.sort();
            got
        };
        let want = vec![("C5".to_string(), 0, 480), ("C5".to_string(), 0, 2880)];
        assert_eq!(oto(toks), want);
        assert_eq!(oto(toks), want);
    }

    #[test]
    fn triplet_decodes_to_performed_onsets() {
        // Three triplet eighths in the time of one quarter: performed
        // onsets 0, 1/3, 2/3 (not the written 0, 1/2, 1).
        let score = decode_bekern(
            "clef-G-2 key+0 time-4-4 voice-1 \
             tuplet-start-3-2 eighth C 4 eighth D 4 a-1 eighth E 4 tuplet-stop \
             quarter F 4 bar",
        )
        .unwrap();
        let notes = &score.parts[0].notes;
        assert_eq!(notes.len(), 4);
        let onsets: Vec<f64> = notes.iter().map(|n| n.onset).collect();
        assert!(approx(onsets[0], 0.0), "{onsets:?}");
        assert!(approx(onsets[1], 1.0 / 3.0), "{onsets:?}");
        assert!(approx(onsets[2], 2.0 / 3.0), "{onsets:?}");
        assert!(approx(onsets[3], 1.0), "{onsets:?}");
        assert!(approx(notes[1].dur, 1.0 / 3.0));
        assert_eq!(notes[0].pitch.as_deref(), Some("C4"));
        // Alter token sits before the octave in pitch spelling.
        assert_eq!(notes[1].pitch.as_deref(), Some("Db4"));
    }

    #[test]
    fn ties_merge_and_bar_rest_jumps_to_bar_end() {
        let score = decode_bekern(
            "clef-G-2 key+0 time-4-4 voice-1 \
             quarter C 4 tie-start quarter C 4 tie-stop \
             bar-rest voice-2 quarter D 4 bar",
        )
        .unwrap();
        let notes = &score.parts[0].notes;
        // Tie chain merges (2.0 beats). The bar rest fills voice 1; the
        // marker opens voice 2, so D starts at offset 0. Rests are kept
        // as rest notes.
        assert_eq!(notes.len(), 3);
        assert!(approx(notes[0].onset, 0.0));
        assert!(approx(notes[0].dur, 2.0));
        assert_eq!(notes[0].voice, 1);
        assert!(notes[1].rest);
        assert!(approx(notes[2].onset, 0.0));
        assert_eq!(notes[2].voice, 2);
        assert_eq!(notes[2].pitch.as_deref(), Some("D4"));
    }

    #[test]
    fn unmerged_decode_preserves_tie_links() {
        // Same stream, both semantics: merged collapses the chain
        // (note-list path), unmerged keeps every link with its flags
        // (notation path for the strict-xml oracle).
        use super::decode_bekern_unmerged;
        let toks = "clef-G-2 key+0 time-4-4 voice-1 \
                    quarter C 4 tie-start bar quarter C 4 tie-stop bar";
        let merged = decode_bekern(toks).unwrap();
        assert_eq!(merged.parts[0].notes.len(), 1);
        assert!(approx(merged.parts[0].notes[0].dur, 2.0));
        let unmerged = decode_bekern_unmerged(toks).unwrap();
        let notes = &unmerged.parts[0].notes;
        assert_eq!(notes.len(), 2);
        assert!(approx(notes[0].onset, 0.0));
        assert!(approx(notes[0].dur, 1.0));
        assert!(notes[0].tie_start && !notes[0].tie_stop);
        assert!(approx(notes[1].onset, 4.0));
        assert!(approx(notes[1].dur, 1.0));
        assert!(notes[1].tie_stop && !notes[1].tie_start);
    }

    #[test]
    fn unmerged_decode_keeps_grace_zero_width() {
        // A grace eighth steals no time: the following quarter keeps
        // the grace's onset, and the grace carries its display
        // duration. The merged path drops grace (its reference has
        // none) but still records the onset for chord runs.
        use super::decode_bekern_unmerged;
        let toks = "clef-G-2 key+0 time-4-4 voice-1 \
                    grace eighth E 5 quarter C 5 chord quarter G 4 bar";
        let unmerged = decode_bekern_unmerged(toks).unwrap();
        let notes = &unmerged.parts[0].notes;
        assert_eq!(notes.len(), 3);
        assert!(notes[0].grace);
        assert_eq!(notes[0].pitch.as_deref(), Some("E5"));
        assert!(approx(notes[0].dur, 0.5));
        assert!(approx(notes[1].onset, notes[0].onset));
        assert!(!notes[1].grace);
        // The chord tone inherits the root's non-grace run.
        assert!(!notes[2].grace);
        assert!(approx(notes[2].onset, notes[1].onset));
        let merged = decode_bekern(toks).unwrap();
        // Grace dropped; quarter + chord tone remain.
        assert_eq!(merged.parts[0].notes.len(), 2);
        assert!(!merged.parts[0].notes[0].grace);
    }

    /// Synthetic bekern+ score: non-dyadic divisions with a jittered
    /// triplet member (tick quantum), a grace note, slurs, a staccato,
    /// a rest fermata, dynamics, hairpins, a double bar, a final bar,
    /// and a mid-score key/time/clef change. Exercises the full
    /// MusicXML -> bekern+ -> decode -> MusicXML path.
    const PLUS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Solo</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>10080</divisions><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>10080</duration><voice>1</voice><type>quarter</type></note>
      <note><grace/><pitch><step>D</step><octave>5</octave></pitch><voice>1</voice><type>eighth</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>10080</duration><voice>1</voice><type>quarter</type><notations><slur type="start" number="1"/><articulations><staccato/></articulations></notations></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>10080</duration><voice>1</voice><type>quarter</type><tie type="start"/><notations><slur type="stop" number="1"/></notations></note>
      <direction placement="below"><direction-type><dynamics><f/></dynamics></direction-type></direction>
      <direction placement="below"><direction-type><wedge type="crescendo"/></direction-type></direction>
    </measure>
    <measure number="2">
      <attributes><key><fifths>2</fifths></key><time><beats>3</beats><beat-type>4</beat-type></time><clef><sign>F</sign><line>4</line></clef></attributes>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>6720</duration><voice>1</voice><type>quarter</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><tie type="stop"/><notations><tuplet type="start" number="1" bracket="yes"/></notations></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>6710</duration><voice>1</voice><type>quarter</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>6730</duration><voice>1</voice><type>quarter</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="stop" number="1"/></notations></note>
      <barline location="right"><bar-style>light-light</bar-style></barline>
    </measure>
    <measure number="3">
      <note><pitch><step>A</step><octave>3</octave></pitch><duration>20160</duration><voice>1</voice><type>half</type></note>
      <note><rest/><duration>10080</duration><voice>1</voice><type>quarter</type><notations><fermata/></notations></note>
      <direction placement="below"><direction-type><wedge type="stop"/></direction-type></direction>
      <barline location="right"><bar-style>light-heavy</bar-style></barline>
    </measure>
  </part>
</score-partwise>"#;

    /// Scorer-style comparison: exact measure + MIDI, tolerant onset +
    /// duration (tick-quantum jitter never exceeds a few 10^-3 beats).
    fn events_keyed(
        xml: &str,
    ) -> Vec<(u32, f64, i32, f64)> {
        use super::super::musicxml_import::score_events;
        score_events(xml)
            .unwrap()
            .into_iter()
            .map(|e| (e.measure, e.onset, e.midi, e.dur))
            .collect()
    }

    fn assert_events_close(got: &[(u32, f64, i32, f64)], want: &[(u32, f64, i32, f64)]) {
        assert_eq!(got.len(), want.len(), "event count: {got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.0, w.0, "measure for {g:?} vs {w:?}");
            assert!((g.1 - w.1).abs() < 0.01, "onset for {g:?} vs {w:?}");
            assert_eq!(g.2, w.2, "midi for {g:?} vs {w:?}");
            assert!((g.3 - w.3).abs() < 0.01, "dur for {g:?} vs {w:?}");
        }
    }

    #[test]
    fn bekern_plus_marks_round_trip() {
        // MusicXML -> bekern+ -> unmerged decode -> MusicXML preserves
        // every pitched event (grace included) and re-renders every
        // mark the fixture carries.
        use super::super::bekern::to_bekern;
        use super::super::musicxml_import::{bekern_events, staff_headers};
        use super::super::{parse_score, to_musicxml, LossTable};
        use super::decode_bekern_unmerged;
        let headers = staff_headers(PLUS_XML).unwrap();
        let events = bekern_events(PLUS_XML).unwrap();
        let (toks, graces) = to_bekern(&events, &headers).unwrap();
        assert_eq!(graces, 0, "only grace rests skip; the fixture has none");
        let joined = toks.join(" ");
        for mark in [
            "grace eighth D 5",
            "slur-start",
            "slur-stop",
            "stacc",
            "fermata",
            "dyn-f",
            "hairpin-cresc-start",
            "hairpin-stop",
            "bar-double",
            "bar-light-heavy",
            "clef-F-4",
            "key+2",
            "time-3-4",
            "tie-start",
            "tie-stop",
            "tuplet-start-3-2",
            "tuplet-stop",
        ] {
            assert!(
                joined.contains(mark),
                "token stream carries {mark}: {joined}"
            );
        }
        let input = decode_bekern_unmerged(&joined).unwrap();
        // Mid-stream state lands on the right measures.
        let at = |m: u32| input.measure_attrs.iter().find(|a| a.measure == m);
        assert_eq!(at(2).and_then(|a| a.key_fifths), Some(2));
        assert_eq!(
            at(2).and_then(|a| a.time),
            Some(super::super::TimeSig { beats: 3, beat_type: 4 })
        );
        assert_eq!(at(2).and_then(|a| a.clef_sign), Some('F'));
        assert_eq!(at(2).and_then(|a| a.clef_line), Some(4));
        assert_eq!(at(1).and_then(|a| a.dynamic.as_deref()), Some("f"));
        assert_eq!(at(1).and_then(|a| a.hairpin.as_deref()), Some("cresc"));
        assert_eq!(at(2).and_then(|a| a.barline_right.as_deref()), Some("double"));
        assert_eq!(at(3).and_then(|a| a.hairpin.as_deref()), Some("stop"));
        assert_eq!(
            at(3).and_then(|a| a.barline_right.as_deref()),
            Some("light-heavy")
        );
        let (score, mut losses) = parse_score(&input).unwrap();
        assert!(losses.entries.is_empty(), "unexpected losses: {losses:?}");
        let hyp = to_musicxml(&score, &mut LossTable::default());
        for frag in [
            "<grace/>",
            "<slur type=\"start\"",
            "<slur type=\"stop\"",
            "<staccato/>",
            "<fermata/>",
            "<dynamics><f/></dynamics>",
            "<wedge type=\"crescendo\"/>",
            "<wedge type=\"stop\"/>",
            "<bar-style>light-light</bar-style>",
            "<bar-style>light-heavy</bar-style>",
            "<time><beats>3</beats><beat-type>4</beat-type></time>",
            "<key><fifths>2</fifths></key>",
            "<clef><sign>F</sign><line>4</line></clef>",
            "<tie type=\"start\"/>",
            "<tie type=\"stop\"/>",
        ] {
            assert!(hyp.contains(frag), "rendered MusicXML carries {frag}");
        }
        assert_events_close(&events_keyed(&hyp), &events_keyed(PLUS_XML));
    }

    #[test]
    fn pickup_measure_has_no_trailing_padding() {
        // An anacrusis must not force trailing padding rests: the token
        // stream carries no `bar-rest` for the short measure, and the
        // next measure still lands on the grid.
        use super::super::bekern::to_bekern;
        use super::super::musicxml_import::{bekern_events, staff_headers};
        use super::super::{parse_score, to_musicxml, LossTable};
        use super::decode_bekern_unmerged;
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Solo</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>480</divisions><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>480</duration><voice>1</voice><type>quarter</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>960</duration><voice>1</voice><type>half</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>960</duration><voice>1</voice><type>half</type></note>
    </measure>
  </part>
</score-partwise>"#;
        let headers = staff_headers(xml).unwrap();
        let events = bekern_events(xml).unwrap();
        let (toks, _) = to_bekern(&events, &headers).unwrap();
        assert!(
            !toks.iter().any(|t| t == "bar-rest"),
            "no padding rests forced: {toks:?}"
        );
        let input = decode_bekern_unmerged(&toks.join(" ")).unwrap();
        let d4 = input.parts[0]
            .notes
            .iter()
            .find(|n| n.pitch.as_deref() == Some("D4"))
            .expect("D4 decodes");
        assert!(approx(d4.onset, 4.0), "measure 2 lands on the grid");
        let (score, _) = parse_score(&input).unwrap();
        let hyp = to_musicxml(&score, &mut LossTable::default());
        assert_events_close(&events_keyed(&hyp), &events_keyed(xml));
    }
}
