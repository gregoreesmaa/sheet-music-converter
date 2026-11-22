//! bekern target tokens for the MVD learned decoder (bekern+).
//!
//! Semantic encoding: one token per duration/step/octave/
//! accidental/dot, then slur/modifier marks, then tie marks;
//! `rest` + duration + dots (+ modifiers) for rests, `bar` (or a
//! `bar-*-*` variant) after each measure's events, per-staff
//! `clef-*-*` / `key-*` / `time-*-*-*` headers up front plus mid-stream
//! changes at any measure start (or mid-measure voice cursor).
//! Measure dynamics (`dyn-*`), hairpins (`hairpin-*-start|stop`) and an
//! opening repeat (`bar-repeat-left`) ride at measure scope. Events sort
//! by (measure, onset, staff, voice, grace-first, midi); chord tones are
//! full note tokens each (no spine markers in v1). Grace notes emit
//! their duration + pitch (zero advance on decode); only grace rests are
//! un-notatable and skipped (counted). Tuplet members carry written
//! durations inside `tuplet-start-A-N` ... `tuplet-stop` spans.
//! Genuinely untypeable durations are an error, never rounded (near
//! misses within [`DURATION_SNAP_BEATS`] snap to the table).
//!
//! Timing authority is `<time-modification>` (per-note tick scaling),
//! never the visual `<tuplet>` bracket alone: MuseScore split exports
//! carry visual-only brackets whose members lack time-modification
//! (straight ticks under a triplet bracket, e.g. ds0091 m37, ds0324-p4
//! split-m8). Opening a scaling span for those would phantom-shift
//! onsets away from the engraved positions (0.167-beat lies that no
//! quantizer can honestly fix), so markers without a ratio only balance
//! against each other and never scale. A no-ratio stop with no pending
//! visual start on its key still fails closed (stray-stop).

use super::musicxml_import::{step_semitone, BekernNote, StaffHeader};

/// Open tuplet spans keyed by `(staff, voice)`: a stack of
/// `((actual, normal), number)` so nested tuplets stay balanced.
type TupletSpans = std::collections::HashMap<(u32, u32), Vec<((u32, u32), u32)>>;

/// Tick-quantum tolerance (beats) for snapping an engraved duration to
/// the duration table. MuseScore distributes triplet rounding across
/// beam-group members (suite31: 1654/1673/1683/1693 ticks at divisions
/// 10080 for true triplet 16ths, up to 39 written-ticks = 0.00387 beats
/// off the dyadic value), so written durations near the table are
/// quantization of it, never distinct values. 0.005 sits ~3x below the
/// half-gap to the nearest distinct table entry (0.0156 between 32nd..
/// and 16th), so genuine values (triplet quarters at 0.667, ...) can
/// never snap: they fail closed as before.
pub const DURATION_SNAP_BEATS: f64 = 0.005;

/// Map a beat duration to (type token, dot count): the nearest table
/// entry within [`DURATION_SNAP_BEATS`]; anything farther (tuplets,
/// genuinely un-notatable values) is `None`.
pub fn duration_token(dur_beats: f64) -> Option<(&'static str, u8)> {
    const TABLE: &[(&str, f64, u8)] = &[
        ("128th", 0.03125, 0),
        ("64th", 0.09375, 1),
        ("64th", 0.0625, 0),
        ("32nd", 0.125, 0),
        ("16th", 0.25, 0),
        ("eighth", 0.5, 0),
        ("quarter", 1.0, 0),
        ("half", 2.0, 0),
        ("whole", 4.0, 0),
        ("breve", 8.0, 0),
        ("32nd", 0.1875, 1),
        ("16th", 0.375, 1),
        ("eighth", 0.75, 1),
        ("quarter", 1.5, 1),
        ("half", 3.0, 1),
        ("whole", 6.0, 1),
        ("32nd", 0.21875, 2),
        ("16th", 0.4375, 2),
        ("eighth", 0.875, 2),
        ("quarter", 1.75, 2),
        ("half", 3.5, 2),
        ("whole", 7.0, 2),
    ];
    let mut best: Option<(&'static str, u8)> = None;
    let mut best_dist = f64::INFINITY;
    for (t, d, dots) in TABLE {
        let dist = (d - dur_beats).abs();
        if dist < best_dist {
            best_dist = dist;
            best = Some((*t, *dots));
        }
    }
    match best {
        Some(b) if best_dist <= DURATION_SNAP_BEATS => Some(b),
        _ => None,
    }
}

fn key_token(fifths: i32) -> String {
    if fifths < 0 {
        format!("key{fifths}")
    } else {
        format!("key+{fifths}")
    }
}

fn midi_of(step: char, octave: i32, alter: i32) -> i32 {
    (octave + 1) * 12 + step_semitone(step) + alter
}

/// Gaps at or below this (beats) are swallowed, not filled: onsets are
/// quantized to 0.001 and every fill injects up to half a quantum, so
/// sub-quantum gaps are indistinguishable from rounding noise. Real
/// silence is at least a tick (1/division beats); swallowing only
/// affects sub-tick slivers the scorer cannot see anyway.
const GAP_MIN_BEATS: f64 = 0.0015;

/// Emit one or more `rest` tokens summing to `written` beats (greedy
/// largest-first, same table as [`duration_token`]). Gaps spanning
/// several beats need several rests. A leftover at or below a quantum
/// is rounding slop (quantum error amplifies under nested scales),
/// not real time, and is swallowed; anything larger below a 128th is
/// un-notatable and fails loudly.
///
/// `resid` carries the residual-dust bound when the emitting voice sits
/// inside a real (ratio-backed) tuplet span: `Some(max_snap)` allows the
/// sub-64th greedy remainder to snap to the 64th grid (drop it when it
/// rounds to zero, else one 64th rest) provided the snap moves timing by
/// at most `max_snap` beats. Callers pass `None` in straight time, so
/// straight-time untypeable gaps still fail closed. `None` also keeps
/// structural remainders loud: the corpus-censused bound
/// (`4 * scale * GAP_MIN_BEATS`, at most ~0.0135 beats) sits well below
/// half a 64th, so only quantum-amplified dust (ds0324-7: 0.065 written
/// for a true 64th under a 1.40625 nested scale) snaps — never a real
/// non-dyadic silence.
fn emit_rest(
    out: &mut Vec<String>,
    written: f64,
    measure: u32,
    tol: f64,
    resid: Option<f64>,
) -> Result<(), String> {
    const UNITS: &[(&str, f64, u8)] = &[
        ("breve", 8.0, 0),
        ("whole", 7.0, 2),
        ("whole", 6.0, 1),
        ("whole", 4.0, 0),
        ("half", 3.5, 2),
        ("half", 3.0, 1),
        ("half", 2.0, 0),
        ("quarter", 1.75, 2),
        ("quarter", 1.5, 1),
        ("quarter", 1.0, 0),
        ("eighth", 0.875, 2),
        ("eighth", 0.75, 1),
        ("eighth", 0.5, 0),
        ("16th", 0.4375, 2),
        ("16th", 0.375, 1),
        ("16th", 0.25, 0),
        ("32nd", 0.21875, 2),
        ("32nd", 0.1875, 1),
        ("32nd", 0.125, 0),
        ("64th", 0.09375, 1),
        ("64th", 0.0625, 0),
        ("128th", 0.03125, 0),
    ];
    // Atomic: callers emit a wrapper bracket for the whole gap when
    // this fails, so a partial prefix here would double-fill decode's
    // cursor (e.g. half-dot + 16th + 64th, then a whole+quarter wrapper
    // covering the same gap again — observed as +150-tick shifts and
    // cross-bar spills on triplet-grid late entries). Stage into a
    // local buffer and commit only on full success.
    let mut staged: Vec<String> = Vec::new();
    let mut rem = written;
    while rem > tol {
        // Match within the tolerance: quantum-rounded values land on
        // rounding ties (0.333 x 3/2 = 0.4995 vs an eighth), and
        // snapping recovers the tick-exact intent. Anything farther
        // than the tolerance from every unit is genuinely un-notatable.
        // Callers pass a scale-aware tolerance (gap-space quantum times
        // the written-space amplification): residue dust grows with the
        // span product, so a fixed quantum would reject exactly-typed
        // gaps under nested brackets (ds0134: 0.498 vs an eighth under
        // a (3,2) wrapper). Decoded timing shifts stay within one
        // gap-space quantum either way.
        let Some(&(typ, val, dots)) = UNITS.iter().find(|(_, v, _)| *v <= rem + tol) else {
            // Greedy is stuck with a sub-128th remainder (failure implies
            // `rem < 128th - tol`: anything larger would have matched the
            // 128th row). Inside a real tuplet span this is quantum dust
            // amplified by the nesting product, so snap it to the grid;
            // in straight time (or past the dust bound) stay loud.
            const SIXTY_FOURTH: f64 = 0.0625;
            if let Some(cap) = resid {
                let n = (rem / SIXTY_FOURTH).round();
                if (n == 0.0 || n == 1.0) && (rem - n * SIXTY_FOURTH).abs() <= cap {
                    if n == 1.0 {
                        staged.push("rest".to_string());
                        staged.push("64th".to_string());
                    }
                    out.extend(staged);
                    return Ok(());
                }
            }
            return Err(format!(
                "untypeable rest gap {written} at measure {measure}"
            ));
        };
        staged.push("rest".to_string());
        staged.push(typ.to_string());
        for _ in 0..dots {
            staged.push("dot".to_string());
        }
        rem -= val;
    }
    out.extend(staged);
    Ok(())
}

/// Written-to-performed scale of a voice's open spans: the product of
/// actual/normal over the stack. A note's performed duration is its
/// written beats divided by this; a silent gap's written rest is the
/// gap times this.
fn span_scale_an(open: &TupletSpans, key: (u32, u32)) -> f64 {
    open.get(&key).map_or(1.0, |s| {
        s.iter().map(|((a, n), _)| *a as f64 / *n as f64).product()
    })
}

/// Emit a deferred `tuplet-stop`: pop the key's span (validating the
/// number the stop was deferred with) and push the token. Decode sees
/// the token while positioned in the stopped note's chord-run segment,
/// so the pop routes to the right stack.
fn flush_stop(
    out: &mut Vec<String>,
    open: &mut TupletSpans,
    key: (u32, u32),
    num: u32,
    measure: u32,
) -> Result<(), String> {
    match open.get_mut(&key).and_then(|s| s.pop()) {
        Some((_, open_num)) if open_num == num => {
            out.push("tuplet-stop".to_string());
            Ok(())
        }
        _ => Err(format!(
            "unmatched tuplet-stop (number {num}) at measure {measure}"
        )),
    }
}

/// Flatten events + headers to bekern tokens. Returns the tokens and
/// the skip count: grace rests (no engraved duration to steal) plus
/// bar-overflow spacing rests (written rests starting at or past their
/// bar's nominal capacity, e.g. invisible `print-object="no"` layout
/// spacing — keeping them would leak them into the next bar's rendering
/// and shift its voice).
pub fn to_bekern(
    events: &[BekernNote],
    headers: &[StaffHeader],
) -> Result<(Vec<String>, usize), String> {
    let mut heads: Vec<&StaffHeader> = headers.iter().collect();
    heads.sort_by_key(|h| h.staff);
    let mut out = Vec::new();
    for h in &heads {
        out.push(format!("clef-{}-{}", h.clef_sign, h.clef_line));
        out.push(key_token(h.key_fifths));
        out.push(format!("time-{}-{}", h.beats, h.beat_type));
    }
    // Nominal bar capacity in beats per measure: the score-initial
    // signature, overridden from each mid-stream time change's carrier
    // measure on (last-wins per measure in walk order). Rests starting
    // at or past their measure's capacity are bar-overflow spacing
    // (e.g. invisible `print-object="no"` layout rests): they belong to
    // no bar's content — a rest exactly at the barline would otherwise
    // render at the next bar's head and shift its voice.
    let initial_cap = heads
        .first()
        .map(|h| h.beats as f64 * 4.0 / h.beat_type.max(1) as f64)
        .unwrap_or(4.0);
    let mut sig_by_measure: std::collections::BTreeMap<u32, f64> =
        std::collections::BTreeMap::new();
    for e in events.iter() {
        if let Some((b, t)) = e.time_change {
            sig_by_measure.insert(e.measure, b as f64 * 4.0 / t.max(1) as f64);
        }
    }
    let cap_for = |m: u32| -> f64 {
        let mut cap = initial_cap;
        for (&cm, &sig) in sig_by_measure.iter() {
            if cm > m {
                break;
            }
            cap = sig;
        }
        cap
    };
    let mut evs: Vec<&BekernNote> = Vec::with_capacity(events.len());
    let mut graces = 0usize;
    // Carrier marks attached to skipped grace rests or bar-overflow
    // spacing rests are preserved per measure so dynamics, hairpins,
    // mid-stream headers, and barlines are never silently dropped.
    let mut skipped_clef: std::collections::HashMap<u32, (char, u32)> =
        std::collections::HashMap::new();
    let mut skipped_key: std::collections::HashMap<u32, i32> = std::collections::HashMap::new();
    let mut skipped_time: std::collections::HashMap<u32, (u32, u32)> =
        std::collections::HashMap::new();
    let mut skipped_dyn: std::collections::HashMap<u32, Vec<String>> =
        std::collections::HashMap::new();
    let mut skipped_hairpin: std::collections::HashMap<u32, Vec<String>> =
        std::collections::HashMap::new();
    let mut skipped_bleft: std::collections::HashMap<u32, String> =
        std::collections::HashMap::new();
    let mut skipped_bright: std::collections::HashMap<u32, String> =
        std::collections::HashMap::new();
    let mut stash_skipped = |e: &BekernNote| {
        if let Some(c) = e.clef_change {
            skipped_clef.insert(e.measure, c);
        }
        if let Some(k) = e.key_change {
            skipped_key.insert(e.measure, k);
        }
        if let Some(t) = e.time_change {
            skipped_time.insert(e.measure, t);
        }
        if let Some(d) = &e.dynamic {
            skipped_dyn.entry(e.measure).or_default().push(d.clone());
        }
        if let Some(h) = &e.hairpin {
            skipped_hairpin
                .entry(e.measure)
                .or_default()
                .push(h.clone());
        }
        if let Some(b) = &e.barline_left {
            skipped_bleft.insert(e.measure, b.clone());
        }
        if let Some(b) = &e.barline_right {
            skipped_bright.insert(e.measure, b.clone());
        }
    };
    for e in events {
        // Only grace rests are un-notatable (no engraved duration to
        // steal); pitched grace notes emit `grace` + duration + pitch.
        if e.grace && e.pitch.is_none() {
            stash_skipped(e);
            graces += 1;
            continue;
        }
        // Whole-bar rests start at zero and never hit the overflow bar.
        if e.pitch.is_none() && !e.measure_rest && e.onset >= cap_for(e.measure) {
            stash_skipped(e);
            graces += 1;
            continue;
        }
        evs.push(e);
    }
    // Grace notes sort before same-onset plain events (after staff
    // and voice, before MIDI pitch): they steal time from the
    // following note, so decoding them after it would shift their
    // onset past it. The sort is stable, so all other ties keep
    // document order (chord runs stay adjacent).
    evs.sort_by(|a, b| {
        a.measure
            .cmp(&b.measure)
            .then(a.onset.partial_cmp(&b.onset).unwrap())
            .then(a.staff.cmp(&b.staff))
            .then(a.voice.cmp(&b.voice))
            .then((!a.grace).cmp(&!b.grace))
            .then(midi_key(a).cmp(&midi_key(b)))
    });
    let mut measure = 0u32;
    let mut started = false;
    // Open tuplet spans per (staff, voice): `(ratio, number)` stack, so
    // nested tuplets and page-cut continuations stay deterministic.
    let mut open: TupletSpans = std::collections::HashMap::new();
    // A stop is deferred past the stopped note's same-onset chord run:
    // the chord tones belong to the stopped note's bracket, so the span
    // must stay open (and scaled) through them. The pop happens now
    // (bookkeeping), the token is emitted at the next non-chord event,
    // at the barline, or at end of stream — never mid-chord. Without
    // this, chord tones after a stop would open phantom implicit spans.
    let mut deferred_stop: Option<((u32, u32), u32)> = None;
    // Visual-only bracket balance per (staff, voice): numbers of
    // `<tuplet type="start">` markers seen with NO time-modification
    // (no timing span opened for them). A later no-ratio stop consumes
    // one pending number of the same value; a stop with nothing pending
    // is a stray and fails closed. Chord-tone duplicates of the root's
    // markers are idempotent via `last_visual_*` (same (key, number) on
    // the same onset skips), mirroring the ratio-stop single slot.
    let mut pending_visual: std::collections::HashMap<(u32, u32), Vec<u32>> =
        std::collections::HashMap::new();
    let mut last_visual_start: Option<((u32, u32), u32)> = None;
    let mut last_visual_stop: Option<((u32, u32), u32)> = None;
    // Per-voice performed-time cursor (beats, measure-relative): the
    // import cursor advances through un-notated silence (late-entering
    // voices, `<forward>` gaps) with no same-key events, but token
    // streams carry no positions — without an explicit rest here,
    // decode would place every later note of this voice too early.
    // Tracks decode's cursor (performed durations under open spans),
    // reset each bar like the import cursor.
    let mut pos: std::collections::HashMap<(u32, u32), f64> = std::collections::HashMap::new();
    // Measure of each voice's last event: a span left open across a
    // barline whose voice then stays silent for whole bars is unclosed
    // in the source (its stop marker is missing), not a bracket
    // continuing here. Recency (not a same-measure flag) tells a live
    // cross-bar bracket from a stale one.
    let mut live: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    // Spans split at barlines (decode resets per-measure cursors and
    // cannot carry a span across `bar`): pre-bar tails closed here,
    // reopened below on the key's next event with identical stacks.
    // Sorted keys keep emission deterministic.
    let mut reopen: TupletSpans = std::collections::HashMap::new();
    // Previous emitted event's (measure, onset, staff, voice, pitched):
    // a pitched note repeating all five is a chord tone, marked `chord`
    // (without the marker the stream cannot tell chords from sequential
    // same-voice notes).
    // Previous event key: (measure, onset, staff, voice, pitched,
    // grace). The grace bit keeps chord detection honest: a grace note
    // and the note it ornaments share onset/staff/voice but are
    // sequential, never a chord (only grace+grace pairs chord).
    let mut prev: Option<(u32, f64, u32, u32, bool, bool)> = None;
    // Voice segments per measure: (staff, voice) -> 1-based appearance
    // number, kept for the whole measure. A marker is emitted on EVERY
    // voice run (including reappearances with their original number):
    // token order interleaves voices by onset, so first-appearance-only
    // markers would strand later runs in the wrong decode segment.
    // Without these markers decode cannot invert multi-voice bars.
    let mut seg_of: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    let mut prev_key: Option<(u32, u32)> = None;
    // Closing barline style of the measure being built (last-wins
    // across its events): emitted instead of `bar` at the close.
    let mut pending_right: Option<String> = None;
    for e in evs {
        // Flush a deferred stop at the first event that is not a
        // same-onset chord continuation of the stopped note. Grace
        // parity: only grace+grace pairs chord (a grace and its
        // ornamented note share the key but are sequential).
        let continues_chord = e.pitch.is_some()
            && prev
                == Some((
                    e.measure, e.onset, e.staff, e.voice, true, e.grace,
                ));
        // Key whose span the marker path just closed this round: the
        // marker owns that key, so the synthetic precise stop below
        // must not pop a level behind it (nested brackets).
        let mut flushed_key: Option<(u32, u32)> = None;
        if !continues_chord {
            if let Some((dkey, dnum)) = deferred_stop.take() {
                flush_stop(&mut out, &mut open, dkey, dnum, e.measure)?;
                flushed_key = Some(dkey);
            }
            // A new chord-run root ends visual-marker idempotence for the
            // previous run (same rule as the deferred-stop single slot).
            last_visual_start = None;
            last_visual_stop = None;
        }
        if e.measure != measure {
            if started {
                let mut keys: Vec<(u32, u32)> = open.keys().cloned().collect();
                keys.sort();
                for key in keys {
                    if let Some(stack) = open.remove(&key) {
                        if stack.is_empty() {
                            continue;
                        }
                        // Key had events this measure, so it has a segment:
                        // seg_of gains an entry for every event (before
                        // the measure_rest early-continue), and spans
                        // never cross `bar` by construction. The lookup
                        // stays graceful in release, but a miss means the
                        // invariant broke and deserves a loud test failure.
                        debug_assert!(
                            seg_of.contains_key(&key),
                            "open span with no segment this measure"
                        );
                        let Some(&seg) = seg_of.get(&key) else {
                            continue;
                        };
                        out.push(format!("voice-{seg}"));
                        for _ in &stack {
                            out.push("tuplet-stop".to_string());
                        }
                        reopen.insert(key, stack);
                    }
                }
                out.push(pending_right.take().unwrap_or_else(|| "bar".to_string()));
            }
            measure = e.measure;
            started = true;
            seg_of.clear();
            pos.clear();
            prev_key = None;
        }
        // The current event's closing barline (if it carries one)
        // applies to the close of THIS measure, so it is collected
        // after the previous measure closed above.
        if let Some(b) = skipped_bright.remove(&e.measure) {
            pending_right = Some(b);
        }
        if let Some(b) = &e.barline_right {
            pending_right = Some(b.clone());
        }
        // Mid-stream notation state (bekern+): attribute and direction
        // changes attached to this event emit here, in bound position
        // before the voice marker, in fixed order. Any marks rescued
        // from skipped grace/overflow rests in this measure flush first.
        if let Some((sign, line)) = skipped_clef.remove(&e.measure) {
            out.push(format!("clef-{sign}-{line}"));
        }
        if let Some((sign, line)) = e.clef_change {
            out.push(format!("clef-{sign}-{line}"));
        }
        if let Some(k) = skipped_key.remove(&e.measure) {
            out.push(key_token(k));
        }
        if let Some(k) = e.key_change {
            out.push(key_token(k));
        }
        if let Some((b, t)) = skipped_time.remove(&e.measure) {
            out.push(format!("time-{b}-{t}"));
        }
        if let Some((b, t)) = e.time_change {
            out.push(format!("time-{b}-{t}"));
        }
        if let Some(ds) = skipped_dyn.remove(&e.measure) {
            out.extend(ds);
        }
        if let Some(d) = &e.dynamic {
            out.push(d.clone());
        }
        if let Some(hs) = skipped_hairpin.remove(&e.measure) {
            out.extend(hs);
        }
        if let Some(h) = &e.hairpin {
            out.push(h.clone());
        }
        if let Some(b) = skipped_bleft.remove(&e.measure) {
            out.push(b);
        }
        if let Some(b) = &e.barline_left {
            out.push(b.clone());
        }
        let key = (e.staff, e.voice);
        let seg_n = seg_of.len() as u32 + 1;
        let seg = *seg_of.entry(key).or_insert(seg_n);
        if prev_key != Some(key) {
            out.push(format!("voice-{seg}"));
            prev_key = Some(key);
        }
        // Whole-bar rest: one token, no duration (a corrupt tuplet
        // marker on one is ignored; open spans auto-close at the end).
        if e.measure_rest {
            out.push("bar-rest".to_string());
            prev = Some((e.measure, e.onset, e.staff, e.voice, false, false));
            live.insert(key, measure);
            continue;
        }
        // Per-event span-restore bookkeeping: whether this event
        // reattached a barline-split span (barline reads nest by
        // default), and whether the voice cursor abuts this onset
        // (no silence since the last note — evidence the previous run
        // ended exactly here, used by the sequential-bracket rule).
        let mut restored_this_event = false;
        let mut cursor_abuts = false;
        // Cancel reopened spans for voices silent for a bar or more:
        // the voice has no events since an earlier measure, so the
        // unclosed span is a missing stop marker in the source, not a
        // bracket continuing here — restoring it would double-scale
        // every note below (decode opens the restored start on the
        // next event, so a later close cannot match either). The
        // pre-bar tail was already closed with a stop, so discarding
        // the reopen is balanced by construction. Live cross-bar
        // brackets (voice active last bar) restore normally. Skipped
        // while a deferred stop is pending: the stack settles at the
        // next non-chord event first.
        if deferred_stop.is_none() && live.get(&key).map_or(false, |&m| measure > m + 1) {
            reopen.remove(&key);
        }
        // Reopen spans split at the previous barline: identical stacks,
        // emitted bottom-first so decode reattaches in order. Runs
        // before the per-event logic so the implicit-member check
        // below sees the restored top (no duplicate start). Member-
        // recency: a reopened span survives only when this key's next
        // event is a bracket member (ratio/start/stop marker present).
        // A same-key non-member means the bracket did not continue
        // across the barline (the voice went plain) — dropping the
        // reopen stays balanced (the pre-bar tail already closed it)
        // and a stale leak can no longer rescale whole bars below.
        // A ratio mismatch drops the reopen even with an explicit
        // start or stop on the new note: at a barline the old content
        // ends by construction, so a new ratio is always a new bracket
        // (nesting across the barline would need old-ratio members
        // below, which re-open implicitly). The old bracket ended (its
        // stop printed in an earlier measure, possibly on another key);
        // reattaching would double-scale every note below (ds0026
        // m51->m52: a restored (3,2) under (2,1) duplets is scale 3,
        // amplifying quantum dust into an untypeable rest). A bare
        // start/stop marker with no ratio never drops (nothing to
        // compare). The pre-bar tail already closed these levels
        // token-side, and the wrapper/implicit paths below open the new
        // bracket, so dropping stays balanced by construction.
        // Timing evidence only: a continuing bracket member carries
        // its ratio (time-modification). Bare visual markers (no ratio)
        // neither restore a split span nor drop it — the reopen entry
        // stays pending for the next ratio-backed member. (A visual
        // marker alone must not resurrect a stale span and rescale whole
        // bars below.)
        if e.tuplet_ratio.is_some() {
            if let Some(stack) = reopen.remove(&key) {
                let stale = match (e.tuplet_ratio, e.tuplet_start, e.tuplet_stop) {
                    (Some(r), _, _) => stack.last().is_some_and(|(sr, _)| *sr != r),
                    _ => false,
                };
                if !stale {
                    for (ratio, num) in &stack {
                        open.entry(key).or_default().push((*ratio, *num));
                        out.push(format!("tuplet-start-{}-{}", ratio.0, ratio.1));
                    }
                    restored_this_event = true;
                }
            }
        }
        // Fill a silent gap before this event: the voice was quiet while
        // time passed (late entry, `<forward>`, un-notated intra-voice
        // gap). The rest is written under the currently open spans, so
        // decode's cursor lands exactly on this event.
        if !continues_chord {
            let cur = pos.get(&key).copied().unwrap_or(0.0);
            // Compare against the rounded cursor with the import's own
            // function; anything at or below a quantum is rounding
            // noise (triplet grids quantize asymmetrically: 0.333 +
            // 0.333 reads back as 0.666 against an onset of 0.667).
            let gap = e.onset - super::musicxml_import::round3(cur);
            cursor_abuts = gap.abs() <= GAP_MIN_BEATS;
            if gap > GAP_MIN_BEATS {
                // The gap rest joins the currently open spans (a voice
                // silent inside its own bracket), so decode scales it
                // back to exactly the gap; multi-beat gaps decompose.
                // Quantum-round the emitted value (float dust like
                // 1.0005 for a true quarter would otherwise leave an
                // untypeable sliver); the cursor chain stays exact.
                let fill_scale = span_scale_an(&open, key);
                // Tolerance cap (arch-review-gap-encoding Q3): the
                // scale-aware quantum compensates dust amplified by
                // nesting; corpus census (2026-11-07) finds max
                // fill_scale 2.25, so anything above 3.0 is novel input
                // that deserves a conscious decision, not silent
                // tolerance growth. Release builds skip the check.
                debug_assert!(
                    fill_scale <= 3.0,
                    "fill_scale {fill_scale} exceeds censused bound"
                );
                let written = super::musicxml_import::round3(gap * fill_scale);
                let emit_ctx = || {
                    format!(
                        "gap {gap:.17} onset {:.17} scale {fill_scale} key {key:?} pitch {:?} tm {:?}",
                        e.onset, e.pitch, e.tuplet_ratio
                    )
                };
                // Residual-dust bound for the snap inside `emit_rest`:
                // only a voice under a real (ratio-backed) span may snap
                // a sub-64th remainder (quantum dust amplified by the
                // nesting product, ds0324-7); straight-time gaps stay
                // fail-closed (`None`).
                let resid = (span_scale_an(&open, key) != 1.0)
                    .then(|| 4.0 * fill_scale * GAP_MIN_BEATS);
                match emit_rest(
                    &mut out,
                    written,
                    e.measure,
                    fill_scale * GAP_MIN_BEATS,
                    resid,
                )
                .map_err(|inner| format!("{inner} | {}", emit_ctx()))
                {
                    Ok(()) => {}
                    Err(_) => {
                        // Non-dyadic silence with no open span (e.g. a
                        // voice entering on a triplet gridline): wrap it
                        // in a self-contained bracket of the upcoming
                        // note's ratio so timing stays exact. Straight
                        // non-dyadic silence is un-notatable — fail loud.
                        if span_scale_an(&open, key) != 1.0 {
                            return Err(format!(
                                "untypeable rest gap {written} at measure {} | {}",
                                e.measure,
                                emit_ctx()
                            ));
                        }
                        let (r, n) = match (e.tuplet_start, e.tuplet_ratio) {
                            (_, Some(t)) => (t, e.tuplet_start.unwrap_or(1)),
                            // A bare visual start (no time-modification)
                            // carries no ratio, so it cannot anchor a
                            // timing wrapper — inventing (3,2) here would
                            // fake a bracket the source never timed.
                            (Some(_), None) | (None, None) => {
                                return Err(format!(
                                    "untypeable rest gap {written} at measure {} | {}",
                                    e.measure,
                                    emit_ctx()
                                ));
                            }
                        };
                        out.push(format!("tuplet-start-{}-{}", r.0, r.1));
                        open.entry(key).or_default().push((r, n));
                        // Round from the unrounded gap (0.667 x 3/2 is
                        // exactly a quarter; rounding `written` first
                        // would bake the quantum in as 1.0005).
                        let inner_written =
                            super::musicxml_import::round3(gap * r.0 as f64 / r.1 as f64);
                        let inner = emit_rest(
                            &mut out,
                            inner_written,
                            e.measure,
                            r.0 as f64 / r.1 as f64 * GAP_MIN_BEATS,
                            Some(4.0 * r.0 as f64 / r.1 as f64 * GAP_MIN_BEATS),
                        )
                        .map_err(|inner_err| format!("{inner_err} | {}", emit_ctx()));
                        open.get_mut(&key).and_then(|s| s.pop());
                        out.push("tuplet-stop".to_string());
                        inner?;
                    }
                }
                pos.insert(key, cur + gap);
            }
            // Precise per-voice stop (cross-staff brackets): the stopped
            // note's marker can live on another key, leaving this key's
            // span open past its last member. A plain event on a key
            // with an open span ends that span here — the gap above was
            // already filled under the span (decode's cursor still lands
            // exactly), but this note and everything after it decode
            // unscaled. Never mid-chord-run (this whole block is
            // skipped there — the second and later chord tones all
            // satisfy `continues_chord`, and the chord root is a genuine
            // event that must close the span), and never for the key
            // the marker path just flushed (the marker owns it). The
            // stop routes to this event's segment: the voice marker
            // above already switched `seg` to this key.
            if flushed_key != Some(key)
                && e.tuplet_ratio.is_none()
                && e.tuplet_start.is_none()
                && e.tuplet_stop.is_none()
                && open.get(&key).map_or(false, |s| !s.is_empty())
            {
                open.get_mut(&key).and_then(|s| s.pop());
                out.push("tuplet-stop".to_string());
            }
        }
        if let Some(num) = e.tuplet_start {
            if e.tuplet_ratio.is_none() {
                // Visual-only bracket start (no time-modification): the
                // engraved ticks are straight, so no scaling span opens —
                // opening one would phantom-shift every member onset
                // (ds0091 m37: 0.167-beat lies). The number is recorded so
                // the bracket's own stop balances explicitly below; chord
                // tones repeating the root's marker are idempotent.
                if !(continues_chord && last_visual_start == Some((key, num))) {
                    pending_visual.entry(key).or_default().push(num);
                    last_visual_start = Some((key, num));
                }
            } else {
                let ratio = e.tuplet_ratio.unwrap();
                // Same bracket as a reopened cross-bar span (identical ratio
                // AND number): the reopen above already restored + emitted
                // it — pushing again would double-scale every member note
                // (e.g. 16th x2/3 twice = 53 ticks instead of 80).
                // Sequential, not nested: an explicitly starting bracket of
                // a different ratio arriving where the voice cursor abuts
                // means the previous run ended exactly here (an unmarked
                // duplet run giving way to triplets mid-measure, ds0226) —
                // nesting it would double-scale below. Only the unnumbered
                // top closes (implicit runs always carry number 1; a
                // numbered top is surely an explicit bracket genuinely
                // containing the new one). Skipped for just-restored spans:
                // the barline reads nest by default (the pre-bar content
                // ends there by construction, so abutment carries no
                // information).
                // `cursor_abuts` is false mid-chord by construction (the
                // gap block is skipped for chord tones), so this never
                // fires mid-chord-run: the root decides for the chord.
                if !restored_this_event
                    && cursor_abuts
                    && matches!(open.get(&key).and_then(|s| s.last().copied()),
                        Some(((a, n), 1)) if (a, n) != ratio)
                {
                    open.get_mut(&key).and_then(|s| s.pop());
                    out.push("tuplet-stop".to_string());
                }
                let top = open.get(&key).and_then(|s| s.last().copied());
                if top != Some((ratio, num)) {
                    open.entry(key).or_default().push((ratio, num));
                    out.push(format!("tuplet-start-{}-{}", ratio.0, ratio.1));
                }
            }
        } else if let Some(ratio) = e.tuplet_ratio {
            // Marker-less member (page-cut continuation or unmarked
            // inner note): open implicitly unless this exact ratio is
            // already on top.
            let top = open.get(&key).and_then(|s| s.last().map(|t| t.0));
            if top != Some(ratio) {
                open.entry(key).or_default().push((ratio, 1));
                out.push(format!("tuplet-start-{}-{}", ratio.0, ratio.1));
            }
        }
        // NOTE: no fallback decomposition (e.g. breve = 2 x whole):
        // the decoder must learn the engraved duration, not an
        // arithmetic alias of it.
        let (typ, dots) = duration_token(e.dur_beats).ok_or_else(|| {
            format!(
                "untypeable duration {} at measure {}",
                e.dur_beats, e.measure
            )
        })?;
        let is_chord_tone = e.pitch.is_some()
            && prev
                == Some((
                    e.measure, e.onset, e.staff, e.voice, true, e.grace,
                ));
        // The `grace` marker leads the chord run once (chord tones
        // inherit grace from their root on decode).
        if e.grace && !is_chord_tone {
            out.push("grace".to_string());
        }
        if is_chord_tone {
            out.push("chord".to_string());
        }
        match e.pitch {
            Some((step, octave, alter)) => {
                out.push(typ.to_string());
                out.push(step.to_string());
                out.push(octave.to_string());
                if alter != 0 {
                    out.push(format!("a{alter}"));
                }
            }
            None => {
                out.push("rest".to_string());
                out.push(typ.to_string());
            }
        }
        for _ in 0..dots {
            out.push("dot".to_string());
        }
        // Modifiers precede ties (decode's pitch-tail consumes them in
        // this order); rests carry them too (fermatas over rests).
        if e.slur_start {
            out.push("slur-start".to_string());
        }
        if e.slur_stop {
            out.push("slur-stop".to_string());
        }
        for a in &e.articulations {
            out.push(a.clone());
        }
        if e.pitch.is_some() {
            if e.tie_start {
                out.push("tie-start".to_string());
            }
            if e.tie_stop {
                out.push("tie-stop".to_string());
            }
        }
        // Advance this voice's cursor by the note's performed duration
        // (written undo-scaled by the still-open spans — the stop below
        // only defers, so the stack is intact here). Chord tones share
        // their root's onset and advance nothing; grace notes steal no
        // time (decode's cursor holds too, so the gap-fill below sees
        // the true silence). Full precision: the chain carries its own
        // residue and only the gap comparison is quantized, so triplet
        // grids (0.333 + 0.333 = 0.667 in float) never phantom-fill;
        // the threshold swallows the straddle.
        if !continues_chord && !e.grace {
            let cur = pos.get(&key).copied().unwrap_or(0.0);
            let performed = e.dur_beats / span_scale_an(&open, key);
            pos.insert(key, cur + performed);
        }
        live.insert(key, measure);
        // Single-slot invariant for the deferred stop below: two keys
        // can never have stops pending at once — the loop-top flush runs
        // before any new event is processed, and same-onset chord runs
        // (the only events that skip the flush) are forced same-key by
        // `continues_chord`. So the take-then-set never silently
        // discards another key's pending stop.
        if let Some(num) = e.tuplet_stop {
            // Validate now (loud on mismatch), but emit the token later:
            // the stopped note's same-onset chord tones still belong to
            // this bracket, so the span stays open (and scaled) through
            // them. The pop + token happen at the loop-top flush, at the
            // barline, or at end of stream — never mid-chord. Emitting
            // here would let a chord tone re-open a phantom implicit
            // span with the wrong number and displace every later note.
            match open.get(&key).and_then(|s| s.last()) {
                // No open span on this key. Two honest sub-cases, never
                // a dangling stop and never a silent skip:
                // * the stopped note carries its ratio (time-modification):
                //   a page-truncated bracket whose opener lives off-page.
                //   The note's own timing is exact, so the span opens
                //   implicitly here (explicit start token, like the
                //   implicit-member rule) and closes through the normal
                //   deferred path — balanced by construction.
                // * no ratio (visual-only bracket): the stop balances
                //   against a pending visual start of the same number on
                //   this key (ds0324-p4 split-m8) and emits nothing — the
                //   note stays straight, exactly as engraved. With nothing
                //   pending it is a stray and fails closed.
                None => {
                    if let Some(ratio) = e.tuplet_ratio {
                        if deferred_stop != Some((key, num)) {
                            if let Some((dkey, dnum)) = deferred_stop.take() {
                                flush_stop(&mut out, &mut open, dkey, dnum, e.measure)?;
                            }
                            debug_assert!(deferred_stop.is_none());
                            open.entry(key).or_default().push((ratio, num));
                            out.push(format!("tuplet-start-{}-{}", ratio.0, ratio.1));
                            deferred_stop = Some((key, num));
                        }
                    } else {
                        if continues_chord && last_visual_stop == Some((key, num)) {
                            // Chord-tone duplicate of the root's visual
                            // stop: idempotent, like the ratio path.
                        } else if let Some(pending) = pending_visual.get_mut(&key) {
                            if let Some(i) =
                                pending.iter().rposition(|&n| n == num)
                            {
                                pending.remove(i);
                                last_visual_stop = Some((key, num));
                            } else {
                                return Err(format!(
                                    "unmatched tuplet-stop (number {}) at measure {}",
                                    num, e.measure
                                ));
                            }
                        } else {
                            return Err(format!(
                                "unmatched tuplet-stop (number {}) at measure {}",
                                num, e.measure
                            ));
                        }
                    }
                }
                Some(&(_, open_num)) if open_num == num => {
                    // A repeated stop marker on a later tone of the same
                    // chord run is idempotent (mirrors the explicit-start
                    // dedup): the first marker already deferred this
                    // stop, so skip instead of flushing mid-chord.
                    if deferred_stop != Some((key, num)) {
                        if let Some((dkey, dnum)) = deferred_stop.take() {
                            flush_stop(&mut out, &mut open, dkey, dnum, e.measure)?;
                        }
                        // Single-slot invariant: the take above cleared
                        // any pending stop, so two keys never have stops
                        // pending at once (loop-top flush runs before any
                        // new event is processed).
                        debug_assert!(deferred_stop.is_none());
                        deferred_stop = Some((key, num));
                    }
                }
                _ => {
                    return Err(format!(
                        "unmatched tuplet-stop (number {}) at measure {}",
                        num, e.measure
                    ));
                }
            }
        }
        prev = Some((
            e.measure,
            e.onset,
            e.staff,
            e.voice,
            e.pitch.is_some(),
            e.grace,
        ));
    }
    // A stop deferred past the final event's chord run flushes here.
    if let Some((dkey, dnum)) = deferred_stop.take() {
        flush_stop(&mut out, &mut open, dkey, dnum, measure)?;
    }
    // Page-cut leftovers stay open: no stops are emitted here. A stop
    // would need the key's current segment for routing, but keys with
    // no events in the final measure have none (and decode grows fresh
    // empty stacks on unknown voice markers, so any routing would
    // error or mis-pop). Open spans at EOS are tolerated on both sides:
    // decode ignores them, the mask still allows EOS, and the next
    // page's implicit-member rule reopens the span by ratio.
    if started {
        out.push(pending_right.take().unwrap_or_else(|| "bar".to_string()));
    }
    Ok((out, graces))
}

fn midi_key(e: &BekernNote) -> i32 {
    match e.pitch {
        Some((step, octave, alter)) => midi_of(step, octave, alter),
        None => i32::MAX,
    }
}

// ---------------------------------------------------------------------------
// Deterministic GT Coverage Sidecar (dynamics/hairpins, lyrics/text, Gate-D
// opt-in drum/tab notation).
//
// Keeps the closed bekern+ token vocabulary and 31/31 suite31 round-trip
// untouched while extracting full measure/beat-aligned ground truth for
// dynamics, hairpins, lyrics, text directions, and (when opted in for
// Gate D) unpitched percussion, slash notation, and guitar tab.
// ---------------------------------------------------------------------------

use serde::{Deserialize, Serialize};

/// Schema version for [`GtCoverageSidecar`] JSON serialization.
pub const GT_SIDECAR_SCHEMA_VERSION: u32 = 1;

/// Extraction options for [`extract_gt_sidecar`].
///
/// `include_percussion_tab` defaults to `false` to enforce the Gate-D
/// boundary (no drum-kit / slash-note / tab events in Gate A-C training).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GtCoverageOptions {
    pub include_percussion_tab: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DynamicEvent {
    pub measure: u32,
    pub onset: f64,
    pub staff: u32,
    pub voice: u32,
    pub mark: String,
    pub bekern_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HairpinEvent {
    pub measure: u32,
    pub onset: f64,
    pub staff: u32,
    pub voice: u32,
    pub wedge_type: String,
    pub number: u32,
    pub bekern_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LyricEvent {
    pub measure: u32,
    pub onset: f64,
    pub staff: u32,
    pub voice: u32,
    pub verse: String,
    pub syllabic: Option<String>,
    pub text: String,
    pub extend: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextDirectionEvent {
    pub measure: u32,
    pub onset: f64,
    pub staff: u32,
    pub voice: u32,
    pub kind: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PercussionEvent {
    pub measure: u32,
    pub onset: f64,
    pub dur_beats: f64,
    pub staff: u32,
    pub voice: u32,
    pub display_step: char,
    pub display_octave: i32,
    pub notehead: Option<String>,
    pub instrument_id: Option<String>,
    pub instrument_name: Option<String>,
    pub chord: bool,
    pub grace: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TabEvent {
    pub measure: u32,
    pub onset: f64,
    pub dur_beats: f64,
    pub staff: u32,
    pub voice: u32,
    pub string: u32,
    pub fret: u32,
    pub pitch: Option<String>,
    pub chord: bool,
    pub grace: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlashSpanEvent {
    pub measure: u32,
    pub onset: f64,
    pub staff: u32,
    pub span_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GtCoverageSidecar {
    pub schema_version: u32,
    pub include_percussion_tab: bool,
    pub has_percussion_or_tab: bool,
    pub excluded_staves: Vec<u32>,
    pub dynamics: Vec<DynamicEvent>,
    pub hairpins: Vec<HairpinEvent>,
    pub lyrics: Vec<LyricEvent>,
    pub text_directions: Vec<TextDirectionEvent>,
    pub percussion: Vec<PercussionEvent>,
    pub tab: Vec<TabEvent>,
    pub slash_spans: Vec<SlashSpanEvent>,
}

fn xml_attr(tag: &str, name: &str) -> Option<String> {
    let pat = format!("{name}=\"");
    let start = tag.find(&pat)? + pat.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

fn xml_tag_text(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}");
    let mut pos = 0;
    while let Some(p) = s[pos..].find(&open) {
        let abs = pos + p;
        let after = &s[abs + open.len()..];
        if after.starts_with('>')
            || after.starts_with(' ')
            || after.starts_with('\t')
            || after.starts_with('\n')
            || after.starts_with('\r')
        {
            let gt = s[abs..].find('>')? + abs;
            if s.as_bytes().get(gt.saturating_sub(1)) == Some(&b'/') {
                return None;
            }
            let close = format!("</{tag}>");
            let end = s[gt + 1..].find(&close)? + gt + 1;
            let trimmed = s[gt + 1..end].trim();
            return if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        }
        pos = abs + open.len();
    }
    None
}

fn xml_blocks<'a>(s: &'a str, tag: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut pos = 0;
    while let Some(p) = s[pos..].find(&open) {
        let abs = pos + p;
        let after = &s[abs + open.len()..];
        let boundary = after.starts_with('>')
            || after.starts_with(' ')
            || after.starts_with('\t')
            || after.starts_with('\n')
            || after.starts_with('\r')
            || after.starts_with('/');
        if !boundary {
            pos = abs + open.len();
            continue;
        }
        let Some(gt_rel) = s[abs..].find('>') else {
            break;
        };
        let gt = abs + gt_rel;
        if s.as_bytes().get(gt.saturating_sub(1)) == Some(&b'/') {
            out.push(&s[abs..=gt]);
            pos = gt + 1;
            continue;
        }
        let Some(end_rel) = s[gt + 1..].find(&close) else {
            break;
        };
        let end = gt + 1 + end_rel + close.len();
        out.push(&s[abs..end]);
        pos = end;
    }
    out
}

fn bekern_dyn_token(mark: &str) -> Option<String> {
    match mark {
        "pp" | "p" | "mp" | "mf" | "f" | "ff" | "sfz" => Some(format!("dyn-{mark}")),
        _ => None,
    }
}

fn bekern_wedge_token(wtype: &str) -> Option<String> {
    match wtype {
        "crescendo" => Some("hairpin-cresc-start".to_string()),
        "diminuendo" => Some("hairpin-dim-start".to_string()),
        "stop" => Some("hairpin-stop".to_string()),
        _ => None,
    }
}

fn sidecar_grace_beats(note_type: Option<&str>) -> f64 {
    match note_type {
        Some("breve") => 8.0,
        Some("whole") => 4.0,
        Some("half") => 2.0,
        Some("quarter") => 1.0,
        Some("eighth") => 0.5,
        Some("16th") => 0.25,
        Some("32nd") => 0.125,
        Some("64th") => 0.0625,
        Some("128th") => 0.03125,
        _ => 0.5,
    }
}

/// Extract a deterministic GT coverage sidecar for the full MusicXML score.
pub fn extract_gt_sidecar(
    xml: &str,
    opts: &GtCoverageOptions,
) -> Result<GtCoverageSidecar, String> {
    extract_gt_sidecar_inner(xml, None, opts)
}

/// Extract a deterministic GT coverage sidecar for a 1-based inclusive
/// measure range `start_measure..=end_measure`.
pub fn extract_gt_sidecar_measures(
    xml: &str,
    start_measure: u32,
    end_measure: u32,
    opts: &GtCoverageOptions,
) -> Result<GtCoverageSidecar, String> {
    if start_measure == 0 || end_measure < start_measure {
        return Err(format!(
            "invalid measure range {start_measure}..{end_measure}"
        ));
    }
    extract_gt_sidecar_inner(xml, Some((start_measure, end_measure)), opts)
}

fn extract_gt_sidecar_inner(
    xml: &str,
    range: Option<(u32, u32)>,
    opts: &GtCoverageOptions,
) -> Result<GtCoverageSidecar, String> {
    if !xml.contains("<score-partwise") {
        return Err("not a score-partwise document".to_string());
    }
    let mut inst_names: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for si in xml_blocks(xml, "score-instrument") {
        if let (Some(id), Some(name)) = (xml_attr(si, "id"), xml_tag_text(si, "instrument-name")) {
            inst_names.insert(id, name);
        }
    }

    let mut dynamics = Vec::new();
    let mut hairpins = Vec::new();
    let mut lyrics = Vec::new();
    let mut text_directions = Vec::new();
    let mut percussion = Vec::new();
    let mut tab = Vec::new();
    let mut slash_spans = Vec::new();
    let mut excluded_staves: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();

    let mut staff_base = 0u32;
    for part in xml_blocks(xml, "part") {
        let measures = xml_blocks(part, "measure");
        let mut divisions = 1i64;
        let mut part_staves = 1u32;
        let mut probe_staff = 1u32;
        for m in &measures {
            for attr in xml_blocks(m, "attributes") {
                if let Some(s) = xml_tag_text(attr, "staves").and_then(|v| v.parse::<u32>().ok()) {
                    part_staves = part_staves.max(s.max(1));
                }
            }
            for note in xml_blocks(m, "note") {
                let local = xml_tag_text(note, "staff")
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(probe_staff)
                    .max(1);
                probe_staff = local;
                part_staves = part_staves.max(local);
            }
        }

        for (ord0, meas) in measures.iter().enumerate() {
            let measure = ord0 as u32 + 1;
            let in_range = range.map_or(true, |(s, e)| measure >= s && measure <= e);
            let mut cursor = 0i64;
            let mut last_onset = 0i64;
            let mut last_staff = 1u32;

            // Walk top-level elements inside <measure> in document order.
            let mut rest = match meas.find('>') {
                Some(q) => &meas[q + 1..],
                None => continue,
            };
            while let Some(lt) = rest.find('<') {
                rest = &rest[lt..];
                if rest.starts_with("</measure>") {
                    break;
                } else if rest.starts_with("<attributes") {
                    let Some(end) = rest.find("</attributes>").map(|q| q + "</attributes>".len())
                    else {
                        break;
                    };
                    let blk = &rest[..end];
                    if let Some(d) =
                        xml_tag_text(blk, "divisions").and_then(|v| v.parse::<i64>().ok())
                    {
                        if d > 0 {
                            divisions = d;
                        }
                    }
                    for clef in xml_blocks(blk, "clef") {
                        let cnum = xml_attr(clef, "number")
                            .and_then(|v| v.parse::<u32>().ok())
                            .unwrap_or(1)
                            .max(1);
                        if let Some(sign) = xml_tag_text(clef, "sign") {
                            if sign.eq_ignore_ascii_case("percussion")
                                || sign.eq_ignore_ascii_case("TAB")
                            {
                                excluded_staves.insert(staff_base + cnum - 1);
                            }
                        }
                    }
                    for ms in xml_blocks(blk, "measure-style") {
                        for sl in xml_blocks(ms, "slash") {
                            let stype = xml_attr(sl, "type").unwrap_or_else(|| "start".to_string());
                            let gstaff = staff_base + last_staff - 1;
                            excluded_staves.insert(gstaff);
                            if opts.include_percussion_tab && in_range {
                                slash_spans.push(SlashSpanEvent {
                                    measure,
                                    onset: super::musicxml_import::round3(
                                        cursor as f64 / divisions as f64,
                                    ),
                                    staff: gstaff,
                                    span_type: stype,
                                });
                            }
                        }
                    }
                    for sd in xml_blocks(blk, "staff-details") {
                        if sd.contains("<staff-tuning") {
                            let snum = xml_attr(sd, "number")
                                .and_then(|v| v.parse::<u32>().ok())
                                .unwrap_or(1)
                                .max(1);
                            excluded_staves.insert(staff_base + snum - 1);
                        }
                    }
                    rest = &rest[end..];
                } else if rest.starts_with("<backup>") || rest.starts_with("<backup ") {
                    let Some(end) = rest.find("</backup>").map(|q| q + "</backup>".len()) else {
                        break;
                    };
                    let d = xml_tag_text(&rest[..end], "duration")
                        .and_then(|v| v.parse::<i64>().ok())
                        .unwrap_or(0);
                    cursor -= d;
                    rest = &rest[end..];
                } else if rest.starts_with("<forward>") || rest.starts_with("<forward ") {
                    let Some(end) = rest.find("</forward>").map(|q| q + "</forward>".len()) else {
                        break;
                    };
                    let d = xml_tag_text(&rest[..end], "duration")
                        .and_then(|v| v.parse::<i64>().ok())
                        .unwrap_or(0);
                    cursor += d;
                    rest = &rest[end..];
                } else if rest.starts_with("<direction") {
                    let Some(end) = rest.find("</direction>").map(|q| q + "</direction>".len())
                    else {
                        break;
                    };
                    let blk = &rest[..end];
                    if in_range {
                        let local = xml_tag_text(blk, "staff")
                            .and_then(|v| v.parse::<u32>().ok())
                            .unwrap_or(last_staff)
                            .max(1);
                        let gstaff = staff_base + local - 1;
                        let voice = xml_tag_text(blk, "voice")
                            .and_then(|v| v.parse::<u32>().ok())
                            .unwrap_or(1);
                        let onset =
                            super::musicxml_import::round3(cursor as f64 / divisions as f64);

                        for dyn_blk in xml_blocks(blk, "dynamics") {
                            let mut inner = match dyn_blk.find('>') {
                                Some(q) => &dyn_blk[q + 1..],
                                None => continue,
                            };
                            while let Some(p) = inner.find('<') {
                                inner = &inner[p..];
                                if inner.starts_with("</dynamics>") {
                                    break;
                                }
                                if inner.starts_with("</") {
                                    if let Some(q) = inner.find('>') {
                                        inner = &inner[q + 1..];
                                        continue;
                                    }
                                    break;
                                }
                                let name: String = inner[1..]
                                    .chars()
                                    .take_while(|c| {
                                        !matches!(c, ' ' | '\t' | '\n' | '\r' | '/' | '>')
                                    })
                                    .collect();
                                let mark = if name == "other-dynamics" {
                                    xml_tag_text(inner, "other-dynamics")
                                        .unwrap_or_else(|| "other".to_string())
                                } else {
                                    name.clone()
                                };
                                if !mark.is_empty() {
                                    dynamics.push(DynamicEvent {
                                        measure,
                                        onset,
                                        staff: gstaff,
                                        voice,
                                        bekern_token: bekern_dyn_token(&mark),
                                        mark,
                                    });
                                }
                                if name == "other-dynamics" {
                                    if let Some(q) = inner.find("</other-dynamics>") {
                                        inner = &inner[q + "</other-dynamics>".len()..];
                                        continue;
                                    }
                                }
                                match inner.find('>') {
                                    Some(q) => inner = &inner[q + 1..],
                                    None => break,
                                }
                            }
                        }

                        for wedge in xml_blocks(blk, "wedge") {
                            if let Some(wtype) = xml_attr(wedge, "type") {
                                if matches!(
                                    wtype.as_str(),
                                    "crescendo" | "diminuendo" | "stop" | "continue"
                                ) {
                                    let number = xml_attr(wedge, "number")
                                        .and_then(|v| v.parse::<u32>().ok())
                                        .unwrap_or(1)
                                        .max(1);
                                    hairpins.push(HairpinEvent {
                                        measure,
                                        onset,
                                        staff: gstaff,
                                        voice,
                                        bekern_token: bekern_wedge_token(&wtype),
                                        wedge_type: wtype,
                                        number,
                                    });
                                }
                            }
                        }

                        for w in xml_blocks(blk, "words") {
                            if let Some(txt) = xml_tag_text(w, "words") {
                                text_directions.push(TextDirectionEvent {
                                    measure,
                                    onset,
                                    staff: gstaff,
                                    voice,
                                    kind: "words".to_string(),
                                    text: txt,
                                });
                            }
                        }
                        for r in xml_blocks(blk, "rehearsal") {
                            if let Some(txt) = xml_tag_text(r, "rehearsal") {
                                text_directions.push(TextDirectionEvent {
                                    measure,
                                    onset,
                                    staff: gstaff,
                                    voice,
                                    kind: "rehearsal".to_string(),
                                    text: txt,
                                });
                            }
                        }
                        for m in xml_blocks(blk, "metronome") {
                            let bu = xml_tag_text(m, "beat-unit").unwrap_or_default();
                            let pm = xml_tag_text(m, "per-minute").unwrap_or_default();
                            let txt = if !bu.is_empty() && !pm.is_empty() {
                                format!("{bu}={pm}")
                            } else {
                                format!("{bu}{pm}")
                            };
                            if !txt.is_empty() {
                                text_directions.push(TextDirectionEvent {
                                    measure,
                                    onset,
                                    staff: gstaff,
                                    voice,
                                    kind: "metronome".to_string(),
                                    text: txt,
                                });
                            }
                        }
                    }
                    rest = &rest[end..];
                } else if rest.starts_with("<harmony") {
                    let Some(end) = rest.find("</harmony>").map(|q| q + "</harmony>".len()) else {
                        break;
                    };
                    let blk = &rest[..end];
                    if in_range {
                        let local = xml_tag_text(blk, "staff")
                            .and_then(|v| v.parse::<u32>().ok())
                            .unwrap_or(last_staff)
                            .max(1);
                        let gstaff = staff_base + local - 1;
                        if let Some(step) = xml_tag_text(blk, "root-step") {
                            let alter = xml_tag_text(blk, "root-alter")
                                .and_then(|v| v.parse::<i32>().ok())
                                .unwrap_or(0);
                            let acc = match alter {
                                1 => "#",
                                2 => "##",
                                -1 => "b",
                                -2 => "bb",
                                _ => "",
                            };
                            let mut htxt = format!("{step}{acc}");
                            if let Some(kind_blk) = xml_blocks(blk, "kind").into_iter().next() {
                                let k = xml_attr(kind_blk, "text")
                                    .or_else(|| xml_tag_text(kind_blk, "kind"))
                                    .unwrap_or_default();
                                if !k.is_empty() && k != "major" {
                                    htxt.push(':');
                                    htxt.push_str(&k);
                                }
                            }
                            if let Some(bstep) = xml_tag_text(blk, "bass-step") {
                                let balter = xml_tag_text(blk, "bass-alter")
                                    .and_then(|v| v.parse::<i32>().ok())
                                    .unwrap_or(0);
                                let bacc = match balter {
                                    1 => "#",
                                    2 => "##",
                                    -1 => "b",
                                    -2 => "bb",
                                    _ => "",
                                };
                                htxt.push('/');
                                htxt.push_str(&bstep);
                                htxt.push_str(bacc);
                            }
                            text_directions.push(TextDirectionEvent {
                                measure,
                                onset: super::musicxml_import::round3(
                                    cursor as f64 / divisions as f64,
                                ),
                                staff: gstaff,
                                voice: 1,
                                kind: "harmony".to_string(),
                                text: htxt,
                            });
                        }
                    }
                    rest = &rest[end..];
                } else if rest.starts_with("<note") {
                    let Some(end) = rest.find("</note>").map(|q| q + "</note>".len()) else {
                        break;
                    };
                    let blk = &rest[..end];
                    let is_chord = blk.contains("<chord/>") || blk.contains("<chord ");
                    let is_grace = blk.contains("<grace/>") || blk.contains("<grace ");
                    let dur = xml_tag_text(blk, "duration")
                        .and_then(|v| v.parse::<i64>().ok())
                        .unwrap_or(0);
                    let dur_beats = if is_grace {
                        sidecar_grace_beats(xml_tag_text(blk, "type").as_deref())
                    } else {
                        dur as f64 / divisions as f64
                    };
                    let onset_ticks = if is_chord { last_onset } else { cursor };
                    let onset =
                        super::musicxml_import::round3(onset_ticks as f64 / divisions as f64);
                    let local = xml_tag_text(blk, "staff")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(last_staff)
                        .max(1);
                    last_staff = local;
                    let gstaff = staff_base + local - 1;
                    let voice = xml_tag_text(blk, "voice")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(1);

                    if in_range {
                        for lyr in xml_blocks(blk, "lyric") {
                            let verse =
                                xml_attr(lyr, "number").unwrap_or_else(|| "1".to_string());
                            let syllabic = xml_tag_text(lyr, "syllabic");
                            let text = xml_tag_text(lyr, "text");
                            let extend = lyr.contains("<extend");
                            if text.is_some() || extend {
                                lyrics.push(LyricEvent {
                                    measure,
                                    onset,
                                    staff: gstaff,
                                    voice,
                                    verse,
                                    syllabic,
                                    text: text.unwrap_or_default(),
                                    extend,
                                });
                            }
                        }
                    }

                    if let Some(unp) = xml_blocks(blk, "unpitched").into_iter().next() {
                        excluded_staves.insert(gstaff);
                        if opts.include_percussion_tab && in_range {
                            let dstep = xml_tag_text(unp, "display-step")
                                .and_then(|s| s.chars().next())
                                .unwrap_or('C')
                                .to_ascii_uppercase();
                            let doct = xml_tag_text(unp, "display-octave")
                                .and_then(|v| v.parse::<i32>().ok())
                                .unwrap_or(4);
                            let notehead = xml_tag_text(blk, "notehead")
                                .map(|s| s.to_ascii_lowercase());
                            let inst_id = xml_blocks(blk, "instrument")
                                .into_iter()
                                .next()
                                .and_then(|el| xml_attr(el, "id"));
                            let instrument_name =
                                inst_id.as_ref().and_then(|id| inst_names.get(id).cloned());
                            percussion.push(PercussionEvent {
                                measure,
                                onset,
                                dur_beats: super::musicxml_import::round3(dur_beats),
                                staff: gstaff,
                                voice,
                                display_step: dstep,
                                display_octave: doct,
                                notehead,
                                instrument_id: inst_id,
                                instrument_name,
                                chord: is_chord,
                                grace: is_grace,
                            });
                        }
                    }

                    if let Some(tech) = xml_blocks(blk, "technical").into_iter().next() {
                        if let (Some(s_num), Some(f_num)) = (
                            xml_tag_text(tech, "string").and_then(|v| v.parse::<u32>().ok()),
                            xml_tag_text(tech, "fret").and_then(|v| v.parse::<u32>().ok()),
                        ) {
                            excluded_staves.insert(gstaff);
                            if opts.include_percussion_tab && in_range {
                                let pitch = xml_blocks(blk, "pitch").into_iter().next().map(|p| {
                                    let st = xml_tag_text(p, "step")
                                        .unwrap_or_else(|| "C".to_string());
                                    let al = xml_tag_text(p, "alter")
                                        .and_then(|v| v.parse::<i32>().ok())
                                        .unwrap_or(0);
                                    let oc = xml_tag_text(p, "octave")
                                        .unwrap_or_else(|| "4".to_string());
                                    let acc = match al {
                                        1 => "#",
                                        2 => "x",
                                        -1 => "b",
                                        -2 => "bb",
                                        _ => "",
                                    };
                                    format!("{st}{acc}{oc}")
                                });
                                tab.push(TabEvent {
                                    measure,
                                    onset,
                                    dur_beats: super::musicxml_import::round3(dur_beats),
                                    staff: gstaff,
                                    voice,
                                    string: s_num,
                                    fret: f_num,
                                    pitch,
                                    chord: is_chord,
                                    grace: is_grace,
                                });
                            }
                        }
                    }

                    if !is_chord {
                        last_onset = cursor;
                        if !is_grace {
                            cursor += dur;
                        }
                    }
                    rest = &rest[end..];
                } else {
                    match rest.find('>') {
                        Some(q) => rest = &rest[q + 1..],
                        None => break,
                    }
                }
            }
        }
        staff_base += part_staves;
    }

    Ok(GtCoverageSidecar {
        schema_version: GT_SIDECAR_SCHEMA_VERSION,
        include_percussion_tab: opts.include_percussion_tab,
        has_percussion_or_tab: !excluded_staves.is_empty(),
        excluded_staves: excluded_staves.into_iter().collect(),
        dynamics,
        hairpins,
        lyrics,
        text_directions,
        percussion,
        tab,
        slash_spans,
    })
}

#[cfg(test)]
mod tests {
    use super::super::musicxml_import::{BekernNote, StaffHeader};
    use super::duration_token;
    use super::{midi_of, to_bekern};

    #[test]
    fn duration_table_covers_dotted_values() {
        assert_eq!(duration_token(0.5), Some(("eighth", 0)));
        assert_eq!(duration_token(1.5), Some(("quarter", 1)));
        assert_eq!(duration_token(3.5), Some(("half", 2)));
        assert_eq!(duration_token(4.0), Some(("whole", 0)));
        assert_eq!(duration_token(8.0), Some(("breve", 0)));
        assert_eq!(duration_token(0.0625), Some(("64th", 0)));
        // Triplet quarter: not representable, never rounded.
        assert_eq!(duration_token(2.0 / 3.0), None);
    }

    #[test]
    fn duration_table_covers_128th_and_dotted_64th() {
        // Leland cull recovery (2026-11-22): 308 pages failed on 128th
        // notes (0.03125, below the 64th table floor) and 8 on dotted
        // 64ths (0.09375, the one missing dotted row).
        assert_eq!(duration_token(0.03125), Some(("128th", 0)));
        assert_eq!(duration_token(0.09375), Some(("64th", 1)));
        // Snap edges: within tolerance snaps, past it fails closed.
        assert_eq!(duration_token(0.03125 + 0.004), Some(("128th", 0)));
        assert_eq!(duration_token(0.03125 + 0.006), None);
        // Neighbors unchanged: 64th still wins its own cell.
        assert_eq!(duration_token(0.0625), Some(("64th", 0)));
        assert_eq!(duration_token(0.125), Some(("32nd", 0)));
    }

    #[test]
    fn duration_token_snaps_tick_quantized_triplets() {
        // suite31 tick-quantized triplet 16ths (divisions 10080):
        // cpebach-h186 m3 + weber-concertino m18 (1683 ticks),
        // schubert-lindenbaum m1 (1654 and 1693 ticks). True written
        // value is a 16th (0.25) in all three cases.
        assert_eq!(duration_token(5049.0 / 20160.0), Some(("16th", 0)));
        assert_eq!(duration_token(4962.0 / 20160.0), Some(("16th", 0)));
        assert_eq!(duration_token(5079.0 / 20160.0), Some(("16th", 0)));
        // Genuine non-dyadic values still fail closed: 0.26 sits 0.01
        // off the table, far past the snap tolerance.
        assert_eq!(duration_token(0.26), None);
    }

    #[allow(clippy::too_many_arguments)]
    fn note(
        measure: u32,
        onset: f64,
        step: char,
        octave: i32,
        alter: i32,
        dur: f64,
        voice: u32,
        staff: u32,
    ) -> BekernNote {
        BekernNote {
            measure,
            onset,
            pitch: Some((step, octave, alter)),
            dur_beats: dur,
            voice,
            staff,
            tie_start: false,
            tie_stop: false,
            grace: false,
            slur_start: false,
            slur_stop: false,
            articulations: Vec::new(),
            tuplet_ratio: None,
            tuplet_start: None,
            tuplet_stop: None,
            measure_rest: false,
            clef_change: None,
            key_change: None,
            time_change: None,
            dynamic: None,
            hairpin: None,
            barline_left: None,
            barline_right: None,
        }
    }

    #[test]
    fn token_stream_orders_and_marks_everything() {
        let headers = vec![
            StaffHeader {
                staff: 1,
                clef_sign: 'F',
                clef_line: 4,
                key_fifths: 0,
                beats: 4,
                beat_type: 4,
            },
            StaffHeader {
                staff: 0,
                clef_sign: 'G',
                clef_line: 2,
                key_fifths: 0,
                beats: 4,
                beat_type: 4,
            },
        ];
        let mut tied = note(1, 0.0, 'C', 4, 0, 1.0, 1, 0);
        tied.tie_start = true;
        let events = vec![
            // Unsorted input: rest on staff 1 sorts after staff-0 notes.
            BekernNote {
                measure: 1,
                onset: 0.0,
                pitch: None,
                dur_beats: 3.0,
                voice: 1,
                staff: 1,
                tie_start: false,
                tie_stop: false,
                grace: false,
                slur_start: false,
                slur_stop: false,
                articulations: Vec::new(),
                tuplet_ratio: None,
                tuplet_start: None,
                tuplet_stop: None,
                measure_rest: false,
                clef_change: None,
                key_change: None,
                time_change: None,
                dynamic: None,
                hairpin: None,
                barline_left: None,
                barline_right: None,
            },
            note(1, 0.0, 'E', 4, 1, 0.5, 1, 0),
            tied,
            note(1, 1.0, 'G', 4, 0, 1.5, 1, 0),
        ];
        let (toks, graces) = to_bekern(&events, &headers).unwrap();
        assert_eq!(graces, 0);
        assert_eq!(
            toks,
            vec![
                "clef-G-2",
                "key+0",
                "time-4-4",
                "clef-F-4",
                "key+0",
                "time-4-4",
                "voice-1",
                "quarter",
                "C",
                "4",
                "tie-start",
                "chord",
                "eighth",
                "E",
                "4",
                "a1",
                "voice-2",
                "rest",
                "half",
                "dot",
                "voice-1",
                "quarter",
                "G",
                "4",
                "dot",
                "bar",
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
        );
    }

    // Cross-staff triplet (ds0037 ordinal 39 skeleton): bracket 1 opens
    // on key A, middle members + stop live on key B, then key A plays a
    // plain rest and a plain quarter. The span must close right after
    // key A's gap fill so the trailing quarter decodes unscaled; the
    // bar must carry no tail stop and no reopen. Rests are unscored but
    // cursor-exactness is asserted through the note positions.
    #[test]
    fn cross_staff_bracket_closes_before_trailing_plain_quarter() {
        let headers = vec![
            StaffHeader {
                staff: 0,
                clef_sign: 'G',
                clef_line: 2,
                key_fifths: 0,
                beats: 4,
                beat_type: 4,
            },
            StaffHeader {
                staff: 1,
                clef_sign: 'G',
                clef_line: 2,
                key_fifths: 0,
                beats: 4,
                beat_type: 4,
            },
        ];
        let mut e4b = note(1, 0.0, 'E', 4, -1, 0.5, 1, 0);
        e4b.tuplet_ratio = Some((3, 2));
        e4b.tuplet_start = Some(1);
        let mut f3a = note(1, 0.333, 'F', 3, 0, 0.5, 1, 1);
        f3a.tuplet_ratio = Some((3, 2));
        let mut ab3a = note(1, 0.667, 'A', 3, -1, 0.5, 1, 1);
        ab3a.tuplet_ratio = Some((3, 2));
        ab3a.tuplet_stop = Some(1);
        let mut d4 = note(1, 1.0, 'D', 4, 0, 0.5, 1, 0);
        d4.tuplet_ratio = Some((3, 2));
        d4.tuplet_start = Some(1);
        let mut f3b = note(1, 1.333, 'F', 3, 0, 0.5, 1, 1);
        f3b.tuplet_ratio = Some((3, 2));
        let mut ab3b = note(1, 1.667, 'A', 3, -1, 0.5, 1, 1);
        ab3b.tuplet_ratio = Some((3, 2));
        ab3b.tuplet_stop = Some(1);
        let events = vec![
            e4b,
            f3a,
            ab3a,
            d4,
            f3b,
            ab3b,
            note(1, 2.0, 'R', 4, 0, 0.0, 1, 0),
            note(1, 3.0, 'F', 5, 0, 1.0, 1, 0),
        ];
        // Plain rest: pitch None.
        let mut evs = events;
        evs[6].pitch = None;
        evs[6].dur_beats = 1.0;
        let (toks, _) = to_bekern(&evs, &headers).unwrap();
        assert_eq!(
            toks,
            vec![
                "clef-G-2",
                "key+0",
                "time-4-4",
                "clef-G-2",
                "key+0",
                "time-4-4",
                "voice-1",
                "tuplet-start-3-2",
                "eighth",
                "E",
                "4",
                "a-1",
                "voice-2",
                "tuplet-start-3-2",
                "rest",
                "eighth",
                "tuplet-stop",
                "tuplet-start-3-2",
                "eighth",
                "F",
                "3",
                "eighth",
                "A",
                "3",
                "a-1",
                "tuplet-stop",
                "voice-1",
                "rest",
                "quarter",
                "eighth",
                "D",
                "4",
                "voice-2",
                "tuplet-start-3-2",
                "rest",
                "eighth",
                "tuplet-stop",
                "tuplet-start-3-2",
                "eighth",
                "F",
                "3",
                "eighth",
                "A",
                "3",
                "a-1",
                "tuplet-stop",
                "voice-1",
                "rest",
                "quarter",
                // Synthetic precise stop: key A's span closes after its
                // gap fill, so the rest and the trailing quarter below
                // decode unscaled.
                "tuplet-stop",
                "rest",
                "quarter",
                "quarter",
                "F",
                "5",
                "bar",
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn midi_of_matches_scorer_arithmetic() {
        assert_eq!(midi_of('C', 4, 0), 60);
        assert_eq!(midi_of('A', 3, 1), 58);
    }

    const TRIPLET: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>12</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="start" number="1"/></notations></note>
<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="stop" number="1"/></notations></note>
<note><rest/><duration>36</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn triplet_emits_written_durations_inside_span_tokens() {
        // Triplet eighths perform 4 ticks at divisions=12 (0.333 beats):
        // untypeable as-is. The written duration (4*3/2 = 6 ticks = an
        // eighth) plus tuplet span tokens must come out instead of the
        // whole page erroring.
        use super::super::musicxml_import::{bekern_events, staff_headers};
        let headers = staff_headers(TRIPLET).unwrap();
        let events = bekern_events(TRIPLET).unwrap();
        assert_eq!(events.len(), 4, "3 triplet notes + closing rest");
        let (toks, graces) = to_bekern(&events, &headers).unwrap();
        assert_eq!(graces, 0);
        assert_eq!(
            toks,
            vec![
                "clef-G-2",
                "key+0",
                "time-4-4",
                "voice-1",
                "tuplet-start-3-2",
                "eighth",
                "C",
                "4",
                "eighth",
                "D",
                "4",
                "eighth",
                "E",
                "4",
                "tuplet-stop",
                "rest",
                "half",
                "dot",
                "bar",
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn tuplet_stop_deferred_past_chord_run() {
        // The stop belongs to the whole chord: emitted after the run,
        // never between the root and its chord tone.
        use super::super::musicxml_import::{bekern_events, staff_headers};
        let headers = staff_headers(CHORD_STOP_TRIPLET).unwrap();
        let events = bekern_events(CHORD_STOP_TRIPLET).unwrap();
        let (toks, _) = to_bekern(&events, &headers).unwrap();
        assert_eq!(
            toks,
            vec![
                "clef-G-2",
                "key+0",
                "time-4-4",
                "voice-1",
                "tuplet-start-3-2",
                "eighth",
                "C",
                "4",
                "eighth",
                "D",
                "4",
                "eighth",
                "E",
                "4",
                "chord",
                "eighth",
                "G",
                "4",
                "tuplet-stop",
                "rest",
                "half",
                "dot",
                "bar",
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn repeated_stop_on_chord_tone_is_idempotent() {
        // Engraver repeated the stop on the chord tone: same tokens as
        // the single-stop page, no error.
        use super::super::musicxml_import::{bekern_events, staff_headers};
        let headers = staff_headers(REPEATED_STOP_TRIPLET).unwrap();
        let events = bekern_events(REPEATED_STOP_TRIPLET).unwrap();
        let (toks, _) = to_bekern(&events, &headers).unwrap();
        assert_eq!(
            toks,
            vec![
                "clef-G-2",
                "key+0",
                "time-4-4",
                "voice-1",
                "tuplet-start-3-2",
                "eighth",
                "C",
                "4",
                "eighth",
                "D",
                "4",
                "eighth",
                "E",
                "4",
                "chord",
                "eighth",
                "G",
                "4",
                "tuplet-stop",
                "rest",
                "half",
                "dot",
                "bar",
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn stray_stop_with_no_open_span_errors() {
        use super::super::musicxml_import::{bekern_events, staff_headers};
        let headers = staff_headers(STRAY_STOP).unwrap();
        let events = bekern_events(STRAY_STOP).unwrap();
        let err = to_bekern(&events, &headers).unwrap_err();
        assert!(
            err.contains("unmatched tuplet-stop"),
            "fail closed, got: {err}"
        );
    }

    fn plain_headers() -> Vec<StaffHeader> {
        vec![StaffHeader {
            staff: 0,
            clef_sign: 'G',
            clef_line: 2,
            key_fifths: 0,
            beats: 4,
            beat_type: 4,
        }]
    }

    fn toks(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn visual_bracket_without_time_mod_encodes_straight() {
        // Regression: ds0091 m37 part-2 staff-1 (frozen pages 3/5, spike
        // pages p2/p6). The engraved ticks are straight eighths under a
        // visual-only triplet bracket (markers, no time-modification);
        // opening a scaling span for it phantom-shifted every member
        // (0.167-beat "untypeable rest gap 0.166" lies). Straight in,
        // straight out: no tuplet tokens, no gap fills.
        let mut evs = vec![
            note(1, 0.0, 'D', 4, 0, 0.5, 1, 0),
            note(1, 0.0, 'F', 4, 0, 0.5, 1, 0),
            note(1, 0.0, 'C', 5, 0, 0.5, 1, 0),
            note(1, 0.5, 'E', 4, 0, 0.5, 1, 0),
            note(1, 0.5, 'G', 4, 0, 0.5, 1, 0),
            note(1, 0.5, 'C', 5, 0, 0.5, 1, 0),
            note(1, 1.0, 'F', 4, 0, 0.5, 1, 0),
            note(1, 1.0, 'A', 4, 0, 0.5, 1, 0),
            note(1, 1.0, 'C', 5, 0, 0.5, 1, 0),
        ];
        evs[0].tuplet_start = Some(1);
        evs[3].tuplet_stop = Some(1);
        let mut rest = note(1, 1.5, 'C', 4, 0, 0.5, 1, 0);
        rest.pitch = None;
        evs.push(rest);
        let (t, _) = to_bekern(&evs, &plain_headers()).unwrap();
        assert_eq!(
            t,
            toks(&[
                "clef-G-2", "key+0", "time-4-4", "voice-1",
                "eighth", "D", "4",
                "chord", "eighth", "F", "4",
                "chord", "eighth", "C", "5",
                "eighth", "E", "4",
                "chord", "eighth", "G", "4",
                "chord", "eighth", "C", "5",
                "eighth", "F", "4",
                "chord", "eighth", "A", "4",
                "chord", "eighth", "C", "5",
                "rest", "eighth", "bar",
            ])
        );
    }

    #[test]
    fn visual_bracket_stop_balances_start_without_tokens() {
        // Regression: ds0324-p4 split-m8. Four straight 16ths under a
        // visual-only bracket (start on F, stop on E, no time-mod). No
        // span opens, so no early close dangles: the stop balances the
        // pending visual start and emits nothing. The real eighth rest
        // before the run still fills explicitly.
        let mut evs = vec![
            note(1, 0.0, 'C', 4, 0, 0.5, 1, 0),
            note(1, 0.5, 'E', 4, 0, 0.5, 1, 0),
            note(1, 1.5, 'F', 4, 0, 0.25, 1, 0),
            note(1, 1.75, 'A', 4, 0, 0.25, 1, 0),
            note(1, 2.0, 'C', 5, 0, 0.25, 1, 0),
            note(1, 2.25, 'E', 4, 0, 0.25, 1, 0),
        ];
        evs[2].tuplet_start = Some(1);
        evs[5].tuplet_stop = Some(1);
        let (t, _) = to_bekern(&evs, &plain_headers()).unwrap();
        assert_eq!(
            t,
            toks(&[
                "clef-G-2", "key+0", "time-4-4", "voice-1",
                "eighth", "C", "4", "eighth", "E", "4",
                "rest", "eighth",
                "16th", "F", "4", "16th", "A", "4",
                "16th", "C", "5", "16th", "E", "4", "bar",
            ])
        );
    }

    #[test]
    fn truncated_bracket_stop_with_ratio_reopens_implicitly() {
        // Page-truncated real bracket: the opener lives off-page, but the
        // stopped note carries its ratio, so timing is exact. The span
        // opens implicitly (explicit start token) and closes through the
        // normal deferred path — balanced, never dangling.
        let mut e = note(1, 0.0, 'C', 4, 0, 0.5, 1, 0);
        e.tuplet_ratio = Some((3, 2));
        e.tuplet_stop = Some(1);
        let (t, _) = to_bekern(std::slice::from_ref(&e), &plain_headers()).unwrap();
        assert_eq!(
            t,
            toks(&[
                "clef-G-2", "key+0", "time-4-4", "voice-1",
                "tuplet-start-3-2", "eighth", "C", "4", "tuplet-stop", "bar",
            ])
        );
    }

    #[test]
    fn residual_dust_gap_snaps_to_64th_grid() {
        // Regression: ds0324-7 (0.065 written for a true 64th under a
        // 1.40625 nested scale). Sub-64th dust inside a real span snaps;
        // straight-time gaps and structural remainders stay loud.
        use super::emit_rest;
        let scale = 1.40625f64;
        let dust_tol = scale * 0.0015;
        let dust_cap = 4.0 * scale * 0.0015;
        let mut out = Vec::new();
        emit_rest(&mut out, 0.065, 7, dust_tol, Some(dust_cap)).unwrap();
        assert_eq!(out, toks(&["rest", "64th"]));
        // Straight time: identical gap fails closed.
        let mut out2 = Vec::new();
        let err = emit_rest(&mut out2, 0.065, 7, 0.0015, None).unwrap_err();
        assert!(err.contains("untypeable rest gap"), "got: {err}");
        assert!(out2.is_empty(), "failed emits stay atomic");
        // Structural remainder (0.166 -> 32nd + 0.041) stays loud even
        // with span context: the snap bound only covers dust.
        let mut out3 = Vec::new();
        let err = emit_rest(&mut out3, 0.166, 3, 0.0015, Some(4.0 * 0.0015)).unwrap_err();
        assert!(err.contains("untypeable rest gap"), "got: {err}");
        assert!(out3.is_empty(), "failed emits stay atomic");
    }

    #[test]
    fn rest_tiler_covers_128th_and_dotted_64th() {
        // Same Leland recovery as duration_token: exact 128th and
        // dotted-64th gaps tile instead of failing loudly.
        use super::emit_rest;
        let mut out = Vec::new();
        emit_rest(&mut out, 0.03125, 7, 0.0015, None).unwrap();
        assert_eq!(out, toks(&["rest", "128th"]));
        let mut out = Vec::new();
        emit_rest(&mut out, 0.09375, 7, 0.0015, None).unwrap();
        assert_eq!(out, toks(&["rest", "64th", "dot"]));
        // Off-grid remainder still fails closed (no partial fill).
        let mut out = Vec::new();
        let err = emit_rest(&mut out, 0.05, 7, 0.0015, None).unwrap_err();
        assert!(err.contains("untypeable rest gap"), "got: {err}");
        assert!(out.is_empty(), "failed emits stay atomic");
    }

    // Triplet whose final note is a two-tone chord (stop marker on the
    // root only, MuseScore style): the stop must come AFTER the whole
    // chord run, so the chord tone stays inside the bracket and scaled.
    const CHORD_STOP_TRIPLET: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>12</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="start" number="1"/></notations></note>
<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="stop" number="1"/></notations></note>
<note><chord/><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><rest/><duration>36</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    // Same page, but the engraver repeated the stop marker on the chord
    // tone too: idempotent like the start path (single stop after the
    // run), never a mid-chord flush or an error.
    const REPEATED_STOP_TRIPLET: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>12</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="start" number="1"/></notations></note>
<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="stop" number="1"/></notations></note>
<note><chord/><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="stop" number="1"/></notations></note>
<note><rest/><duration>36</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    // Stop marker with no open span (no time-mod anywhere): unbalanced
    // input must fail closed, like a number mismatch or a decode pop
    // of an empty stack — never a silent skip.
    const STRAY_STOP: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>12</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration><voice>1</voice><notations><tuplet type="stop" number="1"/></notations></note>
<note><rest/><duration>36</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    const ODD_DIV_TRIPLET: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>12</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="start" number="1"/></notations></note>
<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>7</duration><voice>1</voice><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="stop" number="1"/></notations></note>
</measure>
</part>
</score-partwise>"#;

    const BREVE_NOTE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>4</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>32</duration><voice>1</voice><type>breve</type></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn breve_note_emits_breve_token() {
        // 8-beat note (e.g. ds0349): decomposing into whole notes would
        // teach the decoder an arithmetic alias of the engraved glyph.
        use super::super::musicxml_import::{bekern_events, staff_headers};
        let headers = staff_headers(BREVE_NOTE).unwrap();
        let events = bekern_events(BREVE_NOTE).unwrap();
        assert_eq!(events.len(), 1);
        let (toks, _) = to_bekern(&events, &headers).unwrap();
        assert_eq!(
            toks,
            vec!["clef-G-2", "key+0", "time-4-4", "voice-1", "breve", "C", "4", "bar",]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn fractional_tick_triplet_recovers_exact_written_value() {
        // Double-dotted-eighth triplet member at divisions=12: performed
        // 7 ticks, written 7*3/(2*12) = 0.875 exactly in f64. A
        // tick-exactness gate would reject this (21 % 2 != 0); the float
        // quotient hits the duration table dead on.
        use super::super::musicxml_import::{bekern_events, staff_headers};
        let headers = staff_headers(ODD_DIV_TRIPLET).unwrap();
        let events = bekern_events(ODD_DIV_TRIPLET).unwrap();
        assert_eq!(events.len(), 3);
        let (toks, _) = to_bekern(&events, &headers).unwrap();
        assert_eq!(
            toks,
            vec![
                "clef-G-2",
                "key+0",
                "time-4-4",
                "voice-1",
                "tuplet-start-3-2",
                "eighth",
                "C",
                "4",
                "eighth",
                "D",
                "4",
                "eighth",
                "E",
                "4",
                "dot",
                "dot",
                "tuplet-stop",
                "bar",
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
        );
    }

    const BAR_REST_98: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>8</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>9</beats><beat-type>8</beat-type></time></attributes>
<note><rest measure="yes"/><duration>36</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn full_bar_rest_in_compound_meter_is_one_token() {
        // 9/8 whole-bar rest: 4.5 beats, no plain duration token.
        use super::super::musicxml_import::{bekern_events, staff_headers};
        let headers = staff_headers(BAR_REST_98).unwrap();
        let events = bekern_events(BAR_REST_98).unwrap();
        assert_eq!(events.len(), 1);
        let (toks, _) = to_bekern(&events, &headers).unwrap();
        assert_eq!(
            toks,
            vec!["clef-G-2", "key+0", "time-9-8", "voice-1", "bar-rest", "bar",]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }

    const FINE_DURATIONS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>128</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice><type>64th</type></note>
<note><pitch><step>D</step><octave>4</octave></pitch><duration>24</duration><voice>1</voice><type>32nd</type><dot/></note>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>28</duration><voice>1</voice><type>32nd</type><dot/><dot/></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn sixtyfourth_and_dotted_thirtyseconds_survive() {
        // 64ths (0.0625) and dotted 32nds (0.1875/0.21875) must not be
        // mangled by decimal rounding: the tokenizer matches exact tick
        // quotients (the old round3 turned 0.0625 into 0.062).
        use super::super::musicxml_import::{bekern_events, staff_headers};
        let headers = staff_headers(FINE_DURATIONS).unwrap();
        let events = bekern_events(FINE_DURATIONS).unwrap();
        assert_eq!(events.len(), 3);
        let (toks, _) = to_bekern(&events, &headers).unwrap();
        assert_eq!(
            toks,
            vec![
                "clef-G-2", "key+0", "time-4-4", "voice-1", "64th", "C", "4", "32nd", "D", "4",
                "dot", "32nd", "E", "4", "dot", "dot", "bar",
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn chord_tones_carry_an_explicit_marker() {
        // Same (measure, onset, staff, voice), both pitched: the second
        // tone must be marked — without it the token stream cannot tell
        // a chord from sequential same-voice notes.
        let headers = vec![StaffHeader {
            staff: 0,
            clef_sign: 'G',
            clef_line: 2,
            key_fifths: 0,
            beats: 4,
            beat_type: 4,
        }];
        let events = vec![
            note(1, 0.0, 'C', 4, 0, 1.0, 1, 0),
            note(1, 0.0, 'E', 4, 0, 1.0, 1, 0),
            note(1, 1.0, 'G', 4, 0, 1.0, 1, 0),
        ];
        let (toks, _) = to_bekern(&events, &headers).unwrap();
        assert_eq!(
            toks,
            vec![
                "clef-G-2", "key+0", "time-4-4", "voice-1", "quarter", "C", "4", "chord",
                "quarter", "E", "4", "quarter", "G", "4", "bar",
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn ds0001_page_one_tokenizes_to_golden_count() {
        // Golden pin: tokenizer output on real corpus data. Page-1 event
        // counts (179/11) triple-reconcile with ds0001-p1.json ScoreInput.
        // Breakdown, each verified against the MusicXML independently:
        // headers 3 staves x 3 = 9; pitched 168 x 3 base + 33 alter +
        // 22 dots (18 pitched + 4 on rests) = 559; plain rests 7 x 2 =
        // 14 (6 written + 1 gap-fill `rest quarter` for voice 2's
        // `<forward>`-covered entry silence in m1); whole-bar rests
        // (measure="yes") 5 x 1 = 5; `chord` markers 47 (one per 2nd+
        // chord tone); `voice-N` markers 128 (one per voice run: token
        // order interleaves voices by onset, max voice-6 here);
        // barlines 12 `bar` + 1 `bar-double` (measure 5 closes
        // light-light in the score); bekern+ marks 25 (5 slur-start
        // 2 fermata, dyn-p + dyn-mf, 3 hairpin-cresc-start +
        // 4 hairpin-dim-start + 4 hairpin-stop): 9 + 559 + 14 + 5 + 47
        // + 128 + 13 + 25 = 800.
        // The 5 bar rests previously emitted `rest quarter dot` (their
        // stored duration is a dotted quarter); `bar-rest` matches the
        // rendered bar-rest glyph. A re-render with different pagination
        // updates the 800.
        let xml =
            std::fs::read_to_string("../../data-gen/score-corpus/big500-renders/ds0001.musicxml")
                .expect("corpus fixture ds0001.musicxml");
        let breaks = super::super::musicxml_import::page_breaks(&xml);
        assert!(
            breaks.len() >= 2,
            "expected multi-page score, got {breaks:?}"
        );
        let p1_end = breaks[1].1 - 1;
        let headers = super::super::musicxml_import::staff_headers(&xml).unwrap();
        assert_eq!(headers.len(), 3, "page-1 staves");
        let events: Vec<super::super::musicxml_import::BekernNote> =
            super::super::musicxml_import::bekern_events(&xml)
                .unwrap()
                .into_iter()
                .filter(|e| e.measure <= p1_end)
                .collect();
        assert_eq!(events.len(), 179, "page-1 events vs ds0001-p1.json");
        assert_eq!(
            events.iter().filter(|e| e.pitch.is_none()).count(),
            11,
            "page-1 rests vs ds0001-p1.json"
        );
        let (toks, graces) = to_bekern(&events, &headers).unwrap();
        assert_eq!(graces, 0, "unexpected grace notes on ds0001-1");
        let bars = toks.iter().filter(|t| t.as_str() == "bar").count();
        assert_eq!(bars, 12, "plain barlines (measure 5 closes light-light)");
        assert_eq!(
            toks.iter().filter(|t| t.as_str() == "bar-double").count(),
            1,
            "measure-5 light-light close"
        );
        assert_eq!(
            toks.iter().filter(|t| t.as_str() == "bar-rest").count(),
            5,
            "whole-bar rests become one token each"
        );
        assert_eq!(
            toks.iter().filter(|t| t.as_str() == "chord").count(),
            47,
            "one marker per 2nd+ chord tone"
        );
        assert_eq!(
            toks.iter().filter(|t| t.starts_with("voice-")).count(),
            128,
            "one marker per voice run (reappearances re-marked)"
        );
        // bekern+ marks on page 1 (each verified against the score):
        // 5 slur runs, 2 fermatas, p + mf dynamics, 3 cresc + 4 dim
        // openings closed by 4 stops.
        for (tok, n) in [
            ("slur-start", 5),
            ("slur-stop", 5),
            ("fermata", 2),
            ("dyn-p", 1),
            ("dyn-mf", 1),
            ("hairpin-cresc-start", 3),
            ("hairpin-dim-start", 4),
            ("hairpin-stop", 4),
        ] {
            assert_eq!(
                toks.iter().filter(|t| t.as_str() == tok).count(),
                n,
                "bekern+ mark {tok}"
            );
        }
        assert_eq!(toks.len(), 800, "golden token total");
    }

    #[test]
    fn overflow_rest_preserves_attached_dynamic_and_hairpin() {
        // An invisible bar-overflow spacing rest (onset >= measure cap) is
        // skipped by to_bekern, but any dynamic/hairpin attached to it must
        // survive on the measure rather than being silently dropped.
        let mut overflow_rest = note(1, 4.0, 'C', 4, 0, 1.0, 1, 0);
        overflow_rest.pitch = None;
        overflow_rest.dynamic = Some("dyn-ff".to_string());
        overflow_rest.hairpin = Some("hairpin-stop".to_string());
        let events = vec![
            note(1, 0.0, 'C', 4, 0, 4.0, 1, 0),
            overflow_rest,
        ];
        let (toks, skipped) = to_bekern(&events, &plain_headers()).unwrap();
        assert_eq!(skipped, 1);
        assert!(toks.iter().any(|t| t == "dyn-ff"), "got: {toks:?}");
        assert!(toks.iter().any(|t| t == "hairpin-stop"), "got: {toks:?}");
    }

    #[test]
    fn sidecar_extracts_full_dynamics_hairpins_lyrics_and_text() {
        use super::{extract_gt_sidecar, extract_gt_sidecar_measures, GtCoverageOptions};
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
  <attributes><divisions>4</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
  <direction><direction-type><rehearsal>A</rehearsal></direction-type></direction>
  <direction><direction-type><words>dolce</words></direction-type></direction>
  <direction><direction-type><dynamics><ppp/></dynamics></direction-type></direction>
  <direction><direction-type><wedge type="crescendo" number="1"/></direction-type></direction>
  <harmony><root><root-step>C</root-step></root><kind text="maj7">major-seventh</kind></harmony>
  <note><pitch><step>C</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice>
    <lyric number="1"><syllabic>begin</syllabic><text>Hel</text></lyric>
    <lyric number="2"><syllabic>single</syllabic><text>La</text><extend/></lyric>
  </note>
  <direction><direction-type><wedge type="continue" number="1"/></direction-type></direction>
  <direction><direction-type><dynamics><f/></dynamics></direction-type></direction>
  <note><pitch><step>D</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice>
    <lyric number="1"><syllabic>end</syllabic><text>lo</text></lyric>
  </note>
</measure>
<measure number="2">
  <direction><direction-type><wedge type="stop" number="1"/></direction-type></direction>
  <direction><direction-type><dynamics><rfz/></dynamics></direction-type></direction>
  <note><pitch><step>E</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;
        let sc = extract_gt_sidecar(xml, &GtCoverageOptions::default()).unwrap();
        assert!(!sc.has_percussion_or_tab);
        assert!(sc.excluded_staves.is_empty());
        assert_eq!(sc.dynamics.len(), 3);
        assert_eq!(sc.dynamics[0].mark, "ppp");
        assert_eq!(sc.dynamics[0].bekern_token, None);
        assert_eq!(sc.dynamics[1].mark, "f");
        assert_eq!(sc.dynamics[1].bekern_token.as_deref(), Some("dyn-f"));
        assert_eq!(sc.dynamics[2].mark, "rfz");
        assert_eq!(sc.hairpins.len(), 3);
        assert_eq!(sc.hairpins[0].wedge_type, "crescendo");
        assert_eq!(sc.hairpins[1].wedge_type, "continue");
        assert_eq!(sc.hairpins[1].bekern_token, None);
        assert_eq!(sc.hairpins[2].wedge_type, "stop");
        assert_eq!(sc.lyrics.len(), 3);
        assert_eq!(sc.lyrics[0].text, "Hel");
        assert_eq!(sc.lyrics[0].syllabic.as_deref(), Some("begin"));
        assert_eq!(sc.lyrics[1].verse, "2");
        assert!(sc.lyrics[1].extend);
        assert_eq!(sc.lyrics[2].onset, 2.0);
        assert_eq!(sc.text_directions.len(), 3);
        assert!(sc.text_directions.iter().any(|t| t.kind == "rehearsal" && t.text == "A"));
        assert!(sc.text_directions.iter().any(|t| t.kind == "words" && t.text == "dolce"));
        assert!(sc.text_directions.iter().any(|t| t.kind == "harmony" && t.text == "C:maj7"));

        // Measure slicing keeps only measure 2 events.
        let m2 = extract_gt_sidecar_measures(xml, 2, 2, &GtCoverageOptions::default()).unwrap();
        assert_eq!(m2.dynamics.len(), 1);
        assert_eq!(m2.dynamics[0].mark, "rfz");
        assert_eq!(m2.lyrics.len(), 0);
    }

    #[test]
    fn sidecar_enforces_gate_d_boundary_for_percussion_and_tab() {
        use super::{extract_gt_sidecar, GtCoverageOptions};
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list>
  <score-part id="P1"><part-name>Drums</part-name>
    <score-instrument id="P1-I36"><instrument-name>Bass Drum 1</instrument-name></score-instrument>
    <score-instrument id="P1-I42"><instrument-name>Closed Hi-Hat</instrument-name></score-instrument>
  </score-part>
  <score-part id="P2"><part-name>Guitar TAB</part-name></score-part>
</part-list>
<part id="P1">
<measure number="1">
  <attributes><divisions>4</divisions><clef><sign>percussion</sign><line>2</line></clef><measure-style><slash type="start"/></measure-style></attributes>
  <note><unpitched><display-step>F</display-step><display-octave>4</display-octave></unpitched><duration>4</duration><instrument id="P1-I36"/><voice>1</voice></note>
  <note><chord/><unpitched><display-step>G</display-step><display-octave>5</display-octave></unpitched><duration>4</duration><instrument id="P1-I42"/><voice>1</voice><notehead>x</notehead></note>
</measure>
</part>
<part id="P2">
<measure number="1">
  <attributes><divisions>4</divisions><clef><sign>TAB</sign><line>5</line></clef></attributes>
  <note><pitch><step>E</step><octave>3</octave></pitch><duration>4</duration><voice>1</voice>
    <notations><technical><string>6</string><fret>0</fret></technical></notations>
  </note>
</measure>
</part>
</score-partwise>"#;
        // Default (Gate A-C): detects and excludes staves, emits empty payload.
        let def = extract_gt_sidecar(xml, &GtCoverageOptions::default()).unwrap();
        assert!(!def.include_percussion_tab);
        assert!(def.has_percussion_or_tab);
        assert_eq!(def.excluded_staves, vec![0, 1]);
        assert!(def.percussion.is_empty());
        assert!(def.tab.is_empty());
        assert!(def.slash_spans.is_empty());

        // Opt-in (Gate D): extracts full percussion, tab, and slash events.
        let opt = extract_gt_sidecar(
            xml,
            &GtCoverageOptions {
                include_percussion_tab: true,
            },
        )
        .unwrap();
        assert!(opt.include_percussion_tab);
        assert_eq!(opt.percussion.len(), 2);
        assert_eq!(opt.percussion[0].display_step, 'F');
        assert_eq!(opt.percussion[0].instrument_name.as_deref(), Some("Bass Drum 1"));
        assert!(opt.percussion[1].chord);
        assert_eq!(opt.percussion[1].notehead.as_deref(), Some("x"));
        assert_eq!(opt.tab.len(), 1);
        assert_eq!((opt.tab[0].string, opt.tab[0].fret), (6, 0));
        assert_eq!(opt.tab[0].pitch.as_deref(), Some("E3"));
        assert_eq!(opt.slash_spans.len(), 1);
        assert_eq!(opt.slash_spans[0].span_type, "start");
    }

    #[test]
    fn sidecar_extracts_real_lyrics_and_dynamics_from_ds0001() {
        use super::{extract_gt_sidecar, GtCoverageOptions};
        let xml =
            std::fs::read_to_string("../../data-gen/score-corpus/big500-renders/ds0001.musicxml")
                .expect("corpus fixture ds0001.musicxml");
        let sc = extract_gt_sidecar(&xml, &GtCoverageOptions::default()).unwrap();
        assert!(!sc.has_percussion_or_tab);
        assert!(!sc.lyrics.is_empty(), "ds0001 is a vocal lied with lyrics");
        assert!(!sc.dynamics.is_empty(), "ds0001 has dynamics");
        assert!(!sc.hairpins.is_empty(), "ds0001 has hairpins");
    }
}
