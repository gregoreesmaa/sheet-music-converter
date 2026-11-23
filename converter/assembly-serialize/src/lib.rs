//! assembly-serialize: flat note list -> MusicXML 3.1-first + loss table +
//! validators -> `.mscz` (MuseScore) zip.
//!
//! Pipeline:
//! 1. [`parse_score`] turns a [`ScoreInput`] (JSON note list) into a tick-based
//!    [`Score`]. Anything that cannot be represented exactly is kept going
//!    with a nearest approximation and recorded in the [`LossTable`].
//! 2. [`to_musicxml`] renders MusicXML 3.1 partwise (lossless except for the
//!    entries already in the loss table; barline crossings become ties).
//! 3. [`to_mscx`] renders a minimal MuseScore 3-compatible `.mscx` subset.
//!    Ties are re-articulated there (one [`LossEntry`] per tie) because the
//!    subset does not carry spanners.
//! 4. [`pack_mscz`] zips `META-INF/container.xml` + `*.mscx` into `.mscz`.
//! 5. [`validate_notes`], [`validate_musicxml`], [`validate_mscz`] check each
//!    stage; [`Issue`] lists use `error` / `warning` severities.
//!
//! All durations are integer ticks: `ticks = round(beats * divisions)` with
//! `divisions` (default 480) quarter-note ticks per quarter note.

use std::collections::BTreeMap;
use std::io::{Cursor, Write};

use serde::{Deserialize, Serialize};

pub mod bekern;
pub mod decode;
pub mod lattice;
pub mod musicxml_import;

// ---------------------------------------------------------------------------
// Input model (JSON note list)
// ---------------------------------------------------------------------------

/// House tick grid: every [`ScoreInput`] producer (mapper, MusicXML import)
/// emits this, so SER tick symbols compare across pipelines. Matches
/// MuseScore's native divisions.
pub const SCORE_DIVISIONS: u32 = 480;

fn default_divisions() -> u32 {
    SCORE_DIVISIONS
}

fn default_voice() -> u32 {
    1
}

/// Time signature, e.g. `{"beats": 6, "beat_type": 8}` for 6/8.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeSig {
    #[serde(default = "default_beats")]
    pub beats: u32,
    #[serde(default = "default_beat_type")]
    pub beat_type: u32,
}

fn default_beats() -> u32 {
    4
}
fn default_beat_type() -> u32 {
    4
}

/// One note (or rest) of the input list.
///
/// * `pitch`: scientific pitch like `"C4"`, `"F#5"`, `"Bb3"`, `"Gx2"`
///   (`x` = double sharp). A trailing `+NNc`/`-NNc` cents suffix
///   (e.g. `"A4+30c"`) is rounded to the nearest semitone and logged as loss.
/// * `rest: true` (or missing pitch with `rest: true`) marks a rest.
/// * `onset` / `dur` are in beats (quarter-note units).
/// * `voice` is 1-based; simultaneous same-voice onsets become chords.
/// * `grace`: a grace note (occupies no time; `dur` is the engraved
///   display value used for `<type>`). Renders with `<grace/>` and no
///   `<duration>`, so it scores as a zero-duration event.
/// * `slur_start` / `slur_stop`: slur spanner endpoints on this note.
/// * `articulations`: closed subset (`"stacc"`, `"accent"`, `"tenuto"`,
///   `"fermata"`), rendered into `<notations>`.
/// * `tie_start` / `tie_stop`: tie links (rendered as `<tie>`/`<tied>`;
///   link-faithful, never merged here — merging is the caller's choice).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteInput {
    #[serde(default)]
    pub pitch: Option<String>,
    #[serde(default)]
    pub rest: bool,
    pub onset: f64,
    pub dur: f64,
    #[serde(default = "default_voice")]
    pub voice: u32,
    #[serde(default)]
    pub lyric: Option<String>,
    #[serde(default)]
    pub grace: bool,
    #[serde(default)]
    pub slur_start: bool,
    #[serde(default)]
    pub slur_stop: bool,
    #[serde(default)]
    pub articulations: Vec<String>,
    #[serde(default)]
    pub tie_start: bool,
    #[serde(default)]
    pub tie_stop: bool,
}

/// One named part (staff system) holding an unordered note list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartInput {
    #[serde(default = "default_part_name")]
    pub name: String,
    #[serde(default)]
    pub notes: Vec<NoteInput>,
}

fn default_part_name() -> String {
    "Part 1".to_string()
}

/// Measure-scoped notation state changes (bekern+ mid-stream headers,
/// dynamics, hairpins, barlines). `measure` is 1-based. Every field is
/// a last-wins override for that measure; absent fields inherit the
/// score state. Short spellings: `dynamic` is one of
/// `pp|p|mp|mf|f|ff|sfz`, `hairpin` one of `cresc|dim|stop`,
/// `barline_left` is `repeat-left`, `barline_right` one of
/// `light-heavy|repeat-right|double`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeasureAttr {
    pub measure: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clef_sign: Option<char>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clef_line: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_fifths: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time: Option<TimeSig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hairpin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub barline_left: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub barline_right: Option<String>,
}

/// Top-level JSON document. Either `parts` or the flat `notes` shorthand
/// (which becomes one part named `part_name`) must be non-empty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreInput {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub composer: Option<String>,
    #[serde(default = "default_divisions")]
    pub divisions: u32,
    #[serde(default)]
    pub time: Option<TimeSig>,
    /// Key signature in fifths: 0 = C major, 1 = G, -1 = F, ...
    #[serde(default)]
    pub key_fifths: i32,
    #[serde(default)]
    pub parts: Vec<PartInput>,
    #[serde(default)]
    pub notes: Vec<NoteInput>,
    #[serde(default)]
    pub part_name: Option<String>,
    /// Measure-scoped overrides (see [`MeasureAttr`]); empty when the
    /// score carries none.
    #[serde(default)]
    pub measure_attrs: Vec<MeasureAttr>,
}

// ---------------------------------------------------------------------------
// Internal model (integer ticks)
// ---------------------------------------------------------------------------

/// A pitch or rest with absolute tick positions.
///
/// `grace` notes are zero-width (they occupy no time): `dur_ticks`
/// carries the engraved display value for `<type>` only, and layout
/// never advances the cursor for them.
#[derive(Debug, Clone)]
pub struct ScoreNote {
    pub step: char,
    pub alter: i32,
    pub octave: i32,
    pub midi: i32,
    pub start_ticks: i64,
    pub dur_ticks: i64,
    pub voice: u32,
    pub rest: bool,
    pub lyric: Option<String>,
    pub grace: bool,
    pub slur_start: bool,
    pub slur_stop: bool,
    pub articulations: Vec<String>,
    pub tie_start: bool,
    pub tie_stop: bool,
}

/// Simultaneous same-voice notes sharing one onset/duration.
#[derive(Debug, Clone)]
pub struct ChordGroup {
    pub start_ticks: i64,
    pub dur_ticks: i64,
    pub voice: u32,
    pub members: Vec<ScoreNote>,
}

#[derive(Debug, Clone)]
pub struct Part {
    pub name: String,
    pub groups: Vec<ChordGroup>,
    pub voices: Vec<u32>,
    pub end_ticks: i64,
}

/// Fully resolved score plus the loss table accumulated while building it.
#[derive(Debug, Clone)]
pub struct Score {
    pub title: String,
    pub composer: Option<String>,
    pub divisions: u32,
    pub beats: u32,
    pub beat_type: u32,
    pub key_fifths: i32,
    pub parts: Vec<Part>,
    pub measure_attrs: Vec<MeasureAttr>,
}

impl Score {
    pub fn measure_ticks(&self) -> i64 {
        self.beats as i64 * self.divisions as i64 * 4 / self.beat_type as i64
    }

    /// Per-measure `[start_ticks, end_ticks)` windows respecting any
    /// mid-stream time-signature changes in `measure_attrs`.
    pub fn measure_windows(&self) -> Vec<(i64, i64)> {
        let end = self.parts.iter().map(|p| p.end_ticks).max().unwrap_or(0).max(1);
        let mut windows = Vec::new();
        let mut cur = 0i64;
        let mut beats = self.beats;
        let mut beat_type = self.beat_type;
        let mut m = 1u32;
        while cur < end {
            if let Some(t) = measure_attr(self, m).and_then(|a| a.time) {
                beats = t.beats;
                beat_type = t.beat_type;
            }
            let mlen = (beats as i64 * self.divisions as i64 * 4 / beat_type.max(1) as i64).max(1);
            windows.push((cur, cur + mlen));
            cur += mlen;
            m += 1;
        }
        windows
    }

    pub fn measure_count(&self) -> usize {
        self.measure_windows().len()
    }
}

// ---------------------------------------------------------------------------
// Loss table
// ---------------------------------------------------------------------------

/// One lossy (or informational) approximation. `severity` is `"loss"` when
/// the output no longer encodes the input exactly, `"info"` for lossless
/// normalisation (gap rests, sorting) worth knowing about.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossEntry {
    pub severity: String,
    pub kind: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note_index: Option<usize>,
}

/// Ordered loss/info log for one conversion run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LossTable {
    pub entries: Vec<LossEntry>,
}

impl LossTable {
    fn push(
        &mut self,
        severity: &str,
        kind: &str,
        message: String,
        part: Option<String>,
        note_index: Option<usize>,
    ) {
        self.entries.push(LossEntry {
            severity: severity.to_string(),
            kind: kind.to_string(),
            message,
            part,
            note_index,
        });
    }

    pub fn loss_count(&self) -> usize {
        self.entries.iter().filter(|e| e.severity == "loss").count()
    }

    pub fn is_lossless(&self) -> bool {
        self.loss_count() == 0
    }
}

// ---------------------------------------------------------------------------
// Validation issues
// ---------------------------------------------------------------------------

/// One validator finding; `severity` is `"error"` or `"warning"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    pub severity: String,
    pub code: String,
    pub message: String,
}

fn error(code: &str, message: String) -> Issue {
    Issue {
        severity: "error".to_string(),
        code: code.to_string(),
        message,
    }
}

fn warning(code: &str, message: String) -> Issue {
    Issue {
        severity: "warning".to_string(),
        code: code.to_string(),
        message,
    }
}

pub fn has_errors(issues: &[Issue]) -> bool {
    issues.iter().any(|i| i.severity == "error")
}

// ---------------------------------------------------------------------------
// Pitch parsing
// ---------------------------------------------------------------------------

fn step_semitone(step: char) -> i32 {
    match step {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => 0,
    }
}

/// Sharp-spelled `(step, alter)` for a pitch class.
fn pitch_class_to_step_alter(pc: i32) -> (char, i32) {
    match pc.rem_euclid(12) {
        0 => ('C', 0),
        1 => ('C', 1),
        2 => ('D', 0),
        3 => ('D', 1),
        4 => ('E', 0),
        5 => ('F', 0),
        6 => ('F', 1),
        7 => ('G', 0),
        8 => ('G', 1),
        9 => ('A', 0),
        10 => ('A', 1),
        _ => ('B', 0),
    }
}

/// Parse `"F#5"`, `"Bb3"`, `"Gx2"`, `"A4+30c"`.
/// Returns `(step, alter, octave, midi, cents_rounded: Option<i32>)`.
fn parse_pitch(s: &str) -> Result<(char, i32, i32, i32, Option<i32>), String> {
    let t = s.trim();
    let mut it = t.chars();
    let step = it
        .next()
        .ok_or_else(|| format!("empty pitch {s:?}"))?
        .to_ascii_uppercase();
    if !"ABCDEFG".contains(step) {
        return Err(format!("bad step in pitch {s:?}"));
    }
    let mut rest = it.as_str();
    let mut alter = 0i32;
    let mut sharp_count = 0;
    let mut flat_count = 0;
    loop {
        if let Some(r) = rest.strip_prefix('#') {
            sharp_count += 1;
            alter += 1;
            rest = r;
        } else if let Some(r) = rest.strip_prefix('x').or_else(|| rest.strip_prefix('X')) {
            alter += 2;
            rest = r;
        } else if let Some(r) = rest.strip_prefix('b').or_else(|| rest.strip_prefix('B')) {
            // 'B' as accidental only valid after a step letter, which we
            // already consumed, so a bare 'B' here is a flat sign.
            flat_count += 1;
            alter -= 1;
            rest = r;
        } else if let Some(r) = rest.strip_prefix('n').or_else(|| rest.strip_prefix('N')) {
            rest = r;
        } else {
            break;
        }
    }
    if sharp_count > 2 {
        return Err(format!("triple sharp unsupported in pitch {s:?}"));
    }
    if flat_count > 2 {
        return Err(format!("triple flat unsupported in pitch {s:?}"));
    }
    // Octave: optional '-', then digits.
    let mut dig = String::new();
    let mut tail = rest;
    if let Some(r) = tail.strip_prefix('-') {
        dig.push('-');
        tail = r;
    } else if let Some(r) = tail.strip_prefix('+') {
        // A leading '+' starts a cents suffix only when no digits precede;
        // here it means the octave is missing -> parse digits below fails.
        tail = r;
    }
    while let Some(c) = tail.chars().next() {
        if c.is_ascii_digit() {
            dig.push(c);
            tail = &tail[1..];
        } else {
            break;
        }
    }
    if dig.is_empty() || dig == "-" {
        return Err(format!("missing octave in pitch {s:?}"));
    }
    let octave: i32 = dig
        .parse()
        .map_err(|_| format!("bad octave in pitch {s:?}"))?;
    if !(-1..=9).contains(&octave) {
        return Err(format!("octave out of range in pitch {s:?}"));
    }
    let mut midi = (octave + 1) * 12 + step_semitone(step) + alter;
    let mut cents = None;
    let tail_t = tail.trim();
    if !tail_t.is_empty() {
        // Cents suffix: "[+-]NN[c]", e.g. "+30c", "-45".
        let (num, ok) = match tail_t
            .strip_suffix('c')
            .or_else(|| tail_t.strip_suffix('C'))
        {
            Some(n) => (n, true),
            None => (tail_t, tail_t.starts_with('+') || tail_t.starts_with('-')),
        };
        let c: i32 = if ok {
            num.parse()
                .map_err(|_| format!("bad cents suffix in pitch {s:?}"))?
        } else {
            return Err(format!("trailing garbage in pitch {s:?}"));
        };
        if c.abs() >= 100 {
            return Err(format!("cents out of range in pitch {s:?}"));
        }
        if c != 0 {
            cents = Some(c);
            midi = ((midi as f64) + (c as f64) / 100.0).round() as i32;
        }
    }
    if !(0..=127).contains(&midi) {
        return Err(format!("pitch {s:?} out of MIDI range (got {midi})"));
    }
    if cents.is_some() {
        let (st, al) = pitch_class_to_step_alter(midi.rem_euclid(12));
        let oct = midi.div_euclid(12) - 1;
        return Ok((st, al, oct, midi, cents));
    }
    Ok((step, alter, octave, midi, None))
}

// ---------------------------------------------------------------------------
// Duration -> MusicXML type + dots
// ---------------------------------------------------------------------------

/// All (type name, base ticks, dots 0..=3) candidates for `divisions`.
///
/// A base of `divisions / 2^k` ticks is only exact when `divisions` is a
/// multiple of `2^k`; otherwise that type (and shorter ones) cannot be
/// written exactly and is skipped, so quantisation records real loss.
fn duration_candidates(divisions: u32) -> Vec<(&'static str, i64, u8, i64)> {
    let d = divisions as i64;
    // (type name, divisor of divisions for one base unit)
    let bases: &[(&str, i64)] = &[
        ("whole", -4), // whole = 4 * divisions (negative = multiply)
        ("half", -2),
        ("quarter", 1),
        ("eighth", 2),
        ("16th", 4),
        ("32nd", 8),
        ("64th", 16),
    ];
    let mut out = Vec::new();
    for (name, unit) in bases {
        let base = if *unit < 0 {
            d * -unit
        } else {
            if d % *unit != 0 {
                continue;
            }
            d / *unit
        };
        if base <= 0 {
            continue;
        }
        for dots in 0u8..=3 {
            // value = base * (2 - 2^-dots)
            let num = base * ((1 << (dots + 1)) - 1);
            let den = 1 << dots;
            if num % den != 0 {
                continue;
            }
            out.push((*name, base, dots, num / den));
        }
    }
    out
}

/// Map tick duration to `(type name, dots, actual ticks)`.
/// `actual != ticks` means the caller must log `duration-quantized` loss.
pub fn quantize_duration(ticks: i64, divisions: u32) -> (&'static str, u8, i64) {
    let mut best: Option<(&'static str, u8, i64)> = None;
    let mut best_dist = i64::MAX;
    let mut best_dots = u8::MAX;
    for (name, _base, dots, value) in duration_candidates(divisions) {
        if value <= 0 {
            continue;
        }
        let dist = (value - ticks).abs();
        if dist < best_dist || (dist == best_dist && dots < best_dots) {
            best_dist = dist;
            best_dots = dots;
            best = Some((name, dots, value));
        }
    }
    best.unwrap_or(("64th", 0, divisions as i64 / 16))
}

// ---------------------------------------------------------------------------
// Score assembly
// ---------------------------------------------------------------------------

/// True when the input carries no notes at all (mirrors [`parse_score`]'s
/// `"score has no notes"` gate, without string-matching it): either `parts`
/// is non-empty-but-all-empty, or the flat `notes` shorthand is empty.
/// Scoring CLIs use this to route empty references to the exact-match
/// column instead of erroring.
pub fn is_empty_score_input(input: &ScoreInput) -> bool {
    if !input.parts.is_empty() {
        input.parts.iter().all(|p| p.notes.is_empty())
    } else {
        input.notes.is_empty()
    }
}

/// Count pitched (non-rest) chord members across all parts. Empty
/// references (zero here) score through the exact-match column, never a
/// bare F1 float; a zero here with a non-empty reference stays an
/// included F1 = 0.0.
pub fn score_pitched_count(score: &Score) -> usize {
    score
        .parts
        .iter()
        .flat_map(|p| p.groups.iter())
        .flat_map(|g| g.members.iter())
        .filter(|m| !m.rest)
        .count()
}

/// Build a [`Score`] + [`LossTable`] from validated-or-raw input.
///
/// Hard errors (`Err`): non-positive divisions/time, negative onsets,
/// non-positive durations, unparseable or out-of-range pitches, empty score.
/// Approximations (`LossTable`): microtone rounding, duration quantisation,
/// overlap clamping/drops, inserted gap rests.
pub fn parse_score(input: &ScoreInput) -> Result<(Score, LossTable), String> {
    if input.divisions == 0 {
        return Err("divisions must be positive".to_string());
    }
    let beats = input.time.map(|t| t.beats).unwrap_or(4);
    let beat_type = input.time.map(|t| t.beat_type).unwrap_or(4);
    if beats == 0 || beat_type == 0 {
        return Err("time beats and beat_type must be positive".to_string());
    }
    if !matches!(beat_type, 1 | 2 | 4 | 8 | 16 | 32) {
        return Err(format!("unsupported beat_type {beat_type}"));
    }

    let raw_parts: Vec<(String, &[NoteInput])> = if !input.parts.is_empty() {
        input
            .parts
            .iter()
            .map(|p| {
                (
                    if p.name.is_empty() {
                        "Part 1".to_string()
                    } else {
                        p.name.clone()
                    },
                    p.notes.as_slice(),
                )
            })
            .collect()
    } else {
        vec![(
            input
                .part_name
                .clone()
                .unwrap_or_else(|| "Part 1".to_string()),
            input.notes.as_slice(),
        )]
    };
    if raw_parts.iter().all(|(_, n)| n.is_empty()) {
        return Err("score has no notes".to_string());
    }

    let mut losses = LossTable::default();
    let mut parts = Vec::new();
    for (pname, notes) in &raw_parts {
        if notes.is_empty() {
            continue;
        }
        // Materialise tick notes.
        let mut tick_notes: Vec<(usize, ScoreNote)> = Vec::new();
        for (i, n) in notes.iter().enumerate() {
            if !n.onset.is_finite() || !n.dur.is_finite() {
                return Err(format!("part {pname:?} note {i}: onset/dur must be finite"));
            }
            if n.onset < 0.0 {
                return Err(format!(
                    "part {pname:?} note {i}: negative onset {}",
                    n.onset
                ));
            }
            if n.dur <= 0.0 {
                return Err(format!(
                    "part {pname:?} note {i}: non-positive dur {}",
                    n.dur
                ));
            }
            if n.voice == 0 {
                return Err(format!("part {pname:?} note {i}: voice must be >= 1"));
            }
            let start = (n.onset * input.divisions as f64).round() as i64;
            let dur = (n.dur * input.divisions as f64).round() as i64;
            if dur <= 0 {
                return Err(format!(
                    "part {pname:?} note {i}: duration {} beats rounds to zero ticks at divisions {}",
                    n.dur, input.divisions
                ));
            }
            let is_rest = n.rest
                || n.pitch
                    .as_ref()
                    .map(|p| p.trim().is_empty())
                    .unwrap_or(true);
            if is_rest {
                tick_notes.push((
                    i,
                    ScoreNote {
                        step: 'C',
                        alter: 0,
                        octave: 4,
                        midi: 60,
                        start_ticks: start,
                        dur_ticks: dur,
                        voice: n.voice,
                        rest: true,
                        lyric: n.lyric.clone(),
                        grace: n.grace,
                        slur_start: n.slur_start,
                        slur_stop: n.slur_stop,
                        articulations: n.articulations.clone(),
                        tie_start: n.tie_start,
                        tie_stop: n.tie_stop,
                    },
                ));
                continue;
            }
            let pitch = n.pitch.clone().unwrap();
            let (mut step, mut alter, mut octave, midi, cents) =
                parse_pitch(&pitch).map_err(|e| format!("part {pname:?} note {i}: {e}"))?;
            if let Some(c) = cents {
                losses.push(
                    "loss",
                    "pitch-microtone-rounded",
                    format!("pitch {pitch:?} carries {c:+} cents; rounded to MIDI {midi}"),
                    Some(pname.clone()),
                    Some(i),
                );
                let (st, al) = pitch_class_to_step_alter(midi.rem_euclid(12));
                step = st;
                alter = al;
                octave = midi.div_euclid(12) - 1;
            }
            // Note: `dur` stays exact in ticks. Only the displayed
            // MusicXML <type> is approximated at render time (logged there);
            // <duration> always carries the exact tick count, so measure
            // sums hold exactly.
            tick_notes.push((
                i,
                ScoreNote {
                    step,
                    alter,
                    octave,
                    midi,
                    start_ticks: start,
                    dur_ticks: dur,
                    voice: n.voice,
                    rest: false,
                    lyric: n.lyric.clone(),
                    grace: n.grace,
                    slur_start: n.slur_start,
                    slur_stop: n.slur_stop,
                    articulations: n.articulations.clone(),
                    tie_start: n.tie_start,
                    tie_stop: n.tie_stop,
                },
            ));
        }

        // Sort by (start, voice, grace-first, midi); input order need
        // not be sorted. Grace leads its onset so the scorer reads it
        // at the pre-note cursor, matching document order on the way
        // in (a grace sorted after its note would read a full note
        // later). `sort_by_key` is stable, so remaining ties keep
        // input order.
        tick_notes.sort_by_key(|(_, n)| (n.start_ticks, n.voice, !n.grace, n.midi));

        // Group simultaneous same-voice notes into chords; clamp overlaps.
        // Grace notes are zero-width: they never clamp, drop, or advance
        // the voice cursor, and chord-merge only with matching grace-ness
        // (a real note sharing a grace onset is sequential, not a chord).
        let mut groups: Vec<ChordGroup> = Vec::new();
        let mut cursor: BTreeMap<u32, i64> = BTreeMap::new();
        for (idx, mut n) in tick_notes {
            if n.grace {
                let merge = match groups.last() {
                    Some(g) => {
                        !g.members.is_empty()
                            && g.members[0].grace
                            && g.voice == n.voice
                            && g.start_ticks == n.start_ticks
                            && g.members[0].dur_ticks == n.dur_ticks
                    }
                    None => false,
                };
                if merge {
                    groups.last_mut().unwrap().members.push(n);
                } else {
                    groups.push(ChordGroup {
                        start_ticks: n.start_ticks,
                        dur_ticks: 0,
                        voice: n.voice,
                        members: vec![n],
                    });
                }
                continue;
            }
            // Chord check first: a note starting exactly with the previous
            // group in the same voice extends the chord, it is not overlap.
            if !n.rest {
                if let Some(g) = groups.last() {
                    if !g.members[0].rest
                        && !g.members[0].grace
                        && g.voice == n.voice
                        && g.start_ticks == n.start_ticks
                        && g.dur_ticks == n.dur_ticks
                    {
                        groups.last_mut().unwrap().members.push(n.clone());
                        continue;
                    }
                }
            }
            let cur = *cursor.get(&n.voice).unwrap_or(&0);
            if n.start_ticks < cur {
                let shift = cur - n.start_ticks;
                if n.dur_ticks <= shift {
                    losses.push(
                        "loss",
                        "note-dropped-overlap",
                        format!(
                            "note fully covered by an earlier voice-{} note; dropped",
                            n.voice
                        ),
                        Some(pname.clone()),
                        Some(idx),
                    );
                    continue;
                }
                losses.push(
                    "loss",
                    "overlap-clamped",
                    format!(
                        "onset moved +{shift} ticks to resolve voice-{} overlap",
                        n.voice
                    ),
                    Some(pname.clone()),
                    Some(idx),
                );
                n.start_ticks = cur;
                n.dur_ticks -= shift;
            }
            let end = n.start_ticks + n.dur_ticks;
            cursor.insert(n.voice, end);
            groups.push(ChordGroup {
                start_ticks: n.start_ticks,
                dur_ticks: n.dur_ticks,
                voice: n.voice,
                members: vec![n],
            });
        }

        let end_ticks = groups
            .iter()
            .map(|g| g.start_ticks + g.dur_ticks)
            .max()
            .unwrap_or(0);
        let mut voices: Vec<u32> = groups.iter().map(|g| g.voice).collect();
        voices.sort();
        voices.dedup();
        parts.push(Part {
            name: pname.clone(),
            groups,
            voices,
            end_ticks,
        });
    }

    Ok((
        Score {
            title: input
                .title
                .clone()
                .unwrap_or_else(|| "Untitled".to_string()),
            composer: input.composer.clone(),
            divisions: input.divisions,
            beats,
            beat_type,
            key_fifths: input.key_fifths,
            parts,
            measure_attrs: input.measure_attrs.clone(),
        },
        losses,
    ))
}

// ---------------------------------------------------------------------------
// MusicXML 3.1 rendering
// ---------------------------------------------------------------------------

/// XML-escape text content.
pub fn escape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

fn accidental_name(alter: i32) -> Option<&'static str> {
    match alter {
        1 => Some("sharp"),
        2 => Some("double-sharp"),
        -1 => Some("flat"),
        -2 => Some("flat-flat"),
        0 => None,
        _ => None,
    }
}

/// One rendered note/rest segment inside a measure.
struct Segment {
    note: ScoreNote,
    chord: bool, // <chord/> marker (non-first chord member)
    dur_ticks: i64,
    tie_in: bool,  // continues a previous segment -> <tie type="stop"/>
    tie_out: bool, // continues into the next segment -> <tie type="start"/>
}

/// Split a part's chord groups into per-measure, per-voice segments.
/// Groups crossing a barline are split with ties (lossless). Gaps are
/// filled with rests; every voice sums to exactly `measure_ticks`.
fn layout_part(
    score: &Score,
    part: &Part,
    losses: &mut LossTable,
) -> Vec<Vec<(u32, Vec<Segment>)>> {
    let windows = score.measure_windows();
    // voice -> sorted (start, end, member_idx, group_idx)
    let mut result: Vec<Vec<(u32, Vec<Segment>)>> = Vec::new();
    for (m, &(m0, m1)) in windows.iter().enumerate() {
        let mut voices: Vec<(u32, Vec<Segment>)> = Vec::new();
        for v in &part.voices {
            let mut segs: Vec<Segment> = Vec::new();
            for g in part.groups.iter().filter(|g| g.voice == *v) {
                // Grace groups are zero-width: they render in the one
                // bar window containing their onset (a grace exactly on
                // a barline belongs to the following measure) and never
                // split.
                let is_grace = g.members.first().map_or(false, |m| m.grace);
                if is_grace && (g.start_ticks < m0 || g.start_ticks >= m1) {
                    continue;
                }
                let gs = g.start_ticks.max(m0);
                let ge = if is_grace {
                    gs
                } else {
                    (g.start_ticks + g.dur_ticks).min(m1)
                };
                if ge <= gs && !is_grace {
                    continue;
                }
                let total = g.start_ticks + g.dur_ticks;
                for (mi, member) in g.members.iter().enumerate() {
                    segs.push(Segment {
                        note: member.clone(),
                        chord: mi > 0,
                        dur_ticks: ge - gs,
                        tie_in: gs > g.start_ticks,
                        tie_out: ge < total,
                    });
                }
            }
            segs.sort_by_key(|s| (s.note.start_ticks.max(m0), s.chord));
            // Fill gaps with rests so the voice sums to mlen.
            let mut filled: Vec<Segment> = Vec::new();
            let mut cursor = m0;
            let mut rest_inserted = false;
            for s in segs {
                let ss = s.note.start_ticks.max(m0);
                if ss > cursor {
                    filled.push(Segment {
                        note: ScoreNote {
                            step: 'C',
                            alter: 0,
                            octave: 4,
                            midi: 60,
                            start_ticks: cursor,
                            dur_ticks: ss - cursor,
                            voice: *v,
                            rest: true,
                            lyric: None,
                            grace: false,
                            slur_start: false,
                            slur_stop: false,
                            articulations: Vec::new(),
                            tie_start: false,
                            tie_stop: false,
                        },
                        chord: false,
                        dur_ticks: ss - cursor,
                        tie_in: false,
                        tie_out: false,
                    });
                    rest_inserted = true;
                }
                cursor = cursor.max(ss + s.dur_ticks);
                filled.push(s);
            }
            if cursor < m1 {
                filled.push(Segment {
                    note: ScoreNote {
                        step: 'C',
                        alter: 0,
                        octave: 4,
                        midi: 60,
                        start_ticks: cursor,
                        dur_ticks: m1 - cursor,
                        voice: *v,
                        rest: true,
                        lyric: None,
                        grace: false,
                        slur_start: false,
                        slur_stop: false,
                        articulations: Vec::new(),
                        tie_start: false,
                        tie_stop: false,
                    },
                    chord: false,
                    dur_ticks: m1 - cursor,
                    tie_in: false,
                    tie_out: false,
                });
                rest_inserted = true;
            }
            if rest_inserted {
                losses.push(
                    "info",
                    "rest-inserted",
                    format!("voice {v} measure {}: gap filled with rests", m + 1),
                    Some(part.name.clone()),
                    None,
                );
            }
            voices.push((*v, filled));
        }
        result.push(voices);
    }
    result
}

/// `<tie>` elements for one segment: barline-split articulation wins;
/// note-level tie flags render only when the segment is unsplit (a
/// split link already carries its own ties; doubling them would fake
/// extra segments to the scorer).
fn tie_elements(seg: &Segment) -> (bool, bool) {
    if seg.tie_in || seg.tie_out {
        (seg.tie_out, seg.tie_in)
    } else {
        (seg.note.tie_start, seg.note.tie_stop)
    }
}

/// `<tied>` elements for one segment (same sourcing as [`tie_elements`]).
fn tied_xml(seg: &Segment) -> String {
    let (start, stop) = tie_elements(seg);
    let mut out = String::new();
    if start {
        out.push_str("<tied type=\"start\"/>");
    }
    if stop {
        out.push_str("<tied type=\"stop\"/>");
    }
    out
}

/// `<notations>` block for one segment: tied + slurs + articulations +
/// fermata. Empty when the segment carries none. Rests only ever carry
/// fermatas (the corpus has no slurred/articulated rests), but the
/// rendering is uniform.
fn notations_xml(seg: &Segment) -> String {
    let tied = tied_xml(seg);
    let mut slurs = String::new();
    if seg.note.slur_start {
        slurs.push_str("<slur type=\"start\" number=\"1\"/>");
    }
    if seg.note.slur_stop {
        slurs.push_str("<slur type=\"stop\" number=\"1\"/>");
    }
    let mut arts = String::new();
    for a in &seg.note.articulations {
        match a.as_str() {
            "stacc" => arts.push_str("<staccato/>"),
            "accent" => arts.push_str("<accent/>"),
            "tenuto" => arts.push_str("<tenuto/>"),
            _ => {}
        }
    }
    let fermata = seg
        .note
        .articulations
        .iter()
        .any(|a| a == "fermata");
    if tied.is_empty() && slurs.is_empty() && arts.is_empty() && !fermata {
        return String::new();
    }
    let mut out = String::from("        <notations>");
    out.push_str(&tied);
    out.push_str(&slurs);
    if !arts.is_empty() {
        out.push_str("<articulations>");
        out.push_str(&arts);
        out.push_str("</articulations>");
    }
    if fermata {
        out.push_str("<fermata/>");
    }
    out.push_str("</notations>\n");
    out
}

fn push_note_xml(
    out: &mut String,
    seg: &Segment,
    divisions: u32,
    losses: &mut LossTable,
    part_name: &str,
) {
    // Grace notes occupy no time: `<grace/>` plus the engraved `<type>`
    // for display, and no `<duration>` (which is also what the scorer
    // reads, so grace round-trips as zero-duration events).
    if seg.note.grace {
        out.push_str("      <note>\n");
        if seg.chord {
            out.push_str("        <chord/>\n");
        }
        out.push_str("        <grace/>\n");
        if seg.note.rest {
            out.push_str("        <rest/>\n");
        } else {
            out.push_str(&format!(
                "        <pitch><step>{}</step>{}<octave>{}</octave></pitch>\n",
                seg.note.step,
                if seg.note.alter != 0 {
                    format!("<alter>{}</alter>", seg.note.alter)
                } else {
                    String::new()
                },
                seg.note.octave,
            ));
        }
        let (typename, dots, shown) = quantize_duration(seg.note.dur_ticks, divisions);
        if shown != seg.note.dur_ticks {
            losses.push(
                "loss",
                "note-type-approximated",
                format!(
                    "{} ticks shown as {typename}{} ({} ticks)",
                    seg.note.dur_ticks,
                    ".".repeat(dots as usize),
                    shown
                ),
                Some(part_name.to_string()),
                None,
            );
        }
        out.push_str(&format!("        <voice>{}</voice>\n", seg.note.voice));
        out.push_str(&format!("        <type>{typename}</type>\n"));
        for _ in 0..dots {
            out.push_str("        <dot/>\n");
        }
        if !seg.note.rest {
            if let Some(acc) = accidental_name(seg.note.alter) {
                out.push_str(&format!("        <accidental>{acc}</accidental>\n"));
            }
        }
        out.push_str(&notations_xml(seg));
        out.push_str("      </note>\n");
        return;
    }
    // `<duration>` carries the exact tick count (timing is lossless).
    // `<type>` is display-only: the nearest type+dots value is shown and
    // any discrepancy is logged as display loss.
    let (typename, dots, shown) = quantize_duration(seg.dur_ticks, divisions);
    if shown != seg.dur_ticks {
        losses.push(
            "loss",
            "note-type-approximated",
            format!(
                "{} ticks shown as {typename}{} ({} ticks)",
                seg.dur_ticks,
                ".".repeat(dots as usize),
                shown
            ),
            Some(part_name.to_string()),
            None,
        );
    }
    let dur = seg.dur_ticks;
    out.push_str("      <note>\n");
    if seg.chord {
        out.push_str("        <chord/>\n");
    }
    if seg.note.rest {
        out.push_str("        <rest/>\n");
    } else {
        out.push_str(&format!(
            "        <pitch><step>{}</step>{}<octave>{}</octave></pitch>\n",
            seg.note.step,
            if seg.note.alter != 0 {
                format!("<alter>{}</alter>", seg.note.alter)
            } else {
                String::new()
            },
            seg.note.octave,
        ));
    }
    out.push_str(&format!("        <duration>{dur}</duration>\n"));
    let (tie_start, tie_stop) = tie_elements(seg);
    if tie_start {
        out.push_str("        <tie type=\"start\"/>\n");
    }
    if tie_stop {
        out.push_str("        <tie type=\"stop\"/>\n");
    }
    out.push_str(&format!("        <voice>{}</voice>\n", seg.note.voice));
    out.push_str(&format!("        <type>{typename}</type>\n"));
    for _ in 0..dots {
        out.push_str("        <dot/>\n");
    }
    let notations = notations_xml(seg);
    if !seg.note.rest {
        if let Some(acc) = accidental_name(seg.note.alter) {
            out.push_str(&format!("        <accidental>{acc}</accidental>\n"));
        }
        out.push_str(&notations);
        if let Some(ly) = &seg.note.lyric {
            out.push_str(&format!(
                "        <lyric><text>{}</text></lyric>\n",
                escape_xml(ly)
            ));
        }
    } else {
        out.push_str(&notations);
    }
    out.push_str("      </note>\n");
}

/// `<clef>` element for a recorded sign/line. The token spelling for a
/// percussion clef is `p` (single char); MusicXML spells it
/// `percussion`, and the pair re-parses to the same token.
fn clef_xml(sign: char, line: u32) -> String {
    let name = match sign {
        'p' => "percussion".to_string(),
        s => s.to_string(),
    };
    format!("        <clef><sign>{name}</sign><line>{line}</line></clef>\n")
}

/// Measure-scoped override for one 1-based measure number, if any.
fn measure_attr(score: &Score, measure: u32) -> Option<&MeasureAttr> {
    score
        .measure_attrs
        .iter()
        .find(|a| a.measure == measure)
}

/// `<direction>` carrying one measure-level dynamic mark.
fn dyn_direction_xml(dynamic: &str) -> String {
    format!(
        "      <direction placement=\"below\"><direction-type><dynamics><{dynamic}/></dynamics></direction-type></direction>\n"
    )
}

/// `<direction>` carrying one hairpin wedge event.
fn hairpin_direction_xml(hairpin: &str) -> String {
    let wedge = match hairpin {
        "cresc" => "crescendo",
        "dim" => "diminuendo",
        _ => "stop",
    };
    format!(
        "      <direction placement=\"below\"><direction-type><wedge type=\"{wedge}\"/></direction-type></direction>\n"
    )
}

/// `<barline>` for a recorded bar style. The rendering re-parses to the
/// same token (repeat direction wins over style, like the importer), so
/// barlines are idempotent across round trips.
fn barline_xml(right: bool, style: &str) -> String {
    let loc = if right { "right" } else { "left" };
    match style {
        "repeat-left" => format!(
            "      <barline location=\"{loc}\"><bar-style>heavy-light</bar-style><repeat direction=\"forward\"/></barline>\n"
        ),
        "repeat-right" => format!(
            "      <barline location=\"{loc}\"><bar-style>light-heavy</bar-style><repeat direction=\"backward\"/></barline>\n"
        ),
        "light-heavy" => {
            format!("      <barline location=\"{loc}\"><bar-style>light-heavy</bar-style></barline>\n")
        }
        "double" => {
            format!("      <barline location=\"{loc}\"><bar-style>light-light</bar-style></barline>\n")
        }
        _ => String::new(),
    }
}

/// Render the score as MusicXML 3.1 partwise.
///
/// The output starts with the XML declaration and
/// `<score-partwise version="3.1">`. One `<part>` per input part, one
/// `<measure>` per bar, first voice written straight, further voices after
/// `<backup>`, each voice rest-filled to the full bar. Measure-scoped
/// overrides ([`MeasureAttr`]: mid-stream clef/key/time changes,
/// dynamics, hairpins, barlines) render as `<attributes>`,
/// `<direction>` and `<barline>` elements at their measure.
pub fn to_musicxml(score: &Score, losses: &mut LossTable) -> String {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<!DOCTYPE score-partwise PUBLIC \"-//Recordare//DTD MusicXML 3.1 Partwise//EN\" \"http://www.musicxml.org/dtds/partwise.dtd\">\n");
    out.push_str("<score-partwise version=\"3.1\">\n");
    out.push_str(&format!(
        "  <work><work-title>{}</work-title></work>\n",
        escape_xml(&score.title)
    ));
    if let Some(c) = &score.composer {
        out.push_str(&format!(
            "  <creator type=\"composer\">{}</creator>\n",
            escape_xml(c)
        ));
    }
    out.push_str("  <part-list>\n");
    for (i, part) in score.parts.iter().enumerate() {
        out.push_str(&format!(
            "    <score-part id=\"P{}\"><part-name>{}</part-name></score-part>\n",
            i + 1,
            escape_xml(&part.name),
        ));
    }
    out.push_str("  </part-list>\n");

    let windows = score.measure_windows();
    for (i, part) in score.parts.iter().enumerate() {
        out.push_str(&format!("  <part id=\"P{}\">\n", i + 1));
        let layout = layout_part(score, part, losses);
        for (m, voices) in layout.iter().enumerate() {
            let mlen = windows
                .get(m)
                .map(|&(a, b)| b - a)
                .unwrap_or_else(|| score.measure_ticks());
            out.push_str(&format!("    <measure number=\"{}\">\n", m + 1));
            let attrs = measure_attr(score, m as u32 + 1);
            if m == 0 {
                let (clef_s, clef_l) = match attrs {
                    Some(a) => (
                        a.clef_sign.unwrap_or('G'),
                        a.clef_line.unwrap_or(2),
                    ),
                    None => ('G', 2),
                };
                let key = attrs
                    .and_then(|a| a.key_fifths)
                    .unwrap_or(score.key_fifths);
                let (beats, beat_type) = match attrs.and_then(|a| a.time) {
                    Some(t) => (t.beats, t.beat_type),
                    None => (score.beats, score.beat_type),
                };
                out.push_str("      <attributes>\n");
                out.push_str(&format!(
                    "        <divisions>{}</divisions>\n",
                    score.divisions
                ));
                out.push_str(&format!("        <key><fifths>{key}</fifths></key>\n"));
                out.push_str(&format!(
                    "        <time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time>\n",
                ));
                out.push_str(&clef_xml(clef_s, clef_l));
                out.push_str("      </attributes>\n");
            } else if let Some(a) = attrs {
                let has_clef = a.clef_sign.is_some() || a.clef_line.is_some();
                if has_clef || a.key_fifths.is_some() || a.time.is_some() {
                    out.push_str("      <attributes>\n");
                    if has_clef {
                        out.push_str(&clef_xml(
                            a.clef_sign.unwrap_or('G'),
                            a.clef_line.unwrap_or(2),
                        ));
                    }
                    if let Some(k) = a.key_fifths {
                        out.push_str(&format!("        <key><fifths>{k}</fifths></key>\n"));
                    }
                    if let Some(t) = a.time {
                        out.push_str(&format!(
                            "        <time><beats>{}</beats><beat-type>{}</beat-type></time>\n",
                            t.beats, t.beat_type,
                        ));
                    }
                    out.push_str("      </attributes>\n");
                }
            }
            if let Some(b) = attrs.and_then(|a| a.barline_left.as_deref()) {
                out.push_str(&barline_xml(false, b));
            }
            if let Some(d) = attrs.and_then(|a| a.dynamic.as_deref()) {
                out.push_str(&dyn_direction_xml(d));
            }
            if let Some(h) = attrs.and_then(|a| a.hairpin.as_deref()) {
                out.push_str(&hairpin_direction_xml(h));
            }
            for (vi, (voice, segs)) in voices.iter().enumerate() {
                let _ = voice;
                if vi > 0 {
                    out.push_str(&format!(
                        "      <backup><duration>{mlen}</duration></backup>\n",
                    ));
                }
                for s in segs {
                    push_note_xml(&mut out, s, score.divisions, losses, &part.name);
                }
            }
            if let Some(b) = attrs.and_then(|a| a.barline_right.as_deref()) {
                out.push_str(&barline_xml(true, b));
            }
            out.push_str("    </measure>\n");
        }
        out.push_str("  </part>\n");
    }
    out.push_str("</score-partwise>\n");
    out
}

// ---------------------------------------------------------------------------
// MuseScore (.mscx subset) rendering + .mscz packing
// ---------------------------------------------------------------------------

/// Tonal pitch class for MuseScore: C=14 in the line of fifths.
fn tpc(step: char, alter: i32) -> i32 {
    let base = match step {
        'C' => 14,
        'D' => 16,
        'E' => 18,
        'F' => 13,
        'G' => 15,
        'A' => 17,
        'B' => 19,
        _ => 14,
    };
    base + alter * 7
}

fn mscx_duration_type(dur_ticks: i64, divisions: u32) -> (&'static str, u8) {
    let (name, dots, _) = quantize_duration(dur_ticks, divisions);
    (name, dots)
}

/// Render a minimal MuseScore 3-compatible `.mscx` document.
///
/// One `<Staff>` per part, one `<Measure>` per bar, one `<voice>` per
/// MusicXML voice. Chord/rest/durationType/dots mirror the MusicXML layout,
/// so the two renderings stay consistent.
///
/// Known subset limits (each logged to `losses` as `mscx-…` loss):
/// * ties are re-articulated (no spanners in this subset);
/// * exact tick durations are shown with the nearest `durationType`;
/// * lyrics, accidentals-as-text and multi-staff parts are dropped.
pub fn to_mscx(score: &Score, losses: &mut LossTable) -> String {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<museScore version=\"3.06\">\n");
    out.push_str("  <Score>\n");
    out.push_str("    <LayerTag id=\"0\" tag=\"default\"></LayerTag>\n");
    out.push_str("    <currentLayer>0</currentLayer>\n");
    out.push_str(&format!("    <Division>{}</Division>\n", score.divisions));
    out.push_str(&format!(
        "    <metaTag name=\"workTitle\">{}</metaTag>\n",
        escape_xml(&score.title)
    ));
    if let Some(c) = &score.composer {
        out.push_str(&format!(
            "    <metaTag name=\"composer\">{}</metaTag>\n",
            escape_xml(c)
        ));
    }
    for (i, part) in score.parts.iter().enumerate() {
        out.push_str("    <Part>\n");
        out.push_str(&format!("      <Staff id=\"{}\">\n", i + 1));
        out.push_str("        <StaffType group=\"pitched\">\n");
        out.push_str("          <name>stdNormal</name>\n");
        out.push_str("          </StaffType>\n");
        out.push_str("        <bracket type=\"-1\" span=\"0\" col=\"0\"/>\n");
        out.push_str("        </Staff>\n");
        out.push_str(&format!(
            "      <trackName>{}</trackName>\n",
            escape_xml(&part.name)
        ));
        out.push_str("      </Part>\n");
    }
    for (i, part) in score.parts.iter().enumerate() {
        out.push_str(&format!("    <Staff id=\"{}\">\n", i + 1));
        let layout = layout_part(score, part, losses);
        let mut cur_sig = (score.beats, score.beat_type);
        let mut cur_key = score.key_fifths;
        for (m, voices) in layout.iter().enumerate() {
            out.push_str("      <Measure>\n");
            // Mid-stream time/key changes render like the opening ones;
            // every other measure attr (clefs, dynamics, hairpins,
            // barlines, grace) is outside the .mscx subset and logged.
            if let Some(a) = measure_attr(score, m as u32 + 1) {
                if let Some(t) = a.time {
                    cur_sig = (t.beats, t.beat_type);
                }
                if let Some(k) = a.key_fifths {
                    cur_key = k;
                }
            }
            if m == 0
                || measure_attr(score, m as u32 + 1).is_some_and(|a| a.time.is_some())
            {
                out.push_str(&format!(
                    "        <TimeSig><sigN>{}</sigN><sigD>{}</sigD></TimeSig>\n",
                    cur_sig.0, cur_sig.1
                ));
            }
            if m == 0
                || measure_attr(score, m as u32 + 1).is_some_and(|a| a.key_fifths.is_some())
            {
                out.push_str(&format!(
                    "        <KeySig><accidental>{}</accidental></KeySig>\n",
                    cur_key
                ));
            }
            if let Some(a) = measure_attr(score, m as u32 + 1) {
                for (present, kind) in [
                    (
                        a.clef_sign.is_some() || a.clef_line.is_some(),
                        "mscx-clef-dropped",
                    ),
                    (a.dynamic.is_some(), "mscx-dynamic-dropped"),
                    (a.hairpin.is_some(), "mscx-hairpin-dropped"),
                    (
                        a.barline_left.is_some() || a.barline_right.is_some(),
                        "mscx-barline-dropped",
                    ),
                ] {
                    if present {
                        losses.push(
                            "loss",
                            kind,
                            format!(
                                "measure {}: {kind} (outside the .mscx subset)",
                                m + 1
                            ),
                            Some(part.name.clone()),
                            None,
                        );
                    }
                }
            }
            for (_voice, segs) in voices {
                out.push_str("        <voice>\n");
                // Group chord runs: a head segment (chord=false) plus its
                // following chord=true members form one <Chord>.
                let mut k = 0;
                while k < segs.len() {
                    let head = &segs[k];
                    let (dtype, dots) = mscx_duration_type(head.dur_ticks, score.divisions);
                    if head.note.rest {
                        if head.note.grace {
                            losses.push(
                                "loss",
                                "mscx-grace-dropped",
                                "grace rest dropped in .mscx (subset has no grace)".to_string(),
                                Some(part.name.clone()),
                                None,
                            );
                            k += 1;
                            continue;
                        }
                        out.push_str("          <Rest>\n");
                        out.push_str(&format!(
                            "            <durationType>{dtype}</durationType>\n"
                        ));
                        for _ in 0..dots {
                            out.push_str("            <dots>1</dots>\n");
                        }
                        out.push_str("            </Rest>\n");
                        k += 1;
                        continue;
                    }
                    let mut run: Vec<&Segment> = vec![head];
                    while k + run.len() < segs.len() && segs[k + run.len()].chord {
                        run.push(&segs[k + run.len()]);
                    }
                    if run.iter().any(|s| s.note.grace) {
                        losses.push(
                            "loss",
                            "mscx-grace-dropped",
                            format!(
                                "grace run of {} notes dropped in .mscx (subset has no grace)",
                                run.len()
                            ),
                            Some(part.name.clone()),
                            None,
                        );
                        k += run.len();
                        continue;
                    }
                    for s in &run {
                        if s.tie_in
                            || s.tie_out
                            || s.note.tie_start
                            || s.note.tie_stop
                        {
                            losses.push(
                                "loss",
                                "mscx-tie-dropped",
                                "tied segment re-articulated in .mscx (subset has no spanners)"
                                    .to_string(),
                                Some(part.name.clone()),
                                None,
                            );
                        }
                        if s.note.lyric.is_some() {
                            losses.push(
                                "loss",
                                "mscx-lyric-dropped",
                                "lyrics are not carried by the .mscx subset".to_string(),
                                Some(part.name.clone()),
                                None,
                            );
                        }
                    }
                    out.push_str("          <Chord>\n");
                    out.push_str(&format!(
                        "            <durationType>{dtype}</durationType>\n"
                    ));
                    for _ in 0..dots {
                        out.push_str("            <dots>1</dots>\n");
                    }
                    for s in &run {
                        out.push_str("            <Note>\n");
                        out.push_str(&format!("              <pitch>{}</pitch>\n", s.note.midi));
                        out.push_str(&format!(
                            "              <tpc>{}</tpc>\n",
                            tpc(s.note.step, s.note.alter)
                        ));
                        out.push_str("              </Note>\n");
                    }
                    out.push_str("            </Chord>\n");
                    k += run.len();
                }
                out.push_str("        </voice>\n");
            }
            out.push_str("      </Measure>\n");
        }
        out.push_str("    </Staff>\n");
    }
    out.push_str("  </Score>\n");
    out.push_str("</museScore>\n");
    out
}

/// File stem for the score inside `.mscz`: ASCII alphanumerics, `-`, `_`.
pub fn score_stem(title: &str) -> String {
    let mut s: String = title
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    while s.contains("--") {
        s = s.replace("--", "-");
    }
    if s.is_empty() {
        s = "score".to_string();
    }
    s
}

fn container_xml(mscx_name: &str) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<container>\n  <rootfiles>\n    <rootfile full-path=\"{mscx_name}\"/>\n  </rootfiles>\n</container>\n"
    )
}

// ---------------------------------------------------------------------------
// SER (Symbol Error Rate)
// ---------------------------------------------------------------------------

/// Symbol Error Rate report: Levenshtein distance over symbol sequences,
/// normalised by the reference length (`ser = distance / ref_len`).
///
/// A symbol is one note/rest event rendered as
/// `"p{part}:R|m{midi}:d{dur_ticks}:v{voice}"`, in score order (parts in
/// input order, then `(start_ticks, voice, midi)`). Rests count as symbols,
/// so a dropped rest is one deletion. Values above 1.0 are possible when
/// the hypothesis inserts more symbols than the reference holds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerReport {
    pub distance: usize,
    pub ref_len: usize,
    pub hyp_len: usize,
    pub ser: f64,
}

/// Levenshtein edit distance over symbol sequences (insert/delete/substitute
/// cost 1). Time O(a*b), memory O(min(a,b)).
pub fn levenshtein(a: &[String], b: &[String]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    // Keep the shorter sequence on the inner axis.
    let (short, long) = if b.len() < a.len() { (b, a) } else { (a, b) };
    let mut prev: Vec<usize> = (0..=short.len()).collect();
    let mut curr = vec![0usize; short.len() + 1];
    for (i, l) in long.iter().enumerate() {
        curr[0] = i + 1;
        for (j, s) in short.iter().enumerate() {
            let sub = prev[j] + usize::from(l != s);
            let ins = curr[j] + 1;
            let del = prev[j + 1] + 1;
            curr[j + 1] = sub.min(ins).min(del);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[short.len()]
}

/// Canonical symbol sequence for a parsed [`Score`].
pub fn score_symbols(score: &Score) -> Vec<String> {
    let mut out = Vec::new();
    for (pi, part) in score.parts.iter().enumerate() {
        for g in &part.groups {
            for m in &g.members {
                if m.rest {
                    out.push(format!("p{pi}:R:d{}:v{}", g.dur_ticks, g.voice));
                } else {
                    out.push(format!("p{pi}:m{}:d{}:v{}", m.midi, g.dur_ticks, g.voice));
                }
            }
        }
    }
    out
}

/// SER over two symbol sequences.
pub fn ser_for_symbols(reference: &[String], hypothesis: &[String]) -> SerReport {
    let distance = levenshtein(reference, hypothesis);
    let ser = if reference.is_empty() {
        if hypothesis.is_empty() {
            0.0
        } else {
            hypothesis.len() as f64
        }
    } else {
        distance as f64 / reference.len() as f64
    };
    SerReport {
        distance,
        ref_len: reference.len(),
        hyp_len: hypothesis.len(),
        ser,
    }
}

/// Parse both inputs with [`parse_score`] (canonical tick order) and report
/// SER of hypothesis against reference, with the two symbol sequences.
pub fn ser_for_inputs(
    reference: &ScoreInput,
    hypothesis: &ScoreInput,
) -> Result<(SerReport, Vec<String>, Vec<String>), String> {
    let (ref_score, _) = parse_score(reference).map_err(|e| format!("reference: {e}"))?;
    let (hyp_score, _) = parse_score(hypothesis).map_err(|e| format!("hypothesis: {e}"))?;
    let ref_syms = score_symbols(&ref_score);
    let hyp_syms = score_symbols(&hyp_score);
    let report = ser_for_symbols(&ref_syms, &hyp_syms);
    Ok((report, ref_syms, hyp_syms))
}

/// Pack rendered `.mscx` into `.mscz` bytes (deflated zip with
/// `META-INF/container.xml` + `<stem>.mscx`).
pub fn pack_mscz(score: &Score, mscx: &str) -> Result<Vec<u8>, String> {
    let stem = score_stem(&score.title);
    let mscx_name = format!("{stem}.mscx");
    let mut buf = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buf);
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip.start_file("META-INF/container.xml", opts)
            .map_err(|e| format!("zip container.xml: {e}"))?;
        zip.write_all(container_xml(&mscx_name).as_bytes())
            .map_err(|e| format!("zip container.xml: {e}"))?;
        zip.start_file(mscx_name, opts)
            .map_err(|e| format!("zip score: {e}"))?;
        zip.write_all(mscx.as_bytes())
            .map_err(|e| format!("zip score: {e}"))?;
        zip.finish().map_err(|e| format!("zip finish: {e}"))?;
    }
    Ok(buf.into_inner())
}

// ---------------------------------------------------------------------------
// Validators
// ---------------------------------------------------------------------------

/// Validate a raw [`ScoreInput`] before conversion.
/// Errors: structural problems that make conversion fail; warnings: things
/// conversion will normalise (unsorted onsets, explicit rests, gaps).
pub fn validate_notes(input: &ScoreInput) -> Vec<Issue> {
    let mut out = Vec::new();
    if input.divisions == 0 {
        out.push(error(
            "notes-divisions",
            "divisions must be positive".to_string(),
        ));
    }
    let beats = input.time.map(|t| t.beats).unwrap_or(4);
    let beat_type = input.time.map(|t| t.beat_type).unwrap_or(4);
    if beats == 0 || beat_type == 0 {
        out.push(error(
            "notes-time",
            "time beats and beat_type must be positive".to_string(),
        ));
    } else if !matches!(beat_type, 1 | 2 | 4 | 8 | 16 | 32) {
        out.push(error(
            "notes-time",
            format!("unsupported beat_type {beat_type}"),
        ));
    }
    let parts: Vec<(&str, &[NoteInput])> = if !input.parts.is_empty() {
        input
            .parts
            .iter()
            .map(|p| (p.name.as_str(), p.notes.as_slice()))
            .collect()
    } else {
        vec![(
            input.part_name.as_deref().unwrap_or("Part 1"),
            input.notes.as_slice(),
        )]
    };
    if parts.iter().all(|(_, n)| n.is_empty()) {
        out.push(error("notes-empty", "score has no notes".to_string()));
        return out;
    }
    for (pname, notes) in &parts {
        let mut prev_onset = -1.0f64;
        let mut seen_gap = false;
        let mut cursor = 0.0f64;
        for (i, n) in notes.iter().enumerate() {
            let here = format!("part {pname:?} note {i}");
            if !n.onset.is_finite() || !n.dur.is_finite() {
                out.push(error(
                    "notes-finite",
                    format!("{here}: onset/dur must be finite"),
                ));
                continue;
            }
            if n.onset < 0.0 {
                out.push(error(
                    "notes-onset",
                    format!("{here}: negative onset {}", n.onset),
                ));
            }
            if n.dur <= 0.0 {
                out.push(error(
                    "notes-duration",
                    format!("{here}: non-positive dur {}", n.dur),
                ));
            }
            if n.voice == 0 {
                out.push(error("notes-voice", format!("{here}: voice must be >= 1")));
            }
            if n.onset < prev_onset {
                out.push(warning(
                    "notes-unsorted",
                    format!("{here}: onsets not in order (converter sorts)"),
                ));
            }
            prev_onset = n.onset;
            if n.onset > cursor + 1e-9 {
                seen_gap = true;
            }
            cursor = cursor.max(n.onset + n.dur.max(0.0));
            let is_rest = n.rest
                || n.pitch
                    .as_ref()
                    .map(|p| p.trim().is_empty())
                    .unwrap_or(true);
            if !is_rest {
                match parse_pitch(n.pitch.as_ref().unwrap()) {
                    Ok((_, _, _, midi, cents)) => {
                        if !(0..=127).contains(&midi) {
                            out.push(error("notes-range", format!("{here}: out of MIDI range")));
                        }
                        if cents.is_some() {
                            out.push(warning(
                                "notes-microtone",
                                format!("{here}: cents suffix will be rounded to MIDI {midi}"),
                            ));
                        }
                    }
                    Err(e) => out.push(error("notes-pitch", format!("{here}: {e}"))),
                }
            }
        }
        if seen_gap {
            out.push(warning(
                "notes-gaps",
                format!("part {pname:?}: onset gaps will be filled with rests"),
            ));
        }
    }
    out
}

fn read_tag_value(xml: &str, start: usize, tag: &str) -> Option<(i64, usize)> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let s = xml[start..].find(&open)? + start + open.len();
    let e = xml[s..].find(&close)? + s;
    let v = xml[s..e].trim().parse::<i64>().ok()?;
    Some((v, e + close.len()))
}

/// Validate a MusicXML document (ours or foreign):
/// declaration, `score-partwise version="3.1"`, per-measure duration sums
/// honouring `<backup>`/`<forward>`, positive durations, pitched notes with
/// step+octave, and no unescaped `&` in text.
pub fn validate_musicxml(xml: &str) -> Vec<Issue> {
    let mut out = Vec::new();
    if !xml.starts_with("<?xml") {
        out.push(error(
            "xml-declaration",
            "missing <?xml declaration".to_string(),
        ));
    }
    match xml.find("<score-partwise") {
        None => {
            out.push(error(
                "xml-root",
                "missing <score-partwise> root".to_string(),
            ));
            return out;
        }
        Some(p) => {
            let head = &xml[p..(p + 200).min(xml.len())];
            if !head.contains("version=\"3.1\"") {
                out.push(error(
                    "xml-version",
                    "score-partwise version must be \"3.1\" for this pipeline".to_string(),
                ));
            }
        }
    }
    // Crude unescaped-& scan: every '&' must start a known entity.
    let bytes = xml.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'&' {
            let tail = &xml[i..(i + 8).min(xml.len())];
            if !(tail.starts_with("&amp;")
                || tail.starts_with("&lt;")
                || tail.starts_with("&gt;")
                || tail.starts_with("&quot;")
                || tail.starts_with("&apos;"))
            {
                out.push(error("xml-escape", format!("unescaped '&' at byte {i}")));
                break;
            }
        }
        i += 1;
    }
    // Per-measure duration simulation.
    let divisions = read_tag_value(xml, 0, "divisions")
        .map(|(v, _)| v)
        .unwrap_or(1);
    if divisions <= 0 {
        out.push(error(
            "xml-divisions",
            "divisions must be positive".to_string(),
        ));
        return out;
    }
    let beats = read_tag_value(xml, 0, "beats").map(|(v, _)| v).unwrap_or(4);
    let beat_type = read_tag_value(xml, 0, "beat-type")
        .map(|(v, _)| v)
        .unwrap_or(4);
    if beats <= 0 || beat_type <= 0 {
        out.push(error(
            "xml-time",
            "time beats/beat-type must be positive".to_string(),
        ));
        return out;
    }
    let mut expected = beats * divisions * 4 / beat_type;
    let mut pos = 0;
    let mut mnum = 0;
    let mut note_count = 0;
    while let Some(mp) = xml[pos..].find("<measure") {
        mnum += 1;
        let mstart = pos + mp;
        let mend = match xml[mstart..].find("</measure>") {
            Some(e) => mstart + e,
            None => {
                out.push(error(
                    "xml-measure",
                    format!("measure {mnum}: missing </measure>"),
                ));
                break;
            }
        };
        let body = &xml[mstart..mend];
        if let (Some((mb, _)), Some((mbt, _))) = (
            read_tag_value(body, 0, "beats"),
            read_tag_value(body, 0, "beat-type"),
        ) {
            if mb > 0 && mbt > 0 {
                expected = mb * divisions * 4 / mbt;
            }
        }
        let mut t = 0i64;
        let mut k = 0;
        let mut ctx = 0u8; // 0 none, 1 note, 2 backup, 3 forward, 4 attributes(skip)
        let mut chord_tone = false;
        while k < body.len() {
            if body[k..].starts_with("<note>") || body[k..].starts_with("<note ") {
                ctx = 1;
                chord_tone = false;
                k += 6;
            } else if body[k..].starts_with("<chord") && ctx == 1 {
                // A chord tone shares the previous note's onset: per the
                // MusicXML spec its duration does not advance time.
                chord_tone = true;
                k += 6;
            } else if body[k..].starts_with("<backup>") {
                ctx = 2;
                k += 8;
            } else if body[k..].starts_with("<forward>") {
                ctx = 3;
                k += 9;
            } else if body[k..].starts_with("<attributes>") {
                ctx = 4;
                k += 12;
            } else if body[k..].starts_with("</note>")
                || body[k..].starts_with("</backup>")
                || body[k..].starts_with("</forward>")
                || body[k..].starts_with("</attributes>")
            {
                ctx = 0;
                k += 1;
            } else if body[k..].starts_with("<duration>") {
                match read_tag_value(body, k, "duration") {
                    Some((v, next)) => {
                        if ctx == 4 {
                            // divisions inside attributes: not a time advance.
                        } else if v <= 0 {
                            out.push(error(
                                "xml-duration",
                                format!("measure {mnum}: non-positive duration {v}"),
                            ));
                        } else if ctx == 2 {
                            t -= v;
                        } else if ctx == 1 && chord_tone {
                            // Chord tone: shares onset, no time advance.
                        } else {
                            t += v;
                        }
                        k = next;
                    }
                    None => {
                        out.push(error(
                            "xml-duration",
                            format!("measure {mnum}: bad <duration>"),
                        ));
                        k += 1;
                    }
                }
            } else {
                k += 1;
            }
        }
        if t != expected {
            out.push(error(
                "xml-measure-sum",
                format!("measure {mnum}: durations sum to {t} ticks, expected {expected}"),
            ));
        }
        // Pitched-note completeness within this measure.
        let mut n = 0;
        let mut nn = 0;
        while let Some(p) = body[nn..].find("<note>") {
            n += 1;
            let ns = nn + p;
            let ne = body[ns..]
                .find("</note>")
                .map(|e| ns + e)
                .unwrap_or(body.len());
            let nb = &body[ns..ne];
            note_count += 1;
            if !nb.contains("<rest") && !nb.contains("<rest/>") {
                if !(nb.contains("<step>") && nb.contains("<octave>")) {
                    out.push(error(
                        "xml-pitch",
                        format!("measure {mnum}: pitched note without step+octave"),
                    ));
                }
                // Grace notes correctly carry no `<duration>` (they steal
                // no time); requiring one would reject valid notation.
                if !nb.contains("<grace") && !nb.contains("<duration>") {
                    out.push(error(
                        "xml-duration",
                        format!("measure {mnum}: note without duration"),
                    ));
                }
            }
            nn = ne + 1;
        }
        if n == 0 {
            out.push(warning(
                "xml-empty-measure",
                format!("measure {mnum}: no notes"),
            ));
        }
        pos = mend + 1;
    }
    if mnum == 0 {
        out.push(error(
            "xml-nomeasures",
            "no <measure> elements found".to_string(),
        ));
    }
    if note_count == 0 {
        out.push(warning(
            "xml-nonotes",
            "no <note> elements found".to_string(),
        ));
    }
    out
}

/// Validate `.mscz` bytes: readable zip, `META-INF/container.xml` present,
/// rootfile entry present and ending in `.mscx`, score entry parseable with
/// `<museScore>` root and at least one `<Measure>`.
pub fn validate_mscz(data: &[u8]) -> Vec<Issue> {
    let mut out = Vec::new();
    let mut zip = match zip::ZipArchive::new(Cursor::new(data)) {
        Ok(z) => z,
        Err(e) => {
            out.push(error("mscz-zip", format!("not a readable zip: {e}")));
            return out;
        }
    };
    let container = match zip.by_name("META-INF/container.xml") {
        Ok(mut f) => {
            use std::io::Read;
            let mut s = String::new();
            match f.read_to_string(&mut s) {
                Ok(_) => s,
                Err(e) => {
                    out.push(error(
                        "mscz-container",
                        format!("cannot read container.xml: {e}"),
                    ));
                    String::new()
                }
            }
        }
        Err(_) => {
            out.push(error(
                "mscz-container",
                "missing META-INF/container.xml".to_string(),
            ));
            String::new()
        }
    };
    if !container.is_empty() {
        match container.find("full-path=\"") {
            None => out.push(error(
                "mscz-container",
                "container.xml has no rootfile full-path".to_string(),
            )),
            Some(p) => {
                let s = p + "full-path=\"".len();
                let e = container[s..]
                    .find('"')
                    .map(|q| s + q)
                    .unwrap_or(container.len());
                let target = &container[s..e];
                if !target.ends_with(".mscx") {
                    out.push(error(
                        "mscz-container",
                        format!("rootfile {target:?} is not .mscx"),
                    ));
                } else {
                    match zip.by_name(target) {
                        Ok(mut f) => {
                            use std::io::Read;
                            let mut s = String::new();
                            if f.read_to_string(&mut s).is_err() {
                                out.push(error("mscz-score", format!("cannot read {target}")));
                            } else {
                                if !s.contains("<museScore") {
                                    out.push(error(
                                        "mscz-score",
                                        format!("{target}: missing <museScore> root"),
                                    ));
                                }
                                if !s.contains("<Measure>") {
                                    out.push(error(
                                        "mscz-score",
                                        format!("{target}: no <Measure> found"),
                                    ));
                                }
                            }
                        }
                        Err(_) => out.push(error(
                            "mscz-score",
                            format!("container points at {target:?} but it is not in the zip"),
                        )),
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Example + tests
// ---------------------------------------------------------------------------

/// Small two-bar example input (also used by the `example` CLI subcommand).
pub fn example_input() -> ScoreInput {
    serde_json::from_value(serde_json::json!({
        "title": "Assembly Example",
        "composer": "Converter",
        "divisions": 480,
        "time": {"beats": 4, "beat_type": 4},
        "key_fifths": 0,
        "parts": [
            {"name": "Piano", "notes": [
                {"pitch": "C4", "onset": 0.0, "dur": 1.0},
                {"pitch": "E4", "onset": 1.0, "dur": 1.0},
                {"pitch": "G4", "onset": 2.0, "dur": 1.0},
                {"pitch": "C5", "onset": 3.0, "dur": 1.0},
                {"pitch": "G4", "onset": 4.0, "dur": 1.5},
                {"pitch": "E4", "onset": 5.5, "dur": 0.5},
                {"pitch": "D4", "onset": 6.0, "dur": 1.0},
                {"pitch": "C4", "onset": 7.0, "dur": 1.0}
            ]}
        ]
    }))
    .expect("example input is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_score() -> (Score, LossTable) {
        parse_score(&example_input()).expect("example parses")
    }

    #[test]
    fn pitch_shapes() {
        assert!(parse_pitch("C4").is_ok());
        assert!(parse_pitch("F#5").is_ok());
        assert!(parse_pitch("Bb3").is_ok());
        assert!(parse_pitch("Gx2").is_ok());
        assert!(parse_pitch("H4").is_err());
        assert!(parse_pitch("C").is_err());
        assert!(parse_pitch("C###4").is_err());
        let (_, _, _, midi, cents) = parse_pitch("A4+80c").unwrap();
        assert_eq!(cents, Some(80));
        assert_eq!(midi, 70); // A#4
    }

    #[test]
    fn quantize_common_values() {
        assert_eq!(quantize_duration(480, 480), ("quarter", 0, 480));
        assert_eq!(quantize_duration(720, 480), ("quarter", 1, 720));
        assert_eq!(quantize_duration(1920, 480), ("whole", 0, 1920));
    }

    #[test]
    fn musicxml_roundtrip_validates() {
        let (score, _) = simple_score();
        let mut losses = LossTable::default();
        let xml = to_musicxml(&score, &mut losses);
        assert!(xml.contains("<score-partwise version=\"3.1\">"));
        let issues = validate_musicxml(&xml);
        let errs: Vec<_> = issues.iter().filter(|i| i.severity == "error").collect();
        assert!(errs.is_empty(), "errors: {errs:?}");
    }

    #[test]
    fn simultaneous_notes_form_chord() {
        let input: ScoreInput = serde_json::from_value(serde_json::json!({
            "notes": [
                {"pitch": "C4", "onset": 0.0, "dur": 1.0},
                {"pitch": "E4", "onset": 0.0, "dur": 1.0}
            ]
        }))
        .unwrap();
        let (score, losses) = parse_score(&input).unwrap();
        assert_eq!(score.parts[0].groups.len(), 1);
        assert_eq!(score.parts[0].groups[0].members.len(), 2);
        assert!(losses.is_lossless());
        let mut l2 = LossTable::default();
        let xml = to_musicxml(&score, &mut l2);
        assert!(xml.contains("<chord/>"));
        assert!(!has_errors(&validate_musicxml(&xml)));
    }

    #[test]
    fn ties_span_barlines() {
        let input: ScoreInput = serde_json::from_value(serde_json::json!({
            "notes": [{"pitch": "C4", "onset": 3.0, "dur": 2.0}]
        }))
        .unwrap();
        let (score, _) = parse_score(&input).unwrap();
        let mut losses = LossTable::default();
        let xml = to_musicxml(&score, &mut losses);
        assert!(xml.contains("<tie type=\"start\"/>"));
        assert!(xml.contains("<tie type=\"stop\"/>"));
        let issues = validate_musicxml(&xml);
        assert!(!has_errors(&issues), "{issues:?}");
    }

    #[test]
    fn loss_table_records_microtone_and_clamp() {
        let input: ScoreInput = serde_json::from_value(serde_json::json!({
            "notes": [
                {"pitch": "A4+30c", "onset": 0.0, "dur": 1.0},
                {"pitch": "C4", "onset": 0.5, "dur": 1.0}
            ]
        }))
        .unwrap();
        let (_, losses) = parse_score(&input).unwrap();
        assert!(losses
            .entries
            .iter()
            .any(|e| e.kind == "pitch-microtone-rounded"));
        assert!(losses.entries.iter().any(|e| e.kind == "overlap-clamped"));
    }

    #[test]
    fn mscz_roundtrip_validates() {
        let (score, _) = simple_score();
        let mut losses = LossTable::default();
        let mscx = to_mscx(&score, &mut losses);
        let data = pack_mscz(&score, &mscx).expect("pack");
        let issues = validate_mscz(&data);
        assert!(!has_errors(&issues), "{issues:?}");
    }

    #[test]
    fn validators_catch_bad_input() {
        let bad: ScoreInput = serde_json::from_value(serde_json::json!({
            "notes": [{"pitch": "H9", "onset": -1.0, "dur": 0.0}]
        }))
        .unwrap();
        let issues = validate_notes(&bad);
        assert!(has_errors(&issues));
        assert!(validate_musicxml("<plumbing/>")
            .iter()
            .any(|i| i.severity == "error"));
        assert!(validate_mscz(b"nope").iter().any(|i| i.severity == "error"));
    }

    fn syms(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn ser_identical_is_zero() {
        let (score, _) = simple_score();
        let s = score_symbols(&score);
        let r = ser_for_symbols(&s, &s);
        assert_eq!(r.distance, 0);
        assert_eq!(r.ref_len, s.len());
        assert_eq!(r.ser, 0.0);
    }

    #[test]
    fn ser_counts_substitution_insertion_deletion() {
        let r = ser_for_symbols(&syms(&["a", "b", "c"]), &syms(&["a", "x", "c"]));
        assert_eq!(r.distance, 1);
        assert!((r.ser - 1.0 / 3.0).abs() < 1e-12);
        let ins = ser_for_symbols(&syms(&["a", "b"]), &syms(&["a", "x", "b"]));
        assert_eq!(ins.distance, 1);
        assert!((ins.ser - 0.5).abs() < 1e-12);
        let del = ser_for_symbols(&syms(&["a", "x", "b"]), &syms(&["a", "b"]));
        assert_eq!(del.distance, 1);
        assert!((del.ser - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn ser_empty_reference_edge() {
        let both = ser_for_symbols(&[], &[]);
        assert_eq!(both.ser, 0.0);
        let hyp_only = ser_for_symbols(&[], &syms(&["a", "b"]));
        assert_eq!(hyp_only.distance, 2);
        assert_eq!(hyp_only.ser, 2.0);
    }

    #[test]
    fn ser_for_inputs_uses_canonical_ticks() {
        // 0.0001 beats rounds to the same tick as 0.0, so SER is 0.
        let reference: ScoreInput = serde_json::from_value(serde_json::json!({
            "notes": [{"pitch": "C4", "onset": 0.0, "dur": 1.0}]
        }))
        .unwrap();
        let hypothesis: ScoreInput = serde_json::from_value(serde_json::json!({
            "notes": [{"pitch": "C4", "onset": 0.0001, "dur": 1.0}]
        }))
        .unwrap();
        let (report, _, _) = ser_for_inputs(&reference, &hypothesis).unwrap();
        assert_eq!(report.ser, 0.0);
        let wrong: ScoreInput = serde_json::from_value(serde_json::json!({
            "notes": [{"pitch": "D4", "onset": 0.0, "dur": 1.0}]
        }))
        .unwrap();
        let (report, _, _) = ser_for_inputs(&reference, &wrong).unwrap();
        assert_eq!(report.distance, 1);
        assert_eq!(report.ser, 1.0);
    }
}
