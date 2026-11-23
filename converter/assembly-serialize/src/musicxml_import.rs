//! MusicXML Partwise 3.1 -> [`ScoreInput`](crate::ScoreInput) reference adapter.
//!
//! Ground-truth `.musicxml` (e.g. MuseScore exports in
//! `data-gen/score-corpus/renders/`) becomes the note-list JSON that
//! `assembly-serialize ser` compares against. Ties merge into single
//! logical notes; page breaks split per-page inputs.
//!
//! Scope: pitched notes, rests, chords, backup/forward voices, ties,
//! per-part divisions, first time/key signature wins. Grace notes are
//! skipped; `<print new-page>` tracks pagination; a tie chain crossing a
//! page break merges onto its start page.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{NoteInput, PartInput, ScoreInput, TimeSig, SCORE_DIVISIONS};

/// One page of a paginated import: 1-based page number plus its note list.
#[derive(Debug, Clone)]
pub struct PageScore {
    pub page: u32,
    pub input: ScoreInput,
}

/// Whole-score import: all pages concatenated in document order.
pub fn score_input_from_musicxml(xml: &str) -> Result<ScoreInput, String> {
    let mut pages = score_inputs_by_page(xml)?;
    if pages.is_empty() {
        return Err("no notes found".to_string());
    }
    let first = pages.remove(0);
    let mut head = first.input;
    for p in pages {
        for (dst, src) in head.parts.iter_mut().zip(p.input.parts.iter()) {
            dst.notes.extend(src.notes.iter().cloned());
        }
    }
    Ok(head)
}

/// Per-page import split at `<print new-page="yes">` markers.
pub fn score_inputs_by_page(xml: &str) -> Result<Vec<PageScore>, String> {
    let doc = parse_document(xml)?;
    if doc.parts.iter().all(|p| p.notes.is_empty()) {
        return Err("no notes found".to_string());
    }
    // Collect page numbers in order of first appearance.
    let mut order: Vec<u32> = Vec::new();
    for part in &doc.parts {
        for n in &part.notes {
            if !order.contains(&n.page) {
                order.push(n.page);
            }
        }
    }
    order.sort();
    Ok(order
        .into_iter()
        .map(|page| {
            let parts = doc
                .parts
                .iter()
                .map(|p| PartInput {
                    name: p.name.clone(),
                    notes: p
                        .notes
                        .iter()
                        .filter(|n| n.page == page)
                        .map(|n| n.note.clone())
                        .collect(),
                })
                .collect();
            PageScore {
                page,
                input: ScoreInput {
                    title: doc.title.clone(),
                    composer: doc.composer.clone(),
                    // House tick grid (beats are grid-independent, so the
                    // file's divisions only matter for ticks_to_beats).
                    divisions: SCORE_DIVISIONS,
                    time: Some(doc.time),
                    key_fifths: doc.key_fifths,
                    parts,
                    notes: Vec::new(),
                    part_name: None,
                    measure_attrs: Vec::new(),
                },
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Internal parse model (ticks; converted to beats at the end)
// ---------------------------------------------------------------------------

struct RawNote {
    page: u32,
    note: NoteInput,
}

struct DocPart {
    name: String,
    divisions: u32,
    notes: Vec<RawNote>,
}

struct Doc {
    title: Option<String>,
    composer: Option<String>,
    time: TimeSig,
    key_fifths: i32,
    parts: Vec<DocPart>,
}

pub(crate) fn step_semitone(step: char) -> i32 {
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

/// Text of the first `<tag ...>text</tag>` occurrence, trimmed.
fn tag_text(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}");
    let mut pos = s.find(&open)?;
    pos = s[pos..].find('>')? + pos + 1;
    let end = s[pos..].find(&format!("</{tag}>"))? + pos;
    Some(s[pos..end].trim().to_string())
}

/// True when `<tag .../>` or `<tag ...>` opens inside `s`.
fn has_open(s: &str, tag: &str) -> bool {
    match s.find(&format!("<{tag}")) {
        None => false,
        Some(p) => {
            let rest = &s[p + tag.len() + 1..];
            rest.starts_with('>')
                || rest.starts_with(' ')
                || rest.starts_with('/')
                || rest.starts_with('\t')
                || rest.starts_with('\n')
                || rest.starts_with('\r')
        }
    }
}

/// Split `s` into element bodies for `<name ...>...</name>` in order.
fn elements<'a>(s: &'a str, name: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut rest = s;
    let close = format!("</{name}>");
    while let Some(start) = rest.find(&format!("<{name}")) {
        let body_start = match rest[start..].find('>') {
            Some(q) => start + q + 1,
            None => break,
        };
        let end = match rest[body_start..].find(&close) {
            Some(q) => body_start + q,
            None => break,
        };
        out.push(&rest[start..end]);
        rest = &rest[end + close.len()..];
    }
    out
}

fn accidental_suffix(alter: i32) -> Result<&'static str, String> {
    match alter {
        0 => Ok(""),
        1 => Ok("#"),
        2 => Ok("##"),
        -1 => Ok("b"),
        -2 => Ok("bb"),
        _ => Err(format!("unsupported alter {alter}")),
    }
}

fn parse_document(xml: &str) -> Result<Doc, String> {
    if !xml.contains("<score-partwise") {
        return Err("not a score-partwise document".to_string());
    }
    // Part names from the part list.
    let mut names: HashMap<String, String> = HashMap::new();
    for sp in elements(xml, "score-part") {
        if let Some(id) = attr_value(sp, "id") {
            names.insert(
                id,
                tag_text(sp, "part-name").unwrap_or_else(|| "Part".to_string()),
            );
        }
    }
    let title = tag_text(xml, "movement-title").or_else(|| tag_text(xml, "work-title"));
    let composer = tag_text(xml, "creator");
    let mut doc = Doc {
        title,
        composer,
        time: TimeSig {
            beats: 4,
            beat_type: 4,
        },
        key_fifths: 0,
        parts: Vec::new(),
    };
    let mut first_header = true;
    for part in elements(xml, "part") {
        let id = attr_value(part, "id").unwrap_or_default();
        let name = names.remove(&id).unwrap_or_else(|| "Part".to_string());
        let mut dp = DocPart {
            name,
            divisions: 480,
            notes: Vec::new(),
        };
        let mut page: u32 = 1;
        let mut cursor: i64 = 0; // part-level absolute ticks
                                 // Pending tie chains keyed by (staff, voice, step, alter, octave).
                                 // Staff-qualified: two staves sharing a voice number tie the same
                                 // pitch independently (cross-staff unison); voice-only keying
                                 // collides them and drops a chain.
        let mut ties: HashMap<(u32, u32, char, i32, i32), usize> = HashMap::new();
        // Nominal-grid start of the current measure, in file ticks. Short
        // measures (pickups) must not drag later measures off the grid:
        // score order and durations are grid-independent, but absolute
        // onsets have to sit on nominal bar lines to stay comparable.
        let mut nominal: i64 = 0;
        let mut beats: u32 = 4;
        let mut beat_type: u32 = 4;
        for meas in elements(part, "measure") {
            let mut items = measure_items(meas);
            // The signature in effect for this measure sets its nominal
            // length; attributes normally lead the measure body.
            let mut file_div = dp.divisions;
            for it in &items {
                if let Item::Attributes(a) = it {
                    if let Some(d) = a.divisions {
                        file_div = d;
                    }
                    if let Some(t) = &a.time {
                        beats = t.beats;
                        beat_type = t.beat_type;
                    }
                }
            }
            let barlen = beats as i64 * file_div as i64 * 4 / beat_type.max(1) as i64;
            let mstart = cursor.max(nominal);
            let bar_end = mstart + barlen.max(0);
            let mut pitched_mend = mstart;
            // Per-voice (next start, last onset): chord tones reuse the
            // last onset instead of the advanced cursor. Deliberately
            // NOT staff-qualified (unlike `ties`): this map is the
            // document-time clock, and MusicXML time advances globally
            // through the note sequence unless `<backup>` rewinds it —
            // the backup arm below rewinds all voices equally. Splitting
            // the clock per staff strands cross-staff voices on stale
            // clocks (ds0037 m39 D4 reads 160 instead of the true 480:
            // its staff's clock missed the other staff's triplet time).
            // The encoder (`bekern_events`) uses the same single-clock
            // model, so the oracle must agree with it, not diverge.
            let mut ends: HashMap<u32, (i64, i64)> = HashMap::new();
            for it in items.drain(..) {
                match it {
                    Item::Attributes(a) => {
                        if let Some(d) = a.divisions {
                            dp.divisions = d;
                        }
                        if first_header {
                            if let Some(t) = a.time {
                                doc.time = t;
                            }
                            if let Some(k) = a.key_fifths {
                                doc.key_fifths = k;
                            }
                        }
                    }
                    Item::NewPage(n) => {
                        page = n;
                    }
                    Item::NewSystem => {}
                    Item::Direction(_) | Item::Barline(_) => {}
                    Item::Backup(d) => {
                        for v in ends.values_mut() {
                            v.0 = (v.0 - d).max(mstart);
                        }
                    }
                    Item::Forward(d) => {
                        for v in ends.values_mut() {
                            v.0 += d;
                        }
                    }
                    Item::Note(n) => {
                        if n.grace {
                            continue;
                        }
                        let dur = n.duration.ok_or("note without duration")?;
                        let voice = n.voice;
                        // A voice entering late starts at the current
                        // document time (other voices' notes, `<forward>`
                        // gaps), not at the measure start: initializing
                        // to mstart would silently drop pre-entry
                        // silence and displace every note of the voice
                        // (the encoder fills the same gap with a rest,
                        // so the oracle must agree with it, not with 0).
                        let init = ends.values().map(|v| v.0).max().unwrap_or(mstart);
                        let (next, last) = ends.get(&voice).copied().unwrap_or((init, init));
                        let start = if n.chord { last } else { next };
                        if !n.chord {
                            ends.insert(voice, (start + dur, start));
                        }
                        // Staff-qualified (unlike `ends` above): tie chains
                        // merge per (staff, voice, pitch), so cross-staff
                        // unisons sharing a voice number merge
                        // independently instead of colliding.
                        let staff = n.staff.unwrap_or(1).max(1);
                        let key = n.pitch.map(|(st, al, oc)| (staff, voice, st, al, oc));
                        let is_stop = n.tie_stop;
                        let is_start = n.tie_start;
                        if is_stop {
                            if let Some(k) = key {
                                if let Some(&idx) = ties.get(&k) {
                                    dp.notes[idx].note.dur += ticks_to_beats(dur, dp.divisions);
                                    if !is_start {
                                        ties.remove(&k);
                                    }
                                    continue;
                                }
                            }
                        }
                        let (pitch, rest) = match n.pitch {
                            Some((st, al, oc)) => {
                                pitched_mend = pitched_mend.max(start + dur);
                                (Some(format!("{st}{}{oc}", accidental_suffix(al)?)), false)
                            }
                            None => (None, true),
                        };
                        // Bar-overflow spacing rests (e.g. Finale's trailing
                        // `print-object="no"` rests starting at or past the
                        // nominal bar end) advance `ends` so `<backup>` rewinds
                        // cleanly, but must not be emitted or push the next
                        // measure off the nominal barline grid (matches bekern).
                        let emit_dur = if rest {
                            if start >= bar_end {
                                continue;
                            }
                            dur.min(bar_end - start)
                        } else {
                            dur
                        };
                        dp.notes.push(RawNote {
                            page,
                            note: NoteInput {
                                pitch,
                                rest,
                                onset: ticks_to_beats(start, dp.divisions),
                                dur: ticks_to_beats(emit_dur, dp.divisions),
                                voice,
                                lyric: None,
                                grace: false,
                                slur_start: false,
                                slur_stop: false,
                                articulations: Vec::new(),
                                tie_start: false,
                                tie_stop: false,
                            },
                        });
                        let idx = dp.notes.len() - 1;
                        if is_start {
                            if let Some(k) = key {
                                ties.insert(k, idx);
                            }
                        }
                    }
                }
            }
            first_header = false;
            cursor = pitched_mend.max(cursor);
            nominal += barlen.max(0);
        }
        doc.parts.push(dp);
    }
    if doc.parts.is_empty() {
        return Err("no parts found".to_string());
    }
    Ok(doc)
}

fn ticks_to_beats(ticks: i64, divisions: u32) -> f64 {
    ticks as f64 / divisions as f64
}

fn attr_value(s: &str, attr: &str) -> Option<String> {
    let pat = format!("{attr}=\"");
    let p = s.find(&pat)? + pat.len();
    let e = s[p..].find('"')? + p;
    Some(s[p..e].to_string())
}

struct NoteEv {
    pitch: Option<(char, i32, i32)>,
    duration: Option<i64>,
    voice: u32,
    staff: Option<u32>,
    chord: bool,
    grace: bool,
    /// `<type>` text of a grace note (`None` for real notes); the
    /// engraved display value, since grace notes carry no `<duration>`.
    grace_type: Option<String>,
    tie_start: bool,
    tie_stop: bool,
    /// Slur spanner endpoints from `<notations><slur type=...>`.
    slur_start: bool,
    slur_stop: bool,
    /// Closed-subset modifiers in fixed order (`stacc`, `accent`,
    /// `tenuto`, `fermata`); unknown children are ignored, never errors.
    articulations: Vec<String>,
    /// `(actual, normal)` from `<time-modification>`; `None` = straight.
    tuplet_ratio: Option<(u32, u32)>,
    /// Tuplet `number` attribute of a `<tuplet type="start">` marker.
    tuplet_start: Option<u32>,
    /// Tuplet `number` attribute of a `<tuplet type="stop">` marker.
    tuplet_stop: Option<u32>,
    /// Whole-bar rest (`<rest measure="yes"/>`).
    measure_rest: bool,
}

/// Measure-level dynamic mark (`dyn-pp` … `dyn-ff`, `dyn-sfz`) and/or
/// hairpin wedge event (`hairpin-cresc-start`, `hairpin-dim-start`,
/// `hairpin-stop`) from a `<direction>` block. Last wins per block.
/// `notation_dynamic` carries the extended MusicXML dynamic set for
/// `notation_events`, while `dynamic` stays on bekern's closed subset.
struct DirEv {
    dynamic: Option<String>,
    notation_dynamic: Option<String>,
    wedge: Option<String>,
}

/// A `<barline>` element: `left` is the `location="left"` attribute,
/// `style` the `<bar-style>` text, `repeat` the `<repeat>` direction.
struct BarEv {
    left: bool,
    style: Option<String>,
    repeat: Option<String>,
}

struct AttrEv {
    divisions: Option<u32>,
    time: Option<TimeSig>,
    key_fifths: Option<i32>,
    staves: Option<u32>,
    /// `(part-local staff number, sign, line)` in document order.
    clefs: Vec<(u32, char, u32)>,
}

/// Engraved beats for a grace `<type>`; grace notes carry no duration,
/// so an absent or unknown type defaults to the common eighth.
fn grace_beats(note_type: &Option<String>) -> f64 {
    match note_type.as_deref() {
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

/// Right-barline token for a parsed `<barline>`: repeat direction wins
/// over style (mirroring the renderer's repeat-first mapping, so the
/// pair is idempotent); plain styles emit no token.
fn right_barline_token(b: &BarEv) -> Option<String> {
    if b.left {
        return None;
    }
    if b.repeat.as_deref() == Some("backward") {
        Some("bar-repeat-right".to_string())
    } else if b.style.as_deref() == Some("light-heavy") {
        Some("bar-light-heavy".to_string())
    } else if b.style.as_deref() == Some("light-light") {
        Some("bar-double".to_string())
    } else {
        None
    }
}

/// Left-barline token: only a forward repeat opens a measure.
fn left_barline_token(b: &BarEv) -> Option<String> {
    if b.left && b.repeat.as_deref() == Some("forward") {
        Some("bar-repeat-left".to_string())
    } else {
        None
    }
}

/// One pitched note event in the reference scorer's event model:
/// measure ordinal, onset within the measure (beats), MIDI pitch,
/// duration (beats), document-global staff index, step, alter.
/// Rests and grace notes are not emitted; chord tones share the
/// previous onset; backup/forward move a per-measure cursor.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreEvent {
    pub measure: u32,
    pub onset: f64,
    pub midi: i32,
    pub dur: f64,
    pub staff: u32,
    pub step: char,
    pub alter: i32,
}

/// First-seen header per document-global staff: initial clef, key and
/// time signature. Later mid-score changes are ignored — bekern headers
/// are score-initial. Missing clefs default to G/2.
#[derive(Debug, Clone, PartialEq)]
pub struct StaffHeader {
    pub staff: u32,
    pub clef_sign: char,
    pub clef_line: u32,
    pub key_fifths: i32,
    pub beats: u32,
    pub beat_type: u32,
}

/// Per-part header state while walking: global base staff, clefs keyed
/// by part-local staff number, first key/time, highest staff seen.
type PartHeader = (
    u32,
    Vec<(u32, char, u32)>,
    Option<i32>,
    Option<TimeSig>,
    u32,
);

/// Collect [`StaffHeader`]s in document-global staff order. Staff
/// indexing mirrors [`score_events`] exactly (same `staff_base`
/// accumulation), so headers line up with event `staff` fields.
pub fn staff_headers(xml: &str) -> Result<Vec<StaffHeader>, String> {
    let mut parts: Vec<PartHeader> = Vec::new();
    let mut staff_base = 0u32;
    for part_xml in elements(xml, "part") {
        let mut clefs: Vec<(u32, char, u32)> = Vec::new();
        let mut key: Option<i32> = None;
        let mut time: Option<TimeSig> = None;
        let mut part_staves = 1u32;
        let mut last_staff = 1u32;
        for meas in elements(part_xml, "measure") {
            for it in measure_items(meas) {
                match it {
                    Item::Attributes(a) => {
                        if let Some(s) = a.staves {
                            part_staves = part_staves.max(s.max(1));
                        }
                        for (n, sign, line) in a.clefs {
                            if !clefs.iter().any(|c| c.0 == n) {
                                clefs.push((n, sign, line));
                            }
                        }
                        if key.is_none() {
                            key = a.key_fifths;
                        }
                        if time.is_none() {
                            time = a.time;
                        }
                    }
                    Item::Note(n) => {
                        // Same staff tracking as score_events.
                        let local = n.staff.unwrap_or(last_staff).max(1);
                        last_staff = local;
                        part_staves = part_staves.max(local);
                    }
                    _ => {}
                }
            }
        }
        parts.push((staff_base, clefs, key, time, part_staves));
        staff_base += part_staves;
    }
    Ok(emit_headers(&parts))
}

/// Collect [`StaffHeader`]s with clef, key and time signature changes in
/// `measure <= start_measure` carried forward (last wins). Staff indexing
/// matches [`staff_headers`] exactly (same full-score `staff_base`
/// accumulation), so headers line up with [`bekern_events`] `staff`
/// fields for any measure-span slice. `start_measure` is 1-based; values
/// below 1 clamp to 1, values past the last measure behave like
/// [`staff_headers`] except clefs are last-wins instead of first-wins.
pub fn staff_headers_at_measure(
    xml: &str,
    start_measure: u32,
) -> Result<Vec<StaffHeader>, String> {
    let start = start_measure.max(1);
    let mut parts: Vec<PartHeader> = Vec::new();
    let mut staff_base = 0u32;
    for part_xml in elements(xml, "part") {
        let measures = elements(part_xml, "measure");
        // Full-score staff count for stable global indexing (same as
        // `staff_headers`): the header walk below only *applies*
        // attributes from measures at or before `start`.
        let mut part_staves = 1u32;
        let mut probe_staff = 1u32;
        for meas in &measures {
            for it in measure_items(meas) {
                match it {
                    Item::Attributes(a) => {
                        if let Some(s) = a.staves {
                            part_staves = part_staves.max(s.max(1));
                        }
                    }
                    Item::Note(n) => {
                        let local = n.staff.unwrap_or(probe_staff).max(1);
                        probe_staff = local;
                        part_staves = part_staves.max(local);
                    }
                    _ => {}
                }
            }
        }
        let mut clefs: Vec<(u32, char, u32)> = Vec::new();
        let mut key: Option<i32> = None;
        let mut time: Option<TimeSig> = None;
        for (ordinal0, meas) in measures.iter().enumerate() {
            if ordinal0 as u32 + 1 > start {
                break;
            }
            for it in measure_items(meas) {
                if let Item::Attributes(a) = it {
                    for (n, sign, line) in a.clefs {
                        match clefs.iter_mut().find(|c| c.0 == n) {
                            Some(slot) => *slot = (n, sign, line),
                            None => clefs.push((n, sign, line)),
                        }
                    }
                    if a.key_fifths.is_some() {
                        key = a.key_fifths;
                    }
                    if a.time.is_some() {
                        time = a.time;
                    }
                }
            }
        }
        parts.push((staff_base, clefs, key, time, part_staves));
        staff_base += part_staves;
    }
    Ok(emit_headers(&parts))
}

/// Expand per-part header state into document-global [`StaffHeader`]s.
/// Shared by [`staff_headers`] and [`staff_headers_at_measure`] so both
/// index staves identically.
fn emit_headers(parts: &[PartHeader]) -> Vec<StaffHeader> {
    let mut out = Vec::new();
    for (base, clefs, key, time, max_staff) in parts {
        let t = time.unwrap_or(TimeSig {
            beats: 4,
            beat_type: 4,
        });
        for local in 1..=*max_staff {
            let (sign, line) = clefs
                .iter()
                .find(|c| c.0 == local)
                .map(|c| (c.1, c.2))
                .unwrap_or(('G', 2));
            out.push(StaffHeader {
                staff: base + local - 1,
                clef_sign: sign,
                clef_line: line,
                key_fifths: key.unwrap_or(0),
                beats: t.beats,
                beat_type: t.beat_type,
            });
        }
    }
    out
}

/// Measure-scoped header state for one document-global staff: active
/// clef, key signature, and time signature at a given 1-based measure.
/// Emitted at measure 1 (initial state) and at every subsequent measure
/// where `<attributes>` changes clef, key, or time on that staff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasureStaffHeader {
    pub measure: u32,
    pub staff: u32,
    pub clef_sign: char,
    pub clef_line: u32,
    pub key_fifths: i32,
    pub beats: u32,
    pub beat_type: u32,
}

/// Collect [`MeasureStaffHeader`]s in document-global staff order:
/// measure 1 establishes each staff's opening clef/key/time, and every
/// later measure with an `<attributes>` change emits an updated record
/// for each affected staff. Staff indexing mirrors [`staff_headers`]
/// and [`notation_events`] exactly.
pub fn measure_staff_headers(xml: &str) -> Result<Vec<MeasureStaffHeader>, String> {
    let mut out: Vec<MeasureStaffHeader> = Vec::new();
    let mut staff_base = 0u32;
    for part_xml in elements(xml, "part") {
        let measures = elements(part_xml, "measure");
        let mut part_staves = 1u32;
        let mut probe_staff = 1u32;
        for meas in &measures {
            for it in measure_items(meas) {
                match it {
                    Item::Attributes(a) => {
                        if let Some(s) = a.staves {
                            part_staves = part_staves.max(s.max(1));
                        }
                    }
                    Item::Note(n) => {
                        let local = n.staff.unwrap_or(probe_staff).max(1);
                        probe_staff = local;
                        part_staves = part_staves.max(local);
                    }
                    _ => {}
                }
            }
        }
        let n_staves = part_staves as usize;
        let mut cur_clefs: Vec<(char, u32)> = vec![('G', 2); n_staves];
        let mut cur_key: i32 = 0;
        let mut cur_time: (u32, u32) = (4, 4);
        for (ordinal0, meas) in measures.iter().enumerate() {
            let measure = ordinal0 as u32 + 1;
            let mut changed = vec![measure == 1; n_staves];
            for it in measure_items(meas) {
                if let Item::Attributes(a) = it {
                    for (n, sign, line) in a.clefs {
                        let idx = n.saturating_sub(1) as usize;
                        if idx < n_staves && cur_clefs[idx] != (sign, line) {
                            cur_clefs[idx] = (sign, line);
                            changed[idx] = true;
                        }
                    }
                    if let Some(k) = a.key_fifths {
                        if cur_key != k {
                            cur_key = k;
                            changed.fill(true);
                        }
                    }
                    if let Some(t) = a.time {
                        let v = (t.beats, t.beat_type);
                        if cur_time != v {
                            cur_time = v;
                            changed.fill(true);
                        }
                    }
                }
            }
            for idx in 0..n_staves {
                if changed[idx] {
                    out.push(MeasureStaffHeader {
                        measure,
                        staff: staff_base + idx as u32,
                        clef_sign: cur_clefs[idx].0,
                        clef_line: cur_clefs[idx].1,
                        key_fifths: cur_key,
                        beats: cur_time.0,
                        beat_type: cur_time.1,
                    });
                }
            }
        }
        staff_base += part_staves;
    }
    Ok(out)
}

/// One engraved staff system: the measure span on one page between
/// `<print new-system="yes">` / `<print new-page="yes">` markers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemSpan {
    /// 1-based page number (from `page-number`, else sequential).
    pub page: u32,
    /// 0-based system index within `page`.
    pub system_idx: u32,
    /// 1-based inclusive first measure of the system.
    pub start_measure: u32,
    /// 1-based inclusive last measure of the system.
    pub end_measure: u32,
}

/// List the `(page, system, measure-span)` breaks of part 1, sourced
/// from `<print new-system="yes">` and `<print new-page="yes">` tags.
/// A new page also starts system 0 on that page. Breaks at measure 1
/// are ignored (measure 1 always opens the first span); the final span
/// closes at the last measure of part 1.
pub fn system_breaks(xml: &str) -> Vec<SystemSpan> {
    let part_xml = match elements(xml, "part").into_iter().next() {
        Some(p) => p,
        None => return Vec::new(),
    };
    let measures = elements(part_xml, "measure");
    let total = measures.len() as u32;
    if total == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut cur_page = 1u32;
    let mut sys_idx = 0u32;
    let mut span_start = 1u32;
    for (ordinal0, meas) in measures.iter().enumerate() {
        let measure = ordinal0 as u32 + 1;
        if measure <= 1 {
            continue;
        }
        let mut is_page = false;
        let mut is_system = false;
        for it in measure_items(meas) {
            match it {
                Item::NewPage(_) => is_page = true,
                Item::NewSystem => is_system = true,
                _ => {}
            }
        }
        if is_page {
            out.push(SystemSpan {
                page: cur_page,
                system_idx: sys_idx,
                start_measure: span_start,
                end_measure: measure - 1,
            });
            // Prefer the engraved `page-number`; fall back to sequential.
            cur_page = page_number_attr(meas).unwrap_or(cur_page + 1).max(1);
            sys_idx = 0;
            span_start = measure;
        } else if is_system {
            out.push(SystemSpan {
                page: cur_page,
                system_idx: sys_idx,
                start_measure: span_start,
                end_measure: measure - 1,
            });
            sys_idx += 1;
            span_start = measure;
        }
    }
    out.push(SystemSpan {
        page: cur_page,
        system_idx: sys_idx,
        start_measure: span_start,
        end_measure: total,
    });
    out
}

/// The `page-number` attribute of the first `<print ...>` tag in a
/// measure body, if present and numeric.
fn page_number_attr(meas: &str) -> Option<u32> {
    let p = meas.find("<print")?;
    let tag_end = meas[p..].find('>')? + p;
    let tag = &meas[p..tag_end];
    tag.split("page-number=\"")
        .nth(1)?
        .split('"')
        .next()?
        .parse::<u32>()
        .ok()
}

/// One note or rest for bekern emission: document-global staff, beats,
/// ties kept (the scorer's [`ScoreEvent`] drops ties and rests).
///
/// Mid-stream notation state rides on the first non-chord event at or
/// after its position: clef/key/time changes (`clef_change`,
/// `key_change`, `time_change`), measure dynamics/hairpins (`dynamic`
/// as a `dyn-*` token, `hairpin` as a `hairpin-*-start|stop` token) and
/// an opening repeat (`barline_left`). A closing barline
/// (`barline_right`) rides on the measure's last event instead. Grace
/// notes are kept (flagged); only grace rests are un-notatable.
#[derive(Debug, Clone, PartialEq)]
pub struct BekernNote {
    pub measure: u32,
    pub onset: f64,
    /// `None` = rest.
    pub pitch: Option<(char, i32, i32)>,
    /// Written beats (tuplet scaling already undone in tick space).
    /// For grace notes: the engraved `<type>` value (display only).
    pub dur_beats: f64,
    pub voice: u32,
    pub staff: u32,
    pub tie_start: bool,
    pub tie_stop: bool,
    pub grace: bool,
    pub slur_start: bool,
    pub slur_stop: bool,
    /// Closed-subset modifiers (`stacc`, `accent`, `tenuto`, `fermata`).
    pub articulations: Vec<String>,
    /// `(actual, normal)` tuplet ratio of this note, if any.
    pub tuplet_ratio: Option<(u32, u32)>,
    /// Tuplet `number` of a span starting on this note, if any.
    pub tuplet_start: Option<u32>,
    /// Tuplet `number` of a span ending on this note, if any.
    pub tuplet_stop: Option<u32>,
    /// Whole-bar rest: emitted as a lone `bar-rest` token.
    pub measure_rest: bool,
    /// Mid-stream clef change taking effect here, if any.
    pub clef_change: Option<(char, u32)>,
    /// Mid-stream key change (`fifths`), if any.
    pub key_change: Option<i32>,
    /// Mid-stream time change (`beats`, `beat-type`), if any.
    pub time_change: Option<(u32, u32)>,
    /// Measure dynamic (`dyn-*` token spelling), if any.
    pub dynamic: Option<String>,
    /// Hairpin event (`hairpin-*-start|stop` token spelling), if any.
    pub hairpin: Option<String>,
    /// Opening repeat (`bar-repeat-left`), if any.
    pub barline_left: Option<String>,
    /// Closing barline style token, if any.
    pub barline_right: Option<String>,
}

/// Flatten a Partwise 3.1 document to [`BekernNote`]s: same cursors,
/// backup/forward handling and staff indexing as [`score_events`], but
/// rests are emitted, ties are kept, and grace notes are kept flagged
/// (only grace rests are un-notatable and skipped by the tokenizer).
/// Mid-stream clef/key/time changes, measure dynamics/hairpins, and
/// barlines ride on their carrier events (see [`BekernNote`]).
pub fn bekern_events(xml: &str) -> Result<Vec<BekernNote>, String> {
    let mut out: Vec<BekernNote> = Vec::new();
    let mut staff_base = 0u32;
    for part_xml in elements(xml, "part") {
        let mut divisions = 1u32;
        let mut part_staves = 1u32;
        for meas in elements(part_xml, "measure") {
            let items = measure_items(meas);
            for it in &items {
                if let Item::Attributes(a) = it {
                    if let Some(d) = a.divisions {
                        if d > 0 {
                            divisions = d;
                        }
                    }
                    if let Some(s) = a.staves {
                        part_staves = part_staves.max(s.max(1));
                    }
                }
            }
        }
        // Notation state in effect at the part start (seeded silently by
        // the first attributes block) versus pending changes (diffs that
        // ride on the next non-chord event).
        let mut cur_clefs: HashMap<u32, (char, u32)> = HashMap::new();
        let mut cur_key: Option<i32> = None;
        let mut cur_time: Option<(u32, u32)> = None;
        let mut pend_clef: Option<(char, u32)> = None;
        let mut pend_key: Option<i32> = None;
        let mut pend_time: Option<(u32, u32)> = None;
        let mut pend_dyn: Option<String> = None;
        let mut pend_hair: Option<String> = None;
        let mut pend_bleft: Option<String> = None;
        // A closing barline seen before any carrier event of its measure
        // (unusual order): held until the measure's next event arrives.
        let mut held_bright: Option<(u32, String)> = None;
        // Index of the last pushed non-chord event of this part:
        // measure-end marks retro-stamp onto it (chord tones never
        // carry marks, keeping chord runs token-adjacent).
        let mut last_plain: Option<usize> = None;
        // Last measure ordinal of this part (for the end-of-part
        // flush guard).
        let mut last_ordinal = 0u32;
        for (ordinal0, meas) in elements(part_xml, "measure").into_iter().enumerate() {
            let measure = ordinal0 as u32 + 1;
            last_ordinal = measure;
            // A held barline whose measure produced no carrier event is
            // lost (an event-free measure carries nothing by design).
            if let Some((hm, _)) = &held_bright {
                if *hm < measure {
                    held_bright = None;
                }
            }
            // Measure-end marks (dynamics, hairpins, opening repeats)
            // pending with no later same-measure event stamp onto the
            // closing measure's last plain event; unclaimed marks die
            // with their measure (an event-free measure carries
            // nothing). Attribute changes (clef/key/time) instead keep
            // flowing forward — they take effect at their position,
            // which may be the next measure.
            if measure > 1 {
                let prev = measure - 1;
                if let Some(idx) = last_plain {
                    if out.get(idx).map(|e| e.measure) == Some(prev) {
                        if let Some(tok) = pend_dyn.take() {
                            out[idx].dynamic = Some(tok);
                        }
                        if let Some(tok) = pend_hair.take() {
                            out[idx].hairpin = Some(tok);
                        }
                        if let Some(tok) = pend_bleft.take() {
                            out[idx].barline_left = Some(tok);
                        }
                    }
                }
                pend_dyn = None;
                pend_hair = None;
                pend_bleft = None;
            }
            let div = divisions as f64;
            let mut cursor = 0i64;
            let mut last_onset = 0i64;
            let mut last_staff = 1u32;
            for it in measure_items(meas) {
                match it {
                    Item::Attributes(a) => {
                        for (n, sign, line) in a.clefs {
                            if cur_clefs.get(&n).copied() != Some((sign, line)) {
                                if cur_clefs.contains_key(&n) {
                                    pend_clef = Some((sign, line));
                                }
                                cur_clefs.insert(n, (sign, line));
                            }
                        }
                        if let Some(k) = a.key_fifths {
                            if cur_key != Some(k) {
                                if cur_key.is_some() {
                                    pend_key = Some(k);
                                }
                                cur_key = Some(k);
                            }
                        }
                        if let Some(t) = a.time {
                            let v = (t.beats, t.beat_type);
                            if cur_time != Some(v) {
                                if cur_time.is_some() {
                                    pend_time = Some(v);
                                }
                                cur_time = Some(v);
                            }
                        }
                    }
                    Item::NewPage(_) | Item::NewSystem => {}
                    Item::Direction(d) => {
                        if d.dynamic.is_some() {
                            pend_dyn = d.dynamic;
                        }
                        if d.wedge.is_some() {
                            pend_hair = d.wedge;
                        }
                    }
                    Item::Barline(b) => {
                        if let Some(tok) = left_barline_token(&b) {
                            pend_bleft = Some(tok);
                        }
                        if let Some(tok) = right_barline_token(&b) {
                            // Right barlines close the current measure:
                            // stamp its last plain event, else hold for
                            // its next event.
                            match last_plain {
                                Some(idx)
                                    if out.get(idx).map(|e| e.measure) == Some(measure) =>
                                {
                                    out[idx].barline_right = Some(tok);
                                }
                                _ => {
                                    held_bright = Some((measure, tok));
                                }
                            }
                        }
                    }
                    Item::Backup(d) => {
                        cursor -= d;
                    }
                    Item::Forward(d) => {
                        cursor += d;
                    }
                    Item::Note(n) => {
                        let mut dur = n.duration.unwrap_or(0);
                        if n.grace {
                            dur = (grace_beats(&n.grace_type) * div).round() as i64;
                        }
                        // Tuplet members carry performed ticks; recover the
                        // written duration as an exact float quotient.
                        // Tick-exactness is the wrong gate (a
                        // double-dotted-eighth triplet at divisions=12
                        // performs 7 ticks for a written 10.5 ticks):
                        // quotients that hit the duration table are
                        // exactly representable, everything else fails
                        // loudly at `duration_token`.
                        let written_beats = match n.tuplet_ratio {
                            Some((actual, normal)) => {
                                dur as f64 * actual as f64 / (normal as f64 * div)
                            }
                            None => dur as f64 / div,
                        };
                        let onset = if n.chord { last_onset } else { cursor };
                        let local = n.staff.unwrap_or(last_staff).max(1);
                        last_staff = local;
                        part_staves = part_staves.max(local);
                        let mut ev = BekernNote {
                            measure,
                            onset: round3(onset as f64 / div),
                            // NoteEv stores (step, alter, octave); normalize.
                            pitch: n.pitch.map(|(s, a, o)| (s, o, a)),
                            // No round3: decimal rounding mangles 64ths
                            // (0.0625 -> 0.062) and dotted 32nds. Exact
                            // tick quotients hit the duration table dead
                            // on; onsets keep round3 for scorer parity.
                            dur_beats: written_beats,
                            voice: n.voice,
                            staff: staff_base + local - 1,
                            tie_start: n.tie_start,
                            tie_stop: n.tie_stop,
                            grace: n.grace,
                            slur_start: n.slur_start,
                            slur_stop: n.slur_stop,
                            articulations: n.articulations,
                            tuplet_ratio: n.tuplet_ratio,
                            tuplet_start: n.tuplet_start,
                            tuplet_stop: n.tuplet_stop,
                            measure_rest: n.measure_rest,
                            clef_change: None,
                            key_change: None,
                            time_change: None,
                            dynamic: None,
                            hairpin: None,
                            barline_left: None,
                            barline_right: None,
                        };
                        // State changes attach to the next non-chord
                        // event, keeping chord runs token-adjacent.
                        if !n.chord {
                            ev.clef_change = pend_clef.take();
                            ev.key_change = pend_key.take();
                            ev.time_change = pend_time.take();
                            ev.dynamic = pend_dyn.take();
                            ev.hairpin = pend_hair.take();
                            ev.barline_left = pend_bleft.take();
                            if let Some((hm, tok)) = held_bright.take() {
                                if hm == measure {
                                    ev.barline_right = Some(tok);
                                }
                            }
                        }
                        out.push(ev);
                        if !n.chord {
                            last_plain = Some(out.len() - 1);
                            last_onset = cursor;
                            // Grace notes steal no time (mirrors
                            // `score_events`, whose grace durations are
                            // zero): the cursor holds so the following
                            // note keeps the grace's onset.
                            if !n.grace {
                                cursor += dur;
                            }
                        }
                    }
                }
            }
        }
        // Flush trailing measure-end marks of the part's last measure
        // (dynamics/hairpins/repeats after its final event ride it).
        if let Some(idx) = last_plain {
            if out.get(idx).map(|e| e.measure) == Some(last_ordinal) {
                if let Some(tok) = pend_dyn.take() {
                    out[idx].dynamic = Some(tok);
                }
                if let Some(tok) = pend_hair.take() {
                    out[idx].hairpin = Some(tok);
                }
                if let Some(tok) = pend_bleft.take() {
                    out[idx].barline_left = Some(tok);
                }
            }
        }
        staff_base += part_staves;
    }
    Ok(out)
}

/// Split a Partwise score into consecutive measure-window chunks for
/// single-system strip training (E5 Path B).
///
/// Returns `(first_measure_1based, chunk_xml)` per chunk; every chunk
/// re-emits the score's `<part-list>` and carries the running
/// clef/key/time/divisions headers so `bekern --page 1` reads it
/// standalone. Chunk `i` holds measures
/// `[i*size+1, min((i+1)*size, n)]`.
pub fn split_measures(xml: &str, size: u32) -> Result<Vec<(u32, String)>, String> {
    if size == 0 {
        return Err("split_measures: size must be >= 1".to_string());
    }
    let rs = xml
        .find("<score-partwise")
        .ok_or("split_measures: no score-partwise root")?;
    let root_open = {
        let q = xml[rs..]
            .find('>')
            .ok_or("split_measures: bad score-partwise tag")?;
        xml[rs..rs + q + 1].to_string()
    };
    let part_list = block_inclusive(xml, "part-list")
        .ok_or("split_measures: no part-list")?
        .to_string();
    let mut parts: Vec<(String, Vec<String>)> = Vec::new();
    for part_xml in elements(xml, "part") {
        let q = part_xml
            .find('>')
            .ok_or("split_measures: bad part tag")?;
        let open = part_xml[..q + 1].to_string();
        let meas: Vec<String> = elements(part_xml, "measure")
            .into_iter()
            .map(|m| m.to_string())
            .collect();
        if meas.is_empty() {
            continue;
        }
        parts.push((open, meas));
    }
    if parts.is_empty() {
        return Err("split_measures: no measures found".to_string());
    }
    let n = parts.iter().map(|(_, m)| m.len()).max().unwrap_or(0);
    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < n {
        let mut chunk =
            String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        chunk.push_str(&root_open);
        chunk.push('\n');
        chunk.push_str(&part_list);
        chunk.push('\n');
        for (open, meas) in &parts {
            // Running header state: last <attributes> block strictly
            // before this chunk; prepended when the chunk's first
            // measure carries none of its own.
            let mut carry: Option<String> = None;
            for m in meas.iter().take(idx) {
                if let Some(a) = attributes_block(m) {
                    carry = Some(a);
                }
            }
            chunk.push_str(open);
            for m in meas.iter().skip(idx).take(size as usize) {
                chunk.push_str(&emit_measure(m, carry.as_deref()));
            }
            chunk.push_str("</part>");
        }
        chunk.push_str("</score-partwise>");
        out.push((idx as u32 + 1, chunk));
        idx += size as usize;
    }
    Ok(out)
}

/// `<name ...>...</name>` slice, open tag through close tag inclusive.
fn block_inclusive<'a>(s: &'a str, name: &str) -> Option<&'a str> {
    let start = s.find(&format!("<{name}"))?;
    let after_open = start + s[start..].find('>')? + 1;
    let close = format!("</{name}>");
    let end = after_open + s[after_open..].find(&close)? + close.len();
    Some(&s[start..end])
}

/// Full `<attributes>...</attributes>` block of one measure body.
fn attributes_block(meas: &str) -> Option<String> {
    let s = meas.find("<attributes")?;
    let e = meas[s..].find("</attributes>")? + "</attributes>".len();
    Some(meas[s..s + e].to_string())
}

/// Measure body closed and de-paginated for standalone rendering:
/// `<print new-page>` becomes plain `<print>` so `page_breaks` sees a
/// single page, and the running header block is injected when the
/// measure carries no `<attributes>` of its own.
fn emit_measure(raw: &str, carry: Option<&str>) -> String {
    let mut m = raw.replace("new-page=\"yes\"", "new-page=\"no\"");
    if let Some(attrs) = carry {
        if !m.contains("<attributes") {
            let gt = m.find('>').expect("measure open tag");
            m.insert_str(gt + 1, attrs);
        }
    }
    m.push_str("</measure>");
    m
}

/// `(page number, first measure)` in page order, always starting with
/// `(1, 1)`. Sourced from `<print new-page="yes" page-number>` markers
/// (first occurrence per page across parts wins).
pub fn page_breaks(xml: &str) -> Vec<(u32, u32)> {
    let mut out = vec![(1u32, 1u32)];
    for part_xml in elements(xml, "part") {
        for (ordinal0, meas) in elements(part_xml, "measure").into_iter().enumerate() {
            let measure = ordinal0 as u32 + 1;
            for it in measure_items(meas) {
                if let Item::NewPage(n) = it {
                    if !out.iter().any(|(p, _)| *p == n) {
                        out.push((n, measure));
                    }
                }
            }
        }
    }
    out.sort();
    out
}

/// Round to 3 decimals with ties-to-even, matching the reference
/// scorer's Python `round` exactly.
pub(crate) fn round3(x: f64) -> f64 {
    (x * 1000.0).round_ties_even() / 1000.0
}

/// Flatten a Partwise 3.1 document to [`ScoreEvent`]s, mirroring the
/// reference strict-F1 scorer's event extraction (measure-relative
/// onsets, per-measure divisions, document-global staff index).
pub fn score_events(xml: &str) -> Result<Vec<ScoreEvent>, String> {
    let mut out: Vec<ScoreEvent> = Vec::new();
    let mut staff_base = 0u32;
    for part_xml in elements(xml, "part") {
        let mut divisions = 1u32;
        let mut part_staves = 1u32;
        for (ordinal0, meas) in elements(part_xml, "measure").into_iter().enumerate() {
            let measure = ordinal0 as u32 + 1;
            let items = measure_items(meas);
            for it in &items {
                if let Item::Attributes(a) = it {
                    if let Some(d) = a.divisions {
                        if d > 0 {
                            divisions = d;
                        }
                    }
                    if let Some(s) = a.staves {
                        part_staves = part_staves.max(s.max(1));
                    }
                }
            }
            let div = divisions as f64;
            let mut cursor = 0i64;
            let mut last_onset = 0i64;
            let mut last_staff = 1u32;
            for it in items {
                match it {
                    Item::Attributes(_)
                    | Item::NewPage(_)
                    | Item::NewSystem
                    | Item::Direction(_)
                    | Item::Barline(_) => {}
                    // No lower clamp: the reference lets the cursor go
                    // negative on over-long backups.
                    Item::Backup(d) => {
                        cursor -= d;
                    }
                    Item::Forward(d) => {
                        cursor += d;
                    }
                    Item::Note(n) => {
                        // Grace notes keep flowing with duration zero, so
                        // they still emit pitched events and move last_onset.
                        let mut dur = n.duration.unwrap_or(0);
                        if n.grace {
                            dur = 0;
                        }
                        let onset = if n.chord { last_onset } else { cursor };
                        let local = n.staff.unwrap_or(last_staff).max(1);
                        last_staff = local;
                        part_staves = part_staves.max(local);
                        if let Some((step, alter, octave)) = n.pitch {
                            out.push(ScoreEvent {
                                measure,
                                onset: round3(onset as f64 / div),
                                midi: (octave + 1) * 12 + step_semitone(step) + alter,
                                dur: round3(dur as f64 / div),
                                staff: staff_base + local - 1,
                                step,
                                alter,
                            });
                        }
                        if !n.chord {
                            last_onset = cursor;
                            cursor += dur;
                        }
                    }
                }
            }
        }
        staff_base += part_staves;
    }
    out.sort_by(|a, b| {
        a.measure
            .cmp(&b.measure)
            .then(a.onset.partial_cmp(&b.onset).unwrap())
            .then(a.midi.cmp(&b.midi))
            .then(a.dur.partial_cmp(&b.dur).unwrap())
            .then(a.staff.cmp(&b.staff))
            .then(a.step.cmp(&b.step))
            .then(a.alter.cmp(&b.alter))
    });
    Ok(out)
}

/// One notation-as-written event: the full engraved identity of a note,
/// rest or grace note. Unlike [`ScoreEvent`] (the reference scorer's
/// pitch/onset/duration triple), this preserves spelling (`step` +
/// `alter` + `octave`: B# is NOT C), voice, staff, rests, tuplet
/// brackets, grace notes and attachment marks. The scored page must look
/// identical to what the model saw, so enharmonic equivalence is an
/// error and every visible mark counts.
#[derive(Debug, Clone, PartialEq)]
pub struct NotationEvent {
    pub measure: u32,
    pub onset: f64,
    pub midi: i32,
    pub dur: f64,
    pub staff: u32,
    pub step: char,
    pub alter: i32,
    pub octave: i32,
    pub voice: u32,
    pub rest: bool,
    pub tuplet: bool,
    pub grace: bool,
    pub articulations: Vec<String>,
    pub slur_start: bool,
    pub slur_stop: bool,
    pub dynamics: Vec<String>,
    pub hairpins: Vec<String>,
}

/// [`score_events`] with nothing dropped: rests are emitted, voice is
/// kept, spelling is verbatim, tuplet brackets are flagged, grace notes
/// are flagged, attachment marks (articulations, slurs, dynamics,
/// hairpins) ride along. Same cursors, same staff indexing, same sort
/// family — only the event record is complete.
pub fn notation_events(xml: &str) -> Result<Vec<NotationEvent>, String> {
    let mut out: Vec<NotationEvent> = Vec::new();
    let mut staff_base = 0u32;
    for part_xml in elements(xml, "part") {
        let mut divisions = 1u32;
        let mut part_staves = 1u32;
        for (ordinal0, meas) in elements(part_xml, "measure").into_iter().enumerate() {
            let measure = ordinal0 as u32 + 1;
            let items = measure_items(meas);
            for it in &items {
                if let Item::Attributes(a) = it {
                    if let Some(d) = a.divisions {
                        if d > 0 {
                            divisions = d;
                        }
                    }
                    if let Some(s) = a.staves {
                        part_staves = part_staves.max(s.max(1));
                    }
                }
            }
            let div = divisions as f64;
            let mut cursor = 0i64;
            let mut last_onset = 0i64;
            let mut last_staff = 1u32;
            let mut pend_dyns: Vec<String> = Vec::new();
            let mut pend_hairpins: Vec<String> = Vec::new();
            let mut last_plain: Option<usize> = None;
            for it in items {
                match it {
                    Item::Attributes(_)
                    | Item::NewPage(_)
                    | Item::NewSystem
                    | Item::Barline(_) => {}
                    Item::Direction(d) => {
                        if let Some(tok) = d.notation_dynamic {
                            pend_dyns.push(tok);
                        }
                        if let Some(tok) = d.wedge {
                            pend_hairpins.push(tok);
                        }
                    }
                    Item::Backup(d) => {
                        cursor -= d;
                    }
                    Item::Forward(d) => {
                        cursor += d;
                    }
                    Item::Note(n) => {
                        let mut dur = n.duration.unwrap_or(0);
                        if n.grace {
                            dur = 0;
                        }
                        let onset = if n.chord { last_onset } else { cursor };
                        let local = n.staff.unwrap_or(last_staff).max(1);
                        last_staff = local;
                        part_staves = part_staves.max(local);
                        let (step, alter, octave, midi, rest) = match n.pitch {
                            Some((st, al, oc)) => (
                                st,
                                al,
                                oc,
                                (oc + 1) * 12 + step_semitone(st) + al,
                                false,
                            ),
                            None => ('R', 0, -1, -1, true),
                        };
                        let (mut dynamics, mut hairpins) = if !n.chord {
                            (
                                std::mem::take(&mut pend_dyns),
                                std::mem::take(&mut pend_hairpins),
                            )
                        } else {
                            (Vec::new(), Vec::new())
                        };
                        dynamics.sort();
                        hairpins.sort();
                        out.push(NotationEvent {
                            measure,
                            onset: round3(onset as f64 / div),
                            midi,
                            dur: round3(dur as f64 / div),
                            staff: staff_base + local - 1,
                            step,
                            alter,
                            octave,
                            voice: n.voice,
                            rest,
                            tuplet: n.tuplet_ratio.is_some(),
                            grace: n.grace,
                            articulations: n.articulations,
                            slur_start: n.slur_start,
                            slur_stop: n.slur_stop,
                            dynamics,
                            hairpins,
                        });
                        if !n.chord {
                            last_plain = Some(out.len() - 1);
                            last_onset = cursor;
                            cursor += dur;
                        }
                    }
                }
            }
            // Trailing directions after the measure's last event (e.g.
            // `<wedge type="stop"/>` at bar end) retro-stamp onto the
            // measure's final non-chord event.
            if (!pend_dyns.is_empty() || !pend_hairpins.is_empty()) && last_plain.is_some() {
                let idx = last_plain.unwrap();
                out[idx].dynamics.extend(pend_dyns.drain(..));
                out[idx].dynamics.sort();
                out[idx].hairpins.extend(pend_hairpins.drain(..));
                out[idx].hairpins.sort();
            }
        }
        staff_base += part_staves;
    }
    out.sort_by(|a, b| {
        a.measure
            .cmp(&b.measure)
            .then(a.onset.partial_cmp(&b.onset).unwrap())
            .then(a.midi.cmp(&b.midi))
            .then(a.dur.partial_cmp(&b.dur).unwrap())
            .then(a.staff.cmp(&b.staff))
            .then(a.step.cmp(&b.step))
            .then(a.alter.cmp(&b.alter))
            .then(a.voice.cmp(&b.voice))
            .then(a.rest.cmp(&b.rest))
            .then(a.tuplet.cmp(&b.tuplet))
            .then(a.grace.cmp(&b.grace))
    });
    Ok(out)
}

enum Item {
    Attributes(AttrEv),
    NewPage(u32),
    NewSystem,
    Backup(i64),
    Forward(i64),
    Note(NoteEv),
    Direction(DirEv),
    Barline(BarEv),
}

fn measure_items(meas: &str) -> Vec<Item> {
    // Walk the measure body in document order, dispatching on each
    // top-level element.
    let mut out = Vec::new();
    let mut rest = meas;
    // Skip the opening <measure ...> tag itself.
    if let Some(q) = rest.find('>') {
        rest = &rest[q + 1..];
    }
    while let Some(p) = rest.find('<') {
        rest = &rest[p..];
        // Exact `</measure>` only: layout tags like `</measure-layout>`
        // and `</measure-distance>` share the prefix and must not end the
        // measure (they swallowed whole MuseScore measures, hiding all
        // notes and the divisions that followed).
        if rest.starts_with("</measure>") {
            break;
        } else if rest.starts_with("<attributes") {
            let end = rest
                .find("</attributes>")
                .map(|q| q + "</attributes>".len());
            if let Some(e) = end {
                out.push(Item::Attributes(parse_attributes(&rest[..e])));
                rest = &rest[e..];
            } else {
                break;
            }
        } else if rest.starts_with("<print") {
            let e = rest.find('>').map(|q| q + 1).unwrap_or(rest.len());
            let tag = &rest[..e];
            if tag.contains("new-page=\"yes\"") {
                let n = tag
                    .split("page-number=\"")
                    .nth(1)
                    .and_then(|s| s.split('"').next())
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(1);
                out.push(Item::NewPage(n.max(1)));
                // A print tag never carries notes; continue after it.
            } else if tag.contains("new-system=\"yes\"") {
                out.push(Item::NewSystem);
            }
            rest = &rest[e..];
        } else if rest.starts_with("<note") {
            let e = rest.find("</note>").map(|q| q + "</note>".len());
            if let Some(e) = end_some(e) {
                out.push(Item::Note(parse_note(&rest[..e])));
                rest = &rest[e..];
            } else {
                break;
            }
        } else if rest.starts_with("<backup>") || rest.starts_with("<backup ") {
            let e = rest.find("</backup>").map(|q| q + "</backup>".len());
            if let Some(e) = end_some(e) {
                let d = tag_text(&rest[..e], "duration")
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(0);
                out.push(Item::Backup(d));
                rest = &rest[e..];
            } else {
                break;
            }
        } else if rest.starts_with("<forward>") || rest.starts_with("<forward ") {
            let e = rest.find("</forward>").map(|q| q + "</forward>".len());
            if let Some(e) = end_some(e) {
                let d = tag_text(&rest[..e], "duration")
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(0);
                out.push(Item::Forward(d));
                rest = &rest[e..];
            } else {
                break;
            }
        } else if rest.starts_with("<direction") {
            match rest.find("</direction>") {
                Some(q) => {
                    let e = q + "</direction>".len();
                    out.push(Item::Direction(parse_direction(&rest[..e])));
                    rest = &rest[e..];
                }
                // Malformed direction block: skip the open tag rather
                // than truncating the measure's notes.
                None => match rest.find('>') {
                    Some(q) => rest = &rest[q + 1..],
                    None => break,
                },
            }
        } else if rest.starts_with("<barline") {
            match rest.find("</barline>") {
                Some(q) => {
                    let e = q + "</barline>".len();
                    let open = rest.find('>').map(|qq| qq + 1).unwrap_or(e);
                    out.push(Item::Barline(parse_barline(
                        &rest[..open.min(e)],
                        &rest[..e],
                    )));
                    rest = &rest[e..];
                }
                None => match rest.find('>') {
                    Some(q) => rest = &rest[q + 1..],
                    None => break,
                },
            }
        } else {
            // Unknown element (direction, harmony, figured-bass, ...):
            // skip one tag.
            match rest.find('>') {
                Some(q) => rest = &rest[q + 1..],
                None => break,
            }
        }
    }
    return out;

    fn end_some(e: Option<usize>) -> Option<usize> {
        e
    }
}

fn parse_attributes(s: &str) -> AttrEv {
    let divisions = tag_text(s, "divisions").and_then(|v| v.parse::<u32>().ok());
    let time = match (tag_text(s, "beats"), tag_text(s, "beat-type")) {
        (Some(b), Some(t)) => match (b.parse::<u32>(), t.parse::<u32>()) {
            (Ok(beats), Ok(beat_type)) if beats > 0 => Some(TimeSig { beats, beat_type }),
            _ => None,
        },
        _ => None,
    };
    let key_fifths = tag_text(s, "fifths").and_then(|v| v.parse::<i32>().ok());
    let staves = tag_text(s, "staves").and_then(|v| v.parse::<u32>().ok());
    // `elements` drops opening tags, so scan clefs with attributes intact.
    let mut clefs = Vec::new();
    let mut rest = s;
    while let Some(p) = rest.find("<clef") {
        let body = &rest[p..];
        let open_end = body.find('>').unwrap_or(body.len());
        let open_tag = &body[..open_end];
        let num = open_tag
            .split("number=\"")
            .nth(1)
            .and_then(|t| t.split('"').next())
            .and_then(|t| t.parse::<u32>().ok())
            .unwrap_or(1)
            .max(1);
        let end = body.find("</clef>").map(|q| q + "</clef>".len());
        if let Some(e) = end {
            let inner = &body[..e];
            let sign = tag_text(inner, "sign")
                .and_then(|t| t.chars().next())
                .unwrap_or('G');
            let line = tag_text(inner, "line")
                .and_then(|t| t.parse::<u32>().ok())
                .unwrap_or(2);
            clefs.push((num, sign, line));
            rest = &body[e..];
        } else {
            break;
        }
    }
    AttrEv {
        divisions,
        time,
        key_fifths,
        staves,
        clefs,
    }
}

fn parse_note(s: &str) -> NoteEv {
    let grace = has_open(s, "grace");
    let chord = has_open(s, "chord");
    let rest = has_open(s, "rest");
    // Whole-bar rest (`<rest measure="yes"/>`): a single `bar-rest`
    // token regardless of meter (4.5 beats in 9/8 has no plain token).
    let measure_rest = match s.find("<rest") {
        Some(p) => {
            let t = &s[p..];
            let e = t.find('>').map(|q| q + 1).unwrap_or(t.len());
            t[..e].contains("measure=\"yes\"")
        }
        None => false,
    };
    let (mut tie_start, mut tie_stop) = (false, false);
    let mut pos = 0;
    while let Some(p) = s[pos..].find("<tie") {
        let t = &s[pos + p..];
        let e = t.find('>').map(|q| q + 1).unwrap_or(t.len());
        let tag = &t[..e];
        if tag.contains("type=\"start\"") {
            tie_start = true;
        }
        if tag.contains("type=\"stop\"") {
            tie_stop = true;
        }
        pos += p + e;
    }
    let pitch = if rest {
        None
    } else {
        let step = tag_text(s, "step")
            .and_then(|v| v.chars().next())
            .unwrap_or('C');
        let alter = tag_text(s, "alter")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0);
        let octave = tag_text(s, "octave")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(4);
        let _midi_check = (octave + 1) * 12 + step_semitone(step.to_ascii_uppercase()) + alter;
        let _ = _midi_check;
        Some((step.to_ascii_uppercase(), alter, octave))
    };
    let tuplet_ratio = match (
        tag_text(s, "actual-notes").and_then(|v| v.parse::<u32>().ok()),
        tag_text(s, "normal-notes").and_then(|v| v.parse::<u32>().ok()),
    ) {
        (Some(a), Some(n)) if a > 0 && n > 0 => Some((a, n)),
        _ => None,
    };
    let (mut tuplet_start, mut tuplet_stop) = (None, None);
    let mut tpos = 0;
    while let Some(p) = s[tpos..].find("<tuplet") {
        let t = &s[tpos + p..];
        let e = t.find('>').map(|q| q + 1).unwrap_or(t.len());
        let tag = &t[..e];
        // Malformed marker (no number): default 1, like MuseScore.
        let num = tag
            .split("number=\"")
            .nth(1)
            .and_then(|v| v.split('"').next())
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(1)
            .max(1);
        if tag.contains("type=\"start\"") {
            tuplet_start = Some(num);
        }
        if tag.contains("type=\"stop\"") {
            tuplet_stop = Some(num);
        }
        tpos += p + e;
    }
    let duration = tag_text(s, "duration").and_then(|v| v.parse::<i64>().ok());
    let grace_type = if grace { tag_text(s, "type") } else { None };
    let (slur_start, slur_stop, articulations) = parse_notations(s);
    let voice = tag_text(s, "voice")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(1);
    // Digits-only, mirroring the reference scorer's `isdigit` check
    // (a leading `+` or whitespace never counts as a staff number).
    let staff = tag_text(s, "staff").and_then(|v| {
        if !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) {
            v.parse::<u32>().ok()
        } else {
            None
        }
    });
    NoteEv {
        pitch,
        duration,
        voice,
        staff,
        chord,
        grace,
        grace_type,
        tie_start,
        tie_stop,
        slur_start,
        slur_stop,
        articulations,
        tuplet_ratio,
        tuplet_start,
        tuplet_stop,
        measure_rest,
    }
}

/// Slur endpoints + closed-subset modifiers from a note body. Unknown
/// articulation children are ignored (never errors); modifiers emit in
/// fixed order (`stacc`, `accent`, `tenuto`, `fermata`) for determinism.
fn parse_notations(s: &str) -> (bool, bool, Vec<String>) {
    let Some(block) = block_inclusive(s, "notations") else {
        return (false, false, Vec::new());
    };
    let mut start = false;
    let mut stop = false;
    let mut pos = 0;
    while let Some(p) = block[pos..].find("<slur") {
        let tag = &block[pos + p..];
        let end = tag.find('>').map(|q| q + 1).unwrap_or(tag.len());
        let t = &tag[..end];
        if t.contains("type=\"start\"") {
            start = true;
        }
        if t.contains("type=\"stop\"") {
            stop = true;
        }
        pos += p + end;
    }
    let mut arts = Vec::new();
    if let Some(ab) = block_inclusive(block, "articulations") {
        if ab.contains("<staccato") {
            arts.push("stacc".to_string());
        }
        if ab.contains("<accent") || ab.contains("strong-accent") {
            arts.push("accent".to_string());
        }
        if ab.contains("<tenuto") {
            arts.push("tenuto".to_string());
        }
    }
    if block.contains("<fermata") {
        arts.push("fermata".to_string());
    }
    (start, stop, arts)
}

/// Closed dynamic level spellings (`<f/>` -> `dyn-f`); anything else
/// (`<other-dynamics>`, …) is not in the vocabulary and stays `None`.
fn dyn_token_name(tag: &str) -> Option<&'static str> {
    match tag {
        "pp" => Some("dyn-pp"),
        "p" => Some("dyn-p"),
        "mp" => Some("dyn-mp"),
        "mf" => Some("dyn-mf"),
        "f" => Some("dyn-f"),
        "ff" => Some("dyn-ff"),
        "sfz" => Some("dyn-sfz"),
        _ => None,
    }
}

/// Extended dynamic spellings for notation-as-written scoring (superset of
/// [`dyn_token_name`]; bekern stays on the closed 7-token subset).
fn notation_dyn_token_name(tag: &str) -> Option<String> {
    if let Some(t) = dyn_token_name(tag) {
        return Some(t.to_string());
    }
    match tag {
        "pppp" | "ppp" | "fff" | "ffff" | "fp" | "sf" | "sfp" | "sfpp"
        | "sffz" | "fz" | "rf" | "rfz" | "pf" | "n" => Some(format!("dyn-{tag}")),
        _ => None,
    }
}

fn parse_direction(s: &str) -> DirEv {
    let mut dynamic = None;
    let mut notation_dynamic = None;
    if let Some(dd) = block_inclusive(s, "dynamics") {
        let open_end = dd.find('>').map(|q| q + 1).unwrap_or(dd.len());
        let inner = &dd[open_end..];
        if let Some(lt) = inner.find('<') {
            let name: String = inner[lt + 1..]
                .chars()
                .take_while(|c| !matches!(c, ' ' | '\t' | '\n' | '\r' | '/' | '>'))
                .collect();
            if name != "/dynamics" {
                dynamic = dyn_token_name(&name).map(str::to_string);
                notation_dynamic = notation_dyn_token_name(&name);
            }
        }
    }
    let mut wedge = None;
    let mut pos = 0;
    while let Some(p) = s[pos..].find("<wedge") {
        let tag = &s[pos + p..];
        let end = tag.find('>').map(|q| q + 1).unwrap_or(tag.len());
        let t = &tag[..end];
        if t.contains("type=\"crescendo\"") {
            wedge = Some("hairpin-cresc-start".to_string());
        } else if t.contains("type=\"diminuendo\"") {
            wedge = Some("hairpin-dim-start".to_string());
        } else if t.contains("type=\"stop\"") {
            wedge = Some("hairpin-stop".to_string());
        }
        pos += p + end;
    }
    DirEv {
        dynamic,
        notation_dynamic,
        wedge,
    }
}

fn parse_barline(open_tag: &str, body: &str) -> BarEv {
    let left = open_tag.contains("location=\"left\"");
    let style = tag_text(body, "bar-style");
    let mut repeat = None;
    if let Some(p) = body.find("<repeat") {
        let tag = &body[p..];
        let end = tag.find('>').map(|q| q + 1).unwrap_or(tag.len());
        let t = &tag[..end];
        if t.contains("direction=\"forward\"") {
            repeat = Some("forward".to_string());
        } else if t.contains("direction=\"backward\"") {
            repeat = Some("backward".to_string());
        }
    }
    BarEv {
        left,
        style,
        repeat,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bekern_events, page_breaks, parse_attributes, score_input_from_musicxml,
        score_inputs_by_page, split_measures, staff_headers, staff_headers_at_measure,
        system_breaks, SystemSpan,
    };
    use crate::SCORE_DIVISIONS;

    const TIED: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>4</divisions><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note default-x="109.89"><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><tie type="start"/></note>
</measure>
<measure number="2">
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><tie type="stop"/></note>
<note><pitch><step>D</step><alter>1</alter><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
<note><rest/><duration>4</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn import_merges_ties_and_keeps_rests() {
        let input = score_input_from_musicxml(TIED).expect("import parses");
        // House grid, not the file's divisions=4: beats are unaffected.
        assert_eq!(input.divisions, SCORE_DIVISIONS);
        assert_eq!(input.parts.len(), 1);
        let notes = &input.parts[0].notes;
        assert_eq!(notes.len(), 3);
        // Measure 1 holds one quarter of 4/4 (a pickup-length bar), so
        // measure 2 starts on the nominal grid at 4.0 beats. The tie merge
        // (dur 2.0) and the kept rest are unaffected by the snap.
        assert_eq!(notes[0].pitch.as_deref(), Some("C4"));
        assert_eq!((notes[0].onset, notes[0].dur), (0.0, 2.0));
        assert_eq!(notes[1].pitch.as_deref(), Some("D#4"));
        assert_eq!((notes[1].onset, notes[1].dur), (5.0, 1.0));
        assert!(notes[2].rest);
        assert_eq!((notes[2].onset, notes[2].dur), (6.0, 1.0));
    }

    const PAGED: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>2</divisions><time><beats>2</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
</measure>
<measure number="2">
<print new-page="yes" page-number="2"/>
<note><pitch><step>G</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    const PICKUP: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
</measure>
<measure number="2">
<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    const EVENTS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list>
<score-part id="P1"><part-name>Voice</part-name></score-part>
<score-part id="P2"><part-name>Bass</part-name></score-part>
</part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>32</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice></note>
</measure>
<measure number="2">
<note><pitch><step>D</step><octave>4</octave></pitch><duration>32</duration><voice>1</voice></note>
<note><grace/><pitch><step>D</step><octave>5</octave></pitch><voice>1</voice></note>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>32</duration><voice>1</voice></note>
<note><chord/><pitch><step>G</step><octave>4</octave></pitch><duration>32</duration><voice>1</voice></note>
<note><pitch><step>F</step><octave>4</octave></pitch><duration>32</duration><voice>1</voice></note>
<note><pitch><step>G</step><octave>4</octave></pitch><duration>32</duration><voice>1</voice></note>
</measure>
<measure number="3">
<attributes><time><beats>2</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>A</step><octave>4</octave></pitch><duration>32</duration><voice>1</voice></note>
<note><pitch><step>B</step><octave>4</octave></pitch><duration>32</duration><voice>1</voice></note>
</measure>
<measure number="4">
<attributes><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>5</octave></pitch><duration>128</duration><voice>1</voice></note>
<backup><duration>128</duration></backup>
<note><pitch><step>A</step><octave>3</octave></pitch><duration>64</duration><voice>2</voice><staff>1</staff></note>
<note><pitch><step>G</step><octave>3</octave></pitch><duration>64</duration><voice>2</voice><staff>1</staff></note>
</measure>
</part>
<part id="P2">
<measure number="1">
<attributes><divisions>32</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>3</octave></pitch><duration>32</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    // Cross-staff unison ties (ds0462 shape): staff 1 and staff 2 tie
    // the same pitch with overlapping chains. Voice-only tie/cursor
    // maps collide them; staff-qualified maps merge each chain
    // independently.
    const CROSS_STAFF_TIES: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>4</divisions><staves>2</staves><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff><tie type="start"/></note>
<backup><duration>16</duration></backup>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>2</staff><tie type="start"/></note>
</measure>
<measure number="2">
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff><tie type="stop"/></note>
<backup><duration>16</duration></backup>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>2</staff><tie type="stop"/></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn cross_staff_unison_ties_merge_independently() {
        let input = score_input_from_musicxml(CROSS_STAFF_TIES).expect("import parses");
        let mut got: Vec<(String, f64, f64)> = input.parts[0]
            .notes
            .iter()
            .filter(|n| !n.rest)
            .map(|n| (n.pitch.clone().unwrap(), n.onset, n.dur))
            .collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            got,
            vec![("C4".to_string(), 0.0, 2.0), ("C4".to_string(), 0.0, 2.0),]
        );
    }

    #[test]
    fn score_events_mirror_the_reference_event_model() {
        use super::{score_events, ScoreEvent};
        let ev = score_events(EVENTS).expect("events parse");
        // Sorted by (measure, onset, midi, …); grace note absent.
        // Staff indices are 0-based, matching the reference scorer
        // (document-global staff base plus staff minus one).
        let want = vec![
            ScoreEvent {
                measure: 1,
                onset: 0.0,
                midi: 48,
                dur: 1.0,
                staff: 1,
                step: 'C',
                alter: 0,
            },
            ScoreEvent {
                measure: 1,
                onset: 0.0,
                midi: 60,
                dur: 0.5,
                staff: 0,
                step: 'C',
                alter: 0,
            },
            ScoreEvent {
                measure: 2,
                onset: 0.0,
                midi: 62,
                dur: 1.0,
                staff: 0,
                step: 'D',
                alter: 0,
            },
            ScoreEvent {
                measure: 2,
                onset: 1.0,
                midi: 64,
                dur: 1.0,
                staff: 0,
                step: 'E',
                alter: 0,
            },
            ScoreEvent {
                measure: 2,
                onset: 1.0,
                midi: 67,
                dur: 1.0,
                staff: 0,
                step: 'G',
                alter: 0,
            },
            ScoreEvent {
                measure: 2,
                onset: 1.0,
                midi: 74,
                dur: 0.0,
                staff: 0,
                step: 'D',
                alter: 0,
            },
            ScoreEvent {
                measure: 2,
                onset: 2.0,
                midi: 65,
                dur: 1.0,
                staff: 0,
                step: 'F',
                alter: 0,
            },
            ScoreEvent {
                measure: 2,
                onset: 3.0,
                midi: 67,
                dur: 1.0,
                staff: 0,
                step: 'G',
                alter: 0,
            },
            ScoreEvent {
                measure: 3,
                onset: 0.0,
                midi: 69,
                dur: 1.0,
                staff: 0,
                step: 'A',
                alter: 0,
            },
            ScoreEvent {
                measure: 3,
                onset: 1.0,
                midi: 71,
                dur: 1.0,
                staff: 0,
                step: 'B',
                alter: 0,
            },
            ScoreEvent {
                measure: 4,
                onset: 0.0,
                midi: 57,
                dur: 2.0,
                staff: 0,
                step: 'A',
                alter: 0,
            },
            ScoreEvent {
                measure: 4,
                onset: 0.0,
                midi: 72,
                dur: 4.0,
                staff: 0,
                step: 'C',
                alter: 0,
            },
            ScoreEvent {
                measure: 4,
                onset: 2.0,
                midi: 55,
                dur: 2.0,
                staff: 0,
                step: 'G',
                alter: 0,
            },
        ];
        assert_eq!(ev, want);
    }

    // Notation-as-written fixture: B-sharp (NOT C), a second voice, a
    // rest, a triplet span, a staccato mark and a slur pair. The
    // notation scorer must price the page as engraved: B#4 (midi 60)
    // is a different event from C4 (midi 60).
    const NOTATION: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>4</divisions><staves>2</staves><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>B</step><alter>1</alter><octave>3</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff><notations><articulations><staccato/></articulations></notations></note>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>2</voice><staff>2</staff></note>
<note><rest/><duration>4</duration><voice>1</voice><staff>1</staff></note>
<note><pitch><step>D</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice><staff>1</staff><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="start" number="1"/><slur type="start" number="1"/></notations></note>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice><staff>1</staff><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="stop" number="1"/><slur type="stop" number="1"/></notations></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn notation_events_preserve_spelling_voice_and_marks() {
        use super::notation_events;
        let ev = notation_events(NOTATION).expect("notation events parse");
        // B#3 (midi 60) keeps step B / alter +1: never normalized to C.
        let bsharp = ev
            .iter()
            .find(|e| e.midi == 60 && !e.rest)
            .expect("B# event present");
        assert_eq!((bsharp.step, bsharp.alter, bsharp.octave), ('B', 1, 3));
        // Second voice on staff 2 keeps its voice number.
        assert!(ev.iter().any(|e| e.voice == 2 && e.staff == 1));
        // The rest is emitted with its duration.
        let rest = ev.iter().find(|e| e.rest).expect("rest present");
        assert_eq!((rest.dur, rest.voice), (1.0, 1));
        // Both triplet notes carry the tuplet flag.
        assert_eq!(ev.iter().filter(|e| e.tuplet).count(), 2);
        // Staccato rides on the B#, slur spans D-E.
        assert_eq!(bsharp.articulations, vec!["stacc".to_string()]);
        let d = ev.iter().find(|e| e.midi == 62).expect("D present");
        let e = ev.iter().find(|e| e.midi == 64).expect("E present");
        assert!(d.slur_start && !d.slur_stop);
        assert!(e.slur_stop && !e.slur_start);
    }

    #[test]
    fn notation_events_extract_dynamics_and_hairpins_including_trailing() {
        use super::notation_events;
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<direction><direction-type><dynamics><mf/></dynamics></direction-type></direction>
<direction><direction-type><wedge type="crescendo"/></direction-type></direction>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
<note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
<direction><direction-type><dynamics><ppp/></dynamics></direction-type></direction>
<note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
<direction><direction-type><wedge type="stop"/></direction-type></direction>
</measure>
</part>
</score-partwise>"#;
        let ev = notation_events(xml).expect("notation events parse");
        assert_eq!(ev.len(), 3);
        let c4 = ev.iter().find(|e| e.step == 'C').expect("C4");
        let e4 = ev.iter().find(|e| e.step == 'E').expect("E4 chord tone");
        let g4 = ev.iter().find(|e| e.step == 'G').expect("G4");
        assert_eq!(c4.dynamics, vec!["dyn-mf".to_string()]);
        assert_eq!(c4.hairpins, vec!["hairpin-cresc-start".to_string()]);
        // Chord tone never steals directions.
        assert!(e4.dynamics.is_empty() && e4.hairpins.is_empty());
        // Extended dynamic + trailing wedge stop retro-stamped onto last plain event.
        assert_eq!(g4.dynamics, vec!["dyn-ppp".to_string()]);
        assert_eq!(g4.hairpins, vec!["hairpin-stop".to_string()]);
    }

    #[test]
    fn notation_events_flag_grace_notes_distinct_from_normal() {
        use super::notation_events;
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><grace/><pitch><step>D</step><octave>5</octave></pitch><voice>1</voice><type>eighth</type></note>
<note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;
        let ev = notation_events(xml).expect("notation events parse");
        assert_eq!(ev.len(), 2);
        let grace = ev.iter().find(|e| e.grace).expect("grace note present");
        let normal = ev.iter().find(|e| !e.grace).expect("normal note present");
        assert_eq!((grace.step, grace.octave, grace.onset, grace.dur), ('D', 5, 0.0, 0.0));
        assert_eq!((normal.step, normal.octave, normal.onset, normal.dur), ('C', 5, 0.0, 1.0));
    }

    #[test]
    fn measure_staff_headers_capture_midstream_clef_key_time_changes() {
        use super::{measure_staff_headers, MeasureStaffHeader};
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>4</divisions><staves>2</staves><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time><clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff></note>
</measure>
<measure number="2">
<attributes><clef number="1"><sign>G</sign><line>2</line></clef></attributes>
<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff></note>
</measure>
<measure number="3">
<attributes><clef number="2"><sign>G</sign><line>2</line></clef></attributes>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff></note>
</measure>
<measure number="4">
<attributes><key><fifths>-2</fifths></key><time><beats>3</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>F</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff></note>
</measure>
</part>
</score-partwise>"#;
        let got = measure_staff_headers(xml).expect("measure headers parse");
        assert_eq!(
            got,
            vec![
                MeasureStaffHeader {
                    measure: 1,
                    staff: 0,
                    clef_sign: 'G',
                    clef_line: 2,
                    key_fifths: 0,
                    beats: 4,
                    beat_type: 4,
                },
                MeasureStaffHeader {
                    measure: 1,
                    staff: 1,
                    clef_sign: 'F',
                    clef_line: 4,
                    key_fifths: 0,
                    beats: 4,
                    beat_type: 4,
                },
                // Measure 2 had a redundant G/2 on staff 1: no transition emitted.
                // Measure 3 changed only staff 2 (index 1) from F/4 to G/2.
                MeasureStaffHeader {
                    measure: 3,
                    staff: 1,
                    clef_sign: 'G',
                    clef_line: 2,
                    key_fifths: 0,
                    beats: 4,
                    beat_type: 4,
                },
                // Measure 4 changed key + time for both staves of the part.
                MeasureStaffHeader {
                    measure: 4,
                    staff: 0,
                    clef_sign: 'G',
                    clef_line: 2,
                    key_fifths: -2,
                    beats: 3,
                    beat_type: 4,
                },
                MeasureStaffHeader {
                    measure: 4,
                    staff: 1,
                    clef_sign: 'G',
                    clef_line: 2,
                    key_fifths: -2,
                    beats: 3,
                    beat_type: 4,
                },
            ]
        );
    }

    #[test]
    fn import_snaps_measures_to_the_nominal_grid() {
        // Measure 1 is a half-beat pickup: measure 2 must start on the
        // nominal grid (4.0 beats), not at the running cursor (0.5).
        let input = score_input_from_musicxml(PICKUP).expect("import parses");
        let notes = &input.parts[0].notes;
        assert_eq!(notes.len(), 3);
        assert_eq!((notes[0].onset, notes[0].dur), (0.0, 0.5));
        assert_eq!((notes[1].onset, notes[1].dur), (4.0, 1.0));
        assert_eq!((notes[2].onset, notes[2].dur), (5.0, 1.0));
    }

    #[test]
    fn import_emits_house_grid_divisions() {
        // Beats are grid-independent; the emitted tick grid must be the
        // house 480 so SER tick symbols compare with mapper output
        // (boxnote-mapper also emits 480), whatever the file used.
        let input = score_input_from_musicxml(TIED).expect("import parses");
        assert_eq!(input.divisions, 480);
        assert_eq!(
            (input.parts[0].notes[0].onset, input.parts[0].notes[0].dur),
            (0.0, 2.0)
        );
        let pages = score_inputs_by_page(PAGED).expect("import parses");
        assert!(pages.iter().all(|p| p.input.divisions == 480));
    }

    #[test]
    fn import_splits_pages_at_page_breaks() {
        let pages = score_inputs_by_page(PAGED).expect("import parses");
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].page, 1);
        assert_eq!(pages[0].input.parts[0].notes.len(), 1);
        assert_eq!(
            pages[0].input.parts[0].notes[0].pitch.as_deref(),
            Some("E4")
        );
        assert_eq!(pages[1].page, 2);
        assert_eq!(
            pages[1].input.parts[0].notes[0].pitch.as_deref(),
            Some("G4")
        );
        // Whole-score import ignores page structure.
        let whole = score_input_from_musicxml(PAGED).expect("import parses");
        assert_eq!(whole.parts[0].notes.len(), 2);
    }

    const CHORD_BACKUP: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>4</divisions></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
<note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
<backup><duration>4</duration></backup>
<note><pitch><step>G</step><octave>3</octave></pitch><duration>4</duration><voice>5</voice></note>
</measure>
</part>
</score-partwise>"#;

    /// MuseScore real-world header: `<print>` carries nested layout tags
    /// (`system-layout`, `measure-layout`) before `<attributes>`, and the
    /// score uses divisions=6. The nested print content must not swallow
    /// the attributes: duration 24 ticks at divisions 6 is 4.0 beats, not
    /// 24/480 = 0.05 (sample01 refs came out 80x too small this way).
    const PRINT_HEADER_DIV6: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1" width="178.5">
<print>
<system-layout><system-margins><left-margin>115.34</left-margin><right-margin>0</right-margin></system-margins>
<top-system-distance>434.17</top-system-distance></system-layout>
<measure-layout><measure-distance>30</measure-distance></measure-layout>
</print>
<attributes><divisions>6</divisions><key><fifths>0</fifths></key><time><beats>2</beats><beat-type>2</beat-type></time></attributes>
<direction placement="above"><direction-type><words>Lento non troppo.</words></direction-type><sound tempo="108"/></direction>
<note><rest measure="yes"/><duration>24</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn import_reads_divisions_after_print_header() {
        let input = score_input_from_musicxml(PRINT_HEADER_DIV6).expect("import parses");
        assert_eq!(input.parts.len(), 1);
        let notes = &input.parts[0].notes;
        assert_eq!(notes.len(), 1);
        assert!(notes[0].rest);
        assert_eq!((notes[0].onset, notes[0].dur), (0.0, 4.0));
    }

    #[test]
    fn import_bisect_print_vs_direction() {
        // Same note with NO print header and NO direction: must parse.
        let bare = PRINT_HEADER_DIV6
            .replace("<print>\n<system-layout><system-margins><left-margin>115.34</left-margin><right-margin>0</right-margin></system-margins>\n<top-system-distance>434.17</top-system-distance></system-layout>\n<measure-layout><measure-distance>30</measure-distance></measure-layout>\n</print>\n", "")
            .replace("<direction placement=\"above\"><direction-type><words>Lento non troppo.</words></direction-type><sound tempo=\"108\"/></direction>\n", "");
        let input = score_input_from_musicxml(&bare).expect("bare parses");
        assert_eq!(input.parts[0].notes.len(), 1);
        // With direction but no print.
        let nodir = PRINT_HEADER_DIV6.replace("<direction placement=\"above\"><direction-type><words>Lento non troppo.</words></direction-type><sound tempo=\"108\"/></direction>\n", "");
        let input = score_input_from_musicxml(&nodir).expect("noprint-direction parses");
        assert_eq!(input.parts[0].notes.len(), 1);
    }

    #[test]
    fn import_handles_chords_and_backup_voices() {
        let input = score_input_from_musicxml(CHORD_BACKUP).expect("import parses");
        let notes = &input.parts[0].notes;
        assert_eq!(notes.len(), 3);
        assert_eq!(notes[0].pitch.as_deref(), Some("C4"));
        assert_eq!(notes[0].voice, 1);
        assert_eq!(notes[1].pitch.as_deref(), Some("E4"));
        assert_eq!((notes[1].onset, notes[1].dur), (0.0, 1.0));
        assert_eq!(notes[2].pitch.as_deref(), Some("G3"));
        assert_eq!(notes[2].voice, 5);
        assert_eq!((notes[2].onset, notes[2].dur), (0.0, 1.0));
    }

    const CLEFS_KEYS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Piano</part-name></score-part><score-part id="P2"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>2</divisions><staves>2</staves><clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef><key><fifths>-1</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice><staff>1</staff></note>
<note><pitch><step>E</step><octave>2</octave></pitch><duration>2</duration><voice>1</voice><staff>2</staff></note>
</measure>
</part>
<part id="P2">
<measure number="1">
<attributes><divisions>2</divisions><clef><sign>C</sign><line>3</line></clef><key><fifths>0</fifths></key><time><beats>3</beats><beat-type>4</beat-type></time></attributes>
<note><rest/><duration>6</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn attributes_parse_clefs_per_staff() {
        let a = parse_attributes(
            "<divisions>2</divisions><staves>2</staves>\
             <clef number=\"1\"><sign>G</sign><line>2</line></clef>\
             <clef number=\"2\"><sign>F</sign><line>4</line></clef>",
        );
        assert_eq!(a.clefs, vec![(1, 'G', 2), (2, 'F', 4)]);
        // Missing number defaults to staff 1.
        let b = parse_attributes("<clef><sign>C</sign><line>3</line></clef>");
        assert_eq!(b.clefs, vec![(1, 'C', 3)]);
    }

    #[test]
    fn page_breaks_start_at_one_one() {
        // No print markers: single page.
        assert_eq!(page_breaks(CLEFS_KEYS), vec![(1, 1)]);
    }

    const THREE_BARS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>2</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
</measure>
<measure number="2">
<note><pitch><step>D</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
</measure>
<measure number="3">
<note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn split_measures_chunks_consecutive_bars_with_headers() {
        let chunks = split_measures(THREE_BARS, 2).expect("split parses");
        assert_eq!(chunks.len(), 2);
        assert_eq!((chunks[0].0, chunks[1].0), (1, 3));
        for (first, xml) in &chunks {
            // Each chunk is standalone: part-list present, events parse,
            // headers carried (G/2, key+0, 4/4 on staff 0).
            assert!(xml.contains("<part-list>"), "chunk {first}: part-list");
            let evs = bekern_events(xml).expect("chunk parses");
            assert!(!evs.is_empty(), "chunk {first}: notes");
            let h = staff_headers(xml).expect("headers parse");
            assert_eq!((h[0].clef_sign, h[0].clef_line), ('G', 2));
            assert_eq!((h[0].key_fifths, h[0].beats, h[0].beat_type), (0, 4, 4));
        }
        // Chunk 1 holds bars 1-2 (C4, D4), chunk 2 holds bar 3 (E4).
        let names = |xml: &str| {
            bekern_events(xml)
                .unwrap()
                .iter()
                .filter_map(|e| e.pitch)
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&chunks[0].1), vec![('C', 4, 0), ('D', 4, 0)]);
        assert_eq!(names(&chunks[1].1), vec![('E', 4, 0)]);
    }

    #[test]
    fn bekern_events_keep_rests_ties_and_flags_grace() {
        let evs = bekern_events(CLEFS_KEYS).expect("events parse");
        // Two pitched notes + one rest; no ties, no grace here.
        assert_eq!(evs.len(), 3);
        assert_eq!(evs[0].pitch, Some(('C', 4, 0)));
        assert_eq!((evs[0].onset, evs[0].dur_beats), (0.0, 1.0));
        assert_eq!((evs[0].voice, evs[0].staff), (1, 0));
        assert_eq!(evs[1].pitch, Some(('E', 2, 0)));
        assert_eq!(evs[1].staff, 1);
        assert_eq!(evs[2].pitch, None);
        assert_eq!((evs[2].onset, evs[2].dur_beats), (0.0, 3.0));
        assert!(!evs.iter().any(|e| e.tie_start || e.tie_stop || e.grace));
    }

    #[test]
    fn staff_headers_carry_first_clef_key_time_per_staff() {
        let h = staff_headers(CLEFS_KEYS).expect("headers parse");
        assert_eq!(h.len(), 3);
        assert_eq!((h[0].staff, h[0].clef_sign, h[0].clef_line), (0, 'G', 2));
        assert_eq!((h[0].key_fifths, h[0].beats, h[0].beat_type), (-1, 4, 4));
        assert_eq!((h[1].staff, h[1].clef_sign, h[1].clef_line), (1, 'F', 4));
        assert_eq!((h[2].staff, h[2].clef_sign, h[2].clef_line), (2, 'C', 3));
        assert_eq!((h[2].key_fifths, h[2].beats, h[2].beat_type), (0, 3, 4));
    }

    /// Two pages, two systems each, with a key change at the page break:
    /// system spans split at `new-system` / `new-page`, and headers at a
    /// mid-score measure carry the change (score-initial headers do not).
    const SYSTEMS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>2</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice></note>
</measure>
<measure number="2">
<print new-system="yes"/>
<note><pitch><step>D</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice></note>
</measure>
<measure number="3">
<print new-page="yes" page-number="2"/>
<attributes><key><fifths>2</fifths></key></attributes>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice></note>
</measure>
<measure number="4">
<print new-system="yes"/>
<note><pitch><step>F</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn system_breaks_split_multi_page_multi_system_snippet() {
        assert_eq!(
            system_breaks(SYSTEMS),
            vec![
                SystemSpan {
                    page: 1,
                    system_idx: 0,
                    start_measure: 1,
                    end_measure: 1,
                },
                SystemSpan {
                    page: 1,
                    system_idx: 1,
                    start_measure: 2,
                    end_measure: 2,
                },
                SystemSpan {
                    page: 2,
                    system_idx: 0,
                    start_measure: 3,
                    end_measure: 3,
                },
                SystemSpan {
                    page: 2,
                    system_idx: 1,
                    start_measure: 4,
                    end_measure: 4,
                },
            ]
        );
    }

    #[test]
    fn staff_headers_at_measure_carries_mid_score_key_change() {
        // Score-initial headers keep the opening key.
        let h0 = staff_headers(SYSTEMS).expect("headers parse");
        assert_eq!((h0[0].key_fifths, h0[0].beats, h0[0].beat_type), (0, 4, 4));
        // Span starts carry changes at or before the span start.
        let h1 = staff_headers_at_measure(SYSTEMS, 1).expect("headers parse");
        assert_eq!(h1[0].key_fifths, 0);
        let h3 = staff_headers_at_measure(SYSTEMS, 3).expect("headers parse");
        assert_eq!((h3[0].key_fifths, h3[0].beats, h3[0].beat_type), (2, 4, 4));
        // Clef from measure 1 still applies; staff indexing is unchanged.
        assert_eq!((h3[0].staff, h3[0].clef_sign, h3[0].clef_line), (0, 'G', 2));
        let h4 = staff_headers_at_measure(SYSTEMS, 4).expect("headers parse");
        assert_eq!(h4[0].key_fifths, 2);
    }

    const DS0001: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data-gen/score-corpus/big500-renders/ds0001.musicxml"
    );

    #[test]
    fn system_breaks_match_ds0001_engraving() {
        let xml = std::fs::read_to_string(DS0001).expect("read ds0001.musicxml");
        let spans = system_breaks(&xml);
        let head: Vec<SystemSpan> = spans.into_iter().take(7).collect();
        assert_eq!(
            head,
            vec![
                SystemSpan {
                    page: 1,
                    system_idx: 0,
                    start_measure: 1,
                    end_measure: 5,
                },
                SystemSpan {
                    page: 1,
                    system_idx: 1,
                    start_measure: 6,
                    end_measure: 9,
                },
                SystemSpan {
                    page: 1,
                    system_idx: 2,
                    start_measure: 10,
                    end_measure: 13,
                },
                SystemSpan {
                    page: 2,
                    system_idx: 0,
                    start_measure: 14,
                    end_measure: 17,
                },
                SystemSpan {
                    page: 2,
                    system_idx: 1,
                    start_measure: 18,
                    end_measure: 21,
                },
                SystemSpan {
                    page: 2,
                    system_idx: 2,
                    start_measure: 22,
                    end_measure: 25,
                },
                SystemSpan {
                    page: 2,
                    system_idx: 3,
                    start_measure: 26,
                    end_measure: 28,
                },
            ]
        );
    }

    #[test]
    fn bekern_measures_six_to_nine_decodes_cleanly() {
        use crate::bekern::to_bekern;
        use crate::decode::decode_bekern;
        let xml = std::fs::read_to_string(DS0001).expect("read ds0001.musicxml");
        let events: Vec<_> = bekern_events(&xml)
            .expect("events parse")
            .into_iter()
            .filter(|e| e.measure >= 6 && e.measure <= 9)
            .collect();
        assert!(!events.is_empty(), "measures 6..9 hold notes");
        assert!(
            events.iter().all(|e| (6..=9).contains(&e.measure)),
            "span filter keeps only 6..9"
        );
        let headers = staff_headers_at_measure(&xml, 6).expect("headers parse");
        assert!(!headers.is_empty());
        let (toks, _) = to_bekern(&events, &headers).expect("bekern emits");
        assert!(!toks.is_empty());
        let decoded = decode_bekern(&toks.join(" ")).expect("measures 6..9 decode");
        let n: usize = decoded.parts.iter().map(|p| p.notes.len()).sum();
        assert!(n > 0, "decoded span holds notes");
    }

    #[test]
    fn import_ignores_bar_overflow_spacing_rests_for_grid() {
        // Finale/music21 exports trailing `print-object="no"` spacing rests
        // past the nominal bar end (e.g. weber-concertino Piano m1: 30240
        // ticks of 3/4 + 1260-tick invisible 32nd rest, followed by
        // `<backup><duration>31500</duration></backup>`). Those rests must
        // keep `<backup>` aligned inside m1 without pushing m2 off the
        // 3.0-beat nominal grid.
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>10080</divisions>
        <key><fifths>-3</fifths></key>
        <time><beats>3</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>30240</duration><voice>1</voice><type>half</type><dot/><staff>1</staff></note>
      <note print-object="no" print-spacing="yes"><rest/><duration>1260</duration><voice>1</voice><type>32nd</type><staff>1</staff></note>
      <backup><duration>31500</duration></backup>
      <note><pitch><step>C</step><octave>3</octave></pitch><duration>30240</duration><voice>2</voice><type>half</type><dot/><staff>2</staff></note>
      <note print-object="no" print-spacing="yes"><rest/><duration>1260</duration><voice>2</voice><type>32nd</type><staff>2</staff></note>
    </measure>
    <measure number="2">
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>10080</duration><voice>1</voice><type>quarter</type><staff>1</staff></note>
    </measure>
  </part>
</score-partwise>"#;
        let input = score_input_from_musicxml(xml).expect("import parses");
        let notes = &input.parts[0].notes;
        assert_eq!(notes.len(), 3, "overflow spacing rests are not emitted");
        assert_eq!((notes[0].pitch.as_deref(), notes[0].onset), (Some("C5"), 0.0));
        assert_eq!((notes[1].pitch.as_deref(), notes[1].onset), (Some("C3"), 0.0));
        assert_eq!((notes[2].pitch.as_deref(), notes[2].onset), (Some("G4"), 3.0));
    }
}
