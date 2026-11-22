//! Glyph boxes/labels plus staff layout -> note-list JSON.
//!
//! Naive rules now (the honest-baseline mapper): notehead glyphs become
//! quarter (black) or half (void) notes at treble-clef diatonic pitches,
//! `rest` glyphs become quarter rests, everything else is ignored. One
//! monophonic voice, onsets accumulate in reading order. A learned mapper
//! replaces this behind the same function signature later.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct BBox {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GlyphIn {
    pub bbox: BBox,
    #[serde(default)]
    pub label: String,
    /// Detector confidence in [0, 1] when the glyph doc records it
    /// (symbol-detect serializes `confidence`); absent in older docs.
    #[serde(default)]
    pub confidence: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Crop {
    #[serde(default)]
    pub glyphs: Vec<GlyphIn>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GlyphDoc {
    #[serde(default)]
    pub crops: Vec<Crop>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Staff {
    #[serde(default)]
    pub bbox: [f64; 4],
    #[serde(default)]
    pub lines: Vec<f64>,
    /// Tilt of the staff lines as dy/dx (0 = horizontal). `lines` holds
    /// y-values at x == `pivot` (e.g. page rotation from photography).
    #[serde(default)]
    pub slope: f64,
    /// X coordinate at which `lines` values hold. Defaults to 0.
    #[serde(default)]
    pub pivot: f64,
}

/// Y of a staff line at glyph x with tilt correction.
fn line_at(staff: &Staff, line: f64, x: f64) -> f64 {
    line + staff.slope * (x - staff.pivot)
}

/// Distance from a glyph center to the nearest tilted staff line.
fn staff_line_distance(staff: &Staff, cx: f64, cy: f64) -> f64 {
    staff
        .lines
        .iter()
        .map(|l| (cy - line_at(staff, *l, cx)).abs())
        .fold(f64::INFINITY, f64::min)
}

#[derive(Debug, Clone, Deserialize)]
pub struct System {
    #[serde(default)]
    pub staves: Vec<Staff>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayoutDoc {
    #[serde(default)]
    pub systems: Vec<System>,
}

/// One mapped note event: pitch (None = rest), onset and duration in beats.
#[derive(Debug, Clone, PartialEq)]
pub struct MappedNote {
    pub pitch: Option<String>,
    pub onset: f64,
    pub dur: f64,
}

/// One voiced note event for multi-part output: like [`MappedNote`] plus
/// the MusicXML voice number.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceNote {
    pub pitch: Option<String>,
    pub onset: f64,
    pub dur: f64,
    pub voice: u32,
}

/// One output part: a name plus its voice-assigned notes.
#[derive(Debug, Clone, PartialEq)]
pub struct PartNotes {
    pub name: String,
    pub notes: Vec<VoiceNote>,
}

/// Naive mapping: noteheads/rest glyphs in reading order, treble diatonic
/// pitch from the nearest staff, cumulative onsets, single voice.
pub fn map_naive(
    glyphs: &GlyphDoc,
    layout: &LayoutDoc,
    staff_spacing_fallback: f64,
) -> Vec<MappedNote> {
    // Flatten staves in document order: (system, staff) for reading order.
    let mut staves: Vec<(usize, &Staff)> = Vec::new();
    for (si, sys) in layout.systems.iter().enumerate() {
        for st in &sys.staves {
            staves.push((si, st));
        }
    }
    if staves.is_empty() {
        return Vec::new();
    }
    // One event per notehead/rest glyph, ordered by system then x.
    let mut events: Vec<(usize, f64, &GlyphIn)> = Vec::new();
    for crop in &glyphs.crops {
        for g in &crop.glyphs {
            let keep = matches!(
                g.label.as_str(),
                "notehead-black" | "notehead-void" | "rest"
            );
            if keep {
                let (si, _) = nearest_staff(&staves, g);
                events.push((si, g.bbox.x + g.bbox.w / 2.0, g));
            }
        }
    }
    events.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
    let mut out = Vec::new();
    let mut onset = 0.0;
    for (_, _, g) in events {
        let is_rest = g.label == "rest";
        let dur = match g.label.as_str() {
            "notehead-void" => 2.0,
            _ => 1.0,
        };
        let pitch = if is_rest {
            None
        } else {
            let (_, staff) = nearest_staff(&staves, g);
            Some(staff_step_pitch_base(
                staff,
                g,
                staff_spacing_fallback,
                false,
                &[],
                &[],
            ))
        };
        out.push(MappedNote { pitch, onset, dur });
        onset += dur;
    }
    out
}

/// Pianoform mapping: like [`map_naive`], but the bottom staff of each
/// multi-staff system reads as bass clef (top line = A3) and
/// x-overlapping noteheads share an onset (chords). Single voice;
/// durations stay nominal.
pub fn map_pianoform(
    glyphs: &GlyphDoc,
    layout: &LayoutDoc,
    staff_spacing_fallback: f64,
) -> Vec<MappedNote> {
    // (system, index-in-system, is-bass, staff), document order.
    let mut staves: Vec<(usize, usize, bool, &Staff)> = Vec::new();
    for (si, sys) in layout.systems.iter().enumerate() {
        let n = sys.staves.len();
        for (k, st) in sys.staves.iter().enumerate() {
            staves.push((si, k, n >= 2 && k + 1 == n, st));
        }
    }
    if staves.is_empty() {
        return Vec::new();
    }
    let mut events: Vec<(usize, f64, &GlyphIn)> = Vec::new();
    for crop in &glyphs.crops {
        for g in &crop.glyphs {
            if matches!(
                g.label.as_str(),
                "notehead-black" | "notehead-void" | "rest"
            ) {
                let (si, _, _, _) = nearest_staff_sys(&staves, g);
                events.push((si, g.bbox.x + g.bbox.w / 2.0, g));
            }
        }
    }
    events.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
    let mut out = Vec::new();
    let mut onset = 0.0;
    // Open chord group: (system, x0, max_dur).
    let mut group: Option<(usize, f64, f64)> = None;
    let close_group = |onset: &mut f64, group: &mut Option<(usize, f64, f64)>| {
        if let Some((_, _, d)) = group.take() {
            *onset += d;
        }
    };
    for (si, x, g) in events {
        let is_rest = g.label == "rest";
        let dur = match g.label.as_str() {
            "notehead-void" => 2.0,
            _ => 1.0,
        };
        if is_rest {
            close_group(&mut onset, &mut group);
            out.push(MappedNote {
                pitch: None,
                onset,
                dur,
            });
            onset += dur;
            continue;
        }
        let (_, _, is_bass, staff) = nearest_staff_sys(&staves, g);
        let spacing = staff_spacing(staff).unwrap_or(staff_spacing_fallback.max(1.0));
        let pitch = staff_step_pitch_base(staff, g, staff_spacing_fallback, is_bass, &[], &[]);
        match group {
            Some((gsi, x0, ref mut gd)) if gsi == si && x - x0 < spacing => {
                *gd = gd.max(dur);
                out.push(MappedNote {
                    pitch: Some(pitch),
                    onset,
                    dur,
                });
            }
            _ => {
                close_group(&mut onset, &mut group);
                group = Some((si, x, dur));
                out.push(MappedNote {
                    pitch: Some(pitch),
                    onset,
                    dur,
                });
            }
        }
    }
    out
}

/// Nearest staff with its system/role context.
fn nearest_staff_sys<'a>(
    staves: &[(usize, usize, bool, &'a Staff)],
    g: &GlyphIn,
) -> (usize, usize, bool, &'a Staff) {
    let cx = g.bbox.x + g.bbox.w / 2.0;
    let cy = g.bbox.y + g.bbox.h / 2.0;
    let mut best = staves[0];
    let mut best_d = f64::INFINITY;
    for s in staves {
        let d = staff_line_distance(s.3, cx, cy);
        if d < best_d {
            best_d = d;
            best = *s;
        }
    }
    best
}

/// Nearest staff by distance from the glyph center to the nearest tilted
/// staff line. Ties go to the earlier staff.
fn nearest_staff<'a>(staves: &[(usize, &'a Staff)], g: &GlyphIn) -> (usize, &'a Staff) {
    let cx = g.bbox.x + g.bbox.w / 2.0;
    let cy = g.bbox.y + g.bbox.h / 2.0;
    let mut best = staves[0];
    let mut best_d = f64::INFINITY;
    for s in staves {
        let d = staff_line_distance(s.1, cx, cy);
        if d < best_d {
            best_d = d;
            best = *s;
        }
    }
    best
}

/// Diatonic pitch for a glyph center: half-spaces below the top staff
/// line. Bass staves descend from A3, treble from F5: the semitone map
/// follows the actual top degree, not a fixed table. Ledger lines
/// continue the pattern.
/// Accidentals at the system start recur on every system while
/// note-attached accidentals do not, so a flat/sharp-plausible box at a
/// (clef, degree-step) seen at system starts across systems is a key
/// signature accidental. The aux classifier does not distinguish
/// accidental kinds: flats and sharps are detected independently, each
/// gated by its order prefix (flats B,E,A,...; sharps F,C,G,...). A lone
/// C accidental is clef junk, never a key, no matter how often it
/// recurs.
const KEY_START_FRAC: f64 = 0.12;
const KEY_MIN_WIDTH_SP: f64 = 0.25;
const KEY_MAX_WIDTH_SP: f64 = 1.8;
const KEY_MIN_HEIGHT_SP: f64 = 1.5;
const KEY_MAX_HEIGHT_SP: f64 = 3.5;
/// Order of flats: the detected set must be a prefix of this.
const FLAT_ORDER: [char; 7] = ['B', 'E', 'A', 'D', 'G', 'C', 'F'];
/// Order of sharps: same prefix rule mirrored.
const SHARP_ORDER: [char; 7] = ['F', 'C', 'G', 'D', 'A', 'E', 'B'];
/// Height fraction marking a flat glyph's bulb row. Flat signs are
/// stem-heavy: the box center sits ~1 half-space above the modified line
/// while the bulb wraps it, so the step comes from this row, not the
/// center. Sharps are symmetric and keep the center. Validated on 8
/// symbox Bb flats (bulb 0.65–0.8 reads B, center reads C).
const KEY_BULB_FRAC: f64 = 0.7;
/// Treble diatonic frame top (F5) stepping down.
const TREBLE_STEPS: [char; 7] = ['F', 'E', 'D', 'C', 'B', 'A', 'G'];
/// Bass diatonic frame top (A3) stepping down.
const BASS_STEPS: [char; 7] = ['A', 'G', 'F', 'E', 'D', 'C', 'B'];

/// Raw key-candidate hits: (system, bass, step) from accidental boxes at
/// system starts. `bulb` selects the bulb row (flats) or center (sharps).
fn key_hits(
    aux_boxes: &[&GlyphIn],
    layout: &LayoutDoc,
    roles: &[Vec<(StaffRole, bool)>],
    fallback_spacing: f64,
    bulb: bool,
) -> Vec<(usize, bool, i32)> {
    let mut hits: Vec<(usize, bool, i32)> = Vec::new();
    for (si, sys) in layout.systems.iter().enumerate() {
        let sys_roles = roles.get(si);
        for (sti, st) in sys.staves.iter().enumerate() {
            let bass = sys_roles.and_then(|r| r.get(sti)).is_some_and(|(_, b)| *b);
            let spacing = staff_spacing(st).unwrap_or(fallback_spacing.max(1.0));
            let sp = spacing.max(1.0);
            let x0 = st.bbox.first().copied().unwrap_or(0.0);
            let width = st.bbox.get(2).copied().unwrap_or(0.0) - x0;
            let y0 = st.bbox.get(1).copied().unwrap_or(0.0);
            let y1 = st.bbox.get(3).copied().unwrap_or(0.0);
            let top_line = match st.lines.first() {
                Some(&l) => l,
                None => continue,
            };
            for b in aux_boxes.iter() {
                if b.label != "accidental" {
                    continue;
                }
                let cx = b.bbox.x + b.bbox.w / 2.0;
                let cy = b.bbox.y + b.bbox.h / 2.0;
                if cy < y0 - sp || cy > y1 + sp {
                    continue;
                }
                if cx < x0 || cx > x0 + KEY_START_FRAC * width {
                    continue;
                }
                if b.bbox.w < KEY_MIN_WIDTH_SP * sp
                    || b.bbox.w > KEY_MAX_WIDTH_SP * sp
                    || b.bbox.h < KEY_MIN_HEIGHT_SP * sp
                    || b.bbox.h > KEY_MAX_HEIGHT_SP * sp
                {
                    continue;
                }
                // Flat signs are stem-heavy: step from the bulb row, not
                // the box center — but only for full-height flat boxes
                // (~2.5sp engraved; shorter boxes keep the center so
                // clef-curl junk on C cannot promote to B). Sharps are
                // symmetric: always keep the center.
                let ky = if bulb && b.bbox.h >= 2.0 * sp {
                    b.bbox.y + KEY_BULB_FRAC * b.bbox.h
                } else {
                    cy
                };
                let step = ((ky - line_at(st, top_line, cx)) / (sp / 2.0)).round() as i32;
                hits.push((si, bass, step.rem_euclid(7)));
            }
        }
    }
    hits
}

/// Degrees (clef, diatonic step mod 7) flattened by the key signature,
/// from accidental boxes recurring at system starts.
fn detect_key_flats(
    aux_boxes: &[&GlyphIn],
    layout: &LayoutDoc,
    roles: &[Vec<(StaffRole, bool)>],
    fallback_spacing: f64,
) -> Vec<(bool, i32)> {
    prefix_key_letters(
        key_hits(aux_boxes, layout, roles, fallback_spacing, true),
        FLAT_ORDER,
    )
}

/// Degrees sharpened by the key signature. Sharps are vertically
/// symmetric, so the box center marks the row (no bulb correction).
fn detect_key_sharps(
    aux_boxes: &[&GlyphIn],
    layout: &LayoutDoc,
    roles: &[Vec<(StaffRole, bool)>],
    fallback_spacing: f64,
) -> Vec<(bool, i32)> {
    prefix_key_letters(
        key_hits(aux_boxes, layout, roles, fallback_spacing, false),
        SHARP_ORDER,
    )
}

/// Letters recurring on 2+ systems, then the longest order prefix among
/// them: junk letters (clef curls on C) drop out.
fn prefix_key_letters(hits: Vec<(usize, bool, i32)>, order: [char; 7]) -> Vec<(bool, i32)> {
    let mut groups: std::collections::HashMap<(bool, i32), Vec<usize>> =
        std::collections::HashMap::new();
    for (si, bass, step) in hits {
        groups.entry((bass, step)).or_default().push(si);
    }
    let mut letters: Vec<char> = groups
        .into_iter()
        .filter(|(_, systems)| {
            let mut uniq = systems.clone();
            uniq.sort_unstable();
            uniq.dedup();
            uniq.len() >= 2
        })
        .map(|((bass, step), _)| {
            if bass {
                BASS_STEPS[step as usize]
            } else {
                TREBLE_STEPS[step as usize]
            }
        })
        .collect();
    letters.sort_unstable();
    letters.dedup();
    let mut out: Vec<(bool, i32)> = Vec::new();
    for letter in order {
        if !letters.contains(&letter) {
            break;
        }
        for (bass, steps) in [(false, TREBLE_STEPS), (true, BASS_STEPS)] {
            for (i, &deg) in steps.iter().enumerate() {
                if deg == letter {
                    out.push((bass, i as i32));
                }
            }
        }
    }
    out
}

fn staff_step_pitch_base(
    staff: &Staff,
    g: &GlyphIn,
    fallback_spacing: f64,
    is_bass: bool,
    key_flats: &[(bool, i32)],
    key_sharps: &[(bool, i32)],
) -> String {
    let cx = g.bbox.x + g.bbox.w / 2.0;
    let top = line_at(staff, staff.lines.first().copied().unwrap_or(0.0), cx);
    let spacing = staff_spacing(staff).unwrap_or(fallback_spacing.max(1.0));
    let cy = g.bbox.y + g.bbox.h / 2.0;
    let step = ((cy - top) / (spacing / 2.0)).round() as i32;
    let (top_step, top_midi) = if is_bass { ('A', 57) } else { ('F', 77) };
    let mut midi = top_midi + diatonic_chroma_from(top_step, step);
    if key_flats.contains(&(is_bass, step.rem_euclid(7))) {
        midi -= 1;
    }
    if key_sharps.contains(&(is_bass, step.rem_euclid(7))) {
        midi += 1;
    }
    // Clamp to the MIDI range: far-off-staff blobs (text, page furniture,
    // wrong-staff association) still emit a valid note-list symbol.
    midi_to_pitch(midi.clamp(0, 127))
}

/// Role of a staff inside its system for multi-part mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StaffRole {
    Voice,
    PianoUpper,
    PianoLower,
}

/// Role plus bass-clef flag for every staff, in layout order.
fn assign_roles(layout: &LayoutDoc) -> Vec<Vec<(StaffRole, bool)>> {
    layout
        .systems
        .iter()
        .map(|sys| {
            let n = sys.staves.len();
            (0..n)
                .map(|i| {
                    let role = match n {
                        1 => StaffRole::Voice,
                        2 => {
                            if i == 0 {
                                StaffRole::PianoUpper
                            } else {
                                StaffRole::PianoLower
                            }
                        }
                        _ => {
                            if i == 0 {
                                StaffRole::Voice
                            } else if i == n - 1 {
                                StaffRole::PianoLower
                            } else {
                                StaffRole::PianoUpper
                            }
                        }
                    };
                    (role, n > 1 && i == n - 1)
                })
                .collect()
        })
        .collect()
}

fn bbox_key(b: &BBox) -> (u64, u64, u64, u64) {
    (b.x.to_bits(), b.y.to_bits(), b.w.to_bits(), b.h.to_bits())
}

/// Resolve a primary label against the aux detector on the exact same box.
/// PatchNet labels some filled heads `rest`, but both backends share the
/// connected-components proposer, so an exact-bbox aux notehead overrules
/// a primary rest. Anything else keeps the primary label: eighth rests
/// and flats are geometrically confusable and the aux classifier owns no
/// rest class to break the tie.
/// Multi-part voiced mapping: voice staff and piano grand staff become
/// separate parts (Voice first when present), piano lower staff takes
/// voice 5 (OpenScore piano convention), durations come from the aux
/// detector's stem/beam/dot boxes joined on exact bbox, and each
/// (part, voice) advances its own onset cursor.
pub fn map_voiced(
    primary: &GlyphDoc,
    aux: &GlyphDoc,
    layout: &LayoutDoc,
    fallback_spacing: f64,
) -> Vec<PartNotes> {
    use std::collections::HashMap;
    // Aux labels keyed by exact bbox (same proposer => identical boxes).
    let mut aux_by_box: HashMap<(u64, u64, u64, u64), &str> = HashMap::new();
    let mut aux_boxes: Vec<&GlyphIn> = Vec::new();
    for crop in &aux.crops {
        for g in &crop.glyphs {
            aux_by_box.insert(bbox_key(&g.bbox), g.label.as_str());
            aux_boxes.push(g);
        }
    }
    // Learned beam evidence from the primary doc (4-class PatchNet). A
    // primary beam on a box the aux detector positively labeled otherwise
    // (head, rest, dot, stem, clef, accidental, or its own beam) is a
    // contested detection, not evidence: only aux-silent (`unknown`) or
    // wholly new boxes join the beam pool. Pooled boxes stay out of
    // `aux_boxes` (their levels carry source identity so quarters of one
    // regressed box cannot self-pair; aux boxes keep legacy free pairing
    // for merged symbox doubles).
    let mut pooled: Vec<&GlyphIn> = Vec::new();
    // Primary dots/accidentals were tried as duration/key evidence and
    // reverted: heat lookalikes fabricate keys (essen-ballad 0.278 ->
    // 0.075) and dot FPs cascade through the onset cursor (mean -0.008).
    // Duration-critical features need precision-gated evidence; revisit
    // with per-class gates if detector precision improves.
    for crop in &primary.crops {
        for g in &crop.glyphs {
            if g.label != "beam" {
                continue;
            }
            match aux_by_box.get(&bbox_key(&g.bbox)) {
                None | Some(&"unknown") => pooled.push(g),
                _ => {}
            }
        }
    }
    // Flat staff table: (staff, role, is_bass, part_index, voice).
    // Part 0 is always Voice, part 1 always Piano (reference order).
    let roles = assign_roles(layout);
    let mut table: Vec<(&Staff, StaffRole, bool, usize, u32)> = Vec::new();
    for (sys, sys_roles) in layout.systems.iter().zip(roles.iter()) {
        for (st, (role, is_bass)) in sys.staves.iter().zip(sys_roles.iter()) {
            let (pi, voice) = match role {
                StaffRole::Voice => (0, 1),
                StaffRole::PianoUpper => (1, 1),
                StaffRole::PianoLower => (1, 5),
            };
            table.push((st, *role, *is_bass, pi, voice));
        }
    }
    // Key signature flats, detected once per page from accidental boxes
    // recurring at system starts (aux accidentals elsewhere are noise).
    // Primary accidentals stay out of key detection for now: heat fires
    // too many accidental lookalikes at system starts (clef curls,
    // key-stack fragments) and junk recurrence fabricates flat keys that
    // poison every note (essen-ballad 0.278 -> 0.075). Revisit with a
    // precision gate (confidence/size/shape) if key recall is needed.
    let key_flats = detect_key_flats(&aux_boxes, layout, &roles, fallback_spacing);
    let key_sharps = detect_key_sharps(&aux_boxes, layout, &roles, fallback_spacing);
    // Clef spans for the system-start barline veto: heat fires Rest on
    // thick system-start barlines and key-signature stacks. Collected once
    // per page; every system's clef zone shares the same x-range.
    let clef_zones = clef_zone_spans(primary, fallback_spacing);
    let part_name = ["Voice", "Piano"];
    // Collect events per (part, voice) in system order, then advance each
    // group's own onset cursor in x order with chord sharing.
    let mut parts: Vec<PartNotes> = Vec::new();
    let mut index: HashMap<usize, usize> = HashMap::new();
    let mut cursors: HashMap<(usize, u32), f64> = HashMap::new();
    for (sys, sys_roles) in layout.systems.iter().zip(roles.iter()) {
        let mut groups: HashMap<(usize, u32), Vec<(f64, VoiceNote)>> = HashMap::new();
        let mut order: Vec<(usize, u32)> = Vec::new();
        for (st, (role, is_bass)) in sys.staves.iter().zip(sys_roles.iter()) {
            let (pi, voice) = match role {
                StaffRole::Voice => (0, 1),
                StaffRole::PianoUpper => (1, 1),
                StaffRole::PianoLower => (1, 5),
            };
            let spacing = staff_spacing(st).unwrap_or(fallback_spacing.max(1.0));
            let bass = *is_bass;
            let ti = table
                .iter()
                .position(|(s, _, _, _, _)| std::ptr::eq(*s, st));
            let mut staff_evs: Vec<(f64, VoiceNote, &GlyphIn, bool)> = Vec::new();
            for crop in &primary.crops {
                for g in &crop.glyphs {
                    if !matches!(
                        g.label.as_str(),
                        "notehead-black" | "notehead-void" | "rest"
                    ) {
                        continue;
                    }
                    if nearest_staff_index(&table, g) != ti {
                        continue;
                    }
                    let is_rest = g.label == "rest";
                    if is_rest {
                        let cx = g.bbox.x + g.bbox.w / 2.0;
                        let cy = g.bbox.y + g.bbox.h / 2.0;
                        if !rest_in_staff_span(st, cx, cy, spacing) {
                            continue;
                        }
                        if rest_box_is_staff_artifact(st, g, spacing) {
                            continue;
                        }
                        if rest_box_is_clef_zone_artifact(g, &clef_zones, spacing) {
                            continue;
                        }
                    }
                    let pitch = if is_rest {
                        None
                    } else {
                        Some(staff_step_pitch_base(
                            st,
                            g,
                            spacing,
                            bass,
                            &key_flats,
                            &key_sharps,
                        ))
                    };
                    let dur = if is_rest {
                        rest_dur(g, *role, spacing)
                    } else {
                        note_dur(g, &aux_by_box, &aux_boxes, &pooled, spacing)
                    };
                    staff_evs.push((
                        g.bbox.x,
                        VoiceNote {
                            pitch,
                            onset: 0.0,
                            dur,
                            voice,
                        },
                        g,
                        is_rest,
                    ));
                }
            }
            share_beam_runs(&mut staff_evs, &aux_boxes, &pooled, spacing);
            let key = (pi, voice);
            if !groups.contains_key(&key) {
                order.push(key);
            }
            groups
                .entry(key)
                .or_default()
                .extend(staff_evs.drain(..).map(|(x, n, _, _)| (x, n)));
        }
        for key in order {
            if let Some(mut evs) = groups.remove(&key) {
                evs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let cursor = cursors.entry(key).or_insert(0.0);
                let mut group_x0 = f64::NEG_INFINITY;
                let mut group_onset = 0.0;
                for (x, mut n) in evs {
                    if x - group_x0 < fallback_spacing.max(1.0) {
                        n.onset = group_onset;
                    } else {
                        n.onset = *cursor;
                        group_onset = *cursor;
                        group_x0 = x;
                    }
                    *cursor = group_onset.max(*cursor).max(group_onset + n.dur);
                    let part_idx = *index.entry(key.0).or_insert_with(|| {
                        parts.push(PartNotes {
                            name: part_name[key.0].to_string(),
                            notes: Vec::new(),
                        });
                        parts.len() - 1
                    });
                    parts[part_idx].notes.push(n);
                }
            }
        }
    }
    parts
}

/// True when a rest center sits inside the staff's tilted span (top to
/// bottom line at the glyph's x, plus [`REST_SPAN_MARGIN`] staff-spaces
/// each side for bar rests jutting past the outer lines). Rests never
/// take ledger lines, so a rest outside every staff's span is inter-staff
/// junk (lyrics, dynamics) and must not enter the onset cursor. Glyphs
/// keep flowing when the staff has no lines rather than black-holing.
const REST_SPAN_MARGIN: f64 = 0.75;

fn rest_in_staff_span(staff: &Staff, cx: f64, cy: f64, spacing: f64) -> bool {
    if staff.lines.is_empty() || spacing <= 0.0 {
        return true;
    }
    let top = line_at(staff, staff.lines[0], cx);
    let bottom = line_at(staff, staff.lines[staff.lines.len() - 1], cx);
    let margin = REST_SPAN_MARGIN * spacing;
    cy >= top - margin && cy <= bottom + margin
}

/// True when a rest box's geometry is incompatible with any musical rest:
/// either it covers the staff's outer lines (a barline/brace fragment),
/// or it sticks out past them (a stem/stack fragment). GT symbox rests
/// sit ~1sp inside the outer lines, so the stick-out margin is generous:
/// whole rests hang from the second line down, half rests sit on the
/// middle line, quarter rests center on it. Dropping these keeps phantom
/// whole-note durations out of the onset cursor.
fn rest_box_is_staff_artifact(st: &Staff, g: &GlyphIn, spacing: f64) -> bool {
    if st.lines.len() < 2 || spacing <= 0.0 {
        return false;
    }
    let sp = spacing.max(1.0);
    let cx = g.bbox.x + g.bbox.w / 2.0;
    let top = line_at(st, st.lines[0], cx);
    let bottom = line_at(st, st.lines[st.lines.len() - 1], cx);
    let covers = g.bbox.y <= top + 0.5 * sp && g.bbox.y + g.bbox.h >= bottom - 0.5 * sp;
    // 0.75sp keeps tight detector boxes (pinned rest fixtures sit
    // 0.6sp out) while dropping the observed heat FPs (0.9sp+ out).
    let sticks_out = g.bbox.y < top - 0.75 * sp || g.bbox.y + g.bbox.h > bottom + 0.75 * sp;
    covers || sticks_out
}

/// Key-signature allowance past a clef's right edge, in staff spaces: a
/// seven-accidental key spans ~7sp, and music never starts inside it.
const KEY_ALLOW_SP: f64 = 7.0;

/// Clef x-spans (plus key allowance) for the system-start barline veto,
/// collected once per page from primary clef boxes.
fn clef_zone_spans(primary: &GlyphDoc, fallback_spacing: f64) -> Vec<(f64, f64)> {
    let allow = KEY_ALLOW_SP * fallback_spacing.max(1.0);
    let mut zones: Vec<(f64, f64)> = Vec::new();
    for crop in &primary.crops {
        for g in &crop.glyphs {
            if g.label == "clef" {
                zones.push((g.bbox.x, g.bbox.x + g.bbox.w + allow));
            }
        }
    }
    zones
}

/// True when a rest box is a system-start barline/brace FP: heat fires
/// Rest on thick system-start barlines and key-signature stacks at high
/// confidence. No rest glyph is both wide and tall (whole/half rests are
/// wide-but-short; quarter/eighth rests are tall-but-narrow), and music
/// never sits inside the clef span, so a both-wide-and-tall rest
/// overlapping a clef zone is an artifact. Dropping these keeps phantom
/// whole-bar durations out of the onset cursor.
fn rest_box_is_clef_zone_artifact(g: &GlyphIn, clef_zones: &[(f64, f64)], spacing: f64) -> bool {
    let sp = spacing.max(1.0);
    if g.bbox.w <= 1.2 * sp || g.bbox.h <= 2.0 * sp {
        return false;
    }
    clef_zones
        .iter()
        .any(|(x0, x1)| g.bbox.x < *x1 && g.bbox.x + g.bbox.w > *x0)
}

/// Index of the nearest staff in `table` by tilted-line distance.
fn nearest_staff_index(
    table: &[(&Staff, StaffRole, bool, usize, u32)],
    g: &GlyphIn,
) -> Option<usize> {
    let cx = g.bbox.x + g.bbox.w / 2.0;
    let cy = g.bbox.y + g.bbox.h / 2.0;
    table
        .iter()
        .enumerate()
        .min_by(|(_, (a, _, _, _, _)), (_, (b, _, _, _, _))| {
            let da = staff_line_distance(a, cx, cy);
            let db = staff_line_distance(b, cx, cy);
            da.partial_cmp(&db).unwrap()
        })
        .map(|(i, _)| i)
}

/// Vertical reach of beam association in staff spaces. Stems run past an
/// octave, so symbox GT beams sit up to ~6sp from their heads (gap
/// histogram on ds0005-1: 68/46/35 heads in bins 2/3/4sp, then 8/10
/// at 5/6sp); the stock 4sp window drops the long-stem half of beamed
/// groups. 6sp stays clear of adjacent staves' beams (~8sp+ away).
const BEAM_ASSOC_SP: f64 = 6.0;

/// Source group of a beam level. Aux-detector levels share one group
/// (legacy free pairing: a merged symbox double-beam box pairs with
/// itself). Primary-pooled levels start in their own box's group; dedup
/// unions groups whose boxes overlap, so heat duplicate detections of one
/// bar (y-jitter up to ~1.5sp) never pair with each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LevelGroup {
    Aux,
    Pooled((u64, u64, u64, u64)),
}

/// A beam level with its source identity and box y-range. `tall` marks
/// levels from a pooled (heatmap) box at least 1.25sp high: regressed-thick
/// bars whose quarter levels must not pair into 16ths (heat duplicates of
/// one bar are tall; true split double bars are thin). Aux levels are
/// never tall (legacy free pairing for merged symbox doubles).
#[derive(Debug, Clone, Copy)]
struct Level {
    y: f64,
    group: LevelGroup,
    y0: f64,
    y1: f64,
    tall: bool,
}

impl Level {
    /// True when the two source boxes' y-ranges overlap: same bar seen
    /// twice (heat jitter), not two stacked bars. Strict overlap only;
    /// touching ranges stay distinct so blurred true doubles still pair.
    fn same_bar(self, other: Level) -> bool {
        self.y0 < other.y1 && other.y0 < self.y1
    }
}

/// Beam levels near a head's x-column, split by side. Beam boxes fragment
/// under degradation, so the x-window is dilated half a spacing; boxes at
/// the same height are one level. Levels on opposite sides of the head are
/// different staves' beams in the same y-window, not a double beam: only
/// a same-side pair about a spacing apart marks a 16th.
fn beam_levels(
    g: &GlyphIn,
    aux_boxes: &[&GlyphIn],
    pooled_boxes: &[&GlyphIn],
    spacing: f64,
) -> (Vec<Level>, Vec<Level>) {
    let sp = spacing.max(1.0);
    let cx = g.bbox.x + g.bbox.w / 2.0;
    let mut raw: Vec<Level> = aux_boxes
        .iter()
        .map(|b| (*b, LevelGroup::Aux))
        .chain(
            pooled_boxes
                .iter()
                .map(|b| (*b, LevelGroup::Pooled(bbox_key(&b.bbox)))),
        )
        .filter(|(b, _)| {
            b.label == "beam"
                && b.bbox.x - 0.5 * sp < cx + 0.5 * sp
                && b.bbox.x + b.bbox.w + 0.5 * sp > cx - 0.5 * sp
                && (b.bbox.y + b.bbox.h < g.bbox.y
                    && g.bbox.y - (b.bbox.y + b.bbox.h) < BEAM_ASSOC_SP * sp
                    || b.bbox.y > g.bbox.y + g.bbox.h
                        && b.bbox.y - (g.bbox.y + g.bbox.h) < BEAM_ASSOC_SP * sp)
        })
        .flat_map(|(b, group)| {
            // A merged multi-beam box (double 16th bars span ~1.5sp)
            // carries two levels at its quarters; a single bar (~0.5sp)
            // carries its center. Boxes under 1.25sp stay single: 1.0sp
            // boxes over-fire pairs across whole runs.
            let tall = matches!(group, LevelGroup::Pooled(_)) && b.bbox.h >= 1.25 * sp;
            let level = |y: f64| Level {
                y,
                group,
                y0: b.bbox.y,
                y1: b.bbox.y + b.bbox.h,
                tall,
            };
            if b.bbox.h >= 1.25 * sp {
                vec![
                    level(b.bbox.y + b.bbox.h / 4.0),
                    level(b.bbox.y + 3.0 * b.bbox.h / 4.0),
                ]
            } else {
                vec![level(b.bbox.y + b.bbox.h / 2.0)]
            }
        })
        .collect();
    raw.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());
    let mut above: Vec<Level> = Vec::new();
    let mut below: Vec<Level> = Vec::new();
    for level in raw {
        let side = if level.y < g.bbox.y {
            &mut above
        } else {
            &mut below
        };
        side.push(level);
    }
    dedup_levels(&mut above, sp);
    dedup_levels(&mut below, sp);
    (above, below)
}

/// Merge duplicate views of one beam bar into a single level: same-height
/// levels merge, and overlapping source ranges merge when a pooled level
/// is involved (heat duplicates). The survivor adopts the newcomer's group
/// and the union of both source ranges, so a jitter cloud chains into one
/// level instead of leaving its extremes as a false pair. Shared by
/// per-head levels and run pools: a run pool concatenates several heads'
/// views of the same bars, so it must re-dedup before pairing, or
/// cross-head views of one bar form false 16th pairs.
fn dedup_levels(side: &mut Vec<Level>, sp: f64) {
    side.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());
    let mut out: Vec<Level> = Vec::new();
    for level in side.drain(..) {
        match out.last_mut() {
            // Same height: one level. The survivor adopts the union
            // group so later pairs see the merged identity.
            Some(last) if level.y - last.y <= 0.5 * sp => {
                last.group = level.group;
                last.y0 = last.y0.min(level.y0);
                last.y1 = last.y1.max(level.y1);
            }
            // Same bar (overlapping source ranges) with a pooled level
            // involved: one heat duplicate bar, one level. Aux-aux pairs
            // keep legacy gap-only merging so merged symbox doubles
            // still split into quarters.
            Some(last)
                if !matches!(
                    (last.group, level.group),
                    (LevelGroup::Aux, LevelGroup::Aux)
                ) && level.same_bar(*last) =>
            {
                last.group = level.group;
                last.y0 = last.y0.min(level.y0);
                last.y1 = last.y1.max(level.y1);
            }
            _ => out.push(level),
        }
    }
    *side = out;
}

/// True when one side carries two levels about a spacing apart: a real
/// second beam for 16ths. Pairs closer than half a spacing are one
/// fragmented box; pairs further than two spacings are different groups.
/// Two levels of the same source group never pair (single-box quarters,
/// heat duplicates); levels from tall pooled boxes never pair
/// (regressed-thick duplicate bars); aux levels keep legacy free pairing.
fn second_beam(levels: &[Level], spacing: f64) -> bool {
    let sp = spacing.max(1.0);
    levels.windows(2).any(|w| {
        let gap = w[1].y - w[0].y;
        gap > 0.5 * sp
            && gap <= 2.0 * sp
            && !w[0].tall
            && !w[1].tall
            && !matches!(
                (&w[0].group, &w[1].group),
                (LevelGroup::Pooled(a), LevelGroup::Pooled(b)) if a == b
            )
    })
}

/// Heads closer than this many staff-spaces (bbox gap) belong to one
/// beamed run. Engraved beamed groups sit ~1sp apart; separate beats are
/// wider. Chords share x (negative gap) and stay in the run.
const RUN_GAP: f64 = 2.0;

/// Share beam evidence across an x-run of heads on one staff. Beamed
/// groups usually show only beam fragments, so members outside every
/// fragment's x-window would default to quarters. A run covered by pooled
/// beam levels takes the run duration (halves and wider keep their base:
/// only quarters and shorter are shortened); uncovered runs keep per-head
/// durations. Rests break runs.
fn share_beam_runs(
    evs: &mut [(f64, VoiceNote, &GlyphIn, bool)],
    aux_boxes: &[&GlyphIn],
    pooled_boxes: &[&GlyphIn],
    spacing: f64,
) {
    let sp = spacing.max(1.0);
    let mut idx: Vec<usize> = (0..evs.len()).collect();
    idx.sort_by(|&a, &b| evs[a].0.partial_cmp(&evs[b].0).unwrap());
    let mut run: Vec<usize> = Vec::new();
    let flush = |run: &mut Vec<usize>, evs: &mut [(f64, VoiceNote, &GlyphIn, bool)]| {
        if run.len() < 2 {
            run.clear();
            return;
        }
        let mut above: Vec<Level> = Vec::new();
        let mut below: Vec<Level> = Vec::new();
        for &i in run.iter() {
            let (_, _, g, _) = &evs[i];
            let (a, b) = beam_levels(g, aux_boxes, pooled_boxes, spacing);
            above.extend(a);
            below.extend(b);
        }
        // The pool concatenates several heads' views of the same bars: re-dedup
        // (with range union) before pairing, or cloud extremes survive as
        // false 16th pairs.
        dedup_levels(&mut above, sp);
        dedup_levels(&mut below, sp);
        let pair = second_beam(&above, sp) || second_beam(&below, sp);
        let cover = if pair {
            Some(0.25)
        } else if !above.is_empty() || !below.is_empty() {
            Some(0.5)
        } else {
            None
        };
        if let Some(base) = cover {
            for &i in run.iter() {
                let (_, n, g, _) = &mut evs[i];
                if n.dur <= 1.5 {
                    let dotted = has_dot(g, aux_boxes, spacing);
                    n.dur = if dotted { base * 1.5 } else { base };
                }
            }
        }
        run.clear();
    };
    let mut prev_right = f64::NEG_INFINITY;
    for i in idx {
        let is_rest = evs[i].3;
        let gap_ok = evs[i].0 - prev_right < RUN_GAP * sp;
        if is_rest || !gap_ok {
            flush(&mut run, evs);
            if !is_rest {
                run.push(i);
            }
        } else {
            run.push(i);
        }
        if !is_rest {
            let g = evs[i].2;
            prev_right = evs[i].0.max(g.bbox.x + g.bbox.w);
        } else {
            prev_right = f64::NEG_INFINITY;
        }
    }
    flush(&mut run, evs);
}

/// Flagged stem tip: a high-confidence learned `flag` box in the head's
/// x-column, above or below within four staff spaces. Argmax flag
/// precision is only ~0.5 (small-blob confusion: dots, articulations),
/// so low-confidence flags are ignored (VAL: p >= 0.9 keeps flag recall
/// ~0.9 at precision ~0.7+).
const FLAG_CONFIDENCE: f64 = 0.9;

fn has_flag(g: &GlyphIn, aux_boxes: &[&GlyphIn], spacing: f64) -> bool {
    let sp = spacing.max(1.0);
    let cx = g.bbox.x + g.bbox.w / 2.0;
    aux_boxes.iter().any(|b| {
        b.label == "flag"
            && b.confidence >= FLAG_CONFIDENCE
            && b.bbox.x - 0.5 * sp < cx + 0.5 * sp
            && b.bbox.x + b.bbox.w + 0.5 * sp > cx - 0.5 * sp
            && (b.bbox.y + b.bbox.h < g.bbox.y && g.bbox.y - (b.bbox.y + b.bbox.h) < 4.0 * sp
                || b.bbox.y > g.bbox.y + g.bbox.h && b.bbox.y - (g.bbox.y + g.bbox.h) < 4.0 * sp)
    })
}

/// Augmentation dot: right of the head, vertically aligned.
fn has_dot(g: &GlyphIn, aux_boxes: &[&GlyphIn], spacing: f64) -> bool {
    let sp = spacing.max(1.0);
    let cy = g.bbox.y + g.bbox.h / 2.0;
    aux_boxes.iter().any(|d| {
        d.label == "dot"
            && d.bbox.x > g.bbox.x + g.bbox.w
            && d.bbox.x < g.bbox.x + g.bbox.w + 2.5 * sp
            && (d.bbox.y + d.bbox.h / 2.0 - cy).abs() < sp
    })
}

/// Duration of a notehead from aux stem/beam/dot evidence.
fn note_dur(
    g: &GlyphIn,
    aux_by_box: &std::collections::HashMap<(u64, u64, u64, u64), &str>,
    aux_boxes: &[&GlyphIn],
    pooled_boxes: &[&GlyphIn],
    spacing: f64,
) -> f64 {
    let is_void = g.label == "notehead-void";
    let sp = spacing.max(1.0);
    // Same-blob stem (head + stem merged by connected components): a true
    // half note. Separate nearby stems are ignored: stem recall is low and
    // most void heads with a stem nearby are faded filled heads.
    if is_void && aux_by_box.get(&bbox_key(&g.bbox)).copied() == Some("stem") {
        return if has_dot(g, aux_boxes, spacing) {
            3.0
        } else {
            2.0
        };
    }
    let (above, below) = beam_levels(g, aux_boxes, pooled_boxes, spacing);
    let pair = second_beam(&above, sp) || second_beam(&below, sp);
    // No whole notes: the corpus has none, and degraded boxes inflate
    // past the whole-width cue, so isolated voids are faded quarters.
    let mut dur = if pair {
        0.25
    } else if !above.is_empty() || !below.is_empty() {
        0.5
    } else if has_flag(g, aux_boxes, spacing) {
        // Flagged (unbeamed) eighth: the common vocal-note shape.
        0.5
    } else {
        1.0
    };
    if has_dot(g, aux_boxes, spacing) {
        dur *= 1.5;
    }
    dur
}

/// Duration of a rest: piano staves carry eighth rests here; a wide
/// voice-staff rest fills the bar.
fn rest_dur(g: &GlyphIn, role: StaffRole, spacing: f64) -> f64 {
    if role == StaffRole::Voice && g.bbox.w >= 1.2 * spacing.max(1.0) {
        3.0
    } else {
        0.5
    }
}

/// Median line gap of a staff, if it carries at least two lines.
fn staff_spacing(staff: &Staff) -> Option<f64> {
    if staff.lines.len() < 2 {
        return None;
    }
    let mut gaps: Vec<f64> = staff.lines.windows(2).map(|w| w[1] - w[0]).collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(gaps[gaps.len() / 2])
}

/// Chromatic offset of `step` diatonic half-spaces below the staff's top
/// line (negative steps rise above it). The semitone drops depend on the
/// top degree: descending from F loses a semitone on F->E and C->B, from A
/// on C->B and F->E, at different positions in the cycle.
fn diatonic_chroma_from(top_step: char, step: i32) -> i32 {
    const ORDER: [char; 7] = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
    /// Semitones lost stepping down FROM each degree.
    const DROP: [i32; 7] = [1, 2, 2, 1, 2, 2, 2];
    let pos = ORDER.iter().position(|&c| c == top_step).unwrap_or(3); // default F (treble)
    if step >= 0 {
        let mut acc = 0;
        for i in 0..step {
            let deg = (pos as i32 - i).rem_euclid(7) as usize;
            acc -= DROP[deg];
        }
        acc
    } else {
        // Ascending: gaining what the descent into each degree loses.
        let mut acc = 0;
        for i in 0..-step {
            let next = (pos as i32 + i + 1).rem_euclid(7) as usize;
            acc += DROP[next];
        }
        acc
    }
}

/// MIDI note to sharp-spelled pitch (`parse_pitch`-compatible).
fn midi_to_pitch(midi: i32) -> String {
    const NAMES: [&str; 12] = [
        "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
    ];
    let pc = midi.rem_euclid(12) as usize;
    let octave = midi.div_euclid(12) - 1;
    format!("{}{}", NAMES[pc], octave)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_one_staff() -> LayoutDoc {
        serde_json::from_value(serde_json::json!({
            "systems": [{"staves": [
                {"bbox": [0.0, 80.0, 500.0, 160.0], "lines": [100.0, 110.0, 120.0, 130.0, 140.0]}
            ]}]
        }))
        .unwrap()
    }

    fn glyphs_doc(glyphs: serde_json::Value) -> GlyphDoc {
        serde_json::from_value(serde_json::json!({"crops": [{"glyphs": glyphs}]})).unwrap()
    }

    #[test]
    fn naive_maps_two_noteheads_to_treble_pitches() {
        let glyphs = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 90.0, "y": 99.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let notes = map_naive(&glyphs, &layout_one_staff(), 10.0);
        // Top line treble = F5; one half-space below = E5.
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].pitch.as_deref(), Some("F5"));
        assert_eq!((notes[0].onset, notes[0].dur), (0.0, 1.0));
        assert_eq!(notes[1].pitch.as_deref(), Some("E5"));
        assert_eq!((notes[1].onset, notes[1].dur), (1.0, 1.0));
    }

    #[test]
    fn naive_rests_become_rests_and_other_labels_skip() {
        let glyphs = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 10.0, "y": 100.0, "w": 6.0, "h": 20.0}, "label": "stem"},
            {"bbox": {"x": 30.0, "y": 110.0, "w": 10.0, "h": 10.0}, "label": "rest"},
            {"bbox": {"x": 60.0, "y": 100.0, "w": 12.0, "h": 10.0}, "label": "notehead-void"}
        ]));
        let notes = map_naive(&glyphs, &layout_one_staff(), 10.0);
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].pitch, None);
        assert_eq!((notes[0].onset, notes[0].dur), (0.0, 1.0));
        // Void noteheads are half notes.
        assert_eq!((notes[1].onset, notes[1].dur), (1.0, 2.0));
    }

    #[test]
    fn naive_clamps_far_off_staff_pitches_to_midi_range() {
        let glyphs = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 2000.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let notes = map_naive(&glyphs, &layout_one_staff(), 10.0);
        assert_eq!(notes.len(), 1);
        let pitch = notes[0].pitch.clone().unwrap();
        let octave: i32 = pitch
            .trim_matches(|c: char| c.is_ascii_alphabetic() || c == '#')
            .parse()
            .unwrap();
        assert!((-1..=9).contains(&octave), "pitch {pitch} out of range");
    }

    fn layout_two_staves() -> LayoutDoc {
        serde_json::from_value(serde_json::json!({
            "systems": [{"staves": [
                {"bbox": [0.0, 0.0, 500.0, 60.0], "lines": [0.0, 10.0, 20.0, 30.0, 40.0]},
                {"bbox": [0.0, 100.0, 500.0, 160.0], "lines": [100.0, 110.0, 120.0, 130.0, 140.0]}
            ]}]
        }))
        .unwrap()
    }

    fn layout_two_systems() -> LayoutDoc {
        serde_json::from_value(serde_json::json!({
            "systems": [
                {"staves": [
                    {"bbox": [0.0, 0.0, 500.0, 60.0], "lines": [0.0, 10.0, 20.0, 30.0, 40.0]}
                ]},
                {"staves": [
                    {"bbox": [0.0, 200.0, 500.0, 260.0], "lines": [200.0, 210.0, 220.0, 230.0, 240.0]}
                ]}
            ]
        }))
        .unwrap()
    }

    // Flat-plausible box on the middle line (step 4 = B in treble)
    // inside the system-start zone of the system whose top line is `top`.
    fn flat_at_start(top: f64) -> serde_json::Value {
        serde_json::json!({"bbox": {"x": 50.0, "y": top + 12.0, "w": 8.0, "h": 16.0}, "label": "accidental"})
    }

    #[test]
    fn voiced_recurring_start_accidental_flattens_degree() {
        // The same flat at the system start in both systems is a key
        // signature: B heads become Bb.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 200.0, "y": 14.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let aux = aux_doc(serde_json::json!([
            flat_at_start(0.0),
            flat_at_start(200.0)
        ]));
        let parts = map_voiced(&primary, &aux, &layout_two_systems(), 10.0);
        assert_eq!(parts[0].notes[0].pitch.as_deref(), Some("Bb4"));
    }

    #[test]
    fn voiced_offset_flat_bulb_still_flattens_degree() {
        // Real flat glyphs are stem-heavy (~2.5sp tall): the box center
        // sits ~1 half-step above the modified line while the bulb wraps
        // it. A full-height flat whose center reads C but whose bulb row
        // reads B is a key B.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 200.0, "y": 14.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        // Bulb row top+19.5 ~= B line 20; center top+14.5 reads C.
        let mk = |top: f64| {
            serde_json::json!(
            {"bbox": {"x": 50.0, "y": top + 2.0, "w": 8.0, "h": 25.0}, "label": "accidental"})
        };
        let aux = aux_doc(serde_json::json!([mk(0.0), mk(200.0)]));
        let parts = map_voiced(&primary, &aux, &layout_two_systems(), 10.0);
        assert_eq!(parts[0].notes[0].pitch.as_deref(), Some("Bb4"));
    }

    #[test]
    fn voiced_recurring_sharp_raises_degree() {
        // Sharp keys mirror flats: F# at the system start in both systems
        // sharpens F heads (midi 66 spells Gb4). Sharps are vertically
        // symmetric, so the box center marks the modified row directly.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 200.0, "y": 29.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        // F line is the top line (step 0): centered sharp box on row 0.
        let mk = |top: f64| {
            serde_json::json!(
            {"bbox": {"x": 50.0, "y": top - 8.0, "w": 8.0, "h": 16.0}, "label": "accidental"})
        };
        let aux = aux_doc(serde_json::json!([mk(0.0), mk(200.0)]));
        let parts = map_voiced(&primary, &aux, &layout_two_systems(), 10.0);
        assert_eq!(parts[0].notes[0].pitch.as_deref(), Some("Gb4"));
    }

    #[test]
    fn voiced_recurring_nonflat_degree_is_not_a_key() {
        // Accidentals recurring on C (step 3) in both systems are clef
        // junk: a key signature can never start on C, so B stays natural.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 200.0, "y": 14.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let c0 = serde_json::json!({"bbox": {"x": 50.0, "y": 7.0, "w": 8.0, "h": 16.0}, "label": "accidental"});
        let c1 = serde_json::json!({"bbox": {"x": 50.0, "y": 207.0, "w": 8.0, "h": 16.0}, "label": "accidental"});
        let aux = aux_doc(serde_json::json!([c0, c1]));
        let parts = map_voiced(&primary, &aux, &layout_two_systems(), 10.0);
        assert_eq!(parts[0].notes[0].pitch.as_deref(), Some("B4"));
    }

    #[test]
    fn voiced_lone_start_accidental_is_not_a_key() {
        // One accidental on a single-system page is a cautionary or noise,
        // not a key signature: B stays natural.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 200.0, "y": 14.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let aux = aux_doc(serde_json::json!([flat_at_start(0.0)]));
        let parts = map_voiced(&primary, &aux, &layout_two_systems(), 10.0);
        assert_eq!(parts[0].notes[0].pitch.as_deref(), Some("B4"));
    }

    #[test]
    fn voiced_mid_staff_accidental_is_not_a_key() {
        // Accidentals far from the system start are note-attached (which the
        // noisy aux classifier cannot be trusted for), never key.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 200.0, "y": 14.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let mid0 = serde_json::json!({"bbox": {"x": 400.0, "y": 12.0, "w": 8.0, "h": 16.0}, "label": "accidental"});
        let mid1 = serde_json::json!({"bbox": {"x": 400.0, "y": 212.0, "w": 8.0, "h": 16.0}, "label": "accidental"});
        let aux = aux_doc(serde_json::json!([mid0, mid1]));
        let parts = map_voiced(&primary, &aux, &layout_two_systems(), 10.0);
        assert_eq!(parts[0].notes[0].pitch.as_deref(), Some("B4"));
    }

    #[test]
    fn midi_to_pitch_spells_flats() {
        assert_eq!(midi_to_pitch(58), "Bb3");
        assert_eq!(midi_to_pitch(61), "Db4");
        assert_eq!(midi_to_pitch(60), "C4");
    }

    fn layout_voice_piano() -> LayoutDoc {
        serde_json::from_value(serde_json::json!({
            "systems": [{"staves": [
                {"bbox": [0.0, 0.0, 500.0, 60.0], "lines": [0.0, 10.0, 20.0, 30.0, 40.0]},
                {"bbox": [0.0, 100.0, 500.0, 160.0], "lines": [100.0, 110.0, 120.0, 130.0, 140.0]},
                {"bbox": [0.0, 200.0, 500.0, 260.0], "lines": [200.0, 210.0, 220.0, 230.0, 240.0]}
            ]}]
        }))
        .unwrap()
    }

    #[test]
    fn voiced_drops_gap_rests_but_keeps_gap_noteheads() {
        // Rest on the voice staff (cy=20) is kept; an eighth-shaped rest in
        // the voice/piano gap (cy=70, 3sp below the voice staff) is lyric
        // or dynamics junk and must not shift the voice onset cursor. A
        // notehead at the same gap height is kept: low voice notes and
        // ledger lines live outside staff spans.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 14.0, "w": 10.0, "h": 12.0}, "label": "rest"},
            {"bbox": {"x": 90.0, "y": 64.0, "w": 10.0, "h": 12.0}, "label": "rest"},
            {"bbox": {"x": 130.0, "y": 64.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_voice_piano(),
            10.0,
        );
        let voice = parts.iter().find(|p| p.name == "Voice").unwrap();
        assert_eq!(voice.notes.len(), 2);
        assert!(voice.notes[0].pitch.is_none());
        assert!(voice.notes[1].pitch.is_some());
    }

    #[test]
    fn voiced_primary_beam_evidence_shortens_head() {
        // A primary-doc beam box (learned 4-class PatchNet) counts as beam
        // evidence even with no aux beam: the head takes 0.5, not 1.0.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 14.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 45.0, "y": 2.0, "w": 20.0, "h": 8.0}, "label": "beam"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_voice_piano(),
            10.0,
        );
        let voice = parts.iter().find(|p| p.name == "Voice").unwrap();
        assert_eq!(voice.notes.len(), 1);
        assert_eq!(voice.notes[0].dur, 0.5);
    }

    #[test]
    fn voiced_aux_flag_evidence_shortens_head() {
        // A high-confidence aux flag box (learned 5-class PatchNet) near
        // the head marks a flagged eighth: the head takes 0.5, not 1.0.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 14.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let aux = aux_doc(serde_json::json!([
            {"bbox": {"x": 45.0, "y": 2.0, "w": 20.0, "h": 8.0}, "label": "flag", "confidence": 0.95}
        ]));
        let parts = map_voiced(&primary, &aux, &layout_voice_piano(), 10.0);
        let voice = parts.iter().find(|p| p.name == "Voice").unwrap();
        assert_eq!(voice.notes.len(), 1);
        assert_eq!(voice.notes[0].dur, 0.5);
    }

    #[test]
    fn voiced_low_confidence_flag_is_ignored() {
        // Flag precision at argmax is ~0.5 (small-blob confusion), so only
        // high-confidence flags count: a 0.4-confidence flag leaves 1.0.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 14.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let aux = aux_doc(serde_json::json!([
            {"bbox": {"x": 45.0, "y": 2.0, "w": 20.0, "h": 8.0}, "label": "flag", "confidence": 0.4}
        ]));
        let parts = map_voiced(&primary, &aux, &layout_voice_piano(), 10.0);
        let voice = parts.iter().find(|p| p.name == "Voice").unwrap();
        assert_eq!(voice.notes.len(), 1);
        assert_eq!(voice.notes[0].dur, 1.0);
    }

    #[test]
    fn voiced_primary_beam_defers_to_aux_positive_vote() {
        // The same box labeled notehead by aux is a contested detection,
        // not beam evidence: no shortening.
        let bb = serde_json::json!({"x": 45.0, "y": 2.0, "w": 20.0, "h": 8.0});
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 14.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": bb.clone(), "label": "beam"}
        ]));
        let aux = aux_doc(serde_json::json!([{"bbox": bb, "label": "notehead-black"}]));
        let parts = map_voiced(&primary, &aux, &layout_voice_piano(), 10.0);
        let voice = parts.iter().find(|p| p.name == "Voice").unwrap();
        assert_eq!(voice.notes.len(), 1);
        assert_eq!(voice.notes[0].dur, 1.0);
    }

    #[test]
    fn pianoform_bottom_staff_uses_bass_clef() {
        let glyphs = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let notes = map_pianoform(&glyphs, &layout_two_staves(), 10.0);
        // Top line of the bottom (bass) staff = A3, not treble F5.
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].pitch.as_deref(), Some("A3"));
        // Same height on the top staff stays treble.
        let glyphs_top = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": -6.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let top = map_pianoform(&glyphs_top, &layout_two_staves(), 10.0);
        assert_eq!(top[0].pitch.as_deref(), Some("F5"));
    }

    #[test]
    fn pianoform_groups_x_overlapping_noteheads_as_chord() {
        let glyphs = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 56.0, "y": 99.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 90.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let notes = map_pianoform(&glyphs, &layout_one_staff(), 10.0);
        assert_eq!(notes.len(), 3);
        assert_eq!((notes[0].onset, notes[1].onset), (0.0, 0.0));
        assert_eq!(notes[2].onset, 1.0);
        assert_eq!(notes[0].pitch.as_deref(), Some("F5"));
        assert_eq!(notes[1].pitch.as_deref(), Some("E5"));
    }

    #[test]
    fn naive_skips_glyphs_when_no_staves() {
        let glyphs = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let empty: LayoutDoc = serde_json::from_value(serde_json::json!({"systems": []})).unwrap();
        assert!(map_naive(&glyphs, &empty, 10.0).is_empty());
    }

    fn layout_three_staves() -> LayoutDoc {
        serde_json::from_value(serde_json::json!({
            "systems": [{"staves": [
                {"bbox": [0.0, 0.0, 500.0, 60.0], "lines": [0.0, 10.0, 20.0, 30.0, 40.0]},
                {"bbox": [0.0, 100.0, 500.0, 160.0], "lines": [100.0, 110.0, 120.0, 130.0, 140.0]},
                {"bbox": [0.0, 200.0, 500.0, 260.0], "lines": [200.0, 210.0, 220.0, 230.0, 240.0]}
            ]}]
        }))
        .unwrap()
    }

    fn aux_doc(glyphs: serde_json::Value) -> GlyphDoc {
        serde_json::from_value(serde_json::json!({"crops": [{"glyphs": glyphs}]})).unwrap()
    }

    #[test]
    fn voiced_splits_three_staff_system_into_voice_and_piano() {
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": -6.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 50.0, "y": 194.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_three_staves(),
            10.0,
        );
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].name, "Voice");
        assert_eq!(parts[1].name, "Piano");
        // Voice staff is treble: top line F5. Piano RH treble, LH bass A3.
        assert_eq!(parts[0].notes[0].pitch.as_deref(), Some("F5"));
        assert_eq!(parts[0].notes[0].voice, 1);
        assert_eq!(parts[1].notes[0].pitch.as_deref(), Some("F5"));
        assert_eq!(parts[1].notes[0].voice, 1);
        assert_eq!(parts[1].notes[1].pitch.as_deref(), Some("A3"));
        assert_eq!(parts[1].notes[1].voice, 5);
    }

    #[test]
    fn voiced_tall_beam_box_splits_into_sixteenth() {
        // Symbox merges a 16th double-beam into one ~1.5sp box. A tall
        // beam box carries two levels, so the head is a 16th.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let aux = aux_doc(serde_json::json!([
            {"bbox": {"x": 46.0, "y": 130.0, "w": 30.0, "h": 15.0}, "label": "beam"}
        ]));
        let parts = map_voiced(&primary, &aux, &layout_one_staff(), 10.0);
        assert_eq!(parts[0].notes[0].dur, 0.25);
    }

    #[test]
    fn voiced_primary_single_box_quarters_do_not_self_pair() {
        // Heat regresses a single beam bar thick (~1.8sp): its quarter
        // levels must not pair with each other, or every beamed eighth
        // becomes a 16th. The head stays an eighth.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 46.0, "y": 130.0, "w": 30.0, "h": 18.0}, "label": "beam"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_one_staff(),
            10.0,
        );
        assert_eq!(parts[0].notes[0].dur, 0.5);
    }

    #[test]
    fn voiced_primary_two_distinct_boxes_still_pair() {
        // Two separate primary beam boxes ~1sp apart are a real double
        // beam: the head is a 16th. Only same-box quarters are barred.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 46.0, "y": 130.0, "w": 30.0, "h": 8.0}, "label": "beam"},
            {"bbox": {"x": 46.0, "y": 140.0, "w": 30.0, "h": 8.0}, "label": "beam"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_one_staff(),
            10.0,
        );
        assert_eq!(parts[0].notes[0].dur, 0.25);
    }

    #[test]
    fn voiced_primary_two_tall_boxes_do_not_pair() {
        // Two tall (regressed-thick, ~1.8sp) primary beam boxes with disjoint
        // ranges ~1sp apart are heat duplicates of one bar, not a double
        // beam: their quarter levels must not pair, or beamed eighths become
        // 16ths. The head stays an eighth. (Thin split boxes still pair.)
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 46.0, "y": 130.0, "w": 30.0, "h": 18.0}, "label": "beam"},
            {"bbox": {"x": 46.0, "y": 150.0, "w": 30.0, "h": 18.0}, "label": "beam"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_one_staff(),
            10.0,
        );
        assert_eq!(parts[0].notes[0].dur, 0.5);
    }

    #[test]
    fn voiced_run_pool_rededups_before_pairing() {
        // Three overlapping thin primary boxes staggered across two heads'
        // x-windows: each head merges its own two views into one level, but
        // the run pool concatenates head 1's survivor (y=134) with head 2's
        // (y=140). Without re-dedup the pool pairs them into false 16ths;
        // with re-dedup the overlapping ranges merge into one level and both
        // heads stay eighths.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 65.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 40.0, "y": 128.0, "w": 20.0, "h": 12.0}, "label": "beam"},
            {"bbox": {"x": 40.0, "y": 134.0, "w": 40.0, "h": 12.0}, "label": "beam"},
            {"bbox": {"x": 66.0, "y": 140.0, "w": 20.0, "h": 12.0}, "label": "beam"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_one_staff(),
            10.0,
        );
        assert_eq!(parts[0].notes.len(), 2);
        assert_eq!(parts[0].notes[0].dur, 0.5);
        assert_eq!(parts[0].notes[1].dur, 0.5);
    }

    #[test]
    fn voiced_dedup_unions_ranges_across_chain() {
        // A jitter cloud of three thin boxes where A overlaps B and B
        // overlaps C, but A and C are disjoint: without range union the
        // survivors A and C pair into a false 16th; with union the chain
        // merges into one level and the head stays an eighth.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 46.0, "y": 120.0, "w": 30.0, "h": 12.0}, "label": "beam"},
            {"bbox": {"x": 46.0, "y": 130.0, "w": 30.0, "h": 12.0}, "label": "beam"},
            {"bbox": {"x": 46.0, "y": 138.0, "w": 30.0, "h": 12.0}, "label": "beam"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_one_staff(),
            10.0,
        );
        assert_eq!(parts[0].notes[0].dur, 0.5);
    }

    #[test]
    fn voiced_drops_full_span_rest_as_barline_artifact() {
        // A rest box spanning the staff top-to-bottom is a barline/brace
        // fragment (heat Rest FPs fire on system edges at ~0.93 conf), not
        // a rest: dropping it keeps a +3.0 onset shift out of the cursor.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 98.0, "w": 26.0, "h": 44.0}, "label": "rest"},
            {"bbox": {"x": 90.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_one_staff(),
            10.0,
        );
        assert_eq!(parts[0].notes.len(), 1);
        assert_eq!(parts[0].notes[0].onset, 0.0);
    }

    #[test]
    fn voiced_drops_clef_zone_wide_tall_rest() {
        // A both-wide-and-tall rest overlapping the clef span is a thick
        // system-start barline / key-stack artifact, not a rest (no rest
        // glyph is wide and tall at once). Dropping it keeps a +3.0 onset
        // shift out of the cursor; the head starts at 0.0.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 100.0, "w": 30.0, "h": 40.0}, "label": "clef"},
            {"bbox": {"x": 55.0, "y": 98.0, "w": 20.0, "h": 30.0}, "label": "rest"},
            {"bbox": {"x": 200.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_one_staff(),
            10.0,
        );
        assert_eq!(parts[0].notes.len(), 1);
        assert_eq!(parts[0].notes[0].onset, 0.0);
    }

    #[test]
    fn voiced_primary_accidentals_do_not_form_key() {
        // Deliberately aux-only: heat fires accidental lookalikes at system
        // starts (clef curls, key-stack fragments) and junk recurrence
        // fabricates flat keys that poison every note (essen-ballad
        // 0.278 -> 0.075). Primary accidentals must not move pitch until a
        // precision gate lands; the head stays B natural.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 200.0, "y": 14.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 50.0, "y": 12.0, "w": 8.0, "h": 16.0}, "label": "accidental"},
            {"bbox": {"x": 50.0, "y": 212.0, "w": 8.0, "h": 16.0}, "label": "accidental"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_two_systems(),
            10.0,
        );
        assert_eq!(parts[0].notes[0].pitch.as_deref(), Some("B4"));
    }

    #[test]
    fn voiced_primary_dot_does_not_mark_duration() {
        // Deliberately aux-only: heat dot lookalikes near heads cascade
        // wrong 1.5x durations through the onset cursor (mean -0.008).
        // Primary dots must not move duration until a precision gate lands.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 63.0, "y": 97.0, "w": 4.0, "h": 4.0}, "label": "dot"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_one_staff(),
            10.0,
        );
        assert_eq!(parts[0].notes[0].dur, 1.0);
    }

    #[test]
    fn voiced_primary_overlapping_boxes_share_one_source() {
        // Two heat duplicate boxes of one bar: ranges overlap (130..148
        // vs 138..152) while centers sit 0.6sp apart, so plain dedup keeps
        // both. They are one source: no pair, the head stays an eighth.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 46.0, "y": 130.0, "w": 30.0, "h": 18.0}, "label": "beam"},
            {"bbox": {"x": 46.0, "y": 138.0, "w": 30.0, "h": 14.0}, "label": "beam"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_one_staff(),
            10.0,
        );
        assert_eq!(parts[0].notes[0].dur, 0.5);
    }

    #[test]
    fn voiced_drops_rest_sticking_out_of_staff() {
        // A rest box extending past the outer staff lines (here ~1sp
        // below the bottom line, e.g. a down-stem classified rest) is
        // not a rest: no rest leaves the staff by half a spacing.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 136.0, "w": 18.0, "h": 18.0}, "label": "rest"},
            {"bbox": {"x": 90.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_one_staff(),
            10.0,
        );
        assert_eq!(parts[0].notes.len(), 1);
        assert_eq!(parts[0].notes[0].onset, 0.0);
    }

    #[test]
    fn voiced_opposite_side_beams_count_as_one_level() {
        // A beam above and a beam below the head (adjacent staves' beams
        // both in the y-window) is one beamed side, not a 16th: 0.5.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let aux = aux_doc(serde_json::json!([
            {"bbox": {"x": 46.0, "y": 60.0, "w": 30.0, "h": 5.0}, "label": "beam"},
            {"bbox": {"x": 46.0, "y": 120.0, "w": 30.0, "h": 5.0}, "label": "beam"}
        ]));
        let parts = map_voiced(&primary, &aux, &layout_one_staff(), 10.0);
        assert_eq!(parts[0].notes[0].dur, 0.5);
    }

    #[test]
    fn voiced_long_stem_beam_still_shortens() {
        // Stems run past an octave, so beams sit 5sp below the head:
        // inside the association window, the head is an eighth.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let aux = aux_doc(serde_json::json!([
            {"bbox": {"x": 46.0, "y": 156.0, "w": 30.0, "h": 5.0}, "label": "beam"}
        ]));
        let parts = map_voiced(&primary, &aux, &layout_one_staff(), 10.0);
        assert_eq!(parts[0].notes[0].dur, 0.5);
    }

    #[test]
    fn voiced_same_side_beam_pair_is_sixteenth() {
        // Two beams ~1sp apart on the same side: a real 16th group.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let aux = aux_doc(serde_json::json!([
            {"bbox": {"x": 46.0, "y": 60.0, "w": 30.0, "h": 5.0}, "label": "beam"},
            {"bbox": {"x": 46.0, "y": 70.0, "w": 30.0, "h": 5.0}, "label": "beam"}
        ]));
        let parts = map_voiced(&primary, &aux, &layout_one_staff(), 10.0);
        assert_eq!(parts[0].notes[0].dur, 0.25);
    }

    #[test]
    fn voiced_void_with_separate_stem_falls_back_to_quarter() {
        // A void head with only a nearby separate stem is usually a faded
        // filled head (stem recall is low), not a half note: 1.0.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-void"}
        ]));
        let aux = aux_doc(serde_json::json!([
            {"bbox": {"x": 52.0, "y": 40.0, "w": 3.0, "h": 100.0}, "label": "stem"}
        ]));
        let parts = map_voiced(&primary, &aux, &layout_one_staff(), 10.0);
        assert_eq!(parts[0].notes[0].dur, 1.0);
    }

    #[test]
    fn voiced_void_isolated_defaults_quarter() {
        // Isolated voids default to quarters: the corpus has no whole
        // notes, and degraded boxes inflate past the whole-width cue, so a
        // wide void is a faded quarter, not a whole.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 16.0, "h": 12.0}, "label": "notehead-void"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_one_staff(),
            10.0,
        );
        assert_eq!(parts[0].notes[0].dur, 1.0);
    }

    #[test]
    fn voiced_beamed_run_shares_duration() {
        // Two heads 1.5sp apart form a run; the beam box only reaches the
        // first head's x-window, but the run shares its level: both 0.5,
        // including the narrow void (faded) head.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 65.0, "y": 99.0, "w": 10.0, "h": 12.0}, "label": "notehead-void"}
        ]));
        let aux = aux_doc(serde_json::json!([
            {"bbox": {"x": 48.0, "y": 60.0, "w": 10.0, "h": 5.0}, "label": "beam"}
        ]));
        let parts = map_voiced(&primary, &aux, &layout_one_staff(), 10.0);
        assert_eq!(parts[0].notes.len(), 2);
        assert_eq!(parts[0].notes[0].dur, 0.5);
        assert_eq!(parts[0].notes[1].dur, 0.5);
    }

    #[test]
    fn voiced_wide_gap_breaks_beamed_run() {
        // Heads 2.5sp apart are separate groups: the beam over the first
        // does not shorten the second.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            {"bbox": {"x": 85.0, "y": 99.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let aux = aux_doc(serde_json::json!([
            {"bbox": {"x": 48.0, "y": 60.0, "w": 10.0, "h": 5.0}, "label": "beam"}
        ]));
        let parts = map_voiced(&primary, &aux, &layout_one_staff(), 10.0);
        assert_eq!(parts[0].notes[0].dur, 0.5);
        assert_eq!(parts[0].notes[1].dur, 1.0);
    }

    #[test]
    fn voiced_marks_rhythmic_durations() {
        let primary = glyphs_doc(serde_json::json!([
            // Beamed eighth (beam above the head).
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            // Dotted quarter (dot right of the head).
            {"bbox": {"x": 90.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"},
            // Half note (void + stem in the same blob).
            {"bbox": {"x": 130.0, "y": 94.0, "w": 10.0, "h": 40.0}, "label": "notehead-void"},
            // Isolated void (no stem anywhere near): quarter fallback.
            {"bbox": {"x": 170.0, "y": 94.0, "w": 16.0, "h": 12.0}, "label": "notehead-void"}
        ]));
        let aux = aux_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "stem"},
            {"bbox": {"x": 46.0, "y": 60.0, "w": 30.0, "h": 5.0}, "label": "beam"},
            {"bbox": {"x": 101.0, "y": 97.0, "w": 4.0, "h": 4.0}, "label": "dot"},
            {"bbox": {"x": 130.0, "y": 94.0, "w": 10.0, "h": 40.0}, "label": "stem"}
        ]));
        let parts = map_voiced(&primary, &aux, &layout_one_staff(), 10.0);
        assert_eq!(parts.len(), 1);
        let durs: Vec<f64> = parts[0].notes.iter().map(|n| n.dur).collect();
        assert_eq!(durs, vec![0.5, 1.5, 2.0, 1.0]);
        // One voice cursor advances by durations: 0, 0.5, 2.0, 4.0.
        let onsets: Vec<f64> = parts[0].notes.iter().map(|n| n.onset).collect();
        assert_eq!(onsets, vec![0.0, 0.5, 2.0, 4.0]);
    }

    #[test]
    fn bass_clef_diatonic_descent_uses_minor_thirds_from_a() {
        // Descent from A (bass top line): A G F E D C B A.
        assert_eq!(diatonic_chroma_from('A', 0), 0);
        assert_eq!(diatonic_chroma_from('A', 1), -2);
        assert_eq!(diatonic_chroma_from('A', 2), -4);
        assert_eq!(diatonic_chroma_from('A', 3), -5);
        assert_eq!(diatonic_chroma_from('A', 4), -7);
        assert_eq!(diatonic_chroma_from('A', 5), -9);
        assert_eq!(diatonic_chroma_from('A', 6), -10);
        assert_eq!(diatonic_chroma_from('A', 7), -12);
        // Ascent from A: A B C D E F G A.
        assert_eq!(diatonic_chroma_from('A', -1), 2);
        assert_eq!(diatonic_chroma_from('A', -2), 3);
        assert_eq!(diatonic_chroma_from('A', -7), 12);
        // Treble descent from F is unchanged.
        assert_eq!(diatonic_chroma_from('F', 0), 0);
        assert_eq!(diatonic_chroma_from('F', 1), -1);
        assert_eq!(diatonic_chroma_from('F', 4), -6);
        assert_eq!(diatonic_chroma_from('F', -1), 2);
    }

    #[test]
    fn bass_staff_maps_f3_to_53_not_54() {
        // Second space of the bass staff is F3 (MIDI 53); the old
        // F-rooted table emitted 54 (F#).
        // cy=110 is one spacing below the bass top line: step 2 -> F3.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 104.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let notes = map_pianoform(&primary, &layout_two_staves(), 10.0);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].pitch.as_deref(), Some("F3"));
    }

    #[test]
    fn tilted_staff_assigns_gap_note_to_true_staff() {
        // Staff B tilts down to the right; a note at x=450 sits 10px below
        // B's tilted top line but flat geometry puts it nearer staff A.
        let layout: LayoutDoc = serde_json::from_value(serde_json::json!({
            "systems": [{"staves": [
                {"bbox": [0.0, -20.0, 500.0, 60.0], "lines": [0.0, 10.0, 20.0, 30.0, 40.0]},
                {"bbox": [0.0, 100.0, 500.0, 160.0], "lines": [100.0, 110.0, 120.0, 130.0, 140.0],
                 "slope": -0.1, "pivot": 250.0}
            ]}]
        }))
        .unwrap();
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 445.0, "y": 64.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let parts = map_voiced(&primary, &aux_doc(serde_json::json!([])), &layout, 10.0);
        // Nearest tilted line is B's top line (80 at x=450): 10 below it.
        // Flat bbox distance would pick A (10 vs 30 the other way), which
        // is the upper staff (voice 1); B is the lower staff (voice 5).
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].name, "Piano");
        assert_eq!(parts[0].notes.len(), 1);
        assert_eq!(parts[0].notes[0].voice, 5);
    }

    #[test]
    fn tilted_staff_pitches_from_line_at_glyph_x() {
        // Slope 0.1 rising to the right, pivot x=0: top line at x=100 is 110.
        let layout: LayoutDoc = serde_json::from_value(serde_json::json!({
            "systems": [{"staves": [
                {"bbox": [0.0, 80.0, 500.0, 160.0], "lines": [100.0, 110.0, 120.0, 130.0, 140.0],
                 "slope": 0.1, "pivot": 0.0}
            ]}]
        }))
        .unwrap();
        // Notehead centered (105, 110.5): on the tilted top line -> F5.
        // Flat geometry would read step 2 (D5).
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 100.0, "y": 104.5, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let parts = map_voiced(&primary, &aux_doc(serde_json::json!([])), &layout, 10.0);
        assert_eq!(parts[0].notes[0].pitch.as_deref(), Some("F5"));
    }

    #[test]
    fn untilted_layouts_keep_legacy_assignment() {
        // No slope/pivot keys: identical to the old flat behavior.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 70.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_two_staves(),
            10.0,
        );
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].name, "Piano");
    }

    #[test]
    fn voiced_beamed_eighth_matches_short_beam_fragment() {
        // Beam fragment ends 2px before the head column: degradation gaps
        // must not hide the beam. Head cx=55, window starts at 50.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let aux = aux_doc(serde_json::json!([
            {"bbox": {"x": 20.0, "y": 60.0, "w": 28.0, "h": 5.0}, "label": "beam"}
        ]));
        let parts = map_voiced(&primary, &aux, &layout_one_staff(), 10.0);
        assert_eq!(parts[0].notes[0].dur, 0.5);
    }

    #[test]
    fn voiced_distant_beam_fragment_does_not_leak() {
        // Beam ends a full spacing short of the head column: still a quarter.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let aux = aux_doc(serde_json::json!([
            {"bbox": {"x": 20.0, "y": 60.0, "w": 20.0, "h": 5.0}, "label": "beam"}
        ]));
        let parts = map_voiced(&primary, &aux, &layout_one_staff(), 10.0);
        assert_eq!(parts[0].notes[0].dur, 1.0);
    }

    #[test]
    fn voiced_duplicate_beam_boxes_count_one_level() {
        // Two overlapping boxes from one fragmented beam: eighth, not 16th.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 94.0, "w": 10.0, "h": 12.0}, "label": "notehead-black"}
        ]));
        let aux = aux_doc(serde_json::json!([
            {"bbox": {"x": 40.0, "y": 60.0, "w": 30.0, "h": 5.0}, "label": "beam"},
            {"bbox": {"x": 46.0, "y": 61.0, "w": 30.0, "h": 5.0}, "label": "beam"}
        ]));
        let parts = map_voiced(&primary, &aux, &layout_one_staff(), 10.0);
        assert_eq!(parts[0].notes[0].dur, 0.5);
    }

    #[test]
    fn voiced_piano_rests_default_eighth_voice_wide_rests_fill_bar() {
        // NOTE: the wide rest sits just inside the top staff (y=8..14 vs
        // lines 0..40): a rest fully outside the staff is a staff artifact
        // (see `rest_box_is_staff_artifact`) and is dropped, so the
        // fixture keeps the wide rest where engraved rests live.
        let primary = glyphs_doc(serde_json::json!([
            {"bbox": {"x": 50.0, "y": 194.0, "w": 10.0, "h": 12.0}, "label": "rest"},
            {"bbox": {"x": 50.0, "y": 8.0, "w": 14.0, "h": 6.0}, "label": "rest"}
        ]));
        let parts = map_voiced(
            &primary,
            &aux_doc(serde_json::json!([])),
            &layout_three_staves(),
            10.0,
        );
        let rd: Vec<f64> = parts[1]
            .notes
            .iter()
            .chain(parts[0].notes.iter())
            .map(|n| n.dur)
            .collect();
        assert_eq!(rd, vec![0.5, 3.0]);
    }
}
