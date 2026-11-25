# assembly-serialize

> **Upstream engine repo:** Core MusicXML/MSCX/MSCZ serialization and `bekern`
> engine code has been extracted into the standalone companion repository
> **`mscz-rs`** (`C:\Users\grego\projects\mscz-rs`). Engine fixes, new notation
> features, and optimizations belong upstream there rather than being
> duplicated here.

Rust lib + CLI that turns a flat note list into **MusicXML 3.1-first**
notation plus a **loss table**, stage **validators**, and a **`.mscz`**
(MuseScore) zip.

```bash
assembly-serialize example --out notes.json
assembly-serialize pack --input notes.json --out out/
```

`pack` writes:

```text
out/
  <stem>.musicxml    # MusicXML 3.1 partwise (primary output)
  <stem>.mscx        # minimal MuseScore 3-compatible score
  <stem>.mscz        # zip: META-INF/container.xml + <stem>.mscx
  loss.json          # loss table (every approximation, with location)
  validation.json    # validator issues over notes + musicxml + mscz
```

`<stem>` derives from the score title (non-alphanumerics become `-`).

## Input schema (note list JSON)

```json
{
  "title": "Etude",
  "composer": "Anon",
  "divisions": 480,
  "time": {"beats": 4, "beat_type": 4},
  "key_fifths": 0,
  "parts": [
    {"name": "Piano", "notes": [
      {"pitch": "C4", "onset": 0.0, "dur": 1.0},
      {"pitch": "E4", "onset": 0.0, "dur": 1.0},
      {"rest": true, "onset": 1.0, "dur": 1.0},
      {"pitch": "G4", "onset": 2.0, "dur": 2.0, "voice": 1, "lyric": "la"}
    ]}
  ]
}
```

A flat shorthand is accepted instead of `parts`:

```json
{"title": "Solo", "part_name": "Flute", "notes": [ ... ]}
```

Field notes:

- `pitch`: scientific pitch, `C4` .. `B8`, sharps `#`/`##`/`x`, flats
  `b`/`bb`, natural `n`. A trailing cents suffix (`"A4+30c"`,
  `"G3-45"`) is rounded to the nearest semitone and logged as loss.
- `onset` / `dur` are beats (quarter-note units). They are kept as exact
  integer ticks (`round(beats * divisions)`); `<duration>` always carries
  the exact count, so measure sums hold exactly.
- `voice` is 1-based (default 1). Same-voice notes sharing an onset and
  duration become a chord (`<chord/>`). Same-voice overlaps are clamped
  (or dropped when fully covered) and logged.
- Gaps between notes are filled with rests (logged as `info`).
- Notes crossing a barline are split with ties in MusicXML (lossless).

## Commands

```bash
# Note list -> <stem>.musicxml + loss.json
assembly-serialize serialize --input notes.json --out out/ [--stem etude]

# Validators only (exit 1 when errors are found)
assembly-serialize validate --input notes.json \
  [--musicxml out/etude.musicxml] [--mscz out/etude.mscz] \
  [--report validation.json]

# Note list -> .mscz directly
assembly-serialize mscz --input notes.json --out etude.mscz

# Full pipeline (musicxml + mscx + mscz + loss.json + validation.json)
assembly-serialize pack --input notes.json --out out/ [--stem etude]

# MusicXML page -> bekern target tokens (reference adapter for training/scoring)
assembly-serialize bekern --input ref.musicxml --out refs/ --page 1 --stem ds0001-1

# bekern token file -> note-list JSON (decode only, for scoring)
assembly-serialize bekern-decode --input page-bekern.txt --out notes.json

# bekern token file -> pack outputs (decode + parse + serialize in one step;
# loss table + validators identical to pack by construction)
assembly-serialize bekern2musicxml --input page-bekern.txt --out out/ [--stem page]

# Symbol Error Rate of hyp.json against ref.json (both note-list JSON)
assembly-serialize ser --reference ref.json --hypothesis hyp.json [--report ser.json]

# Write a small example input file
assembly-serialize example [--out example-notes.json]

# MusicXML ground truth to note-list JSON (reference adapter)
assembly-serialize import --input ref.musicxml --out refs/ [--pages]
```

## Import (`import`)

Parses Partwise 3.1 (pitched notes, rests, chords, backup/forward voices,
per-part divisions). Ties merge into single logical notes; grace notes are
skipped; the first time/key signature wins. With `--pages`, one file per
`<print new-page>` page (`<stem>-p<page>.json`) for page-level scoring.

## SER (`ser`)

SER is Levenshtein distance over canonical symbol sequences divided by
reference length. One symbol is one note/rest event:
`"p{part}:m{MIDI}:d{dur_ticks}:v{voice}"` (rests use `R` for pitch), in
score order. Both files are parsed with `parse_score`, so tick rounding
and sorting apply before scoring. `ser.json` carries
`{distance, ref_len, hyp_len, ser}`.

## MusicXML 3.1-first design

- Output starts with the XML declaration, the Partwise 3.1 DOCTYPE, and
  `<score-partwise version="3.1">`.
- One `<part>` per input part, one `<measure>` per bar, treble clef, and
  the input key/time signature on measure 1.
- First voice is written straight; further voices follow `<backup>`; every
  voice is rest-filled to the full bar.
- `<type>`/`<dot>` are display-only: the nearest type+dots value is shown
  and any discrepancy against the exact `<duration>` is logged as
  `note-type-approximated` loss (timing itself is never altered).

## Loss table (`loss.json`)

```json
{"entries": [
  {"severity": "loss", "kind": "pitch-microtone-rounded",
   "message": "pitch \"A4+30c\" carries +30 cents; rounded to MIDI 69",
   "part": "Right", "note_index": 3}
]}
```

| kind | severity | meaning |
|---|---|---|
| `pitch-microtone-rounded` | loss | cents suffix rounded to nearest semitone |
| `note-type-approximated` | loss | exact ticks shown with nearest type+dots |
| `overlap-clamped` | loss | onset moved forward to resolve same-voice overlap |
| `note-dropped-overlap` | loss | note fully covered by an earlier note, dropped |
| `mscx-tie-dropped` | loss | tie re-articulated in `.mscx` (no spanners in subset) |
| `mscx-lyric-dropped` | loss | lyric not carried by the `.mscx` subset |
| `rest-inserted` | info | gap filled with rests (lossless normalisation) |

Hard errors (negative onsets, non-positive durations, bad pitches,
out-of-range MIDI, empty score) abort conversion instead of logging.

## Validators

- `validate_notes` — input structure: `notes-*` codes (negative onset,
  bad pitch, pitch range, voice 0, ...). Warnings for unsorted onsets,
  gaps, and microtones that conversion will normalise.
- `validate_musicxml` — works on our output or foreign files: XML
  declaration, `version="3.1"`, per-measure duration sums honouring
  `<backup>`/`<forward>` and the no-advance rule for `<chord/>` tones,
  positive durations, pitched notes with step+octave, escaped entities.
  Codes: `xml-*`.
- `validate_mscz` — readable zip, `META-INF/container.xml` present,
  rootfile ends in `.mscx` and exists, `<museScore>` root with at least
  one `<Measure>`. Codes: `mscz-*`.

Every issue is `{"severity": "error"|"warning", "code": "...", "message": "..."}`.

## `.mscz` layout

```text
<stem>.mscx                  # minimal MuseScore 3-compatible score
META-INF/container.xml      # <rootfile full-path="<stem>.mscx"/>
```

The `.mscx` subset carries one `<Staff>` per part, `<TimeSig>`/`<KeySig>`
on the first measure, and one `<voice>` per MusicXML voice with
`<Chord>` (`<pitch>` + line-of-fifths `<tpc>`) and `<Rest>` elements. It
opens for interchange and re-export; ties, lyrics, and display-only
nuances beyond `durationType`+`dots` are intentionally not carried (each
is a logged `mscx-*` loss). The `.musicxml` file is the lossless primary
output.

## Library use

```rust
let input: ScoreInput = serde_json::from_str(&text)?;
let (score, mut losses) = parse_score(&input)?;
let xml = to_musicxml(&score, &mut losses);
let mscx = to_mscx(&score, &mut losses);
let mscz = pack_mscz(&score, &mscx)?;
let issues = validate_musicxml(&xml);
```

Run the tests with `cargo test`. Formatting: `cargo fmt`; lints:
`cargo clippy --all-targets`.
