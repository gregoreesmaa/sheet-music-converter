# boxnote-mapper

Rust lib + CLI that maps glyph boxes/labels plus a staff layout to a
note-list JSON file for `assembly-serialize`.

```bash
boxnote-mapper --glyphs glyphs.json --layout page_layout.json --out notes.json
```

## Backends

- `naive` (default): notehead glyphs become quarter (`notehead-black`)
  or half (`notehead-void`) notes at treble-clef diatonic pitches from the
  nearest staff (top line = F5); `rest` glyphs become quarter rests; all
  other labels are ignored. One monophonic voice with cumulative onsets.
  The honest-baseline mapper.
- `pianoform`: the bottom staff of each multi-staff system reads as bass
  clef (top line = A3); x-overlapping noteheads share an onset (chords).
  Still one voice with nominal durations — a stepping stone.
- Future learned mappers plug in behind `--backend` with the same I/O.

## Checks

```sh
cargo test
cargo fmt --check
cargo clippy --all-targets
```
