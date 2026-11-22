# score-render

Batch-render `MusicXML` / `MEI` / `ABC` / `MSCX` / `MSCZ` scores from
`input/scores` into clean `PNG` page images plus ground-truth `JSON` sidecars.

Zero-dependency Rust CLI (only `std`, no crates.io fetches, builds offline).

## Layout

```text
data-gen/score-render/
  Cargo.toml
  src/main.rs
  README.md
input/scores/          # you provide: *.musicxml, *.xml, *.mei, *.abc, *.mscx, *.mscz (recursive)
output/renders/        # generated: <stem>.png + <stem>.musicxml + <stem>.json
```

## Build

```sh
cargo build --release -p score-render
```

Always use `--release`: pixel loops are 10–50× slower in debug.

## Run

```sh
# defaults: --input input/scores --out output/renders --renderer auto
cargo run --release -- --input ../../input/scores --out ../../output/renders

# explicit backend + image size (size applies to the stub only)
cargo run --release -- --input ./input/scores --out ./output/renders \
  --renderer verovio --width 1200 --height 900

# real engraving via local MuseScore, all music x text font combos
cargo run --release -- --renderer musescore \
  --musescore-fonts "Bravura,Emmentaler,Gonville,Leland,MuseJazz,Petaluma" \
  --musescore-text-fonts "Edwin,Times New Roman" \
  --input ./input/scores --out ./output/renders-fonts
```

Flags: `-i/--input`, `-o/--out/--output`, `-r/--renderer
(auto|stub|verovio|lilypond|musescore)`, `--musescore-bin PATH`,
`--musescore-fonts A,B,C`, `--musescore-text-fonts X,Y`,
`--no-batch`, `--musescore-dpi N` (default 300 ≈ final training size),
`--width`, `--height`, `-h/--help`.

## MuseScore backend

Headless export (`-F -f -j <jobfile>`): each page becomes
`<stem>-<p>.png` plus a `<stem>.musicxml` ground-truth file, so every
rendered page pairs 1:1 with symbolic notation. `--musescore-bin` pins
the binary; otherwise `MuseScore4`/`musescore`/`mscore` on `PATH`, then
the well-known install paths, are probed. `auto` prefers musescore over
verovio/lilypond/stub.

Batch mode (default) groups units by style and runs ONE MuseScore
launch per group via a `-j` job file. Before each export, previous
outputs for that stem are deleted, so a re-render into a dirty out dir
cannot silently reuse stale pages. Never set `QT_QPA_PLATFORM=offscreen`
(the stock Windows install ships only `qwindows.dll`; a missing platform
plugin hangs the process) — the native platform exports headlessly.
Every invocation runs with a per-process `TEMP`/`TMP` shard so parallel
runs get separate single-instance sockets, and no `MuseScore4` process
may linger before re-running.

### Font matrix

`--musescore-fonts` selects engraving (SMuFL) fonts,
`--musescore-text-fonts` selects score-text fonts; the two form the full
cross product. Each combo is applied through a generated `.mss` style in
`<out>/.styles/` (`musicalSymbolFont` + auto-paired `musicalTextFont`
for music-only runs; `musicalTextFont` plus all 60 `*FontFace` keys for
text runs). Output stems gain slug suffixes
(`<stem>_<Music>_<Text>`), and each JSON sidecar records the combo in
its `"font"` + `"text_font"` fields (`null` when that axis is default).

Verified on MuseScore 4.7.5: engraving fonts Bravura, Emmentaler,
Gonville, Leland, MuseJazz, Petaluma (anything else falls back to
Bravura — verified byte-identical). Text companions Bravura/Leland/
MuseJazz/Petaluma Text plus Edwin plus system fonts (e.g. Times New
Roman, Georgia, Palatino Linotype, Arial, Courier New, Comic Sans MS)
all apply distinctly; "Emmentaler Text"/"Gonville Text" do not resolve
(byte-identical to a bogus control). One-command driver for the sample
corpus matrix: `bash scripts/run-font-matrix.sh`
(see that script's header for the verified lists).

## Renderer trait

`src/main.rs` defines:

```rust
trait ScoreRenderer {
    fn name(&self) -> &'static str;
    fn render(&self, job: &RenderJob, png: &Path, width: u32, height: u32)
        -> Result<RenderOutcome, String>;
}
```

- `StubRenderer` — always available. Writes a valid white-page PNG
  (two 5-line staves) via a dependency-free encoder, plus the JSON sidecar.
- `VerovioRenderer` — shells out to `verovio -o <png> <src>` when the binary
  is on `PATH`; falls back to the stub with a recorded warning otherwise.
- `LilyPondRenderer` — probes `lilypond`; there is no direct
  MusicXML/MEI/ABC path wired, so it records the hook point
  (`musicxml2ly` / custom converter) and uses the stub. This is where a real
  `.ly` conversion step plugs in.
- `auto` (default) prefers `verovio`, then `lilypond`, then `stub`.

To add a backend, implement `ScoreRenderer` and add a match arm in `main`.

## Ground-truth JSON

Each `<stem>.json` next to its `<stem>.png`:

```json
{
  "source_file": "bach/bwv0772.musicxml",
  "format": "musicxml",
  "renderer": "stub",
  "image": "bwv0772.png",
  "image_width": 1000,
  "image_height": 700,
  "note_count": 42,
  "symbols": [],
  "warnings": ["stub renderer: placeholder image, not engraved notation"]
}
```

`note_count` is a heuristic (XML `<note>` elements / ABC note letters).
The `musescore` backend fills `symbols` with per-page glyph boxes
(`{"page": N, "cls": "NoteBlack", "bbox": [x0, y0, x1, y1]}` in PNG
pixels, classes `NoteBlack/NoteVoid/Rest/Beam/Hook/Accidental/NoteDot/
Clef/Stem/Flag/BarLine/...`) by exporting per-page SVG alongside the PNG
and running `symbox-extract` (`--symbox-bin PATH`, probed on `PATH`
when unset) on each PNG/SVG pair. Missing SVG export or binary only
empties `symbols` with a recorded warning; PNG + MusicXML are unaffected.
Other backends leave `symbols` empty.

## Notes

- Unknown extensions are ignored; undecodable files are skipped, not fatal.
- No Python involved: `cargo run` is the whole pipeline step.
