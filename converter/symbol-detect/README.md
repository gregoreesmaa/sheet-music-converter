# symbol-detect

Rust CLI that maps staff crops plus a staff map to glyph boxes/labels JSON.
Each crop image is decoded, run through a `GlyphDetector` backend
([`src/main.rs`](src/main.rs)), and reported with its glyph detections:
bounding box, label, confidence, staff-relative position, and ink area.

## Inputs

### Crops directory

`--crops-dir` holds one image per staff crop (`.png`, `.jpg`, `.jpeg`,
`.bmp`, `.tif`, `.tiff`, `.webp`, plus netpbm `.pbm`/`.pgm`/`.ppm`/`.pnm`).
The crop id is the file stem (`m1.png` -> `m1`), and files are processed
in sorted name order for stable output.

### Staff map

`--staff-map` is JSON in any of three shapes — wrapped, `staves`-wrapped,
or a bare array — with an optional UTF-8 BOM tolerated:

```json
{
  "systems": [
    {
      "id": "sys0",
      "crop_id": "m1",
      "staff_lines": [12.0, 22.0, 32.0, 42.0, 52.0],
      "staff_spacing": 10.0
    }
  ]
}
```

Fields:

- `id` (string, optional): entry label, informational only.
- `crop_id` / `crop` (string, optional): crop stem or file name this entry
  applies to. An entry without one is global and applies to every crop
  that has no dedicated entry.
- `staff_lines` (array of numbers, default `[]`): staff-line y positions
  in crop pixels, top line first. Used to snap each glyph to a
  half-space `staff_step` below the top line.
- `staff_spacing` (number, optional): line-to-line distance in pixels;
  falls back to `--staff-spacing` when absent.

A crop with no matching entry and no global entry uses `--staff-spacing`
with no staff lines (no `staff_step` values).

## Usage

```sh
cargo run -- --crops-dir crops --staff-map staff-map.json --out glyphs.json
```

Full options:

```sh
cargo run -- --crops-dir crops --staff-map staff-map.json --out glyphs.json \
  --backend classical --threshold 128 --min-area 16 --staff-spacing 10.0 \
  --labels notehead-black,notehead-void,stem,beam,dot,accidental,clef,rest
```

`--threshold` has no effect on two-valued ingest-binarized input (pixels are
already 0/255); `--min-area 16` is big500-calibrated
(see `../TRAINING_BIG500.md`).
```

## Backends

- `classical` (default): binarizes each crop at `--threshold`, extracts
  8-connected ink blobs, drops blobs under `--min-area` pixels as noise,
  and labels each blob by staff-normalized shape rules: tiny blobs are
  `dot`s, tall-narrow bars are `stem`s, wide-short bars are `beam`s,
  tall-wide blocks are `clef`s, staff-sized dense blobs are
  `notehead-black` (sparse ones `notehead-void`), mid-size blocks are
  `accidental`, anything else is `unknown`. Rule labels outside `--labels`
  also become `unknown`. Fully deterministic: same pixels plus same
  staff map give identical output. `confidence` is a stub ranking signal
  from ink density (denser blobs score higher), not a probability.
- `onnx` (hook): requires `--model <model.onnx>` and goes through
  `OnnxDetectorHook`, which shares the `GlyphDetector` trait, CLI, and
  output schema with the classical backend. This crate ships with no ONNX
  runtime, so the hook returns an error until one is wired in. To enable:

  1. Uncomment the `ort` optional dependency and `onnx` feature in
     `Cargo.toml`.
  2. Load the session in `OnnxDetectorHook::load` (e.g. with
     `ort::session::Session::builder`).
  3. Replace the stub `detect` body with preprocess -> session run ->
     box decoding, emitting the same `Glyph` records.

## Output

`--out` receives a JSON document:

```json
{
  "backend": "classical-stub",
  "crops": [
    {
      "crop_id": "m1",
      "source": "m1.png",
      "width": 512,
      "height": 128,
      "glyphs": [
        {"bbox": {"x": 12, "y": 30, "w": 14, "h": 10}, "label": "notehead-black", "confidence": 0.87, "staff_step": 2.0, "area": 98}
      ]
    }
  ]
}
```

- `backend`: detector name (`classical-stub`; the ONNX hook reports
  `onnx-hook` once implemented).
- `glyphs`: detections sorted top-to-bottom, left-to-right. `bbox` is in
  crop pixels; `staff_step` is half staff-spaces below the top staff
  line (omitted when the crop has no staff lines); `area` is the ink
  pixel count.

The run also prints a one-line summary, e.g.
`symbol-detect: backend=classical-stub crops=4 glyphs=63 -> glyphs.json`.

## Checks

```sh
cargo build
cargo test
```
