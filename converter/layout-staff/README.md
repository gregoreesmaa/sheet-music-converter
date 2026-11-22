# layout-staff

Rust CLI that maps staff layout in a **cleaned sheet-music image**
(dark ink on light paper, e.g. deskewed/denoised output): it detects
staff-line rows, groups them into 5-line staves, groups nearby staves into
systems, and writes a systems/staves JSON file plus a staff-probability
map PNG.

Classical fallback (compiled in): binarize at `--threshold`, take the
per-row ink-fill profile, pick staff-line rows with local-maximum peak
detection (`--min-fill`, `--merge-px`), group rows into staves by the
estimated staff space, then group staves closer than
`--system-gap-spaces` staff spaces into systems.

Neural hook: the `StaffLineModel` trait in `src/main.rs` abstracts the
"image -> per-row staff probability" step (`predict_rows`). The compiled-in
`ClassicalModel` implements it with the projection method above; a learned
backend (e.g. ONNX/Tch) plugs in by implementing the same trait and adding
a `--model` arm in `select_model`. Downstream grouping is backend-agnostic.

## Build

```sh
cargo build --release
```

Requires Rust 1.70+ and network access to fetch crates on first build
(`image`, `clap`, `serde`/`serde_json`).

## Usage

Single file:

```sh
layout-staff --input clean.png --out-dir out
```

Batch directory (non-recursive, `*.png`/`*.jpg`/`*.jpeg`, sorted):

```sh
layout-staff --input clean-dir/ --out-dir layouts/ --threshold 140 --min-fill 0.3
```

Options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--input` | (required) | Cleaned image file or directory of images |
| `--out-dir` | `out` | Output directory (created if missing) |
| `--threshold` | `140` | Binarization threshold; pixels darker than this count as ink (no effect on two-valued ingest-binarized input) |
| `--min-fill` | `0.10` | Minimum row ink-fill fraction for a line candidate, in (0, 1] (big500-calibrated; see `../TRAINING_BIG500.md`) |
| `--merge-px` | `3` | Merge peak rows at most this far apart into one staff line |
| `--system-gap-spaces` | `4.0` | Max inter-staff gap, in staff spaces, before a new system starts |
| `--model` | `classical` | Detector backend (only `classical` is compiled in) |
| `--weights` | — | Reserved for a neural backend; errors when set |

## Output (paired)

For input `page.png` in `--out-dir out/`:

- `out/page_layout.json` — systems, staves, line rows, and counts:

```json
{
  "input": "page.png",
  "model": "classical",
  "width": 800,
  "height": 1100,
  "threshold": 140,
  "min_fill": 0.3,
  "staff_space_px": 9.0,
  "systems": [
    {
      "id": 0,
      "bbox": [20, 100, 780, 260],
      "staves": [
        {
          "id": 0,
          "bbox": [20, 100, 780, 140],
          "lines": [102, 111, 120, 129, 138],
          "confidence": 0.97,
          "complete": true
        }
      ]
    }
  ],
  "counts": { "lines": 10, "staves": 2, "systems": 1 }
}
```

Bboxes are `[x_min, y_min, x_max, y_max)` in image pixels. `complete` is
true for full 5-line staves; shorter groups (3–4 lines, `confidence`
penalized ×0.6) are kept, smaller ones discarded as noise.

- `out/page_staffprob.png` — grayscale map the same size as the input:
  each row holds its staff probability (0–255). Bright bands mark detected
  staff lines; dimmer bands reflect other horizontal ink density.

A blank page yields empty `systems` (not an error). Exit code is non-zero
with a message on stderr for missing input, unreadable images, unknown
`--model`, `--weights` without a neural backend, or unwritable output.

## Tests

```sh
cargo test
```

Covers a synthetic two-staff page (exact line rows, staff space, system
grouping), trait-vs-direct equivalence, the blank-page case, and model
selection.
