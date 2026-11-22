# ingest-cleanup

First `converter/` stage for sheet-music photos: turns one photographed (or
scanned) page into a **deskewed binary (black-on-white) image** ready for
staff detection, with **quality gates** that reject unusable shots.

Pipeline (in order):

1. **EXIF normalize** — read the EXIF orientation flag (JPEG/TIFF) and
   rotate/flip the raw pixels upright. Files without EXIF (e.g. PNG) pass
   through unchanged.
2. **Quality gates** — reject (non-zero exit, no output written):
   - **blur**: variance of the Laplacian below `--blur-threshold`
     (exit code 2);
   - **glare** (exit code 3): specular blowout is *local* — hot pixels
     (at/above `--glare-level`) only count when the page median sits well
     below saturation, so uniformly bright clean scans pass. A page where
     almost no ink survives Otsu fails as washed out regardless.
3. **Deskew** — estimate the skew angle by maximizing the variance of the
   horizontal projection profile over `±--max-skew-deg`, then rotate back.
   Out-of-frame pixels are filled with paper white. `--no-deskew` skips this.
4. **Binarize** — Otsu threshold (or `--threshold`) to pure black/white.

## Build

```sh
cargo build --release
```

Requires Rust 1.70+ and network access to fetch crates on first build
(`image`, `clap`, `kamadak-exif`).

## Usage

```sh
ingest-cleanup --input photo.jpg --output clean.png
```

With tuned gates:

```sh
ingest-cleanup --input photo.jpg --output clean.png \
  --blur-threshold 120 --glare-threshold 0.03 --glare-level 245 \
  --max-skew-deg 15 --skew-step-deg 0.25
```

Options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--input` | (required) | Input image (PNG/JPEG/…) |
| `--output` | (required) | Output binary image (format from extension) |
| `--blur-threshold` | `6` | Min Laplacian variance (below → blur reject) |
| `--glare-threshold` | `0.12` | Max fraction of glare pixels (above → glare reject) |
| `--glare-level` | `250` | Luma value treated as blown-out highlight |
| `--max-skew-deg` | `10` | Deskew search range, ±degrees |
| `--skew-step-deg` | `0.5` | Deskew search step, degrees |
| `--threshold` | Otsu | Fixed binarization threshold 0–255 |
| `--no-deskew` | off | Skip deskew, only binarize |
| `--force` | off | Write output even when a gate fails (exit stays 0) |

Stdout reports each stage (`PASS`/`REJECT`, skew angle, threshold, output
dimensions); failures also print to stderr.

## Output

Single file: the deskewed binary image at `--output` (canvas grows slightly
when rotated, same as any deskew).

## Exit codes

- `0` — success (or `--force` with a failed gate, plus a stderr warning).
- `1` — usage / I/O error (missing input, bad flags, unwritable output).
- `2` — blur reject.
- `3` — glare reject.

## Notes

- Thresholds depend on camera and lighting: calibrate `--blur-threshold`
  on a few known-good/known-bad shots before batch use (defaults are
  calibrated on 398 fresh big500 phone photos — see `../TRAINING_BIG500.md`).
- Skew estimation assumes the page carries roughly horizontal structure
  (staff lines, text); on a blank page the angle is meaningless and snaps
  to `0`.
- Binarization maps paper to white, so rotation fill and paper agree.
