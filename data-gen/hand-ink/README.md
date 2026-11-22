# hand-ink

Rust stage that turns a **clean render PNG** into a **handwritten-ink variant**,
for synthetic training-data generation.

Deterministic Rust port of the `v3_high` transform from
`tmp-train/hw-synth/v3/v3_levels.py` (level `"v3_high"`):

1. **Smooth wobble** — two Gaussian-smoothed white-noise fields
   (field sigma 10 px), normalized so the peak absolute displacement is
   5 px, then a nearest-neighbor remap (coordinates clipped to the frame).
2. **Ink jitter** — a sigma-40-px smoothed field modulates per-pixel gain
   (`1 + 0.14·s`, clamped to `[0.93, 1.238]`); dark channels (`< 100`) take
   the raw gain, light ones take `max(gain, 1)`.
3. **Paper grain** — warm tint (`1.0, 0.986, 0.945` around mid-grey) plus
   per-pixel grain (sigma 2.2, shared across channels).

Topology-preserving by construction: nothing is deleted, every dark pixel
survives moved by at most 5 px, so ground-truth notation labels still apply
(dark-pixel-count ratio stays within 1% of the input; recorded per file).

Sibling levels from the same reference script are supported via `--level`:
`med` (3 px / sigma 14 / ink 0.10 / grain 1.6) and
`scribbly` (8 px / sigma 8 / ink 0.20 / grain 3.0).

## Determinism

Every byte derives from `(seed, file index)`: file `i` of a sorted batch uses
`seed + i * 0x9E3779B97F4A7C15` (same stride convention as photo-degrade), and
all noise inside one file comes from a SplitMix64 stream on that per-file
seed. Same seed + inputs ⇒ bit-identical outputs. Artifacts are written to
`*.tmp` then renamed, so a killed run never leaves a half-file that a later
`--resume` would trust.

Pipeline placement: `render -> hand-ink -> vary -> degrade -> pack`.
hand-ink consumes clean render PNGs and its outputs feed `vary`/`degrade`;
see `data-gen/pipeline-orchestrator/README.md`. (The orchestrator still
sequences placeholder stage bodies; run this binary between `render` and
`vary`, or point later stages at its outputs.)

## Build

```sh
cargo build --release   # pixel loops are 10-50x slower in debug
```

Requires Rust 1.70+ and network access on first build
(`image`, `clap`, `serde_json`; all MIT/Apache-2.0).

## Usage

Single file:

```sh
hand-ink --input clean.png --out-dir out --seed 1234 --level high
```

Batch directory (non-recursive, `*.png`, sorted):

```sh
hand-ink --input renders/ --out-dir hand-inked/ --seed 1234
```

Resume after an interruption (skips files whose PNG + params JSON exist):

```sh
hand-ink --input renders/ --out-dir hand-inked/ --seed 1234 --resume
```

| Flag | Effect |
|---|---|
| `--input <PATH>` | Clean PNG file or directory of PNGs |
| `--out-dir <PATH>` | Receives `<stem>{suffix}.png` + `<stem>{suffix}_params.json` |
| `--seed <U64>` | Master seed (default `1234`, the v3 reference seed) |
| `--level <med\|high\|scribbly>` | Strength (default `high` = `v3_high`) |
| `--suffix <S>` | Output filename suffix (default `_hand`) |
| `--resume` | Skip files already fully written |
| `--fail-after <N>` | Exit(2) after N files — test aid for `--resume` |
| `-h, --help` | Print help |

Exit codes: `0` ok · `1` error · `2` `--fail-after` trip.

## Regenerable sample

`samples/ds0002-1_hand.png` (+ `_params.json`) is the checked-in v3_high
sample, regenerated from the existing big500 render with the reference seed:

```sh
cargo run --release -- --input ../score-corpus/big500-renders/ds0002-1.png \
  --out-dir samples --seed 1234 --level high
```

(dark-count ratio 0.9992 — inside the 1% label-preservation band).
