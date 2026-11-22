# dataset-pack

Rust CLI that turns a directory of sheet-music pairs into a versioned,
DVC-friendly dataset package.

## Pair convention

Each sample is two files sharing one stem:

```text
input/
  0001.png
  0001.musicxml
  0002.jpg
  0002.krn
```

- Image side: `png jpg jpeg tif tiff bmp webp`
- Notation side: `musicxml xml mxl mei krn kern abc mid midi`
- Exactly one image + one notation file per stem = valid.
- Anything else (orphan image, orphan notation, duplicates, empty files)
  is reported by `validate` and blocks `pack` unless `--allow-partial`.

Search is recursive; the sample id is the path stem relative to `--input`
(parent dirs are joined with `/`, e.g. `batch1/0001`).

## Commands

```bash
# Check pairs (exit 1 when problems are found)
dataset-pack validate --input data/raw --report data-pack/validation.json

# Deterministic train/val/test split of the valid ids
dataset-pack split --input data/raw --out data-pack \
  --train 0.8 --val 0.1 --test 0.1 --seed 42

# Full package: validate + split + manifest + card + dvc.yaml
dataset-pack pack --input data/raw --out data-pack \
  --name sheet-music-v1 --seed 42 --allow-partial
```

`split` is deterministic: `sha256("{seed}/{id}")` decides the bucket, so the
same `--seed` and ratios always give the same splits on any machine.

## `pack --strict` (training-data gates)

Legacy `pack` only checks pairing/emptiness/hashes and hashes **each sample
id independently**, so pages of one score and `.aug<N>` variants of one page
routinely land in different splits. `--strict` adds four gates and fails the
pack (exit 1, nothing written) when any fires. Default runs are byte-identical
to before.

```bash
dataset-pack pack --input data/raw --out data-pack --allow-partial --strict
# optional: override the kept-class list (default: the 10 detector classes
# NoteBlack,Rest,Beam,Hook,Accidental,NoteDot,Clef,NoteVoid,Stem,Flag)
dataset-pack pack --input data/raw --out data-pack --strict \
  --strict-classes NoteBlack,Rest
```

1. **Box schema + bounds** — every `*-boxes.json` sidecar (`<stem>-boxes.json`,
   symbox `glyphs[]` format) is validated: each glyph needs a non-empty `cls`
   string and a `bbox` of 4 finite numbers with positive area, inside the page
   bounds (`png_size`, else same-dir `<stem>.png` IHDR). Problems:
   `box-schema`, `box-degenerate`, `box-out-of-bounds`, `box-unreadable`.
   When any sidecar exists, every valid pair (aug/degraded variants share
   their page's labels) needs one (`box-missing`).
2. **Class histogram** — the old imbalance warning, escalated: a kept class
   with zero boxes (`class-absent`) or one class holding >95% of boxes
   (`class-dominance`) fails the pack.
3. **Piece-level split integrity** — all aug variants (`.aug<N>`/`_aug<N>`,
   `_d<N>`, `_degraded`) and all pages (`-<N>`) sharing a score stem must land
   in the SAME split; straddling pieces fail with `split-leakage`.
4. **Degradation-param linkage** — every degraded image (`*_degraded.*`,
   `*_d<N>.*`) must sit next to its `<stem>{suffix}_params.json` recording a
   `file_seed`/`seed` and an `input`/`output` referencing the base stem.
   Problems: `params-missing`, `params-seed-missing`, `params-stem-mismatch`,
   `params-unreadable`.

### Current verdict: `--strict` FAILS on the big500 page-level layout

Packing per-page pairs (`ds0001-1`, `ds0001-2`, … plus `.aug<N>` variants)
passes loose but fails strict with `split-leakage`: e.g. `ds0001-1.aug0` →
train while `ds0001-1` → val and `ds0001-1.aug1` → test, and `ds0002-1` →
test while `ds0002-2` → train. This is the known hole — the per-id hash in
`assign_splits` must become piece-aware (hash the piece key, not the page/aug
id) before any detector training split is cut. No data files were touched;
the verdict was reproduced on a throwaway miniature of the layout in
`$TEMP/big500det-mini` (8 pairs + 5 box sidecars + 1 linked params file).

## `pack` outputs

```text
<out>/
  manifest.jsonl        # one JSON object per sample (committed to git)
  dataset-meta.json     # counts, ratios, seed, timestamps
  splits/train.txt      # one sample id per line
  splits/val.txt
  splits/test.txt
  validation.json       # full pair validation report
  DATASET_CARD.md       # dataset card (fill in the TODOs)
  dvc.yaml              # `validate`/`pack` stages for `dvc repro`
  .gitignore            # created when missing: keeps raw data out of git
```

`manifest.jsonl` row:

```json
{"id":"0001","split":"train","image":"0001.png","notation":"0001.musicxml","image_sha256":"…","notation_sha256":"…","image_bytes":1234,"notation_bytes":567}
```

Paths in the manifest are POSIX-style, relative to `--input`.

## DVC workflow

```bash
dataset-pack pack --input data/raw --out data-pack --name sheet-music-v1
git add data-pack/manifest.jsonl data-pack/DATASET_CARD.md data-pack/dvc.yaml
dvc add data/raw                 # or: dvc import-url, dvc push to your remote
dvc repro --project data-pack    # re-runs the pack stage
```

Large files stay in DVC remotes; the manifest, splits, card, and `dvc.yaml`
stay in git so any commit fully describes the dataset version.
