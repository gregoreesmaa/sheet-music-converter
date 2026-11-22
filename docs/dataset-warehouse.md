# Dataset warehouse: generate once, train forever

Policy: **training runs never regenerate pixels.** Every dataset version is
generated once (single machine or farm-sharded per
[farm-scaleout](farm-scaleout.md)), packed, content-addressed, and stored;
training consumes the warehouse read-only.

## Layout

```text
warehouses/<name>/            # e.g. warehouses/sheet-music-v1/
  raw/                        # DVC-tracked: <id>.jpg + <id>.musicxml pairs
  manifest.jsonl              # git: one JSON object per sample (id, split,
                              #   image, notation, sha256+bytes of each)
  dataset-meta.json           # git: name, seed, ratios, counts
  splits/train.txt|val.txt|test.txt   # git: sample ids per split
  validation.json             # git: pair-convention validation report
  DATASET_CARD.md             # git: fill in the TODOs per version
  dvc.yaml                    # git: validate/pack stages for `dvc repro`
  .gitignore                  # keeps raw pixels out of git
```

Produced by one command (see [runbook](farm-runbook.md) §6):

```powershell
dataset-pack.exe pack --input pairs/ --out warehouses/<name> --name <name> --seed <SPLIT_SEED>
```

`split`/`pack` bucket by `sha256("{seed}/{id}")`, so splits are
deterministic per id on any machine — re-running `pack` on the same pairs
with the same seed reproduces the warehouse descriptors exactly.

## Versioning rules

1. **Immutable versions.** A warehoused `<name>` is never edited in place.
   New data, new flags, or a new split seed ⇒ new `<name>`.
2. **Git stores descriptors, DVC stores bytes.** Commit `manifest.jsonl`,
   `splits/`, `DATASET_CARD.md`, `dvc.yaml`, `dataset-meta.json`;
   `dvc add raw/` + `dvc push` the pixels. Any commit fully describes its
   dataset; any dataset is fetchable without rerunning the farm.
3. **One backend per version.** CPU and `--gpu` pixels differ at the same
   seed by design (GPU hash streams; `backend` in each params JSON).
   Record the backend in the dataset card; never mix backends in a version.
4. **Provenance per file.** Keep each sample's `<stem>_params.json`
   (photo-degrade) and render sidecar JSON (`source_file`, `renderer`,
   `font`, `musicxml`) alongside or referenced from the warehouse so any
   pixel is traceable to `(seed, stage, key)`.
5. **Training reads the warehouse.** Trainers take `--data warehouses/<name>`
   (manifest + splits) as input. "Regenerate on the fly" is not a fallback —
   if the warehouse lacks a needed variant, cut a new version through the
   farm procedure instead.

## Pilot proof (100 pages, 2026-10-25)

`tmp-train/farm-pilot/warehouse/sheet-music-pilot/` (scratch, not committed):

```powershell
dataset-pack.exe pack --input tmp-train/farm-pilot/pairs `
  --out tmp-train/farm-pilot/warehouse/sheet-music-pilot `
  --name sheet-music-pilot --seed 20261022
# pack: 100 valid (+0 invalid skipped)
#   manifest.jsonl splits={"test": 13, "train": 77, "val": 10}
```

Manifest row shape (paths POSIX-style, relative to `--input`):

```json
{"id":"p000_degraded","split":"train","image":"p000_degraded.jpg",
 "notation":"p000_degraded.musicxml",
 "image_sha256":"28eacda45396cf727c8f5e125790e0bf45ac1cb4c7bf502fae5e53b9ae358d2c",
 "notation_sha256":"b97e255de65e064a6566c6e621930fc4999335f0d3cd71926382f128ef7259b4",
 "image_bytes":288205,"notation_bytes":1160}
```

Followed by the standard DVC flow: `git add` the descriptors,
`dvc add` the raw pairs dir, `dvc push`, `dvc repro` to re-verify.
