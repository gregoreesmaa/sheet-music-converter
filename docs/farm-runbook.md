# Farm runbook: sharded render + degrade, merge, verify, warehouse

End-to-end operator procedure. Theory (why each step preserves
bit-identity) is in [farm-scaleout](farm-scaleout.md); storage policy after
`pack` is in [dataset-warehouse](dataset-warehouse.md).

All paths relative to the repo root. Build once per machine (release —
pixel loops are 10–50× slower in debug):

```powershell
cargo build --release --manifest-path data-gen/photo-degrade/Cargo.toml
cargo build --release --manifest-path data-gen/score-render/Cargo.toml
cargo build --release --manifest-path data-gen/dataset-pack/Cargo.toml
```

Binaries: `data-gen/<crate>/target/release/<crate>.exe`.

## 1. Freeze the dataset version

Pick and record: `MASTER_SEED`, backend flags (`--format`, `--strength`,
`--max-width`, `--gpu` on/off — uniform across all shards), and the exact
**global sorted input file list**. The list defines global indices; any
later addition/rename changes indices, so freeze it first:

```powershell
Get-ChildItem <input-dir>/*.png | Sort-Object Name | ForEach-Object { $_.Name } | Out-File frozen-list.txt
```

## 2. Render shards (score-render)

Partition the input score set across machines (any partition — render is
per-score deterministic). One MuseScore invocation per style group stays
per machine (see repo notes on batch mode and per-process TEMP shards);
verify no `MuseScore4` process lingers before re-running on a worker.

## 3. Degrade shards (photo-degrade)

On the coordinator, split `frozen-list.txt` into contiguous chunks
`[start_k, end_k)` — one chunk dir per worker, same filenames:

```powershell
# per worker k: copy its chunk into shard_k/, then on worker k run:
photo-degrade.exe --input shard_k --out-dir out_k --seed <SHARD_SEED_k> --format jpeg
```

with `SHARD_SEED_k = (MASTER_SEED + start_k * 0x9E3779B97F4A7C15) mod 2^64`:

```powershell
$M = [bigint]::Pow(2,64)
$S = [bigint]::Parse('09E3779B97F4A7C15', 'AllowHexSpecifier')
$shardSeed = ([bigint]$MASTER_SEED + [bigint]$start_k * $S) % $M
```

Keep `--suffix` identical on all shards (default `_degraded`) and use a
distinct `--out-dir` per shard. Params files already carry the run suffix
(`<stem>{suffix}_params.json`) so parallel passes never share them.

## 4. Merge

Copy (or `robocopy`) every `out_k/*` into one `merged/` dir. Outputs are
stem-keyed, so this is conflict-free. Merge orchestrator state files, if
used, by concatenating row lines under one `v1` + `seed=` header.

## 5. Verify (required before warehousing)

```powershell
# every shard output must exist exactly once in merged/
# spot-check bit-identity: re-run any one shard locally and compare SHA-256
Get-ChildItem merged/*.jpg | Get-FileHash -Algorithm SHA256 | Sort-Object Path |
  Export-Csv merged-sha256.csv
```

Gate: file count equals the frozen list count, and any re-run shard is
byte-identical. A mismatch means the shard used a wrong seed, a wrong file
subset/order, or mixed backend flags — fix and re-run that shard only.

## 6. Pack + warehouse

```powershell
# stage image+notation pairs sharing one stem per sample, then:
dataset-pack.exe pack --input pairs/ --out warehouses/<name> --name <name> --seed <SPLIT_SEED>
```

Commit `manifest.jsonl`, `dataset-meta.json`, `splits/`, `DATASET_CARD.md`,
`dvc.yaml`, `validation.json` to git; push raw data with DVC
(`dvc add` the raw dir, `dvc push`). Never edit a warehoused version —
cut a new `<name>` (and new seed if resplitting).

## 7. Failure handling

- Worker dies mid-shard: re-run that shard's exact command (same files,
  same `--seed`) — outputs are deterministic, resume is a plain overwrite.
- `--resume` (orchestrator stages): reuses `<out>/.pipeline-state`; a seed
  mismatch aborts instead of mixing datasets.
- Straggler worker: its chunk is contiguous and self-contained — reassign
  the whole chunk (with its `SHARD_SEED`) to another machine, never split
  a started chunk across two seeds.
- Kill safety: artifacts use tmp+rename, so a killed run never leaves a
  half-file that a later pass mistakes for complete.
