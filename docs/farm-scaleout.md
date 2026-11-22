# Farm scale-out: shard / seed allocation

How to parallelize `render` + `photo-degrade` across farm machines with
**bit-identical** results to a single-machine run. Companion docs:
[the farm runbook](farm-runbook.md) (operator commands) and
[the dataset warehouse](dataset-warehouse.md) (store once, never regenerate).

## Why sharding is exact

Every output byte derives from a `(seed, stage, unit key)` contract, and no
stage reads another unit's output:

| Stage | Unit key | Seed derivation | Shard rule |
|---|---|---|---|
| `score-render` | input score file stem | per-score deterministic export (MuseScore headless export of that score; stub likewise) — no cross-file seed | partition the input score set arbitrarily |
| `pipeline-orchestrator` stages | `r{N}`, `r{N}v{j}`, … | `stream_for(master, stage, key)` in `data-gen/pipeline-orchestrator/src/main.rs` — key-based, order-independent | slice by key range freely |
| `photo-degrade` | **position `i` in the sorted input listing** | `file_seed = seed + i * 0x9E3779B97F4A7C15` (wrapping `u64`; see `run()` in `data-gen/photo-degrade/src/main.rs`, inputs from sorted `collect_inputs`) | shard = **contiguous filename chunk** `[start, end)` of the global sorted list, run with `--seed shard_seed(start)` (formula below) |

The photo-degrade row is the only subtle one: a naive "split files into
per-machine dirs and reuse `--seed S`" reindexes `i` inside each shard and
silently changes every seed. The fix needs no code changes — pass a
per-shard seed that restores the global index:

```text
shard_seed(start) = (master_seed + start * 0x9E3779B97F4A7C15) mod 2^64
```

because local index `j` then yields
`shard_seed + j*stride = master_seed + (start+j)*stride`, the global seed.
Requirements: shards are contiguous chunks of the **globally sorted**
filename list, and filenames are identical in the shard dir and the full
set (local sorted order then equals global order on that chunk). Outputs
are keyed by input stem (`{stem}{suffix}.jpg` + `{stem}{suffix}_params.json`,
default suffix `_degraded`), so merging shards is a plain file copy with no
clobbering — run each shard into its own `--out-dir`.

Example for `master_seed = 20261022`, 100 files, 4 shards:

| Shard | Global indices | `--seed` |
|---|---|---|
| 0 | 0–24 | 20261022 |
| 1 | 25–49 | 8316709377456948907 |
| 2 | 50–74 | 16633418754893636792 |
| 3 | 75–99 | 6503384058620773061 |

(PowerShell: `$s = ([bigint]20261022 + [bigint]$start * [bigint]::Parse('09E3779B97F4A7C15','AllowHexSpecifier')) % [bigint]::Pow(2,64)`.)

A useful special case: any **prefix** of the sorted input run with the
master seed reproduces the first `n` baseline outputs exactly
(`shard_seed(0) = master_seed`). Use prefixes for cheap scaling probes.

`--resume` composes with sharding: the orchestrator state file
(`<out>/.pipeline-state`) is append-only `v1` / `seed=` / `stage\tkey`
rows and refuses a seed mismatch, so per-shard state files merge by
concatenation (keep one `pack` row). `photo-degrade` has no state file —
re-running a shard dir simply overwrites identical bytes.

## 100-page pilot (2026-10-25, this machine)

Setup: `tmp-train/farm-pilot/input100` = 100 full-size pages
(`tmp-train/bench10/biginput` × 10, stems `p000`–`p099`),
release `photo-degrade`, `--format jpeg`, `--seed 20261022`, CPU backend.
Machine: Windows 11, Ryzen 7 7800X3D (8C/16T), RTX 4080 SUPER.

| Run | Wall | s/page |
|---|---|---|
| full 100 (baseline) | 38 023.0 ms | 0.380 |
| prefix 10 | 4 639.8 ms | 0.464 |
| prefix 25 | 10 577.2 ms | 0.423 |
| prefix 50 | 19 806.2 ms | 0.396 |
| 4 × 25 shards, sequential | 10.6 / 11.0 / 15.1 / 17.4 s | — |
| 2 × 25 shards, concurrent processes | 17.8 s wall (vs 28.4 s sequential) | core-bound locally |

Linearity: wall ≈ `0.373·n + ~1 s` startup across 10/25/50/100 —
per-page cost is constant, so farm wall for `n` pages on `M` machines is
`≈ (0.38·n/M + 1)` s. Single-machine throughput: `86400 / 0.38 ≈ 227k`
pages/day; `M` independent machines ⇒ `≈ 227k·M` pages/day (target
~150M/day ⇒ ~660 machines at current per-page cost — this is why the
per-page wins in subtasks 1–5 multiply the farm lever, not replace it).

Exactness (SHA-256 over every output file):

- merged 4 shards vs unsharded baseline: **100/100 bit-identical**;
- prefixes 10/25/50 vs baseline first-n: **exact**;
- concurrent-process shards vs sequential shards: **exact**.

The concurrent run is only ~1.6× faster than sequential here because two
16-thread processes share 16 logical cores — that contention is a
single-box artifact. Farm workers are independent machines, so each shard
runs at full single-machine speed and aggregate throughput scales ~linearly
with `M`; the merge step is O(n) file copies plus a manifest write.

## Scaling limits and guidance

- Shard granularity: keep shards ≥ ~25 full-size pages so the ~1 s process
  startup stays < 10% overhead. Shard count = machine count (or a small
  multiple for straggler tolerance — re-running a shard is idempotent).
- Balance by file count, not bytes: per-page cost is uniform enough
  (shard walls above differ mostly by content mix and local machine noise).
- `--gpu` is per-machine opt-in; grain/noise use GPU hash streams, so GPU
  pixels differ from CPU pixels at the same seed by design (`backend` field
  in params records the path). **Never mix backends inside one dataset
  version** — shard with a uniform backend flag.
- `dataset-pack split` buckets by `sha256("{seed}/{id}")`, i.e. per-id
  deterministic: per-shard splits merge exactly, but simplest is one final
  `pack` over the merged tree (see warehouse doc).
