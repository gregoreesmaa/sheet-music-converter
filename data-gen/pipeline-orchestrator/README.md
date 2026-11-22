# pipeline-orchestrator

Deterministic, seeded, resumable driver for the synthetic sheet-music data
pipeline:

```
render -> vary -> degrade -> pack
```

- **Config-driven**: one `config.toml` controls seeds, counts, and per-stage
  parameters.
- **Seeded**: every output byte derives from `(seed, stage, unit key)` via an
  independent SplitMix64 stream, so the same config always yields bit-identical
  artifacts — regardless of resume, `--from`, or `--only` slicing.
- **Resumable**: each finished work unit is appended to a state file
  (`<out_dir>/.pipeline-state`). Rerun with `--resume` to skip completed units.
- **Offline build**: std-only, zero crates, so render-farm workers compile
  without crates.io access.

## Quickstart

```sh
cd data-gen/pipeline-orchestrator
cp config.example.toml config.toml   # edit seed / count as needed
cargo run --release -- --config config.toml
```

Resume after an interruption (or after `--fail-after`, or a killed process):

```sh
cargo run --release -- --config config.toml --resume
```

Preview without writing anything:

```sh
cargo run --release -- --config config.toml --dry-run
```

## CLI

| Flag | Effect |
|---|---|
| `-c, --config <PATH>` | Config file (default `config.toml`) |
| `--resume` | Skip units already recorded in the state file |
| `--from <STAGE>` | Start at `render\|vary\|degrade\|pack`, skipping earlier stages |
| `--only <STAGE>` | Run just one stage (its inputs must exist from a prior run) |
| `--dry-run` | Print the unit plan, write nothing |
| `--seed <U64>` | Override `[global].seed` for this run |
| `--fail-after <N>` | Exit(2) after N units — test aid for `--resume` |
| `-h, --help` | Print help |

Exit codes: `0` ok · `1` config/stage error · `2` `--fail-after` trip.

## Config reference (`config.example.toml`)

| Key | Default | Meaning |
|---|---|---|
| `[global].seed` | `1` | Master seed; `--seed` overrides per run |
| `[global].count` | `16` | Base scores rendered (1–100000) |
| `[global].out_dir` | `"out"` | Stage outputs; relative to the config file dir |
| `[global].state_file` | `<out>/.pipeline-state` | Completion log |
| `[render].enabled/backend/width/height` | true/`"placeholder"`/2480/3508 | Render toggle and page geometry |
| `[vary].enabled/variants_per_sample` | true/`3` | Engraving variants per sample (1–64) |
| `[degrade].enabled/level` | true/`0.4` | Artifact density in 0.0–1.0 |
| `[degrade].degrades_per_variant` | `1` | Degrade variants per vary output, 1–64 (render-once-degrade-many; render cost per training page drops ~N×) |
| `[pack].enabled/format` | true/`"manifest"` | Manifest/summary writer |

Total work units ≈ `count × (1 + V + V × D) + 1` (pack), where
`V = [vary].variants_per_sample` and `D = [degrade].degrades_per_variant`.

The plan runs sample-major (render sample 0, its varies, its degrades, then
sample 1, ...), not stage-serial: the first degrade finishes long before the
last render, so render and degrade overlap in wall-clock pipeline terms. Each
run prints per-stage wall totals (`timing ms: render=… vary=… degrade=…
pack=…`); render cost per training page is `render ÷ degrade pages` and
drops ~D× under fan-out. Changing `D` between runs only adds new degrade
keys — fresh runs truncate the state file, and `--resume` reruns just the
new keys.

## Layout of `out_dir`

```
out/
  render/r0000.page ...      # base pages (pseudo-engraving note lists)
  vary/r0000v0.page ...      # transposed / respaced variants
  degrade/r0000v0d.page ...  # blots + coordinate jitter (D=1 legacy keys;
                             # D>1 gives r0000v0d0.jpg ... r0000v0d{N-1}.jpg)
  pack/manifest.txt          # FNV-1a hash + relative path per artifact
  pack/summary.txt           # seed, count, artifact total, manifest hash
  .pipeline-state            # append-only completion log (v1 + seed= + rows)
```

## Resuming

State rows look like `render\tr0003`. A fresh run (no `--resume`) truncates
the state file; `--resume` keeps it and skips recorded units. Resume is safe
because:

1. Artifacts are written to `*.tmp` then renamed — a killed run never leaves
   a half-written file that resume would trust.
2. Each unit's RNG stream depends only on `(seed, stage, key)`, never on
   run order or history.
3. `--resume` refuses a state file whose `seed=` line differs from this run.

Demo the crash path: `--fail-after 5`, then rerun with `--resume` and confirm
the second run reports skipped units and identical hashes.

## Hand-ink stage (external)

Clean render PNGs can be turned into handwritten-ink variants between
`render` and `vary` with the `data-gen/hand-ink` binary (deterministic Rust
port of `tmp-train/hw-synth/v3` level `v3_high`; seeded smooth wobble ≤ 5 px,
ink jitter, paper grain; topology-preserving so labels still apply):

```sh
cargo run --release --manifest-path ../hand-ink/Cargo.toml -- \
  --input out/render --out-dir out/hand-ink --seed <SEED> --level high
```

Full pipeline order with hand-ink: `render -> hand-ink -> vary -> degrade ->
pack`. The orchestrator still sequences only its own placeholder stage
bodies (`render/vary/degrade/pack`); feed its `vary` stage from `hand-ink`
outputs, or run hand-ink standalone over `score-corpus` renders. See
`../hand-ink/README.md` for flags, exit codes, and the regenerable sample.

## Determinism check

```sh
cargo run -q -- --config config.toml
sha256sum out/pack/manifest.txt   # note H1
rm -rf out
cargo run -q -- --config config.toml
sha256sum out/pack/manifest.txt   # must equal H1
```

## Swapping in real backends

Each `run_<stage>` function in [src/main.rs](src/main.rs) is the seam: keep
its signature, the `(seed, stage, key)` → `stream_for(...)` derivation, the
`write_artifact` (tmp+rename) helper, and the `mark_done` call; replace only
the payload generation (e.g. shell out to LilyPond/Verovio, run an
augmentation script, call ImageMagick, write a real tarball in `pack`).
Downstream stages read the previous stage's files, so keep the
`<stage>/<key>.page` naming or update all readers together.
