# pipeline-orchestrator (converter)

Deterministic, seeded, resumable driver for the sheet-music conversion
pipeline:

```
ingest -> layout -> symbols -> serialize
```

- **Config-driven**: one `config.toml` selects the input pages, seeds, and
  per-stage parameters.
- **Seeded**: every layout box and symbol derives from `(seed, stage, page)`
  via an independent SplitMix64 stream, so the same config + inputs always
  yield bit-identical outputs — regardless of resume, `--from`, or `--only`
  slicing.
- **Resumable**: each finished work unit is appended to a state file
  (`<out_dir>/.pipeline-state`). Rerun with `--resume` to skip completed
  units.
- **Offline build**: std-only, zero crates, so workers compile without
  crates.io access.

The `layout` and `symbols` stages are deterministic placeholders behind a
stable `(seed, page)` contract: swap in a real staff detector / recognizer
without changing the CLI, config schema, state format, or on-disk layout.

## Quickstart

```sh
cd converter/pipeline-orchestrator
cp config.example.toml config.toml   # point [ingest].input_dir at page images
cargo run --release -- --config config.toml
```

Preview without writing anything:

```sh
cargo run --release -- --config config.toml --dry-run
```

Resume after an interruption (or after `--fail-after`, or a killed process):

```sh
cargo run --release -- --config config.toml --resume
```

## CLI

| Flag | Effect |
|---|---|
| `-c, --config <PATH>` | Config file (default `config.toml`) |
| `--resume` | Skip units already recorded in the state file |
| `--from <STAGE>` | Start at `ingest\|layout\|symbols\|serialize`, skipping earlier stages |
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
| `[global].out_dir` | `out` | Root for stage outputs; paths relative to the config file |
| `[global].state_file` | `<out>/.pipeline-state` | Completion records (`v1 <stage> <unit>` per line) |
| `[ingest].input_dir` | `samples` | Page images; each matching file is one unit (sorted) |
| `[ingest].extensions` | `png,jpg,jpeg,tif,tiff,bmp` | Comma-separated file extensions |
| `[layout].systems_per_page` | `4` | Staff systems per page (1–32) |
| `[layout].page_width/height` | `2480/3508` | Reference page size boxes are computed against |
| `[symbols].symbols_per_system` | `24` | Symbols emitted per system (1–512, pre-cutoff) |
| `[symbols].min_confidence` | `0.5` | Symbols below this confidence are dropped |
| `[serialize].format` | `json` | Final score format: `json` or `musicxml` |

Set `enabled = false` under any stage to skip it. Downstream stages read
the skipped stage's files from disk, so skipping requires a prior run that
produced them.

## Outputs

```
<out>/
  ingest/<page>.json    # source path, byte size, content id, seed
  layout/<page>.json    # staff-system boxes {id,x,y,w,h}
  symbols/<page>.json   # symbol stream {id,system,kind,pitch,x,confidence}
  score/<page>.json     # final score (or .musicxml with format="musicxml")
  manifest.json         # seed, stages run, page list
  .pipeline-state       # completion records for --resume
```

## Replacing the placeholders

- `derive_systems(cfg, page)` is the canonical `(seed, page)` → boxes
  function. A real detector must reproduce its outputs for supported seeds
  or version the state file.
- `run_symbols` consumes `&[SystemBox]` and emits the same `Symbol` shape;
  keep the per-`(seed, system, index)` stream when swapping recognizers.
