# Experiments

Structured, reproducible experiment ledger for `sheet-music-converter`.
Every experiment follows the [running-an-experiment](../.agents/skills/running-an-experiment/SKILL.md) skill.

## Directory Convention

```text
experiments/
├── README.md                              # This index
└── YYYY-MM-DDTHH-MM-<short-slug>/         # One folder per isolated experiment (minute precision)
    ├── README.md                          # Goal, hypothesis, baseline SHA, status
    ├── PLAN.md                            # Platform, query IDs, data/HF shards, 5-min ckpt, next actions
    └── OUTCOME.md                         # Measured results, suite31_hw delta, verdict (written on completion)
```

## Standing Rules
- **Minute-precision timestamp prefix:** `YYYY-MM-DDTHH-MM-<slug>` (e.g., `2026-11-26T00-39-vjepa21-rism-v2`).
- **5-minute checkpoints mandatory:** Every training/adaptation run uses `--save-every-minutes 5.0` (atomic `.tmp` → rename + batched HF Hub sync to `gregoreesmaa/sheet-music-converter-omr`).
- **Git checkpointing:** Commit and push `README.md` + `PLAN.md` *before* launching, and commit + push `OUTCOME.md` immediately *after* evaluation completes.
- **No secrets or large weights in git:** Store checkpoints/shards on Hugging Face (`gregoreesmaa/sheet-music-converter-omr`) or `D:\tmp\`, never inside `experiments/`.
