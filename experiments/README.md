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
- **Zero-Prior-Context Handoff:** Every experiment folder includes its own launcher script, live query IDs, and copy-paste monitor/resume/eval commands.
- **Git checkpointing:** Commit and push `README.md` + `PLAN.md` *before* launching, and commit + push `OUTCOME.md` immediately *after* evaluation completes.
- **No secrets or large weights in git:** Store checkpoints/shards on Hugging Face (`gregoreesmaa/sheet-music-converter-omr`) or `D:\tmp\`, never inside `experiments/`.

---

## Active & Queued Experiments

| # | Experiment / Slug | Platform | Wave | Status |
|---|---|---|---|---|
| 0 | [2026-11-26T00-26-iter10-5-v1-redteam-fixes](./2026-11-26T00-26-iter10-5-v1-redteam-fixes/OUTCOME.md) | Local RTX 4080 SUPER | Active | `COMPLETED (PROMOTE)` — new record `all45_clean=0.6984` (`0.7004` clean_gt), `all45_photo=0.6832`, `hw_photo=0.2174`, printed fixes `12.0`; promoted as Iter 11 `v1` teacher |
| 1 | `cut-rism-real-ink-v06` (`train_cut_rism.py` + translate `hw-sim-v06`) | Kaggle `2× T4` (FP16) | Wave 1 | `QUEUED` (waiting on HF shards) |
| 2 | [2026-11-26T00-54-vjepa21-gram-rism-v2](./2026-11-26T00-54-vjepa21-gram-rism-v2/OUTCOME.md) (`pretrain_ijepa_rism_v2.py`, 1.5k steps) | Modal `L4` (BF16) | Wave 1 | `COMPLETED (ITERATE)` — rank collapse (`mem_erank=2.6`); DO NOT warm-start Iter 11 |
| 2b | [2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2](./2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2/) (`VICReg + KL-router + no-random-gram` + `suite31_hw` transfer) | Colab `L4` (BF16) | Wave 7 | `QUEUED` (successor to Exp 2) |
| 3 | [2026-11-26T00-54-ablation-vir-vico-router](./2026-11-26T00-54-ablation-vir-vico-router/) (`ViRTwoTierRouter+ViCO` vs `InkTokenRouterV2`) | Modal `2× L4` (BF16) | Wave 1 | `COMPLETED` (`ITERATE` — Keep `InkTokenRouterV2` for Iter 11; successor [`2026-11-26T02-18-ablation-vir-router-bucket-parity`](./2026-11-26T02-18-ablation-vir-router-bucket-parity/) queued) |
| 4 | `ablation-muon-vs-adamw-v2` (`Muon+AdamW` vs `AdamW` on `v2`) | Modal `L4` (BF16) | Wave 1 | `QUEUED` (waiting on HF shards) |
| 5 | `ablation-unlabeled-rism-rlvr` (`rlvr_grpo.py` on Iter-10 `soup`) | Modal `L4` (BF16) | Wave 1 | `QUEUED` (waiting on HF shards) |
| 6 | `trial-scribe-ttt-suite31-hw` (`trial_2028_ttt_adapt.py` on 14 HW pages) | Modal `L4` (BF16) | Wave 2 | `QUEUED` |
| 7 | `trial-render-diff-rlvr` (`trial_2027_render_diff_rlvr.py`) | Modal `L4` (BF16) | Wave 2 | `QUEUED` |
| 8 | `ablation-spinquant-int4-parity` (`spinquant_fold.py` INT4 `suite31_hw` parity) | Modal `L4` / Kaggle `T4` | Wave 2 | `QUEUED` |
| 9 | `ablation-curriculum-fullpage100` (pure fullpage `1.0` vs mixed `0.80`) | Kaggle `2× T4` (FP16) | Wave 2 | `QUEUED` |
| 10 | `ablation-eagle3-speculative-speed` (`eagle3_drafter.py` vs `MTP-4`) | Modal `L4` (BF16) | Wave 3 | `QUEUED` (after `v2` ckpt) |
| 11 | `rism-multiview-pseudolabels` (`generate_rism_pseudolabels.py` on 5,707 RISM) | Kaggle `2× T4` (FP16) | Wave 3 | `QUEUED` (after Iter 10.5) |
| 12 | `ablation-dinov3-dense-distill-v2` (offline dense teacher → `v2` 5M backbone) | Modal `L4` (BF16) | Wave 3 | `QUEUED` |
