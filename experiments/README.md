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
| 4 | [2026-11-26T00-54-ablation-muon-vs-adamw](./2026-11-26T00-54-ablation-muon-vs-adamw/OUTCOME.md) (`Muon+AdamW` vs `AdamW` on `v2`, 1.2k steps) | Modal `L4` (BF16) | Wave 1 | `COMPLETED (ITERATE)` — `AdamW` RAW wins `val_ce=1.3769` (`-0.0251`), `token_f1=0.3609` (`+0.0294`), `sys_exact=0.375`; `Muon` wins `val_loss=2.0908` (`-0.0215`), `raw_ser=0.8738` (`-0.0044`), `-128 MB` VRAM, `+7.15%` time; keep Iter 11 running; successor `exp4b` queued |
| 4b | [2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep](./2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep/) (decoupled `--muon-lr ∈ {6e-4, 1e-3, 1.5e-3}` + AdamW `3e-4`, un-lagged EMA `0.99/1`) | Colab `L4` (BF16) | Wave 4 | `QUEUED` (successor to Exp 4) |
| 5 | `ablation-unlabeled-rism-rlvr` (`rlvr_grpo.py` on Iter-10 `soup`) | Modal `L4` (BF16) | Wave 1 | `QUEUED` (waiting on HF shards) |
| 6 | `trial-scribe-ttt-suite31-hw` (`trial_2028_ttt_adapt.py` on 14 HW pages) | Modal `L4` (BF16) | Wave 2 | `QUEUED` |
| 7 | `trial-render-diff-rlvr` (`trial_2027_render_diff_rlvr.py`) | Modal `L4` (BF16) | Wave 2 | `QUEUED` |
| 8 | `ablation-spinquant-int4-parity` (`spinquant_fold.py` INT4 `suite31_hw` parity) | Modal `L4` / Kaggle `T4` | Wave 2 | `QUEUED` |
| 9 | `ablation-curriculum-fullpage100` (pure fullpage `1.0` vs mixed `0.80`) | Kaggle `2× T4` (FP16) | Wave 2 | `QUEUED` |
| 10 | `ablation-eagle3-speculative-speed` (`eagle3_drafter.py` vs `MTP-4`) | Modal `L4` (BF16) | Wave 3 | `QUEUED` (after `v2` ckpt) |
| 11 | `rism-multiview-pseudolabels` (`generate_rism_pseudolabels.py` on 5,707 RISM) | Kaggle `2× T4` (FP16) | Wave 3 | `QUEUED` (after Iter 10.5) |
| 12 | `ablation-dinov3-dense-distill-v2` (offline dense teacher → `v2` 5M backbone) | Modal `L4` (BF16) | Wave 3 | `QUEUED` |
| 35 | [2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep](./2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep/OUTCOME.md) (`max_dim ∈ {1280,1536,1792}` × `max_len ∈ {3840,5120,6144}`) | Colab `L4` (BF16) | Wave 4 | `COMPLETED (ITERATE)` — `1280px/3840` wins `all45_clean=0.7069` (`+0.0126`), `hw_clean=0.2799` (`+0.0181`), `printed_clean=0.8997` (`+0.0102`); `max_len>3840` strictly harmful; successor `ov-35b` queued |
| 35b | [2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep](./2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep/) (`max_dim ∈ {1024,1152,1280,1408}` × `max_len ∈ {2560,3072,3584,3840}`) | Colab `L4` (BF16) | Wave 4 | `QUEUED` (successor to Exp 35) |
| 36 | [2026-11-26T01-38-ov-36-hw-multiview-tta-voting](./2026-11-26T01-38-ov-36-hw-multiview-tta-voting/OUTCOME.md) (6-view HW TTA + FSA/MBR voting) | Colab `L4` (BF16) | Wave 4 | `COMPLETED (ITERATE)` — Oracle `+0.0419` (`hw_clean 0.3030`) / `+0.0327` (`hw_photo 0.2243`); `clahe`/`deskew` poisoned 6-view rerankers; offline scale-only `fsa_beat` hits **`0.2954` (`+0.0343`)** clean / **`0.1969` (`+0.0053`)** photo; successor `ov-36b` queued |
| 36b | [2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta](./2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta/) (clean 5-scale `{1024..1536}` + `mean_logprob` + FSA TTA at `max_len=3840`) | Colab `L4` (BF16) | Wave 4 | `QUEUED` (successor to Exp 36) |
