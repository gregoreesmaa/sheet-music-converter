# Experiment `2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch` — Transfer Probe: V-JEPA 2.1 RISM-Pretrained Encoder vs Random Init (`iter10_5_proxy_2k` + `suite31_hw`)

- **Status:** `PLANNED` (queued in `experiments/overnight_queue/batch_07.json`, wave 7)
- **Platform:** Google Colab `L4` (`24 GB` VRAM, native BF16, ~35–45 min total, ~3.6 CU)
- **Baseline Commit SHA:** `164852f`

## Summary & Hypothesis

Controlled 2-arm supervised transfer probe comparing **`UniOMR v2 small_45m`** trained for `800` steps on `datasets/iter10_5_proxy_2k.tar` (`fullpage_max_dim=1280`, `batch_size=2`, `grad_accum=4`, `BF16`, `Muon+AdamW`, `WSD`, `seed=42`) from:
- **Arm A (`arm_a_scratch`):** Fresh random initialization.
- **Arm B (`arm_b_vjepa_warmstart`):** Vision encoder (`backbone` + `encoder`) warm-started via `--init-encoder-from` using our RISM-pretrained `V-JEPA 2.1 + DINOv3 Gram + Stroke-HOG` checkpoint (automatically fetched from HF Exp 2 `checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/uni_omr_v2_ijepa_{best,last}.pt` if present, otherwise generated on-the-fly via a 400-step RISM warmup on `datasets/rism_hw_5707.tar`).

Both arms are evaluated head-to-head on `datasets/suite31_hw_eval.tar` (45 pieces: 31 printed + 14 real historical handwritten manuscripts, `clean` + `photo` = 90 pages) using `tmp-train/mvd/eval_uni_suite31.py`.

**Hypothesis:** Unlabeled RISM V-JEPA 2.1 pretraining transfers iron-gall ink stroke, staff-line geometry, and parchment-invariance priors directly into the `UniOMR v2` encoder, yielding faster supervised convergence and a measurable `>= +0.015` Strict Note F1 / Token F1 gain on `suite31_hw` real handwritten splits (`hw_clean`, `hw_photo`) without regressing printed splits.

## Quick Links

- Plan, launch commands & decision tree: [PLAN.md](PLAN.md)
- Standalone Colab L4 runner: [run_colab.py](run_colab.py)
- Outcome (written after completion): `OUTCOME.md` (pending)
- Overnight queue entry: `experiments/overnight_queue/batch_07.json`
- Hugging Face Hub (`gregoreesmaa/sheet-music-converter-omr`, private dataset):
  - Eval JSON: `evals/2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch.json`
  - Run ledger: `runs/2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch.json`
