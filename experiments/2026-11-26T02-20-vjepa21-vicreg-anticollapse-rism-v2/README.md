# Experiment `2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2` — V-JEPA 2.1 + VICReg Anti-Collapse + KL-Router RISM SSL Pretraining & `suite31_hw` Transfer Probe

- **Status:** `QUEUED` (DAG successor to `2026-11-26T00-54-vjepa21-gram-rism-v2`)
- **Platform:** Google Colab `L4` (24 GB VRAM, native BF16) via `experiments/overnight_queue/queue.json`
- **Parent Experiment:** [`2026-11-26T00-54-vjepa21-gram-rism-v2`](../2026-11-26T00-54-vjepa21-gram-rism-v2/OUTCOME.md) (`ITERATE`)

## Summary & Hypothesis

Exp 2 (`2026-11-26T00-54-vjepa21-gram-rism-v2`) completed `1,500` steps cleanly on Modal `L4`, but CPU SVD scrutiny uncovered **severe representation rank collapse** in `p3` (`erank = 2.1 / 384`) and `mem_final` (`erank = 2.6 / 384`, `cross_page_cos = 0.9993`) at `best.pt` (step 385), driven by three root causes:
1. Random-init DINOv3 Gram anchoring (`within_cos = 0.958` at step 0) actively rewarded collapsed Gram matrices (`gram = 4e-5`).
2. Zero explicit variance/covariance anti-collapse regularization on `p3`/post-merger and `mem_final` tokens.
3. Router BCE consistency against soft `p ≈ 0.5` teacher targets created a `ln(2) ≈ 0.693` constant floor (~80% of total loss), masking JEPA collapse and causing `best.pt` to save at peak collapse (step 385).

**Hypothesis:** Running `pretrain_ijepa_rism_v2.py` on RISM 5,707 with `--vicreg-var-weight 1.0 --vicreg-cov-weight 0.04 --router-loss-mode kl --disable-random-gram` will:
1. Keep encoder effective rank healthy throughout pretraining (`post_erank >= 15`, `mem_erank >= 25`, `cross_page_cos < 0.985`), and
2. Beat scratch initialization on downstream 800-step `UniOMR v2 small_45m` `suite31_hw` `hw_clean` / `hw_photo` Strict Note F1 without printed regression.

## Quick Links

- Plan & decision tree: [PLAN.md](PLAN.md)
- Colab runner: [run_colab.py](run_colab.py)
- Outcome: `OUTCOME.md` (pending)
- HF artifacts (`gregoreesmaa/sheet-music-converter-omr`, private dataset):
  - `evals/2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2.json`
  - `runs/2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2.json`
  - `checkpoints/2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2/uni_omr_v2_ijepa_last.pt`
