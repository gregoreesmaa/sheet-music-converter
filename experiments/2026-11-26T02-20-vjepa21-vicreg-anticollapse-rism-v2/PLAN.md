# PLAN — `2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2`

DAG follow-up to [`2026-11-26T00-54-vjepa21-gram-rism-v2`](../2026-11-26T00-54-vjepa21-gram-rism-v2/OUTCOME.md) targeting the diagnosed representation rank collapse (`p3_erank=2.1`, `mem_erank=2.6`).

## 1. Platform & Budget
- **Provider:** Google Colab `L4` (`24 GB` VRAM, native BF16) dispatched via `experiments/overnight_queue/queue.json` (`depends_on: ["2026-11-26T00-54-vjepa21-gram-rism-v2"]`).
- **Estimated Wall-Clock & Compute:** `~45 min` (`~3.6 CU`):
  - Stage 1: `600`-step RISM 5,707 V-JEPA 2.1 + VICReg + KL-router + no-random-gram SSL pretraining (`~18 min`).
  - Stage 2: SVD effective-rank health audit on 6 RISM pages (`< 10 s`).
  - Stage 3: Matched `800`-step `UniOMR v2 small_45m` supervised transfer probe on `datasets/iter10_5_proxy_2k.tar` (warm-start vs scratch) + 45-piece `suite31_hw` evaluation (`~26 min`).

## 2. Exact SSL Hyperparameters (Anti-Collapse Fix)
```bash
python -u tmp-train/mvd/pretrain_ijepa_rism_v2.py \
  --preset small_45m \
  --image-dir /content/vol/datasets/rism_hw_5707 \
  --out-dir /content/vol/ckpts/exp2b_vicreg/ssl \
  --steps 600 \
  --batch-size 4 --grad-accum 2 --max-dim 1024 \
  --bf16 --optimizer muon \
  --gram-weight 1.0 --disable-random-gram \
  --stroke-hog-weight 0.25 \
  --router-cons-weight 0.2 --router-loss-mode kl \
  --vicreg-var-weight 1.0 --vicreg-cov-weight 0.04 \
  --save-every-minutes 5.0 --log-every 25 --seed 42
```

## 3. Checkpointing & HF Artifacts
- 5-minute atomic rolling autosave (`--save-every-minutes 5.0`) on both SSL and supervised arms.
- Uploads `checkpoints/2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2/uni_omr_v2_ijepa_last.pt`, `vjepa21_train_log.json`, `evals/2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2.json`, and `runs/2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2.json` to `gregoreesmaa/sheet-music-converter-omr`.

## 4. Evaluation Gate & Decision Tree
- **Gate 1 (Anti-Collapse Health):** `post_erank >= 15.0` AND `mem_erank >= 25.0` AND `p3_erank >= 10.0` (vs Exp 2's collapsed `2.1 / 2.6`).
- **Gate 2 (Downstream `suite31_hw` Transfer):**
  - **PROMOTE:** Gate 1 passes AND warm-start beats scratch on `hw_clean`/`hw_photo` Strict Note F1 by `>= +0.015` (or Token F1 `>= +0.015`) with `<= 0.005` printed regression → scale to 2,000 steps and use as standard `UniOMR v2` warm-start.
  - **ITERATE:** Gate 1 passes (`erank` healthy) but transfer delta `< +0.015` → run longer SSL (`2000` steps) or initialize SSL from a supervised printed checkpoint with Gram anchoring enabled.
  - **DISCARD:** Collapse persists despite VICReg or warm-start regresses both printed and HW → drop scratch SSL and rely on supervised printed pretraining + RISM pseudo-labeling / RLVR.
