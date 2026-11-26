# PLAN — `2026-11-26T07-40-i13-04-lejepa-sigreg-vs-vicreg-rism-ssl` (`i13-04`)

## 1. Platform & Budget
- **Provider:** Google Colab `L4` (`24 GB` VRAM, `cc=8.9`, native BF16).
- **Estimated Wall-Clock & Compute:** `~42 min` (`~3.4 CU`):
  - Stage 1: `3 × 400`-step RISM `5,707` V-JEPA 2.1 SSL arms (`small_45m`, `max_dim=1024`, `batch_size=4 × grad_accum=2`, `muon`, `--router-loss-mode kl`, `--disable-random-gram`, `stroke_hog=0.25`) (`~22 min`).
  - Stage 2: SVD effective-rank (`p2_erank`, `p3_erank`, `mem_erank`, `cross_page_cos`, Epps-Pulley statistic) on `12` RISM pages (`< 15 s`).
  - Stage 3: Matched `600`-step `UniOMR v2 small_45m` supervised transfer probe on `datasets/iter10_5_proxy_2k.tar` (`arm_vicreg` vs best `arm_sigreg_*`) evaluated on `suite31_hw` (`~19 min`).

## 2. Single-Variable Protocol
- **Controlled Constants:** Identical `UniOMRv2Config.small_45m()`, seed `42`, RISM image order, masking/corruption schedule, `muon` LR `3e-4`, `stroke_hog=0.25`, `--router-loss-mode kl`, `--disable-random-gram`, and identical `train_uni_v2.py` supervised transfer hyperparameters.
- **Manipulated Variable:** Anti-collapse regularizer on `(mem_post_merger, mem_final)`:
  - `arm_vicreg`: `compute_vicreg_loss(var_weight=1.0, cov_weight=0.04)`
  - `arm_sigreg_w05`: `compute_sigreg_loss(weight=0.5, num_slices=64, num_knots=17, t_max=3.0)`
  - `arm_sigreg_w15`: `compute_sigreg_loss(weight=1.5, num_slices=64, num_knots=17, t_max=3.0)`

## 3. HF Identifiers & Checkpointing
- **HF Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
- **Eval JSON:** `evals/2026-11-26T07-40-i13-04-lejepa-sigreg-vs-vicreg-rism-ssl.json`
- **Run Ledger:** `runs/2026-11-26T07-40-i13-04-lejepa-sigreg-vs-vicreg-rism-ssl.json`
- **Winning SSL Ckpt:** `checkpoints/2026-11-26T07-40-i13-04-lejepa-sigreg-vs-vicreg-rism-ssl/best_ssl_encoder.pt`
- Atomic 5-minute rolling autosave on both SSL and supervised arms; `/tmp/.hf_token` purged immediately on read and in `finally`.

## 4. Evaluation Gate & Decision Tree
- **PROMOTE:** Best SIGReg arm passes health gate (`p3_erank >= 12.0`, `mem_erank >= 30.0`, `cross_page_cos <= 0.90`) AND beats `arm_vicreg` on `suite31_hw` `hw_clean`/`hw_photo` Strict Note F1 by `>= +0.010` (or Token F1 `>= +0.015`) with `<= 0.005` printed regression → add `--anticollapse-mode sigreg` to `pretrain_ijepa_rism_v2.py` and scale to `2,000`-step RISM SSL for Iter 13 warm-start.
- **ITERATE:** SIGReg beats VICReg on effective rank (`>= +20%` `p3_erank`/`mem_erank`) but transfer delta `< +0.010` → combine SIGReg + supervised-init DINOv3 Gram anchoring (`--init-from` printed checkpoint).
- **DISCARD:** SIGReg matches or underperforms VICReg on both effective rank and transfer → keep VICReg as the default anti-collapse regularizer.
