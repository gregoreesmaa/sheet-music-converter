# Experiment Plan — `2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840`

## 1. Platform & Budget
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe`.
- **Estimated Wall-Clock:** `~38–42 min` (`~3.1–3.4 CU`):
  - Baseline eval (`14` HW + `12` stratified Printed at `max_dim=1280, eval_max_len=3840`): `~3 min`
  - 3 RLVR arms × `160` updates (`max_len=512`, `max_dim=1024`): `~24 min`
  - 3 post-arm evals (`26` pieces at `1280px/3840`): `~9 min`
  - Full 45-piece confirmation eval on winner (`90` pages at `1280px/3840`): `~5 min`

## 2. Identifiers
- **Experiment ID:** `2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840`
- **Depends On:** `2026-11-26T01-38-ov-37-rlvr-kl-group-sweep`
- **HF Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840.json`
  - Run Ledger JSON: `runs/2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840.json`
  - Winning Checkpoint (if Promote gate passes): `checkpoints/2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840/best_rlvr_arm.pt`

## 3. Exact Arms & Improvements Over `ov-37`
- **3 Pareto Arms (`updates=160`, `max_len=512`, `lr=1.0e-5`, warmup `16` + cosine to `2.0e-6`):**
  1. `kl0.06_G8_u160`: `ov-37` composite winner (`hw_photo +0.0301`)
  2. `kl0.15_G4_u160`: `ov-37` most balanced & lowest-KL arm (`hw_clean +0.0178`, `hw_photo +0.0130`)
  3. `kl0.10_G8_u160`: midpoint KL with low-variance `G=8` rollouts
- **4 Methodological Fixes from `ov-37` Scrutiny:**
  1. Canonical **`max_dim=1280, eval_max_len=3840`** for baseline + all arms (matching `ov-35` winner; removes `1536`-token truncation ceiling).
  2. Saves **per-piece `note_f1_py` rows** for baseline and every arm so we verify multi-piece win rate (`n_improved / 14`) instead of relying on aggregate mean alone.
  3. Automatic platform detection (`kaggle` vs `colab`) and `FP16 + GradScaler` fallback when `cc[0] < 8` (never slow FP32 on T4).
  4. Atomic 5-minute checkpointing + arm-level resume.

## 4. Decision Tree
- **PROMOTE:** Winning arm achieves `delta_hw_clean >= +0.020` AND `delta_hw_photo >= +0.020` at `1280px/3840` with `>= 8/14` HW pieces improving and `printed_clean` regression `<= 0.010` → wire winning `(kl_beta, G, max_len=512)` as mandatory post-training stage after Iter 11.
- **ITERATE:** If `kl_beta` still drifts or `hw_clean`/`hw_photo` diverge at `160` updates → test mixed labeled+unlabeled RLVR (`unlabeled_frac=0.50`) or visual render-diff verifier (`trial_2027_render_diff_rlvr.py`).
- **DISCARD:** If `ov-37`'s gains vanish at `1280px/3840` (indicating a truncation artifact) → drop pure unlabeled structural RLVR and focus on visual render-diff RLVR + pseudo-labeling (`ov-39`).
