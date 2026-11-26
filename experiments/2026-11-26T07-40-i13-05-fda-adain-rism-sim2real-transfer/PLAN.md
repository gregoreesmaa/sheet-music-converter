# PLAN — `2026-11-26T07-40-i13-05-fda-adain-rism-sim2real-transfer` (`i13-05`)

## 1. Platform & Budget
- **Provider:** Google Colab `L4` (`24 GB` VRAM, `cc=8.9`, native BF16; FP32 fallback on T4).
- **Estimated Wall-Clock & Compute:** `~34 min` (`~2.8 CU`):
  - Baseline `uni_omr_soup.pt` eval + `4` fine-tuning arms (`120` optimizer steps × `grad_accum=4` = `480` page draws each from `iter10_5_proxy_2k`, `lr=1.5e-5` cosine → `3e-6`, `label_smoothing=0.02`) evaluated on `suite31_hw` (`14` HW + `12` stratified printed = `52` pages, `max_dim=1024`, `eval_max_len=1536`, MTP-4 + FSA grammar).

## 2. Single-Variable Protocol
- **Controlled Constants:** Starting checkpoint `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`, exact same `480` labeled `iter10_5_proxy_2k` training page sequence (`seed=42`), exact same AdamW optimizer (`lr=1.5e-5`, `wd=0.01`), exact same `52`-page `suite31_hw` eval set.
- **Manipulated Variable:** Unpaired RISM-5707 visual style-transfer transform applied with `p=0.5` to each training page:
  - `arm_ctrl_standard`: Contrast/brightness jitter only (`0.88..1.12` / `0.94..1.06`).
  - `arm_fda_b002`: 2D FFT low-frequency amplitude swap (`beta=0.02`, square central radius `max(1, int(min(H,W)*beta))`) from a random real `rism_hw_5707` page.
  - `arm_fda_b006`: Same 2D FFT FDA with `beta=0.06`.
  - `arm_color_adain`: Per-channel mean/std matching to a random real `rism_hw_5707` page blended with `alpha ~ U(0.4, 0.8)`.

## 3. HF Identifiers & Checkpointing
- **HF Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
- **Eval JSON:** `evals/2026-11-26T07-40-i13-05-fda-adain-rism-sim2real-transfer.json`
- **Run Ledger:** `runs/2026-11-26T07-40-i13-05-fda-adain-rism-sim2real-transfer.json`
- Atomic JSON checkpointing after baseline and every completed arm; `/tmp/.hf_token` purged immediately on read and in `finally`.

## 4. Evaluation Gate & Decision Tree
- **PROMOTE:** Any unpaired RISM transfer arm beats `arm_ctrl_standard` by `>= +0.015` on `hw_clean` OR `hw_photo` (and `hw_avg_delta >= +0.010` vs control) with `<= 0.005` drop on `printed_clean`/`printed_photo` → add `--rism-fda-prob` / `--rism-adain-prob` online augmentation directly into `train_uni_v2.py` for Iter 13 SFT.
- **ITERATE:** Gains appear mainly on `hw_photo` (`>= +0.010`) while `hw_clean` is flat → test combining FDA (`beta=0.02`) + Color-AdaIN + `ov-39` RISM pseudo-labels in a joint mix.
- **DISCARD:** All 3 transfer arms fall within `±0.005` of `arm_ctrl_standard` or regress printed `> -0.010` → photometric/spectral transfer is insufficient; prioritize geometric/handwriting stroke synthesis (`hw-score-sim`) and CycleGAN-Turbo/CUT glyph-level transfer.
