# PLAN — `2026-11-26T07-40-i13-06-memo-sar-rhythm-gated-ttt` (`i13-06`)

## 1. Platform & Budget
- **Provider:** Google Colab `L4` (`24 GB` VRAM, `cc=8.9`, native BF16; FP32 fallback on T4).
- **Estimated Wall-Clock & Compute:** `~26 min` (`~2.2 CU`):
  - All `14` real historical HW manuscripts in `suite31_hw` (`clean` + `photo` = `28` pages) × `4` TTT arms + baseline at canonical `max_dim=1280`, `max_len=3840`, MTP-4 + FSA grammar.
  - Top-2 HW arms automatically verified on the first `10` printed pieces (`20` pages) as a printed safety gate.

## 2. Single-Variable Protocol
- **Controlled Constants:** Checkpoint `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`, adapted scope `full_encoder` (`backbone` + `router` + `enc_layers` + `enc_norm`), `lr=3e-5`, `steps=2`, `l2_weight=0.1`, `cons_weight=1.0`, bit-exact (`atol=0`) CPU weight reset before every page/arm, canonical `max_dim=1280, max_len=3840`.
- **Manipulated Variable:** Test-time adaptation loss & update geometry:
  - `arm_ctrl_ov38`: Single-view `ent=1.0, kl=0.5` (exact `ov-38` winner).
  - `arm_memo_marginal`: Marginal distribution $\bar{p} = \frac{1}{2}(p_{\text{clean}} + p_{\text{aug}})$ entropy $H(\bar{p}) + 0.5 \cdot \text{JSD}(p_{\text{clean}}, p_{\text{aug}}) + 0.5 \cdot \text{KL}$.
  - `arm_sar_sam`: Sharpness-aware (SAM, $\rho=0.01$) 1-step ascent-descent on `full_encoder` with `ent=0.5, kl=0.5`.
  - `arm_rhythm_gated`: Page-level structural reliability gate (`mean_logprob >= -0.35`, `beat_valid_frac >= 0.70`, `r_noloop == 1.0`) toggling `ent=0.5, kl=0.5` vs pure visual consistency (`ent=0.0, kl=0.0`).

## 3. HF Identifiers & Checkpointing
- **HF Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
- **Eval JSON:** `evals/2026-11-26T07-40-i13-06-memo-sar-rhythm-gated-ttt.json`
- **Run Ledger:** `runs/2026-11-26T07-40-i13-06-memo-sar-rhythm-gated-ttt.json`
- Atomic per-page partial JSON checkpointing; `/tmp/.hf_token` purged immediately on read and in `finally`.

## 4. Evaluation Gate & Decision Tree
- **PROMOTE:** Any arm achieves `delta_note_f1_py >= +0.020` on `hw_clean` OR `hw_photo` at canonical `max_len=3840` (and beats `arm_ctrl_ov38` by `>= +0.005` avg), with `wins > losses`, no regression `> -0.005` on the other HW domain, AND `delta >= -0.005` on both `printed_clean` and `printed_photo` → wire directly into `trial_2028_ttt_adapt.py` as default inference TTT.
- **ITERATE:** `arm_rhythm_gated` eliminates hard-manuscript losses (`losses <= 2/14`) and `arm_sar_sam` or `arm_memo_marginal` raises peak gain → combine `rhythm_gated + MEMO/SAR` at `steps=3..4`.
- **DISCARD:** None beats `arm_ctrl_ov38` by `>= +0.003` → stick with `ov-38b`'s winner.
