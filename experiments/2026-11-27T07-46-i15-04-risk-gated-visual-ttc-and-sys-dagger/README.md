# `2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger`

**Wave 13 · Exp `i15-04` — Risk-Gated Visual-Coverage TTC (Pivot 3) + `<sys>`-Aligned On-Policy DAgger & Counterfactual Anti-Prior (Pivot 2)**

- **Status:** `COMPLETED` — Verdict: **`DISCARD`** (see [OUTCOME.md](OUTCOME.md))
- **Platform:** Google Colab `L4` (`provider: "colab_l4"`, BF16 native, 24 GB VRAM)
- **Estimated Wall-Clock / Budget:** `~60 min` / `~4.5 CU`
- **Baseline Commit SHA:** `e426c28`
- **Baseline Checkpoint (HF):** `checkpoints/iter13_v2_mid/uni_omr_v2_ema_latest.pt` (overridable via `UNIOMR_HF_CKPT`)

## Summary & Hypothesis

This Wave 13 experiment formalizes Pivot Strategist 4's Objective, RLVR & Test-Time Compute (TTC) recommendations into two controlled, self-contained parts that directly fix all 5 forensic defects from [`i14-14`](../2026-11-26T22-18-i14-14-cascade-ttt-mbr-zoom-chamfer-rerank/OUTCOME.md) and the **`bsvr = 0.997` blind spot** (where dropped systems or skipped measures still pass syntactic beat-sum verification):

1. **Part A (`part_a_risk_gated_visual_ttc`, Pivot 3 — Zero-Train Eval on OMR-NorthStar `dev-v1`, 125p):**
   Compares `arm0_greedy` (`1408px`, FSA+MTP greedy) against `arm1_risk_gated_ttc`, which escalates **ONLY** when a page trips a calibrated visual/structural risk flag (`has_loop`, `truncated`, `|n_sys - n_sys_img| >= 1`, `attn_ink_cov < 0.80`, or `bsvr < 0.80`), **ALWAYS keeps `y_0`** in the candidate pool, uses **zero encoder TTT**, and reranks candidates by `r_attn_cov + r_sys_vis + norm_lp`, replacing `y_0` only on a hard CFR flag or a `>= +0.08` composite win margin.
2. **Part B (`part_b_sys_dagger_probe`, Pivot 2 — `1,200`-Step Controlled Probe on `suite31_hw`):**
   Compares a `1,200`-step SFT control (`arm_b0_sft_control`) against a `1,200`-step `<sys>`-aligned On-Policy DAgger + Counterfactual Anti-Prior arm (`arm_b1_sys_dagger_antiprior`: `75%` SFT + `25%` DAgger/Anti-Prior, `ignore_index=-100` on the model-generated prefix, and `3×` loss weight on `<sys>` recovery header tokens) to eliminate exposure-bias compounding across handwritten manuscript systems without hurting printed scores.

## Quick Links

- Plan & Decision Tree: [PLAN.md](PLAN.md)
- Colab L4 Runner: [run_colab.py](run_colab.py)
- Queue Entry Fragment: [queue_entry.json](queue_entry.json)
- Outcome (written upon completion): `OUTCOME.md`
- HF Eval Target: `evals/2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger.json`
- HF Run Ledger: `runs/2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger.json`
