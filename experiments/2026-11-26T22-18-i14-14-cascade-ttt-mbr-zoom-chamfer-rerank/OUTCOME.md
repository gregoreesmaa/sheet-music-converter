# OUTCOME — `2026-11-26T22-18-i14-14-cascade-ttt-mbr-zoom-chamfer-rerank`

- **Status:** `COMPLETED`
- **Verdict:** **`DISCARD`**
- **Completed At:** `2026-11-27T01:45:57Z` (started `2026-11-27T00:52:37Z`)
- **Platform / GPU:** Kaggle `2× T4` (`Tesla T4`, `cc=7.5`, `native_bf16=False`, FP32 fallback; kernel `gregoreesmaa/omr-i14-14`)
- **Elapsed Time:** `3,265.5 s` (`54.4 min`, script `3,228 s`, exit code `0`)
- **Checkpoint Evaluated:** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (zero-retrain)
- **Dataset Slice:** `52` `suite31_hw` pages (all `14` real historical HW manuscripts = `28` pages + `12` stratified printed pieces = `24` pages, `clean` + `photo`)
- **HF Artifacts:**
  - Eval JSON: `evals/2026-11-26T22-18-i14-14-cascade-ttt-mbr-zoom-chamfer-rerank.json`
  - Run Ledger: `runs/2026-11-26T22-18-i14-14-cascade-ttt-mbr-zoom-chamfer-rerank.json`
  - Local Log: `D:\tmp\overnight-queue-logs\2026-11-26T22-18-i14-14-cascade-ttt-mbr-zoom-chamfer-rerank.kaggle.log`

---

## Measured Results (4-Arm Cascade Ablation, `n_pages = 52`)

| Arm | All Note F1 | Δ All vs Arm 0 | HW Note F1 | Δ HW vs Arm 0 | Printed Note F1 | Δ Printed vs Arm 0 | Escalation Rate | Latency Mult |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`arm0_tier0_fast1x`** (`1408px`, FSA+MTP greedy) | **0.44180** | — | **0.21903** | — | **0.70170** | — | — | `1.000×` |
| `arm1_tier0_plus_tier1_ttt_mbr` (+ 2-step Scribe-TTT + 3-scale MBR `{1152,1280,1536}`) | `0.41937` | `-0.02243` (`-2.24` pp) | `0.20008` | `-0.01895` (`-1.90` pp) | `0.67521` | `-0.02649` (`-2.65` pp) | Tier 1: `96.15%` (`50/52`) | `5.320×` |
| `arm2_full_cascade_verifier_only` (+ Tier 2 `{1664,1792}px` Verifier+LP) | `0.41839` | `-0.02341` (`-2.34` pp) | `0.20272` | `-0.01631` (`-1.63` pp) | `0.67001` | `-0.03169` (`-3.17` pp) | Tier 2: `94.23%` (`49/52`) | — |
| `arm3_full_cascade_chamfer_iou` (+ Tier 2 Chamfer / 3px-dilated Ink-IoU) | `0.41543` | `-0.02637` (`-2.64` pp) | `0.19723` | `-0.02180` (`-2.18` pp) | `0.67001` | `-0.03169` (`-3.17` pp) | Tier 2: `94.23%` (`49/52`) | `7.587×` |

**Gate Evaluation (per [PLAN.md](PLAN.md)):**
- **PROMOTE gate** (`arm3` HW `>= +3.0` pp AND `arm3 > arm2` at `<= 2.2×` latency): **FAILED** (`-2.18` pp HW, `arm3 < arm2` by `-0.55` pp, `7.587×` latency).
- **DISCARD gate** (fails to beat `arm0` by `>= +1.5` pp HW OR causes `> 0.3` pp printed regression): **TRIGGERED ON ALL THREE ESCALATION ARMS** (HW regressed `-1.63` to `-2.18` pp; Printed regressed `-2.65` to `-3.17` pp, ~10× worse than the `0.3` pp tolerance).

---

## Scrutiny & Anomaly Check

1. **Coverage & Execution Integrity:**
   - All `52/52` pages completed cleanly (`rc=0`, zero NaNs, zero crashes, both HF artifacts uploaded).
   - Ran on Kaggle `2× T4` (`native_bf16=False`, FP32 fallback) instead of Modal `L4` (`BF16`), finishing in `54.4 min` (`~62 s/page`).
   - Minor reporting gap: `eval_payload` logged overall/HW/printed Note F1, escalation rates, and latency multipliers, but did not break out `clean` vs `photo` or `CFR`/`beat_ok_rate` aggregates.

2. **Root-Cause Forensic Diagnosis (5 Defects Explaining the Regression):**
   - **Defect 1 — Miscalibrated Cascade Gates (`tier1_rate = 96.15%`, `tier2_rate = 94.23%`):**
     The Tier-1 trigger (`beat_ok_rate < 0.92 or has_loop or not hit_eos or mean_entropy > 0.35`) fired on `50/52` pages (including almost all easy printed pages), and the Tier-2 trigger (`beat_ok_rate < 0.95 or has_loop`) fired on `49/52` pages. Instead of `< 1.6×` compute, latency ballooned to **`5.32×`** (Tier 1) and **`7.59×`** (Tier 2).
   - **Defect 2 — Tier 0 (`1408px`) Hypothesis Discarded on Escalation:**
     When `need_t1` fired (`96.15%` of pages), Tier 1 ran MBR strictly over the 3 post-TTT views `{1152, 1280, 1536}` and **never included `t0_res` (`1408px`)** in the candidate pool or enforced a verifier/logprob win-margin gate before replacing Tier 0.
   - **Defect 3 — Ungated Full-Encoder Mean-Pool Scribe-TTT Drift:**
     `run_2step_scribe_ttt` updated the entire encoder (`backbone + router + enc_layers + enc_norm`) for 2 steps at `lr=3e-5` with `1 - cos_sim(mean_pool(enc(1152)), mean_pool(enc(1280)))` without any anchor regularization or consistency/rollback gate. Running this on `96.15%` of pages (including clean printed scores) degraded Printed Note F1 by **`-2.65` pp** and HW Note F1 by **`-1.90` pp**.
   - **Defect 4 — Lower-Res MBR Bias & OOD Full-Page Tier 2 (Not `2×` Zoom Crops):**
     Tier 1 voted over `{1152, 1280, 1536}`, where pairwise LCS between the two lower-res scales (`1152`, `1280`) pulls consensus away from `1408px`/`1536px`. In Tier 2, `run_colab.py` never actually cropped suspect measures or halves — it ran **full-page** decodes at out-of-distribution resolutions `{1664, 1792}px`, further dropping Printed Note F1 from `0.67521` to `0.67001` (`-0.52` pp).
   - **Defect 5 — Geometrically Misaligned Single-Staff Proxy Rasterizer (`arm3 < arm2`):**
     `render_candidate_ink_raster` rendered all notes from the entire multi-system page onto a **single horizontal 5-line staff** (`512×384`), and `chamfer_and_dilated_ink_iou` compared that single-staff cartoon directly against the **full multi-system page image `img_pil`**. That geometric mismatch acted as pure noise, degrading `arm3` HW Note F1 by `-0.55` pp below `arm2` (`0.19723` vs `0.20272`).

---

## Verdict & Follow-Up Action

- **Verdict:** **`DISCARD`** — Do **not** promote this 3-tier cascade or the single-staff proxy Chamfer reranker.
- **Why DISCARD instead of queuing another monolithic cascade:**
  - The clean component ablations are already isolated in dedicated experiments:
    - [`2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta`](../2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta/) tests clean multi-scale FSA+logprob TTA without TTT drift.
    - [`2026-11-26T06-25-ov-38b-scribe-ttt-budget-cons-3840`](../2026-11-26T06-25-ov-38b-scribe-ttt-budget-cons-3840/) tests Scribe-TTT with entropy/consistency budgets and a printed safety slice.
  - Any future escalation cascade must (a) calibrate gates to trigger on `<= 25%` of pages, (b) always retain `arm0` (`1408px`) inside the reranker pool with a strict replacement margin, and (c) only apply visual Chamfer/Ink-IoU reranking on **staff-line-registered per-system/per-measure crops** rendered with a real engraver (`verovio`), never a single-staff cartoon vs a full multi-system page.
