# `2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode` — Mathpix-Style Confidence-Gated Per-Bar Beam Re-Decoding on `suite31_hw`

- **Status:** `PROPOSED` (Iter 13 Competitor Scout 3 — Frontier Document/Math/HTR Transfer, Google Colab `NVIDIA L4`, `BF16`)
- **Inspiration:** **Mathpix SuperNet** (`confidence` vs `layout_confidence` separation + localized confidence-gated re-decoding of ambiguous math/handwriting spans) & **TrOCR / PyLaia** selective beam re-decoding.
- **Baseline:** HF `checkpoints/iter10_5_best/uni_omr_ema_best.pt` (fallback `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`) evaluated at canonical `max_dim=1280`, `max_len=3840` on `suite31_hw` (14 HW pieces × 2 domains = 28 pages + 10 printed pieces × 2 domains = 20 pages = 48 pages).
- **Single Variable Tested:** **Per-bar token-confidence trigger threshold** for KV-cached rollback + localized `beam=3` (length-normalized) re-decoding of visually uncertain bars, orthogonal to `ov-34`/`ov-34b`'s rhythm-only FSA triggers:
  - `baseline_greedy` (off — greedy FSA baseline)
  - `conf_minp_0.65` (trigger localized `beam=3` rollback when a bar's bottom-2 mean `p_max < 0.65`, cap `6` bars/page)
  - `conf_minp_0.75` (`bottom-2 mean p_max < 0.75`, cap `6` bars/page)
  - `conf_minp_0.82` (`bottom-2 mean p_max < 0.82`, cap `6` bars/page)
  - `conf_margin_0.25` (trigger when a bar's minimum top-1 minus top-2 probability margin `p1 - p2 < 0.25`, cap `6` bars/page)

## Summary & Hypothesis
Existing measure-beam experiments (`ov-34` / `ov-34b`) only fire when a bar violates rhythmic beat conservation or voice count. However, forensic audits of `suite31_hw` show that many pitch, accidental, clef, and handwritten-manuscript errors are **rhythmically valid** (e.g., wrong diatonic step, missed sharp/flat, or ambiguous iron-gall ledger line) and therefore never trigger FSA rhythm repair.

**Mathpix SuperNet** explicitly decouples recognition confidence from structural layout/grammar validation and triggers localized re-decoding only on low-confidence spans.

**Hypothesis:** Tracking per-bar token uncertainty (`bottom-2 mean p_max` or top-1/top-2 `margin`) during KV-cached FSA decoding and rolling back only the `<= 6` lowest-confidence bars per page for localized `beam=3` re-decoding will recover visually ambiguous pitch/accidental/HW errors missed by rhythm-only repair (`>= +0.012` HW Note F1 or `>= +0.006` Printed Note F1) while keeping latency overhead `< 25%` on printed pages.

## Quick Links
- Execution Plan, Arms & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Hugging Face Eval Target: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode.json`
- Hugging Face Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode.json`
