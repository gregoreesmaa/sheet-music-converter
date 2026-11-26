# `2026-11-26T07-40-i13-17-paddleocr-vrfm-band-router` — PaddleOCR-VL Valid Region Focus Module (VRFM) Border & Staff-Band Routing on `suite31_hw`

- **Status:** `PROPOSED` (Iter 13 Competitor Scout 3 — Frontier Document/Math/HTR Transfer, Google Colab `NVIDIA L4`, `BF16`)
- **Inspiration:** **PaddleOCR-VL (v1.5 / v1.6) Valid Region Focus Module (VRFM)** + **Surya / Marker / Docling** coarse layout region focusing & contiguous line/band retention.
- **Baseline:** HF `checkpoints/iter10_5_best/uni_omr_ema_best.pt` (fallback `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`) evaluated at canonical `max_dim=1280`, `max_len=3840` on `suite31_hw` (all 14 HW pieces + 10 printed pieces = 48 pages).
- **Single Variable Tested:** **Zero-retrain inference-time VRFM spatial saliency shaping inside `InkTokenRouter`** (prior to deterministic top-$k$ selection), testing:
  1. **Border-ring clutter suppression** (`border_frac ∈ {0.0, 0.03, 0.06}`): suppresses dark desk borders, archival colorbars, and corner stamps so they cannot steal router budget (`k_kept`) from notation.
  2. **Horizontal staff-band energy smoothing** (`band_alpha ∈ {0.0, 0.30, 0.60}`): adds `band_alpha * row_mean_saliency` (smoothed across a 3-row kernel) before top-$k$, keeping contiguous horizontal staff ribbons intact instead of fragmented patches.
  3. **Combined VRFM + slight keep-budget lift** (`border=0.03, band=0.30, keep_ratio=0.55`).

## Summary & Hypothesis
Post-Iter-6 and Post-Iter-8 forensic audits (`progress.md`) identified two persistent `InkTokenRouter` failure modes on real phone photos and archival historical manuscripts:
1. **Border-ring token theft:** Dark desk borders, archival colorbars, foxing, and corner stamps score high in ink saliency and displace faint staff/notehead tokens from the top-$k$ budget (`max_keep_ratio=0.45`).
2. **Staff-band fragmentation:** Global unconstrained top-$k$ has no horizontal line/system prior, allowing a dark stained top region to starve a fainter lower staff system.

**PaddleOCR-VL's VRFM** discards non-valid background/border regions via coarse layout energy while preserving contiguous horizontal content bands and strict reading order.

**Hypothesis:** Injecting a lightweight, zero-latency VRFM spatial prior (border-ring suppression + horizontal staff-band smoothing) into `InkTokenRouter`'s pre-top-$k$ saliency map will improve `hw_photo` and `printed_photo` Strict Note F1 (`>= +0.010` on HW photo or `>= +0.005` on printed photo) with **0% inference latency overhead**.

## Quick Links
- Execution Plan, Arms & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Hugging Face Eval Target: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T07-40-i13-17-paddleocr-vrfm-band-router.json`
- Hugging Face Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T07-40-i13-17-paddleocr-vrfm-band-router.json`
