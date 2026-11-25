# `2026-11-26T01-38-ov-21-paper-gate-border-threshold` — Overnight Exp 21: Edge-Connected Flood-Fill `paper_gate` Dark-Border Threshold Sweep

- **Status:** `PLANNED` (`QUEUED` in `experiments/overnight_queue/batch_03.json`, Wave 5)
- **Platform:** Google Colab `1× NVIDIA L4` (`24 GB`, ~15–25 min, ~1.5–2.0 CU)
- **Baseline Commit SHA:** `164852f`
- **Target Component:** `compute_ink_and_paper_gate_f32` (`tmp-train/mvd/vir_router_v2.py:140–215`) & `InkTokenRouterV2._compute_ink_and_paper_gate` (`tmp-train/mvd/uni_omr_v2.py:662–723`)
- **Datasets (100% permissive, 290 pages total):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (all **90 pages**: 31 printed + 14 real historical handwritten manuscripts, `clean` + `photo`)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/rism_hw_5707.tar` (**200 deterministic seeded-sampled** real historical handwritten RISM manuscript pages, `seed=42`)

## Summary & Hypothesis
The router's pure-float32 edge-connected dark-border `paper_gate` scans inward from all 4 image edges until hitting the first bright pixel (`luma >= luma_stop`, currently hardcoded `0.45`), flags dark exterior pixels (`luma < dark_thr`, currently hardcoded `0.30`), dilates by 16 px (`1×33` + `33×1` max-pool) to remove the desk-to-paper halo, and gates out exterior border patches (`paper_gate = 0.05`).

On dark aged parchment (RISM manuscripts) or heavily shadowed phone photos (`suite31_hw` `photo`), an overly aggressive `(luma_stop, dark_thr)` can bleed inward and mask out real musical notation near page margins, whereas an overly timid pair leaves dark desk/scanner borders unmasked, wasting router token budget on non-musical background.

**Exhaustive `3 × 3 = 9` Grid Sweep (`1280px` max-dim, GPU-accelerated float32):**
- `luma_stop ∈ {0.35, 0.45, 0.55}`
- `dark_thr ∈ {0.22, 0.30, 0.38}`
- Includes an automatic bit-exact parity assertion (`max_abs_diff < 1e-6`) at the current default `(0.45, 0.30)` against `vir_router_v2.compute_ink_and_paper_gate_f32`.

**Metrics across 5 cohorts (`printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`, `rism_real_hw_200`):**
1. **Interior Ink Recall** (pixel-level & patch-level fraction of true interior musical ink preserved with `effective_paper_f == 1.0`, plus simulated top-40% routed ink recall)
2. **Desk-Border Suppression Rate** (fraction of exterior dark desk/border pixels suppressed on `photo` & RISM pages)
3. **Clean-Page False-Masking Rate** (fraction of patches wrongly masked on borderless `suite31_hw` `clean` pages — ideal `0.00%`)
4. **Collapsed-Page Rate** (pages where `effective_paper_f.mean() < 0.50`, catching catastrophic flood-fill bleed-in on dark aged parchment) and worst-5 pages per config.

**Hypothesis:** Lowering `luma_stop` from `0.45` to `0.35` (or pairing `luma_stop=0.45` with `dark_thr=0.22`) will substantially reduce false bleed-in on dark aged RISM manuscripts while preserving `>= 98%` desk-border suppression on `suite31_hw` `photo` pages.

## Quick Links
- Reference Implementation: [vir_router_v2.py](../../tmp-train/mvd/vir_router_v2.py) & [uni_omr_v2.py](../../tmp-train/mvd/uni_omr_v2.py)
- Launcher: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Entry: [batch_03.json](../overnight_queue/batch_03.json)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-21-paper-gate-border-threshold.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-21-paper-gate-border-threshold.json`
