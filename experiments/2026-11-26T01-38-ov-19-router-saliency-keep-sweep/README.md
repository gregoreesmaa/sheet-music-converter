# `2026-11-26T01-38-ov-19-router-saliency-keep-sweep` — Overnight Exp 19: Router `saliency_thr` × `max_keep_ratio` Sweep

- **Status:** `PLANNED` (`QUEUED` in `experiments/overnight_queue/batch_03.json`, Wave 5)
- **Platform:** Google Colab `1× NVIDIA L4` (`24 GB`, native `BF16`, ~30–35 min, ~2.5–3.0 CU)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params, `InkTokenRouterV2`, `dynamic_routing=True`, `min_keep_ratio=0.15`, `max_mem_tokens=3072`)
- **Dataset (100% permissive):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, zero `suite31_hw` overlap)

## Summary & Hypothesis
In `InkTokenRouterV2` (`tmp-train/mvd/uni_omr_v2.py`), dynamic token routing is governed by two coupled thresholds:
- `max_keep_ratio` caps the per-page token budget during both training (`k = ceil(N * max_keep_ratio)`) and evaluation (`k_max`).
- `saliency_thr` controls the evaluation-time adaptive selection gate (`score_flat > median + saliency_thr`).

Currently `UniOMRv2Config` defaults to `saliency_thr=0.05` and `max_keep_ratio=0.75`, which keeps far more empty-margin tokens than necessary on sparse pages while costing quadratic global-encoder and decoder cross-attention FLOPs.

**Controlled 3-Arm Sweep (`600` steps each, identical `seed=42`, `lr=3e-4`, `Muon + AdamW`, `dim=1280`, BF16 on Colab `L4`):**
1. **`conservative` (`saliency_thr=0.03`, `max_keep_ratio=0.55`):** High ink-recall ceiling, moderate compression (~1.8×–2.2×).
2. **`balanced` (`saliency_thr=0.05`, `max_keep_ratio=0.45`):** Target Pareto operating point (~2.2×–2.8× compression).
3. **`aggressive` (`saliency_thr=0.08`, `max_keep_ratio=0.35`):** High compression (~2.9×–3.8×), tests whether faint accidentals/dots/staff lines are dropped.

We also run a **zero-shot cross-threshold evaluation matrix** on the `balanced` EMA checkpoint across all three `(saliency_thr, max_keep_ratio)` settings to isolate training-time budget effects from inference-time gating.

**Hypothesis:** `balanced (0.05 / 0.45)` will achieve `>= 0.96` ink recall and `>= 2.3×` token compression while matching or beating `conservative (0.03 / 0.55)` on `proxy_val` Token F1 and SER, whereas `aggressive (0.08 / 0.35)` will reveal the exact ink-recall cliff where accuracy begins to degrade.

## Quick Links
- Launcher: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Entry: [batch_03.json](../overnight_queue/batch_03.json)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-19-router-saliency-keep-sweep.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-19-router-saliency-keep-sweep.json`
