# `2026-11-26T01-38-ov-42-v1-to-v2-distill-temp-jsd` — Overnight Exp 42: `v1 → v2` Online Distillation Temperature × G-JSD/KL × Alpha Sweep

- **Status:** `PLANNED` (`QUEUED` in `experiments/overnight_queue/batch_10.json`, Wave 8)
- **Baseline Commit SHA:** `164852f`
- **Teacher (Frozen BF16):** `gregoreesmaa/sheet-music-converter-omr` → `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`UniOMR v1` soup)
- **Student:** `UniOMR v2 small_45m` (`~43M` total params)
- **Dataset (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages)

## Summary & Hypothesis
Transferring dark knowledge from our strongest `UniOMR v1` checkpoint (`uni_omr_soup.pt`) into the new `UniOMR v2 small_45m` architecture can accelerate convergence and improve generalization, but online distillation is sensitive to softmax temperature `T`, divergence geometry (**Generalized JSD `beta=0.5`** vs **Forward KL `beta=1.0`**), and loss weight `alpha` (cosine-annealed to `0` over the final 25% of steps).

We run a controlled **2-stage factorial sweep** (`350` steps per arm, identical `seed=42`, `lr=3e-4`, `B=2`, `grad_accum=4`, `fullpage_max_dim=1280`, BF16 on Colab `L4`) with automatic vocab-size alignment between `v1` teacher and `v2` student:
1. **Control:** `alpha = 0.0` (pure supervised `small_45m`).
2. **Stage 1 (`T × Objective` at `alpha = 0.50`, 6 arms):** `T ∈ {1.0, 2.0, 4.0}` × `{G-JSD (beta=0.5), Forward KL (beta=1.0)}`.
3. **Stage 2 (`Alpha` sweep at winning `(T*, beta*)`, 2 arms):** `alpha ∈ {0.25, 0.75}` (completing the full `{0.0, 0.25, 0.50, 0.75}` curve).

**Hypothesis:** Generalized JSD (`beta=0.5`) at moderate temperature (`T=2.0`) and `alpha ∈ {0.25, 0.50}` will outperform both pure supervised training (`alpha=0.0`) and mode-covering Forward KL (`beta=1.0`) on `proxy_val` Token F1 and CE loss.

## Quick Links
- Colab Launcher: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Entry: [batch_10.json](../overnight_queue/batch_10.json)
- Distillation Harness: [train_uni_v2.py](../../tmp-train/mvd/train_uni_v2.py)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-42-v1-to-v2-distill-temp-jsd.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-42-v1-to-v2-distill-temp-jsd.json`
