# `2026-11-26T07-40-i13-29-grad-clip-threshold-sweep` — Global Gradient Clipping Threshold Audit on `UniOMR v2 small_45m`

- **Status:** `QUEUED` (Colab `L4`, Iter 13 Batch 10)
- **Baseline Commit SHA:** `9303e51`
- **Model:** `UniOMR v2 small_45m` (`~41.3M` params)
- **Dataset:** `gregoreesmaa/sheet-music-converter-omr` -> `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, 100% permissive)

## Audit Finding & Hypothesis
In `tmp-train/mvd/train_uni_v2.py` (line 2099), global gradient clipping is **hardcoded** to `torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)` (not even exposed as a CLI argument). Because `UniOMR v2` sums 5 active losses (`CE + 0.30*MTP + 0.10*Saliency + 0.15*SysCount + 0.15*StrokeHOG`) across 41M parameters, the joint L2 norm regularly exceeds `1.0`. Under `Muon+AdamW`, Muon already spectral-normalizes its own 2D Transformer updates via Newton-Schulz, so an overly tight global `1.0` clip disproportionately **squelches the AdamW CNN backbone, router, and embedding gradients**!

**Hypothesis:** Relaxing `grad_clip` from hardcoded `1.0` to `2.0` or `5.0` will stop choking early/mid-stage AdamW backbone and decoder embedding updates, lowering `proxy_val` CE and improving `val_token_f1` without inducing gradient spikes or NaNs.

## Quick Links
- Launcher script: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T07-40-i13-29-grad-clip-threshold-sweep.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T07-40-i13-29-grad-clip-threshold-sweep.json`
