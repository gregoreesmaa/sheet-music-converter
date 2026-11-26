# `2026-11-26T07-40-i13-28-decoupled-wd-norm-nodecay` — Decoupled Muon/AdamW Weight Decay & 1D Norm/Bias No-Decay Audit

- **Status:** `QUEUED` (Colab `L4`, Iter 13 Batch 10)
- **Baseline Commit SHA:** `9303e51`
- **Model:** `UniOMR v2 small_45m` (`~41.3M` params)
- **Dataset:** `gregoreesmaa/sheet-music-converter-omr` -> `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, 100% permissive)

## Audit Finding & Hypothesis
In `tmp-train/mvd/train_uni_v2.py` (`build_optimizer_v2`, lines 779-808), a single uniform `--weight-decay 0.01` is applied indiscriminately to **every** parameter — including 1D `RMSNorm`/`LayerNorm`/`BatchNorm` gains, biases, `tok_emb`, and both `Muon` 2D Transformer matrices and `AdamW` 4D FPN convs:
1. Decaying 1D normalization scales and biases distorts RMSNorm/LayerNorm calibration and DeepStack residual gating.
2. Under `Muon(adjust_lr_fn="match_rms_adamw")`, orthogonalized spectral updates benefit from stronger weight decay (`0.05`-`0.10`, Moonlight/Kimi-K2 recipe) to bound weight norms in BF16 and improve generalization, whereas `0.01` is severely under-regularized.

**Hypothesis:** Excluding 1D norms/biases/embeddings from weight decay (`wd=0.0`) and decoupling Muon 2D weight decay (`0.05` or `0.10`) will lower `proxy_val` CE and improve `val_token_f1` and `val_ser` over the uniform `0.01` default at zero compute cost.

## Quick Links
- Launcher script: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T07-40-i13-28-decoupled-wd-norm-nodecay.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T07-40-i13-28-decoupled-wd-norm-nodecay.json`
