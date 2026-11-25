# `2026-11-26T01-38-ov-27-mtp-k-weight-sweep` — Overnight Exp 27: MTP-k × MTP Loss Weight Decoder Sweep

- **Status:** `PLANNED` (`QUEUED` in `experiments/overnight_queue/batch_05.json`, Wave 6)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params, sweeping `mtp_k` heads)
- **Dataset (100% permissive):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages)

## Summary & Hypothesis
`UniOMR v2` trains auxiliary Multi-Token Prediction (`MTP-k`) heads (`mtp_k=4`, `mtp_loss_weight=0.30` by default) both as a dense lookahead regularizer that strengthens decoder representations and as a zero-draft-model **self-speculative decoder** for faster full-page inference. However, the optimal horizon `k` and loss weight for musical BeKern sequences (where notes are structured 2–4 token groups like `quarter C 4`) have never been swept on `UniOMR v2`.

This experiment runs a controlled **3×3 full-factorial sweep** (`9` arms × `300` steps/arm, identical `seed=42`, `lr=3e-4`, `dim=1280`, BF16 on Colab `L4`) on `UniOMR v2 small_45m`:
- **`mtp_k`:** `{2, 4, 6}`
- **`mtp_loss_weight`:** `{0.15, 0.30, 0.50}`

For every arm, `run_colab.py` measures BOTH:
1. **Accuracy (`proxy_val`, 64 pages):** `val_token_f1`, `val_ser`, `val_ce_loss`, `val_mtp_loss`.
2. **Lossless Self-Speculative Decoding Benchmark (`16` val pages, verified bit-exact vs `model.generate()`):** `draft_acceptance_rate`, `mean_accepted_tokens_per_step`, `greedy_tok_per_sec`, `spec_tok_per_sec`, and wall-clock `speedup_x`.

**Hypothesis:** `mtp_k=4` with `mtp_loss_weight=0.30` or `0.50` will lie on the Pareto frontier — matching or beating `mtp_k=2` on `val_token_f1` while delivering significantly higher `mean_accepted_tokens_per_step` and wall-clock `tokens/sec` speedup than `mtp_k=2` without the diminishing draft acceptance of `mtp_k=6`.

## Quick Links
- Colab Runner: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Manifest: [batch_05.json](../overnight_queue/batch_05.json)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-27-mtp-k-weight-sweep.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-27-mtp-k-weight-sweep.json`
