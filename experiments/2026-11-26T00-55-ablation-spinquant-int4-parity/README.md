# Experiment 8 — `2026-11-26T00-55-ablation-spinquant-int4-parity`

- **Experiment ID:** `2026-11-26T00-55-ablation-spinquant-int4-parity`
- **Status:** `RUNNING` on Modal `L4` (`ap-OrTAfTJlPV0o9aSWO0r59l`, `https://modal.com/apps/gregoreesmaa1/main/ap-OrTAfTJlPV0o9aSWO0r59l`)
- **Baseline Commit SHA:** `9d73170`
- **Platform:** Modal `L4` (`24 GB`, BF16/FP32 native, `~$0.80/hr`, est. ~10–15 min / `~$0.15–$0.20`)

## Summary & Hypothesis

This experiment benchmarks **SpinQuant Offline Hadamard Rotation Folding** (`tmp-train/mvd/spinquant_fold.py`, zero runtime ops) and **Multi-Target ONNX Export** (`tmp-train/mvd/export_uni_v2_onnx.py --targets all`) on full-size **`UniOMR v2`** architectures: **`small_45m`** (`~41.3M`) and **`edge85m`** (`~78.1M`).

We test three concrete hypotheses on real `suite31_hw` pages (`datasets/suite31_hw_eval.tar`, 12 pages across `printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`):
1. **Exact FP32 Functional Invariance:** `fold_hadamard_v2` (R1 decoder residual-stream rotation with `RMSNorm` absorption + R2 per-head V-O rotation across encoder & decoder) preserves FP32 logits to numerical precision (`max_abs_diff < 1e-4`) on both `small_45m` and `edge85m`.
2. **INT4 Quantization Parity Gain (`block_size=32`):** Folding orthogonal Walsh-Hadamard rotations significantly reduces symmetric group-wise INT4 logit MSE, forward KL divergence, and max-abs error while improving top-1 token agreement against FP32 — especially under realistic channel-wise outlier-stressed weights (`12×` `RMSNorm` gamma + `12×` linear channel spikes).
3. **Browser `< 55 MB` Budget Verification:** Full-size `small_45m` and `edge85m` export cleanly across all 3 ONNX targets (`webgpu` INT4 `MatMulNBits`, `npu` static opset-17 INT8 `DequantizeLinear`, `cpu` dynamic INT8), with `webgpu` total (`encoder + decoder`) staying strictly under the **`< 55 MB`** WebGPU download budget.

## Quick Links

- **Plan & Follow-Up Commands:** [PLAN.md](PLAN.md)
- **Self-Contained Modal Launcher:** [run_modal.py](run_modal.py)
- **Outcome (written on completion):** `OUTCOME.md` *(pending)*
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr`
  - Eval JSON: `evals/2026-11-26T00-55-ablation-spinquant-int4-parity.json`
  - Run Ledger: `runs/2026-11-26T00-55-ablation-spinquant-int4-parity.json`
