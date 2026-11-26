# Iter 13 Proposal `i13-09`: ParetoQ Mixed-Bit (INT3/INT2/Ternary FFN + INT4 Attn) & AWQ Channel-Scaled QAT for Sub-25 MB Edge `UniOMR v2`

- **Experiment ID:** `2026-11-26T07-40-i13-09-paretoq-awq-edge-qat`
- **Track:** Sub-55 MB Edge Quantization & Compression (Iter 12 → Iter 13)
- **Platform:** Google Colab `L4` (`24 GB`, native BF16)
- **HF Eval Artifact:** `evals/2026-11-26T07-40-i13-09-paretoq-awq-edge-qat.json`

## Academic Motivation (2024–2026)
1. **ParetoQ** (*Liu et al., Meta AI, NeurIPS 2025, arXiv:2502.02631*): Unified sub-4-bit QAT scaling framework showing a representation transition between 2-bit and 3-bit and proving that **3-bit / 2-bit / 1.58-bit ternary QAT on large FFN layers** dominates uniform 4-bit on the accuracy-vs-MB Pareto frontier.
2. **AWQ (Activation-Aware Weight Quantization)** (*Lin et al., MLSys 2024 Best Paper*) & **Modality-Aware VLM Quantization (2025–2026)**: Exp 8 (`2026-11-26T00-55-ablation-spinquant-int4-parity`) proved offline SpinQuant R1 Hadamard folding *hurts* weight-only W4A16 quantization (`0.59×–0.72×` worse MSE) because absorbing `RMSNorm.weight` smears spikes across groups. AWQ avoids smearing by applying smooth per-channel activation salience scaling (`s_j = mean(|X_j|)^alpha`) that protects the top `~1%` salient channels without dense rotation.

## Hypothesis
Branching from a single 600-step trained `UniOMRv2Net(small_45m)` checkpoint:
1. **AWQ activation-salience channel scaling (`alpha=0.5`)** before W4A16 QAT will beat uniform INT4 QAT on `logit_kl_vs_fp32` and `f1_retention_vs_fp32` (solving Exp 8's weight-only outlier gap).
2. **ParetoQ mixed-bit QAT** (`INT3` or `INT2` or `1.58-bit` ternary on SwiGLU FFN matrices `w_gate_up`/`w_down` + `INT4` on attention & `lm_head`) will preserve `>= 98.0%` FP32 F1 while shrinking `small_45m` from `28.04 MB → ~21–24 MB` and `edge85m` from `50.73 MB → ~36–41 MB` (well inside the `< 55 MB` WebGPU budget, unlocking `edge85m` for Iter 13 browser deployment!).

## Controlled 5-Arm QAT Matrix (Shared 600-Step Trained `small_45m` Base)
| Arm | Attention / `lm_head` | SwiGLU FFN (`w_gate_up`, `w_down`) | Pre-QAT Calibration | Est. `small_45m` / `edge85m` WebGPU MB |
|---|---|---|---|---|
| `int4_uniform_qat` | Symmetric INT4 (`b=32`) | Symmetric INT4 (`b=32`) | None (baseline) | `28.0 MB` / `50.7 MB` |
| `int4_awq_scaled_qat` | Symmetric INT4 (`b=32`) | Symmetric INT4 (`b=32`) | AWQ `s_j = mean(\|X_j\|)^0.5` | `28.0 MB` / `50.7 MB` |
| `paretoq_w3_ffn_w4_attn` | Symmetric INT4 (`b=32`) | Symmetric INT3 (`qmax=3, b=32`) | None | `~25.0 MB` / `~44.6 MB` |
| `paretoq_w2_ffn_w4_attn` | Symmetric INT4 (`b=32`) | ParetoQ 2-bit (`±0.5, ±1.5, b=32`) | None | `~22.0 MB` / `~38.5 MB` |
| `paretoq_ternary_ffn_w4_attn` | Symmetric INT4 (`b=32`) | BitNet 1.58-bit (`{-1,0,+1}`) | None | `~20.5 MB` / `~35.4 MB` |

## Files
- [PLAN.md](PLAN.md) — exact launch commands, HF paths, checkpointing, and promotion gates.
- [run_colab.py](run_colab.py) — self-contained Colab L4 runner (zero edits to `train_uni_v2.py` or `uni_omr_v2.py`).
