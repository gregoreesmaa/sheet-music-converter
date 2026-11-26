# Iter 13 Proposal `i13-07`: DAPO + Dr. GRPO Unbiased RLVR on Unlabeled RISM

- **Experiment ID:** `2026-11-26T07-40-i13-07-dapo-drgrpo-rlvr-rism`
- **Track:** Neuro-Symbolic RLVR Post-Training (Iter 12 → Iter 13)
- **Platform:** Google Colab `L4` (`24 GB`, native BF16; automatic FP16+GradScaler fallback on Turing T4)
- **HF Eval Artifact:** `evals/2026-11-26T07-40-i13-07-dapo-drgrpo-rlvr-rism.json`

## Academic Motivation (2025–2026)
1. **DAPO** (*Yu et al., ByteDance Seed / Tsinghua AIR, arXiv:2503.14476, 2025*): Vanilla GRPO suffers from **entropy collapse** (symmetric `clip=0.2` suppresses rare, correct exploratory tokens) and **zero-variance batch waste** (prompt groups where all `G` rollouts achieve near-identical rewards produce zero gradient). DAPO fixes both with **Clip-Higher** (`eps_low=0.20, eps_high=0.28`), **Dynamic Sampling** (filtering/resampling groups with `std(R) < 1e-3`), and **token-level policy gradient loss**.
2. **Dr. GRPO** (*Liu et al., Sea AI Lab / NUS, arXiv:2503.20783, 2025*): Vanilla GRPO divides advantages by `std(R)` (causing difficulty bias that over-weights easy low-variance pages) and normalizes per-sequence loss by variable response length (causing length inflation on wrong/looping sequences). Dr. GRPO removes the `std(R)` denominator (`A_i = R_i - mean(R)`) and normalizes sequence loss by a fixed constant (`max_len`).

## Hypothesis
Our current `rlvr_grpo.py` implements vanilla DeepSeek-R1 GRPO (`(R - mean)/(std + eps)`, symmetric `clip=0.2`, no dynamic sampling). Replacing vanilla GRPO with **Dr. GRPO unbiased advantages** and/or **DAPO Clip-Higher + Dynamic Sampling** on unlabeled RISM handwritten pages (`updates=120`, `G=8`, `kl_beta=0.06`, `max_len=512`) will prevent entropy collapse and length inflation, yielding higher `hw_clean` / `hw_photo` Strict Note F1 at canonical `1280px / 3840` with zero printed regression.

## Controlled 4-Arm Matrix (Single-Variable Objective Ablation)
| Arm | Advantage Norm | Clipping (`low / high`) | Dynamic Sampling (`std >= 1e-3`) | Loss Norm |
|---|---|---|---|---|
| `vanilla_grpo` | `(R - mean) / (std + 1e-6)` | `0.20 / 0.20` | Off | Per-token active mask |
| `dr_grpo` | `R - mean` (no `std` div) | `0.20 / 0.20` | Off | Constant `max_len` |
| `dapo_clip_dyn` | `(R - mean) / (std + 1e-6)` | `0.20 / 0.28` | On (up to 2× resample) | Token-level PG |
| `dapo_drgrpo_hybrid` | `R - mean` (no `std` div) | `0.20 / 0.28` | On (up to 2× resample) | Constant `max_len` |

## Files
- [PLAN.md](PLAN.md) — exact launch commands, HF paths, checkpointing, and promotion gates.
- [run_colab.py](run_colab.py) — self-contained Colab L4 runner (zero edits to `train_uni_v2.py` or `uni_omr_v2.py`).
