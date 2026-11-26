# Iter 13 Proposal `i13-08`: EAGLE-3 Multi-Layer Speculative Drafter + FSA Constrained Decoding vs MTP-4

- **Experiment ID:** `2026-11-26T07-40-i13-08-eagle3-fsa-spec-decode`
- **Track:** Constrained Decoding & Speculative Multi-Token Prediction (Iter 12 → Iter 13)
- **Platform:** Google Colab `L4` (`24 GB`, native BF16)
- **HF Eval Artifact:** `evals/2026-11-26T07-40-i13-08-eagle3-fsa-spec-decode.json`

## Academic Motivation (2025–2026)
1. **EAGLE-3** (*Li et al., arXiv:2503.01840, 2025*): Replaces feature-regression drafting with direct token prediction fueled by **multi-layer hidden feature fusion** `(low=0, mid=L//2, high=L-1)` and **Training-Time Test (TTT)** unrolled multi-step KL distillation, yielding `3–6.5×` lossless speculative decoding speedups.
2. **XGrammar / Grammar-Constrained Speculative Decoding** (*Dong et al., 2024–2025*): Applying deterministic FSM/FSA syntax masks (`PrefixState`) *inside* the speculative drafter eliminates illegal `bekern` proposals before target verification, increasing acceptance length while preserving **100% bit-exact greedy parity**.

## Hypothesis
Our repo already has a unit-tested, zero-edit `Eagle3DraftHead` (`tmp-train/mvd/eagle3_drafter.py`, `~1.33M` params at `d=384` ≈ `0.7 MB` INT4), but it has never been benchmarked on real `suite31_hw` pages against `UniOMRv2Net`'s built-in MTP-4 heads (`~3.5M` params). Training `Eagle3DraftHead` for 300 TTT steps on a frozen `UniOMRv2Net` (`small_45m`) and coupling it with `PrefixState` FSA masking will achieve **higher draft acceptance rate and `>= 1.4×` faster lossless `tok/s`** than both greedy decoding and MTP-4 while maintaining **100% bit-exact token parity**.

## Controlled 5-Arm Decode Benchmark (Shared Frozen `small_45m` Target)
| Arm | Speculator | `K_draft` | Draft FSA Mask | Verify FSA Mask |
|---|---|---:|---|---|
| `greedy_fsa_baseline` | None | `1` | — | `PrefixState` |
| `mtp4_fsa` | Built-in `UniOMRv2Net` MTP-4 | `4` | `PrefixState` | `PrefixState` |
| `eagle3_k4_no_fsa` | `Eagle3DraftHead` (300 TTT steps) | `4` | Off | `PrefixState` |
| `eagle3_k4_fsa` | `Eagle3DraftHead` (300 TTT steps) | `4` | `PrefixState` | `PrefixState` |
| `eagle3_k6_fsa` | `Eagle3DraftHead` (300 TTT steps) | `6` | `PrefixState` | `PrefixState` |

## Files
- [PLAN.md](PLAN.md) — exact launch commands, HF paths, checkpointing, and promotion gates.
- [run_colab.py](run_colab.py) — self-contained Colab L4 runner (zero edits to `train_uni_v2.py` or `uni_omr_v2.py`).
