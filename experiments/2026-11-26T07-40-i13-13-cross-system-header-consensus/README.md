# `2026-11-26T07-40-i13-13-cross-system-header-consensus` — Commercial-Inspired Cross-System Header Consensus & Propagation (`i13-13`)

- **Status:** `PROPOSED` (Iter 13 Competitor Scout 2 — Commercial Sheet Music Scanners, Google Colab `NVIDIA L4`, `BF16`)
- **Commercial Inspiration:** **SmartScore**, **ScanScore**, **PhotoScore (Neuratron)**, and **Soundslice** global state propagation. Commercial OMR engines treat clef, key signature, and time signature as document-level structural anchors that propagate across systems, preventing catastrophic *all-measures-wrong* pitch/meter drift when a continuation system's left margin is shadowed, bent, or cropped in a phone photo.
- **Baseline Checkpoint:** `HF_CKPT_PATH` (default `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`; auto-compatible with `UniOMR v2` checkpoints via `eval_uni_suite31.load_uni_checkpoint`) evaluated at canonical `max_dim=1280`, `max_len=3840` on all 45 `suite31_hw` pieces (`clean` + `photo` = 90 pages).
- **Single Variable Tested:** Cross-system `(clef, key, time)` header propagation & consensus locking across `<sys>` boundaries during decoding:
  1. `baseline`: Standard greedy + MTP-4 + FSA (`PrefixState.reset_system()` resets header state independently per system).
  2. `online_sys1_bias_1.5_c85`: Online confidence-gated logit bias (`+1.5`) toward System 1's leading header tokens on Systems `2..N` only when top-1 header confidence `< 0.85`.
  3. `online_sys1_bias_3.0_c90`: Stronger online confidence-gated logit bias (`+3.0`, threshold `< 0.90`).
  4. `twopass_majority_header`: SmartScore/PhotoScore-style 2-pass page-wide majority-vote header lock — computes confidence-weighted majority `(clef_seq, key)` across systems, and re-decodes ONLY low-confidence outlier systems (`< 0.85` header prob) from their `<sys>` prefix with the consensus header forced.

## Hypothesis
In phone photos (`printed_photo`, `hw_photo`) with spine shadows, page curl, or tight left-margin crops, a single misread continuation clef or dropped key accidental shifts every note in that system. Confidence-gated cross-system header propagation will rescue shadowed continuation headers without overriding genuine high-confidence mid-piece key/clef/time changes, lifting `all45_photo` and `hw_photo` Strict Note F1 by `>= +0.005` with `< 15%` latency overhead and zero regression on `clean`.

## Quick Links
- Execution Plan, Arms & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- HF Eval Target: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T07-40-i13-13-cross-system-header-consensus.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T07-40-i13-13-cross-system-header-consensus.json`
