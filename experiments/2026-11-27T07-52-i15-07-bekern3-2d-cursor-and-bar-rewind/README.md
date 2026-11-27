# `2026-11-27T07-52-i15-07-bekern3-2d-cursor-and-bar-rewind` — `bekern3` 2D Spatial Cursor, Macro-Topology & Local Bar KV-Rewind

- **Experiment ID:** `2026-11-27T07-52-i15-07-bekern3-2d-cursor-and-bar-rewind`
- **Short ID / Wave:** `i15-07` (Wave 13 — `bekern3` Architect 2: 2D Spatial Cursor, Macro-Topology & Local Bar KV-Rewind)
- **Status:** `QUEUED`
- **Baseline Commit SHA:** `e426c28`
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, BF16 native, est. ~65 min / ~4.8 CU)
- **Datasets (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`1,200` steps/arm training + `proxy_val`)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 31 printed + 14 real HW, `clean` + `photo` = 90 pages, incl. targeted diagnostics on `ryans-jig`, `mozart-k387`, and `aloha-oe`)

## Summary & Hypothesis
Ungrounded 1D `bekern` v2a decoding suffers from three structural failure modes on full-page OMR: (1) unconstrained cross-attention drifts backward into copy-loops and `max_len` truncations (currently patched with brittle post-hoc n-gram heuristics), (2) cutaway/multi-staff systems and anacrusis/mid-bar line breaks lack explicit macro-topology markers (`aloha-oe`), and (3) a single misread duration inside a dense measure corrupts rhythmic alignment for the rest of the bar without a low-cost recovery mechanism.

This 2-arm, 1,200-step Colab L4 ablation (`arm0_bekern_v2a` baseline vs `arm1_bekern3_spatial`) formalizes `bekern3` Architect 2's three coupled hypotheses:
1. **2D Spatial Cursor (`@y00..@y63` + `@x00..@x63`) + Strict 2D Monotonic FSA:** Emit quantized vertical cursor `@y00..@y63` after `<bos>`/`<sys>` and horizontal cursor `@x00..@x63` after every barline (`bar` / `BarClose`), using train-only CE-masked (`ignore_index`) `@y?`/`@x?` fallbacks for noisy/unaligned handwritten staves. At eval, a strict 2D monotonic Finite-State Automaton (FSA) enforces `y[s] > y[s-1]` across systems and `x[b] > x[b-1]` within each system (forbidding `@y?`/`@x?`), driving `loop_truncation_count` to `0%` structurally with zero post-hoc n-gram hacks.
2. **Macro-Topology & Exact `PPQ=40320` Tick Accounting:** Add explicit `<part>`, `<s1>..<s12>`, `<pickup>`, and `bar-split` tokens backed by exact integer `PPQ = 40320` tick accounting (`lcm` of all standard/dotted/tuplet divisions) to rescue cutaway/multi-staff layouts and pickup/split measures (`aloha-oe` `S_play`).
3. **Eval-Time Local Single-Bar KV-Cache Rewind (`beam=4`):** Whenever the exact `PPQ=40320` tick accumulator detects a beat-sum mismatch at `BarClose` (excluding valid `<pickup>` and `bar-split`), slice the decoder self-attention KV cache back to the start of that single bar (reusing precomputed cross-attention K/V at zero encoder cost) and run a local `beam=4` search to select the highest-probability beat-consistent bar — targeting `<= +12%` latency overhead.

## Quick Links
- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Queue Entry (pending merge into `overnight_queue/queue.json`): [queue_entry.json](./queue_entry.json)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T07-52-i15-07-bekern3-2d-cursor-and-bar-rewind.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-27T07-52-i15-07-bekern3-2d-cursor-and-bar-rewind.json`
