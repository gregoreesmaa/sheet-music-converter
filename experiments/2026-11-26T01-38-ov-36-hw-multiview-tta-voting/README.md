# `2026-11-26T01-38-ov-36-hw-multiview-tta-voting` — Multi-View TTA & FSA/Beat-Conservation Reranking on Historical HW `suite31_hw`

- **Status:** `COMPLETED (ITERATE)` — Oracle headroom `+0.0419` (`hw_clean 0.3030`) / `+0.0327` (`hw_photo 0.2243`); destructive `clahe`/`deskew` views poisoned 6-view rerankers, while offline scale-only `{1280,1536,1792}` `best_fsa_beat` hits **`0.2954` (`+0.0343`)** clean / **`0.1969` (`+0.0053`)** photo. Successor [`2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta`](../2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta/README.md) queued.
- **Baseline:** Commit `164852f` | Checkpoint: HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`hw_clean 0.2550`, `hw_photo 0.1516` at `1536px`, `max_len=4096`)
- **Single Variable Tested:** Zero-training Test-Time Augmentation (6 deterministic views: `base_1536`, `scale_1280`, `scale_1792`, `clahe_1536`, `deskew_1536`, `deskew_clahe_1536`) combined with unsupervised **FSA + beat-conservation reranking**, **multi-view MBR consensus voting**, and **combined FSA+MBR reranking** on all **14 real historical handwritten manuscripts** (`clean` + `photo` = 28 pages).

## Summary & Hypothesis
Historical handwritten manuscripts suffer from uneven faded ink, yellowed paper background, slight staff-line skew, and variable staff-space density — causing single-view decoding at `1536px` to fail catastrophically on `hw_photo` (`0.1516`) and struggle on `hw_clean` (`0.2550`). Unlike test-time gradient adaptation, multi-view Test-Time Augmentation (deskew + CLAHE local contrast + multi-scale `1280/1536/1792px`) requires zero weight updates and can be scored purely with unsupervised symbolic music constraints (`bekern-decode` validity, FSA grammar integrity, per-bar meter & cross-voice beat conservation) plus Minimum Bayes Risk (MBR) pairwise token-LCS voting across views.

**Hypothesis:** Multi-view TTA with FSA/beat-conservation + MBR consensus reranking will improve Strict Note F1 on the 14 real historical HW pieces (`hw_clean` and/or `hw_photo`) by `>= +0.015` over single-view `base_1536`, and reveal the exact oracle headroom across the 6 views.

## Quick Links
- Execution Plan & Copy-Paste Colab Commands: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Queue Entry: [batch_08.json](../overnight_queue/batch_08.json)
- Outcome: [OUTCOME.md](./OUTCOME.md)
- Successor Experiment: [2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta](../2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta/README.md)
- Hugging Face Eval Target: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T01-38-ov-36-hw-multiview-tta-voting.json`
- Hugging Face Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T01-38-ov-36-hw-multiview-tta-voting.json`
