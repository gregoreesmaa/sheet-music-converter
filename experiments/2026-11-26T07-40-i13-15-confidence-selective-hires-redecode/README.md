# `2026-11-26T07-40-i13-15-confidence-selective-hires-redecode` — Commercial-Inspired Confidence-Triggered Selective High-Res Re-Decoding (`i13-15`)

- **Status:** `PROPOSED` (Iter 13 Competitor Scout 2 — Commercial Sheet Music Scanners, Google Colab `NVIDIA L4`, `BF16`)
- **Commercial Inspiration:** **Soundslice** targeted ambiguity review, **PhotoScore (Neuratron)** Bad Timing / Suspect Navigator, **ScanScore** confidence highlighting, and **Newzik** selective local constraint resolution. Commercial OMR engines recognize that 80–90% of a page is high-confidence and should remain untouched at fast 1x speed, while errors concentrate in a small fraction of low-confidence / rhythm-inconsistent regions that warrant targeted high-res re-inspection or human-in-the-loop flagging.
- **Baseline Checkpoint:** `HF_CKPT_PATH` (default `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`; auto-compatible with `UniOMR v2` checkpoints) evaluated at canonical `max_dim=1280`, `max_len=3840` on all 45 `suite31_hw` pieces (`clean` + `photo` = 90 pages).
- **Single Variable Tested:** Confidence-triggered selective refinement (using exact per-token log-prob, entropy, and FSA beat violation signals from the fast `1280px` pass) vs uniform single-pass `1280px`:
  1. `baseline_1280`: Single-pass `1280px` greedy + MTP-4 + FSA (1.0x compute).
  2. `conf_triage_page_p25`: Trigger 3-scale `{1152, 1280, 1536}` `best_fsa_beat + logprob` re-decode ONLY on pages flagged as low-confidence (`mean_logprob < -0.18` OR `beat_ok_rate < 0.85` OR `has_loop` OR `not bekern_decode_ok`).
  3. `conf_triage_page_p40`: Broader confidence gate (`mean_logprob < -0.12` OR `beat_ok_rate < 0.92` OR `has_loop` OR `not bekern_decode_ok`).
  4. `conf_triage_system_hires`: **Surgical system-level 1536px re-decode** — keep all high-confidence `1280px` systems intact; ONLY for flagged low-confidence systems (`sys_mean_lp < -0.20` OR `sys_meter_err > 0.01`, capped at `<= 4` systems/page), encode at `1536px`, prefill KV-cache up to the `<sys>` prefix, re-decode ONLY that system, and splice it back in if its joint `fsa_beat + mean_logprob` improves!
- **UX / Product Calibration Bonus:** Also computes AUROC and Precision/Recall of our confidence flagger as a Soundslice/PhotoScore "Suspect Page" detector (`note_f1 < 0.85`), directly informing the `fixes-per-page` correction-UX gate.

## Hypothesis
Confidence-triggered selective re-decoding will capture `>= 80%` of full 3-scale TTA's Strict Note F1 gain (`>= +0.010` on `hw_clean` / `hw_photo`, `>= +0.005` on `all45_photo`) while cutting extra compute from `+200%` (full 3-scale TTA) down to `< 45%` average latency overhead across all 90 pages, AND yield `>= 0.85` AUROC for flagging low-F1 suspect pages.

## Quick Links
- Execution Plan, Arms & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- HF Eval Target: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T07-40-i13-15-confidence-selective-hires-redecode.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T07-40-i13-15-confidence-selective-hires-redecode.json`
