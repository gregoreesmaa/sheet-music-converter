# `2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed`

- **Status:** `PLANNED` (clean `ITERATE` successor to `i15-44` `2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr`)
- **Short ID:** `i15-44b`
- **Baseline Commit SHA:** `ca8826e`
- **Cross-Field Origin:** Compilers (`VerMCTS`/FIM), Drift/RL (`Verifier-Triggered KV-Cache Rewind`), Connectomics (`NEURD`), HITL (`Structure-Conditional MBR`).

## Summary & Hypothesis
Clean bug-fixed `ITERATE` successor to `i15-44` evaluating **Violation-Only Verifier-Triggered KV-Cache Rewind + Structure-Conditional MBR (`kv_rewind_mbr`)** against standard FSA-constrained greedy decoding (`greedy_fsa`) on the exact same 12-page challenging subset of `suite31_hw` (6 printed + 6 real handwritten photo pages) using our best existing Iter 14 checkpoint.

### What `i15-44b` Fixes vs `i15-44`
Forensic scrutiny of `i15-44` (`OUTCOME.md`) uncovered two bugs that confounded `i15-44` despite a promising `-40.0%` relative HW `BSVR` reduction:
1. **Proper `suite31_hw` MusicXML Note F1 & `wFPP` Scoring:** `i15-44` erroneously tokenized raw `.musicxml` ground truth via `piece["gt"].read_text().split()`, yielding `0` reference notes (`note_f1 == 0.0` on all 12 pages) and spurious `wFPP`. `i15-44b` uses the real `eval_uni_suite31` / `eval_sys_gate` Rust MusicXML note-matching pipeline (`suite31_page_refs` + `bekern-decode` + `score_notes_pair`) when compiled binaries are present, with a verified pure-Python `xml.etree.ElementTree` MusicXML note/header fallback on Colab Linux.
2. **Ban Premature `<eos>` Truncation + Violation-Only Trigger + `K=2` Low Latency (`<= 1.35×`):**
   - Hard-bans `eos_id` inside `_rollout_bar_candidate` (`ban_eos=True` in `_apply_fsa_mask`) AND heavily penalizes any premature-`<eos>` (`-100.0`) or unclosed (`-50.0`) rollout in `_select_mbr_bar_candidate` so it can never receive `cons_term=+2.0` or truncate pages.
   - Removes the noisy `peak_ent > 1.60` entropy-only trigger (which doubled printed `BSVR` and slowed HW to `3.30×`) — rewinds trigger **ONLY** on actual bar conservation violations (`meter_err > 1e-3` or `voice_err > 1e-3`).
   - Reduces candidate fanout `K=4 -> K=2` (1 peak-entropy top-1 masked greedy + 1 temperature-sampled FSA continuation), `MAX_RETRIES_PER_BAR=1`, and `MAX_REWINDS_PER_PAGE=8` to target `<= 1.35×` wall-clock overhead.

## Quick Links
- [PLAN.md](PLAN.md)
- [run_colab.py](run_colab.py)
- Predecessor: [`../2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr/OUTCOME.md`](../2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr/OUTCOME.md)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed.json`
- HF Partial Sync: `checkpoints/2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed/`
