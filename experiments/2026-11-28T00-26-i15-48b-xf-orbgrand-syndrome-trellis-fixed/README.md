# `2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed`

- **Status:** `PLANNED` (QUEUED in `experiments/overnight_queue/queue.json`, Wave 0 / P0)
- **Short ID:** `i15-48b`
- **Predecessor:** [`2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma`](../2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma/OUTCOME.md) (`ITERATE`)
- **Baseline Commit SHA:** `dddf9a2`
- **Cross-Field Origin:** Information Theory / IDS Coding (`ORBGRAND` / `SOGRAND` IEEE TWC 2024–25 + `Trellis-BMA` multi-view IDS consensus).

## Summary & Fixes vs `i15-48`
Clean `ITERATE` successor to `i15-48` fixing the forensic root cause documented in `i15-48/OUTCOME.md`:
1. **Multi-Staff / Multi-Voice `compute_bar_duration_syndrome()` Fix:** `i15-48` called `break` on `voice-*` and `<staff>`, which immediately aborted at index `0` of every real bekern bar (since every bar starts with `voice-1`), falsely reporting `syndrome=0` on `230/230` bars and turning Arms 1 & 2 into a 100% no-op. `i15-48b` segments each bar by `(staff_idx, voice_id)` and computes per-voice duration conservation vs `bar_capacity` (handling `bar-rest` per segment, `chord` = 0 advance, `dot` = `1.5×`/chained, `tuplet` scaling, and skipping `grace`), returning non-zero syndrome whenever **any** `(staff, voice)` segment violates `bar_capacity`.
2. **Full OMR Metrics (`note_f1`, `bsvr`, `wfpp`, `gate_pass`):** Integrates the clean Rust (`assembly-serialize` + `eval-harness`) + pure-Python `xml.etree.ElementTree` MusicXML scorer from `i15-44b` alongside Token LCS F1 and SOGRAND `mean_p_not_found`, plus an explicit `gate_pass` evaluation block in `eval.json`.
3. **Hardened CPU `--smoke-test`:** Verifies realistic `voice-1` + `voice-2` + `<staff>` multi-voice bars trigger non-zero syndrome when corrupted, `ORBGRAND` repairs them, `SOGRAND` flags unrepairable pages, `Trellis-BMA` 3-view consensus recovers exact GT on flagged pages, and the MusicXML scorer returns `note_f1 == 1.0` / `wfpp == 0.0` on exact matches.

## Quick Links
- [PLAN.md](PLAN.md)
- [run_colab.py](run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed.json`
- HF Partial Sync: `checkpoints/2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed/`
