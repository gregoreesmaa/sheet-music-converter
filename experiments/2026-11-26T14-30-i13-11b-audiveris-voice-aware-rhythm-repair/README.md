# `2026-11-26T14-30-i13-11b-audiveris-voice-aware-rhythm-repair` (`i13-11b`)

- **Parent Experiment:** `2026-11-26T07-40-i13-11-audiveris-humdrum-postproc` (`ITERATE` — 0 edits due to `<sys>` vocab mismatch and `voice-*` bail-out)
- **Goal:** Properly test Audiveris-inspired deterministic symbolic post-processing on full 45-piece `suite31_hw` (90 pages) after fixing both silent 0-edit bugs:
  1. **Bar-indexed clef/key outlier smoothing** (no `<sys>` dependency — splits directly on the 5 real 237-tok barline tokens).
  2. **Voice-aware per-bar rhythm repair** (segments each bar per `voice-*` stream instead of bailing on `voice-1`, repairing each voice independently).
  3. Caches `pred_raw` tokens in the eval JSON so future symbolic post-proc ablations run on CPU in `< 5 s` with `0` GPU cost.
- **Platform:** Modal / Colab `L4` (`BF16`), est. `~12 min` (`~1.0 CU`, single GPU decode pass).
- **Checkpoint:** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`.
- **Assertions:** `assert total_scored > 0` AND `assert total_edits > 0` (fail-fast against silent no-ops).
