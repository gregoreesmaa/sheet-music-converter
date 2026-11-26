# Experiment Plan — `2026-11-26T14-30-i13-11b-audiveris-voice-aware-rhythm-repair`

## 1. Root Causes Fixed from `i13-11`
In `i13-11`, all 90 pages decoded cleanly, but `total_edits == 0` across every symbolic rule due to two bugs:
1. `rule_audiveris_header_continuity` split on `"<sys>"`, which does not exist in `UniOMR`'s 237-token vocabulary (`len(systems) == 1` on every page).
2. `rule_audiveris_rhythm_repair` skipped any bar containing `t.startswith("voice-")`, rejecting nearly every polyphonic / grand-staff measure (`voice-1`).
Meanwhile, `humdrum_accidental_norm` was confirmed redundant under FSA decoding (`0` edits legitimately) and is dropped.

## 2. Fixed Symbolic Post-Processing Arms (Single GPU Pass, 0 ms GPU Overhead)
All arms evaluate identical FSA-decoded `pred_raw` streams (`max_dim=1280, max_len=3840`, MTP-4 + FSA) across all `90` pages of `suite31_hw`:
1. `raw_fsa_baseline` — untouched FSA output.
2. `bar_header_smoothing` — splits directly on real barline tokens `{"bar", "bar-double", "bar-light-heavy", "bar-repeat-left", "bar-repeat-right"}`, tracks clef/key headers across bars, and smooths isolated 1-bar singleton clef/key glitches when surrounded by a strict consensus (`count >= 3`, singleton `count == 1`).
3. `voice_rhythm_last_event` — segments each bar into independent `voice-*` sub-streams (preserving non-voice prefix tokens and skipping only `tuplet-*`, `grace`, `bar-rest`), checks beat sum vs active `time-N-D`, and applies a conservative 1-edit fix to the final event (double/halve duration, add dot, or remove stray dot).
4. `voice_rhythm_single_edit` — same voice-aware segmentation, extending 1-edit repair to the last matching event in the voice when the final event alone cannot close the beat gap.
5. `combined_header_plus_rhythm` — `bar_header_smoothing` followed by `voice_rhythm_last_event`.

## 3. Caching & Safety Assertions
- Stores `pred_raw` per page in the uploaded HF eval JSON (`evals/2026-11-26T14-30-i13-11b-audiveris-voice-aware-rhythm-repair.json`) so future symbolic post-proc iterations require **0 GPU minutes**.
- Enforces `assert total_scored > 0` and `assert total_edits > 0` before marking `COMPLETED`.

## 4. Promotion / Discard Gates
- **PROMOTE:** Any rule improves `all45_clean` or `hw_clean`/`hw_photo` Note F1 by `>= +0.002` (or `r_rhythm` by `>= +0.015` with non-negative Note F1) and causes `<= -0.001` regression on `pr_clean` and `pr_photo`.
- **ITERATE:** Positive `r_rhythm` gain (`>= +0.015`) with mixed Note F1 -> refine edit selectivity using cached `pred_raw` on CPU (`0` GPU cost).
- **DISCARD:** All arms degrade Note F1 or remain within `[-0.001, +0.001]` despite `total_edits > 0`.
