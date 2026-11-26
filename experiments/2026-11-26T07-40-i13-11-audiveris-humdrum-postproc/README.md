# Iter 13 (`i13-11`) — Audiveris & Humdrum/Verovio Deterministic Symbolic Post-Processing

- **Experiment ID:** `2026-11-26T07-40-i13-11-audiveris-humdrum-postproc`
- **Status:** `QUEUED`
- **Source Inspiration:** Open-source OMR competitors **Audiveris** (`HEADERS` -> `MEASURES` -> `RHYTHMS` steps) and **Humdrum `**kern` / Verovio** post-processing pipelines
- **Platform / GPU:** Google Colab `L4` (`24 GB`, BF16 native, ~12 min, ~1.0 CU — single GPU decode pass + 5 CPU symbolic arms!)
- **HF Eval Target:** `evals/2026-11-26T07-40-i13-11-audiveris-humdrum-postproc.json`
- **Core Idea:** Unlike online measure-beam search (`ov-34`/`ov-34b`), which incurs GPU latency and alters autoregressive distributions, Audiveris and Humdrum/Verovio run **zero-GPU-cost deterministic symbolic validation & repair** on the transcribed token stream before MusicXML serialization:
  1. **Audiveris System Header Continuity (`header_continuity`):** Enforce page-wide clef & key-signature persistence across `<sys>` boundaries (recover dropped continuation headers or 1-system outlier clefs inconsistent with consensus).
  2. **Humdrum Accidental-Scope Normalization (`accidental_norm`):** Remove redundant duplicate alterations within the same onset/voice and normalize stray key-contradicting alters inside a measure.
  3. **Audiveris Measure-Rhythm Repair (`rhythm_repair`):** Audit per-voice beat sums against active `time-B-T` and apply minimal 1-edit repairs (snapping a single off-by-dot or 2x-duration final note/rest when exact bar duration is restored without dropping notes).
- **Arms (Single-Variable Ablation on Identical Decoded Streams):**
  - `raw_fsa_baseline`
  - `audiveris_header_continuity`
  - `humdrum_accidental_norm`
  - `audiveris_rhythm_repair`
  - `combined_all_rules`
