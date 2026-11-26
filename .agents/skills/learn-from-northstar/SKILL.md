---
name: learn-from-northstar
description: >-
  Mandatory post-evaluation forensic learning loop for OMR-NorthStar benchmarks
  in sheet-music-converter (eval -> learn -> categorize -> fix -> repeat).
  Activate IMMEDIATELY whenever ANY OMR-NorthStar (eval_northstar_runner.py /
  eval-northstar-*) evaluation finishes (for UniOMR or a competitor baseline):
  record the full apples-to-apples NorthStar ledger, spawn 6 parallel read-only
  CPU forensic scouts across CDR/RWP/ALS/HWM, expressive/structural axes,
  negative controls (NMHR), and competitor differentials, separate fixes into
  Model/Export/Inference/Harness buckets, unit-test on CPU, commit & push, and
  feed into the next training iteration or omr-northstar update.
---

# Learn From OMR-NorthStar — Mandatory Post-Evaluation Forensic Learning Loop

**Trigger:** Activate this skill **IMMEDIATELY** whenever ANY **OMR-NorthStar** (`eval_northstar_runner.py` / `eval-northstar-*`) evaluation finishes — whether for a UniOMR checkpoint or a competitor baseline (`homr`, `Audiveris`, `oemer`, etc.). Never launch the next training run or publish a benchmark update without completing this loop + committing & pushing first.

## Core Principles (Standing Rules)

1. **Full-spectrum NorthStar scoreboard — never cherry-pick `S_note`.** Always report the full headline composite (`NS Score`, `95% CI`, `Tier-Weighted NS`, `M_geom`, `Worst Stratum Q`, `CFR_max`), all 4 domain strata (`CDR`, `ALS`, `RWP`, `HWM`), `Δ_photo`, all 4 fidelity axes (`S_note`, `S_struct`, `S_expr`, `S_play`), human utility (`median_wFPP`, `mean_NTSR`, `NUPR`), and `Negative Controls (NMHR)`.
2. **Never conflate a scorer or exporter bug with a model bug.** Every failure must be traced across the full chain: `raw PNG → UniOMR beKern → bekern2musicxml → MusicXML → cer.py / omr-northstar scorer`. Fix the exact layer that broke.
3. **Visually eyeball actual PNGs and inspect raw `beKern` / MusicXML diffs (`read_file`).** Summary metrics hide catastrophic alignment bugs, silent `bekern2musicxml` drops, blank-page hallucinations, and archival scan failure modes.
4. **Verify hypotheses with fast CPU ablations — never guess.** Run read-only CPU checks (`CUDA_VISIBLE_DEVICES=""`) to confirm whether a `CFR` spike comes from model decoding, XML serialization, or CER alignment.
5. **Protect frozen evals & GPU.** Never modify frozen OMR-NorthStar ground-truth assets or manifests to flatter scores; verify `0` eval overlap on any training data; keep all 6 forensic scouts strictly read-only and CPU-only.

---

## Step 1 — Apples-to-Apples NorthStar Ledger

Present a compact comparison table benchmarking **Previous UniOMR Iter(s) → Current UniOMR Iter → Competitors (`homr`, `Audiveris`, `oemer`) → `Oracle GT`** across every mandatory NorthStar metric:

- **Headline Composite:** `NS Score`, `95% CI`, `Tier-Weighted NS`, `M_geom`, `Worst Stratum Q`, `CFR_max`.
- **Per-Domain Breakdown (`Q / CFR`):** `CDR` (Clean Digital Renders), `ALS` (Archival Library Scans), `RWP` (Real-World Photos), `HWM` (Handwritten Manuscripts), plus `Δ_photo` (`CDR → RWP` / photo drop).
- **4 Fidelity Axes:** `S_note`, `S_struct`, `S_expr`, `S_play`.
- **Human Utility:** `median_wFPP`, `mean_NTSR`, `NUPR`.
- **Pillar 3B Negative Controls:** `NMHR` (Non-Music Hallucination Rate) and non-music false-positive breakdown.
- **Sanity Callouts:** Flag any `Oracle GT < 1.0`, partial coverage, timeout/crash spikes, or `CFR` jumps immediately before attributing them to the model.

---

## Step 2 — Mandatory 6-Scout Parallel CPU Forensic Swarm

Spawn **6 parallel read-only CPU subagents** (`typeName: "self"`, explicit constraint: *STRICTLY READ-ONLY, ZERO FILE EDITS, CPU-only*). Each scout MUST cite exact piece IDs, measured deltas, and visual/token evidence:

1. **Scout 1 — `CDR`/`RWP` `CFR` & MusicXML Export / CER Alignment Autopsy:**
   - Inspect every `CDR` and `RWP` catastrophic failure (`CFR`) and worst-scoring outlier; eyeball input PNGs (`read_file`) alongside predicted `beKern`, exported MusicXML, and GT MusicXML.
   - Strictly distinguish **real model recognition errors** from **`bekern2musicxml` serialization bugs** (dropped voices, broken measure durations, clef/key/meter export bugs) and **`cer.py` / alignment scorer bugs**.
2. **Scout 2 — `ALS` (Archival Scans) Domain-Drop Autopsy (`CDR → ALS` Paired Delta):**
   - Rank the worst `CDR → ALS` paired drops; visually inspect paired clean vs archival scan PNGs (`read_file`).
   - Diagnose scan-specific bottlenecks: yellowed/bleed-through paper, low contrast, engraving/font shifts, border margins, skew, router saliency loss, and layout/system-break drift.
3. **Scout 3 — `HWM` (Handwritten) `CFR` & Human-Utility (`wFPP` / `NTSR` / `NUPR`) Autopsy:**
   - Rank all `HWM` pieces by `Q`, `CFR`, `wFPP`, `NTSR`, and `NUPR`; eyeball worst vs best handwritten PNGs (`read_file`).
   - Identify which failure modes drive `wFPP` / `NUPR` penalties (early `<eos>`, looping, wrong staff count, onset drift, catastrophic measure misalignment) vs usable near-miss outputs.
4. **Scout 4 — Expressive & Structural Fidelity (`S_expr`, `S_struct`, `S_play` vs `S_note`) Autopsy:**
   - Quantify the gap between `S_note` and `S_struct` / `S_expr` / `S_play` across all 4 domains.
   - Audit missing or mis-exported clefs, key/time signatures, barlines, ties, slurs, articulations, dynamics, beaming, tuplets, and multi-voice playability errors (`S_play`).
5. **Scout 5 — Pillar 3B Negative Controls (`NMHR`) & OOD Hallucination Autopsy:**
   - Inspect every non-music / negative-control page where the system emitted spurious notes (`NMHR > 0`); eyeball the actual PNGs (`read_file`).
   - Diagnose whether hallucinations originate from router threshold/saliency leakage, missing negative-control training exposure, decoder grammar forcing note emission, or empty-page post-processing.
6. **Scout 6 — Head-to-Head Competitor Differential (`omr_northstar.compare` vs `homr`, `Audiveris`, `oemer`, etc.):**
   - Run/read `omr_northstar.compare` piece-by-piece differentials against every evaluated competitor (`homr`, `Audiveris`, `oemer`, etc.).
   - Isolate the exact pieces, domains, and axes where competitors still beat UniOMR — and diagnose **why** (layout/staff-count stability, archival contrast robustness, expressive symbols, MusicXML validity, or negative-control rejection).

---

## Step 3 — Actionable Fix Categorization

Synthesize all 6 scout reports into a single ranked action table, strictly separating every finding into **four non-overlapping buckets** (never conflate a scorer/exporter bug with a model bug!):

| Bucket | Scope | Examples |
|---|---|---|
| **(A) Model / Training Fixes** | Architecture, router, loss, curriculum, data mix, augmentation | Archival scan augmentation, negative-control training samples, expressive token weighting, HW exposure |
| **(B) Export / Serialization (`bekern2musicxml`) Fixes** | `beKern → MusicXML` conversion & `mscz-rs` serialization | Multi-voice duration sync, ties/slurs/dynamics export, valid measure divisions, empty-score XML handling |
| **(C) Inference / Post-Processing Fixes** | Decoding, FSA grammar, tiling/scaling, empty-page gating | Negative-control confidence/ink gate, loop breaker, `max_len` / resolution handling |
| **(D) NorthStar Harness / CER Fixes** | `omr-northstar` runner, `cer.py`, alignment, or metric edge cases | Scorer alignment bug, Oracle GT `< 1.0` defect, parser crash on valid MusicXML |

For each item, record: **Root Cause → Affected Metric(s) → Exact Bucket `(A/B/C/D)` → Measured Evidence → Expected Gain**.

---

## Step 4 — Verify, Commit & Feed Into Next Iteration

1. **Implement fixes in parallel non-overlapping CPU subagents** respecting bucket boundaries (separate repo commits if upstream fixes belong in `omr-northstar` or `mscz-rs`).
2. **Unit-test every fix on CPU** before touching the GPU:
   - Run relevant unit suites (`test_uni_omr.py`, `test_uni_decode.py`, `bekern2musicxml` round-trip tests, `omr-northstar` / `cer.py` tests).
   - Re-score affected NorthStar pieces on CPU (or verify `Oracle GT == 1.0` for Bucket D harness fixes) to prove the fix works without regressions.
   - Verify `0` eval overlap for any new training data added under Bucket A.
3. **Update ledgers (`progress.md`, `todo.md`, and `experiments/<exp_id>/OUTCOME.md` if run as an experiment)** with the full Step 1 NorthStar ledger and Step 3 categorized root-cause summary.
4. **Commit & push** using `Gregor Eesmaa <6941817+gregoreesmaa@users.noreply.github.com>`.
5. **Feed directly into the next iteration:** wire Bucket A/C wins into the next UniOMR training/eval run and Bucket B/D wins into the next `omr-northstar` / `bekern2musicxml` update (and note that when `suite31_hw` runs alongside OMR-NorthStar, run `learn-from-suite31` in tandem so both benchmarks stay synchronized).
