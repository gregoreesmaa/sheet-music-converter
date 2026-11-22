# Phase 4: Closed-Loop Render-and-Diff Verifier, Measure Lattice UI & Rust Packaging (`T4.1` – `T4.3`)

Back to master checklist: [../todo.md](../todo.md)

**PREREQUISITE GATE:** Do NOT start any task in this file until `T3.3` has passed **Gate 5 (`strict-F1 > 0.975`, `OMR-NED < 0.08`)** in `todo/phase-3-lossless-bekern-scale.md`.

---

## `[ ]` Task T4.1 — Build Closed-Loop Visual Measure Verifier (`tmp-train/mvd/verify_measures.py`)

### Objective
Catch the last ~2% of errors (dense chords, inner voices, ledger lines, accidentals) at test time by comparing the predicted notes in each measure against the visual `HeatNet` notehead/accidental peaks on the rectified staff strip, and re-decoding any mismatched measure with visual-coverage-constrained beam search (`beam=5`).

### Exact Implementation (`tmp-train/mvd/verify_measures.py`)
1. For each decoded measure $m$ in a system strip:
   - Parse the predicted notes in measure $m$: each pitched note has a known `(staff, onset_fraction, staff_step, accidental)` which maps deterministically to an expected `(x_col, y_canon_row)` on the rectified staff strip (`T2.1`).
   - Compare predicted `(x_col, y_canon_row)` noteheads against the detected `HeatNet` (`Note` class) local peaks in measure $m$'s horizontal slice `[x_bar_start .. x_bar_end]`.
   - Flag measure $m$ as **discrepant** if:
     - Any `HeatNet` notehead peak (`prob >= 0.85`) has no predicted note at that `staff_step` within `±1.5 * staff_space` horizontally (**missing note**), OR
     - Any predicted note has `HeatNet` notehead probability `< 0.20` at its `(x_col, y_canon_row)` (**hallucinated pitch/note**), OR
     - Mean token log-probability in measure $m$ is below `-0.35`.
2. For any discrepant measure $m$ (typically < 10% of measures on a page):
   - Re-run measure decoding with `beam=5` + a **visual pitch-prior bonus** (`+alpha * log(heat_prob[staff_step])` added to the pitch logits matching the `HeatNet` peak `staff_step`s in that measure's $x$-window).
   - Keep the candidate beam that minimizes `(unexplained_peaks + missing_peaks) - 0.5 * logprob`.
3. Create `tmp-train/mvd/test_verify_measures.py` testing that a synthetic measure where greedy decoding dropped 1 chord tone in a 3-note triad is detected as discrepant and recovered by the visual pitch-prior beam re-decoder.

### Verification Command
```powershell
.\.venv\Scripts\python.exe -m unittest tmp-train/mvd/test_verify_measures.py -v
```

### Pass Criteria
- `test_verify_measures.py` passes (`OK`).

---

## `[x]` Task T4.2 — Add Measure-Atomic Confidence & Top-2 Measure Lattice Sidecar to `converter/assembly-serialize` (DONE 2026-11-14: cargo 71+8 / 46 green; e2e total=1 per=[0,1])

### Objective
Emit `<stem>_lattice.json` alongside `.musicxml` and `.mscz` in `bekern2musicxml` so any measure with residual uncertainty carries its pixel bounding box, confidence score, and top-2 alternative measure transcriptions for 1-click user review.

### Files to Read First
- `converter/assembly-serialize/src/main.rs`
- `converter/assembly-serialize/src/lib.rs`

### Exact Changes to Make
1. In `converter/assembly-serialize`:
   - Add an optional `--lattice-in <path.json>` flag to `bekern2musicxml` (and `pack`) that validates and copies/embeds per-measure confidence + top-2 `bekern+`/MusicXML measure alternatives ranked by `cluster_size * (1.0 - confidence)` into `<out>/<stem>_lattice.json`.
   - Add a `fixes-estimate` report field in `eval-harness` that computes **measure-atomic gestures** (1 click if the ground-truth measure equals candidate #2 in the lattice, else note-level edits inside that measure).
2. Add unit tests in `converter/assembly-serialize` and `converter/eval-harness`.

### Verification Command
```powershell
cargo test --manifest-path converter/assembly-serialize/Cargo.toml
cargo test --manifest-path converter/eval-harness/Cargo.toml
```

### Pass Criteria
- All tests in `assembly-serialize` and `eval-harness` pass (`0` failures).

---

## `[ ]` Task T4.3 — Export ONNX Pipeline, Wire `converter/pipeline-orchestrator`, and Pass Gate 6 (10× Lower Error Rate Than Sheetly)

### Objective
Replace the `SplitMix64` RNG placeholders in `converter/pipeline-orchestrator` with the real end-to-end pipeline, fix the `layout-staff` ↔ `symbol-detect` JSON schema mismatch, and verify **Gate 6** on the sealed `gate20` + 31-piece benchmarks.

### Files to Read First
- `converter/pipeline-orchestrator/src/main.rs`
- `converter/layout-staff/src/main.rs`
- `converter/symbol-detect/src/main.rs`

### Exact Steps
1. Fix the schema mismatch between `layout-staff` and `symbol-detect` (`symbol-detect` `StaffEntry` accepting integer or string `id` via custom serde deserializer and reading nested `systems[].staves[].lines` directly without needing an external adapter).
2. Wire `converter/pipeline-orchestrator` so that when real stage binaries/models are configured, it executes `ingest-cleanup` → `layout-staff` + `system-crop` (rectified) → ONNX/Python `MvdNet` v3 + `verify_measures` → `assembly-serialize bekern2musicxml` while preserving the `(seed, stage, unit)` `.pipeline-state` resume contract and `tmp+rename` artifact writes.
3. Run the final evaluation across the 31-piece benchmark and the single pre-registered `gate20` full (`dev-12` + `test-8`) benchmark.

### Verification Command
```powershell
cargo test --manifest-path converter/layout-staff/Cargo.toml
cargo test --manifest-path converter/symbol-detect/Cargo.toml
cargo test --manifest-path converter/pipeline-orchestrator/Cargo.toml
.\.venv\Scripts\python.exe tmp-train/mvd/eval_sys_gate.py --ckpt tmp-train/mvd-runs/sys-bekernplus-v3/ckpt-best.pt --hybrid --rectify --verify-measures --suite31 --include-test-sealed --out data-gen/score-corpus/gate20/probe_report_final_gate6.json
.\.venv\Scripts\python.exe -c "import json; r=json.load(open('data-gen/score-corpus/gate20/probe_report_final_gate6.json')); print(r['summary']); assert r['summary']['suite31_photo_f1_mean'] >= 0.995 and r['summary']['suite31_photo_omrned_mean'] < 0.02 and r['summary']['median_fixes_per_page'] <= 2 and r['summary']['median_secs_per_page'] <= 2.0"
```

### Pass Criteria
- **Gate 6 passed:**
  - `suite31_photo_f1_mean >= 0.995` (< 0.5% note error rate vs. Sheetly's 2.7%).
  - `suite31_photo_omrned_mean < 0.02` (vs. Sheetly's 0.185).
  - Median fixes/page `<= 2` (zero pages > 10).
  - 100% valid MusicXML 3.1 + `.mscz`.
  - Median `secs/page <= 2.0 s`.
- Update `progress.md`, mark all tasks `[x]` in `todo.md`, and `git commit`.
