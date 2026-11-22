# Phase 3: Lossless `bekern+` Vocabulary & `mscx-5000` Scale (`T3.1` – `T3.3`)

Back to master checklist: [../todo.md](../todo.md)

**PREREQUISITE GATE:** Do NOT start any task in this file until `T2.4` has passed **Gate 4 (`strict-F1 > 0.90`)** in `todo/phase-2-hybrid-dewarp.md`.

---

## `[ ]` Task T3.1 — Extend `bekern` to Lossless `bekern+` in `converter/assembly-serialize` (Lift `suite31` Oracle Ceiling `0.8765` → `>= 0.995`)

### Objective
On the 31-piece Sheetly benchmark (`data-gen/score-corpus/suite31/gt/gt-p1-*.musicxml`), our current `bekern` v1 round-trip (`MusicXML → bekern → bekern2musicxml → strict-xml`) achieves **`0.8765` Mean Strict-F1** (`11/31` at `1.000`, `3/31` failing on `cpebach-h186`, `schubert-lindenbaum`, and `weber-concertino` due to `untypeable duration 0.2504464` / `0.2461309` beats against `1e-9` tolerance in `duration_token`, and `17/31` losing 1–17% of notes from skipped grace notes, mid-stream clef/key/time changes, or pickup bars), plus ~15–18% of symbols scored by `musicdiff` (`OMR-NED`) such as slurs, articulations, dynamics, and repeat barlines. Extend `bekern.rs` and `decode.rs` so `31/31` `suite31` GT scores round-trip with `mean strict-F1 >= 0.995` and `OMR-NED < 0.02`.

### Files to Read First
- `converter/assembly-serialize/src/bekern.rs`
- `converter/assembly-serialize/src/decode.rs`
- `converter/assembly-serialize/src/musicxml_import.rs`
- `converter/assembly-serialize/src/lib.rs`

### Exact Additions to Make
1. **Fix Non-Dyadic `<divisions>` Tick-Quantum Tolerance in `duration_token` (`converter/assembly-serialize/src/bekern.rs` line 56):**
   - In `duration_token(dur_beats: f64)`, change `if (d - dur_beats).abs() < 1e-9` to pick the nearest table entry in `DURATION_TABLE` when `(d - dur_beats).abs() <= GAP_MIN_BEATS` (`0.002` beats). This fixes tick-quantized triplet/sixteenth durations in `cpebach-h186` (`0.2504464`), `schubert-lindenbaum` (`0.2504464`), and `weber-concertino` (`0.2461309`).
2. **Mid-Stream Clef / Key / Time Changes & Pickup Bars:**
   - Allow `clef-<sign>-<line>`, `key<fifths>`, and `time-<beats>-<type>` tokens at the start of any measure (or mid-measure voice cursor) in `bekern.rs` and `decode.rs`, emitting `<attributes>` changes in `to_musicxml` / `to_mscx` and updating the active time-signature capacity for subsequent measures (fixing `gounod-meditation`, `mozart-k581`, `schubert-winterreise-01`, `Sibelius-Op5`, etc.).
   - Allow incomplete pickup/anacrusis measures (first measure or post-repeat measure with beat sum `< measure_capacity`) without forcing trailing padding rests that shift note onsets.
3. **Grace Notes:**
   - Instead of skipping grace notes in `to_bekern`, emit `grace` before the grace note's duration + pitch tokens (`grace eighth C 5`), and emit `<grace/>` in `to_musicxml` (fixing `bartok-romanian`, `mozart-k488`, `scarlatti-k1`, `chopin-op28-20`).
4. **Slurs, Articulations, Dynamics, Fermata, and Repeat Barlines:**
   - Add closed-vocab modifiers after note tokens: `slur-start`, `slur-stop`, `stacc`, `accent`, `tenuto`, `fermata`.
   - Add measure-level dynamic tokens (`dyn-pp`, `dyn-p`, `dyn-mp`, `dyn-mf`, `dyn-f`, `dyn-ff`, `dyn-sfz`, `hairpin-cresc-start`, `hairpin-dim-start`, `hairpin-stop`) and barline variants (`bar-light-heavy`, `bar-repeat-left`, `bar-repeat-right`, `bar-double`).
5. Synchronize `tmp-train/mvd/data.py` (`VOCAB`) and `tmp-train/mvd/grammar.py` with the expanded `bekern+` token list.
6. Add round-trip unit tests in `converter/assembly-serialize` verifying `MusicXML → bekern+ → decode_bekern → to_musicxml` preserves non-dyadic divisions, grace notes, mid-score key/time/clef changes, slurs, articulations, and dynamics.

### Verification Command
```powershell
cargo test --manifest-path converter/assembly-serialize/Cargo.toml
.\.venv\Scripts\python.exe -m unittest discover -s tmp-train/mvd -p "test_*.py" -v
```

### Pass Criteria
- All Rust and Python unit tests pass (`0` failures), `e4_corpus_roundtrip.py` verifies `0` blocked files across `big500`, and all `31/31` `data-gen/score-corpus/suite31/gt/gt-p1-*.musicxml` files round-trip through `bekern` → `bekern2musicxml` → `strict-xml` with `0` errors and `mean strict-F1 >= 0.995`.

---

## `[ ]` Task T3.2 — Scale System Dataset Across `mscx-5000` (~75,000 Systems, 6 Fonts, On-the-Fly GPU Augmentations)

### Objective
Scale training diversity from 500 scores to all ~2,514 `mscx-5000` scores across 6 engraving fonts (`Leland`, `Bravura`, `Gonville`, `MuseJazz`, `Petaluma`, `Gootville`) with fast on-the-fly PyTorch GPU augmentations.

### Files to Read First
- `data-gen/score-corpus/mscx-5000/`
- `data-gen/score-render/src/main.rs`
- `tmp-train/mvd/build_sys_data.py`

### Exact Steps
1. Render `data-gen/score-corpus/mscx-5000/` via `score-render` (`--musescore-dpi 300 --musescore-fonts Leland,Bravura,Gonville,MuseJazz,Petaluma,Gootville`), strictly excluding any work in `tmp-train/mscx-5000-test-works.txt` and `gate20/exclusion.json`.
2. Run `build_sys_data.py` over the rendered `mscx-5000` outputs to produce ≥50,000 aligned `(system_strip, bekern+)` pairs.
3. Add on-the-fly GPU photometric/geometric augmentations in `tmp-train/mvd/train.py` (random contrast/gamma, illumination gradient, Gaussian/defocus blur, subtle affine shear, random erasing of tiny background patches) in addition to `photo-degrade` JPEGs.

### Verification Command
```powershell
.\.venv\Scripts\python.exe -c "import json; m=json.load(open('tmp-train/mvd-sysdata/manifest.json')); print(m); assert m['clean_systems'] >= 25000 and m['gate20_leaked'] == 0"
```

### Pass Criteria
- `clean_systems >= 25000` (≥75,000 total with degraded variants) and `gate20_leaked == 0`.

---

## `[ ]` Task T3.3 — Train `bekern+` Hybrid Model at Scale & Pass Gate 5 (Beat Sheetly: `strict-F1 > 0.975`, `OMR-NED < 0.08`)

### Objective
Train the Phase 3 `bekern+` Hybrid `MvdNet` on the 75k-system multi-font corpus and evaluate on both `gate20` and the 31-piece head-to-head benchmark (`EVAL-ORACLE-BASELINE.md`) using `strict-xml` and `musicdiff` (`OMR-NED`).

### Verification Command
```powershell
.\.venv\Scripts\python.exe tmp-train/mvd/eval_sys_gate.py --ckpt tmp-train/mvd-runs/sys-bekernplus-v3/ckpt-best.pt --hybrid --rectify --suite31 --out data-gen/score-corpus/gate20/probe_report_v3.json
.\.venv\Scripts\python.exe -c "import json; r=json.load(open('data-gen/score-corpus/gate20/probe_report_v3.json')); print(r['summary']); assert r['summary']['suite31_photo_f1_mean'] > 0.975 and r['summary']['suite31_photo_omrned_mean'] < 0.08"
```

### Pass Criteria
- **Gate 5 passed:** 31-piece photo mean `strict-F1 > 0.975` (beating Sheetly's `0.973`) AND mean `OMR-NED < 0.08` (more than 2× lower than Sheetly's `0.185`).
- Record results in `progress.md` and commit to Git.
