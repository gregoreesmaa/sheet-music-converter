# Phase 2: Canonical 4-Slot Staff Dewarping + `HeatNet` Keypoint Fusion + Measure Anchoring (`T2.0` – `T2.4`)

Back to master checklist: [../todo.md](../todo.md)

**PREREQUISITE GATE:** Start `T2.0` after `T1.6d` passes **Gate 2 (`strict-F1 > 0.22`)** in `todo/phase-1-system-decoder.md` (or if `T1.6d` unrectified strips plateau below Gate 3 `0.50` F1 due to `photo-degrade` perspective/curl on 4-staff systems, in which case `T2.0`–`T2.4` clear Gate 3 and Gate 4 together).

---

## `[x]` Task T2.0 — Stage Permanent In-Tree Copy of the 31-Piece Sheetly Benchmark (`data-gen/score-corpus/suite31/`)

### Objective
Currently the 31-piece Sheetly benchmark inputs and page-1 ground-truth cuts reside in `%TEMP%\ours`, `%TEMP%\oursph`, `%TEMP%\omrleg`, and `%TEMP%\sheetly\bundle\bundle`, where a Windows temp cleanup would erase them. Stage an immutable, SHA-256-verified copy into `data-gen/score-corpus/suite31/` (git-ignored binary images, tracked `suite31_manifest.json`).

### Files to Read First
- `converter/eval-harness/suites/README.md`
- `converter/EVAL-ORACLE-BASELINE.md`

### Exact Steps
1. Create a small script `tmp-train/mvd/stage_suite31.py` that copies:
   - The 31 clean page-1 PNGs from `%TEMP%\ours` → `data-gen/score-corpus/suite31/clean/<slug>.png`
   - The 31 photo page-1 images from `%TEMP%\oursph` → `data-gen/score-corpus/suite31/photo/<slug>.png` (or `.jpg`)
   - The 31 page-1 GT MusicXML cuts from `%TEMP%\omrleg\gt-p1-<slug>.musicxml` → `data-gen/score-corpus/suite31/gt/gt-p1-<slug>.musicxml`
   - Competitor published reference scores from `%TEMP%\sheetly\bundle\bundle\manifest.json` (or `benchmark-results.json`) → `data-gen/score-corpus/suite31/suite31_manifest.json`, recording SHA-256 hashes of all 93 files (`31 clean + 31 photo + 31 gt`).
2. Add `data-gen/score-corpus/suite31/clean/` and `data-gen/score-corpus/suite31/photo/` to `.gitignore` (keeping `suite31_manifest.json` and `gt/` tracked or verified).

### Verification Command
```powershell
.\.venv\Scripts\python.exe tmp-train/mvd/stage_suite31.py
.\.venv\Scripts\python.exe -c "import json; m=json.load(open('data-gen/score-corpus/suite31/suite31_manifest.json')); print('pieces:', len(m['pieces'])); assert len(m['pieces']) == 31"
```

### Pass Criteria
- All 31 clean images, 31 photo images, and 31 GT MusicXML files exist in `data-gen/score-corpus/suite31/` with verified SHA-256 checksums (`len(pieces) == 31`).

---

## `[x]` Task T2.1 — Build Canonical 4-Slot Staff-Line Rectifier (`tmp-train/mvd/rectify.py`)

### Objective
Eliminate perspective warp, page curl, `±10°` rotation, and **multi-staff vertical/horizontal scale crush** by dewarping each system strip onto a **fixed 4-staff canonical canvas (`H = 256`, `target_sp = 10.0 px`)**:
- Staff slot $s \in \{0, 1, 2, 3\}$ ALWAYS places its 5 staff lines at fixed rows:
  $$y_{\text{canon}}(s, k) = 12 + 64 \cdot s + 10 \cdot k, \quad k \in \{0, 1, 2, 3, 4\}$$
  (Slot 0: `y = 12..52`, Slot 1: `y = 76..116`, Slot 2: `y = 140..180`, Slot 3: `y = 204..244`).
- If a system has $S < 4$ staves (e.g., $S=1$ melody, $S=2$ piano, $S=3$ Lieder), staves $0..S-1$ occupy slots $0..S-1$ and unused slots $S..3$ are filled with clean paper white (`255`).
- Horizontal width is resampled by `target_sp / measured_sp` (`10.0 / measured_sp`), NOT `256 / raw_h`, so 3- and 4-staff systems never have their horizontal width crushed!

### Files to Read First
- `converter/layout-staff/src/main.rs`
- `tmp-train/row-retrain/` (`rownet_v2.onnx`, `rownet_v3.onnx`)

### Exact Implementation (`tmp-train/mvd/rectify.py`)
1. Create `rectify_system_strip(img: Image.Image, staves_lines: list[list[float]], target_sp: float = 10.0, height: int = 256) -> tuple[Image.Image, torch.Tensor]`:
   - Accepts a PIL image and `staves_lines` (`1` to `4` staves, each with 5 vertical $y$-coordinates or slice curves).
   - Maps each active staff $s \in \{0..S-1\}$ ($S \le 4$) so its 5 lines land on `y_canon(s, k) = 12.0 + 64.0 * s + k * target_sp` (and `±2.0 * target_sp` above/below each staff for ledger lines/stems), filling unused staff slots $s \ge S$ with paper white `(255, 255, 255)`.
   - Scales width to `W_out = round(W_in * target_sp / median_measured_sp)`.
   - Also returns the `[1, height, W_out]` binary/float **Canonical Staff-Line Reference Channel** (`+1.0` on the 5 canonical staff lines of each active staff $0..S-1$, `0.0` elsewhere, including `0.0` in inactive slots $s \ge S$ so the encoder knows how many staves are active!).
2. Create `tmp-train/mvd/test_rectify.py`:
   - Generate a synthetic 2-staff and 4-staff image with a sinusoidal vertical bend of `±8 px` and a `3°` tilt, plus known notehead dots on line 1 (`E4`) and space 1 (`F4`).
   - Run `rectify_system_strip` and assert that all 5 staff lines of each active staff sit within `±0.5 px` of `y_canon(s, k)` across the entire width, inactive slots are pure white (`255`), and the `E4` and `F4` centers sit within `±0.5 px` of `y_canon(0, 4)` and `y_canon(0, 3.5)`.

### Verification Command
```powershell
.\.venv\Scripts\python.exe -m unittest tmp-train/mvd/test_rectify.py -v
```

### Pass Criteria
- `test_rectify.py` passes with 0 errors (`max row residual <= 0.5 px`).

---

## `[x]` Task T2.2 — Fuse `HeatNet` 7-Channel Spatial Keypoints + Staff-Grid Channel into `MvdNet` (`tmp-train/mvd/heat_features.py` + `model.py`)

### Objective
Condition the `MvdNet` encoder directly on `HeatNet`'s **0.982 Note-F1** spatial probability maps (`Note`, `Rest`, `Beam`, `Hook`, `Accidental`, `NoteDot`, `Clef`) plus the 1-channel Canonical Staff-Line Reference Map from `T2.1` alongside the 3 RGB image channels (`extra_chans = 8` → `in_chans = 11`).

### Files to Read First
- `converter/symbol-detect/src/onnx_heat.rs` (see exact preprocessing and 7-class channel order of `tmp-train/det-data/heatnet.onnx`)
- `tmp-train/mvd/model.py`

### Exact Changes to Make
1. Create `tmp-train/mvd/heat_features.py`:
   - Loads `tmp-train/det-data/heatnet.onnx` via `onnxruntime` (or `tmp-train/det-data/heatnet.pt` on CUDA) and computes the `[7, H, W]` spatial probability tensor (`sigmoid` of logits) for a system strip image, concatenated with the `[1, H, W]` canonical staff-line map from `rectify.py` → `[8, H, W]`.
2. In `tmp-train/mvd/model.py`:
   - Add parameter `extra_chans: int = 0` to `MvdNet.__init__`.
   - When `extra_chans > 0` (e.g. `extra_chans = 8`), replace the stem `Conv2d(3, 96, kernel_size=4, stride=4)` with `Conv2d(3 + extra_chans, 96, kernel_size=4, stride=4)`.
   - Copy the pretrained RGB `[:, :3, :, :]` weights and bias from the 3-channel stem and **zero-initialize** `[:, 3:, :, :]` so starting behavior at step 0 is bitwise identical to the pretrained 3-channel model when extra channels are zero!
   - Add `'extra_chans'` to `RESUME_KEYS` and `meta.json` in `train.py`.
3. Add unit test `test_hybrid_heat_channels_forward` in `tmp-train/mvd/test_model.py` verifying that a `[2, 11, 256, 512]` input tensor passes through `MvdNet(vocab=113, anisotropic=True, pretrained=True, extra_chans=8)` and produces identical initial logits (`max diff < 1e-5`) to `extra_chans=0` when the 8 extra channels are zero.

### Verification Command
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tmp-train/mvd -p "test_*.py" -v
```

### Pass Criteria
- All unit tests pass (`OK`), including zero-init parity between `extra_chans=0` and `extra_chans=8`.

---

## `[x]` Task T2.3 — Barline-Anchored Measure Decoding & `S > 4` Staff Windowing (`tmp-train/mvd/measure_decode.py`)

### Objective
1. Prevent a single rhythmic error in one measure from shifting `onset_in_bar` for subsequent measures by enforcing exact measure-duration conservation per bar (`PrefixState(strict=True)`).
2. Support systems with $S > 4$ staves (such as the 5-, 6-, and 7-staff pieces in `suite31`: `palestrina-agnus`, `beach-prayer`, `aloha-oe`) by decoding staves in chunks of up to 4 staves (`0..3`, then `4..S-1`) and merging their emitted `bekern` streams measure-by-measure!

### Files to Read First
- `tmp-train/mvd/grammar.py`
- `tmp-train/mvd/model.py`

### Exact Implementation (`tmp-train/mvd/measure_decode.py` + `grammar.py`)
1. In `tmp-train/mvd/grammar.py`:
   - Verify that when `PrefixState(strict=True)` is active:
     - Tracks the active time signature (`time-B-T` → expected measure duration in quarter beats) and running voice cursor within the current measure.
     - Forbids note/rest durations that overshoot the remaining duration in the current measure.
2. Create `tmp-train/mvd/measure_decode.py`:
   - `decode_system_with_measure_fallback(net, img_tensor, beam=3)`:
     - Runs `net.beam_search(img_tensor, beam=beam, strict=True, max_len=564)`.
     - Checks each emitted measure via `assembly-serialize bekern-decode`. If any measure fails closure, splices a valid `bar-rest` fallback for that measure so downstream measures stay bar-aligned.
   - `decode_multi_staff_system(net, page_img, system_staves_lines, beam=3)`:
     - If `len(system_staves_lines) <= 4`, rectifies once and decodes.
     - If `len(system_staves_lines) > 4`, slices `system_staves_lines` into chunks of up to 4 staves (`[0:4]`, `[4:8]`), rectifies and decodes each chunk independently, and merges their headers and per-measure voice runs (`voice-1..k` + shifted `voice-(k+1)..m`) at each `bar` token!
3. Add unit tests in `tmp-train/mvd/test_grammar.py` verifying strict bar-duration conservation and multi-chunk (`4 + 2 = 6` staves) measure merging.

### Verification Command
```powershell
.\.venv\Scripts\python.exe -m unittest tmp-train/mvd/test_grammar.py -v
```

### Pass Criteria
- `test_grammar.py` passes 100% (`OK`).

---

## `[ ]` Task T2.4 — Train Hybrid Rectified `MvdNet` v2.2 & Pass Gate 4 (`strict-F1 > 0.90` on `gate20` Degraded, Beat Audiveris on `suite31`)

### Objective
Train `MvdNet` v2.2 with Canonical 4-Slot Staff Rectification (`T2.1`), 8-channel `HeatNet` + Staff-Grid fusion (`T2.2`), and Barline-Anchored Measure Decoding (`T2.3`), and pass **Gate 3 (`>0.50` F1)** and **Gate 4 (`strict-F1 > 0.90`, `proxy-fixes/page <= 15` on `gate20` degraded, and `suite31` photo mean `strict-F1 > 0.88` beating Audiveris)**.

### Exact Steps
1. Rectify `tmp-train/mvd-sysdata-full` and `data-gen/score-corpus/gate20/sys-dev` strips using `rectify.py` and precompute/cache the 8 extra channels (`HeatNet` 7-ch + 1 staff-grid ch).
2. Train `MvdNet` v2.2 (`--extra-chans 8 --init-from tmp-train/mvd-runs/sys-v2g/ckpt-45000.pt --steps 30000 --batch-size 4 --accum 2 --lr 3e-4 --trunk-lr-mult 0.1 --tok-weight 3.0 --tok-dropout 0.20 --out tmp-train/mvd-runs/sys-hybrid-v2`).
3. Evaluate on `gate20/sys-dev`, stitched `gate20` dev pages, and `data-gen/score-corpus/suite31/`.

### Verification Command
```powershell
.\.venv\Scripts\python.exe tmp-train/mvd/eval_sys_gate.py --ckpt tmp-train/mvd-runs/sys-hybrid-v2/ckpt-30000.pt --hybrid --rectify --suite31 --out data-gen/score-corpus/gate20/probe_report_hybrid_v2.json
.\.venv\Scripts\python.exe -c "import json; r=json.load(open('data-gen/score-corpus/gate20/probe_report_hybrid_v2.json')); print(r['summary']); assert r['summary']['degraded_sys_f1_median'] > 0.90 and r['summary']['degraded_page_proxy_fixes_median'] <= 15 and r['summary']['suite31_photo_f1_mean'] > 0.88"
```

### Pass Criteria
- Held-out `gate20` degraded system median `strict-F1 > 0.90`, stitched full-page median `proxy-fixes/page <= 15`, and `suite31` photo mean `strict-F1 > 0.88` (beating Audiveris's `0.504` photo and `0.884` clean).
- Update `progress.md` and commit to Git.
