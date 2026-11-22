# System-level learned decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a system-level `MvdNet` v2 that passes Gate 2 (held-out system SER < 0.48 AND strict-F1 > 0.22 on `gate20` degraded) and advances to Gate 3 (SER < 0.15, F1 > 0.50), then Gate 4 full-page assembly.

**Architecture:** Engraver-exact system crops + `bekern` spans from renderer truth (never classical barlines) feed a high-res pretrained ConvNeXt-Tiny AR decoder trained 1-clean:3-degraded; page output is system-decode concatenation through `bekern2musicxml`.

**Tech Stack:** Python `torch` 2.14.1+cu126 / `timm` Apache-2.0 ConvNeXt-Tiny, Pillow + numpy, Rust `score-render` / `photo-degrade --gpu` / `assembly-serialize bekern|bekern-decode|bekern2musicxml`, `converter/system-crop` geometry only.

**Spec:** [goal.md](../../../goal.md), [todo.md](../../../todo.md), [progress.md](../../../progress.md) — gates, kill criteria, and bans in those files are normative.

## Global Constraints

- Never train on `gate20` (20 frozen pages) or `exclusion.json` stems (`tmp-train/mvd-data/quarantine.txt`, 26 stems); `train.list_pairs` must load 0 quarantined and 0 gate IDs.
- System training targets come deterministically from engraver/renderer layout truth; 0 `STUB` spans, never classical barline guessing.
- Keep AGENTS.md seams: `(seed, stage, key)` determinism, `write_artifact` tmp+rename, `mark_done`, fixed on-disk naming; extend tmp+rename + fsync + single-writer lock + config-fingerprint refusal to `train.py`.
- Licenses stay permissive (MIT/BSD/Apache); approved trunk init is Apache-2.0 `timm` ConvNeXt-Tiny only — no torchvision-distributed weights, no ConvNeXt-V2 timm, no Nougat/Donut-v2/DiT, no GrandStaff archive, no `abc2xml.py`.
- Gate on strict-F1 + fixes/proxy-fixes with CIs on `gate20` dev; TER/SER are diagnostics only; NOREF pages excluded never zero-filled; EMPTY refs use type-level `StrictOutcome`.
- Never set `QT_QPA_PLATFORM=offscreen`; MuseScore runs use per-process TEMP shards; verify no `MuseScore4` lingers before re-runs.
- Banned: page-level from-scratch AR at any step budget; polishing decoding knobs on ungrounded checkpoints; bare scheduled sampling without EOS handling; beams >5 without length norm; removing the grammar mask; runs >2k steps without blank/shuffled grounding probes first.
- Rust pixel loops build/run with `--release`; Git Bash at `C:\Program Files\Git\bin\bash.exe` on Windows.

## Review Focus

- Blank/shuffled system image still decodes a plausible score (ungrounded encoder) — expect ~0.00 strict-F1 on blank input when grounded.
- System crop cuts through a staff or drops a barline (poisoned supervision) — expect pixel bbox to match renderer symbox within rounding and `bekern-decode` round-trip F1 = 1.0.
- Degraded strip unreadable to a human but paired as supervision — expect 1 clean : 3 degraded per target and human-readable degraded strips.
- `bekern` span off by one bar shifts every later `onset_in_bar` — expect exact `[bar_start, bar_end]` + carried `clef/key/time` headers per system.
- Concurrent `--resume` or mismatched config silently appends to one `--out` dir — expect refusal, not a merged run.

---

### Task 1: §0a repo safety baseline

**Files:**
- Create: `.gitignore`
- Create: `.git/` via `git init` (repo state, not a file edit)
- Modify: none

**Interfaces:**
- Consumes: nothing
- Produces: clean `git status` baseline commit other tasks build on

- [ ] **Step 1: Write `.gitignore` with the spec's exact exclusions**

```gitignore
.venv/
**/target/
**/*.pt
**/*.onnx
```

plus bulk render/degrade/output lines covering `data-gen/score-corpus/big500-renders/`, `renders-fonts/`, `degraded10/`, `tmp-train/mvd-runs/`, `converter/*/out/`, `data-gen/*/out/`.

- [ ] **Step 2: `git init`, add tracked baseline, commit**

Run: `git init && git add .gitignore AGENTS.md goal.md todo.md progress.md tmp-train/mvd converter data-gen scripts && git commit -m "chore: baseline before system-decoder work"`
Expected: PASS — `git status` clean, `git log --oneline` shows 1 commit.

### Task 2: §0b trainer + eval hardening

**Files:**
- Modify: `tmp-train/mvd/train.py`
- Test: `tmp-train/mvd/test_schedule.py`, probe script writing `probe_report_*.json` with `gate20` dev strict-F1 + `proxy-fixes` + blank-image delta

**Interfaces:**
- Consumes: Task 1 clean tree
- Produces: `main()` refuses concurrent writers and config-mismatched `--resume`; checkpoint save in `main()` does tmp+rename + fsync; every probe reports `strict-F1 + proxy-fixes + blank-delta`

- [ ] **Step 1: Write the failing test for single-writer lock + fingerprint refusal**

```python
def test_second_writer_refuses(tmp_path):
    # acquire lock in-process A, second main() with same --out exits 1
def test_resume_mismatched_config_refuses(tmp_path):
    # write meta.json lr=3e-4, resume with lr=1e-4 exits 1 (config-error class)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tmp-train/mvd/test_schedule.py -x -q`
Expected: FAIL — lock/fingerprint symbols missing.

- [ ] **Step 3: Implement lock + fingerprint + fsync saves in `tmp-train/mvd/train.py`**

Single-writer file lock per `--out` dir; full config fingerprint in `meta.json` refused on `--resume` mismatch (exit 1, config-error class); checkpoints via `ckpt-<step>.pt.tmp` + rename + fsync; wire blank-image contrast + cross-attention entropy/coverage + EOS-mass trace into eval probes.

- [ ] **Step 4: Run tests and probes**

Run: `.venv/Scripts/python.exe -m pytest tmp-train/mvd/ -q`
Expected: PASS — all green including new lock/fingerprint tests.

- [ ] **Step 5: Commit**

```bash
git add tmp-train/mvd/train.py tmp-train/mvd/test_schedule.py
git commit -m "feat: trainer single-writer lock, fingerprint refusal, fsync saves"
```

### Task 3: E5-sys-data engraver-exact system pairs + 1:3 degraded mix

**Files:**
- Create: `tmp-train/mvd-data-sys/` output layout (script or `score-render` batch + `assembly-serialize bekern --measures M1..M2`)
- Modify: `converter/system-crop/src/main.rs` (geometry only; bar spans from renderer truth, no detector)
- Test: round-trip script asserting 100% `bekern-decode` validity and F1 = 1.0 on a 50-score pilot (~1,000 systems)

**Interfaces:**
- Consumes: `assembly-serialize bekern --input ref.musicxml --out refs/ --page N --stem S` and `bekern-decode --input S-bekern.txt --out notes.json`; `score-render` single-system strips or symbox-banded crops; `photo-degrade --gpu --format jpeg --max-width 2480`
- Produces: `pair(sys_png, sys_bekern.txt)` records with `bar_start/bar_end`, carried headers, `bbox_xyxy`, `staff_space_px`; max length ≤ 512 tokens; ≥10,000 pairs at full-corpus scale

- [ ] **Step 1: Write the failing round-trip check on 50-score pilot**

```python
def test_sys_roundtrip_f1_is_one():
    # for each pilot (sys_png, sys_bekern.txt): bekern-decode round-trips, len <= 512, 0 STUB
    assert stub_count == 0 and min_f1 == 1.0 and max_len <= 512
```

- [ ] **Step 2: Run pilot generation to verify it fails (STUB spans present)**

Run: `bash scripts/run-data-generator.sh` on pilot config + round-trip script
Expected: FAIL — `.bekern` files carry `# STUB` comment lines.

- [ ] **Step 3: Implement Path A (symbox/y-band slice of `big500`/`mscx-5000` renders) or Path B (MusicXML measure-span slice + `score-render` single-system strip batch) with exact `bekern --measures M1..M2` targets**

Derive per-system `[bar_start, bar_end]` + pixel bbox from MuseScore layout/symbox or single-system renders only; carry `clef/key/time` headers; record `frame_*`/`warp_*`/`rotation_deg` params; run `photo-degrade` so each target pairs 1 clean + 3 degraded.

- [ ] **Step 4: Verify pilot + degraded mix**

Run: round-trip script + `cargo test -p pipeline-orchestrator` determinism check
Expected: PASS — ~1,000 systems, 0 STUBs, 100% `bekern-decode` validity, F1 = 1.0, max ≤ 512, 1:3 clean:degraded on disk.

- [ ] **Step 5: Commit**

```bash
git add converter/system-crop/src/main.rs <generator-script> tmp-train/mvd-data-sys/README.md
git commit -m "feat: engraver-exact system pairs with degraded mix"
```

### Task 4: E6-sys-overfit MvdNet v2 (H=256–384, stride-8/16, pretrained trunk)

**Files:**
- Modify: `tmp-train/mvd/model.py` (`MvdNet.__init__`, `MvdNet.encode`)
- Test: `tmp-train/mvd/test_model.py` (8-strip overfit: >0.90 strict-F1 train, ~0.00 blank/shuffled probe at ≤500 steps)

**Interfaces:**
- Consumes: Task 3 system pairs (`load_image(path, height=256)`, `data.encode + EOS_ID`)
- Produces: `MvdNet(vocab, d_model=256, layers=4, heads=8)` with stride-8/16 vertical features and Apache-2.0 `timm` init ready for Task 5

- [ ] **Step 1: Write the failing overfit test**

```python
def test_sys_overfit_8_strips():
    # train 500 steps, batch>=8, H=256, timm init: train strict-F1 > 0.90, blank-image F1 ~0.00
```

- [ ] **Step 2: Run to verify it fails on v1 trunk**

Run: `.venv/Scripts/python.exe tmp-train/mvd/train.py --renders <sys-clean> --targets <sys-bekern> --height 256 --steps 500 --out tmp-train/mvd-runs/sys-overfit`
Expected: FAIL — stride-32 `convnext_tiny().features` (`H/32 × W/32`) cannot clear 0.90 / blank probe ungrounded.

- [ ] **Step 3: Implement `MvdNet` v2 in `tmp-train/mvd/model.py`**

Feed `H = 256..384` (aspect width `1536..2048`, 12–16 px/staff-space); cut vertical stride 32→8/16 (late-stage stride edit or lightweight FPN merge before `self.proj`); init `self.encoder` from license-clean Apache-2.0 `timm` ConvNeXt-Tiny weights; keep `PAD/BOS/EOS = vocab-3/-2/-1`, `loss()` teacher-force contract, `greedy_kv`/`beam_search` parked.

- [ ] **Step 4: Run 500-step 8-strip overfit + blank/shuffled probe**

Run: same `train.py` line as Step 2 plus blank-image probe
Expected: PASS — train strict-F1 > 0.90, blank/shuffled ~0.00.

- [ ] **Step 5: Commit**

```bash
git add tmp-train/mvd/model.py tmp-train/mvd/test_model.py
git commit -m "feat: MvdNet v2 high-res stride-8/16 with timm init"
```

### Task 5: E7-sys-train toward Gate 2 → Gate 3

**Files:**
- Modify: `tmp-train/mvd/train.py` (system corpus loader, `batch_size >= 8`, `bf16/fp16` + grad clip, warmup + cosine, `trunk_lr_mult = 1.0`)
- Test: held-out `gate20` dev-system probe reporting strict-F1 + SER + CIs + `proxy-fixes` + blank delta (E4 knobs `beam=3` only after grounding)

**Interfaces:**
- Consumes: Tasks 3–4 (full 1:3 corpus, v2 model)
- Produces: `ckpt-*.pt` + `probe_report_sys.json` evidencing Gate 2 (SER < 0.48 AND F1 > 0.22) then Gate 3 (SER < 0.15 AND F1 > 0.50)

- [ ] **Step 1: Write the failing gate-probe assertion**

```python
def test_gate2_probe():
    # probe_report_sys.json: degraded-dev SER < 0.48 and strict-F1 > 0.22 with CIs
    assert ser < 0.48 and f1 > 0.22
```

- [ ] **Step 2: Run baseline probe to verify it fails pre-training**

Run: probe over `gate20` dev systems with v2-init (no full training)
Expected: FAIL — below Gate 2.

- [ ] **Step 3: Implement curriculum in `train.py`**

1-system overfit sanity → short/medium bands (exact `--pages-file` lists) → full corpus 1 clean : 3 degraded; `batch_size >= 8`, `bf16`/`fp16` + grad clip, warmup + cosine LR, uniform `trunk_lr_mult = 1.0`; flat-F1-past-15%-budget and ungrounded-probe kill criteria honored.

- [ ] **Step 4: Train and probe; stop/continue per kill criteria**

Run: full `train.py` + dev-system probe with CIs
Expected: PASS at Gate 2 before claiming Gate 3; E4 `greedy_kv`/`beam_search` (`beam=3`) re-enabled only on the grounded v2 model.

- [ ] **Step 5: Commit**

```bash
git add tmp-train/mvd/train.py probe_report_sys.json
git commit -m "feat: system v2 training run with gate2 probe"
```

### Task 6: Page stitcher → Gate 4 (only after Task 5 passes Gate 2)

**Files:**
- Modify: `converter/pipeline-orchestrator` (`ingest-cleanup → layout-staff --model neural --weights rownet_v3.onnx → system-crop → MvdNet v2 → bekern2musicxml → .mscz`)
- Test: `gate20` full-page run: median `proxy-fixes/page <= 50`, 100% valid MusicXML + `.mscz`, `secs/page <= 5` published

**Interfaces:**
- Consumes: Task 5 system checkpoints + `bekern2musicxml --input sys-bekern.txt --out out/`
- Produces: concatenated system `bekern` → page MusicXML 3.1-first + `.mscz` via `pack`

- [ ] **Step 1: Write the failing Gate 4 probe (12 dev pages)**

```python
def test_gate4_page():
    assert median_proxy_fixes <= 50 and valid_xml == "100%" and secs_per_page <= 5
```

- [ ] **Step 2: Run to verify it fails pre-stitcher**
- [ ] **Step 3: Implement page stitcher (geometry from system-crop, decode per crop, concatenate streams)**
- [ ] **Step 4: Run Gate 4 probe and publish `secs/page`**
- [ ] **Step 5: Commit**
