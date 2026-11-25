# Sheet Music Converter

**Phone photo / scan / handwritten sheet music in → MuseScore 4 (`.mscz`) & MusicXML 3.1 out.**

A permissive-licensed (`MIT / BSD / Apache / CC0 / CC-BY / PD` — zero AGPL/GPL in training or inference) optical music recognition (OMR) system and synthetic training-data factory built in **Rust** and **PyTorch**.

- **Accuracy target:** `>= 0.90` Strict Note F1 (`0` median fixes/page) across clean prints, phone photos, and handwritten manuscripts.
- **Deployment target:** Autonomous, zero-human-in-the-loop transcription compact enough for **old-device in-browser WASM/WebGPU** execution (`< 55 MB` budget; current `UniOMR` core is **`43.5M` params ≈ `21.7 MB` INT4 / `43.5 MB` INT8**).

---

## Measured `suite31_hw` Scoreboard (45 Unseen Pieces × Clean/Photo = 90 Pages)

Apples-to-apples **Strict Musical Note F1** (exact pitch, duration, onset, and voice via `assembly-serialize` + `eval-harness`) on `data-gen/score-corpus/suite31/suite31_hw_manifest.json` (31 unseen printed pieces + 14 unseen historical handwritten pieces):

| Model / Pipeline | Params / Size (INT4) | Printed Clean | Printed Photo | HW Clean | HW Photo | Printed DecOK | Speed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Best 2-Stage (`Arm C` + `exp2B`)** (`StaffUNet2D` → `MvdNet`) | 2-stage pipeline | **0.7482** | 0.6949 | **0.0568** | **0.0387** | 100% | ~0.16 s/page |
| **`UniOMR` Iter 5** (`stride=8`, `FSA + MTP-4`, `1024px`) | `43.5M` (`21.7 MB`) | 0.2863 | 0.0407 | 0.0255 | 0.0077 | 100% / 90.3% | ~355 tok/s |
| **`UniOMR` Iter 6** (`4×8`, `FSA + MTP-4`, `1280px`, zero Stage-1) | `43.5M` (`21.7 MB`) | 0.7014 | **0.7213** | 0.0101 | 0.0023 | **100%** | **447 tok/s** |

> **Highlight:** In a single iteration (Iter 5 → Iter 6), **`UniOMR`** — a single-stage, zero-cropper full-page model — jumped **`+0.6806` on Printed Photo (`0.0407 → 0.7213`)**, **beating our best 2-stage pipeline (`0.6949`)** with `100%` printed `DecOK` and `1.000` Strict Note F1 on `repeats-melody`, `mozart-k458-1`, `airds-air`, and `foster-leadsheet`. See [`docs/UNIOMR.md`](docs/UNIOMR.md) for the full architecture, forensic lessons, and Iter 7 plan.

---

## Ecosystem & Repository Layout

Three standalone engines split out into dedicated companion repositories, paired with the Rust data-gen/converter crates and neural OMR trainer in this repo:

### Extracted Companion Repos

| Repository | Language | Role |
| :--- | :---: | :--- |
| **[`mscz-rs`](https://github.com/gregoreesmaa/mscz-rs)** | Rust | MusicXML 3.1/4.0 ↔ MuseScore 4 (`.mscx` / `.mscz`) serializer, validator, and `bekern` notation compiler |
| **[`hw-score-sim`](https://github.com/gregoreesmaa/hw-score-sim)** | Python | Multi-style vector-stroke handwritten sheet-music simulator (polyphony, slurs, tuplets, lyrics, `bekern` export) |
| **[`doc-photo-degrade`](https://github.com/gregoreesmaa/doc-photo-degrade)** | Rust | Deterministic CPU/WGPU phone-photo degradation engine (paper aging, bend, perspective, optics, noise, UV masks) |

### In-Repo Rust Crates & Neural Trainer

| Path | What It Does |
| :--- | :--- |
| `data-gen/score-render` | MusicXML/MEI/ABC/MSCX/MSCZ → clean PNG + ground-truth JSON (MuseScore batch / Verovio / LilyPond / stub; multi-font + European metric paper presets) |
| `data-gen/hand-ink` | Deterministic Rust handwritten-ink raster variant (`render → hand-ink → vary → degrade → pack`) |
| `data-gen/notation-vary` | Transpose/respace variants + copy-paste rare symbols (dots, staccato) |
| `data-gen/photo-degrade` | Clean PNG → phone-photo JPEG/PNG + JSON param records (CPU + WGPU `--gpu` backend, geometry-preserving) |
| `data-gen/dataset-pack` | Pair validation by stem, train/val/test split, SHA-256 manifest, DVC-friendly layout |
| `data-gen/pipeline-orchestrator` | Std-only, zero-dep deterministic orchestrator (`render → vary → degrade → pack`) with `--resume` |
| `converter/ingest-cleanup` | Photo → deskewed binary image with EXIF normalization + blur/glare quality gates |
| `converter/layout-staff` | Cleaned image → staff systems JSON + staff-probability map |
| `converter/symbol-detect` | Crops + staff map → glyph boxes/labels JSON (`GlyphDetector` trait + ONNX hook) |
| `converter/assembly-serialize` | Note list / `bekern` → MusicXML 3.1 (4.0 opt-in) + validators → `.mscx` / `.mscz` |
| `converter/eval-harness` | Clean-room `strict-f1`, `strict-xml`, and `notation-f1` musical evaluators |
| `converter/pipeline-orchestrator` | Std-only, zero-dep converter orchestrator (`ingest → layout → symbols → serialize`) |
| `tmp-train/mvd/` | Neural OMR trainers & evaluators: **`uni_omr.py`** (`UniOMR`), `eval_uni_suite31.py`, `grammar.py`, `model.py`, `train.py` |

---

## European Metric Paper Size Presets (`score-render`)

`data-gen/score-render` supports European metric paper sizes via `--musescore-page-sizes` (plus arbitrary `WxH` in mm, `80..=600`):

| Preset | Dimensions (mm) | Notes |
| :--- | :---: | :--- |
| `a4` | `210 × 297` | Standard European portrait sheet music |
| `a4_land` (`a4-land`) | `297 × 210` | Landscape orchestral / manuscript layout |
| `a5` (`pocket`) | `148 × 210` | Pocket / miniature score (`spatium = 1.45 mm`) |
| `a5_land` (`excerpt`) | `210 × 148` | Landscape musical excerpt |
| `b4` | `250 × 353` | ISO B4 conductor / chamber score (`spatium = 1.55 mm`) |
| `folio` | `240 × 355` | Concert folio (`spatium = 1.55 mm`) |
| `WxH` | Custom mm | e.g., `210x148` |

---

## Quick Start

> **Windows note:** Use Git Bash (`C:\Program Files\Git\bin\bash.exe`) for `.sh` wrappers and `.\.venv\Scripts\python.exe` for PyTorch/CUDA scripts.

```bash
# 1. Run the synthetic data-gen pipeline (release build)
bash scripts/run-data-generator.sh

# 2. Run the converter pipeline (MusicXML test config)
bash scripts/run-converter.sh converter/pipeline-orchestrator/test-musicxml.toml

# 3. Evaluate UniOMR on the 45-piece (90-page) suite31_hw benchmark with FSA grammar + MTP-4
.\.venv\Scripts\python.exe tmp-train/mvd/eval_uni_suite31.py \
    --ckpt D:/tmp/mvd-runs/uni-omr-iter6-clean/uni_omr_best.pt \
    --max-dim 1280 --max-len 3072 --use-grammar --use-mtp-spec --device cuda \
    --out D:/tmp/uni-omr-suite31-ood/eval_out.json

# 4. Run Rust unit tests across core crates
cargo test --manifest-path data-gen/score-render/Cargo.toml
cargo test --manifest-path data-gen/photo-degrade/Cargo.toml
cargo test --manifest-path converter/assembly-serialize/Cargo.toml
cargo test --manifest-path converter/eval-harness/Cargo.toml
```

---

## Key Documentation

- **[`docs/UNIOMR.md`](docs/UNIOMR.md)** — Complete `UniOMR` technical architecture, Iter 1–6 benchmarks, forensic lessons, and Iter 7 + `~20M` browser-student roadmap.
- **[`AGENTS.md`](AGENTS.md)** — Repository architecture, determinism/resume contracts, companion repo rules, and engineering conventions.
- **[`goal.md`](goal.md)** — Active session goals, accuracy gates, and standing rules.
- **[`todo.md`](todo.md)** — Master task checklist and ordered workstreams.
- **[`progress.md`](progress.md)** — Append-only log of measured on-disk experiment results.
- **[`.agents/skills/learn-from-suite31/SKILL.md`](.agents/skills/learn-from-suite31/SKILL.md)** — Post-eval forensic learning loop (parallel eyeball scouts + CPU ablations before every GPU run).
