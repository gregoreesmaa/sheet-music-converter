# AGENTS.md — sheet-music-converter

Image of sheet music in, MuseScore file (`.mscz`) out. Two pipelines: a
synthetic training-data generator and the image-to-notation converter.
Everything is Rust, split into small single-purpose crates.

## Layout

| Path | What it does |
|---|---|
| `data-gen/score-render` | MusicXML/MEI/ABC/MSCX/MSZ → clean PNG + ground-truth JSON. Prefers the `musescore` backend (headless export, real engraving, plus `.musicxml` ground truth), then `verovio`/`lilypond`, else built-in `StubRenderer`. Zero-dep. `--renderer`, `--musescore-bin`, `--musescore-fonts A,B` (engraving fonts), `--musescore-text-fonts X,Y` (full cross product; sets all 60 `*FontFace` keys), `--musescore-page-sizes` (European metric mm: `a4`, `a4_land`, `a5`, `a5_land`, `b4`, `folio`, `WxH`). JSON records `font` + `text_font`. |
| `data-gen/notation-vary` | Transpose/respace variants + copy-paste rare symbols (dots, staccato). |
| `data-gen/photo-degrade` | Clean PNG → degraded phone-photo PNG/JPEG (`--format png\|jpeg`, `--max-width` default 2480). Stages: paper grain → `degrade_paper` (foxing, mottling, fibers, water damage) → `degrade_ink` (fading, dropout, bleed-through, toner) → page bend (edge/spine/ripple, tear-free) → perspective → ink-aware artifacts (tears, coffee, pencil, folds, highlighter) → `degrade_wear` (heavy ink loss, printer banding/streaks/toner/misregistration, eraser swirl+smudge, water/oil stains) → lens distortion → tilt-shift/uniform focus → lighting → chromatic aberration → `degrade_capture` (motion blur, prints, tape, holes, exposure) → grain → JPEG round-trip. Seeded ChaCha8 RNG; every output paired with a JSON param record. |
| `data-gen/hand-ink` | Clean render PNG → handwritten-ink-variant PNG + per-file JSON params. Deterministic Rust port of `tmp-train/hw-synth/v3` level `v3_high` (seeded smooth wobble ≤5 px, ink jitter, paper grain; topology-preserving, labels still apply). SplitMix64 per-file streams, tmp+rename, `--resume`. Sits between `render` and `vary` (`render → hand-ink → vary → degrade → pack`); run the binary directly, the orchestrator does not sequence it. |
| `data-gen/dataset-pack` | Validate image↔notation pairs by stem, train/val/test split, SHA-256 manifest, DVC-friendly layout. |
| `data-gen/score-corpus` | Real training source: 4 OpenScore Lieder `.mscx` (voice + piano, provenance in `README-source.txt`), `renders/` (MuseScore PNGs + `.musicxml` + JSON), `renders-fonts/` (corpus × 6 engraving fonts), `degraded10/` (10 phone-photo JPEGs + params JSON). Not fixtures — regenerable. |
| `data-gen/pipeline-orchestrator` | Drives `render → vary → degrade → pack` from one TOML config. **Std-only, zero-dep** (builds offline on render farms). Stage bodies are deterministic placeholders behind the `(seed, stage, key)` contract — see below. |
| `converter/ingest-cleanup` | Photo → deskewed binary image. EXIF normalize, then quality gates: Laplacian-variance blur reject, glare-fraction reject. |
| `converter/layout-staff` | Cleaned image → staff systems JSON + staff-probability map (classical fallback; neural hook). |
| `converter/symbol-detect` | Crops + staff map → glyph boxes/labels JSON. `GlyphDetector` trait: `classical` blob stub default, `OnnxDetectorHook` for learned models. |
| `converter/assembly-serialize` | Note list → MusicXML **3.1-first** (4.0 opt-in) + loss table + validators → `.mscx` / `.mscz` (zip). Subcommands: `serialize`, `validate`, `mscz`, `pack`, `example`. |
| `converter/pipeline-orchestrator` | Drives `ingest → layout → symbols → serialize` from one TOML config. Std-only like its data-gen twin; layout/symbol stages are deterministic placeholders. |
| `scripts/run-data-generator.sh` | `bash scripts/run-data-generator.sh [CONFIG] [EXTRA...]` → cargo run data-gen orchestrator. Defaults to `config.example.toml`. |
| `scripts/run-converter.sh` | Same shape for the converter orchestrator. |
| `converter/pipeline-orchestrator/samples/` | Local 1×1 test PNGs (`sample-page.png`, `sample-page2.png`). Synthetic only. |
| `converter/pipeline-orchestrator/test-musicxml.toml` | Local test config: `format = "musicxml"`, `input_dir = "samples"`. Not the default. |

## Run

```bash
bash scripts/run-data-generator.sh            # 16 scores → out/{render,vary,degrade,pack}
bash scripts/run-converter.sh converter/pipeline-orchestrator/test-musicxml.toml
cargo run --release -p photo-degrade --manifest-path data-gen/photo-degrade/Cargo.toml -- --input IN --out-dir OUT --format jpeg   # fast path: release + threads
cargo run -p assembly-serialize --manifest-path converter/assembly-serialize/Cargo.toml -- pack --input notes.json --out outdir
```

Both `run-*.sh` wrappers pass `--release`: pixel loops are 10–50× slower
in debug. `photo-degrade` also threads across files (per-file seeds, so
output is bit-identical to sequential runs). Measured: 10 full pages in
~6 s release-parallel vs ~13 min debug-sequential (~130×).

On Windows use Git Bash (`C:\Program Files\Git\bin\bash.exe`); bare `bash`
resolves to the WSL stub and fails. Each crate also runs standalone —
`--help` on any binary lists its flags.

## Conventions (do not break these)

- **Determinism:** every byte derives from `(seed, stage, unit key)` via
  SplitMix64 (`stream_for`) in the orchestrators, ChaCha8 in `photo-degrade`.
  Same config + inputs ⇒ bit-identical outputs. Keep per-unit streams
  independent; never mix run order or history into the RNG.
- **Resume:** stage completion is append-only in `<out>/.pipeline-state`
  (`v1 <stage> <unit>` rows + `seed=` header). Fresh runs truncate it;
  `--resume` skips recorded units and refuses a seed mismatch. Artifacts are
  written to `*.tmp` then renamed so a killed run never leaves a half-file.
- **Seams:** to swap in real backends (Verovio, ONNX detector, tar packer),
  keep the function signatures, the `(seed, …)` stream derivation, the
  `write_artifact` tmp+rename helper, `mark_done` calls, and on-disk
  naming/layout. Downstream stages read fixed filenames — change readers and
  writers together.
- **CLI shape:** `-c/--config`, `--resume`, `--from`, `--only`, `--dry-run`,
  `--seed`, `--fail-after`, `-h/--help`; exit `0` ok / `1` config error /
  `2` fail-after trip. Orchestrator configs: paths resolve relative to the
  config file. `test-musicxml.toml`-style throwaway configs live next to
  `config.example.toml` — don't commit secrets in them (there are none).
- **Deps:** orchestrators stay std-only. Stage crates may use crates.io
  (`image 0.25` with `png`+`jpeg`, `clap 4` derive, `zip 2` for `.mscz`,
  `kamadak-exif`, `rand`/`rand_chacha`, `serde`, `sha2`, `walkdir`).
- **Licenses:** keep the stack permissive (MIT/BSD/Apache). No AGPL/GPL
  tools in the path (no Ghostscript, no MuseScore binary dependency).
- **Notation:** MusicXML 3.1 is the default emit; 4.0 is opt-in with an
  explicit loss table. `assembly-serialize pack` is the reference for
  `.mscz` output. Fixes-per-page vs manual-entry time is the correction-UX
  gate; `fixes-per-page` beats raw SER for product decisions.
- **Development loop (`train -> learn -> change -> repeat`):** The core
  engineering loop of this repo is **train → evaluate on `suite31_hw`
  (our modified 45-piece Suite31 benchmark: 31 printed + 14 real historical
  handwritten manuscript additions, `clean` + `photo` = 90 pages via
  `suite31_hw_manifest.json` — never the legacy 31-piece printed-only
  manifest) → forensic learn (`.agents/skills/learn-from-suite31/SKILL.md`:
  eyeball worst vs best printed & HW, run CPU ablations, diagnose root
  causes) → change (fix code/data/tests, commit & push) → repeat**. Never
  launch a new training run without completing the forensic learn step
  first.
- **Git checkpoint before training:** Always run tests, commit, and push
  (using `Gregor Eesmaa <6941817+gregoreesmaa@users.noreply.github.com>`)
  after every set of code/doc changes **before** launching a new GPU
  training run, so every run maps cleanly to an exact pushed commit.

## Split-off companion repos

- The three sibling repos split off from this project are **OK to edit
  directly from within tasks in this repo** whenever engine fixes, new
  features, or optimizations belong upstream (never duplicate their engine
  code back into `sheet-music-converter`):
  - `C:\Users\grego\projects\mscz-rs` (Rust MusicXML/MSCX/MSCZ + `bekern` crate)
  - `C:\Users\grego\projects\hw-score-sim` (Python handwritten score simulator)
  - `C:\Users\grego\projects\doc-photo-degrade` (Rust phone-photo degradation engine)
- Run the companion repo's test suite before committing/pushing, and use
  git attribution `Gregor Eesmaa <6941817+gregoreesmaa@users.noreply.github.com>`.

## Session docs (read these first, keep them current)

- `todo.md` — the plan of record: verdicts, workstreams, experiment log.
- `goal.md` — the active session goal with gates and standing rules.
- `progress.md` — the current measured state: what is done and verified
  on disk, what is queued, what is next. Numbers there are measured, not
  projected.
- `docs/UNIOMR.md` — UniOMR neural OMR architecture, training, and eval reference.
- `.agents/skills/learn-from-suite31/SKILL.md` — workspace skill for the
  post-evaluation forensic learning loop on `suite31_hw` runs.
- Read `todo.md`, `goal.md`, and `progress.md` at the start of any session or
  goal continuation before touching code. Update `progress.md` whenever a worker
  result is verified on disk (files changed, test counts, metric values, open
  gaps). Update `todo.md` when a workstream changes state. Never edit `goal.md`
  gates without an explicit user decision.

## External consultation

- Antigravity (`agy` CLI at `C:\Users\grego\AppData\Local\agy\bin\agy.exe`,
  v1.5.3, seen 2026-11-07): a stronger-model consultant the user offers
  for advice/ideas. Consultation only — never delegate implementation work
  to it. Usage: `agy -p "<self-contained question>"` (single
  non-interactive turn; add `--effort high` for hard calls). Rate-limited
  with ~5 h cooldown: spend calls on high-leverage verdicts (architecture
  decisions, design-doc review), not routine debugging; one focused
  evidence-packed question per call. Record every consultation in
  `progress.md` (question, verdict, effect on the decision).

## Current state / known gaps

- Stage crates do the real work; orchestrators only sequence. `cargo test -p
  photo-degrade` covers determinism + bend-mode recording + warp-fit/rotation
  bounds + GPU-backend parity (21 main tests, 9 `gpu_backend` tests);
  `cargo test -p score-render` covers fonts/style/combo/
  slug/job-file/stem-clearing/page-sizes (17 tests); `cargo test -p
  pipeline-orchestrator` covers JPEG magic/dims + determinism.
  `cargo test -p degrade-gpu` covers the 7 WGPU kernels + light+CA and
  bend+perspective fusions + fused bench, plus three test-verified but
  not yet pipeline-wired modules: `chain.rs` (resident single-submit run
  of all 8 dense stages, bit-exact vs separate calls), `sparse.rs`
  (dense halves of ink/wear/capture with bounded CPU parity), `sepblur.rs`
  (separable two-pass blur, u8 diff ≤ 1 vs single-pass). 28 passed +
  3 ignored (2 benches + 1 timing probe). The noise uniform is 16 floats
  (64 B `Nse` struct) — an 8-float buffer fails bind-group validation.
  `converter/.../from-degraded10.toml` runs the converter over the 10
  realistic degraded JPEGs. `data-gen/.../samples10.toml` is the small
  seeded demo config (placeholder pixels, not for training).
- MuseScore headless: NEVER set `QT_QPA_PLATFORM=offscreen` — the stock
  Windows install ships only `qwindows.dll`, and a missing platform plugin
  hangs the process on an invisible dialog (plus the zombie then swallows
  later invocations via single-instance forwarding). The native platform
  exports headlessly. Every score-render MuseScore invocation runs with a
  per-process TEMP/TMP shard (`score-render-ms-<pid>`) so parallel runs get
  separate single-instance sockets and a stale lock can't silently swallow a
  launch. Always verify no `MuseScore4` process lingers before re-running.
- score-render batch mode (`-j` job file, one MuseScore launch per style
  group, default on): `--musescore-dpi N` (default 300 ≈ final training size,
  ~16x fewer pixels than the 1200 dpi default; `-r` is NOT passed in job
  mode, so without the flag every page renders at 9924x14028). Before each
  export, previous outputs for that stem are deleted
  (`clear_stem_outputs`) — without this, re-renders into a dirty out dir are
  invisible to the path-diff and wrongly fall back to stub. Regression: all
  500 big500 scores render via `musescore-batch`, zero stubs.
- photo-degrade geometry never loses notation: perspective warp is fit per
  corner (inward motion kept whole, outward trimmed to a 0.4% bleed sliver;
  `warp_fit_scale`/`warp_out_px` recorded), barrel distortion is rescaled to
  the same sliver budget (`dist_fit_scale`), inward (pincushion) pull is
  folded outward at sample time, rotation is +/-10 deg with the
  canvas expanded to fit (dark desk fill, `rotation_deg` recorded). Photo
  framing (pre-rotation) samples an inset (desk pad 0.8–4.5% of min-dim
  per side, ~70% of pages) or a tight crop (frame cuts into the page,
  ~30%); tight cuts are clamped per edge to the measured quiet margin
  minus a safety buffer (>= 3 px), so notation never leaves the frame.
  `frame_mode`, `frame_pad_px`, `frame_cut_px`, `frame_notation_bbox`,
  `frame_quiet_px`, `frame_safe_px` are recorded. BG fill
  outside the page is paper tone. Crease lines (`fold_creases`), tape
  residue rects (`residue_rects`), and partial-span edge tape
  (`tape_edge/start/len_frac`, never full-width top band) are recorded in
  params. Params files carry the run suffix
  (`<stem>{suffix}_params.json`) so parallel passes never share them.
- `converter/.../out/`, `data-gen/.../out/`, `samples/`, and
  `test-musicxml.toml` are local run scaffolding from earlier verification,
  not fixtures — don't treat 1×1 PNGs or `.page` files as real data.
- Dev machine: Windows 11, Ryzen 7 7800X3D, 64 GB RAM, RTX 4080 SUPER 16 GB.
  Python is installed in `.venv\Scripts\python.exe` (Python 3.12 + PyTorch
  2.14.1 CUDA 12.6 on RTX 4080 SUPER) and used for neural OMR training/eval
  (`tmp-train/mvd/uni_omr.py`, `eval_uni_suite31.py`; see `docs/UNIOMR.md` and
  `.agents/skills/learn-from-suite31/SKILL.md`). CPU inference is the default,
  GPU optional.
- **Windows + RTX 4080 SUPER GPU gotchas (do not regress):**
  - Keep **CUTLASS `EFFICIENT_ATTENTION`** for SDPA on variable `(mem, T)`
    batches — NEVER force `CUDNN_ATTENTION` (cuDNN 9.10 recompiles a CPU
    execution plan ~640 ms/shape, causing a 49× slowdown under dynamic routing
    + multi-scale + variable sequence lengths).
  - NEVER call `.pin_memory()` inside `ThreadPoolExecutor` background threads
    on Windows WDDM (`cudaHostAlloc` takes a global CUDA driver lock and stalls
    the main GPU thread).
  - Always launch background GPU runs with `.\.venv\Scripts\python.exe -u`
    (unbuffered stdout), keep VRAM `< 14 GB` (Windows WDDM silently swaps to
    system RAM instead of throwing OOM), and use PowerShell `;` (never `&&`).
- **Active external `D:\tmp` data & checkpoint map (outside git — never delete):**
  - Best checkpoint: `D:\tmp\mvd-runs\uni-omr-iter6-clean\uni_omr_best.pt`
    (`0.7014` PR clean / `0.7213` PR photo Note F1 on `suite31_hw`).
  - `suite31_hw` OOD reports: `D:\tmp\uni-omr-suite31-ood\`.
  - Clean Iter-7 full-page manifest: `D:\tmp\uni_fullpage_iter7.jsonl`
    (`24,392` rows, `15.3%` real HW, `0` OOV, `0` eval overlap).
  - Clean Iter-7 HW strips: `D:\tmp\real-hw-strips-iter7\` (`6,213` strips,
    `35.3%` real HW, `0` barline corruption, `0` `mozart-k458`).
  - Real HW sources: `D:\tmp\debussy-omr\train_pages\` (`84` uncomposed
    archival full pages), `D:\tmp\real-hw-train-debussy` (`440` strips),
    `D:\tmp\real-hw-train` (`36` Mozart SQ strips; exclude `hw-mozart-k458`),
    `D:\tmp\unlabeled-10k-real\rism_hw` (`5,707` unlabeled RISM pages).
  - **QUARANTINED (DO NOT USE FOR FULL-PAGE / TRAINING):**
    `D:\tmp\corpus10x-NC\cvc-muscima` (`CC-BY-NC`), `D:\tmp\rism-pdm-hw-intl`
    (opening-staff incipits only, NOT full-page aligned), and the `80.8%`
    barline-corrupted strips in `D:\tmp\hw-sim-train-v03-clean\stage2_strips\`.
- photo-degrade `--gpu` (off by default) runs the dense stages
  (paper grain, page bend, defocus blur, perspective, rotation,
  lighting + chromatic aberration fused in one round-trip, noise) on the
  `degrade-gpu` WGPU backend with identical gates/fills/knobs; sparse
  ink-aware effects, paper aging, lens distortion, capture, wear, and
  downscale stay on CPU. Grain and sensor noise use GPU hash streams
  (distinct stream ids per stage, not ChaCha), so pixels differ from CPU
  runs at the same seed — deterministic per backend, recorded in the
  `backend` params field; same-seed cross-backend knob equality ends at
  stage 1 by design. One shared `GpuContext` serves all worker threads;
  `GpuContext::try_new()` (fallible) keeps `--gpu` from panicking on
  GPU-less machines. Release end-to-end (10 full-size pages): fused GPU
  4.3 s (4.3–5.0 s across runs) vs CPU 5.4–5.5 s vs pre-fusion GPU 5.9 s;
  single-submit light+CA and bend+perspective fusions are bit-identical
  to the separate calls. Wiring `blur_sep` into `--gpu` measured 6.1 s
  (own device + two round-trips) and was reverted; the resident-chain /
  sparse-stage pipeline integration is still open (needs shared-device
  submit + params/knob plumbing + end-to-end determinism proofs).
