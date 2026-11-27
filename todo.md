# Master Task Checklist — `todo.md`

Target: [goal.md](goal.md) | Measured State: [progress.md](progress.md)

Detailed task specifications are broken down by phase in the [`todo/`](todo/) folder so each task is **small, self-contained, unambiguous, and verifiable with a single command**—designed so any coding agent can execute one task at a time without guesswork.

---

## Mandatory Execution Rules for the Agent (Read Before Doing Anything)

1. **Strict Sequential Order:** Complete one `[ ]` task at a time in exact order (`T1.6a` → `T1.6b` → `T1.6c` → `T1.6d` → `T2.0` → `T2.1` → ... → `T4.3`). Never start a later task before all earlier tasks are marked `[x]`.
2. **Read the Linked Spec First:** Before starting any task `TX.Y`, open its specification file in [`todo/`](todo/) and read the **Files to Read First**, **Exact Changes to Make**, **Verification Command**, and **Pass Criteria** sections. Follow the spec verbatim.
3. **Verify Before Checking Off:** You may ONLY change `[ ]` to `[x]` in this file (`todo.md`) and its `todo/phase-*.md` file **after** running the task's exact **Verification Command** and confirming exit code `0` and all numeric assertions pass.
4. **Update `progress.md` & Commit Named Files After Every Task:** After each task passes verification:
   - Append the measured result (files changed, test counts, metric values) to [progress.md](progress.md).
   - Mark `[x]` in both `todo/phase-*.md` and `todo.md`.
   - Commit **only the named files** for that task (`git add <files> todo.md todo/phase-*.md progress.md && git commit -m "Complete TX.Y: <short summary>"`). Do NOT run `git add -A` while `tmp-train/hw-synth/.hf-cache/` deletions are unstaged.
5. **Hard Guardrails (Never Violate):**
   - **No Stubs or Faked Metrics:** Never write placeholder/stub `.bekern` files, never hardcode test return values, and never fabricate reference files.
   - **Never Train on Held-Out Data:** Never load any score or page listed in `data-gen/score-corpus/gate20/exclusion.json` (`do_not_train`) or `tmp-train/mvd-data/quarantine.txt` or `data-gen/score-corpus/suite31/` into training.
   - **Keep `gate20` Sealed:** Never modify existing files in `data-gen/score-corpus/gate20/` (`gate_manifest.json` SHA-256 must remain `68a0b070277bda6b2e8fd89565fa8dc3952ac1fb29e25e5403c3d78bc4ed18eb`), and never touch `test-8` pages until the final sealed `T4.3` run.
   - **No Unrectified 1D Column Collapse:** Do NOT replace `MvdNet.proj` with a static `Conv2d(768, d, (32, 1))` on unrectified strips (keep 2D `H/8=32` memory in Phase 1; vertical collapse is only valid after canonical slot rectification in Phase 2).
   - **Domain-Separated Metrics Only:** Never average `clean` and `degraded` accuracy together when evaluating checkpoints; always report `clean` and `degraded` separately.
   - **Python Interpreter:** Always use `.\.venv\Scripts\python.exe` (which has PyTorch + CUDA active). Never use system `python`.

---

## Ordered Task List (`[ ]` = Pending, `[x]` = Verified & Committed)

### Phase 0: Repo Safety & Trainer Hardening (COMPLETE)
Spec file: **[`todo/phase-0-safety.md`](todo/phase-0-safety.md)**

- [x] **T0.1 — Initialize Git Repository and Commit Baseline** ([spec](todo/phase-0-safety.md#task-t01--initialize-git-repository-and-commit-baseline))
- [x] **T0.2 — Harden `tmp-train/mvd/train.py` (Single-Writer Lock, Atomic Saves, Config-Fingerprint Refusal)** ([spec](todo/phase-0-safety.md#task-t02--harden-tmp-trainmvdtrainpy-single-writer-lock-atomic-saves-config-fingerprint-refusal))
- [x] **T0.3 — Add Standing Blank-Image & Shuffled-Image Grounding Probe to `tmp-train/mvd/train.py`** ([spec](todo/phase-0-safety.md#task-t03--add-standing-blank-image--shuffled-image-grounding-probe-to-tmp-trainmvdtrainpy))

---

### Phase 1: Break the Deadlock — Fix Pretrained ConvNeXt `LayerScale`, Eval Crop Parity & `MvdNet` v2.1 (Gates 1 → 3)
Spec file: **[`todo/phase-1-system-decoder.md`](todo/phase-1-system-decoder.md)**

- [x] **T1.1 — Add System-Level `bekern --measures` and `system-breaks` to `converter/assembly-serialize`** ([spec](todo/phase-1-system-decoder.md#task-t11--add-system-level-bekern---measures-and-system-breaks-to-converterassembly-serialize))
- [x] **T1.2 — Build Engraver-Exact System-Crop + Degraded Dataset (`tmp-train/mvd/build_sys_data.py`)** ([spec](todo/phase-1-system-decoder.md#task-t12--build-engraver-exact-system-crop--degraded-dataset-tmp-trainmvdbuild_sys_datapy))
- [x] **T1.3 — Build Held-Out `gate20` Dev System Reference Set (`data-gen/score-corpus/gate20/sys-dev/`)** ([spec](todo/phase-1-system-decoder.md#task-t13--build-held-out-gate20-dev-system-reference-set-data-genscore-corpusgate20sys-dev))
- [x] **T1.4 — Upgrade `MvdNet` in `tmp-train/mvd/model.py` (Anisotropic Stride `8×16` + Apache-2.0 `timm` Pretrained Init)** ([spec](todo/phase-1-system-decoder.md#task-t14--upgrade-mvdnet-in-tmp-trainmvdmodelpy-anisotropic-stride-816--apache-20-timm-pretrained-init))
- [x] **T1.5 — Update `train.py` for System-Strip Batching (`H=256`, `1:3` Degraded Mix) & Pass the 8-Strip Overfit Gate (`E6-sys-overfit`)** ([spec](todo/phase-1-system-decoder.md#task-t15--update-trainpy-for-system-strip-batching-h256-13-degraded-mix--pass-the-8-strip-overfit-gate-e6-sys-overfit))
- [x] **T1.6a — Fix Pretrained ConvNeXt `LayerScale` (`gamma` → `layer_scale`), Add Spatial `LayerNorm` + DC-Centering, and Add Content-Only `--tok-dropout` (`model.py` + `train.py`)** ([spec](todo/phase-1-system-decoder.md#task-t16a--fix-pretrained-convnext-layerscale-gamma--layer_scale-add-spatial-layernorm--dc-centering-and-add-content-only---tok-dropout-modelpy--trainpy))
  - Load all 18 `timm` `stages.{s}.blocks.{i}.gamma` vectors (`.view(-1, 1, 1)`) into `{2*s+1}.{i}.layer_scale` (`178/178` tensors loaded, `0` skipped; fixes the `1e-6` dead-trunk bug), add `self.mem_norm = nn.LayerNorm(d_model)` + spatial mean subtraction in `MvdNet.encode`, set `STRUCT_WEIGHT_IDS` to `('bar',)` only, add `--tok-dropout` on `CONTENT_IDS` only, and exclude 1D `norm`/`bias`/`layer_scale` from AdamW `weight_decay`.
- [x] **T1.6b — Fix `gate20/sys-dev` Tight-Crop Parity (`build_gate20_sys_dev.py`) & Commit Domain-Separated `probe_gate_acc.py`** ([spec](todo/phase-1-system-decoder.md#task-t16b--fix-gate20sys-dev-tight-crop-parity-build_gate20_sys_devpy--commit-domain-separated-probe_gate_accpy))
  - Fix `build_gate20_sys_dev.py` so `sys-dev` strips use tight `[y1 - 2.5*space, y2 + 2.5*space]` and `[x1 - 1.5*space, x2 + 1.5*space]` bounds (matching `build_sys_data.py` instead of pinning `s0` to `y=0` and `s_last` to `y=H_page`), commit `tmp-train/mvd/probe_gate_acc.py` reporting `clean` and `degraded` separately against `train.WHITE` and `swap_other_strip`, and record the re-evaluated `sys-v2f-A/ckpt-30000.pt` clean baseline.
- [x] **T1.6c — Add Non-Excluded 4-Staff String Quartet (`sq*`) Systems to Training Corpus (`tmp-train/mvd/build_sys_data.py`)** ([spec](todo/phase-1-system-decoder.md#task-t16c--add-non-excluded-4-staff-string-quartet-sq-systems-to-training-corpus-tmp-trainmvdbuild_sys_datapy))
  - Render/extract ≥1,000 clean + ≥3,000 degraded 4-staff string quartet system strips from `data-gen/score-corpus/mscx-5000/OpenScore-StringQuartets` (strictly excluding all `gate20/exclusion.json` and `mscx-5000-test-works.txt` stems) so training covers 4-staff systems and alto clef (`clef-C-3`), which make up 50% of `gate20/sys-dev`.
- [ ] **T1.6d — Train `MvdNet` v2.1 on Full Mixed Corpus (`sys-v2g`) & Pass Gate 2 (`>0.22` F1) and Gate 3 (`>0.50` F1)** ([spec](todo/phase-1-system-decoder.md#task-t16d--train-mvdnet-v21-on-full-mixed-corpus-sys-v2g--pass-gate-2-022-f1-and-gate-3-050-f1))
  - Gate 2 MISS at 45k (SER 0.633 vs 0.48; F1 0.524 passes) — committed `6022866`, agy verdict B, route T2 rectification. Stays open; revisit after S13. (T1.6a/b/c verified: `43d9e11` / `cfb367b` / `1db8175`.)
  - Train `MvdNet` v2.1 (`--pretrained --trunk-lr-mult 0.1 --tok-weight 3.0 --tok-dropout 0.20 --batch-size 4 --accum 2`) on the full `1:3` clean+degraded corpus and verify **Gate 2** (`SER < 0.48`, `strict-F1 > 0.22`) and **Gate 3** (`SER < 0.15`, `strict-F1 > 0.50`, full-page `proxy-fixes <= 50`).

---

### Phase 2: Canonical Multi-Staff Slot Dewarping + `HeatNet` Keypoint Fusion + Measure Anchoring (Gate 4: `>0.90` F1, Beat Audiveris)
Spec file: **[`todo/phase-2-hybrid-dewarp.md`](todo/phase-2-hybrid-dewarp.md)**

- [x] **T2.0 — Stage Permanent In-Tree Copy of the 31-Piece Sheetly Benchmark (`data-gen/score-corpus/suite31/`)** ([spec](todo/phase-2-hybrid-dewarp.md#task-t20--stage-permanent-in-tree-copy-of-the-31-piece-sheetly-benchmark-data-genscore-corpussuite31))
  - Copy the 31 clean PNGs, 31 photo PNGs/JPEGs, 31 page-1 GT MusicXML cuts, and `manifest.json` from `%TEMP%\ours`, `%TEMP%\oursph`, `%TEMP%\omrleg`, and `%TEMP%\sheetly\bundle\bundle` into `data-gen/score-corpus/suite31/` with SHA-256 verification so benchmark files cannot be lost on temp cleanup.
- [x] **T2.1 — Build Canonical 4-Slot Staff-Line Rectifier (`tmp-train/mvd/rectify.py`)** ([spec](todo/phase-2-hybrid-dewarp.md#task-t21--build-canonical-4-slot-staff-line-rectifier-tmp-trainmvdrectifypy))
  - Dewarp curved/tilted 5-line staves onto 4 fixed canonical staff slots on `H=256` (`target_sp = 10.0 px`, unused slots filled with paper white `255`, horizontal width scaled by `target_sp / measured_sp` so 3–4 staff widths are never crushed) and pass `test_rectify.py` (`<= 0.5 px` row residual).
- [x] **T2.2 — Fuse `HeatNet` 7-Channel Spatial Keypoints + Staff-Grid Channel into `MvdNet` (`tmp-train/mvd/heat_features.py` + `model.py`)** ([spec](todo/phase-2-hybrid-dewarp.md#task-t22--fuse-heatnet-7-channel-spatial-keypoints--staff-grid-channel-into-mvdnet-tmp-trainmvdheat_featurespy--modelpy))
  - Concatenate `HeatNet`'s 7-channel (`0.982` Note-F1) spatial heatmaps + 1 canonical staff-line reference channel with the 3 RGB image channels (`extra_chans=8`, zero-initialized extra stem weights) and pass unit tests.
- [x] **T2.3 — Barline-Anchored Measure Decoding & `S > 4` Staff Windowing (`tmp-train/mvd/measure_decode.py`)** ([spec](todo/phase-2-hybrid-dewarp.md#task-t23--barline-anchored-measure-decoding--s--4-staff-windowing-tmp-trainmvdmeasure_decodepy))
  - Enforce measure-duration conservation (`PrefixState(strict=True)`), add per-measure fallback decoding, and support `S > 4` staff systems (e.g., 5–7 staff `suite31` pieces) via 4-staff sliding windows merged by measure.
- [ ] **T2.4 — Train Hybrid Rectified `MvdNet` v2.2 & Pass Gate 4 (`strict-F1 > 0.90` on `gate20` Degraded, Beat Audiveris on `suite31`)** ([spec](todo/phase-2-hybrid-dewarp.md#task-t24--train-hybrid-rectified-mvdnet-v22--pass-gate-4-strict-f1--090-on-gate20-degraded-beat-audiveris-on-suite31))
  - Train the rectified + `HeatNet`-conditioned `MvdNet` v2.2 and verify held-out `gate20` degraded `strict-F1 > 0.90` (`proxy-fixes/page <= 15`) and `suite31` photo mean `strict-F1 > 0.88`.
  - Gate 4 MISS (v24: clean 0.71 / degraded 0.30; criterion-2 PASS). Stays open; revisit after S13.

---

### Phase 3: Lossless `bekern+` (Fixing the `0.8765` `suite31` Ceiling) & `mscx-5000` Multi-Font Scale (Gate 5: Beat Sheetly)
Spec file: **[`todo/phase-3-lossless-bekern-scale.md`](todo/phase-3-lossless-bekern-scale.md)**

- [x] **T3.1** — verified + committed `03463b1` (cargo 72, suite 139, oracle 31/31 F1 1.0)
  - Add tick-quantum tolerance (`<= 0.002` beats) in `duration_token` (fixing `cpebach-h186`, `schubert-lindenbaum`, `weber-concertino`), mid-stream `clef-*`/`key*`/`time-*-*` changes, pickup-bar duration handling, `grace` notes, slurs, articulations, dynamics, and repeats, and verify **`31/31` Sheetly GT pieces round-trip with mean `strict-F1 >= 0.995` and `OMR-NED < 0.02`**.
- [x] **T3.2** — dataset + 100k run complete, committed `71909fb` (renders 8/8, sidecars, 79k-pair 100k run, photo F1 0.352; on-the-fly augmentation covered by dropped S8, superseded by S10/S13)
  - Render `mscx-5000` (Lieder + StringQuartets + single-staff sub-extracts) across 6 engraving fonts, extract ≥25,000 clean systems (≥75,000 with degraded variants, `0` `gate20` or `suite31` leakage), and add on-the-fly GPU augmentations to `train.py`.
- [ ] **T3.3 — SUPERSEDED as standalone task (old Gate 5 retired 2026-11-15).** Old Gate 5 (`> 0.975`) is vanity: a 0.975 open-loop reader cannot be verified up to 10× (S6/S7/S15 falsified the family). Intent folds into Gate C (`F1 >= 0.990` open-loop). History kept: Gate 5 MISS at ckpt-100000 (photo 0.352 vs 0.975). Retry vehicle is S13-warm (Step 2 below), not cold S13.

---

### Phase 4: Closed-Loop Render-and-Diff Verifier, Measure Lattice UI & Rust Packaging (Gate 6: 10× Lower Error Rate Than Sheetly)
Spec file: **[`todo/phase-4-verifier-packaging.md`](todo/phase-4-verifier-packaging.md)**

- [ ] **T4.1 — Build Closed-Loop Visual Measure Verifier (`tmp-train/mvd/verify_measures.py`)** ([spec](todo/phase-4-verifier-packaging.md#task-t41--build-closed-loop-visual-measure-verifier-tmp-trainmvdverify_measurespy))
  - Cross-check predicted measure notes against rectified `HeatNet` notehead peaks at test time and re-decode discrepant measures with visual-prior `beam=5`.
- [x] **T4.2 — Add Measure-Atomic Confidence & Top-2 Measure Lattice Sidecar to `converter/assembly-serialize`** ([spec](todo/phase-4-verifier-packaging.md#task-t42--add-measure-atomic-confidence--top-2-measure-lattice-sidecar-to-converterassembly-serialize))
  - DONE 2026-11-14: `--lattice-in` on `bekern2musicxml` + `pack`, `fixes-estimate` subcommand + report field; cargo 71+8 / 46 green; e2e total=1 per=[0,1] on disk.
- [ ] **T4.3 — Export ONNX Pipeline, Wire `converter/pipeline-orchestrator`, and Pass Gate 6 (`>=0.995` F1, `<0.02` OMR-NED)** ([spec](todo/phase-4-verifier-packaging.md#task-t43--export-onnx-pipeline-wire-converterpipeline-orchestrator-and-pass-gate-6-0995-f1-002-omr-ned))
  - Fix `layout-staff` ↔ `symbol-detect` JSON schema, replace `SplitMix64` placeholders in `converter/pipeline-orchestrator`, and pass **Gate 6** (`strict-F1 >= 0.995`, `OMR-NED < 0.02`, median `<= 2` fixes/page, `<= 2.0 s/page`).

---

### Spike backlog (bets farm → spikes, post-100k)
Sources: **[`bets.md`](bets.md)** (100 bets) → **[`spikes.md`](spikes.md)** (S1–S18).
Gate: solo GPU; one spike at a time; record PASS/DROP + evidence in
`progress.md`. Promote at most 2 winners to phase tasks. Specs live in
`spikes.md`; this list tracks state only.

- [x] **S1 — Unconsumed-ink EOS veto** → DROP (over-generation; photo 0.259 vs 0.352)
- [x] **S2 — ILME λ-sweep** → DROP (+0.014, bar +0.03)
- [x] **S3 — Sibling-margin contrastive fine-tune** → DROP (gaps 20 nats vs margin 1.0; run killed)
- [x] **S4 — MWER + CE mixed fine-tune** → DROP (photo 0.276 vs 0.352)
- [x] **S5 — Medusa atom-tuple heads** → DROP without build (joint-accept ~0.21, infeasible)
- [x] **S6 — MaxSAT confusion-lattice repair** → DROP (−4%, bar −25%)
- [x] **S7 — Fixes-per-page action loss** → DROP (sys-F1 +0.02 but photo −0.021, fixes +6%)
- [x] **S8 — Pitch-shear SpecAugment** → DROP (photo −0.012 vs baseline, fixes +10%)
- [x] **S9 — Test-time per-strip adapter** → DROP (photo +0.004, bar +0.02; 40× over time budget)
- [x] **S10 — Scribble/hand-ink data** → PASS (photo 0.356 vs 0.327, +0.029 over control; PROMOTED to training mix)
- [ ] **S11 — Staff-indexed gaze** (queued; weeks build, after S10)
- [ ] **S12 — RNN-T transducer** (queued; biggest build, last)
- [x] **S13 — Full-corpus bundled retrain** (bundle code-complete 2026-11-14; ran from scratch `D:/tmp/t32s13` with stride 4x8 + aux-w 0.3 + coord-w 1.0 + handink, NO simshots (S16b DROP), synthetic-only. EXECUTED 20000 steps = 2500 updates (bs1/acc8 VRAM fallback matched S10's fine-tune update count — 1/10th of the 100k-step plan). Verdict FAIL at bar 2026-11-15: suite31 photo F1 0.0325 vs 0.356 (bar ≥0.456), gate `probe_report_hybrid_t32s13_20000.json`. Miss = step budget, NOT bundle (on-curve vs v2b-2500; all bundle signals live). Bundle verdict UNPROVEN — phase 2 (→100k via --init-from) or pivot, user decision pending.)
- [ ] **S14 — Trunk capacity swap** (CONDITIONAL on Gate C stall, after S18; agy: swap Tiny→Base at 4x8 only if photo plateaus 0.96–0.985)
- [ ] **S15 — Piece-level priors rescore** (slice 1 DONE 2026-11-14: 26×top-4 lattices, byte-identical, all-diverse; slice 2 verdict DONE: no headroom +0.004, LM train NOT justified — parked behind S13)
- [x] **S16 — Real print-and-shoot pairs** (DROPPED entirely 2026-11-14 per user decision: print variant rejected as too costly, sim gate failed +0.02 bar, no real captures taken. S13 proceeds synthetic-only.)
- [ ] **S17 — Measure-local render-and-compare verifier** (slices 1+2 DONE 2026-11-14 TDD: scorer/beam-pick/spread, 273/273 green; 26-strip probes: resid r=-0.13, miss r=-0.29, spread r=+0.46 — no discriminating signal, errors are identity misreads; slice 3 re-decode queued behind S13 retrain)
- [ ] **S18 — Factored event-tuple heads** (PROMOTED ahead of S14/S11/S12 per agy+persona-3: cascade-breaking is load-bearing for 0.9973; flat strings provably cannot reach it)

### Forward program (10×-Sheetly track, Gates A–D in order — supersedes the list below 2026-11-15)
0. **Step 0: coordinate-alignment patch + 2 h diagnostic (Gate A prerequisite, agy red-team).** `_sinusoidal_2d` uses raw grid indices so naive S10→4x8 transfer mis-addresses 2x in y AND 2x in x (4px/8px cells vs 8px/16px); patch scales ys AND xs x0.5 in the 4x8 path (strict refinement, verified against encode() grid math). A/B diagnostic (naive vs patched, zero-shot + 1k updates, gate20/sys-dev): PASS = patched arm >= S10's 0.548 degraded F1 by update 1k; FAIL = 4x8 trunk transfer falsified, train 4x8 only alongside S18, bundle verdict stays negative.
1. ~~S10 verdict~~ DONE PASS → ~~S16~~ DROPPED → S13 cold FAIL (step budget) → **Step 1: fix page stitching + 31/31 scoring (~4 h, zero training)** → Gate A.
2. **Step 2: S13-warm from S10** (`--init-from ckpt-8000`, 22000 fresh micro-steps = 30000 total, binding Gate A verdict photo `>= 0.55`, REQUIRES Step 0 pass) — NOT cold phase-2 (rejected: 3750 updates = 0.35 epochs). **VERDICT 2026-11-20: FAIL AT BAR** (`probe_gateA_final_22000.json`: suite31 photo F1 mean **0.3332** vs bar 0.55 — below even the S10 parent 0.356; clean mean 0.5709 vs 0.75 bar; 31/31 scored ✓; sys-dev degraded sys-F1 0.5168 vs S10 0.548; seal intact `test_pages_touched=0`). Warm bundle could not beat its parent on old data at this horizon. Route per program: Step 3 corpus (DONE — quota composed) → Step 4 S18 → corpus-trained next run; agy consulted on ordering.
3. **Step 3: corpus strata build, ranked (agy genre consult 2026-11-15; 0% model style vector — data + grammar only).** +16k clean / +48k total (1:2 clean : photo+handink sidecars), hard-boundary: NO drum-kit/slash-note staves until Gate D. (1) Solo-piano polyphony 25% (+0.16-0.20: collisions, dense chords, grace bursts — fixes FP4). (2) Mid-piece meter/key/retuning 15% (+0.08-0.11: time-5-8/7-8/11-8, clef swaps, key-change walls — fixes FP3 clock). (3) Regional folk/dance 15% (+0.06-0.08: anacrusis bars, repeat splits incl. light-heavy/double, beaming groups, folk ornaments; needs bar-repeat tokens). (4) 5-7 staff choral 15% (+0.05-0.07: S>4 windows, broken barlines, C-clefs, inter-staff lyrics). (5) Jazz lead sheets 10% (+0.03-0.05: syncopation chains, cross-barline ties, jazz articulations, chord symbols as clutter). Kill gate: 10k-probe clean F1 must reach >= 0.67 (from 0.55); gain < +0.05 freezes data-gen. Eval: tag every suite31 + gate20 piece with its primary stratum (solo-piano / lieder-vocal / chamber-quartet / folk-repeats-irregular / large-ensemble-5plus); per-stratum floors ride along Gates B/C/D. STATUS 2026-11-22: quota renders done (8000 bulk + 598×2 Leland/Emmentaler, spec-gate fails=0); stratum tagger v2 tagged 8000/8000 stems (`D:/tmp/tag_strata_v2.py` → `stratum_tags_v2.json`) — measured piano 0.309 / folk 0.327 / chamber 0.174 / jazz 0.081 / meter 0.051 / choral 0.058 (targets .25/.15/.20/.10/.15/.15; choral+meter need ~2.6–3x upsample, folk downsamples free); Leland-598 is 62% piano + 14.5% choral + 11.9% meter. Remaining: sampler build with upsample map + full 1,196-render strip-build (symbox-extract first: pilot showed 1 silent neural-path skip). STATUS 2026-11-22 (run-4): strip-build LANDED (25,925 clean strips from 23,165 systems via 128th table + bar-anchored column splits; 77,775 degraded; `gate20_leaked: 0`); S2 re-verify ALL PASS (G1 silent 0/dead 0, G3 12/562 unequal all loud-explained, G2 636 all-symmetric music-driven, G4/G5 pass). Remaining now: sampler/materialization refresh on run-4 strips (chunk-aware), chunk markers/tracks, sidecars; Route-A 10k probe queued on operator GPU word.
4. **Step 4: S18 factored event tuples** (measure-atomic decoding + zero-prefix visual readout head (no AR prefix — the comparator dissenter) + per-facet duration/diatonic_step/voice priors) → Gate B/C; S12 stays parked unless S18 misses its gate. BUILD ORDER 2026-11-22 (agy S4): beam-5 + decode-time voice/measure masking first (zero-train control) → frozen-encoder zero-prefix visual probe with KILL tripwires (inner-voice recall <0.75 @ P≥0.85, miss-IoU >0.60) → factored tuples + facet priors → measure-atomic LAST. Oracle-prefix injection gate decides encoder-vs-decoder first (drops persist ≥50% + diatonic errors ≥60% under clean prefixes = fix encoder, stop S18).
5. **Step 5: S14 trunk swap ONLY if Gate C stalls** below 0.990; T4.1 verifier + T4.2 lattice + T4.3 Rust → Gate D.
6. **Music-theory lean (new 2026-11-15, user proposal):** model leans toward diatonically-correct notes when glyphs are ambiguous — NOT via more-of-same-data (scaling falsified) but via (a) diatonic-spelling minimal pairs in training, (b) explicit key/mode conditioning + hard diatonic constraints in the lattice (S18-adjacent), (c) post-hoc corrector on (hyp, ref) pairs. First: quantify theory-recoverable share by mining diatonic violations from (hyp, ref) pairs (queued after warm diagnostic frees the GPU). KILLED 2026-11-22 (agy S5, user-delegated): zero misspellings in 87 strips (pitch F1 == spell F1 == 0.732; errors are diatonic seconds, never enharmonic) — spelling rides free with staff-line pitch + key signature. No spelling models, minimal-pair mining, or post-hoc correctors.
7. **Style-deviation back-off (new 2026-11-15, user proposal):** tunes that fit a style but systematically violate specific patterns must be RECOGNIZED, not normalized — model backs off to an autonomous fallback (literal glyph-level reading + constraint repair: diatonic spelling, bar-duration conservation, voice assignment by rule) instead of hallucinating conformity. NO HUMAN IN THE LOOP — the system ships final notation autonomously; the p95 time budget (<= 5.0 s/page) IS the auto-repair compute budget. Architecture: reader stays style-BLIND and pixel-grounded; a SEPARATE music-LLM prior is KILLED as specified (agy red-team 2026-11-15: a fluent-but-wrong note is by definition already fluent under P_LLM, so a text-only prior can only punish correct literal readings and love fluent hallucinations — comparator polarity inversion). INSTEAD, two mechanisms with conditionally INDEPENDENT errors: (1) INSIDE S18: a zero-prefix visual readout head (no AR prefix, HeatNet keypoints + render-and-diff ink residual) — the ONLY allowed dissenter from the AR decoder, because its errors are independent of AR cascade errors; per-facet S18 priors for duration / diatonic_step / voice (accidentals are glyph evidence, never priors). (2) INSIDE T4.1 (Gate D only): render-and-diff ink-residual flagging as the visual witness. Rationale unchanged: baked-in conditioning would override genuine deviations; the fix is architectural independence (zero-prefix visual readout), not a scalar weight. Falsifier: fluent-but-wrong mining — the AR-with-prefix vs visual-only-head comparator must auto-fix a majority with zero human review, else kill the anti-normalization path and fall back to literal readings + S18 factorization alone. Sits with verifier track (Gate D, verifier = repairer not flagger); the 100:1 pixel-beats-prior rule stands, enforced by architecture rather than by weight.
8. Revisit T1.6d / T2.4 only by supersession (unrectified trunk is dead-end scaffolding; close, don't retry).
9. **Ruler replacement rollout (notation-as-written, user decision 2026-11-21).** `notation-f1` built+TDD+committed (spelling-exact B#≠C, staff/voice/rest/tuplet/attachment/headers priced; strict ports frozen) + **2026-11-22 gaps CLOSED** (dynamics/hairpins P/R, measure-scoped mid-stream clef/key/time `measure_header_match_rate`, explicit `grace_f1` + LCS distinction; `eval-harness` 65/65, `assembly-serialize` 89/89) → rescore 87 strips + suite31 under both rulers → confirm re-based gate bars.
10. **Bar-reset decomposition + paired suite31 counterfactual (G1 gate, CPU-first).** Per-measure-pooled `drift_f1` vs full-identity `notation_f1` on ckpt-22000 (gap = barline/drift share); render suite31 GT clean + synth-degraded, score {real, synth, clean} × {global, drift} (branches A/B/C per agy S3).
11. **Fast training stack + compact sidecars LANDED (2026-11-23, ~2.75× faster, disk blocker cleared).** Legacy-default-tolerant `check_resume` DONE; single-pass shared `MvdNet.loss_shared` (1× encode + 1× decode for `CE+aux+coord`) + `fused=True` AdamW + `--bf16` + `--channels-last` + `--width-bucket` + `--prefetch 4` + `--batch-size 6` (`20.9 strips/s`, `11.9 GB` VRAM) + 16-stage physical & full-page off-center 3D geometric per-epoch GPU aug (`--epoch-aug-p 0.8`, clean floor untouched) wired as defaults in `train.py` + `scripts/run-train.ps1`. Compact `.hybrid.npz` (`compact_sidecar.py`: 321.9 GB → 25.0 GB PNG / 16.5 GB WebP, or 0 GB on-the-fly) validated in live 8-step GPU trial; 246.64 GB of legacy `.hybrid.pt` deleted. Route-A 10k probe only after G1/G2 preconditions. **First-GPU-use order (agy): (a) bar-reset decode-side eval ~15-30 min inference-only → (b) S18 zero-prefix falsifier (0.1-0.2 d, >3x bar) → (c) 10k probe.** Dual-ruler (c) gates — **ENFORCE RUTHLESSLY** — KILL at 10k: strict clean < 0.67 OR photo gain < +0.08 (photo < 0.413), notation clean < 0.49 OR photo gain < +0.06 (photo < 0.308); PLUS 2026-11-22 probe-hardening (agy S3): gate OOD-21 separately (photo >= 0.36, in-domain regression <= -0.02) + dropout-only control on old mix (multi-lever confound); 1k/2.5k plasticity tripwires (old-clean drop >0.08 = optimizer shock → LR warmup/freeze-decoder; new-strata TER >2.5x = lock-in → kill does NOT indict dataset → cold-start fallback); paired clean→photo gap must narrow-or-hold (Δ<=+0.02) + font-holdout stratum. PROMOTE: strict ≥ 0.72 / ≥ 0.50 AND notation ≥ 0.52 / ≥ 0.37.
12. **Neural-first lock, multi-engraver diversity & shipping discipline (2026-11-22/23, user decision).** Target locked: autonomous, permissive-licensed (MIT/BSD/Apache) phone-photo → `.mscz` neural Rust pipeline beating Sheetly. Audiveris fork / data-augmentation / neural-restorer→Audiveris paths are **CLOSED** (`X1` Docker black-box benchmark only). Binding sequence: **`Route-A` 10k probe (ruthless kill gate) → `I1` label-normalization audit → `S18` factored heads (`s18_factored.py`) → multi-engraver Verovio/LilyPond expansion (`multi_engrave.py`)**. Milestone & compute discipline: **beat Audiveris first** (Gate B: `> 88.43%` clean / `> 50.43%` photo) before chasing 10× Sheetly; validate Route-A + S18 on local RTX 4080 first, then rent burst H100s post-Gate-A/B; treat Gate C (`~0.98–0.99`) as a viable v1 shipping milestone ahead of Gate D (`0.9973`). GitHub clean `main` branch live at `https://github.com/gregoreesmaa/sheet-music-converter`.
Old order (retired): S10 → S16 → S13-cold → T3.3(Gate 5) → S17 → S14 → S11/S12/S15/S18 → T4.3(Gate 6).

### Companion tracks (parallel, CPU-only — do NOT block gate order; one item at a time within each track)
Sources: file-backed workflow findings (`D:/tmp/omr-trimmed/*.md`); public numbers from the Sheetly vendor benchmark (Nov 2026).

- [ ] **I1 — Label-normalization audit (workflow lever #1).** Audit our bekern target form for non-determinism (duplicate surface forms for identical notation); measure the normalization gap on the 87 strips. Biggest published single lever (Transcoda OMR-NED 18.71→82.51 without it). Queued immediately behind the Route-A 10k verdict (before full S18 training); zero-train, CPU-only, no architecture change.
- [ ] **I2 — Cheap-affine-augmentation gap check (workflow lever #8).** Test whether a Zeus-style affine/photometric stack closes any of our synth→photo gap on frozen ckpt-22000 before touching the photo-degrade recipe. Zero-train, CPU-only.
- [ ] **C1 — Lean-tree policy (ongoing).** Delete only on zero live references + proven regenerability; untracked user files never touched; last-5-commit protection. Suites green after every removal. (Landed 2026-11-22: `7669e93` + `55fe9bf` + `3eb7886`, ~1,930 files, 22 tests green.)
- [ ] **X1 — Docker black-box evals of oemer + Audiveris on suite31.** Run both engines as black-box containers over `suite31/clean` + `suite31/photo`, score with our `strict-f1` + `notation-f1` harness for directly comparable numbers. AGPL Audiveris stays container-black-box only — never linked, vendored, forked, or placed in the inference/training path (fork/augmentation/restorer-hybrid explicitly CLOSED 2026-11-22). Recorded public baselines: oemer 38.45/9.81, Audiveris 88.43/50.43.
- [x] **X2 — Write `bets+v2.md`.** Best-directions doc fusing the 10 workflow levers with spike evidence (S1–S10 DROP/PASS record): what to do, what to kill, and the named gaps that must be re-read before citing. DONE 2026-11-22 (`bets+v2.md`, 108 lines, content-checked).

---

### UniOMR Full-Page Track & Iter 7 Workstream (2026-11-24 / 2026-11-25)

#### Completed Milestones
- [x] **4-Repo Optimization Sweep & Pushes** — DONE 2026-11-24/25: `mscz-rs` (`ec51183`, 146/146 + WASM), `hw-score-sim` (`80708c0`, 111/111, `6.2–9.7×` faster `clean_paper=True`), `doc-photo-degrade` (`2305fd8` / `v0.3.0`, all 6 gates), `sheet-music-converter` (`staff_unet2d.py` OOM + `--sp-mode auto` fix, `score-render` European metric `--musescore-page-sizes`, 17/17 tests).
- [x] **Stage-1 Controlled 2×2 Ablation (`D:\tmp\ablation-stage1-2x2\summary.json`)** — DONE 2026-11-24: Arm C (`Printed + 1,153 Hero HW`, Unsup OFF) hit new best 2-stage Printed Clean `0.7482` (`+6.3%` vs base); Unsup EMA hurt `-4.7%` to `-8.0%`.
- [x] **UniOMR Iter 1 → Iter 6 (`D:\tmp\mvd-runs\uni-omr-iter6-clean\`)** — DONE 2026-11-25: Fixed inverted `InkTokenRouter` (`23.8% → 95.2%`), `128×` `_sinusoidal_pos2d` aliasing, dark-desk/perspective/1-bit aug, `24,577`-page balanced 1/2/3/4-staff manifest, `max_seq_len=2560`, `4×8` stride, `MixStyle2d`, `CASS`, `SWAD`, CUTLASS `EFFICIENT_ATTENTION` (`11.5–14.7×`), 6-thread async prefetcher. `suite31_hw` (`1280px`, `3072` tok, `FSA + MTP-4`, `447 tok/s`, `100%` printed DecOK): **Printed Clean `0.7014` (`+0.415`) | Printed Photo `0.7213` (`+0.681`, beats best 2-stage `0.6949`!)** | HW Clean `0.0101` | HW Photo `0.0023`.
- [x] **UniOMR Evaluator (`eval_uni_suite31.py`) & Post-Eval Forensic Skill (`.agents/skills/learn-from-suite31/SKILL.md`)** — DONE 2026-11-25: Standardized `suite31_hw` apples-to-apples Strict Note F1 eval + 9-scout forensic audit workflow.

#### Ranked Iter 7 Workstream (Post-Iter-6 9-Scout Audit)
1. [ ] **U7.1 — Fix `grammar.py` `CHORD` loop bug** (eliminate infinite/spurious chord-loop emissions in FSA decoding).
2. [ ] **U7.2 — Upgrade `InkTokenRouter` (`0.50` cap + border-ring mask)** (raise keep cap from `0.25 → 0.50` and mask border-ring leak tokens on dense/5–7 staff pages).
3. [ ] **U7.3 — Calibration & Domain Generalization (`aug_prob=0.50` + domain-mixed `MixStyle`)** (ensure printed & HW features mix across domains during `MixStyle2d` and raise online photo/paper aug rate).
4. [ ] **U7.4 — Option D1 Vocab Expansion (`182 → 204`)** (add missing `bekern+` expressive/meter tokens while keeping base IDs stable).
5. [ ] **U7.5 — Quarantine & Fix `hw-sim-v03` Barline Corruption (`80.8%` affected)** (purge corrupted synthetic HW barlines and regenerate clean `hw-score-sim` pages with valid barlines + expressive `bekern+` tokens).
6. [ ] **U7.6 — Upsample Real Handwritten Data to `15–18%` of Training Steps** (up from `1.17%`: compose and wire `84` uncomposed Debussy full pages + `497` `rism-pdm-hw-intl` + `440` Debussy strips + landscape OpenScore string quartets, `0` eval overlap).
7. [ ] **U7.7 — Rebalance 5–7 Staff Systems & Rare Meters (`4/2`, `3/2`, `1/32`)** (close remaining printed `suite31` tail errors).
8. [ ] **U7.8 — Train & Evaluate UniOMR Iter 7 on `suite31_hw`** (run `eval_uni_suite31.py` + `.agents/skills/learn-from-suite31` forensic loop).
9. [ ] **U7.9 — Future Browser Student (`~20M` Distilled Model)** (post-Iter-7 distillation of UniOMR teacher into a `~20M` WASM/WebGPU browser-deployable student).

---

### Fresh-Session Handoff, `D:\tmp` Map & Windows/GPU Gotchas (2026-11-25)

**Primary Goal (User-Specified):** **`UniOMR` Strict Note F1 `>= 0.90` across ALL scores** on the modified 45-piece `suite31_hw` benchmark (`printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`).

**Already implemented & CPU-tested in working tree (uncommitted):**
- `tmp-train/mvd/data.py`: Option D1 vocab appended (`182 -> 237`, IDs `0..177` intact).
- `tmp-train/mvd/grammar.py`: `CHORD` & `grace` capped at `<= 8`, `BAR_REST` blocked on full/non-empty bar, `cycle_forbidden()` extended to period `1..24`.
- `tmp-train/mvd/eval_uni_suite31.py`: backward-compat `cfg.vocab_size`-aware special IDs (`182` & `237`), `cycle_forbidden()` enabled in both `strict=False` & `strict=True`, clean event-boundary trim on `max_len`.
- `tmp-train/mvd/uni_omr.py`: lossless tail-relocating `load_uni_weights_compat`, border-cliff-proof relative-contrast `InkTokenRouter` (`local_bg > 0.35`, `quantile(0.99)`), clean-page scale jitter + archival parchment/iron-gall/verso-bleed/colorbar/blank-staff aug, domain-mixed `MixStyle2d`.
- Verified: `test_uni_omr.py` (`21/21`), `test_uni_decode.py` (`11/11`), `test_grammar_splitbar.py` (`9/9`).

**Immediate Next Sequence (in order):**
1. **Approved Hybrid B+C Repo Restructure:** Root `Cargo.toml` (only the 4 active crates: `runtime/assembly-serialize`, `eval/eval-harness`, `datagen/score-render`, `datagen/photo-degrade`) + root `pyproject.toml`; shallow workflow slices `uniomr/`, `eval/`, `datagen/`, `runtime/`, `web/`, `benchmarks/` (frozen `suite31/` & `gate20/`), `docs/` (`UNIOMR.md`, `architecture/`, `runbooks/`, `phases/`, `archive/`), and `legacy/` (quarantined 2-stage MVD + 8 retired crates, read-only). Verify `cargo test --workspace` + Python tests green.
2. **Git commit & push** (per `AGENTS.md` rule: `Gregor Eesmaa <6941817+gregoreesmaa@users.noreply.github.com>`) BEFORE launching GPU training.
3. **5-second GPU preflight + Launch UniOMR Iter 7** warm-started from `D:\tmp\mvd-runs\uni-omr-iter6-clean\uni_omr_best.pt` using `D:\tmp\uni_fullpage_iter7.jsonl`, `--max-keep-ratio 0.45`, `--aug-prob 0.50`, `--fullpage-frac 0.65`, `--bf16`, `python -u`.
4. When Iter 7 finishes, `.agents/skills/learn-from-suite31/SKILL.md` triggers automatically (Step 0: run 45-piece `suite31_hw`!).

**Active `D:\tmp` Map (external to git — NEVER delete):**
- Latest best checkpoint: `D:\tmp\mvd-runs\uni-omr-iter6-clean\uni_omr_best.pt` (`0.7014` PR clean / `0.7213` PR photo Note F1)
- Suite31 OOD JSON reports: `D:\tmp\uni-omr-suite31-ood\`
- Iter-7 clean/balanced/real-HW manifest: `D:\tmp\uni_fullpage_iter7.jsonl` (`24,392` rows: `83.9%` printed w/ 5–7st & rare-meter upsampling, `15.3%` real HW incl. `84` uncomposed Debussy archival full pages + `180` landscape 4-staff real HW, `0.8%` bar-clean `hw-sim`; `0` OOV, `0` eval overlap)
- Iter-7 clean HW strip dir (`--hw-strip-dirs`): `D:\tmp\real-hw-strips-iter7\` (`6,213` strips: `35.3%` real HW + `4,021` bar-clean `hw-sim`, `0` corrupted, `0` `mozart-k458`)
- Real HW assets: `D:\tmp\debussy-omr\train_pages\` (84 uncomposed archival full pages), `D:\tmp\real-hw-train-debussy` (440 strips), `D:\tmp\real-hw-train` (36 Mozart SQ strips; exclude `hw-mozart-k458`), `D:\tmp\rism-pdm-hw-intl` (497 PDM-1.0 **opening-staff incipits only** — DO NOT use for full-page!), `D:\tmp\unlabeled-10k-real\rism_hw` (5,707 unlabeled RISM).
- **QUARANTINED (DO NOT USE):** `D:\tmp\corpus10x-NC\cvc-muscima` (CC-BY-NC), and the 80.8% barline-corrupted strips in `D:\tmp\hw-sim-train-v03-clean\stage2_strips\`.

**Windows + RTX 4080 SUPER (16 GB) Hard-Won Gotchas:**
- **SDPA:** Keep **CUTLASS `EFFICIENT_ATTENTION`** (NEVER `CUDNN_ATTENTION` on variable `(mem, T)` batches — cuDNN 9.10 recompiles a CPU plan ~640 ms/shape = 49× slower).
- **Prefetcher:** NEVER call `.pin_memory()` inside `ThreadPoolExecutor` background threads on Windows WDDM (global `cudaHostAlloc` lock stalls GPU).
- **VRAM / Logging:** Always run `.\.venv\Scripts\python.exe -u` (unbuffered), keep VRAM `< 14 GB` (Windows WDDM silently swaps instead of OOMing), and use PowerShell `;` (never `&&`).

---

### Queued 2027–2028 Frontier Trials (Post-Iter 10 / Post-v2)

Standing frontier prototypes built & CPU-verified ahead of schedule; queue on GPU **only after** Iter 10 / UniOMR v2 closes out (one solo-GPU trial at a time; promote only on strict `suite31_hw` GO pass).

1. [ ] **F1 — Render-and-Diff Visual RLVR (`tmp-train/mvd/trial_2027_render_diff_rlvr.py`)**
   - **Idea:** Closed-loop verifiable RL (GRPO / RLVR) combining `bekern` grammar validity + bar-duration conservation + fast render-and-diff ink-IoU reward against the input page so the decoder optimizes pixel-level notation fidelity directly on unlabeled / hard OOD pages.
   - **Ready-to-run commands:**
     - CPU unit test: `$env:CUDA_VISIBLE_DEVICES=""; .\.venv\Scripts\python.exe -m unittest tmp-train/mvd/test_trial_2027_render_diff_rlvr.py -v`
     - GPU trial (post-Iter 10): `.\.venv\Scripts\python.exe -u tmp-train/mvd/trial_2027_render_diff_rlvr.py --ckpt <best_uniomr_ckpt.pt> --out-dir D:\tmp\mvd-runs\trial-2027-rlvr`
   - **GO / NO-GO Criteria:**
     - **GO:** `suite31_hw` Strict Note F1 improves `>= +0.015` on `printed_photo` OR `hw_photo` with `<= -0.005` regression on any other split, `0` grammar/DecOK regressions, and VRAM `< 14 GB`.
     - **NO-GO (KILL):** Gain `< +0.010` after reward-convergence window, reward hacking (ink-IoU rises while Strict Note F1 drops), or wall-clock overhead `> 3×` supervised fine-tuning per update.

2. [ ] **F2 — Test-Time Training / Per-Page Adaptation (`tmp-train/mvd/trial_2028_ttt_adapt.py`)**
   - **Idea:** Lightweight test-time adaptation (TTT / LoRA or LayerNorm-only self-supervised entropy + masked-patch / render-consistency adaptation) for `1–5` gradient steps per hard OOD/manuscript page before decoding, resetting weights between pages.
   - **Ready-to-run commands:**
     - CPU unit test: `$env:CUDA_VISIBLE_DEVICES=""; .\.venv\Scripts\python.exe -m unittest tmp-train/mvd/test_trial_2028_ttt_adapt.py -v`
     - GPU trial (post-Iter 10): `.\.venv\Scripts\python.exe -u tmp-train/mvd/trial_2028_ttt_adapt.py --ckpt <best_uniomr_ckpt.pt> --manifest data-gen/score-corpus/suite31/suite31_hw_manifest.json`
   - **GO / NO-GO Criteria:**
     - **GO:** `suite31_hw` `hw_clean` or `hw_photo` Strict Note F1 improves `>= +0.020` (or `printed_photo` `>= +0.015`) with `<= -0.003` regression on clean printed pages AND added latency `<= +1.5 s/page` (staying inside the `<= 5.0 s/page` p95 autonomous repair budget).
     - **NO-GO (KILL):** Gain `< +0.010`, catastrophic drift on single-page adaptation, or latency `> 3.0 s/page` overhead.

3. [ ] **F3 — Masked Block-Diffusion Measure Refinement (`tmp-train/mvd/trial_2028_block_diffusion.py`)**
   - **Idea:** Lightweight bidirectional 4-layer Transformer denoiser (`BlockDiffusionMeasureRefiner`, `d_model=384`, `n_heads=6`, `5.69M` params, `< 6M` budget) trained with MDLM/LLaDA `1/t` weighted masked diffusion (`t ~ U(0.15, 0.85)`) over measure blocks (`L <= 128`) cross-attending to encoder `mem`, resolving low-confidence / duration-conflicting measures bidirectionally in `4` parallel confidence-guided passes instead of `~60` sequential AR steps.
   - **Ready-to-run commands:**
     - CPU unit test (verified `6/6` green): `$env:CUDA_VISIBLE_DEVICES=""; .\.venv\Scripts\python.exe -m unittest tmp-train/mvd/test_trial_2028_block_diff.py -v`
     - CPU smoke check: `$env:CUDA_VISIBLE_DEVICES=""; .\.venv\Scripts\python.exe tmp-train/mvd/trial_2028_block_diffusion.py --smoke`
   - **GO / NO-GO Criteria:**
     - **GO:** On duration-conflicting / low-confidence `suite31_hw` measures, 4-step parallel block-diffusion refinement fixes `>= 30%` of flagged measure errors (net page Strict Note F1 `>= +0.010`), runs `>= 3×` faster than AR beam-5 measure re-decoding, and adds `< 6M` parameters.
     - **NO-GO (KILL):** Measure repair rate `< 15%`, introduces new bar-duration violations on already-valid measures, or fails to beat AR beam-5 at equal wall-clock latency.

---

### Pre-Iter 15 Wave-0 Architecture Proxies (2026-11-27)

- [x] **`i15-37` (`2026-11-27T10-30-i15-37-modegate-dualhead-text-and-chained-mtp`)** — `COMPLETED` → **`ITERATE`** (CONFOUNDED POSITIVE: `ema_val_ce 1.19205 vs 1.29243`, `val_ce 1.03970 vs 1.08365`, `spec_accept_len 1.1319 vs 0.8838` EMA / `1.3443 vs 1.1603` raw, `54.050 MB` INT4 w/ MTP, but Arm 1 fell back to `opt=adamw` via `"base."` prefix bug fixed in `95446af`/`b64fd64`).
- [ ] **`i15-37b` (`2026-11-27T14-30-i15-37b-modegate-cmtp-muon-clean`)** — `QUEUED` (`wave=0`, `LOCAL_ELIGIBLE`): clean matched `opt.mode == "muon+adamw"` + `QATInt4ControllerV2` (`int4_dequant_val_ce`) + `EMA_DECAY=0.99` verification of ModeGate Dual-Head + Chained MTP-4.
