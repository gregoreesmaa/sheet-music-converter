# suite31 & suite31+hw — Frozen Printed & Companion Handwritten OMR Eval Suite

This directory holds the **frozen 31-piece head-to-head printed OMR benchmark** (`suite31_manifest.json`) and its **modified 45-piece handwritten-expanded benchmark** (`suite31_hw_manifest.json`, adding 14 real historical handwritten manuscripts; `suite31_hw_only_manifest.json` for the 14 HW pieces alone).

> [!IMPORTANT]
> **Active `UniOMR` Goal:** Reach **`Strict Note F1 >= 0.90` across ALL 45 pieces in `suite31_hw`** (`printed_clean`, `printed_photo`, `hw_clean`, and `hw_photo` via `suite31_hw_manifest.json`).

---

## 1. Frozen vs. Companion Manifest Rules

| Manifest | Pieces | Mutability | Purpose |
| :--- | :---: | :--- | :--- |
| `suite31_manifest.json` | **31** | **FROZEN (READ-ONLY)** | Head-to-head benchmark vs. Sheetly, Audiveris, and oemer. **NEVER edit this file or any file under `clean/`, `photo/`, or `gt/`.** |
| `suite31_hw_manifest.json` | **45** | Companion (append-only via verified scripts) | Exact 31 frozen printed pieces (identical order) + 14 held-out real handwritten manuscripts (`hw/`). |
| `suite31_hw_only_manifest.json` | **14** | Companion (HW slice only) | Fast evaluation of the 14 held-out real handwritten manuscripts only. |

- **Pinned Frozen SHA-256 (`suite31_manifest.json`):**
  `b58b4a36b70ddf553cc2bbb63396868971a6284a21d5813d6b11891fc8a7b3ac`
- Every evaluation report written by `tmp-train/mvd/eval_sys_gate.py` records `suite31_manifest_sha256` (top-level and in `summary`) and `test_pages_touched: 0` (confirming sealed `gate20/test-8` was untouched).

---

## 2. Exact Stratum Breakdown (45 Pieces)

Never pool `clean + photo` or `typeset + hw` into a single headline metric. `eval_sys_gate.py` tags every piece with `stratum` and emits `summary.suite31_strata` with clean and photo metrics strictly separated:

| Stratum Key | Count | Source & License | Notes |
| :--- | :---: | :--- | :--- |
| `typeset` | **31** | Frozen `suite31_manifest.json` (Sheetly benchmark bundle) | 1–5+ staves: monophonic folk/repeats, lead sheet, piano (classical/romantic/ragtime/atonal), Lieder/aria/choral, string quartets, renaissance 5-voice. Head-to-head comparable to Audiveris (`0.884` clean / `0.504` photo) and Sheetly (`0.974` clean / `0.973` photo). |
| `hw_landscape` | **5** | IMSLP Public Domain holographs + OpenScore `CC0-1.0` | Landscape string quartet manuscripts (`hw-beethoven-op135`, `hw-mozart-k464`, `hw-mozart-k387`, `hw-mozart-k421`, `hw-schubert-d810`). |
| `hw_portrait_debussy` | **4** | `HugoSchtr/debussy-omr` (BnF Gallica), `CC-BY-4.0` | Portrait piano manuscripts (`hw-debussy-vetement-du-blesse`, `hw-debussy-berceuse-heroique`, `hw-debussy-la-plus-que-lente`, `hw-debussy-lindaraja`). |
| `hw_portrait_pdm` | **5** | SBB Berlin `PDM 1.0` + OpenScore Lieder `CC0-1.0` | Commercial-safe portrait Lieder manuscripts, zero Gallica caveat (`hw-hensel-morgengruss-d-b`, `hw-mendelssohn-herbstlied-d-b`, `hw-beethoven-das-blumchen-wunderhold-d-b`, `hw-cornelius-botschaft-d-b`, `hw-hensel-im-herbste-d-b`). |
| `hw_all` | **14** | Union of all 3 handwritten sub-strata | Always reported alongside the 3 individual HW sub-strata. |

All 14 handwritten pieces are **strictly non-NC** (`Public Domain` / `PDM 1.0` / `CC0-1.0` / `CC-BY-4.0`) and **work-level disjoint** from all training corpora (`D:/tmp/real-hw-corpus/`, `D:/tmp/real-hw-train-debussy/`, `D:/tmp/unlabeled-10k-real/`).

---

## 3. Scoring Contract & Comparability Rules

1. **Strict-F1 (`eval-harness strict-f1` vs. `strict-xml`):**
   - `eval_sys_gate.py` scores full-page Strict-F1 via `eval-harness strict-f1` on note-list JSON (`assembly-serialize import` for GT vs. `assembly-serialize bekern-decode` for hypothesis): a note matches iff MIDI pitch is exact, onset-in-bar matches within ±1/16 beat, and duration matches within ±1/16 beat (LCS alignment, rests unscored).
   - **Comparability caveat (documented honestly):** `eval-harness strict-xml` is the exact MusicXML-vs-MusicXML port of the Sheetly reference scorer (including measure-local onset handling for pickup/anacrusis bars), whereas `strict-f1` computes onset-in-bar modulo a uniform bar grid. On pickup pieces (`fiddle-strathspey`, `oneills-tune`, `mozart-k458-1`, `schumann-dichterliebe-2`, `aloha-oe`), `strict-f1` can be slightly harsher than `strict-xml` if an anacrusis shifts bar phase; for final head-to-head publication tables vs. Sheetly/Audiveris, confirm MusicXML pairs with `eval-harness strict-xml` and `eval-harness suite-summary`.
2. **Notation-F1 (`eval-harness notation-f1` — Replacement Ruler):**
   - Spelling-exact (`B# != C`), pricing staff, voice, rests, tuplets, grace notes, dynamics/hairpins, and mid-stream clef/key/time changes.
   - Both reference (`gt-p1-*.musicxml` via `assembly-serialize bekern --measures 1..999999`) and stitched hypothesis route through `bekern2musicxml` so both sides share one voice/staff numbering scheme.
3. **SER, Proxy Fixes/Page, and OMR-NED:**
   - **SER:** `assembly-serialize ser` (note-list edit distance / `ref_len`).
   - **Proxy Fixes/Page:** `(ref_len - matches) + (hyp_len - matches)` from `strict-f1`. Reported as median per stratum; proxy until measure-atomic `eval-harness fixes-estimate` is run on a top-K lattice.
   - **OMR-NED:** Scored offline via pinned `musicdiff` and attached via `eval-harness suite-summary` (`missing = 1.0` for no-output pieces).
4. **Zero-Exclusion / Denominator Rule:**
   - Every piece in the manifest stays in the denominator (`n_scored == n_pieces`).
   - Blind detection misses (`blind-miss` / `staff-unet-miss`) and `bekern-decode` failures score **`f1 = 0.0`, `ser = 1.0`, `notation_f1 = 0.0`**, and are **never dropped** from means.
5. **Required Reported Columns:**
   - Clean and Photo **Strict-F1 mean**, **Notation-F1 mean**, **SER mean**, **Fixes/page median**, **blind detection rate** (`blind_photo_detected_rate`), `n_scored / n_pieces`, `test_pages_touched == 0`, and `suite31_manifest_sha256` — broken down by `typeset` (31), `hw_all` (14), `hw_landscape` (5), `hw_portrait_debussy` (4), and `hw_portrait_pdm` (5).

---

## 4. Anti-Overfitting & Eval Hygiene Discipline

- **Strict Blind-Photo Mode:** Always evaluate with `--blind-photo --staff-unet-ckpt <stage1_ckpt>`. Never use GT `s_ref`, scaled-clean box fallbacks, or `--use-manifest-boxes` for headline scores.
- **Zero Piece-Specific Heuristics:** Never add printed-score image heuristics (e.g., left-barline rules, comb-snaps tuned on Suite31) or post-decode grammar patches.
- **Uncapped Decode Length:** Pass `--max-len 768` for 768-token checkpoints (e.g., `exp2B` and later). Both `measure_decode` (`beam_search`) and `trim_max_len_repeat_loop` honor `--max-len` without silently clamping at 564.
- **Cross-Check `gate20/sys-dev`:** For milestone evaluations, run without `--suite31-only` so `gate20/sys-dev` (clean + degraded) is scored alongside `suite31`, while keeping `gate20/test-8` strictly sealed (`test_pages_touched == 0`).
- **No Training Leakage:** Zero `suite31` or `suite31_hw` works may ever enter any labeled or unlabeled training pool.

---

## 5. Exact Evaluation Commands (PowerShell, Repo Root)

### Full 45-Piece `suite31+hw` + `gate20/sys-dev` (Blind, Heuristic-Free)
```powershell
.\.venv\Scripts\python.exe tmp-train/mvd/eval_sys_gate.py `
    --ckpt <STAGE2_CKPT.pt> `
    --staff-unet-ckpt <STAGE1_STAFF_UNET2D.pt> `
    --suite31 --suite31-manifest data-gen/score-corpus/suite31/suite31_hw_manifest.json `
    --blind-photo --stride 4x8 --max-len 768 --beam 3 `
    --out data-gen/score-corpus/gate20/probe_report_<RUN_NAME>.json
```

### Frozen 31-Piece Printed `suite31` Only (Fast Check)
```powershell
.\.venv\Scripts\python.exe tmp-train/mvd/eval_sys_gate.py `
    --ckpt <STAGE2_CKPT.pt> `
    --staff-unet-ckpt <STAGE1_STAFF_UNET2D.pt> `
    --suite31-only --blind-photo --stride 4x8 --max-len 768 --beam 3 `
    --out data-gen/score-corpus/gate20/probe_report_<RUN_NAME>_s31.json
```

### 14-Piece Real Handwritten Slice Only (`suite31_hw_only_manifest.json`)
```powershell
.\.venv\Scripts\python.exe tmp-train/mvd/eval_sys_gate.py `
    --ckpt <STAGE2_CKPT.pt> `
    --staff-unet-ckpt <STAGE1_STAFF_UNET2D.pt> `
    --suite31-only --suite31-manifest data-gen/score-corpus/suite31/suite31_hw_only_manifest.json `
    --blind-photo --stride 4x8 --max-len 768 --beam 3 `
    --out data-gen/score-corpus/gate20/probe_report_<RUN_NAME>_hw14.json
```

### Protocol Hygiene Verification (CPU-Only)
```powershell
$env:CUDA_VISIBLE_DEVICES=""; .\.venv\Scripts\python.exe -m unittest tmp-train/mvd/test_suite31_protocol.py -v
```
