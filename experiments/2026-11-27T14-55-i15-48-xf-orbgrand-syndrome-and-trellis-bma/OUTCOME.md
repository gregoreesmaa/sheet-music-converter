# OUTCOME — `2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma` (`i15-48`)

- **Status:** `COMPLETED` (`rc=0`, `3/3` arms completed, `12/12` pages/arm, zero NaNs, `error: null`)
- **Verdict:** **`ITERATE`** *(silent no-op bug in `compute_bar_duration_syndrome` (`run_colab.py:164-170`) broke on the leading `voice-1` token of every bekern bar, falsely marking `230/230` bars (`100%`) as `syndrome=0`; as a result, `orbgrand_sogrand` executed `0` queries / `0` repairs / `0` flagged pages, and `trellis_bma_3view` decoded `0` extra views — neither hypothesis was actually tested)*
- **Platform / GPU:** Local Windows 11 — `NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`)
- **Started / Finished (UTC):** `2026-11-27T22:05:52Z` → `2026-11-27T22:09:32Z` (`219.74 s` / `3.7 min` total)
- **Checkpoint Evaluated:** `hf:checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt` (`arch=uni_omr_v2`, `step=6874`, `missing_keys=[]`, `unexpected_keys=[]`)
- **Baseline Commit SHA:** `750810d` *(plan baseline `6055c49` + manifest/GT loader fix in `750810d`)*
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma.json` (pulled to [`run_hf.json`](run_hf.json))
  - Arm Syncs: `checkpoints/2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma/{greedy_fsa,orbgrand_sogrand,trellis_bma_3view}_arm_result.json`

---

## Measured Results (3 Arms, 12 Hard `suite31_hw` Pages: 6 Printed + 6 HW, `seed=42`)

| Metric | `greedy_fsa` (Arm 0) | `orbgrand_sogrand` (Arm 1) | `trellis_bma_3view` (Arm 2) | Delta (Arm 1 / Arm 2 vs Arm 0) | Gate (`PLAN.md`) | Pass? |
|---|---:|---:|---:|---|---|---|
| **`printed_token_f1`** *(n=6, LCS)* | `0.96052` | `0.96052` | `0.96052` | `0.00000` / `0.00000` | — | **NO-OP** |
| **`hw_token_f1`** *(n=6, LCS)* | `0.18474` | `0.18474` | `0.18474` | `0.00000` / `0.00000` | — | **NO-OP** |
| **`all12_token_f1`** *(n=12, LCS)* | `0.57263` | `0.57263` | `0.57263` | `0.00000` / `0.00000` | `>= +0.005` (Arm 1) / `>= +0.010` flagged (Arm 2) | **INVALID (NO-OP)** |
| **`syndrome_rate`** *(BSVR proxy)* | `0.00000` (`0/230`) | `0.00000` (`0/230`) | `0.00000` (`0/230`) | `0.00000` / `0.00000` | `<= -25%` rel. | **INVALID (BUG)** |
| `n_repaired` / `queries` | N/A | `0` / `0` | N/A | `0` repairs | `>= 10%` repair rate | **INVALID (NO-OP)** |
| **`mean_p_not_found`** *(SOGRAND)* | `0.00000` | `0.00000` | `0.00000` | `0.00000` | calibrated `> 0.15` on hard bars | **INVALID (BUG)** |
| `n_flagged_pages` *(for 3-view)* | `0/12` | `0/12` | `0/12` | `0` pages routed | `> 0` | **INVALID (NO-OP)** |
| **`mean_ms_per_page`** | `10,161.2 ms` | `10,161.4 ms` | `10,161.2 ms` | `+0.2 ms` (`+0.002%`) / `+0.0 ms` | `< +5%` (Arm 1) | **NO-OP** |
| `note_f1` / `wfpp` / `gate_pass` | *not emitted* | *not emitted* | *not emitted* | — | — | **MISSING IN HARNESS** |

### Per-Page Baseline Breakdown (`greedy_fsa` == `orbgrand_sogrand` == `trellis_bma_3view`)
- **Printed (`n=6`, mean `token_f1 = 0.96052`, `114` bars, `0` flagged):** `repeats-melody` (`0.98056`, 16 bars, 3.95 s), `fiddle-strathspey` (`0.97944`, 27 bars, 8.96 s), `joplin-maple-leaf` (`0.97813`, 29 bars, 25.06 s), `beethoven-grosse-fuge` (`0.97543`, 23 bars, 11.00 s), `bach-bwv66.6` (`0.93905`, 10 bars, 7.04 s), `schoenberg-op19-2` (`0.91053`, 9 bars, 5.64 s).
- **Handwritten (`n=6`, mean `token_f1 = 0.18474`, `116` bars, `0` flagged):** `hw-beethoven-op135` (`0.36801`, 12 bars, 5.05 s), `hw-debussy-vetement-du-blesse` (`0.23557`, 31 bars, 15.34 s), `hw-schubert-d810` (`0.21036`, 30 bars, 17.79 s), `hw-mozart-k387` (`0.14735`, 23 bars, 14.40 s), `hw-mozart-k464` (`0.09728`, 15 bars, 5.22 s), `hw-mozart-k421` (`0.04989`, 5 bars, 2.49 s).

---

## Scrutiny & Anomaly Check

1. **Root-Cause Silent No-Op Bug — Leading `voice-1` Triggers Immediate `break` (`0/230` bars evaluated):**
   - In `bekern` streams, **every bar begins with `voice-1`** immediately after `bar` / `<sys>` or header attributes (verified directly on `suite31` GT streams: `16/16` bars in `airds-air` and `8/8` bars in `aloha-oe` start with `voice-1`).
   - In `compute_bar_duration_syndrome()` (`run_colab.py:164-170`):
     ```python
     if tok.startswith("voice-") or tok == "<staff>":
         # Reset voice-level accumulator conservatively by returning best match so far
         break
     if total <= 1e-9:
         return 0.0, 0.0, True
     ```
   - Because `voice-1` occurs at index `0` of every bar (and before the first note of bar 0), the loop **immediately `break`s before seeing a single duration token**, leaving `total == 0.0 <= 1e-9` and returning `(0.0, 0.0, True)` on **all `230/230` bars (`100%` false-valid)**.
2. **Cascading 100% No-Op Across Arm 1 (`orbgrand_sogrand`) and Arm 2 (`trellis_bma_3view`):**
   - Because `orig_ok == True` for every bar, `orbgrand_repair_bar()` returned early on line 229 with `queries=0`, `repaired=False`, `p_not_found=0.0`.
   - Consequently, `apply_orbgrand_page()` produced `mean_p_not_found = 0.0` and `flagged_for_3view = False` on all `12/12` pages (even on `hw-mozart-k421` with `token_f1 = 0.04989`!).
   - `trellis_bma_3view` therefore entered the `else: fused = rep["tokens"]` branch on all `12/12` pages, decoding **zero** contrast/scale views and producing byte-identical outputs to `greedy_fsa`.
3. **Why `--smoke-test` Missed This Bug:**
   - `run_smoke_test()` (`run_colab.py:936`) constructed a synthetic bar `["time-4-4", "quarter", "C", "4", ...]` that omitted `voice-1` and `<staff>`, so the `voice-*` / `<staff>` branch was never tested.
4. **Additional Harness Gaps Identified:**
   - **Multi-voice / multi-staff duration accounting:** Even after skipping leading `voice-1`, breaking on `voice-2` or `<staff>` ignores all subsequent voices/staves in polyphonic/grand-staff bars. Each `(staff, voice)` segment inside a bar must be summed independently against `bar_capacity` (and bar 0 pickup/anacrusis should be exempt from underfull-bar false positives).
   - **Missing primary OMR metrics:** `run_colab.py` only logs Token LCS F1 (`token_f1`) and `syndrome_rate` — it does not emit MusicXML `note_f1`, `wfpp`, or a `gate_pass` summary field.
5. **Platform & Coverage Sanity:**
   - All `12/12` pages decoded cleanly with `missing_keys=[]`, `unexpected_keys=[]`, zero NaNs, and Windows `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION` on RTX 4080 SUPER (`3.7 min` total wall-clock).

---

## Verdict & Follow-Up Action / Successor Experiment

- **Verdict:** **`ITERATE`** — Do **not** promote and do **not** discard `ORBGRAND`/`SOGRAND` or `Trellis-BMA` yet: neither algorithm executed on a single bar (`0/230` bars evaluated, `0` queries, `0` multi-view decodes) due to the `voice-1` early-`break` bug.
- **Proposed Successor Experiment (`i15-48b`, to be queued by main thread):**
  1. **Fix `compute_bar_duration_syndrome` for real bekern bars:** Split each bar into `(staff, voice)` segments on `<staff>` and `voice-*` tokens, compute exact duration conservation per non-empty segment (with `bar-rest` and tuplet support, exempting bar 0 pickup underfills), and report the worst/aggregate segment syndrome.
  2. **Harden `--smoke-test`:** Include realistic `voice-1`, multi-voice (`voice-1` + `voice-2`), and multi-staff (`<staff>`) bekern bars so a `voice-1` no-op can never pass again.
  3. **Emit full OMR metrics:** Add MusicXML `note_f1`, `bsvr`, `wfpp`, and `gate_pass` alongside `token_f1` and SOGRAND `p_not_found` calibration (`AUROC` / correlation with per-page error), and re-run all 3 arms on the same 12 hard `suite31_hw` pages.
