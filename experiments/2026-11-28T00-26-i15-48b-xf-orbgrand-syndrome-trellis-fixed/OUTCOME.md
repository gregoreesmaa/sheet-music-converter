# OUTCOME — `2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed` (`i15-48b`)

- **Status:** `COMPLETED` (`rc=0`, `3/3` arms completed, `12/12` pages/arm, zero NaNs, `error: null`)
- **Verdict:** **`DISCARD`** *(the `voice-1` no-op bug from `i15-48` is cleanly fixed — `42/230` violating bars detected, `18` repaired by ORBGRAND (`-42.86%` rel. syndrome) at `+0.035%` latency, and `7/12` pages routed to `trellis_bma_3view` — yet BOTH arms fail `gate_pass` and regress accuracy on every metric: `token_f1` (`-0.00050` / `-0.00363`), decode-independent `py_note_f1` (`-0.00127` / `-0.01022`), and Rust strict `note_f1` (`-0.18637` / `-0.10369` due to grammar-breaking flips on `5/5` repaired pages))*
- **Platform / GPU:** Local Windows 11 — `NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`)
- **Started / Finished (UTC):** `2026-11-27T22:51:45Z` → `2026-11-27T22:56:00Z` (`254.79 s` / `4.2 min` total)
- **Checkpoint Evaluated:** `hf:checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt` (`arch=uni_omr_v2`, `step=6874`, `missing_keys=[]`, `unexpected_keys=[]`)
- **Predecessor:** [`2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma`](../2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma/OUTCOME.md) (`ITERATE`)
- **Baseline Commit SHA:** `dddf9a2`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed.json` (pulled to [`run_hf.json`](run_hf.json))
  - Arm Syncs: `checkpoints/2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed/{greedy_fsa,orbgrand_sogrand,trellis_bma_3view}_arm_result.json`

---

## Measured Results (3 Arms, 12 Hard `suite31_hw` Pages: 6 Printed + 6 HW, `seed=42`)

| Slice | Metric | `greedy_fsa` (Arm 0) | `orbgrand_sogrand` (Arm 1) | `trellis_bma_3view` (Arm 2) | Delta (Arm 1 / Arm 2 vs Arm 0) | Gate (`PLAN.md`) | Pass? |
|---|---|---:|---:|---:|---|---|---|
| **`all12`** | **`note_f1`** *(Rust strict)* | `0.52275` | `0.33638` | `0.41906` | **`-0.18637` / `-0.10369`** | `>= +0.005` / `>= +0.010` flagged | **FAIL** |
| `all12` | `py_note_f1` *(decode-free)* | `0.54739` | `0.54612` | `0.53717` | **`-0.00127` / `-0.01022`** | higher is better | **FAIL** |
| `all12` | **`token_f1`** *(LCS)* | `0.57860` | `0.57810` | `0.57497` | **`-0.00050` / `-0.00363`** | `>= +0.005` / `>= +0.010` flagged | **FAIL** |
| `all12` | **`bsvr`** *(lower is better)* | `0.16808` | `0.12713` | `0.10177` | `-24.36%` / `-39.45%` rel. | lower is better | **PASS** |
| `all12` | `syndrome_rate` *(multi-voice)* | `0.18261` (`42/230`) | `0.10435` (`24/230`) | `0.07306` (`16/219`) | **`-42.86%` / `-59.99%` rel.** | `<= -25%` rel. | **PASS** |
| `all12` | **`wfpp`** *(lower is better)* | `316.58` | `295.17` | `292.58` | `-21.41` / `-24.00` *(BSVR term)* | lower is better | **CONFOUNDED** |
| `all12` | `n_repaired` / `queries` | `0` / `0` | **`18`** / `1,675` | `0` / `0` | `18/42` (`42.9%`) bars repaired | `>= 10%` | **PASS** |
| `all12` | `mean_p_not_found` (`flagged`) | `0.00000` (`0/12`) | `0.16904` (`7/12`) | `0.16904` (`7/12`) | `7/12` pages routed | `> 0` | **PASS** |
| `all12` | **`ms/page`** | `9,670.1 ms` | `9,673.5 ms` | `20,072.0 ms` | **`+0.035%`** / **`2.076×`** | `< +5%` (Arm 1) | **PASS / SLOW** |
| `all12` | `bekern_decode_ok` | `12/12` | **`7/12`** | **`8/12`** | **`-5` / `-4` broken pages** | `12/12` | **FAIL** |
| **`printed`** *(n=6)* | `note_f1` / `py_note_f1` | `0.97952` / `0.98635` | `0.65018` / `0.98543` | `0.81555` / `0.98586` | `-0.32934` (`-0.00092` py) / `-0.16397` (`-0.00049` py) | — | **FAIL** |
| `printed` | `token_f1` / `bsvr` / `wfpp` | `0.97133` / `0.08804` / `12.67` | `0.97085` / `0.08804` / `10.67` | `0.97105` / `0.08804` / `10.50` | `-0.00048` / `-0.00028` tok | — | **FAIL** |
| **`hw`** *(n=6)* | `note_f1` / `py_note_f1` | `0.06598` / `0.10844` | `0.02257` / `0.10681` | `0.02257` / `0.08848` | `-0.04341` (`-0.00163` py) / `-0.04341` (`-0.01996` py) | — | **FAIL** |
| `hw` | `token_f1` / `bsvr` / `wfpp` | `0.18586` / `0.24812` / `620.50` | `0.18534` / `0.16621` / `579.67` | `0.17889` / `0.11551` / `574.67` | `-0.00052` / `-0.00697` tok | — | **FAIL** |
| **Flagged** *(n=7)* | `token_f1` / `note_f1` / `py_note_f1` | `0.53285` / `0.45186` / `0.49246` | `0.53224` / `0.27292` / `0.49070` | `0.52688` / `0.41465` / `0.47536` | Arm 2: **`-0.00597` tok / `-0.03721` note / `-0.01710` py** | `>= +0.010` (Arm 2) | **FAIL** |
| **Overall Gate** | `gate_pass` (`eval_hf.json`) | baseline | `orbgrand_gate_pass=false` | `trellis_gate_pass=false` | `overall_gate_pass=false` | PROMOTE | **`DISCARD`** |

---

## Scrutiny & Anomaly Check

1. **`i15-48` No-Op Bug Confirmed Fixed:**
   - Multi-staff / multi-voice `compute_bar_duration_syndrome` correctly detected `42/230` (`18.26%`) violating bars in `greedy_fsa` (`9` printed, `33` HW).
   - `orbgrand_sogrand` actively executed `1,675` CPU queries (`+3.4 ms/page`, `+0.035%` overhead), repaired `18/42` violating bars (`-42.86%` rel. `syndrome_rate`), and produced calibrated `mean_p_not_found = 0.16904`, flagging `7/12` pages for `trellis_bma_3view`.
2. **Anomaly #1 — Unconstrained ORBGRAND Flips Break `bekern` Grammar (`bekern_decode_ok: 12/12 → 7/12`):**
   - In `orbgrand_repair_bar()` (`run_colab.py:332–338`), the candidate filter checks `tok in WRITTEN_DUR_BEATS or alt_tok in WRITTEN_DUR_BEATS or tok in ("dot", "bar-rest") or alt_tok in ("dot", "bar-rest")` using **`or`** instead of **`and`**, without checking `PrefixState` FSA validity.
   - This allows replacing a rhythm token with a non-rhythm token (pitch/octave/control) or replacing a non-rhythm token with a rhythm token, emitting syntactically invalid `bekern` streams.
   - **100% correlation (`5/5` repaired pages broke):** Every single page where ORBGRAND applied ≥1 repair (`fiddle-strathspey`: 1, `joplin-maple-leaf`: 2, `hw-beethoven-op135`: 3, `hw-schubert-d810`: 1, `hw-debussy-vetement-du-blesse`: 11) flipped `bekern_decode_ok` from `true` → `false`, collapsing Rust `note_f1` to `0.0` (driving printed `note_f1` `0.97952 → 0.65018`).
3. **Why This Is Still a Clean `DISCARD` (Not Another `ITERATE`):**
   - Even on the **decode-independent** metrics (`token_f1` and `py_note_f1`, which parse tokens directly without Rust `bekern-decode`), **neither arm improves accuracy on any slice**:
     - `orbgrand_sogrand`: `token_f1` **`-0.00050`** (`-0.00048` printed, `-0.00052` HW) and `py_note_f1` **`-0.00127`** (`-0.00092` printed, `-0.00163` HW). On every single repaired page, `py_note_f1` and `token_f1` were flat or strictly worse (`fiddle-strathspey` `1.0000 → 0.9974`, `joplin-maple-leaf` `0.9981 → 0.9952`, `hw-schubert-d810` `0.1243 → 0.1220`, `hw-debussy-vetement-du-blesse` `0.2107 → 0.2032`).
     - `trellis_bma_3view`: Doubles latency (`2.076×`, `9.67 s → 20.07 s/page`) while **regressing** flagged-page `token_f1` (`-0.00597`), `py_note_f1` (`-0.01710`), and HW `py_note_f1` (`0.10844 → 0.08848`, `-0.01996`). Only 2/7 flagged pages saw modest `token_f1` gains (`hw-schubert-d810` `+0.0317`, `hw-debussy-vetement-du-blesse` `+0.0381`), overwhelmed by `hw-beethoven-op135` (`token_f1 0.3712 → 0.2595`, `py_note_f1 0.1629 → 0.0886`) and `hw-debussy-vetement-du-blesse` (`py_note_f1 0.2107 → 0.1272`).
   - **Falsified Core Hypothesis:** In autoregressive OMR, bar-duration violations overwhelmingly stem from **insertions/deletions (missing/extra notes, skipped voices, or staff misalignment)**, not isolated duration-token substitutions. Post-hoc ORBGRAND duration flipping satisfies algebraic bar conservation (`-42.86%` syndrome) by mutating valid tokens, and 3-view bar-level Trellis-BMA consensus degrades accuracy when views disagree on bar segmentation (`n_bars_total` shrank `230 → 219`).
4. **Spurious `wfpp` Drop:** The `-21.41` / `-24.00` `wfpp` reduction is an artifact of the `2.0 * n_bar_violations` penalty term dropping when `bsvr` decreases (plus `bekern_decode_ok=false` pages falling back to the Python note count) while actual note/token accuracy worsened.

---

## Verdict & Follow-Up Action

- **Verdict:** **`DISCARD`** — Directly meets `PLAN.md` §6 `DISCARD` criteria (`overall_gate_pass=false`; neither `orbgrand_sogrand` nor `trellis_bma_3view` improves `token_f1`, `py_note_f1`, or `note_f1` on any slice).
- **Follow-Up Action:** Drop post-hoc ORBGRAND/SOGRAND duration flipping and 3-view Trellis-BMA bar voting; do not port into `eval_uni_suite31.py` or the Rust/ONNX pipeline. No successor experiment needed.
