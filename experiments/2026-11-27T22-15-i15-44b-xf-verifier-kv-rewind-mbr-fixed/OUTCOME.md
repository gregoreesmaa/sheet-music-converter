# OUTCOME — `2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed` (`i15-44b`)

- **Status:** `COMPLETED` (`rc=0`, `2/2` arms completed, `12/12` pages/arm, `12/12` `bekern_decode_ok=true`, zero NaNs, `error: null`)
- **Verdict:** **`DISCARD`** *(all bugs from `i15-44` cleanly fixed — true Rust MusicXML `note_f1` / `wfpp` scored, premature `<eos>` eliminated, latency brought within `1.324× <= 1.35×` — yet `kv_rewind_mbr` regressed `note_f1` (`-0.00482`), `tok_lcs_f1` (`-0.01519`), and `wfpp` (`+9.84` worse) across both printed and handwritten slices)*
- **Platform / GPU:** Local Windows 11 — `NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`)
- **Started / Finished (UTC):** `2026-11-27T22:45:21Z` → `2026-11-27T22:51:45Z` (`383.7 s` / `6.4 min` total)
- **Checkpoint Evaluated:** `hf:checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt (step=6874)`
- **Predecessor:** [`2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr`](../2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr/OUTCOME.md) (`ITERATE`)
- **Baseline Commit SHA:** `ca8826e`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed.json` (pulled to [`run_hf.json`](run_hf.json))
  - Arm Syncs: `checkpoints/2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed/{greedy_fsa,kv_rewind_mbr}_arm_result.json`

---

## Measured Results (`greedy_fsa` vs `kv_rewind_mbr`, 12 `suite31_hw` photo pages: 6 printed + 6 HW, `seed=42`)

| Slice | Metric | `greedy_fsa` (Arm 0) | `kv_rewind_mbr` (Arm 1) | Delta (`Arm 1 - Arm 0`) | Gate (`PLAN.md`) | Pass? |
|---|---|---:|---:|---:|---|---|
| **`all12`** | **`note_f1`** *(Rust strict, primary)* | `0.70037` | `0.69555` | **`-0.00482`** | `>= +0.008` PROMOTE | **FAIL** |
| `all12` | **`tok_lcs_f1`** | `0.73067` | `0.71548` | **`-0.01519`** | higher is better | **FAIL** |
| `all12` | **`bsvr`** *(lower is better)* | `0.18137` | `0.16153` | `-0.01984` (`-10.94%` rel.) | `<= -25%` PROMOTE / `<= -15%` ITERATE | **FAIL** |
| `all12` | **`wfpp`** *(lower is better)* | `219.58` | `229.42` | **`+9.84`** *(worse)* | lower is better | **FAIL** |
| `all12` | **`ms/page`** | `13,186.8 ms` | `17,458.2 ms` | `+4,271.4 ms` (`1.324×`) | `<= 1.35×` | **PASS** |
| `all12` | `rewinds/page` (`accept_rate`) | `0.00` (`0.0000`) | `5.75` (`0.8696`, `60/69`) | `+5.75` | — | — |
| **`printed`** *(n=6)* | **`note_f1`** | `0.97338` | `0.96725` | **`-0.00613`** | `>= -0.003` guard | **FAIL** |
| `printed` | `tok_lcs_f1` | `0.94850` | `0.93578` | `-0.01272` | higher is better | **FAIL** |
| `printed` | `bsvr` | `0.08388` | `0.08551` | `+0.00163` (`+1.94%` worse) | lower is better | **FAIL** |
| `printed` | `wfpp` | `16.33` | `32.17` | **`+15.84`** *(+97% worse)* | lower is better | **FAIL** |
| `printed` | `ms/page` / `rewinds` | `10,958.1 ms` / `0.00` | `12,433.0 ms` (`1.135×`) / `4.00` (`95.8%` acc) | `+1,474.9 ms` | — | — |
| **`hw`** *(n=6)* | **`note_f1`** | `0.42735` | `0.42385` | **`-0.00350`** | higher is better | **FAIL** |
| `hw` | `tok_lcs_f1` | `0.51283` | `0.49518` | `-0.01765` | higher is better | **FAIL** |
| `hw` | `bsvr` | `0.27886` | `0.23754` | `-0.04132` (`-14.82%` rel.) | lower is better | **PARTIAL** |
| `hw` | `wfpp` | `422.83` | `426.67` | `+3.84` *(worse)* | lower is better | **FAIL** |
| `hw` | `ms/page` / `rewinds` | `15,415.5 ms` / `0.00` | `22,483.4 ms` (`1.458×`) / `7.50` (`82.2%` acc) | `+7,067.9 ms` | — | — |
| **Overall Gate** | `gate_pass` | baseline | — | — | PROMOTE / ITERATE | **`DISCARD`** |

### Per-Page Delta Highlights (`greedy_fsa` → `kv_rewind_mbr`)
- **Printed (`n=6`, 0 improved, 4 regressed, 2 tied):** `schumann-op41n1-3` (`1.0000 → 0.9887`, `wfpp 0 → 24`, `3/3` rewinds despite `bsvr=0.000`), `mozart-k155-2` (`1.0000 → 0.9837`, `wfpp 0 → 26`, `2/2` rewinds despite `bsvr=0.000`), `mozart-k458-1` (`1.0000 → 0.9917`, `wfpp 2 → 15`, `3/3` rewinds), `schumann-dichterliebe-2` (`0.8448 → 0.8438`, `wfpp 83 → 97`), `schubert-lindenbaum` (`0.9956 → 0.9956`), `airds-air` (`1.0000 → 1.0000`, `wfpp 8 → 26`).
- **Handwritten (`n=6`, 3 improved, 2 regressed, 1 tied):** Improved: `hw-cornelius-botschaft-d-b` (`0.6216 → 0.7019`, `+0.0803`), `hw-beethoven-op135` (`0.0156 → 0.0828`, `+0.0673`, `bsvr 0.147 → 0.000`), `hw-debussy-berceuse-heroique` (`0.4677 → 0.5032`, `+0.0355`). Regressed: `hw-beethoven-das-blumchen-wunderhold-d-b` (`0.8756 → 0.7273`, **`-0.1483`**, `wfpp 65 → 156`), `hw-debussy-la-plus-que-lente` (`0.3946 → 0.3389`, **`-0.0557`**, `wfpp 476 → 503`). Tied: `hw-debussy-lindaraja` (`0.1890 → 0.1890`).

---

## Scrutiny & Anomaly Check

1. **Clean Verification of Both `i15-44` Bug Fixes (Zero Harness Confounds):**
   - `12/12` pages in both arms scored via `eval_uni_suite31_rust` with `bekern_decode_ok: true`, real non-zero `ref_notes`, and consistent `note_f1` / `py_note_f1` / `tok_lcs_f1` / `wfpp`.
   - Premature `<eos>` truncation is completely eliminated (`hit_eos=true` on all 9 non-`MAX_LEN` pages; printed bar counts preserved except `18 → 17` on `schumann-dichterliebe-2`).
   - Violation-only triggering with `K=2, MAX_REWINDS_PER_PAGE=8` cut latency overhead from `2.594×` (`i15-44`) to `1.324×` (passing the `<= 1.35×` latency gate).
2. **Root Cause of Printed Regression (`note_f1 -0.00613`, `wfpp 16.33 → 32.17`):**
   - On `schumann-op41n1-3` and `mozart-k155-2` — both of which achieved **perfect `note_f1 = 1.0000` and `bsvr = 0.000` under `greedy_fsa`** — `kv_rewind_mbr` still triggered `3` and `2` rewinds respectively and accepted `100%` of them (`5/5`), introducing `24` and `26` fixes per page!
   - This proves the lightweight inline verifier (`_measure_violation_errors`) suffers false-positive triggers on legitimate musical constructs (pickup/anacrusis, multi-voice/grand-staff or tuplet bars), and replacing an exact greedy bar with a `T=0.70` sampled continuation degrades notation accuracy even when the sampled rollout scores higher on the MBR heuristic.
3. **Root Cause of HW Regression (`note_f1 -0.00350`, `tok_lcs_f1 -0.01765`):**
   - Even when `bsvr` genuinely drops (`-14.82%` rel. on HW, e.g., `hw-debussy-la-plus-que-lente` `0.438 → 0.290`), satisfying local bar-sum conservation via `T=0.70` resampling does not reliably recover the true visual notes — it frequently hallucinates rhythmically valid but pitch/note-wrong bars (`hw-beethoven-das-blumchen-wunderhold-d-b` dropped `-0.1483` `note_f1`, `hw-debussy-la-plus-que-lente` dropped `-0.0557` `note_f1`).
4. **Platform & Coverage Sanity:** All `12/12` pages ran cleanly with zero crashes/NaNs under `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION` on RTX 4080 SUPER (`6.4 min` total). The 3 dense Debussy HW pages hit `MAX_LEN=2048` (`hit_eos=false`) identically in both arms.

---

## Verdict & Follow-Up Action

- **Verdict:** **`DISCARD`** — Falsified cleanly under the bug-fixed Rust MusicXML scorer. Hits `PLAN.md` §6 `DISCARD` criteria (`note_f1` regresses `-0.00482` overall, `-0.00613` printed, `-0.00350` HW; `wfpp` worsens `+9.84`; `bsvr` reduction `-10.94%` is well below the `-25%` PROMOTE / `-15%` ITERATE bars).
- **Follow-Up Action:** Do **not** wire inline `kv_rewind_mbr` into `eval_uni_suite31.py` or the production decoder. Retain standard FSA-constrained greedy decoding and the already-promoted `MeasureBeamHeaderGuard` (`mbeam2`). No successor experiment needed.
