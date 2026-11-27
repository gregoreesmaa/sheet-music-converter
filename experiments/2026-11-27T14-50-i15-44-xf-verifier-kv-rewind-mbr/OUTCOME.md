# OUTCOME — `2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr` (`i15-44`)

- **Status:** `COMPLETED` (`rc=0`, `2/2` arms completed, `12/12` pages/arm, zero NaNs, `error: null`)
- **Verdict:** **`ITERATE`** *(eval harness GT-format bug invalidated `Note F1` / `TokLCS F1` / `wFPP`, and early-`<eos>` reward hacking corrupted printed-page bar rollouts; however HW `BSVR` dropped `-40.0%` relative, warranting a clean bug-fixed v2 rerun)*
- **Platform / GPU:** Local Windows 11 — `NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`)
- **Started / Finished (UTC):** `2026-11-27T18:15:57Z` → `2026-11-27T18:26:51Z` (`653.47 s` / `10.9 min` total)
- **Checkpoint Evaluated:** `hf:checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt (step=6874)`
- **Baseline Commit SHA:** `6055c49`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr.json` (pulled to [`run_hf.json`](run_hf.json))
  - Arm Syncs: `checkpoints/2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr/{greedy_fsa,kv_rewind_mbr}_arm_result.json`

---

## Measured Results (`greedy_fsa` vs `kv_rewind_mbr`, 12 `suite31_hw` photo pages: 6 printed + 6 HW, `seed=42`)

| Metric | `greedy_fsa` (Arm 0) | `kv_rewind_mbr` (Arm 1) | Delta (`Arm 1 - Arm 0`) | Gate (`PLAN.md`) | Pass? |
|---|---:|---:|---:|---|---|
| **`note_f1`** *(overall, primary)* | `0.00000` | `0.00000` | `0.00000` | `>= +0.010` PROMOTE | **INVALID** *(GT XML bug)* |
| **`tok_lcs_f1`** *(overall)* | `0.00006` | `0.00006` | `0.00000` | higher is better | **INVALID** *(GT XML bug)* |
| **`bsvr`** *(overall, lower is better)* | `0.17581` | `0.16587` | **`-0.00994` (`-5.65%` rel.)** | `<= -25%` PROMOTE / `<= -15%` ITERATE | **FAIL** |
| `bsvr` *(printed, n=6)* | `0.08388` | `0.17113` | **`+0.08725` (`+104.0%` worse!)** | lower is better | **FAIL** |
| `bsvr` *(HW, n=6)* | `0.26774` | `0.16062` | **`-0.10712` (`-40.0%` rel.)** | lower is better | **PASS (HW only)** |
| **`wfpp`** *(lower is better)* | `348.83` | `293.33` | `-55.50` *(spurious!)* | lower is better | **INVALID** *(empty ref + early EOS)* |
| **`ms_per_page`** *(overall)* | `14,511.0 ms` | `37,638.4 ms` | **`+23,127.4 ms` (`2.594×`)** | `<= 1.8×` | **FAIL** |
| `ms_per_page` *(printed / HW)* | `12,109.6 / 16,912.4 ms` | `19,526.6 / 55,750.2 ms` | `1.61×` / **`3.30×`** | `<= 1.8×` | **FAIL (HW)** |
| `rewinds_per_page` *(overall / PR / HW)* | `0.00` | `11.42` (`6.33` / `16.50`) | `+11.42` | — | — |
| `rewind_accept_rate` | `0.0000` | **`1.0000`** (`137/137`) | `+1.0000` | — | **ANOMALY** |
| **Overall Gate** | baseline | — | — | PROMOTE / ITERATE | **`ITERATE`** *(fix 2 bugs + re-run)* |

---

## Scrutiny & Anomaly Check

1. **Silent Eval Harness Bug (`note_f1 == 0.0`, `tok_lcs_f1 == 0.00006` across all 12 pages):**
   - `run_colab.py:976` loaded ground truth via `ref_toks = piece["gt"].read_text(encoding="utf-8", errors="ignore").split()`, assuming `piece["gt"]` was a whitespace-separated `.bekern` token file.
   - In `suite31_hw_manifest.json`, `piece["gt"]` is a **MusicXML (`.musicxml`)** file (`<?xml version='1.0' ...><score-partwise version="4.0">...`). Consequently, `rlvr_grpo.extract_pitch_dur_notes(ref_toks)` returned `[]` (0 reference notes) on all 12 pages!
   - This forced `note_f1 = 0.0` on every page and reduced `wFPP` to `1.0 * len(pred_notes) + 2.0 * n_bar_violations + 3.0 * len(pred_headers)` — making the `-55.5` `wFPP` "improvement" a pure artifact of shorter/truncated predictions.
2. **Decoder Bug — Premature `<eos>` Reward Hacking & `100%` Rewind Accept Rate:**
   - In `_rollout_bar_candidate` (`run_colab.py:607–609`), when a temperature-sampled (`T=0.7`) bar continuation emitted `<eos>`, the loop broke immediately without calling `_measure_violation_errors`, leaving `c_m_err = 0.0, c_v_err = 0.0`.
   - `_select_mbr_bar_candidate` (`run_colab.py:499–500`) then awarded that premature-`<eos>` rollout the full `cons_term = +2.0` conservation bonus, beating the original bar and immediately terminating the entire page (`if winner["hit_eos"]: hit_eos = True; break`).
   - Direct evidence on printed pages: `schumann-op41n1-3` collapsed from `17 bars / 757 tok` → `5 bars / 226 tok`, `schubert-lindenbaum` collapsed from `18 bars / 1580 tok` → `11 bars / 1159 tok` (while `bsvr` spiked `0.0556 → 0.4545`), and `mozart-k458-1` collapsed from `17 bars / 1227 tok` → `12 bars / 805 tok`. All 137/137 rewinds (`1.0000` accept rate) were accepted because `orig_cand` always had `cons_term <= 0` whenever a violation triggered rewind.
3. **Domain Asymmetry & Latency Blowup (`2.594×` overall, `3.30×` HW):**
   - Printed `BSVR` **more than doubled** (`0.08388 → 0.17113`, `+104.0%` worse) due to `peak_ent > 1.60` firing on already-valid bars (`6.33 rewinds/page`) and replacing greedy bars with noisy `T=0.7` samples.
   - Conversely, on real handwritten pages (`hw`), `BSVR` dropped **`-40.0%` relative** (`0.26774 → 0.16062`, e.g., `hw-beethoven-op135`: `0.147 → 0.000`, `hw-debussy-la-plus-que-lente`: `0.438 → 0.258`, `hw-debussy-lindaraja`: `0.600 → 0.357`), confirming the core KV-cache rewind mechanism *can* fix real bar-sum violations when violations actually exist.
   - Latency hit `37.6 s/page` (`2.594×` vs `<= 1.8×` gate), with the 3 Debussy HW pages taking `72.7–104.8 s/page` (`22–24` rewinds × `K=4` candidates).
4. **Coverage & Platform Sanity:** All `12/12` pages ran cleanly with zero crashes/NaNs under `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION` on RTX 4080 SUPER. Note that the 3 dense Debussy HW pages hit `MAX_LEN=2048` (`hit_eos=false`) in both arms.

---

## Verdict & Follow-Up Action / Successor Experiment

- **Verdict:** **`ITERATE`** — Do **not** promote the current buggy decoder, and do **not** discard the underlying KV-rewind hypothesis yet because `Note F1` was never validly measured and HW `BSVR` fell `-40.0%`.
- **Proposed Successor Experiment (`i15-44b`, to be queued by main thread):**
  1. **Fix GT scoring:** Score predictions against `piece["gt"]` MusicXML using `eval_uni_suite31`'s real MusicXML note-extraction / Strict Note F1 pipeline (never `piece["gt"].read_text().split()`).
  2. **Fix premature `<eos>` reward hacking:** Ban `<eos>` inside `_rollout_bar_candidate` (or evaluate `_measure_violation_errors` on `<eos>` and forbid `<eos>` unless `orig_cand` also ended in `<eos>`).
  3. **Violation-only trigger + lower latency (`<= 1.8×`):** Remove the `peak_ent > 1.60` entropy-only trigger (rewind *only* on `m_err > 1e-3 or v_err > 1e-3`), reduce `K=4 → K=2` candidates, and cap `MAX_REWINDS_PER_PAGE = 8` (1 retry/bar).
