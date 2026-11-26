# Experiment Outcome — `2026-11-26T01-38-ov-36-hw-multiview-tta-voting`

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T00:46:17Z` (started `2026-11-26T00:16:26Z`)
- **Platform / Hardware:** Google Colab `NVIDIA L4` (`24 GB`, `cc=8.9`, `BF16`, `cudnn_sdp=False`, CUTLASS/Flash SDPA)
- **Elapsed Time:** `1789.96 s` (`29.8 min` compute; `3183.38 s` total queue wall including setup/Rust build; `~2.4 CU`). Colab session `ov-36` verified stopped (`colab.exe sessions`).
- **Checkpoint Evaluated:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`step=12000`, `val_ser=0.192301`, `missing_keys=[]`, `unexpected_keys=[]`)
- **HF Artifacts:**
  - Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-36-hw-multiview-tta-voting.json` (pulled to [`eval_result.json`](./eval_result.json))
  - Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-36-hw-multiview-tta-voting.json` (pulled to [`run_ledger.json`](./run_ledger.json))
- **Verdict:** **`ITERATE`** *(Oracle headroom exceeds `+0.030` on **both** domains — `+0.0419` on `hw_clean` (`0.3030`) and `+0.0327` on `hw_photo` (`0.2243`) — while 6-view unsupervised selectors captured only `25.8%` on `hw_clean` and `-2.8%` on `hw_photo`. Forensic audit pinpointed the exact cause: destructive `clahe` and `deskew` views poisoned the rerankers; offline simulation of **scale-only `{1280, 1536, 1792}` + `best_fsa_beat`** immediately jumps to **`0.2954` on `hw_clean` (`+0.0343` vs `base_1536`, 7W/5T/2L)** and **`0.1969` on `hw_photo` (`+0.0053` vs `base_1536`)**. Immediately queuing `ov-36b` to test pure in-distribution multi-scale TTA `{1024, 1152, 1280, 1408, 1536}` at `max_len=3840` with length-normalized model log-prob + FSA reranking).*

---

## Measured Results (14 Real Historical HW Pieces × 2 Domains = 28 Pages, 6 Views = 168 Decodes)

### 1. Individual Views vs `base_1536` Control (`max_len=4096`, MTP-4)

| View | `hw_clean` Note F1 (Δ vs base) | `hw_clean` Tok LCS | `hw_clean` EOS / Mean s | `hw_photo` Note F1 (Δ vs base) | `hw_photo` Tok LCS | `hw_photo` EOS / Mean s |
|---|---:|---:|---:|---:|---:|---:|
| *`base_1536` (ctrl)* | *0.2611* | *0.3890* | *13/14 / 8.17s* | ***0.1916*** | *0.3327* | *12/14 / 8.44s* |
| **`scale_1280`** | **0.2794** (**+0.0183**) | **0.4268** | 13/14 / 8.26s | 0.1863 (-0.0053) | **0.3417** | 13/14 / 8.43s |
| `scale_1792` | 0.2359 (-0.0252) | 0.3694 | 13/14 / 8.11s | 0.1847 (-0.0069) | 0.3154 | 12/14 / 9.21s |
| `clahe_1536` | 0.2473 (-0.0138) | 0.3857 | 13/14 / 8.63s | 0.1626 (-0.0290) | 0.2807 | 11/14 / 9.02s |
| `deskew_1536` | 0.2509 (-0.0103) | 0.3951 | 13/14 / 7.96s | 0.1604 (-0.0312) | 0.2860 | 12/14 / 8.92s |
| `deskew_clahe_1536` | 0.2328 (-0.0284) | 0.3816 | 12/14 / 9.59s | 0.1379 (-0.0538) | 0.2394 | 12/14 / 9.97s |

### 2. Unsupervised Selectors vs Oracle Upper Bound (All 6 Views vs Offline Scale-Only `{1280, 1536, 1792}`)

| Selector (View Pool) | `hw_clean` Note F1 (Δ vs base) | `hw_clean` W / T / L | `hw_photo` Note F1 (Δ vs base) | `hw_photo` W / T / L | Total TTA Decode Time |
|---|---:|---:|---:|---:|---:|
| *`base_1536` (1 view)* | *0.2611* | *—* | *0.1916* | *—* | *232.6 s (28p)* |
| `best_fsa_beat` (all 6) | 0.2720 (+0.0108) | 6 / 2 / 6 | 0.1668 (-0.0248) | 2 / 6 / 6 | 1465.8 s (168 decodes) |
| `best_mbr_vote` (all 6) | 0.2581 (-0.0030) | 4 / 7 / 3 | 0.1822 (-0.0095) | 2 / 7 / 5 | 1465.8 s (168 decodes) |
| `best_combined` (all 6) | 0.2643 (+0.0032) | 4 / 5 / 5 | 0.1907 (-0.0009) | 2 / 8 / 4 | 1465.8 s (168 decodes) |
| `oracle_best_view` (all 6) | **0.3030 (+0.0419)** | **9 / 5 / 0** | **0.2243 (+0.0327)** | **7 / 7 / 0** | — |
| **`best_fsa_beat` (scale-only 3, offline)** | **0.2954 (+0.0343)** | **7 / 5 / 2** | **0.1969 (+0.0053)** | **2 / 10 / 2** | **~709 s (84 decodes)** |
| `best_combined` (scale-only 3, offline) | 0.2755 (+0.0143) | — | 0.1969 (+0.0053) | — | ~709 s (84 decodes) |
| `oracle_best_view` (scale-only 3, offline) | 0.3005 (+0.0394) | — | 0.2122 (+0.0205) | — | — |

*(Scale-only `{1280, 1536, 1792}` `best_fsa_beat` also beats the strong `scale_1280` single view by **`+0.0160`** on `hw_clean` (`5W / 8T / 1L`) and **`+0.0106`** on `hw_photo` (`3W / 7T / 4L`)!)*

---

## Scrutiny & Anomaly Check

1. **Coverage & Silent Failures — PASS (0 anomalies):**
   - All `14` real historical HW pieces × `2` domains = `28` pages × `6` views = `168` decodes completed cleanly.
   - `decode_ok_rate == 1.000`, `bekern_decode_ok == 14/14` across all 6 views on both domains, `broken_steps == 0` globally, `0` NaN/None scores, `missing_keys == []`, `unexpected_keys == []`.
2. **Apples-to-Apples Sanity:**
   - `base_1536` (`0.2611` clean / `0.1916` photo) and `scale_1280` (`0.2794` clean / `0.1863` photo) match `ov-35` (`(1536, 3840)`: `0.2618` / `0.1921`; `(1280, 3840)`: `0.2799` / `0.1867`) within `0.0007` (the tiny difference is `max_len=4096` vs `3840` on the 3 long Debussy manuscripts).
   - *(Note: README's cited `0.2550` / `0.1516` was from an earlier non-soup/different-decode baseline; all comparisons here use the exact in-run `base_1536` and `scale_1280` controls.)*
3. **Root Cause 1 — Destructive `deskew` & `clahe` Views Poisoned All 3 Unsupervised Rerankers:**
   - **Deskew boundary-clamp failure on `hw_photo`:** Projection-profile deskew slammed into the `±4.00°` clamp on **8 of 14 `hw_photo` pages** (`hw-beethoven-op135`, `hw-mozart-k387`, `hw-debussy-berceuse-heroique`, `hw-debussy-la-plus-que-lente`, `hw-mendelssohn-herbstlied-d-b`, `hw-beethoven-das-blumchen-wunderhold-d-b`, `hw-cornelius-botschaft-d-b`, `hw-hensel-im-herbste-d-b`) and `-3.75°`/`-3.25°` on 2 more. Dark desk borders and perspective margins dominate horizontal ink-profile variance, causing deskew to rotate *toward* the desk edge instead of the staff lines (e.g., `hw-beethoven-das-blumchen-wunderhold-d-b` photo plunges `0.468 → 0.190`).
   - **CLAHE amplifies paper noise:** Local histogram equalization turns yellowed paper grain, foxing, and bleed-through into dark false-ink pixels, hurting both `hw_clean` (`-0.0138`) and `hw_photo` (`-0.0290`), with `deskew_clahe_1536` compounding both (`-0.0284` clean, `-0.0538` photo).
   - **Reranker poisoning:** `best_fsa_beat` picked a CLAHE/deskew view on **6/14 `hw_clean` pages** and **6/14 `hw_photo` pages**. When a corrupted view hallucinates a simpler rhythmically regular sequence, `fsa_beat_score` rewards its high beat-conservation rate (e.g., picking `deskew_1536` at `0.190` over `base_1536` at `0.468` and `scale_1280` at `0.504` on `hw-beethoven-das-blumchen-wunderhold-d-b` photo).
4. **Root Cause 2 — Pure Multi-Scale TTA Has Huge Capturable Signal:**
   - In `hw_clean`, `13/14` oracle winners come from `{scale_1280, base_1536, scale_1792}` (`oracle = 0.3005`, capturing `94.0%` of 6-view oracle). Simply removing the 3 destructive preprocessing views turns `best_fsa_beat` into **`0.2954` (`+0.0343` vs `base_1536`, `+0.0160` vs `scale_1280`, capturing `87.1%` of scale-only oracle!)**.
   - On `hw_photo`, removing `clahe`/`deskew` flips `best_fsa_beat` from `-0.0248` to **`+0.0053` (`0.1969` vs `0.1916`, and `+0.0106` over `scale_1280`)**, leaving `+0.0153` uncaptured headroom vs scale-only oracle (`0.2122`) because `fsa_beat_score` has no model confidence term (`mean_logprob`) to reject low-confidence hallucinations.
5. **Latency & Hardware:**
   - Single-view HW decode averages `~8.2–8.4 s/page` on L4 BF16 (`~49 s/page` for 6 views). Pruning to 2–4 clean scale views cuts latency by `1.5–3×`. Zero OOMs, zero hangs.

---

## Follow-Up Action / Successor Experiment

- **Verdict:** **`ITERATE`** (matches PLAN.md criterion: `oracle_best_view` beats `base_1536` by `+0.0419` clean / `+0.0327` photo, both `>= +0.030`, while 6-view unsupervised selectors captured `< 50%` of headroom).
- **Queued Successor Experiment:** [`2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta`](../2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta/README.md) (`depends_on: ["2026-11-26T01-38-ov-36-hw-multiview-tta-voting"]` in `experiments/overnight_queue/queue.json`).
  - **Drop destructive `clahe`, `deskew`, and OOD `1792px` views completely.**
  - Evaluate **5 clean in-distribution scale views** (`{1024, 1152, 1280, 1408, 1536}`) at `max_len=3840` (per `ov-35`) across all 14 real historical HW pieces (`28` pages, `140` decodes, `~12 min` / `~1.0 CU` on Colab L4).
  - Add **exact teacher-forced length-normalized model log-probability (`mean_logprob`) and entropy (`mean_entropy`)** (~30 ms/view) combined with `fsa_beat_score` and `mbr_vote_score`, plus a fast 2-view `{1280, 1536}` Pareto slice, to lock a production-ready multi-scale TTA reranker.
