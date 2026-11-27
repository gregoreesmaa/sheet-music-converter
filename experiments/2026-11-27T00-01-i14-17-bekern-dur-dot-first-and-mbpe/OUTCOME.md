# OUTCOME — `2026-11-27T00-01-i14-17-bekern-dur-dot-first-and-mbpe`

- **Status:** `COMPLETED` *(Scientifically **INVALID** — Control Baseline Collapsed to `0.0000` Note F1)*
- **Verdict:** **`ITERATE`**
- **Started / Completed:** `2026-11-27T17:00:00Z` → `2026-11-27T19:59:51Z`
- **Elapsed Time:** `~10,791s` (`~3.0 hr`; ~`670s` train/arm + ~`2,900s` collapsed greedy decode/arm)
- **Platform / GPU:** Google Colab `L4` (BF16, `600` steps/arm, `small_45m`, custom `AdamW` loop)
- **HF Artifacts:**
  - Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T00-01-i14-17-bekern-dur-dot-first-and-mbpe.json`
  - Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-27T00-01-i14-17-bekern-dur-dot-first-and-mbpe.json`

---

## Measured Results

| Arm | Vocab Size | Round-Trip Exact | Mean Len (Orig → Enc) | Compression Ratio | `T > 3840` Rate | Train Elapsed (s) | Final 25-Step Loss | `proxy_val` Token F1 | `proxy_val` Note F1 | `suite31_hw` Overall Note F1 | `suite31_hw` Rhythm F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0_control_bekern` | `265` | `N/A` | `876.1 → 876.1` | `1.0000` | `0.0%` | `654.8` | **`1.2620`** | `0.0082` | `0.0000` | `0.0000` | `0.0000` |
| `arm1_dur_dot_first_chord_drop` | `265` | **`50/50` (`100%`)** | `876.1 → 835.1` | `0.9531` (`-4.7%`) | `0.0%` | `680.3` | `1.2954` | `0.0085` | `0.0000` | `0.0000` | `0.0000` |
| `arm2_sticky_barrest_mbpe` | `408` (`+143`) | **`37/50` (`74%` LOSSY)** | `876.1 → 435.4` | **`0.4969` (`-50.3%`)** | `0.0%` | `678.2` | `2.3366` | `0.0082` | `0.0000` | `0.0000` | `0.0000` |

*(All 4 `suite31_hw` slices — `pr_clean` `n=31`, `pr_photo` `n=31`, `hw_clean` `n=14`, `hw_photo` `n=14` — scored `0.0000` Strict Note F1 and `0.0000` Rhythm F1 across all 3 arms.)*

---

## Scrutiny & Anomaly Check

1. **CRITICAL BUG #1 — Total Free-Running Decode Collapse Across All 3 Arms (Including Control):**
   - `arm0_control_bekern` scored `0.0000` Strict Note F1 on all 90 `suite31_hw` pages and `0.0082` `token_f1` on `proxy_val`, rendering the OMR accuracy comparison completely non-diagnostic.
   - **Root cause (`run_colab.py:866-943`):** Instead of calling `train_uni_v2.run_train_v2` (Muon + WSD decay + MTP-4 + aux losses + `batch_size=2 × grad_accum=4`), the script ran a bare custom `AdamW(lr=3e-4)` loop with `batch_size=1 × grad_accum=4` (`2,400` page views from random init), NO LR decay, NO aux/MTP losses, and then **overwrote the model with uncorrected `decay=0.99` EMA** (`model.load_state_dict(ema_state)`) initialized from random step-0 weights without ever evaluating raw weights. Every arm then spent ~40 minutes greedily decoding empty/garbage sequences to `EVAL_MAX_LEN`.
2. **CRITICAL BUG #2 — Arm 2 Round-Trip Is Lossy (`37/50` = `74%`, Not `100%`):**
   - In `encode_sticky_and_barrest` (`run_colab.py:451-455`), encountering `"chord"` only emits `"chord"` and increments `i += 1`. On the next iteration, any chord with an explicit duration override (`chord <dur> [dot]* <step> <oct>`) falls into the main `DURS_SET` branch, overwriting `sticky_dur` or stripping the chord's explicit duration — breaking inversion in `decode_sticky_and_barrest` (which consumes the full chord without touching `sticky_dur`).
   - Startup logged `arm2: 37/50` instead of asserting `50/50` before launching GPU training.
3. **Zero `T > 3840` Truncation Exercised on `proxy_2k`:**
   - `trunc_gt_3840_rate_orig = 0.0` (mean length `876.1`), so `iter10_5_proxy_2k` never exercised the `T > 3840` truncation hypothesis.
4. **Salvaged Structural Signal:**
   - **Arm 1 (`dur [dot]*` + chord dur drop):** Verified **`50/50` (`100%`) lossless**, zero vocab growth (`265`), `-4.7%` token reduction (`876.1 → 835.1`), and virtually identical teacher-forced CE loss (`1.2954` vs `1.2620`).
   - **Arm 2 (sticky dur + `<bar-rest-N>` + `128` intra-bar mBPE):** Demonstrated massive **`-50.3%` sequence compression** (`876.1 → 435.4`), confirming huge length headroom, but `+143` new vocab tokens (`265 → 408`) severely underfit at 600 steps from scratch (`loss 2.3366` vs `1.2620`).

---

## Verdict Rationale (`ITERATE`)

- **Why not `PROMOTE` or `DISCARD`:** The control baseline collapsed to `0.0000` Note F1 due to a runner bug (bare `AdamW` + uncorrected EMA-only eval), so the hypothesis (`dur [dot]*` + sticky duration + intra-bar mBPE) was **never validly tested** on OMR accuracy. Meanwhile, Arm 1's lossless `-4.7%` compression and Arm 2's `-50.3%` compression show clear structural upside once the bugs are fixed.

---

## Follow-Up Action / Successor Experiment

- **Next step (to be queued in `experiments/overnight_queue/queue.json` by coordinator):** Create `i14-17b` with 4 mandatory fixes:
  1. Fix `encode_sticky_and_barrest` to skip/preserve explicit chord overrides atomically and **hard-assert `100%` round-trip** across the entire dataset before training starts.
  2. Train via `train_uni_v2.py` (Muon + WSD + MTP/aux + `batch_size=2 × grad_accum=4`) or warm-start from a pretrained `UniOMRv2` checkpoint, and evaluate **both RAW and EMA** weights (aborting if `arm0_control` `val_token_f1 < 0.20`).
  3. Split Arm 2 into **Arm 2A (Arm 1 + sticky duration + `<bar-rest-N>`, no BPE)** vs **Arm 2B (Arm 2A + `+32` BPE merges)** to isolate sticky-duration gains from `+128` BPE underfitting.
  4. Measure `T > 3840` truncation reduction on dense full-page manifests (`iter12` / `suite31_hw` dense piano/orchestral pages) in addition to `proxy_2k`.
