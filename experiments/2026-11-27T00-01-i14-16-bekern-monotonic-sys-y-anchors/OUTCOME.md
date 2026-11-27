# Outcome — `2026-11-27T00-01-i14-16-bekern-monotonic-sys-y-anchors`

- **Status:** `COMPLETED` (Wave 12, `colab_l4`)
- **Started / Completed (UTC):** `2026-11-27T07:31:50Z` → `2026-11-27T10:13:16Z`
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`device=cuda`, native `bf16=true`)
- **Elapsed Time:** `9,686 s` (`161.4 min` = `2.69 h` total wall-clock)
  - **Training (`600` steps/arm, `batch_size=1 × grad_accum=4`):** `arm0_control_ungrounded_sys` = `815.09 s` (`13.6 min`, `1.358 s/step`) | `arm1_sys_ybin64` = `995.67 s` (`16.6 min`, `1.659 s/step`, `+22.2%` slower)
  - **Evaluation (`3` arms × `138` pages = `414` greedy decodes) + I/O:** `~7,875 s` (`~131.3 min`, `81.3%` of total runtime — see Scrutiny #2)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T00-01-i14-16-bekern-monotonic-sys-y-anchors.json` (pulled to [eval_hf.json](./eval_hf.json))
  - Run Ledger: `runs/2026-11-27T00-01-i14-16-bekern-monotonic-sys-y-anchors.json` (pulled to [run_ledger_hf.json](./run_ledger_hf.json))

---

## 1. Training & `proxy_val` Results (`iter10_5_proxy_2k`, `600` Scratch Steps, `48` Val Pages)

| Metric | `arm0_control_ungrounded_sys` | `arm1_sys_ybin64_unmasked` | `arm1_sys_ybin64_monotonic_mask` | Delta (`arm1_mask` vs `arm0`) |
|---|---:|---:|---:|---:|
| Vocab size | `265` | `329` (`+64` y-bins) | `329` (`+64` y-bins) | `+64` (weight-incompatible) |
| Train wall-time (`600` steps) | `815.09 s` (`1.358 s/step`) | `995.67 s` (`1.659 s/step`) | shared `arm1` weights | `+22.2%` |
| `final_25step_loss` | **`1.2602`** | `1.2641` | `1.2641` | `+0.0039` (worse) |
| `proxy_val` `n_pages` | `48` | `48` | `48` | full coverage |
| `proxy_val` `token_f1` | `0.0080` | `0.0410` | **`0.0448`** | `+0.0368` (`5.6×`) |
| `proxy_val` `sys_count_mae` | **`2.4583`** | `36.4167` (`14.8×` worse!) | `2.7500` | `+0.2917` (**`+11.9%` worse**) |
| `proxy_val` `copy_loop_rate` | `1.0000` (`100%`) | `0.2500` (`25.0%`) | **`0.0000` (`0.0%`)** | **`-1.0000` (`-100%` rel)** |
| `proxy_val` `monotonic_violation_rate` | `0.0000` | `0.2500` | **`0.0000`** | `0.0000` (vs `0.2500` unmasked) |

---

## 2. `suite31_hw` OOD Readout (`90` Pages: `31` PR Clean, `31` PR Photo, `14` HW Clean, `14` HW Photo)

| Metric | `arm0_control_ungrounded_sys` | `arm1_sys_ybin64_unmasked` | `arm1_sys_ybin64_monotonic_mask` | Delta (`arm1_mask` vs `arm0`) |
|---|---:|---:|---:|---:|
| `n_pages` | `90` | `90` | `90` | full coverage (`31/31/14/14`) |
| `pr_clean` Strict Note F1 (`n=31`) | `0.0000` | `0.0011` | `0.0011` | `+0.0011` (noise floor) |
| `pr_photo` Strict Note F1 (`n=31`) | `0.0000` | `0.0057` | `0.0057` | `+0.0057` (noise floor) |
| `hw_clean` Strict Note F1 (`n=14`) | `0.0000` | `0.0045` | `0.0045` | `+0.0045` (noise floor) |
| `hw_photo` Strict Note F1 (`n=14`) | `0.0000` | `0.0046` | `0.0046` | `+0.0046` (noise floor) |
| **`overall` Strict Note F1 (`n=90`)** | `0.0000` | `0.0038` | **`0.0038`** | **`+0.0038`** (identical to unmasked) |
| **`sys_count_mae`** | **`2.2222`** | `40.6333` (`18.3×` worse!) | `2.8889` | `+0.6667` (**`+30.0%` worse**) |
| `dropped_system_rate` | `0.3778` | **`0.2667`** | `0.3000` | `-0.0778` (`-20.6%` rel) |
| **`copy_loop_rate`** | `1.0000` (`100%`) | `0.2889` (`28.9%`) | **`0.0000` (`0.0%`)** | **`-1.0000` (`-100%` rel)** |
| `monotonic_violation_rate` | `0.0000` | `0.2889` | **`0.0000`** | `0.0000` (vs `0.2889` unmasked) |

---

## 3. Scrutiny & Anomaly Check

1. **Full Coverage & Zero Crashes:** All 3 evaluation arms scored `48/48` `proxy_val` pages and `90/90` `suite31_hw` pages (`31` PR clean, `31` PR photo, `14` HW clean, `14` HW photo) with zero NaNs and clean HF artifact uploads.
2. **Severe Undertraining / Noise-Floor Confound (Why Literal Gate Math Is Misleading):**
   - Although `arm1_sys_ybin64_monotonic_mask` technically satisfies the literal `OR` clause in `PLAN.md` (`copy_loop_rate` `-100%` relative `>= 50%` and `strict_note_f1 = 0.0038 >= 0.0000 - 0.002`), **both arms are at the scratch noise floor (`CE ~ 1.26`, `Note F1 = 0.0000–0.0038`)**.
   - `run_colab.py` trained `UniOMRv2Net(small_45m)` **from random initialization** for only `600` steps with **`batch_size=1 × grad_accum=4 = 4` pages/step** (`2,400` page-views total — note `PLAN.md` stated `batch_size=2, grad_accum=4`, so actual effective batch size was half of plan). At `CE ~ 1.26`, `arm0` had not yet learned `<eos>` stopping and hit `100%` copy-loops to `max_len=1536` (which also caused eval to consume `~131 min` / `81.3%` of total wall-clock).
   - A `0.0038` Note F1 scratch proxy **cannot** validate a weight-incompatible vocabulary expansion (`VOCAB_SIZE: 265 → 329`) for Iter 15.
3. **Unmasked `<sys>` Explosion & `+30.0%` Masked `sys_count_mae` Regression (Noisy 1D Ink-Projection Labels):**
   - Without the monotonic mask, `arm1_sys_ybin64_unmasked` exploded `sys_count_mae` by **`18.3×`** on `suite31_hw` (`2.2222 → 40.6333`) and **`14.8×`** on `proxy_val` (`2.4583 → 36.4167`), with `28.9%` / `25.0%` monotonic violations.
   - Turning on the monotonic mask capped the `<sys>` spam and eliminated copy-loops (`1.0000 → 0.0000`, `monotonic_violation_rate: 0.0000`), **but `sys_count_mae` still regressed by `+30.0%` on `suite31_hw` (`2.2222 → 2.8889`) and `+11.9%` on `proxy_val` (`2.4583 → 2.7500`)**, while `final_25step_loss` was slightly worse (`1.2641` vs `1.2602`).
   - Root cause matches `PLAN.md`'s exact `ITERATE` diagnosis: `detect_monotonic_system_ybins()` uses a coarse `256×384` 1D horizontal ink projection with an arbitrary threshold and uniform `linspace` fallback whenever detected bands ≠ `n_sys`. On skewed, degraded, or handwritten pages, 1D horizontal projection smears staff bands and feeds noisy `<y-bin-*>` targets during training.
4. **Zero Note-F1 Benefit from the Mask Itself:** Across all 4 `suite31_hw` slices, `arm1_sys_ybin64_unmasked` and `arm1_sys_ybin64_monotonic_mask` produced **bit-identical Note F1 to 4 decimal places** (`0.0011 / 0.0057 / 0.0045 / 0.0046 / overall 0.0038`), while `dropped_system_rate` ticked slightly up under the mask (`0.2667 → 0.3000`).

---

## 4. Verdict: **`ITERATE`** (DO NOT Promote into Iter 15)

- **Mechanism verified:** The decode-time monotonic logit mask cleanly enforces `0.0%` monotonic violations and cuts `copy_loop_rate` from `1.0000 → 0.0000` (`-100%` rel) and `dropped_system_rate` from `0.3778 → 0.3000` (`-20.6%` rel).
- **Blocked from `PROMOTE`:** Training CE worsened (`1.2602 → 1.2641`), `sys_count_mae` regressed `+30.0%` on `suite31_hw` (`2.2222 → 2.8889`) and `+11.9%` on `proxy_val` (`2.4583 → 2.7500`), unmasked `arm1` suffered catastrophic `<sys>` spam (`sys_count_mae = 40.63`), and Note F1 (`0.0000` vs `0.0038`) sits entirely in the undertrained scratch noise floor.
- **Iter 15 Weight-Compatibility Decision:** **NO — do NOT wire append-only `<y-bin-00..63>` (`VOCAB_SIZE: 265 → 329`) into Iter 15.** Keep Iter 15 on the standard `265`-token `bekern+` vocabulary so checkpoints remain clean and we do not risk a `+30%` system-count regression from noisy 1D ink-projection anchors.

---

## 5. Follow-Up Action / Successor Experiment

Queue a targeted warm-started follow-up (post-Iter 15 or on converged Iter 14/15 weights, **not** from random scratch):
1. **Warm-start from a converged `UniOMRv2` checkpoint** (`Note F1 > 0.70`) with append-only `<y-bin-00..63>` embedding/head rows so A/B differences are measured in the real operating regime rather than `0.0038` scratch noise.
2. **Replace 1D horizontal ink projection** (`detect_monotonic_system_ybins`) with **exact staff-band y-centers** extracted from `staff_seg` masks / MuseScore layout metadata.
3. **Include a zero-vocab-change control arm** (standard `265`-token `<sys>` + decode-time repetition-loop guard / `staff_seg` system-count prior) so `<y-bin>` must prove net Note F1 + `sys_count_mae` gain over non-breaking inference guards before earning a vocab expansion.
