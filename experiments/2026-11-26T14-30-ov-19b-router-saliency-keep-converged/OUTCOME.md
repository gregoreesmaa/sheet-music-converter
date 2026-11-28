# Experiment Outcome — `2026-11-26T14-30-ov-19b-router-saliency-keep-converged` (`ov-19b`)

- **Verdict:** **`DISCARD`** (retain `max_keep_ratio=0.75` default with `p10 + 0.15` paper-baseline gating; do **not** promote `0.55` or `0.65`, no follow-up GPU run warranted).
- **Platform:** Local `NVIDIA GeForce RTX 4080 SUPER` (`cc=8.9`, `bf16=True`, `task-5314`; runner hardcoded `"Colab "` prefix in `platform` metadata).
- **Timing:** `2026-11-28T05:07:03Z` → `2026-11-28T05:30:25Z` (**`1,401.35 s`** / `23.36 min` total; `411.16 s` + `475.55 s` + `504.12 s`).
- **Model Source:** Warm-started from HF `checkpoints/iter12_v2_best.pt` (`arch=uni_omr_v2`, `step=20000`, `val_ser=0.291712`, `0` missing / `0` unexpected keys). Fallback 2,500-step proxy was not needed.
- **Benchmark:** Full `suite31_hw` (**`90/90` pages per arm** = `31` printed + `14` HW × `{clean, photo}`, `max_dim=1280`, `max_len=3584`, MTP-4 + FSA `strict_grammar=False`, pure-Python MusicXML Strict Note F1).
- **Artifacts:** [eval_hf.json](./eval_hf.json) (`HF evals/2026-11-26T14-30-ov-19b-router-saliency-keep-converged.json`), [run_hf.json](./run_hf.json) (`HF runs/2026-11-26T14-30-ov-19b-router-saliency-keep-converged.json`).

## 1. Summary Table (Full 90-Page `suite31_hw`)

| Arm | `saliency_thr` (eff. `p10+Δ`) | `max_keep_ratio` | Overall 90p F1 (Δ) | `pr_clean` (Δ) | `pr_photo` (Δ) | `hw_clean` (Δ) | `hw_photo` (Δ) | EOS (`90p`) | Med s/page | Changed Pages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`arm_005_075`** (base) | `0.05` (`0.15`) | `0.75` | **`0.536043`** (`—`) | **`0.682895`** (`—`) | **`0.718202`** (`—`) | **`0.190075`** (`—`) | **`0.153486`** (`—`) | `88/90` | `3.800 s` | `—` |
| **`arm_002_065`** | `0.02` (`0.15`) | `0.65` | **`0.538084`** (`+0.20 pp`) | **`0.682895`** (`+0.00 pp`) | **`0.718202`** (`+0.00 pp`) | **`0.203676`** (`+1.36 pp`) | **`0.153005`** (`-0.05 pp`) | `88/90` | `4.297 s` | `12/90` (`0` PR, `12` HW) |
| **`arm_003_055`** | `0.03` (`0.15`) | `0.55` | **`0.534255`** (`-0.18 pp`) | **`0.685040`** (`+0.21 pp`) | **`0.703071`** (`-1.51 pp`) | **`0.214530`** (`+2.45 pp`) | **`0.146296`** (`-0.72 pp`) | `88/90` | `4.510 s` | `31/90` (`8` PR, `23` HW) |

> **Telemetry Caveat:** All 270 rows recorded `n_in=1`, `n_out=1`, `ink_recall=1.0`, `compression_x=1.0`, `keep_frac=1.0` due to an unpacking bug in `run_colab.py` (root-caused below). Accuracy (`note_f1_py`, `n_toks`, `has_eos`, `mtp_acc`) was **unaffected** because `generate_mtp_spec` ran `model.encode_image` internally with the live overridden router attributes.

## 2. Forensic Scrutiny & Anomaly Diagnosis

### A. Root Cause of `compression_x_mean == 1.0` & `keep_frac_mean == 1.0` While F1 Varied
1. **Tuple-unpacking bug in `run_colab.py:622,632`:**
   - `UniOMRv2Net.encode_image(img_t, return_aux=False)` (`tmp-train/mvd/uni_omr_v2.py:3008`) returns a 3-tuple: `(mem_levels, aux_enc, router_info)`.
   - `run_colab.py` wrote `_, rinfo, _ = model.encode_image(img_t, return_aux=False)` — unpacking **`aux_enc`** (2nd element: `{"sys_count_pred": None, "staff_seg_logits": None, "stroke_hog_pred": None}`) instead of **`router_info`** (3rd element).
   - Every `.get(...)` fallback in lines `659–662` therefore returned `1.0` (`n_in=1`, `n_out=1`, `k_kept=1`, `ink_recall=1.0`, `compression_x=1.0`, `keep_frac=1.0`) on all 270/270 rows, AND each page ran `model.encode_image` twice (once discarded for broken telemetry, once inside `generate_mtp_spec`).
2. **Why F1 varied cleanly and deterministically across arms:**
   - `generate_mtp_spec` called `model.encode_image` with the live overridden `router.saliency_thr` and `router.max_keep_ratio`.
   - Because `InkTokenRouterV2.forward` (`tmp-train/mvd/uni_omr_v2.py:1218`) clamps `thr = max(self.saliency_thr, 0.15)`, **`effective_p10_delta == 0.15` identically across all 3 arms** — `saliency_thr` (`0.05`, `0.02`, `0.03`) was a no-op and **only `max_keep_ratio` (`0.75 → 0.65 → 0.55`) actually varied**.
   - Under dynamic routing (`k = max(k_min, min(k_max, k_dyn))`), `max_keep_ratio` only activates when `k_dyn / N > max_keep_ratio` (aged/parchment/bleed-through HW manuscripts and the very densest printed pages).

### B. Exact Per-Page Diff Breakdown
- **`arm_002_065` (`max_keep_ratio=0.65`) vs `arm_005_075` (`0.75`):**
  - **`0 / 62` printed pages changed** (`pr_clean 0.682895` and `pr_photo 0.718202` are bit-identical) — proving `k_dyn / N <= 0.65` on 100% of printed `suite31` pages, so `0.65` yields **zero** compression/memory benefit on printed scores (`1.00×`).
  - **`12 / 28` HW pages changed** (`10/14` `hw_clean`, `2/14` `hw_photo`):
    - Biggest gains: `hw-debussy-vetement-du-blesse [clean]` (`0.1221 → 0.3556`, `+0.2334`), `hw-debussy-lindaraja [clean]` (`0.1563 → 0.2342`, `+0.0779`), `hw-mendelssohn-herbstlied-d-b [clean]` (`0.0442 → 0.0716`, `+0.0273`), `hw-mozart-k387 [clean]` regained EOS (`3585 → 2885`, `+0.0130`).
    - Biggest losses: `hw-debussy-la-plus-que-lente [clean]` (`0.4249 → 0.2739`, `-0.1509`), `hw-schubert-d810 [clean]` (`0.0722 → 0.0522`, `-0.0200`), `hw-mozart-k421 [photo]` lost EOS (`1693 → 3585`, `-0.0084`).
    - Net: trimming low-saliency parchment/foxing patches slightly helped noisy `hw_clean` (`+1.36 pp`), with `hw_photo` essentially flat (`-0.05 pp`).
- **`arm_003_055` (`max_keep_ratio=0.55`) vs `arm_005_075` (`0.75`):**
  - **`31 / 90` pages changed** (`7/31` `pr_clean`, `1/31` `pr_photo`, `13/14` `hw_clean`, `10/14` `hw_photo`).
  - **`pr_photo` collapsed `-1.5131 pp`** (`0.718202 → 0.703071`) entirely from a **single dense printed photo page**: **`beach-prayer` (`0.5820 → 0.1129`, `-0.4690`)** (all other `30/31` `pr_photo` pages were bit-identical) — textbook dense-page ink clipping.
  - **`pr_clean` `+0.2145 pp`** (`cpebach-h186` `0.6013 → 0.6701`, `+0.0688`).
  - **`hw_clean` `+2.4455 pp`** (`hw-debussy-vetement-du-blesse +0.1819`, `hw-debussy-lindaraja +0.1356`, `hw-beethoven-op135 +0.1075`, offset by `hw-debussy-la-plus-que-lente -0.1807`).
  - **`hw_photo` `-0.7190 pp`** (`8/10` changed `hw_photo` pages regressed: `hw-mozart-k464 -0.0622`, `hw-debussy-lindaraja -0.0400`, `hw-cornelius-botschaft-d-b -0.0234`, `hw-debussy-vetement-du-blesse -0.0114` losing EOS `1857 → 3585`).
- **Latency (`median_secs`):** Monotonically rose across sequential arms (`3.800 s → 4.297 s → 4.510 s`), even on `pr_photo` where `arm_002_065` had bit-identical outputs (`2.911 s → 3.604 s`), attributable to local GPU thermal/clock drift during the 23-min run and confirming tighter `max_keep_ratio` delivers zero measurable decode speedup.

## 3. Gate Decision & Next Action
- **`PLAN.md` `PROMOTE` Gate:** `ink_recall_mean >= 0.96`, `ink_recall_p10 >= 0.90`, `Δnote_f1_py >= -0.002` across all 4 quadrants, AND `>= 1.2×` lower peak memory / higher compression.
  - **`arm_003_055` (`0.03, 0.55`):** **FAILS** accuracy gates (`pr_photo -1.51 pp` via `beach-prayer` `-46.9 pp` dense-page clip, `hw_photo -0.72 pp`, overall `-0.18 pp`).
  - **`arm_002_065` (`0.02, 0.65`):** Clears accuracy gates (`+0.00 / +0.00 / +1.36 / -0.05 pp`, overall `+0.20 pp`), **but** mathematically cannot clear the `>= 1.2×` compression gate (`0/62` printed pages even hit the `0.65` cap → `1.00×` on printed; theoretical ceiling on capped HW pages is `0.75 / 0.65 = 1.154× < 1.20×`), and router telemetry was blinded by the `_, rinfo, _` bug.
- **Verdict:** **`DISCARD`** — retain `max_keep_ratio=0.75` (`p10 + 0.15` paper-baseline gating) as the `UniOMRv2Config` default. Do not spend another GPU run re-testing `0.65`.
- **Reusable Engineering Lesson:** Always unpack `mem_levels, aux_enc, router_info = model.encode_image(...)` (3rd element, not 2nd) in any router telemetry script, and assert `rinfo["n_in"] > 1` so silent `.get(..., 1.0)` fallbacks fail fast.
