# OUTCOME — `2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode` (`i13-16`)

- **Status:** `COMPLETED` (100% silent no-op due to vocabulary-string bug)
- **Completed UTC:** `2026-11-26T11:01:43Z` (started `09:33:40Z`)
- **Platform / GPU:** `Colab Tesla T4 (cc=7.5, bf16=False)` (FP32 fallback)
- **Elapsed Time:** `5283.08 s` (`88.1 min`)
- **Base Checkpoint:** `checkpoints/iter10_5_best/uni_omr_ema_best.pt` (`uni_omr_v1`, step `8000`, `val_ser=0.314308`, `missing_keys=[]`, `unexpected_keys=[]`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - `evals/2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode.json`
  - `runs/2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode.json`

## Measured Results (`max_dim=1280`, `max_len=3840`, `beam=3`, `max_redecodes=6`)

| Arm | `hw_clean` F1 (`n=14`) | `hw_photo` F1 (`n=14`) | `pr_clean` F1 (`n=10`) | `pr_photo` F1 (`n=10`) | Mean Triggers | Mean Accepted |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_greedy` | `0.264263` | `0.187995` | `0.582040` | `0.578630` | `0.0` | `0.0` |
| `conf_minp_0.65` | `0.264263` (`+0.0000`) | `0.187995` (`+0.0000`) | `0.582040` (`+0.0000`) | `0.578630` (`+0.0000`) | **`0.0`** | **`0.0`** |
| `conf_minp_0.75` | `0.264263` (`+0.0000`) | `0.187995` (`+0.0000`) | `0.582040` (`+0.0000`) | `0.578630` (`+0.0000`) | **`0.0`** | **`0.0`** |
| `conf_minp_0.82` | `0.264263` (`+0.0000`) | `0.187995` (`+0.0000`) | `0.582040` (`+0.0000`) | `0.578630` (`+0.0000`) | **`0.0`** | **`0.0`** |
| `conf_margin_0.25` | `0.264263` (`+0.0000`) | `0.187995` (`+0.0000`) | `0.582040` (`+0.0000`) | `0.578630` (`+0.0000`) | **`0.0`** | **`0.0`** |

## Scrutiny & Anomaly Check

1. **Coverage & Integrity:** All `48` pages × `5` arms ran with `0` decode failures and `0` NaNs, **but every single treatment arm was a 100% no-op (`mean_triggers == 0.0`, `mean_accepted == 0.0`, 100% ties across all 48 pages).**
2. **CRITICAL ROOT CAUSE — Vocabulary Token Mismatch (`run_colab.py:55`):**
   - Line 55 hardcoded `BAR_BOUNDARY_TOKS = {"|", "<bar>", "<sys>"}`.
   - **None of `"|"`, `"<bar>"`, or `"<sys>"` exist in `UniOMR`'s 237-token vocabulary** (verified via `eval_uni_suite31._vocab_spec(237)`). The actual barline tokens are `{"bar", "bar-double", "bar-light-heavy", "bar-repeat-left", "bar-repeat-right"}`.
   - As a result, `boundary_ids` resolved to `set()`, `bar_start_kv_len` remained `0` for the entire decode, and line 448 (`if should_trigger and bar_start_kv_len > 0:`) never evaluated to `True` even at `eos_id`.
3. **Platform & Checkpoint Drift:** Ran on T4 FP32 (`88.1 min`, `23.5–26.6 s/page` on HW) and picked `checkpoints/iter10_5_best/uni_omr_ema_best.pt` (`val_ser=0.3143`) rather than `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`val_ser=0.1923`).

## Verdict: **`ITERATE`**

- The Mathpix confidence-gated per-bar beam re-decoding hypothesis is **completely untested** (0 triggers fired across 48 pages due to the `BAR_BOUNDARY_TOKS` bug), NOT falsified.

## Follow-Up Action / Successor Experiment

- **Proposed Successor:** `i13-16b-mathpix-conf-gated-redecode-fixed` (depends on `2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode`):
  1. Fix `BAR_BOUNDARY_TOKS = {"bar", "bar-double", "bar-light-heavy", "bar-repeat-left", "bar-repeat-right"}` and add `assert len(boundary_ids) >= 3` at startup.
  2. Fix bar-1 rollback guard (track BOS KV len `>= 1` after step 0, or trigger cleanly from bar 1 onward) and assert `total_triggers > 0` after the first HW page.
  3. Run on L4 BF16 against `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`~15–20 min`).
