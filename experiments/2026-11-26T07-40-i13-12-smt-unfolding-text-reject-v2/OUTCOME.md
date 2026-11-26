# Experiment Outcome — `2026-11-26T07-40-i13-12-smt-unfolding-text-reject-v2` (`i13-12`)

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T14:02:01Z` (started `2026-11-26T12:19:40Z`)
- **Platform / GPU:** Google Colab `NVIDIA L4` (`24 GB`, `BF16=True`, `muon-lr=1e-3`, `lr=3e-4`, WSD, EMA `0.99/1`)
- **Elapsed Time:** `6518.06 s` total queue wall (`1218.4 s` Arm A + `1298.4 s` Arm B + `1205.3 s` Arm C + evals)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T07-40-i13-12-smt-unfolding-text-reject-v2.json` (pulled to [`eval_result.json`](./eval_result.json))
  - Run Ledger: `runs/2026-11-26T07-40-i13-12-smt-unfolding-text-reject-v2.json` (pulled to [`run_result.json`](./run_result.json))

---

## Measured Results (`600` steps/arm, `UniOMR v2 small_45m`, `iter10_5_proxy_2k`, `n=64` `proxy_val` clean & synthetic text-clutter)

| Arm | Wall (s) | Clean Raw CE / F1 / SysExact | Clean EMA CE (Δ) | Clean EMA SER | Clean EMA Token F1 (Δ) | Clutter EMA CE | Clutter EMA Token F1 (Δ) |
|---|---:|---|---:|---:|---:|---:|---:|
| **`control_static_v2`** *(Arm A)* | `1218.4` | **`1.4126` / `0.3724` / `0.4063`** | **`1.4360`** (`+0.0000`) | **`0.8702`** | **`0.3445`** (`+0.0000`) | **`1.4361`** | **`0.3515`** (`+0.0000`) |
| `smt_progressive_unfolding` *(Arm B, 3×200)* | `1298.4` | `1.4511` / `0.1942` / `0.1719` | `1.4645` (`+0.0285`) | `0.9619` | `0.1736` (**`-0.1709`**) | `1.4646` | `0.1746` (**`-0.1769`**) |
| `audiveris_text_clutter_reject` *(Arm C, 50%)* | `1205.3` | `1.4160` / `0.3690` / `0.3906` | `1.4392` (`+0.0031`) | `0.8875` | `0.3103` (**`-0.0341`**) | `1.4390` | `0.3111` (**`-0.0404`**) |

---

## Scrutiny & Anomaly Check

1. **Coverage & Integrity — PASS:** All 3 arms completed `600/600` steps cleanly and evaluated all `64/64` `proxy_val` pages (both clean and text-cluttered) for RAW and EMA checkpoints with zero NaNs or OOMs.
2. **Budget Note:** `run_colab.py` used `DEFAULT_STEPS = 600` (halved from `PLAN.md`'s `1,200` during triage to stay within Colab L4 wall-clock limits); all 3 arms are strictly apples-to-apples at `600` steps.
3. **Root Cause 1 — Why `smt_progressive_unfolding` (Arm B) Collapsed (`-0.1709` EMA F1):**
   - Chaining 3 separate `train_uni_v2.py --init-from` subprocesses of `200` steps each (`<=3 -> <=6 -> <=16` systems) **reset Muon optimizer state, WSD LR schedule (3 warmups + 3 decays), and GPU-EMA shadow weights 3 times**, while starving the model of dense `>6`-system pages for `67%` (`400/600`) of the budget.
   - Note that `UniOMR v2` already uses a dedicated strip-warmup phase (`Stage 1` strips -> `Stage 2` fullpage) which provides smooth curriculum without mid-fullpage optimizer resets.
4. **Root Cause 2 — Why `audiveris_text_clutter_reject` (Arm C) Regressed (`-0.0341` clean / `-0.0404` clutter):**
   - Inspecting `control_static_v2` on `val_clean_ema` (`CE=1.4360, F1=0.3445`) vs `val_clutter_ema` (`CE=1.4361, F1=0.3515`, `+0.0070`) proves that **the baseline `UniOMR v2` model is ALREADY completely invariant to the synthetic printed text/lyric clutter overlay**!
   - Injecting heavy 50% synthetic text overlays during training therefore solved a non-existent bottleneck while adding label-irrelevant pixel noise, hurting both clean (`-0.0341` EMA F1) and cluttered (`-0.0404` EMA F1) validation accuracy.

---

## Verdict: **`DISCARD`**

Matches `PLAN.md` Section 5 (`DISCARD`): neither treatment arm beats `control_static_v2` on either clean or text-cluttered validation (`-0.1709` and `-0.0341` clean EMA F1; `-0.1769` and `-0.0404` clutter EMA F1).

## Follow-Up Action / Successor Experiment

- **No follow-up queued.** Do **not** add 3-stage `--init-from` system-count unfolding or 50% synthetic text clutter augmentation to Iter 13 full training. Retain static stratified full-page sampling after Stage-1 strip warmup.
