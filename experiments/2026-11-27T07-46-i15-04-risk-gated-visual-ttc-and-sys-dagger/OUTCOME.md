# OUTCOME — `2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger` (`i15-04`)

- **Status:** `COMPLETED`
- **Verdict:** **`DISCARD`** (Part A: **`DISCARD`** — failed all 3 gates; Part B: **`DISCARD`** — failed both gates)
- **Wave / Short ID:** Wave `13` / `i15-04`
- **Platform / GPU:** Local `NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, BF16 native; ran locally as `task-5314` after fixing 3 bugs from the earlier failed Colab A100 attempt: `vocab_size=237` compat encoding, BF16 `q_proj`/`k_proj` dtype alignment, and `_pil_to_tensor` `[0,1]` normalization)
- **Timestamps (UTC):** `2026-11-28T02:07:54Z` → `2026-11-28T05:06:51Z`
- **Elapsed Time:** `10,737.82 s` (`178.96 min` / `2.98 h` total: Part A `3,584.70 s` [`59.7 min`], Part B `arm_b0` `2,915.82 s` [`48.6 min`], Part B `arm_b1` `4,222.83 s` [`70.4 min`])
- **Baseline Checkpoint:** `checkpoints/iter13_v2_mid/uni_omr_v2_ema_latest.pt` (`uni_omr_v2`, `step=744`, `vocab_size=237`)
- **HF Artifacts:**
  - Eval JSON: `evals/2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger.json`
  - Run Ledger: `runs/2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger.json`

---

## 1. Measured Results

### Part A — `part_a_risk_gated_visual_ttc` (Zero-Train, OMR-NorthStar `dev-v1`, 125 pages, `rust:assembly-serialize.exe`)

| Metric | `arm0_greedy` (`1408px`) | `arm1_risk_gated_ttc` | Delta (`arm1 - arm0`) | Gate (`PLAN.md`) | Gate Status |
|---|---:|---:|---:|---|---|
| **NorthStar (`NS`)** | **`34.9265`** `[27.67, 43.09]` | `34.5117` `[27.21, 43.08]` | **`-0.4148`** | `>= +3.0` | **FAIL** |
| **Tier-Weighted NS** | **`55.7121`** | `55.5763` | `-0.1358` | — | regress |
| **`M_geom`** | `59.2440` | **`59.9198`** | `+0.6758` | — | slight gain |
| **`worst_stratum_q` (`HWM`)** | `22.8714` | **`23.6777`** | `+0.8063` | — | slight gain |
| **`cfr_max` (`HWM` CFR)** | **`0.6071`** (`60.71%`) | `0.6429` (`64.29%`) | **`+0.0358` (`+3.58 pp` worse)** | — | **REGRESS** |
| **CDR** | `85.4316` (CFR `8.57%`) | **`85.8437`** (CFR `8.57%`) | `+0.4121` | `>= 0.0` | pass |
| **ALS** | `75.3789` (CFR `16.13%`) | **`75.7412`** (CFR `16.13%`) | `+0.3623` | — | slight gain |
| **RWP** | `83.6408` (CFR `9.68%`) | **`83.7339`** (CFR `9.68%`) | `+0.0931` | — | flat |
| **HWM** | `22.8714` (CFR `60.71%`) | **`23.6777`** (CFR `64.29%`) | `+0.8063` (CFR `+3.58 pp`) | — | CFR worse |
| **`S_note` / `S_struct` / `S_expr` / `S_play`** | `0.7337 / 0.8318 / 0.4961 / 0.7060` | `0.7409 / 0.8278 / 0.4971 / 0.7153` | `+0.0072 / -0.0040 / +0.0010 / +0.0093` | — | mixed |
| **`median_wFPP` / `p90_wFPP`** | `118.0 / 1275.3` | **`114.0 / 1215.7`** | `-4.0 / -59.6` | — | slight gain |
| **`mean_NTSR` / `NUPR`** | `0.5889 / 0.1760` | **`0.6012 / 0.1680`** | `+0.0123 / -0.0080` | — | slight gain |
| **Negative Control (`NMHR`)** | `0.0%` (`4/4` PASS) | `0.0%` (`4/4` PASS) | `0.0` | `0.0%` | PASS |
| **Escalation Rate (All 125p)** | `0.0%` | **`89.6%`** (`112/125`) | `+89.6 pp` | — | over-triggered |
| **CDR Escalation Rate** | `0.0%` | **`80.0%`** | `+80.0 pp` | **`<= 15.0%`** | **FAIL (5.33× over)** |
| **Replacement Rate (All 125p)** | `0.0%` | **`68.8%`** (`86/125`) | `+68.8 pp` | — | over-replaced |
| **Mean Latency (`s/page`)** | **`3.9553 s`** | `28.4737 s` | **`7.1989×`** | **`<= 1.45×`** | **FAIL (4.96× over)** |

### Part B — `part_b_sys_dagger_probe` (`1,200` Steps/Arm on `iter10_5_proxy_2k`, `suite31_hw` 90 pages)

| Metric (`suite31_hw`, 90p) | `arm_b0_sft_control` (RAW) | `arm_b1_sys_dagger` (RAW) | Delta RAW (`b1 - b0`) | `arm_b0_sft_control` (EMA) | `arm_b1_sys_dagger` (EMA) | Delta EMA (`b1 - b0`) | Gate (EMA) | Gate Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **`printed_clean`** | **`0.6889`** | `0.6466` | `-0.0423` (`-4.23 pp`) | **`0.6861`** | `0.6451` | `-0.0410` (`-4.10 pp`) | — | regress |
| **`printed_photo`** | `0.6600` | **`0.6902`** | `+0.0302` (`+3.02 pp`) | **`0.6785`** | `0.6429` | `-0.0356` (`-3.56 pp`) | — | regress |
| **`printed_mean`** | **`0.6744`** | `0.6684` | `-0.0060` (`-0.60 pp`) | **`0.6823`** | `0.6440` | **`-0.0383` (`-3.83 pp`)** | **`>= -0.3 pp`** | **FAIL** |
| **`hw_clean`** | **`0.2091`** | `0.2012` | `-0.0079` (`-0.79 pp`) | `0.1998` | **`0.2007`** | `+0.0009` (`+0.09 pp`) | — | flat |
| **`hw_photo`** | `0.1538` | **`0.1797`** | `+0.0259` (`+2.59 pp`) | **`0.1508`** | `0.1459` | `-0.0049` (`-0.49 pp`) | — | regress |
| **`hw_mean`** | `0.1814` | **`0.1905`** | `+0.0091` (`+0.91 pp`) | **`0.1753`** | `0.1733` | **`-0.0020` (`-0.20 pp`)** | **`>= +3.0 pp`** | **FAIL** |
| **`all90`** | **`0.5210`** | `0.5197` | `-0.0013` (`-0.13 pp`) | **`0.5246`** | `0.4976` | **`-0.0270` (`-2.70 pp`)** | — | **FAIL** |
| `train_loss_tail50` | `0.12154` | `0.15398` | `+0.03244` | — | — | — | — | higher |
| Micro-batches (`sft / dag / ap`) | `4800 / 0 / 0` | `3604 / 544 / 652` | `75.1% / 11.3% / 13.6%` | — | — | — | — | nominal |
| Elapsed (`s`) | `2,915.82 s` | `4,222.83 s` | `+44.8%` (`1.45×`) | — | — | — | — | slower |
| Eval latency (`s/page`) | `11.105 s` | `11.258 s` | `+1.4%` | `10.836 s` | `11.435 s` | `+5.5%` | — | — |

---

## 2. Scrutiny & Anomaly Check

1. **Coverage & Execution Integrity:**
   - **Part A:** All `125/125` OMR-NorthStar `dev-v1` pages were scored cleanly on both arms with the canonical Rust exporter (`rust:assembly-serialize.exe`) and `4/4` negative-control pages passed (`NMHR = 0.0%`). Zero crashes or missing pages.
   - **Part B:** Both arms completed `1,200/1,200` optimizer steps (`4,800` micro-batches, `0` NaNs) and evaluated both RAW and EMA checkpoints across all `90/90` `suite31_hw` pages.

2. **Part A Root Causes (Why All 3 Gates Failed):**
   - **Gate over-triggering (`89.6%` overall / `80.0%` CDR escalation, `7.20×` latency — reproducing `i14-14`'s `7.59×` blowup):** `_estimate_visual_sys_count()` uses a crude horizontal ink-projection heuristic with a fixed `2.2%` merge gap, so routine grand-staff spacing or title/lyric ink shifts `n_sys_img` by `±1` on clean CDR pages. Because `abs(n_sys - n_sys_img) >= 1` was classified as `hard_cfr = True`, it both escalated `80.0%` of CDR pages and **bypassed the `+0.08` composite win-margin guard**, causing **`68.8%` of all pages** to replace `y_0`.
   - **Unreliable `attn_ink_cov` proxy:** Computing coverage from only the last decoder layer's head-averaged cross-attention mass (`mass >= 0.25 * Lq / Lk`) with a 1D-interpolated ink grid when `keep_indices` is absent from `router_info` fails to distinguish true missing systems from sharp sparse attention.
   - **Net negative `NS` (`-0.4148`) despite `7.20×` compute:** Although `M_geom` nudged up `+0.68` and `median_wFPP` improved by `-4.0` (`118.0 → 114.0`), replacing `y_0` pushed an additional HWM page into catastrophic failure (`HWM CFR 60.71% → 64.29%`, `+3.58 pp`; `T4 HWM 16.55 → 15.04`) and regressed `T2 CDR` (`78.67 → 73.08`), `T4 CDR` (`95.93 → 94.57`), and `T5` across all three printed strata (`CDR -0.49`, `ALS -1.50`, `RWP -0.89`), dragging `NS` down `-0.4148` and `Tier-W NS` down `-0.1358`.

3. **Part B Root Causes (Why Both Gates Failed):**
   - **EMA regression across both domains (`printed_mean -3.83 pp`, `hw_mean -0.20 pp`, `all90 -2.70 pp`):** Even on RAW weights, `hw_mean` gained only `+0.91 pp` (far below the `+3.0 pp` gate) while `printed_clean` dropped `-4.23 pp` (`printed_mean -0.60 pp`).
   - **Misaligned DAgger splice:** `_build_sys_dagger_example()` rolls out a fixed-budget greedy prefix (`gen_budget = min(256, gt_prefix_len)`) without stopping at system boundary `split_k`, then splices ground-truth `<sys>` + system `split_k` directly after an arbitrary mid-system prefix. This teaches discontinuous system jumps rather than true system-boundary recovery.
   - **Destructive Anti-Prior token corruption + missing MTP loss:** `_build_antiprior_example()` corrupts `15%` of decoder inputs into arbitrary uniform random tokens `[0, pad_id)` (not valid counterfactual notation edits), and both `dagger` and `antiprior` micro-batches (`24.9%` of training) drop `mtp_loss_weight=0.25` (`return_mtp=False`), destabilizing the decoder and EMA trajectory (`+44.8%` wall-clock overhead for `-2.70 pp` EMA `all90`).
   - **Proxy fine-tuning drift note:** Both `arm_b0` and `arm_b1` dropped on `suite31_hw` relative to the full `iter13_v2_mid` baseline because fine-tuning at `LR_MAX=1e-4` on the small `iter10_5_proxy_2k` (`1280px`, `max_len=1536`) and evaluating with unmasked `model.generate` (`1280px`, `max_len=2048`) narrows generalization; however, the apples-to-apples `arm_b1 - arm_b0` delta is cleanly negative on EMA (`-3.83 pp` printed, `-0.20 pp` HW) and fails all `PLAN.md` gates.

---

## 3. Verdict & Follow-Up Action

- **Verdict:** **`DISCARD`** (both Part A and Part B meet `PLAN.md` Section 7 `DISCARD` criteria: Part A `NS = -0.4148 < +0.5` and `latency = 7.20× > 1.8×`; Part B EMA `HWM = -0.20 pp < +0.5 pp` and `Printed = -3.83 pp < -0.5 pp`).
- **Action:**
  - Do **NOT** wire heuristic visual-projection TTC (`_estimate_visual_sys_count` + last-layer `attn_ink_cov` reranking) into UniOMR production inference.
  - Do **NOT** wire unaligned fixed-budget `<sys>`-DAgger or `15%` random-token Anti-Prior into the Iter 15 Stage-2 training recipe.
  - No follow-up experiment queued; falsified hypotheses documented here and in `progress.md` so future iterations do not repeat heuristic pixel-projection system-count escalation or unaligned prefix-spliced DAgger.
