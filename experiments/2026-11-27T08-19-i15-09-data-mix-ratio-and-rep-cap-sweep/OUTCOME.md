# OUTCOME — `2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep` (`i15-09`)

**Verdict: `PROMOTE` (`arm1_hw_heavy_30_70`)**

---

## 1. Execution Summary & Provenance

- **Experiment ID:** `2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep` (`i15-09`, Wave 13)
- **Platform:** Local `NVIDIA GeForce RTX 4080 SUPER` (BF16 native)
- **Window:** `2026-11-27T22:57:23Z` → `2026-11-28T01:23:08Z` (`8,745.02 s` / `2.43 h`)
- **Warm-Start Checkpoint:** `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` (`UniOMRv2`, `step=20,000`, `vocab_size=237`, `d_model=384`)
- **2-Page Baseline Sanity Check:** `1.0000` (`printed_clean=1.0`, `printed_photo=1.0`, `0` truncated, `3.418 s/page`)
- **Protocol:** `800` optimizer steps/arm (`grad_accum=4` → `3,200` draws/arm), `AdamW(lr=5e-5, wd=0.01)`, `80`-step warmup, WSD cosine cooldown (`25%`, `min_lr_ratio=0.10`), `GPU_EMA_DECAY=0.99`, `train_max_dim=1280`, `eval_max_dim=1408`, `eval_max_len=3584`, `seed=42`.
- **Pool Composition (`iter10_5_proxy_2k.tar`):** `1,360` printed, `520` synth-HW, `120` unique `real_hw_archival`, `16` synthesized `neg_control`, `0` `rism_pseudo`, `0` `real_hw_stitched` (`rism_pseudo` & `real_hw_stitched` target weights spilled proportionally into `synth_hw` + `real_hw_archival`). Hard layouts detected: `665` pages (`485` dense 5+ staff, `257` landscape 4-staff).
- **Artifacts:** [`eval_hf.json`](eval_hf.json), [`run_hf.json`](run_hf.json) (pulled from `gregoreesmaa/sheet-music-converter-omr`).

---

## 2. Full 3-Arm Results (`suite31_hw`, All 90 Pages = 62 Printed + 28 HW)

| Arm | Ckpt | Printed Clean | Printed Photo | **Printed All** | HW Clean | HW Photo | **HW All** | HW Land | HW Port | **All 90** | Trunc | Train Loss (tail50) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm0_v3_50_50` | RAW | 0.6540 | 0.6565 | 0.6552 | 0.1706 | 0.1099 | 0.1403 | 0.0636 | 0.1829 | 0.4950 | 10 | 0.17566 |
| `arm0_v3_50_50` | **EMA** | 0.6592 | 0.6551 | **0.6572** | 0.1760 | 0.1234 | **0.1497** | 0.0626 | 0.1980 | **0.4993** | 8 | 0.17566 |
| `arm1_hw_heavy_30_70` | RAW | 0.6940 | 0.6906 | 0.6923 | 0.2061 | 0.1334 | 0.1698 | 0.0704 | 0.2249 | 0.5297 | 10 | 0.18360 |
| `arm1_hw_heavy_30_70` | **EMA** | 0.6935 | 0.6882 | **0.6909** | 0.2082 | 0.1499 | **0.1790** | 0.0677 | 0.2409 | **0.5316** | 7 | 0.18360 |
| `arm2_rep_cap_10x_hard_upsample` | RAW | 0.6826 | 0.6362 | 0.6594 | 0.1535 | 0.1032 | 0.1283 | 0.0540 | 0.1696 | 0.4942 | 11 | 0.18662 |
| `arm2_rep_cap_10x_hard_upsample` | **EMA** | 0.6982 | 0.6387 | **0.6685** | 0.1826 | 0.1185 | **0.1505** | 0.0590 | 0.2014 | **0.5073** | 10 | 0.18662 |

### Deltas vs `arm0_v3_50_50` (Best = EMA for All 3 Arms)

| Candidate Arm | Δ `printed_all` | Δ `hw_all` | Δ `all90` | EMA − RAW (`all90`) | Gate Check (`HW >= +2.0 pp` & `PR >= -0.5 pp`) |
|---|---:|---:|---:|---:|---|
| **`arm1_hw_heavy_30_70`** | **`+3.37 pp`** | **`+2.93 pp`** | **`+3.23 pp`** | `+0.19 pp` | **PASS (PROMOTE)** |
| `arm2_rep_cap_10x_hard_upsample` | `+1.13 pp` | `+0.08 pp` | `+0.80 pp` | `+1.31 pp` | FAIL |

### Sampler Telemetry (3,200 Draws/Arm)

| Arm | Printed | Synth-HW | Real-HW Archival | Neg | Archival Unique / Max / Mean Reps | Spilled | Hard-Layout Frac |
|---|---:|---:|---:|---:|---|---:|---:|
| `arm0_v3_50_50` | `50.16%` | `34.06%` | `15.12%` | `0.66%` | `119 / 11 / 4.07` | `0` | `30.06%` |
| `arm1_hw_heavy_30_70` | `29.69%` | `45.44%` | `24.22%` | `0.66%` | `120 / 16 / 6.46` | `0` | `29.00%` |
| `arm2_rep_cap_10x_hard_upsample` | `50.16%` | `34.06%` | `15.12%` | `0.66%` | `120 / 10 / 4.03` | `0` | `45.94%` |

---

## 3. Forensic Scrutiny & Anomaly Check

1. **No RAW-vs-EMA divergence red flags:** All 3 arms have `EMA > RAW` on `all90` with spread `<= 1.31 pp` (below the `1.5 pp` warning threshold). `arm1` is the tightest (`+0.19 pp`) and fewest truncated pages (`7` vs `8` in `arm0` and `10` in `arm2`).
2. **Why `arm1_hw_heavy_30_70` improved BOTH HW (`+2.93 pp`) AND Printed (`+3.37 pp`):**
   - `arm1` lifted every single sub-bucket simultaneously: `printed_clean` (`+3.43 pp`), `printed_photo` (`+3.31 pp`), `hw_clean` (`+3.22 pp`), `hw_photo` (`+2.65 pp`), `hw_portrait` (`+4.29 pp`), `hw_landscape` (`+0.51 pp`).
   - Increasing HW exposure from `~49%` (`34.1% synth + 15.1% real`) to `~70%` (`45.4% synth + 24.2% real`) acted as a strong visual regularizer on the warm-started model, preventing printed-domain overfitting on the 1,360 proxy printed pages while directly strengthening handwritten generalization.
3. **Why `arm2_rep_cap_10x_hard_upsample` underperformed `arm1` (important methodological note):**
   - Because `iter10_5_proxy_2k` has `120` unique `real_hw_archival` pages and `arm2` only drew `484` archival samples (`mean_reps=4.03`), the `<= 10×` repetition cap barely engaged (`max_reps=10` vs `11` in `arm0`, **`spilled_draws=0`**).
   - Thus `arm2` effectively isolated **`2×` hard-layout upsampling** (`30.06% → 45.94%` hard-layout draws) under the `50/50` mix: it boosted `printed_clean` (`+3.90 pp`) but regressed `printed_photo` (`-1.64 pp`) and left `hw_all` flat (`+0.08 pp`, `hw_landscape` `-0.36 pp`).
   - Hard-layout `2×` upsampling on its own is **not** promoted. Note that at full Iter 15 scale (30k+ steps), a per-unique-page archival repetition cap should still be monitored even though it did not bind in this 800-step proxy.
4. **Missing `rism_pseudo` / `real_hw_stitched` in proxy tar:** Neither bucket was present in `iter10_5_proxy_2k.tar`, so `arm1`'s winning realized ratio is **`30% Printed / 45% Synth-HW / 24% Real-HW / 0.7% Neg`** (which maps to the planned `30/70` Printed/HW macro split when RISM-pseudo and real-stitched are present in full Iter 15).

---

## 4. Decision & Next Action

- **Verdict:** **`PROMOTE`** **`arm1_hw_heavy_30_70`** (`30% Printed / 70% HW+Neg` macro data mix: `30% Printed / 25% Synth-HW / 18% Real-HW / 20% RISM-Pseudo / 6.5% Real-Stitched / 0.5% Neg`).
- **Action for Iter 15:** Adopt the `30/70` HW-heavy macro mix in `tmp-train/mvd/train_uni_omr_v2.py` (do **not** adopt `2×` hard-layout upsampling from `arm2`).
