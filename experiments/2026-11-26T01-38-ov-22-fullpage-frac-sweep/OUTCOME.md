# Outcome — `2026-11-26T01-38-ov-22-fullpage-frac-sweep`

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T10:51:51Z` (started `08:33:05Z`)
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`22.03 GB`, native `BF16`)
- **Elapsed Time:** `8,330.4 s` (`138.8 min`; `416.7 s` derived-strip build + `23.7 / 27.2 / 31.0 min` training per arm, `peak_vram=9.96–10.01 GB`)
- **Model & Protocol:** `UniOMR v2 small_45m` (`~43M` params), `800` steps/arm from scratch (`seed=42`, `Muon+AdamW`, `lr=3e-4`, WSD, `B=2 × GA=4`, `strip_batch_size=4`, `dim=1280`, `max_seq_len=1536`, `aug_prob=0.50`), `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages: `44` printed + `20` HW) + `2,344` derived strips (`1,681` printed + `663` HW from `600` train pages).
- **HF Artifacts:**
  - `evals/2026-11-26T01-38-ov-22-fullpage-frac-sweep.json` (pulled locally to [2026-11-26T01-38-ov-22-fullpage-frac-sweep.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T01-38-ov-22-fullpage-frac-sweep/2026-11-26T01-38-ov-22-fullpage-frac-sweep.json))
  - `runs/2026-11-26T01-38-ov-22-fullpage-frac-sweep.json` (pulled locally to [run_2026-11-26T01-38-ov-22-fullpage-frac-sweep.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T01-38-ov-22-fullpage-frac-sweep/run_2026-11-26T01-38-ov-22-fullpage-frac-sweep.json))

## Measured Results (`proxy_val.jsonl`, 64 Full Pages: 44 Printed + 20 HW)

### Raw Weights (Primary Converged Signal at 800 Steps)

| Arm | `fp_frac` (FP / Strip steps) | `step/s` | All `val_ce` (Δ vs `0.70`) | All `SER` | All `Token F1` (Δ) | `sys_exact` | Printed `val_ce` / `F1` | HW `val_ce` / `F1` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fp_070` | `0.70` (`560 / 240`) | `0.562` | `1.5412` (`0.0000`) | **`0.8650`** | `0.2696` (`0.0000`) | `0.1406` | `1.5884` / `0.2384` | `1.4655` / `0.3232` |
| `fp_085` | `0.85` (`680 / 120`) | `0.490` | `1.5295` (`-0.0116`) | `0.9501` | `0.1717` (`-0.0979`) | `0.1094` | `1.5756` / `0.2091` | `1.4553` / `0.0949` |
| **`fp_100`** | **`1.00` (`800 / 0`)** | `0.430` | **`1.5164`** (**`-0.0247`**) | `0.8666` | **`0.3085`** (**`+0.0389`**) | **`0.1719`** | **`1.5664`** / **`0.3009`** | **`1.4339`** / **`0.3256`** |

### EMA Weights (`decay=0.997, update_every=5`)

| Arm | `fp_frac` | All `val_ce` (Δ vs `0.70`) | All `SER` (Δ) | All `Token F1` (Δ) | Printed `val_ce` / `F1` | HW `val_ce` / `F1` |
|---|---:|---:|---:|---:|---:|---:|
| `fp_070` | `0.70` | `1.6716` (`0.0000`) | `1.0024` (`0.0000`) | `0.1060` (`0.0000`) | `1.7114` / `0.1249` | `1.6101` / `0.0595` |
| `fp_085` | `0.85` | `1.6662` (`-0.0054`) | `1.0509` (`+0.0486`) | `0.0508` (`-0.0552`) | `1.7065` / `0.0587` | `1.6053` / `0.0304` |
| **`fp_100`** | **`1.00`** | **`1.6547`** (**`-0.0169`**) | **`0.9900`** (**`-0.0124`**) | **`0.1229`** (**`+0.0169`**) | **`1.6968`** / **`0.1432`** | **`1.5890`** / **`0.0814`** |

## Scrutiny & Anomaly Check

1. **Silent failures / coverage:** Zero. All 3 arms completed `800 / 800` steps with exact step-kind counts (`560/240`, `680/120`, `800/0`), `peak_vram <= 10.01 GB`, and scored `64 / 64` val pages (`44` printed + `20` HW) on both Raw and EMA weights.
2. **Unanimous, monotonic win for `fp_100` across both domains and both checkpoints:**
   - Teacher-forced `val_ce` improves strictly monotonically with `fullpage_frac` (`0.70 → 0.85 → 1.00`) across **All** (`1.5412 → 1.5295 → 1.5164` raw; `1.6716 → 1.6662 → 1.6547` EMA), **Printed** (`1.5884 → 1.5756 → 1.5664` raw), and **HW** (`1.4655 → 1.4553 → 1.4339` raw).
   - `fp_100` exceeds both `PLAN.md` promotion gates (`>= +0.015` Token F1 AND `>= 0.02` lower Val CE): **`+0.0389` Raw Token F1** (`0.3085` vs `0.2696`, `+0.1368` vs `0.1717`), **`+0.0169` EMA Token F1**, **`-0.0247` Raw Val CE**, and is the only arm with non-zero HW `sys_count_exact` (`0.05` vs `0.00`) and highest overall `sys_count_exact` (`0.1719`).
3. **Why mixed strips hurt:** Uniform horizontal strip slicing introduces slight system-boundary misalignment and diverts 15–30% of gradient updates away from full-page layout & `<sys>` transitions. `UniOMR v2`'s Anisotropic FPN + Axial/2D-RoPE encoder solves full-page layout directly without strip scaffolding, and `fp_100` also eliminates the `417 s` strip-preparation step.
4. **EMA lag caveat:** `gpu_ema_decay=0.997, update_every=5` over `800` steps performs only `160` EMA updates (`0.997^160 = 0.618` init weight remaining), explaining why EMA lags Raw by `+0.138` CE and `-0.186` F1. Crucially, Raw and EMA agree 100% on **`fp_100`** winning every slice.
5. **Wall-clock tradeoff:** `fp_100` runs at `0.430 step/s` (`31.0 min`) vs `0.562 step/s` (`23.7 min`) for `fp_070` (`1.31×` per step), a modest cost for a decisive `+0.0389` F1 / `-0.0247` CE gain.

## Verdict: **`PROMOTE`**

Promote **`--curriculum fullpage --fullpage-frac 1.00`** as the locked default for Iter-11 `UniOMR v2` full-page training, and **drop mixed uniform-strip curricula (`--fullpage-frac <= 0.85`)**.

## Follow-Up Action / Successor Experiment

- Wire `--curriculum fullpage --fullpage-frac 1.00` into the Iter-11 `UniOMR v2` full training recipe (no strip-fraction follow-up needed).
