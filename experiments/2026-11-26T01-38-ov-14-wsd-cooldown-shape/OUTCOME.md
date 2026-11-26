# Outcome — `2026-11-26T01-38-ov-14-wsd-cooldown-shape`

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T05:58:29Z`
- **Platform / GPU:** Google Colab `NVIDIA L4` (`cc=8.9`, BF16, `torch 2.14.1+cu130`, peak VRAM `9.94 GB`)
- **Elapsed Time:** `10,338.8 s` (`172.3 min`, `~1,192–1,206 s/arm`, `~6.7 CU`)
- **Baseline Commit SHA:** `164852f`
- **HF Artifacts:**
  - `evals/2026-11-26T01-38-ov-14-wsd-cooldown-shape.json` (pulled locally to [2026-11-26T01-38-ov-14-wsd-cooldown-shape.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T01-38-ov-14-wsd-cooldown-shape/2026-11-26T01-38-ov-14-wsd-cooldown-shape.json))
  - `runs/2026-11-26T01-38-ov-14-wsd-cooldown-shape.json`

## Measured Results (`2 × 3` Factorial, `500` steps/arm, `seed=42`, `lr=3.0e-4`, `muon+adamw`, `proxy_val n=64`)

| Arm | `wsd_decay_frac` | Curve | Last-100 Train CE | RAW `val_ce` | RAW `val_ser` | RAW `val_f1` | EMA `val_ce` | EMA `val_ser` | EMA `val_f1` |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `frac015_cosine` | `0.15` | `cosine` | `1.8093` | `1.6630` | `1.0182` | `0.1084` | `1.9752` | `0.8980` | `0.2255` |
| `frac030_cosine` | `0.30` | `cosine` | `1.8192` | `1.6866` | `0.9211` | `0.2530` | `1.9836` | `0.8979` | `0.2251` |
| **`frac015_linear`** | **`0.15`** | **`linear`** | **`1.8056`** | **`1.6603`** | `0.9353` | `0.2147` | **`1.9729`** | **`0.8978`** | `0.2261` |
| `frac030_linear` | `0.30` | `linear` | `1.8214` | `1.6865` | **`0.9097`** | **`0.2663`** | `1.9816` | `0.8979` | `0.2251` |
| `frac015_sqrt` | `0.15` | `1_minus_sqrt` | `1.8092` | `1.6700` | `0.9482` | `0.1921` | `1.9757` | `0.8987` | **`0.2263`** |
| `frac030_sqrt` | `0.30` | `1_minus_sqrt` | `1.8293` | `1.7051` | `1.0368` | `0.0637` | `1.9860` | `0.8981` | `0.2241` |

**Marginal Means (`EMA val_ce`):**
- By `wsd_decay_frac`: `0.15` = **`1.9746`** vs `0.30` = `1.9838` (`Δ = -0.0091`; on RAW `val_ce`: `1.6644` vs `1.6927`, `Δ = -0.0283`).
- By curve: `linear` = **`1.9773`**, `cosine` = `1.9794` (`Δ = -0.0022`), `1_minus_sqrt` = `1.9809` (`Δ = +0.0014`).

## Scrutiny & Anomaly Check

1. **Silent failures / coverage:** Zero. All 6 arms completed `500/500` steps, `n_val_samples = 64`, `val_sys_count_exact = 0.109` across all arms, zero NaNs, identical peak VRAM (`9.942 GB`).
2. **EMA lag confound (consistent with `ov-15`):** `EMA val_ce` (`1.973–1.986`) lags `RAW val_ce` (`1.660–1.705`) by `+0.31` CE across all arms because fixed `ema_decay=0.999` has a 1,000-step time constant on a 500-step run, compressing EMA differences into `0.013` CE / `0.002` F1. However, **RAW `val_ce` confirms the exact same ranking**: `frac015_linear` (`1.6603`) ≈ `frac015_cosine` (`1.6630`, `Δ = -0.0027`), while `1_minus_sqrt` is worst (`1.6700` / `1.7051`, with RAW greedy F1 collapsing to `0.0637` at `0.30`).
3. **Cooldown window effect:** `wsd_decay_frac=0.15` beats `0.30` on **all three curves** on both RAW (`-0.0283` CE) and EMA (`-0.0091` CE) because holding peak LR for `75` additional stable steps matters more than extending the decay tail at 500 steps.
4. **Falsification gate check:** README promote gate required `EMA val_ce <= -0.02` AND `EMA F1 >= +0.005` over `cosine`. Observed best delta (`frac015_linear` vs `frac015_cosine`) is `-0.0023` EMA CE / `+0.0006` EMA F1 (`-0.0027` RAW CE), squarely inside the null/noise band (`|Δ CE| < 0.015`, `|Δ F1| < 0.008`).

## Verdict: **`DISCARD`** (Keep `cosine` curve; prefer `wsd_decay_frac = 0.15–0.20` over `0.30`)

WSD cooldown curve shape (`linear` vs `cosine` vs `1 - sqrt(t)`) is a negligible (`< 0.003` CE) second-order effect on `UniOMR v2 small_45m`, whereas `1_minus_sqrt` hurts stability. Do not add extra decay-curve complexity to `uni_omr.py` / `train_uni_v2.py`; keep `cosine` with `wsd_decay_frac` in `0.15–0.20`.

## Follow-Up Action / Successor Experiment

- None required (hypothesis cleanly falsified). EMA lag is already addressed by queued successor `2026-11-26T05-33-ov-15b-ema-warmup-late-cooldown-soup`.
