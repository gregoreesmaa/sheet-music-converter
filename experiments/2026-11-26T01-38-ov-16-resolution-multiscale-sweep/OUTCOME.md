# Outcome — `2026-11-26T01-38-ov-16-resolution-multiscale-sweep`

- **Status:** `COMPLETED` (`rc=0`, `3/3` arms completed, `0` NaNs)
- **Verdict:** **`DISCARD`** *(450-step from-scratch `small_45m` proxy falsified the `>= +0.5 pp` F1 / `>= -0.005` SER promotion gate and is saturated by `eval_max_len=512` truncation [`0/64` EOS] + un-warmed `0.997` EMA lag; keep current resolution defaults and do not burn cloud compute repeating a short from-scratch `small_45m` resolution sweep. Hardware/router Pareto telemetry is clean and retained below.)*
- **Completed At:** `2026-11-28T02:25:49Z` (`started_at: 2026-11-28T01:43:59Z`, total queue wall-clock `2509.53 s` including one transient Colab `503` retry; script compute elapsed `2013.69 s`)
- **Platform / GPU:** Google Colab `NVIDIA A100-SXM4-40GB` (`torch 2.14.1+cu130`, `BF16`, `mode=muon+adamw`, `seed=42`, session `ov-16` cleanly terminated on finish)
- **HF Private Hub Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T01-38-ov-16-resolution-multiscale-sweep.json` (mirrored locally at `eval_hf.json`)
  - Run Ledger: `runs/2026-11-26T01-38-ov-16-resolution-multiscale-sweep.json` (mirrored locally at `run_hf.json`)

---

## Measured Results (`proxy_val.jsonl`, `n=64` full pages, `450` steps/arm, `bs=2 × ga=4`)

### 1. Accuracy Metrics (Raw vs EMA @ step 450)

| Arm | `fullpage_max_dim` | `multi_scale` | `final_25step_loss` | Raw Token F1 | Raw SER | EMA Token F1 | EMA SER | `sys_count_exact` (Raw / EMA) | `ink_recall` (Raw / EMA) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **`1024px_fixed`** | `1024` | `False` | **`2.6971`** | **`0.1024`** | **`1.0027`** | **`0.2305`** | `0.8989` | `0.1094` / `0.1094` (`7/64`) | `0.9988` / `0.9988` |
| **`1280px_fixed`** *(control)* | `1280` | `False` | `2.7053` (`+0.0082`) | `0.0829` (`-1.95 pp`) | `1.0196` (`+0.0170`) | `0.2301` (`-0.03 pp`) | `0.8988` (`-0.0001`) | `0.1094` / `0.1094` (`7/64`) | `0.9993` / `0.9990` |
| **`1536px_multiscale`** | `1536` | `True` | `2.7146` (`+0.0175`) | `0.0749` (`-2.75 pp` vs 1024, `-0.80 pp` vs 1280) | `1.0287` (`+0.0260` vs 1024, `+0.0090` vs 1280) | `0.2291` (`-0.13 pp` vs 1024, **`-0.10 pp` vs 1280**) | **`0.8981`** (`-0.0008` vs 1024, **`-0.0007` vs 1280**) | `0.1094` / `0.1094` (`7/64`) | `0.9994` / `0.9988` |

### 2. Token Budget, Throughput & VRAM Pareto (A100-SXM4-40GB BF16, `41.36M` total / `40.51M` infer params)

| Arm | Train `n_in -> n_out` (keep) | Eval Comp (`Raw / EMA`) | Elapsed (`s`) | `sec/step` (`steps/s`) | Slowdown vs `1280px` | Peak VRAM (`MB`) | EMA Decode `tok/s` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`1024px_fixed`** | `2982.0 -> 2240.5` (`0.7513`) | `2.07× / 2.03×` | `599.39` | `1.3320` (`0.7508`) | `0.902×` (`1.108×` faster) | `6,461.2` | `307.45` |
| **`1280px_fixed`** | `4697.1 -> 3526.8` (`0.7509`) | `2.31× / 2.22×` | `664.25` | `1.4761` (`0.6775`) | `1.000×` | `9,300.4` | `304.96` |
| **`1536px_multiscale`** | `5563.8 -> 4176.9` (`0.7507`) | **`2.44× / 2.34×`** | `712.45` | `1.5832` (`0.6316`) | **`1.073×`** (`1.189×` vs 1024) | **`11,780.9`** | `297.25` (`-2.5%`) |

---

## Scrutiny & Anomaly Check

1. **Coverage & runtime hygiene:** All 3 arms completed `450/450` steps (`1,992` train pages after filtering 8 rows with `>16` systems; `64/64` held-out `proxy_val` pages; `total_ref_tokens = 52,018`; `0` NaNs; matched `mode=muon+adamw` with `114` Muon / `235` AdamW params; identical `seed=42`). Log confirms `MultiScaleBatchSampler` was genuinely active on Arm C (`n_in` varied from `3,008` to `6,912` across steps; `n_out` from `2,260` to `5,188`). Colab session `ov-16` was terminated cleanly (`exit_code=0`, verified via `colab sessions`).
2. **CRITICAL ANOMALY — 100% `eval_max_len=512` truncation (`0/64` EOS across all 6 evals):** Every single arm (both Raw and EMA) produced `total_pred_tokens == 32,768 == 64 × 512` against `total_ref_tokens = 52,018` (mean `812.8` tok/page). Zero validation pages emitted `EOS`, capping recall at `~0.21` and compressing all 3 EMA arms into a `0.13 pp` Token F1 / `0.0008` SER noise band (37–40 edits apart out of 52,018 tokens). Likewise, `sys_count_exact = 0.109375` (`7/64`) is bit-identical across all 3 arms and both Raw/EMA, confirming the system-count head has not yet separated beyond the modal class at 450 steps (`1.8` epochs).
3. **CRITICAL ANOMALY — Un-warmed `gpu_ema_decay=0.997` over 450 steps:** At `0.997^450 = 25.87%` step-0 random-init retention, EMA is heavily lag-smoothed (yet still beats noisy Raw greedy decode by `+12.8` to `+15.4 pp` F1). Meanwhile, Raw greedy F1 (`0.1024 -> 0.0829 -> 0.0749`) and `final_25step_loss` (`2.6971 -> 2.7053 -> 2.7146`) degrade slightly with higher resolution at 450 steps — the expected early-stage cross-attention convergence lag when attending over `4,177` variable-scale routed tokens vs `2,241` fixed-scale tokens, not a spatial-resolution ceiling.
4. **Protocol gap vs `PLAN.md`:** `PLAN.md` §3 specified an additional common `1280px` reference eval for cross-scale comparison, which `run_colab.py` did not implement (each arm was evaluated only at its native `fullpage_max_dim`).
5. **Clean, reusable systems/router finding:** Even though accuracy is saturated at 450 steps, the **compute, VRAM, and router compression Pareto curve is valid and informative**:
   - `1536px_multiscale` adds only **`+7.3%` step time** over `1280px_fixed` (`1.073×`, well below the `1.45×–1.60×` budget) and peaks at **`11.78 GB` VRAM** (fits comfortably under the `14 GB` local RTX 4080 SUPER WDDM ceiling and `20 GB` gate).
   - `InkTokenRouterV2` scales sub-linearly as designed: eval compression improves monotonically with resolution (`2.03×` @ 1024 → `2.22×` @ 1280 → **`2.34×–2.44×`** @ 1536) while holding **`ink_recall >= 0.9988`**, keeping EMA decode throughput within `-2.5%` (`297.25` vs `304.96 tok/s`).
   - `1024px_fixed` yields only a `1.108×` speedup over `1280px_fixed` on A100 (short of the `>= 1.4×` speedup bar for switching fast proxies to `1024px`), though it trims `-2.84 GB` VRAM (`6.46` vs `9.30 GB`).

---

## Verdict & Follow-Up Action

- **Verdict: `DISCARD`**
  - **Promotion gate (`1536px_multiscale` beats `1280px_fixed` by `>= +0.5 pp` `token_f1` or `>= -0.005` SER) FAILED:** observed deltas are `-0.10 pp` EMA F1 (`-0.80 pp` Raw F1) and `-0.0007` EMA SER (`+0.0090` Raw SER).
  - **No successor queued for this legacy Wave-4 `small_45m` proxy:** `ov-16` was designed for Iter 11 `small_45m`, whereas the active mainline is now Iter 15 (`edge85m + ax0_gl6`), and converged resolution effects (`1280px` / `1408px` / `1536px`) have already been measured on converged checkpoints (`ov-35` / `ov-35b` / Iter 13).
  - **Guardrail for any future resolution sweep on `edge85m` / `pro200m`:** Do **not** evaluate resolution from scratch at `450` steps with `eval_max_len=512` and un-warmed `0.997` EMA. Use teacher-forced `val_ce` at `>= 1,200` steps (or fine-tune from a converged checkpoint with `eval_max_len >= 2048` and warmed EMA) plus a matched common-resolution cross-scale eval.
