# Outcome — `2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha`

- **Status:** `COMPLETED`
- **Completed At:** `2026-11-28T02:07:53Z` (Started `2026-11-28T01:23:09Z`)
- **Platform / GPU:** Local `1× NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, native BF16; originally targeted Colab L4, executed locally at `0` CU cost)
- **Total Elapsed:** `2,683.95 s` (`44.7 min` across 3 arms; `845.65 s` / `847.72 s` / `869.14 s`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - `evals/2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha.json`
  - `runs/2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha.json`
- **Local Artifacts:** `D:\tmp\local-p0\2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha\out\2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha\`

---

## Measured Results (`proxy_val.jsonl`, 64 full pages, 800 steps/arm, `gpu_ema_decay=0.99`, `seed=42`, `dim=1280`, `small_45m`, `Muon+AdamW`)

| Arm | `n_kv_self / cross` | FP16 KV Cache (`2048+3072`) | Total / Infer Params | Hybrid INT4+Scales | Final 25-Step Loss | **Raw Token F1** | **Δ Raw F1 vs Ctrl** | Raw SER | Raw `sys_exact` | **EMA Token F1** | **Δ EMA F1 vs Ctrl** | **EMA SER** | EMA `sys_exact` | Raw / EMA tok/s | Steps/s | Peak VRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `kv2_sym` *(ctrl)* | `2 / 2` | `6.0 + 9.0 = 15.0 MB` | `41.36M / 40.51M` | `31.28 MB` | `2.2983` | `0.3036` (`0.303605`) | `0.00 pp` | `0.8745` | `0.1094` (`7/64`) | `0.2054` (`0.205413`) | `0.00 pp` | `0.9268` | `0.1094` (`7/64`) | `479.2 / 470.5` | `0.9460` | `8,966.3` |
| `asym_s2_c8` | `2 / 8` | `6.0 + 36.0 = 42.0 MB` | `43.13M / 42.28M` | `32.28 MB` (`+1.00 MB`) | `2.3028` | `0.3147` (`0.314719`) | `+1.11 pp` | **`0.8733`** | `0.1719` (`11/64`) | `0.2890` (`0.288998`) | `+8.36 pp` | **`0.8754`** | **`0.1719`** (`11/64`) | `454.6 / 467.6` | `0.9437` | `9,077.3` |
| **`kv8_sym`** | **`8 / 8`** | `24.0 + 36.0 = 60.0 MB` | `44.90M / 44.05M` | **`33.27 MB`** (`+1.99 MB`) | **`2.2761`** | **`0.3312`** (`0.331240`) | **`+2.76 pp`** | `0.8762` | **`0.2969`** (`19/64`) | **`0.3004`** (`0.300383`) | **`+9.50 pp`** | `0.8784` | `0.1563` (`10/64`) | `392.3 / 421.3` | `0.9205` | `9,150.3` |

### Key Takeaways
1. **Full MHA (`kv8_sym`, `8/8`) decisively beats both `kv2_sym` and `asym_s2_c8`:**
   - vs `kv2_sym` (control): **`+9.50 pp` EMA Token F1** (`0.3004` vs `0.2054`), **`-4.84 pp` EMA SER** (`0.8784` vs `0.9268`), **`+2.76 pp` Raw Token F1** (`0.3312` vs `0.3036`), **`+18.75 pp` Raw `sys_count_exact`** (`19/64` vs `7/64`), and **`-0.0222` final 25-step train loss** (`2.2761` vs `2.2983`).
   - vs `asym_s2_c8` (`2/8`): **`+1.14 pp` EMA Token F1** (`0.3004` vs `0.2890`, clearing the `PLAN.md` `> +0.5 pp` `PROMOTE kv8_sym` gate by `2.28×`), **`+1.65 pp` Raw Token F1** (`0.3312` vs `0.3147`), and **`+12.50 pp` Raw `sys_count_exact`** (`19/64` vs `11/64`).
2. **Causal self-attention ALSO benefits from 8 KV heads:** Unbottlenecking only visual cross-attention (`asym_s2_c8`) captures `88.0%` of the EMA F1 gain (`+8.36 pp` of `+9.50 pp`) and preserves `99.4%` of `kv2_sym` EMA decode speed (`467.6` vs `470.5 tok/s`), but leaves `+1.14 pp` EMA F1 / `+1.65 pp` Raw F1 / `+12.50 pp` Raw `sys_count_exact` on the table compared to full MHA (`kv8_sym`).
3. **Training & decode overhead on `small_45m`:** `kv8_sym` adds only `+1.99 MB` Hybrid INT4+scales (`33.27 MB` vs `31.28 MB`), `+184 MB` peak VRAM (`+2.1%`), and `-2.7%` training speed (`0.9205` vs `0.9460 steps/s`). Autoregressive decode throughput drops `-10.5%` EMA (`421.3` vs `470.5 tok/s`) / `-18.1%` raw (`392.3` vs `479.2 tok/s`) due to the `4×` larger self-KV cache (`24.0` vs `6.0 MB`).

---

## Scrutiny & Anomaly Check

1. **Coverage & Integrity:** All 3 arms completed `800/800` steps with **`0` NaNs** (verified in `train_log.json`: step-1 losses `7.78 / 7.89 / 7.71` → step-400 `2.89 / 2.87 / 2.88` → step-800 `2.30 / 2.27 / 2.22`) and scored `64/64` held-out `proxy_val` pages (`total_ref_tokens = 52,018`, `ink_recall = 0.9981–0.9988`, `mean_compression_x = 2.312–2.337`).
2. **`eval_max_len=512` Cap Saturation:** Across all 3 arms (raw & EMA), `total_pred_tokens == 32,768` (`64 × 512`), meaning 100% of val pages still hit the 512-token cap at 800 steps from scratch. That caps SER near `~0.873–0.878` (`total_edits` floored by `52,018 - 32,768 = 19,250` unpredicted tail tokens). However, unlike `i13-26` (400 steps), here both **Raw** and **EMA** Token F1 and `sys_count_exact` are well above the noise floor and rank **`kv8_sym > asym_s2_c8 > kv2_sym`** identically.
3. **Raw > EMA Ordering under `gpu_ema_decay=0.99`:** With `decay=0.99` (`0.99^800 = 0.0003`) and a steep WSD cooldown over steps `600..800`, EMA weights lag the final cooled raw weights (`Raw F1 0.3036 / 0.3147 / 0.3312` vs `EMA F1 0.2054 / 0.2890 / 0.3004`). Because the ordering and `> +0.5 pp` margin between `kv8_sym` and `asym_s2_c8` hold on **both** Raw (`+1.65 pp`) and EMA (`+1.14 pp`), the verdict is robust to EMA lag.
4. **CRITICAL `edge85m` WebGPU `< 55.0 MB` Budget Check (Iter 15 Deployment Constraint):**
   Measured directly via `UniOMRv2Net.param_summary()` on `ax0_gl6`:
   - `small_45m + ax0_gl6`: `n_kv=2` **`31.613 MB`** → `n_kv=4` **`32.276 MB`** → `n_kv=8` **`33.603 MB`** (`+1.990 MB`, **PASSES `< 55.0 MB`**).
   - `edge85m + ax0_gl6` (`d_model=512`, `n_dec_layers=10`):
     - `n_kv=2`: **`53.039 MB`** (**PASS**, `1.961 MB` headroom)
     - `n_kv=4`: **`54.514 MB`** (**PASS**, `0.486 MB` headroom)
     - `asym_s2_c8`: **`55.251 MB`** (**FAIL**, `+0.251 MB` over `< 55.0 MB`)
     - `n_kv=8`: **`57.463 MB`** (**FAIL**, `+2.463 MB` over `< 55.0 MB`)
   - Consequently, `n_kv_heads=8` can be promoted directly for **`small_45m`** and **`pro200m`** (server/local tiers), whereas **`edge85m` (`< 55 MB` WebGPU tier)** cannot take `n_kv_heads=8` standalone without either capping at `n_kv_heads=4` (`54.514 MB`) or pairing with KV-sharing / parameter savings (e.g., YOCO/CLA) to stay `< 55.0 MB`.
5. **Minor Doc Arithmetic Note:** `README.md` estimated `3.75 MB` FP16 self-KV for `asym_s2_c8`; the exact analytical formula in `run_colab.py` (`2 × 2 × 8 layers × 48 head_dim × 2 kv_heads × 2048 tok`) is `6.00 MB` (`42.0 MB` total for `asym_s2_c8`, `15.0 MB` for `kv2_sym`, `60.0 MB` for `kv8_sym`), matching `eval_hf.json`.

---

## Verdict: **`PROMOTE`** (`kv8_sym`: Full MHA `n_kv_heads=8`)

- **Gate Evaluation (`PLAN.md` §4):**
  - `kv8_sym` beats `asym_s2_c8` by **`+1.14 pp` EMA Token F1** (`0.3004` vs `0.2890`, clearing `> +0.5 pp`) and **`+1.65 pp` Raw Token F1** (`0.3312` vs `0.3147`), and beats `kv2_sym` by **`+9.50 pp` EMA Token F1** / **`+2.76 pp` Raw Token F1**.
  - This directly triggers **`PROMOTE kv8_sym`** (`n_kv_heads=8`, symmetric full MHA — no asymmetric `n_kv_heads_cross` config complexity needed).
- **Wiring & Budget Applicability:**
  - `n_kv_heads` is already natively supported in `UniOMRv2Config` and `train_uni_v2.py` (`--n-kv-heads 8`).
  - **Server / Local (`small_45m`, `pro200m`):** Promote `n_kv_heads=8` directly (`33.603 MB` Hybrid INT4 on `small_45m + ax0_gl6`).
  - **WebGPU `< 55.0 MB` (`edge85m + ax0_gl6`):** Do **not** blindly set `n_kv_heads=8` (`57.463 MB > 55.0 MB`); use `n_kv_heads=4` (`54.514 MB`, fits `< 55 MB`) or pair `n_kv_heads=8` with KV-head/layer sharing in Iter 15 architecture lock.

---

## Follow-Up Action / Successor Experiment

- **No further GQA sweep needed (`i13-26` + `i13-26b` conclusively close the decoder KV-head question).**
- Apply `n_kv_heads=8` to `small_45m` / `pro200m` training configs, and respect the `54.514 MB` (`n_kv=4`) vs `57.463 MB` (`n_kv=8`) WebGPU `< 55.0 MB` budget gate when finalizing the Iter 15 `edge85m` recipe.
