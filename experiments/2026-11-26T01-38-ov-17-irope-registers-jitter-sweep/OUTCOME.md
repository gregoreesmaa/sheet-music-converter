# Outcome — `2026-11-26T01-38-ov-17-irope-registers-jitter-sweep`

- **Status:** `COMPLETED` (`rc=0`, `6/6` arms completed, `0` NaNs)
- **Verdict:** **`PROMOTE`** *(`n_register_tokens=8`, keep `rope_jitter=0.08` — `reg8_jit008` beats `reg0_jit008` by **`+4.05 pp`** EMA Token F1 [`-0.0371` SER] and current default `reg4_jit008` by **`+0.50 pp`** EMA Token F1 [`-0.0017` SER], clearing the `>= +0.4 pp` promotion gate with **zero speed or VRAM overhead** and only `+1,536` params [`+0.003 MB`]. Wired `n_register_tokens=8` into `UniOMRv2Config.iter15_pro200m()` and exposed `--n-register-tokens` / `--rope-jitter` CLI overrides in `train_uni_v2.py` while keeping base `UniOMRv2Config.n_register_tokens=4` for backward compatibility with existing checkpoints.)*
- **Completed At:** `2026-11-28T06:20:23Z` (`started_at: 2026-11-28T04:40:16Z`, total queue wall-clock `6006.72 s` including ~50.5 min Colab `TooManyAssignmentsError` retry wait; script compute `05:30:50Z -> 06:20:17Z`, elapsed `2966.99 s` = `49.45 min`)
- **Platform / GPU:** Google Colab `NVIDIA A100-SXM4-40GB` (`torch 2.14.1+cu130`, `BF16`, `mode=muon+adamw`, `seed=42`, session `ov-17` cleanly terminated on finish)
- **HF Private Hub Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T01-38-ov-17-irope-registers-jitter-sweep.json` (mirrored locally at `eval_hf.json`)
  - Run Ledger: `runs/2026-11-26T01-38-ov-17-irope-registers-jitter-sweep.json` (mirrored locally at `run_hf.json`)

---

## Measured Results (`proxy_val.jsonl`, `n=64` full pages, `300` steps/arm, `1280px`, `bs=2 × ga=4`)

### 1. Sub-Sweep A — RoPE-Exempt Register Tokens (`rope_jitter=0.08`, `irope_nope_every=4`)

| Arm | `n_reg` | `final_25step_loss` | Raw Token F1 | Raw SER | EMA Token F1 | EMA SER | Δ EMA F1 vs `reg0` | Δ EMA F1 vs `reg4` | `sys_exact` (Raw / EMA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reg0_jit008` | `0` | `2.9469` | `0.0244` | `1.0704` | `0.2330` | `0.9001` | `0.00 pp` | `-3.55 pp` | `0.1094` / `0.1094` |
| `reg4_jit008` *(control)* | `4` | `2.9348` | `0.0289` | `1.0667` | `0.2685` | `0.8647` | `+3.55 pp` | `0.00 pp` | `0.1094` / `0.1094` |
| **`reg8_jit008`** *(winner)* | **`8`** | `2.9425` | **`0.0325`** | `1.0650` | **`0.2735`** | **`0.8629`** | **`+4.05 pp`** | **`+0.50 pp`** | `0.1094` / `0.1094` |
| `reg16_jit008` | `16` | **`2.9295`** | `0.0321` | **`1.0591`** | `0.1467` | `0.9762` | `-8.63 pp` | `-12.19 pp` | `0.1094` / `0.1094` |

### 2. Sub-Sweep B — V2PE RoPE Box-Jitter (`n_register_tokens=4`, `irope_nope_every=4`)

| Arm | `rope_jitter` | `final_25step_loss` | Raw Token F1 | Raw SER | EMA Token F1 | EMA SER | Δ EMA F1 vs `jit008` | `ink_recall` (Raw / EMA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `reg4_jit000` | `0.00` | `2.9371` | `0.1633` | `0.9512` | `0.2675` | `0.8643` | `-0.10 pp` | `0.9995` / `0.9982` |
| **`reg4_jit008`** *(control)* | **`0.08`** | `2.9348` | `0.0289` | `1.0667` | **`0.2685`** | `0.8647` | `0.00 pp` | `0.9992` / `0.9984` |
| `reg4_jit016` | `0.16` | **`2.9267`** | **`0.2291`** | **`0.8981`** | `0.2684` | **`0.8638`** | `-0.01 pp` | `0.9990` / `0.9978` |

### 3. Throughput, VRAM & Footprint (`small_45m`, A100-SXM4-40GB BF16)

| Arm | Elapsed (`s`) | `steps/s` | Peak VRAM (`MB`) | Raw / EMA `tok/s` | Eval Comp (`Raw / EMA`) | Total Params | Hybrid INT4+Scales (`MB`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `reg0_jit008` | `462.68` | `0.6484` | `9,298.4` | `352.91` / `291.63` | `2.26×` / `2.14×` | `41,354,763` | `31.2797` |
| `reg4_jit008` | `492.71` | `0.6089` | `9,301.0` | `239.58` / `300.29` | `2.30×` / `2.11×` | `41,356,299` | `31.2828` |
| **`reg8_jit008`** | **`461.40`** | **`0.6502`** | `9,300.9` | `312.48` / `299.44` | `2.29×` / `2.11×` | `41,357,835` | `31.2858` |
| `reg16_jit008` | `594.02` | `0.5050` | `9,304.7` | `313.35` / `135.08` | `2.29×` / `2.15×` | `41,360,907` | `31.2920` |
| `reg4_jit000` | `462.79` | `0.6482` | `9,337.3` | `298.91` / `304.20` | `2.24×` / `2.13×` | `41,356,299` | `31.2828` |
| `reg4_jit016` | `460.19` | `0.6519` | `9,298.5` | `310.03` / `302.67` | `2.29×` / `2.10×` | `41,356,299` | `31.2828` |

---

## Scrutiny & Anomaly Check

1. **Coverage & runtime hygiene:** All 6 arms completed `300/300` steps (`1,992` train pages, `64/64` held-out `proxy_val` pages, `total_ref_tokens = 52,018`, `0` NaNs, matched `mode=muon+adamw` [`114` Muon / `235` AdamW params, `234` on `reg0`], identical `seed=42`). Colab session `ov-17` was cleanly terminated (`colab sessions` confirmed zero idle `ov-17` VM).
2. **`eval_max_len=512` truncation & un-warmed `0.997` EMA caveat:** Like `ov-16`, all 12 evals (6 Raw + 6 EMA) produced `total_pred_tokens == 32,768` (`64 × 512`, `0/64` EOS) and bit-identical `sys_count_exact = 0.109375` (`7/64`) under un-warmed `gpu_ema_decay=0.997` (`0.997^300 = 40.6%` step-0 retention). However, unlike `ov-16` (where all EMA arms sat inside a `0.13 pp` noise band), the register sweep separated cleanly and monotonically across `0 -> 4 -> 8` on **both** EMA (`0.2330 -> 0.2685 -> 0.2735`, `+4.05 pp` / `+0.50 pp`) and Raw (`0.0244 -> 0.0289 -> 0.0325`) before collapsing at `16` (`0.1467` EMA F1, `-12.68 pp` vs `reg8`).
3. **Raw greedy threshold sensitivity on `rope_jitter` arms:** `reg4_jit000` (`0.1633`) and `reg4_jit016` (`0.2291`) escaped early-step Raw greedy repetition collapse while `reg4_jit008` (`0.0289`) did not, even though `final_25step_loss` (`2.9267–2.9371`) and EMA Token F1 (`0.2675` / `0.2685` / `0.2684`, `0.10 pp` spread) were essentially identical. EMA smooths out this 300-step greedy bifurcation; `rope_jitter=0.08` retains a slight `+0.10 pp` EMA edge over `0.00` and matches `0.16` (`-0.01 pp`), so keeping `rope_jitter=0.08` is the conservative choice.
4. **`reg16_jit008` collapse & decode slowdown:** At `n_register_tokens=16`, EMA F1 drops `-12.19 pp` vs `reg4` (`0.1467` vs `0.2685`) and EMA decode throughput falls to `135.08 tok/s` (`594.02 s` arm time vs `~461 s`), confirming `16` registers overshoots the sweet spot.

---

## Verdict & Follow-Up Action

- **Verdict: `PROMOTE` (`n_register_tokens=8`, keep `rope_jitter=0.08`)**
  - **Gate check (`PLAN.md` §6):** `reg8_jit008` beats `reg0_jit008` by **`+4.05 pp`** EMA Token F1 and current default `reg4_jit008` by **`+0.50 pp`** EMA Token F1 (clearing `>= +0.4 pp`) with **zero speed cost** (`0.6502` vs `0.6089` / `0.6484 steps/s`) and **zero VRAM cost** (`9300.9` vs `9301.0 MB`). `jit016` did not beat `jit008` on EMA F1 (`-0.01 pp`), so the `ITERATE` branch is not triggered.
  - **Immediate implementation landed:**
    1. Set `n_register_tokens=8` in `UniOMRv2Config.iter15_pro200m()` (`tmp-train/mvd/uni_omr_v2.py`) — adds only `4 × 768 = 3,072` params (`+0.006 MB` FP16, keeping `iter15_pro200m` well under the `135 MB` Hybrid INT4 budget) while preserving base `UniOMRv2Config.n_register_tokens=4` for 100% backward compatibility with existing `iter12`–`iter14` checkpoints.
    2. Exposed `--n-register-tokens` and `--rope-jitter` CLI overrides (`default=None`) in `tmp-train/mvd/train_uni_v2.py` so any preset (`small_45m`, `edge85m`, `pro200m`, `iter15_pro200m`) can override them directly without monkey-patching.
    3. Verified on CPU (`CUDA_VISIBLE_DEVICES=""`) via `test_iter15_arch.py` and `test_uni_omr_v2.py`.
