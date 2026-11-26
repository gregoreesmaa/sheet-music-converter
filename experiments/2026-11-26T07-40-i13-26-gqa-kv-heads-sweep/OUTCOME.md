# Outcome — `2026-11-26T07-40-i13-26-gqa-kv-heads-sweep`

- **Status:** `COMPLETED`
- **Completed At:** `2026-11-26T19:20:54Z`
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (24 GB VRAM, BF16)
- **Total Elapsed:** `4514.33 s` (`75.2 min` across 4 arms; ~18.3–19.4 min/arm)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - `evals/2026-11-26T07-40-i13-26-gqa-kv-heads-sweep.json`
  - `runs/2026-11-26T07-40-i13-26-gqa-kv-heads-sweep.json`

---

## Measured Results (`proxy_val.jsonl`, 64 full pages, 400 steps/arm, `seed=42`, `dim=1280`)

| Arm | `n_kv_heads` | `kv_dim` | FP16 KV Cache (`2048+3072`) | Hybrid INT4+Scales | Final 25-Step Loss | Raw Token F1 | Raw SER | **EMA Token F1** | **Δ EMA F1 vs Ctrl** | **EMA SER** | EMA tok/s | Steps/s | Peak VRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `kv1_mqa` | `1` | `48` | `7.5 MB` | `30.92 MB` | `2.8624` | `0.0676` | `1.0386` | `0.1772` | `-5.27 pp` | `0.9433` | `321.3` | `0.3439` | `9519.0` |
| `kv2_gqa4x` *(ctrl)* | `2` | `96` | `15.0 MB` | `31.26 MB` | `2.8256` | `0.0474` | `1.0529` | `0.2299` | `0.00 pp` | `0.8993` | `324.2` | `0.3637` | `9566.9` |
| `kv4_gqa2x` | `4` | `192` | `30.0 MB` | `31.92 MB` | **`2.7900`** | `0.0190` | `1.0721` | `0.2245` | `-0.54 pp` | `0.9079` | `320.4` | `0.3648` | `9636.2` |
| **`kv8_mha`** | **`8`** | **`384`** | `60.0 MB` | **`33.25 MB`** | `2.8167` | `0.0454` | `1.0557` | **`0.2718`** | **`+4.19 pp`** | **`0.8626`** | **`325.3`** | `0.3573` | `9781.8` |

### Key Takeaways
1. **Severe visual bottleneck at `n_kv_heads=1` (MQA):** Dropping from `2` to `1` KV head collapses EMA Token F1 by `-5.27 pp` (`0.1772` vs `0.2299`) and worsens EMA SER by `+4.40 pp` (`0.9433` vs `0.8993`).
2. **Full MHA (`n_kv_heads=8`) wins decisively:** `kv8_mha` beats control `kv2_gqa4x` by **`+4.19 pp` EMA Token F1** (`0.2718` vs `0.2299`) and **`-3.67 pp` EMA SER** (`0.8626` vs `0.8993`), and beats `kv4_gqa2x` by **`+4.73 pp` EMA Token F1** (`0.2718` vs `0.2245`).
3. **Negligible model-size and throughput cost:** `kv8_mha` adds only `+1.99 MB` Hybrid INT4+scales (`33.25 MB` vs `31.26 MB`, well under `< 55 MB`), `+215 MB` peak VRAM (`+2.2%`), and matches decode throughput (`325.3` vs `324.2 tok/s`, `+0.3%`). Its sole trade-off is a `4×` larger FP16 KV cache (`60.0 MB` vs `15.0 MB` at `L_self=2048, L_cross=3072`).

---

## Scrutiny & Anomaly Check

1. **Coverage & Integrity:** All 4 arms completed 400/400 steps and scored `64/64` held-out `proxy_val` pages (`total_ref_tokens=52,018`, zero NaNs, zero missing keys). Verified via `colab.exe sessions` that the Colab L4 VM is terminated (zero idle CU burn).
2. **Early-Training `eval_max_len=512` Saturation Caveat:** Across all 4 arms, `total_pred_tokens == 32,768` (`64 × 512` — 100% of pages hit the `512`-token cap without emitting EOS) and `sys_count_exact == 0.109375` (`7/64`) is identical across arms. At 400 steps from scratch (`final_25step_loss ~ 2.79–2.86`), raw weights have very low Token F1 (`0.0190–0.0676`, SER `> 1.03`), while EMA (`decay=0.997`) stabilizes predictions (`0.1772–0.2718`).
3. **Non-Monotonic `kv4_gqa2x` Loss vs. EMA F1:** `kv4_gqa2x` achieved the lowest `final_25step_loss` (`2.7900`) yet trailed `kv2_gqa4x` slightly on EMA F1 (`0.2245` vs `0.2299`), whereas `kv8_mha` had `2.8167` loss yet `+4.19 pp` higher EMA F1. Because `0.997^400 = 0.301` retains 30% step-0 random weight in EMA, the follow-up MUST use `800` steps with `gpu_ema_decay=0.99` (`0.99^800 = 0.0003`) to confirm the gain past the early-training regime.

---

## Verdict: **`ITERATE`** *(Provisional Iter 14 Winner: `n_kv_heads=8` MHA)*

- **Why `ITERATE` (per `PLAN.md` Decision Tree):** `kv8_mha` beats `kv2_gqa4x` by `+4.19 pp` (passing the `>= +0.5 pp` promotion bar at `33.25 MB` Hybrid INT4 and `325.3 tok/s`), **AND** beats `kv4_gqa2x` by `+4.73 pp >= +0.5 pp`, which directly triggers the `PLAN.md` **`ITERATE`** branch:
  > *"If `kv8_mha` beats `kv4_gqa2x` by `>= +0.5 pp`, test asymmetric GQA (`n_kv_heads_self=2`, `n_kv_heads_cross=8`) to keep self-attention KV cache tiny while unbottlenecking visual cross-attention."*
- **Iter 14 Default Guidance:** If Iter 14 launches before `i13-26b` completes, `n_kv_heads=8` (full MHA) is the provisional winning hyperparameter (`+4.19 pp` EMA Token F1, `33.25 MB` Hybrid INT4). If `i13-26b` confirms `asym_s2_c8` matches `kv8_mha` within `0.3 pp`, promote `n_kv_heads_self=2, n_kv_heads_cross=8` to capture the full visual cross-attention gain with `4×` smaller autoregressive self-KV bandwidth (`39.75 MB` total vs `60.0 MB`, and only `3.75 MB` self-KV).

---

## Follow-Up Action / Successor Experiment

- **Queued Successor:** [`2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha`](../2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha/) (registered in `experiments/overnight_queue/queue.json` with `depends_on: ["2026-11-26T07-40-i13-26-gqa-kv-heads-sweep"]`).
  - Tests `kv2_sym` (`2/2`) vs `asym_s2_c8` (`self=2, cross=8`) vs `kv8_sym` (`8/8`) for `800` steps/arm with `gpu_ema_decay=0.99` on Colab L4.
