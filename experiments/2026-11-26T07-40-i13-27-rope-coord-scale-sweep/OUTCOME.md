# Outcome — `2026-11-26T07-40-i13-27-rope-coord-scale-sweep`

- **Status:** `COMPLETED`
- **Completed At:** `2026-11-26T19:26:57Z`
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (24 GB VRAM, BF16)
- **Total Elapsed:** `4572.09 s` (`76.2 min` across 4 arms; ~18.4–19.2 min/arm)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - `evals/2026-11-26T07-40-i13-27-rope-coord-scale-sweep.json`
  - `runs/2026-11-26T07-40-i13-27-rope-coord-scale-sweep.json`

---

## Measured Results (`proxy_val.jsonl`, 64 full pages, 400 steps/arm, `seed=42`, `dim=1280`)

| Arm | `rope_coord_scale` | Max `/16` Phase Step (`h_p=80`) | Final 25-Step Loss | Raw Token F1 | Raw SER | **EMA Token F1** | **Δ EMA F1 vs Ctrl** | **EMA SER** | EMA tok/s | Steps/s | Peak VRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scale_32` | `32.0` | `0.41 rad/patch` | `2.8215` | `0.0512` | `1.0499` | `0.2288` | `-0.10 pp` | `0.8988` | `324.2` | `0.3494` | `9558.0` |
| `scale_64` | `64.0` | `0.81 rad/patch` | `2.8238` | `0.0549` | `1.0444` | `0.2288` | `-0.10 pp` | `0.8988` | `321.3` | `0.3481` | `9564.0` |
| **`scale_128`** *(ctrl)* | **`128.0`** | `1.62 rad/patch` | **`2.8124`** | `0.0326` | `1.0644` | `0.2299` | `0.00 pp` | `0.8993` | `314.3` | `0.3519` | `9559.5` |
| `scale_256` | `256.0` | `3.24 rad/patch` | `2.8253` | `0.0213` | `1.0711` | `0.2301` | `+0.02 pp` | `0.8981` | `318.5` | `0.3633` | `9563.5` |

### Key Takeaways
1. **Null effect across an 8× coordinate-scale sweep:** Across `rope_coord_scale ∈ {32, 64, 128, 256}`, all 4 arms fall within a **`0.12 pp`** EMA Token F1 band (`0.2288`–`0.2301`) and **`0.12 pp`** EMA SER band (`0.8981`–`0.8993`).
2. **Control `scale_128` ties best and has lowest training loss:** `scale_128` achieves the lowest `final_25step_loss` (`2.8124`) and ties `scale_256` within `+0.02 pp` EMA Token F1 (`0.2299` vs `0.2301`), well below the `+0.40 pp` promotion threshold and inside the `0.20 pp` discard band.
3. **Zero parameter/VRAM delta:** All 4 arms have identical parameter counts (`41,334,795` total / `31.26 MB` Hybrid INT4+scales) and peak VRAM (`9558–9564 MB`).

---

## Scrutiny & Anomaly Check

1. **Coverage & Integrity:** All 4 arms completed 400/400 steps and scored `64/64` held-out `proxy_val` pages (`total_ref_tokens=52,018`, zero NaNs, zero missing keys). Verified via `colab.exe sessions` that the Colab L4 VM is terminated (zero idle CU burn).
2. **Early-Training Saturation & EMA Identicality:**
   - All 4 arms hit `eval_max_len=512` on every page (`total_pred_tokens == 32,768 = 64 × 512`) and had identical `sys_count_exact == 0.109375` (`7/64`).
   - `scale_32` and `scale_64` produced **bit-identical EMA eval metrics** (`token_f1=0.228833`, `ser=0.898785`, `total_edits=46,753`) despite differing raw metrics (`54,616` vs `54,329` edits) and `mean_compression_x` (`2.139` vs `2.069`). This occurs because at 400 steps with `gpu_ema_decay=0.997` (`0.997^400 = 0.301` step-0 weight), early EMA decoding is dominated by frequent structural templates rather than fine-grained 2D pitch/staff spatial alignment.
   - Note that raw Token F1 did show a slight monotonic preference for lower scales (`scale_64=0.0549`, `scale_32=0.0512` vs `scale_128=0.0326`, `scale_256=0.0213`), consistent with reduced high-frequency aliasing, but raw SER was `> 1.04` across all arms (too noisy to promote).

---

## Verdict: **`DISCARD`** *(Retain default `rope_coord_scale = 128.0`)*

- **Per `PLAN.md` Decision Tree:** `scale_128` ties all treatments within `0.02 pp` EMA Token F1 (and all 4 arms lie within `0.12 pp < 0.20 pp`), with `scale_128` achieving the lowest `final_25step_loss` (`2.8124`). Retain `UniOMRv2Config.rope_coord_scale = 128.0` for Iter 14.
- **Note for future work:** Do not re-run 400-step from-scratch RoPE scale sweeps; if spatial RoPE aliasing is ever revisited, only test it from a converged/warm-started checkpoint (`loss < 0.8`) where the decoder actively relies on sub-staff 2D positional precision.

---

## Follow-Up Action / Successor Experiment

- **None (`DISCARD`).** Retain `rope_coord_scale = 128.0`.
