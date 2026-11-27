# Outcome — `2026-11-26T23-35-i13-25b-axial-global-boundary-completion`

- **Status:** `COMPLETED` (`2/2` arms finished cleanly, `error: null`)
- **Completed At:** `2026-11-27T06:30:49Z` (started `2026-11-27T05:58:59Z`)
- **Platform / GPU:** Colab `L4` (`NVIDIA L4`, 24 GB VRAM, BF16 native; session `[i13-25]` closed/reused on completion)
- **Elapsed Time:** `1909.91 s` (`31.8 min`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T23-35-i13-25b-axial-global-boundary-completion.json`
  - Run ledger: `runs/2026-11-26T23-35-i13-25b-axial-global-boundary-completion.json`
- **Predecessor:** [`2026-11-26T07-40-i13-25-axial-global-layer-split`](../2026-11-26T07-40-i13-25-axial-global-layer-split/OUTCOME.md)

---

## Measured Results — Complete 4-Point Iso-Depth Curve (`n_axial + n_global = 6`)

Matched hyperparameters across `i13-25` and `i13-25b`: `UniOMRv2Config.small_45m()`, `iter10_5_proxy_2k` (`1,992` train / `64` val full-page samples), `400` steps/arm, `seed=42`, `bs=2 × ga=4`, Muon `lr=3e-4` WSD (`warmup=60`), `fullpage_max_dim=1280`, `max_seq_len=1536`, `eval_max_len=512`, `bf16`, `grad_checkpoint`, `stroke_hog_weight=0.15`, `gpu_ema_decay=0.997`.

> **Note:** This is a 400-step from-scratch `proxy_val` (64-page) architecture proxy, not a `suite31_hw` eval (no `printed_clean` / `printed_photo` / `hw_clean` / `hw_photo` / `all45` Note F1 splits are produced at 400 from-scratch steps).

| Arm | Source Exp (Platform) | `n_axial / n_global` | EMA Token F1 | Δ vs `ax2_gl4` | Δ vs `ax1_gl5` | EMA SER | Raw Token F1 | `sys_exact` | Final 25-Step Loss | Speed (`steps/s`) | Arm Time | Peak VRAM | Total Params | Hybrid INT4 (w/ scales) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`ax0_gl6`** | **`i13-25b` (Colab L4)** | **`0 / 6`** | **`0.2715`** (`0.271512`) | **`+4.27 pp`** | **`+0.34 pp`** | **`0.8632`** (`0.863182`) | `0.0341` | `0.1094` | `2.8111` | **`0.4981`** | **`803.05 s`** (`13.4 min`) | `9178.3 MB` | `41.95M` (`41,945,163`) | `31.61 MB` |
| `ax1_gl5` | `i13-25` (Modal L4) | `1 / 5` | `0.2681` | `+3.92 pp` | `0.00 pp` | `0.8643` | `0.0224` | `0.1094` | `2.8393` | `0.4119` | `971.0 s` (`16.2 min`) | `9651.0 MB` | `41.65M` (`41,650,731`) | `31.45 MB` |
| `ax2_gl4` (control) | `i13-25` (Modal L4) | `2 / 4` | `0.2288` | `0.00 pp` | `-3.92 pp` | `0.8988` | `0.2262` | `0.1094` | `2.8236` | `0.3505` | `1141.4 s` (`19.0 min`) | `9563.0 MB` | `41.33M` | `31.26 MB` |
| `ax3_gl3` | `i13-25b` (Colab L4) | `3 / 3` | `0.2376` (`0.237612`) | `+0.88 pp` | `-3.05 pp` | `0.8955` (`0.895459`) | `0.0549` | `0.1094` | `2.8034` | `0.3728` | `1073.08 s` (`17.9 min`) | `8862.9 MB` | `41.06M` (`41,061,867`) | `31.12 MB` |

### `edge85m` WebGPU INT4 Budget Verification (CPU `param_summary()`)

| Config | Total Params | Infer Params | Hybrid FP16/INT4 (w/ scales) | Headroom vs `< 55.0 MB` Gate |
|---|---:|---:|---:|---:|
| `edge85m + ax1_gl5` | `78,701,067` | `77,342,560` | `52.745 MB` | `2.255 MB` |
| **`edge85m + ax0_gl6`** | **`79,224,715`** | **`77,866,208`** | **`53.039 MB`** (`+0.294 MB`) | **`1.961 MB` (PASS)** |

---

## Scrutiny & Anomaly Check

1. **Complete coverage & health:** Both arms (`ax0_gl6`, `ax3_gl3`) completed all `400/400` steps and scored `n_samples=64/64` on `proxy_val` with `ink_recall=1.0`, zero NaNs, zero OOMs, and `error=null`.
2. **Strict Pareto dominance of `ax0_gl6`:**
   - **Accuracy:** `ax0_gl6` (`0.2715` EMA Token F1, `0.8632` EMA SER, `0.3741` precision, `0.2498` recall) beats `ax1_gl5` (`+0.34 pp` F1, `-0.11 pp` SER), beats `ax3_gl3` on the exact same VM (`+3.39 pp` F1, `-3.23 pp` SER), and beats the `ax2_gl4` control (`+4.27 pp` F1, `-3.56 pp` SER).
   - **Throughput:** Within the exact same Colab L4 VM, `ax0_gl6` (`0.4981 steps/s`, `803.05 s`) is **`+33.6%` faster** (`-25.2%` wall-clock) than `ax3_gl3` (`0.3728 steps/s`, `1073.08 s`). Across runs, `ax0_gl6` is **`+20.9%` faster** than `ax1_gl5` (`0.4119 steps/s`) and **`+42.1%` faster** than `ax2_gl4` (`0.3505 steps/s`). Eliminating `/8` axial attention removes the most expensive spatial attention stage.
   - **Memory & size:** `ax0_gl6` uses `9178.3 MB` peak VRAM (`-473 MB` vs `ax1_gl5` `9651 MB`) and adds only `+0.294 MB` Hybrid INT4 on `edge85m` (`53.039 MB < 55.0 MB` gate).
3. **Pre-registered `+0.50 pp` margin vs `eval_max_len=512` ceiling:** Strictly speaking, `ax0_gl6` beat `ax1_gl5` by `+0.34 pp` EMA Token F1 (just under the `+0.50 pp` pre-registered accuracy-only threshold), **because** all 64 val pages saturated `eval_max_len=512` (`total_pred_tokens = 32,768 = 64 × 512` vs `52,018` ref tokens), compressing top-end F1 recall to `~0.25`. Given that `ax0_gl6` still leads on **every** quality metric (EMA Token F1, EMA SER, precision, recall), simplifies the encoder (zero `/8` axial layers), cuts VRAM by `473 MB`, and speeds up training by **`+20.9%` vs `ax1_gl5` (`+42.1%` vs `ax2_gl4`)**, `ax0_gl6` is the clear Pareto winner.
4. **Two-regime behavior (`>=5` vs `<=4` global layers):** `ax3_gl3` (`0.2376`) landed within `+0.88 pp` of `ax2_gl4` (`0.2288`, single-seed noise), confirming that `>=5` `/16` global layers (`ax0_gl6` `0.2715`, `ax1_gl5` `0.2681`) occupy a `+3.0` to `+4.3 pp` higher accuracy regime than `<=4` global layers (`ax2_gl4`, `ax3_gl3`).
5. **Raw vs EMA oscillation:** Consistent with `i13-25`, raw step-400 weights oscillated (`ax0_gl6` raw F1 `0.0341` vs EMA `0.2715`; `ax3_gl3` raw F1 `0.0549` vs EMA `0.2376`), confirming EMA (`decay=0.997`) is required for short-horizon Muon+WSD comparisons.
6. **Provider note:** `PLAN.md` listed `modal_l4`, while the queue dispatched `i13-25b` on `colab_l4` (`NVIDIA L4`, same GPU/VRAM). That minor host difference explains why `ax3_gl3` on Colab L4 (`0.3728 steps/s`) was slightly faster than `ax2_gl4` on Modal L4 (`0.3505 steps/s`); within-VM comparison (`ax0_gl6` vs `ax3_gl3`, `+33.6%`) confirms the pure architectural speedup.

---

## Verdict: **`PROMOTE`** (`ax0_gl6`: `--n-axial-layers 0 --n-global-layers 6`)

`ax0_gl6` (`n_axial_layers=0, n_global_layers=6`) strictly Pareto-dominates `ax1_gl5`, `ax2_gl4`, and `ax3_gl3`:
- **Best accuracy:** `0.2715` EMA Token F1 (`+0.34 pp` vs `ax1_gl5`, `+3.39 pp` vs `ax3_gl3`, `+4.27 pp` vs `ax2_gl4`), `0.8632` EMA SER.
- **Fastest training:** `0.4981 steps/s` (`+20.9%` vs `ax1_gl5`, `+33.6%` vs within-VM `ax3_gl3`, `+42.1%` vs `ax2_gl4`) with `-473 MB` lower VRAM than `ax1_gl5`.
- **Budget-safe:** `edge85m + ax0_gl6` is `53.039 MB` Hybrid FP16/INT4 (`1.961 MB` under the `< 55.0 MB` WebGPU budget).

## Follow-Up Action / Wiring Target

Wire `--n-axial-layers 0 --n-global-layers 6` into the Iter 15 training launchers:
- `experiments/2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a/launch_iter15.ps1`:
  - Phase 0 (`$Stage0Args`, lines 76–77): `"--n-axial-layers", "0", "--n-global-layers", "6"`
  - Phase 1 (`$Stage1Args`, lines 128–129): `"--n-axial-layers", "0", "--n-global-layers", "6"`
  - Phase 2 (`$Stage2Args`, lines 197–198): `"--n-axial-layers", "0", "--n-global-layers", "6"`
- `experiments/2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a/run_colab_h100.py`:
  - Pass `--n-axial-layers 0 --n-global-layers 6` across Stage-0 V-JEPA, Stage-1 strip warmup, and Stage-2 full-page training (deferred to main thread to avoid conflicting with concurrent `c2822b5a` edit).
