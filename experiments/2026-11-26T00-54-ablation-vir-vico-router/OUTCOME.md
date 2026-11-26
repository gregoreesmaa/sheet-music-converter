# Experiment Outcome — `2026-11-26T00-54-ablation-vir-vico-router`

- **Status:** `COMPLETED`
- **Completed At:** `2026-11-26T00:12:30Z` (`2026-11-26 02:12:30 EET`)
- **Platform / GPU:** Modal `2× NVIDIA L4` (`24 GB`, `cc=8.9`, native BF16, parallel arms) — `app_id: ap-8ZDdiVYjK1SGxUf0v43qH7` (verified `stopped`, `Tasks=0`)
- **Elapsed Time:** `486.9s` (`8.1 min`) final resumed segment (`baseline` `424.9s`, `mean_step_sec=3.466s`; `vir_vico` `427.7s`, `mean_step_sec=2.874s`; ~59 min total wall-clock across original + resumed segment)
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`n_train=2,000`, `n_val=64`, `fullpage_max_dim=1280`, `max_seq_len=1536`, `seed=42`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`, `repo_type="dataset"`):**
  - Eval JSON: `evals/2026-11-26T00-54-ablation-vir-vico-router.json` (pulled to [eval_result.json](./eval_result.json))
  - Run Ledger: `runs/2026-11-26T00-54-ablation-vir-vico-router.json`
  - Best EMA Checkpoints: `checkpoints/2026-11-26T00-54-ablation-vir-vico-router/{baseline,vir_vico}/uni_omr_v2_ema_best.pt`

---

## Measured Results (`proxy_val.jsonl`, `n=64`, Step `1,200`)

### 1. Primary EMA Comparison (Step 1,200)

| Metric | Arm 1: `InkTokenRouterV2` (Baseline) | Arm 2: `ViRTwoTierRouter+ViCO` | Delta (`ViR - Base`) | Winner |
|---|---:|---:|---:|---|
| `val_loss` ↓ | **1.68974** | 1.70258 | `+0.01284` | Baseline |
| `val_ce` ↓ | **1.08248** | 1.08548 | `+0.00300` | Baseline |
| `val_mtp` ↓ | **1.87730** | 1.88316 | `+0.00586` | Baseline |
| `val_saliency` ↓ | **0.05523** | 0.06335 | `+0.00812` | Baseline |
| `val_stroke_hog` ↓ | **0.02289** | 0.02345 | `+0.00056` | Baseline |
| `SER` ↓ | **0.868005** | 0.932919 | `+0.064914` | Baseline |
| `token_f1` ↑ | **0.277658** | 0.193663 | **`-0.083995`** | Baseline |
| `token_precision` ↑ | **0.379669** | 0.253316 | `-0.126353` | Baseline |
| `token_recall` ↑ | **0.255644** | 0.188793 | `-0.066851` | Baseline |
| `sys_count_exact` ↑ | **0.171875** (`11/64`) | 0.140625 (`9/64`) | `-0.031250` | Baseline |
| `ink_recall` ↑ | **0.9658** | 0.9030 | **`-0.0628`** | Baseline |
| `mean_compression_x` ↑ | 2.140× (`2186.4` toks) | **2.327×** (`2096.0` toks) | `+0.187×` | ViR+ViCO |
| `tokens_per_sec` ↑ | **224.93** | 221.50 | `-3.43` (`-1.5%`) | Baseline |

### 2. Raw (Un-Lagged) Weights Comparison (Step 1,200)

| Metric | Arm 1: `InkTokenRouterV2` (Baseline) | Arm 2: `ViRTwoTierRouter+ViCO` | Delta (`ViR - Base`) | Winner |
|---|---:|---:|---:|---|
| `val_loss` ↓ | **1.58527** | 1.59191 | `+0.00664` | Baseline |
| `val_ce` ↓ | **1.01478** | 1.01841 | `+0.00363` | Baseline |
| `val_mtp` ↓ | **1.77821** | 1.78407 | `+0.00586` | Baseline |
| `val_saliency` ↓ | **0.04631** | 0.05288 | `+0.00657` | Baseline |
| `SER` ↓ | **0.851399** | 0.861785 | `+0.010386` | Baseline |
| `token_f1` ↑ | **0.350249** | 0.289824 | **`-0.060425`** | Baseline |
| `token_precision` ↑ | **0.477330** | 0.412964 | `-0.064366` | Baseline |
| `token_recall` ↑ | **0.327172** | 0.259785 | `-0.067387` | Baseline |
| `sys_count_exact` ↑ | **0.281250** (`18/64`) | 0.187500 (`12/64`) | **`-0.093750`** | Baseline |
| `ink_recall` ↑ | **0.9643** | 0.9033 | **`-0.0610`** | Baseline |
| `mean_compression_x` ↑ | 2.149× (`2171.8` toks) | **2.327×** (`2096.0` toks) | `+0.178×` | ViR+ViCO |
| `tokens_per_sec` ↑ | 224.71 | **225.20** | `+0.49` (`+0.2%`) | Parity |
| `mean_step_sec` (train) ↓ | 3.466s | **2.874s** | **`-0.592s` (`-17.1%`)** | ViR+ViCO |
| `router_params` | 518,210 (`41.33M` tot) | **369,986** (`41.19M` tot) | `-148,224` | ViR+ViCO |

---

## Scrutiny & Anomaly Check

1. **Coverage & Integrity:** `64/64` held-out validation pages scored on both arms (`total_ref_tokens = 53,055`), zero NaNs, zero crashes, Modal app `ap-8ZDdiVYjK1SGxUf0v43qH7` verified `stopped` (`Tasks=0`). No collapsed router — `ViRTwoTierRouter` produced `2096.0` mean output tokens (`2.327×` compression) and `vico_loss` fired cleanly on every `step % 4 == 0` (`0.258–0.271` at steps `1184–1196`).
2. **CRITICAL Confounder #1 — Static-Bucket Ceiling (`2048`) Starved `ink_recall` by `-6.28 pp`:**
   - `DEFAULT_STATIC_BUCKETS` in `tmp-train/mvd/vir_router_v2.py:46` is `(256, 512, 768, 1024, 1536, 2048)`, whose maximum bucket is **`2048`**, even though `max_mem_tokens = 3072`.
   - On dense `1280px` validation pages (`mean_n_in = 5102.5`), `snap_to_static_bucket` hard-truncated `k_fine` to `2048` (`n_out <= 2144` including `k_coarse=96`), dropping **6.1–6.3 pp of ink pixels** (`ink_recall = 0.9030` vs `0.9658` for `InkTokenRouterV2`). Every single training step in `vir_vico`'s `tail_train_history` was clamped to `n_out = 1632` (`1536+96`) or `2144` (`2048+96`), whereas `baseline` dynamically kept up to `3072` tokens!
3. **CRITICAL Confounder #2 — Training Token Budget Mismatch (`0.40` vs `0.75`):**
   - Under `dynamic_routing=True`, `InkTokenRouterV2.forward` (`uni_omr_v2.py:787`) keeps `min(ceil(N * max_keep_ratio), 3068)` = **`0.75 * N`** (~2,750 tokens/page) during training.
   - `ViRTwoTierRouter.resolve_k_fine` (`vir_router_v2.py:343`) instead kept `snap(ceil(N * keep_ratio))` = **`0.40 * N`** (~1,950 tokens/page — **~30% fewer visual tokens per step**). This explains BOTH why `vir_vico` ran **`17.1%` faster per step** (`2.874s` vs `3.466s` even with the dense ViCO teacher forward every 4 steps) AND why it under-performed on `val_ce`, `token_f1`, and `sys_count_exact`.
4. **EMA Half-Life Lag at 1,200 Steps:**
   - `GPUEmaShadow(decay=0.997, update_every=5)` executed only `240` EMA updates over `1,200` steps (half-life = `231` updates = `1,155` steps), so EMA lagged Raw on **both** arms (`baseline` raw F1 `0.3502` vs EMA `0.2777`; `vir_vico` raw F1 `0.2898` vs EMA `0.1937`). Crucially, `baseline` beat `vir_vico` decisively on **both** Raw (`-6.04 pp` F1, `-9.38 pp` `sys_count_exact`) and EMA (`-8.40 pp` F1).
5. **Mid-Run Resume & Eval Cap Telemetry:**
   - Both arms resumed symmetrically from 5-min checkpoints on `/vol/ckpts/exp3_vir_vico/` around step `~1080` (`wall_clock_sec = 486.9s`, `eval_history` contains only step `1200`), and `run_modal.py` resume did not restore optimizer state (`opt`). Because both arms resumed symmetrically during the low-LR WSD tail (`1.5e-5`), the comparison remained fair, but future scripts should save/restore `opt.state_dict()`.
   - Proxy generation used `eval_max_len = 512` (`total_pred_tokens = 32,679–32,768` vs `64 × 512 = 32,768` cap, `total_ref_tokens = 53,055`), capping long pages identically on both arms.

---

## Verdict & Iter 11 (`UniOMR v2`) Router Decision

- **Verdict:** **`ITERATE`**
- **Iter 11 (`UniOMR v2`) Router Decision:** **KEEP `InkTokenRouterV2` as the default router for Iter 11.** `ViRTwoTierRouter+ViCO` failed the promotion gate across every accuracy metric on both Raw and EMA (`val_ce +0.0030/+0.0036`, `SER +0.0649/+0.0104`, `token_f1 -0.0840/-0.0604`, `ink_recall -0.0628/-0.0610`).
- **Why `ITERATE` (not `DISCARD`):** Two concrete, fixable confounders in `ViRTwoTierRouter` (`2048` static-bucket ceiling + `0.40` vs `0.75` training keep-ratio mismatch) directly caused the `-6.28 pp` `ink_recall` loss and ~30% training token starvation, while `ViRTwoTierRouter` still demonstrated `-17.1%` faster step time and pure-float32 static-bucket WebGPU compatibility.

## Follow-Up Action / Successor Experiment

- **Queued Successor:** [`2026-11-26T02-18-ablation-vir-router-bucket-parity`](../2026-11-26T02-18-ablation-vir-router-bucket-parity/) (registered in `experiments/overnight_queue/queue.json` with `depends_on: ["2026-11-26T00-54-ablation-vir-vico-router"]`).
- **What it tests (3 arms on Colab L4, 600 steps each, `seed=42`):**
  1. `baseline_ink_router`: `InkTokenRouterV2` (`max_keep_ratio=0.75`, `max_mem_tokens=3072`) — exact Iter 11 reference.
  2. `vir_parity_no_vico`: `ViRTwoTierRouter` with expanded buckets `(256, 512, 768, 1024, 1536, 2048, 2560, 2976)`, `keep_ratio=0.75` during training (matching `InkTokenRouterV2` token budget), `k_coarse=96`, **`vico_weight=0.0`** (pure 2-tier router isolation).
  3. `vir_parity_vico005`: Same bucket-parity `ViRTwoTierRouter` + gentle ViCO (`vico_weight=0.05`, `vico_every=2`, dense teacher `0.85`).
- If `vir_parity_no_vico` or `vir_parity_vico005` matches `ink_recall >= 0.96` AND beats `baseline_ink_router` on `token_f1`/`SER`, promote `ViRTwoTierRouter` for **Iter 12**; otherwise **DISCARD** `ViRTwoTierRouter` permanently and keep `InkTokenRouterV2`.
