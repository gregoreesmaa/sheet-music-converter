# Outcome — `2026-11-26T22-18-i14-05-edge85m-int4-capacity-proxy`

- **Status:** `COMPLETED` (`returncode=0`, Colab session `2026-11-26t22-18-i14-05-` cleanly terminated)
- **Started / Completed (UTC):** `2026-11-26T21:34:44Z` → `2026-11-27T03:35:43Z`
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, native `BF16=true`)
- **Elapsed Time:** `21,658.89 s` (`361.0 min` = `6.02 h`; queue wall-time `21,844.94 s`)
  - **Training time (1,200 steps, `grad_accum=4`):** Arm A `small_45m` = **`1,978.2 s` (`33.0 min`, `1.648 s/step`)** | Arm B `edge85m` = **`2,176.3 s` (`36.3 min`, `1.814 s/step`, only `+10.0%` overhead!)**
  - **Eval time (`3 × 90` pages):** Arm A = `7,651.4 s` (`127.5 min`) | Arm B = `9,812.1 s` (`163.5 min`) — see Scrutiny #2
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T22-18-i14-05-edge85m-int4-capacity-proxy.json` (pulled to [eval.json](./eval.json))
  - Run Ledger: `runs/2026-11-26T22-18-i14-05-edge85m-int4-capacity-proxy.json` (pulled to [run_ledger.json](./run_ledger.json))

---

## 1. Footprint & Exact On-Disk Hybrid FP16/INT4 Verification (`< 55.0 MB` Hard Gate)

| Metric | Arm A (`small_45m`) | Arm B (`edge85m`) | Ratio / Delta (`B vs A`) | `< 55.0 MB` Gate |
|---|---:|---:|---:|:---:|
| Total params | `41,334,795` (`41.33M`) | `78,148,747` (`78.15M`) | `1.89×` | — |
| Infer params (`strip_aux_heads` + no MTP) | `40,493,280` (`40.49M`) | `76,790,240` (`76.79M`) | `1.90×` | — |
| Breakdown (backbone / enc / router / deepstack / dec) | `5.12M / 14.98M / 0.52M / 0.15M / 20.40M` | `5.18M / 26.52M / 0.82M / 0.26M / 45.09M` | Decoder `2.21×` | — |
| `int4_weight_params` (`n_int4_tensors`) | `34,595,712` (`115`) | `70,375,936` (`135`) | `2.03×` | — |
| `fp16_keep_params` (`n_fp16_tensors`) | `5,897,568` (`208`) | `6,414,304` (`222`) | `1.09×` | — |
| Uncompressed FP16 size | `80.987 MB` | `153.580 MB` | `1.90×` | — |
| **Exact packed Hybrid FP16/INT4 on disk (`group_size=32`)** | **`31.255 MB`** (`31,255,224 B`) | **`52.415 MB`** (`52,415,072 B`) | `+21.160 MB` | **PASS (`2.585 MB` headroom)** |
| Packed breakdown (INT4 weights + FP16 scales + FP16 keep) | `17.298 + 2.162 + 11.795 MB` | `35.188 + 4.398 + 12.829 MB` | Exact byte match to formula | **PASS** |
| Training step wall-time (`1,200` steps, L4 BF16) | `1.648 s/step` (`33.0 min`) | `1.814 s/step` (`36.3 min`) | **`+10.0%` (`1.10×`)** | — |

---

## 2. Loss & Convergence Results (`iter10_5_proxy_2k`, `1,200` Steps From Scratch, QAT ON @ Step `840`)

| Metric | Arm A (`small_45m`) | Arm B (`edge85m`) | Delta (`B − A`) | `PLAN.md` PROMOTE Bar (`<= -0.05`) |
|---|---:|---:|---:|:---:|
| `train_loss` @ step 100 | `3.1055` | `2.9334` | `-0.1721` | — |
| `train_loss` @ step 400 | `1.9474` | `1.8517` | `-0.0957` | — |
| `train_loss` @ step 800 (pre-QAT) | `1.6394` | `1.5805` | `-0.0589` | — |
| `train_loss` @ step 1000 (mid-QAT) | `1.5462` | `1.4841` | `-0.0621` | — |
| `train_loss_tail50` @ step 1200 | `1.44084` | **`1.36673`** | **`-0.07411`** | — |
| **Held-out `raw_val_ce` (`64` val pages)** | `1.04693` | **`0.98506`** | **`-0.06187`** | **PASS (`124%` of gate)** |
| **Held-out `ema_val_ce` (`64` val pages, `decay=0.99`)** | `1.05353` | **`0.99114`** | **`-0.06239`** | **PASS (`125%` of gate)** |

---

## 3. `suite31_hw` OOD Readout (`90` Pages, Unmasked Greedy `max_len=2048` After `1,200` Scratch Steps)

| Arm / Checkpoint | `printed_clean` | `printed_photo` | `hw_clean` | `hw_photo` | `all90` | Mean Latency (`s/page`) |
|---|---:|---:|---:|---:|---:|---:|
| `arm_a_small45m_qat` — **RAW** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `27.975 s` |
| `arm_a_small45m_qat` — **EMA** | `0.0067` | `0.0052` | `0.0000` | `0.0019` | `0.0044` | `28.107 s` |
| `arm_a_small45m_qat` — **INT4-dequant** | `0.0014` | `0.0000` | `0.0000` | `0.0000` | `0.0005` | `27.959 s` |
| `arm_b_edge85m_qat` — **RAW** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `35.827 s` |
| `arm_b_edge85m_qat` — **EMA** | `0.0013` | `0.0000` | `0.0000` | `0.0000` | `0.0004` | `36.378 s` |
| `arm_b_edge85m_qat` — **INT4-dequant** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `35.882 s` |
| Delta (`B − A`) — **EMA / INT4** | `-0.0054 / -0.0014` | `-0.0052 / 0.0000` | `0.0000 / 0.0000` | `-0.0019 / 0.0000` | `-0.0040 / -0.0005` | `+28.3%` (`1.28×`) |

---

## 4. Scrutiny & Anomaly Check

1. **Full Coverage & Zero Crashes:** Both arms completed all `1,200` optimizer steps (`4,800` page-views, `2,000` train / `64` val), enabled `QATInt4ControllerV2` cleanly at step `840` with zero NaNs or loss spikes, wrote exact packed Hybrid FP16/INT4 binaries to disk (`31,255,224 B` and `52,415,072 B`, matching `param_summary` to the **exact byte**), and terminated the Colab L4 VM cleanly.
2. **Why `suite31_hw` Note F1 Is Near-Zero (`0.0000–0.0067`) & Why Eval Took `4.85 h` (`80.8%` of Runtime):**
   - Both arms trained from **random initialization** for only `1,200` steps (`2.4` epochs on `2k` pages), converging to `val_ce ~ 0.99–1.05` (perplexity `~2.7–2.9`). At `CE ~ 1.0`, unmasked greedy autoregressive decoding (`model.generate(max_len=2048)`, no FSA, no MTP-4) has not yet learned reliable `<eos>` stopping on full multi-system pages, so almost every page ran to `max_len=2048` (`28–36 s/page`) and failed strict note-sequence alignment.
   - This is an **undertrained-proxy floor**, not an `edge85m` regression: `val_ce` is the valid capacity signal at `1,200` steps from scratch, and `edge85m` beat `small_45m` monotonically from step `100` (`-0.172`) through step `1200` (`-0.074` train / **`-0.0624` EMA `val_ce`** / **`-0.0619` RAW `val_ce`**).
   - **Process lesson for future short scratch proxies:** Skip full 90-page `max_len=2048` greedy `suite31_hw` decoding when `val_ce > 0.65` (or evaluate `int4_dequant_val_ce` on the 64-page val set instead) to avoid wasting `~4.8 h` of GPU time at the F1 noise floor.
3. **INT4 Quantization Parity Caveat:** Numerically, the `ema → int4_dequant` F1 drop is `-0.0004` (`<= 0.015` gate), and QAT stayed completely stable from step `840 → 1200` (`edge85m` widened its train-loss lead from `-0.0589` at step `800` to `-0.0741` at step `1200`). However, because `suite31_hw` F1 was at the noise floor and `int4_dequant_val_ce` was not logged, high-F1 (`> 0.75`) INT4 parity for `edge85m` must still be verified on the converged Iter 15 checkpoint (using the proven Iter 13 last-15% `QATInt4ControllerV2` schedule).
4. **Apples-to-Apples Input Normalization:** `_pil_to_tensor` used `(t - 0.5) / 0.5` consistently across both training and evaluation in both arms, so the A/B comparison is strictly controlled.

---

## 5. Verdict: **`PROMOTE`** (Scale Iter 15 to `edge85m`!)

`edge85m` cleanly clears the `PLAN.md` **`PROMOTE`** gate:
- **`val_ce_loss` gain:** **`-0.06239` EMA `val_ce`** (`1.05353 → 0.99114`) and **`-0.06187` RAW `val_ce`** (`1.04693 → 0.98506`), exceeding the `>= -0.05` promotion threshold by `+25%`.
- **Hard `< 55.0 MB` WebGPU budget:** Verified on disk at **`52.415 MB`** (`52,415,072` bytes exact with real 4-bit nibble packing + FP16 group-32 scales + FP16 keep layers) — **`2.585 MB` headroom below `55.0 MB`**.
- **Near-free training compute cost:** Scaling `1.90×` in inference parameters (`40.49M → 76.79M`, decoder `20.40M → 45.09M`) cost only **`+10.0%` training wall-time** on L4 BF16 (`1.648 → 1.814 s/step`) and **`+28.3%` decode latency** (`1.28×`, nowhere near `2×`).

## 6. Follow-Up Action / Iter 15 Recommendation

- **YES — scale Iter 15 to `UniOMRv2Config.edge85m()` (`d_model=512`, `n_dec_layers=10`, `n_heads=8`, `n_kv_heads=2`, `76.79M` infer params, `52.415 MB` Hybrid FP16/INT4 on disk).**
- Keep `QATInt4ControllerV2(block_size=32)` enabled in the final `15–20%` of Iter 15 training and verify both `int4_dequant_val_ce` and converged `suite31_hw` / `OMR-NorthStar` INT4 Note F1 parity before WebGPU ONNX export.
- Note for future `small_45m` vs `edge85m` proxy scripts: always log `int4_dequant_val_ce` and gate full `suite31_hw` autoregressive decoding behind `val_ce < 0.65`.
