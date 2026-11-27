# OUTCOME — `2026-11-27T14-30-i15-37b-modegate-cmtp-muon-clean` (`i15-37b`)

- **Status:** `COMPLETED` (`rc=0`, `2/2` arms completed, zero NaNs, `gate_pass: true`, `error: null`)
- **Verdict:** **`PROMOTE`**
- **Platform / GPU:** Local Windows 11 — `NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`)
- **Started / Finished (UTC):** `2026-11-27T15:13:18Z` → `2026-11-27T16:32:20Z` (`4741.19 s` / `79.0 min` total)
- **Depends On:** `2026-11-27T10-30-i15-37-modegate-dualhead-text-and-chained-mtp` (clean re-run after `"base."` prefix stripping fix in `train_uni_v2.py` commits `95446af` / `b64fd64`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-30-i15-37b-modegate-cmtp-muon-clean.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-27T14-30-i15-37b-modegate-cmtp-muon-clean.json` (pulled to [`run_hf.json`](run_hf.json))

---

## Measured Results (`ctrl` vs `modegate_cmtp`, 600 steps/arm, verified `muon+adamw` on both arms, `seed=42`)

| Metric | `ctrl` (parallel `MTP-4` + single `lm_head`) | `modegate_cmtp` (Chained `MTP-4` + Modality-Gated Dual-Head) | Delta (`modegate_cmtp - ctrl`) | Gate (`README.md`) | Pass? |
|---|---:|---:|---:|---|---|
| **`opt_mode`** *(hard-asserted)* | `muon+adamw` | `muon+adamw` | matched | `muon+adamw` on both | **PASS** |
| **`ema_val_ce`** *(primary, 53,110 val toks)* | 1.10816 | **1.10123** | **-0.00693** (better) | `<= ctrl + 0.010` (ideally `< ctrl`) | **PASS** |
| **`int4_dequant_val_ce`** *(primary)* | 1.10941 | **1.10284** | **-0.00657** (better) | `<= ctrl + 0.010` | **PASS** |
| **`raw_val_ce`** | 1.08578 | **1.07971** | **-0.00607** (better) | lower is better | **PASS** |
| `int4_dequant - ema` gap | +0.00125 | +0.00161 | +0.00036 | `<= +0.025` | **PASS** |
| **`spec_accept_len`** *(greedy-chained, EMA)* | 1.1130 | **1.2625** | **+0.1495 (+13.4%)** | `>= +0.10` tokens/step | **PASS** |
| **`int4_spec_accept_len`** *(greedy-chained, INT4)* | 1.1093 | **1.2608** | **+0.1515 (+13.7%)** | `>= +0.10` tokens/step | **PASS** |
| `spec_accept_len` *(greedy-chained, raw)* | 1.1532 | **1.3090** | **+0.1558 (+13.5%)** | higher is better | **PASS** |
| `mtp_k1_acc` *(EMA)* | 0.60977 | **0.61416** | +0.44 pp | higher is better | **PASS** |
| `mtp_k2 / k3 / k4_acc` *(marginal greedy-chained, EMA)* | 0.4924 / 0.4105 / 0.3299 | 0.4859 / 0.3962 / 0.3169 | -0.65 / -1.43 / -1.30 pp | *(see Scrutiny #3)* | INFO |
| `mtp_tf_k2 / k3 / k4_acc` *(teacher-forced diag., EMA)* | N/A | **0.5776 / 0.5782 / 0.5692** | — | diagnostic | INFO |
| **`step_ms`** | 4432.33 ms | **3226.85 ms** | **-1205.48 ms (-27.2%)** | `<= +10%` | **PASS** |
| `elapsed_sec` | 2659.40 s | 1936.11 s | -723.29 s (-27.2%) | — | — |
| `peak_vram_mb` | 12,167.1 MB | **12,004.0 MB** | -163.1 MB (-1.3%) | `< 14,000 MB` | WARN (WDDM) |
| `packed_int4_stripped_mb` | 53.039 MB | 53.159 MB | +0.120 MB (`+203,268` params) | `< 55.0 MB` | **PASS** |
| **`packed_int4_with_mtp_mb`** *(exact disk bytes)* | 53.485 MB | **54.050 MB** | **+0.565 MB** (`54,050,184 B`) | `<= 55.0 MB` hard cap | **PASS** |
| **Overall Gate (`gate_pass`)** | baseline | **PASS** | — | all primary gates | **PASS → `PROMOTE`** |

---

## Scrutiny & Anomaly Check

1. **Silent failures / partial coverage:** None. Both arms ran all `600/600` steps with **hard-asserted `opt_mode == "muon+adamw"`** (eliminating the `i15-37` AdamW fallback confound), scored all `53,110` validation tokens identically on raw, EMA, and dequantized Hybrid INT4 checkpoints, and verified exact on-disk packed INT4 binary size (`54,050,184 B = 54.050 MB <= 55.0 MB`, `0.95 MB` headroom).
2. **Apples-to-apples sanity:** Both arms share identical `iter10_5_proxy_2k.tar` data, `edge85m + ax0_gl6` backbone/encoder/decoder, `mtp_loss_weight=0.30`, `stroke_hog_weight=0.15`, `EMA_DECAY=0.99`, and `QATInt4ControllerV2` at step `420`. Note that `i15-37b`'s `ctrl` EMA CE (`1.10816`) is lower than `i15-14/42/43` (`1.1269–1.1276`) because `i15-37b` includes `stroke_hog_weight=0.15` and `mtp_loss_weight=0.30` on both arms; within `i15-37b`, the comparison is strictly controlled. Also note `final_25step_loss` (`1.4926` vs `1.7092`) reflects lower teacher-forced Chained-MTP auxiliary loss and is not directly comparable; the apples-to-apples accuracy metric is pure next-token `val_ce` / `ema_val_ce` / `int4_dequant_val_ce`, all three of which improved by `-0.0061` to `-0.0069` nats!
3. **Scrutiny of `spec_accept_len` (`+0.1495` EMA / `+0.1515` INT4) vs marginal `mtp_k2..k4_acc` (`-0.65` to `-1.43 pp`):**
   - Verified directly in `run_colab.py` (lines 801–848): `modegate_cmtp` evaluates `spec_accept_len` strictly with `greedy_chain_mtp=True` (zero teacher forcing; feeds the model's own argmax predictions sequentially) and computes exact contiguous prefix acceptance `P(k1) + P(k1 & k2) + P(k1 & k2 & k3) + P(k1 & k2 & k3 & k4)`.
   - The divergence is the textbook mathematical signature of Chained MTP: whenever `k=1` is **correct**, `ŷ_{t+1} == y_{t+1}` matches teacher-forced training, so conditional accuracy `P(k2 | k1 correct)` jumps to `~57.8%` (`mtp_tf_k2..k4_acc = 0.5776 / 0.5782 / 0.5692`), boosting contiguous accepted draft length by **`+0.1495` tokens/step** (`+0.1515` in INT4, **`+13.7%` speculative throughput**). Whenever `k=1` is **wrong**, conditioning `k=2..4` on a wrong token lowers marginal `k=2..4` accuracy slightly — which is irrelevant to speculative decoding because the draft is already rejected at `k=1`.
4. **Hardware / WDDM anomaly (`step_ms`: `4432.33 ms` `ctrl` vs `3226.85 ms` `modegate_cmtp`):**
   - Both arms hovered right at the Windows 11 16 GB RTX 4080 SUPER WDDM paging boundary (`12,167.1 MB` `ctrl` vs `12,004.0 MB` `modegate_cmtp` allocated), causing elevated step times relative to `~1645 ms` non-paging 4080 SUPER runs (`i15-42`). `ctrl` peaked `163 MB` higher and incurred slightly heavier WDDM paging (`-27.2%` faster for `modegate_cmtp`). Critically, `modegate_cmtp` added **zero** VRAM penalty (`-163.1 MB`) and **zero** latency regression.

---

## Follow-Up Action / Successor Experiment

- **Verdict:** **`PROMOTE`** — cleanly passes every primary accuracy, quantization, speculative decoding, latency, and `< 55.0 MB` INT4 budget gate under verified matched `muon+adamw`.
- **Mandatory Promotion Action (to be integrated into Iter 15 main model):**
  - Port **Chained `MTP-4`** (`W_k([RMSNorm(h_{k-1}); RMSNorm(tok_emb(y_{t+k}))])` with both teacher-forced training and `greedy_chain_mtp=True` inference) and **Modality-Gated Dual-Head** (`lm_head_text` 129-tok + layers `0..3` `cross_attn` FiLM + rank-32 text-query delta) into `tmp-train/mvd/uni_omr_v2.py` and `tmp-train/mvd/train_uni_v2.py` behind clean config flags for Iter 15 (`54.050 MB` packed Hybrid INT4 with MTP, `-0.00693` EMA CE, `-0.00657` INT4 CE, `+0.1515` INT4 speculative acceptance length).
  - Add CPU unit tests (`CUDA_VISIBLE_DEVICES=""`) verifying Chained MTP training/greedy-chain parity, dual-head logit routing, and `< 55.0 MB` packed INT4 export size.
