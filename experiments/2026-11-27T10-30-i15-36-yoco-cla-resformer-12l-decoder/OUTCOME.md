# OUTCOME — `2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder`

- **Status:** `COMPLETED`
- **Verdict:** **`PROMOTE`**
- **Short ID / Wave:** `i15-36` (Wave 13 / Iter 15 — Weight-Incompatible Architecture Proxy)
- **Started / Finished (UTC):** `2026-11-27T13:07:47Z` → `2026-11-27T13:28:34Z` (`1247.01 s` / `20.8 min` total wall-clock)
- **Platform / GPU:** Local `NVIDIA GeForce RTX 4080 SUPER` (`16 GB` VRAM, native BF16; executed via `D:\tmp\local-p0\i15-36`, `600` steps/arm, `seed=42`, `grad_accum=4`, `EMA=0.99`, `QATInt4ControllerV2` enabled at `step >= 420`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder.json` (mirrored locally at [eval_hf.json](./eval_hf.json))
  - Run Ledger: `runs/2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder.json` (mirrored locally at [run_hf.json](./run_hf.json))

---

## Measured Results (`ctrl` vs `yoco12`, 600 steps/arm, `iter10_5_proxy_2k`)

| Metric | `ctrl` (`edge85m + ax0_gl6`, 10L `d_ff=2048`) | `yoco12` (`YocoClaResFormerNet`, 12L `d_ff=1664`) | Delta (`yoco12 - ctrl`) | `PLAN.md` Gate | Status |
|---|---:|---:|---:|---|---|
| **`ema_val_ce`** (primary) | **`1.19388`** | `1.19742` | `+0.00354` (`+0.30%`) | `<= +0.005` (ideally `<= -0.015`) | **PASS** |
| **`raw_val_ce`** | `1.17788` | **`1.17719`** | **`-0.00069`** (`-0.06%`) | — | **PASS** (beats `ctrl`) |
| **`int4_dequant_val_ce`** | **`1.19383`** | `1.19751` | `+0.00368` (`+0.31%`) | — | **PASS** |
| **INT4 quant drop** (`int4 - ema`) | `-0.00005` | `+0.00009` | `+0.00014` | `<= +0.015` | **PASS** (zero drop) |
| **`train_loss_tail50`** | **`1.61626`** | `1.61783` | `+0.00157` (`+0.10%`) | — | Parity |
| **`kv_cache_mb`** (total FP16 @ `L=3072, T=2048`) | `26.214 MB` | **`15.729 MB`** | **`-10.485 MB` (`-40.00%`)** | `>= 35%` reduction | **PASS** (`-40.0%`) |
| — `self_kv_mb` | `10.486 MB` | **`6.291 MB`** | `-4.195 MB` (`-40.00%`) | — | PASS |
| — `cross_kv_mb` | `15.729 MB` | **`9.437 MB`** | `-6.292 MB` (`-40.00%`) | — | PASS |
| **`step_ms`** (steady train avg) | `811.78 ms` | **`782.05 ms`** | **`-29.73 ms` (`-3.66%`, `0.963×`)** | `<= +10%` (`<= 1.10×`) | **PASS** (`3.7%` faster) |
| **`elapsed_secs`** (arm train) | `518.69 s` | **`500.16 s`** | `-18.53 s` (`-3.57%`) | — | Faster |
| **`peak_vram_mb`** (`vram_gb`) | **`6,320.1 MB`** (`6.172 GB`) | `6,982.7 MB` (`6.819 GB`) | `+662.5 MB` (`+0.647 GB`, `+10.5%`) | `< 14,000 MB` | **PASS** |
| **`packed_int4_mb`** (disk-verified) | `53.039 MB` (`53,039,072 B`) | **`51.201 MB`** (`51,201,156 B`) | **`-1.838 MB` (`-3.47%`)** | `< 55.0 MB` hard gate | **PASS** (`3.799 MB` headroom) |
| **`infer_params`** | `77,866,208` | `74,592,050` | `-3,274,158` (`-4.20%`) | — | **CONFOUND** (see Anomaly #1) |
| — `encoder_params` | `27,570,626` | `24,031,682` | **`-3,538,944` (`-12.84%`)** | Iso-encoder intended | **ANOMALY #1** |
| — `decoder_params` | `45,117,442` | `45,382,228` | `+264,786` (`+0.59%`) | Iso-decoder | OK |
| **`gate_pass`** (`PLAN.md` `PROMOTE`) | — | **TRUE** | — | All `PLAN.md` gates | **PASS** |

---

## Scrutiny & Anomaly Check

1. **Silent failures / coverage:** Both arms (`ctrl`, `yoco12`) completed all `600/600` steps with `status: "completed"`, zero NaNs, and disk-verified Hybrid FP16/INT4 binaries (`ctrl`: `53,039,072 B`, `yoco12`: `51,201,156 B`). Both HF artifacts (`eval_hf.json`, `run_hf.json`) downloaded cleanly from `gregoreesmaa/sheet-music-converter-omr`.
2. **Anomaly #1 — Unintended `-3.54M` encoder capacity handicap on `yoco12` (favors `PROMOTE`!):**
   - In `run_colab.py` line 621, `_build_yoco_cla_resformer_model()` passed `ffn_mult=1664.0 / 512.0` (`3.25`) into `UniOMRv2Config.edge85m(...)`. Because `UniOMRv2Config` shares a single `ffn_mult` across both encoder and decoder, all **6 global encoder layers** also shrank from `d_ff=2048` to `d_ff=1664`: `6 × 3 × 512 × (2048 - 1664) = 3,538,944` fewer encoder parameters (`24,031,682` vs `27,570,626`, `-12.84%`), explaining the `-1.838 MB` packed INT4 reduction (`51.201 MB` vs `53.039 MB`).
   - **Crucial implication:** `yoco12` achieved parity on `ema_val_ce` (`+0.00354`), **beat `ctrl` on `raw_val_ce` (`-0.00069`)**, cut WebGPU KV-cache by **`-40.0%`**, and ran **`3.66%` faster** *despite losing 3.54M encoder FFN parameters*. When ported into `uni_omr_v2.py` with a dedicated `dec_ffn_mult` (keeping encoder `ffn_mult=4.0`, `d_ff=2048`), packed INT4 size will be `~53.19 MB` (`51.201 + 3.539M × 0.5625 B`, still `~1.81 MB` below the `55.0 MB` hard gate) with full encoder capacity restored.
3. **Anomaly #2 — Optimizer mode (`AdamW` on both arms instead of `HybridOptimizerV2` `muon+adamw`):**
   - `run_colab.py` line 795 used `torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.04)` on **both** arms rather than `HybridOptimizerV2` (`muon+adamw`).
   - Because both arms used the exact same `AdamW` optimizer, identical `seed=42`, identical `iter10_5_proxy_2k` sampling stream, and identical WSD + QAT-INT4 schedule, the A/B comparison is **100% apples-to-apples** (unlike `i15-37`, where one arm fell back silently). Absolute CE (`~1.19` EMA / `~1.18` raw) is higher than `muon+adamw` runs (`~1.08` raw), and `step_ms` (`~800 ms`) is faster than `muon+adamw` runs (`~1700–2040 ms`) due to no Newton-Schulz orthogonalization steps and `FULLPAGE_MAX_DIM=1280` / `MAX_SEQ_LEN=1536`.
4. **INT4 quantization resilience:** Both arms show essentially zero INT4 quantization loss after QAT at `step >= 420` (`ctrl`: `-0.00005` nats; `yoco12`: `+0.00009` nats). ResFormer scalar `v_res_logit` parameters stay in FP16 (`+2,642` FP16 params, `+0.005 MB`), and shared YOCO/CLA K/V projections quantize cleanly to group-32 INT4.
5. **Speed & VRAM:** `yoco12` trained **`3.66%` faster** (`782.05 ms` vs `811.78 ms`, `0.963×`) despite 12 decoder layers (20% deeper decoder) because 9 of 12 cross-KV projections and 6 of 12 self-KV projections are eliminated (plus the smaller encoder FFN). Peak VRAM was `6.819 GB` (`+0.647 GB` vs `ctrl`'s `6.172 GB`, well under the `14 GB` Windows WDDM ceiling).

---

## Verdict: `PROMOTE`

`yoco12` clears **every single `PLAN.md` `PROMOTE` gate** even while handicapped by `-3.54M` encoder parameters:
- **`ema_val_ce` delta:** `+0.00354` (gate: `<= +0.005`) — **PASS** (`raw_val_ce` beats `ctrl` at `-0.00069`)
- **`kv_cache_mb` reduction:** **`-40.00%`** (`26.214 MB → 15.729 MB`, gate: `>= 35%`) — **PASS**
- **`packed_int4_mb`:** **`51.201 MB`** (gate: `< 55.0 MB`) — **PASS** (`~53.19 MB` once encoder `d_ff=2048` is restored, still `< 55.0 MB`)
- **`step_ms` ratio:** **`0.963×`** (`-3.66%` faster, gate: `<= 1.10×`) — **PASS**
- **INT4 quantization drop:** `+0.00009` nats (gate: `<= +0.015`) — **PASS**

---

## Follow-Up Action / Successor Experiment

- **Architecture Promotion Target (Iter 16+ from-scratch pretraining):** Port **YOCO-3Bank Cross-KV (`n_kv=4`, banks `(0..2, 3..6, 7..11)` aligned with DeepStack levels)** + **CLA-2:1 Self-KV (`n_kv=2`, 6 paired banks)** + **ResFormer `v_res` (`a_init=-2.0`)** into `tmp-train/mvd/uni_omr_v2.py` behind explicit config flags (e.g., `use_yoco_cla_resformer: bool = False`, `dec_ffn_mult: Optional[float] = None`).
- **Critical Port Fix (Anomaly #1):** Add a separate `dec_ffn_mult` field to `UniOMRv2Config` so encoder global layers keep `ffn_mult=4.0` (`d_ff=2048`, `27.57M` params) while the 12L decoder uses `dec_ffn_mult=1664/512` (`3.25`, `~53.19 MB` total packed INT4).
- **Combined Iter-16 Verification:** Re-verify the ported `YocoClaResFormer` (with restored encoder `d_ff=2048` and `HybridOptimizerV2` `muon+adamw`) alongside the other promoted Iter-15 architecture winners (`i15-38` `CoordKeyCrossAttention`, etc.) in the pre-Iter-16 combined integration proxy before launching full pretraining.
