# OUTCOME — `2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn`

- **Status:** `COMPLETED`
- **Verdict:** **`PROMOTE`**
- **Short ID / Wave:** `i15-38` (Iter 15 — Weight-Incompatible Architecture Proxy)
- **Started / Finished (UTC):** `2026-11-27T10:45:09Z` → `2026-11-27T11:22:26Z` (`2236.48 s` / `37.3 min` total wall-clock)
- **Platform / GPU:** Local `NVIDIA GeForce RTX 4080 SUPER` (`16 GB` VRAM, native BF16; executed via `D:\tmp\local-p0\i15-38`, same `seed=42` / `600` steps/arm protocol as `PLAN.md`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn.json` (mirrored locally at [eval_hf.json](./eval_hf.json))
  - Run Ledger: `runs/2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn.json` (mirrored locally at [run_hf.json](./run_hf.json))

---

## Measured Results (`ctrl` vs `coordkey_ca`, 600 steps/arm, `iter10_5_proxy_2k`)

| Metric | `ctrl` (`edge85m + ax0_gl6`) | `coordkey_ca` (`+ FourierCoordMLP + 16d spatial QK`) | Delta (`coordkey_ca - ctrl`) | `PLAN.md` Gate | Status |
|---|---:|---:|---:|---|---|
| **`ema_val_ce`** (primary) | `1.12739` | **`1.10725`** | **`-0.02014`** (`-1.79%`) | `<= -0.015` | **PASS** |
| **`int4_dequant_val_ce`** (primary) | `1.12911` | **`1.10678`** | **`-0.02233`** (`-1.98%`) | `<= -0.015` | **PASS** |
| **INT4 quant delta (`int4 - ema`)** | `+0.00172` | **`-0.00047`** | `-0.00219` | `<= +0.025` | **PASS** |
| **`raw_val_ce`** | `1.10440` | **`1.07805`** | **`-0.02635`** (`-2.39%`) | — | **PASS** |
| **`train_loss_tail50`** | `1.53813` | **`1.50958`** | **`-0.02855`** | — | **PASS** |
| **`step_ms`** (steady-state) | `1720.64 ms` | **`1696.28 ms`** | **`-24.36 ms`** (`-1.42%`) | `<= +10%` | **PASS** |
| **`train_elapsed_sec`** | `1034.33 s` | **`1018.42 s`** | `-15.91 s` (`-1.54%`) | — | **PASS** |
| **`packed_int4_mb`** (exact disk bytes) | `53.039 MB` (`53,039,072 B`) | **`53.482 MB`** (`53,481,568 B`) | `+0.443 MB` (`+442,496 B`) | `< 55.0 MB` | **PASS** (`1.518 MB` headroom) |
| **`infer_params`** | `77,866,208` | `78,587,936` | `+721,728` (`+0.93%`) | — | OK |
| **`peak_vram_mb`** | `11,095.1 MB` | `11,571.9 MB` | `+476.8 MB` (`+4.30%`) | `< 14,000 MB` | **PASS** |

---

## Scrutiny & Anomaly Check

1. **Silent failures / coverage:** Both arms (`ctrl`, `coordkey_ca`) completed all `600/600` steps with `error: null`, zero NaNs, and exact disk-verified Hybrid FP16/INT4 binaries (`ctrl_hybrid_int4.bin` = `53,039,072 B`, `coordkey_ca_hybrid_int4.bin` = `53,481,568 B`). Downloaded HF `eval_hf.json` and `run_hf.json` match local `D:\tmp\local-p0\i15-38\out\{eval,run}.json` byte-for-byte.
2. **Apples-to-apples sanity:** Both arms ran sequentially on the same GPU (`RTX 4080 SUPER`, BF16) with identical `seed=42`, identical `iter10_5_proxy_2k` (`2,000` train / `64` val) manifest, identical per-step weighted sampling sequence (`rng = random.Random(42)` reset per arm), identical Muon+AdamW (`lr=3e-4`, `wd=0.04`, `no_decay_1d=True`), identical WSD cosine schedule, and identical `QATInt4ControllerV2` enablement at step `420`.
3. **`raw_val_ce` vs `ema_val_ce` gap:** Both arms show `raw_val_ce < ema_val_ce` (`1.10440` vs `1.12739` for `ctrl`; `1.07805` vs `1.10725` for `coordkey_ca`). This is expected in a short 600-step from-scratch run where `EMA_DECAY=0.99` (~100-step window) lags behind rapid convergence during the final 150-step WSD decay. `coordkey_ca` beats `ctrl` decisively across **all** validation metrics (`raw_val_ce: -0.02635`, `ema_val_ce: -0.02014`, `int4_dequant_val_ce: -0.02233`) and `train_loss_tail50` (`-0.02855`).
4. **INT4 quantization resilience:** All 20 new `q_coord_proj`/`k_coord_proj` linear weights (`n_int4_tensors: 151` vs `131`) were quantized to group-32 INT4, while the tiny shared `coord_mlp` (`fc1`, `fc2`) and RMSNorm scales stayed in FP16 (`n_fp16_tensors: 240` vs `216`, `+0.051 MB`). Quantization caused **zero** degradation (`int4_dequant_val_ce - ema_val_ce = -0.00047`, better than `ctrl`'s `+0.00172`), so no FP16 override is needed for the spatial projections.
5. **Speed & VRAM:** `coordkey_ca` ran `-1.42%` faster (`1696.28 ms` vs `1720.64 ms`, within normal back-to-back local GPU jitter), confirming that asymmetric `d_qk=80, d_v=64` with explicit KV expansion incurs negligible runtime overhead under CUTLASS SDPA. Peak VRAM stayed comfortably below the 14 GB Windows WDDM ceiling (`11,571.9 MB`, `+476.8 MB` vs `ctrl`).

---

## Verdict: `PROMOTE`

`coordkey_ca` clears every `PLAN.md` `PROMOTE` gate with margin:
- `ema_val_ce`: **`-0.02014`** (gate: `<= -0.015`)
- `int4_dequant_val_ce`: **`-0.02233`** (gate: `<= -0.015`, zero INT4 degradation)
- `packed_int4_mb`: **`53.482 MB`** (`+0.443 MB`, gate: `< 55.0 MB`)
- `step_ms`: **`-1.42%`** (`-24.36 ms`, gate: `<= +10%`)

Decoupling a 16-dim RMSNorm'd 2D Fourier spatial QK subspace from the 64-dim content QK subspace in decoder cross-attention yields a clean, quantization-friendly architectural win on `edge85m + ax0_gl6`.

---

## Follow-Up Action / Successor Experiment

- **Architecture Promotion Target:** Port `FourierCoordMLP` (`66 -> 128 -> 128`, 16 log-spaced frequencies per axis, `persistent=False` frequency buffer) and `CoordKeyCrossAttention` (`d_qk = 64 + 16 = 80, d_v = 64`, RMSNorm'd spatial Q/K) into `tmp-train/mvd/uni_omr_v2.py` behind a config flag (e.g., `use_coordkey_ca: bool = False`, `coord_dim: int = 16`, `coord_mlp_dim: int = 128`) for the next from-scratch `edge85m` pretraining run (Iter 16+).
- **Production Port Note:** Pass `coord_feat` explicitly as a tensor argument through `decode_tokens` / `DecoderLayer.forward` / `precompute_cross_kv` (rather than mutating an ephemeral `layer.cross_attn._coord_feat` attribute) so gradient checkpointing, `torch.compile`, and KV-cached beam/greedy decoding stay pure and side-effect-free.
