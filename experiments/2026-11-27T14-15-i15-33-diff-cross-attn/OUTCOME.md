# OUTCOME — `2026-11-27T14-15-i15-33-diff-cross-attn` (`i15-33`)

- **Status:** `COMPLETED` (`rc=0`, 2/2 arms completed, zero errors)
- **Completed At:** `2026-11-27T14:21:37Z` (started `2026-11-27T13:46:54Z`)
- **Platform / GPU:** Local `NVIDIA GeForce RTX 4080 SUPER` (16 GB, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`)
- **Elapsed Time:** `2,082.45 s` (`34.7 min` total across 2 arms × 600 steps)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - `evals/2026-11-27T14-15-i15-33-diff-cross-attn.json` → local copy `eval_hf.json`
  - `runs/2026-11-27T14-15-i15-33-diff-cross-attn.json` → local copy `run_hf.json`

---

## Measured Results (`ctrl` vs `diff_ca`, 600 steps, `seed=42`, verified `muon+adamw`)

| Metric | `ctrl` (stock GQA CA) | `diff_ca` (DiffCrossAttn, 10L) | Delta (`diff_ca - ctrl`) | Gate Check |
|---|---:|---:|---:|---|
| `optimizer_mode` | `muon+adamw` | `muon+adamw` | — | Verified |
| `ema_val_ce` | `1.12713` | **`1.08967`** | **`-0.03746`** | **PASS** (2.5× better than `-0.015` PROMOTE gate) |
| `raw_val_ce` | `1.10350` | **`1.06301`** | **`-0.04049`** | **PASS** |
| `int4_dequant_val_ce` | `1.12882` | **`1.08915`** | **`-0.03967`** | **PASS** (zero INT4 drop: `-0.00052` vs EMA FP16) |
| `train_loss_tail50` | `1.53757` | **`1.49598`** | **`-0.04159`** | Consistent train+val improvement |
| `step_ms` | `1583.88 ms` | **`1559.98 ms`** | **`-23.90 ms` (`-1.5%`)** | **PASS** (faster than ctrl; gate `<= +10%`) |
| `train_elapsed_sec` | `953.59 s` | `938.62 s` | `-14.97 s` | Consistent |
| `peak_vram_mb` | `10623.6 MB` | `11258.1 MB` | `+634.5 MB` (`+6.0%`) | Safe (`< 14 GB` Windows WDDM ceiling) |
| `packed_int4_mb` | `53.039 MB` (`53,039,072 B`) | `53.043 MB` (`53,042,912 B`) | **`+0.004 MB`** (`+3,840 B`) | **PASS** (`< 55.0 MB` hard budget) |
| `passes_55mb_budget` | `true` | `true` | — | **PASS** |
| `gate_pass` (all gates) | — | **`PASS`** | — | **ALL PROMOTE GATES CLEARED** |

### Learned Per-Layer `lambda` Trajectory (`diff_ca`, decoder layers 0..9)
```text
L0: 0.2646  L1: 0.5399  L2: 0.6535  L3: 0.7579  L4: 0.5840
L5: 0.8134  L6: 0.8036  L7: 0.8774  L8: 0.9336  L9: 0.9904
```
Healthy, smooth depth progression (`0.26 → 0.99`) matching Ye et al. (2024): shallower decoder layers retain broader visual context (`lambda ≈ 0.26–0.58`), while deeper layers apply stronger noise-cancellation subtraction (`lambda ≈ 0.80–0.99`) to suppress adjacent-staff / background-clutter attention leakage.

---

## Scrutiny & Anomaly Check

1. **Silent failures / partial coverage:** Both arms ran all `600` steps (`n_arms_completed=2`, `error=null`), zero NaNs, finite train/raw/EMA/INT4-dequant CE on both arms. Exact Hybrid INT4 packing serialized and verified on disk (`n_int4_tensors=131` identical in both arms; `n_fp16_tensors=216 -> 286` for the 10-layer per-half RMSNorms and scalar `lambda` vectors = `+3,840 B` / `+0.004 MB`).
2. **Apples-to-apples sanity:** Both arms used identical `edge85m + ax0_gl6`, `seed=42`, `bf16=true`, `steps_per_arm=600`, `iter10_5_proxy_2k.tar`, `QATInt4ControllerV2` at step 420, and verified `optimizer_mode == "muon+adamw"` (no silent AdamW fallback).
3. **INT4 quantization behavior:** Remarkably, `diff_ca` suffers **zero INT4 degradation** (`int4_dequant_val_ce=1.08915` vs `ema_val_ce=1.08967`, `-0.00052`), widening its lead over `ctrl` from `-0.03746` (FP16 EMA) to **`-0.03967`** (INT4 dequant). Subtracting the common-mode attention noise map naturally acts as an outlier/noise dampener under INT4 weight quantization.
4. **Speed & memory:** Splitting `Q`/`K` into two `32`-dim halves (`d_qk=32`) keeps the two SDPA calls slightly faster in aggregate than a single `64`-dim SDPA (`-1.5%` wall-clock) while keeping KV-cache shape `(B, n_kv_heads, Lk, 64)` unchanged. Peak VRAM rose by only `+634.5 MB` (`+6.0%`), well within the 14 GB local RTX 4080 SUPER budget.
5. **No anomalies found.**

---

## Verdict: **`PROMOTE`**

**Strong, unambiguous `PROMOTE`.** `DiffCrossAttention` clears every `PLAN.md` promotion gate with massive margin:
- **`-0.03746`** on `ema_val_ce` (vs `-0.015` threshold — **2.5× gate**)
- **`-0.03967`** on `int4_dequant_val_ce` (zero INT4 loss)
- **`-1.5%` faster** (`-23.90 ms/step`, vs `<= +10%` threshold)
- **`+0.004 MB`** packed INT4 size (`53.043 MB < 55.0 MB` hard budget, zero KV-cache growth)

## Follow-Up Action / Successor Experiment
- **Target integration:** Port `DiffCrossAttention` (with per-half RMSNorm, depth-initialized `lambda`, and unchanged `(B, n_kv_heads, Lk, 64)` KV-cache layout) into `tmp-train/mvd/uni_omr_v2.py` behind a config flag (`diff_cross_attn: bool = False`, enabled by default in Iter 15 from-scratch pretraining config) during the pre-Iter-15 winner consolidation pass, accompanied by CPU unit tests (`CUDA_VISIBLE_DEVICES=""`) verifying forward/backward, KV-cache equivalence, and Hybrid INT4 packing.
- **Synergy note:** Because `DiffCrossAttention` operates purely on cross-attention Q/K/V subtraction and is orthogonal to `i15-38` (`CoordKey-CA`), `i15-31` (`HRNet-Music`), and `i15-36` (`YOCO/CLA/ResFormer`), it can be cleanly composed in the Iter 15 architecture integration run.
