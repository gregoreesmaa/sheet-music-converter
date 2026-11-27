# OUTCOME — `2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder`

- **Status:** `COMPLETED`
- **Verdict:** **`DISCARD`**
- **Short ID / Wave:** `i15-39` (Wave 13 / Iter 15 — Weight-Incompatible Architecture Proxy)
- **Started / Finished (UTC):** `2026-11-27T13:29:05Z` → `2026-11-27T13:45:58Z` (`1012.61 s` / `16.9 min` total wall-clock)
- **Platform / GPU:** Local `NVIDIA GeForce RTX 4080 SUPER` (`16 GB` VRAM, native BF16; `600` steps/arm, `seed=42`, `grad_accum=4`, `EMA=0.99`, `QATInt4ControllerV2` enabled at `step >= 420`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder.json` (mirrored locally at [eval_hf.json](./eval_hf.json))
  - Run Ledger: `runs/2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder.json` (mirrored locally at [run_hf.json](./run_hf.json))

---

## Measured Results (`ctrl` vs `sysslot_mem`, 600 steps/arm, `iter10_5_proxy_2k`)

| Metric | `ctrl` (`edge85m + ax0_gl6`, flat `L_k`) | `sysslot_mem` (`SysSlotUniOMRv2Net`, `L_k + 16`) | Delta (`sysslot_mem - ctrl`) | `PLAN.md` Gate | Status |
|---|---:|---:|---:|---|---|
| **`ema_val_ce`** (primary) | **`1.19410`** | `1.20454` | **`+0.01044`** (`+0.87%`) | `<= -0.015` (`DISCARD` if `> -0.005`) | **FAIL** (regression) |
| **`raw_val_ce`** | **`1.17814`** | `1.18773` | **`+0.00959`** (`+0.81%`) | — | **FAIL** (regression) |
| **`int4_dequant_val_ce`** | **`1.19424`** | `1.20455` | **`+0.01031`** (`+0.86%`) | — | **FAIL** (regression) |
| **INT4 quant drop** (`int4 - ema`) | `+0.00014` | **`+0.00001`** | `-0.00013` | `<= +0.015` | **PASS** (zero drop) |
| **`train_loss_tail50`** | **`1.61609`** | `1.63085` | **`+0.01476`** (`+0.91%`) | — | **FAIL** (slower fit) |
| **`step_ms`** (steady train avg) | **`775.49 ms`** | `806.98 ms` | `+31.49 ms` (`+4.06%`, `1.041×`) | `< +5%` (`< 1.05×`) | **PASS** (slower) |
| **`elapsed_secs`** (arm train) | **`496.13 s`** | `515.45 s` | `+19.32 s` (`+3.89%`) | — | Slower |
| **`peak_vram_mb`** (`vram_gb`) | **`6,924.3 MiB`** (`6.762 GB`) | `7,016.4 MiB` (`6.852 GB`) | `+92.2 MiB` (`+0.090 GB`, `+1.33%`) | `< 14,000 MiB` | **PASS** |
| **`packed_int4_mb`** (disk-verified) | **`53.039 MB`** (`53,039,072 B`) | `53.798 MB` (`53,797,858 B`) | `+0.759 MB` (`+1.43%`) | `< 55.0 MB` hard gate | **PASS** (`1.202 MB` headroom) |
| **`infer_params`** (`total_params`) | `77,866,208` (`79,224,715`) | `79,187,681` (`80,546,188`) | `+1,321,473` (`+1.70%`) | — | `+1.32M` slot params |
| **`gate_pass`** (`PLAN.md` `PROMOTE`) | — | **FALSE** | — | All `PLAN.md` gates | **FAIL → `DISCARD`** |

---

## Scrutiny & Anomaly Check

1. **Silent failures / coverage & reproducibility:** Both arms (`ctrl`, `sysslot_mem`) completed all `600/600` steps with `status: "completed"`, zero NaNs, and disk-verified Hybrid FP16/INT4 binaries (`ctrl`: `53,039,072 B`, `sysslot_mem`: `53,797,858 B`). Both HF artifacts (`eval_hf.json`, `run_hf.json`) downloaded cleanly. Notably, `ctrl` reproduced the sibling `i15-36` `ctrl` baseline almost bit-for-bit (`ema_val_ce` `1.19410` vs `1.19388`, `raw_val_ce` `1.17814` vs `1.17788`, `train_loss_tail50` `1.61609` vs `1.61626`), confirming a clean, uncorrupted measurement.
2. **Anomaly #1 — `param_summary` vs disk-verified `packed_int4_mb` discrepancy (`55.682 MB` vs `53.798 MB`):**
   - `UniOMRv2Net.param_summary()` does not register the subclass attribute `self.sys_slot_attn` in its INT4 roots, so it counted all `+1,321,473` `SystemSlotAttention` parameters as `fp16_keep_params` (`6,427,360` → `7,748,833`), inflating `param_summary.hybrid_fp16_int4_with_scales_mb` to `55.682 MB`.
   - `_pack_and_verify_hybrid_int4()` explicitly added `infer_model.sys_slot_attn` to `int4_roots` (`run_colab.py` line 532), producing the true disk-verified binary of **`53.798 MB`** (`int4_weight_mb`: `36.375` vs `35.719`, `int4_scales_mb`: `4.547` vs `4.465`, `fp16_keep_mb`: `12.876` vs `12.855`). Quantization was lossless (`int4 - ema = +0.00001`).
3. **Anomaly #2 — Optimizer mode (`AdamW` on both arms instead of `HybridOptimizerV2` `muon+adamw`):**
   - `run_colab.py` line 664 used `torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.04)` on **both** arms (same as `i15-36`). Because both arms share the exact same optimizer, `seed=42`, `iter10_5_proxy_2k` data stream, and WSD + QAT-INT4 schedule, the A/B comparison is 100% apples-to-apples (`tok_dropout=0.0` on `edge85m`, so `_decode_tokens_with_slots` omitting `apply_tok_dropout` had zero effect).
4. **Root cause of the `+0.01044` `ema_val_ce` / `+0.01476` `train_loss_tail50` regression:**
   - `SystemSlotAttention` appends 16 **unsupervised, unit-RMS (`RMSNorm`-normalized)** slot tokens directly to `mem_final` (`torch.cat([mem_final, slots], dim=1)`) from step 1 with no zero-init residual gate, no modality/slot type embedding, and no auxiliary vertical-system supervision (`SysYHead`/`StaffSegHead`).
   - Early in training, these 16 randomly-initialized full-norm tokens inject uncalibrated distractors into every decoder cross-attention layer's softmax over `L_k + 16` memory keys, slowing optimization (`train_loss_tail50` worse by `+0.01476` nats) and hurting validation CE across all 3 eval modes (`raw`, `ema`, `int4`).
   - Meanwhile, `ax0_gl6`'s 6 global self-attention encoder layers already propagate page-level vertical/system context directly into the routed spatial tokens `mem_final`, making an unsupervised 16-slot sidecar both redundant and actively harmful at `+4.06%` step latency and `+0.759 MB` INT4 cost.

---

## Verdict: `DISCARD`

`sysslot_mem` **regressed on every quality metric** (`ema_val_ce` **`+0.01044`**, `raw_val_ce` **`+0.00959`**, `int4_dequant_val_ce` **`+0.01031`**, `train_loss_tail50` **`+0.01476`**), while adding `+4.06%` step latency (`1.041×`), `+92.2 MiB` peak VRAM, and `+0.759 MB` packed INT4 size. This cleanly falsifies the hypothesis that appending unsupervised competitive System Slots (`S=16`) to `mem_final` improves `edge85m + ax0_gl6` full-page OMR.

---

## Follow-Up Action / Successor Experiment

- **Action:** **DISCARD** — do **not** port `SystemSlotAttention` into `tmp-train/mvd/uni_omr_v2.py`. Keep flat routed `mem_final` (`L_k` tokens) paired with the promoted **`i15-38` `CoordKeyCrossAttention`** (which gives decoder cross-attention explicit 2D spatial Fourier keys without appending noisy summary tokens) and **`i15-36` `YocoClaResFormer`** for Iter 16 pretraining.
- **No successor experiment queued.** If hierarchical system-level memory is ever revisited post-Iter-16, it must be (a) zero-initialized / gated rather than raw `RMSNorm`-appended, and (b) explicitly supervised against ground-truth vertical system bands (`SysYHead` / `StaffSegHead`).
