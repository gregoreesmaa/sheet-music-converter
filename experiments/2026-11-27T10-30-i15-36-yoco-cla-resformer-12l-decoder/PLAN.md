# PLAN — `2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA H100` (`80 GB` VRAM, native BF16; Modal L4/A100 compatible).
- **Estimated Wall-Clock:** ~10–12 min total (`600` steps × 2 arms).
- **Baseline Commit SHA:** `cfc94aa`.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder.json`
  - Run Ledger: `runs/2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder.json`

## 3. Data & Training Protocol
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val, 100% permissive), `2×` sampling weight on dense multi-staff (`>= 4` systems) and handwritten pages.
- **Controlled 2-Arm Setup (`600` steps/arm, `seed=42`, BF16, `grad_accum=4`, `lr=3e-4`, `weight_decay=0.04`, WSD cosine cooldown `warmup=60`, `decay_frac=0.25`, `EMA_DECAY=0.99`, `QATInt4ControllerV2` enabled at `step >= 420`):**
  - **Arm 0 (`ctrl`):** `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, n_dec_layers=10, ffn_mult=4.0, grad_checkpoint=True)` — 10L decoder (`d_ff=2048`), independent GQA-2 self-KV & cross-KV.
  - **Arm 1 (`yoco12`):** Self-contained `YocoClaResFormerNet` inside `run_colab.py` on `edge85m + ax0_gl6`:
    - 12L decoder (`d_ff=1664`).
    - **YOCO-3Bank Cross-KV (`n_kv=4`):** 3 cross-KV producer banks at layers `0` (DeepStack `mem_post_merger`, shared by `0..2`), `3` (DeepStack `mem_global_mid`, shared by `3..6`), and `7` (`mem_final`, shared by `7..11`). Consumer layers omit `k_proj`/`v_proj`/`k_norm`.
    - **CLA-2:1 Self-KV (`n_kv=2`):** 6 self-KV producer banks at even layers `0, 2, 4, 6, 8, 10`, shared with paired odd layers `1, 3, 5, 7, 9, 11` (matching RoPE/NoPE per pair).
    - **ResFormer `v_res`:** `v_l = (1 - sigmoid(a)) * v_l + sigmoid(a) * v_0` on self-KV producer layers `l in {2, 4, 6, 8, 10}` and cross-KV bank producers `l in {3, 7}` (`a` initialized to `-2.0`).
- **Metrics:** `ema_val_ce`, `raw_val_ce`, `int4_dequant_val_ce`, exact `kv_cache_mb` (FP16 at `L_mem=3072, T=2048`), `step_ms`, `vram_gb`, and `packed_int4_mb` (`< 55.0 MB` hard gate).

## 4. Exact Launch Command
```bash
python3 experiments/2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder/run_colab.py
```
Local CPU smoke test (no HF token needed):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder\run_colab.py --smoke-test
```

## 5. 5-Minute Checkpointing & Resume Plan
- Atomic checkpoint save (`latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm.
- Completed arm results cached to `arm_result.json` so preemption never repeats finished arms.

## 6. Evaluation Gate & Decision Tree
- **PROMOTE:** `yoco12` matches or beats `ctrl` on `ema_val_ce` (`delta <= +0.005`, ideally `<= -0.015`), reduces `kv_cache_mb` by `>= 35%`, stays `< 55.0 MB` packed INT4, and maintains `step_ms` within `+10%` of `ctrl` → port into `uni_omr_v2.py` for Iter 16 flagship pretraining.
- **ITERATE:** `yoco12` improves `ema_val_ce` or KV cache strongly but shows `> 0.015` INT4 drop or `> 10%` latency overhead → ablate CLA-only vs YOCO-only or tune `d_ff`.
- **DISCARD:** `yoco12` degrades `ema_val_ce` by `> +0.020` vs `ctrl` → keep standard 10L GQA-2 decoder.
