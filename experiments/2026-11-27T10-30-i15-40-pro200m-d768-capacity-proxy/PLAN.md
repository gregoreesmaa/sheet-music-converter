# PLAN — `2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA H100` (`80 GB` VRAM, native BF16; Modal A100/H100 compatible).
- **Estimated Wall-Clock:** ~14 min total (`600` steps × 2 arms).
- **Baseline Commit SHA:** `cfc94aa`.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy.json`
  - Run Ledger: `runs/2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy.json`

## 3. Data & Training Protocol
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val, 100% permissive), `2×` sampling weight on dense multi-staff (`>= 4` systems) and handwritten pages.
- **Controlled 2-Arm Setup (`600` steps/arm, `seed=42`, BF16, `grad_accum=4`, `lr=3e-4`, `weight_decay=0.04`, WSD cosine cooldown `warmup=60`, `decay_frac=0.25`, `EMA_DECAY=0.99`, `QATInt4ControllerV2` enabled at `step >= 420`):**
  - **Arm 0 (`edge85m_ctrl`):** `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=True)` (`d_model=512`, `n_heads=8`, `n_kv_heads=2`, `n_dec_layers=10`, `ffn_mult=4.0`, `~78M` infer params, `~53 MB` packed INT4).
  - **Arm 1 (`pro200m`):** `UniOMRv2Config.edge85m(d_model=768, n_heads=12, n_kv_heads=4, n_axial_layers=0, n_global_layers=8, n_dec_layers=12, ffn_mult=3.5, grad_checkpoint=True)` (`~185–205M` infer params, `~115–128 MB` packed INT4).
- **Exact Serialized INT4 Verification:** Strip aux/MTP heads, pack all `QATInt4ControllerV2` target `nn.Linear` weights into real 4-bit nibbles + FP16 group-32 scales + FP16 remaining tensors, write binary file to disk, and measure `int4_dequant_val_ce` on the dequantized model.
- **Metrics:** `ema_val_ce`, `raw_val_ce`, `int4_dequant_val_ce`, `step_ms`, `vram_gb`, and `packed_int4_mb`.

## 4. Exact Launch Command
```bash
python3 experiments/2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy/run_colab.py
```
Local CPU smoke test (no HF token needed):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy\run_colab.py --smoke-test
```

## 5. 5-Minute Checkpointing & Resume Plan
- Atomic checkpoint save (`latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm.
- Completed arm results cached to `arm_result.json` so preemption never repeats finished arms.

## 6. Evaluation Gate & Decision Tree
- **PROMOTE:** `pro200m` beats `edge85m_ctrl` by `<= -0.04` `ema_val_ce`, QAT-INT4 degradation (`int4_dequant_val_ce - ema_val_ce`) is `<= 0.015`, AND exact `packed_int4_mb <= 135.0 MB` → add `UniOMRv2Config.pro200m()` preset to `uni_omr_v2.py` as our flagship Pro-tier & KD-teacher architecture.
- **ITERATE:** `pro200m` beats `edge85m_ctrl` by `-0.02` to `-0.04` or needs lower LR (`2e-4`) at `d=768` → run LR/warmup tuning proxy.
- **DISCARD:** `pro200m` shows `< -0.02` `ema_val_ce` gain at 600 steps → focus compute on `edge85m` + data scaling.
