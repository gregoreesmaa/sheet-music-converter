# PLAN — `2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA H100` (`80 GB` VRAM, native BF16; Modal L4/A100 compatible).
- **Estimated Wall-Clock:** ~10–12 min total (`600` steps × 2 arms).
- **Baseline Commit SHA:** `cfc94aa`.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder.json`
  - Run Ledger: `runs/2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder.json`

## 3. Data & Training Protocol
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val, 100% permissive), `2×` sampling weight on dense multi-staff (`>= 4` systems) and handwritten pages.
- **Controlled 2-Arm Setup (`600` steps/arm, `seed=42`, BF16, `grad_accum=4`, `lr=3e-4`, `weight_decay=0.04`, WSD cosine cooldown `warmup=60`, `decay_frac=0.25`, `EMA_DECAY=0.99`, `QATInt4ControllerV2` enabled at `step >= 420`):**
  - **Arm 0 (`ctrl`):** `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=True)` — flat `mem_final` (`L_k` tokens).
  - **Arm 1 (`sysslot_mem`):** Self-contained `SysSlotUniOMRv2Net` inside `run_colab.py` on `edge85m + ax0_gl6`:
    - `SystemSlotAttention(d_model=512, n_slots=16, d_attn=256, n_iters=2, ffn_hidden=512)`:
      - 16 learned slot seeds + fixed vertical prior centers `y_centers = linspace(0.03, 0.97, 16)` with learnable vertical sharpness `y_beta`.
      - 2 iterations of competitive Slot-Attention (softmax over `S=16` slots `dim=1`, normalized weighted mean over `L_k` tokens, linear projection back to `512`, residual + `SwiGLU(512, 512)`).
      - Output slots `(B, 16, 512)` appended to `mem_final` → `(B, L_k + 16, 512)`.
      - DeepStack layers 0 & 1 fuse `mem_post_merger` / `mem_global_mid` into the spatial prefix `[:, :L_k, :]` and preserve the 16 appended system slots unchanged.
- **Metrics:** `ema_val_ce`, `raw_val_ce`, `int4_dequant_val_ce`, `step_ms`, `vram_gb`, and `packed_int4_mb` (`< 55.0 MB` hard gate).

## 4. Exact Launch Command
```bash
python3 experiments/2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder/run_colab.py
```
Local CPU smoke test (no HF token needed):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder\run_colab.py --smoke-test
```

## 5. 5-Minute Checkpointing & Resume Plan
- Atomic checkpoint save (`latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm.
- Completed arm results cached to `arm_result.json` so preemption never repeats finished arms.

## 6. Evaluation Gate & Decision Tree
- **PROMOTE:** `sysslot_mem` beats `ctrl` by `<= -0.015` `ema_val_ce` while staying `< 55.0 MB` packed INT4 and `< +5%` `step_ms` → port `SystemSlotAttention` into `uni_omr_v2.py` for Iter 16 flagship pretraining.
- **ITERATE:** `sysslot_mem` helps slightly (`-0.005` to `-0.015`) → test `n_iters=3` or supervise slot attention maps with vertical system bands (`SysYHead` / `StaffSegHead`).
- **DISCARD:** `sysslot_mem` shows no `ema_val_ce` gain (`delta > -0.005`) → keep flat `mem_final`.
