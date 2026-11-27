# PLAN — `2026-11-27T10-30-i15-31-hrnet-music-dualres-s4`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA H100` (`80 GB` VRAM, native BF16).
- **Estimated Wall-Clock:** ~12–14 min total (2 arms × 600 steps + val CE + exact Hybrid INT4 packing & dequant eval).
- **Baseline Commit SHA:** `cfc94aa`.
- **Self-Containment:** All experimental `HRNetMusicStage4` / `HRNetMusicBackbone` modules live strictly inside `run_colab.py`; `tmp-train/mvd/uni_omr_v2.py` is untouched.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T10-30-i15-31-hrnet-music-dualres-s4`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T10-30-i15-31-hrnet-music-dualres-s4.json`
  - Run Ledger: `runs/2026-11-27T10-30-i15-31-hrnet-music-dualres-s4.json`

## 3. Data & Controlled 2-Arm Protocol
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, `2×` dense multi-staff & HW sampling weight, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`).
- **Shared Hyperparameters:** `600` steps/arm, `seed=42`, BF16 autocast, `batch_size=2`, `grad_accum=4` (eff. batch 8), Muon + AdamW (`lr=3e-4`, `weight_decay=0.04`, `no_decay_1d=True`), WSD cosine schedule (`warmup=60`, `decay_frac=0.25`, `min_lr_ratio=0.05`), EMA (`decay=0.99` every step), `QATInt4ControllerV2` enabled at step `420` (final 30%), `grad_checkpoint=True`.
- **Arm 0 (`ctrl`):** `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=True)` with stock `AnisotropicMusicBackbone`.
- **Arm 1 (`hrnet_music_s4`):** Same `edge85m + ax0_gl6` config, replacing serial `stage4 (448ch)` + `p2_to_p4` + `proj` with parallel dual-branch `/4 (96ch) || /8 (320ch)` bidirectional-exchange `stage4` (3 blocks, `PixelUnshuffle(2)` `/4 -> /8` + bilinear `/8 -> /4` exchange, final `704 -> 512` `1×1` projection). Preserves `(out, p2, p3)` pyramid interface for `StaffSegHead`/`StrokeHogHead`.

## 4. Verification & Launch Commands
CPU smoke test (local, zero GPU):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T10-30-i15-31-hrnet-music-dualres-s4\run_colab.py --smoke-test
```
Colab H100 run:
```bash
python3 -u experiments/2026-11-27T10-30-i15-31-hrnet-music-dualres-s4/run_colab.py
```

## 5. Checkpointing & Resume Plan
- Atomic checkpoint save (`latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm under `/content/i15_31/ckpts/<arm>/`.
- Per-arm `arm_result.json` cache skips finished arms on resume.
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600) or env, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** `ema_val_ce`, `int4_dequant_val_ce`, `step_ms`, and exact serialized `packed_int4_mb` (expect `~ -1.94 MB` vs `ctrl`, hard `< 55.0 MB` budget).
- **PROMOTE:** `hrnet_music_s4` achieves `ema_val_ce <= ctrl - 0.01` (or iso-CE within `+0.005` while saving `>= 1.5 MB` FP16 backbone budget), `int4_dequant_val_ce` gap `<= 0.02`, and `step_ms` overhead `<= +15%` → port `HRNetMusicStage4` into `uni_omr_v2.py` for next from-scratch pretraining run.
- **ITERATE:** Saves `~1.94 MB` and improves `ema_val_ce`, but `step_ms` overhead `> +15%` → test 2-block exchange or lightweight `ch_hi=64` variant.
- **DISCARD:** `ema_val_ce` regresses by `> +0.015` vs `ctrl` → keep stock `AnisotropicMusicBackbone`.
