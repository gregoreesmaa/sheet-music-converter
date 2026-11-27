# PLAN — `2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA A100` (or Local `RTX 4080 SUPER` 16 GB, native BF16).
- **Estimated Wall-Clock:** ~6–10 min on A100 / 4080 SUPER (when loading Iter 14 checkpoint; ~18 min if fallback short-train runs).
- **Baseline Commit SHA:** `6055c49`.
- **Self-Containment:** All QVLA sensitivity ranking, mixed-bit INT4/FP16 binary packing, and QuantVLA per-head attention temperature calibration live strictly inside `run_colab.py`; `tmp-train/mvd/` is untouched.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant.json`
  - Run Ledger: `runs/2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant.json`
  - Partial Arm Sync: `checkpoints/2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant/{uniform_int4,qvla_mixed_int4}_arm_result.json`

## 3. Data, Checkpoint & 2-Arm Protocol
- **Weights:** Loads `checkpoints/iter14-v2a-final/uni_omr_v2_ema_latest.pt` (fallback `checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt`; if neither is available or vocab/arch mismatches, short-trains a 300-step `edge85m` proxy on `iter10_5_proxy_2k.tar`).
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`32` calibration pages from `proxy_train.jsonl`, `64` held-out validation pages from `proxy_val.jsonl`, permissive licenses only, zero `CC-BY-NC`, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`).
- **Windows RTX 4080 SUPER Safety:** Disables `torch.backends.cudnn.enabled` and forces CUTLASS `EFFICIENT_ATTENTION` on `win32`.
- **Arm 0 (`uniform_int4`):** Standard uniform symmetric group-32 INT4 PTQ over encoder/decoder/lm_head `nn.Linear` layers + exact packed binary serialization + dequantized validation CE.
- **Arm 1 (`qvla_mixed_int4`):**
  1. Compute channel-wise output sensitivity `S[layer, out_c] = E[sum_j |grad_W * W|]` over 32 calibration pages using `5×`-weighted CE on structural/header/pitch tokens (`clef-*`, `key-*`, `time-*`, `bar*`, `<sys>`, `staff-*`, `voice-*`, `A`..`G`, `a-2`..`a2`, octaves `0`..`9`).
  2. Keep the top `5%` most sensitive output channels in per-channel INT8 (255 levels + FP16 scale/channel); quantize the remaining `95%` to group-32 INT4. Write exact packed binary (INT4 nibbles + FP16 scales + INT8 kept channels + channel bitmask) to disk and measure exact `packed_int4_mb` (`< 55.0 MB`).
  3. Calibrate per-head attention logit scales (`QuantVLA`, fused into `q_norm.weight` per head) over 32 calibration pages (40 Adam steps, token-weighted KL distillation to FP16 teacher).

## 4. Verification & Launch Commands
CPU smoke test (zero GPU):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant\run_colab.py --smoke-test
```
Colab A100 / Local 4080 SUPER run:
```bash
python3 -u experiments/2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant/run_colab.py
```

## 5. Checkpointing & Remote HF Auto-Resume Plan
- **1-Arm HF Partial Sync & Startup Auto-Resume:** On startup, checks HF Private Hub for `evals/<EXP_ID>.json` and `checkpoints/<EXP_ID>/{uniform_int4,qvla_mixed_int4}_arm_result.json` and skips completed arms. Uploads Arm 0 (`uniform_int4`) immediately before starting Arm 1.
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600), `HF_TOKEN` env, or `~/.cache/huggingface/token`, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** `int4_dequant_val_ce`, `dequant_ce_delta` (`int4 - fp16`), `struct_pitch_val_ce`, and exact serialized `packed_int4_mb` (`< 55.0 MB` hard budget).
- **PROMOTE:** `qvla_mixed_int4` recovers `>= 30%` of the uniform INT4 CE degradation (or `>= -0.010` on `int4_dequant_val_ce`) while staying `< 55.0 MB` packed (`<= +3.5 MB` over uniform INT4) → wire QVLA mixed-bit + QuantVLA head-scale export into `export_uni_v2_onnx.py`.
- **ITERATE:** Recovers `15–30%` of INT4 degradation or exceeds `55.0 MB` → sweep `2%` vs `5%` FP16 channel budget or INT8 instead of FP16 for sensitive channels.
- **DISCARD:** `< 15%` degradation recovery and `< -0.005` `int4_dequant_val_ce` gain → keep uniform group-32 QAT-INT4.
