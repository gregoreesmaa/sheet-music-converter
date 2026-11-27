# PLAN — `2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora` (`i15-05`)

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, native BF16) via `experiments/overnight_queue/run_queue_colab.py` (also Modal L4 compatible; FP16 fallback on cc < 8).
- **Estimated Wall-Clock & Cost:** ~75 min (~5.5 CU).
  - Baseline `arm_0_base_1408` + `arm_a_spx13_band_tiling` (zero-train inference on 14 HW + `beach-prayer` + printed safety slice): ~15 min.
  - `arm_b_hw_lora_r16` (`1,500` steps rank-16 LoRA training on HW `s_px=13 px` bands + post-train eval): ~60 min.
- **Baseline Commit SHA:** `e426c28`.

## 2. Identifiers for Follow-Up Queries
- **Queue Entry:** `experiments/2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora/queue_entry.json` (`id: "2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora"`, `short_id: "i15-05"`, `wave: 13`, `provider: "colab_l4"`, `gpu: "L4"`).
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora.json`
  - Run Ledger: `runs/2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora.json`
  - HW-LoRA Checkpoint: `checkpoints/2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora/hw_lora_r16_final.pt`

## 3. Data & Frozen Base Checkpoint
- **Frozen Base Checkpoint** (loaded via `eval_uni_suite31.load_uni_checkpoint`, hard `assert len(missing_keys) == 0`, canonical `[0, 1]` input norm):
  1. `checkpoints/iter14_v2a_best/uni_omr_v2_ema_best.pt`
  2. `checkpoints/iter13_v2_best/uni_omr_v2_ema_best.pt`
  3. `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt`
  4. `checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt`
  5. `checkpoints/iter10_5_v2_best/uni_omr_v2_ema_best.pt`
- **Eval Dataset:** `datasets/suite31_hw_eval.tar` (100% permissive):
  - All 14 `suite31_hw` handwritten pieces (`clean` + `photo` = 28 pages), split into **`hw_landscape`** (`W > H`, including the landscape BL quartet manuscripts that suffer the `~4.5 px` Nyquist cliff) and **`hw_portrait`**.
  - **`beach-prayer`** (`clean` + `photo` = 2 pages, dense printed/archival edge case).
  - **Printed safety slice:** first 8 printed `suite31_hw` pieces (`clean` + `photo` = 16 pages) to verify `0.00 pp` printed regression.
- **LoRA Training Dataset (Arm B):** `datasets/iter10_5_proxy_2k.tar` (100% permissive), filtered strictly to real + simulated handwritten pages, cropped/rescaled into canonical `s_px = 13.0 px` 1–2 system bands with corresponding `<sys>`-aligned `bekern` targets.

## 4. Experimental Protocol (Pivots 1 & 3A)
- **Control (`arm_0_base_1408`):** Standard single-pass `max_dim = 1408` (`max_len = 3584`) on the exact evaluation slice, with a **2-page fail-fast sanity gate** on printed clean pages (`dec_ok > 0` and `mean_note_f1 >= 0.05`).
- **Arm A (`arm_a_spx13_band_tiling`, Pivot 1 — Zero-Train):**
  1. Estimate native staff-space `s_px` via calibrated lag-0-lobe-bypassed autocorrelation on the high-pass vertical ink projection.
  2. Compute projected `s_px_1408 = s_px_native * min(1.0, 1408 / max(H, W))`.
  3. **Bit-identical passthrough gate:** If `s_px_1408 >= 10.0 px`, run `n_bands = 1` single-pass at `1408` with zero cropping/rescaling (exact parity with `arm_0_base_1408`, guaranteeing `0.00 pp` printed regression).
  4. **Nyquist rescue path (`s_px_1408 < 10.0 px`):** Apply conservative dark-border crop (trimming outer black scan/camera borders up to 15% per side without clipping notation), re-estimate `s_px_native` and inter-system ink valleys, split vertically into 1–2 system bands (`overlap = 3.5%`), rescale each band to canonical `s_px = 13.0 px` (capped at `BAND_MAX_WIDTH = 1664`, `BAND_MAX_HEIGHT = 1280`), decode each band (`max_len = 1792`), and stitch via `<sys>` with boundary deduplication and header carry-forward.
- **Arm B (`arm_b_hw_lora_r16`, Pivot 3A — Isolated HW Capacity Inside `< 55 MB`):**
  1. Freeze 100% of base model parameters (`requires_grad = False`).
  2. Attach `LoRALinear(r=16, alpha=32.0, dropout=0.05)` to every `nn.Linear` inside `model.encoder.global_layers` and `model.dec_layers` (self-attn, cross-attn, and SwiGLU FFN). Wrapper sets `_qat_enabled = True` when active so `GroupedQueryAttention` does not bypass LoRA via weight-concatenation shortcuts, and supports `set_lora_enabled(False)` for domain-gated `0.00 pp` printed passthrough.
  3. Log exact LoRA parameter count and INT4-packed byte size (group-32 scales) verifying `~1.29 MB` on `edge85m` (`base_int4 + lora_int4 < 55.0 MB`).
  4. Train `1,500` optimizer steps (`grad_accum = 4`, `lr = 2e-4` WSD cosine cooldown, `warmup = 100`, `weight_decay = 0.01`, `EMA_DECAY = 0.99` every step, BF16) strictly on real+sim HW `s_px = 13 px` 1–2 system bands.
  5. Evaluate `base + HW-LoRA` (both RAW and EMA) on all 14 `suite31_hw` HW pieces under `spx13` band tiling, plus domain-gated printed safety slice (`lora_enabled = False` → guaranteed `0.00 pp` printed delta).

## 5. Exact Launch Command
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\overnight_queue\run_queue_colab.py --only 2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora
```

## 6. 5-Minute Checkpointing & Resume Plan
- Atomic checkpoint save (`latest_lora.pt.tmp` → `os.replace`) every `5.0` minutes during Arm B training on `/vol/ckpts/2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora/arm_b_hw_lora_r16/`, resuming automatically from `latest_lora.pt` if preempted.
- Completed arm JSONs (`arm_0_base_1408`, `arm_a_spx13_band_tiling`, `arm_b_hw_lora_r16`) are atomically cached (`arm_result.json.tmp` → `os.replace`) so finished arms are never recomputed.
- Final LoRA adapter, `evals/<exp_id>.json`, and `runs/<exp_id>.json` are uploaded to HF on completion.

## 7. Evaluation Gate & Decision Tree
- **Arm A (`arm_a_spx13_band_tiling`) Gate:**
  - `hw_landscape` Note F1 `>= +5.0 pp` (`+0.0500`) over `arm_0_base_1408` AND `0.00 pp` printed safety regression (`printed_delta == 0.0000`).
- **Arm B (`arm_b_hw_lora_r16`) Gate:**
  - `hw_clean` and/or `hw_photo` Note F1 `>= +4.0 pp` (`+0.0400`) over `arm_a_spx13_band_tiling`, `0.00 pp` printed regression via domain gating, AND `edge85m_total_int4_mb < 55.0 MB`.
- **PROMOTE:** Both Arm A and Arm B pass their gates → wire canonical `s_px=13` band tiling + domain-gated rank-16 HW-LoRA into Iter 15 (`edge85m`) inference & WebGPU export pipeline.
- **ITERATE:** Arm A passes (`hw_landscape >= +5.0 pp`) but Arm B gains `< +4.0 pp` (or vice versa: Arm B wins on bands while zero-train Arm A suffers `<sys>` stitching mismatch) → spawn `i15-05b` with band-boundary continuation tokens / expanded real-HW band dataset (`1,466` archival pages) or rank-32 LoRA.
- **DISCARD:** Both arms gain `< +1.5 pp` on `hw_landscape` / `hw_clean` despite verified `s_px = 13 px` rendering → abandon band-tiling + LoRA and invest in native high-resolution full-page training (`max_dim=1792` + `edge85m` foveated routing).
