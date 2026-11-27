# PLAN — `2026-11-27T07-46-i15-01-bandnet-strip-oracle-and-gauss-ptr` (`i15-01`)

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, native `torch.bfloat16`) via `experiments/overnight_queue/run_queue_colab.py` (also FP16-safe on T4).
- **Estimated Wall-Clock & Cost:** ~20 min (~1.5 CU). Pure inference (zero training) over 48 pages × 3 arms = 144 page decodes.
- **Baseline Commit SHA:** `e426c28`.

## 2. Identifiers for Follow-Up Queries
- **Queue Entry:** [queue_entry.json](./queue_entry.json) (to be merged into `experiments/overnight_queue/queue.json` as Wave 13, `priority: 1`, `provider: "colab_l4"`, `status: "QUEUED"`).
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T07-46-i15-01-bandnet-strip-oracle-and-gauss-ptr.json`
  - Run Ledger: `runs/2026-11-27T07-46-i15-01-bandnet-strip-oracle-and-gauss-ptr.json`

## 3. Data & Checkpoint
- **Eval Dataset (100% permissive):** `datasets/suite31_hw_eval.tar` (`suite31/suite31_hw_manifest.json`).
  - **Slice (48 pages):** All **14 real historical handwritten pieces** (`hw_clean`, `hw_photo` = 28 pages) + the **first 10 printed pieces** in manifest order as a regression safety slice (`printed_clean`, `printed_photo` = 20 pages).
- **Frozen Checkpoint (loaded via `eval_uni_suite31.load_uni_checkpoint`, `len(missing_keys) == 0` asserted):**
  1. `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt`
  2. `checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt`
  3. `checkpoints/iter10_5_v2_best/uni_omr_v2_ema_best.pt`
  4. `checkpoints/iter10_v2_best/uni_omr_v2_ema_best.pt`

## 4. Experimental Arms (Zero Retrain)
1. **`arm0_fullpage_baseline`:**
   - Canonical `1408px` resize + `[0, 1]` CHW tensor.
   - Standard full-page FSA grammar-constrained MTP-4 self-speculative decode (`eval_uni_suite31.generate_mtp_spec(..., max_len=3584, use_grammar=True, strict_grammar=False)`).
2. **`arm1_staffseg_gauss_ptr`:**
   - Runs `model.encode_image(inp_t, return_aux=True)` to obtain `aux_enc["staff_seg_logits"]` (`(1, 1, H/4, W/4)`).
   - Extracts normalized vertical system bands `(y0, y1, y_center)` from the width-averaged sigmoid `StaffSegHead` profile (with automatic ink-projection fallback if `staff_seg_head` is absent or yields 0 bands).
   - Computes normalized vertical coordinate `y_mem` for every routed memory token `(kept_idx // W_p) / max(1, H_p - 1)` (background summary tokens receive `0.0` bias).
   - Injects additive Gaussian cross-attention bias `bias = -((y_mem - y_sys[cur])^2) / (2 * sigma_y^2)` (with `sigma_y = clamp(0.75 * band_h, 0.06, 0.18)`, bias clamped to `[-6.0, 0.0]`) into every `GQADecoderLayer.cross_attn` via a scoped wrapper, advancing `cur += 1` whenever `<sys>` is emitted (and relaxing bias to `0.0` if `cur >= len(y_sys)`).
   - Decodes with KV-cached FSA grammar-constrained decoding (`max_len=3584`).
3. **`arm2_band_crop_oracle`:**
   - Uses the exact same detected system bands (`StaffSegHead` / valley fallback), crops each band with `±20%` vertical padding (`pad = round(0.20 * band_h_px)`), resizes each single-system strip to `max_dim=1280`, decodes each strip independently via FSA + MTP-4 (`max_len=1024`), and stitches strips with `<sys>` (carrying forward the first strip's clef/key/time header if a downstream strip omits it).

## 5. Exact Launch Command
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\overnight_queue\run_queue_colab.py --only 2026-11-27T07-46-i15-01-bandnet-strip-oracle-and-gauss-ptr
```

## 6. Checkpointing, Resume & Sanity Gate
- **Per-Arm Atomic Caching:** Each arm writes `arm_result.json.tmp` → `os.replace` under `/vol/ckpts/<exp_id>/<arm>/` (or `/content/vol/...`) so a preempted run resumes at the next unfinished arm.
- **5-Minute Progress Heartbeat:** Writes `progress.json` every `5.0` minutes (`SAVE_EVERY_MINUTES = 5.0`).
- **2-Page Fail-Fast Sanity Gate:** After the first 2 `printed_clean` pages of `arm0_fullpage_baseline`, asserts `dec_ok > 0` and `mean_note_f1 >= 0.05` (aborts in `< 30 s` if checkpoint loading or preprocessing is broken).

## 7. Evaluation Gate & Decision Tree
- **PROMOTE:** Either `arm1_staffseg_gauss_ptr` or `arm2_band_crop_oracle` improves `hw_clean` or `hw_photo` Strict Note F1 by **`>= +5.0 pp` (`>= +0.0500`)** over `arm0_fullpage_baseline` while keeping `printed_clean` and `printed_photo` drop **`<= 0.5 pp` (`>= -0.0050`)**:
  - If `arm1` wins → wire `StaffSegHead` Gaussian cross-attention pointer directly into `eval_uni_suite31.py` / inference decoder (0% retraining cost!).
  - If `arm2` wins → launch Wave-13/14 `UniOMR-BandNet` strip-adapted fine-tuning + learned band detector.
- **ITERATE:** `arm1` or `arm2` yields `+1.5 pp` to `+5.0 pp` on HW (or improves `sys_count` alignment / reduces truncation) but suffers from noisy `StaffSegHead` band boundaries, overly tight `sigma_y`, or strip-crop domain shift (printed drop `> 0.5 pp`) → spawn `i15-01b` tuning `sigma_y` / layer-selective bias (early vs late decoder layers) or strip-augmented fine-tuning.
- **DISCARD:** Both `arm1` and `arm2` gain `< +1.5 pp` on `hw_clean` & `hw_photo` (or degrade HW Note F1) despite accurate band detection → vertical pointer drift is not the primary zero-retrain bottleneck; pivot to data/representation interventions.
