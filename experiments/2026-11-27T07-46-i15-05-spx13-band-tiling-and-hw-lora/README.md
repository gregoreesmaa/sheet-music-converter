# `2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora` — Canonical `s_px=13` Band Tiling + Rank-16 HW-Only LoRA

- **Experiment ID:** `2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora`
- **Short ID / Wave:** `i15-05` (Wave 13 — Pivot Strategist 5: Resolution & Edge Systems, Pivots 1 & 3A)
- **Status:** `QUEUED`
- **Baseline Commit SHA:** `e426c28`
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, BF16 native, est. ~75 min / ~5.5 CU)
- **Datasets (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 14 HW incl. landscape BL quartets + `beach-prayer` + printed safety slice)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (real+sim HW full-page pairs sliced into `s_px=13 px` 1–2 system bands for Arm B LoRA fine-tuning)

## Summary & Hypothesis
12-staff landscape British Library manuscripts and dense historical scores collapse at `max_dim=1408` because downscaling compresses the inter-line staff spacing `s_px` down to `~4.5 px` — below the Nyquist pitch-aliasing cliff where half-space step intervals (`~2.25 px`) alias inside the `/4` and `/8` backbone strides. Simultaneously, full-model fine-tuning on handwritten scores risks eroding printed accuracy and blowing past our `< 55 MB` WebGPU INT4 edge footprint.

This 2-arm experiment formalizes Pivot Strategist 5's Pivots 1 & 3A:
1. **`arm_a_spx13_band_tiling` (Zero-Train Inference):** Dark-border crop → native staff-space `s_px` estimation → if `s_px < 10 px` at `1408`, rescale to canonical `s_px = 13 px` and split vertically into 1–2 system bands (`n_bands = 1` bit-identical passthrough whenever `s_px >= 10 px`, guaranteeing `0.00 pp` printed regression) → decode & stitch `<sys>`.
2. **`arm_b_hw_lora_r16` (Domain-Gated HW-Only LoRA):** Freeze the base `UniOMRv2Net` weights, attach rank-16 LoRA (`~1.29 MB` INT4 on `edge85m`, fitting comfortably inside the `< 55 MB` edge budget) on global encoder + decoder attention/FFN `nn.Linear` layers, and train for `1,500` steps strictly on real+sim HW `s_px = 13 px` bands. Domain gating disables LoRA on printed pages for guaranteed `0.00 pp` printed regression while specializing capacity for calligraphic handwritten notation.

## Quick Links
- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Queue Entry (pending merge into `overnight_queue/queue.json`): [queue_entry.json](./queue_entry.json)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora.json`
- HF Adapter Checkpoint: `gregoreesmaa/sheet-music-converter-omr` → `checkpoints/2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora/hw_lora_r16_final.pt`
