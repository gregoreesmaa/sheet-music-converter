# `2026-11-27T07-46-i15-01-bandnet-strip-oracle-and-gauss-ptr` — Zero-Retrain BandNet Strip Oracle & StaffSeg Gaussian Cross-Attention Pointer

- **Experiment ID:** `2026-11-27T07-46-i15-01-bandnet-strip-oracle-and-gauss-ptr`
- **Short ID / Wave:** `i15-01` (Wave 13 — Pivot Strategist 1: Architecture & Decoding)
- **Status:** `QUEUED`
- **Baseline Commit SHA:** `e426c28`
- **Platform / GPU:** Colab `1× NVIDIA L4` (`24 GB` VRAM, native BF16; ~20 min, ~1.5 CU; FP16-safe on T4)
- **Model Footprint:** `0` new parameters — evaluates frozen `UniOMRv2` EMA checkpoint (`checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt`, falling back to `iter11` / `iter10_5` / `iter10`)
- **Dataset (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (14 real historical HW pieces + 10 printed safety-slice pieces = 24 pieces × `{clean, photo}` = 48 pages)

## Summary & Hypothesis
Full-page handwritten OMR suffers from severe vertical pointer drift, skipped systems, and premature `<eos>` when decoding dense multi-system manuscripts in a single unguided pass. Before investing GPU budget in retraining a dedicated BandNet / pointer architecture, this zero-retrain 3-arm diagnostic isolates **how much handwritten Note F1 is recoverable purely from vertical system localization** on our frozen `UniOMRv2` EMA checkpoint:

1. **`arm0_fullpage_baseline`:** Standard `1408px` full-page FSA + MTP-4 decode.
2. **`arm1_staffseg_gauss_ptr`:** Keep `StaffSegHead` active at inference (`return_aux=True`) → extract vertical system bands `y_1..y_N` → inject a per-`<sys>` vertical Gaussian cross-attention bias `-((y_mem - y_sys[cur])^2) / (2 * sigma_y^2)` into every `GQADecoderLayer.cross_attn` (advancing `cur` each time `<sys>` is emitted).
3. **`arm2_band_crop_oracle`:** Crop each page into individual system strips (`StaffSegHead` bands / valley peaks with `±20%` vertical padding) → decode each strip independently with FSA + MTP-4 → stitch `<sys>` (`UniOMR-BandNet` zero-retrain oracle).

**Hypothesis:** If `arm1` or `arm2` lifts `hw_clean` / `hw_photo` Strict Note F1 by `>= +5.0 pp` with `<= 0.5 pp` printed regression, vertical system localization is confirmed as a dominant bottleneck — justifying immediate promotion of the inference-time pointer (`arm1`) or BandNet strip training (`arm2`).

## Quick Links
- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Queue Entry (Wave 13): [queue_entry.json](./queue_entry.json)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T07-46-i15-01-bandnet-strip-oracle-and-gauss-ptr.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-27T07-46-i15-01-bandnet-strip-oracle-and-gauss-ptr.json`
