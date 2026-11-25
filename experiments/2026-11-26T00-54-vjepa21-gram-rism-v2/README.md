# Experiment `2026-11-26T00-54-vjepa21-gram-rism-v2` — V-JEPA 2.1 + DINOv3 Gram Anchoring + Stroke-HOG SSL Pretraining on RISM HW (`UniOMR v2 small_45m`)

- **Status:** `RUNNING` (launched detached on Modal `L4`)
- **Platform:** Modal (`L4`, 24 GB VRAM, native BF16, profile `gregoreesmaa1`)
- **Baseline Commit SHA:** `9d73170`

## Summary & Hypothesis

Pretrain the `UniOMR v2 small_45m` vision encoder (`AnisotropicMusicBackbone` + `HybridAxialMergerEncoder`) from scratch on all **5,707 unlabeled real RISM historical handwritten manuscripts** (`datasets/rism_hw_5707.tar`) for `1,500` optimizer steps (`batch_size=4`, `grad_accum=2`, effective batch 8, `max_dim=1024`, BF16) using `tmp-train/mvd/pretrain_ijepa_rism_v2.py`:

1. **V-JEPA 2.1 Dense Predictive Loss + 3-Tap DeepStack Supervision** (`post_merger`, `global_mid`, `final`; masked + visible ink-weighted loss).
2. **DINOv3 Ink-Token Gram Anchoring** (`--gram-weight 1.0`) to prevent dense pitch/staff-line geometric collapse during unlabeled domain adaptation.
3. **MonkeyOCRv2 / MaskFeat Sobel-HOG Stroke Reconstruction** (`--stroke-hog-weight 0.25`) on `/4` `p2` features to preserve 1–2 px iron-gall staff/ledger/accidental strokes.
4. **MiniCPM-V 4.5 Local Ink-Patch Corruption + HTR-VT Span Masking** on student inputs while the EMA teacher sees clean archival pages.
5. **Native `torch.optim.Muon` + fused `AdamW`** for 2D Transformer encoder weights and convs/norms/predictors respectively.

**Hypothesis:** Warm-starting downstream `UniOMR v2 small_45m` supervised OMR training from this RISM-adapted SSL encoder will improve real historical handwritten OOD accuracy (`suite31_hw` `hw_clean` / `hw_photo` Strict Note F1) without degrading printed-score accuracy or suffering dense geometric collapse.

## Quick Links

- Plan, live cloud IDs, & copy-paste monitor/resume commands: [PLAN.md](PLAN.md)
- Outcome (written after completion & downstream evaluation): `OUTCOME.md` (pending)
- Launcher script: [run_modal.py](run_modal.py)
- Pretrainer script: `tmp-train/mvd/pretrain_ijepa_rism_v2.py`
- Hugging Face Hub (`gregoreesmaa/sheet-music-converter-omr`, private dataset):
  - Ledger: `runs/2026-11-26T00-54-vjepa21-gram-rism-v2.json`
  - Final checkpoint: `checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/uni_omr_v2_ijepa_last.pt`
  - Best EMA checkpoint: `checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/uni_omr_v2_ijepa_best.pt`
  - Training log: `checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/vjepa21_train_log.json`
