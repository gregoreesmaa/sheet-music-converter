# Experiment `2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep` — DINOv3 Gram Anchoring Weight Sweep (`0.0`, `0.1`, `0.5`) on Unlabeled RISM 5,707

- **Status:** `PLANNED` (queued in `experiments/overnight_queue/batch_07.json`, wave 7)
- **Platform:** Google Colab `L4` (`24 GB` VRAM, native BF16, ~30–40 min total, ~3.2 CU)
- **Baseline Commit SHA:** `164852f`

## Summary & Hypothesis

Sweep the DINOv3 Ink-Token Gram Anchoring weight (`--gram-weight`) in `tmp-train/mvd/pretrain_ijepa_rism_v2.py` across **`0.0`** (no Gram anchoring control), **`0.1`** (light anchoring), and **`0.5`** (moderate anchoring) for `500` optimizer steps per arm on `datasets/rism_hw_5707.tar` (`5,707` unlabeled real historical RISM handwritten manuscript pages, `UniOMR v2 small_45m`, `BF16`, `Muon+AdamW`, `seed=42`).

**Hypothesis:** Pure V-JEPA dense prediction without Gram anchoring (`gram_weight=0.0`) risks dimensional collapse or smoothing away fine spatial pitch/staff-line geometry on unlabeled manuscripts, while overly strong Gram anchoring (`1.0`) may freeze the encoder too close to its random step-0 geometry. Measuring dense ink-token **effective rank (`erank`)**, **normalized spectral entropy**, **mean off-diagonal pairwise cosine similarity**, and **Gram drift** on 128 held-out RISM pages will identify the sweet spot (`0.1` vs `0.5`) that prevents collapse while allowing rich RISM domain adaptation.

## Quick Links

- Plan, launch commands & decision tree: [PLAN.md](PLAN.md)
- Standalone Colab L4 runner: [run_colab.py](run_colab.py)
- Outcome (written after completion): `OUTCOME.md` (pending)
- Overnight queue entry: `experiments/overnight_queue/batch_07.json`
- Hugging Face Hub (`gregoreesmaa/sheet-music-converter-omr`, private dataset):
  - Eval JSON: `evals/2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep.json`
  - Run ledger: `runs/2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep.json`
