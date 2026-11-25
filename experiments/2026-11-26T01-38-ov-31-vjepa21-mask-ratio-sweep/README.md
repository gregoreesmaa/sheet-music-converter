# Experiment `2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep` — V-JEPA 2.1 Masking Ratio Sweep (`0.45`, `0.60`, `0.75`) on Unlabeled RISM 5,707

- **Status:** `PLANNED` (queued in `experiments/overnight_queue/batch_07.json`, wave 7)
- **Platform:** Google Colab `L4` (`24 GB` VRAM, native BF16, ~30–40 min total, ~3.2 CU)
- **Baseline Commit SHA:** `164852f`

## Summary & Hypothesis

Sweep the HTR-VT anisotropic ink-span masking ratio in our `V-JEPA 2.1 + DINOv3 Gram Anchoring + Stroke-HOG` self-supervised encoder pretrainer (`tmp-train/mvd/pretrain_ijepa_rism_v2.py`) across three controlled arms — **`0.45`** (baseline range `[0.40, 0.50]`), **`0.60`** (`[0.55, 0.65]`), and **`0.75`** (`[0.70, 0.80]`) — for `500` optimizer steps per arm on `datasets/rism_hw_5707.tar` (`5,707` unlabeled real historical RISM handwritten manuscript pages, `UniOMR v2 small_45m`, `BF16`, `Muon+AdamW`, `seed=42`).

**Hypothesis:** Historical manuscripts have dense, repetitive horizontal staff-line structure and sparse vertical melodic/accidental events. While `0.45` masking may be too easy to interpolate from adjacent pixels, `0.75` may starve the sparse ink-token router of sufficient context. A `0.60` masking ratio is hypothesized to maximize semantic representation quality (measured via linear Ridge and `k=5` cosine kNN staff-density probes on a held-out 256-page RISM split) while maintaining stable `jepa_masked`, `deep_sup`, and `gram` loss trajectories.

## Quick Links

- Plan, launch commands & decision tree: [PLAN.md](PLAN.md)
- Standalone Colab L4 runner: [run_colab.py](run_colab.py)
- Outcome (written after completion): `OUTCOME.md` (pending)
- Overnight queue entry: `experiments/overnight_queue/batch_07.json`
- Hugging Face Hub (`gregoreesmaa/sheet-music-converter-omr`, private dataset):
  - Eval JSON: `evals/2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep.json`
  - Run ledger: `runs/2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep.json`
