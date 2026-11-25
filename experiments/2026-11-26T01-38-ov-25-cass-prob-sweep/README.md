# `2026-11-26T01-38-ov-25-cass-prob-sweep` — Overnight Exp 25: CASS Scheduled Sampling Probability Sweep

- **Status:** `PLANNED` (`QUEUED` in `experiments/overnight_queue/batch_05.json`, Wave 6)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params, `tok_dropout=0.0` to isolate CASS cleanly)
- **Dataset (100% permissive):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages)

## Summary & Hypothesis
Autoregressive full-page OMR decoders suffer from **exposure bias** on long multi-system pages (`> 512` tokens): during teacher-forced training the decoder always sees ground-truth prefix tokens, whereas at inference a single mistyped duration, barline, or `<sys>` separator cascades into compounding pitch/rhythm drift or premature `<eos>`.

`train_uni_v2.py` includes our fixed **no-tok-dropout CASS** (Context-Aware Scheduled Sampling, `apply_tok_dropout=False` on both the no-grad draft pass and the second decoder pass, never replacing `BOS`/`PAD` and never injecting `EOS`). This experiment runs a controlled 3-arm sweep on `UniOMR v2 small_45m` (`700` steps/arm, identical `seed=42`, `lr=3e-4`, `dim=1280`, BF16 on Colab `L4`):
- **Arm 1 (`cass_0.00` — Control):** `cass_prob=0.0` (pure teacher forcing, `tok_dropout=0.0`).
- **Arm 2 (`cass_0.10`):** `cass_prob=0.10` (10% self-predicted prefix tokens on full pages).
- **Arm 3 (`cass_0.25`):** `cass_prob=0.25` (25% self-predicted prefix tokens on full pages).

**Hypothesis:** Moderate CASS (`cass_prob=0.10` or `0.25`) will improve autoregressive `proxy_val` Token F1 and SER — especially on **long full pages (`> 512` ref tokens)** — and reduce premature `<eos>` truncation (`len_ratio` closer to `1.0`) relative to pure teacher forcing (`cass_prob=0.0`), without hurting short-page accuracy.

## Quick Links
- Colab Runner: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Manifest: [batch_05.json](../overnight_queue/batch_05.json)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-25-cass-prob-sweep.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-25-cass-prob-sweep.json`
