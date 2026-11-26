# `2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840` — Scaled Label-Free RISM RLVR (`updates=160`, `max_len=512`, Canonical `1280px/3840` Eval)

- **Status:** `QUEUED` (Wave 7, successor to [`2026-11-26T01-38-ov-37-rlvr-kl-group-sweep`](../2026-11-26T01-38-ov-37-rlvr-kl-group-sweep/OUTCOME.md) — Google Colab `NVIDIA L4`, `cc=8.9`, BF16)
- **Depends On:** `2026-11-26T01-38-ov-37-rlvr-kl-group-sweep` (`COMPLETED`)
- **Baseline:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`
- **Single Variable Tested:** Scaling the two Pareto-winning RLVR configs from `ov-37` (`kl0.06_G8`, `kl0.15_G4`) plus their midpoint (`kl0.10_G8`) from `40` updates / `256`-token rollouts to **`160` updates / `512`-token rollouts**, evaluated apples-to-apples at the `ov-35` optimal **`max_dim=1280, eval_max_len=3840`** with full per-piece logging.

## Summary & Hypothesis
In `ov-37`, 4/6 label-free RISM RLVR arms cleared `+0.015` HW Note F1 with `<= 0.0012` printed drift, and training reward was still climbing steeply at step 40 (`0.096 → 0.201`). However, `ov-37` had three bottlenecks: (1) `updates=40` only touched `0.7%` of the 5,707 RISM pages, (2) `max_len=256` truncated rollouts at ~1–2 systems, and (3) `eval_max_len=1536` truncated `50–57%` of HW eval pages before EOS and omitted per-piece deltas.

**Hypothesis:** Scaling to `updates=160` and `max_len=512` rollouts with `kl_beta ∈ {0.06, 0.10, 0.15}` will produce a broad, multi-piece HW Note F1 gain (`>= +0.025` on `hw_clean` and `hw_photo` at `1280px/3840` with `>= 9/14` HW pieces improving) while keeping `printed_clean` regression `<= 0.010`, confirming RLVR post-training as a turnkey Iter 11 booster.

## Quick Links
- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Predecessor Outcome: [`ov-37/OUTCOME.md`](../2026-11-26T01-38-ov-37-rlvr-kl-group-sweep/OUTCOME.md)
- HF Eval Target: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840.json`
