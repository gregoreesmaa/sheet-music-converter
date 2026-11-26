# `2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep` — Decoupled `--muon-lr` Sweep with Un-Lagged EMA on `UniOMR v2 small_45m`

- **Status:** `QUEUED` (Colab `L4`, Wave 4, `depends_on: ["2026-11-26T00-54-ablation-muon-vs-adamw", "2026-11-26T01-38-ov-13-muon-lr-sweep"]`)
- **Predecessors:** [`2026-11-26T00-54-ablation-muon-vs-adamw`](../2026-11-26T00-54-ablation-muon-vs-adamw/OUTCOME.md) (`ITERATE`), [`2026-11-26T01-38-ov-13-muon-lr-sweep`](../2026-11-26T01-38-ov-13-muon-lr-sweep/OUTCOME.md) (`ITERATE`)
- **Baseline Commit SHA:** `9d3f4c3`
- **Model:** `UniOMR v2 small_45m` (`~41.3M` params)
- **Dataset:** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, 100% permissive)

## Summary & Hypothesis
Exp 4 (`2026-11-26T00-54-ablation-muon-vs-adamw`) revealed two confounds when comparing `Muon+AdamW` vs `AdamW` at coupled `--lr 3e-4`:
1. **Under-tuned Muon LR:** With `--muon-lr` tied to `3e-4`, `Muon` lagged `AdamW` by `+0.709` CE in steps `1–100` and closed monotonically to `+0.025` RAW `val_ce` at step `1,200` while already beating `AdamW` on total `val_loss` (`-0.0215`) and `raw_val_ser` (`-0.0044`).
2. **EMA Lag Confound:** `--gpu-ema-decay 0.997 --gpu-ema-every 5` over `1,200` steps left `48.6%` random step-0 initialization in the EMA weights (`0.997^240 = 0.486`).

**Hypothesis:** Decoupling `--muon-lr ∈ {6e-4, 1.0e-3, 1.5e-3}` (`2×`, `3.3×`, `5×` AdamW's `3e-4`) while holding AdamW `--lr 3e-4` fixed and using un-lagged EMA (`--gpu-ema-decay 0.99 --gpu-ema-every 1`, `0.99^1200 = 5.8e-6`) will eliminate Muon's early-step lag and decisively beat Exp 4's `AdamW` RAW baseline (`val_ce=1.376862`, `val_token_f1=0.360878`, `val_sys_count_exact=0.375000`, `val_ser=0.878187`) on both RAW and EMA `proxy_val` metrics.

## Quick Links
- Launcher script: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep.json`
