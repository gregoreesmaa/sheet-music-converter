---
id: 2026-11-26T01-38-ov-13-muon-lr-sweep
title: "Overnight Exp 13: Muon+AdamW Peak LR Sweep on UniOMR v2 small_45m"
status: QUEUED
wave: 4
priority: 1
provider: colab_l4
gpu: L4
est_minutes: 30
est_cu: 1.5
baseline_sha: 164852f
---

# `2026-11-26T01-38-ov-13-muon-lr-sweep` — Muon+AdamW Peak LR Sweep (`UniOMR v2 small_45m`)

- **Status:** `QUEUED` (Wave 4 overnight batch 01, Google Colab `L4`)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params, `bf16`, `fullpage_max_dim=1280`)
- **Dataset (100% permissive):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, zero `CC-BY-NC`, zero `suite31_hw` overlap)

## Summary & Hypothesis

`UniOMR v2` routes 2D Transformer weight matrices to native `torch.optim.Muon(adjust_lr_fn="match_rms_adamw", momentum=0.95, nesterov=True, ns_steps=5)` and 4D convolutions/embeddings/norms/biases/aux heads to fused `AdamW`. While `lr=3.0e-4` was inherited from `UniOMR v1` pure AdamW runs, orthogonalized Muon updates often tolerate a higher peak learning rate (`5.0e-4`) for faster convergence or require a lower peak (`1.5e-4`) to avoid late-stage variance on dense full-page OMR sequences (`max_seq_len=2048`).

**Hypothesis:** Sweeping peak LR across `{1.5e-4, 3.0e-4, 5.0e-4}` (matched for both Muon RMS-scaled 2D blocks and AdamW non-2D blocks under identical `seed=42`, 800 steps, WSD schedule) will identify a strictly superior peak LR for Wave 5 full-scale Iter-11 training on `A100/H100`, reducing `proxy_val` EMA CE loss by `>= 0.03` and/or improving greedy `proxy_val` Token F1 by `>= +1.5%` relative to the default `3.0e-4` anchor.

## Parameter Grid (Single-Variable Isolation)

| Arm ID | Peak LR (`--lr`) | Optimizer | Steps | Warmup | WSD Decay Frac | EMA Decay | Resolution | Seed |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `lr_1p5e-4` | `1.5e-4` | `muon` (`muon+adamw`) | `800` | `80` | `0.25` | `0.997` | `1280px` | `42` |
| `lr_3p0e-4` | `3.0e-4` | `muon` (`muon+adamw`) | `800` | `80` | `0.25` | `0.997` | `1280px` | `42` |
| `lr_5p0e-4` | `5.0e-4` | `muon` (`muon+adamw`) | `800` | `80` | `0.25` | `0.997` | `1280px` | `42` |

## Falsification Criteria

- **Promote non-default LR (`1.5e-4` or `5.0e-4`):** Beats `lr_3p0e-4` on BOTH `proxy_val` EMA CE loss (`delta <= -0.02`) and greedy EMA Token F1 (`delta >= +0.01`), with zero NaN/Inf steps and `p95 grad_norm < 5.0`.
- **Keep default `3.0e-4` (Null / Falsified):** `lr_3p0e-4` wins or all arms fall within noise (`|delta CE| < 0.02` and `|delta F1| < 0.01`), or `5.0e-4` exhibits gradient spikes (`p95 grad_norm > 8.0`) or loss divergence.
- **Iterate:** If `5.0e-4` wins cleanly without instability, spawn a follow-up probe at `7.0e-4` or decoupled `--muon-lr` ratio sweep.

## Quick Links

- Plan & Copy-Paste Launch Commands: [PLAN.md](./PLAN.md)
- Colab Runner: [run_colab.py](./run_colab.py)
- Outcome: `OUTCOME.md` *(written after HF eval JSON is retrieved)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-13-muon-lr-sweep.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-13-muon-lr-sweep.json`
