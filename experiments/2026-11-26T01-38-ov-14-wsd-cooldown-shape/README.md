---
id: 2026-11-26T01-38-ov-14-wsd-cooldown-shape
title: "Overnight Exp 14: WSD Cooldown Fraction & Decay Curve Sweep on UniOMR v2 small_45m"
status: QUEUED
wave: 4
priority: 2
provider: colab_l4
gpu: L4
est_minutes: 35
est_cu: 1.8
baseline_sha: 164852f
---

# `2026-11-26T01-38-ov-14-wsd-cooldown-shape` — WSD Cooldown Fraction & Decay Curve Sweep (`UniOMR v2 small_45m`)

- **Status:** `QUEUED` (Wave 4 overnight batch 01, Google Colab `L4`)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params, `bf16`, `fullpage_max_dim=1280`)
- **Dataset (100% permissive):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, zero `CC-BY-NC`, zero `suite31_hw` overlap)

## Summary & Hypothesis

Warmup-Stable-Decay (WSD) training lets us hold peak learning rate flat through the stable plateau and perform a rapid final cooldown into a sharp-yet-flat valley. Currently `train_uni_v2.py` hardcodes a `cosine` cooldown over `wsd_decay_frac=0.25`. Recent WSD scaling studies (Hägele et al., 2024; MiniCPM) show that `1 - sqrt(t)` or `linear` cooldowns — especially paired with a tighter (`0.15`) or longer (`0.30`) decay window — often outperform cosine cooldown by spending more steps at lower effective LR near the end of training.

**Hypothesis:** A full `2 × 3` unconfounded factorial sweep over `wsd_decay_frac ∈ {0.15, 0.30}` and decay curve `∈ {cosine, linear, 1_minus_sqrt}` (`500` steps/arm, identical `seed=42`, `lr=3.0e-4`, `optimizer=muon`) will show that `1_minus_sqrt` or `linear` cooldown reduces `proxy_val` EMA CE loss by `>= 0.02` and improves greedy Token F1 by `>= +1.0%` over the `cosine` baseline, with `wsd_decay_frac=0.30` benefiting shorter runs while `0.15` preserves more high-LR exploration.

## Parameter Grid (`2 × 3` Factorial)

| Arm ID | `wsd_decay_frac` | Cooldown Curve | Formula for `d_prog ∈ [0, 1]` | Steps | Warmup | Peak LR | Seed |
|---|---:|---|---|---:|---:|---:|---:|
| `frac015_cosine` | `0.15` | `cosine` | `0.5 * (1 + cos(pi * d_prog))` | `500` | `50` | `3.0e-4` | `42` |
| `frac030_cosine` | `0.30` | `cosine` | `0.5 * (1 + cos(pi * d_prog))` | `500` | `50` | `3.0e-4` | `42` |
| `frac015_linear` | `0.15` | `linear` | `1.0 - d_prog` | `500` | `50` | `3.0e-4` | `42` |
| `frac030_linear` | `0.30` | `linear` | `1.0 - d_prog` | `500` | `50` | `3.0e-4` | `42` |
| `frac015_sqrt` | `0.15` | `1_minus_sqrt` | `1.0 - sqrt(d_prog)` | `500` | `50` | `3.0e-4` | `42` |
| `frac030_sqrt` | `0.30` | `1_minus_sqrt` | `1.0 - sqrt(d_prog)` | `500` | `50` | `3.0e-4` | `42` |

## Falsification Criteria

- **Promote winning `(wsd_decay_frac, curve)`:** Beats both `cosine` arms on `proxy_val` EMA CE loss (`delta <= -0.02`) and achieves equal-or-better greedy Token F1 (`delta >= +0.005`), warranting adding `--wsd-decay-curve {cosine,linear,sqrt}` directly to `tmp-train/mvd/uni_omr.py` and `train_uni_v2.py` before Wave 5.
- **Keep `cosine` (Null / Falsified):** `cosine` wins or all 6 cells fall within noise (`|delta CE| < 0.015` and `|delta F1| < 0.008`).
- **Iterate:** If `1_minus_sqrt` wins at `0.30` with strong monotonic improvement in the final 100 steps, test `wsd_decay_frac=0.40` or pair with Exp 13's winning peak LR.

## Quick Links

- Plan & Copy-Paste Launch Commands: [PLAN.md](./PLAN.md)
- Colab Runner: [run_colab.py](./run_colab.py)
- Outcome: `OUTCOME.md` *(written after HF eval JSON is retrieved)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-14-wsd-cooldown-shape.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-14-wsd-cooldown-shape.json`
