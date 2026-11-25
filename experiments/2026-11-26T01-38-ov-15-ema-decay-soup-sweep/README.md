---
id: 2026-11-26T01-38-ov-15-ema-decay-soup-sweep
title: "Overnight Exp 15: GPU EMA Decay & k=3 Checkpoint Souping Sweep on UniOMR v2 small_45m"
status: QUEUED
wave: 4
priority: 3
provider: colab_l4
gpu: L4
est_minutes: 32
est_cu: 1.6
baseline_sha: 164852f
---

# `2026-11-26T01-38-ov-15-ema-decay-soup-sweep` — GPU EMA Decay & `k=3` Checkpoint Souping Sweep (`UniOMR v2 small_45m`)

- **Status:** `QUEUED` (Wave 4 overnight batch 01, Google Colab `L4`)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params, `bf16`, `fullpage_max_dim=1280`)
- **Dataset (100% permissive):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, zero `CC-BY-NC`, zero `suite31_hw` overlap)

## Summary & Hypothesis

`UniOMR v2` maintains a device-resident `GPUEmaShadow` updated every `5` steps with default decay `0.997` (effective horizon `~333` EMA updates = `~1,665` optimizer steps). Under WSD cooldown, too-fast EMA (`0.995`) tracks noisy late gradients, while too-slow EMA (`0.9995`) lags behind the cooldown trajectory and under-converges. Additionally, uniform weight-space checkpoint souping (`k=3` snapshots across the cooldown window or across EMA horizons) can flatten the loss basin at zero inference cost.

**Hypothesis:** (1) `gpu_ema_decay=0.9985` will outperform both `0.995` (too noisy) and `0.9995` (too laggy) on `proxy_val` CE loss and Token F1; (2) uniform `k=3` checkpoint souping — tested both as **intra-arm cooldown trajectory soup** (steps `600, 700, 800`) and **cross-arm EMA soup** (averaging `0.995 + 0.9985 + 0.9995` EMA checkpoints) — will beat the best single raw checkpoint by `<= -0.02` val CE and match or beat the best individual EMA checkpoint at zero extra inference cost.

## Parameter Grid

| Arm ID | `gpu_ema_decay` | `gpu_ema_every` | `k=3` Cooldown Soup Snapshots | Steps | Peak LR | Seed |
|---|---:|---:|---|---:|---:|---:|
| `ema_0p995` | `0.995` | `5` | steps `600, 700, 800` | `800` | `3.0e-4` | `42` |
| `ema_0p9985` | `0.9985` | `5` | steps `600, 700, 800` | `800` | `3.0e-4` | `42` |
| `ema_0p9995` | `0.9995` | `5` | steps `600, 700, 800` | `800` | `3.0e-4` | `42` |
| `cross_ema_soup_k3` | `avg(0.995, 0.9985, 0.9995)` | — | Uniform FP32 average of all 3 final EMA ckpts | — | — | `42` |

## Falsification Criteria

- **Promote winning EMA decay and/or `k=3` soup:** Winning candidate beats the best raw checkpoint by `<= -0.02` `proxy_val` CE loss and `>= +0.01` Token F1; if `k=3` trajectory soup beats the best single EMA by `<= -0.01` val CE, add automatic `k=3` cooldown snapshot souping to `train_uni_v2.py` before Wave 5.
- **Keep default `0.997` without souping (Null / Falsified):** All EMA decays and `k=3` soups fall within noise (`|delta CE| < 0.01`) or lag behind raw weights (`ema_val_ce > raw_val_ce`).
- **Iterate:** If `0.9995` lags on 800 steps because of cold EMA initialization, note EMA warmup (`min(decay, (1+t)/(10+t))`) as a Wave 5 follow-up.

## Quick Links

- Plan & Copy-Paste Launch Commands: [PLAN.md](./PLAN.md)
- Colab Runner: [run_colab.py](./run_colab.py)
- Outcome: `OUTCOME.md` *(written after HF eval JSON is retrieved)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-15-ema-decay-soup-sweep.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-15-ema-decay-soup-sweep.json`
