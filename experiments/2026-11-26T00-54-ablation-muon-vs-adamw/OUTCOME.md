# Outcome — `2026-11-26T00-54-ablation-muon-vs-adamw`

- **Status:** `COMPLETED`
- **Verdict:** **`ITERATE`** *(do NOT kill currently running Iter 11 `UniOMR v2` 24k-step `Muon+AdamW` run; run decoupled `--muon-lr` + un-lagged EMA follow-up `2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep`)*
- **Completed At (UTC):** `2026-11-26T01:04:59Z` (started `2026-11-25T23:14:10Z`)
- **Platform / GPU:** Modal `NVIDIA L4` (`24 GB`, Ada `cc=8.9`, `torch=2.15.0+cu132`, native `torch.optim.Muon`, native BF16) — verified **stopped** (`ap-6azxTTf3jTgmB8eoXTxJA1`, `0` tasks)
- **Elapsed Wall-Clock:** `5,504.37 s` (`91.7 min` total: `muon` `2,847.20 s` + `adamw` `2,657.17 s`; `~$1.22` Modal Starter credit)
- **Baseline Commit SHA:** `9d73170`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`, `dataset`):**
  - Eval JSON: `evals/2026-11-26T00-54-ablation-muon-vs-adamw.json` (pulled to [eval_result.json](./eval_result.json))
  - Run ledger: `runs/2026-11-26T00-54-ablation-muon-vs-adamw.json` (pulled to [run_ledger.json](./run_ledger.json))
  - EMA checkpoints: `checkpoints/2026-11-26T00-54-ablation-muon-vs-adamw/{muon,adamw}_ema_latest.pt`

---

## Measured Results (`1,200` steps/arm, `iter10_5_proxy_2k`, `seed=42`, `lr=3e-4`, `eval_max_len=512`)

### 1. Proxy Validation (`64` held-out full pages, `proxy_val.jsonl`)

| Checkpoint & Metric | `Muon+AdamW` (`lr=3e-4`) | `AdamW` (`lr=3e-4`) | Delta (`Muon - AdamW`) | Better |
|---|---:|---:|---:|---|
| **RAW `val_loss`** | **`2.090752`** | `2.112267` | **`-0.021515`** | `Muon+AdamW` |
| **RAW `val_ce_loss`** | `1.401941` | **`1.376862`** | `+0.025079` | `AdamW` |
| **RAW `val_mtp_loss`** | `2.120608` | **`2.073071`** | `+0.047537` | `AdamW` |
| **RAW `val_stroke_hog_loss`** | `0.021261` | **`0.021055`** | `+0.000206` | Tie |
| **RAW `val_ser`** ↓ | **`0.873788`** | `0.878187` | **`-0.004399`** | `Muon+AdamW` |
| **RAW `val_mean_sample_ser`** ↓ | **`1.371373`** | `1.409388` | **`-0.038015`** | `Muon+AdamW` |
| **RAW `val_token_f1`** ↑ | `0.331521` | **`0.360878`** | `-0.029357` | `AdamW` |
| **RAW `val_sys_count_exact`** ↑ | `0.234375` (`15/64`) | **`0.375000`** (`24/64`) | `-0.140625` | `AdamW` |
| **EMA `val_loss`** *(confounded — see below)* | **`2.185460`** | `2.207815` | **`-0.022355`** | `Muon+AdamW` |
| **EMA `val_ce_loss`** *(confounded)* | `1.467558` | **`1.428714`** | `+0.038844` | `AdamW` |
| **EMA `val_mtp_loss`** *(confounded)* | `2.204574` | **`2.146670`** | `+0.057904` | `AdamW` |
| **EMA `val_ser`** ↓ *(confounded)* | **`0.885197`** | `1.028862` | **`-0.143665`** | `Muon+AdamW` |
| **EMA `val_token_f1`** ↑ *(confounded)* | **`0.245284`** | `0.086467` | **`+0.158817`** | `Muon+AdamW` |
| **EMA `val_sys_count_exact`** ↑ *(confounded)* | **`0.171875`** (`11/64`) | `0.109375` (`7/64`) | **`+0.062500`** | `Muon+AdamW` |

### 2. Training Convergence, Speed & Memory

| Metric | `Muon+AdamW` | `AdamW` | Delta (`Muon - AdamW`) |
|---|---:|---:|---:|
| Steps `1–100` Train CE (`lr` warmup) | `3.86648` | **`3.15739`** | `+0.70909` |
| Steps `501–600` Train CE | `1.74294` | **`1.64225`** | `+0.10069` |
| Steps `801–900` Train CE (end of plateau) | `1.56533` | **`1.51672`** | `+0.04861` |
| Steps `1101–1200` Train Loss | **`2.13724`** | `2.13867` | **`-0.00143`** |
| Steps `1101–1200` Train CE | `1.43545` | **`1.40596`** | `+0.02949` |
| Steps `1101–1200` Train MTP | `2.16708` | **`2.11920`** | `+0.04788` |
| Grad Norm Mean / p95 (steps `1101–1200`) | `1.074 / 1.571` | `0.968 / 1.566` | `+0.106 / +0.005` (both clean) |
| Mean Step Time (`mean_step_sec`) | `2.359 s` (`2359.1 ms`) | **`2.209 s`** (`2208.6 ms`) | `+0.151 s` (`+6.82%` step / `+7.15%` wall) |
| Wall-Clock (`1,200` steps) | `2,847.20 s` (`0.4215 st/s`) | **`2,657.17 s`** (`0.4516 st/s`) | `+190.03 s` (`+7.15%`, passes `<= 15%` gate) |
| Peak VRAM Allocated / Reserved | **`9.516 / 9.721 GB`** | `9.644 / 9.807 GB` | **`-0.128 / -0.086 GB`** (`Muon` uses less) |

---

## Scrutiny & Anomaly Check

1. **Coverage & Sanity:** All `64/64` held-out validation pages scored for both RAW and EMA checkpoints on both arms; exact same `41,334,795`-param `small_45m` model, identical `seed=42`, identical CLI except `--optimizer muon` vs `--optimizer adamw` (verified `optimizer_mode == "muon+adamw"` vs `"adamw"`). Zero NaNs, zero OOMs, zero grad-norm blowups (`p95 <= 2.87` after warmup).
2. **CRITICAL ANOMALY #1 — Severe EMA Lag Confound (`decay=0.997, every=5` over `1,200` steps):**
   - `1,200` steps with `update_every=5` executes only `240` EMA updates, so `0.997^240 = 0.486` — **48.6% of the EMA checkpoint weight is still the random step-0 initialization!**
   - Consequently, EMA degraded BOTH arms severely relative to RAW, and catastrophically collapsed `AdamW` EMA (`token_f1: 0.3609 -> 0.0865`, `val_ser: 0.8782 -> 1.0289`, `sys_count_exact: 0.375 -> 0.109`).
   - Because `AdamW` RAW is actually the best overall checkpoint in Exp 4 (`val_ce=1.3769`, `token_f1=0.3609`, `sys_count_exact=0.3750`), the apparent `Muon` EMA victory on `val_ser` (`-0.1437`) and `token_f1` (`+0.1588`) is largely an **EMA-lag artifact** and cannot be used to promote `Muon(lr=3e-4)`.
3. **ANOMALY #2 — Muon Under-LR Signature (`--muon-lr` tied to `--lr 3e-4`):**
   - `train_uni_v2.py` sets `eff_muon_lr = lr = 3e-4` when `--muon-lr` is omitted.
   - `Muon` started `+0.709` CE behind `AdamW` in steps `1–100`, then monotonically closed the gap every 100 steps (`+0.709 -> +0.165 -> +0.166 -> +0.148 -> +0.128 -> +0.101 -> +0.089 -> +0.070 -> +0.049 -> +0.038 -> +0.025 -> +0.029`).
   - This is the textbook signature of **under-tuned Muon LR**: even with `adjust_lr_fn="match_rms_adamw"`, orthogonalized 2D matrix updates commonly prefer `2×–5×` higher LR than AdamW (`--muon-lr ∈ {6e-4, 1.0e-3, 1.5e-3}` while holding AdamW at `3e-4`).
4. **ANOMALY #3 — Total `val_loss` vs `val_ce + val_mtp` Divergence:**
   - Despite higher `val_ce` (`+0.0251`) and `val_mtp` (`+0.0475`), `Muon` achieved **lower total `val_loss`** (`2.0908` vs `2.1123`, `-0.0215`) on RAW (and `-0.0224` on EMA), implying `~0.061` lower auxiliary encoder losses (`saliency + sys_y + sys_count + staff_seg`). Muon is learning spatial encoder features faster while decoder CE lags at `muon_lr=3e-4`.
5. **ANOMALY #4 — First-100-Step Muon Warmup Overhead:**
   - Steps `1–100` of `Muon` averaged `2785.2 ms/step` vs `2245.6 ms/step` for `AdamW` (initial Newton-Schulz kernel/allocator warmup), settling to `2218–2412 ms/step` thereafter (`+7.15%` overall wall-clock overhead, comfortably inside the `<= 15%` budget, and `-128 MB` lower peak VRAM because Muon keeps 1 momentum buffer vs AdamW's 2).

---

## Verdict & Impact on Live Iter 11 (`UniOMR v2`, 24k Steps)

- **Verdict: `ITERATE`** (neither `PROMOTE` nor `DISCARD` gate was cleanly met: `AdamW` RAW wins `val_ce` by `-0.0251`, `val_token_f1` by `+0.0294`, and `val_sys_count_exact` by `+0.1406`, while `Muon+AdamW` wins total `val_loss` by `-0.0215`, `raw_val_ser` by `-0.0044`, and uses `-0.128 GB` less VRAM with only `+7.15%` wall-clock overhead while still rapidly catching up).
- **Decision on Currently Running Iter 11 (`UniOMR v2`, 24k steps with `Muon+AdamW`):** **DO NOT STOP / KEEP RUNNING.**
  1. `Muon+AdamW` is 100% stable (zero NaNs, clean grad norms, lower VRAM, `+7.15%` speed overhead).
  2. Over `24,000` steps (`20×` Exp 4's horizon), EMA lag vanishes (`0.997^4800 = 5.4e-7`) and Muon's monotonic catch-up trajectory + stronger spatial encoder losses have ample horizon to overtake or match AdamW.
  3. Before Iter 12 (or any future `v2` run), we must test **decoupled `--muon-lr`** without EMA lag so we know whether `Muon(muon_lr > 3e-4) + AdamW(3e-4)` beats `AdamW(3e-4)` cleanly on RAW & EMA `val_ce` + `token_f1` + `sys_count_exact`.

---

## Follow-Up Action / Successor Experiment

1. **Complementary running sweep (`ov-13`):** [`2026-11-26T01-38-ov-13-muon-lr-sweep`](../2026-11-26T01-38-ov-13-muon-lr-sweep/PLAN.md) (currently `RUNNING` on Colab L4) sweeps *coupled* `--lr ∈ {1.5e-4, 3.0e-4, 5.0e-4}` at `800` steps (still with `gpu_ema_decay=0.997/5`).
2. **Queued Mandatory `ITERATE` Successor (`exp4b`):** [`2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep`](../2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep/PLAN.md) — appended to `experiments/overnight_queue/queue.json` (`depends_on: ["2026-11-26T00-54-ablation-muon-vs-adamw"]`). Isolates **decoupled `--muon-lr ∈ {6e-4, 1.0e-3, 1.5e-3}`** (holding AdamW `--lr 3e-4` constant) for `1,200` steps on `iter10_5_proxy_2k` with **un-lagged EMA (`--gpu-ema-decay 0.99 --gpu-ema-every 1`)**, directly comparable against Exp 4's `AdamW` RAW baseline (`val_ce=1.3769`, `token_f1=0.3609`, `sys_count_exact=0.3750`).
3. **Skill Amended:** Appended the short-proxy EMA lag gotcha (`0.997^240 = 48.6%` random-init retention at 1,200 steps) to `.agents/skills/running-an-experiment/SKILL.md`.
