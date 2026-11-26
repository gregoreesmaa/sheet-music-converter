# PLAN — `2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep`

Zero-prior-context operational handoff for **Exp 4b: Decoupled `--muon-lr` Sweep with Un-Lagged EMA** on `UniOMR v2 small_45m` (mandatory `ITERATE` successor to `2026-11-26T00-54-ablation-muon-vs-adamw`).

---

## 1. Platform & Budget
- **Provider:** Google Colab (`L4`, 24 GB VRAM, `cc=8.9`, BF16 native, native `torch.optim.Muon`)
- **Estimated Wall-Clock:** `~45 min` (`3 arms × 1,200 steps` = `3,600` steps + 6 `proxy_val` evals)
- **Estimated Compute Units:** `~3.0 CU` (auto-teardown on exit)
- **Baseline Commit SHA:** `9d3f4c3`
- **Predecessor:** `2026-11-26T00-54-ablation-muon-vs-adamw` (`AdamW` RAW baseline: `val_ce=1.376862`, `val_mtp=2.073071`, `val_ser=0.878187`, `val_token_f1=0.360878`, `val_sys_count_exact=0.375000`)

---

## 2. Identifiers & Artifact Paths
- **Experiment ID:** `2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep`
- **HF Repo (`dataset`):** `gregoreesmaa/sheet-music-converter-omr`
  - Input dataset: `datasets/iter10_5_proxy_2k.tar`
  - Eval JSON: `evals/2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep.json`
  - Run ledger: `runs/2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep.json`
  - Best EMA ckpt: `checkpoints/2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep/best_ema_latest.pt`

---

## 3. Exact Parameter Table (Isolated Variable: `--muon-lr`)

| Hyperparameter | Value | Notes |
|---|---|---|
| `--preset` | `small_45m` | `41.3M` params (`UniOMRv2Net`) |
| `--optimizer` | `muon` | Native `torch.optim.Muon` (2D Transformer) + fused `AdamW` (rest) |
| `--lr` (AdamW) | `3.0e-4` | Held constant (identical to Exp 4) |
| `--muon-lr` (swept) | `6.0e-4`, `1.0e-3`, `1.5e-3` | `2×`, `3.3×`, `5×` AdamW LR — single isolated variable |
| `--steps` | `1200` | Identical to Exp 4 for apples-to-apples comparison vs `AdamW` RAW |
| `--batch-size` / `--grad-accum` | `2` / `4` | Effective batch = `8` full pages |
| `--lr-schedule` | `wsd` | `--warmup-steps 120 --wsd-decay-frac 0.25 --min-lr-ratio 0.05` |
| `--gpu-ema-decay` / `--gpu-ema-every` | `0.99` / `1` | **Un-lagged EMA** (`0.99^1200 = 5.8e-6` init weight vs `48.6%` in Exp 4) |
| `--save-every-minutes` | `5.0` | Atomic `.tmp -> os.replace` |
| `--seed` | `42` | Identical to Exp 4 |
| Proxy Val Eval | `64` pages (`proxy_val.jsonl`) | Both RAW & EMA: `val_loss`, `val_ce`, `val_mtp`, `val_ser`, `val_token_f1`, `val_sys_count_exact` (`eval_max_len=512`) |

---

## 4. Decision Tree
1. **Promote (`Muon + AdamW` with winning `--muon-lr`):** If any `--muon-lr` arm beats Exp 4's `AdamW` RAW (`val_ce < 1.3769` AND `val_token_f1 > 0.3609` AND `val_ser <= 0.8782`) with `grad_norm_p95 < 5.0` and zero NaNs → promote `--optimizer muon --muon-lr <winner>` for Iter 12+ `UniOMR v2` training.
2. **Discard `Muon` (Switch to Pure Fused `AdamW`):** If ALL three decoupled `--muon-lr` arms still trail Exp 4's `AdamW` RAW on `val_ce` and `val_token_f1` (or show instability at `>= 1e-3`) → switch default `UniOMR v2` optimizer to `--optimizer adamw` (`+7.15%` faster, better CE/F1/sys-count).
3. **Iterate:** If `1.5e-3` is still monotonically improving at the boundary with low grad norms → probe `2.0e-3`–`3.0e-3` in a 2-arm follow-up.
