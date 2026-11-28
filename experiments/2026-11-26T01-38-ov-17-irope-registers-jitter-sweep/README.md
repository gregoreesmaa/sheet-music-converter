# `2026-11-26T01-38-ov-17-irope-registers-jitter-sweep`

**Title:** Overnight Exp 17 — ViT-5/DINOv3 Register Tokens (`0`, `4`, `8`, `16`) & RoPE Box-Jitter Sweep on `UniOMR v2 small_45m`
**Status:** `COMPLETED` — **Verdict: `PROMOTE`** (`n_register_tokens=8`, keep `rope_jitter=0.08`; `+4.05 pp` vs `reg0`, `+0.50 pp` vs `reg4` EMA Token F1, zero speed/VRAM cost)
**Baseline Commit SHA:** `164852f`
**Target Platform:** Google Colab `L4` (24 GB VRAM, BF16 native, ~38–42 min total, ~3.4 CU)

---

## Summary & Hypothesis

`UniOMR v2` (`HybridAxialMergerEncoder`) includes three 2025–2026 frontier visual upgrades in `UniOMRv2Config` (currently at untuned defaults `irope_nope_every=4`, `n_register_tokens=4`, `rope_jitter=0.08`):

1. **RoPE-exempt Register Tokens (`n_register_tokens`):** Appended after `InkTokenRouterV2` inside `GlobalEncoderLayer` to absorb high-norm background/global summary spikes (ViT-5 / DINOv3 finding). We sweep `n_register_tokens in {0, 4, 8, 16}` at fixed `rope_jitter=0.08`.
2. **V2PE RoPE Box-Jitter (`rope_jitter`):** Applies multiplicative training-time coordinate jitter to `AspectNormalizedRoPE1D` and `AspectNormalizedRoPE2D` to improve scale/aspect generalization across varied page aspect ratios and camera framing. We sweep `rope_jitter in {0.0, 0.08, 0.16}` at fixed `n_register_tokens=4` (sharing the `(reg=4, jitter=0.08)` control arm for 6 total arms).

**Hypothesis:** Adding `4–8` RoPE-exempt register tokens will reduce global attention norm spikes and improve held-out `proxy_val` Token F1 / `sys_count_exact` over `0` registers, while moderate RoPE jitter (`0.08`) will beat both `0.0` (overfit to fixed aspect) and `0.16` (excessive coordinate noise on tight chord stacks).

---

## Quick Links

- **Plan & Runbook:** [PLAN.md](./PLAN.md)
- **Standalone Colab Runner:** [run_colab.py](./run_colab.py)
- **Outcome:** `OUTCOME.md` *(written after Colab run completes)*
- **HF Private Hub Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T01-38-ov-17-irope-registers-jitter-sweep.json`
  - Run Ledger: `runs/2026-11-26T01-38-ov-17-irope-registers-jitter-sweep.json`
