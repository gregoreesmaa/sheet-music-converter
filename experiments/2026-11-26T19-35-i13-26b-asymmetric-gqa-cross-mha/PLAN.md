# Plan — `2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA L4` (24 GB VRAM, native BF16).
- **Budget:** 3 arms × 800 steps (`~16–18 min/arm` ≈ `50 min`, `~3.8 CU`).
- **Session Name:** `i13-26b`
- **HF Targets:**
  - `evals/2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha.json`
  - `runs/2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha.json`

## 2. Architecture Injection (Zero Edits to `uni_omr_v2.py` / `train_uni_v2.py`)
- Sets `cfg.n_kv_heads = n_kv_heads_self` via `train_uni_v2.build_v2_config`.
- Wraps `uni_omr_v2.UniOMRv2Net.__init__` so that when `n_kv_heads_cross != n_kv_heads_self`, each `layer.cross_attn` in `self.dec_layers` is replaced with `GroupedQueryAttention(cfg.d_model, n_heads=cfg.n_heads, n_kv_heads=n_kv_heads_cross, use_rope=False)` (preserving exact seed initialization).
- `precompute_cross_kv` and `GroupedQueryAttention.forward` read `ca.n_kv_heads` dynamically, so KV-caching and `param_report()` work natively.

## 3. Matched Hyperparameters
- `--preset small_45m`, `--optimizer muon`, `--lr 3e-4`, `--weight-decay 0.01`, `--lr-schedule wsd`, `--warmup-steps 80`, `--wsd-decay-frac 0.25`, `--min-lr-ratio 0.05`
- `--steps 800`, `--batch-size 2`, `--grad-accum 4`, `--curriculum fullpage`, `--fullpage-max-dim 1280`, `--max-seq-len 1536`, `--bf16`, `--grad-checkpoint`
- `--stroke-hog-weight 0.15`, `--gpu-ema-decay 0.99`, `--save-every-minutes 5.0`, `--eval-every 800`, `--eval-max-len 512`, `--seed 42`
- Dataset: `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` held-out val).

## 4. Promotion / Discard Gates
- **PROMOTE `asym_s2_c8`:** Beats `kv2_sym` by `>= +0.5 pp` EMA Token F1 AND stays within `0.3 pp` of `kv8_sym` -> add `n_kv_heads_cross: Optional[int] = None` to `UniOMRv2Config` and set `(self=2, cross=8)` for Iter 14.
- **PROMOTE `kv8_sym`:** `kv8_sym` beats `asym_s2_c8` by `> 0.5 pp` EMA Token F1 (meaning self-attention also needs 8 KV heads) -> set `n_kv_heads=8` for Iter 14.
- **DISCARD:** Both `asym_s2_c8` and `kv8_sym` fail to beat `kv2_sym` by `>= +0.5 pp` at 800 steps -> retain `n_kv_heads=2`.
