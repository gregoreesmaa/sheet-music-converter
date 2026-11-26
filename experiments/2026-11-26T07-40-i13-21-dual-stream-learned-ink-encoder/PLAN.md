# Experiment Plan — `2026-11-26T07-40-i13-21-dual-stream-learned-ink-encoder`

> **Zero-Prior-Context Handoff:** Self-contained proxy ablation for **Dual-Stream Raw-RGB + Differentiable Learned-Binarized-Ink Encoder (`DualInkStream`)** (`i13-21`). Does not modify `uni_omr_v2.py` or `train_uni_v2.py`.

## 1. Platform & Budget
- **Provider:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, native BF16).
- **Estimated Runtime & Cost:** `~42 min` (`~3.5 CU`) — ~13 min Arm A (1,000 steps), ~18 min Arm B (1,000 steps + RISM SSL every 2 steps), ~11 min Rust scorer + `proxy_val` (clean + archival bleed-through stress) + 45-piece `suite31_hw` eval.
- **Warm-Start Resolution:** Auto-resolves `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` → fallback `checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt` → fallback matched 600-step `UniOMRv2Config.small_45m()` proxy warm-up.

## 2. Architecture Delta (`+0.86M` params, `< 0.9 MB` INT8)
1. **`LearnedInkBinarizer` (`~0.18M` params):**
   - Inputs: grayscale luma `Y`, `15×15` local high-pass residual `hp15`, `41×41` local high-pass residual `hp41`, and local std `sigma15` (4 channels).
   - Compact 3-stage residual Conv-GroupNorm-SiLU U-Net (channels `24 → 48 → 24`) predicting soft ink probability `I_ink = sigmoid(logit_ink) ∈ [0, 1]^(B, 1, H, W)`.
2. **`InkStreamEncoder` + Zero-Init Gated FiLM Fusion (`~0.68M` params):**
   - Lightweight 3-stage anisotropic conv encoder (`1 → 32 (/2) → 64 (/4) → 128 (/8)`) producing ink features `f_ink` at `/8` (`d_model=384` projected).
   - Fuses into `AnisotropicMusicBackbone` output `feat_2d` via zero-initialized gates:
     `feat_fused = feat_2d * (1 + tanh(g_scale) * gamma(f_ink)) + tanh(g_bias) * beta(f_ink)`.
   - Exact functional bit-identity with `UniOMR v2` at initialization (`g_scale = g_bias = 0`).
3. **Unlabeled RISM Photometric-Invariance SSL (`5,707` real historical manuscripts):**
   - Every `2` steps, sample `B=2` real RISM pages (`768px`), apply severe archival perturbation `x'` (parchment yellowing, flipped/shifted low-opacity verso bleed-through ghost, contrast washout, radial illumination gradient), and minimize:
     `L_rism = L1(I_ink(x'), stopgrad(I_ink(x))) + 0.10 * L_tv(I_ink(x')) + 0.15 * (mean(I_ink) - target_ink_density)^2`.

## 3. Controlled 2-Arm Ablation (`seed=42`, BF16)

| Hyperparameter | Arm A (`control_rgb_only`) | Arm B (`dual_stream_learned_ink`) |
|---|---|---|
| Base Model | `UniOMR v2 small_45m` (`~41.3M`) | Identical + `DualInkStream` (`+0.86M`, zero-init gates) |
| Steps / Eff. Batch | `1,000` steps, `B_eff=8` (`2×4`) | Identical |
| Learning Rate | `lr=3e-5` (WSD: 100 warmup, 25% cosine) | `lr_backbone=3e-5`, `lr_ink=3e-4`, `lr_gate=3e-3` (shared WSD) |
| Train Photometric Aug | `p=0.50` archival parchment + bleed-through + lighting | Identical |
| RISM SSL | Off | Every `2` steps (`weight=0.25`, `B=2`, `768px`) |

## 4. Checkpointing & Artifacts
- Atomic checkpoints every `5.0 min` to `/content/vol/ckpts/2026-11-26T07-40-i13-21-dual-stream-learned-ink-encoder/<arm>/`.
- Uploads `evals/2026-11-26T07-40-i13-21-dual-stream-learned-ink-encoder.json` and `runs/2026-11-26T07-40-i13-21-dual-stream-learned-ink-encoder.json` to HF.

## 5. Kill / Promote Decision Tree
- **PROMOTE:** Mean fusion gate opens (`mean |tanh(g)| >= 0.05`) AND Arm B lifts `suite31_hw` `hw_clean` or `hw_photo` Strict Note F1 by `>= +0.020` (and cuts archival bleed-through CE degradation on `proxy_val` by `>= 20%`) with `printed_clean` delta `>= -0.005` → wire `DualInkStream` into `AnisotropicMusicBackbone` for Iter 13.
- **ITERATE:** Clear gains on `hw_photo` / bleed-through stress (`>= +0.025`) while `hw_clean` is flat (`< +0.010`) → add direct `StaffSegHead` + `StrokeHogHead` supervision onto `I_ink`.
- **DISCARD:** Fusion gates stay collapsed (`< 0.02`) OR `I_ink` drops faint ledger lines causing `hw_clean` regression (`< 0.0`).
