# Experiment Plan — `2026-11-26T07-40-i13-20-masked-diffusion-measure-denoiser`

> **Zero-Prior-Context Handoff:** Self-contained proxy ablation for **Bidirectional Masked-Diffusion Measure Denoising (`MDM-MeasureRefiner`)** (`i13-20`). Does not modify `uni_omr_v2.py` or `train_uni_v2.py`.

## 1. Platform & Budget
- **Provider:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, native BF16).
- **Estimated Runtime & Cost:** `~40 min` (`~3.4 CU`) — ~13 min Arm A (1,000 steps), ~16 min Arm B (1,000 steps joint AR + MDM), ~11 min Rust scorer + `proxy_val` (including oracle mask-recovery probe) + 45-piece `suite31_hw` eval (`K=0`, `K=4`, `K=6`).
- **Warm-Start Resolution:** Auto-resolves `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` → fallback `checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt` → fallback matched 600-step `UniOMRv2Config.small_45m()` proxy warm-up.

## 2. Architecture Delta (`+2.75M` params, `< 1.8 MB` INT4)
- **`MDMMeasureRefiner`:**
  - Dedicated learned `[MASK]` embedding vector (`d_model=384`) + sinusoidal noise-level `t` MLP injection (`AdaRMSNorm` scale/shift).
  - `4` bidirectional (non-causal) GQA Transformer layers (`d_model=384`, `n_heads=8`, `n_kv_heads=2`, `d_ff=768` SwiGLU, `RoPE1D` + cross-attention to `UniOMRv2`'s `mem_final`, QK-Norm).
  - Weight-tied input `tok_emb` and output `lm_head` with `UniOMR v2` (0 extra embedding/vocab params).
- **Masked-Diffusion Training Objective:**
  - Sample `t ~ Uniform(0.15, 0.70)`; mix `50%` i.i.d. Bernoulli token masking and `50%` measure-span masking (masking 1–2 contiguous bar-delimited spans per system).
  - Weighted cross-entropy on masked positions: `L_mdm = mean(min(4.0, 1/t) * CE(logits[mask], target[mask]))`.
  - Total loss: `L = L_ar + 0.5 * L_mdm` (keeping AR draft quality identical while training the bidirectional denoiser).
- **Inference (`K=4` Cosine Unmasking + FSA Guard):**
  - Mask tokens with AR draft `p_max < tau` (`tau=0.65`, max `35%` masked).
  - Over `K` steps, compute bidirectional denoiser probabilities, keep tokens whose refined confidence exceeds the cosine schedule threshold, and finalize any remaining masked tokens on step `K`.
  - Run `PrefixState` FSA validation/fallback per system so `DecOK` never regresses.

## 3. Controlled 2-Arm Ablation (`seed=42`, BF16)

| Hyperparameter | Arm A (`control_causal_ar`) | Arm B (`mdm_measure_refiner`) |
|---|---|---|
| Base Model | `UniOMR v2 small_45m` (`~41.3M`) | Identical + `MDMMeasureRefiner` (`+2.75M`) |
| Steps / Eff. Batch | `1,000` steps, `B_eff=8` (`2×4`) | Identical |
| Learning Rate | `lr=3e-5` (WSD: 100 warmup, 25% cosine) | `lr_backbone=3e-5`, `lr_mdm=3e-4` (shared WSD) |
| Eval Modes | Causal FSA greedy (`K=0`) | `K=0` (AR draft), `K=4` (`tau=0.65`), `K=6` (`tau=0.75`) + Oracle 30% mask-fill probe |

## 4. Checkpointing & Artifacts
- Atomic checkpoints every `5.0 min` to `/content/vol/ckpts/2026-11-26T07-40-i13-20-masked-diffusion-measure-denoiser/<arm>/`.
- Uploads `evals/2026-11-26T07-40-i13-20-masked-diffusion-measure-denoiser.json` and `runs/2026-11-26T07-40-i13-20-masked-diffusion-measure-denoiser.json` to HF.

## 5. Kill / Promote Decision Tree
- **PROMOTE:** `K=4` denoising lifts `suite31_hw` `hw_clean` Note F1 by `>= +0.025` (or Pitch/Token F1 `>= +0.030`) over `K=0` AND maintains `DecOK >= 98%` printed / `>= 90%` HW AND latency overhead `< 25%` → integrate `MDMMeasureRefiner` into Iter 13 decoding stack.
- **ITERATE:** Oracle 30% mask-fill probe improves over AR by `>= +0.05` Token F1, but confidence-gated `tau=0.65` net gain is `< +0.015` → replace static `tau` with a learned token-error mask predictor.
- **DISCARD:** `DecOK` drops below `90%`, `hw_clean` delta `< +0.005`, or denoiser over-smooths rare accidentals/ornaments.
