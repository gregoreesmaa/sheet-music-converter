# Experiment Plan — `2026-11-26T07-40-i13-19-coarse2fine-zoom-omr`

> **Zero-Prior-Context Handoff:** Self-contained proxy ablation for **Coarse-to-Fine 2-Pass System-Plan → Zoomed-System Foveated Decoding** (`i13-19`). Does not modify `uni_omr_v2.py` or `train_uni_v2.py`; all wrapper modules live inside `run_colab.py`.

## 1. Platform & Budget
- **Provider:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, native BF16).
- **Estimated Runtime & Cost:** `~42 min` (`~3.5 CU`) — ~14 min Arm A (1,000 steps), ~17 min Arm B (1,000 steps), ~11 min Rust scorer + `proxy_val` + 45-piece `suite31_hw` eval.
- **Warm-Start Resolution:** Auto-resolves `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` → fallback `checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt` → fallback matched 600-step `UniOMRv2Config.small_45m()` proxy warm-up so the ablation is always apples-to-apples.

## 2. Architecture Delta (`+0.24M` params, `< 0.5 MB` INT8)
1. **`SystemBandProposalHead` (`~0.19M` params):**
   - Pools Pass-1 `768px` backbone `P3 (/8)` features horizontally (`mean + max` → `2*d_model -> 128`), applies 3 layers of 1D dilated conv (`k=5`, `d=1,2,4`) + 2-head 1D self-attention along the vertical axis `y ∈ [0, H/8)`, and outputs per-row logits `(p_interior(y), p_boundary(y))`.
   - Supervised via vertical ink-energy band decomposition aligned to ground-truth `N_sys` (exact system count from `<sys>` delimiters in `bekern`).
   - At inference, connected components of `p_interior(y) > 0.45` (with `±12%` height padding and NMS merge for tiny fragments) yield `[y_top, y_bot]` boxes.
2. **`GlobalContextCarryProj` (`~0.05M` params):**
   - Attentive pooling of Pass-1 `mem_final` into `4` learned global context tokens (`(B, 4, d_model)`), zero-init gated and concatenated to each Pass-2 zoomed band's `mem_final` so every zoomed system retains page-level key/clef/time awareness.

## 3. Controlled 2-Arm Ablation (`seed=42`, BF16)
Both arms start from the exact same weights and consume the exact same `iter10_5_proxy_2k` page stream:

| Hyperparameter | Arm A (`control_singlepass_1280`) | Arm B (`coarse2fine_zoom_2pass`) |
|---|---|---|
| Base Model | `UniOMR v2 small_45m` (`~41.3M`) | Identical + `SystemBandProposalHead` + `GlobalContextCarryProj` (`+0.24M`) |
| Steps / Eff. Batch | `1,000` steps, `B_eff=8` (`2×4`) | Identical |
| Learning Rate | `lr=3e-5` (WSD: 100 warmup, 25% cosine decay) | `lr_backbone=3e-5`, `lr_head=3e-4` (shared WSD) |
| Training Views | Full page `1280px` | 50% Pass-1 (`768px`) + band BCE/Dice + Pass-2 zoomed bands (`256×1280`); 50% full page `1280px` |
| Inference Mode | Single-pass `1280px` FSA greedy | Pass-1 `768px` band proposal → Pass-2 `256×1280` zoomed bands + Global Context → join `<sys>` |

## 4. Checkpointing & Artifacts
- Atomic checkpoints every `5.0 min` to `/content/vol/ckpts/2026-11-26T07-40-i13-19-coarse2fine-zoom-omr/<arm>/`.
- Uploads `evals/2026-11-26T07-40-i13-19-coarse2fine-zoom-omr.json` and `runs/2026-11-26T07-40-i13-19-coarse2fine-zoom-omr.json` to `gregoreesmaa/sheet-music-converter-omr`.

## 5. Kill / Promote Decision Tree
- **PROMOTE:** `suite31_hw` `hw_clean` Note F1 `>= +0.030` (or `hw_clean` Token F1 `>= +0.040`) over Arm A AND Pass-1 system band recall `>= 0.92` on real HW AND `printed_clean` Note F1 delta `>= -0.010` → wire native 2-pass foveated zoom into `uni_omr_v2.py` for Iter 13.
- **ITERATE:** Zoomed-system token F1 improves by `>= +0.030` on correctly localized bands, but Pass-1 horizontal boxes miss/merge curved grand-staff systems (`band_recall < 0.92`) → replace horizontal `[y_top, y_bot]` boxes with 2D curved seam / polygon crops.
- **DISCARD:** Arm B fails to beat Arm A (`hw_clean` delta `< +0.010`) OR Pass-2 latency exceeds `2.5×` Arm A without `>= +0.020` HW Note F1 gain.
