# Iter 12 — `UniOMR v2` (`small_45m`, 2-Stage `4k Strip Warmup + 20k Mixed`, All Tonight's Learnings)

- **Experiment ID:** `2026-11-26T07-15-iter12-v2-strip-warmup-24k`
- **Status:** `LAUNCHING`
- **Supersedes:** `2026-11-26T02-40-iter11-v2-24k` (stopped at step 8k due to cold-start full-page + under-driven `muon_lr=3e-4`)
- **Platform / GPU:** Local `RTX 4080 SUPER` (`16 GB` VRAM, BF16 native)
- **Output Dirs:**
  - Stage 1 (4k strip warmup): `D:\tmp\mvd-runs\uni-omr-iter12-v2-stage1-strip4k`
  - Stage 2 (20k mixed fullpage): `D:\tmp\mvd-runs\uni-omr-iter12-v2-stage2-20k`
- **Incorporated Learnings from Tonight's 17 Experiments:**
  1. **2-Stage Strip→Fullpage Curriculum (Iter 11 lesson):** `4,000` strip-only steps (`eff_batch = 8×4 = 32` strips/step, ~45 min) to reach `strip F1 > 0.90` *before* touching full pages, then `20,000` `mixed` full-page steps.
  2. **Decoupled `Muon` LR (`ov-13` + Exp 4):** `--muon-lr 1.0e-3 --lr 3.0e-4` in Stage 1 and `--muon-lr 8.0e-4 --lr 3.0e-4` in Stage 2 (unlocks Muon's fast convergence without pushing AdamW `lm_head`/embeddings into RAW decode jitter).
  3. **Un-lagged EMA (`Exp 4` + `ov-13`):** `--gpu-ema-decay 0.995 --gpu-ema-every 2` in Stage 1 (`0.995^2000 = 4.4e-5` initial weight) and `0.9985 / 5` in Stage 2.
  4. **`1280px` Training Resolution (`ov-35`):** `--fullpage-max-dim 1280 --multi-scale` (`+1.26 pp` `all45_clean` & `~35%` faster step time than `1536px`).
  5. **Online `Iter 10.5 → v2` Hybrid KD (`Iter 10.5` record):** `--teacher-ckpt D:\tmp\mvd-runs\uni-omr-iter10_5-v1\uni_omr_ema_best.pt --kd-beta 0.5 --kd-alpha 0.30 --kd-temp 1.5`.
  6. **Architecture Guardrails (Exps 2, 3, 8):** Random init (NO collapsed Exp 2 JEPA), `InkTokenRouterV2` with 4-edge flood-fill `paper_gate` + `max_mem_tokens=10,240` (Exp 3), unrotated last-15% `INT4 QAT` (`28.04 MB` WebGPU, Exp 8).
