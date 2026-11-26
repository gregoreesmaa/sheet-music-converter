# Iter 11 — `UniOMR v2` (`small_45m`, `24,000` steps) with `Iter 10.5 → v2` Hybrid Distillation

- **Experiment ID:** `2026-11-26T02-40-iter11-v2-24k`
- **Status:** `LAUNCHING`
- **Platform / GPU:** Local `RTX 4080 SUPER` (`16 GB` VRAM, Ada SM 8.9, native BF16)
- **Target Steps:** `24,000` (WSD schedule: `500` warmup, `17,500` plateau, `6,000` cooldown)
- **Teacher Checkpoint:** `D:\tmp\mvd-runs\uni-omr-iter10_5-v1\uni_omr_ema_best.pt` (`suite31_hw` `all45_clean=0.6984`, `all45_photo=0.6832`, `printed_clean=0.8968`, `printed_photo=0.8935`, `hw_clean=0.2591`, `hw_photo=0.2174`)
- **Output Dir:** `D:\tmp\mvd-runs\uni-omr-iter11-v2-24k`
- **Key Cloud Decisions Incorporated:**
  - **Random init (NO Exp 2 `ijepa_best.pt` warm-start):** Exp 2 `OUTCOME.md` caught severe JEPA rank collapse (`p3_erank=2.1/384`, `mem_erank=2.6/384`).
  - **Keep `InkTokenRouterV2` (NO Exp 3 `ViR+ViCO`):** Exp 3 `OUTCOME.md` showed `InkTokenRouterV2` beat `ViR+ViCO` by `+6.04 pp` Raw F1 / `+6.10 pp` `ink_recall`. Upgraded with 4-edge flood-fill `paper_gate` + `max_mem_tokens=10,240`.
  - **Unrotated `QATInt4ControllerV2` (NO Exp 8 `SpinQuant`):** Exp 8 `OUTCOME.md` proved SpinQuant degrades W4A16 parity by `-6.4 to -30.0 pp`, while unrotated `small_45m` exports cleanly to `28.04 MB` WebGPU INT4 (`< 55 MB` gate!).
  - **Clean Iter 11 Manifest:** `D:\tmp\uni_fullpage_iter11_clean.jsonl` (`44,734` rows, all `>16`-sys `__vrv_` rows purged, `--max-systems-per-page 16`).
