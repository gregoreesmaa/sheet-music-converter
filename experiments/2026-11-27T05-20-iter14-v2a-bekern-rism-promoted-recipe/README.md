# Iter 14 — `UniOMR v2` (`small_45m`, `bekern+v2` Phase A + `ov-39` RISM Pseudo-Labels + Promoted Recipe)

- **Experiment ID:** `2026-11-27T05-20-iter14-v2a-bekern-rism-promoted-recipe`
- **Status:** `PLANNED`
- **Baseline Commit SHA:** `37e8387` *(+ Iter-14 warm-start & vocab-expand preflight commit)*
- **Warm-Start Checkpoint:** `D:\tmp\mvd-runs\uni-omr-iter13-v2-stage2-24k\uni_omr_v2_ema_latest.pt` (`vocab_size=237` -> append-only `bekern+v2` Phase A `vocab_size=265`, copying `0..232` verbatim and remapping `PAD/BOS/EOS/SYS` `233..236 -> 261..264`)
- **Platform / GPU:** Local `RTX 4080 SUPER` (`16 GB` VRAM, BF16 native, `$0` cloud cost)
- **Quick Links:**
  - Plan & Decision Tree: [PLAN.md](./PLAN.md)
  - Verified 3-Phase Launcher (RISM Pseudo-Labels + Stage-2 Train + Auto Evals): [launch_iter14.ps1](./launch_iter14.ps1)
  - Outcome *(written on completion)*: `OUTCOME.md`
  - HF Repo Paths (`gregoreesmaa/sheet-music-converter-omr`):
    - Ledger: `runs/2026-11-27T05-20-iter14-v2a-bekern-rism-promoted-recipe.json`
    - Checkpoints: `checkpoints/2026-11-27T05-20-iter14-v2a-bekern-rism-promoted-recipe/`
    - Eval JSONs: `evals/2026-11-27T05-20-iter14-v2a-bekern-rism-promoted-recipe_suite31_hw.json`
- **Local Output Dirs:**
  - Phase 0 RISM `ov-39` Pseudo-Labels: `D:\tmp\rism-pseudolabels-ov39\uni_rism_pseudolabels_ov39.jsonl`
  - Phase 0 Combined Manifest: `D:\tmp\uni_fullpage_iter14_v2a_rism.jsonl`
  - Phase 1 Stage 2 (`16,000` steps): `D:\tmp\mvd-runs\uni-omr-iter14-v2a-stage2-16k`
  - Phase 2 Eval Reports:
    - `suite31_hw` Fast (`1408px/3584, FSA+MTP-4, mem=4096`): `D:\tmp\uni-omr-suite31-ood\iter14_v2a_stage2_16k_ema_latest_1408_fsa_mtp.json`
    - `suite31_hw` `+ILMS (0.40/0.85, mem=4096)`: `D:\tmp\uni-omr-suite31-ood\iter14_v2a_stage2_16k_ema_latest_1408_fsa_mtp_ilms.json`
    - `OMR-NorthStar` (`125` pages, `ema_latest`, `--neg-control-gate`): `D:\tmp\northstar-eval-runs\iter14-v2a-last\report.json`

---

## Summary & Hypothesis

Iter 14 warm-starts directly from our all-time `UniOMR v2` champion (`Iter 13` `uni_omr_v2_ema_latest.pt`, `41.3M` params, `NS = 34.08`, `Tier-W NS = 36.45`, `suite31_hw` `0.7137 / 0.7059`) into **`bekern+v2` Phase A** (`vocab_size = 265`, `+28` append-only expressive & structural tokens covering `8` extended dynamics, `9` articulations/ornaments/technical marks, and `11` structural markers including `<staff>`, `xstaff-up/down`, `ottava-8va/8vb/stop`, `cue`, `volta-1/2`, `tup-open/close`) and integrates the **Iter 13 12-scout forensic learnings** + **`ov-39` RISM semi-supervised pseudo-labels**:

1. **Phase 0 (Data — `ov-39` RISM Pseudo-Labels + `bekern+v2` Phase A Manifest):** Generates high-confidence RISM archival handwritten pseudo-labels (`--limit 1200 --bf16 --max-dim 1408 --resume`, `mean_logprob >= -0.35`, `beat_valid_frac >= 0.70` / `0.85`) using Iter 13 `ema_latest` and merges them with the `47,727`-row `bekern+v2` Phase A + `250` `neg_control` manifest (`0` OOV, `0/45` `suite31_hw` overlap, `0/125` `OMR-NorthStar` overlap) into `D:\tmp\uni_fullpage_iter14_v2a_rism.jsonl`.
2. **Phase 1 (Training — `16,000` Full-Page Steps):** Warm-starts with exact append-only vocab expansion (`0..232` copied verbatim, `PAD/BOS/EOS/SYS` remapped `233..236 -> 261..264`, `233..260` fresh-init), `--stratified-preset v3` (8-bucket sampler including `real_hw_pseudo` and `neg_control`), `--expr-token-loss-weight 2.5` (targeting NorthStar `S_expr = 0.5544`), `--tok-dropout 0.12`, `--adaptive-ink-target`, `--save-ema-milestones 4000 --best-min-step 4000`, and all Tier 1–3 speedups (`bs=2 ga=4`, cuDNN NCHW + `--fixed-hw-buckets`, `BatchedMuon`, `--kd-stop-frac 0.65`, WSD `0.20` cooldown + last-15% INT4 QAT).
3. **Phase 2 (Auto-Eval on `ema_latest`):** Runs `suite31_hw` Fast (`FSA+MTP-4`, `--max-mem-tokens 4096`), `suite31_hw` `+ILMS 0.40/0.85` (`--max-mem-tokens 4096`), and 125-page `OMR-NorthStar` (`--neg-control-gate` + rebuilt `assembly-serialize.exe`).

**Hypothesis:** Expanding the vocabulary losslessly to `bekern+v2` Phase A (`718,267` new expressive/structural token occurrences across `26,546` re-exported pages) with `2.5x` expressive-token CE upweighting, `ov-39` RISM handwritten pseudo-labels, `neg_control` pages, and `ema_latest` post-WSD evaluation will lift `OMR-NorthStar` `S_expr` (`0.5544 -> >= 0.65`), `S_struct` (`0.7905 -> >= 0.82`), `HWM` (`23.31 -> >= 28.0`), and headline `NS` (`34.08 -> >= 39.0`) while maintaining or beating Iter 13's `suite31_hw` champion marks (`all45_clean >= 0.715`, `all45_photo >= 0.708`).

---

## Baselines to Beat

| Checkpoint | Benchmark / Arm | Headline / Key Metrics |
|---|---|---|
| **Iter 13 `v2` (`ema_best`)** | `suite31_hw` Fast (`1408/3584, FSA+MTP-4`) | `pr_clean=0.9193`, `pr_photo=0.9103`, `hw_clean=0.2448`, `hw_photo=0.2533`, **`all45_clean=0.7094`**, **`all45_photo=0.7059`** |
| **Iter 13 `v2` (`ema_best`)** | `suite31_hw` High-Acc (`+dewarp+mbeam2`) | `pr_clean=0.9153`, `pr_photo=0.9130`, `hw_clean=0.2672`, `hw_photo=0.2464`, **`all45_clean=0.7137`**, **`all45_photo=0.7056`** |
| **Iter 13 `v2` (`ema_latest`)** | `OMR-NorthStar` (`125p, --neg-control-gate`) | **`NS=34.08`**, `Tier-W NS=36.45`, `M_geom=59.20`, `CDR=82.94`, `ALS=78.47`, `RWP=80.93`, `HWM=23.31`, `S_expr=0.5544`, `S_struct=0.7905`, `NMHR=0.0%` |
| **Iter 14 `v2a` Target (`PROMOTE`)** | `suite31_hw` + `OMR-NorthStar` (`ema_latest`) | `all45_clean >= 0.715`, `all45_photo >= 0.708`, `NS >= 39.0`, `S_expr >= 0.65`, `HWM >= 28.0`, `NMHR == 0.0%` |
