# Experiments

Structured, reproducible experiment ledger for `sheet-music-converter`.
Every experiment follows the [running-an-experiment](../.agents/skills/running-an-experiment/SKILL.md) skill.

## Directory Convention

```text
experiments/
├── README.md                              # This index
└── YYYY-MM-DDTHH-MM-<short-slug>/         # One folder per isolated experiment (minute precision)
    ├── README.md                          # Goal, hypothesis, baseline SHA, status
    ├── PLAN.md                            # Platform, query IDs, data/HF shards, 5-min ckpt, next actions
    └── OUTCOME.md                         # Measured results, suite31_hw delta, verdict (written on completion)
```

## Standing Rules
- **Minute-precision timestamp prefix:** `YYYY-MM-DDTHH-MM-<slug>` (e.g., `2026-11-26T00-39-vjepa21-rism-v2`).
- **5-minute checkpoints mandatory:** Every training/adaptation run uses `--save-every-minutes 5.0` (atomic `.tmp` → rename + batched HF Hub sync to `gregoreesmaa/sheet-music-converter-omr`).
- **Zero-Prior-Context Handoff:** Every experiment folder includes its own launcher script, live query IDs, and copy-paste monitor/resume/eval commands.
- **Git checkpointing:** Commit and push `README.md` + `PLAN.md` *before* launching, and commit + push `OUTCOME.md` immediately *after* evaluation completes.
- **No secrets or large weights in git:** Store checkpoints/shards on Hugging Face (`gregoreesmaa/sheet-music-converter-omr`) or `D:\tmp\`, never inside `experiments/`.

---

## Active & Queued Experiments

| # | Experiment / Slug | Platform | Wave | Status |
|---|---|---|---|---|
| 0 | [2026-11-26T00-26-iter10-5-v1-redteam-fixes](./2026-11-26T00-26-iter10-5-v1-redteam-fixes/OUTCOME.md) | Local RTX 4080 SUPER | Active | `COMPLETED (PROMOTE)` — new record `all45_clean=0.6984` (`0.7004` clean_gt), `all45_photo=0.6832`, `hw_photo=0.2174`, printed fixes `12.0`; promoted as Iter 11 `v1` teacher |
| 1 | `cut-rism-real-ink-v06` (`train_cut_rism.py` + translate `hw-sim-v06`) | Kaggle `2× T4` (FP16) | Wave 1 | `QUEUED` (waiting on HF shards) |
| 2 | [2026-11-26T00-54-vjepa21-gram-rism-v2](./2026-11-26T00-54-vjepa21-gram-rism-v2/OUTCOME.md) (`pretrain_ijepa_rism_v2.py`, 1.5k steps) | Modal `L4` (BF16) | Wave 1 | `COMPLETED (ITERATE)` — rank collapse (`mem_erank=2.6`); DO NOT warm-start Iter 11 |
| 2b | [2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2](./2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2/) (`VICReg + KL-router + no-random-gram` + `suite31_hw` transfer) | Colab `L4` (BF16) | Wave 7 | `QUEUED` (successor to Exp 2) |
| 3 | [2026-11-26T00-54-ablation-vir-vico-router](./2026-11-26T00-54-ablation-vir-vico-router/) (`ViRTwoTierRouter+ViCO` vs `InkTokenRouterV2`) | Modal `2× L4` (BF16) | Wave 1 | `COMPLETED` (`ITERATE` — Keep `InkTokenRouterV2` for Iter 11; successor [`2026-11-26T02-18-ablation-vir-router-bucket-parity`](./2026-11-26T02-18-ablation-vir-router-bucket-parity/) queued) |
| 4 | [2026-11-26T00-54-ablation-muon-vs-adamw](./2026-11-26T00-54-ablation-muon-vs-adamw/OUTCOME.md) (`Muon+AdamW` vs `AdamW` on `v2`, 1.2k steps) | Modal `L4` (BF16) | Wave 1 | `COMPLETED (ITERATE)` — `AdamW` RAW wins `val_ce=1.3769` (`-0.0251`), `token_f1=0.3609` (`+0.0294`), `sys_exact=0.375`; `Muon` wins `val_loss=2.0908` (`-0.0215`), `raw_ser=0.8738` (`-0.0044`), `-128 MB` VRAM, `+7.15%` time; keep Iter 11 running; successor `exp4b` queued |
| 4b | [2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep](./2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep/) (decoupled `--muon-lr ∈ {6e-4, 1e-3, 1.5e-3}` + AdamW `3e-4`, un-lagged EMA `0.99/1`) | Colab `L4` (BF16) | Wave 4 | `QUEUED` (successor to Exp 4 & `ov-13`) |
| 5 | `ablation-unlabeled-rism-rlvr` (`rlvr_grpo.py` on Iter-10 `soup`) | Modal `L4` (BF16) | Wave 1 | `QUEUED` (waiting on HF shards) |
| 6 | `trial-scribe-ttt-suite31-hw` (`trial_2028_ttt_adapt.py` on 14 HW pages) | Modal `L4` (BF16) | Wave 2 | `QUEUED` |
| 7 | `trial-render-diff-rlvr` (`trial_2027_render_diff_rlvr.py`) | Modal `L4` (BF16) | Wave 2 | `QUEUED` |
| 8 | `ablation-spinquant-int4-parity` (`spinquant_fold.py` INT4 `suite31_hw` parity) | Modal `L4` / Kaggle `T4` | Wave 2 | `QUEUED` |
| 9 | `ablation-curriculum-fullpage100` (pure fullpage `1.0` vs mixed `0.80`) | Kaggle `2× T4` (FP16) | Wave 2 | `QUEUED` |
| 10 | `ablation-eagle3-speculative-speed` (`eagle3_drafter.py` vs `MTP-4`) | Modal `L4` (BF16) | Wave 3 | `QUEUED` (after `v2` ckpt) |
| 11 | `rism-multiview-pseudolabels` (`generate_rism_pseudolabels.py` on 5,707 RISM) | Kaggle `2× T4` (FP16) | Wave 3 | `QUEUED` (after Iter 10.5) |
| 12 | `ablation-dinov3-dense-distill-v2` (offline dense teacher → `v2` 5M backbone) | Modal `L4` (BF16) | Wave 3 | `QUEUED` |
| 34 | [2026-11-26T01-38-ov-34-measure-beam-header-guard-sweep](./2026-11-26T01-38-ov-34-measure-beam-header-guard-sweep/OUTCOME.md) (`measure_beam_size ∈ {1,2,4,8}` × `const_offset_limit ∈ {2,3,5}`) | Colab `L4` (BF16) | Wave 4 | `COMPLETED (ITERATE)` — `mbeam2_lim3` hits `all45_clean=0.7017` (`+0.0077`), `printed_clean=0.8966` (`+0.0071`), `hw_clean=0.2700` (`+0.0089`), but `hw_photo` regresses `-0.0177` at `+81.8%` overhead (unbounded HW triggers); successor `ov-34b` queued |
| 34b | [2026-11-26T05-35-ov-34b-budgeted-logprob-measure-beam](./2026-11-26T05-35-ov-34b-budgeted-logprob-measure-beam/) (budgeted `cap ∈ {4,6}` + `max_lp_drop` guardrail + softer/voice-only penalties at `mbeam=2`, `1280px/3840`) | Colab `L4` (BF16) | Wave 4 | `QUEUED` (successor to Exp 34) |
| 13 | [2026-11-26T01-38-ov-13-muon-lr-sweep](./2026-11-26T01-38-ov-13-muon-lr-sweep/OUTCOME.md) (coupled Muon+AdamW `--lr ∈ {1.5e-4, 3.0e-4, 5.0e-4}`, 800 steps) | Colab `L4` (BF16) | Wave 4 | `COMPLETED (ITERATE)` — `5.0e-4` won monotonically at upper boundary (`EMA val_ce -0.1106`, `EMA token_f1 +0.0690`, `grad_norm_p95=3.66`), but `61.8%` EMA init lag + coupled-AdamW RAW greedy jitter (`0.1670` F1); successor `exp4b` queued |
| 15 | [2026-11-26T01-38-ov-15-ema-decay-soup-sweep](./2026-11-26T01-38-ov-15-ema-decay-soup-sweep/OUTCOME.md) (`gpu_ema_decay ∈ {0.995, 0.9985, 0.9995}` + `k=3` traj/cross-EMA soups, 800 steps) | Colab `L4` (BF16) | Wave 4 | `COMPLETED (ITERATE)` — un-warmed EMA lagged on all 3 arms (`1.8%–67.0%` init wt); `k=3` traj soup (`600,700,800`) beat RAW 3-for-3 (`+0.0787` mean F1, peak `0.2977`; `-0.0504` SER) at `+0.0225` CE penalty; successor `ov-15b` queued |
| 15b | [2026-11-26T05-33-ov-15b-ema-warmup-late-cooldown-soup](./2026-11-26T05-33-ov-15b-ema-warmup-late-cooldown-soup/) (single-trajectory warmed-up EMA `{0.99, 0.995, 0.9985}` + late-cooldown soups `{700..800}`) | Colab `L4` (BF16) | Wave 4 | `QUEUED` (successor to Exp 15) |
| 35 | [2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep](./2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep/OUTCOME.md) (`max_dim ∈ {1280,1536,1792}` × `max_len ∈ {3840,5120,6144}`) | Colab `L4` (BF16) | Wave 4 | `COMPLETED (ITERATE)` — `1280px/3840` wins `all45_clean=0.7069` (`+0.0126`), `hw_clean=0.2799` (`+0.0181`), `printed_clean=0.8997` (`+0.0102`); `max_len>3840` strictly harmful; successor `ov-35b` queued |
| 35b | [2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep](./2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep/OUTCOME.md) (`max_dim ∈ {1024,1152,1280,1408}` × `max_len ∈ {2560,3072,3584,3840}`) | Kaggle `2× T4` (FP32) | Wave 4 | `COMPLETED (PROMOTE)` — `(1408px, 3072–3584)` brackets global 2D peak: `all45_clean=0.7016` (`+0.0074`), `all45_photo=0.6833` (`+0.0216`), 90p mean `0.6924` (`+0.0144` vs `1536/3840`), `hw_photo=0.2047` (`+0.0180` vs `1280`), `printed_photo=0.8994`; promote `1408px/3584` as Iter 12 default |
| 36 | [2026-11-26T01-38-ov-36-hw-multiview-tta-voting](./2026-11-26T01-38-ov-36-hw-multiview-tta-voting/OUTCOME.md) (6-view HW TTA + FSA/MBR voting) | Colab `L4` (BF16) | Wave 4 | `COMPLETED (ITERATE)` — Oracle `+0.0419` (`hw_clean 0.3030`) / `+0.0327` (`hw_photo 0.2243`); `clahe`/`deskew` poisoned 6-view rerankers; offline scale-only `fsa_beat` hits **`0.2954` (`+0.0343`)** clean / **`0.1969` (`+0.0053`)** photo; successor `ov-36b` queued |
| 36b | [2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta](./2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta/) (clean 5-scale `{1024..1536}` + `mean_logprob` + FSA TTA at `max_len=3840`) | Colab `L4` (BF16) | Wave 4 | `QUEUED` (successor to Exp 36) |
| 37 | [2026-11-26T01-38-ov-37-rlvr-kl-group-sweep](./2026-11-26T01-38-ov-37-rlvr-kl-group-sweep/OUTCOME.md) (label-free RISM RLVR `kl_beta ∈ {0.02,0.06,0.15}` × `G ∈ {4,8}`, 40 updates) | Kaggle `2× T4` (FP32) | Wave 7 | `COMPLETED (ITERATE)` — 4/6 arms beat `+0.015` HW F1 with `<= 0.0012` printed drift; `kl0.06_G8` (`hw_photo +0.0301`, `hw_clean +0.0035`, ckpt on HF) & `kl0.15_G4` (`hw_clean +0.0178`, `hw_photo +0.0130`, `KL=0.0574`); successor `ov-37b` queued |
| 37b | [2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840](./2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840/) (`kl0.06_G8` / `kl0.15_G4` / `kl0.10_G8` × `160` updates, `max_len=512`, canonical `1280px/3840` eval) | Colab `L4` (BF16) | Wave 7 | `QUEUED` (successor to Exp 37) |
| 38 | [2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep](./2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep/OUTCOME.md) (SCRIBE/DocTTT `steps ∈ {2,4,8}` × `lr ∈ {1e-5,3e-5,1e-4}` × `scope ∈ {norm,full}`, 18 cells) | Kaggle `2× T4` (FP32) | Wave 7 | `COMPLETED (ITERATE)` — `full_encoder_lr3e-05_s2` hits `hw_photo +0.0181` (`9W/1T/4L`), `hw_clean +0.0044` (`9W/0T/5L`), avg `+0.0113`; `full_encoder_lr1e-05_s4` hits `hw_clean +0.0124` (`11W/0T/3L`); `encoder_norm_only` inert; sharp cliff above `lr×s=6e-5`; successor `ov-38b` queued |
| 38b | [2026-11-26T06-25-ov-38b-scribe-ttt-budget-cons-3840](./2026-11-26T06-25-ov-38b-scribe-ttt-budget-cons-3840/) (SCRIBE/DocTTT `ent_weight ∈ {0.0,0.25,1.0}` × `full_encoder`/`backbone_router` at canonical `1280px/3840` + printed safety slice) | Colab `L4` (BF16) | Wave 7 | `QUEUED` (successor to Exp 38) |
| NS-1 | [2026-11-26T14-40-eval-northstar-iter12-v2](./2026-11-26T14-40-eval-northstar-iter12-v2/) (UniOMR Iter 12 Stage 2 `20k` EMA on 125-page OMR-NorthStar pack) | Local RTX 4080 SUPER (BF16) | NorthStar | `PLANNED` (ready after Iter 12 Stage 2 finishes) |
| NS-2 | [2026-11-26T14-40-eval-northstar-audiveris](./2026-11-26T14-40-eval-northstar-audiveris/) (Audiveris 5.3.1 black-box Docker AGPL-3.0 baseline on OMR-NorthStar) | Local CPU + Docker | NorthStar | `PLANNED` |
| NS-3 | [2026-11-26T14-40-eval-northstar-oemer](./2026-11-26T14-40-eval-northstar-oemer/) (`oemer` MIT black-box CLI baseline on OMR-NorthStar) | Local CPU | NorthStar | `PLANNED` |
| NS-4 | [2026-11-26T14-40-eval-northstar-homr](./2026-11-26T14-40-eval-northstar-homr/) (`homr` MIT isolated-venv neural OMR baseline on OMR-NorthStar) | Local CPU | NorthStar | `PLANNED` |
| i13-26 | [2026-11-26T07-40-i13-26-gqa-kv-heads-sweep](./2026-11-26T07-40-i13-26-gqa-kv-heads-sweep/) (Decoder GQA `n_kv_heads` ∈ `{1,2,4,8}` sweep; `kv8_mha` +4.19 pp EMA F1) | Colab L4 | proxy_val | `ITERATE` -> `i13-26b` |
| i13-27 | [2026-11-26T07-40-i13-27-rope-coord-scale-sweep](./2026-11-26T07-40-i13-27-rope-coord-scale-sweep/) (RoPE `rope_coord_scale` ∈ `{32,64,128,256}` sweep; 0.12 pp spread, retain 128.0) | Colab L4 | proxy_val | `DISCARD` |
| i13-26b | [2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha](./2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha/) (Asymmetric GQA `self=2, cross=8` vs `2/2` vs `8/8` at 800 steps) | Colab L4 | proxy_val | `QUEUED` |
| i14-04 | [2026-11-26T22-18-i14-04-adaptive-band-tiling-eval](./2026-11-26T22-18-i14-04-adaptive-band-tiling-eval/OUTCOME.md) (Zero-retrain adaptive band-tiling probe; invalidated by `missing=349` ckpt loader + `[-1,1]` norm + T4 FP32 bugs, `0.0000` F1 across all 3 arms) | Kaggle `2× T4` | Wave 11 | `COMPLETED (ITERATE → i14-04b)` |
| i14-04b | [2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed](./2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed/) (Fixed zero-retrain adaptive band-tiling probe: canonical `load_uni_checkpoint` + `missing==0` assert + `[0,1]` norm + BF16/FP16 + 2-page fail-fast gate) | Colab L4 | Wave 11 | `QUEUED` (successor to `i14-04`) |
| i14-14 | [2026-11-26T22-18-i14-14-cascade-ttt-mbr-zoom-chamfer-rerank](./2026-11-26T22-18-i14-14-cascade-ttt-mbr-zoom-chamfer-rerank/OUTCOME.md) (Zero-retrain 3-tier TTT+MBR+zoom+Chamfer cascade on 52p `suite31_hw`; all escalation arms regressed vs `arm0` `0.4418` all / `0.2190` HW / `0.7017` PR at `96.2%` T1 / `94.2%` T2 / `7.59×` latency) | Kaggle `2× T4` | Wave 11 | `COMPLETED (DISCARD)` |
