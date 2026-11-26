# Outcome — `2026-11-26T01-38-ov-20-vir-vico-weight-schedule`

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T10:36:54Z` (started `09:10:02Z`)
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`24 GB`, native `BF16` — note: ledger reports `colab-NVIDIA L4`)
- **Elapsed Time:** `5,211.9 s` (`86.9 min`; `~18.5–19.0 min` per 500-step arm, `peak_vram=9.62–9.63 GB`)
- **Model & Protocol:** `UniOMR v2 small_45m` (`41.19M` total params) + `ViRTwoTierRouter` (`k_coarse=96`, `static_buckets=True`, `dynamic_routing=True`, `max_keep_ratio=0.60`, `saliency_thr=0.05`) + `compute_vico_consistency_loss` (`vico_every=2`, `temp=1.5`, `dense_teacher_keep_ratio=0.85`), `500` steps/arm from scratch (`seed=42`, `Muon+AdamW`, `lr=3e-4`, WSD, `B=2 × GA=4`, `dim=1280`, `max_seq_len=1536`, `VAL_EVAL_MAX_LEN=512`), `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages).
- **HF Artifacts:**
  - `evals/2026-11-26T01-38-ov-20-vir-vico-weight-schedule.json` (pulled locally to [2026-11-26T01-38-ov-20-vir-vico-weight-schedule.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T01-38-ov-20-vir-vico-weight-schedule/2026-11-26T01-38-ov-20-vir-vico-weight-schedule.json))
  - `runs/2026-11-26T01-38-ov-20-vir-vico-weight-schedule.json` (pulled locally to [run_2026-11-26T01-38-ov-20-vir-vico-weight-schedule.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T01-38-ov-20-vir-vico-weight-schedule/run_2026-11-26T01-38-ov-20-vir-vico-weight-schedule.json))

## Measured Results (`proxy_val.jsonl`, 64 Full Pages, EMA Weights)

| Arm | `vico_w` | `keep_ratio` | `vico_step50` (`nz%`) | `eval_vico_kl` nat / `kr025` | `comp_x` (train `k_fine` → eval `k_fine`) | `ink_rec` | `val_ce` (Δ vs `w002`) | `val_token_f1` (Δ) | `val_ser` (Δ) | `tok/s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`vico_w002_kr035`** | **`0.02`** | `0.35` | `0.0462` (`100%`) | **`0.00060`** / `0.00126` | `2.40×` (`1698` → `2032`) | `0.9101` | **`1.6588`** (`0.0000`) | **`0.2385`** (`0.0000`) | **`0.8876`** (`0.0000`) | `237.7` |
| `vico_w008_kr035` | `0.08` | `0.35` | `0.0457` (`100%`) | `0.00063` / **`0.00121`** | `2.42×` (`1698` → `2016`) | `0.9103` | `1.6619` (`+0.0031`) | `0.2254` (`-0.0131`) | `0.8958` (`+0.0082`) | `236.9` |
| `vico_w020_kr035` | `0.20` | `0.35` | `0.0443` (`100%`) | `0.00062` / `0.00131` | `2.40×` (`1698` → `2032`) | `0.9100` | `1.6730` (`+0.0142`) | `0.2268` (`-0.0117`) | `0.8950` (`+0.0075`) | `237.8` |
| `vico_w008_kr025` | `0.08` | `0.25` | `0.0489` (`100%`) | `0.00066` / `0.00126` | `2.42×`* (`1209` → `2016`*) | `0.9103` | `1.6677` (`+0.0089`) | `0.2252` (`-0.0133`) | `0.8958` (`+0.0083`) | `238.7` |

*(Reference: `ov-19` `InkTokenRouterV2` `arm_003_055` on the same dataset reached `val_ce=1.5108` EMA / `1.2546` raw, `ink_recall=0.9461` TF / `0.9716` gen, `comp=2.70×`, `256.4 tok/s`, `1.91 s/step` — vs `2.26 s/step` here.)*

## Scrutiny & Anomaly Check

1. **Exp 3 `vico=0.0000` bug fully fixed & verified:** With `vico_every=2`, all 4 arms fired `250 / 250` active ViCO steps (`vico_nonzero_frac = 1.0`, `vico_step50_val = 0.0443–0.0489 > 0.0`, `mean_k_fine_routed_train < mean_k_fine_dense_train`).
2. **Hypothesis falsified — higher `vico_loss_weight` monotonically hurts `val_ce` and `F1` without lowering `eval_vico_kl`:**
   - `val_ce` degrades monotonically: `1.6588` (`w=0.02`) → `1.6619` (`w=0.08`, `+0.0031`) → `1.6730` (`w=0.20`, `+0.0142`).
   - `val_token_f1` drops `-0.0131` (`w=0.08`) and `-0.0117` (`w=0.20`), and `val_ser` worsens `+0.0082` / `+0.0075`.
   - `eval_vico_kl_native` is already `~0.0006` at `w=0.02` and does **not** decrease at `w=0.08` (`0.00063`) or `w=0.20` (`0.00062`).
   - Worse, the dense-teacher forward pass (`vico_every=2`) adds **`+19%` training step overhead** (`2.26 s/step` vs `1.87 s/step` in `ov-19`) and `ViRTwoTierRouter` decodes slower (`237–239 tok/s` vs `254–256 tok/s`) with lower compression (`2.40×` vs `2.70×`) and lower ink recall (`0.910` vs `0.946`).
3. **Silent eval routing bug on `vico_w008_kr025` (`*`):** During training, `vico_w008_kr025` routed `mean_k_fine_routed_train = 1208.7` (vs `1698.3` for `kr=0.35`). Yet at eval time, `mean_k_fine_routed` jumped to **`2016.0`** (`compression_x_mean = 2.4216`), identical to `vico_w008_kr035`! Because `ViRTwoTierRouter` was constructed with `dynamic_routing=True, max_keep_ratio=0.60, saliency_thr=0.05`, eval-mode adaptive gating ignored `keep_ratio=0.25`.
4. **Short-run confounds (consistent across all 4 arms):** All 4 arms hit the `VAL_EVAL_MAX_LEN = 512` generation cap (`total_pred_tokens = 32,768 = 64 × 512`, `sys_count_exact = 0.1094`) and used `GPUEmaShadow(decay=0.997, update_every=5)` over `500` steps (`100` updates → `74%` init weight), depressing absolute scores equally across arms.

## Verdict: **`DISCARD`**

Per `PLAN.md`'s decision tree (*"If higher `vico_loss_weight` monotonically degrades `val_ce` and `val_token_f1` while failing to reduce `eval_vico_kl`, keep `ViRTwoTierRouter` with `vico_loss_weight=0.0` (router-only) or stick with `InkTokenRouterV2`"*):
- **Discard `ViCO` consistency loss** from Iter-11 training: `w=0.08` and `w=0.20` monotonically hurt `val_ce` and `val_token_f1` while failing to reduce dense-vs-routed KL (`0.00060` vs `0.00063`/`0.00062`) and burning `+19%` training compute.
- **Stick with `InkTokenRouterV2`** over `ViRTwoTierRouter` for Iter-11 (higher ink recall, higher compression, faster decode, and zero dense-teacher overhead).

## Follow-Up Action / Successor Experiment

- None required for `ViCO` (falsified; documented here to prevent re-running). Router optimization continues on `InkTokenRouterV2` via the `ov-19b` warm-started high-recall follow-up.
