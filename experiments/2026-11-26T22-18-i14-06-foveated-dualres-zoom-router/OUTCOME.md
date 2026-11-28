# OUTCOME — `2026-11-26T22-18-i14-06-foveated-dualres-zoom-router` (`i14-06`)

- **Status:** `COMPLETED` (`rc=0`, `2/2` arms finished, zero NaNs; Colab session `2026-11-26t22-18-i14-06-` confirmed stopped)
- **Verdict:** **`DISCARD`** *(warm-start silently loaded `0` weights due to missing `"model"` key lookup in `_build_base_model`, collapsing both arms to scratch-trained `0.017–0.030` Note F1 noise floor; even apples-to-apples between the two scratch-trained arms, Arm B worsened `val_ce_loss` (`+0.00101` EMA / `+0.00184` RAW), regressed on both HW slices (`hw_clean -0.0163`, `hw_photo -0.0023` EMA), and slowed training by `1.427×`)*
- **Platform / GPU:** Google Colab — `1× NVIDIA A100-SXM4-40GB` (BF16 native)
- **Started / Finished (UTC):** `2026-11-27T21:04:45Z` → `2026-11-28T00:52:31Z` (`13,665.9 s` wall / `13,482.7 s` script runtime / `3.75 h`)
- **Warm-Start Declared:** `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` *(NOTE: silently failed to load — see Scrutiny §1)*
- **Baseline Commit SHA:** `87b1ec8` (+ 1-line `autocast` fix)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T22-18-i14-06-foveated-dualres-zoom-router.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-26T22-18-i14-06-foveated-dualres-zoom-router.json` (pulled to [`run_hf.json`](run_hf.json))

---

## Measured Results (`arm_a_singlepass_1408_control` vs `arm_b_foveated_dualres_singlepass`, `1,000` steps, BF16, `seed=42`)

| Metric | Eval Mode | `arm_a_singlepass_1408_control` | `arm_b_foveated_dualres_singlepass` | Delta (`B - A` or `B / A`) | Gate (`PLAN.md`) | Pass? |
|---|---|---:|---:|---:|---|---|
| `total_params` (`extra_params`) | — | `41,356,299` (`0`) | `41,504,907` (`+148,608`) | `+0.36%` | — | — |
| `train_loss_tail50` | Train | `1.49002` | `1.49062` | `+0.00060` *(worse)* | lower is better | **FAIL** |
| **`val_ce_loss`** | **EMA** | `1.49360` | `1.49461` | **`+0.00101`** *(worse)* | lower is better | **FAIL** |
| `val_ce_loss` | RAW | `1.47978` | `1.48162` | `+0.00184` *(worse)* | lower is better | **FAIL** |
| **`suite31_hw` `all90` Note F1** | **EMA** | `0.0174` | `0.0299` | **`+0.0125`** *(noise floor)* | `>= +0.020` PROMOTE | **FAIL** |
| `suite31_hw` `printed_clean` Note F1 | EMA | `0.0143` | `0.0415` | `+0.0272` *(noise floor)* | `>= +0.025` PROMOTE | **CONFOUNDED** |
| `suite31_hw` `printed_photo` Note F1 | EMA | `0.0217` | `0.0391` | `+0.0174` *(noise floor)* | higher is better | **CONFOUNDED** |
| `suite31_hw` `hw_clean` Note F1 | EMA | `0.0242` | `0.0079` | **`-0.0163`** *(worse)* | `>= +0.025` PROMOTE | **FAIL** |
| `suite31_hw` `hw_photo` Note F1 | EMA | `0.0080` | `0.0057` | **`-0.0023`** *(worse)* | higher is better | **FAIL** |
| `suite31_hw` `all90` Note F1 | RAW | `0.0229` | `0.0282` | `+0.0053` *(noise floor)* | higher is better | **CONFOUNDED** |
| `suite31_hw` `printed_clean` Note F1 | RAW | `0.0355` | `0.0333` | **`-0.0022`** *(worse)* | higher is better | **FAIL** |
| `suite31_hw` `printed_photo` Note F1 | RAW | `0.0231` | `0.0447` | `+0.0216` *(noise floor)* | higher is better | **CONFOUNDED** |
| `suite31_hw` `hw_clean` Note F1 | RAW | `0.0122` | `0.0081` | **`-0.0041`** *(worse)* | higher is better | **FAIL** |
| `suite31_hw` `hw_photo` Note F1 | RAW | `0.0052` | `0.0008` | **`-0.0044`** *(worse)* | higher is better | **FAIL** |
| `mean_mem_tokens` | EMA / RAW | `2,686.7` / `2,687.5` | `2,990.0` / `2,988.6` | `+303.3` | `<= 3,072` | **PASS** |
| `mean_latency_s` (per page, A100) | EMA / RAW | `27.258 s` / `27.199 s` | `27.549 s` / `27.599 s` | `1.011×` *(MAX_LEN bound)* | `< 1.25×` | **CONFOUNDED** |
| `train_step_ms` *(derived from 1k-step log)* | Train (`ga=4`) | `1,423.3 ms/step` | `2,030.7 ms/step` | **`1.427×`** *(slower)* | — | — |
| `peak_vram_mb` / `gate_pass` | — | *not logged* | *not logged* | — | — | **FAIL** |
| `arm_elapsed_secs` (train + 2× eval) | Total | `6,375.98 s` | `7,063.95 s` | `+687.97 s` (`1.108×`) | — | — |

---

## Scrutiny & Anomaly Check

1. **CRITICAL ANOMALY — Silent Zero-Weight Warm-Start Failure (`step=1 loss=5.705`):**
   - `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` stores its weights under top-level key **`"model"`** (top keys: `['step', 'val_ser', 'ema_decay', 'cfg', 'config', 'vocab', 'model']`) with `vocab_size = 237` (vs `UniOMRv2Config.small_45m()` default `vocab_size = 265`).
   - `_build_base_model()` in [`run_colab.py`](run_colab.py#L496-L500) executed:
     ```python
     state = raw.get("ema_state_dict") or raw.get("model_state_dict") or raw.get("state_dict") or raw
     model.load_state_dict(state, strict=False)
     ```
     Because `"model"` was absent from the fallback chain, `state` resolved to the 7-key metadata dict `raw`, and `strict=False` silently loaded **0 pretrained weights** without warning!
   - **Direct log confirmation:** Both arms started at `step=1/1000 loss=5.7050` (Arm A) and `5.6972` (Arm B) — random initialization (`≈ ln(265) = 5.58`) — and trained for 1,000 steps at a fine-tuning learning rate (`LR_BACKBONE = 3e-5`), converging only to `train_loss = 1.490` / `val_ce = 1.48–1.49` and near-zero `suite31_hw` Note F1 (`0.017–0.030`).
2. **Runaway Eval Latency (`27.2–27.6 s/page` on A100; `3.75 h` total vs `42 min` est.):**
   - Because the scratch-undertrained models rarely emitted `<EOS>`, almost all `90 × 4 = 360` eval decodes ran all the way to `SUITE31_MAX_LEN = 2048` (`~9,860 s` in eval alone), masking real single-pass encoder overhead (`1.011×` decode latency ratio is dominated by 2048-step autoregressive decoding, whereas actual training step time was **`1.427×` slower** on Arm B: `2,030.7` vs `1,423.3 ms/step`).
3. **Domain / Slice Asymmetry & Noise-Floor Confound:**
   - Despite the `+0.0272` `printed_clean` EMA delta appearing to touch the `+0.025` threshold, RAW `printed_clean` moved **negatively** (`0.0355 → 0.0333`, `-0.0022`), `val_ce_loss` worsened on both RAW (`+0.00184`) and EMA (`+0.00101`), and **both handwritten slices regressed** under both RAW and EMA (`hw_clean -0.0163`, `hw_photo -0.0023` EMA; `-0.0041`, `-0.0044` RAW). At `0.01–0.04` F1 this is pure noise-floor variance.
4. **Missing Telemetry Fields:**
   - `eval_hf.json` omitted `peak_vram_mb`, `step_ms`, `n_scored`, `decode_ok_rate`, and `gate_pass`. Future scripts must always log these plus `loaded_keys / total_keys` and assert `step1_loss < 2.0` on warm-started runs.
5. **Compute Hygiene:**
   - Colab A100 session `2026-11-26t22-18-i14-06-` exited cleanly (`rc=0`) and was terminated (`colab sessions` verified empty of `i14-06`).

---

## Verdict & Follow-Up Action

- **Verdict:** **`DISCARD`** — Fails `PLAN.md` §6 gates (`val_ce_loss` worsened `+0.00101` EMA / `+0.00184` RAW, `all90` F1 `+0.0125 < +0.020`, HW regressed across both slices, `1.427×` slower training, and warm-start was invalidated by the silent `"model"` key bug).
- **Follow-Up Action:** Do **not** wire `FoveatedDualResZoomRouter` into `UniOMRv2Net` and do **not** spend further Wave-11 compute re-running `i14-06` on `small_45m`, as the optical-resolution wall is already superseded by the Iter-15 dual-res / band-tiling workstreams (`i15-31` HRNet-Music, `i15-05` `spx13` band-tiling, `pro200m`). Key engineering takeaway preserved: **always include `"model"` in checkpoint state-dict extraction and assert non-zero matched keys + `vocab_size` alignment on warm-start.**
