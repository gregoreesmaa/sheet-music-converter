# OUTCOME — `2026-11-26T22-18-i14-07-retinex-norm-and-clean-router-masks` (`i14-07`)

- **Status:** `COMPLETED` (`rc=0`)
- **Completed At:** `2026-11-28T04:40:04Z` (session stopped cleanly at `2026-11-28T04:40:14Z`)
- **Platform / GPU:** Google Colab **NVIDIA A100-SXM4-40GB** (`BF16`, session `2026-11-26t22-18-i14-07-`)
- **Elapsed Time:** `2,748.22 s` (`45.8 min`) GPU compute (`1,370.73 s` Control / `1,330.51 s` Treatment); `4,913.51 s` total queue wall time (includes `~35.6 min` of Colab A100 `Service Unavailable` retries prior to VM assignment)
- **Warm-Start Checkpoint:** `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` (`small_45m`, `vocab=237`)
- **HF Artifacts:**
  - `evals/2026-11-26T22-18-i14-07-retinex-norm-and-clean-router-masks.json` (local copy: [eval_hf.json](eval_hf.json))
  - `runs/2026-11-26T22-18-i14-07-retinex-norm-and-clean-router-masks.json` (local copy: [run_hf.json](run_hf.json))
- **Verdict:** **`DISCARD`**

---

## Measured Results (2-Arm Controlled Ablation, Seed=42, `600` Steps/Arm, `grad_accum=4`)

Both arms trained from `iter12_v2_best` on `datasets/iter10_5_proxy_2k.tar` (`p=0.65` online synthetic ALS Sepia+Gutter+Bleed, `max_dim=1280`, `max_seq_len=1536`, `0` NaN skips) and evaluated on the stratified `suite31_hw` subset (`12` printed + `14` real HW, clean & photo, `SUITE31_MAX_LEN=2560`, pitch+dur sequence F1) plus deterministic synthetic ALS stress (`26` clean pages):

| Metric | Control (`arm_ctrl_standard_router`) | Treatment (`arm_retinex_clean_router_mask`) | Delta (Treat − Ctrl) | Gate (`PLAN.md`) |
| :--- | ---: | ---: | ---: | :--- |
| **`als_stress_note_f1`** (`n=26`) | **0.4106** | 0.3978 | **-0.0128** (`-1.28 pp`) | `>= +0.040` (**FAIL**) |
| **`hw_photo_note_f1`** (`n=14`) | **0.1409** | 0.1376 | **-0.0033** (`-0.33 pp`) | `>= +0.020` (**FAIL**) |
| **`hw_clean_note_f1`** (`n=14`) | **0.2010** | 0.1715 | **-0.0295** (`-2.95 pp`) | — (**REGRESSION**) |
| **`printed_clean_note_f1`** (`n=12`) | 0.6336 | **0.6704** | **+0.0368** (`+3.68 pp`) | `>= -0.005` (**PASS**) |
| **`printed_photo_note_f1`** (`n=12`) | 0.6550 | **0.6939** | **+0.0389** (`+3.89 pp`) | — (**GAIN**) |
| **`router_notation_recall`** (`n=26`) | 0.8579 | **0.8629** | **+0.0050** (`+0.50 pp`) | `>= +0.050` (**FAIL**) |
| **`gutter_eviction_rate`** (`n=26`) | 0.3374 | **0.5077** | **+0.1703** (`+17.03 pp`) | — (strong gain) |
| `final_ce_loss` (last 20 steps) | **0.2653** | 0.2732 | +0.0079 | — |
| `final_mask_bce` (last 20 steps) | — | 0.1245 (step 600: `0.0157`) | — | — |
| `retinex_params` | 0 | 146,372 (`0.146M`) | +146,372 | `< 0.5M` (**PASS**) |
| `final_retinex_gate_tanh` | 0.0 | **-0.06534** | **-0.06534** | (sign inversion!) |
| `nan_skips` | 0 | 0 | 0 | `0` (**PASS**) |
| `elapsed_sec` (train + eval) | 1,370.73 s | 1,330.51 s | -40.22 s (`-2.9%`) | — |

---

## Scrutiny & Anomaly Check

1. **Coverage & numerical integrity (PASS):** Both arms completed `600/600` optimizer steps with `nan_skips = 0` and scored identical complete buckets (`printed_clean=12`, `printed_photo=12`, `hw_clean=14`, `hw_photo=14`, `als_stress=26`). Colab session terminated cleanly (`colab stop exit_code=0`, verified absent from `colab sessions`).
2. **CRITICAL ANOMALY — Retinex residual gate inverted sign (`tanh(res_gate) = -0.06534`):**
   - `RetinexBackgroundNormalizer` blended via `out = x + tanh(res_gate) * (x / (B + 0.08) - x)`.
   - Across training, `tanh(res_gate)` drifted monotonically **negative** (`step 50: -0.0019` → `step 200: -0.0241` → `step 400: -0.0432` → `step 600: -0.06534`).
   - Because `B(x,y) + 0.08 <= 1.08`, `x_norm > x` on most pixels; a **negative** gate moves *away* from illumination division (slightly darkening/shifting contrast instead of normalizing sepia parchment). An unconstrained `tanh` gate with `LR_GATE=2e-3` (66× `LR_BASE`) fails to enforce the physical Retinex prior.
3. **CRITICAL ANOMALY — Sharp Printed vs. HW domain asymmetry (`printed +3.68 / +3.89 pp` vs. `hw_clean -2.95 pp`, `hw_photo -0.33 pp`, `als_stress -1.28 pp`):**
   - `_compute_clean_ink_mask_grid` defines the supervision target as `(1.0 - luma) > 0.12`.
   - On pure-white synthetic printed renders (`luma ≈ 1.0`), this threshold yields a crisp notation mask — driving the **`+17.03 pp` `gutter_eviction_rate`** jump (`0.3374 → 0.5077`) and **`+3.68 / +3.89 pp`** printed clean/photo Note F1 lift.
   - However, on real archival handwritten manuscripts in `iter10_5_proxy_2k` (whose "clean" source scans already have sepia parchment `luma < 0.88`) and faint-ink HW eval pages (`14/26` of `als_stress`), `(1 - luma) > 0.12` is miscalibrated (marking parchment as ink or causing over-eviction of faint strokes), hurting `hw_clean` (`-2.95 pp`), `hw_photo` (`-0.33 pp`), and `als_stress` (`-1.28 pp`).
4. **Router baseline recall already high (`0.8579`):** Control `iter12_v2_best` already achieved `0.8579` `router_notation_recall` under synthetic ALS stress, leaving limited headroom for the `+0.05` recall gate (`+0.0050` observed). The router's real failure mode on gutters/bleed-through was **false-positive retention** (`gutter_eviction_rate = 0.3374`), not recall.
5. **Apples-to-apples scale note:** Both arms share identical seed, data stream, checkpoint, and `rlvr_grpo.compute_sequence_f1` pitch+dur scorer, so arm-vs-arm deltas are valid. Absolute F1s (`0.6336` control printed clean) are lower than full `suite31_hw` MusicXML strict F1 because of the `600`-step `p=0.65` heavy-ALS fine-tune, `1280px` / `max_len=2560` cap, and sequence-F1 scorer.

---

## Verdict & Follow-Up Action

- **Verdict: `DISCARD`** — Falsified the Theorist 4 hypothesis as implemented: the combined `<0.5M` Retinex normalizer + clean-render router BCE failed every primary ALS/HW gate (`als_stress -1.28 pp`, `hw_photo -0.33 pp`, `hw_clean -2.95 pp`, `router_notation_recall +0.50 pp`). Do **not** wire `RetinexBackgroundNormalizer` into `uni_omr_v2.py`.
- **Recorded negative lessons (do not repeat):**
  1. Never pair an unconstrained `tanh` residual gate (`alpha` able to go negative) with Retinex division — if illumination normalization is ever revisited, constrain `alpha >= 0` (`sigmoid` / `softplus`) or use parameter-free morphology/homomorphic preprocessing.
  2. Never supervise `InkTokenRouterV2` with a raw `(1 - luma) > 0.12` threshold on mixed datasets containing real aged-parchment HW pages — restrict clean-mask router BCE strictly to synthetic white-background renders (`domain == "printed"` / `hw_sim`), where clean-render supervision cleanly boosted gutter eviction (`+17.03 pp`) and printed F1 (`+3.68 / +3.89 pp`).
- **Successor:** None queued (`DISCARD` closes `i14-07`; ALS/HW archival recovery is pursued via `hw-score-sim` sepia/verso data + `i15-05` canonical `s_px=13` band tiling + `i15-42` ink-unmix/cbDice).
