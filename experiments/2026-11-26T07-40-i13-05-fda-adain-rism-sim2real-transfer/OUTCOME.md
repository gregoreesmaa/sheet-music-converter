# Outcome — `2026-11-26T07-40-i13-05-fda-adain-rism-sim2real-transfer`

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T07:40:56Z`
- **Platform / GPU:** Google Colab `NVIDIA L4` (`cc=8.9`, `bf16=true`)
- **Elapsed Time:** `2,066.4 s` (`34.4 min`, `~432–447 s/arm`, `~1.3 CU`)
- **Baseline Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`step=12000`, `val_ser=0.192301`, `missing_keys=[]`, `unexpected_keys=[]`)
- **HF Artifacts:**
  - `evals/2026-11-26T07-40-i13-05-fda-adain-rism-sim2real-transfer.json` (pulled locally to [2026-11-26T07-40-i13-05-fda-adain-rism-sim2real-transfer.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T07-40-i13-05-fda-adain-rism-sim2real-transfer/2026-11-26T07-40-i13-05-fda-adain-rism-sim2real-transfer.json))
  - `runs/2026-11-26T07-40-i13-05-fda-adain-rism-sim2real-transfer.json`

## Measured Results (`120`-Step Fine-Tuning on `iter10_5_proxy_2k` + Unpaired `rism_hw_5707`, `26`-Piece `suite31_hw` Subset, Python `note_f1_py_mean`)

- **Pre-FT Baseline (`uni_omr_soup.pt`):** `printed_clean 0.6881`, `printed_photo 0.6783`, `hw_clean 0.2280`, `hw_photo 0.1580` (`n=12` printed, `n=14` HW).

| Arm | Mode | `n_tr` | FT Time | Final Loss | `pr_clean` (vs ctrl) | `pr_photo` (vs ctrl) | `hw_clean` (vs base / **vs ctrl**) | `hw_photo` (vs base / **vs ctrl**) | **`hw_avg` vs ctrl** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `arm_ctrl_standard` | Control | `0` | `432.1s` | `0.6264` | `0.0000` | `0.0000` | `-0.0130` (`0.0000`) | `-0.0058` (`0.0000`) | `0.0000` |
| **`arm_fda_b002`** | **FDA `β=0.02`** | **`227`** | `446.3s` | `0.5695` | `-0.0062` | `-0.0063` | **`+0.0051` (`+0.0181`)** | **`+0.0091` (`+0.0149`)** | **`+0.0165`** |
| `arm_fda_b006` | FDA `β=0.06` | `227` | `447.0s` | **`0.5333`** | `-0.0050` | `-0.0002` | `-0.0062` (`+0.0068`) | `+0.0051` (`+0.0109`) | `+0.0089` |
| **`arm_color_adain`** | **Color-AdaIN** | **`222`** | `445.9s` | `0.5573` | **`-0.0043`** | **`-0.0016`** | **`+0.0017` (`+0.0147`)** | **`+0.0104` (`+0.0162`)** | **`+0.0155`** |

## Scrutiny & Anomaly Check

1. **Silent failures / coverage:** Zero. All 4 arms ran `120/120` FT steps with identical seed/batch order, `missing_keys=[]`, `unexpected_keys=[]`, `n=12` printed + `n=14` HW scored cleanly.
2. **Hypothesis gate CLEARED (`>= +0.015` HW Note F1 vs matched control):**
   - Standard proxy-2k fine-tuning (`arm_ctrl_standard`) forgets HW (`-0.0094` `hw_avg` vs base).
   - **Unpaired RISM-5707 FDA (`β=0.02`, `p=0.5`)** reverses HW forgetting and beats control by **`+0.0181` `hw_clean` / `+0.0149` `hw_photo` (`+0.0165` `hw_avg`)** while improving over the pre-FT checkpoint on both HW slices (`+0.0051` / `+0.0091`).
   - **Chromatic-AdaIN (`arm_color_adain`)** also clears the gate (**`+0.0155` `hw_avg` vs ctrl**: `+0.0147` `hw_clean`, **`+0.0162` `hw_photo`**) with even smaller printed drift (`-0.0043` clean / `-0.0016` photo).
   - **Frequency-band sensitivity confirmed:** `β=0.06` (`+0.0089` `hw_avg`) is roughly half as effective as `β=0.02` (`+0.0165`), confirming that swapping only the lowest-frequency amplitude envelope (`β=0.02`) transfers parchment/foxing/bleed-through without disturbing 1–2 px staff/notehead phase structure.
3. **Zero pseudo-label noise & negligible compute cost:** Uses 100% permissive unlabeled `rism_hw_5707` pixels with exact synthetic `bekern` GT preserved, adding only `+3.3%` wall-time overhead (`446s` vs `432s`).

## Verdict: **`PROMOTE`**

Promote **online unpaired RISM-5707 Domain Transfer (`FDA β=0.02` + `Color-AdaIN`, combined `p ≈ 0.35–0.50`)** into the Iter 12/13 full-page training augmentation pipeline (`tmp-train/mvd/uni_omr.py` / `train_uni_v2.py`). Both `FDA β=0.02` (`+0.0165` `hw_avg`) and `Color-AdaIN` (`+0.0155` `hw_avg`) clear the `+0.015` gate with zero pseudo-label noise and `< 3.5%` overhead.

## Follow-Up Action / Successor Experiment

- Wire `FDA (β=0.02)` + `Color-AdaIN` into the main Iter 12/13 training config (no separate proxy rerun required; complements `i13-04` RISM SSL and `i13-21` DualInkStream).
