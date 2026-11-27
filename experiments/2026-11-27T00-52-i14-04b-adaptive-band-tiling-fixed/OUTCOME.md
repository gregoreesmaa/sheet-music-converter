# OUTCOME — `2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed` (`i14-04b`)

- **Experiment ID:** `2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed`
- **Short ID / Wave:** `i14-04b` (Wave 11 — Optical Resolution & Token-Cap Wall; fixed rerun of `i14-04`)
- **Status:** `COMPLETED` (`rc=0`)
- **Verdict:** **`DISCARD`** (Zero-retrain adaptive band-tiling fails the dense-subset gate by `-0.1266` Note F1 vs `-0.01` discard threshold)
- **Platform / GPU:** Colab `1× NVIDIA A100-SXM4-40GB` (`amp_dtype=torch.bfloat16`, session `2026-11-27t00-52-i14-04b`, terminated cleanly)
- **Checkpoint:** `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` (`arch=uni_omr_v2`, `step=20000`, `missing_keys=0`, `unexpected_keys=0`)
- **Timestamps (UTC):** Started `2026-11-27T12:24:53Z` → Completed `2026-11-27T14:11:17Z` (session wall-clock `11:41:09Z` → `14:11:55Z`)
- **Elapsed Time:** `6383.9 s` inner eval (`Arm A: 1770.09 s`, `Arm B: 1864.54 s`, `Arm C: 2665.24 s`; `9046.56 s` total Colab wall-clock)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed.json` → local [`eval_hf.json`](./eval_hf.json)
  - Run Ledger: `runs/2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed.json` → local [`run_hf.json`](./run_hf.json)
  - Local Log: `D:\tmp\overnight-queue-logs\2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed.log`

---

## Measured Results (`suite31_hw`, 90 pages = 31 printed + 14 HW × `{clean, photo}`)

| Slice | N | Arm A (`1408px` control) F1 / `dec_ok` / `ink_rec` / `lat_s` | Arm B (`1792px` single-pass) F1 / `dec_ok` / `ink_rec` / `lat_s` | Arm C (Adaptive Band-Tiling) F1 / `dec_ok` / `ink_rec` / `lat_s` | Δ (B − A) F1 | Δ (C − A) F1 | Gate |
|---|---:|---|---|---|---|---|---|
| **`all90`** | 90 | `0.4924` / `0.8222` / `0.9970` / `19.501s` | **`0.5009`** / `0.8000` / `0.9969` / `20.546s` | `0.4108` / `0.6222` / `0.9955` / `29.433s` | `+0.0085` | **`-0.0816`** | **FAIL** (`>= +0.01`) |
| **`printed` (all)** | 62 | `0.6535` / `0.8548` / `0.9994` / `13.508s` | **`0.6709`** / `0.8387` / `0.9993` / `13.407s` | `0.5395` / `0.6613` / `0.9993` / `24.981s` | `+0.0174` | **`-0.1140`** | — |
| `printed_clean` | 31 | `0.6524` / `0.8710` / `0.9997` / `13.577s` | **`0.6686`** / `0.8065` / `0.9994` / `13.495s` | `0.5624` / `0.7419` / `0.9996` / `22.970s` | `+0.0162` | `-0.0900` | — |
| `printed_photo` | 31 | `0.6547` / `0.8387` / `0.9991` / `13.439s` | **`0.6733`** / `0.8710` / `0.9992` / `13.319s` | `0.5166` / `0.5806` / `0.9991` / `26.993s` | `+0.0186` | `-0.1381` | — |
| **`hw` (all)** | 28 | **`0.1355`** / `0.7500` / `0.9918` / `32.770s` | `0.1245` / `0.7143` / `0.9916` / `36.355s` | `0.1258` / `0.5357` / `0.9870` / `39.289s` | `-0.0110` | **`-0.0097`** | — |
| `hw_clean` | 14 | **`0.1469`** / `0.7857` / `0.9864` / `35.556s` | `0.1351` / `0.7143` / `0.9860` / `37.374s` | `0.1335` / `0.4286` / `0.9768` / `47.422s` | `-0.0118` | `-0.0134` | — |
| `hw_photo` | 14 | **`0.1240`** / `0.7143` / `0.9971` / `29.984s` | `0.1138` / `0.7143` / `0.9972` / `35.336s` | `0.1181` / `0.6429` / `0.9972` / `31.156s` | `-0.0102` | `-0.0059` | — |
| **`dense_subset`** | 58 | `0.4735` / `0.7931` / `0.9960` / `20.845s` | **`0.4901`** / `0.7759` / `0.9957` / `20.622s` | `0.3469` / `0.4828` / `0.9937` / `36.464s` | `+0.0166` | **`-0.1266`** | **FAIL** (`>= +0.03`, DISCARD `< -0.01`) |
| **`sparse_subset`** | 32 | `0.5265` / `0.8750` / `0.9988` / `17.064s` | `0.5207` / `0.8438` / `0.9990` / `20.410s` | `0.5265` / `0.8750` / `0.9988` / `16.689s` | `-0.0058` | `0.0000` | **PASS** (`>= -0.005`) |

- **Dense latency ratio (Arm C / Arm A):** `36.464s / 20.845s = 1.75×` (within `< 2.2×`, but at a catastrophic `-0.1266` F1 and `-0.3103` `dec_ok` cost).

---

## Scrutiny & Anomaly Check

1. **Harness Validity Confirmed (Zero Silent Failures):**
   - All four `i14-04` harness bugs were resolved: canonical `load_uni_checkpoint` loaded `iter12_v2_best` cleanly (`missing_keys=0, unexpected_keys=0`), `[0, 1]` normalization and native `bfloat16` on A100 ran cleanly, the 2-page sanity check passed, and all 90 pages (`58 dense + 32 sparse`) completed with `rc=0` and zero NaNs.
   - `sparse_subset` is exact bit-for-bit parity between Arm A and Arm C (`0.5265` F1, `0.8750` `dec_ok`, `0.9988` `ink_recall`), verifying that Arm C cleanly passed sparse pages through unchanged.
   - Colab session `2026-11-27t00-52-i14-04b` terminated cleanly (`colab-stop exit_code=0`).

2. **Root Cause of Arm C Collapse (`-0.1266` dense F1, `-0.3103` dense `dec_ok`):**
   - **Frozen full-page model fails on 2–5 system crops + stitching breaks syntax:** Even on clean printed pages where Arm A scores `1.0000` (`fiddle-strathspey` clean: `1.0000 → 0.8319` at 3 tiles; `joplin-maple-leaf` photo: `1.0000 → 0.6993` at 3 tiles, `dec_ok: 0.0`; `beethoven-grosse-fuge` clean: `0.3254 → 0.1310` at 3 tiles; `mozart-k155-2` clean: `0.3619 → 0.2430`, `dec_ok: 1.0 → 0.0`), independently decoding horizontal crops and stitching `**kern` fragments destroys spine/clef/barline consistency (`dense_subset` `dec_ok` drops from `0.7931` to `0.4828`).
   - **Photo noise inflates system-count heuristic:** On `per_page_sample`, `bach-bwv66.6` is classified sparse on clean (`n_systems_est=2`, 1 tile) but dense on photo (`n_systems_est=9`, 3 tiles); `joplin-maple-leaf` is sparse on clean (`5`, 1 tile) vs dense on photo (`9`, 3 tiles, F1 drops `1.0000 → 0.6993`); `beethoven-grosse-fuge` jumps from `8` (3 tiles) to `14` (5 tiles). This explains why `printed_photo` regressed even worse than `printed_clean` (`-0.1381` vs `-0.0900`).
   - **Router `ink_recall` was not the bottleneck:** Arm A already achieves `0.9960` `ink_recall` on `dense_subset` (`0.9997` on `printed_clean`, `0.9991` on `printed_photo`), and Arm C slightly lowers it (`0.9937`).

3. **Secondary Insight — Domain Asymmetry in Arm B (`1792px` Single-Pass):**
   - Uniform `1792px` single-pass improves `printed` by **`+0.0174` F1** (`0.6535 → 0.6709`; `printed_clean +0.0162`, `printed_photo +0.0186`, `dense_subset +0.0166`) at essentially identical latency (`13.41s vs 13.51s` on printed; `20.62s vs 20.85s` on dense). Individual dense printed pieces see large wins (`schoenberg-op19-2` clean `0.7228 → 0.8442`, photo `0.7426 → 0.8812`; `mozart-k155-2` photo `0.1673 → 0.3619` with `dec_ok 0.0 → 1.0`).
   - However, `1792px` regresses **`hw` by `-0.0110` F1** (`0.1355 → 0.1245`, `dec_ok 0.7500 → 0.7143`, latency `+3.59s`), netting `+0.0085` on `all90` (`0.4924 → 0.5009`) with `dec_ok` slipping `0.8222 → 0.8000`. Higher resolution helps dense printed engraving without retraining, but OOD historical handwriting degrades when pushed past training resolution.

---

## Verdict: **`DISCARD`**

Zero-retrain adaptive band-tiling (Arm C) is **falsified and discarded** per `PLAN.md` §6 (`dense_subset` Δ = `-0.1266`, well below the `< -0.01` discard threshold). A frozen full-page UniOMR v2 model cannot generalize to 2–5 system horizontal band crops + post-hoc `**kern` stitching without crop-aware training; it slashes `dense_subset` Note F1 by `-0.1266`, drops `dec_ok` from `79.3%` to `48.3%`, and adds `1.75×` latency.

---

## Follow-Up Action / Successor Experiment

- **No `i14-04c` queued** (hypothesis cleanly falsified).
- Rely on single-pass capacity and architectural resolution scaling already in the pipeline:
  - **`i14-05` (`edge85m`) / `i15-40` (`pro200m`):** Higher model capacity without fracturing full-page context.
  - **`i14-06` (Single-Pass Foveated Zoom Router) & `i15-31` (HRNet-Music DualRes /4‖/8):** Preserve full-page global context while giving dense regions higher effective spatial resolution natively in training.
  - Keep `1792px` printed-only inference note in mind for printed-specialist evaluation, but do not raise global default inference resolution above `1408px` without HW-balanced high-res training.
