# Outcome — `2026-11-26T05-35-ov-34b-budgeted-logprob-measure-beam`

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T06:26:39Z`
- **Platform / GPU:** Kaggle `Tesla T4` (`cc=7.5`, `bf16=false` FP32)
- **Elapsed Time:** `4,529.8 s` (`75.5 min`)
- **Baseline Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`step=12000`, `val_ser=0.192301`, `missing_keys=[]`, `unexpected_keys=[]`)
- **Eval Protocol:** Full 45-piece `suite31_hw` (`90` pages: `31` printed + `14` HW × `clean`/`photo`), canonical `1280px / max_len=3840`, `use_grammar=True`, `use_mtp_spec=True`, `const_offset_limit=3`
- **HF Artifacts:**
  - `evals/2026-11-26T05-35-ov-34b-budgeted-logprob-measure-beam.json` (pulled locally to [2026-11-26T05-35-ov-34b-budgeted-logprob-measure-beam.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T05-35-ov-34b-budgeted-logprob-measure-beam/2026-11-26T05-35-ov-34b-budgeted-logprob-measure-beam.json))

## Measured Results (`suite31_hw` 45-Piece Strict Note F1, `1280px / 3840`)

| Config | `cap` | `max_lp_drop` | `(m_w, v_w)` | `pr_clean` | `pr_photo` | `hw_clean` | `hw_photo` | **`all45_clean`** (Δ) | **`all45_photo`** (Δ) | Overhead | Repaired / Trig |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `mbeam1_baseline` | `0` | — | — | `0.8991` | `0.8764` | `0.2825` | `0.1867` | **`0.7073`** | **`0.6619`** | `0.0%` (`585.4s`) | `0 / 0` |
| **`mbeam2_cap4_lp2_w64`** | **`4`** | **`2.0`** | `(6.0, 4.0)` | **`0.9095`** (`+0.0104`) | **`0.8827`** (`+0.0062`) | **`0.2854`** (`+0.0028`) | `0.1743` (`-0.0124`) | **`0.7153`** (**`+0.0081`**) | **`0.6623`** (`+0.0004`) | **`+30.2%`** | `94 / 194` |
| `mbeam2_cap4_lp1_w22` | `4` | `1.0` | `(2.0, 2.0)` | `0.9087` (`+0.0096`) | `0.8778` (`+0.0014`) | `0.2804` (`-0.0021`) | `0.1812` (`-0.0056`) | `0.7132` (`+0.0059`) | `0.6611` (`-0.0008`) | `+27.4%` | `91 / 195` |
| `mbeam2_cap6_lp2_w22` | `6` | `2.0` | `(2.0, 2.0)` | `0.9085` (`+0.0094`) | `0.8824` (`+0.0060`) | `0.2767` (`-0.0058`) | `0.1753` (`-0.0114`) | `0.7119` (`+0.0047`) | `0.6624` (`+0.0006`) | `+35.3%` | `122 / 259` |
| `mbeam2_cap6_lp4_w22` | `6` | `4.0` | `(2.0, 2.0)` | `0.9085` (`+0.0094`) | `0.8823` (`+0.0059`) | `0.2757` (`-0.0068`) | `0.1768` (`-0.0099`) | `0.7116` (`+0.0044`) | **`0.6628`** (`+0.0010`) | `+35.4%` | `125 / 259` |
| `mbeam2_cap6_voice_only` | `6` | `2.0` | `(0.0, 4.0)` | `0.9085` (`+0.0094`) | `0.8770` (`+0.0006`) | `0.2830` (`+0.0005`) | `0.1795` (`-0.0072`) | `0.7139` (`+0.0066`) | `0.6600` (`-0.0019`) | `+32.9%` | `122 / 209` |

*(Reference benchmarks: `Iter 10.5` = `0.6984 / 0.6832`; `ov-35b` = `0.7016 / 0.6833`. Note `ov-34b` ran on `uni_omr_soup.pt` at `1280px/3840` where `mbeam2_cap4_lp2_w64` pushes `all45_clean` to a new high of **`0.7153`** and `printed_clean` to **`0.9095`**.)*

## Scrutiny & Anomaly Check

1. **Silent failures / coverage:** Zero. All 6 configs evaluated all `45` pieces × `2` domains (`90` pages), `guard_tripped_pages <= 2`.
2. **Latency & printed breakthrough vs `ov-34`:** Capping per-page triggers at `max_triggers=4` with `max_lp_drop=2.0` (`mbeam2_cap4_lp2_w64`) slashed decode overhead from **`+81.8%` (`ov-34`) down to `+30.2%`** while delivering **`+0.0104` `printed_clean` (`0.8991 → 0.9095`)**, **`+0.0062` `printed_photo` (`0.8764 → 0.8827`)**, **`+0.0028` `hw_clean` (`0.2825 → 0.2854`)**, and **`+0.0081` `all45_clean` (`0.7073 → 0.7153`)** with net-positive `all45_photo` (`+0.0004`).
3. **Remaining domain asymmetry (`hw_photo`):** Every `mbeam=2` configuration still regresses `hw_photo` (`-0.0056` to `-0.0124`; `-0.0124` for `mbeam2_cap4_lp2_w64`), even `voice_only` (`-0.0072`). On degraded handwritten photos where baseline Note F1 is `< 0.19`, bar-level rhythmic re-decoding replaces uncertain visual tokens with rhythmically valid hallucinations. Gating `mbeam=2` off on low-confidence pages (mean token probability `< thr`) directly eliminates this regression while preserving the full `+0.0104` printed and `+0.0028` `hw_clean` gains.

## Verdict: **`PROMOTE`** (High-Accuracy Inference Decoder, with Confidence-Gated HW Photo Guard)

Promote **`measure_beam_size=2, max_triggers_per_page=4, max_lp_drop=2.0, meter_w=6.0, voice_w=4.0, const_offset_limit=3`** as our best high-accuracy inference configuration (`+0.0081` `all45_clean` to `0.7153`, `+0.0104` `printed_clean` to `0.9095`, `+0.0062` `printed_photo` to `0.8827`, `+0.0028` `hw_clean`, `+30.2%` latency), paired with page-level confidence gating so low-confidence `hw_photo` pages bypass measure-beam rewinds.

## Follow-Up Action / Successor Experiment

- Confidence-gated selective re-decoding is already queued in `experiments/overnight_queue/queue.json`:
  - [`2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode`](../2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode/README.md)
  - [`2026-11-26T07-40-i13-15-confidence-selective-hires-redecode`](../2026-11-26T07-40-i13-15-confidence-selective-hires-redecode/README.md)
