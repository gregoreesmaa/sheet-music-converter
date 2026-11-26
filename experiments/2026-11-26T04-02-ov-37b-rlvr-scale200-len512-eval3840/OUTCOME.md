# Outcome — `2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840`

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T05:45:45Z`
- **Platform / GPU:** Kaggle `Tesla T4` (`cc=7.5`, FP16 + `GradScaler`)
- **Elapsed Time:** `4,791.7 s` (`79.9 min`)
- **Baseline Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`step=12000`, `val_ser=0.192301`, `missing_keys=[]`, `unexpected_keys=[]`)
- **HF Artifacts:**
  - `evals/2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840.json` (pulled locally to [2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840/2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840.json))

## Measured Results (`updates=160`, `max_len=512`, `eval_max_dim=1280`, `eval_max_len=3840`, Python `note_f1_py_mean`)

### 26-Piece Sweep Subset (`12` printed + `14` HW)

| Arm | `kl_beta` | `G` | `printed_clean` (Δ) | `printed_photo` (Δ) | `hw_clean` (Δ) | `hw_photo` (Δ) | HW Wins (`cln`/`ph`) | Final KL | Composite |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Baseline (`uni_omr_soup.pt`) | — | — | `0.7062` | `0.6017` | `0.2599` | `0.1795` | — | — | `0.0000` |
| `kl0.06_G8_u160` | `0.06` | `8` | `-0.0237` | `-0.0230` | `-0.0003` | `-0.0054` | `6/14`, `6/14` | `0.0735` | `-0.0147` |
| `kl0.15_G4_u160` | `0.15` | `4` | `-0.0321` | `+0.0082` | `+0.0039` | `-0.0123` | `7/14`, `6/14` | `0.0620` | `-0.0202` |
| **`kl0.10_G8_u160`** | **`0.10`** | **`8`** | **`0.7065` (`+0.0002`)** | `0.5920` (`-0.0097`) | **`0.2749` (`+0.0150`)** | **`0.1901` (`+0.0106`)** | **`9/14`, `7/14`** | `0.0598` | **`+0.0128`** |

### Winner `kl0.10_G8_u160` on Full 45-Piece `suite31_hw` (Python `note_f1_py_mean`)

- `printed_clean` (`n=31`): `0.6596` (`r_parse=0.8387`, `r_rhythm=0.8330`, `eos_rate=1.000`)
- `printed_photo` (`n=31`): `0.6144` (`r_parse=0.8387`, `r_rhythm=0.8067`, `eos_rate=1.000`)
- `hw_clean` (`n=14`): **`0.2749`** (`+0.0150` vs baseline `0.2599`; `eos_rate` improved **`0.8571 → 1.0000`**, `r_noloop` `0.7857 → 0.8571`)
- `hw_photo` (`n=14`): **`0.1901`** (`+0.0106` vs baseline `0.1795`; `r_parse` improved **`0.7500 → 0.8571`**)

## Scrutiny & Anomaly Check

1. **Silent failures / coverage:** Zero. All 3 arms completed `160` updates on Kaggle T4, all `26` sweep pieces + `45` winner pieces scored with zero NaNs and `missing_keys=[]`.
2. **Apples-to-apples metric note:** `ov-37b` uses the fast Python `note_f1_py_mean` scorer (not the Rust `eval-harness` strict Note F1 where `Iter 10.5` is `0.6984/0.6832` and `ov-35b` is `0.7016/0.6833`), so deltas are strictly intra-run against the identical `1280px/3840` baseline.
3. **Standard GRPO scaling ceiling & KL sensitivity:** Scaling from `40` (`ov-37`) to `160` updates did **not** scale HW gains past `+0.015`:
   - `kl0.06_G8` and `kl0.15_G4` both over-drifted (`printed_clean -0.0237` and `-0.0321`, `hw_photo` negative).
   - `kl0.10_G8_u160` is a genuine sweet spot (`+0.0150` `hw_clean` across `9/14` pieces, `+0.0106` `hw_photo`, `+0.0002` `printed_clean`, `hw_clean eos_rate 0.857 → 1.000`), but `printed_photo` dipped `-0.0097` and `hw_clean r_rhythm` **dropped `-0.0382`** (`0.5227 → 0.4845`), indicating length-normalization / EOS-shortcut bias in vanilla GRPO.
4. **Promote gate check:** Missed the `>= +0.025` `hw_clean` and `hw_photo` promote gate (`+0.0150 / +0.0106`), so it does not promote as-is.

## Verdict: **`ITERATE`**

Label-free RISM RLVR (`kl=0.10, G=8`) delivers a real, multi-piece (`9/14`) `+0.0150` `hw_clean` / `+0.0106` `hw_photo` lift with zero `printed_clean` drift, but vanilla GRPO plateaus at `+0.015` and degrades `r_rhythm` (`-0.0382`) under 160 updates. Moving from vanilla GRPO to **DAPO Clip-Higher + Dr.GRPO unbiased length normalization** and **per-measure process-supervised GRPO** directly targets both the entropy collapse and the rhythm-vs-EOS reward shortcut.

## Follow-Up Action / Successor Experiment

- Already queued in `experiments/overnight_queue/queue.json`:
  - [`2026-11-26T07-40-i13-07-dapo-drgrpo-rlvr-rism`](../2026-11-26T07-40-i13-07-dapo-drgrpo-rlvr-rism/README.md) (DAPO Clip-Higher + Dr.GRPO unbiased length normalization on RISM-5707)
  - [`2026-11-26T07-40-i13-23-per-measure-process-grpo-curriculum`](../2026-11-26T07-40-i13-23-per-measure-process-grpo-curriculum/README.md) (Per-measure process-supervised GRPO curriculum)
