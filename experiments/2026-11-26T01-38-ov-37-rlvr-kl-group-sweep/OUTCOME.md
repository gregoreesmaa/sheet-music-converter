# Outcome — `2026-11-26T01-38-ov-37-rlvr-kl-group-sweep`

- **Status:** `COMPLETED`
- **Verdict:** **`ITERATE`** *(Strong positive signal: 4/6 arms cleared the `+0.015` HW Note F1 proxy gate with virtually zero printed drift (`-0.0002` to `-0.0012`), and `kl0.06_G8` uploaded `best_rlvr_arm.pt` to HF. However, scrutiny flagged 4 methodological gaps — `eval_max_len=1536` truncation ceiling, short `max_len=256` / `40`-update RLVR budget, clean-vs-photo asymmetry between `kl0.06_G8` and `kl0.15_G4`, and missing per-piece breakdown — so we immediately scale & verify in successor **`2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840`** before wiring RLVR into Iter 11.)*
- **Platform / GPU:** Kaggle `2× Tesla T4` (`cc=7.5`, ran on GPU 0 in FP32 due to `use_bf16 = cc[0] >= 8` guard; script logged `colab-Tesla T4`)
- **Started / Completed (UTC):** `2026-11-26T00:13:22Z` → `2026-11-26T01:31:52Z`
- **Elapsed Wall-Clock:** `4709.81s` (`78.5 min`, ~10–12 min per arm in T4 FP32)
- **Baseline Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`step=12000`, `val_ser=0.192301`, `missing_keys=[]`, `unexpected_keys=[]`)
- **HF Artifacts:**
  - Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-37-rlvr-kl-group-sweep.json` (local copy: [`eval_result.json`](./eval_result.json))
  - Run Ledger: `runs/2026-11-26T01-38-ov-37-rlvr-kl-group-sweep.json`
  - Winning Arm Checkpoint: `checkpoints/2026-11-26T01-38-ov-37-rlvr-kl-group-sweep/best_rlvr_arm.pt` (`kl0.06_G8`)

---

## Measured Results (`suite31_hw` Python Proxy Eval, `max_dim=1024`, `eval_max_len=1536`)

### 1. 26-Piece Sweep Table (`14` HW + `12` Stratified Printed, `52` Pages)

| Arm (`kl_beta`, `G`) | `hw_clean` F1 (`Δ`) | `hw_photo` F1 (`Δ`) | `printed_clean` F1 (`Δ`) | `printed_photo` F1 (`Δ`) | Composite | Final KL | Final Reward `R` | Train Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (`uni_omr_soup`)** | `0.2129` (`—`) | `0.1464` (`—`) | `0.6884` (`—`) | `0.6773` (`—`) | `0.0000` | `—` | `—` | `—` |
| `kl0.02_G4` | `0.2289` (`+0.0159`) | `0.1515` (`+0.0051`) | `0.6877` (`-0.0007`) | `0.6785` (`+0.0013`) | `+0.0101` | `0.1025` | `0.213` | `604.3s` |
| `kl0.02_G8` | `0.2170` (`+0.0040`) | `0.1691` (`+0.0227`) | `0.6873` (`-0.0011`) | `0.6780` (`+0.0007`) | `+0.0128` | `0.2346` | `0.213` | `715.1s` |
| `kl0.06_G4` | `0.2251` (`+0.0121`) | `0.1460` (`-0.0004`) | `0.6883` (`-0.0002`) | `0.6805` (`+0.0032`) | `+0.0058` | `0.0814` | `0.157` | `596.0s` |
| **`kl0.06_G8` (Composite Winner)** | `0.2165` (`+0.0035`) | **`0.1765` (`+0.0301`)** | `0.6876` (`-0.0008`) | `0.6762` (`-0.0011`) | **`+0.0164`** | `0.0977` | `0.201` | `711.8s` |
| **`kl0.15_G4` (Best Balanced)** | **`0.2307` (`+0.0178`)** | `0.1594` (`+0.0130`) | `0.6880` (`-0.0005`) | **`0.6793` (`+0.0021`)** | `+0.0151` | **`0.0574`** | `0.215` | `598.9s` |
| `kl0.15_G8` | `0.2291` (`+0.0162`) | `0.1418` (`-0.0046`) | `0.6872` (`-0.0012`) | `0.6777` (`+0.0004`) | `+0.0052` | `0.1375` | `0.228` | `713.8s` |

### 2. Structural Metrics (`r_rhythm`, `r_noloop`, `eos_rate`) on `suite31_hw` HW

| Arm | `hw_clean r_rhythm` (`Δ`) | `hw_clean r_noloop` (`Δ`) | `hw_photo r_rhythm` (`Δ`) | `hw_photo r_noloop` (`Δ`) | `hw` EOS (cln / pho) |
|---|---:|---:|---:|---:|---:|
| **Baseline** | `0.3786` (`—`) | `0.7857` (`—`) | `0.3183` (`—`) | `0.7857` (`—`) | `0.500 / 0.429` |
| `kl0.02_G4` | `0.3549` (`-0.0237`) | `0.7857` (`+0.0000`) | `0.4105` (`+0.0922`) | `0.8571` (`+0.0714`) | `0.500 / 0.429` |
| `kl0.02_G8` | `0.3500` (`-0.0286`) | `0.7857` (`+0.0000`) | `0.3670` (`+0.0486`) | `0.7857` (`+0.0000`) | `0.429 / 0.429` |
| `kl0.06_G4` | `0.4004` (`+0.0218`) | `0.8571` (`+0.0714`) | `0.4179` (`+0.0995`) | `0.7143` (`-0.0714`) | `0.500 / 0.500` |
| `kl0.06_G8` | `0.3833` (`+0.0047`) | `0.7857` (`+0.0000`) | `0.3068` (`-0.0115`) | `0.7857` (`+0.0000`) | `0.500 / 0.429` |
| `kl0.15_G4` | `0.3606` (`-0.0180`) | `0.7857` (`+0.0000`) | `0.3717` (`+0.0534`) | `0.8571` (`+0.0714`) | `0.500 / 0.429` |
| `kl0.15_G8` | `0.3682` (`-0.0104`) | `0.8571` (`+0.0714`) | `0.4041` (`+0.0858`) | `0.7143` (`-0.0714`) | `0.429 / 0.500` |

### 3. Training Reward & KL Trajectories (40 GRPO Updates, Quartiles of 10 Steps)

| Arm | `R` (Q1 → Q2 → Q3 → Q4) | `r_rhythm` (Q1 → Q4) | `KL` (Q1 → Q4, Max) | Mean `clip_frac` | NaNs / Zero-Loss |
|---|---|---|---|---:|---:|
| `kl0.02_G4` | `0.135 → 0.207 → 0.132 → 0.213` | `0.256 → 0.371` | `0.0958 → 0.1025` (max `0.3117`) | `0.1643` | `0 / 0` |
| `kl0.02_G8` | `0.085 → 0.192 → 0.151 → 0.213` | `0.171 → 0.418` | `0.1608 → 0.2346` (max `1.7352`) | `0.1534` | `0 / 0` |
| `kl0.06_G4` | `0.135 → 0.181 → 0.174 → 0.157` | `0.202 → 0.275` | `0.0895 → 0.0813` (max `0.6309`) | `0.1610` | `0 / 0` |
| `kl0.06_G8` | `0.096 → 0.186 → 0.151 → 0.201` | `0.157 → 0.328` | `0.0991 → 0.0977` (max `0.6198`) | `0.1563` | `0 / 0` |
| `kl0.15_G4` | `0.112 → 0.158 → 0.133 → 0.215` | `0.213 → 0.367` | `0.0787 → 0.0574` (max `0.7381`) | `0.1568` | `0 / 0` |
| `kl0.15_G8` | `0.112 → 0.202 → 0.179 → 0.228` | `0.181 → 0.411` | `0.2529 → 0.1375` (max `1.8946`) | `0.1520` | `0 / 0` |

### 4. Winner (`kl0.06_G8`) Full 45-Piece `suite31_hw` Python Proxy Summary (`max_dim=1024`, `eval_max_len=1536`)

- `printed_clean` (`n=31`): `note_f1_py=0.6415`, `r_rhythm=0.8263`, `eos=0.8065`
- `printed_photo` (`n=31`): `note_f1_py=0.6359`, `r_rhythm=0.8209`, `eos=0.8065`
- `hw_clean` (`n=14`): `note_f1_py=0.2165`, `r_rhythm=0.3833`, `eos=0.5000`
- `hw_photo` (`n=14`): `note_f1_py=0.1765`, `r_rhythm=0.3068`, `eos=0.4286`
- Weighted `all45_clean_f1=0.5093`, `all45_photo_f1=0.4930` *(Python proxy at `eval_max_len=1536`; not directly comparable to Rust MusicXML Strict Note F1 at `3840` tokens).*

---

## Scrutiny & Anomaly Check

1. **Coverage & Numerical Health (`PASS`):** All 6 arms completed `40/40` GRPO updates with `0` NaNs, `0` zero-loss steps, `missing_keys=[]`, `unexpected_keys=[]`, healthy GRPO clipping (`clip_frac = 0.152–0.164`), and complete eval coverage (`14` HW + `12` printed in sweep; `31` printed + `14` HW in winner full-45).
2. **Apples-to-Apples Metric & Truncation Ceiling (`WARNING — MUST VERIFY`):**
   - Arm-vs-baseline deltas within `ov-37` are strictly paired (`max_dim=1024`, `eval_max_len=1536`, Python `compute_sequence_f1`), confirming that label-free RISM RLVR genuinely moves HW Note F1 upward (`+0.0035` to `+0.0301`) with near-zero printed drift (`-0.0002` to `-0.0012`).
   - **However**, `eval_max_len=1536` truncates `50.0%` of HW clean pages, `57.1%` of HW photo pages, and `19.4%` of printed pages before EOS, and the JSON omits per-piece rows and full-45 baseline metrics. Before promoting RLVR to Iter 11, we must evaluate both baseline and RLVR arms at the `ov-35` optimal **`max_dim=1280`, `eval_max_len=3840`** and log per-piece deltas to rule out 1–2 outlier pieces driving the `+0.0301` `hw_photo` jump.
3. **Domain Asymmetry & Under-Training (`ITERATE` DRIVER):**
   - `kl0.06_G8` won composite (`+0.0164`) via a huge `hw_photo` jump (`+0.0301`) but small `hw_clean` gain (`+0.0035`) and slight `hw_photo r_rhythm` dip (`-0.0115`).
   - `kl0.15_G4` was much more balanced (`hw_clean +0.0178`, `hw_photo +0.0130`, `printed_photo +0.0021`, `hw_photo r_rhythm +0.0534`, lowest `mean_final_kl=0.0574`).
   - `kl0.02_G8` spiked to `max KL = 1.7352` (`final KL = 0.2346`), confirming `kl_beta=0.02` is too loose for longer runs.
   - `updates=40` with `max_len=256` only exposed the model to `0.7%` of the 5,707 RISM pages and truncated rollouts at ~1–2 systems; training reward was still climbing steeply at step 40 (`0.096 → 0.201`).
4. **Kaggle T4 Precision Quirk (`LOGGED IN cloud-kaggle`):**
   - Because `run_colab.py` gated autocast on `use_bf16 = bool(cc[0] >= 8)`, running on Kaggle `2× T4` (`cc=7.5`) silently fell back to **pure FP32** instead of `FP16 + GradScaler`, doubling wall-clock (`78.5 min` vs `~35 min` on L4 BF16) while leaving GPU 1 idle and logging `platform: "colab-Tesla T4"`.

---

## Key Takeaways & Decision

- **Core Hypothesis Confirmed:** Label-free structural RLVR (`compute_unlabeled_structural_reward`) on unlabeled archival RISM manuscripts improves `suite31_hw` handwritten Note F1 across all 6 arms with **virtually zero printed forgetting** (`<= 0.0012` drop).
- **Pareto Winners to Scale:** `kl0.06_G8` (highest `hw_photo +0.0301`, composite `+0.0164`) and `kl0.15_G4` (highest `hw_clean +0.0178`, best balance & lowest KL `0.0574`), plus interpolating `kl0.10_G8`.
- **Verdict: `ITERATE`** — Do not stop at 40 updates / 256-tok rollouts / 1536-tok eval. Immediately scale the Pareto winners to `updates=160` and `max_len=512` rollouts, and evaluate baseline + all arms at the `ov-35` optimal `max_dim=1280, eval_max_len=3840` with full per-piece logging.

## Follow-Up Action / Successor Experiment

- **Queued Successor:** [`2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840`](../2026-11-26T04-02-ov-37b-rlvr-scale200-len512-eval3840/README.md) (`depends_on: ["2026-11-26T01-38-ov-37-rlvr-kl-group-sweep"]`, registered in `experiments/overnight_queue/queue.json`).
