# OUTCOME — `2026-11-26T07-40-i13-06-memo-sar-rhythm-gated-ttt` (`i13-06`)

- **Status:** `COMPLETED`
- **Completed UTC:** `2026-11-26T08:56:26Z` (started `08:21:45Z`)
- **Platform / GPU:** `colab-NVIDIA L4` (`bf16=True`)
- **Elapsed Time:** `2081.57 s` (`34.7 min`)
- **Base Checkpoint:** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`uni_omr_v1`, step `12000`, `val_ser=0.192301`, `missing_keys=[]`, `unexpected_keys=[]`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - `evals/2026-11-26T07-40-i13-06-memo-sar-rhythm-gated-ttt.json`
  - `runs/2026-11-26T07-40-i13-06-memo-sar-rhythm-gated-ttt.json`

## Measured Results (`max_dim=1280`, `max_len=3840`, MTP-4 + FSA)

### Handwritten Manuscripts (`suite31_hw`, `n=14` clean + `n=14` photo)

| Arm | `hw_clean` F1 | `hw_clean` Δ (W/T/L) | `hw_photo` F1 | `hw_photo` Δ (W/T/L) | HW Avg Δ |
|---|---:|---:|---:|---:|---:|
| `baseline` | `0.262143` | `+0.000000` | `0.144274` | `+0.000000` | `+0.00000` |
| `arm_ctrl_ov38` (`ent=1.0, kl=0.5`) | **`0.282023`** | **`+0.019881`** (`10/0/4`) | `0.171680` | `+0.027406` (`9/2/3`) | **`+0.02364`** |
| `arm_sar_sam` (`ent=0.5, kl=0.5, rho=0.01`) | `0.270908` | `+0.008766` (`8/0/6`) | **`0.182099`** | **`+0.037825`** (`8/2/4`) | `+0.02330` |
| `arm_memo_marginal` (`ent=0.5, kl=0.5`) | `0.265748` | `+0.003606` (`7/0/7`) | `0.172344` | `+0.028070` (`9/2/3`) | `+0.01584` |
| `arm_rhythm_gated` (`ent=0.5, kl=0.5`) | `0.245221` | `-0.016922` (`7/0/7`) | `0.162616` | `+0.018342` (`7/2/5`) | `+0.00071` |

### Printed Safety Gate (`n=10` clean + `n=10` photo, top-2 HW arms)

| Arm | `pr_clean` F1 | `pr_clean` Δ (W/T/L) | `pr_photo` F1 | `pr_photo` Δ (W/T/L) | Gate (`>= -0.005`) |
|---|---:|---:|---:|---:|---|
| `baseline` | `0.573594` | `+0.000000` | `0.559013` | `+0.000000` | — |
| `arm_ctrl_ov38` | `0.573978` | `+0.000384` (`1/9/0`) | `0.548775` | **`-0.010239`** (`0/8/2`) | **FAIL (`pr_photo`)** |
| `arm_sar_sam` | `0.572993` | `-0.000601` (`1/8/1`) | `0.550270` | **`-0.008744`** (`1/7/2`) | **FAIL (`pr_photo`)** |

## Scrutiny & Anomaly Check

1. **Coverage & Integrity:** All `28` HW pages × `4` arms and `20` printed safety pages × `2` arms scored cleanly (`0` decode failures, `0` NaN/Inf, `missing_keys=[]`).
2. **Apples-to-Apples Hyperparameter Confound:** `arm_ctrl_ov38` used `ent_w=1.0`, whereas `arm_sar_sam` and `arm_memo_marginal` used `ent_w=0.5`. Even at half entropy weight, **`arm_sar_sam` crushed `arm_ctrl_ov38` on `hw_photo` by `+0.0104`** (`+0.0378` vs `+0.0274`, with `+0.2105` on `hw-beethoven-das-blumchen-wunderhold-d-b` and `+0.2087` on `hw-cornelius-botschaft-d-b`), while trailing on `hw_clean` (`+0.0088` vs `+0.0199`).
3. **Why `arm_rhythm_gated` Failed (`-0.0169` on `hw_clean`):** Only `3/14` `hw_clean` and `4/14` `hw_photo` pages passed the reliability gate (`beat_valid_frac >= 0.70` was frequently `0.0` on unmeasured/complex HW manuscripts). Failing the gate dropped `kl_w` to `0.0` (un-anchored visual consistency), causing encoder drift on clean HW (`-0.0759` on `hw-debussy-vetement-du-blesse`, `-0.0741` on `hw-debussy-berceuse-heroique`).
4. **Printed Photo Gate Failure Root Cause:** Both `arm_ctrl_ov38` and `arm_sar_sam` failed `pr_photo >= -0.005` almost entirely because of a single dense printed outlier: `beethoven-grosse-fuge` photo (`-0.0936` on both arms; `8/10` other printed photo pages were exact ties or wins). Running TTT unconditionally on high-confidence printed pages is unnecessary and risky.

## Verdict: **`ITERATE`**

- `arm_sar_sam` demonstrates a **+3.78 pp** breakthrough on `hw_photo` (beating `arm_ctrl_ov38` by `+1.04 pp`) with tighter worst-case losses (`-0.0264` vs `-0.0392`), but does not pass PROMOTE because (a) `arm_sar_sam` trailed `arm_ctrl_ov38` on `hw_clean` under mismatched `ent_w=0.5` vs `1.0`, and (b) both arms failed the `pr_photo >= -0.005` safety gate due to `beethoven-grosse-fuge`.

## Follow-Up Action / Successor Experiment

- **Proposed Successor:** `i13-06b-sar-sam-conf-skip-ttt` (depends on `2026-11-26T07-40-i13-06-memo-sar-rhythm-gated-ttt`):
  1. Match `ent_w=1.0, kl_w=0.5` on `arm_sar_sam` (`rho in {0.005, 0.01}`) to recover the `hw_clean` gap while keeping the `+0.0378` `hw_photo` gain.
  2. Replace the broken rhythm gate (`kl=0` fallback) with a **confidence skip gate** (if baseline `mean_logprob > thr`, skip TTT entirely with 0 weight updates) so dense printed pages like `beethoven-grosse-fuge` stay untouched (`0.000` printed regression).
