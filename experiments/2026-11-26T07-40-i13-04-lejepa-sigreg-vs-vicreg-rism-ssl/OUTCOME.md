# OUTCOME — `2026-11-26T07-40-i13-04-lejepa-sigreg-vs-vicreg-rism-ssl` (`i13-04`)

- **Status:** `COMPLETED` (Stage 1 & 2 SSL rank audit valid; Stage 3 `suite31_hw` eval hit missing `.exe` path bug)
- **Completed:** `2026-11-26T08:13:13Z` (`elapsed_sec = 4377.81` wall / `4293.53` in-script, ~73 min)
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`24 GB`, BF16 native; session `i13-04` confirmed stopped)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T07-40-i13-04-lejepa-sigreg-vs-vicreg-rism-ssl.json` (downloaded to `eval_result.json`)
  - Run Ledger: `runs/2026-11-26T07-40-i13-04-lejepa-sigreg-vs-vicreg-rism-ssl.json` (downloaded to `run_meta.json`)
  - SSL Ckpt: `checkpoints/2026-11-26T07-40-i13-04-lejepa-sigreg-vs-vicreg-rism-ssl/best_ssl_encoder.pt` (**DO NOT USE** — contains collapsed `arm_sigreg_w15`)
- **Verdict:** **`DISCARD`** — Keep VICReg (`var_w=1.0, cov_w=0.04`) as the default anti-collapse regularizer for RISM V-JEPA 2.1 SSL.

---

## 1. Measured Results

### Stage 1 & 2: RISM 5,707 SSL (400 steps, `small_45m`, BF16, `0` OOM skips) & 12-Page SVD Rank Probe

| Arm | Mode | `p2_erank` | `p3_erank` (gate `>= 12`) | `mem_erank` (gate `>= 30`) | `mem_tok_std` | `within_page_cos` | `cross_page_cos` (gate `<= 0.90`) | `epps_pulley_stat` | Final EMA Loss | Health Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`arm_vicreg`** (baseline) | VICReg (`1.0 / 0.04`) | 28.11 | **15.91** | **54.25** | 0.7663 | 0.4349 | **0.8578** | 0.28031 | 0.9028 | **PASS** |
| `arm_sigreg_w05` | SIGReg (`w=0.5`) | 29.86 | 1.79 (`-88.7%`) | 5.30 (`-90.2%`) | 0.6790 | 0.4652 | 0.5820 | 0.36206 | 0.5357 | **FAIL (collapsed)** |
| `arm_sigreg_w15` (selected) | SIGReg (`w=1.5`) | 32.36 | 2.61 (`-83.6%`) | 9.11 (`-83.2%`) | 0.7913 | 0.3105 | 0.6609 | 0.24216 | 0.7307 | **FAIL (collapsed)** |

### Stage 3: Matched 600-Step `iter10_5_proxy_2k` Supervised Transfer Probe (`fullpage_val` from run log)

| Transfer Arm | Init SSL Encoder | Step 300 Val F1 (raw / EMA) | Step 600 Val F1 (raw / EMA) | Step 600 Val SER (raw / EMA) | `suite31_hw` |
|---|---|---:|---:|---:|---|
| **`transfer_vicreg`** | `arm_vicreg` | `0.1537` / **`0.2715`** | **`0.3074`** / `0.1994` | **`0.8444`** / `0.8944` | `null` (0/45 scored; see anomaly below) |
| `transfer_arm_sigreg_w15` | `arm_sigreg_w15` | `0.0382` / `0.2678` | `0.2342` / `0.2337` | `0.8729` / `0.8755` | `null` (0/45 scored; see anomaly below) |
| **Delta (`SIGReg - VICReg`)** | — | `-0.1155` / `-0.0037` | **`-0.0732`** / `+0.0343` | `+0.0285` / `-0.0189` | N/A |

---

## 2. Scrutiny & Anomaly Check

1. **Silent `suite31_hw` Evaluation Failure (0/45 pieces scored):** Both Stage 3 `eval_uni_suite31.py` calls completed in 5–6 s with `rc=0` while skipping all 45 pieces with:
   `SKIP ref-error: [Errno 2] No such file or directory: '/content/sheet-music-converter/converter/assembly-serialize/target/release/assembly-serialize.exe'`
   All 6 `suite31_hw` slices in `eval_result.json` are `null` and `comparison_sigreg_minus_vicreg` is `{}`. This did **not** obscure the verdict because Stage 1/2 rank collapse (`p3_erank 2.61 vs 15.91`, `mem_erank 9.11 vs 54.25`) and Stage 3 in-training `fullpage_val` (`0.2342 vs 0.3074` raw F1) already decisively falsify the hypothesis.
2. **Root Cause of SIGReg Dimensional Collapse:** Mean-normalized Epps-Pulley quadrature over `K=64` random 1D slices in `D=512` space without `N`-scaling produces a weak anti-collapse gradient relative to JEPA MSE + Sobel-HOG + KL router losses. Both `w=0.5` (`p3_erank=1.79`) and `w=1.5` (`p3_erank=2.61`) collapsed nearly to the unregularized Exp 2 level (`p3_erank=2.1`), whereas VICReg (`var_w=1.0, cov_w=0.04`) cleanly passed every health gate (`p3_erank=15.91`, `mem_erank=54.25`, `cross_page_cos=0.8578`).
3. **Platform Tag Cosmetic Bug:** `eval_result.json` logged `"platform": "kaggle-NVIDIA L4"` even though the job ran on Google Colab L4, because `run_colab.py` checks `Path("/kaggle").exists()` before `Path("/content").exists()` and the Colab image contains `/kaggle`.
4. **Quarantined HF Checkpoint:** `checkpoints/2026-11-26T07-40-i13-04-lejepa-sigreg-vs-vicreg-rism-ssl/best_ssl_encoder.pt` is the rank-collapsed `arm_sigreg_w15` checkpoint — do not warm-start any run from it.

---

## 3. Verdict & Follow-Up Action

- **Verdict:** **`DISCARD`** (per `PLAN.md` Section 4: SIGReg underperforms VICReg on both effective rank and transfer).
- **Action:** Retain **VICReg (`var_w=1.0, cov_w=0.04`)** as the default anti-collapse regularizer in `pretrain_ijepa_rism_v2.py`. No follow-up SIGReg sweep queued.
- **Infra Hygiene Note:** Ensure future Linux cloud runs invoking `eval_uni_suite31.py` use the clean-GT/bekern manifest or a Linux-compiled `assembly-serialize` binary rather than a hardcoded `.exe` path.
