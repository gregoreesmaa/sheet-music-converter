# OUTCOME — Exp 5 (`2026-11-26T00-54-ablation-unlabeled-rism-rlvr`)

- **Status:** `COMPLETED` (`2026-11-25T23:40:54Z`, `1,056.4 s` / `17.6 min` on Modal `L4`)
- **Modal App ID:** `ap-ozqdCx4lsObGII8YLKdwFV`
- **HF Artifacts:**
  - `evals/2026-11-26T00-54-ablation-unlabeled-rism-rlvr.json`
  - `runs/2026-11-26T00-54-ablation-unlabeled-rism-rlvr.json`
  - `checkpoints/2026-11-26T00-54-ablation-unlabeled-rism-rlvr/uni_omr_rlvr_final.pt`
- **Verdict:** **PARTIAL / DIAGNOSTIC CONFIRMED (DISCARD structural-only RLVR as standalone F1 booster; USE as Arm B3 baseline for Dual-Verifier Render-Cycle RLVR)**

## Measured Results (`suite31_hw`, 90 pages, `max_dim=1024`, Python Note F1 proxy)

| Bucket | Metric | Baseline (`iter10_soup`) | Post-RLVR (120 GRPO steps, RISM) | Delta |
|---|---|---:|---:|---:|
| `printed_clean` (31) | Note F1 / Struct Reward | `0.6394` / `0.6588` | `0.6394` / `0.6588` | `+0.0000` / `+0.0000` |
| `printed_photo` (31) | Note F1 / Struct Reward | `0.6375` / `0.6624` | `0.6377` / `0.6624` | `+0.0002` / `+0.0000` |
| `hw_clean` (14) | Note F1 / Struct Reward | `0.2175` / `0.3028` | `0.2181` / `0.3144` | `+0.0006` / **`+0.0116`** |
| `hw_clean` (14) | `r_rhythm` | `0.3751` | `0.3901` | **`+0.0150`** |
| `hw_photo` (14) | Note F1 / Struct Reward | `0.1739` / `0.3056` | `0.1718` / `0.3140` | `-0.0020` / **`+0.0084`** |
| `hw_photo` (14) | `r_rhythm` / `r_noloop` | `0.3594` / `0.7143` | `0.3737` / `0.7857` | **`+0.0143`** / **`+0.0714`** |
| `all45_clean` (45) | Note F1 / Struct Reward | `0.5082` / `0.5480` | `0.5083` / `0.5516` | `+0.0002` / `+0.0036` |
| `all45_photo` (45) | Note F1 / Struct Reward | `0.4933` / `0.5514` | `0.4928` / `0.5540` | `-0.0005` / `+0.0026` |

## Scrutiny & Anomaly Check

1. **Zero silent failures / clean execution:** `missing_keys=[]`, `unexpected_keys=[]`, 5,707 unlabeled RISM pages loaded, `kl_beta=0.04` prevented any printed drift (`printed_clean` identical to 6 decimals).
2. **Scientific finding (confirms Academic Exp A hypothesis):** Purely structural label-free RLVR ($R_{\text{sym}}$ alone) measurably improved **structural validity on real HW** (`r_rhythm +1.5 pp`, `r_noloop +7.1 pp` on `hw_photo`), **without moving Note F1 (`±0.001`)**. Without visual grounding ($R_{\text{vis}}$ render-cycle verification), the policy learns to make bars metrically valid and avoid loops, but receives zero signal about *which pitches/rhythms are actually written on the page*.
3. **Training-budget caveat:** The proxy run used short rollouts (`max_len=384`, `max_dim=1024`), small `lr=1e-5`, and `120` updates (`G=4`), keeping total weight movement tiny (which is why `printed_clean` was unchanged). Larger updates (`300–500` steps, `max_len=1024+`) MUST be paired with visual grounding ($R_{\text{vis}}$, tested in **Exp 7** & **Academic Exp A**).
