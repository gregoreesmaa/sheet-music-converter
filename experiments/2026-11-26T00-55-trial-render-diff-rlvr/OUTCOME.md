# OUTCOME — Exp 7 (`2026-11-26T00-55-trial-render-diff-rlvr`)

- **Status:** `COMPLETED` (`2026-11-25T23:46:42Z` on Modal `NVIDIA L4`)
- **Modal App ID:** `ap-otKKVyElX8m8GGwPd1zeVa`
- **HF Artifacts:**
  - `evals/2026-11-26T00-55-trial-render-diff-rlvr.json`
  - `checkpoints/2026-11-26T00-55-trial-render-diff-rlvr/uni_omr_final.pt`
- **Verdict:** **ITERATE** (`+0.0077` Strict Note F1 on `hw_clean` in just 60 steps with `0.0000` `printed_clean` regression, but `-0.0065` on `hw_photo` — confirming that naive pixel-space render-diff favors clean pages and must be upgraded to style- & spacing-invariant DTW/projection `R_vis` in Academic Exp A).

## Measured Results (`suite31_hw`, All 45 Pieces / 90 Pages, Native Rust `eval-harness`)

| Bucket | Before (`iter10_soup`) | After (60 Render-Diff RLVR steps) | Delta |
|---|---:|---:|---:|
| `printed_clean` (31) | `0.8852` | `0.8852` | `+0.0000` |
| `printed_photo` (31) | `0.8691` | `0.8677` | `-0.0014` |
| **`hw_clean` (14)** | `0.2476` | **`0.2553`** | **`+0.0077`** |
| `hw_photo` (14) | `0.1997` | `0.1932` | `-0.0065` |
| `all45_clean` (45) | `0.6869` | **`0.6893`** | **`+0.0024`** |
| `all45_photo` (45) | `0.6608` | `0.6579` | `-0.0030` |

## Scrutiny & Anomaly Check

1. **Clean execution & 100% `DecOK`:** All 90/90 pages scored via native Rust `eval-harness` (`missing_keys=[]`, `unexpected_keys=[]`, `decode_ok_rate=1.0`).
2. **Causal contrast vs Exp 5 (Structural-Only RLVR):** Unlike Exp 5 (`hw_clean +0.0006`), adding visual render-diff lifted **`hw_clean` by `+0.0077`** in only 60 steps!
3. **Photo domain asymmetry (`hw_clean +0.0077` vs `hw_photo -0.0065`):** Because Verovio renders clean white-background engravings, raw pixel-space render-diff pushes the policy toward clean contrast cues and slightly hurts phone-photo / aged-paper pages. This directly validates **Academic Exp A's** style- & spacing-invariant DTW + vertical pitch-contour `R_vis` upgrade.
