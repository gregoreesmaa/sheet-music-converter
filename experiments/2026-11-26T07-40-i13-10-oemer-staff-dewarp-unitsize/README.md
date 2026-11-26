# Iter 13 (`i13-10`) — OEMER / `muskew` Staffline Dewarping & Homr `unit_size` Scale Normalization

- **Experiment ID:** `2026-11-26T07-40-i13-10-oemer-staff-dewarp-unitsize`
- **Status:** `QUEUED`
- **Source Inspiration:** Open-source OMR competitors **OEMER / `muskew`** (`BreezeWhite/oemer`, `dewarp.py`) and **Homr** (`liebharc/homr`)
- **Platform / GPU:** Google Colab `L4` (`24 GB`, BF16 native, ~28 min, ~2.2 CU)
- **HF Eval Target:** `evals/2026-11-26T07-40-i13-10-oemer-staff-dewarp-unitsize.json`
- **Core Idea:** Test two zero-retraining, deterministic image front-end tricks from OEMER and Homr on full 90-page `suite31_hw` (`31` printed + `14` real handwritten manuscripts, `clean` + `photo`):
  1. **OEMER / `muskew` Staffline-Aware Vertical Dewarping:** Extract horizontal staffline ridges, estimate smooth per-column vertical displacement $y(x)$ via robust median + Savitzky-Golay/Gaussian smoothing (clamped to $\le 3\%$ page height), and remap columns to flatten curved/drooping staves without cropping notation.
  2. **Homr `unit_size` Scale Normalization:** Estimate each page's dominant inter-staffline spacing (`unit_size` in px) via vertical 1D autocorrelation of horizontal ink profiles, and rescale the page to a canonical target `unit_size` (clamped within `max_dim in [960, 1600]`) so dense 12-staff scores get higher resolution while sparse 4-staff scores avoid overscaling.
- **Arms (Single-Variable 2x2 Factorial):**
  - `baseline_1280`: Canonical `max_dim=1280`, no dewarping, no `unit_size` norm.
  - `oemer_dewarp_only`: Staffline vertical $y(x)$ dewarping only (`max_dim=1280`).
  - `homr_unitsize_only`: Autocorrelation `unit_size` target normalization only (clamped `max_dim in [960, 1600]`).
  - `oemer_dewarp_plus_unitsize`: Combined vertical dewarping + `unit_size` normalization.
