# Plan — Academic Experiment A v2 (`2026-11-26T02-20-academic-dual-verifier-v2-fixed-render`)

## 1. Platform & Budget

- **Platform:** Google Colab **`NVIDIA L4`** (`24 GB`, BF16 native) via overnight queue (`experiments/overnight_queue/queue.json`, `provider: "colab_l4"`).
- **Estimated Time / Cost:** ~35 min (`~2.8 CU`).
- **Predecessor:** `2026-11-26T01-30-academic-dual-verifier-rlvr-ttc` (`ITERATE`).

## 2. Exact Changes vs v1

| Axis | v1 (`01-30`) | v2 (`02-20`) |
|---|---|---|
| Runtime `pip install` cache | Missing `importlib.invalidate_caches()` (`verovio=None`) | `importlib.invalidate_caches()` + fail-fast startup render `assert` |
| `verovio.toolkit()` thread safety | Unsynchronized in `ThreadPoolExecutor(4)` (font race) | Module-level `_VEROVIO_LOCK` singleton in `dual_verifier.py` |
| `rollout_max_len` | `1536` (truncated 41.8% of rollouts vs `2048` Greedy) | **`2048`** (exact match with `greedy_max_len = 2048`) |
| Fusion calibration | Fixed `0.2 + 0.8 * R_vis` only | Default `R_dual` + offline grid over multiplicative & additive fusions |
| Per-page render telemetry | Not printed | Printed every page (`rend=16/16`) + fail-fast if `render_ok_rate < 0.5` on first page |

## 3. Go/No-Go Gate (Phase 1 v2 → Phase 2 5-Arm GRPO)

- **Hard Precondition:** `render_ok_rate >= 0.85` across both Printed and HW rollouts.
- **GO (`GO_PHASE_2`):** On HW (`n=28`), `R_dual` (or calibrated fusion) achieves `rho_dual > rho_sym` (pooled or within-page, with `rho >= 0.25`) **AND** `Bo16+G` improves over Greedy `K=1` (`delta > 0.0`) with Oracle headroom `>= +0.03`.
- **ITERATE:** `render_ok_rate >= 0.85` and Oracle headroom `>= +0.03`, but `R_vis` still has weak/negative $\rho$ on HW → inspect worst `R_vis` failures and replace DTW/projection with learned visual-cycle feature similarity (e.g. V-JEPA / Sobel-HOG cosine).
- **DISCARD:** `R_vis` actively hurts Printed and HW across all fusion weights.
