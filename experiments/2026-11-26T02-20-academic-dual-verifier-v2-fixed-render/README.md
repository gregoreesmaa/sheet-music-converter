# Academic Experiment A v2: Dual-Verifier Gate & TTC Scaling (Fixed Render + Matched 2048 Length) (`2026-11-26T02-20-academic-dual-verifier-v2-fixed-render`)

**Status:** `QUEUED` (Colab `L4`, successor to [`2026-11-26T01-30-academic-dual-verifier-rlvr-ttc`](../2026-11-26T01-30-academic-dual-verifier-rlvr-ttc/))

## Motivation & Fixes Over v1

Academic Experiment A v1 (`2026-11-26T01-30-academic-dual-verifier-rlvr-ttc`) suffered two concrete bugs caught in post-run scrutiny:
1. **`render_ok_rate = 0.0` (`R_vis = 0.0` on all 816/816 candidates):** Missing `importlib.invalidate_caches()` after in-cell `pip install` PLUS a `verovio.toolkit()` C++ font-table race under `ThreadPoolExecutor(max_workers=4)` caused `R_vis` to silently return `0.0` everywhere.
2. **Length asymmetry (`greedy_max_len = 2048` vs `rollout_max_len = 1536`):** `15/28` HW pages had Greedy length `>= 1536` tokens while `41.8%` of rollouts were truncated at `1536`, causing `R_sym`-only to short-sequence-hack `19/28` HW pages (`-0.0134` HW F1 despite `+0.0361` Oracle headroom and `+0.0233` Printed Bo16+G gain).

**This v2 run fixes all three issues:**
- Calls `importlib.invalidate_caches()` after `pip install` and executes a **fail-fast startup smoke test** asserting `dv.verovio is not None`, `dv.resvg_py is not None`, and `dv.render_bekern_to_gray(...)` succeeds across `ThreadPoolExecutor(max_workers=4)` before any GPU rollout starts.
- Uses the thread-safe `_VEROVIO_LOCK` singleton in `tmp-train/mvd/dual_verifier.py`.
- Matches `rollout_max_len = 2048` to `greedy_max_len = 2048` for apples-to-apples dense HW evaluation.
- Performs a zero-extra-GPU-cost offline calibration sweep over `R_dual` fusion formulas (multiplicative `sym_floor ∈ {0.1, 0.2, 0.4}` and additive `(1-w)*R_sym + w*R_vis`) to find the optimal Phase 2 GRPO reward before spending compute on 5-arm GRPO.

## Quick Links

- Predecessor Outcome: [`../2026-11-26T01-30-academic-dual-verifier-rlvr-ttc/OUTCOME.md`](../2026-11-26T01-30-academic-dual-verifier-rlvr-ttc/OUTCOME.md)
- Plan: [PLAN.md](PLAN.md)
- Colab Launcher: [run_colab.py](run_colab.py)
- HF Eval Target: `gregoreesmaa/sheet-music-converter-omr:evals/2026-11-26T02-20-academic-dual-verifier-v2-fixed-render.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr:runs/2026-11-26T02-20-academic-dual-verifier-v2-fixed-render.json`
