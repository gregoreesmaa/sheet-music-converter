# Outcome — Academic Experiment A (`2026-11-26T01-30-academic-dual-verifier-rlvr-ttc`)

- **Status:** `COMPLETED` — **Verdict: `ITERATE`** (follow-up queued as [`2026-11-26T02-20-academic-dual-verifier-v2-fixed-render`](../2026-11-26T02-20-academic-dual-verifier-v2-fixed-render/))
- **Completed (UTC):** `2026-11-26T00:14:31Z`
- **Platform / GPU:** Google Colab `NVIDIA L4` (`24 GB`, BF16 native, session `acad-a-dual-rlvr`, verified stopped cleanly via `colab.exe ls`)
- **Elapsed Time:** `1,914.65 s` (`31.9 min`, `~2.55 CU`)
- **Baseline Checkpoint:** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (commit `164852f`)
- **HF Eval Artifact:** `gregoreesmaa/sheet-music-converter-omr:evals/2026-11-26T01-30-academic-dual-verifier-rlvr-ttc.json`
- **HF Run Ledger:** `gregoreesmaa/sheet-music-converter-omr:runs/2026-11-26T01-30-academic-dual-verifier-rlvr-ttc.json`

---

## Measured Results (48 `suite31_hw` Pages × 17 Candidates = 816 Sequences)

> [!WARNING]
> **Critical Finding (see Scrutiny below):** `render_ok_rate == 0.0` across all 816 candidates due to two bugs (`importlib.invalidate_caches()` missing after runtime `pip install` + `verovio.toolkit()` C++ font-table race under `ThreadPoolExecutor`). Consequently, `R_vis = 0.0` everywhere and `R_dual = 0.2 * R_sym` identically. The numbers below therefore measure **`R_sym`-only** ranking + **Oracle-of-16 headroom** — `R_vis` and true `R_dual` were **not** exercised in this run.

### 1. Ranking Correlation (Spearman $\rho$ vs Ground-Truth Strict Note F1)

| Slice | Pages | Rollouts | `render_ok` | `decode_ok` | Pooled $\rho(R_{\text{sym}})$ | Within-Page $\rho(R_{\text{sym}})$ | $\rho(R_{\text{vis}})$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 48 | 768 | **0.0000** | 1.0000 | `+0.6109` | `+0.4459` | `null` (0.0) |
| **HW All** | 28 | 448 | **0.0000** | 1.0000 | `+0.3627` | `+0.3533` | `null` (0.0) |
| HW Clean | 14 | 224 | **0.0000** | 1.0000 | `+0.3009` | `+0.3119` | `null` (0.0) |
| HW Photo | 14 | 224 | **0.0000** | 1.0000 | `+0.4412` | `+0.3948` | `null` (0.0) |
| **Printed All** | 20 | 320 | **0.0000** | 1.0000 | `+0.6913` | `+0.5984` | `null` (0.0) |
| Printed Clean | 10 | 160 | **0.0000** | 1.0000 | `+0.6916` | `+0.5944` | `null` (0.0) |
| Printed Photo | 10 | 160 | **0.0000** | 1.0000 | `+0.6991` | `+0.6019` | `null` (0.0) |

### 2. Test-Time Compute (TTC) Scaling Curve (Strict Note F1)

| Slice | K=1 Greedy | K=1 Sampled | Bo4 ($R_{\text{sym}}$) | Bo16 ($R_{\text{sym}}$) | Bo16+G ($R_{\text{sym}}$) | $\Delta$ Bo16+G vs Greedy | Oracle16+G | Oracle Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Overall (48)** | `0.5041` | `0.4765` | `0.4912` | `0.5015` | `0.5060` | **`+0.0019`** | `0.5410` | **`+0.0369`** |
| **HW All (28)** | `0.2255` | `0.1955` | `0.2045` | `0.2106` | `0.2121` | **`-0.0134`** | `0.2616` | **`+0.0361`** |
| HW Clean (14) | `0.2545` | `0.2268` | `0.2366` | `0.2424` | `0.2424` | `-0.0121` | `0.2959` | `+0.0414` |
| HW Photo (14) | `0.1965` | `0.1641` | `0.1723` | `0.1787` | `0.1818` | `-0.0147` | `0.2273` | `+0.0309` |
| **Printed All (20)** | `0.8942` | `0.8700` | `0.8926` | `0.9089` | `0.9175` | **`+0.0233`** | `0.9322` | **`+0.0380`** |
| Printed Clean (10) | `0.8906` | `0.8679` | `0.8856` | `0.9008` | `0.9171` | `+0.0266` | `0.9283` | `+0.0377` |
| Printed Photo (10) | `0.8979` | `0.8721` | `0.8995` | `0.9169` | `0.9178` | `+0.0200` | `0.9362` | `+0.0384` |

---

## Scrutiny & Anomaly Check

1. **Coverage & Parse Integrity:** All `48/48` pages (`28` HW + `20` Printed) and all `816/816` candidates (`48` greedy + `768` rollouts) decoded cleanly (`decode_ok_rate = 1.0`, `parse_ok = 816/816`, zero NaNs). Loop detectors fired on `111/816` (`has_tok_loop`) and `19/816` (`has_sys_loop`) candidates.
2. **CRITICAL SILENT FAILURE #1 — `render_ok_rate = 0.0` (`R_vis = 0.0` on 816/816 candidates):**
   - **Root Cause 1A (`importlib.invalidate_caches()` missing in `run_colab.py`):** `run_colab.py` probed `__import__("verovio")` and `__import__("resvg_py")` *before* `pip install -q verovio resvg-py`, without calling `importlib.invalidate_caches()` afterward. Python's finder cache returned stale `ImportError` when `dual_verifier.py` imported them, silently setting `verovio = None` and `resvg_py = None`.
   - **Root Cause 1B (`verovio.toolkit()` C++ font-table race under `ThreadPoolExecutor`):** Reproduced locally in Python: calling `verovio.toolkit()` concurrently across `ThreadPoolExecutor(max_workers=4)` triggers `[Error] Bravura font could not be loaded` / `[Error] Leipzig font could not be loaded` and returns `None` for 8/8 threads. Fixed in `tmp-train/mvd/dual_verifier.py` by serializing `verovio.toolkit()` through a singleton guarded by `_VEROVIO_LOCK` (while keeping `resvg_py` rasterization unlocked in parallel) and verified with new unit test `test_threadpool_verovio_render_no_race`.
   - **Impact:** `R_vis` was identically `0.0` everywhere, collapsing `R_dual` to `0.2 * R_sym`. Thus the core hypothesis of `DualVerifier` (visual-cycle grounding via `R_vis`) was **not yet tested**.
3. **APPLES-TO-APPLES LENGTH ASYMMETRY #2 — `greedy_max_len = 2048` vs `rollout_max_len = 1536`:**
   - `15/28` HW pages (`19/48` overall) had Greedy length `>= 1536` tokens (reaching `2046–2048`), while `321/768` (`41.8%`) rollouts were truncated at `1536` tokens.
   - Without `R_vis`, `R_sym`-only exhibited **short-sequence reward hacking** on HW: it selected a shorter sequence than Greedy on **`19/28` HW pages**, causing severe regressions on dense manuscripts:
     - `hw-mozart-k464 [clean]`: Greedy `0.401` (`r_sym=0.72`, `2046` tok) → Bo16+G `0.223` (`r_sym=0.82`, `860` tok, **`-0.1784` F1**).
     - `hw-debussy-lindaraja [photo]`: Greedy `0.249` (`2046` tok) → Bo16+G `0.088` (`1536` tok, **`-0.1605` F1**).
     - `hw-debussy-la-plus-que-lente [clean]`: Greedy `0.564` (`2048` tok) → Bo16+G `0.440` (`1533` tok, **`-0.1237` F1**).
   - This empirically confirms *why* `R_sym`-only is insufficient for HW RLVR and why visual coverage grounding (`R_vis`) + matched `rollout_max_len = 2048` are essential.
4. **High-Value Positive Findings Already Confirmed:**
   - **Printed TTC scaling works strongly even with `R_sym`-only:** `0.8942 → 0.9175` (**`+0.0233` Strict Note F1**, clean `+0.0266`, photo `+0.0200`) with within-page $\rho = +0.5984$!
   - **Strong HW Oracle headroom (`+0.0361`, clean `+0.0414`, photo `+0.0309`):** Even under the `1536`-token rollout cap, `iter10_soup` at `temp=0.7` has ample exploration diversity on real historical manuscripts (`0.2255 → 0.2616`), validating the base policy for RLVR/TTC once `R_vis` is actually active.
5. **Hardware / Session Hygiene:** Colab L4 session `acad-a-dual-rlvr` terminated cleanly (`colab.exe ls` confirms zero lingering session). Skill `.agents/skills/cloud-colab/SKILL.md` amended with the `verovio.toolkit()` `ThreadPoolExecutor` race gotcha.

---

## Verdict: `ITERATE`

Do **not** launch Phase 2 (5-arm GRPO) yet, and do **not** discard `DualVerifier`. Because `render_ok_rate = 0.0` disabled `R_vis` completely and `rollout_max_len = 1536` handicapped dense HW rollouts against `2048`-token Greedy, we must re-run Phase 1 with both bugs fixed to obtain the first genuine measurement of `R_vis` and `R_dual`.

## Follow-Up Action / Successor Experiment

- **Queued Successor:** [`2026-11-26T02-20-academic-dual-verifier-v2-fixed-render`](../2026-11-26T02-20-academic-dual-verifier-v2-fixed-render/) (registered in `experiments/overnight_queue/queue.json` with `depends_on: ["2026-11-26T01-30-academic-dual-verifier-rlvr-ttc"]`).
- **Fixes Applied:**
  1. `importlib.invalidate_caches()` after `pip install` + fail-fast startup `assert` verifying `verovio`, `resvg_py`, and `render_bekern_to_gray(...)` return a valid grayscale image before GPU rollouts begin.
  2. Thread-safe singleton `_VEROVIO_LOCK` in `tmp-train/mvd/dual_verifier.py` (covered by `test_dual_verifier.py`).
  3. Matched `rollout_max_len = 2048` (equal to `greedy_max_len = 2048`) to eliminate length-truncation asymmetry on dense HW manuscripts.
  4. Zero-extra-GPU-cost offline sweep over `R_dual` fusion weights (`sym_floor`, multiplicative vs additive `R_sym + w * R_vis`) across all 816 scored candidates.
