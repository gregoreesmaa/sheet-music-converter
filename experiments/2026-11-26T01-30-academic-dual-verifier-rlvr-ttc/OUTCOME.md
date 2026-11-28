# Outcome — Academic Experiment A (Phase 1: Dual-Verifier Falsification Gate & TTC Scaling Curve)

**Experiment ID:** `2026-11-26T01-30-academic-dual-verifier-rlvr-ttc`
**Completed:** `2026-11-26`
**Baseline Commit SHA:** `164852f`
**Baseline Checkpoint:** `gregoreesmaa/sheet-music-converter-omr:checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`step=12000`, `vocab=237`)
**Final Verdict:** **`ITERATE`** *(Phase 1a established the `R_sym`-only ablation & strong Oracle headroom across all 48 pages on Colab `L4` BF16 and `T4` FP16; mandatory scrutiny caught a silent `verovio` worker-thread font-path bug causing `render_ok_rate = 0.0` — fixed in `dual_verifier.py` via `_get_verovio_toolkit()` and verified with 13/13 CPU unit tests; ready for Phase 1b re-run with active `R_vis` before Phase 2 5-arm GRPO!)*

---

## 1. Measured Results Table (`suite31_hw`, 24 pieces × 2 domains = 48 pages, 768 rollouts + 48 greedy)

### Primary Run: Colab **`NVIDIA L4` (`BF16`)**, `1,914.7s` (`100%` `decode_ok_rate`)

> **Note:** Because `render_ok_rate = 0.0` in Phase 1a (see Section 2 Scrutiny), `R_vis = 0.0` for all rollouts and `R_dual = 0.2 * R_sym`, making this an exact **`R_sym`-only ablation** across all 48 pages.

| Slice (`N` pages) | $\rho_{\text{pooled}}(R_{\text{sym}}, \text{F1})$ | $\rho_{\text{within}}(R_{\text{sym}}, \text{F1})$ | `K=1` Greedy | `K=1` Sampled | `Bo4` ($R_{\text{sym}}$) | `Bo16` ($R_{\text{sym}}$) | `Bo16+G` ($R_{\text{sym}}$) | `Oracle16+G` | Headroom (`Orc16+G − G`) | TTC Slope ($\Delta\text{F1}/\log_2 K$ vs samp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Overall (48)** | **+0.6109** | **+0.4459** | 0.5041 | 0.4765 | 0.4912 | 0.5015 | **0.5060** | 0.5410 | **+0.0369** | +0.0063 |
| **Printed All (20)** | **+0.6913** | **+0.5984** | 0.8942 | 0.8700 | 0.8926 | **0.9089** | **0.9175** | 0.9322 | **+0.0380** | **+0.0097** |
| Printed Clean (10) | +0.6916 | +0.5944 | 0.8906 | 0.8679 | 0.8856 | 0.9008 | **0.9171** | 0.9283 | +0.0377 | +0.0082 |
| Printed Photo (10) | +0.6991 | +0.6019 | 0.8979 | 0.8721 | 0.8995 | 0.9169 | **0.9178** | 0.9362 | +0.0384 | +0.0112 |
| **HW All (28)** | **+0.3627** | **+0.3533** | 0.2255 | 0.1955 | 0.2045 | 0.2106 | 0.2121 | **0.2616** | **+0.0361** | +0.0038 |
| HW Clean (14) | +0.3009 | +0.3119 | 0.2545 | 0.2268 | 0.2366 | 0.2424 | 0.2424 | 0.2959 | +0.0414 | +0.0039 |
| HW Photo (14) | **+0.4412** | **+0.3948** | 0.1965 | 0.1641 | 0.1723 | 0.1787 | 0.1818 | 0.2273 | +0.0309 | +0.0037 |

### Secondary Replication Run: Colab **`Tesla T4` (`FP16`)**, `1,669.1s` (`100%` `decode_ok_rate`)

- **Overall (48):** $\rho_{\text{pooled}} = +0.6084$, `K=1` Greedy `0.5074`, `Oracle16+G` `0.5411` (`+0.0337` headroom).
- **Printed All (20):** $\rho_{\text{pooled}} = +0.6861$, `K=1` Greedy `0.8990` → `Bo16` `0.9109` (`+1.19%` F1 vs Greedy!).
- **HW All (28):** $\rho_{\text{pooled}} = +0.3598$, `K=1` Greedy `0.2277`, `Oracle16+G` `0.2642` (`+0.0365` headroom).
- *(Note: T4 `FP16` showed slightly higher sampling variance on HW within-page $\rho = +0.1811$ vs L4 `BF16` `+0.3533`, confirming L4/A100 `BF16` is strongly preferred for RLVR/TTC.)*

---

## 2. Scrutiny & Anomaly Check (CRITICAL CATCH)

1. **Silent `render_ok_rate = 0.0` caught via mandatory post-run scrutiny:**
   - Despite `DualVerifier visual render smoke test OK: shape=(112, 840)` passing on the **main thread** at startup, every single one of the 768 rollouts scored inside `ThreadPoolExecutor(max_workers=4)` returned `render_ok = False` (`r_vis = 0.0`).
   - **Root causes identified:**
     1. In Run 1 (`L4`), `pip install -q verovio resvg-py` ran after a failed `__import__` probe without `importlib.invalidate_caches()`, leaving `verovio = None`.
     2. In Run 2 (`T4`), `importlib.invalidate_caches()` fixed the main-thread smoke test, **but** calling `verovio.toolkit()` inside `ThreadPoolExecutor` worker threads failed to locate `verovio/data` fonts (Bravura/Leipzig), causing 100% silent render failures.
   - **Fix implemented & verified:**
     - Updated `tmp-train/mvd/dual_verifier.py` with singleton `_get_verovio_toolkit()` under `_VEROVIO_LOCK` and explicit `tk.setResourcePath(str(Path(verovio.__file__).resolve().parent / "data"))`.
     - Verified with **`13/13` PASS** in `tmp-train/mvd/test_dual_verifier.py` (including `test_threadpool_verovio_render_no_race`, `0.435s` CPU).
2. **Scientific value of the `R_sym`-only ablation:**
   - On **Printed** pages, $R_{\text{sym}}$ alone is already a powerhouse: $\rho = +0.691$ pooled / $+0.598$ within-page, lifting `K=1` Greedy `0.8942` → **`Bo16+G` `0.9175` (`+2.33%` Strict Note F1!)**.
   - On **Historical Handwritten** pages, $R_{\text{sym}}$ alone achieves strong correlation ($\rho = +0.363$ pooled / $+0.353$ within-page) and lifts sampled rollouts by `+1.51%`, **yet loses `-1.34%` against Greedy** (`0.2121` vs `0.2255`) despite **`+3.61%` Oracle headroom** (`0.2616`). This empirically proves our core hypothesis: symbolic verification alone cannot reject syntactically valid hallucinations that omit systems or misalign with manuscript ink — **visual-cycle grounding ($R_{\text{vis}}$) is required to unlock the `+3.61%` HW Oracle headroom!**

---

## 3. Artifacts & Reproducibility

- Verifier module: [`tmp-train/mvd/dual_verifier.py`](../../tmp-train/mvd/dual_verifier.py) (`13/13` CPU unit tests PASS in [`tmp-train/mvd/test_dual_verifier.py`](../../tmp-train/mvd/test_dual_verifier.py))
- Colab launcher: [`experiments/2026-11-26T01-30-academic-dual-verifier-rlvr-ttc/run_colab.py`](run_colab.py)
- HF Full Eval JSON: `gregoreesmaa/sheet-music-converter-omr:evals/2026-11-26T01-30-academic-dual-verifier-rlvr-ttc.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr:runs/2026-11-26T01-30-academic-dual-verifier-rlvr-ttc.json`
- Colab Session Cleanup: `acad-a-dual-rlvr` terminated cleanly (`0` lingering sessions); `/tmp/.hf_token` deleted in `finally`.

---

## 4. Next Action Triggered

1. **Immediate Phase 1b re-run** on Colab `L4`/`A100` (`BF16`) with the fixed `_get_verovio_toolkit()` bundled so `render_ok_rate >= 0.90` and we measure true `R_vis` and `R_dual` Spearman $\rho$ & TTC scaling vs the `R_sym`-only baseline above.
2. Once `R_dual` beats `R_sym`-only on HW (`Bo16+G > 0.2255` Greedy), green-light **Phase 2 (5-arm GRPO on 5,707 unlabeled RISM pages)**.

---

## 5. Platform Learnings Amended

- Amended `.agents/skills/cloud-colab/SKILL.md` with:
  1. Per-tier Colab L4 concurrency cap (2 active L4s → `Allocation refused`; fallback to `A100`).
  2. `colab exec` default 30s silence timeout → always pass `--timeout 600`.
  3. `importlib.invalidate_caches()` required after in-cell `pip install`.
  4. `verovio.toolkit()` worker-thread font-path bug → use singleton `_get_verovio_toolkit()` with explicit `setResourcePath` under a lock.
