# Academic Experiment A: Dual-Verified Render-Cycle RLVR & Test-Time Compute Scaling (`2026-11-26T01-30-academic-dual-verifier-rlvr-ttc`)

**Status:** `COMPLETED (ITERATE)` — `render_ok_rate = 0.0` (`R_vis` disabled by import-cache + `verovio` thread-race bugs); `R_sym`-only achieved `+0.0233` Printed F1 and `+0.0361` HW Oracle headroom. Follow-up queued: [`2026-11-26T02-20-academic-dual-verifier-v2-fixed-render`](../2026-11-26T02-20-academic-dual-verifier-v2-fixed-render/).

## Summary & Hypothesis

Label-free RLVR (GRPO) on unlabeled historical handwritten sheet music (e.g., 5,707 RISM manuscript pages) requires a verifier that (1) prevents reward hacking via syntactically valid empty/hallucinated bars, and (2) correlates strongly with true ground-truth **Strict Note F1** despite severe style differences (yellowed parchment, brown iron-gall ink, uneven lighting) and scribal horizontal spacing stretch.

We introduce **`DualVerifier`** (`tmp-train/mvd/dual_verifier.py`), combining:
1. **$R_{\text{sym}}$ (Exact Symbolic Process Verifier):** Measure-level `bekern` grammar parse + interior-measure beat conservation + cross-voice agreement + balanced ties/tuplets + token & `<sys>`-loop penalties.
2. **$R_{\text{vis}}$ (Style- & Spacing-Invariant Visual-Cycle Verifier):** Re-renders predicted `bekern` → MusicXML → SVG/PNG via `verovio` + `resvg-py`, removes paper yellowing/shadows via local 90th-percentile illumination normalization, suppresses staff lines, and computes **Sakoe-Chiba banded DTW-aligned onset-column 1D Earth Mover's Distance (EMD)** + **vertical pitch-density profile correlation & EMD**.
3. **Multiplicative Grounded Gate:**
   $$R_{\text{dual}} = R_{\text{sym}} \cdot (0.2 + 0.8 \cdot R_{\text{vis}})$$

**Phase 1 Hypothesis:** On `suite31_hw` (all 14 historical HW pieces + 10 diverse printed pieces, `clean` + `photo` = 48 pages) sampled from baseline `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (commit `164852f`), $R_{\text{dual}}$ will achieve higher within-page and pooled Spearman $\rho$ against true Strict Note F1 than either $R_{\text{sym}}$ or $R_{\text{vis}}$ alone, and **Best-of-K** (`K ∈ {1, 4, 16}`) selection with $R_{\text{dual}}$ will yield a positive Test-Time Compute (TTC) scaling curve over Greedy `K=1`, validating the Go/No-Go gate for Phase 2 (5-arm GRPO on 5,707 unlabeled RISM pages).

## Quick Links

- Plan & Execution Spec: [PLAN.md](PLAN.md)
- Colab Launcher: [run_colab.py](run_colab.py)
- Outcome & Analysis: [OUTCOME.md](OUTCOME.md) *(written upon completion)*
- Verifier Implementation: [`tmp-train/mvd/dual_verifier.py`](../../tmp-train/mvd/dual_verifier.py)
- Verifier CPU Unit Tests: [`tmp-train/mvd/test_dual_verifier.py`](../../tmp-train/mvd/test_dual_verifier.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr:evals/2026-11-26T01-30-academic-dual-verifier-rlvr-ttc.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr:runs/2026-11-26T01-30-academic-dual-verifier-rlvr-ttc.json`
