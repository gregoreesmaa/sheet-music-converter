# `2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr`

- **Status:** `PLANNED` (QUEUED in `experiments/overnight_queue/queue.json`, Wave 0 / P0)
- **Short ID:** `i15-47`
- **Baseline Commit SHA:** `6055c49`
- **Cross-Field Origin:** Sim2Real (`DeYO` ICLR 2024 PLPD stability) + HMER/Tables/TikZ/Drift (`GRPO` / `RLVR` with deterministic conservation rewards).

## Summary & Hypothesis
Controlled 2-arm 600-step proxy on `UniOMRv2Net(edge85m, ax0_gl6, d=512)` using `iter10_5_proxy_2k.tar` comparing a 600-step SFT baseline (`sft_ctrl`) against **450-step SFT + 150-step Conservation-Gated `DeYO` Consistency + Critic-Free GRPO (`deyo_rlvr`)**.

Pure teacher-forced SFT suffers from exposure bias and cannot optimize non-differentiable musical conservation laws (bar-duration beat sums, syntax parseability, anti-looping). In Phase 2 (steps 451–600), `deyo_rlvr` combines:
1. **DeYO PLPD-Gated Visual Consistency:** Perturbs the input page (mild photometric/patch-jitter view) and applies symmetric KL consistency gated by Pseudo-Label Probability Difference (PLPD) so only visually grounded, conservation-valid predictions are reinforced.
2. **Critic-Free GRPO (`G=4` rollouts):** Samples `G=4` short-horizon rollouts per page under KV caching, scores each rollout with deterministic `rlvr_grpo.compute_omr_rlvr_reward` (exact syntax `r_parse` + bar-duration conservation `r_meter` + anti-loop `r_noloop` + `<sys>` match `r_sys` + token/note edit similarity `r_note`), and optimizes clipped group-relative policy gradient with `k3` KL regularization to the frozen step-450 reference snapshot.

## Quick Links
- [PLAN.md](PLAN.md)
- [run_colab.py](run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr.json`
- HF Partial Sync: `checkpoints/2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr/`
