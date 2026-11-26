# `2026-11-26T14-30-i13-06b-sar-sam-conf-skip-ttt` (`i13-06b`)

- **Parent Experiment:** `2026-11-26T07-40-i13-06-memo-sar-rhythm-gated-ttt` (`ITERATE`)
- **Goal:** Recover `hw_clean` parity on `arm_sar_sam` by matching `ent_w=1.0, kl_w=0.5` across `rho in {0.005, 0.01}`, and eliminate the `pr_photo` safety gate regression (`beethoven-grosse-fuge` `-0.0936`) via a zero-update **confidence skip gate** (`mean_logprob > thr`).
- **Platform:** Modal / Colab `L4` (`BF16`), est. `~26 min` (`~2.2 CU`).
- **Checkpoint:** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`uni_omr_v1`, `val_ser=0.1923`).
- **Evaluation:** Full 14 HW manuscripts (`28` pages: `hw_clean` + `hw_photo`) + 10 printed safety pieces (`20` pages: `pr_clean` + `pr_photo`) from `datasets/suite31_hw_eval.tar` at `max_dim=1280, max_len=3840`, MTP-4 + FSA.
- **Files:**
  - `PLAN.md` — experimental design, matched grid, skip-gate thresholds, and promotion gates.
  - `run_colab.py` — self-contained runner (zero `cargo`/Rust deps, atomic partial resume, `assert total_scored > 0`).
