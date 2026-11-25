# `2026-11-26T01-38-ov-24-mixstyle-domain-invariance` — `MixStyle2d` Probability Sweep (`0.0` vs `0.20` vs `0.40`) for Printed->HW Cross-Domain Generalization

- **Status:** `PLANNED` (Overnight Queue Batch 04, Wave 5 — `experiments/overnight_queue/batch_04.json`)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params, Anisotropic FPN with `MixStyle2d` + Hybrid Axial/Merger/2D-RoPE Encoder + DeepStack GQA MTP-4 Decoder)
- **Datasets:**
  - `gregoreesmaa/sheet-music-converter-omr` -> `datasets/iter10_5_proxy_2k.tar` (**printed-only** train subset so HW is strictly unseen during training + `64`-page `proxy_val.jsonl` split into `printed` in-domain vs `hw` zero-shot cross-domain)
  - `gregoreesmaa/sheet-music-converter-omr` -> `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 31 printed + 14 real historical HW, `clean` + `photo` = 90 pages)

## Summary & Hypothesis
1-variable controlled domain-invariance ablation for Iter-11 `UniOMR v2`. All three arms train `UniOMR v2 small_45m` for `800` steps from identical scratch initialization (`seed=42`) on the **printed-only** subset of `proxy_train.jsonl` (`B=2 GA=4`, `fullpage_max_dim=1280`, `max_seq_len=1536`, native `BF16`, `--optimizer muon`, WSD `lr=3e-4`, `--augment --aug-prob 0.50`). **The ONLY variable changed is `--mixstyle-prob` (`0.0` vs `0.20` vs `0.40`):**
- **Arm A (`ms_00`):** `--mixstyle-prob 0.0` (no feature-level style mixing; pure pixel augmentation baseline)
- **Arm B (`ms_20`):** `--mixstyle-prob 0.20` (moderate `MixStyle2d` feature-statistics mixing)
- **Arm C (`ms_40`):** `--mixstyle-prob 0.40` (strong `MixStyle2d` feature-statistics mixing)

**Hypothesis:** Mixing per-instance CNN feature-map mean/std statistics across pages (`MixStyle2d` at `p=0.20`) regularizes the Anisotropic FPN against paper/ink contrast shifts, shrinking the zero-shot **printed->HW generalization gap** on both `proxy_val` HW pages and `suite31_hw` (`hw_clean` / `hw_photo`) without hurting in-domain printed Token F1 — whereas `p=0.40` may over-regularize fine stem/beam contrast and hurt printed engraving F1.

## Quick Links
- Self-contained Colab L4 runner: [run_colab.py](./run_colab.py)
- Experiment Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Manifest: [batch_04.json](../overnight_queue/batch_04.json)
- Outcome: `OUTCOME.md` *(to be written immediately after HF eval JSON is retrieved)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T01-38-ov-24-mixstyle-domain-invariance.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T01-38-ov-24-mixstyle-domain-invariance.json`
