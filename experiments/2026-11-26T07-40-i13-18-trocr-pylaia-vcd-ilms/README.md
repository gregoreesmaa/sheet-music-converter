# `2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms` — TrOCR / PyLaia Visual-Contrastive Decoding (VCD) & Internal-LM Calibration on `suite31_hw`

- **Status:** `PROPOSED` (Iter 13 Competitor Scout 3 — Frontier Document/Math/HTR Transfer, Google Colab `NVIDIA L4`, `BF16`)
- **Inspiration:** **PyLaia / TrOCR** Internal Language Model Subtraction (ILMS) for domain-shifted historical handwriting + frontier VLM **Visual Contrastive Decoding (VCD)** (`logits_vcd = (1 + α) * logits(x) - α * logits(x_ref)` with adaptive plausibility truncation).
- **Baseline:** HF `checkpoints/iter10_5_best/uni_omr_ema_best.pt` (fallback `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`) evaluated at canonical `max_dim=1280`, `max_len=3840` on `suite31_hw` (all 14 HW pieces + 10 printed pieces = 48 pages).
- **Single Variable Tested:** **Contrastive weight $\alpha$ and visual reference view $x_{\text{ref}}$** (with adaptive plausibility constraint $\beta_{\text{plaus}}=0.10$ so impossible tokens are never promoted):
  - `baseline_alpha0.0` (standard KV-cached FSA greedy control)
  - `vcd_blur2_a0.15` (`x_ref = GaussianBlur(r=2.0)`, `α=0.15` — isolates fine accidentals, dots, ledger lines, and faint HW strokes)
  - `vcd_blur2_a0.30` (`x_ref = GaussianBlur(r=2.0)`, `α=0.30`)
  - `vcd_blur4_a0.25` (`x_ref = GaussianBlur(r=4.0)`, `α=0.25` — stronger low-pass reference)
  - `ilms_blank_a0.15` (`x_ref = uniform paper blank`, `α=0.15` — pure PyLaia/TrOCR Internal Language Model Subtraction)
  - `ilms_blank_a0.30` (`x_ref = uniform paper blank`, `α=0.30`)

## Summary & Hypothesis
On out-of-distribution historical handwritten manuscripts (`suite31_hw` HW), `UniOMR` achieves high Token F1 (`~0.76`) but low Strict Note F1 (`~0.22–0.26`). Like **TrOCR** on historical scripts, an autoregressive decoder trained predominantly on clean printed notation relies heavily on its **internal printed-music language model prior** whenever visual features are noisy or faint, hallucinating canonical printed rhythms/pitches instead of following subtle ink cues.

Subtracting $\alpha \cdot \text{logits}(x_{\text{ref}})$ (either a blank-paper image for pure **Internal-LM Subtraction**, or a Gaussian-blurred image for **Visual Contrastive Decoding**) gated by the adaptive plausibility mask $\{v : P(v|x) \ge \beta_{\text{plaus}} \max_w P(w|x)\}$ requires **zero external data** (0% leakage), only **1 extra encoder pass** per page, and batched `B=2` KV-cached decoding.

**Hypothesis:** Mild Visual Contrastive Decoding (`blur r=2.0`, `α ∈ [0.15, 0.30]`) or Internal-LM Subtraction (`blank`, `α=0.15`) will suppress printed-corpus hallucination on faint historical handwriting and sharpen high-frequency visual cues (dots, accidentals, ledger lines), lifting `hw_clean` / `hw_photo` Note F1 by `>= +0.012` without degrading printed pages (`>= -0.003`).

## Quick Links
- Execution Plan, Arms & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Hugging Face Eval Target: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms.json`
- Hugging Face Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms.json`
