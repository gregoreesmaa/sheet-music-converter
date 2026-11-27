# `2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair`

- **Status:** `PLANNED` (Wave 0 P0 cross-field proxy)
- **Short ID:** `i15-46`
- **Baseline Commit SHA:** `6055c49`
- **Cross-Field Origin:** Discrete Diffusion (`BD3-LM`, ICLR 2025 Oral — Block Discrete Denoising Diffusion Language Models + `ReMDM-conf` confidence-guided remasking repair).

## Summary & Hypothesis
Tests attaching a lightweight **bidirectional intra-cell masked-diffusion / remasking repair head (`BD3ReMDMRepairHead`)** (`< 1.5 MB` INT4: 2 tiny `d_rep=256` Transformer layers with block-local bidirectional self-attention + visual cross-attention + SwiGLU sharing `tok_emb` and `lm_head`) alongside the standard causal AR decoder of `UniOMRv2Net(edge85m, ax0_gl6, d=512)` under a controlled 2-arm 600-step BF16 Muon+AdamW proxy on `iter10_5_proxy_2k.tar`.

During training, the repair head is trained jointly with complementary masking (`15-40%` masked spans + `10%` synthetic single-token corruptions) so it learns both bidirectional infilling and local error correction grounded in the encoder's visual memory `mem_final`. At evaluation, we measure both standard causal AR `ema_val_ce` / `int4_dequant_val_ce` AND **2-step `ReMDM-conf` bidirectional repair accuracy** on (1) corrupted/masked ground-truth spans and (2) actual lowest-confidence AR-predicted spans.

## Quick Links
- [PLAN.md](PLAN.md)
- [run_colab.py](run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair.json`
- HF Partial Sync: `checkpoints/2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair/`
