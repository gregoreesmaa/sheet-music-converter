# `2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder` — Competitive System Slot-Attention Memory Proxy

- **Experiment ID:** `2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder`
- **Short ID / Wave:** `i15-39` (Wave 13 — Weight-Incompatible Architecture Proxies)
- **Status:** `QUEUED`
- **Baseline Commit SHA:** `cfc94aa`
- **Platform / GPU:** Colab `1× NVIDIA H100` (`80 GB` VRAM, BF16 native, ~10–12 min total)
- **Arms (`600` steps/arm, BF16, `seed=42`, `grad_accum=4`, WSD, EMA `0.99`, QAT-INT4 at `step >= 420`):**
  - **Arm 0 (`ctrl`):** `edge85m + ax0_gl6` flat `mem_final` (`L_k` routed visual tokens).
  - **Arm 1 (`sysslot_mem`):** `edge85m + ax0_gl6` + 2-iteration vertically-ordered competitive **System Slot-Attention** (`S=16` slots, `d_attn=256`, `SwiGLU(512, 512)`) appended to `mem_final` (`L_k + 16` tokens).
- **Dataset (100% permissive):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, `2×` dense multi-staff & HW upsampling).

## Summary & Hypothesis
Flat routed visual memory (`L_k <= 3,072` patch tokens) forces the decoder to re-discover page-level system structure (system count, vertical staff bands, clef/key-signature context per system) from scratch at every `<sys>` boundary.

Appending `S=16` vertically-ordered competitive **System Slots** (2-iteration competitive Slot-Attention with softmax competition over slots `dim=1`, Gaussian vertical band prior `y_centers = linspace(0.03, 0.97, 16)`, `d_attn=256`, and `SwiGLU(512, 512)`) directly to `mem_final` (`L_k + 16`) adds `< 0.5%` memory tokens and `~1.1M` params (`~0.6 MB` INT4), giving every decoder layer a compact, top-to-bottom hierarchical summary of the page's systems.

## Quick Links
- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab H100 Runner: [run_colab.py](./run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder.json`
