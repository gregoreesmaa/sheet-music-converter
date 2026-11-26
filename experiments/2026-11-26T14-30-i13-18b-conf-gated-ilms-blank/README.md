# `2026-11-26T14-30-i13-18b-conf-gated-ilms-blank` — Confidence- & Token-Gated `ilms_blank` + `vrfm_band_0.30` Composition

- **Status:** `PLANNED` (`ITERATE` follow-up to `2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms`)
- **Platform:** Google Colab `1× NVIDIA L4` (`24 GB`, native `BF16`, `~30–40 min`, `~2.5–3.3 CU`)
- **Baseline Commit SHA:** `36484e2`
- **Base Checkpoint:** `gregoreesmaa/sheet-music-converter-omr` → `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (fallback: `checkpoints/iter10_5_best/uni_omr_ema_best.pt`)
- **Eval Dataset:** `datasets/suite31_hw_eval.tar` (`14` HW + `10` printed × `{clean, photo}` = `48` pages, `max_dim=1408`, `max_len=3584`, pure-Python MusicXML Strict Note F1, zero `cargo` dependency)

## Motivation
In `i13-18` (`2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms`), Internal-LM Subtraction against a blank paper page (`ilms_blank_a0.30`) delivered the largest inference-time `hw_photo` gain of the batch (**`+2.68 pp`**, `8/0/6` W/T/L) and also improved `pr_photo` (`+0.21 pp`), whereas `vcd_blur` was noisy and discarded. However, unconditional step-by-step `ilms_blank_a0.30` suffered two deployment blockers:
1. A tiny `-0.36 pp` dip on `pr_clean` (missing the `>= -0.30 pp` gate by `0.06 pp`), despite changing only `1.3` tokens/page.
2. `+50%` latency overhead from encoding a blank image per page and running the blank decoder branch on every token step.

## Method
1. **Single Cached Blank Encoder Memory:** Encode one canonical uniform paper-tone blank image ONCE per session (`0 ms` per-page reference encoder overhead).
2. **Confidence-Gated Lazy ILMS (`max_prob < 0.85`):** Only invoke the blank-LM KV branch (via lazy chunked KV catch-up) and apply `(1 + α) * logits_orig - α * logits_blank` when `max_prob < 0.85`. Confident printed tokens (`max_prob >= 0.85`) are guaranteed **`0.000` drift** and near-zero latency overhead.
3. **Pitch / Duration / Accidental Token-Gating:** Compare `token_gate="all"` vs `token_gate="pitch_dur"` (contrasting only pitch steps, octaves, durations, dots, and accidentals while leaving structural tokens `bar`, `<sys>`, `<eos>`, `clef-*`, `key*`, `time-*`, `rest`, `bar-rest`, `tuplet-*`, `tie-*`, `chord` uncontrasted).
4. **Composition with Promoted `i13-17` `vrfm_band_0.30`:** Compose the best `ilms_blank` arm with `i13-17`'s promoted 3-row horizontal staff-band saliency prior (`band_alpha=0.30`, `+1.42 pp` `hw_clean` / `+0.97 pp` `hw_photo` / `0.000` printed regression) in a clean 2×2 factorial attribution.

## Quick Links
- Launcher: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Parent Outcome: [../2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms/OUTCOME.md](../2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms/OUTCOME.md)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T14-30-i13-18b-conf-gated-ilms-blank.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T14-30-i13-18b-conf-gated-ilms-blank.json`
