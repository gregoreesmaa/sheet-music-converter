# OUTCOME — Exp 6 (`2026-11-26T00-55-trial-scribe-ttt-suite31-hw`)

- **Status:** `COMPLETED` (`2026-11-25T23:46Z`, `1,294.95 s` / `21.6 min` on Google Colab `NVIDIA L4`, `exp6-scribe-ttt` terminated cleanly)
- **HF Artifacts:**
  - `evals/2026-11-26T00-55-trial-scribe-ttt-suite31-hw.json`
  - `runs/2026-11-26T00-55-trial-scribe-ttt-suite31-hw.json`
- **Verdict:** **ITERATE** (`s=3` gives modest net gain on `hw_photo` `+0.0077` & `hw_clean` `+0.0013`; `s=5` over-adapts and regresses `-0.0219` on `hw_clean` → follow up with lower-LR / `encoder_norm_only` sweep in Overnight Exp 38).

## Measured Results (`suite31_hw`, All 14 Real Historical HW Manuscripts, Native Rust `eval-harness`)

| Domain | n | Baseline (`iter10_soup` + `<sys>` fix) | `s=3` Note F1 (`Δ`) | `s=3` W/T/L | `s=3` Med. Time | `s=5` Note F1 (`Δ`) | `s=5` W/T/L | `s=5` Med. Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hw_clean` | 14 | `0.2611` | **`0.2625`** (`+0.0013`) | `7 / 3 / 4` | `6.90 s` | `0.2393` (`-0.0219`) | `4 / 3 / 7` | `8.02 s` |
| `hw_photo` | 14 | `0.1916` | **`0.1994`** (`+0.0077`) | `6 / 5 / 3` | `6.78 s` | `0.1909` (`-0.0007`) | `6 / 3 / 5` | `7.81 s` |

## Scrutiny & Anomaly Check

1. **Bonus confirmation of our `<sys>`-loop fix (`9d73170`):** Even on the unchanged `iter10_soup` checkpoint, `hw_photo` jumped from **`0.1516` → `0.1916` (`+0.0400` Strict Note F1!)** purely from fixing the `<sys>`-loop loophole in `grammar.py` + `eval_uni_suite31.py`.
2. **Over-adaptation threshold clearly observed:** `s=3` beats baseline on both `hw_clean` (`7W/4L`) and `hw_photo` (`6W/3L`, with `hw-debussy-lindaraja photo +0.0702` and `hw-schubert-d810 photo +0.0241`), whereas `s=5` collapses clean pieces (`hw-debussy-lindaraja clean -0.1488`, `hw-debussy-la-plus-que-lente clean -0.1025`).
3. **Actionable next step:** Do NOT enable `s=5` TTA in production; Overnight **Exp 38** will sweep gentler `lr ∈ {1e-5, 3e-5, 1e-4}`, `steps ∈ {2, 4, 8}`, and `encoder_norm_only` vs `full_encoder` to capture the `+0.07` photo upside without clean regression.
