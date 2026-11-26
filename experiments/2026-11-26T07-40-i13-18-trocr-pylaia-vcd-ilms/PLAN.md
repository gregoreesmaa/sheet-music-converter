# Experiment Plan — `2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms` (`i13-18`)

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe`.
- **Rationale:** Batched `B=2` KV-cached decoding (`[img_orig, img_ref]`) shares a single decoder forward pass per step and encodes each reference view only once per page. Evaluating 6 arms across 14 HW manuscripts (28 pages) + 10 printed pieces (20 pages) finishes in `~25–32 min` (`~2.0–2.6 CU`).

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms.json`
  - Run Ledger JSON: `runs/2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms.json`

## 3. Single-Variable Sweep Grid (`max_dim=1280`, `max_len=3840`, `β_plaus=0.10`, FSA)
| Arm Key | Reference View `x_ref` | Contrastive `α` | Plausibility `β` | Purpose |
|---|---|---:|---:|---|
| `baseline_alpha0.0` | none | `0.00` | — | Exact KV-cached FSA greedy control |
| `vcd_blur2_a0.15` | `GaussianBlur(r=2.0)` | `0.15` | `0.10` | Gentle high-frequency ink contrast (dots, accidentals, ledger lines) |
| `vcd_blur2_a0.30` | `GaussianBlur(r=2.0)` | `0.30` | `0.10` | Stronger high-frequency ink contrast |
| `vcd_blur4_a0.25` | `GaussianBlur(r=4.0)` | `0.25` | `0.10` | Medium-frequency staff/notehead contrast |
| `ilms_blank_a0.15` | uniform paper blank | `0.15` | `0.10` | Gentle PyLaia/TrOCR Internal-LM Subtraction |
| `ilms_blank_a0.30` | uniform paper blank | `0.30` | `0.10` | Stronger Internal-LM Subtraction |

- **Zero-Leakage Guarantee:** Uses strictly the test image $x$ and its self-derived blurred/blank counterpart $x_{\text{ref}}$; zero external eval data or GT statistics are ever touched.
- **Plausibility Guardrail:** Before applying `(1 + α) * logits_orig - α * logits_ref`, tokens with `P_orig(v) < 0.10 * max_w P_orig(w)` are masked to `-inf` (standard VCD adaptive plausibility constraint), preventing low-probability noise tokens from being boosted when `logits_ref` is very negative.

## 4. Checkpointing & Resume Plan
- Atomic JSON checkpointing (`_atomic_write_json`) to `/content/cache/evals/2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms_partial.json` every `<= 5.0` min and after every piece.

## 5. Evaluation Gate & Decision Tree
- **PROMOTE:** Any arm achieves `delta_note_f1_py >= +0.012` on `hw_clean` OR `hw_photo` (or `>= +0.006` on printed) with `wins > losses` and printed regression `>= -0.003` → wire `--vcd-alpha` / `--vcd-ref` into `eval_uni_suite31.py` (with optional confidence-gating so VCD only runs on uncertain steps for near-zero overhead).
- **ITERATE:** If `vcd_blur2_a0.15` helps HW pitch/accidental tokens while hurting structural tokens (`<bar>`, `<sys>`, headers) → gate VCD strictly to pitch/duration/accidental token IDs (leaving structural tokens uncontrasted).
- **DISCARD:** If all contrastive arms regress or yield `< +0.004` Note F1 → `UniOMR` decoder is already well-grounded visually and internal-LM subtraction is unnecessary.
