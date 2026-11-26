# Experiment Plan — `2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode` (`i13-16`)

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe`.
- **Rationale:** Native BF16 + KV-cached prefix rollback keeps localized `beam=3` re-decoding fast (`<= 6` bars/page × `max_bar_toks=120`). Evaluating 5 arms across all 14 HW manuscripts (28 pages) + 10 printed pieces (20 pages) finishes in `~24–30 min` (`~2.0–2.4 CU`).

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode.json`
  - Run Ledger JSON: `runs/2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode.json`

## 3. Single-Variable Sweep Grid (`max_dim=1280`, `max_len=3840`, `beam=3`, `max_redecodes=6`)
| Arm Key | Uncertainty Metric | Trigger Threshold | Beam Size | Max Redecodes / Page | Purpose |
|---|---|---:|---:|---:|---|
| `baseline_greedy` | none | off | 1 | 0 | Exact KV-cached FSA greedy control |
| `conf_minp_0.65` | `bottom2_mean_pmax` | `< 0.65` | 3 | 6 | Conservative trigger (only severe ambiguity) |
| `conf_minp_0.75` | `bottom2_mean_pmax` | `< 0.75` | 3 | 6 | Balanced Mathpix confidence trigger |
| `conf_minp_0.82` | `bottom2_mean_pmax` | `< 0.82` | 3 | 6 | Sensitive trigger (catches subtle accidental/pitch hesitation) |
| `conf_margin_0.25` | `min_p1_minus_p2` | `< 0.25` | 3 | 6 | Top-2 margin trigger (specifically targets 2-way pitch/accidental ties) |

- **Acceptance Guardrail:** A re-decoded bar replaces the greedy bar only if its length-normalized log-probability strictly exceeds the greedy bar's length-normalized log-probability (zero risk of worse sequence likelihood).
- **Zero-Edit Architecture:** Implemented entirely inside `run_colab.py` using `model.encode_image` and `model.decode_tokens(..., use_cache=True)` without modifying `uni_omr.py`, `uni_omr_v2.py`, or `train_uni_v2.py`.

## 4. Checkpointing & Resume Plan
- Atomic JSON checkpointing (`_atomic_write_json`) to `/content/cache/evals/2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode_partial.json` every `<= 5.0` min and after every piece.

## 5. Evaluation Gate & Decision Tree
- **PROMOTE:** Any arm achieves `delta_note_f1_py >= +0.012` on `hw_clean` OR `hw_photo` (or `>= +0.006` on printed) with `wins > losses`, printed regression `>= -0.003`, and median printed latency overhead `<= +25%` → wire `--conf-redecode-thr` into `eval_uni_suite31.py` and compose with `ov-34b` rhythm repair.
- **ITERATE:** If `conf_margin_0.25` beats `bottom2_mean_pmax` on pitch/accidental recovery → test Mathpix's second trick: re-decoding triggered bars against a `1.25×` high-resolution memory bank (`1600px` encoder cache).
- **DISCARD:** If all arms yield `< +0.004` Note F1 or regress printed `> -0.005` → greedy FSA + MTP-4 already matches local beam optimality on single-bar horizons.
