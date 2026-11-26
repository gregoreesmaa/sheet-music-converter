# Experiment Plan — `2026-11-26T14-30-i13-18b-conf-gated-ilms-blank` (`i13-18b`)

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe`.
- **Rationale:** Native L4 BF16 is `~2.5–3×` faster than T4 FP32 (`i13-18`), and caching a single blank-page encoder memory + lazy chunked KV catch-up on uncertain steps (`max_prob < 0.85`) eliminates redundant decoder calls on confident printed tokens. Full 48-page sweep + composition phase finishes in `~30–40 min` (`~2.5–3.3 CU`).

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-26T14-30-i13-18b-conf-gated-ilms-blank`
- **Depends On:** `2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms`, `2026-11-26T07-40-i13-17-paddleocr-vrfm-band-router`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T14-30-i13-18b-conf-gated-ilms-blank.json`
  - Run Ledger JSON: `runs/2026-11-26T14-30-i13-18b-conf-gated-ilms-blank.json`

## 3. Sweep Grid (`iter10_soup_baseline`, `max_dim=1408`, `max_len=3584`, `β_plaus=0.10`, `conf_thr=0.85`, FSA)

### Phase 1 — Confidence-Gated & Token-Gated `ilms_blank` Sweep + `vrfm_band_0.30` Control
| Arm Key | `alpha` | `conf_thr` | `token_gate` | `vrfm_band_alpha` | Purpose |
|---|---:|---:|---|---:|---|
| `baseline_alpha0.0` | `0.00` | — | — | `0.00` | Exact KV-cached FSA greedy control |
| `ilms_conf85_a0.15` | `0.15` | `0.85` | `all` | `0.00` | Gentle confidence-gated ILMS |
| `ilms_conf85_a0.25` | `0.25` | `0.85` | `all` | `0.00` | Mid confidence-gated ILMS |
| `ilms_conf85_a0.30` | `0.30` | `0.85` | `all` | `0.00` | `i13-18` winner `α=0.30` with `max_prob < 0.85` gate |
| `ilms_conf85_a0.40` | `0.40` | `0.85` | `all` | `0.00` | Stronger confidence-gated ILMS ceiling probe |
| `ilms_conf85_pd_a0.30` | `0.30` | `0.85` | `pitch_dur` | `0.00` | Pitch/duration/accidental token-gated ILMS (`α=0.30`) |
| `ilms_conf85_pd_a0.40` | `0.40` | `0.85` | `pitch_dur` | `0.00` | Pitch/duration/accidental token-gated ILMS (`α=0.40`) |
| `vrfm_band_0.30_only` | `0.00` | — | — | `0.30` | Promoted `i13-17` VRFM control on `iter10_soup_baseline` |

### Phase 2 — Composition Arm (`vrfm_band_0.30` + Best `ilms_blank`)
- Automatically selects the top Phase-1 `ilms_blank` arm by `0.5 * (Δhw_clean + Δhw_photo)` subject to `Δpr_clean >= -0.002` (fallback: top HW average).
- Evaluates `comp_vrfm0.30_best_ilms` combining `vrfm_band_0.30` (3-row horizontal `F.avg_pool2d` hook on `saliency_net`) with the winning `ilms_blank` config across all `48` pages.

## 4. Integrity & Checkpointing
- **Pure-Python Scoring:** `_extract_musicxml_notes` + `rlvr_grpo.extract_pitch_dur_notes` + `rlvr_grpo.compute_sequence_f1` (zero `cargo` binary dependency).
- **Hard Coverage Assertion:** `assert n_scored > 0` and `assert n_scored == expected_pages` before uploading final JSON.
- **Checkpointing:** Atomic JSON save (`_atomic_write_json`) to `/content/cache/evals/2026-11-26T14-30-i13-18b-conf-gated-ilms-blank_partial.json` every `<= 5.0 min` and after every piece.

## 5. Evaluation Gate & Decision Tree
- **PROMOTE:** Winning `ilms_blank` arm (or `comp_vrfm0.30_best_ilms`) achieves `Δhw_photo >= +0.015` (or `Δhw_avg >= +0.012`) with `wins > losses`, `Δpr_clean >= -0.001`, `Δpr_photo >= -0.001`, and `< 15%` median printed latency overhead → promote confidence-gated ILMS (+ `vrfm_band_0.30` if additive) into `eval_uni_suite31.py`.
- **ITERATE:** If `conf_thr=0.85` still triggers on too many printed steps or `comp_vrfm0.30_best_ilms` saturates → tighten `conf_thr=0.70` or test on converged `UniOMR v2`.
- **DISCARD:** If `iter10_soup_baseline` already absorbs the `ilms_blank` gain (`Δhw_photo < +0.005`) → drop ILMS and keep `vrfm_band_0.30` alone.
