# OUTCOME — `2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta` (`i15-49`)

- **Status:** `COMPLETED` (`rc=0`, `2/2` arms completed, `600/600` steps/arm, `muon+adamw`, `bf16=True`, `seed=42`, `error: null`)
- **Verdict:** **`DISCARD`** *(fails `PLAN.md` gate decisively on both components: `CrossMPTRefiner` slightly regressed `ema_val_ce` by `+0.00129` (`1.12668 → 1.12797`, missing the `-0.015` PROMOTE / `-0.005` DISCARD threshold) and was completely flat on `int4_dequant_val_ce` (`0.00000`, `1.12868 → 1.12868`) while adding `+58.02 ms/step` (`+3.24%`) and `+0.188 MB` INT4; 3-step zero-label Turbo TTA also slightly regressed degraded val CE by `+0.00038` (`1.14978 → 1.15016`, missing the `-0.010` PROMOTE / `-0.005` DISCARD threshold) with negligible syndrome reduction (`0.31462 → 0.31314`, `-0.00148`))*
- **Platform / GPU:** Local Windows 11 — `NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`)
- **Started / Finished (UTC):** `2026-11-27T21:26:28Z` → `2026-11-27T22:05:52Z` (`2,363.53 s` / `39.4 min` total)
- **Baseline Commit SHA:** `6055c49`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta.json` (pulled to [`run_hf.json`](run_hf.json))
  - Arm Syncs: `checkpoints/2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta/{ctrl,crossmpt_turbo}_arm_result.json`

---

## Measured Results (`ctrl` vs `crossmpt_turbo`, 600 steps/arm, `muon+adamw`, `seed=42`)

| Metric | `ctrl` (`edge85m + ax0_gl6`) | `crossmpt_turbo` (`+2L CrossMPT + 3-step Turbo TTA`) | Delta (`Arm 1 - Arm 0`) | Gate (`PLAN.md`) | Pass? |
|---|---:|---:|---:|---|---|
| **`ema_val_ce`** *(=`crossmpt_refined_ce` on Arm 1)* | `1.12668` | `1.12797` | **`+0.00129`** | `<= -0.015` PROMOTE (`> -0.005` DISCARD) | **FAIL** |
| **`int4_dequant_val_ce`** | `1.12868` | `1.12868` | **`0.00000`** | `<= -0.015` PROMOTE | **FAIL** |
| **`raw_val_ce`** | `1.10364` | `1.10371` | `+0.00007` | — | **FLAT** |
| `int4_dequant - ema` gap | `+0.00200` | `+0.00071` | `-0.00129` | `<= +0.025` | **PASS** |
| `degraded_val_ce_before_tta` (`n=16` pages) | N/A | `1.14978` | — | — | — |
| **`turbo_tta_ce`** (`degraded_val_ce_after_tta`, `n=16`) | N/A | `1.15016` | **`+0.00038`** *(after - before)* | `<= -0.010` PROMOTE (`> -0.005` DISCARD) | **FAIL** |
| `syndrome_before_tta` → `syndrome_after_tta` | N/A | `0.31462 → 0.31314` | `-0.00148` | down | **NEGLIGIBLE** |
| `train_loss_tail50` | `1.53779` | `1.55378` | `+0.01599` *(incl. aux `syn_loss`)* | — | — |
| **`packed_int4_mb`** *(exact disk bytes)* | `53.039 MB` (`53,039,072 B`) | `53.227 MB` (`53,227,026 B`) | `+0.188 MB` (`+331,521` params) | `< 55.0 MB` (`+refiner < 1.5 MB`) | **PASS** |
| **`step_ms`** *(steady mean)* | `1,789.90 ms` | `1,847.92 ms` | `+58.02 ms` (`+3.24%`) | — | **OVERHEAD** |
| `train_elapsed_sec` | `1,074.84 s` | `1,108.90 s` | `+34.06 s` (`+3.17%`) | — | — |
| **`peak_vram_mb`** | `11,115.0 MB` | `11,163.2 MB` | `+48.2 MB` (`+0.43%`) | `< 14,000 MB` WDDM safe | **PASS** |
| **`gate_pass`** | baseline | — | — | PROMOTE gate | **FALSE → `DISCARD`** |

---

## Scrutiny & Anomaly Check

1. **Clean execution & apples-to-apples integrity:**
   - Both arms completed `600/600` steps with verified `optimizer_mode == "muon+adamw"`, `bf16 == true`, `seed == 42`, zero NaNs, `error: null`, and safe VRAM (`11,115.0 MB` / `11,163.2 MB`, well below the `14,000 MB` Windows WDDM spill boundary).
   - All `16/16` degraded validation pages were scored for Turbo TTA. Note that `eval_hf.json` does not emit a separate `crossmpt_refined_ce` or `gate_pass` key: because `apply_crossmpt_refiner` wraps `model.decode_tokens`, Arm 1's `ema_val_ce` (`1.12797`) *is* the CrossMPT-refined EMA val CE, and `turbo_tta_degraded_val.degraded_val_ce_after_tta` (`1.15016`) is the Turbo TTA CE.
2. **Why `CrossMPTRefiner` failed to improve val CE (`+0.00129` EMA, `0.00000` INT4):**
   - **Hardcoded `bar_capacity = 4.0`:** `compute_bar_syndromes(..., bar_capacity=4.0)` assumes every bar sums to 4 quarter-note beats, ignoring 3/4, 6/8, 2/4, pickup bars, and multi-spine layouts in `iter10_5_proxy_2k.tar`. This injects noisy, mis-specified syndrome targets into both the attention K/V and the auxiliary `syn_loss`.
   - **Logit-magnitude projection (`logits.abs()`) loses class identity:** Unlike binary BPSK LLR decoding in CrossMPT (ICLR 2025), multi-class OMR logits over `V` tokens lose which class is winning when passed through `mag_proj(logits.abs())`, leaving the refiner unable to make targeted duration/dot corrections.
   - **Teacher-forced vs cached-inference gap:** `_decode_with_crossmpt` skips the refiner during single-token cached decoding (`if not use_cache and logits.shape[1] > 1`) and relies on `tgt_ids` for barline positions during teacher-forced CE eval; even with oracle bar boundaries, it still could not beat `ctrl`.
3. **Why 3-step zero-label Turbo TTA failed (`+0.00038` CE on degraded pages):**
   - `run_turbo_tta_page` generates only a `<= 32`-token greedy prefix (`pseudo_len = min(32, tgt.shape[1] - 1)`), which on full pages (`MAX_SEQ_LEN = 1536`) mostly spans clef/key/meter headers and rarely closes even a single complete bar. Optimizing `syn_loss + 0.15 * ent` on 32-token prefixes barely shifted the conservation syndrome (`0.31462 → 0.31314`, `-0.00148`) and slightly degraded full-page CE (`1.14978 → 1.15016`).

---

## Verdict & Follow-Up Action

- **Verdict:** **`DISCARD`** — Do **not** port `CrossMPTRefiner` or 3-step syndrome Turbo TTA into Iter 15 (`uni_omr_v2.py` or inference). Both components failed their `PLAN.md` gates (`ema_val_ce` `+0.00129`, `int4_dequant_val_ce` `0.00000`, `turbo_tta_ce` `+0.00038`) while adding `+58 ms/step` and `+0.188 MB` INT4.
- **Successor / Alternative:** Conservation-aware correction is better served by exact meter-aware discrete verification at decode time (`i15-44b` verifier-triggered KV-cache rewind + MBR, `i15-48` ORBGRAND/SOGRAND syndrome decoding) and `BD3ReMDMRepairHead` (`i15-46`) rather than a fixed-capacity `|z|` logit refiner or 32-token prefix TTA.
