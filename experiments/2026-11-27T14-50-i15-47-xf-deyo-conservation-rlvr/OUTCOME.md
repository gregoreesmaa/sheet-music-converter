# OUTCOME — `2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr` (`i15-47`)

- **Status:** `COMPLETED` (`rc=0`, `2/2` arms completed, `600/600` steps/arm, `muon+adamw`, `bf16=True`, `seed=42`, `error: null`)
- **Verdict:** **`DISCARD`** *(fails `PLAN.md` gate decisively: `val_rlvr_reward` **regressed** `-0.02060` (`0.06572 → 0.04512`, below the `< +0.010` DISCARD threshold), `val_r_note` regressed `-0.01869`, `val_r_parse` and `val_r_meter` remained `0.00000` on both arms, `ema_val_ce` regressed `+0.00328`, `peak_vram_mb` spiked to `25,083.2 MB` causing WDDM system-RAM thrashing, and `step_ms` increased `5.75×` overall / `~20×` during GRPO steps)*
- **Platform / GPU:** Local Windows 11 — `NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`)
- **Started / Finished (UTC):** `2026-11-27T20:08:46Z` → `2026-11-27T21:26:13Z` (`4647.3 s` / `77.5 min` total; `sft_ctrl` auto-resumed from HF Private Hub, `deyo_rlvr` resumed at step `489/600` and finished `600/600`)
- **Baseline Commit SHA:** `6055c49`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr.json` (pulled to [`run_hf.json`](run_hf.json))
  - Arm Syncs: `checkpoints/2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr/{sft_ctrl,deyo_rlvr}_arm_result.json`

---

## Measured Results (`sft_ctrl` vs `deyo_rlvr`, 600 steps/arm, `muon+adamw`, `seed=42`)

| Metric | `sft_ctrl` (`edge85m + ax0_gl6`) | `deyo_rlvr` (`450 SFT + 150 DeYO+GRPO`) | Delta (`Arm 1 - Arm 0`) | Gate (`PLAN.md`) | Pass? |
|---|---:|---:|---:|---|---|
| **`val_rlvr_reward`** *(primary)* | `0.06572` | `0.04512` | **`-0.02060`** | `>= +0.025` PROMOTE (`< +0.010` DISCARD) | **FAIL** |
| **`val_r_parse`** | `0.00000` | `0.00000` | `0.00000` | `> 0` / up | **FAIL** |
| **`val_r_meter`** *(bar conservation)* | `0.00000` | `0.00000` | `0.00000` | up (`>= +0.05` ITERATE) | **FAIL** |
| **`val_r_note`** *(edit similarity)* | `0.05385` | `0.03516` | **`-0.01869`** | up | **FAIL** |
| **`ema_val_ce`** | `1.12714` | `1.13042` | `+0.00328` | `<= +0.03` | **PASS** |
| **`int4_dequant_val_ce`** | `1.12954` | `1.13157` | `+0.00203` | `<= +0.03` | **PASS** |
| `raw_val_ce` | `1.10440` | `1.10755` | `+0.00315` | — | — |
| `int4_dequant - ema` gap | `+0.00240` | `+0.00115` | `-0.00125` | `<= +0.025` | **PASS** |
| `train_grpo_reward_tail20` (`grpo_R`) | N/A | `0.2642` | — | — | **FLAT** |
| `train_mean_plpd` | N/A | `0.0240` | — | `> 0.02` gate | **MARGINAL** |
| **`packed_int4_mb`** *(exact disk bytes)* | `53.039 MB` | `53.039 MB` | `0.000 MB` | `< 55.0 MB` | **PASS** |
| **`step_ms`** *(steady mean)* | `1,956.38 ms` | `11,252.40 ms` | **`+9,296.02 ms` (`5.75×`)** | — | **FAIL (severe overhead)** |
| `train_elapsed_sec` | `1,173.86 s` | `4,526.76 s` | `+3,352.90 s` (`3.86×`) | — | **FAIL** |
| **`peak_vram_mb`** | `11,150.1 MB` | **`25,083.2 MB`** | **`+13,933.1 MB` (`2.25×`)** | `< 14,000 MB` WDDM safe | **FAIL (WDDM spill)** |
| **`gate_pass`** | baseline | — | — | PROMOTE gate | **FALSE → `DISCARD`** |

---

## Scrutiny & Anomaly Check

1. **Cold-start reward collapse (`val_r_parse = 0.0`, `val_r_meter = 0.0`):**
   - At step `450` of a scratch 600-step proxy (`ema_val_ce ~ 1.13`), the model cannot yet produce syntactically complete multi-bar pages under free-running rollout decoding (`val_r_parse == 0.0`, `val_r_meter == 0.0` on both `sft_ctrl` and `deyo_rlvr`).
   - Consequently, the conservation reward signal is completely absent (`r_meter = 0`), and GRPO's group-normalized advantages on `G=4` noisy `T=0.8` rollouts (`grpo_R` frozen at `0.264` from step 500 to 600) only optimize noisy edit/anti-loop heuristics while halving the SFT weight (`LOSS_W_SFT = 0.50`), causing `val_r_note` to drop `-0.01869` (`0.05385 → 0.03516`) and `val_rlvr_reward` to drop `-0.02060`.
2. **Marginal DeYO PLPD signal (`train_mean_plpd = 0.0240`):**
   - Average PLPD barely cleared `DEYO_PLPD_MIN = 0.02`, indicating weak visual-vs-patch-shuffled probability separation at step 450–600 and adding negligible regularization benefit while paying 2 extra encoder+decoder passes per step.
3. **Severe VRAM blowup & WDDM thrashing (`peak_vram_mb = 25,083.2 MB`):**
   - Holding the frozen `ref_model` copy on GPU alongside differentiable `G=4` rollout decoding and DeYO dual-view forward passes pushed `peak_vram_mb` to `25.08 GB` on a `16 GB` RTX 4080 SUPER, triggering silent Windows WDDM shared-RAM paging. This blew step latency during steps `451..600` to `37,703–41,620 ms/step` (`~20×` slower than SFT's `1,956 ms/step`).
4. **Apples-to-apples integrity:**
   - Both arms verified `optimizer_mode == "muon+adamw"`, `bf16 == true`, `seed == 42`, `600/600` steps, and identical `53.039 MB` (`53,039,072 B`) Hybrid INT4 disk footprint. Note `train_loss_tail50` (`1.53803` vs `1.07101`) is not comparable across arms because Arm 1 scales `L_sft` by `0.50` and mixes in `0.25 * L_deyo + 0.25 * L_grpo`.

---

## Verdict & Follow-Up Action

- **Verdict:** **`DISCARD`** — Do **not** include DeYO+GRPO in the Iter 15 Stage-1/Stage-2 training recipe. Keep standard SFT (`muon+adamw` + QAT + EMA) plus zero-training inference-time verification (`i15-44b` / `i15-48`) and `BD3ReMDMRepairHead` (`i15-46`).
- **Operational note:** If conservation-reward RLVR is ever reconsidered after full Iter 15 convergence, only launch it from a converged checkpoint with `val_r_parse > 0.7` and `val_r_meter > 0.5`, offload `ref_model` log-probs or use LoRA/CPU-ref to keep VRAM `< 14 GB`, and never run it from a 450-step scratch proxy.
