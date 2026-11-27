# OUTCOME — `2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape` (`i15-50`)

- **Status:** `COMPLETED` (`rc=0`, `2/2` arms completed, `600/600` steps/arm, `muon+adamw`, `bf16=True`, `seed=42`, `error: null`)
- **Verdict:** **`DISCARD`** *(fails `PLAN.md` PROMOTE gate across all CE metrics and fails scrutiny on both auxiliary heads: overall `ema_val_ce` regressed by `+0.00235` (`1.12695 → 1.12930`), `raw_val_ce` regressed by `+0.00387` (`1.10372 → 1.10759`), `int4_dequant_val_ce` regressed by `+0.00109` (`1.12876 → 1.12985`), and `modifier_val_ce` gained only `-0.00715` (`1.34247 → 1.33532`, missing both the `-0.040` PROMOTE gate and the `-0.015` DISCARD floor); meanwhile the apparent `daq_register_shift_acc = 0.92969` is invalidated as an additive token-histogram shortcut rather than true visual-vs-symbol register verification, and `FAPE2DOffsetHead` suffered an adjacency-1 bug on non-adjacent pitch/octave tokens)*
- **Platform / GPU:** Local Windows 11 — `NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`)
- **Started / Finished (UTC):** `2026-11-27T22:10:03Z` → `2026-11-27T22:45:03Z` (`2,100.43 s` / `35.0 min` total)
- **Baseline Commit SHA:** `6055c49`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape.json` (pulled to [`run_hf.json`](run_hf.json))
  - Arm Syncs: `checkpoints/2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape/{ctrl,remora_daq_fape}_arm_result.json`

---

## Measured Results (`ctrl` vs `remora_daq_fape`, 600 steps/arm, `muon+adamw`, `seed=42`)

| Metric | `ctrl` (`edge85m + ax0_gl6`) | `remora_daq_fape` (`+Remora + 2D-FAPE + DAQ`) | Delta (`Arm 1 - Arm 0`) | Gate (`PLAN.md`) | Pass? |
|---|---:|---:|---:|---|---|
| **`ema_val_ce`** | `1.12695` | `1.12930` | **`+0.00235`** | `<= -0.015` PROMOTE (`<= 0` w/ mod CE) | **FAIL** (regressed) |
| **`int4_dequant_val_ce`** | `1.12876` | `1.12985` | **`+0.00109`** | `<= -0.015` PROMOTE | **FAIL** (regressed) |
| **`raw_val_ce`** | `1.10372` | `1.10759` | **`+0.00387`** | — | **REGRESSED** |
| `int4_dequant - ema` gap | `+0.00181` | `+0.00055` | `-0.00126` | `<= +0.025` | **PASS** |
| **`modifier_val_ce`** | `1.34247` | `1.33532` | **`-0.00715`** | `<= -0.040` PROMOTE (`< -0.015` DISCARD) | **FAIL** (`< -0.015` floor) |
| **`daq_register_shift_acc`** *(no `daq_auroc` logged)* | N/A | `0.92969` (`92.97%`) | — | `>= 0.80` | **INVALIDATED** *(shortcut; see Scrutiny #2)* |
| `train_loss_tail50` | `1.53764` | `1.55455` | `+0.01691` *(incl. aux FAPE+DAQ)* | — | — |
| **`packed_int4_mb`** *(exact disk bytes)* | `53.039 MB` (`53,039,072 B`) | `53.234 MB` (`53,234,008 B`) | `+0.195 MB` (`+412,164` pre-strip params) | `< 55.0 MB` (`FAPE=0.00 MB` stripped) | **PASS** |
| **`step_ms`** *(steady mean)* | `1,713.63 ms` | `1,513.30 ms` | `-200.33 ms` (`-11.69%`) | — | **HOST JITTER** *(see Scrutiny #4)* |
| `train_elapsed_sec` | `1,030.03 s` | `911.22 s` | `-118.81 s` (`-11.53%`) | — | — |
| **`peak_vram_mb`** | `11,087.0 MB` | `11,459.7 MB` | `+372.7 MB` (`+3.36%`) | `< 14,000 MB` WDDM safe | **PASS** |
| **`gate_pass`** *(evaluated vs `PLAN.md`; not emitted in JSON)* | baseline | — | — | PROMOTE gate | **FALSE → `DISCARD`** |

---

## Scrutiny & Anomaly Check

1. **Clean execution & budget integrity:**
   - Both arms completed `600/600` steps with verified `optimizer_mode == "muon+adamw"`, `bf16 == true`, `seed == 42`, zero NaNs, `error: null`, and safe VRAM (`11,087.0 MB` / `11,459.7 MB`, well below the `14,000 MB` Windows WDDM spill boundary).
   - Exact serialized Hybrid INT4 packing confirmed `FAPE2DOffsetHead` stripped cleanly (`0.00 MB`) and total `packed_int4_mb = 53.234 MB` (`+0.195 MB` vs `ctrl`'s `53.039 MB`, within the `< 55.0 MB` budget).
   - Note that `eval_hf.json` does not emit `daq_auroc`, `modifier_acc`, or `gate_pass` keys (only `modifier_val_ce` and `daq_register_shift_acc`).

2. **Why `daq_register_shift_acc = 0.92969` is a spurious token-histogram shortcut (NOT visual-vs-symbol verification):**
   - `DAQRegisterCriticHead.score_consistency` (`run_colab.py:211-216`) mean-pools `tok_proj(tok_embs)` and `vis_proj(mem_final)`, concatenates `[t_pool, v_pool]`, and applies a **single linear layer** `self.cls(self.norm([t_pool, v_pool]))`. A linear layer over concatenated features is purely additive (`w_t^T t_pool + w_v^T v_pool + b`) and **cannot compute any visual-vs-token cross-interaction**.
   - Simultaneously, `_get_step_and_oct_maps` (`run_colab.py:98-114`) constructs negatives by mapping octaves `0..3 → +1` and `4..8 → -1`, which collapses octaves `3` and `5` onto `4` and completely eliminates octaves `0` and `8`. Thus `w_t^T t_pool` trivially separates shifted negatives from real sequences by global octave-frequency histogram alone without inspecting the image `mem_final`.

3. **Why `FAPE2DOffsetHead` contributed near-zero signal and `RemoraModifierHead` underperformed:**
   - **Adjacency-1 bug in `FAPE2DOffsetHead` (`run_colab.py:185-188`):** `compute_fape_loss` only compares `sv[:, 1:]` vs `sv[:, :-1]` and `ov[:, 1:]` vs `ov[:, :-1]` (distance `= 1`), despite the comment claiming `distance <= 4`. In `uni_omr` beKern sequences (`[duration, step, octave, accidental?, dot?]`), pitch steps (`C..B`) and octaves (`0..8`) are separated by duration/separator tokens and are almost never immediately adjacent, leaving `valid_step` / `valid_oct` near zero.
   - **Small modifier gain & `use_cache` bypass in `RemoraModifierHead` (`run_colab.py:300-301`):** `RemoraModifierHead` attends globally over the same `mem_final` as the decoder (no local move-table window) with a small `tanh(gate)` residual (`init=0.05`), yielding only `-0.00715` on `modifier_val_ce` while regressing overall `ema_val_ce` (`+0.00235`), `raw_val_ce` (`+0.00387`), and `int4_dequant_val_ce` (`+0.00109`). Moreover, `_decode_with_remora` gates on `not use_cache`, so cached autoregressive decoding would bypass `RemoraModifierHead` altogether.

4. **Why `step_ms` was `-200.33 ms` faster on Arm 1 (`1,513.30 ms` vs `1,713.63 ms`) — host/cache jitter, not a speedup:**
   - Arm 1 adds `+412,164` parameters, `+372.7 MB` peak VRAM, an extra cross-attention layer, and two auxiliary heads, so it strictly executes more FLOPs/step than `ctrl`. On Local Windows 11 RTX 4080 SUPER, Arm 0 ran first on cold OS page cache / host contention while Arm 1 benefited from warm file cache over the `2,000` proxy PNGs (consistent with `i15-49` `ctrl` at `1,789.90 ms` on the same GPU).

---

## Verdict & Follow-Up Action

- **Verdict:** **`DISCARD`** — Do **not** port `RemoraModifierHead`, `FAPE2DOffsetHead`, or `DAQRegisterCriticHead` into Iter 15 (`uni_omr_v2.py`). Overall CE regressed across every split (`raw +0.00387`, `ema +0.00235`, `int4_dequant +0.00109`), `modifier_val_ce` improved by only `-0.00715` (failing both the `-0.040` PROMOTE gate and `-0.015` DISCARD floor), and the `0.92969` DAQ accuracy is a linear token-histogram artifact.
- **Follow-Up / Lessons Recorded:** No successor experiment is queued for Iter 15. If a visual-vs-symbol register critic or local-frame FAPE loss is ever revisited in a future iteration, (1) the critic must use cross-attention or bilinear visual-token matching with distribution-preserving negatives (random per-bar `±1` step/octave shifts), (2) 2D-FAPE must pair consecutive *note* pitch/octave tokens across intervening duration/modifier tokens (not `t` vs `t-1`), and (3) any modifier head must execute under `use_cache=True` and attend over local visual windows rather than duplicating full-page cross-attention.
