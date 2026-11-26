# Experiment Outcome — `2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch` (`ov-33`)

- **Status:** `COMPLETED` *(training succeeded cleanly; `suite31_hw` OOD evaluation hit a silent Linux `.exe` path bug -> `0/45` pieces scored)*
- **Completed (UTC):** `2026-11-26T13:06:06Z` (started `2026-11-26T12:09:48Z`)
- **Platform / GPU:** Google Colab `NVIDIA L4` (`24 GB`, `BF16=True`)
- **Elapsed Time:** `3377.22 s` (`56.3 min` compute; `3443.77 s` total queue wall)
- **Upstream SSL Init:** `hf:checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/uni_omr_v2_ijepa_best.pt` (`loaded=176` encoder tensors, `missing=173` decoder/aux heads, `unexpected=0`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch.json` (pulled to [`eval_result.json`](./eval_result.json))
  - Run Ledger: `runs/2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch.json` (pulled to [`run_result.json`](./run_result.json))

---

## Measured Results (`800` steps/arm, `small_45m`, `iter10_5_proxy_2k`, `1280px`, `BF16`)

| Arm | Init | Train Time (s) | Step-1 Loss (`ce`) | Step-400 Loss (`ce`) | Step-800 Loss (`ce`) | `suite31_hw` (`all45` / `hw` / `pr`) |
|---|---|---:|---:|---:|---:|---|
| `arm_a_scratch` | Random | `1635.5` | `7.8625` (`5.5864`) | `2.4667` (`1.6937`) | `2.3285` (`1.5614`) | **`null` (`0/45` scored — eval bug)** |
| `arm_b_vjepa_warmstart` | `exp2` V-JEPA 2.1 | `1637.6` | `7.8503` (`5.5783`) | `2.5047` (`1.6878`) | **`2.2603`** (**`1.5556`**) | **`null` (`0/45` scored — eval bug)** |
| **Delta (`B - A`)** | — | `+2.1 s` | `-0.0122` (`-0.0081`) | `+0.0380` (`-0.0059`) | **`-0.0682`** (`-0.0058`) | **UNSCORED** |

---

## Scrutiny & Anomaly Check

1. **CRITICAL SILENT FAILURE — `0/45` `suite31_hw` Pieces Scored (`eval_secs = 5.68 s / 5.10 s`):**
   - Although `run_colab.py` compiled the Linux Rust scorers (`has_rust_scorer: true`), `tmp-train/mvd/eval_sys_gate.py` hardcoded Windows `.exe` filenames (`ASER = .../assembly-serialize.exe`, `EHARNESS = .../eval-harness.exe`).
   - Every piece `[1/45]..[45/45]` in `eval_uni_suite31.py` logged `SKIP ref-error: [Errno 2] No such file or directory: '.../assembly-serialize.exe'` and exited with `rc=0`, writing an empty summary (`all 6 splits == null`).
   - **Root Fix Applied Immediately:** Patched `tmp-train/mvd/eval_sys_gate.py` lines 52–53 to use `_BIN_EXT = '.exe' if os.name == 'nt' else ''`, permanently fixing `eval_uni_suite31.py` on Linux (Colab/Modal/Kaggle).
2. **Secondary Checkpoint Selection Anomaly:** Both arms evaluated `uni_omr_v2_ema_best.pt (step=400)` instead of `step=800` (because `--eval-every 400` saved `ema_best` at step 400 and step 800 did not overwrite it), AND `arm_b` trained with full `lr=3e-4` on the pretrained encoder rather than a conservative `0.3×` encoder LR multiplier.
3. **Training Signal:** Both arms trained identically with `0` NaNs and `0` OOMs; `arm_b_vjepa_warmstart` finished `-0.0682` lower total loss (`2.2603` vs `2.3285`) at step 800, warranting a clean re-run with working OOD scoring.

---

## Verdict: **`ITERATE`**

Cannot promote or discard because the primary OOD metric (`suite31_hw` Strict Note F1) was completely unscored (`0/45` pieces) due to the Linux `.exe` path bug in `eval_sys_gate.py`.

## Follow-Up Action / Successor Experiment

- **Infra Fix Committed:** Patched `tmp-train/mvd/eval_sys_gate.py` (`ASER`/`EHARNESS` cross-platform extension).
- **Queued Successor:** [`2026-11-26T16-20-ov-33b-vjepa-warmstart-fixed-eval`](../2026-11-26T16-20-ov-33b-vjepa-warmstart-fixed-eval/README.md) (`depends_on: ["2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch"]` in `experiments/overnight_queue/queue.json`).
  - Re-runs `arm_a_scratch` vs `arm_b_vjepa_warmstart` with the fixed Linux Rust scorers, evaluates `uni_omr_v2_ema_latest.pt` (step 800) on BOTH `proxy_val` (CE/SER/Token-F1) AND `suite31_hw` (`assert n_scored == 45` fail-closed!), so we get a definitive apples-to-apples transfer verdict.
