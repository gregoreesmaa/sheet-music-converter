# PLAN — `2026-11-27T08-47-eval-iter14-mid-suite31-and-northstar`

## 1. Platform Chosen & Rationale

- **Platform:** Colab L4 (`provider: "colab_l4"`, `gpu: "L4"`, Wave 13, Priority 0 — TOP priority; BF16-safe for both L4 cc 8.9 and H100 cc 9.0).
- **Estimated Wall-Clock / Budget:** ~15 minutes (~1.2 CU on L4) for both 90-page `suite31_hw` and 125-page `OMR-NorthStar`.
- **Local RTX 4080 SUPER Impact:** Zero (strictly cloud GPU; checkpoint was snapshotted read-only to `D:\tmp\iter14_mid_snapshot.pt` without touching the active Iter 14 training directory).

## 2. Identifiers & HF Artifacts

- **Experiment ID:** `2026-11-27T08-47-eval-iter14-mid-suite31-and-northstar`
- **System Name:** `uni-omr-iter14-v2a-step6874-ema`
- **HF Checkpoint:** `checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt`
  - Size: `165,552,472` bytes
  - SHA-256: `9e35c717b60f7d49cf9d8241c27b6b43459fece5de17232d0824ae04be9a7bf3`
  - Step: `6874` EMA (`ema_decay=0.9985`, `best_val_ser=0.49964`)
  - Vocab: `265` (`bekern+v2` Phase A), `41,356,299` params
- **HF Eval Datasets:**
  - `datasets/suite31_hw_eval.tar` (45 pieces × 2 = 90 pages)
  - `datasets/northstar_eval_pack_125p.tar.gz` (125 pages)
- **External Repos (public read-only clones on cloud worker):**
  - `https://github.com/gregoreesmaa/omr-northstar.git` (`1beaec4`+) for latest `cer.py` scorer.
  - `https://github.com/gregoreesmaa/mscz-rs.git` (`b8cb626`+) built via `cargo` if `tools/assembly-serialize-linux-x86_64` lacks `bekern+v2` dynamics/expressive support, with clean fallback to `tools/assembly-serialize-linux-x86_64` and `dual_verifier.bekern_to_musicxml_fallback`.
- **HF Upload Targets (`gregoreesmaa/sheet-music-converter-omr`):**
  - `evals/2026-11-27T08-47-eval-iter14-mid-suite31-and-northstar.json`
  - `evals/2026-11-27T08-47-eval-iter14-mid-suite31-and-northstar_per_page.json`
  - `evals/2026-11-27T08-47-eval-iter14-mid-suite31-and-northstar_bekern.json`
  - `runs/2026-11-27T08-47-eval-iter14-mid-suite31-and-northstar.json`

## 3. Two-Pass Evaluation Contract

### Pass 1: `suite31_hw` (90 pages)
- Invokes `tmp-train/mvd/eval_uni_suite31.py` with:
  - `--max-dim 1408 --max-len 3072 --max-mem-tokens 4096 --use-grammar --use-mtp-spec --bf16` (on cc>=8).
- Prints and records exact `printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`, `all45_clean`, and `all45_photo` metrics (`note_f1_mean`, `tok_f1_mean`, `tok_ser_mean`, `decode_ok_rate`).

### Pass 2: `OMR-NorthStar` (125 pages)
- Full-page inference with `max_dim=1408`, `max_len=3584`, `max_mem_tokens=4096`, `use_grammar=True`, `use_mtp_spec=True`, `neg_control_gate=True`, `bf16` on cc>=8.
- Exports MusicXML via `bekern+v2`-capable Rust exporter (`mscz-rs` `b8cb626`+ if built/available, else `tools/assembly-serialize-linux-x86_64`, else Python fallback) and persists `pred_raw` + `bekern` per page for exact local re-export if needed.
- Scores via `eval_northstar_runner.evaluate_manifest` against latest `omr-northstar` (`1beaec4`+) and prints exact `NS Score`, `Tier-W NS`, `CDR/ALS/RWP/HWM`, `S_note/S_struct/S_expr/S_play`, `median_wFPP`, `mean_NTSR`, and `NMHR`.

## 4. Decision Tree

- **RECORD & COMPARE:** Benchmark Iter 14 v2a (`step 6874` EMA, `265`-vocab `bekern+v2`) against Iter 13 v2 (`237`-vocab) on both `suite31_hw` and `OMR-NorthStar` (`NS`, `S_expr`, `S_note`, `HWM`, `NMHR`).
- **VERIFY:** Confirm `bekern+v2` Phase A expressive tokens (`S_expr`) are translating into real MusicXML gains without regressing `S_note` or `suite31_hw` Strict Note F1.
