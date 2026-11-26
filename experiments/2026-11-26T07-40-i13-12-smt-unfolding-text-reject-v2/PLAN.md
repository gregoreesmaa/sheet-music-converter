# Experiment Plan — `2026-11-26T07-40-i13-12-smt-unfolding-text-reject-v2`

## 1. Competitor Mechanism & Hypothesis
- **PRAIG Sheet Music Transformer++ (`SMT++`):** Prevents vertical reading-order collapse on dense full-page polyphonic scores via **progressive system-count unfolding**—starting on low-system-count pages before progressively unlocking dense multi-system pages.
- **Audiveris (`TEXTS` Rejection):** Explicitly segregates titles, composer credits, tempo markings, page numbers, and lyrics so non-music glyphs never corrupt pitch/rhythm transcription.
- **Hypothesis:**
  1. **Arm B (`smt_progressive_unfolding`):** Ramping `max_systems` (`<=3 -> <=6 -> <=16`) over 1,200 steps improves `UniOMR v2` `val_ce_loss`, `val_token_f1`, and `val_sys_count_exact` over static full-page sampling (`Arm A`) at identical compute budget.
  2. **Arm C (`audiveris_text_clutter_reject`):** Injecting synthetic non-music text clutter (headers, page numbers, inter-staff lyrics, archival stamps) onto 50% of training pages with unchanged GT `bekern` dramatically improves `val_token_f1` and `val_sys_count_exact` on text-heavy scores (`proxy_val_text_clutter`) with zero regression on clean `proxy_val`.

## 2. Platform & Identifiers
- **Platform:** Google Colab `L4` (`24 GB`, `cc=8.9`, BF16 native), est. **~36 min** (`~2.9 CU`).
- **HF Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`).
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`proxy_train.jsonl` + `proxy_val.jsonl`).
- **Eval Artifact:** `evals/2026-11-26T07-40-i13-12-smt-unfolding-text-reject-v2.json`.
- **Run Ledger:** `runs/2026-11-26T07-40-i13-12-smt-unfolding-text-reject-v2.json`.

## 3. Fixed Hyperparameters & Controlled 3-Arm Design
All 3 arms run `train_uni_v2.py` (`--preset small_45m`, `--optimizer muon --muon-lr 1.0e-3 --lr 3.0e-4 --lr-schedule wsd --wsd-decay-frac 0.25 --min-lr-ratio 0.05 --gpu-ema-decay 0.99 --gpu-ema-every 1 --fullpage-max-dim 1280 --max-seq-len 2048 --batch-size 2 --grad-accum 4 --bf16 --grad-checkpoint --seed 42`) for `1,200` total steps:
1. **Arm A (`control_static_v2`):** `1,200` steps on `proxy_train_abs.jsonl` (`max_systems=16`).
2. **Arm B (`smt_progressive_unfolding`):** 3 chained stages of `400` steps each (`1,200` total):
   - Stage 1 (steps `1..400`): `proxy_train_maxsys3.jsonl` (`<= 3` systems, falls back to lowest-50% system count if `< 64` rows).
   - Stage 2 (steps `401..800`, `--init-from stage1/uni_omr_v2_latest.pt`): `proxy_train_maxsys6.jsonl` (`<= 6` systems).
   - Stage 3 (steps `801..1200`, `--init-from stage2/uni_omr_v2_latest.pt`): full `proxy_train_abs.jsonl` (`<= 16` systems).
3. **Arm C (`audiveris_text_clutter_reject`):** `1,200` steps on `proxy_train_text_clutter.jsonl` where a deterministic 50% of training images have synthetic non-music title blocks, composer credits, page numbers, tempo markings, and inter-staff lyric lines rendered onto paper margins/inter-staff bands (GT `bekern` unchanged).

## 4. Checkpointing & Safety
- `train_uni_v2.py` enforces `--save-every-minutes 5.0` atomic autosave; `run_colab.py` writes per-arm JSON results atomically and auto-resumes completed arms.
- `/tmp/.hf_token` read with `chmod 600` and deleted immediately and in `finally:`.
- `torch.backends.cuda.enable_cudnn_sdp(False)` enforced.

## 5. Promotion / Discard Gate
- **PROMOTE Arm B:** `smt_progressive_unfolding` beats `control_static_v2` by `<= -0.025` `val_ce_loss` OR `>= +0.015` `val_token_f1` (EMA) on `proxy_val` -> adopt progressive system-count unfolding (`max_systems` ramp) in Iter 13 full 24k training.
- **PROMOTE Arm C:** `audiveris_text_clutter_reject` beats `control_static_v2` by `>= +0.020` `val_token_f1` on `proxy_val_text_clutter` with `<= -0.005` max regression on clean `proxy_val` -> add synthetic text/lyric clutter augmentation to `UniFullPageDataset` for Iter 13.
- **DISCARD:** Neither treatment arm beats `control_static_v2` beyond noise (`< 0.005` F1 and `< 0.01` CE).
