# Experiment Plan — `2026-11-26T07-40-i13-14-photo-illum-flatten-autocrop`

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe` (`~4.8 CU/hr`).
- **Rationale:** Pure inference-time preprocessing sweep across 5 arms on 90 `suite31_hw` pages at canonical `1280px / 3840` using fast MTP-4 decoding. Estimated wall-clock: **~20–25 min** (`~1.8–2.0 CU`).

## 2. Identifiers for Follow-Up Queries
- **Colab Session Name (recommended):** `i13-14-photo-prep-l4`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T07-40-i13-14-photo-illum-flatten-autocrop.json`
  - Run Ledger JSON: `runs/2026-11-26T07-40-i13-14-photo-illum-flatten-autocrop.json`
- **Depends On:** None (runs immediately against `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`, and supports `HF_CKPT_PATH` env override for `UniOMR v2` checkpoints).

## 3. Required Data & Checkpoint (100% Permissive)
- **Eval Dataset:** HF `datasets/suite31_hw_eval.tar` -> `suite31/suite31_hw_manifest.json` (45 pieces x `clean`/`photo` = 90 pages).
- **Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (or `HF_CKPT_PATH` override).
- **Scoring Binaries:** Native Linux release builds of `assembly-serialize` and `eval-harness`.

## 4. Exact Sweep Grid & Mechanism
- **Fixed Inference Settings:** `max_dim=1280`, `max_len=3840`, `use_grammar=True`, `strict_grammar=False`, `use_mtp_spec=True`, `bf16=True`, `torch.backends.cuda.enable_cudnn_sdp(False)`.
- **5 Arms:**
  1. `baseline`: Raw RGB resize (`1280px`).
  2. `bg_divide_soft_a50`: Downsample 4x -> `ImageFilter.GaussianBlur(radius=max(8, round(0.01 * max_dim)))` -> bilinear upsample -> divide `img / max(bg, 32) * p98`, blend `alpha=0.50` with original. Preserves stroke sharpness and avoids CLAHE paper-grain amplification.
  3. `bg_divide_full_a85`: Same background division blended at `alpha=0.85`.
  4. `desk_autocrop_safe`: Scan inward from each of the 4 edges while outer row/col has `>= 65%` dark pixels (`luma < 90`) or `mean_luma < 85`, capped at `<= 18%` per edge; if any edge trims `>= 1%`, crop with a `2%` quiet safety margin. Guaranteed no-op on clean white-background pages.
  5. `bg_soft_plus_autocrop`: `desk_autocrop_safe` followed by `bg_divide_soft_a50`.
- **Implementation:** Self-contained in `run_colab.py` (pure PIL + NumPy, `< 15 ms/page`).

## 5. 5-Minute Checkpointing & Resume
- Atomic JSON checkpointing (`_partial.json` via `.tmp` -> `os.replace`) every `<= 5.0` min and after every completed arm; auto-resumes completed arms.

## 6. Evaluation Gate & Decision Tree
- **PROMOTE:** Any arm achieves `>= +0.008` Strict Note F1 on `printed_photo` OR `hw_photo` (or `>= +0.005` on `all45_photo`) with zero regression `> -0.002` on `clean` -> wire winning preprocessor into `eval_uni_suite31.py` and converter ingest stage.
- **ITERATE:** `desk_autocrop_safe` wins on `printed_photo` while `bg_divide` shifts `hw_clean` color distribution -> gate preprocessing on measured photo-shadow / dark-border detector so clean pages bypass untouched.
- **DISCARD:** All arms regress or stay within `[-0.002, +0.002]` -> confirm `UniOMR`'s trained photo augmentations already dominate classical preprocessing.
