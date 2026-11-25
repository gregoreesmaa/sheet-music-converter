# Plan — `2026-11-26T01-38-ov-18-sobel-hog-weight-sweep`

## 1. Platform Chosen & Rationale

- **Platform:** Google Colab `L4` (24 GB VRAM, `cc=8.9`, native BF16, `torch 2.14.1+cu130` pre-installed).
- **Rationale:** 4 controlled arms × `400` optimizer steps at `1280px` + standard & faint-ink validation probes complete in ~35–40 min (~3.2 CU) on a single L4.

---

## 2. Identifiers for Follow-Up Queries

- **Experiment ID:** `2026-11-26T01-38-ov-18-sobel-hog-weight-sweep`
- **Colab Session Name (recommended):** `ov-18-sobel-hog`
- **Colab Commands:**
  - Status: `.\.venv\Scripts\colab.exe sessions`
  - Usage: `.\.venv\Scripts\colab.exe usage`
  - Stop (MANDATORY on finish): `.\.venv\Scripts\colab.exe stop -s ov-18-sobel-hog`
- **Hugging Face Private Dataset Hub (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T01-38-ov-18-sobel-hog-weight-sweep.json`
  - Run JSON: `runs/2026-11-26T01-38-ov-18-sobel-hog-weight-sweep.json`

---

## 3. Required Training & Eval Data

- **Dataset Tar:** `datasets/iter10_5_proxy_2k.tar` on `gregoreesmaa/sheet-music-converter-omr` (permissive license only; zero `CC-BY-NC`, zero `suite31_hw` eval overlap).
  - `proxy_train.jsonl`: 2,000 full-page training samples.
  - `proxy_val.jsonl`: 64 held-out full-page validation samples (evaluated unmodified AND under deterministic Faint-Ink Perturbation: `img_faint = 0.55 + 0.45 * img` in `[0, 1]`, preserving exact geometry/labels while halving stroke contrast).
- **Model Init:** Fresh `UniOMRv2Config.small_45m()` (`seed=42`) across all 4 arms.

---

## 4. Exact Launch Command & Hyperparameters

```powershell
# 1. Provision L4 session & upload code bundle + HF token (zero secrets in git)
.\.venv\Scripts\colab.exe new -s ov-18-sobel-hog --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-18-sobel-hog /path/to/mvd_bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-18-sobel-hog /path/to/local_hf_token /tmp/.hf_token

# 2. Execute standalone experiment runner
.\.venv\Scripts\colab.exe exec -s ov-18-sobel-hog -f experiments/2026-11-26T01-38-ov-18-sobel-hog-weight-sweep/run_colab.py

# 3. Tear down immediately after completion (zero idle CU burn)
.\.venv\Scripts\colab.exe stop -s ov-18-sobel-hog
```

### Matched Hyperparameters Across All 4 Arms (`train_uni_v2.py`)

- `--preset small_45m`
- `--optimizer muon`, `--lr 3e-4`, `--weight-decay 0.01`, `--lr-schedule wsd`, `--warmup-steps 50`, `--wsd-decay-frac 0.25`, `--min-lr-ratio 0.05`
- `--steps 400`, `--batch-size 2`, `--grad-accum 4`, `--curriculum fullpage`, `--fullpage-max-dim 1280`, `--max-seq-len 1536`, `--bf16`, `--grad-checkpoint`
- `--mtp-loss-weight 0.30`, `--saliency-loss-weight 0.10`, `--sys-y-loss-weight 0.25`, `--sys-count-loss-weight 0.15`, `--staff-seg-loss-weight 0.20`
- `--gpu-ema-decay 0.997`, `--save-every-minutes 5.0`, `--eval-every 400`, `--eval-max-len 512`, `--seed 42`

### Swept Variable (1 Variable Only: `--stroke-hog-weight`)

| Arm | `--stroke-hog-weight` | `use_stroke_hog_head` |
|---|---:|---|
| `hog_000` | `0.00` | `False` |
| `hog_005` | `0.05` | `True` |
| `hog_015` | `0.15` (default) | `True` |
| `hog_030` | `0.30` | `True` |

---

## 5. 5-Minute Checkpointing & Resume Plan

- Every arm runs with `--save-every-minutes 5.0`, triggering atomic `.tmp -> os.replace` rolling saves of `uni_omr_v2_latest.pt` and `uni_omr_v2_ema_latest.pt` under `/content/out/<exp_id>/<arm>/`.
- Completed arms write `<arm>/arm_done.json` and immediately upload updated `evals/<exp_id>.json` to HF Private Hub; re-running `run_colab.py` automatically skips completed arms.

---

## 6. Evaluation Gate & Decision Tree

- **Measured Metrics per Arm:**
  1. Standard `proxy_val.jsonl` (`64` pages): `token_f1`, `ser`, `sys_count_exact` (raw & EMA).
  2. **Faint-Ink Perturbation Probe** (`64` contrast-washed pages, `0.55 + 0.45 * img`): `faint_token_f1`, `faint_ser`, and robustness drop `delta_faint_f1 = faint_token_f1 - val_token_f1`.
  3. Final 25-step `ce_loss` & `stroke_hog_loss`, `steps_per_sec`, `peak_vram_mb`.
- **Promote:** Lock the winning `--stroke-hog-weight` into `train_uni_v2.py` default for Iter-11 if it improves `faint_token_f1` by `>= +0.5 pp` over `0.0` without regressing standard `val_token_f1` by more than `0.2 pp`.
- **Iterate:** If `0.30` is still improving both standard and faint-ink F1 monotonically, schedule a targeted follow-up testing `0.45` and `0.60`.
- **Discard:** If `0.0` matches or beats all non-zero weights on BOTH standard and faint-ink probes (within `0.2 pp`), set default `--stroke-hog-weight 0.0` to reclaim training throughput.
