# Plan — `2026-11-26T07-40-i13-03-window2d-highres-encoder`

## 1. Academic Motivation & Literature Grounding

- **Primary Papers:**
  - Shuai Bai et al. (Alibaba Qwen Team), *"Qwen2.5-VL Technical Report"*, `arXiv:2502.13923` (Feb 2025): replaces full self-attention in high-resolution ViT layers with native **2D Window Attention** before the `2×2` patch merger for linear scaling and superior fine-grained OCR localization.
  - Haoran Wei et al. (DeepSeek-AI), *"DeepSeek-OCR: Contexts Optical Compression"* (2025): combines high-resolution SAM local window attention with global attention in **DeepEncoder**.
  - Yixuan Li, Huaping Liu, Qiang Jin, Miaomiao Cai, and Peng Li, *"TrOMR: Transformer-Based Polyphonic Optical Music Recognition"*, ICASSP 2024 (`arXiv:2308.09370`): demonstrates 2D spatial transformer perception for polyphonic camera-captured sheet music.
- **Relevance to `UniOMR v2` (`<55 MB` Edge OMR on `suite31_hw`):**
  - At `1280px` input (`160×120` `/8` grid), `AxialEncoderLayer` runs two separate attention passes (`row` of length 120 + `col` of length 160 = 280 keys/token, 2 QKV/O projections) and slices across tilted handwritten staves.
  - A shifted `8×16` 2D local window (`64px × 128px`) spans a complete 5-line staff + ledger lines + 2–3 beats of polyphonic chords/accidentals/slurs even under `±10°` camera skew, using a single 2D-RoPE attention pass (`128` keys/token, half the attention projection weights of `AxialEncoderLayer`).

---

## 2. Platform Chosen & Identifiers

- **Platform:** Google Colab `L4` (24 GB VRAM, `cc=8.9`, native BF16).
- **Budget:** 4 controlled arms × `300` steps (`1280px` full-page) ≈ `30–34 min` (~`2.6 CU`).
- **Experiment ID:** `2026-11-26T07-40-i13-03-window2d-highres-encoder`
- **Colab Session Name (recommended):** `i13-03-win2d-enc`
- **HF Private Dataset Hub (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T07-40-i13-03-window2d-highres-encoder.json`
  - Run JSON: `runs/2026-11-26T07-40-i13-03-window2d-highres-encoder.json`

---

## 3. Required Data & Single-Variable Design

- **Dataset Tar:** `datasets/iter10_5_proxy_2k.tar` on `gregoreesmaa/sheet-music-converter-omr` (permissive license only; 0 `CC-BY-NC`, 0 `suite31_hw` overlap).
- **Single Variable Tested:** High-resolution `/8` encoder layer topology (`model.encoder.axial_layers`), injected cleanly via `train_uni_v2.UniOMRv2Net` factory hook with exact 5-arg `forward(x, row_cos, row_sin, col_cos, col_sin)` signature parity (zero edits to `uni_omr_v2.py` or `train_uni_v2.py`).

### Sweep Matrix (4 Arms)

| Arm | Layer 0 (`/8`) | Layer 1 (`/8`) | Window Shift (L0 / L1) | Param Delta vs Axial |
|---|---|---|---|---:|
| `axial_control` | `AxialEncoderLayer` | `AxialEncoderLayer` | N/A | `0` (baseline) |
| `win2d_8x8_shifted` | `Window2D(8×8)` | `Window2D(8×8)` | `(0,0)` / `(4,4)` | `-1.18M` (`-2.7%`) |
| `win2d_8x16_staff_strip` | `Window2D(8×16)` | `Window2D(8×16)` | `(0,0)` / `(4,8)` | `-1.18M` (`-2.7%`) |
| `hybrid_win8x16_then_axial` | `Window2D(8×16)` | `AxialEncoderLayer` | `(0,0)` / N/A | `-0.59M` (`-1.4%`) |

### Matched Hyperparameters Across All 4 Arms (`train_uni_v2.py`)

- `--preset small_45m`, `--optimizer muon`, `--lr 3e-4`, `--weight-decay 0.01`, `--lr-schedule wsd`, `--warmup-steps 45`, `--wsd-decay-frac 0.25`, `--min-lr-ratio 0.05`
- `--steps 300`, `--batch-size 2`, `--grad-accum 4`, `--curriculum fullpage`, `--fullpage-max-dim 1280`, `--max-seq-len 1536`, `--bf16`, `--grad-checkpoint`
- `--keep-ratio 0.40`, `--dynamic-routing`, `--max-mem-tokens 10240`, `--stroke-hog-weight 0.15`, `--gpu-ema-decay 0.997`, `--save-every-minutes 5.0`, `--eval-every 300`, `--eval-max-len 512`, `--seed 42`

---

## 4. Launch Commands

```powershell
.\.venv\Scripts\colab.exe new -s i13-03-win2d-enc --gpu L4
.\.venv\Scripts\colab.exe upload -s i13-03-win2d-enc /path/to/mvd_bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s i13-03-win2d-enc /path/to/local_hf_token /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s i13-03-win2d-enc -f experiments/2026-11-26T07-40-i13-03-window2d-highres-encoder/run_colab.py
.\.venv\Scripts\colab.exe stop -s i13-03-win2d-enc
```

---

## 5. Evaluation Gate & Decision Tree

- **Measured Metrics per Arm:** Held-out `proxy_val.jsonl` (`64` pages): `token_f1`, `ser`, `sys_count_exact` (raw & EMA), `final_25step_loss`, `steps_per_sec`, `peak_vram_mb`, `params`.
- **PROMOTE:** Promote the winning 2D-window or hybrid `/8` encoder topology into `UniOMRv2Config` for Iter 13 if it beats `axial_control` by `>= +0.4 pp` EMA Token F1 (or matches within `0.1 pp` while delivering `>= 10%` faster `steps_per_sec` AND lower parameter count).
- **ITERATE:** If `win2d_8x16_staff_strip` beats `8x8`, test a 3-layer `[Win(8×16), Win(8×16, shifted), Axial]` stack (matching baseline param count) to combine 2D skew-robust local perception with global staff-line propagation.
- **DISCARD:** If `axial_control` outperforms all 2D-window arms by `> 0.2 pp` Token F1, retain pure 2× `AxialEncoderLayer`.
