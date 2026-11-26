# Plan — `2026-11-26T07-40-i13-02-anisotropic-patch-merger`

## 1. Academic Motivation & Literature Grounding

- **Primary Papers:**
  - Shuai Bai et al. (Alibaba Qwen Team), *"Qwen2.5-VL Technical Report"*, `arXiv:2502.13923` (Feb 2025): introduces the native dynamic-resolution MLP Patch Merger (`2×2` spatial grouping) for 4× visual token compression.
  - Antonio Ríos-Vila, Jorge Calvo-Zaragoza, David Rizo, and Thierry Paquet, *"Sheet Music Transformer++: End-to-End Full-Page Optical Music Recognition for Pianoform Sheet Music"*, `arXiv:2405.12105` (2024) & *"Sheet Music Transformer"*, ICDAR 2024: demonstrates end-to-end full-page polyphonic OMR where vertical staff-line/pitch resolution is the primary accuracy bottleneck.
  - Kenton Lee et al., *"Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding"*, ICML 2023 (`arXiv:2210.03347`): aspect-ratio-preserving variable-resolution patching for dense structured documents.
- **Relevance to `UniOMR v2` (`<55 MB` Edge OMR on `suite31_hw`):**
  - `PatchMerger2x2` downsamples `/8 -> /16` vertically, collapsing 3–4 diatonic pitch positions per token on `1280px` full pages (especially 10–14 staff historical manuscripts).
  - Switching to `PatchMergerAniso(kh=1, kw=4)` preserves `/8` vertical pitch resolution with **zero extra parameters** and **identical 4× token compression**, while `PatchMergerAniso(kh=1, kw=2)` (both unconstrained and budget-matched) tests whether preserving `/16` horizontal resolution alongside `/8` vertical resolution yields further gains on tight chord clusters and accidentals.

---

## 2. Platform Chosen & Identifiers

- **Platform:** Google Colab `L4` (24 GB VRAM, `cc=8.9`, native BF16).
- **Budget:** 4 controlled arms × `300` steps (`1280px` full-page) ≈ `30–34 min` (~`2.6 CU`).
- **Experiment ID:** `2026-11-26T07-40-i13-02-anisotropic-patch-merger`
- **Colab Session Name (recommended):** `i13-02-aniso-merge`
- **HF Private Dataset Hub (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T07-40-i13-02-anisotropic-patch-merger.json`
  - Run JSON: `runs/2026-11-26T07-40-i13-02-anisotropic-patch-merger.json`

---

## 3. Required Data & Single-Variable Design

- **Dataset Tar:** `datasets/iter10_5_proxy_2k.tar` on `gregoreesmaa/sheet-music-converter-omr` (permissive license only; 0 `CC-BY-NC`, 0 `suite31_hw` overlap).
- **Single Variable Tested:** Patch merger spatial kernel `(kh, kw)` in `HybridAxialMergerEncoder.patch_merger` (plus a budget-matched `1×2` arm isolating vertical resolution from token count), injected via `train_uni_v2.UniOMRv2Net` factory hook (zero edits to `uni_omr_v2.py` or `train_uni_v2.py`).

### Sweep Matrix (4 Arms)

| Arm | `(kh, kw)` | Effective Stride `(y × x)` | Compression | `keep_ratio / max_keep` | Merger Params Delta |
|---|---|---|---:|---|---:|
| `merger_2x2_control` | `(2, 2)` | `/16 × /16` | `4×` | `0.40 / 0.75` | `0` (baseline) |
| `merger_1x4_pitch_hi` | `(1, 4)` | `/8 × /32` | `4×` | `0.40 / 0.75` | `0` (exact parity) |
| `merger_1x2_pitch_hi` | `(1, 2)` | `/8 × /16` | `2×` | `0.40 / 0.75` | `-0.29M` (smaller) |
| `merger_1x2_matched_budget` | `(1, 2)` | `/8 × /16` | `2×` | `0.20 / 0.375` | `-0.29M` (same `k`) |

### Matched Hyperparameters Across All 4 Arms (`train_uni_v2.py`)

- `--preset small_45m`, `--optimizer muon`, `--lr 3e-4`, `--weight-decay 0.01`, `--lr-schedule wsd`, `--warmup-steps 45`, `--wsd-decay-frac 0.25`, `--min-lr-ratio 0.05`
- `--steps 300`, `--batch-size 2`, `--grad-accum 4`, `--curriculum fullpage`, `--fullpage-max-dim 1280`, `--max-seq-len 1536`, `--bf16`, `--grad-checkpoint`
- `--dynamic-routing`, `--max-mem-tokens 10240`, `--stroke-hog-weight 0.15`, `--gpu-ema-decay 0.997`, `--save-every-minutes 5.0`, `--eval-every 300`, `--eval-max-len 512`, `--seed 42`

---

## 4. Launch Commands

```powershell
.\.venv\Scripts\colab.exe new -s i13-02-aniso-merge --gpu L4
.\.venv\Scripts\colab.exe upload -s i13-02-aniso-merge /path/to/mvd_bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s i13-02-aniso-merge /path/to/local_hf_token /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s i13-02-aniso-merge -f experiments/2026-11-26T07-40-i13-02-anisotropic-patch-merger/run_colab.py
.\.venv\Scripts\colab.exe stop -s i13-02-aniso-merge
```

---

## 5. Evaluation Gate & Decision Tree

- **Measured Metrics per Arm:** Held-out `proxy_val.jsonl` (`64` pages): `token_f1`, `ser`, `sys_count_exact` (raw & EMA), `final_25step_loss`, `steps_per_sec`, `peak_vram_mb`, `params`.
- **PROMOTE:** Replace `PatchMerger2x2` default with the winning anisotropic `(kh, kw)` in `UniOMRv2Config` for Iter 13 if it beats `merger_2x2_control` by `>= +0.5 pp` EMA Token F1 (and lowers SER) while keeping `peak_vram_mb` within `< 15%` and `< 55 MB` INT4 edge size.
- **ITERATE:** If `merger_1x2_pitch_hi` wins strongly over both `2x2` and `1x4` (proving both `/8 y` AND `/16 x` are needed), test a dual-branch gated `1x2 + 2x2` merger or `keep_ratio` tuning (`0.25–0.35`) to recover full `2x2` speed.
- **DISCARD:** If `merger_2x2_control` matches or beats all anisotropic arms within `0.2 pp` Token F1, retain `PatchMerger2x2`.
