# Plan — `2026-11-26T07-40-i13-01-visual-causal-flow-encoder`

## 1. Academic Motivation & Literature Grounding

- **Primary Paper:** Haoran Wei, Yaofeng Sun, and Yukun Li (DeepSeek-AI), *"DeepSeek-OCR 2: Visual Causal Flow"*, `arXiv:2601.20552` (Jan 2026). Introduces **DeepEncoder V2**, replacing rigid raster-scan visual tokenization with learnable **causal flow queries** and a **hybrid bidirectional-visual + causal-flow attention mask** so the encoder performs 2D-to-1D semantic reading-order reordering before the autoregressive decoder.
- **Supporting Paper:** Haoran Wei et al., *"General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model (GOT-OCR2.0)"*, `arXiv:2409.01704` (2024). Demonstrates unified end-to-end optical compression (`~256` visual tokens/page) across dense structured formats including sheet music (`**kern`).
- **Relevance to `UniOMR v2` (`<55 MB` Edge OMR on `suite31_hw`):**
  - `HybridAxialMergerEncoder` currently outputs raster-sorted `InkTokenRouterV2` tokens (`k <= 3,072`). On multi-system full-page printed and handwritten scores, raster order decouples simultaneous staves and forces decoder cross-attention to track 2D reading-order jumps solely from scratch.
  - Injecting `M` learnable causal flow queries inside the existing 4 `GlobalEncoderLayer` blocks adds `<0.1M` parameters (`<0.1 MB` INT4) and preserves `DeepStackFusion` alignment across `(mem_post_merger, mem_global_mid, mem_final)`.

---

## 2. Platform Chosen & Identifiers

- **Platform:** Google Colab `L4` (24 GB VRAM, `cc=8.9`, native BF16).
- **Budget:** 4 controlled arms × `300` steps (`1280px` full-page) ≈ `30–34 min` (~`2.6 CU`).
- **Experiment ID:** `2026-11-26T07-40-i13-01-visual-causal-flow-encoder`
- **Colab Session Name (recommended):** `i13-01-vcf`
- **HF Private Dataset Hub (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T07-40-i13-01-visual-causal-flow-encoder.json`
  - Run JSON: `runs/2026-11-26T07-40-i13-01-visual-causal-flow-encoder.json`

---

## 3. Required Data & Single-Variable Design

- **Dataset Tar:** `datasets/iter10_5_proxy_2k.tar` on `gregoreesmaa/sheet-music-converter-omr` (permissive license only; 0 `CC-BY-NC`, 0 `suite31_hw` overlap).
  - `proxy_train.jsonl`: 2,000 full-page training samples.
  - `proxy_val.jsonl`: 64 held-out full-page validation samples.
- **Single Variable Tested:** Visual Causal Flow query count `n_flow_queries` & memory fusion mode (`append` vs `replace`) inside `HybridAxialMergerEncoder.global_layers`, injected via a clean runtime wrapper around `train_uni_v2.UniOMRv2Net` (zero edits to `uni_omr_v2.py` or `train_uni_v2.py`).

### Sweep Matrix (4 Arms)

| Arm | `n_flow_queries` | `vcf_mode` | Memory Length Passed to Decoder | Extra Params |
|---|---:|---|---|---:|
| `control_raster` | `0` | `none` | `L_mem` (raster-ordered) | `0` |
| `vcf_append_64` | `64` | `append` | `L_mem + 64` (2D raster + 1D flow) | `+49.2k` (`+0.11%`) |
| `vcf_append_128` | `128` | `append` | `L_mem + 128` (2D raster + 1D flow) | `+98.3k` (`+0.23%`) |
| `vcf_compress_256` | `256` | `replace` | `256` (pure DeepEncoder-V2 compression) | `+196.6k` (`+0.45%`) |

### Matched Hyperparameters Across All 4 Arms (`train_uni_v2.py`)

- `--preset small_45m`, `--optimizer muon`, `--lr 3e-4`, `--weight-decay 0.01`, `--lr-schedule wsd`, `--warmup-steps 45`, `--wsd-decay-frac 0.25`, `--min-lr-ratio 0.05`
- `--steps 300`, `--batch-size 2`, `--grad-accum 4`, `--curriculum fullpage`, `--fullpage-max-dim 1280`, `--max-seq-len 1536`, `--bf16`, `--grad-checkpoint`
- `--keep-ratio 0.40`, `--dynamic-routing`, `--max-mem-tokens 10240`, `--stroke-hog-weight 0.15`, `--gpu-ema-decay 0.997`, `--save-every-minutes 5.0`, `--eval-every 300`, `--eval-max-len 512`, `--seed 42`

---

## 4. Launch Commands

```powershell
.\.venv\Scripts\colab.exe new -s i13-01-vcf --gpu L4
.\.venv\Scripts\colab.exe upload -s i13-01-vcf /path/to/mvd_bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s i13-01-vcf /path/to/local_hf_token /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s i13-01-vcf -f experiments/2026-11-26T07-40-i13-01-visual-causal-flow-encoder/run_colab.py
.\.venv\Scripts\colab.exe stop -s i13-01-vcf
```

---

## 5. Evaluation Gate & Decision Tree

- **Measured Metrics per Arm:** Held-out `proxy_val.jsonl` (`64` pages): `token_f1`, `ser`, `sys_count_exact` (raw & EMA), `final_25step_loss`, `steps_per_sec`, `peak_vram_mb`, `params`.
- **PROMOTE:** Wire `n_flow_queries` + `vcf_mode` into `UniOMRv2Config` / `HybridAxialMergerEncoder` for Iter 13 if the best VCF arm beats `control_raster` by `>= +0.5 pp` EMA Token F1 (or `>= +2.0 pp` `sys_count_exact` with non-regressing Token F1) at `< 5%` throughput cost.
- **ITERATE:** If `vcf_append_128` beats `vcf_append_64` monotonically (or `vcf_compress_256` achieves near-parity at `>1.3x` faster decoding), queue a follow-up sweep on `M in {192, 256}` + auxiliary `<sys>`-alignment supervision on flow queries.
- **DISCARD:** If `control_raster` matches or beats all VCF arms within `0.2 pp` Token F1 and `sys_count_exact`, keep pure raster-ordered 2D-RoPE memory.
