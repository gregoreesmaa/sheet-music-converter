# Experiment Outcome — `2026-11-26T00-55-ablation-spinquant-int4-parity`

## Verdict

- **Multi-Target ONNX Export (`export_uni_v2_onnx.py --targets all`):** **PROMOTE (PASS)** — Both `small_45m` (`28.04 MB`) and `edge85m` (`50.73 MB`) WebGPU INT4 (`MatMulNBits`, `block_size=32`) bundles comfortably beat the **`< 55 MB`** WebGPU budget, with NPU INT8 (`40.90 MB` / `76.41 MB`) and CPU INT8 (`40.46 MB` / `75.69 MB`) exporting cleanly in `~21–25 s` each.
- **Offline SpinQuant Folding (`fold_hadamard_v2`) for Weight-Only INT4 (`QATInt4ControllerV2` / WebGPU `MatMulNBits`):** **DISCARD as default pre-export step** — While exact FP32 functional invariance passed (`max_abs_diff < 1e-3` across all presets and regimes), `fold_hadamard_v2` **degraded** weight-only INT4 (`block_size=32`) logit MSE, KL divergence, and top-1 agreement on **both** presets, **both** clean and `12×` outlier-stressed weights, and **all 4** `suite31_hw` buckets (`0.59×`–`0.72×` clean, `0.28×`–`0.35×` outlier-stressed).
- **Decision:** Keep standard unrotated `QATInt4ControllerV2` QAT training + direct `export_uni_v2_onnx.py` export as the default path for Iter-11 `UniOMR v2`. Retain `spinquant_fold.py` strictly as an **opt-in** utility for future W4A4 / W8A8 *activation-quantized* (e.g. NPU static INT8) pipelines.

---

## Execution Metadata

| Field | Value |
|---|---|
| Experiment ID | `2026-11-26T00-55-ablation-spinquant-int4-parity` |
| Baseline SHA | `9d73170` |
| Platform | Google Colab `NVIDIA L4` (`22.03 GB`, `cc=[8, 9]`), session `exp8-spinquant` (stopped immediately on completion) |
| Runtime Versions | `torch=2.14.1+cu130`, `onnx=1.23.2`, `onnxruntime=1.31.1` |
| Timestamps (UTC) | `2026-11-25T23:38:27Z` → `2026-11-25T23:39:39Z` (**`72.86 s`** total wall-clock; `~0.10 CU`) |
| Eval Pages | 12 real `suite31_hw` pages (`1024px` max-dim, multiples of 16): 3 `printed_clean` (`repeats-melody`, `fiddle-strathspey`, `bach-bwv66.6`), 3 `printed_photo`, 3 `hw_clean` (`hw-beethoven-op135`, `hw-mozart-k464`, `hw-mozart-k387`), 3 `hw_photo` |
| HF Eval Artifact | `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T00-55-ablation-spinquant-int4-parity.json` |
| HF Run Ledger | `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T00-55-ablation-spinquant-int4-parity.json` |

---

## Exact Results

### 1. FP32 Invariance & Weight-Only INT4 (`block_size=32`) Parity

| Preset | Params | Regime | FP32 Worst `max_abs_diff` | Orig INT4 MSE | Folded INT4 MSE | MSE Ratio (`orig/fold`, `>1` better) | Orig INT4 KL | Folded INT4 KL | KL Ratio | Orig Top-1 | Folded Top-1 | Top-1 Δ |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `small_45m` | `40.94M` | Clean (`seed=42`) | `1.013e-06` | `1.0955e-02` | `1.8621e-02` | **`0.588×`** | `5.3482e-03` | `9.2729e-03` | **`0.577×`** | `75.65%` | `69.27%` | **`-6.38 pp`** |
| `small_45m` | `40.94M` | 12× Outlier (`seed=1234`) | `3.633e-04` | `12.825` | `46.636` | **`0.275×`** | `1.3645` | `5.4724` | **`0.249×`** | `54.82%` | `24.87%` | **`-29.95 pp`** |
| `edge85m` | `77.58M` | Clean (`seed=42`) | `1.073e-06` | `1.4292e-02` | `1.9852e-02` | **`0.720×`** | `7.3798e-03` | `1.0082e-02` | **`0.732×`** | `70.44%` | `62.37%` | **`-8.07 pp`** |
| `edge85m` | `77.58M` | 12× Outlier (`seed=1234`) | `8.974e-04` | `20.083` | `56.610` | **`0.355×`** | `2.2125` | `6.0104` | **`0.368×`** | `53.39%` | `25.46%` | **`-27.93 pp`** |

*Consistency across all 4 `suite31_hw` buckets:*
- `small_45m` clean MSE ratios: `printed_clean=0.602×`, `printed_photo=0.591×`, `hw_clean=0.586×`, `hw_photo=0.576×`.
- `small_45m` outlier MSE ratios: `printed_clean=0.284×`, `printed_photo=0.278×`, `hw_clean=0.265×`, `hw_photo=0.273×`.
- `edge85m` clean MSE ratios: `printed_clean=0.704×`, `printed_photo=0.693×`, `hw_clean=0.745×`, `hw_photo=0.735×`.
- `edge85m` outlier MSE ratios: `printed_clean=0.335×`, `printed_photo=0.341×`, `hw_clean=0.368×`, `hw_photo=0.376×`.

### 2. Multi-Target ONNX Export (`export_uni_v2_onnx.py --targets all`)

| Preset | Params | Export Wall | Target | Quant Format | Encoder (MB) | Decoder (MB) | Total (MB) | `< 55 MB` WebGPU Gate |
|---|---:|---:|---|---|---:|---:|---:|---|
| `small_45m` | `40.94M` | `20.93 s` | `webgpu` | `int4_matmul_nbits_b32` | `14.932` | `13.106` | **`28.038`** | **PASS** |
| `small_45m` | `40.94M` | `20.93 s` | `npu` | `int8_dequantize_linear` | `20.791` | `20.109` | `40.900` | *(NPU target)* |
| `small_45m` | `40.94M` | `20.93 s` | `cpu` | `int8_dynamic_quint8` | `20.373` | `20.091` | `40.463` | *(CPU target)* |
| `edge85m` | `77.58M` | `24.78 s` | `webgpu` | `int4_matmul_nbits_b32` | `22.585` | `28.150` | **`50.735`** | **PASS** |
| `edge85m` | `77.58M` | `24.78 s` | `npu` | `int8_dequantize_linear` | `32.855` | `43.551` | `76.406` | *(NPU target)* |
| `edge85m` | `77.58M` | `24.78 s` | `cpu` | `int8_dynamic_quint8` | `32.245` | `43.449` | `75.694` | *(CPU target)* |

---

## Scrutiny & Anomaly Check

1. **Rotation math is 100% bug-free — FP32 invariance passed cleanly:**
   - `small_45m` (`d_model=384`, `r1_blocks=[128,128,128]`, `r2_blocks=[16,16,16]`, `65` R1 linears, `24` R2 attention layers): worst FP32 `max_abs_diff = 1.013e-06` (clean) and `3.633e-04` (12× outlier).
   - `edge85m` (`d_model=512`, `r1_blocks=[512]`, `r2_blocks=[64]`, `81` R1 linears, `28` R2 attention layers): worst FP32 `max_abs_diff = 1.073e-06` (clean) and `8.974e-04` (12× outlier).
   - Therefore, the INT4 regression is **NOT** an implementation bug in `fold_hadamard_v2`; it is a **genuine mathematical mismatch** between SpinQuant R1 and weight-only W4A32/W4A16 group-wise quantization.

2. **Root Cause #1 (Outlier-stressed regime: `2.8×`–`3.6×` worse MSE): RMSNorm `gamma` absorption amplifies & smears spikes under W4A32/W4A16:**
   - SpinQuant (ICLR 2025) was designed for **W4A4 (weight + activation + KV-cache INT4)**, where `RMSNorm.weight` (`gamma`) *must* be absorbed into `W_in` (`W_in <- (W_in * gamma) @ Q`) so orthogonal `Q` commutes past `RMSNorm` to suppress *activation* quantization outliers.
   - Under our **weight-only W4A32/W4A16** quantizer (`QATInt4ControllerV2` and WebGPU `MatMulNBits`), activations and `RMSNorm.weight` stay in **exact FP32/FP16**!
   - When `_absorb_rmsnorm_into_linears` multiplies `gamma = 12.0` into `W_in` columns that *also* have `12× std` spikes on the same residual channels, it creates **`144×` spikes** inside the INT4-quantized weight matrix. Multiplying by `Q` then spreads that `144×` energy across **all `block_size=32` groups** in every row — whereas unrotated W4A32 keeps `gamma` exact in FP32 and isolates weight spikes to only the ~2% of channels / small fraction of `block_size=32` groups where they occur.

3. **Root Cause #2 (Clean regime: `1.39×`–`1.70×` worse MSE): Uniform → Gaussian CLT kurtosis penalty:**
   - Fresh PyTorch `nn.Linear` weights are initialized Kaiming-uniform (`U(-a, a)`, kurtosis `1.8`, bounded `max/std = sqrt(3) ≈ 1.73`), which is near-optimal for symmetric INT4 quantization.
   - Rotating uniform weights by a dense Walsh-Hadamard matrix `Q` invokes the Central Limit Theorem, converting bounded uniform weights into Gaussian weights (kurtosis `3.0`, heavier tails per 32-element block), inflating each block's absmax scale `s = max(|w|)/7` and increasing INT4 quantization error by `~1.4×–1.7×`.

4. **ONNX Export Log Check (Benign):**
   - `onnxruntime` logged `Ignore MatMul due to non constant B: /[/lm_head_1/MatMul]`, `/lm_head_2/MatMul`, `/lm_head_3/MatMul` during dynamic INT8 quantization of MTP auxiliary heads (weight-tied / non-constant B). All 3 targets (`webgpu`, `npu`, `cpu`) exported cleanly and passed size verification.

5. **Actionable Follow-Ups:**
   - **Iter-11 WebGPU/NPU/CPU release:** Use standard unrotated `QATInt4ControllerV2` QAT + `export_uni_v2_onnx.py --targets all` (already verified at **`28.04 MB`** for `small_45m` and **`50.73 MB`** for `edge85m`, both `< 55 MB`).
   - **If weight-only INT4 outlier mitigation is ever needed on a *trained* Iter-11 checkpoint:** Benchmark on **real trained weights** (not uniform init) and evaluate **AWQ / channel-wise scaling** or **R2-only** (without absorbing `RMSNorm.weight` into quantized weights), rather than full SpinQuant R1.
