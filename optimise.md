# Browser & Older-Device Model Optimization Plan (`optimise.md`)

**Goal:** Shrink the 2-stage neural OMR pipeline (`StaffUNet2D` + `4ch MvdNet` + `assembly-serialize`) to run 100% client-side in a web browser on **older devices (2019–2021 phones / 4 GB laptops, iOS Safari ~600–900 MB tab limit, WASM-SIMD CPU fallback + WebGPU acceleration)** while staying within **`~1–1.5%` F1** of the 34M flagship teacher.

*Source: 5-expert architectural & runtime audit (2026-11-23). Ideas & impact estimates only; execute after the flagship teacher clears our `0.90–0.95+` F1 target.*

---

## 1. End-to-End Target Comparison

| Stack | Total Params | Download (Brotli) | Peak Tab RAM | 2020 Phone (WASM-SIMD) | Modern WebGPU | Est. F1 Δ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flagship Baseline (`StaffUNet2D` v3 + `exp2B` FP32)** | `34.0M` | `~118 MB` | `~480–640 MB` *(Safari OOM)* | `~25–40 s` *(or OOM)* | `~3–4 s` | `0.0%` (ref) |
| **Flagship INT8 QAT (no arch change)** | `34.0M` | `~26 MB` | `~190–250 MB` | `~14–24 s` | `~2.0–3.0 s` | `-0.3%` |
| **Browser-Nano Distilled + INT8 QAT (`S1-Nano` + `S2-Femto/MNV4` + `3L-GQA` + `MTP`)** | **`~6.5–7.5M`** | **`~7–8 MB`** *(+3.5 MB ORT-WASM + 0.3 MB Rust `.mscz`)* | **`~95 MB`** *(safe on 2019 iOS)* | **`~3.0–5.2 s`** | **`~0.8–1.2 s`** | **`-1.0 to -1.5%`** |

---

## 2. Stage-2 Encoder (`27.8M` = 85% of Current Weights)

**Root bottleneck:** Under `stride=4x8`, ConvNeXt-Tiny Stage 4 (`3×768`) consumes **~55% of both encoder params (`15.4M / 27.8M`) and FLOPs**, and Stage-3 `mlp.fc1` peaks at **`~640 MB`** activation RAM on `W=3072` (instant iOS Safari OOM).

| Option | Total Params (Enc) | FP32 / INT8 | Enc FLOPs Δ | Est. F1 Δ (Raw / Distilled from `exp2B`) |
| :--- | :---: | :---: | :---: | :---: |
| **Drop Stage 4 only** (keep S1–S3, `384→256`) | `17.0M` (`12.4M`) | `68 / 18 MB` | `-55%` | `-0.4..-0.8%` / **`-0.1..+0.1%`** |
| **ConvNeXt-Pico** (`timm`, full) | `13.3M` (`8.6M`) | `53 / 14 MB` | `-66%` | `-1.0..-1.8%` / **`-0.1..-0.4%`** |
| **ConvNeXt-Femto** (`timm`, full) — *best acc/size* | **`9.5M` (`4.8M`)** | **`38 / 10 MB`** | **`-80%`** | `-1.8..-2.8%` / **`-0.3..-0.7%`** |
| **ConvNeXt-Atto** (`timm`, 3-stage no S4) | **`6.3M` (`1.6M`)** | **`25 / 6.5 MB`** | **`-93%`** | `-3.0..-4.5%` / **`-0.8..-1.5%`** |
| **MobileNetV4-Conv-S** (`timm`) — *fastest WASM-SIMD INT8* | **`7.2M` (`2.5M`)** | **`29 / 7.5 MB`** | **`-88%`** | `-2.5..-4.0%` / **`-0.7..-1.3%`** |
| **RepViT-M0.9 / StarNet-S1** | `7.6–9.8M` | `30–39 / 8–10 MB` | `-85%` | `-2.0..-3.5%` / `-0.5..-1.1%` |

- **Micro-glyph preservation (`foveal_tap` at `C0=40–64`):**
  - Replace the `4×4` patchify stem with two `3×3` stride-2 convs (or add a `20K`-param parallel `3×3` anti-alias micro-branch) so 1–2 px dots/stems/staff lines survive pre-P2.
  - Insert a `3×3` DWConv before `1×1` in `foveal_proj` (`+45K` params).
  - L2-distill `[256, H/4, W/8]` encoder memory directly against `exp2B`'s memory map.
  - Cap browser Stage-2 stride at `8x16` (or `4x8` with `W ≤ 2048`) to keep peak activation RAM `< 170 MB`.

---

## 3. Stage-2 Decoder & Autoregressive Decoding

**Root bottleneck:** `_layer_step_comer` runs `Conv2d(8→32, 5×5)` at **every token step × 4 layers** (`~472M` MACs/step = **~9× the rest of the decoder**, adding `+4–8 s` on mobile WASM). Also `beam=3` currently duplicates `mem_kv` (`151 MB` FP32).

| Idea | Size Δ | Speedup | KV-Cache RAM Δ | Est. F1 Δ |
| :--- | :---: | :---: | :---: | :---: |
| **1. Export pruning + shared `mem_kv`** (drop training-only `S18`, `aux`, `coord`; don't duplicate `mem_kv` across beams) | `-0.65 MB` FP32 (`-0.17 MB` INT8) | `1.05×` | **`3×` smaller** (`151→50 MB`) | **`0.0%`** (exact) |
| **2. `3L d=192` + GQA (`8Q/2KV`) + 1D Horizontal DW-CoMER** (pool `H_m` → `dw-Conv1d(k=5) + 1×1`, broadcast vertically: `>99%` fewer CoMER MACs) | `-10.4 MB` FP32 (`4.5M→1.9M`, `-2.6 MB` INT8) | **`4.0–6.0×`** (`~3×` from CoMER) | **`6.2×` smaller** (`18.9→3.0 MB`) | `-0.3 to -0.8%` (with KD) |
| **3. 3-Head Multi-Token Prediction (MTP) for `bekern`** (draft `(dur, step, oct, alter)` in 1 pass; verify 4 tokens in one batched grammar-masked step) | `+0.5 MB` FP32 (`+0.12 MB` INT8) | **`2.0–2.6×`** fewer WASM steps | `0%` | **`0.0%`** (exact verify) |
| **4. Adaptive greedy + selective bar-level `beam=3`** (default `greedy_kv` `B=1`; snapshot KV at `BAR`; re-decode only low-conf ~15–20% bars with `beam=3`) | `0 MB` | **`2.2–2.6×`** | **`3×` smaller** typical | `-0.1 to -0.3%` |

- **Combined decoder (INT8 WASM):** `~2.0M` params (`~2 MB`), **`~20–35×` faster**, `< 15 MB` total KV RAM, `< 1%` F1 delta.

---

## 4. Stage-1 `StaffUNet2D-Nano`

**Root bottleneck:** At `1536×1152` (1.77 MP), `dec0` concat (`[1, 60, 768, 576]`) alone peaks at **`220–400 MB` RAM** and takes **`~1.8–2.4 s`** in WASM (`26.4 GMACs`).

| Idea | Params | INT8 Size | Peak RAM | WASM Latency | Est. F1 Δ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (`base=40`, `row_ctx`, `1.77 MP`)** | `1.54M` | `1.55 MB` | `~220–400 MB` | `~1800–2400 ms` | `0.0%` (ref) |
| **1. DW-Sep / MBConv (`expand=2`, `base=24–28`)** (keep 3×3 `stem1`, dilated DW `bottleneck`, DW `row_ctx`; ~8× fewer MACs) | `0.30–0.35M` | `0.38 MB` | `~140 MB` | `~280–350 ms` | `-0.8 to -1.5%` (scratch) |
| **2. Adaptive Browser Pixel Budget (`0.8–1.0 MP`, e.g. `1024×768`)** *(zero retraining via `dynamic_res.py` + `log_hm`!; re-run at `1.77 MP` only if `median(sp_hm) < 3.5 px`)* | `1.54M` | `1.55 MB` | `~100–120 MB` | `~850–1100 ms` | `-0.3 to -0.7%` |
| **3A. `out_stride=4` + `dy_subpx` 5th channel** (drops `s2` skip & `dec0` completely) | `-11K` | `-0.01 MB` | **`~55 MB` (`-75%`)** | `-18–30%` | `-0.2%` (`-0.6%` on 6-staff) |
| **3B. Keep `stride=2` + `1×1` proj→8ch `add` skip + DW `3×3`** | `-5K` | `-0.01 MB` | **`~75 MB`** | `-15%` | `-0.1 to -0.2%` |
| **4. KD from `StaffUNet2D` v3 (`base=40`, `1.77 MP`) + QAT** | `0` overhead | `0` | `0` | `0` | **`+0.8 to +1.3%`** recovery |
| **Combined `StaffUNet2D-Nano` (1+2+3A/B+4 @ `0.9 MP`)** | **`~0.30M`** | **`~0.38 MB`** | **`~25–40 MB`** | **`~80–120 ms`** | **`-0.3 to -0.6%`** |

---

## 5. Quantization & Knowledge Distillation Recipes

| Recipe (Full S1 + S2) | Brotli Size | Peak RAM | WASM / WebGPU Speedup | Est. F1 Δ |
| :--- | :---: | :---: | :---: | :---: |
| FP32 Baseline (`33.9M`) | `~118 MB` | `~480 MB` | `1.0× / 1.0×` | `0.0%` |
| FP16 | `~58 MB` | `~290 MB` | `1.0× / 1.6×` | `-0.05%` |
| INT8 Dynamic PTQ (Linear only) | `~29 MB` | `~230 MB` | `1.4× / 1.2×` | `-0.5%` |
| **INT8 Static QDQ QAT** *(winner on older WASM-SIMD)* | `~26 MB` | `~190 MB` | **`2.3× / 1.8×`** | **`-0.3%`** |
| INT4 `MatMulNBits` (W4A16/W4A8) | `~16 MB` | `~175 MB` | `1.3× / 2.0×` | `-1.2%` |
| Structured 35% Channel Prune (S3–S4) + INT8 QAT | `~18 MB` | `~155 MB` | `2.9× / 2.3×` | `-0.9%` |
| **KD Student (`~7M`) + INT8 QAT** *(overall winner)* | **`~8 MB`** | **`~95 MB`** | **`4.5× / 3.5×`** | **`-1.2%`** |

- **Quantization-sensitive OMR layers (keep FP16/FP32):** `stem`, `foveal_tap` (1–2 px stems/dots), Stage-1 `head` / `log_hm` (peak drift), decoder `out` logits (rare accidentals), `LayerNorm`, and GDCM attention.
- **KD protocol (`32.5M exp2B → 6–8M` student):** Sequence-level KD + token KL (`T=2`) + encoder `mem` cosine + `S18`/`GDCM` auxiliary distillation recovers **65–80%** of the small-model F1 gap.

---

## 6. Browser Runtime & Deployment Architecture

1. **Inference runtime:** **`onnxruntime-web`** with **WebGPU FP16** primary and **WASM-SIMD INT8** automatic fallback (skip WebNN — no iOS Safari; skip `tract` — slower INT8 SIMD).
2. **Progressive caching:** Store Brotli-compressed ONNX models in Cache API / IndexedDB. Load Stage 1 (`~0.4–1.2 MB`) immediately; prefetch Stage 2 (`~6 MB`) in a Web Worker while the user frames the sheet-music photo.
3. **Sequential session lifecycle (critical for 3–4 GB phones / iOS Safari):** Call `session.release()` on Stage 1 *before* initializing Stage 2 so peak activation RAM never stacks.
4. **Zero-ONNX client glue:**
   - Implement `BezierRibbonSampler` as a ~30-line pure JS or Rust-WASM bilinear sampler directly over the RGBA canvas buffer (`< 2 ms`, no full-page float tensor copy).
   - Compile **`converter/assembly-serialize`** to Rust-WASM (gating `clap`, keeping pure `serde`/`zip`) → **`~300 KB` Brotli**, generating valid `.mscz` files 100% client-side in `< 10 ms`.
