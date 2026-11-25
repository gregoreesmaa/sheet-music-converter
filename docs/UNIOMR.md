# UniOMR — Unified Full-Page Optical Music Recognition

**Code:** [`tmp-train/mvd/uni_omr.py`](../tmp-train/mvd/uni_omr.py) · **Evaluator:** [`tmp-train/mvd/eval_uni_suite31.py`](../tmp-train/mvd/eval_uni_suite31.py) · **FSA Grammar:** [`tmp-train/mvd/grammar.py`](../tmp-train/mvd/grammar.py)

`UniOMR` is a single-stage, dynamic-resolution encoder–decoder OMR model that reads single-system strips and full pages natively — **with zero Stage-1 staff detector or system cropper** — and outputs `bekern` tokens separated by `<sys>` (`vocab_size = 182`). Its **`43.5M`-parameter core** quantizes to **`21.7 MB` INT4 / `43.5 MB` INT8**, fitting well inside our `< 55 MB` client-side browser WASM/WebGPU target.

---

## 1. Architecture (`43.5M` Core)

```text
Input RGB (B, 3, H, W)  [arbitrary aspect ratio, e.g. 256×1024 strip or 1280×960 full page]
   │
   ▼
DynamicConvBackbone (stride '4x8' anisotropic, GroupNorm + SiLU, stages 64→128→256→512 → d=384)
   │  └─ Early-layer MixStyle2d (p=0.3 on stages 1–2; 0 extra params, 0% inference cost)
   ▼
InkTokenRouter (local 31×31 high-pass ink prior + 2-layer conv saliency head, BCE supervised)
   │  ├─ Dynamic top-k selection in row-major order (min_keep=0.12 .. max_keep=0.25 in Iter 6)
   │  └─ Pools discarded background into n_bg_summary=4 learned summary tokens
   ▼
SparseTransformerEncoder (4 layers, d_model=384, n_heads=6, SwiGLU, RMSNorm)
   │  ├─ RoPE2D on native (y, x) coordinates
   │  └─ Normalized [0, 1] 2D sinusoidal pos2d (projected via zero-init pos2d_proj, added pre & post)
   │  └─ [Training-only] UVFlowHead (u, v, valid) dewarping aux head (0% inference cost)
   ▼
UniTransformerDecoder (6 layers, d_model=384, n_heads=6, RoPE1D + cross-attn, SwiGLU, RMSNorm)
   │  ├─ Shared lm_head over bekern vocab + <sys> (182 tokens)
   │  └─ MTP-4 heads (3 lightweight residual MTPHead projections for offsets +2, +3, +4)
   ▼
bekern + <sys> sequence  ──▶  assembly-serialize  ──▶  MusicXML 3.1 / MuseScore 4 (.mscz)
```

| Footprint | Precision | Size |
| :--- | :---: | :---: |
| Core inference weights (`backbone + router + encoder + decoder + MTP-4`) | FP32 | `~174 MB` |
| Quantized (browser / desktop) | **INT8** | **`43.5 MB`** |
| Quantized (old-device browser WASM/WebGPU) | **INT4** | **`21.7 MB`** |

---

## 2. Training & Decoding Stack

- **CASS (Curriculum/Confidence-Aware Scheduled Sampling) + Token Dropout (`tok_dropout=0.15`):** Combats teacher-forcing exposure bias on long multi-system sequences while forcing visual grounding over language-model hallucination.
- **SWAD / Checkpoint Souping (`k=5`):** Averages checkpoints (`step 4000..12000`) for flatter minima and OOD stability.
- **Formal `PrefixState` FSA Grammar (`--use-grammar`):** Constrains decoding to structurally valid `bekern` sequences with clean state resets on `<sys>` (non-strict FSA lifts `DecOK` to `100%` on printed pages without over-rejecting valid rhythms).
- **MTP-4 Self-Speculative Decoding (`--use-mtp-spec`):** Drafts up to 4 tokens per forward pass using `mtp_heads` with exact greedy/FSA parity, yielding a **`~2.3–2.9×` speedup** (`~155 tok/s` standard greedy → **`447 tok/s`** with `88.6–89.3%` MTP acceptance on Iter 6 printed pages).
- **SDPA Kernel Selection (`CUTLASS EFFICIENT_ATTENTION` vs cuDNN trap):** Under dynamic routing + multi-scale full pages + variable sequence lengths, cuDNN 9.10 SDPA recompiles a new CPU execution plan (`~640 ms/shape`) on every `(mem, T)` shape change — causing a **`49×` slowdown**. Explicitly prioritizing `SDPBackend.EFFICIENT_ATTENTION` (CUTLASS) with `MATH` fallback eliminates the stall completely.

---

## 3. Measured Benchmarks

### 3.1 In-Distribution Validation Progression (Iter 1 → Iter 6)

Measured on held-out in-distribution validation sets (`eval_summary.json`, `n=100` each):

| Iter | Run | Key Change | Best Step | Printed Strip Tok F1 (SER) | HW Strip Tok F1 | Full-Page Tok F1 (SER) | Mean Val SER |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | `uni-omr-iter1-strip` | Baseline strip model (`stride=16`) | 6,000 | 0.4023 (`0.9485`) | 0.3200 | — | 1.1510 |
| **2** | `uni-omr-iter2-s8` | Higher-res `stride=8` backbone | 1,000 | 0.4317 (`0.9217`) | 0.3552 | — | 1.0349 |
| **3** | `uni-omr-iter3-pos2d` | Explicit 2D sinusoidal `pos2d` into memory | 10,000 | 0.5870 (`0.6998`) | 0.3586 | — | 0.9871 |
| **4** | `uni-omr-iter4-mixed` | Mixed strips + full pages (`<sys>`) | 12,000 | 0.8230 (`0.3514`) | 0.5234 | 0.4837 (`0.7607`) | 0.7717 |
| **5** | `uni-omr-iter5-fp1024` | `1024px` full pages + `tok_dropout=0.15` | 12,000 | 0.9235 (`0.1495`) | 0.6685 | 0.8275 (`0.3899`) | 0.5011 |
| **6** | `uni-omr-iter6-clean` | `4×8` stride, high-pass BCE router, `MixStyle`, normalized `pos2d`, `1280px` | 12,000 | **0.9595** (`0.0766`) | **0.7772** | **0.8375** (`0.3625`) | **0.3872** |

### 3.2 Unseen OOD `suite31_hw` Scoreboard (45 Pieces × Clean/Photo = 90 Pages)

| Model / Decode Mode | Printed Clean Note F1 | Printed Photo Note F1 | HW Clean Note F1 | HW Photo Note F1 | Printed DecOK | Speed (Photo) | MTP Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Best 2-Stage (`Arm C` + `exp2B`)** | **0.7482** | 0.6949 | **0.0568** | **0.0387** | 100% | ~0.16 s/p | — |
| `UniOMR` Iter 5 (greedy, `1024px`) | 0.2200 | 0.0335 | 0.0143 | 0.0056 | 77.4% | 153 tok/s | — |
| `UniOMR` Iter 5 (`FSA + MTP-4`, `1024px`) | 0.2863 | 0.0407 | 0.0255 | 0.0077 | 100% / 90.3% | 363 tok/s | 78.1% |
| `UniOMR` Iter 6 (`strict FSA + MTP-4`, `1280px`) | 0.6794 | 0.6719 | 0.0081 | 0.0055 | **100%** | 420 tok/s | 82.8% |
| **`UniOMR` Iter 6 (`FSA + MTP-4`, `1280px`)** | **0.7014** | **0.7213** | 0.0101 | 0.0023 | **100%** | **447 tok/s** | **88.6%** |

### 3.3 Iter 6 Standout Printed Pieces (`suite31_hw`, Strict Note F1)

| Tier | Pieces (Strict Note F1) |
| :--- | :--- |
| **Perfect (`1.000`)** | `repeats-melody`, `mozart-k458-1`, `airds-air`, `foster-leadsheet` |
| **Near-Perfect (`0.92 – 0.97`)** | `bach-bwv66.6`, `schumann` (Lieder/piano), `haydn`, `joplin`, `chopin` |

---

## 4. Key Forensic Lessons (Iter 5 & Iter 6)

Following the [`learn-from-suite31`](../.agents/skills/learn-from-suite31/SKILL.md) eyeball + CPU-ablation protocol uncovered five concrete root causes:

1. **Router Inversion & `max_keep_ratio=0.25` Cap:**
   - In Iter 5, raw `1 - luma` inverted on dark phone borders and shadows (stealing routed tokens away from interior ink). Replacing it with a `31×31` local high-pass ink map + BCE saliency supervision fixed border theft in Iter 6.
   - However, Iter 6's tight `max_keep_ratio=0.25` cap starves dense multi-staff pages (4–7 staves, dense piano, crowded historical HW), dropping interior ink tokens on clean high-density prints.
2. **`128×` `pos2d` Frequency Aliasing:**
   - Scaling normalized `[0, 1]` coordinates by `128.0` in `_sinusoidal_pos2d` aliases high-frequency sinusoidal channels across dense pages and height/aspect shifts, degrading vertical pitch disambiguation on tall/multi-staff systems.
3. **Clean-vs-Photo Paradox (`0.7213` Photo > `0.7014` Clean) & Dark-Border Augmentation:**
   - Iter 6 scores *higher* on real phone photos than clean white renders (`0.7213` vs `0.7014`) because training leaned heavily on degraded/bordered framing; dark-border and clean-margin augmentation must be balanced so clean white pages and bordered photos sit in the same visual distribution.
4. **`grammar.py` `CHORD` Loop Bug:**
   - Under OOD historical handwriting, a loophole in `PrefixState` allowed infinite `CHORD` repetition loops (`sys=1`, hitting `max_len=3072`, causing `bekern-decode-rc-1` and `35.7%` HW `DecOK`). Tightening `PrefixState` chord-loop bounds eliminates runaway hangs.
5. **`80.8%` `hw-sim-v03` Barline Corruption & Real-HW Starvation:**
   - Forensic inspection of synthetic handwritten training pages (`hw-sim-v03`) revealed **`80.8%` barline corruption** (broken/misaligned barlines vs ground-truth `bekern`), teaching the model to hallucinate barlines and collapse on real historical manuscripts (`0.0101` HW Clean / `0.0023` HW Photo despite `0.7772` synthetic HW strip val Tok F1).

---

## 5. Roadmap: Iter 7 & Future `~20M` Distilled Browser Student

### Iter 7 (Next Immediate Run)
1. **Fix `grammar.py` `CHORD` loop bug** + unit tests in `test_uni_decode.py` so OOD HW never loops to `max_len`.
2. **Raise router capacity (`max_keep_ratio` `0.25 → 0.50+`)** and remove `128×` `pos2d` aliasing so dense 4–7 staff pages and crowded manuscripts retain `>= 98%` interior ink recall.
3. **Clean-margin + dark-border parity augmentation** to close the clean-vs-photo inversion and lift Printed Clean past `0.85+` alongside Printed Photo.
4. **Replace corrupted `hw-sim-v03` barlines** with clean barline-verified `hw-score-sim` renders + higher real historical HW sampling weight (strictly `0` overlap with `suite31_hw` and `gate20/test-8`).

### Future `~20M` Distilled Browser Student
Once `UniOMR` (`43.5M` teacher) clears `>= 0.90` Strict Note F1 across both printed and handwritten `suite31_hw`:
- **Knowledge distillation** (sequence-level + logit KD + MTP-4 heads) into a **`~20M`-parameter student** (`d_model=256`, `4` enc / `4` dec layers).
- **Target browser footprint:** **`~10 MB` INT4 / `~20 MB` INT8** in ONNX / WebGPU / WASM — instant cold-start download and real-time full-page OMR on older laptops and mobile browsers.
