# OUTCOME — `2026-11-26T22-18-i14-03-ocr-mask-declutter-pointer-fusion`

- **Status:** `COMPLETED`
- **Verdict:** **`ITERATE`**
- **Started / Finished:** `2026-11-27T15:24:03Z` → `2026-11-27T16:15:05Z`
- **Elapsed Time:** `3,061.79s` (`51.0 min`)
- **Platform / GPU:** `NVIDIA A100-SXM4-40GB` (BF16, `600` steps/arm, `small_45m`, Muon `lr=3e-4`, WSD, `gpu_ema_decay=0.99`)
- **HF Artifacts:**
  - Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T22-18-i14-03-ocr-mask-declutter-pointer-fusion.json`
  - Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T22-18-i14-03-ocr-mask-declutter-pointer-fusion.json`

---

## Measured Results

### OCR De-Cluttering Coverage (`RapidOCR`, Staff-Safe Filter)
| Metric | Value |
|---|---|
| Total proxy pages (`train + val`) | `2,064` |
| Pages with >=1 staff-safe box erased | `469` (`22.7%` of total; capped by `MAX_OCR_SCAN_PAGES=600` on train) |
| Total OCR boxes detected | `11,699` |
| Staff-safe boxes erased | `7,262` (`62.1%`) |
| Mean erased pixel area (on inpainted pages) | `0.807%` |

### 3-Arm Controlled Proxy (`600` Steps/Arm, `iter10_5_proxy_2k`)
| Arm | Pointer Tokens Wired? | Elapsed (s) | Peak VRAM (MB) | Final 25-Step Loss | RAW `val_token_f1` | Δ RAW F1 | RAW `val_ser` | EMA `val_token_f1` | Δ EMA F1 | EMA `val_ser` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `control_raw_pixels` | No | `711.1` | `8,964.7` | `2.5376` | `0.2296` | `—` | `0.9079` | **`0.2275`** | `—` | **`0.9123`** |
| `ocr_inpaint_declutter_only` | No | `729.3` | `8,964.4` | **`2.5148`** | `0.2782` | **`+0.0486`** | `0.8755` | `0.1425` | `-0.0850` | `0.9788` |
| `ocr_declutter_plus_pointer_fusion` | **No (see Scrutiny)** | `714.3` | `8,963.5` | `2.5343` | **`0.3036`** | **`+0.0740`** | **`0.8527`** | `0.1810` | `-0.0465` | `0.9443` |

- **Synthetic Pointer Probe (`_probe_pointer_attachment_head`, Arm 3 post-hoc only):** `299,136` params (`0.571 MB` FP16), `pointer_attach_top1_acc = 0.9102`.

---

## Scrutiny & Anomaly Check

1. **CRITICAL BUG — Pointer token fusion was NEVER wired into Arm 3 (`run_colab.py:289-399`):**
   - `inject_pointer_tokens=True` was accepted by `_run_arm()` but **never passed** to `train_uni_v2.run_train_v2(args)` or `UniOMRv2Net`. Both Arm 2 and Arm 3 trained identical `40,514,784`-param models on identical `d_train`/`d_val` manifests.
   - `pointer_attach_top1_acc = 0.9102` came purely from a standalone 200-step synthetic 2D-point MLP toy benchmark (`_probe_pointer_attachment_head`), NOT real OCR boxes fused into `UniOMRv2` encoder memory.
   - Therefore, the `+0.0740` RAW F1 in Arm 3 vs `+0.0486` in Arm 2 is just run-to-run variance on identical decluttered training — both independently confirm a **real RAW-weight lift (`+4.9` to `+7.4 pp` `val_token_f1`, `-3.2` to `-5.5 pp` SER)** from staff-safe OCR de-cluttering.
2. **Train/Val OCR Coverage Asymmetry:**
   - `MAX_OCR_SCAN_PAGES = 600` scanned `100%` of `proxy_val` (`64/64`) but only `30%` of `proxy_train` (`600/2,000`), creating a train/val distribution mismatch (`70%` of training pages still had raw text).
3. **Sharp RAW vs EMA Divergence (`-8.50 pp` / `-4.65 pp` EMA drop):**
   - On both decluttered arms, `ema_val_token_f1` (`0.1425`, `0.1810`) trailed RAW `val_token_f1` (`0.2782`, `0.3036`) by `12–13.5 pp` and fell below `control_raw_pixels` EMA (`0.2275`). At `600` steps with `gpu_ema_decay = 0.99` (effective window ~100 steps) + train/val mismatch, EMA lagged heavily behind the active weights.
4. **No OOD `suite31_hw` / OMR-NorthStar Eval:** Only `proxy_val` (`max_len=512`) was scored; no check yet for accidental erasure of dynamics (`f`, `p`, `mf`) or tempo markings on real engraved/handwritten scores.

---

## Verdict Rationale (`ITERATE`)

- **Why not `PROMOTE`:** Pointer fusion was never actually wired into `UniOMRv2Net`, EMA regressed `-4.65` to `-8.50 pp` vs control, only `30%` of training pages were OCR-inpainted, and no OOD `suite31_hw` check was run.
- **Why not `DISCARD`:** Across two independent training runs, staff-safe OCR de-cluttering erased only `0.807%` of page area yet lifted RAW `val_token_f1` by **`+4.86 pp` and `+7.40 pp`** (`0.2296 → 0.2782 / 0.3036`) and cut RAW SER by `3.2–5.5 pp` — strong evidence that freeing `InkTokenRouterV2` tokens from lyric/title clutter genuinely helps music recognition.

---

## Follow-Up Action / Successor Experiment

- **Next step (to be queued in `experiments/overnight_queue/queue.json` by coordinator):** Create a clean `i14-03b` follow-up that:
  1. Removes the `MAX_OCR_SCAN_PAGES=600` cap so `100%` of `proxy_train` and `proxy_val` are consistently OCR-decluttered (plus caches boxes for `suite31_hw`).
  2. Genuinely wires `TextBoxPointerEncoder` (`<= 64` normalized box tokens concatenated to `mem_final`) and `TextPointerAttachHead` inside `UniOMRv2Net` (instead of a post-hoc synthetic probe).
  3. Runs `1,000` steps with warmup-corrected EMA and evaluates **both RAW and EMA** on `proxy_val` AND `suite31_hw` (explicitly checking dynamics/articulation/lyric-heavy vocal pages).
