# OUTCOME — `2026-11-27T10-30-i15-31-hrnet-music-dualres-s4` (`i15-31`)

- **Status:** `COMPLETED` (`2 / 2` arms completed, `error: null`)
- **Completed At:** `2026-11-27T10:44:39Z` (started `2026-11-27T10:00:13Z`)
- **Total Wall-Clock:** `2,666.25 s` (`44.44 min`)
- **Platform / GPU:** Local `NVIDIA GeForce RTX 4080 SUPER` (`16 GB` VRAM, native BF16; originally planned for Colab H100, executed locally via `D:\tmp\local-p0\i15-31`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T10-30-i15-31-hrnet-music-dualres-s4.json` (pulled locally to [eval_hf.json](./eval_hf.json); verified bit-identical to `D:\tmp\local-p0\i15-31\out\eval.json`)
  - Run Ledger: `runs/2026-11-27T10-30-i15-31-hrnet-music-dualres-s4.json` (pulled locally to [run_hf.json](./run_hf.json); verified bit-identical to `D:\tmp\local-p0\i15-31\out\run.json`)
- **Verdict:** **`ITERATE`**

---

## Measured Results (`ctrl` vs `hrnet_music_s4`, 600 steps/arm, `seed=42`, BF16, `muon+adamw`)

| Metric | `ctrl` (Stock `AnisotropicMusicBackbone`) | `hrnet_music_s4` (`/4 96ch \|\| /8 320ch`, 3 blk) | Delta (`hrnet - ctrl`) | `PLAN.md` Gate | Pass? |
|---|---:|---:|---:|---|---|
| **`ema_val_ce`** | `1.12744` | `1.12802` | **`+0.00058`** | `<= -0.01` OR iso-CE `<= +0.005` w/ `>= 1.5 MB` saved | **PASS** (iso-CE) |
| **`int4_dequant_val_ce`** | `1.12755` | `1.12763` | **`+0.00008`** | Dequant gap `<= 0.02` | **PASS** (gap `-0.00039`) |
| `raw_val_ce` | `1.10378` | `1.10523` | `+0.00145` | Diagnostic | Iso |
| `train_loss_tail50` | `1.53799` | `1.53778` | `-0.00021` | Diagnostic | Iso |
| **`packed_int4_mb`** (exact disk) | `53.039 MB` (`53,039,072 B`) | `51.099 MB` (`51,098,576 B`) | **`-1.940 MB`** (`-1,940,496 B`) | `< 55.0 MB` & save `>= 1.5 MB` | **PASS** |
| `fp16_keep_mb` (`backbone` + norms) | `12.855 MB` (`216` tensors) | `10.914 MB` (`261` tensors) | `-1.941 MB` (`+45` tensors) | FP16 backbone budget | **PASS** |
| `backbone_params` / `infer_params` | `5,178,140` / `77,866,208` | `4,207,892` / `76,895,960` | `-970,248` (`-18.7%` backbone) | Parameter savings | **PASS** |
| **`step_ms`** (steady-state) | `1,776.85 ms` | `2,366.05 ms` | **`+589.20 ms` (`+33.16%`)** | **`<= +15%` overhead** | **FAIL** |
| `train_elapsed_sec` | `1,066.83 s` (`17.78 min`) | `1,415.97 s` (`23.60 min`) | `+349.14 s` (`+32.73%`) | Throughput | **FAIL** |
| `peak_vram_mb` | `10,623.6 MB` (`10.37 GiB`) | `13,091.1 MB` (`12.78 GiB`) | `+2,467.5 MB` (`+23.23%`) | `< 14 GB` WDDM safety | **WARN** |

---

## Scrutiny & Anomaly Check

1. **Silent failures / partial coverage:** None. Both arms completed all `600 / 600` steps (`n_arms_completed == 2`, `error == null`), zero NaNs, exact Hybrid FP16/INT4 `.bin` files serialized and verified on disk (`passes_55mb_budget == true` for both arms), and INT4 tensor counts/sizes are bit-identical across arms (`131` INT4 tensors, `35.719 MB` weights + `4.465 MB` scales). HF artifacts match local `D:\tmp\local-p0\i15-31\out\{eval,run}.json` byte-for-byte.
2. **Apples-to-apples sanity:** Both arms ran sequentially on the identical local RTX 4080 SUPER with identical `seed=42`, `600` steps, `batch_size=2 × grad_accum=4`, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`, `muon+adamw`, WSD cosine schedule, `EMA_DECAY=0.99`, and `QATInt4ControllerV2` starting at step `420`.
   - *EMA vs raw lag:* In both arms, `ema_val_ce` is `~0.023` higher than `raw_val_ce` (`1.12744` vs `1.10378` for `ctrl`; `1.12802` vs `1.10523` for `hrnet_music_s4`). This is expected for `EMA_DECAY=0.99` over only 600 steps (EMA lags behind the rapid WSD decay tail over steps 450–600) and affects both arms identically.
3. **Why did `hrnet_music_s4` hit `-1.94 MB` at iso-CE (`+0.00058`) yet regress `step_ms` by `+33.16%` and VRAM by `+2.47 GB`?**
   - **4× spatial activation cost at `/4`:** Running 3 full `AnisotropicMusicBlock(ch_hi=96, expand_ratio=2.0)` blocks plus 3 bidirectional `PixelUnshuffle(2)` + `bilinear` cross-resolution exchanges at `/4` resolution operates on `4×` as many spatial pixels as `/8` (`76.8K` vs `19.2K` pixels on a `1280×960` page).
   - **Uncheckpointed backbone activations:** `grad_checkpoint=True` only checkpoints the transformer encoder/decoder layers, leaving all 3 `/4` branch blocks' intermediate activations in VRAM during backward (`+2,467.5 MB` peak VRAM, pushing RTX 4080 SUPER to `13.09 GB` — close to the `14 GB` Windows WDDM swap boundary).
   - **Windows cuDNN factor:** `run_colab.py` was authored for Linux Colab H100 and did not disable cuDNN on Windows, where extra `/4` variable-shape `Conv2d` ops in `HRNetMusicBackbone` can also amplify backward wall-clock time.
   - **No CE gain over stock:** Stock `AnisotropicMusicBackbone` already fuses `/4` features into `/8` via `p2_to_p3` + `p2_to_p4`, so 3 full bidirectional `/4 || /8` exchange blocks did not lower `ema_val_ce` beyond iso-CE (`+0.00058`, `int4_dequant_val_ce +0.00008`).

---

## Verdict: **`ITERATE`**

- **Why NOT `PROMOTE`:** Although `hrnet_music_s4` passes the iso-CE + size gate (`ema_val_ce +0.00058 <= +0.005`, `int4_dequant_val_ce +0.00008`, exact **`-1.940 MB`** FP16 backbone savings down to `51.099 MB`), it **fails the `step_ms <= +15%` throughput gate by more than 2× (`+33.16%`)** and adds **`+2.47 GB`** peak VRAM.
- **Why NOT `DISCARD`:** It does **not** trigger the `DISCARD` gate (`ema_val_ce` delta is `+0.00058`, far below `> +0.015`), and it proves that the FP16 backbone can be trimmed by **`~1.94 MB` (`-18.7%` backbone params)** with zero quality or INT4-quantization loss (`int4_dequant_val_ce` gap `-0.00039`).

---

## Follow-Up Action / Successor Experiment

- **Recommended Successor (to be queued by orchestrator):** Lightweight dual-res or slim-`/8` stage4 ablation (`i15-31b-hrnet-music-lite-s4`) comparing:
  1. **Lightweight HRNet-Music (`ch_hi=64`, `n_exchange=2` or single final `/4` fuse, `expand_ratio=1.5` on `/4`, optional backbone grad-checkpointing):** target `~ -2.0 MB` packed INT4 savings at iso-CE with `step_ms` overhead **`<= +10%`** and VRAM overhead `< +0.5 GB`.
  2. **Clean baseline control (`slim_s4_352ch`):** Simply narrowing stock serial `stage4` from `448ch -> 352ch` (keeping existing `p2_to_p4` lateral fuse) to test whether the `-1.94 MB` iso-CE win comes from dual-res exchange or purely from `stage4` channel slimming at **negative** `step_ms` and VRAM overhead.
