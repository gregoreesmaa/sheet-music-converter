# Experiment Plan — `2026-11-26T07-40-i13-17-paddleocr-vrfm-band-router` (`i13-17`)

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe`.
- **Rationale:** Every arm uses fast MTP-4 speculative FSA decoding (`~380–420 tok/s` on L4 BF16) with zero extra decoder steps. Running 7 arms across 14 HW manuscripts (28 pages) + 10 printed pieces (20 pages) finishes in `~18–24 min` (`~1.5–2.0 CU`).

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-26T07-40-i13-17-paddleocr-vrfm-band-router`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T07-40-i13-17-paddleocr-vrfm-band-router.json`
  - Run Ledger JSON: `runs/2026-11-26T07-40-i13-17-paddleocr-vrfm-band-router.json`

## 3. Single-Variable Sweep Grid (`max_dim=1280`, `max_len=3840`, MTP-4 + FSA)
| Arm Key | `border_frac` | `band_alpha` | `keep_ratio_override` | Purpose |
|---|---:|---:|---:|---|
| `baseline_global_topk` | `0.00` | `0.00` | `None` (0.45) | Unmodified `InkTokenRouter` control |
| `vrfm_border_3pct` | `0.03` | `0.00` | `None` | Pure outer-3% desk/colorbar border suppression |
| `vrfm_border_6pct` | `0.06` | `0.00` | `None` | Stronger outer-6% archival margin suppression |
| `vrfm_band_0.30` | `0.00` | `0.30` | `None` | Pure horizontal staff-band smoothing (`3-row` kernel, `α=0.30`) |
| `vrfm_band_0.60` | `0.00` | `0.60` | `None` | Stronger horizontal staff-band cohesion (`α=0.60`) |
| `vrfm_b3_band0.30` | `0.03` | `0.30` | `None` | Combined VRFM (border suppression + staff-band cohesion) |
| `vrfm_b3_band0.30_k55` | `0.03` | `0.30` | `0.55` | Combined VRFM with expanded `0.55` keep budget |

- **Zero-Edit Implementation:** Wraps `model.router.saliency_head` (or patches `router.forward` cleanly with context-managed restoration) inside `run_colab.py` to apply the 2D VRFM border mask + horizontal 1D band prior strictly to the saliency scores used for top-$k$ selection, while keeping `uni_omr.py` / `uni_omr_v2.py` completely untouched. Also logs per-page `k_kept / n_in` and `ink_recall`.

## 4. Checkpointing & Resume Plan
- Atomic JSON checkpointing (`_atomic_write_json`) to `/content/cache/evals/2026-11-26T07-40-i13-17-paddleocr-vrfm-band-router_partial.json` every `<= 5.0` min and after every piece.

## 5. Evaluation Gate & Decision Tree
- **PROMOTE:** Any arm achieves `delta_note_f1_py >= +0.010` on `hw_photo` OR `hw_clean` (or `>= +0.005` on `printed_photo`) with `wins > losses` and printed clean regression `>= -0.003` → promote VRFM border+band prior into `InkTokenRouterV2` for Iter 13 training & inference.
- **ITERATE:** If `vrfm_band_0.30` lifts `hw_photo` but hurts tilted/rotated pages (`rotation_deg > 5°`) → replace axis-aligned row smoothing with a lightweight learned `1×K` dilated depthwise conv (`VRFM-Lite`) trained end-to-end in Iter 13.
- **DISCARD:** If all VRFM arms are neutral (`< +0.003`) or regress printed `> -0.005` → `InkTokenRouter`'s learned saliency already handles border clutter and horizontal staff bands sufficiently.
