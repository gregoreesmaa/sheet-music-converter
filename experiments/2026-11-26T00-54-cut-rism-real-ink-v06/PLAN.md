# PLAN — `2026-11-26T00-54-cut-rism-real-ink-v06`

> **Zero-Prior-Context Handoff:** Everything required to monitor, inspect, cancel, resume, or consume outputs of this experiment is documented in this file and [run_modal.py](./run_modal.py).

---

## 1. Platform Chosen & Rationale

- **Platform:** Modal `L4` (`24 GB VRAM`, Ada Lovelace `cc=8.9`, **native BF16**).
- **Rationale:** `train_cut_rism.py` runs `512×512` random-crop CUT training (`batch_size=4`, ~11.4M-param generator + PatchGAN + 5-layer PatchNCE + Sobel-Y geometry lock) followed by full-page Hann-windowed tiled translation of 500 pages. Modal `L4` (~$0.80/hr, `$30/mo` free tier) supports native BF16 and keeps the entire train + translate + tar/upload pipeline well within `timeout=7200` (~45–65 min wall-clock, ~$0.60–$0.85 credit). Local RTX 4080 SUPER remains untouched for Iter 10.5.

---

## 2. Live Identifiers for Follow-Up Queries

| Field | Value |
|---|---|
| **Modal Profile** | `gregoreesmaa1` |
| **Modal App Name** | `exp1-cut-rism-real-ink-v06` |
| **Modal `app_id`** | `ap-6sVnb4KRrDaXVEHZUrYmQv` |
| **Modal Dashboard URL** | https://modal.com/apps/gregoreesmaa1/main/ap-6sVnb4KRrDaXVEHZUrYmQv |
| **Modal Volume** | `omr-cache` (mounted at `/vol`: `/vol/datasets/`, `/vol/ckpts/2026-11-26T00-54-cut-rism-real-ink-v06/`, `/vol/out/cut_rism_v06/`) |
| **Modal Secret** | `hf-secret` (`HF_TOKEN`) |
| **HF Dataset Repo** | `gregoreesmaa/sheet-music-converter-omr` |
| **HF Run Ledger** | `runs/2026-11-26T00-54-cut-rism-real-ink-v06.json` |
| **HF Checkpoints** | `checkpoints/2026-11-26T00-54-cut-rism-real-ink-v06/` (`cut_rism_best.pt`, `cut_rism_latest.pt`, `train_log.jsonl`, `previews/`) |
| **HF Output Shard** | `datasets/cut_rism_v06_500.tar` |

### Copy-Paste Monitor / Cancel / Resume Commands

```powershell
# Check live Modal app status
.\.venv\Scripts\modal.exe app list

# Tail live container logs
.\.venv\Scripts\modal.exe app logs ap-6sVnb4KRrDaXVEHZUrYmQv

# Stop / cancel runaway job
.\.venv\Scripts\modal.exe app stop ap-6sVnb4KRrDaXVEHZUrYmQv

# Resume cleanly (auto-loads /vol/ckpts/.../cut_rism_latest.pt or HF latest, skips translated PNGs)
.\.venv\Scripts\modal.exe run --detach experiments/2026-11-26T00-54-cut-rism-real-ink-v06/run_modal.py

# Re-run translation & tar upload only (skipping training)
.\.venv\Scripts\modal.exe run --detach experiments/2026-11-26T00-54-cut-rism-real-ink-v06/run_modal.py --skip-train
```

---

## 3. Required Training & Source Data

- **Source domain (`X`):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/hw_sim_v06_500.tar` (`500` synthetic historical-clean handwritten pages + `manifest.jsonl` + `stage2_strips/*.txt` `bekern+` GT). Permissive/synthetic, zero `CC-BY-NC`, zero `suite31_hw` eval overlap.
- **Target domain (`Y`):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/rism_hw_5707.tar` (`5,707` unlabeled real historical RISM manuscript pages). Unlabeled archival scans; zero `CC-BY-NC`.
- **Baseline commit SHA:** `9d73170` (uses `tmp-train/mvd/train_cut_rism.py` directly, no baseline checkpoint required — trains CUT generator from scratch with identity-residual init).

---

## 4. Exact Launch Command & Hyperparameters

```powershell
.\.venv\Scripts\modal.exe run --detach experiments/2026-11-26T00-54-cut-rism-real-ink-v06/run_modal.py
```

- **Training:** `steps=12000`, `batch_size=4`, `crop_size=512`, `lr=2e-4` (linear decay after step 6000), `ngf=64`, `ndf=64`, `n_blocks=9`, `gan_weight=1.0`, `nce_weight=1.0`, `geom_lock_weight=4.0`, `sobel_weight=1.0`, `bf16=True` (gated on `cc >= 8`), `seed=42`.
- **Checkpoints & Previews:** `save_every=500`, `preview_every=500`, `log_every=50`.
- **Translation:** Full-page `max_side=1536`, Hann-windowed overlap blend `tile_size=768`, `tile_overlap=96`, `bf16=True`, `--resume-translate` enabled.
- **Output Tar (`datasets/cut_rism_v06_500.tar`):** Portable relative paths (`manifest.jsonl`, `pages/*.png`, and `stage2_strips/*.txt` if present).

---

## 5. 5-Minute Checkpointing & Resume Plan

- `train_cut_rism.py` writes `cut_rism_latest.pt` and `cut_rism_best.pt` atomically (`.tmp` → `os.replace`) every `500` steps (~2–3 min on L4).
- A background daemon thread in [run_modal.py](./run_modal.py) calls `vol.commit()` every **5 minutes (`300s`)** so `/vol/ckpts/2026-11-26T00-54-cut-rism-real-ink-v06/` survives preemption or timeout.
- Every **15 minutes (`900s`)**, plus immediately on training completion and run exit, the launcher uploads `cut_rism_best.pt`, `cut_rism_latest.pt`, `train_log.jsonl`, latest preview PNG, and `runs/2026-11-26T00-54-cut-rism-real-ink-v06.json` to HF `gregoreesmaa/sheet-music-converter-omr`.
- On restart, `run_modal.py` automatically detects `/vol/ckpts/.../cut_rism_latest.pt` (or pulls it from HF) and resumes seamlessly, and skips already-translated PNGs during translation.

---

## 6. Evaluation Gate & Decision Tree

- **Geometry Lock Gate (Measured in `train_log.jsonl` & translation logs):**
  - Mean vertical ink drift `<= 1.5 px` across all 500 translated pages.
  - Sobel-Y correlation `sobel_y_corr >= 0.85` at convergence.
  - Visual eyeball check of `checkpoints/2026-11-26T00-54-cut-rism-real-ink-v06/previews/step_012000.png` and 5 sampled translated pages from `datasets/cut_rism_v06_500.tar`: authentic RISM parchment/iron-gall ink texture with zero hallucinated noteheads or erased staff lines.
- **Promote:** If geometry drift `<= 1.5 px` AND visual eyeball passes → promote `datasets/cut_rism_v06_500.tar` into the next UniOMR training manifest (Iter 11 / `v2`) and measure `suite31_hw` `hw_clean` / `hw_photo` Strict Note F1 delta.
- **Iterate:** If texture transfer is too weak (looks nearly identical to `hw_sim_v06`) → lower `geom_lock_weight` from `4.0` to `2.0`; if staff/notehead drift `> 1.5 px` or artifacts appear → raise `geom_lock_weight` to `6.0` or `sobel_weight` to `2.0`.
- **Discard:** If mode collapse or unreadable ink degradation persists after one weight iteration, discard CUT-RISM and rely on procedural `hw-score-sim` + real RISM self-training.
