# Experiment Plan — `2026-11-26T07-40-i13-10-oemer-staff-dewarp-unitsize`

## 1. Competitor Mechanism & Hypothesis
- **OEMER / `muskew` (`dewarp.py`):** Instead of fragile 3D TPS warps, OEMER tracks horizontal staffline trajectories across columns and computes a smooth 1D vertical displacement field $y(x)$ to flatten curved book-spine and phone-photo staves.
- **Homr (`liebharc/homr`):** Computes the local inter-staffline spacing (`unit_size`) from staff anchors and normalizes every page/staff to a consistent vertical pitch scale before Transformer decoding.
- **Hypothesis:** On `suite31_hw` (especially `pr_photo`, `hw_clean`, and `hw_photo` where book-spine curvature and wide staff-density variation occur), staffline-aware vertical dewarping reduces pitch-line drift (`+0.005` to `+0.015` Note F1 on `photo`/`hw`), and `unit_size`-normalized scaling boosts dense multi-system pages without hurting sparse pages.

## 2. Platform & Identifiers
- **Platform:** Google Colab `L4` (`24 GB`, `cc=8.9`, BF16 native), est. **~28 min** (`~2.2 CU`).
- **HF Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`).
- **Eval Artifact:** `evals/2026-11-26T07-40-i13-10-oemer-staff-dewarp-unitsize.json`.
- **Run Ledger:** `runs/2026-11-26T07-40-i13-10-oemer-staff-dewarp-unitsize.json`.
- **Checkpoint:** Defaults to `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (overrideable via `HF_CKPT_PATH` env var to `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` once Iter 12 uploads).
- **Dataset:** `datasets/suite31_hw_eval.tar` (`45` pieces x `clean`/`photo` = `90` pages).

## 3. Single-Variable 2x2 Experimental Grid
All arms hold the checkpoint, BF16 autocast, `max_len=3840`, MTP-4 self-speculative decoding, and FSA grammar (`use_grammar=True, strict_grammar=False`) identical:
1. `baseline_1280`: `dewarp=False, unitsize_norm=False, max_dim=1280`.
2. `oemer_dewarp_only`: `dewarp=True, unitsize_norm=False, max_dim=1280` (horizontal-morphology ridge extraction + per-column phase/vertical correlation shift relative to page center + Gaussian smoothing + clamped `<= 0.03 * H` vertical remap with edge-replicate paper fill).
3. `homr_unitsize_only`: `dewarp=False, unitsize_norm=True` (vertical 1D autocorrelation of horizontal ink projection to find dominant staffline period `u` in `[4, 28]` px at 1280px reference, rescaling toward `target_unit_size=7.5` px with `max_dim` clamped to `[960, 1600]`).
4. `oemer_dewarp_plus_unitsize`: `dewarp=True, unitsize_norm=True`.

## 4. Checkpointing & Safety
- Atomic `_partial.json` (`.tmp` -> `os.replace`) saved every `<= 5.0` min and after each piece; auto-resumes completed `(piece, domain, arm)` cells.
- Reads HF token from `/tmp/.hf_token` (`chmod 600`), unlinks immediately and in `finally:`.
- `torch.backends.cuda.enable_cudnn_sdp(False)` enforced.

## 5. Promotion / Discard Gate
- **PROMOTE:** Any treatment arm achieves `>= +0.005` mean Note F1 on `all45_photo` OR `hw_photo`, with `<= -0.002` max regression on `pr_clean`, and `< 150 ms/page` CPU preprocessing overhead -> integrate into `converter/ingest-cleanup` and `eval_uni_suite31.py`.
- **ITERATE:** Positive gain on `hw_photo` / curved pages (`>= +0.005`) but regression on flat `pr_clean` -> add curvature-energy gate (apply dewarp only when measured peak-to-peak vertical droop `>= 4 px`).
- **DISCARD:** All treatment arms within `[-0.003, +0.003]` across slices -> close inference-time heuristic dewarping and rely on learned `Gauge-RoPE` (`academic-gauge-rope-converged-v2`).
