# `2026-11-26T01-38-ov-20-vir-vico-weight-schedule` — Overnight Exp 20: `ViR+ViCO` Loss Weight & Target Compression Ratio Sweep

- **Status:** `PLANNED` (`QUEUED` in `experiments/overnight_queue/batch_03.json`, Wave 5)
- **Platform:** Google Colab `1× NVIDIA L4` (`24 GB`, native `BF16`, ~32–38 min, ~2.7–3.2 CU)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params) + `ViRTwoTierRouter` (`k_coarse=96`, `static_buckets=True`, VisionZip contextual merging) + `compute_vico_consistency_loss` (`temp=1.5`, `symmetric=True`, `detach_dense=True`)
- **Dataset (100% permissive):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, zero `suite31_hw` overlap)

## Summary & Hypothesis
### Root Cause of `vico=0.0000` at Step 50 in Exp 3 (`2026-11-26T00-54-ablation-vir-vico-router`)
In Exp 3 (`run_modal.py:434,511`), `vico_every=4` restricted ViCO consistency updates to `step % 4 == 0` while step logging printed single-step `acc["vico_loss"]` at `step == 1 or step % 50 == 0`. Because `50 % 4 == 2 != 0` (and `150, 250, ...`), every odd-50 log line printed `vico=0.0000`! In addition, Exp 3 tested only a single fixed weight (`0.10`) and `keep_ratio=0.40` without measuring whether ViCO actually closed the dense-vs-routed KL gap or enabled tighter compression (`keep_ratio=0.25`).

### Controlled 4-Arm Sweep (`500` steps/arm, `vico_every=2`, identical `seed=42`, `lr=3e-4`, `Muon+AdamW`, `dim=1280`, BF16)
Every arm uses `vico_every=2` (so every `step % 50 == 0` fires cleanly), logs **windowed active-step ViCO mean**, `vico_steps_fired`, `vico_nonzero_frac`, `vico_step50_val` (explicitly verifying `> 0.0`), and `k_fine_routed` vs `k_fine_dense`:
1. **`vico_w002_kr035`**: `vico_loss_weight=0.02`, `keep_ratio=0.35` (~2.6×–2.9× target compression)
2. **`vico_w008_kr035`**: `vico_loss_weight=0.08`, `keep_ratio=0.35` (~2.6×–2.9× target compression)
3. **`vico_w020_kr035`**: `vico_loss_weight=0.20`, `keep_ratio=0.35` (~2.6×–2.9× target compression)
4. **`vico_w008_kr025`**: `vico_loss_weight=0.08`, `keep_ratio=0.25` (~3.5×–4.0× aggressive compression)

**Hypothesis:** Moderate ViCO weight (`0.08`) will minimize the dense-vs-routed KL gap and achieve the highest `proxy_val` Token F1 / lowest SER without over-regularizing the routed student (unlike `0.20`) or under-aligning it (unlike `0.02`), and will preserve `>= 95%` of `kr=0.35` Token F1 even when pushed to `keep_ratio=0.25`.

## Quick Links
- Router & Loss Implementation: [vir_router_v2.py](../../tmp-train/mvd/vir_router_v2.py)
- Launcher: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Entry: [batch_03.json](../overnight_queue/batch_03.json)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-20-vir-vico-weight-schedule.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-20-vir-vico-weight-schedule.json`
