# `2026-11-26T00-54-ablation-vir-vico-router` — ViRTwoTierRouter + ViCO vs InkTokenRouterV2 (1-Variable Ablation)

- **Status:** `RUNNING` (launched `2026-11-26 00:59 EET` / `2026-11-25T23:00:20Z`, Modal `app_id`: `ap-s2HoXUXOCRrpAI3CMHFQJF`)
- **Baseline:** Commit `9d73170` | Architecture: `UniOMRv2Net` (`small_45m`, ~43M params) with standard binary `InkTokenRouterV2` (`n_bg_summary=4`)
- **Platform:** Modal `2× L4` (`24 GB`, native BF16, parallel arms) — profile `gregoreesmaa1`

## Summary & Hypothesis
Isolate a **single architectural variable** on `UniOMR v2 small_45m` (`1,200` steps each on our stratified 2,000-page proxy shard `datasets/iter10_5_proxy_2k.tar`, identical `seed=42`, identical `Muon + AdamW` WSD schedule, identical augmentations & bit-identical batch ordering):
- **Arm 1 (`baseline`):** Standard binary keep/drop `InkTokenRouterV2` (`keep_ratio=0.40`, `n_bg_summary=4`, `max_mem_tokens=3072`).
- **Arm 2 (`vir_vico`):** `ViRTwoTierRouter` (`tmp-train/mvd/vir_router_v2.py`, Tier-1 fine `/16` static token buckets + Tier-2 coarse `/32` VisionZip contextual merging `k_coarse=96`, pure float32 ONNX/WebGPU native) + training-only InternVL 3.5 ViCO KL consistency loss (`vico_weight=0.10`, `vico_temp=1.5`, every 4th step vs dense `keep_ratio=0.85` teacher).

**Hypothesis:** Replacing binary keep/drop routing with 2-tier fine + coarse VisionZip contextual merging + ViCO consistency loss will preserve faint handwritten/printed staff lines, slurs, and margin context at zero extra WebGPU graph-capture cost, reducing `proxy_val.jsonl` CE/MTP loss and SER while improving Token F1 and `sys_count_exact` compared to `InkTokenRouterV2`.

## Quick Links
- Launcher script: [run_modal.py](./run_modal.py)
- Plan, Live Query IDs & Decision Tree: [PLAN.md](./PLAN.md)
- Outcome: `OUTCOME.md` *(to be written immediately after both L4 arms complete and HF eval JSON lands)*
- Hugging Face Eval Summary: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T00-54-ablation-vir-vico-router.json`
- Hugging Face Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T00-54-ablation-vir-vico-router.json`
- Hugging Face Checkpoints: `gregoreesmaa/sheet-music-converter-omr` → `checkpoints/2026-11-26T00-54-ablation-vir-vico-router/{baseline,vir_vico}/uni_omr_v2_ema_best.pt`
