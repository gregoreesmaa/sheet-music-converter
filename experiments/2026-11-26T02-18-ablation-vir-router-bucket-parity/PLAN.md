# Experiment Plan — `2026-11-26T02-18-ablation-vir-router-bucket-parity`

## 1. Platform & Budget
- **Provider:** Google Colab (`1× NVIDIA L4`, `24 GB`, native BF16)
- **Estimated Wall-Clock & Cost:** ~28–34 min (~2.5 Compute Units) for 3 arms × 600 steps
- **Queue Registration:** `experiments/overnight_queue/queue.json` (`depends_on: ["2026-11-26T00-54-ablation-vir-vico-router"]`)

## 2. Controlled 3-Arm Design
All 3 arms share `UniOMRv2Net` (`small_45m`), `seed=42` (re-seeded pre-dataloader), `steps=600`, `B=2 × grad_accum=4` (`eff_batch=8`), `Muon+AdamW`, WSD `lr=3e-4` (warmup `60`, decay `25%`), `fullpage_max_dim=1280`, `max_seq_len=1536`, `max_mem_tokens=3072`, `ema_decay=0.99` (`update_every=5`).

| Arm | Router | Static Buckets | Train `keep_ratio` | Eval `dynamic_routing` | `vico_weight` |
|---|---|---|---|---|---:|
| `baseline_ink_router` | `InkTokenRouterV2` (`n_bg=4`) | N/A | `max_keep_ratio=0.75` | `0.15..0.75` | `0.00` |
| `vir_parity_no_vico` | `ViRTwoTierRouter` (`k_coarse=96`) | `(256, 512, 768, 1024, 1536, 2048, 2560, 2976)` | `0.75` (parity) | `0.15..0.75` | `0.00` |
| `vir_parity_vico005` | `ViRTwoTierRouter` (`k_coarse=96`) | `(256, 512, 768, 1024, 1536, 2048, 2560, 2976)` | `0.75` (parity) | `0.15..0.75` | `0.05` (`every=2`) |

## 3. Checkpointing & HF Artifacts
- 5-min rolling checkpoints to `/content/vol/ckpts/2026-11-26T02-18-ablation-vir-router-bucket-parity/<arm>/`
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T02-18-ablation-vir-router-bucket-parity.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T02-18-ablation-vir-router-bucket-parity.json`

## 4. Decision Tree (Final Go/No-Go for `ViRTwoTierRouter` in Iter 12)
- **PROMOTE (for Iter 12):** Either `vir_parity_no_vico` or `vir_parity_vico005` achieves `ink_recall >= 0.955` AND beats `baseline_ink_router` on BOTH `val_ce` and `token_f1` (by `>= +0.005`) on Raw and EMA → update `DEFAULT_STATIC_BUCKETS` in `vir_router_v2.py` to include `2560, 2976` and promote for Iter 12.
- **DISCARD:** Neither parity arm beats `baseline_ink_router` on `val_ce` + `token_f1` even after fixing the bucket ceiling and training budget → permanently discard `ViRTwoTierRouter` and keep `InkTokenRouterV2`.
