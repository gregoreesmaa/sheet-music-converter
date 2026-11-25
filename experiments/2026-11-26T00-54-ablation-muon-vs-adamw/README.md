# `2026-11-26T00-54-ablation-muon-vs-adamw` — Native `Muon + AdamW` vs Pure Fused `AdamW` on `UniOMR v2 small_45m`

- **Status:** `RUNNING` (launched `2026-11-26 01:02`, Modal `L4`, `24 GB`, BF16 native — `ap-tXKsJnindxtL8tMh5RqEt9`)
- **Baseline Commit SHA:** `9d73170`
- **Model:** `UniOMR v2 small_45m` (`~43M` params, Anisotropic FPN + Hybrid Axial/Merger/2D-RoPE Encoder + DeepStack GQA MTP-4 Decoder)
- **Dataset:** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train pages + `64` held-out val pages, 100% permissive)

## Summary & Hypothesis
1-variable controlled optimizer ablation for Iter-11 `UniOMR v2` training. Both arms run `1,200` steps from identical random initialization (`seed=42`) on the exact same `iter10_5_proxy_2k` full-page dataset with identical `B=2 GA=4` (`eff=8`) and WSD learning rate schedule (`lr=3e-4`, `warmup=120`, `wsd_decay_frac=0.25`, `min_lr_ratio=0.05`). **The ONLY variable changed is `--optimizer muon` vs `--optimizer adamw`:**
- **Arm A (`muon`):** Native `torch.optim.Muon(adjust_lr_fn="match_rms_adamw", momentum=0.95, nesterov=True, ns_steps=5)` on all 2D Transformer matrices (`axial_layers`, `patch_merger`, `global_layers`, `deepstack_fusions`, `dec_layers`) + fused `AdamW` on 4D convs, embeddings, norms, biases, and aux heads.
- **Arm B (`adamw`):** Pure fused `torch.optim.AdamW` on all parameters.

**Hypothesis:** Routing 2D Transformer weight matrices through orthogonalized Newton-Schulz updates (`Muon` with `match_rms_adamw`) will converge faster (lower train CE/loss per 100 steps and lower held-out `proxy_val.jsonl` CE, MTP, and greedy SER) than pure fused `AdamW` at negligible wall-clock and VRAM overhead, validating `Muon + AdamW` as the default Iter-11 optimizer.

## Quick Links
- Launcher script (self-contained): [run_modal.py](./run_modal.py)
- Plan, Live Query IDs & Follow-Up Commands: [PLAN.md](./PLAN.md)
- Outcome: `OUTCOME.md` *(to be written immediately after both arms finish and HF eval JSON is retrieved)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T00-54-ablation-muon-vs-adamw.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T00-54-ablation-muon-vs-adamw.json`
- HF EMA Checkpoints: `gregoreesmaa/sheet-music-converter-omr` → `checkpoints/2026-11-26T00-54-ablation-muon-vs-adamw/{muon,adamw}_ema_latest.pt`
