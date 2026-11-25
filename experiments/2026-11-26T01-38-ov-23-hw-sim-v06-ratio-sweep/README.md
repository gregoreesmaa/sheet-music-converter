# `2026-11-26T01-38-ov-23-hw-sim-v06-ratio-sweep` — `hw_sim_v06` Historical-Clean HW Mixing Ratio Sweep (`10%` vs `25%` vs `40%`)

- **Status:** `PLANNED` (Overnight Queue Batch 04, Wave 5 — `experiments/overnight_queue/batch_04.json`)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params, Anisotropic FPN + Hybrid Axial/Merger/2D-RoPE Encoder + DeepStack GQA MTP-4 Decoder)
- **Datasets:**
  - `gregoreesmaa/sheet-music-converter-omr` -> `datasets/iter10_5_proxy_2k.tar` (printed base pool + `64`-page `proxy_val.jsonl`)
  - `gregoreesmaa/sheet-music-converter-omr` -> `datasets/hw_sim_v06_500.tar` (new historical-clean HW-sim v0.6; ~460 train + 40 held-out val pages)
  - `gregoreesmaa/sheet-music-converter-omr` -> `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 31 printed + 14 real historical HW, `clean` + `photo` = 90 pages)

## Summary & Hypothesis
1-variable controlled data-composition ablation for Iter-11 `UniOMR v2` training. All three arms train `UniOMR v2 small_45m` for `800` steps from identical scratch initialization (`seed=42`) on a `2,000`-row full-page manifest (`B=2 GA=4`, `fullpage_max_dim=1280`, `max_seq_len=1536`, native `BF16`, `--optimizer muon`, WSD `lr=3e-4`, `stratified_fullpage=False` so exact manifest ratios are honored). **The ONLY variable changed is the mixing ratio of `hw_sim_v06` pages:**
- **Arm A (`hw_10`):** `90%` printed (`1,800` rows) + **`10%` `hw_sim_v06`** (`200` rows)
- **Arm B (`hw_25`):** `75%` printed (`1,500` rows) + **`25%` `hw_sim_v06`** (`500` rows)
- **Arm C (`hw_40`):** `60%` printed (`1,200` rows) + **`40%` `hw_sim_v06`** (`800` rows)

**Hypothesis:** `25%` `hw_sim_v06` (`hw_25`) hits the Pareto knee between printed val F1 retention and real-HW zero-shot transfer on `suite31_hw` (`hw_clean` / `hw_photo`), whereas `10%` under-exposes the encoder to historical ink/staff wobble and `40%` begins to dilute printed engraving precision.

## Quick Links
- Self-contained Colab L4 runner: [run_colab.py](./run_colab.py)
- Experiment Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Manifest: [batch_04.json](../overnight_queue/batch_04.json)
- Outcome: `OUTCOME.md` *(to be written immediately after HF eval JSON is retrieved)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T01-38-ov-23-hw-sim-v06-ratio-sweep.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T01-38-ov-23-hw-sim-v06-ratio-sweep.json`
