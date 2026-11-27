# Experiment: `2026-11-27T07-52-i15-08-bekern3-expressive-oracle-and-compress`

- **Title:** `bekern3` Architect 3 (Wave 13) — Expressive/Text 125-Page NorthStar Oracle Closure & Profile-A / Core-Pitch Compression + MTP-4 Speedup Bake-Off (`600`-Step 3-Arm Colab L4)
- **Status:** `QUEUED` (Wave 13, `colab_l4`, ~60 min / ~4.5 CU; staged via `queue_entry.json`)
- **Baseline Commit SHA:** `e426c28`
- **Baseline Architecture:** `UniOMRv2Net(small_45m)` (`~43M` params, GQA `8Q/2KV`, DeepStack, MTP-4 `mtp_k=4`)

## Summary & Hypotheses

`bekern_v2a` leaves two high-leverage bottlenecks open on `OMR-NorthStar` and edge/cloud decoding:
1. **Expressive & Text Oracle Ceiling (`S_expr`, `S_struct`):** Missing pickup-measure flags, deferred system lyrics, harmony/chord symbols, tempo/rehearsal markings, numbered overlapping slurs/hairpins, `15ma`/`15mb` ottavas, extended articulations (`+18` `art-*`), extended dynamics (`+7` `dyn-*`/`dyn-other`), and `bar-dashed` cap the MusicXML round-trip Oracle below `98.0` NS.
2. **Sequence Length & AR Decode Latency:** Verbose `voice-1`, split `rest <dur>`, split `<dur> dot`, repeated chord/sticky durations, and split `<step> <oct>` pitch pairs inflate full-page token counts and linearly slow down autoregressive decoding—even with MTP-4 speculative decoding.

**Architect 3's 3 Formalized Hypotheses:**
- **H1 (`part_a_oracle_closure`):** Extending `bekern3` with `<pickup>`, `<sys_lyr>`, `<harm>`, `<tempo>`, `<reh>`, numbered `slur-1..3` / `hairpin-1..2`, `ottava-15ma`/`ottava-15mb`, `+18` `art-*`, `+7` `dyn-*`/`dyn-other`, and `bar-dashed` closes the 125-page `OMR-NorthStar` round-trip Oracle to **`NS >= 98.0`** and **`S_expr >= 0.980`** (with `CFR == 0.0`).
- **H2 (`arm1_profile_a_comp`):** Lossless Profile-A compression (sticky duration + chord duration drop + implicit `voice-1` + fused `rest-<dur>` + fused `dur.`) cuts sequence length by **`~20.2%`** with `100%` exact round-trip and `>= -0.2 pp` Strict Note F1 parity.
- **H3 (`arm2_profile_a_plus_core_pitch`):** Adding `C2..B6` core pitch fusion (35 diatonic core tokens + compositional `<step> <oct>` fallback outside `C2..B6`) on top of Profile A cuts sequence length by **`~41.1%`** and achieves **`>= 1.45×`** wall-clock MTP-4 speculative decode speedup with **`>= -0.2 pp`** Strict Note F1 parity vs `arm0_bekern_v2a`.

## Quick Links

- Plan & launch spec: [PLAN.md](PLAN.md)
- Runner script: [run_colab.py](run_colab.py)
- Queue entry (staging): [queue_entry.json](queue_entry.json)
- Outcome (written upon completion): `OUTCOME.md`
- HF Eval JSON: `evals/2026-11-27T07-52-i15-08-bekern3-expressive-oracle-and-compress.json`
- HF Run Ledger: `runs/2026-11-27T07-52-i15-08-bekern3-expressive-oracle-and-compress.json`
