# Iter 13 (`i13-12`) — PRAIG SMT++ Progressive System Unfolding & Audiveris Non-Music Text Rejection on `UniOMR v2`

- **Experiment ID:** `2026-11-26T07-40-i13-12-smt-unfolding-text-reject-v2`
- **Status:** `QUEUED`
- **Source Inspiration:** Open-source OMR competitors **PRAIG Sheet Music Transformer++ (`SMT++`, Ríos-Vila et al.)** (progressive system-count unfolding curriculum) and **Audiveris (`TEXTS` rejection)** (isolating & ignoring non-music titles, headers, lyrics, page numbers)
- **Platform / GPU:** Google Colab `L4` (`24 GB`, BF16 native, ~36 min, ~2.9 CU)
- **HF Eval Target:** `evals/2026-11-26T07-40-i13-12-smt-unfolding-text-reject-v2.json`
- **Core Idea:** Controlled 3-arm single-variable training ablation on `UniOMR v2 small_45m` (`1,200` steps/arm on `datasets/iter10_5_proxy_2k.tar`, using our winning decoupled `--muon-lr 1.0e-3 --lr 3.0e-4` + un-lagged EMA `0.99 / 1` from `exp4b` and `1280px` from `ov-35`):
  - **Arm A (`control_static_v2`):** Standard static full-page sampling on `proxy_train`, no synthetic text clutter.
  - **Arm B (`smt_progressive_unfolding`):** Single-variable **SMT++ progressive system-count unfolding** (`<= 3` systems for steps `1..400`, `<= 6` systems for steps `401..800`, full `<= 16` systems for steps `801..1200`), zero code changes to `train_uni_v2.py` (chained via `--init-from`).
  - **Arm C (`audiveris_text_clutter_reject`):** Single-variable **Audiveris-style non-music text/lyric rejection training** — overlays synthetic title headers, composer credits, page numbers, tempo markings, and inter-staff lyric strips onto 50% of training pages with unchanged GT `bekern`, teaching `InkTokenRouterV2` and the GQA decoder to ignore non-music text clutter.
- **Evaluation:** All 3 arms (both RAW and EMA checkpoints) evaluated on **both** standard `proxy_val` AND a deterministic text-cluttered `proxy_val_text_clutter` slice (`val_ce_loss`, `val_ser`, `val_token_f1`, `val_sys_count_exact`).
