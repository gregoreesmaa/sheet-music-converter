# OUTCOME — `2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe` (`i15-03`)

- **Status:** `COMPLETED` (`rc=0`, Colab session `2026-11-27t07-46-i15-03-` terminated cleanly with `exit_code=0`)
- **Verdict:** **`ITERATE` (CONFOUNDED — ZERO TEXT SUPERVISION + RANDOM-INIT BASE)**
- **Platform / GPU:** Google Colab `1× NVIDIA A100-SXM4-40GB` (`bf16=true`)
- **Run Window:** `2026-11-27T20:16:13Z` → `2026-11-28T01:43:48Z` (`elapsed_secs = 19,655.77 s` / `327.6 min` / `5.46 h`; queue wall-clock `20,100.98 s`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe.json` (local copy: `eval_hf.json`)
  - Run Ledger: `runs/2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe.json` (local copy: `run_hf.json`)

---

## Measured Results (`suite31_hw`, `n_vocal_pages = 64` / `90` total pages)

| Metric | Random-Init Base (`base_eval`) | Arm A: `arm_a_textspotter_head` (EMA / Raw) | Arm B: `arm_b_deferred_sys_lyr_bpe` (EMA / Raw) | Target / Hard Gate |
| :--- | :---: | :---: | :---: | :---: |
| **Held-out `text` F1** | `0.0000` | **`0.0000`** / `0.0000` | **`0.0000`** / `0.0000` | `>= 0.70` (FAIL — 0 train XML events) |
| **Vocal `S_note` F1** | `0.0000` | `0.0000` / `0.0000` (`+0.00 pp`) | `0.0042` / `0.0044` (`+0.42 pp`) | `>= -0.5 pp` (vacuous on `0.0` base) |
| **All-90 `S_note` F1** | `0.0000` | `0.0000` / `0.0000` | `0.0034` / `0.0035` | Regression check |
| **Train Loss (`tail50`)** | — | `0.00004` (collapsed to `<t_eos>`) | `1.30563` (music-only from scratch) | — |
| **Vocal Mean Latency** | `34.252 s` | `36.126 s` / `36.241 s` | `34.692 s` / `34.731 s` | Lower is better (inflated by random-init `max_len`) |
| **Trainable / Added Params** | `edge85m` (`265` vocab) | `2,303,744` head + `821,954` router | `+257` vocab (`265 → 522`) | — |
| **Exact Packed Hybrid INT4** | `52.415 MB` (`i14-05` ref) | **`55.081 MB`** (`55,080,928 B`, `+2.629 MB` head) | **`52.789 MB`** (`52,788,992 B`, `+0.374 MB` vocab) | **`< 55.0 MB`** (Arm A **FAIL**, Arm B **PASS**) |
| **INT4 Breakdown (`W / S / FP16`)** | — | `35.883 / 4.485 / 14.712 MB` | `35.261 / 4.408 / 13.120 MB` | — |
| **Arm Elapsed Time** | — | `8,063.21 s` (`1,565.8 s` train) | `8,430.09 s` (`2,141.9 s` train) | — |

---

## Scrutiny & Anomaly Check

1. **Fatal Data-Pipeline Confound — `0 / 2,000` Training Pages Had XML Text Events:**
   - Log proof (`D:\tmp\overnight-queue-logs\2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe.log`):
     `Loaded 2000 train pages (0 with XML text events) on NVIDIA A100-SXM4-40GB`
   - `datasets/iter10_5_proxy_2k.tar` ships only PNGs and `.bekern` files — **zero** `.musicxml` / `.xml` sidecars — so `extract_xml_text_events(...)` returned `[]` for 100% of training pages.
   - **Arm A (`arm_a_textspotter_head`)** therefore trained for 1,500 steps on degenerate 2-token `[TEXT_BOS_ID, TEXT_EOS_ID]` targets (`step 1: 5.7734 → step 150: 0.0021 → step 1200–1500: 0.0000`, `train_loss_tail50 = 4e-05`), memorizing immediate `<t_eos>` and scoring `heldout_text_f1 = 0.0000`.
   - **Arm B (`arm_b_deferred_sys_lyr_bpe`)** received zero `<sys_lyr>` text targets across all 1,500 steps (`train_loss_tail50 = 1.30563`, purely music-only training), also scoring `heldout_text_f1 = 0.0000`.
   - **Conclusion:** Neither architecture was actually tested on lyrics/harmony/tempo prediction.

2. **Fatal Init Confound — Random-Init `edge85m` Base (No Pretrained Checkpoint):**
   - `run_colab.py` instantiated a fresh random-init `UniOMRv2Net(edge85m)` (`base_eval`: `all90_s_note_f1 = 0.0000`, `vocal_s_note_f1 = 0.0000`) instead of warm-starting from a trained checkpoint.
   - Arm A froze this random-init base encoder + decoder, making `vocal_s_note_delta_pp = 0.0` vacuous (`0.0 - 0.0`) and providing random-noise `mem_post_merger` and `anchor_hidden` features to `TextSpotterHead`.
   - Arm B trained the full `edge85m` model from scratch for only 1,500 steps (`all90_s_note_f1 = 0.0034 EMA`, `vocal_s_note_f1 = 0.0042 EMA`).

3. **Wall-Clock Anomaly (`5.46 h` vs `75 min` Estimated):**
   - Training itself took only `1,565.8 s` (Arm A) + `2,141.9 s` (Arm B) = `3,707.7 s` (`61.8 min`).
   - The remaining `~15,948 s` (`~4.43 h`) was burned in 5 unmasked `suite31_hw` greedy eval sweeps (`base_eval` + Arm A `raw/ema` + Arm B `raw/ema`), where the random-init decoder ran to `max_len` garbage tokens on every page (`34.25 s`–`36.24 s` per vocal page).

4. **Valid, Non-Trivial Finding — Exact On-Disk Hybrid FP16/INT4 Footprint:**
   - **Arm A (`d_text=256`, 2 layers)** actually contains **`2,303,744` params (`2.304M` vs planned `~1.8M`)** and adds **`+2.629 MB`** (`14.712 MB` FP16 keep vs `13.120 MB` for Arm B, because `nn.MultiheadAttention` in-proj weights and `text_emb`/`pos_emb` stayed FP16), pushing total packed size to **`55.081 MB` (`55,080,928` bytes)** — **over the `< 55.0 MB` WebGPU ceiling by `+0.081 MB`**.
   - **Arm B (`+257` syllable BPE vocab tokens, `265 → 522`)** adds only **`+0.374 MB`** over `52.415 MB` `edge85m`, packing to **`52.789 MB` (`52,788,992` bytes)** — **comfortably under `< 55.0 MB` with `2.211 MB` headroom**.

---

## Follow-Up Action / Successor Experiment

- **Verdict:** **`ITERATE`** — do **not** promote or discard either text representation based on this confounded run.
- **Required Fixes for Successor (`i15-03b`, to be queued once a vocal/XML-sidecar tar + pretrained `edge85m` checkpoint are uploaded to HF):**
  1. **Fail-fast data guard:** `assert n_with_text >= 200` right after loading train rows, and bundle a dedicated permissive vocal/leadsheet dataset with actual `.musicxml` sidecars (OpenScore Lieder + Choral + vocal leadsheets) on HF.
  2. **Pretrained warm-start:** Load a trained `edge85m` (or `small_45m`) checkpoint so `base_eval` has real `S_note > 0.70`, meaningful `mem_post_merger` + `anchor_hidden` features, and fast `<eos>`-terminated eval.
  3. **Budget-compliant Arm A (`< 55.0 MB`):** Shrink `TextSpotterHead` to `d_text=192` (or use explicit `nn.Linear` Q/K/V/O projections quantized to INT4) so Arm A clears `< 55.0 MB` with `>= 1.0 MB` headroom alongside Arm B (`52.789 MB`).
