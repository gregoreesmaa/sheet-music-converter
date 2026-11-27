# OUTCOME — `2026-11-27T07-52-i15-08-bekern3-expressive-oracle-and-compress`

- **Status:** `COMPLETED` (with **invalid Part B bake-off** due to from-scratch under-training collapse and **Part A fallback/gate failure**)
- **Verdict:** **`ITERATE`** — **Do NOT wire into Iter 15 yet.**
- **Platform / Hardware:** Google Colab `L4` (`cuda`, `bf16: true`, `steps_per_arm: 600`)
- **Timestamps:** Started `2026-11-27T07:24:09Z` → Completed `2026-11-27T09:15:46Z` (**`111.6 min`** wall-clock; Part A `1.41 s`, Part B train `40.2 min`, Part B eval `~71 min` due to 1536-token runaway decoding)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T07-52-i15-08-bekern3-expressive-oracle-and-compress.json` → local [eval_hf.json](eval_hf.json)
  - Run Ledger: `runs/2026-11-27T07-52-i15-08-bekern3-expressive-oracle-and-compress.json` → local [run_hf.json](run_hf.json)

---

## 1. Measured Results

### Part A — `part_a_oracle_closure` (H1: Expressive/Text Oracle Closure)

> **Corpus/Scorer Fallback:** `datasets/northstar_eval_pack.tar` was unavailable on HF and `omr_northstar` was not installed on the VM, so Part A fell back to `corpus_source = "suite31_hw_fallback"` (`n_pages = 45`, not 125) and `scorer_mode = "stdlib_fallback"`.

| Metric | Measured (45 pages, `stdlib_fallback`) | PLAN.md Gate | Pass? |
|---|---:|---:|:---:|
| **`NS`** | **`95.962`** | `>= 98.0` | **FAIL** (`-2.038`) |
| `S_note` | `1.0000` | — | OK |
| `S_struct` | `0.9556` | — | Bottleneck (repeat/barline styles) |
| **`S_expr`** | **`0.8426`** | `>= 0.980` | **FAIL** (`-0.1374`) |
| `S_play` | `1.0000` | — | OK |
| **`CFR`** | **`0.0000`** | `== 0.0` | PASS |
| **Part A Gate** | **`false`** | `NS >= 98.0 & S_expr >= 0.980 & CFR == 0.0` | **FAIL** |

**Worst Part A Round-Trip Pages:**
- **Structural failures (`S_struct`):** `gt-p1-chopin-mazurka06-2` (`NS=83.038`, `S_struct=0.1519`, `S_expr=1.0`), `gt-p1-palestrina-agnus` (`NS=87.500`, `S_struct=0.3750`, `S_expr=1.0`), `gt-p1-ryans-jig` (`S_struct=0.7500`), `gt-p1-foster-leadsheet` (`S_struct=0.8000`).
- **Expressive/Text failures (`S_expr`):** `gt-p1-schumann-op41n1-3` (`S_expr=0.5000`), `gt-p1-hw-beethoven-das-blumchen-wunderhold-d-b` (`S_expr=0.5508`), `gt-p1-fiddle-strathspey` (`S_expr=0.5714`), `gt-p1-ryans-hornpipe` (`S_expr=0.5714`), `gt-p1-oneills-tune` (`S_expr=0.5833`), `gt-p1-clara-op17-3` (`S_expr=0.5882`).

---

### Part B — `part_b_compression_mtp_bakeoff` (H2 & H3: Compression + MTP-4 Speedup)

| Metric | `arm0_bekern_v2a` (Control) | `arm1_profile_a_comp` (H2) | `arm2_profile_a_plus_core_pitch` (H3) |
|---|---:|---:|---:|
| Vocab size | `265` | `307` (`+42`) | `342` (`+77`) |
| Lossless round-trip (`proxy_val[:50]`) | N/A | **`50/50`** (`100%`) | **`50/50`** (`100%`) |
| Mean tokens/page (`orig -> enc`) | `876.1 -> 876.1` | `876.1 -> 723.1` | `876.1 -> 533.2` |
| Compression ratio / token reduction | `1.0000` (`0.00%`) | **`0.8254` (`-17.46%`)** | **`0.6086` (`-39.14%`)** |
| `T > 3840` truncation rate | `0.0000` | `0.0000` | `0.0000` |
| Train steps / elapsed | `600` / `809.25 s` | `600` / `801.81 s` | `600` / `798.65 s` |
| Final 25-step train loss | `1.8855` | `2.1864` (`+0.3009`) | `2.8635` (`+0.9780`) |
| `proxy_val` Token F1 (`n=48`) | `0.0082` | `0.0070` | `0.0078` |
| `proxy_val` Strict Note F1 / Rhythm F1 | **`0.0000` / `0.0000`** | **`0.0000` / `0.0000`** | **`0.0000` / `0.0000`** |
| `suite31_hw` Overall Strict Note F1 (`n=90`) | **`0.0000`** | **`0.0000`** | **`0.0000`** |
| `suite31_hw` Slices (`pr_clean`/`pr_photo`/`hw_clean`/`hw_photo`) | `0.0`/`0.0`/`0.0`/`0.0` | `0.0`/`0.0`/`0.0`/`0.0` | `0.0`/`0.0`/`0.0`/`0.0` |
| Mean AR generated tokens (`bench_pages=16`) | **`1536.0`** *(hit cap)* | **`1536.0`** *(hit cap)* | **`1536.0`** *(hit cap)* |
| Mean MTP-4 generated tokens (`bench_pages=16`) | **`1536.0`** *(hit cap)* | **`1536.0`** *(hit cap)* | **`1536.0`** *(hit cap)* |
| Mean AR latency (`ms/page`, `n=16`) | `20,727.78` | `20,980.03` | `21,269.54` |
| Mean MTP-4 latency (`ms/page`, `n=16`) | `6,154.78` | `6,253.49` | `10,688.04` |
| Mean `suite31_hw` MTP-4 latency (`ms/page`, `n=90`) | `6,118.87` | `6,243.29` | `10,711.16` |
| Intra-arm MTP-4 speedup (`AR / MTP-4`) | `3.368×` | `3.355×` | `1.990×` |
| MTP-4 mean accepted tokens/step | `3.966` *(degenerate loop)* | `3.964` *(degenerate loop)* | `2.3645` |
| **Wall-clock speedup vs `arm0` MTP-4** | `1.000×` | **`0.984×`** *(SLOWER)* | **`0.576×`** *(42% SLOWER)* |
| Script's `mtp4_speedup_vs_arm0_ar` | `3.368×` | `3.315×` *(wrong denom)* | `1.939×` *(wrong denom)* |
| Script's `gate_pass` | — | `true` *(FALSE POSITIVE)* | `true` *(FALSE POSITIVE)* |

---

## 2. Scrutiny & Anomaly Check

### Critical Anomaly 1: Part B Total Collapse (`0.0000` Strict Note F1) & False-Positive `gate_pass: true`
1. **Severe under-training from scratch:** All three arms trained `UniOMRv2Net(small_45m)` (`~43M` params) **from random initialization** for only `600` optimizer steps (`grad_accum=4` → `2,400` full-page views at `1280px`) with `gpu_ema_decay=0.99`. Final losses remained at `1.8855` / `2.1864` / `2.8635` — far from convergence.
2. **100% EOS-free runaway loops:** Every arm hit `EVAL_MAX_LEN = 1536.0` tokens on 100% of evaluated pages (`mean_ar_gen_tokens == 1536.0`, `mean_mtp4_gen_tokens == 1536.0`), producing `0.0000` Strict Note F1, `0.0000` Rhythm F1, and `0.0070–0.0082` Token F1 across all `48` `proxy_val` and `90` `suite31_hw` pages. The seemingly impressive `3.966` / `3.964` MTP-4 acceptance rate on `arm0`/`arm1` is a **collapse artifact** (speculating a trivially repeating token loop).
3. **False-positive gate bug (`run_colab.py:1978`):**
   - `delta_note_f1_pp = 100.0 * (0.0 - 0.0) = 0.0 >= -0.2` passed vacuously because there was no baseline accuracy floor (e.g., `arm0_nf1 >= 0.30`).
   - Speedup was checked against `arm0_ar` (`spd_vs_ar >= 1.45`) instead of `arm0_mtp4`. Against `arm0_mtp4`, `arm1` was **`0.984×`** and `arm2` was **`0.576×`** (both slower because neither terminated before 1536 tokens, so shorter target sequences yielded zero decode-length savings).

### Critical Anomaly 2: Part A Corpus/Scorer Fallback & Specific Compiler Gaps
1. **Not apples-to-apples with 125-page `OMR-NorthStar`:** `datasets/northstar_eval_pack.tar` was missing from HF and `omr_northstar` was absent on Colab, causing fallback to 45 `suite31_hw` pieces with `stdlib_fallback`.
2. **Concrete `musicxml_to_bekern3` / `bekern3_to_musicxml` bugs explaining `S_struct=0.9556` and `S_expr=0.8426`:**
   - **Barlines (`S_struct`):** Only `dashed`, `light-heavy`, `light-light` are emitted (`run_colab.py:575-580`); `heavy-light`, `regular`, `none`, `dotted`, `heavy`, `heavy-heavy`, `tick`, `short` and repeat barlines are silently dropped (causing `chopin-mazurka06-2` `0.1519` and `palestrina-agnus` `0.3750`).
   - **Multi-verse `<sys_lyr>` overflow (`S_expr`):** `bekern3_to_musicxml` attaches deferred `lyr:<num>:<txt>` tokens sequentially to `note_els_in_measure[l_idx]` (`run_colab.py:738-745`); any multi-verse piece (e.g. `beethoven-das-blumchen-wunderhold` `0.5508`, `foster-leadsheet` `0.7831`) overflows `len(note_els_in_measure)` and drops extra verses.
   - **Dropped MusicXML tags (`S_expr`):** `EXT_ART_SET` omits tags counted by `_extract_musicxml_axes` (`fingering`, `string`, `fret`, `wavy-line`, `accidental-mark`, etc.); `slur`/`wedge` `type="continue"` are dropped and spanner numbers clamped (`<=3` / `<=2`); `dyn-*` outside `EXT_DYN_SET` (e.g., `pppp`, `ffff`, `sfp`) collapse into `other-dynamics`; `ottava-stop` loses its `size` attribute.

### Valid Empirical Signal Salvaged
- **Lossless Profile-A & Core-Pitch transforms are bijectively verified (`50/50` exact round-trip on `proxy_val`):**
  - `arm1_profile_a_comp`: **`-17.46%`** tokens (`876.1 -> 723.1`, ratio `0.8254`, `+42` vocab).
  - `arm2_profile_a_plus_core_pitch`: **`-39.14%`** tokens (`876.1 -> 533.2`, ratio `0.6086`, `+77` vocab).
- **Early entropy warning on `arm2`:** `arm2` train loss lagged control by `+0.978` (`2.8635` vs `1.8855`) and MTP-4 acceptance fell to `2.3645` vs `3.964` (`arm1`), indicating higher per-token entropy for fused `C2..B6` pitch tokens that must be tested on a properly converged/warm-started model.

---

## 3. Decision & Iter 15 Wiring Recommendation

- **Verdict:** **`ITERATE`**
- **Wire into Iter 15?** **NO.** Neither Part A nor Part B passed valid gates; do **not** freeze `bekern3` expressive tokens or Profile-A / Core-Pitch compression into Iter 15 yet.

## 4. Follow-Up Action / Successor Experiment

Queue a focused successor (e.g. `i15-08b-bekern3-oracle-and-warmstart-compress`):
1. **Part A (CPU-only, ~2 min):** Fix the `musicxml_to_bekern3` / `bekern3_to_musicxml` bugs above (all `bar-style` values + repeats, note-anchored/verse-indexed `<sys_lyr>`, full `EXT_ART_SET`/`EXT_DYN_SET`, `slur`/`wedge` `continue` + unclipped numbers, `ottava-stop` size), upload/mount `northstar_eval_pack` + official `omr-northstar` scorer, and verify `NS >= 98.0`, `S_expr >= 0.980`, `CFR == 0.0` on all 125 `OMR-NorthStar` pages.
2. **Part B (GPU bake-off):** **Warm-start** all 3 arms from a converged `UniOMRv2Net` checkpoint (expanding only token embedding + LM/MTP heads for `+42`/`+77` new tokens) so `arm0` achieves real `suite31_hw` Strict Note F1 (`>= 0.40`) and real EOS termination (`eos_rate >= 0.80`), and enforce strict gates: `arm0_nf1 >= 0.30`, `delta_note_f1_pp >= -0.2`, and `mtp4_speedup_vs_arm0_mtp4 >= 1.45×` (comparing apples-to-apples MTP-4 vs MTP-4).
