# Experiment Outcome — `2026-11-26T00-26-iter10-5-v1-redteam-fixes`

- **Status:** `COMPLETED`
- **Verdict:** **`PROMOTE`** — New all-time `suite31_hw` record (`all45_clean = 0.6984` / `0.7004` clean_gt, `all45_photo = 0.6832` / `0.6850` clean_gt, `hw_clean = 0.2591` / `0.2654`, `hw_photo = 0.2174` / `0.2232`, printed median fixes `12.0 / 12.0`). Promoted as the new **`v1` teacher** for Iter 11 (`UniOMR v2`).
- **Completed At:** `2026-11-26 02:25:43 EET` (`2026-11-26T00:25:43Z`, OOD evals completed `02:37 EET`)
- **Platform / GPU:** Local `RTX 4080 SUPER` (`16 GB` VRAM, Ada SM 8.9, native `BF16`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)
- **Elapsed Time:** `~119 min` (`00:26 → 02:25 EET`, `8,000 / 8,000` steps, `best_ema_ser = 0.314308`, `best_val_ser = 0.377377`)
- **Baseline Commit / Warm-Start:** Commit `052e210` | `D:\tmp\mvd-runs\uni-omr-iter10-v1\uni_omr_soup.pt`
- **Local Artifacts:**
  - Best EMA Checkpoint: `D:\tmp\mvd-runs\uni-omr-iter10_5-v1\uni_omr_ema_best.pt` (`179.16 MB`, `step=8000`)
  - Standard OOD Eval (`suite31_hw_manifest.json`): `D:\tmp\uni-omr-suite31-ood\iter10_5_ema_best_suite31_hw_1536.json`
  - Clean-GT OOD Eval (`suite31_hw_clean_gt_manifest.json`): `D:\tmp\uni-omr-suite31-ood\iter10_5_ema_best_suite31_hw_clean_gt_1536.json`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`, `repo_type="dataset"`):**
  - Checkpoint: `checkpoints/iter10_5_best/uni_omr_ema_best.pt` (commit `c6a8c77a`)
  - Eval JSON: `evals/2026-11-26T00-26-iter10-5-v1-redteam-fixes.json` (commit `90340d3b`)

---

## 1. Measured `suite31_hw` OOD Results (`1536px`, `max_len=4096`, `FSA + MTP-4`, `BF16`)

### A. Primary 6-Slice Comparison (Strict Note F1)

| Slice (`n`) | Iter 10 `best.pt` (12k) | Iter 10 `soup` (`k=4`) | **Iter 10.5 `ema_best` (8k)** | $\Delta$ vs `best.pt` | $\Delta$ vs `soup` | **Iter 10.5 `clean_gt`** | Iter 10.5 Tok F1 | Iter 10.5 Fixes (med) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`printed_clean`** (`31/31`) | **0.8993** | 0.8895 | **0.8968** | `-0.0024` | **`+0.0073`** | 0.8968 | **0.9554** | **12.0** *(was 19–21)* |
| **`printed_photo`** (`31/31`) | **0.8942** | 0.8738 | **0.8935** | `-0.0007` | **`+0.0197`** | 0.8935 | **0.9451** | **12.0** *(was 20)* |
| **`hw_clean`** (`14/14`) | 0.1953 | 0.2550 | **0.2591** | **`+0.0638`** | **`+0.0041`** | **0.2654** | **0.7640** *(cgt `0.7692`)* | 475.0 |
| **`hw_photo`** (`14/14`) | 0.1376 | 0.1516 | **0.2174** | **`+0.0798`** | **`+0.0658`** | **0.2232** | **0.7490** *(cgt `0.7514`)* | **488.0** *(was 569–606)* |
| **`all45_clean`** (`45/45`) | 0.6802 | 0.6921 | **0.6984** | **`+0.0182`** | **`+0.0063`** | **0.7004** | **0.8958** *(cgt `0.8975`)* | **32.0** *(was 40–43)* |
| **`all45_photo`** (`45/45`) | 0.6588 | 0.6491 | **0.6832** | **`+0.0244`** | **`+0.0340`** | **0.6850** | **0.8841** *(cgt `0.8848`)* | 41.0 |

- **Decode Speed & Speculative Acceptance:** `419.0 tok/s` (`97.03%` MTP-4 accept) on `printed_clean`, `417.9 tok/s` (`96.69%`) on `printed_photo`, `370.9 tok/s` (`75.03%`) on `hw_clean`, `374.0 tok/s` (`74.83%`) on `hw_photo`; `100%` (`90/90`) `decode_ok_rate`.
- **Internal Val (`eval_summary.json`, step `8000` EMA):** `printed_strip_val` SER `0.0529` (`tok_f1 0.9733`, `sys_count_exact 1.00`), `hw_strip_val` SER `0.5127` (`tok_f1 0.8499`, `ink_recall 0.9857`), `fullpage_val` SER `0.3774` (`tok_f1 0.8243`), `mean_val_ser = 0.314308`.

### B. Per-Piece Top Movers (Strict Note F1)

#### Handwritten Top Wins & Regressions (vs Iter 10 `soup` & `best.pt`)

| Piece (Domain) | Iter 10 `best` | Iter 10 `soup` | **Iter 10.5** | $\Delta$ vs `soup` | $\Delta$ vs `best` | `clean_gt` | Tok F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hw-debussy-la-plus-que-lente` (**photo**) | `0.0896` | `0.0000` | **`0.5678`** | **`+0.5678`** | **`+0.4782`** | `0.5678` | `0.9314` |
| `hw-beethoven-das-blumchen-wunderhold-d-b` (**photo**) | `0.4251` | `0.4397` | **`0.6259`** | **`+0.1862`** | **`+0.2008`** | **`0.6612`** | `0.8519` |
| `hw-debussy-vetement-du-blesse` (**clean**) | `0.0140` | `0.1967` | **`0.3480`** | **`+0.1513`** | **`+0.3340`** | `0.3480` | `0.7720` |
| `hw-debussy-berceuse-heroique` (**photo**) | `0.3858` | `0.3206` | **`0.4505`** | **`+0.1299`** | **`+0.0647`** | `0.4505` | `0.9020` |
| `hw-schubert-d810` (**clean**) | `0.0664` | `0.0589` | **`0.1810`** | **`+0.1221`** | **`+0.1146`** | `0.1810` | `0.6733` |
| `hw-cornelius-botschaft-d-b` (**clean / photo**) | `0.2396 / 0.2717` | `0.1201 / 0.2320` | **`0.2396 / 0.3409`** | `+0.1195 / +0.1089` | `+0.0000 / +0.0692` | **`0.2723 / 0.3866`** | `0.8229 / 0.8413` |
| `hw-beethoven-das-blumchen-wunderhold-d-b` (**clean**) | `0.4706` | `0.7696` | **`0.7933`** | `+0.0238` | **`+0.3228`** | **`0.8493`** | `0.8540` |
| `hw-debussy-la-plus-que-lente` (**clean**, reg.) | `0.5587` | `0.6229` | `0.4306` | `-0.1923` | `-0.1281` | `0.4306` | `0.9473` |
| `hw-mozart-k464` (**clean / photo**, reg.) | `0.2273 / 0.1148` | `0.3894 / 0.2007` | `0.2416 / 0.0594` | `-0.1479 / -0.1413` | `+0.0143 / -0.0553` | `0.2416 / 0.0594` | `0.8414 / 0.5880` |
| `hw-mozart-k421` (**clean**, reg.) | `0.0923` | `0.2123` | `0.1521` | `-0.0602` | `+0.0598` | `0.1521` | `0.6684` |

#### Printed Top Wins & Regressions (vs Iter 10 `best.pt`)

- **Top Wins:** `schoenberg-op19-2` (`0.8939 → 0.9727` ph `+7.88 pp`, `0.8667 → 0.9348` cl `+6.81 pp`), `cpebach-h186` (`0.7439 → 0.8266` ph `+8.27 pp`, `0.7737 → 0.8394` cl `+6.56 pp`), `weber-concertino` (`0.8715 → 0.9469` cl `+7.54 pp`, `0.8456 → 0.8951` ph `+4.96 pp`), `clara-op17-3` (`0.8764 → 0.9459` cl `+6.95 pp`, `0.9060 → 0.9570` ph `+5.11 pp`), `clara-polonaise-op1n1` (`0.7730 → 0.8385` ph `+6.54 pp`), `beach-prayer` (`0.9313 → 0.9867` cl `+5.54 pp`).
- **Top Regressions:** `essen-ballad` clean (`0.9444 → 0.5152`, `-42.93 pp`), `beethoven-op18n1-1` photo (`0.9755 → 0.7432`, `-23.22 pp`), `essen-altdeu` photo (`0.6102 → 0.4381`, `-17.21 pp` vs `best`, though **`+40.53 pp`** vs `soup` `0.0328`), `handel-lascia` (`0.9422 → 0.8207` cl `-12.16 pp`, `0.9264 → 0.8207` ph `-10.57 pp`).

---

## Scrutiny & Anomaly Check

1. **Coverage & Parse Integrity:** All `90/90` pages (`31` printed + `14` HW × `clean`/`photo`) scored cleanly on both `suite31_hw_manifest.json` and `suite31_hw_clean_gt_manifest.json`: `decode_ok_rate = 1.0`, `missing_keys = []`, `unexpected_keys = []`, `0` NaN/Inf values, `0` `zero_reason` failures, and `0` `max_len=4096` truncations (`clean=[]`, `photo=[]`).
2. **Apples-to-Apples Parity:** `eval_args` match Iter 10 `step12k` and `soup` identically (`max_dim=1536`, `max_len=4096`, `use_grammar=True`, `strict_grammar=False`, `use_mtp_spec=True`, `beam_size=1`, `measure_beam_size=0`, `BF16`).
3. **Domain / Slice Asymmetry & Outlier Audit:**
   - **HW Photo Breakthrough (`+6.58 pp` vs `soup`, `+7.98 pp` vs `best`):** Removing the 5 GT-destructive augmentations + flood-fill `paper_gate` + ink-density-adaptive routing slashed the `hw_clean → hw_photo` drop from `10.34 pp` (`soup`) down to **`4.17 pp`** and recovered `hw-debussy-la-plus-que-lente [photo]` from `0.0000 → 0.5678`.
   - **Printed Fixes/Page Nearly Halved (`19–21 → 12.0`):** Even though mean `printed_clean` (`0.8968`) and `printed_photo` (`0.8935`) stayed just below `0.90` due to 3 isolated bar-shift outliers (`essen-ballad` clean `-42.93 pp` alone accounts for `-1.38 pp` of the 31-piece mean; without it, `printed_clean` is `0.9107`), median user corrections dropped by **~40%**, printed `tok_f1` set new records (`0.9554` cl / `0.9451` ph), and MTP acceptance reached `97.0%`.
   - **Mozart SQ Clean Tradeoff (`hw-mozart-k464/k421/k387`):** Capping `real_hw` strip reuse (`220× → 4×`) at `0.06` weight traded some Mozart SQ clean memorization (`-6.0` to `-14.8 pp` vs `soup`) for huge gains on Debussy, Beethoven Lieder (`0.8493` clean_gt!), Schubert, and Cornelius across both clean and photo.
   - **Token-vs-Note Gap Still Dominates HW:** `hw-debussy-la-plus-que-lente [clean]` had `tok_f1 = 0.9473` yet `note_f1 = 0.4306` (`-0.1923`), confirming that single-bar onset/duration alignment drift remains the #1 lever for `hw_clean` (directly targeted by `DualVerifier` RLVR/TTC and `UniOMR v2` Iter 11).
4. **Hardware / Platform Check:** Local RTX 4080 SUPER completed all `8,000` steps with zero OOMs, zero WDDM stalls, and deterministic 5-minute atomic checkpoints.

---

## Verdict: `PROMOTE`

Iter 10.5 strictly beats Iter 10 `soup` on **6/6** `suite31_hw` slices, beats Iter 10 `best.pt` by **`+1.82 pp` `all45_clean`** and **`+2.44 pp` `all45_photo`**, crosses **`0.7004` `all45_clean`** on `clean_gt`, sets an all-time **`0.2174` (`0.2232` `clean_gt`) `hw_photo`** record, and cuts printed median fixes/page from `19–21` down to **`12.0`**.

## Follow-Up Action / Successor Experiment

- **Promoted as `v1` Teacher Checkpoint:** `D:\tmp\mvd-runs\uni-omr-iter10_5-v1\uni_omr_ema_best.pt` (synced to HF `checkpoints/iter10_5_best/uni_omr_ema_best.pt`) is now wired as the teacher for **Iter 11 (`UniOMR v2` `small_45m`, `24,000` steps)**: [`2026-11-26T02-40-iter11-v2-24k`](../2026-11-26T02-40-iter11-v2-24k/README.md).
- **Pending Follow-Ups:** Run 6-scout `learn-from-suite31` eyeball check on the `essen-ballad [clean]` / `beethoven-op18n1-1 [photo]` / `hw-debussy-la-plus-que-lente [clean]` bar-shift outliers, and re-export `web/models/` ONNX weights once Iter 11 completes (or promote Iter 10.5 directly if a browser release is needed earlier).
