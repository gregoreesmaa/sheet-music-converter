# Experiment Outcome — `2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep`

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T00:41:17Z` (started `2026-11-26T00:15:20Z`)
- **Platform / Hardware:** Google Colab `NVIDIA L4` (`24 GB`, `cc=8.9`, `BF16`, `cudnn_sdp=False`, CUTLASS/Flash SDPA)
- **Elapsed Time:** `1555.08 s` (`25.9 min` compute; `2883 s` total queue wall including setup/build; `~2.1 CU`). Colab session `ov-35` verified stopped (`colab.exe sessions`).
- **Checkpoint Evaluated:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`step=12000`, `val_ser=0.192301`, `missing_keys=[]`, `unexpected_keys=[]`, trained at `1280px`, `max_dec_len=4096`)
- **HF Artifacts:**
  - Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep.json` (pulled to [`eval_result.json`](./eval_result.json))
  - Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep.json` (pulled to [`run_ledger.json`](./run_ledger.json))
- **Verdict:** **`ITERATE`** *(with immediate finding: **`1280px / max_len=3840`** beats the `1536px / 3840` control by **`+0.0126`** on `all45_clean` (`0.7069` vs `0.6942`), **`+0.0181`** on `hw_clean`, **`+0.0102`** on `printed_clean`, and **`+0.0080`** on `printed_photo`; however, both `1280px` and `3840` hit the **lower boundary** of the grid and `hw_photo` showed a `-0.0054` asymmetry favoring `1536px`, so we immediately queue `ov-35b` to locate the true lower-resolution / tight-`max_len` peak before locking defaults).*

---

## Measured Results (3 × 3 `max_dim × max_len` Matrix on Full 45-Piece `suite31_hw`, 90 Pages)

Control cell: **`(1536px, 3840)`** (`all45_clean=0.6942`, `all45_photo=0.6617`). All deltas `Δ` are vs `(1536, 3840)`.

| `max_dim` | `max_len` | `printed_clean` | `printed_photo` | `hw_clean` | `hw_photo` | `all45_clean` (Δ vs ctrl) | `all45_photo` (Δ vs ctrl) | Trunc (`/90`) | Decode `90p` (`s`) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1280** | **3840** | **0.8997** (+0.0102) | **0.8818** (+0.0080) | **0.2799** (+0.0181) | 0.1867 (-0.0054) | **0.7069** (**+0.0126**) | **0.6655** (**+0.0038**) | 3 (3.3%) | 485.1 |
| 1280 | 5120 | 0.8997 (+0.0102) | 0.8818 (+0.0080) | 0.2783 (+0.0165) | 0.1849 (-0.0071) | 0.7063 (+0.0121) | 0.6650 (+0.0033) | 2 (2.2%) | 485.1 |
| 1280 | 6144 | 0.8997 (+0.0102) | 0.8818 (+0.0080) | 0.2773 (+0.0155) | 0.1843 (-0.0078) | 0.7060 (+0.0118) | 0.6648 (+0.0030) | 2 (2.2%) | 485.1 |
| *1536 (ctrl)* | *3840* | *0.8895* | *0.8738* | *0.2618* | ***0.1921*** | *0.6942* | *0.6617* | *3 (3.3%)* | *483.5* |
| 1536 | 5120 | 0.8895 (0.0000) | 0.8738 (0.0000) | 0.2588 (-0.0030) | 0.1862 (-0.0058) | 0.6933 (-0.0009) | 0.6599 (-0.0018) | 3 (3.3%) | 483.5 |
| 1536 | 6144 | 0.8895 (0.0000) | 0.8738 (0.0000) | 0.2576 (-0.0043) | 0.1836 (-0.0084) | 0.6929 (-0.0013) | 0.6591 (-0.0026) | 2 (2.2%) | 483.5 |
| 1792 | 3840 | 0.8502 (-0.0392) | 0.8602 (-0.0136) | 0.2370 (-0.0249) | 0.1865 (-0.0056) | 0.6594 (-0.0348) | 0.6506 (-0.0111) | 3 (3.3%) | 479.1 |
| 1792 | 5120 | 0.8502 (-0.0392) | 0.8602 (-0.0136) | 0.2322 (-0.0297) | 0.1791 (-0.0130) | 0.6579 (-0.0363) | 0.6483 (-0.0134) | 1 (1.1%) | 479.1 |
| 1792 | 6144 | 0.8502 (-0.0392) | 0.8602 (-0.0136) | 0.2314 (-0.0305) | 0.1791 (-0.0130) | 0.6577 (-0.0365) | 0.6483 (-0.0134) | 0 (0.0%) | 479.1 |

### Throughput & Per-Piece Resolution Headroom (`max_len=3840`)

| Slice | `1280px` Med Latency / TPS | `1536px` Med Latency / TPS | `1792px` Med Latency / TPS | Best-Res Wins (`1280 / 1536 / 1792 / tie`) | `Oracle(1280/1536)` | `Oracle(all 3)` |
|---|---:|---:|---:|---:|---:|---:|
| `printed_clean` (`n=31`) | `3.62s` / `266.0` | `3.41s` / `268.5` | `3.52s` / `257.0` | **`13` / `6` / `0` / `12`** | `0.9037` | `0.9037` |
| `printed_photo` (`n=31`) | `3.57s` / `266.4` | `3.62s` / `262.1` | `3.41s` / `256.8` | **`14` / `3` / `1` / `13`** | `0.8837` | `0.8839` |
| `hw_clean` (`n=14`) | `9.07s` / `245.6` | `7.76s` / `242.0` | `9.02s` / `235.7` | `4` / `2` / `5` / `3` | `0.2926` | `0.3005` |
| `hw_photo` (`n=14`) | `9.31s` / `244.0` | `7.90s` / `241.7` | `11.15s` / `238.8` | `4` / `3` / `2` / `5` | **`0.2091`** | `0.2116` |

---

## Scrutiny & Anomaly Check

1. **Coverage & Silent Failures — PASS (0 anomalies):**
   - All 45 pieces × 2 domains = 90 pages scored at all 9 `(max_dim, max_len)` cells (`n_pieces=31` printed, `14` HW).
   - `decode_ok_rate == 1.000` across every single arm (`0` decode failures, `0` NaN/None scores, `missing_keys == []`, `unexpected_keys == []`).
2. **Apples-to-Apples Sanity & `max_dec_len` Extrapolation Discovery:**
   - `(1536, 3840)` reproduces the Iter-10 soup printed baseline bit-for-bit (`printed_clean=0.8895`, `printed_photo=0.8738`).
   - **Critical config finding:** `uni_omr_soup.pt` has `model_config.max_dec_len = 4096` (`ckpt_dec_tokens = 4000`) and was trained at `1280px`. Both `max_len > 4096` (`5120`, `6144`) and `max_dim > 1280` (`1536`, `1792`) are **out-of-distribution extrapolations** relative to training!
3. **Why `max_len > 3840` Strictly Hurts (Zero Truncation Benefit, Pure Repetition-Loop Harm):**
   - **Printed (`62` pages):** Maximum generated length across all 3 resolutions is **`2400` tokens** (`max_pr_len = 2393 / 2392 / 2400`). **0/62 printed pages** ever hit `3840`, so `3840`, `5120`, and `6144` are bit-identical on printed.
   - **Handwritten (`28` pages):** Only `3/90` pages (`3.3%`, all Debussy manuscripts: `hw-debussy-lindaraja`, `hw-debussy-la-plus-que-lente`, `hw-debussy-berceuse-heroique`) exceed `3840` tokens. Extending `max_len` past `3840` **monotonically degrades Note F1 and inflates SER** on 8 of 9 `(res, page)` instances (e.g., at `1536px`, `hw-debussy-berceuse-heroique` photo drops `0.3217 → 0.2626 → 0.2486`; `hw-debussy-lindaraja` clean drops `0.2785 → 0.2359 → 0.2187`; `hw_photo` SER balloons `1.1176 → 1.1843 → 1.2176`). Extending `max_len` beyond `3840` is **falsified and discarded**.
4. **Resolution Monotonicity & Domain Asymmetry:**
   - **Massive `1280px` win over `1536px`:** `1280px` beats `1536px` by **`+0.0126`** on `all45_clean` (`0.7069` vs `0.6942`, surpassing even the Iter-10.5 `1536px` record `0.6984`!), **`+0.0102`** on `printed_clean` (`clara-op17-3` `+0.1248`, `verdi-la-donna` `+0.0895`), **`+0.0080`** on `printed_photo` (`aloha-oe` `+0.0545`, `schoenberg-op19-2` `+0.0440`), and **`+0.0181`** on `hw_clean` (`hw-cornelius-botschaft-d-b` `+0.2555`).
   - **`1792px` collapses OOD:** `all45_clean` plunges `-0.0348` (`0.6594`), `printed_clean` plunges `-0.0392` (`essen-altdeu` collapses `0.6102 → 0.0328`), and `hw_clean` plunges `-0.0249` (`hw-beethoven-das-blumchen-wunderhold-d-b` drops `0.8258 → 0.3561`).
   - **Boundary & `hw_photo` asymmetry:**
     1. Both `1280px` and `3840` sit at the **lower boundary** of the sweep; we have not yet bracketed the peak on the low side (`1024px`, `1152px`, `1408px` and `max_len` in `{2560, 3072, 3584, 3840}`).
     2. On **`hw_photo`**, `1536px` beats `1280px` by `+0.0054` (`0.1921` vs `0.1867`, driven by `hw-debussy-la-plus-que-lente` photo `0.4802` vs `0.3086` and `hw-debussy-lindaraja` photo `0.2475` vs `0.1689`), and `Oracle(1280/1536)` reaches **`0.2091`** (`+0.0170` above `1536px`, `+0.0224` above `1280px`) — directly motivating multi-scale TTA (`ov-36`).
5. **Hardware / Platform:** Zero OOMs, zero hangs; `1280px` and `1536px` decode in virtually identical wall-clock (`485.1s` vs `483.5s` for 90 pages at horizon `6144`, ~`260 tok/s`).

---

## Follow-Up Action / Successor Experiment

- **Verdict:** **`ITERATE`**
- **Immediate Takeaway for Iter 11 & Eval:** Never evaluate a `1280px`-trained checkpoint at `1792px` or `max_len > 3840`; `1280px / 3840` is already `+0.0126` better on `all45_clean` than `1536px / 3840`.
- **Queued Successor Experiment:** [`2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep`](../2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep/README.md) (`depends_on: ["2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep"]` in `experiments/overnight_queue/queue.json`).
  - Brackets the lower boundary discovered here by sweeping `max_dim in {1024, 1152, 1280, 1408}` × `max_len in {2560, 3072, 3584, 3840}` (16 cells from 4 fast GPU passes at horizon `3840`, ~14 min / ~1.2 CU on Colab L4) to lock the exact global Pareto-optimal `(max_dim, max_len)` before updating `eval_uni_suite31.py` and `docs/UNIOMR.md`.
