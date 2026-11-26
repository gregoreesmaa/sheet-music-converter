# OUTCOME — `2026-11-26T01-38-ov-21-paper-gate-border-threshold` (`ov-21`)

- **Status:** `COMPLETED` (all 9 grid configs × 290 pages = 2,610 evaluations scored; bit-exact parity verified)
- **Completed:** `2026-11-26T08:29:53Z` (`elapsed_sec = 203.22` wall / `68.24` in-script)
- **Platform / GPU:** Modal `1× NVIDIA L4` (`ap-pq1rnjRxfRTxZ1k9EIRnLX`, confirmed `stopped`, 0 tasks)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T01-38-ov-21-paper-gate-border-threshold.json` (downloaded to `eval_result.json`)
  - Run Ledger: `runs/2026-11-26T01-38-ov-21-paper-gate-border-threshold.json` (downloaded to `run_meta.json`)
- **Verdict:** **`PROMOTE`** — Promote **`(luma_stop=0.35, dark_thr=0.30)` (`luma35_dark30`)** over current default `(0.45, 0.30)` in `vir_router_v2.py` and `uni_omr_v2.py`.

---

## 1. Measured Results (290 Pages: 90 `suite31_hw` + 200 Real RISM HW, `1280px`, `parity_max_diff = 0.0`)

| Config (`luma_stop`, `dark_thr`) | All-290 Ink Rec (mean / p05 / **min**) | RISM-200 Ink Rec (mean / p05 / **min**) | `hw_photo` Ink Rec (mean / **p05**) | `hw_clean` Ink Rec (mean / p05) | `printed_photo` Border Sup | `hw_photo` Border Sup | RISM Border Sup | `printed_clean` False Mask | Collapsed Pages | Top-40% Ink Rec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `luma35_dark22` (`0.35, 0.22`) | 0.99838 / 1.00000 / **0.8352** | 0.99870 / 1.00000 / **0.8950** | 0.99684 / 0.98445 | 0.98823 / 0.94232 | 0.98784 | 0.96115 | 0.85479 | 0.00000 | 0 | 0.70690 |
| **`luma35_dark30` (`0.35, 0.30`) ★** | **0.99831 / 1.00000 / 0.8352** | **0.99863 / 1.00000 / 0.8950** | **0.99639 / 0.98228** | **0.98823 / 0.94232** | **0.98784** | **0.96260** | **0.88476** | **0.00000** | **0** | **0.71088** |
| `luma35_dark38` (`0.35, 0.38`) | 0.99829 / 1.00000 / 0.8352 | 0.99862 / 1.00000 / 0.8950 | 0.99614 / 0.98027 | 0.98823 / 0.94232 | 0.98784 | 0.96273 | 0.88582 | 0.00000 | 0 | 0.71124 |
| `luma45_dark22` (`0.45, 0.22`) | 0.99643 / 0.99952 / 0.6599 | 0.99594 / 0.99999 / 0.6599 | 0.99612 / 0.98090 | 0.98782 / 0.93898 | 0.99056 | 0.96228 | 0.86485 | 0.00000 | 0 | 0.70748 |
| `luma45_dark30` (`0.45, 0.30`, **default**) | 0.99613 / 0.99952 / 0.6341 | 0.99559 / 0.99999 / 0.6341 | 0.99566 / 0.97871 | 0.98718 / 0.93274 | 0.99057 | 0.96370 | 0.89443 | 0.00000 | 0 | 0.71184 |
| `luma45_dark38` (`0.45, 0.38`) | 0.99600 / 0.99952 / 0.6166 | 0.99543 / 0.99991 / 0.6166 | 0.99528 / 0.97572 | 0.98710 / 0.93179 | 0.99057 | 0.96387 | 0.89552 | 0.00000 | 0 | 0.71295 |
| `luma55_dark22` (`0.55, 0.22`) | 0.99271 / 0.99451 / 0.0628 | 0.99070 / 0.99600 / 0.0628 | 0.99406 / 0.97231 | 0.98769 / 0.93745 | 0.99057 | 0.96390 | 0.87262 | 0.00000 | **2** | 0.70701 |
| `luma55_dark30` (`0.55, 0.30`) | 0.99230 / 0.99313 / 0.0446 | 0.99022 / 0.99322 / 0.0446 | 0.99323 / 0.97165 | 0.98694 / 0.93063 | 0.99057 | 0.96522 | 0.90163 | 0.00000 | **2** | 0.71146 |
| `luma55_dark38` (`0.55, 0.38`) | 0.99196 / 0.98926 / 0.0042 | 0.98978 / 0.99157 / 0.0042 | 0.99267 / 0.97049 | 0.98678 / 0.92913 | 0.99057 | 0.96537 | 0.90220 | 0.00000 | **2** | 0.71360 |

### Delta: `luma35_dark30` vs Current Default `luma45_dark30`
- **All-290 Interior Ink Recall:** mean `+0.00218` (`0.99831` vs `0.99613`), p05 `+0.00048` (`1.00000` vs `0.99952`), **worst-page min `+0.2011` (`0.8352` vs `0.6341`)**.
- **RISM-200 Dark Aged Parchment Tail:** worst-page min jumps **`+0.2609`** (`0.8950` vs `0.6341`); specifically `rism-de_211004870` recovers from **`0.6341 → 0.9227` (`+0.2886`)** and `rism-de_211005477` recovers from **`0.6976 → >0.96`**, with `0` collapsed pages.
- **`hw_photo` & `hw_clean`:** `hw_photo` p05 ink recall improves **`+0.00357`** (`0.98228` vs `0.97871`); `hw_clean` p05 improves **`+0.00958`** (`0.94232` vs `0.93274`).
- **Desk-Border Suppression Gate (`>= 0.95`):** `printed_photo = 0.98784` (`-0.00273`), `hw_photo = 0.96260` (`-0.00110`) — both comfortably pass `>= 0.95`.
- **Why `dark_thr=0.30` over `0.22`:** Retains `+0.0300` RISM border suppression (`0.88476` vs `0.85479`) and `+0.0040` `top40_ink_recall` (`0.71088` vs `0.70690`) at virtually identical ink recall (`0.99831` vs `0.99838`, `p05=1.00000`).

---

## 2. Scrutiny & Anomaly Check

1. **Coverage & Parity:** All `290/290` pages (`31` `printed_clean`, `31` `printed_photo`, `14` `hw_clean`, `14` `hw_photo`, `200` `rism_real_hw_200`) scored across all 9 grid points with zero NaNs and exact bit-level parity (`parity_max_diff_vs_vir_router_v2 = 0.0`) at the `(0.45, 0.30)` default.
2. **`summary_grid.clean_false_mask` (`0.034–0.038`) Explained:** `_collect_eval_pages` marked all `domain == "clean"` pages (both `printed_clean` and `hw_clean`) as `is_clean_borderless=True`, even though `12/14` real historical `hw_clean` scans contain genuine dark archival scanner borders (`paper_frac_mean = 0.88575`, `desk_border_suppression_mean = 0.95085`). On truly borderless `printed_clean` (31 pages), `clean_false_mask_rate_mean = 0.00000` (`0.00%`) across all 9 configs, cleanly passing the `<= 0.005` gate.
3. **Catastrophic Failure at `luma_stop=0.55`:** Raising `luma_stop` to `0.55` causes 2 RISM dark-parchment pages (`rism-de_211005477`, `rism-de_211004870`) to collapse (`0.0042–0.0628` ink recall). Never raise `luma_stop` above `0.45`.
4. **Platform Tag Cosmetic Note:** `run_colab.py` hardcoded `"platform": "colab-NVIDIA L4"` even though the job was dispatched on Modal L4 (`ap-pq1rnjRxfRTxZ1k9EIRnLX`).

---

## 3. Verdict & Follow-Up Action

- **Verdict:** **`PROMOTE`** `(luma_stop=0.35, dark_thr=0.30)` (`luma35_dark30`).
- **Action (to wire in next code-change window before Iter 13 training):** Update `luma >= 0.45` → `luma >= 0.35` (keeping `dark_thr = 0.30`) in:
  - `tmp-train/mvd/vir_router_v2.py` (`compute_ink_and_paper_gate_f32`)
  - `tmp-train/mvd/uni_omr_v2.py` (`InkTokenRouterV2._compute_ink_and_paper_gate`)
  - Matching unit tests (`test_vir_router_v2.py` / `test_uni_omr_v2.py`).
  *(Note: `tmp-train/mvd/` was intentionally not modified during this CPU-only post-run triage pass.)*
