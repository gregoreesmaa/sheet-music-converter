# Experiment Plan — `2026-11-26T14-30-i13-16b-mathpix-conf-gated-redecode-fixed`

## 1. Root Causes Fixed from `i13-16`
In `i13-16`, every single treatment arm produced `mean_triggers == 0.0` across all 48 pages because:
1. `BAR_BOUNDARY_TOKS = {"|", "<bar>", "<sys>"}` matched `0` tokens in `UniOMR`'s 237-token vocabulary (`boundary_ids == set()`).
2. `bar_start_kv_len > 0` blocked bar-1 rollbacks even if boundaries were found.
3. The script ran on T4 FP32 (`88.1 min`) against `iter10_5_best` (`val_ser=0.3143`) instead of `iter10_soup_baseline` (`val_ser=0.1923`) on L4 BF16.

## 2. Experimental Grid (`max_dim=1280, max_len=3840, max_bar_toks=120`)
- **Base Checkpoint:** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` on L4 BF16 (`~15–20 min`).
- **Slices:** All `14` HW manuscripts (`28` pages) + `10` printed safety pieces (`20` pages) = `48` pages.
- **Arms:**
  1. `baseline_greedy` (`mode="off"`)
  2. `conf_minp_0.65` (`bottom2_pmax < 0.65`, `beam=3, max_redecodes=6`)
  3. `conf_minp_0.75` (`bottom2_pmax < 0.75`, `beam=3, max_redecodes=6`)
  4. `conf_minp_0.82` (`bottom2_pmax < 0.82`, `beam=3, max_redecodes=6`)
  5. `conf_margin_0.25` (`min_margin < 0.25`, `beam=3, max_redecodes=6`)

## 3. Hard Safety Assertions
- Startup: `assert len(boundary_ids) >= 3`.
- Post-HW split: `assert total_triggers > 0` across treatment arms.
- Pre-upload: `assert total_scored > 0`.

## 4. Promotion / Discard Gates
- **PROMOTE:** Best arm improves `hw_clean` OR `hw_photo` by `>= +0.008` Note F1 with `pr_clean >= -0.002`, `pr_photo >= -0.002`, and median latency overhead `< 1.35×`.
- **ITERATE:** Positive HW gain (`>= +0.005`) but overhead `> 1.35×` -> tighten `max_redecodes=3` or combine with MTP-4 fast path.
- **DISCARD:** All arms within `[-0.003, +0.003]` despite `total_triggers > 0`.
