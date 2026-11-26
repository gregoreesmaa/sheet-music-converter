# Outcome — `2026-11-26T02-40-iter11-v2-24k` (`UniOMR v2` Iter 11 Cold-Start)

- **Status:** `STOPPED_EARLY` at step `8,000 / 24,000` (`33.3%`, ~4.5h RTX 4080 SUPER)
- **Verdict:** **`ITERATE`** → Superseded immediately by [`2026-11-26T07-15-iter12-v2-strip-warmup-24k`](../2026-11-26T07-15-iter12-v2-strip-warmup-24k/)

## Measured Trajectory (`step 1 → 8,000`)
- Training loss dropped monotonically (`7.5901 → 1.4918`, `ce 5.63 → 0.994`, `mtp 5.62 → 1.426`, `hog 0.307 → 0.035`, `kd 0.645 → 0.189`), zero OOMs, VRAM `10.6 / 16.0 GB`.
- **In-training validation (`step 4,000 → 8,000`):**
  - `strip_val` (Raw / EMA): `F1 0.3582 → 0.4530` / `0.3760 → 0.4275` (`SER 1.14 → 0.85`)
  - `fullpage_val` (Raw): `F1 0.3180 → 0.3586`, `sys_exact 0.273 → 0.409`
  - `hw_strip_val` (Raw / EMA): `F1 0.1783 → 0.2071` / `0.2209 → 0.2147`

## Scrutiny & Anomaly Check
1. **No bug or collapse — pure cold-start curriculum + LR bottleneck:**
   - Throwing a freshly initialized `UniOMR v2` (`41.3M`) straight into `80%` `1536px` full pages (`~4,900` visual tokens/page, `eff_batch=4`) meant 8,000 steps was only `0.57` of ONE epoch over `44,645` pages.
   - Concurrently, `ov-13` and Exp 4 proved `--lr 3.0e-4` under-drives `Muon` (`5.0e-4` beat it by `-0.111` CE and `+6.9 pp` F1 with *lower* grad norms, and decoupling `--muon-lr` from AdamW avoids RAW decode jitter).
   - Concurrently, `ov-35` proved `1280px` beats `1536px` by `+1.26 pp` `all45_clean` (`0.7069`) while running `~35%` faster per step.
2. **Decision:** Stop at step `8,000` (saving ~8h GPU time) and launch **Iter 12** (`4k` strip-only warmup with decoupled `--muon-lr 1.0e-3` → `20k` `mixed` full-page at `1280px` + `Iter 10.5` KD + unrotated INT4 QAT).
