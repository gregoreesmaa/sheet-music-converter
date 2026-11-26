# `2026-11-26T14-30-i13-16b-mathpix-conf-gated-redecode-fixed` (`i13-16b`)

- **Parent Experiment:** `2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode` (`ITERATE` — 100% silent no-op due to `BAR_BOUNDARY_TOKS` vocabulary mismatch)
- **Goal:** Properly evaluate Mathpix-style confidence-gated per-bar beam re-decoding (`beam=3, max_redecodes=6`) on `suite31_hw` (`14` HW × 2 + `10` printed × 2 = `48` pages) after fixing:
  1. `BAR_BOUNDARY_TOKS = {"bar", "bar-double", "bar-light-heavy", "bar-repeat-left", "bar-repeat-right"}` + `assert len(boundary_ids) >= 3`.
  2. Bar-1 rollback support (`kv_prefix = None`, `memory = mem` when rolling back to BOS) so bar 1 is eligible for re-decoding too.
  3. Target `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` on L4 BF16 (`~15–20 min`).
  4. Fail-fast assertions: `assert total_triggers > 0` and `assert total_scored > 0`.
