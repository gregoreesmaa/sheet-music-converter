# OUTCOME — `2026-11-26T07-40-i13-11-audiveris-humdrum-postproc` (`i13-11`)

- **Status:** `COMPLETED` (with silent no-op bugs in 2 of 3 symbolic rules)
- **Completed UTC:** `2026-11-26T09:06:40Z` (started `08:58:02Z`)
- **Platform / GPU:** `colab-NVIDIA L4` (`bf16=True`)
- **Elapsed Time:** `518.31 s` (`8.6 min`)
- **Base Checkpoint:** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`uni_omr_v1`, step `12000`, `val_ser=0.192301`, `missing_keys=[]`, `unexpected_keys=[]`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - `evals/2026-11-26T07-40-i13-11-audiveris-humdrum-postproc.json`
  - `runs/2026-11-26T07-40-i13-11-audiveris-humdrum-postproc.json`

## Measured Results (`max_dim=1280`, `max_len=3840`, MTP-4 + FSA, all 90 pages)

| Slice (`n`) | Baseline Note F1 | Baseline `r_rhythm` | `header_continuity` Δ / Edits | `accidental_norm` Δ / Edits | `rhythm_repair` Δ / Edits | `combined_all` Δ / Edits |
|---|---:|---:|---:|---:|---:|---:|
| `all45_clean` (`45`) | `0.541416` | `0.7342` | `+0.000000` / `0` | `+0.000000` / `0` | `+0.000000` / `0` | `+0.000000` / `0` |
| `all45_photo` (`45`) | `0.471400` | `0.7198` | `+0.000000` / `0` | `+0.000000` / `0` | `+0.000000` / `0` | `+0.000000` / `0` |
| `pr_clean` (`31`) | `0.667540` | `0.8339` | `+0.000000` / `0` | `+0.000000` / `0` | `+0.000000` / `0` | `+0.000000` / `0` |
| `pr_photo` (`31`) | `0.619134` | `0.8650` | `+0.000000` / `0` | `+0.000000` / `0` | `+0.000000` / `0` | `+0.000000` / `0` |
| `hw_clean` (`14`) | `0.262143` | `0.5134` | `+0.000000` / `0` | `+0.000000` / `0` | `+0.000000` / `0` | `+0.000000` / `0` |
| `hw_photo` (`14`) | `0.144274` | `0.3983` | `+0.000000` / `0` | `+0.000000` / `0` | `+0.000000` / `0` | `+0.000000` / `0` |

## Scrutiny & Anomaly Check

1. **Coverage & Integrity:** All `90/90` pages decoded cleanly (`0` decode failures, `0` NaNs, `missing_keys=[]`). Baseline numbers match `iter10_soup_baseline` cleanly.
2. **CRITICAL SILENT NO-OP BUG #1 (`audiveris_header_continuity` = `0` edits):** `rule_audiveris_header_continuity` splits `pred_raw` on `sys_tok = "<sys>"` and immediately returns `0` edits when `len(systems) < 2`. **The token `"<sys>"` does not exist in `UniOMR`'s 237-token vocabulary** (verified via `eval_uni_suite31._vocab_spec(237)`), so every page was treated as a 1-system score and skipped!
3. **CRITICAL SILENT NO-OP BUG #2 (`audiveris_rhythm_repair` = `0` edits):** Line 389 of `run_colab.py` bails out of any bar containing `t.startswith("voice-")` (which rejects every measure emitting `voice-1`, i.e., nearly all polyphonic/grand-staff scores!), `tuplet-*`, `grace`, or `bar-rest`. Even on remaining bars, it only tests exact single-edit repair on `events[-1]`.
4. **Legitimate Zero-Edit Finding (`humdrum_accidental_norm` = `0` edits):** FSA grammar decoding already enforces `<step> <octave> [alter]` ordering, so orphan/duplicate alteration tokens never occur post-FSA. This rule is confirmed unnecessary and can be dropped.

## Verdict: **`ITERATE`**

- The Audiveris header-continuity and measure-rhythm-repair hypotheses were **not actually exercised** (`total_edits == 0` across all 90 pages due to the non-existent `"<sys>"` token and the overly broad `voice-*` bail-out). Only `humdrum_accidental_norm` is falsified/redundant under FSA decoding.

## Follow-Up Action / Successor Experiment

- **Proposed Successor:** `i13-11b-audiveris-voice-aware-rhythm-repair` (depends on `2026-11-26T07-40-i13-11-audiveris-humdrum-postproc`):
  1. Fix `audiveris_rhythm_repair` to segment each bar **per `voice-*` stream** (repairing each voice independently instead of skipping bars with `voice-1`), and add a pre-flight assertion that `total_edits > 0` on a smoke slice.
  2. Replace `<sys>`-based header continuity with `bar`-indexed clef/key outlier smoothing (detecting isolated 1-bar or mid-stream clef/key glitches without relying on `<sys>`).
  3. Cache `pred_raw` tokens in the eval artifact so future deterministic symbolic post-processing ablations run in `< 5 s` on CPU with `0` GPU cost.
