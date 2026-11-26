# Outcome — `2026-11-26T14-30-i13-11b-audiveris-voice-aware-rhythm-repair` (`i13-11b`)

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T18:01:36Z` (`started_utc=2026-11-26T17:54:05Z`)
- **Platform / GPU:** Google Colab `NVIDIA L4` (`BF16`; note: runner logged cosmetic string `"kaggle-NVIDIA L4"`)
- **Elapsed Time:** `451.32 s` (`7.5 min`, single GPU decode pass + CPU symbolic post-processing)
- **Checkpoint:** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`uni_omr_v1`, step `12000`, `val_ser=0.192301`, `missing_keys=[]`, `unexpected_keys=[]`)
- **Config:** Canonical `max_dim=1280`, `max_len=3840`, MTP-4 + FSA on all `90` pages of `suite31_hw` (`45 clean + 45 photo`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON (includes cached `pred_raw` for all 90 pages): `evals/2026-11-26T14-30-i13-11b-audiveris-voice-aware-rhythm-repair.json`
  - Run Ledger JSON: `runs/2026-11-26T14-30-i13-11b-audiveris-voice-aware-rhythm-repair.json`

---

## Measured Results (`suite31_hw`, 90 pages)

### Overall (`all45_clean`, `n=45` & `all45_photo`, `n=45`)

| Arm | `all45_clean` Note F1 | Delta | `r_rhythm` | Edits | W/T/L | `all45_photo` Note F1 | Delta | `r_rhythm` | Edits | W/T/L |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| `raw_fsa_baseline` | `0.542235` | `0.000000` | `0.7250` | `0` | `0/45/0` | `0.470512` | `0.000000` | `0.7403` | `0` | `0/45/0` |
| `bar_header_smoothing` | `0.542235` | `0.000000` | `0.7250` | `0` | `0/45/0` | `0.470512` | `0.000000` | `0.7403` | `0` | `0/45/0` |
| `voice_rhythm_last_event` | `0.533305` | **`-0.008930`** | `0.6776` | `344` | `1/22/22` | `0.463472` | **`-0.007040`** | `0.6858` | `324` | `2/24/19` |
| `voice_rhythm_single_edit` | `0.530283` | **`-0.011952`** | `0.7035` | `420` | `2/21/22` | `0.463554` | **`-0.006958`** | `0.7138` | `392` | `4/21/20` |
| `combined_header_plus_rhythm` | `0.533305` | **`-0.008930`** | `0.6776` | `344` | `1/22/22` | `0.463472` | **`-0.007040`** | `0.6858` | `324` | `2/24/19` |

### Domain Slices (`pr_*`, `n=31` & `hw_*`, `n=14`)

| Arm | `pr_clean` (d) | `pr_photo` (d) | `hw_clean` (d) | `hw_clean r_rhy` | `hw_photo` (d) | `hw_photo r_rhy` |
|---|---|---|---|---:|---|---:|
| `raw_fsa_baseline` | `0.667540` (`0.0000`) | `0.619288` (`0.0000`) | `0.264775` (`0.0000`) | `0.4800` | `0.141080` (`0.0000`) | `0.4603` |
| `bar_header_smoothing` | `0.667540` (`0.0000`) | `0.619288` (`0.0000`) | `0.264775` (`0.0000`) | `0.4800` | `0.141080` (`0.0000`) | `0.4603` |
| `voice_rhythm_last_event` | `0.656048` (`-0.0115`) | `0.609418` (`-0.0099`) | `0.261517` (`-0.0033`) | `0.5577` | `0.140304` (`-0.0008`) | `0.5681` |
| `voice_rhythm_single_edit` | `0.652859` (`-0.0147`) | `0.608616` (`-0.0107`) | `0.258867` (`-0.0059`) | `0.6188` | `0.142347` (`+0.0013`) | `0.6527` |

---

## Scrutiny & Anomaly Check

1. **Coverage & integrity:** Clean — `90/90` pages scored, `has_pred_raw=90/90`, `0` NaNs, `missing_keys=[]`, `total_edits=2148` across all arms (fixed the `i13-11` 0-edit bug).
2. **`bar_header_smoothing` legitimately fired `0` times:** Even after removing `<sys>` and splitting directly on real barline tokens, `UniOMR` under FSA decoding never produced an isolated 1-bar singleton `clef-*`/`key*` glitch flanked by a `>= 3` consensus. Hypothesis falsified.
3. **Why `voice_rhythm_*` degraded BOTH Note F1 AND printed `r_rhythm`:**
   - On printed scores (`pr_clean r_rhythm 0.8357 -> 0.7317`, `pr_photo 0.8668 -> 0.7390`), naive per-voice bar summing broke **already rhythmically valid (`r_rhythm=1.000`)** pieces (`schumann-op41n1-3`: `r_rhythm 1.000 -> 0.000`, `98` edits, Note F1 `-0.0993`; `handel-lascia`: `r_rhythm 1.000 -> 0.429`, `24` edits, Note F1 `-0.0457`) and pickup/anacrusis measures (`essen-altdeu`: `-0.1000`; `airds-air`: `1.0000 -> 0.9762`).
   - Even on HW where `r_rhythm` rose substantially (`+0.078` to `+0.192`), forcing beat-sum closure by blindly mutating a note's duration **does not recover missed/hallucinated notes** — it corrupts true durations (`hw_clean` `-0.0033` to `-0.0059`, `hw_photo` `-0.0008` to `+0.0013`).
4. **Hardware / platform:** Colab L4 VM stopped cleanly (`colab sessions` confirmed zero lingering `i13-11` VM).

---

## Verdict: **`DISCARD`**

- All active symbolic rhythm-repair arms clearly fail the `PLAN.md` promotion gate, degrading `all45_clean` (`-0.0089` to `-0.0120`), `all45_photo` (`-0.0070`), `pr_clean` (`-0.0115` to `-0.0147`), `pr_photo` (`-0.0099` to `-0.0107`), and `hw_clean` (`-0.0033` to `-0.0059`), while `bar_header_smoothing` makes `0` edits.
- **Falsified hypothesis:** Deterministic Audiveris-style post-hoc duration mutation / header smoothing on top of FSA-decoded `UniOMR` streams is counterproductive; rhythm accuracy must come from model-side learning (or logit-aware decoding) rather than blind post-hoc token rewriting.

## Follow-Up Action / Successor Experiment

- **None queued** (`DISCARD`).
- **Reusable asset preserved:** All `90` `pred_raw` streams at canonical `1280px / 3840` are cached in `evals/2026-11-26T14-30-i13-11b-audiveris-voice-aware-rhythm-repair.json` so any future symbolic/export analysis can run on CPU in `< 5 s` with `0` GPU cost.
