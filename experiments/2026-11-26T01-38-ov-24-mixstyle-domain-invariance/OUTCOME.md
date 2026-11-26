# Outcome — `2026-11-26T01-38-ov-24-mixstyle-domain-invariance`

- **Status:** `COMPLETED` *(with silent `suite31_hw` eval bug — `0/90` pages scored; `proxy_val` `64/64` scored cleanly)*
- **Completed (UTC):** `2026-11-26T11:08:18Z` (started `09:22:55Z`)
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`22.03 GB`, native `BF16`)
- **Elapsed Time:** `6,327.8 s` (`105.5 min`; `~30.5 min` per 800-step arm, `peak_vram=10.03–10.05 GB`)
- **Model & Protocol:** `UniOMR v2 small_45m` (`~43M` params, `MixStyle2d` in Anisotropic FPN), `800` steps/arm from scratch (`seed=42`, `Muon+AdamW`, `lr=3e-4`, WSD, `B=2 × GA=4`, `dim=1280`, `max_seq_len=1536`, `aug_prob=0.50`), trained on **printed-only** `proxy_train` (`1,360` rows) for strict zero-shot `printed → HW` transfer.
- **HF Artifacts:**
  - `evals/2026-11-26T01-38-ov-24-mixstyle-domain-invariance.json` (pulled locally to [2026-11-26T01-38-ov-24-mixstyle-domain-invariance.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T01-38-ov-24-mixstyle-domain-invariance/2026-11-26T01-38-ov-24-mixstyle-domain-invariance.json))
  - `runs/2026-11-26T01-38-ov-24-mixstyle-domain-invariance.json` (pulled locally to [run_2026-11-26T01-38-ov-24-mixstyle-domain-invariance.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T01-38-ov-24-mixstyle-domain-invariance/run_2026-11-26T01-38-ov-24-mixstyle-domain-invariance.json))

## Measured Results (`proxy_val.jsonl`, 64 Pages = 44 Printed In-Domain + 20 HW Zero-Shot, EMA Weights)

| Arm | `mixstyle_prob` | `step/s` | Printed `val_ce` | Printed `SER` | Printed `Token F1` (Δ vs `ms_00`) | Zero-Shot HW `val_ce` | Zero-Shot HW `SER` | Zero-Shot HW `Token F1` (Δ) | `HW − PR` F1 Gap | `suite31_hw` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`ms_00`** | **`0.00`** | `0.437` | `1.6932` | **`0.9577`** | **`0.2234`** (`0.0000`) | `1.6987` | **`0.8967`** | **`0.1799`** (`0.0000`) | `-0.0435` | `0/90` *(bug)* |
| `ms_20` | `0.20` | `0.438` | **`1.6918`** | `0.9875` | `0.1756` (`-0.0477`) | **`1.6945`** | `0.9124` | `0.1455` (`-0.0343`) | `-0.0301`* | `0/90` *(bug)* |
| `ms_40` | `0.40` | `0.436` | `1.6927` | `0.9865` | `0.1807` (`-0.0427`) | `1.6970` | `0.9190` | `0.1327` (`-0.0471`) | `-0.0479` | `0/90` *(bug)* |

*(\*Note: `ms_20`'s smaller `HW − PR` gap `-0.0301` is purely an artifact of printed F1 collapsing `-0.0477` even harder than HW F1's `-0.0343` drop.)*

## Scrutiny & Anomaly Check

1. **CRITICAL SILENT FAILURE — `suite31_hw_cross_domain` scored `0 / 90` pages across all 3 arms:**
   - All 4 `suite31_hw` buckets (`printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`) returned `n_pages: 0` and `0.0` F1, making `winning_arm_by_suite31_hw_f1: "ms_00"` a `0.0 == 0.0 == 0.0` tie-breaker.
   - **Root cause** (`run_colab.py:372-388`): `_eval_suite31_hw` looked up `piece.get("gt_bekern") or piece.get("bekern")` and `piece.get(f"{variant}_image") or piece.get(variant)` and silently `continue`'d when paths/keys did not match `suite31_hw_manifest.json` without asserting `n_scored > 0`. Future scripts MUST assert `n_scored == 90` and reuse canonical manifest loading.
2. **Clear falsification on `proxy_val` (`64 / 64` pages scored cleanly):**
   - `ms_00` (`--mixstyle-prob 0.0`) decisively beats both `ms_20` and `ms_40` on **both** in-domain printed (`0.2234` vs `0.1756`/`0.1807` F1; `0.9577` vs `0.9875`/`0.9865` SER) and **zero-shot HW** (`0.1799` vs `0.1455`/`0.1327` F1; `0.8967` vs `0.9124`/`0.9190` SER).
   - Zero-shot HW F1 degrades **monotonically** with `mixstyle_prob` (`0.1799 → 0.1455 → 0.1327`), while `val_ce` barely moves (`<= 0.004`). Mixing feature-map channel mean/std statistics blurs fine ink/stem/accidental contrast that OMR relies on, hurting both printed and handwritten decoding.
3. **EMA-only eval caveat:** `run_colab.py` evaluated only `uni_omr_v2_ema_latest.pt` (`decay=0.997, update_every=5` over 800 steps → `62%` init weight retained) and skipped raw weights. Because all 3 arms share identical EMA/init/seed/data, the relative defeat of `MixStyle2d` on both domains is clean and unambiguous.

## Verdict: **`DISCARD`**

Per `PLAN.md`'s decision tree (*"If `ms_00` matches or beats both `ms_20` and `ms_40` on both printed and zero-shot HW transfer -> disable `MixStyle2d` (`--mixstyle-prob 0.0`) in Iter-11 `UniOMR v2`"*):
- **Disable `MixStyle2d` (`--mixstyle-prob 0.0`)** in Iter-11 `UniOMR v2` (replacing the unablated `0.30` default). Pixel-level `--augment` + real/synthetic HW data is strictly superior to feature-statistics mixing for OMR.

## Follow-Up Action / Successor Experiment

- Set `--mixstyle-prob 0.0` in Iter-11 `UniOMR v2` configs (no MixStyle follow-up needed).
- **Tooling guardrail:** Whenever a cloud eval script scores `suite31_hw`, enforce `assert total_scored == 90` before writing the eval JSON so manifest-key mismatches fail loudly instead of silently emitting `n_pages: 0`.
