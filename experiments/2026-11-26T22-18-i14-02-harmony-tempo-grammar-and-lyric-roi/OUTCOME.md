# OUTCOME — `2026-11-26T22-18-i14-02-harmony-tempo-grammar-and-lyric-roi`

- **Status:** `COMPLETED` (`rc=0`)
- **Started / Finished (UTC):** `2026-11-27T06:34:39Z` → `2026-11-27T07:20:49Z`
- **Platform / GPU:** Google Colab `NVIDIA L4` (re-routed from planned Modal L4)
- **Elapsed Time:** `2769.72 s` (`46.2 min` inner script; `2888.3 s` wrapper wall time)
- **HF Artifacts (pulled locally):**
  - Eval JSON: `evals/2026-11-26T22-18-i14-02-harmony-tempo-grammar-and-lyric-roi.json` → [`eval_hf.json`](eval_hf.json)
  - Run Ledger: `runs/2026-11-26T22-18-i14-02-harmony-tempo-grammar-and-lyric-roi.json` → [`run_ledger_hf.json`](run_ledger_hf.json)
- **Verdict:** **`ITERATE`** (**DO NOT wire into Iter 15 as-is**)

---

## 1. Measured Results (Arm vs Arm, 600 Steps/Arm on `iter10_5_proxy_2k`)

| Metric | `control_base_v2` (`vocab=237`) | `safe_b_harmony_tempo_roi_ctc` (`vocab=285`) | Delta (Safe B − Control) | `PLAN.md` Gate | Status |
|---|---:|---:|---:|---|---|
| Steps completed | `600` | `600` | `0` | `600` | PASS |
| Elapsed time (`s`) | `1373.61` | `1355.28` | `-18.33 s` (`-1.33%`) | `< +8%` latency | PASS |
| Peak VRAM (`MB`) | `8970.1` | `8970.6` | `+0.5 MB` | `< 14 GB` | PASS |
| Final 25-step train loss | `2.5290` | `2.5202` | `-0.0088` | parity | PASS |
| `val_token_f1` | `0.235758` | `0.313256` | `+0.077498` | — | PASS* |
| `val_ser` | `0.906359` | `0.873082` | `-0.033277` | — | PASS* |
| `ema_val_token_f1` | `0.154458` | `0.240520` | `+0.086062` | `>= -0.005` | PASS* |
| `ema_val_ser` | `0.967953` | `0.896171` | `-0.071782` | — | PASS* |
| Music `total_params` | `41,356,299` | `41,393,163` | `+36,864` (`+0.089%`) | — | PASS |
| Music `fp16_mb` | `81.030` | `81.103` | `+0.074 MB` | — | PASS |
| Music `hybrid_fp16_int4_with_scales_mb` | `31.283` | `31.330` | `+0.047 MB` | — | PASS |
| `lyric_head_params` | — | `723,564` | `+723,564` | — | PASS |
| Combined extra FP16 size (`MB`) | `0.000` | `1.454` (`0.074` + `1.380`) | `+1.454 MB` | `< 3.0 MB` | PASS |
| `lyric_head_int4_mb` | — | `0.388` | `+0.388 MB` | — | PASS |
| `lyric_ctc_final_loss` (`300` steps, `8.39 s`) | — | `3.1458` | — | — | **FAIL** |
| **`lyric_ctc_cer`** (`128` val RoIs) | — | **`0.8883`** | — | **`<= 0.15`** | **FAIL** |
| **`lyric_ctc_exact_match`** | — | **`0.0000`** (`0/128`) | — | — | **FAIL** |

*\*See Anomaly #2 & #3 below: the `+0.0861` `ema_val_token_f1` jump is confounded by unused vocab slots and 600-step from-scratch init variance, not a genuine gain from harmony/tempo supervision.*

---

## 2. Scrutiny & Anomaly Check

1. **Silent Coverage / Execution Integrity:**
   - Both arms ran all `600` steps with zero NaNs and valid JSON outputs (`eval_hf.json`, `run_ledger_hf.json`).
   - Platform shifted cleanly from planned Modal L4 to Colab L4 (`NVIDIA L4`), same GPU class and BF16 precision.

2. **CRITICAL ANOMALY #1 — `LyricRoICTCHead` failed to converge (`CER = 0.8883`, `exact_match = 0.0`) AND was never trained jointly:**
   - Despite `PLAN.md` specifying joint training (`loss_total = loss_music + 0.15 * loss_lyric_ctc`) on `UniOMRv2`'s `P3 (/8)` backbone feature map, [`run_colab.py`](run_colab.py#L235-L293) executed `_train_and_eval_lyric_head()` as a **standalone 8.39 s post-hoc script** on raw `(1, 32, 160)` grayscale PIL crops, completely disconnected from `UniOMRv2Net`.
   - It achieved `lyric_ctc_final_loss = 3.1458` (barely below blank-collapse on a 96-class alphabet, `ln(96) = 4.56`), **`lyric_ctc_cer = 0.8883`** (failing the `<= 0.15` PROMOTE gate by `+0.7383`), and **`0.0` exact match** even on in-vocabulary `SAMPLE_LYRICS` words.
   - **Root causes in `run_colab.py`:**
     1. **Severely under-trained:** Only `300` AdamW steps from scratch for a 2-layer BiGRU + 3-conv CNN over `96` multilingual CTC classes (CTC with BiGRU typically needs `2,000–3,000` steps to exit blank-collapse).
     2. **Tiny 10 px bitmap font (`ImageFont.load_default()`):** Rendered into a `160×32` strip, then downsampled `stride=(8, 4)` (`T=40`) and collapsed vertically via `AdaptiveAvgPool2d((1, None))` — averaging ~10 px of glyphs with ~22 px of white margin and leaving ~1.5 timesteps per narrow character (violating CTC's `>= 2` frames/char requirement for repeated characters).

3. **CRITICAL ANOMALY #2 — The `+48` harmony/tempo vocab slots received ZERO supervision:**
   - [`run_colab.py`](run_colab.py#L367-L371) only patched `cfg.vocab_size += 48` (`237 → 285`) while training on unmodified `iter10_5_proxy_2k` `bekern` targets (token IDs `0..236`).
   - No harmony, metronome, rehearsal, or `<lyric-box>` tokens were present in the training or validation targets. Thus Arm 2 only verified the **memory/compute footprint** of `+48` extra output slots (`+36,864` params, `+0.074 MB` FP16, `+0.047 MB` hybrid INT4, `+0.5 MB` VRAM, `-1.33%` latency), NOT multi-task harmony/tempo emission accuracy or interference.

4. **CRITICAL ANOMALY #3 — Explaining the `+0.0861` `ema_val_token_f1` delta (`0.2405` vs `0.1545`):**
   - Both arms converged to nearly identical train loss (`2.5290` vs `2.5202`, delta `-0.0088`) on identical data with `SEED=42`.
   - Changing `vocab_size` (`237 → 285`) shifts the parameter-initialization RNG sequence for `embed`/`lm_head`, and at `600` steps from scratch on `1280px` full pages (`loss ~2.52`, `SER ~0.90–0.97`), greedy autoregressive decoding sits on a steep phase-transition cliff where minor EOS/init differences cause large `token_f1` variance. Do **not** misattribute this `+0.0861` jump to harmony/tempo grammar.

---

## 3. Verdict & Iter 15 Recommendation

- **Verdict: `ITERATE`** (fails `PLAN.md` PROMOTE gate `lyric_ctc_cer <= 0.15` at `0.8883`; joint training and harmony/tempo supervision were not actually exercised).
- **Iter 15 Wiring Decision: DO NOT wire any `i14-02` code changes into Iter 15.**
  - **What is validated for future iterations:** Expanding `UniOMRv2` vocab by `+48` compositional slots (`237 → 285`) is effectively free (`+0.074 MB` FP16, `+0.047 MB` hybrid INT4, `+0.5 MB` VRAM, zero latency overhead) — safe to adopt **once** real `<harmony>` / `<metronome>` / `<rehearsal>` targets are extracted from OpenScore Lieder / PDMX MusicXML into the training manifest.
  - **What needs iteration before promotion:** `LyricRoICTCHead` must be fixed and actually wired to `UniOMRv2` backbone features (`P2/P3` RoIAlign) with real/realistic TrueType score-font RoIs and sufficient CTC training steps.

---

## 4. Follow-Up Action / Successor Experiment

Recommended successor (`i14-02b-lyric-roi-ctc-and-real-harmony-targets`, to be queued by the orchestrator when prioritizing lyric/harmony support):
1. **Real Harmony/Tempo Targets:** Extract real `<harmony>` chord symbols and `<metronome>` BPM/beat-unit tokens from OpenScore Lieder + PDMX MusicXML into `bekern+` (`+48` slots) so the compositional grammar is actually supervised and evaluated on chord/tempo F1.
2. **Fix `LyricRoICTCHead`:**
   - Render lyric RoIs with real serif/italic TrueType fonts (`18–24 px` glyph height on `48×256` RoI, tight vertical crop) + photo degradation jitter.
   - Train for `2,500+` steps with OneCycleLR (or tap `P2 (/4)` lateral features + joint `loss = loss_music + 0.15 * loss_lyric_ctc` via `RoIAlign` on actual lyric bounding boxes).
   - Gate on `lyric_ctc_cer <= 0.15` on **unseen OOD words** (disjoint train/val lexicon) AND `harmony_f1 >= 0.70` with music `ema_val_token_f1` delta `>= -0.005`.
