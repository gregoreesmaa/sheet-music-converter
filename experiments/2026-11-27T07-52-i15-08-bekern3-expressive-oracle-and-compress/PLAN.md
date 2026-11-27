# PLAN — `2026-11-27T07-52-i15-08-bekern3-expressive-oracle-and-compress`

## 1. Platform Chosen & Rationale

- **Platform:** Google Colab `L4` (`provider: "colab_l4"`, Wave 13, BF16 native, 24 GB VRAM; also Modal L4 compatible).
- **Rationale:** Combines a fast CPU-side 125-page `OMR-NorthStar` round-trip Oracle verification (`part_a_oracle_closure`, ~2 min) with a controlled 3-arm `600`-step MTP-4 proxy bake-off (`part_b_compression_mtp_bakeoff`, `UniOMRv2Net(small_45m)`, `grad_accum=4`, `lr=3e-4`, `gpu_ema_decay=0.99`, `mtp_loss_weight=0.30`, `max_dim=1280`) + `suite31_hw` accuracy & wall-clock MTP-4 decode benchmark.
- **Estimated Wall-Clock & Quota:** **~60 min** (~**`4.5` CU**).

## 2. Identifiers for Follow-Up Queries

- **Queue Staging Entry:** [queue_entry.json](queue_entry.json) (`id: "2026-11-27T07-52-i15-08-bekern3-expressive-oracle-and-compress"`, `short_id: "i15-08"`, `wave: 13`, `provider: "colab_l4"`, `gpu: "L4"`).
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T07-52-i15-08-bekern3-expressive-oracle-and-compress.json`
  - Run ledger: `runs/2026-11-27T07-52-i15-08-bekern3-expressive-oracle-and-compress.json`
- **Live Cloud IDs (filled at launch):**
  - Colab session / notebook ID: `TBD`

## 3. Required Data & Code

- **Training / Val Data (Part B):** `datasets/iter10_5_proxy_2k.tar` (permissive corpus, 0 `CC-BY-NC`, 0 eval overlap).
- **OOD Eval Data (Part B):** `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw_manifest.json`, clean + photo = 90 pages).
- **Oracle Eval Data (Part A):** `datasets/northstar_eval_pack.tar` (125-page `OMR-NorthStar` manifest when present on HF/volume; falls back cleanly to `suite31_hw` MusicXML ground truths if the pack tar is absent, recording `"scorer_mode"` explicitly in the eval JSON).
- **Architecture:** `UniOMRv2Net(small_45m)` (`mtp_k=4`, GQA `8Q/2KV`, DeepStack) from `tmp-train/mvd/uni_omr_v2.py` (commit `e426c28`).

## 4. Exact Two-Part Protocol & 3 Hypotheses

Executed via `experiments/overnight_queue/run_queue_colab.py` or standalone:

```bash
python -u experiments/2026-11-27T07-52-i15-08-bekern3-expressive-oracle-and-compress/run_colab.py
```

### Part A — `part_a_oracle_closure` (Hypothesis 1: Expressive/Text Oracle Closure)
Runs lossless MusicXML → `bekern3` → MusicXML round-trip over the 125-page `OMR-NorthStar` corpus and scores `(NS, S_note, S_struct, S_expr, S_play, CFR)` with the following `bekern3` extensions enabled:
- **Structural / Pickup / Barlines:** `<pickup>` (anacrusis/implicit measure marker), `bar-dashed` (alongside `bar`, `bar-double`, `bar-final`, `bar-repeat-*`).
- **Deferred Text & Semantic Anchors:** `<sys_lyr>` (system-level lyric lines), `<harm>` (harmony/chord symbols), `<tempo>` (tempo words & metronome marks), `<reh>` (rehearsal marks).
- **Numbered Overlapping Spanners:** `slur-1-start/stop`..`slur-3-start/stop`, `hairpin-1-cresc/dim/stop`..`hairpin-2-cresc/dim/stop`, `ottava-15ma`, `ottava-15mb` (plus `ottava-8va`/`8vb`/`stop`).
- **Expanded Articulations (`+18` `art-*`):** `art-staccato`, `art-staccatissimo`, `art-accent`, `art-strong-accent`, `art-tenuto`, `art-detached-legato`, `art-spiccato`, `art-scoop`, `art-plop`, `art-doit`, `art-falloff`, `art-breath-mark`, `art-caesura`, `art-stress`, `art-unstress`, `art-fermata`, `art-trill`, `art-mordent`, `art-turn`, `art-arpeggiate`, `art-glissando`, `art-tremolo`, `art-harmonic`, `art-up-bow`, `art-down-bow`, `art-open-string`, `art-snap-pizzicato`.
- **Expanded Dynamics (`+7` `dyn-*` + `dyn-other`):** `dyn-ppp`, `dyn-pp`, `dyn-p`, `dyn-mp`, `dyn-mf`, `dyn-f`, `dyn-ff`, `dyn-fff`, `dyn-fp`, `dyn-sf`, `dyn-sfz`, `dyn-sffz`, `dyn-rf`, `dyn-rfz`, `dyn-fz`, `dyn-n`, `dyn-other`.

**Part A Gate:** Oracle **`NS >= 98.0`** AND **`S_expr >= 0.980`** (and `CFR == 0.0`).

### Part B — `part_b_compression_mtp_bakeoff` (Hypotheses 2 & 3: Compression + MTP-4 Speedup)
Controlled 3-arm `600`-step bake-off (`UniOMRv2Net(small_45m)`, `mtp_k=4`, `return_mtp=True`, `mtp_loss_weight=0.30`, `grad_accum=4`, `lr=3e-4`, `warmup=60`, `gpu_ema_decay=0.99`, BF16):

1. **`arm0_bekern_v2a` (Control):** Standard `bekern_v2a` representation (`vocab_size = uni_omr.VOCAB_SIZE`), trained with CE + MTP-4 loss (`0.30`).
2. **`arm1_profile_a_comp` (H2 — `-20.2%` target tokens):** Lossless Profile-A transform:
   - Zero-lag fused dotted durations (`quarter.`, `eighth.`, `half.`, `whole.`, `16th.`, `32nd.`, plus double-dot `dur..`)
   - Drop redundant `dur[.]` on `chord` notes matching chord root
   - Sticky duration within bar/voice (omit unchanged `dur[.]` on consecutive pitched notes; resets on bar/system/header/voice/staff boundaries)
   - Implicit `voice-1` (omit `voice-1` at bar/system start when voice is already 1)
   - Fused `rest-<dur>` / `rest-<dur>.` tokens (collapsing `rest <dur> [dot]*` into 1 token)
3. **`arm2_profile_a_plus_core_pitch` (H3 — `-41.1%` target tokens):** Profile A + `C2..B6` core diatonic pitch fusion (`C2`, `D2`, ..., `B6` = 35 single-token core pitches) with exact compositional `<step> <oct>` fallback for octaves `< 2` or `> 6`.

Startup verifies `100%` lossless round-trip on `proxy_val` for both Arm 1 and Arm 2. At evaluation, each arm runs **both** standard KV-cached greedy AR decode and **MTP-4 speculative draft+verify decode** (drafting `1 + 3` tokens per step from `logits` + `mtp_logits[:3]` and verifying with KV-cache rollback) over `suite31_hw` and `proxy_val`, recording exact Strict Note F1, Rhythm F1, compression ratio, `T > 3840` truncation rate, MTP-4 mean accepted tokens/step, and wall-clock `ms/page` speedup.

**Part B Gate:** **`>= 1.45×`** wall-clock MTP-4 decode speedup vs `arm0_bekern_v2a` with **`>= -0.2 pp`** Strict Note F1 parity on `suite31_hw`.

## 5. 5-Minute Checkpointing & Resume Plan

- Atomic checkpointing (`.tmp` → `os.replace`) every `5.0` minutes (`SAVE_EVERY_MINUTES = 5.0`) to `<vol>/out/2026-11-27T07-52-i15-08-bekern3-expressive-oracle-and-compress/<arm>/ckpt_latest.pt` (storing `step`, `model`, `ema`, `opt`).
- Completed stages write atomic sentinels (`part_a_oracle_done.json`, `<arm>/arm_done.json`) so re-launching `run_colab.py` after preemption resumes cleanly from the last completed step/arm.
- Final `evals/<EXP_ID>.json` and `runs/<EXP_ID>.json` are uploaded to HF Hub (`gregoreesmaa/sheet-music-converter-omr`).

## 6. Evaluation Gate & Decision Tree

- **PROMOTE:**
  - Part A hits `NS >= 98.0` AND `S_expr >= 0.980` → freeze `bekern3` expressive/text token spec into `mscz-rs/bekern` and Iter-15 compiler.
  - Part B winning compression arm hits `>= 1.45×` wall-clock MTP-4 decode speedup AND `suite31_hw` Strict Note F1 `>= arm0 - 0.002` (`>= -0.2 pp`) → promote winning compression profile into Iter-15 tokenizer & MTP-4 runtime.
- **ITERATE:**
  - Part A `NS < 98.0` or `S_expr < 0.980` → inspect worst NorthStar round-trip pages in `part_a_oracle_closure.worst_pages` and queue targeted compiler patch.
  - Part B `arm1_profile_a_comp` preserves Note F1 (`>= -0.2 pp`) while `arm2_profile_a_plus_core_pitch` regresses Note F1 `> 0.2 pp` (or misses `1.45×` speedup due to lower MTP-4 acceptance on fused pitches) → promote Profile A and queue follow-up testing accidental-aware pitch fusion or MTP-4 temperature/weight tuning.
- **DISCARD:**
  - Both compression arms regress `suite31_hw` Strict Note F1 by `> 1.0 pp` vs `arm0_bekern_v2a` or fail to beat `1.15×` wall-clock speedup.
