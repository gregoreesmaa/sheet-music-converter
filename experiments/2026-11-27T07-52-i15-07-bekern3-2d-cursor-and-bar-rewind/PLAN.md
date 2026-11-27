# PLAN — `2026-11-27T07-52-i15-07-bekern3-2d-cursor-and-bar-rewind` (`i15-07`)

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, native BF16) via `experiments/overnight_queue/run_queue_colab.py` (also Modal L4 compatible; automatic FP16 fallback on `cc < 8`).
- **Estimated Wall-Clock & Cost:** ~65 min (~4.8 CU).
  - `arm0_bekern_v2a` (`1,200` steps on `iter10_5_proxy_2k`, `UniOMRv2Net(small_45m)`, `grad_accum=4`, BF16, EMA `0.99` every step): ~24 min.
  - `arm1_bekern3_spatial` (`1,200` steps with 2D `@y/@x` spatial cursors, CE-masked `@y?`/`@x?`, and `<part>/<s1..s12>/<pickup>/bar-split` topology tokens): ~25 min.
  - Full `suite31_hw` (90 pages) + `proxy_val` evaluation across 3 decode configs (`arm0_bekern_v2a`, `arm1_bekern3_spatial_fsa`, `arm1_bekern3_spatial_kv_rewind`, plus RAW vs EMA sanity check): ~16 min.
- **Baseline Commit SHA:** `e426c28`.

## 2. Identifiers for Follow-Up Queries
- **Queue Entry:** `experiments/2026-11-27T07-52-i15-07-bekern3-2d-cursor-and-bar-rewind/queue_entry.json` (`id: "2026-11-27T07-52-i15-07-bekern3-2d-cursor-and-bar-rewind"`, `short_id: "i15-07"`, `wave: 13`, `provider: "colab_l4"`, `gpu: "L4"`).
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T07-52-i15-07-bekern3-2d-cursor-and-bar-rewind.json`
  - Run Ledger: `runs/2026-11-27T07-52-i15-07-bekern3-2d-cursor-and-bar-rewind.json`

## 3. Required Training & Eval Data (100% Permissive)
- **Training / Proxy Val:** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`proxy_train.jsonl` + `proxy_val.jsonl`; zero `CC-BY-NC`, zero `suite31_hw` overlap).
- **OOD Evaluation:** `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (`suite31_hw_manifest.json`: 31 printed + 14 real historical handwritten pieces, `clean` + `photo` = 90 pages).
- **Model Backbone:** `UniOMRv2Net(UniOMRv2Config.small_45m())` trained from scratch for controlled 1,200-step A/B attribution (`seed=42`, `LR=3e-4`, `warmup=100`, cosine decay, `grad_accum=4`, `MAX_SEQ_LEN=1536`, `FULLPAGE_MAX_DIM=1280`, `GPU_EMA_DECAY=0.99` updated every step so `0.99^1200 = 5.8e-6` — zero EMA lag).

## 4. Experimental Protocol (Architect 2's 3 Hypotheses)

### Vocabulary Extension (Append-Only on `arm1_bekern3_spatial`)
Appended strictly after `uni_omr.VOCAB_SIZE` so base token IDs never shift:
- **Vertical system cursor (65 tokens):** `@y00`..`@y63` + `@y?` (train-only unknown vertical cursor).
- **Horizontal barline cursor (65 tokens):** `@x00`..`@x63` + `@x?` (train-only unknown horizontal cursor).
- **Macro-topology & measure boundary markers (15 tokens):** `<part>`, `<s1>`..`<s12>`, `<pickup>`, `bar-split`.

### Hypothesis 1 — 2D Spatial Cursor & Strict 2D Monotonic FSA
- **Target Synthesis:**
  - Vertical `@y` bins: 1D horizontal ink-projection / StaffUNet2D-compatible band peak finder producing strictly monotonic `0 <= y_0 < ... < y_{S-1} < 64` aligned to the page's `<sys>` count. On noisy/low-contrast handwritten images where ink peaks cannot be confidently matched, emits `@y?`.
  - Horizontal `@x` bins: Vertical ink-column projection inside each system's `[y_lo, y_hi]` band detecting tall barline-x peaks, aligned strictly monotonically (`0 <= x_0 < ... < x_{B-1} < 64`) to the `bar` (`BarClose`) tokens within that system. On noisy/ambiguous HW bands, emits `@x?`.
- **Train-Only CE Masking:** `@y?` and `@x?` are kept in `tgt_in` (context continuity) but masked to `ignore_index = -100` in `tgt_out` so cross-entropy never rewards predicting `?`.
- **Strict 2D Monotonic FSA (Eval):**
  - After `<bos>` or `<sys>`, allow ONLY `@y_k` with `k > prev_y` (or `<eos>` if `prev_y >= 63`). Reset `prev_x = -1`.
  - After `bar` (`BarClose`), allow ONLY `@x_k` with `k > prev_x` (or `<sys>`/`<eos>` if `prev_x >= 63`).
  - Everywhere else, mask all `@y*` and `@x*` tokens to `-inf`. Always mask `@y?` and `@x?` to `-inf` at eval.
  - **Zero post-hoc n-gram repetition hacks** are applied in `arm1` — loop elimination must come purely from the 2D monotonic cursor FSA.

### Hypothesis 2 — Macro-Topology & Exact `PPQ=40320` Tick Accounting
- Exact integer tick engine at `PPQ = 40320` (`quarter = 40320`, `whole = 161280`, `half = 80640`, `eighth = 20160`, `16th = 10080`, `32nd = 5040`, `64th = 2520`, `breve = 322560`, exact integer dotted sums and `tuplet-start-A-N` scaling `ticks * N // A`).
- Training encoder prepends `<part> <s1>..<sK>` per system and tags anacrusis measures with `<pickup>` and mid-bar system-break continuations with `bar-split`.

### Hypothesis 3 — Eval-Time Local Single-Bar KV-Cache Rewind (`beam=4`)
- During `arm1_bekern3_spatial_kv_rewind` decoding, snapshot `(kv_len, fsa_state, out_len, meter_state)` at the start of every measure (immediately after `@y_k` or `@x_k`).
- Track the current measure's exact `PPQ=40320` tick sum. When `bar` (`BarClose`) is emitted, check `bar_ticks == expected_bar_ticks` (exempting valid `<pickup>` on measure 0, `bar-split`, and whole-bar `bar-rest`).
- On beat-sum mismatch (capped at `MAX_REWINDS_PER_PAGE = 12` to bound tail latency), slice each decoder layer's self-attention KV cache `(k[:, :, :L, :], v[:, :, :L, :])` back to `bar_start_kv_len` while reusing the precomputed cross-attention K/V (`0` encoder recomputation), and run a local `beam=4` search (`MAX_BAR_TOKENS = 96`) under the strict 2D FSA until `bar`/`<sys>`/`<eos>`.
- Accept the highest-log-prob beat-valid bar candidate (or retain greedy if none is beat-valid) and splice its KV cache back into the main stream.

### Arms Evaluated
1. `arm0_bekern_v2a`: Baseline `bekern` v2a + `<sys>`, standard greedy decode (no spatial cursors, no FSA, no rewind).
2. `arm1_bekern3_spatial_fsa`: `bekern3` trained EMA weights decoded with strict 2D monotonic FSA only (isolates Hypotheses 1 + 2).
3. `arm1_bekern3_spatial_kv_rewind`: Same `bekern3` EMA weights + strict 2D monotonic FSA + local single-bar KV-cache rewind (`beam=4`, isolates Hypothesis 3).

## 5. Exact Launch Command
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\overnight_queue\run_queue_colab.py --only 2026-11-27T07-52-i15-07-bekern3-2d-cursor-and-bar-rewind
```

## 6. 5-Minute Checkpointing & Resume Plan
- Atomic checkpoint save (`ckpt_latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm (`model`, `ema`, `opt`, `step`, `vocab_size`), resuming automatically from `step` if preempted.
- Completed training (`train_done.json`) and evaluation (`arm_done.json`) artifacts are atomically cached per arm so completed work is never recomputed.
- On completion, uploads `evals/2026-11-27T07-52-i15-07-bekern3-2d-cursor-and-bar-rewind.json` and `runs/2026-11-27T07-52-i15-07-bekern3-2d-cursor-and-bar-rewind.json` to HF `gregoreesmaa/sheet-music-converter-omr`.

## 7. Evaluation Gate & Decision Tree
- **Primary Metrics:**
  - `loop_truncation_count` & `loop_truncation_rate` (target `0%` without post-hoc n-gram hacks)
  - `sys_count_exact` (exact system-count match rate)
  - Targeted pieces: `ryans-jig` Note F1, `mozart-k387` Note F1, `aloha-oe` cutaway & pickup `S_play` (beat-valid playable bar fraction) + Note F1
  - Rewind telemetry: `pct_bars_rewound`, `rewind_beat_fix_rate`, and `latency_overhead_pct` (hard ceiling `<= +12%`)
  - Global OOD quality: `suite31_hw` `overall`, `pr_clean`, `pr_photo`, `hw_clean`, `hw_photo` Strict Note F1 + `proxy_val` token F1 (RAW vs EMA verified).
- **PROMOTE:** `arm1_bekern3_spatial_kv_rewind` achieves `loop_truncation_rate == 0.0%` (or `>= 80%` reduction vs `arm0`), `sys_count_exact` improves `>= +5.0 pp`, `suite31_hw` overall Note F1 improves `>= +1.5 pp` (with positive gains on `ryans-jig` / `mozart-k387` / `aloha-oe` `S_play`), AND `latency_overhead_pct <= +12.0%` → promote `bekern3` 2D cursor + macro-topology + local bar KV-rewind into Iter 16 full-scale training & export.
- **ITERATE:** `arm1_bekern3_spatial_fsa` eliminates loops and improves `sys_count_exact`, but either (a) `@x` barline peak synthesis is too noisy on HW (`@x?` rate > 40%), or (b) KV-rewind latency exceeds `+12%` or under-fixes beat mismatches → spawn `i15-07b` replacing ink-projection `@x` with StaffUNet2D barline-heatmaps or tightening rewind trigger entropy thresholds.
- **DISCARD:** `arm1` regresses `suite31_hw` Note F1 by `> 1.0 pp` vs `arm0_bekern_v2a` (token overhead hurts sequence budget) without eliminating copy loops → keep `bekern` v2a and test 1D `<y-bin>`-only cursor (`i14-16`) instead.
