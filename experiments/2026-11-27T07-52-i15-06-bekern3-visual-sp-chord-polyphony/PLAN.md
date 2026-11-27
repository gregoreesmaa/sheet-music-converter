# PLAN — `2026-11-27T07-52-i15-06-bekern3-visual-sp-chord-polyphony` (`i15-06`)

## 1. Platform Chosen & Rationale

- **Platform:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, BF16 native; also Modal L4 compatible with automatic FP16 fallback on `cc < 8`).
- **Estimated Wall-Clock & Cost:** **~65 min (~4.8 CU)**.
  - Phase 0 (Deterministic MusicXML Round-Trip Oracle on all 45 `suite31_hw` pieces + corpus compression audit on `proxy_train`/`proxy_val`): ~1 min (CPU).
  - `arm0_bekern_v2a` (`1,200` optimizer steps on `UniOMRv2Net(small_45m)`, `grad_accum=4`, `MTP-4`, BF16, EMA `0.99` + `suite31_hw` & MTP-4 throughput eval): ~33 min.
  - `arm1_bekern3_core` (`1,200` matched steps on `UniOMRv2Net(small_45m)`, `grad_accum=4`, `MTP-4`, BF16, EMA `0.99` + `suite31_hw` & MTP-4 throughput eval): ~31 min (faster due to `>= 30%` shorter target sequences).
- **Baseline Commit SHA:** `e426c28`.

---

## 2. Identifiers for Follow-Up Queries

- **Queue Entry:** `experiments/2026-11-27T07-52-i15-06-bekern3-visual-sp-chord-polyphony/queue_entry.json` (`id: "2026-11-27T07-52-i15-06-bekern3-visual-sp-chord-polyphony"`, `short_id: "i15-06"`, `wave: 13`, `priority: 6`, `provider: "colab_l4"`, `gpu: "L4"`).
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T07-52-i15-06-bekern3-visual-sp-chord-polyphony.json`
  - Run Ledger: `runs/2026-11-27T07-52-i15-06-bekern3-visual-sp-chord-polyphony.json`

---

## 3. Required Training & Eval Data

- **Training / Validation Corpus:** `datasets/iter10_5_proxy_2k.tar` (`proxy_train.jsonl` + `proxy_val.jsonl`, 100% permissive, `0` `CC-BY-NC`, `0` eval overlap).
- **OOD Evaluation Benchmark:** `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw_manifest.json`: 31 printed + 14 real historical handwritten manuscripts, `clean` + `photo` = 90 pages + MusicXML ground truth).
- **Architecture:** `UniOMRv2Net(UniOMRv2Config.small_45m())` from `tmp-train/mvd/uni_omr_v2.py` (`~43M` params, GQA `8Q/2KV`, DeepStack, `mtp_k=4`).

---

## 4. Formalized Hypotheses & 2-Arm Protocol

### Architect 1's 3 Hypotheses (`bekern3_core`)
1. **H1 — Visual Staff-Position + Printed-Only Accidentals + Deterministic `Ratio<i64>` Resolver:**
   - Replace sounding `<STEP> <OCT> [a±1|2]` with core visual staff-relative positions `sp-14..sp+20` (35 tokens, `sp+0` = bottom staff line; lossless `sp-oct-dn` / `sp-oct-up` ±7 diatonic escape prefixes for extreme cross-staff ledger lines) plus 5 printed-only accidental tokens (`acc-2`, `acc-1`, `acc0`, `acc+1`, `acc+2`) emitted strictly when a note's sounding alteration differs from the active `(key_signature, in_measure)` state.
   - Reconstruct sounding pitch and exact measure onset times at decode time via a zero-parameter deterministic `Ratio<i64>` (`fractions.Fraction`) pitch/accidental/beat resolver.
2. **H2 — Duration-First Implicit Chords + Fused Dotted Durations:**
   - Fuse base duration + augmentation dots into zero-lag tokens (`breve..128th`, `<dur>.`, `<dur>..`) emitted *before* noteheads.
   - Represent same-duration chords implicitly as a single `<DUR_FUSED>` followed by consecutive `sp<k> [acc-*]` noteheads (eliminating both `chord` and repeated per-tone durations; retaining explicit `chord <DUR_FUSED>` only as a rare mixed-duration lossless fallback).
3. **H3 — Descending-`mean(sp)` Polyphonic `<v>` + `skip <DUR>` & Rational `d0..d9 /` Tuplets/Meters:**
   - Canonicalize multi-voice measures by sorting voice segments in **descending `mean(sp)`** (highest visual voice first), separated by `<v>`, with `skip <DUR_FUSED>` for invisible voice spacers, eliminating arbitrary `voice-1..voice-27` IDs.
   - Replace all closed-set `time-N-D` and `tuplet-start-A-N`/`tuplet-stop` tokens with compositional rational grammar `time <d0..d9>+ / <d0..d9>+` and `tup <d0..d9>+ / <d0..d9>+ ... tup-end` (`0` OOVs on any meter/tuplet ratio, exact `Fraction(actual, normal)` arithmetic).

### Matched 2-Arm Ablation (`1,200` Steps / Arm)
- **`arm0_bekern_v2a` (Control):**
  - Standard `bekern+v2` Phase A vocabulary (`vocab_size = uni_omr.VOCAB_SIZE = 265`).
  - `UniOMRv2Net(small_45m)`, `1,200` optimizer steps, `grad_accum = 4`, `lr = 3e-4` (warmup `80` steps + cosine decay to `3e-5`), `weight_decay = 0.01`, `mtp_loss_weight = 0.30`, `GPU_EMA_DECAY = 0.99` updated every step (`0.99^1200 = 5.8e-6`, zero EMA lag), `FULLPAGE_MAX_DIM = 1280`, `MAX_SEQ_LEN = 1536`, `SEED = 42`.
- **`arm1_bekern3_core` (Treatment):**
  - Exact matched `UniOMRv2Net(small_45m)` architecture, seed, optimizer, LR schedule, MTP-4 loss weight, EMA decay, and `1,200` steps — differing **solely** in `bekern3_core` tokenization (`vocab_size ~ 183`) and deterministic `bekern3_core -> bekern_v2a` resolution before stitching & scoring.
- **Scrutiny Discipline:** Both **RAW** and **EMA** checkpoints are evaluated on `proxy_val` (and the better/EMA checkpoint on full 45-piece `suite31_hw` + MTP-4 throughput benchmark) so any EMA lag is immediately visible.

---

## 5. Exact Launch Command

```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\overnight_queue\run_queue_colab.py --only 2026-11-27T07-52-i15-06-bekern3-visual-sp-chord-polyphony
```

Standalone direct execution (on Colab L4 / Modal L4 with `HF_TOKEN` or `/tmp/.hf_token`):
```bash
python3 -u experiments/2026-11-27T07-52-i15-06-bekern3-visual-sp-chord-polyphony/run_colab.py
```

---

## 6. 5-Minute Checkpointing & Resume Plan

- Every `5.0` minutes (`SAVE_EVERY_MINUTES = 5.0`), each arm writes an atomic checkpoint (`ckpt_latest.pt.tmp` → `os.replace` → `ckpt_latest.pt`) containing `{step, model, ema, opt, rng_state}` under `<vol>/out/2026-11-27T07-52-i15-06-bekern3-visual-sp-chord-polyphony/<arm>/`.
- Completed arms write an atomic `arm_done.json` and immediately push an interim `runs/<EXP_ID>.json` heartbeat to HF Hub so preemption never loses a completed arm.
- Re-running the exact same command automatically resumes from the last saved step or skips completed arms.

---

## 7. Evaluation Gates & Decision Tree

### Three Primary Gates
1. **Gate A — Deterministic MusicXML Round-Trip Oracle on `suite31_hw`:** `>= 99.5%` exact note-tuple F1 (`(dur, dots, step, octave, alter)`) across all 45 `suite31_hw` MusicXML ground-truth scores.
2. **Gate B — Exact Token-Length Reduction:** `>= 30.0%` mean token-length reduction (`bekern3_core` vs `bekern_v2a`) on `proxy_train`, `proxy_val`, and `suite31_hw` GT.
3. **Gate C — Held-Out OOD Accuracy & MTP-4 Throughput:**
   - **`suite31_hw` Handwritten (`HWM`) Note F1:** `>= +3.0 pp` (`+0.0300`) over `arm0_bekern_v2a`, **OR**
   - **`suite31_hw` Printed Note F1:** `>= -0.5 pp` (`-0.0050`) vs `arm0_bekern_v2a` **AND** MTP-4 speculative decoding effective **`notes/sec` gain `>= +30.0%`**.

### Decision Tree
- **PROMOTE:** Gates A, B, and C all pass → promote `bekern3_core` compiler + deterministic `Ratio<i64>` resolver into Rust (`mscz-rs`/`bekern`) and Python (`tmp-train/mvd/bekern3.py`) as the canonical Iter 15+ tokenization format.
- **ITERATE:** Gates A and B pass (`>= 99.5%` oracle, `>= 30%` compression), and either (i) `arm1` wins on `HWM` / C-clef / chord slices but loses slightly on clef-header misreads (amplifying visual→semantic drift), or (ii) MTP-4 throughput gains `< +30%` due to acceptance-rate drop on implicit chords → queue `i15-06b` pairing `bekern3_core` with cross-system clef/key consensus or explicit chord-count hint.
- **DISCARD:** Gate A fails (`< 99.5%` round-trip oracle after fixing compiler bugs), OR Gate B fails (`< 30%` compression), OR `arm1` regresses `> 1.5 pp` on printed Note F1 while failing to gain on `HWM` or MTP-4 throughput.
