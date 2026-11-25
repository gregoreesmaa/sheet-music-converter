---
name: learn-from-suite31
description: >-
  Mandatory post-training and post-evaluation learning loop for
  sheet-music-converter (train -> learn -> change -> repeat). Trigger
  IMMEDIATELY whenever ANY OMR training run finishes (to automatically run the
  45-piece suite31_hw OOD benchmark first) OR whenever a suite31_hw benchmark
  finishes: compare apples-to-apples Strict Note F1, spawn 6 parallel read-only
  CPU eyeball scouts on worst vs best printed and handwritten pieces, run
  controlled ablations, diagnose sim-to-real / router / grammar / data gaps,
  implement & test fixes, commit & push, and only then launch the next GPU run.
---

# Learn From `suite31_hw` — Mandatory Post-Training Learning Loop

**Trigger:** Activate this skill **automatically** the moment ANY OMR training run finishes (starting at Step 0) OR whenever a `suite31_hw` benchmark finishes (starting at Step 1). Never launch a new training run without completing this loop + committing & pushing first.

## Core Principles (Standing Rules)

1. **Strict Musical `Note F1` is the only real scoreboard.** Never let flattering in-distribution val `Tok F1` mask OOD `suite31_hw` `Note F1`. Always report all 4 slices (`printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`) side-by-side against the previous iteration and our best 2-stage baseline.
2. **Visually eyeball actual PNGs (`read_file`) — worst vs best.** Numbers alone hide the real bugs (inverted routers, black border rings, upside-down crops, giant watercolor paintings, rare time signatures, jazz boxes on Bach).
3. **Verify hypotheses with 5-second CPU ablations — never guess.** Load the checkpoint on CPU (`CUDA_VISIBLE_DEVICES=""`) to measure exact `ink_recall`, border token theft, `max_keep_ratio` sweeps, and token-to-note waterfalls (`Tok F1 → PitchBag → Pitch+Dur → Strict Note F1`).
4. **Keep the model learned, not heuristic.** Prefer fixing data, augmentation, router supervision, loss, and formal `bekern` FSA grammar over brittle inference-time pixel hacks.
5. **Protect frozen evals & GPU.** Never touch `suite31_hw_manifest.json` or `gate20/test-8`; verify `0` eval overlap on any new training data; keep all forensic scouts strictly read-only and CPU-only.

---

## Step 0 — Run `suite31_hw` Immediately After Training Finishes

> [!IMPORTANT]
> Always evaluate on **`suite31_hw`** (`suite31_hw_manifest.json`) — our **modified 45-piece Suite31 benchmark with 14 real historical handwritten manuscript additions** on top of the 31 printed pieces (`45 × clean/photo = 90 pages`), NOT the legacy 31-piece printed-only `suite31_manifest.json`.

The moment a training run completes on GPU (do NOT pause to ask the user):
1. Run the full 45-piece (`90`-page) `suite31_hw` OOD benchmark on `uni_omr_best.pt` (and `uni_omr_soup.pt` if souped):
   ```powershell
   .\.venv\Scripts\python.exe -u eval/eval_uni_suite31.py `
     --ckpt <RUN_DIR>\uni_omr_best.pt `
     --max-dim 1280 --max-len 3072 --use-grammar --use-mtp-spec `
     --device cuda --bf16 --out D:\tmp\uni-omr-suite31-ood\<RUN_TAG>_fsa_mtp.json
   ```
   *(Use `tmp-train/mvd/eval_uni_suite31.py` if running prior to the `eval/` migration.)*
2. Proceed straight to **Step 1** as soon as the JSON report is written.

---

## Step 1 — Apples-to-Apples Scoreboard

Present a compact table comparing **Previous Iter → Current Iter → Best 2-Stage** on all 4 `suite31_hw` slices (`printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`) with:
- `Note F1`, `Tok F1`, `DecOK %`, `Tok/s`, `MTP Acc`.
- Top 5 standout individual pieces (`Note F1`) and worst 5 failure pieces.
- Call out any clean-vs-photo paradox (`photo > clean` or `clean >> photo`) and any decode/truncation crashes (`FAIL(bekern-decode-rc-1)` or hitting `max_len`).

---

## Step 2 — Spawn 6 Parallel Read-Only CPU Forensic Eyeball Scouts (`3 Printed + 3 HW`)

Always split **Non-HW Printed** (`pieces 1..31`) and **Real Historical HW** (`pieces 32..45`) into separate dedicated read-only CPU subagents (`typeName: "self"`, explicit constraint: *STRICTLY READ-ONLY, ZERO FILE EDITS, CPU-only*):

### Non-HW Printed (3 Scouts)
1. **Printed Scout 1 — Worst vs Best `printed_clean` Eyeball:**
   - Rank all 31 printed pieces by `clean` Note F1; visually inspect (`read_file`) at least 4 worst + 3 best PNGs.
   - Compare predicted vs GT `bekern`, `<sys>` count, staff count, clefs, keys, time signatures, and barlines.
2. **Printed Scout 2 — `photo` vs `clean` Divergences & Router Check:**
   - Inspect top 4 `photo > clean` and top 4 `clean > photo` pieces; eyeball PNG pairs.
   - Run checkpoint `InkTokenRouter` on CPU to measure `ink_recall`, interior recall, border-token %, and `max_keep_ratio` saturation.
3. **Printed Scout 3 — Remaining `Tok F1 → Note F1` Waterfall & Grammar Audit:**
   - Break down by staff count (`1, 2, 3, 4, 5+`).
   - Solve every high-`Tok F1` / low-`Note F1` mystery (`>0.85 Tok F1`, `<0.50 Note F1`).
   - Quantify early `<eos>`, onset/barline drift, clef/key/meter errors, and FSA `strict` vs non-strict behavior.

### Real Historical HW (3 Scouts)
4. **HW Scout 1 — Worst vs Best Historical HW Eyeball & Loop Anatomy:**
   - Rank all 14 HW pieces (`pieces 32–45`); visually inspect 3 best vs 4 worst PNGs.
   - Inspect exact decoded `bekern` tokens on looping/failing pieces (`sys=1`, `max_len`, `rc=1`) to expose grammar loopholes or OOD collapse.
5. **HW Scout 2 — Router & Visual Domain Autopsy on Real HW (CPU):**
   - Run checkpoint `InkTokenRouter` on CPU across 6 `suite31_hw` pages (`clean` + `photo`) vs synthetic HW vs composed real HW.
   - Measure `local_ink`, `saliency_logits`, `PgRec`, border/colorbar/sticky-note theft %, and test higher `max_keep_ratio` / contrast ablations.
6. **HW Scout 3 — Real HW Exposure, Data Inventory & Curriculum Audit:**
   - Compute exact % of training gradient steps spent on real historical HW vs synthetic HW.
   - Audit unused real HW assets on disk (`0` eval overlap), OOV tokens, and visual/label gaps vs `suite31_hw`.

---

## Step 3 — Conditional Deep-Dive Scouts (Spawn When Triggered)

Spawn additional read-only 3-scout panels whenever the eyeball reports expose a deeper systemic blocker:
- **Sim-to-Real Gap Panel (3 scouts):** When synthetic val is high (`>0.70` F1) while real OOD stalls (`<0.15` Note F1) → audit (1) visual/glyph physics, (2) macro-layout & label parity, (3) ML shortcut-learning & transfer theory.
- **Training Speed Panel (3 scouts):** When GPU util is low or step time regresses → audit (1) DataLoader/aug/CPU-PCIe, (2) GPU kernels/SDPA/optimizer, (3) algorithmic FLOPs/bucketing/eval overhead.
- **Design/Vocab Decision Panel (3 scouts):** Before breaking checkpoint compatibility (e.g. vocab/OOV or arch changes) → audit (1) exact corpus census, (2) ML/sequence modeling impact, (3) red-team engineering-ROI & blast radius.

---

## Step 4 — Synthesize Ranked Action Plan, Implement, Commit & Push

1. **Synthesize all scout reports into one ranked root-cause table** (Bugs → Router/Arch → Data/Aug → Grammar/Decode → Vocab), citing exact measured numbers and expected `Note F1` gain.
2. **Proceed autonomously** (do not pause for routine confirmation unless hitting a genuinely irreversible or ambiguous fork).
3. **Implement fixes in parallel non-overlapping CPU subagents** with mandatory unit tests (`test_uni_omr.py`, `test_uni_decode.py`, `bekern-decode` round-trip, `0` eval overlap, and a 5-second GPU VRAM/speed preflight).
4. **Update `progress.md` / `todo.md`, commit, and push** (using `Gregor Eesmaa <6941817+gregoreesmaa@users.noreply.github.com>`) **before** launching the next GPU training run.
