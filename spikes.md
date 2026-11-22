# Spikes — falsifiable ≤1-GPU-day experiments from the bets catalog

Source: [bets.md](bets.md) ranked shortlist. Run AFTER the 100k run
finishes or pauses (solo-sequencing rule: never concurrent with
training). Base checkpoint: best `D:/tmp/t32-100k/sys-hybrid-t32/ckpt-*.pt`.
Eval harness: `tmp-train/mvd/eval_sys_gate.py --hybrid --hybrid-cache
D:/tmp/t32hybrid-full/hybrid_staff_sysdev.jsonl --rectify --suite31`
(read-only GT; new probe filename per spike). venv:
`.\.venv\Scripts\python.exe`. No commits inside spikes; record outcomes
in `progress.md`.

## S1 — Unconsumed-ink EOS veto (zero-train)
Bet bets.md#1. Implement HeatNet foreground-integral EOS mask in
`grammar.py` (+ wire into `measure_decode.py`): forbid EOS while
unread notehead ink mass > 2 noteheads right of the attention peak.
Eval ckpt on gate20 + suite31-photo. PASS: hip-short rollouts drop
≥70% AND photo F1 gain ≥+0.03. Cost ~2 h (1 h code + 1 h eval).
Decision: ship if PASS; else drop (evidence against ink-mass signal).
**Outcome 2026-11-12: DROP.** Implemented in `model.py`
(`count_notehead_peaks`, `ink_eos_veto` in `beam_search`,
`_apply_knobs` mask; threaded through `measure_decode` +
`--ink-veto`), 5/5 unit tests green, full suite 148/148 green, eval
`probe_report_hybrid_t32_S1.json` vs baseline `_100000.json`:
degraded F1 0.362 vs 0.515, clean 0.596 vs 0.794, photo 0.259 vs
0.352. Forcing EOS open over-generates (mean hyp/ref 85/65):
HeatNet peak counts systematically exceed decoded step counts, so
the veto fires on nearly every strip and the tail is garbage.
Do not retune without a peak↔note alignment study first.

## S2 — ILME λ-sweep (zero-train, then short tune)
Bet bets.md shortlist #4. Dual-pass decode (zeroed cross-attention =
ILM estimate), sweep λ ∈ {0.1…0.6} on modifier sub-vocab + EOS.
PASS: modifier F1 (`. # - n [ ]`) +0.03 on suite31-photo. On PASS,
3 h ILM-dropout fine-tune for the exact estimate. Cost ~4 h.
**Outcome 2026-11-12: DROP (missed bar).** Built TDD
(`apply_ilme`, `default_ilme_ids`, `modifier_f1` alignment metric;
test_ilme.py 9/9, suite 157/157). Restricted set adapted — vocab
has no `#/-/n` tokens (pitch = letter + octave digit), so
`ILME_MODIFIERS` = dot/tie/slur/stacc/accent/tenuto/fermata/grace
+ EOS. Full gate (ckpt-100000): λ=0.1 vs λ=0: photo_mod **0.404
vs 0.390 (+0.014, bar was +0.03)**, clean_mod 0.603 vs 0.624
(−0.021), sys-F1 photo 0.346 vs 0.352, clean 0.535 vs 0.539.
Pilots agreed (λ=0.1 weakly +, λ=0.3 cliff: everything DOWN).
Real but sub-bar photo-mod gain that costs clean modifiers and
sys-F1 both domains — the zero-memory ILM estimate is too crude a
prior (load-bearing, not just hallucination). Code kept
(default-off: `ilme_lambda=0`), no ILM-dropout fine-tune.

## S3 — Sibling-margin contrastive fine-tune (10k steps, ~5 h)
Bet shortlist #2. Add 8-sibling beat-cell mutator + margin loss
(weight 0.5), fine-tune best ckpt. PASS: held-out degraded token
accuracy on `{., #, b, n}` +8% absolute. Else drop.
**Outcome 2026-11-12: DROP (vacuous term, run killed at step 350).**
Built TDD (`mutate_modifier_siblings`, `sibling_margin_loss`,
`MvdNet.decode_logits`; test_s3_margin 10/10, suite 167/167,
committed 553f745; smoke caught tuple-seed TypeError +
decode(EOS) ValueError). Calibration: ckpt-100000 teacher-forced
modifier acc on 26 held-out degraded strips = 0.625 (headroom OK).
But the gap diagnostic (/tmp/diag_margin.py, n=48 pairs) shows
natural ref-vs-sibling gaps of min 2.7 / median 20.5 / max 65.8
nats vs margin 1.0 — satisfied from the start (marg 0.0000 steps
50-350). Random corruptions are far too easy; the term cannot move
the model. Lesson: contrastive needs hard negatives (siblings the
model scores HIGH), not random edits. No S3 ckpt promoted
(D:/tmp/t32s3 dead end, killed, GPU freed). Follow-up idea (not
built): mine hardest-of-8 by gap, margin on those.

## S4 — MWER + CE mixed fine-tune (12 h)
Bet shortlist #3. B=4 grammar beams on the fly, d_w with 3× cost on
dropped barlines/premature EOS, 2× on dropped micro-glyphs. PASS:
photo F1 +0.03 AND early-EOS frequency −60%. Else drop.
**Outcome 2026-11-12: DROP (regression).** Built TDD
(`weighted_risk`, `mwer_loss`, `beam_hyp`, `beam_search(return_all)`
with default-path bitwise proof; test_s4_mwer 14/14, suite 181/181,
committed 7dcd164). Key save: best-only beams hid diversity;
`return_all` showed 24/24 distinct beams on ckpt-100000, so the term
was live (smoke mwer −3.2 in loss). Fine-tune D:/tmp/t32s4, 6000
steps, 4 beams/weight 0.1/every-2nd, ckpt-6000.pt on disk, train-eval
TER 0.234. Gate verdict (probe_report_hybrid_t32_S4.json vs S2base):
photo F1 0.276 vs 0.352 (−0.076, bar +0.03), clean 0.458 vs 0.539,
photo_mod 0.375 vs 0.390, degraded sys-F1 0.502 vs 0.515 / SER 0.764
vs 0.701, short rate 0.231 vs 0.231 (bar −60%). Everything down or
flat. Lesson: beam-risk pressure distorted the distribution (cost
weights over-emit structure at content's expense); MWER needs
length/risk normalization + lower lr, not raw mixed CE at full
schedule. No S4 ckpt promoted.

## S5 — Medusa atom-tuple heads (≤1 day)
Bet shortlist #5. Parallel heads emitting chord/note-complexes in one
step; measure steps ratio + F1 delta. PASS: ≥2.5× fewer decoder steps
at F1 neutral-or-better. Else drop (keep AR).
**Outcome 2026-11-12: DROP (calibration falsifies, no build).**
Teacher-forced tuple-continuation accuracy on held-out degraded
strips (/tmp/probe_tuple2.py, ckpt-100000; real bekern order is
duration-pitch-octave-dot, verified against sys-dev refs):
P(letter|duration) **0.421** (748/1776), P(octave|letter) 0.787,
P(dot|octave) 0.640. A Medusa tuple (duration AR, then
letter+octave+dot in parallel) accepts at ~0.42×0.79×0.64 ≈ 0.21
— constant rejection would run SLOWER than AR, before even facing
grammar-masked tree verification. Draft heads predict further
ahead than the main head and can only do worse. Pitch class
genuinely needs per-position image grounding, so no free 2.5×
exists at this sharpness. Keep AR. No code changes. (An earlier
probe revision assumed pitch-first order and was discarded.)

## S6 — MaxSAT confusion-lattice repair (no retrain, ~4 h)
Bet shortlist #6. Post-decode CP-SAT repair over beam lattice: bar
math, spine sync, tie closure. PASS: fixes-per-page −25% on 20
sampled photo pages. Else drop.
**Outcome 2026-11-12: DROP (missed bar, −4%).** Built TDD
(`fallback_repair` + stats, `ties_balanced`, `rank_candidates`
lexicographic splice/strict/tie/logprob, `row_to_ids`/`ids_to_tokens`
split, `repair_beams` rescoring; test_s6_repair 13/13, suite 190/190,
`--lattice-repair` eval flag; committed 7def69e after fixing a real
str-tensor crash the first repair run exposed: rescoring fed token
strings into `torch.tensor`, every strip decode-crashed). Repair eval
on the same 20 pieces (probe_report_hybrid_t32_S6repair.json vs
S6base): fixes median 227.5 vs 237.0 (−4%, bar ≤178), photo F1 0.421
vs 0.409 (+0.012), clean 0.605 vs 0.587, photo_mod 0.435 vs 0.444,
clean_mod 0.621 vs 0.689 (−0.068), degraded sys-F1 0.497 vs 0.515.
Switch rate 0.39 (58/150 strips pick a non-top beam) — the mechanism
is active, but fixes are dominated by content misreads, not
constraint violations, so reranking barely moves fixes and costs
clean modifiers. Lesson: post-decode repair can't fix what the
lattice doesn't contain; the errors are recognition errors.
No promotion (default path unchanged).

## S7 — Fixes-per-page action loss (8 h fine-tune)
Bet shortlist #8. Keystroke-risk + bar-conservation surrogate loss.
PASS: sampled fixes-per-page −20% at F1 neutral-or-better. Else drop.
**Outcome 2026-11-13: DROP (missed bar).** Built TDD
(`keystroke_cost_matrix`, `expected_keystroke_risk`,
`--action-weight/--action-every`; test_s7_action 7/7, suite 201/201;
committed a80da01). Fine-tune D:/tmp/t32s7b 10k steps from ckpt-100000
(weight 0.3, resumed from ckpt-3000 after CPU-collate OOM, finished
step 10000, train-eval TER 0.1916). Full suite31 gate eval
(probe_report_hybrid_t32_S7.json, identical beam-3 strict decode) vs
100k baseline: degraded sys-F1 0.535 vs 0.515 (+0.020), clean sys-F1
0.806 vs 0.794 (+0.012), SER down both — but photo F1 0.330 vs 0.352
(−0.021), photo fixes median 153 vs 144 (+6%), clean F1 mean 0.517 vs
0.539, page fixes 257.5 vs 281.0 (−8%, bar needed −20%). Lesson:
token-level cost shaping moves sys-level likelihood slightly but does
not touch the photo-domain recognition errors that dominate fixes —
same conclusion as S6 from the training side. No promotion.

## S8 — Pitch-shear SpecAugment (10 h fine-tune)
Supporting bet. Latent vertical shear + staff-band + micro-time masks
after CNN block 2. PASS: photo F1 +0.025 over identical-step baseline
continuation. Else drop.
**Build 2026-11-13: GREEN, committed 7d0edc6.** TDD
(`specaugment_latent` pure-fn: per-column integerized shear
zero-fill, fraction bands/slices; `sample_specaug_params` +
`maybe_specaug` on the `seed:step:specaug` stream;
`encode/forward/loss(augment=None)` default legacy-bitwise;
`--specaug-p/--specaug-slope` + fingerprint + resume keys;
test_s8_augment 13/13, suite 214/214). Smoke D:/tmp/s8smoke 60
steps p=1.0: `aug` live every step, loss stepping, TER 0.1986.
Runs: D:/tmp/t32s8 (specaug p=1.0, 10k steps) vs D:/tmp/t32s8base
(identical-step no-aug baseline), both --init-from ckpt-100000.
**Outcome 2026-11-13: DROP (missed bar, sign-negative).** Gate
evals probe_report_hybrid_t32_S8.json vs _S8base.json (identical
beam-3 strict decode): photo F1 0.315 vs 0.327 (−0.012, bar needed
≥ +0.025), photo fixes 178 vs 162 (+10%), photo_mod 0.377 vs 0.401,
clean mean 0.517 vs 0.529; sys-level only: degraded sys-F1 0.484 vs
0.474 (+0.010), page fixes 273 vs 271. Note both continuations
regressed photo vs ckpt-100000 itself (0.352 → 0.327/0.315):
extra 10k steps at full LR from a converged ckpt drifts. Lesson:
latent shear/band/time masks at this strength destroy more
pitch-grounding signal than the robustness they buy — same family
as S6/S7 (recognition errors, not solved by perturbation). No
promotion; augmentation flags stay default-off.

## S9 — Test-time per-strip adapter (no retrain)
Shortlist #11. Rank-8 low-rank residual map on the encoder memory,
fit unsupervised per strip (greedy pseudo-targets → entropy steps,
trunk+decoder frozen). PASS: photo F1 +0.02 over no-adapter on the
same ckpt at ≤25 ms/strip overhead. Else drop.
**Outcome 2026-11-13: DROP (missed bar, over budget).** TDD
(`tt_adapter.StripAdapter` identity-at-init/deterministic/isolated;
`adapt_strip` entropy fit; `beam_search(adapt_steps/adapt_lr/
adapt_rank)`, threaded through measure_decode wrappers +
eval `--adapt-steps/--adapt-lr`; test_s9_adapter 9/9, suite
223/223; committed ef27e29). Probe caught a real dead-init bug:
LoRA-scale A (1e-3) made lr=0.1 a bitwise no-op (identical sha);
unit-gain A (1/sqrt(d)) makes steps=2 move decodes. 5-piece pilot
(ckpt-100000): s0 0.314 → s1 0.318 → s2 0.329 → s3 0.331
(monotone). Full gate probe_report_hybrid_t32_S9.json (steps=3)
vs 100k baseline: photo F1 0.355 vs 0.352 (+0.004, bar +0.02),
photo fixes 153 vs 144 (+6%), degraded sys-F1 0.498 vs 0.515,
clean mean 0.549 vs 0.539 (+0.011), page fixes 269.5 vs 281.0
(−4%). Pilot gain didn't transfer (subset noise). Overhead
~+1 s/strip measured (greedy pseudo + K forwards/backwards) vs
25 ms budget — 40× over. Lesson: unsupervised entropy on weak
pseudo-targets sharpens the wrong transcripts as often as the
right ones; per-strip fitting without a trust signal can't buy
photo F1. No promotion; flags default off.

## Provable queue (re-ranked 2026-11-12: provability over cheapness)

S1–S6 sequenced by falsification cost and all DROPPED. The remaining
queue is ordered by P(success) × goal-impact, cost as tiebreak:
- **S8 SpecAugment** — next after S7. Augmentation-for-robustness is
  the most bankable bet class; 10 h, same gate protocol.
- **S9 Test-time adapter** (shortlist #11): rank-8 per-strip linear
  adapter, <25 ms/strip, absorbs unseen blur/noise with zero
  retraining. Smallest build with a direct photo-F1 path — should
  have run earlier. Spec TBD at build time.
- **S10 Scribble/hand-ink data**: wire `data-gen/hand-ink` variants
  into training, fine-tune, read photo F1. S6 proved the errors are
  recognition errors — matched data is the direct fix.
  **Spec 2026-11-13 (build in progress).** Wiring committed 11fed64
  (list_pairs include_handink + handink fingerprint/resume key,
  4/4, suite 227/227). Data DONE: 4836 scribbly variants (every
  7th clean stem, master seed 777, global per-file seeds,
  pixels bit-identical to a monolithic run), 4776 sidecars via
  source-table rows (49 rectify/width + 11 missing-source dropped
  by hybrid filter). Runs: D:/tmp/t32s10 (83899 pairs) vs control
  D:/tmp/t32s8base/ckpt-10000.pt (S8 no-variant continuation,
  already gate-evaled photo 0.327). PASS: photo F1 +0.02 over
  control. `hand-ink --level
  scribbly` (8 px wobble) on every-7th clean stem of
  mvd-sysdata-t32 (~4.8k stems) with global per-file seeds →
  `<stem>_h0.png` alongside renders (degraded-`_dN` precedent);
  `train.py --handink` pairs them via `list_pairs(include_handink)`
  (clean, degraded, handink order per stem) + `handink` fingerprint/
  resume key; sidecars `<stem>_h0.hybrid.pt` from source-table rows
  (img swapped) via `stage_sidecars` (wobble ≤8 px, w_in exact).
  Protocol (S8-style): 10k-step fine-tune from ckpt-100000 WITH
  variants vs 10k-step control WITHOUT (same runbook otherwise);
  PASS: photo F1 +0.02 over control. Else drop.
  **Outcome 2026-11-13: PASS.** D:/tmp/t32s10 finished 10000/10000.
  Gate eval probe_report_hybrid_t32_S10.json vs S8base control
  (identical decode, no-variant 10k continuation): photo F1 0.356
  vs 0.327 (+0.029, bar +0.02), clean mean 0.550 vs 0.529
  (+0.021), photo_mod 0.421 vs 0.401, photo fixes 153 vs 162
  (−5.6%), degraded sys-F1 0.548 vs 0.474 (+0.074); clean
  sys-F1 0.789 vs 0.800 (−0.011), page fixes 283.5 vs 271.0
  (+4.6%). First PASS of the program — matched scribble data
  moves photo F1 where 9 loss/decode/augmentation spikes could
  not. PROMOTED: hand-ink variants stay in the training mix
  (`--handink` default TBD at S13 build); S16/S13 consume this
  result directly.
- **S11 Staff-indexed gaze** (shortlist #9): per-spine attention
  windows replacing rectify. Weeks, not days — after S10.
- **S12 RNN-T transducer** (shortlist #10): frame-synchronous
  decode, EOS structurally impossible. Biggest build, last.
- Medusa stays DROPPED (quality ceiling is AR parity by
  construction; an ensemble-of-heads variant would be a new bet,
  not a revival).
- **S13 Full-corpus scale retrain, bundled** (new 2026-11-13,
  highest EV; agy consult 2026-11-13 agrees but corrects the
  spec): 79k pairs → full big500 with v2.4+S10 stack, from
  scratch — BUT NOT same-architecture (agy: weeks of GPU on the
  unanchored 8×16 grid leaves a resolution ceiling). Bundle in:
  (a) 4×8 trunk stride (one feature row per diatonic step);
  (b) dense ink/keypoint auxiliary loss (1000× gradient bits to
  early layers); (c) cross-attention (x,y) coordinate supervision
  vs engraver symboxes. PASS: photo F1 +0.10 over ckpt-100000.
  Weeks of GPU; the main event.
  **S13 verdict FAIL at bar (2026-11-15, gate
  `probe_report_hybrid_t32s13_20000.json`, identical manifest/decode/
  quarantine to S10): suite31 strict photo F1 0.0325 vs S10 0.356
  (bar ≥0.456 missed by 0.42); photo_mod 0.0941, clean 0.0401,
  fixes 304, gate2_pass=False. Cause is step budget, NOT the
  bundle: executed 20000 steps = 2500 updates (bs1/acc8 matched
  S10's fine-tune update count, 1/10th of the 100k-step plan) from
  scratch, and at matched data volume the bundle learns NORMALLY
  (degraded_page_f1 0.061 vs v2b-2500's 0.071; loss 5.71→2.39,
  aux 0.70→0.37, TER 0.88→0.57, EOS learned). Verdict on the
  bundle itself is UNPROVEN either way — needs phase 2
  (→100k steps via --init-from) for a fair test.**
- **S16 Real print-and-shoot pairs** (new 2026-11-13, agy):
  print 200–500 corpus pages, photograph with 2–3 phones under
  desk lighting, auto-align via homography+TPS. Even 5% real
  captures close the synthetic→real optical gap photo-degrade
  misses. Cheap (days), feeds S13. Before S13.
  **S16b screen-shoot variant (2026-11-14, user rejected the print
  cost): same hypothesis, NO printer + ONE phone — photograph the
  `s16b_slideshow.html` fullscreen display (~100 pages, 3 lightings,
  ~1 h). Same PASS bar (+0.02 photo F1 over control). If it fails,
  S16 is marked DROP and S13 runs synthetic-only.**
  **S16b-sim verdict DROP (2026-11-14, gate
  `probe_report_hybrid_t32_S16sim.json`, identical manifest/decode/
  quarantine to S10): simulated screen captures add NOTHING —
  photo F1 0.3506 vs S10 control 0.356 (−0.006, bar +0.02 missed);
  clean −0.002, photo_mod +0.007 (noise), fixes 152 vs 153,
  degraded sys-F1 −0.035. The simulator faithfully reproduces
  photometrics but not the sensor/lens statistics that matter:
  only REAL captures can test the spike now. Real-capture arm
  (one phone hour) stays open; otherwise S13 runs synthetic-only.
  **S16 marked DROP entirely 2026-11-14 per user decision (no real
  captures will be taken). S13 proceeds synthetic-only.**
- **S14 Trunk capacity swap** (new 2026-11-13): ConvNeXt-Tiny is
  small for photo blur; swap Base (or frozen SigLIP-2 + finetune
  per the NCM consult) with the same decoder, retrain. PASS:
  photo F1 +0.05 over an identical-data Tiny retrain. After S13
  (needs the big corpus to separate capacity from data).
- **S15 Piece-level priors rescore** (new 2026-11-13): learned
  bekern LM over whole pieces (key/time continuity, voice
  alignment, cross-strip bar math) rescoring beam lattices —
  the human-transcriber global priors our strip-local reader
  lacks. Decode-side like S6 but with learned priors instead of
  constraints. PASS: fixes −15% at F1 neutral-or-better. No
  retrain; after S13. (Agy ranks this lowest-EV right now:
  priors only arbitrate candidates already in the beam, and S2/S6
  showed the beam is the problem. Keep, but late.)
- **S17 Measure-local render-and-compare verifier** (new
  2026-11-13, agy: worth-it): test-time only — project each
  decoded measure's notes back onto rectified notehead/ink maps;
  where prediction and ink disagree, re-decode that measure with
  visual-prior constraints (or pick the min-residual beam hyp).
  No retrain. After S13.
- **S18 Factored event-tuple heads** (new 2026-11-13, agy,
  parked): replace flat bekern strings with per-event tuples
  (delta_time, voice, diatonic_step, accidental, duration,
  dot/tie bitmask) so a missed dot can't derail the stream.
  Vocab/grammar rewrite + retrain — biggest change on the board,
  after S13–S17 prove out. (Agy: VLMs and diffusion are hype
  for staff notation — avoid both; exact rational-clock
  enforcement belongs online per-measure, which S17 covers.)

## Spike hygiene
- One spike at a time on the single GPU; training paused via `--resume`
  afterwards (proven mechanism).
- Each spike gets a dated `progress.md` entry: command, cost, numbers,
  PASS/DROP + evidence. A DROP is a result — record it.
- Promote at most 2 winners to `todo.md` phase tasks; the rest stay spikes.
