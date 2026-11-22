# Corpus 10x Design — breadth over depth, flat training budget

Date: 2026-11-18. Status: DRAFT, awaiting user approval (hard gate — no
downloads/renders until approved).

## Intent and why

Worst-photo mining (2026-11-18) proved our failures are context-transfer,
not token absence: staccato/dots/slurs/rests/repeats all exist in the
10.6M-token mix yet drop to zero on folk/clarinet pages, while a Lieder
control with identical degradation scores 0.62. The corpus is 1,207
pieces of 19th-century classical (Lieder + string quartets) × 6 fonts —
deep, not wide. Ten-x the pieces across new sources and styles while
holding GPU-days flat, so Gate A and the strata kill-gates are fed by
breadth instead of repetition.

## Goals

- ~12,000 distinct pieces (≈10x), new strata: folk/regional (Irish/Scottish
  ABC), lead sheets, choral (C-clesfs, lyrics), solo-piano polyphony,
  woodwind+transposing, mid-piece meter/key changes.
- Total pair budget flat at ~85–100k (≈185 GB today, cap v2 at 200 GB):
  multiplicity per piece drops ~10x to pay for breadth.
- All sources permissive (CC0/PD/traditional) — the NC question stays
  parked (approach B) unless the user opts in.
- Every pair keeps the deterministic contract (seeded SplitMix64 streams,
  tmp+rename, resume, SHA-256 manifest) and a stratum tag for per-stratum
  floors on Gates B/C/D.

## Non-goals / anti-goals

- No NC-SA data in the main mix (CVC-MUSCIMA/HOMUS stay out).
- No rare-ornament vocab work (agy: <0.002 F1, deferred to Gate C).
- No resolution increase (agy: misdiagnosed; VRAM-prohibitive).
- Failure = 10x pairs instead of 10x pieces (that just repeats Lieder
  louder and blows the GPU budget the user explicitly capped).

## Constraints

- Format rule is convertibility, not nativeness (user decision 2026-11-18):
  - Tier 1 (direct, zero conversion): MusicXML/MEI/ABC/MSCX natives —
    PDMX, Session.org, Nottingham, OpenScore extensions, OpenEWLD PD.
  - Tier 2 (one proven CLI step): Humdrum/KernScores (.krn) → MEI via a
    verovio binary install (shell-out only, no linking — LGPL-safe;
    neither verovio nor lilypond is currently installed). Spike first:
    convert a sample, validate MEI, then bulk.
  - Tier 3 (parked): Mutopia (.ly) — needs a Python toolchain
    (music21/python-ly; Python not installed) for LOW marginal diversity
    over Tier-1 classical. Revisit only if Tier 1 leaves a classical
    hole.
  - OUT: MIDI-derived (quantization destroys engraving ground truth —
    not "reasonable" conversion), Wikifonia-full (license), NC corpora
    (license, parked under B).
- Training cost per run must not grow: pair budget flat ⇒ per-piece
  multiplicity falls ~10x (1–2 fonts sampled round-robin per piece, not
  the full 6-font cross; clean + 1 degraded + handink subset only).
- D: 366 GB free; v2 capped at 200 GB; user files untouched.

## Approaches considered

- **A. Curated-permissive 10x (RECOMMENDED, default).** PDMX v8
  mxl.tar.gz (254,035 engraved MusicXML, CC-0, VERIFIED on disk) sampled
  from the rated_deduplicated frame (13,187) with stratum quotas +
  OpenEWLD (502 PD lead sheets, VERIFIED) + OpenScore extensions +
  expanded synthetic hand-ink. (CORRECTION 2026-11-18: the
  "symbolic-only" verdict applied solely to the v5 artifact; v8 carries
  the real scores — user was right.) Session.org (LLM-training ban),
  Nottingham (©), Wikifonia-full (expired), BEAT (no license),
  CVC-MUSCIMA/HOMUS + NORBECK ABC (all 72 files carry per-file NC
  notices — audited 2026-11-18, quarantined) EXCLUDED with skip
  manifests. Folk coverage now depends on PDMX content-mining
  (single-staff + 6/8 + repeat-bar heuristics), NOT on a folk-native
  source — the folk gap is re-opened as a sampling problem.
- **Multiplicity (user decision 2026-11-18, flat GPU budget).** Default
  x1 degraded per strip (was x3); x3 kept ONLY as targeted oversampling
  on photo-collapse strata (sparse lead sheets, folk). Fonts sampled
  1-2 per piece round-robin (was full 6-cross). Hand-ink on a sampled
  fraction (clean always survives alongside). Pair budget stays ~85k.
- **Decode freeze during Gate A.** Agy's voice-cap + HeatNet EOS veto
  are queued AFTER the 30k verdict — changing decode constraints
  mid-run would muddy Gate A comparability. No eval/train-code changes
  until the binding number lands (ruling 2026-11-18).
- **B. A + real handwritten (CVC-MUSCIMA/HOMUS).** Best handwriting
  realism; model inherits CC BY-NC-SA non-commercial-only terms.
  PARKED — needs explicit user opt-in (asked 2026-11-18, unanswered).
- **C. Synthetic-only 10x.** 10x vary/degrade/hand-ink on the same 1,207
  scores. REJECTED: same music underneath, fails the broadening goal,
  and the user explicitly capped training-time growth.

## What the user already decided

- 10x the corpus by broadening sources/styles (2026-11-18).
- Flat training budget: reduce augmentation as pieces grow (2026-11-18).
- Handwritten additions wanted (bonus) — synthetic path under A;
  real-NC path parked under B.
- Decode-first fixes (agy voice-cap, HeatNet EOS veto) land before the
  strata build pays off fully — separate code track, not this doc.

## Left to the builder

Source-by-source fetch+license receipts, ABC/MusicXML ingest smoke
tests per source, font-sampling + multiplicity schedule, stratum tag
taxonomy, render-farm throughput measurement, DVC layout, and the
10k-probe-style clean≥0.67 validation gate before full mix-in.

## Pipeline answers (user review 2026-11-18, verified in code)

- Fonts: score-render ALREADY varies fonts (`--musescore-fonts A,B` +
  text fonts, full cross product, recorded in JSON) and notation-vary
  ALSO samples `--fonts`. For 10x: sample 1–2 fonts per piece
  round-robin instead of the full cross (multiplicity control).
- Hand-ink skip: NO probability knob exists (`--resume` only skips
  already-done files). Clean variants always survive alongside (hand-ink
  ADDS suffixed copies), so "chance not to run" = run it on a sampled
  input subset; optionally add a `--fraction` flag (tiny change).
  DECISION 2026-11-19 (user: don't wobblify everything): hand-ink subset
  capped at ≤20% of composed files, biased to single-staff folk/dance
  strata where handwritten sources actually exist (MUSCIMA-style); the
  other ≥80% stay clean engraved. Wobble is a minority accent, never the
  default look.
- Vary-after-render GAP (real): notation-vary consumes render's
  ground-truth JSON and emits `.aug.json` + `pairs.jsonl`, but NOTHING
  consumes those (verified zero references; orchestrator stages are
  placeholders; the t32 corpus bypassed vary entirely). Close the loop
  for 10x: vary must emit MusicXML (new) → re-render via score-render,
  or vary moves pre-render operating on scores.
- Strip-before-degrade CONFIRMED (already so): build_sys_data cuts
  strips from CLEAN renders first (symbox truth boxes, RowNet fallback —
  predictable layout), THEN photo-degrades the strips ×3. Full-page
  degraded10 sets are eval-side only, never training strips.
- Real-page stripping engine EXISTS: RowNet v2 ONNX
  (`neural_detect_staves`) + `group_page` + pad boxes, shared by the
  train-strip fallback and the full eval path (plus the Nov-16
  density-shed/text-discard repair rules).
- Representativeness: enforced by stratum quotas at sample time +
  per-stratum floors on Gates B/C/D — not by hoping the pool averages
  out (the current pool proves hope fails: 100% Lieder/quartet).

## Approval

User approval of THIS doc authorizes downloads + renders. Reply with
approved/changes.

APPROVED 2026-11-18: approach A (user: "option A is fine"). NC track
stays parked pending the legal read below.

## Legal read: NC training data and the model (informational, NOT legal
advice — counsel decides before any product use)

Q1 — Does NC training data truly make the MODEL NC? No court has ruled
directly on whether weights are an Adaptation of training data, so the
law is unsettled. The conservative compliance read (BLOOM/BigScience
practice) is YES, treat NC-trained weights as NC-encumbered: CC's own
position is that commercial use of a model trained on NC works violates
the NC term if weights count as an adaptation, and CC BY-NC-SA adds a
ShareAlike viral clause on top. Rights holders (MUSCIMA authors) offered
research-only terms; commercial use invites a claim even if you might
ultimately win it. Verdict for this project: NC data in ⇒ NC model out
of the commercial path, unless counsel says otherwise.

Q2 — Two versions (NC model + clean model, NC omitted from clean)? YES,
this is the standard compliant shape, with one strict rule: provenance
must be total. Separate runs from separate manifests; the clean run must
NEVER --init-from (or otherwise load) an NC-trained checkpoint, or the
taint carries over. Shared code/pipeline is fine (code isn't data).
Implement as license-per-score tags in the corpus manifest + a pack-time
filter for the commercial build. Cost: 2x training GPU for two full
runs, plus manifest discipline. Feasible, no legal creativity needed.

Q3 — Can the NC model serve a commercial setting as an open-source
addition? NO. Open-sourcing does not cure commercial USE: NC forbids use
"primarily intended for or directed toward commercial advantage",
including internal commercial R&D and shipping it inside a commercial
product, whether or not you charge for the model itself. What IS allowed:
research/eval use, academic publication, and releasing the NC weights
for others' non-commercial research (with BY attribution + SA terms).
Compliant product shape: ship the CLEAN model commercially, publish the
NC model separately as a research artifact.

Practical consequence for the 10x: approach A stays NC-free; IF the user
later opts into B, it runs as a parallel fork (own out-dirs, own
manifests, own checkpoints, never merged into the commercial line).
