# Bets v2 — best directions after the evidence (2026-11-22)

Fuses the file-backed workflow's 10 ranked levers (`D:/tmp/omr-trimmed/*.md`,
5 agents + synthesis, all `complete`) with the spike verdicts (S1–S10, S13,
S16, Gate-A) and the 5 agy program consults. Objective: the 10x-Sheetly
program in [goal.md](goal.md) (Gates A–D; photo strict-F1 ≥ 0.9973,
OMR-NED ≤ 0.0185, fixes median 0). Numbers below are measured unless marked.

## Scoreboard (current state)

| System | Clean strict-F1 | Photo strict-F1 | OMR-NED |
| :--- | :--- | :--- | :--- |
| Ours (ckpt-22000, Gate-A final) | 0.5709 | 0.3332 | — |
| Ours sys-dev degraded sys-F1 | 0.7664 clean / 0.5168 degr | — | — |
| Sheetly (vendor benchmark) | 97.37 | 97.34 | ~0.19 |
| Audiveris 5.11 | 88.43 | 50.43 | 0.216 / 0.538 |
| oemer | 38.45 | 9.81 | 0.734 / 0.944 |
| SMT-NeXt (single-system) | (3.9/5.1/13.1 ideal CER/SER/LER) | (5.3/6.2/13.5 camera) | — |
| SMT++-NeXt (full-page) | 5.6 / 6.9 / 12.9 | — | — |

## What the evidence killed (do not spend here)

- Decoder/loss/rerank shaping without recognition change: S1 EOS veto (−0.09),
  S2 ILME (+0.014, sub-bar), S3 margin (vacuous, gaps 20 nats vs margin 1.0),
  S4 MWER (−0.076), S6 MaxSAT (−4%, bar −25%), S7 action loss (photo −0.021),
  S8 SpecAugment (−0.012), S9 adapter (+0.004, 40x over budget). S5 Medusa
  unfalsifiable-by-build (joint-accept ~0.21). Pattern: all beams make the
  SAME systematic misreads — priors and reranks cannot fix an empty beam.
- Post-decode grammar guards (−0.1126, scorer-blind: strict-F1 ignores rests
  and attachments), text-only music-LLM priors (polarity inversion: the prior
  backs fluent-but-wrong normalizations), theory-lean spelling work (zero
  misspellings in 87 strips — spelling rides free), pre-Gate-B photo-geometry
  (clean geometry already 31/31 + 99.75%), hand-ink expansion, notation-vary
  emitter (lossy), S12 RNN-T now, S14 trunk swap now, T4.1 render-and-diff now.
- Scaling alone (FP7: 10x needs ~100M micro-steps ≈ 4.4 GPU-years).
- S13-cold FAIL (0.0325, step budget — never cold-start a bundle again);
  S16 print path DROPPED (user + sim-gate FAIL); S18-before-corpus ordering
  (agy: hygiene demands the Route-A probe first or cold-head confounds data).
- **Forking Audiveris / augmenting Audiveris's training data / neural-restorer →
  Audiveris (CLOSED 2026-11-22, 4-expert review + user lock):** ~95% of
  Audiveris is hard-coded classical CV & heuristics (`BINARY`/`SCALE`/`GRID`/
  `LINKS`/`RHYTHMS`), so training data cannot fix its 38-pt photo cliff
  (`88.43% → 50.43%`). Even with a neural front-end it hits an 88.43% clean
  assembly ceiling + 1-bit bottleneck + AGPL-3.0 network copyleft. Keep
  strictly as a Docker black-box benchmark (`X1`).

## Execution lock (2026-11-22, user decision)

Target: autonomous, permissive-licensed (MIT/BSD/Apache) phone-photo → `.mscz`
neural Rust pipeline beating Sheetly. Order: **`Route-A` 10k probe (ruthless
kill gate) → `I1` label-normalization audit → `S18` factored heads**. Beat
Audiveris first (Gate B: `> 88.43%` clean / `> 50.43%` photo) before chasing
10× Sheetly; validate Route-A + S18 on the local RTX 4080 first, then rent
burst H100s post-gate; treat Gate C (`~0.98–0.99`) as a shippable v1 milestone
ahead of Gate D (`0.9973`).

## Ranked directions v2

1. **Label-normalization audit (NEW, todo I1).** Transcoda: removing target
   normalization raises OMR-NED 18.71→82.51 (+63.8 pts, sequence collapse) —
   the biggest single published lever, and we have never audited bekern for
   duplicate surface forms. Zero-train, CPU-only, queued behind Route-A.
2. **Corpus mix + fonts + tok-dropout 0.15 (IN PROGRAM, Route-A probe).**
   agy S5: +0.18–0.22, the biggest remaining lever (in-domain 0.757 vs
   OOD-21 0.482 proves starvation). Engraver-diversity (+MuseScore→MuseJazz/
   Leland/Emmentaler) and degradation-trained clean→photo hold (Sheetly
   97.37→97.34) are the same bet from two sides. Kill-gated 10k probe with
   OOD-21 ≥ 0.36 + plasticity tripwires.
3. **Factored event tuples S18 (IN PROGRAM, step 4).** TrOMR's decoupled
   rhythm/pitch/accidental heads + consistency loss collapsed merge-SER
   0.294→0.005 — the direct template for our voice-explosion + rest-flood +
   spray census (voice net −6% is strict-invisible; hypotheses carry 259
   spray marks vs 7 in refs). Build order stands: beam-5/masking → visual
   probe with KILL tripwires → tuples → measure-atomic last.
4. **Keep 2D positional encoding + fine-grained tokenization (HAVE).**
   2D-PE SER 6.9 vs 1D 15.57 vs none 78.4; bekern-style splitting −29.6% vs
   raw kern. Our ConvNeXt-Tiny AR reader already matches this recipe —
   do not regress it (no 1D collapse before rectification, per gate rule).
5. **Real-page structural density over raster noise (VALIDATES STRATA).**
   Removing score-concatenation +14.34 OMR-NED vs raster degradation +11.19
   vs semantic aug +13.23: the quota strata (piano/folk/choral/meter/jazz
   with per-stratum floors) outrank further photo-degrade tuning. Do not
   touch the degrade recipe before Gate B.
6. **Photo/ageing augmentation as data, not domain adaptation (HAVE).**
   Tuggener: 36.0→73.3% AP vs 48.9% for adversarial adaptation. Our
   photo-degrade suite is the mechanism; Greve-style fresh-draws-per-epoch
   is the scaling direction post-Route-A.
7. **Cheap-affine-stack gap check (NEW, todo I2).** Zeus closes synth→scan
   full-TEDn 44.4→18.4% with 9 affine/photometric ops. Zero-train test on
   frozen ckpt-22000 before touching any recipe.
8. **Frozen pretrained vision + scale (HAVE + DIRECTION).** Legato's frozen
   Llama-3.2-Vision beats SMT++ OOD; our timm trunk transfer is the same
   bet at 1/60th the size. Trunk swap (S14) stays parked until Gate C
   stalls — capacity is not the bottleneck (train TER 0.058, flat).
9. **XML-native linearization as a LATER target, not now (DEFERRED).**
   Zeus LMX eliminates the SMT failure mode (92.9%/88.8% of SMT++ outputs
   unconvertible to MusicXML). Our bekern+ → MusicXML path converts today
   (31/31 oracle 1.000), so this is a Gate-D-adjacent option, never a
   retrain trigger.
10. **System→page curriculum with decaying synth mix (FUTURE).** SMT++
    P 0.9→0.2 over 200k steps is the closest published analog of our track.
    Relevant only after the reader clears Gate B on systems — never as a
    substitute for the mix.

## Traps (load-bearing)

- Hand-ruled assembly past the split point: oemer's deskew→`unit_size`→
  geometry chain posts photo OMR-NED 0.944 (≈ total failure). Every
  deskew/staff-line heuristic stays proposal-only; decoding stays learned
  end-to-end. This is the standing verdict behind killing post-decode guards.
- kern-style outputs that cannot convert to notation score ~0 on notation-F1
  no matter their SER. Our ruler (notation-as-written, B#≠C) already prices
  this; never trade convertibility for token SER.
- A fluent-but-wrong note is fluent by definition under a text prior: priors
  punish literal readings and reward hallucinations. Pixel evidence beats
  priors 100:1 by architecture (zero-prefix visual head), never by weight.

## Named gaps (re-read before citing)

Legato/TrOMR/Zeus cell values garbled in conversion; Clarity-OMR vendor-blog
only; PlayScore/ScanScore/PhotoScore/Newzik unmeasured; SMT/SMT++ zero
real-photo numbers; Sheetly figures vendor-published; Torras/Bas 76.9%
coverage row not inspected. Suite31 "ours" figures above are binding-eval
measured (probe JSONs in gate20/), public figures are as-published.
