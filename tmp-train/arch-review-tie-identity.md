# Arch review: tie-chain identity in bekern decode

**Verdict: implement candidate B** — per-pitch open-chain *stack* with
same-segment match and a deterministic fallback. Decode-only change, no
token-stream churn, no vocab or grammar-mask change.

## 1. Mechanism (verified in code)

- Encoder (`converter/assembly-serialize/src/bekern.rs`, `to_bekern`):
  ties are per-note `tie-start`/`tie-stop` flags from `bekern_events`
  (`src/musicxml_import.rs`, staff-qualified `(staff, voice)` keys).
  Voice segments are first-appearance order **per measure**
  (`seg_of`, bekern.rs:230, cleared at every `bar`, bekern.rs:281).
- Decoder (`src/decode.rs:50-110`, `tie_note`): merges chains in
  `pending_tie: BTreeMap<(step, octave, alter), (onset, acc)>` — **pitch
  only, no voice/segment**, page-global across all parts/staves.
- Failure: two simultaneously-open chains on one pitch overwrite each
  other. ds0462 m19 (verified against the corpus file): part P1 voice-1
  ties C5 (dur 42 + 28 ticks) while part P2 voice-1 ties chord-tone C5
  (dur 28 + 7 ticks) at the same onset. The second `tie-start`
  overwrites the first, so decode emits a `[0..600]` fragment plus a
  literal `[720..1200]` instead of two merged notes. Note this is a
  **cross-part** (voice+piano doubling) collision, not same-staff voices:
  the token stream (`voice-1 … voice-2 chord …`) only reflects decode
  segments, and `pending_tie` is global to the page.
- Failed experiment (code comment on `tie_note`): keying by
  (decode-seg, pitch) with same-pitch fallback fixed 135 audit ROWs but
  added 25 new ones with whole-page displacements. Diagnosed cause is
  agreed: decode segment numbers are per-measure appearance order, which
  churns across barlines with voice entry order, so same-voice cross-bar
  chains land on different segs and the fallback misfires amid normal
  same-pitch overlaps.

## 2. Corpus evidence (read-only census, `big500-renders/*.musicxml`)

Method: regex note scan with per-voice cursor tracking
(`backup`/`forward` honored, chord onsets shared per voice, ticks
normalized to beats by per-measure divisions), chains keyed by
`(part, staff, voice, pitch)`, overlap = tick-interval intersection.
Score-level (not page-sliced); page cuts and tie-severed skips only
reduce audit-visible impact from these numbers.

- 500 scores, **14,787 tie chains, 6,731 (46%) cross a barline**.
  Within-page cross-bar merging is load-bearing and common; the audit's
  `tie_clean` skip only removes page-severed chains.
- **137/500 scores contain simultaneously-open same-pitch chains on
  different keys (572 pairs).** 40 pairs are same-`(part, staff)`,
  different voice (the ds0462 shape as described); the majority are
  cross-part doublings (singer+piano unisons — expected in Lieder).
- **101 of the 137 scores have a cross-bar chain involved** in an
  overlap. ds0462 itself contributes 4 overlap pairs, including
  `C5 (P1,1,1) x (P2,1,1) spans=(19,19)/(19,19)`.
- Consequence for the premise: "unison overlaps are rare (1 known page)"
  understates the collision domain by ~100x. The 135 ROWs the reverted
  seg-keying experiment fixed are the same order of magnitude as the
  affected pages. Pitch-only keying is broken on the common case, not
  just ds0462 — this raises the priority of the fix and rules out any
  design that only handles the single-page shape.

## 3. Candidate A — page-stable segment numbering: REJECT

Assessment against each asked point:

- **Correctness mechanism.** Would work: encoder keeps `seg_of` across
  `bar` (first-appearance order across the page), decoder persists
  `voices`/`last_onset`/`spans` vectors across `bar` (reset values, keep
  indices). Same `(staff, voice)` maps to the same seg in every bar, so
  `(seg, pitch)` keying is exact within the page, including unison +
  cross-bar + churn conjunctions.
- **Appearance-order stability truly holds.** Yes: `(staff, voice)`
  identity is stable page-wide (`staff_base` accumulates document-global
  in `bekern_events`/`staff_headers`; MusicXML voice numbers are stable
  within the page; both sides are page-scoped). A voice absent in bar 1
  that enters in bar 2 gets a late number — harmless. Decode/grammar
  have no dense-per-measure assumption (both grow vectors on demand at
  `voice-N`), so sparse bars (e.g. first marker `voice-3`) decode fine.
- **But the churn cost is prohibitive.** Every multi-voice page whose
  bar-2+ entry order differs from page-first-appearance order gets
  different `voice-N` numbers in every later bar — i.e. nearly all
  multi-voice pages retokenize. Consequences: full training-data regen,
  MVD decoder retrain, `MVD-DECODER-SPEC.md` 7b rewrite ("appearance
  order per measure" is pinned in spec, code comments, and the grammar
  docstring), test goldens regen
  (`token_stream_orders_and_marks_everything` et al.),
  `test_grammar.py` real-file walks regen.
- **Vocab-cap hazard (new finding).** `data.py` sizes `voice-1..12`
  from the per-measure maximum ("12 is the big500 maximum observed over
  the 2529 emitted pages"). Page-level distinct `(staff, voice)` counts
  are >= per-measure counts and can exceed 12 on pages that never exceed
  it per bar; those pages would start failing loudly in `to_bekern` on
  previously-good inputs. A requires a page-level key census plus a
  vocab extension (shifting nothing only if appended — but the voice
  range is mid-vocab, so ids after it shift; see `data.py` comments on
  append-only stability).
- **Two concrete footguns.** (i) `prev_key` must *still* reset at `bar`
  (markers are emitted per run) while `seg_of` persists — forgetting
  either half strands decode in the wrong segment silently. (ii) The
  grammar `_State` must mirror the persist-vectors change exactly
  (`seg_last`/`depth` keep lengths, reset values); any drift between
  mask and decoder re-opens the class of bug this review is closing.
- Verdict: exact but maximal blast radius for a problem B solves
  decode-side. Do not pay for new numbering when the numbers are not
  the problem — the *lookup discipline* is.

## 4. Candidate B — per-pitch stack + same-seg match: ACCEPT

Design: `pending_tie: BTreeMap<Pitch, Vec<(seg, onset, acc)>>`.

- `tie_start` (no stop): push `(seg, onset, performed)`.
- `tie_stop`: scan the pitch stack top-down for the first entry with
  `entry.seg == current seg`; pop and merge it (`total = acc +
  performed`; if `tie_start` also set, re-push `(seg, orig_onset,
  total)` — the middle-link case). If no seg matches and exactly one
  entry is open, take it (cross-bar single-chain case: churn moved the
  stop to a new seg, but there is no competing chain). If the stack is
  empty, emit the note literally (unchanged generator-error path). If
  several entries are open and none matches, take the stack top (LIFO)
  — arbitrary but deterministic; see residual below.
- Why this avoids the reverted experiment's failure: the failed attempt
  keyed *primarily* by seg, so every cross-bar chain on a churned seg
  missed and fell into a global same-pitch fallback. B keys primarily
  by pitch (chains never change key) and uses seg only to *disambiguate
  simultaneously-open* chains. Within a bar segs are stable, so every
  same-bar stop routes exactly; cross-bar single-open chains resolve via
  the exactly-one fallback; only the conjunction below reaches LIFO.
- Residual risk (honest bound): a cross-bar stop arriving while >= 2
  same-pitch chains are open *and* entry-order churn maps the stopping
  voice onto a seg recorded by a different chain (same-seg hits wrong)
  or onto a seg with no record while >= 2 are open (LIFO guesses).
  Census upper bound for the conjunction shape: 101 scores have an
  overlap involving a cross-bar chain (score-level; page slicing, seg
  churn between the specific open/stop bars, and tie-severed skips all
  shrink the audit-visible set further). The audit gate (§6) decides:
  zero new ROWs over the 2530-page baseline is the acceptance proof,
  not this estimate.
- No other component moves: token stream byte-identical, vocab
  unchanged, grammar mask unchanged (§5), training data unchanged,
  `tie_clean` unchanged. Blast radius is one function plus tests.

Token trace for the ds0462 m19 segment under B
(`quarter C5 dot tie-start | … chord quarter C5 tie-start |
16th C5 tie-stop | quarter C5 tie-stop`, `|` = voice markers):

1. `voice-1 quarter C 5 dot tie-start` → seg 0, push `(0, 0, 1.5)`.
2. `voice-2 quarter E 4` → seg 1, literal `E4 [0..480]` (ticks).
3. `voice-2 chord quarter C 5 tie-start` → seg 1, onset = seg-1 cursor
   0, push `(1, 0, 1.0)`. Stack (C5): `[(0,0,1.5), (1,0,1.0)]`.
4. `voice-2 16th C 5 tie-stop` → seg 1, performed 0.25; same-seg scan
   hits `(1,0,1.0)` → emit `C5 [0..600]`.
5. `voice-1 quarter C 5 tie-stop` → seg 0, performed 1.0; same-seg scan
   hits `(0,0,1.5)` → emit `C5 [0..1200]`.
6. `bar` → stacks empty, nothing flushed.

Result `{(C5,0,600), (C5,0,1200), (E4,0,480)}` — exactly the ignored
test's expectation.

## 5. Grammar-mask implications (B: none; recorded for A-rejection)

`tmp-train/mvd/grammar.py` tracks only boundary tokens, span depths per
segment, and pending starts; tie tokens are accepted in `pitch-tail` /
`tiestart` modes with no seg or tie-state tracking (`allowed_next`
adds `TIE_START`/`TIE_STOP` unconditionally there). Under B the token
language is unchanged, so **no mask change, no VOCAB change, no
`test_grammar.py` regen**. (Under A, `_State` would need the same
persist-vectors change as decode plus goldens regen — one more reason
to reject A.)

## 6. Function-level changes (B only)

1. `src/decode.rs::tie_note` — add `seg: usize` parameter; change
   `pending` to `BTreeMap<(char,i32,i32), Vec<(usize, f64, f64)>>`;
   implement §4 routing; replace the `KNOWN LIMIT` comment with the
   stack invariant (same-seg exact within bar; exactly-one fallback for
   churned cross-bar singles; LIFO documented for the ambiguous
   remainder). Both call sites (chord branch ~line 348, non-chord
   ~line 371) pass the already-in-scope `seg`.
2. EOS flush (~line 393): iterate stacks oldest-first per pitch
   (deterministic under `BTreeMap` iteration + `Vec` order); keep the
   hardcoded `voice = 1` (unscored by design — no change).
3. Tests: un-ignore `unison_ties_across_voices_merge_independently`;
   add (a) stops-in-reverse-order unison (16th-stop before quarter-stop
   already covers; add voice-1-stop-first variant pinning order
   independence); (b) cross-bar same-voice tie on churned segs
   (`voice-1 … bar … voice-2 <same voice>` synthetic pair merging via
   the exactly-one fallback); (c) ambiguous-remainder determinism test
   (two opens, stop on a third seg → top-of-stack wins, twice).
4. Audit gate: run the corpus round-trip audit; require **zero new ROWs**
   versus the reverted baseline (1524 exact / 607 skipped-tie / 399
   failed over 2530 tie-clean pages) *and* the ds0462-p2 ROW fixed. The
   399 pre-existing failures are out of scope and must not move.
5. Sequencing: land B alone. Do NOT bundle with finding 2's native-key
   change (it moves the GT reference; re-baseline the audit after B).

## 7. Candidate C — explicit chain ids / scope restriction: REJECT

- Chain ids at encode (`tie-start-N` pairing) cost an open-ended vocab
  dimension for a bounded problem, make id assignment a new learning
  target for the generator, require pairing-balance rules in the grammar
  mask and id-pool tracking in decode, and still leave the reference
  path (finding 2) colliding. Strictly worse than B on every axis.
- Scope restriction ("how often do unison overlaps vs cross-bar chains
  co-occur — can we ignore it?"): answered by §2 — no. Overlaps touch
  137/500 scores, cross-bar chains are 46% of all chains, and the
  conjunction appears in 101 scores. The restriction sub-question
  collapses into B's residual (§4), which the audit gate bounds.

## 8. Ranked additional findings

1. **The collision domain is page-global and 100x the premise.**
   `pending_tie` ignores part/staff/voice, so singer+piano doublings
   (the ds0462 shape: `(P1,1,1) x (P2,1,1)`) collide exactly like
   same-staff unisons. Census: 572 simultaneously-open same-pitch pairs
   in 137 scores; only 40 pairs are same-staff/different-voice. Any fix
   must be keyed to disambiguate *any* two keys — B's `(seg, pitch)`
   routing does; a narrower "same-staff" fix would not.
2. **The native reference path has the same collision class.**
   `score_inputs` (`src/musicxml_import.rs:220-221,299-304`) keys both
   the tie-merge map and the per-voice cursor map (`ends`) by MusicXML
   `voice` alone — no staff. Cross-staff same-voice unison ties merge
   wrongly in GT itself, and `ends` sharing across staves is one
   `<backup>`-pattern away from onset corruption. The audit's
   `tie_clean` already keys `(staff, voice, pitch)` — the reference
   should match it. Recommend staff-qualifying both maps as a
   separately-audited follow-up *after* B (it shifts GT; re-baseline).
3. **EOS flush misattributes voice (`decode.rs:393-396`).**
   Page-severed tie starts flush with hardcoded `voice = 1` regardless
   of origin segment. Unscored today (voice differs by design) and the
   audit skips severed pages, so: leave as-is, do not bundle. Noted so a
   future voice-scored metric does not rediscover it.

## Follow-up: B fails gate — mirror-oracle hypothesis CONFIRMED; pair with finding 2 (sequence in §6.4/§8.2 was wrong)

Context: stack B implemented exactly as §4 specified (same-seg top-down
match, exactly-one fallback, LIFO remainder, decode-only), plus a
mid-course `bar` gate on same-seg matches. Result: exact 1524 -> 1634
(+110 incl. ds0462-p2 fixed), failed 399 -> 289, but 25 brand-new ROW
lines vs the pitch-only baseline (incl. catastrophic ONSET Ndeltas=50/78
on ds0131 p1, ds0225 p7, ds0090 p12/p13; new COUNT ds0231 p3 327v328,
ds0245 p3 140v142; new DUR ds0003 p4, ds0054 p1). Reverted. The 25 are
byte-identical with and without the bar gate. Tree is back at
pitch-only baseline (`decode.rs:50-67` now carries a `KNOWN LIMIT`
comment pointing here).

### 1. Oracle re-verified in code — the hypothesis is correct on specifics

- `score_inputs` (`src/musicxml_import.rs:210-221`): `ties:
  HashMap<(voice, step, alter, octave), usize>` is declared **inside**
  the per-part loop — per-part, single slot. Key has voice but **no
  staff** (line 299); `ends` cursor map likewise voice-only (line 249),
  reset per measure. `bekern_events` carries staff
  (`staff_base + local - 1`, line 598) but the reference merge ignores
  it. Finding 2 confirmed as written.
- Disposition (lines 302-311, 331-335): stop extends the head
  (`dp.notes[idx].note.dur += …`, 305), keeps the slot iff start also
  set (middle link, 306-308), else removes; start-only **overwrites**
  (`ties.insert`, 331-335) with no prior-slot handling — recency.
  Orphan stop falls through to a literal emit (313-330). Baseline
  decode (`decode.rs:79-94`) is the same machine minus key components:
  single pitch-only page-global slot, overwrite on start (91), literal
  on orphan stop (88). Cursors advance regardless of tie decisions
  (392; `last_onset` likewise, 387) — merges move only merged-note
  onset/dur, never downstream onsets. No cascade is possible through
  cursors.
- ds0462 asymmetry confirmed: oracle separates P1/P2 by per-part maps;
  decode collides page-globally. That remains the ONLY structural
  decode/oracle tie asymmetry — everything else below is B diverging
  from a lossy-but-shared oracle behavior.

### 2. The 25, by mechanism (all three are keep-both vs single-slot)

(a) **Recency-overwrite pages (e.g. ds0003 p4 DUR).** Same
`(part, voice, pitch)` restarts while a chain is open: oracle drops the
first head (overwrite), B keeps both chains. Same grand total, different
split — the DUR signature. Not an oracle bug: both chains share the full
`(staff, voice, pitch)` identity, so single-slot recency IS the contract
there.
(b) **Finding-2 pages (cross-staff same-voice unison).** Oracle collides
(voice-only key) exactly the way baseline decode collides (pitch-only
key) — two wrongs agree, page reads exact. B separates via distinct segs
and diverges from the lossy GT. B is musically right; the GT is wrong.
(c) **Orphan-stop attachment → COUNT deficit.** B has no literal path
while its stack is non-empty; the oracle emits literally whenever its
slot is empty. Restart-inflated stack depth (mechanism (a)) makes a
later true stop attach in B where the oracle goes literal — so B emits
FEWER notes (327 vs 328, 140 vs 142; direction matches). This is the
discriminating proof: pure routing bugs (same-seg/LIFO misfire) preserve
note count — every stop still attaches somewhere. A COUNT deficit can
only come from open/closed-status disagreement at a stop, i.e. from the
recency/collision difference, never from seg routing. The bar-gate
invariance (identical 25 with/without) independently rules out cross-bar
same-seg coincidence: the variable is push-vs-overwrite, not matching.

Ndeltas scale: with no cursor cascade possible, each ONSET delta is one
merged note attached to a different head. Ndeltas=2 pages are single
head-reassignments; catastrophic pages (50/78) are many reassigned heads
in repeated same-pitch textures (piano doubling figures), NOT one error
cascading. One open check for the audit owner: if the scorer
index-aligns notes, a single swallowed literal (mechanism (c)) could
present as whole-page ONSET displacement — confirm the catastrophic
pages have equal note totals (if any coincides with a ±1 count delta, it
is (c), not many-heads).

### 3. Caveat: per-page cause-assignment needs the audit corpus

The big500 corpus is not on disk in this environment, and the on-disk
`tmp-train/mvd-data` streams do NOT reproduce the implicated pages:
ds0086-3 and ds0121-5 show zero overlapping same-pitch chains under any
discipline (all stops same-seg, depth never exceeds 1), and ds0003-4
lacks the reported m41/m42 half-D5 chain — so mvd-data is not the audit
input (or its pagination differs). I therefore assert the mechanism
classes and the two anchored pages (ds0003 = class (a) per the verified
token trace; ds0462 = fixed cross-part), but the (a)-vs-(b) split of the
remaining 23 must come from a join against the audit corpus, not from
this review. Classifier query per new-ROW page: does the source contain
a cross-staff same-voice same-pitch tie overlap on that page? Yes →
class (b), reclassify as oracle bug. No → class (a), genuine decode
regression under the token→oracle contract.

### 4. Q3 answered by impossibility: no decode-only design passes the gate

Within one bar, decode segs are an exact proxy for
`(part, staff, voice)` (appearance-order numbering is injective over
`(staff, voice)` keys with document-global staff ids). But the token
pattern "two same-pitch starts on different segs + stops" arises from
both oracle-separates sources (cross-part, cross-voice) and
oracle-merges sources (cross-staff same-voice). Same tokens, different
oracle behavior — the oracle key is not a function of token-visible
information. Hence **no decode-only function fixes ds0462 without
diverging on class-(b) pages**: zero-new-ROWs + ds0462-fixed is jointly
unsatisfiable pre-finding-2. This supersedes the §1 churn diagnosis for
the original seg-keying experiment's 25 and explains the bar-gate
invariance. The recommended design is therefore not "B decode-only" but:

1. Land finding 2 first (staff-qualify `ties` AND `ends` keys in
   `score_inputs`). Within-bar, `(seg, pitch)` then coincides exactly
   with the oracle key, which is what makes a decode mirror possible at
   all — this is why §6.4/§8.2 sequencing (B alone, finding 2 later) was
   wrong: the two changes share one information bottleneck.
2. Land the mirror: per-pitch stack with same-seg routing AND
   **recency** — a start on `(seg, pitch)` with an open same-`(seg,
   pitch)` entry overwrites instead of pushing (mirrors oracle
   single-slot where seg is an exact voice proxy); keep exactly-one
   cross-bar fallback + LIFO remainder. I.e. B with push→overwrite on
   same-seg collision.
3. Residual is then exactly the §4 conjunction in cross-bar form only
   (≥2 open same-pitch chains + churned stop seg): irreducible, because
   cross-bar voice identity is absent from tokens. Bound it by audit.

### 5. Re-baseline plan (concrete)

1. Apply finding 2, regen GT. Re-run the **pitch-only baseline** decode
   audit to get the new triple replacing (1524 / 607 / 399).
   Prediction: `tie_clean` already keys `(staff, voice, pitch)`
   (`decode.rs:501-507`), so the skip set (~607) should barely move;
   exact/failed redistribute as class-(b) pages flip.
2. Static census on sources (extend the §2 method): pages containing a
   cross-staff same-voice same-pitch tie overlap. Audit diff (new vs old
   baseline) must be ⊆ census; any diff page outside the census is a
   genuine regression — that subset relation is the gate that separates
   GT-shift churn from regressions.
3. Land the §4-mirror design against the NEW baseline: require zero new
   ROWs + ds0462-p2 fixed + all class-(a) pages of the old 25 gone
   (recency now mirrored). Class-(b) pages are absorbed by the GT shift
   (step 2), not counted as decode regressions.

### 6. Gate verdict (Q4)

Zero-new-ROWs stays the right **regression** gate — do not weaken it.
But pre-finding-2 it partly guards oracle bugs: class-(b) pages among
the 25 are GT errors under the audit's own staff-qualified chain
ontology, and B's output there is more correct than the reference.
Reclassify only via the step-2 census join (no blanket amnesty: class
(a), incl. ds0003 and the COUNT rows, are genuine decode regressions
under the token→oracle contract and must read exact post-mirror).

**Verdict: pair with finding 2 — do not reland B decode-only, do not
keep pitch-only. Land staff-qualification first, then the recency
mirror (§4 items 1–3 above), against a re-baselined audit.**
