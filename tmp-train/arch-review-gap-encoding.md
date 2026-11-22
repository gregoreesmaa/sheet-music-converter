# Arch review: gap-encoding robustness (bekern.rs gap-fill / tol / S1 / S2)

Read-only review of `converter/assembly-serialize/src/bekern.rs` (gap-encoding
concept: `emit_rest` + `tol`, gap-fill block, reopen/S1-stale rule, S2
sequential-close rule, `deferred_stop`/`flush_stop`, precise-stop), the three new
TDD tests in `src/decode.rs`, prior reviews `tmp-train/arch-review-span-model.md`
(+ its §4 minimal design, now evidently landed as the precise-stop) and
`tmp-train/arch-review-tie-identity.md` (+ Follow-up: mirror-oracle, recency,
staff-qualified ties). Verified by reading code; the three TDD tests were re-run
green in this session (3 passed). The audit triple (1641 exact / 282 fail, zero
new failing pages vs baseline by page-key diff) is parent-reported, not re-run
here — the big500 corpus is not on disk in this environment.

Field best practices used as yardstick: (i) encoding/invertibility design —
`decode(encode(x)) == x` must hold by construction, with fail-closed errors, never
silent rescaling; (ii) error taxonomy — distinguish loud failures (encode error,
audit-visible) from silent wrong-but-plausible outputs (audit-visible only if the
data exists) from learning-signal-only degradations (bracket fragmentation the
note-level audit cannot see); (iii) tolerance budgeting — every quantum must
carry a proved bound in audit-visible units (ticks) and must not accumulate
across a page; (iv) span/bracket modeling precedent — MusicXML brackets are
identified by `number` within a part, **kern spans are per-spine (per voice
lane), MEI uses explicit member containment (prior span-model review §1, §4).

## Q1 — Ratio-mismatch-drop (S1): sound

Rule (`bekern.rs:358-377`): at the key's first event after a barline, a
barline-split `reopen` stack is restored only if the event is a bracket member
(ratio/start/stop present); if the event carries a `tuplet_ratio` differing from
the restored top, the reopen is dropped. Dropping stays balanced because the
pre-bar tail already closed those levels token-side, and the new ratio reopens
implicitly below (`:519-527`).

**Sound, and ratio (not number) is the right key.** The span-model review
established the load-bearing corpus fact: 470/474 ds0037 `<tuplet>` elements
carry no `number` (all default to 1). A number-aware cross-key bracket identity
cannot disambiguate anything in the wild — it matches on a constant. Ratio is
the only per-member signal that varies, so keying the stale decision on ratio is
the correct choice, not a shortcut. The cross-key case (start on key A, stop on
key B) does not need cross-key number routing either: each key carries its own
implicit span and the precise-stop (`:466-487`) closes the start key after its
last member — per-spine spans, per the **kern precedent.

What breaks if TRUE cross-bar nesting with mismatched ratios exists: less than
it appears. Dropping the outer reopen loses bracket *continuity*, not timing:
later outer-ratio members implicitly restart the span (`:519-527`), so decoded
note onsets/durations stay exact. The damage is learning-signal-only (one
bracket becomes two fragments) and **audit-invisible**. Two genuine holes, both
bounded:

- (a) The drop discards the **whole stack** on a top-ratio mismatch. With a
  nested stack `[(3,2),(2,1)]` and next-event ratio `(3,2)`, the outer `(3,2)`
  is lost along with the stale `(2,1)` top. A pop-one-level refinement would
  preserve it. No test pins this; add one (see §5).
- (b) The stale check only fires when `tuplet_ratio` is present (`_ => false`
  arm, `:366`). A bare explicit `<tuplet type="start">` with no
  time-modification restores the stale span, then the start path defaults to
  `(3,2)` — a nested scale-3 that fails **loudly** at gap-fill, not silently.
  Fail-closed, rare in practice (starts without time-mod are near-absent in
  real engraving). The code comment ("drops the reopen even with an explicit
  start or stop", `:351-354`) overclaims slightly: it drops on ratio mismatch
  *regardless of accompanying* start/stop, but a start/stop *alone* never drops.
  Fix the comment or the arm; I recommend fixing the comment (bare starts are
  too rare to deserve new behavior, and loud failure is acceptable there).

## Q2 — S2 sequential-close: risky-with-condition (accepted heuristic, pin the residual)

Rule (`:506-513`): an explicitly starting bracket of a different ratio closes
the open top iff the voice cursor abuts (`gap <= GAP_MIN_BEATS`), the top is
unnumbered (`number == 1`, i.e. implicit runs), and the span was not just
restored (`!restored_this_event`). The two TDD tests pin the intended
discriminator exactly: gap → nest (`explicit_start_after_gap_nests_under_open_span`),
abutment → close (`explicit_start_after_abutting_run_closes_stale_span`).

**The discriminator (abutment) is a 1-quantum threshold standing in for
bracket-interval containment, and it will flatten genuine aligned nesting.**
A duplet genuinely containing triplets whose inner start lands exactly on a
duplet gridline (no `<forward>` silence, unnumbered outer — the common case
given Q1's number finding) satisfies all three guards and gets flattened: the
inner notes then decode scaled by 2/3 instead of 1/3. Unlike S1's drop, this
is **silently wrong timing** (audit-visible ONSET/DUR *if* such data exists —
the zero-new-ROWs gate says none has been seen yet, which bounds the wild
frequency but does not prove absence).

Why it is still the right call today: the alternative (always nest) was
observed to triple-scale real sequential runs into `emit_rest` failures
(ds0226, pre-fix error). Both directions cannot be distinguished from the
token cursor alone — sequential-vs-nested is a source-bracket-interval fact
(does the outer ratio reappear after the inner bracket closes?), which needs
lookahead the streaming encoder does not do. The guards are well-chosen
minimizers: `!restored_this_event` protects cross-bar nesting (barline reads
nest by default), the unnumbered-top guard preserves explicitly numbered
nesting, and gap-vs-abutment is correct whenever engraving puts silence
between sequential brackets (the `<forward>`/late-entry shape).

How to detect it in the wild (source query, no audit needed): on the corpus
sources, find an explicit `tuplet-start` event whose onset lies **strictly
inside** another bracket's `[start-onset, stop-onset]` interval on the same
`(staff, voice)` with a different ratio, with `|onset − voice-cursor| <= 1
quantum` (abutment) and outer `number` absent/1. Any hit is a page S2 currently
flattens. The audit gate cannot substitute for this query — the audit compares
notes, and S2's output *is* wrong notes, so a hit would surface as ONSET/DUR;
but absence of such ROWs is weak evidence without the source join (cf. the
tie-identity Follow-up §3 lesson: per-page cause-assignment needs the corpus).

Principled fix (not requested now): a per-measure pre-pass recording, per key,
the set of ratios appearing after each position (the span-model review §4
already proposed a backward pass for the precise-stop family); close on
abutting explicit start only if the outer ratio never reappears later in the
measure. Bounded, offline, decode- and grammar-neutral.

## Q3 — Scale-aware tolerance: sound bound, wrong layer (keep with a cap)

Mechanism: `emit_rest(written, tol = fill_scale * GAP_MIN_BEATS)` where
`fill_scale` is the open-span actual/normal product (`:397-410`); the wrapper
fallback uses `r.0/r.1 * GAP_MIN_BEATS` (`:454`). Dust source is correctly
diagnosed in comments: 3dp onset quantization (`round3` of tick/division —
triplet grids read 0.333+0.333 back as 0.666 against onset 0.667) plus
full-precision float `pos` chains, so residue grows with the span product
(ds0134: 0.498 vs an eighth under a `(3,2)` wrapper).

**The bound is sound; the layer is compensatory.** The audit-visible guarantee
holds by construction and is worth stating exactly:

- Per filled gap, the snapped written rest differs from the true
  `gap × scale` by at most `tol = scale × 0.0015` beats (the loop exits with
  `rem <= tol`, swallowed). Decoded performed error is therefore at most
  `tol / scale = GAP_MIN_BEATS = 0.0015` beats = **0.72 ticks → ≤ 1 tick after
  the audit's rounding**. This is why "decoded timing shifts stay within one
  gap-space quantum either way" (`:120-121`) is literally true.
- Non-cascading on the encode side: `pos` advances by the exact `gap`
  (`:464`), never by the snapped fill, so encoder-side error does not
  accumulate. Decode-side cursors accumulate snapped rests, so drift can sum
  to **≤ 0.72 ticks × (number of fills in the voice's dense region)** —
  bounded, and each term is usually ~0 (dust, not worst case). A corpus
  census of max fills-per-voice-per-measure would turn this into a number;
  with gaps as rare as late-entries/`<forward>`s, expect single digits.
- Failure mode is fail-closed: genuinely untypeable gaps still error
  (`:425-439`, straight silence with no ratio), so `tol` growth under deep
  nesting cannot silently notate the un-notatable — it can only snap within
  1 performed quantum. The residual audit-invisible effect is a ≤
  `scale × 0.0015`-beat imprecision in the *written* rest token (the learning
  target), e.g. ~2 written ticks under scale 3.

Fixing dust at its source (integer tick-space gap computation: `gap_ticks =
onset_ticks − cursor_ticks`, then exact rational × ratio arithmetic) is the
principled layer and would make scale-aware `tol` unnecessary — but it is a
large refactor touching `bekern_events` (float onsets are the scorer-parity
contract), the `pos` chain, and the audit baseline. Condition for keeping the
current layer: assert the bound (a test that every gap-fill in the corpus
fixtures shifts decoded timing by ≤ 1 tick — implied by the audit but never
pinned per-gap) and cap the nesting depth the tolerance may amplify (today
unbounded span products linearly widen `tol`; a `debug_assert!(fill_scale <=
K)` with K from a source census makes the compensation explicit rather than
open-ended).

## Q4 — State plumbing: sound, with two documentation-grade smells

Checked interactions, all sound:

- `restored_this_event` / `cursor_abuts` are *set* in the reopen/gap blocks
  and *consumed* by S2 two blocks later — but both are freshly initialized per
  event (`:316-317`) and S2's guards fail safe on the defaults. In particular
  `cursor_abuts` stays `false` mid-chord-run (gap block skipped), which is
  load-bearing for S2 never firing mid-chord — correct, but implicit. One-line
  comment at the S2 guard stating "abuts is false mid-chord by construction"
  would stop the next editor breaking it.
- `deferred_stop` / `flushed_key`: single-slot invariant holds — loop-top
  flush precedes both the bar split and the seg switch in the same iteration,
  so the flushed token is always seen by decode in the stopped note's segment
  (the stated routing invariant, `:151-157`), and same-onset chord runs
  (which skip the flush) are forced same-key by `continues_chord`. The
  span-model review's finding 3 asked for a `debug_assert!` + comment; the
  comment landed (`:579-584`) but no assert. Add
  `debug_assert!(deferred_stop.is_none())` at the take-then-set site — cheap.
- No unsound `deferred_stop` × `reopen` interaction: flush runs before the
  reopen-cancel/restore of the same event, and the cancel is skipped while a
  stop is pending (`:329-330`), so the number-validated pop (`flush_stop`)
  never meets a restored stack. Cross-bar deferred stops are safe for the
  same ordering reason (flush precedes the split within the iteration).
- Gap-fill-then-close ordering (fill under the stale span, S2 close after)
  is *required*, not incidental: decode scales the emitted rest by the span
  it sits inside, so closing first and filling unscaled would desync the
  cursor. Do not "fix" the order.

Smell worth one line each: (i) S2 lives outside the `!continues_chord` gate
  while two of its three guards are computed inside it — works, fragile to
  refactor; (ii) the S1 comment overclaims on bare start/stop (Q1-b).

## Verdicts

| Question | Verdict |
|---|---|
| Q1 ratio-mismatch-drop (S1) | **sound** — ratio is the only discriminating signal (numbers are constant in the wild); whole-stack drop (a) and bare-marker arm (b) are bounded holes |
| Q2 sequential-close (S2) | **risky-with-condition** — correct fix for observed sequential runs, but abutment is a 1-quantum proxy for bracket containment and flattens genuine aligned unnumbered nesting silently; condition: run the §2 source query, pin the residual test |
| Q3 scale-aware tolerance | **sound bound, compensatory layer** — ≤1 tick per fill, non-cascading encode-side; keep with a depth cap + per-gap bound test; tick-exact arithmetic is the principled (larger) fix |
| Q4 lockstep/state plumbing | **sound** — orderings verified (flush→split→restore; fill→close); two comment-grade smells, one missing `debug_assert!` |

## Highest-risk mechanism

**S2's abutment guard.** It is the only new mechanism that can emit
wrong-but-plausible timing on real data without failing loudly (S1 degrades to
decode-exact restarts; tolerance errors fail closed; plumbing is verified).
Its trigger condition — explicit start exactly on the voice cursor under an
unnumbered different-ratio span — is indistinguishable in the token stream
from the sequential runs it was built to fix. The zero-new-ROWs gate bounds
its wild frequency today; the §2 source query is the instrument that keeps it
bounded tomorrow.

## Smallest test to add

Abutting genuine-nesting pin (decode.rs, next to the other two TDD tests):
same-key unnumbered duplet run, then an explicitly started triplet whose first
note lands exactly on the duplet gridline (no `<forward>`, cursor abuts),
with duplet-ratio members resuming after the triplet closes. Assert current
behavior explicitly — today that is flattening (one `tuplet-start-2-1` … stop
before the triplet, triplet at top level) — with a comment marking it the
accepted S2 residual and pointing at the §2 source query. Rationale: the
cheapest invariant with the most information. If such data ever appears in the
wild, the audit flips these pages to ONSET/DUR and this test names the culprit
mechanism instantly; if the lookahead fix ever lands, this test is the
acceptance criterion (both levels survive, inner 16ths at 1/6 beat). Second,
one line: `debug_assert!(fill_scale <= 4.5)`-class cap or the per-gap ≤1-tick
assertion of §Q3 — the tolerance deserves its bound in code, not just prose.
