# Arch review: bekern tuplet-span model vs per-note time-modification

Read-only review of `converter/assembly-serialize` (`src/bekern.rs`,
`src/decode.rs`, `src/musicxml_import.rs`) plus the MVD training target
(`tmp-train/mvd/grammar.py`, `tmp-train/mvd/data.py`).
No source files were modified; probes and token dumps lived in `%TEMP%`.

**Review-context warning (please read first).**
The tree was being edited concurrently with this review: `bekern.rs`
mtime moved 00:48 → 01:00 and `decode.rs` 00:56/12:56 → 12:58 during the
session (now ≈ 01:10). Failure signatures moved with it:

- The delegated report (m38 drop 19→17; m39 dup/shift +150 with
  `rest16+rest64`; m40/m41 phantom 480→320) reproduced exactly once, on
  the first full-suite run, then never again.
- Current source is stable (1× full suite + 4× single-test + CLI
  encode/decode round-trip all agree): ordinal-38 triplets exact,
  ordinal-39 wrong only in `F5 480→320`, ordinals 40/41 wrong only in
  phantom `480→320` on `(staff 1, voice 1)`-key notes. Suite: 44/45 pass;
  the single failure is `roundtrip_ds0037_nested_tuplets_are_note_exact`.
- Corpus-wide audit (`tmp_corpus_roundtrip_audit`, decode.rs:1008):
  **1474 exact / 598 tie-skipped / 458 failed** over tie-clean pages,
  176 scores affected, dominated by ONSET with COUNT/MIX/DUR minorities.
  The disease is corpus-wide, not one page.

Measure numbering below is global 1-based ordinals (the test harness's
`measure {m}` labels; "page 4" in test output is 0-based page index 4 =
5th page = ordinals 35–43). File `ds0037.musicxml` is 12/8,
divisions=12, bar=72 ticks. House grid is 480 ticks/beat.

Key corpus fact: of 474 `<tuplet …>` start/stop elements in ds0037,
**470 carry no `number` attribute** (parse defaults to 1; only the m27
nested pair has explicit numbers). Any matching keyed on `number` is matching on a
constant in 99% of real data.

## 1. Verdict: the implicit-span model is salvageable — do not add per-note ratio markers

The observed failures are all **encoder stop-placement** errors, not
model-expressiveness limits. Every pathology here is representable with
per-voice precise stops; per-note ratio tokens buy nothing on current
evidence and cost real vocabulary/sequence budget:

| Dimension | Per-voice precise stops (recommended) | Per-note ratio markers (not recommended now) |
|---|---|---|
| Vocab | Zero new tokens (`tuplet-stop` already exists) | ~+17 ids (one per attested ratio, cf. `data.py:92-97`) |
| Sequence length | Same stops, better positions; a few more stops (one per key per bracket) | +1 token per member note (~+10–20% on tuplet-dense pages) |
| Grammar mask | Unchanged (stop allowed iff per-seg depth > 0, `grammar.py:97-99`) | Trivially extendable, but mask cannot constrain markers to true members |
| Invertibility proof | Decoder untouched; encoder rule is a pure offline function of (key, ratio, markers) + bounded lookahead | Perfect per-note, but bracket identity still needs spans — ends as a hybrid, strictly more complex |
| Precedent | **kern: span state is per-spine (voice lane), stops are per-voice precise by construction | MEI: explicit member containment; MusicXML: per-note `time-modification` + markers attached to specific notes; Audiveris: per-note ratios grouped into explicit member lists |

The outlier is not spans per se, it is bekern's *cross-voice-covering*
span: one `(staff, voice)` key's stop marker routinely lives on another
key (ds0037 ord-38/39 bracket 4 starts on `(st1,v1)` E4b/D4 and stops on
`(st2,v1)` A3b — a cross-staff triplet). MusicXML attaches each marker
to its own note; **kern re-emits the state per spine. Bekern should do
the per-key equivalent: close each key's span after that key's last
member. Revisit per-note markers only if the corpus ever shows same-key
*non-contiguous* membership whose resulting span fragmentation hurts
learnability — exactness does not require it.

## 2. Verified mechanism (current source, token-grounded)

Token dump: `bekern --page 5` (827 tokens) → `bekern-decode`; outputs
agree with the test harness exactly.

- Ordinal 38 tail: `… voice-4 tuplet-start-3-2 rest whole rest quarter
  tuplet-stop tuplet-start-3-2 eighth F 3 eighth A 3 a-1 tuplet-stop
  voice-2 tuplet-stop bar`. The deferred flush correctly precedes the
  bar split (flush-before-split ordering, bekern.rs:237-241 before
  :242-272), and decode of this bar is now exact.
- Ordinal 39: `(st1,v1)` holds the reopened bracket-4 span across
  `E4b D4 … REST F5`. F5 has no tuplet markers (verified in source XML)
  but is emitted while the span is open and only closed by the bar-end
  tail: `… voice-2 rest eighth quarter F 5 voice-2 tuplet-stop bar`.
  Decode scales F5 ×2/3 → 320. **This is the F5 bug: over-cover, not a
  decode bug.**
- The same unclosed `(st1,v1)` span is re-split at every barline
  (`live` recency, bekern.rs:306-310, keeps it "live" because the voice
  has *events* every bar) and reopened each bar (bekern.rs:315-320),
  poisoning ordinals 40 (`voice-2 tuplet-start-3-2 quarter A 4 a-1 …
  voice-2 tuplet-stop`) and 41 identically. The 2/3 factor fits `(3,2)`
  *and* `(6,4)`; the token stream proves the leak is the **(3,2) span
  from ordinal 38's cross-staff bracket**, not 6:4 material from
  ordinals 35/36 (whose spans all close cleanly in bars 0–2).
- The older m38-drop/m39-dup/shift signature (extra `310/470/940/1100`
  onsets, count 19→17 / 11→13) does not reproduce on current source and
  is consistent with a transitional revision, not the current code.

## 3. Stacked-fix review, ranked

1. **HIGH — leaked spans propagate across barlines indefinitely while
   the voice stays active.** Stale-cancel (`bekern.rs:306-310`) keys on
   *event* recency (`live`: last bar with any event of the key), but a
   leaked span's own reopened members keep the voice "live" forever, so
   `reopen` (`:315-320`) restores it every bar. Verified: ordinals
   40/41 phantom 320s, and the DUR audit rows. Fix: restore a reopened
   span only when the key's next event is a *member* (ratio present) —
   a same-key non-member means the bracket did not continue, and
   dropping the reopen stays balanced (the pre-bar tail already
   closed it).
2. **HIGH — no precise per-voice stop.** A key's span closes only on its
   own marker (`bekern.rs:472-509`) or at bar end, so a key whose
   notational stop sits on another key (cross-staff brackets) over-covers
   everything after its last member. Verified: F5 480→320. Fix: §4 rule.
   Findings 1+2 are one fix family; either alone halves the residual,
   together they zero the ds0037 page (see §4 traces).
3. **MEDIUM — `deferred_stop` is a single global slot whose correctness
   rests on undocumented ordering.** Safe today only because (a) flush
   runs before the bar split (`:237-241` vs `:242-272`), so the deferred
   key's stack is still present, and (b) `continues_chord` (`:235-236`)
   forces same-key chord runs, so two keys can never have stops pending
   at once. Add the `debug_assert!` + comment, or key the slot by
   `(staff, voice)`. No behavior change needed.
4. **MEDIUM — validate against the corpus audit, not one page.** 458
   failing tie-clean pages (mostly ONSET displacement, some COUNT/MIX)
   share the span-coverage family; the ds0037 unit test alone cannot
   guard the fix. Gate the change on `tmp_corpus_roundtrip_audit`
   showing DUR rows flipping exact with zero new ROW lines.
5. **LOW (latent) — decode's `pending_spans` is global.**
   `tuplet-start … voice-X … event` drains the start into X's segment
   (`decode.rs:212` push; drains at `:247,:321`), which mis-scales.
   The encoder never emits that shape (starts immediately precede their
   member), and the grammar mask (`grammar.py:40,95-99,195-206`) is
   *documented* wider-than-encoder,
   so this is a latent generator-misroute, not a live bug. Document the
   "start immediately precedes its member event" invariant next to the
   drain.
6. **LOW — oracle/encoder cursor-model divergence on over-long backup.**
   The oracle clamps at bar start (`musicxml_import.rs:268-271`) while
   `bekern_events` does not (`:559-562`); the reference `score_events`
   also does not clamp. Benign on well-formed input; note for the next
   person touching backup handling. (`measure_rest` pos staleness,
   `bekern.rs:289-294`, is likewise benign: a whole-bar rest precludes
   further same-key events that bar.)
7. **INFO — hygiene.** `tmp_page_diag` (`decode.rs:865`) and
   `tmp_corpus_roundtrip_audit` (`decode.rs:1008`, +39 s suite time) are
   self-marked TEMPORARY scaffolding; keep the audit until §4 lands,
   then move it out of the unit-test binary. Gap-fill
   (`GAP_MIN_BEATS`, `bekern.rs:67`; self-bracket fallback `:352-396`)
   and the keyed tie map (`decode.rs:57-83,146-147`) reviewed sound.

## 4. Minimal design (decode, grammar, vocab all unchanged)

**Change one function: `to_bekern` (`bekern.rs:156-527`).** Add a
synthetic precise stop at loop-top (`:237-241`, same site as the
deferred flush): when the previous emitted event's key `pk` still has an
open span *in the same measure*, close its top level iff (a) the current
event is a different key **and no later same-key event in this measure
carries the same ratio** (one backward pass per measure precomputes
this; cross-staff brackets close, continuous brackets survive), or
(b) the current event is the same key but a non-member (no
`tuplet_ratio`). Skip when `e.measure` changed (bar split owns it) and
for a key just marker-flushed this round (marker owns it); never fires
mid-chord-run (whole block is already skipped). The stop token is the
existing bare `tuplet-stop`, routed to the stopped note's segment by
the same mechanism as deferred flushes. Then finding-1's fix (member-
recency reopen) is three lines at `:315`.

Expected ordinal-39 bar under the proposal (diffs only): `… eighth D 4
tuplet-stop voice-3 …` (new precise stop; F3@16 is a different key and
no later `(st1,v1)` member exists) and the bar ends `… quarter F 5 bar`
with no trailing `voice-2 tuplet-stop` — F5 decodes 480, no reopen into
ordinal 40, so the ordinal-40/41 bars lose their reopen spans (and their
tail stops) entirely. Ordinal-38's bar loses both barline tails the same
way (E4b@36 is `(st1,v1)`'s last member; its span closes at C3@36's
loop-top; the `(st2,v1)` side stays marker-balanced). Bracket
fragmentation is confined to genuinely non-contiguous same-key runs,
which stay decode-exact via the existing implicit-member restart
(`:412-421`).

**Tests:** add one unit fixture shaped like ordinal 38/39 bars (cross-
staff bracket + trailing plain quarter on the start key) asserting the
exact token stream including the synthetic stop; keep all 45 tests
green; re-run `tmp_corpus_roundtrip_audit` expecting the DUR rows
(ds0037 p4 et al.) to flip exact with no new ROW lines; re-run the MVD
grammar walk (parent cites 0/13027) since stop *positions* move.
