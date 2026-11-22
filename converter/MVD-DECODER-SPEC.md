# MVD — Minimum Viable Decoder (learned system/page sequence stage)

Status: spec (2026-11-05). Decision driver: rule-mapper oracle ceiling
F1 0.10–0.22; detection healthy (HeatNet tile Note-F1 0.982).
Target: TrOMR-style hybrid — frozen proposer + learned decoder.

## 1. Target format: bekern (decided)

Token-format spike (ds0001-1, 179 events) measured:

| | bekern | Linearized-MusicXML |
|---|---|---|
| Tokens/page | **608 (3.40/event)** | 6297 (35.2/event); note-subtree 5730 |
| Vocab | ~40 closed | ~40 tags + open text |
| Importer gap | small (below) | large (beams/stems/notations/lyrics/layout) |
| ONNX `sequence` fit | small softmax, short AR roll | 10× decode, heterogeneous schema |

bekern rules (SMT++ "basic extended kern", arXiv 2405.12105): note =
duration + step + octave + accidental(iff altered) + 1/dot + 1/tie-mark;
rest = `rest` + duration + dots, whole-bar rest = `bar-rest`;
chord tones = full note tokens with `chord` before the 2nd+ tone;
header per staff = `clef-*-*-*` + `key-*` + `time-*-*-*` (3 tokens);
barline = 1/measure; tuplets wrap written durations in
`tuplet-start-A-N` ... `tuplet-stop`; voices marked per run.
Headers 9 + pitched 555 + plain rests 16 + bar-rests 5 +
chord markers 47 + voice markers 128 + barlines 13 = 773
on ds0001-1. (The spike's hand count said 623: it double-counted
barlines per part, mis-added headers, and missed rest dots. The
committed golden test pins 773 with per-category asserts.)

Triple-reconciled GT on ds0001-1: MusicXML parse = 179 events / 11 rests
/ voices {1,2,3,5,6}; `big500-symbox/ds0001/ds0001-p1.json` parts-notes =
179 / 11 (notes live under `parts[i].notes`, top-level `notes` is empty);
symbox `Note` boxes = 168 = 179 − 11. Per-page `ScoreInput` GT
(`ds000X-pN.json` in `big500-symbox/dsXXXX/`) is real and is the v1
supervision source — no alignment problem at page level.

## 2. Importer gaps to close (assembly-serialize)

bekern needs, beyond today (`musicxml_import.rs`): `clef` parse per
staff (currently skipped); `type`/`dot` (derivable from tick durations
via existing `quantize_duration`, lib.rs:451); `staff` in the
`NoteInput` path (parsed but only used by `score_events`); grace-note
policy (currently dropped, :280-282). LMX-only needs (beams, stems,
notations, lyrics) stay out of scope.

## 3. Data contract (v1: page-level; v2: system-level)

- v1 pairs: page PNG (existing renders, 1240px tiles OK) + bekern from
  the same page's `ScoreInput` GT (`ds000X-pN.json`, 30 pages) +
  GrandStaff 53k system pairs (license-check before download) +
  OLiMPiC-synth. Degraded variants via photo-degrade with reprojected
  labels (0.17px) + occluder masking.
- v2 systems: single-system crops aligned by Verovio SVG `xml:id`
  (second-engraver backend doubles as alignment source). Verovio
  integration is a data prerequisite, not a decoder dependency.
- Splits: piece-level integrity via `pack --strict` (leakage gate proven).
  ds0005 held out (zero voids — void-sensitive checks run on ds0003).

## 4. Model (one RTX 4080 SUPER, repo-root `.venv`)

Env verified: `torch 2.14.1+cu126`, CUDA True at `.venv/Scripts`
(the `Python312` named interpreter has no torch — never train there).
- Encoder: ConvNeXt-Tiny from scratch (SMT-NeXt scale; GrandStaff-scale
  data makes from-scratch viable, SMB-SOTA precedent).
- Decoder: 4–6-layer causal transformer, 256-d, 8 heads (TrOMR/SMT
  scale), cross-attention to tile features; ~40-class softmax head;
  auxiliary beam/rest loss.
- Curriculum: GrandStaff systems → our 30 pages (page-level) →
  degraded tiles → Verovio-aligned systems.
- Box-geometry variant as ablation (HeatNet features + box embeddings)
  vs pure pixels; keep the winner.

## 5. Deploy shape (new backend, not a GlyphDetector impl)

`GlyphDetector::detect` returns `Vec<Glyph>` per crop — a sequence
decoder emits page-level token strings via an AR loop. It needs a new
`sequence` backend shape (new trait method or trait), exported to ONNX
(AR loop + ~40-class head; `ort 2.0.0-rc.15` precedent in
symbol-detect). `OnnxDetectorHook` is the wrong seam to reuse.

## 6. Gates (falsifiable; kill or promote)

- Adopt on ds0005 iff system SER **below the oracle band 0.48–0.75**
  **AND** strict-F1 **above 0.22**. Else kill the decoder, keep rules.
- Ablations required: (a) no-box pure-feature variant; (b) fine-tune
  on 10 real degraded pages (must move real-F1, never Otsu parity).
- Product gates unchanged: median fixes/page, zero pages >50, valid
  MusicXML + .mscz, seconds/page published.

## 7. File plan

- `tmp-train/mvd/tok.py` — UNNEEDED: tokenizer lives in
  `assembly-serialize/src/bekern.rs` (Rust, tested). The golden test
  pins ds0001 page 1 at 608 tokens.
- `tmp-train/mvd/data.py` — page/system pair loader (COCO boxes +
  bekern + degraded variants), piece-level splits.
- `tmp-train/mvd/model.py` — ConvNeXt-Tiny + transformer (venv torch).
- `tmp-train/mvd/train.py`, `export.py` — curriculum + ONNX AR export.
- `converter/symbol-detect` — `sequence` backend shape + ONNX hookup.
- `converter/eval-harness` — decoder scoring (bekern→notes→strict-F1).

## 7b. Canonical token table (normative for tok + data.py)

Whitespace-separated, one page per file. Exact strings:

- Step: `C D E F G A B`; octave: bare `0`..`8` (e.g. `C 4` = middle C);
  altered only: `a-2 a-1 a1 a2` (flat= a-1, sharp = a1).
- Duration: `whole half quarter eighth 16th 32nd 64th breve`; dots:
  one `dot` per dot (breve: plain only, 14 events attested in big500).
  Genuinely untypeable durations are an error, never rounded.
- Tuplets: members carry written (unscaled) durations inside
  `tuplet-start-A-N` ... `tuplet-stop` spans (`A` = actual notes,
  `N` = normal notes, e.g. triplet eighth = `tuplet-start-3-2`
  `eighth` ... `tuplet-stop`). 17 corpus-attested ratios in vocab;
  unseen ratios fail loudly at data build.
- Ties: `tie-start`, `tie-stop` after the note tokens.
- Chords: `chord` before the 2nd+ tone of a same-onset same-staff
  same-voice group (without the marker the stream cannot tell a chord
  from sequential same-voice notes; the decoder needs it exact).
- Voices: `voice-N` on every voice run (appearance order per measure,
  1..12 attested) — token order interleaves voices by onset, so decode
  switches segments on each marker and reappearances continue their
  cursor. Exact inversion (round-trip F1 1.0 on ds0001-1, pinned in
  `decode.rs` tests); heuristic segmentation was tried and capped the
  round-trip at 0.48.
- Rest: `rest` + duration + dots; whole-bar rest (`<rest
  measure="yes"/>`, any meter) is the single token `bar-rest`.
- Known gap (1 page in big500): triple-and-beyond dots (e.g. ds0033
  quadruple-dotted half rest, 3.875 beats) have no token and error
  loudly. Deliberately ungrown: 1 attested note in 500 scores.
- Barline: `bar` after each measure's events (incl. the last).
- Header, per staff in global staff order: `clef-G-2` (`clef-<sign>-<line>`,
  all 15 G/F/C x line-1..5 in vocab), key `key+0 key+2 key-1`
  (explicit sign, `key<fifths>`), time `time-4-4` (`time-<beats>-<type>`,
  15 signatures in vocab, incl. the corpus-attested 4-8/7-8/8-8).
- Order: all header tokens first, then events sorted by
  (measure, onset, staff, voice, midi). Chord tones = full note tokens
  each, 2nd+ tone prefixed with `chord`. Every voice run opens with
  `voice-N` (appearance order, reappearances re-marked). Grace notes
  skipped + counted.

## 8. Open items / risks

- GrandStaff: NOT used — the archive carries no license statement
  (verified Nov 2026; U-MusT notes the same gap), so weights trained
  on it could not ship. Curriculum is self-rendered CC0 scores
  (mscx-5000/big500) instead. DoReMi / SMB license check still open.
- Verovio second-engraver backend (data alignment for v2).
- Time-sig / key-mode parse only if bekern headers need them (key via
  fifths ✓ today; time via beats/beat-type ✓; clef = the gap).
- If v1 page-level stalls (30 pages too few even with GrandStaff),
  the fallback is SMT-family MIT weights via the `sequence` backend,
  not more rule work.
