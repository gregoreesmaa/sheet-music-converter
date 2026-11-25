"""MVD decoder training-data loader (page-level, v1).

Per training item:

  (a) page PNG pixels as float32 HWC in [0, 1];
  (b) target tokens from ``<page-stem>-bekern.txt``
      (whitespace-separated, one file per page);
  (c) auxiliary boxes from a COCO JSON (``images`` / ``annotations`` /
      ``categories`` schema, as in ``tmp-train/det-data/coco.json``).

Splits are read from a dataset-pack ``manifest.jsonl`` (one JSON object
per line, with at least ``id`` and ``split`` fields). Piece-level
integrity is *asserted, not repaired*: all ids sharing a piece key must
land in the same split, otherwise :func:`load_split` raises
``AssertionError`` before returning anything.

Dependencies: Python standard library + numpy + Pillow only. ``torch``
is never imported here (it lives in the repo-root ``.venv`` and belongs
to training code, which this module must not contain).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Union

import numpy as np
from PIL import Image

__all__ = [
    "VOCAB",
    "TOKEN_TO_ID",
    "ID_TO_TOKEN",
    "PAD_ID",
    "BOS_ID",
    "EOS_ID",
    "VOCAB_SIZE",
    "EXT_DYNAMICS_VOCAB",
    "EXT_LYRIC_MARKERS_VOCAB",
    "EXT_PERCUSSION_TAB_VOCAB",
    "build_vocab",
    "AUX_CLASSES",
    "encode",
    "decode",
    "piece_key",
    "load_bekern",
    "load_sidecar",
    "load_item",
    "load_split",
]

PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# Vocabulary.
#
# Closed bekern token set. Canonical spellings (normative:
# ``converter/MVD-DECODER-SPEC.md`` 7b, pinned by the
# ``assembly-serialize`` golden test): keys carry an explicit sign
# (``key+0``, ``key+2``, ``key-1``) and time signatures are
# ``time-BEATS-TYPE``. Clefs are single tokens (``clef-G-2``). A note
# is a duration + step + octave token run (``quarter C 4``), accidentals
# only when altered (``a1``). Tuplet members carry written durations
# inside ``tuplet-start-A-N`` ... ``tuplet-stop`` spans. A whole-bar
# rest is the single token ``bar-rest`` (no duration).
# ---------------------------------------------------------------------------

_STEPS = ["A", "B", "C", "D", "E", "F", "G"]
_OCTAVES = [str(n) for n in range(0, 9)]
_DURATIONS = ["whole", "half", "quarter", "eighth", "16th", "32nd", "64th"]
_MARKS = ["dot", "tie-start", "tie-stop"]
_ALTERS = ["a-2", "a-1", "a1", "a2"]
_SYMBOLS = ["rest", "bar"]
_CLEFS = ["clef-%s-%d" % (s, n) for s in ("G", "F", "C") for n in range(1, 6)]
_KEYS = (
    ["key%d" % n for n in range(-7, 0)]
    + ["key+0"]
    + ["key+%d" % n for n in range(1, 8)]
)
# 15 signatures: the 12 common ones plus the -8 family members the
# big500 corpus actually uses (``time-4-8`` ds0196, ``time-7-8``,
# ``time-8-8`` — 9 pages total, found by full-corpus census 2026-11).
_TIMES = [
    "time-%s" % t
    for t in (
        "4-4", "3-4", "2-4", "5-4", "6-4",
        "2-2", "3-2", "4-2",
        "3-8", "4-8", "6-8", "7-8", "8-8", "9-8", "12-8",
    )
]

# Tuplet span markers. ``tuplet-start-A-N`` opens a span of ``actual:A``
# notes in the time of ``normal:N`` written durations; ``tuplet-stop``
# closes the innermost open span. Ratios are the closed set attested in
# the big500 corpus census (triplet 3:2 dominates); a new ratio fails
# loudly in encode() and must be added here deliberately. Appended last
# so existing ids never shift under saved checkpoints.
_TUPLET_RATIOS = [
    (3, 2), (6, 4), (2, 1), (5, 4), (2, 3), (4, 6), (7, 4), (5, 2),
    (9, 8), (7, 6), (7, 8), (5, 3), (4, 3), (12, 8), (10, 8), (9, 4),
    (8, 2),
]
_TUPLETS = ["tuplet-start-%d-%d" % r for r in _TUPLET_RATIOS] + ["tuplet-stop"]

# Whole-bar rest (``<rest measure="yes"/>`` in any meter). Appended last
# like the tuplet markers so all earlier ids stay put.
_BAR_REST = ["bar-rest"]

# Breve (8-beat) note value, attested in big500 (ds0349). Appended last
# for the same id-stability reason; dotted breves stay a loud error.
_BREVE = ["breve"]

# Chord-tone marker: emitted before the 2nd+ note of a same-onset
# same-staff same-voice group, so decode need not guess chords.
_CHORD = ["chord"]

# Voice-segment markers ``voice-1`` .. ``voice-12`` (appearance order
# per measure; 12 is the big500 maximum observed over the 2529 emitted
# pages — a 9th voice (ds0127-2) broke the old voice-8 cap, so the
# range now covers the corpus max. A 13th voice fails loudly in
# encode(); extend deliberately if the corpus grows one.
_VOICES = ["voice-%d" % i for i in range(1, 13)]

# String-quartet (``sq``) corpus additions (full-corpus census 2026-11-08:
# 242 files, 0 remaining unknowns after this). Percussion clef
# ``clef-p-2`` (114x, e.g. sq7358708); 5 meters — ``time-12-16`` (272x),
# ``time-5-8`` (28x), ``time-15-8`` (20x), ``time-2-8`` (4x),
# ``time-1-2`` (4x); 17 tuplet ratios — ``12:4`` (307x), ``8:6`` (27x),
# ``6:5`` / ``12:5`` / ``8:12`` (4x each), ``13:8`` (3x), ``24:4`` (2x),
# ``10:4`` / ``18:8`` / ``9:2`` / ``15:8`` / ``5:6`` / ``11:8`` /
# ``33:8`` / ``3:4`` / ``19:16`` / ``11:6`` (1x each). Lives AFTER
# _VOICES so every pre-existing id stays put under saved checkpoints;
# a new token fails loudly in encode() and must be added here.
_SQ_ADD = (
    ["clef-p-2"]
    + ["time-%s" % t for t in (
        "12-16", "5-8", "15-8", "2-8", "1-2",
    )]
    + ["tuplet-start-%d-%d" % r for r in (
        (12, 4), (8, 6), (6, 5), (12, 5), (8, 12), (13, 8), (24, 4),
        (10, 4), (18, 8), (9, 2), (15, 8), (5, 6), (11, 8), (33, 8),
        (3, 4), (19, 16), (11, 6),
    )]
)

# bekern+ (lossless MusicXML round-trip), Nov 2026: grace marker, slur
# endpoints, closed articulation subset, measure dynamics, hairpin wedge
# events, closing/opening barline styles, plus suite31-attested time
# signatures (``time-1-16``, ``time-1-32``: grace-only pickup bars).
# Appended last so every pre-existing id stays put under saved
# checkpoints; a new token fails loudly in encode() and must be added
# here deliberately.
_BEKERNPLUS = (
    ["grace"]
    + ["slur-start", "slur-stop", "stacc", "accent", "tenuto", "fermata"]
    + ["dyn-%s" % d for d in ("pp", "p", "mp", "mf", "f", "ff", "sfz")]
    + ["hairpin-cresc-start", "hairpin-dim-start", "hairpin-stop"]
    + ["bar-light-heavy", "bar-repeat-left", "bar-repeat-right",
       "bar-double"]
    + ["time-1-16", "time-1-32"]
)
assert all(
    t not in (
        _STEPS + _OCTAVES + _DURATIONS + _MARKS + _ALTERS + _SYMBOLS
        + _CLEFS + _KEYS + _TIMES + _TUPLETS + _BAR_REST + _BREVE + _CHORD
        + _VOICES + _SQ_ADD
    ) for t in _BEKERNPLUS
), "bekern+ must be append-only (id stability)"

# T3.2 closed-vocab completion (full t32-corpus census 2026-11-11:
# 33852 bekern GT files, 19 unknown types in ~200 stems): 9 meters --
# ``time-9-4`` (229x/50 stems), ``time-4-16`` (96x/32),
# ``time-9-16`` (66x/32), ``time-7-4`` (64x/21), ``time-6-16``
# (49x/12), ``time-8-16`` (36x/9), ``time-1-4`` (12x/6),
# ``time-12-4``, ``time-27-8``; 10 tuplet ratios -- ``11:12``,
# ``10:12``, ``16:8``, ``4:2``, ``10:6``, ``18:4``, ``6:8``,
# ``24:8``, ``7:1``, ``35:16``. Lives AFTER _BEKERNPLUS (the
# _SQ_ADD/_BEKERNPLUS precedent): new rows at the end, so every
# pre-existing id stays put under saved checkpoints and ckpt prefix
# rows stay byte-identical through load_weights_compat's vocab-expand
# (overlapping rows copied verbatim, new rows fresh-init); a new token
# fails loudly in encode() and must be added here deliberately. (The
# time/tuplet TYPES extend the closed sets _TIMES/_TUPLET_RATIOS
# define; they are spelled out here, not appended to those lists, so
# the mid-VOCAB runs do not shift and prefix byte-identity holds.)
_T32_ADD = (
    ["time-%s" % t for t in (
        "9-4", "4-16", "9-16", "7-4", "6-16",
        "8-16", "1-4", "12-4", "27-8",
    )]
    + ["tuplet-start-%d-%d" % r for r in (
        (11, 12), (10, 12), (16, 8), (4, 2), (10, 6), (18, 4),
        (6, 8), (24, 8), (7, 1), (35, 16),
    )]
)
assert all(
    t not in (
        _STEPS + _OCTAVES + _DURATIONS + _MARKS + _ALTERS + _SYMBOLS
        + _CLEFS + _KEYS + _TIMES + _TUPLETS + _BAR_REST + _BREVE + _CHORD
        + _VOICES + _SQ_ADD + _BEKERNPLUS
    ) for t in _T32_ADD
), "T3.2 add must be append-only (id stability)"
# Leland cull recovery (2026-11-22, 308 pages failed on 128th notes):
# the 128th duration TYPE. Lives AFTER _T32_ADD (same precedent: new
# rows at the end, pre-existing ids stay put, ckpt prefix rows stay
# byte-identical through load_weights_compat vocab-expand). The type
# is spelled out here, NOT appended to _DURATIONS, so the mid-VOCAB
# runs do not shift and prefix byte-identity holds.
_LELAND_ADD = ["128th"]
assert all(
    t not in (
        _STEPS + _OCTAVES + _DURATIONS + _MARKS + _ALTERS + _SYMBOLS
        + _CLEFS + _KEYS + _TIMES + _TUPLETS + _BAR_REST + _BREVE + _CHORD
        + _VOICES + _SQ_ADD + _BEKERNPLUS + _T32_ADD
    ) for t in _LELAND_ADD
), "leland add must be append-only (id stability)"

# Octave-transposing treble clefs (2026-11-24, suite31 luca-gloria
# Parts 2 & 3 carry <clef-octave-change>-1</clef-octave-change>, 61.5%
# of notes). Lives AFTER _LELAND_ADD so every pre-existing id 0..175
# stays put under saved checkpoints (vocab-expand copies prefix rows
# verbatim and fresh-inits the new rows).
_OCTAVE_CLEF_ADD = ["clef-G-2-8vb", "clef-G-2-8va"]
assert all(
    t not in (
        _STEPS + _OCTAVES + _DURATIONS + _MARKS + _ALTERS + _SYMBOLS
        + _CLEFS + _KEYS + _TIMES + _TUPLETS + _BAR_REST + _BREVE + _CHORD
        + _VOICES + _SQ_ADD + _BEKERNPLUS + _T32_ADD + _LELAND_ADD
    ) for t in _OCTAVE_CLEF_ADD
), "octave clef add must be append-only (id stability)"

# Option D1 Iter-7 vocab expansion (2026-11-25, full PDMX/Debussy/corpus
# census): 15 missing voices (voice-13..voice-27), 2 missing clefs
# (clef-T-2, clef-F-4-8vb), 1 missing octave (-1), 26 real meters, and
# 11 Debussy/PDMX tuplet ratios. Appended strictly AFTER _OCTAVE_CLEF_ADD
# so all pre-existing IDs 0..177 stay 100% verbatim under saved checkpoints.
_BASE_PRE_ITER7 = (
    _STEPS + _OCTAVES + _DURATIONS + _MARKS + _ALTERS + _SYMBOLS
    + _CLEFS + _KEYS + _TIMES + _TUPLETS + _BAR_REST + _BREVE + _CHORD
    + _VOICES + _SQ_ADD + _BEKERNPLUS + _T32_ADD + _LELAND_ADD
    + _OCTAVE_CLEF_ADD
)
_ITER7_D1_RAW = (
    ["voice-%d" % i for i in range(13, 28)]
    + ["clef-T-2", "clef-F-4-8vb"]
    + ["-1"]
    + ["time-%s" % t for t in (
        "2-1", "3-1", "1-8", "8-4", "11-8", "13-8", "13-16", "15-16",
        "16-4", "17-16", "3-16", "5-16", "7-16", "10-8", "10-4", "14-4",
        "18-4", "19-16", "2-16", "6-2", "22-8", "17-8", "11-16", "19-32",
        "21-16", "35-32",
    )]
    + ["tuplet-start-%d-%d" % r for r in (
        (12, 16), (17, 16), (17, 2), (18, 16), (8, 4), (9, 6), (3, 6),
        (6, 2), (9, 1), (10, 3), (12, 2),
    )]
)
_ITER7_D1_ADD: List[str] = list(
    dict.fromkeys(t for t in _ITER7_D1_RAW if t not in set(_BASE_PRE_ITER7))
)
assert all(
    t not in _BASE_PRE_ITER7 for t in _ITER7_D1_ADD
), "iter7 D1 add must be append-only (id stability)"

VOCAB: List[str] = _BASE_PRE_ITER7 + _ITER7_D1_ADD
TOKEN_TO_ID: Dict[str, int] = {t: i for i, t in enumerate(VOCAB)}
ID_TO_TOKEN: Dict[int, str] = {i: t for i, t in enumerate(VOCAB)}

# Special ids live after the vocabulary so appended vocab tokens never
# shift them relative to each other (model.py derives the same ids
# from ``vocab = len(VOCAB) + 3``).
PAD_ID: int = len(VOCAB)
BOS_ID: int = len(VOCAB) + 1
EOS_ID: int = len(VOCAB) + 2
VOCAB_SIZE: int = len(VOCAB) + 3

# ---------------------------------------------------------------------------
# Opt-in extension vocabularies (NOT appended to default VOCAB).
#
# Preserves VOCAB IDs 0..177 and len(VOCAB) == 178 byte-identically for
# existing checkpoints (prefix 0..175 unchanged) and respects Step 3's
# Gate-D boundary (no drum-kit / slash-note / tab tokens in Gate A-C
# training). Call ``build_vocab(...)`` when opting in to extended
# dynamics, lyric structural markers, or Gate-D percussion/tab notation.
# ---------------------------------------------------------------------------

EXT_DYNAMICS_VOCAB: List[str] = [
    "dyn-ppp", "dyn-fff", "dyn-fp", "dyn-sf", "dyn-sfp",
    "dyn-sffz", "dyn-rf", "dyn-rfz", "dyn-fz",
    "hairpin-continue",
]

EXT_LYRIC_MARKERS_VOCAB: List[str] = [
    "lyric-begin", "lyric-middle", "lyric-end", "lyric-single", "lyric-extend",
]

EXT_PERCUSSION_TAB_VOCAB: List[str] = (
    ["clef-TAB-5", "clef-TAB-4", "clef-TAB-6", "unpitched",
     "slash-start", "slash-stop"]
    + ["nh-%s" % nh for nh in ("x", "diamond", "slash", "triangle",
                               "circle-x", "normal")]
    + ["str-%d" % s for s in range(1, 7)]
    + ["fret-%d" % f for f in range(0, 25)]
)


def build_vocab(
    *,
    include_ext_dynamics: bool = False,
    include_lyric_markers: bool = False,
    include_percussion_tab: bool = False,
) -> tuple[List[str], Dict[str, int]]:
    """Build an append-only vocabulary preserving base ``VOCAB`` IDs 0..177.

    Default flags return an exact copy of ``VOCAB`` (len 178). Enabling any
    extension appends its tokens strictly after index 177 so existing
    checkpoints load cleanly via vocab-expand.
    """
    vocab = list(VOCAB)
    if include_ext_dynamics:
        vocab.extend(EXT_DYNAMICS_VOCAB)
    if include_lyric_markers:
        vocab.extend(EXT_LYRIC_MARKERS_VOCAB)
    if include_percussion_tab:
        vocab.extend(EXT_PERCUSSION_TAB_VOCAB)
    assert len(vocab) == len(set(vocab)), "extended vocab has duplicates"
    return vocab, {t: i for i, t in enumerate(vocab)}

# Expected 10-class auxiliary-box ontology. Mirrors dataset-pack's
# ``STRICT_DEFAULT_CLASSES``. The loader itself reads class names from
# whichever COCO file it is given (current ``det-data/coco.json`` still
# carries the older 7-class set); this list documents the target.
AUX_CLASSES: List[str] = [
    "NoteBlack", "Rest", "Beam", "Hook", "Accidental",
    "NoteDot", "Clef", "NoteVoid", "Stem", "Flag",
]

_SPLITS = ("train", "val", "test")


def encode(tokens: Sequence[str]) -> List[int]:
    """Map bekern tokens to vocabulary ids.

    Raises:
        ValueError: on the first token outside :data:`VOCAB`.
    """
    if isinstance(tokens, str):
        tokens = tokens.split()
    ids = []
    for tok in tokens:
        try:
            ids.append(TOKEN_TO_ID[tok])
        except KeyError:
            raise ValueError("unknown bekern token: %r" % (tok,)) from None
    return ids


def decode(ids: Sequence[int]) -> List[str]:
    """Map vocabulary ids back to bekern tokens.

    Raises:
        ValueError: on the first id outside ``range(len(VOCAB))``.
    """
    toks = []
    for i in ids:
        if not isinstance(i, int) or isinstance(i, bool):
            raise ValueError("token id must be int, got %r" % (i,))
        if not 0 <= i < len(VOCAB):
            raise ValueError("token id out of range: %r" % (i,))
        toks.append(VOCAB[i])
    return toks


# ---------------------------------------------------------------------------
# Piece keys (Python mirror of dataset-pack's ``piece_of``).
# ---------------------------------------------------------------------------

def _is_digits(s: str) -> bool:
    return bool(s) and all("0" <= c <= "9" for c in s)


def _strip_digit_suffix(s: str, marker: str) -> str:
    pos = s.rfind(marker)
    if pos > 0 and _is_digits(s[pos + len(marker):]):
        return s[:pos]
    return s


def piece_key(sample_id: str) -> str:
    """Map a manifest sample id to its piece key.

    Strips, in order: any ``parent/`` prefix, a trailing ``_degraded``,
    a ``_d<N>`` degrade variant, ``.aug<N>`` / ``_aug<N>`` aug variants,
    and a trailing ``-N`` page number. ``ds0001-1_d0`` -> ``ds0001``;
    ``sample01-1_degraded`` -> ``sample01``.
    """
    stem = sample_id.replace("\\", "/").rsplit("/", 1)[-1]
    if stem.endswith("_degraded"):
        stem = stem[: -len("_degraded")]
    stem = _strip_digit_suffix(stem, "_d")
    stem = _strip_digit_suffix(stem, ".aug")
    stem = _strip_digit_suffix(stem, "_aug")
    head, sep, tail = stem.rpartition("-")
    if sep and head and _is_digits(tail):
        stem = head
    return stem


# ---------------------------------------------------------------------------
# Item loading.
# ---------------------------------------------------------------------------

def load_bekern(path: PathLike) -> List[str]:
    """Read a ``<page-stem>-bekern.txt`` file (whitespace-separated tokens).

    Raises:
        ValueError: on the first token outside :data:`VOCAB`.
    """
    toks = Path(path).read_text(encoding="utf-8").split()
    for tok in toks:
        if tok not in TOKEN_TO_ID:
            raise ValueError("unknown bekern token %r in %s" % (tok, path))
    return toks


def load_sidecar(path: PathLike) -> Dict:
    """Read a deterministic GT coverage sidecar JSON file."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "schema_version" not in doc:
        raise ValueError("invalid GT coverage sidecar: %s" % (path,))
    return doc


def load_item(
    png_path: PathLike,
    bekern_path: PathLike,
    sidecar_path: PathLike | None = None,
) -> Dict:
    """Load one page: ``{"image", "tokens", "page_id"}`` (plus optional ``"sidecar"``).

    ``image`` is float32 HWC in [0, 1] (any PIL-readable mode is
    converted to RGB, so C is always 3). ``tokens`` is the validated
    bekern token list. ``page_id`` is the PNG stem. When ``sidecar_path``
    is provided, the parsed GT coverage sidecar JSON is attached under
    ``"sidecar"``.
    """
    png_path = Path(png_path)
    with Image.open(png_path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.ascontiguousarray(arr)
    item = {
        "image": arr,
        "tokens": load_bekern(bekern_path),
        "page_id": png_path.stem,
    }
    if sidecar_path is not None:
        item["sidecar"] = load_sidecar(sidecar_path)
    return item


# ---------------------------------------------------------------------------
# Manifest + COCO + split loading.
# ---------------------------------------------------------------------------

def _read_manifest(manifest_path: Path) -> List[Dict]:
    rows = []
    with open(manifest_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "%s:%d: bad JSON: %s" % (manifest_path, lineno, exc)
                ) from None
            if not isinstance(rec, dict) or "id" not in rec or "split" not in rec:
                raise ValueError(
                    "%s:%d: row needs 'id' and 'split' fields" % (manifest_path, lineno)
                )
            rows.append(rec)
    if not rows:
        raise ValueError("empty manifest: %s" % (manifest_path,))
    return rows


def _assert_piece_integrity(rows: List[Dict]) -> None:
    """Raise AssertionError if any piece key straddles splits."""
    by_piece: Dict[str, Dict[str, List[str]]] = {}
    for rec in rows:
        pid, split = str(rec["id"]), str(rec["split"])
        by_piece.setdefault(piece_key(pid), {}).setdefault(split, []).append(pid)
    bad = {p: s for p, s in by_piece.items() if len(s) > 1}
    if bad:
        parts = []
        for piece in sorted(bad):
            per = " | ".join(
                "%s: %s" % (sp, ",".join(sorted(ids)))
                for sp, ids in sorted(bad[piece].items())
            )
            parts.append("%s (%s)" % (piece, per))
        raise AssertionError(
            "piece-level split leakage: %d piece(s) straddle splits: %s"
            % (len(bad), "; ".join(parts))
        )


def _read_coco(coco_path: Path):
    doc = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if key not in doc:
            raise ValueError("COCO %s: missing '%s' table" % (coco_path, key))
    cat_name = {}
    for cat in doc["categories"]:
        cat_name[cat["id"]] = cat.get("name", "cat-%s" % (cat["id"],))
    by_id: Dict[int, Dict] = {}
    by_page: Dict[str, List[Dict]] = {}
    by_stem: Dict[str, List[Dict]] = {}
    for im in doc["images"]:
        by_id[im["id"]] = im
        if "page" in im:
            by_page.setdefault(str(im["page"]), []).append(im)
        stem = Path(str(im.get("file_name", ""))).stem
        by_stem.setdefault(stem, []).append(im)
    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in doc["annotations"]:
        if ann["image_id"] not in by_id:
            raise ValueError(
                "COCO %s: annotation %r points at unknown image_id %r"
                % (coco_path, ann.get("id"), ann["image_id"])
            )
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    return by_page, by_stem, anns_by_image, cat_name


def _boxes_for(rec: Dict, by_page, by_stem, anns_by_image, cat_name) -> List[Dict]:
    seen = set()
    imgs = []
    for im in by_page.get(str(rec["id"]), []):
        if im["id"] not in seen:
            seen.add(im["id"])
            imgs.append(im)
    for key in (Path(str(rec.get("image", ""))).stem, Path(str(rec["id"])).stem):
        for im in by_stem.get(key, []):
            if im["id"] not in seen:
                seen.add(im["id"])
                imgs.append(im)
    boxes = []
    for im in imgs:
        for ann in anns_by_image.get(im["id"], []):
            x, y, w, h = (float(v) for v in ann["bbox"])
            cid = ann["category_id"]
            boxes.append({
                "bbox": [x, y, x + w, y + h],
                "category_id": cid,
                "category": cat_name.get(cid, "cat-%s" % (cid,)),
            })
    return boxes


def load_split(
    manifest_path: PathLike, coco_path: PathLike, split: str
) -> Iterator[Dict]:
    """Return an iterator of ``{"image", "tokens", "boxes", "page_id"}``.

    ``manifest_path`` supplies sample ids + split assignment (its
    ``image`` field gives the page PNG path, resolved relative to the
    manifest directory unless absolute). Tokens come from the
    ``<page-stem>-bekern.txt`` file next to each image. Boxes are COCO
    annotations matched by ``page`` == manifest id, else by image-stem,
    converted from xywh to xyxy floats.

    Piece integrity is checked over the *whole* manifest at call time
    (before any row is returned): a straddling piece raises
    ``AssertionError``. A ``split`` name outside train/val/test raises
    ``ValueError``; a valid but absent split yields nothing.
    """
    if split not in _SPLITS:
        raise ValueError(
            "unknown split %r (expected one of %s)" % (split, "/".join(_SPLITS))
        )
    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    rows = _read_manifest(manifest_path)
    _assert_piece_integrity(rows)
    by_page, by_stem, anns_by_image, cat_name = _read_coco(Path(coco_path))

    def _iter() -> Iterator[Dict]:
        for rec in rows:
            if str(rec["split"]) != split:
                continue
            if "image" not in rec:
                raise ValueError("manifest row %r has no 'image' field" % (rec["id"],))
            img_path = Path(str(rec["image"]))
            if not img_path.is_absolute():
                img_path = base / img_path
            bekern_path = img_path.parent / (img_path.stem + "-bekern.txt")
            item = load_item(img_path, bekern_path)
            item["page_id"] = str(rec["id"])
            item["boxes"] = _boxes_for(rec, by_page, by_stem, anns_by_image, cat_name)
            yield item

    return _iter()

