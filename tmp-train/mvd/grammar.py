"""bekern grammar mask for constrained greedy decoding (MVD).

Mirrors ``converter/assembly-serialize/src/decode.rs`` (`decode_bekern`):
given a valid generated prefix, returns the set of token ids that can
extend it toward a decodable stream, plus whether EOS is legal here.
The mask may be STRICTER than decode (it need not cover every exotic
but-decodable stream); it must never block a ground-truth continuation
(pinned by ``test_grammar.py`` walking real bekern files).

Seam: if ``decode.rs`` event order changes, update ``_State`` together.
"""

from __future__ import annotations

import difflib
import hashlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

_T = data.TOKEN_TO_ID

DURATIONS = frozenset(
    ['whole', 'half', 'quarter', 'eighth', '16th', '32nd', '64th',
     '128th', 'breve'])
STEPS = frozenset(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
OCTAVES = frozenset(
    [str(n) for n in range(9)] + (['-1'] if '-1' in data.TOKEN_TO_ID else [])
)
ALTERS = frozenset(['a-2', 'a-1', 'a1', 'a2'])
DOT = 'dot'
TIE_START = 'tie-start'
TIE_STOP = 'tie-stop'
REST = 'rest'
BAR_REST = 'bar-rest'
CHORD = 'chord'
BAR = 'bar'
TUPLET_STOP = 'tuplet-stop'
# Includes octave-transposing treble clefs (`clef-G-2-8vb`, `clef-G-2-8va`)
# appended to `data.VOCAB`.
CLEFS = frozenset(t for t in data.VOCAB if t.startswith('clef-'))
KEYS = frozenset(t for t in data.VOCAB if t.startswith('key'))
TIMES = frozenset(t for t in data.VOCAB if t.startswith('time-'))
VOICES = frozenset(t for t in data.VOCAB if t.startswith('voice-'))
TUPLET_STARTS = frozenset(t for t in data.VOCAB if t.startswith('tuplet-start-'))
GRACE = 'grace'
ARTS = frozenset(t for t in data.VOCAB if t.startswith('art-'))
XSTAVES = frozenset(t for t in ('xstaff-up', 'xstaff-down') if t in data.TOKEN_TO_ID)
CUE = 'cue'
MODS = (
    frozenset(['slur-start', 'slur-stop', 'stacc', 'accent', 'tenuto', 'fermata'])
    | ARTS
    | XSTAVES
    | (frozenset([CUE]) if CUE in data.TOKEN_TO_ID else frozenset())
)
DYNS = frozenset(t for t in data.VOCAB if t.startswith('dyn-'))
HAIRS = frozenset(t for t in data.VOCAB if t.startswith('hairpin-'))
OTTAVAS = frozenset(t for t in data.VOCAB if t.startswith('ottava-'))
VOLTAS = frozenset(t for t in data.VOCAB if t.startswith('volta-'))
BAR_LEFT = frozenset(['bar-repeat-left'])
DIR_TOKENS = DYNS | HAIRS | BAR_LEFT | OTTAVAS | VOLTAS
STAFF_SEP = '<staff>'
TUP_OPEN = 'tup-open'
TUP_CLOSE = 'tup-close'
TUP_OPEN_DIGITS = frozenset(str(d) for d in range(1, 9))
BAR_RIGHT = frozenset(['bar-light-heavy', 'bar-repeat-right', 'bar-double'])
BAR_LIKES = frozenset([BAR]) | BAR_RIGHT

# bekern v2 optional hybrid compositional grammar tokens (present in v2 vocabs;
# absent from legacy v1 vocabs, where vocab_map checks keep 100% compatibility).
TIME_KW = 'time'
TUPLET_KW = 'tuplet-start'
MULTIREST_KW = 'multi-rest'
SLASH = '/'
DIGITS = frozenset(str(d) for d in range(10))
NONZERO_DIGITS = frozenset(str(d) for d in range(1, 10))
TIME_DEN_DIGITS = frozenset(['1', '2', '3', '4', '6', '8'])

# Opt-in Gate-D percussion/tab token families (gated behind
# ``PrefixState(allow_percussion_tab=True)``; never active in Gate A-C).
TAB_CLEFS = frozenset(['clef-TAB-5', 'clef-TAB-4', 'clef-TAB-6'])
UNPITCHED = 'unpitched'
SLASH_MARKS = frozenset(['slash-start', 'slash-stop'])
NOTEHEADS = frozenset(['nh-x', 'nh-diamond', 'nh-slash', 'nh-triangle',
                       'nh-circle-x', 'nh-normal'])
TAB_STRINGS = frozenset('str-%d' % s for s in range(1, 7))
TAB_FRETS = frozenset('fret-%d' % f for f in range(0, 25))

# Written duration values in quarter beats (mirror decode.rs
# written_dur) plus exact reduced integer rationals (num, den) for
# drift-free nested/high-order tuplet arithmetic (e.g. 13:8, 5:4 in 3:2).
_WRITTEN = {'breve': 8.0, 'whole': 4.0, 'half': 2.0, 'quarter': 1.0,
            'eighth': 0.5, '16th': 0.25, '32nd': 0.125, '64th': 0.0625,
            '128th': 0.03125}
_WRITTEN_RAT = {
    'breve': (8, 1), 'whole': (4, 1), 'half': (2, 1), 'quarter': (1, 1),
    'eighth': (1, 2), '16th': (1, 4), '32nd': (1, 8), '64th': (1, 16),
    '128th': (1, 32),
}
_BAR_EPS = 1e-6
MAX_TUPLET_VAL = 32
MAX_TXT_TOKENS = 24
MAX_TEXT_SPANS_PER_BAR = 4
TXT_OPEN = '<txt>'
TXT_CLOSE = '</txt>'
TEXT_TAG_OPENS = frozenset(['<txt>', '<lyr>', '<harm>', '<dir>', '<tempo>'])
_TEXT_FORBIDDEN = (
    BAR_LIKES
    | frozenset([STAFF_SEP, '<sys>', '<pad>', '<bos>', '<eos>', TXT_CLOSE])
    | TEXT_TAG_OPENS
)

# Per-staff u16 active-spanner bit layout for bekern-full numbered spanners
# (slur-1..4, hairpin-1..3, trem2). Legacy unnumbered slur-start/slur-stop
# and hairpin-* stay untouched for 100% v2a backward compatibility.
SPANNER_START_BIT = {
    'slur-1-start': 1 << 0,
    'slur-2-start': 1 << 1,
    'slur-3-start': 1 << 2,
    'slur-4-start': 1 << 3,
    'hairpin-1-start': 1 << 4,
    'hairpin-1-cresc-start': 1 << 4,
    'hairpin-1-dim-start': 1 << 4,
    'hairpin-2-start': 1 << 5,
    'hairpin-2-cresc-start': 1 << 5,
    'hairpin-2-dim-start': 1 << 5,
    'hairpin-3-start': 1 << 6,
    'hairpin-3-cresc-start': 1 << 6,
    'hairpin-3-dim-start': 1 << 6,
    'trem2-start': 1 << 7,
}
SPANNER_STOP_BIT = {
    'slur-1-stop': 1 << 0,
    'slur-2-stop': 1 << 1,
    'slur-3-stop': 1 << 2,
    'slur-4-stop': 1 << 3,
    'hairpin-1-stop': 1 << 4,
    'hairpin-2-stop': 1 << 5,
    'hairpin-3-stop': 1 << 6,
    'trem2-stop': 1 << 7,
}
SPANNER_MOD_TOKENS = frozenset(
    t for t in (*SPANNER_START_BIT.keys(), *SPANNER_STOP_BIT.keys())
    if t.startswith('slur-') or t.startswith('trem2-')
)
SPANNER_DIR_TOKENS = frozenset(
    t for t in (*SPANNER_START_BIT.keys(), *SPANNER_STOP_BIT.keys())
    if t.startswith('hairpin-')
)


def _rat_reduce(num, den):
    """Return reduced exact rational (num, den) with den > 0 via math.gcd."""
    if num == 0:
        return (0, 1)
    if den < 0:
        num = -num
        den = -den
    g = math.gcd(abs(num), den)
    return (num // g, den // g)


def _rat_mul(a, b):
    return _rat_reduce(a[0] * b[0], a[1] * b[1])


def _rat_add(a, b):
    return _rat_reduce(a[0] * b[1] + b[0] * a[1], a[1] * b[1])


def _rat_cmp(a, b):
    """Return <0, 0, or >0 comparing exact rationals a and b (den > 0)."""
    return a[0] * b[1] - b[0] * a[1]


def _dot_mult_rat(dots):
    """Exact rational multiplier for `dots` augmentation dots: (2^(d+1)-1) / 2^d."""
    return ((1 << (dots + 1)) - 1, 1 << dots)


def _float_to_rat(x):
    """Fallback conversion for floats into exact reduced rationals."""
    if x is None:
        return None
    # Powers of 2 up to 1/32 cover all standard durations/meters cleanly.
    den = 1024
    num = int(round(x * den))
    return _rat_reduce(num, den)


def _beats_rat_of(time_tok):
    """Exact rational quarter-beat measure duration for a ``time-B-T`` token."""
    try:
        _, b, t = time_tok.split('-')
        bi = int(b)
        ti = int(t)
        if ti <= 0:
            return None
        return _rat_reduce(bi * 4, ti)
    except (ValueError, ZeroDivisionError):
        return None
MAX_CHORDS_PER_ONSET = 8
MAX_CONSECUTIVE_GRACE = 8
MAX_CONSECUTIVE_BAR_REST = 24
MAX_CONSEC_DIR_TOKENS = 3
MAX_STAVES_PER_BAR = 24
MAX_MODS_PER_EVENT = 8
MAX_DOTS_PER_EVENT = 3
MAX_CONSEC_VOICE_SWITCHES = 2
MAX_ALTERS_LAX = 2
MAX_SYSTEMS = 24
MAX_TOKENS_PER_BAR = 288
MAX_BARS_PER_SYSTEM = 28
MAX_TOKENS_PER_SYSTEM = 1440
RECENT_WINDOW_MIN = 512
RECENT_WINDOW_KEEP = 576
RECENT_WINDOW_CAP = 640
BAR_HASH_CONSEC_LIMIT = 5
BAR_HASH_WINDOW_SIZE = 12
BAR_HASH_WINDOW_LIMIT = 8
MULTI_VOICE_BAR_HASH_CONSEC_LIMIT = 3
MAX_HEADER_STAVES_DEFAULT = 24
SINGLE_STAFF_HEADER_REPEAT_LIMIT = 2
SYS_DUP_SIM_THRESHOLD = 0.90
_COSMETIC_SYS_TOKENS = frozenset(DYNS) | frozenset(['slur-start', 'slur-stop'])
_COSMETIC_BAR_TOKENS = (
    frozenset(DYNS) | frozenset(CLEFS) | frozenset(KEYS) | frozenset(TIMES)
    | frozenset(['slur-start', 'slur-stop'])
)
LAX_BAR_MULT = 2.5
LAX_BAR_MIN_BEATS = 8.0
SHORT_PICKUP_TIMES = frozenset(['time-1-32', 'time-1-16'])


def _beats_of(time_tok):
    """Quarter-beat measure duration for a ``time-B-T`` token."""
    try:
        _, b, t = time_tok.split('-')
        return int(b) * 4.0 / int(t)
    except (ValueError, ZeroDivisionError):
        return None


class PrefixState:
    """Incremental prefix state for cached decoding (O(1) per step).

    Same transitions as the ``_walk`` + ``_State`` pair below (legacy
    ``strict=False`` is behavior-identical to :func:`allowed_next`),
    plus optional validator-predicate rules (``strict=True``) proven
    GT-safe by the full-corpus audit (``D:/tmp`` evidence scripts,
    2523 pages / 2.5M positions):

    * at most ONE alter per note, alter strictly before dots (decode.rs
      consumes a single optional alter before the dot loop; the corpus
      has 0 double-alters and 0 alter-after-dot);
    * voice markers open sequentially (voice-1 free; voice-N needs
      N-1 or N already opened since the last bar — mirrors the
      appearance-order-per-measure rule; corpus has 0 non-sequential
      opens; the voice-1..12 range itself caps N at the corpus max);
    * no consecutive duplicate ``bar`` / ``voice-N`` markers (decode
      no-ops on them; corpus has 0; they are the tight-loop shapes
      ``voice-4 voice-4 voice-4`` seen in failed hyps).
    * bar-duration conservation (T2.3): the active time signature
      (``time-B-T`` -> ``B * 4 / T`` quarter beats) and a running
      per-voice cursor forbid note/rest durations that overshoot the
      remaining measure duration (chord tones add 0; ``bar-rest``
      jumps its voice to the barline; tuplet spans scale by N/A and
      dots count as ``2 - 2**-dots`` — mirroring ``decode.rs``).
      Under-full (partial/pickup) bars — including anacrusis bars,
      mid-bar split repeats closed by ``bar-repeat-right`` or opened
      by ``bar-repeat-left``, and 1-note pickup meters (``time-1-16``,
      ``time-1-32``) — stay legal and are never masked out. Unlike the rules
      above this one is NOT proven GT-safe: the audit found genuine
      GT overlaps (voices the serializer clamp-repairs), so the
      full-corpus strict round-trip flags those pages while legacy
      ``strict=False`` stays the GT-safe mask.

    Deliberately NOT gated (audit-forced omissions, see module doc
    additions below): tie-stop keying (529 GT
    dangling stops incl. 100 with other chains open — decode stays
    literal, validators never error on ties), global trigram blocking
    (70.8% of GT positions complete a seen trigram — music repeats;
    tight loops are caught by the decoder-side run/cycle caps instead).
    """

    __slots__ = ('mode', 'header_open', 'seg_last', 'seg', 'depth',
                 'pending', 'strict', 'broken',
                 'run_tok', 'run_len', 'alter_seen', 'dot_seen',
                 'opened', 'recent',
                 'tsig', 'tsig_rat', 'pending_tsig', 'pending_tsig_rat',
                 'tsig_by_staff', 'tsig_rat_by_staff', 'pending_tsig_by_staff',
                 'vcurs', 'vcurs_rat', 'vscales',
                 'pend_spans', 'max_tuplet_depth',
                 'ev_base', 'ev_base_rat', 'ev_dots', 'ev_chord', 'ev_grace',
                 'ev_mods', 'ev_alters',
                 'chords_on_onset', 'grace_run',
                 'consec_bar_rest_bars', 'bar_has_pitched', 'bar_has_bar_rest',
                 'seg_has_bar_rest', 'staves_in_bar',
                 'dir_run', 'last_dir_tok',
                 'midbar_clefs', 'midbar_keys', 'midbar_times',
                 'voice_switch_run',
                 'sys_count', 'sys_has_event',
                 'bars_in_system', 'tokens_in_system',
                 'max_bars_per_system', 'max_tokens_per_system',
                 'tokens_since_bar', 'cur_sys_tokens', 'completed_system_hashes',
                 'completed_systems',
                 'cur_bar_tokens', 'sys_bar_hashes', 'bar_repeat_forbidden_tok',
                 'last_mv_bar_canon', 'mv_bar_consec',
                 'last_was_time', 'header_clefs', 'header_keys', 'header_times',
                 'first_sys_staves', 'cur_hdr_tuple', 'completed_hdr_tuples',
                 'comp_num', 'comp_digits', 'comp_val',
                 'in_text_span', 'txt_toks', 'bar_txt_spans',
                 'staff_spanners', 'onset_spanner_starts',
                 'allow_percussion_tab', 'vocab_map')

    def __init__(
        self,
        strict=False,
        allow_percussion_tab=False,
        vocab_map=None,
        max_bars_per_system=MAX_BARS_PER_SYSTEM,
        max_tokens_per_system=MAX_TOKENS_PER_SYSTEM,
        max_tuplet_depth=1,
    ):
        self.mode = 'bound'
        self.header_open = True
        self.seg_last = [False]
        self.seg = 0
        self.depth = [0]
        self.pending = 0
        self.strict = strict
        self.broken = False
        self.run_tok = None
        self.run_len = 0
        self.alter_seen = False
        self.dot_seen = False
        self.opened = set()
        self.recent = []
        # T2.3 bar-duration books (kept in both modes so copy() stays
        # uniform; strict=True caps at tsig, strict=False enforces a
        # generous safety ceiling max(8.0, 2.5 * tsig)): active
        # time signature in quarter beats (None until seen),
        # per-staff polymeter time signatures, per-voice running cursors
        # (both float and exact reduced integer rationals), per-voice open
        # tuplet ratio stacks, queued tuplet starts, and the pending event's
        # written base quarters + dot count + chord flag.
        self.tsig = None
        self.tsig_rat = None
        # Courtesy time-sig stashed mid-measure; applied at the next
        # barline (a courtesy change must not retune the running
        # books, but must take effect for the new measure).
        self.pending_tsig = None
        self.pending_tsig_rat = None
        self.tsig_by_staff = {}
        self.tsig_rat_by_staff = {}
        self.pending_tsig_by_staff = {}
        self.vcurs = [0.0]
        self.vcurs_rat = [(0, 1)]
        self.vscales = [[]]
        self.pend_spans = []
        self.max_tuplet_depth = max(1, int(max_tuplet_depth))
        self.ev_base = None
        self.ev_base_rat = None
        self.ev_dots = 0
        self.ev_chord = False
        self.ev_grace = False
        self.ev_mods = 0
        self.ev_alters = 0
        self.chords_on_onset = 0
        self.grace_run = 0
        self.consec_bar_rest_bars = 0
        self.bar_has_pitched = False
        self.bar_has_bar_rest = False
        self.seg_has_bar_rest = False
        self.staves_in_bar = 0
        self.dir_run = 0
        self.last_dir_tok = None
        self.midbar_clefs = 0
        self.midbar_keys = 0
        self.midbar_times = 0
        self.voice_switch_run = 0
        self.sys_count = 0
        self.sys_has_event = False
        self.bars_in_system = 0
        self.tokens_in_system = 0
        self.max_bars_per_system = max_bars_per_system
        self.max_tokens_per_system = max_tokens_per_system
        self.tokens_since_bar = 0
        self.cur_sys_tokens = []
        self.completed_system_hashes = {}
        self.completed_systems = []
        self.cur_bar_tokens = []
        self.sys_bar_hashes = []
        self.bar_repeat_forbidden_tok = None
        self.last_mv_bar_canon = None
        self.mv_bar_consec = 0
        self.last_was_time = False
        self.header_clefs = 0
        self.header_keys = 0
        self.header_times = 0
        self.first_sys_staves = 0
        self.cur_hdr_tuple = None
        self.completed_hdr_tuples = []
        # bekern v2 compositional sub-state counters (flat ints for cheap copy())
        self.comp_num = 0
        self.comp_digits = 0
        self.comp_val = 0
        # bekern-full opt-in text-span anti-loop state
        self.in_text_span = False
        self.txt_toks = 0
        self.bar_txt_spans = 0
        # bekern-full per-staff u16 active-spanner bitmasks + per-onset start guard
        self.staff_spanners = [0]
        self.onset_spanner_starts = 0
        # Gate-D opt-in: percussion/tab grammar transitions (OFF by default).
        self.allow_percussion_tab = bool(allow_percussion_tab)
        if vocab_map is not None:
            self.vocab_map = vocab_map
        elif self.allow_percussion_tab:
            _, self.vocab_map = data.build_vocab(include_percussion_tab=True)
        else:
            self.vocab_map = _T

    def copy(self):
        """Cheap per-beam copy (all fields are small scalars/lists)."""
        c = PrefixState.__new__(PrefixState)
        c.mode = self.mode
        c.header_open = self.header_open
        c.seg_last = list(self.seg_last)
        c.seg = self.seg
        c.depth = list(self.depth)
        c.pending = self.pending
        c.strict = self.strict
        c.broken = self.broken
        c.run_tok = self.run_tok
        c.run_len = self.run_len
        c.alter_seen = self.alter_seen
        c.dot_seen = self.dot_seen
        c.opened = set(self.opened)
        c.recent = list(self.recent)
        c.tsig = self.tsig
        c.tsig_rat = self.tsig_rat
        c.pending_tsig = self.pending_tsig
        c.pending_tsig_rat = self.pending_tsig_rat
        c.tsig_by_staff = dict(self.tsig_by_staff)
        c.tsig_rat_by_staff = dict(self.tsig_rat_by_staff)
        c.pending_tsig_by_staff = dict(self.pending_tsig_by_staff)
        c.vcurs = list(self.vcurs)
        c.vcurs_rat = list(self.vcurs_rat)
        c.vscales = [list(s) for s in self.vscales]
        c.pend_spans = list(self.pend_spans)
        c.max_tuplet_depth = self.max_tuplet_depth
        c.ev_base = self.ev_base
        c.ev_base_rat = self.ev_base_rat
        c.ev_dots = self.ev_dots
        c.ev_chord = self.ev_chord
        c.ev_grace = self.ev_grace
        c.ev_mods = self.ev_mods
        c.ev_alters = self.ev_alters
        c.chords_on_onset = self.chords_on_onset
        c.grace_run = self.grace_run
        c.consec_bar_rest_bars = self.consec_bar_rest_bars
        c.bar_has_pitched = self.bar_has_pitched
        c.bar_has_bar_rest = self.bar_has_bar_rest
        c.seg_has_bar_rest = self.seg_has_bar_rest
        c.staves_in_bar = self.staves_in_bar
        c.dir_run = self.dir_run
        c.last_dir_tok = self.last_dir_tok
        c.midbar_clefs = self.midbar_clefs
        c.midbar_keys = self.midbar_keys
        c.midbar_times = self.midbar_times
        c.voice_switch_run = self.voice_switch_run
        c.sys_count = self.sys_count
        c.sys_has_event = self.sys_has_event
        c.bars_in_system = self.bars_in_system
        c.tokens_in_system = self.tokens_in_system
        c.max_bars_per_system = self.max_bars_per_system
        c.max_tokens_per_system = self.max_tokens_per_system
        c.tokens_since_bar = self.tokens_since_bar
        c.cur_sys_tokens = list(self.cur_sys_tokens)
        c.completed_system_hashes = dict(self.completed_system_hashes)
        c.completed_systems = [list(s) for s in self.completed_systems]
        c.cur_bar_tokens = list(self.cur_bar_tokens)
        c.sys_bar_hashes = list(self.sys_bar_hashes)
        c.bar_repeat_forbidden_tok = self.bar_repeat_forbidden_tok
        c.last_mv_bar_canon = list(self.last_mv_bar_canon) if self.last_mv_bar_canon is not None else None
        c.mv_bar_consec = self.mv_bar_consec
        c.last_was_time = self.last_was_time
        c.header_clefs = self.header_clefs
        c.header_keys = self.header_keys
        c.header_times = self.header_times
        c.first_sys_staves = self.first_sys_staves
        c.cur_hdr_tuple = list(self.cur_hdr_tuple) if self.cur_hdr_tuple is not None else None
        c.completed_hdr_tuples = list(self.completed_hdr_tuples)
        c.comp_num = self.comp_num
        c.comp_digits = self.comp_digits
        c.comp_val = self.comp_val
        c.in_text_span = self.in_text_span
        c.txt_toks = self.txt_toks
        c.bar_txt_spans = self.bar_txt_spans
        c.staff_spanners = list(self.staff_spanners)
        c.onset_spanner_starts = self.onset_spanner_starts
        c.allow_percussion_tab = self.allow_percussion_tab
        c.vocab_map = self.vocab_map
        return c

    @staticmethod
    def _hash_tokens(toks):
        return hashlib.sha1(' '.join(toks).encode('utf-8')).hexdigest()[:16]

    def _canonical_system_tokens(self):
        return [t for t in self.cur_sys_tokens if t not in _COSMETIC_SYS_TOKENS]

    def _canonical_system_hash(self):
        canon = self._canonical_system_tokens()
        if not canon:
            return None
        return self._hash_tokens(canon)

    def _is_near_dup_system(self, cur_canon):
        if len(cur_canon) < 8 or not any(t in STEPS for t in cur_canon):
            return False
        n_cur = len(cur_canon)
        for prev_canon in self.completed_systems:
            n_prev = len(prev_canon)
            if n_prev < 8:
                continue
            hi = n_cur if n_cur >= n_prev else n_prev
            lo = n_prev if n_cur >= n_prev else n_cur
            if (float(lo) / float(hi)) < SYS_DUP_SIM_THRESHOLD:
                continue
            if difflib.SequenceMatcher(None, cur_canon, prev_canon, autojunk=False).ratio() >= SYS_DUP_SIM_THRESHOLD:
                return True
        return False

    def has_open_tuplet(self):
        """Return True iff any tuplet-start/tup-open is queued (pending), mid-ratio, or open (depth > 0)."""
        return (
            self.pending > 0
            or any(d > 0 for d in self.depth)
            or self.mode in ('tup-open-a', 'tup-open-n', 'tuplet-a', 'tuplet-n')
        )

    def _reset_event_guards(self):
        """Reset zero-cursor anti-loop counters on musical/structural progress."""
        self.dir_run = 0
        self.last_dir_tok = None
        self.midbar_clefs = 0
        self.midbar_keys = 0
        self.midbar_times = 0
        self.voice_switch_run = 0

    def _dir_token_ok(self, tok):
        """Anti-loop guard for direction/span/bar-left tokens (Bug A).

        Forbids consecutive identical direction/span/bar-left tokens (e.g.
        `hairpin-stop hairpin-stop` or `dyn-pp dyn-pp`) and caps consecutive
        direction tokens without an intervening note/rest/bar/<staff>/<sys>
        at ``MAX_CONSEC_DIR_TOKENS`` (3).
        """
        if self.dir_run >= MAX_CONSEC_DIR_TOKENS:
            return False
        if self.last_dir_tok == tok:
            return False
        if self.recent and self.recent[-1] == tok:
            return False
        return True

    def _header_staff_cap(self):
        return max(self.first_sys_staves, MAX_HEADER_STAVES_DEFAULT)

    def _single_staff_hdr_repeat_capped(self):
        if not self.header_open or self.first_sys_staves != 1:
            return False
        tuples = list(self.completed_hdr_tuples)
        if self.cur_hdr_tuple is not None:
            tuples.append(tuple(self.cur_hdr_tuple))
        if len(tuples) < SINGLE_STAFF_HEADER_REPEAT_LIMIT:
            return False
        tail = tuples[-SINGLE_STAFF_HEADER_REPEAT_LIMIT:]
        return all(x == tail[0] for x in tail[1:])

    def _header_clef_ok(self):
        if not self.header_open:
            return self.midbar_clefs < 1
        if self.header_clefs >= self._header_staff_cap():
            return False
        if self._single_staff_hdr_repeat_capped():
            return False
        return True

    def _header_key_ok(self):
        if not self.header_open:
            return self.midbar_keys < 1
        if self.header_keys >= self._header_staff_cap():
            return False
        if self._single_staff_hdr_repeat_capped():
            return False
        return True

    def _header_time_ok(self):
        if self.last_was_time:
            return False
        if not self.header_open:
            return self.midbar_times < 1
        if self.header_times >= self._header_staff_cap():
            return False
        if self._single_staff_hdr_repeat_capped():
            return False
        return True

    @staticmethod
    def _is_pure_rest_bar(bar_toks):
        return not any(t in STEPS for t in bar_toks)

    def _record_completed_bar(self):
        if not self.cur_bar_tokens or self._is_pure_rest_bar(self.cur_bar_tokens):
            self.sys_bar_hashes.append(None)
            self.bar_repeat_forbidden_tok = None
            self.last_mv_bar_canon = None
            self.mv_bar_consec = 0
            return
        canon = [t for t in self.cur_bar_tokens if t not in _COSMETIC_BAR_TOKENS]
        if not canon:
            self.sys_bar_hashes.append(None)
            self.bar_repeat_forbidden_tok = None
            self.last_mv_bar_canon = None
            self.mv_bar_consec = 0
            return
        bh = self._hash_tokens(canon)
        self.sys_bar_hashes.append(bh)
        consec = 0
        for prev in reversed(self.sys_bar_hashes):
            if prev == bh:
                consec += 1
            else:
                break
        win = self.sys_bar_hashes[-BAR_HASH_WINDOW_SIZE:]
        win_count = sum(1 for x in win if x == bh)
        if consec >= BAR_HASH_CONSEC_LIMIT or win_count >= BAR_HASH_WINDOW_LIMIT:
            self.bar_repeat_forbidden_tok = canon[0]
        else:
            self.bar_repeat_forbidden_tok = None

        # Multi-voice bar consecutive repetition tracker (>=2 distinct voices, pitched).
        distinct_voices = {t for t in canon if t in VOICES}
        if len(distinct_voices) >= 2:
            mv_canon = _canonicalize_single_measure(canon)
            if mv_canon == self.last_mv_bar_canon:
                self.mv_bar_consec += 1
            else:
                self.last_mv_bar_canon = mv_canon
                self.mv_bar_consec = 1
        else:
            self.last_mv_bar_canon = None
            self.mv_bar_consec = 0

    def can_emit_sys(self):
        """Return True iff the current system has >=1 musical event, no open tuplet/text span,
        page is under MAX_SYSTEMS, and the current system does not duplicate a previously completed system.

        NOTE: Active spanners (`staff_spanners`) deliberately NEVER block `<sys>` so
        system/page-crossing slurs, hairpins, and tremolos stay legal.
        """
        if not self.sys_has_event or self.sys_count >= MAX_SYSTEMS - 1:
            return False
        if self.has_open_tuplet() or self.in_text_span:
            return False
        cur_canon = self._canonical_system_tokens()
        if not cur_canon:
            return False
        cur_h = self._hash_tokens(cur_canon)
        if self.completed_system_hashes.get(cur_h, 0) >= 1:
            return False
        return True

    def _sys_is_capped(self):
        """True if the current system hit MAX_BARS_PER_SYSTEM or MAX_TOKENS_PER_SYSTEM."""
        if (
            self.max_bars_per_system is not None
            and self.bars_in_system >= self.max_bars_per_system
        ):
            return True
        if (
            self.max_tokens_per_system is not None
            and self.tokens_in_system >= self.max_tokens_per_system
        ):
            return True
        return False

    def can_emit_eos(self, cur_len=None, max_len=None):
        """Return True iff <eos> is legal here (guards against empty trailing systems,
        unclosed tuplets, and unclosed text spans; NEVER blocks on page-crossing spanners).
        """
        if cur_len is not None and max_len is not None and cur_len >= max_len:
            return True
        if self.has_open_tuplet() or self.in_text_span:
            return False
        if (self.sys_count + 1) >= MAX_SYSTEMS:
            return True
        if self._sys_is_capped():
            return True
        return bool(self.sys_has_event)

    def reset_system(self):
        """Advance to a new system on ``<sys>`` while preserving cross-system cycle history
        and cross-system spanner / polymeter state.
        """
        if self.sys_has_event:
            cur_canon = self._canonical_system_tokens()
            if cur_canon:
                cur_h = self._hash_tokens(cur_canon)
                self.completed_system_hashes[cur_h] = (
                    self.completed_system_hashes.get(cur_h, 0) + 1
                )
                self.completed_systems.append(cur_canon)
        self.sys_count += 1
        self.sys_has_event = False
        self.bars_in_system = 0
        self.tokens_in_system = 0
        self.tokens_since_bar = 0
        self.cur_sys_tokens = []
        self.cur_bar_tokens = []
        self.sys_bar_hashes = []
        self.bar_repeat_forbidden_tok = None
        self.last_mv_bar_canon = None
        self.mv_bar_consec = 0
        self.last_was_time = False
        self.header_clefs = 0
        self.header_keys = 0
        self.header_times = 0
        self.cur_hdr_tuple = None
        self.completed_hdr_tuples = []
        self.mode = 'bound'
        self.header_open = True
        self.seg_last = [False]
        self.seg = 0
        self.depth = [0]
        self.pending = 0
        self.broken = False
        self.run_tok = None
        self.run_len = 0
        self.alter_seen = False
        self.dot_seen = False
        self.opened = set()
        self.vcurs = [0.0]
        self.vcurs_rat = [(0, 1)]
        self.vscales = [[]]
        self.pend_spans = []
        self.ev_base = None
        self.ev_base_rat = None
        self.ev_dots = 0
        self.ev_chord = False
        self.ev_grace = False
        self.ev_mods = 0
        self.ev_alters = 0
        self.chords_on_onset = 0
        self.grace_run = 0
        self.bar_has_pitched = False
        self.bar_has_bar_rest = False
        self.seg_has_bar_rest = False
        self.staves_in_bar = 0
        self._reset_event_guards()
        self.comp_num = 0
        self.comp_digits = 0
        self.comp_val = 0
        self.in_text_span = False
        self.txt_toks = 0
        self.bar_txt_spans = 0
        self.onset_spanner_starts = 0
        self.recent.append('<sys>')
        if len(self.recent) > RECENT_WINDOW_KEEP:
            del self.recent[:-RECENT_WINDOW_KEEP]

    def _grow(self, n):
        while len(self.seg_last) < n:
            self.seg_last.append(False)
            self.depth.append(0)
            self.vcurs.append(0.0)
            self.vcurs_rat.append((0, 1))
            self.vscales.append([])

    def _ensure_staff_spanner(self, staff_idx=None):
        idx = self.staves_in_bar if staff_idx is None else staff_idx
        while len(self.staff_spanners) <= idx:
            self.staff_spanners.append(0)
        return idx

    def _spanner_ok(self, tok):
        """Return True iff spanner start/stop token is legal on current staff & onset."""
        bit = SPANNER_START_BIT.get(tok)
        if bit is not None:
            idx = self._ensure_staff_spanner()
            if (self.staff_spanners[idx] & bit) or (self.onset_spanner_starts & bit):
                return False
            return True
        # Spanner stops are always legal (supports page-crossing spanners started on prev page).
        return True

    def _apply_spanner(self, tok):
        """Update per-staff u16 active-spanner bitmask and per-onset duplicate guard."""
        bit = SPANNER_START_BIT.get(tok)
        if bit is not None:
            idx = self._ensure_staff_spanner()
            if (self.staff_spanners[idx] & bit) or (self.onset_spanner_starts & bit):
                raise ValueError('duplicate or already-active spanner start %r' % tok)
            self.staff_spanners[idx] = (self.staff_spanners[idx] | bit) & 0xFFFF
            self.onset_spanner_starts = (self.onset_spanner_starts | bit) & 0xFFFF
            return
        stop_bit = SPANNER_STOP_BIT.get(tok)
        if stop_bit is not None:
            idx = self._ensure_staff_spanner()
            self.staff_spanners[idx] = (self.staff_spanners[idx] & ~stop_bit) & 0xFFFF

    def _open_scale_rat(self):
        """Exact reduced rational pending-event scale: open spans x queued starts (N/A each)."""
        num, den = 1, 1
        for a, n in self.vscales[self.seg]:
            num *= n
            den *= a
            g = math.gcd(num, den)
            num //= g
            den //= g
        for a, n in self.pend_spans:
            num *= n
            den *= a
            g = math.gcd(num, den)
            num //= g
            den //= g
        return (num, den)

    def _open_scale(self):
        """Float view of :meth:`_open_scale_rat` (preserved for backward-compatible callers)."""
        num, den = self._open_scale_rat()
        return num / den

    def _drained_scale_rat(self):
        """Exact reduced rational scale after draining pending starts into this segment."""
        self.vscales[self.seg].extend(self.pend_spans)
        del self.pend_spans[:]
        num, den = 1, 1
        for a, n in self.vscales[self.seg]:
            num *= n
            den *= a
            g = math.gcd(num, den)
            num //= g
            den //= g
        return (num, den)

    def _drained_scale(self):
        """Float view of :meth:`_drained_scale_rat`."""
        num, den = self._drained_scale_rat()
        return num / den

    def _pending_value_rat(self):
        """Exact rational projected quarters of the still-open event ((0, 1) if none)."""
        if self.ev_base_rat is None or self.ev_chord or self.ev_grace:
            return (0, 1)
        return _rat_mul(
            _rat_mul(self.ev_base_rat, _dot_mult_rat(self.ev_dots)),
            self._open_scale_rat(),
        )

    def _pending_value(self):
        """Float view of :meth:`_pending_value_rat`."""
        num, den = self._pending_value_rat()
        return num / den

    def _active_tsig_rat(self):
        """Active exact rational time signature for the current staff (polymeter-aware)."""
        r = self.tsig_rat_by_staff.get(self.staves_in_bar)
        if r is not None:
            return r
        if self.tsig_rat is not None:
            return self.tsig_rat
        if self.tsig is not None:
            return _float_to_rat(self.tsig)
        return None

    def _active_tsig(self):
        """Float view of :meth:`_active_tsig_rat`."""
        r = self._active_tsig_rat()
        return None if r is None else (r[0] / r[1])

    def _beat_limit_rat(self):
        """Active exact rational per-voice beat ceiling (strict=tsig, lax=max(8, 5/2*tsig))."""
        ts = self._active_tsig_rat()
        if ts is None:
            return None
        if self.strict:
            return ts
        scaled = _rat_mul((5, 2), ts)
        return scaled if _rat_cmp(scaled, (8, 1)) >= 0 else (8, 1)

    def _beat_limit(self):
        """Float view of :meth:`_beat_limit_rat`."""
        r = self._beat_limit_rat()
        return None if r is None else (r[0] / r[1])

    def _dot_overshoots(self):
        """True if the open event already exceeds the active beat ceiling (exact rational)."""
        limit = self._beat_limit_rat()
        if limit is None or self.ev_chord or self.ev_grace or self.ev_base_rat is None:
            return False
        total = _rat_add(self.vcurs_rat[self.seg], self._pending_value_rat())
        return _rat_cmp(total, limit) > 0

    def _dot_ok(self):
        """True if one more dot keeps the open event within the beat ceiling (exact rational)."""
        if self.ev_dots >= MAX_DOTS_PER_EVENT:
            return False
        limit = self._beat_limit_rat()
        if limit is None:
            return True
        if self.ev_chord or self.ev_grace or self.ev_base_rat is None:
            return True
        val = _rat_mul(
            _rat_mul(self.ev_base_rat, _dot_mult_rat(self.ev_dots + 1)),
            self._open_scale_rat(),
        )
        total = _rat_add(self.vcurs_rat[self.seg], val)
        return _rat_cmp(total, limit) <= 0

    def _blocks_duration(self, base):
        """True if undotted ``base`` (exact rational tuple or float) would overshoot the beat ceiling."""
        limit = self._beat_limit_rat()
        if limit is None:
            return False
        base_rat = base if isinstance(base, tuple) else _float_to_rat(base)
        scale_rat = self._open_scale_rat()
        ev_rat = _rat_mul(base_rat, scale_rat)
        if self.strict and round((ev_rat[0] * 480.0) / ev_rat[1]) <= 0:
            return True
        used = _rat_add(self.vcurs_rat[self.seg], self._pending_value_rat())
        total = _rat_add(used, ev_rat)
        return _rat_cmp(total, limit) > 0

    def _bar_is_full(self):
        """True if no further timed event can fit in the current voice/bar (exact rational)."""
        limit = self._beat_limit_rat()
        if limit is None:
            return False
        used = _rat_add(self.vcurs_rat[self.seg], self._pending_value_rat())
        if self.strict:
            return _rat_cmp(used, limit) >= 0
        # In lax mode, full only once even the shortest duration (128th)
        # would exceed the generous safety ceiling.
        min_ev = _rat_mul(_WRITTEN_RAT['128th'], self._open_scale_rat())
        return _rat_cmp(_rat_add(used, min_ev), limit) > 0

    def _short_pickup_time_ok(self, prev_tok=None, recent_before=None):
        """Allow time-1-32 / time-1-16 in header, after a bar*/<staff> token (allowing clef*/key* between), or when already in a short pickup bar."""
        if self.header_open:
            return True
        ts = self._active_tsig_rat()
        if ts is not None and _rat_cmp(ts, (1, 4)) <= 0:
            return True
        if self.tsig_rat is not None and _rat_cmp(self.tsig_rat, (1, 4)) <= 0:
            return True
        if recent_before is not None:
            for tok in reversed(recent_before):
                if tok in CLEFS or tok in KEYS:
                    continue
                return tok in BAR_LIKES or tok in BAR_LEFT or tok == STAFF_SEP
            return True
        return prev_tok is not None and (
            prev_tok in BAR_LIKES or prev_tok in BAR_LEFT or prev_tok == STAFF_SEP
        )

    def _header_attr_ok(self):
        """Allow key*/time-* at system start, bar start, or full-bar end (courtesy change).

        In lax mode (strict=False), also permits legitimate mid-bar key*/time-* changes.
        In strict mode, forbids stray mid-bar key*/time-* emitted after a note/rest inside
        an incomplete bar (0 < cur_used < tsig), while preserving opening headers,
        post-barline changes, and full-bar courtesy changes.
        """
        if not self.strict or self.header_open:
            return True
        cur_used = _rat_add(self.vcurs_rat[self.seg], self._pending_value_rat())
        if cur_used[0] <= 0:
            return True
        ts = self._active_tsig_rat()
        if ts is not None and _rat_cmp(cur_used, ts) >= 0:
            return True
        return False

    def _drain(self):
        self.depth[self.seg] += self.pending
        self.pending = 0

    def _apply_tsig(self, beats, beats_rat=None):
        if beats_rat is None and beats is not None:
            beats_rat = _float_to_rat(beats)
        if beats is None and beats_rat is not None:
            beats = beats_rat[0] / beats_rat[1]
        staff_idx = max(0, self.header_times - 1) if self.header_open else self.staves_in_bar
        if all(v[0] == 0 for v in self.vcurs_rat):
            self.tsig = beats
            self.tsig_rat = beats_rat
            self.pending_tsig = None
            self.pending_tsig_rat = None
            if beats is not None and beats_rat is not None:
                self.tsig_by_staff[staff_idx] = beats
                self.tsig_rat_by_staff[staff_idx] = beats_rat
                self.pending_tsig_by_staff.pop(staff_idx, None)
        else:
            self.pending_tsig = beats
            self.pending_tsig_rat = beats_rat
            if beats is not None and beats_rat is not None:
                self.pending_tsig_by_staff[staff_idx] = (beats, beats_rat)

    def _bound(self, t):
        if t == TIME_KW:
            if not self._header_attr_ok():
                raise ValueError('spurious mid-bar header %r' % t)
            if not self._header_time_ok():
                raise ValueError('consecutive or capped time header %r' % t)
            if self.header_open:
                if self.cur_hdr_tuple is None:
                    self.cur_hdr_tuple = [None, None, t]
                else:
                    self.cur_hdr_tuple[2] = t
                self.header_times += 1
            else:
                self.midbar_times += 1
            self.last_was_time = False
            self.comp_num = 0
            self.comp_digits = 0
            self.comp_val = 0
            return 'time-num'
        if t in CLEFS or t in KEYS or t in TIMES or (
                self.allow_percussion_tab and t in TAB_CLEFS):
            # Mid-stream attribute changes (bekern+): clef/key/time
            # may appear at any measure start (or courtesy end of a full
            # measure). Forbid stray key*/time-* inside an incomplete bar,
            # consecutive time-* tokens, runaway repeated system headers,
            # and repeated mid-bar clef/key/time without intervening events.
            if (t in KEYS or t in TIMES) and not self._header_attr_ok():
                raise ValueError('spurious mid-bar header %r' % t)
            if t in CLEFS or (self.allow_percussion_tab and t in TAB_CLEFS):
                if not self._header_clef_ok():
                    raise ValueError('header/mid-bar clef cap exceeded %r' % t)
                if self.header_open:
                    if self.cur_hdr_tuple is not None:
                        self.completed_hdr_tuples.append(tuple(self.cur_hdr_tuple))
                    self.cur_hdr_tuple = [t, None, None]
                    self.header_clefs += 1
                else:
                    self.midbar_clefs += 1
                self.last_was_time = False
                return 'bound'
            if t in KEYS:
                if not self._header_key_ok():
                    raise ValueError('header/mid-bar key cap exceeded %r' % t)
                if self.header_open:
                    if self.cur_hdr_tuple is None:
                        self.cur_hdr_tuple = [None, t, None]
                    else:
                        self.cur_hdr_tuple[1] = t
                    self.header_keys += 1
                else:
                    self.midbar_keys += 1
                self.last_was_time = False
                return 'bound'
            # t in TIMES
            if not self._header_time_ok():
                raise ValueError('consecutive or capped time token %r' % t)
            if t in SHORT_PICKUP_TIMES:
                if not self._short_pickup_time_ok(recent_before=self.recent[:-1]):
                    raise ValueError('spurious mid-bar pickup time %r' % t)
            if self.header_open:
                if self.cur_hdr_tuple is None:
                    self.cur_hdr_tuple = [None, None, t]
                else:
                    self.cur_hdr_tuple[2] = t
                self.header_times += 1
            else:
                self.midbar_times += 1
            self._apply_tsig(_beats_of(t), _beats_rat_of(t))
            self.last_was_time = True
            return 'bound'
        self.last_was_time = False
        if self.header_open:
            if self.cur_hdr_tuple is not None:
                self.completed_hdr_tuples.append(tuple(self.cur_hdr_tuple))
                self.cur_hdr_tuple = None
            if self.first_sys_staves == 0:
                self.first_sys_staves = max(
                    1, self.header_clefs, self.header_keys, self.header_times
                )
            self.header_open = False
        if t == BAR or t in BAR_RIGHT:
            if self.has_open_tuplet():
                raise ValueError('unclosed tuplet before bar marker %r' % t)
            # step() appends every token to recent before dispatch, so
            # the previous token is recent[-2] (recent[-1] is t itself).
            if self.strict and len(self.recent) >= 2 \
                    and self.recent[-2] in BAR_LIKES:
                raise ValueError('duplicate bar marker')
            if self.bar_has_bar_rest and not self.bar_has_pitched:
                self.consec_bar_rest_bars += 1
            else:
                self.consec_bar_rest_bars = 0
            self._record_completed_bar()
            self.bars_in_system += 1
            self.cur_bar_tokens = []
            self.tokens_since_bar = 0
            self.bar_has_pitched = False
            self.bar_has_bar_rest = False
            self.seg_has_bar_rest = False
            self.staves_in_bar = 0
            self.bar_txt_spans = 0
            self.onset_spanner_starts = 0
            self._reset_event_guards()
            # A stashed courtesy time-sig takes effect for the new
            # measure whose barline just arrived.
            if self.pending_tsig_by_staff:
                for _s_idx, (_pb, _pbr) in self.pending_tsig_by_staff.items():
                    self.tsig_by_staff[_s_idx] = _pb
                    self.tsig_rat_by_staff[_s_idx] = _pbr
                self.pending_tsig_by_staff.clear()
            if self.pending_tsig is not None:
                self.tsig = self.pending_tsig
                self.tsig_rat = self.pending_tsig_rat
                self.pending_tsig = None
                self.pending_tsig_rat = None
            self.seg_last = [False]
            self.seg = 0
            self.depth = [0]
            self.opened = set()
            self.vcurs = [0.0]
            self.vcurs_rat = [(0, 1)]
            self.vscales = [[]]
            self.ev_base = None
            self.ev_base_rat = None
            self.ev_dots = 0
            self.ev_chord = False
            self.ev_grace = False
            self.ev_mods = 0
            self.ev_alters = 0
            self.chords_on_onset = 0
            self.grace_run = 0
            return 'bound'
        if t == STAFF_SEP:
            if self.has_open_tuplet():
                raise ValueError('unclosed tuplet before staff separator %r' % t)
            if self.staves_in_bar >= MAX_STAVES_PER_BAR:
                raise ValueError('staves per bar cap exceeded (%d)' % MAX_STAVES_PER_BAR)
            self.staves_in_bar += 1
            self._ensure_staff_spanner()
            self.seg_has_bar_rest = False
            self.onset_spanner_starts = 0
            self._reset_event_guards()
            self.seg_last = [False]
            self.seg = 0
            self.depth = [0]
            self.opened = set()
            self.vcurs = [0.0]
            self.vcurs_rat = [(0, 1)]
            self.vscales = [[]]
            self.ev_base = None
            self.ev_base_rat = None
            self.ev_dots = 0
            self.ev_chord = False
            self.ev_grace = False
            self.ev_mods = 0
            self.ev_alters = 0
            self.chords_on_onset = 0
            self.grace_run = 0
            return 'bound'
        if t in VOICES:
            if self.has_open_tuplet():
                raise ValueError('unclosed tuplet before voice marker %r' % t)
            n = int(t.split('-')[1])
            if n == 0:
                raise ValueError('bad voice token %r' % t)
            if len(self.recent) >= 2 and self.recent[-2] == t:
                raise ValueError('duplicate voice marker %r' % t)
            if self.voice_switch_run >= MAX_CONSEC_VOICE_SWITCHES:
                raise ValueError('consecutive voice switch cap exceeded (%d)' % MAX_CONSEC_VOICE_SWITCHES)
            if self.strict:
                if (n != 1 and (n - 1) not in self.opened
                        and n not in self.opened):
                    raise ValueError('non-sequential voice %r' % t)
            self._grow(n)
            self.seg = n - 1
            self.opened.add(n)
            self.chords_on_onset = 0
            self.grace_run = 0
            self.seg_has_bar_rest = False
            self.onset_spanner_starts = 0
            self.voice_switch_run += 1
            return 'bound'
        if t in TUPLET_STARTS:
            if (self.depth[self.seg] + self.pending >= self.max_tuplet_depth) or (
                    self.strict and self._bar_is_full()):
                raise ValueError('tuplet-start blocked (chained/full bar): %r' % t)
            self.pending += 1
            _, _, a, n = t.split('-')
            self.pend_spans.append((int(a), int(n)))
            return 'bound'
        if t == TUP_OPEN:
            if (self.depth[self.seg] + self.pending >= self.max_tuplet_depth) or (
                    self.strict and self._bar_is_full()):
                raise ValueError('tup-open blocked (chained/full bar): %r' % t)
            self.comp_num = 0
            self.comp_digits = 0
            self.comp_val = 0
            return 'tup-open-a'
        if t == TUPLET_KW:
            if (self.depth[self.seg] + self.pending >= self.max_tuplet_depth) or (
                    self.strict and self._bar_is_full()):
                raise ValueError('tuplet-start blocked (chained/full bar): %r' % t)
            self.comp_num = 0
            self.comp_digits = 0
            self.comp_val = 0
            return 'tuplet-a'
        if t == MULTIREST_KW:
            self.sys_has_event = True
            self._reset_event_guards()
            self._drain()
            self._drained_scale_rat()
            self.ev_base = None
            self.ev_base_rat = None
            self.ev_dots = 0
            self.ev_mods = 0
            self.ev_alters = 0
            self.seg_last[self.seg] = False
            self.chords_on_onset = 0
            self.grace_run = 0
            self.onset_spanner_starts = 0
            self.comp_num = 0
            self.comp_digits = 0
            self.comp_val = 0
            return 'multi-rest'
        if t == TUPLET_STOP or t == TUP_CLOSE:
            if self.depth[self.seg] <= 0:
                raise ValueError('unmatched %s' % t)
            self.depth[self.seg] -= 1
            if self.vscales[self.seg]:
                self.vscales[self.seg].pop()
            return 'bound'
        if t == GRACE:
            if self.grace_run >= MAX_CONSECUTIVE_GRACE:
                raise ValueError('grace run cap exceeded (%d)' % MAX_CONSECUTIVE_GRACE)
            self.grace_run += 1
            self.chords_on_onset = 0
            self._reset_event_guards()
            return 'grace'
        if t in TEXT_TAG_OPENS:
            if TXT_CLOSE not in self.vocab_map:
                raise ValueError('text span %r without </txt> in vocab' % t)
            if self.bar_txt_spans >= MAX_TEXT_SPANS_PER_BAR:
                raise ValueError('text spans per bar cap exceeded (%d)' % MAX_TEXT_SPANS_PER_BAR)
            self.in_text_span = True
            self.txt_toks = 0
            self.bar_txt_spans += 1
            return 'text'
        if t in DIR_TOKENS or t in SPANNER_DIR_TOKENS or (
                self.allow_percussion_tab and t in SLASH_MARKS):
            # Measure dynamics, hairpin wedges, ottava spans, volta brackets
            # and opening repeats: measure/staff scope with anti-loop caps.
            if t in DIR_TOKENS or t in SPANNER_DIR_TOKENS:
                # step() appends t to recent before dispatch; compare against
                # recent[-2] and last_dir_tok to block consecutive identical spam.
                prev_tok = self.recent[-2] if len(self.recent) >= 2 else None
                if self.dir_run >= MAX_CONSEC_DIR_TOKENS or self.last_dir_tok == t or prev_tok == t:
                    raise ValueError('consecutive or capped direction token %r' % t)
                if t in SPANNER_DIR_TOKENS:
                    self._apply_spanner(t)
                self.dir_run += 1
                self.last_dir_tok = t
            return 'bound'
        if self.allow_percussion_tab and t == UNPITCHED:
            self._reset_event_guards()
            return 'unpitched'
        if t == CHORD:
            if not self.seg_last[self.seg]:
                raise ValueError('dangling chord marker')
            if self.chords_on_onset >= MAX_CHORDS_PER_ONSET:
                raise ValueError('chord cap exceeded (%d)' % MAX_CHORDS_PER_ONSET)
            self.chords_on_onset += 1
            self.ev_chord = True
            self._reset_event_guards()
            return 'chord'
        if t in DURATIONS or t == REST or t == BAR_REST:
            self.sys_has_event = True
            self._reset_event_guards()
            self._drain()
            if t in DURATIONS:
                if self._blocks_duration(_WRITTEN_RAT[t]):
                    raise ValueError(
                        'duration overshoots bar: %.4f + %.4f > %.4f'
                        % (self.vcurs[self.seg],
                           _WRITTEN[t] * self._open_scale(), self._beat_limit()))
                self.alter_seen = False
                self.dot_seen = False
                self.ev_base = _WRITTEN[t]
                self.ev_base_rat = _WRITTEN_RAT[t]
                self.ev_dots = 0
                self.ev_grace = False
                self.ev_mods = 0
                self.ev_alters = 0
                self.chords_on_onset = 0
                self.grace_run = 0
                self.onset_spanner_starts = 0
                self.bar_has_pitched = True
                self.consec_bar_rest_bars = 0
                self.seg_has_bar_rest = False
                return 'dur'
            if t == REST:
                self.ev_base = None
                self.ev_base_rat = None
                self.ev_dots = 0
                self.ev_mods = 0
                self.ev_alters = 0
                self.chords_on_onset = 0
                self.grace_run = 0
                self.onset_spanner_starts = 0
                self.seg_has_bar_rest = False
                return 'rest'
            if self.seg_has_bar_rest:
                raise ValueError('duplicate bar-rest in same voice segment')
            if not self.bar_has_pitched and self.consec_bar_rest_bars >= MAX_CONSECUTIVE_BAR_REST:
                raise ValueError(
                    'consecutive bar-rest cap exceeded (%d)' % MAX_CONSECUTIVE_BAR_REST
                )
            if self.strict and (self._bar_is_full() or self.vcurs_rat[self.seg][0] > 0):
                raise ValueError('bar-rest blocked in non-empty/full bar')
            self._drained_scale_rat()
            ts_rat = self._active_tsig_rat()
            if ts_rat is not None and _rat_cmp(ts_rat, self.vcurs_rat[self.seg]) > 0:
                self.vcurs_rat[self.seg] = ts_rat
                self.vcurs[self.seg] = ts_rat[0] / ts_rat[1]
            self.ev_base = None
            self.ev_base_rat = None
            self.ev_dots = 0
            self.ev_mods = 0
            self.ev_alters = 0
            self.seg_last[self.seg] = False
            self.chords_on_onset = 0
            self.grace_run = 0
            self.onset_spanner_starts = 0
            self.bar_has_bar_rest = True
            self.seg_has_bar_rest = True
            return 'bound'
        raise ValueError('unexpected token %r' % t)

    def step(self, t):
        """Advance by one token (raises ValueError on invalid)."""
        if self.broken:
            raise ValueError('broken prefix')
        # Unigram runs + recent window update for EVERY token (including
        # the early-return note-tail branches below).
        if t == self.run_tok:
            self.run_len += 1
        else:
            self.run_tok = t
            self.run_len = 1
        self.recent.append(t)
        if len(self.recent) > RECENT_WINDOW_CAP:
            del self.recent[:-RECENT_WINDOW_KEEP]
        self.cur_sys_tokens.append(t)
        self.tokens_in_system += 1
        if t not in BAR_LIKES:
            self.cur_bar_tokens.append(t)
            self.tokens_since_bar += 1
        if self.mode == 'bound':
            self.mode = self._bound(t)
        elif self.mode == 'chord':
            if t not in DURATIONS:
                raise ValueError('chord on non-duration %r' % t)
            self._drain()
            self.alter_seen = False
            self.dot_seen = False
            self.ev_base = _WRITTEN[t]
            self.ev_base_rat = _WRITTEN_RAT[t]
            self.ev_dots = 0
            self.ev_mods = 0
            self.ev_alters = 0
            self.mode = 'dur'
        elif self.mode == 'grace':
            if self.allow_percussion_tab and t == UNPITCHED:
                self.ev_grace = True
                self.mode = 'unpitched'
                return
            if t not in DURATIONS:
                raise ValueError('grace without duration %r' % t)
            self._drain()
            self.alter_seen = False
            self.dot_seen = False
            self.ev_base = _WRITTEN[t]
            self.ev_base_rat = _WRITTEN_RAT[t]
            self.ev_dots = 0
            self.ev_mods = 0
            self.ev_alters = 0
            self.ev_grace = True
            self.mode = 'dur'
        elif self.mode == 'unpitched':
            if t not in DURATIONS:
                raise ValueError('unpitched without duration %r' % t)
            self._drain()
            if not self.ev_grace and self._blocks_duration(_WRITTEN_RAT[t]):
                raise ValueError(
                    'duration overshoots bar: %.4f + %.4f > %.4f'
                    % (self.vcurs[self.seg],
                       _WRITTEN[t] * self._open_scale(), self._beat_limit()))
            self.alter_seen = False
            self.dot_seen = False
            self.ev_base = _WRITTEN[t]
            self.ev_base_rat = _WRITTEN_RAT[t]
            self.ev_dots = 0
            self.ev_mods = 0
            self.ev_alters = 0
            self.mode = 'dur'
        elif self.mode == 'dur':
            if t not in STEPS:
                raise ValueError('bad step token %r' % t)
            self.bar_has_pitched = True
            self.consec_bar_rest_bars = 0
            self.mode = 'step'
        elif self.mode == 'step':
            if t not in OCTAVES:
                raise ValueError('bad octave token %r' % t)
            self.mode = 'pitch-tail'
        elif self.mode == 'pitch-tail':
            if self.strict and t in ALTERS and self.alter_seen:
                raise ValueError('second alter in one note')
            if self.strict and t in ALTERS and self.dot_seen:
                raise ValueError('alter after dot')
            if t in ALTERS or t == DOT:
                if t in ALTERS:
                    if not self.strict and self.ev_alters >= MAX_ALTERS_LAX:
                        raise ValueError('alter cap exceeded (%d)' % MAX_ALTERS_LAX)
                    self.alter_seen = True
                    self.ev_alters += 1
                else:
                    if self.ev_dots >= MAX_DOTS_PER_EVENT:
                        raise ValueError('dot cap exceeded (%d)' % MAX_DOTS_PER_EVENT)
                    self.dot_seen = True
                    self.ev_dots += 1
                    if self._dot_overshoots():
                        raise ValueError(
                            'dot overshoots bar: %.4f > %.4f'
                            % (self.vcurs[self.seg]
                               + self._pending_value(), self._active_tsig()))
                return
            if t in MODS or t in SPANNER_MOD_TOKENS or (self.allow_percussion_tab and (
                    t in NOTEHEADS or t in TAB_STRINGS or t in TAB_FRETS)):
                # Slur/articulation/ornament/cue/xstaff/fermata marks (plus
                # bekern-full numbered spanners & opt-in Gate-D notehead/tab marks):
                # tail position, capped per event to prevent unbounded modifier loops.
                if self.ev_mods >= MAX_MODS_PER_EVENT:
                    raise ValueError('modifier cap exceeded (%d)' % MAX_MODS_PER_EVENT)
                if t in SPANNER_MOD_TOKENS:
                    self._apply_spanner(t)
                self.ev_mods += 1
                return
            if t == TIE_START:
                self.mode = 'tiestart'
                return
            if t == TIE_STOP:
                self._close_event(True)
                self.mode = 'bound'
                return
            self._close_event(True)
            self.mode = self._bound(t)
        elif self.mode == 'tiestart':
            if t == TIE_STOP:
                self._close_event(True)
                self.mode = 'bound'
                return
            if t in MODS or t in SPANNER_MOD_TOKENS:
                if self.ev_mods >= MAX_MODS_PER_EVENT:
                    raise ValueError('modifier cap exceeded (%d)' % MAX_MODS_PER_EVENT)
                if t in SPANNER_MOD_TOKENS:
                    self._apply_spanner(t)
                self.ev_mods += 1
                return
            self._close_event(True)
            self.mode = self._bound(t)
        elif self.mode == 'rest':
            if t not in DURATIONS:
                raise ValueError('bad rest duration %r' % t)
            if self._blocks_duration(_WRITTEN_RAT[t]):
                raise ValueError(
                    'duration overshoots bar: %.4f + %.4f > %.4f'
                    % (self.vcurs[self.seg],
                       _WRITTEN[t] * self._open_scale(), self._beat_limit()))
            self.ev_base = _WRITTEN[t]
            self.ev_base_rat = _WRITTEN_RAT[t]
            self.ev_dots = 0
            self.ev_mods = 0
            self.ev_alters = 0
            self.mode = 'rest-tail'
        elif self.mode == 'rest-tail':
            if t == DOT:
                if self.ev_dots >= MAX_DOTS_PER_EVENT:
                    raise ValueError('dot cap exceeded (%d)' % MAX_DOTS_PER_EVENT)
                self.ev_dots += 1
                if self._dot_overshoots():
                    raise ValueError(
                        'dot overshoots bar: %.4f > %.4f'
                        % (self.vcurs[self.seg]
                           + self._pending_value(), self._active_tsig()))
                return
            if t in MODS or t in SPANNER_MOD_TOKENS:
                # Fermatas / cue / xstaff / articulations over rests ride the rest tail.
                if self.ev_mods >= MAX_MODS_PER_EVENT:
                    raise ValueError('modifier cap exceeded (%d)' % MAX_MODS_PER_EVENT)
                if t in SPANNER_MOD_TOKENS:
                    self._apply_spanner(t)
                self.ev_mods += 1
                return
            self._close_event(False)
            self.mode = self._bound(t)
        elif self.mode == 'text':
            if t == TXT_CLOSE:
                self.in_text_span = False
                self.txt_toks = 0
                self.mode = 'bound'
                return
            if self.txt_toks >= MAX_TXT_TOKENS:
                raise ValueError('text span token cap exceeded (%d)' % MAX_TXT_TOKENS)
            if t in _TEXT_FORBIDDEN:
                raise ValueError('forbidden token %r inside text span' % t)
            self.txt_toks += 1
            return
        elif self.mode == 'tup-open-a':
            if SLASH in self.vocab_map:
                if t == SLASH:
                    if self.comp_digits == 0 or self.comp_val <= 0 or self.comp_val > MAX_TUPLET_VAL:
                        raise ValueError('tup-open slash without valid actual count')
                    self.comp_num = self.comp_val
                    self.comp_digits = 0
                    self.comp_val = 0
                    self.mode = 'tup-open-n'
                elif t in DIGITS:
                    if self.comp_digits == 0 and t == '0':
                        raise ValueError('leading zero in tup-open actual')
                    if self.comp_digits >= 2:
                        raise ValueError('tup-open actual too many digits')
                    new_val = self.comp_val * 10 + int(t)
                    if new_val > MAX_TUPLET_VAL:
                        raise ValueError('tup-open actual %d exceeds cap %d' % (new_val, MAX_TUPLET_VAL))
                    self.comp_val = new_val
                    self.comp_digits += 1
                else:
                    raise ValueError('bad tup-open actual token %r' % t)
            else:
                valid_digs = NONZERO_DIGITS if '9' in self.vocab_map else TUP_OPEN_DIGITS
                if t not in valid_digs:
                    raise ValueError('bad tup-open actual digit %r' % t)
                self.comp_num = int(t)
                self.mode = 'tup-open-n'
        elif self.mode == 'tup-open-n':
            if SLASH in self.vocab_map:
                if t in DIGITS:
                    if self.comp_digits == 0 and t == '0':
                        raise ValueError('leading zero in tup-open normal')
                    if self.comp_digits >= 2:
                        raise ValueError('tup-open normal too many digits')
                    new_val = self.comp_val * 10 + int(t)
                    if new_val > MAX_TUPLET_VAL:
                        raise ValueError('tup-open normal %d exceeds cap %d' % (new_val, MAX_TUPLET_VAL))
                    if self.comp_digits == 1:
                        if new_val == self.comp_num:
                            raise ValueError('identity tuplet %d:%d forbidden' % (self.comp_num, new_val))
                        self.pending += 1
                        self.pend_spans.append((self.comp_num, new_val))
                        self.comp_num = 0
                        self.comp_digits = 0
                        self.comp_val = 0
                        self.mode = 'bound'
                    elif new_val * 10 > MAX_TUPLET_VAL:
                        if new_val == self.comp_num:
                            raise ValueError('identity tuplet %d:%d forbidden' % (self.comp_num, new_val))
                        self.pending += 1
                        self.pend_spans.append((self.comp_num, new_val))
                        self.comp_num = 0
                        self.comp_digits = 0
                        self.comp_val = 0
                        self.mode = 'bound'
                    else:
                        self.comp_val = new_val
                        self.comp_digits = 1
                else:
                    if self.comp_digits == 0:
                        raise ValueError('tup-open normal missing before %r' % t)
                    if self.comp_val == self.comp_num:
                        raise ValueError('identity tuplet %d:%d forbidden' % (self.comp_num, self.comp_val))
                    self.pending += 1
                    self.pend_spans.append((self.comp_num, self.comp_val))
                    self.comp_num = 0
                    self.comp_digits = 0
                    self.comp_val = 0
                    self.mode = self._bound(t)
            else:
                valid_digs = NONZERO_DIGITS if '9' in self.vocab_map else TUP_OPEN_DIGITS
                if t not in valid_digs:
                    raise ValueError('bad tup-open normal digit %r' % t)
                norm_val = int(t)
                if (self.max_tuplet_depth > 1 or '9' in self.vocab_map) and norm_val == self.comp_num:
                    raise ValueError('identity tuplet %d:%d forbidden' % (self.comp_num, norm_val))
                self.pending += 1
                self.pend_spans.append((self.comp_num, norm_val))
                self.comp_num = 0
                self.mode = 'bound'
        elif self.mode == 'time-num':
            if t == SLASH:
                if self.comp_digits == 0 or self.comp_val <= 0:
                    raise ValueError('time slash without numerator')
                self.comp_num = self.comp_val
                self.comp_digits = 0
                self.comp_val = 0
                self.mode = 'time-den'
            elif t in DIGITS:
                if self.comp_digits == 0 and t == '0':
                    raise ValueError('leading zero in time numerator')
                if self.comp_digits >= 2:
                    raise ValueError('time numerator too many digits')
                self.comp_val = self.comp_val * 10 + int(t)
                self.comp_digits += 1
            else:
                raise ValueError('bad time numerator token %r' % t)
        elif self.mode == 'time-den':
            if t in DIGITS:
                if self.comp_digits == 0:
                    if t not in TIME_DEN_DIGITS:
                        raise ValueError('bad time denominator digit %r' % t)
                    self.comp_val = int(t)
                    self.comp_digits = 1
                    if self.comp_val in (2, 4, 8):
                        self._apply_tsig(
                            self.comp_num * 4.0 / self.comp_val,
                            _rat_reduce(self.comp_num * 4, self.comp_val),
                        )
                        self.comp_num = 0
                        self.comp_digits = 0
                        self.comp_val = 0
                        self.last_was_time = True
                        self.mode = 'bound'
                elif self.comp_digits == 1:
                    den = self.comp_val * 10 + int(t)
                    if den not in (16, 32, 64):
                        raise ValueError('bad time denominator %d' % den)
                    self._apply_tsig(
                        self.comp_num * 4.0 / den,
                        _rat_reduce(self.comp_num * 4, den),
                    )
                    self.comp_num = 0
                    self.comp_digits = 0
                    self.comp_val = 0
                    self.last_was_time = True
                    self.mode = 'bound'
                else:
                    raise ValueError('time denominator too many digits')
            else:
                if self.comp_digits == 1 and self.comp_val == 1:
                    self._apply_tsig(
                        self.comp_num * 4.0,
                        _rat_reduce(self.comp_num * 4, 1),
                    )
                    self.comp_num = 0
                    self.comp_digits = 0
                    self.comp_val = 0
                    self.last_was_time = True
                    self.mode = self._bound(t)
                else:
                    raise ValueError('incomplete time denominator before %r' % t)
        elif self.mode == 'tuplet-a':
            if t == SLASH:
                if self.comp_digits == 0 or self.comp_val <= 0:
                    raise ValueError('tuplet slash without actual count')
                self.comp_num = self.comp_val
                self.comp_digits = 0
                self.comp_val = 0
                self.mode = 'tuplet-n'
            elif t in DIGITS:
                if self.comp_digits == 0 and t == '0':
                    raise ValueError('leading zero in tuplet actual')
                if self.comp_digits >= 2:
                    raise ValueError('tuplet actual too many digits')
                self.comp_val = self.comp_val * 10 + int(t)
                self.comp_digits += 1
            else:
                raise ValueError('bad tuplet actual token %r' % t)
        elif self.mode == 'tuplet-n':
            if t in DIGITS:
                if self.comp_digits == 0 and t == '0':
                    raise ValueError('leading zero in tuplet normal')
                if self.comp_digits >= 2:
                    raise ValueError('tuplet normal too many digits')
                self.comp_val = self.comp_val * 10 + int(t)
                self.comp_digits += 1
                if self.comp_digits == 2:
                    self.pending += 1
                    self.pend_spans.append((self.comp_num, self.comp_val))
                    self.comp_num = 0
                    self.comp_digits = 0
                    self.comp_val = 0
                    self.mode = 'bound'
            else:
                if self.comp_digits == 0:
                    raise ValueError('tuplet normal missing before %r' % t)
                self.pending += 1
                self.pend_spans.append((self.comp_num, self.comp_val))
                self.comp_num = 0
                self.comp_digits = 0
                self.comp_val = 0
                self.mode = self._bound(t)
        elif self.mode == 'multi-rest':
            if t in DIGITS:
                if self.comp_digits == 0 and t == '0':
                    raise ValueError('leading zero in multi-rest count')
                if self.comp_digits >= 3:
                    raise ValueError('multi-rest count too many digits')
                self.comp_val = self.comp_val * 10 + int(t)
                self.comp_digits += 1
            elif t in BAR_LIKES:
                if self.comp_digits == 0:
                    raise ValueError('multi-rest bar without count')
                self.comp_num = 0
                self.comp_digits = 0
                self.comp_val = 0
                self.mode = self._bound(t)
            else:
                raise ValueError('bad multi-rest token %r' % t)
        else:
            raise ValueError('bad mode')

    def _close_event(self, placed):
        self.seg_last[self.seg] = placed
        if self.ev_base_rat is not None:
            # Queued tuplet starts open on this event's segment
            # (mirrors decode.rs: spans drain into every event,
            # chord tones included); chord tones add no duration.
            scale_rat = self._drained_scale_rat()
            if not self.ev_chord and not self.ev_grace:
                val_rat = _rat_mul(
                    _rat_mul(self.ev_base_rat, _dot_mult_rat(self.ev_dots)),
                    scale_rat,
                )
                limit_rat = self._beat_limit_rat()
                new_cur = _rat_add(self.vcurs_rat[self.seg], val_rat)
                if limit_rat is not None and _rat_cmp(new_cur, limit_rat) > 0:
                    raise ValueError(
                        'duration overshoots bar: %.4f + %.4f > %.4f'
                        % (self.vcurs[self.seg], val_rat[0] / val_rat[1],
                           limit_rat[0] / limit_rat[1]))
                self.vcurs_rat[self.seg] = new_cur
                self.vcurs[self.seg] = new_cur[0] / new_cur[1]
        self.ev_base = None
        self.ev_base_rat = None
        self.ev_dots = 0
        self.ev_chord = False
        self.ev_grace = False
        self.ev_mods = 0
        self.ev_alters = 0

    def cycle_forbidden(
        self,
        max_repeats=12,
        ngram_min=8,
        ngram_max=16,
        max_ngram_repeats=3,
        non_bar_ngram_repeats=8,
        bar_ngram_max=160,
    ):
        """Token STRING that would extend a tight cycle or n-gram loop.

        Returns the single forbidden continuation token, or None. Checks:
        0. Canonical completed-bar repeat guard within the current system
           (>=5 consecutive identical canonical non-rest bars, or >=8 in
           the last 12 bars, ignoring cosmetic slur/clef/dyn tokens).
        1. Identical 8-to-16-token pattern: requires a bar token (``BAR_LIKES``)
           inside the pattern to trigger at ``max_ngram_repeats`` (3) consecutive
           repeats; non-bar 8..16-token patterns (legitimate musical ostinatos /
           arpeggios) require ``non_bar_ngram_repeats`` (8) consecutive repeats.
        2. Identical bar-containing pattern of length 17..160 repeated
           ``max_ngram_repeats`` (3) consecutive times (kills verbatim
           multi-bar loops without touching non-bar ostinatos).
        3. Exact adjacent repeats of any short period (1..24 tokens) past
           ``max_repeats`` (12) consecutive repeats.

        Suspended while the header is open (headers legitimately cycle
        clef/key/time per staff).
        """
        if self.header_open:
            return None
        if (
            self.bar_repeat_forbidden_tok is not None
            and all(tok in _COSMETIC_BAR_TOKENS for tok in self.cur_bar_tokens)
        ):
            return self.bar_repeat_forbidden_tok
        # 0b. Multi-voice consecutive bar-hash repetition guard (>=3 identical normalized
        # multi-voice bars): once the next bar matches >=6 non-cosmetic tokens verbatim
        # (or len-1 for shorter bars), forbid the next matching token to force divergence
        # without ever blocking a legitimately different bar starting with `voice-1`.
        if (
            self.mv_bar_consec >= MULTI_VOICE_BAR_HASH_CONSEC_LIMIT
            and self.last_mv_bar_canon
            and len(self.last_mv_bar_canon) >= 6
        ):
            cur_non_cos = [tok for tok in self.cur_bar_tokens if tok not in _COSMETIC_BAR_TOKENS]
            m = len(cur_non_cos)
            min_prefix = min(6, len(self.last_mv_bar_canon) - 1)
            if (
                m >= min_prefix
                and m < len(self.last_mv_bar_canon)
                and cur_non_cos == self.last_mv_bar_canon[:m]
            ):
                return self.last_mv_bar_canon[m]
        r = self.recent
        n = len(r)
        # 1. Conservative 8..16-token n-gram loop guard (bar: >3x, non-bar ostinato: >=8x)
        for p in range(ngram_min, ngram_max + 1):
            if n < max_ngram_repeats * p:
                continue
            if r[n - p] != r[n - 2 * p]:
                continue
            base = r[n - p:]
            has_bar = any(tok in BAR_LIKES for tok in base)
            req_reps = max_ngram_repeats if has_bar else non_bar_ngram_repeats
            if n < req_reps * p:
                continue
            ok = True
            for k in range(1, req_reps):
                if r[n - (k + 1) * p:n - k * p] != base:
                    ok = False
                    break
            if ok:
                return base[0]
        # 2. Verbatim bar-level loop guard for longer measures (17..64 tokens, >3x)
        for p in range(ngram_max + 1, bar_ngram_max + 1):
            if n < max_ngram_repeats * p:
                continue
            if r[n - p] != r[n - 2 * p]:
                continue
            base = r[n - p:]
            if not any(tok in BAR_LIKES for tok in base):
                continue
            ok = True
            for k in range(1, max_ngram_repeats):
                if r[n - (k + 1) * p:n - k * p] != base:
                    ok = False
                    break
            if ok:
                return base[0]
        # 3. General short-period tight-cycle guard (1..24 tokens, >= max_repeats)
        for p in range(1, 25):
            if n < max_repeats * p:
                continue
            if r[n - p] != r[n - 2 * p]:
                continue
            base = r[n - p:]
            ok = True
            for k in range(1, max_repeats):
                if r[n - (k + 1) * p:n - k * p] != base:
                    ok = False
                    break
            if ok:
                return base[0]
        return None

    def allowed(self):
        """Allowed next token ids + EOS legality for the current state."""
        if self.broken:
            return _fail_open()
        vm = self.vocab_map
        allow = set()
        bar_tok_capped = self.tokens_since_bar >= MAX_TOKENS_PER_BAR
        sys_capped = self._sys_is_capped()
        eos_ok = (len(self.recent) == 0) or self.can_emit_eos()

        def add_boundary(placed=None):
            open_tup = self.has_open_tuplet()
            if sys_capped:
                if self.depth[self.seg] > 0:
                    if TUPLET_STOP in vm:
                        allow.add(vm[TUPLET_STOP])
                    if TUP_CLOSE in vm:
                        allow.add(vm[TUP_CLOSE])
                elif self.pending > 0:
                    for t in DURATIONS:
                        if t in vm and not self._blocks_duration(_WRITTEN_RAT[t]):
                            allow.add(vm[t])
                return
            if bar_tok_capped:
                if self.depth[self.seg] > 0:
                    if TUPLET_STOP in vm:
                        allow.add(vm[TUPLET_STOP])
                    if TUP_CLOSE in vm:
                        allow.add(vm[TUP_CLOSE])
                elif self.pending > 0:
                    for t in DURATIONS:
                        if t in vm and not self._blocks_duration(_WRITTEN_RAT[t]):
                            allow.add(vm[t])
                else:
                    allow.add(vm[BAR])
                    allow.update(vm[t] for t in BAR_RIGHT if t in vm)
                return
            # Strict parity with step(): a BAR right after a BAR, and
            # a voice marker that step() would reject (non-sequential
            # open or consecutive duplicate), are masked here instead
            # of failing open after the fact. Non-strict keeps the
            # legacy unconditional set. Both modes forbid bar*, <staff>,
            # and voice-* while a tuplet is open.
            if not open_tup and not (self.strict and self.recent
                    and self.recent[-1] in BAR_LIKES):
                allow.add(vm[BAR])
                allow.update(vm[t] for t in BAR_RIGHT if t in vm)
            if not open_tup and STAFF_SEP in vm and self.staves_in_bar < MAX_STAVES_PER_BAR:
                allow.add(vm[STAFF_SEP])
            if not open_tup and self.voice_switch_run < MAX_CONSEC_VOICE_SWITCHES:
                for _vt in VOICES:
                    if _vt not in vm:
                        continue
                    if self.recent and self.recent[-1] == _vt:
                        continue
                    if self.strict:
                        _vn = int(_vt.split('-')[1])
                        if _vn != 1 and (_vn - 1) not in self.opened \
                                and _vn not in self.opened:
                            continue
                    allow.add(vm[_vt])
            if not (self.depth[self.seg] + self.pending >= self.max_tuplet_depth
                    or (self.strict and self._bar_is_full())):
                allow.update(vm[t] for t in TUPLET_STARTS if t in vm)
                if TUP_OPEN in vm:
                    allow.add(vm[TUP_OPEN])
                if TUPLET_KW in vm:
                    allow.add(vm[TUPLET_KW])
            for t in DURATIONS:
                if t in vm and not self._blocks_duration(_WRITTEN_RAT[t]):
                    allow.add(vm[t])
            # T2.4b: a full bar must not offer REST -- entering
            # rest-mode with every duration overshooting traps the
            # decoder with all logits -inf.
            if not self._bar_is_full():
                allow.add(vm[REST])
            # Iter-7/8 anti-loop: forbid BAR_REST when consecutive pure-rest
            # bars hit MAX_CONSECUTIVE_BAR_REST (both modes), when this voice
            # segment already emitted a bar-rest, or when strict bar is full / non-empty.
            cur_used_rat = _rat_add(self.vcurs_rat[self.seg], self._pending_value_rat())
            bar_rest_capped = (
                self.seg_has_bar_rest
                or (
                    not self.bar_has_pitched
                    and self.consec_bar_rest_bars >= MAX_CONSECUTIVE_BAR_REST
                )
            )
            if not bar_rest_capped and not (
                self.strict and (self._bar_is_full() or cur_used_rat[0] > 0)
            ):
                allow.add(vm[BAR_REST])
                if MULTIREST_KW in vm:
                    allow.add(vm[MULTIREST_KW])
            if self.depth[self.seg] > 0:
                if TUPLET_STOP in vm:
                    allow.add(vm[TUPLET_STOP])
                if TUP_CLOSE in vm:
                    allow.add(vm[TUP_CLOSE])
            if (placed if placed is not None else self.seg_last[self.seg]) \
                    and self.chords_on_onset < MAX_CHORDS_PER_ONSET:
                allow.add(vm[CHORD])
            if self._header_clef_ok():
                allow.update(vm[t] for t in CLEFS if t in vm)
            hdr_ok = self._header_attr_ok()
            if hdr_ok:
                if self._header_key_ok():
                    allow.update(vm[t] for t in KEYS if t in vm)
                if self._header_time_ok():
                    short_ok = self._short_pickup_time_ok(recent_before=self.recent)
                    for t in TIMES:
                        if t in vm and (short_ok or t not in SHORT_PICKUP_TIMES):
                            allow.add(vm[t])
                    if TIME_KW in vm:
                        allow.add(vm[TIME_KW])
            if self.grace_run < MAX_CONSECUTIVE_GRACE:
                allow.add(vm[GRACE])
            for t in DIR_TOKENS:
                if t in vm and self._dir_token_ok(t):
                    allow.add(vm[t])
            for t in SPANNER_DIR_TOKENS:
                if t in vm and self._dir_token_ok(t) and self._spanner_ok(t):
                    allow.add(vm[t])
            if TXT_CLOSE in vm and self.bar_txt_spans < MAX_TEXT_SPANS_PER_BAR:
                for t in TEXT_TAG_OPENS:
                    if t in vm:
                        allow.add(vm[t])
            if self.allow_percussion_tab:
                if self._header_clef_ok():
                    allow.update(vm[t] for t in TAB_CLEFS if t in vm)
                allow.update(vm[t] for t in SLASH_MARKS if t in vm)
                if UNPITCHED in vm:
                    allow.add(vm[UNPITCHED])

        if self.mode == 'bound':
            add_boundary()
            return allow, eos_ok
        if self.mode == 'text':
            if TXT_CLOSE in vm:
                allow.add(vm[TXT_CLOSE])
            if self.txt_toks < MAX_TXT_TOKENS and not (bar_tok_capped or sys_capped):
                for tok, tid in vm.items():
                    if tok not in _TEXT_FORBIDDEN:
                        allow.add(tid)
            return allow, False
        if self.mode == 'grace':
            # Display-only duration follows a grace marker (never
            # strict-gated: grace steals no time).
            allow.update(vm[t] for t in DURATIONS if t in vm)
            if self.allow_percussion_tab and UNPITCHED in vm:
                allow.add(vm[UNPITCHED])
            return allow, False
        if self.mode == 'unpitched':
            for t in DURATIONS:
                if t in vm and (self.ev_grace or not self._blocks_duration(_WRITTEN_RAT[t])):
                    allow.add(vm[t])
            return allow, False
        if self.mode == 'chord':
            allow.update(vm[t] for t in DURATIONS if t in vm)
            return allow, False
        if self.mode == 'dur':
            allow.update(vm[t] for t in STEPS if t in vm)
            return allow, False
        if self.mode == 'step':
            allow.update(vm[t] for t in OCTAVES if t in vm)
            return allow, False
        if self.mode == 'pitch-tail':
            if not (bar_tok_capped or sys_capped):
                # decode.rs consumes a single optional alter BEFORE the dot
                # loop; a second alter, or an alter after dots, never parses.
                if (not (self.strict and (self.alter_seen or self.dot_seen))
                        and (self.strict or self.ev_alters < MAX_ALTERS_LAX)):
                    allow.update(vm[t] for t in ALTERS if t in vm)
                if self._dot_ok():
                    allow.add(vm[DOT])
                allow.add(vm[TIE_START])
                allow.add(vm[TIE_STOP])
                if self.ev_mods < MAX_MODS_PER_EVENT:
                    allow.update(vm[t] for t in MODS if t in vm)
                    for t in SPANNER_MOD_TOKENS:
                        if t in vm and self._spanner_ok(t):
                            allow.add(vm[t])
                    if self.allow_percussion_tab:
                        allow.update(vm[t] for t in NOTEHEADS | TAB_STRINGS | TAB_FRETS
                                     if t in vm)
            add_boundary(placed=True)
            return allow, eos_ok
        if self.mode == 'tiestart':
            if not (bar_tok_capped or sys_capped):
                allow.add(vm[TIE_STOP])
                if self.ev_mods < MAX_MODS_PER_EVENT:
                    allow.update(vm[t] for t in MODS if t in vm)
                    for t in SPANNER_MOD_TOKENS:
                        if t in vm and self._spanner_ok(t):
                            allow.add(vm[t])
            add_boundary(placed=True)
            return allow, eos_ok
        if self.mode == 'rest':
            if self._bar_is_full():
                # Full bar: no duration fits (each overshoots);
                # entry is masked at the boundary, so this only
                # fires on paths that bypass the mask. Non-strict
                # never reports full.
                return allow, False
            for t in DURATIONS:
                if t in vm and not self._blocks_duration(_WRITTEN_RAT[t]):
                    allow.add(vm[t])
            return allow, False
        if self.mode == 'rest-tail':
            if not (bar_tok_capped or sys_capped):
                if self._dot_ok():
                    allow.add(vm[DOT])
                if self.ev_mods < MAX_MODS_PER_EVENT:
                    allow.update(vm[t] for t in MODS if t in vm)
                    for t in SPANNER_MOD_TOKENS:
                        if t in vm and self._spanner_ok(t):
                            allow.add(vm[t])
            add_boundary()
            return allow, eos_ok
        if self.mode == 'tup-open-a':
            if SLASH in vm:
                if self.comp_digits == 0:
                    allow.update(vm[t] for t in NONZERO_DIGITS if t in vm)
                elif self.comp_digits == 1:
                    allow.add(vm[SLASH])
                    for d in DIGITS:
                        if d in vm and (self.comp_val * 10 + int(d)) <= MAX_TUPLET_VAL:
                            allow.add(vm[d])
                else:
                    allow.add(vm[SLASH])
            else:
                valid_digs = NONZERO_DIGITS if '9' in vm else TUP_OPEN_DIGITS
                allow.update(vm[t] for t in valid_digs if t in vm)
            return allow, False
        if self.mode == 'tup-open-n':
            if SLASH in vm:
                if self.comp_digits == 0:
                    for d in NONZERO_DIGITS:
                        if d not in vm:
                            continue
                        di = int(d)
                        if di != self.comp_num or di * 10 <= MAX_TUPLET_VAL:
                            allow.add(vm[d])
                    return allow, False
                # comp_digits == 1 (1 <= comp_val <= 3, since >= 4 commits immediately)
                for d in DIGITS:
                    if d not in vm:
                        continue
                    nv = self.comp_val * 10 + int(d)
                    if nv <= MAX_TUPLET_VAL and nv != self.comp_num:
                        allow.add(vm[d])
                if self.comp_val != self.comp_num:
                    self.pending += 1
                    self.pend_spans.append((self.comp_num, self.comp_val))
                    try:
                        add_boundary()
                    finally:
                        self.pending -= 1
                        self.pend_spans.pop()
                return allow, False
            else:
                valid_digs = NONZERO_DIGITS if '9' in vm else TUP_OPEN_DIGITS
                ban_eq = (self.max_tuplet_depth > 1 or '9' in vm)
                for d in valid_digs:
                    if d in vm and (not ban_eq or int(d) != self.comp_num):
                        allow.add(vm[d])
                return allow, False
        if self.mode == 'time-num':
            if self.comp_digits == 0:
                allow.update(vm[t] for t in NONZERO_DIGITS if t in vm)
            elif self.comp_digits == 1:
                allow.update(vm[t] for t in DIGITS if t in vm)
                if SLASH in vm:
                    allow.add(vm[SLASH])
            else:
                if SLASH in vm:
                    allow.add(vm[SLASH])
            return allow, False
        if self.mode == 'time-den':
            if self.comp_digits == 0:
                allow.update(vm[t] for t in TIME_DEN_DIGITS if t in vm)
                return allow, False
            if self.comp_val == 3:
                if '2' in vm:
                    allow.add(vm['2'])
                return allow, False
            if self.comp_val == 6:
                if '4' in vm:
                    allow.add(vm['4'])
                return allow, False
            # comp_val == 1: either '6' (for 16) or a boundary token (for N/1)
            if '6' in vm:
                allow.add(vm['6'])
            add_boundary()
            return allow, eos_ok
        if self.mode == 'tuplet-a':
            if self.comp_digits == 0:
                allow.update(vm[t] for t in NONZERO_DIGITS if t in vm)
            elif self.comp_digits == 1:
                allow.update(vm[t] for t in DIGITS if t in vm)
                if SLASH in vm:
                    allow.add(vm[SLASH])
            else:
                if SLASH in vm:
                    allow.add(vm[SLASH])
            return allow, False
        if self.mode == 'tuplet-n':
            if self.comp_digits == 0:
                allow.update(vm[t] for t in NONZERO_DIGITS if t in vm)
                return allow, False
            allow.update(vm[t] for t in DIGITS if t in vm)
            add_boundary()
            return allow, eos_ok
        if self.mode == 'multi-rest':
            if self.comp_digits == 0:
                allow.update(vm[t] for t in NONZERO_DIGITS if t in vm)
            else:
                if self.comp_digits < 3:
                    allow.update(vm[t] for t in DIGITS if t in vm)
                if BAR in vm:
                    allow.add(vm[BAR])
                allow.update(vm[t] for t in BAR_RIGHT if t in vm)
            return allow, False
        return _fail_open()


class _State:
    """Boundary state mirroring decode.rs cursors (positions only).

    Span depths are per segment (mirrors to_bekern's per-key stacks;
    within a measure segment <-> (staff, voice) is bijective, and spans
    never cross `bar` by encoder construction). Pending starts queue
    and drain into the next event.
    """

    def __init__(self):
        self.header_open = True
        self.seg_last = [False]
        self.seg = 0
        self.depth = [0]
        self.pending = 0

    def _grow(self, n):
        while len(self.seg_last) < n:
            self.seg_last.append(False)
            self.depth.append(0)

    def _drain(self):
        self.depth[self.seg] += self.pending
        self.pending = 0

    def bound(self, t):
        """Process one boundary token; return the next intra-note mode.

        Modes: 'bound' (event-boundary), 'chord', 'dur' (after pitched
        duration, expects step), 'rest' (after `rest`, expects duration).
        """
        if t in CLEFS or t in KEYS or t in TIMES:
            return 'bound'
        self.header_open = False
        if t == BAR or t in BAR_RIGHT or t == STAFF_SEP:
            self.seg_last = [False]
            self.seg = 0
            self.depth = [0]
            # Pending starts survive the bar (decode keeps them; the
            # encoder emits them adjacent to their member event).
            return 'bound'
        if t in VOICES:
            n = int(t.split('-')[1])
            if n == 0:
                raise ValueError('bad voice token %r' % t)
            self._grow(n)
            self.seg = n - 1
            return 'bound'
        if t in TUPLET_STARTS:
            self.pending += 1
            return 'bound'
        if t == TUPLET_STOP or t == TUP_CLOSE:
            if self.depth[self.seg] <= 0:
                raise ValueError('unmatched %s' % t)
            self.depth[self.seg] -= 1
            return 'bound'
        if t == GRACE:
            return 'grace'
        if t in DIR_TOKENS:
            return 'bound'
        if t == CHORD:
            if not self.seg_last[self.seg]:
                raise ValueError('dangling chord marker')
            return 'chord'
        if t in DURATIONS or t == REST or t == BAR_REST:
            self.sys_has_event = True
            self._drain()
            if t in DURATIONS:
                return 'dur'
            if t == REST:
                return 'rest'
            self.seg_last[self.seg] = False
            return 'bound'
        raise ValueError('unexpected token %r' % t)

    def close_event(self, placed):
        """A complete note/rest event ends; placed=True for pitched."""
        self.seg_last[self.seg] = placed


def _walk(tokens):
    """Run the prefix; return (mode, state). Raises ValueError if invalid."""
    st = _State()
    mode = 'bound'
    for t in tokens:
        if mode == 'bound':
            mode = st.bound(t)
        elif mode == 'chord':
            if t not in DURATIONS:
                raise ValueError('chord on non-duration %r' % t)
            st._drain()
            mode = 'dur'
        elif mode == 'grace':
            if t not in DURATIONS:
                raise ValueError('grace without duration %r' % t)
            st._drain()
            mode = 'dur'
        elif mode == 'dur':
            if t not in STEPS:
                raise ValueError('bad step token %r' % t)
            mode = 'step'
        elif mode == 'step':
            if t not in OCTAVES:
                raise ValueError('bad octave token %r' % t)
            mode = 'pitch-tail'
        elif mode == 'pitch-tail':
            if t in ALTERS or t == DOT or t in MODS:
                continue
            if t == TIE_START:
                mode = 'tiestart'
                continue
            if t == TIE_STOP:
                st.close_event(True)
                mode = 'bound'
                continue
            # Event-boundary token: close this note, dispatch it.
            st.close_event(True)
            mode = st.bound(t)
        elif mode == 'tiestart':
            if t == TIE_STOP:
                st.close_event(True)
                mode = 'bound'
                continue
            if t in MODS:
                continue
            st.close_event(True)
            mode = st.bound(t)
        elif mode == 'rest':
            if t not in DURATIONS:
                raise ValueError('bad rest duration %r' % t)
            mode = 'rest-tail'
        elif mode == 'rest-tail':
            if t == DOT or t in MODS:
                continue
            st.close_event(False)
            mode = st.bound(t)
        else:
            raise ValueError('bad mode')
    return mode, st


def _fail_open():
    allow = {data.TOKEN_TO_ID[t] for t in [BAR, REST, BAR_REST]}
    allow.update(data.TOKEN_TO_ID[t] for t in VOICES)
    allow.update(data.TOKEN_TO_ID[t] for t in TUPLET_STARTS)
    allow.update(data.TOKEN_TO_ID[t] for t in DURATIONS)
    return allow, True


def _walk_prefix(prefix_ids, strict):
    """Shared walk: returns PrefixState or None on invalid prefix."""
    try:
        toks = data.decode(list(prefix_ids))
    except ValueError:
        return None
    st = PrefixState(strict=strict)
    try:
        for t in toks:
            st.step(t)
    except ValueError:
        return None
    return st


def allowed_next(prefix_ids):
    """Allowed next token ids + EOS legality for a generated prefix.

    ``prefix_ids`` are VOCAB ids (no BOS/PAD/EOS). Fails open: an
    invalid prefix yields the broad boundary set + EOS so greedy never
    dead-ends on an internal tracking bug. Legacy validator set
    (byte-identical to the pre-incremental ``_walk`` behavior).
    """
    st = _walk_prefix(prefix_ids, False)
    if st is None:
        return _fail_open()
    return st.allowed()


def allowed_next_strict(prefix_ids):
    """Validator-predicate mask: legacy set + GT-safe tightened rules.

    Adds (all proven GT-safe over the 2523-page corpus): single-alter
    + alter-before-dots, sequential voice opening, no consecutive
    duplicate bar/voice markers. Fails open exactly like legacy.
    """
    st = _walk_prefix(prefix_ids, True)
    if st is None:
        return _fail_open()
    return st.allowed()


def _voice_sort_key(tok):
    """Numeric sort key for ``voice-<k>`` tokens (falls back to lexicographic)."""
    try:
        return (0, int(tok.split('-', 1)[1]), tok)
    except (IndexError, ValueError):
        return (1, 0, tok)


def _canonicalize_single_measure(measure_toks):
    """Reorder an onset-interleaved multi-voice measure into canonical voice-major order.

    Measures that are monophonic or already have each ``voice-k`` appearing at most
    once in non-decreasing order are returned verbatim (idempotent, zero perturbation).
    """
    voice_positions = [i for i, t in enumerate(measure_toks) if t in VOICES]
    if len(voice_positions) < 2:
        return list(measure_toks)

    seen_voices = [measure_toks[i] for i in voice_positions]
    if len(seen_voices) == len(set(seen_voices)):
        sort_keys = [_voice_sort_key(v) for v in seen_voices]
        if sort_keys == sorted(sort_keys):
            return list(measure_toks)

    first_v = voice_positions[0]
    prefix = list(measure_toks[:first_v])
    # Collect (voice_tok, segment_toks) runs.
    runs = []
    for idx, pos in enumerate(voice_positions):
        end = voice_positions[idx + 1] if (idx + 1) < len(voice_positions) else len(measure_toks)
        v_tok = measure_toks[pos]
        seg = list(measure_toks[pos + 1 : end])
        runs.append([v_tok, seg])

    # Leading dynamics immediately preceding a voice switch (e.g.
    # ``... tie-start dyn-f voice-2 quarter C 4``) belong to the upcoming voice.
    for idx in range(len(runs) - 1):
        prev_seg = runs[idx][1]
        carry = []
        while prev_seg and prev_seg[-1] in DYNS:
            carry.append(prev_seg.pop())
        if carry:
            carry.reverse()
            runs[idx + 1][1] = carry + runs[idx + 1][1]

    grouped = {}
    for v_tok, seg in runs:
        grouped.setdefault(v_tok, []).extend(seg)

    out = prefix
    for v_tok in sorted(grouped.keys(), key=_voice_sort_key):
        out.append(v_tok)
        out.extend(grouped[v_tok])
    return out


def canonicalize_measure_voices(tokens):
    """Canonicalize per-measure voice ordering to voice-major across ``<sys>`` & barlines.

    Eliminates semantically identical onset-interleaved vs voice-major serialization
    splits (e.g. ``voice-1 ... voice-2 ... voice-1 ... voice-2 ... bar`` ->
    ``voice-1 ... voice-2 ... bar``) while leaving already-canonical measures untouched.
    """
    if not tokens:
        return []

    out = []
    cur_measure = []
    for tok in tokens:
        if tok == '<sys>':
            if cur_measure:
                out.extend(_canonicalize_single_measure(cur_measure))
                cur_measure = []
            out.append(tok)
        elif tok in BAR_LIKES:
            if cur_measure:
                out.extend(_canonicalize_single_measure(cur_measure))
                cur_measure = []
            out.append(tok)
        else:
            cur_measure.append(tok)

    if cur_measure:
        out.extend(_canonicalize_single_measure(cur_measure))
    return out
