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

import hashlib
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
MODS = frozenset(['slur-start', 'slur-stop', 'stacc', 'accent', 'tenuto', 'fermata'])
DYNS = frozenset(t for t in data.VOCAB if t.startswith('dyn-'))
HAIRS = frozenset(t for t in data.VOCAB if t.startswith('hairpin-'))
BAR_LEFT = frozenset(['bar-repeat-left'])
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
# written_dur) plus the bar-closure epsilon for T2.3 strict duration
# books (fp slack for tuplet/dot sums like 3 x (1/2 x 2/3)).
_WRITTEN = {'breve': 8.0, 'whole': 4.0, 'half': 2.0, 'quarter': 1.0,
            'eighth': 0.5, '16th': 0.25, '32nd': 0.125, '64th': 0.0625,
            '128th': 0.03125}
_BAR_EPS = 1e-6
MAX_CHORDS_PER_ONSET = 8
MAX_CONSECUTIVE_GRACE = 8
MAX_CONSECUTIVE_BAR_REST = 24
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
                 'tsig', 'pending_tsig', 'vcurs', 'vscales',
                 'pend_spans',
                 'ev_base', 'ev_dots', 'ev_chord', 'ev_grace',
                 'chords_on_onset', 'grace_run',
                 'consec_bar_rest_bars', 'bar_has_pitched', 'bar_has_bar_rest',
                 'sys_count', 'sys_has_event',
                 'bars_in_system', 'tokens_in_system',
                 'max_bars_per_system', 'max_tokens_per_system',
                 'tokens_since_bar', 'cur_sys_tokens', 'completed_system_hashes',
                 'cur_bar_tokens', 'sys_bar_hashes', 'bar_repeat_forbidden_tok',
                 'comp_num', 'comp_digits', 'comp_val',
                 'allow_percussion_tab', 'vocab_map')

    def __init__(
        self,
        strict=False,
        allow_percussion_tab=False,
        vocab_map=None,
        max_bars_per_system=MAX_BARS_PER_SYSTEM,
        max_tokens_per_system=MAX_TOKENS_PER_SYSTEM,
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
        # per-voice running cursors, per-voice open tuplet ratio
        # stacks, queued tuplet starts, and the pending event's
        # written base quarters + dot count + chord flag.
        self.tsig = None
        # Courtesy time-sig stashed mid-measure; applied at the next
        # barline (a courtesy change must not retune the running
        # books, but must take effect for the new measure).
        self.pending_tsig = None
        self.vcurs = [0.0]
        self.vscales = [[]]
        self.pend_spans = []
        self.ev_base = None
        self.ev_dots = 0
        self.ev_chord = False
        self.ev_grace = False
        self.chords_on_onset = 0
        self.grace_run = 0
        self.consec_bar_rest_bars = 0
        self.bar_has_pitched = False
        self.bar_has_bar_rest = False
        self.sys_count = 0
        self.sys_has_event = False
        self.bars_in_system = 0
        self.tokens_in_system = 0
        self.max_bars_per_system = max_bars_per_system
        self.max_tokens_per_system = max_tokens_per_system
        self.tokens_since_bar = 0
        self.cur_sys_tokens = []
        self.completed_system_hashes = {}
        self.cur_bar_tokens = []
        self.sys_bar_hashes = []
        self.bar_repeat_forbidden_tok = None
        # bekern v2 compositional sub-state counters (flat ints for cheap copy())
        self.comp_num = 0
        self.comp_digits = 0
        self.comp_val = 0
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
        c.pending_tsig = self.pending_tsig
        c.vcurs = list(self.vcurs)
        c.vscales = [list(s) for s in self.vscales]
        c.pend_spans = list(self.pend_spans)
        c.ev_base = self.ev_base
        c.ev_dots = self.ev_dots
        c.ev_chord = self.ev_chord
        c.ev_grace = self.ev_grace
        c.chords_on_onset = self.chords_on_onset
        c.grace_run = self.grace_run
        c.consec_bar_rest_bars = self.consec_bar_rest_bars
        c.bar_has_pitched = self.bar_has_pitched
        c.bar_has_bar_rest = self.bar_has_bar_rest
        c.sys_count = self.sys_count
        c.sys_has_event = self.sys_has_event
        c.bars_in_system = self.bars_in_system
        c.tokens_in_system = self.tokens_in_system
        c.max_bars_per_system = self.max_bars_per_system
        c.max_tokens_per_system = self.max_tokens_per_system
        c.tokens_since_bar = self.tokens_since_bar
        c.cur_sys_tokens = list(self.cur_sys_tokens)
        c.completed_system_hashes = dict(self.completed_system_hashes)
        c.cur_bar_tokens = list(self.cur_bar_tokens)
        c.sys_bar_hashes = list(self.sys_bar_hashes)
        c.bar_repeat_forbidden_tok = self.bar_repeat_forbidden_tok
        c.comp_num = self.comp_num
        c.comp_digits = self.comp_digits
        c.comp_val = self.comp_val
        c.allow_percussion_tab = self.allow_percussion_tab
        c.vocab_map = self.vocab_map
        return c

    @staticmethod
    def _hash_tokens(toks):
        return hashlib.sha1(' '.join(toks).encode('utf-8')).hexdigest()[:16]

    def _canonical_system_hash(self):
        canon = [t for t in self.cur_sys_tokens if t not in _COSMETIC_SYS_TOKENS]
        if not canon:
            return None
        return self._hash_tokens(canon)

    @staticmethod
    def _is_pure_rest_bar(bar_toks):
        return not any(t in STEPS for t in bar_toks)

    def _record_completed_bar(self):
        if not self.cur_bar_tokens or self._is_pure_rest_bar(self.cur_bar_tokens):
            self.sys_bar_hashes.append(None)
            self.bar_repeat_forbidden_tok = None
            return
        canon = [t for t in self.cur_bar_tokens if t not in _COSMETIC_BAR_TOKENS]
        if not canon:
            self.sys_bar_hashes.append(None)
            self.bar_repeat_forbidden_tok = None
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

    def can_emit_sys(self):
        """Return True iff the current system has >=1 musical event, page is under MAX_SYSTEMS,
        and the current system does not duplicate a previously completed system."""
        if not self.sys_has_event or self.sys_count >= MAX_SYSTEMS - 1:
            return False
        cur_h = self._canonical_system_hash()
        if cur_h is not None and self.completed_system_hashes.get(cur_h, 0) >= 1:
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
        """Return True iff <eos> is legal here (guards against empty trailing systems)."""
        if cur_len is not None and max_len is not None and cur_len >= max_len:
            return True
        if (self.sys_count + 1) >= MAX_SYSTEMS:
            return True
        if self._sys_is_capped():
            return True
        return bool(self.sys_has_event)

    def reset_system(self):
        """Advance to a new system on ``<sys>`` while preserving cross-system cycle history."""
        if self.sys_has_event:
            cur_h = self._canonical_system_hash()
            if cur_h is not None:
                self.completed_system_hashes[cur_h] = (
                    self.completed_system_hashes.get(cur_h, 0) + 1
                )
        self.sys_count += 1
        self.sys_has_event = False
        self.bars_in_system = 0
        self.tokens_in_system = 0
        self.tokens_since_bar = 0
        self.cur_sys_tokens = []
        self.cur_bar_tokens = []
        self.sys_bar_hashes = []
        self.bar_repeat_forbidden_tok = None
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
        self.vscales = [[]]
        self.pend_spans = []
        self.ev_base = None
        self.ev_dots = 0
        self.ev_chord = False
        self.ev_grace = False
        self.chords_on_onset = 0
        self.grace_run = 0
        self.bar_has_pitched = False
        self.bar_has_bar_rest = False
        self.comp_num = 0
        self.comp_digits = 0
        self.comp_val = 0
        self.recent.append('<sys>')
        if len(self.recent) > RECENT_WINDOW_KEEP:
            del self.recent[:-RECENT_WINDOW_KEEP]

    def _grow(self, n):
        while len(self.seg_last) < n:
            self.seg_last.append(False)
            self.depth.append(0)
            self.vcurs.append(0.0)
            self.vscales.append([])

    def _open_scale(self):
        """Pending-event scale: open spans x queued starts (N/A each)."""
        s = 1.0
        for a, n in self.vscales[self.seg]:
            s *= n / a
        for a, n in self.pend_spans:
            s *= n / a
        return s

    def _drained_scale(self):
        """Scale after queuing pending starts into this segment."""
        self.vscales[self.seg].extend(self.pend_spans)
        del self.pend_spans[:]
        s = 1.0
        for a, n in self.vscales[self.seg]:
            s *= n / a
        return s

    def _pending_value(self):
        """Projected quarters of the still-open event (0 if none)."""
        if self.ev_base is None or self.ev_chord or self.ev_grace:
            return 0.0
        return (self.ev_base * (2.0 - 2.0 ** (-self.ev_dots))
                * self._open_scale())

    def _beat_limit(self):
        """Active per-voice beat ceiling (strict=tsig, lax=max(8.0, 2.5*tsig))."""
        if self.tsig is None:
            return None
        if self.strict:
            return self.tsig
        return max(LAX_BAR_MIN_BEATS, LAX_BAR_MULT * self.tsig)

    def _dot_overshoots(self):
        """True if the open event already exceeds the active beat ceiling."""
        limit = self._beat_limit()
        return (limit is not None
                and not self.ev_chord and not self.ev_grace
                and self.ev_base is not None
                and self.vcurs[self.seg] + self._pending_value()
                > limit + _BAR_EPS)

    def _dot_ok(self):
        """True if one more dot keeps the open event within the beat ceiling.

        Pure read (no state mutation) mirroring the projection
        :meth:`step` checks via :meth:`_dot_overshoots` after
        incrementing ``ev_dots``: chord tones add no duration and an
        absent time signature disables gating, so both stay legal.
        """
        limit = self._beat_limit()
        if limit is None:
            return True
        if self.ev_chord or self.ev_grace or self.ev_base is None:
            return True
        val = (self.ev_base * (2.0 - 2.0 ** (-(self.ev_dots + 1)))
               * self._open_scale())
        return self.vcurs[self.seg] + val <= limit + _BAR_EPS

    def _blocks_duration(self, base):
        """True if undotted ``base`` quarters would overshoot the beat ceiling."""
        limit = self._beat_limit()
        if limit is None:
            return False
        if self.strict and round(base * self._open_scale() * 480.0) <= 0:
            return True
        return (self.vcurs[self.seg] + self._pending_value()
                + base * self._open_scale() > limit + _BAR_EPS)

    def _bar_is_full(self):
        """True if no further timed event can fit in the current voice/bar.

        ``vcurs`` lags by one event (it only closes on the next
        boundary token), so fullness must add the open event's
        projected value -- otherwise a full bar of notes still offers
        REST and traps the decoder in rest-mode with every duration
        overshooting (all logits -inf). ``_BAR_EPS`` absorbs tuplet/dot
        float dust.
        """
        limit = self._beat_limit()
        if limit is None:
            return False
        if self.strict:
            return (self.vcurs[self.seg] + self._pending_value()
                    >= limit - _BAR_EPS)
        # In lax mode, full only once even the shortest duration (128th)
        # would exceed the generous safety ceiling.
        return (self.vcurs[self.seg] + self._pending_value()
                + _WRITTEN['128th'] * self._open_scale() > limit + _BAR_EPS)

    def _short_pickup_time_ok(self, prev_tok=None, recent_before=None):
        """Allow time-1-32 / time-1-16 in header, after a bar* token (allowing clef*/key* between), or when already in a short pickup bar."""
        if self.header_open:
            return True
        if self.tsig is not None and self.tsig <= 0.25 + _BAR_EPS:
            return True
        if recent_before is not None:
            for tok in reversed(recent_before):
                if tok in CLEFS or tok in KEYS:
                    continue
                return tok in BAR_LIKES or tok in BAR_LEFT
            return True
        return prev_tok is not None and (prev_tok in BAR_LIKES or prev_tok in BAR_LEFT)

    def _header_attr_ok(self):
        """Allow key*/time-* at system start, bar start, or full-bar end (courtesy change).

        In lax mode (strict=False), also permits legitimate mid-bar key*/time-* changes.
        In strict mode, forbids stray mid-bar key*/time-* emitted after a note/rest inside
        an incomplete bar (0 < cur_used < tsig - eps), while preserving opening headers,
        post-barline changes, and full-bar courtesy changes.
        """
        if not self.strict or self.header_open:
            return True
        cur_used = self.vcurs[self.seg] + self._pending_value()
        if cur_used <= _BAR_EPS:
            return True
        if self.tsig is not None and cur_used >= self.tsig - _BAR_EPS:
            return True
        return False

    def _drain(self):
        self.depth[self.seg] += self.pending
        self.pending = 0

    def _apply_tsig(self, beats):
        if all(v == 0.0 for v in self.vcurs):
            self.tsig = beats
            self.pending_tsig = None
        else:
            self.pending_tsig = beats

    def _bound(self, t):
        if t == TIME_KW:
            if not self._header_attr_ok():
                raise ValueError('spurious mid-bar header %r' % t)
            self.comp_num = 0
            self.comp_digits = 0
            self.comp_val = 0
            return 'time-num'
        if t in CLEFS or t in KEYS or t in TIMES or (
                self.allow_percussion_tab and t in TAB_CLEFS):
            # Mid-stream attribute changes (bekern+): clef/key/time
            # may appear at any measure start (or courtesy end of a full
            # measure). Forbid stray key*/time-* inside an incomplete bar.
            # A time token retunes the books at a true barline (all
            # cursors idle); a courtesy change at full-bar end is stashed
            # and applied at the next barline instead.
            if (t in KEYS or t in TIMES) and not self._header_attr_ok():
                raise ValueError('spurious mid-bar header %r' % t)
            if t in SHORT_PICKUP_TIMES:
                if not self._short_pickup_time_ok(recent_before=self.recent[:-1]):
                    raise ValueError('spurious mid-bar pickup time %r' % t)
            if t in TIMES:
                self._apply_tsig(_beats_of(t))
            return 'bound'
        self.header_open = False
        if t == BAR or t in BAR_RIGHT:
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
            # A stashed courtesy time-sig takes effect for the new
            # measure whose barline just arrived.
            if self.pending_tsig is not None:
                self.tsig = self.pending_tsig
                self.pending_tsig = None
            self.seg_last = [False]
            self.seg = 0
            self.depth = [0]
            self.opened = set()
            self.vcurs = [0.0]
            self.vscales = [[]]
            self.ev_base = None
            self.ev_dots = 0
            self.ev_chord = False
            self.ev_grace = False
            self.chords_on_onset = 0
            self.grace_run = 0
            return 'bound'
        if t in VOICES:
            n = int(t.split('-')[1])
            if n == 0:
                raise ValueError('bad voice token %r' % t)
            if len(self.recent) >= 2 and self.recent[-2] == t:
                raise ValueError('duplicate voice marker %r' % t)
            if self.strict:
                if (n != 1 and (n - 1) not in self.opened
                        and n not in self.opened):
                    raise ValueError('non-sequential voice %r' % t)
            self._grow(n)
            self.seg = n - 1
            self.opened.add(n)
            self.chords_on_onset = 0
            self.grace_run = 0
            return 'bound'
        if t in TUPLET_STARTS:
            if self.strict and (self.depth[self.seg] + self.pending >= 1
                    or self._bar_is_full()):
                raise ValueError('tuplet-start blocked (chained/full bar): %r' % t)
            self.pending += 1
            _, _, a, n = t.split('-')
            self.pend_spans.append((int(a), int(n)))
            return 'bound'
        if t == TUPLET_KW:
            if self.strict and (self.depth[self.seg] + self.pending >= 1
                    or self._bar_is_full()):
                raise ValueError('tuplet-start blocked (chained/full bar): %r' % t)
            self.comp_num = 0
            self.comp_digits = 0
            self.comp_val = 0
            return 'tuplet-a'
        if t == MULTIREST_KW:
            self.sys_has_event = True
            self._drain()
            self._drained_scale()
            self.ev_base = None
            self.ev_dots = 0
            self.seg_last[self.seg] = False
            self.chords_on_onset = 0
            self.grace_run = 0
            self.comp_num = 0
            self.comp_digits = 0
            self.comp_val = 0
            return 'multi-rest'
        if t == TUPLET_STOP:
            if self.depth[self.seg] <= 0:
                raise ValueError('unmatched tuplet-stop')
            self.depth[self.seg] -= 1
            if self.vscales[self.seg]:
                self.vscales[self.seg].pop()
            return 'bound'
        if t == GRACE:
            if self.grace_run >= MAX_CONSECUTIVE_GRACE:
                raise ValueError('grace run cap exceeded (%d)' % MAX_CONSECUTIVE_GRACE)
            self.grace_run += 1
            self.chords_on_onset = 0
            return 'grace'
        if t in DYNS or t in HAIRS or t in BAR_LEFT or (
                self.allow_percussion_tab and t in SLASH_MARKS):
            # Measure dynamics, hairpin wedge events and opening
            # repeats: measure scope, no cursor or span state.
            return 'bound'
        if self.allow_percussion_tab and t == UNPITCHED:
            return 'unpitched'
        if t == CHORD:
            if not self.seg_last[self.seg]:
                raise ValueError('dangling chord marker')
            if self.chords_on_onset >= MAX_CHORDS_PER_ONSET:
                raise ValueError('chord cap exceeded (%d)' % MAX_CHORDS_PER_ONSET)
            self.chords_on_onset += 1
            self.ev_chord = True
            return 'chord'
        if t in DURATIONS or t == REST or t == BAR_REST:
            self.sys_has_event = True
            self._drain()
            if t in DURATIONS:
                if self._blocks_duration(_WRITTEN[t]):
                    raise ValueError(
                        'duration overshoots bar: %.4f + %.4f > %.4f'
                        % (self.vcurs[self.seg],
                           _WRITTEN[t] * self._open_scale(), self._beat_limit()))
                self.alter_seen = False
                self.dot_seen = False
                self.ev_base = _WRITTEN[t]
                self.ev_dots = 0
                self.ev_grace = False
                self.chords_on_onset = 0
                self.grace_run = 0
                self.bar_has_pitched = True
                self.consec_bar_rest_bars = 0
                return 'dur'
            if t == REST:
                self.ev_base = None
                self.ev_dots = 0
                self.chords_on_onset = 0
                self.grace_run = 0
                return 'rest'
            if not self.bar_has_pitched and self.consec_bar_rest_bars >= MAX_CONSECUTIVE_BAR_REST:
                raise ValueError(
                    'consecutive bar-rest cap exceeded (%d)' % MAX_CONSECUTIVE_BAR_REST
                )
            if self.strict and (self._bar_is_full() or self.vcurs[self.seg] > _BAR_EPS):
                raise ValueError('bar-rest blocked in non-empty/full bar')
            self._drained_scale()
            if self.tsig is not None:
                self.vcurs[self.seg] = max(self.vcurs[self.seg],
                                           self.tsig)
            self.ev_base = None
            self.ev_dots = 0
            self.seg_last[self.seg] = False
            self.chords_on_onset = 0
            self.grace_run = 0
            self.bar_has_bar_rest = True
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
            self.ev_dots = 0
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
            self.ev_dots = 0
            self.ev_grace = True
            self.mode = 'dur'
        elif self.mode == 'unpitched':
            if t not in DURATIONS:
                raise ValueError('unpitched without duration %r' % t)
            self._drain()
            if not self.ev_grace and self._blocks_duration(_WRITTEN[t]):
                raise ValueError(
                    'duration overshoots bar: %.4f + %.4f > %.4f'
                    % (self.vcurs[self.seg],
                       _WRITTEN[t] * self._open_scale(), self._beat_limit()))
            self.alter_seen = False
            self.dot_seen = False
            self.ev_base = _WRITTEN[t]
            self.ev_dots = 0
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
                    self.alter_seen = True
                else:
                    self.dot_seen = True
                    self.ev_dots += 1
                    if self._dot_overshoots():
                        raise ValueError(
                            'dot overshoots bar: %.4f > %.4f'
                            % (self.vcurs[self.seg]
                               + self._pending_value(), self.tsig))
                return
            if t in MODS or (self.allow_percussion_tab and (
                    t in NOTEHEADS or t in TAB_STRINGS or t in TAB_FRETS)):
                # Slur/articulation/fermata marks (plus opt-in Gate-D
                # notehead/tab string/fret marks): tail position, no
                # cursor, span or book state.
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
            if t in MODS:
                return
            self._close_event(True)
            self.mode = self._bound(t)
        elif self.mode == 'rest':
            if t not in DURATIONS:
                raise ValueError('bad rest duration %r' % t)
            if self._blocks_duration(_WRITTEN[t]):
                raise ValueError(
                    'duration overshoots bar: %.4f + %.4f > %.4f'
                    % (self.vcurs[self.seg],
                       _WRITTEN[t] * self._open_scale(), self._beat_limit()))
            self.ev_base = _WRITTEN[t]
            self.ev_dots = 0
            self.mode = 'rest-tail'
        elif self.mode == 'rest-tail':
            if t == DOT:
                self.ev_dots += 1
                if self._dot_overshoots():
                    raise ValueError(
                        'dot overshoots bar: %.4f > %.4f'
                        % (self.vcurs[self.seg]
                           + self._pending_value(), self.tsig))
                return
            if t in MODS:
                # Fermatas over rests ride the rest tail.
                return
            self._close_event(False)
            self.mode = self._bound(t)
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
                        self._apply_tsig(self.comp_num * 4.0 / self.comp_val)
                        self.comp_num = 0
                        self.comp_digits = 0
                        self.comp_val = 0
                        self.mode = 'bound'
                elif self.comp_digits == 1:
                    den = self.comp_val * 10 + int(t)
                    if den not in (16, 32, 64):
                        raise ValueError('bad time denominator %d' % den)
                    self._apply_tsig(self.comp_num * 4.0 / den)
                    self.comp_num = 0
                    self.comp_digits = 0
                    self.comp_val = 0
                    self.mode = 'bound'
                else:
                    raise ValueError('time denominator too many digits')
            else:
                if self.comp_digits == 1 and self.comp_val == 1:
                    self._apply_tsig(self.comp_num * 4.0)
                    self.comp_num = 0
                    self.comp_digits = 0
                    self.comp_val = 0
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
        if self.ev_base is not None:
            # Queued tuplet starts open on this event's segment
            # (mirrors decode.rs: spans drain into every event,
            # chord tones included); chord tones add no duration.
            scale = self._drained_scale()
            if not self.ev_chord and not self.ev_grace:
                val = (self.ev_base * (2.0 - 2.0 ** (-self.ev_dots))
                       * scale)
                limit = self._beat_limit()
                if (limit is not None
                        and self.vcurs[self.seg] + val
                        > limit + _BAR_EPS):
                    raise ValueError(
                        'duration overshoots bar: %.4f + %.4f > %.4f'
                        % (self.vcurs[self.seg], val, limit))
                self.vcurs[self.seg] += val
        self.ev_base = None
        self.ev_dots = 0
        self.ev_chord = False
        self.ev_grace = False

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
            if sys_capped:
                return
            if bar_tok_capped:
                allow.add(vm[BAR])
                allow.update(vm[t] for t in BAR_RIGHT if t in vm)
                return
            # Strict parity with step(): a BAR right after a BAR, and
            # a voice marker that step() would reject (non-sequential
            # open or consecutive duplicate), are masked here instead
            # of failing open after the fact. Non-strict keeps the
            # legacy unconditional set.
            if not (self.strict and self.recent
                    and self.recent[-1] in BAR_LIKES):
                allow.add(vm[BAR])
                allow.update(vm[t] for t in BAR_RIGHT if t in vm)
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
            if not (self.strict and (self.depth[self.seg] + self.pending >= 1
                    or self._bar_is_full())):
                allow.update(vm[t] for t in TUPLET_STARTS if t in vm)
                if TUPLET_KW in vm:
                    allow.add(vm[TUPLET_KW])
            for t in DURATIONS:
                if t in vm and not self._blocks_duration(_WRITTEN[t]):
                    allow.add(vm[t])
            # T2.4b: a full bar must not offer REST -- entering
            # rest-mode with every duration overshooting traps the
            # decoder with all logits -inf.
            if not self._bar_is_full():
                allow.add(vm[REST])
            # Iter-7/8 anti-loop: forbid BAR_REST when consecutive pure-rest
            # bars hit MAX_CONSECUTIVE_BAR_REST (both modes), or when strict
            # bar is full / non-empty.
            cur_used = self.vcurs[self.seg] + self._pending_value()
            bar_rest_capped = (
                not self.bar_has_pitched
                and self.consec_bar_rest_bars >= MAX_CONSECUTIVE_BAR_REST
            )
            if not bar_rest_capped and not (
                self.strict and (self._bar_is_full() or cur_used > _BAR_EPS)
            ):
                allow.add(vm[BAR_REST])
                if MULTIREST_KW in vm:
                    allow.add(vm[MULTIREST_KW])
            if self.depth[self.seg] > 0:
                allow.add(vm[TUPLET_STOP])
            if (placed if placed is not None else self.seg_last[self.seg]) \
                    and self.chords_on_onset < MAX_CHORDS_PER_ONSET:
                allow.add(vm[CHORD])
            allow.update(vm[t] for t in CLEFS if t in vm)
            hdr_ok = self._header_attr_ok()
            if hdr_ok:
                allow.update(vm[t] for t in KEYS if t in vm)
                short_ok = self._short_pickup_time_ok(recent_before=self.recent)
                for t in TIMES:
                    if t in vm and (short_ok or t not in SHORT_PICKUP_TIMES):
                        allow.add(vm[t])
                if TIME_KW in vm:
                    allow.add(vm[TIME_KW])
            if self.grace_run < MAX_CONSECUTIVE_GRACE:
                allow.add(vm[GRACE])
            allow.update(vm[t] for t in DYNS | HAIRS | BAR_LEFT if t in vm)
            if self.allow_percussion_tab:
                allow.update(vm[t] for t in TAB_CLEFS if t in vm)
                allow.update(vm[t] for t in SLASH_MARKS if t in vm)
                if UNPITCHED in vm:
                    allow.add(vm[UNPITCHED])

        if self.mode == 'bound':
            add_boundary()
            return allow, eos_ok
        if self.mode == 'grace':
            # Display-only duration follows a grace marker (never
            # strict-gated: grace steals no time).
            allow.update(vm[t] for t in DURATIONS if t in vm)
            if self.allow_percussion_tab and UNPITCHED in vm:
                allow.add(vm[UNPITCHED])
            return allow, False
        if self.mode == 'unpitched':
            for t in DURATIONS:
                if t in vm and (self.ev_grace or not self._blocks_duration(_WRITTEN[t])):
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
                if not (self.strict
                        and (self.alter_seen or self.dot_seen)):
                    allow.update(vm[t] for t in ALTERS)
                if self._dot_ok():
                    allow.add(vm[DOT])
                allow.add(vm[TIE_START])
                allow.add(vm[TIE_STOP])
                allow.update(vm[t] for t in MODS)
                if self.allow_percussion_tab:
                    allow.update(vm[t] for t in NOTEHEADS | TAB_STRINGS | TAB_FRETS
                                 if t in vm)
            add_boundary(placed=True)
            return allow, eos_ok
        if self.mode == 'tiestart':
            if not (bar_tok_capped or sys_capped):
                allow.add(vm[TIE_STOP])
                allow.update(vm[t] for t in MODS)
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
                if not self._blocks_duration(_WRITTEN[t]):
                    allow.add(vm[t])
            return allow, False
        if self.mode == 'rest-tail':
            if not (bar_tok_capped or sys_capped):
                if self._dot_ok():
                    allow.add(vm[DOT])
                allow.update(vm[t] for t in MODS)
            add_boundary()
            return allow, eos_ok
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
        if t == BAR or t in BAR_RIGHT:
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
        if t == TUPLET_STOP:
            if self.depth[self.seg] <= 0:
                raise ValueError('unmatched tuplet-stop')
            self.depth[self.seg] -= 1
            return 'bound'
        if t == GRACE:
            return 'grace'
        if t in DYNS or t in HAIRS or t in BAR_LEFT:
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
