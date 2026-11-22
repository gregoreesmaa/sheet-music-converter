"""Spike S6: MaxSAT confusion-lattice repair (post-decode, no retrain).

At this lattice size (K <= 8 beams) exhaustive lexicographic ranking
IS the MaxSAT solve: every candidate is repaired by
:func:`measure_decode.fallback_repair`, then ranked by fewest
``bar-rest`` splices, fewest trimmed measures, tie closure, and
 finally the beam log-prob. Hard constraints (bar math via the
strict walk inside the repair, tie closure) dominate; the log-prob
only breaks exact ties. Pure functions; net-dependent rescoring
lives in the eval wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import measure_decode as mdec  # noqa: E402
import train  # noqa: E402

__all__ = ['ties_balanced', 'rank_candidates', 'row_to_tokens',
           'row_to_ids', 'ids_to_tokens', 'repair_beams', 'stats']

#: Repair-run diagnostics (spike S6 verdict analysis): calls counts
#: strips ranked, switches counts strips where the winner differs
#: from the repaired top beam (the legacy output).
stats = {'calls': 0, 'switches': 0}


def row_to_ids(row, eos_id, nvocab):
    """Beam row -> id list (cut at specials, cutting EOS kept).

    The cutting EOS is retained: ``weighted`` premature-EOS risk and
    rescoring both need it, while trailing PAD/fill carries no
    signal. Rows without EOS keep content until the first special.
    """
    ids = []
    for t in row:
        t = int(t)
        if t == eos_id:
            ids.append(t)
            break
        if t < 0 or t >= nvocab:
            break
        ids.append(t)
    return ids


def ids_to_tokens(ids, nvocab):
    """Id list -> bekern token strings (specials dropped).

    Inverse of :func:`row_to_ids` for ranking: the retained EOS id
    has no vocab string and is dropped here, matching
    ``measure_decode._ids_to_tokens`` cut-at-specials semantics.
    """
    return [mdec.data.VOCAB[i] for i in ids if 0 <= i < nvocab]


def row_to_tokens(row, nvocab):
    """Beam row -> bekern tokens (cut at specials, EOS dropped)."""
    return ids_to_tokens(row_to_ids(row, mdec.data.EOS_ID, nvocab),
                         nvocab)


def ties_balanced(toks):
    """True iff every ``tie-start`` is closed and no lone ``tie-stop``."""
    depth = 0
    for t in toks:
        if t == 'tie-start':
            depth += 1
        elif t == 'tie-stop':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def rank_candidates(cands):
    """Best repaired token list from ``[(toks, logprob), ...]``.

    Lexicographic key
    ``(n_spliced, n_strict_fails, n_touched, tie_open, -logprob)``:
    bar-rest splices dominate (they nuke whole measures), then raw
    strict-closure failures (the bar-math signal — a strict-failing
    measure that loose-closes whole is invisible to trim counts),
    then trims, then tie closure, then the beam posterior. Non-empty
    input required (callers always have >= 1 beam). Returns the
    repaired winner; the repair is applied once here so downstream
    ships exactly what was ranked.
    """
    scored = []
    for toks, lp in cands:
        fixed, touched, spliced = mdec.fallback_repair(toks)
        header, measures, _tb = mdec.split_measures(toks)
        strict_fails = sum(
            1 for mm in measures
            if not mdec.measure_closes(header, mm))
        key = (spliced, strict_fails, touched,
               0 if ties_balanced(fixed) else 1, -lp)
        scored.append((key, fixed))
    scored.sort(key=lambda s: s[0])
    return scored[0][1]


def repair_beams(net, img_tensor, mem_mask, beam_rows, nvocab):
    """Rank one strip's beam lattice, return repaired winner (spike S6).

    ``beam_rows`` are raw id rows (one per beam, rank order from
    ``beam_search(return_all=True)``). Each row is cut to tokens,
    rescored teacher-forced against one shared encode, and ranked by
    :func:`rank_candidates`. Empty rows score ``-inf`` (they never
    win while any content beam exists). Requires grad disabled by
    the caller (eval path runs no-grad).
    """
    mem = net.encode(img_tensor, mem_mask)
    cands = []
    for row in beam_rows:
        ids = row_to_ids(row, net.eos_id, nvocab)
        if not ids:
            cands.append(([], float('-inf')))
            continue
        lp = train.sequence_logprob(net, mem, ids, mem_mask)
        cands.append((ids_to_tokens(ids, nvocab),
                      float(lp.detach())))
    winner = rank_candidates(cands)
    stats['calls'] += 1
    legacy, _, _ = mdec.fallback_repair(cands[0][0])
    if winner != legacy:
        stats['switches'] += 1
    if stats['calls'] % 50 == 0:
        print('lattice_repair: %d strips, %d switches (%.2f)' % (
            stats['calls'], stats['switches'],
            stats['switches'] / stats['calls']), flush=True)
    return winner
