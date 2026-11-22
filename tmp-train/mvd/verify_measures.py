"""S17/T4.1 measure-local render-and-compare verifier, slice 1: scoring.

Projects each decoded measure's notes (via S13c cross-attention
x-centroids) back onto observed HeatNet notehead-peak x-positions and
scores the disagreement. Slice 2 (re-decode of flagged measures with
visual-prior constraints) builds on these scores. Pure functions
(except for caller-supplied tensors); no RNG, no weights.
"""

import torch

__all__ = [
    'note_peak_xs',
    'measure_spans',
    'measure_residual',
    'score_strip_measures',
    'pick_min_residual_hyp',
]


def note_peak_xs(note_map, thresh=0.5):
    """Sorted unique peak-column fractions from a Note prob map.

    ``note_map`` is ``[H, W]`` of HeatNet Note probabilities. A peak
    is a 3x3 local maximum at or above ``thresh`` (same detector as
    ``count_notehead_peaks``); chord notes sharing a column yield one
    x. Returns ``x / W`` fractions in increasing order.
    """
    import torch.nn.functional as F
    x = note_map.detach().float()
    pooled = F.max_pool2d(x.unsqueeze(0).unsqueeze(0), 3, 1, 1)
    pooled = pooled.squeeze(0).squeeze(0)
    peaks = (pooled == x) & (x >= thresh)
    cols = peaks.any(dim=0)
    w = x.shape[1]
    return [c / w for c in range(w) if bool(cols[c])]


def measure_spans(ids, bar_id, step_ids, pad_id):
    """``(start, end)`` token spans per note-bearing measure.

    Splits the id sequence on ``bar_id``; a span is kept iff it holds
    at least one id from ``step_ids`` (headers/pickup rests with no
    notes never verify). ``pad_id`` positions terminate the strip.
    """
    step_ids = set(step_ids)
    spans = []
    start = 0
    for i, t in enumerate(ids):
        if t == pad_id:
            break
        if t == bar_id:
            if any(s in step_ids for s in ids[start:i]):
                spans.append((start, i))
            start = i + 1
    end = len(ids)
    for k in range(start, len(ids)):
        if ids[k] == pad_id:
            end = k
            break
    if any(s in step_ids for s in ids[start:end]):
        spans.append((start, end))
    return spans


def measure_residual(exp_xs, obs_xs, tol):
    """``(mean_match, miss_frac, extra_frac)`` for one measure.

    Every expected note x matches its nearest observed peak x;
    expected notes farther than ``tol`` from any peak are misses, and
    observed peaks farther than ``tol`` from every expected note are
    extras. Empty expected: ``(0.0, 0.0, extra)`` with ``extra`` 1.0
    iff any peak is observed. Empty observed with expected notes:
    miss 1.0 and infinite mean distance (fail loud, never 0).
    """
    if not exp_xs:
        return (0.0, 0.0, 1.0 if obs_xs else 0.0)
    if not obs_xs:
        return (float('inf'), 1.0, 0.0)
    dists = [min(abs(e - o) for o in obs_xs) for e in exp_xs]
    miss = sum(1 for d in dists if d > tol) / len(dists)
    extra_hits = [min(abs(o - e) for e in exp_xs) for o in obs_xs]
    extra = sum(1 for d in extra_hits if d > tol) / len(extra_hits)
    return (sum(dists) / len(dists), miss, extra)


def score_strip_measures(ids, xc, bar_id, step_ids, pad_id, obs_xs,
                         tol=0.02):
    """Per-measure residual rows for one decoded strip.

    ``ids`` is the decoded token-id list, ``xc`` its ``[T]`` (or
    ``[1, T]``) centroid vector. Returns one dict per note-bearing
    span: ``span``, ``n_notes``, ``mean_match``, ``miss``, ``extra``,
    ``residual = (miss + extra) / 2`` (``inf`` mean_match maps to a
    1.0 residual so empty-ink measures always flag).
    """
    if torch.is_tensor(xc):
        xc = xc.reshape(-1).tolist()
    rows = []
    for (s, e) in measure_spans(ids, bar_id, step_ids, pad_id):
        exp = [xc[i] for i in range(s, e) if ids[i] in step_ids]
        mean_match, miss, extra = measure_residual(exp, obs_xs, tol)
        residual = (miss + extra) / 2.0
        if mean_match == float('inf'):
            residual = 1.0
        rows.append({'span': (s, e), 'n_notes': len(exp),
                     'mean_match': mean_match, 'miss': miss,
                     'extra': extra, 'residual': residual})
    return rows


def pick_min_residual_hyp(hyps, bar_id, step_ids, pad_id, obs_xs,
                          tol=0.02):
    """Index of the beam hyp with the lowest mean measure residual.

    ``hyps`` is ``[(ids, xc)]`` per beam (``xc`` the centroid vector
    for those ids). Ties break toward the lower beam index.
    """
    best, best_score = 0, None
    for i, (ids, xc) in enumerate(hyps):
        rows = score_strip_measures(ids, xc, bar_id, step_ids, pad_id,
                                    obs_xs, tol)
        score = (sum(r['residual'] for r in rows) / len(rows)
                 if rows else 1.0)
        if best_score is None or score < best_score:
            best, best_score = i, score
    return best
