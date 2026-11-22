"""S18 poisoned-prefix flip-test harness (CPU-runnable).

S18 asks whether a guarded (PrefixState-constrained) continuation
recovers from a poisoned prefix faster than the flat decoder. Protocol:

1. :func:`inject_flip` poisons a stored hyp prefix at token ``t``
   (pitch shift or duration swap from a flip map).
2. :func:`cascade_length` runs a next-token ``step_fn`` from the
   poisoned prefix and counts emitted tokens until the output resyncs
   with the reference suffix (``resync_k`` consecutive matches).
3. :func:`recovery_ratio` / :func:`merits_full_run` apply the S18 bar:
   flat cascade > 3x the guarded cascade merits a full GPU run.

``step_fn`` is injectable so the harness mechanics are CPU-testable
with stub decoders (see ``test_guard_rescore.py``). Wiring a real
ckpt-22000 net as ``step_fn`` is GPU-GATED: use
:func:`real_net_step_fn` only on an idle-GPU box; it fail-closes
otherwise. Nothing here launches or resumes training.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

__all__ = [
    'inject_flip',
    'cascade_length',
    'recovery_ratio',
    'merits_full_run',
    'compare_recovery',
    'real_net_step_fn',
]

S18_NEED_RATIO = 3.0


def inject_flip(tokens, t, flip_map):
    """Poison ``tokens`` at index ``t`` via ``flip_map``.

    Returns (prefix, old, new): the poisoned prefix ``tokens[:t] +
    [new]`` plus the replaced token. Raises IndexError for an
    out-of-range ``t``, KeyError when ``tokens[t]`` has no flip.
    """
    tokens = list(tokens)
    if not 0 <= t < len(tokens):
        raise IndexError('flip index %d out of range %d' % (t, len(tokens)))
    old = tokens[t]
    if old not in flip_map:
        raise KeyError('no flip for %r' % old)
    new = flip_map[old]
    return tokens[:t] + [new], old, new


def cascade_length(step_fn, poisoned_prefix, reference_suffix,
                   max_steps=64, resync_k=3):
    """Tokens emitted until ``resync_k`` consecutive reference matches.

    ``step_fn(prefix) -> token`` is called with the growing prefix;
    resync is declared when the last ``resync_k`` emitted tokens equal
    ``reference_suffix[:resync_k]``. Returns the emission count, or
    ``max_steps`` when resync never happens inside the budget.
    """
    if resync_k <= 0:
        raise ValueError('resync_k must be positive')
    if len(reference_suffix) < resync_k:
        raise ValueError('reference suffix shorter than resync_k')
    target = list(reference_suffix[:resync_k])
    prefix = list(poisoned_prefix)
    window = []
    for n in range(1, max_steps + 1):
        tok = step_fn(prefix)
        prefix = prefix + [tok]
        window.append(tok)
        if len(window) > resync_k:
            del window[0]
        if window == target:
            return n
    return max_steps


def recovery_ratio(flat_cascade, guarded_cascade):
    """Flat-over-guarded cascade ratio (inf when guarded is 0)."""
    if guarded_cascade <= 0:
        return float('inf')
    return flat_cascade / guarded_cascade


def merits_full_run(flat_cascade, guarded_cascade,
                    need=S18_NEED_RATIO):
    """S18 bar: flat cascade strictly more than ``need``x guarded."""
    return recovery_ratio(flat_cascade, guarded_cascade) > need


def compare_recovery(flat_fn, guarded_fn, cases, max_steps=64,
                     resync_k=3):
    """Run flip cases under both continuations; report ratios.

    ``cases``: [(name, poisoned_prefix, reference_suffix)]. Returns
    ``{'cases': [{name, flat, guarded, ratio, merits}], ...}``.
    """
    out = {'cases': []}
    for name, prefix, ref in cases:
        flat = cascade_length(flat_fn, prefix, ref, max_steps, resync_k)
        guarded = cascade_length(guarded_fn, prefix, ref, max_steps,
                                 resync_k)
        ratio = recovery_ratio(flat, guarded)
        out['cases'].append({'name': name, 'flat': flat,
                             'guarded': guarded, 'ratio': ratio,
                             'merits': merits_full_run(flat, guarded)})
    return out


def real_net_step_fn(ckpt_path, device='cuda'):
    """Build a ckpt step_fn (GPU-GATED, fail-closed without GPU).

    Requires ``GUARD_GPU_OK=1`` in the environment (operator confirms
    the box is idle) and a CUDA device; otherwise raises
    RuntimeError. Exact idle-GPU run::

        nvidia-smi  # operator confirms idle first
        set GUARD_GPU_OK=1
        .\\.venv\\Scripts\\python.exe tmp-train\\mvd\\guard_rescore_fliptest.py \\
            --ckpt D:/tmp/t32s13gateA/ckpt-22000.pt --cases D:/tmp/guard_rescore/flip_cases.json
    """
    import os as _os
    if _os.environ.get('GUARD_GPU_OK') != '1':
        raise RuntimeError('GPU-GATED: set GUARD_GPU_OK=1 only on an '
                           'operator-confirmed idle box')
    import torch  # noqa: E402
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU-GATED: no CUDA device')
    raise NotImplementedError(
        'GPU-GATED stub: wire ckpt %s autoregressive stepping here; '
        'not executed in this session' % ckpt_path)


def main(argv=None):
    """CLI entry (GPU path only; stub harness runs live on CPU)."""
    import argparse as _ap
    ap = _ap.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--cases', default=None)
    a = ap.parse_args(argv)
    if not a.ckpt or not a.cases:
        print('need --ckpt and --cases (GPU-GATED run)')
        raise SystemExit(2)
    import json as _json
    cases = _json.load(open(a.cases))
    step = real_net_step_fn(a.ckpt)
    res = compare_recovery(step, step,
                           [(c['name'], c['prefix'], c['ref'])
                            for c in cases])
    print(_json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
