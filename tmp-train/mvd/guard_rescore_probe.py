"""S18 frozen-encoder facet-probe script (GPU-GATED, do NOT run here).

Question: does the frozen ckpt-22000 encoder memory already linearly
separate the facets the flat-AR decoder fumbles (pitch class, duration
class, voice id, attachment presence)? If yes, a light facet head is
warranted; if no, the encoder itself needs retraining and S18 is dead.

Protocol (per facet):
1. Load ckpt-22000, freeze the encoder (no grad, eval mode).
2. Encode the 87 stored strips (rectified strip tensors via the
   eyeball90 geometry path), keep memory states on CPU.
3. Pool memory frames aligned to decoded note events (nearest grid
   cell to the event's staff position); labels from ref bekern.
4. Train a linear probe (closed-form / few LBFGS steps, CPU) per
   facet; report train + held-out-strip accuracy. Facet merit bar:
   held-out accuracy clearly above the majority-class baseline
   (report both; no fixed threshold -- verdict is comparative).

Fail-closed: :func:`main` refuses to run without
``--i-understand-gpu`` AND ``GUARD_GPU_OK=1`` AND a CUDA device.
The operator owns the GPU; this script never launches or resumes
training (encoder stays frozen, probe is a linear fit).

Exact idle-GPU run::

    nvidia-smi  # operator confirms idle first
    set GUARD_GPU_OK=1
    .\\.venv\\Scripts\\python.exe tmp-train\\mvd\\guard_rescore_probe.py \\
        --i-understand-gpu --ckpt D:/tmp/t32s13gateA/ckpt-22000.pt \\
        --strips D:/tmp/eyeball90/f1_87.json --outdir D:/tmp/guard_rescore/probes
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FACETS = ('pitch', 'duration', 'voice', 'attachment')

__all__ = ['FACETS', 'main']


def main(argv=None):
    """Fail-closed entry: refuses without the explicit GPU flag."""
    import argparse as _ap
    ap = _ap.ArgumentParser()
    ap.add_argument('--i-understand-gpu', action='store_true')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--strips', default=None)
    ap.add_argument('--outdir', default=None)
    a = ap.parse_args(argv)
    if not a.i_understand_gpu:
        print('GPU-GATED: re-run with --i-understand-gpu on an '
              'operator-confirmed idle box. Not executed.')
        raise SystemExit(2)
    import os as _os
    if _os.environ.get('GUARD_GPU_OK') != '1':
        raise RuntimeError('GPU-GATED: GUARD_GPU_OK=1 required')
    import torch as _torch
    if not _torch.cuda.is_available():
        raise RuntimeError('GPU-GATED: no CUDA device')
    # Frozen-encoder probe body (GPU only; never runs in this session):
    #   net = M.MvdNet(...); state = torch.load(ckpt); strict compat
    #   load; net.encoder frozen; per-strip rectified tensor ->
    #   memory grid; nearest-cell pooling per ref note event; linear
    #   probe fit per facet; JSON report to outdir.
    raise NotImplementedError(
        'GPU body not executed here (ckpt=%s)' % a.ckpt)


if __name__ == '__main__':
    main()
