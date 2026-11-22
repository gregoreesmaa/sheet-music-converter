#!/usr/bin/env python3
"""T1.6b domain-separated gate probe: TF top-1 accuracy by token class.

Evaluates a checkpoint on ``data-gen/score-corpus/gate20/sys-dev`` across
4 inputs per system strip, through the TESTED ``train.tf_top1_by_class``
helper:

1. ``clean`` -- real clean strip
2. ``degraded`` -- real degraded strip
3. ``white_blank`` -- true standardized white-paper tensor filled with
   ``train.WHITE`` per channel (the ``train.py`` collate padding
   convention), sized to the system's clean width
4. ``swap`` -- the NEXT system's clean image resized to the same width
   (tests true local spatial grounding vs global manifold statistics)

Reports ``clean_body_acc``, ``clean_body_gap`` (clean minus white_blank),
``clean_swap_gap`` (clean minus swap), ``degraded_body_acc``,
``degraded_body_gap`` (degraded minus white_blank), ``clean_bar_acc``,
and ``degraded_bar_acc`` -- clean and degraded are NEVER pooled.

``--legacy-no-mem-norm`` loads pre-T1.6a checkpoints (e.g.
``sys-v2f-A/ckpt-30000.pt``, 255 tensors, no ``mem_norm``) for the
tight-crop baseline recording: state loads with ``strict=False`` gated on
exactly the two missing ``mem_norm`` keys, and ``encode`` is patched to
the pre-T1.6a path (no DC-centering, no LayerNorm) so the forward matches
training.

Reads only: ``--ckpt`` + ``sys-dev`` (+ frozen ``gate_manifest.json``
read-only for the SHA record). Writes only: ``--out`` JSON + stdout.
Exits 0 once the report is written.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/probe_gate_acc.py \\
        --ckpt tmp-train/mvd-runs/sys-v2f-A/ckpt-30000.pt --legacy-no-mem-norm \\
        --out data-gen/score-corpus/gate20/probe_gate_v2fA_tight.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data  # noqa: E402
import model  # noqa: E402
import train  # noqa: E402

SYS_DEV_DEFAULT = REPO_ROOT / 'data-gen' / 'score-corpus' / 'gate20' / 'sys-dev'
GATE_MANIFEST = (REPO_ROOT / 'data-gen' / 'score-corpus' / 'gate20'
                 / 'gate_manifest.json')

CLASSES = ('header', 'bar', 'body', 'syntax')
DOMAINS = ('clean', 'degraded', 'white_blank', 'swap')

#: The only state-dict keys a pre-T1.6a checkpoint may lack.
LEGACY_MISSING_KEYS = frozenset(('mem_norm.weight', 'mem_norm.bias'))

HEIGHT = 256


# ---------------------------------------------------------------- pure helpers
# (importable by test_probe_gate_acc.py; no checkpoint, GPU, or sys-dev
# needed -- the smoke suite exercises these without the 26-system eval).

def swap_peer(i, n):
    """Index of the swap donor for system ``i`` of ``n`` (next, wrapping)."""
    return (i + 1) % n


def safe_div(ok, n):
    """Accuracy with a zero-count floor (no NaN in the report)."""
    return ok / n if n else 0.0


def make_blank(height, width, device):
    """Standardized white-paper tensor, ``train.WHITE`` per channel.

    Matches the ``train.py`` collate padding convention exactly
    (``WHITE.expand(3, height, w_pad)``), so the blank is what the trunk
    reads as blank paper during training.
    """
    import torch
    return train.WHITE.expand(3, height, width).clone().unsqueeze(0).to(device)


def check_legacy_state(missing, unexpected):
    """Gate a ``strict=False`` load for pre-T1.6a checkpoints.

    Returns ``(True, '')`` iff the ONLY missing keys are the two
    ``mem_norm`` tensors and there are no unexpected keys; otherwise
    ``(False, reason)`` -- a checkpoint missing anything else (or carrying
    foreign keys) must never silently load.
    """
    missing, unexpected = set(missing), set(unexpected)
    if unexpected:
        return False, 'unexpected keys: %s' % sorted(unexpected)[:5]
    extra = missing - set(LEGACY_MISSING_KEYS)
    if extra:
        return False, 'missing non-legacy keys: %s' % sorted(extra)[:5]
    if not missing:
        return False, 'no missing keys: not a pre-T1.6a checkpoint?'
    return True, ''


def apply_legacy_encode(net):
    """Patch a T1.6a ``MvdNet`` instance to the pre-T1.6a ``encode`` path.

    Pre-T1.6a ``encode`` was ``proj.permute + sinusoidal pos`` with no
    DC-centering and no ``mem_norm`` LayerNorm; running a legacy state
    through the new path would NOT reproduce its training forward, so the
    tight-crop baseline would be mismeasured. Verified shape-identical to
    ``git show a498de7:tmp-train/mvd/model.py`` ``encode``.
    """
    import torch  # noqa: F401 -- keeps the patch self-contained

    def _legacy_encode(images):
        feats = net.encoder(images)          # [B, 768, h, w]
        proj = net.proj(feats)               # [B, d, h, w]
        b, d, h, w = proj.shape
        pos = model._sinusoidal_2d(h, w, d, proj.device)
        seq = (proj.permute(0, 2, 3, 1).reshape(b, h * w, d)
               + pos.reshape(1, h * w, d))
        return seq

    # NOTE: plain instance attribute (no __get__ binding): instance
    # attributes are not bound, so net.encode(images) passes images only.
    net.encode = _legacy_encode


def run_variant(net, img, ref, dev):
    """Teacher-forced per-class correct/total counts for one (img, ref)."""
    import torch
    tin = torch.tensor([[data.BOS_ID] + ref[:-1]], dtype=torch.long,
                       device=dev)
    with torch.no_grad():
        logits = net(img, tin)[0]
    return train.tf_top1_by_class(logits, ref)


def summarize(counts):
    """Pool per-class counts per domain; headline metrics, never pooled.

    ``counts`` maps ``domain -> {cls: [ok, n]}``. Returns
    ``(domain_acc, metrics)`` where ``domain_acc[dom][cls]`` is the pooled
    accuracy and ``metrics`` holds the seven headline numbers with clean
    and degraded kept strictly separate.
    """
    domain_acc = {d: {c: safe_div(*counts[d][c]) for c in CLASSES}
                  for d in counts}
    clean = domain_acc['clean']
    degraded = domain_acc['degraded']
    white = domain_acc['white_blank']
    swap = domain_acc['swap']
    metrics = {
        'clean_body_acc': clean['body'],
        'clean_body_gap': clean['body'] - white['body'],
        'clean_swap_gap': clean['body'] - swap['body'],
        'degraded_body_acc': degraded['body'],
        'degraded_body_gap': degraded['body'] - white['body'],
        'clean_bar_acc': clean['bar'],
        'degraded_bar_acc': degraded['bar'],
        # Raw controls, reported so gaps stay interpretable:
        'white_blank_body_acc': white['body'],
        'swap_body_acc': swap['body'],
        'white_blank_bar_acc': white['bar'],
    }
    return domain_acc, metrics


def build_report(ckpt, ckpt_bytes, gate_sha, legacy, height, device, torch_v,
                 systems, per_system, counts):
    """Assemble the JSON-serializable probe report (no I/O)."""
    domain_acc, metrics = summarize(counts)
    per_domain = {}
    for d in DOMAINS:
        per_domain[d] = {c: {'acc': safe_div(*counts[d][c]),
                             'ok': counts[d][c][0],
                             'n': counts[d][c][1]} for c in CLASSES}
    return {
        'probe': 'probe_gate_acc (T1.6b): TF top-1 by class, 4 inputs/system',
        'ckpt': ckpt,
        'ckpt_bytes': ckpt_bytes,
        'legacy_no_mem_norm': bool(legacy),
        'gate_manifest_sha256': gate_sha,
        'test_pages_touched': 0,
        'sys_dev_systems': systems,
        'image_height': height,
        'preprocessing': 'train.load_image (LANCZOS, ImageNet); swap via '
                         'bilinear width-match; blank = train.WHITE',
        'device': device,
        'torch': torch_v,
        'per_domain_class': per_domain,
        'metrics': metrics,
        'per_system': per_system,
    }


# ---------------------------------------------------------------- driver

def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_net(ckpt, legacy, dev):
    """Construct ``MvdNet`` and load ``ckpt`` (strict, or legacy-gated)."""
    import torch
    net = model.MvdNet(vocab=len(data.VOCAB) + 3, anisotropic=True)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    if legacy:
        res = net.load_state_dict(state, strict=False)
        if hasattr(res, 'missing_keys'):  # torch>=1.5 _IncompatibleKeys
            missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
        else:  # pragma: no cover -- very old torch returned a 2-tuple
            missing, unexpected = list(res[0]), list(res[1])
        ok, reason = check_legacy_state(
            [k.replace('module.', '') for k in missing],
            [k.replace('module.', '') for k in unexpected])
        if not ok:
            raise RuntimeError('legacy load refused: %s' % reason)
        apply_legacy_encode(net)
    else:
        net.load_state_dict(state)
    net.to(dev).eval()
    return net


def main():
    ap = argparse.ArgumentParser(description='T1.6b TF-accuracy gate probe')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--sys-dev', default=str(SYS_DEV_DEFAULT))
    ap.add_argument('--height', type=int, default=HEIGHT)
    ap.add_argument('--device', default=None)
    ap.add_argument('--legacy-no-mem-norm', action='store_true',
                    help='load a pre-T1.6a checkpoint (no mem_norm) with '
                         'the legacy encode path for the tight-crop baseline')
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    sys_dev = Path(args.sys_dev)
    manifest = json.loads((sys_dev / 'sys_dev_manifest.json').read_text(
        encoding='utf-8'))
    assert manifest.get('test_pages_touched', 0) == 0, 'test pages touched!'
    systems = sorted(manifest['systems'],
                     key=lambda s: (s['id'], s['page'], s['sys_idx']))
    gate_sha = sha256_file(GATE_MANIFEST)

    net = load_net(args.ckpt, args.legacy_no_mem_norm, device)
    print('ckpt: %s | device: %s | torch: %s | systems: %d | legacy=%s'
          % (args.ckpt, device, torch.__version__, len(systems),
             args.legacy_no_mem_norm), flush=True)

    counts = {d: {c: [0, 0] for c in CLASSES} for d in DOMAINS}
    per_system = []
    n = len(systems)
    for i, s in enumerate(systems):
        stem = '%s-%d-s%d' % (s['id'], s['page'], s['sys_idx'])
        ref = data.encode(data.load_bekern(str(sys_dev / s['bekern'])))
        clean = train.load_image(str(sys_dev / s['clean']),
                                 args.height).unsqueeze(0).to(device)
        degraded = train.load_image(str(sys_dev / s['degraded']),
                                    args.height).unsqueeze(0).to(device)
        blank = make_blank(args.height, clean.shape[3], device)
        donor = systems[swap_peer(i, n)]
        donor_stem = '%s-%d-s%d' % (donor['id'], donor['page'],
                                    donor['sys_idx'])
        swap_img = train.load_image(str(sys_dev / donor['clean']),
                                    args.height).unsqueeze(0).to(device)
        if swap_img.shape[3] != clean.shape[3]:
            swap_img = F.interpolate(swap_img,
                                     size=(args.height, clean.shape[3]),
                                     mode='bilinear', align_corners=False)
        variants = {'clean': clean, 'degraded': degraded,
                    'white_blank': blank, 'swap': swap_img}
        row = {'stem': stem, 'tokens': len(ref), 'swap_donor': donor_stem}
        for dom, img in variants.items():
            got = run_variant(net, img, ref, device)
            row[dom] = {c: {'ok': got[c][0], 'n': got[c][1]} for c in got}
            for c in CLASSES:
                counts[dom][c][0] += got[c][0]
                counts[dom][c][1] += got[c][1]
        per_system.append(row)
        print('%-18s tok=%3d clean=%.3f degr=%.3f blank=%.3f swap=%.3f' % (
            stem, len(ref),
            safe_div(sum(row['clean'][c]['ok'] for c in CLASSES),
                     max(1, sum(row['clean'][c]['n'] for c in CLASSES))),
            safe_div(sum(row['degraded'][c]['ok'] for c in CLASSES),
                     max(1, sum(row['degraded'][c]['n'] for c in CLASSES))),
            safe_div(sum(row['white_blank'][c]['ok'] for c in CLASSES),
                     max(1, sum(row['white_blank'][c]['n'] for c in CLASSES))),
            safe_div(sum(row['swap'][c]['ok'] for c in CLASSES),
                     max(1, sum(row['swap'][c]['n'] for c in CLASSES)))),
            flush=True)

    report = build_report(args.ckpt, os.path.getsize(args.ckpt), gate_sha,
                          args.legacy_no_mem_norm, args.height, device,
                          torch.__version__, len(systems), per_system, counts)
    print(json.dumps(report['metrics'], indent=1), flush=True)
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        tmp = outp.with_suffix(outp.suffix + '.tmp')
        tmp.write_text(json.dumps(report, indent=1), encoding='utf-8')
        os.replace(tmp, outp)
        print('wrote ' + str(outp), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
