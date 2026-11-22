#!/usr/bin/env python3
"""Hybrid rectified-cache builder (Phase 2, Task T2.4).

Transfers exact staff-line geometry onto every training/sys-dev strip so
``train.py --hybrid-cache`` can rectify on the fly (``rectify.py``) and
fuse HeatNet channels (``heat_features.py``) without a 400 GB tensor
cache (76k pairs x [8,256,~1500] fp32; the C: volume holds 9 GB free).

How each strip gets its lines (all deterministic, all validated):

* CLEAN strips: RowNet ``neural_detect_staves`` + inner-line peaks
  (exact on 97/100 sampled strips), fallback ``strip_staff`` blind
  curves, else the whole system (clean + 3 degraded pairs) is skipped.
* DEGRADED strips: marker transfer. Small black dots on the clean
  staff lines are re-run through the ``photo-degrade`` binary with the
  RECORDED per-file ``file_seed`` (single-file ``--seed`` reproduces
  the original warp byte-identically — 5/5 byte-equal pilot,
  2026-11-08; markers do not perturb geometry: every geometric params
  field identical with/without markers). The marker-only difference
  image yields per-line tracks (``marker_tracks``), validated by a
  geometry-identity check plus local-contrast on the real strip.
  Failures skip that degraded pair only.

Stages (run in order; each resumable by existing outputs)::

    markers  clean lines + marker PNGs + worklist (--jobs threads)
    degrade  photo-degrade plain+marked PNG re-runs (--jobs threads)
    tracks   tracks -> curve tables -> hybrid_staff.jsonl (--jobs threads)
    sidecars jsonl tables + corpus strips -> <img>.hybrid.pt (--jobs)
    sysdev   sys-dev clean rows + blind degraded tables -> jsonl
             (reads gate20 only; writes workdir)

Training/eval consume ``hybrid_staff.jsonl`` (one row per image;
``kind=rows`` constant-line tables for clean strips, ``kind=table``
control-point tables for degraded). ``gate20/`` and ``suite31/`` are
only READ (dev pages never test-8); nothing is written outside the
workdir and the sidecar path.

Reference: ``todo/phase-2-hybrid-dewarp.md``, Task T2.4.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import strip_staff as sst  # noqa: E402
import marker_tracks as mt  # noqa: E402
import rectify as R  # noqa: E402 (sidecar validation import)

PDEG = (REPO_ROOT / 'data-gen' / 'photo-degrade' / 'target' / 'release'
        / 'photo-degrade.exe')

# Params fields compared between the original degrade record and the
# marker re-run: pure geometry + sizing. Photometric/artifact fields
# (ink/wear/capture/paper banks, stains, tears, codec) may legitimately
# differ from marker-perturbed ink sampling and are ignored.
GEOM_KEYS = ('src_width', 'src_height', 'rescale_k', 'curl_side', 'curl_px',
             'bend_spine_cx', 'bend_spine_w', 'bend_ripple_fx',
             'bend_ripple_fy', 'bend_ripple_ph', 'perspective_max_frac',
             'corner_offsets', 'warp_fit_scale', 'warp_out_px',
             'rotation_deg', 'rot_scale_fused', 'frame_mode', 'frame_pad_px',
             'frame_cut_px', 'frame_notation_bbox', 'frame_quiet_px',
             'frame_safe_px', 'dist_k1', 'dist_k2', 'dist_fit_scale',
             'focus_mode', 'focus_y', 'focus_h', 'tilt_sigma', 'blur_sigma',
             'scale_x', 'scale_y', 'width', 'height', 'strength',
             'preset', 'train_width')

MARKER_EVERY = 32
MARKER_HALF = 1  # 3x3 dots (minimal ink perturbation; Hough + low
# core tries recover faint cores, keeping warp-divergence low)
MARKER_INK = 140  # draw only on clean ink darker than this
MARKER_EDGE = 12  # inset from image borders (bbox safety)
VARIANTS = ('_d0', '_d1', '_d2')
# Rectified-canvas width cap (px). Normal strips land <= 4000px;
# degenerate tables that survive validation rectify to 10k-1.27M px
# and OOM the trainer. train.py carries the same cap as a batch guard
# (kept as a literal there to avoid importing this builder chain).
MAX_SIDECAR_WIDTH = 8000


def clean_lines(png_path):
    """``(S, lines5, sp, source)`` for a clean strip.

    RowNet boxes + inner-line profile peaks (exact); falls back to the
    blind ``strip_staff`` detector (constant-ified medians); raises
    ``ValueError`` when both fail.
    """
    import build_sys_data as bsd
    img = Image.open(png_path).convert('RGB')
    try:
        staves, space = bsd.neural_detect_staves(png_path)
        gray = img.convert('L')
        probs = bsd.neural_staff_rows(gray, img.size[0], img.size[1])
        h = gray.size[1]
        lines5 = []
        for s in staves:
            y1, y2 = int(round(s['y1'])), int(round(s['y2']))
            lo, hi = max(0, y1 - 2), min(h - 1, y2 + 2)
            inner = _inner_peaks(probs, lo, hi, bsd.NEURAL_THRESHOLD)
            inner = [r for r in inner if y1 < r < y2]
            mg = []
            for p in inner:
                if mg and p - mg[-1] <= 3:
                    if probs[p] > probs[mg[-1]]:
                        mg[-1] = p
                else:
                    mg.append(p)
            if len(mg) == 3:
                lines5.append([float(y1)] + [float(r) for r in mg]
                              + [float(y2)])
            else:
                lines5.append([float(y1 + k * (y2 - y1) / 4.0)
                               for k in range(5)])
        return len(staves), lines5, float(space), 'rownet'
    except (ValueError, OSError):
        pass
    curves, info = sst.detect_strip_curves(img.convert('L'),
                                           _guess_S(png_path))
    med = [[float(np.median(line)) for line in staff]
           for staff in curves]
    return len(curves), med, float(info['sp']), 'stripstaff'


def _inner_peaks(probs, lo, hi, thresh):
    """Row indices in ``(lo, hi)`` that are strict local maxima over the
    ±2 stencil (inner staff-line peaks).

    Rows lacking full stencil context (``r < 2`` or ``r + 2 >= len``)
    can never satisfy the stencil and are excluded -- previously
    ``r = h - 2`` evaluated ``probs[r + 2]`` out of bounds and killed
    the whole markers stage (leland tall strip, h = 3319), while
    ``r = 1`` silently read ``probs[-1]`` through negative wrapping.
    """
    n = len(probs)
    return [r for r in range(max(lo + 1, 2), min(hi, n - 2))
            if probs[r] > thresh
            and probs[r] >= probs[r - 1] and probs[r] > probs[r + 1]
            and probs[r] >= probs[r - 2] and probs[r] >= probs[r + 2]]


def _guess_S(png_path):
    """Staff count for the blind fallback (clean strip only)."""
    import build_sys_data as bsd
    try:
        staves, _ = bsd.neural_detect_staves(png_path)
        return len(staves)
    except (ValueError, OSError):
        pass
    # Last resort: layout-staff classical complete-staff count.
    import subprocess as sp
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        exe = (REPO_ROOT / 'converter' / 'layout-staff' / 'target'
               / 'release' / 'layout-staff.exe')
        r = sp.run([str(exe), '--input', png_path, '--out-dir', td],
                   capture_output=True, text=True)
        if r.returncode == 0:
            outs = glob.glob(os.path.join(td, '*_layout.json'))
            if outs:
                d = json.load(open(outs[0]))
                n = sum(1 for s in d['systems'] for st in s['staves']
                        if st['complete'])
                if 1 <= n <= 4:
                    return n
    raise ValueError('cannot determine S for %s' % png_path)


def draw_markers(png_path, lines5, out_path):
    """Clean strip + black dots every MARKER_EVERY px along staff lines.

    Within each staff, markers are confined to the measured ink span
    (first-to-last ink column across the staff's line rows): staff
    lines are continuous there, so grid positions land on line ink
    even when faint (faint lines need markers most), while the blank
    margins (clef/text zones) stay clean and the notation bbox is
    never extended. The per-strip geometry-identity check backstops
    any residual perturbation by skipping the strip.
    """
    img = Image.open(png_path).convert('RGB')
    a = np.array(img)
    g = np.asarray(img.convert('L'))
    h, w = g.shape
    n = 0
    for staff in lines5:
        rows = [min(h - 1, max(0, int(round(y)))) for y in staff]
        ink_cols = np.nonzero((g[rows, :] < MARKER_INK).any(axis=0))[0]
        if len(ink_cols) == 0:
            continue
        x0 = max(MARKER_EDGE, int(ink_cols[0]) + 2)
        x1 = min(w - MARKER_EDGE, int(ink_cols[-1]) - 2)
        for y in staff:
            yy = int(round(y))
            if not (MARKER_EDGE <= yy < h - MARKER_EDGE):
                continue
            for x in range(x0, x1 + 1, MARKER_EVERY):
                mh = MARKER_HALF
                a[max(0, yy - mh):yy + mh + 1,
                  max(0, x - mh):x + mh + 1] = (0, 0, 0)
                n += 1
    Image.fromarray(a).save(out_path)
    return n


def _table_npts(w_in, stride=8):
    """Control-point count matching ``table_to_curves`` reconstruction."""
    n = len(range(0, w_in, stride))
    return n if (w_in - 1) % stride == 0 else n + 1


def rows_to_table(lines5, w_in):
    """Constant-line rows -> control-point table (uniform sidecar form)."""
    n = _table_npts(int(w_in))
    return {'stride': 8, 'w_in': int(w_in),
            'staves': [[[round(float(y), 2)] * n for y in staff]
                       for staff in lines5]}


def _markers_one(args):
    """One stem: clean lines + marker PNG + rows/worklist payloads.

    Returns ``(stem, kind, payload)`` where kind selects the jsonl
    target (``rows``/``skip``/``work`` list). Pure except the marker
    PNG write (distinct file per stem, pool-safe).
    """
    corpus_s, mk_s, stem = args
    corpus, mk_png = Path(corpus_s), Path(mk_s)
    png = corpus / (stem + '.png')
    try:
        S, lines5, sp, src = clean_lines(str(png))
    except ValueError as e:
        return (stem, 'skip', {'img': stem + '.png', 'skip': True,
                               'reason': 'clean-lines: %s' % e})
    w = Image.open(png).size[0]
    if mk_png.exists():
        mk_png.unlink()
    n_mk = draw_markers(str(png), lines5, str(mk_png))
    table = rows_to_table(lines5, w)
    work_entries = []
    no_params = []
    for v in VARIANTS:
        jp = corpus / (stem + v + '_params.json')
        if not jp.exists():
            no_params.append({'img': stem + v + '.jpg', 'skip': True,
                              'reason': 'no-params'})
            continue
        d = json.loads(jp.read_text(encoding='utf-8'))
        work_entries.append(
            {'img': stem + v + '.jpg', 'clean': stem + '.png',
             'variant': v, 'file_seed': d['file_seed'], 'S': S,
             'sp_clean': round(sp, 2)})
    return (stem, 'ok', {'rows': {'img': stem + '.png', 'S': S,
                                  'kind': 'rows', 'table': table,
                                  'sp': round(sp, 2), 'source': src,
                                  'skip': False},
                         'n_mark': n_mk, 'work': work_entries,
                         'no_params': no_params})


def stage_markers(corpus, workdir, limit=None, prefix=None, jobs=8):
    """Clean lines + marker PNGs + worklist for every bekern stem.

    ``jobs`` threads share one warmed-up RowNet session (onnxruntime
    inference is thread-safe; the module-global lazy session is
    warmed in this thread first so workers never race its creation).
    Results are appended in stem order (``executor.map`` yields in
    order), so the jsonl is bit-identical for any ``jobs`` value.
    """
    import build_gate20_sys_dev as bgate
    bgate.ensure_weights_readable()
    mk_dir = workdir / 'markers'
    mk_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted(glob.glob(str(corpus / '*-bekern.txt')))
    if prefix:
        stems = [s for s in stems
                 if os.path.basename(s).startswith(prefix)]
    if limit:
        stems = stems[:limit]
    side = workdir / 'hybrid_staff.jsonl'
    work = workdir / 'worklist.jsonl'
    done_imgs, done_work = set(), set()
    if side.exists():
        for ln in side.read_text(encoding='utf-8').splitlines():
            if ln.strip():
                done_imgs.add(json.loads(ln).get('img'))
    if work.exists():
        for ln in work.read_text(encoding='utf-8').splitlines():
            if ln.strip():
                e = json.loads(ln)
                done_work.add((e.get('clean'), e.get('variant')))
    todo = []
    for bek in stems:
        stem = os.path.basename(bek)[:-len('-bekern.txt')]
        if (stem + '.png') in done_imgs and all(
                (stem + '.png', v) in done_work for v in VARIANTS):
            continue
        todo.append(stem)
    # Warm the shared RowNet session before threads touch it.
    if todo:
        try:
            clean_lines(str(corpus / (todo[0] + '.png')))
        except ValueError:
            pass
    calls = [(str(corpus), str(mk_dir / (s + '.png')), s) for s in todo]
    n_ok = n_skip = n_mark = 0
    with open(side, 'a', encoding='utf-8') as fs, \
            open(work, 'a', encoding='utf-8') as fw:
        if jobs <= 1:
            results = [_markers_one(c) for c in calls]
        else:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=jobs) as ex:
                results = list(ex.map(_markers_one, calls))
        for stem, kind, pay in results:
            if kind == 'skip':
                fs.write(json.dumps(pay) + '\n')
                n_skip += 1
                continue
            if (stem + '.png') not in done_imgs:
                fs.write(json.dumps(pay['rows']) + '\n')
            n_mark += pay['n_mark']
            n_ok += 1
            for e in pay['work']:
                if (e['clean'], e['variant']) not in done_work:
                    fw.write(json.dumps(e) + '\n')
            for npay in pay['no_params']:
                fs.write(json.dumps(npay) + '\n')
    print('markers: %d systems ok, %d skipped, %d markers drawn'
          % (n_ok, n_skip, n_mark), flush=True)
    return side, work


def _degrade_one(args):
    """Single-file photo-degrade re-run (plain or marked PNG output).

    ``key`` is ``<stem>_<variant>`` (e.g. ``ds0001-1-s0_d0``): every
    variant has its own ``file_seed``/warp, so outputs are per-variant
    (sharing one output name across variants silently keeps only the
    last writer's warp).
    """
    key, tag, seed, in_png, out_dir, strength = args
    # The binary names outputs <input-stem><suffix>: the input stems
    # (corpus clean / marker PNGs) carry NO variant, so the variant
    # rides in the suffix and each warp lands on its own filename.
    stem = key.rsplit('_d', 1)[0]
    suffix = '_d%s_%s' % (key.rsplit('_d', 1)[1], tag)
    out_png = os.path.join(out_dir, stem + suffix + '.png')
    out_js = os.path.join(out_dir, stem + suffix + '_params.json')
    if os.path.exists(out_png) and os.path.exists(out_js):
        return (key, tag, 'cached')
    cmd = [str(PDEG), '--input', in_png, '--out-dir', out_dir,
           '--suffix', suffix, '--seed', str(seed),
           '--format', 'png', '--max-width', '2048',
           '--strength', str(strength)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(REPO_ROOT))
    if r.returncode != 0:
        return (key, tag, 'rc%d: %s' % (r.returncode, (r.stderr or r.stdout)[-200:]))
    return (key, tag, 'ok')


def stage_degrade(workdir, corpus, jobs=8, limit=None, strength=0.5):
    """Re-run warps: plain + marked PNG per worklist entry (parallel)."""
    work = [json.loads(ln) for ln in
            (workdir / 'worklist.jsonl').read_text(encoding='utf-8').splitlines()
            if ln.strip()]
    if limit:
        work = work[:limit]
    mk_dir = workdir / 'markers'
    png_dir = workdir / 'redeg'
    png_dir.mkdir(parents=True, exist_ok=True)
    calls = []
    for e in work:
        stem = e['clean'][:-4]
        key = stem + e['variant']
        calls.append((key, 'plain', e['file_seed'],
                      str(corpus / e['clean']), str(png_dir), strength))
        calls.append((key, 'marked', e['file_seed'],
                      str(mk_dir / (stem + '.png')), str(png_dir), strength))
    n_ok = n_err = n_cached = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        for key, tag, st in ex.map(_degrade_one, calls):
            if st == 'ok':
                n_ok += 1
            elif st == 'cached':
                n_cached += 1
            else:
                n_err += 1
                print('degrade FAIL %s %s: %s' % (key, tag, st), flush=True)
            if (n_ok + n_err + n_cached) % 2000 == 0:
                print('degrade progress: %d ok %d cached %d err' %
                      (n_ok, n_cached, n_err), flush=True)
    print('degrade: %d ok %d cached %d err' % (n_ok, n_cached, n_err),
          flush=True)


def _geom_equal(orig_params, new_params):
    """True when every geometric warp field matches (markers are warp-neutral)."""
    for k in GEOM_KEYS:
        a = json.dumps(orig_params.get(k), sort_keys=True)
        b = json.dumps(new_params.get(k), sort_keys=True)
        if a != b:
            return False, k
    return True, ''


def _tracks_one(args):
    """One worklist entry -> sidecar row (raises nothing; skips as rows)."""
    png_dir_s, corpus_s, e = args
    try:
        return _track_one(Path(png_dir_s), Path(corpus_s), e)
    except ValueError as ex:
        return {'img': e['img'], 'skip': True, 'reason': str(ex)[:160]}


def stage_tracks(workdir, corpus, limit=None, keep_pngs=False, jobs=8):
    """Marker diffs -> validated curve tables -> sidecar rows.

    ``jobs`` threads run ``_track_one`` over the pending entries;
    rows append in worklist order (``executor.map`` yields in order),
    so the jsonl is bit-identical for any ``jobs`` value.
    """
    from scipy import ndimage  # noqa: F401 (backend check for marker_tracks)
    work = [json.loads(ln) for ln in
            (workdir / 'worklist.jsonl').read_text(encoding='utf-8').splitlines()
            if ln.strip()]
    if limit:
        work = work[:limit]
    png_dir = workdir / 'redeg'
    side = workdir / 'hybrid_staff.jsonl'
    done_imgs = set()
    if side.exists():
        for ln in side.read_text(encoding='utf-8').splitlines():
            if ln.strip():
                done_imgs.add(json.loads(ln).get('img'))
    pending = [e for e in work if e['img'] not in done_imgs]
    calls = [(str(png_dir), str(corpus), e) for e in pending]
    if jobs <= 1:
        rows = [_tracks_one(c) for c in calls]
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=jobs) as ex:
            rows = list(ex.map(_tracks_one, calls))
    n_ok = n_skip = 0
    reasons = {}
    with open(side, 'a', encoding='utf-8') as fs:
        for row in rows:
            if row.get('skip'):
                n_skip += 1
                reasons[row['reason'].split(':')[0]] = \
                    reasons.get(row['reason'].split(':')[0], 0) + 1
            else:
                n_ok += 1
            fs.write(json.dumps(row) + '\n')
    print('tracks: %d ok, %d skipped %s' % (n_ok, n_skip, reasons),
          flush=True)
    if not keep_pngs:
        import shutil
        shutil.rmtree(png_dir, ignore_errors=True)
        shutil.rmtree(workdir / 'markers', ignore_errors=True)
        print('tracks: scratch PNGs removed', flush=True)


def sidecar_path_for(img_path, fmt='legacy'):
    """``<stem>.hybrid.pt`` (legacy) or ``<stem>.hybrid.npz`` (compact)."""
    p = Path(img_path)
    suffix = '.hybrid.npz' if fmt == 'compact' else '.hybrid.pt'
    return p.parent / (p.stem + suffix)


def _sidecar_one(args):
    """One jsonl table row -> ``<img>.hybrid.pt`` or ``.hybrid.npz`` (raises nothing).

    Rectifies the corpus strip with its table curves (``rectify.py``
    canonical canvas) and writes either the legacy uncompressed
    ``.hybrid.pt`` (backward-compatible default) or the compact lossless
    ``.hybrid.npz`` (~13-20x smaller, omitting redundant ``gray`` and
    deterministic ``staff``). A table whose ``w_in`` disagrees with the
    image width yields ``(img, 'skip', reason)``: curves sampled at the
    wrong width would silently misplace lines.
    """
    if len(args) == 4:
        corpus_s, row, height, target_sp = args
        fmt, codec = 'legacy', 'png'
    else:
        corpus_s, row, height, target_sp, fmt, codec = args
    from train import atomic_torch_save  # noqa: E402 (tmp+rename helper)
    import torch
    img_name = row['img']
    img_path = Path(corpus_s) / img_name
    dest = sidecar_path_for(img_path, fmt=fmt)
    if dest.exists():
        return (img_name, 'cached', '')
    try:
        img = Image.open(img_path).convert('RGB')
    except OSError as e:
        return (img_name, 'skip', 'unreadable: %s' % e)
    table = row.get('table') or {}
    if int(table.get('w_in', -1)) != img.size[0]:
        return (img_name, 'skip',
                'w_in %s != img width %d'
                % (table.get('w_in'), img.size[0]))
    try:
        curves = sst.table_to_curves(table)
    except (ValueError, KeyError) as e:
        # Malformed table (missing stride/staves, ragged control
        # points): skip, don't crash the stage. Same dense-curve form
        # the eval path uses (eval_sys_gate.rect_staves_for), so train
        # and eval rectify identical geometry; on constant-y rows the
        # restored curves are bit-identical to the control points.
        return (img_name, 'skip', 'table: %s' % e)
    try:
        rect_img, ref = R.rectify_system_strip(
            img, curves, target_sp=target_sp, height=height)
    except (ValueError, TypeError, MemoryError) as e:
        # MemoryError: degenerate table geometry (e.g. collapsed staff
        # lines -> median space ~0.01px -> 1.27M-px canvas attempt,
        # sq7224846-41-s3_d0.jpg). The stage contract is skip-not-crash.
        return (img_name, 'skip', 'rectify: %s' % e)
    # Width cap: tables that rectify "successfully" to absurd canvases
    # (e.g. 162301px for ds0402-1-s2_d2, median space ~0.08px) would
    # later OOM the trainer (batch input alone ~730MB + forward
    # activations on 16GB). Normal strips land <= 4000px; corpus max
    # after the cap: 10 rows skipped, 44705 sidecars kept.
    out_w = rect_img.size[0]
    if out_w > MAX_SIDECAR_WIDTH:
        return (img_name, 'skip',
                'width %d > cap %d' % (out_w, MAX_SIDECAR_WIDTH))
    if fmt == 'compact':
        import compact_sidecar as cs
        try:
            cs.save_compact_sidecar(
                dest,
                rect_img,
                img_name=img_name,
                S=int(row.get('S', 0)),
                sp=float(row.get('sp', 0.0)),
                kind=row.get('kind', ''),
                height=int(height),
                target_sp=float(target_sp),
                codec=codec,
            )
        except OSError as e:
            return (img_name, 'skip', 'write: %s' % e)
        return (img_name, 'ok', '')
    gray = np.array(rect_img.convert('L'), dtype=np.uint8)
    rgb = np.ascontiguousarray(np.array(rect_img, dtype=np.uint8))
    staff = (np.asarray(ref[0], dtype=np.float32) > 0.5).astype(np.uint8)
    blob = {'img': img_name, 'S': int(row.get('S', 0)),
            'sp': float(row.get('sp', 0.0)),
            'kind': row.get('kind', ''), 'height': int(height),
            'rgb': torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
            'gray': torch.from_numpy(gray),
            'staff': torch.from_numpy(staff).unsqueeze(0).contiguous()}
    try:
        atomic_torch_save(blob, dest)
    except OSError as e:
        return (img_name, 'skip', 'write: %s' % e)
    return (img_name, 'ok', '')


def stage_sidecars(workdir, corpus, height=256, target_sp=10.0,
                   limit=None, jobs=8, fmt='legacy', codec='png'):
    """``hybrid_staff.jsonl`` tables -> per-strip sidecars.

    Consumes every row with ``skip == False`` and a ``table`` (both
    ``kind=rows`` clean tables and ``kind=table`` degraded tables);
    already-built sidecars are kept (resume by existence). Rows whose
    image is missing or whose ``w_in`` mismatches are reported, not
    retried: ``train.py --hybrid-cache`` drops pairs without a
    sidecar (skip-only-degraded policy). Pass ``fmt='compact'`` to emit
    compressed ``.hybrid.npz`` sidecars (~13-20x smaller than legacy
    ``.hybrid.pt``).
    """
    side = workdir / 'hybrid_staff.jsonl'
    rows = [json.loads(ln) for ln in
            side.read_text(encoding='utf-8').splitlines() if ln.strip()]
    usable = [r for r in rows
              if not r.get('skip') and isinstance(r.get('table'), dict)]
    if limit:
        usable = usable[:limit]
    calls = [(str(corpus), r, height, target_sp, fmt, codec) for r in usable]
    if jobs <= 1:
        outs = [_sidecar_one(c) for c in calls]
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=jobs) as ex:
            outs = list(ex.map(_sidecar_one, calls))
    n_ok = n_cached = 0
    skips = {}
    for img_name, st, reason in outs:
        if st == 'ok':
            n_ok += 1
        elif st == 'cached':
            n_cached += 1
        else:
            skips[reason.split(':')[0]] = \
                skips.get(reason.split(':')[0], 0) + 1
    print('sidecars: %d built %d cached %d skipped %s'
          % (n_ok, n_cached, sum(skips.values()), skips), flush=True)


def _track_one(png_dir, corpus, e):
    """Sidecar row for one degraded strip (raises ValueError to skip)."""
    stem = e['clean'][:-4]
    key = stem + e['variant']
    img_name = e['img']
    S = e['S']
    A = np.array(Image.open(
        png_dir / ('%s_plain.png' % key)).convert('L')).astype(np.int16)
    B = np.array(Image.open(
        png_dir / ('%s_marked.png' % key)).convert('L')).astype(np.int16)
    if A.shape != B.shape:
        raise ValueError('shape-mismatch: %s vs %s' % (A.shape, B.shape))
    orig = json.loads((corpus / (key + '_params.json')
                       ).read_text(encoding='utf-8'))
    newp = json.loads((png_dir / ('%s_marked_params.json' % key)
                       ).read_text(encoding='utf-8'))
    ok, key = _geom_equal(orig, newp)
    if not ok:
        raise ValueError('warp-drift: %s' % key)
    real = np.array(Image.open(corpus / img_name).convert('L'))
    if real.shape != A.shape:
        raise ValueError('recorded-shape-mismatch')
    w_real = Image.open(corpus / e['clean']).size[0]
    sp_d = max(4.0, min(40.0, e['sp_clean'] * real.shape[1] / w_real))
    curves, info = mt.extract_curves(np.abs(A - B), real, S, sp_d)
    if len(curves) != S * 5:
        raise ValueError('line-count: %d != %d' % (len(curves), S * 5))
    staves = [curves[i * 5:(i + 1) * 5] for i in range(S)]
    table = sst.curves_to_table(staves)
    return {'img': img_name, 'S': S, 'kind': 'table', 'table': table,
            'sp': round(float(sp_d), 2), 'source': 'marker',
            'usage': info.get('usage'),
            'smooth': info.get('smooth_max'),
            'n_blobs': info.get('n_blobs'),
            'fill_resid': info.get('fill_resid'),
            'core': info.get('core'), 'skip': False}


def _blind_ok(gray, staves, sp_prior):
    """Validate blind ``strip_staff`` curves for a sys-dev degraded strip.

    Same optical gates as marker ``_finalize`` minus blob usage (no
    markers here): per-line smoothness ``<= 2.0``, glare-aware
    anti-line floor, and measured spacing within 30% of the clean
    strip's spacing scaled by strip widths. Raises ValueError to skip.
    """
    import marker_tracks as _mt
    gg = np.asarray(gray, dtype=np.float32)
    paper_med = float(np.median(gg))
    diffs = []
    for staff in staves:
        if len(staff) != 5:
            raise ValueError('blind line-count %d != 5' % len(staff))
        arr = [np.asarray(ln, dtype=float) for ln in staff]
        diffs.extend(float(np.median(np.diff(a))) for a in arr)
        for c in arr:
            d2 = np.abs(c[2:] - 2 * c[1:-1] + c[:-2])
            if len(d2) and float(d2.max()) > _mt.MAX_SMOOTH:
                raise ValueError('blind rough line (smooth %.2f)'
                                 % float(d2.max()))
            co, su = _mt.local_contrast(gray, c, sp_prior)
            if (co < _mt.MIN_CONTRAST_HARD
                    and su < paper_med + _mt.GLARE_PAPER_LIFT):
                raise ValueError('blind anti-line (contrast %.1f)' % co)
    sp_got = float(np.median(diffs))
    if not (0.7 * sp_prior <= sp_got <= 1.3 * sp_prior):
        raise ValueError('blind sp %.2f vs prior %.2f'
                         % (sp_got, sp_prior))
    return sp_got


def _global_tilt_search(g, seeds, sp):
    """Joint (offset, slope) fit over ALL seed lines of the strip.

    A strip is one rigid page crop: a single tilt + offset moves
    every staff together (per-staff independent tilts let a staff
    lock onto a horizontal artifact band at slope 0 while its
    siblings tilt, as seen on ds0091-3-s0 staff0). All ``S x 5``
    mapped lines vote together: total darkness-contrast over
    ``dy in ±40`` (2 px) x ``slope in ±0.12`` (0.01). The affine
    seeds already encode relative staff geometry, so one global
    shift suffices; per-line polish absorbs residual perspective.
    Returns ``(dy, slope)``.
    """
    h, w = g.shape
    xs = np.arange(w, dtype=float)
    xi = xs.astype(int)
    off = max(3, int(round(1.5 * sp)))
    xc = xs - w / 2.0
    best = None
    for dy in range(-40, 41, 2):
        for slope100 in range(-12, 13):
            slope = slope100 / 100.0
            tot = 0.0
            for y0 in seeds:
                ys = np.clip((y0 + dy + slope * xc).astype(int),
                             0, h - 1)
                line = g[ys, xi].mean()
                up = g[np.clip(ys - off, 0, h - 1), xi].mean()
                dn = g[np.clip(ys + off, 0, h - 1), xi].mean()
                tot += (up + dn) / 2.0 - line
            if best is None or tot > best[0]:
                best = (tot, dy, slope)
    return best[1], best[2]


def _refine_line(g, y_seed, slope_seed, sp):
    """Per-line polish around the staff fit: dy ±6, slope ±0.02."""
    h, w = g.shape
    xs = np.arange(w, dtype=float)
    xi = xs.astype(int)
    xc = xs - w / 2.0
    off = max(3, int(round(1.5 * sp)))
    best = None
    for dy in range(-6, 7):
        for ds in (-0.02, -0.01, 0.0, 0.01, 0.02):
            slope = slope_seed + ds
            ys = np.clip((y_seed + dy + slope * xc).astype(int), 0, h - 1)
            line = g[ys, xi].mean()
            up = g[np.clip(ys - off, 0, h - 1), xi].mean()
            dn = g[np.clip(ys + off, 0, h - 1), xi].mean()
            sc = (up + dn) / 2.0 - line
            if best is None or sc > best[0]:
                best = (sc, y_seed + dy, slope)
    _, yb, sb = best
    return (yb + sb * xc).astype(float)


def _map_clean_lines(gray, lines5, sy, sp_deg):
    """Affine-mapped clean lines -> refined degraded curves + gate.

    Maps constant clean lines by the strip-height ratio ``sy``, fits
    the strip's offset + tilt jointly over all lines
    (:func:`_global_tilt_search`), polishes each line
    (:func:`_refine_line`), then gates every line (glare-aware
    anti-line floor; linear fits are smooth by construction). Lines
    failing the gate are re-interpolated from passing siblings
    (same-staff, like marker sibling rescue); a staff with fewer
    than 3 passing lines fails the strip. Returns
    ``(staves_curves, sp_got)``; raises ValueError to skip.
    """
    import marker_tracks as _mt
    g = np.asarray(gray, dtype=np.float32)
    paper_med = float(np.median(g))
    seeds_all = [[y * sy for y in staff] for staff in lines5]
    dy, slope = _global_tilt_search(
        g, [y for staff in seeds_all for y in staff], sp_deg)
    xi = np.arange(g.shape[1]).astype(int)
    off = max(3, int(round(1.5 * sp_deg)))
    xc = np.arange(g.shape[1], dtype=float) - g.shape[1] / 2.0
    staves = []
    for seeds in seeds_all:
        # Per-staff offset around the rigid slope: the stripwise
        # affine seed errs differentially per staff under page
        # perspective (staff0 exits the strip top at the global dy).
        best_dy, best_tot = dy, None
        for ddy in range(dy - 30, dy + 31, 2):
            tot = 0.0
            for y0 in seeds:
                ys = np.clip((y0 + ddy + slope * xc).astype(int),
                             0, g.shape[0] - 1)
                line = g[ys, xi].mean()
                up = g[np.clip(ys - off, 0, g.shape[0] - 1), xi].mean()
                dn = g[np.clip(ys + off, 0, g.shape[0] - 1), xi].mean()
                tot += (up + dn) / 2.0 - line
            if best_tot is None or tot > best_tot:
                best_tot, best_dy = tot, ddy
        raw = [_refine_line(g, y + best_dy, slope, sp_deg)
               for y in seeds]
        good = []
        for c in raw:
            # Lines pinned at the image edge are off-strip, not paper:
            # clipped rows read paper contrast and would pass below.
            edge = float(np.mean((c < 2.0) | (c > g.shape[0] - 3.0)))
            if edge > 0.2:
                continue
            co, su = _mt.local_contrast(gray, c, sp_deg)
            if not (co < _mt.MIN_CONTRAST_HARD
                    and su < paper_med + _mt.GLARE_PAPER_LIFT):
                good.append(c)
        if len(good) < 3:
            raise ValueError('mapped staff keeps %d/5 lines' % len(good))
        if len(good) < 5:
            # Re-interpolate failing slots from passing siblings.
            order = np.argsort([c.mean() for c in good])
            good = [good[i] for i in order]
            full = []
            for k in range(5):
                # Nearest-rank mapping: passing lines keep rank order;
                # missing ranks interpolate between neighbors.
                pos = k * (len(good) - 1) / 4.0
                lo, hi = int(pos), min(len(good) - 1, int(pos) + 1)
                wgt = pos - lo
                full.append((1 - wgt) * good[lo] + wgt * good[hi])
            raw = full
        else:
            order = np.argsort([c.mean() for c in raw])
            raw = [raw[i] for i in order]
        staves.append(raw)
    diffs = []
    for staff in staves:
        for k in range(4):
            diffs.append(float(np.median(np.asarray(staff[k + 1])
                                         - np.asarray(staff[k]))))
    return staves, float(np.median(diffs))


def _scaled_clean_fallback(gray, lines5, sy, sp):
    """Tilt-fitted affine-scaled clean lines, clamped to the canvas.

    Last-resort geometry for a degraded strip when mapped-refine and
    the blind detector both fail: clean lines scaled by the strip
    height ratio, shifted by the joint (offset, tilt) fit
    (:func:`_global_tilt_search`), rows clipped to ``[0, h-1]`` so
    rectify always receives on-canvas curves. Returns
    ``(staves_curves, sp_got)`` like :func:`_map_clean_lines` (spacing
    measured off the emitted lines). Never raises — a failed tilt fit
    degrades to a pure affine scale, never a skip.
    """
    g = np.asarray(gray, dtype=np.float32)
    h, w = g.shape
    seeds_all = [[y * sy for y in staff] for staff in lines5]
    try:
        dy, slope = _global_tilt_search(
            g, [y for staff in seeds_all for y in staff], sp)
    except (ValueError, IndexError):
        dy, slope = 0, 0.0
    xc = np.arange(w, dtype=float) - w / 2.0
    staves = []
    for seeds in seeds_all:
        staves.append([np.clip(y0 + dy + slope * xc, 0, h - 1).astype(float)
                       for y0 in seeds])
    diffs = []
    for staff in staves:
        for k in range(4):
            diffs.append(float(np.median(np.asarray(staff[k + 1])
                                         - np.asarray(staff[k]))))
    return staves, float(np.median(diffs))


def _sysdev_one(args):
    """One manifest system -> (clean_row, degraded_row or skip).

    Degraded rows never skip for geometry: mapped-refine, then the
    blind detector, then tilt-fitted scaled clean lines
    (``source='scaled-clean-fallback'``). Only a missing clean
    reference skips the pair.
    """
    sysdev_s, s = args
    sysdev = Path(sysdev_s)
    stem = '%s-%d-s%d' % (s['id'], s['page'], s['sys_idx'])
    clean_img = sysdev / s['clean']
    try:
        S, lines5, sp_clean, src = clean_lines(str(clean_img))
    except ValueError as e:
        sk = {'img': Path(s['clean']).name, 'skip': True,
              'reason': 'clean-lines: %s' % e}
        return (sk, {'img': Path(s['degraded']).name, 'skip': True,
                     'reason': 'clean-lines-missing'})
    w_clean = Image.open(clean_img).size[0]
    clean_row = {'img': Path(s['clean']).name, 'S': S, 'kind': 'rows',
                 'table': rows_to_table(lines5, w_clean),
                 'sp': round(sp_clean, 2), 'source': src, 'skip': False}
    deg_img = sysdev / s['degraded']
    deg_name = Path(s['degraded']).name
    try:
        gray = Image.open(deg_img).convert('L')
        w_deg, h_deg = gray.size
        h_clean = Image.open(clean_img).size[1]
        sp_prior = max(4.0, min(40.0,
                                sp_clean * w_deg / max(1, w_clean)))
        # Primary: seeded mapped-refine (transferred truth). Fallbacks:
        # blind detector with the clean spacing as prior, then
        # tilt-fitted scaled clean lines — degraded rows never skip.
        try:
            staves, sp_got = _map_clean_lines(
                gray, lines5, h_deg / max(1, h_clean), sp_prior)
            source = 'mapped-refine'
        except ValueError:
            try:
                curves, info = sst.detect_strip_curves(
                    gray, S, sp_prior=sp_prior)
                if len(curves) != S:
                    raise ValueError('blind staff-count %d != %d'
                                     % (len(curves), S))
                sp_got = _blind_ok(gray, curves, sp_prior)
                staves = curves
                source = 'stripstaff-blind'
            except ValueError:
                staves, sp_got = _scaled_clean_fallback(
                    gray, lines5, h_deg / max(1, h_clean), sp_prior)
                source = 'scaled-clean-fallback'
        deg_row = {'img': deg_name, 'S': S, 'kind': 'table',
                   'table': sst.curves_to_table(staves),
                   'sp': round(sp_got, 2), 'source': source,
                   'skip': False}
    except ValueError as e:
        deg_row = {'img': deg_name, 'skip': True,
                   'reason': 'sysdev-deg: %s' % e}
    return (clean_row, deg_row)


def stage_sysdev(sysdev_dir, workdir, jobs=8):
    """``sys-dev`` strips -> ``hybrid_staff.jsonl`` (READS gate20 only).

    Clean strips take the RowNet rows path (exact truth, same as
    corpus ``markers``); degraded strips are page-crops without
    ``file_seed`` warps, so marker transfer cannot apply — they take
    mapped-refine off the clean lines, then the blind ``strip_staff``
    detector with the clean strip's spacing as prior (optical gates of
    :func:`_blind_ok`), then tilt-fitted scaled clean lines
    (``source='scaled-clean-fallback'``) so degraded rows never skip.
    Output jsonl lives in ``workdir`` (OUTSIDE gate20; nothing in
    gate20 is written). Every other gate20 file is only read.
    """
    import build_gate20_sys_dev as bgate
    bgate.ensure_weights_readable()
    manifest = json.loads((sysdev_dir / 'sys_dev_manifest.json').read_text(
        encoding='utf-8'))
    assert manifest.get('test_pages_touched', 0) == 0, \
        'test pages touched!'
    systems = sorted(manifest['systems'],
                     key=lambda s: (s['id'], s['sys_idx']))
    calls = [(str(sysdev_dir), s) for s in systems]
    if jobs <= 1:
        outs = [_sysdev_one(c) for c in calls]
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=jobs) as ex:
            outs = list(ex.map(_sysdev_one, calls))
    side = workdir / 'hybrid_staff.jsonl'
    n_clean = n_deg = n_skip = 0
    reasons = {}
    with open(side, 'a', encoding='utf-8') as fs:
        for clean_row, deg_row in outs:
            for row in (clean_row, deg_row):
                fs.write(json.dumps(row) + '\n')
                if row.get('skip'):
                    n_skip += 1
                    reasons[row['reason'].split(':')[0]] = \
                        reasons.get(row['reason'].split(':')[0], 0) + 1
                elif row.get('kind') == 'rows':
                    n_clean += 1
                else:
                    n_deg += 1
    print('sysdev: %d clean rows, %d degraded tables, %d skipped %s'
          % (n_clean, n_deg, n_skip, reasons), flush=True)


def main():
    ap = argparse.ArgumentParser(description='T2.4 hybrid cache builder')
    ap.add_argument('--stage', required=True,
                    choices=('markers', 'degrade', 'tracks', 'sidecars',
                             'sysdev'))
    ap.add_argument('--corpus', default='tmp-train/mvd-sysdata-full')
    ap.add_argument('--workdir',
                    default='tmp-train/mvd-sysdata-full/hybrid')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--prefix', default=None)
    ap.add_argument('--strength', type=float, default=0.5)
    ap.add_argument('--height', type=int, default=256,
                    help='rectified canvas height for sidecars (must '
                         'equal train.py --height)')
    ap.add_argument('--sysdev', default=None,
                    help='sys-dev dir for the sysdev stage '
                         '(default: data-gen/score-corpus/gate20/sys-dev)')
    ap.add_argument('--keep-pngs', action='store_true')
    ap.add_argument('--format', dest='sidecar_format', default='legacy',
                    choices=('legacy', 'compact'),
                    help='sidecar format for --stage sidecars: legacy '
                         '(.hybrid.pt) or compact (.hybrid.npz, ~13-20x '
                         'smaller)')
    ap.add_argument('--codec', default='png', choices=('png', 'webp'),
                    help='lossless image codec for --format compact')
    args = ap.parse_args()
    corpus, workdir = Path(args.corpus), Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if args.stage == 'markers':
        stage_markers(corpus, workdir, limit=args.limit, prefix=args.prefix,
                      jobs=args.jobs)
    elif args.stage == 'degrade':
        stage_degrade(workdir, corpus, jobs=args.jobs, limit=args.limit,
                      strength=args.strength)
    elif args.stage == 'sidecars':
        stage_sidecars(workdir, corpus, height=args.height, limit=args.limit,
                       jobs=args.jobs, fmt=args.sidecar_format,
                       codec=args.codec)
    elif args.stage == 'sysdev':
        sysdev = (Path(args.sysdev) if args.sysdev else
                  (REPO_ROOT / 'data-gen' / 'score-corpus' / 'gate20'
                   / 'sys-dev'))
        stage_sysdev(sysdev, workdir, jobs=args.jobs)
    else:
        stage_tracks(workdir, corpus, limit=args.limit,
                     keep_pngs=args.keep_pngs, jobs=args.jobs)


if __name__ == '__main__':
    main()
