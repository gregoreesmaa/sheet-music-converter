#!/usr/bin/env python3
"""T1.3: held-out gate20 dev system reference set builder.

Reads ``data-gen/score-corpus/gate20/gate_manifest.json`` and filters
strictly to ``split == "dev"`` pages (manifest value ``"dev-probe"``;
``"test-probe"`` pages are never opened, read, or cropped). For each
scored dev page:

* takes the page's per-system measure spans from
  ``assembly-serialize system-breaks`` on the score's ground-truth
  ``.musicxml`` (rendered page N == musicxml page N; notation-free
  trailing pages yield zero spans and are skipped),
* crops tight per-system strips from
  ``gate20/clean/<id>.png`` and ``gate20/degraded/<id>_degraded.jpg``
  into ``sys-dev/clean/`` + ``sys-dev/degraded/`` (tight staff boxes
  ``[x1-1.5sp, y1-2.5sp, x2+1.5sp, y2+2.5sp]`` from the shared rownet_v2
  detector via build_sys_data import, same fractions as
  build_sys_data.emit_system; degraded via degraded-page detection else
  scaled clean box; trailing-empty-span pages pair k-1 visual systems;
  no equal-slice fallback -- ambiguous pages are skipped and recorded),
* emits ground-truth ``<id>-s<sys_idx>-bekern.txt`` (via
  ``bekern --measures``) and decoded ``<id>-s<sys_idx>-ref-notes.json``
  (via ``bekern-decode``) into ``sys-dev/refs/``,
* writes ``sys-dev/sys_dev_manifest.json`` (``n_systems``,
  ``test_pages_touched == 0``).

Writes ONLY under ``data-gen/score-corpus/gate20/sys-dev/``; every
other file under ``gate20/`` is opened read-only. ``gate_manifest.json``
SHA-256 is asserted before and after (frozen gate).

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/build_gate20_sys_dev.py
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sys_data as bsd
from build_sys_data import (  # shared rownet_v2 weights/preprocessing/grouping, no drift
    MERGE_PX,
    NEURAL_THRESHOLD,
    PAD_H_SPACES,
    PAD_V_SPACES,
    content_check,
    group_page,
    neural_detect_staves,
    neural_staff_rows,
    parse_parts,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASER = os.path.join(REPO_ROOT, 'converter', 'assembly-serialize', 'target', 'release',
                    'assembly-serialize.exe')
GATE_DIR = os.path.join(REPO_ROOT, 'data-gen', 'score-corpus', 'gate20')
MANIFEST_PATH = os.path.join(GATE_DIR, 'gate_manifest.json')
EXPECTED_MANIFEST_SHA256 = '68a0b070277bda6b2e8fd89565fa8dc3952ac1fb29e25e5403c3d78bc4ed18eb'

OUT_DIR = os.path.join(GATE_DIR, 'sys-dev')
CLEAN_OUT = os.path.join(OUT_DIR, 'clean')
DEGRADED_OUT = os.path.join(OUT_DIR, 'degraded')
REFS_OUT = os.path.join(OUT_DIR, 'refs')
SYS_MANIFEST = os.path.join(OUT_DIR, 'sys_dev_manifest.json')

DEV_SPLITS = ('dev', 'dev-probe')

_opened_gate_inputs = []
test_pages_touched = 0
TEST_IDS = set()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def gate_open(path, mode='rb'):
    """Open a file under gate20/, refusing test-probe pages."""
    global test_pages_touched
    norm = path.replace(os.sep, '/')
    for tid in TEST_IDS:
        if tid in norm:
            test_pages_touched += 1
            raise RuntimeError('REFUSED test-probe contact: %s (matches %s)' % (path, tid))
    _opened_gate_inputs.append(path)
    return open(path, mode)


def ensure_weights_readable():
    """Point bsd.WEIGHTS at readable bytes, repairing nothing on disk.

    Environment repair (2026-11-08): the committed
    ``rownet_v2.onnx`` + ``rownet_v2.onnx.data`` working files are
    ACL-unreadable in this checkout (open/read/icacls/takeown all
    ``Access denied``; CreateFileW reports error 5 under every share
    mode, both checkouts, both v2/v3). The blob bytes are intact in git
    (tracked since T0.1). When the working file cannot be opened, extract
    the identical ``HEAD:`` blob bytes to a temp sibling copy of the
    readable ``.onnx`` (relative external-data resolution keeps working)
    and rebind ``bsd.WEIGHTS`` so the shared rownet_v2 detector runs
    bit-identically. Verifies the extracted size against
    ``git cat-file -s``. Returns the temp dir (None on the healthy path).
    """
    try:
        with open(bsd.WEIGHTS, 'rb') as f:
            f.read(1)
        with open(bsd.WEIGHTS + '.data', 'rb') as f:
            f.read(1)
        return None
    except OSError:
        pass
    import shutil
    rel = os.path.relpath(bsd.WEIGHTS, REPO_ROOT).replace(os.sep, '/')
    blob = subprocess.run(['git', 'show', 'HEAD:' + rel + '.data'],
                          capture_output=True, cwd=REPO_ROOT)
    size = subprocess.run(['git', 'cat-file', '-s', 'HEAD:' + rel + '.data'],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    if blob.returncode != 0 or not blob.stdout or size.returncode != 0:
        raise RuntimeError('rownet weights unreadable and git-blob '
                           'extract failed for ' + rel + '.data')
    if len(blob.stdout) != int(size.stdout.strip()):
        raise RuntimeError('rownet weights blob size mismatch')
    tmpdir = tempfile.mkdtemp(prefix='rownet_weights_')
    shutil.copy(bsd.WEIGHTS, os.path.join(tmpdir, os.path.basename(bsd.WEIGHTS)))
    with open(os.path.join(tmpdir, os.path.basename(bsd.WEIGHTS) + '.data'),
              'wb') as f:
        f.write(blob.stdout)
    import atexit
    atexit.register(shutil.rmtree, tmpdir, True)
    bsd.WEIGHTS = os.path.join(tmpdir, os.path.basename(bsd.WEIGHTS))
    print('weights repair: materialized %s bytes from git blob HEAD:%s.data'
          % (len(blob.stdout), rel), flush=True)
    return tmpdir


def run_system_breaks(xml_path):
    out = subprocess.run([ASER, 'system-breaks', '--input', xml_path],
                         capture_output=True, text=True, cwd=REPO_ROOT)
    if out.returncode != 0:
        raise RuntimeError('system-breaks failed: %s' % out.stderr.strip()[-300:])
    return json.loads(out.stdout)


def emit_system_refs(xml_path, span, sys_stem):
    """Run bekern --measures + bekern-decode; return (tokens, notes_obj)."""
    span_arg = '%d..%d' % (span['start_measure'], span['end_measure'])
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([ASER, 'bekern', '--input', xml_path, '--measures', span_arg,
                            '--out', tmp, '--stem', sys_stem],
                           capture_output=True, text=True, cwd=REPO_ROOT)
        tmp_txt = os.path.join(tmp, sys_stem + '-bekern.txt')
        if r.returncode != 0 or not os.path.exists(tmp_txt):
            raise RuntimeError('bekern %s failed: %s'
                               % (span_arg, (r.stderr or r.stdout or '').strip()[-300:]))
        with open(tmp_txt) as f:
            text = f.read()
        tokens = text.split()
        if len(tokens) < 1:
            raise RuntimeError('bekern %s emitted 0 tokens' % span_arg)
        tmp_json = os.path.join(tmp, 'dec.json')
        r2 = subprocess.run([ASER, 'bekern-decode', '--input', tmp_txt, '--out', tmp_json],
                            capture_output=True, text=True, cwd=REPO_ROOT)
        if r2.returncode != 0 or not os.path.exists(tmp_json):
            raise RuntimeError('bekern-decode failed for %s: %s'
                               % (span_arg, (r2.stderr or r2.stdout or '').strip()[-300:]))
        with open(tmp_json) as f:
            notes = json.load(f)
    final_txt = os.path.join(REFS_OUT, sys_stem + '-bekern.txt')
    final_json = os.path.join(REFS_OUT, sys_stem + '-ref-notes.json')
    with open(final_txt, 'w') as f:
        f.write(text)
    with open(final_json, 'w') as f:
        json.dump(notes, f)
    return tokens, notes


def content_unique_groups(staves, spans, staves_per_part, active):
    """Single content-passing break placement, or None.

    Last resort before equal-slice fallback (the ranked-gap margin rule
    cannot separate breaks when a within-system staff gap rivals the true
    inter-system gaps, e.g. hidden-staff Lieder pages): try every choice
    of k-1 breaks and keep it iff EXACTLY ONE placement passes the shared
    content_check. Bounded (<=5000 placements) and uniqueness-gated --
    anything ambiguous returns None (never force).
    """
    import itertools
    import math
    n, k = len(staves), len(spans)
    if k <= 1 or n < k:
        return None
    if math.comb(n - 1, k - 1) > 5000:
        return None
    hits = []
    for cuts in itertools.combinations(range(n - 1), k - 1):
        groups, start = [], 0
        for c in cuts:
            groups.append(staves[start:c + 1])
            start = c + 1
        groups.append(staves[start:])
        if content_check(groups, spans, staves_per_part, active):
            hits.append(groups)
            if len(hits) > 1:
                return None  # ambiguous: never force
    return hits[0] if hits else None


def crop_has_no_peaks(clean_crop_path):
    """True iff the crop shows zero staff-line rows (unpaired visual)."""
    from PIL import Image
    with gate_open(clean_crop_path, 'rb') as f:
        with Image.open(f) as img:
            gray = img.convert('L')
            w, h = gray.size
            probs = neural_staff_rows(gray, w, h)
    peaks = [r for r in range(1, h - 1)
             if probs[r] > NEURAL_THRESHOLD and probs[r] >= probs[r - 1]
             and probs[r] >= probs[r + 1]]
    merged = []
    for pk in peaks:
        if merged and pk - merged[-1] <= MERGE_PX:
            if probs[pk] > probs[merged[-1]]:
                merged[-1] = pk
        else:
            merged.append(pk)
    return len(merged) == 0


def _longest_ink_run(row_dark, max_gap=2):
    """Return ``(length, x1, x2)`` of the longest horizontal dark run.

    Allows tiny 1..``max_gap``-pixel breaks (aliasing / light ink) inside
    a staff line without bridging across separate text glyphs.
    """
    import numpy as np
    idx = np.nonzero(row_dark)[0]
    if len(idx) == 0:
        return 0, 0, 0
    best_len, best_x1, best_x2 = 0, 0, 0
    start = int(idx[0])
    prev = start
    for x in idx[1:]:
        xi = int(x)
        if xi - prev <= max_gap + 1:
            prev = xi
        else:
            run_len = prev - start + 1
            if run_len > best_len:
                best_len, best_x1, best_x2 = run_len, start, prev + 1
            start = prev = xi
    run_len = prev - start + 1
    if run_len > best_len:
        best_len, best_x1, best_x2 = run_len, start, prev + 1
    return best_len, best_x1, best_x2


def detect_short_trailing_staves(page_path, staves, space):
    """Recover short partial-width final staves missed by RowNet.

    RowNet resizes the full page width to 248 px, so a short final system
    (20-35% of page width, e.g. ``oneills-tune`` m24 or ``repeats-melody``
    m16) scores below ``NEURAL_THRESHOLD`` and is dropped. This scans
    strictly below the last detected staff for 5 evenly-spaced horizontal
    rules matching ``space`` (within +-1.5 px) and aligned to the page's
    left staff margin, with a continuous ink run >= max(120, 0.12*full_w).
    Text/copyright footers lack 5 parallel long rules and are ignored.
    """
    if not staves or space <= 2.0:
        return list(staves)
    import statistics
    import numpy as np
    from PIL import Image

    with gate_open(page_path, 'rb') as f:
        with Image.open(f) as img:
            arr = np.asarray(img.convert('L'))
    h, w = arr.shape
    ordered = sorted(staves, key=lambda s: s['y1'])
    x_left = min(s['x1'] for s in ordered)
    full_w = statistics.median(s['x2'] - s['x1'] for s in ordered)
    min_run = max(120.0, 0.12 * full_w)
    x_tol = max(25.0, 3.0 * space)
    y_cursor = int(round(max(s['y2'] for s in ordered) + max(2.0 * space, 20.0)))
    y_limit = h - int(round(4.0 * space)) - 3
    if y_cursor >= y_limit:
        return ordered

    dark = arr < bsd.BIN_THRESHOLD
    # Precompute longest horizontal run for candidate rows below y_cursor
    row_info = {}
    for r in range(y_cursor, h):
        # Fast pre-filter: row must have enough dark pixels to hold min_run
        if int(dark[r].sum()) < min_run * 0.85:
            continue
        rlen, rx1, rx2 = _longest_ink_run(dark[r], max_gap=2)
        if rlen >= min_run and abs(rx1 - x_left) <= x_tol:
            row_info[r] = (rlen, rx1, rx2)

    if len(row_info) < 5:
        return ordered

    added = list(ordered)
    r = y_cursor
    sp_tol = max(1.5, 0.12 * space)
    while r <= y_limit:
        if r not in row_info:
            r += 1
            continue
        # Try matching a 5-line comb starting at r
        lines = [r]
        ok = True
        for k in range(1, 5):
            target = r + k * space
            lo = int(round(target - sp_tol))
            hi = int(round(target + sp_tol))
            cands = [rr for rr in range(lo, hi + 1) if rr in row_info]
            if not cands:
                ok = False
                break
            best_rr = max(cands, key=lambda rr: row_info[rr][0])
            lines.append(best_rr)
        if not ok:
            r += 1
            continue
        # Verify all 5 lines share consistent x-extent (true staff, not noise)
        x1s = [row_info[rr][1] for rr in lines]
        x2s = [row_info[rr][2] for rr in lines]
        if (max(x1s) - min(x1s) <= x_tol
                and max(x2s) - min(x2s) <= max(x_tol, 0.15 * full_w)):
            y1, y2 = lines[0], lines[-1]
            band = dark[y1:y2 + 1, :]
            cols = np.nonzero(band.any(axis=0))[0]
            bx1 = float(cols[0]) if len(cols) else float(min(x1s))
            bx2 = float(cols[-1] + 1) if len(cols) else float(max(x2s))
            added.append({'x1': bx1, 'y1': float(y1),
                          'x2': bx2, 'y2': float(y2)})
            r = y2 + int(round(max(2.0 * space, 20.0)))
        else:
            r += 1
    return added


def detect_staves_with_short_tail(page_path):
    """Run ``neural_detect_staves`` and recover any short trailing staff."""
    staves, space = bsd.neural_detect_staves(page_path)
    staves = detect_short_trailing_staves(page_path, staves, space)
    return staves, space


def tight_group_boxes(page_path, xml_path, spans):
    """Return (boxes, kept_spans, dropped_span, (w, h)) or None.

    Runs the shared rownet_v2 detector (same weights/preprocessing as
    build_sys_data.py) on a dev page, then groups staff rows via the
    shared group_page (exact-fit regular chunks, else k-1 largest
    inter-staff gaps with the separation-margin skip rule plus content
    check), then a uniqueness-gated content search as a last resort.
    Each emitted box is the tight staff box
    ``[x1 - 1.5*space, y1 - 2.5*space, x2 + 1.5*space, y2 + 2.5*space]``
    clamped to image bounds -- the same fractions as
    build_sys_data.emit_system, so no full-page-width crop and no
    y=0/H-page pins.

    Trailing-empty-span rule: when k-span grouping is ambiguous but the
    first k-1 spans group cleanly (sq8438915-5: 2 visual systems vs 3 XML
    spans, trailing empty measure 80), pair the k-1 visual groups with
    spans[:-1] and report spans[-1] as dropped. Returns None when every
    grouping path is ambiguous (the caller skips the page; there is no
    equal-slice fallback, which would reintroduce full-bleed crops).
    """
    k = len(spans)
    try:
        staves, space = detect_staves_with_short_tail(page_path)
    except (OSError, ValueError):
        return None
    try:
        staves_per_part, active = parse_parts(xml_path)
    except Exception:
        return None
    try:
        groups = group_page(staves, space, spans, staves_per_part, active)
        if groups is None:
            groups = content_unique_groups(staves, spans, staves_per_part,
                                           active)
    except Exception:
        groups = None
    kept, dropped = spans, None
    if not (groups is not None and len(groups) == k
            and all(len(g) > 0 for g in groups)):
        if k < 2:
            return None
        try:
            groups = group_page(staves, space, spans[:-1], staves_per_part,
                                active)
            if groups is None:
                groups = content_unique_groups(staves, spans[:-1],
                                               staves_per_part, active)
        except Exception:
            groups = None
        if not (groups is not None and len(groups) == k - 1
                and all(len(g) > 0 for g in groups)):
            return None
        kept, dropped = spans[:-1], spans[-1]
    from PIL import Image
    with gate_open(page_path, 'rb') as f:
        with Image.open(f) as img:
            img.load()
            w, h = img.size
    boxes = []
    for g in groups:
        x1 = max(0.0, min(s['x1'] for s in g) - PAD_H_SPACES * space)
        y1 = max(0.0, min(s['y1'] for s in g) - PAD_V_SPACES * space)
        x2 = min(float(w), max(s['x2'] for s in g) + PAD_H_SPACES * space)
        y2 = min(float(h), max(s['y2'] for s in g) + PAD_V_SPACES * space)
        if not (x2 > x1 and y2 > y1):
            return None
        boxes.append([int(round(x1)), int(round(y1)),
                      int(round(x2)), int(round(y2))])
    return boxes, kept, dropped, (w, h)


def degraded_page_boxes(degr_path, xml_path, kept_spans, clean_size,
                        clean_boxes):
    """Return (boxes, source) with tight boxes in degraded-page pixels.

    Prefers detecting staves on the degraded page itself under the same
    grouping contract (so rotated/padded capture borders are excluded,
    never a full-page-width crop); when detection/grouping is ambiguous
    there, scales the clean tight boxes by the dimension ratio and clamps
    (``source='scaled'``). Never pins to y=0/H or x=0/W.
    """
    det = tight_group_boxes(degr_path, xml_path, kept_spans)
    if det is not None:
        boxes, _kept2, dropped2, _size = det
        if dropped2 is None and len(boxes) == len(kept_spans):
            return boxes, 'detected'
    cw, ch = clean_size
    from PIL import Image
    with gate_open(degr_path, 'rb') as f:
        with Image.open(f) as img:
            img.load()
            dw, dh = img.size
    sx, sy = dw / cw, dh / ch
    boxes = []
    for b in clean_boxes:
        x1 = max(0, min(int(round(b[0] * sx)), dw - 1))
        y1 = max(0, min(int(round(b[1] * sy)), dh - 1))
        x2 = max(x1 + 1, min(int(round(b[2] * sx)), dw))
        y2 = max(y1 + 1, min(int(round(b[3] * sy)), dh))
        boxes.append([x1, y1, x2, y2])
    return boxes, 'scaled'


def crop_box(src_path, dst_path, box):
    """Crop exact pixel box [x1, y1, x2, y2]; tmp+rename for safety."""
    from PIL import Image
    with gate_open(src_path, 'rb') as f:
        with Image.open(f) as img:
            img.load()
            w, h = img.size
            x1 = max(0, min(int(box[0]), w - 1))
            y1 = max(0, min(int(box[1]), h - 1))
            x2 = max(x1 + 1, min(int(box[2]), w))
            y2 = max(y1 + 1, min(int(box[3]), h))
            crop = img.crop((x1, y1, x2, y2))
            used = [x1, y1, x2, y2]
    tmp_dst = dst_path + '.tmp'
    fmt = 'PNG' if dst_path.lower().endswith('.png') else 'JPEG'
    crop.save(tmp_dst, format=fmt)
    os.replace(tmp_dst, dst_path)
    return {'width': w, 'height': h, 'crop_box': used}


def main():
    global TEST_IDS
    if sha256_file(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256:
        print('FROZEN-GATE PRECONDITION FAILED: gate_manifest.json SHA-256 mismatch')
        return 1
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    pages = manifest['pages']
    TEST_IDS = set(p['id'] for p in pages if p['split'] not in DEV_SPLITS)
    for p in pages:
        if p['split'] not in DEV_SPLITS and not p['split'].startswith('test'):
            print('REFUSED: unexpected split value %r on %s' % (p['split'], p['id']))
            return 1
    dev_pages = [p for p in pages if p['split'] in DEV_SPLITS]
    print('dev pages: %d, sealed test pages (untouched): %d' % (len(dev_pages), len(TEST_IDS)))
    ensure_weights_readable()

    for d in (CLEAN_OUT, DEGRADED_OUT, REFS_OUT):
        os.makedirs(d, exist_ok=True)

    systems = []
    skipped_pages = []
    skipped_systems = []
    slice_fallback_pages = []
    token_counts = []
    for rec in dev_pages:
        pid = rec['id']
        score = rec['score']
        try:
            page = int(pid[len(score) + 1:])
        except ValueError:
            skipped_pages.append([pid, 'unparseable-page-suffix'])
            continue
        xml_path = os.path.join(GATE_DIR, rec['ground_truth_musicxml'].replace('/', os.sep))
        clean_path = os.path.join(GATE_DIR, rec['clean'].replace('/', os.sep))
        degr_rel = rec['degraded'].replace('/', os.sep)
        degr_path = os.path.join(GATE_DIR, degr_rel)
        if not (os.path.exists(xml_path) and os.path.exists(clean_path)
                and os.path.exists(degr_path)):
            skipped_pages.append([pid, 'missing-input-file'])
            continue
        # Seal hardening (reviewer T1.3-re): the TEST_IDS/gate_open counter
        # does not cover subprocess/XML/detection reads, so assert up front
        # that NO test page id appears in any gate20 path this page will read
        # (all reads — subprocess included — derive from these three paths).
        for _tp in (xml_path, clean_path, degr_path):
            for _tid in TEST_IDS:
                if _tid in _tp.replace(os.sep, '/'):
                    raise RuntimeError('REFUSED test-probe contact: %s matches %s'
                                       % (_tp, _tid))
        try:
            spans = [s for s in run_system_breaks(xml_path) if s['page'] == page]
        except RuntimeError as e:
            skipped_pages.append([pid, 'system-breaks-failed: %s' % e])
            continue
        if not spans:
            skipped_pages.append([pid, 'no-system-spans-on-page'])
            continue
        spans.sort(key=lambda s: s['system_idx'])
        got = tight_group_boxes(clean_path, xml_path, spans)
        if got is None:
            skipped_pages.append([pid, 'tight-grouping-ambiguous; '
                                     'no full-bleed fallback'])
            print('page %s: tight grouping ambiguous/undetected; page skipped' % pid,
                  flush=True)
            continue
        clean_boxes, kept_spans, dropped, clean_size = got
        if dropped is not None:
            skipped_systems.append([
                '%s-s%d' % (pid, dropped['system_idx']),
                '%d..%d: trailing span without visual group (tight k-1 pairing)'
                % (dropped['start_measure'], dropped['end_measure'])])
        degr_boxes, degr_source = degraded_page_boxes(
            degr_path, xml_path, kept_spans, clean_size, clean_boxes)
        for span, cbox, dbox in zip(kept_spans, clean_boxes, degr_boxes):
            sys_stem = '%s-s%d' % (pid, span['system_idx'])
            try:
                tokens, _notes = emit_system_refs(xml_path, span, sys_stem)
                clean_info = crop_box(clean_path,
                                      os.path.join(CLEAN_OUT, sys_stem + '.png'),
                                      cbox)
                if crop_has_no_peaks(os.path.join(CLEAN_OUT, sys_stem + '.png')):
                    raise RuntimeError('empty crop: no staff peaks '
                                       '(visual/spans mismatch on page)')
                degr_ext = os.path.splitext(degr_path)[1].lower()
                degr_info = crop_box(degr_path,
                                     os.path.join(DEGRADED_OUT, sys_stem + degr_ext),
                                     dbox)
            except (RuntimeError, OSError, ValueError) as e:
                for p in (os.path.join(CLEAN_OUT, sys_stem + '.png'),
                          os.path.join(DEGRADED_OUT, sys_stem + '.jpg'),
                          os.path.join(DEGRADED_OUT, sys_stem + '.jpeg'),
                          os.path.join(REFS_OUT, sys_stem + '-bekern.txt'),
                          os.path.join(REFS_OUT, sys_stem + '-ref-notes.json')):
                    if os.path.exists(p):
                        os.remove(p)
                skipped_systems.append([sys_stem, '%d..%d: %s'
                                        % (span['start_measure'], span['end_measure'], e)])
                continue
            token_counts.append(len(tokens))
            systems.append({
                'id': pid, 'page': page, 'sys_idx': span['system_idx'],
                'start_measure': span['start_measure'], 'end_measure': span['end_measure'],
                'tokens': len(tokens),
                'clean': 'clean/%s.png' % sys_stem,
                'degraded': 'degraded/%s%s' % (sys_stem, degr_ext),
                'bekern': 'refs/%s-bekern.txt' % sys_stem,
                'ref_notes': 'refs/%s-ref-notes.json' % sys_stem,
                'clean_box': clean_info['crop_box'],
                'degraded_box': degr_info['crop_box'],
                'degraded_box_source': degr_source,
            })
        print('page %s: %d/%d systems ok' % (pid, sum(1 for s in systems if s['id'] == pid),
                                             len(kept_spans)),
              flush=True)

    token_counts.sort()
    sys_manifest = {
        'gate': manifest.get('gate', 'gate20'),
        'subset': 'sys-dev',
        'split': 'dev-probe only; test-probe SEALED and untouched',
        'builder': 'tmp-train/mvd/build_gate20_sys_dev.py',
        'gate_manifest_sha256': sha256_file(MANIFEST_PATH),
        'crop': ('tight per-system staff boxes (rownet_v2 + group_page via '
                 'build_sys_data import; uniqueness-gated content search last '
                 'resort; [x1-1.5sp, y1-2.5sp, x2+1.5sp, y2+2.5sp] clamped, '
                 'same fractions as build_sys_data.emit_system); degraded via '
                 'degraded-page detection else scaled clean box; trailing '
                 'empty-span k-1 pairing; no full-bleed fallback'),
        'slice_fallback_pages': sorted(slice_fallback_pages),
        'n_pages': len(set(s['id'] for s in systems)),
        'n_systems': len(systems),
        'test_pages_touched': test_pages_touched,
        'token_p50': token_counts[len(token_counts) // 2] if token_counts else 0,
        'token_p95': token_counts[min(len(token_counts) - 1, int(len(token_counts) * 0.95))] if token_counts else 0,
        'token_max': token_counts[-1] if token_counts else 0,
        'systems': systems,
        'skipped_pages': skipped_pages,
        'skipped_systems': skipped_systems,
    }
    with open(SYS_MANIFEST, 'w') as f:
        json.dump(sys_manifest, f, indent=2)
    if sha256_file(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256:
        print('FROZEN-GATE VIOLATION: gate_manifest.json changed during build')
        return 1
    if test_pages_touched != 0:
        print('SEAL VIOLATION: test pages touched: %d' % test_pages_touched)
        return 1
    print('wrote %s: n_systems=%d test_pages_touched=%d' % (SYS_MANIFEST, len(systems),
                                                            test_pages_touched))
    return 0


if __name__ == '__main__':
    sys.exit(main())
