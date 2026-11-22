#!/usr/bin/env python3
"""T1.2: engraver-exact system-crop + degraded dataset builder.

For each big500 score: pair ``assembly-serialize system-breaks`` measure
spans with visual staff-system boxes (exact symbox staves when present,
else neural row-model staff detection), crop system strips, emit exact
``bekern --measures`` targets (decode-validated, 1..512 tokens), render 3
photo-degraded JPEG variants per strip, and write ``manifest.json``.

Exclusion guard: any page stem listed in ``gate20/exclusion.json``
(``do_not_train``) or ``tmp-train/mvd-data/quarantine.txt`` is skipped
before processing. gate20 pages are never read or written here.

Neural fallback note: the spec calls for
``layout-staff --model neural --weights rownet_v2.onnx``, but the checked-in
release binary was built WITHOUT the ``onnx`` feature
(``unknown model 'neural'; rebuild with --features onnx``). This script
therefore runs the same weights with the same preprocessing through the
venv's ``onnxruntime`` (width-248 Triangle resize, binarize @140, edge
replicate, 33-row windows, batch 512, sigmoid) -- the exact contract of
``converter/layout-staff/src/onnx_model.rs``. Verified bit-for-bit on
truth: 40/40 staff-line rows exact on ds0001-1.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/build_sys_data.py --limit-scores 60
"""

import argparse
import concurrent.futures
import glob
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASER = os.path.join(REPO_ROOT, 'converter', 'assembly-serialize', 'target', 'release', 'assembly-serialize.exe')
PDEG = os.path.join(REPO_ROOT, 'data-gen', 'photo-degrade', 'target', 'release', 'photo-degrade.exe')
WEIGHTS = os.path.join(REPO_ROOT, 'tmp-train', 'row-retrain', 'rownet_v2.onnx')
EXCLUSION_JSON = os.path.join(REPO_ROOT, 'data-gen', 'score-corpus', 'gate20', 'exclusion.json')
QUARANTINE_TXT = os.path.join(REPO_ROOT, 'tmp-train', 'mvd-data', 'quarantine.txt')

NEURAL_THRESHOLD = 0.3  # row-prob peak floor (mirrors layout-staff --neural-threshold)
BIN_THRESHOLD = 140     # ink binarization level (mirrors layout-staff --threshold)
MERGE_PX = 3            # merge row peaks at most this far apart
GAP_CUT_SPACES = 3.0    # peak-gap multiple of staff space that starts a new staff
MARGIN_RATIO = 1.25     # ranked-break acceptance: min break >= 1.25x max non-break (ds0008-3 passes at 1.36; wrong-break contests sit ~1.0, e.g. ds0005-1 at 0.91); literal 6.0-space joining was tried and matches zero of 30 symbox pages (voice-piano gaps run 6.5-16sp), so ranked gaps + margin arbitrate and the strict gate skips the rest
PAD_V_SPACES = 2.5      # vertical crop padding (ledger lines, stems, beams, clefs)
PAD_H_SPACES = 1.5      # horizontal crop padding
SHED_SEARCH_PX = 3      # +-rows around a peak when measuring solid-rule density
SHED_EDGE_RATIO = 0.7   # shed an edge peak below this fraction of the group's best row (beach-prayer: 0.57 vs 0.86 = 0.66; true lines sit ~1.0)
SHED_DISCARD_DENSITY = 0.6  # drop a >5 group outright when its best row is below this: a text block (title/composer, best ~0.49), not a staff
SHED_MIN_LINE_DENSITY = 0.7  # ...only when the best row looks like a solid rule
MIN_TOKENS = 1
MAX_TOKENS = 512        # model context gate
DEGRADE_VARIANTS = (('_d0', 0), ('_d1', 1), ('_d2', 2))
DEGRADE_MAX_WIDTH = 2048


# ---------------------------------------------------------------- exclusion

def load_excluded():
    """Return the set of excluded page/score stems (build-time guard only)."""
    with open(EXCLUSION_JSON) as f:
        excl = json.load(f)
    excluded = set(excl.get('do_not_train', []))
    if os.path.exists(QUARANTINE_TXT):
        with open(QUARANTINE_TXT) as f:
            for line in f:
                stem = line.strip()
                if stem and not stem.startswith('#'):
                    excluded.add(stem)
    else:
        print('warning: quarantine.txt missing, using exclusion.json only', flush=True)
    return excluded


def page_excluded(page_stem, excluded):
    if page_stem in excluded:
        return True
    # bare score-level entries (no -<page> suffix) exclude all pages of that score
    if '-' in page_stem:
        score = page_stem.rsplit('-', 1)[0]
        if score in excluded:
            return True
    return False


# ---------------------------------------------------------------- musicxml

def parse_parts(xml_path):
    """Return (staves_per_part, active) where active[p] is a list over that
    part's measures (ordinal, matching system_breaks part-1 convention) of
    whether the part sounds (any non-rest note) in the measure."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    parts = root.findall('part')
    staves_per_part = []
    active = []
    for part in parts:
        measures = part.findall('measure')
        staves = 1
        part_active = []
        for m in measures:
            sounds = False
            for note in m.findall('note'):
                if note.find('rest') is not None:
                    continue
                staves_el = note.find('staff')
                if staves_el is not None and staves_el.text:
                    try:
                        staves = max(staves, int(staves_el.text))
                    except ValueError:
                        pass
                sounds = True
            part_active.append(sounds)
        staves_per_part.append(staves)
        active.append(part_active)
    return staves_per_part, active


def run_system_breaks(xml_path):
    out = subprocess.run([ASER, 'system-breaks', '--input', xml_path],
                         capture_output=True, text=True, cwd=REPO_ROOT)
    if out.returncode != 0:
        raise RuntimeError('system-breaks failed on %s: %s' % (xml_path, out.stderr.strip()))
    return json.loads(out.stdout)


def part_active_in_span(active, part_idx, start, end, n_measures):
    """1-based inclusive measure span -> does part_idx sound anywhere in it?"""
    ms = active[part_idx]
    for m in range(start - 1, end):
        if 0 <= m < len(ms) and ms[m]:
            return True
    return False


# ---------------------------------------------------------------- detection

def load_symbox_staves(box_path, img_w, img_h):
    """Return (staves, staff_space) with staves as {x1,y1,x2,y2} in PNG px."""
    with open(box_path) as f:
        d = json.load(f)
    sx, sy = 1.0, 1.0
    pw, ph = d.get('png_size', [img_w, img_h])
    if abs(pw - img_w) > 1 or abs(ph - img_h) > 1:
        sx, sy = img_w / pw, img_h / ph
    staves = []
    gaps = []
    for s in d['staves']:
        x1, y1, x2, y2 = s['bbox']
        staves.append({'x1': x1 * sx, 'y1': y1 * sy, 'x2': x2 * sx, 'y2': y2 * sy})
        ls = s['lines']
        for i in range(len(ls) - 1):
            gaps.append((ls[i + 1] - ls[i]) * sy)
    staves.sort(key=lambda s: s['y1'])
    return staves, statistics.median(gaps)


_ORT_SESSION = None


def neural_staff_rows(gray, width, height):
    """Row staff-line probabilities via rownet_v2.onnx (onnx_model.rs parity)."""
    global _ORT_SESSION
    import numpy as np
    from PIL import Image
    if _ORT_SESSION is None:
        import onnxruntime as ort
        _ORT_SESSION = ort.InferenceSession(WEIGHTS, providers=['CPUExecutionProvider'])
    small = gray.resize((248, height), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32)
    bina = (a < BIN_THRESHOLD).astype(np.float32)
    pad = np.concatenate([np.repeat(bina[:1], 16, axis=0), bina,
                          np.repeat(bina[-1:], 16, axis=0)], axis=0)
    probs = np.zeros(height, dtype=np.float32)
    for s in range(0, height, 512):
        e = min(s + 512, height)
        n = e - s
        inp = np.stack([pad[s + i:s + i + 33] for i in range(n)])[:, None, :, :].astype(np.float32)
        (logit,) = _ORT_SESSION.run(['logit'], {'patch': inp})
        probs[s:e] = 1.0 / (1.0 + np.exp(-logit))
    return probs


def neural_detect_staves(png_path):
    """Return (staves, staff_space) from the neural row model.

    staves: {x1,y1,x2,y2} in PNG px (x from ink extent of the staff band).
    Raises ValueError when the peak pattern is not clean 5-line staves.
    """
    import numpy as np
    from PIL import Image
    gray = Image.open(png_path).convert('L')
    w, h = gray.size
    probs = neural_staff_rows(gray, w, h)
    peaks = [r for r in range(1, h - 1)
             if probs[r] > NEURAL_THRESHOLD and probs[r] >= probs[r - 1] and probs[r] >= probs[r + 1]]
    merged = []
    for p in peaks:
        if merged and p - merged[-1] <= MERGE_PX:
            if probs[p] > probs[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)
    if len(merged) < 5:
        raise ValueError('only %d staff-line rows' % len(merged))
    gaps = [merged[i + 1] - merged[i] for i in range(len(merged) - 1)]
    med = statistics.median(gaps)
    groups, cur = [], [merged[0]]
    for i, g in enumerate(gaps):
        if g > GAP_CUT_SPACES * med:
            groups.append(cur)
            cur = [merged[i + 1]]
        else:
            cur.append(merged[i + 1])
    groups.append(cur)
    # Repair lyric/noise artifacts (never real staves, verified on truth pages):
    # - groups of <=2 peaks are always spurious (a staff has exactly 5 lines)
    # - an adjacent pair <=6px apart is a doubled detection (true staff lines
    #   sit ~1 staff space apart); keep the stronger copy
    # - an oversized group split at its largest internal gap when that gap is
    #   >=2x the group's median (a lyric/ottava line riding near a staff, or
    #   two tight staves merged by the 3-space cut); halves re-enter repair
    # - a clearly-weaker doubled copy or weak edge peak (lyric baseline) is
    #   shed; anything else keeps the page skipped.
    arr = np.asarray(gray)

    def row_density(r):
        """Max single-row ink fraction within +-SHED_SEARCH_PX of row r."""
        lo = max(0, r - SHED_SEARCH_PX)
        hi = min(arr.shape[0], r + SHED_SEARCH_PX + 1)
        return max(float((arr[rr, :] < BIN_THRESHOLD).mean())
                   for rr in range(lo, hi))

    def repair(g):
        out = []
        stack = [list(g)]
        while stack:
            cur = stack.pop()
            if len(cur) <= 2:
                continue  # spurious fragment, never a staff
            if len(cur) == 5:
                out.append(cur)
                continue
            if len(cur) < 5:
                raise ValueError('short staff group: %d' % len(cur))
            if max(row_density(r) for r in cur) < SHED_DISCARD_DENSITY:
                continue  # text block, never a staff: drop it
            gaps = [cur[i + 1] - cur[i] for i in range(len(cur) - 1)]
            j = min(range(len(gaps)), key=lambda i: gaps[i])
            if gaps[j] <= 6:
                cur.pop(j if probs[cur[j]] < probs[cur[j + 1]] else j + 1)
                stack.append(cur)
                continue
            med_in = statistics.median(gaps)
            k = max(range(len(gaps)), key=lambda i: gaps[i])
            if gaps[k] >= 2.0 * med_in:
                stack.append(cur[:k + 1])
                stack.append(cur[k + 1:])
                continue
            a, b = probs[cur[j]], probs[cur[j + 1]]
            if min(a, b) < 0.98 and max(a, b) >= 0.98:
                cur.pop(j if a < b else j + 1)
                stack.append(cur)
                continue
            if probs[cur[0]] < 0.95 and all(probs[r] >= 0.98 for r in cur[1:]):
                stack.append(cur[1:])
                continue
            if probs[cur[-1]] < 0.95 and all(probs[r] >= 0.98 for r in cur[:-1]):
                stack.append(cur[:-1])
                continue
            # Sparse annotation row peaking at staff spacing (volta/ottava
            # bracket, dense beam row over a folk staff): every prob rule
            # above misses because the row peaks ~1.0. An engraved staff
            # line is a continuous rule (~1.0 density); the annotation is
            # sparse. Shed an EDGE peak far below the group's best row;
            # edge-only and only beside a solid-looking line, so a middle
            # sparse row (or a faded page with no solid line) keeps the
            # page skipped instead of risking a shifted staff.
            shed = None
            for edge in (0, -1):
                rest = cur[1:] if edge == 0 else cur[:-1]
                best = max(row_density(r) for r in rest)
                if (best >= SHED_MIN_LINE_DENSITY
                        and row_density(cur[edge]) < SHED_EDGE_RATIO * best):
                    shed = edge
                    break
            if shed is not None:
                cur.pop(shed)
                stack.append(cur)
                continue
            raise ValueError('unrepairable staff group: %d peaks' % len(cur))
        return out
    fixed = []
    for g in groups:
        fixed.extend(repair(g))
    groups = fixed
    if not groups or any(len(g) != 5 for g in groups):
        raise ValueError('non-5-line staff groups: %s' % sorted(len(g) for g in groups))
    intra = [g[i + 1] - g[i] for g in groups for i in range(4)]
    space = statistics.median(intra)
    staves = []
    for g in groups:
        y1, y2 = int(g[0]), int(g[-1])
        band = arr[y1:y2 + 1, :]
        cols = np.nonzero((band < BIN_THRESHOLD).any(axis=0))[0]
        if len(cols) == 0:
            x1, x2 = 0.0, float(w)
        else:
            x1, x2 = float(cols[0]), float(cols[-1] + 1)
        staves.append({'x1': x1, 'y1': float(y1), 'x2': x2, 'y2': float(y2)})
    return staves, space


# ---------------------------------------------------------------- grouping

def group_regular(staves, chunk):
    return [staves[i:i + chunk] for i in range(0, len(staves), chunk)]


def group_ranked_gaps(staves, k):
    """Split into k systems at the k-1 largest inter-staff edge gaps."""
    if k == 1:
        return [list(staves)]
    gaps = [staves[i + 1]['y1'] - staves[i]['y2'] for i in range(len(staves) - 1)]
    order = sorted(range(len(gaps)), key=lambda i: gaps[i], reverse=True)
    cuts = sorted(order[:k - 1])
    groups, start = [], 0
    for c in cuts:
        groups.append(staves[start:c + 1])
        start = c + 1
    groups.append(staves[start:])
    return groups


def content_check(groups, spans, staves_per_part, active):
    """True iff every visual chunk's staff count matches the parts sounding
    in the paired measure span, with one engraver-truth allowance: a chunk
    showing the FULL staff complement is accepted even when some part is
    silent in that span. Rationale (verified on symbox truth): MuseScore's
    "hide empty staves" is a per-score style choice -- ds0001 hides a silent
    vocal staff on system 0 while ds0005/ds0004 print it (ds0005-1 system 0,
    ds0004-2 systems 0-1, ds0003-2 last system all show 3 staves with the
    voice resting whole measures). A printed-but-empty staff still pairs
    correctly with its (rest-filled) bekern span, so vetoing it only loses
    good data. Misalignment protection is preserved: any wrong break
    placement yields a chunk that is neither the expected count nor full,
    and the ranked-gap margin rule independently rejects pages whose
    chosen breaks are not separated from within-system gaps."""
    if len(groups) != len(spans):
        return False
    full = sum(staves_per_part)
    for chunk, span in zip(groups, spans):
        expect = 0
        for p, s in enumerate(staves_per_part):
            if part_active_in_span(active, p, span['start_measure'], span['end_measure'], None):
                expect += s
        if len(chunk) != expect and len(chunk) != full:
            return False
    return True


def group_page(staves, space, spans, staves_per_part, active):
    """Return system staff-groups or None (STRICT gate: skip, never force).

    Priority: regular S-staves-per-system chunks when the staff count fits
    exactly (n == k*S). Every system then shows a subset of the score's
    parts, so each chunk holds <= S staves; k chunks summing to k*S forces
    every chunk to exactly S -- regular chunking is then provably the only
    consistent grouping, and the content check is a sanity corroboration.
    Otherwise the k-1 largest inter-staff gaps define the breaks, accepted
    only with a separation margin over within-system gaps plus a passing
    content check (a wrong break pick inherently compresses that margin,
    e.g. ds0005-1's 9.8-vs-10.8 contest at ratio 0.91)."""
    k = len(spans)
    if k == 0 or len(staves) < k:
        return None
    total_s = sum(staves_per_part)
    if total_s > 0 and len(staves) == k * total_s:
        cand = group_regular(staves, total_s)
        return cand if content_check(cand, spans, staves_per_part, active) else None
    cand = group_ranked_gaps(staves, k)
    if k > 1 and len(staves) > k:
        gaps = sorted(staves[i + 1]['y1'] - staves[i]['y2'] for i in range(len(staves) - 1))
        smallest_break = gaps[-(k - 1)]
        non_breaks = gaps[:len(gaps) - (k - 1)]
        if smallest_break <= 0 or (non_breaks and smallest_break < MARGIN_RATIO * non_breaks[-1]):
            return None  # breaks not separated: ambiguous page, skip
    return cand if content_check(cand, spans, staves_per_part, active) else None


# ---------------------------------------------------------------- systems

# Glyph classes that span systems by design (sliced at crop time; the
# measure decoder re-anchors bars). They never drag the crop.
_NON_DRAG_CLASSES = ('BarLine', 'Bracket')


def crop_box_for_system(chunk, space, glyphs, ty1, ty2, cap_up, cap_dn, img_w, img_h):
    """Glyph-aware system crop box (strip-clip fix, audit-verified).

    Base = staff bounds +- PAD_H/V_SPACES, clamped to the image. The box
    then expands to cover attributed glyph extents (x-overlapping, center
    in [ty1, ty2)), excluding BarLine/Bracket -- but only along paths free
    of neighbor content (cap_up/cap_dn, None = open) and never contracting
    the base box. Returns (x1, y1, x2, y2) floats.
    """
    x1 = max(0.0, min(s['x1'] for s in chunk) - PAD_H_SPACES * space)
    y1 = max(0.0, min(s['y1'] for s in chunk) - PAD_V_SPACES * space)
    x2 = min(float(img_w), max(s['x2'] for s in chunk) + PAD_H_SPACES * space)
    y2 = min(float(img_h), max(s['y2'] for s in chunk) + PAD_V_SPACES * space)
    own = [g for g in (glyphs or [])
           if g['bbox'][2] >= x1 and g['bbox'][0] <= x2
           and ty1 <= (g['bbox'][1] + g['bbox'][3]) / 2 < ty2
           and g.get('class', g.get('cls', '?')) not in _NON_DRAG_CLASSES]
    if own:
        m1 = min(g['bbox'][1] for g in own)
        m2 = max(g['bbox'][3] for g in own)
        if m1 >= (cap_up if cap_up is not None else float('-inf')):
            y1 = max(0.0, min(y1, m1))
        if m2 <= (cap_dn if cap_dn is not None else float('inf')):
            y2 = min(float(img_h), max(y2, m2))
    return (x1, y1, x2, y2)


SPLIT_TARGET = 448  # chunk token budget (hard gate stays MAX_TOKENS)
CUT_MARGIN_PX = 8.0
BAR_INK = 140       # pixel darker than this counts as ink
BAR_INK_FRAC = 0.5  # min ink fraction in the best cut column


def plan_measure_chunks(measures, per_measure_tokens, target=SPLIT_TARGET,
                        hard_max=MAX_TOKENS):
    """Greedy measure ranges partitioning token load (bar-anchored split).

    Accumulates measures until adding the next would exceed ``target``
    (non-empty chunks); boundaries always fall on barlines. A single
    measure over ``hard_max`` is unsplittable and returns None.
    """
    chunks, cur, cur_n = [], [], 0
    for m, n in zip(measures, per_measure_tokens):
        if n > hard_max:
            return None
        if cur and cur_n + n > target:
            chunks.append(cur)
            cur, cur_n = [], 0
        cur.append(m)
        cur_n += n
    if cur:
        chunks.append(cur)
    return [(c[0], c[-1]) for c in chunks]


def _bar_candidates(glyphs, x1, x2, y_top, y_bot, tol=2.0):
    """BarLine glyphs spanning the full staff height, x-sorted."""
    out = []
    for g in glyphs or []:
        if g.get('class', g.get('cls', '?')) != 'BarLine':
            continue
        bx1, by1, bx2, by2 = g['bbox']
        cx = (bx1 + bx2) / 2
        if x1 <= cx <= x2 and by1 <= y_top + tol and by2 >= y_bot - tol:
            out.append(g)
    return sorted(out, key=lambda g: (g['bbox'][0] + g['bbox'][2]) / 2)


def full_height_bar_xs(glyphs, x1, x2, y_top, y_bot, tol=2.0):
    """Center-x of BarLine glyphs spanning the full staff height."""
    return [(g['bbox'][0] + g['bbox'][2]) / 2
            for g in _bar_candidates(glyphs, x1, x2, y_top, y_bot, tol)]


BAR_COL_X_TOL = 4.0
BAR_COL_COVER = 0.7


def bar_column_xs(glyphs, x1, x2, y_top, y_bot, tol=2.0,
                  x_tol=BAR_COL_X_TOL, cover=BAR_COL_COVER):
    """Center-x of barline COLUMNS: same-x BarLines spanning the system.

    Large-ensemble engraving breaks barlines per staff (or per group), so no
    single glyph spans a 5-10 staff system. A cluster of BarLine glyphs at
    (nearly) the same center-x qualifies as a column anchor when its merged
    y-union starts within ``tol`` of ``y_top``, ends within ``tol`` of
    ``y_bot``, and covers at least ``cover`` of the span.

    Returns [(cx, seg_y1, seg_y2)]: mean cluster center-x plus the tallest
    merged y-segment (the ink-check span for the cut).
    """
    bars = [g for g in (glyphs or [])
            if g.get('class', g.get('cls', '?')) == 'BarLine']
    bars = [g for g in bars
            if x1 <= (g['bbox'][0] + g['bbox'][2]) / 2 <= x2]
    bars.sort(key=lambda g: (g['bbox'][0] + g['bbox'][2]) / 2)
    clusters = []
    for g in bars:
        cx = (g['bbox'][0] + g['bbox'][2]) / 2
        if clusters and abs(cx - clusters[-1][0]) <= x_tol:
            clusters[-1][1].append(g)
        else:
            clusters.append([cx, [g]])
    out = []
    for _, members in clusters:
        segs = sorted((g['bbox'][1], g['bbox'][3]) for g in members)
        merged = []
        for a, b in segs:
            if merged and a <= merged[-1][1] + tol:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        if merged[0][0] > y_top + tol or merged[-1][1] < y_bot - tol:
            continue
        covered = sum(min(b, y_bot) - max(a, y_top) for a, b in merged)
        if covered < cover * (y_bot - y_top):
            continue
        cx = sum((g['bbox'][0] + g['bbox'][2]) / 2
                 for g in members) / len(members)
        tall = max(merged, key=lambda s: s[1] - s[0])
        out.append((cx, tall[0], tall[1]))
    return sorted(out)


def choose_cut_x(target_x, bar_xs, lo_x, hi_x, margin=CUT_MARGIN_PX):
    """Nearest candidate barline x to target inside the margins."""
    cands = [x for x in bar_xs if lo_x + margin <= x <= hi_x - margin]
    if not cands:
        return None
    return min(cands, key=lambda v: abs(v - target_x))


def cut_has_barline(px, x, gy1, gy2, dark=BAR_INK, need=BAR_INK_FRAC):
    """True if image column x shows ink over the glyph's y-range.

    ``px`` is a 2D row-major pixel array (e.g. numpy uint8 gray).
    Guards bar-anchored cuts against off-measure barlines: the cut must
    sit on printed staff-spanning ink, not empty paper.
    """
    import numpy as np
    a = np.asarray(px)
    h, w = a.shape[:2]
    xi = int(round(x))
    if xi < 1 or xi + 1 >= w:
        return False
    y1, y2 = max(0, int(round(gy1))), min(h, int(round(gy2)))
    if y2 - y1 < 4:
        return False
    col = a[y1:y2, max(0, xi - 1):min(w, xi + 2)]
    fracs = [(col[:, k] < dark).mean() for k in range(col.shape[1])]
    return bool(max(fracs) >= need)


def load_symbox_glyphs(box_path, img_w, img_h):
    """Return symbox glyphs as {'bbox': [x1,y1,x2,y2], 'class': ...} in PNG px."""
    with open(box_path) as f:
        d = json.load(f)
    sx, sy = 1.0, 1.0
    pw, ph = d.get('png_size', [img_w, img_h])
    if abs(pw - img_w) > 1 or abs(ph - img_h) > 1:
        sx, sy = img_w / pw, img_h / ph
    out = []
    for g in d.get('glyphs', []):
        x1, y1, x2, y2 = g['bbox']
        out.append({'bbox': [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                    'class': g.get('class', g.get('cls', '?'))})
    return out


_EMIT_FAIL_LOCK = threading.Lock()


def _log_emit_fail(out_dir, sys_stem, reason):
    """Append one loud row per culled strip (S2 zero-silent-drops gate).

    Best-effort: logging must never break the build it instruments.
    """
    try:
        with _EMIT_FAIL_LOCK:
            with open(os.path.join(out_dir, 'emit_failures.jsonl'), 'a',
                      encoding='utf-8') as f:
                f.write(json.dumps({'system': sys_stem,
                                    'reason': reason[:200]}) + '\n')
    except OSError:
        pass


def _bekern_and_decode(span_lo, span_hi, stem, txt_path, xml_path):
    """Run bekern --measures + token read + bekern-decode gate.

    Returns (n_tokens, None) on success or (None, reason) on any gate.
    """
    import subprocess
    import tempfile
    span_arg = '%d..%d' % (span_lo, span_hi)
    r = subprocess.run([ASER, 'bekern', '--input', xml_path, '--measures', span_arg,
                        '--out', os.path.dirname(txt_path), '--stem', stem],
                       capture_output=True, text=True, cwd=REPO_ROOT)
    if r.returncode != 0 or not os.path.exists(txt_path):
        return None, 'bekern-Q:' + (r.stderr or r.stdout or '')[-120:]
    with open(txt_path) as f:
        tokens = f.read().split()
    if not (MIN_TOKENS <= len(tokens) <= MAX_TOKENS):
        return None, 'tokens-%d' % len(tokens)
    with tempfile.TemporaryDirectory() as tmp:
        r2 = subprocess.run([ASER, 'bekern-decode', '--input', txt_path,
                             '--out', os.path.join(tmp, 'dec.json')],
                            capture_output=True, text=True, cwd=REPO_ROOT)
        if r2.returncode != 0:
            return None, 'decode:' + (r2.stderr or r2.stdout or '')[-120:]
    return len(tokens), None


def emit_split_chunks(span, chunk, space, glyphs, ty1, ty2, cap_up, cap_dn,
                      img_w, img_h, out_dir, png_path, x1, y1, x2, y2,
                      sys_stem, xml_path):
    """Split an over-long system into bar-anchored chunk strips.

    Greedy token-budget chunking by measure, image cuts at visually
    verified full-height barlines, per-chunk bekern+decode gates.
    Returns the list of chunk token counts (partial successes kept;
    each failure is logged loud) or None when nothing emitted.
    """
    from PIL import Image
    import numpy as np
    import tempfile
    a, b = span['start_measure'], span['end_measure']
    measures = list(range(a, b + 1))
    per_m = []
    with tempfile.TemporaryDirectory() as tmp:
        for m in measures:
            stem = '%s-m%d' % (sys_stem, m)
            txt = os.path.join(tmp, stem + '-bekern.txt')
            n, reason = _bekern_and_decode(m, m, stem, txt, xml_path)
            if n is None:
                _log_emit_fail(out_dir, sys_stem, 'split-measure-%d:%s' % (m, reason))
                return None
            per_m.append(n)
    chunks = plan_measure_chunks(measures, per_m)
    if chunks is None:
        _log_emit_fail(out_dir, sys_stem,
                       'unsplittable-single-measure-over-%d' % MAX_TOKENS)
        return None
    y_top = min(s['y1'] for s in chunk)
    y_bot = max(s['y2'] for s in chunk)
    cands = _bar_candidates(glyphs, x1, x2, y_top, y_bot)
    if not cands and len(chunks) > 1:
        # Fallback for large-ensemble engraving (per-staff/per-group
        # barlines): column anchors carry the same cut contract, with the
        # ink check running over the column's tallest segment.
        for cx, sy1, sy2 in bar_column_xs(glyphs, x1, x2, y_top, y_bot):
            cands.append({'bbox': [cx - 2.0, sy1, cx + 2.0, sy2],
                          'class': 'BarLine'})
    if not cands and len(chunks) > 1:
        _log_emit_fail(out_dir, sys_stem, 'split-no-barlines')
        return None
    total = sum(per_m)
    cum = 0
    idx_of = {m: i for i, m in enumerate(measures)}
    cuts = [x1]
    ok = True
    for lo, hi in chunks[:-1]:
        cum += sum(per_m[idx_of[m]] for m in range(lo, hi + 1))
        target_x = x1 + (cum / total) * (x2 - x1)
        ordered = sorted(cands, key=lambda g: abs(
            (g['bbox'][0] + g['bbox'][2]) / 2 - target_x))
        cut = None
        with Image.open(png_path) as img:
            gray = np.asarray(img.convert('L'))
            for g in ordered:
                cx = (g['bbox'][0] + g['bbox'][2]) / 2
                if not (cuts[-1] + CUT_MARGIN_PX <= cx <= x2 - CUT_MARGIN_PX):
                    continue
                if cut_has_barline(gray, cx, g['bbox'][1], g['bbox'][3]):
                    cut = cx
                    break
        if cut is None:
            _log_emit_fail(out_dir, sys_stem,
                           'split-no-verified-cut-after-%d' % hi)
            ok = False
            break
        cuts.append(cut)
    cuts.append(x2)
    if not ok or any(cuts[i + 1] - cuts[i] < 32.0 for i in range(len(chunks))):
        if ok:
            _log_emit_fail(out_dir, sys_stem, 'split-chunk-too-narrow')
        return None
    counts = []
    with Image.open(png_path) as img:
        for k, (lo, hi) in enumerate(chunks):
            cstem = '%sc%d' % (sys_stem, k)
            cpath = os.path.join(out_dir, cstem + '.png')
            tpath = os.path.join(out_dir, cstem + '-bekern.txt')
            try:
                img.crop((int(cuts[k]), int(y1), int(cuts[k + 1]), int(y2))).save(cpath)
            except (OSError, ValueError) as e:
                for p in (cpath, tpath):
                    if os.path.exists(p):
                        os.remove(p)
                _log_emit_fail(out_dir, cstem, 'split-crop:%s:%s' % (type(e).__name__, str(e)[:80]))
                continue
            n, reason = _bekern_and_decode(lo, hi, cstem, tpath, xml_path)
            if n is None:
                for p in (cpath, tpath):
                    if os.path.exists(p):
                        os.remove(p)
                _log_emit_fail(out_dir, cstem, 'split-' + reason)
                continue
            counts.append(n)
    return counts or None

def emit_system(score_stem, page, sys_idx, span, chunk, space, xml_path, img_w, img_h, out_dir, png_path,
                 glyphs=None, ty1=None, ty2=None, cap_up=None, cap_dn=None):
    """Crop the system strip and emit its bekern target.

    Returns the list of emitted strip token counts (bar-anchored chunks
    when the system exceeds MAX_TOKENS) or None when no strip survived.
    Every failure is logged loud via emit_failures.jsonl.
    """
    from PIL import Image
    sys_stem = '%s-%d-s%d' % (score_stem, page, sys_idx)
    if ty1 is None or ty2 is None:
        x1 = max(0.0, min(s['x1'] for s in chunk) - PAD_H_SPACES * space)
        y1 = max(0.0, min(s['y1'] for s in chunk) - PAD_V_SPACES * space)
        x2 = min(float(img_w), max(s['x2'] for s in chunk) + PAD_H_SPACES * space)
        y2 = min(float(img_h), max(s['y2'] for s in chunk) + PAD_V_SPACES * space)
    else:
        x1, y1, x2, y2 = crop_box_for_system(chunk, space, glyphs, ty1, ty2,
                                             cap_up, cap_dn, img_w, img_h)
    crop_path = os.path.join(out_dir, sys_stem + '.png')
    txt_path = os.path.join(out_dir, sys_stem + '-bekern.txt')

    def fail(reason):
        for p in (crop_path, txt_path):
            if os.path.exists(p):
                os.remove(p)
        _log_emit_fail(out_dir, sys_stem, reason)
        return None

    try:
        with Image.open(png_path) as img:
            img.crop((int(x1), int(y1), int(x2), int(y2))).save(crop_path)
        lo, hi = span['start_measure'], span['end_measure']
        n, reason = _bekern_and_decode(lo, hi, sys_stem, txt_path, xml_path)
        if n is not None:
            return [n]
        over = False
        if reason.startswith('tokens-'):
            try:
                over = int(reason.split('-')[1]) > MAX_TOKENS
            except ValueError:
                over = False
        if over:
            for p in (crop_path, txt_path):
                if os.path.exists(p):
                    os.remove(p)
            got = emit_split_chunks(span, chunk, space, glyphs, ty1, ty2,
                                    cap_up, cap_dn, img_w, img_h, out_dir,
                                    png_path, x1, y1, x2, y2, sys_stem,
                                    xml_path)
            if got:
                return got
            return fail('split-failed')
        return fail(reason)
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        return fail('%s:%s' % (type(e).__name__, str(e)[:120]))


def degrade_clean_crops(out_dir):
    for suffix, seed in DEGRADE_VARIANTS:
        r = subprocess.run([PDEG, '--input', out_dir, '--out-dir', out_dir,
                            '--suffix', suffix, '--seed', str(seed),
                            '--format', 'jpeg', '--max-width', str(DEGRADE_MAX_WIDTH)],
                           capture_output=True, text=True, cwd=REPO_ROOT)
        if r.returncode != 0:
            raise RuntimeError('photo-degrade %s failed: %s' % (suffix, (r.stderr or r.stdout)[-2000:]))


# ---------------------------------------------------------------- driver

def percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = q / 100.0 * (len(sorted_vals) - 1)
    lo, hi = int(rank), min(int(rank) + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def page_pngs(renders_dir, score_stem):
    pages = {}
    for p in glob.glob(os.path.join(renders_dir, score_stem + '-*.png')):
        tail = os.path.basename(p)[len(score_stem) + 1:-4]
        if tail.isdigit():
            pages[int(tail)] = p
    return pages


def process_page(score_stem, page, png_path, spans, staves_per_part, active, args, out_dir):
    """Return (jobs, skipped) where jobs are emit_system arg tuples."""
    from PIL import Image
    box_path = os.path.join(args.symbox, score_stem, '%s-%d-boxes.json' % (score_stem, page))
    page_glyphs = []
    try:
        with Image.open(png_path) as img:
            img_w, img_h = img.size
        if os.path.exists(box_path):
            staves, space = load_symbox_staves(box_path, img_w, img_h)
            try:
                page_glyphs = load_symbox_glyphs(box_path, img_w, img_h)
            except (OSError, ValueError, KeyError):
                page_glyphs = []
        else:
            staves, space = neural_detect_staves(png_path)
    except (OSError, ValueError, KeyError) as e:
        print('  %s-%d: detection failed (%s)' % (score_stem, page, e), flush=True)
        return [], True
    groups = group_page(staves, space, spans, staves_per_part, active)
    if groups is None:
        return [], True
    xml_path = os.path.join(args.renders, score_stem + '.musicxml')
    n = len(groups)
    bounds = []
    for i, chunk in enumerate(groups):
        ty1 = float('-inf') if i == 0 else (groups[i - 1][-1]['y2'] + chunk[0]['y1']) / 2
        ty2 = float('inf') if i == n - 1 else (chunk[-1]['y2'] + groups[i + 1][0]['y1']) / 2
        if page_glyphs:
            x1b = min(s['x1'] for s in chunk) - PAD_H_SPACES * space
            x2b = max(s['x2'] for s in chunk) + PAD_H_SPACES * space
            prev = [g for g in page_glyphs
                    if g['bbox'][2] >= x1b and g['bbox'][0] <= x2b
                    and (g['bbox'][1] + g['bbox'][3]) / 2 < ty1] if i > 0 else []
            nxt = [g for g in page_glyphs
                   if g['bbox'][2] >= x1b and g['bbox'][0] <= x2b
                   and (g['bbox'][1] + g['bbox'][3]) / 2 >= ty2] if i < n - 1 else []
            cap_up = max([groups[i - 1][-1]['y2']] + [g['bbox'][3] for g in prev]) if i > 0 else None
            cap_dn = min([groups[i + 1][0]['y1']] + [g['bbox'][1] for g in nxt]) if i < n - 1 else None
        else:
            cap_up = cap_dn = None
        bounds.append((ty1, ty2, cap_up, cap_dn))
    jobs = [(score_stem, page, i, span, chunk, space, xml_path, img_w, img_h, out_dir, png_path,
             page_glyphs, ty1, ty2, cap_up, cap_dn)
            for i, ((span, chunk), (ty1, ty2, cap_up, cap_dn))
            in enumerate(zip(zip(spans, groups), bounds))]
    return jobs, False


def main():
    ap = argparse.ArgumentParser(description='T1.2 system-crop + degraded dataset builder')
    ap.add_argument('--renders', default='data-gen/score-corpus/big500-renders')
    ap.add_argument('--symbox', default='data-gen/score-corpus/big500-symbox')
    ap.add_argument('--degraded', default='data-gen/score-corpus/big500-degraded10')
    ap.add_argument('--out', default='tmp-train/mvd-sysdata')
    ap.add_argument('--limit-scores', type=int, default=0)
    args = ap.parse_args()
    for name in ('renders', 'symbox', 'degraded'):
        setattr(args, name, os.path.join(REPO_ROOT, getattr(args, name)))
    out_dir = os.path.join(REPO_ROOT, args.out)
    for p, label in ((ASER, 'assembly-serialize'), (PDEG, 'photo-degrade'), (WEIGHTS, 'rownet weights')):
        if not os.path.exists(p):
            print('error: missing %s: %s' % (label, p))
            return 1
    for d, label in ((args.renders, '--renders'), (args.symbox, '--symbox'), (args.degraded, '--degraded')):
        if not os.path.isdir(d):
            print('error: %s dir missing: %s' % (label, d))
            return 1
    excluded = load_excluded()
    print('exclusion guard: %d page/score stems (%d do_not_train + quarantine)' %
          (len(excluded), len(json.load(open(EXCLUSION_JSON)).get('do_not_train', []))), flush=True)

    xmls = sorted(glob.glob(os.path.join(args.renders, '*.musicxml')))
    if args.limit_scores > 0:
        xmls = xmls[:args.limit_scores]
    print('scores: %d (limit=%d)' % (len(xmls), args.limit_scores), flush=True)
    os.makedirs(out_dir, exist_ok=True)

    skipped_pages = set()
    excluded_pages = 0
    all_jobs = []
    for xml_path in xmls:
        score_stem = os.path.basename(xml_path)[:-len('.musicxml')]
        try:
            spans = run_system_breaks(xml_path)
        except RuntimeError as e:
            print('%s: %s' % (score_stem, e), flush=True)
            continue
        by_page = {}
        for s in spans:
            by_page.setdefault(s['page'], []).append(s)
        for pg in by_page.values():
            pg.sort(key=lambda s: s['system_idx'])
        try:
            staves_per_part, active = parse_parts(xml_path)
        except ET.ParseError as e:
            print('%s: musicxml parse failed (%s)' % (score_stem, e), flush=True)
            continue
        for page, png_path in sorted(page_pngs(args.renders, score_stem).items()):
            page_stem = '%s-%d' % (score_stem, page)
            if page_excluded(page_stem, excluded):
                excluded_pages += 1
                continue
            if page not in by_page:
                skipped_pages.add(page_stem)
                continue
            jobs, skipped = process_page(score_stem, page, png_path, by_page[page],
                                         staves_per_part, active, args, out_dir)
            if skipped:
                skipped_pages.add(page_stem)
            all_jobs.extend(jobs)
        print('%s: %d systems queued' %
              (score_stem, sum(1 for j in all_jobs if j[0] == score_stem)), flush=True)

    token_counts = []
    ok_systems = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(emit_system, *j): j for j in all_jobs}
        for fut in concurrent.futures.as_completed(futs):
            ns = fut.result() or []
            ok_systems += len(ns)
            token_counts.extend(ns)
    print('clean systems emitted: %d/%d (bekern+decode+token gates)' % (ok_systems, len(all_jobs)), flush=True)

    print('degrading %d clean crops x3 (release photo-degrade, jpeg, max-width %d)...' %
          (ok_systems, DEGRADE_MAX_WIDTH), flush=True)
    degrade_clean_crops(out_dir)
    degraded = sorted(glob.glob(os.path.join(out_dir, '*_d[012].jpg')))
    print('degraded images: %d' % len(degraded), flush=True)

    clean_stems = sorted(glob.glob(os.path.join(out_dir, '*.png')))
    n_clean = sum(1 for p in clean_stems
                  if os.path.exists(p[:-len('.png')] + '-bekern.txt'))
    leaked = sum(1 for p in clean_stems
                 if page_excluded(os.path.basename(p)[:-len('.png')].rsplit('-s', 1)[0], excluded))
    token_counts.sort()
    manifest = {
        'clean_systems': n_clean,
        'degraded_systems': len(degraded),
        'token_p50': int(percentile(token_counts, 50)),
        'token_p95': int(percentile(token_counts, 95)),
        'max_tokens': int(token_counts[-1]) if token_counts else 0,
        'min_tokens': int(token_counts[0]) if token_counts else 0,
        'gate20_leaked': leaked,
        'n_scores_attempted': len(xmls),
        'n_skipped_pages': len(skipped_pages),
        'n_excluded_pages': excluded_pages,
        'degrade_variants': [s for s, _ in DEGRADE_VARIANTS],
        'provenance': {'renders': args.renders, 'symbox': args.symbox, 'degraded': args.degraded},
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_dir, 'skipped_pages.txt'), 'w') as f:
        for stem in sorted(skipped_pages):
            f.write(stem + '\n')
    print(json.dumps(manifest, indent=2), flush=True)
    if leaked:
        print('FATAL: gate20 leak detected', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
