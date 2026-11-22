#!/usr/bin/env python3
"""Phase-1 reprojection probe: can we predict where a clean landmark lands
in degraded output, using only the params JSON? (part 1: grid + maps)"""
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
BIN = os.path.join(ROOT, 'data-gen', 'photo-degrade', 'target', 'release',
                   'photo-degrade.exe')
GRID_W, GRID_H = 1200, 1600
DOT_R = 14
COLS, ROWS = 6, 8


def gen_grid(path):
    img = np.full((GRID_H, GRID_W, 3), 200, np.uint8)
    ys = np.linspace(0.1, 0.9, ROWS) * GRID_H
    xs = np.linspace(0.1, 0.9, COLS) * GRID_W
    yy, xx = np.mgrid[0:GRID_H, 0:GRID_W]
    dots = []
    for y in ys:
        for x in xs:
            img[(xx - x) ** 2 + (yy - y) ** 2 <= DOT_R ** 2] = 0
            dots.append((float(x), float(y)))
    Image.fromarray(img).save(path)
    return dots


def solve_h(src, dst):
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(a, float))
    return vt[-1].reshape(3, 3)


def apply_h(h, x, y):
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return ((h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w,
            (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w)


def bend_sample(mode, px, shade, scx, scw, fx, fy, ph, w, h, x, y):
    if mode == 'spine':
        depth = min((px / w) * 2.0, 0.15)
        t = (x - scx) / max(scw, 1.0)
        g = math.exp(-t * t)
        return scx + (x - scx) * (1.0 - depth * g), y
    if mode == 'ripple':
        ox = px * 0.5 * math.sin(math.tau * fx * y / h + ph)
        oy = px * 0.3 * math.cos(math.tau * fy * x / w + ph)
        return x + ox, y + oy
    wave = (0.6 + 0.4 * math.sin(math.pi * y / h) if mode in
            ('left', 'right') else 0.6 + 0.4 * math.sin(math.pi * x / w))
    if mode == 'right':
        edge = x / w
    elif mode == 'top':
        edge = 1.0 - y / h
    elif mode == 'bottom':
        edge = y / h
    else:
        edge = 1.0 - x / w
    shift = px * edge * edge * wave
    if mode == 'right':
        return x - shift, y
    if mode == 'top':
        return x, y + shift
    if mode == 'bottom':
        return x, y - shift
    return x + shift, y


def bend_forward(mode, px, shade, scx, scw, fx, fy, ph, w, h, ix, iy):
    ox, oy = ix, iy
    for _ in range(200):
        sx, sy = bend_sample(mode, px, shade, scx, scw, fx, fy, ph,
                             w, h, ox, oy)
        nx, ny = ox + (ix - sx), oy + (iy - sy)
        if abs(nx - ox) + abs(ny - oy) < 1e-9:
            break
        ox, oy = nx, ny
    return ox, oy


def forward(p, dots, iw, ih):
    out = []
    for (ix, iy) in dots:
        w, h = float(iw), float(ih)
        x, y = ix, iy
        if p['curl_px'] > 0.25:
            x, y = bend_forward(p['curl_side'], p['curl_px'],
                                p['curl_shade'], p['bend_spine_cx'],
                                p['bend_spine_w'], p['bend_ripple_fx'],
                                p['bend_ripple_fy'], p['bend_ripple_ph'],
                                w, h, x, y)
        src = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        dst = [(sx + dx, sy + dy) for (sx, sy), (dx, dy) in
               zip(src, p['corner_offsets'])]
        if '--no-persp' not in sys.argv:
            x, y = apply_h(solve_h(src, dst), x, y)
        if p['frame_mode'] == 'inset':
            l, t, r, b = p['frame_pad_px']
            x, y = x + l, y + t
            w, h = w + l + r, h + t + b
        else:
            l, t, r, b = p['frame_cut_px']
            x, y = x - l, y - t
            w, h = w - l - r, h - t - b
        do_rot = abs(p['rotation_deg']) > 0.05 and w > 4 and h > 4
        if do_rot:
            th = math.radians(p['rotation_deg'])
            snt, cst = math.sin(-th), math.cos(th)
            c0x, c0y = w / 2.0, h / 2.0
            w2 = math.ceil(w * abs(cst) + h * abs(snt))
            h2 = math.ceil(h * abs(cst) + w * abs(snt))
            c1x, c1y = w2 / 2.0, h2 / 2.0
            dx, dy = x - c0x, y - c0y
            x, y = c1x + dx * cst + dy * snt, c1y - dx * snt + dy * cst
            w, h = float(w2), float(h2)
        if abs(p['dist_k1']) > 0.005 or abs(p['dist_k2']) > 0.002:
            n0 = max(min(w, h) / 2.0, 1.0)
            dcx, dcy = w / 2.0, h / 2.0
            rx, ry = (x - dcx) / n0, (y - dcy) / n0
            r2 = rx * rx + ry * ry
            f = 1.0 + p['dist_k1'] * r2 + p['dist_k2'] * r2 * r2
            x, y = dcx + (x - dcx) * f, dcy + (y - dcy) * f
        x *= int(w * p['scale_x']) / w
        y *= int(h * p['scale_y']) / h
        out.append((x, y))
    return out


def detect(path):
    img = np.asarray(Image.open(path).convert('L'), float)
    bg = ndimage.gaussian_filter(img, DOT_R * 2.0)
    fg = bg - img
    lab, n = ndimage.label(fg > 25.0)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    pts = []
    for i, s in enumerate(sizes, 1):
        if 60.0 <= s <= 1500.0:
            cy, cx = ndimage.center_of_mass(fg > 25.0, lab, i)
            pts.append((float(cx), float(cy)))
    return pts


def affine_fit(pred, det, tol=30.0):
    rem = list(det)
    pairs = []
    for q in pred:
        best, bi, bd = tol, -1, None
        for i, d in enumerate(rem):
            e = math.hypot(q[0] - d[0], q[1] - d[1])
            if e < best:
                best, bi, bd = e, i, d
        if bi >= 0:
            pairs.append((q, bd))
            rem.pop(bi)
    if len(pairs) < 6:
        return None, None
    a = []
    b = []
    for (x, y), (u, v) in pairs:
        a.append([x, y, 1, 0, 0, 0])
        a.append([0, 0, 0, x, y, 1])
        b += [u, v]
    coef, *_ = np.linalg.lstsq(np.asarray(a), np.asarray(b), rcond=None)
    res = []
    for (x, y), (u, v) in pairs:
        pu = coef[0] * x + coef[1] * y + coef[2]
        pv = coef[3] * x + coef[4] * y + coef[5]
        res.append(math.hypot(pu - u, pv - v))
    return coef, np.array(res)


def score(pred, det, tol=30.0, dump=False):
    rem = list(det)
    errs = []
    for qi, q in enumerate(pred):
        best, bi, bd = tol, -1, None
        for i, d in enumerate(rem):
            e = math.hypot(q[0] - d[0], q[1] - d[1])
            if e < best:
                best, bi, bd = e, i, d
        if bi >= 0:
            errs.append(best)
            if dump and len(errs) <= 12:
                print(f'  dot{qi}: pred=({q[0]:.1f},{q[1]:.1f}) '
                      f'det=({bd[0]:.1f},{bd[1]:.1f}) '
                      f'd=({bd[0]-q[0]:+.1f},{bd[1]-q[1]:+.1f})')
            rem.pop(bi)
    return errs


def main():
    seeds = [int(s) for s in sys.argv[sys.argv.index('--seeds') + 1].split(',')]
    strength = float(sys.argv[sys.argv.index('--strength') + 1]) \
        if '--strength' in sys.argv else 0.6
    scalemin = float(sys.argv[sys.argv.index('--scale-min') + 1]) \
        if '--scale-min' in sys.argv else 0.5
    grid = os.path.join(HERE, 'grid.png')
    dots = gen_grid(grid)
    outd = os.path.join(HERE, 'probe-out')
    os.makedirs(outd, exist_ok=True)
    all_e, all_m = [], []
    for s in seeds:
        subprocess.run([BIN, '--input', grid, '--out-dir', outd,
                        '--seed', str(s), '--strength', str(strength),
                        '--scale-min', str(scalemin),
                        '--suffix', f'_p{s}', '--format', 'png'],
                       check=True, capture_output=True)
        jp = glob.glob(os.path.join(outd, f'grid_p{s}_params.json'))[0]
        p = json.load(open(jp))
        outs = sorted(glob.glob(os.path.join(outd, f'grid_p{s}.png')))
        pred = forward(p, dots, GRID_W, GRID_H)
        det = detect(outs[0])
        errs = score(pred, det, dump=('--dump' in sys.argv))
        if '--viz' in sys.argv:
            import PIL.Image as I
            viz = Image.open(outs[0]).convert('RGB')
            import PIL.ImageDraw as D
            dr = D.Draw(viz)
            for (x, y) in pred:
                dr.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 0, 0), width=2)
            for (x, y) in det:
                dr.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(0, 255, 0), width=2)
            viz.save(os.path.join(outd, f'viz_p{s}.png'))
        coef, res = affine_fit(pred, det)
        if coef is not None:
            print(f'  affine: [{coef[0]:.4f} {coef[1]:.4f} {coef[2]:+.1f}; '
                  f'{coef[3]:.4f} {coef[4]:.4f} {coef[5]:+.1f}] '
                  f'resid med={np.median(res):.2f}px max={res.max():.2f}px')
        e = np.array(errs)
        md = min(p['width'], p['height'])
        print(f'seed {s}: mode={p["curl_side"]} match={len(errs)}/{len(dots)} '
              f'mean={e.mean():.2f}px med={np.median(e):.2f}px '
              f'max={e.max():.2f}px ({100*e.max()/md:.2f}% mindim)')
        all_e += errs
        all_m.append(len(errs) / len(dots))
    e = np.array(all_e)
    md = 1000.0
    print(f'ALL: n={len(all_e)} matchrate={np.mean(all_m):.3f} '
          f'mean={e.mean():.2f}px med={np.median(e):.2f}px '
          f'max={e.max():.2f}px ({100*e.max()/md:.2f}% of ~1000px)')


if __name__ == '__main__':
    main()
