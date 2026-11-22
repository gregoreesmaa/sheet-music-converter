"""Synthetic verification for the canonical 4-slot rectifier (T2.1).

Builds 2-staff and 4-staff grayscale system strips with an exact-truth
warp (3-degree tilt + +/-8 px sinusoidal bend, staff space 20 px) plus
notehead dots on staff 0 (E4 on bottom line ``k = 4``, F4 in the space
above it at ``k = 3.5``), runs :func:`rectify.rectify_system_strip`, and
asserts:

* every active staff line sits within +/-0.5 px of ``y_canon(s, k)``
  across the full width (8 segments x all lines x all active staves),
* inactive slots are pure paper white (255),
* the E4/F4 dot centroids sit within +/-0.5 px of ``y_canon(0, 4)``
  and ``y_canon(0, 3.5)`` (dots are small on purpose: the vertical map is
  uniform inside a staff but crushes the inter-staff gap, so the scale
  kink at an outer staff line would drag a fat dot's centroid off the
  line; staff interiors, spaces, and the lines themselves are unaffected),
* ``W_out == round(W_in * target_sp / median_measured_sp)`` and the
  ``[1, H, W_out]`` reference channel is +1.0 exactly on active lines.

Run: ``.\\.venv\\Scripts\\python.exe -m unittest tmp-train/mvd/test_rectify.py -v``
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rectify as R

W_IN, H_IN = 1200, 900
SP = 20.0
TILT = float(np.tan(np.deg2rad(3.0)))  # 3-degree tilt slope
BEND_A = 8.0  # +/-8 px sinusoidal page-curl bend
BEND_LAM = 900.0
BEND_PHI = 0.7
DOT_X_E4 = 840  # input column of the E4 notehead dot (staff 0, bottom line)
DOT_X_F4 = 360  # input column of the F4 notehead dot (staff 0, space above E4)
DOT_R = 3.0  # small marker dots: a fat dot straddling the staff/gap scale
# kink at an outer line would pull its own centroid off the line (see below)
LINE_THICK = 3.0
N_SEG = 8
TOL = 0.5


def warp_offset(x):
    """Exact-truth vertical warp (tilt + bend) shared by all lines."""
    xc = (W_IN - 1) / 2.0
    return TILT * (x - xc) + BEND_A * np.sin(2.0 * np.pi * (x - xc) / BEND_LAM + BEND_PHI)


def staff_curves(base_top):
    """(5, W_IN) exact line positions for a staff starting at ``base_top``."""
    x = np.arange(W_IN, dtype=np.float64)
    return base_top + SP * np.arange(5, dtype=np.float64)[:, None] + warp_offset(x)[None, :]


def draw_lines(canvas, curves, thick=LINE_THICK):
    """Paint subpixel-centered dark staff lines (grayscale coverage)."""
    rows = np.arange(canvas.shape[0], dtype=np.float64)[:, None]
    for k in range(curves.shape[0]):
        y = curves[k][None, :]
        cover = np.minimum(y + thick / 2.0, rows + 0.5) - np.maximum(
            y - thick / 2.0, rows - 0.5
        )
        canvas[:] = np.minimum(canvas, 255.0 * (1.0 - np.clip(cover, 0.0, 1.0)))


def draw_dot(canvas, cx, cy, r=DOT_R):
    """Paint a filled notehead dot centered at float ``(cx, cy)``."""
    x0, x1 = max(0, int(cx - r - 1)), min(canvas.shape[1], int(cx + r + 2))
    y0, y1 = max(0, int(cy - r - 1)), min(canvas.shape[0], int(cy + r + 2))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    canvas[y0:y1, x0:x1] = np.where((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r, 0.0,
                                    canvas[y0:y1, x0:x1])


def build_case(bases):
    """Synthetic strip: warped staves + E4/F4 dots on staff 0.

    Returns ``(PIL image, staves_lines curves, dot rows dict)``.
    """
    canvas = np.full((H_IN, W_IN), 255.0, dtype=np.float64)
    curves = [staff_curves(b) for b in bases]
    for c in curves:
        draw_lines(canvas, c)
    # E4 and F4 sit at different columns (as engraved seconds do) so neither
    # dot's patch contaminates the other's centroid measurement.
    e4_row = float(curves[0][4, DOT_X_E4])
    f4_row = float((curves[0][3, DOT_X_F4] + curves[0][4, DOT_X_F4]) / 2.0)
    draw_dot(canvas, float(DOT_X_E4), e4_row)
    draw_dot(canvas, float(DOT_X_F4), f4_row)
    img = Image.fromarray(canvas.astype(np.uint8), mode="L")
    return img, curves, {"E4": e4_row, "F4": f4_row}


def out_x(dot_x, w_out):
    """Output column of an input column under the pixel-center width map."""
    return (dot_x + 0.5) * (w_out / W_IN) - 0.5


def keep_columns(w_out, dot_xs, margin=10.0):
    """Output columns excluding the dot neighborhoods (for line profiles)."""
    xcs = [out_x(d, w_out) for d in dot_xs]
    return np.array([c for c in range(w_out)
                     if all(abs(c - xc) > margin for xc in xcs)])


def line_residuals(arr, expected_rows, keep_cols):
    """Darkness-centroid residual per (row, width-segment); segments span width."""
    dark = 255.0 - arr.astype(np.float64)
    out = {}
    for e in expected_rows:
        lo, hi = int(round(e)) - 3, int(round(e)) + 3
        rows = np.arange(lo, hi + 1, dtype=np.float64)
        for i, cols in enumerate(np.array_split(keep_cols, N_SEG)):
            prof = dark[lo:hi + 1][:, cols].mean(axis=1)
            tot = float(prof.sum())
            if tot <= 0:
                out[(e, i)] = float("nan")
            else:
                out[(e, i)] = float((rows * prof).sum() / tot) - e
    return out


def dot_dy(arr, xc_out, e_row, half_c=8, half_r=6):
    """Vertical centroid residual of a notehead dot in a window around it."""
    dark = (255.0 - arr.astype(np.float64))
    c0, c1 = int(round(xc_out)) - half_c, int(round(xc_out)) + half_c + 1
    r0, r1 = int(round(e_row)) - half_r, int(round(e_row)) + half_r + 1
    patch = dark[r0:r1, c0:c1]
    rows = np.arange(r0, r1, dtype=np.float64)[:, None]
    return float((rows * patch).sum() / patch.sum()) - e_row


class TwoStaffRectify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bases = [200.0, 500.0]
        img, curves, _ = build_case(cls.bases)
        cls.out_img, cls.ref = R.rectify_system_strip(img, curves)
        cls.arr = np.asarray(cls.out_img)
        cls.w_out = cls.arr.shape[1]

    def test_canvas_geometry_and_width(self):
        self.assertEqual(self.out_img.mode, "L")
        self.assertEqual((self.out_img.size[1], self.w_out), (256, 600))
        self.assertEqual(self.w_out, round(W_IN * 10.0 / SP))
        self.assertEqual(self.ref.shape, (1, 256, self.w_out))
        self.assertEqual(self.ref.dtype, torch.float32)
        print(f"2-staff: size={(self.w_out, 256)} W_out==round(1200*10/20)=={self.w_out}")

    def test_line_residuals(self):
        keep = keep_columns(self.w_out, (DOT_X_E4, DOT_X_F4))
        expected = [12.0 + 64.0 * s + 10.0 * k for s in range(2) for k in range(5)]
        res = line_residuals(self.arr, expected, keep)
        worst = max(abs(v) for v in res.values())
        per_line = [max(abs(res[(e, i)]) for i in range(N_SEG)) for e in expected]
        print(f"2-staff line residuals: worst={worst:.4f} "
              f"per-line-max={[round(v, 4) for v in per_line]}")
        self.assertLessEqual(worst, TOL, f"max row residual {worst:.4f} > 0.5 px")

    def test_inactive_slots_white(self):
        for s in (2, 3):
            block = self.arr[s * 64:(s + 1) * 64]
            bad = int((block != 255).sum())
            print(f"2-staff slot {s}: nonwhite px = {bad} / {block.size}")
            self.assertEqual(bad, 0, f"slot {s} not pure white")

    def test_notehead_positions(self):
        d_e4 = dot_dy(self.arr, out_x(DOT_X_E4, self.w_out), R.canon_y(0, 4))
        # Narrow row window: the F4 dot (r=1.5 out) sits between flanking
        # staff lines; rows 44..50 isolate the dot's own centroid.
        d_f4 = dot_dy(self.arr, out_x(DOT_X_F4, self.w_out), R.canon_y(0, 3.5),
                      half_r=3)
        print(f"2-staff noteheads: E4 dy={d_e4:.4f} (expect y=52), F4 dy={d_f4:.4f} (expect y=47)")
        self.assertLessEqual(abs(d_e4), TOL)
        self.assertLessEqual(abs(d_f4), TOL)

    def test_reference_channel(self):
        on = [12.0 + 64.0 * s + 10.0 * k for s in range(2) for k in range(5)]
        on_rows = sorted(int(round(e)) for e in on)
        for r in on_rows:
            self.assertTrue(bool((self.ref[0, r, :] == 1.0).all()), f"ref row {r} not +1")
        mask = torch.ones(256, dtype=torch.bool)
        mask[on_rows] = False
        self.assertTrue(bool((self.ref[0, mask, :] == 0.0).all()), "ref nonzero off-line")
        print(f"2-staff ref: +1 rows={on_rows}, zeros elsewhere incl. slots 2-3")


class FourStaffRectify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bases = [80.0, 290.0, 500.0, 710.0]
        img, curves, _ = build_case(cls.bases)
        # Pass plain python lists (not ndarrays) to cover the list input path.
        cls.out_img, cls.ref = R.rectify_system_strip(img, [c.tolist() for c in curves])
        cls.arr = np.asarray(cls.out_img)
        cls.w_out = cls.arr.shape[1]

    def test_canvas_geometry_and_width(self):
        self.assertEqual((self.out_img.size[1], self.w_out), (256, 600))
        self.assertEqual(self.w_out, round(W_IN * 10.0 / SP))
        print(f"4-staff: size={(self.w_out, 256)}")

    def test_line_residuals(self):
        keep = keep_columns(self.w_out, (DOT_X_E4, DOT_X_F4))
        expected = [12.0 + 64.0 * s + 10.0 * k for s in range(4) for k in range(5)]
        res = line_residuals(self.arr, expected, keep)
        worst = max(abs(v) for v in res.values())
        per_staff = [
            max(abs(res[(12.0 + 64.0 * s + 10.0 * k, i)])
                for k in range(5) for i in range(N_SEG))
            for s in range(4)
        ]
        print(f"4-staff line residuals: worst={worst:.4f} "
              f"per-staff-max={[round(v, 4) for v in per_staff]}")
        self.assertLessEqual(worst, TOL, f"max row residual {worst:.4f} > 0.5 px")

    def test_notehead_positions(self):
        d_e4 = dot_dy(self.arr, out_x(DOT_X_E4, self.w_out), R.canon_y(0, 4))
        # Narrow row window: the F4 dot (r=1.5 out) sits between flanking
        # staff lines; rows 44..50 isolate the dot's own centroid.
        d_f4 = dot_dy(self.arr, out_x(DOT_X_F4, self.w_out), R.canon_y(0, 3.5),
                      half_r=3)
        print(f"4-staff noteheads: E4 dy={d_e4:.4f} (expect y=52), F4 dy={d_f4:.4f} (expect y=47)")
        self.assertLessEqual(abs(d_e4), TOL)
        self.assertLessEqual(abs(d_f4), TOL)

    def test_reference_channel(self):
        on_rows = sorted(int(round(12.0 + 64.0 * s + 10.0 * k))
                         for s in range(4) for k in range(5))
        for r in on_rows:
            self.assertTrue(bool((self.ref[0, r, :] == 1.0).all()), f"ref row {r} not +1")
        mask = torch.ones(256, dtype=torch.bool)
        mask[on_rows] = False
        self.assertTrue(bool((self.ref[0, mask, :] == 0.0).all()), "ref nonzero off-line")
        print(f"4-staff ref: +1 rows 12..244 step pattern, zeros elsewhere")


class ScalarSingleStaff(unittest.TestCase):
    def test_scalar_lines_and_full_white_fill(self):
        canvas = np.full((H_IN, W_IN), 255.0, dtype=np.float64)
        lines = [150.0 + 15.0 * k for k in range(5)]  # scalar format, sp = 15
        c = np.array(lines)[:, None]
        draw_lines(canvas, np.repeat(c, W_IN, axis=1), thick=2.0)
        img = Image.fromarray(canvas.astype(np.uint8), mode="L")
        out_img, ref = R.rectify_system_strip(img, [lines])
        arr = np.asarray(out_img)
        self.assertEqual((img.size[1], arr.shape[1]), (900, round(W_IN * 10.0 / 15.0)))
        keep = np.arange(arr.shape[1])
        res = line_residuals(arr, [12.0 + 10.0 * k for k in range(5)], keep)
        worst = max(abs(v) for v in res.values())
        white = int((arr[64:256] != 255).sum())
        print(f"scalar 1-staff: W_out={arr.shape[1]} worst-line-resid={worst:.4f} "
              f"slots1-3 nonwhite={white}")
        self.assertLessEqual(worst, TOL)
        self.assertEqual(white, 0)
        self.assertTrue(bool((ref[0, [12, 22, 32, 42, 52], :] == 1.0).all()))

    def test_rejects_bad_input(self):
        img = Image.new("L", (100, 100), 255)
        with self.assertRaises(ValueError):
            R.rectify_system_strip(img, [])
        with self.assertRaises(ValueError):
            R.rectify_system_strip(img, [[[10.0] * 5]] * 5)
        with self.assertRaises(ValueError):
            R.rectify_system_strip(img, [[10.0, 20.0, 30.0, 40.0]])
        with self.assertRaises(TypeError):
            R.rectify_system_strip(np.zeros((10, 10)), [[10.0] * 5])


if __name__ == "__main__":
    unittest.main()
