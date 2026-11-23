"""Unit tests for tmp-train/mvd/bezier_ribbon.py (Batch 5-B2)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

MVD_DIR = Path(__file__).resolve().parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import bezier_ribbon as br  # noqa: E402
import staff_unet2d as su  # noqa: E402


def _render_warped_staff_page(
    page_h: int,
    page_w: int,
    staves: list[br.BezierStaff],
    offset_mode: str = "normal",
    line_sigma: float = 0.85,
) -> torch.Tensor:
    """Synthesize a clean [1, H, W] float page in [0, 1] with dark Gaussian staff lines.

    Each staff's 5 lines are drawn directly in page coordinates along its cubic
    Bezier centerline (using either normal or vertical offsets) so round-trip
    unwarping accuracy can be measured to sub-pixel precision.
    """
    canvas = np.ones((page_h, page_w), dtype=np.float32)
    t_dense = torch.linspace(0.0, 1.0, 2048, dtype=torch.float32)

    for st in staves:
        cp = torch.as_tensor(st.ctrl_pts, dtype=torch.float32)
        sp_t = torch.as_tensor(st.sp_ctrl, dtype=torch.float32)
        if sp_t.ndim == 0:
            sp_t = sp_t.expand(4)
        pts = br.eval_cubic_bezier(cp, t_dense)  # (N, 2)
        sp_vals = br.eval_cubic_bezier(sp_t.unsqueeze(-1), t_dense).squeeze(-1)  # (N,)

        if offset_mode == "normal":
            tang = br.eval_cubic_bezier_tangent(cp, t_dense)
            t_hat = tang / torch.linalg.norm(tang, dim=-1, keepdim=True).clamp(min=1e-6)
            n_hat = torch.stack([-t_hat[:, 1], t_hat[:, 0]], dim=-1)
            n_hat = n_hat * torch.where(n_hat[:, 1:2] >= 0, 1.0, -1.0)
        else:
            n_hat = torch.zeros_like(pts)
            n_hat[:, 1] = 1.0

        pts_np = pts.numpy()
        n_np = n_hat.numpy()
        sp_np = sp_vals.numpy()

        for k in range(5):
            off = float(k - 2) * sp_np[:, None]
            lpts = pts_np + off * n_np  # (N, 2)
            # Splat smooth Gaussian dark ink around each line point.
            xs = np.round(lpts[:, 0]).astype(int)
            ys = lpts[:, 1]
            for dy in range(-2, 3):
                yy = np.round(ys).astype(int) + dy
                valid = (xs >= 0) & (xs < page_w) & (yy >= 0) & (yy < page_h)
                if not np.any(valid):
                    continue
                dist = yy[valid] - ys[valid]
                ink = np.exp(-0.5 * (dist / line_sigma) ** 2).astype(np.float32)
                np.minimum.at(canvas, (yy[valid], xs[valid]), 1.0 - 0.9 * ink)

    return torch.from_numpy(canvas[None, :, :])  # [1, H, W]


def _measure_line_errors(
    ribbon: torch.Tensor,
    n_staves: int,
    target_sp: float = 10.0,
    height: int = 256,
    margin_cols: int = 12,
) -> float:
    """Measure max absolute vertical deviation (px) of all 5 staff lines per staff."""
    arr = ribbon[0].detach().cpu().numpy()  # (H, W)
    ink = np.clip(1.0 - arr, 0.0, 1.0)
    h, w = ink.shape
    cols = slice(margin_cols, w - margin_cols)
    max_err = 0.0
    for s in range(n_staves):
        for k in range(5):
            yc = br.canon_y(s, k, target_sp=target_sp, height=height)
            r0 = max(0, int(round(yc)) - 3)
            r1 = min(h, int(round(yc)) + 4)
            band = ink[r0:r1, cols]  # (band_h, W')
            rows = np.arange(r0, r1, dtype=np.float64)[:, None]
            wsum = band.sum(axis=0)
            assert float(wsum.min()) > 0.1, f"missing staff ink at slot {s} line {k}"
            centroids = (rows * band).sum(axis=0) / wsum
            err = float(np.max(np.abs(centroids - yc)))
            max_err = max(max_err, err)
    return max_err


class TestBezierRibbon(unittest.TestCase):
    def test_canonical_layout_matches_rectify(self):
        top, pitch, out_h = br.canonical_layout(target_sp=10.0, height=256)
        self.assertEqual((top, pitch, out_h), (12.0, 64.0, 256))
        self.assertAlmostEqual(br.canon_y(0, 0), 12.0)
        self.assertAlmostEqual(br.canon_y(0, 4), 52.0)
        self.assertAlmostEqual(br.canon_y(1, 0), 76.0)

    def test_roundtrip_10deg_tilted_2staff(self):
        """10-degree tilted 2-staff system (sq7070319-1 regime) recovers flat <= 1.0 px."""
        page_h, page_w = 1024, 1200
        sp = 12.0
        theta = math.radians(10.0)
        cx, cy = 600.0, 460.0
        half_len = 420.0
        tx, ty = math.cos(theta), math.sin(theta)
        nx, ny = -ty, tx

        staves = []
        for s_off in (-70.0, 70.0):
            pts = []
            for frac in np.linspace(-1.0, 1.0, 4):
                x = cx + frac * half_len * tx + s_off * nx
                y = cy + frac * half_len * ty + s_off * ny
                pts.append([x, y])
            staves.append(br.BezierStaff(ctrl_pts=np.array(pts, dtype=np.float32), sp_ctrl=sp))

        page = _render_warped_staff_page(page_h, page_w, staves, offset_mode="normal")
        ribbon, ref = br.sample_bezier_ribbon(page, staves, target_sp=10.0, height=256, offset_mode="normal")

        self.assertEqual(ribbon.shape[0], 1)
        self.assertEqual(ribbon.shape[1], 256)
        self.assertEqual(tuple(ref.shape), (1, 256, ribbon.shape[2]))

        max_err = _measure_line_errors(ribbon, n_staves=2)
        self.assertLessEqual(max_err, 1.0, f"10-deg tilted max error {max_err:.3f} px > 1.0 px")

        # Inactive slots 2..3 must be clean paper white (1.0).
        self.assertAlmostEqual(float(ribbon[:, 128:, :].min()), 1.0, places=5)

    def test_roundtrip_15deg_scurved_2staff(self):
        """15-degree S-curved 2-staff system (ds0324-7 regime) recovers flat <= 1.0 px."""
        page_h, page_w = 1024, 1200
        sp = 11.0
        x0, x1 = 120.0, 1080.0
        xs = np.linspace(x0, x1, 64)
        u = (xs - x0) / (x1 - x0)
        amp = math.tan(math.radians(15.0)) * (x1 - x0) / (2.0 * math.pi)

        staves = []
        for y_base in (380.0, 530.0):
            ys = y_base + amp * np.sin(2.0 * math.pi * u)
            y_ctrl = su.fit_cubic_bezier_1d(u, ys)
            x_ctrl = np.linspace(x0, x1, 4)
            cp = np.stack([x_ctrl, y_ctrl], axis=-1).astype(np.float32)
            staves.append(br.BezierStaff(ctrl_pts=cp, sp_ctrl=sp))

        # Test both normal-offset and vertical-offset S-curve recovery.
        for mode in ("normal", "vertical"):
            page = _render_warped_staff_page(page_h, page_w, staves, offset_mode=mode)
            ribbon, _ = br.sample_bezier_ribbon(page, staves, target_sp=10.0, height=256, offset_mode=mode)
            max_err = _measure_line_errors(ribbon, n_staves=2)
            self.assertLessEqual(
                max_err, 1.0, f"15-deg S-curve ({mode}) max error {max_err:.3f} px > 1.0 px"
            )

    def test_roundtrip_perspective_varying_sp(self):
        """Perspective-tapered staff (sp varying 9 -> 15 px) recovers flat <= 1.0 px."""
        page_h, page_w = 640, 1000
        cp = np.array([[100.0, 260.0], [360.0, 275.0], [640.0, 295.0], [900.0, 320.0]], dtype=np.float32)
        sp_ctrl = np.array([9.0, 11.0, 13.0, 15.0], dtype=np.float32)
        staves = [br.BezierStaff(ctrl_pts=cp, sp_ctrl=sp_ctrl)]

        page = _render_warped_staff_page(page_h, page_w, staves, offset_mode="normal")
        ribbon, _ = br.sample_bezier_ribbon(page, staves, target_sp=10.0, height=256, offset_mode="normal")
        max_err = _measure_line_errors(ribbon, n_staves=1)
        self.assertLessEqual(max_err, 1.0, f"perspective sp max error {max_err:.3f} px > 1.0 px")

    def test_end_to_end_extract_then_ribbon(self):
        """Integration: rasterize_staff_targets -> extract_curved_systems -> sample_bezier_ribbon (normal)."""
        page_h, page_w = 1024, 1024
        sp_normal = 12.0
        x0, x1 = 100.0, 924.0
        xs = np.linspace(x0, x1, 64)
        u = (xs - x0) / (x1 - x0)
        slope = math.tan(math.radians(9.0))
        y0 = 360.0 + slope * (xs - 512.0) + 18.0 * np.sin(2.0 * math.pi * u)
        y1 = 500.0 + slope * (xs - 512.0) + 18.0 * np.sin(2.0 * math.pi * u)

        # Vertical spacing at each x for a curved staff with true normal spacing sp_normal.
        cos_theta = np.cos(np.arctan(np.gradient(y0, xs)))
        sp_vert = sp_normal / cos_theta

        spec = [
            {"y_center": y0, "sp": sp_vert, "x0": x0, "x1": x1, "system_id": 0},
            {"y_center": y1, "sp": sp_vert, "x0": x0, "x1": x1, "system_id": 0},
        ]
        gt_staves = [
            br.BezierStaff(
                ctrl_pts=np.stack([np.linspace(x0, x1, 4), su.fit_cubic_bezier_1d(u, y0)], axis=-1),
                sp_ctrl=sp_normal,
            ),
            br.BezierStaff(
                ctrl_pts=np.stack([np.linspace(x0, x1, 4), su.fit_cubic_bezier_1d(u, y1)], axis=-1),
                sp_ctrl=sp_normal,
            ),
        ]
        page = _render_warped_staff_page(page_h, page_w, gt_staves, offset_mode="normal")

        tgt = su.rasterize_staff_targets(spec, w=page_w, h=page_h, out_h=page_h // 2, out_w=page_w // 2)
        systems = su.extract_curved_systems(tgt, orig_w=page_w, orig_h=page_h)
        self.assertEqual(len(systems), 1)

        ribbon, _ = br.sample_bezier_ribbon(page, systems[0], target_sp=10.0, height=256, offset_mode="normal")
        max_err = _measure_line_errors(ribbon, n_staves=2)
        self.assertLessEqual(max_err, 1.0, f"end-to-end max error {max_err:.3f} px > 1.0 px")

    def test_pil_batch_and_top_bottom_helper(self):
        top = np.array([[50.0, 120.0], [200.0, 130.0], [350.0, 140.0], [500.0, 150.0]], dtype=np.float32)
        bot = top + np.array([0.0, 120.0], dtype=np.float32)
        staves = br.staves_from_top_bottom(top, bot, sp_ctrl=10.0, n_staves=2)
        self.assertEqual(len(staves), 2)

        pil_rgb = Image.new("RGB", (600, 400), color=(255, 255, 255))
        out_img, ref = br.sample_bezier_ribbon(pil_rgb, staves, height=256)
        self.assertIsInstance(out_img, Image.Image)
        self.assertEqual(out_img.mode, "RGB")
        self.assertEqual(out_img.size[1], 256)
        self.assertEqual(ref.shape[1], 256)

        batch = torch.ones(2, 3, 400, 600)
        sampler = br.BezierRibbonSampler(height=256)
        out_b, _ = sampler(batch, staves, width=320)
        self.assertEqual(tuple(out_b.shape), (2, 3, 256, 320))

    def test_chunk_staves_for_ribbon_and_6staff_system(self):
        """6-staff curved system (palestrina/beach/aloha regime) chunks cleanly into <=4-staff ribbons."""
        page_h, page_w = 1280, 1200
        sp = 10.0
        x0, x1 = 100.0, 1100.0
        xs = np.linspace(x0, x1, 64)
        u = (xs - x0) / (x1 - x0)
        slope = math.tan(math.radians(6.0))

        staves = []
        for s_idx in range(6):
            y_base = 180.0 + s_idx * 150.0
            ys = y_base + slope * (xs - 600.0) + 12.0 * np.sin(2.0 * math.pi * u)
            cp = np.stack([np.linspace(x0, x1, 4), su.fit_cubic_bezier_1d(u, ys)], axis=-1).astype(np.float32)
            staves.append(br.BezierStaff(ctrl_pts=cp, sp_ctrl=sp))

        # Raw chunking: 6 staves -> [4, 2] with overlap=0, [4, 3] with overlap=1.
        chunks0 = br.chunk_staves_for_ribbon(staves, max_staves=4, overlap=0)
        self.assertEqual([len(c) for c in chunks0], [4, 2])
        chunks1 = br.chunk_staves_for_ribbon(staves, max_staves=4, overlap=1)
        self.assertEqual([len(c) for c in chunks1], [4, 3])
        self.assertIs(chunks1[0][3], chunks1[1][0])

        # 5-staff system -> [4, 1]; 2-staff system -> [2].
        self.assertEqual([len(c) for c in br.chunk_staves_for_ribbon(staves[:5])], [4, 1])
        self.assertEqual([len(c) for c in br.chunk_staves_for_ribbon(staves[:2])], [2])

        # Invalid overlap raises ValueError.
        with self.assertRaises(ValueError):
            br.chunk_staves_for_ribbon(staves, max_staves=4, overlap=4)

        # Full round-trip sampling on the 6-staff curved page (no ValueError!).
        page = _render_warped_staff_page(page_h, page_w, staves, offset_mode="normal")
        results = br.sample_curved_system_chunks(
            page, staves, max_staves=4, overlap=0, target_sp=10.0, height=256, offset_mode="normal"
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][2], [0, 1, 2, 3])
        self.assertEqual(results[1][2], [4, 5])

        for chunk_idx, (ribbon, ref, idxs) in enumerate(results):
            self.assertEqual(ribbon.shape[1], 256)
            self.assertEqual(ref.shape[1], 256)
            max_err = _measure_line_errors(ribbon, n_staves=len(idxs))
            self.assertLessEqual(
                max_err, 1.0, f"6-staff chunk {chunk_idx} max error {max_err:.3f} px > 1.0 px"
            )
            # Inactive slots in chunk 1 (slots 2..3) must be clean paper white.
            if len(idxs) < 4:
                self.assertAlmostEqual(float(ribbon[:, len(idxs) * 64 :, :].min()), 1.0, places=5)

    def test_differentiability_cpu_and_cuda(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for dev in devices:
            page = torch.rand(1, 3, 256, 384, device=dev, requires_grad=True)
            cp = torch.tensor(
                [[40.0, 100.0], [140.0, 110.0], [240.0, 120.0], [340.0, 115.0]],
                device=dev,
                requires_grad=True,
            )
            sp = torch.tensor([9.0, 9.5, 10.0, 9.5], device=dev, requires_grad=True)
            sampler = br.BezierRibbonSampler(height=256)
            ribbon, _ = sampler(page, [br.BezierStaff(ctrl_pts=cp, sp_ctrl=sp)], width=256)
            loss = ribbon.mean()
            loss.backward()
            self.assertIsNotNone(page.grad)
            self.assertIsNotNone(cp.grad)
            self.assertIsNotNone(sp.grad)
            self.assertGreater(float(page.grad.abs().sum()), 0.0)
            self.assertGreater(float(cp.grad.abs().sum()), 0.0)
            self.assertGreater(float(sp.grad.abs().sum()), 0.0)

    def test_comb_snap_recovers_vertical_and_sp_jitter_on_curved_shadowed_page(self):
        """Comb snap locks perturbed StaffUNet2D curves (< 0.45 px) even with shadow & note blobs."""
        page_h, page_w = 1024, 1200
        sp = 12.0
        x0, x1 = 100.0, 1100.0
        xs = np.linspace(x0, x1, 64)
        u = (xs - x0) / (x1 - x0)
        slope = math.tan(math.radians(7.0))

        gt_staves = []
        perturbed_staves = []
        for idx, y_base in enumerate((340.0, 520.0)):
            ys = y_base + slope * (xs - 600.0) + 14.0 * np.sin(2.0 * math.pi * u)
            cp_gt = np.stack([np.linspace(x0, x1, 4), su.fit_cubic_bezier_1d(u, ys)], axis=-1).astype(np.float32)
            gt_staves.append(br.BezierStaff(ctrl_pts=cp_gt, sp_ctrl=sp))
            # Perturb y by +2.8/-2.4 px + slight tilt and sp by +/-8%.
            cp_bad = cp_gt.copy()
            cp_bad[:, 1] += (2.8 if idx == 0 else -2.4) + np.array([-0.8, -0.2, 0.3, 0.9], dtype=np.float32)
            sp_bad = sp * (1.08 if idx == 0 else 0.93)
            perturbed_staves.append(br.BezierStaff(ctrl_pts=cp_bad, sp_ctrl=sp_bad))

        page = _render_warped_staff_page(page_h, page_w, gt_staves, offset_mode="normal")
        # Add realistic uneven lighting ramp + a few dark notehead blobs.
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, page_h, dtype=np.float32),
            np.linspace(0.0, 1.0, page_w, dtype=np.float32),
            indexing="ij",
        )
        shade = 0.72 + 0.25 * (0.5 + 0.5 * np.cos(math.pi * xx) * np.sin(math.pi * yy))
        degraded = (page * torch.from_numpy(shade[None, :, :])).clamp(0.0, 1.0)
        # Drop dark noteheads on several columns.
        for cx in (240, 420, 610, 780, 950):
            degraded[:, 320:360, cx:cx + 12] *= 0.15

        raw_ribbon, _ = br.sample_bezier_ribbon(degraded, perturbed_staves, target_sp=10.0, height=256, comb_snap=False)
        snapped_ribbon, _ = br.sample_bezier_ribbon(degraded, perturbed_staves, target_sp=10.0, height=256, comb_snap=True)

        # Measure error on clean page using the raw vs snapped staves.
        snapped_staves = br.refine_staves_comb(degraded, perturbed_staves, target_sp=10.0, height=256)
        raw_clean, _ = br.sample_bezier_ribbon(page, perturbed_staves, target_sp=10.0, height=256)
        snap_clean, _ = br.sample_bezier_ribbon(page, snapped_staves, target_sp=10.0, height=256)

        err_before = _measure_line_errors(raw_clean, n_staves=2)
        err_after = _measure_line_errors(snap_clean, n_staves=2)
        self.assertGreater(err_before, 2.0)
        self.assertLessEqual(err_after, 0.55, f"comb_snap max error {err_after:.3f} px > 0.55 px (was {err_before:.3f})")
        self.assertEqual(snapped_ribbon.shape[:2], (1, 256))

    def test_comb_snap_blank_page_safe_fallback(self):
        """Blank/washed-out page leaves input Bezier curves intact (no drift or crash)."""
        cp = np.array([[80.0, 200.0], [260.0, 200.0], [440.0, 200.0], [620.0, 200.0]], dtype=np.float32)
        staves = [br.BezierStaff(ctrl_pts=cp, sp_ctrl=10.0)]
        blank = Image.new("RGB", (700, 400), color=(245, 240, 232))
        refined = br.refine_staves_comb(blank, staves, target_sp=10.0, height=256)
        self.assertEqual(len(refined), 1)
        np.testing.assert_allclose(
            torch.as_tensor(refined[0].ctrl_pts).numpy(), cp, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
