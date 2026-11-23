"""Unit tests for tmp-train/mvd/staff_unet2d.py."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

MVD_DIR = Path(__file__).resolve().parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import staff_unet2d as su  # noqa: E402


class TestStaffUNet2DModel(unittest.TestCase):
    def test_param_count_and_shapes(self):
        # Default is RGB (in_channels=3) with 1/2-res skip-up head (out_stride=2).
        model = su.StaffUNet2D()
        self.assertEqual(model.in_channels, 3)
        self.assertEqual(model.out_stride, 2)
        n_params = model.num_parameters()
        self.assertGreaterEqual(n_params, 1_000_000)
        self.assertLessEqual(n_params, 2_000_000)

        x = torch.randn(2, 3, 256, 192)
        out = model(x)
        self.assertEqual(tuple(out.shape), (2, 4, 128, 96))

        with torch.no_grad():
            probs = model.predict_probs(x)
        self.assertEqual(tuple(probs.shape), (2, 4, 128, 96))
        self.assertGreaterEqual(float(probs[:, :3].min()), 0.0)
        self.assertLessEqual(float(probs[:, :3].max()), 1.0)
        self.assertGreater(float(probs[:, 3].min()), 0.0)

        # Backward compatibility: in_channels=1 and out_stride=4 still work cleanly.
        m_gray = su.StaffUNet2D(in_channels=1, out_stride=4)
        out_g = m_gray(torch.randn(1, 1, 256, 192))
        self.assertEqual(tuple(out_g.shape), (1, 4, 64, 48))

    def test_rgb_and_non_multiple_of_32_input(self):
        model = su.StaffUNet2D(in_channels=3, out_stride=2)
        x = torch.randn(1, 3, 300, 210)
        out = model(x)
        self.assertEqual(tuple(out.shape), (1, 4, 150, 105))

    def test_anti_aliased_resize_preserves_thin_staff_lines(self):
        # High-res page with 1-px black horizontal staff lines downscaled >3x with BOX.
        hi_w, hi_h = 2400, 3200
        arr = np.full((hi_h, hi_w, 3), 255, dtype=np.uint8)
        line_ys = [600 + k * 24 for k in range(5)]
        for y in line_ys:
            arr[y, 200:2200, :] = 0
        img = Image.fromarray(arr, mode="RGB")

        ten = su.preprocess_page_image(img, target_h=1024, target_w=768, in_channels=3)
        self.assertEqual(tuple(ten.shape), (3, 1024, 768))
        # Every single downscaled staff line must remain visibly darker than white paper.
        mid_col = ten[0, :, 384].numpy()
        for y in line_ys:
            ry = int(round(y * 1024.0 / hi_h))
            local_min = float(mid_col[max(0, ry - 1) : min(1024, ry + 2)].min())
            self.assertLess(local_min, 0.85, f"thin 1-px staff line at y={y} washed out ({local_min:.3f})")

    def test_loss_finiteness_and_backward(self):
        model = su.StaffUNet2D(in_channels=3, out_stride=2)
        crit = su.StaffUNetLoss()
        x = torch.randn(2, 3, 128, 96)
        tgt = torch.zeros(2, 4, 64, 48)
        tgt[0, 0, 20:30, 4:40] = 1.0
        tgt[0, 1, 25, 4:40] = 1.0
        tgt[0, 2, 18:32, 4:40] = 1.0
        tgt[0, 3, 18:32, 4:40] = 10.0 / 128.0

        preds = model(x)
        loss, metrics = crit(preds, tgt)
        self.assertTrue(torch.isfinite(loss))
        for k, v in metrics.items():
            self.assertTrue(math.isfinite(v), f"non-finite {k}={v}")
        loss.backward()
        grad_norm = sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad_norm, 0.0)

        # All-empty target batch must still produce finite loss (no 0/0 NaN).
        empty_loss, _ = crit(model(x), torch.zeros_like(tgt))
        self.assertTrue(torch.isfinite(empty_loss))

    def test_onnx_export(self):
        import onnx

        model = su.StaffUNet2D(in_channels=3, out_stride=2).eval()
        dummy = torch.randn(1, 3, 256, 192)
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "staff_unet2d.onnx"
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["image"],
                output_names=["maps"],
                opset_version=17,
                dynamo=False,
            )
            self.assertTrue(onnx_path.exists())
            om = onnx.load(str(onnx_path))
            onnx.checker.check_model(om)

    def test_latency_budget_1024(self):
        model = su.StaffUNet2D(in_channels=3, out_stride=2).eval()
        x_cpu = torch.randn(1, 3, 1024, 768)
        with torch.no_grad():
            out_cpu = model(x_cpu)  # warmup
            self.assertEqual(tuple(out_cpu.shape), (1, 4, 512, 384))
            t0 = time.perf_counter()
            for _ in range(3):
                _ = model(x_cpu)
            cpu_ms = (time.perf_counter() - t0) * 1000.0 / 3.0
        self.assertLess(cpu_ms, 120.0, f"CPU latency {cpu_ms:.1f} ms exceeded 120 ms budget")

        if torch.cuda.is_available():
            model_gpu = model.cuda()
            x_gpu = x_cpu.cuda()
            with torch.no_grad():
                _ = model_gpu(x_gpu)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(5):
                    _ = model_gpu(x_gpu)
                torch.cuda.synchronize()
                gpu_ms = (time.perf_counter() - t0) * 1000.0 / 5.0
            self.assertLess(gpu_ms, 15.0, f"GPU latency {gpu_ms:.1f} ms exceeded 15 ms budget")


class TestRasterizeAndExtract(unittest.TestCase):
    def test_rasterize_hybrid_dict(self):
        w, h = 800, 1024
        out_h, out_w = 256, 200
        sp = 12.0
        s0 = [200.0 + k * sp for k in range(5)]
        s1 = [340.0 + k * sp for k in range(5)]
        tgt = su.rasterize_staff_targets({"staves": [s0, s1]}, w=w, h=h, out_h=out_h, out_w=out_w)
        self.assertEqual(tuple(tgt.shape), (4, out_h, out_w))
        self.assertGreater(float(tgt[0].sum()), 100.0)
        self.assertGreater(float(tgt[1].sum()), 50.0)
        # System body should span between s0 and s1 (around y=270 -> out_y=67).
        self.assertGreater(float(tgt[2, 67, 100]), 0.5)
        # Scale map at staff 0 center (y=224 -> out_y=56) should equal sp / h.
        self.assertAlmostEqual(float(tgt[3, 56, 100]), sp / h, places=4)

    def test_extract_curved_systems_tilted_and_scurved(self):
        w, h = 1024, 1280
        out_h, out_w = h // 4, w // 4
        sp = 12.0
        x0, x1 = 96.0, 928.0
        xs = np.linspace(x0, x1, 64)
        u = (xs - x0) / (x1 - x0)

        # System 0: 10-degree tilted 2-staff grand staff.
        tilt_slope = math.tan(math.radians(10.0))
        y_tilt_s0 = 260.0 + tilt_slope * (xs - 0.5 * (x0 + x1))
        y_tilt_s1 = 400.0 + tilt_slope * (xs - 0.5 * (x0 + x1))

        # System 1: 15-degree max-slope cubic S-curved 2-staff grand staff.
        # Cubic S-curve: f(u) = A * u * (u - 0.5) * (u - 1.0); max |df/du| = A/2 at u=0,1.
        # Setting A / (2 * (x1 - x0)) = tan(15 deg) gives exact 15-deg peak slope.
        amp_cubic = 2.0 * math.tan(math.radians(15.0)) * (x1 - x0)
        scurve = amp_cubic * u * (u - 0.5) * (u - 1.0)
        y_scurve_s0 = 780.0 + scurve
        y_scurve_s1 = 920.0 + scurve

        staves_spec = [
            {"y_center": y_tilt_s0, "sp": sp, "x0": x0, "x1": x1, "system_id": 0},
            {"y_center": y_tilt_s1, "sp": sp, "x0": x0, "x1": x1, "system_id": 0},
            {"y_center": y_scurve_s0, "sp": sp, "x0": x0, "x1": x1, "system_id": 1},
            {"y_center": y_scurve_s1, "sp": sp, "x0": x0, "x1": x1, "system_id": 1},
        ]
        gt_curves = [y_tilt_s0, y_tilt_s1, y_scurve_s0, y_scurve_s1]

        tgt = su.rasterize_staff_targets(staves_spec, w=w, h=h, out_h=out_h, out_w=out_w)
        systems = su.extract_curved_systems(tgt, orig_w=w, orig_h=h)

        self.assertEqual(len(systems), 2)
        self.assertEqual(len(systems[0].staves), 2)
        self.assertEqual(len(systems[1].staves), 2)

        extracted = systems[0].staves + systems[1].staves
        x_eval = np.linspace(x0 + 20.0, x1 - 20.0, 40)
        for idx, (st, gt_y) in enumerate(zip(extracted, gt_curves)):
            gt_interp = np.interp(x_eval, xs, gt_y)
            pred_y = st.eval_y_at_x(x_eval)
            err = float(np.max(np.abs(pred_y - gt_interp)))
            self.assertLess(err, 2.0, f"staff {idx} centerline max error {err:.2f} px >= 2.0 px")
            self.assertLess(abs(st.sp_median - sp), 0.5, f"staff {idx} sp {st.sp_median:.2f} vs {sp}")

        # Verify JSON serialization works cleanly.
        d = systems[0].to_dict()
        self.assertEqual(len(d["staves"]), 2)
        self.assertGreater(len(d["polygon"]), 10)

    def test_normal_sp_accuracy_on_15deg_tilt(self):
        """ch3 must supervise sp_vert * cos(atan(dy/dx)) on a 15-deg tilted staff."""
        w, h = 1024, 1024
        out_h, out_w = h // 2, w // 2
        angle = math.radians(15.0)
        cos_a = math.cos(angle)
        tan_a = math.tan(angle)
        x0, x1 = 120.0, 904.0
        xs = np.linspace(x0, x1, 64)

        # Case A: vertical sp = 12.0 -> normal sp = 12.0 * cos(15 deg).
        sp_vert = 12.0
        yc = 512.0 + tan_a * (xs - 512.0)
        tgt_a = su.rasterize_staff_targets(
            [{"y_center": yc, "sp": sp_vert, "x0": x0, "x1": x1, "system_id": 0}],
            w=w,
            h=h,
            out_h=out_h,
            out_w=out_w,
        )
        expected_a = sp_vert * cos_a
        mid_c = out_w // 2
        mid_r = out_h // 2
        self.assertAlmostEqual(float(tgt_a[3, mid_r, mid_c]) * h, expected_a, delta=0.15)
        sys_a = su.extract_curved_systems(tgt_a, orig_w=w, orig_h=h)
        self.assertEqual(len(sys_a), 1)
        self.assertAlmostEqual(sys_a[0].staves[0].sp_median, expected_a, delta=0.15)

        # Case B: true normal sp = 12.0 rotated by 15 deg -> vertical spacing is 12.0 / cos(15 deg),
        # so rasterize_staff_targets must recover ch3 = 12.0 / h.
        sp_true = 12.0
        tgt_b = su.rasterize_staff_targets(
            [{"y_center": yc, "sp": sp_true / cos_a, "x0": x0, "x1": x1, "system_id": 0}],
            w=w,
            h=h,
            out_h=out_h,
            out_w=out_w,
        )
        self.assertAlmostEqual(float(tgt_b[3, mid_r, mid_c]) * h, sp_true, delta=0.15)
        sys_b = su.extract_curved_systems(tgt_b, orig_w=w, orig_h=h)
        self.assertEqual(len(sys_b), 1)
        self.assertAlmostEqual(sys_b[0].staves[0].sp_median, sp_true, delta=0.15)

    def test_horizontal_fragment_merging_across_stain_gap(self):
        """A stain/crease splitting ch1 horizontally must merge back into 1 staff (not S+1)."""
        w, h = 1024, 1024
        out_h, out_w = h // 2, w // 2
        sp = 12.0
        x0, x1 = 100.0, 924.0
        xs = np.linspace(x0, x1, 64)
        slope = math.tan(math.radians(8.0))
        y0 = 320.0 + slope * (xs - 512.0)
        y1 = 460.0 + slope * (xs - 512.0)

        tgt = su.rasterize_staff_targets(
            [
                {"y_center": y0, "sp": sp, "x0": x0, "x1": x1, "system_id": 0},
                {"y_center": y1, "sp": sp, "x0": x0, "x1": x1, "system_id": 0},
            ],
            w=w,
            h=h,
            out_h=out_h,
            out_w=out_w,
        ).clone()

        # Punch a wide 20-col (40-px page) stain gap right through staff 0's centerline.
        gap_c0, gap_c1 = out_w // 2 - 10, out_w // 2 + 10
        tgt[1, :200, gap_c0:gap_c1] = 0.0

        systems = su.extract_curved_systems(tgt, orig_w=w, orig_h=h)
        self.assertEqual(len(systems), 1)
        self.assertEqual(len(systems[0].staves), 2, "stain gap should not duplicate staff into S+1")

        x_eval = np.linspace(x0 + 20.0, x1 - 20.0, 40)
        err0 = float(np.max(np.abs(systems[0].staves[0].eval_y_at_x(x_eval) - np.interp(x_eval, xs, y0))))
        self.assertLess(err0, 2.0, f"merged staff centerline error {err0:.2f} px >= 2.0 px")

    def test_short_coda_survives_20pct_desk_border(self):
        """Short trailing coda (15% of normal staff width) survives 20% left/right desk border."""
        w, h = 1024, 1280
        out_h, out_w = h // 2, w // 2
        sp = 10.0
        # 20% desk border on both left and right -> normal staves span [205, 819] (60% of W).
        x0, x1 = 0.20 * w, 0.80 * w
        coda_x1 = x0 + 0.15 * (x1 - x0)  # 9% of W (< old 12% fixed threshold, > 6% floor & 10% median)

        staves_spec = [
            {"y_center": 240.0, "sp": sp, "x0": x0, "x1": x1, "system_id": 0},
            {"y_center": 360.0, "sp": sp, "x0": x0, "x1": x1, "system_id": 0},
            {"y_center": 600.0, "sp": sp, "x0": x0, "x1": x1, "system_id": 1},
            {"y_center": 720.0, "sp": sp, "x0": x0, "x1": x1, "system_id": 1},
            {"y_center": 960.0, "sp": sp, "x0": x0, "x1": coda_x1, "system_id": 2},
        ]
        tgt = su.rasterize_staff_targets(staves_spec, w=w, h=h, out_h=out_h, out_w=out_w)
        systems = su.extract_curved_systems(tgt, orig_w=w, orig_h=h)
        self.assertEqual(len(systems), 3, "short trailing coda should survive with 20% desk border")
        self.assertEqual([len(s.staves) for s in systems], [2, 2, 1])

    def test_ch2_vertical_bleed_guard_dense_6staff(self):
        """1-px vertical ch2 bleed between dense 3-staff systems must not merge them."""
        w, h = 1024, 1280
        out_h, out_w = h // 2, w // 2
        sp = 10.0
        x0, x1 = 96.0, 928.0
        staves_spec = [
            {"y_center": 180.0, "sp": sp, "x0": x0, "x1": x1, "system_id": 0},
            {"y_center": 290.0, "sp": sp, "x0": x0, "x1": x1, "system_id": 0},
            {"y_center": 400.0, "sp": sp, "x0": x0, "x1": x1, "system_id": 0},
            {"y_center": 560.0, "sp": sp, "x0": x0, "x1": x1, "system_id": 1},
            {"y_center": 670.0, "sp": sp, "x0": x0, "x1": x1, "system_id": 1},
            {"y_center": 780.0, "sp": sp, "x0": x0, "x1": x1, "system_id": 1},
        ]
        tgt = su.rasterize_staff_targets(staves_spec, w=w, h=h, out_h=out_h, out_w=out_w).clone()
        # Inject a narrow vertical cross-system bleed bridge in ch2 between system 0 and system 1.
        r_bridge0 = int(round(400.0 / 2.0))
        r_bridge1 = int(round(560.0 / 2.0))
        tgt[2, r_bridge0:r_bridge1, out_w // 2 : out_w // 2 + 2] = 0.70

        systems = su.extract_curved_systems(tgt, orig_w=w, orig_h=h)
        self.assertEqual(len(systems), 2, "vertical ch2 bleed must not merge two 3-staff systems")
        self.assertEqual([len(s.staves) for s in systems], [3, 3])


class TestTrainingSmoke(unittest.TestCase):
    def test_dataset_and_one_epoch_train(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            img_path = td_path / "page0.png"
            Image.new("RGB", (256, 320), color=(250, 248, 242)).save(img_path)
            manifest = td_path / "manifest.jsonl"
            row = {
                "image": "page0.png",
                "staves": [{"y_center": 120.0, "sp": 8.0, "x0": 20.0, "x1": 236.0, "system_id": 0}],
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

            ds = su.StaffUNetDataset(manifest, target_h=128, target_w=96)
            self.assertEqual(len(ds), 1)
            x, y = ds[0]
            self.assertEqual(tuple(x.shape), (3, 128, 96))
            self.assertEqual(tuple(y.shape), (4, 64, 48))

            ckpt = su.train_staff_unet2d(
                manifest=manifest,
                out_dir=td_path / "ckpt",
                epochs=1,
                batch_size=1,
                target_h=128,
                target_w=96,
                base_channels=16,
                device="cpu",
            )
            self.assertTrue(ckpt.exists())

    def test_rgb_augmented_dataset_determinism_and_batching(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # Draw a simple RGB page with 2 systems (4 staves).
            arr = np.full((320, 256, 3), 248, dtype=np.uint8)
            staves = [
                {"y_center": 70.0, "sp": 6.0, "x0": 24.0, "x1": 232.0, "system_id": 0},
                {"y_center": 120.0, "sp": 6.0, "x0": 24.0, "x1": 232.0, "system_id": 0},
                {"y_center": 200.0, "sp": 6.0, "x0": 24.0, "x1": 232.0, "system_id": 1},
                {"y_center": 250.0, "sp": 6.0, "x0": 24.0, "x1": 232.0, "system_id": 1},
            ]
            for st in staves:
                for k in range(-2, 3):
                    ry = int(round(st["y_center"] + k * st["sp"]))
                    arr[ry, 24:232, :] = 20
            img_path = td_path / "page_aug.png"
            Image.fromarray(arr, mode="RGB").save(img_path)

            manifest = td_path / "manifest.jsonl"
            row = {"image": "page_aug.png", "staves": staves}
            manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

            ds1 = su.StaffUNetDataset(
                manifest, target_h=128, target_w=96, in_channels=3, out_stride=2, augment=True, seed=123
            )
            ds2 = su.StaffUNetDataset(
                manifest, target_h=128, target_w=96, in_channels=3, out_stride=2, augment=True, seed=123
            )

            # Exact determinism for same (seed, epoch, idx).
            x1, y1 = ds1[0]
            x2, y2 = ds2[0]
            self.assertTrue(torch.equal(x1, x2))
            self.assertTrue(torch.equal(y1, y2))

            # Advancing epoch changes the augmentation output.
            ds1.set_epoch(1)
            x1_e1, _ = ds1[0]
            self.assertFalse(torch.equal(x1, x1_e1))

            # Stress-test 12 epochs: valid shapes, finite values, RGB [0, 1], non-empty targets.
            for ep in range(12):
                ds1.set_epoch(ep)
                xb, yb = ds1[0]
                self.assertEqual(tuple(xb.shape), (3, 128, 96))
                self.assertEqual(tuple(yb.shape), (4, 64, 48))
                self.assertTrue(torch.isfinite(xb).all())
                self.assertTrue(torch.isfinite(yb).all())
                self.assertGreaterEqual(float(xb.min()), 0.0)
                self.assertLessEqual(float(xb.max()), 1.0)
                self.assertGreater(float(yb[1].sum()), 0.0)

            # DataLoader batching + model forward/backward.
            loader = torch.utils.data.DataLoader(ds1, batch_size=2, shuffle=False)
            bx, by = next(iter(loader))
            self.assertEqual(tuple(bx.shape), (2, 3, 128, 96))
            self.assertEqual(tuple(by.shape), (2, 4, 64, 48))
            model = su.StaffUNet2D(in_channels=3, base_channels=16, out_stride=2)
            loss, _ = su.StaffUNetLoss()(model(bx), by)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()


if __name__ == "__main__":
    unittest.main()
