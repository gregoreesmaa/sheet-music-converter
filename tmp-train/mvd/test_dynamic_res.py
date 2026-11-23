"""Unit tests for tmp-train/mvd/dynamic_res.py and dynamic-res Stage-1 hooks."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

MVD_DIR = Path(__file__).resolve().parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import dynamic_res as dr  # noqa: E402
import eval_sys_gate as esg  # noqa: E402
import staff_unet2d as su  # noqa: E402


class TestDynamicGeometryAndPreprocess(unittest.TestCase):
    def test_compute_dynamic_hw_and_preprocess_all_aspect_ratios(self):
        budget = 1536 * 1152
        max_side = 2048
        stride = 32
        cases = [
            ("portrait", 2480, 3508),
            ("landscape_hw", 1191, 835),
            ("spread_2to1", 4000, 2000),
            ("tight_crop_3to1", 1800, 600),
        ]
        for name, ow, oh in cases:
            with self.subTest(case=name, orig_w=ow, orig_h=oh):
                meta = dr.compute_dynamic_hw(
                    ow, oh, pixel_budget=budget, max_side=max_side, stride=stride
                )
                sw, sh = meta["scaled_w"], meta["scaled_h"]
                iw, ih = meta["in_w"], meta["in_h"]

                # Isotropic scaling: sx == sy up to integer pixel rounding.
                sx = sw / float(ow)
                sy = sh / float(oh)
                self.assertAlmostEqual(
                    sx,
                    sy,
                    delta=max(1.0 / ow, 1.0 / oh) + 1e-6,
                    msg=f"{name}: non-isotropic sx={sx:.6f} vs sy={sy:.6f}",
                )
                self.assertAlmostEqual(sx, meta["scale"], delta=1.0 / ow + 1e-6)
                self.assertAlmostEqual(sy, meta["scale"], delta=1.0 / oh + 1e-6)

                # Stride-32 multiples on padded network input.
                self.assertEqual(iw % stride, 0)
                self.assertEqual(ih % stride, 0)
                self.assertGreaterEqual(iw, sw)
                self.assertGreaterEqual(ih, sh)
                self.assertLess(iw - sw, stride)
                self.assertLess(ih - sh, stride)

                # Pre-pad pixel budget and max_side constraints.
                self.assertLessEqual(sw * sh, budget)
                self.assertLessEqual(max(sw, sh), max_side)

        # Verify preprocess_page_dynamic + border-color padding + unpad_probs_dynamic
        # on a warm parchment landscape image (1191x835) and tight crop (1800x600).
        for ow, oh in ((1191, 835), (1800, 600)):
            parchment = np.full((oh, ow, 3), [240, 230, 205], dtype=np.uint8)
            parchment[oh // 2 - 2 : oh // 2 + 2, 100 : ow - 100] = 20
            img = Image.fromarray(parchment, mode="RGB")

            ten, meta = dr.preprocess_page_dynamic(img, pixel_budget=budget, max_side=max_side)
            self.assertEqual(tuple(ten.shape), (3, meta["in_h"], meta["in_w"]))
            self.assertEqual(ten.dtype, torch.float32)
            self.assertGreaterEqual(float(ten.min()), 0.0)
            self.assertLessEqual(float(ten.max()), 1.0)
            # Padded bottom-right corner must match parchment border color, not black 0.
            if meta["in_h"] > meta["scaled_h"] or meta["in_w"] > meta["scaled_w"]:
                corner = ten[:, -1, -1].numpy()
                expected = np.array([240, 230, 205], dtype=np.float32) / 255.0
                np.testing.assert_allclose(corner, expected, atol=0.02)

            # Unpad 1/2-res predictions back to valid scaled region.
            fake_probs = torch.zeros(1, 4, meta["in_h"] // 2, meta["in_w"] // 2)
            unpadded = dr.unpad_probs_dynamic(fake_probs, meta, out_stride=2)
            self.assertEqual(
                tuple(unpadded.shape),
                (1, 4, int(round(meta["scaled_h"] / 2.0)), int(round(meta["scaled_w"] / 2.0))),
            )


class TestLogHmScaleRoundTripAndGutter(unittest.TestCase):
    def test_sp_log_hm_transforms(self):
        for sp_orig, scale, stride in ((14.5, 0.5, 2), (22.0, 1.33, 2), (9.2, 0.75, 4)):
            log_v = dr.sp_orig_to_log_hm(sp_orig, scale, out_stride=stride)
            rec = dr.log_hm_to_sp_orig(log_v, scale, out_stride=stride)
            self.assertAlmostEqual(rec, sp_orig, places=6)

        arr = np.array([8.0, 12.0, 20.0], dtype=np.float64)
        rec_arr = dr.log_hm_to_sp_orig(dr.sp_orig_to_log_hm(arr, 0.6, 2), 0.6, 2)
        np.testing.assert_allclose(rec_arr, arr, rtol=1e-6, atol=1e-6)

    def test_log_hm_rasterize_extract_roundtrip_portrait_and_landscape(self):
        cases = [
            ("portrait", 2480, 3508, 22.0),
            ("landscape", 1191, 835, 11.5),
        ]
        for name, ow, oh, sp_orig in cases:
            with self.subTest(case=name):
                meta = dr.compute_dynamic_hw(ow, oh, pixel_budget=1536 * 1152, max_side=2048)
                scale = meta["scale"]
                out_stride = 2
                hp = int(round(meta["scaled_h"] / float(out_stride)))
                wp = int(round(meta["scaled_w"] / float(out_stride)))

                staves_spec = [
                    {
                        "y_center": 0.30 * oh,
                        "sp": sp_orig,
                        "x0": 0.10 * ow,
                        "x1": 0.90 * ow,
                        "system_id": 0,
                    },
                    {
                        "y_center": 0.65 * oh,
                        "sp": sp_orig,
                        "x0": 0.10 * ow,
                        "x1": 0.90 * ow,
                        "system_id": 1,
                    },
                ]
                tgt = su.rasterize_staff_targets(
                    staves_spec,
                    w=ow,
                    h=oh,
                    out_h=hp,
                    out_w=wp,
                    sp_mode="log_hm",
                    iso_scale=scale,
                    out_stride=out_stride,
                )
                systems = su.extract_curved_systems(
                    tgt,
                    orig_w=ow,
                    orig_h=oh,
                    sp_mode="log_hm",
                    iso_scale=scale,
                    out_stride=out_stride,
                )
                self.assertEqual(len(systems), 2)
                for sys_obj in systems:
                    self.assertEqual(len(sys_obj.staves), 1)
                    sp_rec = sys_obj.staves[0].sp_median
                    self.assertLess(
                        abs(sp_rec - sp_orig),
                        0.2,
                        f"{name}: recovered sp {sp_rec:.3f} vs {sp_orig:.3f}",
                    )

    def test_log_hm_loss_and_predict_probs(self):
        model = su.StaffUNet2D(in_channels=3, base_channels=16, out_stride=2)
        crit = su.StaffUNetLoss(w_sp=1.0, sp_mode="log_hm")
        x = torch.randn(2, 3, 128, 160)
        tgt = torch.zeros(2, 4, 64, 80)
        tgt[:, 0, 20:28, 8:72] = 1.0
        tgt[:, 1, 24, 8:72] = 1.0
        tgt[:, 2, 18:30, 8:72] = 1.0
        tgt[:, 3, 18:30, 8:72] = math.log(4.5)

        preds = model(x)
        loss, metrics = crit(preds, tgt)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["loss_sp"], 0.0)
        loss.backward()

        with torch.no_grad():
            probs_log = model.predict_probs(x, sp_mode="log_hm")
            raw = model(x)
        # In log_hm mode, ch3 must pass raw logits untouched (no softplus).
        self.assertTrue(torch.equal(probs_log[:, 3:4], raw[:, 3:4]))

    def test_double_page_2to1_gutter_does_not_merge(self):
        """A 2:1 book spread with >18% central gutter must stay two separate staves, not bridge."""
        ow, oh = 4000, 2000
        out_h, out_w = 500, 1000
        sp = 24.0
        # Left page [200..1550], right page [2450..3800] -> central gutter = 900 px = 22.5% of orig_w (> 18%).
        staves_spec = [
            {"y_center": 600.0, "sp": sp, "x0": 200.0, "x1": 1550.0, "system_id": 0},
            {"y_center": 600.0, "sp": sp, "x0": 2450.0, "x1": 3800.0, "system_id": 1},
        ]
        tgt = su.rasterize_staff_targets(staves_spec, w=ow, h=oh, out_h=out_h, out_w=out_w)
        systems = su.extract_curved_systems(tgt, orig_w=ow, orig_h=oh)
        total_staves = sum(len(s.staves) for s in systems)
        self.assertEqual(
            total_staves,
            2,
            f"2:1 double-page gutter should not merge left and right staves (got {total_staves})",
        )
        all_staves = sorted([st for s in systems for st in s.staves], key=lambda s: s.x_start)
        self.assertLess(all_staves[0].x_end, 1700.0)
        self.assertGreater(all_staves[1].x_start, 2300.0)


class TestBucketSamplerAndDynamicTraining(unittest.TestCase):
    def test_aspect_bucket_batch_sampler_determinism_and_uniform_batches(self):
        sizes = [
            (2480, 3508),  # portrait
            (2400, 3400),  # portrait
            (2500, 3550),  # portrait
            (1191, 835),   # landscape
            (1200, 850),   # landscape
            (1400, 1400),  # square
            (1350, 1360),  # square
        ]
        s1 = dr.AspectBucketBatchSampler(sizes, batch_size=2, seed=77, shuffle=True)
        s2 = dr.AspectBucketBatchSampler(sizes, batch_size=2, seed=77, shuffle=True)

        b1_e0 = list(s1)
        b2_e0 = list(s2)
        self.assertEqual(b1_e0, b2_e0)
        self.assertEqual(len(b1_e0), len(s1))

        # Every batch must be uniform in aspect bucket.
        for batch in b1_e0:
            buckets_in_batch = {s1.bucket_for_index(i) for i in batch}
            self.assertEqual(len(buckets_in_batch), 1)

        # Every dataset index appears exactly once per epoch.
        flat = sorted(i for batch in b1_e0 for i in batch)
        self.assertEqual(flat, list(range(len(sizes))))

        # Different epoch produces a different batch ordering.
        s1.set_epoch(1)
        b1_e1 = list(s1)
        self.assertNotEqual(b1_e0, b1_e1)

    def test_dynamic_res_one_epoch_train_and_init_from_warm_start(self):
        small_buckets = [(128, 96), (96, 128)]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # Create 2 portrait + 2 landscape pages.
            rows = []
            specs = [
                ("p0.png", 180, 240, 100.0, 6.0),
                ("p1.png", 190, 250, 120.0, 6.0),
                ("l0.png", 260, 180, 90.0, 5.5),
                ("l1.png", 250, 170, 85.0, 5.5),
            ]
            for fname, w, h, yc, sp in specs:
                arr = np.full((h, w, 3), 246, dtype=np.uint8)
                for k in range(-2, 3):
                    ry = int(round(yc + k * sp))
                    arr[ry, 20 : w - 20, :] = 25
                Image.fromarray(arr, mode="RGB").save(td_path / fname)
                rows.append(
                    {
                        "image": fname,
                        "staves": [{"y_center": yc, "sp": sp, "x0": 20.0, "x1": float(w - 20), "system_id": 0}],
                    }
                )
            manifest = td_path / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            # 1. Train a fixed-res rel_h checkpoint to serve as --init-from source.
            v2_ckpt = su.train_staff_unet2d(
                manifest=manifest,
                out_dir=td_path / "v2_rel_h",
                epochs=1,
                batch_size=2,
                target_h=128,
                target_w=96,
                base_channels=16,
                device="cpu",
                augment=False,
            )
            self.assertTrue(v2_ckpt.exists())

            # 2. Warm-start a dynamic_res=True, sp_mode="log_hm" 1-epoch CPU run from v2_ckpt.
            v3_ckpt = su.train_staff_unet2d(
                manifest=manifest,
                out_dir=td_path / "v3_dyn_log_hm",
                epochs=1,
                batch_size=2,
                base_channels=16,
                device="cpu",
                augment=True,
                dynamic_res=True,
                pixel_budget=128 * 96,
                max_side=160,
                sp_mode="log_hm",
                init_from=v2_ckpt,
                buckets=small_buckets,
            )
            self.assertTrue(v3_ckpt.exists())

            # 3. Verify eval_sys_gate.load_staff_unet + extract_page_curved_systems
            #    loads dynamic metadata and runs cleanly on both portrait and landscape.
            net, meta = esg.load_staff_unet(v3_ckpt, device="cpu")
            self.assertTrue(meta["dynamic_res"])
            self.assertEqual(meta["sp_mode"], "log_hm")
            self.assertEqual(meta["pixel_budget"], 128 * 96)
            self.assertEqual(meta["max_side"], 160)

            for fname in ("p0.png", "l0.png"):
                res = esg.extract_page_curved_systems(net, td_path / fname, device="cpu", meta=meta)
                self.assertIsInstance(res, list)

            # Also test zero-retrain force_dynamic=True on the legacy v2 rel_h checkpoint.
            net_v2, meta_v2 = esg.load_staff_unet(v2_ckpt, device="cpu")
            self.assertFalse(meta_v2["dynamic_res"])
            self.assertEqual(meta_v2["sp_mode"], "rel_h")
            meta_v2_small = dict(meta_v2, pixel_budget=128 * 96, max_side=160)
            res_force = esg.extract_page_curved_systems(
                net_v2, td_path / "l0.png", device="cpu", meta=meta_v2_small, force_dynamic=True
            )
            self.assertIsInstance(res_force, list)


class TestDynamicOnnxExport(unittest.TestCase):
    def test_export_staff_unet_dynamic_onnx_portrait_and_landscape(self):
        import onnx

        model = su.StaffUNet2D(in_channels=3, base_channels=16, out_stride=2).eval()
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "staff_unet_dyn.onnx"
            out = dr.export_staff_unet_dynamic_onnx(model, onnx_path, opset=17)
            self.assertEqual(out, onnx_path)
            self.assertTrue(onnx_path.exists())

            om = onnx.load(str(onnx_path))
            onnx.checker.check_model(om)

            try:
                import onnxruntime as ort
            except ImportError:
                return

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            for shape in ((1, 3, 256, 192), (1, 3, 192, 256)):
                with self.subTest(shape=shape):
                    x_np = np.random.default_rng(123).standard_normal(shape).astype(np.float32)
                    (ort_out,) = sess.run(["logits"], {"x": x_np})
                    with torch.no_grad():
                        pt_out = model(torch.from_numpy(x_np)).numpy()
                    self.assertEqual(ort_out.shape, (shape[0], 4, shape[2] // 2, shape[3] // 2))
                    np.testing.assert_allclose(ort_out, pt_out, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
