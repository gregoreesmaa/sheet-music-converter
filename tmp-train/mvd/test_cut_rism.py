"""CPU unit tests for Geometry-Locked CUT RISM Style Transfer Pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import train_cut_rism as cut  # noqa: E402


def _make_synthetic_score_tensor(
    h: int,
    w: int,
    bg_rgb: tuple[float, float, float] = (0.98, 0.98, 0.98),
    ink_rgb: tuple[float, float, float] = (0.05, 0.05, 0.05),
    y_shift: int = 0,
) -> torch.Tensor:
    """Create a synthetic [3, H, W] sheet-music image with 5 staff lines and noteheads."""
    img = torch.empty((3, h, w), dtype=torch.float32)
    for c in range(3):
        img[c].fill_(bg_rgb[c])

    # 5 horizontal staff lines spaced 8 px apart
    base_y = h // 3 + y_shift
    for k in range(5):
        y = base_y + k * 8
        if 0 <= y < h:
            for c in range(3):
                img[c, max(0, y - 1) : min(h, y + 1), 8 : max(9, w - 8)] = ink_rgb[c]

    # Notehead blobs along the staff
    for nx in range(20, max(21, w - 20), 28):
        ny = base_y + ((nx // 28) % 5) * 8
        y0, y1 = max(0, ny - 3), min(h, ny + 4)
        x0, x1 = max(0, nx - 4), min(w, nx + 5)
        for c in range(3):
            img[c, y0:y1, x0:x1] = ink_rgb[c]

    return img.clamp(0.0, 1.0)


class TestCutRismPipeline(unittest.TestCase):
    def setUp(self) -> None:
        cut.seed_everything(1234)
        self.device = torch.device("cpu")

    def test_generator_param_count_and_arbitrary_shapes(self) -> None:
        # Default architecture should be ~11M params (11.38M for ngf=64, n_blocks=9)
        gen_full = cut.ResNetGenerator(ngf=64, n_blocks=9)
        n_params = sum(p.numel() for p in gen_full.parameters())
        self.assertGreaterEqual(n_params, 10_000_000)
        self.assertLessEqual(n_params, 13_000_000)

        # Tiny generator for fast CPU shape checks on odd / non-multiple-of-4 dims
        gen = cut.ResNetGenerator(ngf=16, n_blocks=2)
        for h, w in [(64, 64), (73, 101), (131, 95)]:
            x = torch.rand(1, 3, h, w)
            y = gen(x)
            self.assertEqual(tuple(y.shape), (1, 3, h, w))
            self.assertGreaterEqual(float(y.min().item()), 0.0)
            self.assertLessEqual(float(y.max().item()), 1.0)

    def test_geometry_lock_loss_and_vertical_drift(self) -> None:
        geom = cut.GeometryLockLoss(bg_kernel=21, sobel_weight=1.0)
        src = _make_synthetic_score_tensor(128, 160, y_shift=0).unsqueeze(0)
        recolored_same_geom = _make_synthetic_score_tensor(
            128, 160, bg_rgb=(0.86, 0.79, 0.66), ink_rgb=(0.20, 0.14, 0.09), y_shift=0
        ).unsqueeze(0)
        shifted_4px = _make_synthetic_score_tensor(128, 160, y_shift=4).unsqueeze(0)
        shifted_3px = _make_synthetic_score_tensor(128, 160, y_shift=3).unsqueeze(0)

        # Identical input -> ~0 loss, ~1.0 Sobel-Y correlation
        loss_id, stats_id = geom(src, src)
        self.assertLess(float(loss_id.item()), 1e-4)
        self.assertGreater(stats_id["sobel_y_corr"], 0.999)

        # Parchment tint + brown ink with exact geometry stays low loss & high corr
        loss_recolor, stats_recolor = geom(recolored_same_geom, src)
        self.assertGreater(stats_recolor["sobel_y_corr"], 0.99)

        # Vertical 4px shift causes large geometry lock penalty
        loss_shift, stats_shift = geom(shifted_4px, src)
        self.assertGreater(float(loss_shift.item()), float(loss_recolor.item()) + 0.5)
        self.assertLess(stats_shift["sobel_y_corr"], 0.5)

        # Drift estimator: ~0px for recolored unshifted, ~3px for 3px vertical shift
        drift_same = cut.measure_vertical_ink_drift_px(src, recolored_same_geom, bg_kernel=21)
        drift_3px = cut.measure_vertical_ink_drift_px(src, shifted_3px, bg_kernel=21)
        self.assertLess(drift_same, 0.25)
        self.assertAlmostEqual(drift_3px, 3.0, delta=0.5)

    def test_forward_backward_train_step_updates_all_modules(self) -> None:
        cfg = cut.CutRismConfig(
            ngf=16,
            ndf=16,
            n_blocks=2,
            nce_feat_dim=32,
            num_patches=16,
            geom_lock_weight=4.0,
            ink_bg_kernel=15,
        )
        gen = cut.ResNetGenerator(ngf=cfg.ngf, n_blocks=cfg.n_blocks)
        disc = cut.PatchGANDiscriminator(ndf=cfg.ndf, use_spectral_norm=cfg.use_spectral_norm)
        net_f = cut.PatchSampleMLP(gen.layer_channels(cfg.nce_layers), feat_dim=cfg.nce_feat_dim)
        nce_loss_fn = cut.PatchNCELoss(temperature=cfg.nce_temp)
        geom_loss_fn = cut.GeometryLockLoss(bg_kernel=cfg.ink_bg_kernel, sobel_weight=cfg.sobel_weight)

        opt_g = torch.optim.Adam(list(gen.parameters()) + list(net_f.parameters()), lr=1e-3)
        opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)

        g_before = next(gen.parameters()).detach().clone()
        f_before = next(net_f.parameters()).detach().clone()
        d_before = next(disc.parameters()).detach().clone()

        real_x = torch.stack([_make_synthetic_score_tensor(64, 64), _make_synthetic_score_tensor(64, 64)], dim=0)
        real_y = torch.stack(
            [
                _make_synthetic_score_tensor(64, 64, bg_rgb=(0.85, 0.78, 0.65), ink_rgb=(0.22, 0.15, 0.10)),
                _make_synthetic_score_tensor(64, 64, bg_rgb=(0.82, 0.75, 0.62), ink_rgb=(0.18, 0.12, 0.08)),
            ],
            dim=0,
        )

        metrics, fake_y = cut.train_step(
            gen=gen,
            disc=disc,
            net_f=net_f,
            nce_loss_fn=nce_loss_fn,
            geom_loss_fn=geom_loss_fn,
            opt_g=opt_g,
            opt_d=opt_d,
            real_x=real_x,
            real_y=real_y,
            cfg=cfg,
            use_bf16=False,
        )

        for k, v in metrics.items():
            self.assertTrue(np.isfinite(v), f"Non-finite metric {k}={v}")
        self.assertEqual(tuple(fake_y.shape), (2, 3, 64, 64))

        # Verify all three modules actually received gradient updates
        self.assertFalse(torch.allclose(g_before, next(gen.parameters())))
        self.assertFalse(torch.allclose(f_before, next(net_f.parameters())))
        self.assertFalse(torch.allclose(d_before, next(disc.parameters())))

    def test_checkpoint_save_load_parity(self) -> None:
        cfg = cut.CutRismConfig(ngf=16, ndf=16, n_blocks=2, nce_feat_dim=32, num_patches=16)
        gen = cut.ResNetGenerator(ngf=cfg.ngf, n_blocks=cfg.n_blocks)
        disc = cut.PatchGANDiscriminator(ndf=cfg.ndf)
        net_f = cut.PatchSampleMLP(gen.layer_channels(cfg.nce_layers), feat_dim=cfg.nce_feat_dim)
        opt_g = torch.optim.Adam(list(gen.parameters()) + list(net_f.parameters()), lr=1e-4)
        opt_d = torch.optim.Adam(disc.parameters(), lr=1e-4)

        x = _make_synthetic_score_tensor(77, 93).unsqueeze(0)
        gen.eval()
        with torch.no_grad():
            expected = gen(x)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "cut_rism_test.pt"
            cut.save_checkpoint(ckpt_path, gen, disc, net_f, opt_g, opt_d, cfg, step=7, best_score=0.123)
            loaded_gen, loaded_cfg, ckpt = cut.load_generator_from_checkpoint(ckpt_path, self.device)
            self.assertEqual(ckpt["step"], 7)
            self.assertEqual(loaded_cfg.ngf, 16)
            self.assertEqual(loaded_cfg.n_blocks, 2)
            with torch.no_grad():
                actual = loaded_gen(x)
            self.assertTrue(torch.allclose(expected, actual, atol=0.0, rtol=0.0))

    def test_end_to_end_train_and_translate_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_dir = root / "src"
            tgt_dir = root / "tgt"
            train_out = root / "train_out"
            trans_out = root / "trans_out"
            src_dir.mkdir()
            tgt_dir.mkdir()

            # Write synthetic source & target images with arbitrary non-square dims
            src_img_1 = src_dir / "page_small.png"
            src_img_2 = src_dir / "page_large.png"
            cut.tensor_to_pil(_make_synthetic_score_tensor(113, 149)).save(src_img_1)
            cut.tensor_to_pil(_make_synthetic_score_tensor(221, 187)).save(src_img_2)

            tgt_img_1 = tgt_dir / "rism_0.png"
            cut.tensor_to_pil(
                _make_synthetic_score_tensor(160, 160, bg_rgb=(0.84, 0.77, 0.64), ink_rgb=(0.20, 0.14, 0.09))
            ).save(tgt_img_1)

            # 1) Run --mode train via CLI main() on CPU
            rc = cut.main(
                [
                    "--mode", "train",
                    "--src-dir", str(src_dir),
                    "--tgt-dir", str(tgt_dir),
                    "--out-dir", str(train_out),
                    "--device", "cpu",
                    "--steps", "2",
                    "--batch-size", "1",
                    "--crop-size", "64",
                    "--ngf", "16",
                    "--ndf", "16",
                    "--n-blocks", "2",
                    "--nce-feat-dim", "32",
                    "--num-patches", "16",
                    "--save-every", "2",
                    "--preview-every", "2",
                    "--log-every", "1",
                ]
            )
            self.assertEqual(rc, 0)
            best_ckpt = train_out / "cut_rism_best.pt"
            self.assertTrue(best_ckpt.is_file())
            self.assertTrue((train_out / "cut_rism_latest.pt").is_file())
            self.assertTrue((train_out / "train_log.jsonl").is_file())
            self.assertTrue((train_out / "previews" / "step_000002.png").is_file())

            # 2) Write a JSONL manifest and run --mode translate (covers both direct and tiled paths)
            manifest_in = root / "in_manifest.jsonl"
            orig_records = [
                {
                    "image": str(src_img_1),
                    "bekerns": ["gt/page_small_s0.txt", "gt/page_small_s1.txt"],
                    "domain": "hw",
                    "n_systems": 2,
                },
                {
                    "image": str(src_img_2),
                    "bekerns": ["gt/page_large_s0.txt"],
                    "domain": "hw",
                    "n_systems": 1,
                },
            ]
            with open(manifest_in, "w", encoding="utf-8") as mf:
                for rec in orig_records:
                    mf.write(json.dumps(rec) + "\n")

            # Set --max-side 160 so page_small (113x149) uses direct full-res conv
            # and page_large (221x187) uses Hann-windowed tiled overlap-blend!
            rc_t = cut.main(
                [
                    "--mode", "translate",
                    "--checkpoint", str(best_ckpt),
                    "--manifest", str(manifest_in),
                    "--out-dir", str(trans_out),
                    "--device", "cpu",
                    "--max-side", "160",
                    "--tile-size", "128",
                    "--tile-overlap", "32",
                ]
            )
            self.assertEqual(rc_t, 0)

            manifest_out = trans_out / "manifest.jsonl"
            self.assertTrue(manifest_out.is_file())
            with open(manifest_out, "r", encoding="utf-8") as mf:
                out_records = [json.loads(line) for line in mf if line.strip()]
            self.assertEqual(len(out_records), 2)

            expected_sizes = [(149, 113), (187, 221)]  # PIL (W, H)
            for orig, out_r, exp_wh in zip(orig_records, out_records, expected_sizes):
                self.assertEqual(out_r["bekerns"], orig["bekerns"])
                self.assertEqual(out_r["domain"], orig["domain"])
                self.assertEqual(out_r["n_systems"], orig["n_systems"])
                self.assertEqual(out_r["style_transfer"], "cut_rism")
                out_img_path = Path(out_r["image"])
                self.assertTrue(out_img_path.is_file())
                with Image.open(out_img_path) as im:
                    self.assertEqual(im.size, exp_wh)


if __name__ == "__main__":
    unittest.main()
