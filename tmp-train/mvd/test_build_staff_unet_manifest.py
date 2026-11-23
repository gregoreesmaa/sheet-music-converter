"""Unit tests for tmp-train/mvd/build_staff_unet_manifest.py (CPU-only)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image
import torch

MVD_DIR = Path(__file__).resolve().parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import build_staff_unet_manifest as bsm  # noqa: E402
import staff_unet2d as su  # noqa: E402


def _make_symbox_page(
    symbox_dir: Path,
    render_dir: Path,
    stem: str,
    n_systems: int,
    staves_per_sys: int,
    w: int = 800,
    h: int = 1024,
    include_barlines: bool = True,
) -> tuple[Path, Path]:
    """Write a synthetic PNG + matching *-boxes.json into symbox_dir/render_dir."""
    render_dir.mkdir(parents=True, exist_ok=True)
    score_dir = symbox_dir / stem.split("-")[0]
    score_dir.mkdir(parents=True, exist_ok=True)
    png_path = render_dir / f"{stem}.png"
    Image.new("RGB", (w, h), (255, 255, 255)).save(png_path)

    sp = 10.0
    staves = []
    glyphs = []
    y = 60.0
    for _sys_idx in range(n_systems):
        sys_y0 = y
        for _s in range(staves_per_sys):
            lines = [y + k * sp for k in range(5)]
            staves.append({"lines": lines, "bbox": [60.0, lines[0], float(w - 60), lines[-1]]})
            y += 4.0 * sp + 32.0
        sys_y1 = staves[-1]["bbox"][3]
        if include_barlines and staves_per_sys >= 2:
            glyphs.append({"cls": "BarLine", "bbox": [60.0, sys_y0, 62.0, sys_y1]})
        elif not include_barlines and staves_per_sys >= 2:
            # Bracket only on bottom 2 staves to exercise big500-style gap fallback
            b_y0 = staves[-2]["bbox"][1]
            glyphs.append({"cls": "Bracket", "bbox": [52.0, b_y0, 58.0, sys_y1]})
        y += 90.0

    box_path = score_dir / f"{stem}-boxes.json"
    box_path.write_text(
        json.dumps({"page": 1, "png_size": [w, h], "staves": staves, "glyphs": glyphs}),
        encoding="utf-8",
    )
    return box_path, png_path


class TestBuildStaffUNetManifest(unittest.TestCase):
    def test_group_staves_into_systems_barlines_and_fallback(self):
        # 2 systems x 3 staves with BarLine spans
        staves = []
        y = 50.0
        for _ in range(2):
            for _ in range(3):
                staves.append({"lines": [y + 10 * k for k in range(5)], "bbox": [50, y, 700, y + 40]})
                y += 70.0
            y += 140.0
        glyphs = [
            {"cls": "BarLine", "bbox": [50, 50, 52, 230]},
            {"cls": "BarLine", "bbox": [50, 400, 52, 580]},
        ]
        sids = bsm.group_staves_into_systems(staves, glyphs)
        self.assertEqual(sids, [0, 0, 0, 1, 1, 1])

        # Fallback without BarLine (big500 voice+piano style): bimodal gaps still group cleanly
        glyphs_no_bar = [{"cls": "Bracket", "bbox": [45, 120, 49, 230]}]
        sids_fb = bsm.group_staves_into_systems(staves, glyphs_no_bar)
        self.assertEqual(sids_fb, [0, 0, 0, 1, 1, 1])

    def test_excludes_gate20_suite31_and_hw_leaks(self):
        policy = bsm.load_exclusion_policy()
        with tempfile.TemporaryDirectory(prefix="t_sunet_excl_") as tmp:
            tmpd = Path(tmp)
            sdir, rdir = tmpd / "symbox", tmpd / "renders"
            # Clean training piece
            _make_symbox_page(sdir, rdir, "trainpiece01-1", n_systems=2, staves_per_sys=2)
            # Gate20 score leak (ds0072-3 is in gate20 exclusion.json)
            _make_symbox_page(sdir, rdir, "ds0072-3", n_systems=2, staves_per_sys=2)
            # Suite31 slug leak (with -1 page suffix)
            sample_s31 = sorted(policy.suite31_slugs)[0]
            _make_symbox_page(sdir, rdir, f"{sample_s31}-1", n_systems=2, staves_per_sys=1)
            # Suite31_hw held-out work leak
            _make_symbox_page(sdir, rdir, "beethoven-op135-1", n_systems=2, staves_per_sys=4)

            pairs, stats = bsm.collect_candidate_pairs([("syn", sdir, rdir)], policy)
            stems = [p[2].stem for p in pairs]
            self.assertEqual(stems, ["trainpiece01-1"])
            self.assertEqual(stats["excluded_eval"], 3)

    def test_oversampling_and_dataset_loading_cpu(self):
        with tempfile.TemporaryDirectory(prefix="t_sunet_ds_") as tmp:
            tmpd = Path(tmp)
            sdir, rdir = tmpd / "symbox", tmpd / "renders"
            _make_symbox_page(sdir, rdir, "mono01-1", n_systems=3, staves_per_sys=1)
            _make_symbox_page(sdir, rdir, "piano01-1", n_systems=2, staves_per_sys=2)
            _make_symbox_page(sdir, rdir, "choral01-1", n_systems=1, staves_per_sys=6, h=1400)

            out_manifest = tmpd / "manifest.jsonl"
            summary = bsm.build_manifest(
                out_path=out_manifest,
                sources=[("syn", sdir, rdir)],
                target_pages=20,
                oversample_1staff=3,
                oversample_dense=3,
                seed=7,
                workers=2,
            )
            self.assertEqual(summary["audit"]["leaked_eval"], 0)
            self.assertEqual(summary["audit"]["nc_count"], 0)
            # 1*3 (1-staff) + 1 (2-staff) + 1*3 (6-staff) = 7 rows
            self.assertEqual(summary["total_manifest_rows"], 7)
            self.assertEqual(summary["final_max_staves_per_sys"]["1"], 3)
            self.assertEqual(summary["final_max_staves_per_sys"]["6"], 3)

            # Verify StaffUNetDataset loads with augment=True on CPU
            ds = su.StaffUNetDataset(
                out_manifest, target_h=256, target_w=192, in_channels=3, out_stride=2, augment=True, seed=123
            )
            self.assertEqual(len(ds), 7)
            x, y = ds[0]
            self.assertEqual(tuple(x.shape), (3, 256, 192))
            self.assertEqual(tuple(y.shape), (4, 128, 96))
            self.assertTrue(torch.isfinite(x).all())
            self.assertTrue(torch.isfinite(y).all())
            self.assertGreater(float(y[0].sum()), 0.0)
            self.assertGreater(float(y[2].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
