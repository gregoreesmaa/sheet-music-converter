#!/usr/bin/env python3
"""Unit tests for tmp-train/mvd/harvest_unlabeled_10k.py."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harvest_unlabeled_10k as h


def make_synthetic_music_page(w: int = 1200, h_px: int = 1600, bg: int = 242) -> Image.Image:
    """Create a realistic synthetic sheet-music page with 6 systems of 5-line staves + notes."""
    im = Image.new("RGB", (w, h_px), (bg, bg, bg - 5))
    draw = ImageDraw.Draw(im)
    margin_x = 90
    for sys_idx in range(6):
        y0 = 140 + sys_idx * 230
        for line_i in range(5):
            y = y0 + line_i * 14
            draw.line([(margin_x, y), (w - margin_x, y)], fill=(25, 25, 25), width=2)
        # Draw barlines and noteheads/stems
        for bx in range(margin_x, w - margin_x + 1, 250):
            draw.line([(bx, y0), (bx, y0 + 56)], fill=(20, 20, 20), width=2)
        for nx in range(margin_x + 35, w - margin_x - 30, 55):
            ny = y0 + ((nx // 55) % 5) * 14
            draw.ellipse([(nx - 7, ny - 5), (nx + 7, ny + 5)], fill=(15, 15, 15))
            draw.line([(nx + 7, ny), (nx + 7, ny - 36)], fill=(15, 15, 15), width=2)
    return im


def make_blank_page(w: int = 1200, h_px: int = 1600) -> Image.Image:
    return Image.new("RGB", (w, h_px), (250, 250, 248))


def make_solid_dark_cover(w: int = 1200, h_px: int = 1600) -> Image.Image:
    im = Image.new("RGB", (w, h_px), (45, 38, 52))
    draw = ImageDraw.Draw(im)
    draw.rectangle([(150, 200), (w - 150, 500)], outline=(220, 190, 120), width=6)
    return im


def make_text_only_page(w: int = 800, h_px: int = 1100) -> Image.Image:
    """Simulated prose/preface page with printed text lines (no musical notation)."""
    im = Image.new("RGB", (w, h_px), (245, 242, 235))
    draw = ImageDraw.Draw(im)
    for y in range(80, h_px - 80, 28):
        draw.text(
            (70, y),
            "Chapter 1: Preface and Historical Notes on the Composition of Literature",
            fill=(25, 25, 25),
        )
    return im


class TestMusicPageFilter(unittest.TestCase):
    def test_accepts_real_suite31_clean_and_photo_pages(self):
        repo = Path(__file__).resolve().parents[2]
        clean_imgs = sorted((repo / "data-gen" / "score-corpus" / "suite31" / "clean").glob("*.png"))[:3]
        photo_imgs = sorted((repo / "data-gen" / "score-corpus" / "suite31" / "photo").glob("*.png"))[:3]
        self.assertTrue(clean_imgs, "suite31 clean samples missing")
        self.assertTrue(photo_imgs, "suite31 photo samples missing")
        for p in clean_imgs + photo_imgs:
            with Image.open(p) as im:
                ok, stats = h.is_music_page(im)
            self.assertTrue(ok, f"Expected real printed/photo page {p.name} to pass CLIP gate, stats={stats}")
            self.assertEqual(stats["page_class"], "music_sheet")
            self.assertGreaterEqual(stats["p_music"], h.CLIP_MUSIC_THRESHOLD)

    def test_accepts_real_handwritten_sepia_manuscripts(self):
        repo = Path(__file__).resolve().parents[2]
        hw_imgs = sorted(
            (repo / "data-gen" / "score-corpus" / "handwritten-500" / "handwritten" / "beethoven-op127").glob("*")
        )[2:5]
        hw_imgs += sorted(
            (repo / "data-gen" / "score-corpus" / "handwritten-500" / "handwritten" / "mozart-k428").glob("*")
        )[2:5]
        self.assertTrue(hw_imgs, "handwritten-500 samples missing")
        for p in hw_imgs:
            with Image.open(p) as im:
                ok, stats = h.is_music_page(im)
            self.assertTrue(ok, f"Expected handwritten sepia page {p.name} to pass CLIP gate, stats={stats}")
            self.assertEqual(stats["page_class"], "music_sheet")
            self.assertGreaterEqual(stats["p_music"], 0.95)

    def test_rejects_blank_page(self):
        ok, stats = h.is_music_page(make_blank_page())
        self.assertFalse(ok, f"Expected blank page to be rejected, stats={stats}")
        self.assertEqual(stats["page_class"], "cover_or_blank")

    def test_rejects_solid_cover(self):
        ok, stats = h.is_music_page(make_solid_dark_cover())
        self.assertFalse(ok, f"Expected solid cover to be rejected, stats={stats}")
        self.assertEqual(stats["page_class"], "cover_or_blank")

    def test_rejects_text_only_page(self):
        ok, stats = h.is_music_page(make_text_only_page())
        self.assertFalse(ok, f"Expected text-only page to be rejected, stats={stats}")
        self.assertEqual(stats["page_class"], "text_page")


class TestEvalLeakageFilter(unittest.TestCase):
    def test_blocks_all_five_suite31_hw_works(self):
        for work in (
            "beethoven-op135",
            "mozart-k464",
            "mozart-k387",
            "mozart-k421",
            "schubert-d810",
        ):
            excl, reason = h.is_eval_excluded(f"String Quartet {work}", work)
            self.assertTrue(excl, f"Must block held-out work {work}")
            self.assertTrue(reason)

    def test_blocks_natural_language_held_out_titles(self):
        cases = [
            ("Beethoven: String Quartet No. 16 in F major, Op. 135", "imslp-b135"),
            ("Mozart String Quartet K. 464 in A major", "imslp-m464"),
            ("Schubert - Death and the Maiden Quartet D.810", "imslp-s810"),
            ("Scott Joplin - Maple Leaf Rag (1899)", "imslp-joplin"),
            ("Emmanuel Chabrier - Ballade des gros dindons", "imslp-chab"),
            ("Beethoven - Grosse Fuge Op. 133", "imslp-gf"),
        ]
        for title, ident in cases:
            excl, _ = h.is_eval_excluded(title, ident)
            self.assertTrue(excl, f"Must block eval title: {title}")

    def test_allows_unrelated_public_domain_works(self):
        cases = [
            ("Frederic Chopin - Nocturne in E-flat major, Op. 9 No. 2", "imslp-chopin-op9"),
            ("Claude Debussy - Clair de Lune (Suite bergamasque)", "imslp-debussy-clair"),
            ("J.S. Bach - Goldberg Variations, BWV 988", "imslp-bach-988"),
            ("Franz Liszt - Liebestraum No. 3", "imslp-liszt-3"),
        ]
        for title, ident in cases:
            excl, reason = h.is_eval_excluded(title, ident)
            self.assertFalse(excl, f"Should allow unrelated PD work {title}, got {reason}")


class TestManifestAndState(unittest.TestCase):
    def test_manifest_schema_and_audit(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            state = h.HarvestState(out_dir=out_dir, target=5, resume=False)
            im = make_synthetic_music_page()
            ok, stats = h.is_music_page(im)
            self.assertTrue(ok)
            jb, w, hp = h.resize_and_encode_jpeg(im, max_dim=1600)

            saved = state.try_save_image(
                jpeg_bytes=jb,
                width=w,
                height=hp,
                filename="ia_imslp_test_p0000.jpg",
                source="internet_archive",
                collection="ia_imslp",
                item_id="imslp-chopin-op9",
                title="Chopin Nocturne Op. 9",
                license_str="Public Domain (IMSLP / Internet Archive)",
                url="https://archive.org/download/imslp-chopin-op9/score.pdf",
                page_index=0,
                filter_stats=stats,
            )
            self.assertTrue(saved)
            # Duplicate bytes must be rejected
            saved_dup = state.try_save_image(
                jpeg_bytes=jb,
                width=w,
                height=hp,
                filename="ia_imslp_test_p0001.jpg",
                source="internet_archive",
                collection="ia_imslp",
                item_id="imslp-chopin-op9",
                title="Chopin Nocturne Op. 9",
                license_str="Public Domain (IMSLP / Internet Archive)",
                url="https://archive.org/download/imslp-chopin-op9/score.pdf",
                page_index=1,
                filter_stats=stats,
            )
            self.assertFalse(saved_dup)

            summary = state.finalize_and_audit(extra_slugs=set())
            self.assertEqual(summary["total_images"], 1)
            self.assertEqual(summary["leaked_eval"], 0)
            self.assertEqual(summary["rejected_dup"], 1)

            lines = (out_dir / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            required_keys = {
                "id",
                "image_path",
                "source",
                "collection",
                "item_id",
                "title",
                "license",
                "url",
                "page_index",
                "width",
                "height",
                "sha256",
                "page_class",
                "p_music",
                "filter_stats",
            }
            self.assertTrue(required_keys.issubset(rec.keys()))
            self.assertEqual(rec["page_class"], "music_sheet")
            self.assertGreaterEqual(rec["p_music"], h.CLIP_MUSIC_THRESHOLD)
            self.assertTrue(Path(rec["image_path"]).is_file())

    def test_labeled_item_detection_and_routing(self):
        files = [
            {"name": "score.pdf", "size": "123456"},
            {"name": "score.musicxml", "size": "45000"},
            {"name": "part.mxl", "size": "12000"},
            {"name": "choral.krn", "size": "8000"},
        ]
        found = h.find_labeled_files(files)
        self.assertEqual(found, ["score.musicxml", "part.mxl", "choral.krn"])
        self.assertEqual(h.find_labeled_files([{"name": "score.pdf"}]), [])

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            state = h.HarvestState(out_dir=out_dir, target=5, resume=False)
            state.record_labeled_find(
                source="internet_archive",
                collection="ia_imslp",
                item_id="test-labeled-01",
                title="Test Labeled Score",
                url="https://archive.org/details/test-labeled-01",
                labeled_files=found,
            )
            summary = state.finalize_and_audit(extra_slugs=set())
            self.assertEqual(summary["total_images"], 0)
            self.assertEqual(summary["labeled_finds_routed"], 1)
            lf_lines = (out_dir / "labeled_finds.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lf_lines), 1)
            lf_rec = json.loads(lf_lines[0])
            self.assertEqual(lf_rec["item_id"], "test-labeled-01")
            self.assertEqual(lf_rec["labeled_files"], found)


if __name__ == "__main__":
    unittest.main()
