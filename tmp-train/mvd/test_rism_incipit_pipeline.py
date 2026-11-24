"""Comprehensive unit & real-RISM CPU smoke tests for rism_incipit_pipeline.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

MVD_DIR = Path(__file__).resolve().parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import rism_incipit_pipeline as rip  # noqa: E402
import train as mvd_train  # noqa: E402


def _make_synthetic_strip(
    width: int = 900,
    height: int = 256,
    barline_xs: list[int] | None = None,
    target_sp: float = 10.0,
    canon_top: float = 12.0,
    slant_px: float = 0.0,
    noisy_sepia: bool = False,
    seed: int = 42,
    draw_notes: bool = True,
    blank: bool = False,
) -> Image.Image:
    """Create a canonical 256-px 4-slot ribbon strip with slot-0 5-line staff."""
    rng = np.random.default_rng(seed)
    if noisy_sepia:
        # Aged parchment background with subtle horizontal illumination gradient.
        x_grad = np.linspace(-12.0, 10.0, width, dtype=np.float32)[None, :, None]
        base = np.array([234.0, 224.0, 198.0], dtype=np.float32)[None, None, :] + x_grad
        arr = np.broadcast_to(base, (height, width, 3)).copy()
        arr += rng.normal(0.0, 6.5, size=arr.shape).astype(np.float32)
        arr[64:, :, :] = 255.0
    else:
        arr = np.full((height, width, 3), 255.0, dtype=np.float32)

    if blank:
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    ink_rgb = np.array([48.0, 36.0, 28.0], dtype=np.float32) if noisy_sepia else np.array([15.0, 15.0, 15.0], dtype=np.float32)

    # Draw 5 horizontal staff lines in slot 0.
    for k in range(5):
        y = int(round(canon_top + k * target_sp))
        arr[y : y + 1, 10 : width - 10, :] = ink_rgb

    y_top = int(round(canon_top))
    y_bot = int(round(canon_top + 4.0 * target_sp))
    y_mid = 0.5 * (y_top + y_bot)

    # Draw a synthetic clef/header glyph near x=20..42.
    arr[y_top - 4 : y_bot + 5, 22:27, :] = ink_rgb
    arr[y_top + 8 : y_top + 22, 27:35, :] = ink_rgb

    # Draw barlines (optionally slanted for handwritten realism).
    bar_xs = list(barline_xs or [])
    for bx in bar_xs:
        for y in range(y_top, y_bot + 1):
            dx = int(round(slant_px * (y - y_mid) / max(4.0 * target_sp, 1.0)))
            x = int(np.clip(bx + dx, 0, width - 2))
            arr[y, x : x + 2, :] = ink_rgb

    # Draw synthetic noteheads + short stems between barlines.
    if draw_notes and bar_xs:
        edges = [55] + bar_xs
        for a, b in zip(edges, edges[1:]):
            span = b - a
            if span < 30:
                continue
            for frac in (0.22, 0.48, 0.74):
                nx = int(round(a + frac * span))
                ny = int(rng.integers(y_top + 4, y_bot - 4))
                arr[ny - 3 : ny + 3, nx - 4 : nx + 4, :] = ink_rgb
                # Short stem (~2.2 * sp, NOT spanning all 4 inter-line spaces).
                s0 = max(y_top - 5, ny - int(round(2.2 * target_sp)))
                arr[s0:ny, nx + 3 : nx + 4, :] = ink_rgb

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


class TestShortTailFiltering(unittest.TestCase):
    def test_bar_rest_not_counted_as_barline(self):
        bek = "clef-G-2 key+0 time-4-4 voice-1 bar-rest bar voice-1 whole C 4 bar"
        stats = rip.parse_bekern_stats(bek)
        self.assertEqual(stats["n_bars"], 2)
        self.assertEqual(stats["n_tokens"], 11)
        self.assertTrue(stats["vocab_valid"])

    def test_rejects_short_bars_and_short_tokens(self):
        short_bars = {
            "bekern": "clef-G-2 key+0 time-4-4 voice-1 " + " ".join(["quarter C 4"] * 8) + " bar"
        }
        ok, reason = rip.filter_short_tail(short_bars, min_bars=2, min_tokens=25)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("short_tail_bars"))

        short_toks = {
            "bekern": "clef-G-2 key+0 time-4-4 voice-1 whole C 4 bar voice-1 whole D 4 bar"
        }
        ok2, reason2 = rip.filter_short_tail(short_toks, min_bars=2, min_tokens=25)
        self.assertFalse(ok2)
        self.assertTrue(reason2.startswith("short_tail_tokens"))

    def test_accepts_valid_incipit_and_rejects_unknown_vocab(self):
        valid_bek = (
            "clef-G-2 key-2 time-4-4 "
            "voice-1 quarter D 5 quarter E 5 a-1 quarter F 5 quarter G 5 bar "
            "voice-1 quarter A 5 quarter G 5 quarter F 5 quarter E 5 a-1 bar"
        )
        ok, reason = rip.filter_short_tail({"bekern": valid_bek}, min_bars=2, min_tokens=25)
        self.assertTrue(ok)
        self.assertIsNone(reason)

        bad_vocab = {"bekern": valid_bek + " not_a_real_token_xyz"}
        ok_bad, reason_bad = rip.filter_short_tail(bad_vocab, min_bars=2, min_tokens=25)
        self.assertFalse(ok_bad)
        self.assertEqual(reason_bad, "unknown_bekern_tokens")


class TestEvalLeakageGate(unittest.TestCase):
    def test_detects_rism_id_iiif_sha_and_slug_overlaps(self):
        with tempfile.TemporaryDirectory() as td:
            eval_path = Path(td) / "eval_manifest.json"
            eval_doc = {
                "pieces": [
                    {
                        "slug": "hw-beethoven-test",
                        "rism_url": "https://rism.online/sources/452004437",
                        "iiif_manifest": "https://example.org/iiif/PPN1031115978/manifest",
                        "clean": {"sha256": "aaaa1111bbbb2222"},
                        "photo": {"sha256": "cccc3333dddd4444"},
                    }
                ]
            }
            eval_path.write_text(json.dumps(eval_doc), encoding="utf-8")

            clean_rec = [
                {
                    "sample_id": "CH-Bu_999999999_c01",
                    "rism_id": "999999999",
                    "source_url": "https://rism.online/sources/999999999",
                    "iiif_manifest_url": "https://example.org/iiif/OTHER/manifest",
                    "image_sha256": "eeee5555ffff6666",
                }
            ]
            # Should pass cleanly.
            rip.assert_no_eval_overlap(clean_rec, eval_manifests=[eval_path])

            # RISM ID collision.
            with self.assertRaises(AssertionError):
                rip.assert_no_eval_overlap(
                    [{"sample_id": "x1", "rism_id": "452004437"}],
                    eval_manifests=[eval_path],
                )
            # IIIF manifest collision.
            with self.assertRaises(AssertionError):
                rip.assert_no_eval_overlap(
                    [{"sample_id": "x2", "iiif_manifest_url": "https://example.org/iiif/PPN1031115978/manifest/"}],
                    eval_manifests=[eval_path],
                )
            # SHA-256 collision.
            with self.assertRaises(AssertionError):
                rip.assert_no_eval_overlap(
                    [{"sample_id": "x3", "image_sha256": "CCCC3333DDDD4444"}],
                    eval_manifests=[eval_path],
                )
            # Slug collision.
            with self.assertRaises(AssertionError):
                rip.assert_no_eval_overlap(
                    [{"sample_id": "hw-beethoven-test"}],
                    eval_manifests=[eval_path],
                )


class TestSyntheticBarlineCropAndMarginGate(unittest.TestCase):
    def test_clean_strip_barline_crop_accuracy(self):
        true_bars = [210, 390, 580, 760]
        strip = _make_synthetic_strip(width=920, barline_xs=true_bars, noisy_sepia=False)
        res = rip.crop_and_gate_opening_strip(
            strip, n_bars=3, n_tokens=55, key_fifths=1, target_sp=10.0, pad_right_sp=1.5
        )
        self.assertEqual(res["status"], "accepted", msg=f"unexpected reject: {res}")
        self.assertTrue(res["exact_barline_match"])
        # 3rd barline is at x=580 -> crop_x_end should be ~580 + 15 = 595.
        self.assertAlmostEqual(res["crop_x_end"], 595, delta=3)
        self.assertEqual(res["cropped_img"].size, (res["crop_x_end"], 256))
        self.assertGreaterEqual(res["confidence"], 0.70)

    def test_noisy_slanted_sepia_strip_barline_crop_accuracy(self):
        true_bars = [200, 370, 550, 730]
        strip = _make_synthetic_strip(
            width=900,
            barline_xs=true_bars,
            slant_px=4.0,
            noisy_sepia=True,
            seed=123,
        )
        res = rip.crop_and_gate_opening_strip(
            strip, n_bars=3, n_tokens=50, key_fifths=-2, target_sp=10.0, pad_right_sp=1.5
        )
        self.assertEqual(res["status"], "accepted", msg=f"unexpected reject: {res}")
        self.assertAlmostEqual(res["crop_x_end"], 565, delta=5)
        self.assertGreaterEqual(res["confidence"], 0.65)

    def test_margin_gate_rejects_blank_strip(self):
        strip = _make_synthetic_strip(width=800, blank=True)
        res = rip.crop_and_gate_opening_strip(strip, n_bars=3, n_tokens=50)
        self.assertEqual(res["status"], "rejected")
        self.assertEqual(res["reason"], "blank_staff")
        self.assertIsNone(res["cropped_img"])

    def test_margin_gate_rejects_insufficient_barlines(self):
        # Only 1 barline present when 4 bars are expected -> system-wrap / ambiguous mismatch.
        strip = _make_synthetic_strip(width=800, barline_xs=[260], noisy_sepia=False)
        res = rip.crop_and_gate_opening_strip(strip, n_bars=4, n_tokens=65)
        self.assertEqual(res["status"], "rejected")
        self.assertEqual(res["reason"], "insufficient_barlines")

    def test_margin_gate_rejects_all_black_border_strip(self):
        arr = np.full((256, 800, 3), 15, dtype=np.uint8)
        arr[64:, :, :] = 255
        strip = Image.fromarray(arr, mode="RGB")
        res = rip.crop_and_gate_opening_strip(strip, n_bars=3, n_tokens=50)
        self.assertEqual(res["status"], "rejected")
        self.assertIn(res["reason"], ("blank_staff", "excessive_ink_or_border", "insufficient_barlines"))

    def test_margin_gate_rejects_bad_token_density(self):
        true_bars = [210, 390, 580]
        strip = _make_synthetic_strip(width=900, barline_xs=true_bars)
        # 500 tokens crammed into 595 px -> px_per_tok < 2.0 -> reject.
        res = rip.crop_and_gate_opening_strip(strip, n_bars=3, n_tokens=500)
        self.assertEqual(res["status"], "rejected")
        self.assertEqual(res["reason"], "bad_token_density")


class TestRealRismEndToEndSmoke(unittest.TestCase):
    RISM_INTL_DIR = Path(r"D:\tmp\rism-pdm-hw-intl")
    STAFF_UNET_CKPT = Path(r"D:\tmp\mvd-runs\staff-unet2d-v3-dynres\staff_unet2d.pt")

    @unittest.skipUnless(
        RISM_INTL_DIR.is_dir() and STAFF_UNET_CKPT.is_file(),
        "Real RISM intl dataset or StaffUNet2D v3 checkpoint not present on disk",
    )
    def test_real_rism_10_pages_cpu_smoke_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="rism_smoke_") as td:
            out_dir = Path(td)
            summary = rip.run_pipeline(
                rism_dirs=[self.RISM_INTL_DIR],
                staff_unet_ckpt=self.STAFF_UNET_CKPT,
                out_dir=out_dir,
                limit=10,
                min_bars=2,
                min_tokens=25,
                device="cpu",
                resume=False,
            )

            print(
                "\n[RISM 10-page CPU smoke]",
                json.dumps(summary, indent=2),
                flush=True,
            )

            self.assertEqual(summary["total_records"], 10)
            self.assertEqual(summary["processed_this_run"], 10)
            self.assertEqual(summary["accepted"] + summary["rejected"], 10)
            self.assertEqual(summary["stage1_pages"], 10)
            self.assertGreater(summary["total_unlabeled_strips"], 10)

            # 1. Verify stage1_pages.txt
            stage1_lines = [
                ln.strip()
                for ln in (out_dir / "stage1_pages.txt").read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            self.assertEqual(len(stage1_lines), 10)
            for p in stage1_lines:
                self.assertTrue(Path(p).is_file(), f"missing stage1 page: {p}")

            # 2. Verify unlabeled_strips/ (ALL unwrapped systems across the 10 pages)
            unlabeled_pngs = sorted((out_dir / "unlabeled_strips").glob("*.png"))
            self.assertEqual(len(unlabeled_pngs), summary["total_unlabeled_strips"])
            for up in unlabeled_pngs[:5]:
                with Image.open(up) as im:
                    self.assertEqual(im.height, 256)
                    self.assertEqual(im.mode, "RGB")

            # 3. Verify renders/ + targets/ + manifest.jsonl are train.py-compatible
            manifest_rows = [
                json.loads(ln)
                for ln in (out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            self.assertEqual(len(manifest_rows), 10)
            accepted_rows = [r for r in manifest_rows if r["status"] == "accepted"]
            self.assertEqual(len(accepted_rows), summary["accepted"])
            self.assertGreaterEqual(summary["accepted"], 1, "expected at least 1 accepted real RISM strip")

            hw_pairs = mvd_train.load_real_hw_pairs(str(out_dir), include_degraded=False)
            self.assertEqual(len(hw_pairs), summary["accepted"])
            for img_p, bek_p, _stem in hw_pairs:
                with Image.open(img_p) as im:
                    self.assertEqual(im.height, 256)
                    self.assertGreater(im.width, 60)
                toks = Path(bek_p).read_text(encoding="utf-8").strip().split()
                self.assertGreaterEqual(len(toks), 25)

            # 4. Verify --resume skips all 10 already-processed pages cleanly
            summary_resume = rip.run_pipeline(
                rism_dirs=[self.RISM_INTL_DIR],
                staff_unet_ckpt=self.STAFF_UNET_CKPT,
                out_dir=out_dir,
                limit=10,
                min_bars=2,
                min_tokens=25,
                device="cpu",
                resume=True,
            )
            self.assertEqual(summary_resume["processed_this_run"], 0)
            self.assertEqual(summary_resume["resumed_skipped"], 10)
            self.assertEqual(summary_resume["accepted"], summary["accepted"])


if __name__ == "__main__":
    unittest.main()
