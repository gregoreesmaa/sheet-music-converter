"""CPU-only TDD unit tests for T4.1 render_diff_verify.py."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = MVD_DIR.parents[1]
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

from page_prep import prep_page
from render_diff_verify import (
    bekern_to_strip_png,
    default_assembly_serialize_bin,
    rerank_beams_visual,
    visual_similarity,
)

SUITE31_DIR = REPO_ROOT / 'data-gen' / 'score-corpus' / 'suite31'
REPEATS_GT_XML = SUITE31_DIR / 'gt' / 'gt-p1-repeats-melody.musicxml'
REPEATS_CLEAN_PNG = SUITE31_DIR / 'clean' / 'repeats-melody.png'
REPEATS_PHOTO_PNG = SUITE31_DIR / 'photo' / 'repeats-melody.png'


def _extract_gt_bekern(xml_path: Path, measures: str = '1..5') -> list[str]:
    aser = default_assembly_serialize_bin()
    with tempfile.TemporaryDirectory(prefix='rdv_test_') as td:
        td_path = Path(td)
        proc = subprocess.run(
            [
                str(aser),
                'bekern',
                '--input', str(xml_path),
                '--measures', measures,
                '--out', str(td_path),
                '--stem', 'gt',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert proc.returncode == 0
        return (td_path / 'gt-bekern.txt').read_text(encoding='utf-8').split()


def _load_repeats_strips() -> tuple[np.ndarray, np.ndarray]:
    """Extract system 1 (measures 1..5) from suite31 clean & photo pages."""
    clean_im = Image.open(REPEATS_CLEAN_PNG).convert('L')
    clean_strip = np.asarray(clean_im, dtype=np.uint8)[145:290, :]

    photo_im = Image.open(REPEATS_PHOTO_PNG)
    photo_prepped, _ = prep_page(photo_im)
    photo_strip = np.asarray(photo_prepped.convert('L'), dtype=np.uint8)[174:321, :]
    return clean_strip, photo_strip


class TestRenderDiffVerify(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.aser = default_assembly_serialize_bin()
        assert cls.aser.exists(), f'Missing assembly-serialize binary: {cls.aser}'
        assert REPEATS_GT_XML.exists(), f'Missing suite31 GT XML: {REPEATS_GT_XML}'
        assert REPEATS_CLEAN_PNG.exists(), f'Missing suite31 clean PNG: {REPEATS_CLEAN_PNG}'
        assert REPEATS_PHOTO_PNG.exists(), f'Missing suite31 photo PNG: {REPEATS_PHOTO_PNG}'

        cls.gt_toks = _extract_gt_bekern(REPEATS_GT_XML, measures='1..5')
        cls.clean_strip, cls.photo_strip = _load_repeats_strips()

        # Pitch-mutated hypothesis (shifts every pitch step by a 5th).
        pitch_map = {'C': 'G', 'D': 'A', 'E': 'B', 'F': 'C', 'G': 'D', 'A': 'E', 'B': 'F'}
        cls.mut_pitch_toks = [pitch_map.get(t, t) for t in cls.gt_toks]

        # Truncated hypothesis (only first 2 measures).
        bar_pos = [i for i, t in enumerate(cls.gt_toks) if t == 'bar']
        cls.mut_trunc_toks = cls.gt_toks[:bar_pos[1]]

        # Dropped-notes hypothesis (replaces bars 2-3 with sparse repeated notes).
        cls.mut_drop_toks = (
            cls.gt_toks[:bar_pos[0] + 1]
            + ['voice-1', 'quarter', 'G', '4', 'quarter', 'G', '4', 'half', 'G', '4']
            + cls.gt_toks[bar_pos[2]:]
        )

    def test_gt_renders_fast_and_deterministically_on_cpu(self) -> None:
        # Warm up thread-local Verovio once, then time 3 consecutive renders.
        warm = bekern_to_strip_png(self.gt_toks, aser_bin=self.aser, target_h=256)
        self.assertIsNotNone(warm)
        assert warm is not None
        self.assertEqual(warm.dtype, np.uint8)
        self.assertEqual(warm.ndim, 2)
        self.assertEqual(warm.shape[0], 256)
        self.assertGreater(warm.shape[1], 500)

        times_ms = []
        outputs = []
        for _ in range(3):
            t0 = time.perf_counter()
            out = bekern_to_strip_png(self.gt_toks, aser_bin=self.aser, target_h=256)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            self.assertIsNotNone(out)
            outputs.append(out)

        median_ms = float(np.median(times_ms))
        self.assertLess(
            median_ms,
            200.0,
            f'Expected < 200 ms CPU render latency, got {median_ms:.1f} ms ({times_ms})',
        )
        # Bit-identical determinism across calls.
        np.testing.assert_array_equal(outputs[0], outputs[1])
        np.testing.assert_array_equal(outputs[1], outputs[2])

    def test_visual_similarity_clean_separates_gt_from_corruptions(self) -> None:
        ren_gt = bekern_to_strip_png(self.gt_toks, aser_bin=self.aser)
        ren_pitch = bekern_to_strip_png(self.mut_pitch_toks, aser_bin=self.aser)
        ren_drop = bekern_to_strip_png(self.mut_drop_toks, aser_bin=self.aser)
        ren_trunc = bekern_to_strip_png(self.mut_trunc_toks, aser_bin=self.aser)
        self.assertIsNotNone(ren_gt)
        self.assertIsNotNone(ren_pitch)
        self.assertIsNotNone(ren_drop)
        self.assertIsNotNone(ren_trunc)

        s_self = visual_similarity(ren_gt, ren_gt, n_bars=5)
        self.assertGreaterEqual(s_self['iou'], 0.99)
        self.assertEqual(s_self['flagged_slices'], [])

        s_gt = visual_similarity(self.clean_strip, ren_gt, n_bars=5)
        s_pitch = visual_similarity(self.clean_strip, ren_pitch, n_bars=5)
        s_drop = visual_similarity(self.clean_strip, ren_drop, n_bars=5)
        s_trunc = visual_similarity(self.clean_strip, ren_trunc, n_bars=5)

        self.assertEqual(len(s_gt['slice_ious']), 5)
        self.assertEqual(len(s_gt['slice_residuals']), 5)

        self.assertGreater(
            s_gt['iou'],
            s_pitch['iou'],
            f"GT ({s_gt['iou']}) must beat pitch-mutated ({s_pitch['iou']})",
        )
        self.assertGreater(
            s_gt['iou'],
            s_drop['iou'],
            f"GT ({s_gt['iou']}) must beat dropped-notes ({s_drop['iou']})",
        )
        self.assertGreater(
            s_gt['iou'],
            s_trunc['iou'],
            f"GT ({s_gt['iou']}) must beat truncated ({s_trunc['iou']})",
        )
        # Truncated hypothesis must flag trailing suspect measures.
        self.assertTrue(len(s_trunc['flagged_slices']) >= 1)

    def test_visual_similarity_photo_separates_gt_from_corruptions(self) -> None:
        ren_gt = bekern_to_strip_png(self.gt_toks, aser_bin=self.aser)
        ren_pitch = bekern_to_strip_png(self.mut_pitch_toks, aser_bin=self.aser)
        ren_drop = bekern_to_strip_png(self.mut_drop_toks, aser_bin=self.aser)
        ren_trunc = bekern_to_strip_png(self.mut_trunc_toks, aser_bin=self.aser)

        s_gt = visual_similarity(self.photo_strip, ren_gt, n_bars=5)
        s_pitch = visual_similarity(self.photo_strip, ren_pitch, n_bars=5)
        s_drop = visual_similarity(self.photo_strip, ren_drop, n_bars=5)
        s_trunc = visual_similarity(self.photo_strip, ren_trunc, n_bars=5)

        self.assertGreater(
            s_gt['iou'],
            s_pitch['iou'],
            f"Photo GT ({s_gt['iou']}) must beat pitch-mutated ({s_pitch['iou']})",
        )
        self.assertGreater(
            s_gt['iou'],
            s_drop['iou'],
            f"Photo GT ({s_gt['iou']}) must beat dropped-notes ({s_drop['iou']})",
        )
        self.assertGreater(
            s_gt['iou'],
            s_trunc['iou'],
            f"Photo GT ({s_gt['iou']}) must beat truncated ({s_trunc['iou']})",
        )

    def test_rerank_beams_visual_rescues_gt_over_higher_logprob_corruptions(self) -> None:
        beams = [
            self.mut_pitch_toks,
            self.mut_drop_toks,
            self.mut_trunc_toks,
            ['INVALID_BEKERN_TOKEN', '@@@'],
            self.gt_toks,
        ]
        # Corrupted beams have higher logprob than GT (-1.00 .. -1.03 vs -1.06).
        logprobs = [-1.00, -1.01, -1.02, -0.95, -1.06]

        for label, strip in (('clean', self.clean_strip), ('photo', self.photo_strip)):
            res = rerank_beams_visual(
                strip,
                beams,
                logprobs,
                alpha=2.0,
                aser_bin=self.aser,
                n_bars=5,
            )
            self.assertEqual(
                res['best_idx'],
                4,
                f'{label}: expected GT (idx=4), got idx={res["best_idx"]} scores={res["scores"]}',
            )
            self.assertEqual(res['best_toks'], self.gt_toks)
            # Invalid bekern beam failed soft with visual_iou=0.0 and render_ok=False.
            self.assertFalse(res['scores'][3]['render_ok'])
            self.assertEqual(res['scores'][3]['visual_iou'], 0.0)
            self.assertIsInstance(res['flagged_slices'], list)

    def test_fail_soft_on_invalid_bekern_and_bad_inputs(self) -> None:
        self.assertIsNone(bekern_to_strip_png([], aser_bin=self.aser))
        self.assertIsNone(bekern_to_strip_png('   ', aser_bin=self.aser))
        self.assertIsNone(
            bekern_to_strip_png(['not-valid', '???', '999'], aser_bin=self.aser)
        )
        self.assertIsNone(
            bekern_to_strip_png(self.gt_toks, aser_bin=Path('nonexistent_aser.exe'))
        )
        # Blank image gives 0.0 IoU without crashing.
        blank = np.full((128, 512), 255, dtype=np.uint8)
        sim = visual_similarity(blank, blank, n_bars=4)
        self.assertEqual(sim['iou'], 0.0)
        self.assertEqual(sim['flagged_slices'], [0, 1, 2, 3])

        # Input validation errors on rerank_beams_visual.
        with self.assertRaises(ValueError):
            rerank_beams_visual(self.clean_strip, [], [])
        with self.assertRaises(ValueError):
            rerank_beams_visual(self.clean_strip, [self.gt_toks], [-1.0, -2.0])


if __name__ == '__main__':
    unittest.main()
