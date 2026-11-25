#!/usr/bin/env python3
"""CPU-only unit tests for 2027 Trial: Closed-Loop Render-and-Diff RLVR."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

# Enforce strictly CPU-only execution.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import grammar  # noqa: E402
import trial_2027_render_diff_rlvr as rd_rlvr  # noqa: E402

SUITE31_DIR = REPO_ROOT / "data-gen" / "score-corpus" / "suite31"
REPEATS_MELODY_PNG = SUITE31_DIR / "clean" / "repeats-melody.png"
REPEATS_MELODY_GT_XML = SUITE31_DIR / "gt" / "gt-p1-repeats-melody.musicxml"


def _extract_gt_bekern(musicxml_path: Path) -> str:
    """Extract canonical page-1 GT bekern tokens from a Suite31 MusicXML file."""
    exe = rd_rlvr.find_assembly_serialize_bin()
    with tempfile.TemporaryDirectory(prefix="test_rd_gt_") as td:
        td_path = Path(td)
        proc = subprocess.run(
            [
                str(exe),
                "bekern",
                "--input",
                str(musicxml_path),
                "--page",
                "1",
                "--out",
                str(td_path),
                "--stem",
                "gt",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert proc.returncode == 0
        return (td_path / "gt-bekern.txt").read_text(encoding="utf-8").strip()


def _drop_measures_as_skipped_system(bekern_str: str, keep_bars: int = 8) -> str:
    """Simulate a skipped-system OMR error by keeping only the first ``keep_bars`` measures."""
    toks = bekern_str.split()
    bars_seen = 0
    out = []
    for tok in toks:
        out.append(tok)
        if tok in grammar.BAR_LIKES:
            bars_seen += 1
            if bars_seen >= keep_bars:
                break
    return " ".join(out)


class TestVisualRenderDiffHelpers(unittest.TestCase):
    def test_self_similarity_and_margin_invariance(self) -> None:
        """Identical ink cropped tightly scores 1.0 even with extra white margins."""
        base = np.ones((200, 300), dtype=np.float32)
        # Draw 2 synthetic staff bands + note blobs
        for y in (40, 46, 52, 58, 64, 130, 136, 142, 148, 154):
            base[y : y + 1, 30:270] = 0.05
        for x in range(50, 250, 30):
            base[45:55, x : x + 8] = 0.0
            base[138:148, x + 5 : x + 13] = 0.0

        padded = np.ones((360, 460), dtype=np.float32)
        padded[90:290, 110:410] = base

        s_self = rd_rlvr.compute_visual_render_diff_reward(base, base)
        s_pad = rd_rlvr.compute_visual_render_diff_reward(base, padded)
        self.assertAlmostEqual(s_self, 1.0, places=5)
        self.assertAlmostEqual(s_pad, 1.0, places=4)

        # Blank image must score 0.0
        blank = np.ones((200, 300), dtype=np.float32)
        self.assertEqual(rd_rlvr.compute_visual_render_diff_reward(base, blank), 0.0)
        self.assertEqual(rd_rlvr.compute_visual_render_diff_reward(blank, base), 0.0)

    def test_render_bekern_to_ink_image_invalid_returns_none(self) -> None:
        self.assertIsNone(rd_rlvr.render_bekern_to_ink_image(""))
        self.assertIsNone(rd_rlvr.render_bekern_to_ink_image("quarter ??? invalid-tok"))


class TestClosedLoopSuite31RenderDiff(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert REPEATS_MELODY_PNG.exists(), f"Missing {REPEATS_MELODY_PNG}"
        assert REPEATS_MELODY_GT_XML.exists(), f"Missing {REPEATS_MELODY_GT_XML}"
        cls.input_img_gray = (
            np.asarray(Image.open(REPEATS_MELODY_PNG).convert("L"), dtype=np.float32) / 255.0
        )
        cls.gt_bekern = _extract_gt_bekern(REPEATS_MELODY_GT_XML)

    def test_exact_gt_beats_skipped_system_and_garbled_truncated(self) -> None:
        """Exact GT bekern must score strictly higher than skipped-system and garbled/truncated."""
        # 1. Exact GT bekern (all 16 measures of repeats-melody)
        r_gt = rd_rlvr.compute_label_free_render_diff_reward(
            self.input_img_gray, self.gt_bekern
        )

        # 2. (a) Skipped-system bekern: valid parse & valid meter, but drops half the page's systems
        skipped_sys_bekern = _drop_measures_as_skipped_system(self.gt_bekern, keep_bars=6)
        r_skipped = rd_rlvr.compute_label_free_render_diff_reward(
            self.input_img_gray, skipped_sys_bekern
        )

        # 3. (b1) Truncated bekern: only 2 measures (severely truncated)
        truncated_bekern = _drop_measures_as_skipped_system(self.gt_bekern, keep_bars=2)
        r_truncated = rd_rlvr.compute_label_free_render_diff_reward(
            self.input_img_gray, truncated_bekern
        )

        # 4. (b2) Garbled / unparseable bekern
        garbled_bekern = "clef-G-2 quarter quarter C ??? broken-token bar-light-heavy"
        r_garbled = rd_rlvr.compute_label_free_render_diff_reward(
            self.input_img_gray, garbled_bekern
        )

        # Verify GT renders cleanly and parses validly
        self.assertTrue(r_gt["render_ok"])
        self.assertEqual(r_gt["r_parse"], 1.0)
        self.assertEqual(r_gt["r_noloop"], 1.0)
        self.assertGreater(r_gt["r_visual"], 0.0)

        # Core label-free ordering proofs:
        # Exact GT strictly beats skipped-system on BOTH total reward and visual render-diff
        self.assertGreater(
            r_gt["reward"],
            r_skipped["reward"],
            msg=f"Expected GT ({r_gt}) > skipped-system ({r_skipped})",
        )
        self.assertGreater(
            r_gt["r_visual"],
            r_skipped["r_visual"],
            msg=f"Expected GT visual ({r_gt['r_visual']:.4f}) > skipped visual ({r_skipped['r_visual']:.4f})",
        )

        # Exact GT strictly beats truncated and garbled
        self.assertGreater(
            r_gt["reward"],
            r_truncated["reward"],
            msg=f"Expected GT ({r_gt}) > truncated ({r_truncated})",
        )
        self.assertGreater(
            r_gt["reward"],
            r_garbled["reward"],
            msg=f"Expected GT ({r_gt}) > garbled ({r_garbled})",
        )
        self.assertEqual(r_garbled["r_parse"], 0.0)
        self.assertFalse(r_garbled["render_ok"])

    def test_trial_grpo_step_cpu_smoke(self) -> None:
        """Verify `--mode trial` runs G=4 GRPO step via rlvr_grpo.grpo_step on CPU."""
        summary = rd_rlvr.run_trial_grpo(
            image_dir=REPEATS_MELODY_PNG,
            ckpt_path=None,
            steps=1,
            batch_size=1,
            group_size=4,
            max_len=24,
            img_h=128,
            img_w=128,
            device_str="cpu",
            seed=7,
        )
        self.assertEqual(summary["mode"], "trial")
        self.assertEqual(summary["group_size"], 4)
        self.assertEqual(len(summary["history"]), 1)
        step0 = summary["history"][0]
        self.assertTrue(np.isfinite(step0["loss"]))
        self.assertGreaterEqual(step0["mean_reward"], 0.0)
        self.assertLessEqual(step0["mean_reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
