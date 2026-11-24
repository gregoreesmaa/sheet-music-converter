"""Unit tests for tmp-train/mvd/trial_qwen_vl_omr.py.

Covers:
  1. ``--dry-run`` end-to-end execution in both ``zero_shot`` and ``lora_sft``
     modes, verifying JSON report generation and metric computation.
  2. Prompt formatting (system/user/assistant roles, image payload).
  3. ``bekern`` post-processing (code fences, prose filtering, tail order).
  4. Hard eval-overlap guard ensuring ZERO ``suite31_hw`` pieces can ever
     enter the LoRA SFT training split.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

_MVD_DIR = Path(__file__).resolve().parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import trial_qwen_vl_omr as tq  # noqa: E402


class TestTrialQwenVlOmr(unittest.TestCase):
    """Tests for Qwen2.5-VL-3B OMR value trial harness."""

    def test_hf_home_defaults_to_d_drive(self):
        self.assertTrue(
            os.environ.get("HF_HOME", "").upper().startswith("D:\\"),
            f"Expected HF_HOME on D:, got {os.environ.get('HF_HOME')!r}",
        )

    def test_prompt_formatting_inference_and_sft(self):
        im = Image.new("RGB", (128, 32), color=(255, 255, 255))
        inf_msgs = tq.format_omr_messages(im, target_bekern=None)
        self.assertEqual(len(inf_msgs), 2)
        self.assertEqual(inf_msgs[0]["role"], "system")
        self.assertEqual(inf_msgs[1]["role"], "user")
        user_types = [c["type"] for c in inf_msgs[1]["content"]]
        self.assertEqual(user_types, ["image", "text"])
        self.assertIs(inf_msgs[1]["content"][0]["image"], im)

        target = "clef-G-2 key+0 time-4-4 voice-1 whole C 4 bar"
        sft_msgs = tq.format_omr_messages(im, target_bekern=target)
        self.assertEqual(len(sft_msgs), 3)
        self.assertEqual(sft_msgs[2]["role"], "assistant")
        self.assertEqual(sft_msgs[2]["content"][0]["text"], target)

    def test_bekern_post_processing_strips_fences_and_prose(self):
        raw = (
            "Here is the transcription:\n"
            "```bekern\n"
            "clef-G-2 key+0 time-4-4 voice-1 quarter C 4 stacc a1 dot "
            "quarter D 4 quarter E 4 quarter F 4 bar\n"
            "```\n"
            "Hope this helps!"
        )
        toks = tq.extract_bekern_from_response(raw, strict_vocab=True, canonicalize=True)
        # Prose words are filtered out; alter + dot canonicalized before stacc.
        self.assertEqual(
            toks,
            [
                "clef-G-2",
                "key+0",
                "time-4-4",
                "voice-1",
                "quarter",
                "C",
                "4",
                "a1",
                "dot",
                "stacc",
                "quarter",
                "D",
                "4",
                "quarter",
                "E",
                "4",
                "quarter",
                "F",
                "4",
                "bar",
            ],
        )

    def test_eval_overlap_guard_blocks_all_suite31_hw_pieces(self):
        manifest = tq.SUITE31_HW_MANIFEST_DEFAULT
        self.assertTrue(
            manifest.exists(),
            f"Expected suite31_hw manifest at {manifest}",
        )
        blocklist = tq.build_suite31_blocklist([manifest])
        self.assertGreaterEqual(len(blocklist), 40)

        doc = json.loads(manifest.read_text(encoding="utf-8"))
        # Build synthetic training samples mimicking every single suite31_hw piece
        # (both exact slug and unprefixed HW work name) plus 2 safe training pieces.
        leaked_samples = []
        for piece in doc["pieces"]:
            slug = piece["slug"]
            leaked_samples.append(
                tq.StripSample(
                    sample_id=f"train:{slug}-1-s0",
                    piece_slug=slug,
                    domain="handwritten" if piece.get("handwritten") else "printed",
                    image_path=f"/tmp/{slug}_h0.png",
                    bekern_tokens=["clef-G-2", "key+0", "time-4-4", "bar"],
                )
            )
            if slug.startswith("hw-"):
                unprefixed = slug[3:]
                leaked_samples.append(
                    tq.StripSample(
                        sample_id=f"real_hw:{unprefixed}_p001",
                        piece_slug=unprefixed,
                        domain="handwritten",
                        image_path=f"/tmp/{unprefixed}.jpg",
                        bekern_tokens=["clef-G-2", "key+0", "time-4-4", "bar"],
                    )
                )

        safe_samples = [
            tq.StripSample(
                sample_id="t32:ds0001-1-s0",
                piece_slug="ds0001",
                domain="printed",
                image_path="/tmp/ds0001-1-s0.png",
                bekern_tokens=["clef-G-2", "key+0", "time-4-4", "bar"],
            ),
            tq.StripSample(
                sample_id="t32:lc99999999_mscz_Bravura-1-s0",
                piece_slug="lc99999999",
                domain="printed",
                image_path="/tmp/lc99999999-1-s0.png",
                bekern_tokens=["clef-G-2", "key+0", "time-4-4", "bar"],
            ),
        ]

        kept, dropped = tq.filter_training_samples(leaked_samples + safe_samples, blocklist)
        # ZERO suite31_hw pieces survive filtering; all safe samples survive.
        self.assertEqual(len(kept), len(safe_samples))
        self.assertEqual(len(dropped), len(leaked_samples))
        tq.assert_no_eval_overlap(kept, blocklist)

        # Passing even a single leaked sample directly to assert_no_eval_overlap raises.
        with self.assertRaises(ValueError):
            tq.assert_no_eval_overlap(safe_samples + [leaked_samples[0]], blocklist)

    def test_dry_run_zero_shot_and_lora_sft(self):
        with tempfile.TemporaryDirectory(prefix="test_qwen_vl_") as td:
            td_p = Path(td)

            # 1. zero_shot --dry-run
            zs_json = td_p / "zs_report.json"
            rc_zs = tq.main(
                ["--mode", "zero_shot", "--dry-run", "--n-printed", "2", "--n-hw", "2", "--out-json", str(zs_json)]
            )
            self.assertEqual(rc_zs, 0)
            self.assertTrue(zs_json.exists())
            zs_rep = json.loads(zs_json.read_text(encoding="utf-8"))
            self.assertEqual(zs_rep["mode"], "zero_shot")
            self.assertTrue(zs_rep["dry_run"])
            self.assertEqual(zs_rep["eval"]["overall"]["n"], 4)
            self.assertAlmostEqual(zs_rep["eval"]["overall"]["mean_valid_bar_ratio"], 1.0, places=4)
            self.assertAlmostEqual(zs_rep["eval"]["overall"]["mean_strict_f1"], 1.0, places=4)
            self.assertGreaterEqual(zs_rep["eval"]["overall"]["mean_ms_per_strip"], 0.0)

            # 2. lora_sft --dry-run
            sft_json = td_p / "sft_report.json"
            rc_sft = tq.main(
                [
                    "--mode",
                    "lora_sft",
                    "--dry-run",
                    "--steps",
                    "5",
                    "--n-printed",
                    "2",
                    "--n-hw",
                    "2",
                    "--out-json",
                    str(sft_json),
                ]
            )
            self.assertEqual(rc_sft, 0)
            self.assertTrue(sft_json.exists())
            sft_rep = json.loads(sft_json.read_text(encoding="utf-8"))
            self.assertEqual(sft_rep["mode"], "lora_sft")
            self.assertTrue(sft_rep["dry_run"])
            self.assertEqual(sft_rep["lora_config"]["r"], 16)
            self.assertEqual(sft_rep["lora_config"]["alpha"], 32)
            self.assertEqual(sft_rep["lora_config"]["max_pixels"], 512 * 28 * 28)
            self.assertEqual(sft_rep["data_guard"]["dropped_overlap_count"], 0)
            self.assertGreater(sft_rep["data_guard"]["suite31_blocklist_size"], 40)
            self.assertEqual(sft_rep["train"]["steps_completed"], 5)
            self.assertAlmostEqual(sft_rep["eval"]["overall"]["mean_valid_bar_ratio"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
