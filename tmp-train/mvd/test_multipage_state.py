"""Unit tests for Multi-Page Streaming Symbolic State Prefix helpers in data.py."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402


class TestMultipageState(unittest.TestCase):
    def test_vocab_ids_unchanged(self):
        """Adding MULTIPAGE_STATE_TOKENS must not shift existing VOCAB IDs 0..236."""
        self.assertEqual(len(data.VOCAB), 233)
        self.assertEqual(data.PAD_ID, 233)
        self.assertEqual(data.BOS_ID, 234)
        self.assertEqual(data.EOS_ID, 235)
        self.assertEqual(data.VOCAB_SIZE, 236)
        self.assertEqual(
            data.MULTIPAGE_STATE_TOKENS,
            ["<prev_state>", "</prev_state>", "<partial_bar>"],
        )
        for tok in data.MULTIPAGE_STATE_TOKENS:
            self.assertNotIn(tok, data.TOKEN_TO_ID)

        # Default build_vocab() stays byte-identical to base VOCAB
        base_vocab, base_map = data.build_vocab()
        self.assertEqual(base_vocab, data.VOCAB)
        self.assertEqual(base_map, data.TOKEN_TO_ID)

        # Opt-in build_vocab(include_multipage_state=True) appends strictly at tail
        ext_vocab, ext_map = data.build_vocab(include_multipage_state=True)
        self.assertEqual(ext_vocab[: len(data.VOCAB)], data.VOCAB)
        self.assertEqual(ext_vocab[len(data.VOCAB) :], data.MULTIPAGE_STATE_TOKENS)
        self.assertEqual(ext_map["<prev_state>"], len(data.VOCAB))
        self.assertEqual(ext_map["</prev_state>"], len(data.VOCAB) + 1)
        self.assertEqual(ext_map["<partial_bar>"], len(data.VOCAB) + 2)

    def test_extract_single_staff_multi_system(self):
        """Single-staff multi-system page inherits omitted time-* and tracks key change."""
        page = (
            "clef-G-2 key+1 time-3-4 quarter G 4 quarter A 4 quarter B 4 bar <sys> "
            "clef-G-2 key-2 half B 4 quarter C 5 bar"
        )
        self.assertEqual(
            data.extract_page_tail_state(page),
            ["clef-G-2", "key-2", "time-3-4"],
        )

    def test_extract_grand_staff(self):
        """Grand-staff page tracks active clef/key/time independently per staff."""
        page = (
            "clef-G-2 key-1 time-4-4 quarter F 4 quarter G 4 half A 4 bar <sys> "
            "clef-F-4 key-1 time-4-4 whole F 2 bar <sys> "
            "clef-G-2 key-1 whole A 4 bar <sys> "
            "clef-F-4 key-1 half F 2 half C 3 bar"
        )
        self.assertEqual(
            data.extract_page_tail_state(page),
            ["clef-G-2", "key-1", "time-4-4", "<sys>", "clef-F-4", "key-1", "time-4-4"],
        )

    def test_extract_open_tie_slur_hairpin_and_closed(self):
        """Open tie/slur/hairpin are emitted; closed ones are not."""
        open_page = (
            "clef-G-2 key+0 time-4-4 hairpin-cresc-start "
            "quarter C 4 slur-start quarter E 4 tie-start bar"
        )
        self.assertEqual(
            data.extract_page_tail_state(open_page),
            [
                "clef-G-2",
                "key+0",
                "time-4-4",
                "tie-start",
                "E",
                "4",
                "slur-start",
                "hairpin-cresc-start",
            ],
        )

        closed_page = (
            "clef-G-2 key+0 time-4-4 hairpin-dim-start "
            "quarter C 4 slur-start tie-start quarter C 4 tie-stop slur-stop "
            "hairpin-stop half D 4 bar"
        )
        self.assertEqual(
            data.extract_page_tail_state(closed_page),
            ["clef-G-2", "key+0", "time-4-4"],
        )

    def test_extract_open_repeat_and_split_bar(self):
        """Open bar-repeat-left and unclosed final measure emit repeat + <partial_bar>."""
        page = (
            "clef-G-2 key+2 time-6-8 bar-repeat-left "
            "eighth D 4 eighth E 4 eighth F 4 dot quarter A 4 bar "
            "eighth B 4 eighth A 4"
        )
        self.assertEqual(
            data.extract_page_tail_state(page),
            [
                "clef-G-2",
                "key+2",
                "time-6-8",
                "bar-repeat-left",
                "<partial_bar>",
            ],
        )

        closed_repeat = (
            "clef-G-2 key+2 time-6-8 bar-repeat-left "
            "dotted_ignored eighth D 4 bar-repeat-right"
        )
        self.assertEqual(
            data.extract_page_tail_state(closed_repeat),
            ["clef-G-2", "key+2", "time-6-8"],
        )

    def test_extract_empty_and_prev_state_stripped(self):
        self.assertEqual(data.extract_page_tail_state(""), [])
        self.assertEqual(data.extract_page_tail_state("   "), [])
        prefixed = (
            "<prev_state> clef-F-4 key-3 time-2-4 </prev_state> "
            "clef-G-2 key+0 time-4-4 quarter C 4 bar"
        )
        self.assertEqual(
            data.extract_page_tail_state(prefixed),
            ["clef-G-2", "key+0", "time-4-4"],
        )

    def test_format_prev_state_prefix_modes(self):
        """Verify empty dropout, noisy corruption, clean GT, and 3-way proportions."""
        state = [
            "clef-G-2",
            "key+1",
            "time-3-4",
            "tie-start",
            "E",
            "4",
            "slur-start",
            "<partial_bar>",
        ]
        rng = random.Random(42)

        # Empty mode
        empty = data.format_prev_state_prefix(
            state, rng=rng, empty_prob=1.0, corrupt_prob=0.0
        )
        self.assertEqual(empty, ["<prev_state>", "</prev_state>"])

        # Clean mode
        clean = data.format_prev_state_prefix(
            state, rng=rng, empty_prob=0.0, corrupt_prob=0.0
        )
        self.assertEqual(clean, ["<prev_state>", *state, "</prev_state>"])

        # Noisy mode: always differs from clean GT and never leaves broken tie-start
        for _ in range(50):
            noisy = data.format_prev_state_prefix(
                state, rng=rng, empty_prob=0.0, corrupt_prob=1.0
            )
            self.assertEqual(noisy[0], "<prev_state>")
            self.assertEqual(noisy[-1], "</prev_state>")
            inner = noisy[1:-1]
            self.assertNotEqual(inner, state)
            if "tie-start" in inner:
                idx = inner.index("tie-start")
                self.assertLess(idx + 2, len(inner))
                self.assertEqual(inner[idx + 1], "E")
                self.assertEqual(inner[idx + 2], "4")

        # Default 3-way distribution over 2000 draws (~25% empty, ~20% noisy, ~55% clean)
        rng_dist = random.Random(12345)
        n_empty = n_noisy = n_clean = 0
        for _ in range(2000):
            out = data.format_prev_state_prefix(state, rng=rng_dist)
            inner = out[1:-1]
            if not inner:
                n_empty += 1
            elif inner == state:
                n_clean += 1
            else:
                n_noisy += 1
        self.assertAlmostEqual(n_empty / 2000, 0.25, delta=0.04)
        self.assertAlmostEqual(n_noisy / 2000, 0.20, delta=0.04)
        self.assertAlmostEqual(n_clean / 2000, 0.55, delta=0.04)


if __name__ == "__main__":
    unittest.main()
