#!/usr/bin/env python3
"""CPU-only unit tests for tmp-train/mvd/self_train_omr.py."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

MVD_DIR = Path(__file__).resolve().parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import data  # noqa: E402
import model  # noqa: E402
import self_train_omr as st  # noqa: E402


# Valid 2-measure 4/4 bekern stream (16+ tokens, exact 4.0 beats per bar).
VALID_4_4 = (
    'clef-G-2 key+0 time-4-4 '
    'quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar '
    'half G 4 half A 4 bar'
).split()

# Valid 3-measure 4/4 with pickup bar 1 (1 beat), full interior bar (4 beats),
# and partial final bar (3 beats).
VALID_PICKUP_AND_FINAL = (
    'clef-G-2 key+0 time-4-4 '
    'quarter G 4 bar '
    'quarter C 5 quarter D 5 quarter E 5 quarter F 5 bar '
    'half G 5 quarter E 5 bar'
).split()

# Valid 4/4 with dotted quarter + eighth + triplet eighths + chord + bar-rest.
VALID_COMPLEX_RHYTHM = (
    'clef-G-2 key+0 time-4-4 '
    'quarter C 4 dot eighth D 4 '
    'tuplet-start-3-2 eighth E 4 eighth F 4 eighth G 4 tuplet-stop '
    'quarter A 4 chord quarter C 5 bar '
    'bar-rest bar'
).split()

# Broken interior bar: measure 2 has only 3 quarters in 4/4 (underflow).
BROKEN_UNDERFLOW = (
    'clef-G-2 key+0 time-4-4 '
    'whole C 4 bar '
    'quarter D 4 quarter E 4 quarter F 4 bar '
    'whole G 4 bar'
).split()

# Broken bar: measure 1 has 5 quarters in 4/4 (overflow).
BROKEN_OVERFLOW = (
    'clef-G-2 key+0 time-4-4 '
    'quarter C 4 quarter D 4 quarter E 4 quarter F 4 quarter G 4 bar '
    'whole A 4 bar'
).split()

# Syntax error for assembly-serialize bekern-decode: malformed pitch (no octave).
SYNTAX_INVALID = (
    'clef-G-2 key+0 time-4-4 '
    'quarter C quarter D 4 quarter E 4 quarter F 4 bar'
).split()


class TestSelfTrainVerifier(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            st.DEFAULT_ASER.is_file(),
            f'assembly-serialize binary missing at {st.DEFAULT_ASER}',
        )

    def test_valid_4_4_passes_live_bekern_decode(self):
        res = st.verify_pseudo_bekern(
            VALID_4_4,
            aser_bin=st.DEFAULT_ASER,
            mean_logprob=-0.05,
            loop_trimmed=False,
        )
        self.assertTrue(res.ok, f'expected pass, got {res}')
        self.assertTrue(bool(res))
        self.assertEqual(res.gate, 'passed')
        self.assertEqual(res.n_measures, 2)
        self.assertGreaterEqual(res.n_notes, 6)

    def test_pickup_and_partial_final_bar_pass(self):
        res = st.verify_pseudo_bekern(
            VALID_PICKUP_AND_FINAL,
            aser_bin=st.DEFAULT_ASER,
            mean_logprob=-0.10,
            loop_trimmed=False,
        )
        self.assertTrue(res.ok, f'expected pass, got {res}')
        self.assertEqual(res.n_measures, 3)

    def test_complex_rhythm_dots_triplets_chords_bar_rest_pass(self):
        res = st.verify_pseudo_bekern(
            VALID_COMPLEX_RHYTHM,
            aser_bin=st.DEFAULT_ASER,
            mean_logprob=-0.15,
            loop_trimmed=False,
        )
        self.assertTrue(res.ok, f'expected pass, got {res}')
        self.assertEqual(res.n_measures, 2)

    def test_broken_interior_underflow_fails_gate2(self):
        res = st.verify_pseudo_bekern(
            BROKEN_UNDERFLOW,
            aser_bin=st.DEFAULT_ASER,
            mean_logprob=-0.05,
            loop_trimmed=False,
        )
        self.assertFalse(res.ok)
        self.assertEqual(res.gate, 'gate2_bar_duration')
        self.assertIn('bar_underflow', res.reason)

    def test_broken_overflow_fails_gate2(self):
        res = st.verify_pseudo_bekern(
            BROKEN_OVERFLOW,
            aser_bin=st.DEFAULT_ASER,
            mean_logprob=-0.05,
            loop_trimmed=False,
        )
        self.assertFalse(res.ok)
        self.assertEqual(res.gate, 'gate2_bar_duration')
        self.assertIn('bar_overflow', res.reason)

    def test_repeat_loop_fails_gate3(self):
        res = st.verify_pseudo_bekern(
            VALID_4_4,
            aser_bin=st.DEFAULT_ASER,
            mean_logprob=-0.05,
            loop_trimmed=True,
        )
        self.assertFalse(res.ok)
        self.assertEqual(res.gate, 'gate3_confidence')
        self.assertEqual(res.reason, 'loop_trimmed')

    def test_too_short_fails_gate3(self):
        res = st.verify_pseudo_bekern(
            ['clef-G-2', 'key+0', 'time-4-4', 'whole', 'C', '4', 'bar'],
            aser_bin=st.DEFAULT_ASER,
            mean_logprob=-0.01,
            loop_trimmed=False,
        )
        self.assertFalse(res.ok)
        self.assertEqual(res.gate, 'gate3_confidence')
        self.assertIn('too_short', res.reason)

    def test_low_logprob_fails_gate3(self):
        res = st.verify_pseudo_bekern(
            VALID_4_4,
            aser_bin=st.DEFAULT_ASER,
            mean_logprob=-0.42,
            loop_trimmed=False,
            min_logprob=-0.25,
        )
        self.assertFalse(res.ok)
        self.assertEqual(res.gate, 'gate3_confidence')
        self.assertIn('low_logprob', res.reason)

    def test_syntax_invalid_fails_gate1_live_aser(self):
        res = st.verify_pseudo_bekern(
            SYNTAX_INVALID,
            aser_bin=st.DEFAULT_ASER,
            mean_logprob=-0.05,
            loop_trimmed=False,
        )
        self.assertFalse(res.ok)
        self.assertEqual(res.gate, 'gate1_syntax')
        self.assertIn('bekern_decode_rc_', res.reason)

    def test_held_out_exclusion(self):
        held = st.load_held_out_ids()
        self.assertTrue(st.is_held_out_entry('gate20_001', '', 'unlabeled', held))
        self.assertTrue(st.is_held_out_entry('foo', 'C:/data/suite31/x.png', '', held))
        self.assertTrue(st.is_held_out_entry('strip_1', '', 'suite31_hw', held))
        # Real gate20 ID from gate_manifest.json
        self.assertTrue(st.is_held_out_entry('ds0091-3_sys0', '', 'lieder', held))
        # Real suite31 slug
        self.assertTrue(st.is_held_out_entry('repeats-melody_p1', '', 'folk', held))
        # Clean unlabeled entry is allowed
        self.assertFalse(
            st.is_held_out_entry('imslp_unlabeled_0042', 'D:/tmp/u/0042.png', 'imslp', held)
        )

    def test_atomic_emit_pseudo_dataset(self):
        with tempfile.TemporaryDirectory(prefix='st_emit_') as td:
            root = Path(td)
            src_img = root / 'src.png'
            Image.fromarray(np.full((32, 64, 3), 240, dtype=np.uint8)).save(src_img)
            out_dir = root / 'out'
            mf = st.emit_pseudo_dataset(
                [{
                    'id': 'u_0001',
                    'path': str(src_img),
                    'domain': 'imslp',
                    'tokens': VALID_4_4,
                    'n_notes': 6,
                    'n_measures': 2,
                    'mean_logprob': -0.08,
                }],
                out_dir,
            )
            self.assertTrue(mf.is_file())
            self.assertTrue((out_dir / 'renders' / 'u_0001.png').is_file())
            tok_file = out_dir / 'targets' / 'u_0001-bekern.txt'
            self.assertTrue(tok_file.is_file())
            self.assertEqual(tok_file.read_text(encoding='utf-8').strip().split(), VALID_4_4)
            # No leftover .tmp files
            tmps = list(out_dir.rglob('*.tmp'))
            self.assertEqual(tmps, [])
            rows = [json.loads(line) for line in mf.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['id'], 'u_0001')

    def test_fixmatch_jsd_gradient_flow_cpu(self):
        torch.manual_seed(7)
        net = model.MvdNet(
            vocab=len(data.VOCAB) + 3,
            d_model=32,
            layers=1,
            heads=4,
            dim_ff=64,
            dropout=0.0,
            anisotropic=True,
            stride='4x8',
            pretrained=False,
        )
        net.train()
        weak_imgs = torch.randn(2, 3, 32, 64)
        strong_imgs = (weak_imgs + 0.25 * torch.randn_like(weak_imgs)).requires_grad_(True)
        pseudo_tgt = torch.randint(0, len(data.VOCAB), (2, 10), dtype=torch.long)

        # Low conf_thresh=0.0 guarantees active tokens and non-zero gradients
        loss, stats = st.unlabeled_consistency_step(
            net,
            weak_imgs,
            strong_imgs,
            pad_id=data.PAD_ID,
            temperature=1.5,
            conf_thresh=0.0,
            pseudo_tgt=pseudo_tgt,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)
        self.assertEqual(stats['keep_frac'], 1.0)
        self.assertTrue(net.training)

        loss.backward()
        self.assertIsNotNone(strong_imgs.grad)
        self.assertTrue(torch.isfinite(strong_imgs.grad).all())
        self.assertGreater(float(strong_imgs.grad.abs().sum().item()), 0.0)
        out_grad = net.out.weight.grad
        self.assertIsNotNone(out_grad)
        self.assertGreater(float(out_grad.abs().sum().item()), 0.0)

        # High conf_thresh=1.0 masks everything -> 0.0 loss, backward-safe
        net.zero_grad(set_to_none=True)
        loss_zero, stats_zero = st.unlabeled_consistency_step(
            net,
            weak_imgs,
            strong_imgs.detach().requires_grad_(True),
            pad_id=data.PAD_ID,
            temperature=1.5,
            conf_thresh=1.0,
            pseudo_tgt=pseudo_tgt,
        )
        self.assertEqual(float(loss_zero.item()), 0.0)
        self.assertEqual(stats_zero['n_active'], 0.0)
        loss_zero.backward()


if __name__ == '__main__':
    unittest.main()
