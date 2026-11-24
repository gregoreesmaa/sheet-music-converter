"""Tests for higher-res rectification, P2 foveal tap, and S=5..7 multi-staff windowing."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image

import data
import measure_decode as mdec
import model
import rectify as R
import test_rectify as tr


class HigherResRectifyTest(unittest.TestCase):
    def test_default_layout_exact_legacy(self):
        self.assertEqual(R.canonical_layout(), (12.0, 64.0, 256))
        self.assertEqual(R.canonical_layout(10.0, 256), (12.0, 64.0, 256))

    def test_high_res_target_sp_12_and_14_preserve_subpixel_accuracy(self):
        for target_sp, expected_h in ((12.0, 320), (14.0, 352)):
            top, pitch, out_h = R.canonical_layout(target_sp)
            self.assertEqual(out_h, expected_h)
            self.assertEqual(out_h % 32, 0)
            self.assertGreater(pitch, 4.0 * target_sp)

            bases = [200.0, 500.0]
            img, curves, _ = tr.build_case(bases)
            out_img, ref = R.rectify_system_strip(img, curves, target_sp=target_sp)
            arr = np.asarray(out_img)
            w_out = arr.shape[1]
            self.assertEqual(out_img.size, (round(tr.W_IN * target_sp / tr.SP), out_h))
            self.assertEqual(tuple(ref.shape), (1, out_h, w_out))

            keep = tr.keep_columns(w_out, (tr.DOT_X_E4, tr.DOT_X_F4))
            expected = [R.canon_y(s, k, target_sp) for s in range(2) for k in range(5)]
            res = tr.line_residuals(arr, expected, keep)
            worst = max(abs(v) for v in res.values())
            self.assertLessEqual(worst, tr.TOL, f'target_sp={target_sp} worst={worst}')

            # Inactive slots 2 and 3 must be pure paper white.
            for s in (2, 3):
                r0 = int(round(s * pitch))
                r1 = int(round((s + 1) * pitch))
                self.assertEqual(int((arr[r0:r1] != 255).sum()), 0)

            # Reference channel +1 on canonical active rows only.
            on_rows = sorted(int(round(e)) for e in expected)
            for r in on_rows:
                self.assertTrue(bool((ref[0, r, :] == 1.0).all()))


class FovealTapTest(unittest.TestCase):
    def test_foveal_off_preserves_ckpt22000_state_dict(self):
        net_default = model.MvdNet(vocab=90, d_model=64, layers=1, heads=4,
                                   dim_ff=128, stride='4x8')
        net_explicit = model.MvdNet(vocab=90, d_model=64, layers=1, heads=4,
                                    dim_ff=128, stride='4x8', foveal_tap=False)
        self.assertEqual(set(net_default.state_dict().keys()),
                         set(net_explicit.state_dict().keys()))
        self.assertFalse(any('foveal' in k for k in net_default.state_dict()))

    def test_foveal_on_zero_init_parity_and_stage0_grad_flow(self):
        torch.manual_seed(0)
        base = model.MvdNet(vocab=90, d_model=64, layers=1, heads=4,
                            dim_ff=128, stride='4x8', foveal_tap=False)
        fov = model.MvdNet(vocab=90, d_model=64, layers=1, heads=4,
                           dim_ff=128, stride='4x8', foveal_tap=True)
        # Warm-start from pre-tap state_dict with strict=False.
        missing, unexpected = fov.load_state_dict(base.state_dict(), strict=False)
        self.assertEqual(unexpected, [])
        self.assertTrue(all(k.startswith('foveal_proj.') for k in missing))

        base.eval()
        fov.eval()
        # Works on both H=256 and higher-res H=320 (target_sp=12.0)!
        for h_in in (256, 320):
            img = torch.randn(1, 3, h_in, 256)
            with torch.no_grad():
                m0 = base.encode(img)
                m1 = fov.encode(img)
            self.assertEqual(tuple(m0.shape), tuple(m1.shape))
            self.assertTrue(torch.allclose(m0, m1, atol=1e-5))

        # Non-zero foveal_proj weights must receive gradients and route back to encoder[:2].
        fov.train()
        for p in fov.foveal_proj.parameters():
            torch.nn.init.normal_(p, std=0.05)
        img = torch.randn(1, 3, 320, 256)
        mem = fov.encode(img)
        mem.pow(2).mean().backward()
        self.assertIsNotNone(fov.foveal_proj[0].weight.grad)
        self.assertGreater(float(fov.foveal_proj[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(fov.encoder[0][0].weight.grad.abs().sum()), 0.0)


class _StubMultiStaffNet:
    """Deterministic stub returning scripted bekern tokens per chunk call."""

    def __init__(self, chunk_streams):
        self.chunk_streams = [list(s) for s in chunk_streams]
        self.calls = []
        self.eos_id = data.EOS_ID

    def beam_search(self, img_tensor, beam=3, strict=True, max_len=564, mem_mask=None):
        idx = len(self.calls)
        self.calls.append({
            'shape': tuple(img_tensor.shape),
            'mem_mask_sum': int(mem_mask.sum()) if mem_mask is not None else None,
            'mem_mask_len': int(mem_mask.numel()) if mem_mask is not None else None,
        })
        toks = self.chunk_streams[idx]
        ids = data.encode(toks) + [self.eos_id]
        return torch.tensor([ids], dtype=torch.long)


class MultiStaffWindowingTest(unittest.TestCase):
    def _dummy_page_and_staves(self, n_staves):
        h = 120 * n_staves + 60
        w = 400
        img = Image.new('RGB', (w, h), (255, 255, 255))
        staves = [[30.0 + 120.0 * s + 12.0 * k for k in range(5)]
                  for s in range(n_staves)]
        return img, staves

    def test_window_staves_partitions_5_6_7_8(self):
        for s, want_lens in ((5, [4, 1]), (6, [4, 2]), (7, [4, 3]), (8, [4, 4])):
            wins = mdec.window_staves(list(range(s)))
            self.assertEqual([len(c) for _, _, c in wins], want_lens)
            flat = [x for _, _, c in wins for x in c]
            self.assertEqual(flat, list(range(s)))

    def test_five_six_seven_staff_decode_masks_and_voice_merge(self):
        for n_staves, rem in ((5, 1), (6, 2), (7, 3)):
            img, staves = self._dummy_page_and_staves(n_staves)
            chunk0 = (
                'clef-G-2 key+0 time-4-4 '
                'voice-1 whole C 4 voice-2 whole E 4 '
                'voice-3 whole G 3 voice-4 whole C 3 bar '
                'voice-1 whole D 4 voice-2 whole F 4 bar'
            ).split()
            # Chunk 1 has ``rem`` staves (1, 2, or 3 voices).
            c1_m1 = []
            c1_m2 = []
            for v in range(1, rem + 1):
                c1_m1.extend([f'voice-{v}', 'whole', 'A', '2'])
                c1_m2.extend([f'voice-{v}', 'whole', 'G', '2'])
            chunk1 = ['clef-F-4', 'key+0', 'time-4-4'] + c1_m1 + ['bar'] + c1_m2 + ['bar']

            net = _StubMultiStaffNet([chunk0, chunk1])
            out = mdec.decode_multi_staff_system(
                net, img, staves, stride='4x8', target_sp=12.0
            )
            self.assertEqual(len(net.calls), 2)
            # Chunk 0 has 4 active staves -> 0 masked cells; height=320 at target_sp=12.0.
            self.assertEqual(net.calls[0]['shape'][2], 320)
            self.assertEqual(net.calls[0]['mem_mask_sum'], 0)
            # Chunk 1 has ``rem`` active staves -> (4 - rem)/4 of memory cells masked.
            self.assertEqual(net.calls[1]['shape'][2], 320)
            expected_masked = net.calls[1]['mem_mask_len'] * (4 - rem) // 4
            self.assertEqual(net.calls[1]['mem_mask_sum'], expected_masked)

            header, measures, tb = mdec.split_measures(out)
            self.assertTrue(tb)
            self.assertEqual(len(measures), 2)
            # Measure 0 had 4 voices in chunk 0 -> chunk 1 shifted to voice-5..voice-(4+rem).
            m0_voices = [t for t in measures[0] if t.startswith('voice-')]
            self.assertEqual(m0_voices, [f'voice-{i}' for i in range(1, 4 + rem + 1)])
            # Measure 1 had 2 voices on 4 staves in chunk 0 -> chunk 1 shifted by
            # max(2, 4) = 4 to voice-5..voice-(4+rem) (preserves per-staff IDs).
            m1_voices = [t for t in measures[1] if t.startswith('voice-')]
            self.assertEqual(
                m1_voices,
                ['voice-1', 'voice-2'] + [f'voice-{i}' for i in range(5, 4 + rem + 1)],
            )


if __name__ == '__main__':
    unittest.main()
