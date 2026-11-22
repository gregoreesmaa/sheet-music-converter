"""S13b dense ink/keypoint auxiliary loss + heat-channel dropout.

The 8x16 grid starves early trunk layers (sparse token CE only).
S13b distills the 8 hybrid-extra channels (frozen HeatNet 7-ch probs +
staff grid) into a 2-layer 1x1 aux head on trunk features: dense
per-cell BCE on every step. Companion: per-channel dropout on the 8
heatmap inputs (decouples the encoder from RowNet/HeatNet failures).
"""

import unittest

import torch

import data
import model
import train


def make_hybrid_net(**kw):
    return model.MvdNet(vocab=len(data.VOCAB) + 3, anisotropic=True,
                        extra_chans=8, **kw)


class TestS13AuxHead(unittest.TestCase):
    def test_aux_pred_shapes_default_stride(self):
        net = make_hybrid_net()
        net.eval()
        with torch.no_grad():
            out = net.aux_pred(torch.zeros(1, 11, 256, 512))
        self.assertEqual(list(out.shape), [1, 8, 32, 32])

    def test_aux_pred_shapes_4x8(self):
        net = make_hybrid_net(stride='4x8')
        net.eval()
        with torch.no_grad():
            out = net.aux_pred(torch.zeros(1, 11, 256, 512))
        self.assertEqual(list(out.shape), [1, 8, 64, 64])

    def test_aux_bce_live_and_reaches_stem(self):
        torch.manual_seed(0)
        net = make_hybrid_net(stride='4x8')
        net.train()
        pred = net.aux_pred(torch.randn(1, 11, 256, 512))
        target = torch.rand(1, 8, 64, 64)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred, target)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)
        net.zero_grad()
        loss.backward()
        g = net.encoder[0][0].weight.grad
        self.assertIsNotNone(g)
        self.assertGreater(g.abs().sum().item(), 0.0)

    def test_compat_strict_unchanged(self):
        net = make_hybrid_net()
        state = net.state_dict()
        net2 = make_hybrid_net()
        how = train.load_weights_compat(net2, dict(state))
        self.assertEqual(how, 'strict')


class TestS13StrideTransfer(unittest.TestCase):
    def test_padconv_unwrap_loads_s10_ckpt(self):
        # S10 (8x16) ckpts carry unwrapped downsample keys; the 4x8
        # net wraps feats[2][1] in _PadConv (encoder.2.1.conv.*).
        torch.manual_seed(0)
        old = make_hybrid_net()  # 8x16, S10-like (no aux in ckpt)
        new = make_hybrid_net(stride='4x8')
        wrapped = [k for k in new.state_dict()
                   if '.1.conv.' in k and k.endswith(
                       ('.weight', '.bias'))]
        self.assertTrue(wrapped, 'expected wrapped downsample keys')
        state = {k: v.clone() for k, v in old.state_dict().items()
                 if not k.startswith('aux_head.')}
        how = train.load_weights_compat(new, state)
        self.assertEqual(how, 'strict+aux-fresh+padconv')
        got = new.state_dict()
        # Only blocks the 8x16 ckpt leaves unwrapped (feats[2];
        # feats[4]/[6] are _PadConv-wrapped on both sides already).
        mapped = [k for k in wrapped
                  if k.replace('.1.conv.', '.1.', 1) in state]
        self.assertTrue(mapped, 'expected at least one remapped key')
        for k in mapped:
            ck = k.replace('.1.conv.', '.1.', 1)
            self.assertTrue(torch.equal(got[k], state[ck]),
                            '%s must equal ckpt %s exactly' % (k, ck))

    def test_compat_aux_fresh(self):
        """Old ckpt (no aux keys) loads: trunk verbatim, aux fresh."""
        net = make_hybrid_net()
        state = {k: v.clone() for k, v in net.state_dict().items()
                 if not k.startswith('aux_head.')}
        self.assertTrue(state)
        net2 = make_hybrid_net()
        how = train.load_weights_compat(net2, dict(state))
        self.assertEqual(how, 'strict+aux-fresh')
        sd2 = net2.state_dict()
        for k, v in state.items():
            self.assertTrue(torch.equal(sd2[k], v), k)


class TestS13HeatDropout(unittest.TestCase):
    def test_zero_is_identity(self):
        torch.manual_seed(0)
        imgs = torch.randn(2, 11, 32, 32)
        out = train.heat_channel_dropout(imgs, 0.0)
        self.assertTrue(torch.equal(out, imgs))

    def test_drops_whole_channels_with_scaling(self):
        torch.manual_seed(7)
        imgs = torch.randn(2, 11, 16, 16) + 5.0  # all nonzero
        p = 0.5
        out = train.heat_channel_dropout(imgs, p)
        torch.manual_seed(7)
        _replay = torch.randn(2, 11, 16, 16) + 5.0  # same stream prefix
        keep = (torch.rand(2, 8, 1, 1) >= p).to(imgs.dtype)
        self.assertTrue(torch.equal(_replay, imgs))
        expected = imgs.clone()
        expected[:, 3:] = expected[:, 3:] * keep / (1.0 - p)
        self.assertTrue(torch.equal(out, expected))
        # RGB untouched.
        self.assertTrue(torch.equal(out[:, :3], imgs[:, :3]))
        # Whole-channel structure: each extra channel is all-kept or
        # all-dropped.
        for b in range(2):
            for c in range(8):
                ch = out[b, 3 + c]
                base = imgs[b, 3 + c]
                self.assertTrue(
                    bool((ch == base * 2.0).all())
                    or bool((ch == 0).all()))

    def test_rejects_unit_prob(self):
        with self.assertRaises(ValueError):
            train.heat_channel_dropout(torch.zeros(1, 11, 8, 8), 1.0)


class TestS13AuxMainFlow(unittest.TestCase):
    """Regression guard: hybrid-cache runs MUST filter pairs by sidecar.

    A past edit split the ``if args.hybrid_cache:`` block so the
    index/filter lines silently stopped running; the run then crashed
    on the first sidecar-less strip. This launches the real CLI for 2
    steps and asserts the filter prints and the aux term is logged.
    """

    def test_hybrid_filter_runs_and_aux_logged(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        out = Path(tempfile.mkdtemp(prefix='s13_auxflow_'))
        repo = Path(__file__).resolve().parent.parent.parent
        cmd = [sys.executable, 'tmp-train/mvd/train.py',
               '--renders', 'tmp-train/mvd-sysdata-full',
               '--targets', 'tmp-train/mvd-sysdata-full',
               '--pages', 'ds0001-1-s0', 'ds0001-1-s1',
               '--steps', '2', '--batch-size', '2', '--accum', '4',
               '--eval-every', '100', '--stride', '4x8', '--pretrained',
               '--tok-weight', '3.0', '--tok-dropout', '0.30',
               '--extra-chans', '8',
               '--hybrid-cache',
               'D:/tmp/t24hybrid-full/hybrid_staff.jsonl',
               '--heat-pt', 'tmp-train/det-data/heatnet.pt',
               '--aux-weight', '1.0', '--aux-every', '1',
               '--heat-dropout', '0.3',
               '--shuffle', '--warmup', '50', '--cosine', '--seed', '0',
               '--out', str(out)]
        r = subprocess.run(cmd, cwd=str(repo), capture_output=True,
                           text=True, timeout=900)
        combined = (r.stdout or '') + (r.stderr or '')
        self.assertEqual(r.returncode, 0, combined[-3000:])
        self.assertIn('hybrid index:', combined)
        self.assertIn('hybrid filter: kept 6 pairs', combined)
        self.assertIn(' aux ', combined)


if __name__ == '__main__':
    unittest.main()
