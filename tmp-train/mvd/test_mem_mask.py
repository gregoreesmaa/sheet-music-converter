"""Train/inference mem_mask parity (t24 maskfix probe + regression tests).

Covers the v2.3 mismatch: training masks inactive staff slots
(``y_mem >= 8 * S``) and centers DC over active cells only, while
inference decoded with ``mem_mask=None`` (unmasked white-slot rows +
shifted DC mean). Run with the repo-root .venv interpreter:

    .\\.venv\\Scripts\\python.exe -m unittest tmp-train.mvd.test_mem_mask -v
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data  # noqa: E402
import measure_decode  # noqa: E402
import model  # noqa: E402

VOCAB_SIZE = len(data.VOCAB) + 3  # PAD, BOS, EOS


def _small_net(**kw):
    kw.setdefault('vocab', 40)
    kw.setdefault('d_model', 32)
    kw.setdefault('layers', 1)
    kw.setdefault('heads', 4)
    kw.setdefault('dim_ff', 64)
    net = model.MvdNet(**kw)
    net.eval()
    return net


class TestSlotMaskConstruction(unittest.TestCase):
    def test_mem_dims_match_live_encoder(self):
        net = _small_net()
        with torch.no_grad():
            for w in (64, 100, 101, 104, 128, 200, 256, 400,
                      512, 640, 1000):
                feats = net.encoder(torch.zeros(1, 3, 256, w))
                self.assertEqual(
                    measure_decode.mem_dims_for_strip(256, w),
                    (feats.shape[2], feats.shape[3]),
                    'width %d' % w)

    def test_slot_mask_layout(self):
        h_mem, w_mem = 32, 16
        m2 = measure_decode.slot_mem_mask(2, h_mem, w_mem)
        self.assertEqual(tuple(m2.shape), (h_mem * w_mem,))
        self.assertEqual(m2.dtype, torch.bool)
        grid = m2.reshape(h_mem, w_mem)
        self.assertFalse(grid[:16].any())
        self.assertTrue(grid[16:].all())
        self.assertEqual(int(m2.sum()), 16 * w_mem)
        m4 = measure_decode.slot_mem_mask(4, h_mem, w_mem)
        self.assertFalse(m4.any())
        m1 = measure_decode.slot_mem_mask(1, h_mem, w_mem)
        self.assertEqual(int(m1.sum()), 24 * w_mem)

    def test_mask_for_strip_shape_and_device(self):
        ten = torch.zeros(1, 3, 256, 320)
        mask = measure_decode.mask_for_strip(ten, 2)
        h_mem, w_mem = measure_decode.mem_dims_for_strip(256, 320)
        self.assertEqual(tuple(mask.shape), (1, h_mem * w_mem))
        self.assertEqual(mask.device, ten.device)
        self.assertEqual(int(mask.sum()), (h_mem - 16) * w_mem)

    def test_mem_dims_match_live_4x8_encoder(self):
        net = _small_net(anisotropic=True, stride='4x8')
        with torch.no_grad():
            for w in (64, 100, 101, 104, 128, 200, 256, 400,
                      512, 640, 1000):
                feats = net.encoder(torch.zeros(1, 3, 256, w))
                self.assertEqual(
                    measure_decode.mem_dims_for_strip(
                        256, w, stride='4x8'),
                    (feats.shape[2], feats.shape[3]),
                    'width %d' % w)

    def test_mask_for_strip_4x8_geometry(self):
        ten = torch.zeros(1, 3, 256, 320)
        mask = measure_decode.mask_for_strip(ten, 2, stride='4x8')
        h_mem, w_mem = measure_decode.mem_dims_for_strip(
            256, 320, stride='4x8')
        self.assertEqual((h_mem, w_mem), (64, (320 // 4) // 2))
        self.assertEqual(tuple(mask.shape), (1, h_mem * w_mem))
        # S=2 of 4 slots at 64 rows: rows >= 32 masked.
        self.assertEqual(int(mask.sum()), (h_mem - 32) * w_mem)


class TestLayerStepMask(unittest.TestCase):
    def test_mask_matches_decoder_layer(self):
        torch.manual_seed(0)
        net = _small_net(layers=2)
        layer = net.decoder.layers[0]
        params = model._layer_params(layer)
        b, s, d = 2, 24, 32
        mem = torch.randn(b, s, d)
        tgt = torch.randn(b, 1, d)
        mask = torch.zeros(b, s, dtype=torch.bool)
        mask[:, 12:] = True
        mems = model._mem_kv(mem, params[1])
        with torch.no_grad():
            ref = layer(tgt, mem, memory_key_padding_mask=mask)
            out, _, _ = model._layer_step(layer, params, tgt, None,
                                          None, mems, mask)
        diff = (ref - out).abs().max().item()
        print('layer-step vs decoder-layer maxdiff: %.3g' % diff)
        self.assertLess(diff, 1e-6)

    def test_training_op_gives_zero_weight_to_masked_rows(self):
        torch.manual_seed(1)
        net = _small_net(layers=1)
        layer = net.decoder.layers[0]
        mha = layer.multihead_attn
        b, s, d = 1, 32, 32
        mem = torch.randn(b, s, d)
        qry = torch.randn(b, 1, d)
        mask = torch.zeros(b, s, dtype=torch.bool)
        mask[:, 16:] = True
        with torch.no_grad():
            _, wts = mha(qry, mem, mem,
                         key_padding_mask=mask,
                         need_weights=True,
                         average_attn_weights=False)
        wts = wts.squeeze(0).squeeze(1) if wts.dim() == 4 else wts
        masked_max = wts[..., 16:].abs().max().item()
        live_mass = wts[..., :16].sum(-1).min().item()
        print('masked-row max weight: %.3g, live mass: %.6f'
              % (masked_max, live_mass))
        self.assertLess(masked_max, 1e-9)
        self.assertAlmostEqual(live_mass, 1.0, places=5)

    def test_masked_mem_content_invariance(self):
        torch.manual_seed(2)
        net = _small_net()
        layer = net.decoder.layers[0]
        params = model._layer_params(layer)
        images = torch.randn(1, 3, 128, 128)
        mask = measure_decode.mask_for_strip(images, 2)
        with torch.no_grad():
            mem = net.encode(images, mask)
            tampered = mem.clone()
            tampered[mask.unsqueeze(-1).expand_as(mem)] += 100.0
            mems = [model._mem_kv(mem, params[1])]
            tmems = [model._mem_kv(tampered, params[1])]
            tok = torch.full((1, 1), net.bos_id, dtype=torch.long)
            out, _ = net._cached_step([params], mems, tok, 0,
                                         [(None, None)], mask)
            tout, _ = net._cached_step([params], tmems, tok, 0,
                                          [(None, None)], mask)
            diff = (out - tout).abs().max().item()
            print('masked-content invariance maxdiff: %.3g' % diff)
            self.assertTrue(torch.equal(out, tout))
            bare, _ = net._cached_step([params], mems, tok, 0,
                                          [(None, None)], None)
            tbare, _ = net._cached_step([params], tmems, tok, 0,
                                           [(None, None)], None)
            self.assertFalse(torch.equal(bare, tbare))


class TestLegacyParity(unittest.TestCase):
    def test_none_and_s4_byte_identical(self):
        torch.manual_seed(3)
        net = _small_net()
        net.eval()
        images = torch.randn(1, 3, 128, 128)
        s4 = measure_decode.mask_for_strip(images, 4)
        self.assertFalse(bool(s4.any()))
        with torch.no_grad():
            self.assertTrue(torch.equal(
                net.encode(images), net.encode(images, None)))
            self.assertTrue(torch.equal(
                net.encode(images), net.encode(images, s4)))
            self.assertTrue(torch.equal(
                net.greedy(images, max_len=8),
                net.greedy(images, max_len=8, mem_mask=None)))
            self.assertTrue(torch.equal(
                net.greedy(images, max_len=8, mem_mask=s4),
                net.greedy(images, max_len=8)))
            self.assertTrue(torch.equal(
                net.greedy_kv(images, max_len=8, constrain=False),
                net.greedy_kv(images, max_len=8, constrain=False,
                              mem_mask=None)))
            self.assertTrue(torch.equal(
                net.beam_search(images, max_len=8, constrain=False,
                                beam=2),
                net.beam_search(images, max_len=8, constrain=False,
                                beam=2, mem_mask=None)))


class TestMaskedBeamS2(unittest.TestCase):
    def test_masked_beam_decode_s2_vocab_legal(self):
        torch.manual_seed(4)
        net = model.MvdNet(vocab=VOCAB_SIZE, d_model=64, layers=2,
                           heads=4, dim_ff=128)
        net.eval()
        ten = torch.full((1, 3, 256, 320), 0.94)
        for y in (63, 73, 83, 93, 103, 127, 137, 147, 157, 167):
            ten[:, :, y - 1:y + 2, :] = 0.15
        ten += 0.02 * torch.randn_like(ten)
        mask = measure_decode.mask_for_strip(ten, 2)
        self.assertEqual(tuple(mask.shape), (1, 32 * 20))
        self.assertEqual(int(mask.sum()), 16 * 20)
        with torch.no_grad():
            out = net.beam_search(ten, beam=2, strict=True,
                                  max_len=48, constrain=True,
                                  mem_mask=mask)
        self.assertEqual(out.shape[0], 1)
        ids = [int(i) for i in out[0].tolist()]
        self.assertTrue(all(0 <= i < VOCAB_SIZE for i in ids))
        content = [i for i in ids if i not in (data.PAD_ID,
                                               data.BOS_ID, data.EOS_ID)]
        self.assertGreater(len(content), 0)
        toks = data.decode(content)
        self.assertTrue(all(t in data.TOKEN_TO_ID for t in toks))
        print('S=2 masked beam: %d ids, %d content tokens, all legal'
              % (len(ids), len(content)))


if __name__ == '__main__':
    unittest.main()
