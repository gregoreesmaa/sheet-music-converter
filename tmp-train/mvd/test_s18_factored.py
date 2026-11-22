"""Unit tests for S18 Factored Event-Tuple Head + Zero-Prefix Visual Readout."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import model
import s18_factored as s18
import tuple_events as te


class FactoredCodecTest(unittest.TestCase):
    def test_bekern_to_factors_and_back(self):
        toks = (
            'voice-1 quarter C 4 a1 dot tie-start stacc '
            'chord quarter E 4 '
            'voice-2 rest eighth dot bar'
        ).split()
        evs = te.bekern_to_tuples(toks)
        packed = s18.encode_events_to_tensors(evs, append_eos=True)
        # 5 events (VOICE, NOTE, CHORD_NOTE, VOICE, REST, BAR -> 6 events) + EOS
        self.assertEqual(packed['kind'].shape[0], len(evs) + 1)
        self.assertEqual(int(packed['kind'][-1]), s18.EOS_KIND_ID)
        # Verify note facets (index 1: quarter C 4 a1 dot tie-start stacc)
        self.assertEqual(int(packed['kind'][1]), s18.KIND_TO_ID['NOTE'])
        self.assertEqual(int(packed['voice'][1]), 1)
        self.assertEqual(int(packed['dur'][1]), s18.DUR_TO_ID['quarter'])
        self.assertEqual(int(packed['pitch'][1]), te.diatonic_index('C', 4))
        self.assertEqual(int(packed['alter'][1]), s18.ALTER_TO_ID['a1'])
        self.assertEqual(float(packed['attach'][1, s18.ATTACH_TO_IDX['dot1']]), 1.0)
        self.assertEqual(float(packed['attach'][1, s18.ATTACH_TO_IDX['dot2']]), 0.0)
        self.assertEqual(float(packed['attach'][1, s18.ATTACH_TO_IDX['tie_start']]), 1.0)
        self.assertEqual(float(packed['attach'][1, s18.ATTACH_TO_IDX['stacc']]), 1.0)

        # Round-trip back to event dicts and bekern tokens.
        rebuilt = []
        for t in range(len(evs)):
            ev = s18.factors_to_event(
                int(packed['kind'][t]),
                int(packed['voice'][t]),
                int(packed['dur'][t]),
                int(packed['pitch'][t]),
                int(packed['alter'][t]),
                packed['attach'][t].tolist(),
            )
            rebuilt.append(ev)
        self.assertEqual(te.tuples_to_bekern(rebuilt), toks)


class FactoredHeadTest(unittest.TestCase):
    def test_decoupled_heads_and_bernoulli_attachments(self):
        torch.manual_seed(0)
        head = s18.FactoredEventHead(d_model=64)
        h = torch.randn(2, 10, 64, requires_grad=True)
        out = head(h)
        self.assertEqual(tuple(out['kind'].shape), (2, 10, s18.N_KINDS))
        self.assertEqual(tuple(out['voice'].shape), (2, 10, s18.N_VOICES))
        self.assertEqual(tuple(out['dur'].shape), (2, 10, s18.N_DURS))
        self.assertEqual(tuple(out['pitch'].shape), (2, 10, s18.N_PITCHES))
        self.assertEqual(tuple(out['alter'].shape), (2, 10, s18.N_ALTERS))
        self.assertEqual(tuple(out['attach'].shape), (2, 10, s18.N_ATTACH))

        # Flipping an attachment logit must NOT change duration/pitch/voice/kind predictions.
        out_mod = {k: v.clone() for k, v in out.items()}
        out_mod['attach'][:, :, s18.ATTACH_TO_IDX['dot1']] += 10.0
        out_mod['attach'][:, :, s18.ATTACH_TO_IDX['stacc']] -= 10.0
        for k in ('kind', 'voice', 'dur', 'pitch', 'alter'):
            self.assertTrue(torch.equal(out[k].argmax(-1), out_mod[k].argmax(-1)))

    def test_factored_loss_masks_irrelevant_heads_and_backprops(self):
        torch.manual_seed(1)
        head = s18.FactoredEventHead(d_model=32)
        toks = 'voice-1 quarter C 4 dot bar voice-2 rest half bar'.split()
        packed = s18.encode_events_to_tensors(te.bekern_to_tuples(toks), append_eos=True)
        targets = {k: v.unsqueeze(0) for k, v in packed.items()}
        T = targets['kind'].shape[1]
        h = torch.randn(1, T, 32, requires_grad=True)
        out = head(h)
        total, parts = head.compute_loss(out, targets, attach_pos_weight=2.0)
        self.assertEqual(set(parts.keys()), {'kind', 'voice', 'dur', 'pitch', 'alter', 'attach'})
        self.assertGreater(float(total.detach()), 0.0)
        total.backward()
        self.assertIsNotNone(h.grad)
        self.assertTrue(bool(torch.isfinite(h.grad).all()))

    def test_factored_embedding_shapes(self):
        emb = s18.FactoredEventEmbedding(d_model=48)
        toks = 'voice-1 quarter D 4 stacc bar'.split()
        packed = s18.encode_events_to_tensors(te.bekern_to_tuples(toks), append_eos=True)
        batched = {k: v.unsqueeze(0) for k, v in packed.items()}
        x = emb(batched['kind'], batched['voice'], batched['dur'],
                batched['pitch'], batched['alter'], batched['attach'])
        self.assertEqual(tuple(x.shape), (1, packed['kind'].shape[0], 48))


class ZeroPrefixVisualReadoutTest(unittest.TestCase):
    def test_zero_prefix_visual_independence_and_mem_sensitivity(self):
        torch.manual_seed(2)
        net = model.MvdNet(vocab=90, d_model=64, layers=1, heads=4,
                           dim_ff=128, stride='4x8', foveal_tap=True)
        net.eval()
        probe = s18.ZeroPrefixVisualReadout(
            d_model=64, num_queries=16, heads=4, layers=2, dim_ff=128)
        probe.eval()

        img_a = torch.randn(1, 3, 256, 256)
        img_b = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            mem_a = net.encode(img_a)
            mem_b = net.encode(img_b)
            # Forward takes ONLY visual memory (no token prefix argument exists).
            out_a1, attn_a = probe(mem_a, return_attn=True)
            out_a2 = probe(mem_a)
            out_a3 = probe(mem_a)
            out_b = probe(mem_b)

        # Exact determinism & return_attn numeric parity.
        for k in out_a1:
            self.assertTrue(torch.equal(out_a2[k], out_a3[k]))
            self.assertTrue(torch.allclose(out_a1[k], out_a2[k], atol=1e-5))
            self.assertEqual(tuple(out_a1[k].shape[:2]), (1, 16))
        self.assertEqual(tuple(attn_a.shape), (1, 16, mem_a.shape[1]))
        # Changing visual memory must change readout logits.
        self.assertFalse(torch.allclose(out_a1['pitch'], out_b['pitch']))
        self.assertFalse(torch.allclose(out_a1['attach'], out_b['attach']))

    def test_mem_mask_ignores_masked_slots_exactly(self):
        torch.manual_seed(3)
        probe = s18.ZeroPrefixVisualReadout(
            d_model=32, num_queries=8, heads=4, layers=1, dim_ff=64)
        probe.eval()
        mem = torch.randn(1, 20, 32)
        mask = torch.zeros(1, 20, dtype=torch.bool)
        mask[:, 10:] = True
        tampered = mem.clone()
        tampered[:, 10:, :] += 500.0
        with torch.no_grad():
            o1 = probe(mem, mem_mask=mask)
            o2 = probe(tampered, mem_mask=mask)
        for k in o1:
            self.assertTrue(torch.allclose(o1[k], o2[k], atol=1e-5))


if __name__ == '__main__':
    unittest.main()
