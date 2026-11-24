"""Unit tests for PosFormer-style MusicalPosHead and Column-Pooled Voice-1 CTC head."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as data_mod
import model
import pos_forest_head
import train as train_mod

VOCAB = data_mod.VOCAB
T2I = data_mod.TOKEN_TO_ID
VOCAB_SIZE = len(VOCAB) + 3


def _ids(toks):
    return [T2I[t] for t in toks]


class TestPosForestTargets(unittest.TestCase):
    def test_extract_bekern_pos_targets_structure(self):
        seq = (
            _ids([
                'clef-G-2', 'key+0', 'time-4-4',
                'quarter', 'C', '4',
                'chord', 'quarter', 'E', '4',
                'quarter', 'D', '4', 'dot',
                'eighth', 'F', '4',
                'bar',
                'tuplet-start-3-2', 'eighth', 'G', '4', 'tuplet-stop',
                'voice-2', 'half', 'A', '3',
                'voice-12', 'quarter', 'B', '3',
                'bar-double',
            ])
            + [data_mod.EOS_ID, data_mod.PAD_ID, data_mod.PAD_ID]
        )
        tgt = torch.tensor([seq], dtype=torch.long)
        out = pos_forest_head.extract_bekern_pos_targets(
            tgt, id2tok=VOCAB, pad_id=data_mod.PAD_ID)

        bins = out['beat_phase_bin'][0].tolist()
        voices = out['voice_id'][0].tolist()
        flags = out['flags'][0]
        valid = out['valid_mask'][0].tolist()

        # Shape & range checks.
        self.assertEqual(out['beat_phase_bin'].shape, tgt.shape)
        self.assertEqual(out['voice_id'].shape, tgt.shape)
        self.assertEqual(out['flags'].shape, (1, len(seq), 3))
        self.assertTrue(all(0 <= b <= 15 for b in bins))
        self.assertTrue(all(0 <= v <= 7 for v in voices))
        # Last two tokens are PAD -> valid_mask False.
        self.assertFalse(valid[-1])
        self.assertFalse(valid[-2])
        self.assertTrue(all(valid[:-2]))

        # First note (quarter C 4) at beat 0.0 -> bin 0, not in chord.
        idx_q1 = 3
        self.assertEqual(bins[idx_q1:idx_q1 + 3], [0, 0, 0])
        self.assertEqual(flags[idx_q1, 0].item(), 0.0)

        # Chord tone (chord quarter E 4) stays at beat 0.0 -> bin 0, in_chord=1.
        idx_ch = 6
        self.assertEqual(bins[idx_ch:idx_ch + 4], [0, 0, 0, 0])
        for k in range(idx_ch, idx_ch + 4):
            self.assertEqual(flags[k, 0].item(), 1.0)

        # Next note (quarter D 4 dot) starts at beat 1.0/4.0 -> bin 4, in_chord=0.
        idx_q2 = 10
        self.assertEqual(bins[idx_q2:idx_q2 + 4], [4, 4, 4, 4])
        self.assertEqual(flags[idx_q2, 0].item(), 0.0)

        # Next note (eighth F 4) starts at beat 1.0 + 1.5 = 2.5 -> 2.5/4*16 = 10.
        idx_e = 14
        self.assertEqual(bins[idx_e:idx_e + 3], [10, 10, 10])

        # Barline sets is_bar_close=1 and resets beat_pos for next measure.
        idx_bar = 17
        self.assertEqual(flags[idx_bar, 2].item(), 1.0)
        self.assertEqual(flags[idx_e, 2].item(), 0.0)

        # Tuplet span immediately after bar starts at beat 0 with in_tuplet=1.
        idx_tup = 18
        self.assertEqual(bins[idx_tup], 0)
        for k in range(idx_tup, idx_tup + 5):
            self.assertEqual(flags[k, 1].item(), 1.0)

        # voice-2 -> voice_id=1; voice-12 -> clamped to 7.
        idx_v2 = 23
        self.assertEqual(voices[idx_v2], 1)
        self.assertEqual(voices[idx_v2 + 1], 1)
        self.assertEqual(bins[idx_v2], 0)
        idx_v12 = 27
        self.assertEqual(voices[idx_v12], 7)
        self.assertEqual(voices[idx_v12 + 1], 7)

        # bar-double also sets is_bar_close=1.
        idx_bd = 31
        self.assertEqual(flags[idx_bd, 2].item(), 1.0)

    def test_bar_rest_is_not_barline_and_barline_prefix_supported(self):
        custom_vocab = ['time-4-4', 'quarter', 'bar-rest', 'barline-final']
        tgt = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        out = pos_forest_head.extract_bekern_pos_targets(
            tgt, id2tok=custom_vocab, pad_id=99)
        self.assertEqual(out['flags'][0, 2, 2].item(), 0.0)  # bar-rest != barline
        self.assertEqual(out['flags'][0, 3, 2].item(), 1.0)  # barline* == barline


class TestVoice1CTCTargets(unittest.TestCase):
    def test_single_and_multivoice_monotonic_slicing(self):
        # Row 0: single-voice sequence (no voice-2) -> keeps entire sequence.
        r0 = _ids(['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4', 'bar']) + [
            data_mod.EOS_ID, data_mod.PAD_ID, data_mod.PAD_ID,
            data_mod.PAD_ID, data_mod.PAD_ID,
        ]
        # Row 1: multi-voice sequence -> keeps headers + voice-1 up to voice-2 reset.
        r1 = _ids([
            'clef-G-2', 'voice-1', 'quarter', 'C', '4', 'bar',
            'voice-2', 'half', 'E', '3', 'bar',
        ]) + [data_mod.EOS_ID]
        # Row 2: empty / immediate EOS -> length 0.
        r2 = [data_mod.BOS_ID, data_mod.EOS_ID] + [data_mod.PAD_ID] * 10

        tgt = torch.tensor([r0, r1, r2], dtype=torch.long)
        flat, lengths = model.extract_voice1_ctc_targets(
            tgt, id2tok=VOCAB,
            pad_id=data_mod.PAD_ID, bos_id=data_mod.BOS_ID, eos_id=data_mod.EOS_ID,
        )
        self.assertEqual(lengths.tolist(), [7, 6, 0])
        want_r0 = _ids(['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4', 'bar'])
        want_r1 = _ids(['clef-G-2', 'voice-1', 'quarter', 'C', '4', 'bar'])
        self.assertEqual(flat.tolist(), want_r0 + want_r1)
        # Ensure voice-2 token never leaks into CTC targets.
        self.assertNotIn(T2I['voice-2'], flat.tolist())


class TestPosAndCTCHeadsModelAndCompat(unittest.TestCase):
    def test_forward_backward_gradients_and_edge_cases(self):
        torch.manual_seed(0)
        net = model.MvdNet(
            vocab=VOCAB_SIZE, d_model=64, layers=2, heads=4, dim_ff=128,
            use_pos_head=True, use_ctc=True,
        )
        net.train()
        # W=256 -> w_mem=16 columns (> target length 6..7 for valid CTC alignment).
        img = torch.randn(2, 3, 64, 256)
        r0 = _ids(['clef-G-2', 'key+0', 'quarter', 'C', '4', 'bar']) + [
            data_mod.EOS_ID, data_mod.PAD_ID, data_mod.PAD_ID, data_mod.PAD_ID,
        ]
        r1 = _ids([
            'clef-G-2', 'voice-1', 'quarter', 'D', '4', 'bar',
            'voice-2', 'half', 'E', '3',
        ])
        tgt = torch.tensor([r0, r1], dtype=torch.long)

        # Include a realistic mem_mask (width-padded last 2 columns on row 0).
        h_mem, w_mem = 64 // 8, 256 // 16
        mem_mask = torch.zeros(2, h_mem, w_mem, dtype=torch.bool)
        mem_mask[0, :, -2:] = True
        mem_mask_flat = mem_mask.reshape(2, h_mem * w_mem)

        ce, _, _, h_shared, ctc_l = net.loss_shared(
            img, tgt, mem_mask=mem_mask_flat,
            return_hidden=True, need_ctc=True,
        )
        pos_l = net.pos_loss(h_shared, tgt)
        total = ce + 0.1 * pos_l + 0.1 * ctc_l
        total.backward()

        self.assertTrue(bool(torch.isfinite(total)))
        self.assertTrue(bool(torch.isfinite(pos_l)))
        self.assertTrue(bool(torch.isfinite(ctc_l)))

        for name, mod in (
            ('pos_head', net.pos_head),
            ('ctc_vpool', net.ctc_vpool),
            ('ctc_head', net.ctc_head),
        ):
            grads = [p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None]
            self.assertEqual(len(grads), sum(1 for _ in mod.parameters()), name)
            self.assertGreater(sum(grads), 0.0, f'zero grad in {name}')

        # Edge cases: all-PAD target must return finite 0.0 for both heads.
        all_pad = torch.full_like(tgt, net.pad_id)
        _, _, _, h_p, ctc_p = net.loss_shared(
            img, all_pad, return_hidden=True, need_ctc=True)
        pos_p = net.pos_loss(h_p, all_pad)
        self.assertEqual(float(pos_p.item()), 0.0)
        self.assertEqual(float(ctc_p.item()), 0.0)

        # Edge case: CTC target longer than w_mem must gracefully return finite 0.0.
        long_tgt = torch.full((2, w_mem + 10), T2I['quarter'], dtype=torch.long)
        _, _, _, ctc_long = net.loss_shared(img, long_tgt, need_ctc=True)
        self.assertEqual(float(ctc_long.item()), 0.0)

    def test_zero_inference_overhead(self):
        torch.manual_seed(1)
        net = model.MvdNet(
            vocab=VOCAB_SIZE, d_model=64, layers=1, heads=4, dim_ff=128,
            use_pos_head=True, use_ctc=True,
        )
        calls = {'pos': 0, 'vpool': 0, 'ctc': 0}
        h1 = net.pos_head.register_forward_hook(
            lambda _m, _i, _o: calls.__setitem__('pos', calls['pos'] + 1))
        h2 = net.ctc_vpool.register_forward_hook(
            lambda _m, _i, _o: calls.__setitem__('vpool', calls['vpool'] + 1))
        h3 = net.ctc_head.register_forward_hook(
            lambda _m, _i, _o: calls.__setitem__('ctc', calls['ctc'] + 1))
        img = torch.randn(1, 3, 64, 128)
        tgt = torch.randint(0, len(VOCAB), (1, 6))
        try:
            net.eval()
            with torch.no_grad():
                _ = net(img, tgt)
                _ = net.greedy(img, max_len=4, constrain=False)
                _ = net.greedy_kv(img, max_len=4, constrain=False)
                _ = net.beam_search(img, max_len=4, beam=2, constrain=False)
        finally:
            h1.remove()
            h2.remove()
            h3.remove()
        self.assertEqual(calls, {'pos': 0, 'vpool': 0, 'ctc': 0})

        # Default MvdNet has zero extra params (100% backward compatible).
        base = model.MvdNet(vocab=VOCAB_SIZE, d_model=64, layers=1, heads=4, dim_ff=128)
        self.assertFalse(hasattr(base, 'pos_head'))
        self.assertFalse(hasattr(base, 'ctc_vpool'))
        self.assertFalse(hasattr(base, 'ctc_head'))

    def test_load_weights_compat_warm_start_and_drop_and_optimizer(self):
        torch.manual_seed(2)
        base = model.MvdNet(vocab=VOCAB_SIZE, d_model=64, layers=1, heads=4, dim_ff=128)
        aux = model.MvdNet(
            vocab=VOCAB_SIZE, d_model=64, layers=1, heads=4, dim_ff=128,
            use_pos_head=True, use_ctc=True,
        )

        # Warm-start from legacy checkpoint -> pos-fresh + ctc-fresh.
        how_fresh = train_mod.load_weights_compat(aux, base.state_dict())
        self.assertEqual(how_fresh, 'strict+pos-fresh+ctc-fresh')
        self.assertTrue(torch.equal(aux.out.weight, base.out.weight))

        # Drop aux heads at inference -> pos-dropped + ctc-dropped.
        infer_net = model.MvdNet(
            vocab=VOCAB_SIZE, d_model=64, layers=1, heads=4, dim_ff=128)
        how_drop = train_mod.load_weights_compat(infer_net, aux.state_dict())
        self.assertEqual(how_drop, 'strict+pos-dropped+ctc-dropped')
        self.assertTrue(torch.equal(infer_net.out.weight, aux.out.weight))

        # Vocab expansion with use_ctc=True preserves prefix rows of ctc_head.
        small_ctc = model.MvdNet(
            vocab=VOCAB_SIZE - 2, d_model=64, layers=1, heads=4, dim_ff=128,
            use_pos_head=True, use_ctc=True,
        )
        big_ctc = model.MvdNet(
            vocab=VOCAB_SIZE, d_model=64, layers=1, heads=4, dim_ff=128,
            use_pos_head=True, use_ctc=True,
        )
        how_exp = train_mod.load_weights_compat(big_ctc, small_ctc.state_dict())
        self.assertIn('vocab-expand', how_exp)
        n_old_ctc = small_ctc.ctc_head.weight.shape[0]
        self.assertTrue(torch.equal(
            big_ctc.ctc_head.weight[:n_old_ctc], small_ctc.ctc_head.weight))

        # build_optimizer puts pos_head / ctc_vpool / ctc_head in fresh_lr_mult group.
        opt, _ = train_mod.build_optimizer(aux, lr=1e-4, fresh_lr_mult=2.5)
        fresh_lrs = {round(g['lr'], 10) for g in opt.param_groups}
        self.assertIn(round(2.5e-4, 10), fresh_lrs)

        # Resume/meta contract check.
        for k in ('pos_weight', 'ctc_weight'):
            self.assertIn(k, train_mod.META_KEYS)
            self.assertIn(k, train_mod.RESUME_KEYS)
            self.assertEqual(train_mod.LEGACY_DEFAULTS[k], 0.0)


if __name__ == '__main__':
    unittest.main()
