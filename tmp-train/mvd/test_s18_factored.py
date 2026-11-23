"""Unit tests for S18 Factored Event-Tuple Head, Micro-Glyph Focal & Zero-Prefix Readout."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import data
import model
import s18_factored as s18
import tuple_events as te


class FactoredCodecTest(unittest.TestCase):
    def test_bekern_to_factors_and_back_all_subtypes(self):
        # Covers NOTE, CHORD_NOTE, GRACE_NOTE, REST, BAR_REST, BAR (plain + variants),
        # HEADER (clef, key, time), VOICE, TUPLET_START/STOP, DYN, HAIRPIN,
        # standalone ARTIC, TIE, SLUR, and multi-bit attachments.
        toks = (
            'clef-G-2 key+2 time-6-8 '
            'voice-1 dyn-mf hairpin-cresc-start '
            'tuplet-start-3-2 '
            'quarter C 4 a1 dot dot tie-start stacc '
            'chord quarter E 4 accent '
            'grace eighth F 5 slur-start '
            'tuplet-stop '
            'bar-double '
            'voice-2 rest eighth dot fermata bar-repeat-left '
            'bar-rest bar-repeat-right '
            'dyn-ff hairpin-stop stacc tie-stop slur-stop '
            'bar-light-heavy bar'
        ).split()
        evs = te.bekern_to_tuples(toks)
        packed = s18.encode_events_to_tensors(evs, append_eos=True)
        self.assertEqual(packed['kind'].shape[0], len(evs) + 1)
        self.assertEqual(int(packed['kind'][-1]), s18.EOS_KIND_ID)
        self.assertIn('subtype', packed)

        # Verify header subtypes preserved.
        self.assertEqual(int(packed['kind'][0]), s18.KIND_TO_ID['HEADER'])
        self.assertEqual(s18.SUBTYPE_LIST[int(packed['subtype'][0])], 'clef-G-2')
        self.assertEqual(s18.SUBTYPE_LIST[int(packed['subtype'][1])], 'key+2')
        self.assertEqual(s18.SUBTYPE_LIST[int(packed['subtype'][2])], 'time-6-8')

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
                subtype_id=int(packed['subtype'][t]),
            )
            rebuilt.append(ev)
        self.assertEqual(te.tuples_to_bekern(rebuilt), toks)

    def test_align_token_ids_to_factors_handles_pad_eos_and_truncation(self):
        toks1 = 'clef-G-2 voice-1 quarter C 4 dot bar'.split()
        toks2 = 'voice-1 quarter D 4'.split()
        ids1 = [data.TOKEN_TO_ID[t] for t in toks1] + [data.EOS_ID]
        # Deliberately truncated note (missing octave) to verify safe fallback.
        ids2 = [data.TOKEN_TO_ID[t] for t in toks2] + [data.TOKEN_TO_ID['eighth'], data.TOKEN_TO_ID['E']]
        max_len = max(len(ids1), len(ids2)) + 2
        row1 = ids1 + [data.PAD_ID] * (max_len - len(ids1))
        row2 = ids2 + [data.PAD_ID] * (max_len - len(ids2))
        tgt = torch.tensor([row1, row2], dtype=torch.long)

        fac = s18.align_token_ids_to_factors(tgt)
        self.assertEqual(tuple(fac['kind'].shape), (2, max_len))
        self.assertEqual(int(fac['kind'][0, 0]), s18.KIND_TO_ID['HEADER'])
        self.assertEqual(s18.SUBTYPE_LIST[int(fac['subtype'][0, 0])], 'clef-G-2')
        self.assertEqual(int(fac['kind'][0, len(toks1)]), s18.EOS_KIND_ID)
        self.assertEqual(int(fac['kind'][0, -1]), s18.PAD_KIND_ID)
        # Row 2 recovered prefix before truncated tail.
        self.assertEqual(int(fac['kind'][1, 0]), s18.KIND_TO_ID['VOICE'])
        self.assertEqual(int(fac['kind'][1, 1]), s18.KIND_TO_ID['NOTE'])
        self.assertEqual(int(fac['pitch'][1, 1]), te.diatonic_index('D', 4))


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
        self.assertEqual(tuple(out['subtype'].shape), (2, 10, s18.N_SUBTYPES))
        self.assertEqual(tuple(out['attach'].shape), (2, 10, s18.N_ATTACH))

        # Flipping an attachment logit must NOT change duration/pitch/voice/kind/subtype.
        out_mod = {k: v.clone() for k, v in out.items()}
        out_mod['attach'][:, :, s18.ATTACH_TO_IDX['dot1']] += 10.0
        out_mod['attach'][:, :, s18.ATTACH_TO_IDX['stacc']] -= 10.0
        for k in ('kind', 'voice', 'dur', 'pitch', 'alter', 'subtype'):
            self.assertTrue(torch.equal(out[k].argmax(-1), out_mod[k].argmax(-1)))

    def test_factored_loss_masks_irrelevant_heads_and_backprops(self):
        torch.manual_seed(1)
        head = s18.FactoredEventHead(d_model=32)
        toks = 'clef-G-2 voice-1 quarter C 4 dot bar-double voice-2 rest half bar'.split()
        packed = s18.encode_events_to_tensors(te.bekern_to_tuples(toks), append_eos=True)
        targets = {k: v.unsqueeze(0) for k, v in packed.items()}
        T = targets['kind'].shape[1]
        h = torch.randn(1, T, 32, requires_grad=True)
        out = head(h)
        total, parts = head.compute_loss(out, targets, attach_pos_weight=2.0)
        self.assertEqual(
            set(parts.keys()),
            {'kind', 'voice', 'dur', 'pitch', 'alter', 'subtype', 'attach', 'attach_calib'},
        )
        self.assertGreater(float(total.detach()), 0.0)
        total.backward()
        self.assertIsNotNone(h.grad)
        self.assertTrue(bool(torch.isfinite(h.grad).all()))
        self.assertIsNotNone(head.attach_log_temp.grad)
        self.assertTrue(bool(torch.isfinite(head.attach_log_temp.grad).all()))

    def test_bernoulli_attachment_gate_calibration(self):
        torch.manual_seed(4)
        # Prior bias init keeps initial probabilities near attach_prior (~0.05) on zero input.
        head = s18.FactoredEventHead(d_model=16, attach_prior=0.05)
        probs0 = torch.sigmoid(head(torch.zeros(1, 4, 16))['attach'])
        self.assertTrue(torch.allclose(probs0, torch.full_like(probs0, 0.05), atol=1e-4))

        # Spraying 2x attachment rate incurs a strictly larger attach_calib penalty
        # than matching the ground-truth marginal rate.
        n = 40
        kind = torch.full((1, n), s18.KIND_TO_ID['NOTE'], dtype=torch.long)
        targets = {
            'kind': kind,
            'voice': torch.ones((1, n), dtype=torch.long),
            'dur': torch.full((1, n), s18.DUR_TO_ID['quarter'], dtype=torch.long),
            'pitch': torch.full((1, n), te.diatonic_index('C', 4), dtype=torch.long),
            'alter': torch.zeros((1, n), dtype=torch.long),
            'subtype': torch.zeros((1, n), dtype=torch.long),
            'attach': torch.zeros((1, n, s18.N_ATTACH), dtype=torch.float32),
        }
        # 10% true staccato rate.
        stacc_i = s18.ATTACH_TO_IDX['stacc']
        targets['attach'][0, :4, stacc_i] = 1.0

        def _make_out(stacc_prob: float) -> dict[str, torch.Tensor]:
            base = head(torch.zeros(1, n, 16))
            att = torch.full((1, n, s18.N_ATTACH), -20.0)
            att[:, :, stacc_i] = torch.logit(torch.tensor(stacc_prob))
            base['attach'] = att
            return base

        _, parts_matched = head.compute_loss(_make_out(0.10), targets)
        _, parts_spray2x = head.compute_loss(_make_out(0.20), targets)
        self.assertLess(float(parts_matched['attach_calib']), 1e-6)
        self.assertGreater(float(parts_spray2x['attach_calib']), 1e-4)

        # Threshold calibration suppresses over-confident false-positive spray.
        probs_val = torch.full((200, s18.N_ATTACH), 0.05)
        tgt_val = torch.zeros((200, s18.N_ATTACH))
        # Positives have prob ~0.85, negatives have prob ~0.55 (would spray at 0.50).
        tgt_val[:20, stacc_i] = 1.0
        probs_val[:20, stacc_i] = 0.85
        probs_val[20:, stacc_i] = 0.55
        best_th = head.calibrate_attach_thresholds(probs_val, tgt_val, beta=0.5)
        self.assertGreater(float(best_th[stacc_i]), 0.55)
        self.assertLessEqual(float(best_th[stacc_i]), 0.85)

    def test_micro_glyph_focal_weighting(self):
        # Micro-glyphs (accidentals, short flagged durations, ledger pitches, dots/stacc)
        # must get strictly higher loss than large/in-staff glyphs at equal logit error.
        # 1) Duration: 64th (weight 3.0) vs quarter (weight 1.0)
        logits_dur = torch.zeros(1, s18.N_DURS)
        l_quarter = s18.micro_glyph_focal_ce(
            logits_dur,
            torch.tensor([s18.DUR_TO_ID['quarter']]),
            class_weights=torch.tensor(s18.MICRO_GLYPH_DUR_WEIGHTS),
            gamma=2.0,
        )
        l_64th = s18.micro_glyph_focal_ce(
            logits_dur,
            torch.tensor([s18.DUR_TO_ID['64th']]),
            class_weights=torch.tensor(s18.MICRO_GLYPH_DUR_WEIGHTS),
            gamma=2.0,
        )
        self.assertAlmostEqual(float(l_64th / l_quarter), 3.0, places=5)

        # 2) Pitch: ledger-line pitch (A6) vs in-staff pitch (C4)
        pw = s18.build_pitch_micro_glyph_weights(ledger_weight=2.0)
        logits_p = torch.zeros(1, s18.N_PITCHES)
        l_instaff = s18.micro_glyph_focal_ce(
            logits_p, torch.tensor([te.diatonic_index('C', 4)]),
            class_weights=pw, gamma=2.0,
        )
        l_ledger = s18.micro_glyph_focal_ce(
            logits_p, torch.tensor([te.diatonic_index('A', 6)]),
            class_weights=pw, gamma=2.0,
        )
        self.assertAlmostEqual(float(l_ledger / l_instaff), 2.0, places=5)

        # 3) Focal gamma down-weights confident easy examples vs hard errors.
        logits_easy = torch.tensor([[5.0, -5.0, -5.0, -5.0, -5.0]])
        logits_hard = torch.tensor([[-2.0, 2.0, 0.0, 0.0, 0.0]])
        t0 = torch.tensor([0])
        ratio_ce = (
            s18.micro_glyph_focal_ce(logits_hard, t0, gamma=0.0)
            / s18.micro_glyph_focal_ce(logits_easy, t0, gamma=0.0)
        )
        ratio_focal = (
            s18.micro_glyph_focal_ce(logits_hard, t0, gamma=2.0)
            / s18.micro_glyph_focal_ce(logits_easy, t0, gamma=2.0)
        )
        self.assertGreater(float(ratio_focal), float(ratio_ce))

        # 4) Flat token weights boost micro-glyphs over plain tokens.
        tw = s18.token_micro_glyph_weights()
        self.assertGreater(float(tw[data.TOKEN_TO_ID['dot']]), float(tw[data.TOKEN_TO_ID['quarter']]))
        self.assertGreater(float(tw[data.TOKEN_TO_ID['stacc']]), float(tw[data.TOKEN_TO_ID['bar']]))
        self.assertGreater(float(tw[data.TOKEN_TO_ID['a1']]), float(tw[data.TOKEN_TO_ID['quarter']]))

    def test_s18_factored_head_consistency_and_grad_flow(self):
        torch.manual_seed(5)
        vocab_size = data.EOS_ID + 1
        wrapper = s18.S18FactoredHead(
            d_model=32, vocab_size=vocab_size, consistency_weight=0.5,
        )
        toks = 'clef-G-2 voice-1 quarter C 4 a1 dot bar'.split()
        ids = [data.TOKEN_TO_ID[t] for t in toks] + [data.EOS_ID, data.PAD_ID]
        tgt = torch.tensor([ids], dtype=torch.long)
        T = tgt.shape[1]

        # detach_tok=False: gradients flow into BOTH h and tok_logits.
        h = torch.randn(1, T, 32, requires_grad=True)
        tok_logits = torch.randn(1, T, vocab_size, requires_grad=True)
        total, metrics = wrapper.loss_from_tokens(
            h, tgt, tok_logits=tok_logits, detach_tok=False,
        )
        self.assertIn('s18_total', metrics)
        self.assertIn('s18_consistency', metrics)
        self.assertGreater(metrics['s18_consistency'], 0.0)
        total.backward()
        self.assertIsNotNone(h.grad)
        self.assertIsNotNone(tok_logits.grad)
        self.assertTrue(bool(torch.isfinite(h.grad).all()))
        self.assertTrue(bool(torch.isfinite(tok_logits.grad).all()))
        self.assertGreater(float(tok_logits.grad.abs().sum()), 0.0)

        # detach_tok=True (and default): tok_logits receives NO gradient.
        h2 = torch.randn(1, T, 32, requires_grad=True)
        tok_logits2 = torch.randn(1, T, vocab_size, requires_grad=True)
        total2, _ = wrapper.loss_from_tokens(
            h2, tgt, tok_logits=tok_logits2,
        )
        total2.backward()
        self.assertIsNotNone(h2.grad)
        self.assertIsNone(tok_logits2.grad)

        # Default consistency_weight is 0.0.
        default_wrapper = s18.S18FactoredHead(d_model=32, vocab_size=vocab_size)
        self.assertEqual(default_wrapper.consistency_weight, 0.0)

    def test_subtype_consistency_masked_strictly_to_subtype_kinds(self):
        torch.manual_seed(6)
        vocab_size = data.EOS_ID + 1
        fmaps = {'subtype': s18.build_vocab_factor_maps(vocab_size=vocab_size)['subtype']}
        # Batch of NOTE tokens only (NOT in SUBTYPE_KINDS), with flat tok_logits
        # heavily predicting 'stacc' (which appears in m_subtype for standalone ARTIC).
        kind_note = torch.full((1, 4), s18.KIND_TO_ID['NOTE'], dtype=torch.long)
        tok_logits = torch.full((1, 4, vocab_size), -10.0)
        tok_logits[:, :, data.TOKEN_TO_ID['stacc']] = 10.0
        sub_logits = torch.randn(1, 4, s18.N_SUBTYPES, requires_grad=True)
        fac_out = {
            'kind': torch.zeros(1, 4, s18.N_KINDS),
            'subtype': sub_logits,
        }
        loss_note = s18.factored_token_consistency_loss(
            fac_out, tok_logits, fmaps, kind_targets=kind_note,
        )
        self.assertEqual(float(loss_note), 0.0)
        self.assertFalse(loss_note.requires_grad)

        # On a genuine SUBTYPE_KINDS position (ARTIC), subtype consistency activates.
        kind_artic = torch.full((1, 4), s18.KIND_TO_ID['ARTIC'], dtype=torch.long)
        sub_logits2 = torch.randn(1, 4, s18.N_SUBTYPES, requires_grad=True)
        fac_out2 = {
            'kind': torch.zeros(1, 4, s18.N_KINDS),
            'subtype': sub_logits2,
        }
        loss_artic = s18.factored_token_consistency_loss(
            fac_out2, tok_logits, fmaps, kind_targets=kind_artic,
        )
        self.assertGreater(float(loss_artic), 0.0)
        loss_artic.backward()
        self.assertIsNotNone(sub_logits2.grad)
        self.assertGreater(float(sub_logits2.grad.abs().sum()), 0.0)

    def test_factored_embedding_shapes(self):
        emb = s18.FactoredEventEmbedding(d_model=48)
        toks = 'clef-G-2 voice-1 quarter D 4 stacc bar-double'.split()
        packed = s18.encode_events_to_tensors(te.bekern_to_tuples(toks), append_eos=True)
        batched = {k: v.unsqueeze(0) for k, v in packed.items()}
        x = emb(batched['kind'], batched['voice'], batched['dur'],
                batched['pitch'], batched['alter'], batched['attach'],
                subtype=batched['subtype'])
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
