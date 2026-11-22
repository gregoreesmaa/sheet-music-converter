"""Regression checks confirming dropped spike S1 (ink EOS veto) is removed."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import data
import grammar
import model


class DroppedS1RegressionTest(unittest.TestCase):
    def test_s1_symbols_removed(self):
        self.assertFalse(hasattr(model, 'count_notehead_peaks'))
        self.assertFalse(hasattr(model, 'INK_NOTE_CH'))
        self.assertFalse(hasattr(model, 'STEP_IDS'))

    def test_apply_knobs_allows_grammar_legal_eos(self):
        vocab = len(data.VOCAB) + 3
        net = SimpleNamespace(vocab=vocab, pad_id=vocab - 3,
                              bos_id=vocab - 2, eos_id=vocab - 1)
        toks = ['clef-G-2', 'key+0', 'time-4-4']
        st = grammar.PrefixState(strict=True)
        for t in toks:
            st.step(t)
        row = {'state': st, 'hist': [grammar._T[t] for t in toks],
               'ngrams': set(), 'gen_len': len(toks), 'closed': False}
        logits = torch.zeros(1, net.vocab)
        model.MvdNet._apply_knobs(net, logits, [row], 0, 0.0, 1.0, 0, 0, 0)
        self.assertEqual(logits[0, net.eos_id].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
