"""Modifier-F1 gate metric + regression check that dropped spike S2 ILME is removed."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model


class DroppedS2RegressionTest(unittest.TestCase):
    def test_s2_ilme_helpers_removed(self):
        self.assertFalse(hasattr(model, 'apply_ilme'))
        self.assertFalse(hasattr(model, 'default_ilme_ids'))
        self.assertTrue(hasattr(model, 'MODIFIER_TOKENS'))
        self.assertEqual(model.ILME_MODIFIERS, model.MODIFIER_TOKENS)


class ModifierF1Test(unittest.TestCase):
    def test_perfect_match_scores_one(self):
        import eval_sys_gate as ev
        ref = ['clef-G-2', 'C', '4', 'quarter', 'dot', 'D', '4']
        self.assertAlmostEqual(ev.modifier_f1(ref, list(ref)), 1.0)

    def test_missed_modifier_hurts_recall(self):
        import eval_sys_gate as ev
        ref = ['C', '4', 'quarter', 'dot', 'D', '4', 'quarter', 'stacc']
        hyp = ['C', '4', 'quarter', 'D', '4', 'quarter']
        f1 = ev.modifier_f1(ref, hyp)
        self.assertAlmostEqual(f1, 0.0)

    def test_pitch_only_errors_ignored(self):
        import eval_sys_gate as ev
        ref = ['C', '4', 'quarter', 'dot']
        hyp = ['D', '4', 'quarter', 'dot']
        self.assertAlmostEqual(ev.modifier_f1(ref, hyp), 1.0)

    def test_empty_both_sides_scores_one(self):
        import eval_sys_gate as ev
        self.assertAlmostEqual(ev.modifier_f1(['C', 'quarter'],
                                              ['C', 'quarter']), 1.0)


if __name__ == '__main__':
    unittest.main()
