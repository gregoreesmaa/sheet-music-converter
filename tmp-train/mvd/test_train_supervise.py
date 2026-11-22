"""Supervisor restart-decision contract (no subprocesses spawn here)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_supervise  # noqa: E402


class TestDecide(unittest.TestCase):
    def test_train_done_is_done(self):
        self.assertEqual(train_supervise.decide(0, 1, 20, False),
                         'done')

    def test_stop_file_wins_over_crash(self):
        self.assertEqual(train_supervise.decide(1, 1, 20, True),
                         'stop')

    def test_crash_retries_within_budget(self):
        self.assertEqual(train_supervise.decide(1, 1, 20, False),
                         'retry')
        self.assertEqual(train_supervise.decide(-9, 19, 20, False),
                         'retry')

    def test_budget_exhausted(self):
        self.assertEqual(train_supervise.decide(1, 20, 20, False),
                         'exhausted')
        self.assertEqual(train_supervise.decide(1, 21, 20, False),
                         'exhausted')

    def test_config_error_still_retries(self):
        # A crash is a crash; the runbook flags are fixed, so a
        # nonzero exit means retry, never silent success.
        self.assertEqual(train_supervise.decide(2, 3, 20, False),
                         'retry')


class TestEnsureResume(unittest.TestCase):
    def test_adds_resume_when_absent(self):
        args = ['--steps', '20000', '--out', 'D:/tmp/x']
        self.assertIn('--resume',
                      train_supervise.ensure_resume(args))
        self.assertEqual(args.count('--resume'), 0)

    def test_keeps_single_resume(self):
        args = ['--resume', '--steps', '20000']
        out = train_supervise.ensure_resume(args)
        self.assertEqual(out.count('--resume'), 1)


import unittest


class TestCooldownFor(unittest.TestCase):
    def test_doubles_with_cap(self):
        self.assertEqual(train_supervise.cooldown_for(1), 120)
        self.assertEqual(train_supervise.cooldown_for(2), 240)
        self.assertEqual(train_supervise.cooldown_for(3), 480)
        self.assertEqual(train_supervise.cooldown_for(4), 960)
        self.assertEqual(train_supervise.cooldown_for(5), 1800)
        self.assertEqual(train_supervise.cooldown_for(99), 1800)

    def test_nonpositive_is_base(self):
        self.assertEqual(train_supervise.cooldown_for(0), 120)

if __name__ == '__main__':
    unittest.main()
