"""Regression checks confirming dropped spike S8 (latent SpecAugment) is removed from model.py."""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import model
import train


class DroppedS8RegressionTest(unittest.TestCase):
    def test_s8_symbols_removed_from_model(self):
        self.assertFalse(hasattr(model, 'SPECAUG_SPLIT'))
        self.assertFalse(hasattr(model, 'specaugment_latent'))

    def test_loss_accepts_none_augment_and_rejects_active(self):
        net = model.MvdNet(vocab=90, anisotropic=True, d_model=32, layers=1)
        net.eval()
        tgt = torch.randint(0, 87, (1, 16))
        tgt[:, -1] = net.eos_id
        img = torch.randn(1, 3, 256, 256)
        l0 = net.loss(img, tgt)
        l1 = net.loss(img, tgt, augment=None)
        self.assertTrue(bool(torch.equal(l0, l1)))
        with self.assertRaises(ValueError):
            net.loss(img, tgt, augment={'slope': 0.02})


class SampleSpecAugTest(unittest.TestCase):
    def test_sampler_deterministic_per_stream(self):
        a = train.sample_specaug_params(random.Random('7:42:specaug'))
        b = train.sample_specaug_params(random.Random('7:42:specaug'))
        self.assertEqual(a, b)

    def test_sampler_off_switch(self):
        rng = random.Random('7:42:specaug')
        self.assertIsNone(train.maybe_specaug(rng, 0.0))
        rng = random.Random('7:42:specaug')
        got = train.maybe_specaug(rng, 1.0)
        self.assertIsNotNone(got)
        self.assertIn('slope', got)


if __name__ == '__main__':
    unittest.main()
