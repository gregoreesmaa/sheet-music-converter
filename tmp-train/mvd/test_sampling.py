"""Self-contained checks for MvdNet scheduled sampling (CPU only).

No data files, no GPU (safe while the GPU trains):

    .venv/Scripts/python.exe tmp-train/mvd/test_sampling.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import model  # noqa: E402

V = len(data.VOCAB) + 3


def _net():
    torch.manual_seed(0)
    return model.MvdNet(vocab=V, d_model=32, layers=1)


def _batch():
    torch.manual_seed(1)
    img = torch.randn(1, 3, 32, 48)
    tgt = torch.randint(0, len(data.VOCAB), (1, 24))
    tgt[0, 18:] = data.PAD_ID  # padded tail, as in collate()
    return img, tgt


def test_teacher_path_reproducible():
    net = _net()
    img, tgt = _batch()
    torch.manual_seed(7)
    a = net.loss(img, tgt, sample_p=0.0).item()
    torch.manual_seed(7)
    b = net.loss(img, tgt, sample_p=0.0).item()
    assert a == b, (a, b)
    assert a > 0 and a < 1e6, a
    print('PASS teacher path finite + reproducible: %.4f' % a)


def test_sampling_differs_and_finite():
    net = _net()
    img, tgt = _batch()
    base = net.loss(img, tgt, sample_p=0.0).item()
    torch.manual_seed(11)
    s = net.loss(img, tgt, sample_p=0.5).item()
    assert s > 0 and s < 1e6, s
    assert s != base, 'mixed prefix left logits unchanged'
    torch.manual_seed(11)
    s2 = net.loss(img, tgt, sample_p=0.5).item()
    assert s == s2, 'sampling not seeded-reproducible'
    print('PASS sampling: teacher %.4f vs mixed %.4f, reproducible' % (base, s))


def test_full_replace_runs():
    net = _net()
    img, tgt = _batch()
    torch.manual_seed(3)
    s = net.loss(img, tgt, sample_p=1.0).item()
    assert s > 0 and s < 1e6, s
    print('PASS full replace (p=1.0) finite: %.4f' % s)


if __name__ == '__main__':
    test_teacher_path_reproducible()
    test_sampling_differs_and_finite()
    test_full_replace_runs()
    print('ALL SAMPLING TESTS PASSED')
