"""Self-contained checks for train.build_optimizer (CPU only, no data).

Run with the repo-root .venv python (works while the GPU trains):

    .venv/Scripts/python.exe tmp-train/mvd/test_schedule.py
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model  # noqa: E402
import train  # noqa: E402


def _net():
    return model.MvdNet(vocab=113, d_model=32, layers=1)


def test_groups_cover_all_params():
    net = _net()
    opt, sched = train.build_optimizer(net, 1e-4)
    assert sched is None, 'no schedule requested, expected None'
    assert len(opt.param_groups) == 2
    names0 = {n for n, p in net.named_parameters()
              if any(p is q for q in opt.param_groups[0]['params'])}
    names1 = {n for n, p in net.named_parameters()
              if any(p is q for q in opt.param_groups[1]['params'])}
    assert names0 and names1
    assert all(n.startswith(('encoder.', 'proj.')) for n in names0), names0
    assert not any(n.startswith(('encoder.', 'proj.')) for n in names1)
    assert len(names0) + len(names1) == len(list(net.named_parameters()))
    assert opt.param_groups[0]['lr'] == opt.param_groups[1]['lr'] == 1e-4
    print('PASS groups: trunk=%d decoder=%d params, single LR preserved'
          % (len(names0), len(names1)))


def test_warmup_cosine_shape():
    net = _net()
    opt, sched = train.build_optimizer(net, 1e-3, trunk_mult=10.0,
                                       warmup=100, cosine=True,
                                       total_updates=1000)
    assert sched is not None
    g = opt.param_groups
    first_trunk, first_dec = g[0]['lr'], g[1]['lr']
    assert abs(first_trunk - 1e-3 * 10.0 / 100) < 1e-12, first_trunk
    assert abs(first_dec - 1e-3 / 100) < 1e-12, first_dec
    peak_trunk = peak_dec = 0.0
    mid_ratio = None
    for k in range(1000):
        sched.step()
        peak_trunk = max(peak_trunk, g[0]['lr'])
        peak_dec = max(peak_dec, g[1]['lr'])
        if k == 500:
            mid_ratio = g[0]['lr'] / g[1]['lr']
    assert abs(peak_trunk - 1e-2) / 1e-2 < 0.02, peak_trunk
    assert abs(peak_dec - 1e-3) / 1e-3 < 0.02, peak_dec
    assert abs(mid_ratio - 10.0) < 0.5, mid_ratio
    assert g[0]['lr'] < peak_trunk * 0.05, g[0]['lr']
    assert g[1]['lr'] < peak_dec * 0.05, g[1]['lr']
    print('PASS warmup+cosine: trunk peaks %.5f, ends %.7f (10x held at k=500)'
          % (peak_trunk, g[0]['lr']))


def test_warmup_only_flat_after():
    net = _net()
    opt, sched = train.build_optimizer(net, 2e-4, trunk_mult=10.0,
                                       warmup=50, cosine=False,
                                       total_updates=1000)
    for _ in range(200):
        sched.step()
    assert opt.param_groups[0]['lr'] == 2e-4 * 10.0
    assert opt.param_groups[1]['lr'] == 2e-4
    print('PASS warmup-only flattens at full mult LRs')


def _args(**kw):
    d = dict(height=1024, lr=3e-4, seed=0, exclude=[], shuffle=False,
             no_amp=False, accum=1, init_from=None, trunk_lr_mult=1.0,
             warmup=0, cosine=False, sample_p=0.0, pages_file=None)
    d.update(kw)
    return argparse.Namespace(**d)


def test_resume_fingerprint():
    pairs = [('a.png', 'a-bekern.txt', 'ds0001-1')]
    meta = train.build_meta(_args(), pairs)
    assert set(train.META_KEYS) <= set(meta), 'fingerprint must cover config'
    with tempfile.TemporaryDirectory() as d:
        assert train.check_resume(d, meta) is None, 'no meta yet: proceed'
        train.atomic_write_text(Path(d) / 'meta.json', json.dumps(meta))
        assert train.check_resume(d, meta) is None, 'identical: proceed'
        lr2 = train.build_meta(_args(lr=1e-4), pairs)
        r = train.check_resume(d, lr2)
        assert r is not None and "'lr'" in r, r
        pages2 = train.build_meta(
            _args(), [('b.png', 'b-bekern.txt', 'ds0002-1')])
        r = train.check_resume(d, pages2)
        assert r is not None and "'pages'" in r, r
    print('PASS resume fingerprint: match proceeds, lr/pages mismatch refused')


def test_lock_second_writer_refuses():
    with tempfile.TemporaryDirectory() as d:
        train.acquire_lock(d)
        assert (Path(d) / '.lock').exists()
        try:
            train.acquire_lock(d)
        except SystemExit as e:
            assert 'locked' in str(e.code), e.code
        else:
            raise AssertionError('second writer was not refused')
    print('PASS lock: second writer refused with stale-lock recovery note')


def test_atomic_save_roundtrip():
    import torch
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'ckpt-1.pt'
        train.atomic_torch_save({'step': 1}, str(p))
        assert p.exists() and not Path(str(p) + '.tmp').exists()
        assert torch.load(p, map_location='cpu',
                          weights_only=True) == {'step': 1}
        t = Path(d) / 'meta.json'
        train.atomic_write_text(t, '{"a": 1}')
        assert json.loads(t.read_text(encoding='utf-8')) == {'a': 1}
        assert not Path(str(t) + '.tmp').exists()
    print('PASS atomic saves: tmp+rename round-trips, no half-files left')


if __name__ == '__main__':
    test_groups_cover_all_params()
    test_warmup_cosine_shape()
    test_warmup_only_flat_after()
    test_resume_fingerprint()
    test_lock_second_writer_refuses()
    test_atomic_save_roundtrip()
    print('ALL SCHEDULE TESTS PASSED')
