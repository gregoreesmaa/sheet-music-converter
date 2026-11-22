"""MVD model smoke checks (run with the repo-root .venv interpreter).

1. Forward shapes: encoder feature sequence + decoder logits.
2. Overfit: one page pair, loss must fall below its start after K steps.
Neither test trains a real model; they pin the wiring the trainer needs.
"""

import sys
import unittest

import torch

sys.path.insert(0, r'D:\projects\sheet-music-converter\tmp-train\mvd')

import model
import data as data_mod
from data import VOCAB

VOCAB_SIZE = len(VOCAB) + 3  # PAD, BOS, EOS


def test_forward_shapes():
    net = model.MvdNet(vocab=VOCAB_SIZE, d_model=256, layers=2)
    net.eval()
    img = torch.randn(2, 3, 256, 384)
    tgt = torch.randint(0, VOCAB_SIZE, (2, 16))
    logits = net(img, tgt)
    assert logits.shape == (2, 16, VOCAB_SIZE), logits.shape
    print('PASS forward shapes:', tuple(logits.shape))


def test_long_sequence_no_position_cap():
    # ds0188-4 carries 4414 tokens: fixed positional tables (4096)
    # crash with a device-side assert. Positions must be unbounded.
    net = model.MvdNet(vocab=VOCAB_SIZE, d_model=256, layers=1)
    net.eval()
    img = torch.randn(1, 3, 128, 128)
    tgt = torch.randint(0, VOCAB_SIZE, (1, 5000))
    with torch.no_grad():
        logits = net(img, tgt)
    assert logits.shape == (1, 5000, VOCAB_SIZE), logits.shape
    print('PASS long sequence: 5000 positions, no cap')


def test_causal_no_peek():
    # Same prefix + different suffix must give identical prefix logits.
    net = model.MvdNet(vocab=VOCAB_SIZE, d_model=256, layers=2)
    net.eval()
    img = torch.randn(1, 3, 256, 384)
    a = torch.randint(0, VOCAB_SIZE, (1, 16))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % VOCAB_SIZE
    with torch.no_grad():
        la = net(img, a)
        lb = net(img, b)
    assert torch.equal(la[0, :8], lb[0, :8]), 'decoder peeks at future tokens'
    print('PASS causal mask')


def test_overfit_one_sample():
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = model.MvdNet(vocab=VOCAB_SIZE, d_model=256, layers=2).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    img = torch.randn(1, 3, 256, 384, device=device)
    tgt = torch.randint(0, VOCAB_SIZE, (1, 32), device=device)
    with torch.no_grad():
        start = net.loss(img, tgt).item()
    for _ in range(30):
        opt.zero_grad()
        loss = net.loss(img, tgt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        end = net.loss(img, tgt).item()
    assert end < start, (start, end)
    print('PASS overfit: %.4f -> %.4f on %s' % (start, end, device))


def test_anisotropic_encoder_spatial_resolution():
    # v2: 8x vertical / 16x horizontal downsampling -> H=256 gives 32 rows.
    net = model.MvdNet(vocab=90, anisotropic=True)
    net.eval()
    img = torch.zeros(2, 3, 256, 512)
    with torch.no_grad():
        feats = net.encoder(img)
        mem = net.encode(img)
    assert feats.shape[2:] == (32, 32), feats.shape
    assert mem.shape == (2, 32 * 32, net.d_model), mem.shape
    print('PASS anisotropic resolution:', tuple(feats.shape),
          '->', tuple(mem.shape))


def test_pretrained_init_loads_cleanly():
    # Pretrained trunk: deterministic across inits, non-random, and the
    # full forward + greedy_kv path runs on top of it.
    kw = dict(vocab=90, anisotropic=True, pretrained=True,
              d_model=64, layers=1)
    a = model.MvdNet(**kw)
    b = model.MvdNet(**kw)
    r = model.MvdNet(vocab=90, anisotropic=True,
                     d_model=64, layers=1)
    pa = dict(a.encoder.named_parameters())
    pb = dict(b.encoder.named_parameters())
    pr = dict(r.encoder.named_parameters())
    assert set(pa) == set(pb) == set(pr)
    assert all(torch.equal(pa[k], pb[k]) for k in pa), \
        'pretrained load not deterministic'
    assert any(not torch.equal(pa[k], pr[k]) for k in pa), \
        'pretrained trunk looks random'
    a.eval()
    img = torch.zeros(1, 3, 256, 512)
    tgt = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        logits = a(img, tgt)
    assert logits.shape == (1, 8, 90), logits.shape
    # constrain=False: the 90-token test vocab is a truncation of the
    # 133-token bekern VOCAB, so grammar masking (which emits ids >= 90)
    # cannot apply here; the KV-cached decode path itself is exercised.
    out = a.greedy_kv(img, max_len=4, constrain=False)
    assert out.shape[0] == 1 and out.shape[1] <= 4, out.shape
    print('PASS pretrained init: forward', tuple(logits.shape),
          '+ greedy_kv', tuple(out.shape))


def test_timm_trunk_loads_all_178():
    # T1.6a: all 18 timm LayerScale gamma vectors transfer ([C] -> [C,1,1])
    # so 178/178 trunk tensors load and nothing stays at the 1e-6 init.
    from torchvision.models import convnext_tiny
    feats = convnext_tiny().features
    n_loaded, n_skipped = model._load_timm_trunk_(feats)
    assert (n_loaded, n_skipped) == (178, 0), (n_loaded, n_skipped)
    scales = [p.detach() for n, p in feats.named_parameters()
              if n.endswith('layer_scale')]
    assert len(scales) == 18, len(scales)
    mean_abs = (sum(p.abs().mean().item() for p in scales)
                / len(scales))
    assert mean_abs > 1e-3, mean_abs
    print('PASS trunk load: %d loaded, %d skipped, '
          'layer_scale mean %.4f' % (n_loaded, n_skipped, mean_abs))


def test_encode_has_mem_norm_and_dc_centering():
    # T1.6a: spatial LayerNorm over d_model, and a per-channel
    # spatial-DC shift injected after the encoder must vanish from the
    # memory (no global strip-ID hashing via a DC fingerprint).
    # Seeded: the +2.5 float32 residual varies with random input
    # (measured 8.3e-06..1.6e-05 across seeds 0..9), so the test pins RNG.
    torch.manual_seed(0)
    net = model.MvdNet(vocab=90, d_model=32, layers=1)
    assert isinstance(net.mem_norm, torch.nn.LayerNorm), type(net.mem_norm)
    assert tuple(net.mem_norm.normalized_shape) == (32,)
    net.eval()
    img = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        ref = net.encode(img)
        hook = net.encoder.register_forward_hook(
            lambda m, i, o: o + 2.5)
        try:
            shifted = net.encode(img)
        finally:
            hook.remove()
    # 64x64 input -> h = 64/8, w = 64/16 (anisotropic 8x16 stride).
    assert ref.shape == shifted.shape == (1, 8 * 4, 32), ref.shape
    # atol=1e-4: float32 rounding through proj+LayerNorm leaves ~1e-5
    # residue; without DC-centering the +2.5 shift would surface at O(1),
    # so 1e-4 still proves the property with 4 orders of margin.
    assert torch.allclose(ref, shifted, atol=1e-4, rtol=1e-4), \
        (ref - shifted).abs().max().item()
    print('PASS encode: mem_norm present, DC-shift invariant')


def test_tok_dropout_content_only():
    # T1.6a: tok_dropout replaces ONLY content-id inputs with uniform
    # real ids, never BOS/EOS/PAD/bar/voice-*/header tokens; 0.0 is the
    # bitwise legacy path. Forward is spied to observe the dropped inp.
    import data as _data
    import train as _train
    torch.manual_seed(0)
    net = model.MvdNet(vocab=len(VOCAB) + 3, d_model=32, layers=1,
                       heads=4, dim_ff=64)
    net.eval()
    V = len(VOCAB)
    T = _data.TOKEN_TO_ID
    tgt = torch.tensor([[T['clef-G-2'], T['key+0'], T['time-4-4'],
                         T['quarter'], T['C'], T['4'], T['bar'],
                         T['voice-1'], T['rest'], T['dot'],
                         V + 2, V, V]])
    img = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        base = net.loss(img, tgt)
        same = net.loss(img, tgt, tok_dropout=0.0,
                        content_ids=_train.CONTENT_IDS)
    assert same == base, (same, base)
    captured = {}
    real_forward = net.forward

    def _spy(images, tgt_in, mem_mask=None):
        captured['inp'] = tgt_in.clone()
        return real_forward(images, tgt_in, mem_mask)

    net.forward = _spy
    real_randint = torch.randint
    torch.randint = (lambda lo, hi, shape, **kw:
                     torch.full(shape, 7, dtype=torch.long,
                                device=kw.get('device', 'cpu')))
    try:
        with torch.no_grad():
            dropped = net.loss(img, tgt, tok_dropout=1.0,
                               content_ids=_train.CONTENT_IDS)
    finally:
        torch.randint = real_randint
        net.forward = real_forward
    assert dropped != base, 'full content dropout must change the loss'
    inp = captured['inp']
    want = torch.cat([torch.full_like(tgt[:, :1], V + 1),
                      tgt[:, :-1]], 1)
    cids = set(_train.CONTENT_IDS)
    changed = 0
    for g, w in zip(inp[0].tolist(), want[0].tolist()):
        if w in cids:
            assert g == 7, (g, w)
            changed += (w != 7)
        else:
            assert g == w, (g, w)
    assert changed > 0, 'dropout replaced nothing observable'
    print('PASS tok_dropout: content-only (%d positions), 0.0 legacy exact'
          % changed)


class TestT14Anisotropic(unittest.TestCase):
    def test_anisotropic_encoder_spatial_resolution(self):
        test_anisotropic_encoder_spatial_resolution()

    def test_pretrained_init_loads_cleanly(self):
        test_pretrained_init_loads_cleanly()


class TestT16a(unittest.TestCase):
    def test_timm_trunk_loads_all_178(self):
        test_timm_trunk_loads_all_178()

    def test_encode_has_mem_norm_and_dc_centering(self):
        test_encode_has_mem_norm_and_dc_centering()

    def test_tok_dropout_content_only(self):
        test_tok_dropout_content_only()


def test_hybrid_heat_channels_forward():
    # T2.2: the extra_chans=8 stem is RGB-identical at init (extras
    # zero), so zero extra channels reproduce the 3-channel logits.
    # Live vocab (T1.6d extended VOCAB to 130 -> 133 with specials);
    # never hardcode the stale spec value 113.
    V = len(VOCAB) + 3
    assert V == VOCAB_SIZE, (V, VOCAB_SIZE)
    torch.manual_seed(0)
    net8 = model.MvdNet(vocab=V, d_model=64, layers=1, heads=4,
                        dim_ff=128, extra_chans=8)
    stem = net8.encoder[0][0]
    assert tuple(stem.weight.shape) == (96, 3 + 8, 4, 4), \
        stem.weight.shape
    assert torch.equal(stem.weight[:, 3:],
                       torch.zeros_like(stem.weight[:, 3:])), \
        'hybrid extras must be zero-initialized'
    # Identical shared weights: transplant net8's params into a 3-ch
    # net (RGB slice of the widened stem + every other tensor verbatim).
    st8 = net8.state_dict()
    st0 = {k: (v[:, :3].clone().contiguous()
               if k == 'encoder.0.0.weight' else v.clone())
           for k, v in st8.items()}
    net0 = model.MvdNet(vocab=V, d_model=64, layers=1, heads=4,
                        dim_ff=128, extra_chans=0)
    net0.load_state_dict(st0, strict=True)
    net8.eval()
    net0.eval()
    torch.manual_seed(1)
    rgb = torch.randn(2, 3, 256, 512)
    zeros8 = torch.zeros(2, 8, 256, 512)
    tgt = torch.randint(0, V, (2, 8))
    with torch.no_grad():
        l8 = net8(torch.cat([rgb, zeros8], dim=1), tgt)
        l0 = net0(rgb, tgt)
    assert l8.shape == (2, 8, V), l8.shape
    diff = (l8 - l0).abs().max().item()
    assert diff < 1e-5, diff
    print('PASS hybrid heat channels: stem (96, 11, 4, 4), '
          'zero-init extras, max logit diff %.2e' % diff)


def test_encode_masked_stats_ignore_masked_cells():
    # T2.4 v2.3: with mem_mask given, encode centers proj over active
    # cells only; None (or all-False) keeps the exact legacy mean.
    torch.manual_seed(0)
    net = model.MvdNet(vocab=40, d_model=32, layers=1, heads=4,
                       dim_ff=64)
    net.eval()
    h_mem, w_mem = 256 // 8, 128 // 16
    x = torch.randn(2, 3, 256, 128)
    x[0, :, 128:, :] = 5.0  # bright distractor in the masked half
    mask = torch.zeros(2, h_mem * w_mem, dtype=torch.bool)
    mask[0].reshape(h_mem, w_mem)[16:32, :] = True  # item0: S=2
    with torch.no_grad():
        m_none = net.encode(x)
        m_default = net.encode(x, None)
        assert torch.equal(m_none, m_default), \
            'None mask must keep the legacy path'
        m_clear = net.encode(x, torch.zeros_like(mask))
        assert torch.equal(m_none, m_clear), \
            'all-False mask must equal the legacy mean'
        m = net.encode(x, mask)
        # Independent recomputation from public submodules: masked
        # mean over active cells, then mem_norm + position code.
        proj = net.proj(net.encoder(x))
        active = (~mask).to(proj.dtype).reshape(2, 1, h_mem, w_mem)
        n = active.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        mu = (proj * active).sum(dim=(2, 3), keepdim=True) / n
        pos = model._sinusoidal_2d(h_mem, w_mem, 32, x.device)
        expected = net.mem_norm(
            (proj - mu).permute(0, 2, 3, 1).reshape(2, h_mem * w_mem, 32)
        ) + pos.reshape(1, h_mem * w_mem, 32)
        diff = (m - expected).abs().max().item()
        assert diff < 1e-6, diff
        # Sensitivity: the masked-out bright half must not move the
        # centering statistic.
        assert not torch.allclose(m, m_none, atol=1e-4, rtol=1e-4), \
            'mask had no effect on encode stats'
        # Forward threads mem_mask through to encode (shape smoke).
        tgt = torch.randint(0, 40, (2, 8))
        logits = net(x, tgt, mask)
        assert logits.shape == (2, 8, 40), logits.shape
    print('PASS encode: masked stats ignore masked cells')


class TestT22Hybrid(unittest.TestCase):
    def test_hybrid_heat_channels_forward(self):
        test_hybrid_heat_channels_forward()


class TestT23CollapseFixes(unittest.TestCase):
    def test_encode_masked_stats_ignore_masked_cells(self):
        test_encode_masked_stats_ignore_masked_cells()


class TestT24BeamSelect(unittest.TestCase):
    """T2.4a: closed beams rank by score; the closed flag needs a bar end.

    Old key ``(closed, score)`` let a 1-measure closed stub outrank a
    500-token open rollout on the closed flag alone. The new rank key
    sorts by score first and only credits closed beams whose stream
    ends at a bar boundary, so behavior is identical whenever every
    beam closes at a bar line.
    """

    EOS = len(VOCAB) + 2  # vocab = len(VOCAB)+3; PAD/BOS/EOS trail it

    def _ids(self, toks):
        return [data_mod.TOKEN_TO_ID[t] for t in toks]

    def test_ends_at_bar_true_only_with_trailing_bar(self):
        full = self._ids(['clef-G-2', 'key+0', 'time-4-4', 'quarter',
                          'C', '4', 'bar']) + [self.EOS]
        self.assertTrue(model._beam_ends_at_bar(full, self.EOS))
        mid = self._ids(['clef-G-2', 'key+0', 'time-4-4', 'quarter',
                         'C', '4']) + [self.EOS]
        self.assertFalse(model._beam_ends_at_bar(mid, self.EOS))
        self.assertFalse(model._beam_ends_at_bar([], self.EOS))
        self.assertFalse(model._beam_ends_at_bar([self.EOS], self.EOS))

    def test_short_closed_stub_loses_to_better_open_rollout(self):
        stub = self._ids(['clef-G-2', 'key+0', 'time-4-4', 'quarter',
                          'C', '4']) + [self.EOS]
        rollout = self._ids(
            ['clef-G-2', 'key+0', 'time-4-4'] + ['quarter', 'C', '4'] * 60
            + ['bar'])
        stub_key = model._beam_rank_key(-1.0, True, stub, self.EOS)
        open_key = model._beam_rank_key(-0.9, False, rollout, self.EOS)
        self.assertLess(stub_key, open_key)

    def test_closed_at_bar_still_wins_ties_and_ranks_by_score(self):
        at_bar = self._ids(['clef-G-2', 'key+0', 'time-4-4', 'quarter',
                            'C', '4', 'bar']) + [self.EOS]
        mid = self._ids(['clef-G-2', 'key+0', 'time-4-4', 'quarter',
                         'C', '4']) + [self.EOS]
        # Same score: bar-complete closed outranks mid-measure closed.
        self.assertLess(model._beam_rank_key(-1.0, True, mid, self.EOS),
                        model._beam_rank_key(-1.0, True, at_bar, self.EOS))
        # All closed at bar lines: pure score order (legacy behavior).
        lo = model._beam_rank_key(-2.0, True, at_bar, self.EOS)
        hi = model._beam_rank_key(-1.0, True, at_bar, self.EOS)
        self.assertLess(lo, hi)


def test_stride_grid_scale_values():
    # Reference frame is 8x16 (all pretrained addressing learned it).
    assert model.stride_grid_scale('8x16') == (1.0, 1.0)
    assert model.stride_grid_scale('4x8') == (0.5, 0.5)
    assert model.stride_grid_scale(None) == (1.0, 1.0)


def test_scaled_codes_match_overlap():
    # A 4x8 grid with 0.5/0.5 scaling must reproduce the 8x16 codes
    # exactly on overlapping physical pixels (even rows/cols).
    import torch
    d = 64
    fine = model._sinusoidal_2d(64, 80, d, 'cpu', 0.5, 0.5)
    coarse = model._sinusoidal_2d(32, 40, d, 'cpu')
    assert torch.equal(fine[::2, ::2], coarse), 'overlap mismatch'
    # ...and the naive (unscaled) grid must NOT match (else the
    # transfer bug would not exist).
    naive = model._sinusoidal_2d(64, 80, d, 'cpu')
    assert not torch.equal(naive[::2, ::2], coarse), \
        'naive unexpectedly matches'


class TestStrideGridAlignment(unittest.TestCase):
    def test_scale_values(self):
        test_stride_grid_scale_values()

    def test_scaled_overlap(self):
        test_scaled_codes_match_overlap()


if __name__ == '__main__':
    test_forward_shapes()
    test_long_sequence_no_position_cap()
    test_causal_no_peek()
    test_overfit_one_sample()
    test_anisotropic_encoder_spatial_resolution()
    test_pretrained_init_loads_cleanly()
    test_timm_trunk_loads_all_178()
    test_encode_has_mem_norm_and_dc_centering()
    test_tok_dropout_content_only()
    test_hybrid_heat_channels_forward()
    test_encode_masked_stats_ignore_masked_cells()
    print('ALL MODEL TESTS PASSED')
