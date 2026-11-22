"""Spike S9: test-time per-strip rank-8 adapter (zero retraining).

A strip's encoder memory ``mem [1, S, d]`` is remapped by a low-rank
linear map ``M = I + A @ B`` (``A [d, r]``, ``B [r, d]``), fit
unsupervised on the strip itself: one constrained greedy pass gives
pseudo-targets, then a few full-batch steps minimize the mean
next-token entropy of the teacher-forced decoder on those targets,
updating ONLY ``A, B`` (the trunk and decoder stay frozen and in
eval mode). ``B`` starts at zero so the adapter is exactly the
identity before fitting; ``A`` is drawn from a locally-seeded
generator (seed 0), so the same strip always yields bitwise
identical adapted memory. A fresh adapter is built per strip —
no state leaks across strips.
"""
import math

import torch
import torch.nn as nn


class StripAdapter(nn.Module):
    """Low-rank residual map ``mem + (mem @ A) @ B`` (spike S9)."""

    def __init__(self, d_model, rank=8):
        super().__init__()
        # Unit-gain init: A ~ N(0, 1/d) keeps mem@A at the memory's
        # own scale, so a conventional lr (0.05-0.2) moves the map in
        # 1-4 steps. (An LoRA-style 1e-3 scale needs lr ~ 100 to move
        # at all — probe-verified dead at lr=0.1.) B stays zero, so
        # the adapter is exactly identity before fitting.
        g = torch.Generator().manual_seed(0)
        a = torch.randn(d_model, rank, generator=g) / math.sqrt(d_model)
        self.A = nn.Parameter(a)
        self.B = nn.Parameter(torch.zeros(rank, d_model))

    def forward(self, mem):
        return mem + (mem @ self.A) @ self.B


def _pseudo_targets(net, images, mem_mask, max_len):
    """Constrained greedy ids ``[B, T]`` (PAD-padded past EOS)."""
    with torch.no_grad():
        return net.greedy(images, max_len=max_len, constrain=True,
                          mem_mask=mem_mask)


def _mean_entropy(logits, pseudo, pad_id):
    """Mean next-token entropy over pre-EOS positions (incl. EOS)."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    # Valid = every non-PAD position: greedy PAD-pads past EOS and
    # PAD never precedes EOS, so this covers exactly the pre-EOS
    # prefix including EOS itself. Normalize over valid cells.
    w = (pseudo != pad_id).to(ent.dtype)
    return (ent * w).sum() / w.sum().clamp_min(1.0)


def adapt_strip(net, images, mem_mask=None, steps=2, lr=0.1, rank=8,
                max_len=564):
    """Fit a fresh adapter on one strip; return adapted memory + stats.

    ``steps <= 0`` skips fitting and returns the plain encode
    (bitwise identical to no adapter). Otherwise: greedy
    pseudo-targets (no grad), then ``steps`` SGD steps on mean
    entropy, grads into ``A, B`` only. All work runs with the net
    in eval mode; the net's own params are never stepped.
    """
    was_training = net.training
    net.eval()
    # Freeze the net: grads flow only into A/B (also cheaper).
    frozen = [p for p in net.parameters() if p.requires_grad]
    for p in frozen:
        p.requires_grad_(False)
    try:
        with torch.no_grad():
            mem0 = net.encode(images, mem_mask)
        pseudo = _pseudo_targets(net, images, mem_mask, max_len)
        pseudo_len = int((pseudo[0] != net.pad_id).sum().item())
        if steps <= 0:
            return {'mem': mem0, 'entropy_before': 0.0,
                    'entropy_after': 0.0, 'pseudo_len': pseudo_len}
        adapter = StripAdapter(net.d_model, rank=rank).to(mem0)
        opt = torch.optim.SGD(adapter.parameters(), lr=lr)
        bos = torch.full_like(pseudo[:, :1], net.bos_id)
        tgt_in = torch.cat([bos, pseudo[:, :-1]], dim=1)
        ent_before = None
        with torch.enable_grad():
            for _ in range(steps):
                opt.zero_grad()
                logits = net.decode_logits(adapter(mem0), tgt_in,
                                           mem_mask)
                ent = _mean_entropy(logits, pseudo, net.pad_id)
                if ent_before is None:
                    ent_before = float(ent.detach().item())
                ent.backward()
                opt.step()
            with torch.no_grad():
                final = net.decode_logits(adapter(mem0), tgt_in,
                                          mem_mask)
                ent_after = float(
                    _mean_entropy(final, pseudo, net.pad_id).item())
        with torch.no_grad():
            adapted = adapter(mem0).detach()
        return {'mem': adapted, 'entropy_before': ent_before,
                'entropy_after': ent_after, 'pseudo_len': pseudo_len}
    finally:
        for p in frozen:
            p.requires_grad_(True)
        if was_training:
            net.train()
