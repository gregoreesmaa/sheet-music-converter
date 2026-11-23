"""MVD v1: image -> bekern token sequence (TrOMR/SMT-shaped, pure torch).

Encoder: torchvision ConvNeXt-Tiny feature trunk (no classifier head),
projected to d_model with a 2-D sinusoidal position code.
Decoder: vanilla nn.TransformerDecoder with a causal mask, greedy
inference helper for eval. No third-party model deps beyond torchvision.

Conventions (must match data.py):
  PAD_ID = len(VOCAB), BOS_ID = PAD_ID + 1, EOS_ID = PAD_ID + 2.
  loss(images, tgt) teacher-forces: input = [BOS, t0..t_{n-2}],
  target = [t0..t_{n-1}], PAD positions ignored.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny

import comer_gdcm
import data
import grammar
import s18_factored


def _sinusoidal_1d(n, dim, device):
    """[N, dim] sinusoidal position code (dim even), any length."""
    assert dim % 2 == 0
    q = dim // 2
    pos = torch.arange(n, device=device).float()
    div = torch.exp(torch.arange(q, device=device).float()
                    * -(math.log(10000.0) / q))
    out = torch.empty(n, dim, device=device)
    out[:, 0::2] = torch.sin(pos[:, None] * div)
    out[:, 1::2] = torch.cos(pos[:, None] * div)
    return out


def _sinusoidal_1d_pos(p, dim, device):
    """Position code for a single position p (equals row p above)."""
    assert dim % 2 == 0
    q = dim // 2
    div = torch.exp(torch.arange(q, device=device).float()
                    * -(math.log(10000.0) / q))
    pv = div * float(p)
    out = torch.empty(dim, device=device)
    out[0::2] = torch.sin(pv)
    out[1::2] = torch.cos(pv)
    return out


def _split_qkv(mha):
    """Split a same-embed-dim MultiheadAttention's packed in-projection.

    Returns (Wq, Wk, Wv, bq, bk, bv, out_w, out_b, num_heads) so the
    cached step below can project one query position and append one
    K/V row without re-projecting the history.
    """
    assert mha._qkv_same_embed_dim
    e = mha.embed_dim
    w = mha.in_proj_weight
    b = mha.in_proj_bias
    return (w[:e], w[e:2 * e], w[2 * e:],
            b[:e], b[e:2 * e], b[2 * e:],
            mha.out_proj.weight, mha.out_proj.bias, mha.num_heads)


def _layer_params(layer):
    """Pre-split QKV projections for one decoder layer (eval inference)."""
    return (_split_qkv(layer.self_attn),
            _split_qkv(layer.multihead_attn))


def _mem_kv(mem, xa):
    """Pre-project encoder memory to per-head cross-attention K/V once."""
    _, xk, xv, _, xbk, xbv, _, _, h = xa
    b, s, d = mem.shape
    dh = d // h
    km = (mem @ xk.t() + xbk).view(b, s, h, dh).transpose(1, 2)
    vm = (mem @ xv.t() + xbv).view(b, s, h, dh).transpose(1, 2)
    return km.contiguous(), vm.contiguous()


def _layer_step(layer, p, x, kc, vc, mem_kv, mem_mask=None):
    """One incremental decoder-layer step (norm_first=False branch).

    Mirrors ``nn.TransformerDecoderLayer.forward`` exactly: the same
    submodule calls in the same order (self-attn -> norm1 -> cross-attn
    -> norm2 -> FFN -> norm3, dropouts included and identity in eval).
    ``kc``/``vc`` are the [B, K, D] projected self-attention history
    (None on the first position); returns (out, kc_new, vc_new).
    ``mem_mask`` is an optional bool [B, S] cross-attention key mask
    (True = ignore, ``nn.MultiheadAttention`` key-padding convention,
    as built by the training collates). None runs the exact legacy
    unmasked path. The mask is applied as a float ``-inf`` additive
    bias -- the same merge ``nn.MultiheadAttention`` performs
    internally -- because SDPA bool masks read True = attend (the
    inverse convention), which a bare bool pass-through would flip.
    """
    (wq, wk, wv, bq, bk, bv, ow, ob, h), \
        (xq, _, _, xbq, _, _, xow, xob, _) = p
    b, _, d = x.shape
    dh = d // h
    q = (x @ wq.t() + bq).view(b, 1, h, dh).transpose(1, 2)
    k1 = x @ wk.t() + bk
    v1 = x @ wv.t() + bv
    kc = k1 if kc is None else torch.cat([kc, k1], 1)
    vc = v1 if vc is None else torch.cat([vc, v1], 1)
    kh = kc.view(b, -1, h, dh).transpose(1, 2)
    vh = vc.view(b, -1, h, dh).transpose(1, 2)
    a = F.scaled_dot_product_attention(q, kh, vh)
    sa = a.transpose(1, 2).reshape(b, 1, d) @ ow.t() + ob
    x = layer.norm1(x + layer.dropout1(sa))
    q2 = (x @ xq.t() + xbq).view(b, 1, h, dh).transpose(1, 2)
    km, vm = mem_kv
    if mem_mask is None:
        m = F.scaled_dot_product_attention(q2, km, vm)
    else:
        bias = torch.zeros(b, 1, 1, km.shape[2], device=x.device,
                           dtype=x.dtype)
        bias = bias.masked_fill(mem_mask[:, None, None, :],
                                float('-inf'))
        m = F.scaled_dot_product_attention(q2, km, vm, attn_mask=bias)
    m = m.transpose(1, 2).reshape(b, 1, d) @ xow.t() + xob
    x = layer.norm2(x + layer.dropout2(m))
    f = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    x = layer.norm3(x + layer.dropout3(f))
    return x, kc, vc


def _layer_step_comer(layer, p, x, kc, vc, mem_kv, mem_mask,
                      comer_bias, cum_cov, h_mem, w_mem):
    """One incremental decoder-layer step with CoMER coverage bias.

    Bit-consistent with :func:`_decoder_layer_comer`: computes raw
    cross-attention weights from ``q2``/``km``, injects
    ``comer_bias.step_coverage_bias(cum_cov, h_mem, w_mem)`` into the
    cross-attention logits, and accumulates ``raw_attn`` into
    ``cum_cov``. When ``comer_bias.conv_out`` is zero-initialized,
    step 0 is mathematically identical to :func:`_layer_step`.
    """
    (wq, wk, wv, bq, bk, bv, ow, ob, h), \
        (xq, _, _, xbq, _, _, xow, xob, _) = p
    b, _, d = x.shape
    dh = d // h
    q = (x @ wq.t() + bq).view(b, 1, h, dh).transpose(1, 2)
    k1 = x @ wk.t() + bk
    v1 = x @ wv.t() + bv
    kc = k1 if kc is None else torch.cat([kc, k1], 1)
    vc = v1 if vc is None else torch.cat([vc, v1], 1)
    kh = kc.view(b, -1, h, dh).transpose(1, 2)
    vh = vc.view(b, -1, h, dh).transpose(1, 2)
    a = F.scaled_dot_product_attention(q, kh, vh)
    sa = a.transpose(1, 2).reshape(b, 1, d) @ ow.t() + ob
    x = layer.norm1(x + layer.dropout1(sa))
    q2 = (x @ xq.t() + xbq).view(b, 1, h, dh).transpose(1, 2)
    km, vm = mem_kv
    scores = (q2.float() @ km.float().transpose(-2, -1)) / math.sqrt(dh)
    if mem_mask is not None:
        scores = scores.masked_fill(mem_mask[:, None, None, :], float('-inf'))
    raw_attn = scores.softmax(dim=-1)
    cov_bias = comer_bias.step_coverage_bias(
        cum_cov, h_mem, w_mem, mem_mask=mem_mask, detach_cov=True)
    biased = scores + cov_bias.float()
    if mem_mask is not None:
        biased = biased.masked_fill(mem_mask[:, None, None, :], float('-inf'))
    attn_w = biased.softmax(dim=-1)
    m = (attn_w.to(vm.dtype) @ vm).transpose(1, 2).reshape(b, 1, d) @ xow.t() + xob
    new_cov = comer_bias.update_coverage(
        cum_cov, raw_attn.to(cum_cov.dtype), detach=True)
    x = layer.norm2(x + layer.dropout2(m))
    f = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    x = layer.norm3(x + layer.dropout3(f))
    return x, kc, vc, new_cov


def _comer_xattn_chunk(qq_c, kk, vv, cov_in_c, mem_mask, comer_bias,
                       h_mem, w_mem, dh, drop_p, training):
    """One query-chunk of CoMER cross-attention (checkpoint-friendly)."""
    b, nh, tc, _ = qq_c.shape
    s = kk.shape[2]
    scores_c = (qq_c.float() @ kk.float().transpose(-2, -1)) / math.sqrt(dh)
    cov_2d = cov_in_c.transpose(1, 2).reshape(b * tc, nh, h_mem, w_mem)
    cov_bias_c = comer_bias(cov_2d).reshape(b, tc, nh, s).transpose(1, 2)
    biased_c = scores_c + cov_bias_c.float()
    if mem_mask is not None:
        biased_c = biased_c.masked_fill(mem_mask[:, None, None, :], float('-inf'))
    attn_w_c = biased_c.softmax(dim=-1)
    if drop_p > 0.0 and training:
        attn_w_c = F.dropout(attn_w_c, p=drop_p, training=True)
    return attn_w_c.to(vv.dtype) @ vv


def _decoder_layer_comer(layer, x, mem, tgt_mask, mem_mask,
                         comer_bias, h_mem, w_mem, t_chunk=64):
    """Batched teacher-forced decoder layer with CoMER shifted coverage bias.

    Uses ``layer.self_attn`` on the fast SDPA path and streams cross-attention
    in query chunks of ``t_chunk`` (checkpointed during training) so long/wide
    strips (``T=768, W=3072``) never materialize a full ``[B, H, T, S]`` tensor.
    Step ``t`` sees cumulative raw attention over strictly past steps
    ``0..t-1``, matching :func:`_layer_step_comer` step-for-step.
    """
    sa = layer.self_attn(x, x, x, attn_mask=tgt_mask, need_weights=False)[0]
    x = layer.norm1(x + layer.dropout1(sa))
    attn = layer.multihead_attn
    wq, wk, wv, bq, bk, bv, ow, ob, nh = _split_qkv(attn)
    b, t, d = x.shape
    s = mem.shape[1]
    dh = d // nh
    qq = (x @ wq.t() + bq).view(b, t, nh, dh).transpose(1, 2)
    kk = (mem @ wk.t() + bk).view(b, s, nh, dh).transpose(1, 2)
    vv = (mem @ wv.t() + bv).view(b, s, nh, dh).transpose(1, 2)

    use_ckpt = layer.training and torch.is_grad_enabled()
    running_cov = torch.zeros((b, nh, 1, s), device=x.device, dtype=torch.float32)
    out_chunks = []
    scale = 1.0 / math.sqrt(dh)
    for t0 in range(0, t, t_chunk):
        t1 = min(t0 + t_chunk, t)
        qq_c = qq[:, :, t0:t1, :]
        # Detached shifted cumulative coverage for this chunk (zero autograd RAM).
        with torch.no_grad():
            raw_scores = (qq_c.float() @ kk.float().transpose(-2, -1)) * scale
            if mem_mask is not None:
                raw_scores = raw_scores.masked_fill(
                    mem_mask[:, None, None, :], float('-inf'))
            raw_c = raw_scores.softmax(dim=-1)
            if mem_mask is not None:
                raw_c = raw_c.masked_fill(mem_mask[:, None, None, :], 0.0)
            tc = t1 - t0
            if tc == 1:
                cov_in_c = running_cov.to(x.dtype)
            else:
                csum_prev = torch.cumsum(raw_c[:, :, :-1, :], dim=2)
                cov_in_c = torch.cat(
                    [running_cov, running_cov + csum_prev], dim=2
                ).to(x.dtype)
            running_cov = running_cov + raw_c.sum(dim=2, keepdim=True)
        if use_ckpt:
            out_c = torch.utils.checkpoint.checkpoint(
                _comer_xattn_chunk,
                qq_c, kk, vv, cov_in_c, mem_mask, comer_bias,
                h_mem, w_mem, dh, attn.dropout, layer.training,
                use_reentrant=False,
            )
        else:
            out_c = _comer_xattn_chunk(
                qq_c, kk, vv, cov_in_c, mem_mask, comer_bias,
                h_mem, w_mem, dh, attn.dropout, layer.training,
            )
        out_chunks.append(out_c)
    m = torch.cat(out_chunks, dim=2).transpose(1, 2).reshape(b, t, d) @ ow.t() + ob
    x = layer.norm2(x + layer.dropout2(m))
    f = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    x = layer.norm3(x + layer.dropout3(f))
    return x


def stride_grid_scale(stride):
    """(y_scale, x_scale) mapping a stride's grid onto the 8x16 frame.

    All pretrained cross-attention addressing learned the 8x16
    coordinate frame (cell (i, j) = pixel (8i, 16j)). Denser grids
    must scale their arange indices so each fine cell's code equals
    the overlapping coarse cell's code (strict refinement);
    otherwise the decoder addresses the wrong physical pixels
    (S13-warm transfer bug: 4x8 is off 2x in y and 2x in x).
    """
    if stride is None or stride == '8x16':
        return (1.0, 1.0)
    sh, sw = stride.split('x')
    return (float(sh) / 8.0, float(sw) / 16.0)


def _sinusoidal_2d(h, w, dim, device, y_scale=1.0, x_scale=1.0):
    """[H, W, dim] 2-D sinusoidal position code (dim divisible by 4).

    ``y_scale``/``x_scale`` multiply the raw grid indices, mapping
    denser grids onto the 8x16 reference frame (see
    :func:`stride_grid_scale`). Defaults preserve the exact legacy
    codes.
    """
    assert dim % 4 == 0
    q = dim // 4
    ys = torch.arange(h, device=device).float() * y_scale
    xs = torch.arange(w, device=device).float() * x_scale
    div = torch.exp(torch.arange(q, device=device).float()
                    * -(math.log(10000.0) / q))
    ey = torch.zeros(h, w, dim, device=device)
    ex = torch.zeros(h, w, dim, device=device)
    ey[:, :, 0:q] = torch.sin(ys[:, None, None] * div).expand(h, w, q)
    ey[:, :, q:2 * q] = torch.cos(ys[:, None, None] * div).expand(h, w, q)
    ex[:, :, 2 * q:3 * q] = torch.sin(xs[None, :, None] * div).expand(h, w, q)
    ex[:, :, 3 * q:] = torch.cos(xs[None, :, None] * div).expand(h, w, q)
    return ey + ex


class _PadConv(nn.Module):
    """Zero-pad the input, then apply the wrapped conv.

    Gives exact output sizes for the anisotropic downsamples below:
    a 2x2 kernel with stride (1, 2) drops one row without the bottom
    pad, and stride (1, 1) drops one row and column without padding.
    The wrapped conv object (and its weight shapes) is untouched, so
    pretrained trunk weights load with no shape mismatch.
    """

    def __init__(self, conv, pad):
        super().__init__()
        self.conv = conv
        self.pad = pad

    def forward(self, x):
        if self.pad is not None:
            x = F.pad(x, self.pad)
        return self.conv(x)


def _timm_to_tv_pairs():
    """(timm_key, torchvision-features-key) pairs for ConvNeXt-Tiny.

    timm stage s maps to torchvision features (2*s+1) for blocks and
    (2*s) for the s>0 downsamples; block parts map conv_dw->block.0,
    norm->block.2, mlp.fc1->block.3, mlp.fc2->block.5. timm-only
    tensors (LayerScale ``gamma`` -> ``layer_scale``, [C] reshaped to
    [C, 1, 1] by the loader; only ``head`` tensors are skipped.
    """
    pairs = [('stem.0.weight', '0.0.weight'),
             ('stem.0.bias', '0.0.bias'),
             ('stem.1.weight', '0.1.weight'),
             ('stem.1.bias', '0.1.bias')]
    nblocks = (3, 3, 9, 3)
    for s, n in enumerate(nblocks):
        if s > 0:
            pairs += [(f'stages.{s}.downsample.0.weight',
                       f'{2 * s}.0.weight'),
                      (f'stages.{s}.downsample.0.bias',
                       f'{2 * s}.0.bias'),
                      (f'stages.{s}.downsample.1.weight',
                       f'{2 * s}.1.weight'),
                      (f'stages.{s}.downsample.1.bias',
                       f'{2 * s}.1.bias')]
        for i in range(n):
            pairs += [(f'stages.{s}.blocks.{i}.conv_dw.weight',
                       f'{2 * s + 1}.{i}.block.0.weight'),
                      (f'stages.{s}.blocks.{i}.conv_dw.bias',
                       f'{2 * s + 1}.{i}.block.0.bias'),
                      (f'stages.{s}.blocks.{i}.norm.weight',
                       f'{2 * s + 1}.{i}.block.2.weight'),
                      (f'stages.{s}.blocks.{i}.norm.bias',
                       f'{2 * s + 1}.{i}.block.2.bias'),
                      (f'stages.{s}.blocks.{i}.mlp.fc1.weight',
                       f'{2 * s + 1}.{i}.block.3.weight'),
                      (f'stages.{s}.blocks.{i}.mlp.fc1.bias',
                       f'{2 * s + 1}.{i}.block.3.bias'),
                      (f'stages.{s}.blocks.{i}.mlp.fc2.weight',
                       f'{2 * s + 1}.{i}.block.5.weight'),
                      (f'stages.{s}.blocks.{i}.mlp.fc2.bias',
                       f'{2 * s + 1}.{i}.block.5.bias'),
                      (f'stages.{s}.blocks.{i}.gamma',
                       f'{2 * s + 1}.{i}.layer_scale')]
    return pairs


def _load_timm_trunk_(features, model_name='convnext_tiny.in12k_ft_in1k'):
    """Copy timm pretrained trunk weights into torchvision features.

    Same-shape pairs transfer directly; tensors with equal numel but
    different shape are reshaped (timm 1D gamma -> [C, 1, 1]).
    Anything missing or still mismatched is skipped.
    Returns (n_loaded, n_skipped).
    """
    import timm
    src = timm.create_model(model_name, pretrained=True).state_dict()
    own = dict(features.state_dict())
    n_loaded = n_skipped = 0
    for sk, dk in _timm_to_tv_pairs():
        if sk in src and dk in own:
            t = src[sk]
            if t.shape != own[dk].shape and t.numel() == own[dk].numel():
                t = t.reshape(own[dk].shape)
            if t.shape == own[dk].shape:
                own[dk] = t.clone()
                n_loaded += 1
                continue
            n_skipped += 1
        else:
            n_skipped += 1
    features.load_state_dict(own, strict=True)
    return n_loaded, n_skipped


def _beam_ends_at_bar(hist, eos_id):
    """True iff a beam's token stream ends at a bar boundary.

    Special ids (EOS/PAD/BOS live past ``len(data.VOCAB)``) are
    filtered, so a closed beam's trailing EOS never masks the last
    content token. An empty stream (immediate EOS) is not bar-complete.
    Pure read of ``hist``; used to gate the closed-first beam bonus.
    """
    toks = [data.VOCAB[int(i)] for i in hist
            if 0 <= int(i) < len(data.VOCAB)]
    return bool(toks) and toks[-1] == grammar.BAR


#: Bekern modifier tokens (augmentation dots, ties/slurs, articulations,
#: grace) used by live evaluation metrics and weighted risk losses.
MODIFIER_TOKENS = ('dot', 'tie-start', 'tie-stop', 'slur-start',
                   'slur-stop', 'stacc', 'accent', 'tenuto', 'fermata',
                   'grace')

#: Compat alias for ``eval_sys_gate.py`` (MODIFIER_TOKENS).
ILME_MODIFIERS = MODIFIER_TOKENS

#: Id-level modifier set used by S4 MWER / S7 keystroke action loss.
MODIFIER_IDS = frozenset(data.VOCAB.index(t) for t in MODIFIER_TOKENS
                         if t in data.VOCAB)

#: Voice-token ids (`voice-1` .. `voice-12`): each starts a new staff
#: system whose x-coordinates restart at the left margin.
VOICE_IDS = frozenset(data.VOCAB.index(t) for t in data.VOCAB
                      if t.startswith('voice-'))


def coord_monotonic_penalty(xc, tgt, voice_ids, pad_id, margin=0.0):
    """Mean backward-jump penalty on x-centroids (S13c).

    ``xc`` is ``[B, T]`` normalized (``[0, 1]``) cross-attention
    x-centroids, one per target position; ``tgt`` is ``[B, T]`` token
    ids. Music reads left-to-right within a voice, so a centroid that
    moves left is attention drift (the SER insertion failure mode).
    Voice tokens reset the segment (new staff, x restarts); ``pad_id``
    positions never form pairs. Penalty per valid adjacent pair is
    ``relu(prev - cur + margin)``, averaged over valid pairs (0 when
    there are none). Pure function; no RNG.
    """
    vid = torch.tensor(sorted(voice_ids), device=tgt.device)
    is_voice = torch.isin(tgt, vid)
    valid = (tgt != pad_id) & ~is_voice
    seg = torch.cumsum(is_voice.long(), dim=1)
    prev_ok = valid[:, :-1] & valid[:, 1:] & (seg[:, :-1] == seg[:, 1:])
    if not bool(prev_ok.any()):
        return xc.new_zeros(())
    drop = torch.relu(xc[:, :-1] - xc[:, 1:] + margin)
    return (drop * prev_ok.to(drop.dtype)).sum() / prev_ok.sum()


def attention_spread(w):
    """Normalized attention entropy per position (S13c/S17 slice 2).

    ``w`` is ``[B, T, S]`` head-averaged cross-attention weights;
    returns ``[B, T]`` in ``[0, 1]`` (``1`` = uniform over all ``S``
    cells, ``0`` = single-cell focus). Masked (zero-mass) cells
    contribute ``0``. Pure function.
    """
    safe = torch.where(w > 0, w, torch.ones((), device=w.device,
                                            dtype=w.dtype))
    ent = -(w * safe.log()).sum(dim=-1)
    return ent / math.log(w.shape[-1])


def _beam_rank_key(score, closed, hist, eos_id):
    """Beam selection key, higher = better (T2.4a).

    Ranks by length-normalized score FIRST; the closed flag only counts
    when the beam ends at a bar boundary. A 1-measure closed stub can
    no longer outrank a 500-token open rollout on the closed flag
    alone, while all-closed-at-bar selections reduce to pure score
    order (legacy behavior).
    """
    return (float(score),
            1 if (closed and _beam_ends_at_bar(hist, eos_id)) else 0)


def _cross_weights(attn, q, mem, key_padding_mask=None):
    """Head-averaged cross-attention weights from a decoder layer.

    ``q`` is the layer's cross-attention input (post-``norm2``)
    ``[B, T, E]``; ``mem`` is the encoder memory ``[B, S, E]``.
    Recomputes standard scaled dot-product weights with the layer's
    own ``in_proj`` parameters (torch hardcodes ``need_weights=False``
    in ``TransformerDecoderLayer``, so this is the only way to observe
    them). Only the default ``qkv_same_embed_dim`` layout without
    ``bias_k``/``bias_v``/``add_zero_attn`` is supported — anything
    else fails loud.
    """
    if (not attn._qkv_same_embed_dim or attn.bias_k is not None
            or attn.bias_v is not None or attn.add_zero_attn):
        raise ValueError('unsupported MultiheadAttention layout for '
                         'weight recovery')
    e, nh = attn.embed_dim, attn.num_heads
    dh = e // nh
    b, t, _ = q.shape
    s = mem.shape[1]
    w_q, w_k, _ = attn.in_proj_weight.chunk(3, dim=0)
    if attn.in_proj_bias is not None:
        b_q, b_k, _ = attn.in_proj_bias.chunk(3, dim=0)
    else:
        b_q = b_k = torch.zeros(e, device=q.device, dtype=q.dtype)
    qq = (q.float() @ w_q.t().float()
          + b_q.float()).view(b, t, nh, dh).transpose(1, 2)
    kk = (mem.float() @ w_k.t().float()
          + b_k.float()).view(b, s, nh, dh).transpose(1, 2)
    scores = qq @ kk.transpose(-2, -1) / math.sqrt(dh)
    if key_padding_mask is not None:
        scores = scores.masked_fill(
            key_padding_mask[:, None, None, :], float('-inf'))
    return scores.softmax(dim=-1).mean(dim=1)  # [B, T, S]


class MvdNet(nn.Module):
    def __init__(self, vocab, d_model=256, layers=4, heads=8,
                 dim_ff=1024, dropout=0.1, anisotropic=True,
                 pretrained=False, extra_chans: int = 0,
                 stride: str = '8x16', foveal_tap: bool = False,
                 use_comer: bool = False, use_gdcm: bool = False,
                 use_s18: bool = False):
        # NOTE: no max_tgt — target positions are runtime sinusoids
        # (corpus pages reach 4414 tokens; a fixed table crashes).
        # anisotropic=True (v2): 8x vertical / 16x horizontal encoder
        # stride, so H=256 yields 32 feature rows. False keeps the
        # legacy stride-32 trunk for old checkpoints (e.g. ckpt-13000).
        # pretrained=True inits the trunk from timm ConvNeXt-Tiny
        # weights (Apache-2.0 release) instead of random init.
        # stride='4x8' (S13) halves both axes (H=256 -> 64 rows) for
        # denser visual grounding; only stride/padding hyperparameters
        # move, so timm weight shapes are unchanged. Default '8x16'
        # keeps the exact legacy path.
        # foveal_tap=True taps Stage-0 (P2, 4x4 resolution, 96 ch) via
        # avg+max pooling into the final memory grid so 2-4 px
        # micro-glyphs (dots, accidentals, 128th flags) survive
        # downsampling. Default False adds zero parameters (ckpt-22000
        # state_dict compatible).
        super().__init__()
        if stride not in ('8x16', '4x8'):
            raise ValueError('stride must be 8x16 or 4x8, got %r' % stride)
        if not anisotropic and stride != '8x16':
            raise ValueError('stride is only valid with anisotropic=True')
        self.vocab = vocab
        self.pad_id = vocab - 3
        self.bos_id = vocab - 2
        self.eos_id = vocab - 1
        self.d_model = d_model
        self.heads = heads
        self.anisotropic = anisotropic
        self.stride = stride
        self.pretrained = pretrained
        self.extra_chans = int(extra_chans or 0)
        self.foveal_tap = bool(foveal_tap)
        self.use_comer = bool(use_comer)
        self.use_gdcm = bool(use_gdcm)
        self.use_s18 = bool(use_s18)
        self._last_hw_mem = None

        trunk = convnext_tiny()
        self.encoder = trunk.features  # [B, 768, H/8, W/16] if
        # anisotropic, else legacy [B, 768, H/32, W/32]
        if pretrained:
            n_loaded, n_skipped = _load_timm_trunk_(self.encoder)
            assert n_loaded > 0, 'timm trunk transfer loaded 0 params'
        if self.extra_chans > 0:
            # T2.2 hybrid: widen the stem to 3 + extra_chans input
            # channels (RGB + HeatNet 7-ch + staff grid). RGB weights
            # and bias carry over; extras are zero-init so step 0 is
            # bitwise identical to the 3-channel stem on zero extras.
            self._expand_stem(self.extra_chans)
        if anisotropic:
            if stride == '4x8':
                self._apply_stride_4x8()
            else:
                self._apply_anisotropic_stride()
        self.proj = nn.Conv2d(768, d_model, 1)
        if self.foveal_tap:
            # P2 foveal tap: avg+max pool Stage-0 [B, 96, H/4, W/4] to
            # [B, 192, h, w] -> residual projection into d_model.
            # Zero-initialized final conv makes step 0 bit-identical to
            # the pre-tap checkpoint when warm-starting.
            f_out = nn.Conv2d(d_model, d_model, 1)
            nn.init.zeros_(f_out.weight)
            nn.init.zeros_(f_out.bias)
            self.foveal_proj = nn.Sequential(
                nn.Conv2d(192, d_model, 1), nn.GELU(), f_out)
        self.mem_norm = nn.LayerNorm(d_model)
        # S13b dense auxiliary head: distills the hybrid-extra channels
        # (8 in 11-ch HeatNet+staff mode; 1 in 4-ch staff-only mode; 8 in
        # legacy 3-ch mode for checkpoint compatibility) from raw trunk
        # features. Unused when the train loop passes no aux targets.
        aux_ch = 1 if self.extra_chans == 1 else 8
        self.aux_head = nn.Sequential(
            nn.Conv2d(768, 128, 1), nn.ReLU(), nn.Conv2d(128, aux_ch, 1))

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.out = nn.Linear(d_model, vocab)
        if self.use_gdcm:
            self.gdcm_head = comer_gdcm.GlobalDynamicCountingHead(
                d_model=d_model)
        if self.use_comer:
            self.comer_bias = comer_gdcm.CoverageAttentionBias(n_heads=heads)
        if self.use_s18:
            self.s18_head = s18_factored.S18FactoredHead(
                d_model=d_model, vocab_size=vocab,
                consistency_weight=0.0, eos_id=self.eos_id)

    def _expand_stem(self, extra):
        """Widen the stem conv to ``3 + extra`` input channels.

        Copies the RGB ``[:, :3, :, :]`` weights and bias from the
        3-channel stem and zero-initializes ``[:, 3:, :, :]``. Only the
        input-channel axis changes; kernel, stride, padding, and the
        output width (96) are untouched, so timm/anisotropic handling
        around this call is unaffected.
        """
        old = self.encoder[0][0]
        new = nn.Conv2d(3 + extra, old.out_channels,
                        kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, dilation=old.dilation,
                        groups=old.groups,
                        bias=(old.bias is not None),
                        padding_mode=old.padding_mode)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3] = old.weight
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.encoder[0][0] = new

    def _apply_anisotropic_stride(self):
        """Retune trunk strides to 8x vertical / 16x horizontal.

        Stem (4, 4) and stage-2 downsample (2, 2) are already correct
        (cumulative 8x8) and are set explicitly. Stage-3 downsample
        goes (2, 2) -> (1, 2) and stage-4 (2, 2) -> (1, 1), keeping
        the 2x2 kernels (so pretrained weights still match by shape);
        a 1-px zero pad on the shrinking axes keeps the spatial size
        exact (H/8, W/16). Only stride/padding hyperparameters move —
        no parameter tensors are created or destroyed.
        """
        feats = self.encoder
        feats[0][0].stride = (4, 4)
        feats[2][1].stride = (2, 2)
        norm3, conv3 = feats[4][0], feats[4][1]
        conv3.stride = (1, 2)
        feats[4] = nn.Sequential(
            norm3, _PadConv(conv3, (0, 0, 0, 1)))
        norm4, conv4 = feats[6][0], feats[6][1]
        conv4.stride = (1, 1)
        feats[6] = nn.Sequential(
            norm4, _PadConv(conv4, (0, 1, 0, 1)))

    def _apply_stride_4x8(self):
        """Retune trunk strides to 4x vertical / 8x horizontal (S13).

        Stem (4, 4) is untouched; stage-2 downsample goes (2, 2) ->
        (1, 2) (bottom 1-px pad keeps H/4 exact), stage-3 goes (1, 2) ->
        (1, 1) (right + bottom pad keeps both axes exact), stage-4 stays
        (1, 1). Cumulative: vertical 4*1*1*1 = 4, horizontal 4*2*1*1 =
        8. Like :meth:`_apply_anisotropic_stride`, only stride/padding
        hyperparameters move — no parameter tensors are created or
        destroyed, so timm pretrained weights load with no mismatch.
        """
        feats = self.encoder
        feats[0][0].stride = (4, 4)
        feats[2][1].stride = (1, 2)
        feats[2] = nn.Sequential(
            feats[2][0], _PadConv(feats[2][1], (0, 0, 0, 1)))
        norm3, conv3 = feats[4][0], feats[4][1]
        conv3.stride = (1, 1)
        feats[4] = nn.Sequential(
            norm3, _PadConv(conv3, (0, 1, 0, 1)))
        norm4, conv4 = feats[6][0], feats[6][1]
        conv4.stride = (1, 1)
        feats[6] = nn.Sequential(
            norm4, _PadConv(conv4, (0, 1, 0, 1)))

    def _encode_with_feats(self, images, mem_mask=None, return_counts=False):
        """Encode images once, returning ``(mem_seq, raw_feats, h_mem, w_mem)``.

        When ``self.use_gdcm`` is enabled, predicts 6 strip-level counts from
        ``mem_seq`` and adds the zero-init ``count_to_mem_cond(pred_counts)``
        conditioning vector (100% identity at step 0 on warm-start). When
        ``return_counts=True``, appends ``pred_counts`` (or ``None`` if GDCM
        is disabled) to the returned tuple.
        """
        if mem_mask is not None and not mem_mask.any():
            mem_mask = None
        if self.foveal_tap:
            p2 = self.encoder[:2](images)     # [B, 96, H/4, W/4]
            feats = self.encoder[2:](p2)      # [B, 768, h, w]
            proj = self.proj(feats)           # [B, d, h, w]
            h, w = proj.shape[2], proj.shape[3]
            p2_pool = torch.cat([
                F.adaptive_avg_pool2d(p2, (h, w)),
                F.adaptive_max_pool2d(p2, (h, w)),
            ], dim=1)                         # [B, 192, h, w]
            proj = proj + self.foveal_proj(p2_pool)
        else:
            feats = self.encoder(images)      # [B, 768, h, w]
            proj = self.proj(feats)           # [B, d, h, w]
        if mem_mask is None:
            proj = proj - proj.mean(dim=(2, 3), keepdim=True)
        else:
            b, _, h, w = proj.shape
            active = (~mem_mask).to(proj.dtype).reshape(b, 1, h, w)
            n = active.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            proj = proj - (proj * active).sum(dim=(2, 3),
                                              keepdim=True) / n
        b, d, h, w = proj.shape
        self._last_hw_mem = (h, w)
        y_scale, x_scale = stride_grid_scale(self.stride)
        pos = _sinusoidal_2d(h, w, d, proj.device, y_scale, x_scale)
        seq = self.mem_norm(proj.permute(0, 2, 3, 1).reshape(b, h * w, d)) + pos.reshape(1, h * w, d)
        pred_counts = None
        if self.use_gdcm:
            pred_counts = self.gdcm_head(seq, mem_mask=mem_mask)
            seq = seq + self.gdcm_head.count_to_mem_cond(pred_counts)
        if return_counts:
            return seq, feats, h, w, pred_counts
        return seq, feats, h, w

    def encode(self, images, mem_mask=None):
        """Encode images to a ``[B, h * w, d]`` memory sequence.

        ``mem_mask`` is an optional bool ``[B, h * w]``
        memory_key_padding_mask (True = ignore), as built by the
        train collates. When given, the spatial DC-centering mean is
        computed over active (non-masked) cells only: width padding
        and inactive staff-slot paper must not drag the mean or leak
        padding geometry into the memory. None (default) keeps the
        exact legacy full-spatial mean.

        When ``self.foveal_tap`` is enabled, Stage-0 (P2, 4x4) features
        are avg+max pooled to ``(h, w)`` and added as a residual before
        DC-centering so micro-glyph peaks survive deep downsampling.
        """
        return self._encode_with_feats(images, mem_mask)[0]

    def aux_pred(self, images):
        """S13b: ``[B, 8, h, w]`` dense keypoint/ink logits.

        Two-layer 1x1 head on raw trunk features (pre-``proj``), so
        gradients reach the earliest trunk blocks densely. Grid
        follows the trunk stride (``H // 8`` rows at 8x16, ``H // 4``
        at 4x8). Input is the full model input (``3 + extra_chans``
        channels, 11 in hybrid mode).
        """
        return self.aux_head(self.encoder(images))

    def x_centroids(self, images, tgt, mem_mask=None, layer=-1,
                    with_spread=False):
        """S13c: ``[B, T]`` normalized cross-attention x-centroids.

        Runs the teacher-forced decoder pass (BOS-shifted ``tgt``, same
        alignment as :meth:`loss`) and returns, per target position,
        the head-averaged top-layer cross-attention centroid over
        memory columns in ``[0, 1]``. Weights are recomputed from the
        top layer's own parameters (torch hardcodes
        ``need_weights=False`` in decoder layers, so hooks can never
        observe them); masked columns carry no mass. The query tap is
        a temporary hook, removed before return.
        """
        _, _, h_in, w_in = images.shape
        if not self.anisotropic:
            h_mem, w_mem = h_in // 32, w_in // 32
        elif self.stride == '4x8':
            h_mem, w_mem = h_in // 4, w_in // 8
        else:
            h_mem, w_mem = h_in // 8, w_in // 16
        mem = self.encode(images, mem_mask)
        assert mem.shape[1] == h_mem * w_mem, (mem.shape, h_mem, w_mem)
        held = {}

        def hook(mod, inp, out):
            held['q'] = out.detach().float()

        top = self.decoder.layers[layer]
        handle = top.norm2.register_forward_hook(hook)
        try:
            bos = torch.full_like(tgt[:, :1], self.bos_id)
            inp = torch.cat([bos, tgt[:, :-1]], dim=1)
            n = inp.shape[1]
            emb = self.tok_emb(inp) + _sinusoidal_1d(
                n, self.d_model, inp.device)[None]
            mask = nn.Transformer.generate_square_subsequent_mask(
                n, device=inp.device)
            self.decoder(emb, mem, tgt_mask=mask,
                         memory_key_padding_mask=mem_mask)
        finally:
            handle.remove()
        w = _cross_weights(top.multihead_attn, held['q'], mem.float(),
                           key_padding_mask=mem_mask)
        b, t, _ = w.shape
        xs = (torch.arange(w_mem, device=w.device).float() + 0.5) / w_mem
        xc = (w.reshape(b, t, h_mem, w_mem) * xs).sum(dim=(2, 3))
        if with_spread:
            return xc, attention_spread(w)
        return xc

    def forward(self, images, tgt_in, mem_mask=None):
        """Logits [B, T, V], one per input position (causal).

        ``mem_mask`` is an optional bool [B, S] memory_key_padding_mask
        (True = ignore) for batches with width-padded images.
        """
        mem, _, h_mem, w_mem = self._encode_with_feats(images, mem_mask)
        return self.out(self.decode_hidden(
            mem, tgt_in, mem_mask, hw_mem=(h_mem, w_mem)))

    def decode_hidden(self, mem, tgt_in, mem_mask=None, hw_mem=None):
        """Decoder-only hidden states ``[B, T, d_model]`` for precomputed memory.

        Shared by :meth:`decode_logits` and :meth:`loss_shared` so
        factored heads (e.g. ``S18FactoredHead``) can share one decoder
        pass with ``self.out(h)``. ``tgt_in`` includes the BOS prefix.
        When ``self.use_comer=False``, runs the untouched fast SDPA
        ``self.decoder`` path.
        """
        n = tgt_in.shape[1]
        emb = self.tok_emb(tgt_in) + _sinusoidal_1d(
            n, self.d_model, tgt_in.device)[None]
        mask = nn.Transformer.generate_square_subsequent_mask(
            n, device=tgt_in.device)
        if not self.use_comer:
            return self.decoder(emb, mem, tgt_mask=mask,
                                memory_key_padding_mask=mem_mask)
        if hw_mem is None:
            if (self._last_hw_mem is not None
                    and self._last_hw_mem[0] * self._last_hw_mem[1] == mem.shape[1]):
                hw_mem = self._last_hw_mem
            else:
                hw_mem = (1, mem.shape[1])
        h_mem, w_mem = hw_mem
        x = emb
        use_ckpt = self.training and torch.is_grad_enabled()
        for layer in self.decoder.layers:
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    _decoder_layer_comer,
                    layer, x, mem, mask, mem_mask, self.comer_bias, h_mem, w_mem,
                    use_reentrant=False,
                )
            else:
                x = _decoder_layer_comer(
                    layer, x, mem, mask, mem_mask, self.comer_bias, h_mem, w_mem)
        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return x

    def decode_logits(self, mem, tgt_in, mem_mask=None, hw_mem=None):
        """Decoder-only logits ``[B, T, V]`` for precomputed memory.

        Same math as :meth:`forward` minus the encode, so spike-S3
        sibling scoring (one encode + many teacher-forcing passes)
        stays cheap. ``tgt_in`` includes the BOS prefix.
        """
        return self.out(self.decode_hidden(mem, tgt_in, mem_mask, hw_mem=hw_mem))

    def loss(self, images, tgt, sample_p=0.0, mem_mask=None,
               tok_weight=1.0, weight_ids=(), tok_dropout=0.0,
               content_ids=(), augment=None):
        """NLL with optional two-pass scheduled sampling.

        sample_p=0 (default) is pure teacher-forcing: input
        [BOS, y0..yn-2] predicts y0..yn-1 (legacy path, bitwise same).
        sample_p>0 runs a no-grad teacher pass, then re-predicts from a
        mixed prefix where each non-PAD teacher token is replaced by the
        pass-1 argmax with probability sample_p. This trains recovery
        from the model's own errors (exposure-bias remedy) at 2x forward
        cost instead of a sequential rollout loop. PAD slots and BOS are
        never replaced.
        tok_weight scales the per-token NLL of ``weight_ids`` (structural
        tokens: barlines + key/time signatures). 1.0 (default) builds no
        weight tensor, keeping the legacy path bitwise identical.
        tok_dropout replaces each input token belonging to
        ``content_ids`` with a uniform random real token (never
        PAD/BOS/EOS) with probability ``tok_dropout``. 0.0 (default)
        skips entirely, keeping the legacy path bitwise identical.
        """
        bos = torch.full_like(tgt[:, :1], self.bos_id)
        inp = torch.cat([bos, tgt[:, :-1]], dim=1)
        if tok_dropout > 0.0 and content_ids:
            cid = torch.tensor(list(content_ids), device=inp.device)
            is_content = (inp.unsqueeze(-1) == cid).any(-1)
            drop = (torch.rand(inp.shape, device=inp.device)
                    < tok_dropout) & is_content
            inp = torch.where(
                drop,
                torch.randint(0, self.vocab - 3, inp.shape,
                              device=inp.device),
                inp)
        ce_kw = {}
        if tok_weight != 1.0 and len(weight_ids) > 0:
            w = torch.ones(self.vocab, device=tgt.device)
            w[list(weight_ids)] = tok_weight
            ce_kw['weight'] = w
        if augment is not None:
            raise ValueError('spike-S8 latent SpecAugment was removed; '
                             'pass augment=None')
        if sample_p <= 0.0:
            logits = self(images, inp, mem_mask)
            return nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab), tgt.reshape(-1),
                ignore_index=self.pad_id, **ce_kw)
        with torch.no_grad():
            probe = self(images, inp, mem_mask)
            pred = probe.argmax(-1)
            keep = (torch.rand_like(inp.float()) >= sample_p)
            keep[:, 0] = True  # BOS always teacher
            prev = torch.cat([torch.full_like(tgt[:, :1], -1),
                              tgt[:, :-1]], dim=1)
            keep = keep | (prev == self.pad_id)
            keep = keep | (pred == self.pad_id) | (pred == self.bos_id)
            mixed = torch.where(keep, inp, pred)
        logits = self(images, mixed, mem_mask)
        return nn.functional.cross_entropy(
            logits.reshape(-1, self.vocab), tgt.reshape(-1),
            ignore_index=self.pad_id, **ce_kw)

    def loss_shared(self, images, tgt, sample_p=0.0, mem_mask=None,
                    tok_weight=1.0, weight_ids=(), tok_dropout=0.0,
                    content_ids=(), need_aux=False, need_coord=False,
                    coord_layer=-1, augment=None,
                    return_hidden=False, return_logits=False,
                    need_gdcm=False):
        """Single-pass ``(ce_loss, aux_pred, x_centroids[, h][, logits][, gdcm_loss])`` sharing 1 encoder + 1 decoder.

        Eliminates the 3x ConvNeXt-Tiny encoder + 2x Transformer decoder
        redundant autograd graphs when ``aux_weight > 0`` and/or
        ``coord_weight > 0`` (~2.5x faster, ~2.7x lower peak VRAM, identical
        forward/gradient math when ``tok_dropout == 0``). When
        ``return_hidden=True``, ``return_logits=True``, and/or
        ``need_gdcm=True``, appends the shared decoder hidden states ``h``,
        output ``logits``, and/or ``gdcm_loss`` to the returned tuple in that
        order so ``S18FactoredHead``, ``logit_jsd_consistency``, and GDCM
        reuse the same encoder + decoder pass.
        """
        if augment is not None:
            raise ValueError('spike-S8 latent SpecAugment was removed; '
                             'pass augment=None')
        mem, feats, h_mem, w_mem, pred_counts = self._encode_with_feats(
            images, mem_mask, return_counts=True)
        apred = self.aux_head(feats) if need_aux else None

        bos = torch.full_like(tgt[:, :1], self.bos_id)
        inp = torch.cat([bos, tgt[:, :-1]], dim=1)
        if tok_dropout > 0.0 and content_ids:
            cid = torch.tensor(list(content_ids), device=inp.device)
            is_content = (inp.unsqueeze(-1) == cid).any(-1)
            drop = (torch.rand(inp.shape, device=inp.device)
                    < tok_dropout) & is_content
            inp = torch.where(
                drop,
                torch.randint(0, self.vocab - 3, inp.shape,
                              device=inp.device),
                inp)
        ce_kw = {}
        if tok_weight != 1.0 and len(weight_ids) > 0:
            w = torch.ones(self.vocab, device=tgt.device)
            w[list(weight_ids)] = tok_weight
            ce_kw['weight'] = w

        dec_in = inp
        if sample_p > 0.0:
            with torch.no_grad():
                probe = self.decode_logits(
                    mem, inp, mem_mask, hw_mem=(h_mem, w_mem))
                pred = probe.argmax(-1)
                keep = (torch.rand_like(inp.float()) >= sample_p)
                keep[:, 0] = True
                prev = torch.cat([torch.full_like(tgt[:, :1], -1),
                                  tgt[:, :-1]], dim=1)
                keep = keep | (prev == self.pad_id)
                keep = keep | (pred == self.pad_id) | (pred == self.bos_id)
                dec_in = torch.where(keep, inp, pred)

        held = {}
        handle = None
        top = self.decoder.layers[coord_layer] if need_coord else None
        if need_coord:
            def _hook(mod, _inp, out):
                held['q'] = out.detach().float()
            handle = top.norm2.register_forward_hook(_hook)
        try:
            h = self.decode_hidden(mem, dec_in, mem_mask, hw_mem=(h_mem, w_mem))
            logits = self.out(h)
        finally:
            if handle is not None:
                handle.remove()

        ce = nn.functional.cross_entropy(
            logits.reshape(-1, self.vocab), tgt.reshape(-1),
            ignore_index=self.pad_id, **ce_kw)

        xc = None
        if need_coord:
            w_att = _cross_weights(top.multihead_attn, held['q'], mem.float(),
                                   key_padding_mask=mem_mask)
            b, t, _ = w_att.shape
            xs = (torch.arange(w_mem, device=w_att.device).float() + 0.5) / w_mem
            xc = (w_att.reshape(b, t, h_mem, w_mem) * xs).sum(dim=(2, 3))

        gdcm_loss = None
        if need_gdcm and self.use_gdcm and pred_counts is not None:
            tgt_counts = comer_gdcm.extract_strip_counts(
                tgt, pad_id=self.pad_id).to(mem.device)
            gdcm_loss = self.gdcm_head.compute_loss(pred_counts, tgt_counts)

        out = (ce, apred, xc)
        if return_hidden:
            out = out + (h,)
        if return_logits:
            out = out + (logits,)
        if need_gdcm:
            out = out + (gdcm_loss,)
        return out

    @torch.no_grad()
    def greedy(self, images, max_len=2048, constrain=False, mem_mask=None):
        """Greedy decode one batch; stops every row at EOS.

        With constrain=True, each step masks logits to the bekern
        grammar continuations of the row's prefix (plus EOS at event
        boundaries), so output always parses in `decode_bekern`.
        ``mem_mask`` is an optional bool [B, S] memory key mask
        (True = ignore); forwarded to :meth:`encode` and the decoder's
        ``memory_key_padding_mask``. None keeps the exact legacy path.
        """
        self.eval()
        mem, _, h_mem, w_mem = self._encode_with_feats(images, mem_mask)
        b = images.shape[0]
        device = images.device
        out = torch.full((b, 1), self.bos_id, dtype=torch.long,
                         device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_len):
            nxt = self.out(self.decode_hidden(
                mem, out, mem_mask, hw_mem=(h_mem, w_mem)))[:, -1]
            if constrain:
                for r in range(b):
                    if finished[r]:
                        continue
                    allowed, eos_ok = grammar.allowed_next(
                        out[r, 1:].tolist())
                    if eos_ok:
                        allowed.add(self.eos_id)
                    forb = torch.ones(self.vocab, dtype=torch.bool,
                                      device=device)
                    forb[list(allowed)] = False
                    forb[self.pad_id] = True
                    forb[self.bos_id] = True
                    nxt[r][forb] = float('-inf')
            tok = nxt.argmax(-1)
            tok = torch.where(finished, torch.full_like(tok, self.pad_id),
                              tok)
            out = torch.cat([out, tok[:, None]], dim=1)
            finished = finished | (tok == self.eos_id)
            if finished.all():
                break
        return out[:, 1:]

    def _cached_setup(self, mem):
        """Pre-split layer projections + cross K/V (one mem batch)."""
        params = [_layer_params(layer) for layer in self.decoder.layers]
        mems = [_mem_kv(mem, xa) for (_, xa) in params]
        return params, mems

    def _cached_step(self, params, mems, tok, pos, caches, mem_mask=None,
                     cum_covs=None, hw_mem=None):
        """Advance every layer by one position; returns final hidden."""
        device = tok.device
        x = self.tok_emb(tok) + _sinusoidal_1d_pos(
            pos, self.d_model, device)[None, None, :]
        outs = []
        use_c = self.use_comer and cum_covs is not None and hw_mem is not None
        if use_c:
            h_mem, w_mem = hw_mem
        for li, layer in enumerate(self.decoder.layers):
            kc, vc = caches[li]
            if use_c:
                x, kc, vc, new_cov = _layer_step_comer(
                    layer, params[li], x, kc, vc, mems[li], mem_mask,
                    self.comer_bias, cum_covs[li], h_mem, w_mem)
                cum_covs[li] = new_cov
            else:
                x, kc, vc = _layer_step(layer, params[li], x, kc, vc,
                                        mems[li], mem_mask)
            outs.append((kc, vc))
        return x, outs

    def _apply_knobs(self, logits, rows, min_len, eos_bias, rep_penalty,
                     run_cap, cycle_repeats, no_repeat_ngram):
        """Apply E4 free-gain knobs to one step's logits (in place).

        ``rows`` parallels the batch: dicts with ``state``
        (PrefixState), ``hist`` (emitted id list, no BOS), ``ngrams``
        (seen n-gram set or None), ``gen_len`` (emitted count).
        Neutral defaults (min_len=0, eos_bias=0, rep_penalty=1,
        run_cap=0, cycle_repeats=0, no_repeat_ngram=0) leave logits
        unchanged apart from the grammar mask.
        """
        device = logits.device
        for r, row in enumerate(rows):
            if row.get('closed', False):
                continue
            row['fires'] = row.get('fires', 0)
            if eos_bias:
                logits[r, self.eos_id] += eos_bias
            if rep_penalty != 1.0 and row['hist']:
                seen = torch.tensor(sorted(set(row['hist'])),
                                    dtype=torch.long, device=device)
                lv = logits[r, seen]
                logits[r, seen] = torch.where(lv > 0, lv / rep_penalty,
                                              lv * rep_penalty)
            allow, eos_ok = row['state'].allowed()
            if row['gen_len'] < min_len:
                eos_ok = False
            if eos_ok:
                allow = set(allow) | {self.eos_id}
            if run_cap and row['state'].run_len >= run_cap \
                    and row['state'].run_tok is not None:
                try:
                    xt = grammar._T[row['state'].run_tok]
                    if xt in allow:
                        row['fires'] += 1
                    allow.discard(xt)
                except KeyError:
                    pass
            if cycle_repeats:
                forb = row['state'].cycle_forbidden(
                    max_repeats=cycle_repeats)
                if forb is not None:
                    try:
                        if grammar._T[forb] in allow:
                            row['fires'] += 1
                        allow.discard(grammar._T[forb])
                    except KeyError:
                        pass
            if no_repeat_ngram and not row['state'].header_open \
                    and len(row['hist']) >= no_repeat_ngram - 1:
                ctx = tuple(row['hist'][-(no_repeat_ngram - 1):]) \
                    if no_repeat_ngram > 1 else ()
                for c in list(allow):
                    if ctx + (c,) in row['ngrams']:
                        allow.discard(c)
            forb = torch.ones(self.vocab, dtype=torch.bool, device=device)
            forb[list(allow)] = False
            forb[self.pad_id] = True
            forb[self.bos_id] = True
            logits[r][forb] = float('-inf')

    @torch.no_grad()
    def greedy_kv(self, images, max_len=4608, constrain=True, strict=False,
                  min_len=0, eos_bias=0.0, rep_penalty=1.0, run_cap=0,
                  cycle_repeats=0, no_repeat_ngram=0, fires=None,
                  mem_mask=None):
        """KV-cached greedy decode (O(T) transformer work per page).

        Neutral knob defaults reproduce :meth:`greedy` token-for-token
        (verified by the E4 parity probe); the E4 free gains switch on
        via the keyword knobs. ``strict`` selects the validator-predicate
        grammar mask (E4 S3); loop caps live decoder-side (``run_cap`` =
        max identical-in-a-row, ``cycle_repeats`` = max adjacent cycle
        repeats, ``no_repeat_ngram`` = classic global n-gram block, 0
        disables each).
        ``mem_mask`` is an optional bool [B, S] memory key mask
        (True = ignore); forwarded to :meth:`encode` and the cached
        cross-attention steps. None keeps the exact legacy path.
        """
        self.eval()
        mem, _, h_mem, w_mem = self._encode_with_feats(images, mem_mask)
        b = images.shape[0]
        device = images.device
        params, mems = self._cached_setup(mem)
        caches = [(None, None) for _ in self.decoder.layers]
        cum_covs = (
            [self.comer_bias.init_coverage(b, h_mem, w_mem,
                                           device=device, dtype=mem.dtype)
             for _ in self.decoder.layers]
            if self.use_comer else None
        )
        out = torch.full((b, 1), self.bos_id, dtype=torch.long,
                         device=device)
        rows = [{'state': grammar.PrefixState(strict=strict), 'hist': [],
                 'ngrams': set(), 'gen_len': 0, 'closed': False}
                for _ in range(b)]
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        for pos in range(max_len):
            x, caches = self._cached_step(
                params, mems, out[:, -1:], pos, caches, mem_mask,
                cum_covs=cum_covs, hw_mem=(h_mem, w_mem))
            nxt = self.out(x)[:, -1, :]
            if constrain:
                self._apply_knobs(nxt, rows, min_len, eos_bias,
                                  rep_penalty, run_cap, cycle_repeats,
                                  no_repeat_ngram)
            tok = nxt.argmax(-1)
            tok = torch.where(finished, torch.full_like(tok, self.pad_id),
                              tok)
            out = torch.cat([out, tok[:, None]], dim=1)
            for r in range(b):
                if finished[r]:
                    continue
                t = int(tok[r])
                rows[r]['hist'].append(t)
                rows[r]['gen_len'] += 1
                n = no_repeat_ngram
                if n and len(rows[r]['hist']) >= n:
                    rows[r]['ngrams'].add(tuple(rows[r]['hist'][-n:]))
                if t == self.eos_id:
                    rows[r]['closed'] = True
                else:
                    try:
                        rows[r]['state'].step(
                            data.VOCAB[t] if t < len(data.VOCAB)
                            else None)
                    except (ValueError, TypeError):
                        rows[r]['state'].broken = True
            finished = finished | (tok == self.eos_id)
            if finished.all():
                break
        if fires is not None:
            fires.extend(int(row.get('fires', 0)) for row in rows)
        return out[:, 1:]

    @torch.no_grad()
    def beam_search(self, images, max_len=4608, constrain=True,
                    strict=True, beam=3, len_alpha=0.6, min_len=0,
                    eos_bias=0.0, rep_penalty=1.0, run_cap=2,
                    cycle_repeats=12, no_repeat_ngram=0, fires=None,
                    mem_mask=None, return_all=False, return_beams=False,
                    adapt_steps=0, adapt_lr=0.1, adapt_rank=8):
        """Grammar-masked beam search with KV-cache + length norm.

        E4 S4 only (never before the KV-cache lands: uncached beam
        triples the O(T^2) cost). ``beam`` > 5 without a length norm
        stays banned — the norm exponent ``len_alpha`` (Wu et al.:
        score / ((5 + len) / 6)^alpha) is always applied at selection.
        Returns best-beam ids per page, PAD-padded (greedy-shaped).
        ``mem_mask`` is an optional bool [B, S] memory key mask
        (True = ignore); forwarded to :meth:`encode` and the cached
        cross-attention steps (fanned out across beams). None keeps
        the exact legacy path.
        """
        self.eval()
        mem, _, h_mem, w_mem = self._encode_with_feats(images, mem_mask)
        if adapt_steps is not None and adapt_steps > 0:
            raise ValueError('dropped spike S9 (tt_adapter) was removed; '
                             'pass adapt_steps=0')
        nb = images.shape[0]
        device = images.device
        K = max(1, beam)
        mem_k = mem.repeat_interleave(K, dim=0)
        mask_k = (mem_mask.repeat_interleave(K, dim=0)
                  if mem_mask is not None else None)
        params, mems = self._cached_setup(mem_k)
        caches = [(None, None) for _ in self.decoder.layers]
        b = nb * K
        cum_covs = (
            [self.comer_bias.init_coverage(b, h_mem, w_mem,
                                           device=device, dtype=mem.dtype)
             for _ in self.decoder.layers]
            if self.use_comer else None
        )
        out = torch.full((b, 1), self.bos_id, dtype=torch.long,
                         device=device)
        rows = [{'state': grammar.PrefixState(strict=strict), 'hist': [],
                 'ngrams': set(), 'gen_len': 0, 'closed': False}
                for _ in range(b)]
        cum = torch.zeros(b, device=device)
        # First step is shared (all beams identical): run once via row 0
        # of each page, then fan out to K beams.
        fanned = False
        for pos in range(max_len):
            x, caches = self._cached_step(
                params, mems, out[:, -1:], pos, caches, mask_k,
                cum_covs=cum_covs, hw_mem=(h_mem, w_mem))
            nxt = self.out(x)[:, -1, :]
            if constrain:
                self._apply_knobs(nxt, rows, min_len, eos_bias,
                                  rep_penalty, run_cap, cycle_repeats,
                                  no_repeat_ngram)
            logp = torch.log_softmax(nxt, dim=-1)
            if not fanned:
                # Top-K first tokens per page, then fan caches out.
                first = logp[::K, :]
                topv, topi = first.topk(K, dim=-1)
                new_idx = []
                for pg in range(nb):
                    for k in range(K):
                        new_idx.append(pg * K)
                order = torch.tensor(new_idx, dtype=torch.long,
                                     device=device)
                caches = [(kc.index_select(0, order) if kc is not None
                           else None,
                           vc.index_select(0, order) if vc is not None
                           else None) for kc, vc in caches]
                if cum_covs is not None:
                    cum_covs = [cc.index_select(0, order) for cc in cum_covs]
                mems = [(km.index_select(0, order),
                         vm.index_select(0, order)) for km, vm in mems]
                if mask_k is not None:
                    mask_k = mask_k.index_select(0, order)
                out = out.index_select(0, order)
                base_rows = rows[::K]
                rows = []
                for pg in range(nb):
                    for k in range(K):
                        src = base_rows[pg]
                        rows.append({'state': src['state'].copy(),
                                     'hist': list(src['hist']),
                                     'ngrams': set(src['ngrams']),
                                     'gen_len': src['gen_len'],
                                     'closed': False})
                        rows[-1]['hist'].append(int(topi[pg, k]))
                        rows[-1]['gen_len'] += 1
                        t = int(topi[pg, k])
                        if t == self.eos_id:
                            rows[-1]['closed'] = True
                        else:
                            try:
                                rows[-1]['state'].step(
                                    data.VOCAB[t]
                                    if t < len(data.VOCAB) else None)
                            except (ValueError, TypeError):
                                rows[-1]['state'].broken = True
                        n = no_repeat_ngram
                        if n and len(rows[-1]['hist']) >= n:
                            rows[-1]['ngrams'].add(
                                tuple(rows[-1]['hist'][-n:]))
                cum = topv.reshape(-1)
                out = torch.cat(
                    [out, topi.reshape(-1)[:, None].to(torch.long)], 1)
                fanned = True
                continue
            scores = cum[:, None] + logp
            # Closed beams self-loop on PAD with unchanged score (never
            # re-expand EOS rows); this keeps every page's rows valid
            # until global termination, with no per-page bookkeeping.
            for r in range(b):
                if rows[r]['closed']:
                    scores[r, :] = float('-inf')
                    scores[r, self.pad_id] = float(cum[r])
            # Pick top-K expansions per page.
            sel_tok = torch.zeros(b, dtype=torch.long, device=device)
            sel_src = torch.zeros(b, dtype=torch.long, device=device)
            for pg in range(nb):
                sl = slice(pg * K, (pg + 1) * K)
                flat = scores[sl, :].reshape(-1)
                topv, topi = flat.topk(K)
                for k in range(K):
                    gi = pg * K + k
                    src = int(topi[k]) // self.vocab + pg * K
                    tk = int(topi[k]) % self.vocab
                    sel_src[gi] = src
                    sel_tok[gi] = tk
                    cum[gi] = float(topv[k])
            order = sel_src
            caches = [(kc.index_select(0, order),
                       vc.index_select(0, order)) for kc, vc in caches]
            if cum_covs is not None:
                cum_covs = [cc.index_select(0, order) for cc in cum_covs]
            out = torch.cat([out.index_select(0, order),
                             sel_tok[:, None]], 1)
            new_rows = []
            for gi in range(b):
                src = rows[int(order[gi])]
                cp = {'state': src['state'].copy(),
                      'hist': list(src['hist']),
                      'ngrams': set(src['ngrams']),
                      'gen_len': src['gen_len'],
                      'closed': src['closed']}
                t = int(sel_tok[gi])
                if cp['closed']:
                    # PAD self-loop: output already extended above;
                    # grammar state, history length, and score frozen.
                    new_rows.append(cp)
                    continue
                cp['hist'].append(t)
                cp['gen_len'] += 1
                if t == self.eos_id:
                    cp['closed'] = True
                else:
                    try:
                        cp['state'].step(
                            data.VOCAB[t]
                            if t < len(data.VOCAB) else None)
                    except (ValueError, TypeError):
                        cp['state'].broken = True
                n = no_repeat_ngram
                if n and len(cp['hist']) >= n:
                    cp['ngrams'].add(tuple(cp['hist'][-n:]))
                new_rows.append(cp)
            rows = new_rows
            if all(row['closed'] for row in rows):
                break
        # Select per page by length-normalized score first; the closed
        # flag only counts for beams ending at a bar boundary (T2.4a).
        # Spike S4: ``return_all`` also returns the non-winning beams
        # ([nb, K, L] rank-ordered) for on-the-fly MWER; ``return_beams``
        # returns ``(all_beams, norm_logprobs)`` for T4.1 visual reranking.
        # The default path (best only) is untouched.
        ranked = []
        ranked_scores = []
        for pg in range(nb):
            cands = []
            for k in range(K):
                gi = pg * K + k
                ln = max(1, rows[gi]['gen_len'])
                norm = ((5.0 + ln) / 6.0) ** len_alpha
                norm_score = float(cum[gi]) / norm
                cands.append((_beam_rank_key(norm_score,
                                             rows[gi]['closed'],
                                             rows[gi]['hist'],
                                             self.eos_id), norm_score, gi))
            cands.sort(key=lambda c: c[0], reverse=True)
            ranked.append([out[gi, 1:].tolist() for _key, _ns, gi in cands])
            ranked_scores.append([ns for _key, ns, _gi in cands])
        best = [r[0] for r in ranked]
        L = max(len(x) for x in best)
        res = torch.full((nb, L), self.pad_id, dtype=torch.long,
                         device=device)
        for i, x in enumerate(best):
            res[i, :len(x)] = torch.tensor(x, dtype=torch.long,
                                           device=device)
        allr = None
        if return_all or return_beams:
            La = max(len(x) for r in ranked for x in r)
            allr = torch.full((nb, K, La), self.pad_id,
                              dtype=torch.long, device=device)
            for pg, r in enumerate(ranked):
                for k, x in enumerate(r):
                    allr[pg, k, :len(x)] = torch.tensor(
                        x, dtype=torch.long, device=device)
        if fires is not None:
            for pg, x in enumerate(best):
                # re-derive the winner: score first, closed flag gated
                # on a bar boundary (mirrors above, T2.4a)
                cands = []
                for k in range(K):
                    gi = pg * K + k
                    ln = max(1, rows[gi]['gen_len'])
                    norm = ((5.0 + ln) / 6.0) ** len_alpha
                    cands.append((_beam_rank_key(float(cum[gi]) / norm,
                                                 rows[gi]['closed'],
                                                 rows[gi]['hist'],
                                                 self.eos_id), gi))
                cands.sort(key=lambda c: c[0], reverse=True)
                fires.append(int(rows[cands[0][1]].get('fires', 0)))
        if return_beams:
            return allr, ranked_scores
        return allr if return_all else res
