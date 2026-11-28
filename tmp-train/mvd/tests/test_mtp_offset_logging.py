"""CPU-only: the logged ce/mtp offset subtracted for -1e4 v1 schema masking must
reproduce the true (unmasked) loss scale. Mirrors train_uni_v2.py ~L2295-2336."""
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import train_uni_v2 as T  # noqa: E402
from uni_omr_v2 import UniOMRv2Net as M  # noqa: E402

V, PAD, LS, NH = 265, 0, 0.05, 3


def _run(dtype, B, Tn, seed=0):
    g = torch.Generator().manual_seed(seed)
    dev = torch.device("cpu")
    ids = T._get_v2a_token_ids(V, dev)
    tgt = torch.randint(5, V, (B, Tn), generator=g)
    for i in ids.tolist():
        tgt[tgt == i] = 7
    tgt[:, -5:] = PAD
    is_v1 = torch.tensor([True, True, False, True, True, False][:B])
    base = (torch.randn(B, Tn, V, generator=g) * 3).to(dtype)
    mtp = [(torch.randn(B, Tn, V, generator=g) * 3).to(dtype) for _ in range(NH)]
    col, mv = T._get_v2a_mask_tensors(V, dev, dtype)
    mk = is_v1.view(-1, 1, 1) & col
    fake = types.SimpleNamespace(cfg=types.SimpleNamespace(resolved_pad_id=lambda: PAD, text_aux_loss_weight=0.0, bekern_full=False, vocab_size=V), bd3_repair_head=None, lm_head_text=None)
    masked = {"logits": torch.where(mk, mv, base), "mtp_logits": [torch.where(mk, mv, m) for m in mtp]}
    l = M.compute_loss(fake, masked, tgt, label_smoothing=LS, mtp_loss_weight=0.3)
    # Reference: same loss with masked classes physically excluded on v1 rows
    # (active-class-only smoothing is what "true scale" means: no -1e4 term).
    cpt = LS * (len(ids) / V) * 1e4
    valid = tgt != PAD
    rv = is_v1.unsqueeze(1).expand_as(tgt)
    ce_off = ((valid & rv).sum() / valid.sum()) * cpt
    v1c = tc = 0.0
    for k in range(NH):
        vm = tgt[:, k + 1:] != PAD
        v1c += (vm & rv[:, k + 1:]).sum().item()
        tc += vm.sum().item()
    mtp_off = v1c / tc * cpt
    return float(l["ce_loss"]) - float(ce_off), float(l["mtp_loss"]) - float(mtp_off)


def test_offset_leaves_nonnegative_true_scale_fp32():
    for B, Tn in ((2, 200), (4, 40)):
        ce, mt = _run(torch.float32, B, Tn)
        assert ce > 0 and mt > 0, (B, Tn, ce, mt)


def test_offset_leaves_nonnegative_true_scale_bf16():
    for B, Tn in ((2, 200), (4, 40)):
        ce, mt = _run(torch.bfloat16, B, Tn)
        assert ce > 0 and mt > 0, (B, Tn, ce, mt)


def _corrected_mtp(shift, B=4, Tn=40):
    """Corrected (offset-subtracted) logged MTP loss when every active logit is shifted by `shift`."""
    g = torch.Generator().manual_seed(3)
    dev = torch.device("cpu")
    ids = T._get_v2a_token_ids(V, dev)
    tgt = torch.randint(5, V, (B, Tn), generator=g)
    for i in ids.tolist():
        tgt[tgt == i] = 7
    tgt[:, -5:] = PAD
    col, mv = T._get_v2a_mask_tensors(V, dev, torch.float32)
    mk = torch.ones(B, 1, 1, dtype=torch.bool) & col  # all-v1 (strip batch)
    base = [(torch.randn(B, Tn, V, generator=g) * 2) for _ in range(NH + 1)]
    lg = [(b + shift).requires_grad_(True) for b in base]
    out = {"logits": torch.where(mk, mv, lg[0]), "mtp_logits": [torch.where(mk, mv, m) for m in lg[1:]], "v1_mask_btv": mk}
    fake = types.SimpleNamespace(
        cfg=types.SimpleNamespace(resolved_pad_id=lambda: PAD, text_aux_loss_weight=0.0, bekern_full=False, vocab_size=V),
        bd3_repair_head=None, lm_head_text=None)
    l = M.compute_loss(fake, out, tgt, label_smoothing=LS, mtp_loss_weight=0.3)
    return l["mtp_loss"], lg


def test_mtp_loss_is_invariant_to_constant_logit_shift():
    """Softmax over the ACTIVE classes is shift invariant, so the true loss must be too.
    With -1e4 masked logits + F.cross_entropy(label_smoothing) the smoothing term over the
    28 masked classes is ls*28/V*(lse+1e4): linear in lse -> the optimizer can lower the loss
    without bound by pushing all active logits negative (logged mtp -> -5..-11)."""
    l0, _ = _corrected_mtp(0.0)
    l1, lg = _corrected_mtp(-2000.0)
    assert abs(float(l0) - float(l1)) < 1e-2, (float(l0), float(l1))


def test_mtp_gradient_has_no_constant_shift_component():
    l0, lg = _corrected_mtp(0.0)
    l0.backward()
    # d loss / d (uniform shift of active logits of head 0) must be ~0 for a shift-invariant loss
    g = lg[1].grad
    shift_grad = g[..., torch.arange(V) >= 0].sum().item() - g[..., T._get_v2a_token_ids(V, torch.device("cpu"))].sum().item()
    assert abs(shift_grad) < 1e-4, shift_grad
