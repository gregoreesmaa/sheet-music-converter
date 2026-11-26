#!/usr/bin/env python3
"""2028 Trial: On-Device Test-Time Training (TTT) for Unseen Hands & Publishers.

Adapts a tiny subset of UniOMR parameters (< 0.1% of weights — ``RMSNorm``
scale weights in ``backbone`` / ``enc_layers`` / ``enc_norm``) on a single
unseen manuscript or publisher page at inference time, then restores exact
original weights (``atol=0``) between documents.

Two self-supervised adaptation objectives:
  1. **Multi-View Encoder & Router Consistency:** aligns encoder features and
     router ``saliency_prob`` between the clean page view and a mildly
     contrast/scale-jittered view.
  2. **Confidence-Gated Self-Distillation (Tent / SCRIBE-style):** runs a fast
     ``PrefixState``-constrained greedy decode, filters to high-confidence
     non-loop tokens (``prob >= 0.85``), and minimizes conditional entropy on
     those predictions while anchored to the pre-TTT snapshot via KL divergence
     and L2 weight anchoring + gradient clipping so adaptation cannot diverge.

Usage (A/B trial on the 14 unseen historical handwritten pieces in suite31_hw):
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/trial_2028_ttt_adapt.py \\
        --ckpt D:/tmp/mvd-runs/uni-omr-iter6-clean/uni_omr_best.pt \\
        --ttt-steps 3 --ttt-lr 5e-5 --device cuda --bf16 \\
        --out D:/tmp/uni-omr-suite31-ood/ttt_ab_report.json
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import eval_sys_gate as esg  # noqa: E402
import eval_uni_suite31 as ev  # noqa: E402
import grammar  # noqa: E402
import uni_omr  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: Multi-View Augmentation, Loop Detection, Dense Feature Scattering
# ---------------------------------------------------------------------------

def make_jittered_view(
    image: torch.Tensor,
    step_idx: int = 0,
    contrast_range: Tuple[float, float] = (0.88, 1.12),
    brightness_range: Tuple[float, float] = (-0.04, 0.04),
    scale_range: Tuple[float, float] = (0.94, 1.06),
    seed: int = 2028,
) -> torch.Tensor:
    """Create a mildly contrast/brightness/scale-jittered view of ``image``.

    Args:
        image: ``(B, C, H, W)`` float tensor in ``[0, 1]``.
        step_idx: deterministic step index for reproducible view sampling.
        contrast_range: multiplicative contrast factor bounds.
        brightness_range: additive brightness shift bounds.
        scale_range: spatial scale jitter bounds.
        seed: base PRNG seed.

    Returns:
        Jittered ``(B, C, H_s, W_s)`` tensor in ``[0, 1]`` on the same device/dtype.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected 4D image tensor (B, C, H, W), got shape {tuple(image.shape)}")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 1009 * int(step_idx))

    u_c = float(torch.rand((), generator=gen).item())
    u_b = float(torch.rand((), generator=gen).item())
    u_s = float(torch.rand((), generator=gen).item())

    contrast = contrast_range[0] + u_c * (contrast_range[1] - contrast_range[0])
    brightness = brightness_range[0] + u_b * (brightness_range[1] - brightness_range[0])
    scale = scale_range[0] + u_s * (scale_range[1] - scale_range[0])
    # Ensure scale is visibly non-trivial (not 1.0) so scale jitter always exercises invariance.
    if abs(scale - 1.0) < 0.015:
        scale = scale_range[0] if (step_idx % 2 == 0) else scale_range[1]

    x = image.float()
    mean_val = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean_val) * contrast + mean_val + brightness
    x = x.clamp(0.0, 1.0)

    _, _, h, w = x.shape
    new_h = max(32, int(round(h * scale / 16.0)) * 16)
    new_w = max(32, int(round(w * scale / 16.0)) * 16)
    if new_h != h or new_w != w:
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    return x.to(dtype=image.dtype, device=image.device).clamp(0.0, 1.0)


def detect_loop_mask(
    token_ids: Sequence[int],
    max_single_repeat: int = 4,
    max_ngram_repeats: int = 3,
    ngram_sizes: Tuple[int, ...] = (2, 3, 4, 6, 8),
) -> List[bool]:
    """Flag tokens participating in degenerate repetition loops.

    Returns a list of booleans of length ``len(token_ids)`` where ``True``
    indicates the token is part of a tight repetition cycle and should be
    excluded from self-distillation.
    """
    n = len(token_ids)
    is_loop = [False] * n
    if n == 0:
        return is_loop

    # 1. Single-token consecutive runs >= max_single_repeat
    run = 1
    for i in range(1, n):
        if token_ids[i] == token_ids[i - 1]:
            run += 1
            if run >= max_single_repeat:
                is_loop[i] = True
        else:
            run = 1

    # 2. Short n-gram consecutive cycles >= max_ngram_repeats
    for g in ngram_sizes:
        if n < g * max_ngram_repeats:
            continue
        i = 0
        while i <= n - g * max_ngram_repeats:
            pat = token_ids[i : i + g]
            reps = 1
            j = i + g
            while j + g <= n and list(token_ids[j : j + g]) == list(pat):
                reps += 1
                j += g
            if reps >= max_ngram_repeats:
                # Mark all repeated copies after the first occurrence as loop tokens.
                for k in range(i + g, j):
                    is_loop[k] = True
                i = j
            else:
                i += 1

    return is_loop


def _scatter_encoder_tokens_to_2d(
    mem: torch.Tensor,
    router_info: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scatter routed ink encoder tokens ``mem[:, :k_kept]`` onto a dense 2D grid.

    Returns:
        dense_2d: ``(B, D, H_p, W_p)`` float32 tensor.
        kept_mask_2d: ``(B, 1, H_p, W_p)`` float32 binary mask of routed positions.
    """
    b, _, d = mem.shape
    h_p, w_p = int(router_info["grid_hw"][0]), int(router_info["grid_hw"][1])
    k_kept = int(router_info["k_kept"])
    kept_idx: torch.Tensor = router_info["kept_idx"]  # (B, k_kept)
    n_grid = h_p * w_p

    mem_kept = mem[:, :k_kept, :].float()
    dense_flat = torch.zeros((b, n_grid, d), device=mem.device, dtype=torch.float32)
    idx_exp = kept_idx.unsqueeze(-1).expand(-1, -1, d)
    dense_flat = dense_flat.scatter(1, idx_exp, mem_kept)
    dense_2d = dense_flat.transpose(1, 2).reshape(b, d, h_p, w_p)

    mask_flat = torch.zeros((b, n_grid), device=mem.device, dtype=torch.float32)
    mask_flat = mask_flat.scatter(1, kept_idx, torch.ones_like(kept_idx, dtype=torch.float32))
    kept_mask_2d = mask_flat.reshape(b, 1, h_p, w_p)
    return dense_2d, kept_mask_2d


def compute_multiview_consistency_loss(
    mem_clean: torch.Tensor,
    rinfo_clean: Dict[str, Any],
    mem_aug: torch.Tensor,
    rinfo_aug: Dict[str, Any],
    router_weight: float = 0.5,
) -> Tuple[torch.Tensor, float, float]:
    """Compute Multi-View Encoder & Router Consistency loss.

    Compares:
      1. Encoder representations (both global ink-pooled cosine distance and
         spatially aligned 2D dense feature cosine distance over overlapping
         routed ink patches).
      2. Router ``saliency_prob = sigmoid(saliency_logits)`` spatially aligned
         via bilinear interpolation.

    Returns:
        ``(total_consistency_loss, feat_loss_val, router_loss_val)``
    """
    k_c = int(rinfo_clean["k_kept"])
    k_a = int(rinfo_aug["k_kept"])

    # 1a. Global ink-token mean representation consistency
    g_clean = mem_clean[:, :k_c, :].float().mean(dim=1)
    g_aug = mem_aug[:, :k_a, :].float().mean(dim=1)
    global_cos_loss = (1.0 - F.cosine_similarity(g_clean, g_aug, dim=-1, eps=1e-6)).mean()

    # 1b. Spatially aligned 2D encoder feature consistency
    dense_c, mask_c = _scatter_encoder_tokens_to_2d(mem_clean, rinfo_clean)
    dense_a, mask_a = _scatter_encoder_tokens_to_2d(mem_aug, rinfo_aug)
    if dense_a.shape[-2:] != dense_c.shape[-2:]:
        dense_a = F.interpolate(dense_a, size=dense_c.shape[-2:], mode="bilinear", align_corners=False)
        mask_a = F.interpolate(mask_a, size=dense_c.shape[-2:], mode="bilinear", align_corners=False)

    overlap = ((mask_c > 0.5) & (mask_a > 0.5)).squeeze(1)  # (B, H_p, W_p)
    if bool(overlap.any()):
        cos_map = F.cosine_similarity(dense_c, dense_a, dim=1, eps=1e-6)  # (B, H_p, W_p)
        spatial_cos_loss = (1.0 - cos_map[overlap]).mean()
        feat_loss = 0.5 * global_cos_loss + 0.5 * spatial_cos_loss
    else:
        feat_loss = global_cos_loss

    # 2. Router saliency_prob consistency
    sal_c = torch.sigmoid(rinfo_clean["saliency_logits"].float())
    sal_a = torch.sigmoid(rinfo_aug["saliency_logits"].float())
    if sal_a.shape[-2:] != sal_c.shape[-2:]:
        sal_a = F.interpolate(sal_a, size=sal_c.shape[-2:], mode="bilinear", align_corners=False)
    router_loss = F.mse_loss(sal_c, sal_a)

    total = feat_loss + float(router_weight) * router_loss
    return total, float(feat_loss.detach().item()), float(router_loss.detach().item())


# ---------------------------------------------------------------------------
# Promoted Presets (ov-38b) & 3-Decimal Confidence Skip Gate (i13-06c)
# ---------------------------------------------------------------------------

TTT_MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "backbone_cons": {
        "adapt_scope": "backbone_router",
        "ttt_lr": 3e-5,
        "ttt_steps": 2,
        "cons_weight": 1.0,
        "ent_weight": 0.0,
        "kl_weight": 0.0,
        "l2_weight": 1e-2,
        "grad_clip": 1.0,
    },
}

VALID_TTT_MODES: Tuple[str, ...] = ("custom", "backbone_cons")


def format_conf_skip_thr(thr: float) -> str:
    """Format confidence skip threshold to 3 decimals so -0.035/-0.040/-0.045 never collide."""
    return f"{float(thr):.3f}"


def format_conf_skip_tag(thr: float) -> str:
    """Format a collision-free 3-decimal tag (e.g. -0.040 -> '0040', -0.035 -> '0035')."""
    return f"{abs(float(thr)):.3f}".replace(".", "")


def should_skip_ttt_by_confidence(
    mean_logprob: float,
    conf_skip_thr: Optional[float],
) -> bool:
    """Return True when ``mean_logprob > conf_skip_thr`` (skip TTT with 0 weight updates)."""
    if conf_skip_thr is None:
        return False
    return float(mean_logprob) > float(conf_skip_thr)


def compute_pseudo_mean_logprob(
    pseudo_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]],
) -> float:
    """Compute mean top-token log-probability over valid positions (exact i13-06c formula)."""
    if pseudo_cache is None:
        return -10.0
    inp_ids, _conf_mask, valid_mask, ref_log_probs, _n_conf = pseudo_cache
    n_valid = int(valid_mask.sum().item())
    if n_valid <= 0 or inp_ids.shape[1] < 2:
        return -10.0
    top_lp = ref_log_probs.max(dim=-1).values
    mean_lp = float(((top_lp * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)).item())
    return round(mean_lp, 5)


# ---------------------------------------------------------------------------
# TTTSession — Snapshot, Per-Page Adaptation, Bit-Exact Restore
# ---------------------------------------------------------------------------

class TTTSession:
    """On-device Test-Time Training session for ``UniOMRNet``.

    Snapshots ONLY the adapted parameters (by default ``RMSNorm`` weights in
    ``backbone``, ``enc_layers``, and ``enc_norm`` — ``< 0.1%`` of model
    parameters; or ``backbone_router`` for promoted ``ov-38b`` ``backbone_cons``)
    so ``restore()`` / ``reset()`` is instant and bit-exact (``max_diff == 0.0``)
    between documents.

    Args:
        model: ``UniOMRNet`` instance.
        ttt_lr: AdamW learning rate for per-page adaptation (default ``5e-5``).
        adapt_scope: which parameters to adapt:
            - ``'norms'`` (default): ``RMSNorm`` weights in ``backbone``,
              ``enc_layers``, ``enc_norm``.
            - ``'all_rms_norms'``: all ``RMSNorm`` weights in the model
              (encoder + router + decoder).
            - ``'norms_with_gn'``: ``RMSNorm`` in ``enc_layers``/``enc_norm``
              plus ``GroupNorm`` affine params in ``backbone``/``router``.
            - ``'backbone_router'``: all parameters in ``backbone`` + ``router``
              with ``enc_layers``/``enc_norm``/decoder frozen (promoted ``ov-38b``).
            - ``'full_encoder'``: all parameters in ``backbone``, ``router``,
              ``enc_layers``, and ``enc_norm``.
        conf_threshold: minimum token probability for Loss 2 self-distillation.
        cons_weight: weight on Loss 1 (multi-view encoder & router consistency).
        ent_weight: weight on Loss 2 (confidence-gated conditional entropy).
        kl_weight: weight on KL divergence anchor to snapshot distribution.
        l2_weight: weight on L2 parameter anchor to snapshot weights.
        grad_clip: max L2 gradient norm clipping for stability.
        use_bf16: enable CUDA bfloat16 autocast during forward passes.
        strict_grammar: whether ``PrefixState`` uses strict bar-duration rules.
        max_decode_len: max tokens for the fast pre-TTT pseudo-label decode.
        seed: base seed for deterministic multi-view jitter.
        conf_skip_thr: optional 3-decimal ``mean_logprob`` skip gate (``i13-06c``;
            e.g. ``-0.040`` skips pages with ``mean_logprob > -0.040`` untouched).
    """

    VALID_SCOPES = ("norms", "all_rms_norms", "norms_with_gn", "backbone_router", "full_encoder")

    def __init__(
        self,
        model: uni_omr.UniOMRNet,
        ttt_lr: float = 5e-5,
        adapt_scope: str = "norms",
        conf_threshold: float = 0.85,
        cons_weight: float = 1.0,
        ent_weight: float = 1.0,
        kl_weight: float = 0.5,
        l2_weight: float = 1e-2,
        grad_clip: float = 1.0,
        use_bf16: bool = False,
        strict_grammar: bool = False,
        max_decode_len: int = 512,
        seed: int = 2028,
        conf_skip_thr: Optional[float] = None,
    ) -> None:
        if adapt_scope not in self.VALID_SCOPES:
            raise ValueError(
                f"Unsupported adapt_scope={adapt_scope!r}; expected one of {self.VALID_SCOPES}"
            )
        if ttt_lr <= 0.0:
            raise ValueError(f"ttt_lr must be positive, got {ttt_lr}")

        self.model = model
        self.ttt_lr = float(ttt_lr)
        self.adapt_scope = str(adapt_scope)
        self.conf_threshold = float(conf_threshold)
        self.cons_weight = float(cons_weight)
        self.ent_weight = float(ent_weight)
        self.kl_weight = float(kl_weight)
        self.l2_weight = float(l2_weight)
        self.grad_clip = float(grad_clip)
        self.use_bf16 = bool(use_bf16)
        self.strict_grammar = bool(strict_grammar)
        self.max_decode_len = int(max_decode_len)
        self.seed = int(seed)
        self.conf_skip_thr: Optional[float] = (
            float(conf_skip_thr) if conf_skip_thr is not None else None
        )

        self._adapted_named_params: List[Tuple[str, nn.Parameter]] = self._select_adapted_params()
        if not self._adapted_named_params:
            raise RuntimeError(f"No parameters matched adapt_scope={self.adapt_scope!r}")

        self._adapted_param_ids = {id(p) for _, p in self._adapted_named_params}
        self._snapshot: Dict[str, torch.Tensor] = {
            name: p.detach().clone() for name, p in self._adapted_named_params
        }
        self._total_params: int = sum(p.numel() for p in self.model.parameters())
        self._adapted_param_count: int = sum(p.numel() for _, p in self._adapted_named_params)

    def _select_adapted_params(self) -> List[Tuple[str, nn.Parameter]]:
        selected: List[Tuple[str, nn.Parameter]] = []
        seen: set[int] = set()

        def _collect_from_root(root_name: str, norm_types: Tuple[type, ...]) -> None:
            root_mod = getattr(self.model, root_name, None)
            if root_mod is None:
                return
            for sub_name, mod in root_mod.named_modules():
                if isinstance(mod, norm_types):
                    for pname, p in mod.named_parameters(recurse=False):
                        if id(p) not in seen:
                            seen.add(id(p))
                            full_name = (
                                f"{root_name}.{sub_name}.{pname}"
                                if sub_name
                                else f"{root_name}.{pname}"
                            )
                            selected.append((full_name, p))

        def _collect_all_from_root(root_name: str) -> None:
            root_mod = getattr(self.model, root_name, None)
            prefix = root_name
            if root_mod is None and root_name == "router":
                encoder = getattr(self.model, "encoder", None)
                if encoder is not None:
                    root_mod = getattr(encoder, "router", None)
                    prefix = "encoder.router"
            if root_mod is None:
                return
            for pname, p in root_mod.named_parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    selected.append((f"{prefix}.{pname}", p))

        if self.adapt_scope == "norms":
            for root in ("backbone", "enc_layers", "enc_norm"):
                _collect_from_root(root, (uni_omr.RMSNorm,))
        elif self.adapt_scope == "all_rms_norms":
            for root in ("backbone", "router", "enc_layers", "enc_norm", "dec_layers", "dec_norm"):
                _collect_from_root(root, (uni_omr.RMSNorm,))
        elif self.adapt_scope == "norms_with_gn":
            for root in ("backbone", "router", "enc_layers", "enc_norm"):
                _collect_from_root(root, (uni_omr.RMSNorm, nn.GroupNorm))
        elif self.adapt_scope == "backbone_router":
            for root in ("backbone", "router"):
                _collect_all_from_root(root)
        elif self.adapt_scope == "full_encoder":
            for root in ("backbone", "router", "enc_layers", "enc_norm"):
                _collect_all_from_root(root)

        return selected

    @property
    def adapted_param_names(self) -> List[str]:
        return [name for name, _ in self._adapted_named_params]

    @property
    def adapted_param_count(self) -> int:
        return self._adapted_param_count

    @property
    def adapted_param_fraction(self) -> float:
        return float(self._adapted_param_count) / float(max(1, self._total_params))

    def __enter__(self) -> "TTTSession":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.restore()
        return False

    @torch.no_grad()
    def restore(self) -> None:
        """Restore exact original snapshot weights (``atol=0``) and clear grads."""
        for name, p in self._adapted_named_params:
            snap = self._snapshot[name]
            p.copy_(snap.to(device=p.device, dtype=p.dtype))
            p.grad = None

    def reset(self) -> None:
        """Alias for ``restore()`` for instant reset between documents."""
        self.restore()

    def _autocast_ctx(self, device: torch.device) -> contextlib.AbstractContextManager[Any]:
        if self.use_bf16 and device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def _fast_constrained_pseudo_decode(
        self,
        image_tensor: torch.Tensor,
        max_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Run fast ``PrefixState``-constrained greedy decode to gather Loss 2 targets.

        Returns:
            inp_ids: ``(1, T)`` decoder input token IDs starting with ``BOS``.
            conf_mask: ``(1, T)`` float32 mask of high-confidence non-loop positions.
            valid_mask: ``(1, T)`` float32 mask of all valid (non-pad) decoded positions.
            ref_log_probs: ``(1, T, V)`` float32 log-probs from the pre-TTT snapshot.
            n_confident: integer count of tokens passing ``prob >= conf_threshold`` & non-loop.
        """
        device = image_tensor.device
        vocab_size = int(self.model.cfg.vocab_size)
        pad_id, bos_id, eos_id, _, tok_to_id, _ = ev._vocab_spec(vocab_size)

        with self._autocast_ctx(device):
            mem, _, _ = self.model.encode_image(image_tensor, return_aux=False)

        cur_id = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        past_kvs: Optional[List[Any]] = None
        state = grammar.PrefixState(strict=self.strict_grammar, vocab_map=tok_to_id)

        inp_seq: List[int] = []
        pred_ids: List[int] = []
        pred_probs: List[float] = []
        ref_lps: List[torch.Tensor] = []

        for _ in range(max(1, int(max_len))):
            inp_seq.append(int(cur_id.item()))
            with self._autocast_ctx(device):
                logits, _, past_kvs = self.model.decode_tokens(
                    cur_id,
                    memory=mem if past_kvs is None else None,
                    return_mtp=False,
                    past_kvs=past_kvs,
                    use_cache=True,
                    apply_tok_dropout=False,
                )
            step_logits = logits[0, -1, :].float()
            step_lp = F.log_softmax(step_logits, dim=-1)
            tok_id = ev._select_constrained_token(step_logits, state)
            tok_prob = float(step_lp[tok_id].exp().item())

            pred_ids.append(tok_id)
            pred_probs.append(tok_prob)
            ref_lps.append(step_lp.detach())

            if tok_id == eos_id:
                break
            state = ev._advance_grammar_state(
                state, tok_id, strict=self.strict_grammar, vocab_size=vocab_size
            )
            cur_id = torch.tensor([[tok_id]], dtype=torch.long, device=device)

        loop_flags = detect_loop_mask(pred_ids)
        conf_list = [
            1.0
            if (
                (pred_probs[i] >= self.conf_threshold)
                and (not loop_flags[i])
                and (pred_ids[i] not in (pad_id, bos_id))
            )
            else 0.0
            for i in range(len(pred_ids))
        ]
        valid_list = [
            1.0 if (pred_ids[i] not in (pad_id, bos_id)) else 0.0
            for i in range(len(pred_ids))
        ]
        n_confident = int(sum(1 for v in conf_list if v > 0.5))

        inp_ids = torch.tensor([inp_seq], dtype=torch.long, device=device)
        conf_mask = torch.tensor([conf_list], dtype=torch.float32, device=device)
        valid_mask = torch.tensor([valid_list], dtype=torch.float32, device=device)
        ref_log_probs = torch.stack(ref_lps, dim=0).unsqueeze(0)  # (1, T, V)
        return inp_ids, conf_mask, valid_mask, ref_log_probs, n_confident

    def adapt_on_page(
        self,
        image_tensor: torch.Tensor,
        steps: int = 3,
        max_decode_len: Optional[int] = None,
        mean_logprob: Optional[float] = None,
        conf_skip_thr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Adapt the selected parameters on a single unseen page.

        Args:
            image_tensor: ``(1, 3, H, W)`` (or ``(3, H, W)``) float tensor in ``[0, 1]``.
            steps: number of low-LR TTT gradient steps (typically 1–3).
            max_decode_len: optional override for the pre-TTT greedy decode length.
            mean_logprob: optional precomputed page confidence (``mean_logprob``) for the
                ``i13-06c`` confidence skip gate. When ``None`` and a skip threshold is
                active, it is computed via ``_fast_constrained_pseudo_decode``.
            conf_skip_thr: optional per-call override for ``self.conf_skip_thr``.

        Returns:
            Dictionary of per-step losses, gradient norms, confidence stats,
            skip-gate status, and elapsed wall time.
        """
        if image_tensor.ndim == 3:
            img = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 4:
            if image_tensor.shape[0] != 1:
                raise ValueError(
                    f"adapt_on_page expects batch size 1, got shape {tuple(image_tensor.shape)}"
                )
            img = image_tensor
        else:
            raise ValueError(
                f"Expected 3D or 4D image tensor, got shape {tuple(image_tensor.shape)}"
            )

        t0 = time.perf_counter()
        eff_skip_thr: Optional[float] = (
            self.conf_skip_thr if conf_skip_thr is None else float(conf_skip_thr)
        )
        eff_skip_str: Optional[str] = (
            format_conf_skip_thr(eff_skip_thr) if eff_skip_thr is not None else None
        )
        n_steps = int(steps)
        if n_steps <= 0:
            return {
                "steps": 0,
                "n_decoded_tokens": 0,
                "n_confident_tokens": 0,
                "skipped_by_conf_gate": False,
                "mean_logprob": round(float(mean_logprob), 5) if mean_logprob is not None else None,
                "conf_skip_thr": eff_skip_str,
                "loss_history": [],
                "cons_loss_history": [],
                "ent_loss_history": [],
                "kl_loss_history": [],
                "grad_norm_history": [],
                "elapsed_secs": 0.0,
            }

        # Fast-path confidence skip when caller already supplied mean_logprob
        if eff_skip_thr is not None and mean_logprob is not None:
            if should_skip_ttt_by_confidence(mean_logprob, eff_skip_thr):
                return {
                    "steps": 0,
                    "n_decoded_tokens": 0,
                    "n_confident_tokens": 0,
                    "skipped_by_conf_gate": True,
                    "mean_logprob": round(float(mean_logprob), 5),
                    "conf_skip_thr": eff_skip_str,
                    "loss_history": [],
                    "cons_loss_history": [],
                    "ent_loss_history": [],
                    "kl_loss_history": [],
                    "grad_norm_history": [],
                    "elapsed_secs": 0.0,
                }

        device = next(self.model.parameters()).device
        img = img.to(device=device)
        dec_len = self.max_decode_len if max_decode_len is None else int(max_decode_len)

        was_training = self.model.training
        orig_req_grad = {p: p.requires_grad for p in self.model.parameters()}

        loss_history: List[float] = []
        cons_loss_history: List[float] = []
        ent_loss_history: List[float] = []
        kl_loss_history: List[float] = []
        grad_norm_history: List[float] = []
        n_decoded = 0
        n_confident = 0
        obs_mean_lp: Optional[float] = (
            round(float(mean_logprob), 5) if mean_logprob is not None else None
        )

        try:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(id(p) in self._adapted_param_ids)

            need_pseudo_for_loss = (self.ent_weight > 0.0) or (self.kl_weight > 0.0)
            need_pseudo_for_gate = (eff_skip_thr is not None) and (obs_mean_lp is None)

            inp_ids = conf_mask = valid_mask = ref_log_probs = None
            if need_pseudo_for_loss or need_pseudo_for_gate:
                pseudo_cache = self._fast_constrained_pseudo_decode(img, max_len=dec_len)
                inp_ids, conf_mask, valid_mask, ref_log_probs, n_confident = pseudo_cache
                n_decoded = int(inp_ids.shape[1])
                if obs_mean_lp is None:
                    obs_mean_lp = compute_pseudo_mean_logprob(pseudo_cache)
                if eff_skip_thr is not None and should_skip_ttt_by_confidence(obs_mean_lp, eff_skip_thr):
                    elapsed = max(1e-6, time.perf_counter() - t0)
                    return {
                        "steps": 0,
                        "n_decoded_tokens": n_decoded,
                        "n_confident_tokens": n_confident,
                        "skipped_by_conf_gate": True,
                        "mean_logprob": obs_mean_lp,
                        "conf_skip_thr": eff_skip_str,
                        "loss_history": [],
                        "cons_loss_history": [],
                        "ent_loss_history": [],
                        "kl_loss_history": [],
                        "grad_norm_history": [],
                        "elapsed_secs": round(elapsed, 4),
                    }

            adapted_params = [p for _, p in self._adapted_named_params]
            snap_on_dev = [
                self._snapshot[name].to(device=p.device, dtype=torch.float32)
                for name, p in self._adapted_named_params
            ]

            optimizer = torch.optim.AdamW(
                adapted_params,
                lr=self.ttt_lr,
                betas=(0.9, 0.99),
                weight_decay=0.0,
            )

            with torch.enable_grad():
                for step_idx in range(n_steps):
                    optimizer.zero_grad(set_to_none=True)
                    img_aug = make_jittered_view(img, step_idx=step_idx, seed=self.seed)

                    with self._autocast_ctx(device):
                        mem_clean, _, rinfo_clean = self.model.encode_image(img, return_aux=False)
                        mem_aug, _, rinfo_aug = self.model.encode_image(img_aug, return_aux=False)

                    loss_cons, _, _ = compute_multiview_consistency_loss(
                        mem_clean, rinfo_clean, mem_aug, rinfo_aug
                    )

                    loss_ent = torch.zeros((), device=device, dtype=torch.float32)
                    loss_kl = torch.zeros((), device=device, dtype=torch.float32)

                    if (
                        n_confident > 0
                        and need_pseudo_for_loss
                        and inp_ids is not None
                        and conf_mask is not None
                        and valid_mask is not None
                        and ref_log_probs is not None
                    ):
                        with self._autocast_ctx(device):
                            logits, _ = self.model.decode_tokens(
                                inp_ids,
                                memory=mem_clean,
                                return_mtp=False,
                                use_cache=False,
                                apply_tok_dropout=False,
                            )
                        log_p = F.log_softmax(logits.float(), dim=-1)
                        if self.ent_weight > 0.0:
                            p_cur = log_p.exp()
                            ent_tok = -(p_cur * log_p).sum(dim=-1)  # (1, T)
                            loss_ent = (ent_tok * conf_mask).sum() / conf_mask.sum().clamp(min=1.0)

                        if self.kl_weight > 0.0:
                            ref_p = ref_log_probs.exp()
                            kl_tok = (ref_p * (ref_log_probs - log_p)).sum(dim=-1).clamp(min=0.0)
                            loss_kl = (kl_tok * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

                    loss_l2 = torch.zeros((), device=device, dtype=torch.float32)
                    if self.l2_weight > 0.0:
                        l2_terms = [
                            (p.float() - s0).pow(2).mean()
                            for p, s0 in zip(adapted_params, snap_on_dev)
                        ]
                        loss_l2 = torch.stack(l2_terms).mean()

                    total_loss = (
                        self.cons_weight * loss_cons
                        + self.ent_weight * loss_ent
                        + self.kl_weight * loss_kl
                        + self.l2_weight * loss_l2
                    )

                    if not bool(torch.isfinite(total_loss)):
                        # Safety guard: never apply a non-finite update.
                        optimizer.zero_grad(set_to_none=True)
                        break

                    total_loss.backward()
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        adapted_params, max_norm=self.grad_clip
                    )
                    grad_norm = float(grad_norm_t.item())
                    if not math.isfinite(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        break

                    optimizer.step()

                    loss_history.append(float(total_loss.detach().item()))
                    cons_loss_history.append(float(loss_cons.detach().item()))
                    ent_loss_history.append(float(loss_ent.detach().item()))
                    kl_loss_history.append(float(loss_kl.detach().item()))
                    grad_norm_history.append(grad_norm)

        finally:
            for p, req in orig_req_grad.items():
                p.requires_grad_(req)
            for _, p in self._adapted_named_params:
                p.grad = None
            self.model.train(was_training)

        elapsed = max(1e-6, time.perf_counter() - t0)
        return {
            "steps": len(loss_history),
            "n_decoded_tokens": n_decoded,
            "n_confident_tokens": n_confident,
            "skipped_by_conf_gate": False,
            "mean_logprob": obs_mean_lp,
            "conf_skip_thr": eff_skip_str,
            "loss_history": loss_history,
            "cons_loss_history": cons_loss_history,
            "ent_loss_history": ent_loss_history,
            "kl_loss_history": kl_loss_history,
            "grad_norm_history": grad_norm_history,
            "elapsed_secs": round(elapsed, 4),
        }


def resolve_ttt_mode_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve effective TTT hyperparameters from ``--ttt-mode`` preset or custom CLI args."""
    mode = str(getattr(args, "ttt_mode", "custom")).strip().lower()
    if mode not in VALID_TTT_MODES:
        raise ValueError(f"Unsupported --ttt-mode={mode!r}; expected one of {VALID_TTT_MODES}")
    if mode in TTT_MODE_PRESETS:
        cfg = dict(TTT_MODE_PRESETS[mode])
    else:
        cfg = {
            "adapt_scope": str(args.adapt_scope),
            "ttt_lr": float(args.ttt_lr),
            "ttt_steps": int(args.ttt_steps),
            "cons_weight": float(args.cons_weight),
            "ent_weight": float(args.ent_weight),
            "kl_weight": float(args.kl_weight),
            "l2_weight": float(args.l2_weight),
            "grad_clip": float(args.grad_clip),
        }
    cfg["ttt_mode"] = mode
    cfg["conf_skip_thr"] = (
        float(args.ttt_conf_skip_thr)
        if getattr(args, "ttt_conf_skip_thr", None) is not None
        else None
    )
    return cfg


def build_ttt_session(
    model: Any,
    mode: str = "backbone_cons",
    *,
    use_bf16: bool = False,
    strict_grammar: bool = False,
    conf_skip_thr: Optional[float] = None,
    max_decode_len: int = 512,
    seed: int = 2028,
) -> Tuple[TTTSession, int]:
    """Instantiate a ``TTTSession`` for a named promoted preset (e.g. ``backbone_cons``)."""
    key = str(mode).strip().lower()
    if key not in TTT_MODE_PRESETS:
        raise ValueError(
            f"Unsupported TTT preset mode={mode!r}; expected one of {tuple(TTT_MODE_PRESETS)}"
        )
    preset = TTT_MODE_PRESETS[key]
    session = TTTSession(
        model=model,
        ttt_lr=float(preset["ttt_lr"]),
        adapt_scope=str(preset["adapt_scope"]),
        cons_weight=float(preset["cons_weight"]),
        ent_weight=float(preset["ent_weight"]),
        kl_weight=float(preset["kl_weight"]),
        l2_weight=float(preset["l2_weight"]),
        grad_clip=float(preset["grad_clip"]),
        use_bf16=use_bf16,
        strict_grammar=strict_grammar,
        max_decode_len=max_decode_len,
        seed=seed,
        conf_skip_thr=conf_skip_thr,
    )
    return session, int(preset["ttt_steps"])


# ---------------------------------------------------------------------------
# Suite31+HW Before-vs-After TTT A/B Trial Runner
# ---------------------------------------------------------------------------

def _decode_and_score_page(
    model: uni_omr.UniOMRNet,
    cfg: uni_omr.UniOMRConfig,
    img_t: torch.Tensor,
    ref_page_toks: Sequence[str],
    page_ref_notes: Path,
    workdir: Path,
    tag_id: str,
    *,
    max_len: int,
    use_grammar: bool,
    strict_grammar: bool,
    use_mtp_spec: bool,
    use_bf16: bool,
    device: torch.device,
) -> Dict[str, Any]:
    """Decode one page image and score Token F1/LCS/SER + Note F1/SER."""
    t0 = time.perf_counter()
    with torch.no_grad():
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (use_bf16 and device.type == "cuda")
            else contextlib.nullcontext()
        )
        with ctx:
            if use_mtp_spec:
                seq, _ = ev.generate_mtp_spec(
                    model,
                    img_t,
                    max_len=max_len,
                    use_grammar=use_grammar,
                    strict_grammar=strict_grammar,
                    return_stats=True,
                )
            elif use_grammar:
                seq = ev.generate_constrained(
                    model,
                    img_t,
                    max_len=max_len,
                    strict=strict_grammar,
                )
            elif int(cfg.vocab_size) != uni_omr.VOCAB_SIZE:
                seq = ev.generate_mtp_spec(
                    model,
                    img_t,
                    max_len=max_len,
                    use_grammar=False,
                    return_stats=False,
                )
            else:
                seq = model.generate(img_t, max_len=max_len)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.perf_counter() - t0)

    seq_ids = seq[0].detach().cpu().tolist()
    _, _, eos_id, _, _, _ = ev._vocab_spec(int(cfg.vocab_size))
    pred_raw_toks = ev.decode_uni_compat(
        seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
    )
    if eos_id not in seq_ids[1:]:
        pred_raw_toks = ev.trim_truncated_tokens(pred_raw_toks)
    hyp_page_toks, n_sys = ev.stitch_uni_tokens(pred_raw_toks)

    tok_edits = uni_omr.levenshtein_distance(ref_page_toks, hyp_page_toks)
    tok_ser = tok_edits / max(1, len(ref_page_toks))
    _, _, tok_f1 = uni_omr.compute_token_f1(ref_page_toks, hyp_page_toks)
    tok_lcs = ev.compute_token_lcs_f1(ref_page_toks, hyp_page_toks)

    hyp_txt = workdir / f"{tag_id}-hyp-bekern.txt"
    hyp_txt.write_text(" ".join(hyp_page_toks) + "\n", encoding="utf-8")
    hyp_notes = workdir / f"{tag_id}-hyp-notes.json"
    p_dec = esg.run_cli([
        esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes),
    ])
    if p_dec.returncode != 0 or not hyp_notes.exists():
        note_f1 = 0.0
        note_ser = 1.0
        decode_ok = False
        zero_reason = f"bekern-decode-rc-{p_dec.returncode}"
    else:
        sc = esg.score_notes_pair(
            page_ref_notes,
            hyp_notes,
            workdir,
            tag_id,
            esg.EHARNESS,
            esg.ASER,
        )
        note_f1 = float(sc.get("f1", 0.0))
        note_ser = float(sc.get("ser", 1.0))
        decode_ok = not bool(sc.get("zeroed"))
        zero_reason = sc.get("zero_reason")

    return {
        "tok_f1": round(tok_f1, 6),
        "tok_lcs": round(tok_lcs, 6),
        "tok_ser": round(tok_ser, 6),
        "note_f1": round(note_f1, 6),
        "note_ser": round(note_ser, 6),
        "decode_ok": decode_ok,
        "zero_reason": zero_reason,
        "n_sys": n_sys,
        "n_pred_toks": len(hyp_page_toks),
        "elapsed_secs": round(elapsed, 4),
    }


def _mean_metric(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def run_suite31_ttt_ab(args: argparse.Namespace) -> Dict[str, Any]:
    """Run before-vs-after TTT A/B ablation on ``suite31_hw``."""
    device = torch.device(args.device)
    model, cfg, ckpt_meta = ev.load_uni_checkpoint(args.ckpt, device=device)
    _, pieces = ev.resolve_manifest_pieces(args.manifest)

    if not args.all_pieces:
        pieces = [p for p in pieces if p["handwritten"]]
    if args.pieces:
        wanted = {s.strip() for s in args.pieces.split(",") if s.strip()}
        pieces = [p for p in pieces if p["slug"] in wanted]
    if args.max_pieces is not None:
        pieces = pieces[: int(args.max_pieces)]

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    use_bf16 = bool(args.bf16 and device.type == "cuda")
    eff_cfg = resolve_ttt_mode_config(args)

    ttt = TTTSession(
        model=model,
        ttt_lr=eff_cfg["ttt_lr"],
        adapt_scope=eff_cfg["adapt_scope"],
        conf_threshold=args.conf_threshold,
        cons_weight=eff_cfg["cons_weight"],
        ent_weight=eff_cfg["ent_weight"],
        kl_weight=eff_cfg["kl_weight"],
        l2_weight=eff_cfg["l2_weight"],
        grad_clip=eff_cfg["grad_clip"],
        use_bf16=use_bf16,
        strict_grammar=args.strict_grammar,
        max_decode_len=args.ttt_max_decode_len,
        conf_skip_thr=eff_cfg["conf_skip_thr"],
    )
    skip_str = (
        f" | conf_skip_thr={format_conf_skip_thr(eff_cfg['conf_skip_thr'])}"
        if eff_cfg["conf_skip_thr"] is not None
        else ""
    )

    print(
        f"[2028 TTT A/B] ckpt={args.ckpt} (step={ckpt_meta['step']}) | device={device} | "
        f"mode={eff_cfg['ttt_mode']} | scope={ttt.adapt_scope} ({ttt.adapted_param_count:,} params = "
        f"{ttt.adapted_param_fraction * 100.0:.4f}%) | steps={eff_cfg['ttt_steps']} | "
        f"lr={eff_cfg['ttt_lr']:g}{skip_str} | pieces={len(pieces)} | domains={domains}",
        flush=True,
    )

    workdir = Path(tempfile.mkdtemp(prefix="uni_ttt_ab_"))
    rows: List[Dict[str, Any]] = []

    try:
        for idx, piece in enumerate(pieces, start=1):
            slug = piece["slug"]
            tag = "HW" if piece["handwritten"] else "PR"
            try:
                ref_page_toks, page_ref_notes = esg.suite31_page_refs(
                    piece["gt"], "ttt-" + slug, workdir
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[{idx:02d}/{len(pieces):02d}] {slug:34s} [{tag}] SKIP ref-error: {exc}", flush=True)
                continue

            for domain in domains:
                img_path: Path = piece[domain]
                with Image.open(img_path) as raw_im:
                    pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), args.max_dim)
                img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)

                # Before TTT (exact snapshot)
                ttt.restore()
                before = _decode_and_score_page(
                    model, cfg, img_t, ref_page_toks, page_ref_notes, workdir,
                    f"ttt-before-{slug}-{domain}",
                    max_len=args.max_len,
                    use_grammar=args.use_grammar,
                    strict_grammar=args.strict_grammar,
                    use_mtp_spec=args.use_mtp_spec,
                    use_bf16=use_bf16,
                    device=device,
                )

                # Adapt on page + evaluate After TTT
                ttt.restore()
                ttt_stats = ttt.adapt_on_page(img_t, steps=eff_cfg["ttt_steps"])
                if ttt_stats.get("skipped_by_conf_gate"):
                    after = dict(before)
                else:
                    after = _decode_and_score_page(
                        model, cfg, img_t, ref_page_toks, page_ref_notes, workdir,
                        f"ttt-after-{slug}-{domain}",
                        max_len=args.max_len,
                        use_grammar=args.use_grammar,
                        strict_grammar=args.strict_grammar,
                        use_mtp_spec=args.use_mtp_spec,
                        use_bf16=use_bf16,
                        device=device,
                    )
                ttt.restore()

                d_note = after["note_f1"] - before["note_f1"]
                d_tok = after["tok_f1"] - before["tok_f1"]
                rows.append({
                    "piece": slug,
                    "handwritten": piece["handwritten"],
                    "domain": domain,
                    "before": before,
                    "after": after,
                    "delta_note_f1": round(d_note, 6),
                    "delta_tok_f1": round(d_tok, 6),
                    "ttt_stats": ttt_stats,
                })

                print(
                    f"[{idx:02d}/{len(pieces):02d}] {slug:34s} [{tag}|{domain:5s}] "
                    f"noteF1: {before['note_f1']:.4f} -> {after['note_f1']:.4f} ({d_note:+.4f}) | "
                    f"tokF1: {before['tok_f1']:.4f} -> {after['tok_f1']:.4f} ({d_tok:+.4f}) | "
                    f"conf={ttt_stats['n_confident_tokens']}/{ttt_stats['n_decoded_tokens']} "
                    f"ttt={ttt_stats['elapsed_secs']:.2f}s",
                    flush=True,
                )
    finally:
        ttt.restore()

    summary: Dict[str, Any] = {}
    for domain in domains:
        drows = [r for r in rows if r["domain"] == domain]
        bef_rows = [r["before"] for r in drows]
        aft_rows = [r["after"] for r in drows]
        bef_nf1 = _mean_metric(bef_rows, "note_f1")
        aft_nf1 = _mean_metric(aft_rows, "note_f1")
        bef_tf1 = _mean_metric(bef_rows, "tok_f1")
        aft_tf1 = _mean_metric(aft_rows, "tok_f1")
        summary[domain] = {
            "n": len(drows),
            "before_note_f1": bef_nf1,
            "after_note_f1": aft_nf1,
            "delta_note_f1": round(aft_nf1 - bef_nf1, 6) if (bef_nf1 is not None and aft_nf1 is not None) else None,
            "before_tok_f1": bef_tf1,
            "after_tok_f1": aft_tf1,
            "delta_tok_f1": round(aft_tf1 - bef_tf1, 6) if (bef_tf1 is not None and aft_tf1 is not None) else None,
            "median_ttt_secs": (
                round(float(statistics.median(r["ttt_stats"]["elapsed_secs"] for r in drows)), 4)
                if drows
                else None
            ),
        }

    print("=" * 96, flush=True)
    print("2028 TTT A/B Summary", flush=True)
    print("=" * 96, flush=True)
    for domain, s in summary.items():
        print(
            f"  {domain:6s} (n={s['n']:2d}) | Note F1: {s['before_note_f1']} -> {s['after_note_f1']} "
            f"({s['delta_note_f1']}) | Tok F1: {s['before_tok_f1']} -> {s['after_tok_f1']} "
            f"({s['delta_tok_f1']}) | med_ttt={s['median_ttt_secs']}s",
            flush=True,
        )
    print("=" * 96, flush=True)

    report = {
        "ckpt": str(args.ckpt),
        "ckpt_meta": ckpt_meta,
        "model_config": asdict(cfg),
        "ttt_config": {
            "ttt_mode": str(eff_cfg["ttt_mode"]),
            "ttt_steps": int(eff_cfg["ttt_steps"]),
            "ttt_lr": float(eff_cfg["ttt_lr"]),
            "adapt_scope": str(eff_cfg["adapt_scope"]),
            "adapted_param_count": ttt.adapted_param_count,
            "adapted_param_fraction": round(ttt.adapted_param_fraction, 6),
            "conf_threshold": float(args.conf_threshold),
            "ttt_conf_skip_thr": eff_cfg["conf_skip_thr"],
            "ttt_conf_skip_thr_str": (
                format_conf_skip_thr(eff_cfg["conf_skip_thr"])
                if eff_cfg["conf_skip_thr"] is not None
                else None
            ),
            "cons_weight": float(eff_cfg["cons_weight"]),
            "ent_weight": float(eff_cfg["ent_weight"]),
            "kl_weight": float(eff_cfg["kl_weight"]),
            "l2_weight": float(eff_cfg["l2_weight"]),
            "grad_clip": float(eff_cfg["grad_clip"]),
        },
        "summary": summary,
        "rows": rows,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote TTT A/B report -> {out_path}", flush=True)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="2028 Trial: On-Device Test-Time Training (TTT) A/B Ablation on suite31_hw"
    )
    ap.add_argument("--ckpt", required=True, help="Path to UniOMR .pt checkpoint")
    ap.add_argument(
        "--manifest",
        default=str(ev.DEFAULT_MANIFEST),
        help="Path to suite31_hw_manifest.json",
    )
    ap.add_argument(
        "--ttt-mode",
        default="custom",
        choices=VALID_TTT_MODES,
        help=(
            "TTT preset mode: 'custom' uses explicit --adapt-scope/--ttt-lr/--ttt-steps flags; "
            "'backbone_cons' applies the promoted ov-38b preset "
            "(scope=backbone_router, lr=3e-5, steps=2, ent=0.0, kl=0.0, cons=1.0, l2=1e-2)."
        ),
    )
    ap.add_argument(
        "--ttt-conf-skip-thr",
        type=float,
        default=None,
        help=(
            "Optional 3-decimal mean_logprob confidence skip gate (i13-06c; "
            "promoted: -0.040 skips pages with mean_logprob > -0.040 untouched)."
        ),
    )
    ap.add_argument("--ttt-steps", type=int, default=3, help="TTT gradient steps per page")
    ap.add_argument("--ttt-lr", type=float, default=5e-5, help="TTT AdamW learning rate")
    ap.add_argument(
        "--adapt-scope",
        default="norms",
        choices=TTTSession.VALID_SCOPES,
        help="Parameter adaptation scope (default: norms = RMSNorm in backbone/enc_layers/enc_norm)",
    )
    ap.add_argument("--conf-threshold", type=float, default=0.85, help="Loss 2 confidence gate")
    ap.add_argument("--cons-weight", type=float, default=1.0, help="Loss 1 consistency weight")
    ap.add_argument("--ent-weight", type=float, default=1.0, help="Loss 2 conditional entropy weight")
    ap.add_argument("--kl-weight", type=float, default=0.5, help="KL anchor weight to snapshot")
    ap.add_argument("--l2-weight", type=float, default=1e-2, help="L2 weight anchor to snapshot")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="Max grad norm clip")
    ap.add_argument("--max-dim", type=int, default=1024, help="Full-page resize max dimension")
    ap.add_argument("--max-len", type=int, default=1024, help="Max autoregressive decode tokens")
    ap.add_argument(
        "--ttt-max-decode-len",
        type=int,
        default=512,
        help="Max decode tokens for pre-TTT pseudo-label pass",
    )
    ap.add_argument(
        "--no-grammar",
        dest="use_grammar",
        action="store_false",
        help="Disable FSA grammar-constrained decoding during A/B eval",
    )
    ap.set_defaults(use_grammar=True)
    ap.add_argument("--strict-grammar", action="store_true", help="Use strict bar-duration grammar")
    ap.add_argument("--use-mtp-spec", action="store_true", help="Use MTP-4 speculative decoding")
    ap.add_argument(
        "--all-pieces",
        action="store_true",
        help="Evaluate all 45 suite31_hw pieces (default: 14 unseen historical handwritten pieces)",
    )
    ap.add_argument("--domains", default="clean,photo", help="Comma-separated domains (clean,photo)")
    ap.add_argument("--pieces", default=None, help="Optional comma-separated piece slugs")
    ap.add_argument("--max-pieces", type=int, default=None, help="Cap number of pieces")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
        help="Execution device (cuda or cpu)",
    )
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast on CUDA")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    run_suite31_ttt_ab(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
