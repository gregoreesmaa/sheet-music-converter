"""EAGLE-3 Multi-Layer Speculative Draft Head for UniOMR v2.

Implements a lightweight (~1.33M params at d=384, ~2.36M at d=512; < 3.5% of
target model params) EAGLE-3 speculative draft head that attaches cleanly to
``UniOMRv2Net`` via forward hooks with ZERO edits to ``uni_omr_v2.py``:

  1. ``extract_multilayer_hidden``:
     Forward-hook wrapper tapping 3 decoder layers ``(low=0, mid=L//2, high=L-1)``
     during ``UniOMRv2Net.decode_tokens`` without altering KV-cache tensors.

  2. ``Eagle3DraftHead``:
     - Multi-layer feature fusion: ``Linear(3 * d_model, d_model) + RMSNorm``.
     - Shifted token-embedding fusion: ``Linear(2 * d_model, d_model)``.
     - Single-layer causal GQA Transformer block (QK-Norm + RoPE1D + KV-cache +
       compact SwiGLU FFN) sharing the target model's frozen ``tok_emb`` and
       ``lm_head`` weights.

  3. ``compute_eagle3_ttt_loss``:
     EAGLE-3 Training-Time Test (TTT) loss unrolling ``K_draft`` autoregressive
     draft steps (feeding the drafter's own hidden state ``h_draft_t`` into step
     ``t+1``) and minimizing step-decayed KL divergence against frozen target
     ``UniOMRv2Net`` logits.

  4. ``eagle3_speculative_generate``:
     Lossless speculative decoding: proposes ``K_draft`` tokens autoregressively
     with ``Eagle3DraftHead`` (optionally masked by ``PrefixState`` FSA), verifies
     all ``K_draft`` tokens in 1 parallel forward pass of ``UniOMRv2Net`` with
     exact KV-cache rollback on first mismatch, and appends the bonus/correction
     token — guaranteeing 100% bit-identical greedy output to
     ``UniOMRv2Net.generate()``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from grammar import PrefixState  # noqa: E402
from uni_omr import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SYS_ID,
    UNI_ID_TO_TOKEN,
    RMSNorm,
    SwiGLU,
)
from uni_omr_v2 import (  # noqa: E402
    GroupedQueryAttention,
    UniOMRv2Config,
    UniOMRv2Net,
    _CachedRoPE1D,
)

__all__ = [
    "get_eagle3_tap_indices",
    "extract_multilayer_hidden",
    "Eagle3DraftHead",
    "compute_eagle3_ttt_loss",
    "eagle3_speculative_generate",
]


# ---------------------------------------------------------------------------
# 1. Multi-Layer Hidden Feature Capture (Zero-Edit Forward Hooks)
# ---------------------------------------------------------------------------

def get_eagle3_tap_indices(n_dec_layers: int) -> Tuple[int, int, int]:
    """Return the 3 EAGLE-3 decoder tap indices ``(low=0, mid=L//2, high=L-1)``."""
    if n_dec_layers <= 0:
        raise ValueError(f"n_dec_layers must be positive, got {n_dec_layers}")
    return (0, n_dec_layers // 2, n_dec_layers - 1)


def extract_multilayer_hidden(
    model: UniOMRv2Net,
    tgt_ids: torch.Tensor,
    memory: Optional[Any],
    *,
    past_kvs: Optional[List[Any]] = None,
    use_cache: bool = False,
    return_mtp: bool = False,
    return_aux: bool = False,
    apply_tok_dropout: bool = False,
    detach_hidden: bool = False,
) -> Tuple[torch.Tensor, Any, torch.Tensor]:
    """Run ``model.decode_tokens`` while capturing 3 decoder layer outputs via hooks.

    Taps ``(low=0, mid=L//2, high=L-1)`` from ``model.dec_layers`` without
    modifying layer outputs or breaking KV-cache state.

    Returns:
        ``(logits, past_kvs_or_aux, h_multi)`` where:
          - ``logits``: ``(B, T, V)`` target decoder logits
          - ``past_kvs_or_aux``: ``new_past_kvs`` if ``use_cache=True`` else ``sys_y_pred``
          - ``h_multi``: ``(B, T, 3 * d_model)`` concatenated ``[h_low, h_mid, h_high]``
    """
    n_layers = len(model.dec_layers)
    tap_indices = get_eagle3_tap_indices(n_layers)
    unique_indices = sorted(set(tap_indices))
    captured: Dict[int, torch.Tensor] = {}
    handles: List[Any] = []

    def _make_hook(layer_idx: int):
        def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = h.detach() if detach_hidden else h
        return _hook

    try:
        for idx in unique_indices:
            handles.append(model.dec_layers[idx].register_forward_hook(_make_hook(idx)))
        logits, _mtp_logits, third_out = model.decode_tokens(
            tgt_ids,
            memory=memory,
            return_mtp=return_mtp,
            return_aux=return_aux,
            past_kvs=past_kvs,
            use_cache=use_cache,
            apply_tok_dropout=apply_tok_dropout,
        )
    finally:
        for handle in handles:
            handle.remove()

    h_multi = torch.cat([captured[i] for i in tap_indices], dim=-1)
    return logits, third_out, h_multi


# ---------------------------------------------------------------------------
# 2. Eagle3DraftHead (~1.33M params at d=384, ~2.36M at d=512)
# ---------------------------------------------------------------------------

class Eagle3DraftHead(nn.Module):
    """Lightweight EAGLE-3 speculative draft head for ``UniOMRv2Net``.

    Architecture:
      - Multi-layer feature fusion: ``fc_fuse: Linear(3*D, D)`` + ``RMSNorm(D)``.
      - Shifted next-token fusion: ``input_proj: Linear(2*D, D)`` combining
        ``RMSNorm(tok_emb(next_id))`` and ``h_prev``.
      - Single causal GQA Transformer block (QK-Norm + 1D RoPE + KV-cache +
        compact SwiGLU FFN with ``ffn_mult=0.5``).
      - Projects to vocabulary logits using the target model's shared frozen
        ``lm_head`` weight (zero duplicate vocab parameters).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        ffn_mult: float = 0.5,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = self.d_model // self.n_heads
        self.d_ff = max(32, int(round(self.d_model * ffn_mult)))

        # 1. Multi-layer target hidden state fusion (3*D -> D)
        self.fc_fuse = nn.Linear(3 * self.d_model, self.d_model, bias=False)
        self.fuse_norm = RMSNorm(self.d_model)

        # 2. Token-embedding + hidden state fusion (2*D -> D)
        self.tok_norm = RMSNorm(self.d_model)
        self.input_proj = nn.Linear(2 * self.d_model, self.d_model, bias=False)

        # 3. Single causal GQA Transformer layer with QK-Norm + RoPE1D + KV-cache
        self.rope1d = _CachedRoPE1D(head_dim=self.head_dim, base=rope_base)
        self.norm1 = RMSNorm(self.d_model)
        self.self_attn = GroupedQueryAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            use_rope=True,
        )
        self.norm2 = RMSNorm(self.d_model)
        self.ffn = SwiGLU(self.d_model, self.d_ff, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.out_norm = RMSNorm(self.d_model)

    @classmethod
    def from_config(
        cls,
        cfg: UniOMRv2Config,
        ffn_mult: float = 0.5,
        dropout: float = 0.0,
    ) -> "Eagle3DraftHead":
        return cls(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            ffn_mult=ffn_mult,
            dropout=dropout,
            rope_base=cfg.rope_base,
        )

    @classmethod
    def from_model(
        cls,
        model: UniOMRv2Net,
        ffn_mult: float = 0.5,
        dropout: float = 0.0,
    ) -> "Eagle3DraftHead":
        return cls.from_config(model.cfg, ffn_mult=ffn_mult, dropout=dropout)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def fuse_hidden(self, h_multi: torch.Tensor) -> torch.Tensor:
        """Fuse concatenated 3-layer target hidden states ``(B, T, 3*D) -> (B, T, D)``."""
        return self.fuse_norm(self.fc_fuse(h_multi))

    def forward(
        self,
        h_prev: torch.Tensor,
        tok_emb_next: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Any:
        """Run one EAGLE-3 draft transformer pass.

        Args:
            h_prev: ``(B, T, D)`` fused target hidden state (step 0) or previous
                draft hidden state (step >= 1).
            tok_emb_next: ``(B, T, D)`` embeddings of shifted next tokens.
            past_kv: optional cached ``(k, v)`` tuple for autoregressive drafting.
            use_cache: whether to return updated ``(k, v)`` cache.

        Returns:
            ``h_draft`` ``(B, T, D)`` if ``use_cache=False``, else
            ``(h_draft, present_kv)``.
        """
        _, T, _ = h_prev.shape
        x = self.input_proj(torch.cat([self.tok_norm(tok_emb_next), h_prev], dim=-1))

        past_len = int(past_kv[0].shape[2]) if past_kv is not None else 0
        cos_all, sin_all = self.rope1d(past_len + T, device=x.device, dtype=x.dtype)
        cos1d = cos_all[:, :, past_len : past_len + T, :]
        sin1d = sin_all[:, :, past_len : past_len + T, :]

        h_norm = self.norm1(x)
        if use_cache:
            sa_out, present_kv = self.self_attn(
                h_norm,
                h_norm,
                rope_cos=cos1d,
                rope_sin=sin1d,
                is_causal=True,
                past_kv=past_kv,
                use_cache=True,
                is_cross=False,
            )
            x = x + self.drop(sa_out)
            x = x + self.drop(self.ffn(self.norm2(x)))
            return x, present_kv

        sa_out = self.self_attn(
            h_norm,
            h_norm,
            rope_cos=cos1d,
            rope_sin=sin1d,
            is_causal=True,
            past_kv=None,
            use_cache=False,
            is_cross=False,
        )
        x = x + self.drop(sa_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

    def project_logits(
        self,
        h_draft: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Project draft hidden states to vocab logits using frozen target ``lm_head``."""
        return F.linear(self.out_norm(h_draft), lm_head_weight.detach())


# ---------------------------------------------------------------------------
# 3. EAGLE-3 Training-Time Test (TTT) Loss
# ---------------------------------------------------------------------------

def compute_eagle3_ttt_loss(
    target_model: UniOMRv2Net,
    drafter: Eagle3DraftHead,
    tgt_ids: torch.Tensor,
    *,
    images: Optional[torch.Tensor] = None,
    memory: Optional[Any] = None,
    k_draft: int = 4,
    step_decay: float = 0.8,
    temperature: float = 1.0,
    keep_ratio_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute EAGLE-3 Training-Time Test (TTT) multi-step KL distillation loss.

    Unrolls ``k_draft`` autoregressive draft steps during training:
      - Step 0 takes fused multi-layer hidden states ``h_0 = fuse(h_multi)`` and
        shifted token embeddings ``tok_emb(tgt_ids[:, 1:])`` to predict target
        logits at positions ``1..T-1``.
      - Step ``k >= 1`` feeds the drafter's OWN predicted hidden state
        ``h_draft_{k-1}`` together with ``tok_emb(tgt_ids[:, k+1:])`` to predict
        target logits at positions ``k+1..T-1``.
      - Target ``UniOMRv2Net`` is strictly frozen (forward runs under
        ``torch.no_grad()`` and shared ``tok_emb`` / ``lm_head`` weights are
        detached), so gradients flow ONLY into ``drafter``.

    Args:
        target_model: Frozen target ``UniOMRv2Net``.
        drafter: Trainable ``Eagle3DraftHead``.
        tgt_ids: ``(B, T)`` teacher token sequence starting with ``BOS_ID``.
        images: ``(B, 3, H, W)`` input images (required if ``memory`` is None).
        memory: Optional pre-encoded DeepStack ``mem_levels`` from ``encode_image``.
        k_draft: Number of autoregressive unroll steps (default 4).
        step_decay: Exponential weight decay per draft step ``step_decay ** k``.
        temperature: Distillation temperature for KL divergence.
        keep_ratio_override: Optional router keep ratio passed to ``encode_image``.

    Returns:
        Dict with ``loss``, ``step_losses`` (list of scalar tensors), and
        ``k_steps_computed``.
    """
    if k_draft <= 0:
        raise ValueError(f"k_draft must be positive, got {k_draft}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if memory is None and images is None:
        raise ValueError("Either `memory` or `images` must be provided")

    # 1. Frozen target pass: extract target logits and 3-layer decoder hidden states.
    with torch.no_grad():
        if memory is None:
            memory, _, _ = target_model.encode_image(
                images, keep_ratio_override=keep_ratio_override, return_aux=False
            )
        target_logits, _, h_multi = extract_multilayer_hidden(
            target_model,
            tgt_ids,
            memory=memory,
            use_cache=False,
            return_mtp=False,
            return_aux=False,
            apply_tok_dropout=False,
            detach_hidden=True,
        )
        target_logits = target_logits.detach()
        h_multi = h_multi.detach()

    tok_weight = target_model.tok_emb.weight.detach()
    lm_weight = target_model.lm_head.weight.detach()

    # 2. Fuse multi-layer hidden states and unroll K_draft TTT steps.
    _, T = tgt_ids.shape
    h_cur = drafter.fuse_hidden(h_multi)

    step_losses: List[torch.Tensor] = []
    step_weights: List[float] = []
    inv_t = 1.0 / float(temperature)
    t_sq = float(temperature) ** 2

    for k in range(k_draft):
        shift = k + 1
        if T <= shift:
            break
        L_k = T - shift
        h_in = h_cur[:, :L_k, :]
        tok_in = tgt_ids[:, shift:T]
        tok_emb = F.embedding(tok_in, tok_weight)

        h_next = drafter(h_in, tok_emb, past_kv=None, use_cache=False)
        draft_logits = drafter.project_logits(h_next, lm_weight)

        tgt_slice = target_logits[:, shift:T, :]
        tgt_probs = F.softmax(tgt_slice.float() * inv_t, dim=-1)
        draft_log_probs = F.log_softmax(draft_logits.float() * inv_t, dim=-1)
        kl_per_tok = F.kl_div(draft_log_probs, tgt_probs, reduction="none").sum(dim=-1) * t_sq

        valid = (tok_in != PAD_ID).float()
        step_loss = (kl_per_tok * valid).sum() / valid.sum().clamp(min=1.0)
        step_losses.append(step_loss.to(h_next.dtype))
        step_weights.append(float(step_decay) ** k)

        # Feed drafter's own predicted hidden state into step k + 1.
        h_cur = h_next

    if not step_losses:
        zero = h_cur.new_zeros(())
        return {"loss": zero, "step_losses": [], "k_steps_computed": 0}

    weight_sum = sum(step_weights)
    total_loss = sum((w / weight_sum) * sl for w, sl in zip(step_weights, step_losses))
    return {
        "loss": total_loss,
        "step_losses": step_losses,
        "k_steps_computed": len(step_losses),
    }


# ---------------------------------------------------------------------------
# 4. Lossless Speculative Decoding (100% Exact Greedy Equivalence)
# ---------------------------------------------------------------------------

def _truncate_target_past_kvs(
    past_kvs: List[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]],
    keep_len: int,
) -> List[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
    """Roll back decoder self-attention KV cache to ``keep_len`` (cross-KV untouched)."""
    trimmed: List[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]] = []
    for self_kv, cross_kv in past_kvs:
        if self_kv is not None:
            k, v = self_kv
            self_kv = (k[:, :, :keep_len, :], v[:, :, :keep_len, :])
        trimmed.append((self_kv, cross_kv))
    return trimmed


def _select_greedy_token(
    logits_1d: torch.Tensor,
    grammar_state: Optional[PrefixState] = None,
) -> int:
    """Select greedy next token ID, optionally masked by ``PrefixState`` FSA."""
    if grammar_state is None:
        return int(logits_1d.argmax(dim=-1).item())

    allow_ids, eos_ok = grammar_state.allowed()
    vocab_size = logits_1d.shape[-1]
    mask = torch.full((vocab_size,), float("-inf"), device=logits_1d.device, dtype=logits_1d.dtype)
    valid_list = [tid for tid in allow_ids if 0 <= tid < vocab_size]
    if eos_ok:
        if 0 <= EOS_ID < vocab_size:
            valid_list.append(EOS_ID)
        if 0 <= SYS_ID < vocab_size:
            valid_list.append(SYS_ID)
    if not valid_list:
        return int(logits_1d.argmax(dim=-1).item())
    mask[valid_list] = 0.0
    return int((logits_1d + mask).argmax(dim=-1).item())


def _advance_grammar_state(
    grammar_state: Optional[PrefixState],
    tok_id: int,
    strict: bool = False,
) -> Optional[PrefixState]:
    """Advance (or reset on ``SYS_ID``) ``PrefixState`` by one token ID."""
    if grammar_state is None:
        return None
    if tok_id in (BOS_ID, EOS_ID, PAD_ID):
        return grammar_state
    if tok_id == SYS_ID:
        return PrefixState(strict=strict)
    tok_str = UNI_ID_TO_TOKEN.get(tok_id)
    if tok_str is not None:
        try:
            grammar_state.step(tok_str)
        except ValueError:
            grammar_state.broken = True
    return grammar_state


@torch.no_grad()
def _speculative_generate_single(
    model: UniOMRv2Net,
    drafter: Eagle3DraftHead,
    mem_single: Any,
    *,
    max_len: int = 256,
    k_draft: int = 4,
    use_grammar: bool = False,
    grammar_strict: bool = False,
) -> Tuple[List[int], Dict[str, Any]]:
    """Run lossless EAGLE-3 speculative decoding for a single image (B=1)."""
    device = mem_single[-1].device if isinstance(mem_single, (tuple, list)) else mem_single.device
    tok_weight = model.tok_emb.weight.detach()
    lm_weight = model.lm_head.weight.detach()

    generated: List[int] = [BOS_ID]
    stats: Dict[str, Any] = {
        "n_tokens_generated": 0,
        "n_target_passes": 0,
        "n_verify_rounds": 0,
        "n_draft_tokens": 0,
        "n_accepted_draft_tokens": 0,
        "mean_accept_len": 0.0,
        "tokens_per_target_pass": 0.0,
    }
    if max_len <= 0:
        return generated, stats

    past_kvs = model.precompute_cross_kv(mem_single)
    grammar_state: Optional[PrefixState] = PrefixState(strict=grammar_strict) if use_grammar else None

    # 1. Prefill BOS (y_0) with target model to obtain y_1 and initial h_fuse_0.
    bos_tensor = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    logits, past_kvs, h_multi = extract_multilayer_hidden(
        model,
        bos_tensor,
        memory=None,
        past_kvs=past_kvs,
        use_cache=True,
        return_mtp=False,
        return_aux=False,
        apply_tok_dropout=False,
        detach_hidden=True,
    )
    stats["n_target_passes"] += 1

    first_tok = _select_greedy_token(logits[0, -1], grammar_state)
    generated.append(first_tok)
    grammar_state = _advance_grammar_state(grammar_state, first_tok, strict=grammar_strict)
    if first_tok == EOS_ID or (len(generated) - 1) >= max_len:
        stats["n_tokens_generated"] = len(generated) - 1
        stats["tokens_per_target_pass"] = float(stats["n_tokens_generated"]) / max(1, stats["n_target_passes"])
        return generated, stats

    # Drafter state invariant:
    #   - `draft_kv` holds committed exact (h_fuse_t, y_{t+1}) pairs for t < t_pending.
    #   - `pending_h_fuse` (1, M, D) and `pending_toks` (len M) are exact newly
    #     accepted (h_fuse, next_tok) pairs not yet committed into `draft_kv`.
    #   - `past_kvs` covers [y_0 .. y_{L-1}]; `pending_toks[-1]` is y_L (not yet fed to target).
    draft_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    pending_h_fuse = drafter.fuse_hidden(h_multi)  # (1, 1, D)
    pending_toks: List[int] = [first_tok]

    while (len(generated) - 1) < max_len:
        remaining = max_len - (len(generated) - 1)
        k_eff = min(k_draft, remaining - 1)
        last_accepted = pending_toks[-1]

        # 2. Propose k_eff speculative tokens using tiny Eagle3DraftHead.
        draft_ids: List[int] = []
        if k_eff > 0:
            pend_tok_tensor = torch.tensor([pending_toks], dtype=torch.long, device=device)
            pend_emb = F.embedding(pend_tok_tensor, tok_weight)
            h_committed, draft_kv = drafter(
                pending_h_fuse, pend_emb, past_kv=draft_kv, use_cache=True
            )

            h_cur = h_committed[:, -1:, :]
            spec_kv = draft_kv
            draft_grammar = grammar_state.copy() if grammar_state is not None else None

            for step in range(k_eff):
                d_logits = drafter.project_logits(h_cur, lm_weight)
                d_tok = _select_greedy_token(d_logits[0, -1], draft_grammar)
                draft_ids.append(d_tok)
                draft_grammar = _advance_grammar_state(draft_grammar, d_tok, strict=grammar_strict)
                if d_tok == EOS_ID or step == k_eff - 1:
                    break
                d_emb = F.embedding(
                    torch.tensor([[d_tok]], dtype=torch.long, device=device), tok_weight
                )
                h_cur, spec_kv = drafter(h_cur, d_emb, past_kv=spec_kv, use_cache=True)

            stats["n_draft_tokens"] += len(draft_ids)

        # 3. Verify [last_accepted, *draft_ids] in ONE parallel target forward pass.
        past_len_before = int(past_kvs[0][0][0].shape[2])
        verify_seq = [last_accepted] + draft_ids
        verify_ids = torch.tensor([verify_seq], dtype=torch.long, device=device)
        verify_logits, verify_past_kvs, verify_h_multi = extract_multilayer_hidden(
            model,
            verify_ids,
            memory=None,
            past_kvs=past_kvs,
            use_cache=True,
            return_mtp=False,
            return_aux=False,
            apply_tok_dropout=False,
            detach_hidden=True,
        )
        stats["n_target_passes"] += 1
        if len(draft_ids) > 0:
            stats["n_verify_rounds"] += 1

        # 4. Lossless greedy verification + bonus/correction token.
        n_accept = 0
        bonus_tok: Optional[int] = None
        hit_eos = False

        for j, d_tok in enumerate(draft_ids):
            target_tok = _select_greedy_token(verify_logits[0, j], grammar_state)
            if target_tok == d_tok:
                n_accept += 1
                generated.append(d_tok)
                grammar_state = _advance_grammar_state(grammar_state, d_tok, strict=grammar_strict)
                if d_tok == EOS_ID or (len(generated) - 1) >= max_len:
                    hit_eos = True
                    break
            else:
                bonus_tok = target_tok
                generated.append(bonus_tok)
                grammar_state = _advance_grammar_state(grammar_state, bonus_tok, strict=grammar_strict)
                if bonus_tok == EOS_ID or (len(generated) - 1) >= max_len:
                    hit_eos = True
                break
        else:
            # All proposed draft tokens matched (or k_eff == 0): emit bonus token!
            bonus_tok = _select_greedy_token(verify_logits[0, len(draft_ids)], grammar_state)
            generated.append(bonus_tok)
            grammar_state = _advance_grammar_state(grammar_state, bonus_tok, strict=grammar_strict)
            if bonus_tok == EOS_ID or (len(generated) - 1) >= max_len:
                hit_eos = True

        stats["n_accepted_draft_tokens"] += n_accept
        if hit_eos:
            break

        # 5. Exact KV-cache rollback to keep [y_0 .. y_{L-1}, last_accepted, d_1 .. d_{n_accept}].
        keep_len = past_len_before + n_accept + 1
        past_kvs = _truncate_target_past_kvs(verify_past_kvs, keep_len)

        # Queue exact target fused hidden states for next draft round.
        verify_h_fuse = drafter.fuse_hidden(verify_h_multi[:, : n_accept + 1, :])
        pending_h_fuse = verify_h_fuse
        assert bonus_tok is not None
        pending_toks = draft_ids[:n_accept] + [bonus_tok]

    stats["n_tokens_generated"] = len(generated) - 1
    stats["mean_accept_len"] = (
        float(stats["n_accepted_draft_tokens"]) / max(1, stats["n_verify_rounds"])
    )
    stats["tokens_per_target_pass"] = (
        float(stats["n_tokens_generated"]) / max(1, stats["n_target_passes"])
    )
    return generated, stats


@torch.no_grad()
def eagle3_speculative_generate(
    model: UniOMRv2Net,
    drafter: Eagle3DraftHead,
    images: torch.Tensor,
    *,
    max_len: int = 256,
    k_draft: int = 4,
    keep_ratio_override: Optional[float] = None,
    use_grammar: bool = False,
    grammar_strict: bool = False,
    return_stats: bool = False,
) -> Any:
    """Lossless EAGLE-3 speculative decoding for ``UniOMRv2Net``.

    Guarantees 100% bit-identical greedy token output to ``UniOMRv2Net.generate()``
    (when ``use_grammar=False``), while verifying ``k_draft`` draft tokens in 1
    target forward pass per round.

    Args:
        model: Target ``UniOMRv2Net`` (eval mode recommended).
        drafter: ``Eagle3DraftHead`` attached to ``model``.
        images: ``(B, 3, H, W)`` input images.
        max_len: Maximum tokens to generate after ``BOS_ID``.
        k_draft: Speculative draft horizon per round (typically 4..8).
        keep_ratio_override: Optional router keep ratio passed to ``encode_image``.
        use_grammar: Whether to mask draft & verify logits with ``PrefixState`` FSA.
        grammar_strict: Whether to enable strict validator rules on ``PrefixState``.
        return_stats: If True, returns ``(seq, stats_dict)``; else returns ``seq``.

    Returns:
        ``seq`` tensor of shape ``(B, L)`` (padded with ``PAD_ID`` if ``B > 1``),
        or ``(seq, stats)`` if ``return_stats=True``.
    """
    if k_draft <= 0:
        raise ValueError(f"k_draft must be positive, got {k_draft}")

    mem_levels, _, router_info = model.encode_image(
        images, keep_ratio_override=keep_ratio_override, return_aux=False
    )
    B = int(images.shape[0])
    device = images.device

    seqs: List[List[int]] = []
    per_item_stats: List[Dict[str, Any]] = []
    for b in range(B):
        if isinstance(mem_levels, (tuple, list)):
            mem_b = tuple(m[b : b + 1] for m in mem_levels)
        else:
            mem_b = mem_levels[b : b + 1]
        seq_b, st_b = _speculative_generate_single(
            model,
            drafter,
            mem_b,
            max_len=max_len,
            k_draft=k_draft,
            use_grammar=use_grammar,
            grammar_strict=grammar_strict,
        )
        seqs.append(seq_b)
        per_item_stats.append(st_b)

    max_seq_len = max(len(s) for s in seqs)
    out = torch.full((B, max_seq_len), PAD_ID, dtype=torch.long, device=device)
    for b, s in enumerate(seqs):
        out[b, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)

    if return_stats:
        total_gen = sum(s["n_tokens_generated"] for s in per_item_stats)
        total_passes = sum(s["n_target_passes"] for s in per_item_stats)
        total_rounds = sum(s["n_verify_rounds"] for s in per_item_stats)
        total_draft = sum(s["n_draft_tokens"] for s in per_item_stats)
        total_acc = sum(s["n_accepted_draft_tokens"] for s in per_item_stats)
        agg = {
            "router_info": router_info,
            "n_tokens_generated": total_gen,
            "n_target_passes": total_passes,
            "n_verify_rounds": total_rounds,
            "n_draft_tokens": total_draft,
            "n_accepted_draft_tokens": total_acc,
            "mean_accept_len": float(total_acc) / max(1, total_rounds),
            "tokens_per_target_pass": float(total_gen) / max(1, total_passes),
            "per_item": per_item_stats,
        }
        return out, agg
    return out
