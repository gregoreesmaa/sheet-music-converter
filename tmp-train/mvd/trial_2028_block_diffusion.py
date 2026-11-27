"""2027-2028 Frontier Trial: Masked Block-Diffusion Measure-Refinement Prototype.

Lightweight bidirectional 4-layer Transformer denoiser (d_model=384, n_heads=6,
<6M parameters) that cross-attends to UniOMR encoder memory ``mem`` and
bidirectionally refines a fixed-window measure token block (L <= 128) using
MDLM / LLaDA-style discrete absorbing diffusion:

- Training (``compute_diffusion_loss``): samples random mask ratio
  ``t ~ Uniform(0.15, 0.85)`` per measure, replaces masked tokens with a
  dedicated ``<mask>`` token ID, and optimizes parallel cross-entropy weighted
  by ``1 / t``.
- Inference (``denoise_measure``): starts from either fully masked or
  low-confidence / duration-conflicting masked positions and runs ``steps=4``
  parallel confidence-guided unmasking passes (unmasking the top ``1/steps``
  most confident positions per pass) so conflicting beats and pitches resolve
  bidirectionally in 4 passes instead of ~60 sequential AR steps.

Standalone CPU smoke check:
    CUDA_VISIBLE_DEVICES="" .\\.venv\\Scripts\\python.exe tmp-train/mvd/trial_2028_block_diffusion.py --smoke
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from uni_omr import VOCAB_SIZE as _DEFAULT_VOCAB_SIZE
except Exception:  # noqa: BLE001
    _DEFAULT_VOCAB_SIZE = 265


@dataclass
class BlockDiffusionConfig:
    """Configuration for :class:`BlockDiffusionMeasureRefiner`."""

    vocab_size: int = _DEFAULT_VOCAB_SIZE
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: int = 2
    n_layers: int = 4
    d_ff: int = 768
    max_measure_len: int = 128
    dropout: float = 0.1
    mem_dim: Optional[int] = None
    t_min: float = 0.15
    t_max: float = 0.85
    mask_token_id: Optional[int] = None
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if not (0.0 < self.t_min < self.t_max <= 1.0):
            raise ValueError(
                f"Require 0 < t_min < t_max <= 1, got ({self.t_min}, {self.t_max})"
            )
        if self.mem_dim is None:
            self.mem_dim = self.d_model
        if self.mask_token_id is None:
            self.mask_token_id = self.vocab_size
        if self.mask_token_id < self.vocab_size:
            raise ValueError(
                f"mask_token_id ({self.mask_token_id}) must be >= vocab_size ({self.vocab_size}) "
                "so valid non-mask vocab IDs stay in [0, vocab_size)."
            )


class _GQAttention(nn.Module):
    """Bidirectional grouped-query multi-head attention (self or cross)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_groups = n_heads // n_kv_heads
        self.dropout = dropout

        kv_dim = n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = x.shape
        kv_len = kv.shape[1]

        q = self.q_proj(x).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(bsz, kv_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(bsz, kv_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        attn_mask: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, kv_len), True = pad/ignore -> float -inf bias
            attn_mask = torch.zeros(
                (bsz, 1, 1, kv_len), dtype=q.dtype, device=q.device
            )
            attn_mask = attn_mask.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        return self.o_proj(out)


class _BlockDiffusionLayer(nn.Module):
    """Pre-norm bidirectional self-attention + encoder cross-attention + FFN."""

    def __init__(self, cfg: BlockDiffusionConfig) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(cfg.d_model)
        self.self_attn = _GQAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            dropout=cfg.dropout,
        )

        self.norm_cross = nn.LayerNorm(cfg.d_model)
        self.cross_attn = _GQAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            dropout=cfg.dropout,
        )

        self.norm_ffn = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
            nn.Dropout(cfg.dropout),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        token_padding_mask: Optional[torch.Tensor] = None,
        mem_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.norm_self(x), self.norm_self(x), token_padding_mask))
        x = x + self.drop(self.cross_attn(self.norm_cross(x), mem, mem_padding_mask))
        x = x + self.ffn(self.norm_ffn(x))
        return x


class BlockDiffusionMeasureRefiner(nn.Module):
    """Bidirectional 4-layer Transformer denoiser for measure-block refinement.

    Cross-attends to encoder memory ``mem`` and predicts all masked measure
    tokens in parallel (L <= 128). Output logits cover ``[0, vocab_size)``
    only, so ``<mask>`` is never emitted as a decoded token.
    """

    def __init__(
        self,
        cfg: Optional[BlockDiffusionConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = BlockDiffusionConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either cfg or keyword overrides, not both.")

        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.mask_token_id = int(cfg.mask_token_id)
        self.pad_token_id = cfg.pad_token_id
        self.max_measure_len = cfg.max_measure_len

        total_tokens = self.mask_token_id + 1
        self.tok_emb = nn.Embedding(total_tokens, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_measure_len, cfg.d_model)
        self.t_proj = nn.Sequential(
            nn.Linear(1, cfg.d_model // 4),
            nn.SiLU(),
            nn.Linear(cfg.d_model // 4, cfg.d_model),
        )
        self.emb_drop = nn.Dropout(cfg.dropout)

        if cfg.mem_dim != cfg.d_model:
            self.mem_proj: nn.Module = nn.Linear(cfg.mem_dim, cfg.d_model, bias=False)
        else:
            self.mem_proj = nn.Identity()

        self.layers = nn.ModuleList(
            [_BlockDiffusionLayer(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.out_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        mem: torch.Tensor,
        tokens: torch.Tensor,
        token_padding_mask: Optional[torch.Tensor] = None,
        mem_padding_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-position vocab logits ``(B, L, vocab_size)``.

        Args:
            mem: Encoder memory tensor ``(B, M, mem_dim)``.
            tokens: Measure token IDs ``(B, L)`` with optional ``mask_token_id``.
            token_padding_mask: Optional bool mask ``(B, L)`` where ``True`` is pad.
            mem_padding_mask: Optional bool mask ``(B, M)`` where ``True`` is pad.
            t: Optional mask-ratio tensor ``(B,)`` in ``(0, 1]``.
        """
        bsz, seq_len = tokens.shape
        if seq_len > self.max_measure_len:
            raise ValueError(
                f"Measure block length {seq_len} exceeds max_measure_len={self.max_measure_len}"
            )

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = self.tok_emb(tokens) + self.pos_emb(positions)

        if t is None:
            t_in = (tokens == self.mask_token_id).float().mean(dim=1, keepdim=True)
        else:
            t_in = t.to(device=tokens.device, dtype=x.dtype).view(bsz, 1)
        x = x + self.t_proj(t_in).unsqueeze(1)
        x = self.emb_drop(x)

        mem_h = self.mem_proj(mem)
        for layer in self.layers:
            x = layer(
                x,
                mem_h,
                token_padding_mask=token_padding_mask,
                mem_padding_mask=mem_padding_mask,
            )

        x = self.final_norm(x)
        # Weight-tied output projection restricted strictly to non-mask vocab [0, vocab_size)
        logits = F.linear(x, self.tok_emb.weight[: self.vocab_size], self.out_bias)
        return logits

    def compute_diffusion_loss(
        self,
        mem: torch.Tensor,
        target_tokens: torch.Tensor,
        token_padding_mask: Optional[torch.Tensor] = None,
        mem_padding_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_metrics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """MDLM / LLaDA-style random-ratio masked diffusion training objective.

        Samples ``t ~ Uniform(t_min, t_max)`` (default ``[0.15, 0.85]``) per
        measure, masks each valid token independently with probability ``t``,
        and minimizes ``(1 / t) * (1 / L_valid) * sum_{i in masked} CE(x_i)``.
        """
        bsz, seq_len = target_tokens.shape
        device = target_tokens.device

        if token_padding_mask is not None:
            valid_mask = ~token_padding_mask
        else:
            valid_mask = target_tokens != self.pad_token_id
            # Fallback if a synthetic test passes all zeros as valid tokens
            all_empty = ~valid_mask.any(dim=1)
            if all_empty.any():
                valid_mask = valid_mask.clone()
                valid_mask[all_empty] = True

        if t is None:
            u = torch.rand((bsz,), device=device, generator=generator)
            t_vec = self.cfg.t_min + u * (self.cfg.t_max - self.cfg.t_min)
        else:
            t_vec = t.to(device=device, dtype=torch.float32).view(bsz).clamp(1e-3, 1.0)

        rand_vals = torch.rand((bsz, seq_len), device=device, generator=generator)
        mask = (rand_vals < t_vec.unsqueeze(1)) & valid_mask

        # Ensure at least one valid token is masked per sequence so loss is always finite.
        for b in range(bsz):
            if not mask[b].any():
                valid_idx = valid_mask[b].nonzero(as_tuple=False).squeeze(-1)
                if valid_idx.numel() > 0:
                    pick = torch.randint(
                        0, valid_idx.numel(), (1,), device=device, generator=generator
                    ).item()
                    mask[b, valid_idx[pick]] = True

        noisy_tokens = target_tokens.clone()
        noisy_tokens[mask] = self.mask_token_id

        logits = self.forward(
            mem=mem,
            tokens=noisy_tokens,
            token_padding_mask=token_padding_mask,
            mem_padding_mask=mem_padding_mask,
            t=t_vec,
        )

        ce = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_tokens.reshape(-1),
            reduction="none",
        ).reshape(bsz, seq_len)

        valid_counts = valid_mask.float().sum(dim=1).clamp_min(1.0)
        per_sample_loss = (ce * mask.float()).sum(dim=1) / (valid_counts * t_vec)
        loss = per_sample_loss.mean()

        if not return_metrics:
            return loss

        with torch.no_grad():
            masked_count = int(mask.sum().item())
            total_valid = max(int(valid_mask.sum().item()), 1)
            ce_unweighted = float(ce[mask].mean().item()) if masked_count > 0 else 0.0
            metrics: Dict[str, float] = {
                "loss": float(loss.item()),
                "ce_unweighted": ce_unweighted,
                "t_mean": float(t_vec.mean().item()),
                "masked_frac": float(masked_count / total_valid),
                "masked_count": float(masked_count),
            }
        return loss, metrics

    @torch.no_grad()
    def denoise_measure(
        self,
        mem: torch.Tensor,
        init_tokens: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
        steps: int = 4,
        temperature: float = 0.0,
        token_padding_mask: Optional[torch.Tensor] = None,
        mem_padding_mask: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, float]]]]:
        """Parallel confidence-guided iterative unmasking in ``steps`` passes.

        Args:
            mem: Encoder memory ``(B, M, mem_dim)``.
            init_tokens: Initial measure tokens ``(B, L)``.
            mask_positions: Optional bool mask ``(B, L)`` indicating positions
                to mask and refine (e.g., low-confidence or duration-conflicting
                spans). If omitted, positions equal to ``mask_token_id`` are used.
            steps: Number of parallel unmasking passes (default ``4``).
            temperature: Sampling temperature (``0.0`` = greedy argmax).
            token_padding_mask: Optional bool mask ``(B, L)`` where ``True`` is pad.
            mem_padding_mask: Optional bool mask ``(B, M)`` where ``True`` is pad.
            return_trace: If ``True``, also return per-step diagnostic trace.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        bsz, seq_len = init_tokens.shape
        tokens = init_tokens.clone()

        if mask_positions is None:
            pending = tokens == self.mask_token_id
        else:
            pending = mask_positions.to(device=tokens.device, dtype=torch.bool).clone()
            pending = pending | (tokens == self.mask_token_id)

        if token_padding_mask is not None:
            pending = pending & (~token_padding_mask)

        tokens[pending] = self.mask_token_id
        trace: List[Dict[str, float]] = []

        for step in range(steps):
            steps_left = steps - step
            t_cur = (
                pending.float().sum(dim=1) / max(float(seq_len), 1.0)
            ).clamp(min=1e-3, max=1.0)

            logits = self.forward(
                mem=mem,
                tokens=tokens,
                token_padding_mask=token_padding_mask,
                mem_padding_mask=mem_padding_mask,
                t=t_cur,
            )

            if temperature <= 0.0:
                probs = F.softmax(logits, dim=-1)
                conf, cand_ids = probs.max(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                flat_ids = torch.multinomial(
                    probs.reshape(-1, self.vocab_size), num_samples=1
                ).reshape(bsz, seq_len)
                cand_ids = flat_ids
                conf = probs.gather(-1, cand_ids.unsqueeze(-1)).squeeze(-1)

            step_unmasked = 0
            conf_vals: List[float] = []

            for b in range(bsz):
                rem_idx = pending[b].nonzero(as_tuple=False).squeeze(-1)
                n_rem = int(rem_idx.numel())
                if n_rem == 0:
                    continue

                if steps_left == 1:
                    k = n_rem
                else:
                    k = int(math.ceil(n_rem / float(steps_left)))
                    k = max(1, min(k, n_rem))

                rem_conf = conf[b, rem_idx]
                top_vals, top_rel = torch.topk(rem_conf, k=k, largest=True, sorted=False)
                chosen = rem_idx[top_rel]

                tokens[b, chosen] = cand_ids[b, chosen]
                pending[b, chosen] = False
                step_unmasked += k
                conf_vals.append(float(top_vals.mean().item()))

            trace.append(
                {
                    "step": float(step + 1),
                    "unmasked_count": float(step_unmasked),
                    "remaining_masked": float(pending.sum().item()),
                    "mean_conf": float(sum(conf_vals) / len(conf_vals)) if conf_vals else 1.0,
                }
            )

        if return_trace:
            return tokens, trace
        return tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2027-2028 Trial: Masked Block-Diffusion Measure-Refinement Prototype"
    )
    parser.add_argument("--smoke", action="store_true", help="Run CPU smoke verification")
    parser.add_argument("--vocab-size", type=int, default=_DEFAULT_VOCAB_SIZE)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = BlockDiffusionConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    model = BlockDiffusionMeasureRefiner(cfg)
    n_params = model.count_parameters()
    print(
        f"BlockDiffusionMeasureRefiner: params={n_params:,} ({n_params / 1e6:.2f}M) "
        f"d_model={cfg.d_model} n_heads={cfg.n_heads} n_layers={cfg.n_layers} "
        f"vocab={cfg.vocab_size} mask_id={cfg.mask_token_id}"
    )

    if args.smoke:
        bsz = 2
        mem = torch.randn(bsz, 48, cfg.d_model)
        targets = torch.randint(1, cfg.vocab_size, (bsz, args.seq_len))
        loss, metrics = model.compute_diffusion_loss(mem, targets, return_metrics=True)
        loss.backward()
        print(f"Train step OK: {metrics}")

        model.eval()
        init = torch.full((bsz, args.seq_len), cfg.mask_token_id, dtype=torch.long)
        mask_pos = torch.ones((bsz, args.seq_len), dtype=torch.bool)
        t0 = time.perf_counter()
        out, trace = model.denoise_measure(
            mem, init, mask_pos, steps=args.steps, return_trace=True
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        n_mask_left = int((out == cfg.mask_token_id).sum().item())
        print(
            f"Denoise OK ({args.steps} passes in {dt_ms:.1f} ms CPU): "
            f"remaining_mask={n_mask_left} min_id={int(out.min())} max_id={int(out.max())}"
        )
        for row in trace:
            print(f"  step={int(row['step'])} unmasked={int(row['unmasked_count'])} "
                  f"rem={int(row['remaining_masked'])} conf={row['mean_conf']:.4f}")


if __name__ == "__main__":
    main()
