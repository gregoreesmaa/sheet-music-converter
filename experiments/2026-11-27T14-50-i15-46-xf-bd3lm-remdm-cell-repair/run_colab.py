#!/usr/bin/env python3
"""Colab A100 / Local RTX 4080 SUPER runner — Iter 15 Exp i15-46: BD3-LM + ReMDM-conf Cell Repair.

Experiment ID: 2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair
Cross-field origin:
  - Discrete Diffusion: BD3-LM (ICLR 2025 Oral) + ReMDM-conf confidence-guided remasking repair

Controlled 2-arm proxy on edge85m + ax0_gl6 (600 steps/arm, BF16, seed=42):
  - Arm 0 (ctrl):           Stock edge85m + ax0_gl6 causal AR decoder.
  - Arm 1 (bd3_remdm_head): Attaches a lightweight 2-layer bidirectional intra-cell
                            masked-diffusion / remasking repair head (d_rep=256,
                            n_heads=4, d_ff=512, block_size=32 bidirectional self-attn
                            + visual cross-attn, sharing tok_emb & lm_head; < 1.5 MB INT4)
                            trained with complementary masking (15-40% mask + 10%
                            synthetic single-token corruptions) alongside causal AR.

Measures per arm:
  - raw_val_ce, ema_val_ce, int4_dequant_val_ce (standard causal AR)
  - 2-step ReMDM-conf bidirectional repair accuracy on corrupted/masked spans
    AND on lowest-20%-confidence AR-predicted spans (bd3_remdm_head arm)
  - step_ms, train_elapsed_sec, peak_vram_mb, exact packed_int4_mb (< 55.0 MB)

Includes 1-arm HF partial checkpoint sync + startup auto-resume and Windows cuDNN 9.10 safety.
Security: reads HF_TOKEN from /tmp/.hf_token (chmod 600), HF_TOKEN env, or ~/.cache/huggingface/token;
never prints it, always unlinks /tmp/.hf_token in finally.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXP_ID = "2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair"
EXP_CODE = "i15-46"
ARMS = ("ctrl", "bd3_remdm_head")
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_CKPT_PREFIX = f"checkpoints/{EXP_ID}"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORK_ROOT = Path("/vol/i15_46") if Path("/vol").is_dir() else Path("/content/i15_46")

SEED = 42
TRAIN_STEPS = 600
WARMUP_STEPS = 60
QAT_START_STEP = 420
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR_MAX = 3e-4
WEIGHT_DECAY = 0.04
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
EMA_DECAY = 0.99
INT4_GROUP_SIZE = 32
HARD_BUDGET_MB = 55.0
REPAIR_HEAD_MAX_DELTA_MB = 1.5
SAVE_EVERY_MINUTES = 5.0
REPAIR_LOSS_WEIGHT = 0.25
REPAIR_BLOCK_SIZE = 32


# ---------------------------------------------------------------------------
# Experimental Module: BD3-LM + ReMDM-conf Bidirectional Intra-Cell Repair Head
# ---------------------------------------------------------------------------

def _build_bd3_repair_head_class() -> Any:
    """Factory returning BD3ReMDMRepairHead using uni_omr_v2 primitives (< 1.5 MB INT4)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import uni_omr_v2

    class _BD3RepairLayer(nn.Module):
        """Lightweight bidirectional intra-cell self-attn + visual cross-attn + SwiGLU."""

        def __init__(self, d_rep: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.d_rep = int(d_rep)
            self.n_heads = int(n_heads)
            self.head_dim = self.d_rep // self.n_heads

            self.norm1 = uni_omr_v2.RMSNorm(d_rep)
            self.qkv_proj = nn.Linear(d_rep, 3 * d_rep, bias=False)
            self.sa_o_proj = nn.Linear(d_rep, d_rep, bias=False)
            self.sa_q_norm = uni_omr_v2.RMSNorm(self.head_dim)
            self.sa_k_norm = uni_omr_v2.RMSNorm(self.head_dim)

            self.norm2 = uni_omr_v2.RMSNorm(d_rep)
            self.ca_q_proj = nn.Linear(d_rep, d_rep, bias=False)
            self.ca_kv_proj = nn.Linear(d_rep, 2 * d_rep, bias=False)
            self.ca_o_proj = nn.Linear(d_rep, d_rep, bias=False)
            self.ca_q_norm = uni_omr_v2.RMSNorm(self.head_dim)
            self.ca_k_norm = uni_omr_v2.RMSNorm(self.head_dim)

            self.norm3 = uni_omr_v2.RMSNorm(d_rep)
            self.ffn = uni_omr_v2.SwiGLU(d_rep, d_ff, dropout=dropout)
            self.drop = nn.Dropout(dropout)

        def forward(
            self,
            x: torch.Tensor,
            mem_rep: torch.Tensor,
            rope_cos: torch.Tensor,
            rope_sin: torch.Tensor,
            block_size: int = REPAIR_BLOCK_SIZE,
        ) -> torch.Tensor:
            B, T, D = x.shape

            # 1. Bidirectional intra-cell (block-local) self-attention (BD3-LM style)
            h = self.norm1(x)
            pad_t = (block_size - (T % block_size)) % block_size
            if pad_t > 0:
                h_pad = F.pad(h, (0, 0, 0, pad_t))
                cos_pad = F.pad(rope_cos, (0, 0, 0, pad_t))
                sin_pad = F.pad(rope_sin, (0, 0, 0, pad_t))
            else:
                h_pad = h
                cos_pad = rope_cos
                sin_pad = rope_sin

            T_pad = h_pad.shape[1]
            n_blk = T_pad // block_size
            h_blk = h_pad.view(B * n_blk, block_size, D)
            q, k, v = self.qkv_proj(h_blk).chunk(3, dim=-1)
            q = self.sa_q_norm(q.view(B * n_blk, block_size, self.n_heads, self.head_dim)).transpose(1, 2)
            k = self.sa_k_norm(k.view(B * n_blk, block_size, self.n_heads, self.head_dim)).transpose(1, 2)
            v = v.view(B * n_blk, block_size, self.n_heads, self.head_dim).transpose(1, 2)

            # Slice intra-block RoPE (0..block_size-1) so every cell shares consistent relative phases
            c_blk = cos_pad[:, :, :block_size, :]
            s_blk = sin_pad[:, :, :block_size, :]
            q, k = uni_omr_v2.apply_rope(q, k, c_blk, s_blk)

            sa_out = uni_omr_v2._sdpa(q, k, v, attn_mask=None, is_causal=False, enable_gqa=False)
            sa_out = sa_out.transpose(1, 2).reshape(B, T_pad, D)[:, :T, :]
            x = x + self.drop(self.sa_o_proj(sa_out))

            # 2. Visual cross-attention to projected encoder memory
            h2 = self.norm2(x)
            Lk = mem_rep.shape[1]
            cq = self.ca_q_norm(self.ca_q_proj(h2).view(B, T, self.n_heads, self.head_dim)).transpose(1, 2)
            ck, cv = self.ca_kv_proj(mem_rep).chunk(2, dim=-1)
            ck = self.ca_k_norm(ck.view(B, Lk, self.n_heads, self.head_dim)).transpose(1, 2)
            cv = cv.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
            ca_out = uni_omr_v2._sdpa(cq, ck, cv, attn_mask=None, is_causal=False, enable_gqa=False)
            ca_out = ca_out.transpose(1, 2).reshape(B, T, D)
            x = x + self.drop(self.ca_o_proj(ca_out))

            # 3. SwiGLU FFN
            x = x + self.drop(self.ffn(self.norm3(x)))
            return x

    class BD3ReMDMRepairHead(nn.Module):
        """Lightweight 2-layer bidirectional intra-cell masked-diffusion / remasking repair head.

        At edge85m (d_model=512), using d_rep=256, n_heads=4, d_ff=512, n_layers=2:
        ~2.23M params -> ~1.26 MB packed INT4 (< 1.5 MB budget).
        Shares model.tok_emb and model.lm_head (zero extra vocab table params).
        """

        def __init__(
            self,
            d_model: int,
            d_rep: int = 256,
            n_heads: int = 4,
            d_ff: int = 512,
            n_layers: int = 2,
            block_size: int = REPAIR_BLOCK_SIZE,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.d_model = int(d_model)
            self.d_rep = int(d_rep)
            self.block_size = int(block_size)
            head_dim = self.d_rep // int(n_heads)

            self.in_proj = nn.Linear(self.d_model, self.d_rep, bias=False)
            self.mem_proj = nn.Linear(self.d_model, self.d_rep, bias=False)
            self.mask_emb = nn.Parameter(torch.randn(1, 1, self.d_rep) * 0.02)
            self.rope1d = uni_omr_v2._CachedRoPE1D(head_dim=head_dim, base=10000.0)
            self.layers = nn.ModuleList([
                _BD3RepairLayer(self.d_rep, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
                for _ in range(n_layers)
            ])
            self.out_norm = uni_omr_v2.RMSNorm(self.d_rep)
            self.out_proj = nn.Linear(self.d_rep, self.d_model, bias=False)

        def forward(
            self,
            tok_embeds: torch.Tensor,
            mask_flags: torch.Tensor,
            mem_final: torch.Tensor,
            lm_head: nn.Linear,
        ) -> torch.Tensor:
            """Args:
                tok_embeds: (B, T, D) from model.tok_emb(corrupt_ids)
                mask_flags: (B, T) bool where True -> replace with learned mask_emb
                mem_final:  (B, L_mem, D) encoder visual memory
                lm_head:    shared output projection (D -> vocab_size)
            Returns:
                repair_logits: (B, T, vocab_size)
            """
            compute_dtype = uni_omr_v2._active_compute_dtype(tok_embeds)
            x = self.in_proj(tok_embeds.to(compute_dtype))
            m_emb = self.mask_emb.to(dtype=x.dtype).expand_as(x)
            x = torch.where(mask_flags.unsqueeze(-1), m_emb, x)

            mem_rep = self.mem_proj(mem_final.to(compute_dtype))
            T = x.shape[1]
            cos1d, sin1d = self.rope1d(max(T, self.block_size), device=x.device, dtype=x.dtype)
            cos1d = cos1d[:, :, :T, :]
            sin1d = sin1d[:, :, :T, :]

            for layer in self.layers:
                x = layer(x, mem_rep, cos1d, sin1d, block_size=self.block_size)

            h_out = self.out_proj(self.out_norm(x))
            return lm_head(h_out)

    return BD3ReMDMRepairHead


def _sample_bd3_complementary_views(
    tgt_y: "torch.Tensor",
    vocab_size: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> List[Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]]:
    """Build 2 complementary masked/corrupted views (mask M and complement ~M within 15-40% budget).

    Each view returns (corrupt_ids, mask_flags, supervise_flags):
      - mask_flags: True for [MASK] positions (replaced by learned mask_emb)
      - supervise_flags: True for both [MASK] positions and synthetic single-token corruptions
    """
    import torch

    valid = (tgt_y != pad_id) & (tgt_y != bos_id) & (tgt_y != eos_id)
    p_mask = random.uniform(0.15, 0.40)
    u = torch.rand(tgt_y.shape, device=tgt_y.device)

    # Complementary mask pair: view A masks u < p_mask, view B masks (1 - u) < p_mask
    mask_a = (u < p_mask) & valid
    mask_b = ((1.0 - u) < p_mask) & valid

    views: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for m in (mask_a, mask_b):
        # Also inject 10% single-token random corruptions on unmasked valid positions
        corrupt_unmasked = (torch.rand(tgt_y.shape, device=tgt_y.device) < 0.10) & valid & (~m)
        rand_toks = torch.randint(4, vocab_size, tgt_y.shape, device=tgt_y.device, dtype=tgt_y.dtype)
        corrupt_ids = torch.where(corrupt_unmasked, rand_toks, tgt_y)
        supervise = m | corrupt_unmasked
        if not bool(supervise.any()):
            # Guarantee at least 1 supervised position if valid tokens exist
            supervise = valid
            m = valid
        views.append((corrupt_ids, m, supervise))
    return views


def compute_bd3_repair_loss(
    model: Any,
    mem_final: "torch.Tensor",
    tgt_y: "torch.Tensor",
) -> "torch.Tensor":
    """Complementary masked-diffusion + single-token corruption repair loss."""
    import torch
    import torch.nn.functional as F
    import uni_omr

    views = _sample_bd3_complementary_views(
        tgt_y,
        vocab_size=int(model.cfg.vocab_size),
        pad_id=uni_omr.PAD_ID,
        bos_id=uni_omr.BOS_ID,
        eos_id=uni_omr.EOS_ID,
    )
    total_loss = torch.zeros((), device=tgt_y.device, dtype=torch.float32)
    for corrupt_ids, mask_flags, supervise in views:
        tok_emb = model.tok_emb(corrupt_ids)
        logits = model.bd3_repair_head(tok_emb, mask_flags, mem_final, model.lm_head)
        targets = torch.where(supervise, tgt_y, torch.full_like(tgt_y, uni_omr.PAD_ID))
        loss_v = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=uni_omr.PAD_ID,
        )
        total_loss = total_loss + loss_v
    return 0.5 * total_loss


def eval_remdm_2step_repair(
    model: Any,
    val_samples: List[Tuple["torch.Tensor", "torch.Tensor"]],
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, float]:
    """Evaluate 2-step ReMDM-conf bidirectional repair on:
      1. Deterministically corrupted ground-truth spans (12.5% [MASK] + 12.5% random substitution)
      2. Lowest-20%-confidence AR-predicted spans (actual AR model errors / uncertain tokens)
    """
    import torch
    import uni_omr

    if not hasattr(model, "bd3_repair_head"):
        return {}

    model.eval()
    c_total = 0
    c_hit0 = 0
    c_hit1 = 0
    c_hit2 = 0

    ar_lc_total = 0
    ar_lc_hit_before = 0
    ar_lc_hit_after = 0

    with torch.no_grad():
        for idx, (inp_cpu, tgt_cpu) in enumerate(val_samples):
            inp = inp_cpu.to(device)
            tgt = tgt_cpu.to(device)
            tgt_y = tgt[:, 1:]  # (1, T)
            valid = (tgt_y != uni_omr.PAD_ID) & (tgt_y != uni_omr.BOS_ID) & (tgt_y != uni_omr.EOS_ID)
            if int(valid.sum().item()) < 4:
                continue

            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                mem_lvls, _, _ = model.encode_image(inp, return_aux=False)
                mem_final = mem_lvls[-1]
                ar_logits, _, _ = model.decode_tokens(
                    tgt[:, :-1], mem_lvls, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )

            # --- Task 1: 25% corrupted spans (half [MASK], half single-token substitution) ---
            g = torch.Generator(device="cpu").manual_seed(SEED * 997 + idx * 31 + 7)
            u = torch.rand(tgt_y.shape, generator=g).to(device)
            mask_span = (u < 0.125) & valid
            sub_span = (u >= 0.125) & (u < 0.25) & valid
            corrupt_span = mask_span | sub_span
            if int(corrupt_span.sum().item()) >= 2:
                rand_toks = torch.randint(
                    4, int(model.cfg.vocab_size), tgt_y.shape, generator=g, dtype=torch.long
                ).to(device)
                cur_ids = torch.where(sub_span, rand_toks, tgt_y)
                c_total += int(corrupt_span.sum().item())
                c_hit0 += int(((cur_ids == tgt_y) & (~mask_span) & corrupt_span).sum().item())

                # Step 1: Bidirectional repair pass
                with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    logits1 = model.bd3_repair_head(model.tok_emb(cur_ids), mask_span, mem_final, model.lm_head)
                probs1 = torch.softmax(logits1.float(), dim=-1)
                conf1, pred1 = probs1.max(dim=-1)
                step1_ids = torch.where(corrupt_span, pred1, cur_ids)
                c_hit1 += int(((step1_ids == tgt_y) & corrupt_span).sum().item())

                # ReMDM-conf remask: keep top-50% confident repaired tokens, remask bottom-50%
                span_confs = conf1[corrupt_span]
                med_conf = torch.median(span_confs)
                remask_2 = corrupt_span & (conf1 <= med_conf)

                # Step 2: Second bidirectional repair pass conditioned on high-confidence anchors
                with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    logits2 = model.bd3_repair_head(model.tok_emb(step1_ids), remask_2, mem_final, model.lm_head)
                pred2 = logits2.argmax(dim=-1)
                step2_ids = torch.where(remask_2, pred2, step1_ids)
                c_hit2 += int(((step2_ids == tgt_y) & corrupt_span).sum().item())

            # --- Task 2: Lowest-20%-confidence AR-predicted spans ---
            ar_probs = torch.softmax(ar_logits.float(), dim=-1)
            ar_conf, ar_pred = ar_probs.max(dim=-1)
            valid_confs = ar_conf[valid]
            k_low = max(2, int(round(0.20 * valid_confs.numel())))
            thr = torch.topk(valid_confs, k=k_low, largest=False).values.max()
            low_conf_mask = valid & (ar_conf <= thr)
            if int(low_conf_mask.sum().item()) >= 2:
                ar_lc_total += int(low_conf_mask.sum().item())
                ar_lc_hit_before += int(((ar_pred == tgt_y) & low_conf_mask).sum().item())

                # 2-step ReMDM-conf starting from AR predictions with low-confidence positions masked
                with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    r_log1 = model.bd3_repair_head(model.tok_emb(ar_pred), low_conf_mask, mem_final, model.lm_head)
                r_p1 = torch.softmax(r_log1.float(), dim=-1)
                r_c1, r_id1 = r_p1.max(dim=-1)
                s1 = torch.where(low_conf_mask, r_id1, ar_pred)
                med_r1 = torch.median(r_c1[low_conf_mask])
                rm2 = low_conf_mask & (r_c1 <= med_r1)
                with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    r_log2 = model.bd3_repair_head(model.tok_emb(s1), rm2, mem_final, model.lm_head)
                s2 = torch.where(rm2, r_log2.argmax(dim=-1), s1)
                ar_lc_hit_after += int(((s2 == tgt_y) & low_conf_mask).sum().item())

    return {
        "remdm_corrupt_acc_step0": round(c_hit0 / max(1, c_total), 5),
        "remdm_corrupt_acc_step1": round(c_hit1 / max(1, c_total), 5),
        "remdm_corrupt_acc_step2": round(c_hit2 / max(1, c_total), 5),
        "remdm_corrupt_gain_2step": round((c_hit2 - c_hit0) / max(1, c_total), 5),
        "remdm_ar_lowconf_acc_before": round(ar_lc_hit_before / max(1, ar_lc_total), 5),
        "remdm_ar_lowconf_acc_after": round(ar_lc_hit_after / max(1, ar_lc_total), 5),
        "remdm_ar_lowconf_gain": round((ar_lc_hit_after - ar_lc_hit_before) / max(1, ar_lc_total), 5),
    }


def build_arm_model(arm_name: str, grad_checkpoint: bool = True, tiny: bool = False) -> Any:
    """Build edge85m + ax0_gl6 (or tiny d=64 for smoke test) for 'ctrl' or 'bd3_remdm_head'."""
    import uni_omr_v2

    if tiny:
        cfg = uni_omr_v2.UniOMRv2Config(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            n_axial_layers=0,
            n_global_layers=2,
            n_dec_layers=2,
            ffn_mult=2.0,
            axial_ffn_mult=2.0,
            backbone_channels=(16, 32, 48, 64),
            backbone_blocks=(1, 1, 1, 1),
            backbone_expand=2.0,
            router_hidden=32,
            max_mem_tokens=256,
            mtp_k=4,
            grad_checkpoint=grad_checkpoint,
        )
    else:
        cfg = uni_omr_v2.UniOMRv2Config.edge85m(
            n_axial_layers=0,
            n_global_layers=6,
            grad_checkpoint=grad_checkpoint,
        )
    model = uni_omr_v2.UniOMRv2Net(cfg)
    if arm_name == "bd3_remdm_head":
        BD3ReMDMRepairHead = _build_bd3_repair_head_class()
        if tiny:
            model.bd3_repair_head = BD3ReMDMRepairHead(
                d_model=cfg.d_model, d_rep=32, n_heads=2, d_ff=64, n_layers=2, block_size=8, dropout=cfg.dropout
            )
        else:
            model.bd3_repair_head = BD3ReMDMRepairHead(
                d_model=cfg.d_model,
                d_rep=256,
                n_heads=4,
                d_ff=512,
                n_layers=2,
                block_size=REPAIR_BLOCK_SIZE,
                dropout=cfg.dropout,
            )
    elif arm_name != "ctrl":
        raise ValueError(f"Unknown arm_name: {arm_name}")
    return model


# ---------------------------------------------------------------------------
# Infrastructure Helpers (Windows Safety, HF Auth & Partial Resume, Dataset, INT4)
# ---------------------------------------------------------------------------

def _configure_backend_safety() -> None:
    import torch

    if sys.platform == "win32":
        torch.backends.cudnn.enabled = False
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_hf_token() -> str:
    if HF_TOKEN_FILE.is_file():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        tok = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    cache_tok = Path.home() / ".cache" / "huggingface" / "token"
    if cache_tok.is_file():
        tok = cache_tok.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    raise RuntimeError("HF token missing: provide /tmp/.hf_token, HF_TOKEN env var, or ~/.cache/huggingface/token")


def _purge_hf_token() -> None:
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass


def _locate_mvd() -> Path:
    candidates = [
        Path.cwd() / "tmp-train" / "mvd",
        Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path("/workspace/mvd"),
        Path("/content/mvd"),
        WORK_ROOT / "mvd",
    ]
    for cand in candidates:
        if (cand / "uni_omr_v2.py").is_file() and (cand / "train_uni_v2.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    for bundle in (Path("/content/bundle.tar.gz"), Path("/content/mvd_bundle.tar.gz")):
        if bundle.is_file():
            WORK_ROOT.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORK_ROOT)
            for p in WORK_ROOT.rglob("uni_omr_v2.py"):
                if str(p.parent) not in sys.path:
                    sys.path.insert(0, str(p.parent))
                return p.parent
    raise FileNotFoundError("Could not locate tmp-train/mvd/uni_omr_v2.py")


def _ensure_deps() -> None:
    needed = []
    for mod, pkg in (("huggingface_hub", "huggingface_hub>=0.26"), ("PIL", "pillow"), ("numpy", "numpy")):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])


def _try_resume_from_hf(hf_token: str) -> Dict[str, Dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    recovered: Dict[str, Dict[str, Any]] = {}
    dl_dir = WORK_ROOT / "hf_resume"
    dl_dir.mkdir(parents=True, exist_ok=True)

    try:
        ev_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_EVAL_PATH,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(dl_dir),
        )
        ev_data = json.loads(Path(ev_path).read_text(encoding="utf-8"))
        for arm_name, arm_res in (ev_data.get("arms") or {}).items():
            if arm_name in ARMS and isinstance(arm_res, dict) and "ema_val_ce" in arm_res:
                recovered[arm_name] = arm_res
    except Exception:
        pass

    for arm_name in ARMS:
        if arm_name in recovered:
            continue
        try:
            p = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"{HF_CKPT_PREFIX}/{arm_name}_arm_result.json",
                repo_type="dataset",
                token=hf_token,
                local_dir=str(dl_dir),
            )
            arm_res = json.loads(Path(p).read_text(encoding="utf-8"))
            if isinstance(arm_res, dict) and "ema_val_ce" in arm_res:
                recovered[arm_name] = arm_res
        except Exception:
            pass

    for arm_name, arm_res in recovered.items():
        cached = WORK_ROOT / "ckpts" / arm_name / "arm_result.json"
        if not cached.is_file():
            _atomic_write_json(cached, arm_res)
        print(f"[{EXP_ID}] Auto-resumed completed arm '{arm_name}' from HF Private Hub", flush=True)

    return recovered


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = WORK_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"
    if train_abs.is_file() and val_abs.is_file():
        return train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = WORK_ROOT / "hf_dl"
    dl_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR} from {HF_REPO_ID}...", flush=True)
    tar_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PROXY_TAR,
        repo_type="dataset",
        token=hf_token,
        local_dir=str(dl_dir),
    )
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=ds_dir)
    try:
        os.remove(tar_path)
    except OSError:
        pass

    for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
        src_path = ds_dir / src_name
        out_lines: List[str] = []
        for raw in src_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            rec["image"] = str((ds_dir / rec["image"]).resolve())
            rec["bekerns"] = [str((ds_dir / b).resolve()) for b in rec.get("bekerns", [])]
            out_lines.append(json.dumps(rec))
        dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"[{EXP_ID}] Prepared {dst_path.name} ({len(out_lines)} rows)", flush=True)

    print(f"[{EXP_ID}] Dataset ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return train_abs, val_abs


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _wsd_mult(step: int, total_steps: int, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    decay_steps = max(1, int(round(total_steps * WSD_DECAY_FRAC)))
    decay_start = total_steps - decay_steps
    if step <= decay_start:
        return 1.0
    prog = min(1.0, max(0.0, float(step - decay_start) / float(decay_steps)))
    return MIN_LR_RATIO + 0.5 * (1.0 - MIN_LR_RATIO) * (1.0 + math.cos(math.pi * prog))


def _load_manifest_pairs(manifest_path: Path) -> List[Tuple[Path, List[Path], str, float]]:
    pairs: List[Tuple[Path, List[Path], str, float]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        img_p = Path(rec["image"])
        beks = [Path(b) for b in rec.get("bekerns", [])]
        domain = str(rec.get("domain", "")).strip().lower()
        if img_p.is_file() and beks:
            is_dense_or_hw = (len(beks) >= 4) or ("hw" in domain) or ("hand" in domain) or ("debussy" in domain)
            weight = 2.0 if is_dense_or_hw else 1.0
            pairs.append((img_p, beks, domain, weight))
    return pairs


def _encode_bekerns_full(beks: List[Path]) -> List[int]:
    import uni_omr

    toks: List[str] = []
    for idx, bp in enumerate(beks):
        if idx > 0:
            toks.append(uni_omr.SYS_TOKEN)
        raw = bp.read_text(encoding="utf-8", errors="ignore").strip()
        if raw:
            toks.extend(raw.split())
    return uni_omr.encode_uni(toks)


def _pil_to_tensor(img_pil: Any, max_dim: int) -> "torch.Tensor":
    import numpy as np
    import torch
    from PIL import Image

    rgb = img_pil.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, float(max_dim) / float(max(w, h)))
    nw = max(32, int(round(w * scale / 16.0)) * 16)
    nh = max(32, int(round(h * scale / 16.0)) * 16)
    if (nw, nh) != (w, h):
        rgb = rgb.resize((nw, nh), Image.BICUBIC)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return (t - 0.5) / 0.5


def _attach_qat_with_repair_head(model: Any, block_size: int = INT4_GROUP_SIZE) -> Any:
    """Attach QATInt4ControllerV2 including bd3_repair_head Linear layers when present."""
    import torch.nn as nn
    import uni_omr_v2

    qat = uni_omr_v2.QATInt4ControllerV2(model, block_size=block_size)
    if hasattr(model, "bd3_repair_head") and isinstance(model.bd3_repair_head, nn.Module):
        for sub_name, mod in model.bd3_repair_head.named_modules():
            if isinstance(mod, nn.Linear) and mod not in qat._orig_forwards:
                full_name = f"bd3_repair_head.{sub_name}" if sub_name else "bd3_repair_head"
                qat._orig_forwards[mod] = mod.forward
                qat.target_names.append(full_name)
                qat._patch_linear(mod)
    return qat


def _pack_and_verify_hybrid_int4(
    model: Any,
    arm_name: str,
    out_bin: Path,
    group_size: int = INT4_GROUP_SIZE,
    tiny: bool = False,
) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    infer_model = build_arm_model(arm_name, grad_checkpoint=False, tiny=tiny).cpu().eval()
    infer_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True)
    infer_model.strip_aux_heads()
    infer_model.mtp_heads = nn.ModuleList()

    int4_roots: List[nn.Module] = [
        infer_model.encoder.axial_layers,
        infer_model.encoder.patch_merger,
        infer_model.encoder.global_layers,
        infer_model.deepstack_fusions,
        infer_model.dec_layers,
        infer_model.lm_head,
    ]
    if hasattr(infer_model, "bd3_repair_head") and isinstance(infer_model.bd3_repair_head, nn.Module):
        int4_roots.append(infer_model.bd3_repair_head)

    int4_module_ids = set()
    for root in int4_roots:
        for mod in root.modules():
            if isinstance(mod, nn.Linear):
                int4_module_ids.add(id(mod))

    int4_param_names = set()
    for name, mod in infer_model.named_modules():
        if id(mod) in int4_module_ids:
            int4_param_names.add(f"{name}.weight" if name else "weight")

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    total_int4_bytes = 0
    total_scale_bytes = 0
    total_fp16_bytes = 0
    n_int4_tensors = 0
    n_fp16_tensors = 0

    dequant_sd: Dict[str, torch.Tensor] = {}
    with open(out_bin, "wb") as f:
        for name, param in infer_model.named_parameters():
            t = param.detach().float()
            if name in int4_param_names:
                flat = t.reshape(-1)
                orig_n = flat.numel()
                pad = (group_size - (orig_n % group_size)) % group_size
                if pad:
                    flat = torch.nn.functional.pad(flat, (0, pad))
                g = flat.reshape(-1, group_size)
                amax = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                scale = (amax / 7.0).to(torch.float16)
                q = torch.clamp(torch.round(g / scale.float()), -8, 7).to(torch.int8)
                u = (q.to(torch.int16) + 8).to(torch.uint8).reshape(-1)
                if u.numel() % 2 == 1:
                    u = torch.nn.functional.pad(u, (0, 1))
                packed = (u[0::2] | (u[1::2] << 4)).numpy()
                scales_np = scale.squeeze(-1).numpy()
                f.write(packed.tobytes())
                f.write(scales_np.tobytes())
                total_int4_bytes += int(packed.nbytes)
                total_scale_bytes += int(scales_np.nbytes)
                n_int4_tensors += 1
                dq = (q.float() * scale.float()).reshape(-1)[:orig_n].reshape(t.shape)
                dequant_sd[name] = dq.to(param.dtype)
            else:
                fp16_np = t.to(torch.float16).numpy()
                f.write(fp16_np.tobytes())
                total_fp16_bytes += int(fp16_np.nbytes)
                n_fp16_tensors += 1
                dequant_sd[name] = t.to(torch.float16).to(param.dtype)

    disk_bytes = out_bin.stat().st_size
    disk_mb = disk_bytes / 1_000_000.0
    infer_model.load_state_dict(dequant_sd, strict=False)
    return {
        "disk_bytes": int(disk_bytes),
        "packed_int4_mb": round(disk_mb, 3),
        "passes_55mb_budget": bool(disk_mb < HARD_BUDGET_MB),
        "int4_weight_mb": round(total_int4_bytes / 1_000_000.0, 3),
        "int4_scales_mb": round(total_scale_bytes / 1_000_000.0, 3),
        "fp16_keep_mb": round(total_fp16_bytes / 1_000_000.0, 3),
        "n_int4_tensors": n_int4_tensors,
        "n_fp16_tensors": n_fp16_tensors,
        "dequant_model": infer_model,
    }


def _preload_val_samples(
    val_pairs: List[Tuple[Path, List[Path], str, float]],
) -> List[Tuple["torch.Tensor", "torch.Tensor"]]:
    import torch
    from PIL import Image

    loaded: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for img_p, beks, _, _ in val_pairs:
        ids = _encode_bekerns_full(beks)
        if len(ids) < 3:
            continue
        img = Image.open(img_p).convert("RGB")
        inp = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0)
        tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long).unsqueeze(0)
        loaded.append((inp, tgt))
    return loaded


def _eval_val_ce(
    model: Any,
    val_samples: List[Tuple["torch.Tensor", "torch.Tensor"]],
    device: "torch.device",
    use_bf16: bool,
) -> float:
    import torch
    import torch.nn.functional as F
    import uni_omr

    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for inp_cpu, tgt_cpu in val_samples:
            inp = inp_cpu.to(device)
            tgt = tgt_cpu.to(device)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                mem, _, _ = model.encode_image(inp, return_aux=False)
                logits, _, _ = model.decode_tokens(
                    tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID
                )
            losses.append(float(loss.item()))
    return round(sum(losses) / max(1, len(losses)), 5)


def _collate_microbatch(
    samples: List[Tuple[Path, List[Path], str, float]],
    device: "torch.device",
) -> Optional[Tuple["torch.Tensor", "torch.Tensor"]]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr

    imgs: List[torch.Tensor] = []
    seqs: List[List[int]] = []
    for img_p, beks, _, _ in samples:
        ids = _encode_bekerns_full(beks)
        if len(ids) < 3:
            continue
        img = Image.open(img_p).convert("RGB")
        imgs.append(_pil_to_tensor(img, FULLPAGE_MAX_DIM))
        seqs.append(ids[:MAX_SEQ_LEN])
    if not imgs:
        return None

    max_h = max(t.shape[1] for t in imgs)
    max_w = max(t.shape[2] for t in imgs)
    padded_imgs = [
        F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1]), value=1.0)
        for t in imgs
    ]
    max_t = max(len(s) for s in seqs)
    tgt = torch.full((len(seqs), max_t), uni_omr.PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        tgt[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return torch.stack(padded_imgs, dim=0).to(device), tgt.to(device)


def _train_arm(
    arm_name: str,
    *,
    train_pairs: List[Tuple[Path, List[Path], str, float]],
    val_samples: List[Tuple["torch.Tensor", "torch.Tensor"]],
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    import train_uni_v2

    _seed_everything(SEED)
    arm_dir = WORK_ROOT / "ckpts" / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached = arm_dir / "arm_result.json"
    if cached.is_file():
        print(f"[{EXP_ID}] Reusing cached result for {arm_name}", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))

    model = build_arm_model(arm_name, grad_checkpoint=True).to(device)
    psummary = model.param_summary(int4_block_size=INT4_GROUP_SIZE)
    qat = _attach_qat_with_repair_head(model, block_size=INT4_GROUP_SIZE)
    opt = train_uni_v2.build_optimizer_v2(
        model,
        optimizer_name="muon",
        lr=LR_MAX,
        weight_decay=WEIGHT_DECAY,
        device_type=device.type,
        no_decay_1d=True,
    )
    if device.type == "cuda" and opt.mode != "muon+adamw":
        raise RuntimeError(f"[{EXP_ID}][{arm_name}] Silent fallback detected: expected muon+adamw, got {opt.mode}")
    print(f"[{EXP_ID}][{arm_name}] Optimizer verified: mode={opt.mode}", flush=True)

    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    weights = [p[3] for p in train_pairs]
    rng = random.Random(SEED)
    start_step = 1
    loss_hist: List[float] = []
    repair_loss_hist: List[float] = []
    step_ms_hist: List[float] = []

    latest_pt = arm_dir / "latest.pt"
    if latest_pt.is_file():
        ck = torch.load(str(latest_pt), map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        ema_state = {k: v.to(device) for k, v in ck["ema_state_dict"].items()}
        opt_list = getattr(opt, "optimizers", [o for o, _ in opt._opts])
        if "opt_state_dicts" in ck and len(ck["opt_state_dicts"]) == len(opt_list):
            for o_obj, sd in zip(opt_list, ck["opt_state_dicts"]):
                o_obj.load_state_dict(sd)
        if "rng_state" in ck:
            rng.setstate(ck["rng_state"])
        start_step = int(ck["step"]) + 1
        loss_hist = list(ck.get("loss_hist", []))
        repair_loss_hist = list(ck.get("repair_loss_hist", []))
        step_ms_hist = list(ck.get("step_ms_hist", []))
        if start_step > QAT_START_STEP:
            qat.enable()
        print(f"[{EXP_ID}][{arm_name}] Resumed local checkpoint at step {start_step}/{TRAIN_STEPS}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_arm0 = time.perf_counter()
    t_last_save = time.time()

    model.train()
    for step in range(start_step, TRAIN_STEPS + 1):
        if step == QAT_START_STEP:
            qat.enable()
            print(f"[{EXP_ID}][{arm_name}] Enabled QATInt4ControllerV2 at step {step}", flush=True)

        mult = _wsd_mult(step, TRAIN_STEPS, WARMUP_STEPS)
        opt.set_lr(mult)

        t_step0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_rep_loss = 0.0
        for _ in range(GRAD_ACCUM):
            batch_samples = rng.choices(train_pairs, weights=weights, k=BATCH_SIZE)
            collated = _collate_microbatch(batch_samples, device)
            if collated is None:
                continue
            inp, tgt = collated
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                mem_lvls, aux_enc, router_info = model.encode_image(inp, return_aux=False)
                logits, mtp_logits, sys_y_pred = model.decode_tokens(
                    tgt[:, :-1], mem_lvls, return_mtp=True, return_aux=False
                )
                out = {
                    "logits": logits,
                    "mtp_logits": mtp_logits,
                    "router_info": router_info,
                    "sys_y_pred": sys_y_pred,
                    "sys_count_pred": aux_enc["sys_count_pred"],
                    "staff_seg_logits": aux_enc["staff_seg_logits"],
                    "stroke_hog_pred": aux_enc["stroke_hog_pred"],
                }
                losses = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)
                total_l = losses["loss"]
                if hasattr(model, "bd3_repair_head"):
                    rep_l = compute_bd3_repair_loss(model, mem_lvls[-1], tgt[:, 1:])
                    total_l = total_l + REPAIR_LOSS_WEIGHT * rep_l
                    step_rep_loss += float(rep_l.item()) / GRAD_ACCUM
                loss = total_l / GRAD_ACCUM
            loss.backward()
            step_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_state[k].mul_(EMA_DECAY).add_(v, alpha=1.0 - EMA_DECAY)
                else:
                    ema_state[k].copy_(v)

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_ms_hist.append((time.perf_counter() - t_step0) * 1000.0)
        loss_hist.append(step_loss)
        if hasattr(model, "bd3_repair_head"):
            repair_loss_hist.append(step_rep_loss)

        if step % 100 == 0 or step == 1:
            mean_l = sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:]))
            mean_ms = sum(step_ms_hist[-50:]) / max(1, len(step_ms_hist[-50:]))
            rep_str = (
                f" rep_loss={sum(repair_loss_hist[-50:]) / max(1, len(repair_loss_hist[-50:])):.4f}"
                if repair_loss_hist
                else ""
            )
            print(
                f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} "
                f"loss={mean_l:.4f}{rep_str} step_ms={mean_ms:.1f} elapsed={time.perf_counter() - t_arm0:.1f}s",
                flush=True,
            )

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ck = arm_dir / "latest.pt.tmp"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema_state,
                    "opt_state_dicts": [o.state_dict() for o in getattr(opt, "optimizers", [o for o, _ in opt._opts])],
                    "rng_state": rng.getstate(),
                    "loss_hist": loss_hist,
                    "repair_loss_hist": repair_loss_hist,
                    "step_ms_hist": step_ms_hist,
                },
                tmp_ck,
            )
            os.replace(tmp_ck, latest_pt)
            t_last_save = time.time()

    train_elapsed_sec = time.perf_counter() - t_arm0
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )

    qat.disable()
    raw_val_ce = _eval_val_ce(model, val_samples, device, use_bf16)

    model.load_state_dict(ema_state, strict=True)
    ema_val_ce = _eval_val_ce(model, val_samples, device, use_bf16)
    remdm_metrics_ema = eval_remdm_2step_repair(model, val_samples, device, use_bf16)

    packed_info = _pack_and_verify_hybrid_int4(
        model, arm_name, arm_dir / f"{arm_name}_hybrid_int4.bin", INT4_GROUP_SIZE
    )
    dq_model = packed_info.pop("dequant_model").to(device)
    int4_dequant_val_ce = _eval_val_ce(dq_model, val_samples, device, use_bf16)
    remdm_metrics_int4 = eval_remdm_2step_repair(dq_model, val_samples, device, use_bf16)

    steady_ms = step_ms_hist[10:] if len(step_ms_hist) > 10 else step_ms_hist
    result: Dict[str, Any] = {
        "arm": arm_name,
        "optimizer_mode": opt.mode,
        "steps": TRAIN_STEPS,
        "param_summary": psummary,
        "packed_int4_mb": packed_info["packed_int4_mb"],
        "packed_int4_verification": packed_info,
        "train_loss_tail50": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5),
        "raw_val_ce": raw_val_ce,
        "ema_val_ce": ema_val_ce,
        "int4_dequant_val_ce": int4_dequant_val_ce,
        "step_ms": round(sum(steady_ms) / max(1, len(steady_ms)), 2),
        "train_elapsed_sec": round(train_elapsed_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }
    if repair_loss_hist:
        result["repair_loss_tail50"] = round(sum(repair_loss_hist[-50:]) / max(1, len(repair_loss_hist[-50:])), 5)
    if remdm_metrics_ema:
        result["remdm_2step_ema"] = remdm_metrics_ema
    if remdm_metrics_int4:
        result["remdm_2step_int4"] = remdm_metrics_int4
    _atomic_write_json(cached, result)
    del model, dq_model, opt, ema_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _upload_json_to_hf(hf_token: str, local_path: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"{EXP_CODE}({EXP_ID}): upload {path_in_repo}",
    )
    print(f"[{EXP_ID}] Uploaded {path_in_repo} -> {HF_REPO_ID}", flush=True)


# ---------------------------------------------------------------------------
# CPU 2-Step Smoke Test
# ---------------------------------------------------------------------------

def run_smoke_test() -> int:
    """2-step tiny CPU forward+backward+step + 2-step ReMDM-conf repair + edge85m INT4 budget check."""
    _locate_mvd()
    import tempfile
    import torch
    import train_uni_v2
    import uni_omr

    device = torch.device("cpu")
    _seed_everything(SEED)

    tiny_results: Dict[str, Dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as tmpd:
        for arm_name in ARMS:
            model = build_arm_model(arm_name, grad_checkpoint=False, tiny=True).to(device)
            opt = train_uni_v2.build_optimizer_v2(
                model,
                optimizer_name="muon",
                lr=LR_MAX,
                weight_decay=WEIGHT_DECAY,
                device_type="cpu",
                allow_cpu_muon=True,
                no_decay_1d=True,
            )
            assert opt.mode == "muon+adamw", f"{arm_name}: expected muon+adamw, got {opt.mode}"
            model.train()

            losses_seen: List[float] = []
            smoke_val: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                imgs = torch.randn(1, 3, 64, 96, device=device)
                tgt = torch.randint(4, min(64, uni_omr.VOCAB_SIZE), (1, 18), device=device)
                tgt[:, 0] = uni_omr.BOS_ID
                smoke_val.append((imgs.detach().clone(), tgt.detach().clone()))

                mem_lvls, aux_enc, router_info = model.encode_image(imgs, return_aux=False)
                logits, mtp_logits, sys_y_pred = model.decode_tokens(
                    tgt[:, :-1], mem_lvls, return_mtp=True, return_aux=False
                )
                out = {
                    "logits": logits,
                    "mtp_logits": mtp_logits,
                    "router_info": router_info,
                    "sys_y_pred": sys_y_pred,
                    "sys_count_pred": aux_enc["sys_count_pred"],
                    "staff_seg_logits": aux_enc["staff_seg_logits"],
                    "stroke_hog_pred": aux_enc["stroke_hog_pred"],
                }
                loss = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)["loss"]
                if hasattr(model, "bd3_repair_head"):
                    rep_loss = compute_bd3_repair_loss(model, mem_lvls[-1], tgt[:, 1:])
                    assert torch.isfinite(rep_loss), f"non-finite repair loss {rep_loss}"
                    loss = loss + REPAIR_LOSS_WEIGHT * rep_loss
                assert torch.isfinite(loss), f"{arm_name} step {step}: non-finite loss {loss}"
                loss.backward()

                if arm_name == "bd3_remdm_head":
                    for pname, p in model.bd3_repair_head.named_parameters():
                        assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad in bd3_repair_head.{pname}"
                        assert p.grad.abs().sum().item() > 0.0, f"zero grad in bd3_repair_head.{pname}"
                opt.step()
                losses_seen.append(round(float(loss.item()), 4))

            pack_tiny = _pack_and_verify_hybrid_int4(
                model, arm_name, Path(tmpd) / f"{arm_name}_tiny.bin", INT4_GROUP_SIZE, tiny=True
            )
            dq_model = pack_tiny.pop("dequant_model")
            with torch.no_grad():
                mem, _, _ = dq_model.encode_image(imgs, return_aux=False)
                dq_logits, _, _ = dq_model.decode_tokens(
                    tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
                seq = dq_model.generate(imgs, max_len=4)
            assert dq_logits.shape == (1, 17, uni_omr.VOCAB_SIZE) and torch.isfinite(dq_logits).all()
            assert seq.shape[0] == 1 and seq.shape[1] >= 2

            remdm_res = eval_remdm_2step_repair(dq_model, smoke_val, device, use_bf16=False)
            tiny_results[arm_name] = {
                "opt_mode": opt.mode,
                "losses_2step": losses_seen,
                "tiny_params": sum(p.numel() for p in model.parameters()),
                "remdm_smoke": remdm_res,
            }

        edge_mb: Dict[str, float] = {}
        for arm_name in ARMS:
            m_edge = build_arm_model(arm_name, grad_checkpoint=False, tiny=False)
            pack_edge = _pack_and_verify_hybrid_int4(
                m_edge, arm_name, Path(tmpd) / f"{arm_name}_edge85m.bin", INT4_GROUP_SIZE, tiny=False
            )
            pack_edge.pop("dequant_model")
            assert pack_edge["passes_55mb_budget"], f"{arm_name} exceeded 55 MB budget: {pack_edge['packed_int4_mb']}"
            edge_mb[arm_name] = pack_edge["packed_int4_mb"]
            del m_edge

    delta_mb = round(edge_mb["bd3_remdm_head"] - edge_mb["ctrl"], 3)
    assert delta_mb < REPAIR_HEAD_MAX_DELTA_MB, (
        f"Repair head INT4 delta ({delta_mb} MB) exceeded {REPAIR_HEAD_MAX_DELTA_MB} MB budget!"
    )
    print(f"[{EXP_CODE}][SMOKE] tiny ctrl:           {tiny_results['ctrl']}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] tiny bd3_remdm_head: {tiny_results['bd3_remdm_head']}", flush=True)
    print(
        f"[{EXP_CODE}][SMOKE] edge85m packed_int4_mb: ctrl={edge_mb['ctrl']} MB, "
        f"bd3_remdm_head={edge_mb['bd3_remdm_head']} MB (delta={delta_mb:+.3f} MB < {REPAIR_HEAD_MAX_DELTA_MB} MB)",
        flush=True,
    )
    print(f"[{EXP_CODE}][SMOKE] PASS!", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{EXP_ID} runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-step tiny CPU smoke test and exit")
    args = parser.parse_args()
    if args.smoke_test:
        return run_smoke_test()

    t_start = time.perf_counter()
    started_at = _utc_now()
    hf_token: Optional[str] = None
    status = "RUNNING"
    error_msg: Optional[str] = None
    arm_results: Dict[str, Dict[str, Any]] = {}

    try:
        hf_token = _read_hf_token()
        _ensure_deps()
        _locate_mvd()
        _configure_backend_safety()

        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        print(
            f"[{EXP_ID}] Device={gpu_name} | bf16={use_bf16} | cudnn={torch.backends.cudnn.enabled} | torch={torch.__version__}",
            flush=True,
        )

        arm_results.update(_try_resume_from_hf(hf_token))

        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        val_samples = _preload_val_samples(val_pairs)
        print(f"[{EXP_ID}] Loaded {len(train_pairs)} train / {len(val_samples)} val", flush=True)

        for arm_name in ARMS:
            res = _train_arm(
                arm_name,
                train_pairs=train_pairs,
                val_samples=val_samples,
                device=device,
                use_bf16=use_bf16,
            )
            arm_results[arm_name] = res

            arm_json_local = WORK_ROOT / "ckpts" / arm_name / "arm_result.json"
            _upload_json_to_hf(hf_token, arm_json_local, f"{HF_CKPT_PREFIX}/{arm_name}_arm_result.json")

            eval_payload: Dict[str, Any] = {
                "exp_id": EXP_ID,
                "exp_code": EXP_CODE,
                "status": "RUNNING" if len(arm_results) < len(ARMS) else "COMPLETED",
                "started_at": started_at,
                "updated_at": _utc_now(),
                "gpu": gpu_name,
                "bf16": use_bf16,
                "steps_per_arm": TRAIN_STEPS,
                "seed": SEED,
                "arms": arm_results,
            }
            if "ctrl" in arm_results and "bd3_remdm_head" in arm_results:
                c = arm_results["ctrl"]
                t = arm_results["bd3_remdm_head"]
                eval_payload["delta_bd3_remdm_minus_ctrl"] = {
                    "ema_val_ce": round(t["ema_val_ce"] - c["ema_val_ce"], 5),
                    "int4_dequant_val_ce": round(t["int4_dequant_val_ce"] - c["int4_dequant_val_ce"], 5),
                    "raw_val_ce": round(t["raw_val_ce"] - c["raw_val_ce"], 5),
                    "step_ms": round(t["step_ms"] - c["step_ms"], 2),
                    "packed_int4_mb": round(t["packed_int4_mb"] - c["packed_int4_mb"], 3),
                }
            eval_local = WORK_ROOT / "out" / "eval.json"
            _atomic_write_json(eval_local, eval_payload)
            _upload_json_to_hf(hf_token, eval_local, HF_EVAL_PATH)

        status = "COMPLETED"
    except Exception as exc:
        status = "FAILED"
        error_msg = f"{type(exc).__name__}: {exc}"
        print(f"[{EXP_ID}] ERROR: {error_msg}", flush=True)
        raise
    finally:
        total_sec = time.perf_counter() - t_start
        run_payload = {
            "exp_id": EXP_ID,
            "exp_code": EXP_CODE,
            "status": status,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_sec": round(total_sec, 2),
            "n_arms_completed": len(arm_results),
            "eval_path": HF_EVAL_PATH,
            "error": error_msg,
        }
        run_local = WORK_ROOT / "out" / "run.json"
        _atomic_write_json(run_local, run_payload)
        if hf_token:
            try:
                _upload_json_to_hf(hf_token, run_local, HF_RUN_PATH)
            except Exception as up_exc:
                print(f"[{EXP_ID}] WARN: failed to upload run ledger: {up_exc}", flush=True)
        _purge_hf_token()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
