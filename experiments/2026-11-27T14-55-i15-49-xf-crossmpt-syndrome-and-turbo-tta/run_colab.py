#!/usr/bin/env python3
"""Colab A100 / Local RTX 4080 SUPER runner — Iter 15 Exp i15-49: CrossMPT Syndrome Refiner + Turbo TTA.

Experiment ID: 2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta
Origin: Channel Coding (CrossMPT ICLR 2025 syndrome-masked cross-attention + Zero-Label Turbo TTA).
Controlled 2-arm proxy on edge85m + ax0_gl6 (600 steps/arm, BF16, seed=42, verified muon+adamw):
  - Arm 0 (ctrl):           Standard edge85m + ax0_gl6.
  - Arm 1 (crossmpt_turbo): Tiny 2-layer CrossMPT post-refiner (< 1.5 MB INT4) performing bar-masked
                            cross-attention between token logit magnitudes and per-bar conservation
                            syndrome embeddings, PLUS 3-step zero-label syndrome-minimizing
                            encoder-norm Turbo TTA evaluated on degraded validation pages.

Includes 1-arm HF partial checkpoint sync + startup auto-resume and Windows cuDNN 9.10 safety.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXP_ID = "2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta"
EXP_CODE = "i15-49"
ARMS = ("ctrl", "crossmpt_turbo")
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_CKPT_PREFIX = f"checkpoints/{EXP_ID}"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORK_ROOT = Path("/vol/i15_49") if Path("/vol").is_dir() else Path("/content/i15_49")

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
REFINER_MAX_DELTA_MB = 1.5
SAVE_EVERY_MINUTES = 5.0
TURBO_TTA_STEPS = 3
TURBO_TTA_LR = 1e-3
TURBO_TTA_MAX_PAGES = 16


# ---------------------------------------------------------------------------
# Self-Contained Experimental Module: CrossMPTRefiner (ICLR 2025)
# ---------------------------------------------------------------------------

def _build_crossmpt_refiner_class() -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import uni_omr
    import uni_omr_v2

    # Precompute duration weights per vocab ID (in quarter beats) & barline IDs
    dur_map = {
        "breve": 8.0, "whole": 4.0, "half": 2.0, "quarter": 1.0,
        "eighth": 0.5, "16th": 0.25, "32nd": 0.125, "64th": 0.0625, "128th": 0.03125,
    }
    bar_toks = {"bar", "bar-light-heavy", "bar-repeat-right", "bar-double", "<sys>"}

    class CrossMPTLayer(nn.Module):
        """One CrossMPT layer: bar-masked cross-attention from token logit features to bar syndromes."""

        def __init__(self, d_ref: int, n_heads: int = 4) -> None:
            super().__init__()
            self.n_heads = int(n_heads)
            self.head_dim = d_ref // n_heads
            self.q_norm = uni_omr_v2.RMSNorm(d_ref)
            self.kv_norm = uni_omr_v2.RMSNorm(d_ref)
            self.q_proj = nn.Linear(d_ref, d_ref, bias=False)
            self.k_proj = nn.Linear(d_ref, d_ref, bias=False)
            self.v_proj = nn.Linear(d_ref, d_ref, bias=False)
            self.o_proj = nn.Linear(d_ref, d_ref, bias=False)
            self.ffn_norm = uni_omr_v2.RMSNorm(d_ref)
            self.ffn_up = nn.Linear(d_ref, d_ref * 2, bias=False)
            self.ffn_down = nn.Linear(d_ref * 2, d_ref, bias=False)

        def forward(
            self,
            x: torch.Tensor,
            syn_kv: torch.Tensor,
            bar_mask: torch.Tensor,
        ) -> torch.Tensor:
            """x: (B, T, d_ref), syn_kv: (B, N_bars, d_ref), bar_mask: (B, 1, T, N_bars) bool."""
            B, T, D = x.shape
            N = syn_kv.shape[1]
            q = self.q_proj(self.q_norm(x)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            kv_in = self.kv_norm(syn_kv)
            k = self.k_proj(kv_in).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(kv_in).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

            attn_bias = torch.zeros((B, 1, T, N), device=x.device, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(~bar_mask, -1e4)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=False)
            out = out.transpose(1, 2).reshape(B, T, D)
            x = x + self.o_proj(out)
            x = x + self.ffn_down(F.silu(self.ffn_up(self.ffn_norm(x))))
            return x

    class CrossMPTRefiner(nn.Module):
        """Tiny 2-layer CrossMPT post-refiner (< 1.5 MB INT4) conditioned on per-bar conservation syndromes."""

        def __init__(self, vocab_size: int, d_ref: int = 128, n_heads: int = 4, n_layers: int = 2) -> None:
            super().__init__()
            self.vocab_size = int(vocab_size)
            self.d_ref = int(d_ref)

            dur_vec = torch.zeros(self.vocab_size, dtype=torch.float32)
            bar_mask_vec = torch.zeros(self.vocab_size, dtype=torch.bool)
            for tok, tid in uni_omr.UNI_TOKEN_TO_ID.items():
                if tid < self.vocab_size:
                    if tok in dur_map:
                        dur_vec[tid] = dur_map[tok]
                    if tok in bar_toks:
                        bar_mask_vec[tid] = True
            self.register_buffer("dur_vec", dur_vec, persistent=False)
            self.register_buffer("bar_id_mask", bar_mask_vec, persistent=False)

            # Input: token logit magnitudes |z| (V -> d_ref) + syndrome features (4 -> d_ref)
            self.mag_proj = nn.Linear(self.vocab_size, d_ref, bias=False)
            self.syn_proj = nn.Linear(4, d_ref, bias=True)
            self.layers = nn.ModuleList([CrossMPTLayer(d_ref, n_heads=n_heads) for _ in range(n_layers)])
            self.out_norm = uni_omr_v2.RMSNorm(d_ref)
            self.out_proj = nn.Linear(d_ref, self.vocab_size, bias=False)
            # Zero-init residual gate so initial behavior matches base decoder cleanly
            self.res_gate = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        def compute_bar_syndromes(
            self,
            logits: torch.Tensor,
            tgt_ids: Optional[torch.Tensor] = None,
            bar_capacity: float = 4.0,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Compute differentiable per-bar duration conservation syndromes.

            Returns:
                syn_feats: (B, N_bars, 4)
                bar_mask:  (B, 1, T, N_bars) bool (each token attends to its own bar + neighbor)
                syn_loss:  scalar differentiable conservation loss
            """
            B, T, V = logits.shape
            probs = F.softmax(logits.float(), dim=-1)  # (B, T, V)
            exp_dur = torch.matmul(probs, self.dur_vec.to(probs.device))  # (B, T)
            entropy = -(probs * torch.log(probs.clamp(min=1e-7))).sum(dim=-1)  # (B, T)

            if tgt_ids is not None:
                is_bar = self.bar_id_mask.to(tgt_ids.device)[tgt_ids.clamp(0, V - 1)]
            else:
                pred_ids = logits.argmax(dim=-1)
                is_bar = self.bar_id_mask.to(pred_ids.device)[pred_ids.clamp(0, V - 1)]

            # Bar index per token via cumulative sum of bar delimiters (capped to max 32 bars/seq)
            bar_idx = torch.cumsum(is_bar.to(torch.long), dim=1)
            # Shift so delimiter belongs to the bar it closes
            bar_idx = torch.cat([torch.zeros((B, 1), dtype=torch.long, device=logits.device), bar_idx[:, :-1]], dim=1)
            max_bars = int(min(32, max(1, int(bar_idx.max().item()) + 1)))
            bar_idx = bar_idx.clamp(0, max_bars - 1)

            one_hot = F.one_hot(bar_idx, num_classes=max_bars).float()  # (B, T, N)
            bar_counts = one_hot.sum(dim=1).clamp(min=1.0)  # (B, N)
            bar_dur = torch.einsum("bt,btn->bn", exp_dur, one_hot)  # (B, N)
            bar_ent = torch.einsum("bt,btn->bn", entropy, one_hot) / bar_counts  # (B, N)

            # Active bars (those containing > 0.25 expected beats)
            active = (bar_dur > 0.25).float()
            signed_err = (bar_dur - bar_capacity) / bar_capacity
            abs_err = signed_err.abs()
            ratio = (bar_dur / bar_capacity).clamp(0.0, 4.0)
            syn_feats = torch.stack([signed_err, abs_err, ratio, bar_ent], dim=-1)  # (B, N, 4)

            # Bar-masked attention: each token attends to its own bar (and immediate previous bar)
            idx_grid = torch.arange(max_bars, device=logits.device).view(1, 1, max_bars)
            dist = (bar_idx.unsqueeze(-1) - idx_grid).abs()
            bar_mask = (dist <= 1).unsqueeze(1)  # (B, 1, T, N)

            # Differentiable conservation syndrome loss
            syn_loss = (abs_err * active).sum() / active.sum().clamp(min=1.0)
            return syn_feats.to(logits.dtype), bar_mask, syn_loss

        def forward(
            self,
            logits: torch.Tensor,
            tgt_ids: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            syn_feats, bar_mask, syn_loss = self.compute_bar_syndromes(logits, tgt_ids=tgt_ids)
            # CrossMPT uses logit magnitudes (|z|) as reliability inputs
            x = self.mag_proj(logits.abs())
            syn_kv = self.syn_proj(syn_feats)
            for layer in self.layers:
                x = layer(x, syn_kv, bar_mask)
            delta = self.out_proj(self.out_norm(x))
            refined = logits + torch.tanh(self.res_gate).to(logits.dtype) * delta
            return refined, syn_loss

    return CrossMPTRefiner


def apply_crossmpt_refiner(model: Any, tiny: bool = False) -> Any:
    """Attach CrossMPTRefiner onto UniOMRv2Net and wrap decode_tokens cleanly."""
    CrossMPTRefiner = _build_crossmpt_refiner_class()
    d_ref = 32 if tiny else 128
    n_heads = 2 if tiny else 4
    model.crossmpt_refiner = CrossMPTRefiner(
        vocab_size=int(model.cfg.vocab_size),
        d_ref=d_ref,
        n_heads=n_heads,
        n_layers=2,
    )
    orig_decode = model.decode_tokens

    def _decode_with_crossmpt(
        self: Any,
        tgt_ids: Any,
        memory: Any,
        return_mtp: Optional[bool] = None,
        return_aux: Optional[bool] = None,
        past_kvs: Optional[List[Any]] = None,
        use_cache: bool = False,
        apply_tok_dropout: bool = True,
    ) -> Any:
        out = orig_decode(
            tgt_ids,
            memory,
            return_mtp=return_mtp,
            return_aux=return_aux,
            past_kvs=past_kvs,
            use_cache=use_cache,
            apply_tok_dropout=apply_tok_dropout,
        )
        logits, mtp_logits, third = out
        # Apply CrossMPT refiner on full-sequence decode (skip single-token cached steps)
        if not use_cache and logits.shape[1] > 1:
            refined_logits, syn_loss = self.crossmpt_refiner(logits, tgt_ids=tgt_ids)
            self._last_syn_loss = syn_loss
            return refined_logits, mtp_logits, third
        return logits, mtp_logits, third

    model.decode_tokens = types.MethodType(_decode_with_crossmpt, model)
    return model


def build_arm_model(arm_name: str, grad_checkpoint: bool = True, tiny: bool = False) -> Any:
    import uni_omr_v2

    if tiny:
        cfg = uni_omr_v2.UniOMRv2Config(
            d_model=64, n_heads=4, n_kv_heads=2,
            n_axial_layers=0, n_global_layers=2, n_dec_layers=2,
            ffn_mult=2.0, axial_ffn_mult=2.0,
            backbone_channels=(16, 32, 48, 64), backbone_blocks=(1, 1, 1, 1),
            backbone_expand=2.0, router_hidden=32, max_mem_tokens=256,
            mtp_k=4, grad_checkpoint=grad_checkpoint,
        )
    else:
        cfg = uni_omr_v2.UniOMRv2Config.edge85m(
            n_axial_layers=0, n_global_layers=6, grad_checkpoint=grad_checkpoint,
        )
    model = uni_omr_v2.UniOMRv2Net(cfg)
    if arm_name == "crossmpt_turbo":
        apply_crossmpt_refiner(model, tiny=tiny)
    elif arm_name != "ctrl":
        raise ValueError(f"Unknown arm_name: {arm_name}")
    return model


# ---------------------------------------------------------------------------
# Zero-Label Syndrome-Driven Encoder-Norm Turbo TTA
# ---------------------------------------------------------------------------

def _collect_encoder_norm_params(model: Any) -> List[Tuple[str, "torch.nn.Parameter"]]:
    import torch.nn as nn
    import uni_omr_v2

    norm_types = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, uni_omr_v2.RMSNorm)
    params: List[Tuple[str, Any]] = []
    for root_name in ("backbone", "encoder"):
        root = getattr(model, root_name, None)
        if root is None:
            continue
        for mname, mod in root.named_modules():
            if isinstance(mod, norm_types):
                for pname, p in mod.named_parameters(recurse=False):
                    params.append((f"{root_name}.{mname}.{pname}", p))
    return params


def run_turbo_tta_page(
    model: Any,
    inp: "torch.Tensor",
    tgt: "torch.Tensor",
    use_bf16: bool,
    n_steps: int = TURBO_TTA_STEPS,
    lr: float = TURBO_TTA_LR,
) -> Dict[str, float]:
    """Run 3-step zero-label syndrome + entropy minimizing TTA on encoder norm affine params.

    Never uses target labels during TTA updates (uses greedy prefix / base logits only).
    Restores exact original norm weights before returning.
    """
    import torch
    import torch.nn.functional as F
    import uni_omr

    norm_params = _collect_encoder_norm_params(model)
    saved_state = [(p, p.detach().clone()) for _, p in norm_params]
    saved_req = [(p, p.requires_grad) for p in model.parameters()]

    def _eval_ce_and_syn() -> Tuple[float, float]:
        with torch.no_grad(), torch.amp.autocast(inp.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            mem, _, _ = model.encode_image(inp, return_aux=False)
            logits, _, _ = model.decode_tokens(
                tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
            )
            ce = float(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID
            ).item())
            syn = float(getattr(model, "_last_syn_loss", torch.tensor(0.0)).item())
        return ce, syn

    model.eval()
    ce_before, syn_before = _eval_ce_and_syn()

    try:
        for p in model.parameters():
            p.requires_grad_(False)
        tta_params = [p for _, p in norm_params]
        for p in tta_params:
            p.requires_grad_(True)
        tta_opt = torch.optim.Adam(tta_params, lr=lr)

        # Obtain zero-label pseudo-token prefix from initial model pass (no labels used!)
        with torch.no_grad(), torch.amp.autocast(inp.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            pseudo_len = min(32, int(tgt.shape[1]) - 1)
            pseudo_prefix = model.generate(inp, max_len=max(4, pseudo_len))

        for _ in range(n_steps):
            tta_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(inp.device.type, dtype=torch.bfloat16, enabled=use_bf16):
                mem, _, _ = model.encode_image(inp, return_aux=False)
                logits, _, _ = model.decode_tokens(
                    pseudo_prefix[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
                probs = F.softmax(logits.float(), dim=-1)
                ent = -(probs * torch.log(probs.clamp(min=1e-7))).sum(dim=-1).mean()
                syn_loss = getattr(model, "_last_syn_loss", torch.tensor(0.0, device=inp.device))
                tta_loss = syn_loss + 0.15 * ent
            tta_loss.backward()
            torch.nn.utils.clip_grad_norm_(tta_params, 1.0)
            tta_opt.step()

        ce_after, syn_after = _eval_ce_and_syn()
    finally:
        with torch.no_grad():
            for p, orig in saved_state:
                p.copy_(orig)
        for p, req in saved_req:
            p.requires_grad_(req)

    return {
        "ce_before": ce_before,
        "ce_after": ce_after,
        "syn_before": syn_before,
        "syn_after": syn_after,
    }


def _degrade_pil_for_tta(img_pil: Any) -> Any:
    """Deterministic mild phone-photo degradation (blur + low contrast) for TTA val evaluation."""
    from PIL import ImageEnhance, ImageFilter

    out = img_pil.convert("RGB").filter(ImageFilter.GaussianBlur(radius=0.8))
    out = ImageEnhance.Contrast(out).enhance(0.75)
    return out


# ---------------------------------------------------------------------------
# Infrastructure Helpers (Windows Safety, HF Sync/Resume, Dataset, INT4)
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
            repo_id=HF_REPO_ID, filename=HF_EVAL_PATH, repo_type="dataset",
            token=hf_token, local_dir=str(dl_dir),
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
                repo_id=HF_REPO_ID, filename=f"{HF_CKPT_PREFIX}/{arm_name}_arm_result.json",
                repo_type="dataset", token=hf_token, local_dir=str(dl_dir),
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
    tar_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=HF_PROXY_TAR, repo_type="dataset",
        token=hf_token, local_dir=str(dl_dir),
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
            pairs.append((img_p, beks, domain, 2.0 if is_dense_or_hw else 1.0))
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

    int4_roots = [
        infer_model.encoder.axial_layers,
        infer_model.encoder.patch_merger,
        infer_model.encoder.global_layers,
        infer_model.deepstack_fusions,
        infer_model.dec_layers,
        infer_model.lm_head,
    ]
    if hasattr(infer_model, "crossmpt_refiner"):
        int4_roots.append(infer_model.crossmpt_refiner)

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
                dq = (q.float() * scale.float()).reshape(-1)[:orig_n].reshape(t.shape)
                dequant_sd[name] = dq.to(param.dtype)
            else:
                fp16_np = t.to(torch.float16).numpy()
                f.write(fp16_np.tobytes())
                total_fp16_bytes += int(fp16_np.nbytes)
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
        "dequant_model": infer_model,
    }


def _eval_val_ce(
    model: Any,
    val_pairs: List[Tuple[Path, List[Path], str, float]],
    device: "torch.device",
    use_bf16: bool,
) -> float:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr

    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for img_p, beks, _, _ in val_pairs:
            ids = _encode_bekerns_full(beks)
            if len(ids) < 3:
                continue
            img = Image.open(img_p).convert("RGB")
            inp = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
            tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
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


def _eval_degraded_turbo_tta(
    model: Any,
    val_pairs: List[Tuple[Path, List[Path], str, float]],
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, float]:
    import torch
    from PIL import Image

    ce_before_list: List[float] = []
    ce_after_list: List[float] = []
    syn_before_list: List[float] = []
    syn_after_list: List[float] = []
    for img_p, beks, _, _ in val_pairs[:TURBO_TTA_MAX_PAGES]:
        ids = _encode_bekerns_full(beks)
        if len(ids) < 6:
            continue
        img = _degrade_pil_for_tta(Image.open(img_p))
        inp = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
        tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
        r = run_turbo_tta_page(model, inp, tgt, use_bf16=use_bf16)
        ce_before_list.append(r["ce_before"])
        ce_after_list.append(r["ce_after"])
        syn_before_list.append(r["syn_before"])
        syn_after_list.append(r["syn_after"])
    n = max(1, len(ce_before_list))
    return {
        "n_pages": len(ce_before_list),
        "degraded_val_ce_before_tta": round(sum(ce_before_list) / n, 5),
        "degraded_val_ce_after_tta": round(sum(ce_after_list) / n, 5),
        "delta_tta_ce": round((sum(ce_after_list) - sum(ce_before_list)) / n, 5),
        "syndrome_before_tta": round(sum(syn_before_list) / n, 5),
        "syndrome_after_tta": round(sum(syn_after_list) / n, 5),
    }


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
        imgs.append(_pil_to_tensor(Image.open(img_p), FULLPAGE_MAX_DIM))
        seqs.append(ids[:MAX_SEQ_LEN])
    if not imgs:
        return None
    max_h = max(t.shape[1] for t in imgs)
    max_w = max(t.shape[2] for t in imgs)
    padded_imgs = [F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1]), value=1.0) for t in imgs]
    max_t = max(len(s) for s in seqs)
    tgt = torch.full((len(seqs), max_t), uni_omr.PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        tgt[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return torch.stack(padded_imgs, dim=0).to(device), tgt.to(device)


def _train_arm(
    arm_name: str,
    *,
    train_pairs: List[Tuple[Path, List[Path], str, float]],
    val_pairs: List[Tuple[Path, List[Path], str, float]],
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    import train_uni_v2
    import uni_omr_v2

    _seed_everything(SEED)
    arm_dir = WORK_ROOT / "ckpts" / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached = arm_dir / "arm_result.json"
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))

    model = build_arm_model(arm_name, grad_checkpoint=True).to(device)
    psummary = model.param_summary(int4_block_size=INT4_GROUP_SIZE)
    qat = uni_omr_v2.QATInt4ControllerV2(model, block_size=INT4_GROUP_SIZE)
    opt = train_uni_v2.build_optimizer_v2(
        model, optimizer_name="muon", lr=LR_MAX, weight_decay=WEIGHT_DECAY,
        device_type=device.type, no_decay_1d=True,
    )
    if device.type == "cuda" and opt.mode != "muon+adamw":
        raise RuntimeError(f"[{EXP_ID}][{arm_name}] Silent fallback detected: expected muon+adamw, got {opt.mode}")
    print(f"[{EXP_ID}][{arm_name}] Optimizer verified: mode={opt.mode}", flush=True)

    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    weights = [p[3] for p in train_pairs]
    rng = random.Random(SEED)
    start_step = 1
    loss_hist: List[float] = []
    step_ms_hist: List[float] = []

    latest_pt = arm_dir / "latest.pt"
    if latest_pt.is_file():
        ck = torch.load(str(latest_pt), map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        ema_state = {k: v.to(device) for k, v in ck["ema_state_dict"].items()}
        if "opt_state_dicts" in ck and len(ck["opt_state_dicts"]) == len(opt.optimizers):
            for o_obj, sd in zip(opt.optimizers, ck["opt_state_dicts"]):
                o_obj.load_state_dict(sd)
        if "rng_state" in ck:
            rng.setstate(ck["rng_state"])
        start_step = int(ck["step"]) + 1
        loss_hist = list(ck.get("loss_hist", []))
        step_ms_hist = list(ck.get("step_ms_hist", []))
        if start_step > QAT_START_STEP:
            qat.enable()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_arm0 = time.perf_counter()
    t_last_save = time.time()

    model.train()
    for step in range(start_step, TRAIN_STEPS + 1):
        if step == QAT_START_STEP:
            qat.enable()
        opt.set_lr(_wsd_mult(step, TRAIN_STEPS, WARMUP_STEPS))
        t_step0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            collated = _collate_microbatch(rng.choices(train_pairs, weights=weights, k=BATCH_SIZE), device)
            if collated is None:
                continue
            inp, tgt = collated
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(inp, tgt[:, :-1], return_aux=False, return_mtp=True)
                losses = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)
                total_loss = losses["loss"]
                if arm_name == "crossmpt_turbo" and hasattr(model, "_last_syn_loss"):
                    total_loss = total_loss + 0.05 * model._last_syn_loss
                loss = total_loss / GRAD_ACCUM
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

        if step % 100 == 0 or step == 1:
            print(
                f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} "
                f"loss={sum(loss_hist[-50:])/max(1,len(loss_hist[-50:])):.4f} "
                f"step_ms={sum(step_ms_hist[-50:])/max(1,len(step_ms_hist[-50:])):.1f}",
                flush=True,
            )
        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ck = arm_dir / "latest.pt.tmp"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema_state,
                    "opt_state_dicts": [o.state_dict() for o in opt.optimizers],
                    "rng_state": rng.getstate(),
                    "loss_hist": loss_hist,
                    "step_ms_hist": step_ms_hist,
                },
                tmp_ck,
            )
            os.replace(tmp_ck, latest_pt)
            t_last_save = time.time()

    train_elapsed_sec = time.perf_counter() - t_arm0
    peak_vram_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0) if device.type == "cuda" else 0.0
    qat.disable()
    raw_val_ce = _eval_val_ce(model, val_pairs, device, use_bf16)
    model.load_state_dict(ema_state, strict=True)
    ema_val_ce = _eval_val_ce(model, val_pairs, device, use_bf16)

    turbo_tta_metrics = None
    if arm_name == "crossmpt_turbo":
        turbo_tta_metrics = _eval_degraded_turbo_tta(model, val_pairs, device, use_bf16)

    packed_info = _pack_and_verify_hybrid_int4(model, arm_name, arm_dir / f"{arm_name}_hybrid_int4.bin", INT4_GROUP_SIZE)
    dq_model = packed_info.pop("dequant_model").to(device)
    int4_dequant_val_ce = _eval_val_ce(dq_model, val_pairs, device, use_bf16)

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
    if turbo_tta_metrics is not None:
        result["turbo_tta_degraded_val"] = turbo_tta_metrics
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
        path_or_fileobj=str(local_path), path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID, repo_type="dataset",
        commit_message=f"{EXP_CODE}({EXP_ID}): upload {path_in_repo}",
    )
    print(f"[{EXP_ID}] Uploaded {path_in_repo} -> {HF_REPO_ID}", flush=True)


# ---------------------------------------------------------------------------
# CPU 2-Step Smoke Test
# ---------------------------------------------------------------------------

def run_smoke_test() -> int:
    """2-step tiny CPU forward+backward+step + Turbo TTA + edge85m packed_int4_mb verification."""
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
                model, optimizer_name="muon", lr=LR_MAX, weight_decay=WEIGHT_DECAY,
                device_type="cpu", allow_cpu_muon=True, no_decay_1d=True,
            )
            assert opt.mode == "muon+adamw", f"{arm_name}: expected muon+adamw, got {opt.mode}"
            model.train()

            losses_seen: List[float] = []
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                imgs = torch.randn(1, 3, 64, 96, device=device)
                tgt = torch.randint(4, min(64, uni_omr.VOCAB_SIZE), (1, 14), device=device)
                tgt[:, 0] = uni_omr.BOS_ID
                out = model(imgs, tgt[:, :-1], return_aux=False, return_mtp=True)
                loss = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)["loss"]
                if arm_name == "crossmpt_turbo":
                    loss = loss + 0.05 * model._last_syn_loss
                assert torch.isfinite(loss), f"{arm_name} step {step}: non-finite loss"
                loss.backward()
                if arm_name == "crossmpt_turbo":
                    rg = model.crossmpt_refiner.res_gate.grad
                    assert rg is not None and torch.isfinite(rg).all()
                    assert model.crossmpt_refiner.out_proj.weight.grad is not None
                opt.step()
                losses_seen.append(round(float(loss.item()), 4))

            if arm_name == "crossmpt_turbo":
                tta_res = run_turbo_tta_page(model, imgs, tgt, use_bf16=False, n_steps=3, lr=1e-3)
                assert math.isfinite(tta_res["ce_after"]) and math.isfinite(tta_res["syn_after"])

            pack_tiny = _pack_and_verify_hybrid_int4(
                model, arm_name, Path(tmpd) / f"{arm_name}_tiny.bin", INT4_GROUP_SIZE, tiny=True
            )
            dq_model = pack_tiny.pop("dequant_model")
            with torch.no_grad():
                seq = dq_model.generate(imgs, max_len=4)
            assert seq.shape[0] == 1 and seq.shape[1] >= 2
            tiny_results[arm_name] = {"opt_mode": opt.mode, "losses_2step": losses_seen}

        # Exact edge85m packed_int4_mb verification
        edge_mb: Dict[str, float] = {}
        for arm_name in ARMS:
            m_edge = build_arm_model(arm_name, grad_checkpoint=False, tiny=False)
            pack_edge = _pack_and_verify_hybrid_int4(
                m_edge, arm_name, Path(tmpd) / f"{arm_name}_edge85m.bin", INT4_GROUP_SIZE, tiny=False
            )
            pack_edge.pop("dequant_model")
            assert pack_edge["passes_55mb_budget"], f"{arm_name} exceeded 55 MB: {pack_edge['packed_int4_mb']}"
            edge_mb[arm_name] = pack_edge["packed_int4_mb"]
            del m_edge

    delta_mb = round(edge_mb["crossmpt_turbo"] - edge_mb["ctrl"], 3)
    assert delta_mb < REFINER_MAX_DELTA_MB, f"CrossMPT refiner delta {delta_mb} MB >= {REFINER_MAX_DELTA_MB} MB"
    print(f"[{EXP_CODE}][SMOKE] tiny ctrl:           {tiny_results['ctrl']}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] tiny crossmpt_turbo: {tiny_results['crossmpt_turbo']}", flush=True)
    print(
        f"[{EXP_CODE}][SMOKE] edge85m packed_int4_mb: ctrl={edge_mb['ctrl']} MB, "
        f"crossmpt_turbo={edge_mb['crossmpt_turbo']} MB (delta={delta_mb:+.3f} MB < {REFINER_MAX_DELTA_MB} MB)",
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
        print(f"[{EXP_ID}] Device={gpu_name} | bf16={use_bf16} | cudnn={torch.backends.cudnn.enabled}", flush=True)

        arm_results.update(_try_resume_from_hf(hf_token))
        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)

        for arm_name in ARMS:
            res = _train_arm(
                arm_name, train_pairs=train_pairs, val_pairs=val_pairs, device=device, use_bf16=use_bf16
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
            if "ctrl" in arm_results and "crossmpt_turbo" in arm_results:
                c, t = arm_results["ctrl"], arm_results["crossmpt_turbo"]
                eval_payload["delta_crossmpt_minus_ctrl"] = {
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
