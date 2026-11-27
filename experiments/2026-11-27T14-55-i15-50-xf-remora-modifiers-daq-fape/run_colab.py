#!/usr/bin/env python3
"""Colab A100 / Local RTX 4080 SUPER runner — Iter 15 Exp i15-50: Remora Modifiers + 2D-FAPE + DAQ Critic.

Experiment ID: 2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape
Origin: Nanopore Genomics (Remora Move-Table decoupled modifier head) + Cryo-EM (ModelAngelo 2D-FAPE
        local-frame offset loss + DAQ independent local raster-vs-symbol register-shift critic).
Controlled 2-arm weight-incompatible proxy on edge85m + ax0_gl6 (600 steps/arm, BF16, seed=42):
  - Arm 0 (ctrl):            Standard edge85m + ax0_gl6.
  - Arm 1 (remora_daq_fape): Adds (a) RemoraModifierHead (~0.3 MB INT4) local cross-attention head
                             focused on rare modifier/accidental/dot/articulation tokens, (b) training-only
                             FAPE2DOffsetHead (0.00 MB inference, stripped at export) for local-frame
                             relative pitch-step/octave offsets, and (c) DAQRegisterCriticHead (~0.1 MB INT4)
                             trained with synthetic register-shifted negative bars to flag silent shifts.

Includes 1-arm HF partial checkpoint sync + startup auto-resume and Windows cuDNN 9.10 safety.
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
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXP_ID = "2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape"
EXP_CODE = "i15-50"
ARMS = ("ctrl", "remora_daq_fape")
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_CKPT_PREFIX = f"checkpoints/{EXP_ID}"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORK_ROOT = Path("/vol/i15_50") if Path("/vol").is_dir() else Path("/content/i15_50")

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
SAVE_EVERY_MINUTES = 5.0


# ---------------------------------------------------------------------------
# Vocab Token Class Helpers (Modifiers, Pitch Steps, Octaves)
# ---------------------------------------------------------------------------

def _get_modifier_mask(vocab_size: int) -> "torch.Tensor":
    import torch
    import uni_omr

    exact_mods = {
        "a-2", "a-1", "a1", "a2", "dot", "tie-start", "tie-stop",
        "slur-start", "slur-stop", "stacc", "accent", "tenuto", "fermata",
        "grace", "cue", "xstaff-up", "xstaff-down",
    }
    prefixes = ("art-", "dyn-", "hairpin-", "ottava-")
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for tok, tid in uni_omr.UNI_TOKEN_TO_ID.items():
        if tid < vocab_size and (tok in exact_mods or tok.startswith(prefixes)):
            mask[tid] = True
    return mask


def _get_step_and_oct_maps(vocab_size: int) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Return lookup tables (step_val[V], oct_val[V], shift_map[V]) for 2D-FAPE & DAQ negatives."""
    import torch
    import uni_omr

    step_order = ["C", "D", "E", "F", "G", "A", "B"]
    oct_order = [str(n) for n in range(0, 9)]
    step_val = torch.full((vocab_size,), -1.0, dtype=torch.float32)
    oct_val = torch.full((vocab_size,), -1.0, dtype=torch.float32)
    shift_map = torch.arange(vocab_size, dtype=torch.long)

    tok_to_id = uni_omr.UNI_TOKEN_TO_ID
    for idx, s in enumerate(step_order):
        tid = tok_to_id.get(s)
        if tid is not None and tid < vocab_size:
            step_val[tid] = float(idx)
            # Synthetic register shift: +1 diatonic step
            next_tid = tok_to_id.get(step_order[(idx + 1) % len(step_order)])
            if next_tid is not None and next_tid < vocab_size:
                shift_map[tid] = next_tid
    for idx, o in enumerate(oct_order):
        tid = tok_to_id.get(o)
        if tid is not None and tid < vocab_size:
            oct_val[tid] = float(idx)
            # Synthetic octave shift: +/-1 octave
            next_o = oct_order[min(len(oct_order) - 1, idx + 1) if idx < 4 else max(0, idx - 1)]
            next_tid = tok_to_id.get(next_o)
            if next_tid is not None and next_tid < vocab_size:
                shift_map[tid] = next_tid

    return step_val, oct_val, shift_map


# ---------------------------------------------------------------------------
# Self-Contained Modules: RemoraModifierHead + FAPE2DOffsetHead + DAQCritic
# ---------------------------------------------------------------------------

def _build_remora_daq_fape_classes() -> Tuple[Any, Any, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import uni_omr_v2

    class RemoraModifierHead(nn.Module):
        """Decoupled local cross-attention head (~0.3 MB INT4) specialized for rare modifier tokens."""

        def __init__(self, d_model: int, vocab_size: int, d_mod: int = 128, n_heads: int = 4) -> None:
            super().__init__()
            self.d_mod = int(d_mod)
            self.n_heads = int(n_heads)
            self.head_dim = self.d_mod // self.n_heads
            self.q_proj = nn.Linear(d_model, self.d_mod, bias=False)
            self.k_proj = nn.Linear(d_model, self.d_mod, bias=False)
            self.v_proj = nn.Linear(d_model, self.d_mod, bias=False)
            self.o_proj = nn.Linear(self.d_mod, self.d_mod, bias=False)
            self.norm = uni_omr_v2.RMSNorm(self.d_mod)
            self.mod_logits = nn.Linear(self.d_mod, vocab_size, bias=False)
            self.gate = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
            self.register_buffer("mod_mask", _get_modifier_mask(vocab_size), persistent=False)

        def forward(self, dec_h: torch.Tensor, mem_final: torch.Tensor) -> torch.Tensor:
            B, T, _ = dec_h.shape
            Lk = mem_final.shape[1]
            q = self.q_proj(dec_h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(mem_final).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(mem_final).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            ctx = ctx.transpose(1, 2).reshape(B, T, self.d_mod)
            out = self.norm(self.o_proj(ctx))
            delta = self.mod_logits(out) * self.mod_mask.to(out.device, out.dtype)
            return torch.tanh(self.gate).to(delta.dtype) * delta

    class FAPE2DOffsetHead(nn.Module):
        """Training-only 2D-FAPE local-frame relative pitch/octave offset head (0.00 MB inference)."""

        def __init__(self, d_model: int, vocab_size: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                uni_omr_v2.RMSNorm(d_model),
                nn.Linear(d_model, 128, bias=True),
                nn.SiLU(),
                nn.Linear(128, 2, bias=True),  # (delta_step_norm, delta_oct_norm)
            )
            step_val, oct_val, _ = _get_step_and_oct_maps(vocab_size)
            self.register_buffer("step_val", step_val, persistent=False)
            self.register_buffer("oct_val", oct_val, persistent=False)

        def compute_fape_loss(self, dec_h: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
            """Clamped L1 loss on consecutive relative diatonic step & octave offsets."""
            pred = self.net(dec_h).float()  # (B, T, 2)
            V = self.step_val.numel()
            ids = tgt_ids.clamp(0, V - 1)
            sv = self.step_val.to(ids.device)[ids]  # (B, T)
            ov = self.oct_val.to(ids.device)[ids]   # (B, T)

            # Relative offset to previous token (distance <= 4 tokens back)
            B, T = ids.shape
            if T < 3:
                return pred.sum() * 0.0
            d_step = (sv[:, 1:] - sv[:, :-1]) / 7.0
            d_oct = (ov[:, 1:] - ov[:, :-1]) / 4.0
            valid_step = (sv[:, 1:] >= 0.0) & (sv[:, :-1] >= 0.0)
            valid_oct = (ov[:, 1:] >= 0.0) & (ov[:, :-1] >= 0.0)

            # Clamped L1 (FAPE-style clamp at 1.0)
            err_s = (pred[:, 1:, 0] - d_step).abs().clamp(max=1.0)
            err_o = (pred[:, 1:, 1] - d_oct).abs().clamp(max=1.0)
            loss_s = (err_s * valid_step.float()).sum() / valid_step.float().sum().clamp(min=1.0)
            loss_o = (err_o * valid_oct.float()).sum() / valid_oct.float().sum().clamp(min=1.0)
            # Keep graph connected even if no adjacent step/oct pair in tiny mock batch
            return loss_s + loss_o + 1e-6 * (pred ** 2).mean()

    class DAQRegisterCriticHead(nn.Module):
        """Independent local visual-vs-token register-shift critic (~0.1 MB INT4)."""

        def __init__(self, d_model: int, vocab_size: int, d_daq: int = 96) -> None:
            super().__init__()
            self.d_daq = int(d_daq)
            self.tok_proj = nn.Linear(d_model, self.d_daq, bias=False)
            self.vis_proj = nn.Linear(d_model, self.d_daq, bias=False)
            self.norm = uni_omr_v2.RMSNorm(self.d_daq * 2)
            self.cls = nn.Linear(self.d_daq * 2, 1, bias=True)
            _, _, shift_map = _get_step_and_oct_maps(vocab_size)
            self.register_buffer("shift_map", shift_map, persistent=False)

        def score_consistency(self, tok_embs: torch.Tensor, mem_final: torch.Tensor) -> torch.Tensor:
            """Return per-page register consistency logit (B,)."""
            t_pool = self.tok_proj(tok_embs).mean(dim=1)   # (B, d_daq)
            v_pool = self.vis_proj(mem_final).mean(dim=1)  # (B, d_daq)
            feat = self.norm(torch.cat([t_pool, v_pool], dim=-1))
            return self.cls(feat).squeeze(-1)

        def compute_daq_loss_and_acc(
            self,
            tok_emb_layer: nn.Embedding,
            tgt_ids: torch.Tensor,
            mem_final: torch.Tensor,
        ) -> Tuple[torch.Tensor, float]:
            """BCE loss contrasting positive real token streams vs synthetic register-shifted negatives."""
            V = self.shift_map.numel()
            pos_embs = tok_emb_layer(tgt_ids.clamp(0, V - 1))
            neg_ids = self.shift_map.to(tgt_ids.device)[tgt_ids.clamp(0, V - 1)]
            neg_embs = tok_emb_layer(neg_ids)

            pos_logit = self.score_consistency(pos_embs, mem_final).float()
            neg_logit = self.score_consistency(neg_embs, mem_final).float()
            logits = torch.cat([pos_logit, neg_logit], dim=0)
            labels = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)], dim=0)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            with torch.no_grad():
                preds = (logits > 0.0).float()
                acc = float((preds == labels).float().mean().item())
            return loss, acc

    return RemoraModifierHead, FAPE2DOffsetHead, DAQRegisterCriticHead


def apply_remora_daq_fape(model: Any, tiny: bool = False) -> Any:
    """Attach RemoraModifierHead, FAPE2DOffsetHead (training-only), and DAQRegisterCriticHead."""
    RemoraModifierHead, FAPE2DOffsetHead, DAQRegisterCriticHead = _build_remora_daq_fape_classes()
    c = model.cfg
    model.remora_head = RemoraModifierHead(
        d_model=c.d_model, vocab_size=int(c.vocab_size), d_mod=32 if tiny else 128, n_heads=2 if tiny else 4
    )
    model.fape_head = FAPE2DOffsetHead(d_model=c.d_model, vocab_size=int(c.vocab_size))
    model.daq_critic = DAQRegisterCriticHead(
        d_model=c.d_model, vocab_size=int(c.vocab_size), d_daq=32 if tiny else 96
    )

    orig_strip = model.strip_aux_heads

    def _strip_with_fape(self: Any, strip_mtp: bool = False) -> Any:
        orig_strip(strip_mtp=strip_mtp)
        self.fape_head = None  # 0.00 MB inference cost!
        return self

    model.strip_aux_heads = types.MethodType(_strip_with_fape, model)

    orig_decode = model.decode_tokens

    def _decode_with_remora(
        self: Any,
        tgt_ids: Any,
        memory: Any,
        return_mtp: Optional[bool] = None,
        return_aux: Optional[bool] = None,
        past_kvs: Optional[List[Any]] = None,
        use_cache: bool = False,
        apply_tok_dropout: bool = True,
    ) -> Any:
        # Capture post-dec_norm hidden state via a lightweight forward hook on dec_norm
        captured: Dict[str, Any] = {}

        def _hook(_mod: Any, _inp: Any, out_h: Any) -> None:
            captured["h"] = out_h

        handle = self.dec_norm.register_forward_hook(_hook)
        try:
            out = orig_decode(
                tgt_ids,
                memory,
                return_mtp=return_mtp,
                return_aux=return_aux,
                past_kvs=past_kvs,
                use_cache=use_cache,
                apply_tok_dropout=apply_tok_dropout,
            )
        finally:
            handle.remove()

        logits, mtp_logits, third = out
        dec_h = captured.get("h")
        mem_final = memory[-1] if isinstance(memory, (tuple, list)) else memory

        if dec_h is not None and mem_final is not None and not use_cache:
            logits = logits + self.remora_head(dec_h, mem_final)
            if self.fape_head is not None and self.training:
                self._last_fape_loss = self.fape_head.compute_fape_loss(dec_h, tgt_ids)
            else:
                self._last_fape_loss = None
            if self.daq_critic is not None and self.training:
                daq_loss, daq_acc = self.daq_critic.compute_daq_loss_and_acc(self.tok_emb, tgt_ids, mem_final)
                self._last_daq_loss = daq_loss
                self._last_daq_acc = daq_acc
            else:
                self._last_daq_loss = None
        return logits, mtp_logits, third

    model.decode_tokens = types.MethodType(_decode_with_remora, model)
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
    if arm_name == "remora_daq_fape":
        apply_remora_daq_fape(model, tiny=tiny)
    elif arm_name != "ctrl":
        raise ValueError(f"Unknown arm_name: {arm_name}")
    return model


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
    infer_model.strip_aux_heads()  # strips FAPE2DOffsetHead (0.00 MB inference)
    infer_model.mtp_heads = nn.ModuleList()

    int4_roots = [
        infer_model.encoder.axial_layers,
        infer_model.encoder.patch_merger,
        infer_model.encoder.global_layers,
        infer_model.deepstack_fusions,
        infer_model.dec_layers,
        infer_model.lm_head,
    ]
    if hasattr(infer_model, "remora_head") and infer_model.remora_head is not None:
        int4_roots.append(infer_model.remora_head)
    if hasattr(infer_model, "daq_critic") and infer_model.daq_critic is not None:
        int4_roots.append(infer_model.daq_critic)

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


def _eval_val_metrics(
    model: Any,
    val_pairs: List[Tuple[Path, List[Path], str, float]],
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, float]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr

    model.eval()
    mod_mask = _get_modifier_mask(int(model.cfg.vocab_size)).to(device)
    losses: List[float] = []
    mod_losses: List[float] = []
    daq_accs: List[float] = []

    with torch.no_grad():
        for img_p, beks, _, _ in val_pairs:
            ids = _encode_bekerns_full(beks)
            if len(ids) < 3:
                continue
            inp = _pil_to_tensor(Image.open(img_p), FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
            tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                mem, _, _ = model.encode_image(inp, return_aux=False)
                logits, _, _ = model.decode_tokens(
                    tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
                flat_logits = logits.reshape(-1, logits.size(-1))
                flat_tgt = tgt[:, 1:].reshape(-1)
                loss = F.cross_entropy(flat_logits, flat_tgt, ignore_index=uni_omr.PAD_ID)
                losses.append(float(loss.item()))

                # Modifier-only CE
                is_mod = mod_mask[flat_tgt.clamp(0, logits.size(-1) - 1)] & (flat_tgt != uni_omr.PAD_ID)
                if bool(is_mod.any()):
                    mloss = F.cross_entropy(flat_logits[is_mod], flat_tgt[is_mod])
                    mod_losses.append(float(mloss.item()))

                if hasattr(model, "daq_critic") and model.daq_critic is not None:
                    mem_final = mem[-1] if isinstance(mem, (tuple, list)) else mem
                    _, dacc = model.daq_critic.compute_daq_loss_and_acc(model.tok_emb, tgt[:, 1:], mem_final)
                    daq_accs.append(dacc)

    out = {
        "val_ce": round(sum(losses) / max(1, len(losses)), 5),
        "modifier_val_ce": round(sum(mod_losses) / max(1, len(mod_losses)), 5) if mod_losses else 0.0,
    }
    if daq_accs:
        out["daq_register_shift_acc"] = round(sum(daq_accs) / len(daq_accs), 5)
    return out


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
        opt_list = getattr(opt, "optimizers", [o for o, _ in opt._opts])
        if "opt_state_dicts" in ck and len(ck["opt_state_dicts"]) == len(opt_list):
            for o_obj, sd in zip(opt_list, ck["opt_state_dicts"]):
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
                if arm_name == "remora_daq_fape":
                    if getattr(model, "_last_fape_loss", None) is not None:
                        total_loss = total_loss + 0.15 * model._last_fape_loss
                    if getattr(model, "_last_daq_loss", None) is not None:
                        total_loss = total_loss + 0.10 * model._last_daq_loss
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
                    "opt_state_dicts": [o.state_dict() for o in getattr(opt, "optimizers", [o for o, _ in opt._opts])],
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
    raw_metrics = _eval_val_metrics(model, val_pairs, device, use_bf16)
    model.load_state_dict(ema_state, strict=True)
    ema_metrics = _eval_val_metrics(model, val_pairs, device, use_bf16)

    packed_info = _pack_and_verify_hybrid_int4(model, arm_name, arm_dir / f"{arm_name}_hybrid_int4.bin", INT4_GROUP_SIZE)
    dq_model = packed_info.pop("dequant_model").to(device)
    dq_metrics = _eval_val_metrics(dq_model, val_pairs, device, use_bf16)

    steady_ms = step_ms_hist[10:] if len(step_ms_hist) > 10 else step_ms_hist
    result: Dict[str, Any] = {
        "arm": arm_name,
        "optimizer_mode": opt.mode,
        "steps": TRAIN_STEPS,
        "param_summary": psummary,
        "packed_int4_mb": packed_info["packed_int4_mb"],
        "packed_int4_verification": packed_info,
        "train_loss_tail50": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5),
        "raw_val_ce": raw_metrics["val_ce"],
        "ema_val_ce": ema_metrics["val_ce"],
        "modifier_val_ce": ema_metrics["modifier_val_ce"],
        "int4_dequant_val_ce": dq_metrics["val_ce"],
        "step_ms": round(sum(steady_ms) / max(1, len(steady_ms)), 2),
        "train_elapsed_sec": round(train_elapsed_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }
    if "daq_register_shift_acc" in ema_metrics:
        result["daq_register_shift_acc"] = ema_metrics["daq_register_shift_acc"]
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
    """2-step tiny CPU forward+backward+step + FAPE strip + DAQ check + edge85m packed_int4_mb."""
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
                if arm_name == "remora_daq_fape":
                    assert model._last_fape_loss is not None and model._last_daq_loss is not None
                    loss = loss + 0.15 * model._last_fape_loss + 0.10 * model._last_daq_loss
                assert torch.isfinite(loss), f"{arm_name} step {step}: non-finite loss"
                loss.backward()
                if arm_name == "remora_daq_fape":
                    assert model.remora_head.gate.grad is not None and torch.isfinite(model.remora_head.gate.grad).all()
                    assert model.fape_head.net[1].weight.grad is not None
                    assert model.daq_critic.cls.weight.grad is not None
                opt.step()
                losses_seen.append(round(float(loss.item()), 4))

            pack_tiny = _pack_and_verify_hybrid_int4(
                model, arm_name, Path(tmpd) / f"{arm_name}_tiny.bin", INT4_GROUP_SIZE, tiny=True
            )
            dq_model = pack_tiny.pop("dequant_model")
            if arm_name == "remora_daq_fape":
                assert dq_model.fape_head is None, "FAPE2DOffsetHead must be stripped at export (0.00 MB)"
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

    delta_mb = round(edge_mb["remora_daq_fape"] - edge_mb["ctrl"], 3)
    print(f"[{EXP_CODE}][SMOKE] tiny ctrl:            {tiny_results['ctrl']}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] tiny remora_daq_fape: {tiny_results['remora_daq_fape']}", flush=True)
    print(
        f"[{EXP_CODE}][SMOKE] edge85m packed_int4_mb: ctrl={edge_mb['ctrl']} MB, "
        f"remora_daq_fape={edge_mb['remora_daq_fape']} MB (delta={delta_mb:+.3f} MB)",
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
            if "ctrl" in arm_results and "remora_daq_fape" in arm_results:
                c, t = arm_results["ctrl"], arm_results["remora_daq_fape"]
                eval_payload["delta_remora_minus_ctrl"] = {
                    "ema_val_ce": round(t["ema_val_ce"] - c["ema_val_ce"], 5),
                    "modifier_val_ce": round(t["modifier_val_ce"] - c["modifier_val_ce"], 5),
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
