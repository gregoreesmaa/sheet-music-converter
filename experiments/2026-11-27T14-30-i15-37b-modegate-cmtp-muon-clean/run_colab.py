#!/usr/bin/env python3
"""Colab A100 / Local RTX 4080 SUPER runner — Iter 15 Exp i15-37b:
Clean Muon+AdamW + INT4-QAT verification of Modality-Gated Dual-Head + Chained MTP-4.

Experiment ID: 2026-11-27T14-30-i15-37b-modegate-cmtp-muon-clean
Successor to 2026-11-27T10-30-i15-37-modegate-dualhead-text-and-chained-mtp (confounded by
"base." wrapper prefix -> AdamW fallback on Arm 1, fixed in commit 95446af / b64fd64).

Hard invariants:
  - Asserts `opt.mode == "muon+adamw"` on BOTH `ctrl` and `modegate_cmtp`.
  - Runs QATInt4ControllerV2 from step 420 + exact disk-verified Hybrid FP16/INT4 dequant val_ce.
  - Uses EMA_DECAY=0.99 (matching i15-38 / i15-33 / i15-14).
  - Includes HF Private Hub per-arm partial checkpoint sync + startup auto-resume.
  - Applies Windows cuDNN 9.10 safety when running on Windows.

Security:
  - Reads HF_TOKEN from /tmp/.hf_token (chmod 600), HF_TOKEN env, or ~/.cache/huggingface/token.
  - Never prints tokens; always deletes /tmp/.hf_token in finally.
"""

from __future__ import annotations

import argparse
import contextlib
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-27T14-30-i15-37b-modegate-cmtp-muon-clean"
EXP_CODE = "i15-37b"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_CKPT_PREFIX = f"checkpoints/{EXP_ID}"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORK_ROOT = Path("/vol/i15_37b") if Path("/vol").is_dir() else Path("/content/i15_37b")
OUT_ROOT = WORK_ROOT / "out" / EXP_ID

STEPS_PER_ARM = int(os.environ.get("I15_37B_STEPS", "600"))
SEED = 42
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 3e-4
WEIGHT_DECAY = 0.04
WARMUP_STEPS = 60
QAT_START_STEP = 420
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
GPU_EMA_DECAY = 0.99
INT4_GROUP_SIZE = 32
HARD_BUDGET_MB = 55.0
MTP_LOSS_WEIGHT = 0.30
SALIENCY_LOSS_WEIGHT = 0.10
SYS_COUNT_LOSS_WEIGHT = 0.15
STROKE_HOG_WEIGHT = 0.15
TEXT_AUX_WEIGHT = 0.02
TEXT_VOCAB_SIZE = 129
MODEGATE_LAYERS = 4
MODEGATE_RANK = 32
LOG_EVERY = 25

ARMS: List[Dict[str, Any]] = [
    {"name": "ctrl", "mode": "ctrl", "is_control": True},
    {"name": "modegate_cmtp", "mode": "modegate_cmtp", "is_control": False},
]


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
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
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


def _ensure_deps() -> None:
    needed = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])


def _locate_or_unpack_mvd() -> Path:
    candidates = [
        Path.cwd() / "tmp-train" / "mvd",
        Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        WORK_ROOT / "mvd",
        WORK_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
    ]
    for cand in candidates:
        if (cand / "train_uni_v2.py").is_file() and (cand / "uni_omr_v2.py").is_file():
            return cand

    for bundle in (Path("/content/bundle.tar.gz"), Path("/content/mvd_bundle.tar.gz")):
        if bundle.is_file():
            WORK_ROOT.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORK_ROOT)
            for p in WORK_ROOT.rglob("train_uni_v2.py"):
                return p.parent

    raise FileNotFoundError(
        "Could not locate tmp-train/mvd/train_uni_v2.py. Upload /content/bundle.tar.gz "
        "containing tmp-train/mvd before running."
    )


def _try_resume_from_hf(hf_token: str) -> Dict[str, Dict[str, Any]]:
    """Download any already-completed arm results from HF Private Hub into OUT_ROOT."""
    from huggingface_hub import hf_hub_download

    recovered: Dict[str, Dict[str, Any]] = {}
    dl_dir = WORK_ROOT / "hf_resume"
    dl_dir.mkdir(parents=True, exist_ok=True)
    arm_names = {a["name"] for a in ARMS}

    try:
        ev_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_EVAL_PATH,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(dl_dir),
        )
        ev_data = json.loads(Path(ev_path).read_text(encoding="utf-8"))
        arms_obj = ev_data.get("arms") or []
        if isinstance(arms_obj, list):
            for rec in arms_obj:
                if isinstance(rec, dict) and rec.get("arm") in arm_names and "ema_val_ce" in rec:
                    recovered[rec["arm"]] = rec
        elif isinstance(arms_obj, dict):
            for k, rec in arms_obj.items():
                if k in arm_names and isinstance(rec, dict) and "ema_val_ce" in rec:
                    recovered[k] = rec
    except Exception:
        pass

    for arm in ARMS:
        name = arm["name"]
        if name in recovered:
            continue
        try:
            p = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"{HF_CKPT_PREFIX}/{name}_arm_result.json",
                repo_type="dataset",
                token=hf_token,
                local_dir=str(dl_dir),
            )
            rec = json.loads(Path(p).read_text(encoding="utf-8"))
            if isinstance(rec, dict) and "ema_val_ce" in rec:
                recovered[name] = rec
        except Exception:
            pass

    for name, rec in recovered.items():
        done_file = OUT_ROOT / name / "arm_done.json"
        if not done_file.is_file():
            _atomic_write_json(done_file, rec)
        print(f"[{EXP_ID}] Auto-resumed completed arm '{name}' from HF Private Hub", flush=True)

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


def _upload_json_to_hf(hf_token: str, local_path: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"{EXP_ID}: upload {path_in_repo}",
    )
    print(f"[{EXP_ID}] Uploaded {path_in_repo} -> {HF_REPO_ID}", flush=True)


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


# ---------------------------------------------------------------------------
# Self-Contained Experimental Modules (NO edits to uni_omr_v2.py)
# ---------------------------------------------------------------------------

def _build_experimental_classes() -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from uni_omr_v2 import PAD_ID, RMSNorm, UNI_ID_TO_TOKEN, UniOMRv2Net

    class ChainedMTPHead(nn.Module):
        """DeepSeek-V3 sequential Chained MTP projection:
        ``h_k = h_{k-1} + SiLU(W_k([RMSNorm(h_{k-1}); RMSNorm(tok_emb(y_{t+k}))]))``.
        """

        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.norm_h = RMSNorm(d_model)
            self.norm_e = RMSNorm(d_model)
            self.proj = nn.Linear(2 * d_model, d_model, bias=False)

        def forward(self, h_prev: torch.Tensor, next_tok_emb: torch.Tensor) -> torch.Tensor:
            fused = torch.cat([self.norm_h(h_prev), self.norm_e(next_tok_emb)], dim=-1)
            return h_prev + F.silu(self.proj(fused))

    class ModeGatedCrossAttn(nn.Module):
        """Modality-Gated Cross-Attention wrapper for decoder layers 0..3."""

        def __init__(self, inner: nn.Module, d_model: int, rank: int = MODEGATE_RANK) -> None:
            super().__init__()
            self.inner = inner
            self.d_model = int(d_model)
            self.rank = int(rank)
            self.mode_gate = nn.Linear(d_model, 1, bias=True)
            nn.init.zeros_(self.mode_gate.weight)
            nn.init.constant_(self.mode_gate.bias, -2.0)
            self.film_gamma = nn.Parameter(torch.zeros(1, 1, d_model))
            self.film_beta = nn.Parameter(torch.zeros(1, 1, d_model))
            self.q_down = nn.Linear(d_model, rank, bias=False)
            self.q_up = nn.Linear(rank, d_model, bias=False)
            nn.init.zeros_(self.q_up.weight)

        def forward(self, q_in: torch.Tensor, kv_in: Optional[torch.Tensor], *args: Any, **kwargs: Any) -> Any:
            alpha = torch.sigmoid(self.mode_gate(q_in)).to(q_in.dtype)
            gamma = self.film_gamma.to(q_in.dtype)
            beta = self.film_beta.to(q_in.dtype)
            q_delta = self.q_up(F.silu(self.q_down(q_in)))
            q_mod = q_in * (1.0 + alpha * gamma) + alpha * (beta + q_delta)
            return self.inner(q_mod, kv_in, *args, **kwargs)

    _TEXT_PREFIXES = ("dyn-", "hairpin-", "art-", "clef-", "time-", "key", "ottava-", "volta-")

    def build_text_aux_targets(targets: torch.Tensor, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = targets.device
        lut = torch.full((vocab_size,), 128, dtype=torch.long)
        for tid, tok in UNI_ID_TO_TOKEN.items():
            if 0 <= tid < vocab_size and any(tok.startswith(p) for p in _TEXT_PREFIXES):
                lut[tid] = tid % 128
        lut = lut.to(device)
        safe_t = targets.clamp(0, vocab_size - 1)
        text_tgt = lut[safe_t]
        valid = targets != PAD_ID
        text_tgt = torch.where(valid, text_tgt, torch.full_like(text_tgt, -100))
        is_text = (text_tgt >= 0) & (text_tgt < 128)
        return text_tgt, is_text

    class ModeGateChainedMTPWrapper(nn.Module):
        """Self-contained Arm 1 wrapper around ``UniOMRv2Net``."""

        def __init__(
            self,
            base: UniOMRv2Net,
            text_vocab_size: int = TEXT_VOCAB_SIZE,
            modegate_layers: int = MODEGATE_LAYERS,
            modegate_rank: int = MODEGATE_RANK,
        ) -> None:
            super().__init__()
            self.base = base
            self.cfg = base.cfg
            d_model = int(base.cfg.d_model)
            n_extra_mtp = max(0, int(base.cfg.mtp_k) - 1)

            self.base.mtp_heads = nn.ModuleList([ChainedMTPHead(d_model) for _ in range(n_extra_mtp)])

            n_wrap = min(int(modegate_layers), len(self.base.dec_layers))
            for idx in range(n_wrap):
                layer = self.base.dec_layers[idx]
                layer.cross_attn = ModeGatedCrossAttn(layer.cross_attn, d_model, rank=modegate_rank)

            self.lm_head_text = nn.Linear(d_model, int(text_vocab_size), bias=False)

        def strip_aux_heads(self) -> None:
            self.base.strip_aux_heads()

        def _decode_hidden(
            self,
            tgt_ids: torch.Tensor,
            memory: Any,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            b = self.base
            _, T = tgt_ids.shape
            x = b.tok_emb(tgt_ids)
            compute_dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
            if x.dtype != compute_dtype:
                x = x.to(dtype=compute_dtype)
            cos1d, sin1d = b.rope1d(T, device=x.device, dtype=compute_dtype)

            mem_final = None
            level_mems = None
            if memory is not None:
                if isinstance(memory, (tuple, list)):
                    mem_post_merger, mem_global_mid, mem_final = memory
                    level_mems = (mem_post_merger, mem_global_mid)
                else:
                    mem_final = memory

            bt = int(x.shape[0] * x.shape[1])
            can_dec_ckpt = self.training and b.cfg.grad_checkpoint and torch.is_grad_enabled()
            use_dec_ffn_ckpt = can_dec_ckpt and (4096 < bt <= 8192)
            for idx, layer in enumerate(b.dec_layers):
                if mem_final is None:
                    layer_mem = None
                elif level_mems is not None and idx < len(b.deepstack_fusions):
                    layer_mem = b.deepstack_fusions[idx](mem_final, level_mems[idx])
                else:
                    layer_mem = mem_final
                x = layer(x, memory=layer_mem, rope_cos=cos1d, rope_sin=sin1d, checkpoint_ffn=use_dec_ffn_ckpt)
            h = b.dec_norm(x)
            sys_y_pred = b.sys_y_head(h) if (self.training and b.sys_y_head is not None) else None
            return h, sys_y_pred

        def forward(
            self,
            images: torch.Tensor,
            tgt_ids: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            return_aux: Optional[bool] = None,
            return_mtp: bool = True,
            greedy_chain_mtp: bool = False,
        ) -> Dict[str, Any]:
            mem_levels, aux_enc, router_info = self.base.encode_image(images, return_aux=return_aux)
            h0, sys_y_pred = self._decode_hidden(tgt_ids, mem_levels)
            logits = self.base.lm_head(h0)
            text_logits = self.lm_head_text(h0)

            mtp_logits: List[torch.Tensor] = []
            if return_mtp and len(self.base.mtp_heads) > 0:
                B, T = tgt_ids.shape
                h_prev = h0
                prev_logits = logits
                for k_idx, head in enumerate(self.base.mtp_heads):
                    if greedy_chain_mtp or targets is None:
                        next_tok = prev_logits.detach().argmax(dim=-1)
                    else:
                        if k_idx == 0:
                            next_tok = targets
                        elif T > k_idx:
                            pad_tail = torch.full((B, k_idx), PAD_ID, device=targets.device, dtype=targets.dtype)
                            next_tok = torch.cat([targets[:, k_idx:], pad_tail], dim=1)
                        else:
                            next_tok = torch.full((B, T), PAD_ID, device=targets.device, dtype=targets.dtype)
                    next_emb = self.base.tok_emb(next_tok).to(h_prev.dtype)
                    h_k = head(h_prev, next_emb)
                    logits_k = self.base.lm_head(h_k)
                    mtp_logits.append(logits_k)
                    h_prev = h_k
                    prev_logits = logits_k

            return {
                "logits": logits,
                "text_logits": text_logits,
                "mtp_logits": mtp_logits,
                "sys_y_pred": sys_y_pred,
                "sys_count_pred": aux_enc["sys_count_pred"],
                "staff_seg_logits": aux_enc["staff_seg_logits"],
                "stroke_hog_pred": aux_enc["stroke_hog_pred"],
                "router_info": router_info,
            }

        def compute_loss(self, out: Dict[str, Any], targets: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
            losses = self.base.compute_loss(out, targets, **kwargs)
            text_logits = out.get("text_logits")
            if text_logits is not None and TEXT_AUX_WEIGHT > 0.0:
                text_tgt, _ = build_text_aux_targets(targets, int(self.cfg.vocab_size))
                text_loss = F.cross_entropy(
                    text_logits.reshape(-1, text_logits.shape[-1]),
                    text_tgt.reshape(-1),
                    ignore_index=-100,
                )
                losses["text_aux_loss"] = text_loss
                losses["loss"] = losses["loss"] + TEXT_AUX_WEIGHT * text_loss
            else:
                losses["text_aux_loss"] = losses["loss"].new_zeros(())
            return losses

    return {
        "ChainedMTPHead": ChainedMTPHead,
        "ModeGatedCrossAttn": ModeGatedCrossAttn,
        "ModeGateChainedMTPWrapper": ModeGateChainedMTPWrapper,
    }


def _strip_base_prefix(name: str) -> str:
    return name[5:] if name.startswith("base.") else name


def _compute_packed_int4_stats(model: Any, int4_block_size: int = INT4_GROUP_SIZE) -> Dict[str, Any]:
    import torch.nn as nn

    aux_prefixes = ("sys_y_head", "sys_count_head", "staff_seg_head", "stroke_hog_head")
    int4_prefixes = (
        "encoder.axial_layers",
        "encoder.patch_merger",
        "encoder.global_layers",
        "deepstack_fusions",
        "dec_layers",
        "lm_head",
        "lm_head_text",
        "mtp_heads",
    )

    def _calc(include_mtp: bool) -> Tuple[int, int, float]:
        int4_weight_ids: set[int] = set()
        for m_name, mod in model.named_modules():
            clean = _strip_base_prefix(m_name)
            if any(clean == p or clean.startswith(p + ".") for p in aux_prefixes):
                continue
            if (not include_mtp) and (clean == "mtp_heads" or clean.startswith("mtp_heads.")):
                continue
            if isinstance(mod, nn.Linear) and any(clean == p or clean.startswith(p + ".") for p in int4_prefixes):
                int4_weight_ids.add(id(mod.weight))

        total_p = 0
        int4_p = 0
        seen: set[int] = set()
        for p_name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            clean = _strip_base_prefix(p_name)
            if any(clean == pref or clean.startswith(pref + ".") for pref in aux_prefixes):
                continue
            if (not include_mtp) and (clean == "mtp_heads" or clean.startswith("mtp_heads.")):
                continue
            n = int(p.numel())
            total_p += n
            if id(p) in int4_weight_ids:
                int4_p += n
        fp16_p = total_p - int4_p
        scale_b = 2.0 / float(int4_block_size)
        mb = (fp16_p * 2.0 + int4_p * (0.5 + scale_b)) / 1_000_000.0
        return total_p, int4_p, round(mb, 3)

    stripped_params, _, stripped_mb = _calc(include_mtp=False)
    with_mtp_params, _, with_mtp_mb = _calc(include_mtp=True)
    return {
        "infer_params_stripped": stripped_params,
        "infer_params_with_mtp": with_mtp_params,
        "mtp_params": with_mtp_params - stripped_params,
        "packed_int4_stripped_mb": stripped_mb,
        "packed_int4_with_mtp_mb": with_mtp_mb,
    }


def _pack_and_dequant_hybrid_int4(
    model: Any,
    arm_mode: str,
    cfg: Any,
    out_bin: Path,
    group_size: int = INT4_GROUP_SIZE,
) -> Tuple[Any, Dict[str, Any]]:
    """Pack Hybrid FP16/INT4 (keeping MTP heads for full speculative eval) to exact disk bytes
    and return a dequantized model copy + exact byte statistics."""
    import torch
    import torch.nn as nn

    infer_model = _build_arm_model(arm_mode, cfg).cpu().eval()
    infer_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True)
    infer_model.strip_aux_heads()

    int4_prefixes = (
        "encoder.axial_layers",
        "encoder.patch_merger",
        "encoder.global_layers",
        "deepstack_fusions",
        "dec_layers",
        "lm_head",
        "lm_head_text",
        "mtp_heads",
    )
    int4_param_names: set[str] = set()
    for m_name, mod in infer_model.named_modules():
        clean = _strip_base_prefix(m_name)
        if isinstance(mod, nn.Linear) and any(clean == p or clean.startswith(p + ".") for p in int4_prefixes):
            int4_param_names.add(f"{m_name}.weight" if m_name else "weight")

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    dequant_sd: Dict[str, torch.Tensor] = {}
    n_int4_tensors = 0
    n_fp16_tensors = 0
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
                f.write(packed.tobytes())
                f.write(scale.numpy().tobytes())
                deq = (q.float() * scale.float()).reshape(-1)[:orig_n].reshape(t.shape)
                dequant_sd[name] = deq.to(param.dtype)
                n_int4_tensors += 1
            else:
                fp16 = t.to(torch.float16)
                f.write(fp16.numpy().tobytes())
                dequant_sd[name] = fp16.to(param.dtype)
                n_fp16_tensors += 1

    infer_model.load_state_dict(dequant_sd, strict=False)
    disk_bytes = out_bin.stat().st_size
    return infer_model, {
        "disk_int4_with_mtp_bytes": disk_bytes,
        "disk_int4_with_mtp_mb": round(disk_bytes / 1_000_000.0, 3),
        "n_int4_tensors": n_int4_tensors,
        "n_fp16_tensors": n_fp16_tensors,
    }


# ---------------------------------------------------------------------------
# Self-Contained Data Loader, EMA, and Eval
# ---------------------------------------------------------------------------

class _EMA:
    def __init__(self, model: Any, decay: float = GPU_EMA_DECAY) -> None:
        import torch

        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad
        }

    def update(self, model: Any) -> None:
        import torch

        with torch.no_grad():
            d = self.decay
            for k, p in model.named_parameters():
                if k in self.shadow:
                    self.shadow[k].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextlib.contextmanager
    def apply(self, model: Any):
        backup: Dict[str, Any] = {}
        for k, p in model.named_parameters():
            if k in self.shadow:
                backup[k] = p.detach().clone()
                p.data.copy_(self.shadow[k])
        try:
            yield
        finally:
            for k, p in model.named_parameters():
                if k in backup:
                    p.data.copy_(backup[k])


def _load_manifest_rows(manifest_path: Path) -> List[Tuple[Path, List[Path]]]:
    rows: List[Tuple[Path, List[Path]]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        img_p = Path(rec["image"])
        beks = [Path(b) for b in rec.get("bekerns", [])]
        if img_p.is_file() and beks:
            rows.append((img_p, beks))
    return rows


def _load_page_sample(
    img_path: Path,
    bek_paths: Sequence[Path],
    max_dim: int,
    max_seq_len: int,
) -> Tuple[Any, List[int]]:
    import numpy as np
    import torch
    from PIL import Image
    import data
    from uni_omr import BOS_ID, EOS_ID, SYS_TOKEN, encode_uni

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, float(max_dim) / float(max(w, h)))
        nw = max(32, int(round(w * scale / 16.0)) * 16)
        nh = max(32, int(round(h * scale / 16.0)) * 16)
        if (nw, nh) != (w, h):
            im = im.resize((nw, nh), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    img_t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    toks: List[str] = []
    for i, bp in enumerate(bek_paths):
        if i > 0:
            toks.append(SYS_TOKEN)
        toks.extend(data.load_bekern(bp))
    ids = encode_uni(toks, add_bos_eos=True)
    if len(ids) > max_seq_len:
        ids = ids[: max_seq_len - 1] + [EOS_ID]
    if len(ids) < 2:
        ids = [BOS_ID, EOS_ID]
    return img_t, ids


def _collate_pages(batch: Sequence[Tuple[Any, List[int]]], pad_id: int) -> Tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    max_h = max(int(x[0].shape[1]) for x in batch)
    max_w = max(int(x[0].shape[2]) for x in batch)
    max_h = int(math.ceil(max_h / 16.0) * 16)
    max_w = int(math.ceil(max_w / 16.0) * 16)
    imgs = []
    for img, _ in batch:
        ph = max_h - int(img.shape[1])
        pw = max_w - int(img.shape[2])
        imgs.append(F.pad(img, (0, pw, 0, ph), value=1.0) if (ph or pw) else img)
    img_batch = torch.stack(imgs, dim=0)

    max_l = max(len(x[1]) for x in batch)
    seq_batch = torch.full((len(batch), max_l), pad_id, dtype=torch.long)
    for i, (_, ids) in enumerate(batch):
        seq_batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return img_batch, seq_batch


@contextlib.contextmanager
def _amp_ctx(device: Any, use_bf16: bool):
    import torch

    if use_bf16 and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def _eval_val_ce_and_mtp(
    model: Any,
    val_batches: Sequence[Tuple[Any, Any]],
    device: Any,
    use_bf16: bool,
    is_chained: bool,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from uni_omr_v2 import PAD_ID

    was_training = model.training
    model.eval()
    ce_sum = 0.0
    ce_toks = 0
    k_hits = [0, 0, 0, 0]
    k_tots = [0, 0, 0, 0]
    tf_k_hits = [0, 0, 0, 0]
    tf_k_tots = [0, 0, 0, 0]
    joint_accept_sum = 0.0
    joint_positions = 0

    with torch.no_grad():
        for imgs_cpu, seq_cpu in val_batches:
            imgs = imgs_cpu.to(device=device, dtype=(torch.bfloat16 if (use_bf16 and device.type == "cuda") else torch.float32))
            seq = seq_cpu.to(device=device)
            tgt_in = seq[:, :-1]
            tgt_out = seq[:, 1:]

            with _amp_ctx(device, use_bf16):
                if is_chained:
                    out = model(imgs, tgt_in, targets=tgt_out, return_aux=False, return_mtp=True, greedy_chain_mtp=True)
                    out_tf = model(imgs, tgt_in, targets=tgt_out, return_aux=False, return_mtp=True, greedy_chain_mtp=False)
                else:
                    out = model(imgs, tgt_in, return_aux=False, return_mtp=True)
                    out_tf = out

            logits = out["logits"]
            V = logits.shape[-1]
            valid = tgt_out != PAD_ID
            n_val = int(valid.sum().item())
            if n_val > 0:
                ce_sum += float(F.cross_entropy(logits.float().reshape(-1, V), tgt_out.reshape(-1), ignore_index=PAD_ID, reduction="sum").item())
                ce_toks += n_val

            all_logits = [logits] + list(out.get("mtp_logits", []))[:3]
            all_tf_logits = [out_tf["logits"]] + list(out_tf.get("mtp_logits", []))[:3]
            T = tgt_out.shape[1]

            for idx, lg in enumerate(all_logits):
                if T <= idx:
                    continue
                sub_tgt = tgt_out[:, idx:]
                sub_pred = lg[:, : T - idx, :].argmax(dim=-1)
                m = sub_tgt != PAD_ID
                k_hits[idx] += int(((sub_pred == sub_tgt) & m).sum().item())
                k_tots[idx] += int(m.sum().item())

            for idx, lg in enumerate(all_tf_logits):
                if T <= idx:
                    continue
                sub_tgt = tgt_out[:, idx:]
                sub_pred = lg[:, : T - idx, :].argmax(dim=-1)
                m = sub_tgt != PAD_ID
                tf_k_hits[idx] += int(((sub_pred == sub_tgt) & m).sum().item())
                tf_k_tots[idx] += int(m.sum().item())

            if len(all_logits) >= 4 and T > 3:
                L = T - 3
                m0 = tgt_out[:, :L] != PAD_ID
                match1 = (all_logits[0][:, :L, :].argmax(dim=-1) == tgt_out[:, :L]) & m0
                match2 = match1 & (all_logits[1][:, :L, :].argmax(dim=-1) == tgt_out[:, 1 : L + 1]) & (tgt_out[:, 1 : L + 1] != PAD_ID)
                match3 = match2 & (all_logits[2][:, :L, :].argmax(dim=-1) == tgt_out[:, 2 : L + 2]) & (tgt_out[:, 2 : L + 2] != PAD_ID)
                match4 = match3 & (all_logits[3][:, :L, :].argmax(dim=-1) == tgt_out[:, 3 : L + 3]) & (tgt_out[:, 3 : L + 3] != PAD_ID)
                n_pos = int(m0.sum().item())
                if n_pos > 0:
                    joint_accept_sum += float((match1.float() + match2.float() + match3.float() + match4.float()).sum().item())
                    joint_positions += n_pos

    if was_training:
        model.train()

    res: Dict[str, Any] = {
        "val_ce": round(ce_sum / max(1, ce_toks), 5),
        "val_tokens": ce_toks,
        "mtp_k1_acc": round(k_hits[0] / max(1, k_tots[0]), 5),
        "mtp_k2_acc": round(k_hits[1] / max(1, k_tots[1]), 5),
        "mtp_k3_acc": round(k_hits[2] / max(1, k_tots[2]), 5),
        "mtp_k4_acc": round(k_hits[3] / max(1, k_tots[3]), 5),
        "spec_accept_len": round(joint_accept_sum / max(1, joint_positions), 4),
    }
    if is_chained:
        res["mtp_tf_k2_acc"] = round(tf_k_hits[1] / max(1, tf_k_tots[1]), 5)
        res["mtp_tf_k3_acc"] = round(tf_k_hits[2] / max(1, tf_k_tots[2]), 5)
        res["mtp_tf_k4_acc"] = round(tf_k_hits[3] / max(1, tf_k_tots[3]), 5)
    return res


def _build_arm_model(arm_mode: str, cfg: Any) -> Any:
    from uni_omr_v2 import UniOMRv2Net

    base = UniOMRv2Net(cfg)
    if arm_mode == "ctrl":
        return base
    if arm_mode == "modegate_cmtp":
        cls_map = _build_experimental_classes()
        return cls_map["ModeGateChainedMTPWrapper"](
            base,
            text_vocab_size=TEXT_VOCAB_SIZE,
            modegate_layers=MODEGATE_LAYERS,
            modegate_rank=MODEGATE_RANK,
        )
    raise ValueError(f"Unknown arm_mode={arm_mode!r}")


def _run_arm(
    arm: Dict[str, Any],
    train_rows: Sequence[Tuple[Path, List[Path]]],
    val_batches: Sequence[Tuple[Any, Any]],
    device: Any,
    use_bf16: bool,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    import torch
    import train_uni_v2
    from uni_omr_v2 import PAD_ID, SYS_ID, EOS_ID, UniOMRv2Config

    arm_dir = OUT_ROOT / arm["name"]
    done_file = arm_dir / "arm_done.json"
    if done_file.is_file():
        print(f"[{EXP_ID}] Skipping completed arm {arm['name']} (found {done_file})", flush=True)
        return json.loads(done_file.read_text(encoding="utf-8"))
    arm_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(SEED)
    cfg = UniOMRv2Config.edge85m(
        n_axial_layers=0,
        n_global_layers=6,
        grad_checkpoint=True,
    )
    is_chained = arm["mode"] == "modegate_cmtp"
    model = _build_arm_model(arm["mode"], cfg).to(device)
    size_stats = _compute_packed_int4_stats(model)

    opt = train_uni_v2.build_optimizer_v2(
        model,
        optimizer_name="muon",
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device_type=device.type,
    )
    assert opt.mode == "muon+adamw", (
        f"Expected opt.mode == 'muon+adamw' on arm '{arm['name']}', got {opt.mode!r}! "
        f"(Check train_uni_v2.partition_muon_adamw_params wrapper prefix stripping.)"
    )

    qat_target = model.base if hasattr(model, "base") else model
    qat_ctrl = train_uni_v2.QATInt4ControllerV2(
        qat_target,
        group_size=INT4_GROUP_SIZE,
        start_step=QAT_START_STEP,
        enable=True,
    )
    ema = _EMA(model, decay=GPU_EMA_DECAY)

    rng = random.Random(SEED)
    n_train = len(train_rows)

    def _sample_batch() -> Tuple[Any, Any]:
        items = []
        for _ in range(BATCH_SIZE):
            ip, bps = train_rows[rng.randrange(n_train)]
            items.append(_load_page_sample(ip, bps, FULLPAGE_MAX_DIM, MAX_SEQ_LEN))
        return _collate_pages(items, PAD_ID)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print(
        f"\n[{EXP_ID}] === Arm {arm['name']} | infer_stripped={size_stats['infer_params_stripped']:,} "
        f"({size_stats['packed_int4_stripped_mb']:.2f} MB INT4) | w_mtp={size_stats['packed_int4_with_mtp_mb']:.2f} MB "
        f"| opt={opt.mode} | qat_start={QAT_START_STEP} | steps={STEPS_PER_ARM} ===",
        flush=True,
    )

    model.train()
    losses_tail: List[float] = []
    t_train0 = time.perf_counter()

    for step in range(1, STEPS_PER_ARM + 1):
        qat_ctrl.maybe_enable(step)
        opt.set_lr(_wsd_mult(step, STEPS_PER_ARM, WARMUP_STEPS))
        opt.zero_grad(set_to_none=True)
        step_loss_val = 0.0
        for _ in range(GRAD_ACCUM):
            imgs_cpu, seq_cpu = _sample_batch()
            imgs = imgs_cpu.to(device=device, dtype=(torch.bfloat16 if (use_bf16 and device.type == "cuda") else torch.float32))
            seq = seq_cpu.to(device=device)
            tgt_in = seq[:, :-1]
            tgt_out = seq[:, 1:]
            has_content = ((tgt_out != PAD_ID) & (tgt_out != EOS_ID)).any(dim=1).to(torch.float32)
            sys_count_tgt = has_content * ((seq == SYS_ID).sum(dim=1).to(torch.float32) + 1.0)

            with _amp_ctx(device, use_bf16):
                if is_chained:
                    out = model(imgs, tgt_in, targets=tgt_out, return_aux=True, return_mtp=True, greedy_chain_mtp=False)
                else:
                    out = model(imgs, tgt_in, return_aux=True, return_mtp=True)
                ldict = model.compute_loss(
                    out,
                    tgt_out,
                    sys_count_target=sys_count_tgt,
                    images=imgs,
                    mtp_loss_weight=MTP_LOSS_WEIGHT,
                    saliency_loss_weight=SALIENCY_LOSS_WEIGHT,
                    sys_y_loss_weight=0.0,
                    sys_count_loss_weight=SYS_COUNT_LOSS_WEIGHT,
                    staff_seg_loss_weight=0.0,
                    stroke_hog_loss_weight=STROKE_HOG_WEIGHT,
                )
                loss = ldict["loss"] / float(GRAD_ACCUM)
            loss.backward()
            step_loss_val += float(ldict["loss"].detach().item()) / float(GRAD_ACCUM)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        losses_tail.append(step_loss_val)
        if len(losses_tail) > 25:
            losses_tail.pop(0)

        if step == 1 or step % LOG_EVERY == 0 or step == STEPS_PER_ARM:
            elapsed = time.perf_counter() - t_train0
            step_ms = (elapsed / step) * 1000.0
            print(
                f"  [{arm['name']}] step {step:4d}/{STEPS_PER_ARM} | loss={step_loss_val:.4f} "
                f"| ce={float(ldict['ce_loss'].detach().item()):.4f} "
                f"| mtp={float(ldict['mtp_loss'].detach().item()):.4f} "
                f"| qat={int(qat_ctrl.active)} | {step_ms:.1f} ms/step",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    else:
        peak_vram_mb = 0.0
    train_elapsed = time.perf_counter() - t_train0
    step_ms = (train_elapsed / max(1, STEPS_PER_ARM)) * 1000.0

    qat_ctrl.remove()
    raw_eval = _eval_val_ce_and_mtp(model, val_batches, device, use_bf16, is_chained=is_chained)
    with ema.apply(model):
        ema_eval = _eval_val_ce_and_mtp(model, val_batches, device, use_bf16, is_chained=is_chained)
        int4_bin = arm_dir / f"{arm['name']}_hybrid_int4.bin"
        deq_model, int4_disk = _pack_and_dequant_hybrid_int4(model, arm["mode"], cfg, int4_bin)
        deq_model = deq_model.to(device)
        int4_eval = _eval_val_ce_and_mtp(deq_model, val_batches, device, use_bf16, is_chained=is_chained)
        del deq_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary: Dict[str, Any] = {
        "arm": arm["name"],
        "is_control": bool(arm["is_control"]),
        "opt_mode": opt.mode,
        "steps": STEPS_PER_ARM,
        "elapsed_sec": round(train_elapsed, 2),
        "step_ms": round(step_ms, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "final_25step_loss": round(sum(losses_tail) / max(1, len(losses_tail)), 4),
        "val_ce": raw_eval["val_ce"],
        "ema_val_ce": ema_eval["val_ce"],
        "int4_dequant_val_ce": int4_eval["val_ce"],
        "mtp_k1_acc": ema_eval["mtp_k1_acc"],
        "mtp_k2_acc": ema_eval["mtp_k2_acc"],
        "mtp_k3_acc": ema_eval["mtp_k3_acc"],
        "mtp_k4_acc": ema_eval["mtp_k4_acc"],
        "spec_accept_len": ema_eval["spec_accept_len"],
        "int4_spec_accept_len": int4_eval["spec_accept_len"],
        "raw_eval": raw_eval,
        "ema_eval": ema_eval,
        "int4_eval": int4_eval,
        **size_stats,
        **int4_disk,
    }
    _atomic_write_json(done_file, summary)
    if hf_token:
        try:
            _upload_json_to_hf(hf_token, done_file, f"{HF_CKPT_PREFIX}/{arm['name']}_arm_result.json")
        except Exception as up_exc:
            print(f"[{EXP_ID}] WARN: failed partial arm upload for {arm['name']}: {up_exc}", flush=True)

    del model, opt, ema
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def run_cpu_smoke_test() -> Dict[str, Any]:
    """1-step tiny CPU forward+backward+QAT+INT4+eval smoke test asserting opt.mode == 'muon+adamw'."""
    import tempfile
    import torch
    import train_uni_v2
    from uni_omr_v2 import BOS_ID, EOS_ID

    mvd_dir = _locate_or_unpack_mvd()
    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))

    device = torch.device("cpu")
    results: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmpd:
        for arm in ARMS:
            _seed_everything(SEED)
            cfg = train_uni_v2.uni_omr_v2_tiny(n_axial_layers=0, n_global_layers=2, n_dec_layers=4, mtp_k=4)
            is_chained = arm["mode"] == "modegate_cmtp"
            model = _build_arm_model(arm["mode"], cfg).to(device)
            opt = train_uni_v2.build_optimizer_v2(
                model, optimizer_name="muon", lr=1e-3, device_type="cpu", allow_cpu_muon=True
            )
            assert opt.mode == "muon+adamw", f"Smoke test failed: {arm['name']} got opt.mode={opt.mode!r}"
            assert len(opt.muon_param_names) > 0, f"Smoke test failed: {arm['name']} has 0 Muon params"

            imgs = torch.rand(2, 3, 64, 96)
            seq = torch.randint(0, 40, (2, 16), dtype=torch.long)
            seq[:, 0] = BOS_ID
            seq[:, -1] = EOS_ID
            tgt_in, tgt_out = seq[:, :-1], seq[:, 1:]

            model.train()
            opt.zero_grad(set_to_none=True)
            out = (
                model(imgs, tgt_in, targets=tgt_out, return_aux=True, return_mtp=True, greedy_chain_mtp=False)
                if is_chained
                else model(imgs, tgt_in, return_aux=True, return_mtp=True)
            )
            ldict = model.compute_loss(out, tgt_out, images=imgs, sys_y_loss_weight=0.0, staff_seg_loss_weight=0.0)
            ldict["loss"].backward()
            opt.step()

            eval_res = _eval_val_ce_and_mtp(model, [(imgs, seq)], device, use_bf16=False, is_chained=is_chained)
            deq_model, int4_disk = _pack_and_dequant_hybrid_int4(
                model, arm["mode"], cfg, Path(tmpd) / f"{arm['name']}.bin"
            )
            int4_res = _eval_val_ce_and_mtp(deq_model, [(imgs, seq)], device, use_bf16=False, is_chained=is_chained)
            size_stats = _compute_packed_int4_stats(model)
            assert math.isfinite(float(ldict["loss"].item())), f"Non-finite loss in {arm['name']}"
            assert math.isfinite(float(eval_res["val_ce"])), f"Non-finite val_ce in {arm['name']}"
            assert math.isfinite(float(int4_res["val_ce"])), f"Non-finite int4 val_ce in {arm['name']}"
            results[arm["name"]] = {
                "opt_mode": opt.mode,
                "loss": round(float(ldict["loss"].item()), 4),
                "val_ce": eval_res["val_ce"],
                "int4_val_ce": int4_res["val_ce"],
                "mtp_k1_k4": [eval_res[f"mtp_k{k}_acc"] for k in (1, 2, 3, 4)],
                "spec_accept_len": eval_res["spec_accept_len"],
                **size_stats,
                **int4_disk,
            }
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=f"Runner for {EXP_ID}")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-step tiny CPU smoke test and exit.")
    cli_args = parser.parse_args(argv)

    mvd_dir = _locate_or_unpack_mvd()
    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))

    if cli_args.smoke_test:
        res = run_cpu_smoke_test()
        print(f"[{EXP_ID}] CPU smoke test PASSED: {json.dumps(res)}", flush=True)
        return 0

    _configure_backend_safety()
    t_start = time.perf_counter()
    started_at = _utc_now()
    hf_token: Optional[str] = None
    arm_results: List[Dict[str, Any]] = []
    status = "RUNNING"
    error_msg: Optional[str] = None

    try:
        hf_token = _read_hf_token()
        _ensure_deps()

        import torch
        from uni_omr_v2 import PAD_ID

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda"
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        print(f"[{EXP_ID}] Device={gpu_name} | torch={torch.__version__}", flush=True)

        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        _try_resume_from_hf(hf_token)

        train_manifest, val_manifest = _ensure_proxy_dataset(hf_token)
        train_rows = _load_manifest_rows(train_manifest)
        val_rows = _load_manifest_rows(val_manifest)
        val_samples = [_load_page_sample(ip, bps, FULLPAGE_MAX_DIM, MAX_SEQ_LEN) for ip, bps in val_rows]
        val_batches = [
            _collate_pages(val_samples[i : i + BATCH_SIZE], PAD_ID)
            for i in range(0, len(val_samples), BATCH_SIZE)
        ]

        for arm in ARMS:
            summary = _run_arm(arm, train_rows, val_batches, device, use_bf16=use_bf16, hf_token=hf_token)
            arm_results.append(summary)

            gate_pass = None
            if len(arm_results) == len(ARMS):
                c_arm = next(a for a in arm_results if a["arm"] == "ctrl")
                m_arm = next(a for a in arm_results if a["arm"] == "modegate_cmtp")
                gate_pass = bool(
                    c_arm.get("opt_mode") == "muon+adamw"
                    and m_arm.get("opt_mode") == "muon+adamw"
                    and (m_arm["ema_val_ce"] - c_arm["ema_val_ce"]) <= 0.010
                    and (m_arm["int4_dequant_val_ce"] - c_arm["int4_dequant_val_ce"]) <= 0.010
                    and (m_arm["spec_accept_len"] - c_arm["spec_accept_len"]) >= 0.10
                    and m_arm["packed_int4_with_mtp_mb"] <= HARD_BUDGET_MB
                    and m_arm["step_ms"] <= c_arm["step_ms"] * 1.10
                )

            eval_payload = {
                "exp_id": EXP_ID,
                "status": "RUNNING" if len(arm_results) < len(ARMS) else "COMPLETED",
                "started_at": started_at,
                "updated_at": _utc_now(),
                "gpu": gpu_name,
                "steps_per_arm": STEPS_PER_ARM,
                "seed": SEED,
                "gate_pass": gate_pass,
                "arms": arm_results,
            }
            eval_local = OUT_ROOT / "eval.json"
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
        best_arm = min(
            arm_results,
            key=lambda r: float(r.get("ema_val_ce") if r.get("ema_val_ce") is not None else 1e9),
            default=None,
        )
        run_payload = {
            "exp_id": EXP_ID,
            "status": status,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_sec": round(total_sec, 2),
            "n_arms_completed": len(arm_results),
            "best_arm": best_arm["arm"] if best_arm else None,
            "best_ema_val_ce": best_arm.get("ema_val_ce") if best_arm else None,
            "eval_path": HF_EVAL_PATH,
            "error": error_msg,
        }
        run_local = OUT_ROOT / "run.json"
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
