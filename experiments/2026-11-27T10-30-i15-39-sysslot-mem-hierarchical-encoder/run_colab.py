#!/usr/bin/env python3
"""Colab H100 / Modal runner for Exp i15-39:
Competitive System Slot-Attention Hierarchical Memory Proxy.

Experiment ID: 2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder
Controlled 2-Arm Proxy (BF16, seed=42, 600 steps/arm, EMA=0.99, QAT-INT4 @ step>=420):
  - Arm 0 (`ctrl`):        edge85m + ax0_gl6 flat `mem_final` (L_k tokens).
  - Arm 1 (`sysslot_mem`): edge85m + ax0_gl6 + 2-iteration vertically-ordered competitive
                           System Slot-Attention (S=16 slots, d_attn=256, SwiGLU(512,512))
                           appended to `mem_final` (L_k + 16 tokens).

Self-contained: does NOT edit tmp-train/mvd/uni_omr_v2.py.
Uploads evals/<EXP_ID>.json and runs/<EXP_ID>.json to gregoreesmaa/sheet-music-converter-omr.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXP_ID = "2026-11-27T10-30-i15-39-sysslot-mem-hierarchical-encoder"
EXP_CODE = "i15-39"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

VOL_ROOT = Path("/vol") if Path("/vol").is_dir() else Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

SEED = 42
TRAIN_STEPS = 600
WARMUP_STEPS = 60
QAT_START_STEP = 420
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
GRAD_ACCUM = 4
LR_MAX = 3e-4
WEIGHT_DECAY = 0.04
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
EMA_DECAY = 0.99
INT4_GROUP_SIZE = 32
HARD_BUDGET_MB = 55.0
SAVE_EVERY_MINUTES = 5.0

N_SYS_SLOTS = 16
SLOT_ATTN_DIM = 256
SLOT_ITERS = 2
SLOT_FFN_DIM = 512


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and HF_TOKEN_FILE.exists():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        token = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var")
    return token


def _locate_or_unpack_mvd() -> Path:
    candidates = (
        Path.cwd() / "tmp-train" / "mvd",
        Path("/workspace/mvd"),
        Path("/content/mvd"),
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path("/content/workspace/sheet-music-converter/tmp-train/mvd"),
    )
    for cand in candidates:
        if (cand / "uni_omr_v2.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand

    for bundle in (Path("/content/bundle.tar.gz"), Path("/content/mvd_bundle.tar.gz")):
        if bundle.is_file():
            extract_dir = Path("/content/mvd_unpacked")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=extract_dir)
            for p in extract_dir.rglob("uni_omr_v2.py"):
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


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = VOL_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"
    if train_abs.is_file() and val_abs.is_file():
        return train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    tar_p = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PROXY_TAR,
        repo_type="dataset",
        token=hf_token,
        local_dir="/tmp/hf_dl",
    )
    with tarfile.open(tar_p, "r:*") as tf:
        tf.extractall(path=ds_dir)
    try:
        os.remove(tar_p)
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


# ---------------------------------------------------------------------------
# Self-Contained Experimental Architecture:
# 2-Iteration Vertically-Ordered Competitive System Slot-Attention (S=16)
# ---------------------------------------------------------------------------

def _build_sysslot_model(grad_checkpoint: bool = True) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import uni_omr
    import uni_omr_v2 as v2

    class SystemSlotAttention(nn.Module):
        """2-iteration vertically-ordered competitive Slot-Attention producing S=16 system slots.

        Each visual token competes across the S=16 slots via softmax over ``dim=1`` (slots),
        biased by a learnable vertical Gaussian prior centered at ``linspace(0.03, 0.97, S)``
        so slots specialize in strict top-to-bottom page order.
        """

        def __init__(
            self,
            d_model: int = 512,
            n_slots: int = N_SYS_SLOTS,
            d_attn: int = SLOT_ATTN_DIM,
            n_iters: int = SLOT_ITERS,
            ffn_hidden: int = SLOT_FFN_DIM,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.d_model = int(d_model)
            self.n_slots = int(n_slots)
            self.d_attn = int(d_attn)
            self.n_iters = int(n_iters)
            self.scale = float(d_attn) ** -0.5

            self.slot_init = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)
            self.register_buffer(
                "y_centers",
                torch.linspace(0.03, 0.97, steps=n_slots, dtype=torch.float32).view(1, n_slots, 1),
                persistent=False,
            )
            # softplus(2.0) ~= 2.13 -> gentle initial vertical band bias, fully learnable
            self.y_beta_raw = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))

            self.norm_inputs = v2.RMSNorm(d_model)
            self.norm_slots = v2.RMSNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_attn, bias=False)
            self.k_proj = nn.Linear(d_model, d_attn, bias=False)
            self.v_proj = nn.Linear(d_model, d_attn, bias=False)
            self.q_norm = v2.RMSNorm(d_attn)
            self.k_norm = v2.RMSNorm(d_attn)
            self.out_proj = nn.Linear(d_attn, d_model, bias=False)

            self.norm_ffn = v2.RMSNorm(d_model)
            self.ffn = uni_omr.SwiGLU(d_model, ffn_hidden, dropout=dropout)
            self.out_norm = v2.RMSNorm(d_model)

        def forward(
            self,
            mem_final: torch.Tensor,
            y_norm: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Args: ``mem_final`` ``(B, L_k, D)``, ``y_norm`` ``(B, L_k)`` in ``[0, 1]``.
            Returns: ``(B, S, D)`` vertically-ordered system slot tokens."""
            B, _, _ = mem_final.shape
            inp = self.norm_inputs(mem_final)
            k = self.k_norm(self.k_proj(inp))  # (B, L_k, d_attn)
            v = self.v_proj(inp)               # (B, L_k, d_attn)

            y_bias: Optional[torch.Tensor] = None
            if y_norm is not None:
                beta = F.softplus(self.y_beta_raw).to(torch.float32)
                y_centers = self.y_centers.to(device=mem_final.device, dtype=torch.float32)
                y_bias = -beta * (y_norm.float().unsqueeze(1) - y_centers).pow(2)  # (B, S, L_k)

            slots = self.slot_init.expand(B, -1, -1).to(mem_final.dtype)
            for _ in range(self.n_iters):
                slots_prev = slots
                q = self.q_norm(self.q_proj(self.norm_slots(slots)))  # (B, S, d_attn)
                logits = torch.bmm(q.float(), k.float().transpose(1, 2)) * self.scale  # (B, S, L_k)
                if y_bias is not None:
                    logits = logits + y_bias
                # Competitive softmax over slots (dim=1), then weighted mean over tokens (dim=-1)
                attn = F.softmax(logits, dim=1)
                attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                updates = self.out_proj(torch.bmm(attn.to(v.dtype), v))
                slots = slots_prev + updates
                slots = slots + self.ffn(self.norm_ffn(slots))

            return self.out_norm(slots)

    class SysSlotUniOMRv2Net(v2.UniOMRv2Net):
        """edge85m + ax0_gl6 with S=16 competitive System Slots appended to mem_final (L_k + 16)."""

        def __init__(self, cfg: v2.UniOMRv2Config) -> None:
            super().__init__(cfg)
            self.sys_slot_attn = SystemSlotAttention(
                d_model=cfg.d_model,
                n_slots=N_SYS_SLOTS,
                d_attn=SLOT_ATTN_DIM,
                n_iters=SLOT_ITERS,
                ffn_hidden=SLOT_FFN_DIM,
                dropout=cfg.dropout,
            )

        @staticmethod
        def _router_y_norm(router_info: Dict[str, Any], total_len: int, device: torch.device) -> torch.Tensor:
            kept_idx: torch.Tensor = router_info["kept_idx"]
            h_p, w_p = router_info["grid_hw"]
            n_bg = int(router_info.get("n_bg_summary", 0))
            B = kept_idx.shape[0]
            y_kept = torch.div(kept_idx, max(1, int(w_p)), rounding_mode="floor").float() / float(max(1, int(h_p) - 1))
            if n_bg > 0:
                y_bg = torch.full((B, n_bg), 0.5, device=device, dtype=torch.float32)
                y_all = torch.cat([y_kept, y_bg], dim=1)
            else:
                y_all = y_kept
            if y_all.shape[1] != total_len:
                y_all = torch.linspace(0.0, 1.0, steps=total_len, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
            return y_all

        def encode_image(
            self,
            images: torch.Tensor,
            keep_ratio_override: Optional[float] = None,
            return_aux: Optional[bool] = None,
        ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, Optional[torch.Tensor]], Dict[str, Any]]:
            mem_levels, aux_enc, router_info = super().encode_image(
                images, keep_ratio_override=keep_ratio_override, return_aux=return_aux
            )
            mem_post_merger, mem_global_mid, mem_final = mem_levels
            y_norm = self._router_y_norm(router_info, mem_final.shape[1], mem_final.device)
            slots = self.sys_slot_attn(mem_final, y_norm=y_norm)
            mem_final_aug = torch.cat([mem_final, slots], dim=1)  # (B, L_k + 16, D)
            router_info["n_sys_slots"] = N_SYS_SLOTS
            router_info["n_mem_with_slots"] = int(mem_final_aug.shape[1])
            return (mem_post_merger, mem_global_mid, mem_final_aug), aux_enc, router_info

        def _fuse_deepstack_with_slots(self, idx: int, mem_final_aug: torch.Tensor, mem_level: torch.Tensor) -> torch.Tensor:
            l_k = mem_level.shape[1]
            if mem_final_aug.shape[1] == l_k:
                return self.deepstack_fusions[idx](mem_final_aug, mem_level)
            fused_spatial = self.deepstack_fusions[idx](mem_final_aug[:, :l_k, :], mem_level)
            return torch.cat([fused_spatial, mem_final_aug[:, l_k:, :]], dim=1)

        def precompute_cross_kv(self, mem_levels: Any) -> List[Tuple[None, Tuple[torch.Tensor, torch.Tensor]]]:
            if isinstance(mem_levels, (tuple, list)):
                mem_post_merger, mem_global_mid, mem_final = mem_levels
                level_mems: Optional[Tuple[torch.Tensor, torch.Tensor]] = (mem_post_merger, mem_global_mid)
            else:
                mem_final = mem_levels
                level_mems = None

            past_kvs: List[Tuple[None, Tuple[torch.Tensor, torch.Tensor]]] = []
            for idx, layer in enumerate(self.dec_layers):
                if level_mems is not None and idx < len(self.deepstack_fusions):
                    layer_mem = self._fuse_deepstack_with_slots(idx, mem_final, level_mems[idx])
                else:
                    layer_mem = mem_final
                ca = layer.cross_attn
                B, Lk, _ = layer_mem.shape
                k = ca.k_norm(ca.k_proj(layer_mem).view(B, Lk, ca.n_kv_heads, ca.head_dim)).transpose(1, 2)
                v = ca.v_proj(layer_mem).view(B, Lk, ca.n_kv_heads, ca.head_dim).transpose(1, 2)
                past_kvs.append((None, (k, v)))
            return past_kvs

        def decode_tokens(
            self,
            tgt_ids: torch.Tensor,
            memory: Optional[Any],
            return_mtp: Optional[bool] = None,
            return_aux: Optional[bool] = None,
            past_kvs: Optional[List[Any]] = None,
            use_cache: bool = False,
            apply_tok_dropout: bool = True,
        ) -> Any:
            if isinstance(memory, (tuple, list)) and len(memory) == 3:
                mem_post_merger, mem_global_mid, mem_final_aug = memory
                l_k = mem_post_merger.shape[1]
                if mem_final_aug.shape[1] > l_k:
                    # Pad level_mems with zero-delta slot slice so DeepStackFusion preserves exact length L_k + 16
                    # while keeping slot tokens untouched via explicit fusion below.
                    return self._decode_tokens_with_slots(
                        tgt_ids,
                        mem_post_merger=mem_post_merger,
                        mem_global_mid=mem_global_mid,
                        mem_final_aug=mem_final_aug,
                        return_mtp=return_mtp,
                        return_aux=return_aux,
                        past_kvs=past_kvs,
                        use_cache=use_cache,
                    )
            return super().decode_tokens(
                tgt_ids,
                memory,
                return_mtp=return_mtp,
                return_aux=return_aux,
                past_kvs=past_kvs,
                use_cache=use_cache,
                apply_tok_dropout=apply_tok_dropout,
            )

        def _decode_tokens_with_slots(
            self,
            tgt_ids: torch.Tensor,
            *,
            mem_post_merger: torch.Tensor,
            mem_global_mid: torch.Tensor,
            mem_final_aug: torch.Tensor,
            return_mtp: Optional[bool],
            return_aux: Optional[bool],
            past_kvs: Optional[List[Any]],
            use_cache: bool,
        ) -> Any:
            _, T = tgt_ids.shape
            x = self.tok_emb(tgt_ids)
            compute_dtype = v2._active_compute_dtype(x)
            if x.dtype != compute_dtype:
                x = x.to(dtype=compute_dtype)

            past_len = 0
            if past_kvs is not None and len(past_kvs) > 0 and past_kvs[0] is not None and past_kvs[0][0] is not None:
                past_len = int(past_kvs[0][0][0].shape[2])

            cos_all, sin_all = self.rope1d(past_len + T, device=x.device, dtype=compute_dtype)
            cos1d = cos_all[:, :, past_len : past_len + T, :]
            sin1d = sin_all[:, :, past_len : past_len + T, :]

            level_mems = (mem_post_merger, mem_global_mid)
            new_past_kvs: List[Any] = []
            for idx, layer in enumerate(self.dec_layers):
                layer_past = past_kvs[idx] if past_kvs is not None else None
                need_cross_kv = (not use_cache) or (layer_past is None) or (layer_past[1] is None)
                if not need_cross_kv:
                    layer_mem = None
                elif idx < len(self.deepstack_fusions):
                    layer_mem = self._fuse_deepstack_with_slots(idx, mem_final_aug, level_mems[idx])
                else:
                    layer_mem = mem_final_aug

                if use_cache:
                    x, layer_kv = layer(
                        x, memory=layer_mem, rope_cos=cos1d, rope_sin=sin1d,
                        past_kv=layer_past, use_cache=True,
                    )
                    new_past_kvs.append(layer_kv)
                else:
                    x = layer(x, memory=layer_mem, rope_cos=cos1d, rope_sin=sin1d)

            h = self.dec_norm(x)
            want_mtp = self.training if return_mtp is None else bool(return_mtp)
            if want_mtp and len(self.mtp_heads) > 0:
                all_h = torch.stack([h, *(head(h) for head in self.mtp_heads)], dim=0)
                all_logits = self.lm_head(all_h)
                logits = all_logits[0]
                mtp_logits = list(all_logits[1:].unbind(0))
            else:
                logits = self.lm_head(h)
                mtp_logits = []

            want_aux = self.training if return_aux is None else bool(return_aux)
            sys_y_pred = self.sys_y_head(h) if (want_aux and self.sys_y_head is not None and not use_cache) else None
            if use_cache:
                return logits, mtp_logits, new_past_kvs
            return logits, mtp_logits, sys_y_pred

    cfg = v2.UniOMRv2Config.edge85m(
        n_axial_layers=0,
        n_global_layers=6,
        grad_checkpoint=grad_checkpoint,
    )
    return SysSlotUniOMRv2Net(cfg)


def _pack_and_verify_hybrid_int4(model: Any, out_bin: Path, group_size: int = INT4_GROUP_SIZE) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    infer_model = copy.deepcopy(model).cpu().eval()
    infer_model.strip_aux_heads(strip_mtp=True)

    int4_roots = [
        infer_model.encoder.axial_layers,
        infer_model.encoder.patch_merger,
        infer_model.encoder.global_layers,
        infer_model.deepstack_fusions,
        infer_model.dec_layers,
        infer_model.lm_head,
    ]
    if hasattr(infer_model, "sys_slot_attn") and infer_model.sys_slot_attn is not None:
        int4_roots.append(infer_model.sys_slot_attn)

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
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mem, _, _ = model.encode_image(inp, return_aux=False)
                logits, _, _ = model.decode_tokens(
                    tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID
                )
            losses.append(float(loss.item()))
    return round(sum(losses) / max(1, len(losses)), 5)


def _build_arm_model(arm_name: str, grad_checkpoint: bool = True) -> Any:
    import uni_omr_v2 as v2

    if arm_name == "sysslot_mem":
        return _build_sysslot_model(grad_checkpoint=grad_checkpoint)
    cfg = v2.UniOMRv2Config.edge85m(
        n_axial_layers=0,
        n_global_layers=6,
        grad_checkpoint=grad_checkpoint,
    )
    return v2.UniOMRv2Net(cfg)


def _train_arm(
    arm_name: str,
    *,
    train_pairs: List[Tuple[Path, List[Path], str, float]],
    val_pairs: List[Tuple[Path, List[Path], str, float]],
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    from PIL import Image
    import uni_omr_v2 as v2

    _seed_everything(SEED)
    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached = arm_dir / "arm_result.json"
    if cached.is_file():
        print(f"[{EXP_ID}] Reusing cached result for {arm_name}", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))

    model = _build_arm_model(arm_name, grad_checkpoint=True).to(device)
    psummary = model.param_summary(int4_block_size=INT4_GROUP_SIZE)
    qat = v2.QATInt4ControllerV2(model, block_size=INT4_GROUP_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    ema_state = copy.deepcopy(model.state_dict())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    weights = [p[3] for p in train_pairs]
    rng = random.Random(SEED)
    t_arm0 = time.time()
    t_last_save = time.time()
    loss_hist: List[float] = []
    step_times_ms: List[float] = []

    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        if step == QAT_START_STEP:
            qat.enable()
            print(f"[{EXP_ID}][{arm_name}] Enabled QATInt4ControllerV2 at step {step}", flush=True)

        mult = _wsd_mult(step, TRAIN_STEPS, WARMUP_STEPS)
        for pg in opt.param_groups:
            pg["lr"] = LR_MAX * mult

        t_s0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            img_p, beks, _, _ = rng.choices(train_pairs, weights=weights, k=1)[0]
            ids = _encode_bekerns_full(beks)
            if len(ids) < 3:
                continue
            img = Image.open(img_p).convert("RGB")
            inp = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
            tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(inp, tgt[:, :-1], return_aux=False, return_mtp=True)
                losses = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)
                loss = losses["loss"] / GRAD_ACCUM
            loss.backward()
            step_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times_ms.append((time.perf_counter() - t_s0) * 1000.0)

        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_state[k].mul_(EMA_DECAY).add_(v, alpha=1.0 - EMA_DECAY)
                else:
                    ema_state[k].copy_(v)

        loss_hist.append(step_loss)
        if step % 100 == 0 or step == 1:
            mean_l = sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:]))
            mean_ms = sum(step_times_ms[-50:]) / max(1, len(step_times_ms[-50:]))
            print(
                f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} loss={mean_l:.4f} "
                f"step_ms={mean_ms:.1f} elapsed={time.time()-t_arm0:.1f}s",
                flush=True,
            )

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ck = arm_dir / "latest.pt.tmp"
            torch.save({"step": step, "model_state_dict": model.state_dict(), "ema_state_dict": ema_state}, tmp_ck)
            os.replace(tmp_ck, arm_dir / "latest.pt")
            t_last_save = time.time()

    qat.disable()
    peak_vram_gb = (
        round(float(torch.cuda.max_memory_allocated()) / (1024.0 ** 3), 3) if device.type == "cuda" else 0.0
    )
    raw_val_ce = _eval_val_ce(model, val_pairs, device, use_bf16)

    model.load_state_dict(ema_state, strict=True)
    ema_val_ce = _eval_val_ce(model, val_pairs, device, use_bf16)

    packed_info = _pack_and_verify_hybrid_int4(model, arm_dir / f"{arm_name}_hybrid_int4.bin", INT4_GROUP_SIZE)
    dq_model = packed_info.pop("dequant_model").to(device)
    int4_val_ce = _eval_val_ce(dq_model, val_pairs, device, use_bf16)

    steady_ms = step_times_ms[min(10, len(step_times_ms) - 1) :]
    result = {
        "arm": arm_name,
        "param_summary": psummary,
        "packed_int4_verification": packed_info,
        "packed_int4_mb": packed_info["packed_int4_mb"],
        "step_ms": round(sum(steady_ms) / max(1, len(steady_ms)), 2),
        "vram_gb": peak_vram_gb,
        "train_loss_tail50": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5),
        "raw_val_ce": raw_val_ce,
        "ema_val_ce": ema_val_ce,
        "int4_dequant_val_ce": int4_val_ce,
        "elapsed_secs": round(time.time() - t_arm0, 2),
    }
    _atomic_write_json(cached, result)
    del model, dq_model, opt, ema_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _run_cpu_smoke_test() -> int:
    _locate_or_unpack_mvd()
    import tempfile
    import torch
    import uni_omr

    device = torch.device("cpu")
    print(f"[{EXP_ID}] Running 1-step tiny CPU forward+backward smoke test...", flush=True)
    for arm_name in ("ctrl", "sysslot_mem"):
        _seed_everything(SEED)
        model = _build_arm_model(arm_name, grad_checkpoint=False).to(device)
        ps = model.param_summary(int4_block_size=INT4_GROUP_SIZE)
        model.train()
        img = torch.randn(1, 3, 128, 128, device=device)
        tgt = torch.randint(0, 32, (1, 16), device=device)
        tgt[:, 0] = uni_omr.BOS_ID
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt.zero_grad(set_to_none=True)
        out = model(img, tgt[:, :-1], return_aux=False, return_mtp=True)
        losses = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)
        losses["loss"].backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            mem_levels, _, rinfo = model.encode_image(img, return_aux=False)
            if arm_name == "sysslot_mem":
                assert mem_levels[2].shape[1] == mem_levels[0].shape[1] + N_SYS_SLOTS, (
                    f"Expected L_k + {N_SYS_SLOTS}, got {mem_levels[2].shape[1]} vs {mem_levels[0].shape[1]}"
                )
            seq = model.generate(img, max_len=2)
            assert seq.shape[0] == 1 and seq.shape[1] >= 2
        with tempfile.TemporaryDirectory() as td:
            pack = _pack_and_verify_hybrid_int4(model, Path(td) / f"{arm_name}.bin", INT4_GROUP_SIZE)
            dq = pack.pop("dequant_model")
            del dq
        print(
            f"[{EXP_ID}][smoke][{arm_name}] OK | infer_params={ps['infer_params']/1e6:.2f}M "
            f"| packed_int4_mb={pack['packed_int4_mb']:.2f} MB | mem_len={mem_levels[2].shape[1]} "
            f"| loss={float(losses['loss'].item()):.4f}",
            flush=True,
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true", help="Run 1-step tiny CPU forward+backward smoke test")
    args = ap.parse_args()
    if args.smoke_test:
        return _run_cpu_smoke_test()

    t_start = time.time()
    started_iso = _utc_now()
    hf_token = _read_and_purge_hf_token()
    try:
        _locate_or_unpack_mvd()
        _ensure_deps()
        import torch
        from huggingface_hub import HfApi

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(f"[{EXP_ID}] Loaded {len(train_pairs)} train / {len(val_pairs)} val on {gpu_name}", flush=True)

        arm_ctrl = _train_arm("ctrl", train_pairs=train_pairs, val_pairs=val_pairs, device=device, use_bf16=use_bf16)
        arm_slot = _train_arm(
            "sysslot_mem", train_pairs=train_pairs, val_pairs=val_pairs, device=device, use_bf16=use_bf16
        )

        comparison = {
            "ema_val_ce_delta": round(arm_slot["ema_val_ce"] - arm_ctrl["ema_val_ce"], 5),
            "int4_val_ce_delta": round(arm_slot["int4_dequant_val_ce"] - arm_ctrl["int4_dequant_val_ce"], 5),
            "step_ms_ratio": round(arm_slot["step_ms"] / max(1e-6, arm_ctrl["step_ms"]), 3),
            "sysslot_packed_int4_mb": arm_slot["packed_int4_mb"],
            "sysslot_passes_55mb": arm_slot["packed_int4_verification"]["passes_55mb_budget"],
        }
        report = {
            "exp_id": EXP_ID,
            "exp_code": EXP_CODE,
            "status": "completed",
            "started_utc": started_iso,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"colab-or-modal-{gpu_name}",
            "bf16": use_bf16,
            "steps_per_arm": TRAIN_STEPS,
            "arms": {"ctrl": arm_ctrl, "sysslot_mem": arm_slot},
            "comparison_sysslot_minus_ctrl": comparison,
        }
        eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        run_local = VOL_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(
            run_local,
            {
                "exp_id": EXP_ID,
                "status": "completed",
                "started_utc": started_iso,
                "completed_utc": report["completed_utc"],
                "elapsed_secs": report["elapsed_secs"],
                "eval_path": HF_EVAL_PATH,
                "comparison": comparison,
            },
        )
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload i15-39 competitive System Slot-Attention proxy",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload i15-39 run ledger",
        )
        print(f"[{EXP_ID}] UPLOADED {HF_EVAL_PATH} & {HF_RUN_PATH} TO HF!", flush=True)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
