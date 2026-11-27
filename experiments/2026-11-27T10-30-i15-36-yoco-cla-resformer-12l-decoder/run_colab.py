#!/usr/bin/env python3
"""Colab H100 / Modal runner for Exp i15-36:
YOCO-3Bank Cross-KV + CLA-2:1 Self-KV + ResFormer 12L Decoder Proxy.

Experiment ID: 2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder
Controlled 2-Arm Proxy (BF16, seed=42, 600 steps/arm, EMA=0.99, QAT-INT4 @ step>=420):
  - Arm 0 (`ctrl`):   edge85m + ax0_gl6 (10L, d_ff=2048, GQA-2 independent KV).
  - Arm 1 (`yoco12`): edge85m + ax0_gl6 with 12L decoder (d_ff=1664),
                      YOCO-3Bank Cross-KV (n_kv=4, banks shared by 0..2, 3..6, 7..11),
                      CLA-2:1 Self-KV (n_kv=2, 6 banks),
                      ResFormer v_res (v_l = (1 - sig(a)) * v_l + sig(a) * v_0).

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

EXP_ID = "2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder"
EXP_CODE = "i15-36"
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

# Reference KV-cache geometry for apples-to-apples WebGPU FP16 KV-cache MB reporting
REF_LMEM = 3072
REF_TDEC = 2048


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
        local_dir=str(VOL_ROOT / "hf_dl"),
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
# YOCO-3Bank Cross-KV (n_kv=4) + CLA-2:1 Self-KV (n_kv=2) + ResFormer v_res
# ---------------------------------------------------------------------------

def _build_yoco_cla_resformer_model(grad_checkpoint: bool = True) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as _grad_ckpt
    import uni_omr
    import uni_omr_v2 as v2

    class SharedKVAttention(nn.Module):
        """GQA attention supporting producer (computes K/V + optional ResFormer v_res)
        and consumer (reuses shared K/V bank with zero K/V projection weights)."""

        def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_kv_heads: int,
            *,
            owns_kv: bool,
            use_rope: bool = True,
            use_v_res: bool = False,
            v_res_init: float = -2.0,
        ) -> None:
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads
            self.n_rep = n_heads // n_kv_heads
            self.head_dim = d_model // n_heads
            self.kv_dim = n_kv_heads * self.head_dim
            self.owns_kv = bool(owns_kv)
            self.use_rope = bool(use_rope)
            self.use_v_res = bool(use_v_res and owns_kv)

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.q_norm = v2.RMSNorm(self.head_dim)
            if self.owns_kv:
                self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
                self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
                self.k_norm = v2.RMSNorm(self.head_dim)
            else:
                self.k_proj = None
                self.v_proj = None
                self.k_norm = None
            if self.use_v_res:
                # ResFormer value residual: v_l = (1 - sigmoid(a)) * v_l + sigmoid(a) * v_0
                self.v_res_alpha = nn.Parameter(torch.full((1, n_kv_heads, 1, 1), float(v_res_init)))
            else:
                self.v_res_alpha = None
            self.o_proj = nn.Linear(d_model, d_model, bias=False)

        def compute_kv(
            self,
            kv_in: torch.Tensor,
            v0: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            assert self.owns_kv and self.k_proj is not None and self.v_proj is not None and self.k_norm is not None
            B, Lk, _ = kv_in.shape
            k = self.k_norm(self.k_proj(kv_in).view(B, Lk, self.n_kv_heads, self.head_dim)).transpose(1, 2)
            v = self.v_proj(kv_in).view(B, Lk, self.n_kv_heads, self.head_dim).transpose(1, 2)
            if self.use_v_res and v0 is not None and self.v_res_alpha is not None:
                g = torch.sigmoid(self.v_res_alpha).to(v.dtype)
                v = (1.0 - g) * v + g * v0
            return k, v

        def forward(
            self,
            q_in: torch.Tensor,
            kv_in: Optional[torch.Tensor] = None,
            *,
            shared_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            v0: Optional[torch.Tensor] = None,
            rope_cos: Optional[torch.Tensor] = None,
            rope_sin: Optional[torch.Tensor] = None,
            is_causal: bool = False,
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            is_cross: bool = False,
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
            """Returns ``(out, kv_used_for_sdpa, v_raw_cur)``."""
            B, Lq, _ = q_in.shape
            q = self.q_norm(self.q_proj(q_in).view(B, Lq, self.n_heads, self.head_dim)).transpose(1, 2)
            v_raw_cur: Optional[torch.Tensor] = None

            if shared_kv is not None:
                # Consumer layer: reuse bank K/V directly (apply RoPE to q if self-attn uses RoPE)
                if self.use_rope and rope_cos is not None and rope_sin is not None and not is_cross:
                    q, _ = uni_omr.apply_rope(q, q, rope_cos, rope_sin)
                k, v = shared_kv
            elif is_cross and past_kv is not None:
                k, v = past_kv
            else:
                assert kv_in is not None, "kv_in required for KV producer when past_kv is None"
                k, v = self.compute_kv(kv_in, v0=v0)
                v_raw_cur = v
                if self.use_rope and rope_cos is not None and rope_sin is not None and not is_cross:
                    q, k = uni_omr.apply_rope(q, k, rope_cos, rope_sin)
                if not is_cross and past_kv is not None:
                    k = torch.cat([past_kv[0], k], dim=2)
                    v = torch.cat([past_kv[1], v], dim=2)

            use_causal_flag = is_causal
            eff_mask = None
            if use_causal_flag and not is_cross and k.shape[2] > Lq:
                past_len = k.shape[2] - Lq
                if Lq == 1:
                    use_causal_flag = False
                else:
                    eff_mask = v2._get_causal_offset_mask(Lq, past_len, q.device)
                    use_causal_flag = False

            out = v2._sdpa(
                q, k, v, attn_mask=eff_mask, is_causal=use_causal_flag, enable_gqa=(self.n_rep > 1)
            )
            out = out.transpose(1, 2).reshape(B, Lq, self.d_model)
            out = self.o_proj(out)
            return out, (k, v), v_raw_cur

    class YocoClaDecoderLayer(nn.Module):
        def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_self_kv: int,
            n_cross_kv: int,
            d_ff: int,
            *,
            owns_self_kv: bool,
            owns_cross_kv: bool,
            use_self_v_res: bool,
            use_cross_v_res: bool,
            use_rope: bool,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.owns_self_kv = bool(owns_self_kv)
            self.owns_cross_kv = bool(owns_cross_kv)
            self.use_rope = bool(use_rope)
            self.norm1 = v2.RMSNorm(d_model)
            self.self_attn = SharedKVAttention(
                d_model,
                n_heads=n_heads,
                n_kv_heads=n_self_kv,
                owns_kv=owns_self_kv,
                use_rope=use_rope,
                use_v_res=use_self_v_res,
            )
            self.norm2 = v2.RMSNorm(d_model)
            self.cross_attn = SharedKVAttention(
                d_model,
                n_heads=n_heads,
                n_kv_heads=n_cross_kv,
                owns_kv=owns_cross_kv,
                use_rope=False,
                use_v_res=use_cross_v_res,
            )
            self.norm3 = v2.RMSNorm(d_model)
            self.ffn = uni_omr.SwiGLU(d_model, d_ff, dropout=dropout)
            self.drop = nn.Dropout(dropout)

        def forward(
            self,
            x: torch.Tensor,
            *,
            memory: Optional[torch.Tensor],
            rope_cos: Optional[torch.Tensor],
            rope_sin: Optional[torch.Tensor],
            shared_self_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            shared_cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            self_v0: Optional[torch.Tensor] = None,
            cross_v0: Optional[torch.Tensor] = None,
            past_self_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            past_cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
        ) -> Tuple[
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
        ]:
            h = self.norm1(x)
            r_cos = rope_cos if self.use_rope else None
            r_sin = rope_sin if self.use_rope else None
            sa_out, self_kv_out, self_v_cur = self.self_attn(
                h,
                h if self.owns_self_kv else None,
                shared_kv=None if self.owns_self_kv else shared_self_kv,
                v0=self_v0,
                rope_cos=r_cos,
                rope_sin=r_sin,
                is_causal=True,
                past_kv=past_self_kv if self.owns_self_kv else None,
                use_cache=use_cache,
                is_cross=False,
            )
            x = x + self.drop(sa_out)

            ca_out, cross_kv_out, cross_v_cur = self.cross_attn(
                self.norm2(x),
                memory if self.owns_cross_kv else None,
                shared_kv=None if self.owns_cross_kv else shared_cross_kv,
                v0=cross_v0,
                past_kv=past_cross_kv if self.owns_cross_kv else None,
                use_cache=use_cache,
                is_cross=True,
            )
            x = x + self.drop(ca_out)
            x = x + self.drop(self.ffn(self.norm3(x)))
            return x, self_kv_out, cross_kv_out, self_v_cur, cross_v_cur

    class YocoClaResFormerNet(v2.UniOMRv2Net):
        """12L YOCO-3Bank Cross-KV (n_kv=4) + CLA-2:1 Self-KV (n_kv=2) + ResFormer v_res."""

        # YOCO 3-Bank cross-KV assignment: bank 0 -> layers 0..2, bank 1 -> 3..6, bank 2 -> 7..11
        CROSS_BANK_PRODUCERS: Tuple[int, ...] = (0, 3, 7)
        CROSS_BANK_OF_LAYER: Tuple[int, ...] = (0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2)
        N_CROSS_KV_HEADS: int = 4
        N_SELF_KV_HEADS: int = 2

        def __init__(self, cfg: v2.UniOMRv2Config) -> None:
            super().__init__(cfg)
            assert cfg.n_dec_layers == 12, f"Expected 12 decoder layers, got {cfg.n_dec_layers}"
            d_ff = 1664
            layers = []
            for i in range(12):
                owns_self = (i % 2 == 0)
                owns_cross = i in self.CROSS_BANK_PRODUCERS
                # Keep RoPE/NoPE aligned per CLA-2:1 pair so Q and shared K share rotary space
                pair_idx = i // 2
                use_rope = (cfg.irope_nope_every <= 0) or (((pair_idx + 1) % max(1, cfg.irope_nope_every // 2)) != 0)
                layers.append(
                    YocoClaDecoderLayer(
                        d_model=cfg.d_model,
                        n_heads=cfg.n_heads,
                        n_self_kv=self.N_SELF_KV_HEADS,
                        n_cross_kv=self.N_CROSS_KV_HEADS,
                        d_ff=d_ff,
                        owns_self_kv=owns_self,
                        owns_cross_kv=owns_cross,
                        use_self_v_res=(owns_self and i > 0),
                        use_cross_v_res=(owns_cross and i > 0),
                        use_rope=use_rope,
                        dropout=cfg.dropout,
                    )
                )
            self.dec_layers = nn.ModuleList(layers)

        def precompute_cross_kv(self, mem_levels: Any) -> Dict[str, List[Any]]:
            """Precompute only the 3 YOCO Cross-KV banks (layers 0, 3, 7) with ResFormer v_res."""
            if isinstance(mem_levels, (tuple, list)):
                mem_post_merger, mem_global_mid, mem_final = mem_levels
                bank_mems = (
                    self.deepstack_fusions[0](mem_final, mem_post_merger),
                    self.deepstack_fusions[1](mem_final, mem_global_mid),
                    mem_final,
                )
            else:
                bank_mems = (mem_levels, mem_levels, mem_levels)

            cross_banks: List[Tuple[torch.Tensor, torch.Tensor]] = []
            cross_v0: Optional[torch.Tensor] = None
            for bank_idx, prod_idx in enumerate(self.CROSS_BANK_PRODUCERS):
                ca = self.dec_layers[prod_idx].cross_attn
                k, v = ca.compute_kv(bank_mems[bank_idx], v0=cross_v0)
                if bank_idx == 0:
                    cross_v0 = v
                cross_banks.append((k, v))
            return {
                "self_banks": [None] * 6,
                "cross_banks": cross_banks,
            }

        def decode_tokens(
            self,
            tgt_ids: torch.Tensor,
            memory: Optional[Any],
            return_mtp: Optional[bool] = None,
            return_aux: Optional[bool] = None,
            past_kvs: Optional[Any] = None,
            use_cache: bool = False,
            apply_tok_dropout: bool = True,
        ) -> Any:
            _, T = tgt_ids.shape
            x = self.tok_emb(tgt_ids)
            compute_dtype = v2._active_compute_dtype(x)
            if x.dtype != compute_dtype:
                x = x.to(dtype=compute_dtype)

            past_len = 0
            if isinstance(past_kvs, dict) and past_kvs["self_banks"][0] is not None:
                past_len = int(past_kvs["self_banks"][0][0].shape[2])

            cos_all, sin_all = self.rope1d(past_len + T, device=x.device, dtype=compute_dtype)
            cos1d = cos_all[:, :, past_len : past_len + T, :]
            sin1d = sin_all[:, :, past_len : past_len + T, :]

            # Prepare 3-bank memory inputs when not using precomputed cross banks
            bank_mems: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
            if memory is not None:
                if isinstance(memory, (tuple, list)):
                    mem_post_merger, mem_global_mid, mem_final = memory
                    bank_mems = (
                        self.deepstack_fusions[0](mem_final, mem_post_merger),
                        self.deepstack_fusions[1](mem_final, mem_global_mid),
                        mem_final,
                    )
                else:
                    bank_mems = (memory, memory, memory)

            self_banks_in = past_kvs["self_banks"] if isinstance(past_kvs, dict) else [None] * 6
            cross_banks_in = past_kvs["cross_banks"] if isinstance(past_kvs, dict) else [None] * 3

            new_self_banks: List[Any] = [None] * 6
            new_cross_banks: List[Any] = list(cross_banks_in)
            cur_self_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            cur_cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            self_v0: Optional[torch.Tensor] = None
            cross_v0: Optional[torch.Tensor] = None

            for idx, layer in enumerate(self.dec_layers):
                self_bank_idx = idx // 2
                cross_bank_idx = self.CROSS_BANK_OF_LAYER[idx]

                layer_mem = None
                past_c = cross_banks_in[cross_bank_idx]
                if layer.owns_cross_kv and past_c is None and bank_mems is not None:
                    layer_mem = bank_mems[cross_bank_idx]

                x, self_kv_out, cross_kv_out, self_v_cur, cross_v_cur = layer(
                    x,
                    memory=layer_mem,
                    rope_cos=cos1d,
                    rope_sin=sin1d,
                    shared_self_kv=None if layer.owns_self_kv else cur_self_kv,
                    shared_cross_kv=None if layer.owns_cross_kv else cur_cross_kv,
                    self_v0=self_v0,
                    cross_v0=cross_v0,
                    past_self_kv=self_banks_in[self_bank_idx] if layer.owns_self_kv else None,
                    past_cross_kv=past_c if layer.owns_cross_kv else None,
                    use_cache=use_cache,
                )

                if layer.owns_self_kv:
                    cur_self_kv = self_kv_out
                    new_self_banks[self_bank_idx] = self_kv_out
                    if idx == 0:
                        self_v0 = self_v_cur
                if layer.owns_cross_kv:
                    cur_cross_kv = cross_kv_out
                    new_cross_banks[cross_bank_idx] = cross_kv_out
                    if idx == 0:
                        cross_v0 = cross_v_cur

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
                return logits, mtp_logits, {"self_banks": new_self_banks, "cross_banks": new_cross_banks}
            return logits, mtp_logits, sys_y_pred

    cfg = v2.UniOMRv2Config.edge85m(
        n_axial_layers=0,
        n_global_layers=6,
        n_dec_layers=12,
        ffn_mult=1664.0 / 512.0,
        grad_checkpoint=grad_checkpoint,
    )
    return YocoClaResFormerNet(cfg)


def _compute_kv_cache_mb(arm_name: str, head_dim: int = 64, l_mem: int = REF_LMEM, t_dec: int = REF_TDEC) -> Dict[str, float]:
    """Exact FP16 (2 bytes/elem) KV-cache footprint at reference (L_mem=3072, T_dec=2048)."""
    if arm_name == "yoco12":
        n_self_banks, n_self_kv = 6, 2
        n_cross_banks, n_cross_kv = 3, 4
    else:
        n_self_banks, n_self_kv = 10, 2
        n_cross_banks, n_cross_kv = 10, 2
    self_bytes = n_self_banks * 2 * n_self_kv * t_dec * head_dim * 2
    cross_bytes = n_cross_banks * 2 * n_cross_kv * l_mem * head_dim * 2
    total_bytes = self_bytes + cross_bytes
    return {
        "self_kv_mb": round(self_bytes / 1_000_000.0, 3),
        "cross_kv_mb": round(cross_bytes / 1_000_000.0, 3),
        "total_kv_mb": round(total_bytes / 1_000_000.0, 3),
        "ref_l_mem": l_mem,
        "ref_t_dec": t_dec,
    }


def _pack_and_verify_hybrid_int4(model: Any, out_bin: Path, group_size: int = INT4_GROUP_SIZE) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    infer_model = copy.deepcopy(model).cpu().eval()
    infer_model.strip_aux_heads(strip_mtp=True)

    int4_roots = (
        infer_model.encoder.axial_layers,
        infer_model.encoder.patch_merger,
        infer_model.encoder.global_layers,
        infer_model.deepstack_fusions,
        infer_model.dec_layers,
        infer_model.lm_head,
    )
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

    if arm_name == "yoco12":
        return _build_yoco_cla_resformer_model(grad_checkpoint=grad_checkpoint)
    cfg = v2.UniOMRv2Config.edge85m(
        n_axial_layers=0,
        n_global_layers=6,
        n_dec_layers=10,
        ffn_mult=4.0,
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
    kv_info = _compute_kv_cache_mb(arm_name)
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
        "kv_cache": kv_info,
        "packed_int4_verification": packed_info,
        "packed_int4_mb": packed_info["packed_int4_mb"],
        "kv_cache_mb": kv_info["total_kv_mb"],
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
    for arm_name in ("ctrl", "yoco12"):
        _seed_everything(SEED)
        model = _build_arm_model(arm_name, grad_checkpoint=False).to(device)
        ps = model.param_summary(int4_block_size=INT4_GROUP_SIZE)
        kv_info = _compute_kv_cache_mb(arm_name)
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

        # Verify cached KV generation (2 steps) and Hybrid INT4 packing
        model.eval()
        with torch.no_grad():
            seq = model.generate(img, max_len=2)
            assert seq.shape[0] == 1 and seq.shape[1] >= 2
        with tempfile.TemporaryDirectory() as td:
            pack = _pack_and_verify_hybrid_int4(model, Path(td) / f"{arm_name}.bin", INT4_GROUP_SIZE)
            dq = pack.pop("dequant_model")
            del dq
        print(
            f"[{EXP_ID}][smoke][{arm_name}] OK | infer_params={ps['infer_params']/1e6:.2f}M "
            f"| packed_int4_mb={pack['packed_int4_mb']:.2f} MB | kv_cache_mb={kv_info['total_kv_mb']:.2f} MB "
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
        arm_yoco = _train_arm("yoco12", train_pairs=train_pairs, val_pairs=val_pairs, device=device, use_bf16=use_bf16)

        comparison = {
            "ema_val_ce_delta": round(arm_yoco["ema_val_ce"] - arm_ctrl["ema_val_ce"], 5),
            "int4_val_ce_delta": round(arm_yoco["int4_dequant_val_ce"] - arm_ctrl["int4_dequant_val_ce"], 5),
            "kv_cache_mb_delta": round(arm_yoco["kv_cache_mb"] - arm_ctrl["kv_cache_mb"], 3),
            "kv_cache_reduction_pct": round(100.0 * (1.0 - arm_yoco["kv_cache_mb"] / arm_ctrl["kv_cache_mb"]), 2),
            "step_ms_ratio": round(arm_yoco["step_ms"] / max(1e-6, arm_ctrl["step_ms"]), 3),
            "yoco12_packed_int4_mb": arm_yoco["packed_int4_mb"],
            "yoco12_passes_55mb": arm_yoco["packed_int4_verification"]["passes_55mb_budget"],
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
            "arms": {"ctrl": arm_ctrl, "yoco12": arm_yoco},
            "comparison_yoco12_minus_ctrl": comparison,
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
            commit_message=f"eval({EXP_ID}): upload i15-36 YOCO-3Bank + CLA-2:1 + ResFormer 12L proxy",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload i15-36 run ledger",
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
