#!/usr/bin/env python3
"""Colab A100 / Local RTX 4080 SUPER runner — Iter 15 Exp i15-14: 3-Axis Structural Decoder RoPE.

Experiment ID: 2026-11-27T14-15-i15-14-3axis-structural-dec-rope
Controlled 2-arm weight-incompatible proxy on edge85m + ax0_gl6 (600 steps/arm, BF16, seed=42):
  - Arm 0 (ctrl):   edge85m + ax0_gl6 stock 1D sequential decoder RoPE (_CachedRoPE1D).
  - Arm 1 (rope3d): 3-Axis Structural Decoder RoPE (0.000 MB INT4): splits head_dim=64 into
                    (16, 16, 32) subspaces for (system_idx, measure_idx, intra_measure_pos)
                    derived 100% causally from prefix tokens (<sys> and barline tokens).
                    Bit-identical between parallel training and KV-cached generate().

Measures per arm:
  - raw_val_ce, ema_val_ce, int4_dequant_val_ce
  - step_ms, train_elapsed_sec, peak_vram_mb
  - exact serialized packed_int4_mb (hard < 55.0 MB budget)

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
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXP_ID = "2026-11-27T14-15-i15-14-3axis-structural-dec-rope"
EXP_CODE = "i15-14"
ARMS = ("ctrl", "rope3d")
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_CKPT_PREFIX = f"checkpoints/{EXP_ID}"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORK_ROOT = Path("/vol/i15_14") if Path("/vol").is_dir() else Path("/content/i15_14")

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

BARLINE_TOKENS = (
    "bar",
    "bar-light-heavy",
    "bar-repeat-left",
    "bar-repeat-right",
    "bar-double",
)


# ---------------------------------------------------------------------------
# Self-Contained Experimental Module: StructuralRoPE3D (0.000 MB INT4)
# ---------------------------------------------------------------------------

def _build_structural_rope3d_class() -> Any:
    """Factory returning StructuralRoPE3D using uni_omr token IDs."""
    import torch
    import torch.nn as nn

    class StructuralRoPE3D(nn.Module):
        """Causal 3-Axis Structural Decoder RoPE over (system_idx, measure_idx, intra_measure_pos).

        Splits head_dim into 3 even subspaces:
          - d_sys   (16 dims / 8 pairs for head_dim=64): system index (increments on <sys>)
          - d_meas  (16 dims / 8 pairs for head_dim=64): measure index within system (increments on barlines, resets on <sys>)
          - d_intra (32 dims / 16 pairs for head_dim=64): token position within measure (resets on <sys> or barlines)

        All buffers are non-persistent (0 extra parameters, 0.000 MB INT4).
        Vectorized O(T) parallel path matches incremental KV-cache decoding bit-for-bit.
        """

        def __init__(
            self,
            head_dim: int,
            vocab_size: int,
            sys_id: int,
            bar_ids: Tuple[int, ...],
            base_sys: float = 100.0,
            base_meas: float = 1000.0,
            base_intra: float = 10000.0,
        ) -> None:
            super().__init__()
            if head_dim < 6 or head_dim % 2 != 0:
                raise ValueError(f"StructuralRoPE3D requires even head_dim >= 6, got {head_dim}")
            self.head_dim = int(head_dim)
            self.sys_id = int(sys_id)

            d_sys = max(2, (self.head_dim // 4) // 2 * 2)
            d_meas = max(2, (self.head_dim // 4) // 2 * 2)
            d_intra = self.head_dim - d_sys - d_meas
            if d_intra < 2 or d_intra % 2 != 0:
                raise ValueError(f"Bad 3-axis split ({d_sys}, {d_meas}, {d_intra}) for head_dim={head_dim}")
            self.d_sys = d_sys
            self.d_meas = d_meas
            self.d_intra = d_intra

            inv_sys = 1.0 / (base_sys ** (torch.arange(0, d_sys, 2, dtype=torch.float32) / d_sys))
            inv_meas = 1.0 / (base_meas ** (torch.arange(0, d_meas, 2, dtype=torch.float32) / d_meas))
            inv_intra = 1.0 / (base_intra ** (torch.arange(0, d_intra, 2, dtype=torch.float32) / d_intra))
            self.register_buffer("inv_freq_sys", inv_sys, persistent=False)
            self.register_buffer("inv_freq_meas", inv_meas, persistent=False)
            self.register_buffer("inv_freq_intra", inv_intra, persistent=False)

            bar_lut = torch.zeros(int(vocab_size), dtype=torch.bool)
            for bid in bar_ids:
                if 0 <= int(bid) < int(vocab_size):
                    bar_lut[int(bid)] = True
            self.register_buffer("bar_lut", bar_lut, persistent=False)

        @torch.no_grad()
        def compute_coords(
            self,
            tgt_ids: torch.Tensor,
            prev_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            """Return ((sys_idx, meas_idx, intra_pos), (last_sys, last_meas, last_intra))."""
            B, T = tgt_ids.shape
            device = tgt_ids.device
            lut = self.bar_lut if self.bar_lut.device == device else self.bar_lut.to(device)
            clamped = tgt_ids.clamp(0, lut.numel() - 1)
            is_sys = tgt_ids == self.sys_id
            is_bar = lut[clamped] & (~is_sys)

            if prev_state is None:
                # Vectorized O(T) causal computation (0 Python loops)
                sys_idx = is_sys.to(torch.long).cumsum(dim=1)
                global_bar = is_bar.to(torch.long).cumsum(dim=1)
                bar_at_sys = torch.where(is_sys, global_bar, torch.zeros_like(global_bar))
                last_sys_bar = torch.cummax(bar_at_sys, dim=1).values
                meas_idx = global_bar - last_sys_bar

                is_reset = is_sys | is_bar
                pos = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, T)
                reset_pos = torch.where(is_reset, pos, torch.zeros_like(pos))
                last_reset_pos = torch.cummax(reset_pos, dim=1).values
                intra_pos = pos - last_reset_pos

                new_state = (sys_idx[:, -1], meas_idx[:, -1], intra_pos[:, -1])
                return (sys_idx, meas_idx, intra_pos), new_state

            # Incremental KV-cache path (bit-identical to parallel path)
            cur_sys, cur_meas, cur_intra = prev_state
            cur_sys = cur_sys.to(device)
            cur_meas = cur_meas.to(device)
            cur_intra = cur_intra.to(device)
            zeros = torch.zeros_like(cur_sys)

            if T == 1:
                s0 = is_sys[:, 0]
                b0 = is_bar[:, 0]
                cur_sys = cur_sys + s0.to(torch.long)
                cur_meas = torch.where(s0, zeros, cur_meas + b0.to(torch.long))
                cur_intra = torch.where(s0 | b0, zeros, cur_intra + 1)
                return (cur_sys.unsqueeze(1), cur_meas.unsqueeze(1), cur_intra.unsqueeze(1)), (
                    cur_sys,
                    cur_meas,
                    cur_intra,
                )

            sys_list: List[torch.Tensor] = []
            meas_list: List[torch.Tensor] = []
            intra_list: List[torch.Tensor] = []
            for t in range(T):
                st = is_sys[:, t]
                bt = is_bar[:, t]
                cur_sys = cur_sys + st.to(torch.long)
                cur_meas = torch.where(st, zeros, cur_meas + bt.to(torch.long))
                cur_intra = torch.where(st | bt, zeros, cur_intra + 1)
                sys_list.append(cur_sys)
                meas_list.append(cur_meas)
                intra_list.append(cur_intra)
            sys_idx = torch.stack(sys_list, dim=1)
            meas_idx = torch.stack(meas_list, dim=1)
            intra_pos = torch.stack(intra_list, dim=1)
            return (sys_idx, meas_idx, intra_pos), (cur_sys, cur_meas, cur_intra)

        def forward(
            self,
            tgt_ids: torch.Tensor,
            dtype: torch.dtype,
            prev_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            """Return ``(cos, sin, new_state)`` where cos/sin have shape ``(B, 1, T, head_dim)``."""
            (sys_idx, meas_idx, intra_pos), new_state = self.compute_coords(tgt_ids, prev_state=prev_state)
            device = tgt_ids.device
            inv_s = self.inv_freq_sys if self.inv_freq_sys.device == device else self.inv_freq_sys.to(device)
            inv_m = self.inv_freq_meas if self.inv_freq_meas.device == device else self.inv_freq_meas.to(device)
            inv_i = self.inv_freq_intra if self.inv_freq_intra.device == device else self.inv_freq_intra.to(device)

            f_sys = sys_idx.to(torch.float32).unsqueeze(-1) * inv_s
            f_meas = meas_idx.to(torch.float32).unsqueeze(-1) * inv_m
            f_intra = intra_pos.to(torch.float32).unsqueeze(-1) * inv_i

            freqs = torch.cat([f_sys, f_meas, f_intra], dim=-1)  # (B, T, head_dim // 2)
            cos_h = freqs.cos().to(dtype)
            sin_h = freqs.sin().to(dtype)
            cos = torch.cat([cos_h, cos_h], dim=-1).unsqueeze(1)  # (B, 1, T, head_dim)
            sin = torch.cat([sin_h, sin_h], dim=-1).unsqueeze(1)  # (B, 1, T, head_dim)
            return cos, sin, new_state

    return StructuralRoPE3D


def apply_structural_rope3d(model: Any) -> Any:
    """Attach StructuralRoPE3D and patch decode_tokens to use 3-axis structural RoPE."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as _grad_ckpt
    import data as mvd_data
    import uni_omr
    import uni_omr_v2

    StructuralRoPE3D = _build_structural_rope3d_class()
    c = model.cfg
    head_dim = c.d_model // c.n_heads
    bar_ids = tuple(
        uni_omr.UNI_TOKEN_TO_ID[t] for t in BARLINE_TOKENS if t in uni_omr.UNI_TOKEN_TO_ID
    )
    model.rope3d = StructuralRoPE3D(
        head_dim=head_dim,
        vocab_size=c.vocab_size,
        sys_id=uni_omr.SYS_ID,
        bar_ids=bar_ids,
    )
    model._rope3d_state = None

    def _decode_tokens_rope3d(
        self: Any,
        tgt_ids: torch.Tensor,
        memory: Optional[Any],
        return_mtp: Optional[bool] = None,
        return_aux: Optional[bool] = None,
        past_kvs: Optional[List[Any]] = None,
        use_cache: bool = False,
        apply_tok_dropout: bool = True,
    ) -> Any:
        inp_ids = tgt_ids
        if self.training and apply_tok_dropout and self.cfg.tok_dropout > 0.0 and not use_cache:
            drop_mask = (
                (torch.rand_like(inp_ids, dtype=torch.float32) < self.cfg.tok_dropout)
                & (inp_ids != uni_omr.PAD_ID)
                & (inp_ids != uni_omr.BOS_ID)
                & (inp_ids != uni_omr.EOS_ID)
            )
            rand_toks = torch.randint(0, len(mvd_data.VOCAB), inp_ids.shape, device=inp_ids.device, dtype=inp_ids.dtype)
            inp_ids = torch.where(drop_mask, rand_toks, inp_ids)

        x = self.tok_emb(inp_ids)
        compute_dtype = uni_omr_v2._active_compute_dtype(x)
        if x.dtype != compute_dtype:
            x = x.to(dtype=compute_dtype)

        past_len = 0
        if past_kvs is not None and len(past_kvs) > 0 and past_kvs[0] is not None and past_kvs[0][0] is not None:
            past_len = int(past_kvs[0][0][0].shape[2])

        prev_state = self._rope3d_state if (use_cache and past_len > 0) else None
        cos3d, sin3d, new_rope_state = self.rope3d(tgt_ids, dtype=compute_dtype, prev_state=prev_state)
        if use_cache:
            self._rope3d_state = new_rope_state

        mem_final: Optional[torch.Tensor] = None
        level_mems: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if memory is not None:
            if isinstance(memory, (tuple, list)):
                mem_post_merger, mem_global_mid, mem_final = memory
                level_mems = (mem_post_merger, mem_global_mid)
            else:
                mem_final = memory

        bt = int(x.shape[0] * x.shape[1])
        can_dec_ckpt = self.training and self.cfg.grad_checkpoint and torch.is_grad_enabled()
        use_dec_full_ckpt = can_dec_ckpt and (bt > 8192)
        use_dec_ffn_ckpt = can_dec_ckpt and (4096 < bt <= 8192)
        new_past_kvs: List[Any] = []
        for idx, layer in enumerate(self.dec_layers):
            layer_past = past_kvs[idx] if past_kvs is not None else None
            need_cross_kv = (not use_cache) or (layer_past is None) or (layer_past[1] is None)
            if (not need_cross_kv) or (mem_final is None):
                layer_mem = None
            elif level_mems is not None and idx < len(self.deepstack_fusions):
                layer_mem = self.deepstack_fusions[idx](mem_final, level_mems[idx])
            else:
                layer_mem = mem_final

            if use_cache:
                x, layer_kv = layer(
                    x, memory=layer_mem, rope_cos=cos3d, rope_sin=sin3d,
                    past_kv=layer_past, use_cache=True,
                )
                new_past_kvs.append(layer_kv)
            elif use_dec_full_ckpt:
                x = _grad_ckpt(
                    layer, x, layer_mem, cos3d, sin3d,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    x, memory=layer_mem, rope_cos=cos3d, rope_sin=sin3d,
                    checkpoint_ffn=use_dec_ffn_ckpt,
                )
        h = self.dec_norm(x)

        want_mtp = self.training if return_mtp is None else bool(return_mtp)
        if want_mtp and len(self.mtp_heads) > 0:
            can_fuse_mtp = (
                uni_omr_v2._HAS_F_RMS_NORM
                and (not torch.onnx.is_in_onnx_export())
                and all(
                    (head.proj.bias is None) and (not uni_omr_v2._is_qat_linear_active(head.proj))
                    for head in self.mtp_heads
                )
            )
            if can_fuse_mtp:
                eps0 = float(self.mtp_heads[0].norm.eps)
                if all(float(head.norm.eps) == eps0 for head in self.mtp_heads):
                    B_h, T_h, D_h = h.shape
                    if h.dtype in (torch.bfloat16, torch.float16):
                        with torch.amp.autocast(device_type=h.device.type, enabled=False):
                            h_unit = F.rms_norm(h, (D_h,), None, eps0)
                    else:
                        h_unit = F.rms_norm(h, (D_h,), None, eps0)
                    w_mtp = torch.cat(
                        [head.proj.weight * head.norm.weight.unsqueeze(0) for head in self.mtp_heads],
                        dim=0,
                    )
                    mtp_h = h.unsqueeze(0) + F.silu(
                        F.linear(h_unit, w_mtp)
                        .view(B_h, T_h, len(self.mtp_heads), D_h)
                        .permute(2, 0, 1, 3)
                    )
                    all_h = torch.cat((h.unsqueeze(0), mtp_h), dim=0)
                else:
                    all_h = torch.stack([h, *(head(h) for head in self.mtp_heads)], dim=0)
            else:
                all_h = torch.stack([h, *(head(h) for head in self.mtp_heads)], dim=0)
            all_logits = self.lm_head(all_h)
            logits = all_logits[0]
            mtp_logits: List[torch.Tensor] = list(all_logits[1:].unbind(0))
        else:
            logits = self.lm_head(h)
            mtp_logits = []

        want_aux = self.training if return_aux is None else bool(return_aux)
        sys_y_pred: Optional[torch.Tensor] = None
        if want_aux and self.sys_y_head is not None and not use_cache:
            sys_y_pred = self.sys_y_head(h)

        if use_cache:
            return logits, mtp_logits, new_past_kvs
        return logits, mtp_logits, sys_y_pred

    model.decode_tokens = types.MethodType(_decode_tokens_rope3d, model)
    return model


def build_arm_model(arm_name: str, grad_checkpoint: bool = True, tiny: bool = False) -> Any:
    """Build edge85m + ax0_gl6 (or tiny d=64 for smoke test) model for 'ctrl' or 'rope3d'."""
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
    if arm_name == "rope3d":
        apply_structural_rope3d(model)
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
    """Download any already-completed arm results from HF Private Hub into local cache."""
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


def _pack_and_verify_hybrid_int4(
    model: Any,
    arm_name: str,
    out_bin: Path,
    group_size: int = INT4_GROUP_SIZE,
    tiny: bool = False,
) -> Dict[str, Any]:
    """Strip aux/MTP heads, pack QAT-INT4 Linear weights into 4-bit nibbles + FP16 scales,
    write exact binary file to disk, and dequantize onto a fresh model copy."""
    import torch
    import torch.nn as nn

    infer_model = build_arm_model(arm_name, grad_checkpoint=False, tiny=tiny).cpu().eval()
    infer_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True)
    infer_model.strip_aux_heads()
    infer_model.mtp_heads = nn.ModuleList()

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
        print(f"[{EXP_ID}] Reusing cached result for {arm_name}", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))

    model = build_arm_model(arm_name, grad_checkpoint=True).to(device)
    psummary = model.param_summary(int4_block_size=INT4_GROUP_SIZE)
    qat = uni_omr_v2.QATInt4ControllerV2(model, block_size=INT4_GROUP_SIZE)
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
        for _ in range(GRAD_ACCUM):
            batch_samples = rng.choices(train_pairs, weights=weights, k=BATCH_SIZE)
            collated = _collate_microbatch(batch_samples, device)
            if collated is None:
                continue
            inp, tgt = collated
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(inp, tgt[:, :-1], return_aux=False, return_mtp=True)
                losses = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)
                loss = losses["loss"] / GRAD_ACCUM
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
            mean_l = sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:]))
            mean_ms = sum(step_ms_hist[-50:]) / max(1, len(step_ms_hist[-50:]))
            print(
                f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} "
                f"loss={mean_l:.4f} step_ms={mean_ms:.1f} elapsed={time.perf_counter() - t_arm0:.1f}s",
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
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )

    qat.disable()
    raw_val_ce = _eval_val_ce(model, val_pairs, device, use_bf16)

    model.load_state_dict(ema_state, strict=True)
    ema_val_ce = _eval_val_ce(model, val_pairs, device, use_bf16)

    packed_info = _pack_and_verify_hybrid_int4(
        model, arm_name, arm_dir / f"{arm_name}_hybrid_int4.bin", INT4_GROUP_SIZE
    )
    dq_model = packed_info.pop("dequant_model").to(device)
    int4_dequant_val_ce = _eval_val_ce(dq_model, val_pairs, device, use_bf16)

    steady_ms = step_ms_hist[10:] if len(step_ms_hist) > 10 else step_ms_hist
    result = {
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
# CPU 2-Step Smoke Test + Causality & KV-Cache Parity Verification
# ---------------------------------------------------------------------------

def run_smoke_test() -> int:
    """2-step tiny CPU forward+backward+step + causality + KV-cache parity + edge85m INT4 check."""
    _locate_mvd()
    import tempfile
    import torch
    import train_uni_v2
    import uni_omr

    device = torch.device("cpu")
    _seed_everything(SEED)

    bar_id = uni_omr.UNI_TOKEN_TO_ID["bar"]
    sys_id = uni_omr.SYS_ID
    note_id = uni_omr.UNI_TOKEN_TO_ID["quarter"]

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
                tgt = torch.tensor(
                    [[uni_omr.BOS_ID, note_id, bar_id, note_id, bar_id, sys_id, note_id, bar_id, note_id, note_id, bar_id, sys_id, note_id, uni_omr.EOS_ID]],
                    dtype=torch.long,
                    device=device,
                )
                out = model(imgs, tgt[:, :-1], return_aux=False, return_mtp=True)
                loss = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)["loss"]
                assert torch.isfinite(loss), f"{arm_name} step {step}: non-finite loss {loss}"
                loss.backward()
                assert out["logits"].shape == (1, 13, uni_omr.VOCAB_SIZE)
                opt.step()
                losses_seen.append(round(float(loss.item()), 4))

            model.eval()
            if arm_name == "rope3d":
                # 1. Exact coordinate check on [BOS, n, bar, n, bar, <sys>, n, bar, n]
                probe = torch.tensor(
                    [[uni_omr.BOS_ID, note_id, bar_id, note_id, bar_id, sys_id, note_id, bar_id, note_id]],
                    dtype=torch.long,
                    device=device,
                )
                (s_par, m_par, i_par), _ = model.rope3d.compute_coords(probe)
                assert s_par[0].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1], f"bad sys_idx: {s_par[0].tolist()}"
                assert m_par[0].tolist() == [0, 0, 1, 1, 2, 0, 0, 1, 1], f"bad meas_idx: {m_par[0].tolist()}"
                assert i_par[0].tolist() == [0, 1, 0, 1, 0, 0, 1, 0, 1], f"bad intra_pos: {i_par[0].tolist()}"

                # 2. Strict causality check: mutating tokens at t >= 5 must not affect logits[:, :5]
                probe_mut = probe.clone()
                probe_mut[0, 5:] = torch.tensor([note_id, note_id, sys_id, bar_id], dtype=torch.long)
                with torch.no_grad():
                    mem, _, _ = model.encode_image(imgs, return_aux=False)
                    l_a, _, _ = model.decode_tokens(probe, mem, return_mtp=False, return_aux=False, apply_tok_dropout=False)
                    l_b, _, _ = model.decode_tokens(probe_mut, mem, return_mtp=False, return_aux=False, apply_tok_dropout=False)
                causal_diff = float((l_a[:, :5, :] - l_b[:, :5, :]).abs().max().item())
                assert causal_diff < 1e-6, f"CAUSALITY LEAK: diff={causal_diff}"

                # 3. Parallel vs 1-token-at-a-time KV-cache parity check
                with torch.no_grad():
                    past_kvs = model.precompute_cross_kv(mem)
                    step_logits: List[torch.Tensor] = []
                    for t in range(probe.shape[1]):
                        lt, _, past_kvs = model.decode_tokens(
                            probe[:, t : t + 1],
                            memory=None,
                            return_mtp=False,
                            return_aux=False,
                            past_kvs=past_kvs,
                            use_cache=True,
                            apply_tok_dropout=False,
                        )
                        step_logits.append(lt)
                    l_inc = torch.cat(step_logits, dim=1)
                kv_diff = float((l_a - l_inc).abs().max().item())
                assert kv_diff < 1e-4, f"KV-cache parity mismatch: max_diff={kv_diff}"
                tiny_results["rope3d_checks"] = {
                    "causal_max_diff": causal_diff,
                    "kv_cache_max_diff": round(kv_diff, 7),
                }

            pack_tiny = _pack_and_verify_hybrid_int4(
                model, arm_name, Path(tmpd) / f"{arm_name}_tiny.bin", INT4_GROUP_SIZE, tiny=True
            )
            dq_model = pack_tiny.pop("dequant_model")
            with torch.no_grad():
                seq = dq_model.generate(imgs, max_len=4)
            assert seq.shape[0] == 1 and seq.shape[1] >= 2

            tiny_results[arm_name] = {
                "opt_mode": opt.mode,
                "losses_2step": losses_seen,
                "tiny_params": sum(p.numel() for p in model.parameters()),
            }

        # Exact edge85m packed_int4_mb verification
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

    delta_mb = round(edge_mb["rope3d"] - edge_mb["ctrl"], 3)
    assert delta_mb == 0.0, f"Expected 0.000 MB delta for rope3d, got {delta_mb}"
    print(f"[{EXP_CODE}][SMOKE] tiny ctrl:     {tiny_results['ctrl']}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] tiny rope3d:   {tiny_results['rope3d']}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] rope3d checks: {tiny_results['rope3d_checks']}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] edge85m packed_int4_mb: ctrl={edge_mb['ctrl']} MB, rope3d={edge_mb['rope3d']} MB (delta={delta_mb:+.3f} MB)", flush=True)
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

        # Startup auto-resume from HF Private Hub
        arm_results.update(_try_resume_from_hf(hf_token))

        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(f"[{EXP_ID}] Loaded {len(train_pairs)} train / {len(val_pairs)} val", flush=True)

        for arm_name in ARMS:
            res = _train_arm(
                arm_name,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                device=device,
                use_bf16=use_bf16,
            )
            arm_results[arm_name] = res

            # 1-arm HF partial checkpoint sync
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
            if "ctrl" in arm_results and "rope3d" in arm_results:
                c = arm_results["ctrl"]
                t = arm_results["rope3d"]
                eval_payload["delta_rope3d_minus_ctrl"] = {
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
