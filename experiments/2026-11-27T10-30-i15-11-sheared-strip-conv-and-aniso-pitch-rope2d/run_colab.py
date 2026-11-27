#!/usr/bin/env python3
"""Google Colab H100 runner — Iter 15 Exp i15-11:
Sheared Strip Convs (stage3/stage4) + Pitch-Biased Anisotropic 2D-RoPE.

Experiment ID: 2026-11-27T10-30-i15-11-sheared-strip-conv-and-aniso-pitch-rope2d
Weight-incompatible 2-arm proxy on UniOMRv2 edge85m + ax0_gl6 (600 steps/arm, BF16,
Muon+AdamW, EMA, seed 42, iter10_5_proxy_2k + +/-6 deg tilt stress eval slice):
  - Arm 0 (ctrl): stock AnisotropicMusicBlock + isotropic AspectNormalizedRoPE2D.
  - Arm 1 (sheared_aniso_rope): replaces stage3/stage4 dw_horiz (1,9) with 3-angle
    gated (3,9) sheared strips (0, +/-5 deg) and dw_vert (9,1) with 3-angle gated
    (9,3) slanted strips (0, +/-8 deg) + Pitch-Biased Anisotropic 2D-RoPE
    (62.5% Y dims base_y=2000, coord_scale_y=256 vs 37.5% X dims base_x=10000,
    coord_scale_x=128).

Security:
  - Reads HF_TOKEN from /tmp/.hf_token (chmod 600) or env HF_TOKEN, never prints it.
  - Always deletes /tmp/.hf_token in finally.
  - Uploads evals/<exp_id>.json and runs/<exp_id>.json to gregoreesmaa/sheet-music-converter-omr.
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
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-27T10-30-i15-11-sheared-strip-conv-and-aniso-pitch-rope2d"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORK_ROOT = Path("/content/i15_11")
OUT_ROOT = WORK_ROOT / "out" / EXP_ID

STEPS_PER_ARM = int(os.environ.get("I15_11_STEPS", "600"))
SEED = 42
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 60
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
GPU_EMA_DECAY = 0.997
MTP_LOSS_WEIGHT = 0.30
SALIENCY_LOSS_WEIGHT = 0.10
SYS_COUNT_LOSS_WEIGHT = 0.15
STROKE_HOG_WEIGHT = 0.15
TILT_STRESS_DEG = 6.0
LOG_EVERY = 25

ARMS: List[Dict[str, Any]] = [
    {"name": "ctrl", "mode": "ctrl", "is_control": True},
    {"name": "sheared_aniso_rope", "mode": "sheared_aniso_rope", "is_control": False},
]


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
    if not tok:
        raise RuntimeError("HF token missing: provide /tmp/.hf_token or HF_TOKEN env var")
    return tok


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


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = WORK_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"
    if train_abs.is_file() and val_abs.is_file():
        return train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR} from {HF_REPO_ID}...", flush=True)
    tar_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PROXY_TAR,
        repo_type="dataset",
        token=hf_token,
        local_dir="/tmp/hf_dl",
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

    def _make_sheared_masks(
        orientation: str,
        angles_deg: Tuple[float, float, float],
    ) -> torch.Tensor:
        """Build exact bilinear-interpolated directional strip masks for 3 angles.

        - ``orientation == "horiz"``: kernel ``(3, 9)``, ``0, +5, -5`` deg.
        - ``orientation == "vert"``: kernel ``(9, 3)``, ``0, +8, -8`` deg.
        Returns float32 tensor of shape ``(3, 1, kh, kw)``.
        """
        if orientation == "horiz":
            kh, kw = 3, 9
            masks = torch.zeros((len(angles_deg), 1, kh, kw), dtype=torch.float32)
            for a_idx, deg in enumerate(angles_deg):
                tan_a = math.tan(math.radians(deg))
                for col in range(kw):
                    dx = float(col - 4)
                    y_f = 1.0 + dx * tan_a  # center row = 1
                    y_f = max(0.0, min(2.0, y_f))
                    y0 = int(math.floor(y_f))
                    y1 = min(2, y0 + 1)
                    frac = y_f - float(y0)
                    masks[a_idx, 0, y0, col] += 1.0 - frac
                    if y1 != y0:
                        masks[a_idx, 0, y1, col] += frac
                    else:
                        masks[a_idx, 0, y0, col] += frac
            return masks

        kh, kw = 9, 3
        masks = torch.zeros((len(angles_deg), 1, kh, kw), dtype=torch.float32)
        for a_idx, deg in enumerate(angles_deg):
            tan_a = math.tan(math.radians(deg))
            for row in range(kh):
                dy = float(row - 4)
                x_f = 1.0 + dy * tan_a  # center col = 1
                x_f = max(0.0, min(2.0, x_f))
                x0 = int(math.floor(x_f))
                x1 = min(2, x0 + 1)
                frac = x_f - float(x0)
                masks[a_idx, 0, row, x0] += 1.0 - frac
                if x1 != x0:
                    masks[a_idx, 0, row, x1] += frac
                else:
                    masks[a_idx, 0, row, x0] += frac
        return masks

    class ShearedStripConv2d(nn.Module):
        """3-angle gated depthwise sheared/slanted strip convolution.

        Uses a single grouped ``F.conv2d`` over ``3 * C`` masked directional kernels
        blended per spatial location by a lightweight ``1x1`` 3-way softmax gate.
        """

        def __init__(
            self,
            channels: int,
            orientation: str,
            angles_deg: Tuple[float, float, float],
        ) -> None:
            super().__init__()
            self.channels = int(channels)
            self.orientation = orientation
            self.angles_deg = tuple(float(a) for a in angles_deg)
            kh, kw = (3, 9) if orientation == "horiz" else (9, 3)
            self.padding = (kh // 2, kw // 2)

            self.weight = nn.Parameter(torch.empty(channels * 3, 1, kh, kw))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

            base_masks = _make_sheared_masks(orientation, self.angles_deg)  # (3, 1, kh, kw)
            tiled = base_masks.repeat(channels, 1, 1, 1)  # (3*C, 1, kh, kw)
            self.register_buffer("angle_mask", tiled, persistent=False)

            self.gate = nn.Conv2d(channels, 3, kernel_size=1, bias=True)
            nn.init.zeros_(self.gate.weight)
            # Slightly favor 0-deg upright strip at step 0 while keeping +/- angles active
            with torch.no_grad():
                self.gate.bias.copy_(torch.tensor([0.5, 0.0, 0.0]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            w_eff = self.weight * self.angle_mask.to(dtype=self.weight.dtype)
            y = F.conv2d(x, w_eff, bias=None, stride=1, padding=self.padding, groups=C)
            y = y.view(B, C, 3, H, W)
            g = F.softmax(self.gate(x).float(), dim=1).to(y.dtype).unsqueeze(1)  # (B, 1, 3, H, W)
            return (y * g).sum(dim=2)

    class PitchBiasedAnisoRoPE2D(nn.Module):
        """Pitch-Biased Anisotropic 2D-RoPE.

        Allocates ``62.5%`` of rotary dimensions to vertical pitch (Y) with finer
        frequency spectrum (``base_y=2000, coord_scale_y=256``) and ``37.5%`` to
        horizontal time (X) (``base_x=10000, coord_scale_x=128``). Drop-in
        replacement for ``AspectNormalizedRoPE2D``.
        """

        def __init__(
            self,
            head_dim: int,
            y_frac: float = 0.625,
            base_y: float = 2000.0,
            coord_scale_y: float = 256.0,
            base_x: float = 10000.0,
            coord_scale_x: float = 128.0,
            rope_jitter: float = 0.0,
        ) -> None:
            super().__init__()
            if head_dim % 4 != 0:
                raise ValueError(f"PitchBiasedAnisoRoPE2D requires head_dim % 4 == 0, got {head_dim}")
            half_dim = head_dim // 2
            n_y = max(1, min(half_dim - 1, int(round(half_dim * y_frac))))
            n_x = half_dim - n_y
            self.n_y = n_y
            self.n_x = n_x
            self.coord_scale_y = float(coord_scale_y)
            self.coord_scale_x = float(coord_scale_x)
            self.rope_jitter = float(max(0.0, rope_jitter))

            inv_freq_y = 1.0 / (base_y ** (torch.arange(0, n_y, dtype=torch.float32) / float(n_y)))
            inv_freq_x = 1.0 / (base_x ** (torch.arange(0, n_x, dtype=torch.float32) / float(n_x)))
            self.register_buffer("inv_freq_y", inv_freq_y, persistent=False)
            self.register_buffer("inv_freq_x", inv_freq_x, persistent=False)
            self._grid_cache: "OrderedDict[Tuple[int, int, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
            self._max_cache_size: int = 32

        def _get_grid_cos_sin(
            self,
            h_p: int,
            w_p: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            key = (h_p, w_p, device, dtype)
            cached = self._grid_cache.get(key)
            if cached is not None:
                self._grid_cache.move_to_end(key)
                return cached
            with torch.no_grad():
                inv_y = self.inv_freq_y.to(device=device, dtype=torch.float32)
                inv_x = self.inv_freq_x.to(device=device, dtype=torch.float32)
                ys = torch.arange(h_p, device=device, dtype=torch.float32) / float(max(1, h_p - 1))
                xs = torch.arange(w_p, device=device, dtype=torch.float32) / float(max(1, w_p - 1))
                fy = (ys * self.coord_scale_y).unsqueeze(-1) * inv_y  # (H, n_y)
                fx = (xs * self.coord_scale_x).unsqueeze(-1) * inv_x  # (W, n_x)
                fy_2d = fy[:, None, :].expand(h_p, w_p, -1)
                fx_2d = fx[None, :, :].expand(h_p, w_p, -1)
                emb_half = torch.cat([fy_2d, fx_2d], dim=-1).reshape(h_p * w_p, -1)
                cos_h = emb_half.cos().to(dtype)
                sin_h = emb_half.sin().to(dtype)
                cos_grid = torch.cat([cos_h, cos_h], dim=-1)
                sin_grid = torch.cat([sin_h, sin_h], dim=-1)

                cy = torch.tensor([0.5], device=device, dtype=torch.float32)
                cx = torch.tensor([0.5], device=device, dtype=torch.float32)
                emb_c = torch.cat(
                    [(cy * self.coord_scale_y).unsqueeze(-1) * inv_y, (cx * self.coord_scale_x).unsqueeze(-1) * inv_x],
                    dim=-1,
                )
                cos_ch = emb_c.cos().to(dtype)
                sin_ch = emb_c.sin().to(dtype)
                cos_center = torch.cat([cos_ch, cos_ch], dim=-1)
                sin_center = torch.cat([sin_ch, sin_ch], dim=-1)

                val = (cos_grid, sin_grid, cos_center, sin_center)
                self._grid_cache[key] = val
                self._grid_cache.move_to_end(key)
                while len(self._grid_cache) > self._max_cache_size:
                    self._grid_cache.popitem(last=False)
                return val

        def forward_indexed(
            self,
            kept_idx: torch.Tensor,
            grid_hw: Tuple[int, int],
            n_bg_summary: int,
            dtype: torch.dtype,
            coords: Optional[torch.Tensor] = None,
            is_full_grid: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if self.training and self.rope_jitter > 0.0:
                if coords is None:
                    raise ValueError("coords must be provided when rope_jitter > 0 during training")
                return self.forward(coords, grid_hw=grid_hw, dtype=dtype)
            h_p, w_p = int(grid_hw[0]), int(grid_hw[1])
            cos_grid, sin_grid, cos_center, sin_center = self._get_grid_cos_sin(h_p, w_p, kept_idx.device, dtype)
            B = kept_idx.shape[0]
            if is_full_grid and n_bg_summary == 0:
                cos = cos_grid.unsqueeze(0).expand(B, -1, -1)
                sin = sin_grid.unsqueeze(0).expand(B, -1, -1)
            else:
                cos = cos_grid[kept_idx]
                sin = sin_grid[kept_idx]
                if n_bg_summary > 0:
                    cos = torch.cat([cos, cos_center.expand(B, n_bg_summary, -1)], dim=1)
                    sin = torch.cat([sin, sin_center.expand(B, n_bg_summary, -1)], dim=1)
            return cos.unsqueeze(1), sin.unsqueeze(1)

        def forward(
            self,
            coords: torch.Tensor,
            grid_hw: Tuple[int, int],
            dtype: torch.dtype,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            h_p, w_p = int(grid_hw[0]), int(grid_hw[1])
            inv_y = self.inv_freq_y.to(device=coords.device, dtype=torch.float32)
            inv_x = self.inv_freq_x.to(device=coords.device, dtype=torch.float32)
            coords_f = coords if coords.dtype == torch.float32 else coords.to(torch.float32)
            y_norm = coords_f[..., 0] / float(max(1, h_p - 1))
            x_norm = coords_f[..., 1] / float(max(1, w_p - 1))
            if self.training and self.rope_jitter > 0.0:
                j = self.rope_jitter
                bcast_shape = (y_norm.shape[0],) + (1,) * (y_norm.ndim - 1) if y_norm.ndim >= 2 else ()
                scale_y = 1.0 + (torch.rand(bcast_shape, device=coords.device, dtype=torch.float32) * 2.0 - 1.0) * j
                scale_x = 1.0 + (torch.rand(bcast_shape, device=coords.device, dtype=torch.float32) * 2.0 - 1.0) * j
                shift_y = (torch.rand(bcast_shape, device=coords.device, dtype=torch.float32) * 2.0 - 1.0) * (0.5 * j)
                shift_x = (torch.rand(bcast_shape, device=coords.device, dtype=torch.float32) * 2.0 - 1.0) * (0.5 * j)
                y_norm = y_norm * scale_y + shift_y
                x_norm = x_norm * scale_x + shift_x
            fy = (y_norm * self.coord_scale_y).unsqueeze(-1) * inv_y
            fx = (x_norm * self.coord_scale_x).unsqueeze(-1) * inv_x
            emb_half = torch.cat([fy, fx], dim=-1)
            cos_h = emb_half.cos().to(dtype)
            sin_h = emb_half.sin().to(dtype)
            cos = torch.cat([cos_h, cos_h], dim=-1).unsqueeze(1)
            sin = torch.cat([sin_h, sin_h], dim=-1).unsqueeze(1)
            return cos, sin

    return {
        "ShearedStripConv2d": ShearedStripConv2d,
        "PitchBiasedAnisoRoPE2D": PitchBiasedAnisoRoPE2D,
    }


def _apply_sheared_and_aniso_rope(model: Any) -> Any:
    """In-place architecture patch for Arm 1 (keeps uni_omr_v2.py untouched)."""
    cls_map = _build_experimental_classes()
    ShearedStripConv2d = cls_map["ShearedStripConv2d"]
    PitchBiasedAnisoRoPE2D = cls_map["PitchBiasedAnisoRoPE2D"]

    for stage in (model.backbone.stage3, model.backbone.stage4):
        for blk in stage.blocks:
            c4 = int(blk.split_sizes[0])
            blk.dw_horiz = ShearedStripConv2d(c4, orientation="horiz", angles_deg=(0.0, 5.0, -5.0))
            blk.dw_vert = ShearedStripConv2d(c4, orientation="vert", angles_deg=(0.0, 8.0, -8.0))

    head_dim = int(model.cfg.d_model) // int(model.cfg.n_heads)
    model.encoder.rope2d = PitchBiasedAnisoRoPE2D(
        head_dim=head_dim,
        y_frac=0.625,
        base_y=2000.0,
        coord_scale_y=256.0,
        base_x=10000.0,
        coord_scale_x=128.0,
        rope_jitter=float(model.cfg.rope_jitter),
    )
    return model


def _build_arm_model(arm_mode: str, cfg: Any) -> Any:
    from uni_omr_v2 import UniOMRv2Net

    model = UniOMRv2Net(cfg)
    if arm_mode == "ctrl":
        return model
    if arm_mode == "sheared_aniso_rope":
        return _apply_sheared_and_aniso_rope(model)
    raise ValueError(f"Unknown arm_mode={arm_mode!r}")


# ---------------------------------------------------------------------------
# Self-Contained Data Loader, Tilt Transform, EMA, and Eval
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


def _tilt_batch_images(imgs: Any, deg: float = TILT_STRESS_DEG, batch_idx: int = 0) -> Any:
    """Apply deterministic alternating +deg / -deg aspect-preserving rotation via grid_sample."""
    import torch
    import torch.nn.functional as F

    B, _, H, W = imgs.shape
    thetas = []
    for i in range(B):
        sign = 1.0 if ((batch_idx * B + i) % 2 == 0) else -1.0
        rad = math.radians(sign * deg)
        c = math.cos(rad)
        s = math.sin(rad)
        # Aspect-ratio-corrected rotation in normalized [-1, 1] grid coordinates
        theta = torch.tensor(
            [[c, -s * (float(H) / float(W)), 0.0], [s * (float(W) / float(H)), c, 0.0]],
            device=imgs.device,
            dtype=torch.float32,
        )
        thetas.append(theta)
    theta_b = torch.stack(thetas, dim=0)
    grid = F.affine_grid(theta_b, size=imgs.size(), align_corners=False)
    return F.grid_sample(imgs.float(), grid, mode="bilinear", padding_mode="border", align_corners=False).to(imgs.dtype)


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


def _eval_upright_and_tilted_ce(
    model: Any,
    val_batches: Sequence[Tuple[Any, Any]],
    device: Any,
    use_bf16: bool,
    tilt_deg: float = TILT_STRESS_DEG,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from uni_omr_v2 import PAD_ID

    was_training = model.training
    model.eval()
    up_ce_sum = 0.0
    tilt_ce_sum = 0.0
    tok_count = 0

    with torch.no_grad():
        for b_idx, (imgs_cpu, seq_cpu) in enumerate(val_batches):
            dtype = torch.bfloat16 if (use_bf16 and device.type == "cuda") else torch.float32
            imgs = imgs_cpu.to(device=device, dtype=dtype)
            imgs_tilted = _tilt_batch_images(imgs, deg=tilt_deg, batch_idx=b_idx)
            seq = seq_cpu.to(device=device)
            tgt_in = seq[:, :-1]
            tgt_out = seq[:, 1:]
            n_val = int((tgt_out != PAD_ID).sum().item())
            if n_val == 0:
                continue

            with _amp_ctx(device, use_bf16):
                out_up = model(imgs, tgt_in, return_aux=False, return_mtp=False)
                out_tilt = model(imgs_tilted, tgt_in, return_aux=False, return_mtp=False)

            V = out_up["logits"].shape[-1]
            up_ce_sum += float(
                F.cross_entropy(out_up["logits"].float().reshape(-1, V), tgt_out.reshape(-1), ignore_index=PAD_ID, reduction="sum").item()
            )
            tilt_ce_sum += float(
                F.cross_entropy(out_tilt["logits"].float().reshape(-1, V), tgt_out.reshape(-1), ignore_index=PAD_ID, reduction="sum").item()
            )
            tok_count += n_val

    if was_training:
        model.train()

    up_ce = up_ce_sum / max(1, tok_count)
    tilt_ce = tilt_ce_sum / max(1, tok_count)
    return {
        "val_ce": round(up_ce, 5),
        "val_tilted_ce": round(tilt_ce, 5),
        "tilt_degradation_ce": round(tilt_ce - up_ce, 5),
        "val_tokens": tok_count,
    }


def _run_arm(
    arm: Dict[str, Any],
    train_rows: Sequence[Tuple[Path, List[Path]]],
    val_batches: Sequence[Tuple[Any, Any]],
    device: Any,
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    import train_uni_v2
    from uni_omr_v2 import EOS_ID, PAD_ID, SYS_ID, UniOMRv2Config

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
    model = _build_arm_model(arm["mode"], cfg).to(device)
    pinfo = model.param_summary(int4_block_size=32)
    packed_int4_mb = round(float(pinfo["hybrid_fp16_int4_with_scales_mb"]), 3)

    opt = train_uni_v2.build_optimizer_v2(
        model,
        optimizer_name="muon",
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device_type=device.type,
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
        f"\n[{EXP_ID}] === Arm {arm['name']} | infer_params={pinfo['infer_params']:,} "
        f"({packed_int4_mb:.2f} MB INT4) | backbone={pinfo['backbone_params']:,} "
        f"| opt={opt.mode} | steps={STEPS_PER_ARM} ===",
        flush=True,
    )

    model.train()
    losses_tail: List[float] = []
    t_train0 = time.perf_counter()

    for step in range(1, STEPS_PER_ARM + 1):
        opt.set_lr_scale(_wsd_mult(step, STEPS_PER_ARM, WARMUP_STEPS))
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
                f"| ce={float(ldict['ce_loss'].detach().item()):.4f} | {step_ms:.1f} ms/step",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    else:
        peak_vram_mb = 0.0
    train_elapsed = time.perf_counter() - t_train0
    step_ms = (train_elapsed / max(1, STEPS_PER_ARM)) * 1000.0

    raw_eval = _eval_upright_and_tilted_ce(model, val_batches, device, use_bf16)
    with ema.apply(model):
        ema_eval = _eval_upright_and_tilted_ce(model, val_batches, device, use_bf16)

    summary: Dict[str, Any] = {
        "arm": arm["name"],
        "is_control": bool(arm["is_control"]),
        "steps": STEPS_PER_ARM,
        "elapsed_sec": round(train_elapsed, 2),
        "step_ms": round(step_ms, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "final_25step_loss": round(sum(losses_tail) / max(1, len(losses_tail)), 4),
        "val_ce": raw_eval["val_ce"],
        "val_tilted_ce": raw_eval["val_tilted_ce"],
        "ema_val_ce": ema_eval["val_ce"],
        "ema_val_tilted_ce": ema_eval["val_tilted_ce"],
        "tilt_degradation_ce": ema_eval["tilt_degradation_ce"],
        "infer_params": int(pinfo["infer_params"]),
        "backbone_params": int(pinfo["backbone_params"]),
        "packed_int4_mb": packed_int4_mb,
        "raw_eval": raw_eval,
        "ema_eval": ema_eval,
    }
    _atomic_write_json(done_file, summary)

    del model, opt, ema
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def run_cpu_smoke_test() -> Dict[str, Any]:
    """1-step tiny CPU forward+backward+upright/tilted eval smoke test for BOTH arms."""
    import torch
    import train_uni_v2
    from uni_omr_v2 import BOS_ID, EOS_ID

    mvd_dir = _locate_or_unpack_mvd()
    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))

    device = torch.device("cpu")
    results: Dict[str, Any] = {}
    for arm in ARMS:
        _seed_everything(SEED)
        cfg = train_uni_v2.uni_omr_v2_tiny(n_axial_layers=0, n_global_layers=2, n_dec_layers=2, mtp_k=4)
        model = _build_arm_model(arm["mode"], cfg).to(device)
        opt = train_uni_v2.build_optimizer_v2(model, optimizer_name="muon", lr=1e-3, device_type="cpu")

        imgs = torch.rand(2, 3, 64, 96)
        seq = torch.randint(0, 40, (2, 16), dtype=torch.long)
        seq[:, 0] = BOS_ID
        seq[:, -1] = EOS_ID
        tgt_in, tgt_out = seq[:, :-1], seq[:, 1:]

        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(imgs, tgt_in, return_aux=True, return_mtp=True)
        ldict = model.compute_loss(out, tgt_out, images=imgs, sys_y_loss_weight=0.0, staff_seg_loss_weight=0.0)
        ldict["loss"].backward()
        opt.step()

        eval_res = _eval_upright_and_tilted_ce(model, [(imgs, seq)], device, use_bf16=False)
        pinfo = model.param_summary(int4_block_size=32)
        assert math.isfinite(float(ldict["loss"].item())), f"Non-finite loss in {arm['name']}"
        assert math.isfinite(float(eval_res["val_ce"])), f"Non-finite val_ce in {arm['name']}"
        assert math.isfinite(float(eval_res["val_tilted_ce"])), f"Non-finite val_tilted_ce in {arm['name']}"
        results[arm["name"]] = {
            "loss": round(float(ldict["loss"].item()), 4),
            "val_ce": eval_res["val_ce"],
            "val_tilted_ce": eval_res["val_tilted_ce"],
            "packed_int4_mb": round(float(pinfo["hybrid_fp16_int4_with_scales_mb"]), 3),
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

        train_manifest, val_manifest = _ensure_proxy_dataset(hf_token)
        train_rows = _load_manifest_rows(train_manifest)
        val_rows = _load_manifest_rows(val_manifest)
        val_samples = [_load_page_sample(ip, bps, FULLPAGE_MAX_DIM, MAX_SEQ_LEN) for ip, bps in val_rows]
        val_batches = [
            _collate_pages(val_samples[i : i + BATCH_SIZE], PAD_ID)
            for i in range(0, len(val_samples), BATCH_SIZE)
        ]
        OUT_ROOT.mkdir(parents=True, exist_ok=True)

        for arm in ARMS:
            summary = _run_arm(arm, train_rows, val_batches, device, use_bf16=use_bf16)
            arm_results.append(summary)

            eval_payload = {
                "exp_id": EXP_ID,
                "status": "RUNNING" if len(arm_results) < len(ARMS) else "COMPLETED",
                "started_at": started_at,
                "updated_at": _utc_now(),
                "gpu": gpu_name,
                "steps_per_arm": STEPS_PER_ARM,
                "seed": SEED,
                "tilt_stress_deg": TILT_STRESS_DEG,
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
            "best_ema_val_tilted_ce": best_arm.get("ema_val_tilted_ce") if best_arm else None,
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
