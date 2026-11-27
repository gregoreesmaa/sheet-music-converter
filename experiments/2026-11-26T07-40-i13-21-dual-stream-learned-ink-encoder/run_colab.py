#!/usr/bin/env python3
"""Google Colab L4 runner for Iter 13 Moonshot i13-21:
Dual-Stream Raw-RGB + Differentiable Learned-Binarized-Ink Encoder (DualInkStream)
with Unlabeled RISM Photometric-Invariance SSL.

Experiment ID: 2026-11-26T07-40-i13-21-dual-stream-learned-ink-encoder
Controlled 2-Arm Ablation (BF16, seed=42) on 1x Colab L4:
  - Arm A (Control): Single-stream RGB UniOMR v2 (small_45m, ~41.3M) fine-tuned
    1,000 steps on iter10_5_proxy_2k with 50% archival parchment/bleed-through aug.
  - Arm B (Treatment): Identical UniOMR v2 + DualInkStream (+0.86M params,
    ~42.2M total, < 32 MB Hybrid INT4, exact identity at step 0 via zero-init
    FiLM + residual gates):
      * LearnedInkBinarizer (0.18M): multi-scale adaptive high-pass (15x15, 41x41)
        + 3-stage residual U-Net -> canonical soft-ink probability map I_ink.
      * InkStreamEncoder (0.68M): 3-stage anisotropic conv encoder -> zero-init
        gated FiLM + residual fusion into AnisotropicMusicBackbone /8 features.
      * RISM Photometric-Invariance SSL every 2 steps on 5,707 unlabeled real
        historical RISM manuscripts.

Uploads:
  - evals/2026-11-26T07-40-i13-21-dual-stream-learned-ink-encoder.json
  - runs/2026-11-26T07-40-i13-21-dual-stream-learned-ink-encoder.json
to gregoreesmaa/sheet-music-converter-omr.
"""

from __future__ import annotations

import copy
import difflib
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T07-40-i13-21-dual-stream-learned-ink-encoder"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt",
)
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_RISM_TAR = "datasets/rism_hw_5707.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

VOL_ROOT = Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

SEED = 42
TRAIN_STEPS = 1000
WARMUP_STEPS = 100
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
GRAD_ACCUM = 4
LR_BACKBONE = 3e-5
LR_INK = 3e-4
LR_GATE = 3e-3
WEIGHT_DECAY = 0.04
RISM_SSL_EVERY = 2
RISM_SSL_WEIGHT = 0.25
RISM_MAX_DIM = 768
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
SUITE31_MAX_LEN = 2048
EMA_DECAY = 0.995
SAVE_EVERY_MINUTES = 5.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str] | str, *, shell: bool = False, env: Optional[Dict[str, str]] = None) -> None:
    print(f"[{EXP_ID}] $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        shell=shell,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {cmd}")


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


def _locate_repo_and_mvd() -> Tuple[Path, Path]:
    for repo in (
        Path("/content/sheet-music-converter"),
        Path("/workspace/sheet-music-converter"),
        Path("/content/workspace/sheet-music-converter"),
        Path.cwd(),
    ):
        mvd = repo / "tmp-train" / "mvd"
        if (mvd / "uni_omr_v2.py").is_file():
            if str(mvd) not in sys.path:
                sys.path.insert(0, str(mvd))
            return repo, mvd

    for bundle in (Path("/content/ov_mvd_bundle.tar.gz"), Path("/content/bundle.tar.gz")):
        if bundle.is_file():
            extract_root = Path("/content/workspace")
            extract_root.mkdir(parents=True, exist_ok=True)
            print(f"[{EXP_ID}] Extracting {bundle} -> {extract_root}...", flush=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=extract_root)
            for repo, mvd in (
                (extract_root / "sheet-music-converter", extract_root / "sheet-music-converter" / "tmp-train" / "mvd"),
                (extract_root, extract_root / "mvd"),
            ):
                if (mvd / "uni_omr_v2.py").is_file():
                    if str(mvd) not in sys.path:
                        sys.path.insert(0, str(mvd))
                    return repo, mvd

    raise FileNotFoundError("Could not locate tmp-train/mvd/uni_omr_v2.py")


def _setup_environment() -> Tuple[Path, Optional[Path], Optional[Path]]:
    repo_root, _ = _locate_repo_and_mvd()
    needed = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed])

    aser_manifest = repo_root / "converter" / "assembly-serialize" / "Cargo.toml"
    eharness_manifest = repo_root / "converter" / "eval-harness" / "Cargo.toml"
    aser_bin = aser_manifest.parent / "target" / "release" / "assembly-serialize"
    eharness_bin = eharness_manifest.parent / "target" / "release" / "eval-harness"

    if aser_manifest.is_file() and eharness_manifest.is_file():
        cargo_bin_dir = Path.home() / ".cargo" / "bin"
        env = os.environ.copy()
        env["PATH"] = f"{cargo_bin_dir}:{env.get('PATH', '')}"
        if shutil.which("cargo", path=env["PATH"]) is None:
            _run_cmd(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
                "| sh -s -- -y --default-toolchain stable --profile minimal",
                shell=True,
                env=env,
            )
        if not aser_bin.is_file():
            _run_cmd(["cargo", "build", "--release", "--manifest-path", str(aser_manifest)], env=env)
        if not eharness_bin.is_file():
            _run_cmd(["cargo", "build", "--release", "--manifest-path", str(eharness_manifest)], env=env)
        if aser_bin.is_file() and eharness_bin.is_file():
            return repo_root, aser_bin, eharness_bin

    return repo_root, None, None


def _ensure_assets(hf_token: str) -> Tuple[Optional[Path], Optional[str], Path, Path, List[Path], Path]:
    from huggingface_hub import hf_hub_download

    VOL_ROOT.mkdir(parents=True, exist_ok=True)
    ds_root = VOL_ROOT / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)
    ckpt_root = VOL_ROOT / "pretrained"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    ckpt_local: Optional[Path] = None
    ckpt_source: Optional[str] = None
    for cand in HF_CKPT_CANDIDATES:
        target = ckpt_root / Path(cand).name
        if target.is_file():
            ckpt_local = target
            ckpt_source = cand
            break
        try:
            dl_p = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=cand,
                repo_type="dataset",
                token=hf_token,
                local_dir="/tmp/hf_dl",
            )
            shutil.copy2(dl_p, target)
            try:
                os.remove(dl_p)
            except OSError:
                pass
            ckpt_local = target
            ckpt_source = cand
            print(f"[{EXP_ID}] Using warm-start checkpoint {cand}", flush=True)
            break
        except Exception as exc:
            print(f"[{EXP_ID}] Checkpoint {cand} unavailable ({exc}); trying next...", flush=True)

    proxy_dir = ds_root / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    val_abs = proxy_dir / "proxy_val_abs.jsonl"
    if not (train_abs.is_file() and val_abs.is_file()):
        proxy_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_PROXY_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=proxy_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass
        for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
            src_path = proxy_dir / src_name
            out_lines: List[str] = []
            for raw in src_path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                rec["image"] = str((proxy_dir / rec["image"]).resolve())
                rec["bekerns"] = [str((proxy_dir / b).resolve()) for b in rec.get("bekerns", [])]
                out_lines.append(json.dumps(rec))
            dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    rism_dir = ds_root / "rism_hw_5707"
    rism_sentinel = rism_dir / ".unpacked_ok"
    if not rism_sentinel.is_file():
        rism_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_RISM_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=rism_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass
        rism_sentinel.write_text("ok\n", encoding="utf-8")

    rism_imgs = sorted(
        p for p in rism_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )

    suite31_dir = ds_root / "suite31_hw_eval"
    manifest_path = suite31_dir / "suite31" / "suite31_hw_manifest.json"
    if not manifest_path.is_file():
        suite31_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_SUITE31_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=suite31_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass

    return ckpt_local, ckpt_source, train_abs, val_abs, rism_imgs, manifest_path


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


def _load_manifest_pairs(manifest_path: Path) -> List[Tuple[Path, List[Path], str]]:
    pairs: List[Tuple[Path, List[Path], str]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        img_p = Path(rec["image"])
        beks = [Path(b) for b in rec.get("bekerns", [])]
        domain = str(rec.get("domain", "")).strip().lower()
        if img_p.is_file() and beks:
            pairs.append((img_p, beks, domain))
    return pairs


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


def _apply_archival_stress(img_t: "torch.Tensor", rng: random.Random, *, deterministic: bool = False) -> "torch.Tensor":
    """Apply realistic archival parchment yellowing, verso bleed-through ghost, and uneven vignette lighting."""
    import torch

    # Convert [-1, 1] -> [0, 1]
    x = (img_t * 0.5 + 0.5).clamp(0.0, 1.0)
    _, _, H, W = x.shape

    # 1. Parchment tint (warm sepia / aged paper cast)
    tint_r = 0.94 if deterministic else rng.uniform(0.84, 0.98)
    tint_g = 0.87 if deterministic else rng.uniform(0.76, 0.93)
    tint_b = 0.72 if deterministic else rng.uniform(0.60, 0.84)
    tint = torch.tensor([tint_r, tint_g, tint_b], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = x * tint

    # 2. Verso bleed-through ghost (horizontally flipped + shifted faint ink)
    alpha_bt = 0.22 if deterministic else rng.uniform(0.10, 0.28)
    shift_y = 14 if deterministic else rng.randint(-24, 24)
    ghost = torch.roll(torch.flip(x, dims=[-1]), shifts=shift_y, dims=-2)
    x = (1.0 - alpha_bt) * x + alpha_bt * ghost

    # 3. Uneven illumination gradient / shadow
    yy = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
    xx = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W)
    gx = 0.16 if deterministic else rng.uniform(-0.22, 0.22)
    gy = -0.14 if deterministic else rng.uniform(-0.22, 0.22)
    illum = (1.0 + gx * xx + gy * yy - 0.12 * (xx * xx + yy * yy)).clamp(0.55, 1.15)
    x = (x * illum).clamp(0.0, 1.0)

    # 4. Iron-gall contrast washout
    contrast = 0.72 if deterministic else rng.uniform(0.60, 0.88)
    mean_v = x.mean(dim=(-2, -1), keepdim=True)
    x = ((x - mean_v) * contrast + mean_v).clamp(0.0, 1.0)
    return (x - 0.5) / 0.5


def _encode_bekerns_full(bek_paths: Sequence[Path]) -> List[int]:
    import data
    import uni_omr

    full_toks: List[str] = []
    for idx, bp in enumerate(bek_paths):
        toks = data.load_bekern(bp)
        if not toks:
            continue
        if idx > 0:
            full_toks.append(uni_omr.SYS_TOKEN)
        full_toks.extend(toks)
    return uni_omr.encode_uni(full_toks)


def _build_dual_ink_modules():
    """Define LearnedInkBinarizer (0.18M) + InkStreamEncoder (0.68M) with zero-init FiLM gates."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LearnedInkBinarizer(nn.Module):
        """Differentiable Sauvola/high-pass + compact residual U-Net -> soft ink map I_ink in [0, 1]."""

        def __init__(self, ch: int = 24) -> None:
            super().__init__()
            self.enc1 = nn.Sequential(
                nn.Conv2d(4, ch, 3, padding=1, bias=False),
                nn.GroupNorm(4, ch),
                nn.SiLU(inplace=True),
            )
            self.enc2 = nn.Sequential(
                nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, ch * 2),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch * 2, ch * 2, 3, padding=1, bias=False),
                nn.GroupNorm(8, ch * 2),
                nn.SiLU(inplace=True),
            )
            self.dec1 = nn.Sequential(
                nn.Conv2d(ch * 3, ch, 3, padding=1, bias=False),
                nn.GroupNorm(4, ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch, 1, 1, bias=True),
            )

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            # images in [-1, 1] -> luma in [0, 1]
            x01 = (images * 0.5 + 0.5).clamp(0.0, 1.0)
            luma = 0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]
            bg15 = F.avg_pool2d(luma, kernel_size=15, stride=1, padding=7)
            bg41 = F.avg_pool2d(luma, kernel_size=41, stride=1, padding=20)
            hp15 = (bg15 - luma).clamp(-0.5, 1.0)
            hp41 = (bg41 - luma).clamp(-0.5, 1.0)
            var15 = F.avg_pool2d((luma - bg15).pow(2), kernel_size=15, stride=1, padding=7)
            std15 = torch.sqrt(var15 + 1e-5)
            inp = torch.cat([luma, hp15, hp41, std15], dim=1)
            f1 = self.enc1(inp)
            f2 = self.enc2(f1)
            up = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
            logits = self.dec1(torch.cat([f1, up], dim=1))
            return torch.sigmoid(logits)

    class InkStreamEncoder(nn.Module):
        """Lightweight anisotropic conv encoder for canonical soft-ink canvas I_ink -> /8 FiLM."""

        def __init__(self, d_model: int = 384) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(16, 128),
                nn.SiLU(inplace=True),
            )
            self.h_strip = nn.Conv2d(128, 128, (1, 9), padding=(0, 4), groups=128, bias=False)
            self.v_strip = nn.Conv2d(128, 128, (9, 1), padding=(4, 0), groups=128, bias=False)
            self.norm = nn.GroupNorm(16, 128)
            self.to_film = nn.Conv2d(128, d_model * 2, 1, bias=True)
            self.gate_scale = nn.Parameter(torch.zeros(1))
            self.gate_bias = nn.Parameter(torch.zeros(1))

        def forward(self, i_ink: torch.Tensor, target_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
            h = self.stem(i_ink)
            h = F.silu(self.norm(h + self.h_strip(h) + self.v_strip(h)))
            if h.shape[-2:] != target_hw:
                h = F.interpolate(h, size=target_hw, mode="bilinear", align_corners=False)
            gamma, beta = self.to_film(h).chunk(2, dim=1)
            return gamma, beta

    class DualInkStreamWrapper(nn.Module):
        def __init__(self, base_model: nn.Module) -> None:
            super().__init__()
            self.base = base_model
            self.binarizer = LearnedInkBinarizer(ch=24)
            self.ink_encoder = InkStreamEncoder(d_model=int(base_model.cfg.d_model))

        def encode_dual(self, images: torch.Tensor) -> Tuple[Any, torch.Tensor]:
            feat_2d = self.base.backbone(images, return_pyramid=False)
            i_ink = self.binarizer(images)
            gamma, beta = self.ink_encoder(i_ink, feat_2d.shape[-2:])
            gs = torch.tanh(self.ink_encoder.gate_scale).to(feat_2d.dtype)
            gb = torch.tanh(self.ink_encoder.gate_bias).to(feat_2d.dtype)
            feat_fused = feat_2d * (1.0 + gs * gamma) + gb * beta
            mem_levels, _ = self.base.encoder(feat_fused, images=images)
            return mem_levels, i_ink

        def rism_ssl_loss(self, rism_clean: torch.Tensor, rism_stressed: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                ink_ref = self.binarizer(rism_clean).detach()
            ink_st = self.binarizer(rism_stressed)
            l_inv = F.l1_loss(ink_st, ink_ref)
            l_tv = (
                (ink_st[:, :, :, 1:] - ink_st[:, :, :, :-1]).abs().mean()
                + (ink_st[:, :, 1:, :] - ink_st[:, :, :-1, :]).abs().mean()
            )
            # Soft archival ink sparsity target (~8% foreground ink density)
            l_sparse = (ink_st.mean() - 0.08).pow(2)
            return l_inv + 0.10 * l_tv + 0.15 * l_sparse

        def mean_abs_gate(self) -> float:
            gs = abs(float(torch.tanh(self.ink_encoder.gate_scale).item()))
            gb = abs(float(torch.tanh(self.ink_encoder.gate_bias).item()))
            return round(0.5 * (gs + gb), 5)

    return DualInkStreamWrapper


def _build_base_model(ckpt_path: Optional[Path], device: "torch.device") -> Any:
    import torch
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net

    if ckpt_path is not None and ckpt_path.is_file():
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        raw_cfg = ck.get("cfg") or ck.get("config") or {}
        if isinstance(raw_cfg, UniOMRv2Config):
            cfg = raw_cfg
        elif isinstance(raw_cfg, dict) and raw_cfg:
            valid = set(UniOMRv2Config.__dataclass_fields__.keys())
            filtered = {k: v for k, v in raw_cfg.items() if k in valid}
            for k in ("backbone_channels", "backbone_blocks"):
                if k in filtered and isinstance(filtered[k], list):
                    filtered[k] = tuple(filtered[k])
            cfg = UniOMRv2Config(**filtered)
        else:
            cfg = UniOMRv2Config.small_45m()
        model = UniOMRv2Net(cfg)
        sd = ck.get("ema_state_dict") or ck.get("model_state_dict") or ck.get("state_dict") or ck
        model.load_state_dict(sd, strict=False)
        return model.to(device)
    return UniOMRv2Net(UniOMRv2Config.small_45m()).to(device)


def _token_f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    sm = difflib.SequenceMatcher(a=list(gold), b=list(pred), autojunk=False)
    matches = sum(b.size for b in sm.get_matching_blocks())
    return (2.0 * matches) / float(len(pred) + len(gold))


def _eval_proxy_val(
    wrapper_or_base: Any,
    val_pairs: List[Tuple[Path, List[Path], str]],
    *,
    is_dual: bool,
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr

    wrapper_or_base.eval()
    base = wrapper_or_base.base if is_dual else wrapper_or_base
    clean_ces: List[float] = []
    stress_ces: List[float] = []
    rng = random.Random(SEED + 99)

    with torch.no_grad():
        for img_p, beks, _ in val_pairs[:32]:
            img = Image.open(img_p).convert("RGB")
            full_ids = _encode_bekerns_full(beks)
            if len(full_ids) < 6:
                continue
            inp_clean = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
            inp_stress = _apply_archival_stress(inp_clean, rng, deterministic=True)
            tgt = torch.tensor(full_ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)

            for x_in, bucket in ((inp_clean, clean_ces), (inp_stress, stress_ces)):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    if is_dual:
                        mem, _ = wrapper_or_base.encode_dual(x_in)
                    else:
                        mem, _, _ = base.encode_image(x_in, return_aux=False)
                    out = base.decode_tokens(tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False)
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID)
                bucket.append(float(loss.item()))

    c_mean = sum(clean_ces) / max(1, len(clean_ces))
    s_mean = sum(stress_ces) / max(1, len(stress_ces))
    return {
        "val_clean_ce": round(c_mean, 5),
        "val_archival_stress_ce": round(s_mean, 5),
        "archival_ce_degradation": round(s_mean - c_mean, 5),
        "n_pages": len(clean_ces),
    }


def _eval_suite31(
    wrapper_or_base: Any,
    manifest_path: Path,
    *,
    is_dual: bool,
    device: "torch.device",
    use_bf16: bool,
    aser_bin: Optional[Path],
    eharness_bin: Optional[Path],
) -> Dict[str, Any]:
    import xml.etree.ElementTree as ET
    import torch
    from PIL import Image
    import eval_uni_suite31 as eus
    import rlvr_grpo

    def _extract_musicxml_notes(xml_path: Path) -> List[Tuple[str, int, str, str, str]]:
        notes: List[Tuple[str, int, str, str, str]] = []
        try:
            root = ET.parse(str(xml_path)).getroot()
        except Exception:
            return notes

        def _strip(tag: str) -> str:
            return tag.split("}", 1)[-1] if "}" in tag else tag

        for elem in root.iter():
            if _strip(elem.tag) != "note":
                continue
            is_rest = is_grace = False
            step = octave = alter = dur = ""
            n_dots = 0
            for child in elem:
                ctag = _strip(child.tag)
                if ctag == "rest":
                    is_rest = True
                elif ctag == "grace":
                    is_grace = True
                elif ctag == "pitch":
                    for pc in child:
                        ptag = _strip(pc.tag)
                        text = (pc.text or "").strip()
                        if ptag == "step":
                            step = text.upper()
                        elif ptag == "octave":
                            octave = text
                        elif ptag == "alter":
                            try:
                                av = int(round(float(text)))
                                if av != 0:
                                    alter = f"a{av}"
                            except ValueError:
                                pass
                elif ctag == "type":
                    dur = (child.text or "").strip().lower()
                elif ctag == "dot":
                    n_dots += 1
            if not is_rest and not is_grace and step and octave and dur:
                notes.append((dur, n_dots, step, octave, alter))
        return notes

    def _decode_from_mem(model_base: Any, mem: Any, max_len: int) -> List[int]:
        _, bos_id, eos_id, _, _, _ = eus._vocab_spec(int(model_base.cfg.vocab_size))
        cur_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        past_kvs = None
        out_ids = [bos_id]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            for _ in range(max_len):
                logits, _, past_kvs = model_base.decode_tokens(
                    cur_ids, mem, past_kvs=past_kvs, use_cache=True
                )
                nxt = int(logits[:, -1, :].argmax(dim=-1).item())
                out_ids.append(nxt)
                if nxt == eos_id:
                    break
                cur_ids = torch.tensor([[nxt]], dtype=torch.long, device=device)
        return out_ids

    wrapper_or_base.eval()
    base = wrapper_or_base.base if is_dual else wrapper_or_base
    _, _, eos_id, _, _, id_to_tok = eus._vocab_spec(int(base.cfg.vocab_size))
    _, pieces = eus.resolve_manifest_pieces(manifest_path)

    buckets: Dict[str, List[Dict[str, float]]] = {
        "printed_clean": [],
        "printed_photo": [],
        "hw_clean": [],
        "hw_photo": [],
    }
    latencies: List[float] = []

    with torch.no_grad():
        for entry in pieces:
            is_hw = bool(entry.get("handwritten", False))
            ref_notes = _extract_musicxml_notes(Path(entry["gt"]))

            for mode_key in ("clean", "photo"):
                img_p = Path(entry[mode_key])
                if not img_p.is_file():
                    continue
                img = Image.open(img_p).convert("RGB")
                inp_t = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
                t0 = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    if is_dual:
                        mem, _ = wrapper_or_base.encode_dual(inp_t)
                    else:
                        mem, _, _ = base.encode_image(inp_t, return_aux=False)
                ids = _decode_from_mem(base, mem, SUITE31_MAX_LEN)
                latencies.append(time.perf_counter() - t0)

                hit_eos = eos_id in ids[1:]
                pred_toks = eus.decode_uni_compat(ids, vocab_size=base.cfg.vocab_size, skip_special=True)
                if not hit_eos:
                    pred_toks = eus.trim_truncated_tokens(pred_toks)
                hyp_toks, _ = eus.stitch_uni_tokens(pred_toks)
                hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
                note_f1 = float(rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes))
                struct = rlvr_grpo.compute_unlabeled_structural_reward(
                    hyp_toks, vocab=id_to_tok, terminated_eos=hit_eos
                )
                dec_ok = 1.0 if (len(hyp_notes) > 0 and float(struct.get("r_parse", 0.0)) > 0.0) else 0.0
                tok_f1 = note_f1

                bkey = f"{'hw' if is_hw else 'printed'}_{mode_key}"
                buckets[bkey].append({"tok_f1": tok_f1, "note_f1": note_f1, "dec_ok": dec_ok})

    def _agg(rows: List[Dict[str, float]]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0, "tok_f1": None, "note_f1": None, "dec_ok": None}
        return {
            "n": len(rows),
            "tok_f1": round(sum(r["tok_f1"] for r in rows) / len(rows), 4),
            "note_f1": round(sum(r["note_f1"] for r in rows) / len(rows), 4),
            "dec_ok": round(sum(r["dec_ok"] for r in rows) / len(rows), 4),
        }

    return {
        "printed_clean": _agg(buckets["printed_clean"]),
        "printed_photo": _agg(buckets["printed_photo"]),
        "hw_clean": _agg(buckets["hw_clean"]),
        "hw_photo": _agg(buckets["hw_photo"]),
        "mean_latency_s": round(sum(latencies) / max(1, len(latencies)), 3),
    }


def _train_arm(
    arm_name: str,
    *,
    is_dual: bool,
    ckpt_path: Optional[Path],
    train_pairs: List[Tuple[Path, List[Path], str]],
    val_pairs: List[Tuple[Path, List[Path], str]],
    rism_imgs: List[Path],
    manifest_path: Path,
    device: "torch.device",
    use_bf16: bool,
    aser_bin: Optional[Path],
    eharness_bin: Optional[Path],
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr

    _seed_everything(SEED)
    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached_res = arm_dir / "arm_result.json"
    if cached_res.is_file():
        return json.loads(cached_res.read_text(encoding="utf-8"))

    base = _build_base_model(ckpt_path, device)
    DualCls = _build_dual_ink_modules()
    model = DualCls(base).to(device) if is_dual else base

    if is_dual:
        gate_params = [model.ink_encoder.gate_scale, model.ink_encoder.gate_bias]
        gate_ids = {id(p) for p in gate_params}
        base_ids = {id(p) for p in model.base.parameters()}
        ink_params = [
            p for p in model.parameters()
            if id(p) not in base_ids and id(p) not in gate_ids and p.requires_grad
        ]
        bb_params = [p for p in model.base.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            [
                {"params": bb_params, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
                {"params": ink_params, "lr": LR_INK, "weight_decay": 0.01},
                {"params": gate_params, "lr": LR_GATE, "weight_decay": 0.0},
            ]
        )
        extra_params = sum(p.numel() for p in ink_params) + sum(p.numel() for p in gate_params)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR_BACKBONE, weight_decay=WEIGHT_DECAY)
        extra_params = 0

    ema_state = copy.deepcopy(model.state_dict())
    rng = random.Random(SEED)
    t_last_save = time.time()
    t_arm0 = time.time()
    loss_hist: List[float] = []

    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        mult = _wsd_mult(step, TRAIN_STEPS, WARMUP_STEPS)
        opt.param_groups[0]["lr"] = LR_BACKBONE * mult
        if is_dual:
            opt.param_groups[1]["lr"] = LR_INK * mult
            opt.param_groups[2]["lr"] = LR_GATE * mult

        opt.zero_grad(set_to_none=True)
        step_loss_val = 0.0

        for _ in range(GRAD_ACCUM):
            img_p, beks, _ = train_pairs[rng.randrange(len(train_pairs))]
            img = Image.open(img_p).convert("RGB")
            full_ids = _encode_bekerns_full(beks)
            if len(full_ids) < 6:
                continue
            inp_t = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
            if rng.random() < 0.50:
                inp_t = _apply_archival_stress(inp_t, rng, deterministic=False)
            tgt = torch.tensor(full_ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                if is_dual:
                    mem, _ = model.encode_dual(inp_t)
                    cur_base = model.base
                else:
                    cur_base = model
                    mem, _, _ = cur_base.encode_image(inp_t, return_aux=False)
                out = cur_base.decode_tokens(tgt[:, :-1], mem, return_mtp=False, return_aux=False)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID
                ) / GRAD_ACCUM
            loss.backward()
            step_loss_val += float(loss.item())

        # Unlabeled RISM archival photometric-invariance SSL every 2 steps
        if is_dual and rism_imgs and (step % RISM_SSL_EVERY == 0):
            r_img = Image.open(rism_imgs[rng.randrange(len(rism_imgs))]).convert("RGB")
            r_clean = _pil_to_tensor(r_img, RISM_MAX_DIM).unsqueeze(0).to(device)
            r_stress = _apply_archival_stress(r_clean, rng, deterministic=False)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss_ssl = RISM_SSL_WEIGHT * model.rism_ssl_loss(r_clean, r_stress)
            loss_ssl.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_state[k].mul_(EMA_DECAY).add_(v, alpha=1.0 - EMA_DECAY)
                else:
                    ema_state[k].copy_(v)

        loss_hist.append(step_loss_val)
        if step % 100 == 0 or step == 1:
            mean_l = sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:]))
            gate_str = f" gate={model.mean_abs_gate():.4f}" if is_dual else ""
            print(
                f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} loss={mean_l:.4f}{gate_str} elapsed={time.time()-t_arm0:.1f}s",
                flush=True,
            )

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ck = arm_dir / "latest.pt.tmp"
            torch.save({"step": step, "ema_state_dict": ema_state}, tmp_ck)
            os.replace(tmp_ck, arm_dir / "latest.pt")
            t_last_save = time.time()

    model.load_state_dict(ema_state, strict=True)
    proxy_val = _eval_proxy_val(model, val_pairs, is_dual=is_dual, device=device, use_bf16=use_bf16)
    suite31 = _eval_suite31(
        model, manifest_path, is_dual=is_dual, device=device, use_bf16=use_bf16,
        aser_bin=aser_bin, eharness_bin=eharness_bin,
    )
    result = {
        "arm": arm_name,
        "is_dual": is_dual,
        "extra_params": extra_params,
        "total_params": sum(p.numel() for p in model.parameters()),
        "mean_abs_gate": model.mean_abs_gate() if is_dual else None,
        "train_loss_tail50": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5),
        "proxy_val": proxy_val,
        "suite31_hw": suite31,
        "elapsed_secs": round(time.time() - t_arm0, 2),
    }
    _atomic_write_json(cached_res, result)
    del model, opt, ema_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    t_start = time.time()
    started_iso = _utc_now()
    hf_token = _read_and_purge_hf_token()
    try:
        _, aser_bin, eharness_bin = _setup_environment()
        import torch
        from huggingface_hub import HfApi

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

        ckpt_local, ckpt_source, train_abs, val_abs, rism_imgs, manifest_path = _ensure_assets(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(
            f"[{EXP_ID}] Loaded {len(train_pairs)} train / {len(val_pairs)} val / {len(rism_imgs)} RISM on {gpu_name}",
            flush=True,
        )

        arm_a = _train_arm(
            "arm_a_control_rgb_only",
            is_dual=False,
            ckpt_path=ckpt_local,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            rism_imgs=rism_imgs,
            manifest_path=manifest_path,
            device=device,
            use_bf16=use_bf16,
            aser_bin=aser_bin,
            eharness_bin=eharness_bin,
        )
        arm_b = _train_arm(
            "arm_b_dual_stream_learned_ink",
            is_dual=True,
            ckpt_path=ckpt_local,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            rism_imgs=rism_imgs,
            manifest_path=manifest_path,
            device=device,
            use_bf16=use_bf16,
            aser_bin=aser_bin,
            eharness_bin=eharness_bin,
        )

        def _delta(b: Optional[float], a: Optional[float]) -> Optional[float]:
            return None if (b is None or a is None) else round(float(b) - float(a), 5)

        deg_a = arm_a["proxy_val"]["archival_ce_degradation"]
        deg_b = arm_b["proxy_val"]["archival_ce_degradation"]
        deg_red_pct = round(100.0 * (deg_a - deg_b) / max(1e-5, abs(deg_a)), 2)

        comparison = {
            "val_clean_ce_delta_b_minus_a": _delta(arm_b["proxy_val"]["val_clean_ce"], arm_a["proxy_val"]["val_clean_ce"]),
            "val_stress_ce_delta_b_minus_a": _delta(arm_b["proxy_val"]["val_archival_stress_ce"], arm_a["proxy_val"]["val_archival_stress_ce"]),
            "archival_ce_degradation_reduction_pct": deg_red_pct,
            "hw_clean_note_f1_delta": _delta(arm_b["suite31_hw"]["hw_clean"]["note_f1"], arm_a["suite31_hw"]["hw_clean"]["note_f1"]),
            "hw_photo_note_f1_delta": _delta(arm_b["suite31_hw"]["hw_photo"]["note_f1"], arm_a["suite31_hw"]["hw_photo"]["note_f1"]),
            "printed_clean_note_f1_delta": _delta(arm_b["suite31_hw"]["printed_clean"]["note_f1"], arm_a["suite31_hw"]["printed_clean"]["note_f1"]),
            "learned_mean_abs_gate": arm_b["mean_abs_gate"],
        }

        report = {
            "exp_id": EXP_ID,
            "exp_code": "i13-21",
            "status": "completed",
            "warmstart_ckpt": ckpt_source or "scratch_small_45m",
            "started_utc": started_iso,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"colab-{gpu_name}",
            "bf16": use_bf16,
            "arm_a_control": arm_a,
            "arm_b_dual_ink": arm_b,
            "comparison_b_minus_a": comparison,
        }

        eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        run_local = VOL_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(
            run_local,
            {
                "exp_id": EXP_ID,
                "status": "completed",
                "platform": f"colab-{gpu_name}",
                "started_utc": started_iso,
                "completed_utc": report["completed_utc"],
                "elapsed_secs": report["elapsed_secs"],
                "eval_path": HF_EVAL_PATH,
                "comparison_b_minus_a": comparison,
            },
        )

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload DualInkStream ablation",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload DualInkStream run ledger",
        )
        print(f"[{EXP_ID}] UPLOADED {HF_EVAL_PATH} & {HF_RUN_PATH} TO HF!", flush=True)
        print(json.dumps(comparison, indent=2), flush=True)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
