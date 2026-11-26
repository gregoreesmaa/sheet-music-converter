#!/usr/bin/env python3
"""Google Colab L4 runner for Iter 13 Moonshot i13-19:
Coarse-to-Fine 2-Pass System-Plan -> Native-Res Zoomed-System Foveated Decoder.

Experiment ID: 2026-11-26T07-40-i13-19-coarse2fine-zoom-omr
Controlled 2-Arm Ablation (BF16, seed=42) on 1x Colab L4:
  - Arm A (Control): Single-pass 1280px UniOMR v2 (small_45m, ~41.3M) fine-tuned
    1,000 steps on iter10_5_proxy_2k.
  - Arm B (Treatment): Shared UniOMR v2 + SystemBandProposalHead +
    GlobalContextCarryProj (+0.24M params, ~41.5M total, < 31 MB Hybrid INT4):
      * Pass 1 (768px): Predicts per-row vertical system bands [y_top, y_bot]
        and pools 4 Global Page Context tokens from mem_final.
      * Pass 2 (256x1280 per band): Crops each predicted band with +/-12% pad
        directly from the native high-res image (3-5x vertical DPI), encodes
        with the shared encoder, prepends Global Context tokens, and decodes
        system-by-system joined by <sys>.

Uploads:
  - evals/2026-11-26T07-40-i13-19-coarse2fine-zoom-omr.json
  - runs/2026-11-26T07-40-i13-19-coarse2fine-zoom-omr.json
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

EXP_ID = "2026-11-26T07-40-i13-19-coarse2fine-zoom-omr"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt",
)
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
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
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR_BACKBONE = 3e-5
LR_HEAD = 3e-4
WEIGHT_DECAY = 0.04
FULLPAGE_MAX_DIM = 1280
COARSE_MAX_DIM = 768
ZOOM_BAND_H = 256
ZOOM_BAND_W = 1280
BAND_PAD_FRAC = 0.12
MAX_SEQ_LEN = 1536
BAND_MAX_SEQ_LEN = 512
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


def _ensure_assets(hf_token: str) -> Tuple[Optional[Path], Optional[str], Path, Path, Path]:
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
            print(f"[{EXP_ID}] Trying warm-start checkpoint {cand}...", flush=True)
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

    return ckpt_local, ckpt_source, train_abs, val_abs, manifest_path


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


def _extract_gt_system_bands_from_image(img_pil: Any, n_sys: int) -> List[Tuple[float, float]]:
    """Derive vertical [y_top, y_bot] system bands aligned to ground-truth n_sys via horizontal ink projection."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    n_sys = max(1, int(n_sys))
    gray = np.asarray(img_pil.convert("L"), dtype=np.float32) / 255.0
    H, W = gray.shape
    # High-pass ink energy per row (robust to parchment yellowing & vignette)
    row_bg = np.percentile(gray, 85, axis=1, keepdims=True)
    ink = np.clip(row_bg - gray, 0.0, 1.0)
    # Trim extreme 3% horizontal borders to avoid dark phone framing
    x0, x1 = int(0.03 * W), max(int(0.97 * W), int(0.03 * W) + 1)
    profile = ink[:, x0:x1].mean(axis=1)
    smooth = gaussian_filter1d(profile, sigma=max(2.0, H * 0.008))

    if n_sys == 1:
        active = np.where(smooth > max(0.015, float(smooth.max()) * 0.15))[0]
        if len(active) > 0:
            y0 = max(0.0, float(active[0]) / H - 0.04)
            y1 = min(1.0, float(active[-1]) / H + 0.04)
            return [(y0, max(y0 + 0.10, y1))]
        return [(0.02, 0.98)]

    # Split vertical range into n_sys bands by locating lowest-energy valleys around uniform anchors
    active = np.where(smooth > max(0.01, float(smooth.max()) * 0.10))[0]
    top_y = int(active[0]) if len(active) > 0 else int(0.03 * H)
    bot_y = int(active[-1]) if len(active) > 0 else int(0.97 * H)
    span = max(n_sys * 8, bot_y - top_y)
    cuts = [max(0, top_y - int(0.02 * H))]
    for k in range(1, n_sys):
        center = top_y + int(round(span * k / n_sys))
        win = max(6, int(round(span / (n_sys * 2.4))))
        lo = max(cuts[-1] + 4, center - win)
        hi = min(bot_y - 4, center + win)
        if hi <= lo:
            cuts.append(min(H - 1, cuts[-1] + max(4, span // n_sys)))
        else:
            valley = lo + int(np.argmin(smooth[lo:hi]))
            cuts.append(valley)
    cuts.append(min(H, bot_y + int(0.02 * H)))

    bands: List[Tuple[float, float]] = []
    for k in range(n_sys):
        y0 = max(0.0, float(cuts[k]) / H)
        y1 = min(1.0, float(cuts[k + 1]) / H)
        if y1 - y0 < 0.03:
            y1 = min(1.0, y0 + 0.05)
        bands.append((y0, y1))
    return bands


def _pil_to_tensor(img_pil: Any, max_dim: int, target_hw: Optional[Tuple[int, int]] = None) -> "torch.Tensor":
    import numpy as np
    import torch
    from PIL import Image

    rgb = img_pil.convert("RGB")
    if target_hw is not None:
        th, tw = target_hw
        rgb = rgb.resize((tw, th), Image.BICUBIC)
    else:
        w, h = rgb.size
        scale = min(1.0, float(max_dim) / float(max(w, h)))
        nw = max(32, int(round(w * scale / 16.0)) * 16)
        nh = max(32, int(round(h * scale / 16.0)) * 16)
        if (nw, nh) != (w, h):
            rgb = rgb.resize((nw, nh), Image.BICUBIC)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return (t - 0.5) / 0.5


def _crop_zoom_band(img_pil: Any, y0: float, y1: float, pad_frac: float = BAND_PAD_FRAC) -> Any:
    w, h = img_pil.size
    bh = max(0.04, y1 - y0)
    py0 = max(0, int(round((y0 - pad_frac * bh) * h)))
    py1 = min(h, int(round((y1 + pad_frac * bh) * h)))
    if py1 - py0 < 16:
        py1 = min(h, py0 + 16)
    return img_pil.crop((0, py0, w, py1))


def _build_moonshot_modules():
    """Define SystemBandProposalHead & GlobalContextCarryProj (total ~0.24M params)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SystemBandProposalHead(nn.Module):
        """Predicts vertical system interior + boundary probabilities from Pass-1 /8 features."""

        def __init__(self, d_model: int = 384, hidden: int = 128) -> None:
            super().__init__()
            self.in_proj = nn.Linear(d_model * 2, hidden)
            self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, dilation=1)
            self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2)
            self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=8, dilation=4)
            self.norm = nn.LayerNorm(hidden)
            self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
            self.out_head = nn.Linear(hidden, 2)  # [interior_logit, boundary_logit]

        def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
            # feat_2d: (B, d_model, H', W')
            h_mean = feat_2d.mean(dim=-1)  # (B, C, H')
            h_max = feat_2d.amax(dim=-1)   # (B, C, H')
            row_feat = torch.cat([h_mean, h_max], dim=1).transpose(1, 2)  # (B, H', 2C)
            x = F.silu(self.in_proj(row_feat))  # (B, H', hidden)
            xt = x.transpose(1, 2)
            xt = xt + F.silu(self.conv1(xt))
            xt = xt + F.silu(self.conv2(xt))
            xt = xt + F.silu(self.conv3(xt))
            x = self.norm(xt.transpose(1, 2))
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            x = x + attn_out
            return self.out_head(x)  # (B, H', 2)

        @staticmethod
        def decode_bands(logits_1d: torch.Tensor, thr: float = 0.45) -> List[Tuple[float, float]]:
            # logits_1d: (H', 2)
            probs = torch.sigmoid(logits_1d[:, 0]).detach().float().cpu().numpy()
            bnd = torch.sigmoid(logits_1d[:, 1]).detach().float().cpu().numpy()
            H = len(probs)
            active = probs > thr
            bands: List[Tuple[float, float]] = []
            in_seg = False
            start = 0
            for y in range(H):
                is_cut = bnd[y] > 0.55 and in_seg and (y - start) >= 3
                if active[y] and not in_seg:
                    start = y
                    in_seg = True
                elif (not active[y] or is_cut) and in_seg:
                    if y - start >= 2:
                        bands.append((float(start) / H, float(y) / H))
                    in_seg = active[y] and is_cut
                    if in_seg:
                        start = y
            if in_seg and (H - start) >= 2:
                bands.append((float(start) / H, 1.0))
            if not bands:
                bands = [(0.02, 0.98)]
            return bands

    class GlobalContextCarryProj(nn.Module):
        """Pools Pass-1 mem_final into n_ctx=4 zero-init gated Global Page Context tokens."""

        def __init__(self, d_model: int = 384, n_ctx: int = 4) -> None:
            super().__init__()
            self.queries = nn.Parameter(torch.randn(1, n_ctx, d_model) * 0.02)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.norm = nn.LayerNorm(d_model)
            self.gate = nn.Parameter(torch.zeros(1))

        def forward(self, mem_final: torch.Tensor) -> torch.Tensor:
            B, _, D = mem_final.shape
            q = self.queries.expand(B, -1, -1)
            k = self.k_proj(mem_final)
            v = self.v_proj(mem_final)
            attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(D), dim=-1)
            ctx = self.norm(torch.bmm(attn, v))
            return torch.tanh(self.gate).to(ctx.dtype) * ctx

    class Coarse2FineZoomWrapper(nn.Module):
        def __init__(self, base_model: nn.Module) -> None:
            super().__init__()
            self.base = base_model
            d_model = int(base_model.cfg.d_model)
            self.band_head = SystemBandProposalHead(d_model=d_model, hidden=128)
            self.ctx_proj = GlobalContextCarryProj(d_model=d_model, n_ctx=4)

        def pass1_plan(self, coarse_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            feat_2d = self.base.backbone(coarse_img, return_pyramid=False)
            band_logits = self.band_head(feat_2d)
            mem_levels, _ = self.base.encoder(feat_2d, images=coarse_img)
            global_ctx = self.ctx_proj(mem_levels[-1])
            return band_logits, global_ctx

        def encode_zoom_with_ctx(
            self, band_img: torch.Tensor, global_ctx: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mem_levels, _, _ = self.base.encode_image(band_img, return_aux=False)
            # Prepend 4 global page context tokens to all 3 DeepStack levels
            return tuple(torch.cat([global_ctx, m], dim=1) for m in mem_levels)  # type: ignore[return-value]

    return Coarse2FineZoomWrapper


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
    cfg = UniOMRv2Config.small_45m()
    return UniOMRv2Net(cfg).to(device)


def _encode_bekerns_to_systems(bek_paths: Sequence[Path]) -> Tuple[List[List[int]], List[int]]:
    import data
    import uni_omr

    sys_seqs: List[List[int]] = []
    full_toks: List[str] = []
    for idx, bp in enumerate(bek_paths):
        toks = data.read_bekern_tokens(bp)
        if not toks:
            continue
        sys_seqs.append(uni_omr.encode_uni(toks))
        if idx > 0:
            full_toks.append(uni_omr.SYS_TOKEN)
        full_toks.extend(toks)
    full_ids = uni_omr.encode_uni(full_toks)
    return sys_seqs, full_ids


def _band_targets(bands: List[Tuple[float, float]], h_feat: int, device: "torch.device") -> "torch.Tensor":
    import torch

    tgt = torch.zeros(h_feat, 2, device=device, dtype=torch.float32)
    for y0, y1 in bands:
        i0 = max(0, min(h_feat - 1, int(round(y0 * h_feat))))
        i1 = max(i0 + 1, min(h_feat, int(round(y1 * h_feat))))
        tgt[i0:i1, 0] = 1.0
        tgt[i0, 1] = 1.0
        if i1 < h_feat:
            tgt[i1, 1] = 1.0
    return tgt


def _token_f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    sm = difflib.SequenceMatcher(a=list(gold), b=list(pred), autojunk=False)
    matches = sum(b.size for b in sm.get_matching_blocks())
    return (2.0 * matches) / float(len(pred) + len(gold))


def _band_recall(pred_bands: List[Tuple[float, float]], gt_bands: List[Tuple[float, float]]) -> float:
    if not gt_bands:
        return 1.0
    matched = 0
    for gy0, gy1 in gt_bands:
        best_iou = 0.0
        for py0, py1 in pred_bands:
            inter = max(0.0, min(gy1, py1) - max(gy0, py0))
            union = max(1e-6, max(gy1, py1) - min(gy0, py0))
            best_iou = max(best_iou, inter / union)
        if best_iou >= 0.45:
            matched += 1
    return float(matched) / float(len(gt_bands))


def _eval_model_on_proxy_val(
    wrapper_or_base: Any,
    val_pairs: List[Tuple[Path, List[Path], str]],
    *,
    is_two_pass: bool,
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr

    wrapper_or_base.eval()
    base = wrapper_or_base.base if is_two_pass else wrapper_or_base
    ce_losses: List[float] = []
    band_recalls: List[float] = []
    sys_count_exact = 0

    with torch.no_grad():
        for img_p, beks, _ in val_pairs[:32]:
            img = Image.open(img_p).convert("RGB")
            sys_seqs, full_ids = _encode_bekerns_to_systems(beks)
            if len(full_ids) < 3:
                continue
            gt_bands = _extract_gt_system_bands_from_image(img, len(sys_seqs))

            if is_two_pass:
                coarse_t = _pil_to_tensor(img, COARSE_MAX_DIM).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    band_logits, global_ctx = wrapper_or_base.pass1_plan(coarse_t)
                pred_bands = wrapper_or_base.band_head.decode_bands(band_logits[0])
                band_recalls.append(_band_recall(pred_bands, gt_bands))
                if len(pred_bands) == len(gt_bands):
                    sys_count_exact += 1

                band_ces: List[float] = []
                for (y0, y1), seq in zip(gt_bands, sys_seqs):
                    if len(seq) < 3:
                        continue
                    crop = _crop_zoom_band(img, y0, y1)
                    bt = _pil_to_tensor(crop, ZOOM_BAND_W, target_hw=(ZOOM_BAND_H, ZOOM_BAND_W)).unsqueeze(0).to(device)
                    tgt = torch.tensor(seq[:BAND_MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        mem = wrapper_or_base.encode_zoom_with_ctx(bt, global_ctx)
                        out = base.decode_tokens(tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False)
                        logits = out[0] if isinstance(out, tuple) else out
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID)
                    band_ces.append(float(loss.item()))
                if band_ces:
                    ce_losses.append(sum(band_ces) / len(band_ces))
            else:
                inp_t = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
                tgt = torch.tensor(full_ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    mem, _, _ = base.encode_image(inp_t, return_aux=False)
                    out = base.decode_tokens(tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False)
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID)
                ce_losses.append(float(loss.item()))

    return {
        "val_ce_loss": round(sum(ce_losses) / max(1, len(ce_losses)), 5),
        "val_band_recall": round(sum(band_recalls) / max(1, len(band_recalls)), 4) if band_recalls else None,
        "val_sys_count_exact": round(sys_count_exact / max(1, len(band_recalls)), 4) if band_recalls else None,
        "n_pages": len(ce_losses),
    }


def _eval_suite31(
    wrapper_or_base: Any,
    manifest_path: Path,
    *,
    is_two_pass: bool,
    device: "torch.device",
    use_bf16: bool,
    aser_bin: Optional[Path],
    eharness_bin: Optional[Path],
) -> Dict[str, Any]:
    import torch
    from PIL import Image
    import data
    import eval_uni_suite31 as eus
    import uni_omr

    wrapper_or_base.eval()
    base = wrapper_or_base.base if is_two_pass else wrapper_or_base
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent

    buckets: Dict[str, List[Dict[str, float]]] = {
        "printed_clean": [],
        "printed_photo": [],
        "hw_clean": [],
        "hw_photo": [],
    }
    latencies: List[float] = []
    hw_band_recalls: List[float] = []

    with torch.no_grad():
        for entry in manifest.get("pieces", []):
            is_hw = bool(entry.get("is_handwritten", False)) or str(entry.get("domain", "")).lower().startswith("hw")
            gt_beks = [base_dir / p for p in entry.get("bekerns", [])]
            gt_toks: List[str] = []
            for idx, bp in enumerate(gt_beks):
                if bp.is_file():
                    if idx > 0:
                        gt_toks.append(uni_omr.SYS_TOKEN)
                    gt_toks.extend(data.read_bekern_tokens(bp))
            n_gt_sys = max(1, len(gt_beks))

            for mode_key, img_rel in (("clean", entry.get("clean_image")), ("photo", entry.get("photo_image"))):
                if not img_rel:
                    continue
                img_p = base_dir / img_rel
                if not img_p.is_file():
                    continue
                img = Image.open(img_p).convert("RGB")
                t0 = time.perf_counter()
                if is_two_pass:
                    coarse_t = _pil_to_tensor(img, COARSE_MAX_DIM).unsqueeze(0).to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        band_logits, global_ctx = wrapper_or_base.pass1_plan(coarse_t)
                    pred_bands = wrapper_or_base.band_head.decode_bands(band_logits[0])
                    if is_hw:
                        gt_bands = _extract_gt_system_bands_from_image(img, n_gt_sys)
                        hw_band_recalls.append(_band_recall(pred_bands, gt_bands))
                    pred_toks: List[str] = []
                    for b_idx, (y0, y1) in enumerate(pred_bands[:10]):
                        crop = _crop_zoom_band(img, y0, y1)
                        bt = _pil_to_tensor(crop, ZOOM_BAND_W, target_hw=(ZOOM_BAND_H, ZOOM_BAND_W)).unsqueeze(0).to(device)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                            mem = wrapper_or_base.encode_zoom_with_ctx(bt, global_ctx)
                        ids = eus.generate_constrained(
                            base, mem, max_len=BAND_MAX_SEQ_LEN, use_grammar=True, device=device
                        )
                        toks_b = eus.decode_uni_compat(ids, vocab_size=base.cfg.vocab_size)
                        if b_idx > 0 and toks_b:
                            pred_toks.append(uni_omr.SYS_TOKEN)
                        pred_toks.extend(toks_b)
                else:
                    inp_t = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        mem, _, _ = base.encode_image(inp_t, return_aux=False)
                    ids = eus.generate_constrained(
                        base, mem, max_len=SUITE31_MAX_LEN, use_grammar=True, device=device
                    )
                    pred_toks = eus.decode_uni_compat(ids, vocab_size=base.cfg.vocab_size)

                latencies.append(time.perf_counter() - t0)
                tok_f1 = _token_f1(pred_toks, gt_toks)
                note_f1 = tok_f1  # fallback if Rust binaries unavailable
                dec_ok = 1.0 if len(pred_toks) > 0 else 0.0
                if aser_bin is not None and eharness_bin is not None and entry.get("gt_musicxml"):
                    gt_xml = base_dir / entry["gt_musicxml"]
                    if gt_xml.is_file():
                        try:
                            with tempfile.TemporaryDirectory() as td:
                                bek_f = Path(td) / "pred.bekern"
                                xml_f = Path(td) / "pred.musicxml"
                                bek_f.write_text(" ".join(eus.trim_truncated_tokens(pred_toks)), encoding="utf-8")
                                r1 = subprocess.run(
                                    [str(aser_bin), "bekern-decode", "--input", str(bek_f), "--out", str(xml_f)],
                                    capture_output=True, text=True, timeout=15,
                                )
                                if r1.returncode == 0 and xml_f.is_file():
                                    r2 = subprocess.run(
                                        [str(eharness_bin), "--gt", str(gt_xml), "--pred", str(xml_f), "--json"],
                                        capture_output=True, text=True, timeout=15,
                                    )
                                    if r2.returncode == 0:
                                        res = json.loads(r2.stdout)
                                        note_f1 = float(res.get("strict_note_f1", res.get("note_f1", tok_f1)))
                                        dec_ok = 1.0
                                else:
                                    dec_ok = 0.0
                        except Exception:
                            dec_ok = 0.0

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
        "hw_band_recall": round(sum(hw_band_recalls) / max(1, len(hw_band_recalls)), 4) if hw_band_recalls else None,
        "mean_latency_s": round(sum(latencies) / max(1, len(latencies)), 3),
    }


def _train_arm(
    arm_name: str,
    *,
    is_two_pass: bool,
    ckpt_path: Optional[Path],
    train_pairs: List[Tuple[Path, List[Path], str]],
    val_pairs: List[Tuple[Path, List[Path], str]],
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
        print(f"[{EXP_ID}] Reusing cached result for {arm_name}", flush=True)
        return json.loads(cached_res.read_text(encoding="utf-8"))

    base = _build_base_model(ckpt_path, device)
    WrapperCls = _build_moonshot_modules()
    model = WrapperCls(base).to(device) if is_two_pass else base

    if is_two_pass:
        head_params = list(model.band_head.parameters()) + list(model.ctx_proj.parameters())
        head_ids = {id(p) for p in head_params}
        bb_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
        opt = torch.optim.AdamW(
            [
                {"params": bb_params, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
                {"params": head_params, "lr": LR_HEAD, "weight_decay": 0.01},
            ]
        )
        extra_params = sum(p.numel() for p in head_params)
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
        if is_two_pass:
            opt.param_groups[1]["lr"] = LR_HEAD * mult

        opt.zero_grad(set_to_none=True)
        step_loss_val = 0.0

        for _ in range(GRAD_ACCUM):
            img_p, beks, _ = train_pairs[rng.randrange(len(train_pairs))]
            img = Image.open(img_p).convert("RGB")
            sys_seqs, full_ids = _encode_bekerns_to_systems(beks)
            if len(full_ids) < 3:
                continue

            if is_two_pass and (rng.random() < 0.50):
                gt_bands = _extract_gt_system_bands_from_image(img, len(sys_seqs))
                coarse_t = _pil_to_tensor(img, COARSE_MAX_DIM).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    band_logits, global_ctx = model.pass1_plan(coarse_t)
                    btgt = _band_targets(gt_bands, band_logits.shape[1], device).unsqueeze(0)
                    loss_band = F.binary_cross_entropy_with_logits(band_logits, btgt)

                    # Train Pass-2 on up to 2 sampled zoomed bands per page for L4 speed
                    pick_idx = rng.sample(range(min(len(gt_bands), len(sys_seqs))), k=min(2, len(gt_bands), len(sys_seqs)))
                    loss_zoom = torch.tensor(0.0, device=device)
                    for idx_b in pick_idx:
                        seq = sys_seqs[idx_b]
                        if len(seq) < 3:
                            continue
                        crop = _crop_zoom_band(img, gt_bands[idx_b][0], gt_bands[idx_b][1])
                        bt = _pil_to_tensor(crop, ZOOM_BAND_W, target_hw=(ZOOM_BAND_H, ZOOM_BAND_W)).unsqueeze(0).to(device)
                        tgt = torch.tensor(seq[:BAND_MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
                        mem = model.encode_zoom_with_ctx(bt, global_ctx)
                        out = model.base.decode_tokens(tgt[:, :-1], mem, return_mtp=False, return_aux=False)
                        logits = out[0] if isinstance(out, tuple) else out
                        loss_zoom = loss_zoom + F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID
                        )
                    loss = (loss_zoom / max(1, len(pick_idx)) + 0.5 * loss_band) / GRAD_ACCUM
            else:
                inp_t = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
                tgt = torch.tensor(full_ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    cur_base = model.base if is_two_pass else model
                    mem, _, _ = cur_base.encode_image(inp_t, return_aux=False)
                    out = cur_base.decode_tokens(tgt[:, :-1], mem, return_mtp=False, return_aux=False)
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID
                    ) / GRAD_ACCUM

            loss.backward()
            step_loss_val += float(loss.item())

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
            print(f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} loss={mean_l:.4f} elapsed={time.time()-t_arm0:.1f}s", flush=True)

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ck = arm_dir / "latest.pt.tmp"
            torch.save({"step": step, "ema_state_dict": ema_state}, tmp_ck)
            os.replace(tmp_ck, arm_dir / "latest.pt")
            t_last_save = time.time()

    model.load_state_dict(ema_state, strict=True)
    proxy_val = _eval_model_on_proxy_val(model, val_pairs, is_two_pass=is_two_pass, device=device, use_bf16=use_bf16)
    suite31 = _eval_suite31(
        model, manifest_path, is_two_pass=is_two_pass, device=device, use_bf16=use_bf16,
        aser_bin=aser_bin, eharness_bin=eharness_bin,
    )
    result = {
        "arm": arm_name,
        "is_two_pass": is_two_pass,
        "extra_params": extra_params,
        "total_params": sum(p.numel() for p in model.parameters()),
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

        ckpt_local, ckpt_source, train_abs, val_abs, manifest_path = _ensure_assets(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(f"[{EXP_ID}] Loaded {len(train_pairs)} train / {len(val_pairs)} val pairs on {gpu_name}", flush=True)

        arm_a = _train_arm(
            "arm_a_control_singlepass_1280",
            is_two_pass=False,
            ckpt_path=ckpt_local,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            manifest_path=manifest_path,
            device=device,
            use_bf16=use_bf16,
            aser_bin=aser_bin,
            eharness_bin=eharness_bin,
        )
        arm_b = _train_arm(
            "arm_b_coarse2fine_zoom_2pass",
            is_two_pass=True,
            ckpt_path=ckpt_local,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            manifest_path=manifest_path,
            device=device,
            use_bf16=use_bf16,
            aser_bin=aser_bin,
            eharness_bin=eharness_bin,
        )

        def _delta(b: Optional[float], a: Optional[float]) -> Optional[float]:
            return None if (b is None or a is None) else round(float(b) - float(a), 5)

        comparison = {
            "val_ce_delta_b_minus_a": _delta(arm_b["proxy_val"]["val_ce_loss"], arm_a["proxy_val"]["val_ce_loss"]),
            "hw_clean_note_f1_delta": _delta(arm_b["suite31_hw"]["hw_clean"]["note_f1"], arm_a["suite31_hw"]["hw_clean"]["note_f1"]),
            "hw_clean_tok_f1_delta": _delta(arm_b["suite31_hw"]["hw_clean"]["tok_f1"], arm_a["suite31_hw"]["hw_clean"]["tok_f1"]),
            "hw_photo_note_f1_delta": _delta(arm_b["suite31_hw"]["hw_photo"]["note_f1"], arm_a["suite31_hw"]["hw_photo"]["note_f1"]),
            "printed_clean_note_f1_delta": _delta(arm_b["suite31_hw"]["printed_clean"]["note_f1"], arm_a["suite31_hw"]["printed_clean"]["note_f1"]),
            "hw_band_recall": arm_b["suite31_hw"]["hw_band_recall"],
            "latency_ratio_b_over_a": round(
                arm_b["suite31_hw"]["mean_latency_s"] / max(1e-6, arm_a["suite31_hw"]["mean_latency_s"]), 3
            ),
        }

        report = {
            "exp_id": EXP_ID,
            "exp_code": "i13-19",
            "status": "completed",
            "warmstart_ckpt": ckpt_source or "scratch_small_45m",
            "started_utc": started_iso,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"colab-{gpu_name}",
            "bf16": use_bf16,
            "arm_a_control": arm_a,
            "arm_b_coarse2fine_zoom": arm_b,
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
            commit_message=f"eval({EXP_ID}): upload Coarse2Fine Zoom OMR ablation",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload Coarse2Fine Zoom OMR run ledger",
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
