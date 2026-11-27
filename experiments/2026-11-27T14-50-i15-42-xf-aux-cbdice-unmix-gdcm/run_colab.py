#!/usr/bin/env python3
"""Colab A100 / Local RTX 4080 SUPER runner — Iter 15 Exp i15-42: Aux cbDice + InkUnmix + GDCM.

Experiment ID: 2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm
Cross-field origin:
  - Paleography: Early-stem 2-channel recto-ink vs background/bleed-through unmixing head
  - Medical Vascular Imaging: cbDice soft-skeleton thin-line preservation loss on StaffSegHead
  - HMER (SemiHMER): Global Dynamic Counting Module (GDCM) predicting per-page symbol/structure counts

Controlled 2-arm proxy on edge85m + ax0_gl6 (600 steps/arm, BF16, seed=42):
  - Arm 0 (ctrl):                edge85m + ax0_gl6 stock CE + MTP-4 + router saliency.
  - Arm 1 (aux_topo_unmix_gdcm): Identical inference architecture + 3 training-only auxiliary heads
                                 stripped cleanly via strip_aux_heads() -> EXACT 0.000 MB INT4 delta!

Measures per arm:
  - raw_val_ce, ema_val_ce, int4_dequant_val_ce
  - step_ms, train_elapsed_sec, peak_vram_mb
  - exact serialized packed_int4_mb (verified exact 0.000 MB stripped delta vs ctrl)

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

EXP_ID = "2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm"
EXP_CODE = "i15-42"
ARMS = ("ctrl", "aux_topo_unmix_gdcm")
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_CKPT_PREFIX = f"checkpoints/{EXP_ID}"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
_DEFAULT_WORK = "/vol/i15_42" if Path("/vol").is_dir() else "/content/i15_42"
WORK_ROOT = Path(os.environ.get("WORK_ROOT", _DEFAULT_WORK))

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

AUX_UNMIX_WEIGHT = 0.10
AUX_CBDICE_WEIGHT = 0.10
AUX_GDCM_WEIGHT = 0.10
GDCM_NUM_CLASSES = 6


# ---------------------------------------------------------------------------
# Self-Contained Training-Only Aux Modules & Losses (0.000 MB after stripping)
# ---------------------------------------------------------------------------

def _build_aux_heads_and_losses() -> Tuple[Any, Any, Any, Any, Any]:
    """Factory returning (InkUnmixHead, GDCMCountHead, compute_ink_unmix_loss,
    compute_cbdice_staff_loss, compute_gdcm_count_loss)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import uni_omr
    import uni_omr_v2

    class InkUnmixHead(nn.Module):
        """Paleography-inspired training-only 2-channel ink-vs-background/bleed-through unmixing head on P2."""

        def __init__(self, in_ch: int, mid_ch: int = 64) -> None:
            super().__init__()
            g = uni_omr_v2._group_count(mid_ch, 8)
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(g, mid_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(mid_ch, 2, kernel_size=1, bias=True),
            )

        def forward(self, p2: torch.Tensor) -> torch.Tensor:
            """Returns (B, 2, H/4, W/4): ch0=primary foreground ink logits, ch1=bg/bleed-through logits."""
            return self.net(p2)

    class GDCMCountHead(nn.Module):
        """SemiHMER-inspired training-only Global Dynamic Counting Module over pooled visual memory."""

        def __init__(self, d_model: int, n_classes: int = GDCM_NUM_CLASSES) -> None:
            super().__init__()
            self.d_model = int(d_model)
            self.n_classes = int(n_classes)
            mid = max(64, d_model // 2)
            self.norm = uni_omr_v2.RMSNorm(d_model)
            self.queries = nn.Parameter(torch.randn(1, n_classes, d_model) * 0.02)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.mlp = nn.Sequential(
                nn.Linear(d_model * 2, mid, bias=False),
                nn.SiLU(inplace=True),
                nn.Linear(mid, 1, bias=True),
            )
            self._inv_sqrt_d = 1.0 / math.sqrt(float(d_model))

        def forward(self, memory: torch.Tensor) -> torch.Tensor:
            """Args: memory (B, L, D) -> returns (B, n_classes) non-negative predicted counts."""
            B = memory.shape[0]
            m = self.norm(memory)
            k = self.k_proj(m)  # (B, L, D)
            v = self.v_proj(m)  # (B, L, D)
            q = self.queries.expand(B, -1, -1).to(m.dtype)  # (B, K, D)
            attn_logits = torch.bmm(q, k.transpose(1, 2)) * self._inv_sqrt_d  # (B, K, L)
            attn = F.softmax(attn_logits.float(), dim=-1).to(m.dtype)
            ctx = torch.bmm(attn, v)  # (B, K, D)
            g_mean = m.mean(dim=1, keepdim=True).expand(-1, self.n_classes, -1)
            feat = torch.cat([ctx, g_mean], dim=-1)  # (B, K, 2D)
            return F.softplus(self.mlp(feat).squeeze(-1))  # (B, K)

    @torch.no_grad()
    def _extract_ink_layers(images: torch.Tensor, target_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Derive soft foreground ink mask and background/bleed-through mask from input images."""
        # images are normalized to [-1, 1] -> map to [0, 1]
        img01 = (images.float() * 0.5 + 0.5).clamp(0.0, 1.0)
        if img01.shape[1] == 3:
            luma = img01[:, 0:1] * 0.299 + img01[:, 1:2] * 0.587 + img01[:, 2:3] * 0.114
        else:
            luma = img01.mean(dim=1, keepdim=True)
        local_bg = F.avg_pool2d(
            F.avg_pool2d(luma, kernel_size=(1, 31), stride=1, padding=(0, 15), count_include_pad=False),
            kernel_size=(31, 1),
            stride=1,
            padding=(15, 0),
            count_include_pad=False,
        )
        contrast = ((local_bg - luma) / local_bg.clamp(min=0.15)).clamp(0.0, 1.0)
        fg_px = (contrast > 0.14).float()
        bleed_or_bg_px = (1.0 - fg_px).clamp(0.0, 1.0)
        fg_max = F.adaptive_max_pool2d(fg_px, target_hw)
        fg_avg = F.adaptive_avg_pool2d(fg_px, target_hw)
        fg_tgt = (0.5 * fg_max + 0.5 * fg_avg).clamp(0.0, 1.0)
        bg_tgt = F.adaptive_avg_pool2d(bleed_or_bg_px, target_hw).clamp(0.0, 1.0)
        return fg_tgt, bg_tgt

    def compute_ink_unmix_loss(unmix_logits: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """2-channel BCE unmixing loss supervising primary ink vs background/bleed-through."""
        target_hw = (int(unmix_logits.shape[2]), int(unmix_logits.shape[3]))
        fg_tgt, bg_tgt = _extract_ink_layers(images, target_hw)
        tgt = torch.cat([fg_tgt, bg_tgt], dim=1)  # (B, 2, H/4, W/4)
        loss = F.binary_cross_entropy_with_logits(unmix_logits.float(), tgt.float())
        return loss.to(unmix_logits.dtype)

    def _soft_erode(x: torch.Tensor) -> torch.Tensor:
        p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
        return torch.minimum(p1, p2)

    def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    def _soft_open(x: torch.Tensor) -> torch.Tensor:
        return _soft_dilate(_soft_erode(x))

    def _soft_skeletonize(x: torch.Tensor, iters: int = 3) -> torch.Tensor:
        """Differentiable morphological soft-skeletonization (Shit et al. clDice / cbDice)."""
        x1 = _soft_open(x)
        skel = F.relu(x - x1)
        cur = x
        for _ in range(iters - 1):
            cur = _soft_erode(cur)
            x1 = _soft_open(cur)
            delta = F.relu(cur - x1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def compute_cbdice_staff_loss(staff_logits: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """cbDice-inspired soft-skeleton thin-line preservation loss + BCE/Dice on StaffSegHead."""
        target_hw = (int(staff_logits.shape[2]), int(staff_logits.shape[3]))
        fg_tgt, _ = _extract_ink_layers(images, target_hw)
        logits_f = staff_logits.float()
        tgt_f = fg_tgt.float()

        bce = F.binary_cross_entropy_with_logits(logits_f, tgt_f)
        probs = torch.sigmoid(logits_f)
        inter = (probs * tgt_f).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + tgt_f.sum(dim=(1, 2, 3)) + 1.0
        soft_dice = 1.0 - ((2.0 * inter + 1.0) / denom).mean()

        # Differentiable centerline/skeleton topology preservation (clDice / cbDice)
        skel_pred = _soft_skeletonize(probs, iters=3)
        skel_tgt = _soft_skeletonize(tgt_f, iters=3)
        tprec = ((skel_pred * tgt_f).sum(dim=(1, 2, 3)) + 1.0) / (skel_pred.sum(dim=(1, 2, 3)) + 1.0)
        tsens = ((skel_tgt * probs).sum(dim=(1, 2, 3)) + 1.0) / (skel_tgt.sum(dim=(1, 2, 3)) + 1.0)
        cl_dice = 1.0 - ((2.0 * tprec * tsens) / (tprec + tsens + 1e-6)).mean()

        return (0.5 * (bce + 0.5 * soft_dice) + 0.5 * cl_dice).to(staff_logits.dtype)

    # Precompute token-ID masks for the 6 GDCM musical count classes
    tok2id = uni_omr.UNI_TOKEN_TO_ID
    bar_ids = [idx for tok, idx in tok2id.items() if tok.startswith("bar") and tok != "bar-rest"]
    note_ids = [tok2id[p] for p in ("A", "B", "C", "D", "E", "F", "G") if p in tok2id]
    rest_ids = [tok2id[r] for r in ("rest", "bar-rest") if r in tok2id]
    acc_ids = [tok2id[a] for a in ("a-2", "a-1", "a1", "a2") if a in tok2id]
    dot_tie_ids = [tok2id[d] for d in ("dot", "tie-start", "tie-stop") if d in tok2id]

    _GDCM_MASK_CACHE: Dict[torch.device, torch.Tensor] = {}

    def _get_gdcm_class_lut(device: torch.device) -> torch.Tensor:
        lut = _GDCM_MASK_CACHE.get(device)
        if lut is None:
            v = int(uni_omr.VOCAB_SIZE)
            m = torch.zeros((v, GDCM_NUM_CLASSES), dtype=torch.float32, device=device)
            if uni_omr.SYS_ID < v:
                m[uni_omr.SYS_ID, 0] = 1.0
            for idx in bar_ids:
                if idx < v:
                    m[idx, 1] = 1.0
            for idx in note_ids:
                if idx < v:
                    m[idx, 2] = 1.0
            for idx in rest_ids:
                if idx < v:
                    m[idx, 3] = 1.0
            for idx in acc_ids:
                if idx < v:
                    m[idx, 4] = 1.0
            for idx in dot_tie_ids:
                if idx < v:
                    m[idx, 5] = 1.0
            _GDCM_MASK_CACHE[device] = m
            lut = m
        return lut

    def compute_gdcm_count_loss(gdcm_pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """SmoothL1Loss in log1p count space over 6 per-page musical classes (zero CPU syncs)."""
        lut = _get_gdcm_class_lut(targets.device)
        safe_ids = targets.clamp(min=0, max=lut.shape[0] - 1)
        valid = (targets != uni_omr.PAD_ID).unsqueeze(-1).float()  # (B, T, 1)
        counts = (lut[safe_ids] * valid).sum(dim=1)  # (B, 6)
        # System count is count(<sys>) + 1
        counts[:, 0] = counts[:, 0] + 1.0
        loss = F.smooth_l1_loss(
            torch.log1p(gdcm_pred.float()),
            torch.log1p(counts.float()),
            beta=0.25,
        )
        return loss.to(gdcm_pred.dtype)

    return InkUnmixHead, GDCMCountHead, compute_ink_unmix_loss, compute_cbdice_staff_loss, compute_gdcm_count_loss


def apply_aux_topo_unmix_gdcm(model: Any) -> Any:
    """Attach training-only InkUnmixHead + GDCMCountHead and wire StaffSegHead cbDice cleanly."""
    InkUnmixHead, GDCMCountHead, _, _, _ = _build_aux_heads_and_losses()
    c = model.cfg
    mid_ch = 32 if c.d_model <= 128 else 64

    model.ink_unmix_head = InkUnmixHead(model.backbone.p2_ch, mid_ch=mid_ch)
    model.gdcm_head = GDCMCountHead(c.d_model, n_classes=GDCM_NUM_CLASSES)

    orig_strip = model.strip_aux_heads

    def _strip_aux_heads_ext(self: Any, strip_mtp: bool = False) -> Any:
        orig_strip(strip_mtp=strip_mtp)
        self.ink_unmix_head = None
        self.gdcm_head = None
        return self

    def _encode_image_ext(
        self: Any,
        images: Any,
        keep_ratio_override: Optional[float] = None,
        return_aux: Optional[bool] = None,
    ) -> Any:
        want_aux = self.training if return_aux is None else bool(return_aux)
        need_pyramid = want_aux and (
            (self.staff_seg_head is not None)
            or (self.stroke_hog_head is not None)
            or (self.ink_unmix_head is not None)
        )
        if need_pyramid:
            feat_2d, p2, p3 = self.backbone(images, return_pyramid=True)
        else:
            feat_2d = self.backbone(images, return_pyramid=False)
            p2 = p3 = None

        mem_levels, router_info = self.encoder(
            feat_2d, images=images, keep_ratio_override=keep_ratio_override
        )
        mem_final = mem_levels[-1]

        aux_enc: Dict[str, Any] = {
            "sys_count_pred": None,
            "staff_seg_logits": None,
            "stroke_hog_pred": None,
            "ink_unmix_logits": None,
            "gdcm_counts": None,
        }
        if want_aux:
            if self.staff_seg_head is not None and p2 is not None and p3 is not None:
                aux_enc["staff_seg_logits"] = self.staff_seg_head(p2, p3)
            if self.ink_unmix_head is not None and p2 is not None:
                aux_enc["ink_unmix_logits"] = self.ink_unmix_head(p2)
            if self.gdcm_head is not None:
                aux_enc["gdcm_counts"] = self.gdcm_head(mem_final)

        return mem_levels, aux_enc, router_info

    def _forward_ext(
        self: Any,
        images: Any,
        tgt_ids: Any,
        keep_ratio_override: Optional[float] = None,
        return_aux: Optional[bool] = None,
        return_mtp: Optional[bool] = None,
    ) -> Dict[str, Any]:
        mem, aux_enc, router_info = self.encode_image(
            images, keep_ratio_override=keep_ratio_override, return_aux=return_aux
        )
        logits, mtp_logits, sys_y_pred = self.decode_tokens(
            tgt_ids, mem, return_mtp=return_mtp, return_aux=False
        )
        return {
            "logits": logits,
            "mtp_logits": mtp_logits,
            "sys_y_pred": sys_y_pred,
            "sys_count_pred": aux_enc.get("sys_count_pred"),
            "staff_seg_logits": aux_enc.get("staff_seg_logits"),
            "stroke_hog_pred": aux_enc.get("stroke_hog_pred"),
            "ink_unmix_logits": aux_enc.get("ink_unmix_logits"),
            "gdcm_counts": aux_enc.get("gdcm_counts"),
            "router_info": router_info,
        }

    model.strip_aux_heads = types.MethodType(_strip_aux_heads_ext, model)
    model.encode_image = types.MethodType(_encode_image_ext, model)
    model.forward = types.MethodType(_forward_ext, model)
    return model


def build_arm_model(arm_name: str, grad_checkpoint: bool = True, tiny: bool = False) -> Any:
    """Build edge85m + ax0_gl6 (or tiny d=64 for smoke test) model for 'ctrl' or 'aux_topo_unmix_gdcm'."""
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
    if arm_name == "aux_topo_unmix_gdcm":
        apply_aux_topo_unmix_gdcm(model)
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
    """Strip ALL training-only aux/MTP heads, pack QAT-INT4 Linear weights into 4-bit nibbles +
    FP16 scales, write exact binary file to disk, and dequantize onto a fresh model copy."""
    import torch
    import torch.nn as nn

    infer_model = build_arm_model(arm_name, grad_checkpoint=False, tiny=tiny).cpu().eval()
    infer_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True)
    infer_model.strip_aux_heads(strip_mtp=True)
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

    _, _, compute_ink_unmix_loss, compute_cbdice_staff_loss, compute_gdcm_count_loss = _build_aux_heads_and_losses()

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
    if device.type == "cuda":
        assert opt.mode == "muon+adamw", f"[{EXP_ID}][{arm_name}] Expected muon+adamw, got {opt.mode}"
    print(f"[{EXP_ID}][{arm_name}] Optimizer verified: mode={opt.mode}", flush=True)

    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    weights = [p[3] for p in train_pairs]
    rng = random.Random(SEED)
    start_step = 1
    loss_hist: List[float] = []
    step_ms_hist: List[float] = []
    aux_loss_hist: Dict[str, List[float]] = {"unmix": [], "cbdice": [], "gdcm": []}

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
        print(f"[{EXP_ID}][{arm_name}] Resumed local checkpoint at step {start_step}/{TRAIN_STEPS}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_arm0 = time.perf_counter()
    t_last_save = time.time()

    use_aux = arm_name == "aux_topo_unmix_gdcm"
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
        step_unmix = 0.0
        step_cbdice = 0.0
        step_gdcm = 0.0
        for _ in range(GRAD_ACCUM):
            batch_samples = rng.choices(train_pairs, weights=weights, k=BATCH_SIZE)
            collated = _collate_microbatch(batch_samples, device)
            if collated is None:
                continue
            inp, tgt = collated
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(inp, tgt[:, :-1], return_aux=use_aux, return_mtp=True)
                losses = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)
                total_loss = losses["loss"]
                if use_aux:
                    l_unmix = compute_ink_unmix_loss(out["ink_unmix_logits"], inp)
                    l_cbdice = compute_cbdice_staff_loss(out["staff_seg_logits"], inp)
                    l_gdcm = compute_gdcm_count_loss(out["gdcm_counts"], tgt[:, 1:])
                    total_loss = (
                        total_loss
                        + AUX_UNMIX_WEIGHT * l_unmix.to(total_loss.dtype)
                        + AUX_CBDICE_WEIGHT * l_cbdice.to(total_loss.dtype)
                        + AUX_GDCM_WEIGHT * l_gdcm.to(total_loss.dtype)
                    )
                    step_unmix += float(l_unmix.item()) / GRAD_ACCUM
                    step_cbdice += float(l_cbdice.item()) / GRAD_ACCUM
                    step_gdcm += float(l_gdcm.item()) / GRAD_ACCUM
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
        if use_aux:
            aux_loss_hist["unmix"].append(step_unmix)
            aux_loss_hist["cbdice"].append(step_cbdice)
            aux_loss_hist["gdcm"].append(step_gdcm)

        if step % 100 == 0 or step == 1:
            mean_l = sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:]))
            mean_ms = sum(step_ms_hist[-50:]) / max(1, len(step_ms_hist[-50:]))
            extra = ""
            if use_aux:
                mu = sum(aux_loss_hist["unmix"][-50:]) / max(1, len(aux_loss_hist["unmix"][-50:]))
                mc = sum(aux_loss_hist["cbdice"][-50:]) / max(1, len(aux_loss_hist["cbdice"][-50:]))
                mg = sum(aux_loss_hist["gdcm"][-50:]) / max(1, len(aux_loss_hist["gdcm"][-50:]))
                extra = f" unmix={mu:.4f} cbdice={mc:.4f} gdcm={mg:.4f}"
            print(
                f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} "
                f"loss={mean_l:.4f} step_ms={mean_ms:.1f}{extra} elapsed={time.perf_counter() - t_arm0:.1f}s",
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
    if use_aux:
        result["aux_losses_tail50"] = {
            k: round(sum(v[-50:]) / max(1, len(v[-50:])), 5)
            for k, v in aux_loss_hist.items()
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
# CPU 2-Step Smoke Test
# ---------------------------------------------------------------------------

def run_smoke_test() -> int:
    """2-step tiny CPU forward+backward+step verifying finite grads on all 3 aux heads and
    EXACT 0.000 MB stripped INT4 delta on edge85m."""
    _locate_mvd()
    import tempfile
    import torch
    import train_uni_v2
    import uni_omr

    _, _, compute_ink_unmix_loss, compute_cbdice_staff_loss, compute_gdcm_count_loss = _build_aux_heads_and_losses()

    device = torch.device("cpu")
    _seed_everything(SEED)

    tiny_results: Dict[str, Dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as tmpd:
        for arm_name in ARMS:
            use_aux = arm_name == "aux_topo_unmix_gdcm"
            model = build_arm_model(arm_name, grad_checkpoint=False, tiny=True).to(device)
            opt = train_uni_v2.build_optimizer_v2(
                model, optimizer_name="muon", lr=LR_MAX, weight_decay=WEIGHT_DECAY,
                device_type="cpu", allow_cpu_muon=True, no_decay_1d=True,
            )
            assert opt.mode == "muon+adamw", f"{arm_name}: expected muon+adamw, got {opt.mode}"
            model.train()

            losses_seen: List[float] = []
            aux_seen: Dict[str, float] = {}
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                # Synthetic high-contrast page image (white background + dark horizontal lines/blobs)
                imgs = torch.ones(1, 3, 64, 96, device=device) * 0.85
                imgs[:, :, 16:18, 8:88] = -0.8
                imgs[:, :, 24:26, 8:88] = -0.8
                imgs[:, :, 40:42, 8:88] = -0.8
                imgs[:, :, 14:28, 30:34] = -0.7
                tgt = torch.randint(4, min(64, uni_omr.VOCAB_SIZE), (1, 14), device=device)
                tgt[:, 0] = uni_omr.BOS_ID
                tgt[:, 6] = uni_omr.SYS_ID

                out = model(imgs, tgt[:, :-1], return_aux=use_aux, return_mtp=True)
                loss = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)["loss"]
                if use_aux:
                    l_unmix = compute_ink_unmix_loss(out["ink_unmix_logits"], imgs)
                    l_cbdice = compute_cbdice_staff_loss(out["staff_seg_logits"], imgs)
                    l_gdcm = compute_gdcm_count_loss(out["gdcm_counts"], tgt[:, 1:])
                    for lname, lval in (("unmix", l_unmix), ("cbdice", l_cbdice), ("gdcm", l_gdcm)):
                        assert torch.isfinite(lval), f"{arm_name} step {step}: non-finite {lname} loss {lval}"
                        aux_seen[lname] = round(float(lval.item()), 4)
                    loss = (
                        loss
                        + AUX_UNMIX_WEIGHT * l_unmix.to(loss.dtype)
                        + AUX_CBDICE_WEIGHT * l_cbdice.to(loss.dtype)
                        + AUX_GDCM_WEIGHT * l_gdcm.to(loss.dtype)
                    )
                assert torch.isfinite(loss), f"{arm_name} step {step}: non-finite loss {loss}"
                loss.backward()

                assert out["logits"].shape == (1, 13, uni_omr.VOCAB_SIZE)
                if use_aux:
                    for hname, head in (
                        ("ink_unmix_head", model.ink_unmix_head),
                        ("staff_seg_head", model.staff_seg_head),
                        ("gdcm_head", model.gdcm_head),
                    ):
                        assert head is not None, f"{hname} missing"
                        grads = [p.grad for p in head.parameters() if p.requires_grad]
                        assert all(g is not None and torch.isfinite(g).all() for g in grads), f"{hname} bad grad"
                        assert sum(g.abs().sum().item() for g in grads) > 0.0, f"{hname} zero grad"
                opt.step()
                losses_seen.append(round(float(loss.item()), 4))

            # Verify KV-cached generate() and dequantized INT4 parity after stripping all aux heads
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
            assert dq_logits.shape == (1, 13, uni_omr.VOCAB_SIZE) and torch.isfinite(dq_logits).all()
            assert seq.shape[0] == 1 and seq.shape[1] >= 2

            tiny_results[arm_name] = {
                "opt_mode": opt.mode,
                "losses_2step": losses_seen,
                "tiny_train_params": sum(p.numel() for p in model.parameters()),
                "tiny_stripped_bytes": pack_tiny["disk_bytes"],
                **({"aux_losses_step2": aux_seen} if use_aux else {}),
            }

        # Exact edge85m packed_int4_mb verification — must be EXACT 0.000 MB / 0 bytes delta!
        edge_mb: Dict[str, float] = {}
        edge_bytes: Dict[str, int] = {}
        for arm_name in ARMS:
            m_edge = build_arm_model(arm_name, grad_checkpoint=False, tiny=False)
            pack_edge = _pack_and_verify_hybrid_int4(
                m_edge, arm_name, Path(tmpd) / f"{arm_name}_edge85m.bin", INT4_GROUP_SIZE, tiny=False
            )
            pack_edge.pop("dequant_model")
            assert pack_edge["passes_55mb_budget"], f"{arm_name} exceeded 55 MB budget: {pack_edge['packed_int4_mb']}"
            edge_mb[arm_name] = pack_edge["packed_int4_mb"]
            edge_bytes[arm_name] = pack_edge["disk_bytes"]
            del m_edge

    delta_bytes = edge_bytes["aux_topo_unmix_gdcm"] - edge_bytes["ctrl"]
    delta_mb = round(edge_mb["aux_topo_unmix_gdcm"] - edge_mb["ctrl"], 3)
    assert delta_bytes == 0, f"Expected exact 0-byte stripped INT4 delta, got {delta_bytes} bytes!"
    print(f"[{EXP_CODE}][SMOKE] tiny ctrl:                {tiny_results['ctrl']}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] tiny aux_topo_unmix_gdcm: {tiny_results['aux_topo_unmix_gdcm']}", flush=True)
    print(
        f"[{EXP_CODE}][SMOKE] edge85m packed_int4_mb: ctrl={edge_mb['ctrl']} MB ({edge_bytes['ctrl']} B), "
        f"aux_topo_unmix_gdcm={edge_mb['aux_topo_unmix_gdcm']} MB ({edge_bytes['aux_topo_unmix_gdcm']} B), "
        f"delta={delta_mb:+.3f} MB ({delta_bytes:+d} B — EXACT ZERO PASS)",
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
            if "ctrl" in arm_results and "aux_topo_unmix_gdcm" in arm_results:
                c = arm_results["ctrl"]
                t = arm_results["aux_topo_unmix_gdcm"]
                eval_payload["delta_aux_topo_unmix_gdcm_minus_ctrl"] = {
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
