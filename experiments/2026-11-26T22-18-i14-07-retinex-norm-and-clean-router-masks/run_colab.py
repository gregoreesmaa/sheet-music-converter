#!/usr/bin/env python3
"""Wave 11 Exp i14-07 (Theorist 4 ALS Cure) — Modal/Colab L4 Runner:
< 0.5M Differentiable Retinex Background Normalizer (I / (B(x,y) + eps))
+ Clean-Render Ink-Mask Supervision for InkTokenRouterV2 on Sepia / Dark-Gutter /
Bleed-Through Archival Scans.

Experiment ID: 2026-11-26T22-18-i14-07-retinex-norm-and-clean-router-masks
Controlled 2-Arm Ablation (BF16, seed=42, 600 steps/arm, grad_accum=4):
  - Arm A (arm_ctrl_standard_router): Standard UniOMR v2 with raw RGB input
    and InkTokenRouterV2 self-computed ink prior on degraded images.
  - Arm B (arm_retinex_clean_router_mask): UniOMR v2 + RetinexBackgroundNormalizer
    (~0.14M params, < 0.5M budget, zero-init residual gate -> exact identity at step 0)
    + Clean-Render Ink-Mask BCE supervision (w=0.35) on InkTokenRouterV2 saliency_logits.

Uploads:
  - evals/2026-11-26T22-18-i14-07-retinex-norm-and-clean-router-masks.json
  - runs/2026-11-26T22-18-i14-07-retinex-norm-and-clean-router-masks.json
to gregoreesmaa/sheet-music-converter-omr.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T22-18-i14-07-retinex-norm-and-clean-router-masks"
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
CACHE_ROOT = Path("/content/cache")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

SEED = 42
TRAIN_STEPS = 600
WARMUP_STEPS = 60
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
GRAD_ACCUM = 4
LR_BASE = 3e-5
LR_RETINEX = 3e-4
LR_GATE = 2e-3
WEIGHT_DECAY = 0.04
CLEAN_MASK_BCE_WEIGHT = 0.35
ALS_AUG_PROB = 0.65
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
SUITE31_MAX_LEN = 2560
SAVE_EVERY_MINUTES = 5.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_save_torch(path: Path, payload: Dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
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
        raise RuntimeError("HF token not found in HF_TOKEN env var or /tmp/.hf_token")
    return token


def _locate_mvd() -> Path:
    candidates = (
        Path("/workspace/mvd"),
        Path("/content/mvd"),
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path("/content/workspace/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
    )
    for mvd in candidates:
        if (mvd / "uni_omr_v2.py").is_file():
            if str(mvd) not in sys.path:
                sys.path.insert(0, str(mvd))
            return mvd

    for bundle in (Path("/content/ov_mvd_bundle.tar.gz"), Path("/content/bundle.tar.gz"), Path("/content/repo_bundle.tar.gz")):
        if bundle.is_file():
            extract_root = Path("/content/workspace")
            extract_root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=extract_root)
            for mvd in (
                extract_root / "sheet-music-converter" / "tmp-train" / "mvd",
                extract_root / "tmp-train" / "mvd",
                extract_root / "mvd",
            ):
                if (mvd / "uni_omr_v2.py").is_file():
                    if str(mvd) not in sys.path:
                        sys.path.insert(0, str(mvd))
                    return mvd

    raise FileNotFoundError("Could not locate tmp-train/mvd/uni_omr_v2.py")


def _ensure_packages() -> None:
    needed = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)


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


def _ensure_assets(hf_token: str) -> Tuple[Optional[Path], Optional[str], Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    VOL_ROOT.mkdir(parents=True, exist_ok=True)
    ds_root = VOL_ROOT / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)
    ckpt_root = CACHE_ROOT / "pretrained"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    ckpt_local: Optional[Path] = None
    ckpt_source: Optional[str] = None
    for cand in HF_CKPT_CANDIDATES:
        target = ckpt_root / Path(cand).name
        if target.is_file():
            ckpt_local, ckpt_source = target, cand
            break
        try:
            dl_p = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=cand,
                repo_type="dataset",
                token=hf_token,
                local_dir=str(CACHE_ROOT / "hf_dl"),
            )
            shutil.copy2(dl_p, target)
            ckpt_local, ckpt_source = target, cand
            print(f"[{EXP_ID}] Warm-start checkpoint: {cand}", flush=True)
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
            local_dir=str(CACHE_ROOT / "hf_dl"),
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=proxy_dir)
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

    s31_dir = ds_root / "suite31_hw_eval"
    manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json")) if s31_dir.exists() else []
    if not manifests:
        s31_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_SUITE31_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(CACHE_ROOT / "hf_dl"),
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=s31_dir)
        manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json under {s31_dir}")
    return ckpt_local, ckpt_source, train_abs, val_abs, manifests[0]


def _load_manifest_pairs(manifest_path: Path) -> List[Tuple[Path, List[Path], str]]:
    pairs: List[Tuple[Path, List[Path], str]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        img_p = Path(rec["image"])
        bks = [Path(b) for b in rec.get("bekerns", [])]
        domain = str(rec.get("domain", "printed"))
        if img_p.is_file() and bks and all(b.is_file() for b in bks):
            pairs.append((img_p, bks, domain))
    return pairs


def _apply_als_sepia_gutter_bleed(img_t: "torch.Tensor", rng: random.Random) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Synthesize low-contrast sepia parchment + dark book gutter + show-through bleed-through.

    Args:
        img_t: Clean (3, H, W) tensor in [0, 1].
    Returns:
        degraded: (3, H, W) tensor in [0, 1].
        gutter_mask: (1, H, W) float mask where 1.0 indicates synthetic gutter/bleed distractor pixels.
    """
    import torch
    import torch.nn.functional as F

    c, h, w = img_t.shape
    out = img_t.clone().float()
    gutter_mask = torch.zeros((1, h, w), dtype=torch.float32)

    # 1. Low-contrast aged sepia tint & lifted ink floor
    ink_floor = rng.uniform(0.18, 0.36)
    paper_r = rng.uniform(0.76, 0.90)
    paper_g = paper_r * rng.uniform(0.86, 0.94)
    paper_b = paper_r * rng.uniform(0.68, 0.82)
    tint = torch.tensor([paper_r, paper_g, paper_b], dtype=torch.float32).view(3, 1, 1)
    out = (out * (1.0 - ink_floor) + ink_floor) * tint

    # 2. Smooth uneven scanner illumination gradient (foxing / vignetting)
    ys = torch.linspace(-1.0, 1.0, h).view(1, h, 1)
    xs = torch.linspace(-1.0, 1.0, w).view(1, 1, w)
    gx = rng.uniform(-0.18, 0.18)
    gy = rng.uniform(-0.15, 0.15)
    rad = rng.uniform(0.08, 0.22)
    illum = (1.0 + gx * xs + gy * ys - rad * (xs * xs + ys * ys)).clamp(0.45, 1.10)
    out = out * illum

    # 3. Dark book-binding gutter shadow along left or right edge (or interior spine)
    side = rng.choice(("left", "right", "both"))
    gutter_w = max(8, int(w * rng.uniform(0.04, 0.11)))
    gutter_depth = rng.uniform(0.35, 0.72)
    ramp = torch.linspace(0.0, 1.0, gutter_w).view(1, 1, gutter_w)
    shadow_prof = 1.0 - gutter_depth * torch.exp(-3.2 * ramp)
    if side in ("left", "both"):
        out[:, :, :gutter_w] *= shadow_prof
        gutter_mask[:, :, : int(gutter_w * 0.65)] = 1.0
    if side in ("right", "both"):
        out[:, :, -gutter_w:] *= torch.flip(shadow_prof, dims=[2])
        gutter_mask[:, :, -int(gutter_w * 0.65) :] = 1.0

    # 4. Show-through bleed-through from mirrored / shifted verso page
    if rng.random() < 0.80:
        verso = torch.flip(img_t.float(), dims=[2])
        shift_y = rng.randint(-max(4, h // 40), max(4, h // 40))
        verso = torch.roll(verso, shifts=shift_y, dims=1)
        verso_luma = (0.299 * verso[0] + 0.587 * verso[1] + 0.114 * verso[2]).unsqueeze(0)
        verso_ink = (1.0 - verso_luma).clamp(0.0, 1.0)
        verso_blur = F.avg_pool2d(verso_ink.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
        bleed_alpha = rng.uniform(0.14, 0.30)
        out = out * (1.0 - bleed_alpha * verso_blur)
        gutter_mask = torch.maximum(gutter_mask, (verso_blur > 0.25).float())

    # 5. Subtle sensor grain
    noise_std = rng.uniform(0.01, 0.028)
    out = (out + torch.randn_like(out) * noise_std).clamp(0.0, 1.0)
    return out, gutter_mask


def _build_retinex_normalizer() -> "torch.nn.Module":
    """< 0.5M parameter (~0.14M) differentiable Retinex background normalizer.

    Estimates smooth illumination/parchment field B(x,y) at 1/4 resolution,
    upsamples bilinearly, computes I_norm = clamp(I / (B(x,y) + eps), 0, 1),
    and blends via a zero-initialized residual gate alpha = tanh(g) for exact
    identity initialization at step 0.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RetinexBackgroundNormalizer(nn.Module):
        def __init__(self, hidden: int = 32, eps: float = 0.08) -> None:
            super().__init__()
            self.eps = float(eps)
            self.down1 = nn.Sequential(
                nn.Conv2d(3, hidden, kernel_size=5, stride=2, padding=2, bias=False),
                nn.GroupNorm(8, hidden),
                nn.SiLU(inplace=True),
            )
            self.down2 = nn.Sequential(
                nn.Conv2d(hidden, hidden * 2, kernel_size=5, stride=2, padding=2, bias=False),
                nn.GroupNorm(8, hidden * 2),
                nn.SiLU(inplace=True),
            )
            self.mid = nn.Sequential(
                nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                nn.GroupNorm(8, hidden * 2),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                nn.GroupNorm(8, hidden * 2),
                nn.SiLU(inplace=True),
            )
            self.head = nn.Sequential(
                nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 3, kernel_size=1, bias=True),
            )
            # Zero-init residual gate: step-0 forward is exact identity I_out == I
            self.res_gate = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            h, w = x.shape[-2], x.shape[-1]
            f = self.mid(self.down2(self.down1(x)))
            bg_low = torch.sigmoid(self.head(f)) * 0.85 + 0.15  # B(x,y) in [0.15, 1.0]
            bg = F.interpolate(bg_low, size=(h, w), mode="bilinear", align_corners=False)
            x_norm = (x / (bg + self.eps)).clamp(0.0, 1.0)
            alpha = torch.tanh(self.res_gate).to(x.dtype)
            out = (x + alpha * (x_norm - x)).clamp(0.0, 1.0)
            return out, bg

    return RetinexBackgroundNormalizer(hidden=32, eps=0.08)


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


def _select_eval_pieces(pieces: Sequence[Dict[str, Any]], n_printed: int = 12) -> List[Dict[str, Any]]:
    hw = [p for p in pieces if p["handwritten"]]
    pr = [p for p in pieces if not p["handwritten"]]
    by_stratum: Dict[str, List[Dict[str, Any]]] = {}
    for p in sorted(pr, key=lambda x: x["slug"]):
        by_stratum.setdefault(str(p.get("stratum", "other")), []).append(p)
    strata = sorted(by_stratum.keys())
    chosen_pr: List[Dict[str, Any]] = []
    idx = 0
    while len(chosen_pr) < min(int(n_printed), len(pr)):
        progressed = False
        for s in strata:
            bucket = by_stratum[s]
            if idx < len(bucket) and len(chosen_pr) < int(n_printed):
                chosen_pr.append(bucket[idx])
                progressed = True
        if not progressed:
            break
        idx += 1
    return hw + chosen_pr


def _load_image_tensor(path: Path, max_dim: int) -> "torch.Tensor":
    import numpy as np
    from PIL import Image
    import torch

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, float(max_dim) / float(max(w, h)))
        nw = max(32, int(round(w * scale / 16.0)) * 16)
        nh = max(32, int(round(h * scale / 16.0)) * 16)
        if (nw, nh) != (w, h):
            im = im.resize((nw, nh), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _encode_bekern_files(
    bekern_paths: Sequence[Path],
    eval_s31: Any,
    uni_omr: Any,
    vocab_size: int,
    max_len: int,
) -> "torch.Tensor":
    import torch

    _, bos_id, eos_id, sys_id, tok_to_id, _ = eval_s31._vocab_spec(int(vocab_size))
    ids: List[int] = [bos_id]
    for i, bp in enumerate(bekern_paths):
        if i > 0:
            ids.append(sys_id)
        txt = bp.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            continue
        for tok in txt.split():
            if tok == uni_omr.SYS_TOKEN:
                ids.append(sys_id)
            else:
                tid = tok_to_id.get(tok)
                if tid is not None:
                    ids.append(tid)
    ids.append(eos_id)
    if len(ids) > max_len:
        ids = ids[: max_len - 1] + [eos_id]
    return torch.tensor(ids, dtype=torch.long)


def _compute_clean_ink_mask_grid(clean_img: "torch.Tensor", h_p: int, w_p: int) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F

    luma = (0.299 * clean_img[:, 0:1] + 0.587 * clean_img[:, 1:2] + 0.114 * clean_img[:, 2:3]).float()
    ink_px = ((1.0 - luma) > 0.12).float()
    grid_max = F.adaptive_max_pool2d(ink_px, (h_p, w_p))
    grid_avg = F.adaptive_avg_pool2d(ink_px, (h_p, w_p))
    return ((0.5 * grid_max + 0.5 * grid_avg) > 0.10).float()


def _evaluate_arm(
    model: Any,
    retinex: Optional[Any],
    cfg: Any,
    pieces: Sequence[Dict[str, Any]],
    device: Any,
    *,
    use_bf16: bool,
    eval_s31: Any,
    rlvr_grpo: Any,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    model.eval()
    if retinex is not None:
        retinex.eval()

    buckets: Dict[str, List[float]] = {
        "printed_clean": [],
        "printed_photo": [],
        "hw_clean": [],
        "hw_photo": [],
        "als_stress": [],
    }
    router_recalls: List[float] = []
    gutter_evicts: List[float] = []

    stress_rng = random.Random(SEED + 999)
    for piece in pieces:
        ref_notes = _extract_musicxml_notes(Path(piece["gt"]))
        if not ref_notes:
            continue
        is_hw = bool(piece["handwritten"])
        variants = [("clean", Path(piece["clean"])), ("photo", Path(piece["photo"]))]
        for vname, img_p in variants:
            if not img_p.is_file():
                continue
            img_cpu = _load_image_tensor(img_p, FULLPAGE_MAX_DIM)
            img_t = img_cpu.unsqueeze(0).to(device)
            with torch.no_grad():
                x_in = retinex(img_t)[0] if retinex is not None else img_t
                ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bf16 and device.type == "cuda") else eval_s31._nullcontext()
                with ctx:
                    seq = eval_s31.generate_mtp_spec(
                        model, x_in, max_len=SUITE31_MAX_LEN, use_grammar=True, strict_grammar=False, return_stats=False
                    )
            seq_ids = seq[0].detach().cpu().tolist()
            pred_raw = eval_s31.decode_uni_compat(seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True)
            hyp_notes = rlvr_grpo.extract_pitch_dur_notes(pred_raw)
            f1 = float(rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes))
            bkey = f"{'hw' if is_hw else 'printed'}_{vname}"
            buckets[bkey].append(f1)

            # Synthetic ALS Sepia + Dark Gutter + Bleed-Through stress test on clean pages
            if vname == "clean":
                deg_cpu, gut_cpu = _apply_als_sepia_gutter_bleed(img_cpu, stress_rng)
                deg_t = deg_cpu.unsqueeze(0).to(device)
                clean_t = img_cpu.unsqueeze(0).to(device)
                with torch.no_grad():
                    x_als = retinex(deg_t)[0] if retinex is not None else deg_t
                    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bf16 and device.type == "cuda") else eval_s31._nullcontext()
                    with ctx:
                        _, _, rinfo = model.encode_image(x_als, return_aux=False)
                        seq_als = eval_s31.generate_mtp_spec(
                            model, x_als, max_len=SUITE31_MAX_LEN, use_grammar=True, strict_grammar=False, return_stats=False
                        )
                # Measure router notation recall & gutter eviction rate
                kept_idx = rinfo.get("kept_idx")
                grid_hw = rinfo.get("grid_hw")
                if kept_idx is not None and grid_hw is not None:
                    h_p, w_p = int(grid_hw[0]), int(grid_hw[1])
                    n_tok = h_p * w_p
                    kept_mask = torch.zeros(n_tok, device=device, dtype=torch.float32)
                    kept_mask[kept_idx[0].long().clamp(0, n_tok - 1)] = 1.0
                    clean_ink = _compute_clean_ink_mask_grid(clean_t, h_p, w_p).reshape(-1)
                    gut_grid = (F.adaptive_avg_pool2d(gut_cpu.unsqueeze(0).to(device), (h_p, w_p)).reshape(-1) > 0.35).float()
                    non_ink_gut = gut_grid * (1.0 - clean_ink)
                    if clean_ink.sum().item() > 0:
                        router_recalls.append(float((kept_mask * clean_ink).sum().item() / clean_ink.sum().item()))
                    if non_ink_gut.sum().item() > 0:
                        gutter_evicts.append(float(((1.0 - kept_mask) * non_ink_gut).sum().item() / non_ink_gut.sum().item()))

                pred_als = eval_s31.decode_uni_compat(
                    seq_als[0].detach().cpu().tolist(), vocab_size=int(cfg.vocab_size), skip_special=True
                )
                hyp_als = rlvr_grpo.extract_pitch_dur_notes(pred_als)
                buckets["als_stress"].append(float(rlvr_grpo.compute_sequence_f1(ref_notes, hyp_als)))

    def _mean(xs: List[float]) -> float:
        return round(sum(xs) / max(1, len(xs)), 4) if xs else 0.0

    return {
        "printed_clean_note_f1": _mean(buckets["printed_clean"]),
        "printed_photo_note_f1": _mean(buckets["printed_photo"]),
        "hw_clean_note_f1": _mean(buckets["hw_clean"]),
        "hw_photo_note_f1": _mean(buckets["hw_photo"]),
        "als_stress_note_f1": _mean(buckets["als_stress"]),
        "router_notation_recall": _mean(router_recalls),
        "gutter_eviction_rate": _mean(gutter_evicts),
        "counts": {k: len(v) for k, v in buckets.items()},
    }


def _train_one_arm(
    arm_name: str,
    use_retinex_and_clean_mask: bool,
    ckpt_local: Optional[Path],
    train_pairs: Sequence[Tuple[Path, List[Path], str]],
    eval_pieces: Sequence[Dict[str, Any]],
    run_dir: Path,
    device: Any,
    use_bf16: bool,
    uni_omr: Any,
    uni_v2: Any,
    eval_s31: Any,
    rlvr_grpo: Any,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    arm_done_path = run_dir / f"{arm_name}_result.json"
    if arm_done_path.is_file():
        print(f"[{EXP_ID}] Resuming completed arm {arm_name} from {arm_done_path}", flush=True)
        return json.loads(arm_done_path.read_text(encoding="utf-8"))

    _seed_everything(SEED)
    if ckpt_local is not None and ckpt_local.is_file():
        model, cfg, _ = eval_s31.load_uni_checkpoint(ckpt_local, device)
    else:
        cfg = uni_v2.UniOMRv2Config.small_45m()
        model = uni_v2.UniOMRv2Net(cfg).to(device)

    retinex = _build_retinex_normalizer().to(device) if use_retinex_and_clean_mask else None
    retinex_params = sum(p.numel() for p in retinex.parameters()) if retinex is not None else 0

    param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": LR_BASE}]
    if retinex is not None:
        gate_p = [retinex.res_gate]
        other_p = [p for n, p in retinex.named_parameters() if n != "res_gate"]
        param_groups.append({"params": other_p, "lr": LR_RETINEX})
        param_groups.append({"params": gate_p, "lr": LR_GATE, "weight_decay": 0.0})

    opt = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    ckpt_resume = run_dir / f"{arm_name}_resume.pt"
    start_step = 0
    if ckpt_resume.is_file():
        st = torch.load(str(ckpt_resume), map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        if retinex is not None and "retinex" in st:
            retinex.load_state_dict(st["retinex"], strict=False)
        opt.load_state_dict(st["opt"])
        start_step = int(st.get("step", 0))
        print(f"[{EXP_ID}] Resumed {arm_name} at step {start_step}/{TRAIN_STEPS}", flush=True)

    rng = random.Random(SEED + start_step)
    t_last_save = time.monotonic()
    t0 = time.monotonic()
    losses: List[float] = []
    mask_losses: List[float] = []
    nan_skips = 0

    model.train()
    if retinex is not None:
        retinex.train()

    for step in range(start_step + 1, TRAIN_STEPS + 1):
        mult = _wsd_mult(step, TRAIN_STEPS, WARMUP_STEPS)
        opt.param_groups[0]["lr"] = LR_BASE * mult
        if retinex is not None:
            opt.param_groups[1]["lr"] = LR_RETINEX * mult
            opt.param_groups[2]["lr"] = LR_GATE * mult

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_mask_loss = 0.0
        for _ in range(GRAD_ACCUM):
            img_p, bks, _ = train_pairs[rng.randrange(len(train_pairs))]
            clean_cpu = _load_image_tensor(img_p, FULLPAGE_MAX_DIM)
            tgt_ids = _encode_bekern_files(bks, eval_s31, uni_omr, int(cfg.vocab_size), MAX_SEQ_LEN).unsqueeze(0).to(device)
            if rng.random() < ALS_AUG_PROB:
                inp_cpu, _ = _apply_als_sepia_gutter_bleed(clean_cpu, rng)
            else:
                inp_cpu = clean_cpu

            clean_t = clean_cpu.unsqueeze(0).to(device)
            inp_t = inp_cpu.unsqueeze(0).to(device)
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bf16 and device.type == "cuda") else eval_s31._nullcontext()
            with ctx:
                x_in = retinex(inp_t)[0] if retinex is not None else inp_t
                fwd = model(x_in, tgt_ids[:, :-1], return_aux=False)
                out = model.compute_loss(fwd, tgt_ids[:, 1:])
                ce_loss = out["loss"] if isinstance(out, dict) else out
                total_loss = ce_loss

                if use_retinex_and_clean_mask:
                    # Supervise InkTokenRouterV2 saliency_logits against pre-degradation clean ink mask
                    rinfo = fwd.get("router_info") or {}
                    sal_logits = rinfo.get("saliency_logits")
                    if sal_logits is not None:
                        h_p, w_p = int(sal_logits.shape[-2]), int(sal_logits.shape[-1])
                        clean_mask = _compute_clean_ink_mask_grid(clean_t, h_p, w_p)
                        m_loss = F.binary_cross_entropy_with_logits(
                            sal_logits.float(), clean_mask.float()
                        ).to(sal_logits.dtype)
                        total_loss = total_loss + CLEAN_MASK_BCE_WEIGHT * m_loss
                        step_mask_loss += float(m_loss.item()) / GRAD_ACCUM

            if not torch.isfinite(total_loss):
                nan_skips += 1
                continue
            (total_loss / GRAD_ACCUM).backward()
            step_loss += float(ce_loss.item()) / GRAD_ACCUM

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if retinex is not None:
            torch.nn.utils.clip_grad_norm_(retinex.parameters(), 1.0)
        opt.step()
        losses.append(step_loss)
        if use_retinex_and_clean_mask:
            mask_losses.append(step_mask_loss)

        if step % 50 == 0 or step == TRAIN_STEPS:
            gate_val = float(torch.tanh(retinex.res_gate).item()) if retinex is not None else 0.0
            print(
                f"[{EXP_ID}] {arm_name} step={step}/{TRAIN_STEPS} ce={step_loss:.4f} "
                f"mask_bce={step_mask_loss:.4f} gate={gate_val:.4f} elapsed={time.monotonic() - t0:.1f}s",
                flush=True,
            )

        if (time.monotonic() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            payload: Dict[str, Any] = {"step": step, "model": model.state_dict(), "opt": opt.state_dict()}
            if retinex is not None:
                payload["retinex"] = retinex.state_dict()
            _atomic_save_torch(ckpt_resume, payload)
            t_last_save = time.monotonic()

    metrics = _evaluate_arm(
        model, retinex, cfg, eval_pieces, device,
        use_bf16=use_bf16, eval_s31=eval_s31, rlvr_grpo=rlvr_grpo,
    )
    res = {
        "arm": arm_name,
        "use_retinex_and_clean_mask": use_retinex_and_clean_mask,
        "retinex_params": int(retinex_params),
        "final_retinex_gate_tanh": round(float(torch.tanh(retinex.res_gate).item()), 5) if retinex is not None else 0.0,
        "train_steps": TRAIN_STEPS,
        "final_ce_loss": round(sum(losses[-20:]) / max(1, len(losses[-20:])), 4),
        "final_mask_bce": round(sum(mask_losses[-20:]) / max(1, len(mask_losses[-20:])), 4) if mask_losses else None,
        "nan_skips": nan_skips,
        "elapsed_sec": round(time.monotonic() - t0, 2),
        "metrics": metrics,
    }
    _atomic_write_json(arm_done_path, res)
    if ckpt_resume.exists():
        ckpt_resume.unlink(missing_ok=True)
    del model, retinex, opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def main() -> None:
    t_start = time.monotonic()
    hf_token = _read_and_purge_hf_token()
    _ensure_packages()
    _locate_mvd()

    import torch
    from huggingface_hub import HfApi
    import eval_uni_suite31 as eval_s31
    import rlvr_grpo
    import uni_omr
    import uni_omr_v2 as uni_v2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    print(f"[{EXP_ID}] device={device} bf16={use_bf16}", flush=True)

    ckpt_local, ckpt_source, train_abs, _, manifest_path = _ensure_assets(hf_token)
    train_pairs = _load_manifest_pairs(train_abs)
    _, all_pieces = eval_s31.resolve_manifest_pieces(manifest_path)
    eval_pieces = _select_eval_pieces(all_pieces, n_printed=12)

    run_dir = VOL_ROOT / "runs" / EXP_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    arm_ctrl = _train_one_arm(
        "arm_ctrl_standard_router", False, ckpt_local, train_pairs, eval_pieces,
        run_dir, device, use_bf16, uni_omr, uni_v2, eval_s31, rlvr_grpo,
    )
    arm_treat = _train_one_arm(
        "arm_retinex_clean_router_mask", True, ckpt_local, train_pairs, eval_pieces,
        run_dir, device, use_bf16, uni_omr, uni_v2, eval_s31, rlvr_grpo,
    )

    m_c = arm_ctrl["metrics"]
    m_t = arm_treat["metrics"]
    deltas = {
        "als_stress_note_f1_delta": round(m_t["als_stress_note_f1"] - m_c["als_stress_note_f1"], 4),
        "hw_photo_note_f1_delta": round(m_t["hw_photo_note_f1"] - m_c["hw_photo_note_f1"], 4),
        "hw_clean_note_f1_delta": round(m_t["hw_clean_note_f1"] - m_c["hw_clean_note_f1"], 4),
        "printed_clean_note_f1_delta": round(m_t["printed_clean_note_f1"] - m_c["printed_clean_note_f1"], 4),
        "router_notation_recall_delta": round(m_t["router_notation_recall"] - m_c["router_notation_recall"], 4),
        "gutter_eviction_rate_delta": round(m_t["gutter_eviction_rate"] - m_c["gutter_eviction_rate"], 4),
    }

    promote = (
        (deltas["als_stress_note_f1_delta"] >= 0.040 or deltas["hw_photo_note_f1_delta"] >= 0.020)
        and deltas["router_notation_recall_delta"] >= 0.05
        and deltas["printed_clean_note_f1_delta"] >= -0.005
    )
    verdict = "PROMOTE" if promote else ("ITERATE" if deltas["router_notation_recall_delta"] >= 0.05 else "DISCARD")

    eval_payload = {
        "exp_id": EXP_ID,
        "completed_at": _utc_now(),
        "elapsed_sec": round(time.monotonic() - t_start, 2),
        "baseline_ckpt": ckpt_source,
        "verdict": verdict,
        "deltas": deltas,
        "arms": {"control": arm_ctrl, "treatment": arm_treat},
    }
    run_payload = {
        "exp_id": EXP_ID,
        "status": "COMPLETED",
        "verdict": verdict,
        "completed_at": eval_payload["completed_at"],
        "elapsed_sec": eval_payload["elapsed_sec"],
        "deltas": deltas,
    }

    local_eval = run_dir / f"{EXP_ID}.eval.json"
    local_run = run_dir / f"{EXP_ID}.run.json"
    _atomic_write_json(local_eval, eval_payload)
    _atomic_write_json(local_run, run_payload)

    api = HfApi(token=hf_token)
    api.upload_file(path_or_fileobj=str(local_eval), path_in_repo=HF_EVAL_PATH, repo_id=HF_REPO_ID, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(local_run), path_in_repo=HF_RUN_PATH, repo_id=HF_REPO_ID, repo_type="dataset")
    print(f"[{EXP_ID}] Uploaded {HF_EVAL_PATH} and {HF_RUN_PATH} | verdict={verdict}", flush=True)


if __name__ == "__main__":
    main()
