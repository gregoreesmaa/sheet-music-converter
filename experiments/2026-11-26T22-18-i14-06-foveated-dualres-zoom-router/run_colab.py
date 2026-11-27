#!/usr/bin/env python3
"""Modal / Colab L4 runner for Wave 11 Exp i14-06 (Wild):
Single-Pass Foveated Dual-Res Zoom Router inside max_mem_tokens=3072.

Experiment ID: 2026-11-26T22-18-i14-06-foveated-dualres-zoom-router
Controlled 2-Arm Ablation (BF16, seed=42, 1,000 steps, EMA decay=0.99 every step):
  - Arm A (Control): Standard single-pass 1408px UniOMR v2 (small_45m, max_mem=3072).
  - Arm B (Treatment): Single-pass Foveated Dual-Res Zoom Router:
      1. Low-res global page (1024px) -> shared backbone + axial + PatchMerger2x2
         + InkTokenRouterV2 (capped at <= 1,536 global tokens).
      2. Automatic top-K (K=12) densest high-frequency ink/beam/chord crops
         (128x256 px at 2x native resolution, selected in ~1ms via Sobel+ink
         energy pooling) -> shared backbone + PatchMerger2x2 (+0.0M backbone
         params) + FoveatedScaleFusion (+0.30M params) -> mapped to exact
         global continuous (y, x) in [0, 1] coordinates in AspectNormalizedRoPE2D!
      3. Concatenated inside the EXACT SAME max_mem_tokens=3072 budget ->
         1 single pass through the 4 Global Encoder layers + 1 GQA Decoder pass!

Uploads:
  - evals/2026-11-26T22-18-i14-06-foveated-dualres-zoom-router.json
  - runs/2026-11-26T22-18-i14-06-foveated-dualres-zoom-router.json
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T22-18-i14-06-foveated-dualres-zoom-router"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter10_5_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter10_v2_best/uni_omr_v2_ema_best.pt",
)
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

VOL_ROOT = Path("/vol") if Path("/vol").is_dir() else Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

SEED = 42
TRAIN_STEPS = 1000
WARMUP_STEPS = 100
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
GRAD_ACCUM = 4
LR_BACKBONE = 3e-5
LR_HEAD = 3e-4
WEIGHT_DECAY = 0.04
CONTROL_MAX_DIM = 1408
GLOBAL_LOWRES_DIM = 1024
FOVEA_TOP_K = 12
FOVEA_CROP_H = 128
FOVEA_CROP_W = 256
GLOBAL_TOKEN_CAP = 1536
TOTAL_MEM_CAP = 3072
MAX_SEQ_LEN = 1536
SUITE31_MAX_LEN = 2048
EMA_DECAY = 0.99
SAVE_EVERY_MINUTES = 5.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str] | str, *, shell: bool = False) -> None:
    print(f"[{EXP_ID}] $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        shell=shell,
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
    for cand in (
        Path("/workspace/mvd"),
        Path("/content/mvd"),
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path("/content/workspace/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
    ):
        if (cand / "uni_omr_v2.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand.parent, cand
    raise FileNotFoundError("Could not locate tmp-train/mvd/uni_omr_v2.py")


def _setup_environment() -> None:
    _locate_repo_and_mvd()
    needed = []
    for mod, pkg in (("huggingface_hub", "huggingface_hub>=0.26"), ("PIL", "pillow"), ("numpy", "numpy")):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed])


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
        target = ckpt_root / cand.replace("/", "__")
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


def _select_topk_foveated_crops(
    img_pil: Any,
    top_k: int = FOVEA_TOP_K,
    crop_h: int = FOVEA_CROP_H,
    crop_w: int = FOVEA_CROP_W,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Select top-K densest high-frequency ink/beam/chord regions from native-res image.

    Returns:
      - crops_t: (K, 3, crop_h, crop_w) normalized tensors
      - boxes_norm: (K, 4) float32 [y0, x0, y1, x1] in [0, 1] global page coordinates
    """
    import numpy as np
    import torch
    from PIL import Image

    rgb = img_pil.convert("RGB")
    W, H = rgb.size
    gray = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0

    # High-frequency ink + gradient energy on a coarse non-overlapping tile grid
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    ink = np.clip(np.percentile(gray, 85) - gray, 0.0, 1.0)
    energy = (gx + gy) * 0.7 + ink * 0.3

    # Partition native page into grid cells proportional to 2x native crop aspect
    n_rows = max(4, min(10, int(round(H / max(128.0, H / 8.0)))))
    n_cols = max(3, min(6, int(round(W / max(256.0, W / 4.0)))))
    cell_h = H / float(n_rows)
    cell_w = W / float(n_cols)

    scored: List[Tuple[float, float, float, float, float]] = []
    for r in range(n_rows):
        y0 = int(round(r * cell_h))
        y1 = min(H, int(round((r + 1) * cell_h)))
        for c in range(n_cols):
            x0 = int(round(c * cell_w))
            x1 = min(W, int(round((c + 1) * cell_w)))
            if (y1 - y0) < 8 or (x1 - x0) < 8:
                continue
            sc = float(energy[y0:y1, x0:x1].mean())
            scored.append((sc, y0 / float(H), x0 / float(W), y1 / float(H), x1 / float(W)))

    scored.sort(key=lambda item: item[0], reverse=True)
    picked = scored[: min(top_k, len(scored))]
    # Sort picked crops in reading order (top-to-bottom, left-to-right)
    picked.sort(key=lambda item: (round(item[1], 2), item[2]))

    crop_tensors: List[torch.Tensor] = []
    boxes: List[List[float]] = []
    for _, yn0, xn0, yn1, xn1 in picked:
        px0 = max(0, int(round(xn0 * W)))
        py0 = max(0, int(round(yn0 * H)))
        px1 = min(W, max(px0 + 8, int(round(xn1 * W))))
        py1 = min(H, max(py0 + 8, int(round(yn1 * H))))
        patch = rgb.crop((px0, py0, px1, py1))
        crop_tensors.append(_pil_to_tensor(patch, max(crop_h, crop_w), target_hw=(crop_h, crop_w)))
        boxes.append([yn0, xn0, yn1, xn1])

    return torch.stack(crop_tensors, dim=0), torch.tensor(boxes, dtype=torch.float32)


def _build_foveated_wrapper() -> Any:
    import torch
    import torch.nn as nn
    import uni_omr_v2

    class FoveatedDualResZoomOMR(nn.Module):
        """Single-pass global (1024px) + top-K 2x native-res foveated crops packed inside <= 3072 tokens."""

        def __init__(self, base: Any) -> None:
            super().__init__()
            self.base = base
            d = int(base.cfg.d_model)
            self.fovea_norm = uni_omr_v2.RMSNorm(d)
            self.fovea_proj = nn.Linear(d, d, bias=False)
            self.scale_emb_lo = nn.Parameter(torch.zeros(1, 1, d))
            self.scale_emb_hi = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        def encode_foveated(
            self,
            lowres_img: torch.Tensor,
            fovea_crops: torch.Tensor,
            fovea_boxes: torch.Tensor,
        ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, Any]]:
            """Encode 1 page (B=1) in a single pass combining global 1024px + K 2x-res foveated crops."""
            enc = self.base.encoder
            # 1. Global low-res branch (capped at GLOBAL_TOKEN_CAP)
            feat_lo = self.base.backbone(lowres_img, return_pyramid=False)
            _, _, H_p, W_p = feat_lo.shape
            x_lo = feat_lo.permute(0, 2, 3, 1).contiguous()
            row_cos, row_sin = enc.axial_rope(W_p, device=x_lo.device, dtype=x_lo.dtype)
            col_cos, col_sin = enc.axial_rope(H_p, device=x_lo.device, dtype=x_lo.dtype)
            for layer in enc.axial_layers:
                x_lo = layer(x_lo, row_cos, row_sin, col_cos, col_sin)
            merged_lo = enc.patch_merger(x_lo)
            merged_lo_2d = merged_lo.permute(0, 3, 1, 2)

            orig_cap = enc.router.max_mem_tokens
            enc.router.max_mem_tokens = GLOBAL_TOKEN_CAP
            try:
                toks_lo, coords_lo, rinfo = enc.router(
                    merged_lo_2d, images=lowres_img, feat_nhwc=merged_lo
                )
            finally:
                enc.router.max_mem_tokens = orig_cap

            gh, gw = rinfo["grid_hw"]
            # Normalize global coords to [0, 1]
            coords_lo_norm = torch.stack(
                [
                    coords_lo[..., 0] / float(max(1, gh - 1)),
                    coords_lo[..., 1] / float(max(1, gw - 1)),
                ],
                dim=-1,
            )
            toks_lo = toks_lo + self.scale_emb_lo.to(toks_lo.dtype)

            # 2. High-res 2x foveated crop branch through SHARED backbone + PatchMerger2x2
            K = fovea_crops.shape[0]
            feat_hi = self.base.backbone(fovea_crops, return_pyramid=False)  # (K, D, hc, wc)
            merged_hi = enc.patch_merger(feat_hi.permute(0, 2, 3, 1).contiguous())  # (K, hm, wm, D)
            _, hm, wm, D = merged_hi.shape
            toks_hi = self.fovea_proj(self.fovea_norm(merged_hi.reshape(1, K * hm * wm, D)))
            toks_hi = toks_hi + self.scale_emb_hi.to(toks_hi.dtype)

            # Map every foveated token to its exact global continuous (y, x) in [0, 1]
            ys_local = torch.linspace(0.0, 1.0, hm, device=toks_hi.device, dtype=torch.float32)
            xs_local = torch.linspace(0.0, 1.0, wm, device=toks_hi.device, dtype=torch.float32)
            gy_l, gx_l = torch.meshgrid(ys_local, xs_local, indexing="ij")
            gy_l = gy_l.reshape(1, hm * wm)
            gx_l = gx_l.reshape(1, hm * wm)

            y0 = fovea_boxes[:, 0:1]
            x0 = fovea_boxes[:, 1:2]
            y1 = fovea_boxes[:, 2:3]
            x1 = fovea_boxes[:, 3:4]
            gy_global = y0 + gy_l * (y1 - y0)
            gx_global = x0 + gx_l * (x1 - x0)
            coords_hi_norm = torch.stack([gy_global.reshape(-1), gx_global.reshape(-1)], dim=-1).unsqueeze(0)

            # Respect TOTAL_MEM_CAP = 3072 hard budget
            rem_budget = max(1, TOTAL_MEM_CAP - toks_lo.shape[1])
            if toks_hi.shape[1] > rem_budget:
                toks_hi = toks_hi[:, :rem_budget, :]
                coords_hi_norm = coords_hi_norm[:, :rem_budget, :]

            all_toks = torch.cat([toks_lo, toks_hi], dim=1)
            all_coords_norm = torch.cat([coords_lo_norm, coords_hi_norm], dim=1)

            # Compute continuous global AspectNormalizedRoPE2D + sinusoidal pos2d using virtual [0..1000] grid
            virt_hw = (1001, 1001)
            virt_coords = all_coords_norm * 1000.0
            pos2d = enc.pos2d_proj(enc._sinusoidal_pos2d(virt_coords, grid_hw=virt_hw, dtype=all_toks.dtype))
            cos2d, sin2d = enc.rope2d(virt_coords, grid_hw=virt_hw, dtype=all_toks.dtype)

            mem_post_merger = all_toks
            mem = all_toks + pos2d
            n_global = len(enc.global_layers)
            mid_idx = max(0, (n_global // 2) - 1)
            mem_global_mid = mem_post_merger
            for i, layer in enumerate(enc.global_layers):
                mem = layer(mem, rope_cos=cos2d, rope_sin=sin2d)
                if i == mid_idx:
                    mem_global_mid = mem
            mem_final = enc.enc_norm(mem) + pos2d

            rinfo["n_global_lo"] = int(toks_lo.shape[1])
            rinfo["n_fovea_hi"] = int(toks_hi.shape[1])
            rinfo["n_total_mem"] = int(all_toks.shape[1])
            return (mem_post_merger, mem_global_mid, mem_final), rinfo

    return FoveatedDualResZoomOMR


def _build_base_model(ckpt_path: Optional[Path], device: "torch.device") -> Any:
    import torch
    import uni_omr_v2

    cfg = uni_omr_v2.UniOMRv2Config.small_45m(grad_checkpoint=True)
    model = uni_omr_v2.UniOMRv2Net(cfg).to(device)
    if ckpt_path is not None and ckpt_path.is_file():
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = raw.get("ema_state_dict") or raw.get("model_state_dict") or raw.get("state_dict") or raw
        model.load_state_dict(state, strict=False)
    return model


def _decode_from_mem(base: Any, mem: Any, max_len: int, device: "torch.device", use_bf16: bool) -> List[int]:
    import torch
    import eval_uni_suite31 as eus

    _, bos_id, eos_id, _, _, _ = eus._vocab_spec(int(base.cfg.vocab_size))
    cur_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
    out_ids = [bos_id]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        past_kvs = base.precompute_cross_kv(mem)
        for _ in range(max_len):
            logits, _, past_kvs = base.decode_tokens(
                cur_ids, memory=None, return_mtp=False, return_aux=False, past_kvs=past_kvs, use_cache=True
            )
            nxt = int(logits[:, -1, :].argmax(dim=-1).item())
            out_ids.append(nxt)
            if nxt == eos_id:
                break
            cur_ids = torch.tensor([[nxt]], dtype=torch.long, device=device)
    return out_ids


def _eval_val_and_suite31(
    model_or_wrap: Any,
    *,
    is_foveated: bool,
    val_pairs: List[Tuple[Path, List[Path], str]],
    manifest_path: Path,
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import xml.etree.ElementTree as ET
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import eval_uni_suite31 as eus
    import rlvr_grpo
    import uni_omr

    model_or_wrap.eval()
    base = model_or_wrap.base if is_foveated else model_or_wrap

    # 1. Proxy Val CE
    val_losses: List[float] = []
    with torch.no_grad():
        for img_p, beks, _ in val_pairs:
            ids = _encode_bekerns_full(beks)
            if len(ids) < 3:
                continue
            img = Image.open(img_p).convert("RGB")
            tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                if is_foveated:
                    lo = _pil_to_tensor(img, GLOBAL_LOWRES_DIM).unsqueeze(0).to(device)
                    crops, boxes = _select_topk_foveated_crops(img)
                    mem, _ = model_or_wrap.encode_foveated(lo, crops.to(device), boxes.to(device))
                else:
                    inp = _pil_to_tensor(img, CONTROL_MAX_DIM).unsqueeze(0).to(device)
                    mem, _, _ = base.encode_image(inp, return_aux=False)
                logits, _, _ = base.decode_tokens(
                    tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID
                )
            val_losses.append(float(loss.item()))

    # 2. suite31_hw OOD Note F1
    def _xml_notes(xml_path: Path) -> List[Tuple[str, int, str, str, str]]:
        notes: List[Tuple[str, int, str, str, str]] = []
        try:
            root = ET.parse(str(xml_path)).getroot()
        except Exception:
            return notes
        for elem in root.iter():
            if elem.tag.split("}", 1)[-1] != "note":
                continue
            is_rest = is_grace = False
            step = octave = alter = dur = ""
            n_dots = 0
            for child in elem:
                ctag = child.tag.split("}", 1)[-1]
                if ctag == "rest":
                    is_rest = True
                elif ctag == "grace":
                    is_grace = True
                elif ctag == "pitch":
                    for pc in child:
                        ptag = pc.tag.split("}", 1)[-1]
                        txt = (pc.text or "").strip()
                        if ptag == "step":
                            step = txt.upper()
                        elif ptag == "octave":
                            octave = txt
                        elif ptag == "alter":
                            try:
                                av = int(round(float(txt)))
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

    _, _, eos_id, _, _, _ = eus._vocab_spec(int(base.cfg.vocab_size))
    _, pieces = eus.resolve_manifest_pieces(manifest_path)
    buckets: Dict[str, List[float]] = {
        "printed_clean": [],
        "printed_photo": [],
        "hw_clean": [],
        "hw_photo": [],
        "all90": [],
    }
    latencies: List[float] = []
    mem_counts: List[int] = []

    with torch.no_grad():
        for entry in pieces:
            is_hw = bool(entry.get("handwritten", False))
            ref_notes = _xml_notes(Path(entry["gt"]))
            for mode_key in ("clean", "photo"):
                img_p = Path(entry[mode_key])
                if not img_p.is_file():
                    continue
                img = Image.open(img_p).convert("RGB")
                t0 = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    if is_foveated:
                        lo = _pil_to_tensor(img, GLOBAL_LOWRES_DIM).unsqueeze(0).to(device)
                        crops, boxes = _select_topk_foveated_crops(img)
                        mem, rinfo = model_or_wrap.encode_foveated(lo, crops.to(device), boxes.to(device))
                        mem_counts.append(int(rinfo.get("n_total_mem", 0)))
                    else:
                        inp = _pil_to_tensor(img, CONTROL_MAX_DIM).unsqueeze(0).to(device)
                        mem, _, rinfo = base.encode_image(inp, return_aux=False)
                        mem_counts.append(int(rinfo.get("n_out", 0)))
                ids = _decode_from_mem(base, mem, SUITE31_MAX_LEN, device, use_bf16)
                latencies.append(time.perf_counter() - t0)
                pred_toks = eus.decode_uni_compat(ids, vocab_size=base.cfg.vocab_size, skip_special=True)
                if eos_id not in ids[1:]:
                    pred_toks = eus.trim_truncated_tokens(pred_toks)
                hyp_toks, _ = eus.stitch_uni_tokens(pred_toks)
                hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
                nf1 = float(rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes))
                buckets[f"{'hw' if is_hw else 'printed'}_{mode_key}"].append(nf1)
                buckets["all90"].append(nf1)

    return {
        "val_ce_loss": round(sum(val_losses) / max(1, len(val_losses)), 5),
        "suite31_hw": {k: round(sum(v) / max(1, len(v)), 4) if v else None for k, v in buckets.items()},
        "mean_mem_tokens": round(sum(mem_counts) / max(1, len(mem_counts)), 1),
        "mean_latency_s": round(sum(latencies) / max(1, len(latencies)), 3),
    }


def _train_arm(
    arm_name: str,
    *,
    is_foveated: bool,
    ckpt_path: Optional[Path],
    train_pairs: List[Tuple[Path, List[Path], str]],
    val_pairs: List[Tuple[Path, List[Path], str]],
    manifest_path: Path,
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr

    _seed_everything(SEED)
    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached = arm_dir / "arm_result.json"
    if cached.is_file():
        print(f"[{EXP_ID}] Reusing cached result for {arm_name}", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))

    base = _build_base_model(ckpt_path, device)
    WrapperCls = _build_foveated_wrapper()
    model = WrapperCls(base).to(device) if is_foveated else base

    if is_foveated:
        head_params = [
            *model.fovea_norm.parameters(),
            *model.fovea_proj.parameters(),
            model.scale_emb_lo,
            model.scale_emb_hi,
        ]
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
    t_arm0 = time.time()
    t_last_save = time.time()
    loss_hist: List[float] = []

    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        mult = _wsd_mult(step, TRAIN_STEPS, WARMUP_STEPS)
        opt.param_groups[0]["lr"] = LR_BACKBONE * mult
        if is_foveated:
            opt.param_groups[1]["lr"] = LR_HEAD * mult

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            img_p, beks, _ = train_pairs[rng.randrange(len(train_pairs))]
            ids = _encode_bekerns_full(beks)
            if len(ids) < 3:
                continue
            img = Image.open(img_p).convert("RGB")
            tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                if is_foveated:
                    lo = _pil_to_tensor(img, GLOBAL_LOWRES_DIM).unsqueeze(0).to(device)
                    crops, boxes = _select_topk_foveated_crops(img)
                    mem, _ = model.encode_foveated(lo, crops.to(device), boxes.to(device))
                    logits, _, _ = model.base.decode_tokens(tgt[:, :-1], mem, return_mtp=False, return_aux=False)
                else:
                    inp = _pil_to_tensor(img, CONTROL_MAX_DIM).unsqueeze(0).to(device)
                    mem, _, _ = model.encode_image(inp, return_aux=False)
                    logits, _, _ = model.decode_tokens(tgt[:, :-1], mem, return_mtp=False, return_aux=False)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID
                ) / GRAD_ACCUM
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

        loss_hist.append(step_loss)
        if step % 100 == 0 or step == 1:
            mean_l = sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:]))
            print(f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} loss={mean_l:.4f} elapsed={time.time()-t_arm0:.1f}s", flush=True)

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ck = arm_dir / "latest.pt.tmp"
            torch.save({"step": step, "model_state_dict": model.state_dict(), "ema_state_dict": ema_state}, tmp_ck)
            os.replace(tmp_ck, arm_dir / "latest.pt")
            t_last_save = time.time()

    raw_eval = _eval_val_and_suite31(
        model, is_foveated=is_foveated, val_pairs=val_pairs, manifest_path=manifest_path, device=device, use_bf16=use_bf16
    )
    model.load_state_dict(ema_state, strict=True)
    ema_eval = _eval_val_and_suite31(
        model, is_foveated=is_foveated, val_pairs=val_pairs, manifest_path=manifest_path, device=device, use_bf16=use_bf16
    )

    result = {
        "arm": arm_name,
        "is_foveated": is_foveated,
        "extra_params": extra_params,
        "total_params": sum(p.numel() for p in model.parameters()),
        "train_loss_tail50": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5),
        "raw_eval": raw_eval,
        "ema_eval": ema_eval,
        "elapsed_secs": round(time.time() - t_arm0, 2),
    }
    _atomic_write_json(cached, result)
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
        _setup_environment()
        import torch
        from huggingface_hub import HfApi

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

        ckpt_local, ckpt_source, train_abs, val_abs, manifest_path = _ensure_assets(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(f"[{EXP_ID}] Loaded {len(train_pairs)} train / {len(val_pairs)} val on {gpu_name}", flush=True)

        arm_a = _train_arm(
            "arm_a_singlepass_1408_control",
            is_foveated=False,
            ckpt_path=ckpt_local,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            manifest_path=manifest_path,
            device=device,
            use_bf16=use_bf16,
        )
        arm_b = _train_arm(
            "arm_b_foveated_dualres_singlepass",
            is_foveated=True,
            ckpt_path=ckpt_local,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            manifest_path=manifest_path,
            device=device,
            use_bf16=use_bf16,
        )

        def _delta(b_val: Optional[float], a_val: Optional[float]) -> Optional[float]:
            return None if (b_val is None or a_val is None) else round(float(b_val) - float(a_val), 5)

        ea, eb = arm_a["ema_eval"], arm_b["ema_eval"]
        comparison = {
            "ema_val_ce_b_minus_a": _delta(eb["val_ce_loss"], ea["val_ce_loss"]),
            "ema_all90_f1_b_minus_a": _delta(eb["suite31_hw"]["all90"], ea["suite31_hw"]["all90"]),
            "ema_printed_clean_b_minus_a": _delta(
                eb["suite31_hw"]["printed_clean"], ea["suite31_hw"]["printed_clean"]
            ),
            "ema_hw_clean_b_minus_a": _delta(eb["suite31_hw"]["hw_clean"], ea["suite31_hw"]["hw_clean"]),
            "latency_ratio_b_over_a": round(eb["mean_latency_s"] / max(1e-6, ea["mean_latency_s"]), 3),
            "mean_mem_tokens_b": eb["mean_mem_tokens"],
        }

        report = {
            "exp_id": EXP_ID,
            "exp_code": "i14-06",
            "status": "completed",
            "warmstart_ckpt": ckpt_source or "scratch_small_45m",
            "started_utc": started_iso,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"modal-or-colab-{gpu_name}",
            "bf16": use_bf16,
            "arm_a_singlepass_1408_control": arm_a,
            "arm_b_foveated_dualres_singlepass": arm_b,
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
            commit_message=f"eval({EXP_ID}): upload single-pass foveated dual-res zoom router ablation",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload single-pass foveated dual-res zoom router ledger",
        )
        print(f"[{EXP_ID}] UPLOADED {HF_EVAL_PATH} & {HF_RUN_PATH} TO HF!", flush=True)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
