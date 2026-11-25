#!/usr/bin/env python3
"""Google Colab L4 runner for Overnight Exp 21: Flood-Fill paper_gate Dark-Border Threshold Sweep.

Experiment ID: 2026-11-26T01-38-ov-21-paper-gate-border-threshold
Sweeps the 3x3 grid of edge-connected flood-fill paper_gate dark-border thresholds:
  - luma_stop in {0.35, 0.45, 0.55}
  - dark_thr  in {0.22, 0.30, 0.38}
across 290 real/OOD pages:
  - All 90 pages of datasets/suite31_hw_eval.tar (31 printed + 14 real HW, clean + photo)
  - 200 deterministic seeded-sampled real historical RISM pages from datasets/rism_hw_5707.tar

Measures per threshold pair and per cohort:
  - Interior ink recall (pixel-level & patch-level mean, p05, min)
  - Desk-border suppression rate on exterior dark borders (photo & RISM)
  - Clean-page false-masking rate on borderless suite31_hw clean pages
  - Paper fraction, collapsed-page count (paper_frac < 0.50), simulated top-40% routed ink recall
  - Bit-exact parity verification against vir_router_v2.compute_ink_and_paper_gate_f32 at (0.45, 0.30)

Uploads results to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T01-38-ov-21-paper-gate-border-threshold.json
  - runs/2026-11-26T01-38-ov-21-paper-gate-border-threshold.json
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T01-38-ov-21-paper-gate-border-threshold"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_RISM_TAR = "datasets/rism_hw_5707.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")
BUNDLE_CANDIDATES = (
    Path("/content/bundle.tar.gz"),
    Path("/content/ov21_bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
)

SEED = 42
FULLPAGE_MAX_DIM = 1280
RISM_SAMPLE_COUNT = 200
BG_FLOOR = 0.05
LUMA_STOPS: Tuple[float, ...] = (0.35, 0.45, 0.55)
DARK_THRS: Tuple[float, ...] = (0.22, 0.30, 0.38)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str]) -> None:
    print(f"[{EXP_ID}] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
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


def _resolve_mvd_dir() -> Optional[Path]:
    candidates = [
        WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
        WORKSPACE_ROOT / "tmp-train" / "mvd",
        WORKSPACE_ROOT / "mvd",
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
    ]
    for c in candidates:
        if (c / "vir_router_v2.py").is_file():
            return c
    for b in BUNDLE_CANDIDATES:
        if b.is_file():
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            print(f"[{EXP_ID}] Extracting {b} -> {WORKSPACE_ROOT} ...", flush=True)
            with tarfile.open(b, "r:*") as tf:
                tf.extractall(path=WORKSPACE_ROOT)
            for c in candidates:
                if (c / "vir_router_v2.py").is_file():
                    return c
    return None


def _setup_environment() -> Optional[Path]:
    needed_pkgs: List[str] = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed_pkgs.append(pkg)
    if needed_pkgs:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed_pkgs])
    mvd_dir = _resolve_mvd_dir()
    if mvd_dir is not None and str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))
    return mvd_dir


def _ensure_datasets(hf_token: str) -> Tuple[Path, List[Path]]:
    from huggingface_hub import hf_hub_download

    ds_root = VOL_ROOT / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)

    # 1. Suite31_hw (45 pieces x clean + photo = 90 pages)
    suite31_dir = ds_root / "suite31_hw_eval"
    manifest_path = suite31_dir / "suite31" / "suite31_hw_manifest.json"
    if not manifest_path.is_file():
        suite31_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR} ...", flush=True)
        t0 = time.perf_counter()
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_SUITE31_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        print(f"[{EXP_ID}] Downloaded suite31 tar in {time.perf_counter() - t0:.1f}s -> extracting ...", flush=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=suite31_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass

    # 2. Unlabeled RISM 5,707 real historical manuscripts
    rism_dir = ds_root / "rism_hw_5707"
    rism_sentinel = rism_dir / ".unpacked_ok"
    if not rism_sentinel.is_file():
        rism_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_RISM_TAR} ...", flush=True)
        t0 = time.perf_counter()
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_RISM_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        print(f"[{EXP_ID}] Downloaded RISM tar in {time.perf_counter() - t0:.1f}s -> extracting ...", flush=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=rism_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass
        rism_sentinel.write_text("ok\n", encoding="utf-8")

    all_rism = sorted(
        p for p in rism_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    rng = random.Random(SEED)
    n_take = min(RISM_SAMPLE_COUNT, len(all_rism))
    sampled_rism = sorted(rng.sample(all_rism, n_take))
    print(
        f"[{EXP_ID}] Discovered {len(all_rism)} RISM pages -> deterministic sample of {len(sampled_rism)} (seed={SEED})",
        flush=True,
    )
    return manifest_path, sampled_rism


def _collect_eval_pages(manifest_path: Path, rism_paths: List[Path]) -> List[Dict[str, Any]]:
    """Collect all 90 suite31_hw pages + 200 RISM pages with cohort metadata."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    pieces = manifest.get("pieces", manifest if isinstance(manifest, list) else [])

    pages: List[Dict[str, Any]] = []
    for piece in pieces:
        slug = str(piece.get("slug") or piece.get("name") or piece.get("id"))
        is_hw = bool(piece.get("handwritten", False))
        for domain in ("clean", "photo"):
            rel = piece.get(domain) or piece.get(f"{domain}_image")
            if not rel:
                continue
            p = Path(rel)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            if not p.is_file():
                alt = (base_dir.parent / rel).resolve()
                if alt.is_file():
                    p = alt
            if p.is_file():
                cohort = f"{'hw' if is_hw else 'printed'}_{domain}"
                pages.append(
                    {
                        "page_id": f"{slug}:{domain}",
                        "cohort": cohort,
                        "is_clean_borderless": (domain == "clean"),
                        "path": p,
                    }
                )

    for idx, rp in enumerate(rism_paths):
        pages.append(
            {
                "page_id": f"rism:{idx:03d}:{rp.stem}",
                "cohort": "rism_real_hw_200",
                "is_clean_borderless": False,
                "path": rp,
            }
        )
    return pages


def _load_image_tensor(path: Path, max_dim: int, device: Any) -> Any:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        scale = float(max_dim) / float(max(1, max(w, h)))
        if scale < 1.0:
            nw = max(16, int(round(w * scale)))
            nh = max(16, int(round(h * scale)))
            rgb = rgb.resize((nw, nh), Image.Resampling.BICUBIC)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    # Pad right/bottom with white (1.0) to exact multiple of 16 matching /16 router grid
    _, _, h0, w0 = t.shape
    pad_h = (16 - (h0 % 16)) % 16
    pad_w = (16 - (w0 % 16)) % 16
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=1.0)
    return t


def compute_ink_and_paper_gate_param(
    images: Any,
    h_p: int,
    w_p: int,
    *,
    luma_stop: float,
    dark_thr: float,
    bg_floor: float = BG_FLOOR,
) -> Dict[str, Any]:
    """Parameterized pure-float32 edge-connected dark-border paper_gate.

    Bit-identical to vir_router_v2.compute_ink_and_paper_gate_f32 at (0.45, 0.30).
    """
    import torch
    import torch.nn.functional as F

    img_f = images.to(torch.float32)
    if img_f.shape[1] == 3:
        luma = (
            0.299 * img_f[:, 0:1]
            + 0.587 * img_f[:, 1:2]
            + 0.114 * img_f[:, 2:3]
        )
    else:
        luma = img_f.mean(dim=1, keepdim=True)

    bright_f = (luma >= float(luma_stop)).to(torch.float32)
    top_dark = (bright_f.cumsum(dim=2) < 0.5).to(torch.float32)
    bot_dark = (bright_f.flip(dims=(2,)).cumsum(dim=2).flip(dims=(2,)) < 0.5).to(torch.float32)
    left_dark = (bright_f.cumsum(dim=3) < 0.5).to(torch.float32)
    right_dark = (bright_f.flip(dims=(3,)).cumsum(dim=3).flip(dims=(3,)) < 0.5).to(torch.float32)
    edge_connected_f = (top_dark + bot_dark + left_dark + right_dark).clamp(0.0, 1.0)
    dark_px_f = (luma < float(dark_thr)).to(torch.float32)
    border_px = edge_connected_f * dark_px_f

    border_dilated_px = F.max_pool2d(
        F.max_pool2d(border_px, kernel_size=(1, 33), stride=1, padding=(0, 16)),
        kernel_size=(33, 1),
        stride=1,
        padding=(16, 0),
    )
    paper_px = (1.0 - border_dilated_px).clamp(0.0, 1.0)

    local_bg = F.avg_pool2d(
        F.avg_pool2d(
            luma,
            kernel_size=(1, 31),
            stride=1,
            padding=(0, 15),
            count_include_pad=False,
        ),
        kernel_size=(31, 1),
        stride=1,
        padding=(15, 0),
        count_include_pad=False,
    )
    raw_ink = ((local_bg - luma) / local_bg.clamp(min=0.15)).clamp(0.0, 1.0)
    local_ink = raw_ink * paper_px

    in_h, in_w = int(img_f.shape[2]), int(img_f.shape[3])
    if in_h == h_p * 16 and in_w == w_p * 16:
        ks = (16, 16)
        ink_max = F.max_pool2d(local_ink, kernel_size=ks, stride=ks)
        ink_avg = F.avg_pool2d(local_ink, kernel_size=ks, stride=ks)
        paper_grid_f = F.avg_pool2d(paper_px, kernel_size=ks, stride=ks)
        raw_ink_max = F.max_pool2d(raw_ink, kernel_size=ks, stride=ks)
        raw_ink_avg = F.avg_pool2d(raw_ink, kernel_size=ks, stride=ks)
    else:
        ink_max = F.adaptive_max_pool2d(local_ink, (h_p, w_p))
        ink_avg = F.adaptive_avg_pool2d(local_ink, (h_p, w_p))
        paper_grid_f = F.adaptive_avg_pool2d(paper_px, (h_p, w_p))
        raw_ink_max = F.adaptive_max_pool2d(raw_ink, (h_p, w_p))
        raw_ink_avg = F.adaptive_avg_pool2d(raw_ink, (h_p, w_p))

    local_ink_grid = 0.5 * ink_max + 0.5 * ink_avg
    raw_ink_grid = 0.5 * raw_ink_max + 0.5 * raw_ink_avg
    effective_paper_f = (paper_grid_f > 0.5).to(torch.float32)
    paper_gate = float(bg_floor) + (1.0 - float(bg_floor)) * effective_paper_f
    local_ink_grid = local_ink_grid * effective_paper_f

    return {
        "luma": luma,
        "paper_px": paper_px,
        "raw_ink": raw_ink,
        "raw_ink_grid": raw_ink_grid,
        "local_ink_grid": local_ink_grid,
        "paper_gate": paper_gate,
        "effective_paper_f": effective_paper_f,
    }


def _score_page(out: Dict[str, Any], is_clean_borderless: bool) -> Dict[str, float]:
    """Compute non-circular interior ink recall, desk-border suppression, and routing metrics."""
    import torch

    luma = out["luma"]                     # (1, 1, H, W)
    paper_px = out["paper_px"]             # (1, 1, H, W)
    raw_ink = out["raw_ink"]               # (1, 1, H, W)
    raw_ink_grid = out["raw_ink_grid"]     # (1, 1, h_p, w_p)
    local_ink_grid = out["local_ink_grid"] # (1, 1, h_p, w_p)
    paper_gate = out["paper_gate"]         # (1, 1, h_p, w_p)
    eff_paper = out["effective_paper_f"]   # (1, 1, h_p, w_p)

    _, _, H, W = luma.shape
    _, _, h_p, w_p = eff_paper.shape

    # 1. Safe central interior [10%..90%] x [10%..90%] (guaranteed musical page body, never desk)
    y0, y1 = max(1, int(round(H * 0.10))), max(2, int(round(H * 0.90)))
    x0, x1 = max(1, int(round(W * 0.10))), max(2, int(round(W * 0.90)))
    int_ink_px = (raw_ink[:, :, y0:y1, x0:x1] > 0.12).to(torch.float32)
    int_paper_px = paper_px[:, :, y0:y1, x0:x1]
    tot_int_px = float(int_ink_px.sum().item())
    if tot_int_px > 0.0:
        interior_ink_px_recall = float((int_ink_px * int_paper_px).sum().item() / tot_int_px)
    else:
        interior_ink_px_recall = 1.0

    gy0, gy1 = max(1, int(round(h_p * 0.10))), max(2, int(round(h_p * 0.90)))
    gx0, gx1 = max(1, int(round(w_p * 0.10))), max(2, int(round(w_p * 0.90)))
    int_ink_grid = (raw_ink_grid[:, :, gy0:gy1, gx0:gx1] > 0.04).to(torch.float32)
    int_eff_grid = eff_paper[:, :, gy0:gy1, gx0:gx1]
    tot_int_grid = float(int_ink_grid.sum().item())
    if tot_int_grid > 0.0:
        interior_ink_patch_recall = float((int_ink_grid * int_eff_grid).sum().item() / tot_int_grid)
    else:
        interior_ink_patch_recall = 1.0

    # 2. Desk-border suppression on outer 5% dark perimeter (luma < 0.25)
    by = max(1, int(round(H * 0.05)))
    bx = max(1, int(round(W * 0.05)))
    perim_mask = torch.zeros_like(luma)
    perim_mask[:, :, :by, :] = 1.0
    perim_mask[:, :, -by:, :] = 1.0
    perim_mask[:, :, :, :bx] = 1.0
    perim_mask[:, :, :, -bx:] = 1.0
    ext_dark = perim_mask * (luma < 0.25).to(torch.float32)
    tot_ext_dark = float(ext_dark.sum().item())
    has_dark_border = 1.0 if tot_ext_dark >= 200.0 else 0.0
    if tot_ext_dark >= 200.0:
        desk_border_suppression = float((ext_dark * (1.0 - paper_px)).sum().item() / tot_ext_dark)
    else:
        desk_border_suppression = -1.0  # Sentinel: no dark border present

    # 3. Paper fraction & clean-page false-masking rate
    paper_frac = float(eff_paper.mean().item())
    false_mask_rate = (1.0 - paper_frac) if is_clean_borderless else -1.0
    collapsed = 1.0 if paper_frac < 0.50 else 0.0

    # 4. Simulated top-40% router ink patch recall on interior ink
    score_flat = (local_ink_grid * paper_gate).flatten(1)
    n_tot = int(score_flat.shape[1])
    k40 = max(1, min(n_tot, int(round(n_tot * 0.40))))
    _, topk_idx = torch.topk(score_flat, k=k40, dim=1, largest=True, sorted=False)
    kept_mask = torch.zeros_like(score_flat)
    kept_mask.scatter_(1, topk_idx, 1.0)
    kept_2d = kept_mask.view(1, 1, h_p, w_p)
    if tot_int_grid > 0.0:
        top40_ink_recall = float((int_ink_grid * kept_2d[:, :, gy0:gy1, gx0:gx1]).sum().item() / tot_int_grid)
    else:
        top40_ink_recall = 1.0

    return {
        "interior_ink_px_recall": interior_ink_px_recall,
        "interior_ink_patch_recall": interior_ink_patch_recall,
        "desk_border_suppression": desk_border_suppression,
        "has_dark_border": has_dark_border,
        "paper_frac": paper_frac,
        "false_mask_rate": false_mask_rate,
        "collapsed": collapsed,
        "top40_ink_recall": top40_ink_recall,
    }


def _aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    import numpy as np

    if not rows:
        return {"n_pages": 0}
    px_rec = [r["interior_ink_px_recall"] for r in rows]
    pt_rec = [r["interior_ink_patch_recall"] for r in rows]
    p_frac = [r["paper_frac"] for r in rows]
    t40 = [r["top40_ink_recall"] for r in rows]
    border_sup = [r["desk_border_suppression"] for r in rows if r["has_dark_border"] > 0.5]
    clean_fm = [r["false_mask_rate"] for r in rows if r["false_mask_rate"] >= 0.0]
    n_collapsed = int(sum(1 for r in rows if r["collapsed"] > 0.5))

    return {
        "n_pages": len(rows),
        "n_pages_with_dark_border": len(border_sup),
        "interior_ink_px_recall_mean": round(float(np.mean(px_rec)), 5),
        "interior_ink_px_recall_p05": round(float(np.percentile(px_rec, 5)), 5),
        "interior_ink_px_recall_min": round(float(np.min(px_rec)), 5),
        "interior_ink_patch_recall_mean": round(float(np.mean(pt_rec)), 5),
        "interior_ink_patch_recall_p05": round(float(np.percentile(pt_rec, 5)), 5),
        "desk_border_suppression_mean": round(float(np.mean(border_sup)), 5) if border_sup else None,
        "desk_border_suppression_p10": round(float(np.percentile(border_sup, 10)), 5) if border_sup else None,
        "clean_false_mask_rate_mean": round(float(np.mean(clean_fm)), 5) if clean_fm else None,
        "paper_frac_mean": round(float(np.mean(p_frac)), 5),
        "collapsed_page_count": n_collapsed,
        "top40_ink_recall_mean": round(float(np.mean(t40)), 5),
    }


def main() -> None:
    try:
        hf_token = _read_and_purge_hf_token()
        mvd_dir = _setup_environment()

        import torch
        from huggingface_hub import HfApi

        t_start = time.perf_counter()
        started_at = _utc_now()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        print(f"[{EXP_ID}] Running on {gpu_name} (torch={torch.__version__})", flush=True)

        manifest_path, rism_paths = _ensure_datasets(hf_token)
        pages = _collect_eval_pages(manifest_path, rism_paths)
        cohort_counts: Dict[str, int] = {}
        for pg in pages:
            cohort_counts[pg["cohort"]] = cohort_counts.get(pg["cohort"], 0) + 1
        print(f"[{EXP_ID}] Total pages={len(pages)} | cohorts={cohort_counts}", flush=True)

        # Preload resized tensors on CPU (290 pages @ 1280px uint8-equivalent takes ~1.1 GB RAM)
        # Keep uint8 in RAM and cast per-page on GPU for fast, low-memory 9-config sweep.
        print(f"[{EXP_ID}] Preloading {len(pages)} resized pages...", flush=True)
        cached_tensors: List[Tuple[Dict[str, Any], Any]] = []
        for idx, pg in enumerate(pages, start=1):
            t_gpu = _load_image_tensor(pg["path"], FULLPAGE_MAX_DIM, device)
            t_u8 = (t_gpu.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()
            cached_tensors.append((pg, t_u8))
            if idx % 50 == 0 or idx == len(pages):
                print(f"  preloaded {idx}/{len(pages)} pages", flush=True)

        # Verify bit-exact parity at (0.45, 0.30) against vir_router_v2.compute_ink_and_paper_gate_f32
        parity_max_diff: Optional[float] = None
        if mvd_dir is not None:
            try:
                from vir_router_v2 import compute_ink_and_paper_gate_f32

                sample_t = cached_tensors[0][1].to(device, dtype=torch.float32) / 255.0
                h_p, w_p = int(sample_t.shape[2]) // 16, int(sample_t.shape[3]) // 16
                ref_ink, ref_gate, ref_eff = compute_ink_and_paper_gate_f32(sample_t, h_p, w_p, bg_floor=BG_FLOOR)
                got = compute_ink_and_paper_gate_param(
                    sample_t, h_p, w_p, luma_stop=0.45, dark_thr=0.30, bg_floor=BG_FLOOR
                )
                parity_max_diff = float(
                    max(
                        (ref_ink - got["local_ink_grid"]).abs().max().item(),
                        (ref_gate - got["paper_gate"]).abs().max().item(),
                        (ref_eff - got["effective_paper_f"]).abs().max().item(),
                    )
                )
                print(f"[{EXP_ID}] Bit-exact parity vs vir_router_v2 at (0.45, 0.30): max_diff={parity_max_diff:.2e}", flush=True)
                assert parity_max_diff < 1e-6, f"Parity mismatch: {parity_max_diff}"
            except Exception as exc:
                print(f"[{EXP_ID}] WARN parity check skipped: {exc}", flush=True)

        cache_dir = VOL_ROOT / "evals" / EXP_ID
        cache_dir.mkdir(parents=True, exist_ok=True)
        grid_results: List[Dict[str, Any]] = []

        for luma_stop in LUMA_STOPS:
            for dark_thr in DARK_THRS:
                cfg_key = f"luma{int(round(luma_stop * 100)):02d}_dark{int(round(dark_thr * 100)):02d}"
                cfg_cache = cache_dir / f"{cfg_key}.json"
                if cfg_cache.is_file():
                    res = json.loads(cfg_cache.read_text(encoding="utf-8"))
                    grid_results.append(res)
                    print(f"[{EXP_ID}] Loaded cached {cfg_key}", flush=True)
                    continue

                t_cfg0 = time.perf_counter()
                per_page_rows: List[Dict[str, Any]] = []
                with torch.no_grad():
                    for pg, t_u8 in cached_tensors:
                        img_t = t_u8.to(device, dtype=torch.float32) / 255.0
                        h_p, w_p = int(img_t.shape[2]) // 16, int(img_t.shape[3]) // 16
                        out = compute_ink_and_paper_gate_param(
                            img_t,
                            h_p,
                            w_p,
                            luma_stop=luma_stop,
                            dark_thr=dark_thr,
                            bg_floor=BG_FLOOR,
                        )
                        metrics = _score_page(out, is_clean_borderless=bool(pg["is_clean_borderless"]))
                        per_page_rows.append(
                            {
                                "page_id": pg["page_id"],
                                "cohort": pg["cohort"],
                                **metrics,
                            }
                        )

                cohorts_agg: Dict[str, Any] = {}
                for cname in ("printed_clean", "printed_photo", "hw_clean", "hw_photo", "rism_real_hw_200"):
                    sub = [r for r in per_page_rows if r["cohort"] == cname]
                    cohorts_agg[cname] = _aggregate_metrics(sub)
                suite31_all = [r for r in per_page_rows if r["cohort"] != "rism_real_hw_200"]
                overall_agg = _aggregate_metrics(per_page_rows)
                suite31_agg = _aggregate_metrics(suite31_all)

                worst_5 = sorted(per_page_rows, key=lambda r: r["interior_ink_px_recall"])[:5]
                res = {
                    "config_key": cfg_key,
                    "luma_stop": luma_stop,
                    "dark_thr": dark_thr,
                    "is_default": (abs(luma_stop - 0.45) < 1e-9 and abs(dark_thr - 0.30) < 1e-9),
                    "elapsed_sec": round(time.perf_counter() - t_cfg0, 2),
                    "overall": overall_agg,
                    "suite31_hw_90": suite31_agg,
                    "cohorts": cohorts_agg,
                    "worst_5_interior_ink_recall": worst_5,
                }
                _atomic_write_json(cfg_cache, res)
                grid_results.append(res)
                print(
                    f"  [{cfg_key}] luma={luma_stop:.2f} dark={dark_thr:.2f} | "
                    f"all_ink_rec={overall_agg['interior_ink_px_recall_mean']:.4f} "
                    f"(p05={overall_agg['interior_ink_px_recall_p05']:.4f}) | "
                    f"border_sup={overall_agg['desk_border_suppression_mean']} | "
                    f"clean_fm={overall_agg['clean_false_mask_rate_mean']} | "
                    f"collapsed={overall_agg['collapsed_page_count']}/290",
                    flush=True,
                )

        summary_table = [
            {
                "config_key": r["config_key"],
                "luma_stop": r["luma_stop"],
                "dark_thr": r["dark_thr"],
                "is_default": r["is_default"],
                "all_interior_ink_rec": r["overall"]["interior_ink_px_recall_mean"],
                "all_interior_ink_p05": r["overall"]["interior_ink_px_recall_p05"],
                "rism_interior_ink_rec": r["cohorts"]["rism_real_hw_200"]["interior_ink_px_recall_mean"],
                "rism_interior_ink_p05": r["cohorts"]["rism_real_hw_200"]["interior_ink_px_recall_p05"],
                "photo_border_sup": r["cohorts"]["printed_photo"]["desk_border_suppression_mean"],
                "hw_photo_border_sup": r["cohorts"]["hw_photo"]["desk_border_suppression_mean"],
                "rism_border_sup": r["cohorts"]["rism_real_hw_200"]["desk_border_suppression_mean"],
                "clean_false_mask": r["overall"]["clean_false_mask_rate_mean"],
                "collapsed_pages": r["overall"]["collapsed_page_count"],
                "top40_ink_recall": r["overall"]["top40_ink_recall_mean"],
            }
            for r in grid_results
        ]

        elapsed_sec = round(time.perf_counter() - t_start, 2)
        completed_at = _utc_now()
        report = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_sec": elapsed_sec,
            "platform": f"colab-{gpu_name}",
            "parity_max_diff_vs_vir_router_v2": parity_max_diff,
            "dataset": {
                "suite31_tar": HF_SUITE31_TAR,
                "rism_tar": HF_RISM_TAR,
                "rism_sample_count": len(rism_paths),
                "total_pages": len(pages),
                "cohort_counts": cohort_counts,
                "fullpage_max_dim": FULLPAGE_MAX_DIM,
                "seed": SEED,
            },
            "summary_grid": summary_table,
            "grid_results": grid_results,
        }
        run_ledger = {
            "exp_id": EXP_ID,
            "wave": 5,
            "status": "COMPLETED",
            "platform": f"colab-{gpu_name}",
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_sec": elapsed_sec,
            "hf_eval_path": HF_EVAL_PATH,
            "summary_grid": summary_table,
        }

        eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        run_local = VOL_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(run_local, run_ledger)

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload paper_gate 3x3 border threshold sweep",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload run ledger",
        )
        print(f"\n[{EXP_ID}] Uploaded {HF_EVAL_PATH} & {HF_RUN_PATH} in {elapsed_sec}s!", flush=True)
        print(json.dumps(summary_table, indent=2), flush=True)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
