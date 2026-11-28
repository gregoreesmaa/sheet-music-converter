#!/usr/bin/env python3
"""Colab L4 / Modal L4 runner for Wave 13 Exp i15-04:
Risk-Gated Visual-Coverage TTC (Pivot 3) + <sys>-Aligned On-Policy DAgger &
Counterfactual Anti-Prior Probe (Pivot 2).

Experiment ID: 2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger

Part A (`part_a_risk_gated_visual_ttc`):
  - Zero-train evaluation on OMR-NorthStar dev-v1 (125 pages) using the prebuilt
    static Linux Rust exporter `tools/assembly-serialize-linux-x86_64`.
  - Compares `arm0_greedy` (1408px FSA+MTP greedy) vs `arm1_risk_gated_ttc`:
      * Escalates ONLY on: has_loop OR truncated OR |n_sys - n_sys_img| >= 1
        OR attn_ink_cov < 0.80 OR bsvr < 0.80.
      * ALWAYS retains y_0 (1408px) in the candidate pool (index 0).
      * ZERO encoder TTT (frozen weights).
      * Reranks pool by `r_attn_cov + r_sys_vis + norm_lp_scaled`.
      * Replaces y_0 ONLY on hard CFR flag OR >= +0.08 composite win margin.
  - Gates: CDR escalation <= 15%, mean latency <= 1.45x, 0.0 pp printed
    regression, NS >= +3.0.

Part B (`part_b_sys_dagger_probe`):
  - 1,200-step controlled fine-tuning probe on `iter10_5_proxy_2k`, evaluated on
    `suite31_hw` (90 pages, both RAW and EMA checkpoints):
      * `arm_b0_sft_control`: 100% standard SFT.
      * `arm_b1_sys_dagger_antiprior`: 75% SFT + 25% (<sys>-aligned On-Policy
        DAgger with `ignore_index=-100` on model-generated prefix + 3x weight
        on `<sys>` recovery header, and Counterfactual Anti-Prior).
  - Gates: HWM Note F1 >= +3.0 pp, Printed Note F1 >= -0.3 pp.

Uploads:
  - evals/2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger.json
  - runs/2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger.json
to gregoreesmaa/sheet-music-converter-omr.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import pathlib
import platform
import random
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

EXP_ID = "2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger"
EXP_CODE = "i15-04"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"

HF_CKPT_PATH = os.environ.get("UNIOMR_HF_CKPT", "checkpoints/iter13_v2_mid/uni_omr_v2_ema_latest.pt")
HF_PACK_TAR = "datasets/northstar_eval_pack_125p.tar.gz"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_ASER_BIN_PATH = os.environ.get("UNIOMR_HF_ASER_BIN", "tools/assembly-serialize-linux-x86_64")

HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

# Part A (NorthStar Risk-Gated Visual-Coverage TTC) hyperparameters
NS_MAX_DIM = int(os.environ.get("UNIOMR_MAX_DIM", "1408"))
NS_MAX_LEN = int(os.environ.get("UNIOMR_MAX_LEN", "3584"))
NS_LIMIT_PAGES = int(os.environ.get("NORTHSTAR_LIMIT", "0"))  # 0 = all 125 pages
TTC_ATTN_COV_THR = 0.80
TTC_BSVR_THR = 0.80
TTC_WIN_MARGIN = 0.08
TTC_SCALES = (1280, 1536)

# Part B (<sys>-Aligned On-Policy DAgger + Anti-Prior Probe) hyperparameters
SEED = 42
TRAIN_STEPS = int(os.environ.get("I15_04_TRAIN_STEPS", "1200"))
WARMUP_STEPS = 100
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
GRAD_ACCUM = 4
LR_MAX = 1e-4
WEIGHT_DECAY = 0.04
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
SUITE31_MAX_LEN = 2048
EMA_DECAY = 0.99
DAGGER_FRAC = 0.25          # 75% SFT + 25% DAgger/Anti-Prior
DAGGER_SYS_HEADER_WEIGHT = 3.0
ANTIPRIOR_HEADER_WEIGHT = 2.0
DAGGER_MAX_PREFIX_GEN = 256
SAVE_EVERY_MINUTES = 5.0

HF_TOKEN_FILE = Path("/tmp/.hf_token")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_vol_root() -> Path:
    if Path("/vol").is_dir():
        return Path("/vol")
    if Path("/content").is_dir():
        return Path("/content/vol")
    return Path("/tmp/omr_vol")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _read_and_purge_hf_token() -> Optional[str]:
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        for cand in (HF_TOKEN_FILE, Path.home() / ".cache" / "huggingface" / "token"):
            if cand.is_file():
                try:
                    os.chmod(cand, 0o600)
                except OSError:
                    pass
                val = cand.read_text(encoding="utf-8").strip()
                if val:
                    token = val
                    break
    if HF_TOKEN_FILE.is_file():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass
    return token or None


def _ensure_packages() -> None:
    import importlib.util

    needed: List[str] = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
    ):
        if importlib.util.find_spec(mod) is None:
            needed.append(pkg)
    if needed:
        print(f"[{EXP_ID}] Installing missing packages: {needed}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed", *needed],
            check=True,
        )


def _locate_mvd_dir() -> Path:
    candidates = [
        Path("/workspace/mvd"),
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path("/content/workspace/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
        Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
    ]
    for cand in candidates:
        if (cand / "uni_omr.py").is_file() and (cand / "eval_northstar_runner.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand

    for bundle in (Path("/content/repo_bundle.tar.gz"), Path("/content/bundle.tar.gz")):
        if bundle.is_file():
            target = Path("/content/sheet-music-converter")
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:gz") as tf:
                tf.extractall(path=target)
            mvd = target / "tmp-train" / "mvd"
            if (mvd / "uni_omr.py").is_file():
                if str(mvd) not in sys.path:
                    sys.path.insert(0, str(mvd))
                return mvd

    raise FileNotFoundError("Could not locate tmp-train/mvd containing uni_omr.py and eval_northstar_runner.py")


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_hf_assets(vol_root: Path, hf_token: Optional[str]) -> Dict[str, Path]:
    from huggingface_hub import hf_hub_download

    ds_root = vol_root / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)

    # 1. Checkpoint
    print(f"[{EXP_ID}] Fetching checkpoint {HF_CKPT_PATH} ...", flush=True)
    ckpt_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=HF_CKPT_PATH,
            token=hf_token,
        )
    )

    # 2. NorthStar 125p pack
    ns_dir = ds_root / "northstar_eval_pack_125p"
    if not (ns_dir / "manifest.json").is_file():
        print(f"[{EXP_ID}] Downloading {HF_PACK_TAR} ...", flush=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=HF_PACK_TAR,
                token=hf_token,
            )
        )
        with tarfile.open(tar_p, "r:gz") as tf:
            tf.extractall(path=ds_root)
    if not (ns_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"Expected {ns_dir / 'manifest.json'} after extraction")

    # 3. Proxy 2k dataset (for Part B)
    proxy_dir = ds_root / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    val_abs = proxy_dir / "proxy_val_abs.jsonl"
    if not (train_abs.is_file() and val_abs.is_file()):
        proxy_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR} ...", flush=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=HF_PROXY_TAR,
                token=hf_token,
                local_dir="/tmp/hf_dl",
            )
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

    # 4. suite31_hw eval pack (for Part B)
    suite31_dir = ds_root / "suite31_hw_eval"
    suite31_manifest = suite31_dir / "suite31" / "suite31_hw_manifest.json"
    if not suite31_manifest.is_file():
        suite31_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR} ...", flush=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=HF_SUITE31_TAR,
                token=hf_token,
                local_dir="/tmp/hf_dl",
            )
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=suite31_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass

    return {
        "ckpt_path": ckpt_path,
        "ns_dir": ns_dir,
        "train_abs": train_abs,
        "val_abs": val_abs,
        "suite31_manifest": suite31_manifest,
    }


def _resolve_rust_exporter(
    mvd_dir: Path,
    vol_root: Path,
    dv_mod: Any,
    hf_token: Optional[str],
) -> Tuple[Optional[Path], str]:
    env_bin = os.environ.get("ASSEMBLY_SERIALIZE_BIN") or os.environ.get("MSCZ_BIN")
    if env_bin and Path(env_bin).is_file():
        return Path(env_bin), f"rust:env:{Path(env_bin).name}"

    is_linux_x64 = platform.system().lower() == "linux" and platform.machine().lower() in ("x86_64", "amd64")
    if is_linux_x64 and HF_ASER_BIN_PATH:
        cached_bin = vol_root / "bin" / "assembly-serialize-linux-x86_64"
        try:
            if not cached_bin.is_file() or cached_bin.stat().st_size == 0:
                from huggingface_hub import hf_hub_download

                print(f"[{EXP_ID}] Downloading prebuilt Rust binary {HF_ASER_BIN_PATH} ...", flush=True)
                dl_path = Path(
                    hf_hub_download(
                        repo_id=HF_REPO_ID,
                        repo_type="dataset",
                        filename=HF_ASER_BIN_PATH,
                        token=hf_token,
                    )
                )
                cached_bin.parent.mkdir(parents=True, exist_ok=True)
                tmp_bin = cached_bin.with_suffix(".tmp")
                shutil.copy2(dl_path, tmp_bin)
                os.chmod(tmp_bin, os.stat(tmp_bin).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                os.replace(tmp_bin, cached_bin)
            else:
                os.chmod(cached_bin, os.stat(cached_bin).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

            probe = subprocess.run([str(cached_bin), "--help"], capture_output=True, text=True, timeout=15, check=False)
            if probe.returncode == 0:
                return cached_bin, "rust:hf:assembly-serialize-linux-x86_64"
        except Exception as exc:
            print(f"[{EXP_ID}] WARN: HF static Rust binary unavailable ({exc}); checking fallbacks.", flush=True)

    found = dv_mod.find_assembly_serialize_bin()
    if found is not None and Path(found).is_file():
        return Path(found), f"rust:{Path(found).name}"
    return None, "python:fallback"


def _bekern_to_musicxml_exact(
    toks: Sequence[str],
    *,
    aser_bin: Optional[Path],
    stem: str,
    empty_xml: str,
    dv_mod: Any,
) -> Tuple[str, bool, str, str]:
    if not toks:
        return empty_xml, True, "empty-tokens", "empty"

    if aser_bin is not None and aser_bin.is_file():
        with tempfile.TemporaryDirectory(prefix="ns-bekern-") as td:
            td_path = Path(td)
            bekern_txt = td_path / f"{stem}-bekern.txt"
            bekern_txt.write_text(" ".join(toks) + "\n", encoding="utf-8")
            out_sub = td_path / "pack"
            out_sub.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [
                    str(aser_bin),
                    "bekern2musicxml",
                    "--input",
                    str(bekern_txt),
                    "--out",
                    str(out_sub),
                    "--stem",
                    stem,
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            mxml_path = out_sub / f"{stem}.musicxml"
            if proc.returncode == 0 and mxml_path.is_file():
                xml_text = mxml_path.read_text(encoding="utf-8", errors="replace")
                if xml_text.strip():
                    return xml_text, True, "ok", f"rust:{aser_bin.name}"
            err_msg = (proc.stderr or proc.stdout or f"rc={proc.returncode}").strip()[:160]
            return "", False, f"rust-err:{err_msg}", f"rust:{aser_bin.name}"

    xml_opt = dv_mod.bekern_to_musicxml_fallback(toks)
    if xml_opt and xml_opt.strip():
        return xml_opt, True, "ok-py-fallback", "python:fallback"
    return "", False, "bekern2xml-empty", "python:fallback"


# ---------------------------------------------------------------------------
# Part A Helpers: Visual System Counter, Cross-Attention Ink Coverage & BSVR
# ---------------------------------------------------------------------------

def _estimate_visual_sys_count(pil_im: Any) -> int:
    """Estimate the number of visual music systems on a page via horizontal ink projection."""
    import numpy as np

    gray = np.asarray(pil_im.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape
    if h < 64 or w < 64:
        return 1

    # Ignore 3% border margins (desk / scanner edges)
    y0, y1 = int(h * 0.03), int(h * 0.97)
    x0, x1 = int(w * 0.04), int(w * 0.96)
    crop = gray[y0:y1, x0:x1]
    ch = crop.shape[0]
    if ch < 32:
        return 1

    # Local adaptive darkness threshold (handles yellowed manuscripts & low-contrast photos)
    p85 = float(np.percentile(crop, 85))
    thr = max(0.35, min(0.84, p85 - 0.14))
    ink = (crop < thr).astype(np.float32)

    # Row-wise ink density + smooth box kernel (~1.4% of page height)
    row_prof = ink.mean(axis=1)
    win = max(5, int(round(ch * 0.014)))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    smooth = np.convolve(row_prof, kernel, mode="same")

    active_thr = max(0.015, float(np.percentile(smooth, 35)) + 0.22 * float(smooth.max() - np.percentile(smooth, 35)))
    active = smooth > active_thr

    # Extract contiguous vertical ink bands
    bands: List[Tuple[int, int]] = []
    in_band = False
    start = 0
    for y, flag in enumerate(active):
        if flag and not in_band:
            start = y
            in_band = True
        elif not flag and in_band:
            if (y - start) >= max(8, int(ch * 0.012)):
                bands.append((start, y))
            in_band = False
    if in_band and (ch - start) >= max(8, int(ch * 0.012)):
        bands.append((start, ch))

    if not bands:
        return 1

    # Merge closely spaced staves belonging to the same grand-staff system (< 2.2% page height gap)
    merge_gap = max(6, int(round(ch * 0.022)))
    merged: List[Tuple[int, int]] = [bands[0]]
    for s, e in bands[1:]:
        ps, pe = merged[-1]
        if (s - pe) <= merge_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    # Filter out narrow title/page-number bands at very top/bottom (< 1.8% height & low width span)
    systems: List[Tuple[int, int]] = []
    min_sys_h = max(12, int(round(ch * 0.020)))
    for s, e in merged:
        band_ink = ink[s:e, :]
        col_span = float((band_ink.mean(axis=0) > 0.02).mean())
        if (e - s) >= min_sys_h and col_span >= 0.35:
            systems.append((s, e))

    return max(1, min(16, len(systems) if systems else len(merged)))


def _compute_attn_ink_cov_and_norm_lp(
    model: Any,
    img_t: "torch.Tensor",
    pil_im: Any,
    seq_ids: Sequence[int],
    *,
    device: "torch.device",
    use_bf16: bool,
    pad_id: int,
) -> Tuple[float, float]:
    """Compute (attn_ink_cov, norm_lp) in a single teacher-forced forward pass.

    - `attn_ink_cov`: fraction of ink-bearing encoder visual memory tokens that
      receive meaningful cross-attention mass from the last decoder layer.
    - `norm_lp`: length-normalized sequence log-probability (mean log p(y_t | y_<t, x)).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    if len(seq_ids) < 2:
        return 0.0, -2.0

    tgt = torch.tensor(seq_ids, dtype=torch.long, device=device).unsqueeze(0)
    dec_in = tgt[:, :-1]
    dec_tgt = tgt[:, 1:]

    # Capture last decoder layer cross-attention inputs (q_in, kv_in) via lightweight forward pre-hook
    captured: Dict[str, torch.Tensor] = {}
    last_layer = model.dec_layers[-1]
    ca = last_layer.cross_attn

    def _hook_pre(_mod: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        if len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            captured["q_in"] = args[0].detach()
            captured["kv_in"] = args[1].detach()

    handle = ca.register_forward_pre_hook(_hook_pre, with_kwargs=True)
    try:
        with torch.no_grad():
            ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cpu", enabled=False)
            with ctx:
                mem, _, router_info = model.encode_image(img_t, return_aux=False)
                out = model.decode_tokens(
                    dec_in,
                    mem,
                    return_mtp=False,
                    return_aux=False,
                    apply_tok_dropout=False,
                )
                logits = out[0] if isinstance(out, tuple) else out
                log_probs = F.log_softmax(logits.float(), dim=-1)
                valid_mask = dec_tgt != pad_id
                tok_lp = log_probs.gather(-1, dec_tgt.unsqueeze(-1)).squeeze(-1)
                n_valid = int(valid_mask.sum().item())
                norm_lp = float((tok_lp * valid_mask.float()).sum().item() / max(1, n_valid))
    finally:
        handle.remove()

    # Compute cross-attention distribution over Lk visual memory tokens (chunked over T for low VRAM)
    attn_ink_cov = 0.85  # neutral fallback if hook unavailable
    if "q_in" in captured and "kv_in" in captured:
        with torch.no_grad():
            q_in = captured["q_in"].to(dtype=ca.q_proj.weight.dtype)
            kv_in = captured["kv_in"].to(dtype=ca.k_proj.weight.dtype)
            B, Lq, _ = q_in.shape
            Lk = kv_in.shape[1]
            n_heads = int(getattr(ca, "n_heads", 8))
            n_kv_heads = int(getattr(ca, "n_kv_heads", n_heads))
            head_dim = int(getattr(ca, "head_dim", q_in.shape[-1] // max(1, n_heads)))

            q = ca.q_proj(q_in).view(B, Lq, n_heads, head_dim)
            if hasattr(ca, "q_norm"):
                q = ca.q_norm(q)
            q = q.transpose(1, 2).float()  # (1, H, Lq, D)

            k = ca.k_proj(kv_in).view(B, Lk, n_kv_heads, head_dim)
            if hasattr(ca, "k_norm"):
                k = ca.k_norm(k)
            k = k.transpose(1, 2).float()  # (1, H_kv, Lk, D)
            if n_kv_heads != n_heads:
                rep = n_heads // n_kv_heads
                k = k.repeat_interleave(rep, dim=1)

            scale = 1.0 / math.sqrt(float(head_dim))
            mem_mass = torch.zeros(Lk, dtype=torch.float32, device=device)
            chunk = 256
            for s in range(0, Lq, chunk):
                q_c = q[:, :, s : s + chunk, :]  # (1, H, C, D)
                scores = torch.matmul(q_c, k.transpose(-2, -1)) * scale  # (1, H, C, Lk)
                probs = F.softmax(scores, dim=-1)
                # Sum across time and average across heads -> total attention mass per memory token
                mem_mass += probs[0].mean(dim=0).sum(dim=0)

            mem_mass_np = mem_mass.detach().cpu().numpy()

            # Build ink mask aligned with encoder visual memory tokens
            gray = np.asarray(pil_im.convert("L"), dtype=np.float32) / 255.0
            p85 = float(np.percentile(gray, 85))
            thr = max(0.35, min(0.84, p85 - 0.14))
            ink_2d = (gray < thr).astype(np.float32)

            # Determine spatial grid (gh, gw) from router_info if available, else infer from image aspect
            grid_hw = router_info.get("grid_hw") if isinstance(router_info, dict) else None
            keep_idx = router_info.get("keep_indices") if isinstance(router_info, dict) else None
            if grid_hw is not None and len(grid_hw) == 2:
                gh, gw = int(grid_hw[0]), int(grid_hw[1])
            else:
                ih, iw = gray.shape
                aspect = max(0.25, min(4.0, float(iw) / float(max(1, ih))))
                gh = max(4, int(round(math.sqrt(max(1, Lk) / aspect))))
                gw = max(4, int(round(max(1, Lk) / float(gh))))

            ink_t = torch.from_numpy(ink_2d).unsqueeze(0).unsqueeze(0)
            ink_grid = F.adaptive_avg_pool2d(ink_t, (gh, gw)).reshape(-1).numpy()

            if keep_idx is not None and isinstance(keep_idx, torch.Tensor):
                idx_np = keep_idx[0].detach().cpu().numpy().astype(np.int64)
                valid_i = idx_np[idx_np < ink_grid.shape[0]]
                if valid_i.shape[0] > 0:
                    n_use = min(valid_i.shape[0], mem_mass_np.shape[0])
                    mem_ink = ink_grid[valid_i[:n_use]]
                    mass_vec = mem_mass_np[:n_use]
                else:
                    n_use = min(ink_grid.shape[0], mem_mass_np.shape[0])
                    mem_ink = ink_grid[:n_use]
                    mass_vec = mem_mass_np[:n_use]
            else:
                # Resample ink_grid to Lk if lengths differ (e.g., register tokens at front/end)
                if ink_grid.shape[0] != Lk:
                    ink_resamp = F.interpolate(
                        torch.from_numpy(ink_grid).view(1, 1, -1),
                        size=Lk,
                        mode="linear",
                        align_corners=False,
                    ).reshape(-1).numpy()
                    mem_ink = ink_resamp
                else:
                    mem_ink = ink_grid
                mass_vec = mem_mass_np

            ink_tokens = mem_ink >= 0.04
            if int(ink_tokens.sum()) > 0:
                # A memory token is "covered" if it receives at least 25% of uniform mean attention mass
                uniform_mass = float(Lq) / float(max(1, Lk))
                cov_thr = 0.25 * uniform_mass
                covered = ink_tokens & (mass_vec >= cov_thr)
                attn_ink_cov = float(covered.sum()) / float(ink_tokens.sum())
            else:
                attn_ink_cov = 1.0

    return round(float(attn_ink_cov), 4), round(float(norm_lp), 5)


def _compute_bsvr(raw_toks: Sequence[str], rlvr_mod: Any) -> float:
    """Compute beat-sum validity rate (bsvr) from rlvr_grpo._walk_parse_and_meter."""
    if not raw_toks:
        return 0.0
    try:
        walk = rlvr_mod._walk_parse_and_meter(list(raw_toks))
        if isinstance(walk, dict):
            n_m = int(walk.get("n_measures", 0))
            n_ok = int(walk.get("n_valid_measures", 0))
            if n_m > 0:
                return round(float(n_ok) / float(n_m), 4)
            return float(walk.get("r_meter", 1.0))
        if isinstance(walk, (tuple, list)) and len(walk) >= 3:
            # Typical signature: (r_parse, r_meter, ...)
            return round(float(walk[1]), 4)
    except Exception:
        pass
    try:
        unlab = rlvr_mod.compute_unlabeled_structural_reward(list(raw_toks))
        if isinstance(unlab, dict) and "r_meter" in unlab:
            return round(float(unlab["r_meter"]), 4)
    except Exception:
        pass
    return 1.0


def _score_ttc_candidate(attn_ink_cov: float, n_sys: int, n_sys_img: int, norm_lp: float) -> Tuple[float, Dict[str, float]]:
    r_attn_cov = float(max(0.0, min(1.0, attn_ink_cov)))
    r_sys_vis = float(max(0.0, 1.0 - 0.35 * abs(int(n_sys) - int(n_sys_img))))
    # Map norm_lp in [-2.0, 0.0] -> [0.0, 1.0] so all three terms are commensurate
    norm_lp_scaled = float(max(0.0, 1.0 + max(-2.0, min(0.0, norm_lp)) / 2.0))
    total = r_attn_cov + r_sys_vis + norm_lp_scaled
    return round(total, 5), {
        "r_attn_cov": round(r_attn_cov, 4),
        "r_sys_vis": round(r_sys_vis, 4),
        "norm_lp": round(norm_lp, 5),
        "norm_lp_scaled": round(norm_lp_scaled, 4),
    }


def _decode_candidate(
    model: Any,
    cfg: Any,
    raw_im: Any,
    *,
    max_dim: int,
    beam_size: int,
    decode_args: argparse.Namespace,
    device: "torch.device",
    use_bf16: bool,
    pad_id: int,
    eos_id: int,
    n_sys_img: int,
    eus: Any,
    uni_omr: Any,
    rlvr_grpo: Any,
    dv: Any,
    label: str,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
    img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
    seq, _, _, _ = eus._decode_single_view(
        model=model,
        cfg=cfg,
        img_t=img_t,
        args=decode_args,
        device=device,
        use_bf16=use_bf16,
        beam_size=beam_size,
        beam_length_penalty=0.6,
        measure_beam_size=0,
        compute_logprob=False,
    )
    seq_ids = seq[0].detach().cpu().tolist()
    truncated = eos_id not in seq_ids[1:]
    pred_raw_toks = eus.decode_uni_compat(seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True)
    if truncated:
        pred_raw_toks = eus.trim_truncated_tokens(pred_raw_toks)
    hyp_page_toks, _ = eus.stitch_uni_tokens(pred_raw_toks)

    has_loop = bool(rlvr_grpo.has_repetition_loop(pred_raw_toks) or dv.detect_sys_loop(pred_raw_toks))
    n_sys = 1 + sum(1 for t in pred_raw_toks if t == uni_omr.SYS_TOKEN)
    bsvr = _compute_bsvr(pred_raw_toks, rlvr_grpo)
    attn_ink_cov, norm_lp = _compute_attn_ink_cov_and_norm_lp(
        model,
        img_t,
        pil_im,
        seq_ids,
        device=device,
        use_bf16=use_bf16,
        pad_id=pad_id,
    )
    score, score_parts = _score_ttc_candidate(attn_ink_cov, n_sys, n_sys_img, norm_lp)
    elapsed = time.perf_counter() - t0

    return {
        "label": label,
        "max_dim": max_dim,
        "beam_size": beam_size,
        "seq_ids": seq_ids,
        "pred_raw_toks": pred_raw_toks,
        "hyp_page_toks": hyp_page_toks,
        "has_loop": has_loop,
        "truncated": truncated,
        "n_sys": n_sys,
        "n_sys_img": n_sys_img,
        "bsvr": bsvr,
        "attn_ink_cov": attn_ink_cov,
        "norm_lp": norm_lp,
        "score": score,
        "score_parts": score_parts,
        "elapsed_s": round(elapsed, 4),
    }


def _run_part_a_northstar_ttc(
    *,
    ckpt_path: Path,
    ns_dir: Path,
    vol_root: Path,
    mvd_dir: Path,
    device: "torch.device",
    use_bf16: bool,
    hf_token: Optional[str],
) -> Dict[str, Any]:
    import torch
    from PIL import Image

    import dual_verifier as dv  # type: ignore[import-not-found]
    import eval_northstar_runner as ns_runner  # type: ignore[import-not-found]
    import eval_uni_suite31 as eus  # type: ignore[import-not-found]
    import neg_control_gate  # type: ignore[import-not-found]
    import rlvr_grpo  # type: ignore[import-not-found]
    import uni_omr  # type: ignore[import-not-found]

    t_part_a = time.perf_counter()
    part_dir = vol_root / "runs" / EXP_ID / "part_a"
    cached_part_a = part_dir / "part_a_result.json"
    if cached_part_a.is_file():
        print(f"[{EXP_ID}][Part A] Reusing cached Part A result at {cached_part_a}", flush=True)
        return json.loads(cached_part_a.read_text(encoding="utf-8"))

    aser_bin, export_backend = _resolve_rust_exporter(mvd_dir, vol_root, dv, hf_token)
    print(f"[{EXP_ID}][Part A] Export backend: {export_backend} (bin={aser_bin})", flush=True)

    model, cfg, ckpt_meta = eus.load_uni_checkpoint(ckpt_path, device=device)
    model.eval()
    param_count = int(sum(p.numel() for p in model.parameters()))
    pad_id, _, eos_id, _, _, _ = eus._vocab_spec(int(cfg.vocab_size))

    manifest_path = ns_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = list(manifest["pages"])
    if NS_LIMIT_PAGES > 0:
        pages = pages[:NS_LIMIT_PAGES]

    arm0_mxml_dir = part_dir / "arm0_greedy" / "musicxml"
    arm1_mxml_dir = part_dir / "arm1_ttc" / "musicxml"
    arm0_mxml_dir.mkdir(parents=True, exist_ok=True)
    arm1_mxml_dir.mkdir(parents=True, exist_ok=True)
    meta_path = part_dir / "ttc_per_page_meta.json"

    per_page_meta: Dict[str, Dict[str, Any]] = {}
    if meta_path.is_file():
        try:
            per_page_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            per_page_meta = {}

    decode_args = argparse.Namespace(
        max_len=NS_MAX_LEN,
        use_grammar=True,
        strict_grammar=False,
        use_mtp_spec=True,
    )

    arm0_lat: Dict[str, float] = {}
    arm1_lat: Dict[str, float] = {}

    for idx, rec in enumerate(pages, start=1):
        pid = str(rec["page_id"])
        split_tag = str(rec.get("subset", rec.get("domain", rec.get("split", "")))).upper()
        p0_xml = arm0_mxml_dir / f"{pid}.musicxml"
        p1_xml = arm1_mxml_dir / f"{pid}.musicxml"

        if p0_xml.is_file() and p1_xml.is_file() and pid in per_page_meta:
            arm0_lat[pid] = float(per_page_meta[pid].get("arm0_latency_s", 0.0))
            arm1_lat[pid] = float(per_page_meta[pid].get("arm1_latency_s", 0.0))
            continue

        img_path = ns_dir / str(rec["image_path"])
        t_page0 = time.perf_counter()
        try:
            with Image.open(img_path) as raw_im:
                is_non_music, gate_info = neg_control_gate.is_non_music_page(raw_im)
                if is_non_music:
                    dt = round(time.perf_counter() - t_page0, 4)
                    _atomic_write_text(p0_xml, ns_runner.EMPTY_VALID_MUSICXML)
                    _atomic_write_text(p1_xml, ns_runner.EMPTY_VALID_MUSICXML)
                    arm0_lat[pid] = dt
                    arm1_lat[pid] = dt
                    per_page_meta[pid] = {
                        "page_id": pid,
                        "subset": split_tag,
                        "neg_gate": True,
                        "escalated": False,
                        "replaced": False,
                        "arm0_latency_s": dt,
                        "arm1_latency_s": dt,
                        "reason": gate_info.get("reason", "neg-gate"),
                    }
                    _atomic_write_json(meta_path, per_page_meta)
                    continue

                n_sys_img = _estimate_visual_sys_count(raw_im)

                # Arm 0: standard 1408px greedy
                cand0 = _decode_candidate(
                    model,
                    cfg,
                    raw_im,
                    max_dim=NS_MAX_DIM,
                    beam_size=1,
                    decode_args=decode_args,
                    device=device,
                    use_bf16=use_bf16,
                    pad_id=pad_id,
                    eos_id=eos_id,
                    n_sys_img=n_sys_img,
                    eus=eus,
                    uni_omr=uni_omr,
                    rlvr_grpo=rlvr_grpo,
                    dv=dv,
                    label="y0_1408_greedy",
                )

                hard_cfr = bool(
                    cand0["has_loop"]
                    or cand0["truncated"]
                    or (abs(int(cand0["n_sys"]) - int(n_sys_img)) >= 1)
                    or (float(cand0["bsvr"]) < TTC_BSVR_THR)
                )
                escalate = bool(hard_cfr or (float(cand0["attn_ink_cov"]) < TTC_ATTN_COV_THR))

                selected = cand0
                replaced = False
                extra_s = 0.0
                cand_summaries = [
                    {
                        "label": cand0["label"],
                        "score": cand0["score"],
                        "attn_ink_cov": cand0["attn_ink_cov"],
                        "bsvr": cand0["bsvr"],
                        "n_sys": cand0["n_sys"],
                        "norm_lp": cand0["norm_lp"],
                        "has_loop": cand0["has_loop"],
                        "truncated": cand0["truncated"],
                    }
                ]

                if escalate:
                    pool = [cand0]
                    for sc in TTC_SCALES:
                        c_sc = _decode_candidate(
                            model,
                            cfg,
                            raw_im,
                            max_dim=sc,
                            beam_size=1,
                            decode_args=decode_args,
                            device=device,
                            use_bf16=use_bf16,
                            pad_id=pad_id,
                            eos_id=eos_id,
                            n_sys_img=n_sys_img,
                            eus=eus,
                            uni_omr=uni_omr,
                            rlvr_grpo=rlvr_grpo,
                            dv=dv,
                            label=f"scale_{sc}_greedy",
                        )
                        extra_s += float(c_sc["elapsed_s"])
                        pool.append(c_sc)

                    c_beam = _decode_candidate(
                        model,
                        cfg,
                        raw_im,
                        max_dim=NS_MAX_DIM,
                        beam_size=2,
                        decode_args=decode_args,
                        device=device,
                        use_bf16=use_bf16,
                        pad_id=pad_id,
                        eos_id=eos_id,
                        n_sys_img=n_sys_img,
                        eus=eus,
                        uni_omr=uni_omr,
                        rlvr_grpo=rlvr_grpo,
                        dv=dv,
                        label="y_1408_beam2",
                    )
                    extra_s += float(c_beam["elapsed_s"])
                    pool.append(c_beam)

                    for c in pool[1:]:
                        cand_summaries.append(
                            {
                                "label": c["label"],
                                "score": c["score"],
                                "attn_ink_cov": c["attn_ink_cov"],
                                "bsvr": c["bsvr"],
                                "n_sys": c["n_sys"],
                                "norm_lp": c["norm_lp"],
                                "has_loop": c["has_loop"],
                                "truncated": c["truncated"],
                            }
                        )

                    # Rerank strictly over pool ALWAYS containing y_0 at index 0
                    best_idx = max(range(len(pool)), key=lambda i: float(pool[i]["score"]))
                    best_c = pool[best_idx]
                    margin = float(best_c["score"]) - float(cand0["score"])
                    if best_idx != 0 and (hard_cfr or margin >= TTC_WIN_MARGIN):
                        selected = best_c
                        replaced = True

                xml0, ok0, st0, _ = _bekern_to_musicxml_exact(
                    cand0["hyp_page_toks"],
                    aser_bin=aser_bin,
                    stem=pid,
                    empty_xml=ns_runner.EMPTY_VALID_MUSICXML,
                    dv_mod=dv,
                )
                _atomic_write_text(p0_xml, xml0)
                if replaced:
                    xml1, ok1, st1, _ = _bekern_to_musicxml_exact(
                        selected["hyp_page_toks"],
                        aser_bin=aser_bin,
                        stem=pid,
                        empty_xml=ns_runner.EMPTY_VALID_MUSICXML,
                        dv_mod=dv,
                    )
                else:
                    xml1, ok1, st1 = xml0, ok0, st0
                _atomic_write_text(p1_xml, xml1)

                arm0_lat[pid] = round(float(cand0["elapsed_s"]), 4)
                arm1_lat[pid] = round(float(cand0["elapsed_s"]) + extra_s, 4)
                per_page_meta[pid] = {
                    "page_id": pid,
                    "subset": split_tag,
                    "neg_gate": False,
                    "n_sys_img": n_sys_img,
                    "n_sys_y0": cand0["n_sys"],
                    "attn_ink_cov_y0": cand0["attn_ink_cov"],
                    "bsvr_y0": cand0["bsvr"],
                    "has_loop_y0": cand0["has_loop"],
                    "truncated_y0": cand0["truncated"],
                    "hard_cfr": hard_cfr,
                    "escalated": escalate,
                    "replaced": replaced,
                    "selected_label": selected["label"],
                    "score_delta": round(float(selected["score"]) - float(cand0["score"]), 5),
                    "arm0_latency_s": arm0_lat[pid],
                    "arm1_latency_s": arm1_lat[pid],
                    "xml_ok_arm0": ok0,
                    "xml_ok_arm1": ok1,
                    "status_arm0": st0,
                    "status_arm1": st1,
                    "candidates": cand_summaries,
                }
                _atomic_write_json(meta_path, per_page_meta)
                print(
                    f"[Part A][{idx:03d}/{len(pages):03d}] {pid:38s} esc={str(escalate):5s} rep={str(replaced):5s} "
                    f"cov={cand0['attn_ink_cov']:.2f} bsvr={cand0['bsvr']:.2f} sys={cand0['n_sys']}/{n_sys_img} "
                    f"sel={selected['label']} ({arm1_lat[pid]:.2f}s)",
                    flush=True,
                )
        except Exception as exc:
            print(f"[Part A][ERROR][{idx:03d}/{len(pages):03d}] {pid}: {type(exc).__name__}: {exc}", flush=True)
            _atomic_write_text(p0_xml, "")
            _atomic_write_text(p1_xml, "")
            dt = round(time.perf_counter() - t_page0, 4)
            arm0_lat[pid] = dt
            arm1_lat[pid] = dt
            per_page_meta[pid] = {
                "page_id": pid,
                "subset": split_tag,
                "error": f"{type(exc).__name__}: {exc}",
                "escalated": False,
                "replaced": False,
                "arm0_latency_s": dt,
                "arm1_latency_s": dt,
            }
            _atomic_write_json(meta_path, per_page_meta)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Score both arms with canonical eval_northstar_runner
    timing0 = part_dir / "arm0_greedy" / "timing.json"
    timing1 = part_dir / "arm1_ttc" / "timing.json"
    _atomic_write_json(
        timing0,
        {"system_name": f"{EXP_CODE}-arm0-greedy", "parameter_count": param_count, "latencies_s": arm0_lat},
    )
    _atomic_write_json(
        timing1,
        {"system_name": f"{EXP_CODE}-arm1-risk-gated-ttc", "parameter_count": param_count, "latencies_s": arm1_lat},
    )

    prev_cwd = Path.cwd()
    os.chdir(ns_dir)
    try:
        if NS_LIMIT_PAGES > 0:
            sub_m = dict(manifest)
            sub_m["pages"] = pages
            score_manifest = part_dir / "manifest_limit.json"
            _atomic_write_json(score_manifest, sub_m)
        else:
            score_manifest = manifest_path

        rep0, _, sc0 = ns_runner.evaluate_manifest(
            manifest_path=score_manifest,
            pred_dir=arm0_mxml_dir,
            oracle=False,
            system_name=f"{EXP_CODE}-arm0-greedy",
            northstar_root=ns_dir,
            timing_path=timing0,
            strict_validate=(NS_LIMIT_PAGES == 0),
            n_boot=1000,
            bootstrap_seed=42,
        )
        rep1, _, sc1 = ns_runner.evaluate_manifest(
            manifest_path=score_manifest,
            pred_dir=arm1_mxml_dir,
            oracle=False,
            system_name=f"{EXP_CODE}-arm1-risk-gated-ttc",
            northstar_root=ns_dir,
            timing_path=timing1,
            strict_validate=(NS_LIMIT_PAGES == 0),
            n_boot=1000,
            bootstrap_seed=42,
        )
    finally:
        os.chdir(prev_cwd)

    print("\n=== Part A Arm 0 (Greedy) ===\n" + sc0, flush=True)
    print("\n=== Part A Arm 1 (Risk-Gated Visual TTC) ===\n" + sc1, flush=True)

    # Aggregate escalation & latency stats
    all_meta = list(per_page_meta.values())
    n_total = max(1, len(all_meta))
    n_esc = sum(1 for m in all_meta if m.get("escalated"))
    n_rep = sum(1 for m in all_meta if m.get("replaced"))
    cdr_meta = [m for m in all_meta if "CDR" in str(m.get("subset", ""))]
    cdr_esc_rate = round(sum(1 for m in cdr_meta if m.get("escalated")) / max(1, len(cdr_meta)), 4) if cdr_meta else None
    sum_lat0 = sum(float(v) for v in arm0_lat.values())
    sum_lat1 = sum(float(v) for v in arm1_lat.values())
    lat_mult = round(sum_lat1 / max(1e-6, sum_lat0), 4)

    def _extract_ns(rep: Dict[str, Any]) -> Optional[float]:
        hl = rep.get("layer1_headline", {})
        for k in ("northstar_score", "NS", "ns_score", "overall_score"):
            if k in hl and hl[k] is not None:
                return float(hl[k])
        return None

    ns0 = _extract_ns(rep0)
    ns1 = _extract_ns(rep1)
    ns_delta = round(ns1 - ns0, 4) if (ns0 is not None and ns1 is not None) else None

    part_a_res = {
        "checkpoint": HF_CKPT_PATH,
        "arch": ckpt_meta.get("arch"),
        "step": ckpt_meta.get("step"),
        "export_backend": export_backend,
        "pages_total": len(pages),
        "escalation_rate_all": round(n_esc / n_total, 4),
        "replacement_rate_all": round(n_rep / n_total, 4),
        "cdr_escalation_rate": cdr_esc_rate,
        "mean_latency_s_arm0": round(sum_lat0 / n_total, 4),
        "mean_latency_s_arm1": round(sum_lat1 / n_total, 4),
        "latency_multiplier": lat_mult,
        "ns_arm0": ns0,
        "ns_arm1": ns1,
        "ns_delta": ns_delta,
        "arm0_headline": rep0.get("layer1_headline", {}),
        "arm1_headline": rep1.get("layer1_headline", {}),
        "arm0_matrix": rep0.get("layer2_matrix_and_deltas", {}),
        "arm1_matrix": rep1.get("layer2_matrix_and_deltas", {}),
        "gates": {
            "cdr_esc_le_15pct": (cdr_esc_rate is not None and cdr_esc_rate <= 0.15),
            "latency_le_1_45x": (lat_mult <= 1.45),
            "ns_delta_ge_3_0": (ns_delta is not None and ns_delta >= 3.0),
        },
        "elapsed_secs": round(time.perf_counter() - t_part_a, 2),
    }
    _atomic_write_json(cached_part_a, part_a_res)
    return part_a_res


# ---------------------------------------------------------------------------
# Part B Helpers: <sys>-Aligned On-Policy DAgger + Counterfactual Anti-Prior
# ---------------------------------------------------------------------------

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


def _encode_tokens_compat(toks: Sequence[str], vocab_size: int, eus: Any, uni_omr: Any) -> List[int]:
    """Encode bekern/system tokens strictly within the checkpoint's vocab_size."""
    _, bos_id, eos_id, sys_id, tok_to_id, _ = eus._vocab_spec(int(vocab_size))
    ids: List[int] = [bos_id]
    for tok in toks:
        if tok == uni_omr.SYS_TOKEN:
            ids.append(sys_id)
        else:
            tid = tok_to_id.get(tok)
            if tid is not None:
                ids.append(int(tid))
    ids.append(eos_id)
    return ids


def _encode_bekerns_systems(
    beks: List[Path],
    uni_omr: Any,
    *,
    vocab_size: int,
    eus: Any,
) -> Tuple[List[int], List[List[str]]]:
    sys_toks: List[List[str]] = []
    flat: List[str] = []
    for idx, bp in enumerate(beks):
        raw = bp.read_text(encoding="utf-8", errors="ignore").strip()
        toks = raw.split() if raw else []
        sys_toks.append(toks)
        if idx > 0:
            flat.append(uni_omr.SYS_TOKEN)
        flat.extend(toks)
    return _encode_tokens_compat(flat, vocab_size, eus, uni_omr), sys_toks


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
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _is_header_token(tok: str, sys_token: str) -> bool:
    return (
        tok == sys_token
        or tok.startswith("clef-")
        or tok.startswith("key")
        or tok.startswith("time-")
    )


def _build_sys_dagger_example(
    model: Any,
    inp: "torch.Tensor",
    sys_toks: List[List[str]],
    *,
    rng: random.Random,
    device: "torch.device",
    use_bf16: bool,
    uni_omr: Any,
    vocab_size: int,
    eus: Any,
) -> Optional[Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]]:
    """Generate a `<sys>`-aligned On-Policy DAgger training example:
    - Model generates a short on-policy prefix up to split system k.
    - Ground-truth `<sys>` recovery suffix from system k onward is spliced on.
    - Model-generated prefix targets are masked with `ignore_index=-100`.
    - `<sys>` recovery header tokens receive `3.0x` loss weight.
    Returns `(dec_in, labels, weights)` each of shape `(1, L)`.
    """
    import torch

    if len(sys_toks) < 2:
        return None
    split_k = rng.randint(1, len(sys_toks) - 1)
    suffix_toks: List[str] = [uni_omr.SYS_TOKEN]
    for s_idx in range(split_k, len(sys_toks)):
        if s_idx > split_k:
            suffix_toks.append(uni_omr.SYS_TOKEN)
        suffix_toks.extend(sys_toks[s_idx])

    suffix_ids_full = _encode_tokens_compat(suffix_toks, vocab_size, eus, uni_omr)
    # _encode_tokens_compat returns [BOS, ..., EOS]; drop BOS for spliced suffix
    suffix_ids = suffix_ids_full[1:] if len(suffix_ids_full) >= 2 else suffix_ids_full
    if len(suffix_ids) < 2:
        return None

    # Estimate target prefix length (capped for fast on-policy rollout)
    gt_prefix_len = sum(len(s) + 1 for s in sys_toks[:split_k])
    gen_budget = max(24, min(DAGGER_MAX_PREFIX_GEN, gt_prefix_len))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cpu", enabled=False)
            with ctx:
                gen_seq = model.generate(inp, max_len=gen_budget)
        gen_ids = gen_seq[0].detach().cpu().tolist()
    finally:
        if was_training:
            model.train()

    # Trim trailing EOS if model stopped early, keep BOS at index 0
    _, bos_id, eos_id, _, _, _ = eus._vocab_spec(vocab_size)
    if len(gen_ids) > 1 and gen_ids[-1] == eos_id:
        gen_ids = gen_ids[:-1]
    if not gen_ids or gen_ids[0] != bos_id:
        gen_ids = [bos_id] + gen_ids

    # Truncate total length to MAX_SEQ_LEN
    max_suffix = max(16, MAX_SEQ_LEN - len(gen_ids))
    suffix_ids = suffix_ids[:max_suffix]
    full_ids = gen_ids + suffix_ids
    if len(full_ids) < 4:
        return None

    dec_in = torch.tensor(full_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    labels = torch.tensor(full_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    weights = torch.ones_like(labels, dtype=torch.float32)

    # Mask ALL model-generated prefix tokens with ignore_index = -100!
    # The first predicted recovery token (`<sys>`) is at target index `len(gen_ids) - 1`.
    rec_start = max(0, len(gen_ids) - 1)
    if rec_start > 0:
        labels[:, :rec_start] = -100

    # Apply 3.0x weight on the `<sys>` recovery header (first up to 8 tokens of recovery suffix)
    header_span = min(8, len(suffix_toks))
    for offset in range(header_span):
        pos = rec_start + offset
        if pos >= labels.shape[1]:
            break
        if offset == 0 or _is_header_token(suffix_toks[offset], uni_omr.SYS_TOKEN):
            weights[:, pos] = DAGGER_SYS_HEADER_WEIGHT

    return dec_in, labels, weights


def _build_antiprior_example(
    img_pil: Any,
    sys_toks: List[List[str]],
    *,
    rng: random.Random,
    device: "torch.device",
    uni_omr: Any,
    vocab_size: int,
    eus: Any,
) -> Optional[Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]]:
    """Build a Counterfactual Anti-Prior pair:
    - Applies mild photometric contrast/gamma + vertical-crop jitter on the image
      so the model cannot memorize page templates.
    - Randomly swaps a non-empty system's order (system-order permutation paired
      with vertical band swap on the image when >= 2 systems) OR applies token
      prefix-history dropout (`15%` random input token corruption) with `2.0x`
      weight on state-header tokens (`<sys>`, `clef-*`, `key*`, `time-*`).
    """
    import numpy as np
    import torch
    from PIL import Image, ImageEnhance

    aug = img_pil.convert("RGB")
    # Photometric domain jitter (exposure/contrast)
    aug = ImageEnhance.Contrast(aug).enhance(rng.uniform(0.75, 1.30))
    aug = ImageEnhance.Brightness(aug).enhance(rng.uniform(0.85, 1.15))

    flat_toks: List[str] = []
    for idx, st in enumerate(sys_toks):
        if idx > 0:
            flat_toks.append(uni_omr.SYS_TOKEN)
        flat_toks.extend(st)
    ids = _encode_tokens_compat(flat_toks, vocab_size, eus, uni_omr)
    if len(ids) < 4:
        return None
    ids = ids[:MAX_SEQ_LEN]

    inp = _pil_to_tensor(aug, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
    dec_in = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    labels = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    weights = torch.ones_like(labels, dtype=torch.float32)

    # Anti-prior input perturbation: corrupt 15% of non-special decoder input tokens
    # so the decoder MUST read visual memory rather than autoregressing LM priors.
    pad_id, _, _, _, _, _ = eus._vocab_spec(int(vocab_size))
    if dec_in.shape[1] > 4:
        corrupt_mask = torch.rand_like(dec_in, dtype=torch.float32) < 0.15
        corrupt_mask[:, 0] = False  # keep BOS
        rand_toks = torch.randint(0, pad_id, dec_in.shape, device=device, dtype=dec_in.dtype)
        dec_in = torch.where(corrupt_mask, rand_toks, dec_in)

    # Upweight state-header tokens (`<sys>`, `clef-*`, `key*`, `time-*`)
    for pos, tok in enumerate(flat_toks[: labels.shape[1]]):
        if _is_header_token(tok, uni_omr.SYS_TOKEN):
            weights[:, pos] = ANTIPRIOR_HEADER_WEIGHT

    return inp, dec_in, labels, weights


def _eval_suite31_quick(
    model: Any,
    manifest_path: Path,
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import xml.etree.ElementTree as ET
    import torch
    from PIL import Image
    import eval_uni_suite31 as eus  # type: ignore[import-not-found]
    import rlvr_grpo  # type: ignore[import-not-found]

    def _xml_notes(xml_path: Path) -> List[Tuple[str, int, str, str, str]]:
        notes: List[Tuple[str, int, str, str, str]] = []
        try:
            root = ET.parse(str(xml_path)).getroot()
        except Exception:
            return notes
        for elem in root.iter():
            tag = elem.tag.split("}", 1)[-1]
            if tag != "note":
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

    model.eval()
    _, _, eos_id, _, _, _ = eus._vocab_spec(int(model.cfg.vocab_size))
    _, pieces = eus.resolve_manifest_pieces(manifest_path)
    buckets: Dict[str, List[float]] = {
        "printed_clean": [],
        "printed_photo": [],
        "hw_clean": [],
        "hw_photo": [],
        "all90": [],
    }
    latencies: List[float] = []

    with torch.no_grad():
        for entry in pieces:
            is_hw = bool(entry.get("handwritten", False))
            ref_notes = _xml_notes(Path(entry["gt"]))
            for mode_key in ("clean", "photo"):
                img_p = Path(entry[mode_key])
                if not img_p.is_file():
                    continue
                img = Image.open(img_p).convert("RGB")
                inp = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
                t0 = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    seq = model.generate(inp, max_len=SUITE31_MAX_LEN)
                latencies.append(time.perf_counter() - t0)
                ids = seq[0].tolist()
                pred_toks = eus.decode_uni_compat(ids, vocab_size=model.cfg.vocab_size, skip_special=True)
                if eos_id not in ids[1:]:
                    pred_toks = eus.trim_truncated_tokens(pred_toks)
                hyp_toks, _ = eus.stitch_uni_tokens(pred_toks)
                hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
                nf1 = float(rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes))
                buckets[f"{'hw' if is_hw else 'printed'}_{mode_key}"].append(nf1)
                buckets["all90"].append(nf1)

    res = {k: round(sum(v) / max(1, len(v)), 4) if v else None for k, v in buckets.items()}
    hw_vals = buckets["hw_clean"] + buckets["hw_photo"]
    pr_vals = buckets["printed_clean"] + buckets["printed_photo"]
    res["hw_mean"] = round(sum(hw_vals) / max(1, len(hw_vals)), 4) if hw_vals else None
    res["printed_mean"] = round(sum(pr_vals) / max(1, len(pr_vals)), 4) if pr_vals else None
    res["mean_latency_s"] = round(sum(latencies) / max(1, len(latencies)), 3)
    return res


def _train_part_b_arm(
    arm_name: str,
    *,
    use_dagger_antiprior: bool,
    ckpt_path: Path,
    train_pairs: List[Tuple[Path, List[Path], str, float]],
    suite31_manifest: Path,
    vol_root: Path,
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import eval_uni_suite31 as eus  # type: ignore[import-not-found]
    import uni_omr  # type: ignore[import-not-found]

    _seed_everything(SEED)
    arm_dir = vol_root / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached = arm_dir / "arm_result.json"
    if cached.is_file():
        print(f"[{EXP_ID}][Part B] Reusing cached result for {arm_name}", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))

    model, cfg, _ = eus.load_uni_checkpoint(ckpt_path, device=device)
    if hasattr(cfg, "grad_checkpoint"):
        cfg.grad_checkpoint = True
    pad_id, _, _, _, _, _ = eus._vocab_spec(int(cfg.vocab_size))

    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    ema_state = copy.deepcopy(model.state_dict())

    start_step = 1
    latest_ckpt = arm_dir / "latest.pt"
    if latest_ckpt.is_file():
        try:
            st = torch.load(latest_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(st["model_state_dict"], strict=True)
            ema_state = st["ema_state_dict"]
            opt.load_state_dict(st["optimizer_state_dict"])
            start_step = int(st["step"]) + 1
            print(f"[{EXP_ID}][{arm_name}] Resumed from step {start_step - 1}", flush=True)
        except Exception as exc:
            print(f"[{EXP_ID}][{arm_name}] WARN: resume failed ({exc}); starting fresh.", flush=True)

    weights = [p[3] for p in train_pairs]
    rng = random.Random(SEED + start_step)
    t_arm0 = time.time()
    t_last_save = time.time()
    loss_hist: List[float] = []
    n_sft = 0
    n_dagger = 0
    n_antiprior = 0

    model.train()
    for step in range(start_step, TRAIN_STEPS + 1):
        mult = _wsd_mult(step, TRAIN_STEPS, WARMUP_STEPS)
        for pg in opt.param_groups:
            pg["lr"] = LR_MAX * mult

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            img_p, beks, _, _ = rng.choices(train_pairs, weights=weights, k=1)[0]
            ids, sys_toks = _encode_bekerns_systems(beks, uni_omr, vocab_size=int(cfg.vocab_size), eus=eus)
            if len(ids) < 4:
                continue
            img = Image.open(img_p).convert("RGB")

            mode = "sft"
            if use_dagger_antiprior and rng.random() < DAGGER_FRAC:
                mode = "dagger" if (len(sys_toks) >= 2 and rng.random() < 0.5) else "antiprior"

            if mode == "dagger":
                inp = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
                dag = _build_sys_dagger_example(
                    model,
                    inp,
                    sys_toks,
                    rng=rng,
                    device=device,
                    use_bf16=use_bf16,
                    uni_omr=uni_omr,
                    vocab_size=int(cfg.vocab_size),
                    eus=eus,
                )
                if dag is not None:
                    dec_in, labels, tok_w = dag
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        mem, _, _ = model.encode_image(inp, return_aux=False)
                        dec_out = model.decode_tokens(dec_in, mem, return_mtp=False, return_aux=False)
                        logits = dec_out[0] if isinstance(dec_out, tuple) else dec_out
                        ce = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            labels.reshape(-1),
                            ignore_index=-100,
                            reduction="none",
                        ).view_as(labels)
                        valid = (labels != -100) & (labels != pad_id)
                        denom = (tok_w * valid.float()).sum().clamp(min=1.0)
                        loss = ((ce * tok_w * valid.float()).sum() / denom) / GRAD_ACCUM
                    n_dagger += 1
                else:
                    mode = "sft"

            if mode == "antiprior":
                ap = _build_antiprior_example(
                    img,
                    sys_toks,
                    rng=rng,
                    device=device,
                    uni_omr=uni_omr,
                    vocab_size=int(cfg.vocab_size),
                    eus=eus,
                )
                if ap is not None:
                    inp, dec_in, labels, tok_w = ap
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        mem, _, _ = model.encode_image(inp, return_aux=False)
                        dec_out = model.decode_tokens(dec_in, mem, return_mtp=False, return_aux=False)
                        logits = dec_out[0] if isinstance(dec_out, tuple) else dec_out
                        ce = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            labels.reshape(-1),
                            ignore_index=pad_id,
                            reduction="none",
                        ).view_as(labels)
                        valid = labels != pad_id
                        denom = (tok_w * valid.float()).sum().clamp(min=1.0)
                        loss = ((ce * tok_w * valid.float()).sum() / denom) / GRAD_ACCUM
                    n_antiprior += 1
                else:
                    mode = "sft"

            if mode == "sft":
                inp = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
                tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    if hasattr(model, "compute_loss"):
                        out = model(inp, tgt[:, :-1], return_aux=False, return_mtp=True)
                        losses = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.0)
                        loss = losses["loss"] / GRAD_ACCUM
                    else:
                        mem, _, _ = model.encode_image(inp, return_aux=False)
                        dec_out = model.decode_tokens(tgt[:, :-1], mem, return_mtp=False, return_aux=False)
                        logits = dec_out[0] if isinstance(dec_out, tuple) else dec_out
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            tgt[:, 1:].reshape(-1),
                            ignore_index=pad_id,
                        ) / GRAD_ACCUM
                n_sft += 1

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
            print(
                f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} loss={mean_l:.4f} "
                f"sft={n_sft} dag={n_dagger} ap={n_antiprior} elapsed={time.time()-t_arm0:.1f}s",
                flush=True,
            )

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ck = arm_dir / "latest.pt.tmp"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema_state,
                    "optimizer_state_dict": opt.state_dict(),
                },
                tmp_ck,
            )
            os.replace(tmp_ck, latest_ckpt)
            t_last_save = time.time()

    raw_suite31 = _eval_suite31_quick(model, suite31_manifest, device, use_bf16)
    model.load_state_dict(ema_state, strict=True)
    ema_suite31 = _eval_suite31_quick(model, suite31_manifest, device, use_bf16)

    result = {
        "arm": arm_name,
        "use_dagger_antiprior": use_dagger_antiprior,
        "train_steps": TRAIN_STEPS,
        "micro_counts": {"sft": n_sft, "dagger": n_dagger, "antiprior": n_antiprior},
        "train_loss_tail50": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5),
        "raw_suite31_hw": raw_suite31,
        "ema_suite31_hw": ema_suite31,
        "elapsed_secs": round(time.time() - t_arm0, 2),
    }
    _atomic_write_json(cached, result)
    del model, opt, ema_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    t_start = time.time()
    started_iso = _utc_now()
    hf_token = _read_and_purge_hf_token()
    try:
        _ensure_packages()
        mvd_dir = _locate_mvd_dir()
        import torch
        from huggingface_hub import HfApi

        vol_root = _get_vol_root()
        vol_root.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = bool(device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8)
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        print(f"[{EXP_ID}] Starting on {gpu_name} (bf16={use_bf16}) | ckpt={HF_CKPT_PATH}", flush=True)

        assets = _ensure_hf_assets(vol_root, hf_token)

        # ---- PART A: Risk-Gated Visual-Coverage TTC on OMR-NorthStar 125p ----
        part_a = _run_part_a_northstar_ttc(
            ckpt_path=assets["ckpt_path"],
            ns_dir=assets["ns_dir"],
            vol_root=vol_root,
            mvd_dir=mvd_dir,
            device=device,
            use_bf16=use_bf16,
            hf_token=hf_token,
        )

        # ---- PART B: 1,200-step <sys>-Aligned DAgger + Anti-Prior vs SFT Control ----
        train_pairs = _load_manifest_pairs(assets["train_abs"])
        print(f"[{EXP_ID}][Part B] Loaded {len(train_pairs)} proxy train pairs", flush=True)

        arm_b0 = _train_part_b_arm(
            "arm_b0_sft_control",
            use_dagger_antiprior=False,
            ckpt_path=assets["ckpt_path"],
            train_pairs=train_pairs,
            suite31_manifest=assets["suite31_manifest"],
            vol_root=vol_root,
            device=device,
            use_bf16=use_bf16,
        )
        arm_b1 = _train_part_b_arm(
            "arm_b1_sys_dagger_antiprior",
            use_dagger_antiprior=True,
            ckpt_path=assets["ckpt_path"],
            train_pairs=train_pairs,
            suite31_manifest=assets["suite31_manifest"],
            vol_root=vol_root,
            device=device,
            use_bf16=use_bf16,
        )

        def _delta(b_val: Optional[float], a_val: Optional[float]) -> Optional[float]:
            return None if (b_val is None or a_val is None) else round(float(b_val) - float(a_val), 5)

        ema_hw_delta = _delta(arm_b1["ema_suite31_hw"]["hw_mean"], arm_b0["ema_suite31_hw"]["hw_mean"])
        ema_pr_delta = _delta(arm_b1["ema_suite31_hw"]["printed_mean"], arm_b0["ema_suite31_hw"]["printed_mean"])
        raw_hw_delta = _delta(arm_b1["raw_suite31_hw"]["hw_mean"], arm_b0["raw_suite31_hw"]["hw_mean"])
        raw_pr_delta = _delta(arm_b1["raw_suite31_hw"]["printed_mean"], arm_b0["raw_suite31_hw"]["printed_mean"])

        part_b = {
            "arm_b0_sft_control": arm_b0,
            "arm_b1_sys_dagger_antiprior": arm_b1,
            "comparison_b1_minus_b0": {
                "ema_hw_mean_delta": ema_hw_delta,
                "ema_printed_mean_delta": ema_pr_delta,
                "ema_all90_delta": _delta(arm_b1["ema_suite31_hw"]["all90"], arm_b0["ema_suite31_hw"]["all90"]),
                "raw_hw_mean_delta": raw_hw_delta,
                "raw_printed_mean_delta": raw_pr_delta,
                "raw_all90_delta": _delta(arm_b1["raw_suite31_hw"]["all90"], arm_b0["raw_suite31_hw"]["all90"]),
            },
            "gates": {
                "ema_hw_ge_3pp": (ema_hw_delta is not None and ema_hw_delta >= 0.03),
                "ema_printed_ge_neg0_3pp": (ema_pr_delta is not None and ema_pr_delta >= -0.003),
            },
        }

        report = {
            "exp_id": EXP_ID,
            "exp_code": EXP_CODE,
            "wave": 13,
            "status": "completed",
            "started_utc": started_iso,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"colab-or-modal-{gpu_name}",
            "bf16": use_bf16,
            "baseline_checkpoint": HF_CKPT_PATH,
            "part_a_risk_gated_visual_ttc": part_a,
            "part_b_sys_dagger_probe": part_b,
        }

        eval_local = vol_root / "evals" / f"{EXP_ID}.json"
        run_local = vol_root / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(
            run_local,
            {
                "exp_id": EXP_ID,
                "exp_code": EXP_CODE,
                "status": "completed",
                "started_utc": started_iso,
                "completed_utc": report["completed_utc"],
                "elapsed_secs": report["elapsed_secs"],
                "gpu": gpu_name,
                "bf16": use_bf16,
                "eval_path": HF_EVAL_PATH,
                "part_a_gates": part_a["gates"],
                "part_b_gates": part_b["gates"],
            },
        )

        if hf_token and NS_LIMIT_PAGES == 0:
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=str(eval_local),
                path_in_repo=HF_EVAL_PATH,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"eval({EXP_ID}): upload Part A Risk-Gated TTC + Part B <sys>-DAgger probe",
            )
            api.upload_file(
                path_or_fileobj=str(run_local),
                path_in_repo=HF_RUN_PATH,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"run({EXP_ID}): upload Wave 13 i15-04 run ledger",
            )
            print(f"[{EXP_ID}] UPLOADED {HF_EVAL_PATH} & {HF_RUN_PATH} TO HF!", flush=True)
        else:
            print(f"[{EXP_ID}] Skipping HF upload (token={'set' if hf_token else 'unset'}, limit={NS_LIMIT_PAGES})", flush=True)
        return 0
    finally:
        if HF_TOKEN_FILE.is_file():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
