#!/usr/bin/env python3
"""Colab / Modal L4 / Kaggle T4 runner for Wave 11 Exp i14-04b:
Zero-Retrain Adaptive Band-Tiling Probe for Dense Multi-Staff Pages (Fixed Harness).

Experiment ID: 2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed
Predecessor: 2026-11-26T22-18-i14-04-adaptive-band-tiling-eval (invalidated by
missing=349 ckpt-loader bug + [-1,1] input norm bug + silent FP32 on T4).

Fixes vs i14-04:
  1. Uses canonical `eval_uni_suite31.load_uni_checkpoint` with hard
     `assert len(meta["missing_keys"]) == 0`.
  2. Uses canonical `uni_omr._resize_fullpage_pil` + `uni_omr._pil_to_chw_tensor`
     ([0, 1] input range, matching training & eval_uni_suite31.py).
  3. Uses native `torch.bfloat16` on cc >= 8 (L4/A100) and `torch.float16` on
     cc < 8 (T4) — never silent FP32.
  4. Calibrated staff-space autocorrelation (`min_lag >= max(6, round(H*0.0035))`
     + true local-maximum peak picking past the lag-0 decay lobe).
  5. 2-page fail-fast sanity gate on Arm A (aborts in <30s if dec_ok == 0 or
     mean Note F1 < 0.05 on the first 2 printed clean pages).
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter10_5_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter10_v2_best/uni_omr_v2_ema_best.pt",
)
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_PARTIAL_STATE_PATH = f"checkpoints/{EXP_ID}/partial_state.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_SYNC_EVERY_MINUTES = 10.0

VOL_ROOT = Path("/vol") if Path("/vol").is_dir() else Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

SINGLEPASS_BASE_DIM = 1408
SINGLEPASS_HIRES_DIM = 1792
DENSE_STAFF_SPACE_THR_PX = 7.0
DENSE_SYSTEM_COUNT_THR = 6
TARGET_BAND_STAFF_SPACE_PX = 10.5
BAND_MAX_WIDTH = 1536
BAND_MAX_HEIGHT = 1280
BAND_SYSTEMS_PER_TILE = 2
BAND_OVERLAP_FRAC = 0.04
SUITE31_MAX_LEN = 3584
BAND_MAX_LEN = 1536
SAVE_EVERY_MINUTES = 5.0

_HEADER_PREFIXES = ("clef", "key", "time", "*clef", "*k[", "*M")


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
            return cand.parent.parent if cand.name == "mvd" and cand.parent.name == "tmp-train" else cand.parent, cand

    for bundle in (Path("/content/ov_mvd_bundle.tar.gz"), Path("/content/bundle.tar.gz")):
        if bundle.is_file():
            extract_root = Path("/content/workspace")
            extract_root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=extract_root)
            for mvd in (
                extract_root / "sheet-music-converter" / "tmp-train" / "mvd",
                extract_root / "mvd",
            ):
                if (mvd / "uni_omr_v2.py").is_file():
                    if str(mvd) not in sys.path:
                        sys.path.insert(0, str(mvd))
                    return extract_root, mvd

    raise FileNotFoundError("Could not locate tmp-train/mvd/uni_omr_v2.py")


def _setup_environment() -> Path:
    repo_root, _ = _locate_repo_and_mvd()
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
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed])
    return repo_root


def _ensure_assets(hf_token: str) -> Tuple[Path, str, Path]:
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
            print(f"[{EXP_ID}] Downloading checkpoint {cand}...", flush=True)
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

    if ckpt_local is None or ckpt_source is None:
        raise RuntimeError(f"No frozen UniOMR v2 checkpoint found among {HF_CKPT_CANDIDATES}")

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

    return ckpt_local, ckpt_source, manifest_path


def _load_frozen_model(ckpt_path: Path, device: "torch.device") -> Tuple[Any, Dict[str, Any]]:
    """Load frozen checkpoint via canonical eval_uni_suite31.load_uni_checkpoint with strict check."""
    import eval_uni_suite31 as eus

    model, cfg, meta = eus.load_uni_checkpoint(ckpt_path, device)
    missing = meta.get("missing_keys") or []
    unexpected = meta.get("unexpected_keys") or []
    print(
        f"[{EXP_ID}] Loaded frozen ckpt ({ckpt_path.name}): arch={meta.get('arch')} "
        f"step={meta.get('step')} d_model={getattr(cfg, 'd_model', None)} "
        f"max_mem={getattr(cfg, 'max_mem_tokens', None)} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if len(missing) > 0:
        raise RuntimeError(
            f"Refusing to evaluate partially loaded checkpoint: {len(missing)} missing keys "
            f"(first 5: {missing[:5]}), {len(unexpected)} unexpected keys."
        )
    model.eval()
    return model, meta


def _pil_to_canonical_tensor(img_pil: Any, max_dim: int, max_h: Optional[int] = None) -> "torch.Tensor":
    """Canonical [0, 1] CHW tensor matching uni_omr._resize_fullpage_pil + _pil_to_chw_tensor."""
    import uni_omr

    work = img_pil.convert("RGB")
    if max_h is not None:
        w, h = work.size
        scale = min(1.0, float(max_dim) / float(max(w, h, 1)), float(max_h) / float(max(h, 1)))
        eff_max_dim = max(32, int(round(max(w, h) * scale)))
    else:
        eff_max_dim = int(max_dim)
    pil_im = uni_omr._resize_fullpage_pil(work, eff_max_dim)
    return uni_omr._pil_to_chw_tensor(pil_im)


def _analyze_page_density_and_systems(img_pil: Any) -> Dict[str, Any]:
    """Estimate native staff-space px (local-peak AC past lag-0 lobe) and system bands."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    gray = np.asarray(img_pil.convert("L"), dtype=np.float32) / 255.0
    H, W = gray.shape
    scale_1408 = min(1.0, float(SINGLEPASS_BASE_DIM) / float(max(H, W)))

    # High-pass ink projection (vertically-smoothed 92nd-pct bg so full-width staff lines are preserved)
    x0, x1 = int(0.05 * W), max(int(0.95 * W), int(0.05 * W) + 1)
    crop = gray[:, x0:x1]
    row_bg_raw = np.percentile(crop, 92, axis=1)
    row_bg = gaussian_filter1d(row_bg_raw, sigma=max(12.0, H * 0.025))[:, None]
    ink = np.clip(row_bg - crop, 0.0, 1.0)
    row_prof = ink.mean(axis=1)

    # 1. Estimate staff-space px via first true local maximum of autocorrelation past lag-0 lobe
    row_hp = row_prof - gaussian_filter1d(row_prof, sigma=max(2.0, H * 0.012))
    row_hp = row_hp - row_hp.mean()
    ac = np.correlate(row_hp, row_hp, mode="full")[H - 1 :]
    min_lag = max(6, int(round(H * 0.0035)))
    max_lag = min(H // 12, max(min_lag + 6, int(round(H * 0.025))))
    native_staff_space_px = float(H) / 140.0
    ac_peak = 0.0
    if max_lag > min_lag + 2 and ac[0] > 1e-8:
        ac_norm = ac / ac[0]
        # Advance past any monotonic descent from lag 0
        start_lag = min_lag
        while start_lag + 1 < max_lag and ac_norm[start_lag] >= ac_norm[start_lag + 1]:
            start_lag += 1
        # Find local maxima in [start_lag, max_lag - 1]
        best_lag = -1
        best_val = -1.0
        for lag in range(max(min_lag, start_lag), max_lag - 1):
            val = float(ac_norm[lag])
            if val > float(ac_norm[lag - 1]) and val >= float(ac_norm[lag + 1]) and val > best_val:
                best_val = val
                best_lag = lag
        if best_lag > 0 and best_val > 0.08:
            native_staff_space_px = float(best_lag)
            ac_peak = best_val
    staff_space_px_1408 = native_staff_space_px * scale_1408

    # 2. Detect system bands via smoothed vertical ink projection valleys
    smooth = gaussian_filter1d(row_prof, sigma=max(3.0, native_staff_space_px * 1.2))
    thr = max(0.012, float(np.percentile(smooth, 25)) + 0.14 * float(smooth.max() - smooth.min()))
    active = smooth > thr

    raw_bands: List[Tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for y in range(H):
        if active[y] and not in_seg:
            seg_start = y
            in_seg = True
        elif not active[y] and in_seg:
            if (y - seg_start) >= max(8, int(native_staff_space_px * 2.5)):
                raw_bands.append((seg_start, y))
            in_seg = False
    if in_seg and (H - seg_start) >= max(8, int(native_staff_space_px * 2.5)):
        raw_bands.append((seg_start, H))

    # Merge closely spaced staves belonging to the same grand-staff/piano system
    merge_gap = max(6, int(round(native_staff_space_px * 2.4)))
    systems: List[Tuple[int, int]] = []
    for y0, y1 in raw_bands:
        if systems and (y0 - systems[-1][1]) <= merge_gap:
            systems[-1] = (systems[-1][0], y1)
        else:
            systems.append((y0, y1))

    if not systems:
        systems = [(int(0.02 * H), int(0.98 * H))]

    # Compute valley split points between consecutive systems
    valleys: List[int] = [0]
    for i in range(len(systems) - 1):
        a_end = systems[i][1]
        b_start = systems[i + 1][0]
        if b_start > a_end + 1:
            v = a_end + int(np.argmin(smooth[a_end:b_start]))
        else:
            v = (a_end + b_start) // 2
        valleys.append(v)
    valleys.append(H)

    is_dense = (staff_space_px_1408 < DENSE_STAFF_SPACE_THR_PX) or (len(systems) >= DENSE_SYSTEM_COUNT_THR)
    return {
        "native_hw": (H, W),
        "native_staff_space_px": round(native_staff_space_px, 2),
        "staff_space_px_1408": round(staff_space_px_1408, 2),
        "ac_peak": round(ac_peak, 3),
        "n_systems_est": len(systems),
        "systems_px": systems,
        "valleys_px": valleys,
        "is_dense": bool(is_dense),
    }


def _build_adaptive_tiles(img_pil: Any, analysis: Dict[str, Any]) -> List[Tuple[Any, int]]:
    """Build overlapping 2-3 system crops and target max_dim guaranteeing >= 10 px/staff-space."""
    H, W = analysis["native_hw"]
    n_sys = analysis["n_systems_est"]
    valleys = analysis["valleys_px"]
    native_ss = max(2.0, float(analysis["native_staff_space_px"]))
    pad_px = int(round(H * BAND_OVERLAP_FRAC))

    if n_sys <= 3:
        mid = valleys[len(valleys) // 2] if len(valleys) > 2 else H // 2
        tile_ranges = [(0, min(H, mid + pad_px)), (max(0, mid - pad_px), H)]
    else:
        step = BAND_SYSTEMS_PER_TILE if n_sys <= 6 else 3
        tile_ranges = []
        idx = 0
        while idx < n_sys:
            end_idx = min(n_sys, idx + step)
            if n_sys - end_idx == 1:
                end_idx = n_sys
            y0 = max(0, valleys[idx] - (pad_px if idx > 0 else 0))
            y1 = min(H, valleys[end_idx] + (pad_px if end_idx < n_sys else 0))
            tile_ranges.append((y0, y1))
            idx = end_idx

    tiles: List[Tuple[Any, int]] = []
    req_scale = min(1.0, TARGET_BAND_STAFF_SPACE_PX / native_ss)
    for y0, y1 in tile_ranges:
        crop = img_pil.crop((0, y0, W, y1))
        cw, ch = crop.size
        target_dim = min(BAND_MAX_WIDTH, max(1024, int(round(max(cw, ch) * req_scale))))
        tiles.append((crop, target_dim))
    return tiles


def _extract_header_tokens(tokens: List[str]) -> List[str]:
    """Extract leading clef/key/time-signature tokens from the first system for carry-forward."""
    header: List[str] = []
    for tok in tokens[:24]:
        low = tok.lower()
        if any(low.startswith(p) for p in _HEADER_PREFIXES):
            header.append(tok)
        elif header and (tok in ("=", "|", "<sys>") or re.match(r"^[0-9]+[a-gA-G]", tok)):
            break
    return header


def _stitch_band_tokens_with_header_carry(
    band_token_lists: List[List[str]], sys_token: str
) -> List[str]:
    """Stitch decoded multi-system bands with boundary deduplication & header carry-forward."""
    if not band_token_lists:
        return []
    stitched_systems: List[List[str]] = []
    carry_header: List[str] = []

    for b_idx, toks in enumerate(band_token_lists):
        if not toks:
            continue
        cur_systems: List[List[str]] = [[]]
        for t in toks:
            if t == sys_token:
                if cur_systems[-1]:
                    cur_systems.append([])
            else:
                cur_systems[-1].append(t)
        cur_systems = [s for s in cur_systems if s]
        if not cur_systems:
            continue

        if b_idx == 0:
            carry_header = _extract_header_tokens(cur_systems[0])
            stitched_systems.extend(cur_systems)
            continue

        if stitched_systems and cur_systems:
            prev_s = stitched_systems[-1]
            next_s = cur_systems[0]
            sim = difflib.SequenceMatcher(None, prev_s[:80], next_s[:80]).ratio()
            if sim >= 0.72:
                cur_systems = cur_systems[1:]

        for s_idx, sys_toks in enumerate(cur_systems):
            if s_idx == 0 and carry_header and not _extract_header_tokens(sys_toks):
                sys_toks = list(carry_header) + sys_toks
            stitched_systems.append(sys_toks)

    out: List[str] = []
    for idx, sys_toks in enumerate(stitched_systems):
        if idx > 0:
            out.append(sys_token)
        out.extend(sys_toks)
    return out


def _extract_musicxml_notes(xml_path: Path) -> List[Tuple[str, int, str, str, str]]:
    import xml.etree.ElementTree as ET

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


def _restore_partial_state_from_hf(hf_token: str) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """Best-effort download of partial arm/page state from HF (`checkpoints/<EXP_ID>/partial_state.json` or `evals/<EXP_ID>.json`)."""
    from huggingface_hub import hf_hub_download

    completed_arms: Dict[str, Any] = {}
    in_progress_arms: Dict[str, List[Dict[str, Any]]] = {}
    for repo_path in (HF_PARTIAL_STATE_PATH, HF_EVAL_PATH):
        try:
            dl_p = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=repo_path,
                repo_type="dataset",
                token=hf_token,
                local_dir="/tmp/hf_dl",
            )
            payload = json.loads(Path(dl_p).read_text(encoding="utf-8"))
            if isinstance(payload.get("completed_arms"), dict):
                completed_arms.update(payload["completed_arms"])
            if isinstance(payload.get("in_progress"), dict):
                for k, v in payload["in_progress"].items():
                    if isinstance(v, list) and k not in completed_arms:
                        in_progress_arms[k] = v
            for arm_k in ("arm_a_singlepass_1408", "arm_b_singlepass_1792", "arm_c_adaptive_band_tiling"):
                if isinstance(payload.get(arm_k), dict) and arm_k not in completed_arms:
                    completed_arms[arm_k] = payload[arm_k]
        except Exception:
            pass
    if completed_arms or in_progress_arms:
        print(
            f"[{EXP_ID}][resume] Restored from HF: completed={sorted(completed_arms.keys())} "
            f"in_progress={ {k: len(v) for k, v in in_progress_arms.items()} }",
            flush=True,
        )
    return completed_arms, in_progress_arms


def _sync_partial_state_to_hf(
    *,
    api: Any,
    state_local: Path,
    completed_arms: Dict[str, Any],
    in_progress_arms: Dict[str, List[Dict[str, Any]]],
    reason: str,
) -> None:
    """Best-effort upload of partial arm/page progress to HF `checkpoints/<EXP_ID>/partial_state.json`."""
    try:
        payload = {
            "exp_id": EXP_ID,
            "updated_utc": _utc_now(),
            "completed_arms": completed_arms,
            "in_progress": in_progress_arms,
        }
        _atomic_write_json(state_local, payload)
        api.upload_file(
            path_or_fileobj=str(state_local),
            path_in_repo=HF_PARTIAL_STATE_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"ckpt({EXP_ID}): {reason}",
        )
        print(f"[{EXP_ID}][hf] Synced {HF_PARTIAL_STATE_PATH} ({reason})", flush=True)
    except Exception as exc:
        print(f"[{EXP_ID}][hf] WARN: partial sync failed ({type(exc).__name__}: {str(exc)[:120]})", flush=True)


def _eval_arm(
    arm_name: str,
    *,
    mode: str,
    model: Any,
    manifest_path: Path,
    device: "torch.device",
    amp_dtype: Optional["torch.dtype"],
    sanity_check: bool = False,
    resumed_per_page: Optional[List[Dict[str, Any]]] = None,
    on_hf_sync: Optional[Any] = None,
) -> Dict[str, Any]:
    import torch
    from PIL import Image
    import eval_uni_suite31 as eus
    import rlvr_grpo
    import uni_omr

    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached = arm_dir / "arm_result.json"
    if cached.is_file():
        print(f"[{EXP_ID}] Reusing cached result for {arm_name}", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))

    _, _, eos_id, _, _, id_to_tok = eus._vocab_spec(int(model.cfg.vocab_size))
    _, pieces = eus.resolve_manifest_pieces(manifest_path)

    buckets: Dict[str, List[Dict[str, Any]]] = {
        "printed_clean": [],
        "printed_photo": [],
        "hw_clean": [],
        "hw_photo": [],
        "dense_subset": [],
        "sparse_subset": [],
        "all90": [],
    }
    per_page: List[Dict[str, Any]] = []
    done_keys = set()
    if resumed_per_page:
        for row in resumed_per_page:
            key = (row.get("piece_id"), row.get("mode"))
            if key in done_keys:
                continue
            done_keys.add(key)
            per_page.append(row)
            is_hw_r = bool(row.get("is_hw", False))
            is_dense_r = bool(row.get("is_dense", False))
            mode_r = str(row.get("mode", "clean"))
            buckets[f"{'hw' if is_hw_r else 'printed'}_{mode_r}"].append(row)
            buckets["dense_subset" if is_dense_r else "sparse_subset"].append(row)
            buckets["all90"].append(row)
        print(f"[{EXP_ID}][{arm_name}] Resumed {len(per_page)} pages from HF partial state", flush=True)

    t_arm0 = time.time()
    t_last_save = time.time()
    t_last_hf_sync = time.time()
    amp_enabled = device.type == "cuda" and amp_dtype is not None
    sanity_done = len(buckets["printed_clean"]) >= 2

    with torch.no_grad():
        for p_idx, entry in enumerate(pieces):
            piece_id = str(entry.get("slug") or entry.get("id") or f"piece_{p_idx}")
            is_hw = bool(entry.get("handwritten", False))
            ref_notes = _extract_musicxml_notes(Path(entry["gt"]))

            for mode_key in ("clean", "photo"):
                if (piece_id, mode_key) in done_keys:
                    continue
                img_p = Path(entry[mode_key])
                if not img_p.is_file():
                    continue
                img = Image.open(img_p).convert("RGB")
                analysis = _analyze_page_density_and_systems(img)
                is_dense = bool(analysis["is_dense"])
                t0 = time.perf_counter()

                n_tiles = 1
                ink_recalls: List[float] = []
                hit_eos = True

                if mode == "singlepass_1408" or (mode == "adaptive_band_tiling" and not is_dense):
                    inp_t = _pil_to_canonical_tensor(img, SINGLEPASS_BASE_DIM).unsqueeze(0).to(device)
                    with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
                        seq, rinfo = model.generate(inp_t, max_len=SUITE31_MAX_LEN, return_info=True)
                    ids = seq[0].tolist()
                    hit_eos = eos_id in ids[1:]
                    pred_toks = eus.decode_uni_compat(ids, vocab_size=model.cfg.vocab_size, skip_special=True)
                    if not hit_eos:
                        pred_toks = eus.trim_truncated_tokens(pred_toks)
                    ink_recalls.append(float(rinfo.get("ink_recall", 1.0)))
                elif mode == "singlepass_1792":
                    inp_t = _pil_to_canonical_tensor(img, SINGLEPASS_HIRES_DIM).unsqueeze(0).to(device)
                    with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
                        seq, rinfo = model.generate(inp_t, max_len=SUITE31_MAX_LEN, return_info=True)
                    ids = seq[0].tolist()
                    hit_eos = eos_id in ids[1:]
                    pred_toks = eus.decode_uni_compat(ids, vocab_size=model.cfg.vocab_size, skip_special=True)
                    if not hit_eos:
                        pred_toks = eus.trim_truncated_tokens(pred_toks)
                    ink_recalls.append(float(rinfo.get("ink_recall", 1.0)))
                else:
                    tiles = _build_adaptive_tiles(img, analysis)
                    n_tiles = len(tiles)
                    band_toks: List[List[str]] = []
                    for crop, target_dim in tiles:
                        inp_t = _pil_to_canonical_tensor(crop, target_dim, max_h=BAND_MAX_HEIGHT).unsqueeze(0).to(device)
                        with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
                            seq, rinfo = model.generate(inp_t, max_len=BAND_MAX_LEN, return_info=True)
                        ids = seq[0].tolist()
                        toks_b = eus.decode_uni_compat(ids, vocab_size=model.cfg.vocab_size, skip_special=True)
                        if eos_id not in ids[1:]:
                            toks_b = eus.trim_truncated_tokens(toks_b)
                            hit_eos = False
                        band_toks.append(toks_b)
                        ink_recalls.append(float(rinfo.get("ink_recall", 1.0)))
                    pred_toks = _stitch_band_tokens_with_header_carry(band_toks, uni_omr.SYS_TOKEN)

                latency_s = time.perf_counter() - t0
                hyp_toks, _ = eus.stitch_uni_tokens(pred_toks)
                hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
                note_f1 = float(rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes))
                struct = rlvr_grpo.compute_unlabeled_structural_reward(
                    hyp_toks, vocab=id_to_tok, terminated_eos=hit_eos
                )
                dec_ok = 1.0 if (len(hyp_notes) > 0 and float(struct.get("r_parse", 0.0)) > 0.0) else 0.0
                mean_ir = sum(ink_recalls) / max(1, len(ink_recalls))

                row = {
                    "piece_id": piece_id,
                    "mode": mode_key,
                    "is_hw": is_hw,
                    "is_dense": is_dense,
                    "staff_space_px_1408": analysis["staff_space_px_1408"],
                    "n_systems_est": analysis["n_systems_est"],
                    "n_tiles": n_tiles,
                    "note_f1": round(note_f1, 4),
                    "dec_ok": dec_ok,
                    "ink_recall": round(mean_ir, 4),
                    "latency_s": round(latency_s, 3),
                }
                per_page.append(row)
                buckets[f"{'hw' if is_hw else 'printed'}_{mode_key}"].append(row)
                buckets["dense_subset" if is_dense else "sparse_subset"].append(row)
                buckets["all90"].append(row)

            # Fail-fast sanity gate after first 2 printed clean pages
            if sanity_check and not sanity_done and len(buckets["printed_clean"]) >= 2:
                sanity_done = True
                pc2 = buckets["printed_clean"][:2]
                mean_f1_2 = sum(r["note_f1"] for r in pc2) / len(pc2)
                dec_ok_2 = sum(r["dec_ok"] for r in pc2) / len(pc2)
                print(
                    f"[{EXP_ID}][{arm_name}] 2-page sanity check: "
                    f"mean_note_f1={mean_f1_2:.4f} dec_ok={dec_ok_2:.2f}",
                    flush=True,
                )
                if dec_ok_2 <= 0.0 or mean_f1_2 < 0.05:
                    raise RuntimeError(
                        f"Fail-fast sanity gate tripped on {arm_name}: "
                        f"first 2 printed_clean pages had mean_note_f1={mean_f1_2:.4f}, dec_ok={dec_ok_2:.2f}."
                    )

            if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
                _atomic_write_json(arm_dir / "progress.json", {"done_pieces": p_idx + 1, "pages": len(per_page)})
                t_last_save = time.time()
            if on_hf_sync is not None and (time.time() - t_last_hf_sync) >= HF_SYNC_EVERY_MINUTES * 60.0:
                on_hf_sync(per_page)
                t_last_hf_sync = time.time()

    def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0, "note_f1": None, "dec_ok": None, "ink_recall": None, "latency_s": None}
        return {
            "n": len(rows),
            "note_f1": round(sum(r["note_f1"] for r in rows) / len(rows), 4),
            "dec_ok": round(sum(r["dec_ok"] for r in rows) / len(rows), 4),
            "ink_recall": round(sum(r["ink_recall"] for r in rows) / len(rows), 4),
            "latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 3),
        }

    result = {
        "arm": arm_name,
        "mode": mode,
        "summary": {k: _agg(v) for k, v in buckets.items()},
        "elapsed_secs": round(time.time() - t_arm0, 2),
        "per_page_sample": per_page[:15],
    }
    _atomic_write_json(cached, result)
    print(f"[{EXP_ID}][{arm_name}] done: {json.dumps(result['summary'], indent=2)}", flush=True)
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
        cc_major = torch.cuda.get_device_capability(0)[0] if device.type == "cuda" else 0
        amp_dtype = (
            torch.bfloat16 if (device.type == "cuda" and cc_major >= 8)
            else (torch.float16 if device.type == "cuda" else None)
        )
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

        ckpt_local, ckpt_source, manifest_path = _ensure_assets(hf_token)
        model, ckpt_meta = _load_frozen_model(ckpt_local, device)

        api = HfApi(token=hf_token)
        state_local = VOL_ROOT / "ckpts" / EXP_ID / "partial_state.json"
        completed_arms, in_progress_arms = _restore_partial_state_from_hf(hf_token)

        arm_specs = (
            ("arm_a_singlepass_1408", "singlepass_1408", True),
            ("arm_b_singlepass_1792", "singlepass_1792", False),
            ("arm_c_adaptive_band_tiling", "adaptive_band_tiling", False),
        )
        for arm_name, arm_mode, do_sanity in arm_specs:
            if arm_name in completed_arms:
                print(f"[{EXP_ID}][resume] Skipping {arm_name} (loaded from HF)", flush=True)
                continue

            def _sync_cb(cur_pages: List[Dict[str, Any]], _arm: str = arm_name) -> None:
                in_progress_arms[_arm] = list(cur_pages)
                _sync_partial_state_to_hf(
                    api=api,
                    state_local=state_local,
                    completed_arms=completed_arms,
                    in_progress_arms=in_progress_arms,
                    reason=f"{_arm} progress ({len(cur_pages)} pages)",
                )

            arm_res = _eval_arm(
                arm_name,
                mode=arm_mode,
                model=model,
                manifest_path=manifest_path,
                device=device,
                amp_dtype=amp_dtype,
                sanity_check=do_sanity,
                resumed_per_page=in_progress_arms.get(arm_name),
                on_hf_sync=_sync_cb,
            )
            completed_arms[arm_name] = arm_res
            in_progress_arms.pop(arm_name, None)
            _sync_partial_state_to_hf(
                api=api,
                state_local=state_local,
                completed_arms=completed_arms,
                in_progress_arms=in_progress_arms,
                reason=f"completed {arm_name}",
            )

        arm_a = completed_arms["arm_a_singlepass_1408"]
        arm_b = completed_arms["arm_b_singlepass_1792"]
        arm_c = completed_arms["arm_c_adaptive_band_tiling"]

        def _delta(b_val: Optional[float], a_val: Optional[float]) -> Optional[float]:
            return None if (b_val is None or a_val is None) else round(float(b_val) - float(a_val), 4)

        sa, sb, sc = arm_a["summary"], arm_b["summary"], arm_c["summary"]
        comparison = {
            "dense_note_f1_c_minus_a": _delta(sc["dense_subset"]["note_f1"], sa["dense_subset"]["note_f1"]),
            "dense_note_f1_b_minus_a": _delta(sb["dense_subset"]["note_f1"], sa["dense_subset"]["note_f1"]),
            "sparse_note_f1_c_minus_a": _delta(sc["sparse_subset"]["note_f1"], sa["sparse_subset"]["note_f1"]),
            "all90_note_f1_c_minus_a": _delta(sc["all90"]["note_f1"], sa["all90"]["note_f1"]),
            "printed_clean_c_minus_a": _delta(sc["printed_clean"]["note_f1"], sa["printed_clean"]["note_f1"]),
            "hw_clean_c_minus_a": _delta(sc["hw_clean"]["note_f1"], sa["hw_clean"]["note_f1"]),
            "dense_ink_recall_a": sa["dense_subset"]["ink_recall"],
            "dense_ink_recall_c": sc["dense_subset"]["ink_recall"],
        }

        report = {
            "exp_id": EXP_ID,
            "exp_code": "i14-04b",
            "predecessor": "2026-11-26T22-18-i14-04-adaptive-band-tiling-eval",
            "status": "completed",
            "checkpoint": ckpt_source,
            "ckpt_meta": {
                "arch": ckpt_meta.get("arch"),
                "step": ckpt_meta.get("step"),
                "missing_keys": len(ckpt_meta.get("missing_keys") or []),
                "unexpected_keys": len(ckpt_meta.get("unexpected_keys") or []),
            },
            "started_utc": started_iso,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"cloud-{gpu_name}",
            "amp_dtype": str(amp_dtype),
            "arm_a_singlepass_1408": arm_a,
            "arm_b_singlepass_1792": arm_b,
            "arm_c_adaptive_band_tiling": arm_c,
            "comparison": comparison,
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
            commit_message=f"eval({EXP_ID}): upload fixed zero-retrain adaptive band-tiling probe",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload fixed zero-retrain adaptive band-tiling run ledger",
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
