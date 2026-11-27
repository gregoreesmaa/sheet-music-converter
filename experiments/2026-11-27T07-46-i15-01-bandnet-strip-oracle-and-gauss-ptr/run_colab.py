#!/usr/bin/env python3
"""Colab L4 runner for Wave 13 Exp i15-01:
Zero-Retrain BandNet Strip Oracle & StaffSeg Gaussian Cross-Attention Pointer.

Experiment ID: 2026-11-27T07-46-i15-01-bandnet-strip-oracle-and-gauss-ptr
Evaluates 3 zero-retrain arms on our frozen UniOMRv2 EMA checkpoint over
suite31_hw (all 14 HW pieces + 10 printed safety-slice pieces = 48 pages):
  - arm0_fullpage_baseline: Standard 1408px full-page FSA + MTP-4 decode.
  - arm1_staffseg_gauss_ptr: Keep StaffSegHead active at inference -> extract
    system vertical bands y_1..y_N -> inject per-<sys> vertical Gaussian
    cross-attention bias -((y_mem - y_sys[cur])^2)/(2*sigma_y^2) into
    GQADecoderLayer.cross_attn (advancing cur on <sys>).
  - arm2_band_crop_oracle: Crop each page into individual system strips
    (StaffSegHead bands / valley peaks + +-20% vertical pad) -> decode each
    strip independently -> stitch <sys> (UniOMR-BandNet zero-retrain oracle).
"""

from __future__ import annotations

import contextlib
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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-27T07-46-i15-01-bandnet-strip-oracle-and-gauss-ptr"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter10_5_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter10_v2_best/uni_omr_v2_ema_best.pt",
)
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

VOL_ROOT = Path("/vol") if Path("/vol").is_dir() else Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

FULLPAGE_MAX_DIM = 1408
STRIP_MAX_DIM = 1280
FULLPAGE_MAX_LEN = 3584
STRIP_MAX_LEN = 1024
STRIP_PAD_FRAC = 0.20
PRINTED_SAFETY_COUNT = 10
GAUSS_BIAS_MIN = -6.0
GAUSS_SIGMA_MIN = 0.06
GAUSS_SIGMA_MAX = 0.18
GAUSS_SIGMA_BAND_MULT = 0.75
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
            root = cand.parent.parent if (cand.name == "mvd" and cand.parent.name == "tmp-train") else cand.parent
            return root, cand

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
        raise RuntimeError(f"No frozen UniOMRv2 checkpoint found among {HF_CKPT_CANDIDATES}")

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
    import eval_uni_suite31 as eus

    model, cfg, meta = eus.load_uni_checkpoint(ckpt_path, device)
    missing = meta.get("missing_keys") or []
    unexpected = meta.get("unexpected_keys") or []
    has_staff_seg = getattr(model, "staff_seg_head", None) is not None
    print(
        f"[{EXP_ID}] Loaded frozen ckpt ({ckpt_path.name}): arch={meta.get('arch')} "
        f"step={meta.get('step')} d_model={getattr(cfg, 'd_model', None)} "
        f"staff_seg_head={has_staff_seg} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if len(missing) > 0:
        raise RuntimeError(
            f"Refusing to evaluate partially loaded checkpoint: {len(missing)} missing keys "
            f"(first 5: {missing[:5]}), {len(unexpected)} unexpected keys."
        )
    model.eval()
    return model, meta


def _pil_to_canonical_tensor(img_pil: Any, max_dim: int) -> "torch.Tensor":
    import uni_omr

    work = img_pil.convert("RGB")
    pil_im = uni_omr._resize_fullpage_pil(work, int(max_dim))
    return uni_omr._pil_to_chw_tensor(pil_im)


def _bands_from_1d_profile(
    prof: "np.ndarray",
    *,
    min_band_frac: float = 0.015,
    merge_gap_frac: float = 0.025,
) -> List[Tuple[float, float, float]]:
    """Convert a 1D vertical activity profile (len H) into normalized (y0, y1, y_center) bands."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    H = int(len(prof))
    if H < 8:
        return [(0.02, 0.98, 0.50)]

    smooth = gaussian_filter1d(prof.astype(np.float32), sigma=max(1.5, H * 0.008))
    p_lo = float(np.percentile(smooth, 20))
    p_hi = float(np.max(smooth))
    if (p_hi - p_lo) < 1e-4:
        return []

    thr = p_lo + 0.22 * (p_hi - p_lo)
    active = smooth > thr

    min_len = max(3, int(round(H * min_band_frac)))
    raw_bands: List[Tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for y in range(H):
        if active[y] and not in_seg:
            seg_start = y
            in_seg = True
        elif not active[y] and in_seg:
            if (y - seg_start) >= min_len:
                raw_bands.append((seg_start, y))
            in_seg = False
    if in_seg and (H - seg_start) >= min_len:
        raw_bands.append((seg_start, H))

    if not raw_bands:
        return []

    # Merge closely spaced staves (e.g. grand staff / piano brace pairs)
    merge_gap = max(2, int(round(H * merge_gap_frac)))
    merged: List[Tuple[int, int]] = []
    for y0, y1 in raw_bands:
        if merged and (y0 - merged[-1][1]) <= merge_gap:
            merged[-1] = (merged[-1][0], y1)
        else:
            merged.append((y0, y1))

    bands: List[Tuple[float, float, float]] = []
    denom = float(max(1, H))
    for y0, y1 in merged:
        seg = smooth[y0:y1]
        if seg.sum() > 1e-6:
            ys = np.arange(y0, y1, dtype=np.float32)
            yc = float((ys * seg).sum() / seg.sum()) / denom
        else:
            yc = 0.5 * (y0 + y1) / denom
        bands.append((round(y0 / denom, 5), round(y1 / denom, 5), round(yc, 5)))
    return bands


def _extract_bands_ink_fallback(img_pil: Any) -> List[Tuple[float, float, float]]:
    """Fallback vertical system band detector via high-pass ink projection."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    gray = np.asarray(img_pil.convert("L"), dtype=np.float32) / 255.0
    H, W = gray.shape
    x0, x1 = int(0.05 * W), max(int(0.95 * W), int(0.05 * W) + 1)
    crop = gray[:, x0:x1]
    row_bg = gaussian_filter1d(np.percentile(crop, 92, axis=1), sigma=max(12.0, H * 0.025))[:, None]
    ink = np.clip(row_bg - crop, 0.0, 1.0)
    row_prof = ink.mean(axis=1)
    bands = _bands_from_1d_profile(row_prof, min_band_frac=0.018, merge_gap_frac=0.028)
    return bands if bands else [(0.02, 0.98, 0.50)]


def _extract_system_bands(
    img_pil: Any,
    staff_seg_logits: Optional["torch.Tensor"],
) -> Tuple[List[Tuple[float, float, float]], str]:
    """Extract normalized system vertical bands (y0, y1, y_center) from StaffSegHead or ink fallback."""
    import torch

    if staff_seg_logits is not None and staff_seg_logits.numel() > 0:
        probs = torch.sigmoid(staff_seg_logits.float())[0, 0]  # (H/4, W/4)
        h4, w4 = probs.shape
        x0, x1 = int(0.04 * w4), max(int(0.96 * w4), int(0.04 * w4) + 1)
        row_prof = probs[:, x0:x1].mean(dim=1).detach().cpu().numpy()
        bands = _bands_from_1d_profile(row_prof, min_band_frac=0.015, merge_gap_frac=0.025)
        if len(bands) >= 1:
            return bands, "staff_seg_head"
    return _extract_bands_ink_fallback(img_pil), "ink_projection_fallback"


def _compute_memory_y_norm(router_info: Dict[str, Any], device: "torch.device") -> Tuple["torch.Tensor", int]:
    """Return normalized y_mem in [0, 1] for all L_mem tokens and k_kept count."""
    import torch

    h_p, w_p = router_info["grid_hw"]
    kept_idx = router_info["kept_idx"][0]  # (k_kept,)
    k_kept = int(kept_idx.shape[0])
    n_bg = int(router_info.get("n_bg_summary", 0))
    y_kept = torch.div(kept_idx, int(w_p), rounding_mode="floor").to(torch.float32) / float(max(1, int(h_p) - 1))
    if n_bg > 0:
        y_bg = torch.full((n_bg,), 0.5, device=device, dtype=torch.float32)
        y_mem = torch.cat([y_kept.to(device), y_bg], dim=0)
    else:
        y_mem = y_kept.to(device)
    return y_mem, k_kept


def _make_gaussian_bias(
    y_mem: "torch.Tensor",
    k_kept: int,
    band: Optional[Tuple[float, float, float]],
    dtype: "torch.dtype",
) -> Optional["torch.Tensor"]:
    """Build additive (1, 1, 1, L_mem) Gaussian vertical bias for the active system band."""
    import torch

    if band is None:
        return None
    y0, y1, yc = band
    band_h = max(0.02, float(y1 - y0))
    sigma_y = min(GAUSS_SIGMA_MAX, max(GAUSS_SIGMA_MIN, GAUSS_SIGMA_BAND_MULT * band_h))
    bias = -((y_mem - float(yc)) ** 2) / (2.0 * sigma_y * sigma_y)
    bias = bias.clamp(min=GAUSS_BIAS_MIN, max=0.0)
    if k_kept < int(bias.shape[0]):
        bias[k_kept:] = 0.0  # Keep bg_summary tokens freely attendable
    return bias.to(dtype=dtype).view(1, 1, 1, -1)


@contextlib.contextmanager
def _inject_cross_attn_bias(model: Any, bias_box: Dict[str, Optional["torch.Tensor"]]) -> Iterator[None]:
    """Temporarily inject additive attn_mask from bias_box['bias'] into all GQADecoderLayer.cross_attn."""
    orig_forwards: List[Tuple[Any, Any]] = []
    for layer in model.dec_layers:
        ca = layer.cross_attn
        orig_fwd = ca.forward
        orig_forwards.append((ca, orig_fwd))

        def _make_wrapped(fn: Any) -> Any:
            def _wrapped(
                q_in: Any,
                kv_in: Any,
                rope_cos: Any = None,
                rope_sin: Any = None,
                is_causal: bool = False,
                attn_mask: Any = None,
                past_kv: Any = None,
                use_cache: bool = False,
                is_cross: bool = False,
            ) -> Any:
                eff_mask = attn_mask
                if is_cross and eff_mask is None:
                    b = bias_box.get("bias")
                    if b is not None:
                        eff_mask = b.to(device=q_in.device, dtype=q_in.dtype)
                return fn(
                    q_in,
                    kv_in,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    is_causal=is_causal,
                    attn_mask=eff_mask,
                    past_kv=past_kv,
                    use_cache=use_cache,
                    is_cross=is_cross,
                )

            return _wrapped

        ca.forward = _make_wrapped(orig_fwd)
    try:
        yield
    finally:
        for ca, orig_fwd in orig_forwards:
            ca.forward = orig_fwd


def _generate_with_staffseg_gauss_ptr(
    model: Any,
    img_pil: Any,
    inp_t: "torch.Tensor",
    *,
    max_len: int,
    amp_dtype: Optional["torch.dtype"],
    amp_enabled: bool,
) -> Tuple[List[int], Dict[str, Any]]:
    """Arm 1: StaffSegHead vertical band extraction + per-<sys> Gaussian cross-attention pointer."""
    import torch
    import eval_uni_suite31 as eus
    import grammar

    device = inp_t.device
    vocab_size = int(model.cfg.vocab_size)
    _, bos_id, eos_id, sys_id, tok_to_id, _ = eus._vocab_spec(vocab_size)

    with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
        mem, aux_enc, router_info = model.encode_image(inp_t, return_aux=True)
        past_kvs = model.precompute_cross_kv(mem)

    bands, band_source = _extract_system_bands(img_pil, aux_enc.get("staff_seg_logits"))
    y_mem, k_kept = _compute_memory_y_norm(router_info, device)
    bias_dtype = amp_dtype if (amp_enabled and amp_dtype is not None) else torch.float32

    cur_sys = 0
    overflow_sys = 0
    bias_box: Dict[str, Optional[torch.Tensor]] = {
        "bias": _make_gaussian_bias(y_mem, k_kept, bands[0] if bands else None, bias_dtype)
    }

    state = grammar.PrefixState(strict=False, vocab_map=tok_to_id)
    out_ids: List[int] = [bos_id]
    cur_id = bos_id

    with _inject_cross_attn_bias(model, bias_box):
        for _ in range(int(max_len)):
            cur_t = torch.tensor([[cur_id]], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
                logits, _, past_kvs = model.decode_tokens(
                    cur_t,
                    memory=None,
                    return_mtp=False,
                    return_aux=False,
                    past_kvs=past_kvs,
                    use_cache=True,
                )
            tok_id = eus._select_constrained_token(logits[0, -1, :], state)
            out_ids.append(tok_id)
            if tok_id == eos_id:
                break
            state = eus._advance_grammar_state(state, tok_id, strict=False, vocab_size=vocab_size)
            if tok_id == sys_id:
                cur_sys += 1
                if cur_sys < len(bands):
                    bias_box["bias"] = _make_gaussian_bias(y_mem, k_kept, bands[cur_sys], bias_dtype)
                else:
                    overflow_sys += 1
                    bias_box["bias"] = None
            cur_id = tok_id

    info = {
        "ink_recall": float(router_info.get("ink_recall", 1.0)),
        "n_bands_est": len(bands),
        "band_source": band_source,
        "overflow_sys": overflow_sys,
        "bands": bands,
    }
    return out_ids, info


def _extract_header_tokens(tokens: List[str]) -> List[str]:
    header: List[str] = []
    for tok in tokens[:24]:
        low = tok.lower()
        if any(low.startswith(p) for p in _HEADER_PREFIXES):
            header.append(tok)
        elif header and (tok in ("=", "|", "<sys>") or re.match(r"^[0-9]+[a-gA-G]", tok)):
            break
    return header


def _stitch_strip_tokens(strip_token_lists: List[List[str]], sys_token: str) -> List[str]:
    """Stitch independently decoded system strips with <sys> and clef/key/time header carry."""
    non_empty = [t for t in strip_token_lists if t]
    if not non_empty:
        return []
    carry_header = _extract_header_tokens(non_empty[0])
    out: List[str] = []
    for idx, toks in enumerate(non_empty):
        # Strip any accidental leading/trailing <sys> inside a single-system crop
        cleaned = [t for t in toks if t != sys_token]
        if not cleaned:
            continue
        if idx > 0 and carry_header and not _extract_header_tokens(cleaned):
            cleaned = list(carry_header) + cleaned
        if out:
            out.append(sys_token)
        out.extend(cleaned)
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


def _select_eval_pieces(manifest_path: Path) -> List[Dict[str, Any]]:
    import eval_uni_suite31 as eus

    _, pieces = eus.resolve_manifest_pieces(manifest_path)
    printed = [p for p in pieces if not bool(p.get("handwritten", False))]
    hw = [p for p in pieces if bool(p.get("handwritten", False))]
    # Put printed safety slice first so the 2-page fail-fast sanity gate runs immediately.
    return printed[:PRINTED_SAFETY_COUNT] + hw


def _eval_arm(
    arm_name: str,
    *,
    mode: str,
    model: Any,
    pieces: List[Dict[str, Any]],
    device: "torch.device",
    amp_dtype: Optional["torch.dtype"],
    sanity_check: bool = False,
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
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "printed_clean": [],
        "printed_photo": [],
        "hw_clean": [],
        "hw_photo": [],
        "printed_all": [],
        "hw_all": [],
        "all48": [],
    }
    per_page: List[Dict[str, Any]] = []
    t_arm0 = time.time()
    t_last_save = time.time()
    amp_enabled = device.type == "cuda" and amp_dtype is not None
    sanity_done = False

    with torch.no_grad():
        for p_idx, entry in enumerate(pieces):
            piece_id = str(entry.get("slug") or entry.get("id") or f"piece_{p_idx}")
            is_hw = bool(entry.get("handwritten", False))
            ref_notes = _extract_musicxml_notes(Path(entry["gt"]))

            for mode_key in ("clean", "photo"):
                img_p = Path(entry[mode_key])
                if not img_p.is_file():
                    continue
                img = Image.open(img_p).convert("RGB")
                t0 = time.perf_counter()

                n_bands_est = 1
                band_source = "none"
                hit_eos = True
                ink_recalls: List[float] = []

                if mode == "fullpage_baseline":
                    inp_t = _pil_to_canonical_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
                    with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
                        seq, stats = eus.generate_mtp_spec(
                            model,
                            inp_t,
                            max_len=FULLPAGE_MAX_LEN,
                            use_grammar=True,
                            strict_grammar=False,
                            return_stats=True,
                        )
                    ids = seq[0].tolist()
                    hit_eos = eos_id in ids[1:]
                    pred_toks = eus.decode_uni_compat(ids, vocab_size=model.cfg.vocab_size, skip_special=True)
                    if not hit_eos:
                        pred_toks = eus.trim_truncated_tokens(pred_toks)
                    rinfo = stats.get("router_info") or {}
                    ink_recalls.append(float(rinfo.get("ink_recall", 1.0)))

                elif mode == "staffseg_gauss_ptr":
                    inp_t = _pil_to_canonical_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
                    ids, ptr_info = _generate_with_staffseg_gauss_ptr(
                        model,
                        img,
                        inp_t,
                        max_len=FULLPAGE_MAX_LEN,
                        amp_dtype=amp_dtype,
                        amp_enabled=amp_enabled,
                    )
                    hit_eos = eos_id in ids[1:]
                    pred_toks = eus.decode_uni_compat(ids, vocab_size=model.cfg.vocab_size, skip_special=True)
                    if not hit_eos:
                        pred_toks = eus.trim_truncated_tokens(pred_toks)
                    ink_recalls.append(float(ptr_info["ink_recall"]))
                    n_bands_est = int(ptr_info["n_bands_est"])
                    band_source = str(ptr_info["band_source"])

                elif mode == "band_crop_oracle":
                    inp_full = _pil_to_canonical_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
                    with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
                        _, aux_enc, _ = model.encode_image(inp_full, return_aux=True)
                    bands, band_source = _extract_system_bands(img, aux_enc.get("staff_seg_logits"))
                    n_bands_est = len(bands)
                    W, H = img.size
                    strip_tok_lists: List[List[str]] = []
                    for y0_n, y1_n, _ in bands:
                        y0_px = int(round(y0_n * H))
                        y1_px = int(round(y1_n * H))
                        bh = max(16, y1_px - y0_px)
                        pad = int(round(STRIP_PAD_FRAC * bh))
                        cy0 = max(0, y0_px - pad)
                        cy1 = min(H, y1_px + pad)
                        if cy1 <= cy0 + 8:
                            continue
                        crop = img.crop((0, cy0, W, cy1))
                        inp_s = _pil_to_canonical_tensor(crop, STRIP_MAX_DIM).unsqueeze(0).to(device)
                        with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
                            seq_s, stats_s = eus.generate_mtp_spec(
                                model,
                                inp_s,
                                max_len=STRIP_MAX_LEN,
                                use_grammar=True,
                                strict_grammar=False,
                                return_stats=True,
                            )
                        ids_s = seq_s[0].tolist()
                        toks_s = eus.decode_uni_compat(ids_s, vocab_size=model.cfg.vocab_size, skip_special=True)
                        if eos_id not in ids_s[1:]:
                            toks_s = eus.trim_truncated_tokens(toks_s)
                            hit_eos = False
                        strip_tok_lists.append(toks_s)
                        rinfo_s = stats_s.get("router_info") or {}
                        ink_recalls.append(float(rinfo_s.get("ink_recall", 1.0)))
                    pred_toks = _stitch_strip_tokens(strip_tok_lists, uni_omr.SYS_TOKEN)
                else:
                    raise ValueError(f"Unknown arm mode: {mode}")

                latency_s = time.perf_counter() - t0
                n_sys_pred = 1 + sum(1 for t in pred_toks if t == uni_omr.SYS_TOKEN) if pred_toks else 0
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
                    "n_bands_est": n_bands_est,
                    "band_source": band_source,
                    "n_sys_pred": n_sys_pred,
                    "hit_eos": bool(hit_eos),
                    "note_f1": round(note_f1, 4),
                    "dec_ok": dec_ok,
                    "ink_recall": round(mean_ir, 4),
                    "latency_s": round(latency_s, 3),
                }
                per_page.append(row)
                buckets[f"{'hw' if is_hw else 'printed'}_{mode_key}"].append(row)
                buckets["hw_all" if is_hw else "printed_all"].append(row)
                buckets["all48"].append(row)

            # Fail-fast 2-page sanity gate on Arm 0
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

    def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0, "note_f1": None, "dec_ok": None, "ink_recall": None, "latency_s": None}
        return {
            "n": len(rows),
            "note_f1": round(sum(r["note_f1"] for r in rows) / len(rows), 4),
            "dec_ok": round(sum(r["dec_ok"] for r in rows) / len(rows), 4),
            "ink_recall": round(sum(r["ink_recall"] for r in rows) / len(rows), 4),
            "hit_eos_rate": round(sum(1.0 for r in rows if r["hit_eos"]) / len(rows), 4),
            "mean_sys_pred": round(sum(r["n_sys_pred"] for r in rows) / len(rows), 2),
            "mean_bands_est": round(sum(r["n_bands_est"] for r in rows) / len(rows), 2),
            "latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 3),
        }

    band_sources: Dict[str, int] = {}
    for r in per_page:
        bs = str(r["band_source"])
        band_sources[bs] = band_sources.get(bs, 0) + 1

    result = {
        "arm": arm_name,
        "mode": mode,
        "summary": {k: _agg(v) for k, v in buckets.items()},
        "band_sources": band_sources,
        "elapsed_secs": round(time.time() - t_arm0, 2),
        "per_page": per_page,
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
        pieces = _select_eval_pieces(manifest_path)
        print(
            f"[{EXP_ID}] Selected {len(pieces)} pieces "
            f"(10 printed safety + {len(pieces) - PRINTED_SAFETY_COUNT} HW) on {gpu_name} ({amp_dtype})",
            flush=True,
        )

        arm0 = _eval_arm(
            "arm0_fullpage_baseline",
            mode="fullpage_baseline",
            model=model,
            pieces=pieces,
            device=device,
            amp_dtype=amp_dtype,
            sanity_check=True,
        )
        arm1 = _eval_arm(
            "arm1_staffseg_gauss_ptr",
            mode="staffseg_gauss_ptr",
            model=model,
            pieces=pieces,
            device=device,
            amp_dtype=amp_dtype,
        )
        arm2 = _eval_arm(
            "arm2_band_crop_oracle",
            mode="band_crop_oracle",
            model=model,
            pieces=pieces,
            device=device,
            amp_dtype=amp_dtype,
        )

        def _delta(b_val: Optional[float], a_val: Optional[float]) -> Optional[float]:
            return None if (b_val is None or a_val is None) else round(float(b_val) - float(a_val), 4)

        s0, s1, s2 = arm0["summary"], arm1["summary"], arm2["summary"]
        comparison = {
            "arm1_minus_arm0_hw_clean": _delta(s1["hw_clean"]["note_f1"], s0["hw_clean"]["note_f1"]),
            "arm1_minus_arm0_hw_photo": _delta(s1["hw_photo"]["note_f1"], s0["hw_photo"]["note_f1"]),
            "arm1_minus_arm0_printed_clean": _delta(s1["printed_clean"]["note_f1"], s0["printed_clean"]["note_f1"]),
            "arm1_minus_arm0_printed_photo": _delta(s1["printed_photo"]["note_f1"], s0["printed_photo"]["note_f1"]),
            "arm2_minus_arm0_hw_clean": _delta(s2["hw_clean"]["note_f1"], s0["hw_clean"]["note_f1"]),
            "arm2_minus_arm0_hw_photo": _delta(s2["hw_photo"]["note_f1"], s0["hw_photo"]["note_f1"]),
            "arm2_minus_arm0_printed_clean": _delta(s2["printed_clean"]["note_f1"], s0["printed_clean"]["note_f1"]),
            "arm2_minus_arm0_printed_photo": _delta(s2["printed_photo"]["note_f1"], s0["printed_photo"]["note_f1"]),
            "arm1_minus_arm0_all48": _delta(s1["all48"]["note_f1"], s0["all48"]["note_f1"]),
            "arm2_minus_arm0_all48": _delta(s2["all48"]["note_f1"], s0["all48"]["note_f1"]),
        }

        report = {
            "exp_id": EXP_ID,
            "exp_code": "i15-01",
            "wave": 13,
            "status": "completed",
            "checkpoint": ckpt_source,
            "ckpt_meta": {
                "arch": ckpt_meta.get("arch"),
                "step": ckpt_meta.get("step"),
                "has_staff_seg_head": getattr(model, "staff_seg_head", None) is not None,
                "missing_keys": len(ckpt_meta.get("missing_keys") or []),
                "unexpected_keys": len(ckpt_meta.get("unexpected_keys") or []),
            },
            "started_utc": started_iso,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"cloud-{gpu_name}",
            "amp_dtype": str(amp_dtype),
            "arm0_fullpage_baseline": arm0,
            "arm1_staffseg_gauss_ptr": arm1,
            "arm2_band_crop_oracle": arm2,
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
            commit_message=f"eval({EXP_ID}): upload BandNet strip oracle & StaffSeg Gaussian pointer results",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload BandNet strip oracle & StaffSeg Gaussian pointer run ledger",
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
