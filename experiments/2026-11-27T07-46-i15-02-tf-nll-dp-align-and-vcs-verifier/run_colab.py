#!/usr/bin/env python3
"""Colab L4 runner for Wave 13 Exp i15-02:
AR-Free Teacher-Forced NLL + Cross-Attention Monotonicity DP Alignment
and Visual Consistency Score (VCS) Verifier Probe.

Experiment ID: 2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier
Validates Pivot Strategist 2's Pivots 1 & 3 (100% permissive: PDM 1.0 +
MIT hw-score-sim + MIT/Apache mscz-rs, zero NC/GPL):

  - Part A (`part_a_vcs_verifier_corr`):
      Computes 4-signal Visual Consistency Score (VCS):
        1. High-pass ink IoU (`vcs_ink_iou`)
        2. Barline-X column projection correlation (`vcs_barline_x_corr`)
        3. Cross-scale 1280px vs 1536px Note F1 agreement (`vcs_cross_scale_agree`)
        4. Frozen encoder patch cosine on real page vs synthetic bekern render
           (`vcs_encoder_patch_cos`)
      on all 14 `suite31_hw` real historical HW pieces (clean + photo = 28 pages).
      Measures Pearson r(VCS, Strict Note F1) and false-positive rejection rate
      on `ov-39`-passing RISM-like hallucinations. Gate: `r >= 0.70`.

  - Part B (`part_b_tf_nll_dp_align_bench`):
      Benchmarks AR-free Teacher-Forced NLL + Cross-Attention Monotonicity DP
      (`log P(bekern[m_a..m_b] | image)` + reading-order monotonicity & coverage)
      vs free-decoded alignment on multi-system HW spans (selecting true
      `[m_a..m_b]` among shifted and boundary-perturbed measure windows).
      Measures Top-1 span accuracy and exact measure-boundary precision.
      Gate: `Top-1 >= 0.85`.

Uploads:
  - evals/2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier.json
  - runs/2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier.json
to gregoreesmaa/sheet-music-converter-omr.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier"
EXP_CODE = "i15-02"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter10_5_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter10_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
)
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

VOL_ROOT = Path("/vol") if Path("/vol").is_dir() else Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

SEED = 42
SCALE_HI_DIM = 1536
SCALE_LO_DIM = 1280
DECODE_MAX_LEN = 3072
TF_MAX_LEN = 2048
SAVE_EVERY_MINUTES = 5.0

# Gates
GATE_PART_A_PEARSON_R = 0.70
GATE_PART_B_TOP1_ACC = 0.85

# Cross-attention DP weights
DP_LAMBDA_MONO = 0.35
DP_LAMBDA_COV = 0.25

# ov-39 promoted filter thresholds
OV39_MIN_LOGPROB = -0.35
OV39_MIN_BEAT_VALID = 0.70

_BAR_TOKENS = frozenset({"bar", "dbar", "rbar", "fbar", "=", "==", "=|", "=||", "|"})
_DUR_TOKENS = frozenset(
    {
        "breve",
        "whole",
        "half",
        "quarter",
        "eighth",
        "16th",
        "32nd",
        "64th",
        "128th",
    }
)
_PITCH_RE = re.compile(r"^([A-Ga-g])([0-9])$")


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
        if (cand / "uni_omr_v2.py").is_file() and (cand / "eval_uni_suite31.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            root = cand.parent.parent if cand.name == "mvd" and cand.parent.name == "tmp-train" else cand.parent
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
                if (mvd / "uni_omr_v2.py").is_file() and (mvd / "eval_uni_suite31.py").is_file():
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
        raise RuntimeError(f"No frozen UniOMR checkpoint found among {HF_CKPT_CANDIDATES}")

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
        f"vocab={getattr(cfg, 'vocab_size', None)} "
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


def _pil_to_canonical_tensor(img_pil: Any, max_dim: int) -> "torch.Tensor":
    """Canonical [0, 1] CHW tensor matching uni_omr._resize_fullpage_pil + _pil_to_chw_tensor."""
    import uni_omr

    work = img_pil.convert("RGB")
    pil_im = uni_omr._resize_fullpage_pil(work, int(max_dim))
    return uni_omr._pil_to_chw_tensor(pil_im)


def _xml_notes(xml_path: Path) -> List[Tuple[str, int, str, str, str]]:
    """Extract ordered (dur, n_dots, step, octave, alter) notes from GT MusicXML."""
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


def _musicxml_to_bekern_measures(
    xml_path: Path, tok_to_id: Dict[str, int]
) -> Tuple[List[str], List[List[str]]]:
    """Convert GT MusicXML into header tokens + measure-grouped canonical bekern tokens.

    Every emitted token is validated against ``tok_to_id`` so teacher-forced scoring
    has zero OOV tokens across both v1 (182-vocab) and v2 (237-vocab) checkpoints.
    """
    header: List[str] = []
    measures: List[List[str]] = []
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return header, measures

    # Pick the first <part> (or iterate all parts sequentially for multi-part scores)
    parts = [el for el in root if el.tag.split("}", 1)[-1] == "part"]
    if not parts:
        return header, measures

    bar_tok = "bar" if "bar" in tok_to_id else ("=" if "=" in tok_to_id else "")
    chord_tok = "chord" if "chord" in tok_to_id else ""
    rest_tok = "rest" if "rest" in tok_to_id else ""
    dot_tok = "dot" if "dot" in tok_to_id else ""

    for p_idx, part in enumerate(parts):
        for m_el in part:
            if m_el.tag.split("}", 1)[-1] != "measure":
                continue
            m_toks: List[str] = []
            for child in m_el:
                ctag = child.tag.split("}", 1)[-1]
                if ctag == "attributes":
                    for attr in child:
                        atag = attr.tag.split("}", 1)[-1]
                        if atag == "clef":
                            sign = ""
                            line = ""
                            for cc in attr:
                                cct = cc.tag.split("}", 1)[-1]
                                if cct == "sign":
                                    sign = (cc.text or "").strip().upper()
                                elif cct == "line":
                                    line = (cc.text or "").strip()
                            cand = f"clef-{sign}{line}"
                            if cand in tok_to_id:
                                if not header and p_idx == 0:
                                    header.append(cand)
                                m_toks.append(cand)
                        elif atag == "key":
                            fifths = ""
                            for kc in attr:
                                if kc.tag.split("}", 1)[-1] == "fifths":
                                    fifths = (kc.text or "").strip()
                            cand = f"key-{fifths}"
                            if cand in tok_to_id:
                                if p_idx == 0 and len(header) < 4:
                                    header.append(cand)
                                m_toks.append(cand)
                        elif atag == "time":
                            beats = btype = ""
                            for tc in attr:
                                tct = tc.tag.split("}", 1)[-1]
                                if tct == "beats":
                                    beats = (tc.text or "").strip()
                                elif tct == "beat-type":
                                    btype = (tc.text or "").strip()
                            cand = f"time-{beats}/{btype}"
                            if cand in tok_to_id:
                                if p_idx == 0 and len(header) < 4:
                                    header.append(cand)
                                m_toks.append(cand)
                elif ctag == "note":
                    is_rest = is_grace = is_chord = False
                    step = octave = alter = dur = ""
                    n_dots = 0
                    for nc in child:
                        nct = nc.tag.split("}", 1)[-1]
                        if nct == "rest":
                            is_rest = True
                        elif nct == "grace":
                            is_grace = True
                        elif nct == "chord":
                            is_chord = True
                        elif nct == "type":
                            dur = (nc.text or "").strip().lower()
                        elif nct == "dot":
                            n_dots += 1
                        elif nct == "pitch":
                            for pc in nc:
                                pct = pc.tag.split("}", 1)[-1]
                                txt = (pc.text or "").strip()
                                if pct == "step":
                                    step = txt.upper()
                                elif pct == "octave":
                                    octave = txt
                                elif pct == "alter":
                                    try:
                                        av = int(round(float(txt)))
                                        if av != 0:
                                            alter = f"a{av}"
                                    except ValueError:
                                        pass
                    if is_grace:
                        continue
                    if is_chord and chord_tok:
                        m_toks.append(chord_tok)
                    if dur and dur in tok_to_id:
                        m_toks.append(dur)
                    if dot_tok and n_dots > 0:
                        for _ in range(min(2, n_dots)):
                            m_toks.append(dot_tok)
                    if is_rest:
                        if rest_tok:
                            m_toks.append(rest_tok)
                    elif step and octave:
                        p_tok = f"{step}{octave}"
                        if p_tok in tok_to_id:
                            m_toks.append(p_tok)
                        else:
                            if step in tok_to_id:
                                m_toks.append(step)
                            if octave in tok_to_id:
                                m_toks.append(octave)
                        if alter and alter in tok_to_id:
                            m_toks.append(alter)
            if bar_tok:
                m_toks.append(bar_tok)
            if len(m_toks) >= 2:
                measures.append(m_toks)
        # Use first part for clean 1-voice/grand-staff measure stream if already populated
        if len(measures) >= 4:
            break

    return header, measures


def _pearson_r(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = min(len(xs), len(ys))
    if n < 3:
        return None
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if dx < 1e-12 or dy < 1e-12:
        return 0.0
    return num / (dx * dy)


def _spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = min(len(xs), len(ys))
    if n < 3:
        return None

    def _ranks(vals: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_r = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg_r
            i = j + 1
        return r

    return _pearson_r(_ranks(xs[:n]), _ranks(ys[:n]))


def _compute_auroc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> Optional[float]:
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    total = len(pos_scores) * len(neg_scores)
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / float(total)


# ---------------------------------------------------------------------------
# Visual Consistency Score (VCS) Helpers (100% Pure Python / NumPy / PIL)
# ---------------------------------------------------------------------------

def _extract_real_ink_and_barlines(img_pil: Any, grid_hw: Tuple[int, int] = (64, 64)) -> Dict[str, Any]:
    """Compute high-pass ink map, detected system bands, and vertical barline-X projection."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d, uniform_filter

    gray = np.asarray(img_pil.convert("L"), dtype=np.float32) / 255.0
    H, W = gray.shape
    # High-pass ink via local background subtraction (robust to yellowing/lighting)
    bg = uniform_filter(gray, size=(max(15, H // 40), max(15, W // 40)))
    ink = np.clip((bg - gray) / np.maximum(bg, 0.15), 0.0, 1.0)

    # Mask dark exterior borders (4-edge inward threshold)
    bright = gray >= 0.35
    cs_y = np.cumsum(bright, axis=0)
    cs_x = np.cumsum(bright, axis=1)
    edge_dark = (cs_y == 0) | (cs_y == cs_y[-1:, :]) | (cs_x == 0) | (cs_x == cs_x[:, -1:])
    paper_mask = (~(edge_dark & (gray < 0.22))).astype(np.float32)
    ink = ink * paper_mask

    # Pool ink to (grid_h, grid_w)
    gh, gw = grid_hw
    ink_pil_f = ink
    # Fast block pooling via reshape/interpolate via PIL
    from PIL import Image

    ink_u8 = np.clip(ink_pil_f * 255.0, 0, 255).astype(np.uint8)
    ink_grid = np.asarray(Image.fromarray(ink_u8).resize((gw, gh), Image.BILINEAR), dtype=np.float32) / 255.0
    ink_max = float(ink_grid.max())
    if ink_max > 1e-6:
        ink_grid = ink_grid / ink_max

    # System band detection via horizontal ink projection
    row_prof = ink.mean(axis=1)
    smooth_row = gaussian_filter1d(row_prof, sigma=max(3.0, H * 0.008))
    thr = float(np.percentile(smooth_row, 25)) + 0.15 * float(smooth_row.max() - smooth_row.min())
    active = smooth_row > max(0.01, thr)

    bands: List[Tuple[float, float]] = []
    in_b = False
    y0 = 0
    min_h = max(8, int(H * 0.015))
    for y in range(H):
        if active[y] and not in_b:
            y0 = y
            in_b = True
        elif not active[y] and in_b:
            if (y - y0) >= min_h:
                bands.append((y0 / float(H), y / float(H)))
            in_b = False
    if in_b and (H - y0) >= min_h:
        bands.append((y0 / float(H), 1.0))

    if not bands:
        bands = [(0.08, 0.92)]

    # Barline-X projection: vertical-stroke response inside detected system bands
    # Vertical kernel emphasizes thin tall vertical strokes (barlines/stems) over horizontal staff lines
    gy_abs = np.abs(np.diff(ink, axis=0, prepend=ink[:1, :]))
    gx_abs = np.abs(np.diff(ink, axis=1, prepend=ink[:, :1]))
    vert_ink = np.clip(ink - 0.5 * gy_abs, 0.0, 1.0) * (gx_abs > 0.02).astype(np.float32)

    col_prof = np.zeros(W, dtype=np.float32)
    for b0, b1 in bands:
        iy0, iy1 = int(b0 * H), max(int(b0 * H) + 1, int(b1 * H))
        sub = vert_ink[iy0:iy1, :]
        # Strong vertical runs have high column mean inside the band
        col_prof += sub.mean(axis=0)
    col_prof = col_prof / float(max(1, len(bands)))
    col_prof = gaussian_filter1d(col_prof, sigma=max(1.5, W * 0.004))

    # Resample barline-X profile to length gw
    x_idx = np.linspace(0, W - 1, gw)
    bar_x_prof = np.interp(x_idx, np.arange(W), col_prof).astype(np.float32)
    if bar_x_prof.max() > 1e-6:
        bar_x_prof = bar_x_prof / bar_x_prof.max()

    y_min = bands[0][0]
    y_max = bands[-1][1]
    return {
        "ink_grid": ink_grid,
        "bar_x_prof": bar_x_prof,
        "bands": bands,
        "y_span": (float(y_min), float(y_max)),
    }


def _render_bekern_silhouette_and_image(
    pred_toks: Sequence[str],
    real_bands: Sequence[Tuple[float, float]],
    target_size: Tuple[int, int] = (512, 640),
    grid_hw: Tuple[int, int] = (64, 64),
) -> Tuple[Any, Any, Any]:
    """Deterministically render predicted bekern into:
    1. 2D normalized ink grid `(gh, gw)` for `vcs_ink_iou`
    2. 1D barline-X profile `(gw,)` for `vcs_barline_x_corr`
    3. Synthetic RGB PIL image `(W, H)` for frozen-encoder `vcs_encoder_patch_cos`
    100% pure Python/NumPy/PIL (zero external renderer / zero GPL).
    """
    import numpy as np
    from PIL import Image, ImageDraw
    import uni_omr

    # Split tokens into systems by <sys>
    systems: List[List[str]] = []
    cur: List[str] = []
    for t in pred_toks:
        if t == uni_omr.SYS_TOKEN:
            if cur:
                systems.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        systems.append(cur)

    W, H = target_size
    gh, gw = grid_hw
    img = Image.new("RGB", (W, H), (245, 242, 234))
    draw = ImageDraw.Draw(img)
    bar_x_prof = np.zeros(gw, dtype=np.float32)

    if not systems:
        return np.zeros((gh, gw), dtype=np.float32), bar_x_prof, img

    n_sys = len(systems)
    # Map predicted systems vertically: use real_bands span if available, else [0.06, 0.94]
    y_top = real_bands[0][0] if real_bands else 0.06
    y_bot = real_bands[-1][1] if real_bands else 0.94
    y_top = max(0.03, min(0.25, y_top))
    y_bot = max(y_top + 0.20, min(0.97, y_bot))

    x_left = int(0.06 * W)
    x_right = int(0.94 * W)
    sys_slot_h = (y_bot - y_top) * H / float(max(1, n_sys))
    staff_h = max(14.0, min(sys_slot_h * 0.55, H * 0.07))

    for s_idx, s_toks in enumerate(systems):
        yc = int((y_top * H) + (s_idx + 0.5) * sys_slot_h)
        st_y0 = int(yc - staff_h * 0.5)
        st_y1 = int(yc + staff_h * 0.5)
        # Draw 5 staff lines
        for line_i in range(5):
            ly = int(st_y0 + line_i * (staff_h / 4.0))
            draw.line([(x_left, ly), (x_right, ly)], fill=(45, 42, 38), width=1)

        # Split system into measures by barline tokens
        measures: List[List[str]] = []
        m_cur: List[str] = []
        for t in s_toks:
            if t in _BAR_TOKENS:
                measures.append(m_cur)
                m_cur = []
            else:
                m_cur.append(t)
        if m_cur:
            measures.append(m_cur)
        if not measures:
            continue

        # Weight horizontal width by token density per measure
        m_weights = [max(2, len(m)) for m in measures]
        tot_w = float(sum(m_weights))
        x_cursor = float(x_left)
        usable_w = float(x_right - x_left)

        for m_idx, (m_toks, mw) in enumerate(zip(measures, m_weights)):
            seg_w = usable_w * (float(mw) / tot_w)
            x_end = x_cursor + seg_w
            # Draw notes/rests across the measure
            n_ev = max(1, sum(1 for t in m_toks if t in _DUR_TOKENS or t == "rest" or _PITCH_RE.match(t)))
            ev_i = 0
            for t in m_toks:
                m_pitch = _PITCH_RE.match(t)
                if m_pitch or t == "rest":
                    ex = int(x_cursor + (ev_i + 0.5) * (seg_w / float(n_ev)))
                    if m_pitch:
                        step_idx = "CDEFGAB".index(m_pitch.group(1).upper())
                        oct_v = int(m_pitch.group(2))
                        rel = (oct_v - 4) * 7 + step_idx - 4  # relative to G4
                        ey = int(yc - rel * (staff_h / 10.0))
                        ey = max(st_y0 - int(staff_h * 0.4), min(st_y1 + int(staff_h * 0.4), ey))
                        r = max(2, int(staff_h * 0.14))
                        draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(20, 18, 16))
                        # Stem
                        stem_h = int(staff_h * 0.75)
                        draw.line([(ex + r, ey), (ex + r, ey - stem_h)], fill=(25, 22, 20), width=1)
                    else:
                        r = max(2, int(staff_h * 0.12))
                        draw.rectangle([ex - r, yc - r, ex + r, yc + r], fill=(35, 32, 28))
                    ev_i += 1
            # Draw measure barline at x_end
            bx = int(min(x_right, max(x_left, round(x_end))))
            draw.line([(bx, st_y0), (bx, st_y1)], fill=(18, 16, 14), width=2)
            gx = int(min(gw - 1, max(0, round((bx / float(W)) * (gw - 1)))))
            bar_x_prof[gx] += 1.0
            x_cursor = x_end

    # Convert rendered image into normalized ink grid (gh, gw)
    gray_r = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    ink_r = np.clip(0.95 - gray_r, 0.0, 1.0)
    ink_u8 = np.clip(ink_r * 255.0, 0, 255).astype(np.uint8)
    pred_ink_grid = np.asarray(Image.fromarray(ink_u8).resize((gw, gh), Image.BILINEAR), dtype=np.float32) / 255.0
    if pred_ink_grid.max() > 1e-6:
        pred_ink_grid = pred_ink_grid / pred_ink_grid.max()

    if bar_x_prof.max() > 1e-6:
        from scipy.ndimage import gaussian_filter1d

        bar_x_prof = gaussian_filter1d(bar_x_prof, sigma=1.0)
        if bar_x_prof.max() > 1e-6:
            bar_x_prof = bar_x_prof / bar_x_prof.max()

    return pred_ink_grid, bar_x_prof, img


def _compute_vcs_components(
    real_feats: Dict[str, Any],
    pred_toks_hi: Sequence[str],
    pred_toks_lo: Sequence[str],
    real_mem_vec: "torch.Tensor",
    model: Any,
    device: "torch.device",
    amp_dtype: Optional["torch.dtype"],
    amp_enabled: bool,
) -> Dict[str, float]:
    """Compute the 4 Visual Consistency Score (VCS) components in [0, 1]."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    import eval_uni_suite31 as eus
    import rlvr_grpo

    pred_ink_grid, pred_bar_x, synth_pil = _render_bekern_silhouette_and_image(
        pred_toks_hi, real_feats["bands"]
    )

    # 1. High-pass ink IoU (soft Dice/IoU on normalized (64, 64) ink maps)
    a = real_feats["ink_grid"]
    b = pred_ink_grid
    inter = float(np.minimum(a, b).sum())
    union = float(np.maximum(a, b).sum())
    vcs_ink_iou = inter / max(1e-6, union)

    # 2. Barline-X correlation mapped from [-1, 1] -> [0, 1]
    bx_r = real_feats["bar_x_prof"]
    bx_p = pred_bar_x
    r_bx = _pearson_r(bx_r.tolist(), bx_p.tolist())
    vcs_barline_x_corr = 0.5 * (float(r_bx if r_bx is not None else 0.0) + 1.0)

    # 3. Cross-scale 1280px vs 1536px Note F1 agreement
    st_hi, _ = eus.stitch_uni_tokens(list(pred_toks_hi))
    st_lo, _ = eus.stitch_uni_tokens(list(pred_toks_lo))
    notes_hi = rlvr_grpo.extract_pitch_dur_notes(st_hi)
    notes_lo = rlvr_grpo.extract_pitch_dur_notes(st_lo)
    vcs_cross_scale_agree = float(rlvr_grpo.compute_sequence_f1(notes_hi, notes_lo))

    # 4. Frozen encoder patch cosine similarity (real HW page vs synthetic render)
    synth_t = _pil_to_canonical_tensor(synth_pil, 640).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
            s_mem, _, _ = model.encode_image(synth_t, return_aux=False)
        s_final = s_mem[-1] if isinstance(s_mem, (tuple, list)) else s_mem
        s_vec = F.normalize(s_final.float().mean(dim=1), dim=-1)
        cos_val = float((real_mem_vec * s_vec).sum(dim=-1).item())
    vcs_encoder_patch_cos = max(0.0, min(1.0, 0.5 * (cos_val + 1.0)))

    vcs_total = 0.25 * (
        vcs_ink_iou + vcs_barline_x_corr + vcs_cross_scale_agree + vcs_encoder_patch_cos
    )
    return {
        "vcs_ink_iou": round(vcs_ink_iou, 4),
        "vcs_barline_x_corr": round(vcs_barline_x_corr, 4),
        "vcs_cross_scale_agree": round(vcs_cross_scale_agree, 4),
        "vcs_encoder_patch_cos": round(vcs_encoder_patch_cos, 4),
        "vcs_total": round(vcs_total, 4),
    }


# ---------------------------------------------------------------------------
# Teacher-Forced NLL + Cross-Attention Monotonicity DP (Part B)
# ---------------------------------------------------------------------------

def _extract_routed_norm_coords(router_info: Dict[str, Any], n_mem: int, device: "torch.device") -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Derive normalized (y_norm, x_norm) in [0, 1] for each memory token from router_info."""
    import torch

    grid_hw = router_info.get("grid_hw")
    kept_idx = router_info.get("kept_idx")
    if grid_hw is not None and kept_idx is not None:
        h_p, w_p = int(grid_hw[0]), int(grid_hw[1])
        idx0 = kept_idx[0].to(device=device, dtype=torch.long)
        ys = (idx0 // max(1, w_p)).float() / float(max(1, h_p - 1))
        xs = (idx0 % max(1, w_p)).float() / float(max(1, w_p - 1))
        if idx0.numel() < n_mem:
            pad_n = n_mem - idx0.numel()
            ys = torch.cat([ys, torch.full((pad_n,), 0.5, device=device)], dim=0)
            xs = torch.cat([xs, torch.full((pad_n,), 0.5, device=device)], dim=0)
        return ys[:n_mem], xs[:n_mem]
    # Fallback uniform 1D reading order
    lin = torch.linspace(0.0, 1.0, n_mem, device=device)
    return lin, lin


def _teacher_forced_nll_and_ca_mono(
    model: Any,
    mem_levels: Any,
    router_info: Dict[str, Any],
    tok_ids: List[int],
    measure_ranges: List[Tuple[int, int]],
    page_y_span: Tuple[float, float],
    pad_id: int,
    device: "torch.device",
    amp_dtype: Optional["torch.dtype"],
    amp_enabled: bool,
) -> Dict[str, float]:
    """Run a single AR-free teacher-forced pass over `tok_ids` conditioned on cached `mem_levels`,
    capturing cross-attention weights on the last 2 decoder layers via non-invasive hooks to score
    both NLL (`-mean_ce`) and Cross-Attention Monotonicity DP."""
    import torch
    import torch.nn.functional as F

    if len(tok_ids) < 3:
        return {
            "tf_mean_logprob": -10.0,
            "ca_mono_score": 0.0,
            "ca_cov_score": 0.0,
            "dp_total_score": -10.0,
        }

    tgt = torch.tensor(tok_ids[:TF_MAX_LEN], dtype=torch.long, device=device).unsqueeze(0)
    inp_ids = tgt[:, :-1]
    lbl_ids = tgt[:, 1:]

    # Attach temporary forward pre-hooks to the last 2 decoder cross-attention modules
    dec_layers = getattr(model, "dec_layers", [])
    hook_layers = dec_layers[-2:] if len(dec_layers) >= 2 else dec_layers
    captured_attn: List[torch.Tensor] = []
    handles: List[Any] = []

    def _make_pre_hook(ca_mod: Any):
        def _pre_hook(_mod: Any, args: Tuple[Any, ...]) -> None:
            if len(args) < 2 or args[0] is None or args[1] is None:
                return
            q_in, kv_in = args[0], args[1]
            with torch.no_grad():
                B_c, Lq, _ = q_in.shape
                Lk = kv_in.shape[1]
                if hasattr(ca_mod, "q_proj") and hasattr(ca_mod, "k_proj"):
                    q = ca_mod.q_proj(q_in).view(B_c, Lq, ca_mod.n_heads, ca_mod.head_dim)
                    if hasattr(ca_mod, "q_norm"):
                        q = ca_mod.q_norm(q)
                    q = q.transpose(1, 2).float()
                    n_kv = getattr(ca_mod, "n_kv_heads", ca_mod.n_heads)
                    k = ca_mod.k_proj(kv_in).view(B_c, Lk, n_kv, ca_mod.head_dim)
                    if hasattr(ca_mod, "k_norm"):
                        k = ca_mod.k_norm(k)
                    k = k.transpose(1, 2).float()
                    if n_kv != ca_mod.n_heads and hasattr(ca_mod, "_expand_kv"):
                        k = ca_mod._expand_kv(k)
                    scale = 1.0 / math.sqrt(float(ca_mod.head_dim))
                    logits_ca = torch.matmul(q, k.transpose(-2, -1)) * scale
                    probs_ca = torch.softmax(logits_ca, dim=-1).mean(dim=1)[0]  # (Lq, Lk)
                    captured_attn.append(probs_ca.detach())
        return _pre_hook

    for layer in hook_layers:
        ca = getattr(layer, "cross_attn", None)
        if ca is not None:
            handles.append(ca.register_forward_pre_hook(_make_pre_hook(ca)))

    try:
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
                dec_out = model.decode_tokens(
                    inp_ids,
                    mem_levels,
                    return_mtp=False,
                    return_aux=False,
                    apply_tok_dropout=False,
                )
            logits = dec_out[0] if isinstance(dec_out, (tuple, list)) else dec_out
            nll = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                lbl_ids.reshape(-1),
                ignore_index=pad_id,
                reduction="mean",
            )
            tf_mean_logprob = -float(nll.item())
    finally:
        for h in handles:
            h.remove()

    # Compute Cross-Attention Monotonicity & Coverage score over measure spans
    ca_mono_score = 0.0
    ca_cov_score = 0.0
    if captured_attn:
        attn_avg = torch.stack(captured_attn, dim=0).mean(dim=0)  # (T_in, L_mem)
        T_in, L_mem = attn_avg.shape
        ys_mem, xs_mem = _extract_routed_norm_coords(router_info, L_mem, attn_avg.device)
        tok_y = torch.matmul(attn_avg, ys_mem)  # (T_in,)
        tok_x = torch.matmul(attn_avg, xs_mem)  # (T_in,)

        m_centroids: List[Tuple[float, float]] = []
        for s_i, e_i in measure_ranges:
            cs = max(0, min(T_in - 1, s_i))
            ce = max(cs + 1, min(T_in, e_i))
            m_centroids.append((float(tok_y[cs:ce].mean().item()), float(tok_x[cs:ce].mean().item())))

        if len(m_centroids) >= 2:
            mono_ok = 0.0
            for idx_m in range(len(m_centroids) - 1):
                y0, x0 = m_centroids[idx_m]
                y1, x1 = m_centroids[idx_m + 1]
                same_sys_forward = (abs(y1 - y0) <= 0.09) and (x1 >= x0 - 0.06)
                next_sys_down = (y1 > y0 + 0.03)
                if same_sys_forward or next_sys_down:
                    mono_ok += 1.0
            ca_mono_score = mono_ok / float(len(m_centroids) - 1)

            # Vertical coverage match against detected real ink y_span
            att_y_min = min(c[0] for c in m_centroids)
            att_y_max = max(c[0] for c in m_centroids)
            att_span = max(0.0, att_y_max - att_y_min)
            real_span = max(0.10, page_y_span[1] - page_y_span[0])
            cov_ratio = min(att_span, real_span) / max(att_span, real_span, 1e-6)
            ca_cov_score = float(max(0.0, min(1.0, cov_ratio)))

    dp_total = tf_mean_logprob + DP_LAMBDA_MONO * ca_mono_score + DP_LAMBDA_COV * ca_cov_score
    return {
        "tf_mean_logprob": round(tf_mean_logprob, 5),
        "ca_mono_score": round(ca_mono_score, 4),
        "ca_cov_score": round(ca_cov_score, 4),
        "dp_total_score": round(dp_total, 5),
    }


def _decode_page(
    model: Any,
    img_pil: Any,
    max_dim: int,
    max_len: int,
    vocab_size: int,
    eos_id: int,
    device: "torch.device",
    amp_dtype: Optional["torch.dtype"],
    amp_enabled: bool,
) -> Tuple[List[str], List[int], bool, Any, Dict[str, Any]]:
    """Decode one page canonically at `max_dim`, returning decoded tokens, raw IDs, EOS flag,
    cached encoder memory, and router_info."""
    import torch
    import eval_uni_suite31 as eus

    inp_t = _pil_to_canonical_tensor(img_pil, max_dim).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
            mem_levels, _, router_info = model.encode_image(inp_t, return_aux=False)
            seq = model.generate(inp_t, max_len=max_len)
    ids = seq[0].tolist()
    hit_eos = eos_id in ids[1:]
    pred_toks = eus.decode_uni_compat(ids, vocab_size=vocab_size, skip_special=True)
    if not hit_eos:
        pred_toks = eus.trim_truncated_tokens(pred_toks)
    return pred_toks, ids, hit_eos, mem_levels, router_info


def _run_2page_sanity_gate(
    model: Any,
    pieces: List[Dict[str, Any]],
    device: "torch.device",
    amp_dtype: Optional["torch.dtype"],
    amp_enabled: bool,
) -> Dict[str, Any]:
    """Fail-fast 2-page sanity gate on the first 2 printed clean pages."""
    from PIL import Image
    import eval_uni_suite31 as eus
    import rlvr_grpo

    vsz = int(model.cfg.vocab_size)
    _, _, eos_id, _, _, id_to_tok = eus._vocab_spec(vsz)
    printed = [p for p in pieces if not bool(p.get("handwritten", False))][:2]
    f1s: List[float] = []
    dec_oks: List[float] = []

    for entry in printed:
        img = Image.open(entry["clean"]).convert("RGB")
        ref_notes = _xml_notes(Path(entry["gt"]))
        pred_toks, _, hit_eos, _, _ = _decode_page(
            model, img, SCALE_HI_DIM, DECODE_MAX_LEN, vsz, eos_id, device, amp_dtype, amp_enabled
        )
        hyp_toks, _ = eus.stitch_uni_tokens(pred_toks)
        hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
        nf1 = float(rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes))
        struct = rlvr_grpo.compute_unlabeled_structural_reward(
            hyp_toks, vocab=id_to_tok, terminated_eos=hit_eos
        )
        dec_ok = 1.0 if (len(hyp_notes) > 0 and float(struct.get("r_parse", 0.0)) > 0.0) else 0.0
        f1s.append(nf1)
        dec_oks.append(dec_ok)

    mean_f1 = sum(f1s) / max(1, len(f1s))
    mean_ok = sum(dec_oks) / max(1, len(dec_oks))
    print(
        f"[{EXP_ID}] 2-page printed sanity gate: mean_note_f1={mean_f1:.4f} dec_ok={mean_ok:.2f}",
        flush=True,
    )
    if mean_ok <= 0.0 or mean_f1 < 0.05:
        raise RuntimeError(
            f"Fail-fast 2-page sanity gate tripped: mean_note_f1={mean_f1:.4f}, dec_ok={mean_ok:.2f}"
        )
    return {"mean_note_f1": round(mean_f1, 4), "dec_ok": round(mean_ok, 4), "n": len(f1s)}


def _run_part_a_and_b(
    model: Any,
    pieces: List[Dict[str, Any]],
    device: "torch.device",
    amp_dtype: Optional["torch.dtype"],
    amp_enabled: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Part A (VCS Verifier Correlation + ov-39 Hallucination Rejection) and
    Part B (AR-Free TF-NLL + Cross-Attention Monotonicity DP vs Free-Decoded Alignment)
    over all 14 real historical HW pieces x {clean, photo} = 28 pages."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import eval_uni_suite31 as eus
    import rlvr_grpo

    exp_dir = VOL_ROOT / "ckpts" / EXP_ID
    exp_dir.mkdir(parents=True, exist_ok=True)
    part_a_cache = exp_dir / "part_a_result.json"
    part_b_cache = exp_dir / "part_b_result.json"

    vsz = int(model.cfg.vocab_size)
    pad_id, bos_id, eos_id, _, tok_to_id, id_to_tok = eus._vocab_spec(vsz)

    hw_pieces = [p for p in pieces if bool(p.get("handwritten", False))]
    print(f"[{EXP_ID}] Found {len(hw_pieces)} real HW pieces ({len(hw_pieces) * 2} clean+photo pages)", flush=True)

    # Pre-parse GT MusicXML into measure-grouped bekern for all HW pieces (for Part B distractor pool)
    piece_gt_measures: Dict[str, Tuple[List[str], List[List[str]]]] = {}
    all_donor_measures: List[List[str]] = []
    for entry in hw_pieces:
        pid = str(entry.get("slug") or entry.get("piece_id") or Path(entry["gt"]).stem)
        hdr, meas = _musicxml_to_bekern_measures(Path(entry["gt"]), tok_to_id)
        piece_gt_measures[pid] = (hdr, meas)
        all_donor_measures.extend(meas)

    # Decode all 28 HW pages once (shared between Part A and Part B)
    decoded_pages: List[Dict[str, Any]] = []
    t_last_save = time.time()
    t_part0 = time.time()

    for p_idx, entry in enumerate(hw_pieces):
        pid = str(entry.get("slug") or entry.get("piece_id") or Path(entry["gt"]).stem)
        ref_notes = _xml_notes(Path(entry["gt"]))
        for mode_key in ("clean", "photo"):
            img_p = Path(entry[mode_key])
            if not img_p.is_file():
                continue
            img = Image.open(img_p).convert("RGB")
            t0 = time.perf_counter()

            # High-res 1536px decode + cached encoder features
            pred_hi, ids_hi, hit_eos_hi, mem_hi, rinfo_hi = _decode_page(
                model, img, SCALE_HI_DIM, DECODE_MAX_LEN, vsz, eos_id, device, amp_dtype, amp_enabled
            )
            # Low-res 1280px decode for cross-scale VCS agreement
            pred_lo, _, _, _, _ = _decode_page(
                model, img, SCALE_LO_DIM, DECODE_MAX_LEN, vsz, eos_id, device, amp_dtype, amp_enabled
            )

            # Single teacher-forced pass for exact mean_logprob of base_1536 decode
            tf_self = _teacher_forced_nll_and_ca_mono(
                model=model,
                mem_levels=mem_hi,
                router_info=rinfo_hi,
                tok_ids=ids_hi,
                measure_ranges=[],
                page_y_span=(0.05, 0.95),
                pad_id=pad_id,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )

            hyp_toks_hi, _ = eus.stitch_uni_tokens(pred_hi)
            hyp_notes_hi = rlvr_grpo.extract_pitch_dur_notes(hyp_toks_hi)
            note_f1 = float(rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes_hi))
            struct = rlvr_grpo.compute_unlabeled_structural_reward(
                hyp_toks_hi, vocab=id_to_tok, terminated_eos=hit_eos_hi
            )
            r_parse = float(struct.get("r_parse", 0.0))
            beat_valid = float(struct.get("r_rhythm", 0.0))
            r_noloop = float(struct.get("r_noloop", 1.0))
            mean_lp = float(tf_self["tf_mean_logprob"])
            passes_ov39 = bool(
                mean_lp >= OV39_MIN_LOGPROB
                and beat_valid >= OV39_MIN_BEAT_VALID
                and r_parse >= 0.5
                and r_noloop >= 1.0
                and len(pred_hi) >= 24
            )

            real_feats = _extract_real_ink_and_barlines(img)
            mem_final = mem_hi[-1] if isinstance(mem_hi, (tuple, list)) else mem_hi
            real_mem_vec = F.normalize(mem_final.float().mean(dim=1), dim=-1)

            vcs = _compute_vcs_components(
                real_feats=real_feats,
                pred_toks_hi=pred_hi,
                pred_toks_lo=pred_lo,
                real_mem_vec=real_mem_vec,
                model=model,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )

            decoded_pages.append(
                {
                    "piece_id": pid,
                    "mode": mode_key,
                    "img_path": str(img_p),
                    "note_f1": round(note_f1, 4),
                    "mean_logprob": round(mean_lp, 4),
                    "beat_valid_frac": round(beat_valid, 4),
                    "r_parse": round(r_parse, 4),
                    "r_noloop": round(r_noloop, 4),
                    "passes_ov39": passes_ov39,
                    "pred_hi": pred_hi,
                    "pred_lo": pred_lo,
                    "hyp_notes_hi": hyp_notes_hi,
                    "real_feats": real_feats,
                    "real_mem_vec": real_mem_vec.cpu(),
                    "vcs": vcs,
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                }
            )
            print(
                f"[{EXP_ID}][A] {pid}/{mode_key}: F1={note_f1:.3f} VCS={vcs['vcs_total']:.3f} "
                f"(iou={vcs['vcs_ink_iou']:.2f} barX={vcs['vcs_barline_x_corr']:.2f} "
                f"xsc={vcs['vcs_cross_scale_agree']:.2f} cos={vcs['vcs_encoder_patch_cos']:.2f}) "
                f"lp={mean_lp:.2f} ov39={passes_ov39}",
                flush=True,
            )

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            _atomic_write_json(
                exp_dir / "progress.json",
                {"stage": "part_a_decode", "done_hw_pieces": p_idx + 1, "pages": len(decoded_pages)},
            )
            t_last_save = time.time()

    # -----------------------------------------------------------------------
    # Part A Aggregation & ov-39 Hallucination Rejection Benchmark
    # -----------------------------------------------------------------------
    f1_vals = [d["note_f1"] for d in decoded_pages]
    vcs_vals = [d["vcs"]["vcs_total"] for d in decoded_pages]
    lp_vals = [d["mean_logprob"] for d in decoded_pages]
    iou_vals = [d["vcs"]["vcs_ink_iou"] for d in decoded_pages]
    barx_vals = [d["vcs"]["vcs_barline_x_corr"] for d in decoded_pages]
    xsc_vals = [d["vcs"]["vcs_cross_scale_agree"] for d in decoded_pages]
    cos_vals = [d["vcs"]["vcs_encoder_patch_cos"] for d in decoded_pages]

    r_vcs = _pearson_r(vcs_vals, f1_vals)
    rho_vcs = _spearman_rho(vcs_vals, f1_vals)
    r_lp = _pearson_r(lp_vals, f1_vals)

    # Controlled ov-39 RISM-like hallucination negatives: pair each HW image with a fluent
    # high-rhythm prediction from a different HW piece (passes ov-39 fluency, 0 visual grounding)
    fluent_pool = [d for d in decoded_pages if d["passes_ov39"]] or sorted(
        decoded_pages, key=lambda x: x["beat_valid_frac"], reverse=True
    )[: max(4, len(decoded_pages) // 2)]

    halluc_vcs_scores: List[float] = []
    for idx_d, d in enumerate(decoded_pages):
        # Pick a fluent donor from a DIFFERENT piece
        donor = None
        for offset in range(1, len(fluent_pool) + 1):
            cand = fluent_pool[(idx_d + offset) % len(fluent_pool)]
            if cand["piece_id"] != d["piece_id"]:
                donor = cand
                break
        if donor is None:
            continue
        # Score mismatched fluent prediction against this real image
        neg_vcs = _compute_vcs_components(
            real_feats=d["real_feats"],
            pred_toks_hi=donor["pred_hi"],
            pred_toks_lo=d["pred_lo"],  # cross-scale mismatch against real image's 1280px decode
            real_mem_vec=d["real_mem_vec"].to(device),
            model=model,
            device=device,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        halluc_vcs_scores.append(float(neg_vcs["vcs_total"]))

    # Also include natural ov-39 false positives (passes_ov39 & note_f1 < 0.30)
    natural_ov39_fps = [d["vcs"]["vcs_total"] for d in decoded_pages if d["passes_ov39"] and d["note_f1"] < 0.30]
    valid_high_f1 = [d["vcs"]["vcs_total"] for d in decoded_pages if d["note_f1"] >= 0.35]
    if not valid_high_f1:
        # Fallback to top-quartile Note F1 pages if checkpoint HW F1 is below 0.35
        sorted_by_f1 = sorted(decoded_pages, key=lambda x: x["note_f1"], reverse=True)
        valid_high_f1 = [d["vcs"]["vcs_total"] for d in sorted_by_f1[: max(4, len(sorted_by_f1) // 4)]]

    # Calibrate VCS threshold at 20th percentile of valid_high_f1 (keeps >= 80% true positives)
    sorted_valid = sorted(valid_high_f1)
    vcs_thr = sorted_valid[max(0, int(0.20 * (len(sorted_valid) - 1)))]
    all_halluc = halluc_vcs_scores + natural_ov39_fps
    fp_rej_rate = (
        sum(1 for s in all_halluc if s < vcs_thr) / float(max(1, len(all_halluc)))
        if all_halluc
        else 0.0
    )
    synth_fp_rej_rate = (
        sum(1 for s in halluc_vcs_scores if s < vcs_thr) / float(max(1, len(halluc_vcs_scores)))
        if halluc_vcs_scores
        else 0.0
    )
    nat_fp_rej_rate = (
        sum(1 for s in natural_ov39_fps if s < vcs_thr) / float(max(1, len(natural_ov39_fps)))
        if natural_ov39_fps
        else None
    )
    halluc_auroc = _compute_auroc(valid_high_f1, all_halluc)

    part_a_result: Dict[str, Any] = {
        "n_hw_pages": len(decoded_pages),
        "mean_hw_note_f1": round(sum(f1_vals) / max(1, len(f1_vals)), 4),
        "mean_vcs": round(sum(vcs_vals) / max(1, len(vcs_vals)), 4),
        "pearson_r_vcs_vs_note_f1": round(float(r_vcs), 4) if r_vcs is not None else None,
        "spearman_rho_vcs_vs_note_f1": round(float(rho_vcs), 4) if rho_vcs is not None else None,
        "baseline_pearson_r_ov39_logprob_vs_note_f1": round(float(r_lp), 4) if r_lp is not None else None,
        "component_pearson_r": {
            "vcs_ink_iou": round(float(_pearson_r(iou_vals, f1_vals) or 0.0), 4),
            "vcs_barline_x_corr": round(float(_pearson_r(barx_vals, f1_vals) or 0.0), 4),
            "vcs_cross_scale_agree": round(float(_pearson_r(xsc_vals, f1_vals) or 0.0), 4),
            "vcs_encoder_patch_cos": round(float(_pearson_r(cos_vals, f1_vals) or 0.0), 4),
        },
        "ov39_hallucination_benchmark": {
            "calibrated_vcs_threshold": round(float(vcs_thr), 4),
            "n_valid_high_f1": len(valid_high_f1),
            "n_natural_ov39_fps": len(natural_ov39_fps),
            "n_synthetic_ov39_hallucinations": len(halluc_vcs_scores),
            "ov39_hallucination_fp_rejection_rate": round(float(fp_rej_rate), 4),
            "synthetic_fp_rejection_rate": round(float(synth_fp_rej_rate), 4),
            "natural_ov39_fp_rejection_rate": round(float(nat_fp_rej_rate), 4) if nat_fp_rej_rate is not None else None,
            "hallucination_auroc": round(float(halluc_auroc), 4) if halluc_auroc is not None else None,
        },
        "gate_target_pearson_r": GATE_PART_A_PEARSON_R,
        "gate_pass": bool(r_vcs is not None and r_vcs >= GATE_PART_A_PEARSON_R),
        "elapsed_secs": round(time.time() - t_part0, 2),
        "per_page": [
            {
                "piece_id": d["piece_id"],
                "mode": d["mode"],
                "note_f1": d["note_f1"],
                "mean_logprob": d["mean_logprob"],
                "beat_valid_frac": d["beat_valid_frac"],
                "passes_ov39": d["passes_ov39"],
                **d["vcs"],
            }
            for d in decoded_pages
        ],
    }
    _atomic_write_json(part_a_cache, part_a_result)
    print(
        f"[{EXP_ID}] Part A DONE: Pearson r(VCS, F1)={part_a_result['pearson_r_vcs_vs_note_f1']} "
        f"(baseline lp r={part_a_result['baseline_pearson_r_ov39_logprob_vs_note_f1']}) "
        f"FP_rej={fp_rej_rate:.3f} AUROC={halluc_auroc} gate_pass={part_a_result['gate_pass']}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Part B: AR-Free TF-NLL + Cross-Attention Monotonicity DP Alignment Bench
    # -----------------------------------------------------------------------
    t_part_b0 = time.time()
    rng = random.Random(SEED)
    methods = ("free_decoded_align", "tf_nll_only", "tf_nll_plus_ca_mono_dp")
    method_stats: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}

    for d_idx, d in enumerate(decoded_pages):
        pid = d["piece_id"]
        mode_key = d["mode"]
        hdr, true_meas = piece_gt_measures.get(pid, ([], []))
        if len(true_meas) < 2:
            continue

        # Embed true page span [m_a, m_b] inside a simulated multi-page movement stream
        # with realistic preceding & following donor measures from the corpus.
        n_true = len(true_meas)
        n_pre = 4
        n_post = 6
        donor_pool = [m for m in all_donor_measures if m not in true_meas] or all_donor_measures
        pre_meas = [rng.choice(donor_pool) for _ in range(n_pre)]
        post_meas = [rng.choice(donor_pool) for _ in range(n_post)]
        stream = pre_meas + true_meas + post_meas
        m_a = n_pre
        m_b = n_pre + n_true  # exclusive end index

        # Build candidate measure windows [s, e)
        raw_windows = [
            (m_a, m_b),                                     # true span
            (max(0, m_a - 2), max(1, m_b - 2)),             # shift -2
            (max(0, m_a - 1), max(1, m_b - 1)),             # shift -1
            (min(len(stream) - 1, m_a + 1), min(len(stream), m_b + 1)),  # shift +1
            (min(len(stream) - 1, m_a + 2), min(len(stream), m_b + 2)),  # shift +2
            (m_a, max(m_a + 1, m_b - 1)),                   # end -1
            (m_a, min(len(stream), m_b + 1)),               # end +1
            (min(m_b - 1, m_a + 1), m_b),                   # start +1
            (max(0, m_a - 1), m_b),                         # start -1
            (0, min(len(stream), n_true)),                  # off-page early
            (max(0, len(stream) - n_true), len(stream)),    # off-page late
        ]
        # Deduplicate while keeping true span present
        cand_windows: List[Tuple[int, int]] = []
        seen_w = set()
        for w in raw_windows:
            if w[1] > w[0] and w not in seen_w:
                seen_w.add(w)
                cand_windows.append(w)

        # Single encoder pass for this HW page image (AR-free!)
        img = Image.open(d["img_path"]).convert("RGB")
        inp_t = _pil_to_canonical_tensor(img, SCALE_HI_DIM).unsqueeze(0).to(device)
        t_enc0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
                mem_levels, _, router_info = model.encode_image(inp_t, return_aux=False)
        enc_s = time.perf_counter() - t_enc0

        scores_free: List[Tuple[float, Tuple[int, int]]] = []
        scores_nll: List[Tuple[float, Tuple[int, int]]] = []
        scores_dp: List[Tuple[float, Tuple[int, int]]] = []

        t_tf0 = time.perf_counter()
        for ws, we in cand_windows:
            span_meas = stream[ws:we]
            toks_flat: List[str] = list(hdr)
            m_ranges: List[Tuple[int, int]] = []
            # Token offset in inp_ids starts at 1 (after BOS)
            cur_pos = 1 + len(hdr)
            for m_toks in span_meas:
                toks_flat.extend(m_toks)
                m_ranges.append((cur_pos, cur_pos + len(m_toks)))
                cur_pos += len(m_toks)

            # Method 1: Free-decoded Note F1 alignment
            cand_notes = rlvr_grpo.extract_pitch_dur_notes(toks_flat)
            free_f1 = float(rlvr_grpo.compute_sequence_f1(cand_notes, d["hyp_notes_hi"]))
            scores_free.append((free_f1, (ws, we)))

            # Methods 2 & 3: AR-Free Teacher-Forced NLL + Cross-Attention Monotonicity DP
            cand_ids = [bos_id] + [tok_to_id[t] for t in toks_flat if t in tok_to_id] + [eos_id]
            tf_res = _teacher_forced_nll_and_ca_mono(
                model=model,
                mem_levels=mem_levels,
                router_info=router_info,
                tok_ids=cand_ids,
                measure_ranges=m_ranges,
                page_y_span=d["real_feats"]["y_span"],
                pad_id=pad_id,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            scores_nll.append((tf_res["tf_mean_logprob"], (ws, we)))
            scores_dp.append((tf_res["dp_total_score"], (ws, we)))

        tf_total_s = enc_s + (time.perf_counter() - t_tf0)

        for m_name, sc_list, lat in (
            ("free_decoded_align", scores_free, d["elapsed_s"]),
            ("tf_nll_only", scores_nll, tf_total_s),
            ("tf_nll_plus_ca_mono_dp", scores_dp, tf_total_s),
        ):
            best_score, (ps, pe) = max(sc_list, key=lambda x: x[0])
            top1 = 1.0 if (ps == m_a and pe == m_b) else 0.0
            start_ok = 1.0 if ps == m_a else 0.0
            end_ok = 1.0 if pe == m_b else 0.0
            mae = 0.5 * (abs(ps - m_a) + abs(pe - m_b))
            method_stats[m_name].append(
                {
                    "piece_id": pid,
                    "mode": mode_key,
                    "n_cands": len(cand_windows),
                    "true_span": [m_a, m_b],
                    "pred_span": [ps, pe],
                    "top1": top1,
                    "start_exact": start_ok,
                    "end_exact": end_ok,
                    "boundary_mae": mae,
                    "latency_s": round(lat, 3),
                }
            )

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            _atomic_write_json(
                exp_dir / "progress.json",
                {"stage": "part_b_align", "done_pages": d_idx + 1, "total_pages": len(decoded_pages)},
            )
            t_last_save = time.time()

    def _agg_align(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0, "top1_span_acc": None}
        n = float(len(rows))
        return {
            "n": len(rows),
            "top1_span_acc": round(sum(r["top1"] for r in rows) / n, 4),
            "exact_start_acc": round(sum(r["start_exact"] for r in rows) / n, 4),
            "exact_end_acc": round(sum(r["end_exact"] for r in rows) / n, 4),
            "exact_both_boundaries_acc": round(sum(r["top1"] for r in rows) / n, 4),
            "mean_boundary_mae_measures": round(sum(r["boundary_mae"] for r in rows) / n, 4),
            "mean_latency_s": round(sum(r["latency_s"] for r in rows) / n, 3),
        }

    summary_b: Dict[str, Any] = {}
    for m_name, rows in method_stats.items():
        summary_b[m_name] = {
            "all_hw": _agg_align(rows),
            "hw_clean": _agg_align([r for r in rows if r["mode"] == "clean"]),
            "hw_photo": _agg_align([r for r in rows if r["mode"] == "photo"]),
        }

    dp_top1 = summary_b["tf_nll_plus_ca_mono_dp"]["all_hw"]["top1_span_acc"]
    free_top1 = summary_b["free_decoded_align"]["all_hw"]["top1_span_acc"]
    nll_top1 = summary_b["tf_nll_only"]["all_hw"]["top1_span_acc"]

    part_b_result: Dict[str, Any] = {
        "n_hw_pages": len(method_stats["tf_nll_plus_ca_mono_dp"]),
        "dp_weights": {"lambda_mono": DP_LAMBDA_MONO, "lambda_cov": DP_LAMBDA_COV},
        "methods": summary_b,
        "delta_dp_minus_free_top1": (
            round(float(dp_top1) - float(free_top1), 4)
            if (dp_top1 is not None and free_top1 is not None)
            else None
        ),
        "delta_dp_minus_nll_top1": (
            round(float(dp_top1) - float(nll_top1), 4)
            if (dp_top1 is not None and nll_top1 is not None)
            else None
        ),
        "gate_target_top1_acc": GATE_PART_B_TOP1_ACC,
        "gate_pass": bool(dp_top1 is not None and dp_top1 >= GATE_PART_B_TOP1_ACC),
        "elapsed_secs": round(time.time() - t_part_b0, 2),
        "per_page_dp_sample": method_stats["tf_nll_plus_ca_mono_dp"][:16],
    }
    _atomic_write_json(part_b_cache, part_b_result)
    print(
        f"[{EXP_ID}] Part B DONE: free_top1={free_top1} nll_top1={nll_top1} "
        f"dp_top1={dp_top1} gate_pass={part_b_result['gate_pass']}",
        flush=True,
    )
    return part_a_result, part_b_result


def main() -> None:
    t_start = time.time()
    started_iso = _utc_now()
    hf_token = _read_and_purge_hf_token()
    try:
        _setup_environment()
        import torch
        from huggingface_hub import HfApi
        import eval_uni_suite31 as eus

        random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            cc = torch.cuda.get_device_capability(0)
            amp_dtype: Optional[torch.dtype] = torch.bfloat16 if cc[0] >= 8 else torch.float16
            amp_enabled = True
            gpu_name = torch.cuda.get_device_name(0)
        else:
            amp_dtype = None
            amp_enabled = False
            gpu_name = "cpu"

        ckpt_local, ckpt_source, manifest_path = _ensure_assets(hf_token)
        model, ckpt_meta = _load_frozen_model(ckpt_local, device)
        _, pieces = eus.resolve_manifest_pieces(manifest_path)

        sanity = _run_2page_sanity_gate(model, pieces, device, amp_dtype, amp_enabled)
        part_a, part_b = _run_part_a_and_b(model, pieces, device, amp_dtype, amp_enabled)

        report = {
            "exp_id": EXP_ID,
            "exp_code": EXP_CODE,
            "status": "completed",
            "started_utc": started_iso,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"colab-{gpu_name}",
            "amp_dtype": str(amp_dtype) if amp_dtype is not None else "fp32",
            "checkpoint": {
                "source": ckpt_source,
                "arch": ckpt_meta.get("arch"),
                "step": ckpt_meta.get("step"),
                "missing_keys": len(ckpt_meta.get("missing_keys") or []),
            },
            "sanity_gate_2p": sanity,
            "part_a_vcs_verifier_corr": part_a,
            "part_b_tf_nll_dp_align_bench": part_b,
            "gates": {
                "part_a_pearson_r_ge_070": part_a["gate_pass"],
                "part_b_top1_ge_085": part_b["gate_pass"],
                "both_passed": bool(part_a["gate_pass"] and part_b["gate_pass"]),
            },
        }

        eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        run_local = VOL_ROOT / "runs" / f"{EXP_ID}.json"
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
                "eval_path": HF_EVAL_PATH,
                "gates": report["gates"],
                "part_a_pearson_r": part_a["pearson_r_vcs_vs_note_f1"],
                "part_b_dp_top1": part_b["methods"]["tf_nll_plus_ca_mono_dp"]["all_hw"]["top1_span_acc"],
            },
        )

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload VCS verifier + AR-free TF-NLL DP align bench",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload i15-02 run ledger",
        )
        print(f"[{EXP_ID}] UPLOADED {HF_EVAL_PATH} & {HF_RUN_PATH} TO HF!", flush=True)
    finally:
        gc.collect()
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
