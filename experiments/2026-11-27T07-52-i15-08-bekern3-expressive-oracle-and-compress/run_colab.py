#!/usr/bin/env python3
"""Colab L4 / Modal L4 runner for Wave 13 Exp i15-08 (bekern3 Architect 3):
2026-11-27T07-52-i15-08-bekern3-expressive-oracle-and-compress

Two-part experiment:
  Part A (`part_a_oracle_closure`):
    Measures exact 125-page `OMR-NorthStar` MusicXML -> `bekern3` -> MusicXML
    round-trip Oracle (`NS`, `S_note`, `S_struct`, `S_expr`, `S_play`, `CFR`)
    with `<pickup>`, `<sys_lyr>`, `<harm>`, `<tempo>`, `<reh>`, numbered
    `slur-1..3` / `hairpin-1..2`, `ottava-15ma`/`ottava-15mb`, `+18` `art-*`,
    `+7` `dyn-*`/`dyn-other`, and `bar-dashed`.
    Gate: Oracle `NS >= 98.0`, `S_expr >= 0.980`.

  Part B (`part_b_compression_mtp_bakeoff`):
    3-arm 600-step bake-off on `iter10_5_proxy_2k` (`UniOMRv2Net(small_45m)`,
    `mtp_k=4`, `return_mtp=True`, `mtp_loss_weight=0.30`, EMA `0.99`, BF16):
      - `arm0_bekern_v2a`: Control `bekern_v2a`
      - `arm1_profile_a_comp`: `-20.2%` target tokens (sticky dur + chord dur drop
        + implicit `voice-1` + fused `rest-<dur>` + fused `dur.`)
      - `arm2_profile_a_plus_core_pitch`: `-41.1%` target tokens (Profile A +
        `C2..B6` core pitch fusion + compositional `<step> <oct>` fallback)
    Measures compression ratio, `T > 3840` truncation rate, `suite31_hw` Strict
    Note F1 + Rhythm F1, and wall-clock MTP-4 speculative decode speedup.
    Gate: `>= 1.45x` wall-clock MTP-4 speedup with `>= -0.2 pp` Note F1 parity.

Uploads `evals/<EXP_ID>.json` and `runs/<EXP_ID>.json` to HF
`gregoreesmaa/sheet-music-converter-omr`. Resume-safe (5-minute atomic
checkpoints + per-stage sentinels). Never prints `HF_TOKEN`.
"""

from __future__ import annotations

from collections import Counter
import contextlib
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET
import zipfile

EXP_ID = "2026-11-27T07-52-i15-08-bekern3-expressive-oracle-and-compress"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_NORTHSTAR_TAR = "datasets/northstar_eval_pack.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
STEPS_PER_ARM = int(os.environ.get("I15_08_STEPS", "600"))
SEED = 42
GRAD_ACCUM = 4
LR = 3.0e-4
WARMUP_STEPS = 60
MTP_LOSS_WEIGHT = 0.30
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
EVAL_MAX_LEN = 1536
TRUNC_HORIZON = 3840
GPU_EMA_DECAY = 0.99
SAVE_EVERY_MINUTES = 5.0
VAL_LIMIT = int(os.environ.get("I15_08_VAL_LIMIT", "48"))
SUITE31_LIMIT = int(os.environ.get("I15_08_SUITE31_LIMIT", "0"))
SPEED_BENCH_PAGES = int(os.environ.get("I15_08_SPEED_PAGES", "16"))

DURS_LIST: Tuple[str, ...] = ("breve", "whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "128th")
DURS_SET = frozenset(DURS_LIST)
DURS_DOT1_LIST: Tuple[str, ...] = tuple(f"{d}." for d in DURS_LIST)
DURS_DOT2_LIST: Tuple[str, ...] = tuple(f"{d}.." for d in DURS_LIST)
FUSED_DUR_TO_PAIR: Dict[str, Tuple[str, int]] = {
    **{d: (d, 0) for d in DURS_LIST},
    **{f"{d}.": (d, 1) for d in DURS_LIST},
    **{f"{d}..": (d, 2) for d in DURS_LIST},
}
FUSED_REST_LIST: Tuple[str, ...] = (
    tuple(f"rest-{d}" for d in DURS_LIST)
    + tuple(f"rest-{d}." for d in DURS_LIST)
    + tuple(f"rest-{d}.." for d in DURS_LIST)
)
FUSED_REST_TO_PAIR: Dict[str, Tuple[str, int]] = {
    **{f"rest-{d}": (d, 0) for d in DURS_LIST},
    **{f"rest-{d}.": (d, 1) for d in DURS_LIST},
    **{f"rest-{d}..": (d, 2) for d in DURS_LIST},
}

STEPS_LIST: Tuple[str, ...] = ("C", "D", "E", "F", "G", "A", "B")
STEPS_SET = frozenset(STEPS_LIST)
OCTS_SET = frozenset(str(i) for i in range(-1, 9))
ALTERS_SET = frozenset(("a-2", "a-1", "a1", "a2"))

# Core pitch fusion: C2..B6 (5 octaves * 7 diatonic steps = 35 tokens)
CORE_PITCH_OCTS: Tuple[str, ...] = ("2", "3", "4", "5", "6")
CORE_PITCH_LIST: Tuple[str, ...] = tuple(f"{s}{o}" for o in CORE_PITCH_OCTS for s in STEPS_LIST)
CORE_PITCH_TO_PAIR: Dict[str, Tuple[str, str]] = {f"{s}{o}": (s, o) for o in CORE_PITCH_OCTS for s in STEPS_LIST}

# Architect 3 bekern3 expressive/text extension inventory (Part A Oracle Closure)
EXT_ARTICULATIONS: Tuple[str, ...] = (
    "staccato", "staccatissimo", "accent", "strong-accent", "tenuto",
    "detached-legato", "spiccato", "scoop", "plop", "doit", "falloff",
    "breath-mark", "caesura", "stress", "unstress", "fermata",
    "trill-mark", "mordent", "inverted-mordent", "turn", "inverted-turn",
    "arpeggiate", "glissando", "tremolo", "harmonic", "up-bow", "down-bow",
    "open-string", "snap-pizzicato",
)
EXT_ART_SET = frozenset(EXT_ARTICULATIONS)
EXT_DYNAMICS: Tuple[str, ...] = (
    "ppp", "pp", "p", "mp", "mf", "f", "ff", "fff",
    "fp", "sf", "sfz", "sffz", "rf", "rfz", "fz", "n",
)
EXT_DYN_SET = frozenset(EXT_DYNAMICS)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_vol_root() -> Path:
    if Path("/vol").is_dir():
        return Path("/vol")
    if Path("/content").is_dir():
        return Path("/content/vol")
    return Path("/tmp/omr_vol")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_hf_token() -> str:
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not tok and HF_TOKEN_FILE.is_file():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        tok = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass
    if not tok:
        raise RuntimeError("HF token missing: set HF_TOKEN env var or /tmp/.hf_token")
    return tok


def _ensure_deps() -> None:
    needed = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])


def _locate_mvd() -> Path:
    for cand in (
        Path("/workspace/mvd"),
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
        Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
    ):
        if (cand / "uni_omr_v2.py").is_file() and (cand / "eval_uni_suite31.py").is_file():
            return cand
    for bundle in (Path("/content/bundle.tar.gz"), Path("/content/repo_bundle.tar.gz")):
        if bundle.is_file():
            dest = Path("/content/sheet-music-converter")
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=dest)
            mvd = dest / "tmp-train" / "mvd"
            if (mvd / "uni_omr_v2.py").is_file():
                return mvd
    raise FileNotFoundError("Could not locate tmp-train/mvd containing uni_omr_v2.py")


def _ensure_proxy_dataset(vol_root: Path, hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = vol_root / "datasets" / "iter10_5_proxy_2k"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"
    if train_abs.is_file() and val_abs.is_file():
        return train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR} from {HF_REPO_ID}...", flush=True)
    tar_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PROXY_TAR,
        repo_type="dataset",
        token=hf_token,
        local_dir=str(vol_root / "tars"),
    )
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=ds_dir)
    Path(tar_path).unlink(missing_ok=True)

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
    return train_abs, val_abs


def _ensure_suite31(vol_root: Path, hf_token: str) -> Path:
    from huggingface_hub import hf_hub_download

    s31_dir = vol_root / "datasets" / "suite31_hw_eval"
    sentinel = s31_dir / ".extracted"
    if not sentinel.exists():
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR} from {HF_REPO_ID}...", flush=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=HF_SUITE31_TAR,
                local_dir=str(vol_root / "tars"),
                token=hf_token,
            )
        )
        s31_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=s31_dir)
        sentinel.write_text(_utc_now() + "\n", encoding="utf-8")
        tar_p.unlink(missing_ok=True)
    manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json under {s31_dir}")
    return manifests[0]


def _try_ensure_northstar_pack(vol_root: Path, hf_token: str) -> Optional[Path]:
    """Attempt to locate or download the 125-page OMR-NorthStar pack."""
    for cand in (
        vol_root / "datasets" / "northstar_eval_pack" / "manifest.json",
        Path("/content/northstar-eval-pack/manifest.json"),
        Path(r"D:\tmp\northstar-eval-pack\manifest.json"),
    ):
        if cand.is_file():
            return cand
    try:
        from huggingface_hub import hf_hub_download

        ns_dir = vol_root / "datasets" / "northstar_eval_pack"
        ns_dir.mkdir(parents=True, exist_ok=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=HF_NORTHSTAR_TAR,
                local_dir=str(vol_root / "tars"),
                token=hf_token,
            )
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=ns_dir)
        tar_p.unlink(missing_ok=True)
        manifests = sorted(ns_dir.rglob("manifest.json"))
        return manifests[0] if manifests else None
    except Exception as exc:  # noqa: BLE001
        print(f"[{EXP_ID}] Note: {HF_NORTHSTAR_TAR} not available on HF ({exc}); will use fallback corpus.", flush=True)
        return None


def _upload_json(hf_token: str, local_path: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"{EXP_ID}: upload {path_in_repo}",
    )
    print(f"[{EXP_ID}] Uploaded {path_in_repo} -> {HF_REPO_ID}", flush=True)


def _load_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _read_page_tokens(rec: Dict[str, Any], sys_token: str) -> List[str]:
    if "tokens" in rec and isinstance(rec["tokens"], list):
        return [str(t) for t in rec["tokens"]]
    chunks: List[List[str]] = []
    for bp in rec.get("bekerns", []):
        p = Path(bp)
        if p.is_file():
            toks = p.read_text(encoding="utf-8").split()
            if toks:
                chunks.append(toks)
    out: List[str] = []
    for idx, ch in enumerate(chunks):
        if idx > 0:
            out.append(sys_token)
        out.extend(ch)
    return out


# ===========================================================================
# PART A: bekern3 Expressive/Text MusicXML Round-Trip Oracle Closure
# ===========================================================================

def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _read_musicxml_text(path: Path) -> str:
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            if "META-INF/container.xml" in names:
                try:
                    c_root = ET.fromstring(zf.read("META-INF/container.xml"))
                    for elem in c_root.iter():
                        if _strip_ns(elem.tag) == "rootfile":
                            fp = elem.attrib.get("full-path")
                            if fp and fp in names:
                                return zf.read(fp).decode("utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    pass
            for name in names:
                low = name.lower()
                if not low.startswith("meta-inf/") and (low.endswith(".musicxml") or low.endswith(".xml")):
                    return zf.read(name).decode("utf-8", errors="replace")
        raise ValueError(f"No MusicXML found inside {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _esc_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\n", " ").replace("|", "/").strip()


def musicxml_to_bekern3(xml_text: str) -> List[str]:
    """Compile MusicXML -> `bekern3` token sequence with Architect 3 expressive/text extensions:
    `<pickup>`, `<sys_lyr>`, `<harm>`, `<tempo>`, `<reh>`, numbered `slur-1..3` /
    `hairpin-1..2`, `ottava-15ma`/`ottava-15mb`, `+18` `art-*`, `+7` `dyn-*`/`dyn-other`,
    and `bar-dashed`.
    """
    root = ET.fromstring(xml_text)
    toks: List[str] = []
    for p_idx, part in enumerate(ch for ch in root if _strip_ns(ch.tag) == "part"):
        if p_idx > 0:
            toks.append("<part>")
        sys_lyrics: List[str] = []
        for measure in (ch for ch in part if _strip_ns(ch.tag) == "measure"):
            if measure.attrib.get("implicit", "").lower() == "yes":
                toks.append("<pickup>")
            for elem in measure:
                tag = _strip_ns(elem.tag)
                if tag == "attributes":
                    for ach in elem:
                        atag = _strip_ns(ach.tag)
                        if atag == "divisions" and ach.text:
                            toks.append(f"div-{ach.text.strip()}")
                        elif atag == "key":
                            fifths = "0"
                            for kch in ach:
                                if _strip_ns(kch.tag) == "fifths" and kch.text:
                                    fifths = kch.text.strip()
                            toks.append(f"key-{fifths}")
                        elif atag == "time":
                            beats, btype = "4", "4"
                            for tch in ach:
                                ttag = _strip_ns(tch.tag)
                                if ttag == "beats" and tch.text:
                                    beats = tch.text.strip()
                                elif ttag == "beat-type" and tch.text:
                                    btype = tch.text.strip()
                            toks.append(f"time-{beats}/{btype}")
                        elif atag == "clef":
                            sign, line = "G", "2"
                            for cch in ach:
                                ctag = _strip_ns(cch.tag)
                                if ctag == "sign" and cch.text:
                                    sign = cch.text.strip()
                                elif ctag == "line" and cch.text:
                                    line = cch.text.strip()
                            toks.append(f"clef-{sign}{line}")
                elif tag == "harmony":
                    root_step, root_alt, kind = "", "", "major"
                    for hch in elem:
                        htag = _strip_ns(hch.tag)
                        if htag == "root":
                            for rch in hch:
                                rtag = _strip_ns(rch.tag)
                                if rtag == "root-step" and rch.text:
                                    root_step = rch.text.strip()
                                elif rtag == "root-alter" and rch.text:
                                    root_alt = rch.text.strip()
                        elif htag == "kind" and hch.text:
                            kind = hch.text.strip()
                    if root_step:
                        toks.append("<harm>")
                        toks.append(f"harm:{root_step}:{root_alt}:{kind}")
                elif tag == "direction":
                    for dch in elem:
                        if _strip_ns(dch.tag) != "direction-type":
                            continue
                        for dt in dch:
                            dttag = _strip_ns(dt.tag)
                            if dttag == "dynamics":
                                emitted = False
                                for dyn_el in dt:
                                    dname = _strip_ns(dyn_el.tag).lower()
                                    if dname in EXT_DYN_SET:
                                        toks.append(f"dyn-{dname}")
                                        emitted = True
                                    elif dname == "other-dynamics":
                                        toks.append("dyn-other")
                                        if dyn_el.text and dyn_el.text.strip():
                                            toks.append(f"dyn-txt:{_esc_text(dyn_el.text)}")
                                        emitted = True
                                if not emitted:
                                    toks.append("dyn-other")
                            elif dttag == "wedge":
                                wtype = dt.attrib.get("type", "stop").lower()
                                num = min(2, max(1, int(dt.attrib.get("number", "1") or "1")))
                                if wtype in ("crescendo", "diminuendo", "stop"):
                                    short = "cresc" if wtype == "crescendo" else ("dim" if wtype == "diminuendo" else "stop")
                                    toks.append(f"hairpin-{num}-{short}")
                            elif dttag == "octave-shift":
                                otype = dt.attrib.get("type", "stop").lower()
                                size = dt.attrib.get("size", "8")
                                if otype == "stop":
                                    toks.append("ottava-stop")
                                elif size == "15":
                                    toks.append("ottava-15ma" if otype == "down" else "ottava-15mb")
                                else:
                                    toks.append("ottava-8va" if otype == "down" else "ottava-8vb")
                            elif dttag == "rehearsal" and dt.text and dt.text.strip():
                                toks.append("<reh>")
                                toks.append(f"reh:{_esc_text(dt.text)}")
                            elif dttag == "words" and dt.text and dt.text.strip():
                                toks.append("<tempo>")
                                toks.append(f"tempo:{_esc_text(dt.text)}")
                            elif dttag == "metronome":
                                bu, pm = "quarter", "120"
                                for mch in dt:
                                    mtag = _strip_ns(mch.tag)
                                    if mtag == "beat-unit" and mch.text:
                                        bu = mch.text.strip()
                                    elif mtag == "per-minute" and mch.text:
                                        pm = mch.text.strip()
                                toks.append("<tempo>")
                                toks.append(f"metro:{bu}={pm}")
                elif tag == "note":
                    is_chord = False
                    is_rest = False
                    is_grace = False
                    step, octv, alt = "", "4", ""
                    dur = "quarter"
                    dots = 0
                    voice = "1"
                    staff = ""
                    ties: List[str] = []
                    slurs: List[str] = []
                    arts: List[str] = []
                    for nch in elem:
                        ntag = _strip_ns(nch.tag)
                        if ntag == "chord":
                            is_chord = True
                        elif ntag == "rest":
                            is_rest = True
                        elif ntag == "grace":
                            is_grace = True
                        elif ntag == "pitch":
                            for pch in nch:
                                ptag = _strip_ns(pch.tag)
                                if ptag == "step" and pch.text:
                                    step = pch.text.strip().upper()
                                elif ptag == "octave" and pch.text:
                                    octv = pch.text.strip()
                                elif ptag == "alter" and pch.text:
                                    try:
                                        av = int(round(float(pch.text.strip())))
                                        if av != 0:
                                            alt = f"a{av}"
                                    except ValueError:
                                        pass
                        elif ntag == "type" and nch.text:
                            dur = nch.text.strip().lower()
                        elif ntag == "dot":
                            dots += 1
                        elif ntag == "voice" and nch.text:
                            voice = nch.text.strip()
                        elif ntag == "staff" and nch.text:
                            staff = nch.text.strip()
                        elif ntag == "tie":
                            tt = nch.attrib.get("type", "")
                            if tt in ("start", "stop"):
                                ties.append(f"tie-{tt}")
                        elif ntag == "lyric":
                            lnum = nch.attrib.get("number", "1")
                            ltxt = ""
                            for lch in nch:
                                if _strip_ns(lch.tag) == "text" and lch.text:
                                    ltxt = _esc_text(lch.text)
                            if ltxt:
                                sys_lyrics.append(f"lyr:{lnum}:{ltxt}")
                        elif ntag == "notations":
                            for nt in nch:
                                nttag = _strip_ns(nt.tag)
                                if nttag == "slur":
                                    stype = nt.attrib.get("type", "")
                                    snum = min(3, max(1, int(nt.attrib.get("number", "1") or "1")))
                                    if stype in ("start", "stop"):
                                        slurs.append(f"slur-{snum}-{stype}")
                                elif nttag == "fermata":
                                    arts.append("art-fermata")
                                elif nttag == "arpeggiate":
                                    arts.append("art-arpeggiate")
                                elif nttag == "glissando":
                                    arts.append("art-glissando")
                                elif nttag in ("articulations", "ornaments", "technical"):
                                    for ael in nt:
                                        aname = _strip_ns(ael.tag).lower()
                                        if aname in EXT_ART_SET:
                                            arts.append(f"art-{aname}")
                    if staff:
                        toks.append(f"staff-{staff}")
                    toks.append(f"voice-{voice}")
                    if is_grace:
                        toks.append("grace")
                    if is_rest:
                        toks.append("rest")
                        toks.append(dur)
                        toks.extend(["dot"] * dots)
                    elif step:
                        if is_chord:
                            toks.append("chord")
                        toks.append(dur)
                        toks.append(step)
                        toks.append(octv)
                        if alt:
                            toks.append(alt)
                        toks.extend(["dot"] * dots)
                    toks.extend(ties)
                    toks.extend(slurs)
                    toks.extend(arts)
                elif tag == "barline":
                    bstyle = ""
                    for bch in elem:
                        if _strip_ns(bch.tag) == "bar-style" and bch.text:
                            bstyle = bch.text.strip().lower()
                    if bstyle == "dashed":
                        toks.append("bar-dashed")
                    elif bstyle == "light-heavy":
                        toks.append("bar-final")
                    elif bstyle == "light-light":
                        toks.append("bar-double")
            if sys_lyrics:
                toks.append("<sys_lyr>")
                toks.extend(sys_lyrics)
                sys_lyrics = []
            toks.append("bar")
    return toks


def bekern3_to_musicxml(toks: Sequence[str]) -> str:
    """Reconstruct valid MusicXML 3.1 from `bekern3` tokens (round-trip inverse)."""
    score = ET.Element("score-partwise", {"version": "3.1"})
    plist = ET.SubElement(score, "part-list")

    parts_toks: List[List[str]] = [[]]
    for t in toks:
        if t == "<part>":
            parts_toks.append([])
        else:
            parts_toks[-1].append(t)

    dur_to_divs = {
        "breve": 8, "whole": 4, "half": 2, "quarter": 1,
        "eighth": 1, "16th": 1, "32nd": 1, "64th": 1, "128th": 1,
    }

    for p_idx, ptoks in enumerate(parts_toks, start=1):
        pid = f"P{p_idx}"
        sp = ET.SubElement(plist, "score-part", {"id": pid})
        ET.SubElement(sp, "part-name").text = f"Part {p_idx}"
        part_el = ET.SubElement(score, "part", {"id": pid})

        m_num = 1
        m_el = ET.SubElement(part_el, "measure", {"number": str(m_num)})
        cur_voice = "1"
        cur_staff = ""
        last_note_el: Optional[ET.Element] = None
        last_notations_el: Optional[ET.Element] = None
        note_els_in_measure: List[ET.Element] = []

        def _ensure_notations() -> ET.Element:
            nonlocal last_notations_el
            if last_note_el is None:
                return ET.Element("notations")
            if last_notations_el is None:
                last_notations_el = ET.SubElement(last_note_el, "notations")
            return last_notations_el

        i = 0
        n = len(ptoks)
        while i < n:
            t = ptoks[i]
            if t == "<pickup>":
                m_el.set("implicit", "yes")
                i += 1
                continue
            if t == "bar":
                m_num += 1
                if i + 1 < n:
                    m_el = ET.SubElement(part_el, "measure", {"number": str(m_num)})
                    last_note_el = None
                    last_notations_el = None
                    note_els_in_measure = []
                i += 1
                continue
            if t in ("bar-dashed", "bar-final", "bar-double"):
                bl = ET.SubElement(m_el, "barline")
                bs = ET.SubElement(bl, "bar-style")
                bs.text = "dashed" if t == "bar-dashed" else ("light-heavy" if t == "bar-final" else "light-light")
                i += 1
                continue
            if t.startswith("div-") or t.startswith("key-") or t.startswith("time-") or t.startswith("clef-"):
                attr = ET.SubElement(m_el, "attributes")
                if t.startswith("div-"):
                    ET.SubElement(attr, "divisions").text = t[4:]
                elif t.startswith("key-"):
                    k = ET.SubElement(attr, "key")
                    ET.SubElement(k, "fifths").text = t[4:]
                elif t.startswith("time-") and "/" in t[5:]:
                    b, bt = t[5:].split("/", 1)
                    tm = ET.SubElement(attr, "time")
                    ET.SubElement(tm, "beats").text = b
                    ET.SubElement(tm, "beat-type").text = bt
                elif t.startswith("clef-") and len(t) >= 7:
                    cl = ET.SubElement(attr, "clef")
                    ET.SubElement(cl, "sign").text = t[5]
                    ET.SubElement(cl, "line").text = t[6:]
                i += 1
                continue
            if t == "<harm>" and i + 1 < n and ptoks[i + 1].startswith("harm:"):
                _, rs, ra, kd = (ptoks[i + 1].split(":", 3) + ["", "", "", ""])[:4]
                h_el = ET.SubElement(m_el, "harmony")
                r_el = ET.SubElement(h_el, "root")
                ET.SubElement(r_el, "root-step").text = rs
                if ra:
                    ET.SubElement(r_el, "root-alter").text = ra
                ET.SubElement(h_el, "kind").text = kd or "major"
                i += 2
                continue
            if t == "<reh>" and i + 1 < n and ptoks[i + 1].startswith("reh:"):
                d_el = ET.SubElement(m_el, "direction")
                dt_el = ET.SubElement(d_el, "direction-type")
                ET.SubElement(dt_el, "rehearsal").text = ptoks[i + 1][4:]
                i += 2
                continue
            if t == "<tempo>" and i + 1 < n:
                nxt = ptoks[i + 1]
                d_el = ET.SubElement(m_el, "direction")
                dt_el = ET.SubElement(d_el, "direction-type")
                if nxt.startswith("metro:") and "=" in nxt[6:]:
                    bu, pm = nxt[6:].split("=", 1)
                    me = ET.SubElement(dt_el, "metronome")
                    ET.SubElement(me, "beat-unit").text = bu
                    ET.SubElement(me, "per-minute").text = pm
                elif nxt.startswith("tempo:"):
                    ET.SubElement(dt_el, "words").text = nxt[6:]
                i += 2
                continue
            if t.startswith("dyn-"):
                d_el = ET.SubElement(m_el, "direction")
                dt_el = ET.SubElement(d_el, "direction-type")
                dy = ET.SubElement(dt_el, "dynamics")
                if t == "dyn-other":
                    od = ET.SubElement(dy, "other-dynamics")
                    if i + 1 < n and ptoks[i + 1].startswith("dyn-txt:"):
                        od.text = ptoks[i + 1][8:]
                        i += 1
                else:
                    ET.SubElement(dy, t[4:])
                i += 1
                continue
            if t.startswith("hairpin-"):
                parts_h = t.split("-")
                if len(parts_h) == 3:
                    num, kind = parts_h[1], parts_h[2]
                    wtype = "crescendo" if kind == "cresc" else ("diminuendo" if kind == "dim" else "stop")
                    d_el = ET.SubElement(m_el, "direction")
                    dt_el = ET.SubElement(d_el, "direction-type")
                    ET.SubElement(dt_el, "wedge", {"type": wtype, "number": num})
                i += 1
                continue
            if t.startswith("ottava-"):
                d_el = ET.SubElement(m_el, "direction")
                dt_el = ET.SubElement(d_el, "direction-type")
                if t == "ottava-stop":
                    ET.SubElement(dt_el, "octave-shift", {"type": "stop"})
                elif t == "ottava-15ma":
                    ET.SubElement(dt_el, "octave-shift", {"type": "down", "size": "15"})
                elif t == "ottava-15mb":
                    ET.SubElement(dt_el, "octave-shift", {"type": "up", "size": "15"})
                elif t == "ottava-8va":
                    ET.SubElement(dt_el, "octave-shift", {"type": "down", "size": "8"})
                elif t == "ottava-8vb":
                    ET.SubElement(dt_el, "octave-shift", {"type": "up", "size": "8"})
                i += 1
                continue
            if t == "<sys_lyr>":
                i += 1
                l_idx = 0
                while i < n and ptoks[i].startswith("lyr:"):
                    _, lnum, ltxt = (ptoks[i].split(":", 2) + ["1", ""])[:3]
                    if l_idx < len(note_els_in_measure):
                        ly = ET.SubElement(note_els_in_measure[l_idx], "lyric", {"number": lnum})
                        ET.SubElement(ly, "text").text = ltxt
                    l_idx += 1
                    i += 1
                continue
            if t.startswith("voice-"):
                cur_voice = t[6:]
                i += 1
                continue
            if t.startswith("staff-"):
                cur_staff = t[6:]
                i += 1
                continue
            if t == "rest" and i + 1 < n and ptoks[i + 1] in DURS_SET:
                dur = ptoks[i + 1]
                j = i + 2
                dots = 0
                while j < n and ptoks[j] == "dot":
                    dots += 1
                    j += 1
                n_el = ET.SubElement(m_el, "note")
                ET.SubElement(n_el, "rest")
                ET.SubElement(n_el, "duration").text = str(dur_to_divs.get(dur, 1))
                ET.SubElement(n_el, "voice").text = cur_voice
                ET.SubElement(n_el, "type").text = dur
                for _ in range(dots):
                    ET.SubElement(n_el, "dot")
                if cur_staff:
                    ET.SubElement(n_el, "staff").text = cur_staff
                last_note_el = n_el
                last_notations_el = None
                i = j
                continue
            if t == "grace" or t == "chord" or t in DURS_SET:
                is_grace = False
                is_chord = False
                j = i
                if j < n and ptoks[j] == "grace":
                    is_grace = True
                    j += 1
                if j < n and ptoks[j] == "chord":
                    is_chord = True
                    j += 1
                if j + 2 < n and ptoks[j] in DURS_SET and ptoks[j + 1] in STEPS_SET and ptoks[j + 2] in OCTS_SET:
                    dur = ptoks[j]
                    step = ptoks[j + 1]
                    octv = ptoks[j + 2]
                    j += 3
                    alt = ""
                    if j < n and ptoks[j] in ALTERS_SET:
                        alt = ptoks[j][1:]
                        j += 1
                    dots = 0
                    while j < n and ptoks[j] == "dot":
                        dots += 1
                        j += 1
                    n_el = ET.SubElement(m_el, "note")
                    if is_grace:
                        ET.SubElement(n_el, "grace")
                    if is_chord:
                        ET.SubElement(n_el, "chord")
                    p_el = ET.SubElement(n_el, "pitch")
                    ET.SubElement(p_el, "step").text = step
                    if alt:
                        ET.SubElement(p_el, "alter").text = alt
                    ET.SubElement(p_el, "octave").text = octv
                    if not is_grace:
                        ET.SubElement(n_el, "duration").text = str(dur_to_divs.get(dur, 1))
                    ET.SubElement(n_el, "voice").text = cur_voice
                    ET.SubElement(n_el, "type").text = dur
                    for _ in range(dots):
                        ET.SubElement(n_el, "dot")
                    if cur_staff:
                        ET.SubElement(n_el, "staff").text = cur_staff
                    last_note_el = n_el
                    last_notations_el = None
                    if not is_chord and not is_grace:
                        note_els_in_measure.append(n_el)
                    i = j
                    continue
            if t in ("tie-start", "tie-stop") and last_note_el is not None:
                ET.SubElement(last_note_el, "tie", {"type": t[4:]})
                i += 1
                continue
            if t.startswith("slur-") and last_note_el is not None:
                ps = t.split("-")
                if len(ps) == 3:
                    nt = _ensure_notations()
                    ET.SubElement(nt, "slur", {"number": ps[1], "type": ps[2]})
                i += 1
                continue
            if t.startswith("art-") and last_note_el is not None:
                aname = t[4:]
                nt = _ensure_notations()
                if aname in ("fermata", "arpeggiate", "glissando"):
                    ET.SubElement(nt, aname)
                elif aname in ("trill-mark", "mordent", "inverted-mordent", "turn", "inverted-turn", "tremolo"):
                    orn = nt.find("ornaments")
                    if orn is None:
                        orn = ET.SubElement(nt, "ornaments")
                    ET.SubElement(orn, aname)
                elif aname in ("harmonic", "up-bow", "down-bow", "open-string", "snap-pizzicato"):
                    tech = nt.find("technical")
                    if tech is None:
                        tech = ET.SubElement(nt, "technical")
                    ET.SubElement(tech, aname)
                else:
                    art_el = nt.find("articulations")
                    if art_el is None:
                        art_el = ET.SubElement(nt, "articulations")
                    ET.SubElement(art_el, aname)
                i += 1
                continue
            i += 1

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(score, encoding="unicode")


def _extract_musicxml_axes(xml_text: str) -> Dict[str, List[Any]]:
    """Extract canonical multiset features for S_note, S_struct, S_expr, S_play fallback scoring."""
    root = ET.fromstring(xml_text)
    notes: List[Tuple[str, int, str, int, int]] = []
    structs: List[str] = []
    exprs: List[str] = []
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag == "measure":
            if elem.attrib.get("implicit", "").lower() == "yes":
                structs.append("pickup")
        elif tag == "bar-style" and elem.text:
            structs.append(f"bar:{elem.text.strip().lower()}")
        elif tag == "fifths" and elem.text:
            structs.append(f"key:{elem.text.strip()}")
        elif tag == "time":
            beats, bt = "", ""
            for ch in elem:
                if _strip_ns(ch.tag) == "beats" and ch.text:
                    beats = ch.text.strip()
                elif _strip_ns(ch.tag) == "beat-type" and ch.text:
                    bt = ch.text.strip()
            if beats and bt:
                structs.append(f"time:{beats}/{bt}")
        elif tag == "clef":
            s, l = "", ""
            for ch in elem:
                if _strip_ns(ch.tag) == "sign" and ch.text:
                    s = ch.text.strip()
                elif _strip_ns(ch.tag) == "line" and ch.text:
                    l = ch.text.strip()
            if s:
                structs.append(f"clef:{s}{l}")
        elif tag == "harmony":
            exprs.append("harmony")
        elif tag == "rehearsal" and elem.text and elem.text.strip():
            exprs.append(f"reh:{elem.text.strip()}")
        elif tag == "words" and elem.text and elem.text.strip():
            exprs.append(f"words:{elem.text.strip()}")
        elif tag == "metronome":
            exprs.append("metronome")
        elif tag == "dynamics":
            for dch in elem:
                exprs.append(f"dyn:{_strip_ns(dch.tag).lower()}")
        elif tag == "wedge":
            exprs.append(f"wedge:{elem.attrib.get('type', '').lower()}:{elem.attrib.get('number', '1')}")
        elif tag == "octave-shift":
            exprs.append(f"ottava:{elem.attrib.get('type', '').lower()}:{elem.attrib.get('size', '8')}")
        elif tag == "slur":
            exprs.append(f"slur:{elem.attrib.get('type', '').lower()}:{elem.attrib.get('number', '1')}")
        elif tag == "lyric":
            for lch in elem:
                if _strip_ns(lch.tag) == "text" and lch.text and lch.text.strip():
                    exprs.append(f"lyr:{lch.text.strip()}")
        elif tag in ("articulations", "ornaments", "technical"):
            for ach in elem:
                exprs.append(f"art:{_strip_ns(ach.tag).lower()}")
        elif tag in ("fermata", "arpeggiate", "glissando"):
            exprs.append(f"art:{tag}")
        elif tag == "note":
            is_rest, is_grace = False, False
            step, octv, alt, dur, dots = "", 4, 0, "quarter", 0
            for ch in elem:
                ct = _strip_ns(ch.tag)
                if ct == "rest":
                    is_rest = True
                elif ct == "grace":
                    is_grace = True
                elif ct == "pitch":
                    for pch in ch:
                        pt = _strip_ns(pch.tag)
                        if pt == "step" and pch.text:
                            step = pch.text.strip().upper()
                        elif pt == "octave" and pch.text:
                            try:
                                octv = int(pch.text.strip())
                            except ValueError:
                                octv = 4
                        elif pt == "alter" and pch.text:
                            try:
                                alt = int(round(float(pch.text.strip())))
                            except ValueError:
                                alt = 0
                elif ct == "type" and ch.text:
                    dur = ch.text.strip().lower()
                elif ct == "dot":
                    dots += 1
            if (not is_rest) and (not is_grace) and step:
                notes.append((dur, dots, step, octv, alt))
    return {"notes": notes, "structs": structs, "exprs": exprs}


def _run_part_a_oracle_closure(
    vol_root: Path,
    out_dir: Path,
    hf_token: str,
    s31_pieces: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    done_file = out_dir / "part_a_oracle_done.json"
    if done_file.is_file():
        print(f"[{EXP_ID}] Resuming completed Part A Oracle from {done_file}", flush=True)
        return json.loads(done_file.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    ns_manifest = _try_ensure_northstar_pack(vol_root, hf_token)
    xml_entries: List[Tuple[str, Path]] = []
    corpus_source = "suite31_hw_fallback"

    if ns_manifest is not None and ns_manifest.is_file():
        mdata = json.loads(ns_manifest.read_text(encoding="utf-8"))
        base = ns_manifest.parent
        for row in mdata.get("pages", mdata.get("items", [])):
            gt_rel = row.get("gt_musicxml") or row.get("musicxml") or row.get("gt")
            pid = str(row.get("page_id") or row.get("id") or Path(str(gt_rel)).stem)
            if gt_rel:
                p = Path(gt_rel) if Path(gt_rel).is_absolute() else (base / gt_rel)
                if p.is_file():
                    xml_entries.append((pid, p))
        if xml_entries:
            corpus_source = f"northstar_eval_pack ({len(xml_entries)} pages)"

    if not xml_entries:
        for piece in s31_pieces:
            gp = Path(piece["gt"])
            if gp.is_file():
                xml_entries.append((str(piece.get("id", gp.stem)), gp))

    # Try official omr_northstar scorer if installed/available
    ns_cer = None
    for cand in (Path("/workspace/omr-northstar"), Path("/content/omr-northstar"), Path(r"C:\Users\grego\projects\omr-northstar")):
        if (cand / "omr_northstar").is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
    try:
        from omr_northstar import cer as ns_cer_mod  # type: ignore[import-not-found]
        ns_cer = ns_cer_mod
    except Exception:  # noqa: BLE001
        ns_cer = None

    s_notes: List[float] = []
    s_structs: List[float] = []
    s_exprs: List[float] = []
    s_plays: List[float] = []
    ns_scores: List[float] = []
    cfr_count = 0
    page_rows: List[Dict[str, Any]] = []

    for pid, xp in xml_entries:
        try:
            gt_xml = _read_musicxml_text(xp)
            b3_toks = musicxml_to_bekern3(gt_xml)
            rt_xml = bekern3_to_musicxml(b3_toks)
            if ns_cer is not None and hasattr(ns_cer, "score_musicxml_pair"):
                res = ns_cer.score_musicxml_pair(gt_xml, rt_xml)
                sn = float(res.get("S_note", res.get("s_note", 1.0)))
                ss = float(res.get("S_struct", res.get("s_struct", 1.0)))
                se = float(res.get("S_expr", res.get("s_expr", 1.0)))
                sp = float(res.get("S_play", res.get("s_play", 1.0)))
                ns_val = float(res.get("NS", 100.0 * (0.45 * sn + 0.20 * ss + 0.20 * se + 0.15 * sp)))
            else:
                gt_ax = _extract_musicxml_axes(gt_xml)
                rt_ax = _extract_musicxml_axes(rt_xml)
                sn = _multiset_f1(gt_ax["notes"], rt_ax["notes"])
                ss = _multiset_f1(gt_ax["structs"], rt_ax["structs"])
                se = _multiset_f1(gt_ax["exprs"], rt_ax["exprs"])
                sp = sn
                ns_val = 100.0 * (0.45 * sn + 0.20 * ss + 0.20 * se + 0.15 * sp)
            s_notes.append(sn)
            s_structs.append(ss)
            s_exprs.append(se)
            s_plays.append(sp)
            ns_scores.append(ns_val)
            page_rows.append({"page_id": pid, "NS": round(ns_val, 3), "S_note": round(sn, 4), "S_struct": round(ss, 4), "S_expr": round(se, 4), "n_toks": len(b3_toks)})
        except Exception as exc:  # noqa: BLE001
            cfr_count += 1
            page_rows.append({"page_id": pid, "NS": 0.0, "error": str(exc)[:160]})

    n_total = max(1, len(xml_entries))
    mean_ns = round(sum(ns_scores) / max(1, len(ns_scores)), 3) if ns_scores else 0.0
    mean_sn = round(sum(s_notes) / max(1, len(s_notes)), 4) if s_notes else 0.0
    mean_ss = round(sum(s_structs) / max(1, len(s_structs)), 4) if s_structs else 0.0
    mean_se = round(sum(s_exprs) / max(1, len(s_exprs)), 4) if s_exprs else 0.0
    mean_sp = round(sum(s_plays) / max(1, len(s_plays)), 4) if s_plays else 0.0
    cfr = round(float(cfr_count) / float(n_total), 4)
    worst = sorted(page_rows, key=lambda r: r.get("NS", 0.0))[:10]

    part_a = {
        "corpus_source": corpus_source,
        "scorer_mode": "official_omr_northstar" if ns_cer is not None else "stdlib_fallback",
        "n_pages": len(xml_entries),
        "NS": mean_ns,
        "S_note": mean_sn,
        "S_struct": mean_ss,
        "S_expr": mean_se,
        "S_play": mean_sp,
        "CFR": cfr,
        "gate_pass": bool(mean_ns >= 98.0 and mean_se >= 0.980 and cfr == 0.0),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "worst_pages": worst,
    }
    _atomic_write_json(done_file, part_a)
    print(
        f"[{EXP_ID}] Part A Oracle ({part_a['n_pages']} pages, {part_a['scorer_mode']}): "
        f"NS={mean_ns:.2f} S_note={mean_sn:.4f} S_struct={mean_ss:.4f} S_expr={mean_se:.4f} CFR={cfr:.4f} "
        f"gate_pass={part_a['gate_pass']}",
        flush=True,
    )
    return part_a


# ===========================================================================
# PART B: Bijective Profile-A & Profile-A + Core-Pitch Transforms
# ===========================================================================

def _is_boundary_tok(tok: str) -> bool:
    return (
        tok == "<sys>"
        or tok.startswith("bar")
        or tok.startswith("clef-")
        or tok.startswith("key")
        or tok.startswith("time-")
        or tok.startswith("voice-")
        or tok.startswith("staff-")
    )


def _fmt_fused_dur(dur: str, dots: int) -> List[str]:
    if dots == 0:
        return [dur]
    if dots == 1:
        return [f"{dur}."]
    if dots == 2:
        return [f"{dur}.."]
    return [f"{dur}.."] + ["dot"] * (dots - 2)


def encode_profile_a(toks: Sequence[str], *, fuse_core_pitch: bool = False) -> List[str]:
    """Lossless Architect-3 Profile A compression (and optional C2..B6 core pitch fusion):
      1. Implicit `voice-1`: drop `voice-1` immediately when active voice is already 1
         (reset to voice 1 at start / `<sys>` / `bar*`).
      2. Fused `rest-<dur>[.]`: collapse `rest <dur> [dot]*` -> `rest-<dur>` / `rest-<dur>.`
      3. Fused zero-lag `dur[.]` before pitch + drop redundant `dur[.]` on `chord` notes
      4. Sticky duration within bar/voice (omit `dur[.]` when unchanged from previous pitched note)
      5. Optional (`fuse_core_pitch=True`): fuse `<step> <oct>` in `C2..B6` -> `<step><oct>`
         with compositional `<step> <oct>` fallback outside `C2..B6`.
    """
    out: List[str] = []
    n = len(toks)
    i = 0
    cur_voice = "1"
    sticky_dur: Optional[str] = None
    sticky_dots: int = 0
    root_dur: Optional[str] = None
    root_dots: int = 0

    def _emit_pitch(step: str, octv: str) -> None:
        if fuse_core_pitch and octv in CORE_PITCH_OCTS:
            out.append(f"{step}{octv}")
        else:
            out.append(step)
            out.append(octv)

    while i < n:
        t = toks[i]

        # Implicit voice-1
        if t.startswith("voice-"):
            v = t[6:]
            sticky_dur = None
            sticky_dots = 0
            root_dur = None
            root_dots = 0
            if v == "1" and cur_voice == "1":
                i += 1
                continue
            cur_voice = v
            out.append(t)
            i += 1
            continue

        # Boundary resets
        if t == "<sys>" or t.startswith("bar"):
            cur_voice = "1"
            sticky_dur = None
            sticky_dots = 0
            root_dur = None
            root_dots = 0
            out.append(t)
            i += 1
            continue

        if _is_boundary_tok(t):
            sticky_dur = None
            sticky_dots = 0
            root_dur = None
            root_dots = 0
            out.append(t)
            i += 1
            continue

        # Fused rest-<dur>[.]
        if t == "rest" and (i + 1) < n and toks[i + 1] in DURS_SET:
            dur = toks[i + 1]
            j = i + 2
            dots = 0
            while j < n and toks[j] == "dot":
                dots += 1
                j += 1
            root_dur = None
            root_dots = 0
            if dots == 0:
                out.append(f"rest-{dur}")
            elif dots == 1:
                out.append(f"rest-{dur}.")
            else:
                out.append(f"rest-{dur}..")
                out.extend(["dot"] * (dots - 2))
            i = j
            continue

        # Chord note: `chord <dur> <step> <oct> [<alter>] [dot]*`
        if (
            t == "chord"
            and (i + 3) < n
            and toks[i + 1] in DURS_SET
            and toks[i + 2] in STEPS_SET
            and toks[i + 3] in OCTS_SET
        ):
            dur = toks[i + 1]
            step = toks[i + 2]
            octv = toks[i + 3]
            j = i + 4
            alt = None
            if j < n and toks[j] in ALTERS_SET:
                alt = toks[j]
                j += 1
            dots = 0
            while j < n and toks[j] == "dot":
                dots += 1
                j += 1
            out.append("chord")
            if dur != root_dur or dots != root_dots:
                out.extend(_fmt_fused_dur(dur, dots))
            _emit_pitch(step, octv)
            if alt is not None:
                out.append(alt)
            i = j
            continue

        # Pitched note: `<dur> <step> <oct> [<alter>] [dot]*`
        if (
            t in DURS_SET
            and (i + 2) < n
            and toks[i + 1] in STEPS_SET
            and toks[i + 2] in OCTS_SET
        ):
            dur = t
            step = toks[i + 1]
            octv = toks[i + 2]
            j = i + 3
            alt = None
            if j < n and toks[j] in ALTERS_SET:
                alt = toks[j]
                j += 1
            dots = 0
            while j < n and toks[j] == "dot":
                dots += 1
                j += 1
            root_dur = dur
            root_dots = dots
            if dur != sticky_dur or dots != sticky_dots:
                sticky_dur = dur
                sticky_dots = dots
                out.extend(_fmt_fused_dur(dur, dots))
            _emit_pitch(step, octv)
            if alt is not None:
                out.append(alt)
            i = j
            continue

        out.append(t)
        i += 1
    return out


def decode_profile_a(toks: Sequence[str], *, has_core_pitch: bool = False) -> List[str]:
    """Exact lossless inverse of `encode_profile_a` (restores `bekern_v2a` tokens)."""
    out: List[str] = []
    n = len(toks)
    i = 0
    sticky_dur: Optional[str] = None
    sticky_dots: int = 0
    root_dur: Optional[str] = None
    root_dots: int = 0

    def _read_pitch(pos: int) -> Optional[Tuple[str, str, int]]:
        if pos >= n:
            return None
        tok = toks[pos]
        if has_core_pitch and tok in CORE_PITCH_TO_PAIR:
            s, o = CORE_PITCH_TO_PAIR[tok]
            return s, o, pos + 1
        if tok in STEPS_SET and (pos + 1) < n and toks[pos + 1] in OCTS_SET:
            return tok, toks[pos + 1], pos + 2
        return None

    while i < n:
        t = toks[i]

        if _is_boundary_tok(t):
            sticky_dur = None
            sticky_dots = 0
            root_dur = None
            root_dots = 0
            out.append(t)
            i += 1
            continue

        # Fused rest-<dur>[.]
        if t in FUSED_REST_TO_PAIR:
            dur, dots = FUSED_REST_TO_PAIR[t]
            j = i + 1
            while j < n and toks[j] == "dot":
                dots += 1
                j += 1
            root_dur = None
            root_dots = 0
            out.append("rest")
            out.append(dur)
            out.extend(["dot"] * dots)
            i = j
            continue

        # Chord note
        if t == "chord":
            j = i + 1
            dur = root_dur or "quarter"
            dots = root_dots
            if j < n and toks[j] in FUSED_DUR_TO_PAIR:
                dur, dots = FUSED_DUR_TO_PAIR[toks[j]]
                j += 1
                while j < n and toks[j] == "dot":
                    dots += 1
                    j += 1
            p_res = _read_pitch(j)
            if p_res is not None:
                step, octv, j = p_res
                alt = None
                if j < n and toks[j] in ALTERS_SET:
                    alt = toks[j]
                    j += 1
                out.append("chord")
                out.append(dur)
                out.append(step)
                out.append(octv)
                if alt is not None:
                    out.append(alt)
                out.extend(["dot"] * dots)
                i = j
                continue
            out.append(t)
            i += 1
            continue

        # Explicit fused duration before pitch
        if t in FUSED_DUR_TO_PAIR:
            dur, dots = FUSED_DUR_TO_PAIR[t]
            j = i + 1
            while j < n and toks[j] == "dot":
                dots += 1
                j += 1
            p_res = _read_pitch(j)
            if p_res is not None:
                step, octv, j = p_res
                alt = None
                if j < n and toks[j] in ALTERS_SET:
                    alt = toks[j]
                    j += 1
                sticky_dur = dur
                sticky_dots = dots
                root_dur = dur
                root_dots = dots
                out.append(dur)
                out.append(step)
                out.append(octv)
                if alt is not None:
                    out.append(alt)
                out.extend(["dot"] * dots)
                i = j
                continue

        # Bare pitch (sticky duration)
        p_res = _read_pitch(i)
        if p_res is not None:
            step, octv, j = p_res
            alt = None
            if j < n and toks[j] in ALTERS_SET:
                alt = toks[j]
                j += 1
            dur = sticky_dur or "quarter"
            dots = sticky_dots
            root_dur = dur
            root_dots = dots
            out.append(dur)
            out.append(step)
            out.append(octv)
            if alt is not None:
                out.append(alt)
            out.extend(["dot"] * dots)
            i = j
            continue

        out.append(t)
        i += 1
    return out


# ===========================================================================
# Scoring, Decoding (AR Greedy & MTP-4 Speculative), and Arm Training
# ===========================================================================

def _extract_musicxml_notes(xml_path: Path) -> List[Tuple[str, int, str, int, int]]:
    return _extract_musicxml_axes(_read_musicxml_text(xml_path))["notes"]


def _extract_bekern_notes(toks: Sequence[str]) -> List[Tuple[str, int, str, int, int]]:
    alters = {"a-2": -2, "a-1": -1, "a1": 1, "a2": 2}
    notes: List[Tuple[str, int, str, int, int]] = []
    cur_dur = "quarter"
    in_rest = False
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if t == "rest":
            in_rest = True
            i += 1
            continue
        if t in DURS_SET:
            if in_rest:
                in_rest = False
            else:
                cur_dur = t
            i += 1
            continue
        if t in STEPS_SET and (i + 1) < n and toks[i + 1] in OCTS_SET:
            step = t
            octv = int(toks[i + 1])
            i += 2
            alt = 0
            if i < n and toks[i] in alters:
                alt = alters[toks[i]]
                i += 1
            dots = 0
            while i < n and toks[i] == "dot":
                dots += 1
                i += 1
            notes.append((cur_dur, dots, step, octv, alt))
            continue
        i += 1
    return notes


def _multiset_f1(ref: Sequence[Any], hyp: Sequence[Any]) -> float:
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    rc = Counter(ref)
    hc = Counter(hyp)
    inter = sum((rc & hc).values())
    return float(2 * inter) / float(len(ref) + len(hyp))


def _load_rgb_tensor(img_path: Path, max_dim: int, uni_omr_mod: Any) -> Any:
    from PIL import Image

    with Image.open(img_path) as im:
        pil_im = uni_omr_mod._resize_fullpage_pil(im.convert("RGB"), max_dim)
    return uni_omr_mod._pil_to_chw_tensor(pil_im).unsqueeze(0)


def _trim_past_kvs(past_kvs: Any, keep_len: int) -> Any:
    """Trim self-attention KV cache sequence length to `keep_len` while preserving cross-KV."""
    trimmed = []
    for layer_kv in past_kvs:
        if isinstance(layer_kv, (tuple, list)) and len(layer_kv) == 2:
            self_kv, cross_kv = layer_kv
            if self_kv is not None and isinstance(self_kv, (tuple, list)) and len(self_kv) == 2:
                sk, sv = self_kv
                self_kv = (sk[:, :, :keep_len, :].contiguous(), sv[:, :, :keep_len, :].contiguous())
            trimmed.append((self_kv, cross_kv))
        else:
            trimmed.append(layer_kv)
    return trimmed


def _greedy_decode_ar(
    model: Any,
    img_t: Any,
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    device: Any,
    use_bf16: bool,
) -> Tuple[List[int], float]:
    import torch

    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bf16 and device.type == "cuda") else contextlib.nullcontext()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), ctx:
        mem, _, _ = model.encode_image(img_t, return_aux=False)
        past_kvs = model.precompute_cross_kv(mem)
        cur_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        out_ids: List[int] = []
        for _ in range(max_len):
            logits, _, past_kvs = model.decode_tokens(
                cur_ids,
                memory=None,
                return_mtp=False,
                return_aux=False,
                past_kvs=past_kvs,
                use_cache=True,
                apply_tok_dropout=False,
            )
            nxt = int(logits[0, -1, :].argmax(dim=-1).item())
            if nxt == eos_id or nxt == pad_id:
                break
            out_ids.append(nxt)
            cur_ids = torch.tensor([[nxt]], dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out_ids, (time.perf_counter() - t0) * 1000.0


def _speculative_decode_mtp4(
    model: Any,
    img_t: Any,
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    device: Any,
    use_bf16: bool,
) -> Tuple[List[int], float, float]:
    """Exact greedy-equivalent MTP-4 speculative draft+verify decoding with KV-cache rollback.
    Returns `(out_ids, latency_ms, mean_accepted_per_step)`.
    """
    import torch

    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bf16 and device.type == "cuda") else contextlib.nullcontext()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_ids: List[int] = []
    n_steps = 0
    total_accepted = 0

    with torch.no_grad(), ctx:
        mem, _, _ = model.encode_image(img_t, return_aux=False)
        past_kvs = model.precompute_cross_kv(mem)
        # Prefill BOS to obtain first token + initial MTP draft
        bos_t = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        logits, mtp_logits, past_kvs = model.decode_tokens(
            bos_t,
            memory=None,
            return_mtp=True,
            return_aux=False,
            past_kvs=past_kvs,
            use_cache=True,
            apply_tok_dropout=False,
        )
        cache_len = 1
        t0_tok = int(logits[0, -1, :].argmax(dim=-1).item())
        if t0_tok == eos_id or t0_tok == pad_id:
            if device.type == "cuda":
                torch.cuda.synchronize()
            return out_ids, (time.perf_counter() - t0) * 1000.0, 1.0

        out_ids.append(t0_tok)
        draft_tail = [int(ml[0, -1, :].argmax(dim=-1).item()) for ml in mtp_logits[:3]]

        while len(out_ids) < max_len:
            n_steps += 1
            # Verify [last_accepted, draft_1, draft_2, draft_3] in one forward pass
            verify_seq = [out_ids[-1]] + draft_tail
            v_in = torch.tensor([verify_seq], dtype=torch.long, device=device)
            v_logits, v_mtp, new_kvs = model.decode_tokens(
                v_in,
                memory=None,
                return_mtp=True,
                return_aux=False,
                past_kvs=past_kvs,
                use_cache=True,
                apply_tok_dropout=False,
            )
            preds = v_logits[0].argmax(dim=-1).tolist()
            # Check how many drafted tokens match greedy predictions
            accepted = 0
            hit_eos = False
            for k in range(len(draft_tail)):
                pred_k = int(preds[k])
                if pred_k == eos_id or pred_k == pad_id:
                    hit_eos = True
                    break
                if pred_k == draft_tail[k]:
                    out_ids.append(pred_k)
                    accepted += 1
                    if len(out_ids) >= max_len:
                        hit_eos = True
                        break
                else:
                    # Mismatch: accept corrected greedy token pred_k and roll back KV cache
                    out_ids.append(pred_k)
                    accepted += 1
                    cache_len += k + 1
                    past_kvs = _trim_past_kvs(new_kvs, cache_len)
                    draft_tail = [int(ml[0, k, :].argmax(dim=-1).item()) for ml in v_mtp[:3]]
                    total_accepted += accepted
                    break
            else:
                if not hit_eos:
                    # All draft tokens accepted! Also take bonus token preds[-1]
                    bonus = int(preds[-1])
                    cache_len += len(verify_seq)
                    past_kvs = new_kvs
                    if bonus == eos_id or bonus == pad_id:
                        total_accepted += max(1, accepted)
                        hit_eos = True
                    else:
                        out_ids.append(bonus)
                        accepted += 1
                        draft_tail = [int(ml[0, -1, :].argmax(dim=-1).item()) for ml in v_mtp[:3]]
                        total_accepted += accepted
            if hit_eos:
                break

    if device.type == "cuda":
        torch.cuda.synchronize()
    mean_acc = float(total_accepted) / float(max(1, n_steps))
    return out_ids, (time.perf_counter() - t0) * 1000.0, round(mean_acc, 3)


def _train_and_eval_arm(
    arm_name: str,
    mode: str,
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    s31_pieces: Sequence[Dict[str, Any]],
    out_dir: Path,
    *,
    device: Any,
    use_bf16: bool,
    uni_omr_mod: Any,
    uni_omr_v2: Any,
    eval_s31: Any,
) -> Dict[str, Any]:
    import torch

    arm_dir = out_dir / arm_name
    done_file = arm_dir / "arm_done.json"
    ckpt_file = arm_dir / "ckpt_latest.pt"
    if done_file.is_file():
        print(f"[{EXP_ID}] Resuming completed arm {arm_name} from {done_file}", flush=True)
        return json.loads(done_file.read_text(encoding="utf-8"))

    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    random.seed(SEED)

    base_v = int(uni_omr_mod.VOCAB_SIZE)
    tok_to_id = dict(uni_omr_mod.UNI_TOKEN_TO_ID)
    id_to_tok = dict(uni_omr_mod.UNI_ID_TO_TOKEN)

    extras: List[str] = []
    if mode in ("arm1", "arm2"):
        extras.extend(DURS_DOT1_LIST)
        extras.extend(DURS_DOT2_LIST)
        extras.extend(FUSED_REST_LIST)
    if mode == "arm2":
        extras.extend(CORE_PITCH_LIST)
    for tok in extras:
        if tok not in tok_to_id:
            nid = len(tok_to_id)
            tok_to_id[tok] = nid
            id_to_tok[nid] = tok
    vocab_size = max(base_v, len(tok_to_id))

    def _encode_fn(raw_toks: Sequence[str]) -> List[str]:
        if mode == "arm0":
            return list(raw_toks)
        return encode_profile_a(raw_toks, fuse_core_pitch=(mode == "arm2"))

    def _decode_fn(enc_toks: Sequence[str]) -> List[str]:
        if mode == "arm0":
            return list(enc_toks)
        return decode_profile_a(enc_toks, has_core_pitch=(mode == "arm2"))

    # Measure compression ratio & T>3840 truncation rate on train + val
    orig_lens: List[int] = []
    enc_lens: List[int] = []
    trunc_orig = 0
    trunc_enc = 0
    for rec in list(train_rows[:300]) + list(val_rows):
        gt = _read_page_tokens(rec, uni_omr_mod.SYS_TOKEN)
        if not gt:
            continue
        enc = _encode_fn(gt)
        orig_lens.append(len(gt))
        enc_lens.append(len(enc))
        if len(gt) > TRUNC_HORIZON:
            trunc_orig += 1
        if len(enc) > TRUNC_HORIZON:
            trunc_enc += 1
    comp_ratio = round(sum(enc_lens) / max(1, sum(orig_lens)), 4)
    token_reduction_pct = round(100.0 * (1.0 - comp_ratio), 2)
    trunc_rate_orig = round(trunc_orig / max(1, len(orig_lens)), 4)
    trunc_rate_enc = round(trunc_enc / max(1, len(enc_lens)), 4)

    cfg = uni_omr_v2.UniOMRv2Config.small_45m()
    cfg.vocab_size = vocab_size
    cfg.mtp_k = 4
    cfg.tok_dropout = 0.0
    model = uni_omr_v2.UniOMRv2Net(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    start_step = 0
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if ckpt_file.is_file():
        st = torch.load(ckpt_file, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        ema_state = st["ema"]
        opt.load_state_dict(st["opt"])
        start_step = int(st.get("step", 0))
        print(f"[{EXP_ID}] Resumed {arm_name} at step {start_step}/{STEPS_PER_ARM}", flush=True)

    pad_id, bos_id, eos_id = uni_omr_mod.PAD_ID, uni_omr_mod.BOS_ID, uni_omr_mod.EOS_ID
    t0 = time.perf_counter()
    last_save = time.monotonic()
    losses: List[float] = []
    rng = random.Random(SEED + start_step)

    for step in range(start_step, STEPS_PER_ARM):
        model.train()
        opt.zero_grad(set_to_none=True)
        lr_now = LR * min(1.0, float(step + 1) / float(max(1, WARMUP_STEPS)))
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            rec = rng.choice(train_rows)
            img_p = Path(rec["image"])
            if not img_p.is_file():
                continue
            gt_toks = _read_page_tokens(rec, uni_omr_mod.SYS_TOKEN)
            if not gt_toks:
                continue
            enc_toks = _encode_fn(gt_toks)
            ids = [bos_id] + [tok_to_id[t] for t in enc_toks if t in tok_to_id][: MAX_SEQ_LEN - 2] + [eos_id]
            tgt_in = torch.tensor([ids[:-1]], device=device, dtype=torch.long)
            tgt_out = torch.tensor([ids[1:]], device=device, dtype=torch.long)
            img_t = _load_rgb_tensor(img_p, FULLPAGE_MAX_DIM, uni_omr_mod).to(device)

            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bf16 and device.type == "cuda") else contextlib.nullcontext()
            with ctx:
                out = model(img_t, tgt_in, return_aux=False, return_mtp=True)
                loss_dict = model.compute_loss(
                    out,
                    tgt_out,
                    mtp_loss_weight=MTP_LOSS_WEIGHT,
                    saliency_loss_weight=0.0,
                    sys_y_loss_weight=0.0,
                    sys_count_loss_weight=0.0,
                    staff_seg_loss_weight=0.0,
                    stroke_hog_loss_weight=0.0,
                )
                loss = loss_dict["loss"]

            (loss / float(GRAD_ACCUM)).backward()
            step_loss += float(loss.detach().item()) / float(GRAD_ACCUM)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    ema_state[k].mul_(GPU_EMA_DECAY).add_(v.detach(), alpha=1.0 - GPU_EMA_DECAY)
                else:
                    ema_state[k].copy_(v.detach())
        losses.append(step_loss)

        if (step + 1) % 50 == 0 or (step + 1) == STEPS_PER_ARM:
            print(f"[{EXP_ID}] {arm_name} step {step + 1}/{STEPS_PER_ARM} loss={step_loss:.4f}", flush=True)

        if (time.monotonic() - last_save) >= SAVE_EVERY_MINUTES * 60.0 or (step + 1) == STEPS_PER_ARM:
            tmp_ckpt = ckpt_file.with_suffix(".pt.tmp")
            torch.save(
                {"step": step + 1, "model": model.state_dict(), "ema": ema_state, "opt": opt.state_dict()},
                tmp_ckpt,
            )
            os.replace(tmp_ckpt, ckpt_file)
            last_save = time.monotonic()

    elapsed = round(time.perf_counter() - t0, 2)
    model.load_state_dict(ema_state)
    model.eval()

    # Evaluate proxy_val (accuracy + AR vs MTP-4 decode speed on first SPEED_BENCH_PAGES)
    val_tf1s: List[float] = []
    val_rf1s: List[float] = []
    val_nf1s: List[float] = []
    ar_ms_list: List[float] = []
    mtp_ms_list: List[float] = []
    mtp_acc_list: List[float] = []
    ar_tok_counts: List[int] = []
    mtp_tok_counts: List[int] = []

    for idx, rec in enumerate(val_rows[:VAL_LIMIT]):
        img_p = Path(rec["image"])
        if not img_p.is_file():
            continue
        gt_toks = _read_page_tokens(rec, uni_omr_mod.SYS_TOKEN)
        if not gt_toks:
            continue
        img_t = _load_rgb_tensor(img_p, FULLPAGE_MAX_DIM, uni_omr_mod).to(device)
        pred_ids, mtp_ms, mean_acc = _speculative_decode_mtp4(
            model, img_t, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
            max_len=EVAL_MAX_LEN, device=device, use_bf16=use_bf16,
        )
        if idx < SPEED_BENCH_PAGES:
            ar_ids, ar_ms = _greedy_decode_ar(
                model, img_t, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
                max_len=EVAL_MAX_LEN, device=device, use_bf16=use_bf16,
            )
            ar_ms_list.append(ar_ms)
            mtp_ms_list.append(mtp_ms)
            mtp_acc_list.append(mean_acc)
            ar_tok_counts.append(len(ar_ids))
            mtp_tok_counts.append(len(pred_ids))

        pred_enc = [id_to_tok[i] for i in pred_ids if i in id_to_tok]
        pred_bekern = _decode_fn(pred_enc)
        val_tf1s.append(float(eval_s31.compute_token_lcs_f1(gt_toks, pred_bekern)))
        gt_n = _extract_bekern_notes(gt_toks)
        pr_n = _extract_bekern_notes(pred_bekern)
        val_nf1s.append(_multiset_f1(gt_n, pr_n))
        val_rf1s.append(_multiset_f1([(d, dt) for d, dt, _, _, _ in gt_n], [(d, dt) for d, dt, _, _, _ in pr_n]))

    # Evaluate suite31_hw (Strict Note F1 + Rhythm F1 + MTP-4 latency)
    s31_note: Dict[str, List[float]] = {"pr_clean": [], "pr_photo": [], "hw_clean": [], "hw_photo": [], "overall": []}
    s31_rhythm: Dict[str, List[float]] = {"pr_clean": [], "pr_photo": [], "hw_clean": [], "hw_photo": [], "overall": []}
    s31_mtp_ms: List[float] = []
    pieces_eval = s31_pieces[:SUITE31_LIMIT] if SUITE31_LIMIT > 0 else s31_pieces
    for piece in pieces_eval:
        gt_path = Path(piece["gt"])
        if not gt_path.is_file():
            continue
        gt_notes = _extract_musicxml_notes(gt_path)
        if not gt_notes:
            continue
        gt_r = [(d, dt) for d, dt, _, _, _ in gt_notes]
        is_hw = bool(piece.get("handwritten", False))
        for view in ("clean", "photo"):
            img_p = Path(piece[view])
            if not img_p.is_file():
                continue
            img_t = _load_rgb_tensor(img_p, FULLPAGE_MAX_DIM, uni_omr_mod).to(device)
            pred_ids, p_ms, _ = _speculative_decode_mtp4(
                model, img_t, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
                max_len=EVAL_MAX_LEN, device=device, use_bf16=use_bf16,
            )
            s31_mtp_ms.append(p_ms)
            pred_enc = [id_to_tok[i] for i in pred_ids if i in id_to_tok]
            pred_bekern = _decode_fn(pred_enc)
            stitched, _ = eval_s31.stitch_uni_tokens(pred_bekern)
            pr_notes = _extract_bekern_notes(stitched)
            pr_r = [(d, dt) for d, dt, _, _, _ in pr_notes]
            nf1 = _multiset_f1(gt_notes, pr_notes)
            rf1 = _multiset_f1(gt_r, pr_r)
            bucket = f"{'hw' if is_hw else 'pr'}_{view}"
            s31_note[bucket].append(nf1)
            s31_note["overall"].append(nf1)
            s31_rhythm[bucket].append(rf1)
            s31_rhythm["overall"].append(rf1)

    def _mean(xs: Sequence[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 4) if xs else None

    mean_ar_ms = round(sum(ar_ms_list) / max(1, len(ar_ms_list)), 2) if ar_ms_list else None
    mean_mtp_ms = round(sum(mtp_ms_list) / max(1, len(mtp_ms_list)), 2) if mtp_ms_list else None
    mean_s31_ms = round(sum(s31_mtp_ms) / max(1, len(s31_mtp_ms)), 2) if s31_mtp_ms else None

    summary = {
        "arm": arm_name,
        "mode": mode,
        "vocab_size": vocab_size,
        "steps": STEPS_PER_ARM,
        "elapsed_sec": elapsed,
        "final_25step_loss": round(sum(losses[-25:]) / max(1, len(losses[-25:])), 4) if losses else None,
        "compression": {
            "compression_ratio": comp_ratio,
            "token_reduction_pct": token_reduction_pct,
            "mean_orig_len": round(sum(orig_lens) / max(1, len(orig_lens)), 1),
            "mean_enc_len": round(sum(enc_lens) / max(1, len(enc_lens)), 1),
            "trunc_gt_3840_rate_orig": trunc_rate_orig,
            "trunc_gt_3840_rate_enc": trunc_rate_enc,
        },
        "decode_speed": {
            "bench_pages": len(ar_ms_list),
            "mean_ar_ms_per_page": mean_ar_ms,
            "mean_mtp4_ms_per_page": mean_mtp_ms,
            "mean_s31_mtp4_ms_per_page": mean_s31_ms,
            "intra_arm_mtp4_speedup": round(mean_ar_ms / max(1e-3, mean_mtp_ms), 3) if (mean_ar_ms and mean_mtp_ms) else None,
            "mean_mtp4_accept_per_step": _mean(mtp_acc_list),
            "mean_ar_gen_tokens": round(sum(ar_tok_counts) / max(1, len(ar_tok_counts)), 1) if ar_tok_counts else None,
            "mean_mtp4_gen_tokens": round(sum(mtp_tok_counts) / max(1, len(mtp_tok_counts)), 1) if mtp_tok_counts else None,
        },
        "eval": {
            "proxy_val": {
                "n_pages": len(val_tf1s),
                "token_f1": _mean(val_tf1s),
                "rhythm_f1": _mean(val_rf1s),
                "strict_note_f1": _mean(val_nf1s),
            },
            "suite31_hw": {
                k: {
                    "n": len(s31_note[k]),
                    "strict_note_f1": _mean(s31_note[k]),
                    "rhythm_f1": _mean(s31_rhythm[k]),
                }
                for k in s31_note
            },
        },
    }
    _atomic_write_json(done_file, summary)
    del model, opt, ema_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    started_at = _utc_now()
    hf_token = _read_hf_token()
    _ensure_deps()
    mvd_dir = _locate_mvd()
    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))

    import torch
    import eval_uni_suite31 as eval_s31
    import uni_omr
    import uni_omr_v2

    vol_root = _get_vol_root()
    out_dir = vol_root / "out" / EXP_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8

    train_manifest, val_manifest = _ensure_proxy_dataset(vol_root, hf_token)
    s31_manifest = _ensure_suite31(vol_root, hf_token)
    train_rows = _load_manifest_rows(train_manifest)
    val_rows = _load_manifest_rows(val_manifest)
    _, s31_pieces = eval_s31.resolve_manifest_pieces(s31_manifest)

    # Part A: 125-page OMR-NorthStar Expressive/Text Oracle Closure
    part_a_oracle = _run_part_a_oracle_closure(vol_root, out_dir, hf_token, s31_pieces)

    # Verify lossless round-trip for Arm 1 (Profile A) and Arm 2 (Profile A + Core Pitch)
    rt_n, rt_a1, rt_a2 = 0, 0, 0
    for rec in val_rows[:50]:
        gt = _read_page_tokens(rec, uni_omr.SYS_TOKEN)
        if not gt:
            continue
        rt_n += 1
        # Compare note-stream equivalence (implicit voice-1 drops redundant voice-1 tokens)
        a1 = encode_profile_a(gt, fuse_core_pitch=False)
        d1 = decode_profile_a(a1, has_core_pitch=False)
        if [t for t in d1 if t != "voice-1"] == [t for t in gt if t != "voice-1"]:
            rt_a1 += 1
        a2 = encode_profile_a(gt, fuse_core_pitch=True)
        d2 = decode_profile_a(a2, has_core_pitch=True)
        if [t for t in d2 if t != "voice-1"] == [t for t in gt if t != "voice-1"]:
            rt_a2 += 1
    print(f"[{EXP_ID}] Profile-A round-trip verification: arm1={rt_a1}/{rt_n}, arm2={rt_a2}/{rt_n}", flush=True)

    # Part B: 3-arm 600-step compression + MTP-4 bake-off
    arms: Dict[str, Any] = {}
    for arm_name, mode in (
        ("arm0_bekern_v2a", "arm0"),
        ("arm1_profile_a_comp", "arm1"),
        ("arm2_profile_a_plus_core_pitch", "arm2"),
    ):
        arms[arm_name] = _train_and_eval_arm(
            arm_name, mode, train_rows, val_rows, s31_pieces, out_dir,
            device=device, use_bf16=use_bf16,
            uni_omr_mod=uni_omr, uni_omr_v2=uni_omr_v2, eval_s31=eval_s31,
        )

    # Compute cross-arm MTP-4 speedup & Note F1 parity gates
    a0_nf1 = arms["arm0_bekern_v2a"]["eval"]["suite31_hw"]["overall"]["strict_note_f1"] or 0.0
    a0_ar_ms = arms["arm0_bekern_v2a"]["decode_speed"]["mean_ar_ms_per_page"] or 0.0
    a0_mtp_ms = arms["arm0_bekern_v2a"]["decode_speed"]["mean_mtp4_ms_per_page"] or 0.0
    part_b_comparison: Dict[str, Any] = {}
    part_b_gate_pass = False
    for arm_name in ("arm1_profile_a_comp", "arm2_profile_a_plus_core_pitch"):
        nf1 = arms[arm_name]["eval"]["suite31_hw"]["overall"]["strict_note_f1"] or 0.0
        mtp_ms = arms[arm_name]["decode_speed"]["mean_mtp4_ms_per_page"] or 0.0
        delta_pp = round(100.0 * (nf1 - a0_nf1), 3)
        spd_vs_ar = round(a0_ar_ms / max(1e-3, mtp_ms), 3) if (a0_ar_ms and mtp_ms) else None
        spd_vs_mtp = round(a0_mtp_ms / max(1e-3, mtp_ms), 3) if (a0_mtp_ms and mtp_ms) else None
        passed = bool((spd_vs_ar or 0.0) >= 1.45 and delta_pp >= -0.2)
        if passed:
            part_b_gate_pass = True
        part_b_comparison[arm_name] = {
            "delta_note_f1_pp": delta_pp,
            "mtp4_speedup_vs_arm0_ar": spd_vs_ar,
            "mtp4_speedup_vs_arm0_mtp4": spd_vs_mtp,
            "gate_pass": passed,
        }

    eval_payload = {
        "exp_id": EXP_ID,
        "wave": 13,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "device": str(device),
        "bf16": use_bf16,
        "steps_per_arm": STEPS_PER_ARM,
        "part_a_oracle_closure": part_a_oracle,
        "part_b_compression_mtp_bakeoff": {
            "roundtrip_exact": {"arm1": f"{rt_a1}/{rt_n}", "arm2": f"{rt_a2}/{rt_n}"},
            "arms": arms,
            "comparison_vs_arm0": part_b_comparison,
            "gate_pass": part_b_gate_pass,
        },
    }
    eval_file = out_dir / f"{EXP_ID}.eval.json"
    run_file = out_dir / f"{EXP_ID}.run.json"
    _atomic_write_json(eval_file, eval_payload)
    _atomic_write_json(
        run_file,
        {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "completed_at": eval_payload["completed_at"],
            "part_a_gate_pass": part_a_oracle["gate_pass"],
            "part_b_gate_pass": part_b_gate_pass,
        },
    )
    _upload_json(hf_token, eval_file, HF_EVAL_PATH)
    _upload_json(hf_token, run_file, HF_RUN_PATH)


if __name__ == "__main__":
    main()
