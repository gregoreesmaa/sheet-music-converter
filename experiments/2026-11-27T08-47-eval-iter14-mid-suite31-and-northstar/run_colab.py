#!/usr/bin/env python3
"""Colab L4/H100 dual-benchmark evaluator for UniOMR Iter 14 v2a Mid (step 6874 EMA, 265-vocab bekern+v2):
  - Pass 1: 90-page `suite31_hw` (31 printed + 14 handwritten, clean + photo) via `eval_uni_suite31.py`
  - Pass 2: 125-page `OMR-NorthStar` (`31 CDR + 31 ALS + 31 RWP + 28 HWM + 4 Pillar 3B Negative Controls`)

Strictly internal evaluation without disturbing local RTX 4080 SUPER Iter 14 training.

Pipeline:
  1. Locate `tmp-train/mvd` and ensure required Python packages.
  2. Download `checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt` (step 6874, vocab=265),
     `datasets/suite31_hw_eval.tar`, and `datasets/northstar_eval_pack_125p.tar.gz` from HF Private Dataset Hub
     (`gregoreesmaa/sheet-music-converter-omr`).
  3. Clone/pull latest public `omr-northstar` (`1beaec4`+) for latest `cer.py` scorer.
  4. Resolve Rust MusicXML exporter with `bekern+v2` (`mscz-rs` `b8cb626`+) support:
     - First check prebuilt HF `tools/assembly-serialize-linux-x86_64` and probe `bekern+v2` (`dyn-f` -> `<dynamics>`).
     - If missing `bekern+v2` support and `cargo` + `git` are available, clone/build public `mscz-rs` (`b8cb626`+).
     - Fall back cleanly to `tools/assembly-serialize-linux-x86_64`, and finally `dual_verifier.bekern_to_musicxml_fallback`,
       always persisting `pred_raw` and stitched `bekern` per page for exact local Rust re-export if ever needed.
  5. Pass 1: Run full 90-page `suite31_hw` (`--max-dim 1408 --max-len 3072 --max-mem-tokens 4096
     --use-grammar --use-mtp-spec --bf16` on cc>=8) -> print exact `printed_clean/photo`, `hw_clean/photo`,
     `all45_clean/photo`.
  6. Pass 2: Run full 125-page `OMR-NorthStar` (`--neg-control-gate`, `1408px`, `max_len=3584`,
     `max_mem_tokens=4096`, FSA + MTP-4, BF16 on cc>=8) -> print exact `NS Score`, `Tier-W NS`,
     `CDR/ALS/RWP/HWM`, `S_note/S_struct/S_expr/S_play`, `median_wFPP`, `mean_NTSR`, `NMHR`.
  7. Upload combined `evals/2026-11-27T08-47-eval-iter14-mid-suite31-and-northstar.json` (both `suite31_hw`
     summary and `northstar` report), `*_per_page.json`, `*_bekern.json`, and
     `runs/2026-11-27T08-47-eval-iter14-mid-suite31-and-northstar.json` to HF.

Never echoes secrets (`HF_TOKEN` is read from env or `/tmp/.hf_token`).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

EXP_ID = "2026-11-27T08-47-eval-iter14-mid-suite31-and-northstar"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_PATH = os.environ.get("UNIOMR_HF_CKPT", "checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt")
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_NORTHSTAR_TAR = "datasets/northstar_eval_pack_125p.tar.gz"
HF_ASER_BIN_PATH = os.environ.get("UNIOMR_HF_ASER_BIN", "tools/assembly-serialize-linux-x86_64")

OMR_NORTHSTAR_GIT_URL = "https://github.com/gregoreesmaa/omr-northstar.git"
MSCZ_RS_GIT_URL = "https://github.com/gregoreesmaa/mscz-rs.git"

HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_PER_PAGE_PATH = f"evals/{EXP_ID}_per_page.json"
HF_BEKERN_PATH = f"evals/{EXP_ID}_bekern.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

SYSTEM_NAME = os.environ.get("NORTHSTAR_SYSTEM_NAME", "uni-omr-iter14-v2a-step6874-ema")
MAX_DIM = int(os.environ.get("UNIOMR_MAX_DIM", "1408"))
SUITE31_MAX_LEN = int(os.environ.get("UNIOMR_SUITE31_MAX_LEN", "3072"))
NORTHSTAR_MAX_LEN = int(os.environ.get("UNIOMR_NORTHSTAR_MAX_LEN", "3584"))
MAX_MEM_TOKENS = int(os.environ.get("UNIOMR_MAX_MEM_TOKENS", "4096"))
USE_GRAMMAR = os.environ.get("UNIOMR_USE_GRAMMAR", "1") != "0"
STRICT_GRAMMAR = os.environ.get("UNIOMR_STRICT_GRAMMAR", "0") == "1"
USE_MTP_SPEC = os.environ.get("UNIOMR_USE_MTP_SPEC", "1") != "0"
USE_NEG_GATE = os.environ.get("UNIOMR_NEG_GATE", "1") != "0"
LIMIT_PAGES = int(os.environ.get("NORTHSTAR_LIMIT", "0"))  # 0 = all 125 pages
LIMIT_SUITE31 = int(os.environ.get("SUITE31_LIMIT", "0"))  # 0 = all 45 pieces (90 pages)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_hf_token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    for cand in (pathlib.Path("/tmp/.hf_token"), pathlib.Path.home() / ".cache" / "huggingface" / "token"):
        if cand.is_file():
            val = cand.read_text(encoding="utf-8").strip()
            if val:
                return val
    return None


def _get_vol_root() -> pathlib.Path:
    if pathlib.Path("/vol").is_dir():
        return pathlib.Path("/vol")
    if pathlib.Path("/content").is_dir():
        return pathlib.Path("/content/vol")
    return pathlib.Path("/tmp/omr_vol")


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _locate_mvd_dir() -> pathlib.Path:
    candidates = [
        pathlib.Path("/workspace/mvd"),
        pathlib.Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        pathlib.Path("/content/mvd"),
        pathlib.Path("/content/sheet-music-converter/tmp-train/mvd"),
        pathlib.Path.cwd() / "tmp-train" / "mvd",
        pathlib.Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
    ]
    for cand in candidates:
        if (cand / "uni_omr.py").is_file() and (cand / "eval_northstar_runner.py").is_file():
            return cand

    for bundle in (pathlib.Path("/content/repo_bundle.tar.gz"), pathlib.Path("/content/bundle.tar.gz")):
        if bundle.is_file():
            target = pathlib.Path("/content/sheet-music-converter")
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:gz") as tf:
                tf.extractall(path=target)
            mvd = target / "tmp-train" / "mvd"
            if (mvd / "uni_omr.py").is_file():
                return mvd

    raise FileNotFoundError("Could not locate tmp-train/mvd containing uni_omr.py and eval_northstar_runner.py")


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
        print(f"[setup] Installing missing packages: {needed}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed", *needed],
            check=True,
        )


def _ensure_checkpoint(hf_token: Optional[str]) -> pathlib.Path:
    from huggingface_hub import hf_hub_download

    print(f"[ckpt] Fetching {HF_CKPT_PATH} from {HF_REPO_ID} ...", flush=True)
    ckpt_path = pathlib.Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=HF_CKPT_PATH,
            token=hf_token,
        )
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Downloaded checkpoint missing: {ckpt_path}")
    print(f"[ckpt] Ready at {ckpt_path} ({ckpt_path.stat().st_size:,} bytes)", flush=True)
    return ckpt_path


def _ensure_suite31_manifest(vol_root: pathlib.Path, hf_token: Optional[str]) -> pathlib.Path:
    from huggingface_hub import hf_hub_download

    s31_dir = vol_root / "datasets" / "suite31_hw_eval"
    sentinel = s31_dir / ".extracted"
    if not sentinel.is_file():
        print(f"[data] Downloading {HF_SUITE31_TAR} from {HF_REPO_ID} ...", flush=True)
        tar_p = pathlib.Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=HF_SUITE31_TAR,
                token=hf_token,
            )
        )
        s31_dir.mkdir(parents=True, exist_ok=True)
        print(f"[data] Extracting {tar_p} -> {s31_dir} ...", flush=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=s31_dir)
        _atomic_write_text(sentinel, _utc_now() + "\n")
    else:
        print(f"[data] Using cached suite31_hw_eval at {s31_dir}", flush=True)

    manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json under {s31_dir}")
    print(f"[data] suite31_hw manifest ready at {manifests[0]}", flush=True)
    return manifests[0]


def _ensure_northstar_pack(vol_root: pathlib.Path, hf_token: Optional[str]) -> pathlib.Path:
    from huggingface_hub import hf_hub_download

    pack_dir = vol_root / "datasets" / "northstar_eval_pack_125p"
    manifest_file = pack_dir / "manifest.json"
    if manifest_file.is_file() and (pack_dir / "omr_northstar" / "scoring.py").is_file():
        print(f"[data] Using cached NorthStar 125p pack at {pack_dir}", flush=True)
        return pack_dir

    dl_dir = vol_root / "datasets"
    dl_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] Downloading {HF_NORTHSTAR_TAR} from {HF_REPO_ID} ...", flush=True)
    tar_path = pathlib.Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=HF_NORTHSTAR_TAR,
            token=hf_token,
        )
    )
    print(f"[data] Extracting {tar_path} -> {dl_dir} ...", flush=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=dl_dir)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Expected {manifest_file} after extracting {tar_path}")
    print(f"[data] NorthStar pack ready at {pack_dir}", flush=True)
    return pack_dir


def _ensure_latest_omr_northstar(vol_root: pathlib.Path, pack_dir: pathlib.Path) -> Tuple[pathlib.Path, str]:
    """Clone/pull latest public `omr-northstar` (`1beaec4`+) for latest `cer.py`; fall back to pack_dir."""
    git_bin = shutil.which("git")
    if git_bin:
        ns_repo = vol_root / "src" / "omr-northstar"
        try:
            if (ns_repo / ".git").is_dir():
                subprocess.run(
                    [git_bin, "-C", str(ns_repo), "pull", "--ff-only", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            else:
                ns_repo.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [git_bin, "clone", "--depth", "1", OMR_NORTHSTAR_GIT_URL, str(ns_repo)],
                    capture_output=True,
                    text=True,
                    timeout=90,
                    check=True,
                )
            if (ns_repo / "omr_northstar" / "cer.py").is_file():
                rev = subprocess.run(
                    [git_bin, "-C", str(ns_repo), "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                commit = (rev.stdout or "unknown").strip()
                print(f"[northstar] Using latest cloned omr-northstar at {ns_repo} (commit={commit})", flush=True)
                return ns_repo, f"git:{commit}"
        except Exception as exc:
            print(f"[northstar] WARN: Could not clone/update omr-northstar ({exc}); falling back to pack bundle.", flush=True)

    print(f"[northstar] Using bundled omr_northstar in {pack_dir}", flush=True)
    return pack_dir, "pack:bundled"


def _probe_bekern_v2_support(bin_path: pathlib.Path) -> Tuple[bool, bool]:
    """Probe whether `bin_path` runs `bekern2musicxml` and emits `<dynamics>` for `bekern+v2` (`dyn-f`).

    Returns ``(works_basic, supports_v2_expr)``.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="probe-v2-") as td:
            td_path = pathlib.Path(td)
            in_txt = td_path / "probe.txt"
            in_txt.write_text("*clefG2 *k[] *M4/4 =1 4c dyn-f =|\n", encoding="utf-8")
            out_dir = td_path / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [
                    str(bin_path),
                    "bekern2musicxml",
                    "--input",
                    str(in_txt),
                    "--out",
                    str(out_dir),
                    "--stem",
                    "probe",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            mxml = out_dir / "probe.musicxml"
            if proc.returncode == 0 and mxml.is_file():
                xml_txt = mxml.read_text(encoding="utf-8", errors="replace")
                if "<note" in xml_txt:
                    has_v2 = ("<dynamics" in xml_txt) or ("<f/>" in xml_txt)
                    return True, has_v2
    except Exception:
        pass
    return False, False


def _resolve_or_build_rust_exporter(
    mvd_dir: pathlib.Path,
    vol_root: pathlib.Path,
    dv_mod: Any,
    hf_token: Optional[str] = None,
) -> Tuple[Optional[pathlib.Path], str]:
    """Resolve or build Rust `mscz` / `assembly-serialize` binary with `bekern+v2` (`b8cb626`) support.

    Order:
      1. `ASSEMBLY_SERIALIZE_BIN` / `MSCZ_BIN` env override.
      2. Prebuilt HF `tools/assembly-serialize-linux-x86_64` (used immediately if it supports `bekern+v2`).
      3. If prebuilt binary lacks `bekern+v2` support (or is unavailable) and `cargo` + `git` are installed,
         clone/build public `https://github.com/gregoreesmaa/mscz-rs.git` (`b8cb626`+).
      4. Clean fallback to prebuilt `tools/assembly-serialize-linux-x86_64` (if basic `bekern2musicxml` works),
         then local binaries, then `dual_verifier.bekern_to_musicxml_fallback`.
    """
    env_bin = os.environ.get("ASSEMBLY_SERIALIZE_BIN") or os.environ.get("MSCZ_BIN")
    if env_bin:
        p = pathlib.Path(env_bin)
        if p.is_file():
            ok, has_v2 = _probe_bekern_v2_support(p)
            if ok:
                return p, f"rust:env:{p.name}:v2={int(has_v2)}"

    fallback_hf_bin: Optional[pathlib.Path] = None
    is_linux_x64 = (
        platform.system().lower() == "linux"
        and platform.machine().lower() in ("x86_64", "amd64")
    )
    if is_linux_x64 and HF_ASER_BIN_PATH:
        cached_bin = vol_root / "bin" / "assembly-serialize-linux-x86_64"
        try:
            if not cached_bin.is_file() or cached_bin.stat().st_size == 0:
                from huggingface_hub import hf_hub_download

                print(f"[export] Downloading prebuilt Rust binary {HF_ASER_BIN_PATH} from {HF_REPO_ID} ...", flush=True)
                dl_path = pathlib.Path(
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

            ok, has_v2 = _probe_bekern_v2_support(cached_bin)
            if ok and has_v2:
                print(
                    f"[export] Prebuilt HF Rust binary supports bekern+v2: {cached_bin} ({cached_bin.stat().st_size:,} bytes)",
                    flush=True,
                )
                return cached_bin, "rust:hf:assembly-serialize-linux-x86_64:v2=1"
            if ok:
                print(
                    f"[export] Prebuilt HF Rust binary works for v1 bekern but lacks bekern+v2 dynamics; "
                    f"will try building latest mscz-rs if cargo is available.",
                    flush=True,
                )
                fallback_hf_bin = cached_bin
            else:
                print(f"[export] WARN: Prebuilt HF Rust binary probe failed.", flush=True)
        except Exception as exc:
            print(f"[export] WARN: Could not fetch/verify prebuilt HF Rust binary ({exc}).", flush=True)

    # Check cached built mscz-rs binary before rebuilding
    cached_mscz = vol_root / "bin" / "mscz-v2"
    if cached_mscz.is_file():
        ok, has_v2 = _probe_bekern_v2_support(cached_mscz)
        if ok and has_v2:
            print(f"[export] Using cached built mscz-rs binary: {cached_mscz}", flush=True)
            return cached_mscz, "rust:cached:mscz-v2:v2=1"

    cargo = shutil.which("cargo")
    git_bin = shutil.which("git")
    if cargo and git_bin:
        mscz_repo = vol_root / "src" / "mscz-rs"
        try:
            if (mscz_repo / ".git").is_dir():
                subprocess.run(
                    [git_bin, "-C", str(mscz_repo), "pull", "--ff-only", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            else:
                mscz_repo.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [git_bin, "clone", "--depth", "1", MSCZ_RS_GIT_URL, str(mscz_repo)],
                    capture_output=True,
                    text=True,
                    timeout=90,
                    check=True,
                )
            manifest_toml = mscz_repo / "Cargo.toml"
            if manifest_toml.is_file():
                rev = subprocess.run(
                    [git_bin, "-C", str(mscz_repo), "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                commit = (rev.stdout or "unknown").strip()
                print(f"[export] Building latest mscz-rs (commit={commit}) with cargo ...", flush=True)
                proc = subprocess.run(
                    [cargo, "build", "--release", "--manifest-path", str(manifest_toml)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
                if proc.returncode == 0:
                    for name in ("mscz", "assembly-serialize"):
                        cand = mscz_repo / "target" / "release" / name
                        for c in (cand, cand.with_suffix(".exe")):
                            if c.is_file():
                                ok, has_v2 = _probe_bekern_v2_support(c)
                                if ok:
                                    cached_mscz.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(c, cached_mscz)
                                    os.chmod(
                                        cached_mscz,
                                        os.stat(cached_mscz).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
                                    )
                                    print(
                                        f"[export] Built mscz-rs ({commit}): {cached_mscz} (v2={has_v2})",
                                        flush=True,
                                    )
                                    return cached_mscz, f"rust:built:mscz-rs@{commit}:v2={int(has_v2)}"
                else:
                    print(
                        f"[export] WARN: cargo build mscz-rs failed (rc={proc.returncode}): "
                        f"{(proc.stderr or proc.stdout).strip()[:300]}",
                        flush=True,
                    )
        except Exception as exc:
            print(f"[export] WARN: Could not clone/build mscz-rs ({exc}).", flush=True)

    if fallback_hf_bin is not None:
        print(
            f"[export] Falling back cleanly to prebuilt HF binary: {fallback_hf_bin}",
            flush=True,
        )
        return fallback_hf_bin, "rust:hf:assembly-serialize-linux-x86_64:v2=0"

    found = dv_mod.find_assembly_serialize_bin()
    if found is not None and found.is_file():
        ok, has_v2 = _probe_bekern_v2_support(found)
        if ok:
            return found, f"rust:local:{found.name}:v2={int(has_v2)}"

    for name in ("mscz", "assembly-serialize"):
        w = shutil.which(name)
        if w:
            wp = pathlib.Path(w)
            ok, has_v2 = _probe_bekern_v2_support(wp)
            if ok:
                return wp, f"rust:path:{wp.name}:v2={int(has_v2)}"

    print(
        "[export] WARNING: No Rust `mscz` / `assembly-serialize` binary found! "
        "Falling back to `dual_verifier.bekern_to_musicxml_fallback`. "
        "Raw & stitched bekern strings WILL be persisted in `*_per_page.json` and `*_bekern.json`.",
        flush=True,
    )
    return None, "python:fallback"


def _bekern_tokens_to_musicxml_exact(
    toks: Sequence[str],
    *,
    aser_bin: Optional[pathlib.Path],
    stem: str,
    empty_xml: str,
    dv_mod: Any,
) -> Tuple[str, bool, str, str]:
    if not toks:
        return empty_xml, True, "empty-tokens", "empty"

    if aser_bin is not None and aser_bin.is_file():
        with tempfile.TemporaryDirectory(prefix="ns-bekern-") as td:
            td_path = pathlib.Path(td)
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
            err_msg = (proc.stderr or proc.stdout or f"rc={proc.returncode}").strip()[:200]
            return "", False, f"rust-err:{err_msg}", f"rust:{aser_bin.name}"

    xml_opt = dv_mod.bekern_to_musicxml_fallback(toks)
    if xml_opt and xml_opt.strip():
        return xml_opt, True, "ok-py-fallback", "python:fallback"
    return "", False, "bekern2xml-empty", "python:fallback"


def _run_pass1_suite31(
    *,
    mvd_dir: pathlib.Path,
    ckpt_path: pathlib.Path,
    suite31_manifest: pathlib.Path,
    run_dir: pathlib.Path,
    use_cuda: bool,
    use_bf16: bool,
) -> Dict[str, Any]:
    """Pass 1: Full 90-page suite31_hw via `eval_uni_suite31.py`."""
    t0 = time.monotonic()
    out_json = run_dir / "suite31_hw_report.json"

    cmd = [
        sys.executable,
        "-u",
        str(mvd_dir / "eval_uni_suite31.py"),
        "--ckpt",
        str(ckpt_path),
        "--manifest",
        str(suite31_manifest),
        "--max-dim",
        str(MAX_DIM),
        "--max-len",
        str(SUITE31_MAX_LEN),
        "--max-mem-tokens",
        str(MAX_MEM_TOKENS),
        "--device",
        "cuda" if use_cuda else "cpu",
        "--out",
        str(out_json),
    ]
    if USE_GRAMMAR:
        cmd.append("--use-grammar")
    if STRICT_GRAMMAR:
        cmd.append("--strict-grammar")
    if USE_MTP_SPEC:
        cmd.append("--use-mtp-spec")
    if use_bf16:
        cmd.append("--bf16")
    if LIMIT_SUITE31 > 0:
        cmd.extend(["--limit-pieces", str(LIMIT_SUITE31)])

    print("\n" + "=" * 96, flush=True)
    print(f"[PASS 1] Running suite31_hw (90 pages): {' '.join(cmd)}", flush=True)
    print("=" * 96, flush=True)
    subprocess.run(cmd, check=True)

    if not out_json.is_file():
        raise FileNotFoundError(f"Expected suite31_hw report at {out_json}")
    s31_report = json.loads(out_json.read_text(encoding="utf-8"))
    summary = s31_report.get("summary", {})
    elapsed = round(time.monotonic() - t0, 2)

    print("\n[PASS 1 SUMMARY — suite31_hw]", flush=True)
    for key in ("printed_clean", "printed_photo", "hw_clean", "hw_photo", "all45_clean", "all45_photo"):
        s = summary.get(key, {})
        nf1 = s.get("note_f1_mean")
        tf1 = s.get("tok_f1_mean")
        ser = s.get("tok_ser_mean")
        dec = s.get("decode_ok_rate")
        n_sc = s.get("n_scored", 0)
        nf1_s = f"{nf1:.4f}" if isinstance(nf1, (int, float)) else "n/a"
        tf1_s = f"{tf1:.4f}" if isinstance(tf1, (int, float)) else "n/a"
        ser_s = f"{ser:.4f}" if isinstance(ser, (int, float)) else "n/a"
        dec_s = f"{dec * 100.0:.1f}%" if isinstance(dec, (int, float)) else "n/a"
        print(
            f"  {key:14s} (n={n_sc:2d}): Note_F1={nf1_s} | Tok_F1={tf1_s} | Tok_SER={ser_s} | Decode_OK={dec_s}",
            flush=True,
        )
    print(f"[PASS 1] Completed in {elapsed:.1f}s\n", flush=True)

    return {
        "elapsed_sec": elapsed,
        "max_dim": MAX_DIM,
        "max_len": SUITE31_MAX_LEN,
        "max_mem_tokens": MAX_MEM_TOKENS,
        "use_grammar": USE_GRAMMAR,
        "use_mtp_spec": USE_MTP_SPEC,
        "bf16": use_bf16,
        "summary": summary,
        "pieces": s31_report.get("pieces", []),
    }


def _print_northstar_headline(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract and explicitly print key OMR-NorthStar headline & layer2 metrics."""
    l1 = report.get("layer1_headline", {}) or {}
    l2 = report.get("layer2_matrix_and_deltas", {}) or {}
    l3 = report.get("layer3_component_breakdown", {}) or {}
    adv = l2.get("adversarial_scorecard", {}) or {}
    neg = adv.get("negative_control", {}) or {}
    dom_matrix = l2.get("domain_scores", {}) or l2.get("by_domain", {}) or {}

    def _pick(d: Mapping[str, Any], *keys: str) -> Any:
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    ns_score = _pick(l1, "ns_score", "NS", "overall_ns")
    tier_w_ns = _pick(l1, "tier_weighted_ns", "tier_w_ns", "TierW_NS")
    median_wfpp = _pick(l1, "median_wfpp", "median_wFPP")
    mean_ntsr = _pick(l1, "mean_ntsr", "mean_NTSR")
    nmhr = _pick(neg, "nmhr")

    def _dom_val(code: str) -> Any:
        v = dom_matrix.get(code)
        if isinstance(v, Mapping):
            return _pick(v, "ns_score", "score", "mean", "ns")
        return v

    cdr = _dom_val("CDR")
    als = _dom_val("ALS")
    rwp = _dom_val("RWP")
    hwm = _dom_val("HWM")

    # Component axes (check both layer3 and layer1)
    comp = l3.get("overall", l3) if isinstance(l3, Mapping) else {}
    s_note = _pick(comp, "s_note", "S_note", "note") or _pick(l1, "s_note", "S_note")
    s_struct = _pick(comp, "s_struct", "S_struct", "struct") or _pick(l1, "s_struct", "S_struct")
    s_expr = _pick(comp, "s_expr", "S_expr", "expr") or _pick(l1, "s_expr", "S_expr")
    s_play = _pick(comp, "s_play", "S_play", "play") or _pick(l1, "s_play", "S_play")

    def _fmt(v: Any, digits: int = 2) -> str:
        return f"{v:.{digits}f}" if isinstance(v, (int, float)) else str(v)

    compact = {
        "ns_score": ns_score,
        "tier_weighted_ns": tier_w_ns,
        "CDR": cdr,
        "ALS": als,
        "RWP": rwp,
        "HWM": hwm,
        "S_note": s_note,
        "S_struct": s_struct,
        "S_expr": s_expr,
        "S_play": s_play,
        "median_wFPP": median_wfpp,
        "mean_NTSR": mean_ntsr,
        "NMHR": nmhr,
        "neg_gate_passed": neg.get("gate_passed"),
    }

    print("[PASS 2 SUMMARY — OMR-NorthStar 125p]", flush=True)
    print(
        f"  NS Score={_fmt(ns_score)} | Tier-W NS={_fmt(tier_w_ns)} | "
        f"median_wFPP={_fmt(median_wfpp)} | mean_NTSR={_fmt(mean_ntsr, 4)} | "
        f"NMHR={_fmt(nmhr * 100.0 if isinstance(nmhr, (int, float)) else nmhr, 1)}%",
        flush=True,
    )
    print(
        f"  Domains: CDR={_fmt(cdr)} | ALS={_fmt(als)} | RWP={_fmt(rwp)} | HWM={_fmt(hwm)}",
        flush=True,
    )
    print(
        f"  Axes:    S_note={_fmt(s_note, 4)} | S_struct={_fmt(s_struct, 4)} | "
        f"S_expr={_fmt(s_expr, 4)} | S_play={_fmt(s_play, 4)}",
        flush=True,
    )
    return compact


def main() -> int:
    t_start = time.monotonic()
    started_at = _utc_now()
    print(f"[{started_at}] Starting {EXP_ID} | system={SYSTEM_NAME} | ckpt={HF_CKPT_PATH}", flush=True)

    _ensure_packages()
    mvd_dir = _locate_mvd_dir()
    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))
    print(f"[setup] Using mvd_dir={mvd_dir}", flush=True)

    import torch
    from PIL import Image
    from huggingface_hub import HfApi

    import dual_verifier as dv  # type: ignore[import-not-found]
    import eval_northstar_runner as ns_runner  # type: ignore[import-not-found]
    import eval_uni_suite31 as eus  # type: ignore[import-not-found]
    import neg_control_gate  # type: ignore[import-not-found]
    import uni_omr  # type: ignore[import-not-found]

    hf_token = _get_hf_token()
    vol_root = _get_vol_root()
    ckpt_path = _ensure_checkpoint(hf_token)
    suite31_manifest = _ensure_suite31_manifest(vol_root, hf_token)
    pack_dir = _ensure_northstar_pack(vol_root, hf_token)
    ns_root, ns_source = _ensure_latest_omr_northstar(vol_root, pack_dir)
    aser_bin, export_backend = _resolve_or_build_rust_exporter(mvd_dir, vol_root, dv, hf_token=hf_token)
    print(f"[export] Active MusicXML backend: {export_backend} (bin={aser_bin})", flush=True)

    run_dir = vol_root / "runs" / EXP_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_bf16 = bool(use_cuda and torch.cuda.get_device_capability(0)[0] >= 8)
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    print(f"[hw] device={device} gpu={gpu_name} bf16={use_bf16}", flush=True)

    # ------------------------------------------------------------------
    # PASS 1: Full 90-page suite31_hw
    # ------------------------------------------------------------------
    suite31_result = _run_pass1_suite31(
        mvd_dir=mvd_dir,
        ckpt_path=ckpt_path,
        suite31_manifest=suite31_manifest,
        run_dir=run_dir,
        use_cuda=use_cuda,
        use_bf16=use_bf16,
    )

    if use_cuda:
        torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------
    # PASS 2: Full 125-page OMR-NorthStar
    # ------------------------------------------------------------------
    t_ns = time.monotonic()
    print("=" * 96, flush=True)
    print(
        f"[PASS 2] Running OMR-NorthStar 125p (max_dim={MAX_DIM}, max_len={NORTHSTAR_MAX_LEN}, "
        f"max_mem_tokens={MAX_MEM_TOKENS}, grammar={USE_GRAMMAR}, mtp={USE_MTP_SPEC}, "
        f"neg_gate={USE_NEG_GATE}, bf16={use_bf16})",
        flush=True,
    )
    print("=" * 96, flush=True)

    mxml_dir = run_dir / "musicxml"
    bekern_dir = run_dir / "bekern"
    mxml_dir.mkdir(parents=True, exist_ok=True)
    bekern_dir.mkdir(parents=True, exist_ok=True)
    timing_path = run_dir / "timing.json"

    latencies: Dict[str, float] = {}
    conv_statuses: Dict[str, str] = {}
    page_predictions: Dict[str, Dict[str, Any]] = {}
    if timing_path.is_file():
        try:
            prev = json.loads(timing_path.read_text(encoding="utf-8"))
            latencies.update(prev.get("latencies_s", {}))
            conv_statuses.update(prev.get("conversion_status", {}))
        except Exception:
            pass

    model, cfg, ckpt_meta = eus.load_uni_checkpoint(ckpt_path, device=device)
    if MAX_MEM_TOKENS > 0 and hasattr(eus, "_override_max_mem_tokens"):
        try:
            eus._override_max_mem_tokens(model, MAX_MEM_TOKENS)
            cfg.max_mem_tokens = MAX_MEM_TOKENS
        except Exception as exc:
            print(f"[model] WARN: Could not override max_mem_tokens={MAX_MEM_TOKENS}: {exc}", flush=True)

    param_count = int(sum(p.numel() for p in model.parameters()))
    _, _, eos_id, _, _, _ = eus._vocab_spec(int(cfg.vocab_size))
    print(
        f"[model] loaded arch={ckpt_meta['arch']} step={ckpt_meta['step']} "
        f"max_mem_tokens={getattr(cfg, 'max_mem_tokens', '?')} "
        f"vocab={cfg.vocab_size} params={param_count:,} gpu={gpu_name} bf16={use_bf16}",
        flush=True,
    )

    manifest_path = pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = list(manifest["pages"])
    if LIMIT_PAGES > 0:
        pages = pages[:LIMIT_PAGES]

    decode_args = argparse.Namespace(
        max_len=NORTHSTAR_MAX_LEN,
        use_grammar=USE_GRAMMAR,
        strict_grammar=STRICT_GRAMMAR,
        use_mtp_spec=USE_MTP_SPEC,
    )

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    n_ok = 0
    n_fail = 0
    peak_mem_gb: Optional[float] = None

    for idx, rec in enumerate(pages, start=1):
        pid = str(rec["page_id"])
        out_mxml = mxml_dir / f"{pid}.musicxml"
        out_bekern = bekern_dir / f"{pid}.txt"
        out_raw = bekern_dir / f"{pid}.raw.txt"

        if out_mxml.is_file() and pid in latencies:
            st = conv_statuses.get(pid, "resumed")
            if st in ("ok", "ok-py-fallback", "empty-tokens") or st.startswith("neg-gate:"):
                n_ok += 1
            else:
                n_fail += 1
            bek_str = out_bekern.read_text(encoding="utf-8", errors="replace").strip() if out_bekern.is_file() else ""
            raw_str = out_raw.read_text(encoding="utf-8", errors="replace").strip() if out_raw.is_file() else ""
            page_predictions[pid] = {
                "pred_raw": raw_str,
                "bekern": bek_str,
                "n_toks": len(bek_str.split()) if bek_str else 0,
                "status": st,
                "export_backend": export_backend,
            }
            continue

        img_path = pack_dir / str(rec["image_path"])
        t0 = time.perf_counter()
        n_toks = 0
        raw_str = ""
        bek_str = ""
        used_backend = export_backend
        try:
            with Image.open(img_path) as raw_im:
                if USE_NEG_GATE:
                    is_non_music, gate_info = neg_control_gate.is_non_music_page(raw_im)
                    if is_non_music:
                        elapsed = time.perf_counter() - t0
                        latencies[pid] = round(elapsed, 4)
                        status = f"neg-gate:{gate_info['reason']}"
                        conv_statuses[pid] = status
                        _atomic_write_text(out_raw, "\n")
                        _atomic_write_text(out_bekern, "\n")
                        _atomic_write_text(out_mxml, ns_runner.EMPTY_VALID_MUSICXML)
                        page_predictions[pid] = {
                            "pred_raw": "",
                            "bekern": "",
                            "n_toks": 0,
                            "status": status,
                            "export_backend": "neg-gate",
                        }
                        n_ok += 1
                        _atomic_write_json(
                            timing_path,
                            {
                                "system_name": SYSTEM_NAME,
                                "parameter_count": param_count,
                                "peak_memory_gb": peak_mem_gb,
                                "export_backend": export_backend,
                                "latencies_s": latencies,
                                "conversion_status": conv_statuses,
                            },
                        )
                        print(
                            f"[{idx:03d}/{len(pages):03d}] {pid:42s} toks=   0 "
                            f"xml_ok=True  ({status} v={gate_info['v_frac']:.5f} h={gate_info['h_frac']:.5f}) "
                            f"{elapsed:.2f}s",
                            flush=True,
                        )
                        continue
                pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), MAX_DIM)

            img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
            seq, _, _, _ = eus._decode_single_view(
                model=model,
                cfg=cfg,
                img_t=img_t,
                args=decode_args,
                device=device,
                use_bf16=use_bf16,
                beam_size=1,
                beam_length_penalty=0.6,
                measure_beam_size=0,
                compute_logprob=False,
            )
            seq_ids = seq[0].detach().cpu().tolist()
            pred_raw_toks = eus.decode_uni_compat(
                seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
            )
            if eos_id not in seq_ids[1:]:
                pred_raw_toks = eus.trim_truncated_tokens(pred_raw_toks)
            raw_str = " ".join(pred_raw_toks)
            hyp_page_toks, _ = eus.stitch_uni_tokens(pred_raw_toks)
            bek_str = " ".join(hyp_page_toks)
            n_toks = len(hyp_page_toks)

            _atomic_write_text(out_raw, raw_str + "\n")
            _atomic_write_text(out_bekern, bek_str + "\n")
            xml_str, ok, status, used_backend = _bekern_tokens_to_musicxml_exact(
                hyp_page_toks,
                aser_bin=aser_bin,
                stem=pid,
                empty_xml=ns_runner.EMPTY_VALID_MUSICXML,
                dv_mod=dv,
            )
            _atomic_write_text(out_mxml, xml_str)
        except Exception as exc:
            err_line = f"# ERROR: {type(exc).__name__}: {exc}"
            _atomic_write_text(out_raw, err_line + "\n")
            _atomic_write_text(out_bekern, err_line + "\n")
            _atomic_write_text(out_mxml, "")
            ok = False
            status = f"infer-err({type(exc).__name__}:{str(exc)[:100]})"
            if use_cuda:
                torch.cuda.empty_cache()

        elapsed = time.perf_counter() - t0
        latencies[pid] = round(elapsed, 4)
        conv_statuses[pid] = status
        page_predictions[pid] = {
            "pred_raw": raw_str,
            "bekern": bek_str,
            "n_toks": n_toks,
            "status": status,
            "export_backend": used_backend,
        }
        if ok:
            n_ok += 1
        else:
            n_fail += 1

        if use_cuda:
            peak_mem_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)

        _atomic_write_json(
            timing_path,
            {
                "system_name": SYSTEM_NAME,
                "parameter_count": param_count,
                "peak_memory_gb": peak_mem_gb,
                "export_backend": export_backend,
                "latencies_s": latencies,
                "conversion_status": conv_statuses,
            },
        )
        print(
            f"[{idx:03d}/{len(pages):03d}] {pid:42s} toks={n_toks:4d} "
            f"xml_ok={str(ok):5s} ({status[:48]}) {elapsed:.2f}s",
            flush=True,
        )

    if use_cuda:
        peak_mem_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)
    del model
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    _atomic_write_json(
        timing_path,
        {
            "system_name": SYSTEM_NAME,
            "parameter_count": param_count,
            "peak_memory_gb": peak_mem_gb,
            "export_backend": export_backend,
            "latencies_s": latencies,
            "conversion_status": conv_statuses,
        },
    )

    prev_cwd = pathlib.Path.cwd()
    os.chdir(pack_dir)
    try:
        if LIMIT_PAGES > 0:
            sub_manifest_path = run_dir / "manifest_limit.json"
            sub_m = dict(manifest)
            sub_m["pages"] = pages
            _atomic_write_json(sub_manifest_path, sub_m)
            score_manifest = sub_manifest_path
        else:
            score_manifest = manifest_path

        ns_report, per_page_rows, scorecard = ns_runner.evaluate_manifest(
            manifest_path=score_manifest,
            pred_dir=mxml_dir,
            oracle=False,
            system_name=SYSTEM_NAME,
            northstar_root=ns_root,
            timing_path=timing_path,
            strict_validate=(LIMIT_PAGES == 0),
            n_boot=1000,
            bootstrap_seed=42,
        )
    finally:
        os.chdir(prev_cwd)

    ns_elapsed = round(time.monotonic() - t_ns, 2)
    ns_report["export_backend"] = export_backend
    ns_report["omr_northstar_source"] = ns_source
    ns_report["needs_local_rust_rescore"] = not export_backend.startswith("rust:")
    ns_report["checkpoint"] = HF_CKPT_PATH
    ns_report["step"] = ckpt_meta["step"]
    ns_report["vocab_size"] = int(cfg.vocab_size)
    ns_report["max_mem_tokens"] = getattr(cfg, "max_mem_tokens", None)

    print("\n" + scorecard + "\n", flush=True)
    ns_compact = _print_northstar_headline(ns_report)
    neg_info = ns_report.get("layer2_matrix_and_deltas", {}).get("adversarial_scorecard", {}).get("negative_control")
    if neg_info:
        print(
            f"Negative Controls (Pillar 3B): {neg_info['passed_count']}/{neg_info['n_pages']} passed "
            f"(NMHR={neg_info['nmhr'] * 100:.1f}%, gate_passed={neg_info['gate_passed']})",
            flush=True,
        )

    enriched_rows: List[Dict[str, Any]] = []
    for row in per_page_rows:
        r = dict(row)
        pid = str(r.get("page_id", ""))
        pred_info = page_predictions.get(pid)
        if pred_info:
            r["pred_raw"] = pred_info["pred_raw"]
            r["bekern"] = pred_info["bekern"]
            r["export_backend"] = pred_info["export_backend"]
            r["conversion_status"] = pred_info["status"]
        enriched_rows.append(r)

    combined_eval = {
        "exp_id": EXP_ID,
        "system_name": SYSTEM_NAME,
        "checkpoint": HF_CKPT_PATH,
        "arch": ckpt_meta["arch"],
        "step": ckpt_meta["step"],
        "vocab_size": int(cfg.vocab_size),
        "max_mem_tokens": getattr(cfg, "max_mem_tokens", None),
        "parameter_count": param_count,
        "gpu": gpu_name,
        "bf16": use_bf16,
        "export_backend": export_backend,
        "omr_northstar_source": ns_source,
        "needs_local_rust_rescore": not export_backend.startswith("rust:"),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "elapsed_sec": round(time.monotonic() - t_start, 2),
        "suite31_hw": {
            "elapsed_sec": suite31_result["elapsed_sec"],
            "max_dim": suite31_result["max_dim"],
            "max_len": suite31_result["max_len"],
            "max_mem_tokens": suite31_result["max_mem_tokens"],
            "use_grammar": suite31_result["use_grammar"],
            "use_mtp_spec": suite31_result["use_mtp_spec"],
            "summary": suite31_result["summary"],
            "pieces": suite31_result["pieces"],
        },
        "northstar_summary": ns_compact,
        "northstar": ns_report,
    }

    eval_path = run_dir / "eval_combined.json"
    per_page_path = run_dir / "per_page.json"
    bekern_dump_path = run_dir / "bekern_predictions.json"
    run_ledger_path = run_dir / "run_ledger.json"

    _atomic_write_json(eval_path, combined_eval)
    _atomic_write_json(
        per_page_path,
        {
            "exp_id": EXP_ID,
            "system_name": SYSTEM_NAME,
            "checkpoint": HF_CKPT_PATH,
            "arch": ckpt_meta["arch"],
            "step": ckpt_meta["step"],
            "vocab_size": int(cfg.vocab_size),
            "max_mem_tokens": getattr(cfg, "max_mem_tokens", None),
            "export_backend": export_backend,
            "omr_northstar_source": ns_source,
            "needs_local_rust_rescore": not export_backend.startswith("rust:"),
            "pages": enriched_rows,
        },
    )
    _atomic_write_json(
        bekern_dump_path,
        {
            "exp_id": EXP_ID,
            "system_name": SYSTEM_NAME,
            "checkpoint": HF_CKPT_PATH,
            "arch": ckpt_meta["arch"],
            "step": ckpt_meta["step"],
            "vocab_size": int(cfg.vocab_size),
            "max_mem_tokens": getattr(cfg, "max_mem_tokens", None),
            "export_backend": export_backend,
            "predictions": page_predictions,
            "latencies_s": latencies,
            "parameter_count": param_count,
            "peak_memory_gb": peak_mem_gb,
        },
    )
    _atomic_write_json(
        run_ledger_path,
        {
            "exp_id": EXP_ID,
            "system_name": SYSTEM_NAME,
            "checkpoint": HF_CKPT_PATH,
            "arch": ckpt_meta["arch"],
            "step": ckpt_meta["step"],
            "vocab_size": int(cfg.vocab_size),
            "max_mem_tokens": getattr(cfg, "max_mem_tokens", None),
            "export_backend": export_backend,
            "omr_northstar_source": ns_source,
            "needs_local_rust_rescore": not export_backend.startswith("rust:"),
            "gpu": gpu_name,
            "bf16": use_bf16,
            "max_dim": MAX_DIM,
            "suite31_max_len": SUITE31_MAX_LEN,
            "northstar_max_len": NORTHSTAR_MAX_LEN,
            "use_grammar": USE_GRAMMAR,
            "use_mtp_spec": USE_MTP_SPEC,
            "neg_control_gate": USE_NEG_GATE,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "elapsed_sec": round(time.monotonic() - t_start, 2),
            "suite31_elapsed_sec": suite31_result["elapsed_sec"],
            "northstar_elapsed_sec": ns_elapsed,
            "northstar_pages_total": len(pages),
            "northstar_conversion_ok": n_ok,
            "northstar_conversion_fail": n_fail,
            "telemetry": {
                "peak_memory_gb": peak_mem_gb,
                "parameter_count": param_count,
            },
            "suite31_summary": suite31_result["summary"],
            "northstar_summary": ns_compact,
            "northstar_headline": ns_report.get("layer1_headline", {}),
            "hf_eval_path": HF_EVAL_PATH,
            "hf_per_page_path": HF_PER_PAGE_PATH,
            "hf_bekern_path": HF_BEKERN_PATH,
        },
    )

    if hf_token and LIMIT_PAGES == 0 and LIMIT_SUITE31 == 0:
        api = HfApi(token=hf_token)
        for local_p, repo_p, msg in (
            (eval_path, HF_EVAL_PATH, f"eval: {EXP_ID} suite31_hw + NorthStar 125p combined report"),
            (per_page_path, HF_PER_PAGE_PATH, f"eval: {EXP_ID} NorthStar 125p per-page + bekern"),
            (bekern_dump_path, HF_BEKERN_PATH, f"eval: {EXP_ID} NorthStar 125p bekern dump"),
            (run_ledger_path, HF_RUN_PATH, f"run: {EXP_ID} ledger"),
        ):
            api.upload_file(
                path_or_fileobj=str(local_p),
                path_in_repo=repo_p,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=msg,
            )
            print(f"[hf] Uploaded {repo_p}", flush=True)
    else:
        print(
            f"[hf] Skipping upload (hf_token={'set' if hf_token else 'unset'}, "
            f"limit_pages={LIMIT_PAGES}, limit_suite31={LIMIT_SUITE31})",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
