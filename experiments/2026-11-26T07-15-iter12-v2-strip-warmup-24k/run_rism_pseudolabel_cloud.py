#!/usr/bin/env python3
"""Colab / Kaggle cloud wrapper for sharded ``ov-39`` RISM pseudo-label generation.

Runs ``tmp-train/mvd/build_rism_pseudolabel_manifest.py`` on a shard of all
5,707 archival RISM pages (`datasets/rism_hw_5707.tar` on HF Private Hub),
using either a v1 (Iter 10.5 soup) or v2 (Iter 12) checkpoint, then uploads
the shard bundle (manifest JSONL + per-system ``*-bekern.txt`` + scored ledger
+ summary JSON) back to HF Private Dataset Hub.

Environment / CLI overrides:
  - ``HF_TOKEN`` or ``/tmp/.hf_token`` (chmod 600, purged immediately and in ``finally``)
  - ``--hf-ckpt`` (default: ``checkpoints/iter10_soup_baseline/uni_omr_soup.pt``)
  - ``--shard-idx`` / ``--num-shards`` (e.g. ``0/2`` and ``1/2`` across 2× T4 or 2× L4)
  - ``--hf-out-prefix`` (default: ``datasets/rism_pseudolabels_ov39``)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import tarfile

HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_RISM_TAR = "datasets/rism_hw_5707.tar"
DEFAULT_HF_CKPT = "checkpoints/iter10_soup_baseline/uni_omr_soup.pt"
DEFAULT_HF_OUT_PREFIX = "datasets/rism_pseudolabels_ov39"
TOKEN_PATH = Path("/tmp/.hf_token")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and TOKEN_PATH.exists():
        try:
            os.chmod(TOKEN_PATH, 0o600)
        except OSError:
            pass
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    if TOKEN_PATH.exists():
        try:
            TOKEN_PATH.unlink()
        except OSError:
            pass
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var")
    return token


def _detect_roots() -> tuple[Path, Path]:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/workspace"), Path("/kaggle/working/cache")
    return Path("/content/workspace"), Path("/content/cache")


def _extract_tar_once(tar_path: Path, dest_dir: Path) -> None:
    sentinel = dest_dir / ".extracted"
    if sentinel.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=dest_dir)
    sentinel.write_text(_utc_now() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cloud shard runner for ov-39 RISM pseudo-labeling.")
    ap.add_argument("--hf-ckpt", default=DEFAULT_HF_CKPT, help="HF dataset path to teacher .pt checkpoint.")
    ap.add_argument("--hf-out-prefix", default=DEFAULT_HF_OUT_PREFIX, help="HF dataset output folder.")
    ap.add_argument("--shard-idx", type=int, default=0, help="0-based shard index.")
    ap.add_argument("--num-shards", type=int, default=2, help="Total number of shards.")
    ap.add_argument("--max-dim", type=int, default=1280)
    ap.add_argument("--gen-max-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bundle-path", default="/content/bundle.tar.gz")
    args = ap.parse_args()

    try:
        hf_token = _read_and_purge_hf_token()
        work_root, cache_root = _detect_roots()
        work_root.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)

        mvd_dir = work_root / "sheet-music-converter" / "tmp-train" / "mvd"
        if not mvd_dir.exists():
            alt = Path(__file__).resolve().parents[2] / "tmp-train" / "mvd"
            if alt.exists():
                mvd_dir = alt
            else:
                bp = Path(args.bundle_path)
                if not bp.exists():
                    raise FileNotFoundError(f"Missing code bundle at {bp}")
                with tarfile.open(bp, "r:*") as tf:
                    tf.extractall(path=work_root)

        from huggingface_hub import HfApi, hf_hub_download

        ckpt_path = cache_root / args.hf_ckpt
        if not ckpt_path.exists():
            print(f"[cloud-rism] Downloading {args.hf_ckpt}...", flush=True)
            hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=args.hf_ckpt,
                local_dir=str(cache_root),
                token=hf_token,
            )

        rism_dir = cache_root / "datasets" / "rism_hw"
        if not (rism_dir / ".extracted").exists():
            print(f"[cloud-rism] Downloading {HF_RISM_TAR}...", flush=True)
            tar_p = Path(
                hf_hub_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    filename=HF_RISM_TAR,
                    local_dir=str(cache_root / "tars"),
                    token=hf_token,
                )
            )
            _extract_tar_once(tar_p, rism_dir)
            tar_p.unlink(missing_ok=True)

        shard_tag = f"shard_{args.shard_idx}_of_{args.num_shards}"
        out_dir = cache_root / "rism_ov39_out" / shard_tag
        out_manifest = out_dir / f"uni_rism_ov39_{shard_tag}.jsonl"

        cmd = [
            sys.executable, "-u",
            str(mvd_dir / "build_rism_pseudolabel_manifest.py"),
            "--ckpt", str(ckpt_path),
            "--input-dir", str(rism_dir),
            "--out", str(out_manifest),
            "--device", "cuda",
            "--bf16",
            "--max-dim", str(args.max_dim),
            "--gen-max-len", str(args.gen_max_len),
            "--shard-idx", str(args.shard_idx),
            "--num-shards", str(args.num_shards),
            "--resume",
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])

        print(f"[cloud-rism] Running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

        tar_out = cache_root / f"rism_ov39_{shard_tag}.tar.gz"
        with tarfile.open(tar_out, "w:gz") as tf:
            tf.add(out_dir, arcname=shard_tag)

        api = HfApi(token=hf_token)
        hf_dest = f"{args.hf_out_prefix.rstrip('/')}/{tar_out.name}"
        api.upload_file(
            path_or_fileobj=str(tar_out),
            path_in_repo=hf_dest,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"Upload ov-39 RISM pseudo-label {shard_tag}",
        )
        print(f"[cloud-rism] Uploaded {hf_dest}", flush=True)
    finally:
        if TOKEN_PATH.exists():
            try:
                TOKEN_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
