#!/usr/bin/env python3
"""Black-box CLI evaluator for `oemer` (MIT) on OMR-NorthStar (125-page local pack).

Invokes `oemer` per page in a CPU-only subprocess (`CUDA_VISIBLE_DEVICES=""`)
with per-page timeout and atomic checkpointing, then scores with
`tmp-train/mvd/eval_northstar_runner.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MVD_DIR = REPO_ROOT / "tmp-train" / "mvd"
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

DEFAULT_MANIFEST = pathlib.Path(r"D:\tmp\northstar-eval-pack\manifest.json")
DEFAULT_OUT_DIR = pathlib.Path(r"D:\tmp\northstar-eval-runs\oemer")
DEFAULT_OEMER_BIN = REPO_ROOT / ".venv" / "Scripts" / "oemer.exe"


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def run_oemer_page(
    *,
    oemer_bin: str,
    img_path: pathlib.Path,
    page_id: str,
    out_mxml_dir: pathlib.Path,
    timeout_s: int,
    without_deskew: bool,
    empty_xml: str,
    is_neg: bool,
) -> tuple[bool, str, float]:
    """Run `oemer` on a single page image in a CPU-only subprocess."""
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    with tempfile.TemporaryDirectory(prefix=f"oemer-{page_id[:16]}-") as td:
        td_path = pathlib.Path(td)
        cmd = [oemer_bin, str(img_path), "-o", str(td_path)]
        if without_deskew:
            cmd.append("--without-deskew")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            _atomic_write_text(out_mxml_dir / f"{page_id}.musicxml", "")
            return False, f"timeout>{timeout_s}s", elapsed

        elapsed = time.perf_counter() - t0
        exports = sorted(td_path.rglob("*.musicxml")) + sorted(td_path.rglob("*.xml"))
        if exports:
            src = exports[0]
            dst = out_mxml_dir / f"{page_id}.musicxml"
            tmp = dst.with_suffix(".musicxml.tmp")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            return True, "ok", elapsed

        if is_neg and proc.returncode == 0:
            _atomic_write_text(out_mxml_dir / f"{page_id}.musicxml", empty_xml)
            return True, "neg-empty-ok", elapsed

        _atomic_write_text(out_mxml_dir / f"{page_id}.musicxml", "")
        err = (proc.stderr or proc.stdout or f"rc={proc.returncode}").strip().replace("\n", " ")[:120]
        return False, f"no-export(rc={proc.returncode}:{err})", elapsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Black-box oemer evaluator for OMR-NorthStar")
    parser.add_argument("--oemer-bin", type=str, default=str(DEFAULT_OEMER_BIN))
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--system-name", type=str, default="oemer-cli")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--without-deskew", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args(argv)

    import eval_northstar_runner as ns_runner  # noqa: E402

    oemer_exe = args.oemer_bin
    if not pathlib.Path(oemer_exe).is_file() and shutil.which(oemer_exe) is None:
        raise FileNotFoundError(
            f"oemer executable not found at {oemer_exe!r}. Install with `pip install oemer` "
            "or pass --oemer-bin."
        )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pages = list(manifest["pages"])
    if args.limit is not None:
        pages = pages[: args.limit]

    mxml_dir = args.out_dir / "musicxml"
    mxml_dir.mkdir(parents=True, exist_ok=True)
    timing_path = args.out_dir / "timing.json"
    latencies: dict[str, float] = {}
    if args.resume and timing_path.is_file():
        latencies.update(json.loads(timing_path.read_text(encoding="utf-8")).get("latencies_s", {}))

    for idx, rec in enumerate(pages, start=1):
        pid = str(rec["page_id"])
        if args.resume and ns_runner.find_prediction_file(mxml_dir, pid) is not None:
            continue

        ok, status, elapsed = run_oemer_page(
            oemer_bin=oemer_exe,
            img_path=pathlib.Path(rec["image_path"]),
            page_id=pid,
            out_mxml_dir=mxml_dir,
            timeout_s=args.timeout_s,
            without_deskew=args.without_deskew,
            empty_xml=ns_runner.EMPTY_VALID_MUSICXML,
            is_neg=bool(rec.get("is_negative_control", False)),
        )
        latencies[pid] = round(elapsed, 4)
        payload: dict[str, Any] = {"system_name": args.system_name, "latencies_s": latencies}
        _atomic_write_text(timing_path, json.dumps(payload, indent=2) + "\n")
        print(f"[{idx:03d}/{len(pages):03d}] {pid:42s} ok={str(ok):5s} ({status}) {elapsed:.2f}s", flush=True)

    if args.skip_score:
        return 0

    return ns_runner.main(
        [
            "--manifest",
            str(args.manifest),
            "--pred-dir",
            str(mxml_dir),
            "--system-name",
            args.system_name,
            "--timing-json",
            str(timing_path),
            "--out",
            str(args.out_dir / "report.json"),
            "--per-page-out",
            str(args.out_dir / "per_page.json"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
