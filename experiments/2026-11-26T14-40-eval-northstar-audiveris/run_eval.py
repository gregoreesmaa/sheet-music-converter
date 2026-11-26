#!/usr/bin/env python3
"""Black-box Docker evaluator for Audiveris (AGPL-3.0) on OMR-NorthStar.

STRICT LICENSE ISOLATION:
  Audiveris is AGPL-3.0 and is ONLY invoked inside an ephemeral, network-disabled
  Docker container (`docker run --rm --network none ...`) as a black-box CLI
  benchmark baseline. It is NEVER imported into or linked with training/inference.
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
DEFAULT_OUT_DIR = pathlib.Path(r"D:\tmp\northstar-eval-runs\audiveris")
DEFAULT_IMAGE = "audiveris-northstar:5.3.1"

DOCKERFILE_TEMPLATE = """FROM eclipse-temurin:21-jre
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl unzip tesseract-ocr tesseract-ocr-eng libtesseract-dev \\
    && rm -rf /var/lib/apt/lists/*
ARG AUDIVERIS_DEB_URL=https://github.com/Audiveris/audiveris/releases/download/5.3.1/Audiveris-5.3.1-ubuntu22.04-x86_64.deb
RUN curl -fSL "${AUDIVERIS_DEB_URL}" -o /tmp/audiveris.deb \\
    && apt-get update && apt-get install -y --no-install-recommends /tmp/audiveris.deb \\
    && rm -rf /tmp/audiveris.deb /var/lib/apt/lists/*
ENTRYPOINT ["/opt/audiveris/bin/Audiveris"]
"""


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def build_docker_image(image_tag: str) -> None:
    """Build the isolated Audiveris black-box Docker image."""
    with tempfile.TemporaryDirectory(prefix="audiveris-docker-") as td:
        df = pathlib.Path(td) / "Dockerfile"
        df.write_text(DOCKERFILE_TEMPLATE, encoding="utf-8")
        cmd = ["docker", "build", "-t", image_tag, td]
        print(f"[audiveris] running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


def run_audiveris_page(
    *,
    image_tag: str,
    img_path: pathlib.Path,
    page_id: str,
    out_mxml_dir: pathlib.Path,
    timeout_s: int,
    memory_limit: str,
    empty_xml: str,
    is_neg: bool,
) -> tuple[bool, str, float]:
    """Run Audiveris on a single image in an isolated Docker container."""
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"aud-{page_id[:16]}-") as td:
        td_path = pathlib.Path(td)
        in_dir = td_path / "in"
        out_dir = td_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        staged_img = in_dir / f"{page_id}{img_path.suffix}"
        shutil.copy2(img_path, staged_img)

        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            memory_limit,
            "-v",
            f"{in_dir.resolve()}:/in:ro",
            "-v",
            f"{out_dir.resolve()}:/out:rw",
            image_tag,
            "-batch",
            "-export",
            "-output",
            "/out",
            f"/in/{staged_img.name}",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            _atomic_write_text(out_mxml_dir / f"{page_id}.musicxml", "")
            return False, f"timeout>{timeout_s}s", elapsed

        elapsed = time.perf_counter() - t0
        exports = sorted(out_dir.rglob("*.mxl")) + sorted(out_dir.rglob("*.musicxml")) + sorted(out_dir.rglob("*.xml"))
        if exports:
            src = exports[0]
            dst = out_mxml_dir / f"{page_id}{src.suffix.lower()}"
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            return True, f"ok({src.suffix.lower()})", elapsed

        # No exported score: if negative control and Audiveris cleanly found no staves/music, emit empty valid MusicXML.
        if is_neg and proc.returncode == 0:
            _atomic_write_text(out_mxml_dir / f"{page_id}.musicxml", empty_xml)
            return True, "neg-empty-ok", elapsed

        _atomic_write_text(out_mxml_dir / f"{page_id}.musicxml", "")
        err = (proc.stderr or proc.stdout or f"rc={proc.returncode}").strip().replace("\n", " ")[:120]
        return False, f"no-export(rc={proc.returncode}:{err})", elapsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Black-box Docker Audiveris evaluator for OMR-NorthStar")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--build-image", action="store_true", help="Build isolated Audiveris Docker image")
    parser.add_argument("--skip-eval", action="store_true", help="Exit after --build-image")
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--system-name", type=str, default="audiveris-5.3.1-docker")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--memory", type=str, default="6g")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args(argv)

    if args.build_image:
        build_docker_image(args.image)
        if args.skip_eval:
            return 0

    import eval_northstar_runner as ns_runner  # noqa: E402

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

        ok, status, elapsed = run_audiveris_page(
            image_tag=args.image,
            img_path=pathlib.Path(rec["image_path"]),
            page_id=pid,
            out_mxml_dir=mxml_dir,
            timeout_s=args.timeout_s,
            memory_limit=args.memory,
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
