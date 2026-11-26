#!/usr/bin/env python3
"""Evaluate UniOMR Iter 12 Stage 2 (`v2` EMA) on OMR-NorthStar (125-page local pack).

Runs UniOMR inference -> bekern tokens -> `assembly-serialize.exe bekern2musicxml`
-> `<out-dir>/musicxml/<page_id>.musicxml` -> `eval_northstar_runner.py` -> `report.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MVD_DIR = REPO_ROOT / "tmp-train" / "mvd"
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

DEFAULT_CKPT = pathlib.Path(r"D:\tmp\mvd-runs\uni-omr-iter12-v2-stage2-20k\uni_omr_v2_ema_best.pt")
DEFAULT_MANIFEST = pathlib.Path(r"D:\tmp\northstar-eval-pack\manifest.json")
DEFAULT_OUT_DIR = pathlib.Path(r"D:\tmp\northstar-eval-runs\iter12-v2")
DEFAULT_ASER = REPO_ROOT / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize.exe"


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def bekern_tokens_to_musicxml(
    toks: Sequence[str],
    *,
    aser_bin: pathlib.Path,
    stem: str,
    empty_xml: str,
) -> tuple[str, bool, str]:
    """Convert stitched bekern tokens into MusicXML via `assembly-serialize bekern2musicxml`."""
    if not toks:
        return empty_xml, True, "empty-tokens"

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
            return mxml_path.read_text(encoding="utf-8", errors="replace"), True, "ok"
        err_msg = (proc.stderr or proc.stdout or f"rc={proc.returncode}").strip()[:200]
        return "", False, err_msg


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run UniOMR Iter 12 v2 on OMR-NorthStar")
    parser.add_argument("--ckpt", type=pathlib.Path, default=DEFAULT_CKPT)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--aser-bin", type=pathlib.Path, default=DEFAULT_ASER)
    parser.add_argument("--system-name", type=str, default="uni-omr-iter12-v2-20k-ema")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-dim", type=int, default=1408)
    parser.add_argument("--max-len", type=int, default=3584)
    parser.add_argument("--use-grammar", action="store_true", default=True)
    parser.add_argument("--no-grammar", dest="use_grammar", action="store_false")
    parser.add_argument("--strict-grammar", action="store_true", default=False)
    parser.add_argument("--use-mtp-spec", action="store_true", default=True)
    parser.add_argument("--no-mtp-spec", dest="use_mtp_spec", action="store_false")
    parser.add_argument("--neg-control-gate", action="store_true", default=True)
    parser.add_argument("--no-neg-control-gate", dest="neg_control_gate", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from PIL import Image

    import eval_northstar_runner as ns_runner  # noqa: E402
    import eval_uni_suite31 as eus  # noqa: E402
    import neg_control_gate  # noqa: E402
    import uni_omr  # noqa: E402

    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not args.aser_bin.is_file():
        raise FileNotFoundError(f"assembly-serialize.exe not found: {args.aser_bin}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pages = list(manifest["pages"])
    if args.limit is not None:
        pages = pages[: args.limit]

    mxml_dir = args.out_dir / "musicxml"
    bekern_dir = args.out_dir / "bekern"
    mxml_dir.mkdir(parents=True, exist_ok=True)
    bekern_dir.mkdir(parents=True, exist_ok=True)

    timing_path = args.out_dir / "timing.json"
    latencies: dict[str, float] = {}
    if args.resume and timing_path.is_file():
        prev = json.loads(timing_path.read_text(encoding="utf-8"))
        latencies.update(prev.get("latencies_s", {}))

    device = torch.device(args.device)
    model, cfg, ckpt_meta = eus.load_uni_checkpoint(args.ckpt, device=device)
    param_count = int(sum(p.numel() for p in model.parameters()))
    _, _, eos_id, _, _, _ = eus._vocab_spec(int(cfg.vocab_size))
    use_bf16 = bool(args.bf16 and device.type == "cuda")

    print(
        f"[iter12-v2] loaded {ckpt_meta['arch']} step={ckpt_meta['step']} "
        f"params={param_count:,} device={device} bf16={use_bf16} pages={len(pages)}",
        flush=True,
    )

    decode_args = argparse.Namespace(
        max_len=args.max_len,
        use_grammar=args.use_grammar,
        strict_grammar=args.strict_grammar,
        use_mtp_spec=args.use_mtp_spec,
    )

    peak_mem_gb: float | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for idx, rec in enumerate(pages, start=1):
        pid = str(rec["page_id"])
        out_mxml = mxml_dir / f"{pid}.musicxml"
        if args.resume and out_mxml.is_file():
            continue

        img_path = pathlib.Path(rec["image_path"])
        t0 = time.perf_counter()
        with Image.open(img_path) as raw_im:
            if args.neg_control_gate:
                is_non_music, gate_info = neg_control_gate.is_non_music_page(raw_im)
                if is_non_music:
                    elapsed = time.perf_counter() - t0
                    latencies[pid] = round(elapsed, 4)
                    _atomic_write_text(bekern_dir / f"{pid}.txt", "\n")
                    _atomic_write_text(out_mxml, ns_runner.EMPTY_VALID_MUSICXML)
                    timing_payload: dict[str, Any] = {
                        "system_name": args.system_name,
                        "parameter_count": param_count,
                        "peak_memory_gb": peak_mem_gb,
                        "latencies_s": latencies,
                    }
                    _atomic_write_text(timing_path, json.dumps(timing_payload, indent=2) + "\n")
                    print(
                        f"[{idx:03d}/{len(pages):03d}] {pid:42s} toks=   0 "
                        f"xml_ok=True  (neg-gate:{gate_info['reason']} "
                        f"v={gate_info['v_frac']:.5f} h={gate_info['h_frac']:.5f}) {elapsed:.2f}s",
                        flush=True,
                    )
                    continue
            pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), args.max_dim)
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
        hyp_page_toks, _ = eus.stitch_uni_tokens(pred_raw_toks)

        _atomic_write_text(bekern_dir / f"{pid}.txt", " ".join(hyp_page_toks) + "\n")
        xml_str, ok, status = bekern_tokens_to_musicxml(
            hyp_page_toks,
            aser_bin=args.aser_bin,
            stem=pid,
            empty_xml=ns_runner.EMPTY_VALID_MUSICXML,
        )
        elapsed = time.perf_counter() - t0
        latencies[pid] = round(elapsed, 4)
        _atomic_write_text(out_mxml, xml_str)

        if device.type == "cuda":
            peak_mem_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)

        timing_payload: dict[str, Any] = {
            "system_name": args.system_name,
            "parameter_count": param_count,
            "peak_memory_gb": peak_mem_gb,
            "latencies_s": latencies,
        }
        _atomic_write_text(timing_path, json.dumps(timing_payload, indent=2) + "\n")
        print(
            f"[{idx:03d}/{len(pages):03d}] {pid:42s} toks={len(hyp_page_toks):4d} "
            f"xml_ok={str(ok):5s} ({status[:40]}) {elapsed:.2f}s",
            flush=True,
        )

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
