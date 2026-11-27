#!/usr/bin/env python3
"""Reusable OMR-NorthStar local evaluation runner.

Evaluates ANY OMR system's directory of predicted MusicXML files (`.musicxml`,
`.xml`, or compressed `.mxl`) against an OMR-NorthStar manifest using the
official `omr_northstar` stdlib scorer:
  - `omr_northstar.cer.score_musicxml_pair` (2-stage hierarchical CER alignment)
  - `omr_northstar.scoring.build_report` (4-layer canonical `report.json`)
  - `omr_northstar.stress.score_negative_control` (Pillar 3B NMHR gate)
  - `omr_northstar.validate.validate_manifest` & `validate_report`

Zero GPU usage, zero external dependencies beyond Python 3.10+ stdlib.

Usage:
    # Oracle self-check (GT == Pred -> NS = 100.0):
    python tmp-train/mvd/eval_northstar_runner.py \\
        --manifest D:\\tmp\\northstar-eval-pack\\manifest.json \\
        --oracle --system-name oracle-gt \\
        --out D:\\tmp\\northstar-eval-pack\\oracle_report.json

    # Evaluate a system's predicted MusicXML directory:
    python tmp-train/mvd/eval_northstar_runner.py \\
        --manifest D:\\tmp\\northstar-eval-pack\\manifest.json \\
        --pred-dir D:\\tmp\\northstar-eval-runs\\my-system\\musicxml \\
        --system-name my-system \\
        --out D:\\tmp\\northstar-eval-runs\\my-system\\report.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Mapping, Sequence

DEFAULT_NORTHSTAR_ROOT = pathlib.Path(r"C:\Users\grego\projects\omr-northstar")
DEFAULT_MANIFEST = pathlib.Path(r"D:\tmp\northstar-eval-pack\manifest.json")

EMPTY_VALID_MUSICXML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<score-partwise version="3.1">\n'
    '  <part-list><score-part id="P1"><part-name>Empty</part-name></score-part></part-list>\n'
    '  <part id="P1"/>\n'
    "</score-partwise>\n"
)


def _import_northstar(northstar_root: pathlib.Path):
    root_str = str(northstar_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from omr_northstar import cer, cli, scoring, stress, validate  # type: ignore[import-not-found]

    return cer, cli, scoring, stress, validate


def read_musicxml_file(path: pathlib.Path) -> str:
    """Read plain `.musicxml`/`.xml` or compressed `.mxl` ZIP MusicXML."""
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            # Prefer META-INF/container.xml rootfile declaration when present.
            if "META-INF/container.xml" in names:
                try:
                    c_root = ET.fromstring(zf.read("META-INF/container.xml"))
                    for elem in c_root.iter():
                        if elem.tag.rsplit("}", 1)[-1] == "rootfile":
                            full_path = elem.attrib.get("full-path")
                            if full_path and full_path in names:
                                return zf.read(full_path).decode("utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    pass
            for name in names:
                low = name.lower()
                if not low.startswith("meta-inf/") and (low.endswith(".musicxml") or low.endswith(".xml")):
                    return zf.read(name).decode("utf-8", errors="replace")
        raise ValueError(f"No MusicXML entry found inside {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def find_prediction_file(pred_dir: pathlib.Path, page_id: str) -> pathlib.Path | None:
    """Locate predicted MusicXML for `page_id` in `pred_dir`."""
    for ext in (".musicxml", ".xml", ".mxl"):
        cand = pred_dir / f"{page_id}{ext}"
        if cand.is_file():
            return cand
    # Also check nested subdir `<pred_dir>/<page_id>/*.{musicxml,xml,mxl}` (e.g. Audiveris output)
    sub = pred_dir / page_id
    if sub.is_dir():
        for ext in ("*.musicxml", "*.mxl", "*.xml"):
            matches = sorted(sub.glob(ext))
            if matches:
                return matches[0]
    return None


def _atomic_write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def backfill_legacy_manifest_metadata(manifest: dict[str, Any]) -> int:
    """In-memory backfill of legacy pack metadata fields required by omr-northstar >= db12da9.

    Non-destructive: mutates only the loaded manifest dict (never rewrites manifest.json on disk).
    Returns the number of fields backfilled.
    """
    n_filled = 0
    for rec in manifest.get("pages", []):
        if isinstance(rec, dict):
            opt = rec.get("optics_codec")
            if isinstance(opt, dict) and not (isinstance(opt.get("codec"), str) and opt["codec"].strip()):
                img_ext = pathlib.Path(str(rec.get("image_path", ""))).suffix.lstrip(".").lower()
                opt["codec"] = img_ext or "png"
                n_filled += 1
            prov = rec.get("provenance")
            if isinstance(prov, dict) and not (isinstance(prov.get("source_type"), str) and prov["source_type"].strip()):
                prov["source_type"] = "northstar_eval_pack_125p"
                n_filled += 1
    return n_filled


def evaluate_manifest(
    *,
    manifest_path: pathlib.Path,
    pred_dir: pathlib.Path | None,
    oracle: bool,
    system_name: str,
    northstar_root: pathlib.Path = DEFAULT_NORTHSTAR_ROOT,
    timing_path: pathlib.Path | None = None,
    strict_validate: bool = True,
    n_boot: int = 1000,
    bootstrap_seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Run full OMR-NorthStar evaluation and return `(report, per_page_rows, scorecard_text)`."""
    cer, cli, scoring, stress, validate = _import_northstar(northstar_root)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backfill_legacy_manifest_metadata(manifest)
    if strict_validate:
        m_errs = validate.validate_manifest(manifest)
        if m_errs:
            raise ValueError(
                f"Manifest validation failed ({len(m_errs)} errors):\n  - " + "\n  - ".join(m_errs[:10])
            )

    timing_by_page: dict[str, float] = {}
    telemetry: dict[str, Any] = {}
    if timing_path is not None and timing_path.is_file():
        tdata = json.loads(timing_path.read_text(encoding="utf-8"))
        if isinstance(tdata, dict):
            raw_lat = tdata.get("latencies_s", tdata)
            if isinstance(raw_lat, dict):
                for k, v in raw_lat.items():
                    if isinstance(v, (int, float)):
                        timing_by_page[str(k)] = float(v)
            for tk in ("peak_memory_gb", "parameter_count"):
                if tk in tdata:
                    telemetry[tk] = tdata[tk]

    pages_spec: Sequence[Mapping[str, Any]] = manifest["pages"]
    page_metrics_list = []
    per_page_rows: list[dict[str, Any]] = []
    neg_results: list[dict[str, Any]] = []
    missing_preds = 0

    for rec in pages_spec:
        pid = str(rec["page_id"])
        wid = str(rec["work_id"])
        dom = str(rec["domain"])
        tier = str(rec["complexity_tier"])
        anchor = rec.get("paired_anchor_id")
        is_neg = bool(rec.get("is_negative_control", False))
        neg_type = str(rec.get("negative_control_type", "non_music"))
        prov = rec.get("provenance") if isinstance(rec.get("provenance"), Mapping) else {}
        engraver = prov.get("engraver_or_publisher") or rec.get("engraver")

        gt_path = pathlib.Path(rec["gt_musicxml"])
        gt_xml = read_musicxml_file(gt_path)

        pred_source = "oracle-gt"
        if oracle:
            pred_xml = gt_xml
        else:
            assert pred_dir is not None
            pred_file = find_prediction_file(pred_dir, pid)
            if pred_file is not None:
                pred_source = str(pred_file)
                try:
                    pred_xml = read_musicxml_file(pred_file)
                except Exception:  # noqa: BLE001
                    pred_xml = ""
            else:
                missing_preds += 1
                pred_source = "missing"
                # If a system wrote an explicit `.abstain` marker on a negative control,
                # treat it as a valid empty MusicXML (0 hallucinated notes).
                if is_neg and (pred_dir / f"{pid}.abstain").is_file():
                    pred_xml = EMPTY_VALID_MUSICXML
                    pred_source = "abstain-marker"
                else:
                    pred_xml = ""

        lat = timing_by_page.get(pid)
        pm = cer.score_musicxml_pair(
            pred_xml,
            gt_xml,
            page_id=pid,
            work_id=wid,
            domain=dom,
            tier=tier,
            paired_anchor_id=str(anchor) if anchor else None,
            engraver=str(engraver) if engraver else None,
            latency_s=lat,
        )
        page_metrics_list.append(pm)
        pq = scoring.compute_page_quality(pm)

        row: dict[str, Any] = {
            "page_id": pid,
            "work_id": wid,
            "domain": dom,
            "tier": tier,
            "is_negative_control": is_neg,
            "pred_source": pred_source,
            "parseable": pm.parseable,
            "q_score": round(pq.q_score, 4),
            "s_note": round(pm.s_note, 4),
            "s_struct": round(pm.s_struct, 4),
            "s_expr": round(pm.s_expr, 4),
            "s_play": round(pm.s_play, 4),
            "wfpp": round(pq.effective_wfpp, 4),
            "ntsr": round(pq.ntsr, 4),
            "n_gt": pm.n_gt,
            "n_pred": pm.n_pred,
            "m_gt": pm.m_gt,
            "m_pred": pm.m_pred,
            "bsvr": round(pm.bsvr, 4),
            "catastrophic_failure": pq.catastrophic_failure,
            "cf_reasons": list(pq.cf_reasons),
            "latency_s": lat,
        }

        if is_neg:
            nc = stress.score_negative_control(pred_note_count=pm.n_pred, page_type=neg_type)
            row["negative_control_passed"] = nc.passed
            row["negative_control_failures"] = list(nc.gate_failures)
            neg_results.append(
                {
                    "page_id": pid,
                    "page_type": neg_type,
                    "pred_note_count": nc.pred_note_count,
                    "passed": nc.passed,
                    "gate_failures": list(nc.gate_failures),
                }
            )

        per_page_rows.append(row)

    adversarial: dict[str, Any] = {}
    if neg_results:
        n_neg = len(neg_results)
        n_pass = sum(1 for r in neg_results if r["passed"])
        n_fail = n_neg - n_pass
        adversarial["negative_control"] = {
            "n_pages": n_neg,
            "passed_count": n_pass,
            "nmhr": round(n_fail / n_neg, 4),
            "pass_rate": round(n_pass / n_neg, 4),
            "gate_passed": n_fail == 0,
            "pages": neg_results,
        }
    if missing_preds > 0:
        adversarial["missing_prediction_files"] = missing_preds

    report = scoring.build_report(
        page_metrics_list,
        system_name=system_name,
        split=str(manifest.get("split", "dev-v1")),
        benchmark_version=str(manifest.get("benchmark_version", "1.0.0")),
        adversarial=adversarial if adversarial else None,
        telemetry=telemetry if telemetry else None,
        n_boot=n_boot,
        bootstrap_seed=bootstrap_seed,
    )

    if strict_validate:
        r_errs = validate.validate_report(report)
        if r_errs:
            raise ValueError(
                f"Report validation failed ({len(r_errs)} errors):\n  - " + "\n  - ".join(r_errs[:10])
            )

    scorecard = cli.format_scorecard(report)
    return report, per_page_rows, scorecard


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OMR-NorthStar local evaluation runner")
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pred-dir", type=pathlib.Path, default=None)
    parser.add_argument("--oracle", action="store_true", help="Evaluate GT against itself (expect NS=100.0)")
    parser.add_argument("--system-name", type=str, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True, help="Output path for 4-layer report.json")
    parser.add_argument("--per-page-out", type=pathlib.Path, default=None, help="Optional per-page breakdown JSON")
    parser.add_argument("--timing-json", type=pathlib.Path, default=None, help="Optional latency/telemetry JSON")
    parser.add_argument("--northstar-root", type=pathlib.Path, default=DEFAULT_NORTHSTAR_ROOT)
    parser.add_argument("--no-strict-validate", action="store_true")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.oracle and args.pred_dir is None:
        parser.error("Either --pred-dir or --oracle must be provided.")

    report, per_page_rows, scorecard = evaluate_manifest(
        manifest_path=args.manifest,
        pred_dir=args.pred_dir,
        oracle=bool(args.oracle),
        system_name=args.system_name,
        northstar_root=args.northstar_root,
        timing_path=args.timing_json,
        strict_validate=not args.no_strict_validate,
        n_boot=args.n_boot,
        bootstrap_seed=args.bootstrap_seed,
    )

    _atomic_write_json(args.out, report)
    if args.per_page_out is not None:
        _atomic_write_json(args.per_page_out, {"system_name": args.system_name, "pages": per_page_rows})

    print(scorecard)
    neg_info = report.get("layer2_matrix_and_deltas", {}).get("adversarial_scorecard", {}).get("negative_control")
    if neg_info:
        print(
            f"\nNegative Controls (Pillar 3B): {neg_info['passed_count']}/{neg_info['n_pages']} passed "
            f"(NMHR={neg_info['nmhr'] * 100:.1f}%, gate_passed={neg_info['gate_passed']})"
        )
    print(f"\nWrote report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
