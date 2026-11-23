#!/usr/bin/env python3
"""Consolidate all commercial-safe labeled handwritten training data into ONE place:
D:\\tmp\\real-hw-corpus\\

Uses NTFS hardlinks (with copy fallback) so every image, MusicXML, and bekern file
is physically present under D:\\tmp\\real-hw-corpus\\ with zero extra disk usage and
zero interference with running harvesters (e.g., D:\\tmp\\rism-pdm-hw\\).

Keeps BnF-Gallica-caveat Debussy (D:\\tmp\\real-hw-train-debussy\\) strictly OUTSIDE
this commercial-safe training folder.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil

ROOT = pathlib.Path(r"D:\tmp\real-hw-corpus")
EVAL_MANIFEST = pathlib.Path(
    r"C:\Users\grego\projects\sheet-music-converter\data-gen\score-corpus\suite31\suite31_hw_manifest.json"
)
TRAIN_WORKS_IN = ROOT / "train_works.jsonl"
RISM_DE = pathlib.Path(r"D:\tmp\rism-pdm-hw\manifest.jsonl")
RISM_CH = pathlib.Path(r"D:\tmp\rism-pdm-hw-intl\manifest.jsonl")


def link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return True


def load_jsonl(p: pathlib.Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main() -> None:
    eval_data = json.loads(EVAL_MANIFEST.read_text(encoding="utf-8"))
    eval_ids = set()
    for p in eval_data.get("pieces", []):
        slug = p["slug"]
        eval_ids.add(slug)
        if slug.startswith("hw-"):
            eval_ids.add(slug[3:])

    unified: list[dict] = []
    counts = {
        "openscore_works": 0,
        "openscore_pages": 0,
        "openscore_portrait_pages": 0,
        "openscore_landscape_pages": 0,
        "openscore_bekern_ok": 0,
        "quartet_works": 0,
        "quartet_clean_strips": 0,
        "quartet_deg_strips": 0,
        "rism_de_pairs": 0,
        "rism_ch_pairs": 0,
        "rism_portrait": 0,
        "rism_landscape": 0,
    }

    # 1. Full works & quartet strips from train_works.jsonl
    fw_rows = load_jsonl(TRAIN_WORKS_IN)
    for row in fw_rows:
        wid = row["work_id"]
        assert wid not in eval_ids and f"hw-{wid}" not in eval_ids, f"Eval leak: {wid}"
        src_name = row.get("source", "")
        if src_name == "openscore-lieder-pdm":
            counts["openscore_works"] += 1
            local_pages = []
            for idx, pstr in enumerate(row.get("page_paths", []), 1):
                sp = pathlib.Path(pstr)
                dp = ROOT / "full_works" / "pages" / f"{wid}_p{idx:03d}{sp.suffix.lower()}"
                if link_or_copy(sp, dp):
                    local_pages.append(str(dp.relative_to(ROOT)).replace("\\", "/"))
            counts["openscore_pages"] += len(local_pages)
            counts["openscore_portrait_pages"] += int(row.get("n_portrait_pages", 0))
            counts["openscore_landscape_pages"] += int(row.get("n_landscape_pages", 0))

            local_xml = None
            if row.get("xml_path"):
                sx = pathlib.Path(row["xml_path"])
                dx = ROOT / "full_works" / "xml" / f"{wid}{sx.suffix.lower()}"
                if link_or_copy(sx, dx):
                    local_xml = str(dx.relative_to(ROOT)).replace("\\", "/")

            local_bk = None
            if row.get("bekern_ok") and row.get("bekern_path"):
                counts["openscore_bekern_ok"] += 1
                sb = pathlib.Path(row["bekern_path"])
                db = ROOT / "full_works" / "bekern" / f"{wid}-bekern.txt"
                if link_or_copy(sb, db):
                    local_bk = str(db.relative_to(ROOT)).replace("\\", "/")

            unified.append({
                "id": wid,
                "label_scope": "full_work",
                "source": "openscore-lieder-pdm",
                "country": "DE",
                "library": row.get("library", ""),
                "license": row.get("license", "PDM 1.0"),
                "label_license": row.get("xml_license", "CC0-1.0"),
                "n_pages": len(local_pages),
                "n_portrait_pages": int(row.get("n_portrait_pages", 0)),
                "n_landscape_pages": int(row.get("n_landscape_pages", 0)),
                "image_paths": local_pages,
                "xml_path": local_xml,
                "bekern_path": local_bk,
                "bekern_ok": bool(row.get("bekern_ok", False)),
            })
        elif src_name == "real-hw-train-quartets":
            counts["quartet_works"] += 1
            # Hardlink all clean + degraded strips and targets for this quartet work
            rdir = pathlib.Path(r"D:\tmp\real-hw-train\renders")
            tdir = pathlib.Path(r"D:\tmp\real-hw-train\targets")
            clean_local, deg_local, bk_local = [], [], []
            for sp in sorted(rdir.glob(f"{wid}-*")):
                if sp.is_file():
                    dp = ROOT / "full_works" / "strips" / sp.name
                    if link_or_copy(sp, dp):
                        rel = str(dp.relative_to(ROOT)).replace("\\", "/")
                        if "-deg" in sp.name or sp.suffix.lower() == ".jpg":
                            deg_local.append(rel)
                        else:
                            clean_local.append(rel)
            for tp in sorted(tdir.glob(f"{wid}-*-bekern.txt")):
                dp = ROOT / "full_works" / "bekern" / tp.name
                if link_or_copy(tp, dp):
                    bk_local.append(str(dp.relative_to(ROOT)).replace("\\", "/"))
            counts["quartet_clean_strips"] += len(clean_local)
            counts["quartet_deg_strips"] += len(deg_local)
            unified.append({
                "id": wid,
                "label_scope": "system_strip",
                "source": "real-hw-train-quartets",
                "country": "IMSLP",
                "library": "IMSLP (PD)",
                "license": "PD",
                "label_license": "CC0-1.0",
                "n_clean_strips": len(clean_local),
                "n_deg_strips": len(deg_local),
                "image_paths": clean_local + deg_local,
                "bekern_paths": bk_local,
                "bekern_ok": len(bk_local) > 0,
            })

    # 2. RISM Opening-Incipit Labeled Pairs (DE + CH)
    seen_rism = set()
    for tag, country, mpath in [
        ("rism-de", "DE", RISM_DE),
        ("rism-ch", "CH", RISM_CH),
    ]:
        base = mpath.parent
        for row in load_jsonl(mpath):
            rid = str(row.get("rism_id") or row.get("source_id") or row.get("id") or "")
            if not rid or rid in seen_rism:
                continue
            assert rid not in eval_ids, f"Eval leak: {rid}"

            # Locate page image
            img_rel = row.get("image_path") or row.get("page_path") or f"pages/{rid}.jpg"
            img_src = pathlib.Path(img_rel)
            if not img_src.is_absolute():
                img_src = base / img_src
            if not img_src.exists():
                continue

            # Locate or materialize bekern text
            bk_dst = ROOT / "rism_incipits" / "bekern" / f"{tag}_{rid}-bekern.txt"
            bk_ok = False
            bk_rel = row.get("bekern_path")
            if bk_rel:
                bk_src = pathlib.Path(bk_rel)
                if not bk_src.is_absolute():
                    bk_src = base / bk_src
                bk_ok = link_or_copy(bk_src, bk_dst)
            elif row.get("bekern"):
                bk_dst.parent.mkdir(parents=True, exist_ok=True)
                if not bk_dst.exists():
                    bk_dst.write_text(str(row["bekern"]), encoding="utf-8")
                bk_ok = True
            if not bk_ok:
                continue

            img_dst = ROOT / "rism_incipits" / "pages" / f"{tag}_{rid}{img_src.suffix.lower()}"
            if not link_or_copy(img_src, img_dst):
                continue

            seen_rism.add(rid)
            if tag == "rism-de":
                counts["rism_de_pairs"] += 1
            else:
                counts["rism_ch_pairs"] += 1

            w = int(row.get("width", 0) or 0)
            h = int(row.get("height", 0) or 0)
            orient = row.get("orientation") or ("portrait" if h > w else "landscape")
            if orient == "portrait":
                counts["rism_portrait"] += 1
            else:
                counts["rism_landscape"] += 1

            unified.append({
                "id": f"{tag}_{rid}",
                "label_scope": "opening_incipit",
                "source": tag,
                "country": country,
                "library": row.get("institution", row.get("siglum", "")),
                "license": row.get("license", row.get("rights", "PDM 1.0")),
                "label_license": "CC-BY-4.0 (RISM)",
                "orientation": orient,
                "image_paths": [str(img_dst.relative_to(ROOT)).replace("\\", "/")],
                "bekern_path": str(bk_dst.relative_to(ROOT)).replace("\\", "/"),
                "bekern_ok": True,
                "n_bars": row.get("n_bars"),
                "n_tokens": row.get("n_tokens"),
            })

    out_manifest = ROOT / "all_labeled_hw.jsonl"
    tmp_manifest = ROOT / "all_labeled_hw.jsonl.tmp"
    with tmp_manifest.open("w", encoding="utf-8") as f:
        for r in unified:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_manifest, out_manifest)

    total_items = len(unified)
    total_images = (
        counts["openscore_pages"]
        + counts["quartet_clean_strips"]
        + counts["quartet_deg_strips"]
        + counts["rism_de_pairs"]
        + counts["rism_ch_pairs"]
    )
    summary = {
        "root": str(ROOT),
        "total_labeled_entries": total_items,
        "total_labeled_images_on_disk": total_images,
        "counts": counts,
        "eval_overlap": 0,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
