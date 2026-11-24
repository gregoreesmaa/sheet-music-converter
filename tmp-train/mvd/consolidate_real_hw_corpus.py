#!/usr/bin/env python3
"""Consolidate commercial-safe handwritten data into two strictly separated homes:

1. LABELED (full-work / system-level transcriptions ONLY):
   D:\\tmp\\real-hw-corpus\\all_labeled_hw.jsonl
   - 48 OpenScore Lieder PDM 1.0 x CC0 works (252 full pages; 47/48 bekern OK)
   - 4 Mozart String Quartet PD x CC0 works (12 clean + 24 degraded strips; 4/4 bekern OK)
   - NEVER includes RISM incipits (grouped under unlabeled per user rule).
   - NEVER includes BnF-Gallica-caveat Debussy (kept separate in real-hw-train-debussy).

2. UNLABELED (full-page PD/PDM scans, including RISM DE + CH):
   D:\\tmp\\unlabeled-10k-real\\rism_hw\\
   - 5,000 RISM Germany (D-B, D-Dl, D-Mbs, PDM 1.0)
   - 300 RISM Switzerland (CH-Bu, CH-SO, PDM 1.0)
   (Optional opening-incipit metadata preserved in manifest, but NEVER counted as labeled data.)

Uses NTFS hardlinks (zero extra disk usage).
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil

LABELED_ROOT = pathlib.Path(r"D:\tmp\real-hw-corpus")
UNLABELED_RISM_ROOT = pathlib.Path(r"D:\tmp\unlabeled-10k-real\rism_hw")
EVAL_MANIFEST = pathlib.Path(
    r"C:\Users\grego\projects\sheet-music-converter\data-gen\score-corpus\suite31\suite31_hw_manifest.json"
)
TRAIN_WORKS_IN = LABELED_ROOT / "train_works.jsonl"
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

    # Clean up old rism_incipits hardlinks from LABELED_ROOT so RISM never sits inside labeled
    old_rism_dir = LABELED_ROOT / "rism_incipits"
    if old_rism_dir.exists():
        shutil.rmtree(old_rism_dir)

    # 1. LABELED ONLY: Full works & system strips
    labeled: list[dict] = []
    l_counts = {
        "openscore_works": 0,
        "openscore_pages": 0,
        "openscore_portrait_pages": 0,
        "openscore_landscape_pages": 0,
        "openscore_bekern_ok": 0,
        "quartet_works": 0,
        "quartet_clean_strips": 0,
        "quartet_deg_strips": 0,
    }

    for row in load_jsonl(TRAIN_WORKS_IN):
        wid = row["work_id"]
        assert wid not in eval_ids and f"hw-{wid}" not in eval_ids, f"Eval leak: {wid}"
        src_name = row.get("source", "")
        if src_name == "openscore-lieder-pdm":
            l_counts["openscore_works"] += 1
            local_pages = []
            for idx, pstr in enumerate(row.get("page_paths", []), 1):
                sp = pathlib.Path(pstr)
                dp = LABELED_ROOT / "full_works" / "pages" / f"{wid}_p{idx:03d}{sp.suffix.lower()}"
                if link_or_copy(sp, dp):
                    local_pages.append(str(dp.relative_to(LABELED_ROOT)).replace("\\", "/"))
            l_counts["openscore_pages"] += len(local_pages)
            l_counts["openscore_portrait_pages"] += int(row.get("n_portrait_pages", 0))
            l_counts["openscore_landscape_pages"] += int(row.get("n_landscape_pages", 0))

            local_xml = None
            if row.get("xml_path"):
                sx = pathlib.Path(row["xml_path"])
                dx = LABELED_ROOT / "full_works" / "xml" / f"{wid}{sx.suffix.lower()}"
                if link_or_copy(sx, dx):
                    local_xml = str(dx.relative_to(LABELED_ROOT)).replace("\\", "/")

            local_bk = None
            if row.get("bekern_ok") and row.get("bekern_path"):
                l_counts["openscore_bekern_ok"] += 1
                sb = pathlib.Path(row["bekern_path"])
                db = LABELED_ROOT / "full_works" / "bekern" / f"{wid}-bekern.txt"
                if link_or_copy(sb, db):
                    local_bk = str(db.relative_to(LABELED_ROOT)).replace("\\", "/")

            labeled.append({
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
            l_counts["quartet_works"] += 1
            rdir = pathlib.Path(r"D:\tmp\real-hw-train\renders")
            tdir = pathlib.Path(r"D:\tmp\real-hw-train\targets")
            clean_local, deg_local, bk_local = [], [], []
            for sp in sorted(rdir.glob(f"{wid}-*")):
                if sp.is_file():
                    dp = LABELED_ROOT / "full_works" / "strips" / sp.name
                    if link_or_copy(sp, dp):
                        rel = str(dp.relative_to(LABELED_ROOT)).replace("\\", "/")
                        if "-deg" in sp.name or sp.suffix.lower() == ".jpg":
                            deg_local.append(rel)
                        else:
                            clean_local.append(rel)
            for tp in sorted(tdir.glob(f"{wid}-*-bekern.txt")):
                dp = LABELED_ROOT / "full_works" / "bekern" / tp.name
                if link_or_copy(tp, dp):
                    bk_local.append(str(dp.relative_to(LABELED_ROOT)).replace("\\", "/"))
            l_counts["quartet_clean_strips"] += len(clean_local)
            l_counts["quartet_deg_strips"] += len(deg_local)
            labeled.append({
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

    tmp_l = LABELED_ROOT / "all_labeled_hw.jsonl.tmp"
    with tmp_l.open("w", encoding="utf-8") as f:
        for r in labeled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_l, LABELED_ROOT / "all_labeled_hw.jsonl")

    # 2. UNLABELED: Group RISM DE + CH under D:\tmp\unlabeled-10k-real\rism_hw\
    unlabeled_rism: list[dict] = []
    u_counts = {"rism_de": 0, "rism_ch": 0, "portrait": 0, "landscape": 0}
    seen_rism = set()
    for tag, country, mpath in [("rism-de", "DE", RISM_DE), ("rism-ch", "CH", RISM_CH)]:
        base = mpath.parent
        for row in load_jsonl(mpath):
            rid = str(row.get("rism_id") or row.get("source_id") or row.get("id") or "")
            if not rid or rid in seen_rism:
                continue
            assert rid not in eval_ids, f"Eval leak: {rid}"
            img_rel = row.get("image_path") or row.get("page_path") or f"pages/{rid}.jpg"
            img_src = pathlib.Path(img_rel)
            if not img_src.is_absolute():
                img_src = base / img_src
            if not img_src.exists():
                continue
            img_dst = UNLABELED_RISM_ROOT / "images" / f"{tag}_{rid}{img_src.suffix.lower()}"
            if not link_or_copy(img_src, img_dst):
                continue
            seen_rism.add(rid)
            if tag == "rism-de":
                u_counts["rism_de"] += 1
            else:
                u_counts["rism_ch"] += 1
            w = int(row.get("width", 0) or 0)
            h = int(row.get("height", 0) or 0)
            orient = row.get("orientation") or ("portrait" if h > w else "landscape")
            if orient == "portrait":
                u_counts["portrait"] += 1
            else:
                u_counts["landscape"] += 1
            unlabeled_rism.append({
                "id": f"{tag}_{rid}",
                "corpus_group": "unlabeled",
                "source": tag,
                "country": country,
                "library": row.get("institution", row.get("siglum", "")),
                "license": row.get("license", row.get("rights", "PDM 1.0")),
                "orientation": orient,
                "image_path": str(img_dst.relative_to(UNLABELED_RISM_ROOT)).replace("\\", "/"),
                "optional_opening_incipit_bars": row.get("n_bars"),
                "optional_opening_incipit_tokens": row.get("n_tokens"),
            })

    UNLABELED_RISM_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_u = UNLABELED_RISM_ROOT / "manifest.jsonl.tmp"
    with tmp_u.open("w", encoding="utf-8") as f:
        for r in unlabeled_rism:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_u, UNLABELED_RISM_ROOT / "manifest.jsonl")

    summary = {
        "labeled_root": str(LABELED_ROOT),
        "labeled_works": len(labeled),
        "labeled_images": l_counts["openscore_pages"] + l_counts["quartet_clean_strips"] + l_counts["quartet_deg_strips"],
        "labeled_counts": l_counts,
        "unlabeled_rism_root": str(UNLABELED_RISM_ROOT),
        "unlabeled_rism_pages": len(unlabeled_rism),
        "unlabeled_rism_counts": u_counts,
        "eval_overlap": 0,
    }
    (LABELED_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (UNLABELED_RISM_ROOT / "summary.json").write_text(json.dumps(u_counts, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
