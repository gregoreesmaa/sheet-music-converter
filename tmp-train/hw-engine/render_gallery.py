#!/usr/bin/env python3
"""Render N diverse handwritten score simulations + an HTML eyeball gallery."""
from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import random
import sys
import time

ENGINE_DIR = pathlib.Path(__file__).resolve().parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))


def _render_one(job: tuple[int, str, str, str, int, str, int, int]) -> tuple[int, str, str, str, bool, str]:
    i, sc, sname, mode, seed, out_dir_s, page_w, page_h = job
    if str(ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_DIR))
    import numpy as np
    import styles as _st
    import render_mvp as _rm

    out_dir = pathlib.Path(out_dir_s)
    sc_stem = pathlib.Path(sc).stem.replace("gt-p1-", "")
    if mode == "random_scribe":
        r = np.random.default_rng(seed)
        custom = _st.sample_random_scribe(r, base_style=sname, jitter=0.25)
        reg_name = f"rand_{i:03d}_{sname}"
        _st.register_style(reg_name, custom, overwrite=True)
        use_name = reg_name
        label = f"rand({sname})"
    else:
        use_name = sname
        label = sname
    out_png = out_dir / f"{i + 1:03d}_{sc_stem}__{use_name}.png"
    try:
        _rm.render_score_handwritten(sc, use_name, str(out_png), seed=seed, page_w=page_w, page_h=page_h)
        return (i, out_png.name, sc_stem, label, True, "")
    except Exception as exc:
        return (i, out_png.name, sc_stem, label, False, str(exc))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--out-dir", type=str, default=r"tmp-train\hw-engine\eyeball100")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--page-w", type=int, default=1320)
    ap.add_argument("--page-h", type=int, default=1760)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    import styles as st

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_candidates: list[str] = []
    for p in sorted(pathlib.Path("data-gen/score-corpus").glob("*.mscx")):
        score_candidates.append(str(p))
    for p in sorted(pathlib.Path("data-gen/score-corpus/suite31/gt").glob("gt-p1-*.musicxml")):
        score_candidates.append(str(p))

    all_styles = st.list_styles()
    rng = random.Random(args.seed)

    jobs = []
    for i in range(args.count):
        sc = score_candidates[i % len(score_candidates)]
        if i < len(all_styles):
            sname = all_styles[i]
            mode = "named"
        else:
            sname = all_styles[rng.randrange(len(all_styles))]
            mode = "random_scribe"
        jobs.append((i, sc, sname, mode, 1000 + i, str(out_dir), args.page_w, args.page_h))

    t0 = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_render_one, jobs):
            results.append(r)

    ok = [r for r in results if r[4]]
    fail = [r for r in results if not r[4]]
    cards = []
    for i, fname, sc_stem, label, _is_ok, _err in sorted(ok):
        cards.append(
            f'<div class="card"><a href="{fname}" target="_blank"><img src="{fname}" loading="lazy"/></a>'
            f'<div class="cap"><b>#{i + 1:03d}</b> {label}<br/><span>{sc_stem}</span></div></div>'
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>100 Handwritten Simulations Eyeball Gallery</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#18181b;color:#f4f4f5;margin:20px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}"
        ".card{background:#27272a;border-radius:8px;overflow:hidden;border:1px solid #3f3f46}"
        ".card img{width:100%;display:block;background:#fff}.cap{padding:8px 10px;font-size:12px}"
        ".cap span{color:#a1a1aa}</style></head><body>"
        f"<h2>{len(ok)} Handwritten Score Simulations ({len(all_styles)} Named Styles + {max(0, len(ok) - len(all_styles))} Random Unseen Scribes)</h2>"
        f"<div class='grid'>{''.join(cards)}</div></body></html>"
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"Rendered {len(ok)}/{args.count} PNGs in {time.time() - t0:.1f}s | fails={len(fail)} | gallery={(out_dir / 'index.html').resolve()}")
    if fail:
        print("First 3 fails:", fail[:3])


if __name__ == "__main__":
    main()
