#!/usr/bin/env python3
"""Generate 250 deterministic synthetic negative-control pages for Iter 14 (CPU-only).

Produces:
  - D:\\tmp\\neg-controls-iter14\\pages\\neg_ctrl_000.{png,jpg} .. neg_ctrl_249.{png,jpg}
  - D:\\tmp\\neg-controls-iter14\\empty-bekern.txt (0 bytes -> empty target "" -> <eos>)
  - D:\\tmp\\neg-controls-iter14\\manifest.jsonl (250 rows, domain="neg_control", bucket="neg_control")

Mix (250 pages total):
  (a) 100 blank 5-line manuscript staves (varying staff counts, grand-staff pairs,
      with/without left/right margin barlines, clean + aged/photo variants)
  (b) 100 text-only program notes / lyrics / title pages (with/without decorative
      horizontal rules, double borders, boxes, clean + aged/photo variants)
  (c)  50 aged/yellowed/photo-like & non-music variants (blank parchment, ruled paper,
      heavy sepia/vignette/desk-border blank staves & title pages, JPEG artifacts)
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import io
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WINDOWS_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\timesi.ttf",
    r"C:\Windows\Fonts\timesbd.ttf",
    r"C:\Windows\Fonts\georgia.ttf",
    r"C:\Windows\Fonts\georgiai.ttf",
    r"C:\Windows\Fonts\georgiab.ttf",
    r"C:\Windows\Fonts\cambria.ttc",
    r"C:\Windows\Fonts\garamond.ttf",
    r"C:\Windows\Fonts\palai.ttf",
    r"C:\Windows\Fonts\pala.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\cour.ttf",
]

COMPOSERS = [
    "L. van Beethoven", "W. A. Mozart", "J. S. Bach", "F. Chopin",
    "C. Debussy", "R. Schumann", "F. Schubert", "J. Brahms",
    "F. Mendelssohn", "C. Schumann", "M. Ravel", "A. Dvorak",
    "G. Faure", "H. Wolf", "J. Haydn", "F. Liszt",
]

TITLES = [
    "SONATA IN C MINOR", "QUATUOR A CORDES", "PRELUDES POUR PIANO",
    "LIEDER OHNE WORTE", "NOCTURNES ET BALLADES", "CONCERTO GROSSO",
    "SIX ETUDES DE CONCERT", "PARTITA FOR SOLO VIOLIN", "SUITE FRANCAISE",
    "FANTASIA AND FUGUE", "STRING TRIO IN G MAJOR", "CHAMBER SYMPHONY",
    "VARIATIONS ON AN ORIGINAL THEME", "ALBUMBLATTER", "ROMANCES SANS PAROLES",
]

PROGRAM_PARAGRAPHS = [
    "Composed during the winter of 1842, this work exemplifies the lyrical restraint and contrapuntal clarity characteristic of the composer's mature chamber style. The opening movement introduces a brooding principal theme over a quietly pulsing accompaniment, gradually expanding into a luminous second subject in the relative major.",
    "The manuscript sources preserved in the municipal archive reveal several layers of revision, particularly in the transition leading to the recapitulation. Earlier sketches indicate a shorter coda, later expanded to balance the harmonic weight of the central development section.",
    "Performers should observe the distinction between articulated phrasing marks and broader legato slurs. Tempo modifications are intended to arise organically from the harmonic rhythm rather than abrupt agogic shifts.",
    "First performed in Leipzig on 14 November 1843, the piece drew warm praise from contemporary critics for its balanced scoring and idiomatic writing across all registers. The present edition restores original dynamic indications from the autograph fair copy.",
    "Notes on Editorial Practice: Accidentals in parentheses represent cautionary reminders supplied by the editor. Original beaming and page turns have been preserved wherever practical for modern performance.",
    "Movement I (Allegro moderato) establishes a dialogue between upper and lower registers. Movement II (Adagio cantabile) unfolds as a serene ternary aria. Movement III (Scherzo: Vivace) introduces buoyant syncopated rhythms before the Finale (Allegro con brio) brings the cycle to a decisive close.",
]

LYRIC_STANZAS = [
    [
        "1. Through quiet fields the evening breeze",
        "   Moves softly through the willow trees;",
        "   A distant bell across the hill",
        "   Rings out while all the world is still.",
    ],
    [
        "2. The silver stream reflects the sky,",
        "   Where drifting clouds go sailing by;",
        "   And shadows lengthen on the grass",
        "   As golden hours gently pass.",
    ],
    [
        "3. O stay, sweet light, a moment more",
        "   Upon the silent river shore;",
        "   Before the stars begin to gleam",
        "   Above the softly flowing stream.",
    ],
    [
        "4. Then night descends with gentle grace",
        "   To fold the valley in embrace;",
        "   And dreams awaken, calm and deep,",
        "   While weary hearts drift into sleep.",
    ],
]


def _discover_fonts() -> List[str]:
    return [p for p in WINDOWS_FONT_CANDIDATES if Path(p).is_file()]


def _load_font(font_paths: List[str], rng: random.Random, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_paths:
        for _ in range(3):
            fp = rng.choice(font_paths)
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _make_paper_canvas(
    w: int,
    h: int,
    rng: random.Random,
    np_rng: np.random.Generator,
    aged: bool = False,
) -> Image.Image:
    if aged:
        base_r = rng.randint(215, 245)
        base_g = rng.randint(195, min(232, base_r - 5))
        base_b = rng.randint(155, min(205, base_g - 12))
    else:
        v = rng.randint(242, 255)
        base_r, base_g, base_b = v, v, rng.randint(max(235, v - 6), v)

    arr = np.empty((h, w, 3), dtype=np.float32)
    arr[:, :, 0] = base_r
    arr[:, :, 1] = base_g
    arr[:, :, 2] = base_b

    # Subtle paper grain
    noise_std = rng.uniform(4.0, 11.0) if aged else rng.uniform(0.5, 3.0)
    if noise_std > 0.6:
        # Low-res smooth mottle + fine grain
        sh, sw = max(8, h // 16), max(8, w // 16)
        mottle = np_rng.normal(0.0, noise_std * 0.9, size=(sh, sw, 1)).astype(np.float32)
        mottle_img = Image.fromarray(np.clip(mottle[:, :, 0] + 128.0, 0, 255).astype(np.uint8), mode="L")
        mottle_up = np.asarray(mottle_img.resize((w, h), Image.BILINEAR), dtype=np.float32) - 128.0
        arr += mottle_up[:, :, None]
        fine = np_rng.normal(0.0, noise_std * 0.45, size=(h, w, 1)).astype(np.float32)
        arr += fine

    if aged:
        # Subtle illumination gradient / vignette
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        grad_x = rng.uniform(-16.0, 16.0) * xx
        grad_y = rng.uniform(-16.0, 16.0) * yy
        vig = -rng.uniform(4.0, 22.0) * (xx * xx + yy * yy)
        arr += (grad_x + grad_y + vig)[:, :, None]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _draw_foxing(img: Image.Image, rng: random.Random) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    for _ in range(rng.randint(8, 40)):
        cx = rng.randint(0, w)
        cy = rng.randint(0, h)
        r = rng.randint(2, 18)
        col = (rng.randint(130, 180), rng.randint(95, 145), rng.randint(55, 100), rng.randint(18, 55))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)


def _apply_photo_finish(
    img: Image.Image,
    rng: random.Random,
    np_rng: np.random.Generator,
    heavy: bool = False,
) -> Tuple[Image.Image, str]:
    w, h = img.size
    if heavy or rng.random() < 0.5:
        _draw_foxing(img, rng)

    # Slight rotation + optional dark desk border
    use_desk = heavy or rng.random() < 0.65
    if use_desk:
        desk_col = (
            rng.randint(28, 78),
            rng.randint(24, 66),
            rng.randint(20, 58),
        )
        pad_x = int(w * rng.uniform(0.02, 0.065))
        pad_y = int(h * rng.uniform(0.02, 0.065))
        canvas = Image.new("RGB", (w + 2 * pad_x, h + 2 * pad_y), desk_col)
        canvas.paste(img, (pad_x, pad_y))
        angle = rng.uniform(-3.8, 3.8) if heavy else rng.uniform(-2.2, 2.2)
        img = canvas.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=desk_col)
    else:
        angle = rng.uniform(-1.5, 1.5)
        bg = img.getpixel((5, 5))
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=bg)

    if heavy or rng.random() < 0.45:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))

    # Optional JPEG round-trip for photo realism
    if heavy or rng.random() < 0.6:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(62, 88))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return img, ".jpg"
    return img, ".png"


def _render_blank_staves(
    w: int,
    h: int,
    rng: random.Random,
    np_rng: np.random.Generator,
    font_paths: List[str],
    aged: bool,
) -> Image.Image:
    img = _make_paper_canvas(w, h, rng, np_rng, aged=aged)
    draw = ImageDraw.Draw(img)

    ink_v = rng.randint(45, 115) if aged else rng.randint(15, 75)
    ink = (ink_v, max(0, ink_v - rng.randint(0, 8)), max(0, ink_v - rng.randint(0, 15)))
    line_w = rng.choice([1, 1, 2, 2, 3])
    margin_l = int(w * rng.uniform(0.07, 0.13))
    margin_r = int(w * rng.uniform(0.07, 0.13))
    x0, x1 = margin_l, w - margin_r

    has_header = rng.random() < 0.35
    top_y = int(h * (0.14 if has_header else rng.uniform(0.07, 0.11)))
    bot_y = int(h * rng.uniform(0.89, 0.94))

    if has_header:
        f_hdr = _load_font(font_paths, rng, size=rng.randint(24, 38))
        hdr_text = rng.choice(["MANUSCRIPT PAPER", "Title: ______________________", "No. ____", "Page " + str(rng.randint(1, 24))])
        draw.text((x0, int(h * 0.05)), hdr_text, fill=ink, font=f_hdr)

    grouped_pairs = rng.random() < 0.35
    draw_left_bar = rng.random() < 0.65
    draw_right_bar = rng.random() < 0.60
    draw_mid_bar = rng.random() < 0.18

    if grouped_pairs:
        n_pairs = rng.choice([3, 4, 5])
        staff_space = rng.randint(9, 14)
        staff_h = 4 * staff_space
        pair_gap = int(staff_space * rng.uniform(3.5, 5.5))
        pair_h = 2 * staff_h + pair_gap
        avail = max(pair_h, bot_y - top_y)
        step = avail / max(1, n_pairs)
        for p in range(n_pairs):
            sy0 = int(top_y + p * step)
            sy1 = sy0 + staff_h + pair_gap
            if sy1 + staff_h > bot_y + 20:
                break
            for sy in (sy0, sy1):
                for li in range(5):
                    y = sy + li * staff_space
                    draw.line((x0, y, x1, y), fill=ink, width=line_w)
            if draw_left_bar:
                draw.line((x0, sy0, x0, sy1 + staff_h), fill=ink, width=line_w + 1)
                # Simple bracket tick
                draw.line((x0 - 8, sy0 - 2, x0, sy0), fill=ink, width=line_w + 1)
                draw.line((x0 - 8, sy1 + staff_h + 2, x0, sy1 + staff_h), fill=ink, width=line_w + 1)
            if draw_right_bar:
                draw.line((x1, sy0, x1, sy1 + staff_h), fill=ink, width=line_w + 1)
    else:
        n_staves = rng.choice([6, 8, 10, 12])
        staff_space = rng.randint(8, 15)
        staff_h = 4 * staff_space
        avail = max(staff_h, bot_y - top_y - staff_h)
        step = avail / max(1, n_staves - 1)
        for s in range(n_staves):
            sy = int(top_y + s * step)
            for li in range(5):
                y = sy + li * staff_space
                draw.line((x0, y, x1, y), fill=ink, width=line_w)
            if draw_left_bar:
                draw.line((x0, sy, x0, sy + staff_h), fill=ink, width=line_w + 1)
            if draw_right_bar:
                draw.line((x1, sy, x1, sy + staff_h), fill=ink, width=line_w + 1)
            if draw_mid_bar:
                for frac in (0.33, 0.66):
                    xm = int(x0 + (x1 - x0) * frac)
                    draw.line((xm, sy, xm, sy + staff_h), fill=ink, width=line_w)

    return img


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur: List[str] = []
    for word in words:
        cand = " ".join(cur + [word])
        bbox = draw.textbbox((0, 0), cand, font=font)
        if bbox[2] - bbox[0] <= max_w or not cur:
            cur.append(word)
        else:
            lines.append(" ".join(cur))
            cur = [word]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _render_text_page(
    w: int,
    h: int,
    rng: random.Random,
    np_rng: np.random.Generator,
    font_paths: List[str],
    aged: bool,
    subtype: str,
) -> Image.Image:
    img = _make_paper_canvas(w, h, rng, np_rng, aged=aged)
    draw = ImageDraw.Draw(img)

    ink_v = rng.randint(35, 95) if aged else rng.randint(12, 55)
    ink = (ink_v, max(0, ink_v - rng.randint(0, 6)), max(0, ink_v - rng.randint(0, 12)))
    margin_x = int(w * rng.uniform(0.10, 0.15))
    max_w = w - 2 * margin_x

    # Optional decorative border frame
    if rng.random() < 0.45:
        b0 = int(min(w, h) * 0.05)
        draw.rectangle((b0, b0, w - b0, h - b0), outline=ink, width=rng.choice([2, 3]))
        if rng.random() < 0.6:
            b1 = b0 + rng.randint(8, 16)
            draw.rectangle((b1, b1, w - b1, h - b1), outline=ink, width=1)

    if subtype == "title_page":
        f_comp = _load_font(font_paths, rng, size=rng.randint(34, 48))
        f_title = _load_font(font_paths, rng, size=rng.randint(52, 76))
        f_sub = _load_font(font_paths, rng, size=rng.randint(26, 36))

        comp = rng.choice(COMPOSERS)
        title = rng.choice(TITLES)
        opus = f"Op. {rng.randint(1, 135)}, No. {rng.randint(1, 6)}"
        pub = rng.choice([
            "LEIPZIG: BREITKOPF & HARTEL",
            "PARIS: DURAND & FILS, EDITEURS",
            "WIEN: UNIVERSAL EDITION",
            "LONDON: NOVELLO & CO., LTD.",
            "BERLIN: N. SIMROCK",
        ])

        def _draw_centered(y_pos: int, txt: str, fnt: ImageFont.ImageFont) -> None:
            bb = draw.textbbox((0, 0), txt, font=fnt)
            tw = bb[2] - bb[0]
            draw.text(((w - tw) // 2, y_pos), txt, fill=ink, font=fnt)

        _draw_centered(int(h * 0.18), comp.upper(), f_comp)
        if rng.random() < 0.75:
            y_rule = int(h * 0.25)
            draw.line((int(w * 0.28), y_rule, int(w * 0.72), y_rule), fill=ink, width=2)
            draw.line((int(w * 0.34), y_rule + 8, int(w * 0.66), y_rule + 8), fill=ink, width=1)
        _draw_centered(int(h * 0.34), title, f_title)
        _draw_centered(int(h * 0.44), opus, f_sub)
        if rng.random() < 0.75:
            y_rule2 = int(h * 0.54)
            draw.line((int(w * 0.22), y_rule2, int(w * 0.78), y_rule2), fill=ink, width=2)
        _draw_centered(int(h * 0.78), pub, f_sub)
        _draw_centered(int(h * 0.84), f"Plate No. {rng.randint(1000, 9999)}", f_sub)

    elif subtype == "lyrics":
        f_hdr = _load_font(font_paths, rng, size=rng.randint(36, 50))
        f_body = _load_font(font_paths, rng, size=rng.randint(24, 34))
        y = int(h * 0.10)
        hdr = rng.choice(TITLES)
        draw.text((margin_x, y), hdr, fill=ink, font=f_hdr)
        y += 64
        if rng.random() < 0.7:
            draw.line((margin_x, y, w - margin_x, y), fill=ink, width=2)
            y += 30
        stanzas = list(LYRIC_STANZAS)
        rng.shuffle(stanzas)
        for st in stanzas[: rng.randint(3, 4)]:
            for line in st:
                draw.text((margin_x + 20, y), line, fill=ink, font=f_body)
                y += 42
            y += 36
            if y > int(h * 0.88):
                break
    else:
        # Program notes / critical commentary
        f_hdr = _load_font(font_paths, rng, size=rng.randint(36, 48))
        f_body = _load_font(font_paths, rng, size=rng.randint(22, 30))
        y = int(h * 0.09)
        hdr = rng.choice(["PROGRAM NOTES", "PREFACE", "EDITORIAL COMMENTARY", "CRITICAL REPORT", "PERFORMANCE NOTES"])
        draw.text((margin_x, y), hdr, fill=ink, font=f_hdr)
        y += 62
        if rng.random() < 0.8:
            draw.line((margin_x, y, w - margin_x, y), fill=ink, width=2)
            y += 28
        paras = list(PROGRAM_PARAGRAPHS)
        rng.shuffle(paras)
        for para in paras[: rng.randint(3, 5)]:
            for line in _wrap_text(draw, para, f_body, max_w):
                draw.text((margin_x, y), line, fill=ink, font=f_body)
                y += 38
                if y > int(h * 0.88):
                    break
            y += 26
            if y > int(h * 0.88):
                break

    return img


def _render_ruled_or_blank(
    w: int,
    h: int,
    rng: random.Random,
    np_rng: np.random.Generator,
    aged: bool,
    ruled: bool,
) -> Image.Image:
    img = _make_paper_canvas(w, h, rng, np_rng, aged=aged)
    if ruled:
        draw = ImageDraw.Draw(img)
        col_v = rng.randint(140, 195)
        line_col = (col_v, col_v, min(220, col_v + 15))
        step = rng.randint(36, 60)
        for y in range(int(h * 0.08), int(h * 0.93), step):
            draw.line((int(w * 0.06), y, int(w * 0.94), y), fill=line_col, width=1)
        if rng.random() < 0.5:
            mx = int(w * 0.14)
            draw.line((mx, int(h * 0.05), mx, int(h * 0.95)), fill=(195, 130, 130), width=1)
    return img


def generate_neg_controls(out_dir: Path, count: int = 250, seed: int = 20261127) -> Dict[str, int]:
    t0 = time.time()
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Shared 0-byte empty bekern target -> data.load_bekern() returns [] -> [BOS, EOS]
    empty_bek = out_dir / "empty-bekern.txt"
    empty_bek.write_text("", encoding="utf-8")

    font_paths = _discover_fonts()
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Plan exact category mix for 250 pages (scales cleanly if count != 250)
    n_staves = int(round(count * 0.40))   # 100
    n_text = int(round(count * 0.40))     # 100
    n_aged_mix = count - n_staves - n_text  # 50

    plan: List[Tuple[str, bool, bool]] = []
    # (neg_type, aged, heavy_photo)
    for i in range(n_staves):
        aged = i >= (n_staves // 2)
        plan.append(("blank_staves", aged, aged and (i % 3 == 0)))
    for i in range(n_text):
        sub = ("program_notes", "lyrics", "title_page")[i % 3]
        aged = i >= (n_text // 2)
        plan.append((sub, aged, aged and (i % 3 == 0)))
    for i in range(n_aged_mix):
        kind = ("blank_staves_heavy", "title_page_heavy", "ruled_paper", "blank_parchment")[i % 4]
        plan.append((kind, True, True))

    rng.shuffle(plan)

    manifest_rows: List[Dict[str, object]] = []
    type_counts: Dict[str, int] = {}

    for idx, (neg_type, aged, heavy_photo) in enumerate(plan):
        # Portrait ~85%, landscape ~15%
        if rng.random() < 0.15:
            w, h = rng.choice([(1650, 1200), (1800, 1300), (1500, 1100)])
        else:
            w, h = rng.choice([(1200, 1650), (1300, 1800), (1400, 1950), (1240, 1754)])

        if neg_type in ("blank_staves", "blank_staves_heavy"):
            img = _render_blank_staves(w, h, rng, np_rng, font_paths, aged=aged)
        elif neg_type in ("program_notes", "lyrics", "title_page", "title_page_heavy"):
            sub = "title_page" if neg_type == "title_page_heavy" else neg_type
            img = _render_text_page(w, h, rng, np_rng, font_paths, aged=aged, subtype=sub)
        elif neg_type == "ruled_paper":
            img = _render_ruled_or_blank(w, h, rng, np_rng, aged=True, ruled=True)
        else:
            img = _render_ruled_or_blank(w, h, rng, np_rng, aged=True, ruled=False)

        ext = ".png"
        if aged or heavy_photo:
            img, ext = _apply_photo_finish(img, rng, np_rng, heavy=heavy_photo)

        img_path = pages_dir / f"neg_ctrl_{idx:03d}{ext}"
        if ext == ".jpg":
            img.save(img_path, format="JPEG", quality=rng.randint(75, 92))
        else:
            img.save(img_path, format="PNG", compress_level=1)

        manifest_rows.append(
            {
                "image": str(img_path),
                "bekerns": [str(empty_bek)],
                "domain": "neg_control",
                "bucket": "neg_control",
                "neg_type": neg_type,
                "aged": bool(aged),
            }
        )
        type_counts[neg_type] = type_counts.get(neg_type, 0) + 1

    manifest_path = out_dir / "manifest.jsonl"
    tmp_path = manifest_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, manifest_path)

    elapsed = time.time() - t0
    print(
        f"[neg-controls] Generated {len(manifest_rows)} pages -> {manifest_path} ({elapsed:.1f}s)",
        flush=True,
    )
    print(f"[neg-controls] Breakdown: {dict(sorted(type_counts.items()))}", flush=True)
    return type_counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic negative-control pages for Iter 14 (CPU-only).")
    ap.add_argument("--out-dir", type=Path, default=Path(r"D:\tmp\neg-controls-iter14"))
    ap.add_argument("--count", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261127)
    args = ap.parse_args()
    generate_neg_controls(args.out_dir, count=args.count, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
