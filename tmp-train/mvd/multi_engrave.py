"""Multi-engraver MusicXML renderer & system-strip extractor (Verovio + LilyPond).

Breaks single-engraver (MuseScore-only) bias by rendering the same MusicXML
scores through independent layout/engraving engines:

1. **Verovio** (Python ``verovio`` C++ toolkit + ``pymupdf`` SVG->PNG):
   Supports 5 built-in SMuFL fonts (``Leipzig``, ``Bravura``, ``Gootville``,
   ``Petaluma``, ``Leland``) and extracts **engraver-exact** system boxes and
   per-system measure spans directly from Verovio's SVG DOM (``<g class="system">``,
   ``<g class="measure">``, ``<g class="staff">``).
2. **LilyPond** (``musicxml2ly`` -> ``lilypond --png -dresolution=300``):
   Gold-standard classical engraving look (closest to Henle/Peters/IMSLP plates)
   with automatic binary discovery on Windows (PATH + WinGet install dir) and
   projection-profile + staff-line system-strip extraction.

All outputs use atomic ``.tmp`` + ``os.replace`` writes and deterministic JSON
sidecars recording engraver, font, page dimensions, and system strip boxes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

VEROVIO_FONTS = ('Leipzig', 'Bravura', 'Gootville', 'Petaluma', 'Leland')


def _atomic_save_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.png')
    img.save(tmp, format='PNG')
    os.replace(tmp, path)


def _atomic_write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def svg_to_png(svg_text: str, width_px: int = 2480) -> Image.Image:
    """Rasterize an SVG string to a white-background RGB PIL image via resvg."""
    import io
    import resvg_py

    png_bytes = resvg_py.svg_to_bytes(svg_string=svg_text, width=int(width_px))
    rgba = Image.open(io.BytesIO(png_bytes)).convert('RGBA')
    bg = Image.new('RGB', rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def _parse_verovio_svg_systems(
    svg_text: str,
    img_w: int,
    img_h: int,
    pad_y_sp: float = 3.0,
    pad_x_sp: float = 1.5,
) -> list[dict]:
    """Extract engraver-exact system boxes & measure numbers from Verovio SVG.

    Uses the explicit staff-line ``<path>`` y-coordinates inside each
    ``<g class="system">`` to measure exact staff spacing (``sp``) and
    apply the same tight ``[y1 - 2.5*sp, y2 + 2.5*sp]`` / ``[x1 - 1.5*sp, x2 + 1.5*sp]``
    crop parity used by ``build_sys_data.py``.
    """
    root = ET.fromstring(svg_text)
    vx0 = vy0 = 0.0
    vw = vh = 0.0
    # Verovio puts the coordinate viewBox on the inner <svg class="definition-scale">
    # (outer <svg> may omit viewBox). Prefer inner viewBox, fall back to root.
    for el in root.iter():
        if el.tag.endswith('svg'):
            ivb = el.attrib.get('viewBox', '')
            ip = [float(v) for v in re.split(r'[\s,]+', ivb.strip()) if v]
            if len(ip) == 4 and ip[2] > 0 and ip[3] > 0:
                vx0, vy0, vw, vh = ip
                if el is not root:
                    break
    if vw <= 0 or vh <= 0:
        return []

    # Accumulate page-margin translate(tx, ty) if present.
    off_x = off_y = 0.0
    for el in root.iter():
        if el.tag.endswith('g') and 'page-margin' in el.attrib.get('class', '').split():
            tr = el.attrib.get('transform', '')
            tm = re.search(r'translate\(\s*(-?\d+(?:\.\d+)?)\s*[\s,]\s*(-?\d+(?:\.\d+)?)\s*\)', tr)
            if tm:
                off_x += float(tm.group(1))
                off_y += float(tm.group(2))
            break

    sx = float(img_w) / vw
    sy = float(img_h) / vh

    def _collect_staff_lines(sys_el) -> tuple[list[list[float]], float, float]:
        """Return (list of 5-line y lists in px, min_x_px, max_x_px)."""
        staves_by_n: dict[str, list[float]] = {}
        min_x, max_x = float('inf'), float('-inf')
        for el in sys_el.iter():
            cls = el.attrib.get('class', '')
            if el.tag.endswith('g') and 'staff' in cls.split():
                ys = []
                for child in el:
                    if child.tag.endswith('path'):
                        d = child.attrib.get('d', '')
                        m = re.match(
                            r'M\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+'
                            r'L\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)',
                            d.strip(),
                        )
                        if m:
                            x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
                            if abs(y1 - y2) < 1e-2 and abs(x2 - x1) > 10.0:
                                ys.append((y1 + off_y - vy0) * sy)
                                min_x = min(min_x, (min(x1, x2) + off_x - vx0) * sx)
                                max_x = max(max_x, (max(x1, x2) + off_x - vx0) * sx)
                if len(ys) == 5:
                    ys.sort()
                    bucket = str(round(ys[0] / 8.0))
                    if bucket not in staves_by_n:
                        staves_by_n[bucket] = ys
        staves = sorted(staves_by_n.values(), key=lambda s: s[0])
        return staves, min_x, max_x

    systems = []
    for el in root.iter():
        if el.tag.endswith('g') and 'system' in el.attrib.get('class', '').split():
            staves, min_x, max_x = _collect_staff_lines(el)
            if not staves:
                continue
            spacings = [
                (s[k + 1] - s[k]) for s in staves for k in range(4)
                if s[k + 1] > s[k]
            ]
            sp = float(np.median(spacings)) if spacings else 12.0
            y1 = max(0, int(round(staves[0][0] - pad_y_sp * sp)))
            y2 = min(img_h, int(round(staves[-1][-1] + pad_y_sp * sp)))
            x1 = max(0, int(round(min_x - pad_x_sp * sp))) if np.isfinite(min_x) else 0
            x2 = min(img_w, int(round(max_x + pad_x_sp * sp))) if np.isfinite(max_x) else img_w
            # Count measures in this system.
            n_meas = sum(
                1 for sub in el.iter()
                if sub.tag.endswith('g') and 'measure' in sub.attrib.get('class', '').split()
            )
            systems.append({
                'bbox': [x1, y1, x2, y2],
                'staves': [[round(v, 2) for v in s] for s in staves],
                'staff_space_px': round(sp, 2),
                'n_staves': len(staves),
                'n_measures': n_meas,
            })
    return systems


def render_verovio(
    musicxml_path: str | Path,
    out_dir: str | Path,
    font: str = 'Leipzig',
    width_px: int = 2480,
    extract_strips: bool = True,
    max_pages: int | None = None,
) -> dict:
    """Render a MusicXML file with Verovio into PNG pages + optional system strips."""
    import verovio

    xml_path = Path(musicxml_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{xml_path.stem}__vrv_{font}'

    tk = verovio.toolkit()
    tk.setOptions({
        'font': font,
        'pageWidth': 2100,
        'pageHeight': 2970,
        'scale': 40,
        'adjustPageHeight': False,
        'footer': 'none',
        'header': 'none',
        'breaks': 'auto',
    })
    ok = tk.loadFile(str(xml_path))
    if not ok:
        raise RuntimeError(f'verovio failed to load {xml_path}')
    n_pages = tk.getPageCount()
    if max_pages is not None:
        n_pages = min(n_pages, int(max_pages))

    pages_meta = []
    for p in range(1, n_pages + 1):
        svg = tk.renderToSVG(p)
        img = svg_to_png(svg, width_px=width_px)
        page_png = out / f'{stem}-{p}.png'
        _atomic_save_png(img, page_png)
        systems = _parse_verovio_svg_systems(svg, img.width, img.height)
        strips = []
        if extract_strips:
            for si, sys_info in enumerate(systems):
                x1, y1, x2, y2 = sys_info['bbox']
                if x2 > x1 + 16 and y2 > y1 + 16:
                    crop = img.crop((x1, y1, x2, y2))
                    strip_png = out / f'{stem}-{p}-s{si}.png'
                    _atomic_save_png(crop, strip_png)
                    strips.append({
                        'strip_png': strip_png.name,
                        'width': crop.width,
                        'height': crop.height,
                        **sys_info,
                    })
        pages_meta.append({
            'page': p,
            'png': page_png.name,
            'width': img.width,
            'height': img.height,
            'systems': strips if extract_strips else systems,
        })

    meta = {
        'engraver': 'verovio',
        'verovio_version': tk.getVersion(),
        'font': font,
        'source': str(xml_path),
        'pages': pages_meta,
    }
    _atomic_write_json(meta, out / f'{stem}_meta.json')
    return meta


def find_lilypond_bins() -> tuple[Path | None, Path | None]:
    """Locate ``(lilypond, musicxml2ly)`` executables on PATH or WinGet."""
    lp = shutil.which('lilypond')
    m2l = shutil.which('musicxml2ly')
    if lp and m2l:
        return Path(lp), Path(m2l)
    # Check standard Windows WinGet / Program Files locations.
    roots = [
        Path(os.environ.get('LOCALAPPDATA', '')) / 'Microsoft' / 'WinGet' / 'Packages',
        Path(os.environ.get('LOCALAPPDATA', '')) / 'Microsoft' / 'WinGet' / 'Links',
        Path(r'C:\Program Files'),
        Path(r'C:\Program Files (x86)'),
    ]
    for r in roots:
        if not r.exists():
            continue
        for hit in r.glob('**/lilypond.exe'):
            bin_dir = hit.parent
            cand_m2l = bin_dir / 'musicxml2ly.py'
            if not cand_m2l.exists():
                cand_m2l = bin_dir / 'musicxml2ly'
            return hit, (cand_m2l if cand_m2l.exists() else None)
    return (Path(lp) if lp else None), (Path(m2l) if m2l else None)


def _extract_strips_from_clean_png(
    img: Image.Image,
    pad_y_sp: float = 2.5,
    pad_x_sp: float = 1.5,
) -> list[dict]:
    """Detect 5-line staves & group into systems on a clean rendered page PNG."""
    gray = np.asarray(img.convert('L'), dtype=np.float32)
    h, w = gray.shape
    ink = (gray < 180.0).astype(np.float32)
    row_dens = ink.mean(axis=1)

    # Staff lines are sharp horizontal ink spikes (>22% page width).
    line_rows = np.where(row_dens > 0.22)[0]
    if line_rows.size < 5:
        return []
    # Merge adjacent rows belonging to the same line thickness.
    lines: list[float] = []
    start = int(line_rows[0])
    prev = start
    for r in line_rows[1:]:
        ri = int(r)
        if ri > prev + 2:
            lines.append(0.5 * (start + prev))
            start = ri
        prev = ri
    lines.append(0.5 * (start + prev))

    # Group consecutive lines into 5-line staves via median small gap.
    if len(lines) < 5:
        return []
    diffs = np.diff(lines)
    small = diffs[diffs < np.percentile(diffs, 75) * 1.4]
    sp = float(np.median(small)) if small.size else 12.0

    staves: list[list[float]] = []
    cur = [lines[0]]
    for y in lines[1:]:
        if abs((y - cur[-1]) - sp) <= max(3.5, 0.35 * sp) and len(cur) < 5:
            cur.append(y)
        else:
            if len(cur) == 5:
                staves.append(cur)
            cur = [y]
    if len(cur) == 5:
        staves.append(cur)
    if not staves:
        return []

    # Group staves into systems by checking vertical ink continuity on the left
    # system barline/brace (within the leftmost staff ink column +/- 3*sp).
    col_dens = ink.mean(axis=0)
    ink_cols = np.where(col_dens > 0.005)[0]
    gx1 = int(ink_cols[0]) if ink_cols.size else 0
    gx2 = int(ink_cols[-1]) if ink_cols.size else w - 1

    # Find left edge of staves using staff rows.
    staff_rows_mask = np.zeros(h, dtype=bool)
    for s in staves:
        for y in s:
            r0 = max(0, int(round(y)) - 1)
            r1 = min(h, int(round(y)) + 2)
            staff_rows_mask[r0:r1] = True
    staff_cols = np.where(ink[staff_rows_mask].mean(axis=0) > 0.35)[0]
    sx1 = int(staff_cols[0]) if staff_cols.size else gx1
    sx2 = int(staff_cols[-1]) if staff_cols.size else gx2

    systems_staves: list[list[list[float]]] = [[staves[0]]]
    if len(staves) > 1:
        # Compute all consecutive inter-staff gaps (bottom of prev -> top of next).
        gaps = [staves[i + 1][0] - staves[i][-1] for i in range(len(staves) - 1)]
        # Bimodal split: inter-system gaps are significantly larger than
        # within-system gaps (e.g. Voice<->Piano or Piano RH<->LH).
        # Also detect continuous system barlines at sx1.
        connectors = []
        for i in range(len(staves) - 1):
            y_a = int(round(staves[i][-1]))
            y_b = int(round(staves[i + 1][0]))
            band = ink[max(0, y_a):min(h, y_b), max(0, sx1 - 6):min(w, sx1 + 12)]
            connectors.append(band.size > 0 and float(band.max(axis=1).mean()) > 0.70)

        max_gap = max(gaps)
        min_gap = min(gaps)
        # If there is a clear bimodal jump (inter-system > 1.45x within-system),
        # split at the midpoint between the two clusters; otherwise use 10*sp.
        if max_gap > 1.45 * max(min_gap, 1.0):
            # Find largest gap jump in sorted gaps for robust 1D k=2 split.
            sg = sorted(gaps)
            best_jump, split_thr = 0.0, 0.5 * (min_gap + max_gap)
            for k in range(len(sg) - 1):
                jump = sg[k + 1] - sg[k]
                if jump > best_jump:
                    best_jump = jump
                    split_thr = 0.5 * (sg[k] + sg[k + 1])
        else:
            split_thr = 10.0 * sp

        for i, s in enumerate(staves[1:]):
            if connectors[i] or gaps[i] <= split_thr:
                systems_staves[-1].append(s)
            else:
                systems_staves.append([s])

    out = []
    for group in systems_staves:
        y1 = max(0, int(round(group[0][0] - max(pad_y_sp, 3.0) * sp)))
        y2 = min(h, int(round(group[-1][-1] + max(pad_y_sp, 3.0) * sp)))
        x1 = max(0, int(round(min(gx1, sx1) - pad_x_sp * sp)))
        x2 = min(w, int(round(max(gx2, sx2) + pad_x_sp * sp)))
        out.append({
            'bbox': [x1, y1, x2, y2],
            'staves': [[round(v, 2) for v in s] for s in group],
            'staff_space_px': round(sp, 2),
            'n_staves': len(group),
        })
    return out


def render_lilypond(
    musicxml_path: str | Path,
    out_dir: str | Path,
    dpi: int = 300,
    extract_strips: bool = True,
    max_pages: int | None = None,
) -> dict:
    """Render a MusicXML file via ``musicxml2ly`` + ``lilypond --png``."""
    lp_bin, m2l_bin = find_lilypond_bins()
    if lp_bin is None or m2l_bin is None:
        raise FileNotFoundError(
            f'LilyPond binaries not found (lilypond={lp_bin}, musicxml2ly={m2l_bin})'
        )

    xml_path = Path(musicxml_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{xml_path.stem}__lp'

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        ly_path = td_path / 'score.ly'
        # musicxml2ly is a Python script on Windows; run via lilypond's bundled
        # python.exe if present, else current sys.executable.
        bundled_py = lp_bin.parent / 'python.exe'
        py_exe = str(bundled_py if bundled_py.exists() else Path(os.sys.executable))
        m2l_cmd = ([py_exe, str(m2l_bin)] if m2l_bin.suffix == '.py' or not m2l_bin.suffix
                   else [str(m2l_bin)])
        subprocess.run(
            m2l_cmd + ['--no-articulation-directions', '-o', str(ly_path), str(xml_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        # Disable LilyPond tagline so pages stay clean notation.
        ly_txt = ly_path.read_text(encoding='utf-8', errors='replace')
        ly_txt += '\n\\header { tagline = ##f }\n'
        ly_path.write_text(ly_txt, encoding='utf-8')

        prefix = td_path / 'page'
        subprocess.run(
            [str(lp_bin), '--png', f'-dresolution={int(dpi)}',
             '-o', str(prefix), str(ly_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        # LilyPond emits page.png (single page) or page-page1.png, page-page2.png...
        raw_pngs = sorted(
            td_path.glob('page*.png'),
            key=lambda p: (
                int(re.search(r'(\d+)$', p.stem).group(1))
                if re.search(r'(\d+)$', p.stem) else 0
            ),
        )
        if max_pages is not None:
            raw_pngs = raw_pngs[:int(max_pages)]

        pages_meta = []
        for idx, rp in enumerate(raw_pngs, start=1):
            img = Image.open(rp).convert('RGB')
            page_png = out / f'{stem}-{idx}.png'
            _atomic_save_png(img, page_png)
            systems = _extract_strips_from_clean_png(img)
            strips = []
            if extract_strips:
                for si, sys_info in enumerate(systems):
                    x1, y1, x2, y2 = sys_info['bbox']
                    if x2 > x1 + 16 and y2 > y1 + 16:
                        crop = img.crop((x1, y1, x2, y2))
                        strip_png = out / f'{stem}-{idx}-s{si}.png'
                        _atomic_save_png(crop, strip_png)
                        strips.append({
                            'strip_png': strip_png.name,
                            'width': crop.width,
                            'height': crop.height,
                            **sys_info,
                        })
            pages_meta.append({
                'page': idx,
                'png': page_png.name,
                'width': img.width,
                'height': img.height,
                'systems': strips if extract_strips else systems,
            })

    meta = {
        'engraver': 'lilypond',
        'font': 'Emmentaler',
        'dpi': dpi,
        'source': str(xml_path),
        'pages': pages_meta,
    }
    _atomic_write_json(meta, out / f'{stem}_meta.json')
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description='Multi-engraver MusicXML renderer')
    ap.add_argument('--input', required=True, help='MusicXML file')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--engravers', default='verovio,lilypond')
    ap.add_argument('--verovio-fonts', default='Leipzig,Bravura,Petaluma')
    ap.add_argument('--max-pages', type=int, default=None)
    ap.add_argument('--no-strips', action='store_true')
    args = ap.parse_args()

    engs = [e.strip().lower() for e in args.engravers.split(',') if e.strip()]
    vfonts = [f.strip() for f in args.verovio_fonts.split(',') if f.strip()]
    summary = []
    if 'verovio' in engs:
        for f in vfonts:
            m = render_verovio(args.input, args.out, font=f,
                               extract_strips=not args.no_strips,
                               max_pages=args.max_pages)
            n_strips = sum(len(p['systems']) for p in m['pages'])
            summary.append(f'verovio/{f}: {len(m["pages"])}p, {n_strips} strips')
    if 'lilypond' in engs:
        m = render_lilypond(args.input, args.out,
                            extract_strips=not args.no_strips,
                            max_pages=args.max_pages)
        n_strips = sum(len(p['systems']) for p in m['pages'])
        summary.append(f'lilypond/Emmentaler: {len(m["pages"])}p, {n_strips} strips')
    print(' | '.join(summary))


if __name__ == '__main__':
    main()
