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
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

VEROVIO_FONTS = ('Leipzig', 'Bravura', 'Gootville', 'Petaluma', 'Leland')
MIN_BEKERN_TOKENS = 1
MAX_BEKERN_TOKENS = 512

# LilyPond Scheme hook injected before \score to report the exact 1-based bar
# number starting every engraved system.
_LILYPOND_SYS_BREAK_HOOK = r"""
\layout {
  \context {
    \Score
    \override NonMusicalPaperColumn.after-line-breaking = #(lambda (grob)
      (let ((bd (ly:item-break-dir grob))
            (rl (ly:grob-property grob 'rhythmic-location #f)))
        (when (and (eq? bd 1) (pair? rl))
          (format (current-error-port) "SYS_START ~a~%" (car rl)))))
  }
}
"""


def default_assembly_serialize_bin() -> Path:
    """Locate the release ``assembly-serialize`` binary relative to repo root."""
    repo = Path(__file__).resolve().parents[2]
    exe = 'assembly-serialize.exe' if os.name == 'nt' else 'assembly-serialize'
    return repo / 'converter' / 'assembly-serialize' / 'target' / 'release' / exe


def _table_npts(w_in: int, stride: int = 8) -> int:
    """Control-point count matching ``build_hybrid_cache.rows_to_table``."""
    n = len(range(0, int(w_in), int(stride)))
    return n if (int(w_in) - 1) % int(stride) == 0 else n + 1


def rows_to_table(lines5: list[list[float]], w_in: int, stride: int = 8) -> dict:
    """Convert strip-local 5-line staff rows to ``hybrid_staff.jsonl`` table."""
    n = _table_npts(int(w_in), stride=stride)
    return {
        'stride': int(stride),
        'w_in': int(w_in),
        'staves': [[[round(float(y), 2)] * n for y in staff] for staff in lines5],
    }


def make_staff_jsonl_row(
    strip_png_name: str,
    page_staves: list[list[float]],
    crop_y1: int,
    strip_w: int,
    sp: float,
    source: str,
    skip: bool = False,
    reason: str | None = None,
) -> dict:
    """Build a ``hybrid_staff.jsonl`` row in strip-local pixel coordinates."""
    if skip:
        row = {'img': strip_png_name, 'skip': True, 'source': source}
        if reason:
            row['reason'] = reason
        return row
    local_lines = [
        [round(float(y - crop_y1), 2) for y in staff]
        for staff in page_staves
    ]
    return {
        'img': strip_png_name,
        'S': len(local_lines),
        'kind': 'rows',
        'table': rows_to_table(local_lines, strip_w),
        'sp': round(float(sp), 2),
        'source': source,
        'skip': False,
    }


def _upsert_staff_jsonl(jsonl_path: Path, new_rows: list[dict]) -> None:
    """Atomically merge ``new_rows`` (keyed by ``img``) into ``hybrid_staff.jsonl``."""
    by_img: dict[str, dict] = {}
    if jsonl_path.exists():
        for ln in jsonl_path.read_text(encoding='utf-8').splitlines():
            if ln.strip():
                r = json.loads(ln)
                if 'img' in r:
                    by_img[r['img']] = r
    for r in new_rows:
        if 'img' in r:
            by_img[r['img']] = r
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = jsonl_path.with_suffix(jsonl_path.suffix + '.tmp')
    lines = [json.dumps(by_img[k], separators=(',', ':')) for k in sorted(by_img)]
    tmp.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    os.replace(tmp, jsonl_path)


def get_musicxml_system_breaks(
    musicxml_path: str | Path,
    aser_bin: str | Path | None = None,
) -> list[dict]:
    """Run ``assembly-serialize system-breaks`` and return engraved break spans."""
    exe = Path(aser_bin) if aser_bin else default_assembly_serialize_bin()
    if not exe.exists():
        raise FileNotFoundError(f'assembly-serialize binary not found: {exe}')
    r = subprocess.run(
        [str(exe), 'system-breaks', '--input', str(musicxml_path)],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f'system-breaks failed on {musicxml_path}: {(r.stderr or r.stdout).strip()}'
        )
    return json.loads(r.stdout)


def _total_measures_from_breaks(
    musicxml_path: str | Path,
    aser_bin: str | Path | None = None,
) -> tuple[int, list[dict]]:
    """Return ``(total_measures, system_breaks)`` via ``assembly-serialize``."""
    breaks = get_musicxml_system_breaks(musicxml_path, aser_bin=aser_bin)
    if breaks:
        total = max(int(b['end_measure']) for b in breaks)
        return total, breaks
    # Fallback: count <measure> tags in first <part>.
    root = ET.parse(str(musicxml_path)).getroot()
    part = root.find('part')
    total = len(part.findall('measure')) if part is not None else 0
    return total, breaks


def emit_and_validate_bekern(
    musicxml_path: str | Path,
    start_measure: int,
    end_measure: int,
    out_dir: str | Path,
    stem: str,
    aser_bin: str | Path | None = None,
    min_tokens: int = MIN_BEKERN_TOKENS,
    max_tokens: int = MAX_BEKERN_TOKENS,
) -> tuple[Path | None, int, str | None]:
    """Emit ``<stem>-bekern.txt`` via ``bekern --measures`` + ``bekern-decode`` gate.

    Writes atomically via a temporary directory + ``os.replace`` only when
    both token count and ``bekern-decode`` round-trip validation pass.
    Returns ``(bekern_path_or_None, n_tokens, error_or_None)``.
    """
    exe = Path(aser_bin) if aser_bin else default_assembly_serialize_bin()
    if not exe.exists():
        return None, 0, f'missing-aser:{exe}'
    if start_measure < 1 or end_measure < start_measure:
        return None, 0, f'invalid-span:{start_measure}..{end_measure}'

    out_path = Path(out_dir) / f'{stem}-bekern.txt'
    span_arg = f'{int(start_measure)}..{int(end_measure)}'
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tmp_txt = td_path / f'{stem}-bekern.txt'
        r1 = subprocess.run(
            [str(exe), 'bekern', '--input', str(musicxml_path),
             '--measures', span_arg, '--out', str(td_path), '--stem', stem],
            capture_output=True, text=True, check=False,
        )
        if r1.returncode != 0 or not tmp_txt.exists():
            return None, 0, f'bekern:{(r1.stderr or r1.stdout).strip()[-160:]}'
        raw = tmp_txt.read_text(encoding='utf-8')
        tokens = raw.split()
        if not (min_tokens <= len(tokens) <= max_tokens):
            return None, len(tokens), f'tokens:{len(tokens)}'
        dec_json = td_path / 'decoded.json'
        r2 = subprocess.run(
            [str(exe), 'bekern-decode', '--input', str(tmp_txt),
             '--out', str(dec_json)],
            capture_output=True, text=True, check=False,
        )
        if r2.returncode != 0:
            return None, len(tokens), f'decode:{(r2.stderr or r2.stdout).strip()[-160:]}'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        staged = out_path.with_suffix('.txt.tmp')
        staged.write_text(raw, encoding='utf-8')
        os.replace(staged, out_path)
    return out_path, len(tokens), None


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
    emit_bekern: bool = True,
    write_staff_jsonl: bool = True,
    aser_bin: str | Path | None = None,
) -> dict:
    """Render a MusicXML file with Verovio into PNG pages + training-ready strips.

    When ``extract_strips=True`` and ``emit_bekern=True``, each system strip is
    paired with an engraver-exact ``<strip_stem>-bekern.txt`` (validated via
    ``assembly-serialize bekern --measures`` + ``bekern-decode``) and a
    ``hybrid_staff.jsonl`` row in strip-local coordinates.
    """
    import verovio

    xml_path = Path(musicxml_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{xml_path.stem}__vrv_{font}'

    total_measures = 0
    if extract_strips and emit_bekern:
        total_measures, _ = _total_measures_from_breaks(xml_path, aser_bin=aser_bin)

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
    staff_rows: list[dict] = []
    cur_measure = 1
    for p in range(1, n_pages + 1):
        svg = tk.renderToSVG(p)
        img = svg_to_png(svg, width_px=width_px)
        page_png = out / f'{stem}-{p}.png'
        _atomic_save_png(img, page_png)
        systems = _parse_verovio_svg_systems(svg, img.width, img.height)
        strips = []
        for si, sys_info in enumerate(systems):
            n_meas = int(sys_info.get('n_measures', 0))
            start_m = cur_measure if n_meas > 0 else 0
            end_m = (cur_measure + n_meas - 1) if n_meas > 0 else 0
            if n_meas > 0:
                cur_measure = end_m + 1
            if total_measures > 0 and end_m > total_measures:
                end_m = total_measures

            if not extract_strips:
                sys_info['start_measure'] = start_m
                sys_info['end_measure'] = end_m
                continue

            x1, y1, x2, y2 = sys_info['bbox']
            if x2 > x1 + 16 and y2 > y1 + 16:
                crop = img.crop((x1, y1, x2, y2))
                strip_stem = f'{stem}-{p}-s{si}'
                strip_png = out / f'{strip_stem}.png'
                _atomic_save_png(crop, strip_png)

                bek_name: str | None = None
                n_tok = 0
                bek_err: str | None = None
                if emit_bekern:
                    if 1 <= start_m <= end_m:
                        bek_path, n_tok, bek_err = emit_and_validate_bekern(
                            xml_path, start_m, end_m, out, strip_stem,
                            aser_bin=aser_bin,
                        )
                        if bek_path is not None:
                            bek_name = bek_path.name
                    else:
                        bek_err = f'invalid-span:{start_m}..{end_m}'

                skip_row = bool(emit_bekern and bek_name is None)
                srow = make_staff_jsonl_row(
                    strip_png_name=strip_png.name,
                    page_staves=sys_info['staves'],
                    crop_y1=y1,
                    strip_w=crop.width,
                    sp=float(sys_info['staff_space_px']),
                    source=f'verovio-{font.lower()}',
                    skip=skip_row,
                    reason=(f'bekern:{bek_err}' if skip_row and bek_err else None),
                )
                staff_rows.append(srow)
                strips.append({
                    'strip_png': strip_png.name,
                    'width': crop.width,
                    'height': crop.height,
                    'start_measure': start_m,
                    'end_measure': end_m,
                    'bekern_txt': bek_name,
                    'n_tokens': n_tok,
                    'bekern_error': bek_err,
                    'staff_row': srow,
                    **sys_info,
                })
        pages_meta.append({
            'page': p,
            'png': page_png.name,
            'width': img.width,
            'height': img.height,
            'systems': strips if extract_strips else systems,
        })

    if extract_strips and write_staff_jsonl and staff_rows:
        _upsert_staff_jsonl(out / 'hybrid_staff.jsonl', staff_rows)

    meta = {
        'engraver': 'verovio',
        'verovio_version': tk.getVersion(),
        'font': font,
        'source': str(xml_path),
        'staff_jsonl': 'hybrid_staff.jsonl' if (extract_strips and write_staff_jsonl and staff_rows) else None,
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
        # split at the midpoint between the two clusters. Otherwise, if no
        # adjacent staves are joined by a left system barline (e.g. a 1-staff
        # melody score where all gaps are equal inter-system gaps), each staff
        # is its own system (split_thr = 0.0); else fall back to 10*sp.
        if max_gap > 1.45 * max(min_gap, 1.0):
            # Find largest gap jump in sorted gaps for robust 1D k=2 split.
            sg = sorted(gaps)
            best_jump, split_thr = 0.0, 0.5 * (min_gap + max_gap)
            for k in range(len(sg) - 1):
                jump = sg[k + 1] - sg[k]
                if jump > best_jump:
                    best_jump = jump
                    split_thr = 0.5 * (sg[k] + sg[k + 1])
        elif not any(connectors):
            split_thr = 0.0
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


def _parse_lilypond_sys_starts(stderr_text: str, total_measures: int) -> list[tuple[int, int]]:
    """Parse ``SYS_START <bar>`` lines from LilyPond stderr into ``(start, end)`` spans."""
    raw: list[int] = []
    for ln in stderr_text.splitlines():
        m = re.search(r'\bSYS_START\s+(\d+)\b', ln)
        if m:
            raw.append(int(m.group(1)))
    if not raw:
        return []
    # Deduplicate consecutive identical bar numbers while preserving order.
    starts: list[int] = []
    for b in raw:
        if not starts or b > starts[-1]:
            starts.append(b)
    # Shift anacrusis (bar 0) to 1-based MusicXML ordinal measure index.
    if starts[0] == 0:
        starts = [b + 1 for b in starts]
    spans: list[tuple[int, int]] = []
    for i, s_m in enumerate(starts):
        if i + 1 < len(starts):
            e_m = starts[i + 1] - 1
        else:
            e_m = total_measures if total_measures >= s_m else s_m
        if total_measures > 0:
            e_m = min(e_m, total_measures)
        spans.append((s_m, max(s_m, e_m)))
    return spans


def render_lilypond(
    musicxml_path: str | Path,
    out_dir: str | Path,
    dpi: int = 300,
    extract_strips: bool = True,
    max_pages: int | None = None,
    emit_bekern: bool = True,
    write_staff_jsonl: bool = True,
    aser_bin: str | Path | None = None,
) -> dict:
    """Render a MusicXML file via ``musicxml2ly`` + ``lilypond --png``.

    Injects a lightweight LilyPond Scheme ``after-line-breaking`` hook to
    recover engraver-exact per-system measure spans, then emits validated
    ``<strip_stem>-bekern.txt`` targets and ``hybrid_staff.jsonl`` rows.
    """
    lp_bin, m2l_bin = find_lilypond_bins()
    if lp_bin is None or m2l_bin is None:
        raise FileNotFoundError(
            f'LilyPond binaries not found (lilypond={lp_bin}, musicxml2ly={m2l_bin})'
        )

    xml_path = Path(musicxml_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{xml_path.stem}__lp'

    total_measures = 0
    xml_breaks: list[dict] = []
    if extract_strips and emit_bekern:
        total_measures, xml_breaks = _total_measures_from_breaks(xml_path, aser_bin=aser_bin)

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
        # Inject system-start Scheme hook before \score & disable tagline.
        ly_txt = ly_path.read_text(encoding='utf-8', errors='replace')
        score_m = re.search(r'(?m)^\s*\\score\b', ly_txt)
        if score_m is not None:
            idx_s = score_m.start()
            ly_txt = ly_txt[:idx_s] + _LILYPOND_SYS_BREAK_HOOK + '\n' + ly_txt[idx_s:]
        ly_txt += '\n\\header { tagline = ##f }\n'
        ly_path.write_text(ly_txt, encoding='utf-8')

        prefix = td_path / 'page'
        lp_res = subprocess.run(
            [str(lp_bin), '--png', f'-dresolution={int(dpi)}',
             '-o', str(prefix), str(ly_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        lp_spans = _parse_lilypond_sys_starts(lp_res.stderr or '', total_measures)
        if not lp_spans and xml_breaks:
            lp_spans = [
                (int(b['start_measure']), int(b['end_measure']))
                for b in xml_breaks
            ]

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

        # Pass 1: save rendered page PNGs and detect systems across all pages
        # before indexing lp_spans so we can enforce exact count equality.
        rendered_pages: list[tuple[int, Image.Image, Path, list[dict]]] = []
        for idx, rp in enumerate(raw_pngs, start=1):
            img = Image.open(rp).convert('RGB')
            page_png = out / f'{stem}-{idx}.png'
            _atomic_save_png(img, page_png)
            systems = _extract_strips_from_clean_png(img)
            rendered_pages.append((idx, img, page_png, systems))

        total_detected_systems = sum(len(systems) for _, _, _, systems in rendered_pages)
        spans_aligned = (len(lp_spans) > 0 and total_detected_systems == len(lp_spans))
        mismatch_reason: str | None = None
        if emit_bekern and not spans_aligned:
            mismatch_reason = f'sys-count-mismatch:{total_detected_systems}!={len(lp_spans)}'
            msg = (
                f'[multi_engrave] WARNING: LilyPond system-count mismatch for {xml_path.name}: '
                f'detected={total_detected_systems} vs lp_spans={len(lp_spans)} — '
                f'skipping bekern.txt emission to prevent measure misalignment'
            )
            logging.warning(msg)
            print(msg, file=sys.stderr)

        pages_meta = []
        staff_rows: list[dict] = []
        global_si = 0
        for idx, img, page_png, systems in rendered_pages:
            strips = []
            for si, sys_info in enumerate(systems):
                if spans_aligned:
                    start_m, end_m = lp_spans[global_si]
                else:
                    start_m, end_m = 0, 0
                global_si += 1

                if not extract_strips:
                    sys_info['start_measure'] = start_m
                    sys_info['end_measure'] = end_m
                    continue

                x1, y1, x2, y2 = sys_info['bbox']
                if x2 > x1 + 16 and y2 > y1 + 16:
                    crop = img.crop((x1, y1, x2, y2))
                    strip_stem = f'{stem}-{idx}-s{si}'
                    strip_png = out / f'{strip_stem}.png'
                    _atomic_save_png(crop, strip_png)

                    bek_name: str | None = None
                    n_tok = 0
                    bek_err: str | None = None
                    if emit_bekern:
                        if not spans_aligned:
                            bek_err = mismatch_reason
                        elif 1 <= start_m <= end_m:
                            bek_path, n_tok, bek_err = emit_and_validate_bekern(
                                xml_path, start_m, end_m, out, strip_stem,
                                aser_bin=aser_bin,
                            )
                            if bek_path is not None:
                                bek_name = bek_path.name
                        else:
                            bek_err = f'invalid-span:{start_m}..{end_m}'

                    skip_row = bool(emit_bekern and bek_name is None)
                    srow = make_staff_jsonl_row(
                        strip_png_name=strip_png.name,
                        page_staves=sys_info['staves'],
                        crop_y1=y1,
                        strip_w=crop.width,
                        sp=float(sys_info['staff_space_px']),
                        source='lilypond',
                        skip=skip_row,
                        reason=(f'bekern:{bek_err}' if skip_row and bek_err else None),
                    )
                    staff_rows.append(srow)
                    strips.append({
                        'strip_png': strip_png.name,
                        'width': crop.width,
                        'height': crop.height,
                        'start_measure': start_m,
                        'end_measure': end_m,
                        'bekern_txt': bek_name,
                        'n_tokens': n_tok,
                        'bekern_error': bek_err,
                        'staff_row': srow,
                        **sys_info,
                    })
            pages_meta.append({
                'page': idx,
                'png': page_png.name,
                'width': img.width,
                'height': img.height,
                'systems': strips if extract_strips else systems,
            })

    if extract_strips and write_staff_jsonl and staff_rows:
        _upsert_staff_jsonl(out / 'hybrid_staff.jsonl', staff_rows)

    meta = {
        'engraver': 'lilypond',
        'font': 'Emmentaler',
        'dpi': dpi,
        'source': str(xml_path),
        'staff_jsonl': 'hybrid_staff.jsonl' if (extract_strips and write_staff_jsonl and staff_rows) else None,
        'total_detected_systems': total_detected_systems,
        'total_lp_spans': len(lp_spans),
        'system_count_mismatch': bool(emit_bekern and not spans_aligned),
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
    ap.add_argument('--no-bekern', action='store_true',
                    help='Skip assembly-serialize bekern --measures emission')
    ap.add_argument('--no-staff-jsonl', action='store_true',
                    help='Skip hybrid_staff.jsonl emission')
    ap.add_argument('--aser-bin', default=None,
                    help='Optional path to assembly-serialize executable')
    args = ap.parse_args()

    engs = [e.strip().lower() for e in args.engravers.split(',') if e.strip()]
    vfonts = [f.strip() for f in args.verovio_fonts.split(',') if f.strip()]
    summary = []
    if 'verovio' in engs:
        for f in vfonts:
            m = render_verovio(
                args.input, args.out, font=f,
                extract_strips=not args.no_strips,
                max_pages=args.max_pages,
                emit_bekern=not args.no_bekern,
                write_staff_jsonl=not args.no_staff_jsonl,
                aser_bin=args.aser_bin,
            )
            n_strips = sum(len(p['systems']) for p in m['pages'])
            n_bek = sum(1 for p in m['pages'] for s in p['systems'] if s.get('bekern_txt'))
            summary.append(f'verovio/{f}: {len(m["pages"])}p, {n_strips} strips, {n_bek} bekern')
    if 'lilypond' in engs:
        m = render_lilypond(
            args.input, args.out,
            extract_strips=not args.no_strips,
            max_pages=args.max_pages,
            emit_bekern=not args.no_bekern,
            write_staff_jsonl=not args.no_staff_jsonl,
            aser_bin=args.aser_bin,
        )
        n_strips = sum(len(p['systems']) for p in m['pages'])
        n_bek = sum(1 for p in m['pages'] for s in p['systems'] if s.get('bekern_txt'))
        summary.append(f'lilypond/Emmentaler: {len(m["pages"])}p, {n_strips} strips, {n_bek} bekern')
    print(' | '.join(summary))


if __name__ == '__main__':
    main()
