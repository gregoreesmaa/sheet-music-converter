"""Incorporate 53 verified PDM 1.0 x CC0 OpenScore Lieder handwritten works
(D:\\tmp\\pdm-openscore-hw\\, 271 full-page scans: 87 portrait / 184 landscape)
into our unified handwritten corpus registry (D:\\tmp\\real-hw-corpus\\) and
Suite31+HW companion manifests.

Strict work-level train/eval split (ZERO overlap):
  - Held-out EVAL (5 commercial-safe portrait works, 4 distinct composers, 1 page each,
    PDM 1.0 SBB Berlin + CC0-1.0 OpenScore Lieder, zero Gallica caveat):
      1. hensel-morgengruss-d-b               -> hw-hensel-morgengruss-d-b
      2. mendelssohn-herbstlied-d-b           -> hw-mendelssohn-herbstlied-d-b
      3. beethoven-das-blumchen-wunderhold-d-b -> hw-beethoven-das-blumchen-wunderhold-d-b
      4. cornelius-botschaft-d-b              -> hw-cornelius-botschaft-d-b
      5. hensel-im-herbste-d-b                -> hw-hensel-im-herbste-d-b
  - Train (remaining 48 OpenScore PDM 1.0 x CC0 works + 4 existing real-hw-train quartet works = 52 works):
      * Unified registry -> D:\\tmp\\real-hw-corpus\\train_works.jsonl
      * Verified bekern  -> D:\\tmp\\real-hw-corpus\\bekern\\<work_id>-bekern.txt
      * Documentation    -> D:\\tmp\\real-hw-corpus\\USAGE.md

CPU-only (CUDA_VISIBLE_DEVICES=""). Never modifies frozen suite31_manifest.json.
"""

from __future__ import annotations

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import concurrent.futures
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import recanon_i1  # noqa: E402

ASER = REPO_ROOT / 'converter' / 'assembly-serialize' / 'target' / 'release' / 'assembly-serialize.exe'
PHOTO_DEGRADE = REPO_ROOT / 'data-gen' / 'photo-degrade' / 'target' / 'release' / 'photo-degrade.exe'
SUITE31 = REPO_ROOT / 'data-gen' / 'score-corpus' / 'suite31'

OPENSCORE_HW_ROOT = Path(r'D:\tmp\pdm-openscore-hw')
REAL_HW_TRAIN_ROOT = Path(r'D:\tmp\real-hw-train')
REAL_HW_CORPUS_ROOT = Path(r'D:\tmp\real-hw-corpus')

FROZEN_SUITE31_SHA256 = 'b58b4a36b70ddf553cc2bbb63396868971a6284a21d5813d6b11891fc8a7b3ac'

# 5 commercial-safe portrait eval holdout works (4 distinct composers, 1 page each, PDM 1.0 x CC0).
# Measure counts & vertical system bands verified by visual inspection of page_001.jpg.
EVAL_SPECS = [
    {
        'work_id': 'hensel-morgengruss-d-b',
        'slug': 'hw-hensel-morgengruss-d-b',
        'p1_measures': 10,  # pickup + 4 (sys 1) + 5 (sys 2) out of 35 total
        'n_sys': 2,
        'y_span': (0.16, 0.82),
        'seed': 20261125,
    },
    {
        'work_id': 'mendelssohn-herbstlied-d-b',
        'slug': 'hw-mendelssohn-herbstlied-d-b',
        'p1_measures': 29,  # 6+6+6+5+6 across 5 systems out of 109 total
        'n_sys': 5,
        'y_span': (0.05, 0.88),
        'seed': 20261126,
    },
    {
        'work_id': 'beethoven-das-blumchen-wunderhold-d-b',
        'slug': 'hw-beethoven-das-blumchen-wunderhold-d-b',
        'p1_measures': 21,  # full song (all 21 measures across 4 systems on p.1)
        'n_sys': 4,
        'y_span': (0.07, 0.60),
        'seed': 20261127,
    },
    {
        'work_id': 'cornelius-botschaft-d-b',
        'slug': 'hw-cornelius-botschaft-d-b',
        'p1_measures': 17,  # 3+5+4+5 across 4 systems out of 70 total
        'n_sys': 4,
        'y_span': (0.10, 0.86),
        'seed': 20261128,
    },
    {
        'work_id': 'hensel-im-herbste-d-b',
        'slug': 'hw-hensel-im-herbste-d-b',
        'p1_measures': 18,  # 4+5+4+5 across 4 systems out of 38 total
        'n_sys': 4,
        'y_span': (0.08, 0.91),
        'seed': 20261129,
    },
]
EVAL_WORK_IDS = {sp['work_id'] for sp in EVAL_SPECS}
EVAL_SLUGS = {sp['slug'] for sp in EVAL_SPECS}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def is_music_page(img_path: Path) -> tuple[bool, dict]:
    """Heuristic check that a manuscript scan is a real music page (not a blank cover).

    Checks interior dark-ink fraction and horizontal projection gradient (staff lines
    create consistent row-to-row luminance transitions).
    """
    im = Image.open(img_path).convert('L')
    arr = np.array(im)
    h, w = arr.shape
    crop = arr[int(h * 0.10):int(h * 0.90), int(w * 0.10):int(w * 0.90)]
    dark_frac = float((crop < 140).mean())
    row_means = crop.mean(axis=1)
    row_diff = float(np.abs(np.diff(row_means)).mean())
    ok = (0.015 <= dark_frac <= 0.45) and (row_diff >= 1.5)
    return ok, {
        'width': w,
        'height': h,
        'portrait': h > w,
        'dark_frac': round(dark_frac, 4),
        'row_diff': round(row_diff, 3),
    }


def select_first_music_page(page_paths: list[str]) -> tuple[int, Path, dict]:
    """Return (1-based page_index, Path, metrics) for the first portrait music page."""
    for idx, pp in enumerate(page_paths, start=1):
        p = Path(pp)
        ok, metrics = is_music_page(p)
        if ok and metrics['portrait']:
            return idx, p, metrics
    raise RuntimeError(f'No valid portrait music page found in {page_paths}')


def slice_and_clean_musicxml(src_xml: Path, end_idx: int | None, strip_pno: bool = False) -> tuple[str, int, int]:
    """Slice measures [0:end_idx] (or keep all if end_idx is None) and strip page/system breaks.

    Returns (xml_str, kept_measures, total_measures).
    """
    tree = ET.parse(src_xml)
    root = tree.getroot()
    parts = root.findall('part')
    total_meas = len(parts[0].findall('measure')) if parts else 0
    kept_meas = total_meas if end_idx is None else min(end_idx, total_meas)
    for part in parts:
        measures = part.findall('measure')
        if end_idx is not None and end_idx < len(measures):
            for m in measures[end_idx:]:
                part.remove(m)
        for m in part.findall('measure'):
            for pr in list(m.findall('print')):
                pr.attrib.pop('new-page', None)
                pr.attrib.pop('new-system', None)
            if strip_pno:
                for el in list(m):
                    if el.attrib.get('print-object') == 'no':
                        m.remove(el)
    return ET.tostring(root, encoding='unicode', xml_declaration=True), kept_meas, total_meas


def convert_and_verify_bekern(
    src_xml: Path,
    out_dir: Path,
    stem: str,
    end_idx: int | None = None,
) -> tuple[bool, int, str, Path | None]:
    """Convert MusicXML -> canonical post-I1 <stem>-bekern.txt and verify bekern-decode rc=0."""
    with tempfile.TemporaryDirectory(prefix='os_bk_') as td:
        td_p = Path(td)
        xml_p = td_p / f'{stem}.musicxml'
        rescued = False
        try:
            cleaned, _, _ = slice_and_clean_musicxml(src_xml, end_idx=end_idx, strip_pno=False)
        except Exception as e:
            return False, 0, f'xml_parse:{e}', None
        xml_p.write_text(cleaned, encoding='utf-8')
        r1 = subprocess.run(
            [str(ASER), 'bekern', '--input', str(xml_p), '--page', '1',
             '--out', str(td_p), '--stem', stem],
            capture_output=True, text=True, timeout=60,
        )
        if r1.returncode != 0:
            try:
                cleaned2, _, _ = slice_and_clean_musicxml(src_xml, end_idx=end_idx, strip_pno=True)
            except Exception as e:
                return False, 0, f'xml_parse2:{e}', None
            xml_p.write_text(cleaned2, encoding='utf-8')
            r1b = subprocess.run(
                [str(ASER), 'bekern', '--input', str(xml_p), '--page', '1',
                 '--out', str(td_p), '--stem', stem],
                capture_output=True, text=True, timeout=60,
            )
            if r1b.returncode != 0:
                err = (r1.stderr or r1.stdout).strip().splitlines()
                return False, 0, 'bekern_fail:' + (err[-1] if err else 'unknown')[:180], None
            rescued = True

        bk_tmp = td_p / f'{stem}-bekern.txt'
        if not bk_tmp.exists():
            return False, 0, 'no_bekern_file', None
        txt = bk_tmp.read_text(encoding='utf-8').strip()
        toks = txt.split()
        if not toks:
            return False, 0, 'empty_bekern', None

        if not recanon_i1.is_stream_canonical_i1(txt):
            new_toks, changed = recanon_i1.canonicalize_bekern_tokens(toks)
            if changed:
                txt = ' '.join(new_toks)
                bk_tmp.write_text(txt + '\n', encoding='utf-8')
                toks = new_toks
            if not recanon_i1.is_stream_canonical_i1(txt):
                return False, len(toks), 'non_i1_after_recanon', None

        dec_p = td_p / f'{stem}-dec.json'
        r2 = subprocess.run(
            [str(ASER), 'bekern-decode', '--input', str(bk_tmp), '--out', str(dec_p)],
            capture_output=True, text=True, timeout=60,
        )
        if r2.returncode != 0 or not dec_p.exists():
            err = (r2.stderr or r2.stdout).strip().splitlines()
            return False, len(toks), 'decode_fail:' + (err[-1] if err else 'unknown')[:140], None

        out_dir.mkdir(parents=True, exist_ok=True)
        final_p = out_dir / f'{stem}-bekern.txt'
        tmp_final = out_dir / f'{stem}-bekern.txt.tmp'
        tmp_final.write_text(txt + '\n', encoding='utf-8')
        os.replace(tmp_final, final_p)
        return True, len(toks), ('rescued_pno' if rescued else 'ok'), final_p


def run_photo_degrade(src_png: Path, dst_path: Path, seed: int, fmt: str = 'png') -> None:
    """Run photo-degrade on one image deterministically (CPU) and move to dst_path."""
    with tempfile.TemporaryDirectory(prefix='os_deg_') as td:
        td_p = Path(td)
        in_dir = td_p / 'in'
        out_dir = td_p / 'out'
        in_dir.mkdir()
        out_dir.mkdir()
        shutil.copy2(src_png, in_dir / src_png.name)
        cmd = [
            str(PHOTO_DEGRADE),
            '--input', str(in_dir),
            '--out-dir', str(out_dir),
            '--seed', str(seed),
            '--format', fmt,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            raise RuntimeError(f'photo-degrade failed on {src_png.name}: {r.stderr[-300:]}')
        ext = '.jpg' if fmt == 'jpeg' else '.png'
        cands = list(out_dir.glob('*' + ext))
        if not cands:
            raise RuntimeError(f'photo-degrade produced no {ext} in {out_dir}')
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cands[0]), str(dst_path))


def build_eval_entries(by_slug: dict[str, dict]) -> list[dict]:
    hw_clean = SUITE31 / 'hw' / 'clean'
    hw_photo = SUITE31 / 'hw' / 'photo'
    hw_gt = SUITE31 / 'hw' / 'gt'
    for d in (hw_clean, hw_photo, hw_gt):
        d.mkdir(parents=True, exist_ok=True)

    new_pieces = []
    for sp in EVAL_SPECS:
        wid = sp['work_id']
        slug = sp['slug']
        row = by_slug[wid]
        page_idx, page_src, metrics = select_first_music_page(row['page_paths'])
        assert metrics['portrait'], f'Selected page for {wid} is not portrait: {metrics}'

        dst_clean = hw_clean / f'{slug}.png'
        dst_photo = hw_photo / f'{slug}.png'
        dst_gt = hw_gt / f'gt-p1-{slug}.musicxml'

        img = Image.open(page_src).convert('RGB')
        iw, ih = img.size
        assert ih > iw, f'Eval clean {slug} is NOT portrait: {iw}x{ih}'
        img.save(dst_clean, format='PNG')

        run_photo_degrade(dst_clean, dst_photo, seed=sp['seed'], fmt='png')
        pw, ph = Image.open(dst_photo).size
        assert ph > pw, f'Eval photo {slug} is NOT portrait: {pw}x{ph}'

        src_xml = Path(row['xml_path'])
        cleaned_xml, kept_meas, total_meas = slice_and_clean_musicxml(
            src_xml, end_idx=sp['p1_measures'], strip_pno=False,
        )
        dst_gt.write_text(cleaned_xml, encoding='utf-8')

        # Verify bekern + bekern-decode + import rc=0 on the exact staged GT
        with tempfile.TemporaryDirectory(prefix='os_ev_') as td:
            td_p = Path(td)
            ok, ntok, msg, _ = convert_and_verify_bekern(dst_gt, td_p, slug, end_idx=None)
            assert ok, f'Eval bekern+decode failed on {slug}: {msg}'
            imp_dir = td_p / 'imp'
            imp_dir.mkdir()
            ri = subprocess.run(
                [str(ASER), 'import', '--input', str(dst_gt), '--out', str(imp_dir)],
                capture_output=True, text=True, timeout=60,
            )
            assert ri.returncode == 0, f'Eval import failed on {slug}: {ri.stderr[-200:]}'

        y0_f, y1_f = sp['y_span']
        top, bot = int(ih * y0_f), int(ih * y1_f)
        n_sys = sp['n_sys']
        band = (bot - top) / n_sys
        sys_boxes = [
            [int(iw * 0.05), int(top + i * band), int(iw * 0.95), int(top + (i + 1) * band)]
            for i in range(n_sys)
        ]

        print(
            f'[EVAL] {slug} (p.{page_idx}, {iw}x{ih}, photo {pw}x{ph}): '
            f'meas={kept_meas}/{total_meas}, n_sys={n_sys}, bekern_toks={ntok}, '
            f'ink={metrics["dark_frac"]:.3f}, row_diff={metrics["row_diff"]:.2f}'
        )

        new_pieces.append({
            'slug': slug,
            'work_id': wid,
            'handwritten': True,
            'orientation': 'portrait',
            'source': 'openscore-lieder-pdm',
            'library': row['library'],
            'material_type': row['material_type'],
            'page_index': page_idx,
            'rism_url': row['rism_url'],
            'iiif_manifest': row['iiif_manifest'],
            'work': f'{row["title"]} (p.{page_idx})',
            'composer': row['composer'],
            'license': f'{row["rights"]} ({row["library"]}) + {row["xml_license"]}',
            'p1_measures': kept_meas,
            'system_boxes': sys_boxes,
            'clean': {
                'file': f'hw/clean/{slug}.png',
                'sha256': sha256_file(dst_clean),
                'bytes': dst_clean.stat().st_size,
            },
            'photo': {
                'file': f'hw/photo/{slug}.png',
                'sha256': sha256_file(dst_photo),
                'bytes': dst_photo.stat().st_size,
            },
            'gt': {
                'file': f'hw/gt/gt-p1-{slug}.musicxml',
                'sha256': sha256_file(dst_gt),
                'bytes': dst_gt.stat().st_size,
            },
        })

    # Update suite31_hw_only_manifest.json (idempotent: keep existing 9 non-openscore HW, append 5 new)
    hw_only_path = SUITE31 / 'suite31_hw_only_manifest.json'
    existing_hw_only = json.loads(hw_only_path.read_text(encoding='utf-8'))
    base_hw = [
        p for p in existing_hw_only['pieces']
        if p.get('source') != 'openscore-lieder-pdm' and p['slug'] not in EVAL_SLUGS
    ]
    assert len(base_hw) == 9, f'Expected 9 existing HW pieces, found {len(base_hw)}'
    all_hw = base_hw + new_pieces
    assert len(all_hw) == 14

    gen_by = (
        'tmp-train/mvd/build_real_hw_suite.py + '
        'tmp-train/mvd/build_debussy_hw.py + '
        'tmp-train/mvd/incorporate_openscore_hw.py'
    )
    hw_only_out = {
        'suite': 'suite31-hw-only',
        'generated_by': gen_by,
        'pieces_expected': len(all_hw),
        'license': (
            'Public Domain (IMSLP holographs) + CC0-1.0 (OpenScore StringQuartets & Lieder) + '
            'PDM 1.0 (SBB Berlin) + CC-BY-4.0 (HugoSchtr/debussy-omr, BnF Gallica); zero NC data'
        ),
        'pieces': all_hw,
    }
    hw_only_path.write_text(json.dumps(hw_only_out, indent=1), encoding='utf-8')

    # Update suite31_hw_manifest.json (31 frozen printed + 14 HW = 45; NEVER touch suite31_manifest.json)
    base_manifest = json.loads((SUITE31 / 'suite31_manifest.json').read_text(encoding='utf-8'))
    assert len(base_manifest['pieces']) == 31
    combined = copy.deepcopy(base_manifest)
    combined['suite'] = 'suite31+hw'
    combined['generated_by'] = gen_by
    combined['pieces_expected'] = len(base_manifest['pieces']) + len(all_hw)
    combined['pieces'] = base_manifest['pieces'] + all_hw
    assert len(combined['pieces']) == 45
    (SUITE31 / 'suite31_hw_manifest.json').write_text(
        json.dumps(combined, indent=1), encoding='utf-8'
    )

    return new_pieces


def build_train_registry(all_rows: list[dict]) -> dict:
    REAL_HW_CORPUS_ROOT.mkdir(parents=True, exist_ok=True)
    bekern_dir = REAL_HW_CORPUS_ROOT / 'bekern'
    bekern_dir.mkdir(parents=True, exist_ok=True)

    train_openscore = [r for r in all_rows if r['slug'] not in EVAL_WORK_IDS]
    assert len(train_openscore) == 48, f'Expected 48 OpenScore train works, got {len(train_openscore)}'

    def _process_one(row: dict) -> dict:
        wid = row['slug']
        ok, ntok, status, bk_path = convert_and_verify_bekern(
            Path(row['xml_path']), bekern_dir, wid, end_idx=None,
        )
        return {
            'work_id': wid,
            'source': 'openscore-lieder-pdm',
            'composer': row['composer'],
            'title': row['title'],
            'library': row['library'],
            'license': f'{row["rights"]} ({row["library"]}) + {row["xml_license"]}',
            'rights': row['rights'],
            'xml_license': row['xml_license'],
            'material_type': row['material_type'],
            'orientation': row['orientation'],
            'n_pages': int(row['n_pages']),
            'n_portrait_pages': int(row['n_portrait_pages']),
            'n_landscape_pages': int(row['n_landscape_pages']),
            'page_paths': row['page_paths'],
            'page_dims': row['page_dims'],
            'xml_path': row['xml_path'],
            'bekern_path': str(bk_path) if bk_path else None,
            'bekern_ok': ok,
            'bekern_tokens': ntok,
            'bekern_status': status,
            'rism_url': row['rism_url'],
            'iiif_manifest': row['iiif_manifest'],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        os_records = list(ex.map(_process_one, train_openscore))

    # Incorporate existing 4 real-hw-train quartet works
    quartet_split = json.loads((REAL_HW_TRAIN_ROOT / 'split_manifest.json').read_text(encoding='utf-8'))
    quartet_records = []
    quartet_meta = {
        'hw-mozart-k428': ('Mozart, Wolfgang Amadeus', 'String Quartet No.16 in E-flat major, K.428 (Mvt I p.1)', 'mozart-k428.musicxml'),
        'hw-mozart-k458': ('Mozart, Wolfgang Amadeus', 'String Quartet No.17 in B-flat major, K.458 (Mvt I p.1)', 'mozart-k458.musicxml'),
        'hw-mozart-k465': ('Mozart, Wolfgang Amadeus', 'String Quartet No.19 in C major, K.465 (Mvt I p.1)', 'mozart-k465.musicxml'),
        'hw-mozart-k590': ('Mozart, Wolfgang Amadeus', 'String Quartet No.23 in F major, K.590 (Mvt I p.1)', 'mozart-k590.musicxml'),
    }
    renders_dir = REAL_HW_TRAIN_ROOT / 'renders'
    targets_dir = REAL_HW_TRAIN_ROOT / 'targets'
    for wid in quartet_split['train_works']:
        comp, title, xml_fn = quartet_meta[wid]
        clean_strips = sorted(str(p) for p in renders_dir.glob(f'{wid}-1-s*.png'))
        deg_strips = sorted(str(p) for p in renders_dir.glob(f'{wid}-1-s*_d*.jpg'))
        bk_strips = sorted(str(p) for p in targets_dir.glob(f'{wid}-1-s*-bekern.txt'))
        assert clean_strips and bk_strips, f'Missing strips for {wid}'
        # Verify bekern-decode rc=0 on every existing strip target
        for bk in bk_strips:
            with tempfile.TemporaryDirectory(prefix='q_dec_') as td:
                dec_p = Path(td) / 'dec.json'
                r = subprocess.run(
                    [str(ASER), 'bekern-decode', '--input', bk, '--out', str(dec_p)],
                    capture_output=True, text=True, timeout=60,
                )
                assert r.returncode == 0 and dec_p.exists(), f'Quartet bekern-decode failed on {bk}'
        quartet_records.append({
            'work_id': wid,
            'source': 'real-hw-train-quartets',
            'composer': comp,
            'title': title,
            'library': 'IMSLP (holographs)',
            'license': 'Public Domain (IMSLP) + CC0-1.0 (OpenScore StringQuartets)',
            'rights': 'Public Domain',
            'xml_license': 'CC0-1.0 (OpenScore StringQuartets)',
            'material_type': 'Autograph manuscript (system strips)',
            'orientation': 'landscape',
            'n_pages': 1,
            'n_portrait_pages': 0,
            'n_landscape_pages': 1,
            'page_paths': clean_strips,
            'degraded_strip_paths': deg_strips,
            'xml_path': str(Path(r'D:\tmp\hw-align\xml') / xml_fn),
            'bekern_path': bk_strips[0],
            'bekern_paths': bk_strips,
            'bekern_ok': True,
            'bekern_status': 'ok',
            'hybrid_staff_jsonl': str(REAL_HW_TRAIN_ROOT / 'hybrid_staff.jsonl'),
        })

    all_train = os_records + quartet_records
    train_jsonl = REAL_HW_CORPUS_ROOT / 'train_works.jsonl'
    with open(train_jsonl, 'w', encoding='utf-8') as f:
        for rec in all_train:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # Compute summary stats
    os_bk_ok = sum(1 for r in os_records if r['bekern_ok'])
    os_bk_fail = [r for r in os_records if not r['bekern_ok']]
    os_port_works = sum(1 for r in os_records if r['orientation'] == 'portrait')
    os_land_works = sum(1 for r in os_records if r['orientation'] == 'landscape')
    os_port_pages = sum(r['n_portrait_pages'] for r in os_records)
    os_land_pages = sum(r['n_landscape_pages'] for r in os_records)
    os_total_pages = sum(r['n_pages'] for r in os_records)
    q_strips = sum(len(r['page_paths']) for r in quartet_records)
    q_deg = sum(len(r['degraded_strip_paths']) for r in quartet_records)

    return {
        'train_jsonl': str(train_jsonl),
        'total_train_works': len(all_train),
        'openscore_train_works': len(os_records),
        'openscore_train_bekern_ok': os_bk_ok,
        'openscore_train_bekern_fail': [
            {'work_id': r['work_id'], 'status': r['bekern_status']} for r in os_bk_fail
        ],
        'openscore_train_portrait_works': os_port_works,
        'openscore_train_landscape_works': os_land_works,
        'openscore_train_total_pages': os_total_pages,
        'openscore_train_portrait_pages': os_port_pages,
        'openscore_train_landscape_pages': os_land_pages,
        'quartet_train_works': len(quartet_records),
        'quartet_clean_strips': q_strips,
        'quartet_degraded_strips': q_deg,
    }


def write_usage_md(eval_pieces: list[dict], train_stats: dict, all_rows: list[dict]) -> Path:
    eval_port_pages = sum(
        r['n_portrait_pages'] for r in all_rows if r['slug'] in EVAL_WORK_IDS
    )
    eval_total_pages = sum(
        r['n_pages'] for r in all_rows if r['slug'] in EVAL_WORK_IDS
    )
    total_os_bk_ok = 5 + train_stats['openscore_train_bekern_ok']
    fail_lines = '\n'.join(
        f'- `{x["work_id"]}`: `{x["status"]}` (retained in `train_works.jsonl` with `bekern_ok: false` for layout/staff training)'
        for x in train_stats['openscore_train_bekern_fail']
    ) or '- None (100% conversion)'

    eval_table = '\n'.join(
        f'| `{p["slug"]}` | `{p["work_id"]}` | {p["composer"]} | {p["work"]} | {p["p1_measures"]} | {p["license"]} |'
        for p in eval_pieces
    )

    md = f"""# Unified Real Handwritten Corpus Registry (`D:\\tmp\\real-hw-corpus\\`)

Generated reproducibly by `tmp-train/mvd/incorporate_openscore_hw.py` (CPU-only).

## 1. Overview & Counts

| Split | Source | Works | Pages / Strips | Portrait | Landscape | `bekern` + `bekern-decode` OK |
|---|---|---:|---:|---:|---:|---:|
| **Eval holdout** (added to `suite31_hw`) | OpenScore Lieder × SBB Berlin (`PDM 1.0 × CC0-1.0`) | 5 | 5 staged (from {eval_total_pages} scans) | 5 works (5/5 staged, {eval_port_pages} raw scans) | 0 | **5 / 5 (100.0%)** |
| **Train** (`train_works.jsonl`) | OpenScore Lieder × SBB Berlin / SLUB Dresden (`PDM 1.0 × CC0-1.0`) | {train_stats['openscore_train_works']} | {train_stats['openscore_train_total_pages']} full-page scans | {train_stats['openscore_train_portrait_works']} works ({train_stats['openscore_train_portrait_pages']} pages) | {train_stats['openscore_train_landscape_works']} works ({train_stats['openscore_train_landscape_pages']} pages) | **{train_stats['openscore_train_bekern_ok']} / {train_stats['openscore_train_works']} ({100.0 * train_stats['openscore_train_bekern_ok'] / train_stats['openscore_train_works']:.1f}%)** |
| **Train** (`train_works.jsonl`) | Existing `real-hw-train` String Quartets (`PD IMSLP × CC0-1.0`) | {train_stats['quartet_train_works']} | {train_stats['quartet_clean_strips']} clean strips + {train_stats['quartet_degraded_strips']} degraded | 0 | 4 works (12 strips) | **4 / 4 works (12/12 strips, 100.0%)** |
| **Total OpenScore PDM 1.0** | All harvested OpenScore Lieder manuscripts | 53 | 271 full-page scans | 15 works (87 pages) | 38 works (184 pages) | **{total_os_bk_ok} / 53 ({100.0 * total_os_bk_ok / 53:.1f}%)** |
| **Total Train Registry** | `train_works.jsonl` | **{train_stats['total_train_works']}** | {train_stats['openscore_train_total_pages']} pages + {train_stats['quartet_clean_strips']} strips | {train_stats['openscore_train_portrait_works']} works | {train_stats['openscore_train_landscape_works'] + train_stats['quartet_train_works']} works | **{train_stats['openscore_train_bekern_ok'] + train_stats['quartet_train_works']} / {train_stats['total_train_works']}** |

## 2. Licenses (100% Commercial-Safe, Zero CC-BY-NC, Zero Gallica Caveat)

- **OpenScore Lieder scans (`D:\\tmp\\pdm-openscore-hw\\`)**: `PDM 1.0` (Public Domain Mark 1.0) from SBB Berlin (50 works) and SLUB Dresden (3 works).
- **OpenScore Lieder ground-truth MusicXML**: `CC0-1.0` (OpenScore Lieder).
- **Existing `real-hw-train` quartet strips (`D:\\tmp\\real-hw-train\\`)**: Public Domain (IMSLP holographs) + `CC0-1.0` (OpenScore StringQuartets).

## 3. Held-Out Portrait Eval Works (Added to `suite31_hw` Manifests)

All 5 works are portrait (`H > W`), verified real music pages on `page_001.jpg`, 4 distinct composers, and verified through `assembly-serialize bekern` + `bekern-decode` (`rc == 0`) + `import` (`rc == 0`):

| Eval Slug | OpenScore `work_id` | Composer | Work | `p1_measures` | License |
|---|---|---|---|---:|---|
{eval_table}

Companion manifests updated (frozen `suite31_manifest.json` untouched, SHA-256 `{FROZEN_SUITE31_SHA256}`):
- `data-gen/score-corpus/suite31/suite31_hw_only_manifest.json`: 9 → **14 pieces** (5 IMSLP landscape quartets + 4 Debussy portrait + 5 OpenScore PDM 1.0 portrait)
- `data-gen/score-corpus/suite31/suite31_hw_manifest.json`: 40 → **45 pieces** (31 frozen printed + 14 HW)

## 4. `bekern` Conversion Notes

{fail_lines}

## 5. Split Integrity & Usage

- **Zero work-level overlap** between `D:\\tmp\\real-hw-corpus\\train_works.jsonl` and `data-gen/score-corpus/suite31/suite31_hw_manifest.json` (verified by automated assertions on exact `work_id`, `slug`, and `hw-`-stripped normalized IDs).
- **Directory layout**:
  - `D:\\tmp\\real-hw-corpus\\train_works.jsonl` — unified JSONL (52 works: 48 OpenScore Lieder + 4 quartets).
  - `D:\\tmp\\real-hw-corpus\\bekern\\<work_id>-bekern.txt` — post-I1 canonical `bekern` targets verified with `bekern-decode rc == 0`.
  - `D:\\tmp\\real-hw-corpus\\USAGE.md` — this file.
- **Loading in training scripts**: filter on `row["bekern_ok"]` when training sequence/OMR decoders; use all 52 rows (including `bekern_ok: false`) for page-level staff/layout detectors.
"""
    usage_p = REAL_HW_CORPUS_ROOT / 'USAGE.md'
    usage_p.write_text(md, encoding='utf-8')
    return usage_p


def verify_all(eval_pieces: list[dict], train_stats: dict) -> None:
    # 1. Frozen suite31_manifest.json SHA-256 unchanged
    frozen_sha = sha256_file(SUITE31 / 'suite31_manifest.json')
    assert frozen_sha == FROZEN_SUITE31_SHA256, (
        f'FROZEN suite31_manifest.json SHA256 mismatch: {frozen_sha} != {FROZEN_SUITE31_SHA256}'
    )

    # 2. Manifest counts
    hw_only = json.loads((SUITE31 / 'suite31_hw_only_manifest.json').read_text(encoding='utf-8'))
    hw_full = json.loads((SUITE31 / 'suite31_hw_manifest.json').read_text(encoding='utf-8'))
    assert len(hw_only['pieces']) == 14 and hw_only['pieces_expected'] == 14
    assert len(hw_full['pieces']) == 45 and hw_full['pieces_expected'] == 45

    # 3. Zero work-ID overlap between train_works.jsonl and suite31_hw_manifest.json
    train_rows = [
        json.loads(l)
        for l in Path(train_stats['train_jsonl']).read_text(encoding='utf-8').splitlines()
        if l.strip()
    ]
    assert len(train_rows) == 52

    def _norm(s: str) -> str:
        return s[3:] if s.startswith('hw-') else s

    train_ids_raw = {r['work_id'] for r in train_rows}
    train_ids_norm = {_norm(r['work_id']) for r in train_rows}

    eval_ids_raw = set()
    eval_ids_norm = set()
    for p in hw_full['pieces']:
        eval_ids_raw.add(p['slug'])
        eval_ids_norm.add(_norm(p['slug']))
        if 'work_id' in p:
            eval_ids_raw.add(p['work_id'])
            eval_ids_norm.add(_norm(p['work_id']))

    overlap_raw = train_ids_raw & eval_ids_raw
    overlap_norm = train_ids_norm & eval_ids_norm
    assert not overlap_raw, f'Raw train/eval work-ID overlap: {overlap_raw}'
    assert not overlap_norm, f'Normalized train/eval work-ID overlap: {overlap_norm}'

    # 4. All 5 new portrait eval entries: H > W, valid files + SHA-256, bekern-decode rc=0
    hw_by_slug = {p['slug']: p for p in hw_full['pieces']}
    for sp in EVAL_SPECS:
        slug = sp['slug']
        assert slug in hw_by_slug, f'Missing {slug} in suite31_hw_manifest.json'
        entry = hw_by_slug[slug]
        for key in ('clean', 'photo', 'gt'):
            fp = SUITE31 / entry[key]['file']
            assert fp.exists(), f'Missing {fp}'
            assert fp.stat().st_size == entry[key]['bytes'], f'Size mismatch on {fp}'
            assert sha256_file(fp) == entry[key]['sha256'], f'SHA mismatch on {fp}'
        for key in ('clean', 'photo'):
            w, h = Image.open(SUITE31 / entry[key]['file']).size
            assert h > w, f'{slug} {key} is NOT portrait: {w}x{h}'
        with tempfile.TemporaryDirectory(prefix='ver_ev_') as td:
            ok, ntok, msg, _ = convert_and_verify_bekern(
                SUITE31 / entry['gt']['file'], Path(td), slug, end_idx=None,
            )
            assert ok, f'Verification bekern-decode failed on {slug}: {msg}'

    print('\n=== ALL ASSERTIONS PASSED ===')
    print(f'  0 work-ID overlap (train={len(train_rows)} works vs eval={len(hw_full["pieces"])} pieces)')
    print(f'  5/5 new portrait eval entries verified (H > W clean & photo, valid SHA-256, bekern-decode rc=0)')
    print(f'  suite31_hw_only_manifest.json: {len(hw_only["pieces"])} pieces')
    print(f'  suite31_hw_manifest.json:      {len(hw_full["pieces"])} pieces')
    print(f'  Frozen suite31_manifest.json SHA-256 unchanged: {frozen_sha}')


def main() -> None:
    frozen_before = sha256_file(SUITE31 / 'suite31_manifest.json')
    assert frozen_before == FROZEN_SUITE31_SHA256, f'Unexpected initial frozen SHA: {frozen_before}'

    manifest_p = OPENSCORE_HW_ROOT / 'manifest.jsonl'
    all_rows = [json.loads(l) for l in manifest_p.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(all_rows) == 53, f'Expected 53 OpenScore HW works, found {len(all_rows)}'
    by_slug = {r['slug']: r for r in all_rows}
    for wid in EVAL_WORK_IDS:
        assert wid in by_slug, f'Eval holdout work {wid} not found in {manifest_p}'

    eval_pieces = build_eval_entries(by_slug)
    train_stats = build_train_registry(all_rows)
    usage_p = write_usage_md(eval_pieces, train_stats, all_rows)
    verify_all(eval_pieces, train_stats)

    print('\n=== SUMMARY ===')
    print(f'Eval works added:              {len(eval_pieces)} (all portrait, 4 composers, 5/5 bekern OK)')
    print(f'Train works incorporated:      {train_stats["total_train_works"]} '
          f'({train_stats["openscore_train_works"]} OpenScore PDM 1.0 + {train_stats["quartet_train_works"]} real-hw-train quartets)')
    print(f'OpenScore bekern conversion:   {5 + train_stats["openscore_train_bekern_ok"]}/53 '
          f'(eval 5/5, train {train_stats["openscore_train_bekern_ok"]}/48)')
    print(f'OpenScore train orientation:   {train_stats["openscore_train_portrait_works"]} portrait works '
          f'({train_stats["openscore_train_portrait_pages"]} pages) / '
          f'{train_stats["openscore_train_landscape_works"]} landscape works '
          f'({train_stats["openscore_train_landscape_pages"]} pages)')
    print(f'Registry written to:           {train_stats["train_jsonl"]}')
    print(f'Usage doc written to:          {usage_p}')


if __name__ == '__main__':
    main()
