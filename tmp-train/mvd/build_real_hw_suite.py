"""Build real handwritten training strips + Suite31+HW companion manifests.

Uses ONLY permissive-licensed data from data-gen/score-corpus/handwritten-500/
(Public Domain IMSLP holographs + CC0-1.0 OpenScore StringQuartets).
Zero CC-BY-NC data (CVC-MUSCIMA) is touched.

Strict work-level train/eval split (zero leakage, honors hw-90-10/test_works.txt):
  - Held-out EVAL (5 works, 15 systems, all 3 composers):
      hw-beethoven-op135, hw-mozart-k464, hw-mozart-k387,
      hw-mozart-k421, hw-schubert-d810
  - Training (4 non-overlapping works, 12 systems):
      mozart-k428, mozart-k458, mozart-k465, mozart-k590

Frozen data-gen/score-corpus/suite31/suite31_manifest.json is NEVER modified.
Writes companion manifests:
  - data-gen/score-corpus/suite31/suite31_hw_only_manifest.json (5 HW pieces)
  - data-gen/score-corpus/suite31/suite31_hw_manifest.json (31 + 5 = 36 pieces)
And real handwritten training set:
  - D:/tmp/real-hw-train/{renders/, targets/, hybrid_staff.jsonl, split_manifest.json}
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_gate20_sys_dev as bgate  # noqa: E402
import build_hybrid_cache as bhc  # noqa: E402
import build_sys_data as bsd  # noqa: E402
import eval_sys_gate as esg  # noqa: E402

ASER = REPO_ROOT / 'converter' / 'assembly-serialize' / 'target' / 'release' / 'assembly-serialize.exe'
PHOTO_DEGRADE = REPO_ROOT / 'data-gen' / 'photo-degrade' / 'target' / 'release' / 'photo-degrade.exe'
HW500 = REPO_ROOT / 'data-gen' / 'score-corpus' / 'handwritten-500'
SUITE31 = REPO_ROOT / 'data-gen' / 'score-corpus' / 'suite31'
XML_CACHE = Path('D:/tmp/hw-align/xml')
TRAIN_OUT = Path('D:/tmp/real-hw-train')

# Verified page-1 measure slices (0-based exclusive index slices into <part>'s <measure> list).
# Each boundary was verified against musical landmarks (rests, whole notes, fermatas, time changes).
EVAL_SPECS = [
    {
        'slug': 'hw-beethoven-op135',
        'xml': 'beethoven-op135.musicxml',
        'img': 'handwritten/beethoven-op135-ms/p001.png',
        'work': 'String Quartet No.16 in F major, Op.135 (Mvt I p.1)',
        'composer': 'Beethoven, Ludwig van',
        'license': 'Public Domain (IMSLP) + CC0-1.0 (OpenScore)',
        'p1_measures': 10,
        'sys_slices': [(0, 5), (5, 10)],
    },
    {
        'slug': 'hw-mozart-k464',
        'xml': 'mozart-k464.musicxml',
        'img': 'handwritten/mozart-k464/p001.png',
        'work': 'String Quartet No.18 in A major, K.464 (Mvt I p.1)',
        'composer': 'Mozart, Wolfgang Amadeus',
        'license': 'Public Domain (IMSLP) + CC0-1.0 (OpenScore)',
        'p1_measures': 39,
        'sys_slices': [(0, 17), (17, 28), (28, 39)],
    },
    {
        'slug': 'hw-mozart-k387',
        'xml': 'mozart-k387.musicxml',
        'img': 'handwritten/mozart-k387/p001.png',
        'work': 'String Quartet No.14 in G major, K.387 (Mvt I p.1)',
        'composer': 'Mozart, Wolfgang Amadeus',
        'license': 'Public Domain (IMSLP) + CC0-1.0 (OpenScore)',
        'p1_measures': 28,
        'sys_slices': [(0, 11), (11, 19), (19, 28)],
    },
    {
        'slug': 'hw-mozart-k421',
        'xml': 'mozart-k421.musicxml',
        'img': 'handwritten/mozart-k421/p001.png',
        'work': 'String Quartet No.15 in D minor, K.421 (Mvt I p.1)',
        'composer': 'Mozart, Wolfgang Amadeus',
        'license': 'Public Domain (IMSLP) + CC0-1.0 (OpenScore)',
        'p1_measures': 32,
        'sys_slices': [(0, 11), (11, 22), (22, 32)],
    },
    {
        'slug': 'hw-schubert-d810',
        'xml': 'schubert-d810.musicxml',
        'img': 'handwritten/schubert-d810/p002.png',
        'work': 'String Quartet in D minor, D.810 (Mvt I p.1)',
        'composer': 'Schubert, Franz',
        'license': 'Public Domain (IMSLP) + CC0-1.0 (OpenScore)',
        'p1_measures': 30,
        'sys_slices': [(0, 6), (6, 15), (15, 23), (23, 30)],
    },
]

TRAIN_SPECS = [
    {
        'slug': 'hw-mozart-k428',
        'xml': 'mozart-k428.musicxml',
        'img': 'handwritten/mozart-k428/p001.png',
        'work': 'String Quartet No.16 in E-flat major, K.428 (Mvt I p.1)',
        'sys_slices': [(0, 12), (12, 24), (24, 34)],
    },
    {
        'slug': 'hw-mozart-k458',
        'xml': 'mozart-k458.musicxml',
        'img': 'handwritten/mozart-k458/p001.png',
        'work': 'String Quartet No.17 in B-flat major, K.458 (Mvt I p.1)',
        'sys_slices': [(0, 12), (12, 24), (24, 35)],
    },
    {
        'slug': 'hw-mozart-k465',
        'xml': 'mozart-k465.musicxml',
        'img': 'handwritten/mozart-k465/p001.png',
        'work': 'String Quartet No.19 in C major, K.465 (Mvt I p.1)',
        'sys_slices': [(0, 12), (12, 26), (26, 36)],
    },
    {
        'slug': 'hw-mozart-k590',
        'xml': 'mozart-k590.musicxml',
        'img': 'handwritten/mozart-k590/p001.png',
        'work': 'String Quartet No.23 in F major, K.590 (Mvt I p.1)',
        'sys_slices': [(0, 11), (11, 20), (20, 29)],
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def slice_musicxml(src_xml: Path, dst_xml: Path, start_idx: int, end_idx: int) -> None:
    """Write a valid MusicXML containing measures [start_idx:end_idx] of every part.

    Carries forward active <divisions>, <key>, <time>, <clef> into the first
    kept measure when start_idx > 0, and strips <print new-page="yes"> so
    `assembly-serialize bekern --page 1` reads the entire slice as page 1.
    """
    tree = ET.parse(src_xml)
    root = tree.getroot()
    for part in root.findall('part'):
        measures = part.findall('measure')
        # Track active attributes up to start_idx
        last_div = None
        last_key = None
        last_time = None
        last_clefs = []
        for m in measures[:start_idx]:
            for attr in m.findall('attributes'):
                if attr.find('divisions') is not None:
                    last_div = copy.deepcopy(attr.find('divisions'))
                if attr.find('key') is not None:
                    last_key = copy.deepcopy(attr.find('key'))
                if attr.find('time') is not None:
                    last_time = copy.deepcopy(attr.find('time'))
                clefs = attr.findall('clef')
                if clefs:
                    last_clefs = [copy.deepcopy(c) for c in clefs]
        kept = measures[start_idx:end_idx]
        for m in measures:
            part.remove(m)
        for i, m in enumerate(kept):
            # Remove new-page / new-system breaks so all kept measures stay on page 1
            for pr in list(m.findall('print')):
                if 'new-page' in pr.attrib:
                    del pr.attrib['new-page']
                if 'new-system' in pr.attrib:
                    del pr.attrib['new-system']
            if i == 0 and start_idx > 0:
                attr = m.find('attributes')
                if attr is None:
                    attr = ET.Element('attributes')
                    m.insert(0, attr)
                if attr.find('divisions') is None and last_div is not None:
                    attr.insert(0, copy.deepcopy(last_div))
                if attr.find('key') is None and last_key is not None:
                    attr.append(copy.deepcopy(last_key))
                if attr.find('time') is None and last_time is not None:
                    attr.append(copy.deepcopy(last_time))
                if not attr.findall('clef') and last_clefs:
                    for c in last_clefs:
                        attr.append(copy.deepcopy(c))
            part.append(m)
    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_xml, encoding='utf-8', xml_declaration=True)


def run_photo_degrade(src_png: Path, dst_path: Path, seed: int, fmt: str = 'png') -> None:
    """Run photo-degrade on one image deterministically and move to dst_path."""
    with tempfile.TemporaryDirectory(prefix='hw_deg_') as td:
        td_p = Path(td)
        in_dir = td_p / 'in'
        out_dir = td_p / 'out'
        in_dir.mkdir()
        out_dir.mkdir()
        tmp_in = in_dir / src_png.name
        shutil.copy2(src_png, tmp_in)
        cmd = [
            str(PHOTO_DEGRADE),
            '--input', str(in_dir),
            '--out-dir', str(out_dir),
            '--seed', str(seed),
            '--format', fmt,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f'photo-degrade failed: {r.stderr[-300:]}')
        ext = '.jpg' if fmt == 'jpeg' else '.png'
        cands = list(out_dir.glob('*' + ext))
        if not cands:
            raise RuntimeError(f'photo-degrade produced no {ext} in {out_dir}')
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cands[0]), str(dst_path))


def build_eval_suite() -> list[dict]:
    hw_clean = SUITE31 / 'hw' / 'clean'
    hw_photo = SUITE31 / 'hw' / 'photo'
    hw_gt = SUITE31 / 'hw' / 'gt'
    for d in (hw_clean, hw_photo, hw_gt):
        d.mkdir(parents=True, exist_ok=True)

    hw_pieces = []
    for idx, sp in enumerate(EVAL_SPECS):
        slug = sp['slug']
        src_img = HW500 / sp['img']
        src_xml = XML_CACHE / sp['xml']
        dst_clean = hw_clean / f'{slug}.png'
        dst_photo = hw_photo / f'{slug}.png'
        dst_gt = hw_gt / f'gt-p1-{slug}.musicxml'

        shutil.copy2(src_img, dst_clean)
        run_photo_degrade(src_img, dst_photo, seed=20261123 + idx, fmt='png')
        slice_musicxml(src_xml, dst_gt, 0, sp['p1_measures'])

        # Verify assembly-serialize bekern --page 1 succeeds on the sliced GT
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run(
                [str(ASER), 'bekern', '--input', str(dst_gt), '--page', '1',
                 '--out', td, '--stem', slug],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode != 0:
                raise RuntimeError(f'bekern failed on {slug}: {r.stderr[-300:]}')
            ntok = len((Path(td) / f'{slug}-bekern.txt').read_text(encoding='utf-8').split())
            print(f'[EVAL] {slug}: p1_measures={sp["p1_measures"]}, bekern_tokens={ntok}')

        iw, ih = Image.open(dst_clean).size
        n_sys = len(sp['sys_slices'])
        top, bot = int(ih * 0.06), int(ih * 0.90)
        band = (bot - top) / n_sys
        sys_boxes = [
            [int(iw * 0.05), int(top + i * band), int(iw * 0.90), int(top + (i + 1) * band)]
            for i in range(n_sys)
        ]

        hw_pieces.append({
            'slug': slug,
            'handwritten': True,
            'work': sp['work'],
            'composer': sp['composer'],
            'license': sp['license'],
            'p1_measures': sp['p1_measures'],
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

    # Write companion HW-only manifest
    hw_only_manifest = {
        'suite': 'suite31-hw-only',
        'generated_by': 'tmp-train/mvd/build_real_hw_suite.py',
        'pieces_expected': len(hw_pieces),
        'license': 'Public Domain (IMSLP holographs) + CC0-1.0 (OpenScore StringQuartets); zero NC data',
        'pieces': hw_pieces,
    }
    (SUITE31 / 'suite31_hw_only_manifest.json').write_text(
        json.dumps(hw_only_manifest, indent=1), encoding='utf-8')

    # Write companion Suite31 + HW manifest (preserving frozen suite31_manifest.json untouched)
    base_manifest = json.loads((SUITE31 / 'suite31_manifest.json').read_text(encoding='utf-8'))
    combined = copy.deepcopy(base_manifest)
    combined['suite'] = 'suite31+hw'
    combined['generated_by'] = 'tmp-train/mvd/build_real_hw_suite.py'
    combined['pieces_expected'] = len(base_manifest['pieces']) + len(hw_pieces)
    combined['pieces'] = base_manifest['pieces'] + hw_pieces
    (SUITE31 / 'suite31_hw_manifest.json').write_text(
        json.dumps(combined, indent=1), encoding='utf-8')

    return hw_pieces


def build_train_strips() -> dict:
    renders = TRAIN_OUT / 'renders'
    targets = TRAIN_OUT / 'targets'
    renders.mkdir(parents=True, exist_ok=True)
    targets.mkdir(parents=True, exist_ok=True)

    bgate.ensure_weights_readable()
    hybrid_rows = []
    stats = {'works': 0, 'systems': 0, 'clean_pairs': 0, 'degraded_pairs': 0, 'hybrid_rows': 0}

    for w_idx, sp in enumerate(TRAIN_SPECS):
        slug = sp['slug']
        src_img = HW500 / sp['img']
        src_xml = XML_CACHE / sp['xml']
        w, h = Image.open(src_img).size
        try:
            staves, space = bsd.neural_detect_staves(str(src_img))
            groups = esg.visual_system_groups(staves, space, s_ref=4)
        except (ValueError, OSError):
            staves, space, groups = [], 0.0, None
        if not groups or len(groups) != len(sp['sys_slices']):
            # Fallback: evenly partition vertical music region into len(sys_slices) bands
            n_sys = len(sp['sys_slices'])
            top, bot = int(h * 0.06), int(h * 0.90)
            band = (bot - top) / n_sys
            boxes = [
                [int(w * 0.05), int(top + i * band), int(w * 0.90), int(top + (i + 1) * band)]
                for i in range(n_sys)
            ]
            box_src = f'uniform-fallback (detected {len(groups) if groups else 0} groups)'
        else:
            boxes = esg.pad_staff_boxes(groups, space, w, h)
            box_src = 'neural-4staff'

        print(f'[TRAIN] {slug}: {len(boxes)} systems ({box_src})')
        stats['works'] += 1

        for s_idx, (box, (m0, m1)) in enumerate(zip(boxes, sp['sys_slices'])):
            stem = f'{slug}-1-s{s_idx}'
            clean_png = renders / f'{stem}.png'
            bgate.crop_box(str(src_img), str(clean_png), box)

            # Slice exact system measures -> bekern
            with tempfile.TemporaryDirectory() as td:
                sys_xml = Path(td) / f'{stem}.musicxml'
                slice_musicxml(src_xml, sys_xml, m0, m1)
                r = subprocess.run(
                    [str(ASER), 'bekern', '--input', str(sys_xml), '--page', '1',
                     '--out', str(targets), '--stem', stem],
                    capture_output=True, text=True, timeout=60,
                )
                if r.returncode != 0:
                    raise RuntimeError(f'bekern failed on train {stem}: {r.stderr[-300:]}')

            stats['systems'] += 1
            stats['clean_pairs'] += 1

            # Generate 2 seeded phone-photo degraded variants per clean strip
            deg_names = []
            for d_idx in range(2):
                d_path = renders / f'{stem}_d{d_idx}.jpg'
                run_photo_degrade(clean_png, d_path, seed=20261200 + w_idx * 100 + s_idx * 10 + d_idx, fmt='jpeg')
                deg_names.append(d_path.name)
                stats['degraded_pairs'] += 1

            # Build hybrid_staff.jsonl rows for clean + degraded
            s_clean = {'id': slug, 'page': 1, 'sys_idx': s_idx,
                       'clean': clean_png.name, 'degraded': clean_png.name}
            try:
                c_row, _ = bhc._sysdev_one((str(renders), s_clean))
                if c_row and not c_row.get('skip') and c_row.get('table'):
                    hybrid_rows.append(c_row)
                    stats['hybrid_rows'] += 1
            except Exception as e:
                print(f'  warn: clean hybrid row failed for {stem}: {e}')

            for d_name in deg_names:
                s_deg = {'id': slug, 'page': 1, 'sys_idx': s_idx,
                         'clean': clean_png.name, 'degraded': d_name}
                try:
                    _, d_row = bhc._sysdev_one((str(renders), s_deg))
                    if d_row and not d_row.get('skip') and d_row.get('table'):
                        hybrid_rows.append(d_row)
                        stats['hybrid_rows'] += 1
                except Exception as e:
                    print(f'  warn: deg hybrid row failed for {d_name}: {e}')

    jpath = TRAIN_OUT / 'hybrid_staff.jsonl'
    with open(jpath, 'w', encoding='utf-8') as f:
        for row in hybrid_rows:
            f.write(json.dumps(row) + '\n')

    manifest = {
        'train_works': [sp['slug'] for sp in TRAIN_SPECS],
        'eval_works': [sp['slug'] for sp in EVAL_SPECS],
        'overlap': sorted(set(sp['slug'] for sp in TRAIN_SPECS) & set(sp['slug'] for sp in EVAL_SPECS)),
        'stats': stats,
    }
    assert not manifest['overlap'], 'train/eval work overlap!'
    (TRAIN_OUT / 'split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    hw_pieces = build_eval_suite()
    tr_manifest = build_train_strips()
    print('\n=== DONE ===')
    print('Eval pieces staged:', len(hw_pieces))
    print('Train manifest:', json.dumps(tr_manifest, indent=2))


if __name__ == '__main__':
    main()
