"""Integrate CC-BY-4.0 HugoSchtr/debussy-omr (portrait handwritten corpus).

Strict work-level split (ZERO leakage across 37 works):
  - Held-out EVAL (4 portrait multi-system works, 1 page each -> Suite31+HW):
      12148_btv1b100760232 -> hw-debussy-vetement-du-blesse   (2 staves, 5 sys, 3674x4658)
      12148_btv1b10076394j -> hw-debussy-berceuse-heroique    (3 staves, 5 sys, 4354x6079)
      12148_btv1b52500675r -> hw-debussy-la-plus-que-lente    (2 staves, 6 sys, 3696x5064)
      12148_btv1b55000830z -> hw-debussy-lindaraja            (4 staves, 4 sys, 3358x4305)
  - Train (remaining 33 works):
      * System-level strips -> D:/tmp/real-hw-train-debussy/{renders,targets,hybrid_staff.jsonl,split_manifest.json}
      * Full-page images    -> D:/tmp/debussy-omr/train_pages/{*.png,manifest.json}

CPU-only (CUDA_VISIBLE_DEVICES=""). Never touches frozen suite31_manifest.json.
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
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_gate20_sys_dev as bgate  # noqa: E402
import build_hybrid_cache as bhc  # noqa: E402
import recanon_i1  # noqa: E402

ASER = REPO_ROOT / 'converter' / 'assembly-serialize' / 'target' / 'release' / 'assembly-serialize.exe'
PHOTO_DEGRADE = REPO_ROOT / 'data-gen' / 'photo-degrade' / 'target' / 'release' / 'photo-degrade.exe'
SUITE31 = REPO_ROOT / 'data-gen' / 'score-corpus' / 'suite31'

DEBUSSY_ROOT = Path(r'D:\tmp\debussy-omr')
CACHE_DIR = str(DEBUSSY_ROOT / 'hf_cache')
TRAIN_PAGES_DIR = DEBUSSY_ROOT / 'train_pages'
TRAIN_SYS_DIR = Path(r'D:\tmp\real-hw-train-debussy')

EVAL_SPECS = [
    {
        'document_id': '12148_btv1b100760232',
        'slug': 'hw-debussy-vetement-du-blesse',
        'work': 'Pièce pour l\'oeuvre "Vêtement du blessé" (p.1)',
        'composer': 'Debussy, Claude',
        'license': 'CC-BY-4.0 (HugoSchtr/debussy-omr, BnF Gallica)',
    },
    {
        'document_id': '12148_btv1b10076394j',
        'slug': 'hw-debussy-berceuse-heroique',
        'work': 'Berceuse Héroïque (p.1)',
        'composer': 'Debussy, Claude',
        'license': 'CC-BY-4.0 (HugoSchtr/debussy-omr, BnF Gallica)',
    },
    {
        'document_id': '12148_btv1b52500675r',
        'slug': 'hw-debussy-la-plus-que-lente',
        'work': 'La plus que lente (p.1)',
        'composer': 'Debussy, Claude',
        'license': 'CC-BY-4.0 (HugoSchtr/debussy-omr, BnF Gallica)',
    },
    {
        'document_id': '12148_btv1b55000830z',
        'slug': 'hw-debussy-lindaraja',
        'work': 'Lindaraja (p.1)',
        'composer': 'Debussy, Claude',
        'license': 'CC-BY-4.0 (HugoSchtr/debussy-omr, BnF Gallica)',
    },
]
EVAL_DOC_IDS = {sp['document_id'] for sp in EVAL_SPECS}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def clean_musicxml(xml_str: str, strip_pno: bool = False) -> tuple[str, int]:
    """Strip new-page/new-system print attributes (and optionally print-object='no').

    Returns (cleaned_xml_str, n_measures).
    """
    root = ET.fromstring(xml_str)
    parts = root.findall('part')
    n_meas = len(parts[0].findall('measure')) if parts else 0
    for part in parts:
        for m in part.findall('measure'):
            for pr in list(m.findall('print')):
                if 'new-page' in pr.attrib:
                    del pr.attrib['new-page']
                if 'new-system' in pr.attrib:
                    del pr.attrib['new-system']
            if strip_pno:
                for el in list(m):
                    if el.attrib.get('print-object') == 'no':
                        m.remove(el)
    return ET.tostring(root, encoding='unicode', xml_declaration=True), n_meas


def run_photo_degrade(src_png: Path, dst_path: Path, seed: int, fmt: str = 'png') -> None:
    """Run photo-degrade on one image deterministically (CPU) and move to dst_path."""
    with tempfile.TemporaryDirectory(prefix='deb_deg_') as td:
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


def convert_and_verify_bekern(xml_str: str, out_dir: Path, stem: str) -> tuple[bool, int, str]:
    """Convert MusicXML -> canonical post-I1 <stem>-bekern.txt and verify bekern-decode rc=0."""
    with tempfile.TemporaryDirectory(prefix='deb_bk_') as td:
        td_p = Path(td)
        xml_p = td_p / f'{stem}.musicxml'
        rescued = False
        try:
            cleaned, _ = clean_musicxml(xml_str, strip_pno=False)
        except Exception as e:
            return False, 0, f'xml_parse:{e}'
        xml_p.write_text(cleaned, encoding='utf-8')
        r1 = subprocess.run(
            [str(ASER), 'bekern', '--input', str(xml_p), '--page', '1',
             '--out', str(td_p), '--stem', stem],
            capture_output=True, text=True, timeout=60,
        )
        if r1.returncode != 0:
            try:
                cleaned2, _ = clean_musicxml(xml_str, strip_pno=True)
            except Exception as e:
                return False, 0, f'xml_parse2:{e}'
            xml_p.write_text(cleaned2, encoding='utf-8')
            r1b = subprocess.run(
                [str(ASER), 'bekern', '--input', str(xml_p), '--page', '1',
                 '--out', str(td_p), '--stem', stem],
                capture_output=True, text=True, timeout=60,
            )
            if r1b.returncode != 0:
                err = (r1.stderr or r1.stdout).strip().splitlines()
                return False, 0, (err[-1] if err else 'bekern_fail')[:160]
            rescued = True

        bk_tmp = td_p / f'{stem}-bekern.txt'
        if not bk_tmp.exists():
            return False, 0, 'no_bekern_file'
        txt = bk_tmp.read_text(encoding='utf-8').strip()
        toks = txt.split()
        if not toks:
            return False, 0, 'empty_bekern'

        # Ensure post-I1 canonical form
        if not recanon_i1.is_stream_canonical_i1(txt):
            new_toks, changed = recanon_i1.canonicalize_bekern_tokens(toks)
            if changed:
                txt = ' '.join(new_toks)
                bk_tmp.write_text(txt + '\n', encoding='utf-8')
                toks = new_toks
            if not recanon_i1.is_stream_canonical_i1(txt):
                return False, len(toks), 'non_i1_after_recanon'

        # Verify round-trip bekern-decode rc=0
        dec_p = td_p / f'{stem}-dec.json'
        r2 = subprocess.run(
            [str(ASER), 'bekern-decode', '--input', str(bk_tmp), '--out', str(dec_p)],
            capture_output=True, text=True, timeout=60,
        )
        if r2.returncode != 0 or not dec_p.exists():
            err = (r2.stderr or r2.stdout).strip().splitlines()
            return False, len(toks), 'decode_fail:' + (err[-1] if err else '')[:120]

        out_dir.mkdir(parents=True, exist_ok=True)
        final_p = out_dir / f'{stem}-bekern.txt'
        tmp_final = out_dir / f'{stem}-bekern.txt.tmp'
        tmp_final.write_text(txt + '\n', encoding='utf-8')
        os.replace(tmp_final, final_p)
        return True, len(toks), ('rescued_pno' if rescued else 'ok')


def build_eval_suite(fp_ds, sys_ds) -> list[dict]:
    hw_clean = SUITE31 / 'hw' / 'clean'
    hw_photo = SUITE31 / 'hw' / 'photo'
    hw_gt = SUITE31 / 'hw' / 'gt'
    for d in (hw_clean, hw_photo, hw_gt):
        d.mkdir(parents=True, exist_ok=True)

    # Index fullpages and systems by document_id
    fp_by_doc = defaultdict(list)
    for row in fp_ds:
        fp_by_doc[row['document_id']].append(row)
    sys_per_folio = defaultdict(int)
    for row in sys_ds:
        sys_per_folio[(row['document_id'], row['folio'])] += 1

    new_pieces = []
    for idx, sp in enumerate(EVAL_SPECS):
        doc_id = sp['document_id']
        slug = sp['slug']
        pages = sorted(fp_by_doc[doc_id], key=lambda r: r['folio'])
        assert pages, f'no fullpage found for {doc_id}'
        p0 = pages[0]
        img = p0['image'].convert('RGB')
        iw, ih = img.size
        assert ih > iw, f'Eval piece {slug} ({doc_id}) is NOT portrait: {iw}x{ih}'

        dst_clean = hw_clean / f'{slug}.png'
        dst_photo = hw_photo / f'{slug}.png'
        dst_gt = hw_gt / f'gt-p1-{slug}.musicxml'

        img.save(dst_clean, format='PNG')
        run_photo_degrade(dst_clean, dst_photo, seed=20261124 + idx, fmt='png')
        pw, ph = Image.open(dst_photo).size
        assert ph > pw, f'Photo-degraded {slug} is NOT portrait: {pw}x{ph}'

        cleaned_xml, n_meas = clean_musicxml(p0['musicxml'], strip_pno=False)
        dst_gt.write_text(cleaned_xml, encoding='utf-8')

        # Verify bekern + bekern-decode + import
        with tempfile.TemporaryDirectory(prefix='deb_ev_') as td:
            td_p = Path(td)
            ok, ntok, msg = convert_and_verify_bekern(cleaned_xml, td_p, slug)
            assert ok, f'Eval bekern+decode failed on {slug}: {msg}'
            imp_dir = td_p / 'imp'
            imp_dir.mkdir()
            ri = subprocess.run(
                [str(ASER), 'import', '--input', str(dst_gt), '--out', str(imp_dir)],
                capture_output=True, text=True, timeout=60,
            )
            assert ri.returncode == 0, f'Eval import failed on {slug}: {ri.stderr[-200:]}'

        n_sys = sys_per_folio[(doc_id, p0['folio'])]
        assert n_sys >= 2, f'Expected multi-system page for {slug}, got {n_sys}'
        top, bot = int(ih * 0.06), int(ih * 0.90)
        band = (bot - top) / n_sys
        sys_boxes = [
            [int(iw * 0.05), int(top + i * band), int(iw * 0.90), int(top + (i + 1) * band)]
            for i in range(n_sys)
        ]

        print(f'[EVAL] {slug} ({doc_id} f{p0["folio"]}): {iw}x{ih} (photo {pw}x{ph}), '
              f'n_sys={n_sys}, p1_meas={n_meas}, bekern_toks={ntok}')

        new_pieces.append({
            'slug': slug,
            'handwritten': True,
            'orientation': 'portrait',
            'document_id': doc_id,
            'folio': int(p0['folio']),
            'work': sp['work'],
            'composer': sp['composer'],
            'license': sp['license'],
            'p1_measures': n_meas,
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

    # Update suite31_hw_only_manifest.json (keep existing 5 landscape HW, append 4 portrait Debussy)
    hw_only_path = SUITE31 / 'suite31_hw_only_manifest.json'
    existing_hw_only = json.loads(hw_only_path.read_text(encoding='utf-8'))
    base_hw = [p for p in existing_hw_only['pieces'] if not p['slug'].startswith('hw-debussy-')]
    assert len(base_hw) == 5, f'Expected 5 existing landscape HW pieces, found {len(base_hw)}'
    all_hw = base_hw + new_pieces
    assert len(all_hw) == 9

    hw_only_out = {
        'suite': 'suite31-hw-only',
        'generated_by': 'tmp-train/mvd/build_real_hw_suite.py + tmp-train/mvd/build_debussy_hw.py',
        'pieces_expected': len(all_hw),
        'license': 'Public Domain (IMSLP holographs) + CC0-1.0 (OpenScore StringQuartets) + CC-BY-4.0 (HugoSchtr/debussy-omr, BnF Gallica); zero NC data',
        'pieces': all_hw,
    }
    hw_only_path.write_text(json.dumps(hw_only_out, indent=1), encoding='utf-8')

    # Update suite31_hw_manifest.json (31 frozen printed + 9 HW = 40; NEVER modify suite31_manifest.json)
    base_manifest = json.loads((SUITE31 / 'suite31_manifest.json').read_text(encoding='utf-8'))
    assert len(base_manifest['pieces']) == 31
    combined = copy.deepcopy(base_manifest)
    combined['suite'] = 'suite31+hw'
    combined['generated_by'] = 'tmp-train/mvd/build_real_hw_suite.py + tmp-train/mvd/build_debussy_hw.py'
    combined['pieces_expected'] = len(base_manifest['pieces']) + len(all_hw)
    combined['pieces'] = base_manifest['pieces'] + all_hw
    assert len(combined['pieces']) == 40
    (SUITE31 / 'suite31_hw_manifest.json').write_text(
        json.dumps(combined, indent=1), encoding='utf-8')

    return new_pieces


def _detect_hybrid_one(png_path: Path) -> dict | None:
    try:
        S, lines5, sp_clean, src = bhc.clean_lines(str(png_path))
        if S < 1 or not lines5 or len(lines5) != S:
            return None
        w_clean = Image.open(png_path).size[0]
        if w_clean > bhc.MAX_SIDECAR_WIDTH:
            return None
        table = bhc.rows_to_table(lines5, w_clean)
        if not table or not table.get('staves'):
            return None
        return {
            'img': png_path.name,
            'S': int(S),
            'kind': 'rows',
            'table': table,
            'sp': round(float(sp_clean), 2),
            'source': src,
            'skip': False,
        }
    except Exception:
        return None


def export_train_systems(sys_ds) -> dict:
    renders = TRAIN_SYS_DIR / 'renders'
    targets = TRAIN_SYS_DIR / 'targets'
    renders.mkdir(parents=True, exist_ok=True)
    targets.mkdir(parents=True, exist_ok=True)

    train_items = []
    train_docs = set()
    for i, row in enumerate(sys_ds):
        doc_id = row['document_id']
        if doc_id in EVAL_DOC_IDS:
            continue
        train_docs.add(doc_id)
        doc_short = doc_id.split('_')[-1]
        folio = int(row['folio'])
        sys_num = int(row['system_number'])
        stem = f'hw-debussy-{doc_short}-{folio}-s{sys_num}'
        train_items.append((i, doc_id, folio, sys_num, row['filename'], stem))

    # Convert XML -> bekern in parallel first
    def _conv_item(item):
        i, doc_id, folio, sys_num, orig_fn, stem = item
        row = sys_ds[i]
        ok, ntok, msg = convert_and_verify_bekern(row['musicxml'], targets, stem)
        if not ok:
            return (item, False, ntok, msg)
        png_path = renders / f'{stem}.png'
        row['image'].convert('RGB').save(png_path, format='PNG')
        return (item, True, ntok, msg)

    converted_pngs = []
    skipped = []
    rescued = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for item, ok, ntok, msg in ex.map(_conv_item, train_items):
            if ok:
                converted_pngs.append(renders / f'{item[5]}.png')
                if msg == 'rescued_pno':
                    rescued += 1
            else:
                skipped.append({
                    'document_id': item[1],
                    'filename': item[4],
                    'stem': item[5],
                    'reason': msg,
                })

    print(f'[TRAIN-SYS] converted={len(converted_pngs)}/{len(train_items)} '
          f'(rescued_pno={rescued}, skipped={len(skipped)}) across {len(train_docs)} works')

    # Build validated hybrid_staff.jsonl on CPU
    bgate.ensure_weights_readable()
    hybrid_rows = []
    # Run sequentially or small thread pool (RowNet CPU torch uses OpenMP threads)
    for idx, png_path in enumerate(converted_pngs):
        row = _detect_hybrid_one(png_path)
        if row is not None:
            hybrid_rows.append(row)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(converted_pngs):
            print(f'  [HYBRID] {idx + 1}/{len(converted_pngs)} processed, '
                  f'valid_rows={len(hybrid_rows)}', flush=True)

    jpath = TRAIN_SYS_DIR / 'hybrid_staff.jsonl'
    with open(jpath, 'w', encoding='utf-8') as f:
        for row in hybrid_rows:
            f.write(json.dumps(row) + '\n')

    overlap = sorted(train_docs & EVAL_DOC_IDS)
    assert not overlap, f'Train/eval work overlap: {overlap}'

    manifest = {
        'dataset': 'HugoSchtr/debussy-omr-system-lvl (CC-BY-4.0)',
        'train_works': sorted(train_docs),
        'eval_works': sorted(EVAL_DOC_IDS),
        'eval_slugs': [sp['slug'] for sp in EVAL_SPECS],
        'overlap': overlap,
        'stats': {
            'train_works_count': len(train_docs),
            'eval_works_count': len(EVAL_DOC_IDS),
            'train_systems_total': len(train_items),
            'train_systems_converted': len(converted_pngs),
            'train_systems_rescued_pno': rescued,
            'train_systems_skipped': len(skipped),
            'hybrid_staff_rows': len(hybrid_rows),
        },
        'skipped': skipped,
    }
    (TRAIN_SYS_DIR / 'split_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def export_train_pages(fp_ds) -> dict:
    TRAIN_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    pages_meta = []
    train_docs = set()
    n_port = 0
    n_land = 0
    for row in fp_ds:
        doc_id = row['document_id']
        if doc_id in EVAL_DOC_IDS:
            continue
        train_docs.add(doc_id)
        img = row['image'].convert('RGB')
        w, h = img.size
        portrait = h > w
        if portrait:
            n_port += 1
        else:
            n_land += 1
        fn = f'{row["filename"]}.png'
        out_p = TRAIN_PAGES_DIR / fn
        img.save(out_p, format='PNG')
        pages_meta.append({
            'document_id': doc_id,
            'folio': int(row['folio']),
            'filename': fn,
            'width': w,
            'height': h,
            'portrait': portrait,
        })

    overlap = sorted(train_docs & EVAL_DOC_IDS)
    assert not overlap, f'Fullpage train/eval work overlap: {overlap}'
    info = {
        'dataset': 'HugoSchtr/debussy-omr-fullpage-lvl (CC-BY-4.0)',
        'train_works': sorted(train_docs),
        'eval_works_excluded': sorted(EVAL_DOC_IDS),
        'overlap': overlap,
        'total_pages': len(pages_meta),
        'portrait_pages': n_port,
        'landscape_pages': n_land,
        'pages': pages_meta,
    }
    (TRAIN_PAGES_DIR / 'manifest.json').write_text(
        json.dumps(info, indent=2), encoding='utf-8')
    print(f'[TRAIN-PAGES] exported {len(pages_meta)} pages '
          f'(portrait={n_port}, landscape={n_land}) across {len(train_docs)} works')
    return info


def main():
    frozen_path = SUITE31 / 'suite31_manifest.json'
    frozen_sha_before = sha256_file(frozen_path)

    fp_ds = load_dataset('HugoSchtr/debussy-omr-fullpage-lvl', split='test', cache_dir=CACHE_DIR)
    sys_ds = load_dataset('HugoSchtr/debussy-omr-system-lvl', split='test', cache_dir=CACHE_DIR)

    eval_pieces = build_eval_suite(fp_ds, sys_ds)
    train_pages_info = export_train_pages(fp_ds)
    train_sys_manifest = export_train_systems(sys_ds)

    frozen_sha_after = sha256_file(frozen_path)
    assert frozen_sha_before == frozen_sha_after, 'FROZEN suite31_manifest.json WAS MODIFIED!'

    print('\n=== ALL DONE ===')
    print(f'Frozen suite31_manifest.json SHA256 unchanged: {frozen_sha_after}')
    print(f'Eval portrait Debussy pieces added: {len(eval_pieces)}')
    print(f'Train fullpages exported: {train_pages_info["total_pages"]} '
          f'(portrait={train_pages_info["portrait_pages"]}, landscape={train_pages_info["landscape_pages"]})')
    print(f'Train systems stats: {json.dumps(train_sys_manifest["stats"], indent=2)}')


if __name__ == '__main__':
    main()
