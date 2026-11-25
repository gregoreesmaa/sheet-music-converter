"""Run Docker oemer-compare:0.1.8 baseline on Suite31+HW handwritten pages."""
import json
import pathlib
import subprocess
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SUITE = REPO_ROOT / 'data-gen' / 'score-corpus' / 'suite31'
OUT_ROOT = pathlib.Path('D:/tmp/hw-align/oemer_hw5')
ASER = REPO_ROOT / 'converter' / 'assembly-serialize' / 'target' / 'release' / 'assembly-serialize.exe'
EHARNESS = REPO_ROOT / 'converter' / 'eval-harness' / 'target' / 'release' / 'eval-harness.exe'


def main():
    manifest = json.loads((SUITE / 'suite31_hw_only_manifest.json').read_text(encoding='utf-8'))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for pc in manifest['pieces']:
        slug = pc['slug']
        img_rel = pc['clean']['file']
        gt_xml = SUITE / pc['gt']['file']
        p_out = OUT_ROOT / slug
        p_out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        cmd = [
            'docker', 'run', '--rm',
            '-v', f'{SUITE}:/in:ro',
            '-v', f'{p_out}:/out',
            'oemer-compare:0.1.8',
            '-o', '/out', f'/in/{img_rel}',
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            rc = r.returncode
            err_tail = (r.stderr or r.stdout)[-250:]
        except subprocess.TimeoutExpired:
            rc = -1
            err_tail = 'timeout-240s'
        secs = round(time.time() - t0, 1)
        xmls = list(p_out.glob('*.musicxml')) + list(p_out.glob('*.xml'))
        row = {
            'piece': slug,
            'rc': rc,
            'secs': secs,
            'xml_found': bool(xmls),
            'strict_f1': 0.0,
            'notation_f1': 0.0,
        }
        if xmls:
            hyp_xml = xmls[0]
            nrep = p_out / 'notation.json'
            rn = subprocess.run(
                [str(EHARNESS), 'notation-f1', '--reference', str(gt_xml),
                 '--hypothesis', str(hyp_xml), '--report', str(nrep)],
                capture_output=True, text=True, timeout=60,
            )
            if rn.returncode == 0 and nrep.exists():
                nd = json.loads(nrep.read_text(encoding='utf-8'))
                row['notation_f1'] = float(nd.get('notation_f1', 0.0))
            ref_n = p_out / 'ref_notes'
            hyp_n = p_out / 'hyp_notes'
            ref_n.mkdir(exist_ok=True)
            hyp_n.mkdir(exist_ok=True)
            r1 = subprocess.run([str(ASER), 'import', '--input', str(gt_xml), '--out', str(ref_n)],
                                capture_output=True, text=True, timeout=60)
            r2 = subprocess.run([str(ASER), 'import', '--input', str(hyp_xml), '--out', str(hyp_n)],
                                capture_output=True, text=True, timeout=60)
            rj = list(ref_n.glob('*.json'))
            hj = list(hyp_n.glob('*.json'))
            if r1.returncode == 0 and r2.returncode == 0 and rj and hj:
                frep = p_out / 'strict.json'
                rf = subprocess.run(
                    [str(EHARNESS), 'strict-f1', '--reference', str(rj[0]),
                     '--hypothesis', str(hj[0]), '--report', str(frep)],
                    capture_output=True, text=True, timeout=60,
                )
                if rf.returncode == 0 and frep.exists():
                    fd = json.loads(frep.read_text(encoding='utf-8'))
                    row['strict_f1'] = float(fd.get('f1', 0.0))
        else:
            row['err'] = err_tail
        rows.append(row)
        print(
            f"oemer {slug}: rc={rc} secs={secs} xml={row['xml_found']} "
            f"strict_f1={row['strict_f1']:.4f} notation_f1={row['notation_f1']:.4f}",
            flush=True,
        )

    summary = {
        'tool': 'oemer-compare:0.1.8',
        'n_pieces': len(rows),
        'n_xml': sum(1 for x in rows if x['xml_found']),
        'strict_f1_mean': sum(x['strict_f1'] for x in rows) / max(1, len(rows)),
        'notation_f1_mean': sum(x['notation_f1'] for x in rows) / max(1, len(rows)),
        'rows': rows,
    }
    (OUT_ROOT / 'oemer_hw5_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
