"""Measure-atomic beam-5 re-decode repair loop (S18-adjacent, GPU-gated).

Pure repair core (:func:`repair_measures`) is CPU-testable with stub
nets; the runner (:func:`main`) replays failing strips on the real
checkpoint behind the standard GPU gate (`--i-understand-gpu` +
``GUARD_GPU_OK=1`` + CUDA device) and never trains.

Protocol (per spec): split the beam-3 rollout into measures; keep
strict-closing measures untouched; re-decode ONLY failed spans at
beam=5 (span budget ``min(len+32, 128)``); cut each beam to its first
measure body; rank with ``guard_rescore.rank_beams_by_closure``; splice
the first strict-closing candidate; else fall back to the existing
``_repair_measure`` (trim / ``bar-rest``). No prefix-forcing: ranking
needs no net change.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data  # noqa: E402
import guard_rescore as gr  # noqa: E402
import measure_decode as mdec  # noqa: E402


def first_measure_body(toks):
    """Tokens up to (excluding) the first ``bar``."""
    out = []
    for t in toks:
        if t == 'bar':
            break
        out.append(t)
    return out


def _row_to_toks(row):
    """One id row (torch or list) -> bekern tokens, cut at specials."""
    ids = row.tolist() if hasattr(row, 'tolist') else list(row)
    toks = []
    for i in ids:
        i = int(i)
        if 0 <= i < len(data.VOCAB):
            toks.append(data.VOCAB[i])
        else:
            break
    return toks


def _beam_rows(out, beam):
    """Normalize a ``beam_search(return_all=True)`` result to id rows.

    Real nets return a ``[B, beam, T]`` tensor (batch-first, mirroring
    the S6 ``out[0]`` usage); stub nets in tests return ``[beam, T]``
    lists. Both land here as a list of at most ``beam`` rows.
    """
    try:
        import torch
        is_torch = torch.is_tensor(out)
    except ImportError:  # pragma: no cover - torch always present here
        is_torch = False
    rows = out[0].tolist() if is_torch else list(out[0] if (
        out and isinstance(out[0], (list, tuple)) and out[0]
        and isinstance(out[0][0], (list, tuple))) else [out[0] if out else []])
    if not rows:
        return []
    if isinstance(rows[0], int):
        return [rows]
    return [list(r) for r in rows[:beam]]


def repair_measures(net, ten, mem_mask, header, measures, trailing_bar,
                    beam=5, span_headroom=32, span_cap=128, strict=True,
                    **beam_kwargs):
    """Re-decode failed measures at beam width; return (flat, provenance).

    ``measures`` is the split body list; ``provenance`` aligns with it:
    ``kept`` (strict-closed, no decode), ``beam5[k]`` (spliced
    k-th-ranked closing candidate), ``trimmed`` / ``spliced`` (existing
    :func:`_repair_measure` fallback). Flat output has the
    :func:`join_measures` shape.
    """
    fixed, prov = [], []
    for m in measures:
        if mdec.measure_closes(header, m):
            fixed.append(m)
            prov.append('kept')
            continue
        max_len = min(len(m) + span_headroom, span_cap)
        kw = dict(strict=strict, max_len=max_len, return_all=True)
        if mem_mask is not None:
            kw['mem_mask'] = mem_mask
        kw.update(beam_kwargs)
        out = net.beam_search(ten, beam=beam, **kw)
        cands = []
        for row in _beam_rows(out, beam):
            body = first_measure_body(_row_to_toks(row))
            if not body:
                continue
            try:
                if mdec._measure_closes_loose(header, body):
                    cands.append(body)
            except ValueError:
                continue
        ranked = gr.rank_beams_by_closure(header, cands)
        picked = None
        for k, cand in enumerate(ranked):
            try:
                if mdec.measure_closes(header, cand):
                    picked = (k, cand)
                    break
            except ValueError:
                continue
        if picked is not None:
            k, cand = picked
            fixed.append(cand)
            prov.append('beam5[%d]' % k)
            continue
        r = mdec._repair_measure(header, m)
        fixed.append(r)
        if r == ['bar-rest'] and m != ['bar-rest']:
            prov.append('spliced')
        elif r != m:
            prov.append('trimmed')
        else:
            prov.append('kept')
    return mdec.join_measures(header, fixed, trailing_bar), prov


def _load_net(ckpt):
    import torch
    import model as M
    import train as T
    net = M.MvdNet(vocab=len(data.VOCAB) + 3, anisotropic=True,
                   stride='4x8', extra_chans=8)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError:
        T.load_weights_compat(net, state)
    net.eval()
    return net


SYSDEV_CACHE = 'D:/tmp/t32hybrid-full/hybrid_staff_sysdev.jsonl'
T32_CACHE = 'D:/tmp/t32hybrid-full/hybrid_staff.jsonl'


def _t32_rows_for(want):
    """Stream the 2.3 GB t32 geometry index, keeping only wanted rows."""
    import json as _json
    got = {}
    with open(T32_CACHE, encoding='utf-8') as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                r = _json.loads(ln)
            except ValueError:
                continue
            if (r.get('img') in want and not r.get('skip')
                    and isinstance(r.get('table'), dict)):
                got[r['img']] = r
                if len(got) == len(want):
                    break
    return got


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tags-json', required=True,
                    help='per-strip tags, e.g. D:/tmp/eyeball90/f1_87.json')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--beam', type=int, default=5)
    ap.add_argument('--span-headroom', type=int, default=32)
    ap.add_argument('--picks-json', default='D:/tmp/eyeball90_picks.json')
    ap.add_argument('--i-understand-gpu', action='store_true')
    a = ap.parse_args(argv)
    if not a.i_understand_gpu:
        print('GPU-GATED: re-run with --i-understand-gpu on an '
              'operator-confirmed idle box. Not executed.')
        raise SystemExit(2)
    if os.environ.get('GUARD_GPU_OK') != '1':
        raise RuntimeError('GPU-GATED: GUARD_GPU_OK=1 required')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU-GATED: no CUDA device')
    import eval_sys_gate as E
    import heat_features as HF
    device = 'cuda'
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    spec = json.load(open(a.tags_json))
    if isinstance(spec, list):
        tags = [r['tag'] if isinstance(r, dict) else r for r in spec]
    else:
        tags = [r['tag'] for r in spec.get('rows', spec.get('results', []))]
    picks = {p['kind'] + '-' + p['stem'] + '-' + p['domain']: p
             for p in json.load(open(a.picks_json))['picks']}
    picks = {t: picks[t] for t in tags if t in picks}
    man_rows = {}
    if any(p['kind'] == 'sysdev' for p in picks.values()):
        import re as _re
        mani = json.load(open(E.SYS_DEV / 'sys_dev_manifest.json'))
        for s in mani['systems']:
            man_rows['%s-%d-s%d' % (s['id'], s['page'], s['sys_idx'])] = s
        for p in picks.values():
            if p['kind'] != 'sysdev' or 'img' in p:
                continue
            m = _re.match(r'(.+)-(\d+)-s(\d+)$', p['stem'])
            s = man_rows['%s-%s-s%s' % (m.group(1), m.group(2), m.group(3))]
            p['img'] = s['clean' if p['domain'] == 'clean' else 'degraded']
    T32DIR = E.REPO_ROOT / 'tmp-train' / 'mvd-sysdata-t32'
    sys_geom = E.load_geometry_index(SYSDEV_CACHE)
    t32_geom = _t32_rows_for(
        {p['img'] for p in picks.values() if p['kind'] == 't32'})
    net = _load_net(a.ckpt)
    heat = HF.HeatNet(
        str(E.REPO_ROOT / 'tmp-train' / 'det-data' / 'heatnet.pt'),
        device=device)
    workdir = Path(tempfile.mkdtemp(prefix='beam5repair_'))
    n = 0
    for tag, p in picks.items():
        rec = {'tag': tag}
        try:
            kind = p['kind']
            if kind == 'sysdev':
                img = str(E.SYS_DEV / p['img'])
                row = sys_geom.get(os.path.basename(img))
            else:
                img = str(T32DIR / p['img'])
                row = t32_geom.get(p['img'])
            staves, geom_src = E.rect_staves_for(img, row)
            if staves is None:
                rec.update(zeroed=True,
                            zero_reason='no-geometry:' + geom_src)
                json.dump(rec, open(outdir / (tag + '.json'), 'w'))
                print('%s SKIP %s' % (tag, geom_src), flush=True)
                continue
            ten = E.rectified_input_tensor(img, staves, heat, device,
                                           256, True)
            mem_mask = mdec.mask_for_strip(ten, len(staves), stride='4x8')
            first = mdec.decode_system_with_measure_fallback(
                net, ten, beam=3, mem_mask=mem_mask)
            header, measures, trailing = mdec.split_measures(first)
            fixed, prov = repair_measures(
                net, ten, mem_mask, header, measures, trailing,
                beam=a.beam, span_headroom=a.span_headroom)
            (workdir / (tag + '-hyp-bekern.txt')).write_text(
                ' '.join(fixed) + '\n', encoding='utf-8')
            rec.update(zeroed=False, geom=geom_src, provenance=prov,
                       n_measures=len(measures),
                       n_repaired=sum(1 for x in prov if x != 'kept'))
            json.dump(rec, open(outdir / (tag + '.json'), 'w'))
            print('%s repaired=%d/%d geom=%s' % (
                tag, rec['n_repaired'], len(measures), geom_src), flush=True)
            n += 1
        except Exception as e:  # noqa: BLE001 - record, never crash run
            rec.update(zeroed=True,
                       zero_reason='%s: %s' % (type(e).__name__, e))
            json.dump(rec, open(outdir / (tag + '.json'), 'w'))
            print('%s CRASH %s' % (tag, e), flush=True)
    print('DONE decoded=%d/%d -> %s' % (n, len(picks), outdir), flush=True)
    return 0


if __name__ == '__main__':
    main()
