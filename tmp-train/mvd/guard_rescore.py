"""Post-decode guard filters + stored-hyp rescore harness (CPU).

Context: ckpt-22000 flat-AR reader; Gate-A FAIL (suite31 photo 0.333);
87-strip binding-protocol census in D:/tmp/eyeball90/ shows attachment
spray (stacc 4.8x + slur 3.2x hyp-side) while 0/87 strips hit early EOS.
Per the AGY verdict the HeatNet-peak EOS veto (spike S1, ``ink_eos_veto``)
is KILLED -- it solves a non-problem -- and its signal is reallocated to
attachment-spray grounding. Nothing in this module sets ``ink_eos_veto``.

Filters (all pure functions on token lists, all behind flags in
:func:`apply_guards` -- default OFF):

* :func:`voice_cap_filter` -- cap concurrent open voices at the header
  staff count (clef tokens). Over-cap ``voice-N`` markers are dropped so
  the run merges into the current voice; pitched content is preserved.
* :func:`rest_guard_filter` -- drop ``rest DUR [dot]`` events from a
  measure voice-run that contains zero pitched-note events (``bar-rest``
  is always kept).
* :func:`spray_cap_filter` -- attachment-spray grounding prototype:
  at most ``artic_cap`` articulations (stacc/accent/tenuto) per notehead,
  slur pairs covering < ``slur_min_cover`` pitched notes are dropped,
  at most one open slur per voice (double-opens dropped, stops always
  kept). ``peaks`` is the HeatNet-notehead-evidence hook: a per-notehead
  strength list aligned to pitched-note events; notes below ``peak_min``
  lose all articulations. ``peaks=None`` (no stored peaks) selects the
  pure token-cap path used by the stored-hyp rescore.

:func:`rank_beams_by_closure` ranks candidate beams by strict
measure-closure count for the measure-atomic beam-5 repair loop (see
``D:/tmp/guard_rescore_beam5_spec.md``).

Stored-hyp rescore (CPU, no decoding): :func:`rescore_tags` re-walks the
87 stored ref/hyp pairs from ``D:/tmp/eyeball90/`` + temp workdirs with
:class:`grammar.PrefixState`, applies the guarded filters, and recomputes
note-level F1 through the repo binaries (``assembly-serialize`` +
``eval-harness``), exactly matching the binding recipe. Run::

    .\\.venv\\Scripts\\python.exe tmp-train\\mvd\\guard_rescore.py \\
        --outdir D:/tmp/guard_rescore
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grammar  # noqa: E402

REPO = Path('D:/projects/sheet-music-converter')
ASER = (REPO / 'converter/assembly-serialize/target/release'
        / 'assembly-serialize.exe')
EHARNESS = (REPO / 'converter/eval-harness/target/release'
            / 'eval-harness.exe')

DURS = frozenset(['whole', 'half', 'quarter', 'eighth', '16th', '32nd',
                  '64th', '128th', 'breve'])
ARTICS = frozenset(['stacc', 'accent', 'tenuto'])

__all__ = [
    'header_clef_count',
    'voice_cap_filter',
    'rest_guard_filter',
    'spray_cap_filter',
    'apply_guards',
    'rank_beams_by_closure',
    'walks_loose',
    'load_hyp_map',
    'resolve_ref',
    'rescore_tags',
]


def walks_loose(toks):
    """True iff tokens walk clean under non-strict PrefixState."""
    st = grammar.PrefixState(strict=False)
    try:
        for t in toks:
            st.step(t)
    except ValueError:
        return False
    return True


def _split_measures(toks):
    """Local (header, measures) split on clef/key/time + bar."""
    toks = list(toks)
    i = 0
    while (i < len(toks)
           and (toks[i] in grammar.CLEFS or toks[i] in grammar.KEYS
                or toks[i] in grammar.TIMES)):
        i += 1
    header, body = toks[:i], toks[i:]
    measures, cur = [], []
    for t in body:
        if t == grammar.BAR:
            measures.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        measures.append(cur)
    return header, measures


def _voice_of(tok):
    if tok.startswith('voice-'):
        try:
            return int(tok.split('-')[1])
        except ValueError:
            return None
    return None


def header_clef_count(toks):
    """Number of clef tokens in the leading header run."""
    n = 0
    for t in toks:
        if t in grammar.CLEFS:
            n += 1
        elif t in grammar.KEYS or t in grammar.TIMES:
            continue
        else:
            break
    return n


def voice_cap_filter(toks, cap=None):
    """Drop over-cap voice markers (runs merge into current voice).

    ``cap=None`` resolves to the header clef count; a header with no
    clefs fails open (returns input unchanged, ``cap=None`` reported).
    """
    toks = list(toks)
    if cap is None:
        cap = header_clef_count(toks)
        if cap <= 0:
            return toks, {'cap': None, 'removed_markers': 0,
                          'overcap_measures': 0}
    header, measures = _split_measures(toks)
    rep = {'cap': cap, 'removed_markers': 0, 'overcap_measures': 0}
    fixed = []
    for m in measures:
        out, hit = [], False
        for t in m:
            v = _voice_of(t)
            if v is not None and v > cap:
                rep['removed_markers'] += 1
                hit = True
                continue
            out.append(t)
        if hit:
            rep['overcap_measures'] += 1
        fixed.append(out)
    trailing = len(toks) > 0 and toks[-1] == grammar.BAR
    res = list(header)
    for j, m in enumerate(fixed):
        res.extend(m)
        if j < len(fixed) - 1 or trailing:
            res.append(grammar.BAR)
    return res, rep


def _scan_run_events(run):
    """(pitched_events, rest_spans) for one voice-run token list.

    rest_spans are (start, end) index spans of ``rest DUR [dot]*``.
    A DURATIONS token counts as pitched unless consumed as a rest
    duration or a chord-tone continuation is handled by the caller.
    """
    pitched = 0
    rest_spans = []
    i, n = 0, len(run)
    while i < n:
        t = run[i]
        if t == grammar.REST:
            j = i + 1
            if j < n and run[j] in DURS:
                j += 1
                while j < n and run[j] == grammar.DOT:
                    j += 1
                rest_spans.append((i, j))
                i = j
                continue
            i += 1
        elif t in DURS:
            pitched += 1
            i += 1
        else:
            i += 1
    return pitched, rest_spans


def rest_guard_filter(toks):
    """Drop rest events from voice-runs with zero pitched notes.

    ``bar-rest`` is always kept. A run emptied by rest removal loses
    its voice marker too (except the implicit voice-1 lead run).
    """
    toks = list(toks)
    header, measures = _split_measures(toks)
    rep = {'removed_rests': 0, 'removed_markers': 0}
    fixed = []
    for m in measures:
        runs = []
        cur = [None, []]
        for t in m:
            if _voice_of(t) is not None:
                runs.append(cur)
                cur = [t, []]
            else:
                cur[1].append(t)
        runs.append(cur)
        out = []
        for marker, run in runs:
            pitched, spans = _scan_run_events(run)
            if pitched == 0 and spans:
                drop = set()
                for a, b in spans:
                    drop.update(range(a, b))
                run = [t for k, t in enumerate(run) if k not in drop]
                rep['removed_rests'] += len(spans)
                if not run and marker is not None:
                    rep['removed_markers'] += 1
                    continue
            if marker is not None:
                out.append(marker)
            out.extend(run)
        fixed.append(out)
    trailing = len(toks) > 0 and toks[-1] == grammar.BAR
    res = list(header)
    for j, m in enumerate(fixed):
        res.extend(m)
        if j < len(fixed) - 1 or trailing:
            res.append(grammar.BAR)
    return res, rep


def spray_cap_filter(toks, artic_cap=1, slur_min_cover=2, peaks=None,
                     peak_min=0.5):
    """Ground attachment spray on notehead evidence (prototype).

    Token-cap path (``peaks=None``): per pitched-note event keep at
    most ``artic_cap`` of stacc/accent/tenuto; drop slur-start/stop
    pairs covering fewer than ``slur_min_cover`` pitched notes; drop a
    slur-start while another slur is already open on that voice. A
    surviving span's slur-stop is never dropped (structural); a
    degenerate pair is removed as a whole (start retroactively cut
    from the buffer when its stop closes short, dangling stops kept).

    Peak path (``peaks`` given): per-notehead strengths aligned to
    pitched-note events in order; an event below ``peak_min`` loses
    all its articulations. Missing entries fail open (kept).
    """
    toks = list(toks)
    rep = {'removed_artic': 0, 'removed_slur_pairs': 0,
           'removed_slur_double_open': 0, 'removed_unrounded': 0}
    out = []
    cur_voice = 1
    slur_open = collections.Counter()
    kept_starts = []  # stack of [voice, notes-at-start, out_idx]
    notes = 0
    event_idx = -1   # pitched-note event ordinal (chords share one)
    artic_used = 0
    event_live = False
    prev = None

    def _close_top():
        top_voice, n0, pos = kept_starts.pop()
        slur_open[top_voice] = max(0, slur_open[top_voice] - 1)
        if notes - n0 < slur_min_cover:
            # degenerate: retroactively cut the start, skip the stop
            del out[pos]
            for rec in kept_starts:
                if rec[2] > pos:
                    rec[2] -= 1
            rep['removed_slur_pairs'] += 1
            return False
        return True

    for t in toks:
        if t == grammar.BAR:
            # unclosed starts die at the barline (degenerate by fiat)
            while kept_starts:
                _close_top()
            cur_voice = 1
            slur_open.clear()
            event_live = False
            out.append(t)
        elif _voice_of(t) is not None:
            cur_voice = _voice_of(t)
            event_live = False
            out.append(t)
        elif t in (grammar.REST, grammar.BAR_REST) or t in grammar.CLEFS \
                or t in grammar.KEYS or t in grammar.TIMES:
            event_live = False
            out.append(t)
        elif t in DURS and prev != grammar.CHORD and prev != grammar.REST:
            event_idx += 1
            notes += 1
            artic_used = 0
            event_live = True
            out.append(t)
        elif t in ARTICS and event_live:
            weak = (peaks is not None and event_idx < len(peaks)
                    and peaks[event_idx] < peak_min)
            if weak:
                rep['removed_unrounded'] += 1
            elif artic_used >= artic_cap:
                rep['removed_artic'] += 1
            else:
                artic_used += 1
                out.append(t)
                prev = t
                continue
        elif t == 'slur-start':
            if slur_open[cur_voice] > 0:
                rep['removed_slur_double_open'] += 1
            else:
                slur_open[cur_voice] += 1
                kept_starts.append([cur_voice, notes, len(out)])
                out.append(t)
                prev = t
                continue
        elif t == 'slur-stop':
            if kept_starts:
                if _close_top():
                    out.append(t)
                    prev = t
                    continue
            else:
                # dangling stop: keep (walker consumes it as tail)
                out.append(t)
                prev = t
                continue
        else:
            out.append(t)
            prev = t
            continue
        prev = t
    while kept_starts:
        _close_top()
    return out, rep


def apply_guards(toks, enable_voice_cap=False, voice_cap=None,
                 enable_rest_guard=False, enable_spray_cap=False,
                 spray_kwargs=None):
    """Apply enabled guards in order; fail open on invalid output.

    Order is rest-guard first (it needs the original voice attribution:
    a sprayed ``voice-N rest`` run must be judged before voice-cap
    merging folds it into a pitched run), then voice-cap, then
    spray-cap. Returns (tokens, report). With no guard enabled the
    input is returned unchanged with ``valid=False`` (nothing
    verified). ``valid=True`` means at least one guard ran AND the
    filtered stream walks loose-clean.
    """
    toks = list(toks)
    report = {'applied': [], 'valid': False, 'details': {}}
    out = toks
    if enable_rest_guard:
        out, rep = rest_guard_filter(out)
        report['applied'].append('rest_guard')
        report['details']['rest_guard'] = rep
    if enable_voice_cap:
        out, rep = voice_cap_filter(out, cap=voice_cap)
        report['applied'].append('voice_cap')
        report['details']['voice_cap'] = rep
    if enable_spray_cap:
        out, rep = spray_cap_filter(out, **(spray_kwargs or {}))
        report['applied'].append('spray_cap')
        report['details']['spray_cap'] = rep
    if not report['applied']:
        return toks, report
    if not walks_loose(out):
        bad = dict(report)
        bad['valid'] = False
        return toks, bad
    report['valid'] = True
    return out, report


def rank_beams_by_closure(header, beams):
    """Rank candidate beams: most strict-closing measures first.

    Stable (ties keep input order). Empty input -> [].
    """
    from measure_decode import (  # lazy: keeps import light (torch)
        measure_closes, split_measures)
    scored = []
    for b in beams:
        _, measures, _ = split_measures(list(header) + list(b))
        n = sum(1 for m in measures if measure_closes(list(header), m))
        scored.append((n, b))
    scored.sort(key=lambda p: -p[0])
    return [b for _, b in scored]


# ---------------------------------------------------------------------------
# Stored-hyp rescore harness (CPU; no decoding, scoring via repo binaries).
# ---------------------------------------------------------------------------

def load_hyp_map():
    """Map strip tag -> candidate hyp bekern paths (TEMP workdirs).

    Two filename schemes exist across the eyeball runs:
    ``eyeball90-<tag>-bekern.txt`` and ``<tag>-hyp-bekern.txt``.
    Candidates sort largest-first (crash artifacts are ~2 bytes);
    :func:`rescore_tags` scores each candidate's base F1 and keeps
    the one reproducing the stored census F1.
    """
    import tempfile as _tf
    tmp = _tf.gettempdir()
    hyps = collections.defaultdict(list)
    for dp, _, fn in os.walk(tmp):
        if 'eyeball90' not in dp:
            continue
        for f in fn:
            tag = None
            if f.startswith('eyeball90-') and f.endswith('-bekern.txt'):
                tag = f[len('eyeball90-'):-len('-bekern.txt')]
                if tag.endswith('-hyp'):
                    tag = tag[:-4]
            elif f.endswith('-hyp-bekern.txt'):
                tag = f[:-len('-hyp-bekern.txt')]
            if tag is not None:
                p = os.path.join(dp, f)
                if p not in hyps[tag]:
                    hyps[tag].append(p)
    for tag in hyps:
        hyps[tag].sort(key=lambda p: -os.path.getsize(p))
    return dict(hyps)


def _sysdev_manifest():
    mani = json.load(open(REPO / 'data-gen/score-corpus/gate20/sys-dev'
                          / 'sys_dev_manifest.json'))
    out = {}
    for s in mani['systems']:
        out['%s-%d-s%d' % (s['id'], s['page'], s['sys_idx'])] = s
    return out


def resolve_ref(kind, stem, picks=None):
    """Ref tokens + optional corpus ref-notes path for a strip."""
    if kind == 'sysdev':
        import re as _re
        m = _re.match(r'(.+)-(\d+)-s(\d+)$', stem)
        s = _sysdev_manifest()['%s-%s-s%s' % (m.group(1), m.group(2),
                                             m.group(3))]
        base = REPO / 'data-gen/score-corpus/gate20/sys-dev'
        toks = open(base / s['bekern']).read().split()
        return toks, str(base / s['ref_notes'])
    if picks is None:
        picks = {p['stem']: p
                 for p in json.load(open('D:/tmp/eyeball90_picks.json'))[
                     'picks']}
    bek = picks[stem]['bekern']
    toks = open(REPO / 'tmp-train/mvd-sysdata-t32' / bek).read().split()
    return toks, None


def _run_cli(args, timeout=600):
    pr = subprocess.run(args, capture_output=True, text=True,
                        timeout=timeout)
    return pr


def _bekern_decode(bek_path, out_path):
    pr = _run_cli([str(ASER), 'bekern-decode', '--input', str(bek_path),
                   '--out', str(out_path)])
    return pr.returncode == 0 and Path(out_path).exists()


def _score_hyp(E, ref_notes_src, filt, workdir, tag, name):
    """bekern-decode + strict-f1 one hyp token list; return F1."""
    hp = workdir / ('%s-hyp-%s-bekern.txt' % (tag, name))
    hp.write_text(' '.join(filt) + '\n')
    hn = workdir / ('%s-hyp-%s-notes.json' % (tag, name))
    if not _bekern_decode(hp, hn):
        return 0.0
    sc = E.score_notes_pair(ref_notes_src, str(hn), workdir,
                            '%s-%s' % (tag, name), str(EHARNESS),
                            str(ASER))
    return float(sc.get('f1', 0.0))


def rescore_tags(tags, hyp_map, picks, outdir, workdir, stored=None,
                 configs=(('base', {}), ('guard', {'enable_voice_cap': True,
                                                   'enable_rest_guard': True
                                                   }),
                          ('spray', {'enable_spray_cap': True}),
                          ('combined', {'enable_voice_cap': True,
                                        'enable_rest_guard': True,
                                        'enable_spray_cap': True}))):
    """Score each tag under each guard config; return per-tag F1 table.

    ``configs``: (name, apply_guards kwargs). ``base`` uses no guards
    (sanity: must reproduce the stored F1). ``hyp_map`` maps tag to
    candidate hyp paths; the candidate whose base F1 reproduces the
    stored census F1 (``stored``) is kept and recorded as
    ``row['hyp_src']``. Ref notes come from the sysdev corpus or a
    cached t32 bekern-decode; filtered hyps are decoded + scored per
    config. Fail-soft per strip (error recorded).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval_sys_gate as E  # lazy: pulls torch/model
    outdir, workdir = Path(outdir), Path(workdir)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    stored = stored or {}
    table = {}
    for tag in tags:
        kind = tag.split('-')[0]
        rest = tag[len(kind) + 1:]
        dom = rest.rsplit('-', 1)[1]
        stem = rest[:-(len(dom) + 1)]
        row = {'tag': tag, 'error': None, 'f1': {}, 'n_tok': {},
               'hyp_src': None}
        try:
            ref_toks, ref_notes_src = resolve_ref(kind, stem, picks)
            if ref_notes_src is None:
                ref_notes_src = str(workdir / ('%s-ref-notes.json' % tag))
                if not Path(ref_notes_src).exists():
                    rp = workdir / ('%s-ref-bekern.txt' % tag)
                    rp.write_text(' '.join(ref_toks) + '\n')
                    ok = _bekern_decode(rp, ref_notes_src)
                    if not ok:
                        raise RuntimeError('ref decode failed')
            cands = hyp_map[tag]
            if isinstance(cands, str):
                cands = [cands]
            want = stored.get(tag)
            hyp_toks, src, base_f1 = None, None, None
            scored_cands = []
            for cp in cands:
                ht = open(cp).read().split()
                f1 = _score_hyp(E, ref_notes_src, ht, workdir, tag,
                                'cand%d' % len(scored_cands))
                scored_cands.append((cp, ht, f1))
                if want is not None and abs(f1 - want) < 1e-6:
                    hyp_toks, src, base_f1 = ht, cp, f1
                    break
            if hyp_toks is None:
                # no exact reproduction: keep the best-matching base
                if want is not None:
                    scored_cands.sort(
                        key=lambda r: abs(r[2] - want))
                src, hyp_toks, base_f1 = (scored_cands[0][0],
                                          scored_cands[0][1],
                                          scored_cands[0][2])
            row['hyp_src'] = src
            row['f1']['base'] = base_f1
            row['n_tok']['base'] = len(hyp_toks)
            row['base_repro'] = (want is None or
                                 abs(base_f1 - want) < 1e-6)
            for name, kw in configs:
                if not kw:
                    continue
                filt, grep = apply_guards(hyp_toks, **kw)
                row['guards'] = row.get('guards', {})
                row['guards'][name] = grep['details']
                row['f1'][name] = _score_hyp(E, ref_notes_src, filt,
                                            workdir, tag, name)
                row['n_tok'][name] = len(filt)
        except Exception as e:  # noqa: BLE001 - fail-soft per strip
            row['error'] = '%s: %s' % (type(e).__name__, e)
        table[tag] = row
        print('%s f1=%s repro=%s %s'
              % (tag, row['f1'], row.get('base_repro'),
                 row['error'] or ''), flush=True)
    json.dump(table, open(outdir / 'guard_rescore_table.json', 'w'),
              indent=1)
    return table


def main(argv=None):
    """CLI: rescore stored 87-strip hyps under guard configs (CPU)."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='D:/tmp/guard_rescore')
    ap.add_argument('--workdir', default='D:/tmp/guard_rescore_work')
    ap.add_argument('--tags-json', default='D:/tmp/eyeball90/f1_87.json')
    ap.add_argument('--only-kind', default=None)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args(argv)
    stored = json.load(open(a.tags_json))
    tags = [t for t, _ in stored]
    if a.only_kind:
        tags = [t for t in tags if t.split('-')[0] == a.only_kind]
    if a.limit:
        tags = tags[:a.limit]
    hyp_map = load_hyp_map()
    missing = [t for t in tags if t not in hyp_map]
    if missing:
        raise SystemExit('missing hyps: %s' % missing[:5])
    picks = {p['stem']: p for p in
             json.load(open('D:/tmp/eyeball90_picks.json'))['picks']}
    table = rescore_tags(tags, hyp_map, picks, a.outdir, a.workdir,
                         stored=dict(stored))
    base = [r['f1'].get('base') for r in table.values()
            if r['f1'].get('base') is not None]
    import statistics as _st
    print('scored=%d mean_base=%.4f median_base=%.4f'
          % (len(base), _st.mean(base) if base else 0.0,
             _st.median(base) if base else 0.0))
    return table


if __name__ == '__main__':
    main()