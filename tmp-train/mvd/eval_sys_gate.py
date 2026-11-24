#!/usr/bin/env python3
"""T1.6 system-gate eval (E7-sys-train): MvdNet v2 ckpt on gate20 sys-dev.

Decodes every sys-dev system (clean + degraded) with beam search
(beam=3, strict grammar mask, max_len>=564 to cover the sys-dev token
maximum), scores bekern-decode + SER + strict-F1 per system, stitches
hyp systems per page in reading order (first-system headers + all
bodies), and scores full-page strict-F1, proxy-fixes/page, .mscz
validity and secs/page. Page pairs additionally score MusicXML vs
MusicXML notation-F1 (spelling-exact, --no-notation to skip).
Writes ONE new JSON file (--out); every other gate20 file is only
read.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/eval_sys_gate.py \\
        --ckpt tmp-train/mvd-runs/sys-v2/ckpt-15000.pt \\
        --out data-gen/score-corpus/gate20/probe_report_sys_v2.json
"""

import argparse
import difflib
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data  # noqa: E402
import model  # noqa: E402
import train  # noqa: E402
import heat_features  # noqa: E402
import rectify as rect_mod  # noqa: E402
import strip_staff as sst  # noqa: E402
import build_hybrid_cache as bhc  # noqa: E402
import build_gate20_sys_dev as bgate  # noqa: E402
import measure_decode as mdec  # noqa: E402
import page_prep  # noqa: E402
import bezier_ribbon  # noqa: E402
import dynamic_res  # noqa: E402
import staff_unet2d  # noqa: E402
import verifier_3tier  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ASER = REPO_ROOT / 'converter' / 'assembly-serialize' / 'target' / 'release' / 'assembly-serialize.exe'
EHARNESS = REPO_ROOT / 'converter' / 'eval-harness' / 'target' / 'release' / 'eval-harness.exe'
GATE20 = REPO_ROOT / 'data-gen' / 'score-corpus' / 'gate20'
SYS_DEV = GATE20 / 'sys-dev'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def split_head_body(toks):
    """Leading clef-/key/time- run = staff headers; the rest = body."""
    i = 0
    while i < len(toks) and (toks[i].startswith('clef-')
                             or toks[i].startswith('key')
                             or toks[i].startswith('time-')):
        i += 1
    return toks[:i], toks[i:]


_STEPS = frozenset('ABCDEFG')
_OCTAVES = frozenset(str(d) for d in range(10))
_DURS = frozenset(('long', 'breve', 'whole', 'half', 'quarter', 'eighth',
                   '16th', '32nd', '64th', '128th'))
_NON_TAIL_PREFIXES = ('clef-', 'key', 'time-', 'voice-', 'bar', 'dyn-',
                      'hairpin-', 'tuplet-')
_NON_TAIL_EXACT = (_STEPS | _OCTAVES | _DURS
                   | frozenset(('rest', 'bar-rest', 'chord', 'grace')))


def _is_alter(tok):
    return (len(tok) >= 2 and tok[0] == 'a'
            and (tok[1:].isdigit()
                 or (tok[1] == '-' and len(tok) >= 3 and tok[2:].isdigit())))


def _is_tail_tok(tok):
    if tok in _NON_TAIL_EXACT:
        return False
    return not tok.startswith(_NON_TAIL_PREFIXES)


def canonicalize_bekern_tail_order(toks):
    """Canonicalize note/rest tail token order to match ``decode.rs``.

    ``assembly-serialize bekern-decode`` requires pitched-note tails in
    ``[alter] [dot*] [slurs/mods/ties]`` order and rest tails in
    ``[dot*] [slurs/mods]`` order. When beam search emits an accidental
    or dot after a slur/articulation on the same note (e.g.
    ``16th A 4 slur-stop stacc a-1 dot`` on ``ryans-jig`` / ``airds-air``),
    ``bekern-decode`` exits 1 (``unexpected token "a-1"``) and zeroes the
    entire page. Moving out-of-order alter/dot tokens to their canonical
    slot within the same event is semantics-preserving and a strict no-op
    on already-canonical streams.
    """
    toks = list(toks)
    out = []
    i, n = 0, len(toks)
    while i < n:
        t = toks[i]
        # Pitched note pitch pair: <step> <octave>
        if t in _STEPS and i + 1 < n and toks[i + 1] in _OCTAVES:
            out.append(t)
            out.append(toks[i + 1])
            i += 2
            alters, dots, mods = [], [], []
            while i < n and _is_tail_tok(toks[i]):
                tt = toks[i]
                if _is_alter(tt):
                    alters.append(tt)
                elif tt == 'dot':
                    dots.append(tt)
                else:
                    mods.append(tt)
                i += 1
            if alters:
                out.append(alters[0])
            out.extend(dots)
            out.extend(mods)
        # Plain rest: rest <dur>
        elif t == 'rest' and i + 1 < n and toks[i + 1] in _DURS:
            out.append(t)
            out.append(toks[i + 1])
            i += 2
            dots, mods = [], []
            while i < n and _is_tail_tok(toks[i]):
                tt = toks[i]
                if tt == 'dot':
                    dots.append(tt)
                elif not _is_alter(tt):
                    mods.append(tt)
                i += 1
            out.extend(dots)
            out.extend(mods)
        else:
            out.append(t)
            i += 1
    return out


def trim_max_len_repeat_loop(toks, max_len=564, min_period=8, min_repeats=4):
    """Trim an infinite verbatim repeating tail ONLY when a strip hit max_len.

    Does nothing for normal strips (``len(toks) < int(max_len) - 2``)
    or non-repeating max-length strips. When a strip hits the decode length
    cap with a verbatim ``>=min_period``-token block repeated at least
    ``min_repeats`` times at the tail (allowing a partial final copy cut by
    ``max_len``), keeps the prefix through the first copy of the loop and
    trims the runaway copies.
    """
    toks = list(toks)
    cap = int(max_len)
    if len(toks) < cap - 2:
        return toks
    n = len(toks)
    max_p = n // min_repeats
    for p in range(min_period, max_p + 1):
        match_len = 0
        idx = n - 1
        while idx >= p and toks[idx] == toks[idx - p]:
            match_len += 1
            idx -= 1
        if match_len >= (min_repeats - 1) * p:
            # First occurrence of the repeating period starts at loop_start
            # = n - match_len - p; keep one complete copy of the period.
            # If the backward match walked across a shared trailing 'bar'
            # from the preceding measure, step one token right so we end
            # cleanly on the barline.
            cut = n - match_len
            if cut < n and toks[cut] == 'bar':
                cut += 1
            return toks[:cut]
    return toks


def _header_slots(head_toks):
    """Split a leading header run into per-kind ordered token lists."""
    clefs, keys, times = [], [], []
    for t in head_toks:
        if t.startswith('clef-'):
            clefs.append(t)
        elif t.startswith('key'):
            keys.append(t)
        elif t.startswith('time-'):
            times.append(t)
    return clefs, keys, times


def _diff_header_slots(active, incoming):
    """Compare incoming header tokens of one kind against active slots.

    Updates ``active`` in place and returns deduplicated changed tokens
    (preserving order). Unchanged courtesy headers return ``[]``.
    """
    changed = []
    for idx, tok in enumerate(incoming):
        if idx < len(active):
            if active[idx] != tok:
                active[idx] = tok
                if tok not in changed:
                    changed.append(tok)
        else:
            prev = active[-1] if active else None
            active.append(tok)
            if prev is not None and tok != prev and tok not in changed:
                changed.append(tok)
    return changed


def _update_active_from_body(body_toks, active_clefs, active_keys, active_times):
    """Advance active header state for mid-system clef/key/time changes."""
    for t in body_toks:
        if t.startswith('key'):
            if active_keys:
                for i in range(len(active_keys)):
                    active_keys[i] = t
            else:
                active_keys.append(t)
        elif t.startswith('time-'):
            if active_times:
                for i in range(len(active_times)):
                    active_times[i] = t
            else:
                active_times.append(t)
        elif t.startswith('clef-') and len(active_clefs) <= 1:
            if active_clefs:
                active_clefs[0] = t
            else:
                active_clefs.append(t)


def stitch_page_toks_with_context(sys_toks_list, init_ctx=None):
    """Stitch per-system token lists with ``ScoreContext`` carryforward.

    Returns ``(canonical_tokens, updated_ctx)`` using
    :func:`measure_decode.propagate_across_systems` followed by
    :func:`canonicalize_bekern_tail_order`.
    """
    stitched_str, ctx = mdec.propagate_across_systems(
        sys_toks_list, init_ctx=init_ctx)
    toks = stitched_str.split() if stitched_str else []
    return canonicalize_bekern_tail_order(toks), ctx


def stitch_multipage_toks(page_sys_toks_lists, init_ctx=None):
    """Stitch a multi-page score (list of per-page system token lists).

    Returns ``(canonical_tokens, final_ctx)`` using
    :func:`measure_decode.stitch_multipage_work`.
    """
    stitched_str, ctx = mdec.stitch_multipage_work(
        page_sys_toks_lists, init_ctx=init_ctx)
    toks = stitched_str.split() if stitched_str else []
    return canonicalize_bekern_tail_order(toks), ctx


def stitch_page_toks(sys_toks_list, init_ctx=None):
    """Page token stream from per-system token lists.

    Delegates to :func:`measure_decode.propagate_across_systems` via
    :func:`stitch_page_toks_with_context`: leading headers (``clef-`` /
    ``key`` / ``time-``) are adopted from the first system with a
    non-empty body, unchanged courtesy headers on subsequent systems are
    stripped while genuine mid-page clef/key/meter changes are emitted,
    and cross-system ties/slurs/hairpins are carried cleanly. An empty
    first system (decode crash) does not lock in empty headers. All-empty
    input stitches to ``[]``.
    """
    toks, _ctx = stitch_page_toks_with_context(sys_toks_list, init_ctx=init_ctx)
    return toks


def run_cli(args, timeout=600):
    p = subprocess.run([str(a) for a in args], capture_output=True,
                       text=True, cwd=str(REPO_ROOT), timeout=timeout)
    return p


#: Bekern modifier tokens for the spike-S2 gate metric (mirrors
#: ``model.ILME_MODIFIERS`` as strings; EOS never appears in bekern
#: text so it is excluded here).
MODIFIER_TOKENS = frozenset(model.ILME_MODIFIERS)


def modifier_f1(ref_toks, hyp_toks, mod_set=MODIFIER_TOKENS):
    """Token-level F1 over modifier tokens only (spike S2 metric).

    Aligns the full ref/hyp bekern streams with difflib and counts a
    true positive for each aligned-equal token in ``mod_set``.
    Pitch/duration/header errors are invisible to this metric by
    design: it isolates the articulation/dot/tie hallucinations ILME
    targets. No modifiers on either side scores 1.0 (nothing to fix).
    Pure function.
    """
    ref_toks, hyp_toks = list(ref_toks), list(hyp_toks)
    ref_mod = sum(1 for t in ref_toks if t in mod_set)
    hyp_mod = sum(1 for t in hyp_toks if t in mod_set)
    if not ref_mod and not hyp_mod:
        return 1.0
    sm = difflib.SequenceMatcher(a=ref_toks, b=hyp_toks,
                                 autojunk=False)
    tp = sum(1 for _tag, i1, i2, _j1, _j2 in sm.get_opcodes()
             for t in ref_toks[i1:i2]
             if _tag == 'equal' and t in mod_set)
    prec = tp / hyp_mod if hyp_mod else 0.0
    rec = tp / ref_mod if ref_mod else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def score_notes_pair(ref_notes, hyp_notes, workdir, stem, strict_exe, aser_exe):
    """bekern-decode already done by caller; run ser + strict-f1 on note JSONs.

    Returns dict with ser/f1/matches/ref_len/hyp_len/proxy_fixes (+rc fields).
    Decode failure is handled by the caller (f1=0 rule); here a scorer
    failure also zeroes f1 and pins ser=1.0 so medians stay numeric.
    """
    out = {'ser': 1.0, 'f1': 0.0, 'matches': 0, 'ref_len': None,
           'hyp_len': None, 'proxy_fixes': None, 'ser_rc': None,
           'f1_rc': None, 'zeroed': False, 'zero_reason': None}
    ser_rep = workdir / (stem + '-ser.json')
    p = run_cli([aser_exe, 'ser', '--reference', str(ref_notes),
                 '--hypothesis', str(hyp_notes), '--report', str(ser_rep)])
    out['ser_rc'] = p.returncode
    if p.returncode == 0 and ser_rep.exists():
        try:
            r = json.loads(ser_rep.read_text(encoding='utf-8'))
            out['ser'] = float(r['ser'])
            out['ser_distance'] = r.get('distance')
            out['ser_ref_len'] = r.get('ref_len')
            out['ser_hyp_len'] = r.get('hyp_len')
        except (ValueError, KeyError):
            out['zeroed'] = True
            out['zero_reason'] = 'ser-report-unparseable'
    else:
        out['zeroed'] = True
        out['zero_reason'] = 'ser-rc-%d' % p.returncode
    f1_rep = workdir / (stem + '-f1.json')
    p = run_cli([strict_exe, 'strict-f1', '--reference', str(ref_notes),
                 '--hypothesis', str(hyp_notes), '--report', str(f1_rep)])
    out['f1_rc'] = p.returncode
    if p.returncode == 0 and f1_rep.exists():
        try:
            r = json.loads(f1_rep.read_text(encoding='utf-8'))
            out['f1'] = float(r['f1'])
            out['matches'] = int(r.get('matches', 0))
            out['ref_len'] = r.get('ref_len')
            out['hyp_len'] = r.get('hyp_len')
            if out['ref_len'] is not None and out['hyp_len'] is not None:
                out['proxy_fixes'] = ((out['ref_len'] - out['matches'])
                                      + (out['hyp_len'] - out['matches']))
        except (ValueError, KeyError):
            out['f1'] = 0.0
            out['zeroed'] = True
            out['zero_reason'] = 'f1-report-unparseable'
    else:
        out['f1'] = 0.0
        out['zeroed'] = True
        out['zero_reason'] = 'strict-f1-rc-%d' % p.returncode
    return out


def score_notation_xmls(ref_xml, hyp_xml, workdir, stem, strict_exe):
    """MusicXML-vs-MusicXML notation scoring for one strip/page pair.

    Runs ``eval-harness notation-f1`` (spelling-exact B# != C, staff /
    voice / rests / tuplets priced, attachments priced, headers matched).
    Returns dict with ``notation_*`` fields; conversion/scoring failures
    zero ``notation_f1`` with a reason and never raise. The strict-F1
    columns are untouched: this rides alongside as the replacement
    ruler until gate bars are re-based.
    """
    out = {'notation_f1': 0.0, 'notation_drift': 0.0,
           'notation_zeroed': True, 'notation_zero_reason': None}
    rep = workdir / (stem + '-notation.json')
    try:
        p = run_cli([strict_exe, 'notation-f1', '--reference',
                     str(ref_xml), '--hypothesis', str(hyp_xml),
                     '--report', str(rep)])
    except OSError as e:
        out['notation_zero_reason'] = 'notation-f1-exec: %s' % e
        return out
    if p.returncode != 0 or not rep.exists():
        out['notation_zero_reason'] = 'notation-f1-rc-%d' % p.returncode
        return out
    try:
        r = json.loads(rep.read_text(encoding='utf-8'))
    except ValueError:
        out['notation_zero_reason'] = 'notation-report-unparseable'
        return out
    for k, nk in (('notation_f1', 'notation_f1'),
                  ('drift_f1', 'notation_drift'),
                  ('spell_f1', 'notation_spell'),
                  ('staff_f1', 'notation_staff'),
                  ('voice_f1', 'notation_voice'),
                  ('rest_f1', 'notation_rest'),
                  ('tuplet_f1', 'notation_tuplet'),
                  ('attach_precision', 'notation_attach_p'),
                  ('attach_recall', 'notation_attach_r'),
                  ('measure_exact', 'notation_measure'),
                  ('header_match', 'notation_header')):
        out[nk] = r.get(k)
    out['notation_zeroed'] = False
    return out


def score_notation_bekerns(ref_bekern_txt, hyp_bekern_txt, workdir, stem,
                           aser_exe, strict_exe):
    """Notation scoring for a bekern pair via MusicXML conversion.

    Converts both streams with ``bekern2musicxml`` (same converter both
    sides, so converter numerics cancel) then scores MusicXML vs
    MusicXML. Fail-soft like :func:`score_notes_pair`.
    """
    out = {'notation_f1': 0.0, 'notation_drift': 0.0,
           'notation_zeroed': True, 'notation_zero_reason': None}
    conv = workdir / ('notation_%s' % stem)
    try:
        conv.mkdir(exist_ok=True)
    except OSError as e:  # noqa: BLE001 - record, never crash gate
        out['notation_zero_reason'] = 'notation-convdir: %s' % e
        return out
    for txt, sfx in ((ref_bekern_txt, 'ref'), (hyp_bekern_txt, 'hyp')):
        p = run_cli([aser_exe, 'bekern2musicxml', '--input', str(txt),
                     '--out', str(conv), '--stem', '%s-%s' % (stem, sfx)])
        if p.returncode != 0 or not (conv / ('%s-%s.musicxml'
                                             % (stem, sfx))).exists():
            out['notation_zero_reason'] = (
                'notation-convert-%s-rc-%d' % (sfx, p.returncode))
            return out
    return score_notation_xmls(conv / ('%s-ref.musicxml' % stem),
                               conv / ('%s-hyp.musicxml' % stem),
                               workdir, stem, strict_exe)


def decode_system(net, img_path, height, max_len, beam, device, workdir, stem):
    """Beam-decode one system strip image to a token list + timings."""
    import torch
    t0 = time.time()
    imgs = train.load_image(str(img_path), height).unsqueeze(0).to(device)
    with torch.no_grad():
        hyp = net.beam_search(imgs, beam=beam, strict=True,
                              max_len=max_len, constrain=True)
    secs = time.time() - t0
    hyp = hyp[0].tolist()
    hyp = [x for x in hyp if x != data.PAD_ID]
    eos_miss = data.EOS_ID not in hyp
    if not eos_miss:
        hyp = hyp[:hyp.index(data.EOS_ID)]
    try:
        toks = data.decode(hyp)
        valid = True
    except ValueError:
        toks = []
        valid = False
    # NOTE: caller passes stem='<sys>-<domain>-hyp', so on-disk name is
    # '<sys>-<domain>-hyp-bekern.txt' (single '-hyp-'), matching both
    # readers below. A '-hyp-bekern.txt' suffix here would double it to
    # '-hyp-hyp-bekern.txt' and zero every score + crash page stitch.
    (workdir / (stem + '-bekern.txt')).write_text(' '.join(toks) + '\n',
                                                  encoding='utf-8')
    return {'tokens': toks, 'ntokens': len(toks), 'eos_miss': eos_miss,
            'vocab_valid': valid, 'decode_secs': secs, 'loop_trimmed': False}


def detect_ckpt_extra_chans(state):
    """Infer ``extra_chans`` from checkpoint stem ``encoder.0.0.weight``.

    Returns ``in_chans - 3`` (0 for 3ch RGB, 1 for 4ch RGB+staff, 8 for
    11ch RGB+HeatNet+staff), or ``None`` if the stem weight is absent.
    """
    sd = state.get('model', state) if isinstance(state, dict) else {}
    w = sd.get('encoder.0.0.weight')
    if w is None or getattr(w, 'ndim', 0) < 2:
        return None
    return max(0, int(w.shape[1]) - 3)


def build_arg_parser():
    ap = argparse.ArgumentParser(description='T1.6 sys-dev gate eval')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sys-dev', default=str(SYS_DEV))
    ap.add_argument('--max-len', type=int, default=576)
    ap.add_argument('--beam', type=int, default=3)
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--device', default=None)
    ap.add_argument('--hybrid', action='store_true',
                    help='T2.4: 11ch rectified+HeatNet input, extra_chans=8')
    ap.add_argument('--extra-chans', type=int, default=None,
                    help='extra encoder input channels beyond RGB: 0 (3ch RGB), '
                         '1 (4ch RGB+staff, no HeatNet), 8 (11ch RGB+HeatNet+staff); '
                         'default: auto-detect from checkpoint stem')
    ap.add_argument('--no-heatnet', action='store_true',
                    help='disable HeatNet (4ch RGB+staff with --rectify/--hybrid '
                         'or 3ch RGB; never loads HeatNet weights)')
    ap.add_argument('--blind-photo', action='store_true',
                    help='strict blind photo eval on suite31: no GT s_ref, no '
                         'scaled-clean box fallback (misses score 0.0), and no '
                         'clean-crop pairing in strip staff detection')
    ap.add_argument('--stride', default='8x16', choices=('8x16', '4x8'),
                    help='S13: encoder grid stride of the checkpoint. '
                         'Defaults to the legacy 8x16 path.')
    ap.add_argument('--hybrid-cache', default=None,
                    help='T2.4: hybrid_staff.jsonl staff tables for '
                    'rectification (sys-dev strips)')
    ap.add_argument('--heat-pt', default=heat_features.HEAT_PT_DEFAULT,
                    help='T2.4: HeatNet .pt checkpoint for --hybrid')
    ap.add_argument('--rectify', action='store_true',
                    help='T2.4: rectify strips via cache tables (implied '
                    'by --hybrid) with RowNet/blind fallback')
    ap.add_argument('--staff-unet-ckpt', default=None,
                    help='Exp 1: StaffUNet2D .pt checkpoint for direct '
                         'curved-system extraction + Bezier ribbon unwarping '
                         'on full pages (replaces RowNet + page_prep + scaled '
                         'fallback under --blind-photo)')
    ap.add_argument('--dynamic-staff-unet', action='store_true',
                    help='Force dynamic aspect-preserving StaffUNet2D '
                         'inference (even on fixed-res v2 checkpoints)')
    ap.add_argument('--suite31', action='store_true',
                    help='T2.4: also evaluate suite31 clean+photo pages')
    ap.add_argument('--no-notation', action='store_true',
                    help='skip MusicXML-vs-MusicXML notation scoring '
                    '(eval-harness notation-f1, spelling-exact) on page '
                    'pairs; on by default as the replacement ruler, '
                    'strict-F1 columns untouched')
    ap.add_argument('--ink-veto', action='store_true',
                    help='Spike S1: forbid EOS while HeatNet notehead '
                    'peaks outnumber decoded steps (11ch hybrid input '
                    'only; RGB fails open)')
    ap.add_argument('--ilme-lambda', type=float, default=0.0,
                    help='Spike S2: dual-pass ILM debias weight '
                    '(subtract lam * log P_ILM on modifiers + EOS); '
                    '0.0 keeps the legacy single pass')
    ap.add_argument('--lattice-repair', action='store_true',
                    help='Spike S6: decode 8 beams per strip and ship '
                    'the constraint-ranked winner (bar math, tie '
                    'closure, logprob) instead of the top beam')
    ap.add_argument('--adapt-steps', type=int, default=0,
                    help='Spike S9: test-time per-strip adapter entropy '
                    'steps (0 = off, legacy decode); >0 fits a fresh '
                    'rank-8 map per strip on its own greedy '
                    'pseudo-targets, zero retraining')
    ap.add_argument('--adapt-lr', type=float, default=0.1,
                    help='Spike S9: SGD learning rate for the per-strip '
                    'adapter fit')
    ap.add_argument('--verify-beam', action='store_true',
                    help='T4.1: render-and-diff visual reranking over top-K '
                    'beams via render_diff_verify.rerank_beams_visual '
                    '(CPU-only, requires beam >= 2)')
    ap.add_argument('--verify-alpha', type=float, default=2.0,
                    help='T4.1: visual IoU weight for --verify-beam '
                    '(combined_score = logprob + alpha * visual_iou)')
    ap.add_argument('--verifier-3tier', action='store_true',
                    help='Enable 3-Tier Closed-Loop Verifier (Tier 1 bar-math/'
                    'spanner gate, conditional Tier 2 visual rerank, selective '
                    'Tier 3 geometric/TTA repair & review flagging)')
    ap.add_argument('--verifier-pref-out', default=None,
                    help='Optional JSONL path to emit preference pairs whenever '
                    '--verifier-3tier overrules greedy beam-0')
    ap.add_argument('--page-tta', default='none',
                    choices=('none', 'em', 'come', 'come_sar'),
                    help='Optional episodic LayerNorm TTA mode for Tier 3 '
                    'repair / page adaptation (default: none)')
    ap.add_argument('--suite31-dir', default=str(SUITE31_DEFAULT))
    ap.add_argument('--suite31-manifest', default=None,
                    help='optional path to an alternate Suite31 manifest JSON '
                         '(e.g. suite31_hw_manifest.json); defaults to '
                         '<suite31-dir>/suite31_manifest.json')
    ap.add_argument('--suite31-only', action='store_true',
                    help='evaluate only the Suite31 manifest (skip gate20 '
                         'sys-dev loop; test_pages_touched stays 0)')
    ap.add_argument('--use-manifest-boxes', action='store_true',
                    help='if blind staff grouping fails on a piece, fall back '
                         'to manifest-supplied system_boxes (off by default)')
    ap.add_argument('--limit-systems', type=int, default=None,
                    help='smoke-test: cap sys-dev systems')
    ap.add_argument('--limit-pieces', type=int, default=None,
                    help='smoke-test: cap suite31 pieces')
    ap.add_argument('--pieces', default=None,
                    help='smoke-test: comma-separated suite31 piece slugs '
                         'to evaluate (preserves manifest order)')
    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.suite31_only:
        args.suite31 = True
    if args.hybrid:
        args.rectify = True
        if not args.hybrid_cache and not args.suite31_only:
            ap.error('--hybrid requires --hybrid-cache')
    assert args.max_len >= 564, 'max_len must cover sys-dev token_max 564'

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    sys_dev = Path(args.sys_dev)
    manifest = json.loads((sys_dev / 'sys_dev_manifest.json').read_text(
        encoding='utf-8'))
    assert manifest.get('test_pages_touched', 0) == 0, 'test pages touched!'
    systems = sorted(manifest['systems'],
                     key=lambda s: (s['id'], s['sys_idx']))

    state = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    ckpt_extra = detect_ckpt_extra_chans(state)
    if args.extra_chans is not None:
        extra = int(args.extra_chans)
    elif args.no_heatnet:
        if args.hybrid or args.rectify:
            extra = 1
        elif ckpt_extra in (0, 1):
            extra = ckpt_extra
        else:
            extra = 0
    elif ckpt_extra is not None:
        extra = ckpt_extra
    else:
        extra = heat_features.EXTRA_CHANS if args.hybrid else 0
    if extra not in (0, 1, heat_features.EXTRA_CHANS):
        raise SystemExit(
            'unsupported extra_chans=%d (expected 0, 1, or %d)'
            % (extra, heat_features.EXTRA_CHANS))
    if args.no_heatnet and extra == heat_features.EXTRA_CHANS:
        ap.error('--no-heatnet is incompatible with extra_chans=%d' % extra)
    if extra > 0:
        args.rectify = True
    use_heatnet = (extra == heat_features.EXTRA_CHANS) and not args.no_heatnet

    has_foveal = any(k.startswith('foveal_proj.') for k in state)
    has_comer = any(k.startswith('comer_bias.') for k in state)
    has_gdcm = any(k.startswith('gdcm_head.') for k in state)
    has_s18 = any(k.startswith('s18_head.') for k in state)
    has_rope2d = any(k.startswith('rope2d.') for k in state)
    net = model.MvdNet(vocab=len(data.VOCAB) + 3, anisotropic=True,
                       stride=args.stride, extra_chans=extra,
                       foveal_tap=has_foveal, use_comer=has_comer,
                       use_gdcm=has_gdcm, use_s18=has_s18,
                       rope_2d=has_rope2d)
    try:
        # Compat loader (train.load_weights_compat): strict when shapes
        # match, aux-fresh for pre-S13b ckpts, stem/vocab transplants
        # otherwise. NOTE: a 4x8-stride ckpt in an 8x16 net still fails
        # loud here -- eval-side --stride wiring lands with the S13
        # gate eval, not this slice.
        train.load_weights_compat(net, state)
    except RuntimeError as e:
        raise SystemExit(
            'ckpt %s incompatible with extra_chans=%d: %s'
            % (args.ckpt, extra, str(e)[:200]))
    net.to(device).eval()
    heat_net = None
    staff_unet = None
    staff_unet_meta = None
    geom_index = {}
    if args.rectify and not (args.suite31_only and args.staff_unet_ckpt):
        bgate.ensure_weights_readable()
    if args.staff_unet_ckpt:
        staff_unet, staff_unet_meta = load_staff_unet(
            args.staff_unet_ckpt, device=device)
        print('StaffUNet2D: %s on %s (%dx%d, stride=%d)'
              % (args.staff_unet_ckpt, device,
                 staff_unet_meta['target_h'], staff_unet_meta['target_w'],
                 staff_unet_meta['out_stride']), flush=True)
    if args.rectify and args.hybrid_cache:
        geom_index = load_geometry_index(args.hybrid_cache)
        print('geometry index: %d validated strips in %s'
              % (len(geom_index), args.hybrid_cache), flush=True)
    if use_heatnet:
        heat_net = heat_features.HeatNet(args.heat_pt, device=device)
        print('HeatNet: %s on %s' % (args.heat_pt, heat_net.device),
              flush=True)
    print('ckpt: %s | device: %s | torch: %s | systems: %d | beam=%d max_len=%d'
          % (args.ckpt, device, torch.__version__, len(systems),
             args.beam, args.max_len), flush=True)

    workdir = Path(tempfile.mkdtemp(prefix='t16_gate_'))
    gate_sha = sha256_file(GATE20 / 'gate_manifest.json')
    verifier = None
    if args.verifier_3tier:
        tta_mode = args.page_tta if args.page_tta != 'none' else 'come'
        verifier = verifier_3tier.ClosedLoopVerifier(
            tta_mode=tta_mode,
            pref_out_path=args.verifier_pref_out,
        )

    sys_rows, page_groups = [], {}
    if args.suite31_only:
        systems = []
    elif args.limit_systems:
        systems = systems[:args.limit_systems]
    for s in systems:
        for domain, key in (('clean', 'clean'), ('degraded', 'degraded')):
            stem = '%s-%d-s%d' % (s['id'], s['page'], s['sys_idx'])
            img = sys_dev / s[key]
            try:
                if args.rectify:
                    row = geom_index.get(Path(s[key]).name)
                    dec = decode_system_v2(
                        net, str(img), row, heat_net, device, args.height,
                        args.beam, args.max_len, args.hybrid, workdir,
                        '%s-%s-hyp' % (stem, domain),
                        ink_eos_veto=args.ink_veto,
                        ilme_lambda=args.ilme_lambda,
                        lattice_repair=args.lattice_repair,
                        adapt_steps=args.adapt_steps,
                        adapt_lr=args.adapt_lr, stride=args.stride,
                        extra_chans=extra,
                        verify_beam=args.verify_beam,
                        verify_alpha=args.verify_alpha,
                        verifier=verifier,
                        page_tta=args.page_tta)
                else:
                    dec = decode_system(net, img, args.height, args.max_len,
                                        args.beam, device, workdir,
                                        '%s-%s-hyp' % (stem, domain))
            except Exception as e:  # noqa: BLE001 - record, never crash gate
                sys_rows.append({'system': stem, 'domain': domain,
                                 'error': 'decode-crash: %s' % e,
                                 'ser': 1.0, 'f1': 0.0, 'f1_zeroed': True,
                                 'f1_zero_reason': 'decode-crash',
                                 'ntokens': 0})
                print('%s [%s] DECODE-CRASH %s' % (stem, domain, e),
                      flush=True)
                continue
            hyp_txt = workdir / ('%s-%s-hyp-bekern.txt' % (stem, domain))
            hyp_notes = workdir / ('%s-%s-hyp-notes.json' % (stem, domain))
            ref_notes = sys_dev / s['ref_notes']
            row = {'system': stem, 'domain': domain,
                   'geom': dec.get('geom', 'legacy') if args.rectify else 'legacy',
                   'ref_tokens': s['tokens'],
                   'ntokens': dec['ntokens'], 'eos_miss': dec['eos_miss'],
                   'loop_trimmed': bool(dec.get('loop_trimmed', False)),
                   'decode_secs': dec['decode_secs'],
                   'slice_fallback_page': s['id'] == 'sq8438915-5'}
            p = run_cli([ASER, 'bekern-decode', '--input', str(hyp_txt),
                         '--out', str(hyp_notes)])
            row['decode_rc'] = p.returncode
            if p.returncode != 0 or not hyp_notes.exists():
                row.update({'ser': 1.0, 'f1': 0.0, 'matches': 0,
                            'f1_zeroed': True,
                            'f1_zero_reason': 'bekern-decode-rc-%d'
                             % p.returncode,
                            'decode_stderr': (p.stderr or '')[-300:]})
            else:
                sc = score_notes_pair(ref_notes, hyp_notes, workdir,
                                      '%s-%s' % (stem, domain), EHARNESS,
                                      ASER)
                row.update(sc)
                row['f1_zeroed'] = bool(sc['zeroed']) and sc['f1'] == 0.0
                row['f1_zero_reason'] = sc['zero_reason']
            sys_rows.append(row)
            print('%s [%s] ref_tok=%d hyp_tok=%d ser=%.4f f1=%.4f %s' % (
                stem, domain, s['tokens'], dec['ntokens'],
                row.get('ser'), row.get('f1'),
                'ZEROED(%s)' % row.get('f1_zero_reason')
                if row.get('f1_zero_reason') else ''), flush=True)
        page_groups.setdefault(s['id'], []).append(s)

    # ---- stitch per page in reading order ----
    page_rows = []
    for page_id, ss in sorted(page_groups.items()):
        ss = sorted(ss, key=lambda s: s['sys_idx'])
        t0 = time.time()
        pg = {'page': page_id, 'n_systems': len(ss), 'domains': {}}
        ref_sys_toks = [(sys_dev / s['bekern']).read_text(
            encoding='utf-8').split() for s in ss]
        ref_page = stitch_page_toks(ref_sys_toks)
        ref_bar_end = [bool(b) and b[-1] == 'bar'
                       for _h, b in map(split_head_body, ref_sys_toks)]
        for domain in ('clean', 'degraded'):
            d = {'systems_used': len(ss)}
            hyps = [next(r for r in sys_rows
                         if r['system'] == '%s-%d-s%d'
                         % (s['id'], s['page'], s['sys_idx'])
                         and r['domain'] == domain) for s in ss]
            d['system_decode_secs'] = sum(
                h.get('decode_secs', 0.0) for h in hyps)
            crashed = [h['system'] for h in hyps if 'error' in h]
            hyps_toks = []
            for h in hyps:
                if 'error' not in h:
                    try:
                        ht = (workdir / ('%s-%s-hyp-bekern.txt'
                                         % (h['system'], domain))).read_text(
                                             encoding='utf-8').split()
                    except OSError:
                        # Missing/unreadable hyp -> empty body, page scores
                        # ZEROED downstream; never crash before the report.
                        ht = []
                        d.setdefault('hyp_missing', []).append(h['system'])
                else:
                    ht = []
                hyps_toks.append(ht)
            hyp_page = stitch_page_toks(hyps_toks)
            hyp_bar_end = [bool(b) and b[-1] == 'bar'
                           for _h, b in map(split_head_body, hyps_toks)]
            # Leading header run of the stitched page = adopted heads,
            # unless every body is empty (then the page — and its header
            # run — is empty too).
            d['hyp_header_len'] = len(split_head_body(hyp_page)[0])
            d['hyp_body_bar_end'] = hyp_bar_end
            hyp_txt = workdir / ('%s-%s-page-hyp-bekern.txt'
                                 % (page_id, domain))
            hyp_txt.write_text(' '.join(hyp_page) + '\n', encoding='utf-8')
            ref_txt = workdir / ('%s-%s-page-ref-bekern.txt'
                                 % (page_id, domain))
            ref_txt.write_text(' '.join(ref_page) + '\n', encoding='utf-8')
            hyp_notes = workdir / ('%s-%s-page-hyp-notes.json'
                                   % (page_id, domain))
            ref_notes = workdir / ('%s-%s-page-ref-notes.json'
                                   % (page_id, domain))
            ok = True
            for txt, outp in ((ref_txt, ref_notes), (hyp_txt, hyp_notes)):
                p = run_cli([ASER, 'bekern-decode', '--input', str(txt),
                             '--out', str(outp)])
                if p.returncode != 0 or not outp.exists():
                    ok = False
                    d['decode_failure'] = (
                        'ref' if outp is ref_notes else 'hyp')
                    break
            if not ok or crashed:
                d.update({'ser': 1.0, 'f1': 0.0, 'f1_zeroed': True,
                          'f1_zero_reason': 'page-decode-failure',
                          'crashed_systems': crashed})
                if not args.no_notation:
                    d.update({'notation_f1': 0.0, 'notation_drift': 0.0,
                              'notation_zeroed': True,
                              'notation_zero_reason': 'page-decode-failure'})
            else:
                sc = score_notes_pair(ref_notes, hyp_notes, workdir,
                                      '%s-%s-page' % (page_id, domain),
                                      EHARNESS, ASER)
                d.update(sc)
                if not args.no_notation:
                    nd = score_notation_bekerns(
                        ref_txt, hyp_txt, workdir,
                        '%s-%s-page-notation' % (page_id, domain),
                        ASER, EHARNESS)
                    d.update(nd)
            # pack hyp page -> .mscz validity
            packdir = workdir / ('pack-%s-%s' % (page_id, domain))
            packdir.mkdir(exist_ok=True)
            p = run_cli([ASER, 'bekern2musicxml', '--input', str(hyp_txt),
                         '--out', str(packdir), '--stem', page_id])
            d['pack_rc'] = p.returncode
            mscz = packdir / (page_id + '.mscz')
            d['mscz_packed'] = mscz.exists() and mscz.stat().st_size > 0
            vjson = packdir / 'validation.json'
            d['validation_errors'] = None
            if vjson.exists():
                try:
                    v = json.loads(vjson.read_text(encoding='utf-8'))

                    def count_errors(x):
                        n = 0
                        if isinstance(x, dict):
                            if x.get('severity') == 'error':
                                n += 1
                            n += sum(count_errors(v) for v in x.values())
                        elif isinstance(x, list):
                            n += sum(count_errors(v) for v in x)
                        return n

                    d['validation_errors'] = count_errors(v)
                except ValueError:
                    d['validation_errors'] = 'unparseable'
            d['mscz_valid'] = bool(d['mscz_packed']
                                   and d['validation_errors'] == 0)
            pg['domains'][domain] = d
            print('%s [%s] page_tok=%d ser=%.4f f1=%.4f fixes=%s mscz=%s' % (
                page_id, domain, len(hyp_page), d.get('ser'), d.get('f1'),
                d.get('proxy_fixes'), d.get('mscz_valid')), flush=True)
        pg['secs_per_page'] = time.time() - t0
        pg['ref_body_bar_end'] = ref_bar_end
        page_rows.append(pg)

    def med(xs):
        xs = [x for x in xs if x is not None]
        return statistics.median(xs) if xs else None

    deg = [r for r in sys_rows if r['domain'] == 'degraded' and 'error'
           not in r]
    cln = [r for r in sys_rows if r['domain'] == 'clean' and 'error'
           not in r]
    summary = {
        'n_systems': len(systems),
        'n_sys_rows': len(sys_rows),
        'degraded_sys_ser_median': med([r.get('ser') for r in deg]),
        'degraded_page_notation_f1_median': med(
            [p['domains']['degraded'].get('notation_f1')
             for p in page_rows]),
        'clean_page_notation_f1_median': med(
            [p['domains']['clean'].get('notation_f1') for p in page_rows]),
        'degraded_sys_f1_median': med([r.get('f1') for r in deg]),
        'clean_sys_ser_median': med([r.get('ser') for r in cln]),
        'clean_sys_f1_median': med([r.get('f1') for r in cln]),
        'degraded_page_f1_median': med(
            [p['domains']['degraded'].get('f1') for p in page_rows]),
        'degraded_page_ser_median': med(
            [p['domains']['degraded'].get('ser') for p in page_rows]),
        'degraded_page_proxy_fixes_median': med(
            [p['domains']['degraded'].get('proxy_fixes') for p in page_rows]),
        'degraded_mscz_valid_rate': (
            sum(1 for p in page_rows
                if p['domains']['degraded'].get('mscz_valid')) / len(page_rows)
            if page_rows else None),
        'clean_mscz_valid_rate': (
            sum(1 for p in page_rows
                if p['domains']['clean'].get('mscz_valid')) / len(page_rows)
            if page_rows else None),
        'secs_per_page_median': med([p['secs_per_page'] for p in page_rows]),
        'sysdev_loop_trimmed': sum(
            1 for r in sys_rows if r.get('loop_trimmed')),
    }
    summary['gate2_pass'] = (
        summary['degraded_sys_ser_median'] is not None
        and summary['degraded_sys_ser_median'] < 0.48
        and summary['degraded_sys_f1_median'] is not None
        and summary['degraded_sys_f1_median'] > 0.22)
    summary['gate3_sys'] = {
        'ser_lt_015': (summary['degraded_sys_ser_median'] or 1.0) < 0.15,
        'f1_gt_050': (summary['degraded_sys_f1_median'] or 0.0) > 0.50,
        'page_proxy_fixes_le_50': (
            summary['degraded_page_proxy_fixes_median'] is not None
            and summary['degraded_page_proxy_fixes_median'] <= 50),
        'mscz_100': summary['degraded_mscz_valid_rate'] == 1.0,
    }

    if args.staff_unet_ckpt:
        preproc = ('StaffUNet2D curved-system extraction + '
                   'bezier_ribbon canonical-256 unwarping')
    elif args.rectify:
        preproc = ('rectify_system_strip canonical-256 (cache tables, '
                   'RowNet fallback, legacy on skip)')
    else:
        preproc = 'train.load_image (LANCZOS, ImageNet)'
    if extra == heat_features.EXTRA_CHANS:
        preproc += ' + HeatNet-7ch + staff-grid (11ch input)'
        preproc += '; measure-fallback beam (T2.3)'
    elif extra == 1:
        preproc += ' + staff-grid (4ch input, no HeatNet)'
        preproc += '; measure-fallback beam (T2.3)'
    elif args.rectify or args.staff_unet_ckpt:
        preproc += '; measure-fallback beam (T2.3)'
    suite31_rows, suite31_summary = [], {}
    if args.suite31:
        suite31_rows, suite31_summary = run_suite31(
            args, net, heat_net, workdir, device, extra_chans=extra,
            staff_unet=staff_unet, staff_unet_meta=staff_unet_meta,
            verifier=verifier)
        summary.update(suite31_summary)
    if verifier is not None:
        summary['verifier_3tier'] = verifier.telemetry_dict()
        summary['low_confidence_measures'] = verifier.low_confidence_measures_total
    report = {
        'gate': 'gate20-sys-v2',
        'gate_manifest_sha256': gate_sha,
        'suite31_manifest_sha256': suite31_summary.get('suite31_manifest_sha256'),
        'test_pages_touched': 0,
        'blind_photo': bool(args.blind_photo),
        'ckpt': args.ckpt,
        'ckpt_bytes': os.path.getsize(args.ckpt),
        'hybrid': {'enabled': args.hybrid,
                   'extra_chans': extra,
                   'no_heatnet': not use_heatnet,
                   'blind_photo': bool(args.blind_photo),
                   'hybrid_cache': args.hybrid_cache,
                   'heat_pt': args.heat_pt if use_heatnet else None,
                   'staff_unet_ckpt': args.staff_unet_ckpt,
                   'dynamic_staff_unet': bool(args.dynamic_staff_unet),
                   'rectify': args.rectify,
                   'suite31': args.suite31,
                   'suite31_dir': args.suite31_dir if args.suite31 else None,
                   'suite31_manifest': (
                       suite31_summary.get('suite31_manifest')
                       if args.suite31 else None)},
        'decoding': {'search': 'beam_search', 'beam': args.beam,
                     'strict': True, 'constrain': True,
                     'ilme_lambda': args.ilme_lambda,
                     'lattice_repair': args.lattice_repair,
                     'adapt_steps': args.adapt_steps,
                     'adapt_lr': args.adapt_lr,
                     'verify_beam': bool(args.verify_beam),
                     'verify_alpha': float(args.verify_alpha),
                     'verifier_3tier': bool(args.verifier_3tier),
                     'verifier_pref_out': args.verifier_pref_out,
                     'page_tta': args.page_tta,
                     'max_len': args.max_len, 'image_height': args.height,
                     'preprocessing': preproc,
                     'device': device, 'torch': torch.__version__},
        'stitch': 'reading order by sys_idx; first-system headers adopted + '
                  'subsequent unchanged courtesy headers stripped while '
                  'changed mid-page clef/key/time headers are propagated; '
                  'page ref stitched identically',
        'summary': summary,
        'systems': sys_rows,
        'pages': page_rows,
        'suite31': suite31_rows,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + '.tmp')
    tmp.write_text(json.dumps(report, indent=1), encoding='utf-8')
    os.replace(tmp, outp)
    print(json.dumps(summary, indent=1))
    print('wrote ' + str(outp))
    # Gate verdict lives in the JSON (summary.gate2_pass / gate3_sys);
    # always exit 0 once the report is written so a missed gate can
    # never strand the caller without a report.
    print('gate2_pass=%s' % summary['gate2_pass'])


# ---- T2.4 hybrid / rectify / suite31 (additive; legacy path above untouched) ----

SUITE31_DEFAULT = REPO_ROOT / 'data-gen' / 'score-corpus' / 'suite31'


def load_geometry_index(jsonl_path):
    """``hybrid_staff.jsonl`` -> ``{img basename: row}`` (validated rows only)."""
    index = {}
    with open(jsonl_path, encoding='utf-8') as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if not r.get('skip') and isinstance(r.get('table'), dict):
                index[r['img']] = r
    return index


def _valid_fallback_staves(lines5):
    """True iff live-detected staff lines are geometrically sane.

    Fallback policy (parity evidence 2026-11-21, D:/tmp/parity_diag.json):
    the blind detector with a wrong staff count returns non-monotonic
    garbage (e.g. S=1 rows [286.2, 675.3, 241.7, 253.3, 254.5] on a
    true-S=3 degraded strip). Rectifying against that scores a silent
    0.0, so invalid curves must route to legacy / tagged no-geometry
    instead of a fallback repair decode.
    """
    try:
        staves = list(lines5)
    except TypeError:
        return False
    if not staves:
        return False
    for staff in staves:
        try:
            lines = list(staff)
        except TypeError:
            return False
        if len(lines) != 5:
            return False
        rows = []
        for ln in lines:
            try:
                rows.append(float(np.median(np.asarray(ln, dtype=np.float64))))
            except (TypeError, ValueError):
                return False
        if any(r != r or r in (float('inf'), float('-inf')) for r in rows):
            return False
        if any(b <= a for a, b in zip(rows, rows[1:])):
            return False
    return True


def rect_staves_for(img_path, row):
    """Staff geometry for one strip: cached table curves or RowNet fallback.

    Returns ``(staves_lines, geom_src)``; ``(None, 'legacy')`` when no
    validated table exists (width mismatch or skip) — the caller then
    uses the legacy unrectified path and flags it.
    """
    img_w = Image.open(img_path).size[0]
    if row is not None:
        table = row.get('table') or {}
        if int(table.get('w_in', -1)) == img_w:
            try:
                return sst.table_to_curves(table), 'table'
            except (ValueError, KeyError):
                pass
    try:
        _S, lines5, _sp, _src = bhc.clean_lines(str(img_path))
        if not _valid_fallback_staves(lines5):
            return None, 'legacy'
        return lines5, 'rownet-fallback'
    except ValueError:
        return None, 'legacy'


def rectified_input_tensor(img_path, staves, heat_net, device, height,
                           hybrid, extra_chans=None):
    """Rectified ``[1, C, H, W]`` model input (C=11 hybrid, C=4 staff, C=3 RGB).

    No sidecar reads, no gate writes: rectify in memory via ``rectify.py``,
    optional HeatNet 7-ch from the persistent session, and optional staff
    grid from the canonical reference channel.
    """
    if extra_chans is None:
        extra_chans = heat_features.EXTRA_CHANS if hybrid else 0
    img = Image.open(img_path).convert('RGB')
    rect_img, ref = rect_mod.rectify_system_strip(
        img, staves, target_sp=10.0, height=height)
    return rectified_input_tensor_from_rect(
        rect_img, ref, heat_net, device, extra_chans=extra_chans)


def hybrid_extra_fn(heat_net, device, extra_chans=None):
    """``measure_decode`` extra_fn for 3ch, 4ch (RGB+staff), or 11ch hybrid."""
    def _fn(rect_img, ref):
        return rectified_input_tensor_from_rect(
            rect_img, ref, heat_net, device, extra_chans=extra_chans)
    return _fn


def rectified_input_tensor_from_rect(rect_img, ref, heat_net, device,
                                     extra_chans=None):
    """Same as :func:`rectified_input_tensor` from an in-memory rectify."""
    import torch
    if extra_chans is None:
        extra_chans = (heat_features.EXTRA_CHANS
                       if heat_net is not None else 0)
    arr = np.asarray(rect_img, dtype=np.float32) / 255.0
    rgb = train.standardize_rgb(
        torch.from_numpy(arr).permute(2, 0, 1)).unsqueeze(0).to(device)
    if extra_chans == 0:
        return rgb
    staff = (ref > 0.5).float().unsqueeze(0).to(device)
    if staff.ndim == 3:
        staff = staff.unsqueeze(1)
    if extra_chans == 1:
        return torch.cat([rgb, staff], dim=1)
    gray = torch.from_numpy(
        (np.asarray(rect_img.convert('L'), dtype=np.float32)
         / 255.0)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        heat7 = heat_net.probs(gray.to(device))
    return torch.cat([rgb, heat7.float(), staff], dim=1)


def write_hyp_bekern(workdir, stem, toks):
    """Write ``<stem>-bekern.txt`` (same naming as :func:`decode_system`)."""
    (workdir / (stem + '-bekern.txt')).write_text(' '.join(toks) + '\n',
                                                 encoding='utf-8')


def hyp_tokens_valid(toks):
    """True iff a hypothesis stream is vocabulary-legal.

    T2.3 ``measure_decode`` entry points return bekern STRING token
    lists (via ``_ids_to_tokens``), while the legacy
    :func:`decode_system` path carries INT id lists. Strings must be
    checked by vocabulary membership (``data.decode`` takes INT ids
    only and raises ``ValueError`` on strings, which wrongly zeroed
    every T2.4 hypothesis); ints keep the ``data.decode`` range check.
    """
    if toks and isinstance(toks[0], str):
        return all(t in data.TOKEN_TO_ID for t in toks)
    try:
        data.decode(list(toks))
        return True
    except ValueError:
        return False


def _decode_strip_with_verifier_3tier(net, ten, beam, max_len, mem_mask,
                                      adapt_steps, adapt_lr, verifier,
                                      strip_id, page_tta='none'):
    """Run top-K beam decode + 3-Tier Closed-Loop Verifier on one strip tensor."""
    import torch
    bs_kwargs = {}
    if mem_mask is not None:
        bs_kwargs['mem_mask'] = mem_mask
    if adapt_steps:
        bs_kwargs['adapt_steps'] = adapt_steps
        bs_kwargs['adapt_lr'] = adapt_lr
    beam_k = max(2, int(beam))

    def _run_beams(t_in):
        with torch.no_grad():
            all_beams, all_scores = net.beam_search(
                t_in, beam=beam_k, strict=True, max_len=int(max_len),
                return_beams=True, **bs_kwargs)
        cands = []
        for b_ids, lp in zip(all_beams[0], all_scores[0]):
            toks_i = mdec._ids_to_tokens([b_ids])
            fixed_i, _, _ = mdec.fallback_repair(toks_i)
            cands.append((fixed_i, float(lp)))
        return cands

    candidates = _run_beams(ten)
    if not candidates:
        return [], None
    tta_model = net if (page_tta and page_tta != 'none') else None
    v_out = verifier.verify_strip(
        strip_id, ten, candidates, decode_fn=_run_beams, tta_model=tta_model)
    return list(v_out.tokens), v_out


def decode_system_v2(net, img_path, row, heat_net, device, height, beam,
                     max_len, hybrid, workdir, stem, ink_eos_veto=False,
                     ilme_lambda=0.0, lattice_repair=False,
                     adapt_steps=0, adapt_lr=0.1, stride='8x16',
                     extra_chans=None, verify_beam=False,
                     verify_alpha=2.0, verifier=None, page_tta='none'):
    """T2.4 decode: rectified (+hybrid/4ch) input, measure-fallback beam search.

    Geometry from the cache row (fallback: RowNet detect, then legacy
    ``train.load_image``). ``S > 4`` staves route through
    ``measure_decode.decode_multi_staff_system`` chunking. Returns the
    same dict shape as :func:`decode_system` plus a ``geom`` source tag
    and ``loop_trimmed`` bool.
    Train/inference mask parity: the ``<= 4`` path builds the slot
    memory mask from the known staff count ``S`` (train-matched
    inactive-slot masking); the chunked path builds one per chunk.
    """
    import torch
    if extra_chans is None:
        extra_chans = heat_features.EXTRA_CHANS if hybrid else 0
    t0 = time.time()
    staves, geom = rect_staves_for(img_path, row)
    if staves is None:
        if extra_chans > 0:
            # No validated geometry and no detection: a 4ch/11ch net cannot
            # consume a 3ch strip (and zero-filled extras would lie to
            # the gate), so record an explicit no-geometry zero.
            write_hyp_bekern(workdir, stem, [])
            return {'tokens': [], 'ntokens': 0, 'eos_miss': True,
                    'vocab_valid': True, 'decode_secs': 0.0,
                    'loop_trimmed': False, 'geom': 'no-geometry'}
        dec = decode_system(net, img_path, height, max_len, beam, device,
                            workdir, stem)
        dec['geom'] = 'legacy'
        return dec
    img = Image.open(img_path).convert('RGB')
    v_out = None
    with torch.no_grad():
        if len(staves) > rect_mod.MAX_STAVES:
            extra = hybrid_extra_fn(heat_net, device,
                                    extra_chans=extra_chans)
            toks = mdec.decode_multi_staff_system(
                net, img, staves, beam=beam, extra_fn=extra,
                ink_eos_veto=ink_eos_veto, ilme_lambda=ilme_lambda,
                lattice_repair=lattice_repair, adapt_steps=adapt_steps,
                adapt_lr=adapt_lr, stride=stride,
                verify_beam=verify_beam, verify_alpha=verify_alpha,
                max_len=max_len)
        else:
            if verify_beam:
                rect_img, ref = rect_mod.rectify_system_strip(
                    img, staves, target_sp=10.0, height=height)
                ten = rectified_input_tensor_from_rect(
                    rect_img, ref, heat_net, device, extra_chans=extra_chans)
                obs_gray = rect_img
            else:
                ten = rectified_input_tensor(img_path, staves, heat_net,
                                             device, height, hybrid,
                                             extra_chans=extra_chans)
                obs_gray = None
            mem_mask = mdec.mask_for_strip(ten, len(staves), stride=stride)
            if verifier is not None:
                toks, v_out = _decode_strip_with_verifier_3tier(
                    net, ten, beam=beam, max_len=max_len, mem_mask=mem_mask,
                    adapt_steps=adapt_steps, adapt_lr=adapt_lr,
                    verifier=verifier, strip_id=stem, page_tta=page_tta)
            else:
                toks = mdec.decode_system_with_measure_fallback(
                    net, ten, beam=beam,
                    mem_mask=mem_mask,
                    ink_eos_veto=ink_eos_veto, ilme_lambda=ilme_lambda,
                    lattice_repair=lattice_repair, adapt_steps=adapt_steps,
                    adapt_lr=adapt_lr, verify_beam=verify_beam,
                    verify_alpha=verify_alpha, obs_gray=obs_gray,
                    max_len=max_len)
    secs = time.time() - t0
    valid = hyp_tokens_valid(toks)
    loop_trimmed = False
    if valid:
        trimmed = trim_max_len_repeat_loop(toks, max_len=max_len)
        loop_trimmed = len(trimmed) < len(toks)
        toks = canonicalize_bekern_tail_order(trimmed)
    write_hyp_bekern(workdir, stem, toks if valid else [])
    res_dict = {'tokens': toks if valid else [], 'ntokens': len(toks) if valid else 0,
                'eos_miss': False, 'vocab_valid': valid, 'decode_secs': secs,
                'loop_trimmed': loop_trimmed,
                'geom': geom if len(staves) <= rect_mod.MAX_STAVES
                else geom + '+chunked'}
    if v_out is not None:
        res_dict['verifier_needs_review'] = v_out.needs_review
        res_dict['low_confidence_measures'] = v_out.low_confidence_measures
    return res_dict


def suite31_page_refs(gt_path, sys_stem, workdir):
    """Page-level refs into ``workdir`` (never gate20).

    Files named ``gt-p1-*`` are already cut to page 1 (and may contain
    stray ``<print new-page="yes"/>`` elements such as
    ``gt-p1-luca-gloria.musicxml``), so they convert the full file via
    ``--measures 1..999999`` instead of ``--page 1`` to avoid premature
    truncation. Multi-page raw scores without the ``gt-p1-`` prefix keep
    ``--page 1``. One ``aser import`` gives the reference notes.
    """
    stem = sys_stem + '-page-ref'
    txt = workdir / (stem + '-bekern.txt')
    is_page1_cut = Path(gt_path).name.startswith('gt-p1-')
    scope_args = (['--measures', '1..999999']
                  if is_page1_cut else ['--page', '1'])
    p = run_cli([ASER, 'bekern', '--input', str(gt_path), *scope_args,
                 '--out', str(workdir), '--stem', stem])
    if p.returncode != 0 or not txt.exists():
        raise RuntimeError('suite31 bekern %s failed: %s'
                           % (' '.join(scope_args), (p.stderr or '')[-200:]))
    toks = txt.read_text(encoding='utf-8').split()
    notes = workdir / (stem + '-notes.json')
    p = run_cli([ASER, 'bekern-decode', '--input', str(txt),
                 '--out', str(notes)])
    if p.returncode != 0 or not notes.exists():
        raise RuntimeError('suite31 page bekern-decode failed')
    page_notes = workdir / (sys_stem + '-page-ref-notes.json')
    p = run_cli([ASER, 'import', '--input', str(gt_path),
                 '--out', str(workdir)])
    cand = list(workdir.glob(Path(gt_path).stem + '.json'))
    if p.returncode != 0 or not cand:
        raise RuntimeError('suite31 import failed: %s'
                           % ((p.stderr or '')[-200:]))
    if cand[0] != page_notes:
        if page_notes.exists():
            page_notes.unlink()
        cand[0].rename(page_notes)
    return toks, page_notes


def visual_system_groups(staves, space, factor=3.5, s_ref=None):
    """Gap-rule visual grouping (suite31 pages without engraved breaks).

    Staves sorted by ``y1`` split wherever the inter-staff gap exceeds
    ``factor * space`` (within-system staff gaps are ~1 space; system
    gaps are several). When ``s_ref`` (the reference header staff count,
    i.e. the number of ``clef-*`` tokens) evenly divides the detected
    staff count, staves chunk into consecutive groups of ``s_ref`` --
    the gap rule over-splits quartet/piano systems (e.g. 10 groups where
    the header says 4 staves per system). Returns ``None`` on empty
    input; the result carries no measure-span contract (page-level refs
    only).
    """
    if not staves:
        return None
    ordered = sorted(staves, key=lambda s: s['y1'])
    if s_ref and s_ref > 0 and len(ordered) % s_ref == 0:
        return [ordered[i:i + s_ref]
                for i in range(0, len(ordered), s_ref)]
    groups, cur = [], [ordered[0]]
    for prev, nxt in zip(ordered, ordered[1:]):
        if nxt['y1'] - prev['y2'] > factor * space:
            groups.append(cur)
            cur = [nxt]
        else:
            cur.append(nxt)
    groups.append(cur)
    return groups


def pad_staff_boxes(groups, space, w, h):
    """Tight padded boxes (same fractions as the gate20 builder)."""
    import build_sys_data as bsd
    boxes = []
    for g in groups:
        x1 = max(0.0, min(s['x1'] for s in g) - bsd.PAD_H_SPACES * space)
        y1 = max(0.0, min(s['y1'] for s in g) - bsd.PAD_V_SPACES * space)
        x2 = min(float(w), max(s['x2'] for s in g) + bsd.PAD_H_SPACES * space)
        y2 = min(float(h), max(s['y2'] for s in g) + bsd.PAD_V_SPACES * space)
        if not (x2 > x1 and y2 > y1):
            return None
        boxes.append([int(round(x1)), int(round(y1)),
                      int(round(x2)), int(round(y2))])
    return boxes


def suite31_page_boxes(clean_path, photo_path, gt_path, s_ref=None,
                       blind_photo=False):
    """Visual system boxes for a suite31 piece (clean + photo).

    Tries the proven span-based grouping first (pieces whose GT carries
    print breaks); falls back to gap-rule visual grouping. Photo boxes
    prefer own-page detection (count-matched) else scaled clean boxes.
    When ``blind_photo=True``, photo detection receives ``s_ref=None``,
    does NOT require matching ``len(clean_boxes)``, and NEVER falls back
    to scaled clean boxes (returns ``(clean_boxes, None, 'blind-miss')``
    on failure).
    Returns ``(clean_boxes, photo_boxes, box_source)``; ``None`` pieces
    are skipped by the caller.
    """
    with Image.open(clean_path) as im:
        cw, ch = im.size
    clean_boxes = None
    try:
        spans = bgate.run_system_breaks(str(gt_path))
        got = bgate.tight_group_boxes(str(clean_path), str(gt_path), spans)
        if got is not None:
            clean_boxes, _kept, _dropped, _size = got
    except (RuntimeError, ValueError, OSError):
        pass
    if clean_boxes is None:
        try:
            staves, space = bgate.detect_staves_with_short_tail(str(clean_path))
        except (OSError, ValueError):
            return None
        groups = visual_system_groups(staves, space, s_ref=s_ref)
        if groups is None:
            return None
        clean_boxes = pad_staff_boxes(groups, space, cw, ch)
        if clean_boxes is None:
            return None
    with Image.open(photo_path) as im:
        dw, dh = im.size
    photo_s_ref = None if blind_photo else s_ref
    try:
        dstaves, dspace = bgate.detect_staves_with_short_tail(str(photo_path))
        dgroups = visual_system_groups(dstaves, dspace, s_ref=photo_s_ref)
        if dgroups is not None and (blind_photo
                                    or len(dgroups) == len(clean_boxes)):
            deg_boxes = pad_staff_boxes(dgroups, dspace, dw, dh)
            if deg_boxes is not None:
                return clean_boxes, deg_boxes, 'detected'
    except (OSError, ValueError):
        pass
    if blind_photo:
        return clean_boxes, None, 'blind-miss'
    sx, sy = dw / cw, dh / ch
    deg_boxes = []
    for b in clean_boxes:
        x1 = max(0, min(int(round(b[0] * sx)), dw - 1))
        y1 = max(0, min(int(round(b[1] * sy)), dh - 1))
        x2 = max(x1 + 1, min(int(round(b[2] * sx)), dw))
        y2 = max(y1 + 1, min(int(round(b[3] * sy)), dh))
        deg_boxes.append([x1, y1, x2, y2])
    return clean_boxes, deg_boxes, 'scaled'


def load_staff_unet(ckpt_path, device='cpu'):
    """Load a ``StaffUNet2D`` checkpoint saved by ``staff_unet2d.train_staff_unet2d``.

    Returns ``(model, meta)`` where ``meta`` holds ``in_channels``,
    ``base_channels``, ``out_stride``, ``target_h``, ``target_w``,
    ``sp_mode``, ``dynamic_res``, ``pixel_budget``, and ``max_side``.
    """
    import torch
    state = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
    if isinstance(state, dict) and ('state_dict' in state or 'model' in state):
        sd = state.get('state_dict', state.get('model'))
        in_ch = int(state.get('in_channels', 3))
        base_ch = int(state.get('base_channels', 40))
        out_stride = int(state.get('out_stride', 2))
        target_h = int(state.get('target_h', 1024))
        target_w = int(state.get('target_w', 768))
        sp_mode = str(state.get('sp_mode', 'rel_h'))
        dyn_res = bool(state.get('dynamic_res', False))
        pixel_budget = int(state.get('pixel_budget', 1536 * 1152))
        max_side = int(state.get('max_side', 2048))
    else:
        sd = state
        in_ch, base_ch, out_stride, target_h, target_w = 3, 40, 2, 1024, 768
        sp_mode, dyn_res = 'rel_h', False
        pixel_budget, max_side = 1536 * 1152, 2048
    has_row_ctx = (
        bool(state.get('row_ctx', False)) if isinstance(state, dict) else False
    ) or any(k.startswith('row_ctx.') for k in sd)
    has_strip_conv = (
        bool(state.get('strip_conv', False)) if isinstance(state, dict) else False
    ) or any(k.startswith('strip_dec2.') or k.startswith('strip_dec1.') for k in sd)
    net = staff_unet2d.StaffUNet2D(
        in_channels=in_ch, base_channels=base_ch, out_stride=out_stride,
        row_ctx=has_row_ctx, strip_conv=has_strip_conv)
    net.load_state_dict(sd)
    net.to(device).eval()
    meta = {
        'in_channels': in_ch,
        'base_channels': base_ch,
        'out_stride': out_stride,
        'target_h': target_h,
        'target_w': target_w,
        'sp_mode': sp_mode,
        'dynamic_res': dyn_res,
        'pixel_budget': pixel_budget,
        'max_side': max_side,
        'row_ctx': has_row_ctx,
        'strip_conv': has_strip_conv,
    }
    return net, meta


def extract_page_curved_systems(staff_unet, page_img, device='cpu', meta=None,
                                force_dynamic=False):
    """Run ``StaffUNet2D`` -> ``extract_curved_systems`` on a full-page image."""
    import torch
    if not isinstance(page_img, Image.Image):
        page_img = Image.open(page_img).convert('RGB')
    orig_w, orig_h = page_img.size
    m = meta or {}
    sp_mode = str(m.get('sp_mode', 'rel_h'))
    out_stride = int(m.get('out_stride', getattr(staff_unet, 'out_stride', 2)))
    in_ch = int(m.get('in_channels', getattr(staff_unet, 'in_channels', 3)))
    if bool(m.get('dynamic_res', False)) or bool(force_dynamic):
        pixel_budget = int(m.get('pixel_budget', 1536 * 1152))
        max_side = int(m.get('max_side', 2048))
        x_ten, dyn_meta = dynamic_res.preprocess_page_dynamic(
            page_img, pixel_budget=pixel_budget, max_side=max_side,
            stride=32, in_channels=in_ch)
        x = x_ten.unsqueeze(0).to(device)
        with torch.no_grad():
            probs = staff_unet.predict_probs(x, sp_mode=sp_mode)
        probs = dynamic_res.unpad_probs_dynamic(
            probs, dyn_meta, out_stride=out_stride)
        return staff_unet2d.extract_curved_systems(
            probs, orig_w=orig_w, orig_h=orig_h,
            sp_mode=sp_mode, iso_scale=dyn_meta['scale'],
            out_stride=out_stride)
    target_h = int(m.get('target_h', 1024))
    target_w = int(m.get('target_w', 768))
    x = staff_unet2d.preprocess_page_image(
        page_img, target_h=target_h, target_w=target_w,
        in_channels=in_ch).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = staff_unet.predict_probs(x, sp_mode=sp_mode)
    return staff_unet2d.extract_curved_systems(
        probs, orig_w=orig_w, orig_h=orig_h,
        sp_mode=sp_mode, out_stride=out_stride)


def decode_curved_system(net, page_img, curved_sys, heat_net, device, height,
                         beam, max_len, workdir, stem, ink_eos_veto=False,
                         ilme_lambda=0.0, lattice_repair=False,
                         adapt_steps=0, adapt_lr=0.1, stride='8x16',
                         extra_chans=0, verify_beam=False,
                         verify_alpha=2.0, verifier=None, page_tta='none'):
    """Unwarp a ``CurvedSystem`` via ``bezier_ribbon`` and measure-fallback decode.

    Directly extracts canonical ``height``-px (default 256 px) strips +
    ``ref_channel`` from the full page without rectangular cropping or
    RowNet. Systems with ``> 4`` staves are automatically chunked via
    ``bezier_ribbon.sample_curved_system_chunks`` (``chunk_staves_for_ribbon``)
    and merged at the measure/voice level identically to
    ``measure_decode.decode_multi_staff_system``.
    """
    import torch
    t0 = time.time()
    if not isinstance(page_img, Image.Image):
        page_img = Image.open(page_img).convert('RGB')
    chunks = bezier_ribbon.sample_curved_system_chunks(
        page_img, curved_sys,
        max_staves=bezier_ribbon.MAX_STAVES,
        target_sp=bezier_ribbon.REF_TARGET_SP,
        height=height,
    )
    chunk_streams = []
    chunk_staff_counts = []
    prev_staves_cum = 0
    needs_review_any = False
    low_conf_all = []
    with torch.no_grad():
        for c_idx, (rect_img, ref, staff_indices) in enumerate(chunks):
            if prev_staves_cum >= mdec.MAX_VOICE:
                break
            ten = rectified_input_tensor_from_rect(
                rect_img, ref, heat_net, device, extra_chans=extra_chans)
            mem_mask = mdec.mask_for_strip(
                ten, len(staff_indices), stride=stride)
            if verifier is not None:
                ctoks, v_out = _decode_strip_with_verifier_3tier(
                    net, ten, beam=beam, max_len=max_len, mem_mask=mem_mask,
                    adapt_steps=adapt_steps, adapt_lr=adapt_lr,
                    verifier=verifier, strip_id='%s-c%d' % (stem, c_idx),
                    page_tta=page_tta)
                if v_out is not None and v_out.needs_review:
                    needs_review_any = True
                    low_conf_all.extend(v_out.low_confidence_measures)
            else:
                ctoks = mdec.decode_system_with_measure_fallback(
                    net, ten, beam=beam, mem_mask=mem_mask,
                    ink_eos_veto=ink_eos_veto, ilme_lambda=ilme_lambda,
                    lattice_repair=lattice_repair, adapt_steps=adapt_steps,
                    adapt_lr=adapt_lr, verify_beam=verify_beam,
                    verify_alpha=verify_alpha,
                    obs_gray=(rect_img if verify_beam else None),
                    max_len=max_len)
            chunk_streams.append(ctoks)
            chunk_staff_counts.append(len(staff_indices))
            prev_staves_cum += len(staff_indices)
    if len(chunk_streams) == 1:
        toks = chunk_streams[0]
        geom = 'staff-unet2d'
    else:
        toks = mdec.merge_chunk_streams(
            chunk_streams, chunk_staff_counts=chunk_staff_counts,
            max_voice=mdec.MAX_VOICE)
        geom = 'staff-unet2d+chunked'
    secs = time.time() - t0
    valid = hyp_tokens_valid(toks)
    loop_trimmed = False
    if valid:
        trimmed = trim_max_len_repeat_loop(toks, max_len=max_len)
        loop_trimmed = len(trimmed) < len(toks)
        toks = canonicalize_bekern_tail_order(trimmed)
    write_hyp_bekern(workdir, stem, toks if valid else [])
    n_staves = (len(curved_sys.staves) if hasattr(curved_sys, 'staves')
                else len(curved_sys))
    res = {
        'tokens': toks if valid else [],
        'ntokens': len(toks) if valid else 0,
        'eos_miss': False,
        'vocab_valid': valid,
        'decode_secs': secs,
        'loop_trimmed': loop_trimmed,
        'geom': geom,
        'n_staves': n_staves,
        'n_chunks': len(chunks),
    }
    if verifier is not None:
        res['verifier_needs_review'] = needs_review_any
        res['low_confidence_measures'] = sorted(set(low_conf_all))
    return res


def decode_row_system(net, img_file, row, heat_net, device, height, beam,
                      max_len, hybrid, workdir, stem, ink_eos_veto=False,
                      ilme_lambda=0.0, lattice_repair=False,
                      adapt_steps=0, adapt_lr=0.1, stride='8x16',
                      extra_chans=None, verify_beam=False,
                      verify_alpha=2.0, verifier=None, page_tta='none'):
    """Decode one eval strip file with its geometry row (T2.4 shared tail)."""
    return decode_system_v2(net, img_file, row, heat_net, device, height,
                            beam, max_len, hybrid, workdir, stem,
                            ink_eos_veto=ink_eos_veto,
                            ilme_lambda=ilme_lambda,
                            lattice_repair=lattice_repair,
                            adapt_steps=adapt_steps,
                            adapt_lr=adapt_lr, stride=stride,
                            extra_chans=extra_chans,
                            verify_beam=verify_beam,
                            verify_alpha=verify_alpha,
                            verifier=verifier,
                            page_tta=page_tta)


def classify_suite31_stratum(piece):
    """Classify a Suite31 / Suite31+HW manifest entry into its protocol stratum.

    Returns one of:
    - ``'typeset'``: printed engravings (the 31 frozen Suite31 pieces)
    - ``'hw_landscape'``: IMSLP PD + OpenScore CC0 landscape quartet holographs (5)
    - ``'hw_portrait_debussy'``: CC-BY-4.0 Debussy portrait manuscripts (4)
    - ``'hw_portrait_pdm'``: PDM 1.0 SBB Berlin + CC0 OpenScore portrait Lieder (5)
    - ``'hw_portrait_other'``: any future portrait HW piece outside the above
    """
    if not piece.get('handwritten'):
        return 'typeset'
    slug = str(piece.get('slug', ''))
    source = str(piece.get('source', '') or '')
    lic = str(piece.get('license', '') or '')
    if source == 'openscore-lieder-pdm' or slug.endswith('-d-b'):
        return 'hw_portrait_pdm'
    if 'debussy' in slug or 'debussy-omr' in lic:
        return 'hw_portrait_debussy'
    if piece.get('orientation') == 'portrait':
        return 'hw_portrait_other'
    return 'hw_landscape'


def _summarize_suite31_slice(rows):
    """Compute per-stratum Suite31 metrics (clean & photo strictly separated)."""
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def _med(xs):
        xs = [x for x in xs if x is not None]
        return statistics.median(xs) if xs else None

    ok = [r for r in rows if not r.get('skipped')]
    return {
        'n_pieces': len(rows),
        'n_scored': len(ok),
        'blind_photo_detected_rate': (
            sum(1 for r in ok
                if r.get('deg_box_source') in ('detected', 'staff-unet2d'))
            / len(ok)
            if ok else None),
        'clean_f1_mean': _mean([r.get('clean_f1') for r in ok]),
        'photo_f1_mean': _mean([r.get('photo_f1') for r in ok]),
        'clean_notation_f1_mean': _mean([r.get('clean_notation_f1') for r in ok]),
        'photo_notation_f1_mean': _mean([r.get('photo_notation_f1') for r in ok]),
        'clean_ser_mean': _mean([r.get('clean_ser') for r in ok]),
        'photo_ser_mean': _mean([r.get('photo_ser') for r in ok]),
        'clean_fixes_median': _med([r.get('clean_fixes') for r in ok]),
        'photo_fixes_median': _med([r.get('photo_fixes') for r in ok]),
    }


def run_suite31(args, net, heat_net, workdir, device, extra_chans=None,
                staff_unet=None, staff_unet_meta=None, verifier=None):
    """Suite31 photo+clean page eval (T2.4 / Exp 1): boxes or StaffUNet2D, decode, stitch.

    Pages land in ``workdir`` crops only; ``suite31/`` is read-only.
    Multi-staff systems (5-7 staves) decode via T2.3 / Bezier ribbon chunked
    merging. When ``staff_unet`` is provided, replaces RowNet + page_prep +
    scaled fallback with direct ``StaffUNet2D`` -> ``extract_curved_systems``
    -> ``bezier_ribbon.sample_curved_system_chunks`` extraction.
    Returns ``(piece_rows, summary_dict)``.
    """
    blind_photo = bool(getattr(args, 'blind_photo', False))
    suite = Path(args.suite31_dir)
    manifest_path = (Path(args.suite31_manifest)
                     if getattr(args, 'suite31_manifest', None)
                     else (suite / 'suite31_manifest.json'))
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    pieces = manifest['pieces']
    if getattr(args, 'pieces', None):
        wanted = {s.strip() for s in args.pieces.split(',') if s.strip()}
        pieces = [p for p in pieces if p['slug'] in wanted]
    pieces = pieces[:args.limit_pieces or len(pieces)]
    piece_rows = []
    clean_loop_trimmed = 0
    photo_loop_trimmed = 0
    for piece in pieces:
        slug = piece['slug']
        prow = {
            'piece': slug,
            'handwritten': bool(piece.get('handwritten', False)),
            'stratum': classify_suite31_stratum(piece),
        }
        try:
            gt = suite / piece['gt']['file']
            clean = suite / piece['clean']['file']
            photo = suite / piece['photo']['file']
            ref_page_toks, page_ref_notes = suite31_page_refs(
                gt, 's31-' + slug, workdir)
            if staff_unet is not None:
                # Exp 1: StaffUNet2D + Bezier ribbon directly on raw full pages
                # (bypasses page_prep, RowNet, and scaled-clean fallback).
                prow['photo_prep_meta'] = {
                    'skew_deg': 0.0,
                    'border_crop_box': None,
                    'applied': False,
                    'bypassed_by': 'staff_unet2d',
                }
                clean_pil = Image.open(clean).convert('RGB')
                photo_pil = Image.open(photo).convert('RGB')
                extract_kw = {'device': device, 'meta': staff_unet_meta}
                if getattr(args, 'dynamic_staff_unet', False):
                    extract_kw['force_dynamic'] = True
                clean_sys = extract_page_curved_systems(
                    staff_unet, clean_pil, **extract_kw)
                photo_sys = extract_page_curved_systems(
                    staff_unet, photo_pil, **extract_kw)
                deg_src = 'staff-unet2d' if photo_sys else 'staff-unet-miss'
                domain_specs = (
                    ('clean', clean_pil, clean_sys if clean_sys else None),
                    ('photo', photo_pil, photo_sys if photo_sys else None),
                )
                n_clean_systems = len(clean_sys)
                prow['photo_n_systems'] = len(photo_sys)
            else:
                prepped_photo_img, prep_meta = page_prep.prep_page(photo)
                photo_eval = workdir / ('s31-%s-photo-prepped.png' % slug)
                prepped_photo_img.save(photo_eval)
                prow['photo_prep_meta'] = {
                    'skew_deg': prep_meta['skew_deg'],
                    'border_crop_box': prep_meta['border_crop_box'],
                    'applied': prep_meta['applied'],
                }
                ref_head, _ref_body = split_head_body(ref_page_toks)
                s_ref = sum(1 for _t in ref_head
                            if _t.startswith('clef-')) or None
                got = suite31_page_boxes(str(clean), str(photo_eval), str(gt),
                                         s_ref=s_ref, blind_photo=blind_photo)
                if (got is None and getattr(args, 'use_manifest_boxes', False)
                        and piece.get('system_boxes')):
                    cboxes = piece['system_boxes']
                    if blind_photo:
                        got = (cboxes, None, 'blind-miss')
                    else:
                        with Image.open(clean) as im:
                            cw, ch = im.size
                        with Image.open(photo_eval) as im:
                            dw, dh = im.size
                        sx, sy = dw / cw, dh / ch
                        dboxes = [
                            [max(0, min(int(round(b[0] * sx)), dw - 1)),
                             max(0, min(int(round(b[1] * sy)), dh - 1)),
                             max(1, min(int(round(b[2] * sx)), dw)),
                             max(1, min(int(round(b[3] * sy)), dh))]
                            for b in cboxes
                        ]
                        got = (cboxes, dboxes, 'manifest-boxes')
                if got is None:
                    prow.update({'skipped': True,
                                 'reason': 'system-grouping-ambiguous'})
                    piece_rows.append(prow)
                    print('%s SKIP system-grouping-ambiguous' % slug, flush=True)
                    continue
                clean_boxes, deg_boxes, deg_src = got
                domain_specs = (
                    ('clean', clean, clean_boxes),
                    ('photo', photo_eval, deg_boxes),
                )
                n_clean_systems = len(clean_boxes)
            for domain, page_img, items in domain_specs:
                dskey = domain
                if items is None:
                    miss_reason = ('staff-unet-miss'
                                   if staff_unet is not None else deg_src)
                    prow[dskey + '_f1'] = 0.0
                    prow[dskey + '_ser'] = 1.0
                    prow[dskey + '_zero_reason'] = miss_reason
                    prow[dskey + '_mod_f1'] = modifier_f1(ref_page_toks, [])
                    prow[dskey + '_loop_trimmed'] = 0
                    if not args.no_notation:
                        prow[dskey + '_notation_f1'] = 0.0
                        prow[dskey + '_notation_drift'] = 0.0
                        prow[dskey + '_notation_zeroed'] = True
                        prow[dskey + '_notation_zero_reason'] = miss_reason
                    print('%s [%s] f1=0.0000 ser=1.0000 (%s)' % (
                        slug, domain, miss_reason), flush=True)
                    continue
                hyps_toks = []
                domain_trimmed = 0
                for i, item in enumerate(items):
                    stem = 's31-%s-s%d-%s-hyp' % (slug, i, domain)
                    if staff_unet is not None:
                        try:
                            dec = decode_curved_system(
                                net, page_img, item, heat_net, device,
                                args.height, args.beam, args.max_len,
                                workdir, stem,
                                ink_eos_veto=args.ink_veto,
                                ilme_lambda=args.ilme_lambda,
                                lattice_repair=args.lattice_repair,
                                adapt_steps=args.adapt_steps,
                                adapt_lr=args.adapt_lr, stride=args.stride,
                                extra_chans=(extra_chans if extra_chans is not None
                                             else (heat_features.EXTRA_CHANS
                                                   if args.hybrid else 0)),
                                verify_beam=bool(getattr(args, 'verify_beam', False)),
                                verify_alpha=float(getattr(args, 'verify_alpha', 2.0)),
                                verifier=verifier,
                                page_tta=str(getattr(args, 'page_tta', 'none')))
                        except Exception as e:  # noqa: BLE001 - record, continue
                            prow['%s_s%d_error' % (domain, i)] = (
                                'decode-crash: %s' % e)
                            hyps_toks.append([])
                            continue
                    else:
                        box = item
                        crop = workdir / ('s31-%s-s%d-%s.png' % (slug, i, domain))
                        bgate.crop_box(str(page_img), str(crop), box)
                        clean_name = 's31-%s-s%d-clean.png' % (slug, i)
                        if (domain == 'photo' and not blind_photo
                                and (workdir / clean_name).exists()):
                            # Photo strips pair against the clean crop
                            # (written by the clean pass above) so
                            # mapped-refine transfers clean truth onto the
                            # photo geometry; the degraded row is selected.
                            s = {'id': slug, 'page': 1, 'sys_idx': i,
                                 'clean': clean_name, 'degraded': crop.name}
                            use_degraded = True
                        else:
                            s = {'id': slug, 'page': 1, 'sys_idx': i,
                                 'clean': crop.name, 'degraded': crop.name}
                            use_degraded = False
                        try:
                            c_row, d_row = bhc._sysdev_one((str(workdir), s))
                            want = d_row if use_degraded else c_row
                            row = want if not want.get('skip') else None
                            if row is not None and not row.get('table'):
                                row = None
                        except ValueError:
                            row = None
                        try:
                            dec = decode_row_system(
                                net, str(crop), row, heat_net, device,
                                args.height, args.beam, args.max_len,
                                args.hybrid, workdir, stem,
                                ink_eos_veto=args.ink_veto,
                                ilme_lambda=args.ilme_lambda,
                                lattice_repair=args.lattice_repair,
                                adapt_steps=args.adapt_steps,
                                adapt_lr=args.adapt_lr, stride=args.stride,
                                extra_chans=extra_chans,
                                verify_beam=bool(getattr(args, 'verify_beam', False)),
                                verify_alpha=float(getattr(args, 'verify_alpha', 2.0)),
                                verifier=verifier,
                                page_tta=str(getattr(args, 'page_tta', 'none')))
                        except Exception as e:  # noqa: BLE001 - record, continue
                            prow['%s_s%d_error' % (domain, i)] = (
                                'decode-crash: %s' % e)
                            hyps_toks.append([])
                            continue
                    ht = (workdir / (stem + '-bekern.txt')).read_text(
                        encoding='utf-8').split()
                    hyps_toks.append(ht)
                    prow['%s_s%d_ntok' % (domain, i)] = len(ht)
                    prow['%s_s%d_geom' % (domain, i)] = dec.get('geom')
                    if dec.get('loop_trimmed'):
                        prow['%s_s%d_loop_trimmed' % (domain, i)] = True
                        domain_trimmed += 1
                    if dec.get('verifier_needs_review'):
                        prow['%s_s%d_needs_review' % (domain, i)] = True
                        prow['%s_s%d_low_conf_measures' % (domain, i)] = (
                            dec.get('low_confidence_measures', []))
                prow[dskey + '_loop_trimmed'] = domain_trimmed
                if domain == 'clean':
                    clean_loop_trimmed += domain_trimmed
                else:
                    photo_loop_trimmed += domain_trimmed
                # stitch + score at page level
                hyp_page = stitch_page_toks(hyps_toks)
                hyp_txt = workdir / ('s31-%s-%s-page-hyp-bekern.txt'
                                     % (slug, domain))
                hyp_txt.write_text(' '.join(hyp_page) + '\n',
                                   encoding='utf-8')
                # Reference page bekern for the notation scorer: the GT
                # MusicXML must NOT be scored directly — its voice ids /
                # staff tags use native numbering while bekern-derived
                # XML uses segment-ordinal voices and omits <staff>
                # (bekern carries no staff tokens). Routing the ref
                # through the same bekern round-trip keeps both sides in
                # one scheme, exactly like the sysdev page loop.
                ref_txt = workdir / ('s31-%s-%s-page-ref-bekern.txt'
                                     % (slug, domain))
                ref_txt.write_text(' '.join(ref_page_toks) + '\n',
                                   encoding='utf-8')
                hyp_notes = workdir / ('s31-%s-%s-page-hyp-notes.json'
                                       % (slug, domain))
                p = run_cli([ASER, 'bekern-decode', '--input', str(hyp_txt),
                             '--out', str(hyp_notes)])
                if p.returncode != 0 or not hyp_notes.exists():
                    prow[dskey + '_f1'] = 0.0
                    prow[dskey + '_ser'] = 1.0
                    prow[dskey + '_zero_reason'] = 'page-decode-rc-%d' % (
                        p.returncode)
                    prow[dskey + '_mod_f1'] = modifier_f1(
                        ref_page_toks, hyp_page)
                    if not args.no_notation:
                        prow[dskey + '_notation_f1'] = 0.0
                        prow[dskey + '_notation_drift'] = 0.0
                        prow[dskey + '_notation_zeroed'] = True
                        prow[dskey + '_notation_zero_reason'] = (
                            'page-decode-rc-%d' % p.returncode)
                else:
                    sc = score_notes_pair(page_ref_notes, hyp_notes,
                                          workdir,
                                          's31-%s-%s-page' % (slug, domain),
                                          EHARNESS, ASER)
                    prow[dskey + '_f1'] = sc.get('f1', 0.0)
                    prow[dskey + '_ser'] = sc.get('ser', 1.0)
                    prow[dskey + '_fixes'] = sc.get('proxy_fixes')
                    prow[dskey + '_mod_f1'] = modifier_f1(
                        ref_page_toks, hyp_page)
                    if sc.get('zero_reason'):
                        prow[dskey + '_zero_reason'] = sc['zero_reason']
                    if not args.no_notation:
                        nd = score_notation_bekerns(
                            ref_txt, hyp_txt, workdir,
                            's31-%s-%s-page-notation' % (slug, domain),
                            ASER, EHARNESS)
                        for k, v in nd.items():
                            prow['%s_%s' % (dskey, k)] = v
                print('%s [%s] f1=%.4f ser=%.4f' % (
                    slug, domain, prow.get(dskey + '_f1'),
                    prow.get(dskey + '_ser')), flush=True)
            prow['n_systems'] = n_clean_systems
            prow['deg_box_source'] = deg_src
            prow['ref_tokens'] = len(ref_page_toks)
        except Exception as e:  # noqa: BLE001 - record, never crash gate
            prow.update({'skipped': True,
                         'reason': 'piece-crash: %s' % str(e)[:160]})
            print('%s PIECE-CRASH %s' % (slug, e), flush=True)
        piece_rows.append(prow)

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    scored_rows = [r for r in piece_rows if not r.get('skipped')]
    summary = {
        'suite31_manifest': str(manifest_path),
        'suite31_manifest_sha256': manifest_sha,
        'n_pieces': len(pieces),
        'n_scored': len(scored_rows),
        'blind_photo': blind_photo,
        'blind_photo_detected_rate': (
            sum(1 for r in scored_rows
                if r.get('deg_box_source') in ('detected', 'staff-unet2d'))
            / len(scored_rows)
            if scored_rows else None),
        'clean_loop_trimmed': clean_loop_trimmed,
        'photo_loop_trimmed': photo_loop_trimmed,
        'suite31_photo_f1_mean': _mean(
            [r.get('photo_f1') for r in scored_rows]),
        'suite31_clean_f1_mean': _mean(
            [r.get('clean_f1') for r in scored_rows]),
        'suite31_photo_ser_mean': _mean(
            [r.get('photo_ser') for r in scored_rows]),
        'suite31_clean_ser_mean': _mean(
            [r.get('clean_ser') for r in scored_rows]),
        'suite31_photo_mod_f1_mean': _mean(
            [r.get('photo_mod_f1') for r in scored_rows]),
        'suite31_clean_mod_f1_mean': _mean(
            [r.get('clean_mod_f1') for r in scored_rows]),
        'suite31_photo_notation_f1_mean': _mean(
            [r.get('photo_notation_f1') for r in scored_rows]),
        'suite31_clean_notation_f1_mean': _mean(
            [r.get('clean_notation_f1') for r in scored_rows]),
        'suite31_clean_fixes_median': (
            statistics.median(
                [r['clean_fixes'] for r in scored_rows
                 if r.get('clean_fixes') is not None])
            if any(r.get('clean_fixes') is not None
                   for r in scored_rows) else None),
        'suite31_photo_fixes_median': (
            statistics.median(
                [r['photo_fixes'] for r in scored_rows
                 if r.get('photo_fixes') is not None])
            if any(r.get('photo_fixes') is not None
                   for r in scored_rows) else None),
    }
    strata_summary = {}
    for stratum_name in ('typeset', 'hw_landscape', 'hw_portrait_debussy',
                         'hw_portrait_pdm', 'hw_portrait_other'):
        s_rows = [r for r in piece_rows if r.get('stratum') == stratum_name]
        if s_rows:
            strata_summary[stratum_name] = _summarize_suite31_slice(s_rows)
    hw_rows = [r for r in piece_rows if r.get('handwritten')]
    if hw_rows:
        strata_summary['hw_all'] = _summarize_suite31_slice(hw_rows)
        hw_ok = [r for r in hw_rows if not r.get('skipped')]
        summary.update({
            'suite31_hw_n_pieces': len(hw_rows),
            'suite31_hw_n_scored': len(hw_ok),
            'suite31_hw_clean_f1_mean': _mean([r.get('clean_f1') for r in hw_ok]),
            'suite31_hw_photo_f1_mean': _mean([r.get('photo_f1') for r in hw_ok]),
            'suite31_hw_clean_notation_f1_mean': _mean([r.get('clean_notation_f1') for r in hw_ok]),
            'suite31_hw_photo_notation_f1_mean': _mean([r.get('photo_notation_f1') for r in hw_ok]),
        })
        ts_rows = [r for r in piece_rows if not r.get('handwritten')]
        if ts_rows:
            ts_ok = [r for r in ts_rows if not r.get('skipped')]
            summary.update({
                'suite31_typeset_n_pieces': len(ts_rows),
                'suite31_typeset_n_scored': len(ts_ok),
                'suite31_typeset_clean_f1_mean': _mean([r.get('clean_f1') for r in ts_ok]),
                'suite31_typeset_photo_f1_mean': _mean([r.get('photo_f1') for r in ts_ok]),
                'suite31_typeset_clean_notation_f1_mean': _mean([r.get('clean_notation_f1') for r in ts_ok]),
                'suite31_typeset_photo_notation_f1_mean': _mean([r.get('photo_notation_f1') for r in ts_ok]),
            })
    summary['suite31_strata'] = strata_summary
    return piece_rows, summary


if __name__ == '__main__':
    main()
