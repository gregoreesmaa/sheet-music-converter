#!/usr/bin/env python3
"""Verifier-Gated Self-Training ("Noisy Student") for MVD OMR.

Consumes an unlabeled strip/page manifest (JSONL rows with ``path``,
``domain``, ``id``) and a teacher ``MvdNet`` checkpoint, decodes candidate
bekern sequences, filters them through a 4-gate zero-label symbolic verifier,
and atomically emits accepted pseudo-labeled pairs to ``--out-dir``.

Four verifier gates:
  - Gate 1 (Syntax): ``assembly-serialize.exe bekern-decode`` exits 0 and
    produces non-empty notes JSON.
  - Gate 2 (Exact Bar-Duration Closure): every measure/voice's beat sum
    matches the active ``time-N-D`` signature (pickup bar 1 and final bar
    may be <= full measure; interior measures must close exactly; no
    measure may overflow).
  - Gate 3 (Confidence & Non-Loop): rejects ``loop_trimmed=True``,
    empty/short streams (``< 8`` tokens), or ``mean_logprob < min_logprob``
    (default ``-0.25``).
  - Gate 4 (Weak<->Strong FixMatch JSD Helper):
    :func:`unlabeled_consistency_step` wraps
    ``optics_losses.logit_jsd_consistency`` so high-confidence teacher
    predictions on weakly-augmented unlabeled strips supervise
    strongly-augmented student twins.

Held-out safety: strictly excludes any ``gate20`` / ``suite31`` /
``suite31_hw`` IDs or paths before decoding or emission.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = MVD_DIR.parent.parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import data  # noqa: E402
import grammar  # noqa: E402
import model  # noqa: E402
import optics_losses  # noqa: E402
import train  # noqa: E402

DEFAULT_MANIFEST = Path(r'D:\tmp\unlabeled-10k\unlabeled_10k_manifest.jsonl')
DEFAULT_ASER = (
    REPO_ROOT / 'converter' / 'assembly-serialize' / 'target' / 'release'
    / ('assembly-serialize.exe' if os.name == 'nt' else 'assembly-serialize')
)
GATE20_MANIFEST = (
    REPO_ROOT / 'data-gen' / 'score-corpus' / 'gate20' / 'gate_manifest.json'
)
SUITE31_MANIFEST = (
    REPO_ROOT / 'data-gen' / 'score-corpus' / 'suite31' / 'suite31_manifest.json'
)
SUITE31_HW_MANIFEST = (
    REPO_ROOT / 'data-gen' / 'score-corpus' / 'suite31'
    / 'suite31_hw_only_manifest.json'
)

HELD_OUT_KEYWORDS = ('gate20', 'suite31', 'suite31_hw')

_WRITTEN_DUR: Dict[str, float] = {
    'long': 16.0,
    'breve': 8.0,
    'whole': 4.0,
    'half': 2.0,
    'quarter': 1.0,
    'eighth': 0.5,
    '16th': 0.25,
    '32nd': 0.125,
    '64th': 0.0625,
    '128th': 0.03125,
}
_STEPS = frozenset('ABCDEFG')
_OCTAVES = frozenset(str(i) for i in range(10))
_BAR_CLOSE = frozenset(('bar', 'bar-light-heavy', 'bar-repeat-right', 'bar-double'))
_BAR_OPEN = frozenset(('bar-repeat-left',))


@dataclass
class VerifyResult:
    """Structured result of :func:`verify_pseudo_bekern`."""

    ok: bool
    gate: str
    reason: str
    n_notes: int = 0
    n_measures: int = 0
    details: Dict[str, object] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok


def load_held_out_ids(
    gate20_manifest: Path = GATE20_MANIFEST,
    suite31_manifest: Path = SUITE31_MANIFEST,
    suite31_hw_manifest: Path = SUITE31_HW_MANIFEST,
) -> Set[str]:
    """Collect all held-out piece/page IDs and slugs from gate20 & suite31."""
    blocked: Set[str] = set()
    if gate20_manifest.is_file():
        try:
            obj = json.loads(gate20_manifest.read_text(encoding='utf-8'))
            for page in obj.get('pages', []):
                for key in ('id', 'score'):
                    val = page.get(key)
                    if val:
                        blocked.add(str(val).lower())
        except (OSError, json.JSONDecodeError):
            pass
    for mf in (suite31_manifest, suite31_hw_manifest):
        if mf.is_file():
            try:
                obj = json.loads(mf.read_text(encoding='utf-8'))
                for piece in obj.get('pieces', []):
                    for key in ('slug', 'id'):
                        val = piece.get(key)
                        if val:
                            blocked.add(str(val).lower())
            except (OSError, json.JSONDecodeError):
                pass
    return blocked


def is_held_out_entry(
    entry_id: str,
    path: str = '',
    domain: str = '',
    held_out_ids: Optional[Set[str]] = None,
) -> bool:
    """Return True iff an unlabeled manifest entry touches any held-out set."""
    fields = (
        str(entry_id or '').lower(),
        str(path or '').lower().replace('\\', '/'),
        str(domain or '').lower(),
    )
    for text in fields:
        for kw in HELD_OUT_KEYWORDS:
            if kw in text:
                return True
    if held_out_ids:
        eid = fields[0]
        stem = Path(fields[1]).stem.lower() if fields[1] else ''
        for candidate in (eid, stem):
            if not candidate:
                continue
            if candidate in held_out_ids:
                return True
            for hid in held_out_ids:
                if (candidate.startswith(hid + '-')
                        or candidate.startswith(hid + '_')
                        or candidate.endswith('-' + hid)
                        or candidate.endswith('_' + hid)):
                    return True
    return False


def _parse_time_sig(tok: str) -> Optional[float]:
    """Parse ``time-N-D`` -> quarter-beat measure duration ``N * 4.0 / D``."""
    parts = tok.split('-')
    if len(parts) != 3 or parts[0] != 'time':
        return None
    try:
        num = int(parts[1])
        den = int(parts[2])
    except ValueError:
        return None
    if num <= 0 or den <= 0:
        return None
    return float(num) * 4.0 / float(den)


def _parse_tuplet_start(tok: str) -> Optional[Tuple[int, int]]:
    """Parse ``tuplet-start-A-N`` -> ``(actual, normal)``."""
    parts = tok.split('-')
    if len(parts) != 4 or parts[0] != 'tuplet' or parts[1] != 'start':
        return None
    try:
        actual = int(parts[2])
        normal = int(parts[3])
    except ValueError:
        return None
    if actual <= 0 or normal <= 0:
        return None
    return (actual, normal)


def check_bar_duration_closure(
    toks: Sequence[str],
    eps: float = 1e-4,
) -> Tuple[bool, str, Dict[str, object]]:
    """Gate 2: verify exact per-measure, per-voice beat sums against ``time-N-D``.

    Rules:
      - An active ``time-N-D`` signature must be present before/at musical
        events (yielding ``measure_beats = N * 4 / D``).
      - ``bar-rest`` closes its voice cleanly to ``measure_beats``.
      - ``chord`` notes and ``grace`` notes add ``0`` beats.
      - Dots multiply the written duration by ``2 - 2**(-n_dots)``.
      - Open ``tuplet-start-A-N`` ... ``tuplet-stop`` spans scale duration by
        ``N / A`` (nested spans multiply).
      - Measure 1 (pickup/anacrusis) and the final measure may be partial
        (``0 < dur <= measure_beats + eps``).
      - Every interior measure (and every voice within it) must close
        exactly (``abs(dur - measure_beats) <= eps``).
      - No measure or voice may overflow ``measure_beats + eps``.
    """
    tsig: Optional[float] = None
    pending_tsig: Optional[float] = None
    measures: List[Tuple[float, Dict[int, float]]] = []

    cur_voice = 1
    v_durs: Dict[int, float] = {}
    v_scales: Dict[int, List[float]] = {}
    pend_scales: List[float] = []
    cur_has_events = False

    # Pending note/rest event being accumulated (dots follow pitch/rest).
    ev_voice: Optional[int] = None
    ev_base: Optional[float] = None
    ev_dots: int = 0
    ev_chord: bool = False
    ev_grace: bool = False

    def flush_event() -> None:
        nonlocal ev_voice, ev_base, ev_dots, ev_chord, ev_grace, cur_has_events
        if ev_base is None or ev_voice is None:
            return
        if not ev_chord and not ev_grace:
            dot_mult = 2.0 - (2.0 ** (-ev_dots))
            scale = 1.0
            for s in v_scales.get(ev_voice, ()):
                scale *= s
            dur = ev_base * dot_mult * scale
            v_durs[ev_voice] = v_durs.get(ev_voice, 0.0) + dur
            cur_has_events = True
        else:
            # Ensure chord/grace still marks the measure as active if alone
            cur_has_events = True
        ev_voice = None
        ev_base = None
        ev_dots = 0
        ev_chord = False
        ev_grace = False

    def close_measure() -> Optional[str]:
        nonlocal tsig, pending_tsig, cur_voice, v_durs, v_scales, cur_has_events
        flush_event()
        if cur_has_events:
            if tsig is None:
                return 'missing_time_signature'
            measures.append((tsig, dict(v_durs)))
        if pending_tsig is not None:
            tsig = pending_tsig
            pending_tsig = None
        cur_voice = 1
        v_durs = {}
        v_scales = {}
        cur_has_events = False
        return None

    i = 0
    n = len(toks)
    next_is_chord = False
    next_is_grace = False

    while i < n:
        t = toks[i]
        if t == 'dot':
            if ev_base is not None:
                ev_dots += 1
            i += 1
            continue
        if (t in _STEPS or t in _OCTAVES or t.startswith('a') and t != 'accent'
                and len(t) >= 2 and (t[1:].isdigit()
                                     or (t[1] == '-' and t[2:].isdigit()))):
            i += 1
            continue
        if t in ('tie-start', 'tie-stop', 'slur-start', 'slur-stop',
                 'stacc', 'accent', 'tenuto', 'fermata'):
            i += 1
            continue

        # Any other token terminates an in-progress note/rest event.
        flush_event()

        if t.startswith('time-'):
            beats = _parse_time_sig(t)
            if beats is None:
                return False, f'invalid_time_sig:{t}', {}
            if cur_has_events:
                pending_tsig = beats
            else:
                tsig = beats
            i += 1
            continue

        if t in _BAR_CLOSE:
            err = close_measure()
            if err is not None:
                return False, err, {}
            i += 1
            continue

        if t in _BAR_OPEN or t.startswith('clef-') or t.startswith('key') \
                or t.startswith('dyn-') or t.startswith('hairpin-'):
            i += 1
            continue

        if t.startswith('voice-'):
            try:
                v_num = int(t.split('-', 1)[1])
            except ValueError:
                return False, f'invalid_voice:{t}', {}
            cur_voice = v_num
            i += 1
            continue

        if t.startswith('tuplet-start-'):
            ratio = _parse_tuplet_start(t)
            if ratio is None:
                return False, f'invalid_tuplet:{t}', {}
            actual, normal = ratio
            pend_scales.append(float(normal) / float(actual))
            i += 1
            continue

        if t == 'tuplet-stop':
            stk = v_scales.get(cur_voice)
            if stk:
                stk.pop()
            i += 1
            continue

        if t == 'chord':
            next_is_chord = True
            i += 1
            continue

        if t == 'grace':
            next_is_grace = True
            i += 1
            continue

        if t == 'bar-rest':
            if tsig is None:
                return False, 'missing_time_signature', {}
            if pend_scales:
                v_scales.setdefault(cur_voice, []).extend(pend_scales)
                pend_scales.clear()
            v_durs[cur_voice] = tsig
            cur_has_events = True
            next_is_chord = False
            next_is_grace = False
            i += 1
            continue

        if t == 'rest':
            if i + 1 >= n or toks[i + 1] not in _WRITTEN_DUR:
                return False, 'malformed_rest', {}
            if pend_scales:
                v_scales.setdefault(cur_voice, []).extend(pend_scales)
                pend_scales.clear()
            ev_voice = cur_voice
            ev_base = _WRITTEN_DUR[toks[i + 1]]
            ev_dots = 0
            ev_chord = False
            ev_grace = False
            next_is_chord = False
            next_is_grace = False
            i += 2
            continue

        if t in _WRITTEN_DUR:
            if pend_scales:
                v_scales.setdefault(cur_voice, []).extend(pend_scales)
                pend_scales.clear()
            ev_voice = cur_voice
            ev_base = _WRITTEN_DUR[t]
            ev_dots = 0
            ev_chord = next_is_chord
            ev_grace = next_is_grace
            next_is_chord = False
            next_is_grace = False
            i += 1
            continue

        i += 1

    # Flush trailing measure if tokens did not end with a barline.
    err = close_measure()
    if err is not None:
        return False, err, {}

    n_measures = len(measures)
    if n_measures == 0:
        return False, 'no_measures', {'n_measures': 0}

    for m_idx, (expected, vd) in enumerate(measures):
        is_first = (m_idx == 0)
        is_last = (m_idx == n_measures - 1)
        for v_id, dur in sorted(vd.items()):
            if dur > expected + eps:
                return (
                    False,
                    f'bar_overflow:m{m_idx + 1}:v{v_id}:{dur:.4f}>{expected:.4f}',
                    {'measure': m_idx + 1, 'voice': v_id,
                     'dur': dur, 'expected': expected,
                     'n_measures': n_measures},
                )
            if not is_first and not is_last:
                if abs(dur - expected) > eps:
                    return (
                        False,
                        f'bar_underflow:m{m_idx + 1}:v{v_id}:{dur:.4f}<{expected:.4f}',
                        {'measure': m_idx + 1, 'voice': v_id,
                         'dur': dur, 'expected': expected,
                         'n_measures': n_measures},
                    )
    return True, 'ok', {'n_measures': n_measures}


def _check_bekern_decode(
    toks: Sequence[str],
    aser_bin: Path,
    timeout: float = 15.0,
) -> Tuple[bool, str, int]:
    """Gate 1: run ``assembly-serialize bekern-decode`` and check non-empty notes."""
    aser_path = Path(aser_bin)
    if not aser_path.is_file():
        return False, f'aser_missing:{aser_path}', 0

    with tempfile.TemporaryDirectory(prefix='st_bekern_') as td:
        txt_path = Path(td) / 'cand.txt'
        json_path = Path(td) / 'cand.json'
        txt_path.write_text(' '.join(toks) + '\n', encoding='utf-8')
        try:
            proc = subprocess.run(
                [str(aser_path), 'bekern-decode',
                 '--input', str(txt_path),
                 '--out', str(json_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f'bekern_decode_exec_error:{exc}', 0

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or '').strip().splitlines()
            msg = err[-1] if err else f'rc={proc.returncode}'
            return False, f'bekern_decode_rc_{proc.returncode}:{msg}', 0

        if not json_path.is_file():
            return False, 'bekern_decode_no_json', 0
        try:
            obj = json.loads(json_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f'bekern_decode_bad_json:{exc}', 0

        n_notes = 0
        if isinstance(obj, dict):
            if isinstance(obj.get('parts'), list):
                for part in obj['parts']:
                    if isinstance(part, dict) and isinstance(part.get('notes'), list):
                        n_notes += len(part['notes'])
            elif isinstance(obj.get('notes'), list):
                n_notes = len(obj['notes'])
        elif isinstance(obj, list):
            n_notes = len(obj)
        if n_notes <= 0:
            return False, 'bekern_decode_empty_notes', 0
        return True, 'ok', n_notes


def verify_pseudo_bekern(
    toks: Sequence[str],
    aser_bin: Path = DEFAULT_ASER,
    mean_logprob: Optional[float] = None,
    loop_trimmed: bool = False,
    min_logprob: float = -0.25,
    min_tokens: int = 8,
    eps: float = 1e-4,
    timeout: float = 15.0,
) -> VerifyResult:
    """Run the 3 symbolic/confidence verifier gates on a pseudo-label candidate.

    Gates checked:
      - Gate 3 (Confidence & Non-Loop): rejects ``loop_trimmed=True``,
        ``len(toks) < min_tokens``, or ``mean_logprob < min_logprob``.
      - Gate 2 (Exact Bar-Duration Closure): checks every measure/voice
        against active ``time-N-D`` (pickup bar 1 and final bar <= full
        measure; interior bars exact; no overflows).
      - Gate 1 (Syntax): runs ``assembly-serialize.exe bekern-decode`` and
        requires exit code 0 + non-empty notes JSON.
    """
    tok_list = [str(t) for t in toks if str(t).strip()]

    # Gate 3: Confidence & Non-Loop (cheap check first)
    if loop_trimmed:
        return VerifyResult(
            ok=False, gate='gate3_confidence', reason='loop_trimmed',
        )
    if len(tok_list) < int(min_tokens):
        return VerifyResult(
            ok=False, gate='gate3_confidence',
            reason=f'too_short:{len(tok_list)}<{min_tokens}',
        )
    if mean_logprob is not None:
        if (not math.isfinite(float(mean_logprob))
                or float(mean_logprob) < float(min_logprob)):
            return VerifyResult(
                ok=False, gate='gate3_confidence',
                reason=f'low_logprob:{float(mean_logprob):.4f}<{min_logprob}',
            )

    # Gate 2: Exact Bar-Duration Closure
    ok_bar, bar_reason, bar_details = check_bar_duration_closure(
        tok_list, eps=eps,
    )
    if not ok_bar:
        return VerifyResult(
            ok=False, gate='gate2_bar_duration', reason=bar_reason,
            n_measures=int(bar_details.get('n_measures', 0)),
            details=bar_details,
        )

    # Gate 1: Live assembly-serialize bekern-decode syntax check
    ok_syn, syn_reason, n_notes = _check_bekern_decode(
        tok_list, aser_bin=Path(aser_bin), timeout=timeout,
    )
    if not ok_syn:
        return VerifyResult(
            ok=False, gate='gate1_syntax', reason=syn_reason,
            n_measures=int(bar_details.get('n_measures', 0)),
            details=bar_details,
        )

    return VerifyResult(
        ok=True,
        gate='passed',
        reason='ok',
        n_notes=n_notes,
        n_measures=int(bar_details.get('n_measures', 0)),
        details=bar_details,
    )


def unlabeled_consistency_step(
    net: model.MvdNet,
    weak_imgs: torch.Tensor,
    strong_imgs: torch.Tensor,
    pad_id: int = data.PAD_ID,
    temperature: float = 1.5,
    conf_thresh: float = 0.85,
    max_len: int = 128,
    constrain: bool = False,
    mem_mask_weak: Optional[torch.Tensor] = None,
    mem_mask_strong: Optional[torch.Tensor] = None,
    pseudo_tgt: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Gate 4: Weak<->Strong FixMatch JSD consistency step on unlabeled strips.

    1. Obtains pseudo-target token sequences ``tgt`` (either caller-supplied
       ``pseudo_tgt`` or greedy-decoded from ``weak_imgs`` with ``no_grad``).
    2. Runs a no-grad teacher pass on ``weak_imgs`` to obtain
       ``logits_weak`` and per-position max softmax confidence.
    3. Masks out ``pad_id`` positions and low-confidence positions
       (``conf < conf_thresh``).
    4. Runs a gradient-enabled student pass on ``strong_imgs`` to obtain
       ``logits_strong`` and computes symmetric JSD via
       :func:`optics_losses.logit_jsd_consistency` with ``detach_clean=True``.

    Returns:
        ``(loss, stats)`` where ``loss`` is a scalar ``float32`` tensor with
        gradients flowing strictly through ``strong_imgs`` / ``net`` student
        path, and ``stats`` reports ``keep_frac``, ``n_active``,
        ``mean_conf``, and ``seq_len``.
    """
    if weak_imgs.shape[0] != strong_imgs.shape[0]:
        raise ValueError(
            f'batch size mismatch: weak={weak_imgs.shape[0]} vs '
            f'strong={strong_imgs.shape[0]}'
        )
    if not (0.0 <= conf_thresh <= 1.0):
        raise ValueError(f'conf_thresh must be in [0, 1], got {conf_thresh}')

    was_training = net.training

    if pseudo_tgt is None:
        with torch.no_grad():
            tgt = net.greedy(
                weak_imgs, max_len=max_len, constrain=constrain,
                mem_mask=mem_mask_weak,
            )
        # MvdNet.greedy() flips net.eval(); restore caller mode.
        net.train(was_training)
    else:
        tgt = pseudo_tgt.to(device=weak_imgs.device, dtype=torch.long)

    bos = torch.full(
        (tgt.shape[0], 1), net.bos_id, dtype=torch.long, device=tgt.device,
    )
    tgt_in = torch.cat([bos, tgt[:, :-1]], dim=1)

    with torch.no_grad():
        logits_weak = net(weak_imgs, tgt_in, mem_mask=mem_mask_weak)
        probs_weak = F.softmax(logits_weak.float(), dim=-1)
        conf, _ = probs_weak.max(dim=-1)

    non_pad = (tgt != pad_id)
    keep = non_pad & (conf >= float(conf_thresh))
    ignore_mask = ~keep

    logits_strong = net(strong_imgs, tgt_in, mem_mask=mem_mask_strong)
    loss = optics_losses.logit_jsd_consistency(
        logits_weak,
        logits_strong,
        pad_mask=ignore_mask,
        temperature=float(temperature),
        detach_clean=True,
    )

    n_non_pad = int(non_pad.sum().item())
    n_active = int(keep.sum().item())
    mean_conf = (
        float(conf[non_pad].mean().item()) if n_non_pad > 0 else 0.0
    )
    stats = {
        'keep_frac': float(n_active) / float(max(1, n_non_pad)),
        'n_active': float(n_active),
        'mean_conf': mean_conf,
        'seq_len': float(tgt.shape[1]),
    }
    return loss, stats


def detect_repeat_loop(
    toks: Sequence[str],
    max_len: int = 564,
    min_period: int = 8,
    min_repeats: int = 4,
) -> bool:
    """Return True if a max-length decode ended in a verbatim repeating tail."""
    n = len(toks)
    if n < min(max_len, 564) - 2:
        return False
    max_p = min(64, n // min_repeats)
    for p in range(min_period, max_p + 1):
        block = toks[n - p:n]
        reps = 1
        pos = n - 2 * p
        while pos >= 0 and toks[pos:pos + p] == block:
            reps += 1
            pos -= p
        if reps >= min_repeats:
            return True
    return False


@torch.no_grad()
def decode_teacher_strip(
    net: model.MvdNet,
    img: torch.Tensor,
    max_len: int = 564,
    constrain: bool = True,
    strict: bool = True,
) -> Tuple[List[str], float, bool]:
    """Decode one image tensor ``[1, C, H, W]`` -> ``(toks, mean_logprob, loop_trimmed)``."""
    net.eval()
    ids_tensor = net.greedy_kv(
        img, max_len=max_len, constrain=constrain, strict=strict,
    )
    raw_ids = ids_tensor[0].tolist()
    if net.eos_id in raw_ids:
        raw_ids = raw_ids[:raw_ids.index(net.eos_id)]
    content_ids = [
        t for t in raw_ids
        if t not in (net.pad_id, net.bos_id, net.eos_id)
        and 0 <= t < len(data.VOCAB)
    ]
    if not content_ids:
        return [], float('-inf'), False

    # Compute exact sequence mean log-prob via one teacher-forced forward pass.
    tgt = torch.tensor(
        [content_ids], dtype=torch.long, device=img.device,
    )
    bos = torch.full((1, 1), net.bos_id, dtype=torch.long, device=img.device)
    tgt_in = torch.cat([bos, tgt[:, :-1]], dim=1)
    logits = net(img, tgt_in)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    chosen = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mean_lp = float(chosen.mean().item())

    toks = data.decode(content_ids)
    loop_trimmed = detect_repeat_loop(toks, max_len=max_len)
    return toks, mean_lp, loop_trimmed


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def _atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def load_unlabeled_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    """Read JSONL rows with ``path``, ``domain``, ``id`` fields."""
    rows: List[Dict[str, str]] = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if 'path' not in obj or 'id' not in obj:
                raise ValueError(
                    f'{manifest_path}:{line_no} missing required path/id keys'
                )
            rows.append({
                'id': str(obj['id']),
                'path': str(obj['path']),
                'domain': str(obj.get('domain', 'unknown')),
            })
    return rows


def emit_pseudo_dataset(
    accepted: Sequence[Dict[str, object]],
    out_dir: Path,
) -> Path:
    """Atomically emit ``renders/``, ``targets/*-bekern.txt``, and ``pseudo_manifest.jsonl``."""
    out_dir = Path(out_dir)
    renders_dir = out_dir / 'renders'
    targets_dir = out_dir / 'targets'
    renders_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines: List[str] = []
    for row in accepted:
        eid = str(row['id'])
        src_path = Path(str(row['path']))
        toks = list(row['tokens'])  # type: ignore[arg-type]
        ext = src_path.suffix if src_path.suffix else '.png'
        dst_img = renders_dir / f'{eid}{ext}'
        dst_tok = targets_dir / f'{eid}-bekern.txt'
        if src_path.is_file():
            _atomic_copy_file(src_path, dst_img)
        _atomic_write_text(dst_tok, ' '.join(toks) + '\n')

        record = {
            'id': eid,
            'domain': str(row.get('domain', 'unknown')),
            'source_path': str(src_path),
            'render': str(dst_img),
            'target': str(dst_tok),
            'n_tokens': len(toks),
            'n_notes': int(row.get('n_notes', 0)),
            'n_measures': int(row.get('n_measures', 0)),
            'mean_logprob': float(row.get('mean_logprob', 0.0)),
        }
        manifest_lines.append(json.dumps(record, sort_keys=True))

    manifest_path = out_dir / 'pseudo_manifest.jsonl'
    _atomic_write_text(
        manifest_path,
        ('\n'.join(manifest_lines) + '\n') if manifest_lines else '',
    )
    return manifest_path


def run_self_training_mining(
    manifest_path: Path,
    ckpt_path: Path,
    out_dir: Path,
    aser_bin: Path = DEFAULT_ASER,
    height: int = 256,
    max_len: int = 564,
    min_logprob: float = -0.25,
    min_tokens: int = 8,
    stride: str = '4x8',
    extra_chans: int = 0,
    foveal_tap: bool = False,
    device: str = 'cpu',
    limit: int = 0,
) -> Dict[str, int]:
    """Mine verifier-gated pseudo-labels from an unlabeled manifest."""
    rows = load_unlabeled_manifest(manifest_path)
    held_out = load_held_out_ids()

    net = model.MvdNet(
        vocab=len(data.VOCAB) + 3,
        anisotropic=True,
        stride=stride,
        extra_chans=extra_chans,
        foveal_tap=foveal_tap,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    train.load_weights_compat(net, state)
    net.eval()

    counts: Dict[str, int] = {
        'total': 0,
        'excluded_held_out': 0,
        'missing_image': 0,
        'accepted': 0,
        'rejected_gate1_syntax': 0,
        'rejected_gate2_bar_duration': 0,
        'rejected_gate3_confidence': 0,
    }
    accepted: List[Dict[str, object]] = []

    for row in rows:
        if limit > 0 and counts['total'] >= limit:
            break
        counts['total'] += 1
        if is_held_out_entry(row['id'], row['path'], row['domain'], held_out):
            counts['excluded_held_out'] += 1
            continue
        img_path = Path(row['path'])
        if not img_path.is_file():
            counts['missing_image'] += 1
            continue

        img = train.load_image(str(img_path), height).unsqueeze(0).to(device)
        if extra_chans > 0:
            zeros = torch.zeros(
                (1, extra_chans, img.shape[2], img.shape[3]),
                dtype=img.dtype, device=img.device,
            )
            img = torch.cat([img, zeros], dim=1)

        toks, mean_lp, loop_trimmed = decode_teacher_strip(
            net, img, max_len=max_len, constrain=True, strict=True,
        )
        ver = verify_pseudo_bekern(
            toks,
            aser_bin=aser_bin,
            mean_logprob=mean_lp,
            loop_trimmed=loop_trimmed,
            min_logprob=min_logprob,
            min_tokens=min_tokens,
        )
        if ver.ok:
            counts['accepted'] += 1
            accepted.append({
                'id': row['id'],
                'path': str(img_path),
                'domain': row['domain'],
                'tokens': toks,
                'n_notes': ver.n_notes,
                'n_measures': ver.n_measures,
                'mean_logprob': mean_lp,
            })
        else:
            key = f'rejected_{ver.gate}'
            counts[key] = counts.get(key, 0) + 1

    emit_pseudo_dataset(accepted, out_dir)
    return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description='Verifier-gated self-training (Noisy Student) for MVD OMR.',
    )
    ap.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST,
                    help='Unlabeled JSONL manifest (path, domain, id)')
    ap.add_argument('--ckpt', type=Path, required=True,
                    help='Teacher MvdNet checkpoint (.pt)')
    ap.add_argument('--out-dir', type=Path, required=True,
                    help='Output directory for accepted pseudo-labeled pairs')
    ap.add_argument('--aser', type=Path, default=DEFAULT_ASER,
                    help='Path to assembly-serialize binary')
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--max-len', type=int, default=564)
    ap.add_argument('--min-logprob', type=float, default=-0.25)
    ap.add_argument('--min-tokens', type=int, default=8)
    ap.add_argument('--stride', choices=('8x16', '4x8'), default='4x8')
    ap.add_argument('--extra-chans', type=int, default=0)
    ap.add_argument('--foveal-tap', action='store_true')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--limit', type=int, default=0,
                    help='Optional max manifest rows to process (0 = all)')
    args = ap.parse_args(argv)

    counts = run_self_training_mining(
        manifest_path=args.manifest,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        aser_bin=args.aser,
        height=args.height,
        max_len=args.max_len,
        min_logprob=args.min_logprob,
        min_tokens=args.min_tokens,
        stride=args.stride,
        extra_chans=args.extra_chans,
        foveal_tap=args.foveal_tap,
        device=args.device,
        limit=args.limit,
    )
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
