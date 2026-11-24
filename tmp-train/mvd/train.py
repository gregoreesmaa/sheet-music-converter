"""MVD v1 trainer: page PNGs -> bekern ids (teacher-forced NLL).

Real experiment harness (lives in tmp-train, not shipped). Usage:

    .venv/Scripts/python.exe tmp-train/mvd/train.py \
        --renders data-gen/score-corpus/big500-renders \
        --targets tmp-train/mvd-data \
        --pages ds0001-1 --steps 500 --eval-every 100 --out tmp-train/mvd-runs/overfit1

Pairing: each ``<stem>-bekern.txt`` in targets pairs with
``<stem>.png`` (clean) PLUS every ``<stem>_d*.jpg`` (degraded variants)
in renders (page stems like ``ds0001-1`` and system stems like
``ds0001-1-s0`` alike); stems in ``quarantine.txt``
(pagination-mismatched renders, untypeable pages) are never loaded.
Images are resized to ``--height`` (aspect kept, LANCZOS); batch widths
are padded to the batch max with the standardized white-paper value and
targets with ``data.PAD_ID``. Reports teacher-forced NLL plus, on eval
steps, greedy token error rate (Levenshtein / reference length) on the
first eval page. Checkpoints: ``ckpt-<step>.pt`` + ``meta.json``
(vocab size pinned).
"""

from __future__ import annotations

import argparse
import bisect
import importlib
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import model  # noqa: E402
import heat_features  # noqa: E402
import compact_sidecar  # noqa: E402
import epoch_aug  # noqa: E402
import fast_loader  # noqa: E402
import optics_losses  # noqa: E402


def list_pairs(renders, targets, exclude=(), include_degraded=True,
               include_handink=False, include_simshots=False):
    """Pair every ``<stem>-bekern.txt`` with its images.

    Each bekern stem pairs with ``<stem>.png`` (clean) plus all
    ``<stem>_d*.jpg`` (degraded variants) found in renders — so one
    system ``<stem>-bekern.txt`` yields 1 clean + N degraded pairs
    (``include_degraded=False`` restores clean-only pairing). Stems in
    ``quarantine.txt`` or matching ``exclude`` are skipped before
    pairing. Sorted by bekern stem, clean first then degraded.
    Spike S10: ``include_handink=True`` additionally pairs
    ``<stem>_h*.png/.jpg`` (hand-ink variants, same targets),
    appended after the degraded pairs of their stem.
    S16b: ``include_simshots=True`` additionally pairs
    ``<stem>_s*.png/.jpg`` (simulated screen captures, same targets),
    appended after any hand-ink pairs of their stem.
    """
    renders = Path(renders)
    targets = Path(targets)
    quar = set()
    qf = targets / 'quarantine.txt'
    if qf.exists():
        quar = {ln.strip() for ln in qf.read_text(encoding='utf-8').split()
                if ln.strip()}
    # One sorted listing up front: the old per-stem
    # renders.glob(stem + '_d*.jpg') re-scanned the whole directory per
    # bekern stem (O(stems x files); ~3e9 comparisons on the full
    # corpus, hanging startup for tens of minutes). Prefix-range search
    # over the single listing has identical results (fnmatch '*' here
    # can only match within a filename, i.e. startswith + endswith).
    names = sorted(os.listdir(renders))
    pairs = []
    for bek in sorted(targets.glob('*-bekern.txt')):
        stem = bek.name[:-len('-bekern.txt')]
        if any(s in stem for s in exclude):
            continue
        if stem in quar:
            continue
        clean = renders / (stem + '.png')
        if clean.exists():
            pairs.append((str(clean), str(bek), stem))
        if include_degraded:
            pre = stem + '_d'
            i = bisect.bisect_left(names, pre)
            while i < len(names) and names[i].startswith(pre):
                if names[i].endswith('.jpg'):
                    pairs.append((str(renders / names[i]), str(bek), stem))
                i += 1
        if include_handink:
            pre = stem + '_h'
            i = bisect.bisect_left(names, pre)
            while i < len(names) and names[i].startswith(pre):
                if names[i].endswith(('.png', '.jpg')):
                    pairs.append((str(renders / names[i]), str(bek), stem))
                i += 1
        if include_simshots:
            pre = stem + '_s'
            i = bisect.bisect_left(names, pre)
            while i < len(names) and names[i].startswith(pre):
                if names[i].endswith(('.png', '.jpg')):
                    pairs.append((str(renders / names[i]), str(bek), stem))
                i += 1
    return pairs


def load_real_hw_pairs(hw_dir, exclude=(), include_degraded=True, repeat=1):
    """Load real handwritten training pairs from ``<hw_dir>/{renders,targets}``.

    Follows the exact ``list_pairs`` contract (``quarantine.txt``,
    ``exclude``, clean ``.png`` + optional ``_d*.jpg`` degraded variants,
    deterministic sorted order). ``repeat >= 1`` deterministically tiles
    the sorted pair list (no RNG) so a small real handwritten subset can
    be upsampled cleanly in the mix.
    """
    if repeat < 1:
        raise ValueError('real_hw_repeat must be >= 1, got %r' % repeat)
    root = Path(hw_dir)
    renders = root / 'renders'
    targets = root / 'targets'
    if not renders.is_dir() or not targets.is_dir():
        raise FileNotFoundError(
            'real-hw-dir %s must contain renders/ and targets/' % root)
    base = list_pairs(str(renders), str(targets), exclude=exclude,
                      include_degraded=include_degraded)
    return base * int(repeat)


def load_real_hw_hybrid_index(hw_dir):
    """Load optional ``<hw_dir>/hybrid_staff.jsonl`` for real HW strips."""
    jpath = Path(hw_dir) / 'hybrid_staff.jsonl'
    if not jpath.exists():
        return {}
    return load_hybrid_index(str(jpath))


def parse_extra_data_dirs(raw):
    """Normalize repeatable/comma-separated ``--extra-data-dir`` values."""
    if not raw:
        return []
    items = [raw] if isinstance(raw, str) else list(raw)
    out = []
    for item in items:
        if not item:
            continue
        for part in str(item).split(','):
            p = part.strip()
            if p:
                out.append(p)
    return out


def load_extra_data_dir(dir_path, exclude=(), include_degraded=True,
                        include_handink=False, include_simshots=False):
    """Load pairs + optional ``hybrid_staff.jsonl`` from one extra data dir.

    Supports both flat layout (PNGs, ``*-bekern.txt``, and optional
    ``hybrid_staff.jsonl`` directly inside ``dir_path``) and split layout
    (``<dir_path>/renders/`` + ``<dir_path>/targets/``).
    """
    root = Path(dir_path)
    if not root.is_dir():
        raise FileNotFoundError('extra-data-dir %s does not exist' % root)
    if (root / 'renders').is_dir() and (root / 'targets').is_dir():
        renders, targets = root / 'renders', root / 'targets'
    else:
        renders, targets = root, root
    extra_pairs = list_pairs(
        str(renders), str(targets), exclude=exclude,
        include_degraded=include_degraded,
        include_handink=include_handink,
        include_simshots=include_simshots)
    jpath = root / 'hybrid_staff.jsonl'
    extra_hidx = load_hybrid_index(str(jpath)) if jpath.exists() else {}
    return extra_pairs, extra_hidx


def load_real_hw_corpus_pairs(corpus_path, exclude=(), include_degraded=True):
    """Load strictly labeled real handwritten pairs from ``--real-hw-corpus``.

    Supports:
      1. Unified corpus registry (``<corpus_path>/train_works.jsonl`` or a
         direct ``.jsonl`` path) such as ``D:\\tmp\\real-hw-corpus\\``
         (48 OpenScore Lieder works + 4 Mozart quartet works). Enforces
         ``bekern_ok == True`` and explicitly excludes any RISM-sourced rows
         so unlabeled RISM data can never enter the supervised HW pool.
      2. Split strip layout (``renders/`` + ``targets/`` + optional
         ``hybrid_staff.jsonl``).
      3. Flat strip layout (PNG/JPG + ``*-bekern.txt`` + optional
         ``hybrid_staff.jsonl``).

    Returns ``(pairs, hybrid_index)`` in deterministic sorted order.
    """
    root = Path(corpus_path)
    if not root.exists():
        raise FileNotFoundError('real-hw-corpus %s does not exist' % root)

    manifest = root if root.is_file() else (root / 'train_works.jsonl')
    if manifest.is_file():
        pairs = []
        hidx = {}
        seen = set()
        loaded_hjsonl = set()
        with open(manifest, encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                row = json.loads(ln)
                if not row.get('bekern_ok', False):
                    continue
                src = str(row.get('source', '')).lower()
                if 'rism' in src:
                    continue
                wid = str(row.get('work_id', '')).strip()
                if wid and any(s in wid for s in exclude):
                    continue
                hj = row.get('hybrid_staff_jsonl')
                if hj and hj not in loaded_hjsonl and Path(hj).is_file():
                    hidx.update(load_hybrid_index(hj))
                    loaded_hjsonl.add(hj)

                page_paths = list(row.get('page_paths') or [])
                bekern_paths = list(row.get('bekern_paths') or [])
                single_bek = row.get('bekern_path')

                if bekern_paths and len(bekern_paths) == len(page_paths):
                    stem_to_bek = {}
                    for ip, bp in zip(page_paths, bekern_paths):
                        if not Path(ip).is_file() or not Path(bp).is_file():
                            continue
                        stem = Path(ip).stem
                        if any(s in stem for s in exclude):
                            continue
                        stem_to_bek[stem] = bp
                        key = (str(ip), str(bp))
                        if key not in seen:
                            seen.add(key)
                            pairs.append((str(ip), str(bp), stem))
                    if include_degraded:
                        for dp in row.get('degraded_strip_paths') or []:
                            if not Path(dp).is_file():
                                continue
                            d_stem = Path(dp).stem
                            base_stem = d_stem.rsplit('_d', 1)[0]
                            bp = stem_to_bek.get(base_stem)
                            if not bp or any(s in base_stem for s in exclude):
                                continue
                            key = (str(dp), str(bp))
                            if key not in seen:
                                seen.add(key)
                                pairs.append((str(dp), str(bp), base_stem))
                elif single_bek and Path(single_bek).is_file():
                    for ip in page_paths:
                        if not Path(ip).is_file():
                            continue
                        p_stem = Path(ip).stem
                        stem = ('%s-%s' % (wid, p_stem)) if wid else p_stem
                        if not stem.lower().startswith('hw-'):
                            stem = 'hw-' + stem
                        if any(s in stem for s in exclude):
                            continue
                        key = (str(ip), str(single_bek))
                        if key not in seen:
                            seen.add(key)
                            pairs.append((str(ip), str(single_bek), stem))
        pairs.sort(key=lambda x: (x[2], x[0]))
        return pairs, hidx

    if (root / 'renders').is_dir() and (root / 'targets').is_dir():
        return (
            load_real_hw_pairs(str(root), exclude=exclude,
                               include_degraded=include_degraded, repeat=1),
            load_real_hw_hybrid_index(str(root)),
        )
    return load_extra_data_dir(
        str(root), exclude=exclude, include_degraded=include_degraded)


def load_rism_incipit_pairs(rism_dir, exclude=(), include_degraded=True):
    """Load semi-supervised RISM opening-staff incipit strips.

    Tracked separately from full-work labeled HW (``--real-hw-corpus``).
    Supports flat, ``renders/``+``targets/``, or ``incipit_strips/`` layouts.
    Returns ``(pairs, hybrid_index)``.
    """
    root = Path(rism_dir)
    if not root.is_dir():
        raise FileNotFoundError('rism-incipit-dir %s does not exist' % root)
    if (root / 'incipit_strips').is_dir() and not (
            (root / 'renders').is_dir() and (root / 'targets').is_dir()):
        sub = root / 'incipit_strips'
        if any(sub.glob('*-bekern.txt')) or (sub / 'targets').is_dir():
            root = sub
    return load_extra_data_dir(
        str(root), exclude=exclude, include_degraded=include_degraded)


def _pick_width_matched_pair(pool_pairs, pool_sorted, target_w, rng,
                             width_window=16):
    """Pick a pair from ``pool_pairs`` whose width is closest to ``target_w``.

    When ``target_w`` and ``pool_sorted = (sorted_widths, sorted_pairs)`` are
    available, restricts selection to a local window around
    ``bisect_left(sorted_widths, target_w)`` (preferring candidates within
    ``max(128, 25% of target_w)`` of ``target_w``, or within ``best_dist + 64``
    if none fall inside the tolerance). Falls back to uniform random selection
    from ``pool_pairs`` when width metadata is unavailable.
    """
    n = len(pool_pairs)
    if target_w is None or pool_sorted is None:
        return pool_pairs[rng.randrange(n)]
    widths, pairs_s = pool_sorted
    ns = len(pairs_s)
    if ns == 0:
        return pool_pairs[rng.randrange(n)]
    pos = bisect.bisect_left(widths, target_w)
    k = min(ns, max(1, int(width_window)))
    lo = max(0, pos - k // 2)
    hi = min(ns, lo + k)
    lo = max(0, hi - k)
    tol = max(128, int(round(0.25 * target_w)))
    cands = [i for i in range(lo, hi) if abs(widths[i] - target_w) <= tol]
    if not cands:
        best_d = None
        for idx in (pos - 1, pos):
            if 0 <= idx < ns:
                d = abs(widths[idx] - target_w)
                if best_d is None or d < best_d:
                    best_d = d
        cands = [i for i in range(lo, hi)
                 if abs(widths[i] - target_w) <= best_d + 64]
    if not cands:
        return pairs_s[min(pos, ns - 1)]
    return pairs_s[cands[rng.randrange(len(cands))]]


def _build_sorted_width_pool(pool_pairs, width_by_img):
    """Build ``(sorted_widths, sorted_pairs)`` from ``width_by_img`` if complete."""
    if not pool_pairs or not width_by_img:
        return None
    ws = []
    for p in pool_pairs:
        w = width_by_img.get(p[0])
        if w is None:
            return None
        ws.append(int(w))
    order = sorted(range(len(pool_pairs)), key=lambda i: ws[i])
    return ([ws[i] for i in order], [pool_pairs[i] for i in order])


def mix_supervised_batch(base_batch, step, seed, hw_pairs=(),
                         hw_mix_ratio=0.0, rism_pairs=(),
                         rism_mix_ratio=0.0,
                         width_by_img=None,
                         hw_sorted=None,
                         rism_sorted=None,
                         width_window=16):
    """Deterministically mix labeled HW and/or RISM incipit items into a batch.

    Per-step RNG stream ``f'{seed}:{step}:hw_mix'`` is independent of run
    history so ``--resume`` stays bit-exact. When ``width_by_img`` (and
    optional pre-sorted ``hw_sorted`` / ``rism_sorted``) is provided, picks
    candidates closest to ``base_batch``'s median width so a single wide
    HW/RISM strip never pads a narrow bucketed batch 3x. Returns
    ``(mixed_batch, hw_indices, rism_indices)``.
    """
    hw_r = max(0.0, float(hw_mix_ratio or 0.0))
    rism_r = max(0.0, float(rism_mix_ratio or 0.0))
    if (not hw_pairs or hw_r <= 0.0) and (not rism_pairs or rism_r <= 0.0):
        return base_batch, frozenset(), frozenset()
    rng = random.Random('%d:%d:hw_mix' % (int(seed), int(step)))
    mixed = list(base_batch)
    hw_idx = set()
    rism_idx = set()
    n_hw = len(hw_pairs)
    n_rism = len(rism_pairs)
    eff_hw = hw_r if n_hw > 0 else 0.0
    eff_rism = rism_r if n_rism > 0 else 0.0

    target_w = None
    if width_by_img is not None:
        bws = [width_by_img[p[0]] for p in base_batch if p[0] in width_by_img]
        if bws:
            bws.sort()
            target_w = int(bws[len(bws) // 2])
    if target_w is not None:
        if eff_hw > 0.0 and hw_sorted is None:
            hw_sorted = _build_sorted_width_pool(hw_pairs, width_by_img)
        if eff_rism > 0.0 and rism_sorted is None:
            rism_sorted = _build_sorted_width_pool(rism_pairs, width_by_img)

    for i in range(len(mixed)):
        u = rng.random()
        if eff_hw > 0.0 and u < eff_hw:
            mixed[i] = _pick_width_matched_pair(
                hw_pairs, hw_sorted, target_w, rng, width_window=width_window)
            hw_idx.add(i)
        elif eff_rism > 0.0 and u < eff_hw + eff_rism:
            mixed[i] = _pick_width_matched_pair(
                rism_pairs, rism_sorted, target_w, rng,
                width_window=width_window)
            rism_idx.add(i)
    return mixed, frozenset(hw_idx), frozenset(rism_idx)


def apply_epoch_aug_with_hw_pure(
    rgb_d, batch, seed, epoch, step,
    aug_p=0.0, clean_p=0.0, geom_strength=0.15,
    optics_profile='default', hw_indices=frozenset(),
    hw_pure_frac=0.35, gray_d=None, staff_d=None,
    aug_fn=None,
):
    """Apply ``epoch_aug`` while keeping ``hw_pure_frac`` of HW items pristine.

    For items in ``hw_indices``:
      - with probability ``hw_pure_frac`` (deterministic in
        ``(seed, epoch, step, i, basename)``), ``epoch_aug`` is SKIPPED so
        the model sees pure unaugmented real handwriting scans;
      - with probability ``1 - hw_pure_frac``, ``epoch_aug`` IS applied
        (``aug_p=1.0, clean_p=1.0, clean_anchor_frac=0.0``).
    Non-HW items use the standard ``(aug_p, clean_p)`` path unchanged.
    When ``hw_indices`` is empty, delegates directly to ``aug_fn`` for
    bitwise-identical legacy behavior.

    Returns ``(rgb_out, gray_out, staff_out, stats)`` where ``stats`` is
    ``{'hw_pure': int, 'hw_aug': int}``.
    """
    fn = aug_fn if aug_fn is not None else epoch_aug.augment_standardized_batch
    if not hw_indices:
        if aug_p <= 0.0 and clean_p <= 0.0:
            return rgb_d, gray_d, staff_d, {'hw_pure': 0, 'hw_aug': 0}
        if staff_d is not None:
            rgb_o, gray_o, staff_o = fn(
                rgb_d, batch, seed, epoch,
                aug_p=aug_p, clean_p=clean_p,
                grayb=gray_d, staffb=staff_d,
                geom_strength=geom_strength,
                optics_profile=optics_profile)
            return rgb_o, gray_o, staff_o, {'hw_pure': 0, 'hw_aug': 0}
        rgb_o, gray_o = fn(
            rgb_d, batch, seed, epoch,
            aug_p=aug_p, clean_p=clean_p,
            grayb=gray_d,
            geom_strength=geom_strength,
            optics_profile=optics_profile)
        return rgb_o, gray_o, None, {'hw_pure': 0, 'hw_aug': 0}

    p_pure = max(0.0, min(1.0, float(hw_pure_frac)))
    hw_pure_idx = []
    hw_aug_idx = []
    non_hw_idx = []
    for i, pair in enumerate(batch):
        if i in hw_indices:
            key = Path(pair[0]).name
            rng = random.Random(
                '%d:%d:%d:%d:%s:hw_pure'
                % (int(seed), int(epoch), int(step), i, key))
            if rng.random() < p_pure:
                hw_pure_idx.append(i)
            else:
                hw_aug_idx.append(i)
        else:
            non_hw_idx.append(i)

    out_rgb = rgb_d.clone()
    out_gray = gray_d.clone() if gray_d is not None else None
    out_staff = staff_d.clone() if staff_d is not None else None
    dev = rgb_d.device

    def _run_subset(idxs, sub_aug_p, sub_clean_p, extra_kw=None):
        if not idxs or (sub_aug_p <= 0.0 and sub_clean_p <= 0.0):
            return
        idx_t = torch.tensor(idxs, device=dev, dtype=torch.long)
        sub_rgb = out_rgb.index_select(0, idx_t)
        sub_gray = (out_gray.index_select(0, idx_t)
                    if out_gray is not None else None)
        sub_staff = (out_staff.index_select(0, idx_t)
                     if out_staff is not None else None)
        sub_batch = [batch[j] for j in idxs]
        kw = dict(
            aug_p=sub_aug_p, clean_p=sub_clean_p,
            grayb=sub_gray, geom_strength=geom_strength,
            optics_profile=optics_profile)
        if extra_kw:
            kw.update(extra_kw)
        if sub_staff is not None:
            kw['staffb'] = sub_staff
            r_o, g_o, s_o = fn(sub_rgb, sub_batch, seed, epoch, **kw)
            out_rgb.index_copy_(0, idx_t, r_o)
            if out_gray is not None and g_o is not None:
                out_gray.index_copy_(0, idx_t, g_o)
            if out_staff is not None and s_o is not None:
                out_staff.index_copy_(0, idx_t, s_o)
        else:
            r_o, g_o = fn(sub_rgb, sub_batch, seed, epoch, **kw)
            out_rgb.index_copy_(0, idx_t, r_o)
            if out_gray is not None and g_o is not None:
                out_gray.index_copy_(0, idx_t, g_o)

    _run_subset(non_hw_idx, aug_p, clean_p)
    _run_subset(hw_aug_idx, 1.0, 1.0, extra_kw={'clean_anchor_frac': 0.0})
    return out_rgb, out_gray, out_staff, {
        'hw_pure': len(hw_pure_idx),
        'hw_aug': len(hw_aug_idx),
    }


_IMAGE_EXTS = frozenset(('.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff'))


def list_unlabeled_images(dirs_raw):
    """Collect sorted image paths across one or more comma-separated dirs."""
    dirs = parse_extra_data_dirs(dirs_raw)
    seen = set()
    out = []
    for d in dirs:
        root = Path(d)
        if not root.is_dir():
            continue
        for p in sorted(root.rglob('*')):
            if not p.is_file():
                continue
            if p.name.startswith('.'):
                continue
            if p.suffix.lower() not in _IMAGE_EXTS:
                continue
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                out.append(sp)
    return out


def unsup_round_due(step, every_steps):
    """True when an unsupervised round should fire after supervised ``step``."""
    return int(every_steps) > 0 and int(step) > 0 and (int(step) % int(every_steps) == 0)


def _lazy_import_mvd_module(name):
    """Lazy-import an optional sibling module in ``tmp-train/mvd``."""
    mvd_dir = str(Path(__file__).resolve().parent)
    if mvd_dir not in sys.path:
        sys.path.insert(0, mvd_dir)
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


def run_stage2_unsup_round(
    net, opt, scaler, strip_paths, step, round_steps=100,
    mode='both', weight=0.25, kl_beta=0.10, height=256,
    device='cpu', amp=False, amp_dtype=torch.float16,
    stride='8x16', seed=0, batch_size=2,
    unsup_stage2_mod=None, unsup_bar_math_mod=None,
):
    """Run one scheduled Stage-2 unsupervised round via lazy-imported modules.

    Supports ``mode`` in ``('ema_enc', 'bar_math', 'both')``. Gracefully skips
    if the target optional module (``unsup_stage2`` / ``unsup_bar_math``) is
    not present.
    """
    summary = {
        'step': int(step),
        'mode': str(mode),
        'round_steps': int(round_steps),
        'n_strips': len(strip_paths),
        'ran': [],
        'metrics': {},
    }
    if not strip_paths or int(round_steps) <= 0:
        return summary

    def _invoke(mod, mod_name, fn_names, extra_kw):
        if mod is None:
            print('[unsup] module %s not found; skipping at step %d'
                  % (mod_name, step), flush=True)
            return None
        for fn_name in fn_names:
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn(**extra_kw)
        print('[unsup] module %s has none of %s; skipping at step %d'
              % (mod_name, fn_names, step), flush=True)
        return None

    common_kw = dict(
        net=net, opt=opt, scaler=scaler, strip_paths=list(strip_paths),
        step=int(step), steps=int(round_steps), weight=float(weight),
        kl_beta=float(kl_beta), height=int(height), device=device,
        amp=bool(amp), amp_dtype=amp_dtype, stride=stride,
        seed=int(seed) + int(step), batch_size=int(batch_size),
    )

    if mode in ('ema_enc', 'both'):
        m2 = (unsup_stage2_mod if unsup_stage2_mod is not None
              else _lazy_import_mvd_module('unsup_stage2'))
        res = _invoke(m2, 'unsup_stage2',
                      ('run_unsup_round', 'run_stage2_round', 'unsup_step'),
                      common_kw)
        if res is not None:
            summary['ran'].append('ema_enc')
            summary['metrics']['ema_enc'] = res

    if mode in ('bar_math', 'both'):
        mb = (unsup_bar_math_mod if unsup_bar_math_mod is not None
              else _lazy_import_mvd_module('unsup_bar_math'))
        res = _invoke(mb, 'unsup_bar_math',
                      ('run_bar_math_round', 'run_unsup_round', 'bar_math_step'),
                      common_kw)
        if res is not None:
            summary['ran'].append('bar_math')
            summary['metrics']['bar_math'] = res

    return summary


def filter_pages(pairs, substrings=None, exact_file=None):
    """Select training pairs by page stem.

    ``substrings``: keep pairs whose stem contains any of them (legacy,
    can over-match: 'ds0318-1' also hits 'ds0318-10'). ``exact_file``:
    path to a stem-per-line list, exact match only (curriculum bands).
    When both are given the result is the union (order: substrings
    first, then exact-only additions). Neither given returns pairs.
    """
    if not substrings and not exact_file:
        return pairs
    sel = []
    if substrings:
        sel += [p for p in pairs
                if any(s in p[2] for s in substrings)]
    if exact_file:
        want = {ln.strip() for ln in Path(exact_file).read_text(
            encoding='utf-8').split() if ln.strip()}
        seen = {p[2] for p in sel}
        sel += [p for p in pairs if p[2] in want and p[2] not in seen]
    return sel


META_KEYS = ('vocab', 'pages', 'height', 'lr', 'seed', 'exclude',
             'shuffle', 'no_amp', 'accum', 'init_from', 'trunk_lr_mult',
             'warmup', 'cosine', 'sample_p', 'pages_file', 'batch_size',
             'pretrained', 'no_degraded', 'tok_weight', 'tok_dropout',
             'extra_chans', 'hybrid_cache', 'heat_pt', 'no_heatnet',
             'stride', 'foveal_tap', 'aux_weight', 'aux_every',
             'heat_dropout', 'coord_weight', 'coord_every', 'simshots',
             'epoch_aug_p', 'epoch_aug_clean_p', 'epoch_aug_geom',
             'epoch_redraw_dir', 'optics_profile', 'jsd_weight',
             'jsd_every', 'bf16', 'channels_last', 'width_bucket',
             'prefetch', 'real_hw_dir', 'real_hw_repeat',
             's18_weight', 'comer', 'gdcm_weight', 'extra_data_dir',
             'max_tokens', 'real_hw_corpus', 'hw_mix_ratio', 'hw_pure_frac',
             'rism_incipit_dir', 'rism_mix_ratio', 'unsup_strips_dir',
             'unsup_every_steps', 'unsup_round_steps', 'unsup_mode',
             'unsup_weight', 'unsup_kl_beta')

# Values new flags had implicitly before they existed. An old
# meta.json missing one of these keys resumes iff the current run uses
# the legacy value; any non-default use is a genuine mismatch and is
# refused. Re-apply contract for parked work: landing a new flag MUST
# append it to BOTH RESUME_KEYS and LEGACY_DEFAULTS (with its legacy
# value), or old out dirs brick on --resume.
LEGACY_DEFAULTS = {
    'heat_cache': False,
    'compile': False,
    'grad_checkpoint': False,
    'epoch_aug_p': 0.0,
    'epoch_aug_clean_p': 0.0,
    'epoch_aug_geom': 0.0,
    'epoch_redraw_dir': None,
    'bf16': False,
    'channels_last': False,
    'width_bucket': False,
    'prefetch': 0,
    'real_hw_dir': None,
    'real_hw_repeat': 1,
    'no_heatnet': False,
    'foveal_tap': False,
    'optics_profile': 'default',
    'jsd_weight': 0.0,
    'jsd_every': 4,
    's18_weight': 0.0,
    'comer': False,
    'gdcm_weight': 0.0,
    'extra_data_dir': None,
    'max_tokens': 768,
    'real_hw_corpus': None,
    'hw_mix_ratio': 0.0,
    'hw_pure_frac': 0.35,
    'rism_incipit_dir': None,
    'rism_mix_ratio': 0.0,
    'unsup_strips_dir': None,
    'unsup_every_steps': 0,
    'unsup_round_steps': 100,
    'unsup_mode': 'both',
    'unsup_weight': 0.25,
    'unsup_kl_beta': 0.10,
}

# Exact fingerprint enforced on --resume (T0.2): every key here must
# match meta.json, else the run is refused with a pointer to
# --init-from in a fresh --out dir.
RESUME_KEYS = ('vocab', 'pages', 'height', 'lr', 'seed', 'exclude',
               'shuffle', 'no_amp', 'accum', 'trunk_lr_mult', 'warmup',
               'cosine', 'sample_p', 'batch_size', 'pretrained',
               'no_degraded', 'tok_weight', 'tok_dropout', 'extra_chans',
               'hybrid_cache', 'heat_pt', 'no_heatnet', 'stride',
               'foveal_tap', 'aux_weight', 'aux_every', 'heat_dropout',
               'coord_weight', 'coord_every', 'margin_weight',
               'margin_siblings', 'margin', 'margin_every',
               'mwer_weight', 'mwer_beams', 'mwer_every',
               'mwer_maxlen', 'action_weight', 'action_every',
               'specaug_p', 'specaug_slope', 'handink', 'simshots',
               'epoch_aug_p', 'epoch_aug_clean_p', 'epoch_aug_geom',
               'epoch_redraw_dir', 'optics_profile', 'jsd_weight',
               'jsd_every', 'bf16', 'channels_last', 'width_bucket',
               'prefetch', 'real_hw_dir', 'real_hw_repeat',
               's18_weight', 'comer', 'gdcm_weight', 'extra_data_dir',
               'max_tokens', 'real_hw_corpus', 'hw_mix_ratio', 'hw_pure_frac',
               'rism_incipit_dir', 'rism_mix_ratio', 'unsup_strips_dir',
               'unsup_every_steps', 'unsup_round_steps', 'unsup_mode',
               'unsup_weight', 'unsup_kl_beta')

_LOCK_HELD = []


def acquire_lock(out):
    """Claim ``<out>/.lock`` as this run's single-writer token.

    Refuses with exit 1 when the file already exists. The lock holds
    our pid and is removed at clean exit; a killed run leaves it
    behind, so the refusal names the exact file to delete ONLY after
    confirming no training process is using this ``--out`` dir.
    """
    import atexit
    out = Path(out)
    lock = out / '.lock'
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        sys.exit('refusing: %s is locked by another run '
                 '(delete %s only if no training process is using it)'
                 % (out, lock))
    with os.fdopen(fd, 'w') as f:
        f.write('%d\n' % os.getpid())
    _LOCK_HELD.append(str(lock))

    def _release(path=str(lock)):
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass
    atexit.register(_release)


def _pid_alive(pid):
    """True when ``pid`` names a live process (or liveness unreadable).

    Windows-safe: uses ``OpenProcess`` on ``os.name == 'nt'`` (asking
    the OS instead of signalling), ``os.kill(pid, 0)`` on POSIX. A
    dead PID means the lock is stale (killed run) and may be removed;
    an unreadable PID conservatively counts as alive (refuse).
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return True
    if os.name == 'nt':
        try:
            import ctypes
            _kern = ctypes.windll.kernel32
            handle = _kern.OpenProcess(0x1000, False, pid)
            if handle:
                _kern.CloseHandle(handle)
                return True
            return False
        except Exception:
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    else:
        return True


def acquire_out_lock(out_dir):
    """Claim ``<out_dir>/.train.lock`` as this run's single-writer token.

    Creates the lock atomically (``O_CREAT | O_EXCL``) holding
    ``{"pid": <pid>}``. A second writer while the lock is live is
    refused via ``sys.exit``; a stale lock (dead PID, e.g. a killed
    run) is removed and recreated. Registers an ``atexit`` handler to
    unlink the lock on normal exit (including ``sys.exit`` refusals).
    """
    import atexit
    out_dir = Path(out_dir)
    lock_path = out_dir / '.train.lock'
    for _ in range(2):
        try:
            fd = os.open(str(lock_path),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                raw = lock_path.read_text(encoding='utf-8')
            except OSError:
                sys.exit(f"refusing: active writer lock {lock_path} "
                         f"locked by another run (unreadable lock file)")
            try:
                pid = (json.loads(raw).get('pid') if raw.strip()
                       .startswith('{') else int(raw.strip()))
            except (ValueError, AttributeError, TypeError):
                sys.exit(f"refusing: active writer lock {lock_path} "
                         f"locked by another run (unreadable lock file)")
            if _pid_alive(pid):
                sys.exit(f"refusing: active writer lock {lock_path} "
                         f"locked by another run (pid {pid} still alive)")
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, 'w') as f:
            f.write(json.dumps({"pid": os.getpid()}))
        _LOCK_HELD.append(str(lock_path))

        def _release(path=str(lock_path)):
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
        atexit.register(_release)
        return lock_path
    sys.exit(f"refusing: active writer lock {lock_path} "
             f"locked by another run")


def release_out_lock(out_dir):
    """Remove ``<out_dir>/.train.lock`` held by this process (tests)."""
    try:
        (Path(out_dir) / '.train.lock').unlink()
    except FileNotFoundError:
        pass


MAINTENANCE_EVERY = 500


# S13 4x8 VRAM wall: rectification can stretch an ordinary strip to
# ~5760 px (observed pole, train step 3860 — 5 kills). Typical
# rectified strips are ~1200-2600 px; at B=4 with width-bucketing,
# a W=3664 batch (step 1313) spills 16 GB VRAM into Windows WDDM and
# triggers a TDR driver reset ("CUDA driver error: device not ready").
# Cap at 3072 (skips only ~0.16% outlier strips). Skips are
# deterministic (same pairs -> same skips), so the fixed step->batch
# map stays resume-exact.
BATCH_WIDTH_CAP = 3072


def batch_too_wide(width):
    """True when a rectified batch width must be skipped, not trained."""
    return width > BATCH_WIDTH_CAP


def cuda_mem_stats(device):
    """(reserved_MB, allocated_MB) on CUDA, else (None, None).

    Observability for the S13 4x8 VRAM wall: allocator growth vs a
    genuine leak reads directly off the reserved/allocated gap.
    """
    if device != 'cuda' or not torch.cuda.is_available():
        return None, None
    free = torch.cuda.memory_reserved(device)
    used = torch.cuda.memory_allocated(device)
    return free / 1048576.0, used / 1048576.0


def maintenance_due(step):
    """True on the defrag+checkpoint cadence (every 500 steps)."""
    return step % MAINTENANCE_EVERY == 0


def save_checkpoint(out, step, net, opt, sched, updates, args):
    """Write ``ckpt-<step>.pt`` + ``opt-<step>.pt`` sidecar (atomic).

    Same payload as the eval-block save; factored so the maintenance
    cadence and the eval cadence share one schema.
    """
    atomic_torch_save(net.state_dict(), out / ('ckpt-%d.pt' % step))
    # Optimizer/scheduler sidecar: makes --resume continue the
    # schedule instead of restarting it. ckpt-*.pt stays
    # weights-only for inference loading.
    atomic_torch_save({'step': step, 'updates': updates[0],
                       'opt': opt.state_dict(),
                       'sched': (sched.state_dict()
                                 if sched is not None else None),
                       'hp': {'lr': args.lr,
                              'trunk_mult': args.trunk_lr_mult,
                              'warmup': args.warmup,
                              'accum': args.accum,
                              'cosine': args.cosine}},
                      out / ('opt-%d.pt' % step))


def build_meta(args, pairs):
    """Full config fingerprint recorded in ``<out>/meta.json``."""
    return {
        'vocab': len(data.VOCAB), 'pages': [p[2] for p in pairs],
        'height': args.height, 'lr': args.lr, 'seed': args.seed,
        'exclude': args.exclude, 'shuffle': args.shuffle,
        'no_amp': args.no_amp, 'accum': args.accum,
        'init_from': args.init_from, 'trunk_lr_mult': args.trunk_lr_mult,
        'warmup': args.warmup, 'cosine': args.cosine,
        'sample_p': args.sample_p, 'pages_file': args.pages_file,
        'batch_size': getattr(args, 'batch_size', 8),
        'pretrained': getattr(args, 'pretrained', False),
        'no_degraded': getattr(args, 'no_degraded', False),
        'tok_weight': getattr(args, 'tok_weight', 1.0),
        'tok_dropout': getattr(args, 'tok_dropout', 0.0),
        'margin_weight': getattr(args, 'margin_weight', 0.0),
        'margin_siblings': getattr(args, 'margin_siblings', 8),
        'margin': getattr(args, 'margin', 1.0),
        'margin_every': getattr(args, 'margin_every', 1),
        'mwer_weight': getattr(args, 'mwer_weight', 0.0),
        'mwer_beams': getattr(args, 'mwer_beams', 4),
        'mwer_every': getattr(args, 'mwer_every', 1),
        'mwer_maxlen': getattr(args, 'mwer_maxlen', 576),
        'action_weight': getattr(args, 'action_weight', 0.0),
        'action_every': getattr(args, 'action_every', 1),
        'specaug_p': getattr(args, 'specaug_p', 0.0),
        'specaug_slope': getattr(args, 'specaug_slope', 0.02),
        'extra_chans': getattr(args, 'extra_chans', 0),
        'stride': getattr(args, 'stride', '8x16'),
        'aux_weight': getattr(args, 'aux_weight', 0.0),
        'aux_every': getattr(args, 'aux_every', 1),
        'heat_dropout': getattr(args, 'heat_dropout', 0.0),
        'coord_weight': getattr(args, 'coord_weight', 0.0),
        'coord_every': getattr(args, 'coord_every', 1),
        'hybrid_cache': getattr(args, 'hybrid_cache', None),
        'heat_pt': getattr(args, 'heat_pt', None),
        'no_heatnet': getattr(args, 'no_heatnet', False),
        'foveal_tap': getattr(args, 'foveal_tap', False),
        'handink': getattr(args, 'handink', False),
        'simshots': getattr(args, 'simshots', False),
        'epoch_aug_p': getattr(args, 'epoch_aug_p', 0.0),
        'epoch_aug_clean_p': getattr(args, 'epoch_aug_clean_p', 0.0),
        'epoch_aug_geom': getattr(args, 'epoch_aug_geom', 0.0),
        'epoch_redraw_dir': getattr(args, 'epoch_redraw_dir', None),
        'optics_profile': getattr(args, 'optics_profile', 'default'),
        'jsd_weight': getattr(args, 'jsd_weight', 0.0),
        'jsd_every': getattr(args, 'jsd_every', 4),
        'bf16': getattr(args, 'bf16', False),
        'channels_last': getattr(args, 'channels_last', False),
        'width_bucket': getattr(args, 'width_bucket', False),
        'prefetch': getattr(args, 'prefetch', 0),
        'real_hw_dir': getattr(args, 'real_hw_dir', None),
        'real_hw_repeat': getattr(args, 'real_hw_repeat', 1),
        's18_weight': getattr(args, 's18_weight', 0.0),
        'comer': getattr(args, 'comer', False),
        'gdcm_weight': getattr(args, 'gdcm_weight', 0.0),
        'extra_data_dir': (
            parse_extra_data_dirs(getattr(args, 'extra_data_dir', None))
            or None),
        'max_tokens': getattr(args, 'max_tokens', 768),
        'real_hw_corpus': getattr(args, 'real_hw_corpus', None),
        'hw_mix_ratio': getattr(args, 'hw_mix_ratio', 0.0),
        'hw_pure_frac': getattr(args, 'hw_pure_frac', 0.35),
        'rism_incipit_dir': getattr(args, 'rism_incipit_dir', None),
        'rism_mix_ratio': getattr(args, 'rism_mix_ratio', 0.0),
        'unsup_strips_dir': (
            parse_extra_data_dirs(getattr(args, 'unsup_strips_dir', None))
            or None),
        'unsup_every_steps': getattr(args, 'unsup_every_steps', 0),
        'unsup_round_steps': getattr(args, 'unsup_round_steps', 100),
        'unsup_mode': getattr(args, 'unsup_mode', 'both'),
        'unsup_weight': getattr(args, 'unsup_weight', 0.25),
        'unsup_kl_beta': getattr(args, 'unsup_kl_beta', 0.10),
    }


STRUCT_WEIGHT_IDS = frozenset((data.TOKEN_TO_ID['bar'],))
CONTENT_IDS = tuple(sorted(
    data.TOKEN_TO_ID[tok] for tok in
    # data.py canonical content sets (private _STEPS etc.; no public
    # STEPS/OCTAVES/... aliases exist, so they are referenced here):
    set(data._STEPS) | set(data._OCTAVES) | set(data._ALTERS)
    | set(data._DURATIONS) | set(data._MARKS) | {'rest'}))


def tok_class(tok):
    """Token class for per-class TF accuracy: header / bar / body / syntax.

    Headers (clef/key/time) are prior-saturated and tracked separately
    from the gate; barlines are the structural target; syntax (voices,
    chords, octave digits) is excluded from the grounding gate; body is
    everything else (durations, steps, accidentals, dots, rests, ties,
    tuplets, bar-rest, breve).
    """
    if tok == 'bar':
        return 'bar'
    if tok.startswith(('clef-', 'key', 'time-')):
        return 'header'
    if tok.startswith('voice-') or tok == 'chord' \
            or (len(tok) == 1 and tok.isdigit()):
        return 'syntax'
    return 'body'


def mutate_modifier_siblings(ref_ids, n=8, rng=None, mod_ids=(),
                               vocab_size=0):
    """8 modifier-edited sibling sequences for spike-S3 margin loss.

    Splits ``ref_ids`` on barlines and edits 1-3 modifier tokens inside
    ONE randomly chosen inter-bar span (drop / insert / swap within
    ``mod_ids``); spans without modifiers only take inserts, and the
    pre-first-bar header run (clef/key/time prefix) is never selected.
    Barlines, pitches, durations and headers survive in every sibling.
    Deterministic in ``rng``; every sibling differs from ``ref_ids``.
    ``ref_ids`` must be unpadded (callers strip PAD first). Pure
    function (no torch needed by callers beyond id lists).
    """
    ref = list(ref_ids)
    rng = rng if rng is not None else random.Random()
    mods = sorted(m for m in mod_ids if 0 <= m < vocab_size)
    bar = data.TOKEN_TO_ID.get('bar')
    nv = len(data.VOCAB)
    toks = [data.VOCAB[i] if 0 <= i < nv else '' for i in ref]
    hi = 0
    while hi < len(toks) and (
            toks[hi].startswith(('clef-', 'key', 'time-'))):
        hi += 1
    bounds, start = [], hi
    for i, t in enumerate(ref):
        if i < hi:
            continue
        if t == bar:
            if i > start:
                bounds.append((start, i))
            start = i + 1
    if start < len(ref):
        bounds.append((start, len(ref)))
    if not bounds:
        bounds.append((hi, len(ref)))
    out = []
    for _ in range(max(0, n)):
        lo, end = bounds[rng.randrange(len(bounds))]
        cur = list(ref)
        n_edits = rng.randint(1, 3)
        for _e in range(n_edits):
            kind = rng.choice(('drop', 'insert', 'swap'))
            live = [i for i in range(lo, end) if cur[i] in mod_ids]
            if kind == 'drop' and live:
                del cur[rng.choice(live)]
                end -= 1
            elif kind == 'swap' and live:
                i = rng.choice(live)
                old = cur[i]
                alt = rng.choice(mods)
                if alt == old and len(mods) > 1:
                    alt = rng.choice([m for m in mods if m != old])
                cur[i] = alt
            else:
                at = rng.randint(lo, end)
                cur.insert(at, rng.choice(mods))
                end += 1
        if cur == ref:
            at = rng.randint(lo, end)
            cur.insert(at, rng.choice(mods))
        out.append(cur)
    return out


def sibling_margin_loss(ref_lp, sib_lps, margin=1.0):
    """Mean ``relu(margin + sib - ref)`` over sibling log-probs.

    Zero when the reference beats every sibling by ``margin``;
    positive otherwise. Empty sibling sets score 0. Pure function.
    """
    if sib_lps.numel() == 0:
        return ref_lp.new_zeros(())
    return torch.relu(margin + sib_lps - ref_lp).mean()


def sequence_logprob(net, mem, ids, mem_mask=None, pad_id=None):
    """Teacher-forced total log-prob of one id sequence (spike S3).

    Scores ``ids`` (unpadded) against precomputed ``mem`` via
    :meth:`model.MvdNet.decode_logits` (no re-encode). PAD positions
    never occur in ``ids`` by contract; ``pad_id`` is accepted for
    call-site symmetry and ignored.
    """
    dev = mem.device
    t = torch.tensor(list(ids), dtype=torch.long, device=dev)
    bos = torch.full((1, 1), net.bos_id, dtype=torch.long, device=dev)
    logits = net.decode_logits(
        mem, torch.cat([bos, t.unsqueeze(0)[:, :-1]], dim=1), mem_mask)
    lp = torch.log_softmax(logits[0], dim=-1)
    return lp[torch.arange(t.numel(), device=dev), t].sum()


def weighted_risk(ref_ids, hyp_ids, bar_id, eos_id, micro_ids):
    """Cost-weighted edit risk of hyp vs ref (spike S4).

    difflib alignment, costs on the ref side: dropped barlines 3,
    dropped micro-glyphs (modifier ids) 2, anything else 1;
    substitutions cost ref-side class cost + 1 per hyp token;
    insertions cost 2 (micro) / 1 (else). A hyp EOS landing before
    the ref's last content token adds a premature-EOS 3 (the S1
    hip-short failure). Identical streams score 0. Pure function.
    """
    import difflib as _dl
    ref, hyp = list(ref_ids), list(hyp_ids)

    def _del_cost(t):
        if t == bar_id:
            return 3.0
        if t in micro_ids:
            return 2.0
        return 1.0

    def _ins_cost(t):
        return 2.0 if t in micro_ids else 1.0

    total = 0.0
    sm = _dl.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        if tag == 'delete':
            total += sum(_del_cost(t) for t in ref[i1:i2])
        elif tag == 'insert':
            total += sum(_ins_cost(t) for t in hyp[j1:j2])
        else:  # replace
            total += sum(_del_cost(t) for t in ref[i1:i2])
            total += float(j2 - j1)
    if eos_id in hyp:
        first_eos = hyp.index(eos_id)
        if first_eos < len(ref) - 1:
            total += 3.0
    return total


def beam_hyp(row, eos_id, nvocab):
    """One beam row -> hypothesis ids for MWER risk (spike S4).

    Mirrors ``measure_decode._ids_to_tokens`` (cut at specials) but
    RETAINS a cutting EOS: premature-EOS detection needs it, while
    trailing PAD/fill carries no risk signal. No-EOS beams keep
    content until the first special.
    """
    seq = []
    for t in row:
        t = int(t)
        if t == eos_id:
            seq.append(t)
            break
        if t < 0 or t >= nvocab:
            break
        seq.append(t)
    return seq


def mwer_loss(beam_lps, risks):
    """Minimum-word-error-rate loss with mean-risk baseline (spike S4).

    ``beam_lps`` are total log-probs (one per beam, requires grad);
    ``risks`` are non-differentiable costs. Renormalizes the beam
    posterior via softmax and charges ``sum p_i (r_i - mean(r))``:
    zero when all beams tie, negative pressure on above-average-risk
    beams. Single beams and empty inputs score 0. Pure function.
    """
    if beam_lps.numel() == 0:
        return beam_lps.new_zeros(())
    if beam_lps.numel() == 1:
        return beam_lps.new_zeros(())
    logp = torch.log_softmax(beam_lps, dim=0)
    probs = torch.exp(logp)
    base = risks.mean()
    return (probs * (risks - base)).sum()


def keystroke_cost_matrix(vocab_size, mod_ids):
    """Token substitution cost matrix (spike S7).

    ``C[pred, ref]``: 0 on the diagonal; 3 when a barline is missed
    or hallucinated (bar conservation — a wrong bar count breaks
    every downstream measure); 2 for modifier-involved mismatches
    (dots, ties, articulations); 2 for EOS-involved mistiming;
    1 otherwise. PAD rows/cols are 0 (positions masked by callers,
    never scored). Pure function of ids, built once per run.
    """
    nv = len(data.VOCAB)
    bar = data.VOCAB.index('bar')
    mods = [m for m in mod_ids if 0 <= m < nv]
    c = torch.ones(vocab_size, vocab_size)
    c.fill_diagonal_(0.0)
    c[:, bar] = 3.0
    c[bar, :] = 3.0
    c[bar, bar] = 0.0
    for m in mods:
        c[:, m] = 2.0
        c[m, :] = 2.0
        c[m, m] = 0.0
    eos = data.EOS_ID
    c[:, eos] = 2.0
    c[eos, :] = 2.0
    c[eos, eos] = 0.0
    c[:, data.PAD_ID] = 0.0
    c[data.PAD_ID, :] = 0.0
    return c


def expected_keystroke_risk(logits, tgt, cost):
    """Mean token-level expected substitution cost (spike S7).

    ``risk = mean over non-PAD positions of sum_v p(v) * C[v, ref]``:
    the minimum-Bayes-risk token objective. One forward's logits
    suffice (no beams, unlike S4 MWER); the bar-conservation
    pressure lives in the cost matrix. PAD positions score 0.
    ``tgt`` is ``[B, T]`` ids. Pure function (differentiable in
    ``logits``).
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(logp).to(logits.dtype)
    per_pos = (probs * cost.t()[tgt.clamp_min(0)]).sum(-1)
    mask = (tgt != data.PAD_ID).to(per_pos.dtype)
    return (per_pos * mask).sum() / mask.sum().clamp_min(1.0)


def sample_specaug_params(rng, slope_max=0.02):
    """Sample one latent SpecAugment spec (spike S8).

    ``rng`` is a ``random.Random`` on the ``'seed:step:specaug'``
    stream (deterministic across runs, no run-history input).
    Returns ``{'slope', 'bands', 'slices'}`` for
    :func:`model.specaugment_latent`: uniform slope in
    ``[-slope_max, slope_max]`` px/column, 0-1 staff-band row ranges
    (each 1-9% of height), 0-2 micro-time column ranges (each 1-5%
    of width). Pure function of the rng state.
    """
    slope = rng.uniform(-slope_max, slope_max)
    bands = []
    for _ in range(rng.randrange(2)):
        f0 = rng.random() * 0.9
        bands.append((f0, min(1.0, f0 + 0.01 + rng.random() * 0.08)))
    slices = []
    for _ in range(rng.randrange(3)):
        f0 = rng.random() * 0.9
        slices.append((f0, min(1.0, f0 + 0.01 + rng.random() * 0.04)))
    return {'slope': slope, 'bands': tuple(bands),
            'slices': tuple(slices)}


def maybe_specaug(rng, p, slope_max=0.02):
    """None with probability ``1 - p``, else a sampled spec dict.

    The apply/skip draw comes first on the same rng stream, so the
    per-step outcome is still a pure function of ``(seed, step)``.
    ``p <= 0`` never augments (legacy); ``p >= 1`` always does.
    """
    if rng.random() >= p:
        return None
    return sample_specaug_params(rng, slope_max)


def tf_top1_by_class(logits, tgt):
    """Top-1 accuracy counts per token class for one strip.

    ``logits`` is a [T, V] tensor, ``tgt`` an iterable of ids aligned to
    its rows. PAD and EOS positions are skipped. Returns
    ``{cls: [ok, n]}`` with classes header/bar/body/syntax.
    """
    pred = logits.argmax(-1).tolist()
    ids = tgt if isinstance(tgt, list) else list(tgt)
    out = {'header': [0, 0], 'bar': [0, 0],
           'body': [0, 0], 'syntax': [0, 0]}
    for i, t in enumerate(ids):
        if t in (data.PAD_ID, data.EOS_ID):
            continue
        c = tok_class(data.decode([t])[0])
        out[c][1] += 1
        out[c][0] += (int(pred[i]) == t)
    return out


def check_resume(out, meta):
    """None when ``--resume`` may proceed, else a refusal message.

    Compares the FULL config fingerprint (every key in RESUME_KEYS),
    not just the pair list: any changed field means the checkpoints on
    disk belong to a different run, so resuming would silently corrupt
    the schedule and sidecars.
    """
    mf = Path(out) / 'meta.json'
    if not mf.exists():
        return None
    old = json.loads(mf.read_text(encoding='utf-8'))
    for k in RESUME_KEYS:
        if k not in old:
            if k in LEGACY_DEFAULTS and meta.get(k) == LEGACY_DEFAULTS[k]:
                continue
            return (f"refusing --resume: config key {k!r} missing in "
                    f"{mf} (old run predates it) and new value "
                    f"{meta.get(k)!r} is not the legacy default; "
                    f"use --init-from in a fresh --out dir")
        if old[k] != meta.get(k):
            return (f"refusing --resume: config mismatch on key {k!r}: "
                    f"old={old[k]} vs new={meta.get(k)}; "
                    f"use --init-from in a fresh --out dir")
    return None


def atomic_write_text(path, text):
    """Write text via tmp+rename+fsync so a kill leaves no half-file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(obj, path):
    """torch.save via tmp+rename+fsync so a kill leaves no half-file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_image(path, height):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    nw = max(1, round(w * height / h))
    img = img.resize((nw, height), Image.LANCZOS)
    px = torch.from_numpy(
        __import__('numpy').asarray(img).copy()).float().div_(255.0)
    # ImageNet standardization (statistics only, no pretrained weights:
    # torchvision weights carry ImageNet dataset terms, so the trunk
    # trains from scratch). The 93%-white pages sit at +1.9 sigma
    # background with negative strokes — the trunk's native diet —
    # instead of a [0,1] wash that starves early trunk gradients while
    # the decoder LM learns unconditioned priors (epoch-2 collapse:
    # identical mem for all pages, identical greedy hyps).
    return standardize_rgb(px.permute(2, 0, 1))


def standardize_rgb(chw):
    """``[3, H, W]`` float ``/ 255`` tensor -> ImageNet-standardized.

    Factored from :func:`load_image` so the hybrid collate standardizes
    sidecar RGB identically (same constants, same in-place ops).
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return chw.sub_(mean).div_(std)


# Standardized ImageNet value of pure-white paper, the fill for width
# padding in collate(): (1 - mean) / std per channel, i.e. the +2.2 to
# +2.6 sigma background the trunk reads as blank paper (real page paper
# sits nearby at ~+1.9 sigma).
WHITE = torch.tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224,
                      (1 - 0.406) / 0.225]).view(3, 1, 1)


def sidecar_path_for(img_path):
    """``<stem>.hybrid.pt`` alongside the image (mirrors the cache builder)."""
    p = Path(img_path)
    return p.parent / (p.stem + '.hybrid.pt')


def load_hybrid_index(jsonl_path):
    """``hybrid_staff.jsonl`` -> ``{img basename: row}`` for table rows.

    Provenance log for ``--hybrid-cache`` runs: the training filter
    itself keys on sidecar existence (below), while the index records
    which strips the cache builder validated (S, sp, kind, source).
    """
    index = {}
    with open(jsonl_path, encoding='utf-8') as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if not r.get('skip') and isinstance(r.get('table'), dict):
                index[r['img']] = r
    return index


def filter_hybrid_pairs(pairs, hybrid_index=None, allow_on_the_fly=False):
    """Keep pairs whose image has a sidecar (``.hybrid.npz`` or ``.hybrid.pt``).

    Strips without validated staff tables (marker-transfer skips, ~43%
    of degraded; clean misses are ~0) are dropped from training —
    never zero-filled: the model must learn extras as truth, not as
    sometimes-blank noise. When ``allow_on_the_fly=True`` and
    ``hybrid_index`` is supplied, pairs present in ``hybrid_index`` are
    also kept and rectified in memory when no sidecar file exists on
    disk. Returns ``(kept, n_drop_clean, n_drop_degraded)``; ``kept``
    preserves input order (deterministic).
    """
    kept = []
    dc = dd = 0
    for p in pairs:
        has_file = compact_sidecar.resolve_sidecar_path(p[0]) is not None
        has_otf = False
        if allow_on_the_fly and hybrid_index is not None:
            row = hybrid_index.get(Path(p[0]).name)
            if isinstance(row, dict):
                tbl = row.get('table')
                staves = tbl.get('staves') if isinstance(tbl, dict) else None
                n_s = len(staves) if isinstance(staves, list) else int(row.get('S', 0) or 0)
                has_otf = (1 <= n_s <= 4)
        if has_file or has_otf:
            kept.append(p)
        elif '_d' in Path(p[0]).stem.rsplit('-', 1)[-1]:
            dd += 1
        else:
            dc += 1
    return kept, dc, dd


def load_sidecar(img_path, height, hybrid_index=None):
    """``(rgb, gray, staff, S)`` from compact ``.hybrid.npz``, legacy
    ``.hybrid.pt``, or on-the-fly rectification via ``hybrid_index``.

    ``rgb`` is ImageNet-standardized ``[3, H, W]`` (identical to
    :func:`load_image` output); ``gray`` is ``[1, H, W]`` ``/ 255``
    (HeatNet input convention); ``staff`` is ``[1, H, W]`` ``{0, 1}``
    (rectified staff-grid channel); ``S`` is the strip's staff count
    (``1..4``) for inactive-slot masking in :func:`collate_hybrid`.
    A missing or out-of-range ``S`` falls back to ``4`` (mask nothing),
    preserving legacy behavior. Raises when the sidecar height
    disagrees with the training ``--height``.
    """
    cpath = compact_sidecar.compact_sidecar_path_for(img_path)
    if cpath.exists():
        return compact_sidecar.load_compact_sidecar(
            cpath, height, standardize_fn=standardize_rgb)
    lpath = sidecar_path_for(img_path)
    if lpath.exists():
        blob = torch.load(str(lpath), map_location='cpu', weights_only=True)
        if int(blob['height']) != int(height):
            raise ValueError('sidecar %s height %s != train height %s'
                             % (lpath, blob['height'], height))
        rgb = standardize_rgb(blob['rgb'].float().div_(255.0))
        gray = blob['gray'].float().div_(255.0).unsqueeze(0)
        staff = blob['staff'].float()
        if staff.dim() == 2:
            staff = staff.unsqueeze(0)
        try:
            nslot = int(blob.get('S', 4))
        except (TypeError, ValueError):
            nslot = 4
        if not 1 <= nslot <= 4:
            nslot = 4
        return rgb, gray, staff, nslot
    if hybrid_index is not None and Path(img_path).name in hybrid_index:
        return compact_sidecar.rectify_on_the_fly(
            img_path, hybrid_index[Path(img_path).name], height=height,
            standardize_fn=standardize_rgb)
    raise FileNotFoundError(str(lpath))


def encode_targets(batch, max_tokens=0):
    """Batch bekern targets with EOS + PAD (shared by both collates)."""
    ids = []
    for _, b, _ in batch:
        seq = data.encode(data.load_bekern(b)) + [data.EOS_ID]
        if max_tokens and max_tokens > 0 and len(seq) > max_tokens:
            seq = seq[:max_tokens - 1] + [data.EOS_ID]
        ids.append(seq)
    n = max(len(x) for x in ids)
    tgt = torch.full((len(ids), n), data.PAD_ID, dtype=torch.long)
    for i, x in enumerate(ids):
        tgt[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return tgt


def collate_hybrid(batch, height, stride='8x16', hybrid_index=None,
                   max_tokens=0):
    """Batch rectified sidecar strips: pad widths, pad targets, mask memory.

    Mirrors :func:`collate` exactly (same ``W_max``-to-16 padding, same
    memory key mask, same targets) but loads RGB/gray/staff from the
    ``.hybrid.npz`` / ``.hybrid.pt`` sidecars (or on-the-fly when
    ``hybrid_index`` is supplied) instead of ``load_image``. Gray pads with
    white (1.0, the HeatNet paper value); staff pads with 0. Heat is
    NOT computed here (single CUDA forward per step in the loop via
    :func:`assemble_hybrid_batch`). Inactive staff slots (memory rows
    ``y_mem >= 8 * S`` for an ``S``-staff strip at height 256 — one
    64-px slot is 8 memory rows) are masked alongside width padding:
    they hold canonical white paper, not signal, and unmasked they let
    slot-count leak in as a strip-ID fingerprint (v2.2 collapse).
    Returns ``(rgb, tgt, key_mask, gray, staff)`` (CPU tensors);
    ``key_mask`` is None when nothing is masked (all ``S == 4`` and no
    width padding), exactly like the legacy path. ``stride='4x8'``
    (S13) sizes the memory grid for the 4x8 trunk (``H // 4`` rows,
    ``(w // 4) // 2`` cols; 16 slot rows at height 256). Default keeps
    the exact legacy grid.
    """
    if stride not in ('8x16', '4x8'):
        raise ValueError('stride must be 8x16 or 4x8, got %r' % stride)
    loaded = [load_sidecar(p, height, hybrid_index=hybrid_index)
              for p, _, _ in batch]
    widths = [rgb.shape[2] for rgb, _, _, _ in loaded]
    nslots = [s for _, _, _, s in loaded]
    w_max = max(widths)
    w_pad = (w_max + 15) // 16 * 16
    h_mem = height // (4 if stride == '4x8' else 8)
    w_mem = _mem_width_aniso(w_pad, stride)
    slot_rows = (64 * h_mem) // height  # 8 at height 256 (16 at 4x8)
    if w_pad == min(widths) and all(s == 4 for s in nslots):
        rgb = torch.stack([r for r, _, _, _ in loaded])
        gray = torch.stack([g for _, g, _, _ in loaded])
        staff = torch.stack([st for _, _, st, _ in loaded])
        key_mask = None
    else:
        white = WHITE.expand(3, height, w_pad).clone()
        paper = torch.ones(1, height, w_pad)
        zero = torch.zeros(1, height, w_pad)
        rgb, gray, staff = [], [], []
        for (r, g, st, _), w in zip(loaded, widths):
            d = w_pad - w
            rgb.append(torch.cat([r, white[:, :, :d]], dim=2)
                       if d else r)
            gray.append(torch.cat([g, paper[:, :, :d]], dim=2)
                        if d else g)
            staff.append(torch.cat([st, zero[:, :, :d]], dim=2)
                         if d else st)
        rgb, gray, staff = (torch.stack(rgb), torch.stack(gray),
                            torch.stack(staff))
        key_mask = torch.ones(len(loaded), h_mem * w_mem, dtype=torch.bool)
        for i, (w, s) in enumerate(zip(widths, nslots)):
            valid = _mem_width_aniso(w, stride)
            pad_cols = torch.arange(w_mem) >= valid
            pad_rows = (
                torch.arange(h_mem) >= slot_rows * s
                if s < 4 else torch.zeros(h_mem, dtype=torch.bool)
            )
            key_mask[i] = (pad_rows.unsqueeze(1)
                           | pad_cols.unsqueeze(0)).reshape(-1)
        if not key_mask.any():
            key_mask = None
    return rgb, encode_targets(batch, max_tokens=max_tokens), key_mask, gray, staff


def hybrid_model_input(heat_net, pair, height, device, stride='8x16',
                       hybrid_index=None):
    """Single-pair ``[1, 11 or 4, H, W]`` model input for eval/probes (T2.4)."""
    rgb, _, _, gray, staff = collate_hybrid([pair], height, stride,
                                            hybrid_index=hybrid_index)
    return assemble_hybrid_batch(
        heat_net, rgb.to(device),
        gray.to(device) if heat_net is not None else None,
        staff.to(device))


def assemble_hybrid_batch(heat_net, rgb, gray, staff):
    """``(B,3,H,W)`` + gray + staff -> ``(B,11,H,W)`` or ``(B,4,H,W)`` model input.

    When ``heat_net`` is provided: one batched HeatNet forward (fp32, no_grad
    inside :meth:`heat_features.HeatNet.probs`; call OUTSIDE the AMP autocast
    block) + staff-grid concat (11-ch). When ``heat_net is None`` (4-ch
    ``--extra-chans 1 --no-heatnet`` mode): concatenates RGB + canonical staff
    grid directly without running HeatNet.
    """
    if heat_net is None:
        return torch.cat([rgb, staff.to(rgb.device).float()], dim=1)
    heat7 = heat_net.probs(gray)
    return torch.cat([rgb, heat7.to(rgb.device).float(),
                      staff.to(rgb.device).float()], dim=1)


def heat_channel_dropout(imgs, p):
    """S13b: per-channel inverted dropout on the 8 hybrid-extra maps.

    Drops whole channels ``[:, 3:, :, :]`` (frozen HeatNet 7-ch +
    staff grid) with probability ``p`` each, rescaling kept channels
    by ``1 / (1 - p)``; RGB ``[:, :3]`` is untouched. Forces the trunk
    to read through missing keypoint channels (decoupling from
    RowNet/HeatNet failures). ``p == 0`` returns ``imgs`` unchanged
    (no RNG consumed, legacy path bitwise identical); ``p >= 1`` is
    refused. Uses the global torch RNG (seeded once in ``main``), so
    the step->batch map stays the only run-order input.
    """
    if p == 0.0:
        return imgs
    if not 0.0 < p < 1.0:
        raise ValueError('heat dropout p must be in [0, 1), got %r' % p)
    keep = (torch.rand(imgs.shape[0], imgs.shape[1] - 3, 1, 1,
                       device=imgs.device, dtype=imgs.dtype) >= p)
    out = imgs.clone()
    out[:, 3:] = out[:, 3:] * keep.to(imgs.dtype) / (1.0 - p)
    return out


def _mem_width_aniso(w, stride='8x16'):
    """Encoder output width for input width ``w`` (v2 anisotropic trunk).

    Width path: stem k4/s4 (no pad) -> /2 -> /2 -> s1 (right pad keeps):
    ``((w // 4) // 2) // 2`` (exact conv arithmetic; ``w // 16`` when
    ``w`` is a multiple of 16). Height is exact at ``H // 8``.
    ``stride='4x8'`` (S13) halves the horizontal stride once instead
    of twice: ``(w // 4) // 2``. Default keeps legacy values.
    """
    if stride == '4x8':
        return (w // 4) // 2
    if stride != '8x16':
        raise ValueError('stride must be 8x16 or 4x8, got %r' % stride)
    return ((w // 4) // 2) // 2


def collate(batch, height, stride='8x16', max_tokens=0):
    """Batch variable-width strips: pad widths, pad targets, mask memory.

    Images are padded on the right to the batch ``W_max`` (rounded up to
    a multiple of 16 so the /16 encoder stays exact) with the
    standardized white-paper value; targets are padded with
    ``data.PAD_ID`` (every target still ends with EOS first, so the
    model observes a stop signal). Returns ``(imgs, tgt, key_mask)``
    where ``key_mask`` is None when all widths match, else a bool
    ``[B, H_mem * W_mem]`` memory_key_padding_mask (True = ignore) for
    the decoder's cross-attention over padded memory columns.
    ``stride='4x8'`` (S13) sizes the grid for the 4x8 trunk; default
    keeps the exact legacy grid.
    """
    if stride not in ('8x16', '4x8'):
        raise ValueError('stride must be 8x16 or 4x8, got %r' % stride)
    pages = [load_image(p, height) for p, _, _ in batch]
    widths = [pg.shape[2] for pg in pages]
    w_max = max(widths)
    # Round up to a multiple of 16: keeps every item's feature width
    # exact under the /16 horizontal stride (see _mem_width_aniso).
    w_pad = (w_max + 15) // 16 * 16
    if w_pad == min(widths):
        imgs = torch.stack(pages)
        key_mask = None
    else:
        white = WHITE.expand(3, height, w_pad).clone()
        imgs = torch.stack([
            torch.cat([pg, white[:, :, :w_pad - w]], dim=2)
            if w < w_pad else pg for pg, w in zip(pages, widths)])
        h_mem = height // (4 if stride == '4x8' else 8)
        w_mem = _mem_width_aniso(w_pad, stride)
        key_mask = torch.ones(len(pages), h_mem * w_mem, dtype=torch.bool)
        for i, w in enumerate(widths):
            valid = _mem_width_aniso(w, stride)
            # True = ignore: padded columns beyond the item's own
            # feature width; every row shares the same valid columns.
            pad_cols = torch.arange(w_mem) >= valid
            key_mask[i] = (pad_cols.unsqueeze(0).expand(h_mem, w_mem)
                           .reshape(-1))
    # Every target ends with EOS: without it the model never observes a
    # stop signal and greedy decoding always runs to max_len.
    return imgs, encode_targets(batch, max_tokens=max_tokens), key_mask


def build_optimizer(net, lr, trunk_mult=1.0, warmup=0, cosine=False,
                    total_updates=1, fresh_lr_mult=2.5):
    """Multi-group AdamW (trunk / dec / fresh-heads x decay/plain) + warmup/cosine.

    Trunk groups (``encoder.*``, ``proj.*``) run at ``lr * trunk_mult``;
    decoder groups at ``lr``; fresh Exp-2B heads (``s18_head.*``,
    ``comer_bias.*``, ``gdcm_head.*``) run at ``lr * fresh_lr_mult`` (2.5x
    default) when present so newly attached heads adapt quickly during
    warm-starting. ``plain`` groups hold 1D params, biases,
    ``layer_scale`` and ``norm.weight`` at ``weight_decay=0.0`` (T1.6a);
    ``decay`` groups keep the legacy 0.01. The LambdaLR schedule counts
    OPTIMIZER
    updates (post-accum flushes): linear warmup over ``warmup`` updates,
    then optional cosine decay to ~0 over ``total_updates``. Defaults
    (mult 1.0, no warmup, no cosine, no fresh heads) reproduce the legacy
    single-LR run.

    Scheduler state IS checkpointed (``opt-<step>.pt`` sidecar next to
    each ``ckpt-<step>.pt``): --resume restores optimizer momentum and
    the LambdaLR position when the sidecar exists and the optimizer
    hyperparameters match; otherwise it starts fresh (legacy
    weights-only behavior, with a note). ``ckpt-*.pt`` stays
    weights-only so inference loading never changes.
    """
    def _no_decay(name, p):
        """1D tensors, biases, LayerScale and norm weights skip decay."""
        return (p.ndim <= 1 or name.endswith('layer_scale')
                or name.endswith('bias') or name.endswith('norm.weight'))

    splits = {'trunk_decay': [], 'trunk_plain': [],
              'dec_decay': [], 'dec_plain': [],
              'fresh_decay': [], 'fresh_plain': []}
    for _n, _p in net.named_parameters():
        if not _p.requires_grad:
            continue
        if _n.startswith(('s18_head.', 'comer_bias.', 'gdcm_head.')):
            _key = 'fresh_'
        elif _n.startswith(('encoder.', 'proj.')):
            _key = 'trunk_'
        else:
            _key = 'dec_'
        _key += 'plain' if _no_decay(_n, _p) else 'decay'
        splits[_key].append(_p)
    grouped = [(splits['trunk_decay'], trunk_mult, 0.01),
               (splits['dec_decay'], 1.0, 0.01),
               (splits['trunk_plain'], trunk_mult, 0.0),
               (splits['dec_plain'], 1.0, 0.0),
               (splits['fresh_decay'], fresh_lr_mult, 0.01),
               (splits['fresh_plain'], fresh_lr_mult, 0.0)]
    grouped = [(ps, m, w) for ps, m, w in grouped if ps]
    assert splits['trunk_decay'] and splits['dec_decay'], \
        'param-group split is empty'
    assert (sum(len(v) for v in splits.values())
            == sum(1 for _p in net.parameters() if _p.requires_grad)), \
        'param overlap/leak'
    use_sched = (warmup > 0 or cosine)
    all_cuda = all(p.is_cuda for ps, _, _ in grouped for p in ps)
    opt_kw = {'fused': True} if all_cuda else {}
    param_groups = [
        {'params': ps, 'lr': (lr if use_sched else lr * m), 'weight_decay': w}
        for ps, m, w in grouped
    ]
    try:
        opt = torch.optim.AdamW(param_groups, **opt_kw)
    except (RuntimeError, TypeError):
        opt = torch.optim.AdamW(param_groups)
    if not use_sched:
        return opt, None

    def make_mult(mult):
        def mult_fn(u):
            u = float(u)
            if warmup > 0 and u < warmup:
                return mult * (u + 1.0) / warmup
            if cosine:
                span = max(1, total_updates - warmup)
                t = min(1.0, max(0.0, (u - warmup) / span))
                return mult * (0.5 + 0.5 * math.cos(math.pi * t))
            return mult
        return mult_fn

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, [make_mult(m) for _, m, _ in grouped])
    return opt, sched


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ca != cb))
        prev = cur
    return prev[len(b)]


def grounding_probe(net, eimgs, ref_ids):
    """Standing blank-image grounding probe (T0.3).

    Runs ``net.greedy_kv`` on the real eval image and on a blank
    all-zeros tensor, strips ``EOS_ID``/``PAD_ID`` from both hyps, and
    returns ``(hyp_real, ter_real, ter_blank, blank_diff)``. A model
    that ignores pixels hallucinates the same prior for both inputs
    (``blank_diff`` ~ 0); one that grounds in the image diverges.
    """
    hyp_real = net.greedy_kv(eimgs, max_len=len(ref_ids) + 16,
                             constrain=True)[0].tolist()
    blank = torch.zeros_like(eimgs)
    hyp_blank = net.greedy_kv(blank, max_len=len(ref_ids) + 16,
                              constrain=True)[0].tolist()

    def _strip(hyp):
        hyp = [x for x in hyp if x != data.PAD_ID]
        if data.EOS_ID in hyp:
            hyp = hyp[:hyp.index(data.EOS_ID)]
        return hyp

    hyp_real = _strip(hyp_real)
    hyp_blank = _strip(hyp_blank)
    ter_real = edit_distance(hyp_real, ref_ids) / max(1, len(ref_ids))
    ter_blank = edit_distance(hyp_blank, ref_ids) / max(1, len(ref_ids))
    blank_diff = (edit_distance(hyp_real, hyp_blank)
                  / max(1, max(len(hyp_real), 1)))
    return (hyp_real, ter_real, ter_blank, blank_diff)


def load_weights_compat(net, state):
    """Load a checkpoint into ``net``, transplanting the stem / expanding
    the vocabulary when needed.

    Strict load first (identical shapes, e.g. same ``extra_chans`` and
    same vocab). When that fails, each mismatch class below is handled
    independently (both may apply at once); any other shape mismatch
    re-raises the original error:

    * widened T2.2 stem (``encoder.0.0.weight`` [96, 3, 4, 4]
      checkpoint vs [96, 3 + extra, 4, 4] net): every other tensor
      loads strictly and the checkpoint's RGB ``[:, :3]`` weights +
      bias land in the widened stem (extras keep zero init);
    * T3.2 vocab expansion (``tok_emb.weight`` / ``out.weight`` /
      ``out.bias`` with FEWER ckpt rows than the net at the same width,
      e.g. a 133+3=136-row v2.4 ckpt under the 175+3=178-row T3.2
      vocab): the overlapping prefix rows copy verbatim (existing ids
      never shift — the first 136 rows stay byte-identical, including
      the old PAD/BOS/EOS rows now sitting on ``grace``/``slur-start``/
      ``slur-stop`` ids, while the new PAD/BOS/EOS rows are fresh) and
      the extra rows are fresh-init with the model default init
      (``tok_emb``: normal_(0, 1) like ``nn.Embedding``;
      ``out.weight``: kaiming-uniform a=sqrt(5) and ``out.bias``:
      uniform +/-1/sqrt(fan_in) like ``nn.Linear``).
    * S13 stride transfer (``_PadConv``-wrapped downsample blocks):
      an 8x16 checkpoint's ``encoder.{i}.1.{weight,bias}`` keys map
      onto the 4x8 net's ``encoder.{i}.1.conv.{weight,bias}`` names
      when shapes match exactly (same tensors, renamed by the
      wrapper). Composes with the aux-fresh rule above; the
      ``how`` string gains a ``+padconv`` suffix.
    """
    try:
        net.load_state_dict(state)
        return 'strict'
    except RuntimeError:
        pass
    # S13 stride transfer: ``_apply_stride_4x8`` wraps downsample
    # blocks in ``_PadConv``, renaming e.g. ``encoder.2.1.weight``
    # to ``encoder.2.1.conv.weight`` with IDENTICAL tensors. Map an
    # 8x16 checkpoint's unwrapped keys onto the net's wrapped names
    # first (moved, not copied — a leftover unwrapped key would
    # trip the no-unexpected-keys asserts below); the paths below
    # then see a shape-identical load and the ``how`` string records
    # the rename. Partial matches fall through to the loud re-raise.
    own = net.state_dict()
    renamed = {}
    for k in own:
        if '.1.conv.' not in k or not (
                k.endswith('.weight') or k.endswith('.bias')):
            continue
        ck = k.replace('.1.conv.', '.1.', 1)
        if (ck in state and ck not in own
                and tuple(state[ck].shape) == tuple(own[k].shape)):
            renamed[k] = ck
    padconv = bool(renamed)
    if padconv:
        state = dict(state)
        for k, ck in renamed.items():
            state[k] = state.pop(ck)
    # Detect stem transplant / slicing modes:
    #   * widen:       3 -> >3  (RGB copied, extras keep zero init)
    #   * slice-11to4: 11 -> 4  (RGB [:3] + canonical staff grid [10:11] -> [3:4])
    #   * slice-11to3: 11 -> 3  (RGB [:3] only)
    stem_w, stem_b = 'encoder.0.0.weight', 'encoder.0.0.bias'
    emb_w, out_w, out_b = 'tok_emb.weight', 'out.weight', 'out.bias'
    own = net.state_dict()
    stem_mode = None
    if (stem_w in state and stem_w in own
            and len(state[stem_w].shape) == 4
            and len(own[stem_w].shape) == 4
            and state[stem_w].shape[0] == own[stem_w].shape[0]
            and state[stem_w].shape[2:] == own[stem_w].shape[2:]):
        state_ch = int(state[stem_w].shape[1])
        own_ch = int(own[stem_w].shape[1])
        bias_ok = (
            stem_b not in state or stem_b not in own
            or tuple(state[stem_b].shape) == tuple(own[stem_b].shape)
        )
        if bias_ok:
            if state_ch == 3 and own_ch > 3:
                stem_mode = 'widen'
            elif state_ch == 11 and own_ch == 4:
                stem_mode = 'slice-11to4'
            elif state_ch == 11 and own_ch == 3:
                stem_mode = 'slice-11to3'
    stem_mismatch = stem_mode is not None

    n_old = None
    if all(k in state and k in own for k in (emb_w, out_w, out_b)):
        ce, ne = state[emb_w].shape, own[emb_w].shape
        co, no = state[out_w].shape, own[out_w].shape
        cb, nb = state[out_b].shape, own[out_b].shape
        if (len(ce) == 2 and len(ne) == 2 and len(co) == 2
                and len(no) == 2 and len(cb) == 1 and len(nb) == 1
                and ce[1] == ne[1] == co[1] == no[1]
                and ce[0] == co[0] == cb[0]
                and ne[0] == no[0] == nb[0]
                and ce[0] < ne[0]):
            n_old = ce[0]
    vocab_mismatch = n_old is not None

    # Gracefully skip mismatched/missing aux_head.* weights (e.g. pre-aux
    # ckpts, 8-ch -> 1-ch aux_head on 4-ch models, or disabled aux_head).
    own_aux = {k for k in own if k.startswith('aux_head.')}
    state_aux = {k for k in state if k.startswith('aux_head.')}
    skip_own_aux = {
        k for k in own_aux
        if k not in state or tuple(state[k].shape) != tuple(own[k].shape)
    }
    skip_state_aux = {
        k for k in state_aux
        if k not in own or tuple(state[k].shape) != tuple(own[k].shape)
    }
    aux_skipped = bool(skip_own_aux or skip_state_aux)

    # Gracefully keep fresh zero-init foveal_proj.* weights when warm-starting
    # a --foveal-tap model from a pre-tap checkpoint.
    own_foveal = {k for k in own if k.startswith('foveal_proj.')}
    state_foveal = {k for k in state if k.startswith('foveal_proj.')}
    foveal_fresh = bool(own_foveal and not state_foveal)

    # Gracefully keep fresh Exp-2B heads (s18_head.*, comer_bias.*, gdcm_head.*)
    # when warm-starting from a pre-Exp-2B checkpoint (e.g. exp2A ckpt-15000),
    # or when vocab expansion grows s18_head.subtype_head.* rows.
    own_s18 = {k for k in own if k.startswith('s18_head.')}
    state_s18 = {k for k in state if k.startswith('s18_head.')}
    s18_fresh = bool(own_s18 and not state_s18)
    skip_own_s18 = {
        k for k in own_s18
        if k not in state or tuple(state[k].shape) != tuple(own[k].shape)
    }
    skip_state_s18 = {
        k for k in state_s18
        if k not in own or tuple(state[k].shape) != tuple(own[k].shape)
    }
    s18_resized = bool(not s18_fresh and (skip_own_s18 or skip_state_s18))

    own_comer = {k for k in own if k.startswith('comer_bias.')}
    state_comer = {k for k in state if k.startswith('comer_bias.')}
    comer_fresh = bool(own_comer and not state_comer)

    own_gdcm = {k for k in own if k.startswith('gdcm_head.')}
    state_gdcm = {k for k in state if k.startswith('gdcm_head.')}
    gdcm_fresh = bool(own_gdcm and not state_gdcm)

    # Legacy strict+aux-fresh fast path (preserves exact return string).
    if (not stem_mismatch and not vocab_mismatch and not foveal_fresh
            and not s18_fresh and not s18_resized
            and not comer_fresh and not gdcm_fresh
            and own_aux and not state_aux
            and all(k in state and tuple(state[k].shape) == tuple(own[k].shape)
                    for k in own if k not in own_aux)
            and not (set(state) - set(own))):
        topped = dict(state)
        topped.update({k: own[k] for k in own_aux})
        net.load_state_dict(topped)
        return ('strict+aux-fresh+padconv' if padconv
                else 'strict+aux-fresh')

    if not (padconv or stem_mismatch or vocab_mismatch or aux_skipped
            or foveal_fresh or s18_fresh or s18_resized
            or comer_fresh or gdcm_fresh):
        net.load_state_dict(state)  # re-raise the original error

    skip_own = set(skip_own_aux) | set(skip_own_s18)
    skip_state = set(skip_state_aux) | set(skip_state_s18)
    if foveal_fresh:
        skip_own |= own_foveal
    if s18_fresh:
        skip_own |= own_s18
    if comer_fresh:
        skip_own |= own_comer
    if gdcm_fresh:
        skip_own |= own_gdcm
    if stem_mismatch:
        skip_own |= {stem_w, stem_b}
        skip_state |= {stem_w, stem_b}
    if vocab_mismatch:
        skip_own |= {emb_w, out_w, out_b}
        skip_state |= {emb_w, out_w, out_b}
    with torch.no_grad():
        if stem_mismatch:
            stem = net.encoder[0][0]
            if stem_mode == 'widen':
                stem.weight[:, :3].copy_(state[stem_w])
            elif stem_mode == 'slice-11to4':
                stem.weight[:, :3].copy_(state[stem_w][:, :3])
                stem.weight[:, 3:4].copy_(state[stem_w][:, 10:11])
            elif stem_mode == 'slice-11to3':
                stem.weight[:, :3].copy_(state[stem_w][:, :3])
            if stem.bias is not None and stem_b in state:
                stem.bias.copy_(state[stem_b])
        if vocab_mismatch:
            n_new = net.tok_emb.weight.shape[0]
            n_added = n_new - n_old
            g2_id = data.TOKEN_TO_ID.get('clef-G-2')
            old_n_tok = n_old - 3
            new_n_tok = n_new - 3
            dev = net.tok_emb.weight.device
            if (g2_id is not None and 0 <= g2_id < old_n_tok
                    and n_added == 2):
                # Clef-appendix expansion (e.g. 176+3=179 -> 178+3=181 adding
                # clef-G-2-8vb, clef-G-2-8va right before PAD/BOS/EOS):
                #   * 0..old_n_tok-1  -> copied verbatim
                #   * old specials    -> shifted to new specials (n_new-3..n_new-1)
                #   * new clef rows   -> initialized from clef-G-2 row
                net.tok_emb.weight[:old_n_tok].copy_(state[emb_w][:old_n_tok])
                net.out.weight[:old_n_tok].copy_(state[out_w][:old_n_tok])
                net.out.bias[:old_n_tok].copy_(state[out_b][:old_n_tok])

                net.tok_emb.weight[new_n_tok:].copy_(state[emb_w][old_n_tok:])
                net.out.weight[new_n_tok:].copy_(state[out_w][old_n_tok:])
                net.out.bias[new_n_tok:].copy_(state[out_b][old_n_tok:])

                net.tok_emb.weight[old_n_tok:new_n_tok].copy_(
                    state[emb_w][g2_id:g2_id + 1].expand(n_added, -1))
                net.out.weight[old_n_tok:new_n_tok].copy_(
                    state[out_w][g2_id:g2_id + 1].expand(n_added, -1))
                net.out.bias[old_n_tok:new_n_tok].copy_(
                    state[out_b][g2_id:g2_id + 1].expand(n_added))

                assert torch.equal(net.tok_emb.weight[:old_n_tok],
                                   state[emb_w][:old_n_tok].to(dev)), 'emb prefix shifted'
                assert torch.equal(net.tok_emb.weight[new_n_tok:],
                                   state[emb_w][old_n_tok:].to(dev)), 'emb specials shifted'
                assert torch.equal(net.out.weight[:old_n_tok],
                                   state[out_w][:old_n_tok].to(dev)), 'head prefix shifted'
                assert torch.equal(net.out.weight[new_n_tok:],
                                   state[out_w][old_n_tok:].to(dev)), 'head specials shifted'
                assert torch.equal(net.out.bias[:old_n_tok],
                                   state[out_b][:old_n_tok].to(dev)), 'bias prefix shifted'
                assert torch.equal(net.out.bias[new_n_tok:],
                                   state[out_b][old_n_tok:].to(dev)), 'bias specials shifted'
            else:
                net.tok_emb.weight[:n_old].copy_(state[emb_w])
                net.out.weight[:n_old].copy_(state[out_w])
                net.out.bias[:n_old].copy_(state[out_b])
                torch.nn.init.normal_(net.tok_emb.weight[n_old:])
                torch.nn.init.kaiming_uniform_(net.out.weight[n_old:],
                                               a=math.sqrt(5))
                fan_in = net.out.weight.shape[1]
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(net.out.bias[n_old:], -bound, bound)
                assert torch.equal(net.tok_emb.weight[:n_old],
                                   state[emb_w].to(dev)), 'emb prefix shifted'
                assert torch.equal(net.out.weight[:n_old],
                                   state[out_w].to(dev)), 'head prefix shifted'
                assert torch.equal(net.out.bias[:n_old],
                                   state[out_b].to(dev)), 'bias prefix shifted'
        if s18_resized:
            # Preserve overlapping prefix rows on any resized s18_head tensor
            # (e.g. subtype_head growing when new clef tokens are appended).
            own_params = dict(net.named_parameters())
            for k in skip_own_s18:
                if (k in state and k in own_params
                        and state[k].ndim == own_params[k].ndim
                        and state[k].shape[1:] == own_params[k].shape[1:]
                        and state[k].shape[0] < own_params[k].shape[0]):
                    n_s = state[k].shape[0]
                    own_params[k][:n_s].copy_(state[k])
    rest = {k: v for k, v in state.items() if k not in skip_state}
    missing, unexpected = net.load_state_dict(rest, strict=False)
    assert not unexpected, unexpected
    assert set(missing) == (skip_own & set(own)), missing
    parts = []
    if stem_mode == 'widen':
        parts.append('stem-transplant')
    elif stem_mode is not None:
        parts.append('stem-' + stem_mode)
    if vocab_mismatch:
        parts.append('vocab-expand')
    if not parts:
        parts.append('strict')
    if aux_skipped and not (stem_mode == 'widen' or vocab_mismatch):
        parts.append('aux-fresh')
    if foveal_fresh:
        parts.append('foveal-fresh')
    if s18_fresh:
        parts.append('s18-fresh')
    if comer_fresh:
        parts.append('comer-fresh')
    if gdcm_fresh:
        parts.append('gdcm-fresh')
    if padconv:
        parts.append('padconv')
    return '+'.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--renders', required=True)
    ap.add_argument('--targets', required=True)
    ap.add_argument('--pages', nargs='*', default=None,
                    help='only pages containing one of these substrings')
    ap.add_argument('--pages-file', default=None,
                    help='stem-per-line list for EXACT page selection '
                         '(curriculum bands; avoids --pages substring '
                         'collisions). Unions with --pages when both given.')
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='drop pages containing any of these substrings '
                         '(held-out gate sets, e.g. ds0005)')
    ap.add_argument('--shuffle', action='store_true',
                    help='shuffle pair order once with Random(seed) before '
                         'training; recorded in meta.json and enforced '
                         'on --resume')
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=6,
                    help='true mini-batch over pair order (default 6: fastest '
                         'non-paging batch on 16 GB RTX 4080 SUPER with '
                         'shared pass; use 4 for extra VRAM headroom on '
                         'very wide strips)')
    ap.add_argument('--pretrained', action='store_true',
                    help='init the trunk from timm ConvNeXt-Tiny weights '
                         'with the anisotropic 8x16 encoder (MvdNet v2)')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--no-amp', action='store_true',
                    help='disable fp16 autocast (full fp32). Use after an '
                         'AMP overflow: finite weights but NaN loss/memd '
                         '(epoch-3 extension diverged at step ~5920 with '
                         'all-finite ckpt-6000).')
    ap.add_argument('--accum', type=int, default=1,
                    help='gradient accumulation: optimizer steps every K '
                         'pages (effective batch K at batch-1 memory). '
                         'Pending grads flush at every eval/final step, so '
                         'checkpoints always sit on update boundaries and '
                         '--resume regroups identically.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--resume', action='store_true',
                    help='load the latest ckpt-<step>.pt and continue '
                         'after its step (step->page mapping is '
                         'deterministic, so this is exact)')
    ap.add_argument('--init-from', default=None,
                    help='curriculum handoff: load this ckpt-<step>.pt as '
                         'starting weights for a FRESH run (different page '
                         'set allowed; step counter restarts at 1, '
                         'optimizer restarts). Not combinable with --resume.')
    ap.add_argument('--trunk-lr-mult', type=float, default=1.0,
                    help='LR multiplier for the vision trunk '
                         '(encoder.* + proj.*) vs the decoder. Reviewer '
                         'rec #1: 10x lets the random trunk outrun the '
                         'decoder LM prior. 1.0 = legacy single-LR.')
    ap.add_argument('--warmup', type=int, default=0,
                    help='linear LR warmup over this many OPTIMIZER updates '
                         '(0 = off). Counts optimizer steps (post-accum), '
                         'not pages.')
    ap.add_argument('--cosine', action='store_true',
                    help='cosine-decay LRs to ~0 over the run after warmup.')
    ap.add_argument('--sample-p', type=float, default=0.0,
                    help='two-pass scheduled-sampling probability per input '
                         'token (0.0 = pure teacher forcing, legacy). '
                         'Reviewer rec #2: anneal 0 -> 0.3 across stages '
                         'manually; this flag is the constant-p mechanism.')
    ap.add_argument('--no-degraded', action='store_true',
                    help='pair each bekern stem with its clean PNG only '
                         '(skip <stem>_d*.jpg). Clean-first curriculum '
                         'phase; recorded in meta.json and enforced '
                         'on --resume.')
    ap.add_argument('--handink', action='store_true',
                    help='spike-S10: also pair <stem>_h*.png/.jpg '
                         'hand-ink variants with the same bekern targets '
                         '(appended after degraded pairs per stem). '
                         'Variants need .hybrid.pt sidecars or the '
                         'hybrid filter drops them. Recorded in meta.json '
                         'and enforced on --resume.')
    ap.add_argument('--simshots', action='store_true',
                    help='S16b: also pair <stem>_s*.png/.jpg simulated '
                         'screen captures with the same bekern targets '
                         '(appended after hand-ink pairs per stem). '
                         'Variants need .hybrid.pt sidecars or the '
                         'hybrid filter drops them. Recorded in meta.json '
                         'and enforced on --resume.')
    ap.add_argument('--tok-weight', type=float, default=1.0,
                    help='per-token NLL multiplier for the bar token '
                         '(bar-only since T1.6a). 1.0 = legacy '
                         'unweighted. Recorded in meta.json and enforced '
                         'on --resume.')
    ap.add_argument('--tok-dropout', type=float, default=0.0,
                    help='content-token input dropout probability (0.0 = '
                         'off, legacy). Replaces content-token inputs '
                         '(steps, octaves, durations, marks, rest) with '
                         'uniform random real tokens. Recorded in meta.json '
                         'and enforced on --resume.')
    ap.add_argument('--margin-weight', type=float, default=0.0,
                    help='spike-S3 sibling-margin loss weight (0.0 = off, '
                         'legacy). Adds weight * mean(relu(margin + '
                         'sib_lp - ref_lp)) over --margin-siblings '
                         'modifier-edited siblings of one stream-picked '
                         'sample per margin step. Recorded in meta.json '
                         'and enforced on --resume.')
    ap.add_argument('--margin-siblings', type=int, default=8,
                    help='spike-S3 sibling count per margin step. '
                         'Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--margin', type=float, default=1.0,
                    help='spike-S3 log-prob margin (nats) the reference '
                         'must beat each sibling by. Recorded in meta.json '
                         'and enforced on --resume.')
    ap.add_argument('--margin-every', type=int, default=1,
                    help='spike-S3 margin cadence in steps (1 = every '
                         'step). Recorded in meta.json and enforced '
                         'on --resume.')
    ap.add_argument('--mwer-weight', type=float, default=0.0,
                    help='spike-S4 MWER loss weight (0.0 = off, legacy). '
                         'Adds weight * sum_softmax_p(r - mean(r)) over '
                         '--mwer-beams on-the-fly grammar beams of one '
                         'stream-picked sample per MWER step, mixed with '
                         'CE. Recorded in meta.json and enforced '
                         'on --resume.')
    ap.add_argument('--mwer-beams', type=int, default=4,
                    help='spike-S4 on-the-fly beam count. Recorded in '
                         'meta.json and enforced on --resume.')
    ap.add_argument('--mwer-every', type=int, default=1,
                    help='spike-S4 MWER cadence in steps. Recorded in '
                         'meta.json and enforced on --resume.')
    ap.add_argument('--mwer-maxlen', type=int, default=576,
                    help='spike-S4 beam max_len (must cover the strip '
                         'reference; truncating below ref length would '
                         'fake premature-EOS penalties). Recorded in '
                         'meta.json and enforced on --resume.')
    ap.add_argument('--action-weight', type=float, default=0.0,
                    help='spike-S7 token-level expected keystroke-risk '
                         'weight (0.0 = off, legacy). Adds weight * mean '
                         'E_{v~p}[C[v, ref]] over the CE forward logits '
                         '(bar-involved 3x, modifier/EOS 2x, else 1x); no '
                         'extra decode passes. Recorded in meta.json and '
                         'enforced on --resume.')
    ap.add_argument('--action-every', type=int, default=1,
                    help='spike-S7 action-risk cadence in steps. Recorded '
                         'in meta.json and enforced on --resume.')
    ap.add_argument('--specaug-p', type=float, default=0.0,
                    help='spike-S8 latent SpecAugment probability per step '
                         '(0.0 = off, legacy). When applied, the batch '
                         'forward runs through encoder[:4] + pitch-shear '
                         'SpecAugment + encoder[4:] with a spec sampled '
                         'from the seed:step:specaug stream. Recorded in '
                         'meta.json and enforced on --resume.')
    ap.add_argument('--specaug-slope', type=float, default=0.02,
                    help='spike-S8 max |pitch-shear slope| in latent px '
                         'per column. Recorded in meta.json and enforced '
                         'on --resume.')
    ap.add_argument('--extra-chans', type=int, default=0,
                    help='hybrid extra encoder channels (T2.2: 8 = HeatNet '
                         '7-ch + 1 staff-grid ch, appended after RGB, so '
                         'in_chans = 3 + extra-chans; stem extras are '
                         'zero-init). Recorded in meta.json and enforced '
                         'on --resume.')
    ap.add_argument('--stride', default='8x16', choices=('8x16', '4x8'),
                    help='S13: encoder grid stride. 4x8 doubles both axes '
                         '(H=256 -> 64 rows) for denser visual grounding; '
                         'only stride/padding hyperparameters move, so '
                         'timm transfer arity is unchanged. Recorded in '
                         'meta.json and enforced on --resume.')
    ap.add_argument('--aux-weight', type=float, default=0.0,
                    help='S13b: dense ink/keypoint aux loss weight '
                         '(distills the 8 hybrid-extra channels into a '
                         '2-layer trunk head, every --aux-every steps). '
                         'Requires --hybrid-cache. Recorded in meta.json '
                         'and enforced on --resume.')
    ap.add_argument('--aux-every', type=int, default=1,
                    help='S13b: aux term cadence (every Nth step). '
                         'Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--heat-dropout', type=float, default=0.0,
                    help='S13b: per-channel inverted dropout probability '
                         'on the 8 hybrid-extra input maps (decouples the '
                         'trunk from HeatNet/RowNet failures). 0.0 keeps '
                         'the legacy path bitwise identical. Recorded in '
                         'meta.json and enforced on --resume.')
    ap.add_argument('--coord-weight', type=float, default=0.0,
                    help='S13c: cross-attention x-monotonicity weight '
                         '(penalizes backward reading jumps per voice '
                         'segment; voice tokens reset). 0.0 keeps the '
                         'legacy path bitwise identical. Recorded in '
                         'meta.json and enforced on --resume.')
    ap.add_argument('--coord-every', type=int, default=1,
                    help='S13c: coordinate-term cadence (every Nth step). '
                         'Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--hybrid-cache', default=None,
                    help='hybrid rectified-training mode (T2.4): path to '
                         'hybrid_staff.jsonl. Training pairs are filtered '
                         'to strips with a <img>.hybrid.pt sidecar '
                         '(skip-only-degraded policy); RGB comes from the '
                         'rectified sidecar (not load_image), heat 7-ch '
                         'from one batched CUDA HeatNet forward per step. '
                         'Requires --extra-chans 8. Recorded in meta.json '
                         'and enforced on --resume.')
    ap.add_argument('--heat-pt', default=heat_features.HEAT_PT_DEFAULT,
                    help='HeatNet .pt checkpoint for --hybrid-cache '
                         '(CUDA when available; loaded once, eval mode). '
                         'Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--no-heatnet', action='store_true',
                    help='skip loading/running frozen HeatNet in '
                         '--hybrid-cache mode (auto-enabled with '
                         '--extra-chans 1 for 4-ch RGB+staff). '
                         'Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--foveal-tap', action='store_true',
                    help='enable Stage-0 (P2, 4x4) avg+max foveal tap '
                         'residual in MvdNet for micro-glyph survival. '
                         'Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--optics-profile', default='default',
                    choices=('default', 'calibrated'),
                    help='epoch_aug camera/optics profile (default or '
                         'calibrated). Recorded in meta.json and enforced '
                         'on --resume.')
    ap.add_argument('--jsd-weight', type=float, default=0.0,
                    help='Batch 4-B5 clean-vs-degraded logit JSD consistency '
                         'weight (0.0 = off, legacy; runs clean teacher under '
                         'no_grad on B_jsd=2, skipped when W > 2176). '
                         'Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--jsd-every', type=int, default=4,
                    help='Batch 4-B5 logit JSD cadence in steps (default 4). '
                         'Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--epoch-aug-p', type=float, default=0.0,
                    help='per-epoch deterministic physical aug probability on '
                         'already-degraded _d* strips (default 0.0 to avoid '
                         'double-degrading _d* JPEGs).')
    ap.add_argument('--epoch-aug-clean-p', type=float, default=0.75,
                    help='per-epoch physical+geom aug probability on clean '
                         '.png strips (default 0.75: keeps 25%% clean anchor '
                         'floor, turns 75%% into fresh photo draws each epoch).')
    ap.add_argument('--epoch-aug-geom', type=float, default=0.15,
                    help='3D geometric warp strength in epoch_aug (default '
                         '0.15 = gentle +/-1-2 px spline-jitter preserving '
                         'canonical staff alignment; 0.0 = photometric only).')
    ap.add_argument('--epoch-redraw-dir', default=None,
                    help='optional root holding ep<E>/ fresh photo-degrade '
                         'redraws; substitutes degraded strips per epoch '
                         'while keeping clean .png anchors untouched.')
    ap.add_argument('--bf16', action=argparse.BooleanOptionalAction, default=True,
                    help='use native bfloat16 AMP + TF32 on CUDA (default on; '
                         'pass --no-bf16 for fp16+GradScaler).')
    ap.add_argument('--channels-last', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='use NHWC channels_last memory format on ConvNeXt '
                         'encoder + inputs (default on; --no-channels-last '
                         'for NCHW).')
    ap.add_argument('--width-bucket', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='group similar-width strips into micro-batches per '
                         'epoch (default on; --no-width-bucket for flat).')
    ap.add_argument('--prefetch', type=int, default=4,
                    help='async background collate prefetch depth (default 4; '
                         '0 = synchronous legacy collate).')
    ap.add_argument('--real-hw-dir', default=None,
                    help='optional root of permissive real handwritten training '
                         'strips ({renders/, targets/, hybrid_staff.jsonl}); '
                         'appended deterministically to training pairs.')
    ap.add_argument('--real-hw-repeat', type=int, default=1,
                    help='deterministic repeat factor for --real-hw-dir pairs '
                         '(default 1). Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--s18-weight', type=float, default=0.0,
                    help='Exp-2B S18 factored event-tuple auxiliary head loss '
                         'weight (0.0 = off, legacy). Shares the single decoder '
                         'hidden state pass. Recorded in meta.json and enforced '
                         'on --resume.')
    ap.add_argument('--comer', action='store_true',
                    help='Exp-2B CoMER coverage-guided cross-attention bias '
                         '(zero-init for lossless warm-start). Recorded in '
                         'meta.json and enforced on --resume.')
    ap.add_argument('--gdcm-weight', type=float, default=0.0,
                    help='Exp-2B GDCM global dynamic counting head loss weight '
                         '(0.0 = off, legacy; zero-init memory conditioning). '
                         'Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--extra-data-dir', action='append', default=None,
                    help='repeatable or comma-separated extra data directories '
                         '(each containing strip PNGs, *-bekern.txt, and optional '
                         'hybrid_staff.jsonl) merged into pairs and hybrid_index. '
                         'Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--max-tokens', type=int, default=768,
                    help='maximum target token sequence length (default 768; '
                         '0 = unlimited). Recorded in meta.json and enforced on '
                         '--resume.')
    ap.add_argument('--real-hw-corpus', default=None,
                    help='optional path to strictly labeled real handwritten '
                         'corpus (e.g. D:\\tmp\\real-hw-corpus\\ or extracted '
                         'strip dir; NEVER unlabeled RISM). Works alongside '
                         '--real-hw-dir. Recorded in meta.json and enforced on '
                         '--resume.')
    ap.add_argument('--hw-mix-ratio', type=float, default=0.0,
                    help='fraction of supervised batch items drawn from the '
                         'labeled real HW pool (default 0.0 = legacy; '
                         'recommended 0.12-0.15). Recorded in meta.json and '
                         'enforced on --resume.')
    ap.add_argument('--hw-pure-frac', type=float, default=0.35,
                    help='fraction of labeled HW samples kept pure/unaugmented '
                         '(epoch_aug skipped) while 1 - hw_pure_frac receive '
                         'photometric/ink/camera augmentation (default 0.35). '
                         'Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--rism-incipit-dir', default=None,
                    help='optional path to semi-supervised barline-gated RISM '
                         'opening-staff incipit strips (tracked separately '
                         'from full labeled HW). Recorded in meta.json and '
                         'enforced on --resume.')
    ap.add_argument('--rism-mix-ratio', type=float, default=0.0,
                    help='fraction of supervised batch items drawn from the '
                         'semi-supervised RISM incipit pool (default 0.0; '
                         'recommended 0.05-0.10). Recorded in meta.json and '
                         'enforced on --resume.')
    ap.add_argument('--unsup-strips-dir', default=None,
                    help='comma-separated directories of unlabeled system '
                         'strips for scheduled unsupervised rounds (e.g. '
                         'D:\\tmp\\unlabeled-10k-real\\strips,'
                         'D:\\tmp\\rism-incipit-strips\\unlabeled_strips). '
                         'Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--unsup-every-steps', type=int, default=0,
                    help='run one unsupervised round every N supervised steps '
                         '(default 0 = off; e.g. 1000). Recorded in meta.json '
                         'and enforced on --resume.')
    ap.add_argument('--unsup-round-steps', type=int, default=100,
                    help='steps per scheduled unsupervised round (default 100). '
                         'Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--unsup-mode', default='both',
                    choices=('ema_enc', 'bar_math', 'both'),
                    help='unsupervised round mode: ema_enc, bar_math, or both '
                         '(default both). Recorded in meta.json and enforced '
                         'on --resume.')
    ap.add_argument('--unsup-weight', type=float, default=0.25,
                    help='unsupervised round loss weight (default 0.25). '
                         'Recorded in meta.json and enforced on --resume.')
    ap.add_argument('--unsup-kl-beta', type=float, default=0.10,
                    help='unsupervised round KL regularization beta '
                         '(default 0.10). Recorded in meta.json and enforced '
                         'on --resume.')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device, '| torch:', torch.__version__)

    pairs = list_pairs(args.renders, args.targets, exclude=args.exclude,
                       include_degraded=not args.no_degraded,
                       include_handink=args.handink,
                       include_simshots=args.simshots)
    pairs = filter_pages(pairs, args.pages, args.pages_file)
    hw_pool_pairs = []
    hw_corpus_hidx = {}
    if args.real_hw_dir:
        hw_pairs = load_real_hw_pairs(
            args.real_hw_dir, exclude=args.exclude,
            include_degraded=not args.no_degraded,
            repeat=args.real_hw_repeat)
        print('real-hw pairs: +%d from %s (repeat=%d)'
              % (len(hw_pairs), args.real_hw_dir, args.real_hw_repeat))
        if args.hw_mix_ratio > 0.0:
            hw_pool_pairs.extend(hw_pairs)
        else:
            pairs = pairs + hw_pairs
    if args.real_hw_corpus:
        c_pairs, hw_corpus_hidx = load_real_hw_corpus_pairs(
            args.real_hw_corpus, exclude=args.exclude,
            include_degraded=not args.no_degraded)
        print('real-hw-corpus pairs: +%d (+%d hybrid rows) from %s'
              % (len(c_pairs), len(hw_corpus_hidx), args.real_hw_corpus))
        if args.hw_mix_ratio > 0.0:
            hw_pool_pairs.extend(c_pairs)
        else:
            pairs = pairs + c_pairs
    rism_pool_pairs = []
    rism_hidx = {}
    if args.rism_incipit_dir:
        rism_pool_pairs, rism_hidx = load_rism_incipit_pairs(
            args.rism_incipit_dir, exclude=args.exclude,
            include_degraded=not args.no_degraded)
        print('rism-incipit pairs (semi-supervised): +%d (+%d hybrid rows) '
              'from %s (mix_ratio=%.3f)'
              % (len(rism_pool_pairs), len(rism_hidx), args.rism_incipit_dir,
                 args.rism_mix_ratio))
    unsup_strip_paths = (
        list_unlabeled_images(args.unsup_strips_dir)
        if args.unsup_strips_dir else [])
    if args.unsup_every_steps > 0:
        print('unsup schedule: every=%d steps, round_steps=%d, mode=%s, '
              'strips=%d'
              % (args.unsup_every_steps, args.unsup_round_steps,
                 args.unsup_mode, len(unsup_strip_paths)))
    extra_hidxs = []
    for edir in parse_extra_data_dirs(args.extra_data_dir):
        epairs, ehidx = load_extra_data_dir(
            edir, exclude=args.exclude,
            include_degraded=not args.no_degraded,
            include_handink=args.handink,
            include_simshots=args.simshots)
        print('extra-data-dir: +%d pairs (+%d hybrid rows) from %s'
              % (len(epairs), len(ehidx), edir))
        pairs = pairs + epairs
        if ehidx:
            extra_hidxs.append(ehidx)
    assert pairs, 'no training pairs selected'
    hybrid_index = None
    use_hybrid = bool(args.hybrid_cache)
    use_heatnet = (
        use_hybrid and (args.extra_chans == heat_features.EXTRA_CHANS)
        and not args.no_heatnet
    )
    if use_hybrid:
        if args.extra_chans not in (1, heat_features.EXTRA_CHANS):
            sys.exit('refusing: --hybrid-cache requires --extra-chans 1 '
                     '(RGB+staff) or %d (RGB+HeatNet+staff), got %d'
                     % (heat_features.EXTRA_CHANS, args.extra_chans))
        if args.no_heatnet and args.extra_chans == heat_features.EXTRA_CHANS:
            sys.exit('refusing: --no-heatnet cannot be combined with '
                     '--extra-chans %d (use --extra-chans 1 for 4-ch '
                     'RGB+staff)' % heat_features.EXTRA_CHANS)
        hybrid_index = load_hybrid_index(args.hybrid_cache)
        if args.real_hw_dir:
            hw_hidx = load_real_hw_hybrid_index(args.real_hw_dir)
            if hw_hidx:
                hybrid_index.update(hw_hidx)
                print('real-hw hybrid index: +%d validated strips' % len(hw_hidx))
        if hw_corpus_hidx:
            hybrid_index.update(hw_corpus_hidx)
        if rism_hidx:
            hybrid_index.update(rism_hidx)
        for ehidx in extra_hidxs:
            hybrid_index.update(ehidx)
        print('hybrid index: %d validated strips in %s'
              % (len(hybrid_index), args.hybrid_cache))
        pairs, drop_clean, drop_deg = filter_hybrid_pairs(
            pairs, hybrid_index=hybrid_index, allow_on_the_fly=True)
        if hw_pool_pairs:
            hw_pool_pairs, _, _ = filter_hybrid_pairs(
                hw_pool_pairs, hybrid_index=hybrid_index, allow_on_the_fly=True)
        if rism_pool_pairs:
            rism_pool_pairs, _, _ = filter_hybrid_pairs(
                rism_pool_pairs, hybrid_index=hybrid_index, allow_on_the_fly=True)
        print('hybrid filter: kept %d pairs (dropped clean=%d '
              'degraded=%d without sidecars/tables)'
              % (len(pairs), drop_clean, drop_deg))
        assert pairs, 'no training pairs with sidecars or tables selected'
    hw_img_set = frozenset(p[0] for p in hw_pool_pairs)
    if args.aux_weight > 0 and not use_hybrid:
        sys.exit('refusing: --aux-weight needs --hybrid-cache (aux '
                 'targets are the hybrid-extra channels)')
    if args.shuffle:
        random.Random(args.seed).shuffle(pairs)
    print('pairs:', len(pairs))

    use_gdcm = args.gdcm_weight > 0
    use_s18 = args.s18_weight > 0
    if args.pretrained:
        net = model.MvdNet(vocab=len(data.VOCAB) + 3, anisotropic=True,
                           pretrained=True, stride=args.stride,
                           extra_chans=args.extra_chans,
                           foveal_tap=args.foveal_tap,
                           use_comer=args.comer,
                           use_gdcm=use_gdcm,
                           use_s18=use_s18).to(device)
        print('MvdNet v2: anisotropic %s encoder + timm pretrained trunk'
              % args.stride
              + (' + %d hybrid extra chans' % args.extra_chans
                 if args.extra_chans else '')
              + (' + foveal_tap' if args.foveal_tap else '')
              + (' + CoMER' if args.comer else '')
              + (' + GDCM' if use_gdcm else '')
              + (' + S18' if use_s18 else ''))
    else:
        net = model.MvdNet(vocab=len(data.VOCAB) + 3,
                           stride=args.stride,
                           extra_chans=args.extra_chans,
                           foveal_tap=args.foveal_tap,
                           use_comer=args.comer,
                           use_gdcm=use_gdcm,
                           use_s18=use_s18).to(device)
    K0 = max(1, args.accum)
    total_updates = (args.steps // K0 + 2 * (args.steps //
                     max(1, args.eval_every) + 1) + 2)
    opt, sched = build_optimizer(net, args.lr, args.trunk_lr_mult,
                                 args.warmup, args.cosine, total_updates)
    if args.trunk_lr_mult != 1.0:
        print('trunk LR x%.1f (differential groups)' % args.trunk_lr_mult)
    if args.warmup > 0 or args.cosine:
        print('schedule: warmup %d updates, cosine=%s (horizon ~%d updates)'
              % (args.warmup, args.cosine, total_updates))
    amp = (device == 'cuda') and not args.no_amp
    amp_dtype = (torch.bfloat16 if (amp and args.bf16) else torch.float16)
    if amp and args.bf16:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print('AMP: native bfloat16 + TF32 (GradScaler disabled)')
    if args.no_amp:
        print('AMP disabled: full fp32')
    if args.channels_last and device == 'cuda':
        net.encoder = net.encoder.to(memory_format=torch.channels_last)
        net.proj = net.proj.to(memory_format=torch.channels_last)
        if args.foveal_tap and hasattr(net, 'foveal_proj'):
            net.foveal_proj = net.foveal_proj.to(memory_format=torch.channels_last)
        if args.comer and hasattr(net, 'comer_bias'):
            net.comer_bias = net.comer_bias.to(memory_format=torch.channels_last)
        print('memory format: channels_last (NHWC) on encoder+proj'
              + ('+foveal' if args.foveal_tap else '')
              + ('+comer' if args.comer else ''))
    scaler = torch.amp.GradScaler('cuda', enabled=(amp and amp_dtype == torch.float16))
    heat_net = None
    if use_heatnet:
        heat_net = heat_features.HeatNet(args.heat_pt, device=device)
        print('HeatNet: %s on %s (%d bytes)'
              % (args.heat_pt, heat_net.device,
                 heat_net.param_bytes()), flush=True)
    elif use_hybrid:
        print('hybrid 4-ch mode: RGB + canonical staff grid (HeatNet skipped)',
              flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    acquire_out_lock(out)
    meta = build_meta(args, pairs)
    if args.resume:
        refusal = check_resume(out, meta)
        if refusal is not None:
            sys.exit(refusal)
    atomic_write_text(out / 'meta.json', json.dumps(meta, indent=1))

    start = 1
    resumed_ckpt = None
    if args.resume and args.init_from:
        sys.exit('refusing: --resume and --init-from are exclusive '
                 '(resume continues one run; init starts a fresh one)')
    if args.init_from:
        state = torch.load(args.init_from, map_location='cpu',
                           weights_only=True)
        print('init weights from %s (fresh schedule)' % args.init_from)
    if args.resume:
        ckpts = sorted(out.glob('ckpt-*.pt'),
                       key=lambda p: int(p.stem.split('-')[1]))
        if ckpts:
            last = ckpts[-1]
            state = torch.load(last, map_location='cpu', weights_only=True)
            start = int(last.stem.split('-')[1]) + 1
            resumed_ckpt = start - 1
            print('resuming from %s (step %d)' % (last.name, start - 1))
        else:
            state = None
            print('no checkpoint found, starting fresh')
    elif args.init_from is None:
        state = None

    if state is not None:
        # Weights only here; optimizer/scheduler restore happens below
        # (after `updates` exists) from the opt-<step>.pt sidecar.
        # Compat loader: strict when shapes match, else RGB stem
        # transplant for --extra-chans nets from 3-ch checkpoints.
        how = load_weights_compat(net, state)
        if how != 'strict':
            print('weights: %s' % how)
    n = len(pairs)
    K = max(1, args.accum)
    opt.zero_grad()
    pending = 0
    updates = [0]
    acost = None  # spike-S7 cost matrix, built lazily on device
    if args.resume and resumed_ckpt is not None:
        hp_now = {'lr': args.lr, 'trunk_mult': args.trunk_lr_mult,
                  'warmup': args.warmup, 'accum': args.accum,
                  'cosine': args.cosine}
        sidecar = out / ('opt-%d.pt' % resumed_ckpt)
        if sidecar.exists():
            blob = torch.load(sidecar, map_location='cpu',
                              weights_only=True)
            if blob.get('hp') == hp_now:
                opt.load_state_dict(blob['opt'])
                if (sched is not None
                        and blob.get('sched') is not None):
                    sched.load_state_dict(blob['sched'])
                updates[0] = int(blob.get('updates', 0))
                print('resumed optimizer+scheduler from %s (updates=%d)'
                      % (sidecar.name, updates[0]))
            else:
                print('optimizer hyperparams changed since %s; starting '
                      'fresh optimizer (weights kept)' % sidecar.name)
        else:
            print('no optimizer sidecar for ckpt-%d; starting fresh '
                  'optimizer (legacy weights-only resume)' % resumed_ckpt)

    def flush():
        # One guarded optimizer step: unscale, bound, step.
        # (NaN-cliff note: epoch 3 died at step ~5920 from AMP fp16
        # overflow off exploding trunk activations; fp32 + this
        # clipping runs clean. scaler.step still skips inf/NaN grads.)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        updates[0] += 1
        if sched is not None:
            sched.step()

    B = max(1, args.batch_size)
    nb = (n + B - 1) // B
    if B > 1:
        print('batch-size %d: %d batches/epoch over %d pairs'
              % (B, nb, n))
    pair_widths = None
    width_by_img = None
    hw_sorted = None
    rism_sorted = None
    if args.width_bucket and B > 1:
        pair_widths = fast_loader.measure_pair_widths(
            pairs, height=args.height, hybrid_index=hybrid_index)
        width_by_img = {p[0]: w for p, w in zip(pairs, pair_widths)}
        print('width-bucket: enabled (min=%d, median=%d, max=%d px)'
              % (min(pair_widths), sorted(pair_widths)[len(pair_widths) // 2],
                 max(pair_widths)))
        if hw_pool_pairs and args.hw_mix_ratio > 0.0:
            hw_ws = fast_loader.measure_pair_widths(
                hw_pool_pairs, height=args.height, hybrid_index=hybrid_index)
            hw_order = sorted(range(len(hw_pool_pairs)), key=lambda i: hw_ws[i])
            hw_sorted = ([hw_ws[i] for i in hw_order],
                         [hw_pool_pairs[i] for i in hw_order])
        if rism_pool_pairs and args.rism_mix_ratio > 0.0:
            rism_ws = fast_loader.measure_pair_widths(
                rism_pool_pairs, height=args.height, hybrid_index=hybrid_index)
            rism_order = sorted(range(len(rism_pool_pairs)),
                                key=lambda i: rism_ws[i])
            rism_sorted = ([rism_ws[i] for i in rism_order],
                           [rism_pool_pairs[i] for i in rism_order])

    def _step_batches():
        cached_ep = -1
        ep_batches = None
        for st in range(start, args.steps + 1):
            ep = (st - 1) // nb
            bi = (st - 1) % nb
            if pair_widths is not None:
                if ep != cached_ep:
                    ep_batches = fast_loader.epoch_bucketed_batches(
                        pairs, pair_widths, B, args.seed, ep)
                    cached_ep = ep
                raw_b = ep_batches[bi]
            else:
                b0 = bi * B
                raw_b = pairs[b0:b0 + B]
            if (hw_pool_pairs and args.hw_mix_ratio > 0.0) or (
                    rism_pool_pairs and args.rism_mix_ratio > 0.0):
                raw_b, _, _ = mix_supervised_batch(
                    raw_b, st, args.seed,
                    hw_pairs=hw_pool_pairs, hw_mix_ratio=args.hw_mix_ratio,
                    rism_pairs=rism_pool_pairs,
                    rism_mix_ratio=args.rism_mix_ratio,
                    width_by_img=width_by_img,
                    hw_sorted=hw_sorted,
                    rism_sorted=rism_sorted)
            yield (st, ep, epoch_aug.resolve_epoch_batch(
                raw_b, ep, args.epoch_redraw_dir))

    def _collate_step(_st, _ep, batch):
        if not use_hybrid:
            return collate(batch, args.height, args.stride,
                           max_tokens=args.max_tokens)
        return collate_hybrid(batch, args.height, args.stride,
                              hybrid_index=hybrid_index,
                              max_tokens=args.max_tokens)

    non_block = (device == 'cuda' and args.prefetch > 0)
    if args.prefetch > 0:
        stream_ctx = fast_loader.AsyncBatchPrefetcher(
            _step_batches(), _collate_step,
            num_workers=min(4, max(2, args.prefetch)),
            prefetch=args.prefetch,
            pin_memory=(device == 'cuda'))
    else:
        class _SyncStream:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def __iter__(self):
                for st, ep, batch in _step_batches():
                    yield (st, ep, batch, _collate_step(st, ep, batch))
        stream_ctx = _SyncStream()

    with stream_ctx as batch_stream:
      for step, epoch, batch, collated in batch_stream:
        clean_imgs_jsd = None
        if not use_hybrid:
            imgs, tgt, key_mask = collated
            imgs = imgs.to(device, non_blocking=non_block)
            tgt = tgt.to(device, non_blocking=non_block)
            need_jsd = (
                args.jsd_weight > 0
                and step % max(1, args.jsd_every) == 0
                and imgs.shape[-1] <= (1600 if args.comer else 2176)
            )
            if need_jsd:
                b_jsd = min(2, imgs.shape[0])
                clean_imgs_jsd = imgs[:b_jsd].clone()
            hw_idx = (frozenset(i for i, p in enumerate(batch)
                                if p[0] in hw_img_set)
                      if hw_img_set else frozenset())
            if hw_idx:
                imgs, _, _, _ = apply_epoch_aug_with_hw_pure(
                    imgs, batch, args.seed, epoch, step,
                    aug_p=args.epoch_aug_p, clean_p=args.epoch_aug_clean_p,
                    geom_strength=args.epoch_aug_geom,
                    optics_profile=args.optics_profile,
                    hw_indices=hw_idx, hw_pure_frac=args.hw_pure_frac)
            elif args.epoch_aug_p > 0 or args.epoch_aug_clean_p > 0:
                imgs, _ = epoch_aug.augment_standardized_batch(
                    imgs, batch, args.seed, epoch,
                    aug_p=args.epoch_aug_p, clean_p=args.epoch_aug_clean_p,
                    geom_strength=args.epoch_aug_geom,
                    optics_profile=args.optics_profile)
        else:
            # Hybrid: rectified RGB + staff from sidecars; when HeatNet is
            # active (11-ch), heat comes from one batched fp32 HeatNet forward
            # OUTSIDE autocast. In 4-ch mode (--extra-chans 1), HeatNet is
            # skipped completely and imgs = [rgb, staff].
            rgb, tgt, key_mask, grayb, staffb = collated
            if batch_too_wide(rgb.shape[-1]):
                print('skip step %d: batch width %d > cap %d'
                      % (step, rgb.shape[-1], BATCH_WIDTH_CAP),
                      flush=True)
                continue
            tgt = tgt.to(device, non_blocking=non_block)
            rgb_d = rgb.to(device, non_blocking=non_block)
            gray_d = (grayb.to(device, non_blocking=non_block)
                      if heat_net is not None else None)
            staff_d = staffb.to(device, non_blocking=non_block)
            need_jsd = (
                args.jsd_weight > 0
                and step % max(1, args.jsd_every) == 0
                and rgb_d.shape[-1] <= (1600 if args.comer else 2176)
            )
            if need_jsd:
                b_jsd = min(2, rgb_d.shape[0])
                clean_imgs_jsd = assemble_hybrid_batch(
                    heat_net, rgb_d[:b_jsd].clone(),
                    gray_d[:b_jsd].clone() if gray_d is not None else None,
                    staff_d[:b_jsd].clone())
            hw_idx = (frozenset(i for i, p in enumerate(batch)
                                if p[0] in hw_img_set)
                      if hw_img_set else frozenset())
            if hw_idx:
                rgb_d, gray_d, staff_d, _ = apply_epoch_aug_with_hw_pure(
                    rgb_d, batch, args.seed, epoch, step,
                    aug_p=args.epoch_aug_p, clean_p=args.epoch_aug_clean_p,
                    geom_strength=args.epoch_aug_geom,
                    optics_profile=args.optics_profile,
                    hw_indices=hw_idx, hw_pure_frac=args.hw_pure_frac,
                    gray_d=gray_d, staff_d=staff_d)
            elif args.epoch_aug_p > 0 or args.epoch_aug_clean_p > 0:
                rgb_d, gray_d, staff_d = epoch_aug.augment_standardized_batch(
                    rgb_d, batch, args.seed, epoch,
                    aug_p=args.epoch_aug_p, clean_p=args.epoch_aug_clean_p,
                    grayb=gray_d, staffb=staff_d,
                    geom_strength=args.epoch_aug_geom,
                    optics_profile=args.optics_profile)
            imgs = assemble_hybrid_batch(
                heat_net, rgb_d, gray_d, staff_d)
            # S13b: aux targets are the UNDROPPED extras (denoising:
            # predict full keypoint maps from channel-dropped inputs).
            aux_tgt = imgs[:, 3:].detach() if args.aux_weight > 0 else None
            if args.heat_dropout > 0:
                imgs = heat_channel_dropout(imgs, args.heat_dropout)
        if args.channels_last and device == 'cuda':
            imgs = imgs.to(memory_format=torch.channels_last)
            if clean_imgs_jsd is not None:
                clean_imgs_jsd = clean_imgs_jsd.to(memory_format=torch.channels_last)
        if key_mask is not None:
            key_mask = key_mask.to(device, non_blocking=non_block)
        aug = None
        if args.specaug_p > 0:
            # Spike S8: train-only latent SpecAugment, a pure function
            # of (seed, step) on its own stream — no run history.
            # None keeps the legacy forward for that step.
            aug = maybe_specaug(
                random.Random('%d:%d:specaug' % (args.seed, step)),
                args.specaug_p, args.specaug_slope)
        need_aux = (args.aux_weight > 0 and step % max(1, args.aux_every) == 0)
        need_coord = (args.coord_weight > 0 and step % max(1, args.coord_every) == 0)
        need_s18 = (args.s18_weight > 0)
        need_gdcm = (args.gdcm_weight > 0)
        # Width-aware micro-chunking: at H=256 and B=4, batches wider than
        # 2176 px exceed 16 GB VRAM and spill into Windows WDDM (triggering
        # slow paging or TDR driver resets). Splitting wide batches into
        # B=2 sub-chunks keeps peak VRAM <= 11 GB with mathematically
        # identical gradients and zero skipped strips up to BATCH_WIDTH_CAP.
        _b_cur = imgs.shape[0]
        _no_extra_terms = (
            args.margin_weight <= 0 and args.mwer_weight <= 0
            and args.action_weight <= 0 and not need_jsd)
        _mb = 0
        if _no_extra_terms:
            if _b_cur > 1 and imgs.shape[-1] > 2048 and (args.comer or imgs.shape[-1] > 2560):
                _mb = 1
            elif _b_cur > 2 and (imgs.shape[-1] > 2176 or (args.comer and imgs.shape[-1] > 1408)):
                _mb = 2
        if _mb > 0:
            _loss_acc = torch.zeros((), device=device, dtype=torch.float32)
            _aux_acc = (torch.zeros((), device=device, dtype=torch.float32)
                        if need_aux else None)
            _coord_acc = (torch.zeros((), device=device, dtype=torch.float32)
                          if need_coord else None)
            _s18_acc = (torch.zeros((), device=device, dtype=torch.float32)
                        if need_s18 else None)
            _gdcm_acc = (torch.zeros((), device=device, dtype=torch.float32)
                         if need_gdcm else None)
            mterm = wterm = aterm = jterm = None
            for _s in range(0, _b_cur, _mb):
                _e = min(_b_cur, _s + _mb)
                _wf = (_e - _s) / float(_b_cur)
                _im = imgs[_s:_e]
                _tg = tgt[_s:_e]
                _km = key_mask[_s:_e] if key_mask is not None else None
                with torch.amp.autocast('cuda', enabled=amp, dtype=amp_dtype):
                    _sh = net.loss_shared(
                        _im, _tg, sample_p=args.sample_p,
                        mem_mask=_km, tok_weight=args.tok_weight,
                        weight_ids=STRUCT_WEIGHT_IDS,
                        tok_dropout=args.tok_dropout,
                        content_ids=CONTENT_IDS,
                        need_aux=need_aux, need_coord=need_coord,
                        augment=aug, return_hidden=need_s18,
                        need_gdcm=need_gdcm)
                    _ce, _ap, _xc = _sh[0], _sh[1], _sh[2]
                    _si = 3
                    _h = _sh[_si] if need_s18 else None
                    if need_s18:
                        _si += 1
                    _gd = _sh[_si] if need_gdcm else None
                    _sub = _ce * (_wf / K)
                    if need_aux:
                        _at = aux_tgt[_s:_e]
                        _ag = torch.nn.functional.interpolate(
                            _at, size=_ap.shape[-2:], mode='area')
                        _ab = torch.nn.functional.binary_cross_entropy_with_logits(
                            _ap, _ag, reduction='none').mean(dim=1)
                        if _km is not None:
                            _av = (~_km).to(_ab.dtype).reshape(_ab.shape)
                            _ax = (_ab * _av).sum() / _av.sum().clamp_min(1.0)
                        else:
                            _ax = _ab.mean()
                        _sub = _sub + args.aux_weight * _ax * (_wf / K)
                        _aux_acc = _aux_acc + _ax.detach().float() * _wf
                    if need_coord:
                        _ct = model.coord_monotonic_penalty(
                            _xc, _tg, model.VOICE_IDS, net.pad_id)
                        _sub = _sub + args.coord_weight * _ct * (_wf / K)
                        _coord_acc = _coord_acc + _ct.detach().float() * _wf
                    if need_s18:
                        _s18l, _ = net.s18_head.loss_from_tokens(
                            _h, _tg, pad_id=net.pad_id, eos_id=net.eos_id)
                        _sub = _sub + args.s18_weight * _s18l * (_wf / K)
                        _s18_acc = _s18_acc + _s18l.detach().float() * _wf
                    if need_gdcm and _gd is not None:
                        _sub = _sub + args.gdcm_weight * _gd * (_wf / K)
                        _gdcm_acc = _gdcm_acc + _gd.detach().float() * _wf
                scaler.scale(_sub).backward()
                _loss_acc = _loss_acc + _sub.detach().float()
            loss = _loss_acc
            auxterm = _aux_acc
            cterm = _coord_acc
            s18term = _s18_acc
            gterm = _gdcm_acc
        else:
            with torch.amp.autocast('cuda', enabled=amp, dtype=amp_dtype):
                shared_out = net.loss_shared(
                    imgs, tgt, sample_p=args.sample_p,
                    mem_mask=key_mask, tok_weight=args.tok_weight,
                    weight_ids=STRUCT_WEIGHT_IDS,
                    tok_dropout=args.tok_dropout,
                    content_ids=CONTENT_IDS,
                    need_aux=need_aux, need_coord=need_coord,
                    augment=aug, return_hidden=need_s18,
                    return_logits=need_jsd, need_gdcm=need_gdcm)
                ce_loss, apred_shared, xc_shared = (
                    shared_out[0], shared_out[1], shared_out[2])
                _si = 3
                h_shared = shared_out[_si] if need_s18 else None
                if need_s18:
                    _si += 1
                logits_shared = shared_out[_si] if need_jsd else None
                if need_jsd:
                    _si += 1
                gdcm_shared = shared_out[_si] if need_gdcm else None
                loss = ce_loss / K
                s18term = None
                if need_s18:
                    s18term, _ = net.s18_head.loss_from_tokens(
                        h_shared, tgt, pad_id=net.pad_id, eos_id=net.eos_id)
                    loss = loss + args.s18_weight * s18term / K
                gterm = None
                if need_gdcm and gdcm_shared is not None:
                    gterm = gdcm_shared
                    loss = loss + args.gdcm_weight * gterm / K
                jterm = None
                if need_jsd and clean_imgs_jsd is not None:
                    b_jsd = clean_imgs_jsd.shape[0]
                    with torch.no_grad():
                        _cbos = torch.full_like(tgt[:b_jsd, :1], net.bos_id)
                        _cinp = torch.cat([_cbos, tgt[:b_jsd, :-1]], dim=1)
                        _ckm = key_mask[:b_jsd] if key_mask is not None else None
                        logits_clean = net(clean_imgs_jsd, _cinp, _ckm)
                    _jpad = (tgt[:b_jsd] == net.pad_id)
                    jterm = optics_losses.logit_jsd_consistency(
                        logits_clean, logits_shared[:b_jsd],
                        pad_mask=_jpad, detach_clean=True)
                    loss = loss + args.jsd_weight * jterm / K
                mterm = None
                if args.margin_weight > 0 and step % max(
                        1, args.margin_every) == 0:
                    # Spike S3: sibling margin on one stream-picked sample
                    # ('seed:step' string seed — deterministic across runs,
                    # no run-history input). One extra encode +
                    # (1 + siblings) decoder passes; the batch CE above is
                    # untouched.
                    mrng = random.Random('%d:%d' % (args.seed, step))
                    mi = mrng.randrange(tgt.shape[0])
                    ref_ids = [t for t in tgt[mi].tolist()
                               if t != data.PAD_ID]
                    sibs = mutate_modifier_siblings(
                        ref_ids, args.margin_siblings, mrng,
                        model.MODIFIER_IDS, len(data.VOCAB))
                    mimg = imgs[mi:mi + 1]
                    mmask = (key_mask[mi:mi + 1]
                             if key_mask is not None else None)
                    mem1 = net.encode(mimg, mmask)
                    ref_lp = sequence_logprob(net, mem1, ref_ids, mmask)
                    sib_lps = torch.stack(
                        [sequence_logprob(net, mem1, s, mmask)
                         for s in sibs])
                    mterm = sibling_margin_loss(ref_lp, sib_lps, args.margin)
                    loss = loss + args.margin_weight * mterm / K
                wterm = None
                if args.mwer_weight > 0 and step % max(
                        1, args.mwer_every) == 0:
                    # Spike S4: MWER on one stream-picked sample
                    # ('seed:step:mwer' string seed — deterministic across
                    # runs, no run-history input). B=4 grammar beams decode
                    # no-grad; their risks vs the reference weight the
                    # grad-carrying teacher-forced beam log-probs.
                    wrng = random.Random('%d:%d:mwer' % (args.seed, step))
                    wi = wrng.randrange(tgt.shape[0])
                    wref = [t for t in tgt[wi].tolist()
                            if t != data.PAD_ID]
                    wimg = imgs[wi:wi + 1]
                    wmask = (key_mask[wi:wi + 1]
                             if key_mask is not None else None)
                    with torch.no_grad():
                        wbeams = net.beam_search(
                            wimg, beam=args.mwer_beams, strict=True,
                            max_len=args.mwer_maxlen, mem_mask=wmask,
                            return_all=True)
                    whyps = [beam_hyp(b, net.eos_id, len(data.VOCAB))
                             for b in wbeams[0].tolist()]
                    wmem = net.encode(wimg, wmask)
                    wbeam_lps = torch.stack(
                        [sequence_logprob(net, wmem, h, wmask)
                         for h in whyps])
                    wrisks = torch.tensor(
                        [weighted_risk(wref, h, data.TOKEN_TO_ID['bar'],
                                       net.eos_id, model.MODIFIER_IDS)
                         for h in whyps],
                        device=wbeam_lps.device, dtype=torch.float32)
                    wterm = mwer_loss(wbeam_lps, wrisks)
                    loss = loss + args.mwer_weight * wterm / K
                aterm = None
                if args.action_weight > 0 and step % max(
                        1, args.action_every) == 0:
                    # Spike S7: token-level expected keystroke risk on the
                    # batch (one extra forward, no beams). Cost matrix is
                    # a pure function of ids; build once per run.
                    if acost is None:
                        acost = keystroke_cost_matrix(
                            net.vocab, model.MODIFIER_IDS).to(imgs.device)
                    abos = torch.full_like(tgt[:, :1], net.bos_id)
                    alogits = net(imgs, torch.cat([abos, tgt[:, :-1]],
                                                  dim=1), key_mask)
                    aterm = expected_keystroke_risk(alogits, tgt, acost)
                    loss = loss + args.action_weight * aterm / K
                auxterm = None
                if need_aux:
                    # S13b: dense ink/keypoint distillation from the shared trunk
                    # forward (zero redundant ConvNeXt pass).
                    apred = apred_shared
                    agres = torch.nn.functional.interpolate(
                        aux_tgt, size=apred.shape[-2:], mode='area')
                    abce = torch.nn.functional.binary_cross_entropy_with_logits(
                        apred, agres, reduction='none').mean(dim=1)
                    if key_mask is not None:
                        avalid = (~key_mask).to(abce.dtype).reshape(abce.shape)
                        auxterm = ((abce * avalid).sum()
                                   / avalid.sum().clamp_min(1.0))
                    else:
                        auxterm = abce.mean()
                    loss = loss + args.aux_weight * auxterm / K
                cterm = None
                if need_coord:
                    # S13c: monotonic-reading supervision from the shared decoder
                    # pass (zero redundant encoder/decoder pass).
                    xc = xc_shared
                    cterm = model.coord_monotonic_penalty(
                        xc, tgt, model.VOICE_IDS, net.pad_id)
                    loss = loss + args.coord_weight * cterm / K
            scaler.scale(loss).backward()
        pending += 1
        if pending >= K:
            flush()
            pending = 0
        if step == 1 or step % 50 == 0 or step == args.steps:
            msg = 'step %d/%d loss %.4f' % (step, args.steps,
                                            loss.item() * K)
            if mterm is not None:
                msg += ' marg %.4f' % float(mterm.item())
            if wterm is not None:
                msg += ' mwer %.4f' % float(wterm.item())
            if aterm is not None:
                msg += ' act %.4f' % float(aterm.item())
            if auxterm is not None:
                msg += ' aux %.4f' % float(auxterm.item())
            if cterm is not None:
                msg += ' coord %.4f' % float(cterm.item())
            if jterm is not None:
                msg += ' jsd %.4f' % float(jterm.item())
            if s18term is not None:
                msg += ' s18 %.4f' % float(s18term.item())
            if gterm is not None:
                msg += ' gdcm %.4f' % float(gterm.item())
            if aug is not None:
                msg += ' aug'
            _res, _alc = cuda_mem_stats(device)
            if _res is not None:
                msg += ' mem %.0f/%.0fMB' % (_res, _alc)
            print(msg, flush=True)
            if maintenance_due(step) and step != args.steps:
                # S13 VRAM wall: defrag the expandable-segments pool
                # and bank a ckpt so a kill loses <= 500 steps.
                # Numerics-neutral (cache + identical-schema save).
                torch.cuda.empty_cache()
                save_checkpoint(out, step, net, opt, sched,
                                updates, args)
        if unsup_round_due(step, args.unsup_every_steps) and unsup_strip_paths:
            if pending:
                flush()
                pending = 0
            u_res = run_stage2_unsup_round(
                net=net, opt=opt, scaler=scaler,
                strip_paths=unsup_strip_paths, step=step,
                round_steps=args.unsup_round_steps, mode=args.unsup_mode,
                weight=args.unsup_weight, kl_beta=args.unsup_kl_beta,
                height=args.height, device=device, amp=amp,
                amp_dtype=amp_dtype, stride=args.stride, seed=args.seed,
                batch_size=max(1, min(B, 2)))
            print('  unsup round step=%d mode=%s ran=%s strips=%d'
                  % (step, args.unsup_mode, u_res['ran'],
                     u_res['n_strips']), flush=True)
            net.train()
        if step % args.eval_every == 0 or step == args.steps:
            if pending:
                flush()
                pending = 0
            # Fixed reference page: rotating with the training batch made
            # eval TER incomparable across steps.
            net.eval()
            with torch.no_grad():
                if not use_hybrid:
                    eimgs, etgt, _ = collate([pairs[0]], args.height,
                                             args.stride)
                    eimgs = eimgs.to(device)
                # Encoder-collapse guard (epoch 2: mem went constant
                # across pages while loss kept falling). Two probe
                # images through encode(): pairwise distance ~0 means
                # the decoder is learning an unconditioned prior.
                # Sequential encodes, mean-pooled: page widths vary at
                # fixed height, so neither stacking nor flattening works
                # across pages (seq len h*w differs). The mean-pooled
                # distance still guards the collapse mode (identical mem
                # for all pages -> ~0). ConvNeXt uses LayerNorm (no batch
                # interaction), so sequential == batched per page.
                if not use_hybrid:
                    m0 = net.encode(load_image(
                        pairs[0][0], args.height).unsqueeze(0).to(device))
                    m1 = net.encode(load_image(
                        pairs[1 % len(pairs)][0],
                        args.height).unsqueeze(0).to(device))
                else:
                    ergb, etgt, _, egray, erstaff = collate_hybrid(
                        [pairs[0]], args.height, args.stride,
                        hybrid_index=hybrid_index)
                    eimgs = assemble_hybrid_batch(
                        heat_net, ergb.to(device),
                        egray.to(device) if heat_net is not None else None,
                        erstaff.to(device))
                    m0 = net.encode(eimgs)
                    m1 = net.encode(hybrid_model_input(
                        heat_net, pairs[1 % len(pairs)], args.height,
                        device, args.stride, hybrid_index=hybrid_index))
                memd = (m0.mean(dim=1).reshape(-1)
                        - m1.mean(dim=1).reshape(-1)).norm().item()
                hyp = net.greedy_kv(eimgs, max_len=etgt.shape[1] + 16,
                                    constrain=False)
            ref = etgt[0].tolist()
            ref = [x for x in ref if x != data.PAD_ID]
            hyp = hyp[0].tolist()
            eos_hit = data.EOS_ID in hyp
            if eos_hit:
                hyp = hyp[:hyp.index(data.EOS_ID)]
            ter = edit_distance(hyp, ref) / max(1, len(ref))
            exact = 'EXACT' if hyp == ref else ''
            # Blank-image grounding probe: a white page through encode()
            # with the same standardization as load_image. blankd ~0
            # while memd > 0 means the decoder keys off page identity
            # but not pixels; blankd ~ memd ~ 0 is full collapse.
            _mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            _std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            _blank = torch.full((3, args.height, args.height), 1.0)
            _blank = _blank.sub_(_mean).div_(_std).unsqueeze(0).to(device)
            if args.extra_chans > 0:
                # White page carries no staff lines and no keypoints:
                # the extra channels are exact zeros.
                _blank = torch.cat(
                    [_blank, torch.zeros(1, args.extra_chans,
                                         args.height, args.height,
                                         device=device)], dim=1)
            _mb = net.encode(_blank)
            blankd = (m0.mean(dim=1).reshape(-1)
                      - _mb.mean(dim=1).reshape(-1)).norm().item()
            hyp_real, ter_real, ter_blank, blank_diff = grounding_probe(
                net, eimgs, ref)
            _tfbos = torch.full_like(etgt[:, :1], data.BOS_ID,
                                           device=device)
            _tflogits = net(eimgs, torch.cat(
                [_tfbos, etgt[:, :-1].to(device)], dim=1))[0]
            _tfcls = tf_top1_by_class(_tflogits, etgt[0].tolist())
            _tfacc = {k: v[0] / max(1, v[1]) for k, v in _tfcls.items()}
            print('  eval page=%s len(ref)=%d len(hyp)=%d TER=%.4f '
                  'memd=%.1f blankd=%.1f ter_blank=%.4f blank_diff=%.4f '
                  '%s %s tfacc=h%.3f/b%.3f/q%.3f/s%.3f'
                  % (pairs[0][2], len(ref), len(hyp), ter, memd, blankd,
                     ter_blank, blank_diff,
                     'EOS' if eos_hit else 'NOEOS', exact,
                     _tfacc['header'], _tfacc['bar'], _tfacc['body'],
                     _tfacc['syntax']),
                  flush=True)
            with open(out / 'eval_log.jsonl', 'a',
                      encoding='utf-8') as _elog:
                _elog.write(json.dumps(
                    {'step': step, 'loss': float(loss.item() * K),
                     'ter': ter_real, 'ter_blank': ter_blank,
                     'blank_diff': blank_diff, 'memd': memd,
                     'tf_acc_header': _tfacc['header'],
                     'tf_acc_bar': _tfacc['bar'],
                     'tf_acc_body': _tfacc['body'],
                     'tf_acc_syntax': _tfacc['syntax']}) + '\n')
            save_checkpoint(out, step, net, opt, sched, updates, args)
            # Eval temps (~4 GB: full-res encodes + greedy states) stay
            # referenced as main() locals until the NEXT eval unless
            # released — a permanent floor between evals on a 16 GB
            # card. Numerics-neutral (all values already logged/saved).
            del eimgs, m0, m1, _blank, _mb, _tfbos, _tflogits
            net.train()
    print('done:', args.out)


if __name__ == '__main__':
    main()
