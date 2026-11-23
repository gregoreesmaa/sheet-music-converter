"""Per-epoch fresh redraws and realistic GPU-batched physical OMR augmentation.

Simulates real paper, aging, crumpling/folds, coffee/water stains, ink
degradation/bleed-through, musician usage (pencil/highlighter/smudges/tape),
and smartphone camera optics (phone shadow, spine shadow, chromatic
aberration, tilt-shift defocus, motion blur, white balance, sensor noise,
JPEG softness) — all **vectorized on GPU** and **100% deterministic** in
``(seed, epoch, sample_key)`` so ``--resume`` is bit-exact.

Key design invariant:
- **Zero vertical geometric displacement** on rectified strips, so canonical
  staff-slot rows (``y_canon(s, k)``) and ``staffb`` stay 0.0-px aligned.
- **Clean-floor protection:** clean ``.png`` anchors are preserved untouched
  by default (``clean_p=0.0``).
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LUMA_WEIGHTS = (0.299, 0.587, 0.114)
OPTICS_PROFILES = ('default', 'calibrated', 'heavy')
CALIBRATED_CLEAN_ANCHOR_FRAC = 0.25

_CONST_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_STAGE_CONSTS: dict[torch.device, dict[str, torch.Tensor]] = {}
_GRID_CACHE: dict[tuple[torch.device, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _resolve_optics_profile(optics_profile: str = 'default', heavy_optics: bool = False) -> str:
    """Normalize optics profile name; ``heavy_optics=True`` selects ``'calibrated'``."""
    prof = (optics_profile or 'default').strip().lower()
    if heavy_optics and prof == 'default':
        prof = 'calibrated'
    if prof == 'heavy':
        prof = 'calibrated'
    if prof not in ('default', 'calibrated'):
        raise ValueError(f'unknown optics_profile={optics_profile!r}; expected one of {OPTICS_PROFILES}')
    return prof


def _device_consts(dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = _CONST_CACHE.get(dev)
    if c is None:
        c = (
            torch.tensor(IMAGENET_MEAN, device=dev, dtype=torch.float32).view(3, 1, 1),
            torch.tensor(IMAGENET_STD, device=dev, dtype=torch.float32).view(3, 1, 1),
            torch.tensor(LUMA_WEIGHTS, device=dev, dtype=torch.float32).view(3, 1, 1),
        )
        _CONST_CACHE[dev] = c
    return c


def _device_grid(dev: torch.device, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = (dev, h, w)
    g = _GRID_CACHE.get(key)
    if g is None:
        if len(_GRID_CACHE) > 64:
            _GRID_CACHE.clear()
        g = (
            torch.linspace(-1.0, 1.0, w, device=dev, dtype=torch.float32).view(1, 1, w),
            torch.linspace(-1.0, 1.0, h, device=dev, dtype=torch.float32).view(1, h, 1),
        )
        _GRID_CACHE[key] = g
    return g


def _stage_consts(dev: torch.device) -> dict[str, torch.Tensor]:
    sc = _STAGE_CONSTS.get(dev)
    if sc is None:
        sc = {
            'ink_tint': torch.tensor([0.28, 0.17, 0.08], device=dev, dtype=torch.float32).view(1, 3, 1, 1),
            'sepia': torch.tensor([0.05, 0.16, 0.34], device=dev, dtype=torch.float32).view(1, 3, 1, 1),
            'fox_col': torch.tensor([0.14, 0.30, 0.48], device=dev, dtype=torch.float32).view(1, 3, 1, 1),
            'stain_col': torch.tensor([0.12, 0.32, 0.56], device=dev, dtype=torch.float32).view(1, 3, 1, 1),
            'hi_cal': torch.tensor(
                [[0.96, 0.92, 0.60], [0.98, 0.74, 0.82], [0.72, 0.95, 0.78]],
                device=dev, dtype=torch.float32,
            ).view(3, 3, 1, 1),
            'hi_def': torch.tensor(
                [[0.98, 0.97, 0.68], [0.99, 0.80, 0.86], [0.78, 0.97, 0.82]],
                device=dev, dtype=torch.float32,
            ).view(3, 3, 1, 1),
            'tape_cal': torch.tensor([0.92, 0.86, 0.68], device=dev, dtype=torch.float32).view(3, 1, 1),
            'tape_def': torch.tensor([0.96, 0.92, 0.78], device=dev, dtype=torch.float32).view(3, 1, 1),
        }
        _STAGE_CONSTS[dev] = sc
    return sc


def is_clean_image_path(img_path: str | Path) -> bool:
    """True when ``img_path`` is a clean render (no ``_d*``/``_h*``/``_s*`` tag)."""
    stem = Path(img_path).stem
    for tag in ('_d', '_h', '_s'):
        idx = stem.rfind(tag)
        if idx > 0 and stem[idx + 2:].isdigit():
            return False
    return True


def is_real_hw_image_path(img_path: str | Path) -> bool:
    """True when ``img_path`` is a real handwritten strip (``hw-*``)."""
    return Path(img_path).name.lower().startswith('hw-')


def _stream_seed(base_seed: int, epoch: int, key: str, salt: str = 'eaug') -> int:
    """Deterministic 64-bit integer seed derived purely from (seed, epoch, key)."""
    payload = f'{int(base_seed)}:{int(epoch)}:{salt}:{key}'.encode('utf-8')
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, 'little', signed=False)


def epoch_seed(base_seed: int, epoch: int) -> int:
    """Deterministic uint32 seed for ``photo-degrade --seed`` on a given epoch."""
    return _stream_seed(base_seed, epoch, 'photo-degrade', salt='redraw') & 0xFFFFFFFF


def _gaussian_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(2.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (xs / max(sigma, 1e-4)) ** 2)
    return k / k.sum()


def _batched_separable_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on ``[B, 3, H, W]`` float tensor in 2 convs."""
    k = _gaussian_kernel_1d(sigma, x.device, x.dtype)
    r = k.numel() // 2
    kh = k.view(1, 1, 1, -1).expand(3, 1, 1, -1)
    kv = k.view(1, 1, -1, 1).expand(3, 1, -1, 1)
    out = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), kh, groups=3)
    return F.conv2d(F.pad(out, (0, 0, r, r), mode='reflect'), kv, groups=3)


def _smooth_noise_field(
    m: int, h: int, w: int, gh: int, gw: int,
    dev: torch.device, gen: torch.Generator,
) -> torch.Tensor:
    """Bilinear-upsampled low-frequency 2D field ``[M, 1, H, W]`` in ``[0, 1]``."""
    coarse = torch.empty((m, 1, gh, gw), device=dev, dtype=torch.float32).uniform_(
        0.0, 1.0, generator=gen)
    return F.interpolate(coarse, size=(h, w), mode='bilinear', align_corners=False)


def _augment_batch_01(
    x: torch.Tensor,
    rngs: list[random.Random],
    dev: torch.device,
    gen: torch.Generator,
    s: float,
    geom_strength: float = 1.0,
    staffb: torch.Tensor | None = None,
    optics_profile: str = 'default',
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply rich, vectorized physical paper + ink + usage + camera + geometry aug."""
    m, _, h, w = x.shape
    gx, gy = _device_grid(dev, h, w)  # [1, 1, W], [1, H, 1]
    gx4 = gx.unsqueeze(0)             # [1, 1, 1, W]
    gy4 = gy.unsqueeze(0)             # [1, 1, H, 1]
    _, _, luma = _device_consts(dev)
    sc = _stage_consts(dev)
    gs = float(geom_strength) * s
    cal = (optics_profile == 'calibrated')

    # ------------------------------------------------------------------
    # Sample all per-sample scalar parameters in Python first, then ship
    # them to GPU in ONE single H2D transfer (avoids ~20 WDDM syncs/batch).
    # ------------------------------------------------------------------
    # 3b washout sampled early so fade/thin/shadow/blur compounding caps
    # know which samples have heavy washout (wash_floor > 0.40).
    wash_floor = [0.0] * m
    wash_ceil = [1.0] * m
    if cal:
        for j, rng in enumerate(rngs):
            if rng.random() < 0.45:
                wf = min(0.58, rng.uniform(0.28, 0.56) * s)
                wc = max(wf + 0.24, rng.uniform(0.82, 0.95))
                wash_floor[j] = wf
                wash_ceil[j] = min(1.0, wc)

    bleed_amt = [rng.uniform(0.03, 0.11) * s if rng.random() < 0.45 else 0.0 for rng in rngs]

    morph_mode, morph_amt = [], []
    for j, rng in enumerate(rngs):
        r = rng.random()
        if r < 0.25:
            morph_mode.append(1)  # bleed/thicken
            morph_amt.append(rng.uniform(0.12, 0.32) * s)
        elif r < 0.45 and not (cal and wash_floor[j] > 0.40):
            morph_mode.append(2)  # fade/thin (skip on heavy washout to protect contrast)
            morph_amt.append(rng.uniform(0.10, 0.28) * s)
        else:
            morph_mode.append(0)
            morph_amt.append(0.0)

    fade_p = 0.65 if cal else 0.55
    fade_hi = 0.50 if cal else 0.36
    fade_amt = []
    for j, rng in enumerate(rngs):
        if rng.random() < fade_p:
            fv = rng.uniform(0.10, fade_hi) * s
            fade_amt.append(min(fv, 0.15) if (cal and wash_floor[j] > 0.40) else fv)
        else:
            fade_amt.append(0.0)
    ig_amt = [rng.uniform(0.08, 0.22) * s if (cal and rng.random() < 0.45) else 0.0 for rng in rngs]
    band_amt = [rng.uniform(0.02, 0.07) * s if rng.random() < 0.35 else 0.0 for rng in rngs]
    age_amt = [rng.uniform(0.06, 0.24) * s if rng.random() < 0.70 else 0.0 for rng in rngs]
    fiber_amt = [rng.uniform(0.015, 0.045) * s if rng.random() < 0.55 else 0.0 for rng in rngs]

    stain_p = 0.45 if cal else 0.35
    stain_lo, stain_hi = (0.12, 0.34) if cal else (0.08, 0.24)
    st_cx, st_cy, st_rad, st_asp, st_int = [], [], [], [], []
    for rng in rngs:
        if rng.random() < stain_p:
            st_cx.append(rng.uniform(-0.8, 0.8))
            st_cy.append(rng.uniform(-0.7, 0.7))
            st_rad.append(rng.uniform(0.12, 0.38))
            st_asp.append(rng.uniform(0.35, 0.90))
            st_int.append(rng.uniform(stain_lo, stain_hi) * s)
        else:
            st_cx.append(0.0); st_cy.append(0.0); st_rad.append(0.2)
            st_asp.append(1.0); st_int.append(0.0)

    # 8b desk border params (pad-and-scale in Stage C -> clefs never erased).
    dk_lw, dk_rw, dk_tl, dk_tr, dk_hl, dk_hr, dk_r, dk_g, dk_b = (
        [0.0] * m, [0.0] * m, [0.0] * m, [0.0] * m,
        [0.0] * m, [0.0] * m, [0.1] * m, [0.1] * m, [0.1] * m,
    )
    has_desk = False
    if cal and w >= 32:
        for j, rng in enumerate(rngs):
            if rng.random() < 0.45:
                side = rng.choice((-1, 1, 2))
                lw = rng.uniform(0.015, 0.045) if side in (-1, 2) else 0.0
                rw = rng.uniform(0.015, 0.045) if side in (1, 2) else 0.0
                dk_lw[j] = lw
                dk_rw[j] = rw
                dk_tl[j] = rng.uniform(-min(0.008, lw), min(0.008, lw)) if lw > 0.0 else 0.0
                dk_tr[j] = rng.uniform(-min(0.008, rw), min(0.008, rw)) if rw > 0.0 else 0.0
                dk_hl[j] = 1.0 if lw > 0.0 else 0.0
                dk_hr[j] = 1.0 if rw > 0.0 else 0.0
                base = rng.uniform(0.06, 0.22)
                dk_r[j] = base * rng.uniform(0.90, 1.15)
                dk_g[j] = base * rng.uniform(0.85, 1.05)
                dk_b[j] = base * rng.uniform(0.75, 1.00)
                has_desk = True

    crump_amt = [rng.uniform(0.08, 0.26) * s if rng.random() < 0.55 else 0.0 for rng in rngs]
    cr_lx, cr_ly = [], []
    for rng in rngs:
        ang = rng.uniform(0.0, 2.0 * math.pi)
        cr_lx.append(math.cos(ang)); cr_ly.append(math.sin(ang))

    fl_ca, fl_sa, fl_pos, fl_wf, fl_amp, fl_kink = [], [], [], [], [], []
    for rng in rngs:
        if rng.random() < 0.50:
            ang = rng.uniform(-0.55, 0.55)
            fl_ca.append(math.cos(ang)); fl_sa.append(math.sin(ang))
            fl_pos.append(rng.uniform(-0.75, 0.75))
            fl_wf.append(rng.uniform(0.008, 0.025))
            fl_amp.append(rng.uniform(0.08, 0.24) * s * rng.choice((-1.0, 1.0)))
            fl_kink.append(rng.uniform(1.2, 3.2) * gs * rng.choice((-1.0, 1.0)))
        else:
            fl_ca.append(1.0); fl_sa.append(0.0); fl_pos.append(0.0)
            fl_wf.append(0.02); fl_amp.append(0.0); fl_kink.append(0.0)

    k1_l, k2_l, lcx_l, lcy_l, rot_l, bend_a, bend_f, bend_p, kx_l = (
        [0.0] * m, [0.0] * m, [0.0] * m, [0.0] * m,
        [0.0] * m, [0.0] * m, [1.0] * m, [0.0] * m, [0.0] * m,
    )
    if gs > 0.0:
        for j, rng in enumerate(rngs):
            lm = rng.random()
            if lm < 0.28:
                k1_l[j] = rng.uniform(0.015, 0.055) * gs
                k2_l[j] = rng.uniform(0.0, 0.020) * gs
            elif lm < 0.48:
                k1_l[j] = -rng.uniform(0.015, 0.045) * gs
                k2_l[j] = -rng.uniform(0.0, 0.015) * gs
            elif lm < 0.68:
                sign = rng.choice((-1.0, 1.0))
                k1_l[j] = sign * rng.uniform(0.025, 0.060) * gs
                k2_l[j] = -sign * rng.uniform(0.025, 0.055) * gs
            lcx_l[j] = rng.uniform(-0.45, 0.45)
            lcy_l[j] = rng.uniform(-0.95, 0.95)
            if rng.random() < 0.60:
                rot_l[j] = math.radians(rng.uniform(0.35, 1.55) * gs * rng.choice((-1.0, 1.0)))
            bend_p_thr = 0.70 if cal else 0.55
            bend_lo, bend_hi = (1.8, 5.2) if cal else (1.2, 3.8)
            if rng.random() < bend_p_thr:
                bend_a[j] = rng.uniform(bend_lo, bend_hi) * gs * rng.choice((-1.0, 1.0))
                bend_f[j] = rng.uniform(0.5, 1.6) * math.pi
                bend_p[j] = rng.uniform(0.0, 2.0 * math.pi)
            if cal and rng.random() < 0.55:
                kx_l[j] = rng.uniform(0.04, 0.06) * s * rng.choice((-1.0, 1.0))

    # 11. Lighting params (with washout x shadow compounding cap).
    ax_l, ay_l, cx_l, cy_l, vig_l = [], [], [], [], []
    spine_side, spine_amt = [], []
    sh_ca, sh_sa, sh_pos, sh_soft, sh_amt = [], [], [], [], []
    grad_p = 0.88 if cal else 0.80
    ax_lim = 0.24 if cal else 0.14
    ay_lim = 0.16 if cal else 0.10
    vig_p = 0.70 if cal else 0.55
    vig_lo, vig_hi = (0.12, 0.38) if cal else (0.05, 0.22)
    sh_p = 0.65 if cal else 0.45
    sh_lo, sh_hi = (0.18, 0.52) if cal else (0.12, 0.34)
    sh_soft_lo = 2.5 if cal else 4.0
    for j, rng in enumerate(rngs):
        heavy_wash = (cal and wash_floor[j] > 0.40)
        if rng.random() < grad_p:
            ax_l.append(rng.uniform(-ax_lim, ax_lim) * s)
            ay_l.append(rng.uniform(-ay_lim, ay_lim) * s)
            if rng.random() < vig_p:
                cx_l.append(rng.uniform(-0.4, 0.4))
                cy_l.append(rng.uniform(-0.4, 0.4))
                v_amt = rng.uniform(vig_lo, vig_hi) * s
                vig_l.append(min(v_amt, 0.20) if heavy_wash else v_amt)
            else:
                cx_l.append(0.0); cy_l.append(0.0); vig_l.append(0.0)
        else:
            ax_l.append(0.0); ay_l.append(0.0)
            cx_l.append(0.0); cy_l.append(0.0); vig_l.append(0.0)
        if rng.random() < 0.40:
            spine_side.append(rng.choice((-1.0, 1.0)))
            sp_amt = rng.uniform(0.18, 0.42) * s
            spine_amt.append(min(sp_amt, 0.22) if heavy_wash else sp_amt)
        else:
            spine_side.append(1.0); spine_amt.append(0.0)
        if rng.random() < sh_p:
            ang = rng.uniform(0.0, 2.0 * math.pi)
            sh_ca.append(math.cos(ang)); sh_sa.append(math.sin(ang))
            sh_pos.append(rng.uniform(-0.5, 0.6))
            sh_soft.append(rng.uniform(sh_soft_lo, 14.0))
            s_amt = rng.uniform(sh_lo, sh_hi) * s
            sh_amt.append(min(s_amt, 0.25) if heavy_wash else s_amt)
        else:
            sh_ca.append(1.0); sh_sa.append(0.0); sh_pos.append(0.0)
            sh_soft.append(8.0); sh_amt.append(0.0)
    ramp_lo_l = [0.58 if (cal and wash_floor[j] > 0.40) else (0.26 if cal else 0.42) for j in range(m)]

    # 12. WB + exposure/contrast/gamma.
    tr_l, tg_l, tb_l, gamma_l, contr_l, bright_l = [], [], [], [], [], []
    for j, rng in enumerate(rngs):
        heavy_wash = (cal and wash_floor[j] > 0.40)
        if rng.random() < 0.70:
            tr_l.append(1.0 + rng.uniform(-0.07, 0.07) * s)
            tg_l.append(1.0 + rng.uniform(-0.04, 0.04) * s)
            tb_l.append(1.0 + rng.uniform(-0.10, 0.07) * s)
        else:
            tr_l.append(1.0); tg_l.append(1.0); tb_l.append(1.0)
        if rng.random() < 0.80:
            gm = math.exp(rng.uniform(-0.22, 0.22) * s)
            ct = 1.0 + rng.uniform(-0.16, 0.16) * s
            br = rng.uniform(-0.05, 0.05) * s
            if heavy_wash:
                gm = max(0.92, min(1.10, gm))
                ct = max(0.96, ct)
            gamma_l.append(gm); contr_l.append(ct); bright_l.append(br)
        else:
            gamma_l.append(1.0); contr_l.append(1.0); bright_l.append(0.0)

    # 16. Sensor noise level (sampled here so it rides the same single H2D transfer).
    noise_l = [rng.uniform(0.008, 0.032) * s if rng.random() < 0.80 else 0.0 for rng in rngs]

    # Single packed H2D transfer [59, M] for all scalar per-sample parameters!
    P = torch.tensor([
        bleed_amt, morph_amt, fade_amt, ig_amt, wash_floor, wash_ceil,   # 0..5
        band_amt, age_amt, fiber_amt,                                     # 6..8
        st_cx, st_cy, st_rad, st_asp, st_int,                             # 9..13
        dk_lw, dk_rw, dk_tl, dk_tr, dk_hl, dk_hr, dk_r, dk_g, dk_b,       # 14..22
        crump_amt, cr_lx, cr_ly,                                          # 23..25
        fl_ca, fl_sa, fl_pos, fl_wf, fl_amp, fl_kink,                     # 26..31
        k1_l, k2_l, lcx_l, lcy_l, rot_l, bend_a, bend_f, bend_p, kx_l,    # 32..40
        ax_l, ay_l, cx_l, cy_l, vig_l, spine_side, spine_amt,             # 41..47
        sh_ca, sh_sa, sh_pos, sh_soft, sh_amt, ramp_lo_l,                 # 48..53
        tr_l, tg_l, tb_l, gamma_l, contr_l, bright_l,                     # 54..59
        noise_l,                                                          # 60
    ], device=dev, dtype=torch.float32)

    def _p(k: int) -> torch.Tensor:
        return P[k].view(m, 1, 1, 1)

    # ------------------------------------------------------------------
    # STAGE A: Ink-aware degradation
    # ------------------------------------------------------------------
    # 1. Reverse-side ink bleed-through (fast 3x3 box blur on flipped ghost).
    if any(v > 0.0 for v in bleed_amt):
        ghost = torch.flip(x, dims=[3])
        if w >= 16:
            ghost = torch.roll(ghost, shifts=max(4, w // 37), dims=3)
        ghost = F.avg_pool2d(ghost, kernel_size=3, stride=1, padding=1)
        x.mul_(1.0 - _p(0) * (1.0 - ghost))

    # 2. Ink morphology: ink spread/bleed (min_pool) OR faint thinning (max_pool).
    bleed_rows = [j for j, mm in enumerate(morph_mode) if mm == 1]
    thin_rows = [j for j, mm in enumerate(morph_mode) if mm == 2]
    if bleed_rows:
        idx = torch.tensor(bleed_rows, device=dev, dtype=torch.long)
        sub = x.index_select(0, idx)
        eroded = -F.max_pool2d(-sub, kernel_size=3, stride=1, padding=1)
        a = P[1].index_select(0, idx).view(-1, 1, 1, 1)
        x.index_copy_(0, idx, sub * (1.0 - a) + eroded * a)
    if thin_rows:
        idx = torch.tensor(thin_rows, device=dev, dtype=torch.long)
        sub = x.index_select(0, idx)
        dilated = F.max_pool2d(sub, kernel_size=3, stride=1, padding=1)
        a = P[1].index_select(0, idx).view(-1, 1, 1, 1)
        x.index_copy_(0, idx, sub * (1.0 - a) + dilated * a)

    # 3. Spatial ink fading / toner dropout + 3a. iron-gall/sepia ink tint (calibrated).
    has_fade = any(v > 0.0 for v in fade_amt)
    has_ig = any(v > 0.0 for v in ig_amt)
    patch_cache: torch.Tensor | None = None
    if has_fade or has_ig:
        luma_map = (x * luma.unsqueeze(0)).sum(dim=1, keepdim=True)
        ink_mask = ((0.82 - luma_map) / 0.65).clamp_(0.0, 1.0)
        if has_fade:
            patch_cache = _smooth_noise_field(m, h, w, 4, 10, dev, gen)
            fine = _smooth_noise_field(m, h, w, max(8, h // 8), max(16, w // 16), dev, gen)
            dropout = (0.65 * patch_cache + 0.35 * fine).pow_(1.6)
            lift = _p(2) * ink_mask * dropout
            x.add_((0.86 - x).clamp_(0.0, 1.0) * lift)
        if has_ig:
            w_ig = _p(3) * ink_mask
            x.mul_(1.0 - w_ig).add_(w_ig * sc['ink_tint'])

    # 3b. Extreme ink washout / faded gray toner (calibrated heavy-optics profile).
    if cal and any(v > 0.0 for v in wash_floor):
        wf_t = _p(4)
        wc_t = _p(5)
        base_mod = patch_cache if patch_cache is not None else _smooth_noise_field(m, h, w, 3, 6, dev, gen)
        wf_map = (wf_t * (0.85 + 0.20 * base_mod)).clamp_(0.0, 0.60)
        span_map = (wc_t - wf_map).clamp_(min=0.22)
        active_w = (wf_t > 0.0).float()
        x.mul_(1.0 - active_w + active_w * span_map).add_(active_w * wf_map)

    # 4. Photocopy / printer vertical toner banding streaks.
    if any(v > 0.0 for v in band_amt):
        streak_x = F.interpolate(
            torch.empty((m, 1, 1, max(16, w // 12)), device=dev).uniform_(-1.0, 1.0, generator=gen),
            size=(1, w), mode='bilinear', align_corners=False)
        x.mul_(1.0 - _p(6) * streak_x.clamp_(0.0, 1.0))

    # ------------------------------------------------------------------
    # STAGE B: Real paper aging, fiber grain, foxing, stains & usage
    # ------------------------------------------------------------------
    # 5. Paper yellowing, mottling & foxing spots (warm aged paper).
    if any(v > 0.0 for v in age_amt):
        mottle = _smooth_noise_field(m, h, w, 5, 12, dev, gen)
        fox_raw = _smooth_noise_field(m, h, w, max(8, h // 14), max(16, w // 14), dev, gen)
        fox = ((fox_raw - 0.78) / 0.22).clamp_(0.0, 1.0).square_()
        a_t = _p(7)
        x.mul_(1.0 - a_t * (0.45 + 0.55 * mottle) * sc['sepia'] - (a_t * 0.9) * fox * sc['fox_col'])

    # 6. Paper fiber micro-texture (anisotropic fine grain).
    if any(v > 0.0 for v in fiber_amt):
        fh = max(16, h // 2)
        fw = max(32, w // 6)
        fib = _smooth_noise_field(m, h, w, fh, fw, dev, gen) - 0.5
        x.mul_(1.0 + _p(8) * fib)

    # 7. Coffee / water ring stains (translucent amber fill + darker dried rim).
    if any(v > 0.0 for v in st_int):
        wob = (_smooth_noise_field(m, h, w, 6, 12, dev, gen) - 0.5) * 0.08
        dist = torch.sqrt((gx4 - _p(9)).square() + ((gy4 - _p(10)) * _p(12)).square() + 1e-6) + wob
        interior = torch.sigmoid((_p(11) - dist) * 28.0)
        rim = torch.exp(-((dist - _p(11)) / 0.025).square())
        x.mul_(1.0 - _p(13) * (0.45 * interior + 0.85 * rim) * sc['stain_col'])

    # 8. Musician usage: translucent highlighter band OR faint aged Scotch tape.
    hi_p = 0.24 if cal else 0.20
    tape_p = 0.46 if cal else 0.38
    hi_pal = sc['hi_cal'] if cal else sc['hi_def']
    tape_col = sc['tape_cal'] if cal else sc['tape_def']
    for j, rng in enumerate(rngs):
        u = rng.random()
        if u < hi_p:
            x0 = int(rng.uniform(0.05, 0.65) * w)
            x1 = min(w, x0 + int(rng.uniform(0.08, 0.28) * w))
            y0 = int(rng.uniform(0.08, 0.65) * h)
            y1 = min(h, y0 + int(rng.uniform(0.14, 0.30) * h))
            x[j, :, y0:y1, x0:x1].mul_(hi_pal[rng.randrange(3)])
        elif u < tape_p:
            x0 = int(rng.uniform(0.0, 0.82) * w)
            x1 = min(w, x0 + int(rng.uniform(0.08, 0.18) * w))
            y0 = 0 if rng.random() < 0.5 else max(0, h - int(0.22 * h))
            y1 = min(h, y0 + int(rng.uniform(0.12, 0.24) * h))
            x[j, :, y0:y1, x0:x1].mul_(tape_col)

    # ------------------------------------------------------------------
    # STAGE C: Paper crumpling, fold creases, coupled 3D geometry & shadows
    # ------------------------------------------------------------------
    px_to_nx = 2.0 / max(float(w - 1), 1.0)
    px_to_ny = 2.0 / max(float(h - 1), 1.0)
    dx_grid = torch.zeros((m, 1, h, w), device=dev, dtype=torch.float32)
    dy_grid = torch.zeros((m, 1, h, w), device=dev, dtype=torch.float32)
    has_geom = False

    # 8b pad-and-scale horizontal compression into [desk_L, desk_R] so clefs survive.
    desk_L: torch.Tensor | None = None
    desk_R: torch.Tensor | None = None
    if has_desk:
        desk_L = -1.0 + 2.0 * _p(14) + _p(16) * gy4
        desk_R = 1.0 - 2.0 * _p(15) + _p(17) * gy4
        x_page = -1.0 + 2.0 * (gx4 - desk_L) / (desk_R - desk_L).clamp_(min=0.5)
        dx_grid.add_(x_page - gx4)
        has_geom = True

    # 9. Crumpling / surface waviness.
    if any(v > 0.0 for v in crump_amt):
        h1 = _smooth_noise_field(m, h, w, 4, 8, dev, gen)
        h2 = _smooth_noise_field(m, h, w, 9, 18, dev, gen)
        hf = 0.65 * h1 + 0.35 * h2
        dhx = F.pad(hf[:, :, :, 1:] - hf[:, :, :, :-1], (0, 1, 0, 0))
        dhy = F.pad(hf[:, :, 1:, :] - hf[:, :, :-1, :], (0, 0, 0, 1))
        c_t = _p(23)
        shade = (dhx * _p(24) + dhy * _p(25)) * 18.0
        x.mul_((1.0 + c_t * shade).clamp_(0.72, 1.18))
        if gs > 0.0:
            dx_grid.add_(dhx * (c_t * 28.0 * gs * px_to_nx))
            dy_grid.add_(dhy * (c_t * 24.0 * gs * px_to_ny))
            has_geom = True

    # 10. Sharp fold / crease lines.
    if any(v != 0.0 for v in fl_amp):
        ca = _p(26)
        signed_d = (gx4 * ca + gy4 * _p(27) - _p(28)) / _p(29)
        bevel = signed_d * torch.exp(-0.5 * signed_d.square())
        x.mul_((1.0 + _p(30) * bevel).clamp_(0.68, 1.20))
        if gs > 0.0:
            ridge = torch.exp(-0.5 * (signed_d * 0.65).square())
            kink = _p(31)
            dx_grid.add_(ridge * kink * ca * px_to_nx)
            dy_grid.add_(ridge * kink * (0.75 * px_to_ny))
            has_geom = True

    # 10b. Lens distortion + residual rotation + page-bend wave + keystone shear.
    if gs > 0.0:
        if any(v != 0.0 for v in k1_l) or any(v != 0.0 for v in k2_l):
            strip_h_frac = 0.22
            px = gx4 - _p(34)
            py = gy4 * strip_h_frac - _p(35)
            r2 = px.square() + py.square()
            rad_fac = _p(32) * r2 + _p(33) * r2.square()
            dlx = px * rad_fac
            dly = (py * rad_fac) / strip_h_frac
            dlx = dlx - dlx.mean(dim=(2, 3), keepdim=True)
            dly = (dly - dly.mean(dim=(2, 3), keepdim=True)).clamp_(-0.08, 0.08)
            dx_grid.add_(dlx)
            dy_grid.add_(dly)
            has_geom = True

        if any(v != 0.0 for v in rot_l):
            th = _p(36)
            ct = torch.cos(th) - 1.0
            st = torch.sin(th)
            asp = float(w) / max(float(h), 1.0)
            dx_grid.add_(gx4 * ct - (gy4 / asp) * st)
            dy_grid.add_((gx4 * asp) * st + gy4 * ct)
            has_geom = True

        if any(v != 0.0 for v in bend_a):
            dy_grid.add_(torch.sin(gx4 * _p(38) + _p(39)) * (_p(37) * px_to_ny))
            has_geom = True

        if any(v != 0.0 for v in kx_l):
            dx_grid.add_(0.5 * _p(40) * (gx4.square() - 1.0))
            has_geom = True

    if has_geom:
        base_x = gx4.expand(m, 1, h, w)
        base_y = gy4.expand(m, 1, h, w)
        grid = torch.cat([base_x + dx_grid, base_y + dy_grid], dim=1).permute(0, 2, 3, 1)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        if staffb is not None:
            staffb = F.grid_sample(
                staffb.float(), grid, mode='bilinear',
                padding_mode='zeros', align_corners=False).to(staffb.dtype)

    # Paint dark desk border outside [desk_L, desk_R] AFTER pad-and-scale warp.
    if has_desk and desk_L is not None and desk_R is not None:
        dc_t = P[20:23].transpose(0, 1).view(m, 3, 1, 1)
        sharp = max(40.0, float(w) * 0.25)
        m_left = _p(18) * torch.sigmoid((desk_L - gx4) * sharp)
        m_right = _p(19) * torch.sigmoid((gx4 - desk_R) * sharp)
        desk_m = (m_left + m_right).clamp_(0.0, 1.0)
        x.mul_(1.0 - desk_m).add_(dc_t * desk_m)

    # 11. Lighting (gradient + vignette + book-spine + directional shadow).
    ramp = torch.ones((m, 1, h, w), device=dev, dtype=torch.float32)
    has_grad = any(v != 0.0 for v in ax_l) or any(v != 0.0 for v in ay_l)
    has_vig = any(v != 0.0 for v in vig_l)
    has_spine = any(v > 0.0 for v in spine_amt)
    has_sh = any(v > 0.0 for v in sh_amt)
    if has_grad or has_vig or has_spine or has_sh:
        if has_grad:
            ramp.add_(_p(41) * gx4 + _p(42) * gy4)
        if has_vig:
            r2 = (gx4 - _p(43)).square() + (gy4 - _p(44)).square()
            ramp.mul_(1.0 - _p(45) * r2.clamp_(0.0, 1.5))
        if has_spine:
            edge_dist = (1.0 - _p(46) * gx4) * 0.5
            ramp.mul_(1.0 - _p(47) * torch.exp(-edge_dist * 10.0))
        if has_sh:
            shadow = torch.sigmoid((gx4 * _p(48) + gy4 * _p(49) - _p(50)) * _p(51))
            ramp.mul_(1.0 - _p(52) * shadow)
        ramp = torch.maximum(ramp, _p(53)).clamp_(max=1.25)
    x.mul_(ramp)

    # ------------------------------------------------------------------
    # STAGE D: Real smartphone camera optics
    # ------------------------------------------------------------------
    # 12. White balance + exposure / contrast / gamma.
    x.mul_(P[54:57].transpose(0, 1).view(m, 3, 1, 1))
    x.clamp_(1e-4, 1.0).pow_(_p(57))
    means = x.mean(dim=(1, 2, 3), keepdim=True)
    x.sub_(means).mul_(_p(58)).add_(means + _p(59)).clamp_(0.0, 1.0)

    # 13. Lateral chromatic aberration (horizontal-only R/B shift -> 0 vertical drift).
    for j, rng in enumerate(rngs):
        if rng.random() < 0.35 and w >= 16:
            shift = rng.choice((-1, 1))
            x[j, 0] = torch.roll(x[j, 0], shifts=shift, dims=1)
            x[j, 2] = torch.roll(x[j, 2], shifts=-shift, dims=1)

    # 14. Focus optics: uniform defocus, tilt-shift, or unsharp mask.
    focus_mode, focus_val = [], []
    p_blur = 0.42 if cal else 0.28
    p_tilt = 0.70 if cal else 0.48
    p_sharp = 0.80 if cal else 0.60
    blur_lo, blur_hi = (0.55, 1.85) if cal else (0.35, 1.05)
    tilt_lo, tilt_hi = (1.05, 2.20) if cal else (0.85, 1.45)
    for j, rng in enumerate(rngs):
        heavy_wash = (cal and wash_floor[j] > 0.40)
        pf = rng.random()
        if pf < p_blur:
            focus_mode.append(1)
            sig_v = rng.uniform(blur_lo, blur_hi) * s
            focus_val.append(min(sig_v, 1.2) if heavy_wash else sig_v)
        elif pf < p_tilt:
            focus_mode.append(2)
            sig_v = rng.uniform(tilt_lo, tilt_hi) * s
            focus_val.append(min(sig_v, 1.2) if heavy_wash else sig_v)
        elif pf < p_sharp:
            focus_mode.append(3)
            focus_val.append(rng.uniform(0.20, 0.55) * s)
        else:
            focus_mode.append(0); focus_val.append(0.0)

    # Split blur/tilt groups by heavy_wash so batch sigma averaging never leaks
    # sigma > 1.2 onto an extreme-washout sample.
    blur_wash = [j for j, fm in enumerate(focus_mode) if fm == 1 and cal and wash_floor[j] > 0.40]
    blur_norm = [j for j, fm in enumerate(focus_mode) if fm == 1 and not (cal and wash_floor[j] > 0.40)]
    tilt_wash = [j for j, fm in enumerate(focus_mode) if fm == 2 and cal and wash_floor[j] > 0.40]
    tilt_norm = [j for j, fm in enumerate(focus_mode) if fm == 2 and not (cal and wash_floor[j] > 0.40)]
    sharp_rows = [j for j, fm in enumerate(focus_mode) if fm == 3]
    for grp in (blur_wash, blur_norm):
        if grp:
            idx = torch.tensor(grp, device=dev, dtype=torch.long)
            sig = sum(focus_val[j] for j in grp) / len(grp)
            x.index_copy_(0, idx, _batched_separable_blur(x.index_select(0, idx), sig))
    for grp in (tilt_wash, tilt_norm):
        if grp:
            idx = torch.tensor(grp, device=dev, dtype=torch.long)
            sub = x.index_select(0, idx)
            sig = sum(focus_val[j] for j in grp) / len(grp)
            blurred = _batched_separable_blur(sub, sig)
            fx = torch.tensor([rngs[j].uniform(-0.7, 0.7) for j in grp],
                              device=dev).view(-1, 1, 1, 1)
            blend = ((gx4 - fx).abs() / 1.4).clamp_(0.0, 1.0).pow_(1.5)
            x.index_copy_(0, idx, sub * (1.0 - blend) + blurred * blend)
    if sharp_rows:
        idx = torch.tensor(sharp_rows, device=dev, dtype=torch.long)
        sub = x.index_select(0, idx)
        blurred = F.avg_pool2d(sub, kernel_size=3, stride=1, padding=1)
        amt = torch.tensor([focus_val[j] for j in sharp_rows],
                           device=dev, dtype=torch.float32).view(-1, 1, 1, 1)
        x.index_copy_(0, idx, (sub + amt * (sub - blurred)).clamp_(0.0, 1.0))

    # 15. Down-up camera/JPEG softness.
    scale_l = [
        rng.uniform(max(0.66, 1.0 - 0.30 * s), 0.94)
        if (rng.random() < 0.45 and h >= 16 and w >= 16) else 1.0
        for rng in rngs
    ]
    scale_rows = [j for j, sc_v in enumerate(scale_l) if sc_v < 0.999]
    if scale_rows:
        idx = torch.tensor(scale_rows, device=dev, dtype=torch.long)
        sc_mean = sum(scale_l[j] for j in scale_rows) / len(scale_rows)
        sh = max(8, int(round(h * sc_mean)))
        sw = max(8, int(round(w * sc_mean)))
        small = F.interpolate(x.index_select(0, idx), size=(sh, sw),
                              mode='bilinear', align_corners=False)
        x.index_copy_(0, idx, F.interpolate(small, size=(h, w),
                                            mode='bilinear', align_corners=False))

    # 16. Sensor shot + read Gaussian noise.
    if any(v > 0.0 for v in noise_l):
        noise = torch.empty_like(x).normal_(0.0, 1.0, generator=gen)
        x.add_(noise.mul_(_p(60)))

    return x.clamp_(0.0, 1.0), staffb


def augment_rgb_01(
    rgb: torch.Tensor,
    base_seed: int,
    epoch: int,
    sample_key: str,
    strength: float = 1.0,
    geom_strength: float = 1.0,
    optics_profile: str = 'default',
    heavy_optics: bool = False,
) -> torch.Tensor:
    """Apply deterministic physical paper/ink/camera/geometry aug to ``[3, H, W]``."""
    prof = _resolve_optics_profile(optics_profile, heavy_optics=heavy_optics)
    if strength <= 0.0:
        return rgb.clone()
    seed64 = _stream_seed(base_seed, epoch, sample_key)
    rng = random.Random(seed64)
    dev = rgb.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed64 & 0x7FFFFFFFFFFFFFFF)
    out, _ = _augment_batch_01(
        rgb.clone().float().unsqueeze(0), [rng], dev, gen,
        float(strength), geom_strength=float(geom_strength),
        optics_profile=prof)
    return out.squeeze(0)


def augment_standardized_batch(
    rgb_std: torch.Tensor,
    batch_pairs: list,
    base_seed: int,
    epoch: int,
    aug_p: float = 0.0,
    clean_p: float = 0.0,
    grayb: torch.Tensor | None = None,
    valid_widths: list[int] | None = None,
    strength: float = 1.0,
    geom_strength: float = 1.0,
    staffb: torch.Tensor | None = None,
    optics_profile: str = 'default',
    heavy_optics: bool = False,
    clean_anchor_frac: float | None = None,
):
    """Deterministically augment an ImageNet-standardized batch (vectorized on GPU).

    Supports both 4D ``grayb`` (``[B, 1, H, W]``, ``collate_hybrid`` native)
    and 3D ``grayb`` (``[B, H, W]``). When ``staffb`` (``[B, 1, H, W]``) is
    provided, applies the exact same geometric warp to ``staffb`` and returns
    ``(out_rgb, out_gray, out_staff)``; otherwise returns ``(out_rgb, out_gray)``.

    When ``optics_profile='calibrated'`` (or ``heavy_optics=True``), widens ink
    washout/fade, soft out-of-focus blur, dark shadows/vignettes, dark desk
    borders, and spline-wave geometry while keeping at least ``25%`` of clean
    anchors untouched (``clean_anchor_frac=0.25`` by default in calibrated mode).
    """
    prof = _resolve_optics_profile(optics_profile, heavy_optics=heavy_optics)
    if clean_anchor_frac is None:
        anchor_frac = CALIBRATED_CLEAN_ANCHOR_FRAC if prof == 'calibrated' else 0.0
    else:
        anchor_frac = max(0.0, min(1.0, float(clean_anchor_frac)))

    if (aug_p <= 0.0 and clean_p <= 0.0) or strength <= 0.0:
        return (rgb_std, grayb, staffb) if staffb is not None else (rgb_std, grayb)

    dev = rgb_std.device
    dtype = rgb_std.dtype
    mean, std, luma = _device_consts(dev)
    _, _, _, max_w = rgb_std.shape

    active_idx: list[int] = []
    active_rngs: list[random.Random] = []
    active_seeds: list[int] = []
    seen_keys: dict[str, int] = {}
    for i, pair in enumerate(batch_pairs):
        img_path = pair[0]
        base_key = Path(img_path).name
        occ = seen_keys.get(base_key, 0)
        seen_keys[base_key] = occ + 1
        key = base_key if occ == 0 else f'{base_key}#dup{occ}'

        is_clean = is_clean_image_path(img_path)
        is_hw = is_real_hw_image_path(img_path)
        prob = clean_p if is_clean else aug_p
        if prob <= 0.0:
            continue
        if is_clean and (not is_hw) and anchor_frac > 0.0:
            anchor_rng = random.Random(_stream_seed(base_seed, 0, key, salt='clean_anchor'))
            if anchor_rng.random() < anchor_frac:
                continue
        gate_rng = random.Random(_stream_seed(base_seed, epoch, key, salt='gate'))
        if gate_rng.random() >= prob:
            continue
        seed64 = _stream_seed(base_seed, epoch, key)
        active_idx.append(i)
        active_rngs.append(random.Random(seed64))
        active_seeds.append(seed64)

    if not active_idx:
        return (rgb_std, grayb, staffb) if staffb is not None else (rgb_std, grayb)

    idx_t = torch.tensor(active_idx, device=dev, dtype=torch.long)
    out_rgb = rgb_std.clone()
    out_gray = grayb.clone() if grayb is not None else None
    out_staff = staffb.clone() if staffb is not None else None

    x = out_rgb.index_select(0, idx_t).float().mul_(std).add_(mean).clamp_(0.0, 1.0)
    st_sub = out_staff.index_select(0, idx_t) if out_staff is not None else None
    batch_seed = _stream_seed(
        base_seed, epoch, '|'.join(str(s64) for s64 in active_seeds), salt='bphys')
    gen = torch.Generator(device=dev)
    gen.manual_seed(batch_seed & 0x7FFFFFFFFFFFFFFF)

    x, st_sub = _augment_batch_01(
        x, active_rngs, dev, gen, float(strength),
        geom_strength=float(geom_strength), staffb=st_sub,
        optics_profile=prof)

    x_std = ((x - mean.unsqueeze(0)) / std.unsqueeze(0)).to(dtype)
    if out_gray is not None:
        gray_4d = (out_gray.ndim == 4)
        x_gray = (x * luma.unsqueeze(0)).sum(dim=1, keepdim=gray_4d).to(out_gray.dtype)
    else:
        x_gray = None

    if valid_widths is None:
        out_rgb.index_copy_(0, idx_t, x_std)
        if out_gray is not None:
            out_gray.index_copy_(0, idx_t, x_gray)
        if out_staff is not None and st_sub is not None:
            out_staff.index_copy_(0, idx_t, st_sub)
    else:
        for j, i in enumerate(active_idx):
            w_i = max(1, min(int(valid_widths[i]), max_w))
            out_rgb[i, :, :, :w_i] = x_std[j, :, :, :w_i]
            if out_gray is not None:
                if out_gray.ndim == 4:
                    out_gray[i, :, :, :w_i] = x_gray[j, :, :, :w_i]
                else:
                    out_gray[i, :, :w_i] = x_gray[j, :, :w_i]
            if out_staff is not None and st_sub is not None:
                out_staff[i, :, :, :w_i] = st_sub[j, :, :, :w_i]

    if staffb is not None:
        return out_rgb, out_gray, out_staff
    return out_rgb, out_gray


def default_photo_degrade_bin() -> Path:
    """Locate the release ``photo-degrade`` binary relative to the repo root."""
    repo = Path(__file__).resolve().parents[2]
    exe = 'photo-degrade.exe' if os.name == 'nt' else 'photo-degrade'
    return repo / 'data-gen' / 'photo-degrade' / 'target' / 'release' / exe


def redraw_epoch_dir(
    clean_dir: str | Path,
    redraw_root: str | Path,
    epoch: int,
    base_seed: int,
    photo_degrade_bin: str | Path | None = None,
    train_width: int = 1024,
    cpu: bool = False,
) -> Path:
    """Materialize a fresh ``photo-degrade`` redraw pool for ``epoch``."""
    ep_dir = Path(redraw_root) / f'ep{int(epoch)}'
    done_marker = ep_dir / '.done'
    if done_marker.exists():
        return ep_dir

    ep_dir.mkdir(parents=True, exist_ok=True)
    bin_path = Path(photo_degrade_bin) if photo_degrade_bin else default_photo_degrade_bin()
    if not bin_path.exists():
        raise FileNotFoundError(f'photo-degrade binary not found: {bin_path}')

    seed = epoch_seed(base_seed, epoch)
    cmd = [
        str(bin_path),
        '--input', str(clean_dir),
        '--out-dir', str(ep_dir),
        '--format', 'jpeg',
        '--seed', str(seed),
        '--train-width', str(int(train_width)),
    ]
    if cpu:
        cmd.append('--cpu')
    subprocess.run(cmd, check=True)
    tmp = ep_dir / '.done.tmp'
    tmp.write_text(f'epoch={epoch}\nseed={seed}\n', encoding='utf-8')
    os.replace(tmp, done_marker)
    return ep_dir


def resolve_epoch_batch(
    batch_pairs: list[tuple[str, str, str]],
    epoch: int,
    redraw_root: str | Path | None,
) -> list[tuple[str, str, str]]:
    """Substitute degraded batch items with fresh ``<redraw_root>/ep<E>/`` files."""
    if not redraw_root:
        return batch_pairs
    ep_dir = Path(redraw_root) / f'ep{int(epoch)}'
    if not ep_dir.is_dir():
        return batch_pairs
    resolved = []
    for img_path, bek_path, stem in batch_pairs:
        if is_clean_image_path(img_path):
            resolved.append((img_path, bek_path, stem))
            continue
        p = Path(img_path)
        exact = ep_dir / p.name
        by_stem = ep_dir / f'{stem}.jpg'
        if exact.exists():
            resolved.append((str(exact), bek_path, stem))
        elif by_stem.exists():
            resolved.append((str(by_stem), bek_path, stem))
        else:
            resolved.append((img_path, bek_path, stem))
    return resolved


def _cli_bench() -> None:
    """Quick CLI benchmark/smoke check supporting ``--heavy-optics``."""
    import argparse
    import time

    ap = argparse.ArgumentParser(description='epoch_aug benchmark & smoke test')
    ap.add_argument('--heavy-optics', action='store_true',
                    help="Use optics_profile='calibrated'")
    ap.add_argument('--optics-profile', default='default',
                    choices=OPTICS_PROFILES)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--geom-strength', type=float, default=0.30)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    prof = _resolve_optics_profile(args.optics_profile, heavy_optics=args.heavy_optics)
    dev = torch.device(args.device)
    b, h, w = args.batch_size, args.height, args.width
    rgb = torch.randn((b, 3, h, w), device=dev, dtype=torch.float32)
    gray = torch.zeros((b, 1, h, w), device=dev, dtype=torch.float32)
    staff = torch.zeros((b, 1, h, w), device=dev, dtype=torch.float32)
    pairs = [(f'in/s{i}_d0.jpg', f'in/s{i}-bekern.txt', f's{i}') for i in range(b)]

    # Warmup
    for ep in range(3):
        augment_standardized_batch(
            rgb, pairs, base_seed=42, epoch=ep, aug_p=1.0,
            grayb=gray, staffb=staff, geom_strength=args.geom_strength,
            optics_profile=prof)
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)

    t0 = time.perf_counter()
    for ep in range(args.iters):
        augment_standardized_batch(
            rgb, pairs, base_seed=42, epoch=ep + 10, aug_p=1.0,
            grayb=gray, staffb=staff, geom_strength=args.geom_strength,
            optics_profile=prof)
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)
    ms = (time.perf_counter() - t0) * 1000.0 / max(args.iters, 1)
    print(f'epoch_aug profile={prof} dev={dev} B={b} {h}x{w} geom={args.geom_strength}: {ms:.2f} ms/batch')


if __name__ == '__main__':
    _cli_bench()
