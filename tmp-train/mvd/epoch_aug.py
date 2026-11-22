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

_CONST_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_GRID_CACHE: dict[tuple[torch.device, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


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


def is_clean_image_path(img_path: str | Path) -> bool:
    """True when ``img_path`` is a clean render (no ``_d*``/``_h*``/``_s*`` tag)."""
    stem = Path(img_path).stem
    for tag in ('_d', '_h', '_s'):
        idx = stem.rfind(tag)
        if idx > 0 and stem[idx + 2:].isdigit():
            return False
    return True


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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply rich, vectorized physical paper + ink + usage + camera + geometry aug."""
    m, _, h, w = x.shape
    gx, gy = _device_grid(dev, h, w)  # [1, 1, W], [1, H, 1]
    gx4 = gx.unsqueeze(0)             # [1, 1, 1, W]
    gy4 = gy.unsqueeze(0)             # [1, 1, H, 1]
    _, _, luma = _device_consts(dev)
    gs = float(geom_strength) * s

    # ------------------------------------------------------------------
    # STAGE A: Ink-aware degradation (bleed-through, ink fade/dropout,
    #          ink spread / over-exposed thinning, toner streaks)
    # ------------------------------------------------------------------
    # 1. Reverse-side ink bleed-through (ghosted horizontally-flipped copy).
    bleed_amt = [rng.uniform(0.03, 0.11) * s if rng.random() < 0.45 else 0.0 for rng in rngs]
    if any(v > 0.0 for v in bleed_amt):
        ghost = torch.flip(x, dims=[3])
        if w >= 16:
            # Slight horizontal offset so reverse-side staves don't lock onto front staves.
            ghost = torch.roll(ghost, shifts=max(4, w // 37), dims=3)
        ghost = _batched_separable_blur(ghost, 1.1)
        b_t = torch.tensor(bleed_amt, device=dev, dtype=torch.float32).view(m, 1, 1, 1)
        # Multiply-blend faint ghost ink onto the page.
        x.mul_(1.0 - b_t * (1.0 - ghost))

    # 2. Ink morphology: ink spread/bleed (min_pool) OR faint thinning (max_pool).
    morph_mode = []
    morph_amt = []
    for rng in rngs:
        r = rng.random()
        if r < 0.25:
            morph_mode.append(1)  # bleed/thicken
            morph_amt.append(rng.uniform(0.12, 0.32) * s)
        elif r < 0.45:
            morph_mode.append(2)  # fade/thin (photocopy/overexposed)
            morph_amt.append(rng.uniform(0.10, 0.28) * s)
        else:
            morph_mode.append(0)
            morph_amt.append(0.0)
    bleed_rows = [j for j, mm in enumerate(morph_mode) if mm == 1]
    thin_rows = [j for j, mm in enumerate(morph_mode) if mm == 2]
    if bleed_rows:
        idx = torch.tensor(bleed_rows, device=dev, dtype=torch.long)
        sub = x.index_select(0, idx)
        eroded = -F.max_pool2d(-sub, kernel_size=3, stride=1, padding=1)
        a = torch.tensor([morph_amt[j] for j in bleed_rows],
                         device=dev, dtype=torch.float32).view(-1, 1, 1, 1)
        x.index_copy_(0, idx, sub * (1.0 - a) + eroded * a)
    if thin_rows:
        idx = torch.tensor(thin_rows, device=dev, dtype=torch.long)
        sub = x.index_select(0, idx)
        dilated = F.max_pool2d(sub, kernel_size=3, stride=1, padding=1)
        a = torch.tensor([morph_amt[j] for j in thin_rows],
                         device=dev, dtype=torch.float32).view(-1, 1, 1, 1)
        x.index_copy_(0, idx, sub * (1.0 - a) + dilated * a)

    # 3. Spatial ink fading / toner dropout (targets dark ink pixels specifically).
    fade_amt = [rng.uniform(0.10, 0.36) * s if rng.random() < 0.55 else 0.0 for rng in rngs]
    if any(v > 0.0 for v in fade_amt):
        luma_map = (x * luma.unsqueeze(0)).sum(dim=1, keepdim=True)
        ink_mask = ((0.82 - luma_map) / 0.65).clamp_(0.0, 1.0)
        # Patchy toner-starvation field (smooth blobs + fine grain).
        patch = _smooth_noise_field(m, h, w, 4, 10, dev, gen)
        fine = _smooth_noise_field(m, h, w, max(8, h // 8), max(16, w // 16), dev, gen)
        dropout = (0.65 * patch + 0.35 * fine).pow_(1.6)
        f_t = torch.tensor(fade_amt, device=dev, dtype=torch.float32).view(m, 1, 1, 1)
        # Lift dark ink toward warm gray-white where dropout is high.
        lift = f_t * ink_mask * dropout
        x.add_((0.86 - x).clamp_(0.0, 1.0) * lift)

    # 4. Photocopy / printer vertical or horizontal toner banding streaks.
    band_amt = [rng.uniform(0.02, 0.07) * s if rng.random() < 0.35 else 0.0 for rng in rngs]
    if any(v > 0.0 for v in band_amt):
        # Vertical streaks along x (1D coarse noise broadcast over H).
        streak_x = F.interpolate(
            torch.empty((m, 1, 1, max(16, w // 12)), device=dev).uniform_(-1.0, 1.0, generator=gen),
            size=(1, w), mode='bilinear', align_corners=False)
        ba_t = torch.tensor(band_amt, device=dev, dtype=torch.float32).view(m, 1, 1, 1)
        x.mul_(1.0 - ba_t * streak_x.clamp_(0.0, 1.0))

    # ------------------------------------------------------------------
    # STAGE B: Real paper aging, fiber grain, foxing, stains & usage
    # ------------------------------------------------------------------
    # 5. Paper yellowing, mottling & foxing spots (warm aged paper).
    age_amt = [rng.uniform(0.06, 0.24) * s if rng.random() < 0.70 else 0.0 for rng in rngs]
    if any(v > 0.0 for v in age_amt):
        mottle = _smooth_noise_field(m, h, w, 5, 12, dev, gen)
        fox_raw = _smooth_noise_field(m, h, w, max(8, h // 14), max(16, w // 14), dev, gen)
        fox = ((fox_raw - 0.78) / 0.22).clamp_(0.0, 1.0).square_()
        a_t = torch.tensor(age_amt, device=dev, dtype=torch.float32).view(m, 1, 1, 1)
        # Sepia/cream paper tint: R almost untouched, G slightly darkened, B darkened most.
        sepia = torch.tensor([0.05, 0.16, 0.34], device=dev, dtype=torch.float32).view(1, 3, 1, 1)
        fox_col = torch.tensor([0.14, 0.30, 0.48], device=dev, dtype=torch.float32).view(1, 3, 1, 1)
        x.mul_(1.0 - a_t * (0.45 + 0.55 * mottle) * sepia - (a_t * 0.9) * fox * fox_col)

    # 6. Paper fiber micro-texture (anisotropic fine grain).
    fiber_amt = [rng.uniform(0.015, 0.045) * s if rng.random() < 0.55 else 0.0 for rng in rngs]
    if any(v > 0.0 for v in fiber_amt):
        fh = max(16, h // 2)
        fw = max(32, w // 6)  # stretched horizontally like laid paper fibers
        fib = _smooth_noise_field(m, h, w, fh, fw, dev, gen) - 0.5
        fb_t = torch.tensor(fiber_amt, device=dev, dtype=torch.float32).view(m, 1, 1, 1)
        x.mul_(1.0 + fb_t * fib)

    # 7. Coffee / water ring stains (translucent amber fill + darker dried rim).
    stain_params = []
    for rng in rngs:
        if rng.random() < 0.35:
            stain_params.append((
                rng.uniform(-0.8, 0.8),   # cx
                rng.uniform(-0.7, 0.7),   # cy
                rng.uniform(0.12, 0.38),  # radius
                rng.uniform(0.35, 0.90),  # y aspect (since W >> H)
                rng.uniform(0.08, 0.24) * s,  # intensity
            ))
        else:
            stain_params.append((0.0, 0.0, 0.2, 1.0, 0.0))
    if any(p[4] > 0.0 for p in stain_params):
        cx = torch.tensor([p[0] for p in stain_params], device=dev).view(m, 1, 1, 1)
        cy = torch.tensor([p[1] for p in stain_params], device=dev).view(m, 1, 1, 1)
        rad = torch.tensor([p[2] for p in stain_params], device=dev).view(m, 1, 1, 1)
        asp = torch.tensor([p[3] for p in stain_params], device=dev).view(m, 1, 1, 1)
        inten = torch.tensor([p[4] for p in stain_params], device=dev).view(m, 1, 1, 1)
        # Organic wobble on stain boundary using low-freq noise.
        wob = (_smooth_noise_field(m, h, w, 6, 12, dev, gen) - 0.5) * 0.08
        dist = torch.sqrt((gx4 - cx).square() + ((gy4 - cy) * asp).square() + 1e-6) + wob
        interior = torch.sigmoid((rad - dist) * 28.0)
        rim = torch.exp(-((dist - rad) / 0.025).square())
        stain_col = torch.tensor([0.12, 0.32, 0.56], device=dev).view(1, 3, 1, 1)
        x.mul_(1.0 - inten * (0.45 * interior + 0.85 * rim) * stain_col)

    # 8. Musician usage: translucent highlighter band OR faint aged Scotch tape / thumb smudge.
    for j, rng in enumerate(rngs):
        u = rng.random()
        if u < 0.20:
            # Translucent pastel highlighter across a partial measure span.
            x0 = int(rng.uniform(0.05, 0.65) * w)
            x1 = min(w, x0 + int(rng.uniform(0.08, 0.28) * w))
            y0 = int(rng.uniform(0.08, 0.65) * h)
            y1 = min(h, y0 + int(rng.uniform(0.14, 0.30) * h))
            # Yellow / pink / mint pastel multiply tint (preserves dark ink).
            pal = rng.choice(([0.98, 0.97, 0.68], [0.99, 0.80, 0.86], [0.78, 0.97, 0.82]))
            col = torch.tensor(pal, device=dev, dtype=torch.float32).view(3, 1, 1)
            x[j, :, y0:y1, x0:x1].mul_(col)
        elif u < 0.38:
            # Faint yellowed Scotch tape patch on left/right edge or top margin.
            x0 = int(rng.uniform(0.0, 0.82) * w)
            x1 = min(w, x0 + int(rng.uniform(0.08, 0.18) * w))
            y0 = 0 if rng.random() < 0.5 else max(0, h - int(0.22 * h))
            y1 = min(h, y0 + int(rng.uniform(0.12, 0.24) * h))
            tape = torch.tensor([0.96, 0.92, 0.78], device=dev, dtype=torch.float32).view(3, 1, 1)
            x[j, :, y0:y1, x0:x1].mul_(tape)

    # ------------------------------------------------------------------
    # STAGE C: Paper crumpling, fold creases, coupled 3D geometry & shadows
    # ------------------------------------------------------------------
    # Accumulate coupled pixel displacement in normalized grid coords [-1, 1].
    # dx_norm = 2 * dx_px / max(W-1, 1), dy_norm = 2 * dy_px / max(H-1, 1).
    px_to_nx = 2.0 / max(float(w - 1), 1.0)
    px_to_ny = 2.0 / max(float(h - 1), 1.0)
    dx_grid = torch.zeros((m, 1, h, w), device=dev, dtype=torch.float32)
    dy_grid = torch.zeros((m, 1, h, w), device=dev, dtype=torch.float32)
    has_geom = False

    # 9. Crumpling / surface waviness (coupled heightfield shading + pixel displacement).
    crump_amt = [rng.uniform(0.08, 0.26) * s if rng.random() < 0.55 else 0.0 for rng in rngs]
    if any(v > 0.0 for v in crump_amt):
        h1 = _smooth_noise_field(m, h, w, 4, 8, dev, gen)
        h2 = _smooth_noise_field(m, h, w, 9, 18, dev, gen)
        hf = 0.65 * h1 + 0.35 * h2
        dhx = F.pad(hf[:, :, :, 1:] - hf[:, :, :, :-1], (0, 1, 0, 0))
        dhy = F.pad(hf[:, :, 1:, :] - hf[:, :, :-1, :], (0, 0, 0, 1))
        lx = torch.tensor([math.cos(rng.uniform(0, 2 * math.pi)) for rng in rngs],
                          device=dev).view(m, 1, 1, 1)
        ly = torch.tensor([math.sin(rng.uniform(0, 2 * math.pi)) for rng in rngs],
                          device=dev).view(m, 1, 1, 1)
        shade = (dhx * lx + dhy * ly) * 18.0
        c_t = torch.tensor(crump_amt, device=dev, dtype=torch.float32).view(m, 1, 1, 1)
        x.mul_((1.0 + c_t * shade).clamp_(0.72, 1.18))
        if gs > 0.0:
            # Physically bend staff lines & noteheads along the exact crumple slopes!
            dx_grid.add_(dhx * (c_t * 28.0 * gs * px_to_nx))
            dy_grid.add_(dhy * (c_t * 24.0 * gs * px_to_ny))
            has_geom = True

    # 10. Sharp fold / crease lines (coupled bevel shading + physical crease kink).
    fold_params = []
    for rng in rngs:
        if rng.random() < 0.50:
            ang = rng.uniform(-0.55, 0.55)
            pos = rng.uniform(-0.75, 0.75)
            width_f = rng.uniform(0.008, 0.025)
            amp = rng.uniform(0.08, 0.24) * s * rng.choice((-1.0, 1.0))
            kink_px = rng.uniform(1.2, 3.2) * gs * rng.choice((-1.0, 1.0))
            fold_params.append((math.cos(ang), math.sin(ang), pos, width_f, amp, kink_px))
        else:
            fold_params.append((1.0, 0.0, 0.0, 0.02, 0.0, 0.0))
    if any(p[4] != 0.0 for p in fold_params):
        ca = torch.tensor([p[0] for p in fold_params], device=dev).view(m, 1, 1, 1)
        sa = torch.tensor([p[1] for p in fold_params], device=dev).view(m, 1, 1, 1)
        pos = torch.tensor([p[2] for p in fold_params], device=dev).view(m, 1, 1, 1)
        wf = torch.tensor([p[3] for p in fold_params], device=dev).view(m, 1, 1, 1)
        amp = torch.tensor([p[4] for p in fold_params], device=dev).view(m, 1, 1, 1)
        signed_d = (gx4 * ca + gy4 * sa - pos) / wf
        bevel = signed_d * torch.exp(-0.5 * signed_d.square())
        x.mul_((1.0 + amp * bevel).clamp_(0.68, 1.20))
        if gs > 0.0:
            # Physical crease kink: local step/ridge displacement right across the fold.
            kink = torch.tensor([p[5] for p in fold_params], device=dev).view(m, 1, 1, 1)
            ridge = torch.exp(-0.5 * (signed_d * 0.65).square())
            dx_grid.add_(ridge * kink * ca * px_to_nx)
            dy_grid.add_(ridge * kink * (0.75 * px_to_ny))
            has_geom = True

    # 10b. Full-page off-center lens distortions (barrel / pincushion / mustache)
    #      + residual paper rotation + page-bend wave (book arch / spine curl).
    #      Because each strip is cropped from a full page, the camera optical
    #      center (cx_lens, cy_lens) sits in FULL-PAGE coordinates — often far
    #      above (bottom-of-page strip) or far below (top-of-page strip) the
    #      strip itself, producing realistic asymmetric upward/downward arch!
    if gs > 0.0:
        k1_l, k2_l, lcx_l, lcy_l, rot_l, bend_a, bend_f, bend_p = [], [], [], [], [], [], [], []
        for rng in rngs:
            lm = rng.random()
            if lm < 0.28:  # barrel
                k1_l.append(rng.uniform(0.015, 0.055) * gs)
                k2_l.append(rng.uniform(0.0, 0.020) * gs)
            elif lm < 0.48:  # pincushion
                k1_l.append(-rng.uniform(0.015, 0.045) * gs)
                k2_l.append(-rng.uniform(0.0, 0.015) * gs)
            elif lm < 0.68:  # mustache (wide-angle phone lens)
                sign = rng.choice((-1.0, 1.0))
                k1_l.append(sign * rng.uniform(0.025, 0.060) * gs)
                k2_l.append(-sign * rng.uniform(0.025, 0.055) * gs)
            else:
                k1_l.append(0.0); k2_l.append(0.0)
            # Optical center in full-page normalized coords (x in [-0.5, 0.5],
            # y in [-0.95, +0.95] relative to strip center; strip spans ~0.22 of page H).
            lcx_l.append(rng.uniform(-0.45, 0.45))
            lcy_l.append(rng.uniform(-0.95, 0.95))

            if rng.random() < 0.60:
                deg = rng.uniform(0.35, 1.55) * gs * rng.choice((-1.0, 1.0))
                rot_l.append(math.radians(deg))
            else:
                rot_l.append(0.0)

            if rng.random() < 0.55:
                bend_a.append(rng.uniform(1.2, 3.8) * gs * rng.choice((-1.0, 1.0)))
                bend_f.append(rng.uniform(0.5, 1.6) * math.pi)
                bend_p.append(rng.uniform(0.0, 2.0 * math.pi))
            else:
                bend_a.append(0.0); bend_f.append(1.0); bend_p.append(0.0)

        if any(v != 0.0 for v in k1_l) or any(v != 0.0 for v in k2_l):
            k1_t = torch.tensor(k1_l, device=dev).view(m, 1, 1, 1)
            k2_t = torch.tensor(k2_l, device=dev).view(m, 1, 1, 1)
            lcx_t = torch.tensor(lcx_l, device=dev).view(m, 1, 1, 1)
            lcy_t = torch.tensor(lcy_l, device=dev).view(m, 1, 1, 1)
            # Map strip coords onto a full-page slice (strip height ~ 0.22 of page).
            strip_h_frac = 0.22
            px = gx4 - lcx_t
            py = gy4 * strip_h_frac - lcy_t
            r2 = px.square() + py.square()
            rad_fac = k1_t * r2 + k2_t * r2.square()
            dlx = px * rad_fac
            # Convert page-space vertical displacement back to strip-normalized y,
            # and re-center mean shift so top/bottom strips bow without clipping!
            dly = (py * rad_fac) / strip_h_frac
            dlx = dlx - dlx.mean(dim=(2, 3), keepdim=True)
            dly = (dly - dly.mean(dim=(2, 3), keepdim=True)).clamp_(-0.08, 0.08)
            dx_grid.add_(dlx)
            dy_grid.add_(dly)
            has_geom = True

        if any(v != 0.0 for v in rot_l):
            th = torch.tensor(rot_l, device=dev).view(m, 1, 1, 1)
            ct = torch.cos(th) - 1.0
            st = torch.sin(th)
            # Account for aspect ratio W/H so rotation is true Euclidean angle!
            asp = float(w) / max(float(h), 1.0)
            dx_grid.add_(gx4 * ct - (gy4 / asp) * st)
            dy_grid.add_((gx4 * asp) * st + gy4 * ct)
            has_geom = True

        if any(v != 0.0 for v in bend_a):
            ba_t = torch.tensor(bend_a, device=dev).view(m, 1, 1, 1)
            bf_t = torch.tensor(bend_f, device=dev).view(m, 1, 1, 1)
            bp_t = torch.tensor(bend_p, device=dev).view(m, 1, 1, 1)
            dy_grid.add_(torch.sin(gx4 * bf_t + bp_t) * (ba_t * px_to_ny))
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

    # 11. Real lighting: book-spine shadow + directional phone/hand shadow + vignette.
    ax_l, ay_l, cx_l, cy_l, vig_l = [], [], [], [], []
    spine_side, spine_amt = [], []
    sh_ca, sh_sa, sh_pos, sh_soft, sh_amt = [], [], [], [], []
    for rng in rngs:
        if rng.random() < 0.80:
            ax_l.append(rng.uniform(-0.14, 0.14) * s)
            ay_l.append(rng.uniform(-0.10, 0.10) * s)
            if rng.random() < 0.55:
                cx_l.append(rng.uniform(-0.4, 0.4))
                cy_l.append(rng.uniform(-0.4, 0.4))
                vig_l.append(rng.uniform(0.05, 0.22) * s)
            else:
                cx_l.append(0.0); cy_l.append(0.0); vig_l.append(0.0)
        else:
            ax_l.append(0.0); ay_l.append(0.0)
            cx_l.append(0.0); cy_l.append(0.0); vig_l.append(0.0)

        if rng.random() < 0.40:
            spine_side.append(rng.choice((-1.0, 1.0)))
            spine_amt.append(rng.uniform(0.18, 0.42) * s)
        else:
            spine_side.append(1.0); spine_amt.append(0.0)

        if rng.random() < 0.45:
            ang = rng.uniform(0.0, 2.0 * math.pi)
            sh_ca.append(math.cos(ang)); sh_sa.append(math.sin(ang))
            sh_pos.append(rng.uniform(-0.5, 0.6))
            sh_soft.append(rng.uniform(4.0, 14.0))
            sh_amt.append(rng.uniform(0.12, 0.34) * s)
        else:
            sh_ca.append(1.0); sh_sa.append(0.0); sh_pos.append(0.0)
            sh_soft.append(8.0); sh_amt.append(0.0)

    ramp = torch.ones((m, 1, h, w), device=dev, dtype=torch.float32)
    if any(v != 0.0 for v in ax_l) or any(v != 0.0 for v in ay_l):
        ax_t = torch.tensor(ax_l, device=dev).view(m, 1, 1, 1)
        ay_t = torch.tensor(ay_l, device=dev).view(m, 1, 1, 1)
        ramp.add_(ax_t * gx4 + ay_t * gy4)
    if any(v != 0.0 for v in vig_l):
        cx_t = torch.tensor(cx_l, device=dev).view(m, 1, 1, 1)
        cy_t = torch.tensor(cy_l, device=dev).view(m, 1, 1, 1)
        vig_t = torch.tensor(vig_l, device=dev).view(m, 1, 1, 1)
        r2 = (gx4 - cx_t).square() + (gy4 - cy_t).square()
        ramp.mul_(1.0 - vig_t * r2.clamp_(0.0, 1.5))
    if any(v > 0.0 for v in spine_amt):
        sd = torch.tensor(spine_side, device=dev).view(m, 1, 1, 1)
        sa_t = torch.tensor(spine_amt, device=dev).view(m, 1, 1, 1)
        # Exponential darkening along left (-1) or right (+1) page binding edge.
        edge_dist = (1.0 - sd * gx4) * 0.5  # 0 at binding edge, 1 at far edge
        ramp.mul_(1.0 - sa_t * torch.exp(-edge_dist * 10.0))
    if any(v > 0.0 for v in sh_amt):
        ca_t = torch.tensor(sh_ca, device=dev).view(m, 1, 1, 1)
        sa_t = torch.tensor(sh_sa, device=dev).view(m, 1, 1, 1)
        sp_t = torch.tensor(sh_pos, device=dev).view(m, 1, 1, 1)
        sf_t = torch.tensor(sh_soft, device=dev).view(m, 1, 1, 1)
        am_t = torch.tensor(sh_amt, device=dev).view(m, 1, 1, 1)
        shadow = torch.sigmoid((gx4 * ca_t + gy4 * sa_t - sp_t) * sf_t)
        ramp.mul_(1.0 - am_t * shadow)
    x.mul_(ramp.clamp_(0.42, 1.25))

    # ------------------------------------------------------------------
    # STAGE D: Real smartphone camera optics (WB, exposure, CA,
    #          tilt-shift focus / motion blur, JPEG softness, sensor noise)
    # ------------------------------------------------------------------
    # 12. White balance + exposure / contrast / gamma.
    tint_l, gamma_l, contr_l, bright_l = [], [], [], []
    for rng in rngs:
        if rng.random() < 0.70:
            tint_l.append([
                1.0 + rng.uniform(-0.07, 0.07) * s,
                1.0 + rng.uniform(-0.04, 0.04) * s,
                1.0 + rng.uniform(-0.10, 0.07) * s,
            ])
        else:
            tint_l.append([1.0, 1.0, 1.0])
        if rng.random() < 0.80:
            gamma_l.append(math.exp(rng.uniform(-0.22, 0.22) * s))
            contr_l.append(1.0 + rng.uniform(-0.16, 0.16) * s)
            bright_l.append(rng.uniform(-0.05, 0.05) * s)
        else:
            gamma_l.append(1.0); contr_l.append(1.0); bright_l.append(0.0)

    x.mul_(torch.tensor(tint_l, device=dev, dtype=torch.float32).view(m, 3, 1, 1))
    gamma_t = torch.tensor(gamma_l, device=dev, dtype=torch.float32).view(m, 1, 1, 1)
    contr_t = torch.tensor(contr_l, device=dev, dtype=torch.float32).view(m, 1, 1, 1)
    bright_t = torch.tensor(bright_l, device=dev, dtype=torch.float32).view(m, 1, 1, 1)
    x.clamp_(1e-4, 1.0).pow_(gamma_t)
    means = x.mean(dim=(1, 2, 3), keepdim=True)
    x.sub_(means).mul_(contr_t).add_(means + bright_t).clamp_(0.0, 1.0)

    # 13. Lateral chromatic aberration (horizontal-only R/B shift -> 0 vertical drift).
    for j, rng in enumerate(rngs):
        if rng.random() < 0.35 and w >= 16:
            shift = rng.choice((-1, 1))
            x[j, 0] = torch.roll(x[j, 0], shifts=shift, dims=1)
            x[j, 2] = torch.roll(x[j, 2], shifts=-shift, dims=1)

    # 14. Focus optics: uniform defocus, tilt-shift (one edge soft), horizontal
    #     motion blur, or unsharp camera sharpening.
    focus_mode, focus_val = [], []
    for rng in rngs:
        pf = rng.random()
        if pf < 0.28:
            focus_mode.append(1)  # uniform defocus
            focus_val.append(rng.uniform(0.35, 1.05) * s)
        elif pf < 0.48:
            focus_mode.append(2)  # tilt-shift (sharp in middle/one side, soft other)
            focus_val.append(rng.uniform(0.85, 1.45) * s)
        elif pf < 0.60:
            focus_mode.append(3)  # unsharp mask (phone oversharpening)
            focus_val.append(rng.uniform(0.20, 0.55) * s)
        else:
            focus_mode.append(0); focus_val.append(0.0)

    blur_rows = [j for j, fm in enumerate(focus_mode) if fm == 1]
    tilt_rows = [j for j, fm in enumerate(focus_mode) if fm == 2]
    sharp_rows = [j for j, fm in enumerate(focus_mode) if fm == 3]
    if blur_rows:
        idx = torch.tensor(blur_rows, device=dev, dtype=torch.long)
        sig = sum(focus_val[j] for j in blur_rows) / len(blur_rows)
        x.index_copy_(0, idx, _batched_separable_blur(x.index_select(0, idx), sig))
    if tilt_rows:
        idx = torch.tensor(tilt_rows, device=dev, dtype=torch.long)
        sub = x.index_select(0, idx)
        sig = sum(focus_val[j] for j in tilt_rows) / len(tilt_rows)
        blurred = _batched_separable_blur(sub, sig)
        # Spatially-varying focus blend across x (sharp at focal_x, soft far away).
        fx = torch.tensor([rngs[j].uniform(-0.7, 0.7) for j in tilt_rows],
                          device=dev).view(-1, 1, 1, 1)
        blend = ((gx4 - fx).abs() / 1.4).clamp_(0.0, 1.0).pow_(1.5)
        x.index_copy_(0, idx, sub * (1.0 - blend) + blurred * blend)
    if sharp_rows:
        idx = torch.tensor(sharp_rows, device=dev, dtype=torch.long)
        sub = x.index_select(0, idx)
        blurred = _batched_separable_blur(sub, 0.7)
        amt = torch.tensor([focus_val[j] for j in sharp_rows],
                           device=dev, dtype=torch.float32).view(-1, 1, 1, 1)
        x.index_copy_(0, idx, (sub + amt * (sub - blurred)).clamp_(0.0, 1.0))

    # 15. Down-up camera/JPEG softness.
    scale_l = [
        rng.uniform(max(0.66, 1.0 - 0.30 * s), 0.94)
        if (rng.random() < 0.45 and h >= 16 and w >= 16) else 1.0
        for rng in rngs
    ]
    scale_rows = [j for j, sc in enumerate(scale_l) if sc < 0.999]
    if scale_rows:
        idx = torch.tensor(scale_rows, device=dev, dtype=torch.long)
        sc_mean = sum(scale_l[j] for j in scale_rows) / len(scale_rows)
        sh = max(8, int(round(h * sc_mean)))
        sw = max(8, int(round(w * sc_mean)))
        small = F.interpolate(x.index_select(0, idx), size=(sh, sw),
                              mode='bilinear', align_corners=False)
        x.index_copy_(0, idx, F.interpolate(small, size=(h, w),
                                            mode='bilinear', align_corners=False))

    # 16. Sensor shot + read Gaussian noise (heteroscedastic: brighter -> slightly more shot noise)
    #     + sparse toner dust specks.
    noise_l = [rng.uniform(0.008, 0.032) * s if rng.random() < 0.80 else 0.0 for rng in rngs]
    if any(v > 0.0 for v in noise_l):
        noise = torch.empty_like(x).normal_(0.0, 1.0, generator=gen)
        sig_t = torch.tensor(noise_l, device=dev, dtype=torch.float32).view(m, 1, 1, 1)
        x.add_(noise.mul_(sig_t))

    return x.clamp_(0.0, 1.0), staffb


def augment_rgb_01(
    rgb: torch.Tensor,
    base_seed: int,
    epoch: int,
    sample_key: str,
    strength: float = 1.0,
    geom_strength: float = 1.0,
) -> torch.Tensor:
    """Apply deterministic physical paper/ink/camera/geometry aug to ``[3, H, W]``."""
    if strength <= 0.0:
        return rgb.clone()
    seed64 = _stream_seed(base_seed, epoch, sample_key)
    rng = random.Random(seed64)
    dev = rgb.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed64 & 0x7FFFFFFFFFFFFFFF)
    out, _ = _augment_batch_01(
        rgb.clone().float().unsqueeze(0), [rng], dev, gen,
        float(strength), geom_strength=float(geom_strength))
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
):
    """Deterministically augment an ImageNet-standardized batch (vectorized on GPU).

    Supports both 4D ``grayb`` (``[B, 1, H, W]``, ``collate_hybrid`` native)
    and 3D ``grayb`` (``[B, H, W]``). When ``staffb`` (``[B, 1, H, W]``) is
    provided, applies the exact same geometric warp to ``staffb`` and returns
    ``(out_rgb, out_gray, out_staff)``; otherwise returns ``(out_rgb, out_gray)``.
    """
    if (aug_p <= 0.0 and clean_p <= 0.0) or strength <= 0.0:
        return (rgb_std, grayb, staffb) if staffb is not None else (rgb_std, grayb)

    dev = rgb_std.device
    dtype = rgb_std.dtype
    mean, std, luma = _device_consts(dev)
    _, _, _, max_w = rgb_std.shape

    active_idx: list[int] = []
    active_rngs: list[random.Random] = []
    active_seeds: list[int] = []
    for i, pair in enumerate(batch_pairs):
        img_path = pair[0]
        prob = clean_p if is_clean_image_path(img_path) else aug_p
        if prob <= 0.0:
            continue
        key = Path(img_path).name
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
        geom_strength=float(geom_strength), staffb=st_sub)

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
