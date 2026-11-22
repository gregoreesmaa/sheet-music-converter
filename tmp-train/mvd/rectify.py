"""Canonical 4-slot staff-line rectifier (Phase 2, Task T2.1).

Maps a warped system strip (perspective, page curl, +/-10 deg rotation,
multi-staff scale crush) onto a fixed canonical canvas of ``height`` rows
(default 256) with staff space ``target_sp`` (default 10.0 px):

    y_canon(s, k) = 12.0 + 64.0 * s + k * target_sp      (s = slot, k = line)

Slot ``s`` always carries its 5 staff lines at those rows (slot 0 spans
``y = 12..52`` at default spacing, plus +/-2*target_sp margins above and
below each staff for ledger lines and stems). Staves
``0..S-1`` of an ``S``-staff system (``S <= 4``) occupy slots ``0..S-1``;
slots ``S..3`` are filled with clean paper white (255). Horizontal width
is resampled by ``target_sp / median_measured_sp`` (NOT ``256 / raw_h``),
so 3- and 4-staff systems never have their horizontal width crushed.

Staff-line conventions match ``converter/layout-staff`` and
``tmp-train/row-retrain``: each staff lists its 5 lines top-to-bottom
(``k = 0`` is the top line), ink is dark on a light background, and
``staves_lines`` carries either constant y values (exact-truth rows) or
dense per-column slice curves (e.g. neural row-model output resampled to
input width). In treble clef the bottom line (``k = 4``) is E4 and the
space just above it (``k = 3.5``) is F4.

Reference: ``todo/phase-2-hybrid-dewarp.md``, Task T2.1.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

CANON_TOP = 12.0
SLOT_PITCH = 64.0
MAX_STAVES = 4
LINES_PER_STAFF = 5
PAPER = 255.0
REF_TARGET_SP = 10.0


def canonical_layout(
    target_sp: float = REF_TARGET_SP,
    height: int | None = None,
    n_slots: int = MAX_STAVES,
) -> tuple[float, float, int]:
    """Return ``(canon_top, slot_pitch, canvas_height)`` for ``target_sp``.

    At the default ``target_sp=10.0`` (with ``height=None`` or ``256``),
    returns ``(12.0, 64.0, 256)`` — bit-identical to the legacy layout.
    For higher resolutions (e.g. ``target_sp=12.0`` -> ``height=320``,
    ``target_sp=14.0`` -> ``height=352``), scales ``slot_pitch`` in
    8-px increments so ``canvas_height = n_slots * slot_pitch`` stays
    divisible by 32 (exact for 4x8, 8x16, and 32x32 encoder strides)
    and centers the 4-space staff inside each slot.
    """
    sp = float(target_sp)
    if not np.isfinite(sp) or sp <= 0:
        raise ValueError(f"target_sp must be positive, got {target_sp}")
    if height is None:
        if sp == REF_TARGET_SP and n_slots == MAX_STAVES:
            return CANON_TOP, SLOT_PITCH, int(SLOT_PITCH * MAX_STAVES)
        scale = sp / REF_TARGET_SP
        # Round per-slot pitch to a multiple of 8 px so 4 slots stay
        # divisible by 32, and guarantee >= 1.2 * sp margin per side.
        min_pitch = 4.0 * sp + max(8.0, 2.0 * sp)
        raw_pitch = max(SLOT_PITCH * scale, min_pitch)
        slot_pitch = float(int(round(raw_pitch / 8.0)) * 8)
        if slot_pitch <= 4.0 * sp:
            slot_pitch = float(int(np.ceil((4.0 * sp + 8.0) / 8.0)) * 8)
        out_h = int(round(slot_pitch * n_slots))
    else:
        out_h = int(height)
        if out_h <= 0:
            raise ValueError(f"height must be positive, got {height}")
        slot_pitch = float(out_h) / float(n_slots)
    canon_top = (slot_pitch - 4.0 * sp) / 2.0
    return canon_top, slot_pitch, out_h


def canon_y(
    slot: int,
    line: float,
    target_sp: float = REF_TARGET_SP,
    height: int | None = None,
) -> float:
    """Canonical row of staff ``slot`` at staff-step ``line`` (halves allowed)."""
    top, pitch, _ = canonical_layout(target_sp=target_sp, height=height)
    return top + pitch * slot + line * float(target_sp)


def _as_curve(entry, w_in: int) -> np.ndarray:
    """One staff line as a dense per-input-column ``(W_in,)`` float curve."""
    a = np.asarray(entry, dtype=np.float64)
    if a.ndim == 0:
        return np.full(w_in, float(a), dtype=np.float64)
    a = np.ravel(a)
    if a.shape[0] == w_in:
        return a.astype(np.float64, copy=True)
    if a.shape[0] < 2:
        raise ValueError(f"staff-line curve needs >= 2 samples, got {a.shape[0]}")
    src = np.linspace(0.0, float(w_in - 1), num=int(a.shape[0]))
    return np.interp(np.arange(w_in, dtype=np.float64), src, a.astype(np.float64))


def _staff_array(staff, w_in: int) -> np.ndarray:
    """One staff as a ``(5, W_in)`` float array of per-column line positions."""
    arr = None
    try:
        arr = np.asarray(staff, dtype=np.float64)
    except (ValueError, TypeError):
        arr = None
    if arr is not None and arr.shape == (LINES_PER_STAFF,):
        # Five scalars: constant lines across the strip width.
        return np.repeat(arr[:, None], w_in, axis=1)
    if arr is not None and arr.ndim == 2 and arr.shape[0] == LINES_PER_STAFF:
        return np.stack([_as_curve(arr[k], w_in) for k in range(LINES_PER_STAFF)])
    lines = list(staff)
    if len(lines) != LINES_PER_STAFF:
        raise ValueError(f"each staff needs exactly 5 lines, got {len(lines)}")
    return np.stack([_as_curve(ln, w_in) for ln in lines])


def rectify_system_strip(
    img: Image.Image,
    staves_lines: list,
    target_sp: float = REF_TARGET_SP,
    height: int | None = None,
) -> tuple[Image.Image, torch.Tensor]:
    """Dewarp one system strip onto the canonical 4-slot canvas.

    Args:
        img: PIL image (``L`` or ``RGB``; anything else is converted to
            ``RGB``). Dark ink on a light background.
        staves_lines: ``S`` staves (``1 <= S <= 4``), each a list of 5
            line positions top-to-bottom (``k = 0`` top). Each line is a
            scalar y (constant across the strip) or a per-column curve
            of float y values sampled at input-pixel columns.
        target_sp: canonical staff space in px (default 10.0). Higher
            values (e.g. 12.0, 14.0) preserve micro-glyphs (dots,
            accidentals, 128th flags) when paired with proportional
            slot height (automatic when ``height=None``).
        height: canonical canvas height in rows. ``None`` (default)
            uses :func:`canonical_layout` (256 at ``target_sp=10.0``,
            320 at ``12.0``, 352 at ``14.0``).

    Returns:
        ``(rectified PIL image, reference channel)``. The image has shape
        ``(W_out, out_h)`` with
        ``W_out = round(W_in * target_sp / median_measured_sp)``. The
        reference channel is a float32 ``[1, out_h, W_out]`` tensor with
        ``+1.0`` on the 5 canonical rows of each active staff and ``0.0``
        elsewhere (including inactive slots).
    """
    if not isinstance(img, Image.Image):
        raise TypeError(f"img must be a PIL Image, got {type(img).__name__}")
    staves = list(staves_lines)
    n_staff = len(staves)
    if not 1 <= n_staff <= MAX_STAVES:
        raise ValueError(f"need 1..{MAX_STAVES} staves, got {n_staff}")
    target_sp = float(target_sp)
    canon_top, slot_pitch, out_h = canonical_layout(
        target_sp=target_sp, height=height, n_slots=MAX_STAVES
    )
    if slot_pitch <= 4.0 * target_sp:
        raise ValueError("canonical control rows overlap; slots require pitch > 4*target_sp")
    w_in, h_in = img.size
    if w_in < 2 or h_in < 2:
        raise ValueError(f"image too small to rectify: {(w_in, h_in)}")

    work = img if img.mode in ("L", "RGB") else img.convert("RGB")
    pix = np.asarray(work, dtype=np.float32)
    rgb = pix.ndim == 3

    # (S, 5, W_in) per-column input line positions.
    pos_in = np.stack([_staff_array(st, w_in) for st in staves])
    med_sp = float(np.median(np.diff(pos_in, axis=1)))
    if not np.isfinite(med_sp) or med_sp <= 0:
        raise ValueError(f"non-positive median staff space: {med_sp}")
    w_out = max(1, int(round(w_in * target_sp / med_sp)))

    # Output column -> input column (pixel-center aligned).
    grid = np.arange(w_in, dtype=np.float64)
    xs = (np.arange(w_out, dtype=np.float64) + 0.5) * (w_in / w_out) - 0.5
    xsc = np.clip(xs, 0.0, float(w_in - 1))
    pos = np.stack(
        [
            [np.interp(xsc, grid, pos_in[s, k]) for k in range(LINES_PER_STAFF)]
            for s in range(n_staff)
        ]
    )  # (S, 5, W_out)
    sp_loc = np.maximum(np.mean(np.diff(pos, axis=1), axis=1), 1e-6)

    # Canonical control rows: top margin, all active lines, bottom margin.
    yc = (
        canon_top
        + slot_pitch * np.arange(n_staff, dtype=np.float64)[:, None]
        + target_sp * np.arange(LINES_PER_STAFF, dtype=np.float64)[None, :]
    )
    p_out = np.concatenate(
        ([yc[0, 0] - 2.0 * target_sp], yc.reshape(-1), [yc[-1, -1] + 2.0 * target_sp])
    )
    if np.any(np.diff(p_out) <= 0):
        raise ValueError("canonical control rows overlap; slots require pitch >= 4*target_sp")

    # Backward vertical map per output column (piecewise-linear + clamp).
    rows = np.arange(out_h, dtype=np.float64)
    ys = np.empty((out_h, w_out), dtype=np.float64)
    bump = np.arange(5 * n_staff + 2, dtype=np.float64) * 1e-9
    for j in range(w_out):
        p_in = np.concatenate(
            (
                [pos[0, 0, j] - 2.0 * sp_loc[0, j]],
                pos[:, :, j].reshape(-1),
                [pos[-1, -1, j] + 2.0 * sp_loc[-1, j]],
            )
        )
        p_in = np.maximum.accumulate(p_in) + bump
        ys[:, j] = np.interp(rows, p_out, p_in)

    # Bilinear resample; samples falling outside the page become paper.
    x0 = np.clip(np.floor(xsc).astype(np.int64), 0, w_in - 1)
    x1 = np.minimum(x0 + 1, w_in - 1)
    wx = np.clip(xsc - np.floor(xsc), 0.0, 1.0)
    wx[x0 == w_in - 1] = 0.0
    y0 = np.floor(ys).astype(np.int64)
    wy = np.clip(ys - np.floor(ys), 0.0, 1.0)
    bad = (ys < 0.0) | (ys > float(h_in - 1))
    y0c = np.clip(y0, 0, h_in - 1)
    y1c = np.minimum(y0c + 1, h_in - 1)
    if rgb:
        c00 = pix[y0c, x0]
        c10 = pix[y1c, x0]
        c01 = pix[y0c, x1]
        c11 = pix[y1c, x1]
        wx3 = wx[None, :, None]
        wy3 = wy[:, :, None]
        top = c00 + (c01 - c00) * wx3
        bot = c10 + (c11 - c10) * wx3
        out = top + (bot - top) * wy3
        out[bad] = PAPER
    else:
        c00 = pix[y0c, x0]
        c10 = pix[y1c, x0]
        c01 = pix[y0c, x1]
        c11 = pix[y1c, x1]
        wx2 = wx[None, :]
        top = c00 + (c01 - c00) * wx2
        bot = c10 + (c11 - c10) * wx2
        out = top + (bot - top) * wy
        out[bad] = PAPER

    # Inactive slots are pure paper white over their whole slot block.
    for s in range(n_staff, MAX_STAVES):
        r0 = int(round(s * slot_pitch))
        r1 = int(round((s + 1) * slot_pitch))
        out[r0:r1] = PAPER

    arr = np.clip(np.round(out), 0, 255).astype(np.uint8)
    out_img = Image.fromarray(arr, mode="RGB" if rgb else "L")

    ref = torch.zeros((1, out_h, w_out), dtype=torch.float32)
    for s in range(n_staff):
        for k in range(LINES_PER_STAFF):
            r = int(round(canon_top + slot_pitch * s + k * target_sp))
            if 0 <= r < out_h:
                ref[0, r, :] = 1.0
    return out_img, ref
