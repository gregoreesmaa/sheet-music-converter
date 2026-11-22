"""Hybrid heat features (Phase 2, Task T2.2).

Fuses HeatNet's 7-channel spatial keypoint probabilities with the
1-channel canonical staff-line map from ``rectify.py`` into the
``[8, H, W]`` extra-channel block consumed by
``MvdNet(extra_chans=8)`` (see ``model.py``).

HeatNet contract (must match ``converter/symbol-detect/src/onnx_heat.rs``
and ``tmp-train/det-data/train_heat.py``):

* Weights: ``tmp-train/det-data/heatnet.onnx`` (or
  ``tmp-train/det-data/heatnet.pt`` state dict on CUDA).
* Input: ``[N, 1, H, W]`` float32 grayscale, raw pixels scaled by
  ``1/255`` (no mean/std standardization — the Rust proposer feeds
  ``px / 255.0`` directly).
* Outputs: ``heat`` logits ``[N, 7, H/2, W/2]`` (stride 2) + ``wh``
  box sizes (ignored here). Channel order is the ``HEAT_CLASSES``
  order below: Note, Rest, Beam, Hook, Accidental, NoteDot, Clef.
* This module applies ``sigmoid`` to the logits and bilinearly
  upsamples to full ``[7, H, W]`` probabilities, then concatenates the
  ``[1, H, W]`` canonical staff-line reference channel from
  ``rectify.rectify_system_strip`` (``+1.0`` on active staff lines,
  ``0.0`` elsewhere) to form ``[8, H, W]``.

Reference: ``todo/phase-2-hybrid-dewarp.md``, Task T2.2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rectify import rectify_system_strip  # noqa: E402  (re-exported helper)

__all__ = [
    'HEAT_CLASSES',
    'HEAT_ONNX_DEFAULT',
    'HEAT_PT_DEFAULT',
    'EXTRA_CHANS',
    'HeatNet',
    'preprocess_strip',
    'compute_heat_channels',
    'extra_channels_for_strip',
    'concat_hybrid_input',
    'heat_channels_for_sidecar',
]

#: Heat channel order. Must match ``HEAT_CLASSES`` in
#: ``converter/symbol-detect/src/onnx_heat.rs`` and ``CLASSES`` in
#: ``tmp-train/det-data/train_heat.py``.
HEAT_CLASSES = ('Note', 'Rest', 'Beam', 'Hook', 'Accidental',
                'NoteDot', 'Clef')

#: Number of extra encoder channels: 7 heat + 1 staff-grid.
EXTRA_CHANS = len(HEAT_CLASSES) + 1

HEAT_ONNX_DEFAULT = 'tmp-train/det-data/heatnet.onnx'
HEAT_PT_DEFAULT = 'tmp-train/det-data/heatnet.pt'


def preprocess_strip(img: Image.Image) -> torch.Tensor:
    """Grayscale system strip -> ``[1, 1, H, W]`` float32 ``/ 255`` tensor.

    Mirrors the ``propose`` input packing in ``onnx_heat.rs``: single
    luminance channel, raw byte value divided by 255, no ImageNet
    standardization (that standardization applies to the MvdNet RGB
    path in ``train.load_image``, not to HeatNet).
    """
    if not isinstance(img, Image.Image):
        raise TypeError(f'img must be a PIL Image, got {type(img).__name__}')
    gray = img.convert('L')
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


class _HeatNetTorch(nn.Module):
    """Torch mirror of the current ``train_heat.HeatNet``.

    Architecture is duplicated here (instead of importing
    ``tmp-train/det-data/train_heat.py``) so this module has no
    cross-directory import. Keep in sync with that file: 1->32 (k7/s2),
    32->64 (k3/s2), 64->128 (k3/s2), two k4/s2 transpose convs back to
    stride 2, then ``heat`` (7 ch) and ``wh`` (14 ch) heads.
    """

    def __init__(self, n_classes: int = len(HEAT_CLASSES)):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(1, 32, 7, stride=2, padding=3),
                                nn.BatchNorm2d(32), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                nn.BatchNorm2d(64), nn.ReLU())
        self.c3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                nn.BatchNorm2d(128), nn.ReLU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU())
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU())
        self.heat = nn.Conv2d(32, n_classes, 3, padding=1)
        self.wh = nn.Conv2d(32, 2 * n_classes, 3, padding=1)

    def forward(self, x):
        f = self.up2(self.up(self.c3(self.c2(self.c1(x)))))
        return self.heat(f), self.wh(f)


class _HeatNetTorchV0(nn.Module):
    """Torch mirror of the legacy ``heatnet.pt`` checkpoint arch.

    Same trunk as above but a single transpose conv (stride-4 output)
    and class-agnostic ``wh`` (2 ch shared w/h instead of 2-per-class).
    Selected automatically by :func:`_run_pt_heat` when the checkpoint
    has no ``up2`` keys.
    """

    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(1, 32, 7, stride=2, padding=3),
                                nn.BatchNorm2d(32), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                nn.BatchNorm2d(64), nn.ReLU())
        self.c3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                nn.BatchNorm2d(128), nn.ReLU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU())
        self.heat = nn.Conv2d(64, len(HEAT_CLASSES), 3, padding=1)
        self.wh = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        f = self.up(self.c3(self.c2(self.c1(x))))
        return self.heat(f), self.wh(f)


def _run_onnx_heat(x: torch.Tensor, onnx_path: str) -> torch.Tensor:
    """Raw ``heat`` logits ``[1, 7, h, w]`` for preprocessed strip ``x``."""
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            'onnxruntime is required for heatnet.onnx inference') from e
    sess = ort.InferenceSession(str(onnx_path),
                                providers=['CPUExecutionProvider'])
    (inp,) = sess.get_inputs()
    out = sess.run(['heat'], {inp.name: x.numpy()})[0]
    return torch.from_numpy(out)


def _run_pt_heat(x: torch.Tensor, pt_path: str) -> torch.Tensor:
    """Raw ``heat`` logits ``[1, 7, h, w]`` via the ``.pt`` CUDA path.

    The checkpoint selects the arch: current ``train_heat.py`` weights
    (``up2`` keys present, stride 2) vs the legacy ``heatnet.pt``
    weights (no ``up2``, stride 4, class-agnostic ``wh``). The
    downstream ``sigmoid`` + bilinear upsampling in
    :func:`compute_heat_channels` is stride-agnostic.
    """
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.load(str(pt_path), map_location=dev, weights_only=True)
    net = _HeatNetTorch() if 'up2.0.weight' in state else _HeatNetTorchV0()
    net = net.to(dev)
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        heat, _ = net(x.to(dev))
    return heat.cpu()


class HeatNet:
    """Persistent HeatNet session: weights loaded ONCE (T2.4).

    :func:`compute_heat_channels` / :func:`_run_pt_heat` reload the
    checkpoint per strip — fine for one-off eval, prohibitive at
    30k training steps x batch 4. This class loads the ``.pt`` state
    dict once (arch auto-selected by ``up2`` keys, as in
    :func:`_run_pt_heat`), holds the net in eval mode on ``device``,
    and serves batched ``[B, 1, H, W]`` ``/ 255`` gray inputs ->
    ``[B, 7, H, W]`` sigmoid probabilities (bilinearly upsampled
    from the stride-2/stride-4 head, matching
    :func:`compute_heat_channels`). Fully convolutional: batch
    members may share padded width; white-pad regions carry no ink
    and read ~0 probability. Deterministic: eval mode, no_grad,
    no input-dependent branching.
    """

    def __init__(self, pt_path: str = HEAT_PT_DEFAULT, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available()
                                 else 'cpu')
        state = torch.load(str(pt_path), map_location=self.device,
                           weights_only=True)
        if 'up2.0.weight' in state:
            self.net = _HeatNetTorch()
        else:
            self.net = _HeatNetTorchV0()
        self.net.load_state_dict(state)
        self.net = self.net.to(self.device)
        self.net.eval()

    @torch.no_grad()
    def probs(self, gray: torch.Tensor) -> torch.Tensor:
        """``[B, 1, H, W]`` gray -> ``[B, 7, H, W]`` heat probabilities."""
        if gray.dim() != 4 or gray.shape[1] != 1:
            raise ValueError('gray must be [B, 1, H, W], got %s'
                             % (tuple(gray.shape),))
        b, _, h, w = gray.shape
        logits, _ = self.net(gray.to(self.device))
        p = torch.sigmoid(logits).float()
        if tuple(p.shape[2:]) != (h, w):
            p = torch.nn.functional.interpolate(
                p, size=(h, w), mode='bilinear', align_corners=False)
        return p

    def param_bytes(self) -> int:
        """Total state-dict bytes (runbook/health logging)."""
        return sum(p.numel() * p.element_size()
                   for p in self.net.state_dict().values())


def compute_heat_channels(img: Image.Image,
                          onnx_path: str = HEAT_ONNX_DEFAULT,
                          pt_path: str | None = None) -> torch.Tensor:
    """``[7, H, W]`` sigmoid heat probabilities for a system strip.

    Runs ``heatnet.onnx`` via onnxruntime by default; pass ``pt_path``
    (e.g. ``tmp-train/det-data/heatnet.pt``) to use the torch
    checkpoint instead (CUDA when available). Logits at stride 2 are
    ``sigmoid``-ed (matching the Rust ``decode_tile`` probability
    convention) and bilinearly upsampled to the strip's ``(H, W)``.
    """
    x = preprocess_strip(img)
    _, _, h, w = x.shape
    if pt_path is not None:
        logits = _run_pt_heat(x, pt_path)
    else:
        logits = _run_onnx_heat(x, onnx_path)
    if logits.shape[1] != len(HEAT_CLASSES):
        raise ValueError(
            f'heat head has {logits.shape[1]} channels, '
            f'expected {len(HEAT_CLASSES)} {list(HEAT_CLASSES)}')
    probs = torch.sigmoid(logits).float()
    if tuple(probs.shape[2:]) != (h, w):
        probs = torch.nn.functional.interpolate(
            probs, size=(h, w), mode='bilinear', align_corners=False)
    return probs[0]


def extra_channels_for_strip(img: Image.Image,
                             staves_lines: list,
                             target_sp: float = 10.0,
                             onnx_path: str = HEAT_ONNX_DEFAULT,
                             pt_path: str | None = None) -> torch.Tensor:
    """``[8, H, W]`` hybrid extra block for one system strip.

    Rectifies the strip with ``rectify.rectify_system_strip`` (which
    also yields the ``[1, H, W_out]`` canonical staff-line map),
    computes the ``[7, H, W_out]`` heat probabilities on the rectified
    image, and concatenates them with the staff map. Runs on the
    rectified strip so heat peaks and staff rows share coordinates.
    """
    rect_img, staff_ref = rectify_system_strip(
        img, staves_lines, target_sp=target_sp)
    heat = compute_heat_channels(rect_img, onnx_path=onnx_path,
                                 pt_path=pt_path)
    if tuple(heat.shape[1:]) != tuple(staff_ref.shape[1:]):
        heat = torch.nn.functional.interpolate(
            heat.unsqueeze(0), size=tuple(staff_ref.shape[1:]),
            mode='bilinear', align_corners=False)[0]
    return torch.cat([heat, staff_ref], dim=0)


def concat_hybrid_input(rgb: torch.Tensor,
                        extra8: torch.Tensor) -> torch.Tensor:
    """Append an ``[8, H, W]`` extra block to ``[3, H, W]`` -> ``[11,...]``.

    Both inputs must share ``(H, W)``; the RGB side is expected to
    carry ``train.load_image`` standardization while the extra block
    carries raw probabilities / {0, 1} staff rows.
    """
    if rgb.dim() != 3 or rgb.shape[0] != 3:
        raise ValueError(f'rgb must be [3, H, W], got {tuple(rgb.shape)}')
    if extra8.dim() != 3 or extra8.shape[0] != EXTRA_CHANS:
        raise ValueError(
            f'extra must be [{EXTRA_CHANS}, H, W], got {tuple(extra8.shape)}')
    if tuple(rgb.shape[1:]) != tuple(extra8.shape[1:]):
        raise ValueError(
            f'shape mismatch: rgb {tuple(rgb.shape)} vs '
            f'extra {tuple(extra8.shape)}')
    return torch.cat([rgb, extra8], dim=0)


def heat_channels_for_sidecar(
    img_path,
    gray: torch.Tensor,
    heat_net: HeatNet | None = None,
    onnx_path: str = HEAT_ONNX_DEFAULT,
    pt_path: str | None = None,
) -> torch.Tensor:
    """Load cached quantized ``[7, H, W]`` heatmaps if present, else compute on the fly.

    Checks ``<stem>.hybrid.npz`` for zlib-compressed ``uint8`` heatmaps
    first; when absent (including legacy ``.hybrid.pt`` or sidecar-free
    on-the-fly mode), runs ``heat_net`` (preferred, persistent session)
    or :func:`compute_heat_channels` in memory.
    """
    import compact_sidecar as cs

    if gray.dim() == 2:
        gray_4d = gray.unsqueeze(0).unsqueeze(0)
    elif gray.dim() == 3 and gray.shape[0] == 1:
        gray_4d = gray.unsqueeze(0)
    else:
        raise ValueError(
            f'gray must be [H, W] or [1, H, W], got {tuple(gray.shape)}')
    target_hw = (int(gray_4d.shape[2]), int(gray_4d.shape[3]))
    cached = cs.load_compact_heat(
        cs.compact_sidecar_path_for(img_path), target_hw=target_hw)
    if cached is not None:
        return cached
    if heat_net is not None:
        return heat_net.probs(gray_4d)[0].cpu()
    gray_u8 = torch.clamp(torch.round(gray_4d[0, 0] * 255.0),
                          0.0, 255.0).to(torch.uint8).cpu().numpy()
    pil_gray = Image.fromarray(gray_u8, mode='L')
    return compute_heat_channels(pil_gray, onnx_path=onnx_path, pt_path=pt_path)
