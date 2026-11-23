#!/usr/bin/env python3
"""Unsupervised ConvNeXt-Tiny Masked Autoencoder (MAE) Pre-Training for OMR.

Pure unsupervised visual pre-training on unlabeled sheet-music strips/pages
(zero MusicXML or token labels required):

- Wraps ``MvdNet``'s ConvNeXt-Tiny ``4x8`` encoder (``in_chans=3`` or ``4``).
- Applies random ``60%`` block masking at ``16x16`` patch granularity on the
  input image (replacing masked patches with a learnable mask token).
- Passes the masked image through the ``4x8`` ConvNeXt-Tiny encoder
  (outputting ``[B, 768, H/4, W/8]``).
- Reconstructs pixels via a lightweight 2-layer transposed-conv decoder
  (``stride=(2, 4)`` then ``stride=(2, 2)`` -> ``[B, in_chans, H, W]``).
- Computes per-patch normalized MSE loss strictly on masked patches.
- Atomically saves ``encoder_mae.pt`` (clean ``encoder.state_dict()`` whose
  keys load directly into ``MvdNet.encoder`` via ``train.load_weights_compat``).
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MVD_DIR = Path(__file__).resolve().parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import data  # noqa: E402
import model  # noqa: E402
import train  # noqa: E402

DEFAULT_MANIFEST = Path(r'D:\tmp\unlabeled-10k\unlabeled_10k_manifest.jsonl')


def make_block_mask(
    batch_size: int,
    height: int,
    width: int,
    patch_size: int = 16,
    mask_ratio: float = 0.60,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random per-sample block masks at ``patch_size x patch_size`` grid.

    Returns:
        ``(patch_mask, pixel_mask)`` where:
          - ``patch_mask`` is ``[B, gh, gw]`` float32 in ``{0.0, 1.0}``
            (``1.0`` = masked).
          - ``pixel_mask`` is ``[B, 1, H, W]`` float32 in ``{0.0, 1.0}``.
    """
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f'H={height}, W={width} must be divisible by patch_size={patch_size}'
        )
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError(f'mask_ratio must be in (0, 1), got {mask_ratio}')

    gh = height // patch_size
    gw = width // patch_size
    n_patches = gh * gw
    n_mask = max(1, min(n_patches - 1, int(round(n_patches * mask_ratio))))

    noise = torch.rand(
        (batch_size, n_patches), device=device, generator=generator,
    )
    order = torch.argsort(noise, dim=1)
    mask_flat = torch.zeros((batch_size, n_patches), device=device, dtype=torch.float32)
    mask_flat.scatter_(1, order[:, :n_mask], 1.0)

    patch_mask = mask_flat.view(batch_size, gh, gw)
    pixel_mask = (
        patch_mask
        .unsqueeze(1)
        .repeat_interleave(patch_size, dim=2)
        .repeat_interleave(patch_size, dim=3)
    )
    return patch_mask, pixel_mask


def masked_normalized_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    patch_mask: torch.Tensor,
    patch_size: int = 16,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-patch normalized MSE loss strictly on masked patches.

    Args:
        pred: ``[B, C, H, W]`` reconstructed pixel predictions.
        target: ``[B, C, H, W]`` original unmasked input image.
        patch_mask: ``[B, H//P, W//P]`` float/bool tensor (``1`` = masked).
        patch_size: square patch size ``P`` (default ``16``).
    """
    if pred.shape != target.shape:
        raise ValueError(
            f'pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}'
        )
    b, c, h, w = target.shape
    p = int(patch_size)
    gh, gw = h // p, w // p
    if patch_mask.shape != (b, gh, gw):
        raise ValueError(
            f'patch_mask shape {tuple(patch_mask.shape)} != {(b, gh, gw)}'
        )

    # Reshape into [B, gh, gw, C, P, P]
    tgt_p = (
        target.float()
        .view(b, c, gh, p, gw, p)
        .permute(0, 2, 4, 1, 3, 5)
    )
    pred_p = (
        pred.float()
        .view(b, c, gh, p, gw, p)
        .permute(0, 2, 4, 1, 3, 5)
    )

    mean = tgt_p.mean(dim=(3, 4, 5), keepdim=True)
    var = tgt_p.var(dim=(3, 4, 5), unbiased=False, keepdim=True)
    norm_tgt = (tgt_p - mean) / torch.sqrt(var + eps)

    sq_err = (pred_p - norm_tgt).pow(2).mean(dim=(3, 4, 5))  # [B, gh, gw]
    m = patch_mask.to(device=pred.device, dtype=torch.float32)
    denom = m.sum().clamp_min(1.0)
    return (sq_err * m).sum() / denom


class ConvNeXtMAE(nn.Module):
    """Masked Autoencoder wrapping ``MvdNet``'s ConvNeXt-Tiny ``4x8`` encoder."""

    def __init__(
        self,
        in_chans: int = 3,
        patch_size: int = 16,
        mask_ratio: float = 0.60,
        decoder_dim: int = 192,
        pretrained: bool = False,
    ):
        super().__init__()
        if in_chans not in (3, 4):
            raise ValueError(f'in_chans must be 3 or 4, got {in_chans}')
        self.in_chans = int(in_chans)
        self.patch_size = int(patch_size)
        self.mask_ratio = float(mask_ratio)

        # Build the exact MvdNet 4x8 ConvNeXt-Tiny encoder so state_dict keys
        # and shapes match MvdNet.encoder 1-to-1.
        base = model.MvdNet(
            vocab=len(data.VOCAB) + 3,
            anisotropic=True,
            stride='4x8',
            pretrained=pretrained,
            extra_chans=self.in_chans - 3,
        )
        self.encoder = base.encoder

        # Learnable mask token replacing masked pixels.
        self.mask_token = nn.Parameter(torch.zeros(1, self.in_chans, 1, 1))

        # Lightweight 2-layer transposed-conv decoder:
        # [B, 768, H/4, W/8] -> [B, decoder_dim, H/2, W/2] -> [B, in_chans, H, W]
        gn_groups = 16 if decoder_dim % 16 == 0 else 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                768, decoder_dim, kernel_size=(2, 4), stride=(2, 4),
            ),
            nn.GroupNorm(gn_groups, decoder_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                decoder_dim, self.in_chans, kernel_size=(2, 2), stride=(2, 2),
            ),
        )

    def forward(
        self,
        images: torch.Tensor,
        mask_ratio: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward MAE pass.

        Returns:
            ``(loss, pred, patch_mask)`` where ``loss`` is scalar normalized
            MSE strictly on masked patches, ``pred`` is ``[B, C, H, W]``, and
            ``patch_mask`` is ``[B, H//P, W//P]``.
        """
        b, c, h, w = images.shape
        if c != self.in_chans:
            raise ValueError(f'expected {self.in_chans} channels, got {c}')

        ratio = self.mask_ratio if mask_ratio is None else float(mask_ratio)
        patch_mask, pixel_mask = make_block_mask(
            b, h, w,
            patch_size=self.patch_size,
            mask_ratio=ratio,
            device=images.device,
            generator=generator,
        )
        pm = pixel_mask.to(dtype=images.dtype)
        x_masked = images * (1.0 - pm) + self.mask_token.to(dtype=images.dtype) * pm
        feats = self.encoder(x_masked)          # [B, 768, H/4, W/8]
        pred = self.decoder(feats)              # [B, C, H, W]
        loss = masked_normalized_mse(
            pred, images, patch_mask, patch_size=self.patch_size,
        )
        return loss, pred, patch_mask


def save_encoder_checkpoint(mae: ConvNeXtMAE, path: Path) -> Path:
    """Atomically save clean ``encoder_mae.pt`` loadable into ``MvdNet.encoder``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc_state = {
        k: v.detach().cpu() for k, v in mae.encoder.state_dict().items()
    }
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(enc_state, tmp)
    os.replace(tmp, path)
    return path


def load_mae_encoder_into_mvdnet(
    net: model.MvdNet,
    ckpt_path: Path,
    map_location: str = 'cpu',
) -> str:
    """Load an ``encoder_mae.pt`` checkpoint into ``net.encoder`` via ``load_weights_compat``."""
    state = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    return train.load_weights_compat(net.encoder, state)


class UnlabeledSheetDataset(Dataset):
    """Unlabeled sheet-music image dataset reading ``unlabeled_10k_manifest.jsonl``."""

    def __init__(
        self,
        manifest_path: Path,
        height: int = 256,
        width: int = 512,
        in_chans: int = 3,
    ):
        self.height = int(height)
        self.width = int(width)
        self.in_chans = int(in_chans)
        self.paths: List[Path] = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                p = Path(obj['path'])
                if p.is_file():
                    self.paths.append(p)
        if not self.paths:
            raise ValueError(f'No existing image files found in {manifest_path}')

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert('RGB')
        # Resize preserving height, then center/random crop or pad to self.width
        w, h = img.size
        nw = max(1, round(w * self.height / max(1, h)))
        img = img.resize((nw, self.height), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1)
        t = train.standardize_rgb(t)

        if t.shape[2] >= self.width:
            max_off = t.shape[2] - self.width
            off = random.randint(0, max_off) if max_off > 0 else 0
            t = t[:, :, off:off + self.width]
        else:
            pad_w = self.width - t.shape[2]
            fill = train.WHITE.expand(3, self.height, pad_w)
            t = torch.cat([t, fill], dim=2)

        if self.in_chans == 4:
            # 4th channel: inverted luminance proxy for ink density when no
            # staff-grid sidecar is available.
            gray = t.mean(dim=0, keepdim=True)
            t = torch.cat([t, gray], dim=0)
        return t


def _save_resume_state(
    mae: ConvNeXtMAE,
    opt: torch.optim.Optimizer,
    step: int,
    config: Dict[str, object],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save({
        'step': step,
        'model': {k: v.detach().cpu() for k, v in mae.state_dict().items()},
        'optimizer': opt.state_dict(),
        'config': config,
    }, tmp)
    os.replace(tmp, path)


def train_mae(
    manifest_path: Path,
    out_dir: Path,
    steps: int = 1000,
    batch_size: int = 8,
    in_chans: int = 3,
    height: int = 256,
    width: int = 512,
    patch_size: int = 16,
    mask_ratio: float = 0.60,
    lr: float = 1.5e-4,
    weight_decay: float = 0.05,
    bf16: bool = False,
    resume: bool = False,
    save_every: int = 500,
    seed: int = 42,
    device: str = 'cpu',
    num_workers: int = 0,
) -> Dict[str, object]:
    """Run MAE pre-training loop and save ``encoder_mae.pt``."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dev = torch.device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / 'mae_resume.pt'
    encoder_path = out_dir / 'encoder_mae.pt'

    ds = UnlabeledSheetDataset(
        manifest_path, height=height, width=width, in_chans=in_chans,
    )
    loader = DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    mae = ConvNeXtMAE(
        in_chans=in_chans,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
    ).to(dev)
    opt = torch.optim.AdamW(mae.parameters(), lr=lr, weight_decay=weight_decay)

    start_step = 0
    if resume and resume_path.is_file():
        ckpt = torch.load(resume_path, map_location=dev, weights_only=False)
        mae.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        start_step = int(ckpt.get('step', 0))

    config = {
        'in_chans': in_chans,
        'height': height,
        'width': width,
        'patch_size': patch_size,
        'mask_ratio': mask_ratio,
        'lr': lr,
        'weight_decay': weight_decay,
        'bf16': bf16,
        'seed': seed,
    }

    use_amp = bool(bf16)
    amp_device = 'cuda' if dev.type == 'cuda' else 'cpu'
    mae.train()

    step = start_step
    last_loss = float('nan')
    data_iter = iter(loader)
    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = batch.to(dev)
        opt.zero_grad(set_to_none=True)
        ctx = (
            torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16)
            if use_amp else nullcontext()
        )
        with ctx:
            loss, _, _ = mae(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite MAE loss at step {step + 1}: {loss}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
        opt.step()

        step += 1
        last_loss = float(loss.item())
        if save_every > 0 and (step % save_every == 0):
            save_encoder_checkpoint(mae, encoder_path)
            _save_resume_state(mae, opt, step, config, resume_path)

    save_encoder_checkpoint(mae, encoder_path)
    _save_resume_state(mae, opt, step, config, resume_path)
    return {
        'steps': step,
        'last_loss': last_loss,
        'encoder_ckpt': str(encoder_path),
        'resume_ckpt': str(resume_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description='Unsupervised ConvNeXt-Tiny 4x8 MAE pre-training for OMR.',
    )
    ap.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--in-chans', type=int, choices=(3, 4), default=3)
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--patch-size', type=int, default=16)
    ap.add_argument('--mask-ratio', type=float, default=0.60)
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--num-workers', type=int, default=0)
    args = ap.parse_args(argv)

    summary = train_mae(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        in_chans=args.in_chans,
        height=args.height,
        width=args.width,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        lr=args.lr,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        resume=args.resume,
        save_every=args.save_every,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
