#!/usr/bin/env python3
"""CPU-only unit tests for tmp-train/mvd/mae_pretrain.py."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

MVD_DIR = Path(__file__).resolve().parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import data  # noqa: E402
import mae_pretrain as mae_mod  # noqa: E402
import model  # noqa: E402
import train  # noqa: E402


class TestMaePretrain(unittest.TestCase):
    def test_block_mask_ratio_and_granularity(self):
        gen = torch.Generator().manual_seed(123)
        patch_mask, pixel_mask = mae_mod.make_block_mask(
            batch_size=4,
            height=64,
            width=128,
            patch_size=16,
            mask_ratio=0.60,
            generator=gen,
        )
        self.assertEqual(tuple(patch_mask.shape), (4, 4, 8))
        self.assertEqual(tuple(pixel_mask.shape), (4, 1, 64, 128))
        # 32 patches * 0.60 = 19.2 -> 19 masked patches per sample -> 0.59375
        ratio = float(patch_mask.mean().item())
        self.assertAlmostEqual(ratio, 19.0 / 32.0, places=5)
        self.assertAlmostEqual(float(pixel_mask.mean().item()), ratio, places=5)
        # Verify each 16x16 block in pixel_mask is constant (0 or 1)
        blocks = pixel_mask.view(4, 1, 4, 16, 8, 16)
        self.assertEqual(float(blocks.std(dim=(3, 5)).max().item()), 0.0)

    def test_masked_only_normalized_mse(self):
        torch.manual_seed(0)
        target = torch.randn(2, 3, 32, 64)
        pred = torch.randn(2, 3, 32, 64)
        patch_mask, pixel_mask = mae_mod.make_block_mask(
            batch_size=2, height=32, width=64, patch_size=16, mask_ratio=0.60,
        )
        loss_a = mae_mod.masked_normalized_mse(pred, target, patch_mask, patch_size=16)

        # Perturb ONLY unmasked pixels in pred -> loss must be bit-identical
        pred_unmasked_mod = pred + (1.0 - pixel_mask) * 100.0
        loss_b = mae_mod.masked_normalized_mse(
            pred_unmasked_mod, target, patch_mask, patch_size=16,
        )
        self.assertAlmostEqual(float(loss_a.item()), float(loss_b.item()), places=6)

        # Perturb masked pixels -> loss must change significantly
        pred_masked_mod = pred + pixel_mask * 5.0
        loss_c = mae_mod.masked_normalized_mse(
            pred_masked_mod, target, patch_mask, patch_size=16,
        )
        self.assertNotAlmostEqual(float(loss_a.item()), float(loss_c.item()), places=3)

    def test_forward_backward_cpu_3ch_and_4ch(self):
        for in_chans in (3, 4):
            with self.subTest(in_chans=in_chans):
                torch.manual_seed(42 + in_chans)
                mae = mae_mod.ConvNeXtMAE(
                    in_chans=in_chans,
                    patch_size=16,
                    mask_ratio=0.60,
                    decoder_dim=64,
                )
                x = torch.randn(2, in_chans, 64, 128)
                loss, pred, pmask = mae(x)
                self.assertEqual(tuple(pred.shape), (2, in_chans, 64, 128))
                self.assertEqual(tuple(pmask.shape), (2, 4, 8))
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(float(loss.item()), 0.0)

                loss.backward()
                self.assertIsNotNone(mae.mask_token.grad)
                self.assertTrue(torch.isfinite(mae.mask_token.grad).all())
                stem_grad = mae.encoder[0][0].weight.grad
                self.assertIsNotNone(stem_grad)
                self.assertTrue(torch.isfinite(stem_grad).all())
                self.assertGreater(float(stem_grad.abs().sum().item()), 0.0)

    def test_checkpoint_loads_cleanly_into_mvdnet_encoder(self):
        for in_chans in (3, 4):
            with self.subTest(in_chans=in_chans):
                torch.manual_seed(11 + in_chans)
                mae = mae_mod.ConvNeXtMAE(
                    in_chans=in_chans, patch_size=16, mask_ratio=0.60, decoder_dim=64,
                )
                with tempfile.TemporaryDirectory(prefix='mae_ckpt_') as td:
                    ckpt = Path(td) / 'encoder_mae.pt'
                    mae_mod.save_encoder_checkpoint(mae, ckpt)
                    self.assertTrue(ckpt.is_file())
                    self.assertEqual(list(Path(td).glob('*.tmp')), [])

                    mvd = model.MvdNet(
                        vocab=len(data.VOCAB) + 3,
                        d_model=64,
                        layers=1,
                        heads=4,
                        dim_ff=128,
                        anisotropic=True,
                        stride='4x8',
                        extra_chans=in_chans - 3,
                        pretrained=False,
                    )
                    state = torch.load(ckpt, map_location='cpu', weights_only=True)
                    how = train.load_weights_compat(mvd.encoder, state)
                    self.assertEqual(how, 'strict')
                    # Verify exact tensor equality for stem and deepest block
                    self.assertTrue(
                        torch.equal(
                            mvd.encoder[0][0].weight,
                            mae.encoder[0][0].weight,
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            mvd.encoder[7][2].block[5].weight,
                            mae.encoder[7][2].block[5].weight,
                        )
                    )

    def test_train_mae_loop_and_resume_cpu(self):
        with tempfile.TemporaryDirectory(prefix='mae_train_') as td:
            root = Path(td)
            img_paths = []
            rng = np.random.default_rng(0)
            for idx in range(2):
                p = root / f'sheet_{idx}.png'
                arr = rng.integers(180, 255, size=(64, 128, 3), dtype=np.uint8)
                Image.fromarray(arr).save(p)
                img_paths.append(p)

            mf = root / 'unlabeled_manifest.jsonl'
            mf.write_text(
                '\n'.join(
                    json.dumps({'id': f's{i}', 'domain': 'test', 'path': str(p)})
                    for i, p in enumerate(img_paths)
                ) + '\n',
                encoding='utf-8',
            )
            out_dir = root / 'mae_out'
            res1 = mae_mod.train_mae(
                manifest_path=mf,
                out_dir=out_dir,
                steps=1,
                batch_size=2,
                in_chans=3,
                height=64,
                width=128,
                patch_size=16,
                mask_ratio=0.60,
                save_every=1,
                device='cpu',
            )
            self.assertEqual(res1['steps'], 1)
            self.assertTrue(Path(res1['encoder_ckpt']).is_file())
            self.assertTrue(Path(res1['resume_ckpt']).is_file())

            # Resume for 1 more step
            res2 = mae_mod.train_mae(
                manifest_path=mf,
                out_dir=out_dir,
                steps=2,
                batch_size=2,
                in_chans=3,
                height=64,
                width=128,
                patch_size=16,
                mask_ratio=0.60,
                resume=True,
                save_every=1,
                device='cpu',
            )
            self.assertEqual(res2['steps'], 2)

            # Load resumed encoder_mae.pt cleanly into MvdNet.encoder
            mvd = model.MvdNet(
                vocab=len(data.VOCAB) + 3,
                d_model=64,
                layers=1,
                heads=4,
                dim_ff=128,
                anisotropic=True,
                stride='4x8',
                pretrained=False,
            )
            how = mae_mod.load_mae_encoder_into_mvdnet(
                mvd, Path(res2['encoder_ckpt']), map_location='cpu',
            )
            self.assertEqual(how, 'strict')


if __name__ == '__main__':
    unittest.main()
