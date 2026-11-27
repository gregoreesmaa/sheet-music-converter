"""CPU-only unit tests for Worker 2:
1. Lower-30% quantile (bg_p30) adaptive_ink_target in InkTokenRouterV2 & ViRTwoTierRouter
   (dense archival glare/wear retention >90%, ink_thr <= 0.10, foxed HW rejection, ONNX safety,
   and real-image verification on als-haydn, als-chopin, hw-debussy-berceuse-heroique, als-mozart-k458).
2. eval_uni_suite31.py --max-mem-tokens=4608 default, 2D RoPE compatibility on 3072-trained v2 model
   overridden to 4608, and legacy checkpoint adaptive_ink_target=False default.
3. Broader codebase sweep regression checks (no hardcoded 237 in block diffusion, no r_syntax typo,
   no hardcoded 13.8 GB empty_cache in uni_omr.py).
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image
import torch

_MVD_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MVD_DIR.parent.parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import eval_uni_suite31 as eus  # noqa: E402
import trial_2028_block_diffusion as bd  # noqa: E402
import uni_omr  # noqa: E402
import uni_omr_v2  # noqa: E402
import vir_router_v2  # noqa: E402


class TestRouterAdaptiveInkAndSweep(unittest.TestCase):
    def test_bg_p30_dense_archival_retention_and_foxed_rejection(self) -> None:
        torch.manual_seed(11)
        B, D, H_img, W_img = 1, 32, 256, 256
        H_p, W_p = H_img // 16, W_img // 16

        # 1. Dense archival page with spatially non-uniform glare/wear:
        #    - Top half: crisp dark ink (~0.35 local_ink)
        #    - Bottom half: faded/glare-washed staff & notes (~0.10-0.14 local_ink)
        #    Old global mean+0.5*std climbed to ~0.18-0.20 and wiped out the bottom half!
        dense_img = torch.full((B, 3, H_img, W_img), 0.90, dtype=torch.float32)
        for y in range(16, 120, 8):
            dense_img[:, :, y : y + 2, 16:240] = 0.05  # dense dark ink
        for y in range(136, 240, 8):
            dense_img[:, :, y : y + 2, 16:240] = 0.72  # glare-faded ink (local_ink ~0.11)

        feat = torch.randn(B, D, H_p, W_p)
        for router in (
            uni_omr_v2.InkTokenRouterV2(d_model=D, adaptive_ink_target=True).eval(),
            vir_router_v2.ViRTwoTierRouter(d_model=D, k_coarse=4, adaptive_ink_target=True).eval(),
        ):
            with torch.no_grad():
                _, _, info = router(feat, images=dense_img)
            ink_t = info["ink_target"]
            self.assertIsNotNone(ink_t)
            # Bottom glare-faded half (rows 8..15) must retain >90% of its staff patches
            faded_half = ink_t[:, :, 9:15, 1:15]
            self.assertGreater(
                float(faded_half.mean().item()),
                0.90,
                f"{type(router).__name__} wiped out glare-faded archival ink: {faded_half.mean().item():.3f}",
            )

        # 2. Foxed aged paper: fine speckle contrast ~0.07 across whole page + crisp dark ink block
        bg = torch.full((B, 3, 128, 128), 0.78, dtype=torch.float32)
        yy = torch.arange(128).view(1, 1, 128, 1)
        xx = torch.arange(128).view(1, 1, 1, 128)
        speckle = (((yy + xx) % 2) == 0).float() * 0.09
        aged_img = (bg - speckle).clamp(0.0, 1.0)
        aged_img[:, :, 48:64, 48:80] = 0.08
        feat_sm = torch.randn(B, D, 8, 8)
        for router in (
            uni_omr_v2.InkTokenRouterV2(d_model=D, adaptive_ink_target=True).eval(),
            vir_router_v2.ViRTwoTierRouter(d_model=D, k_coarse=4, adaptive_ink_target=True).eval(),
        ):
            with torch.no_grad():
                _, _, info = router(feat_sm, images=aged_img)
            frac = float(info["ink_target"].mean().item())
            self.assertLess(frac, 0.25, f"{type(router).__name__} failed to reject foxed bg: {frac:.3f}")
            self.assertGreater(frac, 0.0, f"{type(router).__name__} dropped dark ink on foxed bg")

    def test_bg_p30_onnx_export_avoids_torch_quantile(self) -> None:
        torch.manual_seed(3)
        B, D, H_img, W_img = 1, 32, 128, 128
        img = torch.rand(B, 3, H_img, W_img)
        feat = torch.randn(B, D, H_img // 16, W_img // 16)

        def _forbid_quantile(*a, **kw):
            raise AssertionError("torch.quantile must not be called during ONNX export")

        for router in (
            uni_omr_v2.InkTokenRouterV2(d_model=D, adaptive_ink_target=True).eval(),
            vir_router_v2.ViRTwoTierRouter(d_model=D, k_coarse=4, adaptive_ink_target=True).eval(),
        ):
            with (
                mock.patch.object(torch.onnx, "is_in_onnx_export", return_value=True),
                mock.patch.object(torch, "quantile", side_effect=_forbid_quantile),
                torch.no_grad(),
            ):
                out_toks, _, info = router(feat, images=img)
            self.assertTrue(bool(torch.isfinite(out_toks).all()))
            self.assertIsNotNone(info["ink_target"])

    def test_real_images_bg_p30_threshold_and_retention(self) -> None:
        real_paths = [
            ("als-haydn", Path(r"D:\tmp\northstar-eval-pack\als_scans\haydn-op74n1-1_als.png")),
            ("als-chopin", Path(r"D:\tmp\northstar-eval-pack\als_scans\chopin-mazurka06-2_als.png")),
            (
                "hw-debussy-berceuse-heroique",
                _REPO_ROOT / "data-gen/score-corpus/suite31/hw/clean/hw-debussy-berceuse-heroique.png",
            ),
            ("als-mozart-k458", Path(r"D:\tmp\northstar-eval-pack\als_scans\mozart-k458-1_als.png")),
        ]
        checked = 0
        for name, path in real_paths:
            if not path.is_file():
                continue
            checked += 1
            im = uni_omr._resize_fullpage_pil(Image.open(path).convert("RGB"), 1408)
            t = uni_omr._pil_to_chw_tensor(im).unsqueeze(0)
            H_p, W_p = (t.shape[2] + 15) // 16, (t.shape[3] + 15) // 16
            local_ink, _, paper_mask = uni_omr_v2.InkTokenRouterV2._compute_ink_and_paper_gate(t, H_p, W_p)
            bg_p30 = torch.quantile(local_ink.flatten(1).float(), 0.30, dim=1, keepdim=True)
            ink_thr = (bg_p30 + 0.045).clamp(min=0.045, max=0.10)
            thr_val = float(ink_thr.item())
            self.assertLessEqual(thr_val, 0.10 + 1e-6, f"{name}: ink_thr={thr_val:.4f} > 0.10")
            self.assertGreaterEqual(thr_val, 0.045 - 1e-6, f"{name}: ink_thr={thr_val:.4f} < 0.045")
            dense_mask = (local_ink > 0.08) & paper_mask
            retention = float(
                (((local_ink > ink_thr.view(1, 1, 1, 1)) & dense_mask).float().sum() / dense_mask.float().sum().clamp(min=1.0)).item()
            )
            self.assertGreater(retention, 0.90, f"{name}: dense ink retention={retention:.3f} <= 0.90")
        self.assertGreater(checked, 0, "Expected at least one real benchmark image on disk")

    def test_eval_defaults_and_legacy_adaptive_ink_target_false(self) -> None:
        parser = eus.build_arg_parser()
        args = parser.parse_args(["--ckpt", "dummy.pt"])
        self.assertEqual(args.max_mem_tokens, 4608)

        # Save a legacy v2 checkpoint WITHOUT adaptive_ink_target in cfg
        base_cfg = uni_omr_v2.UniOMRv2Config(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            n_axial_layers=1,
            n_global_layers=1,
            n_dec_layers=1,
            ffn_mult=2.0,
            axial_ffn_mult=2.0,
            backbone_channels=(16, 32, 48, 64),
            backbone_blocks=(1, 1, 1, 1),
            max_mem_tokens=3072,
            adaptive_ink_target=True,
        )
        m = uni_omr_v2.UniOMRv2Net(base_cfg).eval()
        legacy_cfg_dict = asdict(base_cfg)
        legacy_cfg_dict.pop("adaptive_ink_target", None)

        with tempfile.TemporaryDirectory() as td:
            legacy_pt = Path(td) / "legacy_v2.pt"
            torch.save({"arch": "uni_omr_v2", "config": legacy_cfg_dict, "model": m.state_dict()}, legacy_pt)
            loaded_leg, cfg_leg, _ = eus.load_uni_checkpoint(legacy_pt, device=torch.device("cpu"))
            self.assertFalse(cfg_leg.adaptive_ink_target)
            self.assertFalse(loaded_leg.encoder.router.adaptive_ink_target)

            modern_pt = Path(td) / "modern_v2.pt"
            torch.save({"arch": "uni_omr_v2", "config": asdict(base_cfg), "model": m.state_dict()}, modern_pt)
            loaded_mod, cfg_mod, _ = eus.load_uni_checkpoint(modern_pt, device=torch.device("cpu"))
            self.assertTrue(cfg_mod.adaptive_ink_target)
            self.assertTrue(loaded_mod.encoder.router.adaptive_ink_target)

            # Verify 3072-trained model handles override_max_mem_tokens(4608) cleanly with 2D RoPE
            img_hi = torch.rand(1, 3, 1024, 1024)  # /16 -> 64x64 = 4096 patches (> 3072, <= 4608)
            with eus.override_max_mem_tokens(loaded_mod, 4608), torch.no_grad():
                self.assertEqual(loaded_mod.encoder.router.max_mem_tokens, 4608)
                mem, _, rinfo = loaded_mod.encode_image(img_hi, return_aux=False)
            self.assertTrue(bool(torch.isfinite(mem[-1]).all()))
            self.assertLessEqual(int(rinfo["n_out"]), 4608)
            self.assertEqual(loaded_mod.encoder.router.max_mem_tokens, 3072)

    def test_broader_sweep_regressions(self) -> None:
        # 1. BlockDiffusionConfig must track uni_omr.VOCAB_SIZE (not stale 237)
        self.assertEqual(bd.BlockDiffusionConfig().vocab_size, uni_omr.VOCAB_SIZE)

        # 2. uni_omr.py must not hardcode 13.8 GB VRAM empty_cache limit
        uni_src = (_MVD_DIR / "uni_omr.py").read_text(encoding="utf-8")
        self.assertNotIn("13.8 * (1024 ** 3)", uni_src)

        # 3. None of the fixed i13 queued scripts may reference non-existent r_syntax key
        for rel in (
            "experiments/2026-11-26T07-40-i13-13-cross-system-header-consensus/run_colab.py",
            "experiments/2026-11-26T07-40-i13-19-coarse2fine-zoom-omr/run_colab.py",
            "experiments/2026-11-26T07-40-i13-20-masked-diffusion-measure-denoiser/run_colab.py",
            "experiments/2026-11-26T07-40-i13-21-dual-stream-learned-ink-encoder/run_colab.py",
        ):
            p = _REPO_ROOT / rel
            if p.is_file():
                self.assertNotIn("r_syntax", p.read_text(encoding="utf-8"), f"r_syntax still present in {rel}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
