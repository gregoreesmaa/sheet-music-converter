"""CPU-only unit tests for Iter 14 training upgrades:
1. Collision-free sample_id in UniFullPageDataset / UniStripDataset + HW skip & fast teacher precompute.
2. Expressive/context token CE loss upweighting (--expr-token-loss-weight) composed in single-pass CE.
3. Configurable encoder axial/global layer split (--n-axial-layers, --n-global-layers).
"""

from __future__ import annotations

import os

# Force strict CPU-only execution (never touch GPU while training runs).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image
import torch
import torch.nn.functional as F

_MVD_DIR = Path(__file__).resolve().parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import data  # noqa: E402
import train_uni_v2 as tv2  # noqa: E402
from uni_omr import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    PAD_ID,
    VOCAB_SIZE,
    UniFullPageDataset,
    UniOMRConfig,
    UniOMRNet,
    UniStripDataset,
    make_sample_id,
)
from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net  # noqa: E402


def _write_dummy_page(img_path: Path, bekern_path: Path) -> None:
    img_path.parent.mkdir(parents=True, exist_ok=True)
    bekern_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 64), (255, 255, 255)).save(img_path)
    bekern_path.write_text("clef-G-2\nkey-3\ntime-4-4\nC\n4\nquarter\n", encoding="utf-8")


class TestIter14TrainUpgrades(unittest.TestCase):
    # ------------------------------------------------------------------
    # 1. Collision-free sample_id + precompute HW skip & fast teacher flags
    # ------------------------------------------------------------------
    def test_sample_id_collision_free_across_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_a = root / "corpus_a" / "page_001.png"
            bk_a = root / "corpus_a" / "page_001_s0.bekern"
            img_b = root / "corpus_b" / "page_001.png"
            bk_b = root / "corpus_b" / "page_001_s0.bekern"
            _write_dummy_page(img_a, bk_a)
            _write_dummy_page(img_b, bk_b)

            fp_ds = UniFullPageDataset(
                [(img_a, [bk_a], "printed"), (img_b, [bk_b], "printed")],
                fullpage_max_dim=128,
                max_seq_len=64,
                augment=False,
                return_uint8=True,
            )
            sid_a = fp_ds[0]["sample_id"]
            sid_b = fp_ds[1]["sample_id"]
            self.assertNotEqual(sid_a, sid_b)
            self.assertTrue(sid_a.startswith("page_001_"))
            self.assertTrue(sid_b.startswith("page_001_"))
            # Deterministic across calls
            self.assertEqual(sid_a, make_sample_id(img_a))
            self.assertEqual(sid_b, make_sample_id(img_b))

            st_ds = UniStripDataset(
                [(img_a, bk_a), (img_b, bk_b)],
                strip_height=64,
                max_strip_width=128,
                max_seq_len=64,
                augment=False,
                return_uint8=True,
            )
            self.assertEqual(st_ds[0]["sample_id"], sid_a)
            self.assertEqual(st_ds[1]["sample_id"], sid_b)

    def test_precompute_teacher_cache_skips_hw_and_uses_fast_router_flags(self) -> None:
        torch.manual_seed(0)
        t_cfg = UniOMRConfig(
            d_model=64,
            n_heads=4,
            n_enc_layers=1,
            n_dec_layers=1,
            ffn_mult=2.0,
            dropout=0.0,
            backbone_stride=16,
            use_uv_aux=False,
            mtp_k=1,
            n_bg_summary=4,
            dynamic_routing=True,
        )
        teacher = UniOMRNet(t_cfg).eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_pr1 = root / "dir1" / "page_001.png"
            bk_pr1 = root / "dir1" / "page_001_s0.bekern"
            img_pr2 = root / "dir2" / "page_001.png"
            bk_pr2 = root / "dir2" / "page_001_s0.bekern"
            img_hw = root / "hw_dir" / "hwsim_001.png"
            bk_hw = root / "hw_dir" / "hwsim_001_s0.bekern"
            _write_dummy_page(img_pr1, bk_pr1)
            _write_dummy_page(img_pr2, bk_pr2)
            _write_dummy_page(img_hw, bk_hw)

            fp_ds = UniFullPageDataset(
                [
                    (img_pr1, [bk_pr1], "printed"),
                    (img_pr2, [bk_pr2], "printed"),
                    (img_hw, [bk_hw], "real_hw"),
                ],
                fullpage_max_dim=128,
                max_seq_len=64,
                augment=False,
                return_uint8=True,
            )
            hw_strip_ds = UniStripDataset(
                [(img_hw, bk_hw)],
                strip_height=64,
                max_strip_width=128,
                max_seq_len=64,
                augment=False,
                kind="hw_strip",
                return_uint8=True,
            )

            seen_kwargs: list = []
            orig_fwd = teacher.forward

            def _spy_fwd(*args, **kwargs):
                seen_kwargs.append(dict(kwargs))
                return orig_fwd(*args, **kwargs)

            cache_dir = root / "cache"
            with mock.patch.object(teacher, "forward", side_effect=_spy_fwd):
                cache = tv2.precompute_teacher_cache(
                    teacher=teacher,
                    datasets=[("fullpage", fp_ds), ("hw_strips", hw_strip_ds)],
                    cache_dir=cache_dir,
                    device=torch.device("cpu"),
                    k=8,
                    batch_size=2,
                    use_bf16=False,
                    skip_hw=True,
                    keep_ratio_override=0.45,
                    compute_ink_recall=False,
                )
            try:
                self.assertEqual(len(cache), 2)
                self.assertTrue(cache.has(make_sample_id(img_pr1)))
                self.assertTrue(cache.has(make_sample_id(img_pr2)))
                self.assertFalse(cache.has(make_sample_id(img_hw)))
                self.assertGreaterEqual(len(seen_kwargs), 1)
                for kw in seen_kwargs:
                    self.assertAlmostEqual(kw["keep_ratio_override"], 0.45)
                    self.assertFalse(kw["compute_ink_recall"])
            finally:
                cache.close()

    # ------------------------------------------------------------------
    # 2. Expressive token CE upweighting (--expr-token-loss-weight)
    # ------------------------------------------------------------------
    def test_expr_token_loss_weight_values_and_single_pass_composition(self) -> None:
        parser = tv2.build_arg_parser()
        args_def = parser.parse_args([])
        self.assertAlmostEqual(args_def.expr_token_loss_weight, 1.0)
        args_25 = parser.parse_args(["--expr-token-loss-weight", "2.5"])
        self.assertAlmostEqual(args_25.expr_token_loss_weight, 2.5)

        w_expr = tv2.build_ce_token_weights(
            VOCAB_SIZE,
            state_weight=1.0,
            expr_weight=2.5,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(tuple(w_expr.shape), (VOCAB_SIZE,))

        # Expressive tokens get exact 2.5x
        expected_expr_tokens = [
            "dyn-p",
            "dyn-f",
            "hairpin-cresc-start",
            "hairpin-stop",
            "slur-start",
            "slur-stop",
            "tie-start",
            "tie-stop",
            "fermata",
            "stacc",
            "accent",
            "tenuto",
        ]
        for tok in expected_expr_tokens:
            self.assertIn(tok, data.TOKEN_TO_ID)
            tid = data.TOKEN_TO_ID[tok]
            self.assertAlmostEqual(float(w_expr[tid]), 2.5, places=6, msg=f"token={tok}")

        # Verify art-* / ottava-* matcher helper too
        self.assertTrue(tv2._is_expr_token("art-staccato"))
        self.assertTrue(tv2._is_expr_token("ottava-8va-start"))
        self.assertFalse(tv2._is_expr_token("quarter"))

        # Note tokens & state tokens remain 1.0 when state_weight=1.0
        for tok in ("C", "4", "quarter", "eighth", "rest", "clef-G-2", "time-4-4", "key-3"):
            tid = data.TOKEN_TO_ID[tok]
            self.assertAlmostEqual(float(w_expr[tid]), 1.0, places=6, msg=f"token={tok}")

        # Composition with state_weight=2.0 sets state=2.0, expr=2.5, note=1.0
        w_both = tv2.build_ce_token_weights(
            VOCAB_SIZE,
            state_weight=2.0,
            expr_weight=2.5,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertAlmostEqual(float(w_both[data.TOKEN_TO_ID["clef-G-2"]]), 2.0, places=6)
        self.assertAlmostEqual(float(w_both[data.TOKEN_TO_ID["dyn-f"]]), 2.5, places=6)
        self.assertAlmostEqual(float(w_both[data.TOKEN_TO_ID["quarter"]]), 1.0, places=6)

        # Verify forward_and_loss_v2 runs F.cross_entropy only ONCE on main logits
        torch.manual_seed(11)
        model = UniOMRv2Net(tv2.uni_omr_v2_tiny())
        B, T = 2, 12
        images = torch.rand(B, 3, 128, 192)
        seq = torch.randint(4, VOCAB_SIZE, (B, T + 1), dtype=torch.long)
        seq[:, 0] = BOS_ID
        seq[:, -1] = EOS_ID

        ce_calls = {"main": 0}
        orig_ce = F.cross_entropy

        def _counting_ce(inp, tgt, *args, **kwargs):
            if inp.shape[0] == B * T:
                ce_calls["main"] += 1
                self.assertIsNotNone(kwargs.get("weight"))
                self.assertAlmostEqual(
                    float(kwargs["weight"][data.TOKEN_TO_ID["dyn-f"]]), 2.5, places=6
                )
            return orig_ce(inp, tgt, *args, **kwargs)

        with mock.patch.object(F, "cross_entropy", side_effect=_counting_ce):
            tv2.forward_and_loss_v2(
                model,
                images,
                seq,
                state_token_loss_weight=2.0,
                expr_token_loss_weight=2.5,
                kd_alpha=0.0,
                cass_prob=0.0,
            )
        self.assertEqual(ce_calls["main"], 1)

    # ------------------------------------------------------------------
    # 3. Configurable encoder layer split (--n-axial-layers, --n-global-layers)
    # ------------------------------------------------------------------
    def test_configurable_axial_global_layer_split(self) -> None:
        parser = tv2.build_arg_parser()
        args_def = parser.parse_args([])
        self.assertIsNone(args_def.n_axial_layers)
        self.assertIsNone(args_def.n_global_layers)

        # Default small_45m preserves 2 axial + 4 global
        cfg_def = tv2.build_v2_config("small_45m")
        self.assertEqual(cfg_def.n_axial_layers, 2)
        self.assertEqual(cfg_def.n_global_layers, 4)

        args_ax1_gl5 = parser.parse_args(
            [
                "--preset", "tiny",
                "--n-axial-layers", "1",
                "--n-global-layers", "5",
                "--device", "cpu",
                "--no-bf16",
                "--dry-run",
                "--steps", "1",
                "--batch-size", "1",
                "--grad-accum", "1",
                "--seq-len", "16",
                "--prefetch-threads", "0",
            ]
        )
        self.assertEqual(args_ax1_gl5.n_axial_layers, 1)
        self.assertEqual(args_ax1_gl5.n_global_layers, 5)

        captured: dict = {}
        orig_init = UniOMRv2Net.__init__

        def _spy_init(mod_self, cfg: UniOMRv2Config) -> None:
            orig_init(mod_self, cfg)
            captured["n_axial"] = len(mod_self.encoder.axial_layers)
            captured["n_global"] = len(mod_self.encoder.global_layers)

        with mock.patch.object(UniOMRv2Net, "__init__", _spy_init):
            res = tv2.run_train_v2(args_ax1_gl5)
        self.assertEqual(res["steps"], 1)
        self.assertEqual(captured["n_axial"], 1)
        self.assertEqual(captured["n_global"], 5)


if __name__ == "__main__":
    unittest.main()
