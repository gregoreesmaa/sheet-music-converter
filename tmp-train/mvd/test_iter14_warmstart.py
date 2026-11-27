"""CPU unit tests for Iter 14 bekern+v2 Phase A warm-start & vocab expansion (237 -> 265)."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

_MVD_DIR = Path(__file__).resolve().parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import data  # noqa: E402
import train_uni_v2  # noqa: E402
import uni_omr  # noqa: E402
import uni_omr_v2  # noqa: E402

ITER13_EMA_LATEST = Path(r"D:\tmp\mvd-runs\uni-omr-iter13-v2-stage2-24k\uni_omr_v2_ema_latest.pt")


class TestIter14WarmStart(unittest.TestCase):
    def test_special_token_ids_bekern_v2a(self) -> None:
        self.assertEqual(len(data.VOCAB), 261)
        self.assertEqual(uni_omr.PAD_ID, 261)
        self.assertEqual(uni_omr.BOS_ID, 262)
        self.assertEqual(uni_omr.EOS_ID, 263)
        self.assertEqual(uni_omr.SYS_ID, 264)
        self.assertEqual(uni_omr.VOCAB_SIZE, 265)

    def test_load_iter13_ckpt_into_265_vocab_small45m(self) -> None:
        if not ITER13_EMA_LATEST.is_file():
            self.skipTest(f"Iter 13 checkpoint not present on disk: {ITER13_EMA_LATEST}")

        ck = torch.load(str(ITER13_EMA_LATEST), map_location="cpu", weights_only=True)
        old_sd = ck["model"]
        self.assertEqual(tuple(old_sd["tok_emb.weight"].shape), (237, 384))
        self.assertEqual(tuple(old_sd["lm_head.weight"].shape), (237, 384))

        torch.manual_seed(1234)
        cfg = uni_omr_v2.UniOMRv2Config.small_45m()
        self.assertEqual(cfg.vocab_size, 265)
        model = uni_omr_v2.UniOMRv2Net(cfg)

        fresh_tok_new = model.tok_emb.weight[233:261].detach().clone()
        fresh_lm_new = model.lm_head.weight[233:261].detach().clone()

        info = train_uni_v2.load_v2_checkpoint(model, ITER13_EMA_LATEST, encoder_only=False)
        self.assertEqual(info["missing_keys"], [])
        self.assertEqual(info["unexpected_keys"], [])
        self.assertEqual(info["shape_skipped"], [])
        self.assertEqual(sorted(info["vocab_expanded"]), ["lm_head.weight", "tok_emb.weight"])

        for key, fresh_slice in (
            ("tok_emb.weight", fresh_tok_new),
            ("lm_head.weight", fresh_lm_new),
        ):
            new_w = model.state_dict()[key]
            old_w = old_sd[key]
            # 0..232 copied verbatim
            self.assertTrue(
                torch.equal(new_w[:233], old_w[:233]),
                f"{key} regular token rows 0..232 mismatch",
            )
            # Old PAD/BOS/EOS/SYS (233..236) -> new PAD/BOS/EOS/SYS (261..264)
            self.assertTrue(
                torch.equal(new_w[261:265], old_w[233:237]),
                f"{key} special token rows 233..236 -> 261..264 mismatch",
            )
            # Newly appended bekern+v2 Phase A rows (233..260) keep fresh init
            self.assertTrue(
                torch.equal(new_w[233:261], fresh_slice),
                f"{key} newly appended rows 233..260 were unexpectedly modified",
            )
            self.assertTrue(torch.isfinite(new_w).all())

    def test_kd_with_smaller_237_v1_teacher_and_265_student(self) -> None:
        torch.manual_seed(7)
        s_cfg = train_uni_v2.build_v2_config("tiny", vocab_size=265)
        student = uni_omr_v2.UniOMRv2Net(s_cfg)
        t_cfg = uni_omr.UniOMRConfig(
            vocab_size=237,
            d_model=64,
            n_heads=4,
            n_enc_layers=1,
            n_dec_layers=1,
            ffn_mult=2.0,
            mtp_k=4,
        )
        teacher = uni_omr.UniOMRNet(t_cfg).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        images = torch.rand(1, 3, 128, 128)
        # Include regular (0..232), new bekern+v2 (233..260), and new specials (261..264)
        seq = torch.tensor(
            [[262, 10, 233, 250, 260, 264, 15, 263, 261, 261]],
            dtype=torch.long,
        )
        losses, _ = train_uni_v2.forward_and_loss_v2(
            student,
            images,
            seq,
            teacher_v1=teacher,
            kd_alpha=0.30,
            kd_beta=0.5,
            kd_temp=1.5,
            kd_teacher_ce_cap=0.0,
            expr_token_loss_weight=2.5,
            use_bf16=False,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(torch.isfinite(losses["kd_loss"]))
        self.assertGreater(float(losses["kd_loss"].item()), 0.0)
        losses["loss"].backward()


if __name__ == "__main__":
    unittest.main(verbosity=2)
