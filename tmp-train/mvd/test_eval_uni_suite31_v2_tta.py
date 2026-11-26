"""CPU unit tests for UniOMR v2 checkpoint support and scale-only best_fsa_beat TTA in eval_uni_suite31.py."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest

import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import eval_uni_suite31 as ev  # noqa: E402
import uni_omr  # noqa: E402
import uni_omr_v2  # noqa: E402


def _make_tiny_v2_cfg() -> uni_omr_v2.UniOMRv2Config:
    return uni_omr_v2.UniOMRv2Config(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_axial_layers=1,
        n_global_layers=1,
        n_dec_layers=2,
        ffn_mult=2.0,
        axial_ffn_mult=2.0,
        backbone_channels=(16, 32, 48, 64),
        backbone_blocks=(1, 1, 1, 1),
        backbone_expand=2.0,
        router_hidden=32,
        max_mem_tokens=128,
        mtp_k=4,
    )


def _make_tiny_v1_cfg() -> uni_omr.UniOMRConfig:
    return uni_omr.UniOMRConfig(
        d_model=64,
        n_heads=4,
        n_enc_layers=1,
        n_dec_layers=2,
        ffn_mult=2.0,
        mtp_k=4,
    )


class TestEvalUniSuite31V2AndTTA(unittest.TestCase):
    def test_load_v2_checkpoint_and_decode_modes(self) -> None:
        torch.manual_seed(7)
        cfg = _make_tiny_v2_cfg()
        orig = uni_omr_v2.UniOMRv2Net(cfg).eval()

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            # Exact payload shape written by train_uni_v2.py (no explicit "arch" key)
            ckpt_train_v2 = tmp_path / "uni_omr_v2_ema_best.pt"
            torch.save(
                {
                    "step": 42,
                    "val_ser": 0.123,
                    "cfg": asdict(cfg),
                    "config": asdict(cfg),
                    "model": orig.state_dict(),
                },
                ckpt_train_v2,
            )

            device = torch.device("cpu")
            model, loaded_cfg, meta = ev.load_uni_checkpoint(ckpt_train_v2, device=device)
            self.assertIsInstance(model, uni_omr_v2.UniOMRv2Net)
            self.assertIsInstance(loaded_cfg, uni_omr_v2.UniOMRv2Config)
            self.assertEqual(meta["arch"], "uni_omr_v2")
            self.assertEqual(meta["step"], 42)
            self.assertEqual(meta["missing_keys"], [])
            self.assertEqual(meta["unexpected_keys"], [])

            # Explicit arch="uni_omr_v2" payload also works (with JSON-style list channels)
            cfg_json = asdict(cfg)
            cfg_json["backbone_channels"] = list(cfg_json["backbone_channels"])
            cfg_json["backbone_blocks"] = list(cfg_json["backbone_blocks"])
            ckpt_arch = tmp_path / "uni_omr_v2_explicit_arch.pt"
            torch.save(
                {"arch": "uni_omr_v2", "config": cfg_json, "model": orig.state_dict()},
                ckpt_arch,
            )
            model2, _, meta2 = ev.load_uni_checkpoint(ckpt_arch, device=device)
            self.assertIsInstance(model2, uni_omr_v2.UniOMRv2Net)
            self.assertEqual(meta2["arch"], "uni_omr_v2")

        # Synthetic image decode across all modes: greedy, grammar, MTP-4 spec, beam, logprob
        img = torch.rand(1, 3, 64, 96)
        seq_greedy = model.generate(img, max_len=12)
        self.assertEqual(seq_greedy.shape[0], 1)
        self.assertGreaterEqual(seq_greedy.shape[1], 2)

        seq_gram = ev.generate_constrained(model, img, max_len=12, strict=False)
        seq_mtp_gram, stats = ev.generate_mtp_spec(
            model, img, max_len=12, use_grammar=True, strict_grammar=False, return_stats=True
        )
        # Exact token-for-token parity between MTP-4 speculative + grammar and constrained greedy!
        self.assertTrue(torch.equal(seq_gram, seq_mtp_gram))
        self.assertEqual(stats["mtp_k"], 4)
        self.assertGreaterEqual(stats["verify_passes"], 1)

        # Unconstrained MTP-4 spec matches model.generate()
        seq_mtp_raw = ev.generate_mtp_spec(
            model, img, max_len=12, use_grammar=False, return_stats=False
        )
        self.assertTrue(torch.equal(seq_greedy, seq_mtp_raw))

        # Beam search on DeepStack 3-tuple memory
        seq_beam = ev.generate_beam(model, img, max_len=10, beam_size=2, use_grammar=True)
        self.assertEqual(seq_beam.shape[0], 1)
        self.assertGreaterEqual(seq_beam.shape[1], 2)

        # Teacher-forced mean logprob works cleanly on v2
        pad_id = ev._vocab_spec(int(loaded_cfg.vocab_size))[0]
        mean_lp = ev.compute_sequence_mean_logprob(model, img, seq_mtp_gram, pad_id=pad_id)
        self.assertIsInstance(mean_lp, float)
        self.assertLessEqual(mean_lp, 0.0)

        # Router max_keep_ratio override works on v2 (model.encoder.router)
        orig_mkr = float(model.encoder.router.max_keep_ratio)
        with ev.override_max_keep_ratio(model, 0.55):
            self.assertAlmostEqual(float(model.encoder.router.max_keep_ratio), 0.55, places=6)
        self.assertAlmostEqual(float(model.encoder.router.max_keep_ratio), orig_mkr, places=6)

    def test_load_v1_checkpoint_backward_compat(self) -> None:
        torch.manual_seed(11)
        cfg = _make_tiny_v1_cfg()
        orig = uni_omr.UniOMRNet(cfg).eval()

        with tempfile.TemporaryDirectory() as td:
            ckpt_v1 = Path(td) / "uni_omr_v1_best.pt"
            torch.save(
                {
                    "step": 100,
                    "val_ser": 0.25,
                    "config": asdict(cfg),
                    "model": orig.state_dict(),
                },
                ckpt_v1,
            )

            device = torch.device("cpu")
            model, loaded_cfg, meta = ev.load_uni_checkpoint(ckpt_v1, device=device)
            self.assertIsInstance(model, uni_omr.UniOMRNet)
            self.assertIsInstance(loaded_cfg, uni_omr.UniOMRConfig)
            self.assertEqual(meta["arch"], "uni_omr_v1")
            self.assertEqual(meta["step"], 100)
            self.assertEqual(meta["missing_keys"], [])
            self.assertEqual(meta["unexpected_keys"], [])

        img = torch.rand(1, 3, 64, 96)
        seq_gram = ev.generate_constrained(model, img, max_len=12, strict=False)
        seq_mtp_gram, stats = ev.generate_mtp_spec(
            model, img, max_len=12, use_grammar=True, strict_grammar=False, return_stats=True
        )
        self.assertTrue(torch.equal(seq_gram, seq_mtp_gram))
        self.assertEqual(stats["mtp_k"], 4)

        pad_id = ev._vocab_spec(int(loaded_cfg.vocab_size))[0]
        mean_lp = ev.compute_sequence_mean_logprob(model, img, seq_mtp_gram, pad_id=pad_id)
        self.assertIsInstance(mean_lp, float)
        self.assertLessEqual(mean_lp, 0.0)

        orig_mkr = float(model.router.max_keep_ratio)
        with ev.override_max_keep_ratio(model, 0.60):
            self.assertAlmostEqual(float(model.router.max_keep_ratio), 0.60, places=6)
        self.assertAlmostEqual(float(model.router.max_keep_ratio), orig_mkr, places=6)

    def test_parse_tta_scales_and_cli_default(self) -> None:
        self.assertEqual(ev.parse_tta_scales(None), [])
        self.assertEqual(ev.parse_tta_scales(""), [])
        self.assertEqual(ev.parse_tta_scales("1152, 1280, 1536, 1280"), [1152, 1280, 1536])
        with self.assertRaises(ValueError):
            ev.parse_tta_scales("32,1280")

        ap = ev.build_arg_parser()
        args_default = ap.parse_args(["--ckpt", "dummy.pt"])
        self.assertIsNone(args_default.tta_scales)
        args_tta = ap.parse_args(["--ckpt", "dummy.pt", "--tta-scales", "1152,1280,1536"])
        self.assertEqual(args_tta.tta_scales, "1152,1280,1536")

    def test_detect_token_loop_and_fsa_beat_metrics(self) -> None:
        good_bar = [
            "clef-G-2",
            "time-4-4",
            "quarter", "C", "4",
            "quarter", "D", "4",
            "quarter", "E", "4",
            "quarter", "F", "4",
            "bar",
        ]
        bad_bar = [
            "clef-G-2",
            "time-4-4",
            "quarter", "C", "4",
            "bar",  # only 1 beat in 4/4 -> meter_err = 3.0
        ]
        loop_toks = ["clef-G-2", "time-4-4"] + ["quarter", "C", "4", "bar"] * 5

        self.assertFalse(ev.detect_token_loop(good_bar))
        self.assertTrue(ev.detect_token_loop(loop_toks))
        self.assertTrue(ev.detect_token_loop(["quarter"] * 8))

        m_good = ev.compute_fsa_beat_metrics(good_bar, hit_eos=True, bekern_decode_ok=True)
        m_bad = ev.compute_fsa_beat_metrics(bad_bar, hit_eos=True, bekern_decode_ok=True)
        m_no_eos = ev.compute_fsa_beat_metrics(good_bar, hit_eos=False, bekern_decode_ok=True)
        m_loop = ev.compute_fsa_beat_metrics(loop_toks, hit_eos=False, bekern_decode_ok=True)

        self.assertAlmostEqual(m_good["beat_conservation_rate"], 1.0, places=6)
        self.assertAlmostEqual(m_good["mean_meter_err"], 0.0, places=6)
        self.assertFalse(m_good["has_loop"])
        self.assertAlmostEqual(m_bad["beat_conservation_rate"], 0.0, places=6)
        self.assertTrue(m_loop["has_loop"])
        self.assertGreater(m_good["fsa_beat_score"], m_no_eos["fsa_beat_score"])
        self.assertGreater(m_no_eos["fsa_beat_score"], m_bad["fsa_beat_score"])
        self.assertGreater(m_good["fsa_beat_score"], m_loop["fsa_beat_score"])

    def test_select_best_fsa_beat_view_priority(self) -> None:
        # 1. Valid parse beats invalid parse regardless of higher logprob/fsa score
        c_invalid = {"scale": 1152, "bekern_decode_ok": False, "has_loop": False, "fsa_beat_score": 2.5, "mean_logprob": -0.05}
        c_valid = {"scale": 1280, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 1.2, "mean_logprob": -0.40}
        self.assertEqual(ev.select_best_fsa_beat_view([c_invalid, c_valid])["scale"], 1280)

        # 2. Non-looping beats looping among valid parses
        c_loop = {"scale": 1152, "bekern_decode_ok": True, "has_loop": True, "fsa_beat_score": 2.5, "mean_logprob": -0.05}
        c_clean = {"scale": 1536, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 1.5, "mean_logprob": -0.35}
        self.assertEqual(ev.select_best_fsa_beat_view([c_loop, c_clean])["scale"], 1536)

        # 3. Higher fsa_beat_score wins among valid non-looping views
        c_low = {"scale": 1152, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 1.80, "mean_logprob": -0.10}
        c_high = {"scale": 1280, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 2.75, "mean_logprob": -0.25}
        self.assertEqual(ev.select_best_fsa_beat_view([c_low, c_high])["scale"], 1280)

        # 4. Exact fsa_beat_score tie broken by mean_logprob
        c_tie_lo = {"scale": 1152, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 2.75, "mean_logprob": -0.32}
        c_tie_hi = {"scale": 1280, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 2.75, "mean_logprob": -0.14}
        c_tie_mid = {"scale": 1536, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 2.75, "mean_logprob": -0.21}
        self.assertEqual(ev.select_best_fsa_beat_view([c_tie_lo, c_tie_hi, c_tie_mid])["scale"], 1280)

        # 5. Full exact tie broken deterministically by earlier scale order
        c_eq1 = {"scale": 1280, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 2.75, "mean_logprob": -0.15}
        c_eq2 = {"scale": 1536, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 2.75, "mean_logprob": -0.15}
        self.assertEqual(ev.select_best_fsa_beat_view([c_eq1, c_eq2])["scale"], 1280)


if __name__ == "__main__":
    unittest.main(verbosity=2)
