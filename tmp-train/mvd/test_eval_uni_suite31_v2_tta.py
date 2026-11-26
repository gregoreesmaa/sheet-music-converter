"""CPU unit tests for UniOMR v2 checkpoint support and scale-only best_fsa_beat TTA in eval_uni_suite31.py."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import eval_uni_suite31 as ev  # noqa: E402
import grammar  # noqa: E402
import preprocess_dewarp  # noqa: E402
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
        self.assertEqual(args_default.tta_scales, "")
        self.assertEqual(ev.parse_tta_scales(args_default.tta_scales), [])
        self.assertEqual(args_default.tta_rerank, "fsa_logprob_mbr")

        args_tta = ap.parse_args([
            "--ckpt", "dummy.pt",
            "--tta-scales", "1024,1152,1280,1408,1536",
            "--tta-rerank", "fsa_logprob",
        ])
        self.assertEqual(args_tta.tta_scales, "1024,1152,1280,1408,1536")
        self.assertEqual(ev.parse_tta_scales(args_tta.tta_scales), [1024, 1152, 1280, 1408, 1536])
        self.assertEqual(args_tta.tta_rerank, "fsa_logprob")

        args_mbr = ap.parse_args([
            "--ckpt", "dummy.pt",
            "--tta-scales", "1280,1408,1536",
            "--tta-rerank", "mbr",
        ])
        self.assertEqual(args_mbr.tta_rerank, "mbr")

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

    def test_select_tta_view_fsa_logprob_mbr_and_tiebreaking(self) -> None:
        # 1. Realistic ov-36b scenario: 3 consensus views vs 1 hallucinated short valid bar
        toks_a = ["clef-G-2", "time-4-4", "quarter", "C", "4", "quarter", "D", "4", "quarter", "E", "4", "quarter", "F", "4", "bar"]
        toks_b = ["clef-G-2", "time-4-4", "quarter", "C", "4", "quarter", "D", "4", "quarter", "E", "4", "quarter", "G", "4", "bar"]
        toks_outlier = ["clef-G-2", "time-4-4", "whole", "A", "5", "bar"]

        cands = [
            {
                "scale": 1152,
                "hyp_page_toks": toks_outlier,
                "bekern_decode_ok": True,
                "has_loop": False,
                "fsa_beat_score": 2.75,   # high FSA on short hallucination
                "mean_logprob": -0.35,    # lower confidence
            },
            {
                "scale": 1280,
                "hyp_page_toks": toks_a,
                "bekern_decode_ok": True,
                "has_loop": False,
                "fsa_beat_score": 2.60,
                "mean_logprob": -0.10,    # high confidence + strong consensus
            },
            {
                "scale": 1408,
                "hyp_page_toks": toks_a,
                "bekern_decode_ok": True,
                "has_loop": False,
                "fsa_beat_score": 2.50,
                "mean_logprob": -0.14,
            },
            {
                "scale": 1536,
                "hyp_page_toks": toks_b,
                "bekern_decode_ok": True,
                "has_loop": False,
                "fsa_beat_score": 2.55,
                "mean_logprob": -0.18,
            },
        ]

        # Verify pairwise MBR downweights failed bekern parses (0.25 vs 1.0)
        mbr_votes = ev.compute_mbr_vote_scores(cands)
        self.assertGreater(mbr_votes[1], mbr_votes[0])
        self.assertAlmostEqual(mbr_votes[1], mbr_votes[2], places=6)

        # fsa_logprob_mbr rejects the 1152 hallucination and picks 1280
        win_flm, _, scores_flm = ev.select_tta_view(cands, rerank="fsa_logprob_mbr")
        self.assertEqual(win_flm["scale"], 1280)
        expected_1280 = 0.5 * 2.60 + 1.5 * (-0.10) + 1.0 * mbr_votes[1]
        self.assertAlmostEqual(scores_flm[1], expected_1280, places=6)

        # fsa_logprob = fsa_beat_score + 2.0 * mean_logprob also picks 1280 (2.40 vs 2.05)
        win_fl, _, scores_fl = ev.select_tta_view(cands, rerank="fsa_logprob")
        self.assertEqual(win_fl["scale"], 1280)
        self.assertAlmostEqual(scores_fl[1], 2.60 + 2.0 * (-0.10), places=6)

        # mbr picks consensus view (1280 beats 1408 on mean_logprob tiebreak when MBR tied)
        win_mbr, _, _ = ev.select_tta_view(cands, rerank="mbr")
        self.assertEqual(win_mbr["scale"], 1280)

        # 2. Failed bekern_decode_ok view gets weight 0.25 in MBR computation
        c_ok1 = {"scale": 1280, "hyp_page_toks": toks_a, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 2.0, "mean_logprob": -0.2}
        c_ok2 = {"scale": 1408, "hyp_page_toks": toks_a, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 2.0, "mean_logprob": -0.2}
        c_bad = {"scale": 1536, "hyp_page_toks": toks_outlier, "bekern_decode_ok": False, "has_loop": False, "fsa_beat_score": -1.0, "mean_logprob": -0.2}
        w_mbr = ev.compute_mbr_vote_scores([c_ok1, c_ok2, c_bad])
        sim_ao = ev.compute_token_lcs_f1(toks_a, toks_outlier)
        expected_ok1_mbr = (1.0 * 1.0 + 0.25 * sim_ao) / 1.25
        self.assertAlmostEqual(w_mbr[0], expected_ok1_mbr, places=6)

        # 3. Tie-breaking on exact equal fsa_logprob_mbr score:
        # 3a. bekern_decode_ok True beats False
        t_bad = {"scale": 1024, "bekern_decode_ok": False, "has_loop": False, "fsa_beat_score": 2.0, "mean_logprob": -0.2, "mbr_vote": 0.5}
        t_ok = {"scale": 1280, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 2.0, "mean_logprob": -0.2, "mbr_vote": 0.5}
        self.assertEqual(ev.select_tta_view([t_bad, t_ok], rerank="fsa_logprob_mbr")[0]["scale"], 1280)

        # 3b. non-looping beats looping on equal rerank score
        t_loop = {"scale": 1024, "bekern_decode_ok": True, "has_loop": True, "fsa_beat_score": 2.0, "mean_logprob": -0.2, "mbr_vote": 0.5}
        t_clean = {"scale": 1408, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 2.0, "mean_logprob": -0.2, "mbr_vote": 0.5}
        self.assertEqual(ev.select_tta_view([t_loop, t_clean], rerank="fsa_logprob_mbr")[0]["scale"], 1408)

        # 3c. higher mean_logprob breaks exact rerank score tie
        # 0.5*2.0 + 1.5*(-0.2) + 1.0*0.8 = 1.50 vs 0.5*2.3 + 1.5*(-0.1) + 1.0*0.5 = 1.50
        t_lp_lo = {"scale": 1280, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 2.0, "mean_logprob": -0.2, "mbr_vote": 0.8}
        t_lp_hi = {"scale": 1408, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 2.3, "mean_logprob": -0.1, "mbr_vote": 0.5}
        self.assertEqual(ev.select_tta_view([t_lp_lo, t_lp_hi], rerank="fsa_logprob_mbr")[0]["scale"], 1408)

        # 3d. full exact tie broken deterministically by earlier scale order
        t_e1 = {"scale": 1280, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 2.0, "mean_logprob": -0.2, "mbr_vote": 0.8}
        t_e2 = {"scale": 1536, "bekern_decode_ok": True, "has_loop": False, "fsa_beat_score": 2.0, "mean_logprob": -0.2, "mbr_vote": 0.8}
        self.assertEqual(ev.select_tta_view([t_e1, t_e2], rerank="fsa_logprob_mbr")[0]["scale"], 1280)

        # 4. Unknown rerank mode raises ValueError
        with self.assertRaises(ValueError):
            ev.select_tta_view([t_e1], rerank="unknown_mode")

    def test_oemer_dewarp_flat_autoskip_and_curved_straighten(self) -> None:
        # 1. Flat synthetic staff image -> must auto-skip and return bit-identical image
        w, h = 640, 800
        flat = np.full((h, w, 3), 255, dtype=np.uint8)
        for sys_y in (160, 360, 560):
            for k in range(5):
                y = sys_y + k * 10
                flat[y : y + 2, 40 : w - 40, :] = 20
        flat_pil = Image.fromarray(flat, mode="RGB")

        out_flat, meta_flat = preprocess_dewarp.oemer_staffline_vertical_dewarp(flat_pil)
        self.assertTrue(meta_flat["skipped"])
        self.assertLess(meta_flat["max_dy_px"], 1.5)
        self.assertIs(out_flat, flat_pil)
        self.assertTrue(np.array_equal(np.asarray(out_flat), flat))

        # 2. Curved synthetic staff image (parabolic bow of ~16 px) -> must dewarp and flatten
        curved = np.full((h, w, 3), 255, dtype=np.uint8)
        xs = np.arange(w, dtype=np.float32)
        bow = 16.0 * ((xs - (w / 2.0)) / (w / 2.0)) ** 2
        for sys_y in (160, 360, 560):
            for k in range(5):
                for x in range(40, w - 40):
                    y = int(round(sys_y + k * 10 + bow[x]))
                    if 0 <= y < h - 1:
                        curved[y : y + 2, x, :] = 20
        curved_pil = Image.fromarray(curved, mode="RGB")

        out_curved, meta_curved = preprocess_dewarp.oemer_staffline_vertical_dewarp(curved_pil)
        self.assertFalse(meta_curved["skipped"])
        self.assertGreater(meta_curved["max_dy_px"], 1.5)
        self.assertEqual(out_curved.size, curved_pil.size)

        # Measure vertical center-of-mass spread of the first staff across columns before vs after
        def _staff_spread(arr_rgb: np.ndarray) -> float:
            band = 255.0 - arr_rgb[130:240, 80 : w - 80, 0].astype(np.float32)
            ys = np.arange(band.shape[0], dtype=np.float32)[:, None]
            col_sum = band.sum(axis=0) + 1e-6
            com = (band * ys).sum(axis=0) / col_sum
            return float(com.max() - com.min())

        spread_before = _staff_spread(curved)
        spread_after = _staff_spread(np.asarray(out_curved))
        self.assertLess(spread_after, spread_before * 0.5)

    def test_estimate_unitsize_first_local_peak_no_lag4_pinning(self) -> None:
        # 1. Synthetic staves with exact 10 px inter-staffline spacing at 1408px
        w, h = 1100, 1408
        arr = np.full((h, w, 3), 255, dtype=np.uint8)
        for sys_y in (200, 450, 700, 950, 1200):
            for k in range(5):
                y = sys_y + k * 10
                arr[y : y + 2, 60 : w - 60, :] = 15
        pil = Image.fromarray(arr, mode="RGB")

        target_dim, meta = preprocess_dewarp.estimate_unitsize_max_dim(
            pil, base_max_dim=1408, min_dim=1152, max_dim=1600
        )
        self.assertAlmostEqual(meta["est_unit_px"], 10.0, delta=1.0)
        self.assertGreater(meta["est_unit_px"], 4.5)
        self.assertGreaterEqual(target_dim, 1152)
        self.assertLessEqual(target_dim, 1600)
        self.assertEqual(target_dim % 32, 0)

        # 2. Monotonically decaying noisy vertical gradient (no staff lines) -> must NOT pin to 4.0
        grad_y = np.linspace(240, 180, h, dtype=np.float32)[:, None]
        noisy = np.clip(np.broadcast_to(grad_y, (h, w)), 0, 255).astype(np.uint8)
        noisy_pil = Image.fromarray(np.stack([noisy, noisy, noisy], axis=-1), mode="RGB")
        dim_fallback, meta_fb = preprocess_dewarp.estimate_unitsize_max_dim(
            noisy_pil, base_max_dim=1408, min_dim=1152, max_dim=1600
        )
        self.assertGreater(meta_fb["est_unit_px"], 4.5)
        self.assertEqual(dim_fallback, 1408)

    def test_crop_dark_borders_and_border_blindness_fix(self) -> None:
        # 1. Clean page (< 1% dark border) -> auto-skip, return bit-identical object
        pw, ph = 1100, 1408
        clean = np.full((ph, pw, 3), 255, dtype=np.uint8)
        for sys_y in (200, 450, 700, 950, 1200):
            for k in range(5):
                y = sys_y + k * 10
                clean[y : y + 2, 40 : pw - 40, :] = 0
        clean_pil = Image.fromarray(clean, mode="RGB")

        out_clean, cmeta_clean = preprocess_dewarp.crop_dark_borders(clean_pil)
        self.assertFalse(cmeta_clean["cropped"])
        self.assertIs(out_clean, clean_pil)
        self.assertTrue(np.array_equal(np.asarray(out_clean), clean))

        # 2. Synthetic 1-bit photo with 100px solid-black padding on all 4 sides
        pad = 100
        padded = np.zeros((ph + 2 * pad, pw + 2 * pad, 3), dtype=np.uint8)
        padded[pad : pad + ph, pad : pad + pw, :] = clean
        padded_pil = Image.fromarray(padded, mode="RGB")

        cropped_pil, cmeta_pad = preprocess_dewarp.crop_dark_borders(padded_pil)
        self.assertTrue(cmeta_pad["cropped"])
        self.assertGreater(cmeta_pad["border_frac"], 0.15)
        # Safety margin (+6px outward) keeps full paper interior without clipping edge ink
        cw, ch = cropped_pil.size
        self.assertGreaterEqual(cw, pw)
        self.assertGreaterEqual(ch, ph)
        self.assertLessEqual(cw, pw + 24)
        self.assertLessEqual(ch, ph + 24)

        # 3. estimate_unitsize_max_dim on 100px black-padded photo -> ac_peak > 0.25 & accurate ~10px period
        target_dim, umeta = preprocess_dewarp.estimate_unitsize_max_dim(
            padded_pil, base_max_dim=1408, min_dim=1152, max_dim=1600
        )
        self.assertTrue(umeta["border_cropped"])
        self.assertGreater(umeta["ac_peak"], 0.25)
        self.assertAlmostEqual(umeta["est_unit_px"], 10.0, delta=1.0)
        self.assertEqual(target_dim % 32, 0)

        # 4. oemer_staffline_vertical_dewarp on 100px black-padded flat photo -> zero false tilt (auto-skipped)
        dew_out, dmeta = preprocess_dewarp.oemer_staffline_vertical_dewarp(padded_pil)
        self.assertTrue(dmeta["skipped"])
        self.assertLess(dmeta["max_dy_px"], 1.5)
        self.assertIs(dew_out, padded_pil)

    def test_measure_beam_header_guard_cap_and_cli_defaults(self) -> None:
        # 1. Trigger cap = 4 allows exactly 4 repairs then blocks further triggers
        guard = ev.MeasureBeamHeaderGuard(const_offset_limit=10, max_triggers=4)
        st = grammar.PrefixState(strict=False)
        st.step("clef-G-2")
        st.step("time-4-4")
        st.step("quarter")
        st.step("C")
        st.step("4")
        results = [guard.should_repair(st, meter_err=3.0, voice_err=0.0) for _ in range(6)]
        self.assertEqual(results, [True, True, True, True, False, False])
        self.assertEqual(guard.trigger_count, 4)

        # 2. Promoted ov-35b / ov-34b / i13-10 CLI defaults
        ap = ev.build_arg_parser()
        args = ap.parse_args(["--ckpt", "dummy.pt"])
        self.assertEqual(args.max_dim, 1408)
        self.assertEqual(args.max_len, 3584)
        self.assertEqual(args.measure_beam_size, 1)
        self.assertEqual(args.measure_beam_cap, 4)
        self.assertAlmostEqual(args.measure_beam_lp, 2.0, places=6)
        self.assertEqual(args.measure_beam_max_window, 64)
        self.assertFalse(args.oemer_dewarp)
        self.assertFalse(args.unitsize_norm)

        # 3. Explicit CLI flags parse cleanly
        args_on = ap.parse_args([
            "--ckpt", "dummy.pt",
            "--oemer-dewarp",
            "--unitsize-norm",
            "--measure-beam-size", "2",
            "--measure-beam-cap", "6",
            "--measure-beam-lp", "1.5",
            "--measure-beam-max-window", "96",
        ])
        self.assertTrue(args_on.oemer_dewarp)
        self.assertTrue(args_on.unitsize_norm)
        self.assertEqual(args_on.measure_beam_size, 2)
        self.assertEqual(args_on.measure_beam_cap, 6)
        self.assertAlmostEqual(args_on.measure_beam_lp, 1.5, places=6)
        self.assertEqual(args_on.measure_beam_max_window, 96)


if __name__ == "__main__":
    unittest.main(verbosity=2)
