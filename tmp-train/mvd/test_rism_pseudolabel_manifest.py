#!/usr/bin/env python3
"""CPU-only unit tests for ``build_rism_pseudolabel_manifest.py`` (promoted ov-39 pipeline).

Uses stdlib ``unittest`` so it runs cleanly under both ``pytest`` and
``python -m unittest``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List
import unittest

from PIL import Image
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import build_rism_pseudolabel_manifest as brm  # noqa: E402
import uni_omr  # noqa: E402
import uni_omr_v2  # noqa: E402


def _make_dummy_images(root: Path, n: int) -> List[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i in range(n):
        p = root / f"rism_{i:04d}.png"
        Image.new("RGB", (64, 48), color=(240, 240, 240)).save(p)
        paths.append(p)
    return paths


def _make_args(
    *,
    input_dir: Path,
    out: Path,
    shard_idx: int = 0,
    num_shards: int = 1,
    resume: bool = False,
    limit: int | None = None,
) -> argparse.Namespace:
    return brm.build_arg_parser().parse_args([
        "--ckpt", "dummy.pt",
        "--input-dir", str(input_dir),
        "--out", str(out),
        "--device", "cpu",
        "--shard-idx", str(shard_idx),
        "--num-shards", str(num_shards),
        "--log-every", "100",
        *(["--resume"] if resume else []),
        *(["--limit", str(limit)] if limit is not None else []),
    ])


class TestRismPseudolabelManifest(unittest.TestCase):
    # -----------------------------------------------------------------------
    # 1. Filtering & Tier Tagging Tests
    # -----------------------------------------------------------------------

    def test_ov39_gate_filtering_and_tier_tagging(self) -> None:
        cfg = brm.Ov39GateConfig()

        # High-rhythm tier (>= 0.85)
        kept, tier, weight, reasons = brm.evaluate_ov39_gate(
            mean_logprob=-0.18,
            beat_valid_frac=0.90,
            r_parse=1.0,
            r_noloop=1.0,
            n_tokens=640,
            n_systems=4,
            cfg=cfg,
        )
        self.assertTrue(kept)
        self.assertEqual(tier, "high_rhythm")
        self.assertAlmostEqual(weight, 1.75)
        self.assertEqual(reasons, [])

        # Exact boundary for high_rhythm (0.85) and min_logprob (-0.35)
        kept, tier, weight, reasons = brm.evaluate_ov39_gate(
            mean_logprob=-0.35,
            beat_valid_frac=0.85,
            r_parse=0.50,
            r_noloop=1.0,
            n_tokens=24,
            n_systems=1,
            cfg=cfg,
        )
        self.assertTrue(kept)
        self.assertEqual(tier, "high_rhythm")
        self.assertAlmostEqual(weight, 1.75)

        # Standard tier (0.70 <= beat_valid_frac < 0.85)
        kept, tier, weight, reasons = brm.evaluate_ov39_gate(
            mean_logprob=-0.34,
            beat_valid_frac=0.70,
            r_parse=0.80,
            r_noloop=1.0,
            n_tokens=120,
            n_systems=2,
            cfg=cfg,
        )
        self.assertTrue(kept)
        self.assertEqual(tier, "standard")
        self.assertAlmostEqual(weight, 1.0)
        self.assertEqual(reasons, [])

        # Rejections for each individual gate criterion
        cases = [
            dict(mean_logprob=-0.36, beat_valid_frac=0.90, r_parse=1.0, r_noloop=1.0, n_tokens=100, n_systems=2, expect="low_logprob"),
            dict(mean_logprob=-0.10, beat_valid_frac=0.69, r_parse=1.0, r_noloop=1.0, n_tokens=100, n_systems=2, expect="low_beat_valid"),
            dict(mean_logprob=-0.10, beat_valid_frac=0.90, r_parse=0.49, r_noloop=1.0, n_tokens=100, n_systems=2, expect="low_r_parse"),
            dict(mean_logprob=-0.10, beat_valid_frac=0.90, r_parse=1.0, r_noloop=0.0, n_tokens=100, n_systems=2, expect="loop_detected"),
            dict(mean_logprob=-0.10, beat_valid_frac=0.90, r_parse=1.0, r_noloop=1.0, n_tokens=23, n_systems=2, expect="too_short"),
            dict(mean_logprob=-0.10, beat_valid_frac=0.90, r_parse=1.0, r_noloop=1.0, n_tokens=50, n_systems=0, expect="empty_systems"),
        ]
        for c in cases:
            expect = c.pop("expect")
            kept, tier, weight, reasons = brm.evaluate_ov39_gate(cfg=cfg, **c)
            self.assertFalse(kept)
            self.assertEqual(tier, "rejected")
            self.assertEqual(weight, 0.0)
            self.assertTrue(
                any(r.startswith(expect) for r in reasons),
                f"Expected {expect} in {reasons}",
            )

    # -----------------------------------------------------------------------
    # 2. Sharding Partitioning Tests
    # -----------------------------------------------------------------------

    def test_sharding_disjoint_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            _make_dummy_images(tmp_path / "rism", 11)
            discovered = brm.discover_images(tmp_path / "rism")
            self.assertEqual(len(discovered), 11)

            s0 = brm.shard_images(discovered, shard_idx=0, num_shards=3)
            s1 = brm.shard_images(discovered, shard_idx=1, num_shards=3)
            s2 = brm.shard_images(discovered, shard_idx=2, num_shards=3)

            self.assertEqual((len(s0), len(s1), len(s2)), (4, 4, 3))
            all_sharded = s0 + s1 + s2
            self.assertEqual(len(set(all_sharded)), len(discovered))
            self.assertEqual(set(all_sharded), set(discovered))

            with self.assertRaises(ValueError):
                brm.shard_images(discovered, shard_idx=3, num_shards=3)
            with self.assertRaises(ValueError):
                brm.shard_images(discovered, shard_idx=0, num_shards=0)

    # -----------------------------------------------------------------------
    # 3. End-to-End Pipeline, Resume Idempotency & Schema Compatibility
    # -----------------------------------------------------------------------

    def test_pipeline_resume_idempotency_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            img_dir = tmp_path / "rism_imgs"
            _make_dummy_images(img_dir, 6)
            out_jsonl = tmp_path / "out" / "rism_ov39.jsonl"

            score_table: Dict[str, Dict[str, Any]] = {
                "rism_0000.png": dict(mean_logprob=-0.12, beat_valid_frac=0.92, r_parse=1.0, r_noloop=1.0, n_tokens=80, has_eos=True),
                "rism_0001.png": dict(mean_logprob=-0.28, beat_valid_frac=0.76, r_parse=1.0, r_noloop=1.0, n_tokens=60, has_eos=True),
                "rism_0002.png": dict(mean_logprob=-0.15, beat_valid_frac=0.95, r_parse=1.0, r_noloop=0.0, n_tokens=90, has_eos=False),
                "rism_0003.png": dict(mean_logprob=-0.35, beat_valid_frac=0.85, r_parse=0.5, r_noloop=1.0, n_tokens=48, has_eos=True),
                "rism_0004.png": dict(mean_logprob=-0.42, beat_valid_frac=0.88, r_parse=1.0, r_noloop=1.0, n_tokens=70, has_eos=True),
                "rism_0005.png": dict(mean_logprob=-0.20, beat_valid_frac=0.71, r_parse=1.0, r_noloop=1.0, n_tokens=55, has_eos=False),
            }
            call_log: List[str] = []

            def fake_decode(p: Path) -> Dict[str, Any]:
                call_log.append(p.name)
                s = score_table[p.name]
                sys0 = ["clef-G-2", "key-0", "time-4-4", "quarter", "C", "4", "bar"]
                sys1 = ["clef-G-2", "half", "D", "4", "half", "E", "4", "bar"]
                return {
                    "raw_tokens": sys0 + [uni_omr.SYS_TOKEN] + sys1,
                    "systems": [sys0, sys1],
                    "n_tokens": s["n_tokens"],
                    "n_systems": 2,
                    "mean_logprob": s["mean_logprob"],
                    "beat_valid_frac": s["beat_valid_frac"],
                    "r_parse": s["r_parse"],
                    "r_noloop": s["r_noloop"],
                    "has_eos": s["has_eos"],
                    "mtp_acc": 0.82,
                    "elapsed_secs": 0.01,
                }

            # Pass 1: process first 3 pages only (--limit 3)
            args1 = _make_args(input_dir=img_dir, out=out_jsonl, limit=3, resume=False)
            sum1 = brm.run_pipeline(args1, decode_fn=fake_decode, ckpt_meta_override={"arch": "v2", "step": 24000})
            self.assertEqual(sum1["counts"]["processed_this_run"], 3)
            self.assertEqual(sum1["counts"]["kept_this_run"], 2)
            self.assertEqual(sum1["counts"]["rejected_this_run"], 1)
            self.assertEqual(call_log, ["rism_0000.png", "rism_0001.png", "rism_0002.png"])

            # Pass 2: --resume over all 6 pages (must skip 0, 1, 2 INCLUDING rejected 2!)
            args2 = _make_args(input_dir=img_dir, out=out_jsonl, resume=True)
            sum2 = brm.run_pipeline(args2, decode_fn=fake_decode, ckpt_meta_override={"arch": "v2", "step": 24000})
            self.assertEqual(sum2["counts"]["skipped_resume"], 3)
            self.assertEqual(sum2["counts"]["processed_this_run"], 3)
            self.assertEqual(sum2["counts"]["kept_this_run"], 2)
            self.assertEqual(sum2["cumulative"]["total_scored"], 6)
            self.assertEqual(sum2["cumulative"]["total_kept"], 4)
            self.assertEqual(sum2["cumulative"]["total_high_rhythm"], 2)
            self.assertEqual(sum2["cumulative"]["total_standard"], 2)
            self.assertEqual(
                call_log,
                [
                    "rism_0000.png", "rism_0001.png", "rism_0002.png",
                    "rism_0003.png", "rism_0004.png", "rism_0005.png",
                ],
            )

            # Pass 3: --resume again must be a complete no-op (0 calls, 0 duplicate rows)
            args3 = _make_args(input_dir=img_dir, out=out_jsonl, resume=True)
            sum3 = brm.run_pipeline(args3, decode_fn=fake_decode, ckpt_meta_override={"arch": "v2", "step": 24000})
            self.assertEqual(sum3["counts"]["processed_this_run"], 0)
            self.assertEqual(sum3["counts"]["skipped_resume"], 6)
            self.assertEqual(len(call_log), 6)

            # Validate JSONL schema compatibility with uni_fullpage_iter11_clean.jsonl / uni_omr.py / train_uni_v2.py
            rows = [json.loads(ln) for ln in out_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(rows), 4)
            tiers_by_name = {Path(r["image"]).name: r["tier"] for r in rows}
            self.assertEqual(
                tiers_by_name,
                {
                    "rism_0000.png": "high_rhythm",
                    "rism_0001.png": "standard",
                    "rism_0003.png": "high_rhythm",
                    "rism_0005.png": "standard",
                },
            )
            for r in rows:
                self.assertTrue(Path(r["image"]).is_file())
                self.assertEqual(r["domain"], "real_hw_pseudo")
                self.assertIsInstance(r["bekerns"], list)
                self.assertEqual(len(r["bekerns"]), 2)
                for bp in r["bekerns"]:
                    p = Path(bp)
                    self.assertTrue(p.is_file())
                    self.assertTrue(p.name.endswith("-bekern.txt"))
                    self.assertGreaterEqual(len(p.read_text(encoding="utf-8").strip().split()), 5)
                self.assertEqual(r["ckpt_arch"], "v2")
                self.assertEqual(r["ckpt_step"], 24000)
                self.assertEqual(r["max_dim"], 1280)
                self.assertEqual(r["gen_max_len"], 2048)

    def test_sharded_runs_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            img_dir = tmp_path / "rism_imgs"
            _make_dummy_images(img_dir, 6)

            def always_keep(p: Path) -> Dict[str, Any]:
                sys0 = ["clef-G-2", "key-0", "time-4-4", "quarter", "C", "4", "bar"]
                return {
                    "raw_tokens": sys0,
                    "systems": [sys0],
                    "n_tokens": 40,
                    "n_systems": 1,
                    "mean_logprob": -0.15,
                    "beat_valid_frac": 0.90 if int(p.stem.split("_")[1]) % 2 == 0 else 0.75,
                    "r_parse": 1.0,
                    "r_noloop": 1.0,
                    "has_eos": True,
                    "mtp_acc": 0.80,
                    "elapsed_secs": 0.01,
                }

            sh0 = tmp_path / "shard0.jsonl"
            sh1 = tmp_path / "shard1.jsonl"
            brm.run_pipeline(_make_args(input_dir=img_dir, out=sh0, shard_idx=0, num_shards=2), decode_fn=always_keep)
            brm.run_pipeline(_make_args(input_dir=img_dir, out=sh1, shard_idx=1, num_shards=2), decode_fn=always_keep)

            merged = tmp_path / "merged.jsonl"
            m_info = brm.merge_shard_manifests([sh0, sh1, sh0], merged, deduplicate=True)
            self.assertEqual(m_info["n_kept"], 6)
            self.assertEqual(m_info["n_high_rhythm"], 3)
            self.assertEqual(m_info["n_standard"], 3)
            self.assertEqual(m_info["n_duplicates_skipped"], 3)

    # -----------------------------------------------------------------------
    # 4. Checkpoint-Agnostic Detection & Loading (Tiny v1 + Tiny v2 on CPU)
    # -----------------------------------------------------------------------

    def test_checkpoint_agnostic_v1_and_v2_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)

            # Build and save a tiny v1 checkpoint
            v1_cfg = uni_omr.UniOMRConfig(
                d_model=64, n_heads=2, n_enc_layers=1, n_dec_layers=1,
                mtp_k=2, dynamic_routing=False,
            )
            v1_model = uni_omr.UniOMRNet(v1_cfg)
            v1_ckpt = tmp_path / "tiny_v1.pt"
            torch.save({"config": v1_cfg.__dict__, "step": 100, "model": v1_model.state_dict()}, v1_ckpt)

            m1, c1, meta1 = brm.load_any_uni_checkpoint(v1_ckpt, device=torch.device("cpu"))
            self.assertEqual(meta1["arch"], "v1")
            self.assertEqual(meta1["step"], 100)
            self.assertIsInstance(m1, uni_omr.UniOMRNet)
            self.assertEqual(c1.d_model, 64)

            # Build and save a tiny v2 checkpoint
            v2_cfg = uni_omr_v2.UniOMRv2Config(
                d_model=64, n_heads=2, n_kv_heads=1,
                n_axial_layers=1, n_global_layers=1, n_dec_layers=1,
                backbone_channels=(16, 32, 48, 64), backbone_blocks=(1, 1, 1, 1),
                mtp_k=2, dynamic_routing=False,
                use_staff_seg_aux=False, use_stroke_hog_head=False,
            )
            v2_model = uni_omr_v2.UniOMRv2Net(v2_cfg)
            v2_ckpt = tmp_path / "tiny_v2.pt"
            torch.save(
                {"cfg": v2_cfg.__dict__, "config": v2_cfg.__dict__, "step": 200, "model": v2_model.state_dict()},
                v2_ckpt,
            )

            m2, c2, meta2 = brm.load_any_uni_checkpoint(v2_ckpt, device=torch.device("cpu"))
            self.assertEqual(meta2["arch"], "v2")
            self.assertEqual(meta2["step"], 200)
            self.assertIsInstance(m2, uni_omr_v2.UniOMRv2Net)
            self.assertEqual(c2.d_model, 64)

            # Verify decode_and_score_page runs end-to-end on CPU for BOTH v1 and v2
            img_p = _make_dummy_images(tmp_path / "one", 1)[0]
            res1 = brm.decode_and_score_page(m1, c1, img_p, torch.device("cpu"), max_dim=64, gen_max_len=8, use_bf16=False)
            res2 = brm.decode_and_score_page(m2, c2, img_p, torch.device("cpu"), max_dim=64, gen_max_len=8, use_bf16=False)
            for res in (res1, res2):
                self.assertIsInstance(res["mean_logprob"], float)
                self.assertGreaterEqual(res["beat_valid_frac"], 0.0)
                self.assertLessEqual(res["beat_valid_frac"], 1.0)
                self.assertIn("r_parse", res)
                self.assertIn("r_noloop", res)

    def test_bf16_t4_cpu_fallback(self) -> None:
        use_bf16, reason = brm.resolve_bf16_support(True, torch.device("cpu"))
        self.assertFalse(use_bf16)
        self.assertIn("fp32", reason)
        use_bf16_off, _ = brm.resolve_bf16_support(False, torch.device("cpu"))
        self.assertFalse(use_bf16_off)


if __name__ == "__main__":
    unittest.main()
