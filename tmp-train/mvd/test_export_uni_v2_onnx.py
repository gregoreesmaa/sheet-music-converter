#!/usr/bin/env python3
"""CPU-only verification suite for UniOMR v2 multi-target ONNX exporter.

Tests all three targets (``webgpu``, ``npu``, ``cpu``) using a tiny
``UniOMRv2Net`` (``d_model=64, n_heads=4, n_kv_heads=2, n_dec_layers=2,
n_axial_layers=1, n_global_layers=1, mtp_k=2``):
  1. Exact FP32 numerical parity (``max_abs_diff < 1e-3``) against PyTorch
     for encoder ``cross_k/v``, decoder ``logits``, and all-T ``mtp_logits``
     across ``T=1`` and ``T=2/4`` steps.
  2. Partial-accept on-GPU KV rewind (``step_pos < past_k.shape[2]``) on
     ``webgpu`` & ``cpu``.
  3. Multi-step static ring-buffer KV cache on ``npu`` (both ``T=1`` and ``T=2``).
  4. Static graph & op-hygiene checks on ``npu`` (0 dynamic dims, 0
     ``MatMulNBits`` ops, 0 ``SpaceToDepth`` ops, 0 ``Loop`` ops).
  5. Quantized exports for all three targets (``webgpu`` INT4 ``MatMulNBits``,
     ``npu`` opset-17 INT8 ``DequantizeLinear``, ``cpu`` dynamic INT8) load
     and run cleanly in ONNX Runtime CPU.
  6. Complete ``manifest.json`` and ``vocab.json`` schema verification.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from typing import Dict, List, Tuple

# Strictly CPU-only! Never touch CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import onnx
import onnxruntime as ort
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import export_uni_v2_onnx as exp  # noqa: E402
import uni_omr_v2  # noqa: E402


def _make_ort_session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _check_no_dynamic_dims(model_path: Path) -> List[str]:
    """Return list of any dynamic dimensions found in graph inputs/outputs."""
    proto = onnx.load(str(model_path))
    dyn: List[str] = []
    for vi in list(proto.graph.input) + list(proto.graph.output):
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            dyn.append(f"{vi.name}:<no_shape>")
            continue
        for idx, dim in enumerate(tt.shape.dim):
            if dim.dim_param or dim.dim_value <= 0:
                dyn.append(f"{vi.name}[{idx}]={dim.dim_param or dim.dim_value}")
    return dyn


def _collect_op_types(model_path: Path) -> Dict[str, int]:
    proto = onnx.load(str(model_path))
    counts: Dict[str, int] = {}
    for node in proto.graph.node:
        key = f"{node.domain}::{node.op_type}" if node.domain else node.op_type
        counts[key] = counts.get(key, 0) + 1
    return counts


class TestExportUniV2ONNX(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(4)
        cls.model, cls.cfg = exp.build_tiny_test_model(seed=1234)
        cls.model.eval()
        # Deterministic synthetic score-like image (white paper with dark ink bands)
        rng = np.random.default_rng(7)
        img = np.full((1, 3, 96, 128), 0.92, dtype=np.float32)
        # Draw dark border & staff-like ink lines so paper_gate + router are exercised
        img[:, :, :4, :] = 0.05
        img[:, :, -4:, :] = 0.05
        for y in range(20, 80, 12):
            img[:, :, y : y + 2, 12:116] = 0.08
        img += rng.normal(0.0, 0.02, size=img.shape).astype(np.float32)
        cls.img_np = np.clip(img, 0.0, 1.0)
        cls.img_t = torch.from_numpy(cls.img_np)

        cls.tmp_fp32 = tempfile.TemporaryDirectory(prefix="uni_v2_fp32_")
        cls.out_fp32 = Path(cls.tmp_fp32.name)
        cls.manifest_fp32 = exp.export_all_targets(
            model=cls.model,
            cfg=cls.cfg,
            out_dir=cls.out_fp32,
            targets=("webgpu", "npu", "cpu"),
            opset=17,
            quantize=False,
            attn_chunk_q=8,
            npu_hw=(96, 128),
            npu_k_mem=20,
            npu_kv_max_len=16,
            npu_dec_t=2,
            dummy_hw=(64, 64),
        )

        cls.tmp_quant = tempfile.TemporaryDirectory(prefix="uni_v2_quant_")
        cls.out_quant = Path(cls.tmp_quant.name)
        cls.manifest_quant = exp.export_all_targets(
            model=cls.model,
            cfg=cls.cfg,
            out_dir=cls.out_quant,
            targets=("webgpu", "npu", "cpu"),
            opset=17,
            quantize=True,
            attn_chunk_q=8,
            block_size=32,
            npu_hw=(96, 128),
            npu_k_mem=20,
            npu_kv_max_len=16,
            npu_dec_t=1,
            dummy_hw=(64, 64),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp_fp32.cleanup()
        cls.tmp_quant.cleanup()

    def test_01_manifest_and_vocab_schema(self) -> None:
        required_target_keys = {
            "target",
            "encoder_file",
            "decoder_file",
            "encoder_bytes",
            "decoder_bytes",
            "total_bytes",
            "d_model",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "n_dec_layers",
            "mtp_k",
            "static_hw",
            "static_k_mem",
            "static_dec_t",
            "kv_max_len",
            "kv_mode",
            "quant_format",
        }
        for manifest, root in ((self.manifest_fp32, self.out_fp32), (self.manifest_quant, self.out_quant)):
            self.assertEqual(manifest["arch"], "uni_omr_v2")
            self.assertEqual(manifest["opset"], 17)
            self.assertTrue((root / "manifest.json").exists())
            self.assertTrue((root / "vocab.json").exists())
            vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
            self.assertEqual(vocab["n_heads"], 4)
            self.assertEqual(vocab["n_kv_heads"], 2)
            self.assertEqual(vocab["n_dec_layers"], 2)
            self.assertEqual(vocab["d_model"], 64)
            self.assertEqual(vocab["mtp_k"], 2)

            for tname in ("webgpu", "npu", "cpu"):
                self.assertIn(tname, manifest["targets"])
                entry = manifest["targets"][tname]
                self.assertTrue(required_target_keys.issubset(entry.keys()))
                self.assertTrue((root / entry["encoder_file"]).exists())
                self.assertTrue((root / entry["decoder_file"]).exists())
                self.assertGreater(entry["encoder_bytes"], 0)
                self.assertGreater(entry["decoder_bytes"], 0)
                self.assertEqual(entry["n_heads"], 4)
                self.assertEqual(entry["n_kv_heads"], 2)
                self.assertEqual(entry["n_dec_layers"], 2)
                self.assertEqual(entry["d_model"], 64)
                self.assertEqual(entry["mtp_k"], 2)

            self.assertEqual(manifest["targets"]["npu"]["static_hw"], [96, 128])
            self.assertEqual(manifest["targets"]["npu"]["kv_max_len"], 16)
            self.assertEqual(manifest["targets"]["npu"]["kv_mode"], "static_ring_buffer")
            self.assertEqual(manifest["targets"]["webgpu"]["kv_mode"], "dynamic_slice")
            self.assertEqual(manifest["targets"]["cpu"]["kv_mode"], "dynamic_slice")

        self.assertIn("int4_matmul_nbits", self.manifest_quant["targets"]["webgpu"]["quant_format"])
        self.assertEqual(self.manifest_quant["targets"]["npu"]["quant_format"], "int8_dequantize_linear")
        self.assertEqual(self.manifest_quant["targets"]["cpu"]["quant_format"], "int8_dynamic_quint8")

    def test_02_npu_static_shapes_and_op_hygiene(self) -> None:
        for manifest, root in ((self.manifest_fp32, self.out_fp32), (self.manifest_quant, self.out_quant)):
            npu = manifest["targets"]["npu"]
            enc_path = root / npu["encoder_file"]
            dec_path = root / npu["decoder_file"]

            enc_dyn = _check_no_dynamic_dims(enc_path)
            dec_dyn = _check_no_dynamic_dims(dec_path)
            self.assertEqual(enc_dyn, [], f"NPU encoder has dynamic dims: {enc_dyn}")
            self.assertEqual(dec_dyn, [], f"NPU decoder has dynamic dims: {dec_dyn}")

            enc_ops = _collect_op_types(enc_path)
            dec_ops = _collect_op_types(dec_path)
            for name, ops in (("npu_enc", enc_ops), ("npu_dec", dec_ops)):
                self.assertNotIn("com.microsoft::MatMulNBits", ops, f"{name} must not have MatMulNBits")
                self.assertNotIn("MatMulNBits", ops, f"{name} must not have MatMulNBits")
                self.assertNotIn("SpaceToDepth", ops, f"{name} must not have SpaceToDepth")
                self.assertNotIn("Loop", ops, f"{name} must not have Loop")

        # Verify quantized NPU uses standard opset-17 DequantizeLinear
        npu_q_enc_ops = _collect_op_types(self.out_quant / self.manifest_quant["targets"]["npu"]["encoder_file"])
        npu_q_dec_ops = _collect_op_types(self.out_quant / self.manifest_quant["targets"]["npu"]["decoder_file"])
        self.assertGreater(npu_q_enc_ops.get("DequantizeLinear", 0), 0)
        self.assertGreater(npu_q_dec_ops.get("DequantizeLinear", 0), 0)

        # Verify webgpu has no SpaceToDepth and no Loop; cpu has Loop (tiled SDPA) and no SpaceToDepth
        wg_enc_ops = _collect_op_types(self.out_fp32 / self.manifest_fp32["targets"]["webgpu"]["encoder_file"])
        cpu_enc_ops = _collect_op_types(self.out_fp32 / self.manifest_fp32["targets"]["cpu"]["encoder_file"])
        self.assertNotIn("SpaceToDepth", wg_enc_ops)
        self.assertNotIn("Loop", wg_enc_ops)
        self.assertNotIn("SpaceToDepth", cpu_enc_ops)
        self.assertGreater(cpu_enc_ops.get("Loop", 0), 0)

    def test_03_webgpu_and_cpu_fp32_parity_and_kv_rewind(self) -> None:
        n_dec = self.cfg.n_dec_layers
        n_kv = self.cfg.n_kv_heads
        hd = self.cfg.d_model // self.cfg.n_heads
        n_mtp = len(self.model.mtp_heads)

        # PyTorch reference on (96, 128) (different from export dummy 64x64!)
        with torch.no_grad():
            pt_mem_levels, _, pt_rinfo = self.model.encode_image(self.img_t, return_aux=False)
            pt_cross_kvs = self.model.precompute_cross_kv(pt_mem_levels)
            pt_cross_ks = [kv[1][0].numpy() for kv in pt_cross_kvs]
            pt_cross_vs = [kv[1][1].numpy() for kv in pt_cross_kvs]

            # Step 0: T=1 [BOS]
            tok0 = torch.tensor([[uni_omr_v2.BOS_ID]], dtype=torch.long)
            pt_logits_0, pt_mtp_0, pt_past_1 = self.model.decode_tokens(
                tok0, memory=None, return_mtp=True, return_aux=False, past_kvs=pt_cross_kvs, use_cache=True
            )
            # Step 1: T=4 [10, 20, 30, 40] from past_len=1
            tok1_4 = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
            pt_logits_1_4, pt_mtp_1_4, pt_past_5 = self.model.decode_tokens(
                tok1_4, memory=None, return_mtp=True, return_aux=False, past_kvs=pt_past_1, use_cache=True
            )
            # Partial-accept KV rewind: keep only first 2 tokens from pt_past_5, then run T=2 [55, 66]
            pt_past_2 = [
                ((lk[0][0][:, :, :2, :], lk[0][1][:, :, :2, :]), lk[1])
                for lk in pt_past_5
            ]
            tok2_2 = torch.tensor([[55, 66]], dtype=torch.long)
            pt_logits_rewind, pt_mtp_rewind, _ = self.model.decode_tokens(
                tok2_2, memory=None, return_mtp=True, return_aux=False, past_kvs=pt_past_2, use_cache=True
            )

        for target in ("webgpu", "cpu"):
            entry = self.manifest_fp32["targets"][target]
            enc_sess = _make_ort_session(self.out_fp32 / entry["encoder_file"])
            dec_sess = _make_ort_session(self.out_fp32 / entry["decoder_file"])

            enc_outs = enc_sess.run(None, {"image": self.img_np})
            ort_cks = enc_outs[:n_dec]
            ort_cvs = enc_outs[n_dec : 2 * n_dec]
            max_enc_diff = 0.0
            for i in range(n_dec):
                self.assertEqual(ort_cks[i].shape, pt_cross_ks[i].shape)
                max_enc_diff = max(
                    max_enc_diff,
                    float(np.max(np.abs(ort_cks[i] - pt_cross_ks[i]))),
                    float(np.max(np.abs(ort_cvs[i] - pt_cross_vs[i]))),
                )
            self.assertLess(max_enc_diff, 1e-3, f"{target} encoder cross_k/v diff={max_enc_diff}")

            def _run_dec(toks: np.ndarray, pos: int, pks: List[np.ndarray], pvs: List[np.ndarray]):
                feed: Dict[str, np.ndarray] = {
                    "token_ids": toks.astype(np.int64),
                    "step_pos": np.array([pos], dtype=np.int64),
                }
                for i in range(n_dec):
                    feed[f"past_k_{i}"] = pks[i]
                    feed[f"past_v_{i}"] = pvs[i]
                    feed[f"cross_k_{i}"] = ort_cks[i]
                    feed[f"cross_v_{i}"] = ort_cvs[i]
                outs = dec_sess.run(None, feed)
                logits = outs[0]
                mtps = outs[1 : 1 + n_mtp]
                nk = outs[1 + n_mtp : 1 + n_mtp + n_dec]
                nv = outs[1 + n_mtp + n_dec : 1 + n_mtp + 2 * n_dec]
                return logits, mtps, nk, nv

            empty_k = [np.zeros((1, n_kv, 0, hd), dtype=np.float32) for _ in range(n_dec)]
            empty_v = [np.zeros((1, n_kv, 0, hd), dtype=np.float32) for _ in range(n_dec)]

            # T=1
            l0, m0, pk1, pv1 = _run_dec(tok0.numpy(), 0, empty_k, empty_v)
            d_l0 = float(np.max(np.abs(l0 - pt_logits_0.numpy())))
            d_m0 = max(float(np.max(np.abs(a - b.numpy()))) for a, b in zip(m0, pt_mtp_0))
            self.assertLess(d_l0, 1e-3, f"{target} T=1 logits diff={d_l0}")
            self.assertLess(d_m0, 1e-3, f"{target} T=1 mtp diff={d_m0}")

            # T=4 (all-T MTP output shape == (1, 4, V)!)
            l14, m14, pk5, pv5 = _run_dec(tok1_4.numpy(), 1, pk1, pv1)
            self.assertEqual(l14.shape, (1, 4, self.cfg.vocab_size))
            self.assertEqual(m14[0].shape, (1, 4, self.cfg.vocab_size))
            self.assertEqual(pk5[0].shape, (1, n_kv, 5, hd))
            d_l14 = float(np.max(np.abs(l14 - pt_logits_1_4.numpy())))
            d_m14 = max(float(np.max(np.abs(a - b.numpy()))) for a, b in zip(m14, pt_mtp_1_4))
            self.assertLess(d_l14, 1e-3, f"{target} T=4 logits diff={d_l14}")
            self.assertLess(d_m14, 1e-3, f"{target} T=4 mtp diff={d_m14}")

            # Zero-copy partial-accept KV rewind: pass pk5 (len=5) with step_pos=2!
            l_rw, m_rw, pk_rw, pv_rw = _run_dec(tok2_2.numpy(), 2, pk5, pv5)
            self.assertEqual(pk_rw[0].shape, (1, n_kv, 4, hd))
            d_rw = float(np.max(np.abs(l_rw - pt_logits_rewind.numpy())))
            d_mrw = max(float(np.max(np.abs(a - b.numpy()))) for a, b in zip(m_rw, pt_mtp_rewind))
            self.assertLess(d_rw, 1e-3, f"{target} KV-rewind logits diff={d_rw}")
            self.assertLess(d_mrw, 1e-3, f"{target} KV-rewind mtp diff={d_mrw}")

            print(
                f"[parity:{target}] enc={max_enc_diff:.2e} T1={d_l0:.2e} "
                f"T4={d_l14:.2e} rewind={d_rw:.2e} mtp_max={max(d_m0, d_m14, d_mrw):.2e}",
                flush=True,
            )

    def test_04_npu_static_ring_buffer_parity(self) -> None:
        n_dec = self.cfg.n_dec_layers
        n_kv = self.cfg.n_kv_heads
        hd = self.cfg.d_model // self.cfg.n_heads
        n_mtp = len(self.model.mtp_heads)
        entry = self.manifest_fp32["targets"]["npu"]
        kv_max_len = entry["kv_max_len"]
        static_k_mem = entry["static_k_mem"]
        n_grid = (96 // 16) * (128 // 16)  # 48
        k_kept = static_k_mem - self.cfg.n_bg_summary
        keep_ratio_override = k_kept / float(n_grid)

        enc_sess = _make_ort_session(self.out_fp32 / entry["encoder_file"])
        dec_sess = _make_ort_session(self.out_fp32 / entry["decoder_file"])

        # PyTorch reference with matching k_kept
        with torch.no_grad():
            pt_mem_levels, _, pt_rinfo = self.model.encode_image(
                self.img_t, keep_ratio_override=keep_ratio_override, return_aux=False
            )
            self.assertEqual(pt_rinfo["n_out"], static_k_mem)
            pt_cross_kvs = self.model.precompute_cross_kv(pt_mem_levels)
            pt_cross_ks = [kv[1][0].numpy() for kv in pt_cross_kvs]
            pt_cross_vs = [kv[1][1].numpy() for kv in pt_cross_kvs]

            # Multi-step T=2 decode: step_pos=0 [BOS, 10], then step_pos=2 [20, 30], then rewind to step_pos=3 [40, 50]
            t_a = torch.tensor([[uni_omr_v2.BOS_ID, 10]], dtype=torch.long)
            t_b = torch.tensor([[20, 30]], dtype=torch.long)
            t_c = torch.tensor([[40, 50]], dtype=torch.long)
            pt_la, pt_ma, pt_p2 = self.model.decode_tokens(
                t_a, memory=None, return_mtp=True, return_aux=False, past_kvs=pt_cross_kvs, use_cache=True
            )
            pt_lb, pt_mb, pt_p4 = self.model.decode_tokens(
                t_b, memory=None, return_mtp=True, return_aux=False, past_kvs=pt_p2, use_cache=True
            )
            # Rewind ring buffer to step_pos=3 (overwriting slot 3 and writing slot 4)
            pt_p3 = [((lk[0][0][:, :, :3, :], lk[0][1][:, :, :3, :]), lk[1]) for lk in pt_p4]
            pt_lc, pt_mc, _ = self.model.decode_tokens(
                t_c, memory=None, return_mtp=True, return_aux=False, past_kvs=pt_p3, use_cache=True
            )

        enc_outs = enc_sess.run(None, {"image": self.img_np})
        ort_cks = enc_outs[:n_dec]
        ort_cvs = enc_outs[n_dec : 2 * n_dec]
        max_enc_diff = 0.0
        for i in range(n_dec):
            self.assertEqual(ort_cks[i].shape, (1, n_kv, static_k_mem, hd))
            max_enc_diff = max(
                max_enc_diff,
                float(np.max(np.abs(ort_cks[i] - pt_cross_ks[i]))),
                float(np.max(np.abs(ort_cvs[i] - pt_cross_vs[i]))),
            )
        self.assertLess(max_enc_diff, 1e-3, f"NPU encoder diff={max_enc_diff}")

        # Initialize static ring buffer with non-zero sentinel noise to prove masked slots never leak!
        rng = np.random.default_rng(99)
        ring_k = [rng.normal(0.0, 5.0, size=(1, n_kv, kv_max_len, hd)).astype(np.float32) for _ in range(n_dec)]
        ring_v = [rng.normal(0.0, 5.0, size=(1, n_kv, kv_max_len, hd)).astype(np.float32) for _ in range(n_dec)]

        def _run_npu_dec(toks: np.ndarray, pos: int, rk: List[np.ndarray], rv: List[np.ndarray]):
            feed: Dict[str, np.ndarray] = {
                "token_ids": toks.astype(np.int64),
                "step_pos": np.array([pos], dtype=np.int64),
            }
            for i in range(n_dec):
                feed[f"past_k_{i}"] = rk[i]
                feed[f"past_v_{i}"] = rv[i]
                feed[f"cross_k_{i}"] = ort_cks[i]
                feed[f"cross_v_{i}"] = ort_cvs[i]
            outs = dec_sess.run(None, feed)
            logits = outs[0]
            mtps = outs[1 : 1 + n_mtp]
            nk = outs[1 + n_mtp : 1 + n_mtp + n_dec]
            nv = outs[1 + n_mtp + n_dec : 1 + n_mtp + 2 * n_dec]
            return logits, mtps, nk, nv

        la, ma, ring_k, ring_v = _run_npu_dec(t_a.numpy(), 0, ring_k, ring_v)
        lb, mb, ring_k, ring_v = _run_npu_dec(t_b.numpy(), 2, ring_k, ring_v)
        lc, mc, ring_k, ring_v = _run_npu_dec(t_c.numpy(), 3, ring_k, ring_v)

        for i in range(n_dec):
            self.assertEqual(ring_k[i].shape, (1, n_kv, kv_max_len, hd))
            self.assertEqual(ring_v[i].shape, (1, n_kv, kv_max_len, hd))

        da = float(np.max(np.abs(la - pt_la.numpy())))
        db = float(np.max(np.abs(lb - pt_lb.numpy())))
        dc = float(np.max(np.abs(lc - pt_lc.numpy())))
        dm = max(
            float(np.max(np.abs(ma[0] - pt_ma[0].numpy()))),
            float(np.max(np.abs(mb[0] - pt_mb[0].numpy()))),
            float(np.max(np.abs(mc[0] - pt_mc[0].numpy()))),
        )
        self.assertLess(da, 1e-3, f"NPU step 0 diff={da}")
        self.assertLess(db, 1e-3, f"NPU step 2 diff={db}")
        self.assertLess(dc, 1e-3, f"NPU step 3 (overwrite) diff={dc}")
        self.assertLess(dm, 1e-3, f"NPU MTP diff={dm}")
        print(
            f"[parity:npu] enc={max_enc_diff:.2e} step0={da:.2e} step2={db:.2e} "
            f"step3_overwrite={dc:.2e} mtp_max={dm:.2e}",
            flush=True,
        )

    def test_05_quantized_all_targets_run_cleanly(self) -> None:
        n_dec = self.cfg.n_dec_layers
        n_kv = self.cfg.n_kv_heads
        hd = self.cfg.d_model // self.cfg.n_heads

        for target in ("webgpu", "npu", "cpu"):
            entry = self.manifest_quant["targets"][target]
            enc_sess = _make_ort_session(self.out_quant / entry["encoder_file"])
            dec_sess = _make_ort_session(self.out_quant / entry["decoder_file"])

            enc_outs = enc_sess.run(None, {"image": self.img_np})
            cks = enc_outs[:n_dec]
            cvs = enc_outs[n_dec : 2 * n_dec]
            for t in cks + cvs:
                self.assertTrue(np.all(np.isfinite(t)), f"{target} quantized encoder non-finite output")

            if target == "npu":
                kv_len = entry["kv_max_len"]
                pks = [np.zeros((1, n_kv, kv_len, hd), dtype=np.float32) for _ in range(n_dec)]
                pvs = [np.zeros((1, n_kv, kv_len, hd), dtype=np.float32) for _ in range(n_dec)]
                toks = np.array([[uni_omr_v2.BOS_ID]], dtype=np.int64)
            else:
                pks = [np.zeros((1, n_kv, 0, hd), dtype=np.float32) for _ in range(n_dec)]
                pvs = [np.zeros((1, n_kv, 0, hd), dtype=np.float32) for _ in range(n_dec)]
                toks = np.array([[uni_omr_v2.BOS_ID, 10]], dtype=np.int64)

            feed: Dict[str, np.ndarray] = {
                "token_ids": toks,
                "step_pos": np.array([0], dtype=np.int64),
            }
            for i in range(n_dec):
                feed[f"past_k_{i}"] = pks[i]
                feed[f"past_v_{i}"] = pvs[i]
                feed[f"cross_k_{i}"] = cks[i]
                feed[f"cross_v_{i}"] = cvs[i]

            dec_outs = dec_sess.run(None, feed)
            self.assertTrue(np.all(np.isfinite(dec_outs[0])), f"{target} quantized decoder non-finite logits")
            print(
                f"[quant:{target}] format={entry['quant_format']} "
                f"enc={entry['encoder_bytes'] / 1024:.1f}KB dec={entry['decoder_bytes'] / 1024:.1f}KB OK",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
