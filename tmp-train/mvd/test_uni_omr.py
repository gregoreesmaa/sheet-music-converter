"""Unit tests for UniOMR (tmp-train/mvd/uni_omr.py).

All tests run on CPU so they never interfere with active GPU training jobs.
"""

from __future__ import annotations

import io
import contextlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402
import uni_omr  # noqa: E402


def _small_cfg(**overrides) -> uni_omr.UniOMRConfig:
    """Fast CPU-friendly config for unit tests (preserves all architectural paths)."""
    base = dict(
        d_model=64,
        n_enc_layers=2,
        n_dec_layers=2,
        n_heads=4,
        ffn_mult=2.0,
        dropout=0.0,
        keep_ratio=0.40,
        n_bg_summary=4,
        use_ink_prior=True,
        use_uv_aux=True,
        mtp_k=4,
    )
    base.update(overrides)
    return uni_omr.UniOMRConfig(**base)


class TestUniOMR(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_vocab_preserves_data_vocab_and_adds_sys(self) -> None:
        self.assertEqual(uni_omr.PAD_ID, data.PAD_ID)
        self.assertEqual(uni_omr.BOS_ID, data.BOS_ID)
        self.assertEqual(uni_omr.EOS_ID, data.EOS_ID)
        self.assertEqual(uni_omr.SYS_ID, len(data.VOCAB) + 3)
        self.assertEqual(uni_omr.VOCAB_SIZE, len(data.VOCAB) + 4)

        sample = ["clef-G-2", "quarter", "C", "4", "bar", "<sys>", "clef-F-4", "bar-rest"]
        ids = uni_omr.encode_uni(sample, add_bos_eos=True)
        self.assertEqual(ids[0], uni_omr.BOS_ID)
        self.assertEqual(ids[-1], uni_omr.EOS_ID)
        self.assertIn(uni_omr.SYS_ID, ids)
        self.assertEqual(uni_omr.decode_uni(ids), sample)

    def test_a_dynamic_resolution_strip_and_fullpage_same_weights(self) -> None:
        """(a) Native dynamic resolution forward pass on strip (2,3,256,512) and fullpage (2,3,768,576)."""
        model = uni_omr.UniOMRNet(_small_cfg())
        model.eval()

        tgt = torch.tensor([[uni_omr.BOS_ID, 0, 1, 2, 3], [uni_omr.BOS_ID, 4, 5, 6, 7]], dtype=torch.long)

        with torch.no_grad():
            strip_img = torch.rand(2, 3, 256, 512)
            out_strip = model(strip_img, tgt, return_aux=False, return_mtp=False)

            page_img = torch.rand(2, 3, 768, 576)
            out_page = model(page_img, tgt, return_aux=False, return_mtp=False)

        self.assertEqual(out_strip["logits"].shape, (2, 5, uni_omr.VOCAB_SIZE))
        self.assertEqual(out_page["logits"].shape, (2, 5, uni_omr.VOCAB_SIZE))
        self.assertEqual(out_strip["router_info"]["grid_hw"], (16, 32))
        self.assertEqual(out_page["router_info"]["grid_hw"], (48, 36))
        self.assertTrue(torch.isfinite(out_strip["logits"]).all())
        self.assertTrue(torch.isfinite(out_page["logits"]).all())

    def test_b_ink_token_router_reduction_determinism_and_gradients(self) -> None:
        """(b) InkTokenRouter reduces tokens (n_out < n_in), is deterministic, and propagates gradients."""
        cfg = _small_cfg(keep_ratio=0.40, n_bg_summary=4)
        model = uni_omr.UniOMRNet(cfg)
        model.train()

        img = torch.rand(2, 3, 128, 256, requires_grad=True)
        tgt_in = torch.randint(0, len(data.VOCAB), (2, 10))
        tgt_out = torch.randint(0, len(data.VOCAB), (2, 10))

        out1 = model(img, tgt_in, return_aux=False, return_mtp=False)
        out2 = model(img, tgt_in, return_aux=False, return_mtp=False)

        rinfo = out1["router_info"]
        # 128x256 / 16 = 8x16 = 128 tokens -> ceil(0.40*128)=52 + 4 bg = 56 < 128
        self.assertEqual(rinfo["n_in"], 128)
        self.assertEqual(rinfo["n_out"], 56)
        self.assertLess(rinfo["n_out"], rinfo["n_in"])

        # Deterministic selection & outputs
        self.assertTrue(torch.equal(out1["router_info"]["kept_idx"], out2["router_info"]["kept_idx"]))
        self.assertTrue(torch.allclose(out1["logits"], out2["logits"]))

        # Gradients flow cleanly to router AND backbone
        losses = model.compute_loss(out1, tgt_out)
        losses["loss"].backward()

        router_grads = [p.grad for p in model.router.parameters() if p.requires_grad]
        backbone_grads = [p.grad for p in model.backbone.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in router_grads))
        self.assertTrue(any(g.abs().sum().item() > 0.0 for g in router_grads))
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in backbone_grads))
        self.assertTrue(any(g.abs().sum().item() > 0.0 for g in backbone_grads))

    def test_c_uv_flow_aux_loss_and_zero_cost_inference_drop(self) -> None:
        """(c) Auxiliary uv_flow head loss computation + zero-cost inference drop."""
        model = uni_omr.UniOMRNet(_small_cfg(use_uv_aux=True))

        img = torch.rand(2, 3, 128, 192)
        tgt_in = torch.randint(0, len(data.VOCAB), (2, 8))
        tgt_out = torch.randint(0, len(data.VOCAB), (2, 8))
        uv_tgt = torch.rand(2, 3, 128, 192)
        uv_tgt[:, 2] = (uv_tgt[:, 2] > 0.2).float()

        # Training mode computes UV head & loss
        model.train()
        out_train = model(img, tgt_in, return_aux=True, return_mtp=False)
        self.assertIsNotNone(out_train["uv_pred"])
        self.assertEqual(out_train["uv_pred"].shape, (2, 3, 8, 12))
        losses = model.compute_loss(out_train, tgt_out, uv_target=uv_tgt, uv_loss_weight=0.20)
        self.assertGreater(losses["uv_loss"].item(), 0.0)
        self.assertTrue(torch.isfinite(losses["uv_loss"]))

        # Eval mode with return_aux=False completely bypasses uv_head (0% infer cost)
        model.eval()
        uv_called = {"count": 0}
        orig_uv_fwd = model.uv_head.forward

        def _spy_uv(*a, **kw):
            uv_called["count"] += 1
            return orig_uv_fwd(*a, **kw)

        model.uv_head.forward = _spy_uv  # type: ignore[method-assign]
        with torch.no_grad():
            out_eval = model(img, tgt_in, return_aux=False, return_mtp=False)
        self.assertIsNone(out_eval["uv_pred"])
        self.assertEqual(uv_called["count"], 0)

        # Also verify load_uv_flow_mask on both .npy and 16-bit PNG
        with tempfile.TemporaryDirectory() as td:
            npy_path = Path(td) / "mask_uv.npy"
            arr = np.random.rand(32, 48, 3).astype(np.float32)
            np.save(npy_path, arr)
            loaded_npy = uni_omr.load_uv_flow_mask(npy_path)
            self.assertEqual(loaded_npy.shape, (3, 32, 48))
            np.testing.assert_allclose(loaded_npy, np.transpose(arr, (2, 0, 1)), atol=1e-6)

            png_path = Path(td) / "mask_uv.png"
            img8 = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
            img8.save(png_path)
            loaded_png = uni_omr.load_uv_flow_mask(png_path)
            self.assertEqual(loaded_png.shape, (3, 32, 48))

    def test_d_mtp4_loss_and_backward_no_nans(self) -> None:
        """(d) MTP-4 loss computation + backward step with no NaNs/Infs."""
        model = uni_omr.UniOMRNet(_small_cfg(mtp_k=4, use_uv_aux=True))
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        img = torch.rand(2, 3, 128, 192)
        seq = torch.randint(0, len(data.VOCAB), (2, 16))
        seq[:, 0] = uni_omr.BOS_ID
        seq[:, -1] = uni_omr.EOS_ID
        uv_tgt = torch.rand(2, 3, 8, 12)
        uv_tgt[:, 2] = 1.0

        opt.zero_grad()
        out = model(img, seq[:, :-1], return_aux=True, return_mtp=True)
        # mtp_k=4 => 3 extra lookahead heads (+2, +3, +4)
        self.assertEqual(len(out["mtp_logits"]), 3)
        for m_log in out["mtp_logits"]:
            self.assertEqual(m_log.shape, out["logits"].shape)

        losses = model.compute_loss(out, seq[:, 1:], uv_target=uv_tgt, mtp_loss_weight=0.30, uv_loss_weight=0.20)
        self.assertGreater(losses["mtp_loss"].item(), 0.0)
        self.assertTrue(torch.isfinite(losses["loss"]))

        losses["loss"].backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Missing grad for {name}")
                self.assertTrue(torch.isfinite(p.grad).all(), f"Non-finite grad in {name}")
        opt.step()

    def test_e_cli_dry_run_and_bench_browser(self) -> None:
        """(e) --dry-run CLI smoke test + bench_browser size/latency readout."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc_dry = uni_omr.main([
                "--mode", "train",
                "--curriculum", "mixed",
                "--d-model", "64",
                "--n-enc-layers", "2",
                "--n-dec-layers", "2",
                "--n-heads", "4",
                "--keep-ratio", "0.40",
                "--mtp-k", "4",
                "--uv-loss-weight", "0.20",
                "--steps", "2",
                "--seq-len", "16",
                "--device", "cpu",
                "--dry-run",
            ])
        self.assertEqual(rc_dry, 0)
        self.assertIn("dry-run", buf.getvalue())

        # Default-config browser budget check (params & quantized sizes)
        default_model = uni_omr.UniOMRNet(uni_omr.UniOMRConfig())
        summary = default_model.param_summary()
        self.assertGreaterEqual(summary["total_params"], 35_000_000)
        self.assertLessEqual(summary["total_params"], 55_000_000)
        self.assertLess(summary["int8_mb"], 55.0)
        self.assertLess(summary["int4_mb"], 28.0)

        # Fast bench_browser CLI invocation (small dims for fast unittest)
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            rc_bench = uni_omr.main([
                "--mode", "bench_browser",
                "--d-model", "64",
                "--n-enc-layers", "2",
                "--n-dec-layers", "2",
                "--n-heads", "4",
                "--device", "cpu",
            ])
        self.assertEqual(rc_bench, 0)
        out_txt = buf2.getvalue()
        self.assertIn("strip_256x1024", out_txt)
        self.assertIn("fullpage_1024x768", out_txt)
        self.assertIn("INT8", out_txt)
        self.assertIn("INT4", out_txt)
        self.assertIn("KV-AR", out_txt)

    def test_f_kv_cache_generate_exact_parity(self) -> None:
        """(f) KV-cached generate() matches uncached step-by-step decode_tokens() identically."""
        model = uni_omr.UniOMRNet(_small_cfg(dropout=0.0))
        model.eval()

        img = torch.rand(2, 3, 128, 256)
        max_len = 16

        with torch.no_grad():
            cached_seq, rinfo = model.generate(img, max_len=max_len, return_info=True)
            self.assertEqual(cached_seq.shape[0], 2)
            self.assertTrue((cached_seq[:, 0] == uni_omr.BOS_ID).all())
            self.assertLess(rinfo["n_out"], rinfo["n_in"])

            # Reference uncached greedy decoding + step-by-step logit parity check
            mem, _, _ = model.encode_image(img, return_aux=False)
            seq_uncached = torch.full((2, 1), uni_omr.BOS_ID, dtype=torch.long)
            finished = torch.zeros(2, dtype=torch.bool)

            # Also step cached decoder manually to compare raw logits at every step
            cur_cached = torch.full((2, 1), uni_omr.BOS_ID, dtype=torch.long)
            past_kvs = None
            max_logit_diff = 0.0

            for _ in range(max_len):
                logits_full, _ = model.decode_tokens(seq_uncached, mem, return_mtp=False)
                step_logits_uncached = logits_full[:, -1, :]

                logits_cached, _, past_kvs = model.decode_tokens(
                    cur_cached,
                    memory=mem if past_kvs is None else None,
                    return_mtp=False,
                    past_kvs=past_kvs,
                    use_cache=True,
                )
                step_logits_cached = logits_cached[:, -1, :]

                # Compare logits on active (non-finished) sequences
                active = ~finished
                if active.any():
                    diff = (step_logits_uncached[active] - step_logits_cached[active]).abs().max().item()
                    max_logit_diff = max(max_logit_diff, diff)

                next_tok = step_logits_uncached.argmax(dim=-1)
                next_tok = torch.where(finished, torch.full_like(next_tok, uni_omr.PAD_ID), next_tok)
                seq_uncached = torch.cat([seq_uncached, next_tok.unsqueeze(1)], dim=1)
                finished = finished | (next_tok == uni_omr.EOS_ID)
                if bool(finished.all()):
                    break
                cur_cached = next_tok.unsqueeze(1)

        self.assertLess(max_logit_diff, 1e-4, f"KV-cache logit drift too large: {max_logit_diff}")
        self.assertTrue(
            torch.equal(cached_seq, seq_uncached),
            f"KV-cached generate() diverged from uncached decode_tokens():\n{cached_seq}\nvs\n{seq_uncached}",
        )

    def test_g_real_strip_and_fullpage_loader_smoke(self) -> None:
        """(g) Real strip + full-page loader smoke test on 2 samples from hw-sim-train-v03-clean."""
        root = Path(r"D:\tmp\hw-sim-train-v03-clean")
        if not root.is_dir():
            self.skipTest(f"Dataset directory not found: {root}")

        strip_pairs = uni_omr.discover_strip_pairs([root / "stage2_strips"], max_samples=2)
        fp_pairs = uni_omr.discover_fullpage_pairs([root / "pages"], max_samples=2)
        self.assertEqual(len(strip_pairs), 2)
        self.assertEqual(len(fp_pairs), 2)

        strip_ds = uni_omr.UniStripDataset(
            strip_pairs, strip_height=192, max_strip_width=1024, max_seq_len=512, augment=False
        )
        strip_ds_aug = uni_omr.UniStripDataset(
            strip_pairs, strip_height=192, max_strip_width=1024, max_seq_len=512, augment=True
        )
        fp_ds = uni_omr.UniFullPageDataset(
            fp_pairs, fullpage_max_dim=768, max_seq_len=1024, augment=False
        )

        # Verify strip shapes, tokens, and online augmentation bounds
        s0 = strip_ds[0]
        self.assertEqual(s0["image"].shape[0], 3)
        self.assertEqual(s0["image"].shape[1], 192)
        self.assertEqual(s0["image"].shape[2] % 16, 0)
        self.assertLessEqual(s0["image"].shape[2], 1024)
        self.assertEqual(int(s0["seq"][0]), uni_omr.BOS_ID)
        self.assertEqual(int(s0["seq"][-1]), uni_omr.EOS_ID)

        s0_aug = strip_ds_aug[0]
        self.assertEqual(s0_aug["image"].shape, s0["image"].shape)
        self.assertGreaterEqual(float(s0_aug["image"].min()), 0.0)
        self.assertLessEqual(float(s0_aug["image"].max()), 1.0)

        # Verify full-page shapes & <sys> stitching
        fp0 = fp_ds[0]
        _, fh, fw = fp0["image"].shape
        self.assertEqual(fh % 16, 0)
        self.assertEqual(fw % 16, 0)
        self.assertLessEqual(max(fh, fw), 768)
        self.assertEqual(int(fp0["seq"][0]), uni_omr.BOS_ID)
        self.assertEqual(int(fp0["seq"][-1]), uni_omr.EOS_ID)
        if len(fp_pairs[0][1]) > 1:
            self.assertIn(uni_omr.SYS_ID, fp0["seq"].tolist())
            decoded_fp = uni_omr.decode_uni(fp0["seq"].tolist())
            self.assertIn(uni_omr.SYS_TOKEN, decoded_fp)

        # Verify clean batch collation + CPU eval_dataset metrics on both modalities
        batch_s = uni_omr.collate_uni_batch([strip_ds[0], strip_ds[1]])
        batch_fp = uni_omr.collate_uni_batch([fp_ds[0], fp_ds[1]])
        self.assertEqual(batch_s["images"].shape[0], 2)
        self.assertEqual(batch_fp["images"].shape[0], 2)

        model = uni_omr.UniOMRNet(_small_cfg(dropout=0.0))
        model.eval()
        m_strip = uni_omr.eval_dataset(model, strip_ds, device=torch.device("cpu"), max_len=16, batch_size=2)
        m_fp = uni_omr.eval_dataset(model, fp_ds, device=torch.device("cpu"), max_len=16, batch_size=2)

        for metrics in (m_strip, m_fp):
            self.assertEqual(metrics["n_samples"], 2)
            self.assertGreaterEqual(metrics["ser"], 0.0)
            self.assertGreaterEqual(metrics["token_f1"], 0.0)
            self.assertLessEqual(metrics["token_f1"], 1.0)
            self.assertGreaterEqual(metrics["sys_count_exact"], 0.0)
            self.assertLessEqual(metrics["sys_count_exact"], 1.0)
            self.assertGreater(metrics["tokens_per_sec"], 0.0)
            self.assertGreater(metrics["mean_compression_x"], 1.0)

    def test_h_mixed_train_real_data_checkpoints_and_eval_summary(self) -> None:
        """(h) End-to-end mixed train loop on 2 real strips + 2 real pages writes checkpoints + eval_summary."""
        root = Path(r"D:\tmp\hw-sim-train-v03-clean")
        if not root.is_dir():
            self.skipTest(f"Dataset directory not found: {root}")

        fp_pairs = uni_omr.discover_fullpage_pairs([root / "pages"], max_samples=2)
        self.assertEqual(len(fp_pairs), 2)
        # Pick 1 strip from each of the 2 distinct pages so piece-aware split produces 1 train / 1 val.
        strip_pairs = []
        for _, sys_paths in fp_pairs:
            b0 = sys_paths[0]
            stem0 = b0.name[: -len("-bekern.txt")]
            img0 = b0.parent / f"{stem0}.png"
            strip_pairs.append((img0, b0))
        self.assertEqual(len(strip_pairs), 2)

        import shutil
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            s_dir = td_path / "stage2_strips"
            p_dir = td_path / "pages"
            out_dir = td_path / "out"
            s_dir.mkdir()
            p_dir.mkdir()

            for img_p, bek_p in strip_pairs:
                shutil.copy2(img_p, s_dir / img_p.name)
                shutil.copy2(bek_p, s_dir / bek_p.name)
            for page_p, sys_paths in fp_pairs:
                shutil.copy2(page_p, p_dir / page_p.name)
                for sp in sys_paths:
                    if not (s_dir / sp.name).exists():
                        shutil.copy2(sp, s_dir / sp.name)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = uni_omr.main([
                    "--mode", "train",
                    "--curriculum", "mixed",
                    "--strip-dirs", str(s_dir),
                    "--fullpage-dirs", str(p_dir),
                    "--strip-height", "128",
                    "--max-strip-width", "384",
                    "--fullpage-max-dim", "256",
                    "--max-seq-len", "64",
                    "--augment",
                    "--d-model", "64",
                    "--n-enc-layers", "1",
                    "--n-dec-layers", "1",
                    "--n-heads", "4",
                    "--steps", "3",
                    "--warmup-steps", "1",
                    "--grad-accum", "2",
                    "--batch-size", "1",
                    "--val-count", "1",
                    "--eval-every", "3",
                    "--eval-max-len", "12",
                    "--device", "cpu",
                    "--out-dir", str(out_dir),
                ])
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "uni_omr_best.pt").is_file())
            self.assertTrue((out_dir / "uni_omr_last.pt").is_file())
            self.assertTrue((out_dir / "train_log.json").is_file())
            self.assertTrue((out_dir / "eval_summary.json").is_file())
            out_txt = buf.getvalue()
            self.assertIn("eval strip_val", out_txt)
            self.assertIn("eval fullpage_val", out_txt)

    def test_i_hw_strip_balanced_sampling_and_aug_prob(self) -> None:
        """(i) --hw-strip-dirs / --hw-frac balanced sampling, separate val metrics, and --aug-prob."""
        # 1. Exact deterministic 70/30 ratio from StripDomainMixer
        mixer = uni_omr.StripDomainMixer(hw_frac=0.30)
        draws = [mixer.next_is_hw() for _ in range(20)]
        self.assertEqual(sum(draws), 6)  # 30% HW
        self.assertEqual(len(draws) - sum(draws), 14)  # 70% Printed anchor replay

        import json
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            pr_dir = td_path / "printed"
            hw_dir = td_path / "hw"
            out_dir = td_path / "out"
            pr_dir.mkdir()
            hw_dir.mkdir()

            bek_txt = "clef-G-2\nquarter\nC\n4\nbar\n"
            for i in range(4):
                arr = np.full((64, 160, 3), 245, dtype=np.uint8)
                arr[20:44, 15 + i * 10 : 140, :] = 30
                Image.fromarray(arr, mode="RGB").save(pr_dir / f"prpiece{i}_p1_s0.png")
                (pr_dir / f"prpiece{i}_p1_s0-bekern.txt").write_text(bek_txt, encoding="utf-8")

                arr_hw = np.full((64, 160, 3), 240, dtype=np.uint8)
                arr_hw[24:40, 20 + i * 8 : 135, :] = 50
                Image.fromarray(arr_hw, mode="RGB").save(hw_dir / f"hwpiece{i}_p1_s0.png")
                (hw_dir / f"hwpiece{i}_p1_s0-bekern.txt").write_text(bek_txt, encoding="utf-8")

            # 2. Verify probabilistic --aug-prob behavior & enriched augment bounds
            pairs = uni_omr.discover_strip_pairs([pr_dir])
            ds_clean = uni_omr.UniStripDataset(pairs, strip_height=64, max_strip_width=160, augment=False)
            ds_p0 = uni_omr.UniStripDataset(pairs, strip_height=64, max_strip_width=160, augment=True, aug_prob=0.0)
            ds_p1 = uni_omr.UniStripDataset(pairs, strip_height=64, max_strip_width=160, augment=True, aug_prob=1.0)
            ds_p75 = uni_omr.UniStripDataset(pairs, strip_height=64, max_strip_width=160, augment=True, aug_prob=0.75)

            base_img = ds_clean[0]["image"]
            self.assertTrue(torch.equal(base_img, ds_p0[0]["image"]))

            aug_img = ds_p1[0]["image"]
            self.assertEqual(aug_img.shape, base_img.shape)
            self.assertGreaterEqual(float(aug_img.min()), 0.0)
            self.assertLessEqual(float(aug_img.max()), 1.0)

            import random as _py_random
            _py_random.seed(123)
            n_changed = sum(0 if torch.equal(base_img, ds_p75[0]["image"]) else 1 for _ in range(30))
            self.assertGreater(n_changed, 0)
            self.assertLess(n_changed, 30)

            # 3. End-to-end CLI run with --strip-dirs + --hw-strip-dirs + --hw-frac 0.30
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = uni_omr.main([
                    "--mode", "train",
                    "--curriculum", "strip",
                    "--strip-dirs", str(pr_dir),
                    "--hw-strip-dirs", str(hw_dir),
                    "--hw-frac", "0.30",
                    "--strip-height", "64",
                    "--max-strip-width", "160",
                    "--max-seq-len", "32",
                    "--augment",
                    "--aug-prob", "0.75",
                    "--d-model", "64",
                    "--n-enc-layers", "1",
                    "--n-dec-layers", "1",
                    "--n-heads", "4",
                    "--steps", "10",
                    "--warmup-steps", "1",
                    "--batch-size", "1",
                    "--val-count", "1",
                    "--eval-every", "10",
                    "--eval-max-len", "10",
                    "--device", "cpu",
                    "--out-dir", str(out_dir),
                ])
            self.assertEqual(rc, 0)
            out_txt = buf.getvalue()
            self.assertIn("eval printed_strip_val", out_txt)
            self.assertIn("eval hw_strip_val", out_txt)

            train_log = json.loads((out_dir / "train_log.json").read_text(encoding="utf-8"))
            kinds = [row["kind"] for row in train_log]
            self.assertEqual(kinds.count("hw_strip"), 3)
            self.assertEqual(kinds.count("strip"), 7)

            eval_sum = json.loads((out_dir / "eval_summary.json").read_text(encoding="utf-8"))
            self.assertIn("printed_strip_val", eval_sum["evals"][0])
            self.assertIn("hw_strip_val", eval_sum["evals"][0])

    def test_j_confidence_gated_ema_unsup_round(self) -> None:
        """(j) Confidence-gated weak->strong EMA unsupervised self-training smoke test (CPU-only)."""
        import random as _py_random
        from unittest import mock

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            unlab_root = td_path / "unlabeled"
            sub_dir = unlab_root / "nested"
            pr_dir = td_path / "printed"
            out_dir = td_path / "out"
            sub_dir.mkdir(parents=True)
            pr_dir.mkdir()

            # Create nested unlabeled PNG + JPG images
            img_arr = np.full((96, 128, 3), 245, dtype=np.uint8)
            img_arr[30:60, 20:100, :] = 25
            Image.fromarray(img_arr, mode="RGB").save(unlab_root / "u0.png")
            Image.fromarray(img_arr, mode="RGB").save(sub_dir / "u1.jpg")

            discovered = uni_omr.discover_unlabeled_images([unlab_root])
            self.assertEqual(len(discovered), 2)

            # Build tiny student + CPU-offloaded EMA teacher
            cfg = _small_cfg(d_model=64, n_enc_layers=1, n_dec_layers=1, n_heads=4, dropout=0.0)
            student = uni_omr.UniOMRNet(cfg)
            ema = uni_omr.CPUEmaTeacher(student, decay=0.995)
            opt = torch.optim.AdamW(student.parameters(), lr=1e-3)

            # Verify EMA shadow is CPU-resident and updates smoothly
            before_w = ema.shadow["lm_head.weight"].clone()
            with torch.no_grad():
                student.lm_head.weight.add_(1.0)
            ema.update(student)
            self.assertEqual(ema.shadow["lm_head.weight"].device.type, "cpu")
            self.assertFalse(torch.equal(before_w, ema.shadow["lm_head.weight"]))

            # 1. Strict confidence gate rejects low-confidence samples without modifying student
            w_snap = student.lm_head.weight.detach().clone()
            stats_strict = uni_omr.run_unsup_ema_round(
                student=student,
                ema_teacher=ema,
                opt=opt,
                unlabeled_paths=discovered,
                device=torch.device("cpu"),
                curriculum="fullpage",
                fullpage_max_dim=128,
                unsup_steps=2,
                unsup_conf_thr=0.9999,
                unsup_weight=0.5,
                max_gen_len=10,
                rng=_py_random.Random(7),
            )
            self.assertEqual(stats_strict["total"], 2)
            self.assertEqual(stats_strict["accepted"], 0)
            self.assertTrue(torch.equal(w_snap, student.lm_head.weight.detach()))

            # 2. High-confidence EOS sequence (len >= 6) passes gate; low-conf & short sequences are rejected
            valid_seq = torch.tensor([[uni_omr.BOS_ID, 0, 1, 2, 3, 4, uni_omr.EOS_ID]], dtype=torch.long)
            short_seq = torch.tensor([[uni_omr.BOS_ID, 0, 1, uni_omr.EOS_ID]], dtype=torch.long)
            fake_returns = [
                (valid_seq, torch.tensor([0.92]), torch.tensor([True])),   # PASS
                (valid_seq, torch.tensor([0.50]), torch.tensor([True])),   # REJECT (conf < 0.85)
                (short_seq, torch.tensor([0.95]), torch.tensor([True])),   # REJECT (len < 6)
                (valid_seq, torch.tensor([0.96]), torch.tensor([False])),  # REJECT (no EOS)
            ]
            with mock.patch.object(uni_omr.UniOMRNet, "generate", side_effect=fake_returns):
                stats_gated = uni_omr.run_unsup_ema_round(
                    student=student,
                    ema_teacher=ema,
                    opt=opt,
                    unlabeled_paths=discovered,
                    device=torch.device("cpu"),
                    curriculum="strip",
                    strip_height=64,
                    max_strip_width=128,
                    unsup_steps=4,
                    unsup_conf_thr=0.85,
                    unsup_weight=0.5,
                    max_gen_len=10,
                    rng=_py_random.Random(11),
                )
            self.assertEqual(stats_gated["total"], 4)
            self.assertEqual(stats_gated["accepted"], 1)
            self.assertGreater(stats_gated["unsup_loss"], 0.0)
            self.assertFalse(torch.equal(w_snap, student.lm_head.weight.detach()))

            # 3. End-to-end CLI smoke test with --unlabeled-dirs and --unsup-every 2
            bek_txt = "clef-G-2\nquarter\nC\n4\nbar\n"
            for i in range(2):
                Image.fromarray(img_arr, mode="RGB").save(pr_dir / f"p{i}_s0.png")
                (pr_dir / f"p{i}_s0-bekern.txt").write_text(bek_txt, encoding="utf-8")

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = uni_omr.main([
                    "--mode", "train",
                    "--curriculum", "strip",
                    "--strip-dirs", str(pr_dir),
                    "--unlabeled-dirs", str(unlab_root),
                    "--unsup-every", "2",
                    "--unsup-steps", "2",
                    "--unsup-conf-thr", "0.85",
                    "--unsup-weight", "0.5",
                    "--strip-height", "64",
                    "--max-strip-width", "128",
                    "--max-seq-len", "32",
                    "--d-model", "64",
                    "--n-enc-layers", "1",
                    "--n-dec-layers", "1",
                    "--n-heads", "4",
                    "--steps", "2",
                    "--warmup-steps", "1",
                    "--batch-size", "1",
                    "--val-count", "1",
                    "--eval-every", "2",
                    "--eval-max-len", "8",
                    "--device", "cpu",
                    "--out-dir", str(out_dir),
                ])
            self.assertEqual(rc, 0)
            out_txt = buf.getvalue()
            self.assertIn("[unsup EMA @ 2]", out_txt)
            self.assertIn("unsup_loss=", out_txt)

    def test_k_anisotropic_stride_4x8_shape_and_weight_compat(self) -> None:
        """(k) backbone_stride='4x8' gives (H/4, W/8) grid & 100% weight compat with stride=8/16."""
        self.assertEqual(_small_cfg(backbone_stride="8").backbone_stride, 8)
        self.assertEqual(_small_cfg(backbone_stride="16").backbone_stride, 16)
        self.assertEqual(_small_cfg(backbone_stride="4x8").backbone_stride, "4x8")

        m8 = uni_omr.UniOMRNet(_small_cfg(backbone_stride=8, use_uv_aux=True))
        m4x8 = uni_omr.UniOMRNet(_small_cfg(backbone_stride="4x8", use_uv_aux=True))

        # Exact state_dict compatibility: missing=0, unexpected=0, zero extra params
        inc = m4x8.load_state_dict(m8.state_dict(), strict=True)
        self.assertEqual(len(inc.missing_keys), 0)
        self.assertEqual(len(inc.unexpected_keys), 0)
        self.assertEqual(m8.param_summary()["total_params"], m4x8.param_summary()["total_params"])

        m4x8.train()
        # Strip (2, 3, 128, 256) -> (H/4, W/8) = (32, 32)
        strip_img = torch.rand(2, 3, 128, 256)
        tgt_in = torch.randint(0, len(data.VOCAB), (2, 8))
        tgt_out = torch.randint(0, len(data.VOCAB), (2, 8))
        uv_tgt = torch.rand(2, 3, 128, 256)
        uv_tgt[:, 2] = 1.0

        out_strip = m4x8(strip_img, tgt_in, return_aux=True, return_mtp=False)
        self.assertEqual(out_strip["router_info"]["grid_hw"], (32, 32))
        self.assertEqual(out_strip["uv_pred"].shape, (2, 3, 32, 32))
        losses = m4x8.compute_loss(out_strip, tgt_out, uv_target=uv_tgt, uv_loss_weight=0.20)
        self.assertTrue(torch.isfinite(losses["loss"]))

        # Fullpage (2, 3, 192, 128) -> (H/4, W/8) = (48, 16)
        page_img = torch.rand(2, 3, 192, 128)
        out_page = m4x8(page_img, tgt_in, return_aux=True, return_mtp=False)
        self.assertEqual(out_page["router_info"]["grid_hw"], (48, 16))
        self.assertEqual(out_page["uv_pred"].shape, (2, 3, 48, 16))
        self.assertTrue(torch.isfinite(out_page["logits"]).all())

    def test_l_mixstyle2d_train_only_behavior(self) -> None:
        """(l) MixStyle2d is train-only, B>=2 only, 0-param, and preserves gradients."""
        ms = uni_omr.MixStyle2d(p=1.0, alpha=0.1)
        self.assertEqual(sum(p.numel() for p in ms.parameters()), 0)
        self.assertEqual(len(list(ms.buffers())), 0)

        # Distinct per-sample channel statistics so mixing always changes activations
        torch.manual_seed(7)
        x = torch.randn(4, 16, 8, 8)
        for b in range(4):
            x[b] = x[b] * (b + 1.0) + (b * 5.0)
        x.requires_grad_(True)

        # Eval mode -> exact identity
        ms.eval()
        self.assertTrue(torch.equal(ms(x), x))

        # Train mode with B=1 -> exact identity
        ms.train()
        x1 = x[:1].detach().clone()
        self.assertTrue(torch.equal(ms(x1), x1))

        # Train mode with p=0.0 -> exact identity
        ms0 = uni_omr.MixStyle2d(p=0.0, alpha=0.1)
        ms0.train()
        self.assertTrue(torch.equal(ms0(x), x))

        # Train mode with B=4, p=1.0 -> modifies activations & propagates finite grads
        torch.manual_seed(123)
        y = ms(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        self.assertFalse(torch.allclose(y, x))
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

        # Weight compatibility with mixstyle_prob=0.0
        m0 = uni_omr.UniOMRNet(_small_cfg(mixstyle_prob=0.0))
        m3 = uni_omr.UniOMRNet(_small_cfg(mixstyle_prob=0.30))
        self.assertEqual(m0.param_summary()["total_params"], m3.param_summary()["total_params"])
        inc = m3.load_state_dict(m0.state_dict(), strict=True)
        self.assertEqual(len(inc.missing_keys), 0)
        self.assertEqual(len(inc.unexpected_keys), 0)

    def test_m_iter6_cass_soup_multiscale_dynamic_routing_and_shared_val(self) -> None:
        """(m) 4-step CPU smoke test: 4x8 + MixStyle + CASS + soup-last-k=2 + multi-scale + dynamic-routing + shared val."""
        import json

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            pr_dir = td_path / "printed"
            hw_dir = td_path / "hw"
            fp_root = td_path / "fullpage"
            fp_pages = fp_root / "pages"
            fp_strips = fp_root / "stage2_strips"
            out_dir = td_path / "out"
            for d in (pr_dir, hw_dir, fp_pages, fp_strips):
                d.mkdir(parents=True)

            bek_txt = "clef-G-2\nquarter\nC\n4\nbar\n"
            # Create 6 pieces across printed, hw, and fullpage (with overlapping piece names
            # to exercise cross-modality shared piece-level val split).
            for i in range(6):
                arr_s = np.full((64, 128, 3), 245, dtype=np.uint8)
                arr_s[20:44, 15 + i * 5 : 110, :] = 30
                Image.fromarray(arr_s, mode="RGB").save(pr_dir / f"piece{i}_p1_s0.png")
                (pr_dir / f"piece{i}_p1_s0-bekern.txt").write_text(bek_txt, encoding="utf-8")

                Image.fromarray(arr_s, mode="RGB").save(hw_dir / f"piece{i}_p1_s0.png")
                (hw_dir / f"piece{i}_p1_s0-bekern.txt").write_text(bek_txt, encoding="utf-8")

                arr_p = np.full((128, 96, 3), 245, dtype=np.uint8)
                arr_p[30:90, 12:84, :] = 25
                Image.fromarray(arr_p, mode="RGB").save(fp_pages / f"piece{i}_p1.png")
                (fp_strips / f"piece{i}_p1_s0-bekern.txt").write_text(bek_txt, encoding="utf-8")

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = uni_omr.main([
                    "--mode", "train",
                    "--curriculum", "mixed",
                    "--strip-dirs", str(pr_dir),
                    "--hw-strip-dirs", str(hw_dir),
                    "--fullpage-dirs", str(fp_root),
                    "--backbone-stride", "4x8",
                    "--mixstyle-prob", "0.30",
                    "--cass-prob", "0.25",
                    "--soup-last-k", "2",
                    "--multi-scale",
                    "--dynamic-routing",
                    "--strip-height", "64",
                    "--max-strip-width", "128",
                    "--fullpage-max-dim", "128",
                    "--max-seq-len", "32",
                    "--d-model", "64",
                    "--n-enc-layers", "1",
                    "--n-dec-layers", "1",
                    "--n-heads", "4",
                    "--steps", "4",
                    "--warmup-steps", "1",
                    "--batch-size", "2",
                    "--val-count", "1",
                    "--eval-every", "1",
                    "--eval-max-len", "8",
                    "--device", "cpu",
                    "--out-dir", str(out_dir),
                ])
            self.assertEqual(rc, 0)

            # Rolling K=2 checkpoints: step1 & step2 pruned, step3 & step4 kept
            self.assertFalse((out_dir / "uni_omr_step1.pt").exists())
            self.assertFalse((out_dir / "uni_omr_step2.pt").exists())
            self.assertTrue((out_dir / "uni_omr_step3.pt").is_file())
            self.assertTrue((out_dir / "uni_omr_step4.pt").is_file())
            self.assertTrue((out_dir / "uni_omr_soup.pt").is_file())

            # Verify souped weights match exact average of step3 & step4
            ck3 = torch.load(str(out_dir / "uni_omr_step3.pt"), map_location="cpu", weights_only=True)["model"]
            ck4 = torch.load(str(out_dir / "uni_omr_step4.pt"), map_location="cpu", weights_only=True)["model"]
            cks = torch.load(str(out_dir / "uni_omr_soup.pt"), map_location="cpu", weights_only=True)["model"]
            for k in cks:
                if cks[k].is_floating_point():
                    expected = 0.5 * (ck3[k].float() + ck4[k].float())
                    self.assertTrue(torch.allclose(cks[k].float(), expected, atol=1e-6), f"Soup mismatch on {k}")

            # Verify eval_summary.json records soup_eval and ink_recall
            eval_sum = json.loads((out_dir / "eval_summary.json").read_text(encoding="utf-8"))
            self.assertIn("soup_eval", eval_sum)
            self.assertEqual(eval_sum["soup_eval"]["soup_ckpts"], ["uni_omr_step3.pt", "uni_omr_step4.pt"])
            self.assertIn("mean_val_ser", eval_sum["soup_eval"])
            self.assertIn("ink_recall", eval_sum["evals"][0]["fullpage_val"])

    def test_n_router_local_highpass_ink_recall_and_dark_border_immunity(self) -> None:
        """(n) InkTokenRouter local high-pass resists dark desk borders & adversarial saliency_net init."""
        cfg = _small_cfg(keep_ratio=0.40, n_bg_summary=4, use_ink_prior=True, backbone_stride=16)
        model = uni_omr.UniOMRNet(cfg)
        model.eval()

        # Synthetic 384x384 page: 15% solid black desk border (~58px) + warm tan paper + thin 2px staff lines
        H, W = 384, 384
        border = int(round(H * 0.15))  # 58px (> 31px high-pass kernel)
        img = torch.zeros(1, 3, H, W, dtype=torch.float32)
        # Warm aged tan parchment interior
        img[:, 0, border : H - border, border : W - border] = 0.94
        img[:, 1, border : H - border, border : W - border] = 0.89
        img[:, 2, border : H - border, border : W - border] = 0.78

        # Draw 2 systems of 5 thin 2-px black staff lines + small notehead blobs
        staff_ys = [110, 120, 130, 140, 150, 220, 230, 240, 250, 260]
        x0, x1 = border + 24, W - border - 24
        for sy in staff_ys:
            img[:, :, sy : sy + 2, x0:x1] = 0.02
        for nx in range(x0 + 16, x1 - 16, 28):
            img[:, :, 124:132, nx : nx + 8] = 0.02
            img[:, :, 234:242, nx : nx + 8] = 0.02

        # Adversarial saliency_net init (simulating Iter 5 unbounded negative logit inversion)
        with torch.no_grad():
            for m in model.router.saliency_net.modules():
                if isinstance(m, torch.nn.Conv2d) and m.bias is not None:
                    m.bias.fill_(-8.0)

        tgt_in = torch.tensor([[uni_omr.BOS_ID, 0, 1, 2, 3]], dtype=torch.long)
        tgt_out = torch.tensor([[0, 1, 2, 3, uni_omr.EOS_ID]], dtype=torch.long)
        out = model(img, tgt_in, return_aux=False, return_mtp=False)
        rinfo = out["router_info"]

        self.assertIn("saliency_logits", rinfo)
        self.assertIn("ink_target", rinfo)
        self.assertIn("ink_recall", rinfo)
        self.assertGreaterEqual(
            rinfo["ink_recall"],
            0.85,
            f"Expected ink_recall >= 0.85 under dark border + adversarial init, got {rinfo['ink_recall']:.4f}",
        )

        # Verify deep black border tokens are NOT preferred over staff-line ink tokens
        score_2d = rinfo["saliency_map"][0, 0]  # (H', W') = (24, 24)
        border_scores = score_2d[:2, :].mean().item()  # top 32px: strictly inside 58px black border
        ink_mask_2d = rinfo["ink_target"][0, 0] > 0.5
        ink_scores = score_2d[ink_mask_2d].mean().item()
        self.assertGreater(
            ink_scores,
            border_scores + 0.15,
            f"Border tokens ({border_scores:.4f}) must not rival ink tokens ({ink_scores:.4f})",
        )

        # Verify bounded score in [0, 2] and finite positive saliency BCE loss
        self.assertGreaterEqual(float(score_2d.min().item()), 0.0)
        self.assertLessEqual(float(score_2d.max().item()), 2.0)
        losses = model.compute_loss(out, tgt_out, saliency_loss_weight=0.10)
        self.assertIn("saliency_loss", losses)
        self.assertTrue(torch.isfinite(losses["saliency_loss"]))
        self.assertGreater(float(losses["saliency_loss"].item()), 0.0)

    def test_o_sinusoidal_pos2d_normalized_no_aliasing(self) -> None:
        """(o) _sinusoidal_pos2d normalizes integer grid coords to [0, 1] before * 128 -> scale-invariant & smooth."""
        model = uni_omr.UniOMRNet(_small_cfg(d_model=64))
        model.eval()

        # Same fractional positions (0.0, 0.5, 1.0) on 128x128 vs 320x320 grids must match identically
        c128 = torch.tensor([[[0.0, 0.0], [63.5, 63.5], [127.0, 127.0]]], dtype=torch.float32)
        c320 = torch.tensor([[[0.0, 0.0], [159.5, 159.5], [319.0, 319.0]]], dtype=torch.float32)

        pe128 = model._sinusoidal_pos2d(c128, dtype=torch.float32, grid_hw=(128, 128))
        pe320 = model._sinusoidal_pos2d(c320, dtype=torch.float32, grid_hw=(320, 320))
        self.assertEqual(pe128.shape, (1, 3, 64))
        self.assertTrue(
            torch.allclose(pe128, pe320, atol=1e-5),
            "Normalized _sinusoidal_pos2d must match across different (H', W') grid sizes",
        )

        # Adjacent rows on a 320x320 grid must be smooth (high cosine similarity, no 128x aliasing)
        ys = torch.arange(320, dtype=torch.float32)
        xs = torch.full_like(ys, 160.0)
        col_coords = torch.stack([ys, xs], dim=-1).unsqueeze(0)  # (1, 320, 2)
        pe_col = model._sinusoidal_pos2d(col_coords, dtype=torch.float32, grid_hw=(320, 320))[0]
        cos_adj = torch.nn.functional.cosine_similarity(pe_col[:-1], pe_col[1:], dim=-1)
        self.assertGreater(
            float(cos_adj.min().item()),
            0.95,
            f"Expected smooth adjacent-row cosine similarity > 0.95, got min={float(cos_adj.min().item()):.4f}",
        )

    def test_p_online_augment_dark_border_and_reset_router(self) -> None:
        """(p) apply_online_augment dark-border/perspective/1-bit + --reset-router + --fullpage-frac."""
        import json
        import random as _py_random

        # 1. Clean synthetic white page with dark staves
        arr = np.full((256, 192, 3), 250, dtype=np.uint8)
        arr[60:200:20, 24:168, :] = 15
        base = Image.fromarray(arr, mode="RGB")

        # Deterministic per RNG seed + exact size preservation
        aug_a = uni_omr.apply_online_augment(base, rng=_py_random.Random(2026))
        aug_b = uni_omr.apply_online_augment(base, rng=_py_random.Random(2026))
        self.assertEqual(aug_a.size, base.size)
        np.testing.assert_array_equal(np.asarray(aug_a), np.asarray(aug_b))

        # Across 30 deterministic seeds, verify dark-border framing & 1-bit scanner variants occur
        saw_dark_corner = False
        saw_binary = False
        for s in range(30):
            im = uni_omr.apply_online_augment(base, rng=_py_random.Random(100 + s))
            self.assertEqual(im.size, base.size)
            a = np.asarray(im)
            if int(a[0, 0].mean()) < 80:
                saw_dark_corner = True
            uniq = np.unique(a)
            if len(uniq) <= 2 and set(uniq.tolist()).issubset({0, 255}):
                saw_binary = True
        self.assertTrue(saw_dark_corner, "Expected dark desk border framing across 30 seeds")
        self.assertTrue(saw_binary, "Expected 1-bit scanner threshold variant across 30 seeds")

        # 2. Verify --reset-router resets saliency_net & ink_prior_scale while preserving backbone/decoder,
        #    and --fullpage-frac controls mixed curriculum split.
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            pr_dir = td_path / "printed"
            fp_root = td_path / "fullpage"
            fp_pages = fp_root / "pages"
            fp_strips = fp_root / "stage2_strips"
            out_dir = td_path / "out"
            for d in (pr_dir, fp_pages, fp_strips):
                d.mkdir(parents=True)

            bek_txt = "clef-G-2\nquarter\nC\n4\nbar\n"
            for i in range(3):
                arr_s = np.full((64, 128, 3), 245, dtype=np.uint8)
                arr_s[20:44, 15:110, :] = 30
                Image.fromarray(arr_s, mode="RGB").save(pr_dir / f"piece{i}_p1_s0.png")
                (pr_dir / f"piece{i}_p1_s0-bekern.txt").write_text(bek_txt, encoding="utf-8")

                arr_p = np.full((128, 96, 3), 245, dtype=np.uint8)
                arr_p[30:90, 12:84, :] = 25
                Image.fromarray(arr_p, mode="RGB").save(fp_pages / f"piece{i}_p1.png")
                (fp_strips / f"piece{i}_p1_s0-bekern.txt").write_text(bek_txt, encoding="utf-8")

            # Save a synthetic checkpoint with sentinel backbone weight & corrupted router
            cfg = _small_cfg(d_model=64, n_enc_layers=1, n_dec_layers=1, n_heads=4, ffn_mult=4.0)
            src_model = uni_omr.UniOMRNet(cfg)
            with torch.no_grad():
                src_model.backbone.proj.weight.fill_(0.1234)
                src_model.lm_head.weight.fill_(0.0567)
                for m in src_model.router.saliency_net.modules():
                    if isinstance(m, torch.nn.Conv2d) and m.bias is not None:
                        m.bias.fill_(-9.5)
                src_model.router.ink_prior_scale.fill_(-7.0)
            init_ckpt = td_path / "init_corrupt_router.pt"
            torch.save({"model": src_model.state_dict()}, init_ckpt)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = uni_omr.main([
                    "--mode", "train",
                    "--curriculum", "mixed",
                    "--fullpage-frac", "0.70",
                    "--strip-dirs", str(pr_dir),
                    "--fullpage-dirs", str(fp_root),
                    "--init-from", str(init_ckpt),
                    "--reset-router",
                    "--strip-height", "64",
                    "--max-strip-width", "128",
                    "--fullpage-max-dim", "128",
                    "--max-seq-len", "32",
                    "--d-model", "64",
                    "--n-enc-layers", "1",
                    "--n-dec-layers", "1",
                    "--n-heads", "4",
                    "--steps", "10",
                    "--lr", "0.0",
                    "--warmup-steps", "1",
                    "--batch-size", "1",
                    "--val-count", "1",
                    "--eval-every", "10",
                    "--eval-max-len", "8",
                    "--device", "cpu",
                    "--out-dir", str(out_dir),
                ])
            self.assertEqual(rc, 0)
            self.assertIn("[UniOMR reset-router]", buf.getvalue())

            # With lr=0.0, saved weights verify exact preservation of backbone/decoder and reset of router
            saved = torch.load(str(out_dir / "uni_omr_last.pt"), map_location="cpu", weights_only=True)["model"]
            self.assertTrue(torch.allclose(saved["backbone.proj.weight"], torch.full_like(saved["backbone.proj.weight"], 0.1234)))
            self.assertTrue(torch.allclose(saved["lm_head.weight"], torch.full_like(saved["lm_head.weight"], 0.0567)))
            self.assertAlmostEqual(float(saved["router.ink_prior_scale"].item()), 1.0, places=5)
            self.assertGreater(float(saved["router.saliency_net.3.bias"].item()), -5.0)

            # Verify --fullpage-frac 0.70 gave 7 fullpage / 3 strip steps
            train_log = json.loads((out_dir / "train_log.json").read_text(encoding="utf-8"))
            kinds = [row["kind"] for row in train_log]
            self.assertEqual(kinds.count("fullpage"), 7)
            self.assertEqual(kinds.count("strip"), 3)

    def test_q_speedups_separable_pool_uint8_multiscale_and_prefetcher(self) -> None:
        """(q) Verify F.rms_norm parity, separable avg_pool2d parity, uint8 H2D, per-batch multi-scale, and AsyncThreadPrefetcher."""
        # 1. RMSNorm parity with manual formula
        norm = uni_omr.RMSNorm(64, eps=1e-6)
        x = torch.randn(2, 12, 64)
        y_fast = norm(x)
        y_ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + norm.eps) * norm.weight
        self.assertTrue(torch.allclose(y_fast, y_ref, atol=1e-5))

        # 2. Separable 2-pass avg_pool2d matches 2D 31x31 box filter identically
        luma = torch.rand(2, 1, 96, 128)
        bg_2d = torch.nn.functional.avg_pool2d(luma, kernel_size=31, stride=1, padding=15, count_include_pad=False)
        bg_sep = torch.nn.functional.avg_pool2d(
            torch.nn.functional.avg_pool2d(luma, kernel_size=(1, 31), stride=1, padding=(0, 15), count_include_pad=False),
            kernel_size=(31, 1), stride=1, padding=(15, 0), count_include_pad=False,
        )
        self.assertTrue(torch.allclose(bg_2d, bg_sep, atol=1e-6))

        # 3. uint8 dataset + per-batch multi-scale + AsyncThreadPrefetcher + eval_dataset
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            pages_dir = td_path / "pages"
            strips_dir = td_path / "stage2_strips"
            pages_dir.mkdir()
            strips_dir.mkdir()
            bek_txt = "clef-G-2\nquarter\nC\n4\nbar\n"
            for i in range(6):
                arr = np.full((1280, 960, 3), 240, dtype=np.uint8)
                arr[200:900:80, 100:860, :] = 20
                Image.fromarray(arr, mode="RGB").save(pages_dir / f"ms{i}_p1.png")
                (strips_dir / f"ms{i}_p1_s0-bekern.txt").write_text(bek_txt, encoding="utf-8")

            fp_pairs = uni_omr.discover_fullpage_pairs([td_path])
            ds_u8 = uni_omr.UniFullPageDataset(
                fp_pairs, fullpage_max_dim=1280, max_seq_len=32, augment=False, multi_scale=True, return_uint8=True
            )
            # MultiScaleBatchSampler assigns identical target_dim to every item in a batch
            sampler = uni_omr.MultiScaleBatchSampler(len(ds_u8), batch_size=3, base_dim=1280, shuffle=True, seed=7)
            for batch_spec in sampler:
                dims = {d for _, d in batch_spec}
                self.assertEqual(len(dims), 1, f"All items in a batch must share target_dim, got {batch_spec}")
                items = [ds_u8[spec] for spec in batch_spec]
                shapes = {tuple(it["image"].shape) for it in items}
                self.assertEqual(len(shapes), 1, f"Zero white-pad waste expected for equal-aspect pages, got {shapes}")
                self.assertEqual(items[0]["image"].dtype, torch.uint8)

            # AsyncThreadPrefetcher yields uint8 batches and shuts down cleanly
            pf = uni_omr.AsyncThreadPrefetcher(
                ds_u8, batch_size=2, shuffle=True, pin_memory=False, max_workers=2, prefetch_batches=2, seed=11
            )
            try:
                b = next(pf)
                self.assertEqual(b["images"].dtype, torch.uint8)
                self.assertEqual(b["images"].shape[0], 2)
                imgs_f = uni_omr._images_to_device_float(b["images"], torch.device("cpu"), non_blocking=False)
                self.assertEqual(imgs_f.dtype, torch.float32)
                self.assertGreaterEqual(float(imgs_f.min()), 0.0)
                self.assertLessEqual(float(imgs_f.max()), 1.0)
            finally:
                pf.close()

            # eval_dataset works seamlessly on return_uint8=True dataset
            ds_val = uni_omr.UniFullPageDataset(
                fp_pairs[:2], fullpage_max_dim=128, max_seq_len=32, augment=False, multi_scale=False, return_uint8=True
            )
            model = uni_omr.UniOMRNet(_small_cfg(dropout=0.0))
            metrics = uni_omr.eval_dataset(model, ds_val, device=torch.device("cpu"), max_len=8, batch_size=2, use_bf16=False)
            self.assertEqual(metrics["n_samples"], 2)
            self.assertGreater(metrics["tokens_per_sec"], 0.0)

    def test_r_vocab_expand_init_from_relocates_tail_specials(self) -> None:
        """(r) load_uni_weights_compat expands 182-vocab checkpoint into current VOCAB_SIZE & relocates tail 4 specials."""
        old_v = 182
        new_v = uni_omr.VOCAB_SIZE if uni_omr.VOCAB_SIZE > old_v else 220

        new_model = uni_omr.UniOMRNet(_small_cfg(vocab_size=new_v, d_model=64))
        old_sd = {k: v.clone() for k, v in new_model.state_dict().items()}

        # Build dummy 182-vocab checkpoint tensors with distinct deterministic sentinel values
        old_tok = torch.empty((old_v, 64), dtype=torch.float32)
        old_lm = torch.empty((old_v, 64), dtype=torch.float32)
        for row in range(old_v):
            old_tok[row].fill_(float(row + 1) * 0.01)
            old_lm[row].fill_(float(row + 1) * -0.02)
        old_sd["tok_emb.weight"] = old_tok
        old_sd["lm_head.weight"] = old_lm
        old_sd["backbone.proj.weight"] = torch.full_like(old_sd["backbone.proj.weight"], 0.31415)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            inc = uni_omr.load_uni_weights_compat(new_model, old_sd)

        self.assertEqual(len(inc.missing_keys), 0)
        self.assertEqual(len(inc.unexpected_keys), 0)
        self.assertIn(
            f"[UniOMR init-from] expanded vocab rows {old_v} -> {new_v} (tail 4 specials relocated)",
            buf.getvalue(),
        )

        # Non-vocab weights copied verbatim
        self.assertTrue(torch.equal(new_model.backbone.proj.weight, old_sd["backbone.proj.weight"]))

        # Base music tokens 0..177 copied bit-for-bit; tail 4 specials (178..181) relocated to new_v-4..new_v-1
        for key in ("tok_emb.weight", "lm_head.weight"):
            w_old = old_sd[key]
            w_new = new_model.state_dict()[key]
            self.assertTrue(
                torch.equal(w_new[: old_v - 4], w_old[: old_v - 4]),
                f"Base rows 0..{old_v - 5} mismatch for {key}",
            )
            self.assertTrue(
                torch.equal(w_new[new_v - 4 :], w_old[old_v - 4 :]),
                f"Tail 4 specials (PAD/BOS/EOS/<sys>) mismatch for {key}",
            )
            mid = w_new[old_v - 4 : new_v - 4]
            self.assertTrue(torch.isfinite(mid).all())
            self.assertLess(float(mid.abs().mean().item()), 0.1)

    def test_s_router_border_cliff_proof_and_faint_ink(self) -> None:
        """(s) Border-masked relative InkTokenRouter ignores dark borders (<5% border tokens) & keeps faint brown ink (>=0.85)."""
        cfg = _small_cfg(
            keep_ratio=0.45,
            dynamic_routing=True,
            min_keep_ratio=0.15,
            max_keep_ratio=0.45,
            saliency_thr=0.05,
            n_bg_summary=4,
            use_ink_prior=True,
            backbone_stride=16,
        )
        model = uni_omr.UniOMRNet(cfg)
        model.eval()

        # Synthetic 384x384 archival photo: 15% solid black border (~58px) + warm tan parchment + faint brown iron-gall ink
        H, W = 384, 384
        border = int(round(H * 0.15))  # 58px
        img = torch.zeros(1, 3, H, W, dtype=torch.float32)
        # Warm aged tan parchment interior
        img[:, 0, border : H - border, border : W - border] = 0.90
        img[:, 1, border : H - border, border : W - border] = 0.84
        img[:, 2, border : H - border, border : W - border] = 0.72

        # Faint brown iron-gall ink (~0.56, 0.48, 0.38) on 2 systems of thin 2px staves + noteheads
        staff_ys = [110, 120, 130, 140, 150, 220, 230, 240, 250, 260]
        x0, x1 = border + 24, W - border - 24
        for sy in staff_ys:
            img[:, 0, sy : sy + 2, x0:x1] = 0.56
            img[:, 1, sy : sy + 2, x0:x1] = 0.48
            img[:, 2, sy : sy + 2, x0:x1] = 0.38
        for nx in range(x0 + 16, x1 - 16, 28):
            img[:, 0, 124:132, nx : nx + 8] = 0.52
            img[:, 1, 124:132, nx : nx + 8] = 0.44
            img[:, 2, 124:132, nx : nx + 8] = 0.34
            img[:, 0, 234:242, nx : nx + 8] = 0.52
            img[:, 1, 234:242, nx : nx + 8] = 0.44
            img[:, 2, 234:242, nx : nx + 8] = 0.34

        # Add a single dark ink speck outlier to verify 99th-percentile normalization resists crushing faint ink
        img[:, :, 180:184, 180:184] = 0.0

        # Neutralize random saliency_net bias to test router ink prior cleanly (even under slight negative bias)
        with torch.no_grad():
            for m in model.router.saliency_net.modules():
                if isinstance(m, torch.nn.Conv2d) and m.bias is not None:
                    m.bias.fill_(-4.0)

        tgt_in = torch.tensor([[uni_omr.BOS_ID, 0, 1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            out = model(img, tgt_in, return_aux=False, return_mtp=False)
        rinfo = out["router_info"]
        H_p, W_p = rinfo["grid_hw"]

        # Compute fraction of kept tokens that fall strictly inside the dark border ring
        kept_idx = rinfo["kept_idx"][0].tolist()
        border_cells = border // 16  # 3 cells strictly inside 58px dark border (48px < 58px)
        n_border_kept = 0
        for idx in kept_idx:
            gy, gx = divmod(int(idx), W_p)
            if gy < border_cells or gy >= H_p - border_cells or gx < border_cells or gx >= W_p - border_cells:
                n_border_kept += 1
        border_frac = n_border_kept / max(1, len(kept_idx))

        self.assertLess(
            border_frac,
            0.05,
            f"Border token fraction must be < 5%, got {border_frac * 100.0:.2f}% ({n_border_kept}/{len(kept_idx)})",
        )
        self.assertGreaterEqual(
            rinfo["ink_recall"],
            0.85,
            f"Expected faint-ink recall >= 0.85 at max_keep_ratio=0.45, got {rinfo['ink_recall']:.4f}",
        )
        # Also verify ink_target assigns 0 to the dark exterior border
        ink_tgt_2d = rinfo["ink_target"][0, 0]
        self.assertEqual(float(ink_tgt_2d[:border_cells, :].max().item()), 0.0)

    def test_t_archival_hw_augment_parchment_and_bleed(self) -> None:
        """(t) apply_online_augment(..., is_hw=True) produces warm parchment (R > B), is fast & deterministic."""
        import random as _py_random
        import time as _time

        arr = np.full((384, 288, 3), 248, dtype=np.uint8)
        for sy in (110, 120, 130, 140, 150, 220, 230, 240, 250, 260):
            arr[sy : sy + 2, 36:252, :] = 20
        base = Image.fromarray(arr, mode="RGB")

        # Deterministic per RNG seed + exact size preservation
        aug1 = uni_omr.apply_online_augment(base, rng=_py_random.Random(777), is_hw=True, hw_archival_prob=1.0)
        aug2 = uni_omr.apply_online_augment(base, rng=_py_random.Random(777), is_hw=True, hw_archival_prob=1.0)
        self.assertEqual(aug1.size, base.size)
        np.testing.assert_array_equal(np.asarray(aug1), np.asarray(aug2))

        # Forced archival mode must produce warm parchment (mean interior R > B)
        a_arch = np.asarray(aug1, dtype=np.float32)
        interior = a_arch[64:-64, 48:-48]
        self.assertGreater(
            float(interior[..., 0].mean()),
            float(interior[..., 2].mean()) + 5.0,
            f"Expected warm parchment R > B, got R={interior[..., 0].mean():.1f}, B={interior[..., 2].mean():.1f}",
        )

        # Speed + stability across 10 seeds with default is_hw=True (p=0.55 archival)
        warm_hits = 0
        t0 = _time.perf_counter()
        for s in range(10):
            im = uni_omr.apply_online_augment(base, rng=_py_random.Random(900 + s), is_hw=True)
            self.assertEqual(im.size, base.size)
            a = np.asarray(im, dtype=np.float32)[64:-64, 48:-48]
            if float(a[..., 0].mean()) > float(a[..., 2].mean()) + 4.0:
                warm_hits += 1
        ms_per_img = (_time.perf_counter() - t0) * 1000.0 / 10.0
        self.assertGreater(warm_hits, 0, "Expected warm parchment archival outputs across 10 is_hw=True seeds")
        self.assertLess(ms_per_img, 100.0, f"apply_online_augment too slow: {ms_per_img:.1f} ms/img")

        # Clean-page mild scale jitter preserves size and stays white-padded (no dark border)
        scaled = uni_omr.apply_clean_scale_jitter(base, rng=_py_random.Random(42))
        self.assertEqual(scaled.size, base.size)
        self.assertGreater(float(np.asarray(scaled)[0, 0].mean()), 230.0)

    def test_u_composer_scratchouts_deterministic_and_fast(self) -> None:
        """(u) _apply_composer_scratchouts preserves (W,H)/RGB, is deterministic, modifies pixels, and runs <20 ms at 1280x960."""
        import random as _py_random
        import time as _time

        arr = np.full((960, 1280, 3), 244, dtype=np.uint8)
        for sy in range(140, 820, 120):
            for k in range(5):
                arr[sy + k * 8 : sy + k * 8 + 2, 90:1190, :] = 25
        base = Image.fromarray(arr, mode="RGB")

        for is_hw in (True, False):
            out_a = uni_omr._apply_composer_scratchouts(base, _py_random.Random(1337), is_hw=is_hw)
            out_b = uni_omr._apply_composer_scratchouts(base, _py_random.Random(1337), is_hw=is_hw)
            out_c = uni_omr._apply_composer_scratchouts(base, _py_random.Random(2027), is_hw=is_hw)

            self.assertEqual(out_a.size, (1280, 960))
            self.assertEqual(out_a.mode, "RGB")
            arr_a = np.asarray(out_a)
            np.testing.assert_array_equal(arr_a, np.asarray(out_b))
            self.assertFalse(np.array_equal(arr_a, arr), f"Expected pixel changes for is_hw={is_hw}")
            self.assertFalse(np.array_equal(arr_a, np.asarray(out_c)), "Expected different seed to produce different output")
            # Keep scratch-outs compact: changed pixel fraction must remain small (<5% of page)
            changed_frac = float((arr_a != arr).any(axis=-1).mean())
            self.assertLess(changed_frac, 0.05, f"Scratch-outs obscured too much of page: {changed_frac:.4f}")

        # Speed check at 1280x960 (<20 ms/img)
        n_runs = 10
        t0 = _time.perf_counter()
        for s in range(n_runs):
            im = uni_omr._apply_composer_scratchouts(base, _py_random.Random(500 + s), is_hw=(s % 2 == 0))
            self.assertEqual(im.size, (1280, 960))
        ms_per_img = (_time.perf_counter() - t0) * 1000.0 / n_runs
        self.assertLess(ms_per_img, 20.0, f"_apply_composer_scratchouts too slow: {ms_per_img:.2f} ms/img")

    def test_v_performer_pencil_markup_deterministic_and_fast(self) -> None:
        """(v) _apply_performer_pencil_markup preserves (W,H)/RGB, is deterministic, modifies pixels, and runs <20 ms at 1280x960."""
        import random as _py_random
        import time as _time

        arr = np.full((960, 1280, 3), 250, dtype=np.uint8)
        for sy in range(140, 820, 120):
            for k in range(5):
                arr[sy + k * 8 : sy + k * 8 + 2, 90:1190, :] = 15
        base = Image.fromarray(arr, mode="RGB")

        out_a = uni_omr._apply_performer_pencil_markup(base, _py_random.Random(4242))
        out_b = uni_omr._apply_performer_pencil_markup(base, _py_random.Random(4242))
        out_c = uni_omr._apply_performer_pencil_markup(base, _py_random.Random(9001))

        self.assertEqual(out_a.size, (1280, 960))
        self.assertEqual(out_a.mode, "RGB")
        arr_a = np.asarray(out_a)
        np.testing.assert_array_equal(arr_a, np.asarray(out_b))
        self.assertFalse(np.array_equal(arr_a, arr), "Expected performer markup to modify pixels")
        self.assertFalse(np.array_equal(arr_a, np.asarray(out_c)), "Expected different seed to produce different output")

        # Speed check at 1280x960 (<20 ms/img)
        n_runs = 10
        t0 = _time.perf_counter()
        for s in range(n_runs):
            im = uni_omr._apply_performer_pencil_markup(base, _py_random.Random(700 + s))
            self.assertEqual(im.size, (1280, 960))
        ms_per_img = (_time.perf_counter() - t0) * 1000.0 / n_runs
        self.assertLess(ms_per_img, 20.0, f"_apply_performer_pencil_markup too slow: {ms_per_img:.2f} ms/img")

    def test_w_struct_tok_weight_loss_and_cli(self) -> None:
        """(w) P0: struct_tok_weight upweights header/structural tokens in CE loss, caches weights, and parses CLI."""
        model = uni_omr.UniOMRNet(_small_cfg(d_model=64, n_enc_layers=1, n_dec_layers=1, n_heads=4, mtp_k=1))
        model.eval()

        # Verify weight tensor contents and caching
        w3 = model._get_struct_token_weights(uni_omr.VOCAB_SIZE, 3.0, torch.device("cpu"), torch.float32)
        w3_cached = model._get_struct_token_weights(uni_omr.VOCAB_SIZE, 3.0, torch.device("cpu"), torch.float32)
        self.assertIs(w3, w3_cached)
        self.assertEqual(w3.shape, (uni_omr.VOCAB_SIZE,))
        for tok in ("clef-G-2", "key+0", "time-4-4", "<sys>", "grace", "bar", "bar-light-heavy"):
            self.assertAlmostEqual(float(w3[uni_omr.UNI_TOKEN_TO_ID[tok]].item()), 3.0, places=5)
        for tok in ("quarter", "C", "4", "dot", "rest"):
            self.assertAlmostEqual(float(w3[uni_omr.UNI_TOKEN_TO_ID[tok]].item()), 1.0, places=5)

        # Verify weighted CE shifts loss toward higher-error structural tokens vs unweighted
        V = uni_omr.VOCAB_SIZE
        clef_id = uni_omr.UNI_TOKEN_TO_ID["clef-G-2"]
        note_id = uni_omr.UNI_TOKEN_TO_ID["quarter"]
        logits = torch.zeros(1, 2, V, dtype=torch.float32)
        # High error on clef_id (pos 0), low error on note_id (pos 1)
        logits[0, 0, clef_id] = -2.0
        logits[0, 1, note_id] = 4.0
        targets = torch.tensor([[clef_id, note_id]], dtype=torch.long)
        out = {"logits": logits, "mtp_logits": [], "uv_pred": None, "router_info": {}}

        l1 = model.compute_loss(out, targets, struct_tok_weight=1.0)
        l3 = model.compute_loss(out, targets, struct_tok_weight=3.0)
        ref_ce = torch.nn.functional.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=uni_omr.PAD_ID)
        self.assertAlmostEqual(float(l1["ce_loss"].item()), float(ref_ce.item()), places=5)
        self.assertGreater(float(l3["ce_loss"].item()), float(l1["ce_loss"].item()))

        # CLI parsing default=1.0 and custom value
        ap = uni_omr.build_arg_parser()
        self.assertAlmostEqual(ap.parse_args([]).struct_tok_weight, 1.0)
        self.assertAlmostEqual(ap.parse_args(["--struct-tok-weight", "2.5"]).struct_tok_weight, 2.5)

    def test_x_router_dark_border_paper_gate_and_bce_target(self) -> None:
        """(x) P4a: paper_gate suppresses high-firing saliency_net on dark borders & ink_target=0 trains it down."""
        cfg = _small_cfg(
            keep_ratio=0.40,
            dynamic_routing=True,
            min_keep_ratio=0.15,
            max_keep_ratio=0.45,
            saliency_thr=0.05,
            n_bg_summary=4,
            use_ink_prior=True,
            backbone_stride=16,
        )
        model = uni_omr.UniOMRNet(cfg)
        model.eval()

        H, W = 384, 384
        border = int(round(H * 0.15))  # 58px dark scanner border
        img = torch.zeros(1, 3, H, W, dtype=torch.float32)
        img[:, 0, border : H - border, border : W - border] = 0.90
        img[:, 1, border : H - border, border : W - border] = 0.84
        img[:, 2, border : H - border, border : W - border] = 0.72
        x0, x1 = border + 24, W - border - 24
        for sy in (110, 120, 130, 140, 150, 220, 230, 240, 250, 260):
            img[:, :, sy : sy + 2, x0:x1] = 0.10

        # Force saliency_net to fire strongly (~0.88) everywhere including dark borders
        with torch.no_grad():
            for m in model.router.saliency_net.modules():
                if isinstance(m, torch.nn.Conv2d) and m.bias is not None:
                    m.bias.fill_(2.0)

        tgt_in = torch.tensor([[uni_omr.BOS_ID, 0, 1, 2, 3]], dtype=torch.long)
        tgt_out = torch.tensor([[0, 1, 2, 3, uni_omr.EOS_ID]], dtype=torch.long)
        out = model(img, tgt_in, return_aux=False, return_mtp=False)
        rinfo = out["router_info"]
        H_p, W_p = rinfo["grid_hw"]
        border_cells = border // 16  # 3 cells strictly inside 58px border

        score_2d = rinfo["saliency_map"][0, 0]
        border_mean = float(score_2d[:border_cells, :].mean().item())
        self.assertLess(
            border_mean,
            0.10,
            f"Expected paper_gate to crush dark-border score < 0.10 even with +2.0 bias, got {border_mean:.4f}",
        )

        # Kept border token fraction must stay < 5% and ink_recall >= 0.85
        kept_idx = rinfo["kept_idx"][0].tolist()
        n_border_kept = sum(
            1
            for idx in kept_idx
            for gy, gx in [divmod(int(idx), W_p)]
            if gy < border_cells or gy >= H_p - border_cells or gx < border_cells or gx >= W_p - border_cells
        )
        border_frac = n_border_kept / max(1, len(kept_idx))
        self.assertLess(border_frac, 0.05, f"Dark border tokens leaked: {border_frac * 100:.2f}%")
        self.assertGreaterEqual(rinfo["ink_recall"], 0.85)

        # ink_target must be strictly 0.0 on dark borders so BCE actively trains saliency_net to 0 there
        ink_tgt_2d = rinfo["ink_target"][0, 0]
        self.assertEqual(float(ink_tgt_2d[:border_cells, :].max().item()), 0.0)
        losses = model.compute_loss(out, tgt_out, saliency_loss_weight=0.10)
        self.assertGreater(float(losses["saliency_loss"].item()), 0.0)

    def test_y_archival_clutter_aug_faint_verse_watermark_fast(self) -> None:
        """(y) P4b: archival verse block, watermark/stain, and faint sepia ink are deterministic and fast."""
        import random as _py_random
        import time as _time

        arr = np.full((960, 1280, 3), 236, dtype=np.uint8)
        for sy in range(180, 720, 140):
            for k in range(5):
                arr[sy + k * 8 : sy + k * 8 + 2, 100:1180, :] = 35
        base = Image.fromarray(arr, mode="RGB")
        ink_u8 = (95, 78, 58)

        # 1. Verse block & watermark/stain determinism, RGB/size preservation, pixel change, speed (<5ms)
        for fn in (uni_omr._apply_archival_verse_block, uni_omr._apply_archival_watermark_stain):
            a = fn(base, _py_random.Random(101), ink_u8)
            b = fn(base, _py_random.Random(101), ink_u8)
            self.assertEqual(a.size, (1280, 960))
            self.assertEqual(a.mode, "RGB")
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            self.assertFalse(np.array_equal(np.asarray(a), arr))

            t0 = _time.perf_counter()
            for s in range(10):
                _ = fn(base, _py_random.Random(200 + s), ink_u8)
            ms = (_time.perf_counter() - t0) * 1000.0 / 10.0
            self.assertLess(ms, 10.0, f"{fn.__name__} too slow: {ms:.2f} ms/img")

        # 2. Integrated archival mode produces faint low-contrast sepia variants among 30 seeds
        small = Image.fromarray(arr[::3, ::4], mode="RGB")
        saw_faint = False
        for s in range(30):
            im = uni_omr.apply_online_augment(
                small, rng=_py_random.Random(3000 + s), is_hw=True, hw_archival_prob=1.0
            )
            self.assertEqual(im.size, small.size)
            interior = np.asarray(im, dtype=np.float32)[40:-40, 40:-40]
            # Faint sepia ink compresses dark ink brightness upward (min luma > 95)
            luma_min = float(interior.mean(axis=-1).min())
            if luma_min > 95.0:
                saw_faint = True
                break
        self.assertTrue(saw_faint, "Expected faint low-contrast sepia ink variant across 30 archival seeds")

    def test_z_tier1_algebraic_bg_pooling_and_channels_last(self) -> None:
        """(z) Tier-1 #1/#2: Algebraic BG pooling matches naive O(N*D^2) formula (<1e-5) & channels_last parity."""
        torch.manual_seed(1234)
        router = uni_omr.InkTokenRouter(d_model=64, keep_ratio=0.40, n_bg_summary=4, use_ink_prior=True)
        router.eval()

        B, D, H_p, W_p = 2, 64, 8, 12
        feat_2d = torch.randn(B, D, H_p, W_p)
        images = torch.rand(B, 3, H_p * 16, W_p * 16)

        with torch.no_grad():
            out_tokens, out_coords, rinfo = router(feat_2d, images=images)
            k = int(rinfo["k_kept"])
            fast_bg = out_tokens[:, k:, :]  # (B, M, D)

            # Reference naive O(N * D^2) formula
            N = H_p * W_p
            tokens_flat = feat_2d.flatten(2).transpose(1, 2)
            score_flat = rinfo["saliency_map"].flatten(1)
            kept_mask = torch.zeros(B, N, dtype=torch.bool)
            kept_mask.scatter_(1, rinfo["kept_idx"], True)

            bg_q = router.bg_queries.expand(B, -1, -1)
            bg_k = router.bg_key_proj(tokens_flat)
            bg_v = router.bg_val_proj(tokens_flat)
            bg_logits = torch.bmm(bg_q, bg_k.transpose(1, 2)) / (D ** 0.5)
            bg_logits = bg_logits - score_flat.unsqueeze(1)
            bg_logits = bg_logits.masked_fill(kept_mask.unsqueeze(1), -1e4)
            bg_attn = torch.nn.functional.softmax(bg_logits, dim=-1)
            naive_bg = router.bg_out_norm(torch.bmm(bg_attn, bg_v))

        max_diff = float((fast_bg - naive_bg).abs().max().item())
        self.assertLess(max_diff, 1e-5, f"Algebraic BG pooling drift too large: {max_diff}")

        # channels_last on DynamicConvBackbone: zero state_dict changes & exact/close logit parity
        model_nchw = uni_omr.UniOMRNet(_small_cfg(d_model=64, dropout=0.0))
        model_nhwc = uni_omr.UniOMRNet(_small_cfg(d_model=64, dropout=0.0))
        model_nhwc.load_state_dict(model_nchw.state_dict(), strict=True)
        model_nhwc.backbone.to(memory_format=torch.channels_last)
        model_nchw.eval()
        model_nhwc.eval()

        self.assertEqual(set(model_nchw.state_dict().keys()), set(model_nhwc.state_dict().keys()))
        for key in model_nchw.state_dict():
            self.assertEqual(model_nchw.state_dict()[key].shape, model_nhwc.state_dict()[key].shape)

        tgt = torch.tensor([[uni_omr.BOS_ID, 0, 1, 2, 3], [uni_omr.BOS_ID, 4, 5, 6, 7]], dtype=torch.long)
        with torch.no_grad():
            o1 = model_nchw(images, tgt, return_aux=False, return_mtp=False)["logits"]
            o2 = model_nhwc(images, tgt, return_aux=False, return_mtp=False)["logits"]
        self.assertLess(float((o1 - o2).abs().max().item()), 1e-4)

    def test_za_tier1_apply_tok_dropout_false_deterministic(self) -> None:
        """(za) Tier-1 #5: apply_tok_dropout=False yields deterministic identical logits when tok_dropout > 0."""
        cfg = _small_cfg(d_model=64, dropout=0.0, tok_dropout=0.50)
        model = uni_omr.UniOMRNet(cfg)
        model.train()

        mem = torch.randn(2, 24, 64)
        tgt_in = torch.tensor(
            [[uni_omr.BOS_ID, 1, 2, 3, 4, 5, 6, 7], [uni_omr.BOS_ID, 8, 9, 10, 11, 12, 13, 14]],
            dtype=torch.long,
        )

        with torch.no_grad():
            logits_a, _ = model.decode_tokens(tgt_in, mem, return_mtp=False, apply_tok_dropout=False)
            logits_b, _ = model.decode_tokens(tgt_in, mem, return_mtp=False, apply_tok_dropout=False)
            self.assertTrue(
                torch.equal(logits_a, logits_b),
                "apply_tok_dropout=False must produce identical logits across repeated calls",
            )

            # With apply_tok_dropout=True and tok_dropout=0.50, repeated calls should differ
            logits_drop1, _ = model.decode_tokens(tgt_in, mem, return_mtp=False, apply_tok_dropout=True)
            logits_drop2, _ = model.decode_tokens(tgt_in, mem, return_mtp=False, apply_tok_dropout=True)
            self.assertFalse(
                torch.equal(logits_drop1, logits_drop2),
                "Expected apply_tok_dropout=True with tok_dropout=0.50 to perturb inputs across calls",
            )

    def test_zb_tier1_orient_buckets_strip_bs_log_every_and_gpu_ema(self) -> None:
        """(zb) Tier-1 #4/#6/#7: Orientation-bucketed B=2 prefetch, --strip-batch-size, --log-every, --gpu-ema-decay."""
        import json

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            pr_dir = td_path / "printed"
            fp_root = td_path / "fullpage"
            fp_pages = fp_root / "pages"
            fp_strips = fp_root / "stage2_strips"
            out_dir = td_path / "out"
            for d in (pr_dir, fp_pages, fp_strips):
                d.mkdir(parents=True)

            bek_short = "clef-G-2\nquarter\nC\n4\nbar\n"
            bek_long = "clef-G-2\nquarter\nC\n4\nquarter\nD\n4\nquarter\nE\n4\nquarter\nF\n4\nbar\n"
            # 6 strips
            for i in range(6):
                arr_s = np.full((64, 128, 3), 245, dtype=np.uint8)
                arr_s[20:44, 15:110, :] = 30
                Image.fromarray(arr_s, mode="RGB").save(pr_dir / f"st{i}_p1_s0.png")
                (pr_dir / f"st{i}_p1_s0-bekern.txt").write_text(
                    bek_short if (i % 2 == 0) else bek_long, encoding="utf-8"
                )

            # 4 portrait (H=128, W=80) + 4 landscape (H=80, W=128) full pages
            for i in range(4):
                arr_p = np.full((128, 80, 3), 245, dtype=np.uint8)
                arr_p[20:100, 10:70, :] = 25
                Image.fromarray(arr_p, mode="RGB").save(fp_pages / f"port{i}_p1.png")
                (fp_strips / f"port{i}_p1_s0-bekern.txt").write_text(
                    bek_short if (i % 2 == 0) else bek_long, encoding="utf-8"
                )

                arr_l = np.full((80, 128, 3), 245, dtype=np.uint8)
                arr_l[10:70, 20:100, :] = 25
                Image.fromarray(arr_l, mode="RGB").save(fp_pages / f"land{i}_p1.png")
                (fp_strips / f"land{i}_p1_s0-bekern.txt").write_text(
                    bek_short if (i % 2 == 0) else bek_long, encoding="utf-8"
                )

            # 1. Verify AsyncThreadPrefetcher orientation bucketing never mixes portrait + landscape
            fp_pairs = uni_omr.discover_fullpage_pairs([fp_root])
            fp_ds = uni_omr.UniFullPageDataset(
                fp_pairs, fullpage_max_dim=128, max_seq_len=32, augment=False, return_uint8=True
            )
            pf = uni_omr.AsyncThreadPrefetcher(
                fp_ds, batch_size=2, shuffle=True, pin_memory=False, max_workers=2, prefetch_batches=2, seed=17
            )
            try:
                for _ in range(8):
                    b = next(pf)
                    _, _, bh, bw = b["images"].shape
                    # Pure portrait is 128x80, pure landscape is 80x128; mixed would pad to 128x128 square!
                    self.assertNotEqual(
                        (bh, bw),
                        (128, 128),
                        f"Orientation bucketing failed: got mixed square batch {(bh, bw)} for {b['sample_ids']}",
                    )
            finally:
                pf.close()

            # 2. End-to-end CLI smoke test with --channels-last, --strip-batch-size, --log-every, --gpu-ema-decay
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = uni_omr.main([
                    "--mode", "train",
                    "--curriculum", "mixed",
                    "--strip-dirs", str(pr_dir),
                    "--fullpage-dirs", str(fp_root),
                    "--channels-last",
                    "--strip-batch-size", "4",
                    "--batch-size", "2",
                    "--grad-accum", "2",
                    "--log-every", "3",
                    "--gpu-ema-decay", "0.997",
                    "--cass-prob", "0.25",
                    "--tok-dropout", "0.10",
                    "--grad-checkpoint",
                    "--strip-height", "64",
                    "--max-strip-width", "128",
                    "--fullpage-max-dim", "128",
                    "--max-seq-len", "32",
                    "--d-model", "64",
                    "--n-enc-layers", "1",
                    "--n-dec-layers", "1",
                    "--n-heads", "4",
                    "--steps", "6",
                    "--warmup-steps", "1",
                    "--val-count", "2",
                    "--eval-every", "6",
                    "--eval-batch-size", "2",
                    "--eval-max-len", "8",
                    "--device", "cpu",
                    "--out-dir", str(out_dir),
                ])
            self.assertEqual(rc, 0)
            out_txt = buf.getvalue()
            # --log-every 3 on 6 steps must print step 1, step 3, step 6 (and NOT step 2, 4, 5)
            self.assertIn("step 1/6", out_txt)
            self.assertIn("step 3/6", out_txt)
            self.assertIn("step 6/6", out_txt)
            self.assertNotIn("step 2/6", out_txt)
            self.assertNotIn("step 4/6", out_txt)
            self.assertNotIn("step 5/6", out_txt)

            # GPU EMA shadow evaluated alongside student and saved uni_omr_ema_best.pt
            self.assertIn("6 EMA", out_txt)
            self.assertTrue((out_dir / "uni_omr_ema_best.pt").is_file())
            ema_ck = torch.load(str(out_dir / "uni_omr_ema_best.pt"), map_location="cpu", weights_only=True)
            self.assertIn("model", ema_ck)
            self.assertAlmostEqual(float(ema_ck["ema_decay"]), 0.997, places=5)

            eval_sum = json.loads((out_dir / "eval_summary.json").read_text(encoding="utf-8"))
            self.assertIn("best_ema_ser", eval_sum)
            self.assertIn("ema_eval", eval_sum["evals"][0])

    def test_zc_iter9_router_erosion_micro_weights_and_augments(self) -> None:
        """(zc) Iter 9: 2-patch router border erosion, micro-rhythm/double-accidental weights, and new HW/printed augs."""
        import random as _py_random

        # 1. Router 2-patch dark-desk halo erosion + all-dark fallback
        H, W = 512, 512
        border = 64  # 4 cells at stride 16
        img = torch.zeros(1, 3, H, W, dtype=torch.float32)
        # Warm parchment interior + faint HW ink in center
        img[:, 0, border : H - border, border : W - border] = 0.88
        img[:, 1, border : H - border, border : W - border] = 0.82
        img[:, 2, border : H - border, border : W - border] = 0.70
        for sy in (180, 190, 200, 210, 220, 280, 290, 300, 310, 320):
            img[:, :, sy : sy + 2, 130:382] = 0.45

        local_ink_grid, paper_gate, eff_paper = uni_omr.InkTokenRouter._compute_ink_and_paper_gate(img, 32, 32)
        # Transition ring (cells 4 and 5 adjacent to border cells 0..3) must be zeroed & gated to 0.05
        self.assertEqual(float(local_ink_grid[0, 0, :6, :].max().item()), 0.0)
        self.assertEqual(float(local_ink_grid[0, 0, -6:, :].max().item()), 0.0)
        self.assertEqual(float(local_ink_grid[0, 0, :, :6].max().item()), 0.0)
        self.assertEqual(float(local_ink_grid[0, 0, :, -6:].max().item()), 0.0)
        self.assertAlmostEqual(float(paper_gate[0, 0, 4, 16].item()), 0.05, places=5)
        self.assertAlmostEqual(float(paper_gate[0, 0, 5, 16].item()), 0.05, places=5)
        # Interior faint ink must remain strongly active
        self.assertTrue(bool(eff_paper[0, 0, 12, 16].item()))
        self.assertGreater(float(local_ink_grid[0, 0, 11:15, 10:22].max().item()), 0.05)

        # Fallback when entire image is dark (no interior paper after erosion): stays finite & non-crashing
        dark_img = torch.full((1, 3, 128, 128), 0.10, dtype=torch.float32)
        li_d, pg_d, ep_d = uni_omr.InkTokenRouter._compute_ink_and_paper_gate(dark_img, 8, 8)
        self.assertTrue(torch.isfinite(li_d).all())
        self.assertTrue(torch.isfinite(pg_d).all())
        self.assertFalse(bool(ep_d.any().item()))

        # 2. Micro-rhythm & double-accidental CE weights = 0.5 * (1 + struct_tok_weight)
        model = uni_omr.UniOMRNet(_small_cfg(d_model=64, n_enc_layers=1, n_dec_layers=1, n_heads=4, mtp_k=1))
        w2 = model._get_struct_token_weights(uni_omr.VOCAB_SIZE, 2.0, torch.device("cpu"), torch.float32)
        for tok in ("tuplet-start-3-2", "tuplet-stop", "32nd", "64th", "a2", "a-2"):
            self.assertAlmostEqual(float(w2[uni_omr.UNI_TOKEN_TO_ID[tok]].item()), 1.5, places=5)
        for tok in ("clef-G-2", "key+0", "time-4-4", "<sys>", "grace", "bar"):
            self.assertAlmostEqual(float(w2[uni_omr.UNI_TOKEN_TO_ID[tok]].item()), 2.0, places=5)
        for tok in ("quarter", "C", "4", "a1", "a-1"):
            self.assertAlmostEqual(float(w2[uni_omr.UNI_TOKEN_TO_ID[tok]].item()), 1.0, places=5)

        # 3. Inter-system empty pre-ruled staves
        luma = np.ones((400, 300), dtype=np.float32)
        # Put dark ink everywhere except the single inter-system quiet band y=135..255
        luma[0:135, 20:280] = 0.1
        luma[255:400, 20:280] = 0.1
        uni_omr._add_archival_empty_staves(luma, _py_random.Random(123))
        # Quiet inter-system band 135..255 must now contain faint pre-ruled staff lines (< 0.90)
        self.assertLess(float(luma[135:255, 50:250].min()), 0.90)

        # 4. System-1 left indentation, corner sticky-note, and HW low-res defocus/compress
        arr = np.full((384, 288, 3), 248, dtype=np.uint8)
        # System 1 at y=40..90, System 2 at y=200..250
        arr[40:90:10, 16:272, :] = 20
        arr[200:250:10, 16:272, :] = 20
        base = Image.fromarray(arr, mode="RGB")

        ind1 = uni_omr._apply_system1_left_indent(base, _py_random.Random(77))
        ind2 = uni_omr._apply_system1_left_indent(base, _py_random.Random(77))
        self.assertEqual(ind1.size, base.size)
        np.testing.assert_array_equal(np.asarray(ind1), np.asarray(ind2))
        ind_arr = np.asarray(ind1)
        # Leftmost 12% of System 1 must now be clean paper (> 235), while System 2 remains untouched
        self.assertGreater(float(ind_arr[40:90, 16:34].mean()), 235.0)
        np.testing.assert_array_equal(ind_arr[200:250], arr[200:250])

        sn1 = uni_omr._apply_corner_sticky_note(base, _py_random.Random(88))
        sn2 = uni_omr._apply_corner_sticky_note(base, _py_random.Random(88))
        self.assertEqual(sn1.size, base.size)
        np.testing.assert_array_equal(np.asarray(sn1), np.asarray(sn2))
        self.assertFalse(np.array_equal(np.asarray(sn1), arr))

        lr1 = uni_omr._apply_hw_lowres_defocus_compress(base, _py_random.Random(99))
        lr2 = uni_omr._apply_hw_lowres_defocus_compress(base, _py_random.Random(99))
        self.assertEqual(lr1.size, base.size)
        np.testing.assert_array_equal(np.asarray(lr1), np.asarray(lr2))
        # Contrast compression lifts darkest ink pixels upward from 20
        self.assertGreater(float(np.asarray(lr1).min()), 30.0)

    def test_zd_qat_int4_fake_quant_ste_and_padding(self) -> None:
        """(zd) fake_quant_int4_ste exact symmetric INT4 rounding, STE grad==1 passthrough, and padding."""
        # 1. Exact rounding on a known 2-group tensor (block_size=4)
        # Group 0: max abs = 7.0 -> scale = 1.0 -> exact integer rounding in [-8, 7]
        # Group 1: max abs = 0.7 -> scale = 0.1
        w = torch.tensor(
            [[7.0, -6.4, 3.2, 0.0, 0.7, -0.36, 0.14, -0.04]],
            dtype=torch.float32,
            requires_grad=True,
        )
        w_q = uni_omr.fake_quant_int4_ste(w, block_size=4)
        expected = torch.tensor(
            [[7.0, -6.0, 3.0, 0.0, 0.7, -0.4, 0.1, 0.0]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(w_q, expected, atol=1e-6), f"Got {w_q} vs {expected}")

        # Exact STE gradient passthrough (d(sum(w_q))/dw == 1 everywhere)
        w_q.sum().backward()
        self.assertIsNotNone(w.grad)
        self.assertTrue(torch.equal(w.grad, torch.ones_like(w)))

        # 2. Non-multiple-of-32 in_features padding (shape 5 x 50, block_size=32)
        torch.manual_seed(77)
        w_pad = torch.randn(5, 50, dtype=torch.float32, requires_grad=True)
        w_pad_q = uni_omr.fake_quant_int4_ste(w_pad, block_size=32)
        self.assertEqual(w_pad_q.shape, (5, 50))
        self.assertTrue(torch.isfinite(w_pad_q).all())
        self.assertFalse(torch.equal(w_pad_q, w_pad))
        # First 32 columns must match independent 32-col quantization
        w_head_q = uni_omr.fake_quant_int4_ste(w_pad[:, :32].detach(), block_size=32)
        self.assertTrue(torch.allclose(w_pad_q[:, :32], w_head_q, atol=1e-6))
        # STE gradient passthrough with non-uniform upstream weights
        upstream = torch.randn_like(w_pad)
        (w_pad_q * upstream).sum().backward()
        self.assertTrue(torch.allclose(w_pad.grad, upstream, atol=1e-6))

        # 3. All-zero tensor stays finite (scale clamped to 1e-8)
        w_zero = torch.zeros(2, 32, dtype=torch.float32, requires_grad=True)
        w_zero_q = uni_omr.fake_quant_int4_ste(w_zero, block_size=32)
        self.assertTrue(torch.isfinite(w_zero_q).all())
        self.assertEqual(float(w_zero_q.abs().max().item()), 0.0)

    def test_ze_qat_int4_controller_targeting_compat_and_cli(self) -> None:
        """(ze) QATInt4Controller targets only Transformer Linears, keeps strict=True state_dict, and runs in CLI."""
        cfg = _small_cfg(d_model=64, n_enc_layers=2, n_dec_layers=2, n_heads=4, mtp_k=4, dropout=0.0)
        model = uni_omr.UniOMRNet(cfg)
        sd_before = {k: v.detach().clone() for k, v in model.state_dict().items()}

        img = torch.rand(2, 3, 128, 192)
        tgt_in = torch.randint(0, len(data.VOCAB), (2, 10))
        tgt_out = torch.randint(0, len(data.VOCAB), (2, 10))

        model.eval()
        with torch.no_grad():
            logits_ref = model(img, tgt_in, return_aux=False, return_mtp=False)["logits"].clone()

        ctrl = uni_omr.QATInt4Controller(model, block_size=32, enabled=False)
        # Expected count: 2*6 (enc) + 2*10 (dec) + 3 (mtp_heads) + 1 (lm_head) = 36
        self.assertEqual(ctrl.num_layers, 36)
        allowed_prefixes = ("enc_layers.", "dec_layers.", "mtp_heads.", "lm_head")
        for name in ctrl.target_names:
            self.assertTrue(
                name == "lm_head" or name.startswith(allowed_prefixes[:3]),
                f"Unexpected QAT target: {name}",
            )
        # Ensure non-target modules are never patched
        for forbidden in ("backbone", "router", "tok_emb", "pos2d_proj", "uv_head", "enc_norm", "dec_norm"):
            self.assertFalse(
                any(n == forbidden or n.startswith(f"{forbidden}.") for n in ctrl.target_names),
                f"Forbidden module {forbidden} was patched by QATInt4Controller",
            )

        # Disabled QAT matches reference bit-identically
        with torch.no_grad():
            logits_disabled = model(img, tgt_in, return_aux=False, return_mtp=False)["logits"]
        self.assertTrue(torch.equal(logits_ref, logits_disabled))

        # Enabled QAT alters logits via INT4 weight quantization while keeping state_dict 100% identical
        ctrl.enable()
        sd_after = model.state_dict()
        self.assertEqual(list(sd_before.keys()), list(sd_after.keys()))
        for k in sd_before:
            self.assertTrue(torch.equal(sd_before[k], sd_after[k]), f"state_dict tensor changed for {k}")

        fresh = uni_omr.UniOMRNet(cfg)
        inc = fresh.load_state_dict(sd_after, strict=True)
        self.assertEqual(len(inc.missing_keys), 0)
        self.assertEqual(len(inc.unexpected_keys), 0)

        with torch.no_grad():
            logits_qat = model(img, tgt_in, return_aux=False, return_mtp=False)["logits"]
        self.assertTrue(torch.isfinite(logits_qat).all())
        self.assertFalse(torch.allclose(logits_ref, logits_qat, atol=1e-6))

        # Training step with QAT active produces finite gradients on all parameters
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.zero_grad()
        out = model(img, tgt_in, return_aux=False, return_mtp=True)
        losses = model.compute_loss(out, tgt_out, mtp_loss_weight=0.30, uv_loss_weight=0.0)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        for name, p in model.named_parameters():
            if p.requires_grad and not name.startswith("uv_head."):
                self.assertIsNotNone(p.grad, f"Missing grad under QAT for {name}")
                self.assertTrue(torch.isfinite(p.grad).all(), f"Non-finite grad under QAT for {name}")
        opt.step()

        # Detach restores original forwards cleanly
        ctrl.detach()
        self.assertEqual(ctrl.num_layers, 0)

        # CLI smoke test: --qat-int4 --qat-int4-last-frac 0.5 --qat-int4-block-size 32
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = uni_omr.main([
                "--mode", "train",
                "--curriculum", "strip",
                "--d-model", "64",
                "--n-enc-layers", "1",
                "--n-dec-layers", "1",
                "--n-heads", "4",
                "--mtp-k", "2",
                "--steps", "2",
                "--seq-len", "16",
                "--qat-int4",
                "--qat-int4-last-frac", "0.5",
                "--qat-int4-block-size", "32",
                "--device", "cpu",
                "--dry-run",
            ])
        self.assertEqual(rc, 0)
        out_txt = buf.getvalue()
        self.assertIn("[UniOMR QAT-INT4] enabled on 18 Linear layers (block_size=32) at step 1/2", out_txt)

    def test_zf_qat_int4_cached_delta_bit_exact_ste_parity(self) -> None:
        """(zf) Iter-10: QATInt4Controller cached deltas match uncached fake_quant_int4_ste bit-identically (fwd + STE grad)."""
        cfg = _small_cfg(d_model=64, n_enc_layers=1, n_dec_layers=1, n_heads=4, mtp_k=2, dropout=0.0)
        torch.manual_seed(2026)
        m_cached = uni_omr.UniOMRNet(cfg)
        m_ref = uni_omr.UniOMRNet(cfg)
        m_ref.load_state_dict(m_cached.state_dict(), strict=True)

        # Reference: uncached fake_quant_int4_ste on every target Linear
        ctrl_cached = uni_omr.QATInt4Controller(m_cached, block_size=32, enabled=True)
        for root_name in uni_omr.QATInt4Controller.TARGET_ROOTS:
            root_mod = getattr(m_ref, root_name, None)
            if root_mod is None:
                continue
            for mod in root_mod.modules():
                if isinstance(mod, torch.nn.Linear):
                    def _uncached_fwd(x: torch.Tensor, _m=mod) -> torch.Tensor:
                        w_q = uni_omr.fake_quant_int4_ste(_m.weight, block_size=32)
                        return torch.nn.functional.linear(x, w_q, _m.bias)
                    mod.forward = _uncached_fwd  # type: ignore[method-assign]

        img = torch.rand(2, 3, 128, 192)
        tgt_in = torch.randint(0, len(data.VOCAB), (2, 10))
        tgt_out = torch.randint(0, len(data.VOCAB), (2, 10))

        m_cached.train()
        m_ref.train()
        out_c = m_cached(img, tgt_in, return_aux=False, return_mtp=True)
        out_r = m_ref(img, tgt_in, return_aux=False, return_mtp=True)
        self.assertTrue(torch.equal(out_c["logits"], out_r["logits"]), "Cached QAT forward must be bit-identical")

        loss_c = m_cached.compute_loss(out_c, tgt_out, mtp_loss_weight=0.30, uv_loss_weight=0.0)["loss"]
        loss_r = m_ref.compute_loss(out_r, tgt_out, mtp_loss_weight=0.30, uv_loss_weight=0.0)["loss"]
        self.assertTrue(torch.equal(loss_c, loss_r))
        loss_c.backward()
        loss_r.backward()

        for (n_c, p_c), (n_r, p_r) in zip(m_cached.named_parameters(), m_ref.named_parameters()):
            self.assertEqual(n_c, n_r)
            if p_c.grad is None:
                self.assertIsNone(p_r.grad)
            else:
                self.assertTrue(
                    torch.equal(p_c.grad, p_r.grad),
                    f"STE grad mismatch on {n_c}: max diff={(p_c.grad - p_r.grad).abs().max().item()}",
                )

        # Verify cache updates cleanly after opt.step() + refresh_cache() and across gpu_ema.swap_into()
        opt_c = torch.optim.SGD(m_cached.parameters(), lr=0.1)
        opt_r = torch.optim.SGD(m_ref.parameters(), lr=0.1)
        opt_c.step()
        opt_r.step()
        ctrl_cached.refresh_cache()
        self.assertEqual(len(ctrl_cached._cached_deltas), ctrl_cached.num_layers)
        with torch.no_grad():
            o2_c = m_cached(img, tgt_in, return_aux=False, return_mtp=False)["logits"]
            o2_r = m_ref(img, tgt_in, return_aux=False, return_mtp=False)["logits"]
        self.assertTrue(torch.equal(o2_c, o2_r))

        ema = uni_omr.GPUEmaShadow(m_cached, decay=0.9, update_every=1)
        with torch.no_grad():
            m_cached.lm_head.weight.add_(0.25)
        ctrl_cached.refresh_cache()
        with ema.swap_into(m_cached, qat_ctrl=ctrl_cached):
            with torch.no_grad():
                o_ema = m_cached(img, tgt_in, return_aux=False, return_mtp=False)["logits"]
            self.assertTrue(torch.equal(o_ema, o2_c))
        ctrl_cached.invalidate_cache()
        self.assertEqual(len(ctrl_cached._cached_deltas), 0)
        ctrl_cached.detach()

    def test_zg_gpu_ema_foreach_lerp_parity_and_ema_only_eval(self) -> None:
        """(zg) Iter-10: GPUEmaShadow _foreach_lerp_ parity + --ema-only-eval + fullpage_val bs cap."""
        import json

        cfg = _small_cfg(d_model=64, n_enc_layers=1, n_dec_layers=1, n_heads=4, mtp_k=2, dropout=0.0)
        torch.manual_seed(11)
        model = uni_omr.UniOMRNet(cfg)
        model.backbone.to(memory_format=torch.channels_last)
        ema = uni_omr.GPUEmaShadow(model, decay=0.95, update_every=2)
        manual_shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        d_eff = 0.95 ** 2

        for step in range(1, 7):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.05)
            updated = ema.update(model, step)
            self.assertEqual(updated, (step % 2 == 0))
            if step % 2 == 0:
                for k, v in model.state_dict().items():
                    if manual_shadow[k].is_floating_point():
                        manual_shadow[k].mul_(d_eff).add_(v.detach(), alpha=1.0 - d_eff)
                    else:
                        manual_shadow[k].copy_(v.detach())

        for k in manual_shadow:
            self.assertTrue(
                torch.allclose(ema.shadow[k], manual_shadow[k], atol=1e-6),
                f"_foreach_lerp_ mismatch on {k}: max diff={(ema.shadow[k] - manual_shadow[k]).abs().max().item()}",
            )

        # CLI test for --ema-only-eval (mid-run EMA-only at step 2, both student+EMA at final step 4)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            pr_dir = td_path / "printed"
            out_dir = td_path / "out"
            pr_dir.mkdir()
            bek_txt = "clef-G-2\nquarter\nC\n4\nbar\n"
            for i in range(4):
                arr = np.full((64, 128, 3), 245, dtype=np.uint8)
                arr[20:44, 15:110, :] = 30
                Image.fromarray(arr, mode="RGB").save(pr_dir / f"s{i}_p1_s0.png")
                (pr_dir / f"s{i}_p1_s0-bekern.txt").write_text(bek_txt, encoding="utf-8")

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = uni_omr.main([
                    "--mode", "train",
                    "--curriculum", "strip",
                    "--strip-dirs", str(pr_dir),
                    "--strip-height", "64",
                    "--max-strip-width", "128",
                    "--max-seq-len", "32",
                    "--d-model", "64",
                    "--n-enc-layers", "1",
                    "--n-dec-layers", "1",
                    "--n-heads", "4",
                    "--steps", "4",
                    "--warmup-steps", "1",
                    "--batch-size", "1",
                    "--val-count", "1",
                    "--eval-every", "2",
                    "--eval-max-len", "8",
                    "--gpu-ema-decay", "0.99",
                    "--ema-only-eval",
                    "--ckpt-enc-tokens", "20000",
                    "--ckpt-dec-tokens", "8000",
                    "--device", "cpu",
                    "--out-dir", str(out_dir),
                ])
            self.assertEqual(rc, 0)
            out_txt = buf.getvalue()
            self.assertNotIn("[eval strip_val @ 2]", out_txt)
            self.assertIn("[eval strip_val @ 2 EMA]", out_txt)
            self.assertIn("[eval strip_val @ 4]", out_txt)
            self.assertIn("[eval strip_val @ 4 EMA]", out_txt)
            ev = json.loads((out_dir / "eval_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(ev["evals"][0].get("ema_only", False))
            self.assertFalse(ev["evals"][1].get("ema_only", False))

    def test_zh_wsd_lr_schedule_and_stratified_fullpage_weights(self) -> None:
        """(zh) Iter-10: WSD LR schedule + 5-bucket domain/renderer-stratified full-page sampler."""
        import random as _py_random

        base_lr = 1e-3
        total = 100
        warmup = 10
        # Warmup: linear ramp
        self.assertAlmostEqual(uni_omr._lr_for_step(5, total, base_lr, warmup, min_lr_ratio=0.05, schedule="wsd", wsd_decay_frac=0.25), 5e-4, places=7)
        self.assertAlmostEqual(uni_omr._lr_for_step(10, total, base_lr, warmup, min_lr_ratio=0.05, schedule="wsd", wsd_decay_frac=0.25), base_lr, places=7)
        # Stable plateau: steps 11..75 stay at base_lr
        for st in (11, 40, 75):
            self.assertAlmostEqual(uni_omr._lr_for_step(st, total, base_lr, warmup, min_lr_ratio=0.05, schedule="wsd", wsd_decay_frac=0.25), base_lr, places=7)
        # Decay phase: steps 76..100 decrease monotonically down to base_lr * 0.05
        lr_mid = uni_omr._lr_for_step(88, total, base_lr, warmup, min_lr_ratio=0.05, schedule="wsd", wsd_decay_frac=0.25)
        lr_end = uni_omr._lr_for_step(100, total, base_lr, warmup, min_lr_ratio=0.05, schedule="wsd", wsd_decay_frac=0.25)
        self.assertLess(lr_mid, base_lr)
        self.assertGreater(lr_mid, lr_end)
        self.assertAlmostEqual(lr_end, base_lr * 0.05, places=7)

        # 5-bucket classifier
        self.assertEqual(uni_omr.classify_fullpage_bucket("D:/tmp/t32farm/sq1_Leland.png", "printed"), "legacy_printed")
        self.assertEqual(uni_omr.classify_fullpage_bucket("D:/tmp/verovio_pages/p1.png", "printed"), "multi_renderer_printed")
        self.assertEqual(uni_omr.classify_fullpage_bucket("D:/tmp/osmd_spacing/p2.png", "printed"), "multi_renderer_printed")
        self.assertEqual(uni_omr.classify_fullpage_bucket("D:/tmp/corpus100k/p3.png", "printed"), "multi_renderer_printed")
        self.assertEqual(uni_omr.classify_fullpage_bucket("D:/tmp/c-8vb_rare/p4.png", "printed"), "rare_printed")
        self.assertEqual(uni_omr.classify_fullpage_bucket("D:/tmp/hw-sim-v05/p5.png", "hw"), "synth_hw")
        self.assertEqual(uni_omr.classify_fullpage_bucket("D:/tmp/debussy-omr/p6.png", "real_hw"), "real_hw")

        # StratifiedFullPageSampler quotas match target weights (26/22/7/27/18%) even under imbalanced counts
        pairs = []
        for i in range(500):
            pairs.append((Path(f"D:/tmp/t32farm/leg_{i}.png"), [], "printed"))
        for i in range(100):
            pairs.append((Path(f"D:/tmp/verovio/mr_{i}.png"), [], "printed"))
        for i in range(20):
            pairs.append((Path(f"D:/tmp/c-8vb/rare_{i}.png"), [], "printed"))
        for i in range(200):
            pairs.append((Path(f"D:/tmp/hw-sim-v05/shw_{i}.png"), [], "hw"))
        for i in range(30):
            pairs.append((Path(f"D:/tmp/debussy/rhw_{i}.png"), [], "real_hw"))

        sampler = uni_omr.StratifiedFullPageSampler(pairs, epoch_size=1000)
        ep = sampler.sample_epoch(_py_random.Random(42))
        self.assertEqual(len(ep), 1000)
        from collections import Counter as _Counter
        counts = _Counter(uni_omr.classify_fullpage_bucket(pairs[i][0], pairs[i][2]) for i in ep)
        for bname, target_w in uni_omr.STRATIFIED_FULLPAGE_WEIGHTS.items():
            self.assertAlmostEqual(counts[bname] / 1000.0, target_w, delta=0.01, msg=f"Bucket {bname} quota drift: {counts}")

    def test_zi_2d_aspect_length_bucketing_and_double_pad_fix(self) -> None:
        """(zi) Iter-10: 2D aspect + bekern_bytes mega-batch bucketing & strip double-padding fix."""
        # 1. Double-padding fix in collate_uni_batch + unpack_collated_batch
        s_short = {
            "image": torch.zeros(3, 64, 128, dtype=torch.uint8),
            "seq": torch.tensor([uni_omr.BOS_ID, 1, 2, uni_omr.EOS_ID], dtype=torch.long),
            "sample_id": "short",
            "kind": "strip",
        }
        s_long = {
            "image": torch.zeros(3, 64, 384, dtype=torch.uint8),
            "seq": torch.tensor([uni_omr.BOS_ID, 1, 2, 3, 4, 5, 6, 7, 8, uni_omr.EOS_ID], dtype=torch.long),
            "sample_id": "long",
            "kind": "strip",
        }
        collated = uni_omr.collate_uni_batch([s_short, s_long])
        self.assertEqual(tuple(collated["images"].shape), (2, 3, 64, 384))
        self.assertEqual(tuple(collated["seq"].shape), (2, 10))
        unpacked = uni_omr.unpack_collated_batch(collated, "strip")
        self.assertEqual(tuple(unpacked[0]["image"].shape), (3, 64, 128))
        self.assertEqual(tuple(unpacked[0]["seq"].shape), (4,))
        self.assertNotIn(uni_omr.PAD_ID, unpacked[0]["seq"].tolist())
        # Re-collating two short strips unpacked from larger batches must stay compact (width 128, len 4)
        re_collated = uni_omr.collate_uni_batch([unpacked[0], unpacked[0]])
        self.assertEqual(tuple(re_collated["images"].shape), (2, 3, 64, 128))
        self.assertEqual(tuple(re_collated["seq"].shape), (2, 4))

        # 2. 2D aspect + bekern_bytes mega-batch bucketing in AsyncThreadPrefetcher
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            fp_pages = td_path / "pages"
            fp_strips = td_path / "stage2_strips"
            fp_pages.mkdir()
            fp_strips.mkdir()

            # 16 portrait (narrow & wide aspect, short & long bekern) + 16 landscape
            for i in range(16):
                w_p = 64 if (i < 8) else 112
                arr_p = np.full((128, w_p, 3), 245, dtype=np.uint8)
                Image.fromarray(arr_p, mode="RGB").save(fp_pages / f"port{i:02d}_p1.png")
                n_rep = 2 if (i % 2 == 0) else 30
                (fp_strips / f"port{i:02d}_p1_s0-bekern.txt").write_text(
                    "clef-G-2\n" + ("quarter\nC\n4\n" * n_rep) + "bar\n", encoding="utf-8"
                )

                h_l = 64 if (i < 8) else 112
                arr_l = np.full((h_l, 128, 3), 245, dtype=np.uint8)
                Image.fromarray(arr_l, mode="RGB").save(fp_pages / f"land{i:02d}_p1.png")
                (fp_strips / f"land{i:02d}_p1_s0-bekern.txt").write_text(
                    "clef-G-2\n" + ("quarter\nC\n4\n" * n_rep) + "bar\n", encoding="utf-8"
                )

            fp_pairs = uni_omr.discover_fullpage_pairs([td_path])
            fp_ds = uni_omr.UniFullPageDataset(
                fp_pairs, fullpage_max_dim=128, max_seq_len=256, augment=False, return_uint8=True
            )
            pf = uni_omr.AsyncThreadPrefetcher(
                fp_ds, batch_size=2, shuffle=True, pin_memory=False, max_workers=2, prefetch_batches=2, seed=42
            )
            try:
                pad_toks = 0
                total_toks = 0
                for _ in range(16):
                    b = next(pf)
                    _, _, bh, bw = b["images"].shape
                    self.assertNotEqual((bh, bw), (128, 128), "Portrait & landscape mixed in 2D bucket")
                    seq = b["seq"]
                    pad_toks += int((seq == uni_omr.PAD_ID).sum().item())
                    total_toks += int(seq.numel())
                # Because short & long bekerns are sorted within each aspect bucket, pad waste must be ~0%!
                self.assertLess(pad_toks / max(1, total_toks), 0.05)
            finally:
                pf.close()

    def test_zj_cv2_aug_speed_and_output_sanity(self) -> None:
        """(zj) Iter-10: OpenCV fast augs match shape/bounds/determinism and fall back cleanly when _cv2 is None."""
        import random as _py_random
        import time as _time
        from unittest import mock

        arr = np.full((384, 288, 3), 246, dtype=np.uint8)
        arr[40:340:12, 24:264, :] = 22
        base = Image.fromarray(arr, mode="RGB")

        # 1. cv2 path (if installed) sanity + determinism + speed
        t0 = _time.perf_counter()
        out_lr1 = uni_omr._apply_hw_lowres_defocus_compress(base, _py_random.Random(123))
        out_lr2 = uni_omr._apply_hw_lowres_defocus_compress(base, _py_random.Random(123))
        out_blur = uni_omr._fast_gaussian_blur(base, 0.8)
        out_rot = uni_omr._fast_rotate(base, 3.0, (240, 240, 240))
        out_persp = uni_omr._fast_perspective(
            base,
            [(0.0, 0.0), (288.0, 0.0), (288.0, 384.0), (0.0, 384.0)],
            [(6.0, 8.0), (280.0, 6.0), (282.0, 376.0), (8.0, 378.0)],
            (25, 25, 25),
        )
        out_full = uni_omr.apply_online_augment(base, rng=_py_random.Random(99), is_hw=True)
        elapsed_ms = (_time.perf_counter() - t0) * 1000.0

        for im in (out_lr1, out_blur, out_rot, out_persp, out_full):
            self.assertEqual(im.size, base.size)
            self.assertEqual(im.mode, "RGB")
        np.testing.assert_array_equal(np.asarray(out_lr1), np.asarray(out_lr2))
        self.assertGreater(float(np.asarray(out_lr1).min()), 30.0)
        self.assertLess(elapsed_ms, 2000.0)

        # 2. Graceful PIL fallback when _cv2 is None
        with mock.patch.object(uni_omr, "_cv2", None):
            fb_lr = uni_omr._apply_hw_lowres_defocus_compress(base, _py_random.Random(123))
            fb_blur = uni_omr._fast_gaussian_blur(base, 0.8)
            fb_rot = uni_omr._fast_rotate(base, 3.0, (240, 240, 240))
            fb_persp = uni_omr._fast_perspective(
                base,
                [(0.0, 0.0), (288.0, 0.0), (288.0, 384.0), (0.0, 384.0)],
                [(6.0, 8.0), (280.0, 6.0), (282.0, 376.0), (8.0, 378.0)],
                (25, 25, 25),
            )
            for im in (fb_lr, fb_blur, fb_rot, fb_persp):
                self.assertEqual(im.size, base.size)
                self.assertEqual(im.mode, "RGB")
            self.assertGreater(float(np.asarray(fb_lr).min()), 30.0)


if __name__ == "__main__":
    unittest.main()
