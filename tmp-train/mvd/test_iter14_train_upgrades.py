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
    SYS_ID,
    VOCAB_SIZE,
    GPUEmaShadow,
    UniFullPageDataset,
    UniOMRConfig,
    UniOMRNet,
    UniStripDataset,
    make_sample_id,
)
from uni_omr_v2 import InkTokenRouterV2, UniOMRv2Config, UniOMRv2Net  # noqa: E402
from vir_router_v2 import ViRTwoTierRouter  # noqa: E402


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

    # ------------------------------------------------------------------
    # 4. Adaptive Router ink_target (InkTokenRouterV2 & ViRTwoTierRouter)
    # ------------------------------------------------------------------
    def test_adaptive_ink_target_suppresses_aged_paper_background(self) -> None:
        torch.manual_seed(7)
        B, D, H_img, W_img = 1, 32, 128, 128
        H_p, W_p = H_img // 16, W_img // 16

        # Simulate aged/foxed archival paper: mottled background with local_ink ~ 0.06-0.08
        # plus crisp dark ink strokes in a small region (local_ink ~ 0.65).
        bg = torch.full((B, 3, H_img, W_img), 0.78, dtype=torch.float32)
        # Checkerboard-like fine foxing speckle with contrast ~0.07 (above legacy 0.04, below adaptive thr)
        yy = torch.arange(H_img).view(1, 1, H_img, 1)
        xx = torch.arange(W_img).view(1, 1, 1, W_img)
        speckle = (((yy + xx) % 2) == 0).float() * 0.09
        aged_img = (bg - speckle).clamp(0.0, 1.0)
        # Add genuine dark ink block (2 patches)
        aged_img[:, :, 48:64, 48:80] = 0.08

        feat = torch.randn(B, D, H_p, W_p)

        # InkTokenRouterV2: legacy (0.04) vs adaptive (clamp(mean+0.5*std, 0.08, 0.20))
        r_legacy = InkTokenRouterV2(d_model=D, adaptive_ink_target=False).eval()
        r_adapt = InkTokenRouterV2(d_model=D, adaptive_ink_target=True).eval()
        with torch.no_grad():
            _, _, info_leg = r_legacy(feat, images=aged_img)
            _, _, info_adp = r_adapt(feat, images=aged_img)
        frac_leg = float(info_leg["ink_target"].mean().item())
        frac_adp = float(info_adp["ink_target"].mean().item())
        self.assertGreater(frac_leg, 0.50, f"Expected legacy 0.04 to over-fire on aged paper, got {frac_leg:.3f}")
        self.assertLess(frac_adp, 0.25, f"Expected adaptive threshold to suppress aged paper bg, got {frac_adp:.3f}")
        self.assertGreater(frac_adp, 0.0, "Adaptive threshold must still retain genuine dark ink strokes")

        # ViRTwoTierRouter: same adaptive behavior in pure float32
        v_legacy = ViRTwoTierRouter(d_model=D, k_coarse=4, adaptive_ink_target=False).eval()
        v_adapt = ViRTwoTierRouter(d_model=D, k_coarse=4, adaptive_ink_target=True).eval()
        with torch.no_grad():
            _, _, vinfo_leg = v_legacy(feat, images=aged_img)
            _, _, vinfo_adp = v_adapt(feat, images=aged_img)
        vfrac_leg = float(vinfo_leg["ink_target"].mean().item())
        vfrac_adp = float(vinfo_adp["ink_target"].mean().item())
        self.assertGreater(vfrac_leg, 0.50)
        self.assertLess(vfrac_adp, 0.25)
        self.assertGreater(vfrac_adp, 0.0)

        # CLI flag wiring (--adaptive-ink-target / --no-adaptive-ink-target)
        parser = tv2.build_arg_parser()
        self.assertTrue(parser.parse_args([]).adaptive_ink_target)
        self.assertFalse(parser.parse_args(["--no-adaptive-ink-target"]).adaptive_ink_target)

    # ------------------------------------------------------------------
    # 5. Decoder --tok-dropout & --sys-count-loss-weight wiring
    # ------------------------------------------------------------------
    def test_tok_dropout_and_sys_count_aux_wiring(self) -> None:
        parser = tv2.build_arg_parser()
        args = parser.parse_args(["--tok-dropout", "0.12", "--sys-count-loss-weight", "0.35"])
        self.assertAlmostEqual(args.tok_dropout, 0.12)
        self.assertAlmostEqual(args.sys_count_loss_weight, 0.35)

        torch.manual_seed(19)
        cfg = tv2.uni_omr_v2_tiny(tok_dropout=0.50)
        model = UniOMRv2Net(cfg).train()
        seen_inp: list = []
        orig_emb = model.tok_emb.forward

        def _spy_emb(ids):
            seen_inp.append(ids.clone())
            return orig_emb(ids)

        seq = torch.full((2, 40), 10, dtype=torch.long)
        seq[:, 0] = BOS_ID
        seq[:, -2] = EOS_ID
        seq[:, -1] = PAD_ID
        mem = torch.randn(2, 16, cfg.d_model)
        with mock.patch.object(model.tok_emb, "forward", side_effect=_spy_emb):
            model.decode_tokens(seq, mem, return_mtp=False, return_aux=False)
        inp_used = seen_inp[0]
        # Special tokens BOS / EOS / PAD must never be corrupted
        self.assertTrue(bool((inp_used[:, 0] == BOS_ID).all()))
        self.assertTrue(bool((inp_used[:, -2] == EOS_ID).all()))
        self.assertTrue(bool((inp_used[:, -1] == PAD_ID).all()))
        # Middle music tokens must have some corruption under tok_dropout=0.50
        n_changed = int((inp_used[:, 1:-2] != 10).sum().item())
        self.assertGreater(n_changed, 5)

    # ------------------------------------------------------------------
    # 6. STRATIFIED_FULLPAGE_WEIGHTS_V3 & graceful renormalization
    # ------------------------------------------------------------------
    def test_stratified_fullpage_weights_v3_and_renormalization(self) -> None:
        # v2 stays untouched for backward compat
        self.assertAlmostEqual(sum(tv2.STRATIFIED_FULLPAGE_WEIGHTS_V2.values()), 1.0, places=6)
        self.assertNotIn("real_hw_pseudo", tv2.STRATIFIED_FULLPAGE_WEIGHTS_V2)
        self.assertNotIn("neg_control", tv2.STRATIFIED_FULLPAGE_WEIGHTS_V2)

        # v3 exact weights
        w3 = tv2.STRATIFIED_FULLPAGE_WEIGHTS_V3
        self.assertAlmostEqual(sum(w3.values()), 1.0, places=6)
        self.assertAlmostEqual(
            w3["legacy_printed"] + w3["multi_renderer_printed"] + w3["rare_printed"], 0.50, places=6
        )
        self.assertAlmostEqual(w3["synth_hw"], 0.195, places=6)
        self.assertAlmostEqual(w3["real_hw_archival"], 0.120, places=6)
        self.assertAlmostEqual(w3["real_hw_pseudo"], 0.120, places=6)
        self.assertAlmostEqual(w3["real_hw_stitched"], 0.060, places=6)
        self.assertAlmostEqual(w3["neg_control"], 0.005, places=6)

        # Full 8-bucket corpus
        pairs_all = [
            (Path("legacy/p1.png"), [Path("p1.bekern")], "printed"),
            (Path("verovio/p2.png"), [Path("p2.bekern")], "multi_renderer_printed"),
            (Path("rare/p3.png"), [Path("p3.bekern")], "rare_printed"),
            (Path("hw-sim-v05/p4.png"), [Path("p4.bekern")], "synth_hw"),
            (Path("debussy/p5.png"), [Path("p5.bekern")], "real_hw"),
            (Path("rism_pseudo/p6.png"), [Path("p6.bekern")], "real_hw_pseudo"),
            (Path("real_hw_stitched/p7.png"), [Path("p7.bekern")], "real_hw"),
            (Path("neg_control/p8.png"), [], "neg_control"),
        ]
        s_all = tv2.ArchivalStratifiedFullPageSampler(pairs_all, preset="v3")
        self.assertEqual(s_all.bucket_counts()["real_hw_pseudo"], 1)
        self.assertEqual(s_all.bucket_counts()["neg_control"], 1)
        self.assertAlmostEqual(sum(s_all.active_weights.values()), 1.0, places=6)
        self.assertAlmostEqual(s_all.active_weights["real_hw_pseudo"], 0.120, places=6)
        self.assertAlmostEqual(s_all.active_weights["neg_control"], 0.005, places=6)

        # Renormalization when 0 real_hw_pseudo and 0 neg_control rows are present
        pairs_no_pseudo_neg = pairs_all[:5] + [pairs_all[6]]
        s_renorm = tv2.ArchivalStratifiedFullPageSampler(pairs_no_pseudo_neg, preset="v3")
        self.assertNotIn("real_hw_pseudo", s_renorm.active_weights)
        self.assertNotIn("neg_control", s_renorm.active_weights)
        self.assertAlmostEqual(sum(s_renorm.active_weights.values()), 1.0, places=6)
        expected_denom = 1.0 - 0.120 - 0.005
        self.assertAlmostEqual(
            s_renorm.active_weights["real_hw_archival"], 0.120 / expected_denom, places=6
        )
        self.assertAlmostEqual(
            s_renorm.active_weights["synth_hw"], 0.195 / expected_denom, places=6
        )

        # split_fullpage_train_val_with_hw must never put real_hw_pseudo or neg_control into hw_val
        _, _, hw_va = tv2.split_fullpage_train_val_with_hw(pairs_all, val_count=1, val_real_hw_count=1, seed=42)
        for entry in hw_va:
            dom = str(entry[2])
            self.assertNotIn(dom, ("real_hw_pseudo", "neg_control"))

    # ------------------------------------------------------------------
    # 7. neg_control empty-target (<bos> <eos>) loss/mask/grad safety
    # ------------------------------------------------------------------
    def test_neg_control_empty_target_forward_and_backward_safe(self) -> None:
        torch.manual_seed(23)
        model = UniOMRv2Net(tv2.uni_omr_v2_tiny(tok_dropout=0.12)).train()

        # Case A: pure neg_control batch with minimal length [BOS, EOS] (T=2 -> tgt_out T=1)
        imgs_a = torch.rand(2, 3, 128, 128)
        seq_a = torch.tensor([[BOS_ID, EOS_ID], [BOS_ID, EOS_ID]], dtype=torch.long)
        losses_a, _ = tv2.forward_and_loss_v2(
            model, imgs_a, seq_a, domains=["neg_control", "neg_control"], cass_prob=0.25
        )
        for k in ("loss", "ce_loss", "mtp_loss", "sys_count_loss"):
            self.assertTrue(bool(torch.isfinite(losses_a[k])), f"{k} not finite: {losses_a[k]}")
        model.zero_grad(set_to_none=True)
        losses_a["loss"].backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                self.assertTrue(bool(torch.isfinite(p.grad).all()), f"NaN/Inf grad in {name}")

        # Case B: pure neg_control batch padded to T=6 (all MTP shifted targets are PAD_ID!)
        seq_b = torch.tensor(
            [
                [BOS_ID, EOS_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
                [BOS_ID, EOS_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
            ],
            dtype=torch.long,
        )
        losses_b, _ = tv2.forward_and_loss_v2(
            model, imgs_a, seq_b, domains=["neg_control", "neg_control"], cass_prob=0.25
        )
        self.assertTrue(bool(torch.isfinite(losses_b["loss"])))
        self.assertAlmostEqual(float(losses_b["mtp_loss"].item()), 0.0, places=6)
        model.zero_grad(set_to_none=True)
        losses_b["loss"].backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                self.assertTrue(bool(torch.isfinite(p.grad).all()), f"NaN/Inf grad in {name}")

        # Case C: UniFullPageDataset with empty bekerns=[] produces exact [BOS_ID, EOS_ID]
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_p = Path(tmp_dir) / "neg_blank.png"
            Image.new("RGB", (96, 64), (245, 240, 230)).save(img_p)
            ds = UniFullPageDataset([(img_p, [], "neg_control")], fullpage_max_dim=128, max_seq_len=64, augment=False)
            item = ds[0]
            self.assertEqual(item["seq"].tolist(), [BOS_ID, EOS_ID])

    # ------------------------------------------------------------------
    # 8. Milestone EMA snapshots & pruning
    # ------------------------------------------------------------------
    def test_ema_milestone_snapshots_and_pruning(self) -> None:
        torch.manual_seed(31)
        cfg = tv2.uni_omr_v2_tiny()
        model = UniOMRv2Net(cfg)
        ema = GPUEmaShadow(model, decay=0.99, update_every=1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            for s in (4000, 8000, 12000, 16000):
                p = tv2.atomic_save_v2_ema_milestone(
                    out_dir, model, cfg, step=s, gpu_ema=ema, best_ema_ser=0.42, keep_last=3
                )
                self.assertIsNotNone(p)
            remaining = sorted(x.name for x in out_dir.glob("uni_omr_v2_ema_step*.pt"))
            self.assertEqual(
                remaining,
                [
                    "uni_omr_v2_ema_step12000.pt",
                    "uni_omr_v2_ema_step16000.pt",
                    "uni_omr_v2_ema_step8000.pt",
                ],
            )
            ck = torch.load(str(out_dir / "uni_omr_v2_ema_step16000.pt"), map_location="cpu", weights_only=True)
            self.assertEqual(ck["step"], 16000)
            self.assertIn("model", ck)

        # CLI defaults
        parser = tv2.build_arg_parser()
        args_def = parser.parse_args([])
        self.assertEqual(args_def.save_ema_milestones, 4000)
        self.assertEqual(args_def.keep_ema_milestones, 0)
        self.assertEqual(args_def.best_min_step, 0)
        self.assertIsNone(args_def.stratified_preset)
        args_v3 = parser.parse_args(["--stratified-preset", "v3", "--best-min-step", "6000"])
        self.assertEqual(args_v3.stratified_preset, "v3")
        self.assertEqual(args_v3.best_min_step, 6000)


if __name__ == "__main__":
    unittest.main()
