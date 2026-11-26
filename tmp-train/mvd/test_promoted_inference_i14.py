"""CPU unit tests for the 3 promoted Iter-13/ov-38b inference upgrades wired for Iter-14.

Covers:
  1. i13-18b Confidence-Gated ILMS Blank Decoding (`--ilms-alpha 0.40 --ilms-conf-thr 0.85`).
  2. ov-38b Backbone+Router Consistency Scribe-TTT (`--ttt-mode backbone_cons`).
  3. i13-06c 3-Decimal Confidence Skip Gate (`--ttt-conf-skip-thr -0.040`).

Runs strictly on CPU (no GPU contact).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import eval_uni_suite31 as ev  # noqa: E402
import trial_2028_ttt_adapt as ttt_mod  # noqa: E402
import uni_omr  # noqa: E402


def _make_tiny_v1_model(seed: int = 123) -> uni_omr.UniOMRNet:
    torch.manual_seed(seed)
    cfg = uni_omr.UniOMRConfig(
        d_model=64,
        n_enc_layers=2,
        n_dec_layers=2,
        n_heads=4,
        ffn_mult=2.0,
        dropout=0.0,
        keep_ratio=0.50,
        n_bg_summary=4,
        use_ink_prior=True,
        use_uv_aux=False,
        mtp_k=2,
    )
    return uni_omr.UniOMRNet(cfg).eval()


def _synthetic_page(seed: int = 7, h: int = 96, w: int = 128) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.full((1, 3, h, w), 0.92, dtype=torch.float32)
    for y in (20, 26, 32, 38, 44, 62, 68, 74, 80, 86):
        img[:, :, y : y + 1, 12 : w - 12] = 0.15
    for x in range(20, w - 20, 14):
        y = 22 + ((x * 7) % 56)
        img[:, :, y : y + 4, x : x + 4] = 0.08
    noise = (torch.rand((1, 3, h, w), generator=g) - 0.5) * 0.03
    return (img + noise).clamp(0.0, 1.0)


class TestPromotedInferenceI14(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    # ------------------------------------------------------------------
    # 1. i13-18b Confidence-Gated ILMS Blank Decoding
    # ------------------------------------------------------------------

    def test_ilms_leaves_confident_logits_bit_identical_and_subtracts_only_uncertain(self) -> None:
        # Confident token: max_prob >> 0.85 -> MUST be bit-identical even with extreme blank logits
        lo_conf = torch.tensor([6.0, 0.0, -1.0, -2.0], dtype=torch.float32)
        p_conf = float(F.softmax(lo_conf, dim=-1).max().item())
        self.assertGreaterEqual(p_conf, 0.85)
        lr_extreme = torch.tensor([100.0, -100.0, 50.0, 0.0], dtype=torch.float32)

        out_conf, applied_conf = ev.apply_conf_gated_ilms_logits(
            lo_conf, lr_extreme, alpha=0.40, conf_thr=0.85
        )
        self.assertFalse(applied_conf)
        self.assertTrue(torch.equal(out_conf, lo_conf))

        # Uncertain token: max_prob < 0.85 -> subtracts alpha * blank_logits
        # Token 0 has slightly higher raw logit (1.0 vs 0.85), but blank prior strongly favors token 0
        lo_unc = torch.tensor([1.0, 0.85, -5.0, -6.0], dtype=torch.float32)
        p_unc = float(F.softmax(lo_unc, dim=-1).max().item())
        self.assertLess(p_unc, 0.85)
        lr_prior = torch.tensor([4.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        out_unc, applied_unc = ev.apply_conf_gated_ilms_logits(
            lo_unc, lr_prior, alpha=0.40, conf_thr=0.85, beta=0.10
        )
        self.assertTrue(applied_unc)
        # Verify exact formula (1 + alpha) * lo - alpha * lr on plausible tokens (0 and 1)
        expected_0 = 1.40 * 1.0 - 0.40 * 4.0
        expected_1 = 1.40 * 0.85 - 0.40 * 0.0
        self.assertAlmostEqual(float(out_unc[0].item()), expected_0, places=5)
        self.assertAlmostEqual(float(out_unc[1].item()), expected_1, places=5)
        # Implausible tokens (< max_logp + log(0.10)) are masked to -1e9
        self.assertLess(float(out_unc[2].item()), -1e8)
        self.assertLess(float(out_unc[3].item()), -1e8)
        # Prior subtraction flips argmax from 0 -> 1
        self.assertEqual(int(torch.argmax(lo_unc).item()), 0)
        self.assertEqual(int(torch.argmax(out_unc).item()), 1)

        # alpha=0.0 disables ILMS even on uncertain tokens
        out_off, applied_off = ev.apply_conf_gated_ilms_logits(
            lo_unc, lr_prior, alpha=0.0, conf_thr=0.85
        )
        self.assertFalse(applied_off)
        self.assertTrue(torch.equal(out_off, lo_unc))

    def test_generate_conf_gated_ilms_alpha0_parity_and_end_to_end(self) -> None:
        model = _make_tiny_v1_model(seed=17)
        img = _synthetic_page(seed=3, h=64, w=96)
        blank_pil = ev.build_canonical_blank_pil(96)
        blank_t = uni_omr._pil_to_chw_tensor(blank_pil).unsqueeze(0)
        with torch.no_grad():
            mem_blank, _, _ = model.encode_image(blank_t, return_aux=False)

        seq_base = ev.generate_constrained(model, img, max_len=10, strict=False)
        seq_a0, st_a0 = ev.generate_conf_gated_ilms(
            model, img, mem_blank, max_len=10, alpha=0.0, conf_thr=0.85,
            use_grammar=True, strict_grammar=False, return_stats=True,
        )
        self.assertTrue(torch.equal(seq_base, seq_a0))
        self.assertEqual(st_a0["n_ref_calls"], 0)
        self.assertEqual(st_a0["n_ilms_applied"], 0)

        seq_ilms, st_ilms = ev.generate_conf_gated_ilms(
            model, img, mem_blank, max_len=10, alpha=0.40, conf_thr=0.85,
            use_grammar=True, strict_grammar=False, return_stats=True,
        )
        self.assertEqual(seq_ilms.shape[0], 1)
        self.assertGreaterEqual(seq_ilms.shape[1], 2)
        self.assertGreater(st_ilms["n_ilms_applied"], 0)
        self.assertGreater(st_ilms["n_ref_calls"], 0)

    # ------------------------------------------------------------------
    # 2. ov-38b Backbone+Router Consistency Scribe-TTT (`backbone_cons`)
    # ------------------------------------------------------------------

    def test_backbone_cons_freezes_enc_layers_updates_backbone_router_and_restores_bit_exact(self) -> None:
        model = _make_tiny_v1_model(seed=55)
        img = _synthetic_page(seed=19)
        full_before = {name: p.detach().clone() for name, p in model.named_parameters()}

        session, preset_steps = ttt_mod.build_ttt_session(model, mode="backbone_cons")
        self.assertEqual(preset_steps, 2)
        self.assertEqual(session.adapt_scope, "backbone_router")
        self.assertAlmostEqual(session.ttt_lr, 3e-5, places=10)
        self.assertAlmostEqual(session.cons_weight, 1.0, places=6)
        self.assertAlmostEqual(session.ent_weight, 0.0, places=6)
        self.assertAlmostEqual(session.kl_weight, 0.0, places=6)
        self.assertAlmostEqual(session.l2_weight, 1e-2, places=6)

        # Verify adapted param names belong strictly to backbone or router (0 enc_layers / decoder)
        adapted_names = set(session.adapted_param_names)
        self.assertGreater(len(adapted_names), 0)
        self.assertTrue(any(n.startswith("backbone.") for n in adapted_names))
        self.assertTrue(any(n.startswith("router.") for n in adapted_names))
        self.assertFalse(any(n.startswith("enc_layers.") for n in adapted_names))
        self.assertFalse(any(n.startswith("enc_norm.") for n in adapted_names))
        self.assertFalse(any(n.startswith("dec_layers.") for n in adapted_names))

        # Attach a backward hook on enc_layers to prove 0 grad ever flows into enc_layers params
        enc_grad_seen = []
        hooks = []
        for p in model.enc_layers.parameters():
            hooks.append(p.register_hook(lambda g: enc_grad_seen.append(g.abs().max().item())))

        try:
            stats = session.adapt_on_page(img, steps=preset_steps)
        finally:
            for h in hooks:
                h.remove()

        self.assertEqual(stats["steps"], 2)
        self.assertFalse(stats["skipped_by_conf_gate"])
        self.assertEqual(enc_grad_seen, [], "enc_layers must remain frozen with 0 grad")
        self.assertEqual(stats["n_decoded_tokens"], 0, "ent=0,kl=0 should skip pseudo-decode when no skip gate")

        # Verify ONLY backbone + router changed; enc_layers / enc_norm / decoder are bit-identical
        bb_changed = False
        for name, p in model.named_parameters():
            diff = float((p.detach() - full_before[name]).abs().max().item())
            if name in adapted_names:
                if diff > 0.0:
                    bb_changed = True
            else:
                self.assertEqual(
                    diff, 0.0, f"Frozen parameter {name} changed during backbone_cons TTT (diff={diff})"
                )
        self.assertTrue(bb_changed, "Expected backbone+router weights to update")

        # Verify bit-exact restore after each page
        session.restore()
        max_diff_all = max(
            float((p.detach() - full_before[name]).abs().max().item())
            for name, p in model.named_parameters()
        )
        self.assertEqual(max_diff_all, 0.0, f"restore() was not bit-exact: max_diff={max_diff_all}")

    # ------------------------------------------------------------------
    # 3. i13-06c 3-Decimal Confidence Skip Gate (`--ttt-conf-skip-thr -0.040`)
    # ------------------------------------------------------------------

    def test_conf_skip_gate_3decimal_formatting_and_skip_vs_run_behavior(self) -> None:
        # 3-decimal formatting ensures -0.035, -0.040, -0.045 never collide
        strs = [ttt_mod.format_conf_skip_thr(t) for t in (-0.035, -0.040, -0.045)]
        tags = [ttt_mod.format_conf_skip_tag(t) for t in (-0.035, -0.040, -0.045)]
        self.assertEqual(strs, ["-0.035", "-0.040", "-0.045"])
        self.assertEqual(tags, ["0035", "0040", "0045"])
        self.assertEqual(len(set(tags)), 3)

        # Boundary checks
        self.assertTrue(ttt_mod.should_skip_ttt_by_confidence(-0.020, -0.040))
        self.assertTrue(ttt_mod.should_skip_ttt_by_confidence(-0.039, -0.040))
        self.assertFalse(ttt_mod.should_skip_ttt_by_confidence(-0.040, -0.040))
        self.assertFalse(ttt_mod.should_skip_ttt_by_confidence(-0.120, -0.040))
        self.assertFalse(ttt_mod.should_skip_ttt_by_confidence(-0.001, None))

        model = _make_tiny_v1_model(seed=88)
        img = _synthetic_page(seed=29)
        full_before = {name: p.detach().clone() for name, p in model.named_parameters()}

        session, steps = ttt_mod.build_ttt_session(
            model, mode="backbone_cons", conf_skip_thr=-0.040
        )

        # High-confidence page (mean_logprob = -0.020 > -0.040): MUST skip untouched!
        st_skip = session.adapt_on_page(img, steps=steps, mean_logprob=-0.020)
        self.assertTrue(st_skip["skipped_by_conf_gate"])
        self.assertEqual(st_skip["steps"], 0)
        self.assertEqual(st_skip["conf_skip_thr"], "-0.040")
        max_diff_skip = max(
            float((p.detach() - full_before[name]).abs().max().item())
            for name, p in model.named_parameters()
        )
        self.assertEqual(max_diff_skip, 0.0, "Skipped page must leave all weights untouched")

        # Low-confidence page (mean_logprob = -0.040 <= -0.040): MUST run TTT!
        st_run = session.adapt_on_page(img, steps=steps, mean_logprob=-0.040)
        self.assertFalse(st_run["skipped_by_conf_gate"])
        self.assertEqual(st_run["steps"], 2)
        self.assertEqual(st_run["conf_skip_thr"], "-0.040")
        max_diff_run = max(
            float((p.detach() - full_before[name]).abs().max().item())
            for name, p in model.named_parameters()
        )
        self.assertGreater(max_diff_run, 0.0, "Page with mean_logprob <= -0.040 must run TTT")

        session.restore()
        max_diff_restored = max(
            float((p.detach() - full_before[name]).abs().max().item())
            for name, p in model.named_parameters()
        )
        self.assertEqual(max_diff_restored, 0.0)

    # ------------------------------------------------------------------
    # 4. CLI Backward Compatibility & Opt-in Flags
    # ------------------------------------------------------------------

    def test_cli_defaults_backward_compat_and_opt_in_flags(self) -> None:
        # eval_uni_suite31.py
        ap_ev = ev.build_arg_parser()
        ev_def = ap_ev.parse_args(["--ckpt", "dummy.pt"])
        self.assertAlmostEqual(ev_def.ilms_alpha, 0.0, places=6)
        self.assertAlmostEqual(ev_def.ilms_conf_thr, 0.85, places=6)
        self.assertEqual(ev_def.ttt_mode, "none")
        self.assertIsNone(ev_def.ttt_conf_skip_thr)

        ev_on = ap_ev.parse_args([
            "--ckpt", "dummy.pt",
            "--ilms-alpha", "0.40",
            "--ilms-conf-thr", "0.85",
            "--ttt-mode", "backbone_cons",
            "--ttt-conf-skip-thr", "-0.040",
        ])
        self.assertAlmostEqual(ev_on.ilms_alpha, 0.40, places=6)
        self.assertAlmostEqual(ev_on.ilms_conf_thr, 0.85, places=6)
        self.assertEqual(ev_on.ttt_mode, "backbone_cons")
        self.assertAlmostEqual(ev_on.ttt_conf_skip_thr, -0.040, places=6)

        # trial_2028_ttt_adapt.py
        ap_ttt = ttt_mod.build_arg_parser()
        ttt_def = ap_ttt.parse_args(["--ckpt", "dummy.pt"])
        self.assertEqual(ttt_def.ttt_mode, "custom")
        self.assertEqual(ttt_def.adapt_scope, "norms")
        self.assertAlmostEqual(ttt_def.ttt_lr, 5e-5, places=10)
        self.assertEqual(ttt_def.ttt_steps, 3)
        self.assertIsNone(ttt_def.ttt_conf_skip_thr)

        ttt_bc = ap_ttt.parse_args([
            "--ckpt", "dummy.pt",
            "--ttt-mode", "backbone_cons",
            "--ttt-conf-skip-thr", "-0.040",
        ])
        eff = ttt_mod.resolve_ttt_mode_config(ttt_bc)
        self.assertEqual(eff["ttt_mode"], "backbone_cons")
        self.assertEqual(eff["adapt_scope"], "backbone_router")
        self.assertAlmostEqual(eff["ttt_lr"], 3e-5, places=10)
        self.assertEqual(eff["ttt_steps"], 2)
        self.assertAlmostEqual(eff["ent_weight"], 0.0, places=6)
        self.assertAlmostEqual(eff["kl_weight"], 0.0, places=6)
        self.assertAlmostEqual(eff["cons_weight"], 1.0, places=6)
        self.assertAlmostEqual(eff["l2_weight"], 1e-2, places=6)
        self.assertAlmostEqual(eff["conf_skip_thr"], -0.040, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
