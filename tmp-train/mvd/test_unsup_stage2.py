"""Unit tests for Stage-2 unsupervised learning primitives (unsup_stage2.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

_MVD_DIR = str(Path(__file__).resolve().parent)
if _MVD_DIR not in sys.path:
    sys.path.insert(0, _MVD_DIR)

import data  # noqa: E402
from model import MvdNet  # noqa: E402
from unsup_stage2 import (  # noqa: E402
    EMAModel,
    episodic_page_ln_tta,
    stage2_encoder_ema_consistency_loss,
)


def _make_tiny_mvdnet(use_gdcm: bool = True, seed: int = 0) -> MvdNet:
    torch.manual_seed(seed)
    vocab = len(data.VOCAB) + 3
    net = MvdNet(
        vocab=vocab,
        d_model=64,
        layers=2,
        heads=4,
        dim_ff=128,
        dropout=0.0,
        anisotropic=True,
        pretrained=False,
        use_gdcm=use_gdcm,
    )
    return net


class TestStage2EncoderEMAConsistency(unittest.TestCase):
    def test_exact_zero_loss_when_student_equals_teacher_and_inputs_match(self) -> None:
        student = _make_tiny_mvdnet(use_gdcm=True, seed=1)
        student.eval()
        teacher = EMAModel(student, decay=0.999)

        strip = torch.randn((2, 3, 32, 64), dtype=torch.float32)
        # 8x16 anisotropic stride -> h_mem=4, w_mem=4 -> S=16.
        mem_mask = torch.zeros((2, 16), dtype=torch.bool)
        mem_mask[0, 12:] = True

        loss, metrics = stage2_encoder_ema_consistency_loss(
            student=student,
            teacher_ema=teacher,
            strip_weak=strip,
            strip_strong=strip.clone(),
            mem_mask=mem_mask,
            return_metrics=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss.item()), 1e-6)
        self.assertLess(metrics["unsup_s2_mem_cos"], 1e-6)
        self.assertLess(metrics["unsup_s2_gdcm"], 1e-6)

    def test_grads_only_on_student_encoder_and_gdcm(self) -> None:
        student = _make_tiny_mvdnet(use_gdcm=True, seed=2)
        student.train()
        teacher = EMAModel(student, decay=0.999)

        strip_weak = torch.randn((2, 3, 32, 64), dtype=torch.float32)
        # Photometric-only jitter (1-to-1 spatial alignment, geom=0).
        strip_strong = strip_weak * 0.85 + 0.12 + 0.05 * torch.randn_like(strip_weak)

        mem_mask = torch.zeros((2, 16), dtype=torch.bool)
        mem_mask[1, 14:] = True

        loss = stage2_encoder_ema_consistency_loss(
            student=student,
            teacher_ema=teacher,
            strip_weak=strip_weak,
            strip_strong=strip_strong,
            mem_mask=mem_mask,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.ndim, 0)
        self.assertGreater(float(loss.item()), 0.0)
        loss.backward()

        # Student encoder, proj, mem_norm, and gdcm_head MUST receive gradients.
        for mod_name in ("encoder", "proj", "mem_norm", "gdcm_head"):
            mod = getattr(student, mod_name)
            grad_sum = sum(
                float(p.grad.abs().sum().item())
                for p in mod.parameters()
                if p.grad is not None
            )
            self.assertGreater(grad_sum, 0.0, f"expected positive grad on student.{mod_name}")

        # Student decoder, tok_emb, out, and aux_head MUST NOT receive gradients.
        for mod_name in ("decoder", "tok_emb", "out", "aux_head"):
            mod = getattr(student, mod_name)
            for p in mod.parameters():
                self.assertIsNone(p.grad, f"unexpected grad on student.{mod_name}")

        # Teacher parameters MUST NOT receive gradients.
        for p in teacher.module.parameters():
            self.assertIsNone(p.grad, "unexpected grad on frozen teacher_ema")


class TestEpisodicPageLnTTA(unittest.TestCase):
    def test_updates_only_ln_inside_and_restores_exact_weights_on_exit(self) -> None:
        model = _make_tiny_mvdnet(use_gdcm=False, seed=3)
        model.train()

        # Snapshot every parameter and its requires_grad flag before TTA.
        orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        orig_rg = {k: p.requires_grad for k, p in model.named_parameters()}
        ln_param_ids = {
            id(p)
            for m in model.modules()
            if isinstance(m, nn.LayerNorm)
            for p in (m.weight, m.bias)
            if p is not None
        }
        self.assertGreater(len(ln_param_ids), 0)

        page_strips = torch.randn((2, 3, 32, 64), dtype=torch.float32)

        # min_conf=0.0 ensures confidence gate fires on randomly initialized logits.
        with episodic_page_ln_tta(
            model, page_strips, steps=2, lr=1e-2, min_conf=0.0, max_len=4
        ) as adapted:
            self.assertIs(adapted, model)
            self.assertFalse(adapted.training)

            ln_changed = False
            for name, p in adapted.named_parameters():
                if id(p) in ln_param_ids:
                    if not torch.equal(p.detach(), orig_state[name]):
                        ln_changed = True
                else:
                    self.assertTrue(
                        torch.equal(p.detach(), orig_state[name]),
                        f"non-LayerNorm param {name} was mutated during TTA",
                    )
            self.assertTrue(ln_changed, "expected at least one LayerNorm param to adapt")

        # After exit: ALL weights, requires_grad flags, and training mode restored bit-for-bit.
        self.assertTrue(model.training)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, orig_state[k]), f"param {k} not restored on exit")
        for k, p in model.named_parameters():
            self.assertEqual(p.requires_grad, orig_rg[k], f"requires_grad for {k} not restored")

    def test_restores_exact_weights_on_exception_and_list_strips(self) -> None:
        model = _make_tiny_mvdnet(use_gdcm=False, seed=4)
        model.eval()
        orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        strips_list = [
            torch.randn((3, 32, 64), dtype=torch.float32),
            torch.randn((1, 3, 32, 80), dtype=torch.float32),
        ]

        with self.assertRaises(RuntimeError):
            with episodic_page_ln_tta(
                model, strips_list, steps=1, lr=1e-2, min_conf=0.0, max_len=3
            ):
                raise RuntimeError("simulated decode failure")

        self.assertFalse(model.training)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, orig_state[k]), f"param {k} not restored after exception")


if __name__ == "__main__":
    unittest.main()
