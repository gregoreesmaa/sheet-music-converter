#!/usr/bin/env python3
"""CPU-only unit tests for Bucket C zero-GPU inference, FSA grammar & neg_control_gate fixes.

Covers all 5 fixes:
  1. FSA Header-Loop & consecutive `time-*` guard (`foster-leadsheet`).
  2. FSA Tuplet balance enforcement (`luca-gloria` photo) + 45/45 `suite31_hw` GT safety.
  3. Near-verbatim (>=0.90) duplicate trailing `<sys>` truncation + >=3 consecutive
     normalized multi-voice bar-hash loop guard (`detect_token_loop` & `cycle_forbidden`).
  4. `neg_control_gate.py` skew-fallback expansion `(-9, -6, -3, 3, 6, 9)` deg
     (4/4 NorthStar negative controls still reject; rotated `debussy-lindaraja` at +-8 deg accepted).
  5. Eval `--max-mem-tokens` override (default 4096, v2 override + restore, v1 safe no-op).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = MVD_DIR.parent.parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

from PIL import Image  # noqa: E402
import torch  # noqa: E402

import data  # noqa: E402
import eval_sys_gate as esg  # noqa: E402
import eval_uni_suite31 as ev  # noqa: E402
import grammar  # noqa: E402
import neg_control_gate as ncg  # noqa: E402
import uni_omr  # noqa: E402
import uni_omr_v2  # noqa: E402

_T = data.TOKEN_TO_ID


class TestBucketCInferenceFixes(unittest.TestCase):
    # ------------------------------------------------------------------
    # Fix 1: FSA Header-Loop & Duplicate time-* Guard
    # ------------------------------------------------------------------

    def test_fix1_consecutive_time_tokens_forbidden(self) -> None:
        """Forbid consecutive time-* tokens in header and mid-stream (both lax and strict)."""
        for strict in (False, True):
            st = grammar.PrefixState(strict=strict)
            st.step("clef-G-2")
            st.step("key+0")
            st.step("time-4-4")
            allow, _ = st.allowed()
            for t in grammar.TIMES:
                if t in _T:
                    self.assertNotIn(_T[t], allow, f"strict={strict}: consecutive {t} allowed")
            with self.assertRaises(ValueError):
                st.step("time-4-4")
            with self.assertRaises(ValueError):
                st.step("time-3-4")

            # After a note + bar, a time change is allowed again, but NOT two in a row.
            for tok in ("quarter", "C", "4", "quarter", "D", "4", "quarter", "E", "4", "quarter", "F", "4", "bar"):
                st.step(tok)
            allow_after_bar, _ = st.allowed()
            self.assertIn(_T["time-3-4"], allow_after_bar)
            st.step("time-3-4")
            allow_after_time, _ = st.allowed()
            self.assertNotIn(_T["time-3-4"], allow_after_time)
            with self.assertRaises(ValueError):
                st.step("time-4-4")

    def test_fix1_header_staff_cap_and_single_staff_repeat_guard(self) -> None:
        """Cap system-header staff count at max(first_sys_staves, 8) and forbid >2x single-staff header loop."""
        # System 0 cap: up to 8 staves allowed, 9th blocked.
        st = grammar.PrefixState(strict=False)
        for _ in range(8):
            st.step("clef-G-2")
            st.step("key+0")
            st.step("time-4-4")
        allow, _ = st.allowed()
        self.assertNotIn(_T["clef-G-2"], allow)
        self.assertNotIn(_T["key+0"], allow)
        self.assertNotIn(_T["time-4-4"], allow)
        with self.assertRaises(ValueError):
            st.step("clef-G-2")
        # Notes are still allowed to exit header mode.
        self.assertIn(_T["quarter"], allow)

        # Single-staff piece: system 0 has 1 staff -> system 1 caps identical header loop at 2x.
        st1 = grammar.PrefixState(strict=False)
        for tok in ("clef-G-2", "key+0", "time-4-4", "quarter", "C", "4", "bar"):
            st1.step(tok)
        self.assertEqual(st1.first_sys_staves, 1)
        st1.reset_system()
        for _ in range(2):
            st1.step("clef-G-2")
            st1.step("key+0")
            st1.step("time-4-4")
        allow_sys1, _ = st1.allowed()
        self.assertNotIn(_T["clef-G-2"], allow_sys1)
        self.assertNotIn(_T["key+0"], allow_sys1)
        self.assertNotIn(_T["time-4-4"], allow_sys1)
        with self.assertRaises(ValueError):
            st1.step("clef-G-2")
        self.assertIn(_T["quarter"], allow_sys1)

    # ------------------------------------------------------------------
    # Fix 2: FSA Tuplet Balance Enforcement
    # ------------------------------------------------------------------

    def test_fix2_tuplet_balance_forbids_voice_bar_sys_eos(self) -> None:
        """While tuplet-start-* is queued or open, voice-*, bar*, <sys>, <eos> are forbidden until tuplet-stop."""
        vocab_size = len(data.VOCAB) + 4
        _, _, eos_id, sys_id, tok_to_id, _ = ev._vocab_spec(vocab_size)

        for strict in (False, True):
            st = grammar.PrefixState(strict=strict, vocab_map=tok_to_id)
            for tok in ("clef-G-2", "key+0", "time-4-4", "voice-1", "tuplet-start-3-2"):
                st.step(tok)

            # Pending tuplet (before first note): voice-*, bar*, <sys>, <eos> forbidden.
            self.assertTrue(st.has_open_tuplet())
            allow, eos_ok = st.allowed()
            self.assertFalse(eos_ok)
            self.assertFalse(st.can_emit_sys())
            self.assertFalse(st.can_emit_eos())
            self.assertNotIn(_T["bar"], allow)
            self.assertNotIn(_T["voice-2"], allow)
            allowed_ids = ev._allowed_ids_for_state(st, vocab_size)
            self.assertNotIn(eos_id, allowed_ids)
            self.assertNotIn(sys_id, allowed_ids)
            with self.assertRaises(ValueError):
                st.step("bar")
            with self.assertRaises(ValueError):
                st.step("voice-2")

            # Inside open tuplet (depth > 0 after notes): still forbidden until tuplet-stop.
            for tok in ("eighth", "C", "4", "eighth", "D", "4", "eighth", "E", "4"):
                st.step(tok)
            self.assertTrue(st.has_open_tuplet())
            allow2, eos_ok2 = st.allowed()
            self.assertFalse(eos_ok2)
            self.assertFalse(st.can_emit_sys())
            self.assertFalse(st.can_emit_eos())
            self.assertNotIn(_T["bar"], allow2)
            self.assertNotIn(_T["bar-light-heavy"], allow2)
            self.assertNotIn(_T["voice-2"], allow2)
            self.assertIn(_T["tuplet-stop"], allow2)
            allowed_ids2 = ev._allowed_ids_for_state(st, vocab_size)
            self.assertNotIn(eos_id, allowed_ids2)
            self.assertNotIn(sys_id, allowed_ids2)

            # Close tuplet -> bar, voice-2, <sys>, <eos> legal again!
            st.step("tuplet-stop")
            self.assertFalse(st.has_open_tuplet())
            allow3, eos_ok3 = st.allowed()
            self.assertTrue(eos_ok3)
            self.assertTrue(st.can_emit_sys())
            self.assertTrue(st.can_emit_eos())
            self.assertIn(_T["bar"], allow3)
            self.assertIn(_T["voice-2"], allow3)
            allowed_ids3 = ev._allowed_ids_for_state(st, vocab_size)
            self.assertIn(eos_id, allowed_ids3)
            self.assertIn(sys_id, allowed_ids3)

    def test_fix2_nested_tuplet_starts_forbidden_both_modes(self) -> None:
        """Forbid chained/nested tuplet-start-* in both lax and strict modes (prevents cpebach-h186 zero-tick spiral)."""
        for strict in (False, True):
            st = grammar.PrefixState(strict=strict)
            for tok in ("clef-G-2", "key+3", "time-4-4", "voice-1", "tuplet-start-3-2"):
                st.step(tok)
            # While pending > 0 (before first note), a second tuplet-start-* must be masked & rejected.
            allow_pend, _ = st.allowed()
            for t in grammar.TUPLET_STARTS:
                if t in _T:
                    self.assertNotIn(_T[t], allow_pend, f"strict={strict}: pending nested {t} allowed")
            with self.assertRaises(ValueError):
                st.step("tuplet-start-3-2")

            # After entering the tuplet (depth > 0), nested tuplet-start-* must still be masked & rejected.
            for tok in ("16th", "B", "4"):
                st.step(tok)
            self.assertEqual(st.depth[st.seg], 1)
            allow_open, _ = st.allowed()
            for t in grammar.TUPLET_STARTS:
                if t in _T:
                    self.assertNotIn(_T[t], allow_open, f"strict={strict}: open nested {t} allowed")
            with self.assertRaises(ValueError):
                st.step("tuplet-start-6-4")

            # Once closed with tuplet-stop (depth == 0), a new tuplet-start-* is allowed again.
            st.step("tuplet-stop")
            self.assertEqual(st.depth[st.seg], 0)
            allow_closed, _ = st.allowed()
            self.assertIn(_T["tuplet-start-3-2"], allow_closed)
            st.step("tuplet-start-3-2")
            self.assertEqual(st.pending, 1)

    def test_fix2_all_45_suite31_hw_gt_sequences_accepted(self) -> None:
        """All 45 suite31_hw GT sequences pass updated PrefixState(strict=False) with 0 blocks."""
        mpath = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"
        self.assertTrue(mpath.is_file(), f"Missing manifest: {mpath}")
        _, pieces = ev.resolve_manifest_pieces(mpath)
        self.assertEqual(len(pieces), 45)

        with tempfile.TemporaryDirectory(prefix="test_bucketc_gt45_") as tmpdir:
            wd = Path(tmpdir)
            for p in pieces:
                slug = p["slug"]
                toks, _ = esg.suite31_page_refs(p["gt"], "bc-" + slug, wd)
                has_sys = "<sys>" in toks
                st = grammar.PrefixState(
                    strict=False,
                    max_bars_per_system=grammar.MAX_BARS_PER_SYSTEM if has_sys else None,
                    max_tokens_per_system=grammar.MAX_TOKENS_PER_SYSTEM if has_sys else None,
                )
                for idx, tok in enumerate(toks):
                    if tok == "<sys>":
                        self.assertTrue(st.can_emit_sys(), f"{slug}@{idx}: can_emit_sys() False")
                        st.reset_system()
                        continue
                    allow, _ = st.allowed()
                    self.assertIn(
                        _T[tok],
                        allow,
                        f"{slug}@{idx}: GT token {tok!r} blocked (open_tup={st.has_open_tuplet()})",
                    )
                    # Also verify cycle_forbidden() never blocks a GT continuation!
                    cyc = st.cycle_forbidden()
                    self.assertNotEqual(
                        cyc,
                        tok,
                        f"{slug}@{idx}: cycle_forbidden() false-blocked GT token {tok!r}",
                    )
                    st.step(tok)
                self.assertFalse(st.broken, f"{slug}: broken=True")
                self.assertFalse(st.has_open_tuplet(), f"{slug}: unclosed tuplet at EOF")
                self.assertTrue(st.can_emit_eos(), f"{slug}: can_emit_eos() False at EOF")

    # ------------------------------------------------------------------
    # Fix 3: Duplicate <sys> & Consecutive Multi-Voice Bar-Hash Guard
    # ------------------------------------------------------------------

    def test_fix3_truncate_duplicate_trailing_systems(self) -> None:
        """Truncate verbatim/near-verbatim (>=0.90) trailing <sys> when prev system ended on a barline."""
        sys1 = [
            "clef-G-2", "key+0", "time-6-8",
            "eighth", "D", "4", "eighth", "F", "4", "eighth", "A", "4", "bar",
            "eighth", "G", "4", "eighth", "B", "4", "eighth", "D", "5", "bar",
            "eighth", "A", "4", "eighth", "F", "4", "eighth", "D", "4", "bar-light-heavy",
        ]
        # Near-verbatim duplicate (1 pitch altered -> ~95% token similarity)
        sys2_near_dup = list(sys1)
        sys2_near_dup[-3] = "E"
        raw = sys1 + ["<sys>"] + sys2_near_dup
        trunc = ev.truncate_duplicate_trailing_systems(raw, sim_threshold=0.90)
        self.assertEqual(trunc, sys1)

        # Stitch also truncates and reports n_sys=1
        _, n_sys = ev.stitch_uni_tokens(raw)
        self.assertEqual(n_sys, 1)

        # Distinct second system is preserved
        sys2_distinct = [
            "clef-G-2", "key+0", "time-6-8",
            "quarter", "C", "5", "eighth", "B", "4", "bar",
            "quarter", "A", "4", "eighth", "G", "4", "bar-light-heavy",
        ]
        raw_ok = sys1 + ["<sys>"] + sys2_distinct
        self.assertEqual(ev.truncate_duplicate_trailing_systems(raw_ok, sim_threshold=0.90), raw_ok)

        # Previous system NOT ending on a barline is NOT truncated
        sys1_nobar = sys1[:-1]
        raw_nobar = sys1_nobar + ["<sys>"] + sys2_near_dup
        self.assertEqual(ev.truncate_duplicate_trailing_systems(raw_nobar, sim_threshold=0.90), raw_nobar)

        # PrefixState.can_emit_sys() also rejects verbatim duplicate system during decoding
        st = grammar.PrefixState(strict=False)
        for tok in sys1:
            st.step(tok)
        self.assertTrue(st.can_emit_sys())
        st.reset_system()
        for tok in sys1:
            st.step(tok)
        self.assertFalse(st.can_emit_sys())

    def test_fix3_multivoice_bar_hash_loop_guard(self) -> None:
        """Detect >=3 consecutive identical normalized multi-voice bars in detect_token_loop and cycle_forbidden."""
        mv_bar = [
            "voice-1", "quarter", "C", "5", "quarter", "D", "5", "quarter", "E", "5", "quarter", "F", "5",
            "voice-2", "half", "C", "4", "half", "G", "3",
            "voice-3", "whole", "C", "3",
            "bar",
        ]
        # 2 consecutive identical multi-voice bars -> NOT a loop
        seq2 = ["clef-G-2", "key+0", "time-4-4"] + mv_bar * 2
        self.assertFalse(ev.detect_token_loop(seq2))

        # 3 consecutive identical multi-voice bars (even with cosmetic dyn perturbation) -> loop detected!
        mv_bar_cos = ["dyn-f"] + mv_bar
        seq3 = ["clef-G-2", "key+0", "time-4-4"] + mv_bar + mv_bar_cos + mv_bar
        self.assertTrue(ev.detect_token_loop(seq3))

        # PrefixState.cycle_forbidden() allows a DIFFERENT bar starting with voice-1 after 3 repeats,
        # but forbids continuing a 4th verbatim copy once 6 tokens match!
        st = grammar.PrefixState(strict=False)
        for tok in seq3:
            st.step(tok)
        self.assertEqual(st.mv_bar_consec, 3)
        # Starting a new bar with voice-1 is NOT blocked
        self.assertIsNone(st.cycle_forbidden())
        for tok in mv_bar[:6]:
            st.step(tok)
        self.assertEqual(st.cycle_forbidden(), mv_bar[6])

    # ------------------------------------------------------------------
    # Fix 4: neg_control_gate.py Skew-Fallback Expansion
    # ------------------------------------------------------------------

    def test_fix4_neg_control_gate_real_northstar_and_rotated_hwm(self) -> None:
        """All 4 NorthStar negative controls still reject; rotated debussy-lindaraja at +-8 deg accepted."""
        neg_dir = Path(r"D:\tmp\northstar-eval-pack\neg_controls")
        if neg_dir.is_dir():
            neg_files = sorted(neg_dir.glob("*.png"))
            self.assertEqual(len(neg_files), 4, f"Expected 4 NorthStar neg controls, got {neg_files}")
            for nf in neg_files:
                with Image.open(nf) as im:
                    rejected, info = ncg.is_non_music_page(im)
                self.assertTrue(rejected, f"NorthStar neg control {nf.name} falsely accepted: {info}")

        # Test rotated sparse HWM & single-staff pages at +-8 deg
        sparse_pages = [
            REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "hw" / "photo" / "hw-debussy-lindaraja.png",
            REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "clean" / "essen-ballad.png",
            REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "clean" / "repeats-melody.png",
        ]
        for sp in sparse_pages:
            if not sp.is_file():
                continue
            with Image.open(sp) as im:
                rgb = im.convert("RGB")
            for angle in (-8.0, 8.0):
                rot = rgb.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(245, 242, 235))
                rejected, info = ncg.is_non_music_page(rot)
                self.assertFalse(rejected, f"{sp.name} rotated {angle} deg falsely rejected: {info}")

    # ------------------------------------------------------------------
    # Fix 5: Eval --max-mem-tokens Override
    # ------------------------------------------------------------------

    def test_fix5_max_mem_tokens_cli_and_override(self) -> None:
        """--max-mem-tokens defaults to 4608, overrides v2 router.max_mem_tokens, and no-ops on v1."""
        ap = ev.build_arg_parser()
        ns_def = ap.parse_args(["--ckpt", "dummy.pt"])
        self.assertEqual(ns_def.max_mem_tokens, 4608)
        ns_custom = ap.parse_args(["--ckpt", "dummy.pt", "--max-mem-tokens", "5120"])
        self.assertEqual(ns_custom.max_mem_tokens, 5120)

        # UniOMRv2Net: router.max_mem_tokens is overridden and restored
        cfg_v2 = uni_omr_v2.UniOMRv2Config(
            d_model=64,
            backbone_channels=(16, 32, 48, 64),
            backbone_blocks=(1, 1, 1, 1),
            n_axial_layers=1,
            n_global_layers=1,
            n_dec_layers=1,
            n_heads=4,
            n_kv_heads=2,
            max_mem_tokens=256,
            n_bg_summary=2,
            mtp_k=1,
        )
        model_v2 = uni_omr_v2.UniOMRv2Net(cfg_v2).eval()
        router_v2 = ev._get_model_router(model_v2)
        self.assertEqual(router_v2.max_mem_tokens, 256)
        with ev.override_max_mem_tokens(model_v2, 4096):
            self.assertEqual(router_v2.max_mem_tokens, 4096)
        self.assertEqual(router_v2.max_mem_tokens, 256)

        # Legacy UniOMRNet v1: no router.max_mem_tokens -> safe no-op without error
        cfg_v1 = uni_omr.UniOMRConfig(
            d_model=64, n_enc_layers=1, n_dec_layers=1, n_heads=4, ffn_mult=2.0,
            dropout=0.0, keep_ratio=0.5, n_bg_summary=2, use_uv_aux=False, mtp_k=1,
        )
        model_v1 = uni_omr.UniOMRNet(cfg_v1).eval()
        with ev.override_max_mem_tokens(model_v1, 4096):
            self.assertIsNotNone(model_v1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
