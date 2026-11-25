#!/usr/bin/env python3
"""CPU-only unit tests for UniOMR FSA grammar decoding, MTP-4 self-speculative
decoding, and suite31_hw manifest resolution."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402
import eval_uni_suite31 as ev  # noqa: E402
import grammar  # noqa: E402
import uni_omr  # noqa: E402


def _make_tiny_model(
    seed: int = 123,
    mtp_k: int = 4,
    vocab_size: int | None = None,
) -> uni_omr.UniOMRNet:
    torch.manual_seed(seed)
    cfg_kwargs = dict(
        d_model=64,
        n_enc_layers=2,
        n_dec_layers=2,
        n_heads=4,
        ffn_mult=2.0,
        dropout=0.0,
        keep_ratio=0.5,
        n_bg_summary=2,
        use_uv_aux=False,
        mtp_k=mtp_k,
    )
    if vocab_size is not None:
        cfg_kwargs["vocab_size"] = vocab_size
    cfg = uni_omr.UniOMRConfig(**cfg_kwargs)
    model = uni_omr.UniOMRNet(cfg).to("cpu").eval()
    return model


class TestUniDecodeCPU(unittest.TestCase):
    def test_mtp_spec_exact_parity_single_and_batched(self) -> None:
        """generate_mtp_spec() must match model.generate() token-for-token on CPU."""
        for seed, batch_size, max_len in ((7, 1, 24), (19, 2, 18), (42, 1, 7)):
            model = _make_tiny_model(seed=seed, mtp_k=4)
            g = torch.Generator(device="cpu").manual_seed(seed + 100)
            imgs = torch.rand((batch_size, 3, 64, 64), generator=g)

            ref_seq = model.generate(imgs, max_len=max_len)
            spec_seq, stats = ev.generate_mtp_spec(
                model, imgs, max_len=max_len, use_grammar=False, return_stats=True
            )

            self.assertEqual(
                ref_seq.tolist(),
                spec_seq.tolist(),
                f"Token mismatch for seed={seed}, B={batch_size}, max_len={max_len}",
            )
            self.assertEqual(stats["mtp_k"], 4)
            self.assertGreaterEqual(stats["acceptance_rate"], 0.0)
            self.assertLessEqual(stats["acceptance_rate"], 1.0)
            self.assertGreater(stats["tokens_per_sec"], 0.0)

    def test_mtp_spec_high_acceptance_when_mtp_matches_lm_head(self) -> None:
        """When MTP heads predict the exact next tokens, acceptance_rate > 0 and parity holds."""
        model = _make_tiny_model(seed=55, mtp_k=4)
        # Force a constant token preference (id 0) so main head and MTP heads agree identically
        # at every position regardless of vocab_size or hidden-state sign.
        with torch.no_grad():
            model.lm_head.weight.zero_()
            for head in model.mtp_heads:
                head.proj.weight.zero_()

        imgs = torch.rand((1, 3, 64, 64))
        ref_seq = model.generate(imgs, max_len=16)
        spec_seq, stats = ev.generate_mtp_spec(
            model, imgs, max_len=16, use_grammar=False, return_stats=True
        )
        self.assertEqual(ref_seq.tolist(), spec_seq.tolist())
        self.assertGreater(stats["draft_accepted"], 0)
        self.assertAlmostEqual(stats["acceptance_rate"], 1.0, places=4)
        # With k=4 and 100% acceptance, 16 tokens take far fewer than 16 forward passes.
        self.assertLess(stats["verify_passes"], 10)

    def test_mtp_spec_with_grammar_exact_parity(self) -> None:
        """generate_mtp_spec(use_grammar=True) matches generate_constrained() token-for-token."""
        model = _make_tiny_model(seed=99, mtp_k=4)
        g = torch.Generator(device="cpu").manual_seed(202)
        imgs = torch.rand((1, 3, 64, 64), generator=g)

        ref_seq = ev.generate_constrained(model, imgs, max_len=20, strict=False)
        spec_seq, _ = ev.generate_mtp_spec(
            model, imgs, max_len=20, use_grammar=True, strict_grammar=False, return_stats=True
        )
        self.assertEqual(ref_seq.tolist(), spec_seq.tolist())

    def test_constrained_grammar_valid_and_sys_reset(self) -> None:
        """generate_constrained() emits grammar-legal bekern and resets PrefixState on <sys>."""
        model = _make_tiny_model(seed=31, mtp_k=4)
        g = torch.Generator(device="cpu").manual_seed(303)
        imgs = torch.rand((2, 3, 64, 64), generator=g)

        seq = ev.generate_constrained(model, imgs, max_len=24, strict=False)
        self.assertEqual(seq.shape[0], 2)
        for row in seq.tolist():
            toks = uni_omr.decode_uni(row, skip_special=True)
            # Every system chunk between <sys> markers must walk cleanly through PrefixState.
            st = grammar.PrefixState(strict=False)
            for tok in toks:
                if tok == uni_omr.SYS_TOKEN:
                    st = grammar.PrefixState(strict=False)
                else:
                    st.step(tok)  # Must not raise ValueError

        # Explicitly test <sys> state reset helper (e.g. bar -> <sys> -> fresh header).
        st = grammar.PrefixState(strict=False)
        for tok in ("clef-G-2", "time-4-4", "quarter", "C", "4", "bar"):
            st.step(tok)
        self.assertFalse(st.header_open)
        allowed = ev._allowed_ids_for_state(st, uni_omr.VOCAB_SIZE)
        self.assertIn(uni_omr.SYS_ID, allowed)
        self.assertIn(uni_omr.EOS_ID, allowed)

        st_reset = ev._advance_grammar_state(st, uni_omr.SYS_ID, strict=False)
        self.assertTrue(st_reset.header_open)
        self.assertEqual(st_reset.mode, "bound")
        self.assertEqual(st_reset.recent, [])

    def test_constrained_all_false_fallback_never_crashes(self) -> None:
        """All-NaN or empty-mask logits safely fall back to unconstrained argmax without crashing."""
        st = grammar.PrefixState(strict=True)
        # Force an impossible/empty-mask state (full bar in rest mode)
        st.tsig = 4.0
        st.vcurs = [4.0]
        st.mode = "rest"
        logits = torch.zeros(uni_omr.VOCAB_SIZE)
        logits[5] = 3.14
        picked = ev._select_constrained_token(logits, st)
        self.assertEqual(picked, 5)

        # Also test NaN logits inside allowed subset
        st2 = grammar.PrefixState(strict=False)
        bad_logits = torch.full((uni_omr.VOCAB_SIZE,), float("-inf"))
        bad_logits[uni_omr.PAD_ID] = 1.0  # only PAD is finite (excluded from allowed)
        picked2 = ev._select_constrained_token(bad_logits, st2)
        self.assertEqual(picked2, uni_omr.PAD_ID)

    def test_suite31_hw_manifest_resolves_all_45_pieces(self) -> None:
        """All 45 pieces (31 printed + 14 HW) in suite31_hw_manifest.json must resolve on disk."""
        self.assertTrue(ev.DEFAULT_MANIFEST.is_file(), f"Missing manifest: {ev.DEFAULT_MANIFEST}")
        _, pieces = ev.resolve_manifest_pieces(ev.DEFAULT_MANIFEST)
        self.assertEqual(len(pieces), 45)
        n_hw = sum(1 for p in pieces if p["handwritten"])
        n_pr = sum(1 for p in pieces if not p["handwritten"])
        self.assertEqual(n_pr, 31)
        self.assertEqual(n_hw, 14)

        for p in pieces:
            for key in ("clean", "photo", "gt"):
                path = p[key]
                self.assertTrue(
                    path.is_file(),
                    f"Missing {key} file for piece {p['slug']}: {path}",
                )

    def test_iter7_d1_vocab_and_grammar_acceptance(self) -> None:
        """Option D1 tokens live after index 177 (preserving 0..177) and walk PrefixState."""
        # Prefix 0..177 preserved verbatim
        self.assertEqual(data.VOCAB[175], "128th")
        self.assertEqual(data.VOCAB[176], "clef-G-2-8vb")
        self.assertEqual(data.VOCAB[177], "clef-G-2-8va")
        self.assertEqual(len(data.VOCAB), len(set(data.VOCAB)))
        self.assertEqual(data.PAD_ID, len(data.VOCAB))
        self.assertEqual(data.BOS_ID, len(data.VOCAB) + 1)
        self.assertEqual(data.EOS_ID, len(data.VOCAB) + 2)
        self.assertEqual(data.VOCAB_SIZE, len(data.VOCAB) + 3)
        self.assertEqual(data.ID_TO_TOKEN[data.TOKEN_TO_ID["voice-13"]], "voice-13")

        samples = ("voice-13", "voice-27", "clef-T-2", "clef-F-4-8vb", "time-13-8",
                   "tuplet-start-17-16", "-1")
        for tok in samples:
            self.assertIn(tok, data.TOKEN_TO_ID)
            self.assertGreaterEqual(data.TOKEN_TO_ID[tok], 178)

        # Walk a stream exercising all Option D1 token families in both lax & strict modes.
        voices_1_to_13 = []
        for v in range(1, 14):
            voices_1_to_13.extend([f"voice-{v}", "eighth", "C", "-1"])
        stream = (
            ["clef-T-2", "key+0", "time-13-8"]
            + voices_1_to_13
            + ["bar", "clef-F-4-8vb", "voice-1", "tuplet-start-17-16",
               "16th", "A", "-1", "tuplet-stop", "bar"]
        )
        for strict in (False, True):
            st = grammar.PrefixState(strict=strict)
            for tok in stream:
                allow, _ = st.allowed()
                self.assertIn(data.TOKEN_TO_ID[tok], allow, f"strict={strict} blocked {tok!r}")
                st.step(tok)

    def test_chord_grace_and_bar_rest_anti_loop_caps(self) -> None:
        """CHORD capped at <= 8 per onset, GRACE <= 8 consecutive, BAR_REST forbidden on full/non-empty bar."""
        chord_id = data.TOKEN_TO_ID["chord"]
        grace_id = data.TOKEN_TO_ID["grace"]
        bar_rest_id = data.TOKEN_TO_ID["bar-rest"]

        # 1. CHORD cap <= 8 per onset (both strict=False and strict=True) + resets on new note
        for strict in (False, True):
            st = grammar.PrefixState(strict=strict)
            for tok in ("clef-G-2", "key+0", "time-4-4", "quarter", "C", "4"):
                st.step(tok)
            for _ in range(8):
                self.assertIn(chord_id, st.allowed()[0])
                for tok in ("chord", "quarter", "E", "4"):
                    st.step(tok)
            # 9th chord on same onset must be masked and rejected
            self.assertNotIn(chord_id, st.allowed()[0])
            with self.assertRaises(ValueError):
                st.step("chord")
            # A fresh non-chord note resets chords_on_onset to 0
            for tok in ("quarter", "D", "4"):
                st.step(tok)
            self.assertEqual(st.chords_on_onset, 0)
            self.assertIn(chord_id, st.allowed()[0])

        # 2. GRACE cap <= 8 consecutive (both strict=False and strict=True)
        for strict in (False, True):
            st = grammar.PrefixState(strict=strict)
            for tok in ("clef-G-2", "key+0", "time-4-4"):
                st.step(tok)
            for _ in range(8):
                self.assertIn(grace_id, st.allowed()[0])
                for tok in ("grace", "16th", "C", "5"):
                    st.step(tok)
            self.assertNotIn(grace_id, st.allowed()[0])
            with self.assertRaises(ValueError):
                st.step("grace")

        # 3. BAR_REST forbidden when bar has notes or is full (strict=True)
        st_full = grammar.PrefixState(strict=True)
        for tok in ("clef-G-2", "key+0", "time-4-4", "quarter", "C", "4"):
            st_full.step(tok)
        self.assertNotIn(bar_rest_id, st_full.allowed()[0])
        with self.assertRaises(ValueError):
            st_full.step("bar-rest")

    def test_period16_cycle_broken_in_both_strict_and_lax(self) -> None:
        """Period-16 cycle must be broken by cycle_forbidden() in BOTH strict=False and strict=True."""
        # Period-16 bar cycle: 4 quarter notes + bar = 13 tokens, or 5 eighth notes + bar = 16 tokens
        bar_cycle = [
            "eighth", "C", "4",
            "eighth", "D", "4",
            "eighth", "E", "4",
            "eighth", "F", "4",
            "eighth", "G", "4",
            "bar",
        ]
        self.assertEqual(len(bar_cycle), 16)
        for strict in (False, True):
            st = grammar.PrefixState(strict=strict)
            for tok in ("clef-G-2", "key+0", "time-4-4"):
                st.step(tok)
            for _ in range(12):
                for tok in bar_cycle:
                    st.step(tok)
            self.assertEqual(st.cycle_forbidden(), "eighth")
            allowed = ev._allowed_ids_for_state(st, uni_omr.VOCAB_SIZE)
            self.assertNotIn(data.TOKEN_TO_ID["eighth"], allowed)

    def test_backward_compat_legacy_182_vocab_decoding(self) -> None:
        """Legacy vocab_size=182 checkpoints decode cleanly with pad=178, bos=179, eos=180, sys=181."""
        legacy_vsz = 182
        pad_id, bos_id, eos_id, sys_id, tok_to_id, id_to_tok = ev._vocab_spec(legacy_vsz)
        self.assertEqual((pad_id, bos_id, eos_id, sys_id), (178, 179, 180, 181))
        self.assertEqual(len(tok_to_id), 178)
        self.assertEqual(id_to_tok[177], "clef-G-2-8va")
        self.assertEqual(id_to_tok[sys_id], uni_omr.SYS_TOKEN)

        # Decode a legacy-ID sequence including <sys> and EOS=180
        legacy_ids = [
            bos_id,
            tok_to_id["clef-G-2"], tok_to_id["time-4-4"],
            tok_to_id["quarter"], tok_to_id["C"], tok_to_id["4"], tok_to_id["bar"],
            sys_id,
            tok_to_id["clef-G-2-8vb"], tok_to_id["quarter"], tok_to_id["D"], tok_to_id["4"],
            eos_id, pad_id,
        ]
        self.assertEqual(
            ev.decode_uni_compat(legacy_ids, vocab_size=legacy_vsz, skip_special=True),
            ["clef-G-2", "time-4-4", "quarter", "C", "4", "bar",
             "<sys>", "clef-G-2-8vb", "quarter", "D", "4"],
        )

        # End-to-end generate_constrained + generate_mtp_spec parity on a legacy vocab_size=182 model
        model = _make_tiny_model(seed=77, mtp_k=4, vocab_size=legacy_vsz)
        g = torch.Generator(device="cpu").manual_seed(777)
        imgs = torch.rand((1, 3, 64, 64), generator=g)
        seq_con = ev.generate_constrained(model, imgs, max_len=20, strict=False)
        seq_mtp, _ = ev.generate_mtp_spec(
            model, imgs, max_len=20, use_grammar=True, strict_grammar=False, return_stats=True
        )
        self.assertEqual(seq_con.tolist(), seq_mtp.tolist())
        self.assertEqual(int(seq_con[0, 0].item()), bos_id)
        self.assertTrue(all(0 <= int(x) < legacy_vsz for x in seq_con[0].tolist()))

    def test_trim_truncated_tokens_clean_boundaries(self) -> None:
        """trim_truncated_tokens cuts incomplete note/chord/voice tails back to complete event/bar."""
        self.assertEqual(
            ev.trim_truncated_tokens(
                ["clef-G-2", "time-4-4", "quarter", "C", "4", "quarter", "D"]
            ),
            ["clef-G-2", "time-4-4", "quarter", "C", "4"],
        )
        self.assertEqual(
            ev.trim_truncated_tokens(
                ["clef-G-2", "time-4-4", "quarter", "C", "4", "bar", "voice-2"]
            ),
            ["clef-G-2", "time-4-4", "quarter", "C", "4", "bar"],
        )
        self.assertEqual(
            ev.trim_truncated_tokens(
                ["clef-G-2", "time-4-4", "quarter", "C", "4", "chord", "quarter"]
            ),
            ["clef-G-2", "time-4-4", "quarter", "C", "4"],
        )
        self.assertEqual(
            ev.trim_truncated_tokens(
                ["clef-G-2", "time-4-4", "quarter", "C", "4", "bar", "<sys>", "clef-G-2"]
            ),
            ["clef-G-2", "time-4-4", "quarter", "C", "4", "bar"],
        )

    def test_beam_size_1_matches_generate_constrained_exactly(self) -> None:
        """generate_beam(beam_size=1) matches generate_constrained() token-for-token."""
        for seed, batch_size, strict, vocab_size in (
            (17, 1, False, None),
            (29, 2, False, None),
            (41, 1, True, None),
            (77, 1, False, 182),
        ):
            model = _make_tiny_model(seed=seed, mtp_k=4, vocab_size=vocab_size)
            g = torch.Generator(device="cpu").manual_seed(seed + 500)
            imgs = torch.rand((batch_size, 3, 64, 64), generator=g)

            ref_seq = ev.generate_constrained(model, imgs, max_len=20, strict=strict)
            beam1_seq = ev.generate_beam(
                model,
                imgs,
                max_len=20,
                beam_size=1,
                length_penalty=0.6,
                use_grammar=True,
                strict_grammar=strict,
            )
            self.assertEqual(
                ref_seq.tolist(),
                beam1_seq.tolist(),
                f"beam_size=1 diverged from generate_constrained (seed={seed}, B={batch_size}, strict={strict}, vocab={vocab_size})",
            )

    def test_beam_size_3_grammar_valid_encodes_once_and_beats_or_ties_greedy_logprob(self) -> None:
        """beam_size=3 encodes once, emits valid FSA sequences, and achieves >= greedy log-prob."""
        model = _make_tiny_model(seed=101, mtp_k=4)
        g = torch.Generator(device="cpu").manual_seed(909)
        imgs = torch.rand((1, 3, 64, 64), generator=g)

        encode_calls = {"n": 0}
        orig_encode = model.encode_image

        def _spy_encode(*args, **kwargs):
            encode_calls["n"] += 1
            return orig_encode(*args, **kwargs)

        model.encode_image = _spy_encode  # type: ignore[method-assign]
        try:
            seq_b1 = ev.generate_beam(
                model, imgs, max_len=20, beam_size=1, length_penalty=0.6, use_grammar=True, strict_grammar=False
            )
            seq_b3 = ev.generate_beam(
                model, imgs, max_len=20, beam_size=3, length_penalty=0.6, use_grammar=True, strict_grammar=False
            )
        finally:
            model.encode_image = orig_encode  # type: ignore[method-assign]

        # 1 encode_image call per generate_beam call (2 total)
        self.assertEqual(encode_calls["n"], 2)

        # Verify beam=3 sequence is 100% grammar-valid across <sys> boundaries
        toks_b3 = ev.decode_uni_compat(seq_b3[0].tolist(), vocab_size=model.cfg.vocab_size, skip_special=True)
        st = grammar.PrefixState(strict=False)
        for tok in toks_b3:
            if tok == uni_omr.SYS_TOKEN:
                st = grammar.PrefixState(strict=False)
            else:
                st.step(tok)  # Must not raise ValueError

        # Score both sequences with an independent uncached full-sequence forward pass
        # to verify both KV-cache correctness and that beam=3 >= beam=1 log-prob.
        def _seq_scores(seq_t: torch.Tensor) -> tuple[float, float]:
            with torch.no_grad():
                mem, _, _ = model.encode_image(imgs, return_aux=False)
                logits, _ = model.decode_tokens(seq_t[:, :-1], mem, return_mtp=False)
                lps = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                tgt = seq_t[:, 1:]
                tok_lps = lps.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                raw = float(tok_lps.sum().item())
                norm = ev._beam_norm_score(raw, int(tgt.shape[1]), 0.6)
                return raw, norm

        raw_b1, norm_b1 = _seq_scores(seq_b1)
        raw_b3, norm_b3 = _seq_scores(seq_b3)
        self.assertGreaterEqual(
            norm_b3,
            norm_b1 - 1e-5,
            f"beam=3 norm log-prob ({norm_b3:.4f}) < beam=1 ({norm_b1:.4f})",
        )
        if seq_b1.shape[1] == seq_b3.shape[1]:
            self.assertGreaterEqual(
                raw_b3,
                raw_b1 - 1e-5,
                f"beam=3 raw log-prob ({raw_b3:.4f}) < beam=1 ({raw_b1:.4f})",
            )

    def test_max_keep_ratio_override_changes_routed_token_count_and_restores(self) -> None:
        """override_max_keep_ratio changes dynamic-router k_kept cleanly and restores original value."""
        torch.manual_seed(13)
        cfg = uni_omr.UniOMRConfig(
            d_model=64,
            n_enc_layers=1,
            n_dec_layers=1,
            n_heads=4,
            ffn_mult=2.0,
            dropout=0.0,
            keep_ratio=0.40,
            dynamic_routing=True,
            min_keep_ratio=0.10,
            max_keep_ratio=0.30,
            saliency_thr=-0.5,  # force dense routing so k_dyn hits max_keep_ratio cap
            n_bg_summary=2,
            use_uv_aux=False,
            mtp_k=2,
        )
        model = uni_omr.UniOMRNet(cfg).to("cpu").eval()
        # Synthetic high-contrast ink-dense image (128x128 -> 8x8 = 64 grid tokens)
        img = torch.full((1, 3, 128, 128), 0.95)
        img[:, :, 16:112, 16:112] = 0.05

        with torch.no_grad():
            _, _, info_default = model.encode_image(img, return_aux=False)
            with ev.override_max_keep_ratio(model, 0.75):
                self.assertAlmostEqual(model.router.max_keep_ratio, 0.75)
                _, _, info_high = model.encode_image(img, return_aux=False)
            # Original value restored cleanly on exit
            self.assertAlmostEqual(model.router.max_keep_ratio, 0.30)

        self.assertGreater(info_high["k_kept"], info_default["k_kept"])
        self.assertGreater(info_high["n_out"], info_default["n_out"])

        # Invalid override rejected cleanly
        with self.assertRaises(ValueError):
            with ev.override_max_keep_ratio(model, 0.0):
                pass
        with self.assertRaises(ValueError):
            with ev.override_max_keep_ratio(model, 1.5):
                pass

        # CLI defaults and parsing check
        ap = ev.build_arg_parser()
        ns_def = ap.parse_args(["--ckpt", "dummy.pt"])
        self.assertEqual(ns_def.beam_size, 1)
        self.assertAlmostEqual(ns_def.beam_length_penalty, 0.6)
        self.assertIsNone(ns_def.max_keep_ratio_override)

        ns_custom = ap.parse_args([
            "--ckpt", "dummy.pt",
            "--beam-size", "4",
            "--beam-length-penalty", "0.8",
            "--max-keep-ratio-override", "0.65",
        ])
        self.assertEqual(ns_custom.beam_size, 4)
        self.assertAlmostEqual(ns_custom.beam_length_penalty, 0.8)
        self.assertAlmostEqual(ns_custom.max_keep_ratio_override, 0.65)

    def test_p5_anti_loop_dup_voice_bar_rest_cap_lax_ceiling_and_short_time(self) -> None:
        """P5: duplicate voice-N, consecutive bar-rest cap, lax beat ceiling, and mid-bar time-1-16/1-32."""
        v1_id = data.TOKEN_TO_ID["voice-1"]
        bar_rest_id = data.TOKEN_TO_ID["bar-rest"]
        half_id = data.TOKEN_TO_ID["half"]
        t16_id = data.TOKEN_TO_ID["time-1-16"]
        t32_id = data.TOKEN_TO_ID["time-1-32"]

        # 1. Duplicate consecutive voice-N forbidden in BOTH strict=False and strict=True
        for strict in (False, True):
            st = grammar.PrefixState(strict=strict)
            for tok in ("clef-G-2", "key+0", "time-4-4", "voice-1"):
                st.step(tok)
            self.assertNotIn(v1_id, st.allowed()[0])
            with self.assertRaises(ValueError):
                st.step("voice-1")

        # 2. Consecutive pure-rest (bar-rest-only) bars capped at <= MAX_CONSECUTIVE_BAR_REST (8)
        for strict in (False, True):
            st = grammar.PrefixState(strict=strict)
            for tok in ("clef-G-2", "key+0", "time-4-4"):
                st.step(tok)
            for i in range(grammar.MAX_CONSECUTIVE_BAR_REST):
                self.assertIn(bar_rest_id, st.allowed()[0])
                # Multi-voice pure-rest bar counts as 1 bar, not 2
                for tok in ("voice-1", "bar-rest", "voice-2", "bar-rest", "bar"):
                    st.step(tok)
                self.assertEqual(st.consec_bar_rest_bars, i + 1)
            # 9th consecutive pure-rest bar must be blocked
            self.assertNotIn(bar_rest_id, st.allowed()[0])
            with self.assertRaises(ValueError):
                st.step("bar-rest")
            # A pitched note resets consecutive bar-rest count to 0
            for tok in ("quarter", "C", "4", "bar"):
                st.step(tok)
            self.assertEqual(st.consec_bar_rest_bars, 0)
            self.assertIn(bar_rest_id, st.allowed()[0])

        # 3. Lax (strict=False) beat ceiling: <= max(8.0, 2.5 * tsig) blocks 12x half G4 in 4/4
        st_lax = grammar.PrefixState(strict=False)
        for tok in ("clef-G-2", "key+0", "time-4-4"):
            st_lax.step(tok)
        # 5 half notes = 10.0 quarter-beats == 2.5 * 4.0 (allowed)
        for _ in range(5):
            self.assertIn(half_id, st_lax.allowed()[0])
            for tok in ("half", "G", "4"):
                st_lax.step(tok)
        # 6th half note = 12.0 > 10.0 (blocked!)
        self.assertNotIn(half_id, st_lax.allowed()[0])
        with self.assertRaises(ValueError):
            st_lax.step("half")

        # 4. Mid-bar time-1-32 / time-1-16 forbidden; immediately after bar allowed
        for strict in (False, True):
            st = grammar.PrefixState(strict=strict)
            for tok in ("clef-G-2", "key+0", "time-4-4", "quarter", "C", "4"):
                st.step(tok)
            allow = st.allowed()[0]
            self.assertNotIn(t16_id, allow)
            self.assertNotIn(t32_id, allow)
            with self.assertRaises(ValueError):
                st.step("time-1-32")
            st.step("bar")
            allow_after_bar = st.allowed()[0]
            self.assertIn(t16_id, allow_after_bar)
            self.assertIn(t32_id, allow_after_bar)
            st.step("time-1-32")

    def test_suite31_hw_all_45_gt_pass_lax_prefix_state(self) -> None:
        """All 45 suite31_hw GT bekern sequences must pass PrefixState(strict=False) with 0 rejections."""
        import subprocess
        import tempfile

        aser = Path(__file__).resolve().parents[2] / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize.exe"
        if not aser.is_file() or not ev.DEFAULT_MANIFEST.is_file():
            self.skipTest("assembly-serialize binary or suite31_hw manifest not present")

        _, pieces = ev.resolve_manifest_pieces(ev.DEFAULT_MANIFEST)
        self.assertEqual(len(pieces), 45)
        passed = 0
        with tempfile.TemporaryDirectory() as td:
            for p in pieces:
                slug = p["slug"]
                res = subprocess.run(
                    [str(aser), "bekern", "--input", str(p["gt"]), "--out", td, "--stem", slug, "--page", "1"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertEqual(res.returncode, 0)
                toks = (Path(td) / f"{slug}-bekern.txt").read_text(encoding="utf-8").split()
                st = grammar.PrefixState(strict=False)
                for k, tok in enumerate(toks):
                    allow, _ = st.allowed()
                    tid = data.TOKEN_TO_ID[tok]
                    self.assertIn(
                        tid,
                        allow,
                        f"{slug}: PrefixState(strict=False) blocked token[{k}]={tok!r} after {toks[max(0, k - 6):k]}",
                    )
                    st.step(tok)
                _, eos_ok = st.allowed()
                self.assertTrue(eos_ok, f"{slug}: EOS blocked at end of GT stream")
                passed += 1
        self.assertEqual(passed, 45)

    def test_iter9_ngram_repetition_guard_and_midbar_header_guard(self) -> None:
        """Iter-9 guards: >3x 8..16-token repeat blocked; mid-bar key/time blocked in incomplete bars."""
        # 1. 8..16-token pattern: 3 consecutive repeats allowed, 4th blocked.
        ostinato_12 = [
            "eighth", "C", "4",
            "eighth", "D", "4",
            "eighth", "E", "4",
            "eighth", "F", "4",
        ]
        self.assertEqual(len(ostinato_12), 12)
        st = grammar.PrefixState(strict=False)
        for tok in ("clef-G-2", "key+0", "time-4-4"):
            st.step(tok)
        # 1st and 2nd repeats: cycle_forbidden must still be None
        for rep in range(2):
            for tok in ostinato_12:
                st.step(tok)
            self.assertIsNone(st.cycle_forbidden(), f"Blocked too early after {rep + 1} repeats")
        # 3rd repeat completes -> 4th repeat's opening token ('eighth') must now be forbidden
        for tok in ostinato_12:
            st.step(tok)
        self.assertEqual(st.cycle_forbidden(), "eighth")
        allowed = ev._allowed_ids_for_state(st, uni_omr.VOCAB_SIZE)
        self.assertNotIn(data.TOKEN_TO_ID["eighth"], allowed)

        # 2. Tight note loop `eighth C 4` (period 9 = 3x3) blocked after 9 notes (3x period-9)
        st_note = grammar.PrefixState(strict=False)
        for tok in ("clef-G-2", "key+0", "time-4-4"):
            st_note.step(tok)
        for _ in range(9):
            for tok in ("eighth", "C", "4"):
                st_note.step(tok)
        self.assertEqual(st_note.cycle_forbidden(), "eighth")

        # 3. Verbatim multi-bar loop (21 tokens containing 'bar') blocked after 3 bars
        bar_21 = [
            "voice-1", "quarter", "C", "4", "quarter", "D", "4",
            "quarter", "E", "4", "quarter", "F", "4",
            "voice-2", "half", "G", "3", "half", "A", "3", "bar",
        ]
        self.assertEqual(len(bar_21), 21)
        st_bar = grammar.PrefixState(strict=False)
        for tok in ("clef-G-2", "key+0", "time-4-4"):
            st_bar.step(tok)
        for _ in range(3):
            for tok in bar_21:
                st_bar.step(tok)
        self.assertEqual(st_bar.cycle_forbidden(), "voice-1")

        # 4. Mid-bar time-* / key* forbidden in incomplete bar; allowed in header, after bar, and at full-bar courtesy
        t68_id = data.TOKEN_TO_ID["time-6-8"]
        t58_id = data.TOKEN_TO_ID["time-5-8"]
        k_m2_id = data.TOKEN_TO_ID["key-2"]
        for strict in (False, True):
            st_h = grammar.PrefixState(strict=strict)
            # System header: allowed
            self.assertIn(t68_id, st_h.allowed()[0])
            self.assertIn(k_m2_id, st_h.allowed()[0])
            for tok in ("clef-G-2", "key+0", "time-4-4", "quarter", "C", "4"):
                st_h.step(tok)
            # Incomplete bar (1.0 / 4.0 beats): time-* and key* forbidden
            allow_mid = st_h.allowed()[0]
            self.assertNotIn(t68_id, allow_mid)
            self.assertNotIn(t58_id, allow_mid)
            self.assertNotIn(k_m2_id, allow_mid)
            with self.assertRaises(ValueError):
                st_h.step("time-6-8")
            with self.assertRaises(ValueError):
                st_h.step("key-2")
            # Complete the bar (4.0 / 4.0 beats): courtesy key/time allowed
            for tok in ("half", "D", "4", "quarter", "E", "4"):
                st_h.step(tok)
            allow_full = st_h.allowed()[0]
            self.assertIn(t68_id, allow_full)
            self.assertIn(k_m2_id, allow_full)
            # Immediately after barline: allowed
            st_h.step("bar")
            allow_bar = st_h.allowed()[0]
            self.assertIn(t68_id, allow_bar)
            self.assertIn(k_m2_id, allow_bar)
            st_h.step("key-2")
            st_h.step("time-6-8")

    def test_iter9_measure_beam_and_tok_lcs_metrics(self) -> None:
        """Violation-triggered measure-scoped beam search, TokLCS, and CLI flag."""
        # 1. _measure_violation_errors checks
        st = grammar.PrefixState(strict=False)
        for tok in ("clef-G-2", "key+0", "time-4-4", "half", "C", "4", "half", "D", "4"):
            st.step(tok)
        self.assertEqual(ev._measure_violation_errors(st), (0.0, 0.0))

        st_bad = grammar.PrefixState(strict=False)
        for tok in ("clef-G-2", "key+0", "time-4-4", "voice-1", "whole", "C", "4", "voice-2", "half", "G", "3"):
            st_bad.step(tok)
        m_err, v_err = ev._measure_violation_errors(st_bad)
        self.assertAlmostEqual(m_err, 2.0, places=5)
        self.assertAlmostEqual(v_err, 2.0, places=5)

        # 2. TokLCS penalizes loops while scoring exact match = 1.0
        ref = ["clef-G-2", "time-4-4", "quarter", "C", "4", "quarter", "D", "4", "bar"]
        self.assertAlmostEqual(ev.compute_token_lcs_f1(ref, ref), 1.0, places=5)
        looped = ref + ["quarter", "C", "4"] * 20
        self.assertLess(ev.compute_token_lcs_f1(ref, looped), 0.30)

        # 3. End-to-end generate_mtp_spec with measure_beam_size=4 (B=1 and B=2)
        model = _make_tiny_model(seed=61, mtp_k=4)
        g = torch.Generator(device="cpu").manual_seed(610)
        for batch_size in (1, 2):
            imgs = torch.rand((batch_size, 3, 64, 64), generator=g)
            seq, stats = ev.generate_mtp_spec(
                model,
                imgs,
                max_len=24,
                use_grammar=True,
                strict_grammar=False,
                return_stats=True,
                measure_beam_size=4,
            )
            self.assertEqual(seq.shape[0], batch_size)
            self.assertIn("measure_beam_triggers", stats)
            self.assertIn("measure_beam_repaired", stats)
            for row in seq.tolist():
                toks = ev.decode_uni_compat(row, vocab_size=model.cfg.vocab_size, skip_special=True)
                st_chk = grammar.PrefixState(strict=False)
                for tok in toks:
                    if tok == uni_omr.SYS_TOKEN:
                        st_chk = grammar.PrefixState(strict=False)
                    else:
                        st_chk.step(tok)

        # 4. Direct _beam_repair_measure verification (KV-cache rewind + scoring)
        model_r = _make_tiny_model(seed=73, mtp_k=4)
        img_r = torch.rand((1, 3, 64, 64))
        with torch.no_grad():
            mem_r, _, _ = model_r.encode_image(img_r, return_aux=False)
            prefix_toks = ["clef-G-2", "key+0", "time-4-4", "whole", "C", "4"]
            prefix_ids = [uni_omr.BOS_ID] + [data.TOKEN_TO_ID[t] for t in prefix_toks]
            _, _, kv_pre = model_r.decode_tokens(
                torch.tensor([prefix_ids], dtype=torch.long),
                memory=mem_r,
                return_mtp=False,
                past_kvs=None,
                use_cache=True,
            )
            base_cross = [
                (ck.expand(4, -1, -1, -1).contiguous(), cv.expand(4, -1, -1, -1).contiguous())
                for _, (ck, cv) in kv_pre
            ]
            st_pre = grammar.PrefixState(strict=False)
            for t in prefix_toks + ["bar"]:
                st_pre.step(t)
            best_toks, best_st, kv0, _ = ev._beam_repair_measure(
                model=model_r,
                kv_prefix=kv_pre,
                base_cross_kvs=base_cross,
                trigger_id=data.TOKEN_TO_ID["bar"],
                bar_start_state=st_pre,
                beam_size=4,
                max_bar_toks=24,
                vocab_size=model_r.cfg.vocab_size,
                use_grammar=True,
                strict_grammar=False,
                device=torch.device("cpu"),
            )
            self.assertEqual(int(kv0[0][0][0].shape[2]), len(prefix_ids) + 1)
            if best_toks is not None:
                self.assertIsNotNone(best_st)
                self.assertGreater(len(best_toks), 0)

        # 5. CLI default and custom --measure-beam-size
        ap = ev.build_arg_parser()
        ns_def = ap.parse_args(["--ckpt", "dummy.pt"])
        self.assertEqual(ns_def.measure_beam_size, 0)
        ns_mb = ap.parse_args(["--ckpt", "dummy.pt", "--measure-beam-size", "4"])
        self.assertEqual(ns_mb.measure_beam_size, 4)


if __name__ == "__main__":
    unittest.main()
