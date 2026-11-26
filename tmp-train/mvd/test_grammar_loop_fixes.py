"""CPU-only unit tests for the 4 FSA grammar loop loophole fixes in grammar.py.

Covers:
1. Loophole A: Non-bar period-120 chord-chain loop (`MAX_TOKENS_PER_BAR = 288`).
2. Loophole B: `<sys>` reset preserves >=512 recent window + canonical system hash repeat blocker.
3. Loophole C: Extended `bar_ngram_max=160` catches 74-tok and 148-tok alternating A-B bar loops.
4. Loophole D: Canonical bar hash guard catches single-token-perturbed 41-tok bar loops (>=5 consec or >=8/12) without touching pure rest bars.
5. Empty trailing system guard: `can_emit_eos()` blocks `<eos>` after `<sys>` + header with 0 notes/rests (unless `n_systems >= MAX_SYSTEMS` or `cur_len >= max_len`).
6. Full 45/45 `suite31_hw` GT safety verification (`broken=False`, every GT token allowed, `can_emit_eos() == True` at `<eos>`, 0 false positives).
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

import data  # noqa: E402
import eval_sys_gate as esg  # noqa: E402
import eval_uni_suite31 as eval_s31  # noqa: E402
import grammar  # noqa: E402

_T = data.TOKEN_TO_ID


def _make_pitched_bar(pitches, voice="voice-1", dur="16th", extra_mods=None):
    """Build a valid pitched bar ending with `bar`."""
    toks = [voice]
    for idx, (step, octv) in enumerate(pitches):
        toks.extend([dur, step, str(octv)])
        if extra_mods and idx in extra_mods:
            toks.append(extra_mods[idx])
    toks.append("bar")
    return toks


class GrammarLoopFixesTest(unittest.TestCase):
    def test_loophole_a_non_bar_period120_chord_chain_capped(self):
        """Loophole A: non-bar period-120 chord-chain loop restricted to bar-closing/<sys>/<eos> at cap."""
        st = grammar.PrefixState(strict=False)
        for t in ["clef-G-2", "key+0", "time-4-4"]:
            st.step(t)

        # Build a 120-token non-bar chord-chain motif (10 onsets x 3 chords = 120 tokens, no `bar`).
        motif = []
        steps = ["C", "D", "E", "F", "G", "A", "B", "C", "D", "E"]
        for i, s in enumerate(steps):
            octv = "4" if i < 7 else "5"
            motif.extend(["16th", s, octv])
            motif.extend(["chord", "16th", "E", "4"])
            motif.extend(["chord", "16th", "G", "4"])
            motif.extend(["16th", "A", "4"])
        # 10 * (3 + 4 + 4 + 3) = 140; slice to 120 tokens cleanly at an event boundary (30 events -> 120)
        motif120 = []
        for i in range(10):
            s = steps[i % len(steps)]
            motif120.extend(["16th", s, "4", "chord", "16th", "E", "4", "chord", "16th", "G", "4", "stacc"])
        self.assertEqual(len(motif120), 120)

        bar_like_ids = {_T[t] for t in grammar.BAR_LIKES if t in _T}

        # Step until just before MAX_TOKENS_PER_BAR: notes/chords must still be allowed.
        for t in motif120 * 2:
            st.step(t)
        self.assertLess(st.tokens_since_bar, grammar.MAX_TOKENS_PER_BAR)
        allow_before, eos_before = st.allowed()
        self.assertIn(_T["16th"], allow_before)
        self.assertIn(_T["chord"], allow_before)
        self.assertTrue(eos_before)

        # Continue into the 3rd period until we hit MAX_TOKENS_PER_BAR at a note boundary.
        for t in motif120:
            st.step(t)
            if st.tokens_since_bar >= grammar.MAX_TOKENS_PER_BAR and st.mode in ("bound", "pitch-tail"):
                break
        self.assertGreaterEqual(st.tokens_since_bar, grammar.MAX_TOKENS_PER_BAR)

        allow_capped, eos_capped = st.allowed()
        self.assertEqual(allow_capped, bar_like_ids)
        self.assertNotIn(_T["chord"], allow_capped)
        self.assertNotIn(_T["16th"], allow_capped)
        self.assertNotIn(_T["rest"], allow_capped)
        self.assertTrue(eos_capped)
        self.assertTrue(st.can_emit_sys())

        # Emitting `bar` resets `tokens_since_bar` and restores normal decoding.
        st.step("bar")
        self.assertEqual(st.tokens_since_bar, 0)
        allow_after, _ = st.allowed()
        self.assertIn(_T["quarter"], allow_after)
        self.assertIn(_T["rest"], allow_after)

    def test_loophole_b_sys_reset_window_and_duplicate_system_blocked(self):
        """Loophole B: recent keeps >=512 across reset_system(), identical canonical system blocks <sys>."""
        st = grammar.PrefixState(strict=False)
        sys_body = ["clef-G-2", "key+0", "time-4-4"]
        # Make a ~160-token system
        for i in range(4):
            sys_body.extend(_make_pitched_bar([("C", 4), ("D", 4), ("E", 4), ("F", 4)] * 3, dur="16th"))
        self.assertGreater(len(sys_body), 150)

        # System 1
        for t in sys_body:
            st.step(t)
        self.assertTrue(st.can_emit_sys())
        st.reset_system()
        self.assertEqual(len(st.completed_system_hashes), 1)
        self.assertGreater(len(st.recent), 150)

        # Distinct System 2 (different pitch) -> still allowed to emit <sys>
        sys_body_2 = list(sys_body)
        sys_body_2[5] = "G"  # change one pitch step ('C' -> 'G')
        for t in sys_body_2:
            st.step(t)
        self.assertTrue(st.can_emit_sys())
        st.reset_system()
        self.assertEqual(len(st.completed_system_hashes), 2)

        # Distinct System 3 & System 4 -> push total tokens past 600 (> RECENT_WINDOW_KEEP=576)
        for step_ch in ("A", "B"):
            sys_body_n = list(sys_body)
            sys_body_n[5] = step_ch
            for t in sys_body_n:
                st.step(t)
            self.assertTrue(st.can_emit_sys())
            st.reset_system()
        # Verify recent window retains >= 512 tokens across reset_system()!
        self.assertGreaterEqual(len(st.recent), 512)
        self.assertEqual(len(st.recent), grammar.RECENT_WINDOW_KEEP)

        # Duplicate of System 1 with cosmetic dyn-f + slur-start/slur-stop perturbation
        sys_dup = (
            ["clef-G-2", "key+0", "time-4-4", "dyn-f"]
            + sys_body[3:7]
            + ["slur-start"]
            + sys_body[7:10]
            + ["slur-stop"]
            + sys_body[10:]
        )
        for t in sys_dup:
            st.step(t)
        # Matches earlier System 1 -> count would be >= 2 -> <sys> MUST be forbidden!
        self.assertFalse(st.can_emit_sys())
        self.assertTrue(st.can_emit_eos())

    def test_loophole_c_long_74_and_148_tok_alternating_ab_bars(self):
        """Loophole C: bar_ngram_max=160 catches 74-tok and 148-tok alternating A-B bar loops."""
        # Case 1: 37-tok bar A + 37-tok bar B = 74-tok alternating period
        bar_a37 = _make_pitched_bar(
            [("C", 4), ("D", 4), ("E", 4), ("F", 4), ("G", 4), ("A", 4),
             ("B", 4), ("C", 5), ("D", 5), ("E", 5), ("F", 5), ("G", 5)][:12],
            dur="16th",
        )
        # len = 1 (voice-1) + 12*3 (36) + 1 (bar) = 38; trim 1 note + add dot -> 37
        bar_a37 = ["voice-1"] + [x for pair in [("C", 4), ("D", 4), ("E", 4), ("F", 4), ("G", 4), ("A", 4), ("B", 4), ("C", 5), ("D", 5), ("E", 5), ("F", 5)] for x in ("16th", pair[0], str(pair[1]))] + ["stacc", "stacc", "bar"]
        bar_b37 = ["voice-1"] + [x for pair in [("G", 4), ("F", 4), ("E", 4), ("D", 4), ("C", 4), ("B", 3), ("A", 3), ("G", 3), ("F", 3), ("E", 3), ("D", 3)] for x in ("16th", pair[0], str(pair[1]))] + ["stacc", "stacc", "bar"]
        self.assertEqual(len(bar_a37), 37)
        self.assertEqual(len(bar_b37), 37)
        ab74 = bar_a37 + bar_b37
        self.assertEqual(len(ab74), 74)

        st = grammar.PrefixState(strict=False)
        for t in ["clef-G-2", "key+0", "time-4-4"]:
            st.step(t)
        for rep in range(2):
            for t in ab74:
                st.step(t)
            self.assertIsNone(st.cycle_forbidden(), f"Blocked too early at rep {rep + 1}")
        for t in ab74:
            st.step(t)
        self.assertEqual(st.cycle_forbidden(), "voice-1")

        # Case 2: 74-tok bar A + 74-tok bar B = 148-tok alternating period (<= 160, 3*148=444 <= 576)
        bar_a74 = ["voice-1"] + [x for pair in [("C", 4), ("D", 4), ("E", 4), ("F", 4), ("G", 4), ("A", 4)] * 4 for x in ("32nd", pair[0], str(pair[1]))] + ["bar"]
        bar_b74 = ["voice-1"] + [x for pair in [("G", 4), ("F", 4), ("E", 4), ("D", 4), ("C", 4), ("B", 3)] * 4 for x in ("32nd", pair[0], str(pair[1]))] + ["bar"]
        self.assertEqual(len(bar_a74), 74)
        self.assertEqual(len(bar_b74), 74)
        ab148 = bar_a74 + bar_b74
        self.assertEqual(len(ab148), 148)

        st2 = grammar.PrefixState(strict=False)
        for t in ["clef-G-2", "key+0", "time-4-4"]:
            st2.step(t)
        for rep in range(2):
            for t in ab148:
                st2.step(t)
            self.assertIsNone(st2.cycle_forbidden())
        for t in ab148:
            st2.step(t)
        self.assertEqual(st2.cycle_forbidden(), "voice-1")

    def test_loophole_d_single_token_perturbed_41_tok_bars(self):
        """Loophole D: canonical bar hash catches >=5 consec or >=8/12 perturbed 41-tok bars; ignores pure rest bars."""
        base_pitches = [("C", 4), ("E", 4), ("G", 4), ("C", 5), ("B", 4), ("G", 4),
                        ("E", 4), ("C", 4), ("D", 4), ("F", 4), ("A", 4), ("D", 5), ("B", 4)]
        # 1 (voice-1) + 13*3 (39) + 1 (bar) = 41 tokens
        bar_plain = _make_pitched_bar(base_pitches, dur="16th")
        self.assertEqual(len(bar_plain), 41)

        # Perturb each repeat with different cosmetic tokens (slur-start/slur-stop/dyn-*/clef-*)
        # so verbatim n-gram cycle_forbidden() CANNOT match them!
        variants = [
            ["dyn-p"] + bar_plain,
            ["clef-G-2"] + _make_pitched_bar(base_pitches, dur="16th", extra_mods={0: "slur-start", 3: "slur-stop"}),
            ["dyn-mf"] + bar_plain,
            _make_pitched_bar(base_pitches, dur="16th", extra_mods={1: "slur-start", 5: "slur-stop"}),
            ["dyn-f", "clef-G-2"] + bar_plain,
        ]
        st = grammar.PrefixState(strict=False)
        for t in ["clef-G-2", "key+0", "time-4-4"]:
            st.step(t)
        for idx in range(4):
            for t in variants[idx]:
                st.step(t)
            self.assertIsNone(st.cycle_forbidden(), f"Should not fire at repeat {idx + 1}")
        # 5th consecutive canonical repeat triggers!
        for t in variants[4]:
            st.step(t)
        self.assertEqual(st.cycle_forbidden(), "voice-1")
        # Even if decoder tries a cosmetic dyn-p first, opening body token stays forbidden!
        st.step("dyn-p")
        self.assertEqual(st.cycle_forbidden(), "voice-1")

        # Window rule: >=8 in last 12 bars of the same system (interleaved so never >=5 consec)
        bar_other = _make_pitched_bar([("G", 3), ("B", 3), ("D", 4), ("G", 4)], dur="quarter")
        st_win = grammar.PrefixState(strict=False)
        for t in ["clef-G-2", "key+0", "time-4-4"]:
            st_win.step(t)
        # Pattern: A A B A A B A A B A A (8 copies of A in 11 bars, max consec = 2)
        seq_bars = [
            variants[0], variants[1], bar_other,
            variants[2], variants[3], bar_other,
            variants[4], variants[0], bar_other,
            variants[1], variants[2],
        ]
        for idx, b in enumerate(seq_bars[:-1]):
            for t in b:
                st_win.step(t)
            self.assertIsNone(st_win.bar_repeat_forbidden_tok, f"Fired early at bar {idx + 1}")
        for t in seq_bars[-1]:
            st_win.step(t)
        self.assertEqual(st_win.bar_repeat_forbidden_tok, "voice-1")
        self.assertEqual(st_win.cycle_forbidden(), "voice-1")

        # Pure rest bars (`bar-rest` and `rest whole`) repeated 10x must NEVER trigger bar hash guard
        st_rest = grammar.PrefixState(strict=False)
        for t in ["clef-G-2", "key+0", "time-4-4"]:
            st_rest.step(t)
        for _ in range(10):
            for t in ["bar-rest", "bar"]:
                st_rest.step(t)
        self.assertIsNone(st_rest.bar_repeat_forbidden_tok)
        for _ in range(10):
            for t in ["rest", "whole", "bar"]:
                st_rest.step(t)
        self.assertIsNone(st_rest.bar_repeat_forbidden_tok)

    def test_empty_trailing_system_eos_guard(self):
        """Empty trailing system guard: blocks <eos> after <sys> + header with 0 notes/rests."""
        st = grammar.PrefixState(strict=False)
        # Header-only system 0 (foster-leadsheet 10-tok collapse)
        for t in ["clef-G-2", "key+0", "time-4-4"]:
            st.step(t)
        self.assertFalse(st.can_emit_eos())
        _, eos_ok = st.allowed()
        self.assertFalse(eos_ok)

        # Emit a real note in system 0 -> EOS and <sys> now legal
        for t in ["quarter", "C", "4", "bar"]:
            st.step(t)
        self.assertTrue(st.can_emit_eos())
        self.assertTrue(st.can_emit_sys())
        _, eos_ok = st.allowed()
        self.assertTrue(eos_ok)

        # Emit <sys> + header (0 notes/rests in system 1) -> EOS MUST be blocked!
        st.reset_system()
        self.assertFalse(st.can_emit_eos())
        _, eos_ok = st.allowed()
        self.assertFalse(eos_ok)
        for t in ["clef-G-2", "key+0", "time-4-4"]:
            st.step(t)
        self.assertFalse(st.can_emit_eos())
        _, eos_ok = st.allowed()
        self.assertFalse(eos_ok)

        # Escape hatches: max_len reached OR MAX_SYSTEMS reached -> EOS allowed
        self.assertTrue(st.can_emit_eos(cur_len=3072, max_len=3072))
        st_max = st.copy()
        st_max.sys_count = grammar.MAX_SYSTEMS - 1
        self.assertTrue(st_max.can_emit_eos())

        # Once a rest or note is emitted in system 1 -> EOS legal again
        for t in ["rest", "whole", "bar-light-heavy"]:
            st.step(t)
        self.assertTrue(st.can_emit_eos())
        _, eos_ok = st.allowed()
        self.assertTrue(eos_ok)

    def test_loophole_5_single_system_runaway_capped(self):
        """Loophole #5: MAX_BARS_PER_SYSTEM & MAX_TOKENS_PER_SYSTEM force <sys>/<eos> on single-system runaways."""
        vocab_size = len(data.VOCAB) + 4
        _, _, eos_id, sys_id, _, _ = eval_s31._vocab_spec(vocab_size)
        steps = ["C", "D", "E", "F", "G", "A", "B"]

        # Part 1: Hit MAX_BARS_PER_SYSTEM with distinct non-repeating bars (bypassing bar-hash & n-gram guards)
        st = grammar.PrefixState(strict=False)
        for t in ["clef-G-2", "key+0", "time-4-4"]:
            st.step(t)
        for b_idx in range(grammar.MAX_BARS_PER_SYSTEM - 1):
            pitches = [
                (steps[(b_idx + k) % 7], 4 + ((b_idx // 7 + k) % 2))
                for k in range(4)
            ]
            for t in _make_pitched_bar(pitches, dur="quarter"):
                st.step(t)
        self.assertEqual(st.bars_in_system, grammar.MAX_BARS_PER_SYSTEM - 1)
        allow_before, eos_before = st.allowed()
        self.assertIn(_T["quarter"], allow_before)
        self.assertTrue(eos_before)

        # Emit the bar that hits MAX_BARS_PER_SYSTEM
        last_bar = _make_pitched_bar([("G", 5), ("F", 5), ("E", 5), ("D", 5)], dur="quarter")
        for t in last_bar:
            st.step(t)
        self.assertEqual(st.bars_in_system, grammar.MAX_BARS_PER_SYSTEM)
        allow_capped, eos_capped = st.allowed()
        self.assertEqual(allow_capped, set())
        self.assertTrue(eos_capped)
        self.assertTrue(st.can_emit_sys())
        allowed_ids = eval_s31._allowed_ids_for_state(st, vocab_size)
        self.assertEqual(set(allowed_ids), {eos_id, sys_id})

        # copy() preserves counters; reset_system() zeroes both and restores normal decoding
        st_copy = st.copy()
        self.assertEqual(st_copy.bars_in_system, grammar.MAX_BARS_PER_SYSTEM)
        self.assertEqual(st_copy.tokens_in_system, st.tokens_in_system)
        st.reset_system()
        self.assertEqual(st.bars_in_system, 0)
        self.assertEqual(st.tokens_in_system, 0)
        allow_after, _ = st.allowed()
        self.assertIn(_T["clef-G-2"], allow_after)
        self.assertIn(_T["quarter"], allow_after)

        # Part 2: Hit MAX_TOKENS_PER_SYSTEM with fewer than MAX_BARS_PER_SYSTEM bars (dense varied bars)
        st2 = grammar.PrefixState(strict=False)
        for t in ["clef-G-2", "key+0", "time-4-4"]:
            st2.step(t)
        b_idx = 0
        while st2.tokens_in_system < grammar.MAX_TOKENS_PER_SYSTEM:
            # 25 varied 32nd notes per bar (~77 tokens/bar -> hits 1440 in ~19 bars < 28)
            pitches = [
                (steps[(b_idx * 3 + k) % 7], 3 + ((b_idx + k // 7) % 3))
                for k in range(25)
            ]
            bar_toks = _make_pitched_bar(pitches, dur="32nd")
            for t in bar_toks:
                st2.step(t)
                if st2.tokens_in_system >= grammar.MAX_TOKENS_PER_SYSTEM:
                    break
            b_idx += 1
        self.assertLess(st2.bars_in_system, grammar.MAX_BARS_PER_SYSTEM)
        self.assertGreaterEqual(st2.tokens_in_system, grammar.MAX_TOKENS_PER_SYSTEM)
        # If stopped mid-note (dur/step), current note is still allowed to complete cleanly
        if st2.mode == "dur":
            allow_mid, eos_mid = st2.allowed()
            self.assertIn(_T["C"], allow_mid)
            self.assertFalse(eos_mid)
            st2.step("C")
        if st2.mode == "step":
            allow_mid, eos_mid = st2.allowed()
            self.assertIn(_T["4"], allow_mid)
            self.assertFalse(eos_mid)
            st2.step("4")
        # At boundary/pitch-tail, ONLY <sys> / <eos> remain legal!
        allow_cap2, eos_cap2 = st2.allowed()
        self.assertEqual(allow_cap2, set())
        self.assertTrue(eos_cap2)
        self.assertEqual(set(eval_s31._allowed_ids_for_state(st2, vocab_size)), {eos_id, sys_id})

    def test_suite31_hw_45_gt_sequences_100pct_safe(self):
        """All 45 suite31_hw GT sequences pass PrefixState with broken=False, 0 false blocks, can_emit_eos()==True."""
        import xml.etree.ElementTree as ET

        mpath = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"
        self.assertTrue(mpath.exists(), f"Missing manifest: {mpath}")
        _, pieces = eval_s31.resolve_manifest_pieces(mpath)
        self.assertEqual(len(pieces), 45)

        max_bar_toks = 0
        max_xml_bars_per_sys = 0
        with tempfile.TemporaryDirectory(prefix="test_grammar_s31_") as tmpdir:
            workdir = Path(tmpdir)
            for p in pieces:
                slug = p["slug"]
                toks, _ = esg.suite31_page_refs(p["gt"], "test-" + slug, workdir)
                self.assertGreater(len(toks), 0, f"Empty GT for {slug}")

                # Check per-system bar counts on MusicXMLs that encode <print new-system="yes"/>
                root = ET.parse(p["gt"]).getroot()
                parts = root.findall("part")
                if parts:
                    is_p1 = Path(p["gt"]).name.startswith("gt-p1-")
                    sys_bars = [0]
                    for m_idx, m in enumerate(parts[0].findall("measure")):
                        pr = m.find("print")
                        if m_idx > 0 and pr is not None:
                            if not is_p1 and pr.attrib.get("new-page") == "yes":
                                break
                            if pr.attrib.get("new-system") == "yes":
                                sys_bars.append(0)
                        sys_bars[-1] += 1
                    if len(sys_bars) > 1:
                        max_xml_bars_per_sys = max(max_xml_bars_per_sys, max(sys_bars))

                # suite31_page_refs emits flat page-level bekern without <sys> separators;
                # enforce per-system caps when <sys> is present, and disable only for flat streams.
                has_sys = "<sys>" in toks
                st = grammar.PrefixState(
                    strict=False,
                    max_bars_per_system=grammar.MAX_BARS_PER_SYSTEM if has_sys else None,
                    max_tokens_per_system=grammar.MAX_TOKENS_PER_SYSTEM if has_sys else None,
                )
                for idx, tok in enumerate(toks):
                    if tok == "<sys>":
                        self.assertTrue(st.can_emit_sys(), f"{slug}@{idx}: can_emit_sys() False on GT <sys>")
                        st.reset_system()
                        continue
                    allow, _ = st.allowed()
                    tok_id = _T[tok]
                    self.assertIn(
                        tok_id,
                        allow,
                        f"{slug}@{idx}: GT token {tok!r} blocked (tokens_since_bar={st.tokens_since_bar})",
                    )
                    self.assertIsNone(
                        st.bar_repeat_forbidden_tok,
                        f"{slug}@{idx}: false-positive bar_repeat_forbidden_tok={st.bar_repeat_forbidden_tok!r}",
                    )
                    st.step(tok)
                    if st.tokens_since_bar > max_bar_toks:
                        max_bar_toks = st.tokens_since_bar

                self.assertFalse(st.broken, f"{slug}: PrefixState broken=True")
                self.assertTrue(st.can_emit_eos(), f"{slug}: can_emit_eos() False at <eos>")
                _, eos_ok = st.allowed()
                self.assertTrue(eos_ok, f"{slug}: allowed() eos_ok=False at <eos>")

        self.assertLessEqual(max_bar_toks + 64, grammar.MAX_TOKENS_PER_BAR)
        self.assertLessEqual(max_xml_bars_per_sys + 8, grammar.MAX_BARS_PER_SYSTEM)


if __name__ == "__main__":
    unittest.main(verbosity=2)
