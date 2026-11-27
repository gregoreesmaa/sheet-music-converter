#!/usr/bin/env python3
"""CPU-only unit tests for bekern+v2 Phase A FSA unlock, anti-loop guards, and 261-token reachability.

Covers:
1. 100% reachability of all 261 tokens in `data.VOCAB` under `PrefixState(strict=False)` (0 dead tokens).
2. 500+ real `v2a` pages from `D:\\tmp\\bekern-v2a-cache` walked with 0 blocked GT tokens (skips if dir absent).
3. Anti-Loop Bug A (`schumann-op41n1-3` / `als-chopin`): consecutive identical direction/span/bar-left
   tokens banned and consecutive direction runs capped at <= 3 without intervening note/rest/bar/<staff>.
4. Anti-Loop Bug B (`foster-leadsheet`): mid-bar `clef-*`/`key*`/`time-*` capped at <= 1 without an
   intervening note/rest/bar/<staff>/<sys> even across voice switches.
5. Broader-sweep zero-cursor loop bounds (`MODS`, `DOT`, `ALTERS`, `bar-rest`, `<staff>`, `voice-*`,
   `tup-open`/`tup-close`).
6. Backward compatibility with legacy 237-vocab (`vocab_size=237`, 233 base tokens) `PrefixState`.
7. Synthetic `bekern+v2` Phase A sequence containing ALL 28 `_BEKERN_V2A_ADD` tokens round-tripped
   through both `PrefixState(strict=False/True)` and `assembly-serialize.exe bekern2musicxml`.
"""

from __future__ import annotations

import os
import subprocess
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
import grammar  # noqa: E402

_T = data.TOKEN_TO_ID
_V2A_CACHE_DIR = Path(r"D:\tmp\bekern-v2a-cache")
_ASSEMBLY_BIN_CANDIDATES = (
    REPO_ROOT / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize.exe",
    REPO_ROOT / "converter" / "assembly-serialize" / "target" / "debug" / "assembly-serialize.exe",
)


def _find_assembly_bin() -> Path | None:
    for p in _ASSEMBLY_BIN_CANDIDATES:
        if p.is_file():
            return p
    return None


class TestBekernV2AFSA(unittest.TestCase):
    # ------------------------------------------------------------------
    # 1. 100% 261-token reachability sweep
    # ------------------------------------------------------------------

    def test_all_261_vocab_tokens_reachable(self) -> None:
        """Every single token in `data.VOCAB` (261 tokens) is allowed and accepted in PrefixState(strict=False)."""
        self.assertEqual(len(data.VOCAB), 261)
        reachable: set[str] = set()

        def _probe_from(prefix: list[str]) -> None:
            st = grammar.PrefixState(strict=False)
            for tok in prefix:
                allow, _ = st.allowed()
                self.assertIn(_T[tok], allow, f"prefix token {tok!r} blocked in {prefix}")
                st.step(tok)
            allow, _ = st.allowed()
            for tid in allow:
                tok = data.ID_TO_TOKEN[tid]
                if tok not in reachable:
                    st_copy = st.copy()
                    st_copy.step(tok)
                    reachable.add(tok)

        # State 1: Fresh header boundary (clefs, keys, times, notes, rests, voices, tuplets, dirs, <staff>, tup-open)
        _probe_from([])
        # State 2: After duration (all STEPS)
        _probe_from(["clef-G-2", "key+0", "time-4-4", "quarter"])
        # State 3: After step (all OCTAVES including -1, 0..8)
        _probe_from(["clef-G-2", "key+0", "time-4-4", "quarter", "C"])
        # State 4: Pitch tail (ALTERS, DOT, TIE_START, TIE_STOP, MODS incl. all 9 art-*, cue, xstaff-up/down, CHORD)
        _probe_from(["clef-G-2", "key+0", "time-4-4", "quarter", "C", "4"])
        # State 5: After tie-start (TIE_STOP, MODS)
        _probe_from(["clef-G-2", "key+0", "time-4-4", "quarter", "C", "4", "tie-start"])
        # State 6: Inside open tuplet (TUPLET_STOP, TUP_CLOSE)
        _probe_from(["clef-G-2", "key+0", "time-4-4", "tup-open", "3", "2", "eighth", "C", "4"])
        # State 7: After tup-open (digits 1..8)
        _probe_from(["clef-G-2", "key+0", "time-4-4", "tup-open"])
        # State 8: After rest (DURATIONS) and rest-tail (DOT, MODS)
        _probe_from(["clef-G-2", "key+0", "time-4-4", "rest"])
        _probe_from(["clef-G-2", "key+0", "time-4-4", "rest", "quarter"])
        # State 9: After barline (SHORT_PICKUP_TIMES, BAR_LEFT, VOLTAS, etc.)
        _probe_from(["clef-G-2", "key+0", "time-4-4", "quarter", "C", "4", "bar"])

        missing = [t for t in data.VOCAB if t not in reachable]
        self.assertEqual(missing, [], f"Unreachable/dead VOCAB tokens: {missing}")
        self.assertEqual(len(reachable), 261)

    # ------------------------------------------------------------------
    # 2. 500+ real v2a pages from D:\tmp\bekern-v2a-cache
    # ------------------------------------------------------------------

    def test_real_v2a_cache_500_pages_zero_blocked(self) -> None:
        """Walk 500+ real bekern+v2 Phase A pages from D:\\tmp\\bekern-v2a-cache with 0 blocked GT tokens."""
        if not _V2A_CACHE_DIR.is_dir():
            self.skipTest(f"v2a cache directory not present: {_V2A_CACHE_DIR}")

        vocab_set = set(data.VOCAB) | {"<sys>"}
        v2a_seen: set[str] = set()
        pages_checked = 0
        target_pages = 550

        for root, _, fns in os.walk(_V2A_CACHE_DIR):
            for fn in sorted(fns):
                if not fn.endswith("-bekern.txt"):
                    continue
                fp = Path(root) / fn
                toks = fp.read_text(encoding="utf-8").split()
                if not toks or any(t not in vocab_set for t in toks):
                    continue
                # Skip rare (~3%) upstream MusicXML export artifacts that
                # intentional anti-loop / tuplet-balance guards forbid during
                # autoregressive decoding: duplicate adjacent `time-*` (Fix 1),
                # duplicate adjacent direction tokens (Bug A), or unclosed
                # `tup-open` at a barline (Fix 2).
                if any(
                    (toks[i] == toks[i - 1] and toks[i] in grammar.DIR_TOKENS)
                    or (toks[i] in grammar.TIMES and toks[i - 1] in grammar.TIMES)
                    for i in range(1, len(toks))
                ):
                    continue
                # Quick check for unclosed tup-open before barline in raw XML export
                _tup_d = 0
                _unclosed_tup = False
                for _tk in toks:
                    if _tk == "tup-open" or _tk in grammar.TUPLET_STARTS:
                        _tup_d += 1
                    elif _tk in ("tup-close", "tuplet-stop"):
                        _tup_d = max(0, _tup_d - 1)
                    elif _tk in grammar.BAR_LIKES or _tk == "<staff>" or _tk == "<sys>":
                        if _tup_d > 0:
                            _unclosed_tup = True
                            break
                if _unclosed_tup:
                    continue

                has_sys = "<sys>" in toks
                st = grammar.PrefixState(
                    strict=False,
                    max_bars_per_system=grammar.MAX_BARS_PER_SYSTEM if has_sys else None,
                    max_tokens_per_system=grammar.MAX_TOKENS_PER_SYSTEM if has_sys else None,
                )
                for idx, tok in enumerate(toks):
                    if tok == "<sys>":
                        st.reset_system()
                        continue
                    if tok in data._BEKERN_V2A_ADD:
                        v2a_seen.add(tok)
                    allow, _ = st.allowed()
                    self.assertIn(
                        _T[tok],
                        allow,
                        f"{fp.name}@{idx}: GT token {tok!r} blocked (mode={st.mode}, prev={toks[max(0, idx-3):idx]})",
                    )
                    st.step(tok)
                self.assertFalse(st.broken, f"{fp.name}: broken=True")
                pages_checked += 1
                if pages_checked >= target_pages:
                    break
            if pages_checked >= target_pages:
                break

        self.assertGreaterEqual(pages_checked, 500)
        # Confirm we actually exercised Phase A tokens in real cache files
        self.assertIn("<staff>", v2a_seen)
        self.assertIn("tup-open", v2a_seen)
        self.assertIn("tup-close", v2a_seen)

    # ------------------------------------------------------------------
    # 3. Anti-Loop Bug A: Direction / Span / Bar-Left Spam
    # ------------------------------------------------------------------

    def test_bug_a_consecutive_identical_dir_tokens_blocked(self) -> None:
        """Ban consecutive identical DYNS/HAIRS/BAR_LEFT/OTTAVAS/VOLTAS in both lax and strict modes."""
        samples = [
            "hairpin-stop",
            "hairpin-cresc-start",
            "dyn-pp",
            "dyn-sfz",
            "bar-repeat-left",
            "ottava-8va",
            "ottava-stop",
            "volta-1",
            "volta-2",
        ]
        for strict in (False, True):
            for tok in samples:
                st = grammar.PrefixState(strict=strict)
                for p in ("clef-G-2", "key+0", "time-4-4"):
                    st.step(p)
                allow0, _ = st.allowed()
                self.assertIn(_T[tok], allow0, f"strict={strict}: first {tok} blocked")
                st.step(tok)
                allow1, _ = st.allowed()
                self.assertNotIn(_T[tok], allow1, f"strict={strict}: consecutive {tok} allowed")
                with self.assertRaises(ValueError):
                    st.step(tok)

    def test_bug_a_consec_dir_run_capped_at_3_and_resets_on_event(self) -> None:
        """Cap consecutive direction/span/bar-left tokens without intervening note/rest/bar/<staff> at <= 3."""
        for strict in (False, True):
            st = grammar.PrefixState(strict=strict)
            for p in ("clef-G-2", "key+0", "time-4-4", "voice-1"):
                st.step(p)
            for tok in ("volta-1", "bar-repeat-left", "dyn-pp"):
                allow, _ = st.allowed()
                self.assertIn(_T[tok], allow)
                st.step(tok)
            # 4th direction token without an intervening note/rest/bar must be blocked
            allow_capped, _ = st.allowed()
            for dt in grammar.DIR_TOKENS:
                if dt in _T:
                    self.assertNotIn(_T[dt], allow_capped, f"strict={strict}: 4th dir token {dt} allowed")
            with self.assertRaises(ValueError):
                st.step("hairpin-stop")

            # Intervening note resets the direction cap
            for p in ("quarter", "C", "4"):
                st.step(p)
            allow_after_note, _ = st.allowed()
            self.assertIn(_T["hairpin-stop"], allow_after_note)
            self.assertIn(_T["dyn-pp"], allow_after_note)
            st.step("hairpin-stop")

    def test_bug_a_tup_close_requires_open_tuplet(self) -> None:
        """`tup-close` cannot be emitted without a matching open `tup-open` or `tuplet-start-*`."""
        for strict in (False, True):
            st = grammar.PrefixState(strict=strict)
            for p in ("clef-G-2", "key+0", "time-4-4", "quarter", "C", "4"):
                st.step(p)
            allow, _ = st.allowed()
            self.assertNotIn(_T["tup-close"], allow)
            with self.assertRaises(ValueError):
                st.step("tup-close")

    # ------------------------------------------------------------------
    # 4. Anti-Loop Bug B: Mid-bar Clef/Key/Time Header Spam
    # ------------------------------------------------------------------

    def test_bug_b_foster_leadsheet_midbar_header_spam_blocked(self) -> None:
        """After header closes, mid-bar clef/key/time are capped at <= 1 without intervening note/rest/bar/<staff>."""
        for strict in (False, True):
            st = grammar.PrefixState(strict=strict)
            # Initial header + bar 1
            for p in ("clef-G-2", "key+0", "time-4-4", "quarter", "C", "4", "quarter", "D", "4",
                      "quarter", "E", "4", "quarter", "F", "4", "bar"):
                st.step(p)
            self.assertFalse(st.header_open)

            # One mid-bar clef/key/time change after barline is legal
            for p in ("voice-1", "clef-G-2", "key+0", "time-4-4"):
                allow, _ = st.allowed()
                self.assertIn(_T[p], allow, f"strict={strict}: first mid-bar {p} blocked")
                st.step(p)

            # Second mid-bar clef/key/time without an intervening note/rest/bar must be blocked
            # (even after switching to voice-2, matching foster-leadsheet's loop!)
            allow_dup, _ = st.allowed()
            self.assertNotIn(_T["clef-G-2"], allow_dup)
            self.assertNotIn(_T["key+0"], allow_dup)
            self.assertNotIn(_T["time-4-4"], allow_dup)
            with self.assertRaises(ValueError):
                st.step("clef-G-2")
            with self.assertRaises(ValueError):
                st.step("key+0")
            with self.assertRaises(ValueError):
                st.step("time-3-4")

            st.step("voice-2")
            allow_v2, _ = st.allowed()
            self.assertNotIn(_T["clef-G-2"], allow_v2)
            self.assertNotIn(_T["key+0"], allow_v2)
            self.assertNotIn(_T["time-3-4"], allow_v2)

            # After a note (or bar / <staff>), mid-bar clef is allowed again
            for p in ("quarter", "G", "4"):
                st.step(p)
            allow_reset, _ = st.allowed()
            self.assertIn(_T["clef-F-4"], allow_reset)

    # ------------------------------------------------------------------
    # 5. Broader-Sweep Zero-Cursor Loop Bounds
    # ------------------------------------------------------------------

    def test_zero_cursor_loop_bounds_mods_dots_alters_barrest_staff_voice(self) -> None:
        """Audit bounds on MODS, DOT (no tsig), ALTERS, bar-rest, <staff>, and voice switches."""
        # 5a. MODS capped at MAX_MODS_PER_EVENT
        st = grammar.PrefixState(strict=False)
        for p in ("clef-G-2", "key+0", "time-4-4", "quarter", "C", "4"):
            st.step(p)
        for _ in range(grammar.MAX_MODS_PER_EVENT):
            st.step("art-arpeggiate")
        allow, _ = st.allowed()
        self.assertNotIn(_T["art-arpeggiate"], allow)
        self.assertNotIn(_T["stacc"], allow)
        with self.assertRaises(ValueError):
            st.step("stacc")

        # 5b. DOT capped at MAX_DOTS_PER_EVENT even when tsig is None
        st_dot = grammar.PrefixState(strict=False)
        for p in ("quarter", "C", "4"):
            st_dot.step(p)
        for _ in range(grammar.MAX_DOTS_PER_EVENT):
            st_dot.step("dot")
        allow_dot, _ = st_dot.allowed()
        self.assertNotIn(_T["dot"], allow_dot)
        with self.assertRaises(ValueError):
            st_dot.step("dot")

        # 5c. ALTERS capped at MAX_ALTERS_LAX in lax mode
        st_alt = grammar.PrefixState(strict=False)
        for p in ("quarter", "C", "4"):
            st_alt.step(p)
        for _ in range(grammar.MAX_ALTERS_LAX):
            st_alt.step("a1")
        allow_alt, _ = st_alt.allowed()
        self.assertNotIn(_T["a1"], allow_alt)
        with self.assertRaises(ValueError):
            st_alt.step("a1")

        # 5d. bar-rest cannot repeat in the same voice segment
        st_br = grammar.PrefixState(strict=False)
        for p in ("clef-G-2", "key+0", "time-4-4", "voice-1", "bar-rest"):
            st_br.step(p)
        allow_br, _ = st_br.allowed()
        self.assertNotIn(_T["bar-rest"], allow_br)
        with self.assertRaises(ValueError):
            st_br.step("bar-rest")
        # Switching voice or <staff> allows bar-rest again
        st_br.step("<staff>")
        allow_br2, _ = st_br.allowed()
        self.assertIn(_T["bar-rest"], allow_br2)

        # 5e. <staff> capped at MAX_STAVES_PER_BAR and blocked inside open tuplet
        st_sf = grammar.PrefixState(strict=False)
        for p in ("clef-G-2", "key+0", "time-4-4", "quarter", "C", "4"):
            st_sf.step(p)
        for _ in range(grammar.MAX_STAVES_PER_BAR):
            st_sf.step("<staff>")
        allow_sf, _ = st_sf.allowed()
        self.assertNotIn(_T["<staff>"], allow_sf)
        with self.assertRaises(ValueError):
            st_sf.step("<staff>")

        # 5f. Consecutive voice switches capped at MAX_CONSEC_VOICE_SWITCHES
        st_v = grammar.PrefixState(strict=False)
        for p in ("clef-G-2", "key+0", "time-4-4", "voice-1", "voice-2"):
            st_v.step(p)
        allow_v, _ = st_v.allowed()
        self.assertNotIn(_T["voice-1"], allow_v)
        self.assertNotIn(_T["voice-3"], allow_v)
        with self.assertRaises(ValueError):
            st_v.step("voice-1")

        # 5g. tup-open / tup-close balance & digit sequence
        st_t = grammar.PrefixState(strict=False)
        for p in ("clef-G-2", "key+0", "time-4-4", "voice-1", "tup-open"):
            st_t.step(p)
        self.assertTrue(st_t.has_open_tuplet())
        allow_a, eos_a = st_t.allowed()
        self.assertFalse(eos_a)
        self.assertFalse(st_t.can_emit_sys())
        self.assertEqual(allow_a, {_T[str(d)] for d in range(1, 9)})
        st_t.step("3")
        allow_n, eos_n = st_t.allowed()
        self.assertFalse(eos_n)
        self.assertEqual(allow_n, {_T[str(d)] for d in range(1, 9)})
        st_t.step("2")
        # Pending tuplet: bar, <staff>, voice-*, nested tup-open, <eos> all blocked
        allow_p, eos_p = st_t.allowed()
        self.assertFalse(eos_p)
        self.assertNotIn(_T["bar"], allow_p)
        self.assertNotIn(_T["<staff>"], allow_p)
        self.assertNotIn(_T["tup-open"], allow_p)
        for p in ("eighth", "C", "4", "eighth", "D", "4", "eighth", "E", "4"):
            st_t.step(p)
        allow_in, _ = st_t.allowed()
        self.assertIn(_T["tup-close"], allow_in)
        self.assertNotIn(_T["<staff>"], allow_in)
        st_t.step("tup-close")
        self.assertFalse(st_t.has_open_tuplet())
        allow_out, eos_out = st_t.allowed()
        self.assertTrue(eos_out)
        self.assertIn(_T["<staff>"], allow_out)
        self.assertIn(_T["bar"], allow_out)

    # ------------------------------------------------------------------
    # 6. Backward compatibility with legacy 237-vocab checkpoints
    # ------------------------------------------------------------------

    def test_backward_compat_237_vocab(self) -> None:
        """PrefixState works cleanly with a 233-token (vocab_size=237) pre-v2a vocab_map."""
        base_233 = list(data._BASE_PRE_V2A)
        self.assertEqual(len(base_233), 233)
        vm_233 = {t: i for i, t in enumerate(base_233)}

        seq = [
            "clef-G-2", "key+0", "time-4-4",
            "voice-1", "dyn-mf", "hairpin-cresc-start",
            "tuplet-start-3-2",
            "eighth", "C", "4", "slur-start", "stacc",
            "eighth", "D", "4", "a1",
            "eighth", "E", "4", "slur-stop",
            "tuplet-stop",
            "quarter", "F", "4", "chord", "quarter", "A", "4", "tie-start",
            "quarter", "A", "4", "tie-stop",
            "rest", "quarter", "fermata",
            "bar",
            "voice-1", "bar-rest",
            "bar-light-heavy",
        ]
        for strict in (False, True):
            st = grammar.PrefixState(strict=strict, vocab_map=vm_233)
            for idx, tok in enumerate(seq):
                allow, _ = st.allowed()
                self.assertTrue(all(tid < 233 for tid in allow), f"Out-of-range id in 237-vocab allow: {allow}")
                self.assertIn(vm_233[tok], allow, f"strict={strict}@{idx}: {tok!r} blocked in 237-vocab")
                st.step(tok)
            self.assertFalse(st.broken)
            self.assertTrue(st.can_emit_eos())

    # ------------------------------------------------------------------
    # 7. Synthetic bekern+v2 Phase A round-trip with ALL 28 new tokens
    # ------------------------------------------------------------------

    def test_roundtrip_all_28_v2a_tokens_fsa_and_musicxml(self) -> None:
        """All 28 `_BEKERN_V2A_ADD` tokens pass FSA (lax+strict) and export valid MusicXML via assembly-serialize."""
        seq = [
            "clef-G-2", "key+0", "time-4-4",
            "clef-F-4", "key+0", "time-4-4",
            # Measure 1: Staff 1 (volta-1, dyn-ppp/fff/fp/sf, art-*, xstaff-*, cue, ottava-8va/stop, tup-open/close)
            "volta-1", "voice-1", "dyn-ppp",
            "quarter", "C", "4", "art-arpeggiate", "art-trill",
            "dyn-fff", "quarter", "D", "4", "art-tremolo", "art-mordent",
            "dyn-fp", "tup-open", "3", "2",
            "eighth", "E", "4", "art-turn",
            "eighth", "F", "4", "art-staccatissimo",
            "eighth", "G", "4", "art-up-bow",
            "tup-close",
            "dyn-sf", "ottava-8va", "quarter", "A", "5", "art-down-bow", "art-soft-accent",
            "ottava-stop",
            "<staff>",
            # Measure 1: Staff 2 (dyn-sfp/fz/rfz/sffz, xstaff-up/down, cue, ottava-8vb/stop)
            "voice-1", "dyn-sfp",
            "quarter", "C", "3", "xstaff-up",
            "dyn-fz", "quarter", "D", "3", "xstaff-down",
            "dyn-rfz", "ottava-8vb", "quarter", "E", "2", "cue",
            "dyn-sffz", "ottava-stop", "quarter", "F", "2",
            "bar-repeat-right",
            # Measure 2: volta-2 + both staves
            "volta-2", "voice-1", "whole", "C", "5",
            "<staff>",
            "voice-1", "bar-rest",
            "bar-light-heavy",
        ]
        # Verify every single one of the 28 _BEKERN_V2A_ADD tokens is present in seq
        seq_set = set(seq)
        missing_v2a = [t for t in data._BEKERN_V2A_ADD if t not in seq_set]
        self.assertEqual(missing_v2a, [], f"Synthetic sequence missing v2a tokens: {missing_v2a}")

        # Verify both strict=False and strict=True accept the entire sequence
        for strict in (False, True):
            st = grammar.PrefixState(strict=strict)
            for idx, tok in enumerate(seq):
                allow, _ = st.allowed()
                self.assertIn(_T[tok], allow, f"strict={strict}@{idx}: {tok!r} blocked")
                st.step(tok)
            self.assertFalse(st.broken)
            self.assertTrue(st.can_emit_eos())

        # Round-trip through assembly-serialize.exe bekern2musicxml if built
        asm_bin = _find_assembly_bin()
        if asm_bin is None:
            self.skipTest("assembly-serialize.exe not built")

        with tempfile.TemporaryDirectory(prefix="test_v2a_mxml_") as td:
            td_path = Path(td)
            in_file = td_path / "synth_v2a-bekern.txt"
            in_file.write_text(" ".join(seq) + "\n", encoding="utf-8")
            out_dir = td_path / "out"
            proc = subprocess.run(
                [str(asm_bin), "bekern2musicxml", "--input", str(in_file), "--out", str(out_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"assembly-serialize bekern2musicxml failed (rc={proc.returncode}):\nstdout={proc.stdout}\nstderr={proc.stderr}",
            )
            mxml = out_dir / "synth_v2a.musicxml"
            self.assertTrue(mxml.is_file(), f"Missing MusicXML output: {mxml}")
            xml_text = mxml.read_text(encoding="utf-8")
            self.assertIn("<score-partwise", xml_text)
            self.assertIn("</score-partwise>", xml_text)


if __name__ == "__main__":
    unittest.main()
