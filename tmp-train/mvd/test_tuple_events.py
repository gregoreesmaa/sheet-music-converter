"""Lossless bekern <-> event-tuple compiler (S18 step-1 gate).

Round-trip must be byte-exact; facet parsing is asserted on hand-picked
cases. Run: ``.\\.venv\\Scripts\\python.exe -m unittest tmp-train/mvd/test_tuple_events.py -v``
"""

import glob
import os
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tuple_events as te


def rt(s):
    return te.tuples_to_bekern(te.bekern_to_tuples(s.split()))


class TestTupleCompiler(unittest.TestCase):
    def test_header_and_simple_note(self):
        toks = 'clef-G-2 key+4 time-2-4 voice-1 quarter B 4'.split()
        evs = te.bekern_to_tuples(toks)
        self.assertEqual([e['kind'] for e in evs],
                         ['HEADER', 'HEADER', 'HEADER', 'VOICE', 'NOTE'])
        n = evs[-1]
        self.assertEqual((n['voice'], n['dur'], n['step'], n['oct']), (1, 'quarter', 'B', 4))
        self.assertEqual(n['diatonic_pitch'], 4 * 7 + 6)
        self.assertEqual(te.tuples_to_bekern(evs), toks)

    def test_full_mods_and_chord(self):
        toks = ('voice-2 half E 3 a1 dot dot tie-start slur-start stacc dyn-f '
                'hairpin-cresc-start chord half G 3 a-1 tie-stop slur-stop bar-double').split()
        evs = te.bekern_to_tuples(toks)
        kinds = [e['kind'] for e in evs]
        self.assertEqual(kinds, ['VOICE', 'NOTE', 'CHORD_NOTE', 'BAR'])
        n, c = evs[1], evs[2]
        self.assertEqual((n['dots'], n['alter'], n['voice']), (2, 'a1', 2))
        self.assertEqual(n['ties'], ['start'])
        self.assertEqual(n['slurs'], ['start'])
        self.assertEqual(n['articulations'], ['stacc'])
        self.assertEqual(n['dyn'], 'dyn-f')
        self.assertIsNone(c['voice'])
        self.assertEqual(c['diatonic_pitch'], 3 * 7 + 4)
        self.assertEqual(evs[3]['variant'], 'double')
        self.assertEqual(te.tuples_to_bekern(evs), toks)

    def test_tuplet_rest_bar_rest(self):
        toks = ('tuplet-start-3-2 eighth E 4 a1 voice-6 quarter G 3 chord quarter B 3 '
                'tuplet-stop bar voice-3 rest quarter voice-1 bar-rest').split()
        evs = te.bekern_to_tuples(toks)
        kinds = [e['kind'] for e in evs]
        self.assertEqual(kinds, ['TUPLET_START', 'NOTE', 'VOICE', 'NOTE', 'CHORD_NOTE',
                                 'TUPLET_STOP', 'BAR', 'VOICE', 'REST', 'VOICE', 'BAR_REST'])
        self.assertEqual(evs[0]['ratio'], '3-2')
        self.assertEqual(evs[-1]['voice'], 1)
        self.assertEqual(te.tuples_to_bekern(evs), toks)

    def test_standalone_mods(self):
        toks = 'bar voice-1 bar-rest dyn-pp voice-2 half B 3'.split()
        evs = te.bekern_to_tuples(toks)
        self.assertEqual([e['kind'] for e in evs],
                         ['BAR', 'VOICE', 'BAR_REST', 'DYN', 'VOICE', 'NOTE'])
        self.assertEqual(evs[3]['mark'], 'dyn-pp')
        self.assertEqual(te.tuples_to_bekern(evs), toks)

    def test_grace_and_midstream_header(self):
        toks = ('voice-1 grace eighth G 4 slur-start quarter F 4 '
                'clef-F-4 key-2 time-3-4').split()
        evs = te.bekern_to_tuples(toks)
        self.assertTrue(evs[1]['grace'])
        self.assertFalse(evs[2]['grace'])
        self.assertEqual(te.tuples_to_bekern(evs), toks)

    def test_diatonic_index(self):
        self.assertEqual(te.diatonic_index('C', 4), 28)
        self.assertEqual(te.diatonic_index('B', 3), 27)
        self.assertEqual(te.diatonic_index('A', 2), 19)
        self.assertEqual(te.NO_PITCH, 63)


class TestTupleRoundTripCorpus(unittest.TestCase):
    def test_sysdev_and_corpus_sample_exact(self):
        files = sorted(glob.glob('data-gen/score-corpus/gate20/sys-dev/refs/*-bekern.txt'))
        pool = sorted(glob.glob('tmp-train/mvd-sysdata-t32/*-bekern.txt'))
        files += random.Random(0).sample(pool, 300)
        self.assertGreaterEqual(len(files), 326)
        bad = []
        for f in files:
            with open(f, encoding='utf-8') as fh:
                toks = fh.read().split()
            if te.tuples_to_bekern(te.bekern_to_tuples(toks)) != toks:
                bad.append(os.path.basename(f))
                if len(bad) >= 5:
                    break
        self.assertEqual(bad, [])


if __name__ == '__main__':
    unittest.main()
