"""Self-contained checks for train.filter_pages (no data, no torch).

    .venv/Scripts/python.exe tmp-train/mvd/test_pages.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402

PAIRS = [('a.png', 'a.txt', 'ds0318-1'),
         ('b.png', 'b.txt', 'ds0318-10'),
         ('c.png', 'c.txt', 'ds0001-2')]


def test_substring_legacy_keeps_collision():
    # Legacy behavior preserved (and documented): substring over-matches.
    sel = train.filter_pages(PAIRS, substrings=['ds0318-1'])
    assert [p[2] for p in sel] == ['ds0318-1', 'ds0318-10'], sel
    print('PASS legacy substring behavior pinned (collision included)')


def test_exact_file_avoids_collision():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'band.txt'
        f.write_text('ds0318-1\n', encoding='utf-8')
        sel = train.filter_pages(PAIRS, exact_file=str(f))
    assert [p[2] for p in sel] == ['ds0318-1'], sel
    print('PASS exact file selects one stem only')


def test_union_dedupes():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'band.txt'
        f.write_text('ds0318-1\nds0001-2\n', encoding='utf-8')
        sel = train.filter_pages(PAIRS, substrings=['ds0318-1'],
                                 exact_file=str(f))
    assert [p[2] for p in sel] == ['ds0318-1', 'ds0318-10', 'ds0001-2'], sel
    print('PASS union of substrings + exact, deduped in order')


def test_no_filter_returns_all():
    assert train.filter_pages(PAIRS) == PAIRS
    print('PASS no filter returns input unchanged')


if __name__ == '__main__':
    test_substring_legacy_keeps_collision()
    test_exact_file_avoids_collision()
    test_union_dedupes()
    test_no_filter_returns_all()
    print('ALL PAGES-FILTER TESTS PASSED')
