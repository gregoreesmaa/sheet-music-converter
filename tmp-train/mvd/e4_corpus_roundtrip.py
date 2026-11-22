"""Full-corpus strict-mask round-trip (E4 S3 validity evidence).

Walks every page in tmp-train/mvd-data with PrefixState(strict=True):
every GT continuation must be admitted and EOS legal at end. Any
failure names the file/position/token (the mask then contradicts GT
and must be fixed, never tuned around). Incremental O(n) walk.
"""
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import grammar  # noqa: E402

t0 = time.perf_counter()
files = sorted(glob.glob(str(Path(__file__).resolve().parent / '..'
                              / 'mvd-data' / '*-bekern.txt')))
bad = []
positions = 0
for f in files:
    toks = data.load_bekern(f)
    ids = data.encode(toks)
    st = grammar.PrefixState(strict=True)
    for k, tok in enumerate(toks):
        allow, _ = st.allowed()
        positions += 1
        if ids[k] not in allow:
            bad.append((Path(f).name, k, tok))
            break
        try:
            st.step(tok)
        except ValueError as e:
            bad.append((Path(f).name, k, tok, str(e)))
            break
    else:
        allow, eos_ok = st.allowed()
        if not eos_ok:
            bad.append((Path(f).name, 'END', 'EOS-illegal'))
        continue
print('files=%d positions=%d failures=%d (%.1fs)'
      % (len(files), positions, len(bad), time.perf_counter() - t0))
for b in bad[:20]:
    print('FAIL', b)
sys.exit(1 if bad else 0)
