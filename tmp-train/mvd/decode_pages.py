"""Greedy-decode page PNGs with a trained MVD checkpoint to hyp bekern files.

Counterpart to train.py (same pairing, same image prep, same vocab):
loads ``ckpt-<step>.pt`` weights into ``model.MvdNet`` and runs
``greedy()`` per page. Output ``<base>-hyp-bekern.txt`` files feed
``assembly-serialize bekern-decode`` + ``eval-harness strict-f1``,
which is the learned-decoder measurement chain (goal units, not TER).

Deterministic: greedy search, no sampling. Run with the repo-root
.venv python:

    .venv/Scripts/python.exe tmp-train/mvd/decode_pages.py \\
        --ckpt tmp-train/mvd-runs/epoch1c/ckpt-2529.pt \\
        --renders data-gen/score-corpus/big500-renders \\
        --targets tmp-train/mvd-data --pages ds0001-1 \\
        --out tmp-train/mvd-runs/epoch1c/hyp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import model  # noqa: E402
import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--renders', required=True)
    ap.add_argument('--targets', required=True)
    ap.add_argument('--pages', nargs='*', default=None,
                    help='only pages containing one of these substrings')
    ap.add_argument('--pages-file', default=None,
                    help='stem-per-line list for EXACT page selection '
                         '(train.filter_pages; unions with --pages)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--height', type=int, default=1024)
    ap.add_argument('--max-len', type=int, default=4608,
                    help='hard cap (fair-probe: 4608 covers the longest '
                         'corpus page at 4414 tokens)')
    ap.add_argument('--constrain', action='store_true',
                    help='mask each greedy step to the bekern grammar '
                         '(output always parses in bekern-decode)')
    ap.add_argument('--search', choices=('greedy', 'kv', 'beam'),
                    default='greedy',
                    help='greedy = legacy O(T^2) reference; kv = KV-cached '
                         'greedy (identical tokens, E4 S1); beam = cached '
                         'masked beam + length norm (E4 S4, needs cache)')
    ap.add_argument('--strict-mask', action='store_true',
                    help='validator-predicate grammar mask (E4 S3: '
                         'single-alter + alter-before-dots, sequential '
                         'voices, no dup bar/voice markers)')
    ap.add_argument('--min-len', type=int, default=-1,
                    help='forbid EOS before this many tokens; -1 = auto '
                         '(5 tokens/encoder-cell, floor 64); 0 disables')
    ap.add_argument('--eos-bias', type=float, default=0.0,
                    help='scalar added to the EOS logit each step')
    ap.add_argument('--rep-penalty', type=float, default=1.0,
                    help='CTRL-style repetition penalty over history '
                         '(1.0 disables)')
    ap.add_argument('--run-cap', type=int, default=0,
                    help='max identical tokens in a row (0 disables; '
                         'corpus max run is 2, so 2 blocks the 3rd)')
    ap.add_argument('--cycle-repeats', type=int, default=0,
                    help='max adjacent repeats of a <=8-token cycle '
                         '(0 disables; corpus max is 10)')
    ap.add_argument('--no-repeat-ngram', type=int, default=0,
                    help='classic global n-gram block, header-exempt '
                         '(0 disables; corpus contradicts it: 70.8%% of '
                         'GT positions repeat a seen trigram)')
    ap.add_argument('--beam', type=int, default=3,
                    help='beam width for --search beam (ban >5 without '
                         'length norm; the norm is always applied)')
    ap.add_argument('--len-alpha', type=float, default=0.6,
                    help='length-norm exponent for beam selection')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device, '| torch:', torch.__version__)
    print('search=%s strict=%s min-len=%s eos-bias=%s rep-penalty=%s '
          'run-cap=%d cycle-repeats=%d no-repeat-ngram=%d beam=%d '
          'len-alpha=%s max-len=%d'
          % (args.search, args.strict_mask, args.min_len, args.eos_bias,
             args.rep_penalty, args.run_cap, args.cycle_repeats,
             args.no_repeat_ngram, args.beam, args.len_alpha,
             args.max_len))

    pairs = train.list_pairs(args.renders, args.targets)
    pairs = train.filter_pages(pairs, args.pages, args.pages_file)
    assert pairs, 'no pages selected'
    print('pages:', len(pairs))

    net = model.MvdNet(vocab=len(data.VOCAB) + 3).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    print('loaded:', args.ckpt)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    capped = 0
    with torch.no_grad():
        for png, _, base in pairs:
            img = train.load_image(png, args.height).unsqueeze(0).to(device)
            min_len = args.min_len
            if min_len < 0:
                # Min-length proportional to image width: 5 tokens per
                # encoder cell (calibrated: min healthy dev-degraded
                # ratio is 11.6/cell, so 5x is a 0.43x safety floor),
                # never below 64 (short pages still clear it).
                cells = max(1, img.shape[-1] // 32)
                min_len = max(64, 5 * cells)
            if args.search == 'beam':
                hyp = net.beam_search(
                    img, max_len=args.max_len, constrain=args.constrain,
                    strict=args.strict_mask, beam=args.beam,
                    len_alpha=args.len_alpha, min_len=min_len,
                    eos_bias=args.eos_bias, rep_penalty=args.rep_penalty,
                    run_cap=args.run_cap,
                    cycle_repeats=args.cycle_repeats,
                    no_repeat_ngram=args.no_repeat_ngram)[0].tolist()
            elif args.search == 'kv':
                hyp = net.greedy_kv(
                    img, max_len=args.max_len, constrain=args.constrain,
                    strict=args.strict_mask, min_len=min_len,
                    eos_bias=args.eos_bias, rep_penalty=args.rep_penalty,
                    run_cap=args.run_cap,
                    cycle_repeats=args.cycle_repeats,
                    no_repeat_ngram=args.no_repeat_ngram)[0].tolist()
            else:
                hyp = net.greedy(img, max_len=args.max_len,
                                 constrain=args.constrain)[0].tolist()
            if data.EOS_ID in hyp:
                hyp = hyp[:hyp.index(data.EOS_ID)]
            else:
                capped += 1
            toks = data.decode([t for t in hyp
                                if t not in (data.PAD_ID, data.BOS_ID,
                                             data.EOS_ID)])
            (out / ('%s-hyp-bekern.txt' % base)).write_text(
                ' '.join(toks) + '\n', encoding='utf-8')
    print('wrote %d hyp files -> %s (hit max_len without EOS: %d)'
          % (len(pairs), args.out, capped))


if __name__ == '__main__':
    main()
