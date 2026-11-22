"""E4 S1 parity proof: cached step == full decoder forward.

1. Layer parity on random inputs (full layer forward vs incremental
   _layer_step chain from empty cache): reports max abs diff.
2. End-to-end token parity on real pages (greedy vs greedy_kv with
   neutral knobs): asserts identical id streams.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import model  # noqa: E402
import train  # noqa: E402

CKPT = (Path(__file__).resolve().parents[1] / 'mvd-runs' / 'fullmix'
        / 'ckpt-13000.pt')
GATE = (Path(__file__).resolve().parents[2] / 'data-gen' / 'score-corpus'
        / 'gate20')


def layer_parity(net, device):
    torch.manual_seed(0)
    layer = net.decoder.layers[0]
    p = model._layer_params(layer)
    mem = torch.randn(2, 37, net.d_model, device=device)
    mkv = model._mem_kv(mem, p[1])
    n = 9
    tgt = torch.randint(0, len(data.VOCAB), (2, n), device=device)
    emb = (net.tok_emb(tgt)
           + model._sinusoidal_1d(n, net.d_model, device)[None])
    mask = torch.nn.Transformer.generate_square_subsequent_mask(
        n, device=device)
    ref = layer(emb, mem, tgt_mask=mask)
    kc = vc = None
    outs = []
    for pos in range(n):
        x1 = (net.tok_emb(tgt[:, pos:pos + 1])
              + model._sinusoidal_1d_pos(pos, net.d_model, device)
              [None, None, :])
        x1, kc, vc = model._layer_step(layer, p, x1, kc, vc, mkv)
        outs.append(x1)
    got = torch.cat(outs, 1)
    d = (ref - got).abs().max().item()
    print('layer parity: max abs diff = %.3g' % d)
    return d


def end_to_end_parity(net, device, pages):
    net.eval()
    for pid, dom in pages:
        man = __import__('json').loads(
            (GATE / 'gate_manifest.json').read_text())
        pg = next(p for p in man['pages'] if p['id'] == pid)
        img = train.load_image(str(GATE / pg[dom]), 1024).unsqueeze(0)
        img = img.to(device)
        with torch.no_grad():
            a = net.greedy(img, max_len=700,
                           constrain=True)[0].tolist()
            b = net.greedy_kv(img, max_len=700,
                              constrain=True)[0].tolist()
        match = a == b
        print('%-14s %-8s greedy=%-5d kv=%-5d identical=%s'
              % (pid, dom, len(a), len(b), match))
        if not match:
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print('  first diff at %d: %s vs %s'
                          % (i, data.VOCAB[x] if x < len(data.VOCAB) else x,
                             data.VOCAB[y] if y < len(data.VOCAB) else y))
                    break
            return False
    return True


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    net = model.MvdNet(vocab=len(data.VOCAB) + 3).to(device)
    net.load_state_dict(torch.load(str(CKPT), map_location=device,
                                   weights_only=True))
    net.eval()
    d = layer_parity(net, device)
    assert d < 1e-4, d
    ok = end_to_end_parity(net, device, [('ds0367-2', 'degraded'),
                                         ('ds0372-2', 'degraded')])
    assert ok
    print('PARITY OK (layer diff %.3g, token streams identical)' % d)


if __name__ == '__main__':
    main()
