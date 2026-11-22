"""Row-model retrain v2: dataset + training + ONNX export + gate.

Mirrors converter/layout-staff/src/onnx_model.rs preprocessing exactly:
grayscale -> resize width 248 (BILINEAR ~ Triangle) -> binarize ink at
--threshold -> 33-row replicate-padded windows -> per-row logit.
Contract kept: input `patch` [N,1,33,248] f32, output `logit` [N].

Train: big500 symbox pages ds0001-ds0004 (26 pages). Gate: held-out
ds0005 (the 4 SER pages): AUC + NMS peak-pick line recall vs pilot.
"""
import argparse, json, os, glob
import numpy as np
from PIL import Image

WIN_W = 248
HALF = 16

def load_page(stem, render_dir, symbox_dir, threshold):
    img = Image.open(os.path.join(render_dir, stem + '.png')).convert('L')
    W, H = img.size
    small = img.resize((WIN_W, H), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32)
    ink = (a < threshold).astype(np.float32)  # (H, 248)
    ds = stem.split('-')[0]
    bj = json.load(open(glob.glob(os.path.join(symbox_dir, ds, stem + '-boxes.json'))[0]))
    lines = sorted(set(int(round(y)) for s in bj['staves'] for y in s['lines']))
    lines = [y for y in lines if 0 <= y < H]
    label = np.zeros(H, dtype=np.float32)
    for y in lines:
        label[max(0, y - 1):min(H, y + 2)] = 1.0
    return ink, label, lines

def windows(ink):
    H = ink.shape[0]
    pad = np.pad(ink, ((HALF, HALF), (0, 0)), mode='edge')
    idx = np.arange(H)[:, None] + np.arange(2 * HALF + 1)[None, :]
    return pad[idx]  # (H, 33, 248)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--render-dir', default='data-gen/score-corpus/big500-renders')
    ap.add_argument('--symbox-dir', default='data-gen/score-corpus/big500-symbox')
    ap.add_argument('--train', nargs='+', default=['ds0001', 'ds0002', 'ds0003', 'ds0004'])
    ap.add_argument('--heldout', nargs='+', default=['ds0005'])
    ap.add_argument('--threshold', type=float, default=140.0)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='tmp-train/row-retrain/rownet_v2.onnx')
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def stems(dss):
        out = []
        for ds in dss:
            for b in sorted(glob.glob(os.path.join(args.symbox_dir, ds, '*-boxes.json'))):
                out.append(os.path.basename(b).replace('-boxes.json', ''))
        return out

    tr_stems, ho_stems = stems(args.train), stems(args.heldout)
    print(f'train pages: {len(tr_stems)}, heldout pages: {len(ho_stems)}', flush=True)

    def build(stems_):
        Xs, ys = [], []
        for s in stems_:
            ink, lab, _ = load_page(s, args.render_dir, args.symbox_dir, args.threshold)
            print(f'  loaded {s}: ink {ink.shape}, pos {lab.sum():.0f}', flush=True)
            Xs.append(windows(ink))
            ys.append(lab)
        return (np.concatenate(Xs).astype(np.float32)[:, None, :, :],
                np.concatenate(ys).astype(np.float32))
    trX, try_ = build(tr_stems)
    hoX, hoy = build(ho_stems)
    print(f'trX {trX.shape} ({trX.nbytes / 1e9:.2f} GB), hoX {hoX.shape}', flush=True)

    pos = float(try_.sum())
    neg = float(len(try_) - pos)
    print(f'train rows: {len(try_)}, pos frac: {pos / len(try_):.4f}', flush=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev, flush=True)

    class RowNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(
                nn.Conv2d(1, 16, (3, 9), padding=(1, 4)), nn.ReLU(),
                nn.AvgPool2d((1, 4)),           # 33 x 62
                nn.Conv2d(16, 32, (3, 5), padding=(1, 2)), nn.ReLU(),
                nn.AvgPool2d((1, 2)),           # 33 x 31
                nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((33, 1)),  # width-pooled: 1D over rows
            )
            self.head = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.3),
                nn.Linear(48 * 33, 128), nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.head(self.feat(x)).squeeze(1)

    net = RowNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos]).to(dev))
    trX_t = torch.from_numpy(trX)
    try_t = torch.from_numpy(try_)
    n = len(trX_t)
    rng = np.random.RandomState(args.seed)
    for ep in range(args.epochs):
        net.train()
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = trX_t[idx].to(dev), try_t[idx].to(dev)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            nb += len(idx)
        print(f'ep {ep + 1}/{args.epochs} loss {tot / nb:.4f}', flush=True)

    net.eval()
    with torch.no_grad():
        logits = []
        for i in range(0, len(hoX), 4096):
            logits.append(net(torch.from_numpy(hoX[i:i + 4096]).to(dev)).cpu().numpy())
    lo = np.concatenate(logits)
    prob = 1 / (1 + np.exp(-lo))
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(hoy, prob)
    ap = average_precision_score(hoy, prob)
    print(f'HELDOUT ds0005: AUC {auc:.4f} AP {ap:.4f}', flush=True)

    # NMS peak-pick line recall (mirrors the classical within-2px metric).
    def peaks(p, thr=0.5, dist=5):
        out = []
        for i in range(len(p)):
            if p[i] < thr:
                continue
            lo_i, hi_i = max(0, i - dist), min(len(p), i + dist + 1)
            if p[i] == p[lo_i:hi_i].max() and (not out or i - out[-1] >= dist):
                out.append(i)
        return out

    det = peaks(prob)
    # GT lines for heldout pages (concat with offsets).
    gt, off, rec = [], 0, 0
    for s in ho_stems:
        _, _, lines = load_page(s, args.render_dir, args.symbox_dir, args.threshold)
        gt += [y + off for y in lines]
        H = int(hoy.shape[0] / len(ho_stems))  # all pages same height here
        off += H
    # NOTE: assumes equal heights; verified 3507 for all symbox pages.
    hits = sum(1 for y in gt if any(abs(y - d) <= 2 for d in det))
    print(f'peak-pick: {hits}/{len(gt)} lines within 2px = {hits / len(gt):.3f}, '
          f'{len(det)} peaks total', flush=True)

    dummy = torch.zeros(1, 1, 33, WIN_W)
    torch.onnx.export(net.cpu(), dummy, args.out, input_names=['patch'],
                      output_names=['logit'], opset_version=17,
                      dynamic_axes={'patch': {0: 'N'}, 'logit': {0: 'N'}})
    print('wrote', args.out, flush=True)

if __name__ == '__main__':
    main()
