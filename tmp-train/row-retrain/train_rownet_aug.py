"""Row-model retrain v3: geometry-preserving photometric augmentation.

Clean 300dpi renders (ds0001-ds0004) + 1 augmented copy each (blur,
brightness/contrast, downscale, JPEG, noise — all row-preserving, so
symbox line labels stay exact). Contract kept: `patch` [N,1,33,248]
f32 in, `logit` [N] out. Gate: held-out clean ds0005 (AUC + NMS
recall) + photo probe counts (labels unknown on warped photos).
"""
import argparse, json, os, glob, io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

WIN_W = 248
HALF = 16


def load_gray(stem, render_dir):
    img = Image.open(os.path.join(render_dir, stem + '.png')).convert('L')
    return img


def prep(img, H_fixed=None):
    W, H = img.size
    small = img.resize((WIN_W, H), Image.BILINEAR)
    return np.asarray(small, dtype=np.uint8)


def augment(gray_pil, rng):
    img = gray_pil
    if rng.rand() < 0.8:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.choice([0.5, 1.0, 1.5]))))
    if rng.rand() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.7, 1.3)))
    if rng.rand() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.7, 1.3)))
    if rng.rand() < 0.5:
        w, h = img.size
        dw, dh = max(1, w // 2), max(1, h // 2)
        img = img.resize((dw, dh), Image.BICUBIC).resize((w, h), Image.BICUBIC)
    if rng.rand() < 0.5:
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=int(rng.choice([40, 70])))
        buf.seek(0)
        img = Image.open(buf).convert('L')
    if rng.rand() < 0.6:
        a = np.asarray(img, dtype=np.float32)
        a += rng.normal(0, float(rng.uniform(4, 12)), a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return img


def gt_lines(stem, symbox_dir, H):
    ds = stem.split('-')[0]
    bj = json.load(open(glob.glob(os.path.join(symbox_dir, ds, stem + '-boxes.json'))[0]))
    lines = sorted(set(int(round(y)) for s in bj['staves'] for y in s['lines']))
    return [y for y in lines if 0 <= y < H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--render-dir', default='data-gen/score-corpus/big500-renders')
    ap.add_argument('--symbox-dir', default='data-gen/score-corpus/big500-symbox')
    ap.add_argument('--train', nargs='+', default=['ds0001', 'ds0002', 'ds0003', 'ds0004'])
    ap.add_argument('--heldout', nargs='+', default=['ds0005'])
    ap.add_argument('--threshold', type=float, default=140.0)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='tmp-train/row-retrain/rownet_v3.onnx')
    ap.add_argument('--probe', nargs='*', default=[])
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

    # Build uint8 strip memmap: clean page + 1 augmented copy each.
    strips, labels, heights = [], [], []
    rng = np.random.RandomState(args.seed)
    for i, s in enumerate(tr_stems):
        gray = load_gray(s, args.render_dir)
        H = gray.size[1]
        for v, g in enumerate([gray, augment(gray, rng)]):
            strips.append(prep(g))
            lab = np.zeros(H, dtype=np.float32)
            for y in gt_lines(s, args.symbox_dir, H):
                lab[max(0, y - 1):min(H, y + 2)] = 1.0
            labels.append(lab)
            heights.append(H)
        print(f'  {s}: H {H} pos {labels[-2].sum():.0f}', flush=True)
    S = np.concatenate(strips)  # (rows, 248) uint8
    Y = np.concatenate(labels)
    print(f'strips {S.shape} ({S.nbytes / 1e9:.2f} GB uint8), pos frac {Y.mean():.4f}', flush=True)
    bounds = np.cumsum([0] + heights)

    def batch_windows(idx, thr):
        # idx: global row indices; replicate-pad within each page segment.
        out = np.empty((len(idx), 33, WIN_W), dtype=np.float32)
        for j, r in enumerate(idx):
            seg = int(np.searchsorted(bounds, r, side='right') - 1)
            rr = r - bounds[seg]
            H = heights[seg]
            rows = np.clip(rr + np.arange(-HALF, HALF + 1), 0, H - 1)
            out[j] = S[bounds[seg] + rows].astype(np.float32)
        return (out < thr).astype(np.float32)[:, None, :, :]

    pos = float(Y.sum())
    neg = float(len(Y) - pos)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev, flush=True)

    class RowNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(
                nn.Conv2d(1, 16, (3, 9), padding=(1, 4)), nn.ReLU(),
                nn.AvgPool2d((1, 4)),
                nn.Conv2d(16, 32, (3, 5), padding=(1, 2)), nn.ReLU(),
                nn.AvgPool2d((1, 2)),
                nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((33, 1)),
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
    n = len(Y)
    brng = np.random.RandomState(args.seed)
    for ep in range(args.epochs):
        net.train()
        perm = brng.permutation(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.from_numpy(batch_windows(idx, args.threshold)).to(dev)
            yb = torch.from_numpy(Y[idx]).to(dev)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f'ep {ep + 1}/{args.epochs} loss {tot / n:.4f}', flush=True)

    torch.save(net.state_dict(), args.out + '.pt')
    print('wrote', args.out + '.pt', flush=True)

    # Heldout clean gate (exact labels).
    net.eval()
    with torch.no_grad():
        probs = []
        for s in ho_stems:
            gray = load_gray(s, args.render_dir)
            H = gray.size[1]
            ink = (prep(gray).astype(np.float32) < args.threshold).astype(np.float32)
            pad = np.pad(ink, ((HALF, HALF), (0, 0)), mode='edge')
            idx = np.arange(H)[:, None] + np.arange(2 * HALF + 1)[None, :]
            X = torch.from_numpy(pad[idx][:, None].astype(np.float32))
            lo = []
            for i in range(0, H, 4096):
                lo.append(net(X[i:i + 4096].to(dev)).cpu().numpy())
            probs.append(1 / (1 + np.exp(-np.concatenate(lo))))
    from sklearn.metrics import roc_auc_score, average_precision_score

    def page_labels(s):
        H = load_gray(s, args.render_dir).size[1]
        lab = np.zeros(H, dtype=np.float32)
        for y in gt_lines(s, args.symbox_dir, H):
            lab[max(0, y - 1):min(H, y + 2)] = 1.0
        return lab

    hoy = np.concatenate([page_labels(s) for s in ho_stems])
    hop = np.concatenate(probs)
    print('HELDOUT ds0005: AUC %.4f AP %.4f' % (
        roc_auc_score(hoy, hop), average_precision_score(hoy, hop)), flush=True)

    def peaks(p, thr=0.5, dist=5):
        out = []
        for i in range(len(p)):
            if p[i] < thr:
                continue
            li, hi = max(0, i - dist), min(len(p), i + dist + 1)
            if p[i] == p[li:hi].max() and (not out or i - out[-1] >= dist):
                out.append(i)
        return out

    off = 0
    for s, p in zip(ho_stems, probs):
        H = len(p)
        gt = gt_lines(s, args.symbox_dir, H)
        det = peaks(p)
        hits = sum(1 for y in gt if any(abs(y - x) <= 2 for x in det))
        print(f'  {s}: GT {len(gt)} det {len(det)} recall {hits}/{len(gt)}={hits / len(gt):.3f}',
              flush=True)

    dummy = torch.zeros(1, 1, 33, WIN_W)
    torch.onnx.export(net.cpu(), dummy, args.out, input_names=['patch'],
                      output_names=['logit'], opset_version=18,
                      dynamic_axes={'patch': {0: 'N'}, 'logit': {0: 'N'}})
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
