import glob, json, os, sys
import numpy as np
sys.path.insert(0, 'tmp-train/reproject')
import probe as P

outd = 'tmp-train/reproject/probe-out'
dots = [(float(x), float(y))
        for y in np.linspace(0.1, 0.9, P.ROWS) * P.GRID_H
        for x in np.linspace(0.1, 0.9, P.COLS) * P.GRID_W]
all_e, all_m = [], []
for jp in sorted(glob.glob(os.path.join(outd, 'grid_p*_params.json')),
                 key=lambda q: int(q.split('_p')[1].split('_')[0])):
    tag = jp.split('_p')[1].split('_')[0]
    p = json.load(open(jp))
    outs = sorted(glob.glob(os.path.join(outd, f'grid_p{tag}.png')))
    if not outs:
        print(f'seed {tag}: NO IMAGE'); continue
    pred = P.forward(p, dots, P.GRID_W, P.GRID_H)
    det = P.detect(outs[0])
    errs = P.score(pred, det)
    coef, res = P.affine_fit(pred, det)
    e = np.array(errs) if errs else np.array([float('nan')])
    md = min(p['width'], p['height'])
    aff = (f'affresid med={np.median(res):.2f} max={res.max():.2f}'
           if res is not None else 'affine NA')
    print(f'seed {tag}: mode={p.get("curl_side")}/{p.get("bend_mode")} '
          f'match={len(errs)}/{len(dots)} mean={e.mean():.2f}px '
          f'med={np.median(e):.2f}px max={e.max():.2f}px '
          f'({100*e.max()/md:.2f}% mindim) {aff}')
    all_e += errs
    all_m.append(len(errs) / len(dots))
e = np.array(all_e)
print(f'ALL: n={len(all_e)} matchrate={np.mean(all_m):.3f} '
      f'mean={e.mean():.2f}px med={np.median(e):.2f}px max={e.max():.2f}px')
