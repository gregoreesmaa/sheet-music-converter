import json, math, sys
sys.path.insert(0, 'tmp-train/reproject')
import probe as P

p = json.load(open('tmp-train/reproject/probe-out/grid_p5_params.json'))
print('curl_side =', repr(p.get('curl_side')))
print('keys:', [k for k in p if 'rot' in k or 'bend' in k or 'curl' in k])
ix, iy = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0, float(sys.argv[2]) if len(sys.argv) > 2 else 160.0
w, h = 1200.0, 1600.0
print('start', (ix, iy), 'wh', (w, h))
x, y = P.bend_forward(p['curl_side'], p['curl_px'], p.get('curl_shade'),
                      p['bend_spine_cx'], p['bend_spine_w'], p['bend_ripple_fx'],
                      p['bend_ripple_fy'], p['bend_ripple_ph'], w, h, ix, iy)
print('bend', (x, y))
src = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
dst = [(sx + dx, sy + dy) for (sx, sy), (dx, dy) in zip(src, p['corner_offsets'])]
H = P.solve_h(src, dst)
x, y = P.apply_h(H, x, y)
print('persp', (x, y))
l, t, r, b = p['frame_pad_px']
x, y = x + l, y + t
w, h = w + l + r, h + t + b
print('frame', (x, y), 'wh', (w, h))
th = math.radians(p['rotation_deg'])
snt, cst = math.sin(-th), math.cos(th)
c0x, c0y = w / 2.0, h / 2.0
w2 = math.ceil(w * abs(cst) + h * abs(snt))
h2 = math.ceil(h * abs(cst) + w * abs(snt))
c1x, c1y = w2 / 2.0, h2 / 2.0
dx, dy = x - c0x, y - c0y
x, y = c1x + dx * cst + dy * snt, c1y - dx * snt + dy * cst
print('rot', (x, y), 'canvas', (w2, h2))
w, h = float(w2), float(h2)
print('lens gate:', abs(p['dist_k1']) > 0.005 or abs(p['dist_k2']) > 0.002)
x *= int(w * p['scale_x']) / w
y *= int(h * p['scale_y']) / h
print('final', (x, y))
print('probe.forward:', P.forward(p, [(ix, iy)], 1200, 1600))
