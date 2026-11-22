//! Capture & handling artifacts for OMR training data.
//!
//! Buffer: RGB interleaved `f32` in `0..=255`, row-major, length `w*h*3`.
//! Every effect is margin/edge-biased so notation is never covered.
//! All randomness comes from the passed `rng` (deterministic per seed).

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::f32::consts::PI;

/// Sampled knobs for the capture/handling degradation pass.
#[derive(serde::Serialize, Debug, Clone)]
pub struct CaptureParams {
    pub enabled: bool,
    pub motion_angle_deg: f32,
    pub motion_radius: u32,
    pub fingerprint_edge: u8,
    pub fingerprint_x: u32,
    pub fingerprint_y: u32,
    pub fingerprint_rx: u32,
    pub fingerprint_ry: u32,
    pub fingerprint_oil: f32,
    pub tape_height: u32,
    pub tape_opacity: f32,
    pub tape_edge: u8,
    pub tape_start_frac: f32,
    pub tape_len_frac: f32,
    pub residue_count: u32,
    pub residue_rects: Vec<[u32; 4]>,
    pub hole_x: u32,
    pub hole_y: u32,
    pub hole_radius: u32,
    pub sticky_corner: u8,
    pub sticky_size: u32,
    pub sticky_tone: u8,
    pub exposure_lift: f32,
    pub exposure_compress: f32,
    pub banding_amplitude: f32,
    pub banding_period: u32,
}

fn ru(rng: &mut ChaCha8Rng, lo: u32, hi: u32, max: u32) -> u32 {
    let hi = hi.min(max).max(1);
    let lo = lo.min(hi.saturating_sub(1));
    if lo >= hi { lo } else { rng.gen_range(lo..hi) }
}

fn off() -> CaptureParams {
    CaptureParams {
        enabled: false, motion_angle_deg: 0.0, motion_radius: 0,
        fingerprint_edge: 0, fingerprint_x: 0, fingerprint_y: 0,
        fingerprint_rx: 0, fingerprint_ry: 0, fingerprint_oil: 0.0,
        tape_height: 0, tape_opacity: 0.0,
        tape_edge: 0, tape_start_frac: 0.0, tape_len_frac: 0.0,
        residue_count: 0, residue_rects: Vec::new(),
        hole_x: 0, hole_y: 0, hole_radius: 0,
        sticky_corner: 0, sticky_size: 0, sticky_tone: 0,
        exposure_lift: 0.0, exposure_compress: 1.0,
        banding_amplitude: 0.0, banding_period: 4,
    }
}

/// `px` is the pixel-scale factor (1.0 = native): the motion-blur radius
/// and sensor-banding period scale with it; `px = 1.0` is bit-identical to
/// the legacy sampling. Relative-geometry knobs (tape, holes, fingerprints)
/// and degeneracy floors stay in output pixels by design.
pub fn apply(buf: &mut [f32], w: u32, h: u32, rng: &mut ChaCha8Rng, strength: f32, px: f32) -> CaptureParams {
    let s = strength.clamp(0.0, 1.0);
    if s <= 0.0 || w < 8 || h < 8 || buf.len() < w as usize * h as usize * 3 {
        return off();
    }
    let base = w.min(h) as f32;
    let band = (w as f32 * 0.10) as u32;
    let vband = (h as f32 * 0.10) as u32;
    let fedge = rng.gen_range(0..4u8);
    let (fx, fy) = match fedge {
        0 => (ru(rng, 2, band.max(3), w), ru(rng, vband, h.saturating_sub(vband), h)),
        1 => (ru(rng, w.saturating_sub(band), w.saturating_sub(2), w), ru(rng, vband, h.saturating_sub(vband), h)),
        2 => (ru(rng, band, w.saturating_sub(band), w), ru(rng, 2, vband.max(3), h)),
        _ => (ru(rng, band, w.saturating_sub(band), w), ru(rng, h.saturating_sub(vband), h.saturating_sub(2), h)),
    };
    let frx = (base * rng.gen_range(0.03..0.06)).clamp(3.0, base) as u32;
    let hm = (base * 0.03) as u32;
    let hr = (base * rng.gen_range(0.008..0.02)).clamp(3.0, (base * 0.05).max(4.0)) as u32;
    let hedge = rng.gen_range(0..4u8);
    let (hx, hy) = match hedge {
        0 => (ru(rng, hm, hm + hr + 3, w), ru(rng, hm, h / 3, h)),
        1 => (ru(rng, w.saturating_sub(hm + hr + 3), w.saturating_sub(hm), w), ru(rng, hm, h / 3, h)),
        2 => (ru(rng, hm, w / 3, w), ru(rng, hm, hm + hr + 3, h)),
        _ => (ru(rng, hm, w / 3, w), ru(rng, h.saturating_sub(hm + hr + 3), h.saturating_sub(hm), h)),
    };
    let mut p = CaptureParams {
        enabled: true,
        motion_angle_deg: rng.gen_range(0.0..180.0),
        motion_radius: 1 + (rng.gen::<f32>() * 2.0 * s * px) as u32,
        fingerprint_edge: fedge, fingerprint_x: fx, fingerprint_y: fy,
        fingerprint_rx: frx, fingerprint_ry: (frx as f32 * 1.25).clamp(3.0, base) as u32,
        fingerprint_oil: rng.gen_range(6.0..18.0) * s,
        tape_height: (base * rng.gen_range(0.02..0.05)).clamp(2.0, base) as u32,
        tape_opacity: rng.gen_range(0.15..0.30) * (0.4 + 0.6 * s),
        tape_edge: rng.gen_range(0..4u8),
        tape_start_frac: rng.gen_range(0.0..0.55),
        tape_len_frac: rng.gen_range(0.30..0.55),
        residue_count: 0,
        residue_rects: Vec::new(),
        hole_x: hx, hole_y: hy, hole_radius: hr,
        sticky_corner: rng.gen_range(0..4u8),
        sticky_size: (base * rng.gen_range(0.12..0.22)).clamp(4.0, base) as u32,
        sticky_tone: rng.gen_range(0..3u8),
        exposure_lift: rng.gen_range(-1.0..1.0) * 15.0 * s,
        exposure_compress: 1.0 - rng.gen_range(0.0..0.12) * s,
        banding_amplitude: rng.gen_range(1.5..4.0) * s,
        banding_period: rng.gen_range((3.0 * px).round().max(2.0) as u32..(7.0 * px).round().max(3.0) as u32),
    };
    // Exposure error: global lift + slight contrast compression about mid-grey.
    for v in buf.iter_mut() {
        *v = ((*v - 128.0) * p.exposure_compress + 128.0 + p.exposure_lift).clamp(0.0, 255.0);
    }
    // Faint sensor banding: low-amplitude horizontal stripes.
    for y in 0..h {
        let off = p.banding_amplitude * (2.0 * PI * y as f32 / p.banding_period as f32).sin();
        for x in 0..w {
            let i = (y * w + x) as usize * 3;
            for c in 0..3 { buf[i + c] = (buf[i + c] + off).clamp(0.0, 255.0); }
        }
    }
    // Motion blur: directional box blur along the seeded angle.
    // Tap offsets depend only on `k`, so hoist them out of the pixel loop
    // (same rounded values, same accumulation order — bit-exact).
    let snap = buf.to_vec();
    let th = p.motion_angle_deg * PI / 180.0;
    let (dx, dy) = (th.cos(), th.sin());
    let r = p.motion_radius as i32;
    let taps = (2 * r + 1) as f32;
    let offs: Vec<(i32, i32)> = (-r..=r)
        .map(|k| ((dx * k as f32).round() as i32, (dy * k as f32).round() as i32))
        .collect();
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut acc = [0.0f32; 3];
            for (ox, oy) in offs.iter() {
                let nx = (x + ox).clamp(0, w as i32 - 1);
                let ny = (y + oy).clamp(0, h as i32 - 1);
                let j = (ny as u32 * w + nx as u32) as usize * 3;
                for c in 0..3 { acc[c] += snap[j + c]; }
            }
            let i = (y as u32 * w + x as u32) as usize * 3;
            for c in 0..3 { buf[i + c] = acc[c] / taps; }
        }
    }
    // Fingerprint: fingertip ellipse of concentric arc ridges + oil brightening.
    let (erx, ery) = (p.fingerprint_rx as f32, p.fingerprint_ry as f32);
    for oy in -(ery as i32)..=ery as i32 {
        for ox in -(erx as i32)..=erx as i32 {
            let t = ((ox as f32 / erx).powi(2) + (oy as f32 / ery).powi(2)).sqrt();
            if t <= 1.0 {
                let ridge = 0.5 + 0.5 * (t * 16.0 * PI).sin();
                let add = p.fingerprint_oil * (0.55 + 0.45 * ridge);
                let (px, py) = (p.fingerprint_x as i32 + ox, p.fingerprint_y as i32 + oy);
                if px >= 0 && py >= 0 && px < w as i32 && py < h as i32 {
                    let i = (py as u32 * w + px as u32) as usize * 3;
                    for c in 0..3 { buf[i + c] = (buf[i + c] + add).clamp(0.0, 255.0); }
                }
            }
        }
    }
    // Tape strip: translucent beige band along one edge (partial span, like
    // a real piece of tape) + sheen line. Any edge, never full-width, so it
    // reads as tape rather than a tinted band across the page.
    const BEIGE: [f32; 3] = [225.0, 205.0, 165.0];
    {
        let t = p.tape_height.min(w.min(h));
        let along = if p.tape_edge % 2 == 0 { w } else { h };
        let a0 = ((p.tape_start_frac * along as f32) as u32).min(along);
        let a1 = ((p.tape_start_frac + p.tape_len_frac) * along as f32) as u32;
        let (a0, a1) = (a0.min(along), a1.min(along).max(a0 + 1));
        let mut paint = |px: u32, py: u32| {
            if px < w && py < h {
                let i = (py * w + px) as usize * 3;
                for c in 0..3 {
                    buf[i + c] = buf[i + c] * (1.0 - p.tape_opacity) + BEIGE[c] * p.tape_opacity;
                }
            }
        };
        match p.tape_edge % 4 {
            0 => { for y in 0..t { for x in a0..a1 { paint(x, y); } } }
            1 => { for y in 0..t { for x in a0..a1 { paint(w - 1 - y.min(w - 1), x); } } }
            2 => { for y in 0..t { for x in a0..a1 { paint(x, h - 1 - y.min(h - 1)); } } }
            _ => { for y in 0..t { for x in a0..a1 { paint(y, x); } } }
        }
        // Sheen line down the middle of the strip.
        let mid = t / 2;
        for a in a0..a1 {
            let (sx, sy) = match p.tape_edge % 4 {
                0 => (a, mid),
                1 => (w - 1 - mid.min(w - 1), a),
                2 => (a, h - 1 - mid.min(h - 1)),
                _ => (mid, a),
            };
            if sx < w && sy < h {
                let i = (sy * w + sx) as usize * 3;
                for c in 0..3 { buf[i + c] = (buf[i + c] + 25.0).clamp(0.0, 255.0); }
            }
        }
    }
    // Tape residue: aged adhesive ghosts where tape was removed — yellowed
    // translucent rectangles with a darker adhesive border and slight gloss.
    // Translucent by design: notation stays readable underneath.
    {
        let n_res = if rng.gen::<f32>() < s * 0.45 { rng.gen_range(1..=2) } else { 0 };
        const RESIN: [f32; 3] = [235.0, 220.0, 178.0];
        for _ in 0..n_res {
            let rw = (base * rng.gen_range(0.08..0.25)).clamp(4.0, w as f32) as u32;
            let rh = (base * rng.gen_range(0.03..0.08)).clamp(3.0, h as f32) as u32;
            let edge = rng.gen_range(0..4u8);
            let (rx, ry) = match edge {
                0 => (rng.gen_range(0..w.saturating_sub(rw).max(1)), rng.gen_range(0..(h / 6).max(2))),
                1 => (rng.gen_range(0..w.saturating_sub(rw).max(1)), h.saturating_sub(rh + rng.gen_range(0..(h / 6).max(1)))),
                2 => (rng.gen_range(0..(w / 6).max(2)), rng.gen_range(0..h.saturating_sub(rh).max(1))),
                _ => (w.saturating_sub(rw + rng.gen_range(0..(w / 6).max(1))), rng.gen_range(0..h.saturating_sub(rh).max(1))),
            };
            let alpha = rng.gen_range(0.10..0.22) * (0.4 + 0.6 * s);
            for y in ry..(ry + rh).min(h) {
                for x in rx..(rx + rw).min(w) {
                    let i = (y * w + x) as usize * 3;
                    let border = x == rx || y == ry || x + 1 >= (rx + rw).min(w) || y + 1 >= (ry + rh).min(h);
                    for c in 0..3 {
                        buf[i + c] = buf[i + c] * (1.0 - alpha) + RESIN[c] * alpha;
                        if border {
                            buf[i + c] *= 0.85;
                        } else {
                            buf[i + c] = (buf[i + c] + 8.0 * alpha).clamp(0.0, 255.0);
                        }
                    }
                }
            }
            p.residue_count += 1;
            p.residue_rects.push([rx, ry, rw, rh]);
        }
    }
    // Staple/punch hole: dark circle with a light rim, margin-safe.
    let hrr = p.hole_radius as f32;
    for oy in -(p.hole_radius as i32 + 2)..=p.hole_radius as i32 + 2 {
        for ox in -(p.hole_radius as i32 + 2)..=p.hole_radius as i32 + 2 {
            let d = ((ox * ox + oy * oy) as f32).sqrt();
            let (px, py) = (p.hole_x as i32 + ox, p.hole_y as i32 + oy);
            if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
                continue;
            }
            let i = (py as u32 * w + px as u32) as usize * 3;
            if d <= hrr {
                for c in 0..3 { buf[i + c] = 25.0 + buf[i + c] * 0.1; }
            } else if d <= hrr + 2.0 {
                for c in 0..3 { buf[i + c] = (buf[i + c] + 40.0).clamp(0.0, 255.0); }
            }
        }
    }
    // Sticky note: opaque pastel square overlapping a corner margin.
    const PASTEL: [[f32; 3]; 3] = [
        [255.0, 235.0, 130.0],
        [255.0, 180.0, 200.0],
        [190.0, 240.0, 180.0],
    ];
    let col = PASTEL[p.sticky_tone as usize % 3];
    let sz = p.sticky_size.min(w).min(h);
    let (ox, oy) = match p.sticky_corner % 4 {
        0 => (0, 0),
        1 => (w - sz, 0),
        2 => (0, h - sz),
        _ => (w - sz, h - sz),
    };
    for y in oy..oy + sz {
        for x in ox..ox + sz {
            let i = (y * w + x) as usize * 3;
            buf[i] = col[0];
            buf[i + 1] = col[1];
            buf[i + 2] = col[2];
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn synth(w: u32, h: u32) -> Vec<f32> {
        let mut v = vec![0.0; w as usize * h as usize * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize * 3;
                v[i] = x as f32 * 255.0 / w as f32;
                v[i + 1] = y as f32 * 255.0 / h as f32;
                v[i + 2] = 128.0;
            }
        }
        v
    }

    #[test]
    fn same_seed_same_bytes() {
        let (w, h) = (48, 32);
        let (mut a, mut b) = (synth(w, h), synth(w, h));
        let (mut ra, mut rb) = (ChaCha8Rng::seed_from_u64(7), ChaCha8Rng::seed_from_u64(7));
        let pa = apply(&mut a, w, h, &mut ra, 0.7, 1.0);
        let pb = apply(&mut b, w, h, &mut rb, 0.7, 1.0);
        assert_eq!(a, b);
        assert_eq!(serde_json::to_string(&pa).unwrap(), serde_json::to_string(&pb).unwrap());
    }

    #[test]
    fn output_differs_and_params_in_range() {
        let (w, h) = (48, 32);
        let mut buf = synth(w, h);
        let before = buf.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let p = apply(&mut buf, w, h, &mut rng, 0.8, 1.0);
        assert!(p.enabled && buf != before);
        assert!((0.0..180.0).contains(&p.motion_angle_deg) && (1..=3).contains(&p.motion_radius));
        assert!(p.fingerprint_x < w && p.fingerprint_y < h && (0.0..=18.0).contains(&p.fingerprint_oil));
        assert!(p.tape_height >= 2 && p.tape_height <= w.min(h) && (0.0..=1.0).contains(&p.tape_opacity));
        assert!(p.tape_edge < 4 && (0.0..0.55).contains(&p.tape_start_frac) && (0.30..=0.55).contains(&p.tape_len_frac));
        assert!(p.residue_count <= 2 && p.residue_rects.len() as u32 == p.residue_count);
        for r in &p.residue_rects {
            assert!(r[0] + r[2] <= w && r[1] + r[3] <= h);
        }
        assert!(p.hole_x < w && p.hole_y < h && p.hole_radius >= 3);
        assert!(p.sticky_size >= 4 && p.sticky_size <= w.min(h) && p.sticky_tone < 3);
        assert!((-15.0..=15.0).contains(&p.exposure_lift) && (0.88..=1.0).contains(&p.exposure_compress));
        assert!((0.0..=4.0).contains(&p.banding_amplitude) && (3..7).contains(&p.banding_period));
        assert!(buf.iter().all(|&v| (0.0..=255.0).contains(&v)));
    }
}
