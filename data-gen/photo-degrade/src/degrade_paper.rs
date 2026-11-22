//! Paper aging & texture for OMR training data: yellowing/mottling, aged
//! edge band, clustered foxing spots, fibers + laid lines, cockling waves,
//! and optional water damage with a tide-line rim. All randomness comes from
//! the passed `rng` (deterministic per seed). Buffer: interleaved RGB `f32`.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// Every sampled paper-aging knob, recorded for `params.json`.
#[derive(serde::Serialize, Debug, Clone)]
pub struct PaperParams {
    pub enabled: bool,
    pub yellowing: f32,
    pub mottle: f32,
    pub edge: f32,
    pub foxing_spots: u32,
    pub foxing_amount: f32,
    pub fiber_strokes: u32,
    pub fiber_amount: f32,
    pub cockle: f32,
    pub water_present: bool,
    pub water_x: f32,
    pub water_y: f32,
    pub water_radius: f32,
    pub water_amount: f32,
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Age the paper in `buf` (interleaved RGB `f32`, row-major, `w*h*3`).
/// `strength` is 0..=1 and scales every effect.
/// Full-strength entry point (kept for the unit tests + API stability).
#[allow(dead_code)]
pub fn apply(buf: &mut [f32], w: u32, h: u32, rng: &mut ChaCha8Rng, strength: f32) -> PaperParams {
    apply_with(buf, w, h, rng, strength, false, 1.0)
}

/// Full aging pass when `coarse` is false; when true, only foxing spots +
/// fibers are applied and the full-image tint/mottle/edge/cockling/water
/// loop is skipped. Sampling header identical either way (stream-preserving);
/// the skipped loop consumes no RNG draws.
///
/// `px` is the pixel-scale factor (1.0 = native): absolute-pixel radii and
/// the laid-line period scale with it; `px = 1.0` is bit-identical to legacy.
pub fn apply_with(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    strength: f32,
    coarse: bool,
    px: f32,
) -> PaperParams {
    let strength = strength.clamp(0.0, 1.0);
    // Fixed draw order: every knob is sampled on every call.
    let yellowing = rng.gen::<f32>();
    let mottle = rng.gen::<f32>();
    let edge = rng.gen::<f32>();
    let foxing_u = rng.gen::<f32>();
    let foxing_amount = rng.gen::<f32>();
    let fiber_u = rng.gen::<f32>();
    let fiber_amount = rng.gen::<f32>();
    let cockle = rng.gen::<f32>();
    let water_u = rng.gen::<f32>();
    let water_x = rng.gen::<f32>();
    let water_y = rng.gen::<f32>();
    let water_radius = 0.15 + rng.gen::<f32>() * 0.30;
    let water_amount = rng.gen::<f32>();
    // NOTE(subtask-4): renamed from `px` — the training-resolution scale
    // factor param above now owns that name; area keeps its usize math.
    let area_px = w as usize * h as usize;
    let max_spots = ((area_px / 9000).clamp(8, 900)) as u32;
    let max_fibers = ((area_px / 20000).clamp(10, 600)) as u32;
    let mut p = PaperParams {
        enabled: strength > 0.0,
        yellowing, mottle, edge,
        foxing_spots: (foxing_u * max_spots as f32).round() as u32,
        foxing_amount,
        fiber_strokes: (fiber_u * max_fibers as f32).round() as u32,
        fiber_amount, cockle,
        water_present: water_u < 0.35,
        water_x, water_y, water_radius, water_amount,
    };
    if !p.enabled || buf.is_empty() || w == 0 || h == 0 {
        if !p.enabled { p.water_present = false; }
        return p;
    }
    let wf = w as f32;
    let hf = h as f32;
    let min_dim = wf.min(hf);
    // Coarse mode skips the full-image tint loop below (no RNG draws inside,
    // so the stream is preserved); foxing + fibers still run.
    if !coarse {
    // Deterministic phases derived from sampled knobs (no hidden draws).
    let o1 = p.yellowing * 6.28318;
    let o2 = p.mottle * 6.28318;
    let o3 = p.edge * 6.28318;
    let o4 = p.cockle * 6.28318;
    let (yel, mot, edg, cok) = (p.yellowing * strength, p.mottle * strength,
        p.edge * strength, p.cockle * strength);
    let laid_amp = (0.5 + 1.1 * p.fiber_amount) * strength;
    // Laid lines are a ~5 px period at native res; keep the physical period
    // fixed by scaling the angular frequency (÷1.0 = legacy exactly).
    let laid_w = 1.2566 / px;
    let (wcx, wcy, wr) = (p.water_x * wf, p.water_y * hf, p.water_radius * min_dim);
    for y in 0..h {
        let fy = y as f32 / hf.max(1.0);
        let sy1 = (fy * 18.85 + o1).sin(); // ~3 vertical blotch cycles
        let sy2 = (fy * 50.27 + o2).sin(); // ~8 vertical blotch cycles
        let syc = (fy * 12.57 + o4).sin(); // cockling ~2 cycles
        let laid = (y as f32 * laid_w + o3).sin() * laid_amp; // ~5px laid lines
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            if i + 2 >= buf.len() { continue; }
            let fx = x as f32 / wf.max(1.0);
            // Layered sine blotch field, -1..1.
            let m = (fx * 18.85 + o2).sin() * sy1 * 0.6 + (fx * 43.98 + o1).sin() * sy2 * 0.4;
            // Aged border band: 0 inside, 1 at the sheet rim.
            let d = fx.min(1.0 - fx).min(fy).min(1.0 - fy);
            let band = ((0.06 - d) / 0.06).clamp(0.0, 1.0);
            let band2 = band * band;
            // Sepia/yellow shift, stronger toward the edges.
            let mut r = buf[i] + yel * (12.0 + 10.0 * m) * (1.0 + band);
            let mut g = buf[i + 1] + yel * (7.0 + 6.0 * m) * (1.0 + 0.8 * band);
            let mut b = buf[i + 2] - yel * (10.0 + 8.0 * m) * (1.0 + 0.8 * band);
            let v = mot * 6.0 * m; // neutral mottling
            r += v; g += v; b += v;
            r -= edg * 30.0 * band2; g -= edg * 34.0 * band2; b -= edg * 44.0 * band2;
            let c = 1.0 + (fx * 12.57 + o3).sin() * syc * cok * 0.05; // cockling
            r *= c; g *= c; b *= c;
            r += laid; g += laid; b += laid;
            if p.water_present {
                let dx = x as f32 - wcx;
                let dy = y as f32 - wcy;
                let t = (dx * dx + dy * dy).sqrt() / wr.max(1.0);
                if t < 1.25 {
                    let inside = 1.0 - smoothstep(0.65, 1.1, t);
                    let rim = (-((t - 1.0) / 0.05).powi(2)).exp(); // tide line
                    let wa = p.water_amount * strength;
                    r -= wa * (16.0 * inside + 46.0 * rim);
                    g -= wa * (16.0 * inside + 52.0 * rim);
                    b -= wa * (12.0 * inside + 56.0 * rim);
                }
            }
            buf[i] = r.clamp(0.0, 255.0);
            buf[i + 1] = g.clamp(0.0, 255.0);
            buf[i + 2] = b.clamp(0.0, 255.0);
        }
    }
    } // --coarse: skip tint/mottle/edge/cockling/water loop
    if coarse {
        // Params record what was actually applied.
        p.yellowing = 0.0;
        p.mottle = 0.0;
        p.edge = 0.0;
        p.cockle = 0.0;
        p.water_present = false;
        p.water_amount = 0.0;
    }
    // Foxing: brown spots clustered around a few deterministic centers.
    let n_spots = (p.foxing_spots as f32 * strength).round() as usize;
    if n_spots > 0 {
        let nc = ((n_spots / 10) + 2).min(6).max(2);
        let mut cc = [(0.0f32, 0.0f32); 6];
        for k in 0..nc { cc[k] = (rng.gen::<f32>() * wf, rng.gen::<f32>() * hf); }
        let spread = min_dim * 0.06;
        for _ in 0..n_spots {
            let k = (rng.gen::<f32>() * nc as f32) as usize % nc;
            let mut bell = || rng.gen::<f32>() + rng.gen::<f32>() + rng.gen::<f32>() - 1.5;
            let (px0, py0) = (cc[k].0 + bell() * spread, cc[k].1 + bell() * spread);
            let rad = (0.6 + rng.gen::<f32>() * 1.8) * px;
            let a = p.foxing_amount * strength * (0.25 + 0.75 * rng.gen::<f32>());
            let (x0, x1) = ((px0 - rad).floor().max(0.0) as u32, (px0 + rad).ceil().min(wf - 1.0).max(0.0) as u32);
            let (y0, y1) = ((py0 - rad).floor().max(0.0) as u32, (py0 + rad).ceil().min(hf - 1.0).max(0.0) as u32);
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    let i = ((yy * w + xx) * 3) as usize;
                    if i + 2 >= buf.len() { continue; }
                    let dd = ((xx as f32 - px0).powi(2) + (yy as f32 - py0).powi(2)).sqrt();
                    if dd > rad { continue; }
                    let f = (1.0 - dd / rad).powi(2) * a;
                    buf[i] = (buf[i] - 36.0 * f).clamp(0.0, 255.0);
                    buf[i + 1] = (buf[i + 1] - 66.0 * f).clamp(0.0, 255.0);
                    buf[i + 2] = (buf[i + 2] - 88.0 * f).clamp(0.0, 255.0);
                }
            }
        }
    }
    // Paper fibers: sparse 1px light/dark short horizontal strokes.
    let n_fib = (p.fiber_strokes as f32 * strength).round() as usize;
    for _ in 0..n_fib {
        let (x0, y0) = (rng.gen_range(0..w), rng.gen_range(0..h));
        // Physical fiber length scales with `px` (floor 2 keeps the stroke;
        // truncation, not rounding, so px = 1.0 matches the legacy lengths).
        let len = (((3.0 + rng.gen::<f32>() * 8.0) * px).max(2.0)) as u32;
        let dark = rng.gen_bool(0.5);
        let amp = p.fiber_amount * strength * (5.0 + rng.gen::<f32>() * 9.0);
        for k in 0..len {
            let i = ((y0 * w + (x0 + k).min(w - 1)) * 3) as usize;
            if i + 2 >= buf.len() { continue; }
            for c in 0..3 {
                buf[i + c] = if dark { (buf[i + c] - amp).clamp(0.0, 255.0) }
                    else { (buf[i + c] + amp).clamp(0.0, 255.0) };
            }
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn synthetic(w: u32, h: u32) -> Vec<f32> {
        let mut v = vec![0.0; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let ink = y % 8 == 0 && x % 3 != 0; // dark staff-like bars on paper
                v[i] = if ink { 20.0 } else { 245.0 };
                v[i + 1] = if ink { 20.0 } else { 242.0 };
                v[i + 2] = if ink { 20.0 } else { 235.0 };
            }
        }
        v
    }

    #[test]
    fn determinism_same_seed_identical_bytes() {
        let (w, h) = (32, 24);
        let mut a = synthetic(w, h);
        let mut b = a.clone();
        let p1 = apply(&mut a, w, h, &mut ChaCha8Rng::seed_from_u64(42), 0.7);
        let p2 = apply(&mut b, w, h, &mut ChaCha8Rng::seed_from_u64(42), 0.7);
        assert_eq!(a, b);
        assert_eq!(format!("{p1:?}"), format!("{p2:?}"));
    }

    #[test]
    fn effect_differs_and_params_in_range() {
        let (w, h) = (32, 24);
        let mut buf = synthetic(w, h);
        let before = buf.clone();
        let p = apply(&mut buf, w, h, &mut ChaCha8Rng::seed_from_u64(7), 0.7);
        assert!(p.enabled && buf != before);
        for v in [p.yellowing, p.mottle, p.edge, p.foxing_amount, p.fiber_amount, p.cockle] {
            assert!((0.0..=1.0).contains(&v), "knob out of range: {v}");
        }
        assert!((0.0..=1.0).contains(&p.water_x) && (0.0..=1.0).contains(&p.water_y));
        assert!((0.15..=0.45).contains(&p.water_radius));
        assert!(buf.iter().all(|v| (0.0..=255.0).contains(v)));
        // Zero strength is a no-op.
        let mut flat = synthetic(w, h);
        let p0 = apply(&mut flat, w, h, &mut ChaCha8Rng::seed_from_u64(7), 0.0);
        assert!(!p0.enabled && flat == before);
    }
}
