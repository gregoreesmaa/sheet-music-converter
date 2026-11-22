//! Physical wear for OMR training data: heavy ink loss, printer artifacts,
//! eraser abrasion, and organic stains. Buffer: RGB interleaved `f32`
//! 0..=255, row-major; all randomness from `rng`.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

const PAPER: [f32; 3] = [244.0, 241.0, 235.0];
const PAPER_LUM: f32 = 241.19;

#[derive(serde::Serialize, Debug, Clone)]
pub struct WearParams {
    pub enabled: bool,
    pub heavy_zones: [[f32; 4]; 3],
    pub heavy_lift: f32,
    pub flake_clusters: u32,
    pub band_period: f32,
    pub band_amp: f32,
    pub streaks: u32,
    pub toner_bands: u32,
    pub misreg: [i32; 2],
    pub eraser_count: u32,
    pub stain_water: bool,
    pub stain_oil: bool,
}

impl WearParams {
    fn disabled() -> Self {
        Self {
            enabled: false,
            heavy_zones: [[0.0; 4]; 3],
            heavy_lift: 0.0,
            flake_clusters: 0,
            band_period: 0.0,
            band_amp: 0.0,
            streaks: 0,
            toner_bands: 0,
            misreg: [0, 0],
            eraser_count: 0,
            stain_water: false,
            stain_oil: false,
        }
    }
}

/// Low-toner column restricted to the ink band bbox `[x_lo, x_hi)`.
/// (algo-aging-wear Swap 2.) Outside `|x - cx| >= hw` the multiplier is
/// exactly 1.0, so those columns are skipped. One documented deviation
/// from the legacy full-width loop: there the `m == 1.0` pixels still
/// executed the write `PAPER - (PAPER-buf)*1.0`, which is not the identity
/// in fp (up to ~7.7e-6; measured). Skipping it leaves those dark pixels
/// at their pre-toner value instead — invisible (Otsu bins are ~1 gray
/// level wide) and deterministic; params and RNG draws are untouched.
fn apply_toner_band(
    buf: &mut [f32],
    snap: &[f32],
    wsz: usize,
    hsz: usize,
    cx: f32,
    hw: f32,
    s: f32,
) {
    let x_lo = ((cx - hw).floor().max(0.0)) as usize;
    let x_hi = ((cx + hw).ceil().clamp(0.0, wsz as f32)) as usize;
    let (x_lo, x_hi) = (x_lo.min(wsz), x_hi.min(wsz));
    for y in 0..hsz {
        for x in x_lo..x_hi {
            let t = ((x as f32 - cx) / hw).abs().min(1.0);
            let m = 1.0 - s * 0.5 * (1.0 - t * t);
            let j = (y * wsz + x) * 3;
            let d = ((PAPER_LUM - snap[j / 3]) / PAPER_LUM).clamp(0.0, 1.0);
            if d > 0.02 {
                for c in 0..3 {
                    buf[j + c] = PAPER[c] - (PAPER[c] - buf[j + c]) * m;
                }
            }
        }
    }
}

/// Misregistration echo with fully-paper 8x8 tiles skipped (Swap 2
/// companion). A skipped tile provably has `g <= 0.05` at every pixel, so
/// the legacy `if g > 0.05` write never fires there: bit-exact. No RNG.
fn apply_misreg(buf: &mut [f32], snap: &[f32], wsz: usize, hsz: usize, misreg: [i32; 2]) {
    const TILE: usize = 8;
    for ty in (0..hsz).step_by(TILE) {
        for tx in (0..wsz).step_by(TILE) {
            // Tile-local max of the echo source decides the skip.
            let mut peak = 0.0f32;
            for y in ty..(ty + TILE).min(hsz) {
                for x in tx..(tx + TILE).min(wsz) {
                    let sx = (x as i32 - misreg[0]).clamp(0, wsz as i32 - 1) as usize;
                    let sy = (y as i32 - misreg[1]).clamp(0, hsz as i32 - 1) as usize;
                    let g = ((PAPER_LUM - snap[sy * wsz + sx]) / PAPER_LUM).clamp(0.0, 1.0);
                    if g > peak {
                        peak = g;
                    }
                }
            }
            if peak <= 0.05 {
                continue;
            }
            for y in ty..(ty + TILE).min(hsz) {
                for x in tx..(tx + TILE).min(wsz) {
                    let sx = (x as i32 - misreg[0]).clamp(0, wsz as i32 - 1) as usize;
                    let sy = (y as i32 - misreg[1]).clamp(0, hsz as i32 - 1) as usize;
                    let g = ((PAPER_LUM - snap[sy * wsz + sx]) / PAPER_LUM).clamp(0.0, 1.0);
                    if g > 0.05 {
                        let j = (y * wsz + x) * 3;
                        for c in 0..3 {
                            buf[j + c] = (buf[j + c] - g * 18.0).clamp(0.0, 255.0);
                        }
                    }
                }
            }
        }
    }
}

/// One eraser blob with bbox-only snapshots (algo-aging-wear Swap 4).
/// The swirl reads rotated taps within distance `r` of the center and the
/// box passes read ±1 taps of `d < r` pixels — all inside the `(r+2)`
/// guard bbox — so a cropped snapshot + bbox-clamped sampler returns
/// exactly what the legacy full-buffer clone returned: bit-exact, at
/// ~1/2000th of the memcpy. `rng` draws (per-pixel roughen) run in the
/// same order as the direct loop.
fn apply_eraser(
    buf: &mut [f32],
    w: u32,
    h: u32,
    rng: &mut ChaCha8Rng,
    cx: f32,
    cy: f32,
    r: f32,
    swirl: f32,
) {
    let (wsz, hsz) = (w as usize, h as usize);
    let x0 = (cx - r - 2.0).max(0.0) as usize;
    let x1 = (cx + r + 2.0).min(w as f32 - 1.0) as usize;
    let y0 = (cy - r - 2.0).max(0.0) as usize;
    let y1 = (cy + r + 2.0).min(h as f32 - 1.0) as usize;
    if x1 < x0 || y1 < y0 {
        return;
    }
    let bw = x1 - x0 + 1;
    // Bbox snapshot of the pre-swirl buffer.
    let mut src = vec![0.0f32; bw * (y1 - y0 + 1) * 3];
    for y in y0..=y1 {
        for x in x0..=x1 {
            let s = ((y * wsz + x) * 3) as usize;
            let d = (((y - y0) * bw + (x - x0)) * 3) as usize;
            src[d..d + 3].copy_from_slice(&buf[s..s + 3]);
        }
    }
    // Bilinear sample over the snapshot: image-OOB taps yield paper BG
    // (same as `sample_bilinear`); every in-image tap the swirl can reach
    // lies inside the bbox by the radius argument above.
    let tap = |ix: i64, iy: i64| -> [f32; 3] {
        if ix < 0 || iy < 0 || ix >= w as i64 || iy >= h as i64 {
            return [246.0, 243.0, 236.0];
        }
        let i = (((iy as usize - y0) * bw + (ix as usize - x0)) * 3) as usize;
        [src[i], src[i + 1], src[i + 2]]
    };
    let sample = |x: f32, y: f32| -> [f32; 3] {
        let x0f = x.floor() as i64;
        let y0f = y.floor() as i64;
        let fx = x - x0f as f32;
        let fy = y - y0f as f32;
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            let top = tap(x0f, y0f)[c] * (1.0 - fx) + tap(x0f + 1, y0f)[c] * fx;
            let bot = tap(x0f, y0f + 1)[c] * (1.0 - fx) + tap(x0f + 1, y0f + 1)[c] * fx;
            out[c] = top * (1.0 - fy) + bot * fy;
        }
        out
    };
    for y in y0..=y1 {
        for x in x0..=x1 {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            if d < r {
                // Twist content around the blob center, fading at the rim.
                let a = swirl * (1.0 - d / r);
                let (sa, ca) = a.sin_cos();
                let sx = cx + dx * ca - dy * sa;
                let sy = cy + dx * sa + dy * ca;
                let p = sample(sx, sy);
                let j = (y * wsz + x) * 3;
                // Abrasion: lift toward paper + roughen.
                let lift = 0.30 * (1.0 - d / r);
                for c in 0..3 {
                    buf[j + c] = (p[c] * (1.0 - lift) + PAPER[c] * lift
                        + (rng.gen::<f32>() - 0.5) * 14.0)
                        .clamp(0.0, 255.0);
                }
            }
        }
    }
    // Two 3x3 box passes inside the blob: the erased smudge.
    for _ in 0..2 {
        let mut snap_b = vec![0.0f32; bw * (y1 - y0 + 1) * 3];
        for y in y0..=y1 {
            for x in x0..=x1 {
                let s = ((y * wsz + x) * 3) as usize;
                let d = (((y - y0) * bw + (x - x0)) * 3) as usize;
                snap_b[d..d + 3].copy_from_slice(&buf[s..s + 3]);
            }
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if (dx * dx + dy * dy).sqrt() < r {
                    for c in 0..3 {
                        let mut acc = 0.0f32;
                        for oy in -1i32..=1 {
                            for ox in -1i32..=1 {
                                // Image-clamped taps (same as the direct
                                // loop); clamped coords stay inside the
                                // bbox because the guard is 2 px wide.
                                let xx = (x as i32 + ox).clamp(0, wsz as i32 - 1) as usize;
                                let yy = (y as i32 + oy).clamp(0, hsz as i32 - 1) as usize;
                                acc += snap_b[((yy - y0) * bw + (xx - x0)) * 3 + c];
                            }
                        }
                        buf[(y * wsz + x) * 3 + c] = acc / 9.0;
                    }
                }
            }
        }
    }
}

/// `px` is the pixel-scale factor (1.0 = native): the absolute-pixel
/// banding period and flake radius scale with it; `px = 1.0` is
/// bit-identical to the legacy sampling.
pub fn apply(buf: &mut [f32], w: u32, h: u32, rng: &mut ChaCha8Rng, strength: f32, px: f32) -> WearParams {
    let (wsz, hsz) = (w as usize, h as usize);
    let s = strength.clamp(0.0, 1.0);
    if s <= 0.0 || wsz == 0 || hsz == 0 || buf.len() < wsz * hsz * 3 {
        return WearParams::disabled();
    }
    let area = (wsz * hsz) as f32;

    let mut heavy_zones = [[0.0f32; 4]; 3];
    for z in heavy_zones.iter_mut() {
        *z = [rng.gen(), rng.gen(), rng.gen_range(0.08..0.25), rng.gen_range(0.5..1.0)];
    }
    let heavy_lift = s * rng.gen_range(0.6..0.95);
    let flake_clusters = ((area * s * rng.gen_range(0.00003..0.00012)) as u32).min(1500);
    let band_period = rng.gen_range(8.0..28.0) * px;
    let band_amp = s * rng.gen_range(0.02..0.08);
    let streaks = if rng.gen::<f32>() < s * 0.7 { rng.gen_range(1..=4) } else { 0 };
    let toner_bands = if rng.gen::<f32>() < s * 0.5 { rng.gen_range(1..=2) } else { 0 };
    let misreg = if rng.gen::<f32>() < s * 0.4 {
        *[[1, 0], [0, 1], [1, 1]].get(rng.gen_range(0..3)).unwrap()
    } else {
        [0, 0]
    };
    let eraser_count = if rng.gen::<f32>() < s * 0.65 { rng.gen_range(1..=2) } else { 0 };
    let stain_water = rng.gen::<f32>() < s * 0.5;
    let stain_oil = rng.gen::<f32>() < s * 0.35;

    let params = WearParams {
        enabled: true,
        heavy_zones,
        heavy_lift,
        flake_clusters,
        band_period,
        band_amp,
        streaks,
        toner_bands,
        misreg,
        eraser_count,
        stain_water,
        stain_oil,
    };

    let snap: Vec<f32> = buf
        .chunks_exact(3)
        .map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2])
        .collect();
    let (wf, hf) = (w as f32, h as f32);

    // Heavy ink loss: near-total washout inside soft zones + flaked patches.
    for (i, px) in buf.chunks_exact_mut(3).enumerate() {
        let d = ((PAPER_LUM - snap[i]) / PAPER_LUM).clamp(0.0, 1.0);
        if d > 0.02 {
            let nx = (i % wsz) as f32 / wf;
            let ny = (i / wsz) as f32 / hf;
            let mut field = 0.0f32;
            for z in &heavy_zones {
                let dx = (nx - z[0]) / z[2];
                let dy = (ny - z[1]) / z[2];
                field += z[3] * (-0.5 * (dx * dx + dy * dy)).exp();
            }
            let lift = (heavy_lift * field.min(1.2)).min(0.97);
            let k = (1.0 - lift).clamp(0.0, 1.0);
            for c in 0..3 {
                px[c] = PAPER[c] - (PAPER[c] - px[c]) * k;
            }
        }
    }
    for _ in 0..flake_clusters {
        let cx = rng.gen_range(0..wsz) as i32;
        let cy = rng.gen_range(0..hsz) as i32;
        let rr = rng.gen_range((2.0 * px).round().max(1.0) as i32..=(5.0 * px).round().max(1.0) as i32);
        for oy in -rr..=rr {
            for ox in -rr..=rr {
                if ox * ox + oy * oy > rr * rr {
                    continue;
                }
                let (xx, yy) = (cx + ox, cy + oy);
                if xx < 0 || yy < 0 || xx >= wsz as i32 || yy >= hsz as i32 {
                    continue;
                }
                let j = (yy as usize * wsz + xx as usize) * 3;
                buf[j] = PAPER[0];
                buf[j + 1] = PAPER[1];
                buf[j + 2] = PAPER[2];
            }
        }
    }

    // Printer banding: faint horizontal density waves from head passes.
    let phase = rng.gen_range(0.0..std::f32::consts::TAU);
    for y in 0..hsz {
        let m = 1.0 - band_amp * (0.5 + 0.5 * (std::f32::consts::TAU * y as f32 / band_period + phase).sin());
        for x in 0..wsz {
            let j = (y * wsz + x) * 3;
            buf[j] *= m;
            buf[j + 1] *= m;
            buf[j + 2] *= m;
        }
    }
    // Clogged-nozzle streaks: pale vertical lines through everything.
    for _ in 0..streaks {
        let x0 = rng.gen_range(0..wsz);
        let a = s * rng.gen_range(0.15..0.4);
        for y in 0..hsz {
            let j = (y * wsz + x0) * 3;
            for c in 0..3 {
                buf[j + c] += (255.0 - buf[j + c]) * a;
            }
        }
    }
    // Low-toner columns: soft vertical bands of extra-faded ink.
    // Bbox-clipped (algo-aging-wear Swap 2): outside `|x - cx| >= hw` the
    // multiplier is exactly 1.0, so the x loop only covers the band.
    for _ in 0..toner_bands {
        let cx = rng.gen::<f32>() * wf;
        let hw = wf * rng.gen_range(0.03..0.10);
        apply_toner_band(buf, &snap, wsz, hsz, cx, hw, s);
    }
    // Misregistration echo: faint offset copy of dark content.
    if misreg != [0, 0] {
        apply_misreg(buf, &snap, wsz, hsz, misreg);
    }

    // Eraser abrasion: swirled smear + heavy local blur + roughened paper.
    // Erasers strike through notation, so placement is anywhere.
    for _ in 0..eraser_count {
        let cx = rng.gen::<f32>() * wf;
        let cy = rng.gen::<f32>() * hf;
        let r = w.min(h) as f32 * rng.gen_range(0.015..0.045);
        let swirl = rng.gen_range(0.6..1.4) * if rng.gen::<bool>() { 1.0 } else { -1.0 };
        apply_eraser(buf, w, h, rng, cx, cy, r, swirl);
    }

    // Organic stains: irregular water blotch with tide line, soft oil spot.
    let stain = |buf: &mut [f32], rng: &mut ChaCha8Rng, oily: bool| {
        let cx = rng.gen::<f32>() * wf;
        let cy = rng.gen::<f32>() * hf;
        let r = w.min(h) as f32 * rng.gen_range(0.03..0.09);
        let (a1, p1) = (rng.gen_range(0.10..0.30), rng.gen_range(0.0..std::f32::consts::TAU));
        let (a2, p2) = (rng.gen_range(0.05..0.20), rng.gen_range(0.0..std::f32::consts::TAU));
        let bb = (r * 1.4) as i64 + 2;
        for dy in -bb..=bb {
            for dx in -bb..=bb {
                let (fx, fy) = (dx as f32, dy as f32);
                let d = (fx * fx + fy * fy).sqrt();
                if d < 0.5 {
                    continue;
                }
                let th = fy.atan2(fx);
                let edge = r * (1.0 + a1 * (3.0 * th + p1).sin() + a2 * (7.0 * th + p2).sin());
                if d < edge {
                    let (x, y) = (cx + fx, cy + fy);
                    if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
                        continue;
                    }
                    let i = ((y as usize) * wsz + (x as usize)) * 3;
                    if oily {
                        // Grease: translucent warm darkening, soft rim.
                        let m = (1.0 - d / edge).min(1.0);
                        for c in 0..3 {
                            buf[i + c] = (buf[i + c] * (1.0 - 0.18 * m) + [196.0, 176.0, 128.0][c] * 0.18 * m)
                                .clamp(0.0, 255.0);
                        }
                    } else {
                        // Water: faint interior + darker tide line at the rim.
                        let rim_w = (edge * 0.10).max(2.0);
                        let rim = (-((d - edge) / rim_w).powi(2)).exp();
                        for c in 0..3 {
                            buf[i + c] = (buf[i + c] * 0.94 + [208.0, 192.0, 168.0][c] * 0.06).clamp(0.0, 255.0);
                        }
                        for c in 0..3 {
                            buf[i + c] = (buf[i + c] * (1.0 - 0.45 * rim) + [110.0, 78.0, 44.0][c] * 0.45 * rim)
                                .clamp(0.0, 255.0);
                        }
                    }
                }
            }
        }
    };
    if stain_water {
        stain(buf, rng, false);
    }
    if stain_oil {
        stain(buf, rng, true);
    }

    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn synth(w: usize, h: usize) -> Vec<f32> {
        let mut buf = vec![244.0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                if y % 6 == 0 {
                    let j = (y * w + x) * 3;
                    buf[j] = 30.0;
                    buf[j + 1] = 30.0;
                    buf[j + 2] = 30.0;
                }
            }
        }
        buf
    }

    #[test]
    fn determinism_same_seed_identical() {
        let (w, h) = (48u32, 32u32);
        let mut a = synth(48, 32);
        let mut b = a.clone();
        let pa = apply(&mut a, w, h, &mut ChaCha8Rng::seed_from_u64(11), 0.7, 1.0);
        let pb = apply(&mut b, w, h, &mut ChaCha8Rng::seed_from_u64(11), 0.7, 1.0);
        assert_eq!(a, b);
        assert_eq!(
            serde_json::to_string(&pa).unwrap(),
            serde_json::to_string(&pb).unwrap()
        );
    }

    fn snap_of(buf: &[f32]) -> Vec<f32> {
        buf.chunks_exact(3)
            .map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2])
            .collect()
    }

    /// Legacy full-width toner loop (pre-Swap-2 reference).
    fn toner_direct(
        buf: &mut [f32],
        snap: &[f32],
        wsz: usize,
        hsz: usize,
        cx: f32,
        hw: f32,
        s: f32,
    ) {
        for y in 0..hsz {
            for x in 0..wsz {
                let t = ((x as f32 - cx) / hw).abs().min(1.0);
                let m = 1.0 - s * 0.5 * (1.0 - t * t);
                let j = (y * wsz + x) * 3;
                let d = ((PAPER_LUM - snap[j / 3]) / PAPER_LUM).clamp(0.0, 1.0);
                if d > 0.02 {
                    for c in 0..3 {
                        buf[j + c] = PAPER[c] - (PAPER[c] - buf[j + c]) * m;
                    }
                }
            }
        }
    }

    #[test]
    fn toner_band_clip_covers_band_and_stays_bounded() {
        // Containment: every x with |x-cx| < hw lies inside the clip range.
        for (cx, hw, w) in [(5.0f32, 2.0f32, 32usize), (31.7, 9.6, 64), (0.5, 1.0, 16), (63.9, 30.0, 64)] {
            let x_lo = ((cx - hw).floor().max(0.0)) as usize;
            let x_hi = ((cx + hw).ceil().clamp(0.0, w as f32)) as usize;
            for x in 0..w {
                if ((x as f32 - cx) / hw).abs() < 1.0 {
                    assert!(x >= x_lo && x < x_hi, "cx={cx} hw={hw}: x={x} escapes [{x_lo},{x_hi})");
                }
            }
        }
        // Bounded deviation vs the legacy full-width loop (the documented
        // m==1.0 no-op-write fp rounding, <= ~7.7e-6): inside the band the
        // arithmetic is identical, so diffs can only come from outside it.
        let (w, h) = (64usize, 48usize);
        let mut base = synth(w, h);
        // Darken a patch outside the band so the deviation (if any) shows.
        for y in 0..h {
            for x in 0..10usize {
                let j = (y * w + x) * 3;
                base[j] = 20.0;
                base[j + 1] = 20.0;
                base[j + 2] = 20.0;
            }
        }
        let snap = snap_of(&base);
        let mut a = base.clone();
        let mut b = base.clone();
        apply_toner_band(&mut a, &snap, w, h, 48.0, 6.4, 0.8);
        toner_direct(&mut b, &snap, w, h, 48.0, 6.4, 0.8);
        let max = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-4, "toner clip max deviation {max} exceeds bound");
    }

    /// Legacy full-frame misreg loop (pre-Swap-2 reference).
    fn misreg_direct(buf: &mut [f32], snap: &[f32], wsz: usize, hsz: usize, misreg: [i32; 2]) {
        for y in 0..hsz {
            for x in 0..wsz {
                let sx = (x as i32 - misreg[0]).clamp(0, wsz as i32 - 1) as usize;
                let sy = (y as i32 - misreg[1]).clamp(0, hsz as i32 - 1) as usize;
                let g = ((PAPER_LUM - snap[sy * wsz + sx]) / PAPER_LUM).clamp(0.0, 1.0);
                if g > 0.05 {
                    let j = (y * wsz + x) * 3;
                    for c in 0..3 {
                        buf[j + c] = (buf[j + c] - g * 18.0).clamp(0.0, 255.0);
                    }
                }
            }
        }
    }

    #[test]
    fn misreg_tile_skip_is_bit_exact() {
        let (w, h) = (40usize, 34usize); // non-multiple of 8 exercises edges
        let buf = synth(w, h);
        let snap = snap_of(&buf);
        for misreg in [[1, 0], [0, 1], [1, 1]] {
            let mut a = buf.clone();
            let mut b = buf.clone();
            apply_misreg(&mut a, &snap, w, h, misreg);
            misreg_direct(&mut b, &snap, w, h, misreg);
            assert_eq!(a, b, "misreg tile skip must be bit-exact for {misreg:?}");
        }
    }

    /// Legacy full-clone eraser (pre-Swap-4 reference).
    fn eraser_direct(
        buf: &mut [f32],
        w: u32,
        h: u32,
        rng: &mut ChaCha8Rng,
        cx: f32,
        cy: f32,
        r: f32,
        swirl: f32,
    ) {
        let (wsz, hsz) = (w as usize, h as usize);
        let x0 = (cx - r - 2.0).max(0.0) as u32;
        let x1 = (cx + r + 2.0).min(w as f32 - 1.0) as u32;
        let y0 = (cy - r - 2.0).max(0.0) as u32;
        let y1 = (cy + r + 2.0).min(h as f32 - 1.0) as u32;
        let src = buf.to_vec();
        for y in y0..=y1 {
            for x in x0..=x1 {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let d = (dx * dx + dy * dy).sqrt();
                if d < r {
                    let a = swirl * (1.0 - d / r);
                    let (sa, ca) = a.sin_cos();
                    let sx = cx + dx * ca - dy * sa;
                    let sy = cy + dx * sa + dy * ca;
                    let p = crate::sample_bilinear(&src, w, h, sx, sy);
                    let j = (y as usize * wsz + x as usize) * 3;
                    let lift = 0.30 * (1.0 - d / r);
                    for c in 0..3 {
                        buf[j + c] = (p[c] * (1.0 - lift) + PAPER[c] * lift
                            + (rng.gen::<f32>() - 0.5) * 14.0)
                            .clamp(0.0, 255.0);
                    }
                }
            }
        }
        for _ in 0..2 {
            let snap_b = buf.to_vec();
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    if (dx * dx + dy * dy).sqrt() < r {
                        for c in 0..3 {
                            let mut acc = 0.0f32;
                            for oy in -1i32..=1 {
                                for ox in -1i32..=1 {
                                    let xx = (x as i32 + ox).clamp(0, wsz as i32 - 1) as usize;
                                    let yy = (y as i32 + oy).clamp(0, hsz as i32 - 1) as usize;
                                    acc += snap_b[(yy * wsz + xx) * 3 + c];
                                }
                            }
                            buf[(y as usize * wsz + x as usize) * 3 + c] = acc / 9.0;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn eraser_bbox_snapshot_is_bit_exact() {
        // Blobs at center, edge, and corner stress the bbox clamps.
        let (w, h) = (48u32, 36u32);
        for (cx, cy, r, swirl, seed) in [
            (24.0f32, 18.0f32, 5.0f32, 1.0f32, 7u64),
            (2.0, 3.0, 6.0, -0.8, 8),
            (46.5, 34.5, 4.0, 1.3, 9),
            (0.5, 35.0, 8.0, -1.2, 10),
        ] {
            let buf = synth(w as usize, h as usize);
            let mut a = buf.clone();
            let mut b = buf.clone();
            apply_eraser(&mut a, w, h, &mut ChaCha8Rng::seed_from_u64(seed), cx, cy, r, swirl);
            eraser_direct(&mut b, w, h, &mut ChaCha8Rng::seed_from_u64(seed), cx, cy, r, swirl);
            assert_eq!(a, b, "eraser snapshot must be bit-exact at ({cx},{cy}) r={r}");
        }
    }

    #[test]
    fn effect_differs_and_params_in_range() {
        let (w, h) = (48u32, 32u32);
        let orig = synth(48, 32);
        let mut buf = orig.clone();
        let p = apply(&mut buf, w, h, &mut ChaCha8Rng::seed_from_u64(5), 0.9, 1.0);
        assert!(p.enabled);
        assert_ne!(buf, orig);
        assert!((0.0..=1.0).contains(&p.heavy_lift));
        assert!(p.band_period >= 8.0 && p.band_period < 28.0);
        assert!(buf.iter().all(|v| (0.0..=255.0).contains(v)));
        let mut untouched = orig.clone();
        let off = apply(&mut untouched, w, h, &mut ChaCha8Rng::seed_from_u64(1), 0.0, 1.0);
        assert!(!off.enabled);
        assert_eq!(untouched, orig);
    }
}
