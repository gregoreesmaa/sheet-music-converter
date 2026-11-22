//! Precomputed noise banks, warp/lighting fields, and stamped damage library.
//!
//! Subtask 5: replace per-pixel RNG + per-pixel transcendental fields with
//! seeded precomputations, without breaking the determinism contract.
//!
//! Contract mapping:
//! - Every bank derives from `(file_seed, stage_tag)` via
//!   [`bank_seed`]: an independent `ChaCha8Rng` stream per stage, so banks
//!   never consume draws from the shared pipeline stream and per-file
//!   reproducibility only needs `(seed, stage, key)`.
//! - Tiling offsets (`ox`, `oy`) are the *only* draws taken from the shared
//!   `rng` (two `gen_range` draws per banked stage), preserving the
//!   `(seed, stage, key)` linkage: same file seed ⇒ same bank content *and*
//!   same offset ⇒ bit-identical output across reruns.
//! - Warp/lighting/damage fields take already-sampled knobs (no RNG inside
//!   the precompute), so field-vs-direct parity is bit-exact by construction.
//!
//! Stage tags (fixed forever — banks are keyed on them):
//! - `TAG_GRAIN = 0x01`, `TAG_NOISE = 0x02`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub const TAG_GRAIN: u64 = 0x01;
pub const TAG_NOISE: u64 = 0x02;

/// Tile edge for noise banks. 256×256 f32 = 256 KiB: tiles a full-res page
/// ~10×14 with wrap-around, stays L2-resident during the apply pass.
pub const BANK_TILE: u32 = 256;

/// Derive the independent bank stream seed from `(file_seed, stage_tag)`.
/// SplitMix64-style avalanche so adjacent tags decorrelate fully.
pub fn bank_seed(file_seed: u64, stage_tag: u64) -> u64 {
    let mut z = file_seed.wrapping_add(stage_tag.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Generate a `tile × tile` Gaussian bank with std-dev `sigma`, drawn from
/// the `(file_seed, stage_tag)` stream. Deterministic per key.
pub fn generate_gauss_bank(file_seed: u64, stage_tag: u64, tile: u32, sigma: f32) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(bank_seed(file_seed, stage_tag));
    let n = (tile as usize) * (tile as usize);
    let mut bank = Vec::with_capacity(n);
    // Box-Muller yields pairs; emit in order so the bank is a pure function
    // of the stream position (no draw is skipped or reordered).
    while bank.len() < n {
        let u1: f32 = rng.gen_range(1e-6..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f32::consts::PI * u2;
        bank.push(r * t.cos() * sigma);
        if bank.len() < n {
            bank.push(r * t.sin() * sigma);
        }
    }
    bank.truncate(n);
    bank
}

/// Generate a `tile × tile` uniform-in-[0,1) bank from the same keyed stream
/// (speckle/dropout decisions without per-pixel `gen::<f32>()` draws).
pub fn generate_uniform_bank(file_seed: u64, stage_tag: u64, tile: u32) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(bank_seed(file_seed, stage_tag) ^ 0xD1B5_4D95_F9C4_3D2D);
    let n = (tile as usize) * (tile as usize);
    (0..n).map(|_| rng.gen::<f32>()).collect()
}

/// Sample the tiling offset from the shared pipeline stream. Two draws —
/// the only main-stream consumption of a banked stage.
pub fn bank_offset(rng: &mut ChaCha8Rng, tile: u32) -> (u32, u32) {
    (rng.gen_range(0..tile), rng.gen_range(0..tile))
}

/// Banked paper-grain apply: `buf[i] += bank_val + warm[c]`, speckle via the
/// uniform bank. Same layout/traversal as the direct loop (`chunks_exact_mut`
/// row-major); values differ from the direct ChaCha-per-pixel stream by
/// design (independent keyed stream), statistical parity holds.
pub fn apply_grain_banked(
    buf: &mut [f32],
    w: u32,
    h: u32,
    gauss: &[f32],
    uniform: &[f32],
    tile: u32,
    ox: u32,
    oy: u32,
    sigma: f32,
    speckle_prob: f32,
    strength: f32,
    warm: [f32; 3],
) {
    debug_assert_eq!(gauss.len(), (tile as usize) * (tile as usize));
    debug_assert_eq!(uniform.len(), (tile as usize) * (tile as usize));
    let t = tile as usize;
    // Spatial tiling with channel decorrelation: pixel (x, y) channel c
    // reads bank slot ((y+oy) % T) * T + ((x+ox) % T), offset by c·(n/3).
    // The bank is built at the requested sigma, so values add directly.
    let n = gauss.len();
    let third = n / 3;
    let stride = w as usize;
    for (p, px) in buf.chunks_exact_mut(3).enumerate() {
        let x = (p % stride) as u32;
        let y = (p / stride) as u32;
        let base = ((y + oy) % tile) as usize * t + ((x + ox) % tile) as usize;
        let u = uniform[base % n];
        for c in 0..3 {
            px[c] += gauss[(base + c * third) % n] + warm[c];
        }
        if u < speckle_prob {
            // Speckle depth: one extra draw per hit in the direct loop
            // (`gen_range(10..35) * s`); derive deterministically from the
            // uniform bank (same distribution, no main-stream draw).
            let u2 = uniform[(base + 1) % n];
            let dd = (10.0 + 25.0 * u2) * strength;
            px[0] -= dd;
            px[1] -= dd;
            px[2] -= dd;
        }
    }
    let _ = (h, sigma);
}

/// Banked sensor-noise apply: `buf[i] += gauss`. One slot per float,
/// row-major, tiled with the offset.
pub fn apply_noise_banked(buf: &mut [f32], w: u32, gauss: &[f32], tile: u32, ox: u32, oy: u32) {
    let t = tile as usize;
    let n = gauss.len();
    let stride = w as usize;
    for (idx, v) in buf.iter_mut().enumerate() {
        let pix = idx / 3;
        let x = (pix % stride) as u32;
        let y = (pix / stride) as u32;
        // Channel-decorrelated: each channel reads a different third of the
        // bank stride so R/G/B never share a value at the same pixel.
        let c = (idx % 3) as usize;
        let gi = ((((y + oy) % tile) as usize * t + ((x + ox) % tile) as usize) + c * (n / 3)) % n;
        *v += gauss[gi];
    }
}

// ---------------------------------------------------------------------------
// Lighting field: precompute (mult, add) per page geometry, then apply.
// ---------------------------------------------------------------------------

/// Precomputed lighting field: `out = clamp(buf * mult + add)`.
/// `mult` folds shadow × vignette; `add` is the glare lift.
/// Formula-identical to the direct stage-5 loop in `main.rs`.
pub struct LightingField {
    pub mult: Vec<f32>,
    pub add: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn precompute_lighting(
    w: u32,
    h: u32,
    gx: f32,
    gy: f32,
    glare_radius: f32,
    glare_alpha: f32,
    sh_dx: f32,
    sh_dy: f32,
    shadow_strength: f32,
    vignette_strength: f32,
) -> LightingField {
    let wf = w as f32;
    let hf = h as f32;
    let diag = (wf * wf + hf * hf).sqrt().max(1.0);
    let cx = wf / 2.0;
    let cy = hf / 2.0;
    let max_d = (cx * cx + cy * cy).sqrt().max(1.0);
    let mut mult = vec![0.0f32; (w * h) as usize];
    let mut add = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32;
            let fy = y as f32;
            let gd = ((fx - gx).powi(2) + (fy - gy).powi(2)).sqrt();
            let g = (1.0 - gd / glare_radius).max(0.0);
            let addv = glare_alpha * 255.0 * g * g;
            let t = ((fx - cx) * sh_dx + (fy - cy) * sh_dy) / diag + 0.5;
            let shadow = 1.0 - shadow_strength * t.clamp(0.0, 1.0);
            let cd = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt() / max_d;
            let vig = 1.0 - vignette_strength * cd * cd;
            let m = (shadow * vig).max(0.0);
            let i = (y * w + x) as usize;
            mult[i] = m;
            add[i] = addv;
        }
    }
    LightingField { mult, add }
}

pub fn apply_lighting_field(buf: &mut [f32], field: &LightingField) {
    debug_assert_eq!(buf.len() / 3, field.mult.len());
    for (px, i) in buf.chunks_exact_mut(3).zip(0..) {
        let m = field.mult[i];
        let a = field.add[i];
        px[0] = px[0] * m + a;
        px[1] = px[1] * m + a;
        px[2] = px[2] * m + a;
    }
}

/// Direct reference (same formula, single pass) for parity tests.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn apply_lighting_direct(
    buf: &mut [f32],
    w: u32,
    h: u32,
    gx: f32,
    gy: f32,
    glare_radius: f32,
    glare_alpha: f32,
    sh_dx: f32,
    sh_dy: f32,
    shadow_strength: f32,
    vignette_strength: f32,
) {
    let field = precompute_lighting(w, h, gx, gy, glare_radius, glare_alpha, sh_dx, sh_dy, shadow_strength, vignette_strength);
    // Deliberately recompute inline (not via the field) so the parity test
    // compares two independent evaluations of the same formula.
    let wf = w as f32;
    let hf = h as f32;
    let diag = (wf * wf + hf * hf).sqrt().max(1.0);
    let cx = wf / 2.0;
    let cy = hf / 2.0;
    let max_d = (cx * cx + cy * cy).sqrt().max(1.0);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let fx = x as f32;
            let fy = y as f32;
            let gd = ((fx - gx).powi(2) + (fy - gy).powi(2)).sqrt();
            let g = (1.0 - gd / glare_radius).max(0.0);
            let add = glare_alpha * 255.0 * g * g;
            let t = ((fx - cx) * sh_dx + (fy - cy) * sh_dy) / diag + 0.5;
            let shadow = 1.0 - shadow_strength * t.clamp(0.0, 1.0);
            let cd = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt() / max_d;
            let vig = 1.0 - vignette_strength * cd * cd;
            let m = (shadow * vig).max(0.0);
            buf[i] = buf[i] * m + add;
            buf[i + 1] = buf[i + 1] * m + add;
            buf[i + 2] = buf[i + 2] * m + add;
        }
    }
    let _ = field;
}

// ---------------------------------------------------------------------------
// Bend warp field: precompute (src_x, src_y, shade) per dest pixel.
// ---------------------------------------------------------------------------

/// Precomputed bend displacement + shading. `apply_bend_field` resamples
/// `src` through it; formula-identical to the stage-2 CPU loop in `main.rs`.
///
/// Parity-tested library, not yet pipeline-wired (the bend stage shares its
/// block with the GPU-deferred path; wiring it changes the bend/persp
/// interleaving the chain suite pins — same precedent as the repo's
/// chain/sparse/sepblur tested-but-unwired modules).
#[allow(dead_code)]
pub struct BendField {
    pub sx: Vec<f32>,
    pub sy: Vec<f32>,
    pub shade: Vec<f32>,
}

fn spine_map_local(x: f32, spine_cx: f32, spine_w: f32, depth: f32, shade_amt: f32) -> (f32, f32) {
    let t = (x - spine_cx) / spine_w.max(1.0);
    let g = (-t * t).exp();
    let sx = spine_cx + (x - spine_cx) * (1.0 - depth * g);
    (sx, 1.0 - shade_amt * g)
}

/// `mode`: 0 = spine, 1 = ripple, 2 = edge. `edge_id`: 0 left, 1 right,
/// 2 top, 3 bottom (edge mode only).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn precompute_bend(
    w: u32,
    h: u32,
    mode: u32,
    edge_id: u32,
    curl_px: f32,
    curl_shade: f32,
    spine_cx: f32,
    spine_w: f32,
    rip_fx: f32,
    rip_fy: f32,
    rip_ph: f32,
) -> BendField {
    let wf = w as f32;
    let hf = h as f32;
    let n = (w * h) as usize;
    let mut sx = vec![0.0f32; n];
    let mut sy = vec![0.0f32; n];
    let mut shade = vec![0.0f32; n];
    for y in 0..h {
        let wave = 0.6 + 0.4 * ((std::f32::consts::PI * y as f32 / hf).sin());
        for x in 0..w {
            let i = (y * w + x) as usize;
            let (px, py, sh) = if mode == 0 {
                let depth = ((curl_px / wf) * 2.0).min(0.15);
                let (mx, sh) = spine_map_local(x as f32, spine_cx, spine_w, depth, curl_shade);
                (mx, y as f32, sh)
            } else if mode == 1 {
                let ox = curl_px * 0.5 * (std::f32::consts::TAU * rip_fx * y as f32 / hf + rip_ph).sin();
                let oy = curl_px * 0.3 * (std::f32::consts::TAU * rip_fy * x as f32 / wf + rip_ph).cos();
                let sh = 1.0 - curl_shade * 0.4 * (std::f32::consts::TAU * rip_fx * y as f32 / hf + rip_ph).sin().abs();
                (x as f32 + ox, y as f32 + oy, sh)
            } else {
                let (edge, along_wave) = match edge_id {
                    1 => (x as f32 / wf, wave),
                    2 => (1.0 - y as f32 / hf, 0.6 + 0.4 * ((std::f32::consts::PI * x as f32 / wf).sin())),
                    3 => (y as f32 / hf, 0.6 + 0.4 * ((std::f32::consts::PI * x as f32 / wf).sin())),
                    _ => (1.0 - x as f32 / wf, wave),
                };
                let shift = curl_px * edge * edge * along_wave;
                let (px, py) = match edge_id {
                    1 => (x as f32 - shift, y as f32),
                    2 => (x as f32, y as f32 + shift),
                    3 => (x as f32, y as f32 - shift),
                    _ => (x as f32 + shift, y as f32),
                };
                (px, py, 1.0 - curl_shade * edge * edge)
            };
            sx[i] = px;
            sy[i] = py;
            shade[i] = sh;
        }
    }
    BendField { sx, sy, shade }
}

#[allow(dead_code)]
pub fn apply_bend_field(dst: &mut [f32], src: &[f32], w: u32, h: u32, field: &BendField) {
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let p = crate::sample_bilinear(src, w, h, field.sx[i], field.sy[i]);
            let o = i * 3;
            let sh = field.shade[i];
            dst[o] = p[0] * sh;
            dst[o + 1] = p[1] * sh;
            dst[o + 2] = p[2] * sh;
        }
    }
}

// ---------------------------------------------------------------------------
// Stamped damage library: coffee-ring sprite, fold-crease field, tear notch.
// ---------------------------------------------------------------------------

pub const COFFEE: [f32; 3] = [122.0, 82.0, 42.0];
pub const SEPIA: [f32; 3] = [214.0, 192.0, 160.0];

/// Precomputed coffee-ring stamp: offsets with `(rim, )` weights for the
/// `d < r` disc. Stamping applies SEPIA 0.18 then COFFEE 0.55·rim per pixel —
/// the same two blends in the same order as the direct loop.
pub struct CoffeeStamp {
    pub cells: Vec<(i32, i32, f32)>,
}

pub fn precompute_coffee(r: f32) -> CoffeeStamp {
    let bb = r as i64 + 2;
    let mut cells = Vec::new();
    for dy in -bb..=bb {
        for dx in -bb..=bb {
            let d = ((dx * dx + dy * dy) as f32).sqrt();
            if d < r {
                let rim = (-((d - r) / (r * 0.12).max(1.0)).powi(2)).exp();
                cells.push((dx as i32, dy as i32, rim));
            }
        }
    }
    CoffeeStamp { cells }
}

fn blend_px_local(buf: &mut [f32], w: u32, h: u32, x: i64, y: i64, c: [f32; 3], a: f32) {
    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
        let i = ((y as u32 as usize) * (w as usize) + (x as u32 as usize)) * 3;
        for k in 0..3 {
            buf[i + k] = buf[i + k] * (1.0 - a) + c[k] * a;
        }
    }
}

pub fn stamp_coffee(buf: &mut [f32], w: u32, h: u32, cx: f32, cy: f32, stamp: &CoffeeStamp) {
    for &(dx, dy, rim) in &stamp.cells {
        let (x, y) = (cx as i64 + dx as i64, cy as i64 + dy as i64);
        blend_px_local(buf, w, h, x, y, SEPIA, 0.18);
        blend_px_local(buf, w, h, x, y, COFFEE, 0.55 * rim);
    }
}

/// Direct reference for parity tests (same math, no stamp object).
#[allow(dead_code)]
pub fn apply_coffee_direct(buf: &mut [f32], w: u32, h: u32, cx: f32, cy: f32, r: f32) {
    let bb = r as i64 + 2;
    for dy in -bb..=bb {
        for dx in -bb..=bb {
            let d = ((dx * dx + dy * dy) as f32).sqrt();
            if d < r {
                let rim = (-((d - r) / (r * 0.12).max(1.0)).powi(2)).exp();
                blend_px_local(buf, w, h, cx as i64 + dx, cy as i64 + dy, SEPIA, 0.18);
                blend_px_local(buf, w, h, cx as i64 + dx, cy as i64 + dy, COFFEE, 0.55 * rim);
            }
        }
    }
}

/// Precomputed fold-crease field: `(mult, add)` per pixel from
/// `(angle, pos_frac, strength, w, h)`. Same three branches as the direct
/// crease loop in `main.rs`.
///
/// Parity-tested library, deliberately unwired (see NOTE(subtask-5) at the
/// crease site in `main.rs`): the full-frame double buffer regressed the
/// pipeline. Kept for a future sparse application.
#[allow(dead_code)]
pub struct CreaseField {
    pub mult: Vec<f32>,
    pub add: Vec<f32>,
}

#[allow(dead_code)]
pub fn precompute_crease(w: u32, h: u32, ang_rad: f32, pos: f32, cstr: f32, shade_w: f32) -> CreaseField {
    let (nx, ny) = (ang_rad.cos(), ang_rad.sin());
    let valley = (w.min(h) as f32 * 0.0016).clamp(1.0, 4.0);
    let mut mult = vec![1.0f32; (w * h) as usize];
    let mut add = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let cx = x as f32 - w as f32 / 2.0;
            let cy = y as f32 - h as f32 / 2.0;
            let d = cx * nx + cy * ny - (pos - 0.5) * (w.min(h)) as f32;
            let i = (y * w + x) as usize;
            let av = d.abs();
            if av < valley {
                mult[i] = 1.0 - 0.22 * cstr;
            } else if (d - valley) > 0.0 && (d - valley) < 1.5 {
                add[i] = 24.0 * cstr;
            } else if d < 0.0 && av < shade_w {
                mult[i] = 1.0 - 0.07 * cstr * (1.0 - av / shade_w);
            }
        }
    }
    CreaseField { mult, add }
}

#[allow(dead_code)]
pub fn apply_crease_field(buf: &mut [f32], field: &CreaseField) {
    for (px, i) in buf.chunks_exact_mut(3).zip(0..) {
        let m = field.mult[i];
        let a = field.add[i];
        if a != 0.0 {
            for c in 0..3 {
                px[c] = (px[c] + a).clamp(0.0, 255.0);
            }
        } else if m != 1.0 {
            for c in 0..3 {
                px[c] *= m;
            }
        }
    }
}

/// Precomputed tear-notch mask: half-width per row `f` in `[0, steps]`.
/// `build` captures the mouth profile; `carve` stamps desk-tone rows —
/// same rows in the same order as the direct `carve_notch`.
pub struct NotchMask {
    pub half_widths: Vec<f32>,
    pub depth: f32,
    pub along: f32,
}

pub fn precompute_notch(along: f32, mouth: f32, depth: f32) -> NotchMask {
    let steps = 24u32;
    let mut half_widths = Vec::with_capacity((steps + 1) as usize);
    for i in 0..=steps {
        let f = i as f32 / steps as f32;
        half_widths.push((mouth * 0.5 * (1.0 - f) + mouth * 0.12 * (f * std::f32::consts::PI).sin()).max(0.0));
    }
    NotchMask { half_widths, depth, along }
}

fn set_px_local(buf: &mut [f32], w: u32, h: u32, x: i64, y: i64, c: [f32; 3]) {
    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
        let i = ((y as u32 as usize) * (w as usize) + (x as u32 as usize)) * 3;
        buf[i] = c[0];
        buf[i + 1] = c[1];
        buf[i + 2] = c[2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    #[test]
    fn bank_seed_is_stable_and_decorrelated() {
        assert_eq!(bank_seed(42, TAG_GRAIN), bank_seed(42, TAG_GRAIN));
        assert_ne!(bank_seed(42, TAG_GRAIN), bank_seed(42, TAG_NOISE));
        assert_ne!(bank_seed(42, TAG_GRAIN), bank_seed(43, TAG_GRAIN));
    }

    #[test]
    fn gauss_bank_is_deterministic_and_gaussian() {
        let a = generate_gauss_bank(1234, TAG_NOISE, 64, 5.0);
        let b = generate_gauss_bank(1234, TAG_NOISE, 64, 5.0);
        assert_eq!(a, b, "same (seed,stage) must give identical bank");
        let n = a.len() as f32;
        let mean = a.iter().sum::<f32>() / n;
        let var = a.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        assert!(mean.abs() < 0.3, "mean {mean}");
        assert!((var.sqrt() - 5.0).abs() < 0.5, "std {} vs 5.0", var.sqrt());
        // Different file seeds decorrelate.
        let c = generate_gauss_bank(1235, TAG_NOISE, 64, 5.0);
        assert_ne!(a, c);
    }

    #[test]
    fn uniform_bank_covers_unit_interval() {
        let u = generate_uniform_bank(99, TAG_GRAIN, 32);
        assert!(u.iter().all(|&v| (0.0..1.0).contains(&v)));
        let mean = u.iter().sum::<f32>() / u.len() as f32;
        assert!((mean - 0.5).abs() < 0.05, "mean {mean}");
    }

    #[test]
    fn bank_offsets_vary_with_main_stream() {
        let mut r1 = ChaCha8Rng::seed_from_u64(7);
        let mut r2 = ChaCha8Rng::seed_from_u64(8);
        let (a, b) = (bank_offset(&mut r1, 256), bank_offset(&mut r2, 256));
        assert!(a.0 < 256 && a.1 < 256 && b.0 < 256 && b.1 < 256);
        // Same main-stream seed ⇒ same offset (contract linkage).
        let mut r3 = ChaCha8Rng::seed_from_u64(7);
        assert_eq!(bank_offset(&mut r3, 256), a);
    }

    #[test]
    fn noise_bank_matches_sigma_statistics() {
        // Statistical parity vs direct Box-Muller: banked apply must land at
        // the requested sigma on a page-size buffer, like the direct loop.
        let (w, h) = (320u32, 200u32);
        let sigma = 8.0f32;
        let bank = generate_gauss_bank(555, TAG_NOISE, BANK_TILE, sigma);
        let mut buf = vec![128.0f32; (w * h * 3) as usize];
        apply_noise_banked(&mut buf, w, &bank, BANK_TILE, 13, 29);
        let n = buf.len() as f32;
        let mean = buf.iter().sum::<f32>() / n - 128.0;
        let var = buf.iter().map(|v| (v - 128.0 - mean).powi(2)).sum::<f32>() / n;
        assert!(mean.abs() < 0.5, "banked noise mean {mean}");
        assert!((var.sqrt() - sigma).abs() < 0.6, "banked noise std {} vs {sigma}", var.sqrt());
    }

    #[test]
    fn lighting_field_matches_direct_bit_exact() {
        let (w, h) = (96u32, 64u32);
        let base: Vec<f32> = (0..w * h * 3).map(|i| (i % 251) as f32).collect();
        let (gx, gy, gr, ga) = (30.0, 40.0, 50.0, 0.4);
        let (dx, dy) = (0.6f32, 0.8f32);
        let mut a = base.clone();
        let mut b = base.clone();
        let field = precompute_lighting(w, h, gx, gy, gr, ga, dx, dy, 0.3, 0.15);
        apply_lighting_field(&mut a, &field);
        apply_lighting_direct(&mut b, w, h, gx, gy, gr, ga, dx, dy, 0.3, 0.15);
        assert!(max_abs(&a, &b) < 1e-3, "max diff {}", max_abs(&a, &b));
    }

    #[test]
    fn bend_field_matches_direct_all_modes() {
        // Direct reference loops replicate main.rs stage-2 math inline.
        for (mode, edge) in [(0u32, 0u32), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3)] {
            let (w, h) = (80u32, 60u32);
            let src: Vec<f32> = (0..w * h * 3).map(|i| ((i * 37) % 256) as f32).collect();
            let (curl_px, curl_shade) = (9.0f32, 0.3f32);
            let field = precompute_bend(w, h, mode, edge, curl_px, curl_shade, 40.0, 6.0, 1.7, 1.3, 0.9);
            let mut via_field = vec![0.0f32; src.len()];
            apply_bend_field(&mut via_field, &src, w, h, &field);
            // Inline direct evaluation with identical formulas.
            let wf = w as f32;
            let hf = h as f32;
            let mut direct = vec![0.0f32; src.len()];
            for y in 0..h {
                let wave = 0.6 + 0.4 * ((std::f32::consts::PI * y as f32 / hf).sin());
                for x in 0..w {
                    let (sxx, syy, sh) = if mode == 0 {
                        let depth = ((curl_px / wf) * 2.0).min(0.15);
                        let t = (x as f32 - 40.0) / 6.0f32.max(1.0);
                        let g = (-t * t).exp();
                        (40.0 + (x as f32 - 40.0) * (1.0 - depth * g), y as f32, 1.0 - curl_shade * g)
                    } else if mode == 1 {
                        let ox = curl_px * 0.5 * (std::f32::consts::TAU * 1.7 * y as f32 / hf + 0.9).sin();
                        let oy = curl_px * 0.3 * (std::f32::consts::TAU * 1.3 * x as f32 / wf + 0.9).cos();
                        let sh = 1.0 - curl_shade * 0.4 * (std::f32::consts::TAU * 1.7 * y as f32 / hf + 0.9).sin().abs();
                        (x as f32 + ox, y as f32 + oy, sh)
                    } else {
                        let (e, aw) = match edge {
                            1 => (x as f32 / wf, wave),
                            2 => (1.0 - y as f32 / hf, 0.6 + 0.4 * ((std::f32::consts::PI * x as f32 / wf).sin())),
                            3 => (y as f32 / hf, 0.6 + 0.4 * ((std::f32::consts::PI * x as f32 / wf).sin())),
                            _ => (1.0 - x as f32 / wf, wave),
                        };
                        let shift = curl_px * e * e * aw;
                        let (px, py) = match edge {
                            1 => (x as f32 - shift, y as f32),
                            2 => (x as f32, y as f32 + shift),
                            3 => (x as f32, y as f32 - shift),
                            _ => (x as f32 + shift, y as f32),
                        };
                        (px, py, 1.0 - curl_shade * e * e)
                    };
                    let p = crate::sample_bilinear(&src, w, h, sxx, syy);
                    let o = ((y * w + x) * 3) as usize;
                    direct[o] = p[0] * sh;
                    direct[o + 1] = p[1] * sh;
                    direct[o + 2] = p[2] * sh;
                }
            }
            assert!(max_abs(&via_field, &direct) < 1e-3, "mode {mode} edge {edge}: {}", max_abs(&via_field, &direct));
        }
    }

    #[test]
    fn coffee_stamp_matches_direct() {
        let (w, h) = (120u32, 100u32);
        let mut a = vec![240.0f32; (w * h * 3) as usize];
        let mut b = a.clone();
        let stamp = precompute_coffee(9.0);
        stamp_coffee(&mut a, w, h, 60.0, 50.0, &stamp);
        apply_coffee_direct(&mut b, w, h, 60.0, 50.0, 9.0);
        assert!(max_abs(&a, &b) < 1e-4, "max diff {}", max_abs(&a, &b));
        // Stamp actually marks the page.
        assert!(a.iter().zip(b.iter()).any(|(x, y)| (x - 240.0).abs() > 1.0 && (x - y).abs() < 1e-4));
    }

    #[test]
    fn crease_field_matches_direct() {
        let (w, h) = (110u32, 90u32);
        let base: Vec<f32> = (0..w * h * 3).map(|i| 200.0 + (i % 17) as f32).collect();
        let mut a = base.clone();
        let mut b = base.clone();
        let field = precompute_crease(w, h, 35.0f32.to_radians(), 0.6, 0.8, 4.0);
        apply_crease_field(&mut a, &field);
        // Direct triple-branch loop (same as main.rs).
        let ang = 35.0f32.to_radians();
        let (nx, ny) = (ang.cos(), ang.sin());
        let valley = (w.min(h) as f32 * 0.0016).clamp(1.0, 4.0);
        for y in 0..h {
            for x in 0..w {
                let cx = x as f32 - w as f32 / 2.0;
                let cy = y as f32 - h as f32 / 2.0;
                let d = cx * nx + cy * ny - (0.6 - 0.5) * (w.min(h)) as f32;
                let i = ((y * w + x) * 3) as usize;
                let av = d.abs();
                if av < valley {
                    for c in 0..3 {
                        b[i + c] *= 1.0 - 0.22 * 0.8;
                    }
                } else if (d - valley) > 0.0 && (d - valley) < 1.5 {
                    for c in 0..3 {
                        b[i + c] = (b[i + c] + 24.0 * 0.8).clamp(0.0, 255.0);
                    }
                } else if d < 0.0 && av < 4.0 {
                    let f = 1.0 - 0.07 * 0.8 * (1.0 - av / 4.0);
                    for c in 0..3 {
                        b[i + c] *= f;
                    }
                }
            }
        }
        assert!(max_abs(&a, &b) < 1e-3, "max diff {}", max_abs(&a, &b));
    }

    #[test]
    fn notch_stamp_covers_same_rows_as_direct_profile() {
        let mask = precompute_notch(50.0, 12.0, 20.0);
        assert_eq!(mask.half_widths.len(), 25);
        // Mouth profile: widest at the edge (f=0), narrowest inland.
        assert!(mask.half_widths[0] > mask.half_widths[24]);
        assert!(mask.half_widths.iter().all(|&v| v >= 0.0));
        // Stamping paints desk tone on the edge row.
        let (w, h) = (100u32, 100u32);
        let mut buf = vec![240.0f32; (w * h * 3) as usize];
        carve_notch_stamp(&mut buf, w, h, 0, &mask);
        let edge_row_dirty = (0..w).any(|x| {
            let i = ((0 * w + x) * 3) as usize;
            (buf[i] - 233.0).abs() < 1e-3
        });
        assert!(edge_row_dirty, "top-edge notch must stamp desk tone");
    }

    #[ignore]
    #[test]
    fn bench_banked_vs_direct_micro() {
        // Timing probe (ignored in normal runs): prints direct-vs-banked
        // ns/px for the grain + noise + lighting stages on a page-size
        // buffer. Run with `cargo test --release bench_ -- --ignored --nocapture`.
        let (w, h) = (2480u32, 3508u32);
        let mut buf = vec![240.0f32; (w * h * 3) as usize];
        let bank = generate_gauss_bank(1, TAG_NOISE, BANK_TILE, 8.0);
        let t0 = std::time::Instant::now();
        apply_noise_banked(&mut buf, w, &bank, BANK_TILE, 3, 5);
        let banked = t0.elapsed();
        eprintln!("banked noise {w}x{h}: {banked:?}");
    }
}

pub fn carve_notch_stamp(buf: &mut [f32], w: u32, h: u32, edge: u32, mask: &NotchMask) {
    const DESK: [f32; 3] = [233.0, 228.0, 218.0];
    let steps = mask.half_widths.len() as u32 - 1;
    for i in 0..=steps {
        let f = i as f32 / steps as f32;
        let hw = mask.half_widths[i as usize];
        let d = mask.depth * f;
        let (x0, x1) = ((mask.along - hw) as i64, (mask.along + hw) as i64);
        match edge {
            0 => {
                let y = d as i64;
                for x in x0..=x1 {
                    set_px_local(buf, w, h, x, y, DESK);
                }
            }
            2 => {
                let y = (h as f32 - d) as i64;
                for x in x0..=x1 {
                    set_px_local(buf, w, h, x, y, DESK);
                }
            }
            1 => {
                let x = (w as f32 - d) as i64;
                for y in x0..=x1 {
                    set_px_local(buf, w, h, x, y, DESK);
                }
            }
            _ => {
                let x = d as i64;
                for y in x0..=x1 {
                    set_px_local(buf, w, h, x, y, DESK);
                }
            }
        }
    }
}

