//! Optional WGPU acceleration for the dense per-pixel stages of
//! [`crate::degrade_file`] (paper grain, page bend, defocus blur,
//! perspective warp, rotation, lighting + chromatic aberration fused in one
//! round-trip, sensor noise), enabled with `--gpu`.
//!
//! Semantics mirror the CPU stages exactly (same gates, same fills, same
//! recorded knobs) so params stay comparable across backends, with three
//! documented differences:
//! * paper grain and sensor noise use GPU hash PRNG streams (distinct stream
//!   ids per stage) instead of the shared ChaCha8 stream, so pixels differ
//!   from the CPU run with the same seed (deterministic per backend) and
//!   downstream ChaCha draws shift; `backend` in the params JSON records
//!   which stream produced a file.
//! * rotation pixel work is deferred to the 6b fused rotate+downscale
//!   affine dispatch (single-sample bilinear), so stages 3b-6 run unrotated
//!   and dims-derived knobs shift by the canvas ratio vs the eager CPU
//!   rotation; `rot_scale_fused` in the params JSON records the path.
//! * the CPU downscale uses the SIMD bilinear `fast_image_resize` resample;
//!   the GPU affine kernel is bilinear-only.

use degrade_gpu::chain::ChainConfig;
use degrade_gpu::gpu::{BendKnobs, GrainKnobs, GpuContext, LightingKnobs};

/// Fused rotate+downscale frame math (more-spec-affine option b), re-exported
/// from `degrade-gpu`: R^-1 rows + `(w2, h2, 1/sx, 1/sy)` meta plus the
/// `(w2, h2, nw, nh)` dims bookkeeping.
pub use degrade_gpu::gpu::rotscale_frame;

/// Paper-tone fill used by the CPU bilinear sampler for out-of-source taps.
#[allow(dead_code)] // test-only since chain1 carries its own paper fill; see `chain()`.
pub const PAPER_FILL: [f32; 3] = [246.0, 243.0, 236.0];
/// Dark-desk fill used by the CPU rotation stage for newly exposed canvas.
#[allow(dead_code)] // test-only: production runs affine inside `chain()`.
pub const DESK_FILL: [f32; 3] = [25.0, 25.0, 30.0];

/// Shared WGPU context. `wgpu::Device`/`Queue`/pipelines are `Send + Sync`,
/// so one `Gpu` serves every worker thread in the directory-run scope —
/// but concurrent submit/map/poll traffic from N threads on the one shared
/// device inflates every call ~10x (measured: 10-thread mixed full-res
/// sequence 30.9 s wall vs 2.3 s serialized; single calls stay ~70 ms).
/// The `Mutex` serializes whole device round-trips across threads while
/// CPU stages keep running in parallel; pixels are untouched (per-file
/// seeds), only wait time changes.
pub struct Gpu {
    ctx: std::sync::Mutex<GpuContext>,
}

impl Gpu {
    /// Request the high-performance adapter. Returns `Err` (no panic) on
    /// machines without a usable GPU so `--gpu` fails with a clean message.
    pub fn new() -> Result<Self, String> {
        match pollster::block_on(GpuContext::try_new()) {
            Ok(ctx) => Ok(Self { ctx: std::sync::Mutex::new(ctx) }),
            Err(e) => Err(format!("GPU init failed: {e}")),
        }
    }

    /// Lock the shared device for one whole round-trip (see the struct
    /// docs for why whole-call serialization beats concurrent submit/poll
    /// on one `wgpu` device). The returned guard lives for the enclosing
    /// statement, so `self.lock().grain(..)` holds the gate for the call.
    fn lock(&self) -> std::sync::MutexGuard<'_, GpuContext> {
        self.ctx.lock().expect("GPU mutex poisoned")
    }

    /// Inverse-map resample, same convention as the CPU `apply_h` helper:
    /// row-major 3x3, `sx = (r0*x + r1*y + r2)/w`.
    /// Kept for the unit tests below (the `--gpu` pipeline defers rotation
    /// to the 6b [`Gpu::affine`] dispatch instead of resampling eagerly).
    #[allow(dead_code)]
    pub fn resample(
        &self,
        src: &[f32],
        w: u32,
        h: u32,
        w2: u32,
        h2: u32,
        inv: [f32; 9],
        fill: [f32; 3],
    ) -> Vec<f32> {
        self.lock().resample(src, w, h, w2, h2, inv, fill)
    }

    /// Fused rotate+downscale dispatch (more-spec-affine option b): one
    /// `affine` call reads the pre-rotation buffer and writes `(nw, nh)`
    /// directly, replacing the eager rotation resample plus a separate
    /// downscale. `rows`/`meta` come from [`rotscale_frame`]; fill is desk
    /// tone on both OOB levels, matching the GPU rotation path.
    #[allow(dead_code)] // test-only: production fuses affine into `chain()`.
    pub fn affine(
        &self,
        src: &[f32],
        w: u32,
        h: u32,
        nw: u32,
        nh: u32,
        rows: [f32; 6],
        meta: [f32; 4],
    ) -> Vec<f32> {
        self.lock().affine(src, w, h, nw, nh, rows, meta, DESK_FILL)
    }

    /// Stage-5 lighting with the same knobs the CPU block samples.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // test-only: the --gpu path runs the resident chain.
    pub fn lighting(
        &self,
        src: &[f32],
        w: u32,
        h: u32,
        glare_center: [f32; 2],
        glare_radius: f32,
        glare_alpha: f32,
        shadow_angle_deg: f32,
        shadow_strength: f32,
        vignette_strength: f32,
    ) -> Vec<f32> {
        let theta = shadow_angle_deg.to_radians();
        self.lock().lighting(
            src,
            w,
            h,
            LightingKnobs {
                gx: glare_center[0],
                gy: glare_center[1],
                glare_radius,
                glare_alpha,
                sh_dx: theta.cos(),
                sh_dy: theta.sin(),
                shadow_strength,
                vignette_strength,
            },
        )
    }

    /// Stage-5b chromatic aberration (radial R/B split, `ca_px` at corners).
    /// Kept for tests and direct use; the pipeline fuses it with lighting
    /// via [`Gpu::light_ca`].
    #[allow(dead_code)]
    pub fn chromatic(&self, src: &[f32], w: u32, h: u32, ca_px: f32) -> Vec<f32> {
        self.lock().chromatic(src, w, h, ca_px)
    }

    /// Stage-6 sensor noise. `seed` is the folded 64-bit file seed; the GPU
    /// hash stream is independent of the CPU ChaCha stream by design.
    #[allow(dead_code)] // test-only: the --gpu path runs the resident chain.
    pub fn noise(&self, src: &[f32], w: u32, h: u32, sigma: f32, seed: u32) -> Vec<f32> {
        self.lock().noise(src, w, h, sigma, seed)
    }

    /// Resident single-submit run of a stage subset on the shared device
    /// (see `GpuContext::chain_full`): the `--gpu` path calls this twice
    /// per file — pre-framing (`run_grain/bend/persp`) and post-framing
    /// (`run_rot/blur/light/ca/noise`) — because the CPU framing stage must
    /// stay between the perspective warp and the rotation. Each call is one
    /// submit on this shared context: no second device, no per-stage
    /// round-trips. `cfg` carries the same knobs the params record.
    pub fn chain(
        &self,
        src: &[f32],
        w: u32,
        h: u32,
        cfg: &ChainConfig,
    ) -> (Vec<f32>, u32, u32) {
        self.lock().chain_full(src, w, h, cfg)
    }

    /// Stage-1 paper grain. Same knobs the CPU stage samples (sigma, speckle
    /// probability, warm tint, strength); the per-pixel stream is the GPU
    /// hash variant, so pixels differ from the CPU ChaCha run at the same
    /// seed (deterministic per backend) exactly like sensor noise.
    /// `seed` must differ from the noise stream id (see `grain_seed32`).
    pub fn grain(
        &self,
        src: &[f32],
        w: u32,
        h: u32,
        sigma: f32,
        seed: u32,
        speckle_prob: f32,
        strength: f32,
        warm: [f32; 3],
    ) -> Vec<f32> {
        self.lock().grain(
            src,
            w,
            h,
            GrainKnobs { sigma, seed, speckle_prob, strength, warm },
        )
    }

    /// Stage-2 page bend. `mode`: 0 = spine, 1 = ripple, 2 = edge curl;
    /// `edge`: 0 = left, 1 = right, 2 = top, 3 = bottom. `depth` replicates
    /// the CPU spine-gutter formula `((curl_px / w) * 2).min(0.15)`.
    #[allow(dead_code)] // test-only: the --gpu path runs the resident chain.
    pub fn bend(
        &self,
        src: &[f32],
        w: u32,
        h: u32,
        mode: u32,
        edge: u32,
        curl_px: f32,
        curl_shade: f32,
        spine_cx: f32,
        spine_w: f32,
        rip_fx: f32,
        rip_fy: f32,
        rip_ph: f32,
    ) -> Vec<f32> {
        let wf = w as f32;
        self.lock().bend(
            src,
            w,
            h,
            BendKnobs {
                mode,
                edge_id: edge,
                curl_px,
                curl_shade,
                spine_cx,
                spine_w,
                depth: ((curl_px / wf) * 2.0).min(0.15),
                rip_fx,
                rip_fy,
                rip_ph,
            },
        )
    }

    /// Stage-4 defocus blur (uniform path). `sigma <= 0` is identity.
    #[allow(dead_code)] // test-only: the GPU path runs the resident chain.
    pub fn blur(&self, src: &[f32], w: u32, h: u32, sigma: f32) -> Vec<f32> {
        self.lock().blur(src, w, h, sigma)
    }
}

/// Fold a 64-bit file seed to the 32-bit GPU noise stream id. XOR-fold keeps
/// both halves influential; the CPU path never uses this value.
pub fn noise_seed32(file_seed: u64) -> u32 {
    (file_seed ^ (file_seed >> 32)) as u32
}

/// Fold a 64-bit file seed to the 32-bit GPU *grain* stream id. The grain
/// kernel draws from the same hash family as the noise kernel, so the two
/// stages must never share a stream id for one file (their per-pixel draws
/// would correlate). The fixed stage tag keeps every file's grain stream
/// distinct from its noise stream while staying deterministic per backend.
pub fn grain_seed32(file_seed: u64) -> u32 {
    noise_seed32(file_seed ^ 0x9E3779B97F4A7C15)
}

/// Rotation frame matching the CPU stage-3a canvas expansion: returns the
/// row-major 3x3 inverse map (dst -> src) plus the expanded canvas dims.
/// `rotation_deg` follows the CPU sign convention (positive = clockwise in
/// image coordinates, matching `sx = cx0 + dx*cos + dy*sin`).
///
/// Kept for the unit tests below (the `--gpu` pipeline now defers rotation
/// to the 6b [`Gpu::affine`] dispatch instead of resampling eagerly here).
#[allow(dead_code)]
pub fn rotation_frame(rotation_deg: f32, w: u32, h: u32) -> ([f32; 9], u32, u32) {
    let th = rotation_deg.to_radians();
    let (sn, cs) = (th.sin().abs(), th.cos().abs());
    let w2 = ((w as f32) * cs + (h as f32) * sn).ceil().max(1.0) as u32;
    let h2 = ((w as f32) * sn + (h as f32) * cs).ceil().max(1.0) as u32;
    let (snt, cst) = (th.sin(), th.cos());
    let (cx0, cy0) = (w as f32 / 2.0, h as f32 / 2.0);
    let (cx1, cy1) = (w2 as f32 / 2.0, h2 as f32 / 2.0);
    let inv = [
        cst,
        snt,
        cx0 - cx1 * cst - cy1 * snt,
        -snt,
        cst,
        cy0 + cx1 * snt - cy1 * cst,
        0.0,
        0.0,
        1.0,
    ];
    (inv, w2, h2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_inv(inv: &[f32; 9], x: f32, y: f32) -> (f32, f32) {
        let z = inv[6] * x + inv[7] * y + inv[8];
        (
            (inv[0] * x + inv[1] * y + inv[2]) / z,
            (inv[3] * x + inv[4] * y + inv[5]) / z,
        )
    }

    #[test]
    fn rotation_frame_zero_is_identity() {
        let (inv, w2, h2) = rotation_frame(0.0, 200, 100);
        assert_eq!((w2, h2), (200, 100));
        for (x, y) in [(0.0, 0.0), (199.0, 99.0), (100.0, 50.0), (37.0, 71.0)] {
            let (sx, sy) = apply_inv(&inv, x, y);
            assert!((sx - x).abs() < 1e-4, "x {x}: {sx}");
            assert!((sy - y).abs() < 1e-4, "y {y}: {sy}");
        }
    }

    #[test]
    fn rotation_frame_90_swaps_canvas_and_maps_corners() {
        let (inv, w2, h2) = rotation_frame(90.0, 200, 100);
        // 200x100 rotated 90 deg expands to a ~100x200 canvas (allow a
        // 1px f32 ceil artifact either way; libm results vary by platform).
        assert!((99..=102).contains(&w2), "w2 {w2}");
        assert!((199..=202).contains(&h2), "h2 {h2}");
        // Dst center maps back to src center.
        let (sx, sy) = apply_inv(&inv, w2 as f32 / 2.0, h2 as f32 / 2.0);
        assert!((sx - 100.0).abs() < 2.0, "sx {sx}");
        assert!((sy - 50.0).abs() < 2.0, "sy {sy}");
        // Full visibility: every source corner forward-maps inside dst.
        // Forward map is the inverse rotation: solve the orthonormal part.
        let (c, s) = (inv[0], inv[1]); // cos, sin of the inverse map
        for (px, py) in [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)] {
            let dx = px - 100.0;
            let dy = py - 50.0;
            let fx = w2 as f32 / 2.0 + dx * c - dy * s;
            let fy = h2 as f32 / 2.0 + dx * s + dy * c;
            assert!((-2.0..w2 as f32 + 2.0).contains(&fx), "corner ({px},{py}) -> fx {fx}");
            assert!((-2.0..h2 as f32 + 2.0).contains(&fy), "corner ({px},{py}) -> fy {fy}");
        }
    }

    #[test]
    fn rotation_frame_matches_cpu_stage_formula() {
        // Canvas expansion must equal the CPU stage-3a formula for any angle.
        for deg in [-10.0f32, -3.25, 0.5, 7.75, 10.0] {
            let th = deg.to_radians();
            let (sn, cs) = (th.sin().abs(), th.cos().abs());
            let ew = ((2480.0f32) * cs + (3508.0f32) * sn).ceil().max(1.0) as u32;
            let eh = ((2480.0f32) * sn + (3508.0f32) * cs).ceil().max(1.0) as u32;
            let (_, w2, h2) = rotation_frame(deg, 2480, 3508);
            assert_eq!((w2, h2), (ew, eh), "deg {deg}");
        }
    }

    /// Fused-map evaluator mirroring the affine kernel's expression tree
    /// (`xr = x*sx_inv`, then rows, DESK fill on both OOB levels): the
    /// test-only CPU mirror of the GPU kernel (cf. more-spec-affine §6).
    fn cpu_affine_mirror(
        src: &[f32],
        w: u32,
        h: u32,
        nw: u32,
        nh: u32,
        rows: [f32; 6],
        meta: [f32; 4],
        fill: [f32; 3],
    ) -> Vec<f32> {
        let tap = |ix: i64, iy: i64| -> [f32; 3] {
            if ix < 0 || iy < 0 || ix >= w as i64 || iy >= h as i64 {
                return fill;
            }
            let i = ((iy as u32 * w + ix as u32) * 3) as usize;
            [src[i], src[i + 1], src[i + 2]]
        };
        let mut out = vec![0.0f32; (nw * nh * 3) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let xr = x as f32 * meta[2];
                let yr = y as f32 * meta[3];
                let px = if xr < 0.0 || yr < 0.0 || xr >= meta[0] || yr >= meta[1] {
                    fill
                } else {
                    let sx_ = rows[0] * xr + rows[1] * yr + rows[2];
                    let sy_ = rows[3] * xr + rows[4] * yr + rows[5];
                    let x0 = sx_.floor() as i64;
                    let y0 = sy_.floor() as i64;
                    let (tx, ty) = (sx_ - x0 as f32, sy_ - y0 as f32);
                    let mut p = [0.0f32; 3];
                    for c in 0..3 {
                        let top = tap(x0, y0)[c] * (1.0 - tx) + tap(x0 + 1, y0)[c] * tx;
                        let bot = tap(x0, y0 + 1)[c] * (1.0 - tx) + tap(x0 + 1, y0 + 1)[c] * tx;
                        p[c] = top * (1.0 - ty) + bot * ty;
                    }
                    p
                };
                let i = ((y * nw + x) * 3) as usize;
                out[i..i + 3].copy_from_slice(&px);
            }
        }
        out
    }

    #[test]
    fn rotscale_frame_zero_is_identity() {
        // deg=0, sx=sy=1: identity map, dims unchanged.
        let (rows, meta, w2, h2, nw, nh) = rotscale_frame(0.0, 200, 100, 1.0, 1.0);
        assert_eq!((w2, h2), (200, 100));
        assert_eq!((nw, nh), (200, 100));
        assert_eq!(meta, [200.0, 100.0, 1.0, 1.0]);
        for (x, y) in [(0.0, 0.0), (199.0, 99.0), (100.0, 50.0)] {
            let xr = x * meta[2];
            let yr = y * meta[3];
            let sx_ = rows[0] * xr + rows[1] * yr + rows[2];
            let sy_ = rows[3] * xr + rows[4] * yr + rows[5];
            assert!((sx_ - x).abs() < 1e-4, "x {x}: {sx_}");
            assert!((sy_ - y).abs() < 1e-4, "y {y}: {sy_}");
        }
    }

    #[test]
    fn rotscale_frame_dims_match_legacy_truncation() {
        // Canvas matches the legacy rotation expansion; output dims match
        // the legacy `trunc(w2*sx)` capture-downscale formula exactly.
        for deg in [-10.0f32, -3.25, 0.5, 7.75, 10.0] {
            for (sx, sy) in [(1.0f32, 1.0f32), (0.75, 0.75), (0.5, 0.9), (1.0, 0.6)] {
                for (w, h) in [(2480u32, 3508u32), (1024, 1447)] {
                    let (rows, meta, w2, h2, nw, nh) = rotscale_frame(deg, w, h, sx, sy);
                    let (_, ew2, eh2) = rotation_frame(deg, w, h);
                    assert_eq!((w2, h2), (ew2, eh2), "deg {deg} canvas");
                    assert_eq!(nw, (w2 as f32 * sx) as u32, "deg {deg} scales {sx},{sy}");
                    assert_eq!(nh, (h2 as f32 * sy) as u32, "deg {deg} scales {sx},{sy}");
                    assert_eq!(meta, [w2 as f32, h2 as f32, 1.0 / sx, 1.0 / sy]);
                    assert!(rows[0].is_finite() && meta[2].is_finite());
                }
            }
        }
    }

    #[test]
    fn rotscale_degenerate_matches_rotation_frame_rows() {
        // With sx=sy=1 the composed map must equal the plain rotation
        // inverse (same R^-1 rows up to fp association): dst center maps to
        // src center within 2 px.
        let (rows, meta, w2, h2, nw, nh) = rotscale_frame(5.0, 64, 48, 1.0, 1.0);
        assert_eq!((nw, nh), (w2, h2));
        let (inv, _, _) = rotation_frame(5.0, 64, 48);
        for (x, y) in [(w2 as f32 / 2.0, h2 as f32 / 2.0), (0.0, 0.0)] {
            let xr = x * meta[2];
            let yr = y * meta[3];
            let a = (rows[0] * xr + rows[1] * yr + rows[2], rows[3] * xr + rows[4] * yr + rows[5]);
            let z = inv[6] * x + inv[7] * y + inv[8];
            let b = ((inv[0] * x + inv[1] * y + inv[2]) / z, (inv[3] * x + inv[4] * y + inv[5]) / z);
            assert!((a.0 - b.0).abs() < 2.0, "x {x}: {a:?} vs {b:?}");
            assert!((a.1 - b.1).abs() < 2.0, "y {y}: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn affine_rotation_only_matches_resample_tolerance() {
        // Live-GPU check (skips cleanly without an adapter): rotation-only
        // fused dispatch vs the eager `resample` rotation path, <= 2 LSB
        // (op-order ULP between `x*sx_inv`+rows vs the legacy row eval).
        let g = match Gpu::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        let (w, h) = (96u32, 64u32);
        let mut buf = vec![240.0f32; (w * h * 3) as usize];
        for y in 20..44 {
            for x in 8..88 {
                let i = ((y * w + x) * 3) as usize;
                buf[i] = 30.0;
                buf[i + 1] = 30.0;
                buf[i + 2] = 30.0;
            }
        }
        let (inv, w2, h2) = rotation_frame(5.0, w, h);
        let expect = g.resample(&buf, w, h, w2, h2, inv, DESK_FILL);
        let (rows, meta, _, _, nw, nh) = rotscale_frame(5.0, w, h, 1.0, 1.0);
        assert_eq!((nw, nh), (w2, h2));
        let got = g.affine(&buf, w, h, nw, nh, rows, meta);
        assert_eq!(got.len(), expect.len());
        let d = got
            .iter()
            .zip(expect.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let m = got
            .iter()
            .zip(expect.iter())
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / got.len() as f64;
        assert!(d <= 2.0, "affine rotation-only max drift {d}");
        assert!(m < 1e-3, "affine rotation-only mean drift {m}");
        // The test-only CPU mirror agrees with the GPU kernel closely.
        let mirror = cpu_affine_mirror(&buf, w, h, nw, nh, rows, meta, DESK_FILL);
        let dm = got
            .iter()
            .zip(mirror.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(dm <= 2.0, "affine vs CPU mirror max drift {dm}");
    }

    #[test]
    fn noise_seed32_folds_both_halves() {
        assert_eq!(noise_seed32(0), 0);
        // High half alone still changes the stream id.
        assert_ne!(noise_seed32(1u64 << 32), 0);
        assert_ne!(noise_seed32(0x9E3779B97F4A7C15), noise_seed32(0x9E3779B9));
    }

    #[test]
    fn grain_seed32_differs_from_noise_stream() {
        // Same file must feed different streams into grain and noise, or
        // their per-pixel draws would correlate.
        for seed in [0u64, 1, 4242, 0x9E3779B97F4A7C15, u64::MAX] {
            assert_ne!(grain_seed32(seed), noise_seed32(seed), "seed {seed}");
        }
        assert_eq!(grain_seed32(7), grain_seed32(7), "deterministic");
    }

    #[test]
    fn gpu_grain_is_deterministic_and_zero_is_identity() {
        let g = match Gpu::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        let (w, h) = (64u32, 48u32);
        let src = vec![200.0f32; (w * h * 3) as usize];
        let a = g.grain(&src, w, h, 2.5, 99, 0.001, 0.6, [4.0, 2.0, -1.0]);
        let b = g.grain(&src, w, h, 2.5, 99, 0.001, 0.6, [4.0, 2.0, -1.0]);
        assert_eq!(a, b, "grain must be deterministic");
        assert!(a.iter().all(|&v| (0.0..=255.0).contains(&v)));
        assert_ne!(a, src, "nonzero grain must change pixels");
        // Zero knobs are identity.
        let id = g.grain(&src, w, h, 0.0, 99, 0.0, 0.0, [0.0, 0.0, 0.0]);
        let drift: f64 =
            id.iter().zip(src.iter()).map(|(a, b)| (a - b).abs() as f64).sum::<f64>()
                / id.len() as f64;
        assert!(drift < 1e-4, "zero grain drift {drift}");
    }

    #[test]
    fn gpu_blur_is_deterministic_and_zero_is_identity() {
        let g = match Gpu::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        let (w, h) = (64u32, 48u32);
        let mut src = vec![0.0f32; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                src[i] = x as f32 * 3.0;
                src[i + 1] = y as f32 * 4.0;
                src[i + 2] = 128.0;
            }
        }
        let a = g.blur(&src, w, h, 1.5);
        let b = g.blur(&src, w, h, 1.5);
        assert_eq!(a, b, "blur must be deterministic");
        let id = g.blur(&src, w, h, 0.0);
        assert_eq!(id, src, "zero blur must be identity");
    }

    #[test]
    fn gpu_bend_is_deterministic_and_matches_modes() {
        let g = match Gpu::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        let (w, h) = (64u32, 48u32);
        let mut src = vec![0.0f32; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                src[i] = x as f32 * 3.0;
                src[i + 1] = y as f32 * 4.0;
                src[i + 2] = 128.0;
            }
        }
        // One call per mode; each must be deterministic and move pixels.
        for (mode, edge) in [(0u32, 0u32), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3)] {
            let a = g.bend(&src, w, h, mode, edge, 12.0, 0.25, 32.0, 4.0, 1.5, 1.2, 0.7);
            let b = g.bend(&src, w, h, mode, edge, 12.0, 0.25, 32.0, 4.0, 1.5, 1.2, 0.7);
            assert_eq!(a, b, "bend must be deterministic (mode {mode} edge {edge})");
            assert_ne!(a, src, "bend must move pixels (mode {mode} edge {edge})");
        }
    }

    /// Full-pipeline equivalence probe: same input PNG through the CPU and
    /// GPU backends. Stages that share exact semantics (warp, rotation,
    /// lighting, CA) must agree closely; noise uses an independent stream
    /// so only distribution-level agreement is asserted there. Skips
    /// cleanly on machines without a GPU adapter.
    #[test]
    fn backend_parity_on_synthetic_page() {
        let g = match Gpu::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        // Tiny synthetic "page": white with a dark staff-like band block.
        let (w, h) = (96u32, 64u32);
        let mut buf = vec![240.0f32; (w * h * 3) as usize];
        for y in 20..44 {
            for x in 8..88 {
                let i = ((y * w + x) * 3) as usize;
                buf[i] = 30.0;
                buf[i + 1] = 30.0;
                buf[i + 2] = 30.0;
            }
        }
        // Identity warp must round-trip through the GPU resampler.
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let rt = g.resample(&buf, w, h, w, h, id, PAPER_FILL);
        let mean: f64 =
            rt.iter().zip(buf.iter()).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / rt.len() as f64;
        assert!(mean < 1e-4, "identity roundtrip drift {mean}");
        // Zero-knob lighting is identity; chromatic at 0 is identity.
        let lit = g.lighting(&buf, w, h, [0.0, 0.0], 10.0, 0.0, 0.0, 0.0, 0.0);
        let mean: f64 =
            lit.iter().zip(buf.iter()).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / lit.len() as f64;
        assert!(mean < 1e-4, "zero lighting drift {mean}");
        let ca = g.chromatic(&buf, w, h, 0.0);
        let mean: f64 =
            ca.iter().zip(buf.iter()).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / ca.len() as f64;
        assert!(mean < 1e-4, "zero-CA drift {mean}");
        // Noise at sigma 8 on mid-grey: mean ~128 not required here (buf is
        // 240/30 mix); assert determinism + clamping instead.
        let n1 = g.noise(&buf, w, h, 8.0, 1234);
        let n2 = g.noise(&buf, w, h, 8.0, 1234);
        assert_eq!(n1, n2, "GPU noise must be deterministic");
        assert!(n1.iter().all(|&v| (0.0..=255.0).contains(&v)));
    }
}
