//! hand-ink: clean PNG -> handwritten-ink-variant PNG.
//!
//! Rust reimplementation of the `v3_high` deterministic hand-ink transform
//! from `tmp-train/hw-synth/v3/v3_levels.py` (level `"v3_high"`):
//! a seeded smooth displacement field (max 5 px, field sigma 10 px) wobbles
//! staff lines and stems like hand drawing, then per-region ink-density
//! jitter plus warm paper grain. Nearest-neighbor remap, topology-preserving
//! by construction: nothing is deleted, every dark pixel survives moved by at
//! most `max_disp` px, so ground-truth labels still apply.
//!
//! Also supports the sibling levels `med` (3 px / sigma 14 / ink 0.10 /
//! grain 1.6) and `scribbly` (8 px / sigma 8 / ink 0.20 / grain 3.0) from the
//! same reference script. Default level is `high` = `v3_high`.
//!
//! Determinism: every byte derives from `(seed, file index)`; file `i` of a
//! sorted batch uses `seed + i * 0x9E3779B97F4A7C15` (same stride convention
//! as photo-degrade), and all noise inside one file comes from a SplitMix64
//! stream seeded with that per-file seed. Same seed + inputs => bit-identical
//! outputs, regardless of thread count (output pixels are independent).
//!
//! Usage:
//!   hand-ink --input clean.png --out-dir out --seed 1234 --level high
//!   hand-ink --input clean-dir/ --out-dir out --seed 7 --resume

use clap::{Parser, ValueEnum};
use image::{Rgb, RgbImage};
use std::path::{Path, PathBuf};

/// Paper tint applied around mid-grey (warm paper), from the v3 reference.
const TINT: [f32; 3] = [1.0, 0.986, 0.945];
/// Smoothing scale of the ink-density field, px (INK_SIGMA in v3).
const INK_SIGMA: f32 = 40.0;
/// Channels darker than this get the raw gain; lighter ones get max(gain, 1).
const INK_THRESHOLD: f32 = 100.0;
/// Per-file seed stride (matches photo-degrade's batch convention).
const SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Hand-ink strength level (v3_levels.py LEVELS table).
#[derive(Copy, Clone, Debug, Default, ValueEnum, PartialEq, Eq)]
enum Level {
    /// 3 px displacement, field sigma 14, ink 0.10, grain 1.6.
    Med,
    /// 5 px displacement, field sigma 10, ink 0.14, grain 2.2 (v3_high).
    #[default]
    High,
    /// 8 px displacement, field sigma 8, ink 0.20, grain 3.0.
    Scribbly,
}

impl Level {
    /// (max_disp_px, field_sigma_px, ink_amp, ink_lo, grain_sigma).
    fn params(self) -> (f32, f32, f32, f32, f32) {
        match self {
            Level::Med => (3.0, 14.0, 0.10, 0.95, 1.6),
            Level::High => (5.0, 10.0, 0.14, 0.93, 2.2),
            Level::Scribbly => (8.0, 8.0, 0.20, 0.90, 3.0),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Level::Med => "v3_med",
            Level::High => "v3_high",
            Level::Scribbly => "v3_scribbly",
        }
    }
}

/// Turn clean sheet-music PNGs into handwritten-ink variants.
#[derive(Parser, Debug)]
#[command(name = "hand-ink", version)]
struct Args {
    /// Input clean PNG file, or a directory of PNGs (non-recursive, sorted).
    #[arg(long)]
    input: PathBuf,

    /// Directory that receives `<stem>{suffix}.png` + `<stem>{suffix}_params.json`.
    #[arg(long, default_value = "out")]
    out_dir: PathBuf,

    /// Master seed (matches the v3 reference SEED=1234 by default).
    /// File `i` of a batch uses `seed + i * SEED_STRIDE`.
    #[arg(long, default_value_t = 1234)]
    seed: u64,

    /// Hand-ink strength level.
    #[arg(long, value_enum, default_value_t = Level::High)]
    level: Level,

    /// Filename suffix for hand-inked images (before the extension).
    #[arg(long, default_value = "_hand")]
    suffix: String,

    /// Skip files whose output PNG and params JSON already exist.
    #[arg(long, default_value_t = false)]
    resume: bool,

    /// Exit(2) after N files (test aid for --resume).
    #[arg(long)]
    fail_after: Option<usize>,

    /// Process only this share of inputs (0,1]: every Nth file of the
    /// sorted batch with N = round(1/fraction) (0.2 = every 5th).
    /// Unselected files are left clean (no outputs). Default 1.0 = all.
    #[arg(long, default_value_t = 1.0)]
    fraction: f64,
}

// ---------------------------------------------------------------------------
// Deterministic RNG (SplitMix64, same construction as pipeline-orchestrator)
// ---------------------------------------------------------------------------

struct StreamRng(u64);

impl StreamRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_unit(&mut self) -> f64 {
        // 53-bit precision double in [0, 1).
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }
    /// Standard-normal sample via Box-Muller (one spare discarded; the
    /// stream stays deterministic either way).
    fn next_gauss(&mut self) -> f32 {
        let u1 = self.next_unit().max(f64::MIN_POSITIVE);
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        (r * theta.cos()) as f32
    }
}

// ---------------------------------------------------------------------------
// Gaussian blur (separable, clamp-to-edge, truncate at 4 sigma)
// ---------------------------------------------------------------------------

fn gauss_kernel(sigma: f32) -> Vec<f32> {
    let radius = (4.0 * sigma).ceil().max(1.0) as usize;
    let mut k = Vec::with_capacity(2 * radius + 1);
    let mut sum = 0.0f32;
    for i in -(radius as i32)..=(radius as i32) {
        let v = (-0.5 * (i as f32 / sigma).powi(2)).exp();
        k.push(v);
        sum += v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// Blur one row-major `w x h` field. Pure function of its input.
fn gauss_blur(field: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let k = gauss_kernel(sigma);
    let r = k.len() / 2;
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in k.iter().enumerate() {
                let sx = (x as i64 + ki as i64 - r as i64).clamp(0, w as i64 - 1) as usize;
                acc += field[y * w + sx] * kv;
            }
            tmp[y * w + x] = acc;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in k.iter().enumerate() {
                let sy = (y as i64 + ki as i64 - r as i64).clamp(0, h as i64 - 1) as usize;
                acc += tmp[sy * w + x] * kv;
            }
            out[y * w + x] = acc;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Core transform
// ---------------------------------------------------------------------------

struct HandInk {
    max_disp: f32,
    field_sigma: f32,
    ink_amp: f32,
    ink_lo: f32,
    grain_sigma: f32,
}

impl HandInk {
    fn for_level(level: Level) -> Self {
        let (disp, fsig, amp, lo, gs) = level.params();
        Self { max_disp: disp, field_sigma: fsig, ink_amp: amp, ink_lo: lo, grain_sigma: gs }
    }

    /// Apply the transform. Returns `(output image, dark_count_ratio)`.
    fn apply(&self, src: &RgbImage, file_seed: u64) -> (RgbImage, f64) {
        let (w, h) = (src.width() as usize, src.height() as usize);
        let n = w * h;
        let mut rng = StreamRng::new(file_seed);

        // 1. Raw white-noise fields, then smooth to field_sigma.
        let mut nx = vec![0.0f32; n];
        let mut ny = vec![0.0f32; n];
        for i in 0..n {
            nx[i] = rng.next_gauss();
            ny[i] = rng.next_gauss();
        }
        let mut dx = gauss_blur(&nx, w, h, self.field_sigma);
        let mut dy = gauss_blur(&ny, w, h, self.field_sigma);
        // 2. Normalize so the peak absolute displacement equals max_disp.
        let mut peak = 0.0f32;
        for i in 0..n {
            peak = peak.max(dx[i].abs()).max(dy[i].abs());
        }
        if peak > 0.0 {
            let s = self.max_disp / peak;
            for i in 0..n {
                dx[i] *= s;
                dy[i] *= s;
            }
        }

        // 3. Nearest-neighbor remap (sample coords clipped to the frame).
        let mut warped = vec![0u8; n * 3];
        for y in 0..h {
            for x in 0..w {
                let mx = (x as f32 + dx[y * w + x]).round().clamp(0.0, w as f32 - 1.0) as usize;
                let my = (y as f32 + dy[y * w + x]).round().clamp(0.0, h as f32 - 1.0) as usize;
                let p = src.get_pixel(mx as u32, my as u32);
                let o = (y * w + x) * 3;
                warped[o] = p[0];
                warped[o + 1] = p[1];
                warped[o + 2] = p[2];
            }
        }

        // 4. Ink-density field (sigma 40) -> per-pixel gain.
        let mut ns = vec![0.0f32; n];
        for v in &mut ns {
            *v = rng.next_gauss();
        }
        let smooth = gauss_blur(&ns, w, h, INK_SIGMA);
        let ink_hi = 1.0 + self.ink_amp * 1.7;
        let mut gain = vec![0.0f32; n];
        for i in 0..n {
            gain[i] = (1.0 + self.ink_amp * smooth[i]).clamp(self.ink_lo, ink_hi);
        }

        // 5. Grain field (shared across channels, like the reference).
        let mut ng = vec![0.0f32; n];
        for v in &mut ng {
            *v = rng.next_gauss();
        }

        // 6. Ink + warm tint + grain.
        let mut out = RgbImage::new(src.width(), src.height());
        let mut out_dark = 0u64;
        let mut in_dark = 0u64;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let g = gain[i];
                let gr = ng[i] * self.grain_sigma;
                let mut px = Rgb([0u8; 3]);
                for c in 0..3 {
                    let chan = warped[i * 3 + c] as f32;
                    // PIL "L" luma of the source pixel, for the dark count.
                    let gc = if chan < INK_THRESHOLD { g } else { g.max(1.0) };
                    let ink = 255.0 - (255.0 - chan) * gc;
                    let v = ((ink - 128.0) * TINT[c] + 128.0 + gr).round().clamp(0.0, 255.0);
                    px[c] = v as u8;
                }
                out.put_pixel(x as u32, y as u32, px);
                let sp = src.get_pixel(x as u32, y as u32);
                let luma_in = 0.299 * sp[0] as f64 + 0.587 * sp[1] as f64 + 0.114 * sp[2] as f64;
                let luma_out =
                    0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64;
                if luma_in < 128.0 {
                    in_dark += 1;
                }
                if luma_out < 128.0 {
                    out_dark += 1;
                }
            }
        }
        let ratio = out_dark as f64 / in_dark.max(1) as f64;
        (out, ratio)
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn collect_inputs(input: &Path) -> Result<Vec<PathBuf>, String> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    if !input.is_dir() {
        return Err(format!("--input not found: {}", input.display()));
    }
    let mut paths: Vec<PathBuf> = Vec::new();
    let entries =
        std::fs::read_dir(input).map_err(|e| format!("cannot list {}: {e}", input.display()))?;
    for e in entries {
        let e = e.map_err(|e| format!("cannot list {}: {e}", input.display()))?;
        let p = e.path();
        if p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("png") {
            paths.push(p);
        }
    }
    paths.sort();
    Ok(paths)
}

fn write_artifact_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    // Write-then-rename so a killed run never leaves a half-written file
    // that a later --resume would mistake for complete.
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("cannot rename to {}: {e}", path.display()))?;
    Ok(())
}

fn r4(v: f32) -> f64 {
    ((v as f64) * 10000.0).round() / 10000.0
}

fn params_json(
    input_name: &str,
    level: Level,
    ink: &HandInk,
    seed: u64,
    file_seed: u64,
    dark_count_ratio: f64,
) -> String {
    let v = serde_json::json!({
        "route": "deterministic hand-ink wobble (no generative model); Rust port of tmp-train/hw-synth/v3 level v3_high",
        "input": input_name,
        "seed": seed,
        "file_seed": file_seed,
        "level": level.name(),
        "field_sigma_px": r4(ink.field_sigma),
        "max_abs_displacement_px": r4(ink.max_disp),
        "ink_amplitude": r4(ink.ink_amp),
        "ink_gain_lo": r4(ink.ink_lo),
        "grain_sigma": r4(ink.grain_sigma),
        "dark_count_ratio": (dark_count_ratio * 10000.0).round() / 10000.0,
        "label_note": format!(
            "topology-preserving: dark-count ratio {:.4} (nothing deleted); wobble <= {:.1}px",
            dark_count_ratio, ink.max_disp),
    });
    serde_json::to_string_pretty(&v).unwrap_or_else(|_| "{}".to_string())
}

fn hand_file(
    path: &Path,
    file_seed: u64,
    ink: &HandInk,
    level: Level,
    seed: u64,
    out_path: &Path,
    params_path: &Path,
    resume: bool,
) -> Result<f64, String> {
    if resume && out_path.is_file() && params_path.is_file() {
        return Ok(f64::NAN); // skipped; nothing to record
    }
    let img = image::open(path)
        .map_err(|e| format!("cannot decode {}: {e}", path.display()))?
        .to_rgb8();
    let (out, ratio) = ink.apply(&img, file_seed);
    let mut png = Vec::new();
    let enc = image::codecs::png::PngEncoder::new(&mut png);
    use image::ImageEncoder as _;
    enc.write_image(out.as_raw(), out.width(), out.height(), image::ExtendedColorType::Rgb8)
        .map_err(|e| format!("PNG encode failed for {}: {e}", out_path.display()))?;
    write_artifact_bytes(out_path, &png)?;
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
    let params = params_json(name, level, ink, seed, file_seed, ratio);
    write_artifact_bytes(params_path, params.as_bytes())?;
    Ok(ratio)
}

/// Deterministic stride selection for `--fraction`: with `f`, process every
/// Nth file of the sorted batch (N = max(1, round(1/f))). Position-based,
/// so a file's fate — and its positional `file_seed` — never depends on
/// siblings, on resume state, or on the fraction used by other runs.
fn selected(index: usize, fraction: f64) -> bool {
    if fraction >= 1.0 {
        return true;
    }
    let denom = (1.0 / fraction).round().max(1.0) as usize;
    index % denom == 0
}

fn run(args: &Args) -> Result<i32, String> {
    if !(args.fraction > 0.0 && args.fraction <= 1.0) {
        return Err(format!(
            "--fraction must be in (0, 1], got {}",
            args.fraction
        ));
    }
    let inputs = collect_inputs(&args.input)?;
    if inputs.is_empty() {
        return Err(format!("no PNG files found at {}", args.input.display()));
    }
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|e| format!("cannot create {}: {e}", args.out_dir.display()))?;
    let ink = HandInk::for_level(args.level);
    let mut done = 0usize;
    let mut skipped = 0usize;
    for (i, path) in inputs.iter().enumerate() {
        if !selected(i, args.fraction) {
            skipped += 1;
            continue;
        }
        let file_seed = args.seed.wrapping_add((i as u64).wrapping_mul(SEED_STRIDE));
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("bad file name: {}", path.display()))?;
        let out_path = args.out_dir.join(format!("{stem}{}.png", args.suffix));
        let params_path = args.out_dir.join(format!("{stem}{}_params.json", args.suffix));
        let ratio = hand_file(path, file_seed, &ink, args.level, args.seed, &out_path, &params_path, args.resume)?;
        if ratio.is_nan() {
            skipped += 1;
        } else {
            done += 1;
            println!("{} -> {} (dark-count ratio {:.4})", path.display(), out_path.display(), ratio);
        }
        if let Some(limit) = args.fail_after {
            if done >= limit {
                eprintln!("--fail-after {limit} reached; rerun with --resume");
                return Ok(2);
            }
        }
    }
    println!("done: {done} file(s) inked, {skipped} skipped, fraction={}, seed={}, out={}", args.fraction, args.seed, args.out_dir.display());
    Ok(0)
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(code) => std::process::ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("hand-ink: error: {e}");
            std::process::ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small synthetic "score": white page, staff lines, note blobs.
    fn synthetic_score(w: u32, h: u32) -> RgbImage {
        let mut img = RgbImage::from_pixel(w, h, Rgb([255u8, 255, 255]));
        for s in 0..3 {
            let base = 10 + s * (h / 3).max(16);
            for l in 0..5 {
                let y = base + l * 3;
                if y >= h {
                    continue;
                }
                for x in 4..w.saturating_sub(4) {
                    img.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }
        // Note heads + stems (positions scale with image size).
        let notes = [(w / 5, h / 6), (w * 2 / 5, h * 2 / 5), (w * 3 / 5, h * 3 / 5), (w * 4 / 5, h / 3)];
        for (cx, cy) in notes {
            for dy in -3i32..=3 {
                for dx in -2i32..=2 {
                    let (px, py) = (cx as i32 + dx, cy as i32 + dy);
                    if px >= 0 && py >= 0 && px < w as i32 && py < h as i32 {
                        img.put_pixel(px as u32, py as u32, Rgb([10, 10, 10]));
                    }
                }
            }
            for i in 0..14 {
                let sy = cy as i32 - i;
                if cx + 3 < w && sy >= 0 {
                    img.put_pixel(cx + 3, sy as u32, Rgb([10, 10, 10]));
                }
            }
        }
        img
    }

    fn png_bytes(img: &RgbImage) -> Vec<u8> {
        use image::ImageEncoder as _;
        let mut buf = Vec::new();
        let enc = image::codecs::png::PngEncoder::new(&mut buf);
        enc.write_image(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgb8)
            .unwrap();
        buf
    }

    #[test]
    fn levels_match_v3_reference_params() {
        // Must track tmp-train/hw-synth/v3/v3_levels.py LEVELS exactly.
        assert_eq!(HandInk::for_level(Level::Med).max_disp, 3.0);
        let hi = HandInk::for_level(Level::High);
        assert_eq!((hi.max_disp, hi.field_sigma, hi.ink_amp, hi.ink_lo, hi.grain_sigma),
                   (5.0, 10.0, 0.14, 0.93, 2.2));
        let sc = HandInk::for_level(Level::Scribbly);
        assert_eq!((sc.max_disp, sc.field_sigma), (8.0, 8.0));
        assert_eq!(INK_SIGMA, 40.0);
    }

    #[test]
    fn same_seed_bit_identical() {
        let src = synthetic_score(160, 96);
        let ink = HandInk::for_level(Level::High);
        let (a, ra) = ink.apply(&src, 1234);
        let (b, rb) = ink.apply(&src, 1234);
        assert_eq!(png_bytes(&a), png_bytes(&b));
        assert_eq!(ra, rb);
    }

    #[test]
    fn different_seeds_differ() {
        let src = synthetic_score(160, 96);
        let ink = HandInk::for_level(Level::High);
        let (a, _) = ink.apply(&src, 1234);
        let (b, _) = ink.apply(&src, 1235);
        assert_ne!(png_bytes(&a), png_bytes(&b));
    }

    #[test]
    fn label_preservation_dark_count_within_1pct() {
        // Proxy for "labels still apply": nothing deleted, only moved <= 5px.
        let src = synthetic_score(160, 96);
        for level in [Level::Med, Level::High, Level::Scribbly] {
            let ink = HandInk::for_level(level);
            let (_, ratio) = ink.apply(&src, 1234);
            assert!(
                (0.99..=1.01).contains(&ratio),
                "{level:?}: dark-count ratio {ratio:.4} outside 1% band"
            );
        }
    }

    #[test]
    fn fraction_selects_deterministic_subset() {
        let dir = std::env::temp_dir().join(format!("hand-ink-frac-{}", std::process::id()));
        let in_dir = dir.join("in");
        let out_dir = dir.join("out");
        std::fs::create_dir_all(&in_dir).unwrap();
        let src = synthetic_score(64, 48);
        for name in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"] {
            std::fs::write(in_dir.join(format!("{name}.png")), png_bytes(&src)).unwrap();
        }
        let base = Args {
            input: in_dir.clone(),
            out_dir: out_dir.clone(),
            seed: 7,
            level: Level::High,
            suffix: "_hand".to_string(),
            resume: false,
            fail_after: None,
            fraction: 0.2,
        };
        assert_eq!(run(&base).unwrap(), 0);
        // Every 5th of the sorted batch: indices 0 (a) and 5 (f) only.
        assert!(out_dir.join("a_hand.png").is_file());
        assert!(out_dir.join("f_hand.png").is_file());
        for name in ["b", "c", "d", "e", "g", "h", "i", "j"] {
            assert!(!out_dir.join(format!("{name}_hand.png")).exists(), "{name} must stay clean");
        }
        let before = std::fs::read(out_dir.join("a_hand.png")).unwrap();
        // Same fraction reruns bit-identical (positional seeds, not renumbered).
        assert_eq!(run(&base).unwrap(), 0);
        assert_eq!(std::fs::read(out_dir.join("a_hand.png")).unwrap(), before);
        // Full fraction restores everything.
        let full = Args { fraction: 1.0, ..base };
        assert_eq!(run(&full).unwrap(), 0);
        for name in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"] {
            assert!(out_dir.join(format!("{name}_hand.png")).is_file());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fraction_rejects_out_of_range() {
        let dir = std::env::temp_dir().join(format!("hand-ink-fracbad-{}", std::process::id()));
        let in_dir = dir.join("in");
        std::fs::create_dir_all(&in_dir).unwrap();
        let src = synthetic_score(64, 48);
        std::fs::write(in_dir.join("t.png"), png_bytes(&src)).unwrap();
        for bad in [0.0, -0.5, 1.5, f64::NAN] {
            let args = Args {
                input: in_dir.clone(),
                out_dir: dir.join("out"),
                seed: 7,
                level: Level::High,
                suffix: "_hand".to_string(),
                resume: false,
                fail_after: None,
                fraction: bad,
            };
            assert!(run(&args).is_err(), "fraction {bad} must be rejected");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resume_skips_completed_files() {
        let dir = std::env::temp_dir().join(format!("hand-ink-test-{}", std::process::id()));
        let in_dir = dir.join("in");
        let out_dir = dir.join("out");
        std::fs::create_dir_all(&in_dir).unwrap();
        let src = synthetic_score(64, 48);
        std::fs::write(in_dir.join("t.png"), png_bytes(&src)).unwrap();
        let args = Args {
            input: in_dir.clone(),
            out_dir: out_dir.clone(),
            seed: 7,
            level: Level::High,
            suffix: "_hand".to_string(),
            resume: false,
            fail_after: None,
            fraction: 1.0,
        };
        assert_eq!(run(&args).unwrap(), 0);
        let before = std::fs::read(out_dir.join("t_hand.png")).unwrap();
        // Second run without --resume rewrites identical bytes.
        assert_eq!(run(&args).unwrap(), 0);
        assert_eq!(std::fs::read(out_dir.join("t_hand.png")).unwrap(), before);
        // --resume skips the completed file.
        let resumed = Args { resume: true, ..args };
        assert_eq!(run(&resumed).unwrap(), 0);
        assert_eq!(std::fs::read(out_dir.join("t_hand.png")).unwrap(), before);
        assert!(out_dir.join("t_hand_params.json").is_file());
        std::fs::remove_dir_all(&dir).ok();
    }
}
