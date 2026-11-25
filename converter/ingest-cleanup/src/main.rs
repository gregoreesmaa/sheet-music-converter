//! `ingest-cleanup`: first converter stage for sheet-music photos.
//!
//! Takes one photographed (or scanned) sheet-music image and produces a
//! deskewed, binary (black-on-white) image ready for staff detection:
//!
//! 1. **EXIF normalize** — apply the EXIF orientation flag so the pixels are
//!    upright (the `image` crate decodes the raw stored pixels).
//! 2. **Quality gates** — reject (non-zero exit, no output) images that are
//!    too blurry (variance of Laplacian below `--blur-threshold`) or have too
//!    much specular glare (fraction of pixels at/above `--glare-level` above
//!    `--glare-threshold`).
//! 3. **Deskew** — estimate the skew angle by maximizing the variance of the
//!    horizontal projection profile over a candidate range, then rotate back
//!    (white background fill).
//! 4. **Binarize** — Otsu threshold (or `--threshold`) to pure black/white.

use clap::Parser;
use image::{DynamicImage, GrayImage, Luma};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Phone-photo sheet music -> deskewed binary image, with quality gates.
#[derive(Parser, Debug)]
#[command(
    name = "ingest-cleanup",
    version,
    about = "Deskew + binarize a sheet-music photo; reject blurry / glare-blown images"
)]
struct Args {
    /// Input image (PNG/JPEG/anything the `image` crate decodes).
    #[arg(long)]
    input: PathBuf,

    /// Output binary image path (format guessed from the extension).
    #[arg(long)]
    output: PathBuf,

    /// Reject when the Laplacian variance (sharpness) is below this.
    /// Re-trained on 398 fresh big500 photos (min observed 6.54): 6 passes
    /// them all while still rejecting blank pages (variance exactly 0).
    #[arg(long, default_value_t = 6.0)]
    blur_threshold: f64,

    /// Reject when the fraction of glare pixels (>= --glare-level) exceeds this.
    /// Re-trained on the same corpus (max observed hot fraction 0.1069).
    #[arg(long, default_value_t = 0.12)]
    glare_threshold: f64,

    /// Luma value treated as blown-out highlight / glare.
    #[arg(long, default_value_t = 250)]
    glare_level: u8,

    /// Deskew search range, +/- degrees.
    #[arg(long, default_value_t = 10.0)]
    max_skew_deg: f32,

    /// Deskew search step in degrees.
    #[arg(long, default_value_t = 0.5)]
    skew_step_deg: f32,

    /// Fixed binarization threshold 0..=255 (default: Otsu).
    #[arg(long)]
    threshold: Option<u8>,

    /// Skip deskew, only binarize.
    #[arg(long, default_value_t = false)]
    no_deskew: bool,

    /// Write output even when a quality gate fails (still reported as REJECT).
    #[arg(long, default_value_t = false)]
    force: bool,
}

#[derive(Debug)]
struct GateFailure {
    code: u8,
    message: String,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ingest-cleanup: error: {}", e.message);
            ExitCode::from(e.code)
        }
    }
}

fn run(args: &Args) -> Result<(), GateFailure> {
    let fail = |code: u8, message: String| GateFailure { code, message };

    if !args.input.is_file() {
        return Err(fail(
            1,
            format!("input not found: {}", args.input.display()),
        ));
    }
    if !(0.0..=1.0).contains(&args.glare_threshold) {
        return Err(fail(1, "--glare-threshold must be in [0, 1]".into()));
    }
    if args.max_skew_deg < 0.0 || args.skew_step_deg <= 0.0 {
        return Err(fail(
            1,
            "--max-skew-deg must be >= 0 and --skew-step-deg > 0".into(),
        ));
    }

    // 1. Load + EXIF normalize.
    let decoded = image::open(&args.input)
        .map_err(|e| fail(1, format!("cannot decode {}: {e}", args.input.display())))?;
    let orientation = read_exif_orientation(&args.input).unwrap_or(1);
    let gray = apply_exif_orientation(decoded, orientation).to_luma8();
    let (w, h) = (gray.width(), gray.height());

    // 2. Quality gates (measured on the normalized grayscale image).
    let blur = laplacian_variance(&gray);
    let blur_ok = blur >= args.blur_threshold;
    let glare = assess_glare(&gray, args.glare_level, args.glare_threshold);

    println!("input: {}", args.input.display());
    println!("dimensions: {w}x{h} (exif orientation {orientation})");
    println!(
        "blur: laplacian variance {blur:.2} (threshold {}) -> {}",
        args.blur_threshold,
        verdict(blur_ok)
    );
    println!(
        "glare: hot fraction {:.4} at level >= {} (page median {:.1}, ink {:.4}, threshold {}) -> {} ({})",
        glare.hot_fraction,
        args.glare_level,
        glare.page_median,
        glare.ink_fraction,
        args.glare_threshold,
        verdict(glare.ok),
        glare.reason
    );

    if !blur_ok && !args.force {
        return Err(fail(
            2,
            format!(
                "blur reject: variance {blur:.2} < threshold {}",
                args.blur_threshold
            ),
        ));
    }
    if !glare.ok && !args.force {
        return Err(fail(
            3,
            format!(
                "glare reject: {} (hot {:.4} > threshold {})",
                glare.reason, glare.hot_fraction, args.glare_threshold
            ),
        ));
    }

    // 3. Deskew.
    let mut deskewed = gray;
    let mut angle = 0.0f32;
    if !args.no_deskew && args.max_skew_deg > 0.0 {
        angle = estimate_skew_angle(&deskewed, args.max_skew_deg, args.skew_step_deg);
        println!("deskew: correction angle {angle:+.2} deg");
        if angle != 0.0 {
            deskewed = rotate_white(&deskewed, angle);
        }
    } else {
        println!("deskew: skipped");
    }

    // 4. Binarize.
    let t = args.threshold.unwrap_or_else(|| otsu_threshold(&deskewed));
    println!("binarize: threshold {t}");
    let binary = binarize(&deskewed, t);

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                fail(
                    1,
                    format!("cannot create output dir {}: {e}", parent.display()),
                )
            })?;
        }
    }
    binary
        .save(&args.output)
        .map_err(|e| fail(1, format!("cannot write {}: {e}", args.output.display())))?;
    println!(
        "output: {} ({}x{}, angle {angle:+.2} deg, threshold {t})",
        args.output.display(),
        binary.width(),
        binary.height()
    );

    if (!blur_ok || !glare.ok) && args.force {
        eprintln!("ingest-cleanup: warning: quality gate failed but --force kept the output");
    }
    Ok(())
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "REJECT"
    }
}

/// EXIF orientation flag (1-8), or `None` when the file has no (readable) EXIF.
/// Never fails the run: missing EXIF just means "already upright".
fn read_exif_orientation(path: &Path) -> Option<u32> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let exif = exif::Reader::new().read_from_container(&mut reader).ok()?;
    let field = exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?;
    field.value.get_uint(0)
}

/// Apply an EXIF orientation value to raw decoded pixels.
fn apply_exif_orientation(img: DynamicImage, orientation: u32) -> DynamicImage {
    match orientation {
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => img.rotate90().fliph(), // transpose
        6 => img.rotate90(),          // 90 deg CW
        7 => img.rotate270().fliph(), // transverse
        8 => img.rotate270(),         // 90 deg CCW
        _ => img,                    // 1 (or unknown): already upright
    }
}

/// Sharpness estimate: variance of the Laplacian response.
/// Sharp edges give large responses; blur drives the variance toward zero.
fn laplacian_variance(gray: &GrayImage) -> f64 {
    let (w, h) = gray.dimensions();
    if w < 3 || h < 3 {
        return 0.0;
    }
    let raw = gray.as_raw();
    let wu = w as usize;
    let hu = h as usize;
    // Exact integer sum / sum-of-squares over the 4-neighbour Laplacian
    // (|lap| <= 1020, lap^2 <= 1_040_400; fits i64 for images up to >8T px).
    let mut sum = 0i64;
    let mut sum_sq = 0i64;
    for y in 1..hu - 1 {
        let row = y * wu;
        let prev = row - wu;
        let next = row + wu;
        for x in 1..wu - 1 {
            let lap = raw[row + x - 1] as i64
                + raw[row + x + 1] as i64
                + raw[prev + x] as i64
                + raw[next + x] as i64
                - 4 * raw[row + x] as i64;
            sum += lap;
            sum_sq += lap * lap;
        }
    }
    let n = ((wu - 2) as u64) * ((hu - 2) as u64);
    if n > 1 {
        let nf = n as f64;
        let mean = sum as f64 / nf;
        ((sum_sq as f64 - nf * mean * mean) / (nf - 1.0)).max(0.0)
    } else {
        0.0
    }
}

/// 256-bin grayscale histogram in one linear pass over raw bytes.
fn luma_histogram(gray: &GrayImage) -> [u64; 256] {
    let mut hist = [0u64; 256];
    for &v in gray.as_raw() {
        hist[v as usize] += 1;
    }
    hist
}

/// Fraction of pixels at/above `level`: blown-out specular highlights.
/// A clean scan has almost none; a phone photo with flash glare has blobs.
#[cfg_attr(not(test), allow(dead_code))]
fn glare_fraction(gray: &GrayImage, level: u8) -> f64 {
    let total = gray.as_raw().len() as u64;
    if total == 0 {
        return 0.0;
    }
    let hot = gray.as_raw().iter().filter(|&&v| v >= level).count() as u64;
    hot as f64 / total as f64
}

/// Median of 16x16-tile mean lumas: the page's overall exposure level,
/// robust to staff lines, margins, and small highlights.
fn tile_median_luma(gray: &GrayImage) -> f64 {
    const TILE: usize = 16;
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    if w == 0 || h == 0 {
        return 0.0;
    }
    let raw = gray.as_raw();
    let mut means = Vec::with_capacity(w.div_ceil(TILE) * h.div_ceil(TILE));
    let mut y = 0;
    while y < h {
        let y1 = (y + TILE).min(h);
        let mut x = 0;
        while x < w {
            let x1 = (x + TILE).min(w);
            let mut sum = 0u64;
            let n = ((y1 - y) * (x1 - x)) as u64;
            for yy in y..y1 {
                let row = &raw[yy * w + x..yy * w + x1];
                for &v in row {
                    sum += v as u64;
                }
            }
            means.push(sum as f64 / n as f64);
            x = x1;
        }
        y = y1;
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    means[means.len() / 2]
}

/// Outcome of the glare / overexposure gate.
struct GlareReport {
    hot_fraction: f64,
    page_median: f64,
    ink_fraction: f64,
    ok: bool,
    reason: &'static str,
}

/// Glare is a *local* blowout: hot pixels on an otherwise normally exposed
/// page. A uniformly bright page (clean scan, white paper) is not glare, so
/// the hot fraction only counts when the page median sits well below
/// saturation. A page where almost no ink survived Otsu is washed out (fully
/// overexposed or blank) and fails regardless.
fn assess_glare(gray: &GrayImage, level: u8, threshold: f64) -> GlareReport {
    const UNIFORM_MARGIN: f64 = 20.0;
    const MIN_INK: f64 = 0.002;
    let median = tile_median_luma(gray);
    let total = gray.as_raw().len() as u64;
    let (hot, ink) = if total == 0 {
        (0.0, 0.0)
    } else {
        // Single 256-bin histogram pass feeds glare_fraction, otsu, and dark_fraction.
        let hist = luma_histogram(gray);
        let hot_cnt: u64 = hist[level as usize..].iter().sum();
        let otsu_t = otsu_from_hist(&hist, total);
        let dark_cnt: u64 = hist[..=otsu_t as usize].iter().sum();
        (hot_cnt as f64 / total as f64, dark_cnt as f64 / total as f64)
    };
    let effective_hot = if median >= level as f64 - UNIFORM_MARGIN {
        0.0
    } else {
        hot
    };
    let (ok, reason) = if ink < MIN_INK {
        (false, "washed out: almost no ink survived")
    } else if effective_hot > threshold {
        (false, "specular glare")
    } else {
        (true, "ok")
    };
    GlareReport {
        hot_fraction: hot,
        page_median: median,
        ink_fraction: ink,
        ok,
        reason,
    }
}

/// Otsu's threshold from a precomputed 256-bin histogram.
fn otsu_from_hist(hist: &[u64; 256], total: u64) -> u8 {
    if total == 0 {
        return 127;
    }
    let sum_all: u64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| i as u64 * c)
        .sum();
    let mut sum_bg = 0u64;
    let mut w_bg = 0u64;
    let mut best = -1.0f64;
    let mut threshold = 127u8;
    for (i, &c) in hist.iter().enumerate() {
        w_bg += c;
        if w_bg == 0 {
            continue;
        }
        let w_fg = total - w_bg;
        if w_fg == 0 {
            break;
        }
        sum_bg += i as u64 * c;
        let m_bg = sum_bg as f64 / w_bg as f64;
        let m_fg = (sum_all - sum_bg) as f64 / w_fg as f64;
        let between = w_bg as f64 * w_fg as f64 * (m_bg - m_fg).powi(2);
        if between > best {
            best = between;
            threshold = i as u8;
        }
    }
    threshold
}

/// Otsu's threshold: the histogram split maximizing between-class variance.
fn otsu_threshold(gray: &GrayImage) -> u8 {
    let total = gray.as_raw().len() as u64;
    if total == 0 {
        return 127;
    }
    let hist = luma_histogram(gray);
    otsu_from_hist(&hist, total)
}

/// Threshold to pure black/white (paper stays white).
fn binarize(gray: &GrayImage, threshold: u8) -> GrayImage {
    let (w, h) = gray.dimensions();
    let mut out = GrayImage::new(w, h);
    for (dst, src) in out.pixels_mut().zip(gray.pixels()) {
        *dst = Luma([if src[0] > threshold { 255 } else { 0 }]);
    }
    out
}

/// Rotate by `angle_deg` (positive = page turns clockwise) with inverse
/// mapping + nearest neighbour, expanding the canvas. Unmapped pixels are
/// paper white, so no black wedges leak into the binary output.
fn rotate_white(src: &GrayImage, angle_deg: f32) -> GrayImage {
    let (w, h) = src.dimensions();
    if w == 0 || h == 0 {
        return GrayImage::new(w.max(1), h.max(1));
    }
    let theta = angle_deg.to_radians();
    let (sin, cos) = theta.sin_cos();
    let nw = (w as f32 * cos.abs() + h as f32 * sin.abs()).ceil() as u32;
    let nh = (w as f32 * sin.abs() + h as f32 * cos.abs()).ceil() as u32;
    let mut dst = GrayImage::from_pixel(nw.max(1), nh.max(1), Luma([255u8]));
    let (nw, nh) = (dst.width(), dst.height());
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let (ncx, ncy) = (nw as f32 / 2.0, nh as f32 / 2.0);
    for y in 0..nh {
        for x in 0..nw {
            let dx = x as f32 - ncx;
            let dy = y as f32 - ncy;
            let sx = cos * dx + sin * dy + cx;
            let sy = -sin * dx + cos * dy + cy;
            let (sxi, syi) = (sx.round() as i64, sy.round() as i64);
            if sxi >= 0 && syi >= 0 && (sxi as u32) < w && (syi as u32) < h {
                dst.put_pixel(x, y, *src.get_pixel(sxi as u32, syi as u32));
            }
        }
    }
    dst
}

/// Estimate the deskew correction angle: threshold a downscaled copy (max
/// 400 px) with Otsu, extract dark-pixel coordinates once, and project them
/// onto the rotated Y axis for each angle in `-max_deg..=+max_deg`. Keeps the
/// angle with the largest variance of per-row dark-pixel counts — staff and
/// text lines stack horizontally when the page is upright.
fn estimate_skew_angle(gray: &GrayImage, max_deg: f32, step_deg: f32) -> f32 {
    let (w, h) = gray.dimensions();
    if w == 0 || h == 0 {
        return 0.0;
    }
    let scale = (400.0 / w.max(h) as f32).min(1.0);
    let resized;
    let small: &GrayImage = if scale < 1.0 {
        resized = image::imageops::resize(
            gray,
            ((w as f32 * scale).ceil() as u32).max(1),
            ((h as f32 * scale).ceil() as u32).max(1),
            image::imageops::FilterType::Triangle,
        );
        &resized
    } else {
        gray
    };
    let t = otsu_threshold(small);
    let (sw, sh) = (small.width() as usize, small.height() as usize);
    let (cx, cy) = (sw as f32 * 0.5, sh as f32 * 0.5);
    let raw = small.as_raw();
    let mut dark_pts: Vec<(f32, f32)> = Vec::new();
    for y in 0..sh {
        let dy = y as f32 - cy;
        let row = &raw[y * sw..(y + 1) * sw];
        for (x, &v) in row.iter().enumerate() {
            if v <= t {
                dark_pts.push((x as f32 - cx, dy));
            }
        }
    }
    if dark_pts.is_empty() {
        return 0.0;
    }

    let mut counts: Vec<u32> = Vec::new();
    let mut best_angle = 0.0f32;
    let mut best_score = f64::NEG_INFINITY;
    let mut a = -max_deg;
    while a <= max_deg + 1e-6 {
        let (sin, cos) = a.to_radians().sin_cos();
        let nh = ((sw as f32 * sin.abs() + sh as f32 * cos.abs()).ceil() as usize).max(1);
        let ncy = nh as f32 * 0.5;
        counts.clear();
        counts.resize(nh, 0);
        for &(dx, dy) in &dark_pts {
            let y_dst = (sin * dx + cos * dy + ncy).round() as isize;
            if y_dst >= 0 && (y_dst as usize) < nh {
                counts[y_dst as usize] += 1;
            }
        }
        let score = projection_variance(&counts);
        if score > best_score {
            best_score = score;
            best_angle = a;
        }
        a += step_deg;
    }
    // Snap negligible angles to zero to avoid a needless resample.
    if best_angle.abs() < step_deg / 2.0 {
        0.0
    } else {
        best_angle
    }
}

/// Variance of per-row dark-pixel counts: high when lines are horizontal.
fn projection_variance(counts: &[u32]) -> f64 {
    let n = counts.len();
    if n <= 1 {
        return 0.0;
    }
    let mut sum = 0u64;
    let mut sum_sq = 0u64;
    for &c in counts {
        let v = c as u64;
        sum += v;
        sum_sq += v * v;
    }
    let nf = n as f64;
    let mean = sum as f64 / nf;
    ((sum_sq as f64 - nf * mean * mean) / (nf - 1.0)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fake "staff lines": horizontal black rows on white.
    fn striped(w: u32, h: u32) -> GrayImage {
        let mut img = GrayImage::from_pixel(w, h, Luma([255]));
        for y in (10..h).step_by(20) {
            for x in 0..w {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        img
    }

    #[test]
    fn otsu_splits_bimodal_histogram() {
        let mut img = GrayImage::new(10, 10);
        for (i, p) in img.pixels_mut().enumerate() {
            *p = Luma([if i < 50 { 10 } else { 240 }]);
        }
        let t = otsu_threshold(&img);
        assert!(t >= 10 && t < 240, "threshold {t} should split the modes");
    }

    #[test]
    fn sharp_image_scores_higher_blur_metric_than_blurred() {
        let sharp = striped(120, 120);
        let blurred = image::imageops::blur(&sharp, 3.0);
        assert!(
            laplacian_variance(&sharp) > laplacian_variance(&blurred),
            "sharp={} blurred={}",
            laplacian_variance(&sharp),
            laplacian_variance(&blurred)
        );
    }

    #[test]
    fn glare_fraction_counts_hot_pixels() {
        let mut img = GrayImage::from_pixel(10, 10, Luma([100]));
        for x in 0..10 {
            img.put_pixel(x, 0, Luma([255]));
        }
        assert!((glare_fraction(&img, 250) - 0.1).abs() < 1e-9);
        assert_eq!(glare_fraction(&img, 255), 0.1);
        assert_eq!(glare_fraction(&img, 101), 0.1);
    }

    #[test]
    fn deskew_recovers_known_rotation() {
        let upright = striped(200, 200);
        let skewed = rotate_white(&upright, 5.0);
        let est = estimate_skew_angle(&skewed, 10.0, 0.5);
        assert!(
            (est + 5.0).abs() <= 0.75,
            "correction {est} should undo +5 deg skew"
        );
    }

    #[test]
    fn rotation_keeps_paper_white_background() {
        let img = striped(60, 40);
        let rot = rotate_white(&img, 7.0);
        // Corners of the expanded canvas must be fill, not black wedges.
        assert_eq!(rot.get_pixel(0, 0)[0], 255);
        assert_eq!(rot.get_pixel(rot.width() - 1, rot.height() - 1)[0], 255);
    }

    #[test]
    fn exif_orientation_6_transposes_dimensions() {
        // 6 = 90 deg CW: W x H becomes H x W.
        let out = apply_exif_orientation(DynamicImage::new_luma8(20, 10), 6);
        assert_eq!((out.width(), out.height()), (10, 20));
        // 1 = identity.
        let out = apply_exif_orientation(DynamicImage::new_luma8(20, 10), 1);
        assert_eq!((out.width(), out.height()), (20, 10));
    }

    #[test]
    fn uniform_white_page_with_ink_is_not_glare() {
        // A clean scan is almost entirely hot pixels, but uniformly so.
        let mut img = striped(200, 200);
        let r = assess_glare(&img, 250, 0.05);
        assert!(r.ok, "clean page must pass: {}", r.reason);
        // Fully blank saturated page: nothing survived -> washed out.
        for p in img.pixels_mut() {
            *p = Luma([255]);
        }
        let r = assess_glare(&img, 250, 0.05);
        assert!(!r.ok, "blank saturated page must fail");
    }

    #[test]
    fn bright_blob_on_dark_page_is_glare() {
        let mut img = GrayImage::from_pixel(200, 200, Luma([100]));
        for y in (10..200).step_by(20) {
            for x in 0..200 {
                img.put_pixel(x, y, Luma([10]));
            }
        }
        // Specular disc: saturated pixels on an otherwise dark page.
        for y in 0..200 {
            for x in 0..200 {
                let (dx, dy) = (x as i32 - 150, y as i32 - 50);
                if dx * dx + dy * dy < 60 * 60 {
                    img.put_pixel(x as u32, y as u32, Luma([255]));
                }
            }
        }
        let r = assess_glare(&img, 250, 0.05);
        assert!(!r.ok, "glare blob must fail: {}", r.reason);
    }

    #[test]
    fn defaults_match_big500_training() {
        let args = Args::try_parse_from([
            "ingest-cleanup",
            "--input",
            "photo.jpg",
            "--output",
            "clean.png",
        ])
        .expect("defaults parse");
        assert_eq!(args.blur_threshold, 6.0);
        assert_eq!(args.glare_threshold, 0.12);
        assert_eq!(args.glare_level, 250);
    }
}
