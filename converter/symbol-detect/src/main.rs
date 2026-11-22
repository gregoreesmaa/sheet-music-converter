//! `symbol-detect`: staff crops + staff map -> glyph boxes/labels JSON.
//!
//! For every crop image this tool runs a [`GlyphDetector`] backend and emits
//! one record per crop with its detected glyph boxes, labels, confidences,
//! and staff-relative positions. The default `classical` backend is a small,
//! dependency-light connected-components stub (dark-pixel blobs + rule-based
//! labels); learned inference plugs in through the same [`GlyphDetector`]
//! trait via [`OnnxDetectorHook`]. See `README.md` for schemas and examples.

use clap::Parser;
use image::GrayImage;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

mod onnx_heat;
#[cfg(feature = "onnx")]
mod onnx_patch;

/// Detect music-symbol glyphs from staff crops guided by a staff map.
#[derive(Parser, Debug)]
#[command(
    name = "symbol-detect",
    version,
    about = "Crops + staff map -> glyph boxes/labels JSON"
)]
struct Args {
    /// Directory of crop images (png/jpg/bmp/tiff/webp); crop id = file stem.
    #[arg(long)]
    crops_dir: PathBuf,

    /// Staff-map JSON: `{"systems": [...]}` / `{"staves": [...]}` or a bare array.
    #[arg(long)]
    staff_map: PathBuf,

    /// Destination JSON file for the glyph detections.
    #[arg(long, default_value = "glyphs.json")]
    out: PathBuf,

    /// Detector backend: `classical` (rule stub), `onnx` (patch
    /// classifier via `--model`; requires the `onnx` feature), or
    /// `heatmap` (CenterNet-style proposer via `--model`; requires the
    /// `onnx` feature).
    #[arg(long, default_value = "classical")]
    backend: String,

    /// ONNX model path (required with `--backend onnx` or `heatmap`).
    #[arg(long)]
    model: Option<PathBuf>,

    /// Center-probability floor for `--backend heatmap` peak extraction.
    /// Tile gate (ds0005 heldout, `tmp-train/det-data/sweep_thr.py`):
    /// 0.9 maximizes micro-F1 (0.798; P 0.754 R 0.847, Note F1 0.982).
    /// 0.5 was the eval_heat recall point (R 0.943, F1 0.263).
    #[arg(long, default_value_t = 0.9)]
    heat_threshold: f32,

    /// Separate peak floor for the Beam class: beam peaks run an order of
    /// magnitude colder (tile R 0.51 at 0.5, 0.00 at 0.85), so the page
    /// operating point would erase every duration cue without this.
    #[arg(long, default_value_t = 0.6)]
    heat_threshold_beam: f32,

    /// Ink threshold: pixels darker than this count as ink (0–255).
    /// No effect on two-valued ingest-binarized input (pixels already 0/255).
    #[arg(long, default_value_t = 128)]
    threshold: u8,

    /// Ink blobs with fewer pixels than this are dropped as noise.
    /// Re-trained on 24 fresh crops: 16 wins on every sub-metric (CV 0.25,
    /// corr 0.38, unknown 0.05 vs 24's 0.27/0.36/0.06). Round 1 favored 24
    /// on a noisier 5-point correlation; see TRAINING_BIG500.md sec. 8.
    #[arg(long, default_value_t = 16)]
    min_area: usize,

    /// Fallback staff spacing in pixels when the staff map has no entry.
    #[arg(long, default_value_t = 10.0)]
    staff_spacing: f64,

    /// Comma-separated glyph vocabulary; rule labels outside it become `unknown`.
    #[arg(
        long,
        default_value = "notehead-black,notehead-void,stem,beam,dot,accidental,clef,rest"
    )]
    labels: String,

    /// Second ONNX patch model whose listed labels are appended as extra
    /// evidence boxes (e.g. a flag/beam model backing a classical primary
    /// run). Same proposer, so boxes coincide; the mapper pools them by
    /// label and confidence. Requires the `onnx` feature.
    #[arg(long)]
    evidence_model: Option<PathBuf>,

    /// Evidence labels appended from `--evidence-model`
    /// (comma-separated; default flag,beam).
    #[arg(long, default_value = "flag,beam")]
    evidence_labels: String,
}

/// One entry of the staff map: staff geometry for a crop.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct StaffEntry {
    /// Optional entry id (informational only).
    #[serde(default)]
    id: Option<String>,
    /// Crop this entry applies to (file stem or file name). When absent the
    /// entry is global and applies to every crop lacking its own entry.
    #[serde(default, alias = "crop")]
    crop_id: Option<String>,
    /// Y coordinates of the staff lines in crop pixels, top line first.
    #[serde(default)]
    staff_lines: Vec<f64>,
    /// Staff spacing (line-to-line distance) in crop pixels.
    #[serde(default)]
    staff_spacing: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct StaffMapWrapped {
    #[serde(default)]
    systems: Vec<StaffEntry>,
    #[serde(default)]
    staves: Vec<StaffEntry>,
}

/// Staff geometry resolved for one crop.
#[derive(Debug, Clone)]
pub struct StaffContext {
    lines: Vec<f64>,
    spacing: f64,
}

/// One decoded crop image.
pub struct CropFrame {
    id: String,
    file_name: String,
    gray: GrayImage,
}

/// Axis-aligned glyph box in crop pixels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BBox {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// One detected glyph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Glyph {
    bbox: BBox,
    label: String,
    confidence: f64,
    /// Vertical position in half staff-spaces below the top staff line,
    /// snapped to 0.5; `None` when the crop has no staff lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    staff_step: Option<f64>,
    area: usize,
}

/// Detection record for one crop.
#[derive(Debug, Serialize, Deserialize)]
struct CropResult {
    crop_id: String,
    source: String,
    width: u32,
    height: u32,
    glyphs: Vec<Glyph>,
}

/// Top-level output document.
#[derive(Debug, Serialize, Deserialize)]
struct Output {
    backend: String,
    crops: Vec<CropResult>,
}

// ---------------------------------------------------------------------------
// Detector backends
// ---------------------------------------------------------------------------

/// Inference backend hook: classical CV today, learned model tomorrow.
///
/// Implementors take a decoded crop plus its staff context and return glyph
/// detections in crop-pixel coordinates, sorted top-to-bottom, left-to-right.
pub trait GlyphDetector {
    /// Backend name recorded in the output document.
    fn name(&self) -> &'static str;
    /// Detect glyphs in `crop`. Errors are fatal for the run.
    fn detect(&self, crop: &CropFrame, ctx: &StaffContext) -> Result<Vec<Glyph>, String>;
}

/// Classical connected-components stub detector.
///
/// Binarizes the crop at [`Args::threshold`], extracts 8-connected ink blobs
/// smaller than [`Args::min_area`] are dropped as noise, then labels each
/// blob with aspect-ratio / staff-normalized-size / fill-ratio rules. Fully
/// deterministic: same pixels + same staff map = same output.
pub struct ClassicalStubDetector {
    pub threshold: u8,
    pub min_area: usize,
    pub allowed: Vec<String>,
}

impl GlyphDetector for ClassicalStubDetector {
    fn name(&self) -> &'static str {
        "classical-stub"
    }

    fn detect(&self, crop: &CropFrame, ctx: &StaffContext) -> Result<Vec<Glyph>, String> {
        let mut glyphs = Vec::new();
        for (bbox, area) in connected_components(&crop.gray, self.threshold, self.min_area) {
            let label = self.classify(&bbox, area, ctx);
            let cy = bbox.y as f64 + bbox.h as f64 / 2.0;
            glyphs.push(Glyph {
                staff_step: snap_to_staff(cy, ctx),
                confidence: score(area, &bbox),
                bbox,
                label,
                area,
            });
        }
        // Blobs come out in scan order already; keep output canonical anyway.
        glyphs.sort_by_key(|g| (g.bbox.y, g.bbox.x));
        Ok(glyphs)
    }
}

impl ClassicalStubDetector {
    /// Rule-based label from box shape normalized by the staff spacing.
    /// `hn`/`wn` are height/width in staff spaces; `fill` is ink coverage.
    fn classify(&self, bbox: &BBox, area: usize, ctx: &StaffContext) -> String {
        let sp = ctx.spacing.max(1e-6);
        let hn = bbox.h as f64 / sp;
        let wn = bbox.w as f64 / sp;
        let fill = area as f64 / (bbox.w.max(1) as f64 * bbox.h.max(1) as f64);
        let raw = if hn <= 0.7 && wn <= 0.7 {
            "dot"
        } else if hn >= 3.0 && wn <= 1.2 {
            "stem"
        } else if wn >= 2.0 && hn <= 1.5 {
            "beam"
        } else if hn >= 3.0 {
            "clef"
        } else if hn <= 1.7 && wn <= 1.7 {
            if fill >= 0.45 {
                "notehead-black"
            } else {
                "notehead-void"
            }
        } else if hn <= 3.5 && wn <= 2.5 {
            "accidental"
        } else {
            "unknown"
        };
        if raw == "unknown" || self.allowed.iter().any(|l| l == raw) {
            raw.to_string()
        } else {
            "unknown".to_string()
        }
    }
}

/// Placeholder score for classical blobs: denser blobs score higher.
/// Deterministic; treat as a stub ranking signal, not a probability.
fn score(area: usize, bbox: &BBox) -> f64 {
    let fill = area as f64 / (bbox.w.max(1) as f64 * bbox.h.max(1) as f64);
    let v = 0.55 + 0.4 * fill.clamp(0.0, 1.0);
    (v.clamp(0.5, 0.95) * 1000.0).round() / 1000.0
}

/// Snap a y coordinate to half staff-spaces below the top staff line.
fn snap_to_staff(cy: f64, ctx: &StaffContext) -> Option<f64> {
    if ctx.lines.is_empty() || ctx.spacing <= 0.0 {
        return None;
    }
    let half = ctx.spacing / 2.0;
    Some(((cy - ctx.lines[0]) / half * 2.0).round() / 2.0)
}

/// Chord-stack split tuning, calibrated on degraded phone-photo pages
/// (staff spacing ~8-9 px): stacked thirds put head centers ~1 spacing
/// apart with ~0.9-spacing head height, so each extra head adds ~0.75
/// spacings of wide-span height on average (seconds pack tighter).
const SPLIT_MIN_H_SP: f64 = 1.4;
const SPLIT_MAX_W_SP: f64 = 2.2;
const SPLIT_MIN_FILL: f64 = 0.45;
const SPLIT_MIN_HEADS_SP: f64 = 1.3;
const SPLIT_WIDE_FRAC: f64 = 0.55;
const SPLIT_HEAD_H_SP: f64 = 0.9;
const SPLIT_PER_HEAD_SP: f64 = 0.75;
const SPLIT_MAX_HEADS: usize = 4;
const SPLIT_MIN_DENSITY: f64 = 0.25;
const SPLIT_MIN_PART_H_SP: f64 = 0.3;

/// Split notehead blobs that stack several chord tones into one glyph box
/// per head. Heads in a stack share the blob (connected components merge
/// touching noteheads and their stem), so one box would map to one note
/// and drop the chord tones. Tall narrow notehead blobs with solid fill
/// are divided into equal-count wide-row groups with boundaries anchored
/// to the narrowest nearby row (the neck between heads), so attachments
/// that add wide rows to one head cannot drag a fragment center onto a
/// step boundary; each part inherits the label and confidence and
/// re-snaps its staff step. (Wide merged
/// seconds stay whole: their population is mostly head-plus-attachment
/// merges — beams, dots, slurs share the horizontal orientation and no
/// geometric rule separates them from true pairs.) Returns the input
/// unchanged when the spacing is unknown. Deterministic: same pixels +
/// context give same glyphs. Callers keep their own output ordering.
fn split_chord_blobs(
    gray: &GrayImage,
    threshold: u8,
    glyphs: Vec<Glyph>,
    ctx: &StaffContext,
) -> Vec<Glyph> {
    let sp = ctx.spacing;
    if sp <= 0.0 {
        return glyphs;
    }
    let mut out = Vec::with_capacity(glyphs.len() + 4);
    for g in glyphs {
        let h_sp = f64::from(g.bbox.h) / sp;
        let w_sp = f64::from(g.bbox.w) / sp;
        let is_head = g.label == "notehead-black" || g.label == "notehead-void";
        if is_head && h_sp >= SPLIT_MIN_H_SP && w_sp <= SPLIT_MAX_W_SP {
            if let Some(parts) = split_one_head(gray, threshold, &g, ctx) {
                out.extend(parts);
                continue;
            }
        }
        out.push(g);
    }
    out.sort_by_key(|g| (g.bbox.y, g.bbox.x));
    out
}

/// Erase horizontal staff-line pixels from a grayscale tile, preserving
/// symbol pixels that cross the lines.
///
/// For every dark pixel on one of `line_rows`, the vertical dark run through
/// that pixel is measured; runs of length `<= max_line_thickness` are line
/// pixels (whitened in the returned image), longer runs belong to symbols
/// crossing the line and are kept. The input image is not modified.
fn erase_staff_lines(
    gray: &GrayImage,
    line_rows: &[u32],
    max_line_thickness: u32,
    white: u8,
) -> GrayImage {
    let (w, h) = (gray.width(), gray.height());
    let dark = |x: u32, y: u32| gray.get_pixel(x, y).0[0] < white;
    let mut out = gray.clone();
    for &ly in line_rows {
        if ly >= h {
            continue;
        }
        for x in 0..w {
            if !dark(x, ly) {
                continue;
            }
            let mut top = ly;
            while top > 0 && dark(x, top - 1) {
                top -= 1;
            }
            let mut bot = ly;
            while bot + 1 < h && dark(x, bot + 1) {
                bot += 1;
            }
            if bot - top + 1 <= max_line_thickness {
                out.put_pixel(x, ly, image::Luma([255]));
            }
        }
    }
    out
}

/// Split one tall notehead blob into per-head glyphs, or `None` when the
/// blob does not look like a chord stack (sparse fill, single head height,
/// or a degenerate split).
fn split_one_head(
    gray: &GrayImage,
    threshold: u8,
    g: &Glyph,
    ctx: &StaffContext,
) -> Option<Vec<Glyph>> {
    let sp = ctx.spacing;
    let (w, h) = (gray.width(), gray.height());
    if w == 0 || h == 0 {
        return None;
    }
    let x0 = g.bbox.x.min(w - 1);
    let y0 = g.bbox.y.min(h - 1);
    let x1 = (g.bbox.x + g.bbox.w).min(w);
    let y1 = (g.bbox.y + g.bbox.h).min(h);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    // Per-row ink runs inside the box, plus the ink width of every row
    // (zero for blank rows) so split boundaries can seek the neck.
    let mut rows: Vec<(u32, u32, u32, u32)> = Vec::new();
    let mut widths = vec![0u32; (y1 - y0) as usize];
    let mut ink_total: u64 = 0;
    for y in y0..y1 {
        let mut xmin = u32::MAX;
        let mut xmax = 0u32;
        let mut n = 0u32;
        for x in x0..x1 {
            if gray.get_pixel(x, y).0[0] < threshold {
                xmin = xmin.min(x);
                xmax = xmax.max(x);
                n += 1;
            }
        }
        if n > 0 {
            rows.push((y, xmin, xmax, n));
            widths[(y - y0) as usize] = xmax - xmin + 1;
            ink_total += u64::from(n);
        }
    }
    let area = f64::from(x1 - x0) * f64::from(y1 - y0);
    if ink_total as f64 / area.max(1.0) < SPLIT_MIN_FILL {
        return None;
    }
    let wide_min = SPLIT_WIDE_FRAC * sp;
    let wide: Vec<(u32, u32, u32, u32)> = rows
        .iter()
        .copied()
        .filter(|&(_, xmin, xmax, _)| f64::from(xmax - xmin + 1) >= wide_min)
        .collect();
    if wide.len() < 2 {
        return None;
    }
    let span = f64::from(wide.last().unwrap().0 - wide.first().unwrap().0 + 1);
    let heads_sp = span / sp;
    if heads_sp < SPLIT_MIN_HEADS_SP {
        return None;
    }
    let n = (((heads_sp - SPLIT_HEAD_H_SP) / SPLIT_PER_HEAD_SP).round() as usize + 1)
        .clamp(2, SPLIT_MAX_HEADS);
    // Contiguous groups with balanced wide-row counts; each head feeds
    // about as many wide rows, so quantiles land near head centers.
    let base = wide.len() / n;
    let rem = wide.len() % n;
    if base == 0 {
        return None;
    }
    // Quantile boundary indices into `wide` (group i starts at bounds[i]).
    let quantile = {
        let mut b = Vec::with_capacity(n + 1);
        let mut idx = 0usize;
        b.push(0);
        for i in 0..n {
            idx += base + usize::from(i < rem);
            b.push(idx);
        }
        b
    };
    // Anchor each interior boundary to the narrowest row within a quarter
    // spacing (the neck between heads): equal counts drift when an
    // attachment contributes extra wide rows to one head.
    let mut bounds = quantile.clone();
    let reach = (0.25 * sp).round().max(1.0) as u32;
    let row_width = |y: u32| widths[(y - y0) as usize];
    for k in 1..n {
        let yb = wide[bounds[k]].0;
        let lo = yb.saturating_sub(reach).max(y0 + 1);
        let hi = (yb + reach).min(y1.saturating_sub(1));
        let mut neck = yb;
        for y in lo..=hi {
            if row_width(y) < row_width(neck) {
                neck = y;
            }
        }
        let mut cut = bounds[k];
        for (i, &(ry, _, _, _)) in wide.iter().enumerate() {
            if ry > neck {
                cut = i;
                break;
            }
        }
        if cut > bounds[k - 1] && cut < quantile[k + 1] {
            bounds[k] = cut;
        }
    }
    for k in 1..n {
        if bounds[k] <= bounds[k - 1] {
            bounds[k] = quantile[k];
        }
    }
    let mut parts = Vec::with_capacity(n);
    for i in 0..n {
        let group = &wide[bounds[i]..bounds[i + 1]];
        let gy0 = group.first().unwrap().0;
        let gy1 = group.last().unwrap().0;
        if f64::from(gy1 - gy0 + 1) < SPLIT_MIN_PART_H_SP * sp {
            return None;
        }
        let gx0 = group.iter().map(|&(_, xmin, _, _)| xmin).min().unwrap();
        let gx1 = group.iter().map(|&(_, _, xmax, _)| xmax).max().unwrap();
        let ink: u64 = group.iter().map(|&(_, _, _, n)| u64::from(n)).sum();
        let dens = ink as f64 / (f64::from(gx1 - gx0 + 1) * f64::from(gy1 - gy0 + 1));
        if dens < SPLIT_MIN_DENSITY {
            return None;
        }
        let bbox = BBox {
            x: gx0,
            y: gy0,
            w: gx1 - gx0 + 1,
            h: gy1 - gy0 + 1,
        };
        let cy = f64::from(gy0) + f64::from(gy1 - gy0 + 1) / 2.0;
        parts.push(Glyph {
            bbox,
            label: g.label.clone(),
            confidence: g.confidence,
            staff_step: snap_to_staff(cy, ctx),
            area: ink as usize,
        });
    }
    Some(parts)
}
/// Blob centers must sit within this many staff-spaces beyond a staff's
/// outermost lines to be rescued: rests live inside staves, while
/// rest-shaped lyric fragments and dynamics sit in the gaps between
/// staves (near lines, but inside no staff's span).
const REST_STAFF_MARGIN: f64 = 0.75;

/// Gaps wider than this many staff-spaces split the line union into
/// separate staves. In-staff gaps are ~1sp even with tilt; anything at or
/// above 2sp is an inter-staff (or inter-system) break.
const STAFF_SPLIT_GAP: f64 = 2.0;

/// True when the blob center sits inside some staff's vertical span (top
/// to bottom line, plus [`REST_STAFF_MARGIN`] staff-spaces each side for
/// bar rests that jut past the outer lines). The context lines form a
/// page union, so staves are recovered by splitting at wide gaps. Empty
/// line sets (and non-positive spacing) skip the gate rather than
/// black-holing every rescue.
fn in_staff_span(center_y: f64, ctx: &StaffContext) -> bool {
    if ctx.lines.is_empty() || ctx.spacing <= 0.0 {
        return true;
    }
    let mut ys = ctx.lines.clone();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let margin = REST_STAFF_MARGIN * ctx.spacing;
    let split = STAFF_SPLIT_GAP * ctx.spacing;
    let mut top = ys[0];
    for w in ys.windows(2) {
        if w[1] - w[0] > split {
            if center_y >= top - margin && center_y <= w[0] + margin {
                return true;
            }
            top = w[1];
        }
    }
    center_y >= top - margin && center_y <= ys[ys.len() - 1] + margin
}

/// Rest-shaped `unknown` blobs rescued as rests. PatchNet's rest class
/// misses most degraded rests (they argmax to `other`), but rests have
/// tight size bands per subtype from symbox truth: eighth slashes are
/// ~1.2 x 1.9 spacings, half/whole bars ~1.3 x 0.6 (solid), quarter
/// squiggles ~0.9 x 2.8. Lyric bits (small) and slur segments (long
/// thin) fall outside every band, and rest-shaped fragments far from any
/// staff line (lyrics, dynamics) are excluded by position. Fill comes from
/// the blob area, so no pixel pass is needed. Keeps input order.
fn rescue_rests(glyphs: Vec<Glyph>, ctx: &StaffContext) -> Vec<Glyph> {
    let sp = ctx.spacing;
    if sp <= 0.0 {
        return glyphs;
    }
    glyphs
        .into_iter()
        .map(|mut g| {
            if g.label == "unknown" {
                let w = f64::from(g.bbox.w) / sp;
                let h = f64::from(g.bbox.h) / sp;
                let fill = g.area as f64 / (f64::from(g.bbox.w) * f64::from(g.bbox.h)).max(1.0);
                let eighth = (0.9..=1.5).contains(&w)
                    && (1.5..=2.3).contains(&h)
                    && (0.22..=0.7).contains(&fill);
                let half_whole =
                    (0.9..=1.7).contains(&w) && (0.35..=0.95).contains(&h) && fill >= 0.55;
                let quarter = (0.6..=1.3).contains(&w)
                    && (2.2..=3.4).contains(&h)
                    && (0.22..=0.65).contains(&fill);
                let cy = f64::from(g.bbox.y) + f64::from(g.bbox.h) / 2.0;
                if (eighth || half_whole || quarter) && in_staff_span(cy, ctx) {
                    g.label = "rest".to_string();
                }
            }
            g
        })
        .collect()
}

/// 8-connected ink components of a grayscale image: `(bbox, area)` in scan
/// (top-to-bottom, left-to-right) order, blobs under `min_area` dropped.
fn connected_components(gray: &GrayImage, threshold: u8, min_area: usize) -> Vec<(BBox, usize)> {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let mut visited = vec![false; w * h];
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if visited[idx] || gray.get_pixel(x as u32, y as u32).0[0] >= threshold {
                continue;
            }
            // Flood-fill one blob.
            let (mut x0, mut y0, mut x1, mut y1) = (x, y, x, y);
            let mut area = 0usize;
            let mut stack = vec![(x, y)];
            visited[idx] = true;
            while let Some((px, py)) = stack.pop() {
                x0 = x0.min(px);
                y0 = y0.min(py);
                x1 = x1.max(px);
                y1 = y1.max(py);
                area += 1;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = px as i32 + dx;
                        let ny = py as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let nidx = ny as usize * w + nx as usize;
                        if visited[nidx] {
                            continue;
                        }
                        visited[nidx] = true;
                        if gray.get_pixel(nx as u32, ny as u32).0[0] < threshold {
                            stack.push((nx as usize, ny as usize));
                        }
                    }
                }
            }
            if area >= min_area {
                out.push((
                    BBox {
                        x: x0 as u32,
                        y: y0 as u32,
                        w: (x1 - x0 + 1) as u32,
                        h: (y1 - y0 + 1) as u32,
                    },
                    area,
                ));
            }
        }
    }
    out
}

/// ONNX patch-classifier backend.
///
/// Same proposer as the classical stub (connected components), but every
/// blob is classified by a learned model (notehead / rest / [beam /]
/// other; the class count comes from the model file) and noteheads are
/// split black/void by the interior-whiteness rule.
/// Without the `onnx` feature this is a load-time error stub (see `load`).
pub struct OnnxDetectorHook {
    pub model_path: PathBuf,
    pub threshold: u8,
    pub min_area: usize,
    #[cfg(feature = "onnx")]
    model: onnx_patch::OnnxPatchModel,
}

impl OnnxDetectorHook {
    pub fn load(model_path: PathBuf, threshold: u8, min_area: usize) -> Result<Self, String> {
        if !model_path.is_file() {
            return Err(format!("onnx model not found: {}", model_path.display()));
        }
        #[cfg(feature = "onnx")]
        {
            let model = onnx_patch::OnnxPatchModel::load(&model_path)?;
            Ok(Self {
                model_path,
                threshold,
                min_area,
                model,
            })
        }
        #[cfg(not(feature = "onnx"))]
        Err(format!(
            "onnx backend requires rebuilding with `--features onnx` \
             (model: {}, threshold: {}, min_area: {}); see README.md",
            model_path.display(),
            threshold,
            min_area
        ))
    }
}

/// Black/void split for a notehead blob: the central third of the bbox is
/// mostly white in void (half/whole) noteheads, nearly all ink in filled
/// ones. The 0.35 cut separates the two clean-render modes (<0.1 vs >=0.4).
/// Used by the onnx backend and its tests; the default build has neither.
#[cfg(any(feature = "onnx", test))]
fn is_void_notehead(gray: &GrayImage, bbox: &BBox, threshold: u8) -> bool {
    let (w, h) = (gray.width(), gray.height());
    let x0 = bbox.x + bbox.w / 3;
    let x1 = bbox.x + 2 * bbox.w / 3;
    let y0 = bbox.y + bbox.h / 3;
    let y1 = bbox.y + 2 * bbox.h / 3;
    let mut white = 0u32;
    let mut total = 0u32;
    for y in y0..y1.min(h) {
        for x in x0..x1.min(w) {
            total += 1;
            if gray.get_pixel(x, y).0[0] >= threshold {
                white += 1;
            }
        }
    }
    total > 0 && white as f32 / total as f32 > 0.35
}

/// PatchNet class index to glyph label. Class 0 is notehead in every
/// model (the void/black split is a pixel rule at the call site); `beam`
/// exists only in 4+-class models and `flag` only in 5+-class models, so
/// legacy 3-class outputs keep mapping class 2 to `unknown`.
fn label_for_class(cls: usize, n_logits: usize) -> Option<&'static str> {
    match cls {
        1 => Some("rest"),
        2 if n_logits >= 4 => Some("beam"),
        4 if n_logits >= 5 => Some("flag"),
        _ => None,
    }
}

#[cfg(feature = "onnx")]
impl GlyphDetector for OnnxDetectorHook {
    fn name(&self) -> &'static str {
        "onnx-hook"
    }

    fn detect(&self, crop: &CropFrame, ctx: &StaffContext) -> Result<Vec<Glyph>, String> {
        let comps = connected_components(&crop.gray, self.threshold, self.min_area);
        let boxes: Vec<BBox> = comps.iter().map(|(b, _)| b.clone()).collect();
        let probs = self
            .model
            .classify_batch(&crop.gray, &boxes, self.threshold);
        let mut glyphs = Vec::with_capacity(comps.len());
        for ((bbox, area), p) in comps.into_iter().zip(probs) {
            let cls = p
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(usize::MAX);
            let label = match cls {
                0 if is_void_notehead(&crop.gray, &bbox, self.threshold) => "notehead-void",
                0 => "notehead-black",
                _ => label_for_class(cls, p.len()).unwrap_or("unknown"),
            }
            .to_string();
            let cy = bbox.y as f64 + bbox.h as f64 / 2.0;
            glyphs.push(Glyph {
                staff_step: snap_to_staff(cy, ctx),
                confidence: p.get(cls).copied().unwrap_or(0.0) as f64,
                bbox,
                label,
                area,
            });
        }
        // Rests PatchNet leaves as `other` come back via shape bands.
        let glyphs = rescue_rests(glyphs, ctx);
        // One box per notehead: stacked chord tones share a blob.
        let mut glyphs = split_chord_blobs(&crop.gray, self.threshold, glyphs, ctx);
        glyphs.sort_by_key(|g| (g.bbox.y, g.bbox.x));
        Ok(glyphs)
    }
}

#[cfg(not(feature = "onnx"))]
impl GlyphDetector for OnnxDetectorHook {
    fn name(&self) -> &'static str {
        "onnx-hook"
    }

    fn detect(&self, _crop: &CropFrame, _ctx: &StaffContext) -> Result<Vec<Glyph>, String> {
        Err(format!(
            "onnx backend requires rebuilding with `--features onnx` (model: {}); \
             see README.md",
            self.model_path.display()
        ))
    }
}

/// Heatmap-proposer backend: whole-crop CenterNet-style detection.
///
/// No blob proposer: boxes come from center peaks + regressed sizes, so
/// beamed groups and touching symbols survive as separate detections. Note
/// boxes go through the same central-third void rule as the patch backend.
pub struct HeatmapDetector {
    pub model_path: PathBuf,
    pub heat_threshold: f32,
    pub heat_threshold_beam: f32,
    pub allowed: Vec<String>,
    #[cfg(feature = "onnx")]
    model: onnx_heat::OnnxHeatModel,
}

impl HeatmapDetector {
    pub fn load(
        model_path: PathBuf,
        heat_threshold: f32,
        heat_threshold_beam: f32,
        allowed: Vec<String>,
    ) -> Result<Self, String> {
        if !model_path.is_file() {
            return Err(format!("heatmap model not found: {}", model_path.display()));
        }
        #[cfg(feature = "onnx")]
        {
            let model = onnx_heat::OnnxHeatModel::load(&model_path)?;
            Ok(Self {
                model_path,
                heat_threshold,
                heat_threshold_beam,
                allowed,
                model,
            })
        }
        #[cfg(not(feature = "onnx"))]
        Err(format!(
            "heatmap backend requires rebuilding with `--features onnx` \
             (model: {}, heat_threshold: {}, beam: {}, labels: {}); see README.md",
            model_path.display(),
            heat_threshold,
            heat_threshold_beam,
            allowed.join(",")
        ))
    }
}

/// Heatmap class index (see `onnx_heat::HEAT_CLASSES`) to glyph label.
/// `Hook` (beam hook fragment) maps to `beam`; the void/black split for
/// class 0 happens at the call site via [`is_void_notehead`].
#[cfg(any(feature = "onnx", test))]
fn heat_label(class: usize) -> &'static str {
    match class {
        0 => "notehead-black",
        1 => "rest",
        2 | 3 => "beam",
        4 => "accidental",
        5 => "dot",
        6 => "clef",
        _ => "unknown",
    }
}

#[cfg(feature = "onnx")]
impl GlyphDetector for HeatmapDetector {
    fn name(&self) -> &'static str {
        "heatmap"
    }

    fn detect(&self, crop: &CropFrame, ctx: &StaffContext) -> Result<Vec<Glyph>, String> {
        let proposals =
            self.model
                .propose(&crop.gray, self.heat_threshold, self.heat_threshold_beam)?;
        let (w, h) = (crop.gray.width(), crop.gray.height());
        let mut glyphs = Vec::with_capacity(proposals.len());
        for p in proposals {
            let x0 = p.x0.round().clamp(0.0, w as f32) as u32;
            let y0 = p.y0.round().clamp(0.0, h as f32) as u32;
            let x1 = p.x1.round().clamp(0.0, w as f32) as u32;
            let y1 = p.y1.round().clamp(0.0, h as f32) as u32;
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            let bbox = BBox {
                x: x0,
                y: y0,
                w: x1 - x0,
                h: y1 - y0,
            };
            let mut label = heat_label(p.class);
            if p.class == 0 && is_void_notehead(&crop.gray, &bbox, 128) {
                label = "notehead-void";
            }
            if !self.allowed.iter().any(|l| l == label) {
                label = "unknown";
            }
            let cy = y0 as f64 + (y1 - y0) as f64 / 2.0;
            glyphs.push(Glyph {
                staff_step: snap_to_staff(cy, ctx),
                confidence: p.score as f64,
                area: (bbox.w as usize).saturating_mul(bbox.h as usize),
                bbox,
                label: label.to_string(),
            });
        }
        glyphs.sort_by_key(|g| (g.bbox.y, g.bbox.x));
        Ok(glyphs)
    }
}

#[cfg(not(feature = "onnx"))]
impl GlyphDetector for HeatmapDetector {
    fn name(&self) -> &'static str {
        "heatmap"
    }

    fn detect(&self, _crop: &CropFrame, _ctx: &StaffContext) -> Result<Vec<Glyph>, String> {
        Err(format!(
            "heatmap backend requires rebuilding with `--features onnx` (model: {}); \
             see README.md",
            self.model_path.display()
        ))
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Image file extensions accepted as crops.
fn is_crop_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some(
            "png"
                | "jpg"
                | "jpeg"
                | "bmp"
                | "tif"
                | "tiff"
                | "webp"
                | "pbm"
                | "pgm"
                | "ppm"
                | "pnm"
        )
    )
}

/// Load every crop image in `dir`, sorted by file name for stable output.
/// Second-model evidence appended to a primary detection run.
/// Holds the model plus the label subset to keep; the mapper gates kept
/// boxes by confidence, so the detector appends every listed prediction.
#[cfg(feature = "onnx")]
struct EvidenceModel {
    model: onnx_patch::OnnxPatchModel,
    labels: Vec<String>,
}

/// Classify a crop's proposer boxes with the evidence model and keep the
/// listed labels as extra glyph entries (confidence = argmax probability).
/// Boxes coincide with the primary run's (same proposer); downstream
/// consumers match them by label.
#[cfg(feature = "onnx")]
fn evidence_glyphs(
    model: &onnx_patch::OnnxPatchModel,
    gray: &GrayImage,
    threshold: u8,
    min_area: usize,
    labels: &[String],
    ctx: &StaffContext,
) -> Vec<Glyph> {
    let comps = connected_components(gray, threshold, min_area);
    let boxes: Vec<BBox> = comps.iter().map(|(b, _)| b.clone()).collect();
    let probs = model.classify_batch(gray, &boxes, threshold);
    let mut out = Vec::new();
    for ((bbox, area), p) in comps.into_iter().zip(probs) {
        let cls = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(usize::MAX);
        if let Some(label) = label_for_class(cls, p.len()) {
            if labels.iter().any(|l| l == label) {
                let cy = bbox.y as f64 + bbox.h as f64 / 2.0;
                out.push(Glyph {
                    staff_step: snap_to_staff(cy, ctx),
                    confidence: p.get(cls).copied().unwrap_or(0.0) as f64,
                    bbox,
                    label: label.to_string(),
                    area,
                });
            }
        }
    }
    out
}

fn load_crops(dir: &Path) -> Result<Vec<CropFrame>, String> {
    let mut names: Vec<String> = Vec::new();
    let entries =
        fs::read_dir(dir).map_err(|e| format!("cannot read --crops-dir {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot list {}: {e}", dir.display()))?;
        let path = entry.path();
        if path.is_file() && is_crop_file(&path) {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| format!("non-UTF8 file name: {}", path.display()))?
                .to_string();
            names.push(name);
        }
    }
    names.sort();
    if names.is_empty() {
        return Err(format!("no crop images found in {}", dir.display()));
    }
    let mut crops = Vec::with_capacity(names.len());
    for name in names {
        let path = dir.join(&name);
        let img = image::open(&path)
            .map_err(|e| format!("cannot decode {}: {e}", path.display()))?
            .to_luma8();
        let stem = Path::new(&name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&name)
            .to_string();
        crops.push(CropFrame {
            id: stem,
            file_name: name,
            gray: img,
        });
    }
    Ok(crops)
}

/// Load the staff map, tolerating a UTF-8 BOM and three shapes:
/// `{"systems": [...]}`, `{"staves": [...]}`, or a bare `[...]` array.
fn load_staff_map(path: &Path) -> Result<Vec<StaffEntry>, String> {
    let raw =
        fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let text = raw.strip_prefix('\u{FEFF}').unwrap_or(&raw);
    if let Ok(entries) = serde_json::from_str::<Vec<StaffEntry>>(text) {
        return Ok(entries);
    }
    serde_json::from_str::<StaffMapWrapped>(text)
        .map(|w| w.systems.into_iter().chain(w.staves).collect())
        .map_err(|e| {
            format!(
                "cannot parse {} as `[entries]` or `{{\"systems\": [...]}}`: {e}",
                path.display()
            )
        })
}

/// Resolve the staff context for one crop: an entry naming the crop
/// (by stem or file name) wins, otherwise the first global entry without a
/// `crop_id` applies, otherwise the `--staff-spacing` fallback is used.
fn resolve_context(
    entries: &[StaffEntry],
    crop: &CropFrame,
    fallback_spacing: f64,
) -> Result<StaffContext, String> {
    if fallback_spacing <= 0.0 {
        return Err("--staff-spacing must be positive".to_string());
    }
    let mut global: Option<&StaffEntry> = None;
    for entry in entries {
        match &entry.crop_id {
            Some(id) if *id == crop.id || *id == crop.file_name => {
                return Ok(context_from(entry, fallback_spacing));
            }
            None if global.is_none() => global = Some(entry),
            _ => {}
        }
    }
    Ok(global
        .map(|e| context_from(e, fallback_spacing))
        .unwrap_or(StaffContext {
            lines: Vec::new(),
            spacing: fallback_spacing,
        }))
}

/// Median staff-line gap as the spacing estimate. Most consecutive gaps in
/// a sorted union of staff lines are in-staff spacings, so the median stays
/// correct past a few missed lines and the wide inter-staff gaps; gaps at
/// or below a pixel are fitter duplicates, not spacings. `None` when the
/// lines are too few to trust.
fn median_line_gap(lines: &[f64]) -> Option<f64> {
    if lines.len() < 4 {
        return None;
    }
    let mut ys = lines.to_vec();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut gaps: Vec<f64> = ys
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d > 1.0)
        .collect();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(gaps[gaps.len() / 2])
}

fn context_from(entry: &StaffEntry, fallback_spacing: f64) -> StaffContext {
    // The line gaps are the truth: layout-declared spacings have been
    // observed far from the lines they accompany.
    let spacing = median_line_gap(&entry.staff_lines)
        .or(entry.staff_spacing)
        .unwrap_or(fallback_spacing);
    StaffContext {
        lines: entry.staff_lines.clone(),
        spacing,
    }
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

fn run(args: &Args) -> Result<(), String> {
    if args.min_area == 0 {
        return Err("--min-area must be at least 1".to_string());
    }
    let allowed: Vec<String> = args
        .labels
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if allowed.is_empty() {
        return Err("--labels must list at least one label".to_string());
    }

    // Select the backend up front so flag errors surface before any imaging.
    let detector: Box<dyn GlyphDetector> = match args.backend.as_str() {
        "classical" => Box::new(ClassicalStubDetector {
            threshold: args.threshold,
            min_area: args.min_area,
            allowed,
        }),
        "onnx" => {
            let model = args
                .model
                .clone()
                .ok_or_else(|| "--backend onnx requires --model <model.onnx>".to_string())?;
            Box::new(OnnxDetectorHook::load(
                model,
                args.threshold,
                args.min_area,
            )?)
        }
        "heatmap" => {
            let model = args
                .model
                .clone()
                .ok_or_else(|| "--backend heatmap requires --model <heatnet.onnx>".to_string())?;
            Box::new(HeatmapDetector::load(
                model,
                args.heat_threshold,
                args.heat_threshold_beam,
                allowed,
            )?)
        }
        other => {
            return Err(format!(
                "unknown --backend `{other}`: expected `classical`, `onnx` or `heatmap`"
            ));
        }
    };

    let crops = load_crops(&args.crops_dir)?;
    let entries = load_staff_map(&args.staff_map)?;

    // Optional second-model evidence (e.g. learned flags backing a
    // classical primary). Loaded up front so flag errors surface early.
    #[cfg(feature = "onnx")]
    let evidence: Option<EvidenceModel> = match &args.evidence_model {
        None => None,
        Some(path) => {
            let labels: Vec<String> = args
                .evidence_labels
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            if labels.is_empty() {
                return Err("--evidence-labels must list at least one label".to_string());
            }
            Some(EvidenceModel {
                model: onnx_patch::OnnxPatchModel::load(path)
                    .map_err(|e| format!("evidence model {}: {e}", path.display()))?,
                labels,
            })
        }
    };
    #[cfg(not(feature = "onnx"))]
    if args.evidence_model.is_some() {
        return Err("--evidence-model requires rebuilding with `--features onnx`".to_string());
    }

    let mut results = Vec::with_capacity(crops.len());
    let mut total_glyphs = 0usize;
    for crop in &crops {
        let ctx = resolve_context(&entries, crop, args.staff_spacing)?;
        #[cfg_attr(not(feature = "onnx"), allow(unused_mut))]
        let mut glyphs = detector.detect(crop, &ctx)?;
        #[cfg(feature = "onnx")]
        if let Some(ev) = &evidence {
            glyphs.extend(evidence_glyphs(
                &ev.model,
                &crop.gray,
                args.threshold,
                args.min_area,
                &ev.labels,
                &ctx,
            ));
            glyphs.sort_by_key(|g| (g.bbox.y, g.bbox.x));
        }
        total_glyphs += glyphs.len();
        results.push(CropResult {
            crop_id: crop.id.clone(),
            source: crop.file_name.clone(),
            width: crop.gray.width(),
            height: crop.gray.height(),
            glyphs,
        });
    }

    let output = Output {
        backend: detector.name().to_string(),
        crops: results,
    };
    let text = serde_json::to_string_pretty(&output)
        .map_err(|e| format!("cannot serialize output: {e}"))?;
    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
    }
    fs::write(&args.out, text).map_err(|e| format!("cannot write {}: {e}", args.out.display()))?;

    println!(
        "symbol-detect: backend={} crops={} glyphs={} -> {}",
        detector.name(),
        crops.len(),
        total_glyphs,
        args.out.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("symbol-detect: error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    fn detector() -> ClassicalStubDetector {
        ClassicalStubDetector {
            threshold: 128,
            min_area: 4,
            allowed: vec![
                "notehead-black".to_string(),
                "notehead-void".to_string(),
                "stem".to_string(),
                "beam".to_string(),
                "dot".to_string(),
                "accidental".to_string(),
                "clef".to_string(),
                "rest".to_string(),
            ],
        }
    }

    fn ctx() -> StaffContext {
        StaffContext {
            lines: vec![10.0, 20.0, 30.0, 40.0, 50.0],
            spacing: 10.0,
        }
    }

    fn unk(x: u32, y: u32, w: u32, h: u32, fill: f64) -> Glyph {
        Glyph {
            bbox: BBox { x, y, w, h },
            label: "unknown".to_string(),
            confidence: 0.4,
            staff_step: None,
            area: (fill * w as f64 * h as f64).round() as usize,
        }
    }

    #[test]
    fn rescue_relabels_rest_shaped_unknowns() {
        // sp = 10: eighth slash (11x19), half bar (13x6), quarter squiggle.
        let gs = vec![
            unk(0, 0, 11, 19, 0.4),
            unk(20, 0, 13, 6, 0.7),
            unk(40, 0, 9, 28, 0.4),
            Glyph {
                bbox: BBox {
                    x: 60,
                    y: 0,
                    w: 10,
                    h: 9,
                },
                label: "notehead-black".to_string(),
                confidence: 0.9,
                staff_step: None,
                area: 70,
            },
        ];
        let out = rescue_rests(gs, &ctx());
        assert_eq!(out[0].label, "rest");
        assert_eq!(out[1].label, "rest");
        assert_eq!(out[2].label, "rest");
        assert_eq!(out[3].label, "notehead-black");
    }

    #[test]
    fn rescue_keeps_lyric_bits_and_slurs_unknown() {
        // Small letter fragment and a long thin slur: not rest-shaped.
        let gs = vec![unk(0, 12, 6, 6, 0.55), unk(20, 22, 40, 4, 0.25)];
        let out = rescue_rests(gs, &ctx());
        assert!(out.iter().all(|g| g.label == "unknown"));
    }

    #[test]
    fn rescue_ignores_rest_shaped_blobs_far_from_staves() {
        // Eighth-sized lyric fragment ~3sp below the lowest line: lyrics,
        // dynamics, and page furniture live outside the staff band.
        let gs = vec![unk(0, 100, 11, 19, 0.4)];
        let out = rescue_rests(gs, &ctx());
        assert_eq!(out[0].label, "unknown");
    }

    #[test]
    fn rescue_ignores_gap_junk_between_staves() {
        // Two staves 25px apart (gap ~2.5sp): an eighth-shaped fragment in
        // the middle of the gap is near lines of both staves but inside
        // neither staff's span, so it stays unknown.
        let ctx = StaffContext {
            lines: vec![10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 85.0, 95.0, 105.0, 115.0],
            spacing: 10.0,
        };
        let gs = vec![unk(0, 53, 11, 19, 0.4), unk(200, 86, 11, 19, 0.4)];
        let out = rescue_rests(gs, &ctx);
        assert_eq!(out[0].label, "unknown");
        assert_eq!(out[1].label, "rest");
    }

    #[test]
    fn rescue_needs_positive_spacing() {
        let gs = vec![unk(0, 0, 11, 19, 0.4)];
        let no_sp = StaffContext {
            lines: vec![],
            spacing: 0.0,
        };
        let out = rescue_rests(gs, &no_sp);
        assert_eq!(out[0].label, "unknown");
    }

    /// White canvas with one filled black rectangle (x0..x1, y0..y1).
    fn rect_image(w: u32, h: u32, rects: &[(u32, u32, u32, u32)]) -> GrayImage {
        let mut img = GrayImage::from_pixel(w, h, image::Luma([255]));
        for &(x0, y0, x1, y1) in rects {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    img.put_pixel(x, y, image::Luma([0]));
                }
            }
        }
        img
    }

    fn frame(id: &str, gray: GrayImage) -> CropFrame {
        CropFrame {
            id: id.to_string(),
            file_name: format!("{id}.png"),
            gray,
        }
    }

    #[test]
    fn components_find_two_blobs_with_boxes_in_scan_order() {
        // Two separated squares: upper-right blob must sort after upper-left.
        let gray = rect_image(40, 20, &[(20, 2, 25, 7), (2, 2, 7, 7)]);
        let comps = connected_components(&gray, 128, 4);
        assert_eq!(comps.len(), 2);
        assert_eq!((comps[0].0.x, comps[0].0.y), (2, 2));
        assert_eq!((comps[1].0.x, comps[1].0.y), (20, 2));
        assert_eq!(comps[0].1, 36);
    }

    #[test]
    fn components_drop_blobs_under_min_area() {
        let gray = rect_image(20, 20, &[(2, 2, 3, 3), (10, 10, 15, 15)]);
        assert_eq!(connected_components(&gray, 128, 4).len(), 2);
        assert_eq!(connected_components(&gray, 128, 5).len(), 1);
    }

    #[test]
    fn classify_maps_shape_to_glyph_label() {
        let d = detector();
        let c = ctx();
        // 10x10 filled square at spacing 10 -> notehead-black.
        let bb = |w, h| BBox { x: 0, y: 0, w, h };
        assert_eq!(d.classify(&bb(10, 10), 100, &c), "notehead-black");
        // 10x10 ring (low fill) -> notehead-void.
        assert_eq!(d.classify(&bb(10, 10), 20, &c), "notehead-void");
        // Tall narrow bar -> stem.
        assert_eq!(d.classify(&bb(4, 40), 160, &c), "stem");
        // Wide short bar -> beam.
        assert_eq!(d.classify(&bb(30, 8), 240, &c), "beam");
        // Tiny blob -> dot.
        assert_eq!(d.classify(&bb(5, 5), 25, &c), "dot");
        // Tall wide block -> clef.
        assert_eq!(d.classify(&bb(20, 40), 400, &c), "clef");
        // Mid-size block -> accidental.
        assert_eq!(d.classify(&bb(15, 25), 200, &c), "accidental");
    }

    #[test]
    fn classify_falls_back_to_unknown_outside_vocabulary() {
        let mut d = detector();
        d.allowed = vec!["stem".to_string()];
        let c = ctx();
        assert_eq!(
            d.classify(
                &BBox {
                    x: 0,
                    y: 0,
                    w: 10,
                    h: 10
                },
                100,
                &c
            ),
            "unknown"
        );
        assert_eq!(
            d.classify(
                &BBox {
                    x: 0,
                    y: 0,
                    w: 4,
                    h: 40
                },
                160,
                &c
            ),
            "stem"
        );
    }

    #[test]
    fn detect_is_deterministic_and_snaps_to_staff() {
        let d = detector();
        let c = ctx();
        let crop = frame("m1", rect_image(60, 60, &[(5, 8, 14, 17)]));
        let first = d.detect(&crop, &c).unwrap();
        let second = d.detect(&crop, &c).unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(
            serde_json::to_string(&first).unwrap(),
            serde_json::to_string(&second).unwrap()
        );
        // Blob center y = 13.0; lines start at 10 with spacing 10 ->
        // (13.0 - 10) / 5 = 0.6 half-steps, snapped to 0.5.
        assert_eq!(first[0].staff_step, Some(0.5));
        assert_eq!(first[0].label, "notehead-black");
    }

    #[test]
    fn snap_returns_none_without_staff_lines() {
        let c = StaffContext {
            lines: vec![],
            spacing: 10.0,
        };
        assert_eq!(snap_to_staff(25.0, &c), None);
    }

    /// Black canvas with filled ellipses (cx, cy, rx, ry) plus bars.
    fn stacked_heads_image(
        w: u32,
        h: u32,
        ellipses: &[(u32, u32, u32, u32)],
        bars: &[(u32, u32, u32, u32)],
    ) -> GrayImage {
        let mut img = GrayImage::from_pixel(w, h, image::Luma([255]));
        for &(cx, cy, rx, ry) in ellipses {
            for y in cy.saturating_sub(ry)..=(cy + ry).min(h - 1) {
                for x in cx.saturating_sub(rx)..=(cx + rx).min(w - 1) {
                    let dx = (x as f64 - cx as f64) / rx as f64;
                    let dy = (y as f64 - cy as f64) / ry as f64;
                    if dx * dx + dy * dy <= 1.0 {
                        img.put_pixel(x, y, image::Luma([0]));
                    }
                }
            }
        }
        for &(x0, y0, x1, y1) in bars {
            for y in y0..=y1.min(h - 1) {
                for x in x0..=x1.min(w - 1) {
                    img.put_pixel(x, y, image::Luma([0]));
                }
            }
        }
        img
    }

    fn is_white(img: &GrayImage, x: u32, y: u32) -> bool {
        img.get_pixel(x, y).0[0] >= 128
    }

    fn head_glyph(x: u32, y: u32, w: u32, h: u32, label: &str) -> Glyph {
        Glyph {
            bbox: BBox { x, y, w, h },
            label: label.to_string(),
            confidence: 0.9,
            staff_step: None,
            area: (w * h) as usize,
        }
    }

    #[test]
    fn splitter_divides_stacked_third_into_two_heads() {
        // Two heads a third apart (centers 10px = 1 spacing) plus a stem.
        let gray = stacked_heads_image(
            40,
            60,
            &[(20, 20, 5, 4), (20, 30, 5, 4)],
            &[(23, 8, 25, 44)],
        );
        let c = ctx();
        let g = head_glyph(15, 8, 11, 37, "notehead-black");
        let out = split_chord_blobs(&gray, 128, vec![g], &c);
        assert_eq!(out.len(), 2, "third stack must split in two");
        assert!(out.iter().all(|g| g.label == "notehead-black"));
        // Centers land on the two head centers: steps 2.0 and 4.0.
        let mut steps: Vec<f64> = out.iter().map(|g| g.staff_step.unwrap()).collect();
        steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(steps, vec![2.0, 4.0]);
    }

    #[test]
    fn splitter_anchors_boundary_to_neck_between_heads() {
        // Two heads a third apart (true steps 2.0 and 4.0) with a beam
        // block hanging off the bottom head: the extra wide rows drag the
        // equal-count quantile boundary into the lower head, so the
        // fragment centers drift to steps 2.5/4.5. Anchoring the boundary
        // to the narrow neck row (the stem-only gap) restores 2.0/4.0.
        let gray = stacked_heads_image(
            40,
            60,
            &[(20, 20, 5, 4), (20, 30, 5, 4)],
            &[(23, 8, 25, 44), (15, 32, 30, 35)],
        );
        let c = ctx();
        let out = split_chord_blobs(
            &gray,
            128,
            vec![head_glyph(15, 14, 16, 27, "notehead-black")],
            &c,
        );
        assert_eq!(out.len(), 2, "beam-dragged stack must still split in two");
        let mut steps: Vec<f64> = out.iter().map(|g| g.staff_step.unwrap()).collect();
        steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(steps, vec![2.0, 4.0]);
    }

    #[test]
    fn splitter_keeps_single_head_with_long_stem() {
        // One head plus a 47px stem: narrow fill, must stay whole.
        let gray = stacked_heads_image(40, 70, &[(20, 20, 5, 4)], &[(23, 8, 25, 55)]);
        let c = ctx();
        let out = split_chord_blobs(
            &gray,
            128,
            vec![head_glyph(15, 8, 11, 48, "notehead-black")],
            &c,
        );
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn splitter_rejects_low_fill_tall_blob() {
        // Two thin clef-like strokes: tall but sparse, must stay whole.
        let gray = stacked_heads_image(40, 60, &[], &[(18, 8, 19, 50), (30, 8, 31, 50)]);
        let c = ctx();
        let out = split_chord_blobs(
            &gray,
            128,
            vec![head_glyph(18, 8, 14, 43, "notehead-void")],
            &c,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].label, "notehead-void");
    }

    #[test]
    fn splitter_skips_wide_and_short_blobs() {
        // Wide 2nd-style blob and short single head pass through untouched.
        let gray = stacked_heads_image(60, 60, &[(20, 20, 5, 4), (32, 20, 5, 4)], &[]);
        let c = ctx();
        let wide = head_glyph(14, 15, 25, 11, "notehead-black");
        let short = head_glyph(15, 16, 11, 9, "notehead-black");
        let out = split_chord_blobs(&gray, 128, vec![wide, short], &c);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn staff_map_accepts_wrapped_bare_and_crop_alias() {
        let dir = std::env::temp_dir().join("symbol-detect-test");
        fs::create_dir_all(&dir).unwrap();
        let wrapped = dir.join("wrapped.json");
        fs::write(
            &wrapped,
            "\u{FEFF}{\"systems\": [{\"crop\": \"m1\", \"staff_lines\": [1.0], \"staff_spacing\": 8.0}]}",
        )
        .unwrap();
        let entries = load_staff_map(&wrapped).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].crop_id.as_deref(), Some("m1"));
        let bare = dir.join("bare.json");
        fs::write(&bare, "[{\"crop_id\": \"m2\", \"staff_lines\": []}]").unwrap();
        assert_eq!(load_staff_map(&bare).unwrap().len(), 1);
        fs::remove_file(&wrapped).unwrap();
        fs::remove_file(&bare).unwrap();
    }

    #[test]
    fn context_prefers_median_line_gap_over_declared_spacing() {
        // Neural layouts can declare a wrong global spacing (e.g. 9px when
        // the lines sit 13px apart): the line gaps are the truth.
        let entry = StaffEntry {
            id: None,
            crop_id: None,
            staff_lines: vec![
                0.0, 13.0, 26.0, 39.0, 52.0, 200.0, 213.0, 226.0, 239.0, 252.0,
            ],
            staff_spacing: Some(9.0),
        };
        assert_eq!(context_from(&entry, 10.0).spacing, 13.0);
        // Near-duplicate lines (fitter artifacts) do not drag the median.
        let dup = StaffEntry {
            id: None,
            crop_id: None,
            staff_lines: vec![0.0, 0.2, 13.0, 26.0, 39.0, 52.0],
            staff_spacing: Some(9.0),
        };
        assert_eq!(context_from(&dup, 10.0).spacing, 13.0);
        // Too few lines: declared spacing, then the CLI fallback.
        let one = StaffEntry {
            id: None,
            crop_id: None,
            staff_lines: vec![5.0],
            staff_spacing: Some(9.0),
        };
        assert_eq!(context_from(&one, 10.0).spacing, 9.0);
        let bare = StaffEntry {
            id: None,
            crop_id: None,
            staff_lines: vec![],
            staff_spacing: None,
        };
        assert_eq!(context_from(&bare, 10.0).spacing, 10.0);
    }

    #[test]
    fn resolve_prefers_crop_entry_then_global_then_fallback() {
        let entries = vec![
            StaffEntry {
                id: None,
                crop_id: None,
                staff_lines: vec![0.0],
                staff_spacing: Some(9.0),
            },
            StaffEntry {
                id: None,
                crop_id: Some("m1".to_string()),
                staff_lines: vec![5.0],
                staff_spacing: None,
            },
        ];
        let m1 = frame("m1", GrayImage::new(8, 8));
        let m9 = frame("m9", GrayImage::new(8, 8));
        let c1 = resolve_context(&entries, &m1, 10.0).unwrap();
        assert_eq!(c1.lines, vec![5.0]);
        assert_eq!(c1.spacing, 10.0);
        let c9 = resolve_context(&entries, &m9, 10.0).unwrap();
        assert_eq!(c9.lines, vec![0.0]);
        assert_eq!(c9.spacing, 9.0);
        let c0 = resolve_context(&[], &m9, 10.0).unwrap();
        assert!(c0.lines.is_empty());
    }

    #[test]
    fn void_split_separates_filled_and_hollow() {
        // Filled 12x12 square: central third all ink -> black.
        let filled = rect_image(20, 20, &[(4, 4, 15, 15)]);
        let bb = BBox {
            x: 4,
            y: 4,
            w: 12,
            h: 12,
        };
        assert!(!is_void_notehead(&filled, &bb, 128));
        // Same square with a white 6x6 hole: central third mostly white.
        let mut hollow = filled;
        for y in 7..13 {
            for x in 7..13 {
                hollow.put_pixel(x, y, image::Luma([255]));
            }
        }
        assert!(is_void_notehead(&hollow, &bb, 128));
        // Degenerate empty center (1px blob) never counts as void.
        let tiny = rect_image(8, 8, &[(4, 4, 4, 4)]);
        let tb = BBox {
            x: 4,
            y: 4,
            w: 1,
            h: 1,
        };
        assert!(!is_void_notehead(&tiny, &tb, 128));
    }

    /// Filled ellipse rasterizer for synthetic notehead probes.
    #[cfg(feature = "onnx")]
    fn ellipse_image(w: u32, h: u32, cx: f32, cy: f32, rx: f32, ry: f32, thick: f32) -> GrayImage {
        let mut img = GrayImage::from_pixel(w, h, image::Luma([255]));
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f32 - cx) / rx).powi(2) + ((y as f32 - cy) / ry).powi(2);
                let edge = (d.sqrt() - 1.0).abs() * rx.min(ry);
                if d <= 1.0 && (thick <= 0.0 || edge <= thick) {
                    img.put_pixel(x, y, image::Luma([0]));
                }
            }
        }
        img
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn onnx_hook_without_feature_reports_everything() {
        assert!(OnnxDetectorHook::load(PathBuf::from("no-such-model.onnx"), 128, 16).is_err());
        let dir = std::env::temp_dir().join("symbol-detect-test");
        fs::create_dir_all(&dir).unwrap();
        let model = dir.join("stub.onnx");
        fs::write(&model, b"fake").unwrap();
        // Without the feature even a present file cannot load a session.
        let err = match OnnxDetectorHook::load(model.clone(), 128, 16) {
            Ok(_) => panic!("load must fail without the onnx feature"),
            Err(e) => e,
        };
        assert!(err.contains("requires rebuilding"), "{err}");
        fs::remove_file(&model).unwrap();
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn onnx_hook_still_rejects_missing_model_file() {
        assert!(OnnxDetectorHook::load(PathBuf::from("no-such-model.onnx"), 128, 16).is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn onnx_hook_classifies_synthetic_notehead_and_rest() {
        let hook = OnnxDetectorHook::load(PathBuf::from("tests/glyphnet.onnx"), 128, 16)
            .expect("fixture loads");
        assert_eq!(hook.name(), "onnx-hook");
        let c = ctx();
        // Hollow ellipse (void notehead): ring blob, white center.
        let ring = frame("ring", ellipse_image(40, 30, 20.0, 15.0, 10.0, 7.0, 2.0));
        let glyphs = hook.detect(&ring, &c).expect("detect runs");
        assert_eq!(glyphs.len(), 1, "one ring blob, got {glyphs:?}");
        assert_eq!(glyphs[0].label, "notehead-void");
        assert!(glyphs[0].confidence > 0.5, "{}", glyphs[0].confidence);
        // Small filled square reads as a rest.
        let sq = frame("sq", rect_image(24, 24, &[(8, 8, 15, 15)]));
        let glyphs = hook.detect(&sq, &c).expect("detect runs");
        assert_eq!(glyphs.len(), 1);
        assert_eq!(glyphs[0].label, "rest");
    }

    #[test]
    fn label_for_class_maps_fifth_class_to_flag() {
        // 5-class PatchNet (notehead / rest / beam / flag / other) reports
        // flag boxes; the hook must surface class 4 as `flag`.
        // Class 0 stays None (void/black split is a pixel rule at the call
        // site); legacy guards stay: `beam` needs 4+ logits.
        assert_eq!(label_for_class(0, 5), None);
        assert_eq!(label_for_class(1, 3), Some("rest"));
        assert_eq!(label_for_class(1, 5), Some("rest"));
        assert_eq!(label_for_class(2, 5), Some("beam"));
        assert_eq!(label_for_class(2, 3), None);
        assert_eq!(label_for_class(4, 5), Some("flag"));
        assert_eq!(label_for_class(4, 4), None);
        assert_eq!(label_for_class(3, 5), None);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn onnx_hook_maps_fourth_class_to_beam() {
        // A 4-class PatchNet (notehead / rest / beam / other) reports beam
        // boxes; the hook must surface class 2 as `beam`, not `unknown`.
        // Fixture `glyphnet_beam4.onnx` always predicts class 2.
        let hook = OnnxDetectorHook::load(PathBuf::from("tests/glyphnet_beam4.onnx"), 128, 16)
            .expect("fixture loads");
        let c = ctx();
        // Wide short bar: beam geometry (label comes from the fixture).
        let bar = frame("bar", rect_image(40, 24, &[(4, 10, 36, 14)]));
        let glyphs = hook.detect(&bar, &c).expect("detect runs");
        assert_eq!(glyphs.len(), 1, "one bar blob, got {glyphs:?}");
        assert_eq!(glyphs[0].label, "beam");
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn evidence_model_appends_listed_labels() {
        // `glyphnet_flag5.onnx` always predicts class 4: with "flag" in the
        // evidence set the bar blob gains a `flag` entry; with only "beam"
        // listed nothing is appended.
        let model = onnx_patch::OnnxPatchModel::load(Path::new("tests/glyphnet_flag5.onnx"))
            .expect("fixture loads");
        // Wide short bar: one blob for the evidence model to label.
        let gray = rect_image(40, 24, &[(4, 10, 36, 14)]);
        let c = ctx();
        let flags = evidence_glyphs(&model, &gray, 128, 16, &["flag".to_string()], &c);
        assert_eq!(flags.len(), 1, "one bar blob, got {flags:?}");
        assert_eq!(flags[0].label, "flag");
        let beams = evidence_glyphs(&model, &gray, 128, 16, &["beam".to_string()], &c);
        assert!(beams.is_empty(), "flag prediction is not beam evidence");
    }

    #[test]
    fn staff_removal_whitens_bare_line_pixels() {
        // 2px staff line, no symbols: every line pixel has a vertical dark
        // run of 2, so the whole line is erased.
        let gray = rect_image(40, 24, &[(0, 10, 39, 11)]);
        let clean = erase_staff_lines(&gray, &[10, 11], 2, 128);
        assert!(is_white(&clean, 5, 10));
        assert!(is_white(&clean, 30, 11));
        assert_eq!(connected_components(&clean, 128, 4).len(), 0);
    }

    #[test]
    fn staff_removal_keeps_head_crossing_line() {
        // Filled head block (11x9) centered on the staff line: the head
        // column's vertical run (9) exceeds the line budget, so the head
        // stays dark through the line while bare line pixels far away are
        // whitened.
        let gray = rect_image(40, 24, &[(0, 10, 39, 11), (15, 6, 25, 14)]);
        let clean = erase_staff_lines(&gray, &[10, 11], 2, 128);
        assert!(is_white(&clean, 2, 10), "bare line far from head erased");
        assert!(!is_white(&clean, 20, 10), "head column kept through line");
        assert!(!is_white(&clean, 20, 11), "head column kept through line");
    }

    #[test]
    fn output_schema_round_trips() {
        let doc = Output {
            backend: "classical-stub".to_string(),
            crops: vec![CropResult {
                crop_id: "m1".to_string(),
                source: "m1.png".to_string(),
                width: 60,
                height: 60,
                glyphs: vec![Glyph {
                    bbox: BBox {
                        x: 5,
                        y: 8,
                        w: 10,
                        h: 10,
                    },
                    label: "notehead-black".to_string(),
                    confidence: 0.95,
                    staff_step: Some(0.5),
                    area: 100,
                }],
            }],
        };
        let text = serde_json::to_string(&doc).unwrap();
        let back: Output = serde_json::from_str(&text).unwrap();
        assert_eq!(back.crops[0].glyphs[0].label, "notehead-black");
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn backend_names_are_stable() {
        assert_eq!(detector().name(), "classical-stub");
        let hook = OnnxDetectorHook {
            model_path: PathBuf::from("m.onnx"),
            threshold: 128,
            min_area: 16,
        };
        assert_eq!(hook.name(), "onnx-hook");
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn backend_names_are_stable() {
        assert_eq!(detector().name(), "classical-stub");
        let hook = OnnxDetectorHook::load(PathBuf::from("tests/glyphnet.onnx"), 128, 16)
            .expect("fixture loads");
        assert_eq!(hook.name(), "onnx-hook");
    }

    #[test]
    fn score_stays_in_range_and_prefers_dense_blobs() {
        let dense = score(
            100,
            &BBox {
                x: 0,
                y: 0,
                w: 10,
                h: 10,
            },
        );
        let sparse = score(
            20,
            &BBox {
                x: 0,
                y: 0,
                w: 10,
                h: 10,
            },
        );
        assert!(dense > sparse);
        assert!((0.5..=0.95).contains(&dense));
        assert!((0.5..=0.95).contains(&sparse));
    }

    #[test]
    fn defaults_match_big500_training() {
        let args = Args::try_parse_from([
            "symbol-detect",
            "--crops-dir",
            "crops",
            "--staff-map",
            "map.json",
        ])
        .expect("defaults parse");
        assert_eq!(args.threshold, 128);
        assert_eq!(args.min_area, 16);
        assert_eq!(args.staff_spacing, 10.0);
    }

    #[test]
    fn heat_tiles_cover_every_pixel() {
        use crate::onnx_heat::tile_origins;
        use crate::onnx_heat::{STRIDE_TILE, TILE};
        for len in [1, 500, 1240, 1241, 2000, 2830, 5000] {
            let xs = tile_origins(len);
            assert_eq!(xs[0], 0, "len {len}");
            let edge = len.saturating_sub(TILE);
            assert_eq!(*xs.last().unwrap(), if len <= TILE { 0 } else { edge });
            for w in xs.windows(2) {
                assert!(w[1] - w[0] <= STRIDE_TILE, "len {len}");
                assert!(w[1] > w[0], "len {len}");
            }
            // Rightmost pixel (len-1) is inside the last tile.
            assert!(*xs.last().unwrap() + TILE >= len, "len {len}");
        }
    }

    #[test]
    fn heat_decode_finds_center_peak() {
        use crate::onnx_heat::decode_tile;
        let (w, h) = (5usize, 5usize);
        let mut heat = vec![-10.0f32; 7 * w * h];
        heat[2 * w + 2] = 2.0; // class 0 peak at (2,2)
        let mut wh = vec![0.0f32; 14 * w * h];
        wh[2 * w + 2] = 0.05;
        wh[1 * w * h + 2 * w + 2] = 0.03;
        let boxes = decode_tile(&heat, &wh, w, h, 0, 0, 0.5, 0.5);
        assert_eq!(boxes.len(), 1);
        let b = &boxes[0];
        assert_eq!(b.class, 0);
        assert!((b.x0 + b.x1) / 2.0 - 5.0 < 1e-4);
        assert!((b.x1 - b.x0) - 62.0 < 1e-3);
        // Sub-threshold heat yields nothing.
        let none = decode_tile(&heat, &wh, w, h, 0, 0, 0.99, 0.99);
        assert!(none.is_empty());
        // A higher neighbor kills the peak.
        let mut heat2 = heat.clone();
        heat2[2 * w + 3] = 3.0;
        let mut wh2 = wh.clone();
        wh2[2 * w + 3] = 0.05;
        wh2[1 * w * h + 2 * w + 3] = 0.03;
        let boxes2 = decode_tile(&heat2, &wh2, w, h, 0, 0, 0.5, 0.5);
        assert_eq!(boxes2.len(), 1);
        assert!((boxes2[0].x0 + boxes2[0].x1) / 2.0 - 7.0 < 1e-4);
    }

    #[test]
    fn heat_decode_beam_uses_own_floor() {
        use crate::onnx_heat::decode_tile;
        let (w, h) = (5usize, 5usize);
        // Beam (class 2) peak at logit 0.4: below the global 0.9 floor,
        // above the 0.3 beam floor.
        let mut heat = vec![-10.0f32; 7 * w * h];
        heat[2 * w * h + 2 * w + 2] = 0.4;
        let mut wh = vec![0.0f32; 14 * w * h];
        wh[2 * 2 * w * h + 2 * w + 2] = 0.05;
        wh[(2 * 2 + 1) * w * h + 2 * w + 2] = 0.03;
        let boxes = decode_tile(&heat, &wh, w, h, 0, 0, 0.9, 0.3);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].class, 2);
        // Below the beam floor (sigmoid(0.4) = 0.599 < 0.7) it is dropped too.
        let none = decode_tile(&heat, &wh, w, h, 0, 0, 0.9, 0.7);
        assert!(none.is_empty());
    }

    #[test]
    fn heat_nms_dedups_same_class_overlap() {
        use crate::onnx_heat::{nms, HeatBox};
        let a = HeatBox {
            x0: 0.0,
            y0: 0.0,
            x1: 10.0,
            y1: 10.0,
            class: 0,
            score: 0.9,
        };
        let b = HeatBox {
            x0: 1.0,
            y0: 1.0,
            x1: 11.0,
            y1: 11.0,
            class: 0,
            score: 0.8,
        };
        let c = HeatBox {
            x0: 1.0,
            y0: 1.0,
            x1: 11.0,
            y1: 11.0,
            class: 1,
            score: 0.7,
        };
        let kept = nms(vec![b.clone(), c.clone(), a.clone()], 0.5);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].score, 0.9);
        assert_eq!(kept[1].class, 1);
    }

    #[test]
    fn heat_label_maps_all_seven_classes() {
        assert_eq!(heat_label(0), "notehead-black");
        assert_eq!(heat_label(1), "rest");
        assert_eq!(heat_label(2), "beam");
        assert_eq!(heat_label(3), "beam");
        assert_eq!(heat_label(4), "accidental");
        assert_eq!(heat_label(5), "dot");
        assert_eq!(heat_label(6), "clef");
        assert_eq!(heat_label(7), "unknown");
    }

    #[test]
    fn heat_detector_rejects_missing_model_file() {
        let res = HeatmapDetector::load(PathBuf::from("nope.onnx"), 0.9, 0.6, Vec::<String>::new());
        match res {
            Ok(_) => panic!("missing model must error"),
            Err(err) => assert!(err.contains("not found"), "{err}"),
        }
    }

    #[test]
    fn heat_threshold_defaults_match_tile_gate() {
        let args = Args::try_parse_from([
            "symbol-detect",
            "--crops-dir",
            "crops",
            "--staff-map",
            "map.json",
        ])
        .expect("defaults parse");
        assert_eq!(args.heat_threshold, 0.9);
        assert_eq!(args.heat_threshold_beam, 0.6);
    }
}
