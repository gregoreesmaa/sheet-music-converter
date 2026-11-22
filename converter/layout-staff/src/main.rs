//! layout-staff: cleaned sheet-music image -> systems/staves JSON +
//! staff-probability map PNG.
//!
//! Classical fallback: binarize the cleaned (dark-ink-on-light-paper) image,
//! take the per-row ink-fill profile, pick staff-line rows with local-maximum
//! peak detection, group rows into 5-line staves, then group nearby staves
//! into systems. A per-row staff-probability map (fill profile plus Gaussian
//! bumps at detected lines) is written next to the JSON.
//!
//! Neural hook: [`StaffLineModel`] abstracts the "image -> per-row staff
//! probability" step. [`ClassicalModel`] implements it with the projection
//! method above. A learned backend (e.g. ONNX/Tch) plugs in by implementing
//! the same trait and adding a `--model` arm in [`select_model`].
//!
//! Usage:
//!   layout-staff --input clean.png --out-dir out
//!   layout-staff --input clean-dir/ --out-dir out --threshold 140 --min-fill 0.3

use clap::Parser;
use image::GrayImage;
use serde::Serialize;
use std::path::{Path, PathBuf};

/// Detect staff systems/staves in cleaned sheet-music images.
#[derive(Parser, Debug)]
#[command(name = "layout-staff", version)]
struct Args {
    /// Input cleaned PNG/JPEG file, or a directory of images (non-recursive).
    #[arg(long)]
    input: PathBuf,

    /// Directory that receives `<stem>_layout.json` + `<stem>_staffprob.png`.
    #[arg(long, default_value = "out")]
    out_dir: PathBuf,

    /// Binarization threshold: pixels darker than this count as ink (0..255).
    /// No effect on two-valued ingest-binarized input (pixels already 0/255).
    #[arg(long, default_value_t = 140)]
    threshold: u8,

    /// Minimum row ink-fill fraction for a staff-line candidate (0..1].
    /// Re-trained on 48 big500 photos: 0.10 leaves zero empty pages while
    /// 0.15 misses whole systems on dense scores (3/24 empty). Recall is
    /// favored over precision: a missed system loses notation entirely,
    /// fragments keep low confidence downstream.
    #[arg(long, default_value_t = 0.10)]
    min_fill: f32,

    /// Peak-picking floor for neural row probabilities (0..1]. Classical
    /// `min_fill` is an ink-fill fraction and does not transfer: on 30
    /// exact-truth pages 0.10 over-detects (sparse-page precision ~0.3),
    /// while 0.3 keeps recall and roughly doubles precision.
    #[arg(long, default_value_t = 0.30)]
    neural_threshold: f32,

    /// Merge peak rows at most this far apart (px) into one staff line.
    #[arg(long, default_value_t = 3)]
    merge_px: u32,

    /// Max inter-staff gap, in staff spaces, before a new system starts.
    #[arg(long, default_value_t = 4.0)]
    system_gap_spaces: f32,

    /// Detector backend: `classical` (ink-fill projection) or `neural`
    /// (ONNX row model via `--weights`; requires the `onnx` feature).
    /// New neural backends get an arm in `select_model`.
    #[arg(long, default_value = "classical")]
    model: String,

    /// Row-model weight file for `--model neural` (e.g. `rownet_pilot.onnx`).
    /// Required with a neural backend, rejected with `classical`.
    #[arg(long)]
    weights: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(&args) {
        eprintln!("layout-staff: error: {e}");
        std::process::exit(1);
    }
}

#[cfg(feature = "onnx")]
mod onnx_model;

fn run(args: &Args) -> Result<(), String> {
    if !(0.0 < args.min_fill && args.min_fill <= 1.0) {
        return Err(format!(
            "--min-fill must be in (0, 1], got {}",
            args.min_fill
        ));
    }
    if !(0.0 < args.neural_threshold && args.neural_threshold <= 1.0) {
        return Err(format!(
            "--neural-threshold must be in (0, 1], got {}",
            args.neural_threshold
        ));
    }
    if args.merge_px < 1 {
        return Err("--merge-px must be >= 1".to_string());
    }
    if args.system_gap_spaces <= 0.0 {
        return Err("--system-gap-spaces must be > 0".to_string());
    }
    if args.model != "neural" && args.weights.is_some() {
        return Err(
            "--weights requires a neural --model backend, which is not compiled in".to_string(),
        );
    }
    let model = select_model(&args.model, args.weights.as_deref())?;
    let params = DetectParams {
        threshold: args.threshold,
        min_fill: args.min_fill,
        neural_threshold: args.neural_threshold,
        merge_px: args.merge_px,
        system_gap_spaces: args.system_gap_spaces,
    };
    let inputs = collect_inputs(&args.input)?;
    if inputs.is_empty() {
        return Err(format!("no image files found at {}", args.input.display()));
    }
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|e| format!("cannot create {}: {e}", args.out_dir.display()))?;
    for path in &inputs {
        let gray = load_gray(path)?;
        let layout = detect_with_model(&gray, model.as_ref(), &params);
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("bad file name: {}", path.display()))?;
        let json_name = format!("{stem}_layout.json");
        let map_name = format!("{stem}_staffprob.png");
        let json_path = args.out_dir.join(&json_name);
        let map_path = args.out_dir.join(&map_name);
        let record = layout.to_json(
            path.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            model.name(),
            &params,
        );
        let json = serde_json::to_string_pretty(&record)
            .map_err(|e| format!("cannot serialize layout: {e}"))?;
        std::fs::write(&json_path, json + "\n")
            .map_err(|e| format!("cannot write {}: {e}", json_path.display()))?;
        layout
            .prob_image()
            .save(&map_path)
            .map_err(|e| format!("cannot write {}: {e}", map_path.display()))?;
        println!("{} -> {}", path.display(), json_path.display());
        println!("  probmap -> {}", map_path.display());
    }
    Ok(())
}

/// Pick the detector backend by name. New neural backends get an arm here.
fn select_model(name: &str, weights: Option<&Path>) -> Result<Box<dyn StaffLineModel>, String> {
    match name {
        "classical" => Ok(Box::new(ClassicalModel)),
        "neural" => {
            #[cfg(feature = "onnx")]
            {
                let w = weights
                    .ok_or("--model neural requires --weights <row-model.onnx>".to_string())?;
                Ok(Box::new(onnx_model::OnnxRowModel::load(w)?))
            }
            #[cfg(not(feature = "onnx"))]
            {
                let _ = weights;
                Err("unknown model 'neural'; rebuild with --features onnx \
                     (implement StaffLineModel for a neural backend to add more)"
                    .to_string())
            }
        }
        other => Err(format!(
            "unknown model '{other}'; this build only includes 'classical' \
             (implement StaffLineModel for a neural backend to add more)"
        )),
    }
}

fn collect_inputs(input: &Path) -> Result<Vec<PathBuf>, String> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    if input.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(input)
            .map_err(|e| format!("cannot read {}: {e}", input.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                        e.eq_ignore_ascii_case("png")
                            || e.eq_ignore_ascii_case("jpg")
                            || e.eq_ignore_ascii_case("jpeg")
                    })
            })
            .collect();
        files.sort();
        return Ok(files);
    }
    Err(format!("input not found: {}", input.display()))
}

fn load_gray(path: &Path) -> Result<GrayImage, String> {
    let img = image::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(img.to_luma8())
}

// ---------------------------------------------------------------------------
// Detector core: StaffLineModel hook + classical fallback.
// ---------------------------------------------------------------------------

/// Tuning knobs shared by every [`StaffLineModel`] backend.
#[derive(Clone, Copy, Debug)]
struct DetectParams {
    threshold: u8,
    min_fill: f32,
    neural_threshold: f32,
    merge_px: u32,
    system_gap_spaces: f32,
}

/// Hook for plugging in a learned staff-line detector.
///
/// A neural backend returns a per-row staff-line probability in [0, 1] for
/// the image (one value per row, length == image height). The downstream
/// grouping (rows -> lines -> staves -> systems) is backend-agnostic, so a
/// new model only needs this one method plus an arm in [`select_model`].
/// `params.threshold` is available for backends that still need an
/// ink/background split (e.g. hybrid pre-processing).
trait StaffLineModel {
    fn name(&self) -> &'static str;
    fn predict_rows(&self, gray: &GrayImage, params: &DetectParams) -> Vec<f32>;
    /// Peak-picking floor for this backend's score scale. Classical scores
    /// are ink-fill fractions (`min_fill`); neural scores are probabilities.
    fn line_threshold(&self, params: &DetectParams) -> f32 {
        params.min_fill
    }
}

/// Classical projection fallback: each row's score is its ink-fill fraction
/// (dark pixels / width). Staff lines span the page, so they stand out as
/// sharp full-width peaks against sparser notation rows.
struct ClassicalModel;

impl StaffLineModel for ClassicalModel {
    fn name(&self) -> &'static str {
        "classical"
    }

    fn predict_rows(&self, gray: &GrayImage, params: &DetectParams) -> Vec<f32> {
        row_fill_profile(gray, params.threshold)
    }
}

/// Per-row ink-fill fraction in [0, 1].
fn row_fill_profile(gray: &GrayImage, threshold: u8) -> Vec<f32> {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let raw = gray.as_raw();
    let mut fill = vec![0.0f32; h];
    for (y, row) in fill.iter_mut().enumerate() {
        let off = y * w;
        let ink = raw[off..off + w].iter().filter(|&&v| v < threshold).count();
        *row = ink as f32 / w as f32;
    }
    fill
}

/// Pick staff-line rows: local maxima of the row-score profile at or above
/// `min_fill` (`>=` on the left, `>` on the right so plateaus count once),
/// then merge peaks within `merge_px` into one line at the fill-weighted
/// centroid. Returns `(center_row, fill)` pairs sorted by row.
///
/// NOTE: valley-aware bridging (merging twin peaks past `merge_px` when the
/// valley between them stays shallow) was tried and reverted: on dense
/// pages the whole fill/prob profile floats high, so every adjacent peak
/// pair qualifies and the full page chains into a single line (classical:
/// 1 line/page; neural pilot: 4 lines/page). Twin-peak doublets are real
/// (neural emits ~2.4x lines) but only a sharper row model (AUC 0.67 now)
/// fixes them; grouping must not assume deep inter-line valleys.
fn pick_lines(scores: &[f32], min_fill: f32, merge_px: u32) -> Vec<(u32, f32)> {
    let n = scores.len();
    let mut peaks: Vec<usize> = Vec::new();
    for y in 0..n {
        if scores[y] < min_fill {
            continue;
        }
        let left_ok = y == 0 || scores[y] >= scores[y - 1];
        let right_ok = y + 1 == n || scores[y] > scores[y + 1];
        if left_ok && right_ok {
            peaks.push(y);
        }
    }
    let mut lines: Vec<(u32, f32)> = Vec::new();
    let mut i = 0;
    while i < peaks.len() {
        let mut j = i;
        while j + 1 < peaks.len() && peaks[j + 1] - peaks[j] <= merge_px as usize {
            j += 1;
        }
        let (num, den) = peaks[i..=j]
            .iter()
            .fold((0.0f64, 0.0f64), |(num, den), &y| {
                (num + y as f64 * scores[y] as f64, den + scores[y] as f64)
            });
        let center = if den > 0.0 {
            num / den
        } else {
            peaks[i] as f64
        };
        let fill = peaks[i..=j]
            .iter()
            .map(|&y| scores[y])
            .fold(0.0f32, f32::max);
        lines.push((center.round() as u32, fill));
        i = j + 1;
    }
    lines
}

/// Median consecutive-line gap, restricted to plausible staff spaces
/// (below 1/8 of the page height) so inter-staff/system gaps do not skew it.
/// Falls back to a height-proportional guess when fewer than 2 lines exist.
fn estimate_staff_space(lines: &[(u32, f32)], height: u32) -> f32 {
    let mut gaps: Vec<u32> = lines
        .windows(2)
        .map(|w| w[1].0.saturating_sub(w[0].0))
        .filter(|&g| g > 0)
        .collect();
    if gaps.is_empty() {
        return (height as f32 / 110.0).clamp(4.0, 40.0);
    }
    let cap = (height / 8).max(4);
    let mut plausible: Vec<u32> = gaps.drain(..).filter(|&g| g <= cap).collect();
    if plausible.is_empty() {
        return cap as f32;
    }
    plausible.sort_unstable();
    let m = plausible.len() / 2;
    if plausible.len() % 2 == 1 {
        plausible[m] as f32
    } else {
        (plausible[m - 1] + plausible[m]) as f32 / 2.0
    }
}

struct Staff {
    lines: Vec<(u32, f32)>,
    bbox: [u32; 4], // [x_min, y_min, x_max, y_max), y clamped to image
    confidence: f32,
    complete: bool, // exactly 5 lines
}

struct System {
    bbox: [u32; 4],
    staves: Vec<Staff>,
    /// Barline x-positions (image pixels) spanning this system's staves.
    barlines: Vec<u32>,
}

struct Layout {
    width: u32,
    height: u32,
    staff_space_px: f32,
    row_scores: Vec<f32>,
    systems: Vec<System>,
    line_count: usize,
}

/// Run a [`StaffLineModel`] backend, then group rows -> lines -> staves ->
/// systems. Never fails: a blank page yields zero systems, not an error.
fn detect_with_model(
    gray: &GrayImage,
    model: &dyn StaffLineModel,
    params: &DetectParams,
) -> Layout {
    let (w, h) = (gray.width(), gray.height());
    let scores = model.predict_rows(gray, params);
    assert_eq!(
        scores.len(),
        h as usize,
        "model must return one score per row"
    );
    let lines = pick_lines(&scores, model.line_threshold(params), params.merge_px);
    let space = estimate_staff_space(&lines, h);
    let staves = group_into_staves(&lines, gray, params, space);
    let mut systems = group_into_systems(staves, space, params.system_gap_spaces);
    let barlines = detect_barlines(gray, &systems, params.threshold);
    for (sys, xs) in systems.iter_mut().zip(barlines) {
        sys.barlines = xs;
    }
    Layout {
        width: w,
        height: h,
        staff_space_px: (space * 100.0).round() / 100.0,
        row_scores: scores,
        line_count: lines.len(),
        systems,
    }
}

/// Group consecutive lines into staves. A staff continues while the gap stays
/// within 1.8 staff spaces and the staff has fewer than 5 lines; groups with
/// fewer than 3 lines are discarded as noise.
fn group_into_staves(
    lines: &[(u32, f32)],
    gray: &GrayImage,
    params: &DetectParams,
    space: f32,
) -> Vec<Staff> {
    let max_gap = (1.8 * space).max(3.0);
    let mut groups: Vec<Vec<(u32, f32)>> = Vec::new();
    for &ln in lines {
        let append = groups
            .last()
            .is_some_and(|g| g.len() < 5 && (ln.0 as f32 - g.last().unwrap().0 as f32) <= max_gap);
        if append {
            groups.last_mut().unwrap().push(ln);
        } else {
            groups.push(vec![ln]);
        }
    }
    groups
        .into_iter()
        .filter(|g| g.len() >= 3)
        .map(|g| {
            let complete = g.len() == 5;
            let avg: f32 = g.iter().map(|&(_, f)| f).sum::<f32>() / g.len() as f32;
            Staff {
                bbox: staff_bbox(&g, gray, params.threshold),
                confidence: ((avg.clamp(0.0, 1.0) * 100.0).round() / 100.0)
                    * if complete { 1.0 } else { 0.6 },
                complete,
                lines: g,
            }
        })
        .collect()
}

/// Tight staff bbox: y from first to last line (padded, clamped), x from the
/// ink extent inside that band (full width when the band holds no ink).
fn staff_bbox(lines: &[(u32, f32)], gray: &GrayImage, threshold: u8) -> [u32; 4] {
    let (w, h) = (gray.width(), gray.height());
    let raw = gray.as_raw();
    let y0 = lines[0].0.saturating_sub(2);
    let y1 = (lines.last().unwrap().0 + 3).min(h);
    let mut x0 = w;
    let mut x1 = 0;
    for y in y0..y1 {
        let off = y as usize * w as usize;
        for (x, &v) in raw[off..off + w as usize].iter().enumerate() {
            if v < threshold && (x as u32) < x0 {
                x0 = x as u32;
            }
            if v < threshold && (x as u32) >= x1 {
                x1 = x as u32 + 1;
            }
        }
    }
    if x1 == 0 {
        x0 = 0;
        x1 = w;
    }
    [x0, y0, x1, y1]
}

/// Group staves into systems: a staff joins the current system while the
/// vertical gap to it stays within `gap_spaces` staff spaces.
fn group_into_systems(staves: Vec<Staff>, space: f32, gap_spaces: f32) -> Vec<System> {
    let max_gap = gap_spaces * space;
    let mut systems: Vec<System> = Vec::new();
    for staff in staves {
        let join = systems
            .last()
            .is_some_and(|s: &System| staff.bbox[1].saturating_sub(s.bbox[3]) as f32 <= max_gap);
        if join {
            let s = systems.last_mut().unwrap();
            s.bbox[0] = s.bbox[0].min(staff.bbox[0]);
            s.bbox[1] = s.bbox[1].min(staff.bbox[1]);
            s.bbox[2] = s.bbox[2].max(staff.bbox[2]);
            s.bbox[3] = s.bbox[3].max(staff.bbox[3]);
            s.staves.push(staff);
        } else {
            let bbox = staff.bbox;
            systems.push(System {
                bbox,
                staves: vec![staff],
                barlines: Vec::new(),
            });
        }
    }
    systems
}

/// Minimum ink coverage of a staff's full line band for a barline column.
/// Staff lines alone contribute ~0.1; stems top out near 0.88 (3.5sp of a
/// 4sp band); a solid barline spans the whole band (~1.0).
const BARLINE_MIN_FILL: f32 = 0.95;
/// Barline run width bounds (px): thinner than noteheads/braces, wider
/// than speckle. Final thin+thick pairs merge below.
const BARLINE_MIN_WIDTH: usize = 1;
const BARLINE_MAX_WIDTH: usize = 6;
/// Runs this close (px) merge: thin+thick final barlines and jitter splits.
const BARLINE_MERGE_PX: u32 = 8;

/// Barline x-positions per system: vertical ink runs spanning a staff's
/// full line band (first to last staff line). Stems (3.5sp), beams/digits
/// (short), and noteheads/braces (wide) all fail either the fill or the
/// width gate. Runs merge within [`BARLINE_MERGE_PX`] across staves, so a
/// grand-staff connector or a coinciding single-staff pair yields one x.
fn detect_barlines(gray: &GrayImage, systems: &[System], threshold: u8) -> Vec<Vec<u32>> {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let raw = gray.as_raw();
    systems
        .iter()
        .map(|sys| {
            let mut xs: Vec<u32> = Vec::new();
            for staff in &sys.staves {
                if staff.lines.len() < 3 {
                    continue;
                }
                let y0 = staff.lines[0].0 as usize;
                let y1 = (staff.lines.last().unwrap().0 as usize).min(h.saturating_sub(1));
                if y1 <= y0 {
                    continue;
                }
                let height = (y1 - y0 + 1) as f32;
                let x0 = (staff.bbox[0] as usize).min(w.saturating_sub(1));
                let x1 = (staff.bbox[2] as usize).clamp(1, w);
                let mut run_start: Option<usize> = None;
                let flush = |start: usize, end: usize, out: &mut Vec<u32>| {
                    let width = end.saturating_sub(start);
                    if (BARLINE_MIN_WIDTH..=BARLINE_MAX_WIDTH).contains(&width) {
                        out.push(((start + end) / 2) as u32);
                    }
                };
                let mut staff_xs: Vec<u32> = Vec::new();
                for x in x0..x1 {
                    let mut ink = 0usize;
                    for y in y0..=y1 {
                        if raw[y * w + x] < threshold {
                            ink += 1;
                        }
                    }
                    if ink as f32 / height >= BARLINE_MIN_FILL {
                        if run_start.is_none() {
                            run_start = Some(x);
                        }
                    } else if let Some(s) = run_start.take() {
                        flush(s, x, &mut staff_xs);
                    }
                }
                if let Some(s) = run_start.take() {
                    flush(s, x1, &mut staff_xs);
                }
                xs.extend(staff_xs);
            }
            xs.sort_unstable();
            let mut merged: Vec<u32> = Vec::new();
            for x in xs {
                if merged
                    .last()
                    .is_some_and(|&last| x - last <= BARLINE_MERGE_PX)
                {
                    let last = merged.last_mut().unwrap();
                    *last = (*last + x) / 2;
                } else {
                    merged.push(x);
                }
            }
            merged
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Output: systems/staves JSON + staff-probability map PNG.
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct JsonStaff {
    id: usize,
    bbox: [u32; 4],
    lines: Vec<u32>,
    confidence: f32,
    complete: bool,
}

#[derive(Serialize)]
struct JsonSystem {
    id: usize,
    bbox: [u32; 4],
    staves: Vec<JsonStaff>,
    /// Barline x-positions (image pixels) for this system.
    barlines: Vec<u32>,
}

#[derive(Serialize)]
struct JsonCounts {
    lines: usize,
    staves: usize,
    systems: usize,
}

#[derive(Serialize)]
struct JsonOutput {
    input: String,
    model: String,
    width: u32,
    height: u32,
    threshold: u8,
    min_fill: f32,
    neural_threshold: f32,
    staff_space_px: f32,
    systems: Vec<JsonSystem>,
    counts: JsonCounts,
}

impl Layout {
    /// Bboxes are `[x_min, y_min, x_max, y_max)` in image pixels.
    fn to_json(&self, input: String, model: &str, params: &DetectParams) -> JsonOutput {
        let systems = self
            .systems
            .iter()
            .enumerate()
            .map(|(sid, s)| JsonSystem {
                id: sid,
                bbox: s.bbox,
                barlines: s.barlines.clone(),
                staves: s
                    .staves
                    .iter()
                    .enumerate()
                    .map(|(tid, t)| JsonStaff {
                        id: tid,
                        bbox: t.bbox,
                        lines: t.lines.iter().map(|&(y, _)| y).collect(),
                        confidence: t.confidence,
                        complete: t.complete,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let staves = systems.iter().map(|s| s.staves.len()).sum();
        JsonOutput {
            input,
            model: model.to_string(),
            width: self.width,
            height: self.height,
            threshold: params.threshold,
            min_fill: params.min_fill,
            neural_threshold: params.neural_threshold,
            staff_space_px: self.staff_space_px,
            counts: JsonCounts {
                lines: self.line_count,
                staves,
                systems: systems.len(),
            },
            systems,
        }
    }

    /// Grayscale map, same size as the input: each row is filled with its
    /// staff probability (normalized row score, floored by Gaussian bumps at
    /// detected line rows), scaled to 0..255. Bright bands mark staff lines.
    fn prob_image(&self) -> GrayImage {
        let (w, h) = (self.width, self.height);
        let mut out = GrayImage::new(w, h);
        let mut prob = vec![0.0f32; h as usize];
        // Gaussian bumps (sigma ~1.5 px) at every detected line row.
        for system in &self.systems {
            for staff in &system.staves {
                for &(y, _) in &staff.lines {
                    for d in -4i32..=4 {
                        let yy = y as i32 + d;
                        if yy >= 0 && (yy as u32) < h {
                            let bump = (-(d * d) as f32 / 4.5).exp();
                            let p = &mut prob[yy as usize];
                            *p = p.max(bump);
                        }
                    }
                }
            }
        }
        let raw: &mut [u8] = out.as_mut();
        for (y, (&bump, &score)) in prob.iter().zip(self.row_scores.iter()).enumerate() {
            let base = (score / 0.3).min(1.0) * 0.6;
            let v = (bump.max(base).clamp(0.0, 1.0) * 255.0).round() as u8;
            let off = y * w as usize;
            raw[off..off + w as usize].fill(v);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    fn test_params() -> DetectParams {
        DetectParams {
            threshold: 140,
            min_fill: 0.3,
            neural_threshold: 0.5,
            merge_px: 3,
            system_gap_spaces: 4.0,
        }
    }

    /// Fixed score profile: a 0.35 bump at row 10, a 0.8 peak at row 30.
    struct StubModel {
        scores: Vec<f32>,
        override_threshold: Option<f32>,
    }

    impl StaffLineModel for StubModel {
        fn name(&self) -> &'static str {
            "stub"
        }

        fn predict_rows(&self, _gray: &GrayImage, _params: &DetectParams) -> Vec<f32> {
            self.scores.clone()
        }

        fn line_threshold(&self, params: &DetectParams) -> f32 {
            self.override_threshold.unwrap_or(params.min_fill)
        }
    }

    fn stub_scores() -> Vec<f32> {
        let mut s = vec![0.05f32; 40];
        s[9] = 0.1;
        s[10] = 0.35;
        s[11] = 0.1;
        s[29] = 0.4;
        s[30] = 0.8;
        s[31] = 0.4;
        s
    }

    #[test]
    fn line_picker_uses_backend_threshold() {
        let gray = GrayImage::from_pixel(8, 40, image::Luma([255u8]));
        let params = test_params();
        // Default wiring follows min_fill: both the 0.35 bump and the
        // 0.8 peak clear 0.3.
        let plain = StubModel {
            scores: stub_scores(),
            override_threshold: None,
        };
        assert_eq!(detect_with_model(&gray, &plain, &params).line_count, 2);
        // A backend override (neural probabilities) filters the bump.
        let neural_like = StubModel {
            scores: stub_scores(),
            override_threshold: Some(params.neural_threshold),
        };
        let layout = detect_with_model(&gray, &neural_like, &params);
        assert_eq!(layout.line_count, 1);
    }

    /// White page with two full-width 5-line staves (space 8px, 24px apart).
    fn synthetic_page() -> GrayImage {
        let (w, h) = (200, 120);
        let mut img = GrayImage::from_pixel(w, h, image::Luma([255u8]));
        for top in [10u32, 58] {
            for k in 0..5 {
                for x in 0..w {
                    img.put_pixel(x, top + k * 8, image::Luma([0u8]));
                }
            }
        }
        img
    }

    #[test]
    fn finds_two_staves_in_one_system() {
        let gray = synthetic_page();
        let params = test_params();
        let model = ClassicalModel;
        let layout = detect_with_model(&gray, &model, &params);
        assert_eq!(layout.line_count, 10);
        assert_eq!(layout.systems.len(), 1, "24px gap < 4*8px space");
        assert_eq!(layout.systems[0].staves.len(), 2);
        assert!((layout.staff_space_px - 8.0).abs() < 0.01);
        for staff in &layout.systems[0].staves {
            assert!(staff.complete);
            assert_eq!(staff.lines.len(), 5);
        }
        let ys: Vec<u32> = layout.systems[0].staves[0]
            .lines
            .iter()
            .map(|&(y, _)| y)
            .collect();
        assert_eq!(ys, vec![10, 18, 26, 34, 42]);
    }

    #[test]
    fn trait_hook_matches_direct_call() {
        let gray = synthetic_page();
        let params = test_params();
        let model: &dyn StaffLineModel = &ClassicalModel;
        assert_eq!(model.name(), "classical");
        let via_trait = model.predict_rows(&gray, &params);
        let direct = row_fill_profile(&gray, params.threshold);
        assert_eq!(via_trait, direct);
    }

    #[test]
    fn barline_detected_but_stem_is_not() {
        // Staff 1 (rows 10..=42) gets a 2px full-band barline at x=100
        // and a 2px 24-of-33-row stem-like run at x=150 (0.73 fill):
        // only the barline may survive the fill and width gates.
        let (w, h) = (200, 120);
        let mut img = GrayImage::from_pixel(w, h, image::Luma([255u8]));
        for top in [10u32, 58] {
            for k in 0..5 {
                for x in 0..w {
                    img.put_pixel(x, top + k * 8, image::Luma([0u8]));
                }
            }
        }
        for y in 10..=42 {
            for x in 100..102 {
                img.put_pixel(x, y, image::Luma([0u8]));
            }
        }
        for y in 14..38 {
            for x in 150..152 {
                img.put_pixel(x, y, image::Luma([0u8]));
            }
        }
        let params = test_params();
        let layout = detect_with_model(&img, &ClassicalModel, &params);
        assert_eq!(layout.systems.len(), 1);
        let bars = &layout.systems[0].barlines;
        assert_eq!(bars.len(), 1, "barline only, got {bars:?}");
        assert!((bars[0] as i32 - 100).abs() <= 1);
    }

    #[test]
    fn blank_page_yields_no_systems() {
        let gray = GrayImage::from_pixel(100, 100, image::Luma([255u8]));
        let params = test_params();
        let layout = detect_with_model(&gray, &ClassicalModel, &params);
        assert_eq!(layout.line_count, 0);
        assert!(layout.systems.is_empty());
        let prob = layout.prob_image();
        assert_eq!(prob.dimensions(), (100, 100));
        assert!(prob.as_raw().iter().all(|&v| v == 0));
    }

    #[test]
    fn unknown_model_is_rejected() {
        assert!(select_model("classical", None).is_ok());
        assert!(select_model("nope", None).is_err());
        // Neural needs weights (and the onnx feature); without either it errs.
        assert!(select_model("neural", None).is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn neural_row_model_loads_fixture_and_scores_rows() {
        use std::path::Path;
        let model = crate::onnx_model::OnnxRowModel::load(Path::new("tests/rownet_pilot.onnx"))
            .expect("fixture loads");
        assert_eq!(model.name(), "neural");
        // Synthetic two-staff page (5 lines each); the pilot weights only
        // need to run the seam: shape, range, and peaks at the lines.
        let (w, h) = (200u32, 120u32);
        let mut img = GrayImage::from_pixel(w, h, image::Luma([255u8]));
        for top in [10u32, 58] {
            for k in 0..5 {
                for x in 0..w {
                    img.put_pixel(x, top + k * 8, image::Luma([0u8]));
                }
            }
        }
        let params = test_params();
        let probs = model.predict_rows(&img, &params);
        assert_eq!(probs.len(), h as usize);
        assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
        let median = {
            let mut s = probs.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[s.len() / 2]
        };
        for top in [10u32, 58] {
            for k in 0..5 {
                let y = (top + k * 8) as usize;
                let peak = probs[y.saturating_sub(2)..=(y + 2).min(119)]
                    .iter()
                    .cloned()
                    .fold(0.0f32, f32::max);
                assert!(peak > median, "line {y}: peak {peak} <= median {median}");
            }
        }
    }

    #[test]
    fn defaults_match_big500_training() {
        let args =
            Args::try_parse_from(["layout-staff", "--input", "clean.png"]).expect("defaults parse");
        assert_eq!(args.threshold, 140);
        assert_eq!(args.min_fill, 0.10);
        assert_eq!(args.neural_threshold, 0.30);
        assert_eq!(args.merge_px, 3);
    }
}
