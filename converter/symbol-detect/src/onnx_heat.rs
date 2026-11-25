//! ONNX heatmap proposer behind the [`GlyphDetector`](crate::GlyphDetector) seam.
//!
//! Loads a CenterNet-style fully-convolutional proposer (`input`
//! [N,1,1240,1240] f32 in, `heat` [N,7,620,620] + `wh` [N,14,620,620] f32
//! out, e.g. `tmp-train/det-data/heatnet.onnx`) and turns center peaks +
//! regressed sizes into glyph boxes. The crop is covered with 1240 px tiles
//! on a 1100 px stride (edge tiles flush, mirroring training), peaks are
//! extracted with 3x3 max-pool NMS at `--heat-threshold`, and duplicate
//! boxes from tile overlaps are merged with per-class NMS (IoU 0.5).
//!
//! Deterministic: same pixels + same weights = same boxes. No blob
//! proposer, no randomness.

#[cfg(feature = "onnx")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "onnx")]
use image::GrayImage;

/// Heatmap tile edge, in pixels. Matches `mkdet_data.py --tile`.
pub const TILE: u32 = 1240;
/// Tile stride in pixels. 140 px overlap. Matches `mkdet_data.py --stride`.
pub const STRIDE_TILE: u32 = 1100;
/// Output stride of the heat/wh heads (1240 / 620).
pub const OUT_STRIDE: u32 = 2;
/// Heatmap classes, in head order. Matches `train_heat.py CLASSES`.
pub const HEAT_CLASSES: [&str; 7] = [
    "Note",
    "Rest",
    "Beam",
    "Hook",
    "Accidental",
    "NoteDot",
    "Clef",
];

/// Tile origin covering `len` pixels with [`TILE`] tiles on [`STRIDE_TILE`].
pub fn tile_origins(len: u32) -> Vec<u32> {
    if len <= TILE {
        return vec![0];
    }
    let mut v: Vec<u32> = (0..=(len - TILE)).step_by(STRIDE_TILE as usize).collect();
    let edge = len - TILE;
    if *v.last().unwrap_or(&0) != edge {
        v.push(edge);
    }
    v
}

/// One decoded detection in crop pixels.
#[derive(Debug, Clone)]
pub struct HeatBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub class: usize,
    pub score: f32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn logit(p: f32) -> f32 {
    if p <= 0.0 {
        f32::NEG_INFINITY
    } else if p >= 1.0 {
        f32::INFINITY
    } else {
        (p / (1.0 - p)).ln()
    }
}

/// Extract peak detections from one tile's heat/wh heads.
///
/// `heat` is [7,H,W] sigmoid logits in row-major order, `wh` is [14,H,W]
/// normalized (w/W, h/H) sizes. A cell is a peak when it is the maximum of
/// its 3x3 neighborhood (ties keep the first in scan order) and its
/// probability reaches `thr` (`beam_thr` for class 2, Beam).
pub fn decode_tile(
    heat: &[f32],
    wh: &[f32],
    out_w: usize,
    out_h: usize,
    tile_x: u32,
    tile_y: u32,
    thr: f32,
    beam_thr: f32,
) -> Vec<HeatBox> {
    let mut out = Vec::new();
    let plane = out_w * out_h;
    // Compare in raw logit space (sigmoid is strictly monotonic) and only
    // evaluate exp/sigmoid on surviving 3x3 peaks.
    let logit_thr = logit(thr);
    let beam_logit_thr = logit(beam_thr);
    for c in 0..HEAT_CLASSES.len() {
        // Beam peaks run an order of magnitude colder than noteheads
        // (tile gate: Beam R 0.51 at 0.5, 0.00 at 0.85); they keep their
        // own floor so duration cues survive the page operating point.
        let (t, t_logit) = if c == 2 {
            (beam_thr, beam_logit_thr)
        } else {
            (thr, logit_thr)
        };
        let base = c * plane;
        let ch = &heat[base..base + plane];
        let wh_w = &wh[(2 * c) * plane..(2 * c + 1) * plane];
        let wh_h = &wh[(2 * c + 1) * plane..(2 * c + 2) * plane];
        for yy in 0..out_h {
            let row = yy * out_w;
            let y0 = yy.saturating_sub(1);
            let y1 = (yy + 1).min(out_h - 1);
            for xx in 0..out_w {
                let idx = row + xx;
                let v = ch[idx];
                if v < t_logit {
                    continue;
                }
                // 3x3 max-pool peak check on raw logits; strictly greater neighbor kills.
                let x0 = xx.saturating_sub(1);
                let x1 = (xx + 1).min(out_w - 1);
                let mut is_peak = true;
                'nbr: for ny in y0..=y1 {
                    let nrow = ny * out_w;
                    for nx in x0..=x1 {
                        if nx == xx && ny == yy {
                            continue;
                        }
                        if ch[nrow + nx] > v {
                            is_peak = false;
                            break 'nbr;
                        }
                    }
                }
                if !is_peak {
                    continue;
                }
                let p = sigmoid(v);
                if p < t {
                    continue;
                }
                let wb = wh_w[idx] * TILE as f32;
                let hb = wh_h[idx] * TILE as f32;
                if wb <= 0.0 || hb <= 0.0 {
                    continue;
                }
                let cx = tile_x as f32 + (xx as f32 + 0.5) * OUT_STRIDE as f32;
                let cy = tile_y as f32 + (yy as f32 + 0.5) * OUT_STRIDE as f32;
                out.push(HeatBox {
                    x0: cx - wb / 2.0,
                    y0: cy - hb / 2.0,
                    x1: cx + wb / 2.0,
                    y1: cy + hb / 2.0,
                    class: c,
                    score: p,
                });
            }
        }
    }
    out
}

fn iou(a: &HeatBox, b: &HeatBox) -> f32 {
    let ix0 = a.x0.max(b.x0);
    let iy0 = a.y0.max(b.y0);
    let ix1 = a.x1.min(b.x1);
    let iy1 = a.y1.min(b.y1);
    let iw = (ix1 - ix0).max(0.0);
    let ih = (iy1 - iy0).max(0.0);
    let inter = iw * ih;
    let ua = (a.x1 - a.x0).max(0.0) * (a.y1 - a.y0).max(0.0)
        + (b.x1 - b.x0).max(0.0) * (b.y1 - b.y0).max(0.0)
        - inter;
    if ua <= 0.0 {
        0.0
    } else {
        inter / ua
    }
}

/// Per-class greedy NMS at `iou_thr`, highest score first. Deterministic:
///
/// ties break by (class, y0, x0).
pub fn nms(mut boxes: Vec<HeatBox>, iou_thr: f32) -> Vec<HeatBox> {
    boxes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.class.cmp(&b.class))
            .then_with(|| a.y0.partial_cmp(&b.y0).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.x0.partial_cmp(&b.x0).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut keep: Vec<HeatBox> = Vec::with_capacity(boxes.len());
    for b in boxes {
        let mut clash = false;
        for k in &keep {
            if b.class == k.class && iou(&b, k) > iou_thr {
                clash = true;
                break;
            }
        }
        if !clash {
            keep.push(b);
        }
    }
    keep
}

/// Heatmap proposer session (requires the `onnx` feature; the pure
/// tiling/decode/NMS helpers above it build and test without it).
#[cfg(feature = "onnx")]
pub struct OnnxHeatModel {
    _env: Arc<ort::environment::Environment>,
    session: Mutex<ort::session::Session>,
}

#[cfg(feature = "onnx")]
impl OnnxHeatModel {
    /// Load weights. The file must be a heatmap proposer ONNX with
    /// `input` / `heat` / `wh` tensors.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let env = Arc::new(
            ort::init()
                .with_name("symbol-detect-heat")
                .build()
                .map_err(|e| format!("onnx environment: {e}"))?,
        );
        let session = ort::session::Session::builder(&env)
            .map_err(|e| format!("onnx session builder: {e}"))?
            .commit_from_file(path)
            .map_err(|e| format!("load {}: {e}", path.display()))?;
        Ok(Self {
            _env: env,
            session: Mutex::new(session),
        })
    }

    /// Run the proposer over `gray` and return crop-pixel boxes.
    ///
    /// Tiles are batched in a single session run. A runtime failure is an
    /// `Err` (fatal: silent empty output would masquerade as a blank page).
    pub fn propose(
        &self,
        gray: &GrayImage,
        thr: f32,
        beam_thr: f32,
    ) -> Result<Vec<HeatBox>, String> {
        let (w, h) = (gray.width(), gray.height());
        if w == 0 || h == 0 {
            return Ok(Vec::new());
        }
        let xs = tile_origins(w);
        let ys = tile_origins(h);
        let n = xs.len() * ys.len();
        let raw = gray.as_raw();
        let w_us = w as usize;
        let h_us = h as usize;
        let tile_us = TILE as usize;
        let inv255 = 1.0f32 / 255.0;
        let mut flat = vec![0f32; n * tile_us * tile_us];
        for (ti, (tx, ty)) in xs
            .iter()
            .flat_map(|&tx| ys.iter().map(move |&ty| (tx, ty)))
            .enumerate()
        {
            let base = ti * tile_us * tile_us;
            let tx_us = tx as usize;
            let ty_us = ty as usize;
            for y in 0..tile_us {
                let sy = (ty_us + y).min(h_us - 1);
                let src_row = &raw[sy * w_us..(sy + 1) * w_us];
                let dst_row = &mut flat[base + y * tile_us..base + (y + 1) * tile_us];
                for x in 0..tile_us {
                    let sx = (tx_us + x).min(w_us - 1);
                    dst_row[x] = src_row[sx] as f32 * inv255;
                }
            }
        }
        let array = ndarray::Array4::from_shape_vec((n, 1, TILE as usize, TILE as usize), flat)
            .map_err(|e| format!("batch shape: {e}"))?;
        let tensor = ort::value::Tensor::from_array(array).map_err(|e| format!("tensor: {e}"))?;
        let mut session = self
            .session
            .lock()
            .map_err(|e| format!("session lock: {e}"))?;
        let outputs = session
            .run(ort::inputs!["input" => tensor])
            .map_err(|e| format!("run: {e}"))?;
        let (heat_shape, heat_data) = outputs["heat"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("output heat: {e}"))?;
        let (wh_shape, wh_data) = outputs["wh"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("output wh: {e}"))?;
        if heat_shape.len() != 4 || wh_shape.len() != 4 {
            return Err(format!(
                "expected 4-D heat/wh, got {heat_shape:?} / {wh_shape:?}"
            ));
        }
        let (bn, bc, bh, bw) = (
            heat_shape[0] as usize,
            heat_shape[1] as usize,
            heat_shape[2] as usize,
            heat_shape[3] as usize,
        );
        if bn != n || bc != HEAT_CLASSES.len() {
            return Err(format!(
                "heat shape {heat_shape:?} mismatches {n} tiles / {} classes",
                HEAT_CLASSES.len()
            ));
        }
        let wbn = wh_shape[0] as usize;
        let wbc = wh_shape[1] as usize;
        let wbh = wh_shape[2] as usize;
        let wbw = wh_shape[3] as usize;
        if wbn != n || wbc != 2 * HEAT_CLASSES.len() || wbh != bh || wbw != bw {
            return Err(format!("wh shape {wh_shape:?} mismatches heat"));
        }
        let heat_stride = bc * bh * bw;
        let wh_stride = wbc * wbh * wbw;
        let mut boxes = Vec::new();
        for (ti, (tx, ty)) in xs
            .iter()
            .flat_map(|&tx| ys.iter().map(move |&ty| (tx, ty)))
            .enumerate()
        {
            let h0 = ti * heat_stride;
            let w0 = ti * wh_stride;
            boxes.extend(decode_tile(
                &heat_data[h0..h0 + heat_stride],
                &wh_data[w0..w0 + wh_stride],
                bw,
                bh,
                tx,
                ty,
                thr,
                beam_thr,
            ));
        }
        // Clip to the crop; drop degenerates.
        let boxes: Vec<HeatBox> = boxes
            .into_iter()
            .map(|b| HeatBox {
                x0: b.x0.max(0.0).min(w as f32),
                y0: b.y0.max(0.0).min(h as f32),
                x1: b.x1.max(0.0).min(w as f32),
                y1: b.y1.max(0.0).min(h as f32),
                ..b
            })
            .filter(|b| b.x1 - b.x0 >= 4.0 && b.y1 - b.y0 >= 4.0)
            .collect();
        Ok(nms(boxes, 0.5))
    }
}
