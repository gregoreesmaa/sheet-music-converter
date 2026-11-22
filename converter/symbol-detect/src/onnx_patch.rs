//! ONNX glyph-patch classifier behind the [`GlyphDetector`](crate::GlyphDetector) seam.
//!
//! Loads a patch classifier (`patch` [N,1,48,48] f32 in, `logits` [N,C] f32
//! out, e.g. `tests/glyphnet.onnx`) and scores binarized glyph patches.
//! The class count C comes from the model file (3: notehead/rest/other,
//! 4: notehead/rest/beam/other, 5: notehead/rest/beam/flag/other).
//! Preprocessing mirrors training: 2 px
//! context margin, resize to 48x48, ink -> 1.0. Inference runs in
//! 256-patch batches.

use std::sync::{Arc, Mutex};

use image::GrayImage;

const PATCH: u32 = 48;
const MARGIN: u32 = 2;
const BATCH: usize = 256;

/// Patch classifier (notehead / rest / [beam /] other) as ONNX session.
pub struct OnnxPatchModel {
    _env: Arc<ort::environment::Environment>,
    session: Mutex<ort::session::Session>,
}

impl OnnxPatchModel {
    /// Load weights. The file must be a patch-classifier ONNX.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let env = Arc::new(
            ort::init()
                .with_name("symbol-detect")
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

    /// Score a batch of glyph patches. A runtime failure yields empty rows
    /// (the caller maps those to `unknown`); problems surface at load time
    /// instead.
    pub fn classify_batch(
        &self,
        gray: &GrayImage,
        bboxes: &[crate::BBox],
        threshold: u8,
    ) -> Vec<Vec<f32>> {
        match self.infer(gray, bboxes, threshold) {
            Ok(probs) => probs,
            Err(e) => {
                eprintln!("symbol-detect: onnx inference failed ({e}); yielding unknown");
                vec![Vec::new(); bboxes.len()]
            }
        }
    }
}

impl OnnxPatchModel {
    fn infer(
        &self,
        gray: &GrayImage,
        bboxes: &[crate::BBox],
        threshold: u8,
    ) -> Result<Vec<Vec<f32>>, String> {
        let (w, h) = (gray.width(), gray.height());
        if w == 0 || h == 0 {
            return Ok(Vec::new());
        }
        let mut patches = Vec::with_capacity(bboxes.len() * (PATCH * PATCH) as usize);
        for b in bboxes {
            let x0 = b.x.saturating_sub(MARGIN).min(w.saturating_sub(1));
            let y0 = b.y.saturating_sub(MARGIN).min(h.saturating_sub(1));
            let x1 = (b.x + b.w + MARGIN).min(w);
            let y1 = (b.y + b.h + MARGIN).min(h);
            let cw = x1.saturating_sub(x0).max(1);
            let ch = y1.saturating_sub(y0).max(1);
            let mut view = GrayImage::new(cw, ch);
            for (px, yy) in (y0..y0 + ch).enumerate() {
                for (pxx, xx) in (x0..x0 + cw).enumerate() {
                    view.put_pixel(
                        pxx as u32,
                        px as u32,
                        *gray.get_pixel(xx.min(w - 1), yy.min(h - 1)),
                    );
                }
            }
            let small =
                image::imageops::resize(&view, PATCH, PATCH, image::imageops::FilterType::Triangle);
            patches.extend(
                small
                    .as_raw()
                    .iter()
                    .map(|&v| if v < threshold { 1.0 } else { 0.0 }),
            );
        }
        let mut probs: Vec<Vec<f32>> = Vec::with_capacity(bboxes.len());
        let mut session = self
            .session
            .lock()
            .map_err(|e| format!("session lock: {e}"))?;
        let mut start = 0;
        while start < bboxes.len() {
            let end = (start + BATCH).min(bboxes.len());
            let logits = Self::infer_batch(
                &mut session,
                &patches[start * 48 * 48..end * 48 * 48],
                start..end,
            )?;
            for row in logits.iter() {
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
                let s: f32 = ex.iter().sum();
                probs.push(ex.iter().map(|&v| v / s).collect());
            }
            start = end;
        }
        Ok(probs)
    }

    fn infer_batch(
        session: &mut ort::session::Session,
        flat: &[f32],
        rows: std::ops::Range<usize>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let n = rows.len();
        let array =
            ndarray::Array4::from_shape_vec((n, 1, PATCH as usize, PATCH as usize), flat.to_vec())
                .map_err(|e| format!("batch shape: {e}"))?;
        let tensor = ort::value::Tensor::from_array(array).map_err(|e| format!("tensor: {e}"))?;
        let outputs = session
            .run(ort::inputs!["patch" => tensor])
            .map_err(|e| format!("run: {e}"))?;
        let (_shape, data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("output: {e}"))?;
        if n == 0 || data.len() % n != 0 {
            return Err(format!(
                "expected {n} logit rows, got {} values",
                data.len()
            ));
        }
        let dim = data.len() / n;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let o = i * dim;
            out.push(data[o..o + dim].to_vec());
        }
        Ok(out)
    }
}
