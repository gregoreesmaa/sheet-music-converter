//! ONNX staff-line row model behind the [`StaffLineModel`] seam.
//!
//! Loads a row-window classifier (`patch` [N,1,33,248] f32 in, `logit`
//! [N] out, e.g. `tests/rownet_pilot.onnx`) and scores every image row.
//! Preprocessing mirrors training: width 248 (Triangle), binarize at the
//! detect threshold, edge rows replicate-padded. Inference runs in 512-row
//! batches; a runtime failure prints to stderr and yields zeros (model
//! problems surface at load time instead).

use std::sync::{Arc, Mutex};

use image::GrayImage;

use crate::{DetectParams, StaffLineModel};

const WIN_W: u32 = 248;
const HALF: i64 = 16;
const BATCH: usize = 512;

/// Row-window staff-line model (ONNX).
pub struct OnnxRowModel {
    _env: Arc<ort::environment::Environment>,
    session: Mutex<ort::session::Session>,
}

impl OnnxRowModel {
    /// Load weights. The file must be a row-window classifier ONNX.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let env = Arc::new(
            ort::init()
                .with_name("layout-staff")
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
}

impl StaffLineModel for OnnxRowModel {
    fn name(&self) -> &'static str {
        "neural"
    }

    fn line_threshold(&self, params: &DetectParams) -> f32 {
        params.neural_threshold
    }

    fn predict_rows(&self, gray: &GrayImage, params: &DetectParams) -> Vec<f32> {
        match self.infer(gray, params) {
            Ok(probs) => probs,
            Err(e) => {
                eprintln!("layout-staff: onnx inference failed ({e}); yielding zeros");
                vec![0.0; gray.height() as usize]
            }
        }
    }
}

impl OnnxRowModel {
    fn infer(&self, gray: &GrayImage, params: &DetectParams) -> Result<Vec<f32>, String> {
        let (w, h) = gray.dimensions();
        if w == 0 || h == 0 {
            return Ok(Vec::new());
        }
        let small = image::imageops::resize(gray, WIN_W, h, image::imageops::FilterType::Triangle);
        let bin: Vec<f32> = small
            .as_raw()
            .iter()
            .map(|&v| if v < params.threshold { 1.0 } else { 0.0 })
            .collect();
        let hh = h as usize;
        let mut probs = vec![0.0f32; hh];
        let mut session = self
            .session
            .lock()
            .map_err(|e| format!("session lock: {e}"))?;
        let mut start = 0;
        while start < hh {
            let end = (start + BATCH).min(hh);
            probs[start..end].copy_from_slice(&Self::infer_batch(
                &mut session,
                &bin,
                WIN_W as usize,
                hh,
                start..end,
            )?);
            start = end;
        }
        Ok(probs)
    }

    fn infer_batch(
        session: &mut ort::session::Session,
        bin: &[f32],
        sw: usize,
        hh: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Vec<f32>, String> {
        let n = rows.len();
        let mut input = Vec::with_capacity(n * 33 * sw);
        for r in rows {
            for dr in -HALF..=HALF {
                let rr = (r as i64 + dr).clamp(0, hh as i64 - 1) as usize;
                input.extend_from_slice(&bin[rr * sw..(rr + 1) * sw]);
            }
        }
        let array = ndarray::Array4::from_shape_vec((n, 1, 33, sw), input)
            .map_err(|e| format!("batch shape: {e}"))?;
        let tensor = ort::value::Tensor::from_array(array).map_err(|e| format!("tensor: {e}"))?;
        let outputs = session
            .run(ort::inputs!["patch" => tensor])
            .map_err(|e| format!("run: {e}"))?;
        let (_shape, data) = outputs["logit"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("output: {e}"))?;
        if data.len() != n {
            return Err(format!("expected {n} logits, got {}", data.len()));
        }
        Ok(data.iter().map(|&l| 1.0 / (1.0 + (-l).exp())).collect())
    }
}
