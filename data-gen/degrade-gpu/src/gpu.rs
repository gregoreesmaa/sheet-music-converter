//! GPU compute core: resample, paper, noise, lighting, motion, chromatic,`r`n//! chromatic aberration kernels plus the fused single-upload pipeline.`r`n//! Validated against the CPU implementations (see tests at the bottom).`r`n`r`nuse wgpu::util::DeviceExt;`r`n
// ---------------------------------------------------------------------------
// Milestone 2: generic inverse-map bilinear resample.
//
// One kernel serves perspective warp, rotation, radial distortion and
// chromatic aberration: every destination pixel inverse-maps through a
// 3x3 matrix (row-major) into source space and bilinear-samples with a
// caller-supplied fill color outside the source. Tap order mirrors the
// CPU `sample_bilinear` so outputs match within float-noise tolerance.
// ---------------------------------------------------------------------------

use super::chain::{rotation_dims, rotation_inv, ChainConfig, CHAIN_BLUR};

const RESAMPLE_SHADER: &str = r#"
struct Dims { src_w : u32, src_h : u32, dst_w : u32, dst_h : u32 }
struct Params { row0 : vec4<f32>, row1 : vec4<f32>, row2 : vec4<f32>, fill : vec4<f32> }

@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform> dims : Dims;
@group(0) @binding(3) var<uniform> prm : Params;

fn tap(ix : i32, iy : i32) -> vec3<f32> {
    if (ix < 0 || iy < 0 || ix >= i32(dims.src_w) || iy >= i32(dims.src_h)) {
        return vec3<f32>(prm.fill[0], prm.fill[1], prm.fill[2]);
    }
    let i = (u32(iy) * dims.src_w + u32(ix)) * 3u;
    return vec3<f32>(src[i], src[i + 1u], src[i + 2u]);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.dst_w || y >= dims.dst_h) { return; }
    let fx = f32(x);
    let fy = f32(y);
    let ww = prm.row2.x * fx + prm.row2.y * fy + prm.row2.z;
    let sx = (prm.row0.x * fx + prm.row0.y * fy + prm.row0.z) / ww;
    let sy = (prm.row1.x * fx + prm.row1.y * fy + prm.row1.z) / ww;
    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let tx = sx - f32(x0);
    let ty = sy - f32(y0);
    let top = tap(x0, y0) * (1.0 - tx) + tap(x0 + 1, y0) * tx;
    let bot = tap(x0, y0 + 1) * (1.0 - tx) + tap(x0 + 1, y0 + 1) * tx;
    let out = top * (1.0 - ty) + bot * ty;
    let i = (y * dims.dst_w + x) * 3u;
    dst[i] = out.x;
    dst[i + 1u] = out.y;
    dst[i + 2u] = out.z;
}
"#;

// Fused rotate+downscale affine pass (terminal `--gpu` stage design): ONE
// single-sample bilinear dispatch reads the pre-rotation buffer and writes
// the final (nw,nh) image directly, replacing the eager rotation resample
// plus the CPU Triangle downscale. The inverse map evaluated per pixel is
// M^-1 = R^-1 . S^-1 (pipeline order is rotate-then-downscale, so S
// inverts first): xr = x/sx, yr = y/sy onto the virtual rotated canvas,
// then the R^-1 rows (same convention as `rotation_frame`). The virtual
// canvas (w2,h2) is never materialized — that is the point — so it lives
// in `meta` while `dims` carries the pre-rotation src + (nw,nh) dst via
// the existing Dims struct (unchanged).
//
// r0/r1 = R^-1 rows (cst, snt, tx / -snt, cst, ty); meta = (w2, h2,
// sx_inv = 1/sx, sy_inv = 1/sy); fill = desk tone for both OOB levels
// (virtual canvas + source taps), matching the GPU rotation path.
// Host packing ([f32; 16], see `rotscale_prm`):
// [cst, snt, tx, 0, -snt, cst, ty, 0, w2, h2, sx_inv, sy_inv, f0, f1, f2, 0].
//
// Manual bilinear taps, NOT a texture sampler (same as every existing
// kernel: filtering stays exactly specified and deterministic). No `ww`
// divide: the map is affine (row2 = [0,0,1]), so the perspective divide is
// skipped by construction — do not "fix" it back in.
//
// Fidelity note: fused-vs-sequential max-abs == 0 is NOT claimed and NOT
// tested — sequential bilinear-then-bilinear has up-to-16-px support while
// single-sample affine has 4-px support, so values differ by design
// (LSB-level on gradients, larger on edges). Validation is tolerance +
// Otsu gates (see tests below), never bit-exactness.
const AFFINE_SHADER: &str = r#"
struct Dims { src_w : u32, src_h : u32, dst_w : u32, dst_h : u32 }
struct Aff { r0 : vec4<f32>, r1 : vec4<f32>, aux : vec4<f32>, fill : vec4<f32> }

@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform> dims : Dims;
@group(0) @binding(3) var<uniform> prm : Aff;

fn tap(ix : i32, iy : i32) -> vec3<f32> {
    if (ix < 0 || iy < 0 || ix >= i32(dims.src_w) || iy >= i32(dims.src_h)) {
        return vec3<f32>(prm.fill[0], prm.fill[1], prm.fill[2]);
    }
    let i = (u32(iy) * dims.src_w + u32(ix)) * 3u;
    return vec3<f32>(src[i], src[i + 1u], src[i + 2u]);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.dst_w || y >= dims.dst_h) { return; }
    let xr = f32(x) * prm.aux.z;
    let yr = f32(y) * prm.aux.w;
    var out : vec3<f32>;
    if (xr < 0.0 || yr < 0.0 || xr >= prm.aux.x || yr >= prm.aux.y) {
        out = vec3<f32>(prm.fill[0], prm.fill[1], prm.fill[2]);
    } else {
        let sx_ = prm.r0.x * xr + prm.r0.y * yr + prm.r0.z;
        let sy_ = prm.r1.x * xr + prm.r1.y * yr + prm.r1.z;
        let x0 = i32(floor(sx_));
        let y0 = i32(floor(sy_));
        let tx = sx_ - f32(x0);
        let ty = sy_ - f32(y0);
        let top = tap(x0, y0) * (1.0 - tx) + tap(x0 + 1, y0) * tx;
        let bot = tap(x0, y0 + 1) * (1.0 - tx) + tap(x0 + 1, y0 + 1) * tx;
        out = top * (1.0 - ty) + bot * ty;
    }
    let i = (y * dims.dst_w + x) * 3u;
    dst[i] = out.x;
    dst[i + 1u] = out.y;
    dst[i + 2u] = out.z;
}
"#;

#[allow(dead_code)]
pub fn make_pipeline(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Run a same-size pointwise kernel: upload src, dispatch, download.
/// Shared by paper/noise/lighting/motion (resample keeps its own path
/// because its dims can change).
#[allow(dead_code)]
pub fn run_pointwise(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    tag: &str,
    src_host: &[f32],
    w: u32,
    h: u32,
    prm: [f32; 16],
) -> Vec<f32> {
    use wgpu::util::DeviceExt;
    let n = (w * h * 3) as usize;
    let src = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(src_host),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let dst = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dims: [u32; 4] = [w, h, w, h];
    let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let prm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&prm),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(tag),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dims_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: prm_buf.as_entire_binding() },
        ],
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
    }
    enc.copy_buffer_to_buffer(&dst, 0, &staging, 0, (n * std::mem::size_of::<f32>()) as u64);
    queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
    let view = slice.get_mapped_range().expect("map failed");
    let out: &[f32] = bytemuck::cast_slice(&view);
    let result = out.to_vec();
    drop(view);
    staging.unmap();
    result
}

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    resample_pipeline: wgpu::ComputePipeline,
    noise_pipeline: wgpu::ComputePipeline,
    lighting_pipeline: wgpu::ComputePipeline,
    ca_pipeline: wgpu::ComputePipeline,
    grain_pipeline: wgpu::ComputePipeline,
    bend_pipeline: wgpu::ComputePipeline,
    /// Tilt-capable blur used by both `blur()` (uniform) and `chain_full()`
    /// (uniform or tilt-shift smoothstep focus band).
    chain_blur_pipeline: wgpu::ComputePipeline,
    /// Fused rotate+downscale affine pass (`AFFINE_SHADER`).
    affine_pipeline: wgpu::ComputePipeline,
}

/// Knobs for lighting stage 5: glare disc, shadow direction, vignette.
#[derive(Clone, Copy)]
pub struct LightingKnobs {
    pub gx: f32,
    pub gy: f32,
    pub glare_radius: f32,
    pub glare_alpha: f32,
    /// Shadow direction unit-ish vector + strength.
    pub sh_dx: f32,
    pub sh_dy: f32,
    pub shadow_strength: f32,
    pub vignette_strength: f32,
}

/// Knobs for the stage-1 paper grain field.
#[derive(Clone, Copy)]
pub struct GrainKnobs {
    pub sigma: f32,
    pub seed: u32,
    pub speckle_prob: f32,
    pub strength: f32,
    pub warm: [f32; 3],
}

/// Knobs for the stage-2 page bend warp. `mode`: 0 = spine, 1 = ripple,
/// 2 = edge curl; `edge_id`: 0 = left, 1 = right, 2 = top, 3 = bottom.
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct BendKnobs {
    pub mode: u32,
    pub edge_id: u32,
    pub curl_px: f32,
    pub curl_shade: f32,
    pub spine_cx: f32,
    pub spine_w: f32,
    pub depth: f32,
    pub rip_fx: f32,
    pub rip_fy: f32,
    pub rip_ph: f32,
}

/// Output dims of the fused rotate+downscale pass: the legacy ceil-expanded
/// rotation canvas `(w2, h2)` (same formula as `rotation_frame` /
/// `chain::rotation_dims`), then `as u32` truncation of `(w2*sx, h2*sy)`,
/// matching the CPU downscale dims formula exactly.
#[allow(dead_code)]
pub fn rot_scale_out_dims(
    rot_deg: f32,
    w: u32,
    h: u32,
    scale_x: f32,
    scale_y: f32,
) -> (u32, u32, u32, u32) {
    let (w2, h2) = crate::chain::rotation_dims(rot_deg, w, h);
    let nw = (w2 as f32 * scale_x) as u32;
    let nh = (h2 as f32 * scale_y) as u32;
    (w2, h2, nw, nh)
}

/// R^-1 rows + meta uniform inputs for the fused affine pass. `rows` holds
/// the rotation inverse in `rotation_frame` convention
/// (`[cst, snt, tx, -snt, cst, ty]`, reused verbatim from
/// `chain::rotation_inv` so the degenerate `sx = sy = 1` case agrees with
/// `resample` bit-for-bit); `meta` holds `(w2, h2, 1/sx, 1/sy)`.
/// Also returns `(w2, h2, nw, nh)` for dims bookkeeping.
#[allow(dead_code)]
pub fn rotscale_frame(
    rot_deg: f32,
    w: u32,
    h: u32,
    scale_x: f32,
    scale_y: f32,
) -> ([f32; 6], [f32; 4], u32, u32, u32, u32) {
    let (w2, h2, nw, nh) = rot_scale_out_dims(rot_deg, w, h, scale_x, scale_y);
    let rinv = crate::chain::rotation_inv(rot_deg, w, h, w2, h2);
    let rows = [rinv[0], rinv[1], rinv[2], rinv[3], rinv[4], rinv[5]];
    let meta = [w2 as f32, h2 as f32, 1.0 / scale_x, 1.0 / scale_y];
    (rows, meta, w2, h2, nw, nh)
}

/// Pack the affine uniforms: R^-1 `rows`, `meta`, `fill` into the 16-float
/// `Aff` uniform (64 B; the 16-float shape is load-bearing for bind-group
/// validation, same as every other kernel).
#[allow(dead_code)]
pub fn rotscale_prm(rows: [f32; 6], meta: [f32; 4], fill: [f32; 3]) -> [f32; 16] {
    [
        rows[0], rows[1], rows[2], 0.0,
        rows[3], rows[4], rows[5], 0.0,
        meta[0], meta[1], meta[2], meta[3],
        fill[0], fill[1], fill[2], 0.0,
    ]
}

impl GpuContext {
    pub async fn new() -> Self {
        Self::try_new().await.expect("no GPU adapter")
    }

    /// Fallible constructor for hosts (e.g. photo-degrade --gpu) that must
    /// skip GPU work cleanly on machines without a usable adapter.
    pub async fn try_new() -> Result<Self, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no GPU adapter: {e:?}"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| format!("request_device failed: {e:?}"))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resample"),
            source: wgpu::ShaderSource::Wgsl(RESAMPLE_SHADER.into()),
        });
        let resample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("resample-pipe"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let noise_pipeline = make_pipeline(&device, "noise", NOISE_SHADER);
        let lighting_pipeline = make_pipeline(&device, "lighting", LIGHTING_SHADER);
        let ca_pipeline = make_pipeline(&device, "ca", CA_SHADER);
        let grain_pipeline = make_pipeline(&device, "grain", GRAIN_SHADER);
        let bend_pipeline = make_pipeline(&device, "bend", BEND_SHADER);
        let chain_blur_pipeline = make_pipeline(&device, "chain-blur", CHAIN_BLUR);
        let affine_pipeline = make_pipeline(&device, "affine", AFFINE_SHADER);
        Ok(Self {
            device,
            queue,
            resample_pipeline,
            noise_pipeline,
            lighting_pipeline,
            ca_pipeline,
            grain_pipeline,
            bend_pipeline,
            chain_blur_pipeline,
            affine_pipeline,
        })
    }

    /// Inverse-map `src` (w x h interleaved RGB f32) through row-major 3x3
    /// `inv` into a w2 x h2 image, filling out-of-source taps with `fill`.
    #[allow(dead_code)]
    pub fn resample(
        &self,
        src_host: &[f32],
        w: u32,
        h: u32,
        w2: u32,
        h2: u32,
        inv: [f32; 9],
        fill: [f32; 3],
    ) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let n2 = (w2 * h2 * 3) as usize;
        let src = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rs-src"),
            contents: bytemuck::cast_slice(src_host),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let dst = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rs-dst"),
            size: (n2 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dims: [u32; 4] = [w, h, w2, h2];
        let dims_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rs-dims"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Padded to 4x vec4<f32> so the uniform struct is 16-byte aligned.
        let prm = [
            inv[0], inv[1], inv[2], 0.0,
            inv[3], inv[4], inv[5], 0.0,
            inv[6], inv[7], inv[8], 0.0,
            fill[0], fill[1], fill[2], 0.0,
        ];
        let prm_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rs-prm"),
            contents: bytemuck::cast_slice(&prm),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rs-bind"),
            layout: &self.resample_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: prm_buf.as_entire_binding() },
            ],
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rs-staging"),
            size: (n2 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.resample_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(w2.div_ceil(16), h2.div_ceil(16), 1);
        }
        enc.copy_buffer_to_buffer(&dst, 0, &staging, 0, (n2 * std::mem::size_of::<f32>()) as u64);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
        let view = slice.get_mapped_range().expect("map failed");
        let out: &[f32] = bytemuck::cast_slice(&view);
        let result = out.to_vec();
        drop(view);
        staging.unmap();
        result
    }

    /// Fused rotate+downscale affine pass (chain-callable entry): reads the
    /// pre-rotation `src` (`w` x `h`) and writes the final `(nw, nh)` image
    /// in ONE dispatch. `rows`/`meta` come from `rotscale_frame`
    /// (R^-1 rows + `(w2, h2, 1/sx, 1/sy)`); `fill` is the desk tone for
    /// both OOB levels. Dims travel via the existing `Dims` struct
    /// (`src = (w, h)`, `dst = (nw, nh)`); the virtual canvas `(w2, h2)`
    /// lives in `meta` because no buffer ever has that shape.
    /// With `sx = sy = 1` this is a rotation-only dispatch through the same
    /// kernel (`nw, nh == w2, h2`); callers keep one code path.
    #[allow(dead_code)]
    pub fn affine(
        &self,
        src_host: &[f32],
        w: u32,
        h: u32,
        nw: u32,
        nh: u32,
        rows: [f32; 6],
        meta: [f32; 4],
        fill: [f32; 3],
    ) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let n2 = (nw * nh * 3) as usize;
        let src = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aff-src"),
            contents: bytemuck::cast_slice(src_host),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let dst = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aff-dst"),
            size: (n2 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dims: [u32; 4] = [w, h, nw, nh];
        let dims_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aff-dims"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let prm = rotscale_prm(rows, meta, fill);
        let prm_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aff-prm"),
            contents: bytemuck::cast_slice(&prm),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aff-bind"),
            layout: &self.affine_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: prm_buf.as_entire_binding() },
            ],
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aff-staging"),
            size: (n2 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.affine_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(nw.div_ceil(16), nh.div_ceil(16), 1);
        }
        enc.copy_buffer_to_buffer(&dst, 0, &staging, 0, (n2 * std::mem::size_of::<f32>()) as u64);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
        let view = slice.get_mapped_range().expect("map failed");
        let out: &[f32] = bytemuck::cast_slice(&view);
        let result = out.to_vec();
        drop(view);
        staging.unmap();
        result
    }

    /// Apply lighting: glare + directional shadow + vignette.
    #[allow(dead_code)]
    pub fn lighting(&self, src_host: &[f32], w: u32, h: u32, k: LightingKnobs) -> Vec<f32> {
        let wf = w as f32;
        let hf = h as f32;
        let cx = wf / 2.0;
        let cy = hf / 2.0;
        let diag = (wf * wf + hf * hf).sqrt().max(1.0);
        let max_d = (cx * cx + cy * cy).sqrt().max(1.0);
        let prm = [
            k.gx, k.gy, k.glare_radius, k.glare_alpha,
            k.sh_dx, k.sh_dy, k.shadow_strength, k.vignette_strength,
            cx, cy, diag, max_d,
            0.0, 0.0, 0.0, 0.0,
        ];
        run_pointwise(&self.device, &self.queue, &self.lighting_pipeline, "lighting", src_host, w, h, prm)
    }

    /// Chromatic aberration: R/B sampled along the center radius.
    #[allow(dead_code)]
    pub fn chromatic(&self, src_host: &[f32], w: u32, h: u32, ca_px: f32) -> Vec<f32> {
        let prm = [
            w as f32 / 2.0, h as f32 / 2.0, w as f32, h as f32,
            246.0, 243.0, 236.0, ca_px,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        run_pointwise(&self.device, &self.queue, &self.ca_pipeline, "ca", src_host, w, h, prm)
    }

    /// Stage-1 grain: hash-stream variant of the CPU ChaCha loop.
    #[allow(dead_code)]
    pub fn grain(&self, src_host: &[f32], w: u32, h: u32, k: GrainKnobs) -> Vec<f32> {
        let prm = [
            k.sigma, f32::from_bits(k.seed), k.speckle_prob, k.strength,
            k.warm[0], k.warm[1], k.warm[2], 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        run_pointwise(&self.device, &self.queue, &self.grain_pipeline, "grain", src_host, w, h, prm)
    }

    /// Stage-2 page bend warp (same dims, paper-tone BG fill).
    #[allow(dead_code)]
    pub fn bend(&self, src_host: &[f32], w: u32, h: u32, k: BendKnobs) -> Vec<f32> {
        let prm = [
            k.mode as f32, k.curl_px, k.curl_shade, k.edge_id as f32,
            k.spine_cx, k.spine_w, k.depth, 0.0,
            k.rip_fx, k.rip_fy, k.rip_ph, 0.0,
            w as f32, h as f32, 0.0, 0.0,
        ];
        run_pointwise(&self.device, &self.queue, &self.bend_pipeline, "bend", src_host, w, h, prm)
    }

    /// Stage-4 gaussian defocus blur (clamped edges). Radius 0 = identity.
    #[allow(dead_code)]
    pub fn blur(&self, src_host: &[f32], w: u32, h: u32, sigma: f32) -> Vec<f32> {
        let radius = (3.0 * sigma).ceil().max(0.0);
        let prm = [
            sigma, radius, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        run_pointwise(&self.device, &self.queue, &self.chain_blur_pipeline, "blur", src_host, w, h, prm)
    }

    /// Resident single-submit run of the dense photo-degrade stages on the
    /// SHARED device: grain -> bend -> perspective -> rotation -> blur ->
    /// lighting -> chromatic -> noise, with ONE upload, ONE encoder/submit
    /// (one compute pass per RUNNING stage), and ONE download. Returns
    /// `(pixels, out_w, out_h)`; rotation expands the canvas when enabled.
    ///
    /// Skipped stages (`run_*` false) leave the resident buffers untouched,
    /// exactly as if the stage gate failed in `photo-degrade`. The
    /// `photo-degrade --gpu` path calls this twice per file — pre-framing
    /// (`run_grain/bend/persp`) and post-framing
    /// (`run_rot/blur/light/ca/noise`) — because the CPU framing stage must
    /// stay between the perspective warp and the rotation (the notation
    /// bbox must see paper fill, not the rotation's dark-desk triangles).
    /// Each call is still a single submit on this shared context: no second
    /// device (the `sepblur` lesson: its own device plus two round-trips
    /// measured slower end-to-end), no per-stage round-trips.
    ///
    /// Kernels and param packing mirror the single-call methods above
    /// (`grain`, `bend`, `resample`, `blur`, `lighting`, `chromatic`,
    /// `noise`); the triple-proof test asserts bit-exactness vs the
    /// sequential calls so the copies cannot drift silently. The blur pass
    /// uses the tilt-capable `chain::CHAIN_BLUR` kernel (uniform mode is
    /// bit-exact vs `blur()`; tilt mode blends sharp/blurred in-kernel with
    /// the same smoothstep band the CPU focus stage uses).
    #[allow(dead_code)]
    pub fn chain_full(
        &self,
        src_host: &[f32],
        w: u32,
        h: u32,
        cfg: &ChainConfig,
    ) -> (Vec<f32>, u32, u32) {
        use wgpu::util::DeviceExt;
        assert_eq!(src_host.len(), (w * h * 3) as usize, "chain input size");
        let (w2, h2) = if cfg.run_rot {
            rotation_dims(cfg.rot_deg, w, h)
        } else {
            (w, h)
        };
        let (out_w, out_h) = if cfg.run_affine {
            (cfg.affine_nw.max(1), cfg.affine_nh.max(1))
        } else {
            (w2, h2)
        };
        let nmax = (w * h).max(w2 * h2).max(out_w * out_h) * 3;
        let bytes = (nmax as usize * std::mem::size_of::<f32>()) as u64;
        let up = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chain-up"),
            contents: bytemuck::cast_slice(src_host),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let mk_resident = |label: &str| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mut ping = mk_resident("chain-ping");
        let mut pong = mk_resident("chain-pong");
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut keep: Vec<(wgpu::BindGroup, wgpu::Buffer, wgpu::Buffer)> = Vec::new();
        let mut head = true;
        {
            let mut dispatch = |pipe: &wgpu::ComputePipeline,
                                tag: &str,
                                src: &wgpu::Buffer,
                                dst: &wgpu::Buffer,
                                dims: [u32; 4],
                                prm: [f32; 16]| {
                let dims_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&dims),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let prm_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&prm),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(tag),
                    layout: &pipe.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: dims_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: prm_buf.as_entire_binding() },
                    ],
                });
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(dims[2].div_ceil(16), dims[3].div_ceil(16), 1);
                keep.push((bind, dims_buf, prm_buf));
            };
            // 1. grain (same dims). Packing mirrors `grain()`.
            if cfg.run_grain {
                let prm = [
                    cfg.grain_sigma, f32::from_bits(cfg.grain_seed), cfg.grain_speckle_prob, cfg.grain_strength,
                    cfg.grain_warm[0], cfg.grain_warm[1], cfg.grain_warm[2], 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                ];
                let src = if head { &up } else { &pong };
                dispatch(&self.grain_pipeline, "c-grain", src, &ping, [w, h, w, h], prm);
                std::mem::swap(&mut ping, &mut pong);
                head = false;
            }
            // 2. bend (same dims). Packing mirrors `bend()`.
            if cfg.run_bend {
                let b = &cfg.bend;
                let prm = [
                    b.mode as f32, b.curl_px, b.curl_shade, b.edge_id as f32,
                    b.spine_cx, b.spine_w, b.depth, 0.0,
                    b.rip_fx, b.rip_fy, b.rip_ph, 0.0,
                    w as f32, h as f32, 0.0, 0.0,
                ];
                let src = if head { &up } else { &pong };
                dispatch(&self.bend_pipeline, "c-bend", src, &ping, [w, h, w, h], prm);
                std::mem::swap(&mut ping, &mut pong);
                head = false;
            }
            // 3. perspective warp (same dims, paper-tone fill).
            if cfg.run_persp {
                let inv = cfg.persp_inv;
                let prm = [
                    inv[0], inv[1], inv[2], 0.0,
                    inv[3], inv[4], inv[5], 0.0,
                    inv[6], inv[7], inv[8], 0.0,
                    246.0, 243.0, 236.0, 0.0,
                ];
                let src = if head { &up } else { &pong };
                dispatch(&self.resample_pipeline, "c-persp", src, &ping, [w, h, w, h], prm);
                std::mem::swap(&mut ping, &mut pong);
                head = false;
            }
            // 4. rotation (canvas expands, desk-tone fill).
            if cfg.run_rot {
                let rinv = rotation_inv(cfg.rot_deg, w, h, w2, h2);
                let prm = [
                    rinv[0], rinv[1], rinv[2], 0.0,
                    rinv[3], rinv[4], rinv[5], 0.0,
                    rinv[6], rinv[7], rinv[8], 0.0,
                    25.0, 25.0, 30.0, 0.0,
                ];
                let src = if head { &up } else { &pong };
                dispatch(&self.resample_pipeline, "c-rot", src, &ping, [w, h, w2, h2], prm);
                std::mem::swap(&mut ping, &mut pong);
                head = false;
            }
            // 5. blur (+ optional in-kernel tilt blend). Uniform packing
            // mirrors `blur()`; tilt slots carry the focus band.
            if cfg.run_blur {
                let radius = (3.0 * cfg.blur_sigma).ceil().max(0.0);
                let prm = [
                    cfg.blur_sigma, radius, cfg.focus_y, cfg.focus_h,
                    if cfg.tilt_active { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                ];
                let src = if head { &up } else { &pong };
                dispatch(&self.chain_blur_pipeline, "c-blur", src, &ping, [w2, h2, w2, h2], prm);
                std::mem::swap(&mut ping, &mut pong);
                head = false;
            }
            // 6. lighting. Packing mirrors `lighting()`.
            if cfg.run_light {
                let k = &cfg.light;
                let (wf, hf) = (w2 as f32, h2 as f32);
                let prm = [
                    k.gx, k.gy, k.glare_radius, k.glare_alpha,
                    k.sh_dx, k.sh_dy, k.shadow_strength, k.vignette_strength,
                    wf / 2.0, hf / 2.0,
                    (wf * wf + hf * hf).sqrt().max(1.0),
                    ((wf / 2.0).powi(2) + (hf / 2.0).powi(2)).sqrt().max(1.0),
                    0.0, 0.0, 0.0, 0.0,
                ];
                let src = if head { &up } else { &pong };
                dispatch(&self.lighting_pipeline, "c-light", src, &ping, [w2, h2, w2, h2], prm);
                std::mem::swap(&mut ping, &mut pong);
                head = false;
            }
            // 7. chromatic aberration. Packing mirrors `chromatic()`.
            if cfg.run_ca {
                let prm = [
                    w2 as f32 / 2.0, h2 as f32 / 2.0, w2 as f32, h2 as f32,
                    246.0, 243.0, 236.0, cfg.ca_px,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                ];
                let src = if head { &up } else { &pong };
                dispatch(&self.ca_pipeline, "c-ca", src, &ping, [w2, h2, w2, h2], prm);
                std::mem::swap(&mut ping, &mut pong);
                head = false;
            }
            // 8. sensor noise. Packing mirrors `noise()`.
            if cfg.run_noise {
                let prm = [
                    cfg.noise_sigma, f32::from_bits(cfg.noise_seed), 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                ];
                let src = if head { &up } else { &pong };
                dispatch(&self.noise_pipeline, "c-noise", src, &ping, [w2, h2, w2, h2], prm);
                std::mem::swap(&mut ping, &mut pong);
                head = false;
            }
            // 9. terminal fused rotate+downscale affine pass (desk-tone fill).
            if cfg.run_affine {
                let prm = rotscale_prm(cfg.affine_rows, cfg.affine_meta, [25.0, 25.0, 30.0]);
                let src = if head { &up } else { &pong };
                dispatch(&self.affine_pipeline, "c-affine", src, &ping, [w2, h2, out_w, out_h], prm);
                std::mem::swap(&mut ping, &mut pong);
                head = false;
            }
        }
        // Single download of the final resident buffer (`up` is still
        // current when every stage was skipped).
        let final_n = (out_w * out_h * 3) as usize;
        let final_bytes = (final_n * std::mem::size_of::<f32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chain-staging"),
            size: final_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let done = if head { &up } else { &pong };
        enc.copy_buffer_to_buffer(done, 0, &staging, 0, final_bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
        let view = slice.get_mapped_range().expect("map failed");
        let out: &[f32] = bytemuck::cast_slice(&view);
        let result = out.to_vec();
        drop(view);
        staging.unmap();
        (result, out_w, out_h)
    }

    pub fn noise(&self, src_host: &[f32], w: u32, h: u32, sigma: f32, seed: u32) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let n = (w * h * 3) as usize;
        let src = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("noise-src"),
            contents: bytemuck::cast_slice(src_host),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let dst = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("noise-dst"),
            size: (n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dims: [u32; 4] = [w, h, w, h];
        let dims_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("noise-dims"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Nse struct is 4x vec4 (64 B); shader reads only a=(sigma, seed_bits).
        let prm = [
            sigma, f32::from_bits(seed), 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let prm_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("noise-prm"),
            contents: bytemuck::cast_slice(&prm),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("noise-bind"),
            layout: &self.noise_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: prm_buf.as_entire_binding() },
            ],
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("noise-staging"),
            size: (n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.noise_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }
        enc.copy_buffer_to_buffer(&dst, 0, &staging, 0, (n * std::mem::size_of::<f32>()) as u64);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
        let view = slice.get_mapped_range().expect("map failed");
        let out: &[f32] = bytemuck::cast_slice(&view);
        let result = out.to_vec();
        drop(view);
        staging.unmap();
        result
    }
}

/// Generator tag for the GPU grain/noise shaders below. The shaders use
/// Irwin-Hall-4 Gaussians over a PCG-32 hash stream (`"ih4"`); the CPU
/// bank path still builds Box-Muller banks (`"bm"`). The photo-degrade
/// `Params.noise_gen` field should record this value on every `--gpu` run
/// so benches and dataset manifests can separate IH4 vs BM runs. Seed
/// packing is unchanged (same uniforms, same `(seed, stage, key)`-derived
/// seeds flow in verbatim); pixel values still change at fixed seeds
/// because the tap range extends 6/8 -> 14 taps by design.
/// Sensor noise: per-pixel Irwin-Hall-4 Gaussian over a PCG-32 hash PRNG.
/// Keyed by (seed, x, y, tap) so output is deterministic per seed without
/// any RNG state; the stream differs from the CPU ChaCha stream by design,
/// so validation is statistical (mean/std), not bit-exact.
/// Uniforms: A=(sigma, seed_bits, 0, 0).
/// Paper grain + warm tint + sparse dark speckles (stage 1).
/// Hash-stream variant of the CPU ChaCha loop: same first two moments, not
/// bit-identical. Uniforms: A=(sigma, seed_bits, speckle_prob, strength),
/// B=(warm_r, warm_g, warm_b, 0).
#[allow(dead_code)]
pub const NOISE_GEN: &str = "ih4";
const GRAIN_SHADER: &str = r#"
struct Dims { src_w : u32, src_h : u32, dst_w : u32, dst_h : u32 }
struct Grn { a : vec4<f32>, b : vec4<f32>, c : vec4<f32>, d : vec4<f32> }

@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform> dims : Dims;
@group(0) @binding(3) var<uniform> prm : Grn;

// PCG-32 hash (Jarzynski & Olano JCGT): keyed per (seed, x, y, tap), a pure
// function of its inputs, so output is deterministic per seed with no RNG
// state. The `seed ^ x*k1 ^ y*k2 ^ n*k3` pre-mix keeps the grain/noise
// stream-id separation working unchanged. Unsigned arithmetic wraps mod
// 2^32 in WGSL, which is what this mixer relies on.
fn pcg01(seed : u32, x : u32, y : u32, n : u32) -> f32 {
    var state = (seed ^ (x * 374761393u) ^ (y * 668265263u) ^ (n * 2246822519u))
        * 747796405u + 2891336453u;
    let word = ((state >> (u32(state >> 28u) + 4u)) ^ state) * 277803737u;
    let out = (word >> 22u) ^ word;
    return f32(out) / 4294967295.0;
}

// Irwin-Hall-4 Gaussian: the sum of 4 uniforms has mean 2 and variance
// 4/12 = 1/3, so `(s - 2) * sqrt(3)` is unit-variance. Kurtosis 2.4 vs 3.0
// for Box-Muller, bounded support +/-2*sqrt(3) ~= +/-3.464 sigma vs
// unbounded tails; at production sigmas (grain 2-5 LSB, noise 3-14 LSB on
// 0-255 + clamp) the bound binds only far out, and downstream is
// downscale -> luma -> Otsu, blind to tail shape at these amplitudes.
// Channel c uses taps b+4*c .. b+4*c+3 (12 taps/pixel total).
fn ih4(seed : u32, x : u32, y : u32, b : u32) -> f32 {
    let s = pcg01(seed, x, y, b) + pcg01(seed, x, y, b + 1u)
          + pcg01(seed, x, y, b + 2u) + pcg01(seed, x, y, b + 3u);
    return (s - 2.0) * 1.7320508075688772;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.dst_w || y >= dims.dst_h) { return; }
    let sigma = prm.a.x;
    let seed = bitcast<u32>(prm.a.y);
    let prob = prm.a.z;
    let s = prm.a.w;
    let g0 = ih4(seed, x, y, 0u);
    let g1 = ih4(seed, x, y, 4u);
    let g2 = ih4(seed, x, y, 8u);
    var dd = 0.0;
    // Speckle path unchanged apart from tap renumbering (old taps 6/7 ->
    // 12/13): same comparison, same formula, same uniforms, so speckle
    // statistics are untouched.
    if (pcg01(seed, x, y, 12u) < prob) {
        dd = (10.0 + 25.0 * pcg01(seed, x, y, 13u)) * s;
    }
    let i = (y * dims.src_w + x) * 3u;
    dst[i] = clamp(src[i] + g0 * sigma + prm.b.x - dd, 0.0, 255.0);
    dst[i + 1u] = clamp(src[i + 1u] + g1 * sigma + prm.b.y - dd, 0.0, 255.0);
    dst[i + 2u] = clamp(src[i + 2u] + g2 * sigma + prm.b.z - dd, 0.0, 255.0);
}
"#;

/// Page bend (stage 2): edge curl (any side), book spine, or ripple.
/// Replicates the CPU stage exactly (same formulas, paper-tone BG fill).
/// Uniforms: A=(mode, curl_px, curl_shade, edge_id) with mode 0=spine,
/// 1=ripple, 2=edge and edge_id 0=left,1=right,2=top,3=bottom;
/// B=(spine_cx, spine_w, depth, 0); C=(rip_fx, rip_fy, rip_ph, 0);
/// D=(wf, hf, 0, 0).
const BEND_SHADER: &str = r#"
struct Dims { src_w : u32, src_h : u32, dst_w : u32, dst_h : u32 }
struct Bnd { a : vec4<f32>, b : vec4<f32>, c : vec4<f32>, d : vec4<f32> }

@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform> dims : Dims;
@group(0) @binding(3) var<uniform> prm : Bnd;

fn tap(ix : i32, iy : i32) -> vec3<f32> {
    if (ix < 0 || iy < 0 || ix >= i32(dims.src_w) || iy >= i32(dims.src_h)) {
        return vec3<f32>(246.0, 243.0, 236.0);
    }
    let i = (u32(iy) * dims.src_w + u32(ix)) * 3u;
    return vec3<f32>(src[i], src[i + 1u], src[i + 2u]);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.dst_w || y >= dims.dst_h) { return; }
    let fx = f32(x);
    let fy = f32(y);
    let wf = prm.d.x;
    let hf = prm.d.y;
    let curl_px = prm.a.y;
    let curl_shade = prm.a.z;
    var sx = fx;
    var sy = fy;
    var shade = 1.0;
    let mode = u32(prm.a.x);
    if (mode == 0u) {
        // Book-spine gutter: smooth squeeze + shade, continuous across fold.
        let t = (fx - prm.b.x) / max(prm.b.y, 1.0);
        let g = exp(-t * t);
        sx = prm.b.x + (fx - prm.b.x) * (1.0 - prm.b.z * g);
        shade = 1.0 - curl_shade * g;
    } else if (mode == 1u) {
        // Hand-held ripple: sinusoidal offsets, mild brightness roll.
        let p1 = 6.283185307179586 * prm.c.x * fy / hf + prm.c.z;
        let p2 = 6.283185307179586 * prm.c.y * fx / wf + prm.c.z;
        sx = fx + curl_px * 0.5 * sin(p1);
        sy = fy + curl_px * 0.3 * cos(p2);
        shade = 1.0 - curl_shade * 0.4 * abs(sin(p1));
    } else {
        // Edge curl toward one side.
        let eid = u32(prm.a.w);
        var edge = 0.0;
        var along = 0.0;
        if (eid == 0u) {
            edge = 1.0 - fx / wf;
            along = 0.6 + 0.4 * sin(3.141592653589793 * fy / hf);
            sx = fx + curl_px * edge * edge * along;
        } else if (eid == 1u) {
            edge = fx / wf;
            along = 0.6 + 0.4 * sin(3.141592653589793 * fy / hf);
            sx = fx - curl_px * edge * edge * along;
        } else if (eid == 2u) {
            edge = 1.0 - fy / hf;
            along = 0.6 + 0.4 * sin(3.141592653589793 * fx / wf);
            sy = fy + curl_px * edge * edge * along;
        } else {
            edge = fy / hf;
            along = 0.6 + 0.4 * sin(3.141592653589793 * fx / wf);
            sy = fy - curl_px * edge * edge * along;
        }
        shade = 1.0 - curl_shade * edge * edge;
    }
    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let tx = sx - f32(x0);
    let ty = sy - f32(y0);
    let top = tap(x0, y0) * (1.0 - tx) + tap(x0 + 1, y0) * tx;
    let bot = tap(x0, y0 + 1) * (1.0 - tx) + tap(x0 + 1, y0 + 1) * tx;
    let out = (top * (1.0 - ty) + bot * ty) * shade;
    let i = (y * dims.dst_w + x) * 3u;
    dst[i] = out.x;
    dst[i + 1u] = out.y;
    dst[i + 2u] = out.z;
}
"#;

const NOISE_SHADER: &str = r#"
struct Dims { src_w : u32, src_h : u32, dst_w : u32, dst_h : u32 }
struct Nse { a : vec4<f32>, b : vec4<f32>, c : vec4<f32>, d : vec4<f32> }

@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform> dims : Dims;
@group(0) @binding(3) var<uniform> prm : Nse;

// PCG-32 hash: same mixer as the grain shader (see it for the derivation
// notes). Stream-id separation via the seed pre-mix is unchanged.
fn pcg01(seed : u32, x : u32, y : u32, n : u32) -> f32 {
    var state = (seed ^ (x * 374761393u) ^ (y * 668265263u) ^ (n * 2246822519u))
        * 747796405u + 2891336453u;
    let word = ((state >> (u32(state >> 28u) + 4u)) ^ state) * 277803737u;
    let out = (word >> 22u) ^ word;
    return f32(out) / 4294967295.0;
}

// Irwin-Hall-4 Gaussian, same definition as the grain shader: channel c
// uses taps b+4*c .. b+4*c+3 (12 taps/pixel total).
fn ih4(seed : u32, x : u32, y : u32, b : u32) -> f32 {
    let s = pcg01(seed, x, y, b) + pcg01(seed, x, y, b + 1u)
          + pcg01(seed, x, y, b + 2u) + pcg01(seed, x, y, b + 3u);
    return (s - 2.0) * 1.7320508075688772;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.dst_w || y >= dims.dst_h) { return; }
    let sigma = prm.a.x;
    let seed = bitcast<u32>(prm.a.y);
    // One independent Irwin-Hall-4 Gaussian per channel (12 taps/pixel,
    // base b = 0: R taps 0-3, G taps 4-7, B taps 8-11).
    let g0 = ih4(seed, x, y, 0u);
    let g1 = ih4(seed, x, y, 4u);
    let g2 = ih4(seed, x, y, 8u);
    let i = (y * dims.src_w + x) * 3u;
    dst[i] = clamp(src[i] + g0 * sigma, 0.0, 255.0);
    dst[i + 1u] = clamp(src[i + 1u] + g1 * sigma, 0.0, 255.0);
    dst[i + 2u] = clamp(src[i + 2u] + g2 * sigma, 0.0, 255.0);
}
"#;

/// Chromatic aberration: red sampled inward along the center radius,
/// blue outward, green untouched. Uniforms: A=(cx, cy, wf, hf).
const CA_SHADER: &str = r#"
struct Dims { src_w : u32, src_h : u32, dst_w : u32, dst_h : u32 }
struct Ca { a : vec4<f32>, b : vec4<f32> }

@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform> dims : Dims;
@group(0) @binding(3) var<uniform> prm : Ca;

fn tap(ix : i32, iy : i32) -> vec3<f32> {
    if (ix < 0 || iy < 0 || ix >= i32(dims.src_w) || iy >= i32(dims.src_h)) {
        return vec3<f32>(prm.b.x, prm.b.y, prm.b.z);
    }
    let i = (u32(iy) * dims.src_w + u32(ix)) * 3u;
    return vec3<f32>(src[i], src[i + 1u], src[i + 2u]);
}

fn sample_at(sx : f32, sy : f32) -> vec3<f32> {
    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let tx = sx - f32(x0);
    let ty = sy - f32(y0);
    let top = tap(x0, y0) * (1.0 - tx) + tap(x0 + 1, y0) * tx;
    let bot = tap(x0, y0 + 1) * (1.0 - tx) + tap(x0 + 1, y0 + 1) * tx;
    return top * (1.0 - ty) + bot * ty;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.dst_w || y >= dims.dst_h) { return; }
    let fx = f32(x);
    let fy = f32(y);
    let dx = (fx - prm.a.x) / max(prm.a.z, 1.0);
    let dy = (fy - prm.a.y) / max(prm.a.w, 1.0);
    let k = prm.b.w;
    let r = sample_at(fx - dx * k, fy - dy * k);
    let b = sample_at(fx + dx * k, fy + dy * k);
    let i = (y * dims.src_w + x) * 3u;
    dst[i] = clamp(r.x, 0.0, 255.0);
    dst[i + 1u] = clamp(src[i + 1u], 0.0, 255.0);
    dst[i + 2u] = clamp(b.z, 0.0, 255.0);
}
"#;

/// Lighting: specular glare disc + directional shadow + vignette.
/// Pointwise; ports lighting stage 5 exactly.
/// Uniforms: A=(gx,gy,glare_radius,glare_alpha), B=(sh_dx,sh_dy,
/// shadow_strength,vignette_strength), C=(cx,cy,diag,max_d), D unused.
const LIGHTING_SHADER: &str = r#"
struct Dims { src_w : u32, src_h : u32, dst_w : u32, dst_h : u32 }
struct Lit { a : vec4<f32>, b : vec4<f32>, c : vec4<f32>, d : vec4<f32> }

@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform> dims : Dims;
@group(0) @binding(3) var<uniform> prm : Lit;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= dims.dst_w || y >= dims.dst_h) { return; }
    let fx = f32(x);
    let fy = f32(y);
    let gdx = fx - prm.a.x;
    let gdy = fy - prm.a.y;
    let gd = sqrt(gdx * gdx + gdy * gdy);
    let g = 1.0 - gd / max(prm.a.z, 1e-6);
    let add = prm.a.w * 255.0 * max(g, 0.0) * max(g, 0.0);
    let t = ((fx - prm.c.x) * prm.b.x + (fy - prm.c.y) * prm.b.y) / max(prm.c.z, 1e-6) + 0.5;
    let shadow = 1.0 - prm.b.z * clamp(t, 0.0, 1.0);
    let cdx = fx - prm.c.x;
    let cdy = fy - prm.c.y;
    let cd = sqrt(cdx * cdx + cdy * cdy) / max(prm.c.w, 1e-6);
    let vig = 1.0 - prm.b.w * cd * cd;
    let m = max(shadow * vig, 0.0);
    let i = (y * dims.src_w + x) * 3u;
    dst[i] = clamp(src[i] * m + add, 0.0, 255.0);
    dst[i + 1u] = clamp(src[i + 1u] * m + add, 0.0, 255.0);
    dst[i + 2u] = clamp(src[i + 2u] * m + add, 0.0, 255.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: u32, h: u32) -> Vec<f32> {
        let mut v = vec![0.0f32; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                v[i] = x as f32 * 3.0;
                v[i + 1] = y as f32 * 5.0;
                v[i + 2] = 128.0;
            }
        }
        v
    }

    /// Pure-CPU bilinear reference with the same tap order as the shader.
    fn cpu_resample(src: &[f32], w: u32, h: u32, w2: u32, h2: u32, inv: [f32; 9], fill: [f32; 3]) -> Vec<f32> {
        let at = |xx: i64, yy: i64| -> [f32; 3] {
            if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                fill
            } else {
                let i = ((yy as u32 * w + xx as u32) * 3) as usize;
                [src[i], src[i + 1], src[i + 2]]
            }
        };
        let mut out = vec![0.0f32; (w2 * h2 * 3) as usize];
        for y in 0..h2 {
            for x in 0..w2 {
                let fx = x as f32;
                let fy = y as f32;
                let ww = inv[6] * fx + inv[7] * fy + inv[8];
                let sx = (inv[0] * fx + inv[1] * fy + inv[2]) / ww;
                let sy = (inv[3] * fx + inv[4] * fy + inv[5]) / ww;
                let x0 = sx.floor() as i64;
                let y0 = sy.floor() as i64;
                let tx = sx - x0 as f32;
                let ty = sy - y0 as f32;
                let t00 = at(x0, y0);
                let t10 = at(x0 + 1, y0);
                let t01 = at(x0, y0 + 1);
                let t11 = at(x0 + 1, y0 + 1);
                let i = ((y * w2 + x) * 3) as usize;
                for c in 0..3 {
                    let top = t00[c] * (1.0 - tx) + t10[c] * tx;
                    let bot = t01[c] * (1.0 - tx) + t11[c] * tx;
                    out[i + c] = top * (1.0 - ty) + bot * ty;
                }
            }
        }
        out
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    #[test]
    fn resample_identity_matches_cpu() {
        let ctx = pollster::block_on(GpuContext::new());
        let (w, h) = (64, 48);
        let src = gradient(w, h);
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let gpu = ctx.resample(&src, w, h, w, h, id, [246.0, 243.0, 236.0]);
        let cpu = cpu_resample(&src, w, h, w, h, id, [246.0, 243.0, 236.0]);
        assert_eq!(gpu.len(), cpu.len());
        assert!(max_abs(&gpu, &cpu) < 1e-4, "identity drift");
        // Identity must reproduce the input bits (up to float noise).
        assert!(max_abs(&gpu, &src) < 1e-4);
    }

    /// Bilinear tap with paper-tone BG fill (mirrors the warp shaders).
    fn bilinear_at(src: &[f32], w: u32, h: u32, sx: f32, sy: f32) -> [f32; 3] {
        const BG: [f32; 3] = [246.0, 243.0, 236.0];
        let mut out = [0.0f32; 3];
        let x0 = sx.floor() as i64;
        let y0 = sy.floor() as i64;
        let tx = sx - x0 as f32;
        let ty = sy - y0 as f32;
        for c in 0..3 {
            let t = |xx: i64, yy: i64| -> f32 {
                if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                    BG[c]
                } else {
                    src[((yy as u32 * w + xx as u32) * 3 + c as u32) as usize]
                }
            };
            let top = t(x0, y0) * (1.0 - tx) + t(x0 + 1, y0) * tx;
            let bot = t(x0, y0 + 1) * (1.0 - tx) + t(x0 + 1, y0 + 1) * tx;
            out[c] = top * (1.0 - ty) + bot * ty;
        }
        out
    }

    fn mean_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
    }

    fn cpu_bend(src: &[f32], w: u32, h: u32, k: &BendKnobs) -> Vec<f32> {
        use std::f32::consts::{PI, TAU};
        let wf = w as f32;
        let hf = h as f32;
        let mut out = vec![0.0f32; src.len()];
        for y in 0..h {
            for x in 0..w {
                let fx = x as f32;
                let fy = y as f32;
                let (sx, sy, shade) = match k.mode {
                    0 => {
                        let t = (fx - k.spine_cx) / k.spine_w.max(1.0);
                        let g = (-t * t).exp();
                        (
                            k.spine_cx + (fx - k.spine_cx) * (1.0 - k.depth * g),
                            fy,
                            1.0 - k.curl_shade * g,
                        )
                    }
                    1 => {
                        let p1 = TAU * k.rip_fx * fy / hf + k.rip_ph;
                        let p2 = TAU * k.rip_fy * fx / wf + k.rip_ph;
                        (
                            fx + k.curl_px * 0.5 * p1.sin(),
                            fy + k.curl_px * 0.3 * p2.cos(),
                            1.0 - k.curl_shade * 0.4 * p1.sin().abs(),
                        )
                    }
                    _ => {
                        let (edge, along) = match k.edge_id {
                            1 => (fx / wf, 0.6 + 0.4 * (PI * fy / hf).sin()),
                            2 => (1.0 - fy / hf, 0.6 + 0.4 * (PI * fx / wf).sin()),
                            3 => (fy / hf, 0.6 + 0.4 * (PI * fx / wf).sin()),
                            _ => (1.0 - fx / wf, 0.6 + 0.4 * (PI * fy / hf).sin()),
                        };
                        let shift = k.curl_px * edge * edge * along;
                        let (sx, sy) = match k.edge_id {
                            1 => (fx - shift, fy),
                            2 => (fx, fy + shift),
                            3 => (fx, fy - shift),
                            _ => (fx + shift, fy),
                        };
                        (sx, sy, 1.0 - k.curl_shade * edge * edge)
                    }
                };
                let p = bilinear_at(src, w, h, sx, sy);
                let i = ((y * w + x) * 3) as usize;
                out[i] = p[0] * shade;
                out[i + 1] = p[1] * shade;
                out[i + 2] = p[2] * shade;
            }
        }
        out
    }

    fn bend_knobs(mode: u32, edge_id: u32) -> BendKnobs {
        BendKnobs {
            mode,
            edge_id,
            curl_px: 4.0,
            curl_shade: 0.3,
            spine_cx: 30.0,
            spine_w: 5.0,
            depth: 0.12,
            rip_fx: 1.5,
            rip_fy: 1.2,
            rip_ph: 0.7,
        }
    }

    fn cpu_gauss(src: &[f32], w: u32, h: u32, sigma: f32) -> Vec<f32> {
        let r = (3.0 * sigma).ceil() as i32;
        if r <= 0 {
            return src.to_vec();
        }
        let s2 = 2.0 * sigma * sigma;
        let mut out = vec![0.0f32; src.len()];
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f32; 3];
                let mut norm = 0.0f32;
                for ky in -r..=r {
                    for kx in -r..=r {
                        let wgt = (-((kx * kx + ky * ky) as f32) / s2).exp();
                        let cx = (x as i32 + kx).clamp(0, w as i32 - 1) as u32;
                        let cy = (y as i32 + ky).clamp(0, h as i32 - 1) as u32;
                        let j = ((cy * w + cx) * 3) as usize;
                        for c in 0..3 {
                            acc[c] += src[j + c] * wgt;
                        }
                        norm += wgt;
                    }
                }
                let i = ((y * w + x) * 3) as usize;
                for c in 0..3 {
                    out[i + c] = acc[c] / norm;
                }
            }
        }
        out
    }

    #[test]
    fn grain_stats_and_determinism() {
        let ctx = pollster::block_on(GpuContext::new());
        let (w, h) = (128u32, 128u32);
        let src = vec![128.0f32; (w * h * 3) as usize];
        let k = GrainKnobs { sigma: 4.0, seed: 777, speckle_prob: 0.001, strength: 0.6, warm: [2.0, 1.0, -1.0] };
        let a = ctx.grain(&src, w, h, k);
        let b = ctx.grain(&src, w, h, k);
        assert_eq!(a, b, "grain must be deterministic");
        let c = ctx.grain(&src, w, h, GrainKnobs { seed: 778, ..k });
        assert_ne!(a, c, "different seeds must differ");
        for ch in 0..3 {
            let vals: Vec<f64> = a.iter().skip(ch).step_by(3).map(|&v| v as f64).collect();
            let n = vals.len() as f64;
            let m = vals.iter().sum::<f64>() / n;
            let sd = (vals.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / n).sqrt();
            assert!((m - (128.0 + k.warm[ch] as f64)).abs() < 0.5, "ch{ch} mean {m}");
            assert!((sd - 4.0).abs() < 0.6, "ch{ch} std {sd}");
        }
    }

    #[test]
    fn bend_matches_cpu_all_modes() {
        let ctx = pollster::block_on(GpuContext::new());
        let (w, h) = (64u32, 48u32);
        let src = gradient(w, h);
        for (mode, edge) in [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3)] {
            let k = bend_knobs(mode, edge);
            let gpu = ctx.bend(&src, w, h, k);
            let cpu = cpu_bend(&src, w, h, &k);
            let m = mean_abs(&gpu, &cpu);
            let d = max_abs(&gpu, &cpu);
            assert!(m < 1e-4, "mode {mode} edge {edge}: mean drift {m}");
            assert!(d < 0.05, "mode {mode} edge {edge}: max drift {d}");
        }
    }

    #[test]
    fn blur_radius_zero_is_identity() {
        let ctx = pollster::block_on(GpuContext::new());
        let (w, h) = (64u32, 48u32);
        let src = gradient(w, h);
        let gpu = ctx.blur(&src, w, h, 0.0);
        assert_eq!(gpu, src, "radius 0 must be bit-exact identity");
    }

    #[test]
    fn blur_matches_direct_gaussian() {
        let ctx = pollster::block_on(GpuContext::new());
        let (w, h) = (48u32, 32u32);
        let src = gradient(w, h);
        for sigma in [0.8, 1.5, 2.5] {
            let gpu = ctx.blur(&src, w, h, sigma);
            let cpu = cpu_gauss(&src, w, h, sigma);
            let m = mean_abs(&gpu, &cpu);
            let d = max_abs(&gpu, &cpu);
            assert!(m < 1e-3, "sigma={sigma}: mean drift {m}");
            assert!(d < 0.5, "sigma={sigma}: max drift {d}");
        }
        // Flat field is preserved (kernel is normalized).
        let flat = vec![200.0f32; (w * h * 3) as usize];
        let g = ctx.blur(&flat, w, h, 1.5);
        assert!(g.iter().all(|&v| (v - 200.0).abs() < 1e-3), "flat field must survive");
    }

    fn cpu_chromatic(src: &[f32], w: u32, h: u32, ca: f32) -> Vec<f32> {
        // Transcription of stage 5b with the paper-tone out-of-source fill
        // (per-channel BG, exactly like the CPU sampler: 246/243/236).
        let at = |buf: &[f32], xx: f32, yy: f32| -> [f32; 3] {
            let x0 = xx.floor() as i64;
            let y0 = yy.floor() as i64;
            let tx = xx - x0 as f32;
            let ty = yy - y0 as f32;
            let mut t = [[246.0f32, 243.0, 236.0]; 4];
            for (k, (ox, oy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].iter().enumerate() {
                let (qx, qy) = (x0 + ox, y0 + oy);
                if qx >= 0 && qy >= 0 && qx < w as i64 && qy < h as i64 {
                    let i = ((qy as u32 * w + qx as u32) * 3) as usize;
                    t[k] = [buf[i], buf[i + 1], buf[i + 2]];
                }
            }
            let mut o = [0.0f32; 3];
            for c in 0..3 {
                let top = t[0][c] * (1.0 - tx) + t[1][c] * tx;
                let bot = t[2][c] * (1.0 - tx) + t[3][c] * tx;
                o[c] = top * (1.0 - ty) + bot * ty;
            }
            o
        };
        let mut out = src.to_vec();
        let (wf, hf) = (w as f32, h as f32);
        for y in 0..h {
            for x in 0..w {
                let dx = (x as f32 - wf / 2.0) / wf.max(1.0);
                let dy = (y as f32 - hf / 2.0) / hf.max(1.0);
                let r = at(src, x as f32 - dx * ca, y as f32 - dy * ca);
                let b = at(src, x as f32 + dx * ca, y as f32 + dy * ca);
                let i = ((y * w + x) * 3) as usize;
                out[i] = r[0].clamp(0.0, 255.0);
                out[i + 2] = b[2].clamp(0.0, 255.0);
            }
        }
        out
    }

    #[test]
    fn chromatic_zero_is_identity() {
        let ctx = pollster::block_on(GpuContext::new());
        let (w, h) = (64, 48);
        let src = gradient(w, h);
        let gpu = ctx.chromatic(&src, w, h, 0.0);
        assert!(max_abs(&gpu, &src) == 0.0, "zero CA must be bit-exact identity");
    }

    #[test]
    fn chromatic_matches_cpu() {
        let ctx = pollster::block_on(GpuContext::new());
        let (w, h) = (64, 48);
        let src = gradient(w, h);
        let gpu = ctx.chromatic(&src, w, h, 1.2);
        let cpu = cpu_chromatic(&src, w, h, 1.2);
        let d = max_abs(&gpu, &cpu);
        if d >= 5e-3 {
            let mut bi = 0;
            for (i, (a, b)) in gpu.iter().zip(cpu.iter()).enumerate() {
                if (a - b).abs() == d {
                    bi = i;
                    break;
                }
            }
            let pi = bi / 3;
            println!("worst px=({},{}) c={} gpu={} cpu={}", pi % w as usize, pi / w as usize, bi % 3, gpu[bi], cpu[bi]);
            println!("  R gpu={} cpu={}", gpu[pi * 3], cpu[pi * 3]);
            println!("  G gpu={} cpu={}", gpu[pi * 3 + 1], cpu[pi * 3 + 1]);
            println!("  B gpu={} cpu={}", gpu[pi * 3 + 2], cpu[pi * 3 + 2]);
        }
        assert!(d < 5e-3, "chromatic drift {d}");
    }

    #[test]
    fn noise_is_deterministic_with_gaussian_stats() {
        let ctx = pollster::block_on(GpuContext::new());
        let (w, h) = (128u32, 128u32);
        // Mid-grey avoids clamp distortion so mean/std are clean.
        let src = vec![128.0f32; (w * h * 3) as usize];
        let a = ctx.noise(&src, w, h, 8.0, 1234);
        let b = ctx.noise(&src, w, h, 8.0, 1234);
        assert_eq!(a, b, "same seed must give same bytes");
        let c = ctx.noise(&src, w, h, 8.0, 9999);
        assert_ne!(a, c, "different seeds must differ");
        assert!(a.iter().all(|&v| (0.0..=255.0).contains(&v)));
        let n = a.len() as f64;
        let mean: f64 = a.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var: f64 = a.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();
        assert!((mean - 128.0).abs() < 0.3, "mean {mean}");
        assert!((std - 8.0).abs() < 0.5, "std {std}");
    }

    fn cpu_lighting(src: &[f32], w: u32, h: u32, k: LightingKnobs) -> Vec<f32> {
        let wf = w as f32;
        let hf = h as f32;
        let cx = wf / 2.0;
        let cy = hf / 2.0;
        let diag = (wf * wf + hf * hf).sqrt().max(1.0);
        let max_d = (cx * cx + cy * cy).sqrt().max(1.0);
        let mut out = src.to_vec();
        for y in 0..h {
            for x in 0..w {
                let fx = x as f32;
                let fy = y as f32;
                let gd = ((fx - k.gx).powi(2) + (fy - k.gy).powi(2)).sqrt();
                let g = (1.0 - gd / k.glare_radius).max(0.0);
                let add = k.glare_alpha * 255.0 * g * g;
                let t = ((fx - cx) * k.sh_dx + (fy - cy) * k.sh_dy) / diag + 0.5;
                let shadow = 1.0 - k.shadow_strength * t.clamp(0.0, 1.0);
                let cd = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt() / max_d;
                let vig = 1.0 - k.vignette_strength * cd * cd;
                let m = (shadow * vig).max(0.0);
                let i = ((y * w + x) * 3) as usize;
                out[i] = (src[i] * m + add).clamp(0.0, 255.0);
                out[i + 1] = (src[i + 1] * m + add).clamp(0.0, 255.0);
                out[i + 2] = (src[i + 2] * m + add).clamp(0.0, 255.0);
            }
        }
        out
    }

    #[test]
    fn lighting_zero_knobs_is_identity() {
        let ctx = pollster::block_on(GpuContext::new());
        let (w, h) = (64, 48);
        let src = gradient(w, h);
        let zero = LightingKnobs {
            gx: 0.0, gy: 0.0, glare_radius: 10.0, glare_alpha: 0.0,
            sh_dx: 1.0, sh_dy: 0.0, shadow_strength: 0.0, vignette_strength: 0.0,
        };
        let gpu = ctx.lighting(&src, w, h, zero);
        assert!(max_abs(&gpu, &src) == 0.0, "zero lighting must be bit-exact identity");
    }

    #[test]
    fn lighting_matches_cpu() {
        let ctx = pollster::block_on(GpuContext::new());
        let (w, h) = (64, 48);
        let src = gradient(w, h);
        let th = 290.0f32.to_radians();
        let k = LightingKnobs {
            gx: 20.0, gy: 30.0, glare_radius: 25.0, glare_alpha: 0.3,
            sh_dx: th.cos(), sh_dy: th.sin(), shadow_strength: 0.24, vignette_strength: 0.1,
        };
        let gpu = ctx.lighting(&src, w, h, k);
        let cpu = cpu_lighting(&src, w, h, k);
        let d = max_abs(&gpu, &cpu);
        assert!(d < 1e-2, "lighting drift {d}");
    }

    #[test]
    fn resample_projective_matches_cpu() {
        let ctx = pollster::block_on(GpuContext::new());
        let (w, h) = (64, 48);
        let src = gradient(w, h);
        // Gentle projective warp with in- and out-of-frame taps.
        let inv = [1.04, 0.03, -2.0, -0.02, 0.97, 1.5, 0.0004, -0.0003, 1.0];
        let gpu = ctx.resample(&src, w, h, 70, 52, inv, [246.0, 243.0, 236.0]);
        let cpu = cpu_resample(&src, w, h, 70, 52, inv, [246.0, 243.0, 236.0]);
        let d = max_abs(&gpu, &cpu);
        // GPU compilers may contract a*b+c into FMA; a few ulps on ~200-scale
        // values is inherent GPU-vs-CPU noise, far below JPEG/sensor noise.
        assert!(d < 5e-3, "projective drift {d}");

    }

    // ---- resident chain (`chain_full` on the shared device) ----

    fn chain_cfg_all_on() -> ChainConfig {
        let th = 290.0f32.to_radians();
        ChainConfig {
            grain_sigma: 2.5,
            grain_seed: 0x1234_5678,
            grain_speckle_prob: 0.001,
            grain_strength: 0.6,
            grain_warm: [4.0, 2.0, -1.0],
            bend: bend_knobs(1, 0),
            persp_inv: [1.03, 0.02, -3.0, -0.015, 0.98, 2.0, 0.00003, -0.00002, 1.0],
            rot_deg: 5.0,
            blur_sigma: 1.2,
            light: LightingKnobs {
                gx: 40.0, gy: 30.0, glare_radius: 60.0, glare_alpha: 0.3,
                sh_dx: th.cos(), sh_dy: th.sin(),
                shadow_strength: 0.24, vignette_strength: 0.1,
            },
            ca_px: 0.8,
            noise_sigma: 6.0,
            noise_seed: 0x9abc_def0,
            run_grain: true,
            run_bend: true,
            run_persp: true,
            run_rot: true,
            run_blur: true,
            run_light: true,
            run_ca: true,
            run_noise: true,
            tilt_active: false,
            focus_y: 0.5,
            focus_h: 0.2,
            ..ChainConfig::default()
        }
    }

    /// Independent rotation frame (same formula as `rotation_frame`, written
    /// out so the reference shares no chain executor code).
    fn ref_frame(rot_deg: f32, w: u32, h: u32) -> ([f32; 9], u32, u32) {
        let th = rot_deg.to_radians();
        let (sn, cs) = (th.sin().abs(), th.cos().abs());
        let w2 = ((w as f32) * cs + (h as f32) * sn).ceil().max(1.0) as u32;
        let h2 = ((w as f32) * sn + (h as f32) * cs).ceil().max(1.0) as u32;
        let (snt, cst) = (th.sin(), th.cos());
        let (cx0, cy0) = (w as f32 / 2.0, h as f32 / 2.0);
        let (cx1, cy1) = (w2 as f32 / 2.0, h2 as f32 / 2.0);
        (
            [
                cst, snt, cx0 - cst * cx1 - snt * cy1,
                -snt, cst, cy0 + snt * cx1 - cst * cy1,
                0.0, 0.0, 1.0,
            ],
            w2,
            h2,
        )
    }

    fn try_ctx() -> Option<GpuContext> {
        match pollster::block_on(GpuContext::try_new()) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("SKIP chain test (no GPU): {e}");
                None
            }
        }
    }

    #[test]
    fn shared_chain_bit_exact_vs_separate_calls() {
        let ctx = match try_ctx() {
            Some(c) => c,
            None => return,
        };
        let (w, h) = (64u32, 48u32);
        let cfg = chain_cfg_all_on();
        let src = gradient(w, h);
        let (chained, cw, ch) = ctx.chain_full(&src, w, h, &cfg);
        // Same knobs, same order, one call per stage (the current path).
        let (rinv, w2, h2) = ref_frame(cfg.rot_deg, w, h);
        let r1 = ctx.grain(
            &src,
            w,
            h,
            GrainKnobs {
                sigma: cfg.grain_sigma,
                seed: cfg.grain_seed,
                speckle_prob: cfg.grain_speckle_prob,
                strength: cfg.grain_strength,
                warm: cfg.grain_warm,
            },
        );
        let r2 = ctx.bend(&r1, w, h, cfg.bend);
        let r3 = ctx.resample(&r2, w, h, w, h, cfg.persp_inv, [246.0, 243.0, 236.0]);
        let r4 = ctx.resample(&r3, w, h, w2, h2, rinv, [25.0, 25.0, 30.0]);
        let r5 = ctx.blur(&r4, w2, h2, cfg.blur_sigma);
        let r6 = ctx.lighting(&r5, w2, h2, cfg.light);
        let r7 = ctx.chromatic(&r6, w2, h2, cfg.ca_px);
        let seq = ctx.noise(&r7, w2, h2, cfg.noise_sigma, cfg.noise_seed);
        assert_eq!((cw, ch), (w2, h2), "chain dims must match rotation frame");
        assert_eq!(chained.len(), seq.len());
        assert_eq!(chained, seq, "shared chain must be bit-exact vs separate calls");
    }

    #[test]
    fn shared_chain_is_deterministic() {
        let ctx = match try_ctx() {
            Some(c) => c,
            None => return,
        };
        let (w, h) = (64u32, 48u32);
        let cfg = chain_cfg_all_on();
        let src = gradient(w, h);
        let (a, aw, ah) = ctx.chain_full(&src, w, h, &cfg);
        let (b, bw, bh) = ctx.chain_full(&src, w, h, &cfg);
        assert_eq!((aw, ah), (bw, bh));
        assert_eq!(a, b, "same input must give same bytes");
    }

    #[test]
    fn shared_chain_all_skipped_is_identity() {
        // Every pass skipped: the upload must come back untouched (this is
        // the tiny-image path where no gate fires).
        let ctx = match try_ctx() {
            Some(c) => c,
            None => return,
        };
        let (w, h) = (64u32, 48u32);
        let mut cfg = chain_cfg_all_on();
        cfg.run_grain = false;
        cfg.run_bend = false;
        cfg.run_persp = false;
        cfg.run_rot = false;
        cfg.run_blur = false;
        cfg.run_light = false;
        cfg.run_ca = false;
        cfg.run_noise = false;
        let src = gradient(w, h);
        let (out, ow, oh) = ctx.chain_full(&src, w, h, &cfg);
        assert_eq!((ow, oh), (w, h));
        assert_eq!(out, src, "all-skipped chain must be bit-exact identity");
    }

    #[test]
    fn shared_chain_split_matches_single_submit() {
        // The pipeline seam (CPU framing between perspective and rotation)
        // splits the chain in two submits; the split must be bit-exact vs
        // one submit over all eight stages (f32 upload/download round-trips
        // are exact, no conversion involved).
        let ctx = match try_ctx() {
            Some(c) => c,
            None => return,
        };
        let (w, h) = (64u32, 48u32);
        let full = chain_cfg_all_on();
        let src = gradient(w, h);
        let (whole, ww, wh) = ctx.chain_full(&src, w, h, &full);
        let mut pre = full;
        pre.run_rot = false;
        pre.run_blur = false;
        pre.run_light = false;
        pre.run_ca = false;
        pre.run_noise = false;
        let (mid, mw, mh) = ctx.chain_full(&src, w, h, &pre);
        assert_eq!((mw, mh), (w, h), "pre segment keeps dims");
        let mut post = full;
        post.run_grain = false;
        post.run_bend = false;
        post.run_persp = false;
        let (split, sw, sh) = ctx.chain_full(&mid, mw, mh, &post);
        assert_eq!((sw, sh), (ww, wh));
        assert_eq!(split, whole, "split chain must be bit-exact vs single submit");
    }

    #[test]
    fn shared_chain_tilt_matches_cpu_blend() {
        // Tilt mode blends sharp/blurred in-kernel; the reference is the
        // same blur plus the `photo-degrade` smoothstep blend on CPU. Only
        // the blend arithmetic may differ (GPU FMA contraction), so this is
        // a tight parity bound, not bit-exactness.
        let ctx = match try_ctx() {
            Some(c) => c,
            None => return,
        };
        let (w, h) = (64u32, 48u32);
        let src = gradient(w, h);
        let (sigma, focus_y, focus_h) = (1.0f32, 0.45f32, 0.2f32);
        let blurred = ctx.blur(&src, w, h, sigma);
        let mut expect = vec![0.0f32; src.len()];
        for y in 0..h {
            let d = ((y as f32 / h.max(1) as f32) - focus_y).abs();
            let mut m = ((d - focus_h) / 0.08).clamp(0.0, 1.0);
            m = m * m * (3.0 - 2.0 * m);
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                for c in 0..3 {
                    expect[i + c] = src[i + c] * (1.0 - m) + blurred[i + c] * m;
                }
            }
        }
        let mut cfg = chain_cfg_all_on();
        cfg.run_grain = false;
        cfg.run_bend = false;
        cfg.run_persp = false;
        cfg.run_rot = false;
        cfg.run_blur = true;
        cfg.run_light = false;
        cfg.run_ca = false;
        cfg.run_noise = false;
        cfg.blur_sigma = sigma;
        cfg.tilt_active = true;
        cfg.focus_y = focus_y;
        cfg.focus_h = focus_h;
        let (got, gw, gh) = ctx.chain_full(&src, w, h, &cfg);
        assert_eq!((gw, gh), (w, h));
        let d = max_abs(&got, &expect);
        assert!(d < 1e-3, "tilt blend drift {d}");
        let (again, _, _) = ctx.chain_full(&src, w, h, &cfg);
        assert_eq!(got, again, "tilt chain must be deterministic");
    }

    #[test]
    fn shared_chain_with_affine_matches_separate_calls() {
        // Fusing the terminal rotate+downscale affine pass into `chain_full`
        // must be bit-exact vs running `chain_full` followed by `affine`.
        let ctx = match try_ctx() {
            Some(c) => c,
            None => return,
        };
        let (w, h) = (64u32, 48u32);
        let src = gradient(w, h);
        let (rows, meta, _, _, nw, nh) = rotscale_frame(5.0, w, h, 0.75, 0.75);
        let mut base = chain_cfg_all_on();
        base.run_grain = false;
        base.run_bend = false;
        base.run_persp = false;
        base.run_rot = false;
        let (mid, mw, mh) = ctx.chain_full(&src, w, h, &base);
        assert_eq!((mw, mh), (w, h));
        let seq = ctx.affine(&mid, mw, mh, nw, nh, rows, meta, [25.0, 25.0, 30.0]);
        let mut fused = base;
        fused.run_affine = true;
        fused.affine_rows = rows;
        fused.affine_meta = meta;
        fused.affine_nw = nw;
        fused.affine_nh = nh;
        let (got, gw, gh) = ctx.chain_full(&src, w, h, &fused);
        assert_eq!((gw, gh), (nw, nh));
        assert_eq!(got, seq, "chain_full + run_affine must be bit-exact vs separate affine");
    }

    // ---- fused rotate+downscale affine pass ----

    /// Test-only CPU mirror of the affine kernel math (§3 expression tree):
    /// explicit `xr = x*sx_inv`, then R^-1 rows, desk fill on BOTH OOB
    /// levels, same tap blend order as the WGSL. Models the GPU kernel, NOT
    /// the production CPU rotation loop (which blends inner taps toward
    /// paper) — any "make the mirror match legacy" drift defeats its
    /// purpose.
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
        let at = |xx: i64, yy: i64| -> [f32; 3] {
            if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                fill
            } else {
                let i = ((yy as u32 * w + xx as u32) * 3) as usize;
                [src[i], src[i + 1], src[i + 2]]
            }
        };
        let mut out = vec![0.0f32; (nw * nh * 3) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let xr = x as f32 * meta[2];
                let yr = y as f32 * meta[3];
                let i = ((y * nw + x) * 3) as usize;
                if xr < 0.0 || yr < 0.0 || xr >= meta[0] || yr >= meta[1] {
                    out[i] = fill[0];
                    out[i + 1] = fill[1];
                    out[i + 2] = fill[2];
                    continue;
                }
                let sx_ = rows[0] * xr + rows[1] * yr + rows[2];
                let sy_ = rows[3] * xr + rows[4] * yr + rows[5];
                let x0 = sx_.floor() as i64;
                let y0 = sy_.floor() as i64;
                let tx = sx_ - x0 as f32;
                let ty = sy_ - y0 as f32;
                let t00 = at(x0, y0);
                let t10 = at(x0 + 1, y0);
                let t01 = at(x0, y0 + 1);
                let t11 = at(x0 + 1, y0 + 1);
                for c in 0..3 {
                    let top = t00[c] * (1.0 - tx) + t10[c] * tx;
                    let bot = t01[c] * (1.0 - tx) + t11[c] * tx;
                    out[i + c] = top * (1.0 - ty) + bot * ty;
                }
            }
        }
        out
    }

    #[test]
    fn rotscale_frame_pure_math_no_gpu() {
        // Identity: deg=0, sx=sy=1 -> identity rows, dims unchanged.
        let (rows, meta, w2, h2, nw, nh) = rotscale_frame(0.0, 64, 48, 1.0, 1.0);
        assert_eq!((w2, h2, nw, nh), (64, 48, 64, 48));
        assert_eq!(rows, [1.0, 0.0, 0.0, -0.0, 1.0, 0.0]);
        assert_eq!(meta, [64.0, 48.0, 1.0, 1.0]);
        // Dims match the legacy formula (written out inline, not via the
        // helper, so the check is not circular) at full and train res.
        for (w, h) in [(2480u32, 3508u32), (1024u32, 1447u32)] {
            for deg in [-10.0f32, -3.25, 0.5, 7.75, 10.0] {
                for (sx, sy) in [(1.0, 1.0), (0.75, 0.75), (0.5, 0.9), (1.0, 0.6)] {
                    let th = deg.to_radians();
                    let (sn, cs) = (th.sin().abs(), th.cos().abs());
                    let ew2 = ((w as f32) * cs + (h as f32) * sn).ceil().max(1.0) as u32;
                    let eh2 = ((w as f32) * sn + (h as f32) * cs).ceil().max(1.0) as u32;
                    let enw = (ew2 as f32 * sx) as u32;
                    let enh = (eh2 as f32 * sy) as u32;
                    let (_, _, gw2, gh2, gnw, gnh) = rotscale_frame(deg, w, h, sx, sy);
                    assert_eq!((gw2, gh2, gnw, gnh), (ew2, eh2, enw, enh),
                        "deg={deg} sx={sx} sy={sy}");
                }
            }
        }
        // Degenerate sx=sy=1 rows equal `rotation_inv` bit-for-bit.
        let (rows, _, w2, h2, _, _) = rotscale_frame(5.0, 64, 48, 1.0, 1.0);
        let rinv = crate::chain::rotation_inv(5.0, 64, 48, w2, h2);
        assert_eq!(rows, [rinv[0], rinv[1], rinv[2], rinv[3], rinv[4], rinv[5]]);
        // Dst center maps back to src center within 2 px (truncation only).
        let (w, h) = (1024u32, 1447u32);
        let (rows, meta, _, _, nw, nh) = rotscale_frame(5.0, w, h, 0.75, 0.75);
        let xr = (nw as f32 / 2.0) * meta[2];
        let yr = (nh as f32 / 2.0) * meta[3];
        let sx_ = rows[0] * xr + rows[1] * yr + rows[2];
        let sy_ = rows[3] * xr + rows[4] * yr + rows[5];
        assert!((sx_ - w as f32 / 2.0).abs() < 2.0, "center x {sx_}");
        assert!((sy_ - h as f32 / 2.0).abs() < 2.0, "center y {sy_}");
    }

    #[test]
    fn affine_rotation_only_matches_resample_tolerance() {
        // Fused (deg!=0, sx=sy=1) vs `resample` rotation-only on a gradient
        // AND a luma ramp: max-abs <= 2 LSB. (Bit-exactness is NOT claimed
        // fused-vs-sequential in general — different bilinear support — but
        // the degenerate case shares the expression tree, so expect ~0.)
        let ctx = match try_ctx() {
            Some(c) => c,
            None => return,
        };
        let (w, h) = (64u32, 48u32);
        let src = gradient(w, h);
        let mut ramp = vec![0.0f32; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let v = (x + y * w as u32) as f32 / (w * h) as f32 * 255.0;
                ramp[i] = v;
                ramp[i + 1] = v;
                ramp[i + 2] = v;
            }
        }
        for fixture in [&src, &ramp] {
            let (rinv, w2, h2) = ref_frame(5.0, w, h);
            let expect = ctx.resample(fixture, w, h, w2, h2, rinv, [25.0, 25.0, 30.0]);
            let (rows, meta, _, _, nw, nh) = rotscale_frame(5.0, w, h, 1.0, 1.0);
            assert_eq!((nw, nh), (w2, h2));
            let got = ctx.affine(fixture, w, h, nw, nh, rows, meta, [25.0, 25.0, 30.0]);
            assert_eq!(got.len(), expect.len());
            let d = max_abs(&got, &expect);
            let m = mean_abs(&got, &expect);
            assert!(d <= 2.0, "affine rotation-only max drift {d}");
            assert!(m < 1e-3, "affine rotation-only mean drift {m}");
        }
    }

    #[test]
    fn affine_matches_cpu_mirror_tolerance() {
        // CPU-mirror (§6) vs GPU-fused with an active downscale on both
        // fixtures: same bar. This is the cross-backend bridge — if it
        // passes, the WGSL transcription is proven.
        let ctx = match try_ctx() {
            Some(c) => c,
            None => return,
        };
        let (w, h) = (64u32, 48u32);
        let src = gradient(w, h);
        for deg in [5.0, -7.5] {
            for (sx, sy) in [(0.75, 0.75), (0.5, 0.9)] {
                let (rows, meta, _, _, nw, nh) = rotscale_frame(deg, w, h, sx, sy);
                let fill = [25.0, 25.0, 30.0];
                let gpu = ctx.affine(&src, w, h, nw, nh, rows, meta, fill);
                let cpu = cpu_affine_mirror(&src, w, h, nw, nh, rows, meta, fill);
                assert_eq!(gpu.len(), cpu.len());
                let d = max_abs(&gpu, &cpu);
                assert!(d <= 2.0, "deg={deg} sx={sx} sy={sy}: mirror drift {d}");
            }
        }
    }

    // ---- Irwin-Hall-4 statistical gates (spec §2 G1) ----

    fn flat_stats(out: &[f32]) -> ([f64; 3], [f64; 3]) {
        let mut mean = [0.0f64; 3];
        let mut std = [0.0f64; 3];
        for ch in 0..3 {
            let vals: Vec<f64> = out.iter().skip(ch).step_by(3).map(|&v| v as f64).collect();
            let n = vals.len() as f64;
            let m = vals.iter().sum::<f64>() / n;
            let sd = (vals.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / n).sqrt();
            mean[ch] = m;
            std[ch] = sd;
        }
        (mean, std)
    }

    #[test]
    fn ih4_flat_field_stats() {
        // G1: 256x256 flat 128 field. Grain sigma=3 -> mean 128 +/- 0.15
        // per channel, std 3.0 +/- 0.15; noise sigma=8 -> mean +/- 0.2,
        // std +/- 0.3. Second production sigmas (grain 5, noise 3) in
        // +/-5% std bands. N=196k samples/stage, so sampling error is ~0.02.
        let ctx = match try_ctx() {
            Some(c) => c,
            None => return,
        };
        let (w, h) = (256u32, 256u32);
        let flat = vec![128.0f32; (w * h * 3) as usize];
        let g = ctx.grain(
            &flat, w, h,
            GrainKnobs { sigma: 3.0, seed: 0x1A2B_3C4D, speckle_prob: 0.0, strength: 0.0, warm: [0.0; 3] },
        );
        let (m, s) = flat_stats(&g);
        for ch in 0..3 {
            assert!((m[ch] - 128.0).abs() < 0.15, "grain ch{ch} mean {}", m[ch]);
            assert!((s[ch] - 3.0).abs() < 0.15, "grain ch{ch} std {}", s[ch]);
        }
        let n = ctx.noise(&flat, w, h, 8.0, 0x5EED_1234);
        let (m, s) = flat_stats(&n);
        for ch in 0..3 {
            assert!((m[ch] - 128.0).abs() < 0.2, "noise ch{ch} mean {}", m[ch]);
            assert!((s[ch] - 8.0).abs() < 0.3, "noise ch{ch} std {}", s[ch]);
        }
        let g5 = ctx.grain(
            &flat, w, h,
            GrainKnobs { sigma: 5.0, seed: 42, speckle_prob: 0.0, strength: 0.0, warm: [0.0; 3] },
        );
        let (_, s) = flat_stats(&g5);
        for ch in 0..3 {
            assert!((s[ch] - 5.0).abs() < 0.25, "grain5 ch{ch} std {}", s[ch]);
        }
        let n3 = ctx.noise(&flat, w, h, 3.0, 43);
        let (_, s) = flat_stats(&n3);
        for ch in 0..3 {
            assert!((s[ch] - 3.0).abs() < 0.15, "noise3 ch{ch} std {}", s[ch]);
        }
    }
}
