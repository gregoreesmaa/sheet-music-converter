//! photo-degrade: clean PNG -> degraded "phone photo" PNG.
//!
//! Applies, in order: paper texture, page-curl warp + shading, perspective
//! warp, Gaussian blur, glare + shadow lighting, sensor noise, and a JPEG
//! compression round-trip. All randomness comes from a seeded ChaCha8 RNG so
//! runs are reproducible. Every output image is paired with a JSON file that
//! records the seed and every sampled parameter.
//!
//! Usage:
//!   photo-degrade --input clean.png --out-dir out --seed 42 --strength 0.6
//!   photo-degrade --input clean-dir/ --out-dir out --seed 7

use clap::Parser;
use image::{ImageFormat, RgbImage};

mod banks;
mod degrade_capture;
mod degrade_ink;
mod degrade_paper;
mod degrade_wear;
mod gpu_backend;
use degrade_gpu::chain::ChainConfig;
use degrade_gpu::gpu::{BendKnobs, LightingKnobs};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::path::{Path, PathBuf};

/// Degrade clean sheet-music PNGs into realistic phone photos.
#[derive(Parser, Debug)]
#[command(name = "photo-degrade", version)]
struct Args {
    /// Input clean PNG file, or a directory of PNGs (non-recursive).
    #[arg(long)]
    input: PathBuf,

    /// Directory that receives `<stem>{suffix}.png|jpg` + `<stem>{suffix}_params.json`
    /// (suffix defaults to `_degraded`, so simultaneous runs with different
    /// suffixes never share params files).
    #[arg(long, default_value = "out")]
    out_dir: PathBuf,

    /// Master seed. For directory input, file `i` uses
    /// `seed + i * 0x9E3779B97F4A7C15` (wrapping multiply stride).
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Global effect strength in [0, 1]. Scales every effect.
    #[arg(long, default_value_t = 0.5)]
    strength: f32,

    /// JPEG quality override (1..100). If omitted, sampled per image.
    #[arg(long)]
    jpeg_quality: Option<u8>,

    /// Downscale inputs wider than this to this width (aspect kept, 0 = keep
    /// native). Phone photos are ~12 MP, not 139 MP engraver exports.
    #[arg(long, default_value_t = 2480)]
    max_width: u32,

    /// Degrade at model-training resolution: downscale the (post-maxwidth)
    /// input to this width (aspect kept) BEFORE any degrade stage, and
    /// rescale every absolute-pixel statistic (blur/grain/noise sigmas,
    /// defect radii, banding periods) by k = width/native so physical sizes
    /// stay scale-correct. 0 = off (legacy full-res path, bit-identical).
    /// Canonical training width is 1024: staff space there is ~4.1 px vs
    /// ~10 px at 2480 (see TRAINING_BIG500.md sec. 4), still resolvable,
    /// and 1024 is the standard detector input size. Recorded as
    /// `train_width`/`rescale_k` in params.
    #[arg(long, default_value_t = 0)]
    train_width: u32,

    /// Random output downscale per image, sampled uniformly in
    /// [scale-min, 1.0] for capture-resolution variety (1.0 = keep).
    #[arg(long, default_value_t = 0.5)]
    scale_min: f32,

    /// Filename suffix for degraded images (before the extension).
    #[arg(long, default_value = "_degraded")]
    suffix: String,

    /// Output image format: `png` (lossless) or `jpeg` (phone-photo-like).
    #[arg(long, default_value = "png")]
    format: String,
    /// Require the WGPU GPU backend (fail if no usable GPU adapter is found).
    /// By default, `photo-degrade` auto-enables GPU acceleration for the dense
    /// stages and falls back cleanly to CPU on GPU-less machines.
    #[arg(long, default_value_t = false, conflicts_with = "cpu")]
    gpu: bool,
    /// Force the CPU backend for all stages (disables auto GPU acceleration).
    #[arg(long, default_value_t = false)]
    cpu: bool,
    /// Skip the fine paper-grain stage (sigma 2-5 additive Gaussian +
    /// speckles). Ablation (subtask 1): erased by downscale -> luma -> Otsu
    /// (ink-F1 > 0.997 vs clean on real pages), ~12% of degrade CPU.
    /// NOTE: skipping shifts the shared ChaCha stream for later stages, so
    /// `--skip-grain` outputs differ from full-pipeline outputs beyond the
    /// grain itself. Within a fixed flag set outputs stay bit-identical.
    #[arg(long, default_value_t = false)]
    skip_grain: bool,
    /// Skip chromatic aberration (subpixel red/blue fringing). Ablation:
    /// erased at pipeline magnitudes (ca <= ~0.75 px shifts nothing across
    /// the Otsu threshold), ~9% of degrade CPU. Stream-preserving: the
    /// `ca_px` draw is still consumed in the lighting block.
    #[arg(long, default_value_t = false)]
    skip_chromatic: bool,
    /// Skip sensor-noise stage (per-channel Gaussian, sigma 3-14). Ablation:
    /// erased by downscale -> luma -> Otsu (ink-F1 > 0.993 at sigma 10),
    /// ~13% of degrade CPU. NOTE: like --skip-grain, shifts the shared
    /// stream for the downscale/JPEG draws.
    #[arg(long, default_value_t = false)]
    skip_noise: bool,
    /// Coarsen ink degradation: keep dropout speckles + broken-stroke gaps
    /// (topology changes that survive Otsu) but skip the per-pixel
    /// fade/toner/browning/bleed-through field (uniform lifts Otsu adapts
    /// to; ablation ink-F1 > 0.995). Saves most of the ink stage (~7% CPU).
    /// Stream-preserving: the pixel loop consumes no RNG draws.
    #[arg(long, default_value_t = false)]
    coarsen_ink: bool,
    /// Coarsen paper aging: keep foxing spots + fibers (dark marks that
    /// survive Otsu as nuisance ink) but skip the full-image
    /// yellowing/mottling/edge/cockling/water tint loop (ablation ink-F1 >
    /// 0.993). Saves most of the aging stage (~6% CPU). Stream-preserving:
    /// the tint loop consumes no RNG draws.
    #[arg(long, default_value_t = false)]
    coarsen_aging: bool,
    /// Reduced pipeline preset: `--skip-grain --skip-chromatic --skip-noise
    /// --coarsen-ink --coarsen-aging`. ~40-45% cheaper per page with
    /// downstream Otsu parity (see examples/ablate.rs). Recorded as
    /// `preset="reduced"` in params; any extra flag makes it `"custom"`.
    #[arg(long, default_value_t = false)]
    reduced: bool,
    /// Precomputed fast path (subtask 5): seeded tiled Gaussian banks for
    /// paper grain + sensor noise (offsets sampled from the shared stream,
    /// bank content from the independent `(file_seed, stage_tag)` stream)
    /// and a precomputed lighting field. Same knobs and params, plus
    /// `fast_banks` recorded; legacy default off (bit-identical).
    /// Banked noise differs pixelwise from the direct ChaCha stream by
    /// design (independent stream) — deterministic per `(seed, stage, key)`,
    /// statistical parity proven in `banks.rs` tests.
    #[arg(long, default_value_t = false)]
    fast_banks: bool,
}

/// Which degradations to skip/coarsen. `default()` = full pipeline.
/// `fast_banks` is orthogonal (precomputed banks/fields, same knobs) and
/// does not affect the preset name.
#[derive(Debug, Clone, Copy, Default)]
struct StageToggles {
    skip_grain: bool,
    skip_chromatic: bool,
    skip_noise: bool,
    coarsen_ink: bool,
    coarsen_aging: bool,
    fast_banks: bool,
}

impl StageToggles {
    fn preset_name(&self) -> &'static str {
        let all = self.skip_grain
            && self.skip_chromatic
            && self.skip_noise
            && self.coarsen_ink
            && self.coarsen_aging;
        let any = self.skip_grain
            || self.skip_chromatic
            || self.skip_noise
            || self.coarsen_ink
            || self.coarsen_aging;
        if all {
            "reduced"
        } else if any {
            "custom"
        } else {
            "full"
        }
    }
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(&args) {
        eprintln!("photo-degrade: error: {e}");
        std::process::exit(1);
    }
}

/// Write `final_path` atomically: bytes go to a sibling `*.tmp` file first,
/// then it is renamed over the target, so a killed run never leaves a
/// half-file behind.
fn write_artifact(final_path: &Path, write: impl FnOnce(&Path) -> Result<(), String>) -> Result<(), String> {
    let mut tmp = final_path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp_path = PathBuf::from(tmp);
    write(&tmp_path)?;
    std::fs::rename(&tmp_path, final_path)
        .map_err(|e| format!("cannot rename {} -> {}: {e}", tmp_path.display(), final_path.display()))?;
    Ok(())
}

fn run(args: &Args) -> Result<(), String> {
    let strength = args.strength.clamp(0.0, 1.0);
    let fmt = match args.format.to_ascii_lowercase().as_str() {
        "png" => ImageFormat::Png,
        "jpeg" | "jpg" => ImageFormat::Jpeg,
        other => return Err(format!("--format must be png|jpeg, got `{other}`")),
    };
    let ext = match fmt {
        ImageFormat::Jpeg => "jpg",
        _ => "png",
    };
    if !(0.0..=1.0).contains(&args.scale_min) || args.scale_min <= 0.0 {
        return Err(format!("--scale-min must be in (0, 1], got {}", args.scale_min));
    }
    if let Some(q) = args.jpeg_quality {
        if q < 1 || q > 100 {
            return Err(format!("--jpeg-quality must be 1..=100, got {q}"));
        }
    }
    let inputs = collect_inputs(&args.input)?;
    if inputs.is_empty() {
        return Err(format!(
            "no PNG files found at {}",
            args.input.display()
        ));
    }
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|e| format!("cannot create {}: {e}", args.out_dir.display()))?;
    // One thread per chunk of files: per-file seeds keep output identical to
    // sequential runs, so threads only affect wall time, never pixels.
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1)
        .min(inputs.len().max(1));
    let chunk_size = (inputs.len() + n_threads - 1) / n_threads;
    let stop = std::sync::atomic::AtomicBool::new(false);
    let first_err = std::sync::Mutex::new(None::<String>);
    let print_lock = std::sync::Mutex::new(());
    // Shared by reference: scope guarantees threads join before return.
    let (stop_r, err_r, pl_r) = (&stop, &first_err, &print_lock);
    // One shared GPU context for all worker threads (`Gpu` is Send+Sync).
    // Default (neither `--gpu` nor `--cpu`): auto-enable GPU and fall back
    // cleanly to CPU on GPU-less machines. `--gpu` requires a GPU (fail fast);
    // `--cpu` forces CPU.
    let gpu = if args.cpu {
        None
    } else {
        match gpu_backend::Gpu::new() {
            Ok(g) => {
                println!("GPU backend enabled");
                // Warm the GPU device/pipelines now (untimed, no RNG) so the
                // first file's ticks measure steady state, not one-time
                // pipeline compile. One tiny dispatch per entry point: grain
                // and the full resident chain (all stages including affine).
                let tiny = vec![128.0f32; 8 * 8 * 3];
                let _ = g.grain(&tiny, 8, 8, 3.0, 1, 0.001, 0.5, [4.0, 2.0, -1.0]);
                let (rows, meta, _, _, nw, nh) = gpu_backend::rotscale_frame(2.0, 8, 8, 0.9, 0.9);
                let _ = g.chain(
                    &tiny,
                    8,
                    8,
                    &degrade_gpu::chain::ChainConfig {
                        grain_sigma: 3.0,
                        grain_seed: 1,
                        grain_speckle_prob: 0.001,
                        grain_strength: 0.5,
                        grain_warm: [4.0, 2.0, -1.0],
                        bend: BendKnobs {
                            mode: 2,
                            edge_id: 0,
                            curl_px: 1.0,
                            curl_shade: 0.1,
                            spine_cx: 4.0,
                            spine_w: 1.0,
                            depth: 0.05,
                            rip_fx: 1.5,
                            rip_fy: 1.2,
                            rip_ph: 0.3,
                        },
                        persp_inv: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        rot_deg: 2.0,
                        blur_sigma: 1.5,
                        light: LightingKnobs {
                            gx: 4.0,
                            gy: 4.0,
                            glare_radius: 20.0,
                            glare_alpha: 0.1,
                            sh_dx: 1.0,
                            sh_dy: 0.0,
                            shadow_strength: 0.1,
                            vignette_strength: 0.1,
                        },
                        ca_px: 0.5,
                        noise_sigma: 5.0,
                        noise_seed: 7,
                        run_grain: true,
                        run_bend: true,
                        run_persp: true,
                        run_rot: true,
                        run_blur: true,
                        run_light: true,
                        run_ca: true,
                        run_noise: true,
                        tilt_active: false,
                        focus_y: 0.0,
                        focus_h: 0.0,
                        run_affine: true,
                        affine_rows: rows,
                        affine_meta: meta,
                        affine_nw: nw.max(1),
                        affine_nh: nh.max(1),
                    },
                );
                Some(g)
            }
            Err(e) if args.gpu => return Err(format!("--gpu: {e}")),
            Err(_) => None,
        }
    };
    let toggles = StageToggles {
        skip_grain: args.skip_grain || args.reduced,
        skip_chromatic: args.skip_chromatic || args.reduced,
        skip_noise: args.skip_noise || args.reduced,
        coarsen_ink: args.coarsen_ink || args.reduced,
        coarsen_aging: args.coarsen_aging || args.reduced,
        fast_banks: args.fast_banks,
    };
    if toggles.preset_name() != "full" {
        println!("pipeline preset: {}", toggles.preset_name());
    }
    let (args_r, strength_r) = (&*args, strength);
    let gpu_r = gpu.as_ref();
    std::thread::scope(|scope| {
        for (ci, chunk) in inputs.chunks(chunk_size).enumerate() {
            let base = ci * chunk_size;
            scope.spawn(move || {
                for (j, path) in chunk.iter().enumerate() {
                    if stop_r.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    // Derive a per-file seed so batch runs stay deterministic.
                    let i = base + j;
                    let file_seed = args_r.seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    match degrade_file(path, file_seed, strength_r, args_r.jpeg_quality, args_r.max_width, args_r.train_width, args_r.scale_min, gpu_r, &toggles) {
                        Ok((img, mut record)) => {
                            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                                Some(s) => s.to_string(),
                                None => {
                                    stop_r.store(true, std::sync::atomic::Ordering::Relaxed);
                                    *err_r.lock().unwrap() =
                                        Some(format!("bad file name: {}", path.display()));
                                    break;
                                }
                            };
                            let img_name = format!("{stem}{}.{ext}", args_r.suffix);
                            let json_name = format!("{stem}{}_params.json", args_r.suffix);
                            let img_path = args_r.out_dir.join(&img_name);
                            let json_path = args_r.out_dir.join(&json_name);
                            // tmp+rename so a killed run never leaves a half-file.
                            // JPEG: encode to memory first (nvJPEG GPU, else
                            // jpeg-encoder SIMD; q75 = old `image` default),
                            // then write the bytes; params record the tag.
                            // PNG keeps `image`.
                            let write_res = if fmt == ImageFormat::Jpeg {
                                match encode_jpeg_bytes(
                                    img.as_raw(),
                                    img.width(),
                                    img.height(),
                                    75,
                                ) {
                                    Ok((buf, tag)) => write_artifact(&img_path, |p| {
                                        std::fs::write(p, &buf).map_err(|e| {
                                            format!("cannot write {}: {e}", p.display())
                                        })
                                    })
                                    .map(|()| tag),
                                    Err(e) => Err(e),
                                }
                            } else {
                                write_artifact(&img_path, |p| {
                                    img.save_with_format(p, fmt).map_err(|e| {
                                        format!("cannot write {}: {e}", p.display())
                                    })
                                })
                                .map(|()| "image")
                            };
                            match write_res {
                                Ok(tag) => record.codec = tag.to_string(),
                                Err(e) => {
                                    stop_r.store(true, std::sync::atomic::Ordering::Relaxed);
                                    *err_r.lock().unwrap() = Some(e);
                                    break;
                                }
                            }
                            record.input = path
                                .file_name()
                                .map(|s| s.to_string_lossy().into_owned())
                                .unwrap_or_default();
                            record.output = img_name;
                            record.file_seed = file_seed;
                            let json = match serde_json::to_string_pretty(&record)
                                .map_err(|e| format!("cannot serialize params: {e}"))
                            {
                                Ok(j) => j,
                                Err(e) => {
                                    stop_r.store(true, std::sync::atomic::Ordering::Relaxed);
                                    *err_r.lock().unwrap() = Some(e);
                                    break;
                                }
                            };
                            if let Err(e) = write_artifact(&json_path, |p| {
                                std::fs::write(p, json.clone() + "\n")
                                    .map_err(|e| format!("cannot write {}: {e}", p.display()))
                            }) {
                                stop_r.store(true, std::sync::atomic::Ordering::Relaxed);
                                *err_r.lock().unwrap() = Some(e);
                                break;
                            }
                            let _guard = pl_r.lock().unwrap();
                            println!("{} -> {}", path.display(), img_path.display());
                            println!("  params -> {}", json_path.display());
                        }
                        Err(e) => {
                            stop_r.store(true, std::sync::atomic::Ordering::Relaxed);
                            *err_r.lock().unwrap() =
                                Some(format!("{}: {e}", path.display()));
                            break;
                        }
                    }
                }
            });
        }
    });
    if let Some(e) = err_r.lock().unwrap().take() {
        return Err(e);
    }
    Ok(())
}

/// Every sampled degradation parameter, written next to its image.
#[derive(Serialize)]
struct Params {
    input: String,
    output: String,
    file_seed: u64,
    strength: f32,
    /// Which backend executed the dense stages: `cpu` or `gpu`.
    backend: String,
    /// Pipeline preset: `full` (default), `reduced` (`--reduced`), or
    /// `custom` (individual skip/coarsen flags).
    preset: String,
    /// Subtask-5 precomputed fast path (`--fast-banks`): tiled noise banks
    /// + precomputed lighting field. Same knobs, `backend` unchanged.
    fast_banks: bool,
    width: u32,
    height: u32,
    // training-resolution rescale (0/1.0 = legacy full-res path)
    train_width: u32,   // requested --train-width (0 = off)
    src_width: u32,     // post-maxwidth input width the factor derives from
    src_height: u32,    // post-maxwidth input height
    rescale_k: f32,     // train_width/src_width (1.0 when off or narrower)
    // paper
    paper_grain_sigma: f32,
    paper_speckle_prob: f32,
    // curl
    curl_side: String,
    curl_px: f32,
    curl_shade: f32,
    bend_spine_cx: f32,  // book-spine gutter center (px, spine mode)
    bend_spine_w: f32,   // gutter half-width (px, spine mode)
    bend_ripple_fx: f32, // hand-held wave x-frequency (ripple mode)
    bend_ripple_fy: f32, // hand-held wave y-frequency (ripple mode)
    bend_ripple_ph: f32, // hand-held wave phase (ripple mode)
    // perspective
    perspective_max_frac: f32,
    corner_offsets: [[f32; 2]; 4], // TL, TR, BR, BL in pixels, post-fit
    warp_fit_scale: f32, // shrink-to-fit factor applied to the offsets
    warp_out_px: f32, // worst corner distance outside the frame (<= bleed)
    // rotation (canvas expands so the page stays fully visible)
    rotation_deg: f32,
    /// `--gpu` deferred rotate+downscale fusion ran at 6b (more-spec-affine
    /// option b): one affine dispatch instead of the eager rotation
    /// resample plus a separate downscale. Always false on the CPU path.
    rot_scale_fused: bool,
    // photo framing: desk inset vs tight crop (cuts clamped to quiet margins)
    frame_mode: String,            // "inset" | "tight"
    frame_pad_px: [u32; 4],       // desk pad added [left, top, right, bottom] (inset only)
    frame_cut_px: [u32; 4],       // frame crop [left, top, right, bottom] (tight only)
    frame_notation_bbox: [u32; 4], // dark-pixel bbox at framing time [min_x, min_y, max_x, max_y]
    frame_quiet_px: [u32; 4],     // quiet margins at framing time [left, top, right, bottom]
    frame_safe_px: u32,           // safety buffer kept between any cut edge and notation
    // lens geometry
    dist_k1: f32,
    dist_k2: f32,
    dist_fit_scale: f32, // barrel-outward rescale factor
    // focus
    focus_mode: String,
    focus_y: f32,
    focus_h: f32,
    tilt_sigma: f32,
    // blur
    blur_sigma: f32,
    // lighting
    glare_center: [f32; 2],
    glare_radius: f32,
    glare_alpha: f32,
    shadow_angle_deg: f32,
    shadow_strength: f32,
    vignette_strength: f32,
    // lens fringing
    ca_px: f32,
    // linked degrade modules (stages 1b, 1c, 3b2, 5c)
    ink: degrade_ink::InkParams,
    paper: degrade_paper::PaperParams,
    wear: degrade_wear::WearParams,
    capture: degrade_capture::CaptureParams,
    // output resolution (non-uniform: x/y may differ by up to ~5%)
    scale_x: f32,
    scale_y: f32,
    // physical artifacts
    tear_count: u32,
    stain_count: u32,
    pencil_count: u32,
    fold_corner: String,
    fold_creases: Vec<[f32; 3]>, // (angle_deg, pos_frac, strength)
    highlight: String,
    // sensor + compression
    noise_sigma: f32,
    jpeg_quality: u8,
    /// pixel codec backend ("zune": jpeg-encoder + zune-jpeg/zune-png)
    codec: String,
}

/// Full pipeline for one file. Returns the degraded image + its params.
///
/// `toggles` selects the reduced pipeline (subtask 1): skipped stages record
/// neutral params. Grain/noise skips shift the shared ChaCha stream for later
/// stages (their loops consume draws); ink/aging coarsening and the chromatic
/// skip are stream-preserving (the skipped pixel loops consume no draws).
fn degrade_file(
    path: &Path,
    file_seed: u64,
    strength: f32,
    jpeg_quality: Option<u8>,
    max_width: u32,
    train_width: u32,
    scale_min: f32,
    gpu: Option<&gpu_backend::Gpu>,
    toggles: &StageToggles,
) -> Result<(RgbImage, Params), String> {
    // Fast-codec input load (zune PNG / zune JPEG → RGB8). Same contract as the
    // old `ImageReader` path: any depth becomes RGB8, huge inputs rejected.
    let mut img = decode_input_image(path)?;
    // Pre-degrade sizing: cap to `max_width`, then optionally rescale to
    // `--train-width`. One Triangle resample lands directly on the final
    // size so stacked blurs never soften notation twice. With train off,
    // this is exactly the legacy maxwidth block.
    let (mut tw, mut th) = (img.width(), img.height());
    if max_width > 0 && tw > max_width {
        th = (img.height() as u64 * max_width as u64 / img.width() as u64).max(1) as u32;
        tw = max_width;
    }
    // Post-maxwidth native dims: the scale story is told against these.
    let (src_width, src_height) = (tw, th);
    let rescale_k = if train_width > 0 && tw > train_width {
        th = (th as u64 * train_width as u64 / tw as u64).max(1) as u32;
        tw = train_width;
        train_width as f32 / src_width as f32
    } else {
        1.0f32
    };
    if tw != img.width() || th != img.height() {
        img = resize_rgb_fir(img, tw, th)?;
    }
    let (mut w, mut h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return Err(format!("empty image: {}", path.display()));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(file_seed);
    let s = strength;
    let backend = if gpu.is_some() { "gpu" } else { "cpu" };
    // Per-stage timing probe, enabled with PHOTO_DEGRADE_TIMING=1 (stderr).
    // Zero overhead when off; no signature changes needed.
    let timing = std::env::var("PHOTO_DEGRADE_TIMING").is_ok();
    let mut last = std::time::Instant::now();
    macro_rules! tick {
        ($name:expr) => {
            if timing {
                let now = std::time::Instant::now();
                eprintln!("stage {:>16}: {:7.1} ms", $name, (now - last).as_secs_f64() * 1000.0);
                #[allow(unused_assignments)]
                {
                    last = now;
                }
            }
        };
    }

    // Working buffer in 0..=255 floats, RGB interleaved.
    let mut buf: Vec<f32> = img.pixels().flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();

        tick!("load+maxwidth");

    // --- 1. paper texture: grain + warm tint + sparse speckles ---
    // Skipped under --skip-grain/--reduced (ablation: erased by Otsu).
    // Grain sigma scales with rescale_k: white noise downsampled by k per
    // axis loses amplitude ~k, so train-res grain matches downscaled-full.
    // (Ã—1.0 = legacy exactly.) Speckle probability is per-pixel, already
    // area-proportional; warm tint is a contrast shift, unscaled.
    let (paper_grain_sigma, paper_speckle_prob, warm) = if toggles.skip_grain {
        (0.0f32, 0.0f32, [0.0f32; 3])
    } else {
        (
            s * rng.gen_range(2.0..5.0) * rescale_k,
            s * rng.gen_range(0.0005..0.002),
            [rng.gen_range(2.0..7.0) * s, rng.gen_range(1.0..4.0) * s, -rng.gen_range(0.0..3.0) * s],
        )
    };
    if !toggles.skip_grain {
    if let Some(g) = gpu {
        // Same knobs as above; the per-pixel stream is the GPU hash variant
        // (see grain_seed32), so pixels differ from the CPU ChaCha run at
        // the same seed â€” deterministic per backend, like sensor noise.
        buf = g.grain(
            &buf, w, h,
            paper_grain_sigma,
            gpu_backend::grain_seed32(file_seed),
            paper_speckle_prob, s, warm,
        );
    } else if toggles.fast_banks {
        // Subtask-5 banked path: knobs sampled above are unchanged; the
        // per-pixel ChaCha/Box-Muller stream is replaced by the seeded
        // tiled bank (content from the `(file_seed, TAG_GRAIN)` stream,
        // offset from the shared stream). Pixels differ from the direct
        // run by design; determinism per (seed, stage, key) holds.
        let gauss_bank = banks::generate_gauss_bank(file_seed, banks::TAG_GRAIN, banks::BANK_TILE, paper_grain_sigma);
        let uni_bank = banks::generate_uniform_bank(file_seed, banks::TAG_GRAIN, banks::BANK_TILE);
        let (ox, oy) = banks::bank_offset(&mut rng, banks::BANK_TILE);
        banks::apply_grain_banked(&mut buf, w, h, &gauss_bank, &uni_bank, banks::BANK_TILE, ox, oy, paper_grain_sigma, paper_speckle_prob, s, warm);
    } else {
    for px in buf.chunks_exact_mut(3) {
        for c in 0..3 {
            px[c] += gauss(&mut rng) * paper_grain_sigma + warm[c];
        }
        if rng.gen::<f32>() < paper_speckle_prob {
            let d = rng.gen_range(10.0..35.0) * s;
            px[0] -= d;
            px[1] -= d;
            px[2] -= d;
        }
    }
    }
    clamp_buf(&mut buf);
    } // --skip-grain

        tick!("paper-grain");

    // --- 1b. paper aging (foxing, mottling, fibers, water damage) ---
    // Absolute-pixel radii/periods inside scale with rescale_k (1.0 = legacy).
    let paper_p = degrade_paper::apply_with(&mut buf, w, h, &mut rng, s, toggles.coarsen_aging, rescale_k);

        tick!("paper-aging");

    // --- 1c. ink & print degradation (fading, dropout, bleed-through) ---
    // Runs pre-bend so worn print warps with the page, like real ink.
    // Absolute-pixel radii inside scale with rescale_k (1.0 = legacy).
    let ink_p = degrade_ink::apply_with(&mut buf, w, h, &mut rng, s, toggles.coarsen_ink, rescale_k);

        tick!("ink");

    // --- 2. page bend: edge curl (any side), book spine, or ripple ---
    // Modes: an edge lifts toward the camera (warp + shading), the spine
    // sinks a gutter down the middle (book scan), ripple adds a gentle
    // hand-held wave. One mode per image, sampled from the file seed.
    let bend_roll: f32 = rng.gen();
    let bend_mode: &str = if bend_roll < 0.45 { "edge" } else if bend_roll < 0.70 { "spine" } else { "ripple" };
    let curl_side: String = if bend_mode == "edge" {
        ["left", "right", "top", "bottom"][rng.gen_range(0..4)].into()
    } else {
        bend_mode.into()
    };
    let curl_px = s * w as f32 * rng.gen_range(0.02..0.08);
    let curl_shade = s * rng.gen_range(0.15..0.40);
    let (mut bend_spine_cx, mut bend_spine_w) = (0.0f32, 0.0f32);
    let (mut bend_ripple_fx, mut bend_ripple_fy, mut bend_ripple_ph) = (0.0f32, 0.0f32, 0.0f32);
    // Bend pixel work on the GPU path is deferred to the perspective stage
    // so the two warps fuse into one round-trip (draw order unchanged:
    // resampling consumes no RNG). CPU runs inline exactly as before.
    let bend_active = curl_px > 0.25;
    // F-A fused framing bbox (online-framing-lens §4): the last CPU warp
    // loop before framing folds the exact written-pixel channel sums here,
    // and the framing stage consumes the accumulation instead of a
    // standalone full-frame scan. The GPU path never touches it (chain1 is
    // opaque) and falls back to the scan. Threshold matches the framing
    // site's `rescale_k` rule (200 train-res, 128 legacy). No RNG.
    let mut warp_acc = BBoxAcc::default();
    let warp_thresh3 = if rescale_k < 1.0 { 600.0 } else { 384.0 };
    // Warp mode ids shared by the fused and bend-alone GPU calls below.
    let (bend_mode_id, bend_edge_id) = if bend_mode == "spine" {
        (0u32, 0u32)
    } else if bend_mode == "ripple" {
        (1u32, 0u32)
    } else {
        let eid = match curl_side.as_str() {
            "right" => 1u32,
            "top" => 2u32,
            "bottom" => 3u32,
            _ => 0u32,
        };
        (2u32, eid)
    };
    if bend_active {
        let wf = w as f32;
        let hf = h as f32;
        bend_ripple_fx = rng.gen_range(1.0..2.5);
        bend_ripple_fy = rng.gen_range(1.0..2.0);
        bend_ripple_ph = rng.gen_range(0.0..std::f32::consts::TAU);
        bend_spine_cx = wf * rng.gen_range(0.35..0.65);
        bend_spine_w = wf * rng.gen_range(0.03..0.08);
        let ripple_fx = bend_ripple_fx;
        let ripple_fy = bend_ripple_fy;
        let ripple_ph = bend_ripple_ph;
        let spine_cx = bend_spine_cx;
        let spine_w = bend_spine_w;
        // GPU pixel work deferred to the perspective stage (fused); CPU
        // runs inline exactly as before.
        if gpu.is_none() {
        let src = buf.clone();
        for y in 0..h {
            let wave = 0.6 + 0.4 * ((std::f32::consts::PI * y as f32 / hf).sin());
            for x in 0..w {
                let (sx, sy, shade) = if bend_mode == "spine" {
                    // Gutter: smooth squeeze + shade, continuous across the fold.
                    let depth = ((curl_px / wf) * 2.0).min(0.15);
                    let (sx, shade) = spine_map(x as f32, spine_cx, spine_w, depth, curl_shade);
                    (sx, y as f32, shade)
                } else if bend_mode == "ripple" {
                    // Hand-held wave: sinusoidal offsets, mild brightness roll.
                    let ox = curl_px * 0.5 * (std::f32::consts::TAU * ripple_fx * y as f32 / hf + ripple_ph).sin();
                    let oy = curl_px * 0.3 * (std::f32::consts::TAU * ripple_fy * x as f32 / wf + ripple_ph).cos();
                    let shade = 1.0 - curl_shade * 0.4 * (std::f32::consts::TAU * ripple_fx * y as f32 / hf + ripple_ph).sin().abs();
                    (x as f32 + ox, y as f32 + oy, shade)
                } else {
                    // Edge curl toward any of the four sides.
                    let (edge, along_wave) = match curl_side.as_str() {
                        "right" => (x as f32 / wf, wave),
                        "top" => (1.0 - y as f32 / hf, 0.6 + 0.4 * ((std::f32::consts::PI * x as f32 / wf).sin())),
                        "bottom" => (y as f32 / hf, 0.6 + 0.4 * ((std::f32::consts::PI * x as f32 / wf).sin())),
                        _ => (1.0 - x as f32 / wf, wave), // left
                    };
                    let shift = curl_px * edge * edge * along_wave;
                    let (sx, sy) = match curl_side.as_str() {
                        "right" => (x as f32 - shift, y as f32),
                        "top" => (x as f32, y as f32 + shift),
                        "bottom" => (x as f32, y as f32 - shift),
                        _ => (x as f32 + shift, y as f32),
                    };
                    (sx, sy, 1.0 - curl_shade * edge * edge)
                };
                let p = sample_bilinear(&src, w, h, sx, sy);
                let i = ((y * w + x) * 3) as usize;
                // Exact written values (what a later scan would sum).
                let (q0, q1, q2) = (p[0] * shade, p[1] * shade, p[2] * shade);
                buf[i] = q0;
                buf[i + 1] = q1;
                buf[i + 2] = q2;
                warp_acc.fold(q0 + q1 + q2, x, y, warp_thresh3);
            }
        }
        clamp_buf(&mut buf);
        }
    }

        tick!("bend");

    // --- 3. perspective warp (homography from jittered corners) ---
    let perspective_max_frac = s * 0.07;
    let wf = w as f32;
    let hf = h as f32;
    let corners = [(0.0, 0.0), (wf, 0.0), (wf, hf), (0.0, hf)];
    let mut dst = [(0.0f64, 0.0f64); 4];
    let mut corner_offsets = [[0.0f32; 2]; 4];
    for (i, &(cx, cy)) in corners.iter().enumerate() {
        let dx = rng.gen_range(-1.0..1.0) as f64 * perspective_max_frac as f64 * wf as f64;
        let dy = rng.gen_range(-1.0..1.0) as f64 * perspective_max_frac as f64 * hf as f64;
        dst[i] = (cx as f64 + dx, cy as f64 + dy);
        corner_offsets[i] = [dx as f32, dy as f32];
    }
    // Shrink-to-fit per corner: inward motion is kept whole (a page
    // photographed smaller than the frame loses nothing); only outward
    // motion is trimmed so no corner leaves the frame by more than a
    // sliver (`bleed`, sub-staff-line scale in the quiet margins, filled
    // with paper tone by the sampler). Notation can never be pushed out
    // of the picture; the applied offsets, the worst out-of-frame
    // distance, and the smallest kept offset fraction are recorded.
    let bleed = 0.004 * (wf as f64).min(hf as f64);
    let mut warp_fit_scale = 1.0f32;
    let mut warp_out_px = 0.0f32;
    for i in 0..4 {
        let (cx, cy) = corners[i];
        let (mut dx, mut dy) = (dst[i].0 - cx as f64, dst[i].1 - cy as f64);
        for (c, dim, d) in [(cx as f64, wf as f64, &mut dx), (cy as f64, hf as f64, &mut dy)] {
            let lo = -(c + bleed);
            let hi = dim - c + bleed;
            let clamped = (*d).clamp(lo, hi);
            if d.abs() > 1e-9 {
                warp_fit_scale = warp_fit_scale.min((clamped / *d) as f32);
            }
            *d = clamped;
        }
        dst[i] = (cx as f64 + dx, cy as f64 + dy);
        corner_offsets[i] = [dx as f32, dy as f32];
        warp_out_px = warp_out_px.max(
            (-dst[i].0).max(dst[i].0 - wf as f64).max(-dst[i].1).max(dst[i].1 - hf as f64).max(0.0) as f32,
        );
    }
    // Inverse homography for the warp; `None` when the gate fails or the
    // fit is degenerate. The GPU path consumes it in chain1 below.
    let mut persp_hinv: Option<[f64; 9]> = None;
    if perspective_max_frac * wf.max(hf) > 0.5 {
        let src_c = corners.map(|(x, y)| (x as f64, y as f64));
        // H maps source -> warped; sample with its inverse.
        if let Some(hinv) = solve_homography(src_c, dst).and_then(invert_3x3) {
            persp_hinv = Some(hinv);
            if gpu.is_none() {
            // This loop rewrites every pixel, so it supersedes any bend
            // accumulation: reset, then fold the exact written sums.
            warp_acc.reset();
            let src = buf.clone();
            for y in 0..h {
                for x in 0..w {
                    let (sx, sy) = apply_h(&hinv, x as f64, y as f64);
                    let p = sample_bilinear(&src, w, h, sx as f32, sy as f32);
                    let i = ((y * w + x) * 3) as usize;
                    buf[i] = p[0];
                    buf[i + 1] = p[1];
                    buf[i + 2] = p[2];
                    warp_acc.fold(p[0] + p[1] + p[2], x, y, warp_thresh3);
                }
            }
            clamp_buf(&mut buf);
            }
        }
    }
    // GPU path (chain1): bend + perspective in one resident submit on the
    // shared device. Same kernels and knobs as the separate calls this
    // replaces (`g.bend` + `g.resample`, previously fused as `bend_persp`);
    // runs here (pre-framing, same dims) exactly where the fused warp ran,
    // with or without perspective (the old bend-alone branch is subsumed:
    // `run_persp` is false exactly when the old code skipped the warp).
    // Grain keeps its inline call above, so its input is unchanged.
    if let Some(g) = gpu {
        let depth = ((curl_px / wf) * 2.0).min(0.15);
        let pre_cfg = ChainConfig {
            bend: BendKnobs {
                mode: bend_mode_id,
                edge_id: bend_edge_id,
                curl_px,
                curl_shade,
                spine_cx: bend_spine_cx,
                spine_w: bend_spine_w,
                depth,
                rip_fx: bend_ripple_fx,
                rip_fy: bend_ripple_fy,
                rip_ph: bend_ripple_ph,
            },
            persp_inv: persp_hinv
                .map(|hinv| {
                    [
                        hinv[0] as f32, hinv[1] as f32, hinv[2] as f32,
                        hinv[3] as f32, hinv[4] as f32, hinv[5] as f32,
                        hinv[6] as f32, hinv[7] as f32, hinv[8] as f32,
                    ]
                })
                .unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            run_bend: bend_active,
            run_persp: persp_hinv.is_some(),
            ..ChainConfig::default()
        };
        // Skip the round-trip entirely when no gate fired (tiny images).
        if pre_cfg.run_bend || pre_cfg.run_persp {
            let (chained, nw, nh) = g.chain(&buf, w, h, &pre_cfg);
            buf = chained;
            w = nw;
            h = nh;
            clamp_buf(&mut buf);
        }
    }

        tick!("perspective");

    // --- 3a-frame. photo framing: desk inset or tight crop ---
    // Real phone photos rarely map page corners exactly to image corners:
    // the sheet sits on a desk (dark surround visible) or the frame cuts
    // slightly into the page. Both variants are notation-safe by
    // construction: inset only ADDS desk pixels around the whole page, and
    // tight cuts are clamped per edge to the measured quiet margin minus a
    // safety buffer, so dark (notation) pixels can never leave the frame.
    // Runs pre-rotation so the notation bbox sees paper fill only (the
    // rotation stage's dark desk triangles would read as ink).
    let frame_roll: f32 = rng.gen();
    let frame_mode: &str = if frame_roll < 0.70 { "inset" } else { "tight" };
    let frame_min_dim = w.min(h) as f32;
    // Safety buffer re-derived for training resolution: the 0.002 relative
    // term gives 5 px at full res (0.5 staff spaces at 10 px spacing) but
    // only ~2 px at 1024 wide, so the 3 px floor binds there instead â€” 3 px
    // is ~0.7 staff spaces at the ~4.1 px train-res spacing, strictly more
    // conservative relative to staff geometry than full res. The
    // cut+safe<=quiet law itself is algebraic and holds at any size.
    let frame_safe_px = ((0.002 * frame_min_dim).round() as u32).max(3);
    // Sensitive bbox on the train path (see notation_bbox_thresh): train
    // pre-downscaling pre-grays thin notation, which would blind the
    // legacy-128 detector and return optimistic quiet margins. Legacy 128
    // is untouched when train is off.
    // F-A: consume the warp-loop accumulation when a CPU warp loop ran;
    // otherwise (GPU path, or both warp gates off) fall back to the
    // standalone scan exactly as before.
    let frame_bbox = warp_acc.bbox().or_else(|| {
        if rescale_k < 1.0 {
            notation_bbox_thresh(&buf, w, h, 200.0)
        } else {
            notation_bbox(&buf, w, h)
        }
    });
    let frame_quiet_px = quiet_margins(w, h, frame_bbox);
    let (mut frame_pad_px, mut frame_cut_px) = ([0u32; 4], [0u32; 4]);
    if frame_mode == "inset" {
        for p in frame_pad_px.iter_mut() {
            *p = (rng.gen_range(0.008..0.045) * frame_min_dim).round() as u32;
        }
    } else {
        let mut desired = [0u32; 4];
        for d in desired.iter_mut() {
            *d = (rng.gen_range(0.004..0.028) * frame_min_dim).round() as u32;
        }
        frame_cut_px = clamp_cut_to_quiet(desired, frame_quiet_px, frame_safe_px);
    }
    // No pixel work here: framing materializes in the rotation block below
    // — eagerly via `apply_framing_eager`, or lazily inside the CPU
    // rotation loop (F-B), or already-eager on the deferred-GPU path.

        tick!("framing");

    // --- 3a. page rotation: up to +/-10 deg, canvas expands to fit ---
    // The whole page stays visible (no negative crop); newly exposed
    // canvas is dark desk tone, and the applied angle is recorded.
    let rotation_deg = rng.gen_range(-10.0..10.0) * s;
    let rot_active = rotation_deg.abs() > 0.05 && w > 4 && h > 4;
    // Deferred-rotation stash for the `--gpu` 6b fused affine dispatch
    // (more-spec-affine option b): only the angle travels; stages 3b-6 run
    // on the unrotated buffer. The CPU path keeps legacy sequential
    // behavior below. No draw-order change either way.
    let mut rot_stash: Option<f32> = None;
    let mut rot_scale_fused = false;
    if rot_active {
        if gpu.is_some() {
            // Framing materializes eagerly now (stages 3b-6 run framed,
            // exactly as before); rotation pixel work is deferred to 6b.
            apply_framing_eager(&mut buf, &mut w, &mut h, frame_mode, frame_pad_px, frame_cut_px);
            rot_stash = Some(rotation_deg);
        } else {
        // F-B: fold framing into the rotation resample (CPU only). The
        // loop iterates the canvas of the *framed* dims; tight pages sample
        // the true neighboring page pixels (a <=1 px rim band inside cut
        // edges blends against page instead of paper BG — the accepted
        // F-B difference), inset pages are bit-exact (pad fill == DESK ==
        // rotation fill, same taps in the same order).
        let (fw, fh) = framed_dims(w, h, frame_mode, frame_pad_px, frame_cut_px);
        let th = rotation_deg.to_radians();
        let (sn, cs) = (th.sin().abs(), th.cos().abs());
        let (w2, h2) = (
            ((fw as f32) * cs + (fh as f32) * sn).ceil().max(1.0) as u32,
            ((fw as f32) * sn + (fh as f32) * cs).ceil().max(1.0) as u32,
        );
        const DESK: [f32; 3] = [25.0, 25.0, 30.0];
        const PAPER_BG: [f32; 3] = [246.0, 243.0, 236.0]; // sample_bilinear OOB
        let src = std::mem::take(&mut buf);
        let (fwf, fhf) = (fw as f32, fh as f32);
        let (cx0, cy0) = (fwf / 2.0, fhf / 2.0);
        let (cx1, cy1) = (w2 as f32 / 2.0, h2 as f32 / 2.0);
        let (snt, cst) = (th.sin(), th.cos());
        let mut rot = vec![0.0f32; (w2 * h2 * 3) as usize];
        if frame_mode == "inset" {
            // Padded-coord expression tree exactly as legacy (same frac and
            // weights); only tap VALUES are remapped (desk surround, page
            // content, paper-BG fringe) — bit-exact vs rotating the
            // materialized padded buffer.
            // Interior fast path: tap origin/weights computed exactly as
            // `sample_remapped` does; when all four framed taps land
            // in-page (the pad offset is integral, so the framed-bounds
            // check above plus src-in-bounds here covers every tap), fetch
            // directly and blend — same values, weights, and order.
            let (pl, pt) = (frame_pad_px[0] as i64, frame_pad_px[1] as i64);
            for y in 0..h2 {
                for x in 0..w2 {
                    let dx = x as f32 - cx1;
                    let dy = y as f32 - cy1;
                    let fx = cx0 + dx * cst + dy * snt;
                    let fy = cy0 - dx * snt + dy * cst;
                    let p = if fx >= 0.0 && fx < fwf && fy >= 0.0 && fy < fhf {
                        let x0f = fx.floor() as i64;
                        let y0f = fy.floor() as i64;
                        let (tx, ty) = (fx - x0f as f32, fy - y0f as f32);
                        let jx0 = x0f - pl;
                        let jy0 = y0f - pt;
                        if jx0 >= 0 && jy0 >= 0 && jx0 + 1 < w as i64 && jy0 + 1 < h as i64 {
                            let j00 = ((jy0 as u32 * w + jx0 as u32) * 3) as usize;
                            let t00 = [src[j00], src[j00 + 1], src[j00 + 2]];
                            let t10 = [src[j00 + 3], src[j00 + 4], src[j00 + 5]];
                            let j01 = j00 + w as usize * 3;
                            let t01 = [src[j01], src[j01 + 1], src[j01 + 2]];
                            let t11 = [src[j01 + 3], src[j01 + 4], src[j01 + 5]];
                            blend4(t00, t10, t01, t11, tx, ty)
                        } else {
                            sample_remapped(fx, fy, |ix, iy| {
                                if ix < 0 || iy < 0 || ix >= fw as i64 || iy >= fh as i64 {
                                    PAPER_BG
                                } else {
                                    let jx = ix - pl;
                                    let jy = iy - pt;
                                    if jx < 0 || jy < 0 || jx >= w as i64 || jy >= h as i64 {
                                        DESK
                                    } else {
                                        let j = ((jy as u32 * w + jx as u32) * 3) as usize;
                                        [src[j], src[j + 1], src[j + 2]]
                                    }
                                }
                            })
                        }
                    } else {
                        DESK
                    };
                    let i = ((y * w2 + x) * 3) as usize;
                    rot[i] = p[0];
                    rot[i + 1] = p[1];
                    rot[i + 2] = p[2];
                }
            }
        } else {
            // Cropped-coord expression tree exactly as legacy; tap values
            // read the true neighboring page pixels (a <=1 px rim band
            // inside cut edges blends against page instead of paper BG —
            // the accepted F-B difference; interior is bit-exact).
            // Interior fast path: tap origin/weights exactly as
            // `sample_remapped`; framed-in-bounds taps always land in-page
            // on the tight path (cut offsets point inward), so the
            // framed-bounds check below covers every tap.
            let (cl, ct) = (frame_cut_px[0] as i64, frame_cut_px[1] as i64);
            for y in 0..h2 {
                for x in 0..w2 {
                    let dx = x as f32 - cx1;
                    let dy = y as f32 - cy1;
                    let fx = cx0 + dx * cst + dy * snt;
                    let fy = cy0 - dx * snt + dy * cst;
                    let p = if fx >= 0.0 && fx < fwf && fy >= 0.0 && fy < fhf {
                        let x0f = fx.floor() as i64;
                        let y0f = fy.floor() as i64;
                        let (tx, ty) = (fx - x0f as f32, fy - y0f as f32);
                        if x0f >= 0 && y0f >= 0 && x0f + 1 < fw as i64 && y0f + 1 < fh as i64 {
                            let jx0 = x0f + cl;
                            let jy0 = y0f + ct;
                            let j00 = ((jy0 as u32 * w + jx0 as u32) * 3) as usize;
                            let t00 = [src[j00], src[j00 + 1], src[j00 + 2]];
                            let t10 = [src[j00 + 3], src[j00 + 4], src[j00 + 5]];
                            let j01 = j00 + w as usize * 3;
                            let t01 = [src[j01], src[j01 + 1], src[j01 + 2]];
                            let t11 = [src[j01 + 3], src[j01 + 4], src[j01 + 5]];
                            blend4(t00, t10, t01, t11, tx, ty)
                        } else {
                            sample_remapped(fx, fy, |ix, iy| {
                                let jx = ix + cl;
                                let jy = iy + ct;
                                if jx < 0 || jy < 0 || jx >= w as i64 || jy >= h as i64 {
                                    PAPER_BG
                                } else {
                                    let j = ((jy as u32 * w + jx as u32) * 3) as usize;
                                    [src[j], src[j + 1], src[j + 2]]
                                }
                            })
                        }
                    } else {
                        DESK
                    };
                    let i = ((y * w2 + x) * 3) as usize;
                    rot[i] = p[0];
                    rot[i + 1] = p[1];
                    rot[i + 2] = p[2];
                }
            }
        }
        buf = rot;
        w = w2;
        h = h2;
        }
    } else {
        // Rotation gate off: framing is a plain eager crop/pad on both backends.
        apply_framing_eager(&mut buf, &mut w, &mut h, frame_mode, frame_pad_px, frame_cut_px);
    }
    let wf = w as f32;
    let hf = h as f32;

        tick!("rotation");

    // --- 3b. physical artifacts: tears, stains, pencil, folds, highlighter ---
    // Placement is ink-aware: damage goes where the music isn't, so notation
    // stays readable (highlighter is translucent by nature).
    // The ink grid cell scales with rescale_k so quiet-cell granularity is a
    // fixed physical size (floor 8 keeps tiny images total); Ã—1.0 = 64 legacy.
    let ink_cell = ((64.0 * rescale_k).round() as u32).max(8);
    let (ink, igw, igh) = ink_grid(&buf, w, h, ink_cell);
    let mut tear_count = 0u32;
    let mut stain_count = 0u32;
    let mut pencil_count = 0u32;
    let mut fold_corner = String::from("none");
    let mut highlight = String::from("none");

    // Paper tears: jagged edge notches carved only from quiet margins.
    let n_tears = if rng.gen::<f32>() < s * 0.5 { rng.gen_range(1..=2) } else { 0 };
    for _ in 0..n_tears {
        let edge = rng.gen_range(0..4); // 0 top, 1 right, 2 bottom, 3 left
        for _ in 0..8 {
            let t = rng.gen::<f32>();
            let (gx, gy) = match edge {
                0 => ((t * igw as f32) as u32, 0),
                1 => (igw - 1, (t * igh as f32) as u32),
                2 => ((t * igw as f32) as u32, igh - 1),
                _ => (0, (t * igh as f32) as u32),
            };
            if ink[(gy.min(igh - 1) * igw + gx.min(igw - 1)) as usize] < 0.08 {
                let min_d = w.min(h) as f32;
                let along = t * if edge == 0 || edge == 2 { w as f32 } else { h as f32 };
                let mouth = min_d * rng.gen_range(0.01..0.04);
                let depth = min_d * rng.gen_range(0.01..0.03);
                // Stamped notch (algo-ink-artifacts P3b): precomputed mask,
                // same rows in the same order — bit-exact vs the direct
                // loop below, without per-pixel `sin`.
                let mask = banks::precompute_notch(along, mouth, depth);
                banks::carve_notch_stamp(&mut buf, w, h, edge, &mask);
                tear_count += 1;
                break;
            }
        }
    }

    // Coffee ring + satellite splats, parked on quiet paper.
    if rng.gen::<f32>() < s * 0.45 {
        if let Some((gx, gy)) = pick_quiet_cell(&mut rng, &ink, igw, igh, 0.06) {
            let jit = 20.0 * rescale_k; // absolute-pixel jitter (Ã—1.0 legacy)
            let cx = (gx * ink_cell + ink_cell / 2) as f32 + rng.gen_range(-jit..jit);
            let cy = (gy * ink_cell + ink_cell / 2) as f32 + rng.gen_range(-jit..jit);
            let r = w.min(h) as f32 * rng.gen_range(0.02..0.06);
            const COFFEE: [f32; 3] = [122.0, 82.0, 42.0];
            // Stamped coffee ring (algo-ink-artifacts P3b): precomputed
            // sprite, same two blends in the same order — bit-exact vs the
            // direct loop, without per-pixel `sqrt`+`exp`.
            let stamp = banks::precompute_coffee(r);
            banks::stamp_coffee(&mut buf, w, h, cx, cy, &stamp);
            for _ in 0..rng.gen_range(1..=3) {
                let a = rng.gen_range(0.0..std::f32::consts::TAU);
                let dist = r * rng.gen_range(1.2..2.2);
                let px = cx + dist * a.cos();
                let py = cy + dist * a.sin();
                let pr = (r * rng.gen_range(0.05..0.14)).max(1.0) as i64;
                for dy in -pr..=pr {
                    for dx in -pr..=pr {
                        if dx * dx + dy * dy <= pr * pr {
                            blend_px(&mut buf, w, h, px as i64 + dx, py as i64 + dy, COFFEE, 0.30);
                        }
                    }
                }
            }
            stain_count = 1;
        }
    }

    // Pencil markings: thin gray polylines in quiet areas.
    let n_pencil = if rng.gen::<f32>() < s * 0.6 { rng.gen_range(1..=3) } else { 0 };
    const PENCIL: [f32; 3] = [150.0, 150.0, 156.0];
    for _ in 0..n_pencil {
        if let Some((gx, gy)) = pick_quiet_cell(&mut rng, &ink, igw, igh, 0.06) {
            let mut px = (gx * ink_cell) as f32 + rng.gen_range(0.0..ink_cell as f32);
            let mut py = (gy * ink_cell) as f32 + rng.gen_range(0.0..ink_cell as f32);
            let mut ang = rng.gen_range(0.0..std::f32::consts::TAU);
            for _ in 0..rng.gen_range(3..=5) {
                let len = ink_cell as f32 * rng.gen_range(0.15..0.45);
                let qx = px + len * ang.cos();
                let qy = py + len * ang.sin();
                stroke(&mut buf, w, h, px, py, qx, qy, 1, PENCIL);
                px = qx;
                py = qy;
                ang += rng.gen_range(-0.9..0.9);
            }
            pencil_count += 1;
        }
    }

    // Folded corner: shaded dog-ear + crease highlight, quiet corner only.
    if rng.gen::<f32>() < s * 0.35 {
        let leg = (w.min(h) as f32 * rng.gen_range(0.04..0.10)) as u32;
        let specs = [
            ("tl", 0u32, 0u32, false, false),
            ("tr", w.saturating_sub(leg), 0, true, false),
            ("bl", 0, h.saturating_sub(leg), false, true),
            ("br", w.saturating_sub(leg), h.saturating_sub(leg), true, true),
        ];
        let mut best: Option<(&str, f32)> = None;
        for (name, x0, y0, _, _) in &specs {
            let gx1 = ((x0 + leg).min(w.saturating_sub(1))) / ink_cell;
            let gy1 = ((y0 + leg).min(h.saturating_sub(1))) / ink_cell;
            let (mut sum, mut n) = (0.0f32, 0u32);
            for gy in (y0 / ink_cell)..=gy1 {
                for gx in (x0 / ink_cell)..=gx1 {
                    sum += ink[(gy * igw + gx) as usize];
                    n += 1;
                }
            }
            let avg = sum / n.max(1) as f32;
            if best.map_or(true, |(_, b)| avg < b) {
                best = Some((name, avg));
            }
        }
        if let Some((name, avg)) = best {
            if avg < 0.10 {
                let (_, x0, y0, mx, my) = *specs.iter().find(|s| s.0 == name).unwrap();
                for dy in 0..leg {
                    for dx in 0..leg {
                        let lx = if mx { leg - 1 - dx } else { dx };
                        let ly = if my { leg - 1 - dy } else { dy };
                        if lx + ly < leg {
                            let i = (((y0 + dy) as usize) * (w as usize) + ((x0 + dx) as usize)) * 3;
                            buf[i] *= 0.82;
                            buf[i + 1] *= 0.82;
                            buf[i + 2] *= 0.82;
                            if ((lx + ly) as f32 - leg as f32).abs() < 2.5 {
                                buf[i] = buf[i] * 0.55 + 255.0 * 0.45;
                                buf[i + 1] = buf[i + 1] * 0.55 + 255.0 * 0.45;
                                buf[i + 2] = buf[i + 2] * 0.55 + 255.0 * 0.45;
                            }
                        }
                    }
                }
                fold_corner = name.to_string();
            }
        }
    }

    // Marker highlighter: translucent wash band, notes stay readable.
    if rng.gen::<f32>() < s * 0.4 {
        let palette = [
            ("yellow", [255.0, 232.0, 110.0]),
            ("pink", [255.0, 165.0, 195.0]),
            ("green", [165.0, 232.0, 165.0]),
        ];
        let (cname, col) = palette[rng.gen_range(0..3)];
        let bh = (h as f32 * rng.gen_range(0.02..0.05)) as u32;
        let y0 = (h as f32 * rng.gen_range(0.15..0.80)) as u32;
        for y in y0..(y0 + bh).min(h) {
            for x in (w as f32 * 0.10) as u32..((w as f32 * 0.88) as u32).min(w) {
                let i = ((y as usize) * (w as usize) + (x as usize)) * 3;
                for c in 0..3 {
                    buf[i + c] = buf[i + c] * 0.72 + col[c] * 0.28;
                }
            }
        }
        highlight = cname.to_string();
    }

    // Paper-fold creases: full-span valley (dark) + ridge highlight with a
    // soft one-sided shade. Creases cross notation like real folded sheets;
    // they shade but never hide. (angle_deg, pos_frac, strength) recorded.
    let mut fold_creases: Vec<[f32; 3]> = Vec::new();
    let n_crease = if rng.gen::<f32>() < s * 0.5 { rng.gen_range(1..=2) } else { 0 };
    for _ in 0..n_crease {
        let ang = rng.gen_range(0.0..180.0f32).to_radians();
        let pos = rng.gen_range(0.15..0.85);
        let cstr = rng.gen_range(0.5..1.0) * s;
        let (nx, ny) = (ang.cos(), ang.sin()); // line normal
        let valley = (w.min(h) as f32 * 0.0016).clamp(1.0, 4.0);
        let shade_w = w.min(h) as f32 * rng.gen_range(0.015..0.035);
        // Sparse band-bbox loop (algo-ink-artifacts P3a): every branch
        // needs `|d| < lim` with `lim = max(valley + 1.5, shade_w)` (valley
        // needs `av < valley`, ridge `d < valley + 1.5`, shade `av <
        // shade_w`), so pixels outside that band are provably no-ops and
        // only the intersecting x-range per row is visited (~5-10% of the
        // frame). `d` is evaluated with the exact legacy expression tree,
        // so branch outcomes and writes are bit-exact. (The full-frame
        // `CreaseField` double buffer stays unwired: it regressed this
        // stage 419.6 -> 662.7 ms on the 10-page bench.)
        let lim = (valley + 1.5).max(shade_w);
        let off = (pos - 0.5) * w.min(h) as f32;
        let hw = w as f32 / 2.0;
        let hh = h as f32 / 2.0;
        for y in 0..h {
            // d(x) = nx*x + c is linear in x: solve |nx*x + c| < lim.
            let c = ny * (y as f32 - hh) - nx * hw - off;
            let (mut xa, mut xb) = if nx.abs() < 1e-9 {
                // d is (up to fp dust) constant across the row.
                if (ny * (y as f32 - hh) - off).abs() < lim + 1e-3 {
                    (0i64, w as i64 - 1)
                } else {
                    continue;
                }
            } else {
                let (mut lo, mut hi) = ((-lim - c) / nx, (lim - c) / nx);
                if lo > hi {
                    std::mem::swap(&mut lo, &mut hi);
                }
                // 2 px guard absorbs fp rounding in the range solve; any
                // extra visited pixel recomputes the same `d` and takes the
                // same branches, so the guard cannot change output.
                (lo.floor() as i64 - 2, hi.ceil() as i64 + 2)
            };
            xa = xa.max(0);
            xb = xb.min(w as i64 - 1);
            for x in xa..=xb {
                let d = (x as f32 - hw) * nx + (y as f32 - hh) * ny - off;
                let i = ((y * w + x as u32) * 3) as usize;
                let av = d.abs();
                if av < valley {
                    for c in 0..3 { buf[i + c] *= 1.0 - 0.22 * cstr; }
                } else if (d - valley) > 0.0 && (d - valley) < 1.5 {
                    for c in 0..3 { buf[i + c] = (buf[i + c] + 24.0 * cstr).clamp(0.0, 255.0); }
                } else if d < 0.0 && av < shade_w {
                    let f = 1.0 - 0.07 * cstr * (1.0 - av / shade_w);
                    for c in 0..3 { buf[i + c] *= f; }
                }
            }
        }
        fold_creases.push([ang.to_degrees(), pos, cstr]);
    }
    clamp_buf(&mut buf);

        tick!("artifacts");

    // --- 3b2. heavy wear: ink loss, printer artifacts, eraser, stains ---
    // Runs with the other on-paper damage, before any lens effect sees it.
    // Absolute-pixel period/radius inside scale with rescale_k (1.0 = legacy).
    let wear_p = degrade_wear::apply(&mut buf, w, h, &mut rng, s, rescale_k);

        tick!("wear");

    // --- 3c. lens geometric distortion (outward barrel only) ---
    // Radial Brown-Conrady warp. Outward bow (barrel, k1 < 0) is the
    // phone-photo look; inward pull (pincushion, k1 > 0) bent staff lines
    // into unrealistic arcs (big500-degraded10 ds0424-9_d0), so sampled
    // inward pull is folded into the outward range below.
    let mut dist_k1 = s * rng.gen_range(-0.18..0.30);
    let mut dist_k2 = s * rng.gen_range(-0.05..0.05);
    let norm0 = (w.min(h) as f32 / 2.0).max(1.0);
    let r_max = (((w as f32 / 2.0).powi(2) + (h as f32 / 2.0).powi(2)).sqrt() / norm0).max(0.0);
    // Fold any inward pull outward, mirroring the sampled range. The gate
    // is the sign, not a radius scan: a corner-only tug (k2 > 0 against
    // k1 < 0) starts past the lens-stage frame edge yet lands inside the
    // final canvas after rotation expands it. Unaffected files stay
    // bit-identical; same draw count keeps downstream stages untouched.
    if dist_k1 > 0.0 {
        dist_k1 = -dist_k1 * (0.18 / 0.30);
    }
    if dist_k2 > 0.0 {
        dist_k2 = -dist_k2;
    }
    // Fit barrel outward swell to the same sliver budget as the
    // perspective warp: outward (k1 < 0) bends corners out of frame.
    // Rescale (k1, k2) so the worst outward displacement stays within
    // `bleed` pixels.
    let bleed = 0.004 * w.min(h) as f32;
    let mut dist_fit_scale = 1.0f32;
    {
        let mut peak = 0.0f32;
        let mut r = 0.0f32;
        while r <= r_max {
            let f = (1.0 + dist_k1 * r * r + dist_k2 * r * r * r * r).max(0.2);
            peak = peak.max(((1.0 / f - 1.0).max(0.0)) * r * norm0);
            r += 0.05;
        }
        if peak > bleed && peak > 1e-6 {
            dist_fit_scale = (bleed / peak).clamp(0.0, 1.0);
            dist_k1 *= dist_fit_scale;
            dist_k2 *= dist_fit_scale;
        }
    }
    // L-A (online-framing-lens §5): the gated resample above is provably
    // dead code — post-fold `k1, k2 <= 0`, and the bleed-budget fit forces
    // `|k1| <= 0.00283 < 0.005`, `|k2| <= 0.00141 < 0.002` on every input,
    // so neither gate arm can fire (proof in algo-framing-lens.md §1b).
    // The tripwire fails loudly if a future knob change revives the stage
    // instead of silently adding a full-frame pass back.
    debug_assert!(
        !(dist_k1.abs() > 0.005 || dist_k2.abs() > 0.002),
        "lens stage revived: k1={dist_k1} k2={dist_k2} — see L-A recipe before re-adding a pass"
    );

        tick!("lens");

    // --- 4. focus: uniform defocus or tilt-shift band ---
    // Cheap lenses miss perfect focus; tilt-shift keeps one band sharp while
    // the rest melts, like a phone focused mid-page.
    let focus_mode: &str = if rng.gen::<f32>() < 0.5 { "uniform" } else { "tilt" };
    let focus_y = rng.gen_range(0.25..0.75); // band center, fraction of h
    let focus_h = rng.gen_range(0.10..0.35); // band half-height, fraction of h
    // Defocus sigmas are absolute pixels: scale keeps the circle of
    // confusion a fixed fraction of a staff space (Ã—1.0 = legacy exactly).
    let tilt_sigma = s * rng.gen_range(1.0..3.0) * rescale_k;
    let blur_sigma = s * rng.gen_range(0.5..2.4) * rescale_k;
    // GPU path: blur (+ tilt blend) runs in chain2 at the noise stage.
    if gpu.is_none() {
    if focus_mode == "tilt" && tilt_sigma > 0.05 {
        // CPU smoothstep blend of the blurred copy below.
        let bb: Vec<f32> = {
            let tmp = vec_to_image(&buf, w, h);
            let blurred = image::imageops::blur(&tmp, tilt_sigma);
            blurred.pixels().flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect()
        };
        for y in 0..h {
            let d = ((y as f32 / h.max(1) as f32) - focus_y).abs();
            let m = ((d - focus_h) / 0.08).clamp(0.0, 1.0);
            let m = m * m * (3.0 - 2.0 * m); // smoothstep in/out of focus
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                for c in 0..3 {
                    buf[i + c] = buf[i + c] * (1.0 - m) + bb[i + c] * m;
                }
            }
        }
    } else if blur_sigma > 0.05 {
        {
            let tmp = vec_to_image(&buf, w, h);
            let blurred = image::imageops::blur(&tmp, blur_sigma);
            buf = blurred.pixels().flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
        }
    }
    }

        tick!("focus");

    // --- 5. lighting: specular glare + directional shadow + vignette ---
    let gx = rng.gen_range(0.0..wf);
    let gy = rng.gen_range(0.0..hf);
    let glare_radius = wf.min(hf) * rng.gen_range(0.2..0.6);
    let glare_alpha = s * rng.gen_range(0.25..0.60);
    let shadow_angle_deg: f32 = rng.gen_range(0.0..360.0);
    let shadow_strength = s * rng.gen_range(0.15..0.45);
    let vignette_strength = s * rng.gen_range(0.05..0.25);
    // Sampled here (not inside either backend branch) so the ChaCha stream
    // position stays identical on both backends; the GPU path fuses this
    // stage with lighting below when it runs.
    // Fringing is an absolute-pixel shift: scale with rescale_k (Ã—1.0 legacy).
    let ca_sampled = s * rng.gen_range(0.0..1.5) * rescale_k;
    // Skipped under --skip-chromatic/--reduced (ablation: subpixel fringing
    // is erased by Otsu). The draw above is still consumed, so the shared
    // stream position is unchanged (stream-preserving skip).
    let ca_px = if toggles.skip_chromatic { 0.0 } else { ca_sampled };
    // GPU path: lighting (+ chromatic) runs in chain2 at the noise stage.
    // Subtask-5 banked path (CPU only): same knobs, precomputed field,
    // two-pass apply — bit-exact vs the inline loop below.
    if gpu.is_none() && toggles.fast_banks {
        let theta = shadow_angle_deg.to_radians();
        let field = banks::precompute_lighting(w, h, gx, gy, glare_radius, glare_alpha, theta.cos(), theta.sin(), shadow_strength, vignette_strength);
        banks::apply_lighting_field(&mut buf, &field);
        clamp_buf(&mut buf);
    }
    if gpu.is_none() && !toggles.fast_banks {
    {
        let theta = shadow_angle_deg.to_radians();
        let (dx, dy) = (theta.cos(), theta.sin());
        let diag = (wf * wf + hf * hf).sqrt().max(1.0);
        let cx = wf / 2.0;
        let cy = hf / 2.0;
        let max_d = (cx * cx + cy * cy).sqrt().max(1.0);
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let fx = x as f32;
                let fy = y as f32;
                // Glare: quadratic falloff disc, added white.
                let gd = ((fx - gx).powi(2) + (fy - gy).powi(2)).sqrt();
                let g = (1.0 - gd / glare_radius).max(0.0);
                let add = glare_alpha * 255.0 * g * g;
                // Shadow: linear gradient across the frame.
                let t = ((fx - cx) * dx + (fy - cy) * dy) / diag + 0.5;
                let shadow = 1.0 - shadow_strength * t.clamp(0.0, 1.0);
                // Vignette: quadratic darkening toward frame corners.
                let cd = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt() / max_d;
                let vig = 1.0 - vignette_strength * cd * cd;
                let m = (shadow * vig).max(0.0);
                buf[i] = buf[i] * m + add;
                buf[i + 1] = buf[i + 1] * m + add;
                buf[i + 2] = buf[i + 2] * m + add;
            }
        }
        clamp_buf(&mut buf);
    }
    }

        tick!("lighting");

    // --- 5b. chromatic aberration: radial red/blue fringing, cheap lens ---
    // (ca_px was sampled above lighting so both backends share the stream;
    // the GPU path already applied it fused with lighting.)
    if ca_px > 0.05 {
        // GPU path already applied chromatic fused with lighting above.
        if gpu.is_none() {
        let src = buf.clone();
        for y in 0..h {
            for x in 0..w {
                let dx = (x as f32 - wf / 2.0) / wf.max(1.0);
                let dy = (y as f32 - hf / 2.0) / hf.max(1.0);
                let r = sample_bilinear(&src, w, h, x as f32 - dx * ca_px, y as f32 - dy * ca_px);
                let b = sample_bilinear(&src, w, h, x as f32 + dx * ca_px, y as f32 + dy * ca_px);
                let i = ((y * w + x) * 3) as usize;
                buf[i] = r[0];
                buf[i + 2] = b[2];
            }
        }
        clamp_buf(&mut buf);
        }
    }

        tick!("chromatic");

    // --- 5c. capture optics & handling (motion blur, prints, tape, exposure) ---
    // Runs post-lens so handling marks sit in the photographed scene.
    // Motion radius + banding period inside scale with rescale_k (1.0 = legacy).
    let capture_p = degrade_capture::apply(&mut buf, w, h, &mut rng, s, rescale_k);

        tick!("capture");

    // --- 6. sensor noise + film grain (per-channel Gaussian, stronger) ---
    // Skipped under --skip-noise/--reduced (ablation: erased by Otsu).
    // NOTE: unlike the chromatic skip, this shifts the shared stream for the
    // downscale/JPEG draws below (the per-pixel loop consumes draws).
    // Sensor-noise sigma scales with rescale_k like grain: per-pixel white
    // noise downsampled by k per axis loses amplitude ~k (Ã—1.0 = legacy).
    let noise_sampled = s * rng.gen_range(3.0..14.0) * rescale_k;
    let noise_sigma = if toggles.skip_noise { 0.0 } else { noise_sampled };
    // GPU path (chain2 + affine): blur + lighting + chromatic + noise +
    // rotate/downscale affine in ONE resident submit on the shared device.
    // Sample scale knobs before chain2 (chain2 consumes zero RNG draws) so
    // the affine pass fuses directly into chain2, eliminating an entire host
    // round-trip per page.
    let (scale_x, scale_y) = if let Some(g) = gpu {
        let base_scale = scale_min + rng.gen::<f32>() * (1.0 - scale_min);
        let sx = (base_scale * (1.0 + rng.gen_range(-0.025..0.025))).clamp(scale_min * 0.97, 1.0);
        let sy = (base_scale * (1.0 + rng.gen_range(-0.025..0.025))).clamp(scale_min * 0.97, 1.0);
        let ds_active = sx < 1.0 || sy < 1.0;
        let run_affine = rot_stash.is_some() || ds_active;
        let (affine_rows, affine_meta, affine_nw, affine_nh) = if run_affine {
            let deg = rot_stash.unwrap_or(0.0);
            let (rows, meta, _, _, nw, nh) = gpu_backend::rotscale_frame(deg, w, h, sx, sy);
            (rows, meta, nw.max(1), nh.max(1))
        } else {
            ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0; 4], w, h)
        };
        let theta = shadow_angle_deg.to_radians();
        let tilt_branch = focus_mode == "tilt" && tilt_sigma > 0.05;
        let post_cfg = ChainConfig {
            blur_sigma: if tilt_branch { tilt_sigma } else { blur_sigma },
            light: LightingKnobs {
                gx,
                gy,
                glare_radius,
                glare_alpha,
                sh_dx: theta.cos(),
                sh_dy: theta.sin(),
                shadow_strength,
                vignette_strength,
            },
            ca_px,
            noise_sigma,
            noise_seed: gpu_backend::noise_seed32(file_seed),
            run_blur: tilt_branch || blur_sigma > 0.05,
            run_light: true,
            run_ca: ca_px > 0.05,
            run_noise: noise_sigma > 0.05,
            tilt_active: tilt_branch,
            focus_y,
            focus_h,
            run_affine,
            affine_rows,
            affine_meta,
            affine_nw,
            affine_nh,
            ..ChainConfig::default()
        };
        let (chained, nw, nh) = g.chain(&buf, w, h, &post_cfg);
        buf = chained;
        w = nw;
        h = nh;
        if run_affine {
            rot_scale_fused = rot_stash.is_some();
        }
        clamp_buf(&mut buf);
        (sx, sy)
    } else {
        if toggles.fast_banks && noise_sigma > 0.05 {
            // Subtask-5 banked path: same sigma knob, keyed tiled bank.
            let bank = banks::generate_gauss_bank(file_seed, banks::TAG_NOISE, banks::BANK_TILE, noise_sigma);
            let (ox, oy) = banks::bank_offset(&mut rng, banks::BANK_TILE);
            banks::apply_noise_banked(&mut buf, w, &bank, banks::BANK_TILE, ox, oy);
            clamp_buf(&mut buf);
        } else if noise_sigma > 0.05 {
            for v in buf.iter_mut() {
                *v += gauss(&mut rng) * noise_sigma;
            }
            clamp_buf(&mut buf);
        }
        let base_scale = scale_min + rng.gen::<f32>() * (1.0 - scale_min);
        let sx = (base_scale * (1.0 + rng.gen_range(-0.025..0.025))).clamp(scale_min * 0.97, 1.0);
        let sy = (base_scale * (1.0 + rng.gen_range(-0.025..0.025))).clamp(scale_min * 0.97, 1.0);
        if sx < 1.0 || sy < 1.0 {
            let nw = ((w as f32 * sx) as u32).max(1);
            let nh = ((h as f32 * sy) as u32).max(1);
            let tmp = vec_to_image(&buf, w, h);
            let rs = resize_rgb_fir(tmp, nw, nh)?;
            w = nw;
            h = nh;
            buf = rs.pixels().flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
        }
        (sx, sy)
    };

        tick!("noise");

    // (6b capture downscale / GPU fused affine already executed above.)

        tick!("downscale");

    // --- 7. JPEG compression round-trip ---
    let jpeg_quality: u8 = match jpeg_quality {
        Some(q) => q,
        None => ((80.0 - s * 30.0 + rng.gen_range(-8.0..8.0)).clamp(30.0, 95.0)) as u8,
    };
    let mut out_img = vec_to_image(&buf, w, h);
    if jpeg_quality < 100 {
        // Fast JPEG round-trip (nvJPEG GPU first, jpeg-encoder + zune-jpeg
        // CPU fallback). Same quality number; bytes may differ, visuals match.
        if let Ok((jpeg_bytes, _)) =
            encode_jpeg_bytes(out_img.as_raw(), w, h, jpeg_quality)
        {
            if let Ok(dec) = decode_jpeg_bytes(&jpeg_bytes, "jpeg round-trip") {
                if dec.width() == w && dec.height() == h {
                    out_img = dec;
                }
            }
        }
    }

    tick!("jpeg");

    let params = Params {
        input: String::new(), // filled in by run()
        output: String::new(), // filled in by run()
        file_seed,
        strength: s,
        backend: backend.to_string(),
        preset: toggles.preset_name().to_string(),
        fast_banks: toggles.fast_banks,
        width: w,
        height: h,
        // Training-resolution rescale record (subtask 2); 0/1.0 = legacy path.
        train_width,
        src_width,
        src_height,
        rescale_k,
        paper_grain_sigma,
        paper_speckle_prob,
        curl_side,
        curl_px,
        curl_shade,
        bend_spine_cx,
        bend_spine_w,
        bend_ripple_fx,
        bend_ripple_fy,
        bend_ripple_ph,
        perspective_max_frac,
        corner_offsets,
        warp_fit_scale,
        warp_out_px,
        rotation_deg,
        rot_scale_fused,
        frame_mode: frame_mode.to_string(),
        frame_pad_px,
        frame_cut_px,
        frame_notation_bbox: frame_bbox.unwrap_or([0, 0, 0, 0]),
        frame_quiet_px,
        frame_safe_px,
        dist_k1,
        dist_k2,
        dist_fit_scale,
        focus_mode: focus_mode.to_string(),
        focus_y,
        focus_h,
        tilt_sigma,
        blur_sigma,
        glare_center: [gx, gy],
        glare_radius,
        glare_alpha,
        shadow_angle_deg,
        shadow_strength,
        vignette_strength,
        ca_px,
        ink: ink_p,
        paper: paper_p,
        wear: wear_p,
        capture: capture_p,
        scale_x,
        scale_y,
        tear_count,
        stain_count,
        pencil_count,
        fold_corner,
        fold_creases,
        highlight,
        noise_sigma,
        jpeg_quality,
        codec: "zune".to_string(),
    };
    Ok((out_img, params))
}

/// Standard-normal sample via Box-Muller (avoids an extra dependency).
fn gauss(rng: &mut ChaCha8Rng) -> f32 {
    let u1: f32 = rng.gen_range(1e-6..1.0);
    let u2: f32 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn clamp_buf(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = v.clamp(0.0, 255.0);
    }
}

/// SIMD bilinear resample (`fast_image_resize`, AVX2/SSE/NEON; MIT/Apache).
/// Replaces `image::imageops::resize(Triangle)` for the pre-size + capture
/// downscale: the Triangle filter and bilinear convolution are the same tent
/// kernel, so notation geometry is unchanged (pixels may differ ~1 LSB from
/// rounding/order — outputs are re-baselined, params untouched). No RNG.
fn resize_rgb_fir(img: RgbImage, nw: u32, nh: u32) -> Result<RgbImage, String> {
    use fast_image_resize as fir;
    let (w, h) = (img.width(), img.height());
    let src = fir::images::Image::from_vec_u8(w, h, img.into_raw(), fir::PixelType::U8x3)
        .map_err(|e| format!("fir src: {e:?}"))?;
    let mut dst = fir::images::Image::new(nw, nh, fir::PixelType::U8x3);
    let mut resizer = fir::Resizer::new();
    resizer
        .resize(
            &src,
            &mut dst,
            &fir::ResizeOptions::new()
                .resize_alg(fir::ResizeAlg::Convolution(fir::FilterType::Bilinear)),
        )
        .map_err(|e| format!("fir resize: {e:?}"))?;
    RgbImage::from_raw(nw, nh, dst.into_vec())
        .ok_or_else(|| "fir resize: bad dims".to_string())
}

fn vec_to_image(buf: &[f32], w: u32, h: u32) -> RgbImage {
    let bytes: Vec<u8> = buf.iter().map(|v| v.clamp(0.0, 255.0) as u8).collect();
    RgbImage::from_raw(w, h, bytes).expect("buffer size matches dimensions")
}

// Caps matching the old `image::Limits` contract (40k px/dim, 4 GB alloc).
const MAX_INPUT_DIM: usize = 40_000;
const MAX_INPUT_ALLOC: u64 = 4_000_000_000;

fn check_input_dims(w: usize, h: usize, what: &str) -> Result<(), String> {
    if w == 0
        || h == 0
        || w > MAX_INPUT_DIM
        || h > MAX_INPUT_DIM
        || (w as u64) * (h as u64) * 4 > MAX_INPUT_ALLOC
    {
        return Err(format!("input too large: {what}"));
    }
    Ok(())
}

/// Fast-codec input loader: PNG via zune-png, JPEG via zune-jpeg, anything
/// else rejected. Any depth/layout becomes RGB8, like `to_rgb8()` did.
fn decode_input_image(path: &Path) -> Result<RgbImage, String> {
    if let Some(hit) = decode_cached_get(path) {
        return Ok(hit);
    }
    let img = decode_input_image_uncached(path)?;
    decode_cached_put(path, &img);
    Ok(img)
}

/// Decode-once input cache (online-jpeg-load §3): the same clean PNG is
/// decoded once per distinct `(path, mtime, len)` instead of once per
/// output file. Input decode involves no RNG, so cached bytes are
/// byte-identical to a fresh decode. Bounded (cleared past 32 entries);
/// thread-safe for the worker pool. Residual caveat: a same-path rewrite
/// that preserves both mtime and byte length would read stale — the len
/// disambiguator makes that vanishingly unlikely (a rewrite that changes
/// pixels virtually always changes the compressed length).
fn decode_cache_key(path: &Path) -> Option<(PathBuf, u64, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos()
        .min(u64::MAX as u128) as u64;
    Some((path.to_path_buf(), mtime, meta.len()))
}

static DECODE_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<(PathBuf, u64, u64), RgbImage>>,
> = std::sync::OnceLock::new();

#[cfg(test)]
static DECODE_CACHE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
fn decode_cache_hits() -> u64 {
    DECODE_CACHE_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

fn decode_cached_get(path: &Path) -> Option<RgbImage> {
    let key = decode_cache_key(path)?;
    let hit = DECODE_CACHE
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
        .lock()
        .ok()?
        .get(&key)
        .cloned();
    #[cfg(test)]
    if hit.is_some() {
        DECODE_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    hit
}

fn decode_cached_put(path: &Path, img: &RgbImage) {
    let Some(key) = decode_cache_key(path) else {
        return;
    };
    let Some(mut cache) = DECODE_CACHE
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
        .lock()
        .ok()
    else {
        return;
    };
    if cache.len() >= 32 {
        cache.clear();
    }
    cache.insert(key, img.clone());
}

fn decode_input_image_uncached(path: &Path) -> Result<RgbImage, String> {
    let raw =
        std::fs::read(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let what = path.display().to_string();
    if raw.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        let mut dec =
            zune_png::PngDecoder::new(zune_png::zune_core::bytestream::ZCursor::new(&raw));
        let pixels = dec
            .decode()
            .map_err(|e| format!("cannot decode {what}: {e:?}"))?;
        let (w, h) = dec
            .dimensions()
            .ok_or_else(|| format!("bad PNG dims: {what}"))?;
        check_input_dims(w, h, &what)?;
        match pixels {
            zune_png::zune_core::result::DecodingResult::U8(buf) => {
                rgb8_from_components(&buf, w, h, &what)
            }
            zune_png::zune_core::result::DecodingResult::U16(buf) => {
                let hi: Vec<u8> = buf.iter().map(|v| (v >> 8) as u8).collect();
                rgb8_from_components(&hi, w, h, &what)
            }
            _ => Err(format!("unsupported PNG depth: {what}")),
        }
    } else if raw.starts_with(&[0xFF, 0xD8]) {
        decode_jpeg_bytes(&raw, &what)
    } else {
        Err(format!("unsupported input (need PNG/JPEG): {what}"))
    }
}

fn rgb8_from_components(buf: &[u8], w: usize, h: usize, what: &str) -> Result<RgbImage, String> {
    let n = (w as u64)
        .checked_mul(h as u64)
        .ok_or_else(|| format!("input too large: {what}"))?;
    let rgb: Vec<u8> = match (buf.len() as u64).checked_div(n.max(1)) {
        Some(3) => buf.to_vec(),
        Some(4) => buf.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect(),
        Some(1) => buf.iter().flat_map(|v| [*v, *v, *v]).collect(),
        Some(2) => buf.chunks_exact(2).flat_map(|p| [p[0], p[0], p[0]]).collect(),
        _ => return Err(format!("bad pixel layout: {what}")),
    };
    RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| format!("bad dims: {what}"))
}

/// zune-jpeg decode → RGB8. `what` labels errors (file path or stage tag).
fn decode_jpeg_bytes(raw: &[u8], what: &str) -> Result<RgbImage, String> {
    let mut dec =
        zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(raw));
    dec.decode_headers()
        .map_err(|e| format!("cannot decode {what}: {e:?}"))?;
    let info = dec
        .info()
        .ok_or_else(|| format!("bad JPEG dims: {what}"))?;
    let (w, h) = (info.width as usize, info.height as usize);
    check_input_dims(w, h, what)?;
    let cs = dec.output_colorspace();
    let buf = dec
        .decode()
        .map_err(|e| format!("cannot decode {what}: {e:?}"))?;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    match cs {
        Some(ColorSpace::RGB) => RgbImage::from_raw(w as u32, h as u32, buf)
            .ok_or_else(|| format!("bad JPEG dims: {what}")),
        Some(ColorSpace::Luma) => {
            let rgb: Vec<u8> = buf.iter().flat_map(|v| [*v, *v, *v]).collect();
            RgbImage::from_raw(w as u32, h as u32, rgb)
                .ok_or_else(|| format!("bad JPEG dims: {what}"))
        }
        other => Err(format!("unsupported JPEG colorspace {other:?}: {what}")),
    }
}

/// Encode interleaved RGB → JPEG bytes, GPU first (nvJPEG), CPU fallback
/// (jpeg-encoder SIMD). Same quality number either way; bytes may differ,
/// visuals match. Returns the bytes plus the backend tag for params.
fn encode_jpeg_bytes(
    rgb: &[u8],
    w: u32,
    h: u32,
    quality: u8,
) -> Result<(Vec<u8>, &'static str), String> {
    if jpeg_nv::is_available() {
        if let Ok(buf) = jpeg_nv::encode_rgb_jpeg(rgb, w, h, quality) {
            return Ok((buf, "nvjpeg"));
        }
        // else fall through to CPU (GPU hiccup mid-run must not fail the file)
    }
    let w16 = u16::try_from(w).map_err(|_| "width exceeds JPEG limit".to_string())?;
    let h16 = u16::try_from(h).map_err(|_| "height exceeds JPEG limit".to_string())?;
    let mut buf: Vec<u8> = Vec::new();
    jpeg_encoder::Encoder::new(&mut buf, quality)
        .encode(rgb, w16, h16, jpeg_encoder::ColorType::Rgb)
        .map_err(|e| format!("jpeg encode: {e:?}"))?;
    Ok((buf, "zune"))
}

/// Radial lens distortion factor: 1 + k1*r^2 + k2*r^4 (r normalized).
/// k1 < 0 bows lines outward (barrel), k1 > 0 pulls them in (pincushion).
/// Kept for the `lens_never_pulls_inward` gate (the L-A dead resample is
/// gone, but the fold-into-barrel sampling law is still asserted here).
#[allow(dead_code)]
fn radial_factor(r2: f32, k1: f32, k2: f32) -> f32 {
    1.0 + k1 * r2 + k2 * r2 * r2
}

/// Book-spine gutter: smooth horizontal squeeze toward `spine_cx` plus
/// darkening. Continuous and strictly monotonic in x â€” staff lines
/// compress but never tear, and no notes are swallowed at the gutter.
fn spine_map(x: f32, spine_cx: f32, spine_w: f32, depth: f32, shade_amt: f32) -> (f32, f32) {
    let t = (x - spine_cx) / spine_w.max(1.0);
    let g = (-t * t).exp();
    let sx = spine_cx + (x - spine_cx) * (1.0 - depth * g);
    (sx, 1.0 - shade_amt * g)
}

/// Running dark-pixel bbox (F-A fusion): the perspective/bend warp loops
/// fold each written pixel's channel sum into this instead of paying for a
/// standalone framing-time scan. `fold` takes the exact sum a later
/// `notation_bbox_thresh` scan would see, so the fused result is identical.
#[derive(Default)]
struct BBoxAcc {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    found: bool,
}

impl BBoxAcc {
    fn reset(&mut self) {
        *self = BBoxAcc::default();
    }
    fn fold(&mut self, sum3: f32, x: u32, y: u32, thresh3: f32) {
        if sum3 < thresh3 {
            if !self.found {
                self.x0 = x;
                self.y0 = y;
                self.x1 = x;
                self.y1 = y;
                self.found = true;
            } else {
                self.x0 = self.x0.min(x);
                self.y0 = self.y0.min(y);
                self.x1 = self.x1.max(x);
                self.y1 = self.y1.max(y);
            }
        }
    }
    fn bbox(&self) -> Option<[u32; 4]> {
        if self.found {
            Some([self.x0, self.y0, self.x1, self.y1])
        } else {
            None
        }
    }
}

/// Dark-pixel (notation) bounding box: pixels whose mean channel is < 128,
/// the same ink threshold `ink_grid` uses. Returns
/// `[min_x, min_y, max_x, max_y]` inclusive, or `None` on a blank page.
fn notation_bbox(buf: &[f32], w: u32, h: u32) -> Option<[u32; 4]> {
    notation_bbox_thresh(buf, w, h, 128.0)
}

/// Threshold-parameterized variant. The train-resolution path uses 200:
/// Triangle downscaling pre-grays thin notation (a 1 px staff line covers
/// ~40% of an output pixel at k = 0.4, landing near 150) and ink fade can
/// lift solid blacks toward ~180, so the legacy 128 misses real notation
/// and the quiet margins come back optimistic — letting tight cuts eat
/// faded staves. 200 still clears clean paper (~244, rim aging ≥ ~214)
/// while catching pre-grayed/faded ink; specks it catches (foxing, water
/// rims) only shrink the quiet margins, i.e. the conservative direction.
fn notation_bbox_thresh(buf: &[f32], w: u32, h: u32, thresh: f32) -> Option<[u32; 4]> {
    // Sum form (`sum < 3T`) instead of `(r+g+b)/3 < T`: identical predicate
    // up to the `/3.0` reciprocal rounding, and exactly the form the fused
    // warp-loop accumulation uses, so F-A is bit-exact by construction.
    let thresh3 = 3.0 * thresh;
    let mut acc = BBoxAcc::default();
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            acc.fold(buf[i] + buf[i + 1] + buf[i + 2], x, y, thresh3);
        }
    }
    acc.bbox()
}

/// Quiet (notation-free) margin depth per edge `[left, top, right, bottom]`:
/// distance from each frame edge to the notation bbox. Blank pages report
/// `u32::MAX` (nothing to protect, any in-range cut is safe).
fn quiet_margins(w: u32, h: u32, bbox: Option<[u32; 4]>) -> [u32; 4] {
    match bbox {
        Some([x0, y0, x1, y1]) => [
            x0,
            y0,
            w.saturating_sub(1).saturating_sub(x1),
            h.saturating_sub(1).saturating_sub(y1),
        ],
        None => [u32::MAX; 4],
    }
}

/// Clamp each desired cut so at least `safe` px of quiet margin survive
/// between the frame edge and the notation bbox. Notation can never be
/// cut: `cut[i] + safe <= quiet[i]` holds elementwise by construction.
fn clamp_cut_to_quiet(desired: [u32; 4], quiet: [u32; 4], safe: u32) -> [u32; 4] {
    let mut out = [0u32; 4];
    for i in 0..4 {
        out[i] = desired[i].min(quiet[i].saturating_sub(safe));
    }
    out
}

/// Crop `cut = [left, top, right, bottom]` px off the frame (tight variant).
/// The kept pixels are exactly the shifted page: content at `(x, y)` moves
/// to `(x - cut_left, y - cut_top)`.
fn apply_frame_cut(buf: &[f32], w: u32, h: u32, cut: [u32; 4]) -> (Vec<f32>, u32, u32) {
    let nw = w.saturating_sub(cut[0].saturating_add(cut[2])).max(1);
    let nh = h.saturating_sub(cut[1].saturating_add(cut[3])).max(1);
    // Clamp the copy window to the source so an over-wide cut degrades to
    // the available pixels instead of panicking (clamp_cut_to_quiet makes
    // this unreachable in the pipeline; the helpers stay total anyway).
    let nw = nw.min(w.saturating_sub(cut[0]).max(1));
    let nh = nh.min(h.saturating_sub(cut[1]).max(1));
    let mut out = vec![0.0f32; (nw * nh * 3) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let si = (((y + cut[1]) * w + (x + cut[0])) * 3) as usize;
            let di = ((y * nw + x) * 3) as usize;
            out[di..di + 3].copy_from_slice(&buf[si..si + 3]);
        }
    }
    (out, nw, nh)
}

/// Pad `pad = [left, top, right, bottom]` px of dark desk around the page
/// (inset variant). The page lands intact at `(pad_left, pad_top)`; the
/// surround uses the same desk tone as the rotation stage.
fn apply_frame_pad(buf: &[f32], w: u32, h: u32, pad: [u32; 4]) -> (Vec<f32>, u32, u32) {
    const DESK: [f32; 3] = [25.0, 25.0, 30.0];
    let nw = w.saturating_add(pad[0].saturating_add(pad[2])).max(1);
    let nh = h.saturating_add(pad[1].saturating_add(pad[3])).max(1);
    let mut out = vec![0.0f32; (nw * nh * 3) as usize];
    for i in 0..(nw * nh) as usize {
        out[i * 3] = DESK[0];
        out[i * 3 + 1] = DESK[1];
        out[i * 3 + 2] = DESK[2];
    }
    let cw = w.min(nw.saturating_sub(pad[0]));
    let ch = h.min(nh.saturating_sub(pad[1]));
    for y in 0..ch {
        for x in 0..cw {
            let si = ((y * w + x) * 3) as usize;
            let di = (((y + pad[1]) * nw + (x + pad[0])) * 3) as usize;
            out[di..di + 3].copy_from_slice(&buf[si..si + 3]);
        }
    }
    (out, nw, nh)
}

/// Framed dims without materializing: tight crops, inset pads — the same
/// arithmetic as `apply_frame_cut`/`apply_frame_pad`, for the F-B fused
/// rotation loop.
fn framed_dims(w: u32, h: u32, mode: &str, pad: [u32; 4], cut: [u32; 4]) -> (u32, u32) {
    if mode == "inset" {
        (
            w.saturating_add(pad[0].saturating_add(pad[2])).max(1),
            h.saturating_add(pad[1].saturating_add(pad[3])).max(1),
        )
    } else {
        let nw = w.saturating_sub(cut[0].saturating_add(cut[2])).max(1);
        let nh = h.saturating_sub(cut[1].saturating_add(cut[3])).max(1);
        (
            nw.min(w.saturating_sub(cut[0]).max(1)),
            nh.min(h.saturating_sub(cut[1]).max(1)),
        )
    }
}

/// Materialize pending framing (crop XOR pad). Used on the `--gpu` path
/// (stages 3b-6 run framed while rotation is deferred to 6b) and whenever
/// the rotation gate is off.
fn apply_framing_eager(
    buf: &mut Vec<f32>,
    w: &mut u32,
    h: &mut u32,
    mode: &str,
    pad: [u32; 4],
    cut: [u32; 4],
) {
    if mode == "inset" {
        if pad.iter().sum::<u32>() > 0 {
            let (fb, fw, fh) = apply_frame_pad(buf, *w, *h, pad);
            *buf = fb;
            *w = fw;
            *h = fh;
        }
    } else if cut.iter().sum::<u32>() > 0 {
        let (fb, fw, fh) = apply_frame_cut(buf, *w, *h, cut);
        *buf = fb;
        *w = fw;
        *h = fh;
    }
}

fn set_px(buf: &mut [f32], w: u32, h: u32, x: i64, y: i64, c: [f32; 3]) {
    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
        let i = ((y as u32 as usize) * (w as usize) + (x as u32 as usize)) * 3;
        buf[i] = c[0];
        buf[i + 1] = c[1];
        buf[i + 2] = c[2];
    }
}

fn blend_px(buf: &mut [f32], w: u32, h: u32, x: i64, y: i64, c: [f32; 3], a: f32) {
    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
        let i = ((y as u32 as usize) * (w as usize) + (x as u32 as usize)) * 3;
        for k in 0..3 {
            buf[i + k] = buf[i + k] * (1.0 - a) + c[k] * a;
        }
    }
}

/// Thin polyline stamp (pencil strokes).
fn stroke(buf: &mut [f32], w: u32, h: u32, x0: f32, y0: f32, x1: f32, y1: f32, r: i64, c: [f32; 3]) {
    let steps = ((x1 - x0).abs().max((y1 - y0).abs()) as i64).max(1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let cx = (x0 + (x1 - x0) * t) as i64;
        let cy = (y0 + (y1 - y0) * t) as i64;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    set_px(buf, w, h, cx + dx, cy + dy, c);
                }
            }
        }
    }
}

/// Ink fraction per `cell`-px grid cell (ink = mean channel < 128).
fn ink_grid(buf: &[f32], w: u32, h: u32, cell: u32) -> (Vec<f32>, u32, u32) {
    let gw = ((w + cell - 1) / cell).max(1);
    let gh = ((h + cell - 1) / cell).max(1);
    let mut grid = vec![0f32; (gw * gh) as usize];
    let mut counts = vec![0u32; (gw * gh) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y as usize) * (w as usize) + (x as usize)) * 3;
            let ink = if (buf[i] + buf[i + 1] + buf[i + 2]) / 3.0 < 128.0 { 1.0 } else { 0.0 };
            let gi = ((y / cell) * gw + (x / cell)) as usize;
            grid[gi] += ink;
            counts[gi] += 1;
        }
    }
    for (g, c) in grid.iter_mut().zip(counts.iter()) {
        *g /= (*c).max(1) as f32;
    }
    (grid, gw, gh)
}

/// Random grid cell with ink fraction below `max_ink`, or `None`.
fn pick_quiet_cell(rng: &mut ChaCha8Rng, grid: &[f32], gw: u32, gh: u32, max_ink: f32) -> Option<(u32, u32)> {
    for _ in 0..12 {
        let gx = rng.gen_range(0..gw);
        let gy = rng.gen_range(0..gh);
        if grid[(gy * gw + gx) as usize] < max_ink {
            return Some((gx, gy));
        }
    }
    None
}

/// Bilinear sample with remapped taps: the coordinate expression tree
/// (floor/frac/blend weights) is exactly `sample_bilinear`'s, but each tap
/// value comes from `tap(ix, iy)` instead of a flat buffer. Used by the F-B
/// fused rotation loop so interior pixels are bit-exact vs the legacy loop.
fn sample_remapped(fx: f32, fy: f32, tap: impl Fn(i64, i64) -> [f32; 3]) -> [f32; 3] {
    let x0 = fx.floor() as i64;
    let y0 = fy.floor() as i64;
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let top = tap(x0, y0)[c] * (1.0 - tx) + tap(x0 + 1, y0)[c] * tx;
        let bot = tap(x0, y0 + 1)[c] * (1.0 - tx) + tap(x0 + 1, y0 + 1)[c] * tx;
        out[c] = top * (1.0 - ty) + bot * ty;
    }
    out
}

/// Blend four already-fetched taps with bilinear weights: the same
/// expression tree as `sample_bilinear`/`sample_remapped`, factored out so
/// hot loops fetch each tap once instead of once per channel (bit-exact —
/// same values combined in the same order).
fn blend4(t00: [f32; 3], t10: [f32; 3], t01: [f32; 3], t11: [f32; 3], tx: f32, ty: f32) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let top = t00[c] * (1.0 - tx) + t10[c] * tx;
        let bot = t01[c] * (1.0 - tx) + t11[c] * tx;
        out[c] = top * (1.0 - ty) + bot * ty;
    }
    out
}

/// Bilinear sample; out-of-bounds yields paper white.
/// Fast path: interior pixels fetch the four taps once (the old form
/// re-evaluated every tap per channel — 12 fetches — with identical weights
/// and order, so this is bit-exact) and skip the per-tap bounds checks.
pub(crate) fn sample_bilinear(buf: &[f32], w: u32, h: u32, x: f32, y: f32) -> [f32; 3] {
    const BG: [f32; 3] = [246.0, 243.0, 236.0];
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    // All four taps in-bounds: no tap can yield BG, so fetch directly.
    if x0 >= 0 && y0 >= 0 && x0 + 1 < w as i64 && y0 + 1 < h as i64 {
        let j00 = ((y0 as u32 * w + x0 as u32) * 3) as usize;
        let t00 = [buf[j00], buf[j00 + 1], buf[j00 + 2]];
        let t10 = [buf[j00 + 3], buf[j00 + 4], buf[j00 + 5]];
        let j01 = j00 + w as usize * 3;
        let t01 = [buf[j01], buf[j01 + 1], buf[j01 + 2]];
        let t11 = [buf[j01 + 3], buf[j01 + 4], buf[j01 + 5]];
        return blend4(t00, t10, t01, t11, fx, fy);
    }
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let at = |xx: i64, yy: i64| -> f32 {
            if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                BG[c]
            } else {
                buf[((yy as u32 * w + xx as u32) * 3 + c as u32) as usize]
            }
        };
        let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
        let bot = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
        out[c] = top * (1.0 - fy) + bot * fy;
    }
    out
}

/// Solve for the homography mapping `src -> dst` (h33 fixed to 1).
fn solve_homography(src: [(f64, f64); 4], dst: [(f64, f64); 4]) -> Option<[f64; 9]> {
    // 8 unknowns h11..h32; two rows per correspondence.
    let mut a = [[0.0f64; 9]; 8]; // augmented 8x8 + rhs
    for i in 0..4 {
        let (x, y) = src[i];
        let (xp, yp) = dst[i];
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -xp * x, -xp * y, xp];
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -yp * x, -yp * y, yp];
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..8 {
        let mut piv = col;
        for row in col..8 {
            if a[row][col].abs() > a[piv][col].abs() {
                piv = row;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        let d = a[col][col];
        for k in col..9 {
            a[col][k] /= d;
        }
        for row in 0..8 {
            if row != col {
                let f = a[row][col];
                for k in col..9 {
                    a[row][k] -= f * a[col][k];
                }
            }
        }
    }
    let h = [a[0][8], a[1][8], a[2][8], a[3][8], a[4][8], a[5][8], a[6][8], a[7][8], 1.0];
    Some(h)
}

fn invert_3x3(m: [f64; 9]) -> Option<[f64; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = [
        (m[4] * m[8] - m[5] * m[7]) / det,
        (m[2] * m[7] - m[1] * m[8]) / det,
        (m[1] * m[5] - m[2] * m[4]) / det,
        (m[5] * m[6] - m[3] * m[8]) / det,
        (m[0] * m[8] - m[2] * m[6]) / det,
        (m[2] * m[3] - m[0] * m[5]) / det,
        (m[3] * m[7] - m[4] * m[6]) / det,
        (m[1] * m[6] - m[0] * m[7]) / det,
        (m[0] * m[4] - m[1] * m[3]) / det,
    ];
    Some(inv)
}

fn apply_h(h: &[f64; 9], x: f64, y: f64) -> (f64, f64) {
    let z = h[6] * x + h[7] * y + h[8];
    if z.abs() < 1e-12 {
        return (x, y);
    }
    ((h[0] * x + h[1] * y + h[2]) / z, (h[3] * x + h[4] * y + h[5]) / z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_input(dir: &Path, name: &str) -> PathBuf {
        let mut img = RgbImage::from_pixel(64, 64, image::Rgb([255, 255, 255]));
        for x in 8..56 {
            for t in 0..2 {
                img.put_pixel(x, 20 + t, image::Rgb([0, 0, 0]));
            }
        }
        let p = dir.join(name);
        img.save(&p).unwrap();
        p
    }

    fn test_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("photo-degrade-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Staffed synthetic page: horizontal staff-line bars plus a solid
    /// notation rect on white, for train-resolution tests.
    fn staffed_input(dir: &Path, name: &str, w: u32, h: u32) -> PathBuf {
        let mut img = RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));
        for y in (h / 4..h * 3 / 4).step_by(10) {
            for x in w / 8..w * 7 / 8 {
                img.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
        for y in h / 3..h / 3 + 40 {
            for x in w / 3..w / 3 + 60 {
                img.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
        let p = dir.join(name);
        img.save(&p).unwrap();
        p
    }

    #[test]
    fn degrade_is_deterministic_and_records_bend_mode() {
        let dir = test_dir("det");
        let input = tiny_input(&dir, "in.png");
        let (a_img, a_p) = degrade_file(&input, 42, 0.6, Some(80), 0, 0, 1.0, None, &StageToggles::default()).unwrap();
        let (b_img, _) = degrade_file(&input, 42, 0.6, Some(80), 0, 0, 1.0, None, &StageToggles::default()).unwrap();
        assert_eq!(a_img.as_raw(), b_img.as_raw(), "same seed must give same pixels");
        assert!(
            ["left", "right", "top", "bottom", "spine", "ripple"].contains(&a_p.curl_side.as_str()),
            "unexpected bend mode {}",
            a_p.curl_side
        );
        assert_eq!(a_p.jpeg_quality, 80);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn zune_codecs_roundtrip_rgb() {
        // Fast-codec swap (zune-png in, jpeg-encoder out, zune-jpeg back):
        // a PNG written by `image` must decode pixel-identically via zune,
        // and a q90 JPEG round-trip must keep dims with small mean abs drift.
        let dir = test_dir("codec");
        let input = tiny_input(&dir, "in.png");
        let img = decode_input_image(&input).unwrap();
        assert_eq!((img.width(), img.height()), (64, 64));
        assert_eq!(img.get_pixel(0, 0).0, [255, 255, 255]);
        assert_eq!(img.get_pixel(10, 20).0, [0, 0, 0]);
        let jp = dir.join("round.jpg");
        let (bytes, _) = encode_jpeg_bytes(img.as_raw(), 64, 64, 90).unwrap();
        std::fs::write(&jp, &bytes).unwrap();
        let raw = std::fs::read(&jp).unwrap();
        let back = decode_jpeg_bytes(&raw, "test").unwrap();
        assert_eq!((back.width(), back.height()), (64, 64));
        let drift: u64 = img
            .as_raw()
            .iter()
            .zip(back.as_raw().iter())
            .map(|(a, b)| (*a as i16 - *b as i16).unsigned_abs() as u64)
            .sum();
        let mean = drift as f64 / img.as_raw().len() as f64;
        assert!(mean < 8.0, "JPEG q90 mean abs drift too large: {mean}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn nvjpeg_encode_roundtrip() {
        // GPU JPEG path: nvJPEG encode → zune-jpeg decode keeps dims with
        // small mean abs drift. Skips (passes) on CUDA-less builds/machines;
        // the CPU fallback path is covered by `zune_codecs_roundtrip_rgb`.
        if !jpeg_nv::is_available() {
            eprintln!("SKIP nvjpeg_encode_roundtrip: no CUDA/nvJPEG here");
            return;
        }
        let dir = test_dir("nvjpeg");
        let input = tiny_input(&dir, "in.png");
        let img = decode_input_image(&input).unwrap();
        let (buf, tag) = encode_jpeg_bytes(img.as_raw(), 64, 64, 90).unwrap();
        assert_eq!(tag, "nvjpeg");
        let back = decode_jpeg_bytes(&buf, "test").unwrap();
        assert_eq!((back.width(), back.height()), (64, 64));
        let drift: u64 = img
            .as_raw()
            .iter()
            .zip(back.as_raw().iter())
            .map(|(a, b)| (*a as i16 - *b as i16).unsigned_abs() as u64)
            .sum();
        let mean = drift as f64 / img.as_raw().len() as f64;
        assert!(mean < 12.0, "nvJPEG q90 mean abs drift too large: {mean}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn spine_map_is_continuous_and_monotonic() {
        // The gutter must squeeze, never tear or fold over: adjacent pixels
        // stay within (1+depth)x of each other and order is preserved
        // (regression test for the staff-line jump seen in degraded previews;
        // the old mirrored shift jumped tens of pixels at the gutter).
        let depth = 0.15;
        let mut prev: Option<f32> = None;
        for x in 0..2000 {
            let (sx, shade) = spine_map(x as f32, 1000.0, 80.0, depth, 0.3);
            if let Some(p) = prev {
                assert!((sx - p).abs() <= 1.0 + depth + 1e-3, "tear at x={x}");
                assert!(sx >= p - 1e-6, "fold-over at x={x}");
            }
            assert!((0.0..=1.0).contains(&shade), "shade out of range at x={x}");
            prev = Some(sx);
        }
    }

    #[test]
    fn radial_factor_shapes() {
        assert!((radial_factor(0.0, -0.2, 0.05) - 1.0).abs() < 1e-6, "center undistorted");
        assert!(radial_factor(1.0, -0.2, 0.0) < 1.0, "negative k1 must bow outward");
        assert!(radial_factor(1.0, 0.3, 0.0) > 1.0, "positive k1 must pull inward");
    }

    #[test]
    fn lens_never_pulls_inward() {
        // Regression: strong inward (pincushion) pull bent staff lines into
        // unrealistic arcs (big500-degraded10 ds0424-9_d0, dist_k1 = +0.166
        // at full strength). Only outward (barrel) bow may survive sampling.
        let dir = test_dir("lens");
        let input = tiny_input(&dir, "in.png"); // 64x64
        let (w, h) = (64.0f32, 64.0f32);
        let n0 = (w.min(h) / 2.0).max(1.0);
        let rmax = ((w / 2.0).powi(2) + (h / 2.0).powi(2)).sqrt() / n0;
        for seed in 0..40u64 {
            let (_, p) = degrade_file(&input, seed, 0.9, Some(80), 0, 0, 1.0, None, &StageToggles::default()).unwrap();
            let mut r = 0.05f32;
            while r <= rmax {
                let f = 1.0 + p.dist_k1 * r * r + p.dist_k2 * r * r * r * r;
                assert!(
                    f <= 1.0 + 1e-6,
                    "seed {seed}: inward pull f={f} at r={r} (k1={}, k2={})",
                    p.dist_k1, p.dist_k2
                );
                r += 0.05;
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ink_grid_and_quiet_pick() {
        let (w, h) = (128u32, 128u32);
        let mut buf = vec![255.0f32; (w * h * 3) as usize];
        for y in 0..64 {
            for x in 0..64 {
                let i = ((y * w + x) * 3) as usize;
                buf[i] = 0.0;
                buf[i + 1] = 0.0;
                buf[i + 2] = 0.0;
            }
        }
        let (grid, gw, gh) = ink_grid(&buf, w, h, 64);
        assert_eq!((gw, gh), (2, 2));
        assert!((grid[0] - 1.0).abs() < 1e-6);
        assert_eq!(grid[3], 0.0);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let got = pick_quiet_cell(&mut rng, &grid, gw, gh, 0.06);
        assert!(got.is_some(), "white cell must be pickable");
        let full = vec![1.0f32; 4];
        assert!(pick_quiet_cell(&mut rng, &full, 2, 2, 0.06).is_none());
    }

    #[test]
    fn output_scale_within_bounds_and_near_uniform() {
        let dir = test_dir("scale");
        let input = tiny_input(&dir, "in.png"); // 64x64
        let (img, p) = degrade_file(&input, 7, 0.6, Some(80), 0, 0, 0.5, None, &StageToggles::default()).unwrap();
        assert!((0.48..=1.0).contains(&p.scale_x), "sx={}", p.scale_x);
        assert!((0.48..=1.0).contains(&p.scale_y), "sy={}", p.scale_y);
        let rel = ((p.scale_x - p.scale_y) / ((p.scale_x + p.scale_y) / 2.0)).abs();
        assert!(rel <= 0.051, "x/y variance {rel} exceeds ~5%");
        // Upper bound allows rotation expansion (10 deg: cos+sin ~= 1.16x)
        // plus the framing inset pads (<= 4.5% of min-dim per side); the
        // lower bound also allows the tight-crop variant before downscale.
        assert!(img.width() <= 84 && img.width() >= 28, "w={}", img.width());
        assert!(img.height() <= 84 && img.height() >= 28, "h={}", img.height());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn warp_corners_stay_within_sliver_and_rotation_bounded() {
        let dir = test_dir("fit");
        let input = tiny_input(&dir, "in.png"); // 64x64
        let bleed = 0.004 * 64.0 + 1e-3;
        for seed in [1u64, 7, 42, 1234, 99991] {
            let (img, p) = degrade_file(&input, seed, 0.9, Some(80), 0, 0, 1.0, None, &StageToggles::default()).unwrap();
            assert!((0.0..=1.0).contains(&p.warp_fit_scale), "fit {}", p.warp_fit_scale);
            assert!(p.warp_out_px <= bleed, "out {} > bleed {bleed}", p.warp_out_px, bleed = bleed);
            assert!(p.rotation_deg.abs() <= 9.0 + 1e-3, "rot {}", p.rotation_deg);
            assert!((0.0..=1.0).contains(&p.dist_fit_scale), "dfit {}", p.dist_fit_scale);
            assert!(img.width() <= 84 && img.height() <= 84);
            for c in p.corner_offsets {
                assert!(c[0].abs() <= 64.0 * 0.07 + 1e-3 && c[1].abs() <= 64.0 * 0.07 + 1e-3);
            }
            // Crease records stay in range.
            assert!(p.fold_creases.len() <= 2);
            for k in &p.fold_creases {
                assert!((0.0..180.0).contains(&k[0]) && (0.15..=0.85).contains(&k[1]));
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn frame_cut_clamped_to_quiet_and_keeps_notation() {
        // Synthetic page: black rect (notation stand-in) x 20..=44,
        // y 10..=30 on a 64x64 white sheet.
        let (w, h) = (64u32, 64u32);
        let mut buf = vec![255.0f32; (w * h * 3) as usize];
        for y in 10..=30u32 {
            for x in 20..=44u32 {
                let i = ((y * w + x) * 3) as usize;
                buf[i] = 0.0;
                buf[i + 1] = 0.0;
                buf[i + 2] = 0.0;
            }
        }
        let bbox = notation_bbox(&buf, w, h).expect("must find the rect");
        assert_eq!(bbox, [20, 10, 44, 30]);
        let quiet = quiet_margins(w, h, Some(bbox));
        assert_eq!(quiet, [20, 10, 19, 33]);
        // Desired cuts exceed every margin: clamping keeps `safe` px back.
        let cut = clamp_cut_to_quiet([30, 30, 30, 30], quiet, 3);
        assert_eq!(cut, [17, 7, 16, 30]);
        let (cropped, nw, nh) = apply_frame_cut(&buf, w, h, cut);
        assert_eq!((nw, nh), (64 - 33, 64 - 37));
        // The whole rect survives at its shifted position: no notation cut.
        let kept = notation_bbox(&cropped, nw, nh).expect("notation must survive the crop");
        assert_eq!(kept, [20 - 17, 10 - 7, 44 - 17, 30 - 7]);
    }

    #[test]
    fn frame_helpers_handle_blank_and_empty_cuts() {
        let (w, h) = (64u32, 64u32);
        let blank = vec![255.0f32; (w * h * 3) as usize];
        assert_eq!(notation_bbox(&blank, w, h), None);
        // Blank page: nothing to protect, desired cuts pass through.
        let quiet = quiet_margins(w, h, None);
        assert_eq!(clamp_cut_to_quiet([2, 3, 2, 3], quiet, 3), [2, 3, 2, 3]);
        // Zero cut / zero pad are no-op copies.
        let (same, nw, nh) = apply_frame_cut(&blank, w, h, [0, 0, 0, 0]);
        assert_eq!((nw, nh), (w, h));
        assert_eq!(same.len(), blank.len());
        let (padded, pw, ph) = apply_frame_pad(&blank, w, h, [0, 0, 0, 0]);
        assert_eq!((pw, ph), (w, h));
        assert_eq!(padded.len(), blank.len());
        // Nonzero pad grows the canvas with dark desk fill, page intact at
        // the pad offset.
        let (padded2, pw2, ph2) = apply_frame_pad(&blank, w, h, [2, 3, 4, 5]);
        assert_eq!((pw2, ph2), (70, 72));
        assert_eq!(padded2[0], 25.0, "pad corner must be desk fill");
        let src_i = ((0 * w + 0) * 3) as usize;
        let dst_i = (((0 + 3) * pw2 + (0 + 2)) * 3) as usize;
        assert_eq!(padded2[dst_i..dst_i + 3], blank[src_i..src_i + 3]);
    }

    #[test]
    fn fused_bbox_matches_standalone_scan() {
        // F-A equality gate: folding exact written-pixel sums through
        // `BBoxAcc` (what the warp loops do) must equal the standalone
        // `notation_bbox_thresh` scan on the same buffer — for both
        // thresholds, on structured and random data.
        let (w, h) = (64u32, 48u32);
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        for thresh in [128.0f32, 200.0] {
            for round in 0..3 {
                let mut buf = vec![244.0f32; (w * h * 3) as usize];
                for v in buf.iter_mut() {
                    // Mix of paper, gray, and ink values, some near threshold.
                    let r: f32 = rng.gen();
                    *v = if round == 2 && r < 0.3 {
                        100.0 + rng.gen::<f32>() * 60.0
                    } else if r < 0.15 {
                        rng.gen::<f32>() * 60.0
                    } else if r < 0.25 {
                        200.0 + rng.gen::<f32>() * 55.0
                    } else {
                        244.0
                    };
                }
                let mut acc = BBoxAcc::default();
                for y in 0..h {
                    for x in 0..w {
                        let i = ((y * w + x) * 3) as usize;
                        acc.fold(buf[i] + buf[i + 1] + buf[i + 2], x, y, 3.0 * thresh);
                    }
                }
                assert_eq!(
                    acc.bbox(),
                    notation_bbox_thresh(&buf, w, h, thresh),
                    "thresh {thresh} round {round}"
                );
            }
        }
        // Exact known-answer still holds under the sum form.
        let mut rect = vec![255.0f32; (w * h * 3) as usize];
        for y in 10..30u32 {
            for x in 20..44u32 {
                let i = ((y * w + x) * 3) as usize;
                rect[i] = 0.0;
                rect[i + 1] = 0.0;
                rect[i + 2] = 0.0;
            }
        }
        assert_eq!(notation_bbox(&rect, w, h), Some([20, 10, 43, 29]));
    }

    #[test]
    fn decode_cache_hits_and_sees_rewrites() {
        // Decode-once gate: repeat decode hits the cache with identical
        // bytes; a rewritten file (new mtime/len key) decodes fresh.
        let dir = test_dir("deccache");
        let input = tiny_input(&dir, "in.png");
        let before = decode_cache_hits();
        let a = decode_input_image(&input).unwrap();
        let b = decode_input_image(&input).unwrap();
        assert_eq!(a.as_raw(), b.as_raw());
        assert!(
            decode_cache_hits() >= before + 1,
            "second decode of the same file must hit the cache"
        );
        // Rewrite with different dims (hence a different byte length, and a
        // new mtime): the cache must not serve the stale image.
        let mut black = RgbImage::from_pixel(48, 48, image::Rgb([0, 0, 0]));
        for x in 0..48 {
            black.put_pixel(x, 0, image::Rgb([255, 255, 255]));
        }
        black.save(&input).unwrap();
        let c = decode_input_image(&input).unwrap();
        assert_eq!((c.width(), c.height()), (48, 48));
        assert_ne!(a.as_raw(), c.as_raw(), "rewritten file must decode fresh");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fir_resize_matches_dims_and_stays_close_to_triangle() {
        // fast_image_resize wiring: correct dims on shrink/grow, and close
        // to the legacy Triangle output (same tent kernel, ~1 LSB order).
        let mut img = RgbImage::from_pixel(64, 48, image::Rgb([240, 240, 240]));
        for y in 10..30 {
            for x in 8..56 {
                img.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }
        let legacy = image::imageops::resize(&img, 32, 24, image::imageops::FilterType::Triangle);
        let small = resize_rgb_fir(img, 32, 24).unwrap();
        assert_eq!((small.width(), small.height()), (32, 24));
        let drift: u64 = small
            .as_raw()
            .iter()
            .zip(legacy.as_raw().iter())
            .map(|(a, b)| (*a as i16 - *b as i16).unsigned_abs() as u64)
            .sum();
        let mean = drift as f64 / small.as_raw().len() as f64;
        assert!(mean < 2.0, "FIR vs Triangle mean drift too large: {mean}");
    }

    #[test]
    fn framing_in_rotation_inset_is_bit_exact() {
        // F-B inset gate: the fused rotation+framing loop must equal
        // eager-pad-then-rotate bit-for-bit (pad fill == DESK == rotation
        // fill, same taps in the same order).
        let (w, h) = (48u32, 36u32);
        let mut rng = ChaCha8Rng::seed_from_u64(77);
        let mut buf = vec![240.0f32; (w * h * 3) as usize];
        for v in buf.iter_mut() {
            *v = rng.gen_range(0.0..255.0);
        }
        let pad = [3u32, 5, 4, 6];
        for deg in [0.5f32, 3.0, -7.5, 10.0] {
            // Eager reference: pad, then the legacy rotation loop body.
            let (mut eb, ew, eh) = apply_frame_pad(&buf, w, h, pad);
            let th = deg.to_radians();
            let (sn, cs) = (th.sin().abs(), th.cos().abs());
            let (w2, h2) = (
                ((ew as f32) * cs + (eh as f32) * sn).ceil().max(1.0) as u32,
                ((ew as f32) * sn + (eh as f32) * cs).ceil().max(1.0) as u32,
            );
            const DESK: [f32; 3] = [25.0, 25.0, 30.0];
            let src = std::mem::take(&mut eb);
            let (cx0, cy0) = (ew as f32 / 2.0, eh as f32 / 2.0);
            let (cx1, cy1) = (w2 as f32 / 2.0, h2 as f32 / 2.0);
            let (snt, cst) = (th.sin(), th.cos());
            let mut expect = vec![0.0f32; (w2 * h2 * 3) as usize];
            for y in 0..h2 {
                for x in 0..w2 {
                    let dx = x as f32 - cx1;
                    let dy = y as f32 - cy1;
                    let sx = cx0 + dx * cst + dy * snt;
                    let sy = cy0 - dx * snt + dy * cst;
                    let p = if sx >= 0.0 && sx < ew as f32 && sy >= 0.0 && sy < eh as f32 {
                        sample_bilinear(&src, ew, eh, sx, sy)
                    } else {
                        DESK
                    };
                    let i = ((y * w2 + x) * 3) as usize;
                    expect[i..i + 3].copy_from_slice(&p);
                }
            }
            // Fused path under test: framed dims helper + remapped taps over
            // padded coords (mirrors the pipeline's F-B inset branch).
            let (fw, fh) = framed_dims(w, h, "inset", pad, [0; 4]);
            assert_eq!((fw, fh), (ew, eh), "deg {deg}: framed dims must match eager pad");
            let (cx0f, cy0f) = (fw as f32 / 2.0, fh as f32 / 2.0);
            let (pl, pt) = (pad[0] as i64, pad[1] as i64);
            const PAPER_BG: [f32; 3] = [246.0, 243.0, 236.0];
            let mut got = vec![0.0f32; (w2 * h2 * 3) as usize];
            for y in 0..h2 {
                for x in 0..w2 {
                    let dx = x as f32 - cx1;
                    let dy = y as f32 - cy1;
                    let fx = cx0f + dx * cst + dy * snt;
                    let fy = cy0f - dx * snt + dy * cst;
                    let p = if fx >= 0.0 && fx < fw as f32 && fy >= 0.0 && fy < fh as f32 {
                        sample_remapped(fx, fy, |ix, iy| {
                            if ix < 0 || iy < 0 || ix >= fw as i64 || iy >= fh as i64 {
                                PAPER_BG
                            } else {
                                let jx = ix - pl;
                                let jy = iy - pt;
                                if jx < 0 || jy < 0 || jx >= w as i64 || jy >= h as i64 {
                                    DESK
                                } else {
                                    let j = ((jy as u32 * w + jx as u32) * 3) as usize;
                                    [buf[j], buf[j + 1], buf[j + 2]]
                                }
                            }
                        })
                    } else {
                        DESK
                    };
                    let i = ((y * w2 + x) * 3) as usize;
                    got[i..i + 3].copy_from_slice(&p);
                }
            }
            assert_eq!(got, expect, "deg {deg}: fused inset must be bit-exact");
        }
    }

    #[test]
    fn framing_in_rotation_tight_differs_only_inside_rim() {
        // F-B tight gate: fused sampling reads true neighboring page pixels
        // where the legacy path blended against paper BG — diffs are
        // confined to a 1 px band inside the cut edges, params untouched.
        let (w, h) = (48u32, 36u32);
        let mut buf = vec![240.0f32; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let v = if (x + y) % 3 == 0 { 20.0 } else { 240.0 };
                buf[i] = v;
                buf[i + 1] = v;
                buf[i + 2] = v;
            }
        }
        let cut = [4u32, 5, 3, 6];
        let deg = 5.0f32;
        // Eager reference: cut, then legacy rotation.
        let (mut eb, ew, eh) = apply_frame_cut(&buf, w, h, cut);
        let th = deg.to_radians();
        let (sn, cs) = (th.sin().abs(), th.cos().abs());
        let (w2, h2) = (
            ((ew as f32) * cs + (eh as f32) * sn).ceil().max(1.0) as u32,
            ((ew as f32) * sn + (eh as f32) * cs).ceil().max(1.0) as u32,
        );
        const DESK: [f32; 3] = [25.0, 25.0, 30.0];
        let src = std::mem::take(&mut eb);
        let (cx0, cy0) = (ew as f32 / 2.0, eh as f32 / 2.0);
        let (cx1, cy1) = (w2 as f32 / 2.0, h2 as f32 / 2.0);
        let (snt, cst) = (th.sin(), th.cos());
        let mut expect = vec![0.0f32; (w2 * h2 * 3) as usize];
        // Track, per output pixel, whether every legacy tap stayed strictly
        // inside the cropped buffer (interior) or touched the rim/outside.
        let mut interior = vec![false; (w2 * h2) as usize];
        for y in 0..h2 {
            for x in 0..w2 {
                let dx = x as f32 - cx1;
                let dy = y as f32 - cy1;
                let sx = cx0 + dx * cst + dy * snt;
                let sy = cy0 - dx * snt + dy * cst;
                let inside = sx >= 0.0 && sx < ew as f32 && sy >= 0.0 && sy < eh as f32;
                let p = if inside {
                    sample_bilinear(&src, ew, eh, sx, sy)
                } else {
                    DESK
                };
                let i = ((y * w2 + x) * 3) as usize;
                expect[i..i + 3].copy_from_slice(&p);
                // Interior iff all four taps are >= 1 px inside the cropped
                // frame (then the fused true-neighbor taps agree exactly).
                interior[(y * w2 + x) as usize] = inside
                    && sx >= 1.0
                    && sy >= 1.0
                    && sx < ew as f32 - 1.0
                    && sy < eh as f32 - 1.0;
            }
        }
        // Fused: cropped coords with tap values remapped to the true
        // neighboring page pixels (mirrors the pipeline's F-B tight branch).
        let (fw, fh) = framed_dims(w, h, "tight", [0; 4], cut);
        assert_eq!((fw, fh), (ew, eh));
        let (cl, ct) = (cut[0] as i64, cut[1] as i64);
        const PAPER_BG: [f32; 3] = [246.0, 243.0, 236.0];
        let mut got = vec![0.0f32; (w2 * h2 * 3) as usize];
        for y in 0..h2 {
            for x in 0..w2 {
                let dx = x as f32 - cx1;
                let dy = y as f32 - cy1;
                let fx = cx0 + dx * cst + dy * snt;
                let fy = cy0 - dx * snt + dy * cst;
                let p = if fx >= 0.0 && fx < fw as f32 && fy >= 0.0 && fy < fh as f32 {
                    sample_remapped(fx, fy, |ix, iy| {
                        let jx = ix + cl;
                        let jy = iy + ct;
                        if jx < 0 || jy < 0 || jx >= w as i64 || jy >= h as i64 {
                            PAPER_BG
                        } else {
                            let j = ((jy as u32 * w + jx as u32) * 3) as usize;
                            [buf[j], buf[j + 1], buf[j + 2]]
                        }
                    })
                } else {
                    DESK
                };
                let i = ((y * w2 + x) * 3) as usize;
                got[i..i + 3].copy_from_slice(&p);
            }
        }
        for (k, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
            if interior[k / 3] {
                assert_eq!(g, e, "interior pixel {k} must be bit-exact");
            }
        }
        // And the bulk agrees (rim-only difference by construction; the
        // interior bit-exactness above is the load-bearing assertion).
        let same = got.iter().zip(expect.iter()).filter(|(g, e)| g == e).count();
        assert!(
            same as f64 / got.len() as f64 > 0.8,
            "tight fused must agree on >80% of values, got {same}/{}",
            got.len()
        );
    }

    #[test]
    fn framing_never_cuts_notation() {
        // End-to-end notation-safety gate for the framing stage: across many
        // seeds, every recorded tight cut must stay inside the recorded
        // quiet margin minus the recorded safety buffer, and both framing
        // modes must occur. (Crop correctness â€” kept pixels are exactly the
        // shifted page â€” is proven by
        // `frame_cut_clamped_to_quiet_and_keeps_notation`.)
        let dir = test_dir("frame");
        let input = tiny_input(&dir, "in.png"); // black bars x 8..56, y 20..21
        let mut seen = std::collections::HashSet::new();
        for seed in 0..40u64 {
            let (_, p) = degrade_file(&input, seed, 0.9, Some(80), 0, 0, 1.0, None, &StageToggles::default()).unwrap();
            seen.insert(p.frame_mode.clone());
            assert!(
                p.frame_mode == "inset" || p.frame_mode == "tight",
                "unexpected frame mode {}",
                p.frame_mode
            );
            if p.frame_mode == "inset" {
                assert_eq!(p.frame_cut_px, [0, 0, 0, 0], "inset must not cut");
            } else {
                assert_eq!(p.frame_pad_px, [0, 0, 0, 0], "tight must not pad");
            }
            // The hard law, checked against the run's own record: no cut
            // may reach closer than `frame_safe_px` to the notation bbox.
            for i in 0..4 {
                assert!(
                    p.frame_cut_px[i] + p.frame_safe_px <= p.frame_quiet_px[i],
                    "seed {seed} edge {i}: cut {} + safe {} > quiet {}",
                    p.frame_cut_px[i],
                    p.frame_safe_px,
                    p.frame_quiet_px[i]
                );
            }
            // The synthetic page has real ink, so the bbox must be found.
            assert!(p.frame_notation_bbox[2] > p.frame_notation_bbox[0]);
            assert!(p.frame_notation_bbox[3] > p.frame_notation_bbox[1]);
            assert!(p.frame_safe_px >= 3);
        }
        assert_eq!(seen.len(), 2, "expected both framing modes, got {seen:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn train_width_downscales_and_records_rescale() {
        // Train-resolution path: params record the rescale story, outputs
        // are deterministic, and every absolute-pixel knob fits its k-scaled
        // range (k = 1024/1500 here).
        let dir = test_dir("trainres");
        let input = staffed_input(&dir, "in.png", 1500, 2100);
        let t = StageToggles::default();
        let (a_img, a_p) = degrade_file(&input, 42, 0.6, Some(80), 0, 1024, 1.0, None, &t).unwrap();
        let (b_img, b_p) = degrade_file(&input, 42, 0.6, Some(80), 0, 1024, 1.0, None, &t).unwrap();
        assert_eq!(a_img.as_raw(), b_img.as_raw(), "train path must be bit-identical across reruns");
        assert_eq!(
            serde_json::to_string(&a_p).unwrap(),
            serde_json::to_string(&b_p).unwrap()
        );
        assert_eq!(a_p.train_width, 1024);
        assert_eq!((a_p.src_width, a_p.src_height), (1500, 2100));
        let k = 1024.0f32 / 1500.0;
        assert!((a_p.rescale_k - k).abs() < 1e-6, "k = {}, got {}", k, a_p.rescale_k);
        assert_eq!((a_img.width(), a_img.height()), (a_p.width, a_p.height));
        assert!(a_img.width() <= 1500, "train output must be smaller than native");
        // Scaled ranges (strength 0.6 folded in where the knob carries it).
        assert!(a_p.paper_grain_sigma <= 3.0 * k + 1e-4, "grain {}", a_p.paper_grain_sigma);
        assert!(a_p.noise_sigma <= 14.0 * 0.6 * k + 1e-4, "noise {}", a_p.noise_sigma);
        assert!(a_p.blur_sigma <= 2.4 * 0.6 * k + 1e-4, "blur {}", a_p.blur_sigma);
        assert!(a_p.ca_px <= 1.5 * 0.6 * k + 1e-4, "ca {}", a_p.ca_px);
        assert!(a_p.ink.dropout_radius <= 2.2 * k + 1e-4, "dropout {}", a_p.ink.dropout_radius);
        assert!(a_p.ink.dropout_radius >= 1.0 * k - 1e-4, "dropout {}", a_p.ink.dropout_radius);
        assert!(a_p.capture.banding_period <= (7.0 * k).round().max(3.0) as u32);
        // Legacy-off path records zeros and unity.
        let (_, off_p) = degrade_file(&input, 42, 0.6, Some(80), 0, 0, 1.0, None, &t).unwrap();
        assert_eq!(off_p.train_width, 0);
        assert!((off_p.rescale_k - 1.0).abs() < 1e-9);
        assert_eq!((off_p.src_width, off_p.src_height), (1500, 2100));
        // Narrower-than-train input is a no-op rescale (k stays 1.0).
        let narrow = staffed_input(&dir, "narrow.png", 400, 560);
        let (_, n_p) = degrade_file(&narrow, 42, 0.6, Some(80), 0, 1024, 1.0, None, &t).unwrap();
        assert!((n_p.rescale_k - 1.0).abs() < 1e-9);
        assert_eq!((n_p.src_width, n_p.src_height), (400, 560));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn framing_never_cuts_notation_at_train_res() {
        // Geometry gate re-derived at training resolution (k = 0.4): the
        // cut+safe<=quiet law must hold on every seed, the 3 px floor must
        // bind (0.002*512 rounds to 1), and real ink must still be found by
        // the bbox at the smaller size.
        let dir = test_dir("trainframe");
        let input = staffed_input(&dir, "in.png", 1280, 1800);
        let mut seen = std::collections::HashSet::new();
        for seed in 0..8u64 {
            let (_, p) = degrade_file(&input, seed, 0.9, Some(80), 0, 512, 1.0, None, &StageToggles::default()).unwrap();
            seen.insert(p.frame_mode.clone());
            assert_eq!(p.train_width, 512);
            assert!((p.rescale_k - 0.4).abs() < 1e-6, "k = {}", p.rescale_k);
            assert_eq!(p.frame_safe_px, 3, "3 px floor must bind at 512 wide");
            for i in 0..4 {
                // True no-cut guarantee: the clamp keeps cut <= quiet on
                // every edge (cut = 0 wherever fewer than `safe` px of
                // quiet exist), so dark pixels are never cropped. The
                // cut+safe<=quiet form additionally holds on edges with
                // room to spare.
                assert!(
                    p.frame_cut_px[i] <= p.frame_quiet_px[i],
                    "seed {seed} edge {i}: cut {} > quiet {}",
                    p.frame_cut_px[i],
                    p.frame_quiet_px[i]
                );
                if p.frame_cut_px[i] > 0 {
                    assert!(
                        p.frame_cut_px[i] + p.frame_safe_px <= p.frame_quiet_px[i],
                        "seed {seed} edge {i}: cut {} + safe {} > quiet {}",
                        p.frame_cut_px[i],
                        p.frame_safe_px,
                        p.frame_quiet_px[i]
                    );
                }
            }
            assert!(
                p.frame_notation_bbox[2] > p.frame_notation_bbox[0]
                    && p.frame_notation_bbox[3] > p.frame_notation_bbox[1],
                "seed {seed}: notation bbox must be found at train res"
            );
        }
        assert_eq!(seen.len(), 2, "expected both framing modes, got {seen:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bend_modes_vary_with_seed() {
        let dir = test_dir("modes");
        let input = tiny_input(&dir, "in.png");
        let mut seen = std::collections::HashSet::new();
        for seed in 0..24u64 {
            let (_, p) = degrade_file(&input, seed, 0.9, Some(80), 0, 0, 1.0, None, &StageToggles::default()).unwrap();
            seen.insert(p.curl_side.clone());
        }
        assert!(seen.len() >= 2, "expected several bend modes, got {seen:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gpu_backend_labels_itself_and_stays_deterministic() {
        // Skips cleanly on machines without a GPU adapter.
        let gpu = match gpu_backend::Gpu::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        let dir = test_dir("gpubackend");
        let input = tiny_input(&dir, "in.png");
        let (cpu_img, cpu_p) = degrade_file(&input, 4242, 0.6, Some(80), 0, 0, 1.0, None, &StageToggles::default()).unwrap();
        let (gpu_img, gpu_p) = degrade_file(&input, 4242, 0.6, Some(80), 0, 0, 1.0, Some(&gpu), &StageToggles::default()).unwrap();
        assert_eq!(cpu_p.backend, "cpu");
        assert_eq!(gpu_p.backend, "gpu");
        // Same-seed cross-backend knob equality ends at stage 1: GPU paper
        // grain (like GPU sensor noise) draws from a hash stream instead of
        // the shared ChaCha stream, so every downstream knob legitimately
        // differs â€” deterministic per backend, not across backends. Both
        // runs must still sample within their legal ranges.
        for (tag, p) in [("cpu", &cpu_p), ("gpu", &gpu_p)] {
            assert!(
                p.rotation_deg.abs() <= 10.0 + 1e-4,
                "{tag} rotation out of range: {}",
                p.rotation_deg
            );
            assert!(p.warp_out_px >= 0.0, "{tag} warp_out_px {}", p.warp_out_px);
            let maxd = cpu_img.width().max(cpu_img.height()) as f32;
            let bound = 0.08 * maxd + 8.0;
            for (i, o) in p.corner_offsets.iter().enumerate() {
                assert!(o[0].abs() <= bound, "{tag} corner {i} dx {}", o[0]);
                assert!(o[1].abs() <= bound, "{tag} corner {i} dy {}", o[1]);
            }
        }
        // Per-backend dims law: knob streams diverge at stage 1 by design,
        // so framing/rotation (and hence dims) legitimately differ across
        // backends even when neither downscaled. Each backend's output must
        // match the dims recomputed from its own records instead.
        for (tag, img, p) in [("cpu", &cpu_img, &cpu_p), ("gpu", &gpu_img, &gpu_p)] {
            assert_eq!(
                (img.width(), img.height()),
                expected_test_dims(p),
                "{tag}: dims must follow the recorded framing/rotation/downscale law"
            );
        }
        // GPU run is deterministic for the same seed.
        let (gpu_img2, _) = degrade_file(&input, 4242, 0.6, Some(80), 0, 0, 1.0, Some(&gpu), &StageToggles::default()).unwrap();
        assert_eq!(gpu_img.as_raw(), gpu_img2.as_raw(), "GPU run must be deterministic");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Recompute final dims from a run's own records: framed dims, the
    /// rotation-gate canvas expansion, then the truncating downscale.
    /// Mirrors both backends exactly (the F-B CPU loop and the 6b GPU
    /// affine dispatch share the canvas/downscale formulas), so each
    /// backend's output must match its own records.
    fn expected_test_dims(p: &Params) -> (u32, u32) {
        let (mut w, mut h) = framed_dims(64, 64, &p.frame_mode, p.frame_pad_px, p.frame_cut_px);
        // Rotation gate as in the pipeline (pre-framing dims are the 64x64
        // input here, always > 4).
        if p.rotation_deg.abs() > 0.05 {
            let th = p.rotation_deg.to_radians();
            let (sn, cs) = (th.sin().abs(), th.cos().abs());
            let (w2, h2) = (
                ((w as f32) * cs + (h as f32) * sn).ceil().max(1.0) as u32,
                ((w as f32) * sn + (h as f32) * cs).ceil().max(1.0) as u32,
            );
            w = w2;
            h = h2;
        }
        if p.scale_x < 1.0 || p.scale_y < 1.0 {
            w = ((w as f32 * p.scale_x) as u32).max(1);
            h = ((h as f32 * p.scale_y) as u32).max(1);
        }
        (w, h)
    }

    #[test]
    fn chain_path_stays_within_dims_and_cut_law() {
        // End-to-end contract for the resident-chain `--gpu` path: across
        // seeds, each backend's dims must follow its own records, the
        // recorded framing cut law must hold, and reruns must be identical.
        // Cross-backend exact dims agreement is NOT contracted: knob
        // streams diverge at stage 1 by design (CPU ChaCha vs GPU hash),
        // so framing pads and rotations legitimately differ (e.g. seed 1:
        // CPU 69x68 vs GPU 74x74, both arithmetically correct from their
        // own knobs). Skips cleanly on machines without a GPU adapter.
        let gpu = match gpu_backend::Gpu::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        let dir = test_dir("chainpath");
        let input = tiny_input(&dir, "in.png");
        for seed in [0u64, 1, 7, 42, 4242, 99991] {
            let (cpu_img, cpu_p) =
                degrade_file(&input, seed, 0.6, Some(80), 0, 0, 1.0, None, &StageToggles::default()).unwrap();
            let (gpu_img, gpu_p) =
                degrade_file(&input, seed, 0.6, Some(80), 0, 0, 1.0, Some(&gpu), &StageToggles::default()).unwrap();
            assert_eq!(gpu_p.backend, "gpu", "chain path must label itself");
            // Per-backend dims law (replaces the old cross-backend
            // exact-dims rule): output dims must equal the dims recomputed
            // from that backend's own framing/rotation/scale records.
            for (tag, img, p) in [("cpu", &cpu_img, &cpu_p), ("gpu", &gpu_img, &gpu_p)] {
                assert_eq!(
                    (img.width(), img.height()),
                    expected_test_dims(p),
                    "{tag} seed {seed}: dims must follow the recorded framing/rotation/downscale law"
                );
            }
            // Framing cut law from the run's own record.
            for i in 0..4 {
                assert!(
                    gpu_p.frame_cut_px[i] + gpu_p.frame_safe_px <= gpu_p.frame_quiet_px[i],
                    "seed {seed} edge {i}: cut {} + safe {} > quiet {}",
                    gpu_p.frame_cut_px[i],
                    gpu_p.frame_safe_px,
                    gpu_p.frame_quiet_px[i]
                );
            }
            // Determinism across reruns.
            let (gpu_img2, _) =
                degrade_file(&input, seed, 0.6, Some(80), 0, 0, 1.0, Some(&gpu), &StageToggles::default()).unwrap();
            assert_eq!(gpu_img.as_raw(), gpu_img2.as_raw(), "seed {seed}: GPU run must be deterministic");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gpu_backend_records_backend_field() {
        let dir = test_dir("backendfield");
        let input = tiny_input(&dir, "in.png");
        let (_, p) = degrade_file(&input, 7, 0.6, Some(80), 0, 0, 1.0, None, &StageToggles::default()).unwrap();
        assert_eq!(p.backend, "cpu", "CPU path must label itself");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reduced_preset_is_deterministic_and_records_skips() {
        let dir = test_dir("reduced");
        let input = tiny_input(&dir, "in.png");
        let toggles = StageToggles {
            skip_grain: true,
            skip_chromatic: true,
            skip_noise: true,
            coarsen_ink: true,
            coarsen_aging: true,
            fast_banks: false,
        };
        assert_eq!(toggles.preset_name(), "reduced");
        let (a_img, a_p) =
            degrade_file(&input, 42, 0.6, Some(80), 0, 0, 1.0, None, &toggles).unwrap();
        let (b_img, b_p) =
            degrade_file(&input, 42, 0.6, Some(80), 0, 0, 1.0, None, &toggles).unwrap();
        assert_eq!(a_img.as_raw(), b_img.as_raw(), "reduced must be bit-identical across reruns");
        assert_eq!(
            serde_json::to_string(&a_p).unwrap(),
            serde_json::to_string(&b_p).unwrap()
        );
        assert_eq!(a_p.preset, "reduced");
        // Skipped stages record neutral params.
        assert_eq!(a_p.paper_grain_sigma, 0.0);
        assert_eq!(a_p.noise_sigma, 0.0);
        assert_eq!(a_p.ca_px, 0.0);
        assert_eq!(a_p.ink.fade_amount, 0.0);
        assert_eq!(a_p.ink.ghost_alpha, 0.0);
        assert_eq!(a_p.paper.yellowing, 0.0);
        assert!(!a_p.paper.water_present);
        // The surviving sub-effects still run under coarsening.
        let full =
            degrade_file(&input, 42, 0.6, Some(80), 0, 0, 1.0, None, &StageToggles::default())
                .unwrap()
                .0;
        assert_ne!(a_img.as_raw(), full.as_raw(), "reduced must differ from full");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn partial_skips_label_preset_custom() {
        let t = StageToggles { skip_noise: true, ..StageToggles::default() };
        assert_eq!(t.preset_name(), "custom");
        assert_eq!(StageToggles::default().preset_name(), "full");
    }

    #[test]
    fn fast_banks_is_deterministic_and_labels_itself() {
        // Subtask-5 pipeline gate: the banked path records itself, stays
        // bit-identical across reruns, keeps every knob in its legal range,
        // and leaves the legacy default path untouched (default off).
        let dir = test_dir("fastbanks");
        let input = tiny_input(&dir, "in.png");
        assert!(!StageToggles::default().fast_banks, "legacy default must stay bank-free");
        let banked = StageToggles { fast_banks: true, ..StageToggles::default() };
        assert_eq!(banked.preset_name(), "full", "banks must not change the preset");
        let (a_img, a_p) =
            degrade_file(&input, 42, 0.6, Some(80), 0, 0, 1.0, None, &banked).unwrap();
        let (b_img, b_p) =
            degrade_file(&input, 42, 0.6, Some(80), 0, 0, 1.0, None, &banked).unwrap();
        assert_eq!(a_img.as_raw(), b_img.as_raw(), "banked path must be bit-identical across reruns");
        assert_eq!(
            serde_json::to_string(&a_p).unwrap(),
            serde_json::to_string(&b_p).unwrap()
        );
        assert!(a_p.fast_banks, "params must record the banked path");
        assert_eq!(a_p.preset, "full");
        // Grain knobs are sampled before any banked apply, so they match the
        // legacy run exactly. Everything downstream of the first banked stage
        // legitimately diverges (banked stages consume 2 offset draws where
        // the direct loop consumed millions) — same precedent as the GPU
        // hash-stream backend, whose knob equality also ends at stage 1.
        let (_, legacy_p) =
            degrade_file(&input, 42, 0.6, Some(80), 0, 0, 1.0, None, &StageToggles::default()).unwrap();
        assert!(!legacy_p.fast_banks);
        assert!((a_p.paper_grain_sigma - legacy_p.paper_grain_sigma).abs() < 1e-6);
        assert!((a_p.paper_speckle_prob - legacy_p.paper_speckle_prob).abs() < 1e-6);
        assert!(a_img.width() > 0 && a_img.height() > 0);
        // Framing cut law holds on the banked path too.
        for i in 0..4 {
            assert!(
                a_p.frame_cut_px[i] + a_p.frame_safe_px <= a_p.frame_quiet_px[i],
                "edge {i}: cut {} + safe {} > quiet {}",
                a_p.frame_cut_px[i],
                a_p.frame_safe_px,
                a_p.frame_quiet_px[i]
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_artifact_leaves_no_tmp() {
        let dir = test_dir("artifact");
        let target = dir.join("out.txt");
        write_artifact(&target, |p| {
            std::fs::write(p, b"hello").map_err(|e| e.to_string())
        })
        .unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"hello");
        assert!(!dir.join("out.txt.tmp").exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// Gather input PNGs: a single file, or all `*.png` in a directory.
fn collect_inputs(input: &Path) -> Result<Vec<PathBuf>, String> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    if input.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(input)
            .map_err(|e| format!("cannot list {}: {e}", input.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| e.eq_ignore_ascii_case("png"))
                        .unwrap_or(false)
            })
            .collect();
        files.sort();
        return Ok(files);
    }
    Err(format!("input not found: {}", input.display()))
}
