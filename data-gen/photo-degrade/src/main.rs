//! `photo-degrade` CLI: thin delegator over the published `doc-photo-degrade` crate.

use clap::Parser;
use doc_photo_degrade::{
    collect_inputs, degrade_file_gpu, degrade_image_gpu, encode_jpeg, write_artifact,
    DegradeConfig, DocumentProfile, Gpu, MaskFormat, MaskRequest, OutputFormat, Preset,
    StageToggles,
};
use image::{ImageFormat, Rgb, RgbImage};
use std::path::PathBuf;

/// Degrade clean sheet-music / document images into realistic phone photos
/// with paired JSON parameter sidecars.
#[derive(Parser, Debug)]
#[command(name = "photo-degrade", version, about)]
struct Args {
    /// Input clean PNG/JPEG file, or a directory of images (non-recursive).
    #[arg(long)]
    input: PathBuf,

    /// Output directory receiving `<stem>{suffix}.png|jpg` + `<stem>{suffix}_params.json`.
    #[arg(long, default_value = "out")]
    out_dir: PathBuf,

    /// Domain document profile: `phone-photo`, `archival-manuscript`,
    /// `thermal-receipt`, `office-scan`, `photocopy-xerox`, `book-gutter-bend`,
    /// or `sheet-music`. Explicit CLI flags override profile defaults.
    #[arg(long)]
    profile: Option<String>,

    /// Pipeline preset: `full` or `reduced`.
    #[arg(long)]
    preset: Option<String>,

    /// Master seed. For directory input, file `i` uses
    /// `seed + i * 0x9E3779B97F4A7C15` (wrapping multiply stride).
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Global effect strength in `[0, 1]`. Scales every effect (default: 0.5, or profile default).
    #[arg(long)]
    strength: Option<f32>,

    /// JPEG quality override (`1..=100`). If omitted, sampled per image (or profile default).
    #[arg(long)]
    jpeg_quality: Option<u8>,

    /// Downscale inputs wider than this to this width (aspect kept, `0` = keep native).
    #[arg(long, default_value_t = 2480)]
    max_width: u32,

    /// Degrade at model-training resolution: downscale post-maxwidth input to
    /// this width before any degrade stage and rescale absolute-pixel stats by
    /// `k = train_width / src_width`. `0` = off (full-res path).
    #[arg(long, default_value_t = 0)]
    train_width: u32,

    /// Random output downscale per image, sampled uniformly in `[scale-min, 1.0]`
    /// (`1.0` = keep; default: 0.5, or profile default).
    #[arg(long)]
    scale_min: Option<f32>,

    /// Filename suffix for degraded images (before the extension).
    #[arg(long, default_value = "_degraded")]
    suffix: String,

    /// Output image format: `png` (lossless) or `jpeg`.
    #[arg(long, default_value = "png")]
    format: String,

    /// Require the WGPU GPU backend (fail if no usable GPU adapter is found).
    #[arg(long, default_value_t = false, conflicts_with = "cpu")]
    gpu: bool,

    /// Force the CPU backend for all stages (disables auto GPU acceleration).
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Skip the fine paper-grain stage.
    #[arg(long, default_value_t = false)]
    skip_grain: bool,

    /// Skip the non-linear page-bend stage (enables exact projective/affine
    /// bounding-box reprojection via `reproject`).
    #[arg(long, default_value_t = false)]
    skip_bend: bool,

    /// Skip chromatic aberration (stream-preserving).
    #[arg(long, default_value_t = false)]
    skip_chromatic: bool,

    /// Skip the sensor-noise stage.
    #[arg(long, default_value_t = false)]
    skip_noise: bool,

    /// Coarsen ink degradation: keep dropout + broken strokes, skip smooth fade/toner/ghost field.
    #[arg(long, default_value_t = false)]
    coarsen_ink: bool,

    /// Coarsen paper aging: keep foxing spots + fibers, skip full-frame tint/mottle/water loop.
    #[arg(long, default_value_t = false)]
    coarsen_aging: bool,

    /// Reduced pipeline preset (`--skip-grain --skip-chromatic --skip-noise --coarsen-ink --coarsen-aging`).
    #[arg(long, default_value_t = false)]
    reduced: bool,

    /// Precomputed fast path: seeded tiled Gaussian banks + precomputed lighting field.
    #[arg(long, default_value_t = false)]
    fast_banks: bool,

    /// Export multi-task ground-truth masks alongside each degraded image
    /// (`uv_flow`, `illumination`, `clean_ink`, `artifacts`).
    #[arg(long, default_value_t = false)]
    export_masks: bool,

    /// Format for exported floating-point masks: `npy-png` (default) or `png16`.
    #[arg(long, default_value = "npy-png")]
    mask_format: String,
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(&args) {
        eprintln!("photo-degrade: error: {e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), String> {
    let fmt = match args.format.to_ascii_lowercase().as_str() {
        "png" => OutputFormat::Png,
        "jpeg" | "jpg" => OutputFormat::Jpeg,
        other => return Err(format!("--format must be png|jpeg, got `{other}`")),
    };
    let ext = match fmt {
        OutputFormat::Jpeg => "jpg",
        OutputFormat::Png => "png",
    };

    let mut base_cfg = DegradeConfig {
        seed: args.seed,
        max_width: args.max_width,
        train_width: args.train_width,
        ..DegradeConfig::default()
    };

    let selected_profile = if let Some(ref name) = args.profile {
        let p = DocumentProfile::from_str_name(name).ok_or_else(|| {
            let valid = DocumentProfile::ALL
                .iter()
                .map(|p| p.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            format!("--profile must be one of [{valid}], got `{name}`")
        })?;
        p.apply_to_config(&mut base_cfg);
        Some(p)
    } else {
        None
    };

    if let Some(ref preset_name) = args.preset {
        let preset = match preset_name.trim().to_ascii_lowercase().as_str() {
            "full" => Preset::Full,
            "reduced" => Preset::Reduced,
            other => return Err(format!("--preset must be full|reduced, got `{other}`")),
        };
        let fb = base_cfg.toggles.fast_banks;
        base_cfg = base_cfg.with_preset(preset);
        base_cfg.toggles.fast_banks = fb;
    }

    if let Some(s) = args.strength {
        base_cfg.strength = s.clamp(0.0, 1.0);
    }
    if let Some(sm) = args.scale_min {
        base_cfg.scale_min = sm;
    }
    if let Some(q) = args.jpeg_quality {
        base_cfg.jpeg_quality = Some(q);
    }
    if args.reduced {
        let fb = base_cfg.toggles.fast_banks;
        base_cfg.toggles = StageToggles::reduced();
        base_cfg.toggles.fast_banks = fb;
    }
    base_cfg.toggles.skip_grain |= args.skip_grain;
    base_cfg.toggles.skip_bend |= args.skip_bend;
    base_cfg.toggles.skip_chromatic |= args.skip_chromatic;
    base_cfg.toggles.skip_noise |= args.skip_noise;
    base_cfg.toggles.coarsen_ink |= args.coarsen_ink;
    base_cfg.toggles.coarsen_aging |= args.coarsen_aging;
    if args.fast_banks {
        base_cfg.toggles.fast_banks = true;
    }

    let mask_fmt = MaskFormat::from_str_name(&args.mask_format).ok_or_else(|| {
        format!(
            "--mask-format must be npy-png|png16, got `{}`",
            args.mask_format
        )
    })?;
    if args.export_masks {
        base_cfg.masks = MaskRequest::all();
    }
    base_cfg.validate()?;

    let inputs = collect_inputs(&args.input)?;
    if inputs.is_empty() {
        return Err(format!(
            "no PNG/JPEG files found at {}",
            args.input.display()
        ));
    }
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|e| format!("cannot create {}: {e}", args.out_dir.display()))?;

    let gpu = init_gpu(args)?;

    if let Some(p) = selected_profile {
        println!("document profile: {}", p.as_str());
    }
    if base_cfg.toggles.preset_name() != "full" {
        println!("pipeline preset: {}", base_cfg.toggles.preset_name());
    }

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1)
        .min(inputs.len().max(1));
    let chunk_size = inputs.len().div_ceil(n_threads);
    let stop = std::sync::atomic::AtomicBool::new(false);
    let first_err = std::sync::Mutex::new(None::<String>);
    let print_lock = std::sync::Mutex::new(());
    let (stop_r, err_r, pl_r) = (&stop, &first_err, &print_lock);
    let base_cfg_r = &base_cfg;
    let gpu_r = gpu.as_ref();

    std::thread::scope(|scope| {
        for (ci, chunk) in inputs.chunks(chunk_size).enumerate() {
            let base = ci * chunk_size;
            scope.spawn(move || {
                for (j, path) in chunk.iter().enumerate() {
                    if stop_r.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    let i = base + j;
                    let file_seed = args
                        .seed
                        .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let mut cfg = base_cfg_r.clone();
                    cfg.seed = file_seed;

                    match degrade_file_gpu(path, &cfg, gpu_r) {
                        Ok(mut out) => {
                            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                                Some(s) => s.to_string(),
                                None => {
                                    stop_r.store(true, std::sync::atomic::Ordering::Relaxed);
                                    *err_r.lock().unwrap() =
                                        Some(format!("bad file name: {}", path.display()));
                                    break;
                                }
                            };
                            let img_name = format!("{stem}{}.{ext}", args.suffix);
                            let json_name = format!("{stem}{}_params.json", args.suffix);
                            let img_path = args.out_dir.join(&img_name);
                            let json_path = args.out_dir.join(&json_name);

                            let write_res = match fmt {
                                OutputFormat::Jpeg => {
                                    match encode_jpeg(&out.image, out.params.jpeg_quality) {
                                        Ok(buf) => write_artifact(&img_path, |p| {
                                            std::fs::write(p, &buf).map_err(|e| {
                                                format!("cannot write {}: {e}", p.display())
                                            })
                                        })
                                        .map(|()| "zune"),
                                        Err(e) => Err(e),
                                    }
                                }
                                OutputFormat::Png => write_artifact(&img_path, |p| {
                                    out.image
                                        .save_with_format(p, ImageFormat::Png)
                                        .map_err(|e| format!("cannot write {}: {e}", p.display()))
                                })
                                .map(|()| "image"),
                            };

                            match write_res {
                                Ok(tag) => out.params.codec = tag.to_string(),
                                Err(e) => {
                                    stop_r.store(true, std::sync::atomic::Ordering::Relaxed);
                                    *err_r.lock().unwrap() = Some(e);
                                    break;
                                }
                            }
                            out.params.output = img_name;
                            let json = match serde_json::to_string_pretty(&out.params)
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
                            let mask_paths = if let Some(ref masks) = out.masks {
                                let prefix = format!("{stem}{}", args.suffix);
                                match masks.write_to_dir(&args.out_dir, &prefix, mask_fmt) {
                                    Ok(paths) => paths,
                                    Err(e) => {
                                        stop_r.store(true, std::sync::atomic::Ordering::Relaxed);
                                        *err_r.lock().unwrap() = Some(e);
                                        break;
                                    }
                                }
                            } else {
                                Vec::new()
                            };
                            let _guard = pl_r.lock().unwrap();
                            println!("{} -> {}", path.display(), img_path.display());
                            println!("  params -> {}", json_path.display());
                            for mp in mask_paths {
                                println!("  mask   -> {}", mp.display());
                            }
                        }
                        Err(e) => {
                            stop_r.store(true, std::sync::atomic::Ordering::Relaxed);
                            *err_r.lock().unwrap() = Some(format!("{}: {e}", path.display()));
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

fn init_gpu(args: &Args) -> Result<Option<Gpu>, String> {
    if args.cpu {
        return Ok(None);
    }
    match Gpu::new() {
        Ok(g) => {
            println!("GPU backend enabled");
            // Warm up pipelines once so steady-state timing is clean.
            let tiny = RgbImage::from_pixel(32, 32, Rgb([250, 248, 242]));
            let mut warm_cfg = DegradeConfig::default();
            warm_cfg.seed = 1;
            warm_cfg.strength = 0.5;
            warm_cfg.scale_min = 0.9;
            let _ = degrade_image_gpu(tiny, &warm_cfg, Some(&g));
            Ok(Some(g))
        }
        Err(e) if args.gpu => Err(format!("--gpu: {e}")),
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_all_supported_flags() {
        let args = Args::try_parse_from([
            "photo-degrade",
            "--input",
            "in.png",
            "--out-dir",
            "out_dir",
            "--format",
            "jpeg",
            "--seed",
            "42",
            "--strength",
            "0.6",
            "--max-width",
            "2048",
            "--train-width",
            "1024",
            "--scale-min",
            "0.75",
            "--preset",
            "reduced",
            "--reduced",
            "--fast-banks",
            "--cpu",
            "--skip-bend",
            "--profile",
            "sheet-music",
            "--suffix",
            "_d0",
            "--export-masks",
        ])
        .expect("CLI flags should parse");

        assert_eq!(args.seed, 42);
        assert_eq!(args.format, "jpeg");
        assert_eq!(args.max_width, 2048);
        assert_eq!(args.train_width, 1024);
        assert_eq!(args.suffix, "_d0");
        assert!(args.cpu);
        assert!(args.reduced);
        assert!(args.fast_banks);
        assert!(args.skip_bend);
        assert!(args.export_masks);
    }

    #[test]
    fn facade_cpu_run_is_deterministic() {
        let tmp = std::env::temp_dir().join(format!(
            "photo-degrade-facade-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let in_path = tmp.join("clean.png");
        let mut img = RgbImage::from_pixel(64, 64, Rgb([250, 248, 242]));
        for x in 8..56 {
            img.put_pixel(x, 32, Rgb([20, 20, 20]));
        }
        img.save(&in_path).unwrap();

        let out_a = tmp.join("out_a");
        let out_b = tmp.join("out_b");
        for out_dir in [&out_a, &out_b] {
            let args = Args::try_parse_from([
                "photo-degrade",
                "--input",
                in_path.to_str().unwrap(),
                "--out-dir",
                out_dir.to_str().unwrap(),
                "--format",
                "jpeg",
                "--seed",
                "7",
                "--cpu",
                "--profile",
                "sheet-music",
                "--suffix",
                "_d0",
            ])
            .unwrap();
            run(&args).expect("run should succeed");
        }

        let bytes_a = std::fs::read(out_a.join("clean_d0.jpg")).unwrap();
        let bytes_b = std::fs::read(out_b.join("clean_d0.jpg")).unwrap();
        assert_eq!(bytes_a, bytes_b, "CPU runs must be bit-identical");
        assert!(out_a.join("clean_d0_params.json").exists());

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
