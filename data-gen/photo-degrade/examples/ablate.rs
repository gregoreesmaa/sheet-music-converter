//! Ablation probe for subtask 1: does each photo-degrade effect survive
//! downscale -> luma -> Otsu (the converter/ingest-cleanup binarization)?
//!
//! Loads one real page, takes a center crop, applies each effect in
//! isolation (pipeline-default magnitudes, fixed seed), binarizes clean vs
//! perturbed with Otsu, and reports flip fraction + ink F1, at native res and
//! after a 50% Triangle downscale (capture/model-res proxy).
//!
//! Run: cargo run --release --example ablate -- <page.png> [crop_px]

use image::RgbImage;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn gauss(rng: &mut ChaCha8Rng) -> f32 {
    let a: f32 = rng.gen_range(1e-6..1.0);
    let b: f32 = rng.gen_range(0.0..1.0);
    (-2.0 * a.ln()).sqrt() * (2.0 * std::f32::consts::PI * b).cos()
}

fn load_crop(path: &str, crop: u32) -> (Vec<f32>, u32, u32) {
    let img = image::open(path).expect("open").to_rgb8();
    let (w, h) = (img.width(), img.height());
    let c = crop.min(w.min(h));
    let x0 = (w - c) / 2;
    let y0 = (h - c) / 2;
    let mut buf = vec![0.0f32; (c * c * 3) as usize];
    for y in 0..c {
        for x in 0..c {
            let p = img.get_pixel(x0 + x, y0 + y);
            let i = ((y * c + x) * 3) as usize;
            buf[i] = p[0] as f32;
            buf[i + 1] = p[1] as f32;
            buf[i + 2] = p[2] as f32;
        }
    }
    (buf, c, c)
}

fn luma(buf: &[f32]) -> Vec<f32> {
    buf.chunks_exact(3)
        .map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2])
        .collect()
}

fn otsu_threshold(l: &[f32]) -> u8 {
    let mut hist = [0u64; 256];
    for &v in l {
        hist[v.clamp(0.0, 255.0) as usize] += 1;
    }
    let total = l.len() as f64;
    let mut sum_all = 0.0;
    for (i, &h) in hist.iter().enumerate() {
        sum_all += i as f64 * h as f64;
    }
    let (mut sum_b, mut w_b, mut best, mut thresh) = (0.0, 0u64, -1.0f64, 0u8);
    for (i, &h) in hist.iter().enumerate() {
        w_b += h;
        if w_b == 0 {
            continue;
        }
        let w_f = l.len() as u64 - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += i as f64 * h as f64;
        let m_b = sum_b / w_b as f64;
        let m_f = (sum_all - sum_b) / w_f as f64;
        let between = w_b as f64 * w_f as f64 * (m_b - m_f) * (m_b - m_f);
        if between > best {
            best = between;
            thresh = i as u8;
        }
    }
    let _ = total;
    thresh
}

fn binarize(l: &[f32], t: u8) -> Vec<bool> {
    l.iter().map(|&v| v <= t as f32).collect() // true = ink
}

fn scores(base: &[bool], pert: &[bool]) -> (f64, f64) {
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut f_n = 0u64;
    let mut diff = 0u64;
    for (&b, &p) in base.iter().zip(pert.iter()) {
        if b != p {
            diff += 1;
        }
        match (b, p) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => f_n += 1,
            _ => {}
        }
    }
    let flip = diff as f64 / base.len() as f64;
    let denom = 2 * tp + fp + f_n;
    let f1 = if denom == 0 { 1.0 } else { 2.0 * tp as f64 / denom as f64 };
    (flip, f1)
}

fn downscale_half(buf: &[f32], w: u32, h: u32) -> (Vec<f32>, u32, u32) {
    let img: RgbImage = {
        let mut im = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                im.put_pixel(
                    x,
                    y,
                    image::Rgb([
                        buf[i].clamp(0.0, 255.0) as u8,
                        buf[i + 1].clamp(0.0, 255.0) as u8,
                        buf[i + 2].clamp(0.0, 255.0) as u8,
                    ]),
                );
            }
        }
        im
    };
    let rs = image::imageops::resize(&img, w / 2, h / 2, image::imageops::FilterType::Triangle);
    let (nw, nh) = (rs.width(), rs.height());
    let mut out = vec![0.0f32; (nw * nh * 3) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let p = rs.get_pixel(x, y);
            let i = ((y * nw + x) * 3) as usize;
            out[i] = p[0] as f32;
            out[i + 1] = p[1] as f32;
            out[i + 2] = p[2] as f32;
        }
    }
    (out, nw, nh)
}

fn jpeg_roundtrip(buf: &[f32], w: u32, h: u32, q: u8) -> Vec<f32> {
    use image::codecs::jpeg::JpegEncoder;
    let mut im = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            im.put_pixel(
                x,
                y,
                image::Rgb([
                    buf[i].clamp(0.0, 255.0) as u8,
                    buf[i + 1].clamp(0.0, 255.0) as u8,
                    buf[i + 2].clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    let mut bytes: Vec<u8> = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut bytes, q);
        enc.encode_image(&im).expect("encode");
    }
    let dec = image::load_from_memory_with_format(&bytes, image::ImageFormat::Jpeg)
        .expect("decode")
        .to_rgb8();
    let mut out = vec![0.0f32; buf.len()];
    for y in 0..h {
        for x in 0..w {
            let p = dec.get_pixel(x, y);
            let i = ((y * w + x) * 3) as usize;
            out[i] = p[0] as f32;
            out[i + 1] = p[1] as f32;
            out[i + 2] = p[2] as f32;
        }
    }
    out
}

type Effect = (&'static str, Box<dyn Fn(&[f32], u32, u32, &mut ChaCha8Rng) -> Vec<f32>>);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: ablate <page.png> [crop_px]");
        std::process::exit(1);
    }
    let crop: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1200);
    let (clean, w, h) = load_crop(&args[1], crop);
    println!("input: {} crop {w}x{h}", args[1]);

    let effects: Vec<Effect> = vec![
        ("grain_s2", Box::new(|b, _, _, r| {
            let mut o = b.to_vec();
            for v in o.iter_mut() {
                *v += gauss(r) * 2.0;
            }
            o
        })),
        ("grain_speckle", Box::new(|b, _, _, r| {
            let mut o = b.to_vec();
            for px in o.chunks_exact_mut(3) {
                for c in 0..3 {
                    px[c] += gauss(r) * 2.0;
                }
                if r.gen::<f32>() < 0.001 {
                    for c in 0..3 {
                        px[c] -= 22.0 * 0.5;
                    }
                }
            }
            o
        })),
        ("noise_s6", Box::new(|b, _, _, r| {
            let mut o = b.to_vec();
            for v in o.iter_mut() {
                *v += gauss(r) * 6.0;
            }
            o
        })),
        ("noise_s10", Box::new(|b, _, _, r| {
            let mut o = b.to_vec();
            for v in o.iter_mut() {
                *v += gauss(r) * 10.0;
            }
            o
        })),
        ("chromatic_1px", Box::new(|b, w, h, _| {
            let mut o = b.to_vec();
            for y in 0..h {
                for x in 0..w {
                    let i = ((y * w + x) * 3) as usize;
                    let xr = x.saturating_add(1).min(w - 1) as usize;
                    let xb = x.saturating_sub(1) as usize;
                    o[i] = b[((y * w + xr as u32) * 3) as usize];
                    o[i + 2] = b[((y * w + xb as u32) * 3 + 2) as usize];
                }
            }
            o
        })),
        ("light_gentle", Box::new(|b, w, h, _| {
            let mut o = b.to_vec();
            for y in 0..h {
                for x in 0..w {
                    let i = ((y * w + x) * 3) as usize;
                    let m = 1.0 - 0.08 * (y as f32 / h as f32);
                    for c in 0..3 {
                        o[i + c] *= m;
                    }
                }
            }
            o
        })),
        ("light_mid", Box::new(|b, w, h, _| {
            let mut o = b.to_vec();
            let (gx, gy) = (w as f32 * 0.3, h as f32 * 0.3);
            let rad = w.min(h) as f32 * 0.35;
            for y in 0..h {
                for x in 0..w {
                    let i = ((y * w + x) * 3) as usize;
                    let m = 1.0 - 0.15 * (y as f32 / h as f32);
                    let gd = ((x as f32 - gx).powi(2) + (y as f32 - gy).powi(2)).sqrt();
                    let g = (1.0 - gd / rad).max(0.0);
                    let add = 0.3 * 255.0 * g * g;
                    for c in 0..3 {
                        o[i + c] = o[i + c] * m + add;
                    }
                }
            }
            o
        })),
        ("blur_s0.7", Box::new(|b, w, h, _| {
            let mut im = RgbImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let i = ((y * w + x) * 3) as usize;
                    im.put_pixel(
                        x,
                        y,
                        image::Rgb([
                            b[i].clamp(0.0, 255.0) as u8,
                            b[i + 1].clamp(0.0, 255.0) as u8,
                            b[i + 2].clamp(0.0, 255.0) as u8,
                        ]),
                    );
                }
            }
            let bl = image::imageops::blur(&im, 0.7);
            let mut o = vec![0.0f32; b.len()];
            for y in 0..h {
                for x in 0..w {
                    let p = bl.get_pixel(x, y);
                    let i = ((y * w + x) * 3) as usize;
                    o[i] = p[0] as f32;
                    o[i + 1] = p[1] as f32;
                    o[i + 2] = p[2] as f32;
                }
            }
            o
        })),
        ("blur_s1.5", Box::new(|b, w, h, _| {
            let mut im = RgbImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let i = ((y * w + x) * 3) as usize;
                    im.put_pixel(
                        x,
                        y,
                        image::Rgb([
                            b[i].clamp(0.0, 255.0) as u8,
                            b[i + 1].clamp(0.0, 255.0) as u8,
                            b[i + 2].clamp(0.0, 255.0) as u8,
                        ]),
                    );
                }
            }
            let bl = image::imageops::blur(&im, 1.5);
            let mut o = vec![0.0f32; b.len()];
            for y in 0..h {
                for x in 0..w {
                    let p = bl.get_pixel(x, y);
                    let i = ((y * w + x) * 3) as usize;
                    o[i] = p[0] as f32;
                    o[i + 1] = p[1] as f32;
                    o[i + 2] = p[2] as f32;
                }
            }
            o
        })),
        ("mottle_8", Box::new(|b, w, h, _| {
            let mut o = b.to_vec();
            for y in 0..h {
                for x in 0..w {
                    let i = ((y * w + x) * 3) as usize;
                    let m = 8.0
                        * (x as f32 / 300.0).sin()
                        * (y as f32 / 260.0).cos();
                    o[i] += m + 5.0;
                    o[i + 1] += m + 3.0;
                    o[i + 2] += m - 1.0;
                }
            }
            o
        })),
        ("jpeg_q65", Box::new(|b, w, h, _| jpeg_roundtrip(b, w, h, 65))),
        ("ink_fade40", Box::new(|b, _, _, _| {
            b.chunks_exact(3)
                .flat_map(|p| {
                    [0, 1, 2].map(|c| {
                        let dark = (241.0 - p[c]).max(0.0);
                        241.0 - dark * 0.6
                    })
                })
                .collect()
        })),
        ("shift_6px", Box::new(|b, w, h, _| {
            let mut o = vec![255.0f32; b.len()];
            for y in 0..h {
                for x in 0..w {
                    if x >= 6 {
                        let s = ((y * w + x - 6) * 3) as usize;
                        let d = ((y * w + x) * 3) as usize;
                        o[d] = b[s];
                        o[d + 1] = b[s + 1];
                        o[d + 2] = b[s + 2];
                    }
                }
            }
            o
        })),
    ];

    let clean_luma = luma(&clean);
    let ct = otsu_threshold(&clean_luma);
    let clean_bin = binarize(&clean_luma, ct);
    let ink_frac = clean_bin.iter().filter(|&&b| b).count() as f64 / clean_bin.len() as f64;
    println!("clean otsu={ct} ink_frac={ink_frac:.4}");

    let (half, hw, hh) = downscale_half(&clean, w, h);
    let half_luma = luma(&half);
    let ht = otsu_threshold(&half_luma);
    let half_bin = binarize(&half_luma, ht);
    println!("clean-half otsu={ht}");

    println!(
        "{:<14} {:>10} {:>10} {:>10} {:>10}",
        "effect", "flip_nat%", "F1_nat", "flip_hlf%", "F1_hlf"
    );
    for (name, f) in &effects {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let pert = f(&clean, w, h, &mut rng);
        let pl = luma(&pert);
        let t = otsu_threshold(&pl);
        let pb = binarize(&pl, t);
        let (flip, f1) = scores(&clean_bin, &pb);
        let (hp, nhw, nhh) = downscale_half(&pert, w, h);
        assert_eq!((nhw, nhh), (hw, hh));
        let hl = luma(&hp);
        let htt = otsu_threshold(&hl);
        let hb = binarize(&hl, htt);
        let (hflip, hf1) = scores(&half_bin, &hb);
        println!(
            "{:<14} {:>10.3} {:>10.4} {:>10.3} {:>10.4}  (t={t}/{htt})",
            name,
            flip * 100.0,
            f1,
            hflip * 100.0,
            hf1
        );
    }
}
