//! Downstream-parity probe for subtask 1: compares binarized (luma + Otsu,
//! like `converter/ingest-cleanup`) image pairs on a fixed probe set.
//!
//! Modes:
//!   cmpbin pair <a.jpg> <b.jpg>            single pair: flip frac + ink F1
//!   cmpbin dir <dirA> <dirB>               all shared stems in two dirs
//!   cmpbin selfgn <img>                    image vs itself + synthetic
//!                                          grain (σ2) + noise (σ6) at
//!                                          pipeline-default magnitudes
//!
//! Exit 0 always (metrics on stdout); non-zero on I/O errors.

use image::RgbImage;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::path::Path;

fn gauss(rng: &mut ChaCha8Rng) -> f32 {
    let a: f32 = rng.gen_range(1e-6..1.0);
    let b: f32 = rng.gen_range(0.0..1.0);
    (-2.0 * a.ln()).sqrt() * (2.0 * std::f32::consts::TAU * b).cos()
}

fn load_rgb(path: &Path) -> RgbImage {
    image::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display())).to_rgb8()
}

fn luma(img: &RgbImage) -> Vec<f32> {
    img.pixels().map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).collect()
}

fn otsu_threshold(l: &[f32]) -> u8 {
    let mut hist = [0u64; 256];
    for &v in l {
        hist[v.clamp(0.0, 255.0) as usize] += 1;
    }
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
    thresh
}

fn binarize(l: &[f32], t: u8) -> Vec<bool> {
    l.iter().map(|&v| v <= t as f32).collect()
}

fn scores(base: &[bool], pert: &[bool]) -> (f64, f64, f64, f64) {
    let (mut tp, mut fp, mut f_n, mut diff) = (0u64, 0u64, 0u64, 0u64);
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
    let rec = if tp + f_n == 0 { 1.0 } else { tp as f64 / (tp + f_n) as f64 };
    let prec = if tp + fp == 0 { 1.0 } else { tp as f64 / (tp + fp) as f64 };
    (flip, f1, rec, prec)
}

fn pair(mask_a: &[bool], ta: u8, img_b: &RgbImage, stem: &str) {
    let lb = luma(img_b);
    let tb = otsu_threshold(&lb);
    let mb = binarize(&lb, tb);
    assert_eq!(mask_a.len(), mb.len(), "{stem}: dim mismatch");
    let (flip, f1, rec, prec) = scores(mask_a, &mb);
    println!("{stem:<28} tA={ta:<4} tB={tb:<4} flip%={:>7.3} F1={:.4} R={:.4} P={:.4}",
        flip * 100.0, f1, rec, prec);
}

fn stems(dir: &Path, ext: &str) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("readdir {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some(ext))
        .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).map(|s| s.to_string()))
        .collect();
    v.sort();
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("pair") => {
            let (a, b) = (Path::new(&args[2]), Path::new(&args[3]));
            let ia = load_rgb(a);
            let la = luma(&ia);
            let ta = otsu_threshold(&la);
            let ma = binarize(&la, ta);
            let ib = load_rgb(b);
            pair(&ma, ta, &ib, "pair");
        }
        Some("dir") => {
            let (da, db) = (Path::new(&args[2]), Path::new(&args[3]));
            let ext = args.get(4).map(|s| s.as_str()).unwrap_or("jpg");
            let sa = stems(da, ext);
            let sb: std::collections::HashSet<String> =
                stems(db, ext).into_iter().collect();
            println!("stems_in_A={} shared={}", sa.len(), sa.iter().filter(|s| sb.contains(*s)).count());
            // Strip the suffix asymmetry: callers pass matching dirs.
            for s in &sa {
                if !sb.contains(s) {
                    continue;
                }
                let ia = load_rgb(&da.join(format!("{s}.{ext}")));
                let la = luma(&ia);
                let ta = otsu_threshold(&la);
                let ma = binarize(&la, ta);
                let ib = load_rgb(&db.join(format!("{s}.{ext}")));
                pair(&ma, ta, &ib, s);
            }
        }
        Some("selfgn") => {
            // Image vs itself + pipeline-magnitude grain + noise.
            let img = load_rgb(Path::new(&args[2]));
            let l = luma(&img);
            let t = otsu_threshold(&l);
            let m = binarize(&l, t);
            let mut rng = ChaCha8Rng::seed_from_u64(99);
            let mut pert = l.clone();
            // grain σ2 + warm-ish handled as pure additive (worst case)
            for v in pert.iter_mut() {
                *v += gauss(&mut rng) * 2.0;
            }
            // noise σ6
            for v in pert.iter_mut() {
                *v += gauss(&mut rng) * 6.0;
            }
            let tp = otsu_threshold(&pert);
            let mp = binarize(&pert, tp);
            let (flip, f1, rec, prec) = scores(&m, &mp);
            println!("selfgn t={t} tp={tp} flip%={:.3} F1={:.4} R={:.4} P={:.4}", flip * 100.0, f1, rec, prec);
        }
        _ => {
            eprintln!("usage: cmpbin pair <a> <b> | cmpbin dir <A> <B> [ext] | cmpbin selfgn <img>");
            std::process::exit(1);
        }
    }
}
