//! score-render: batch-render MusicXML / MEI / ABC scores to clean PNGs + ground-truth JSON.
//!
//! Zero-dependency Rust CLI. Real engraving backends are hooked in through the
//! [`ScoreRenderer`] trait: if a `verovio` or `lilypond` binary is on `PATH`
//! it is used, otherwise rendering falls back to the built-in [`StubRenderer`]
//! which emits a valid placeholder PNG (white page + staff lines) so the
//! pipeline stays runnable without external tools.

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Score formats accepted under `input/scores`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    MusicXml,
    Mei,
    Abc,
    Mscx,
    Mscz,
}

impl Format {
    fn as_str(self) -> &'static str {
        match self {
            Format::MusicXml => "musicxml",
            Format::Mei => "mei",
            Format::Abc => "abc",
            Format::Mscx => "mscx",
            Format::Mscz => "mscz",
        }
    }

    /// Detect format from file extension, falling back to content sniffing.
    fn detect(path: &Path, content: &str) -> Option<Format> {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            let ext = ext.to_ascii_lowercase();
            match ext.as_str() {
                "musicxml" | "xml" => return Some(Format::MusicXml),
                "mei" => return Some(Format::Mei),
                "abc" => return Some(Format::Abc),
                "mscx" => return Some(Format::Mscx),
                "mscz" => return Some(Format::Mscz),
                _ => {}
            }
        }
        let head: String = content.chars().take(2000).collect();
        if head.contains("<score-partwise") || head.contains("<score-timewise") {
            Some(Format::MusicXml)
        } else if head.contains("<mei") || head.contains("http://www.music-encoding.org") {
            Some(Format::Mei)
        } else if head.lines().any(|l| {
            let t = l.trim_start();
            t.starts_with("X:") || t.starts_with("M:") || t.starts_with("K:")
        }) {
            Some(Format::Abc)
        } else {
            None
        }
    }
}

fn split_fonts(raw: &str) -> Vec<String> {
    raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
}

struct Config {
    input: PathBuf,
    out: PathBuf,
    renderer: String,
    musescore_bin: Option<PathBuf>,
    /// symbox-extract binary for per-page glyph boxes (musescore backend
    /// only); empty symbols when absent. Probed on PATH if unset.
    symbox_bin: Option<PathBuf>,
    /// Engraving fonts to render with (musescore backend only); empty = as-is.
    musescore_fonts: Vec<String>,
    /// Score-text fonts to render with (musescore backend only); empty = paired "<music> Text" or as-is.
    /// Combined with musescore_fonts as a full cross product.
    musescore_text_fonts: Vec<String>,
    /// Paper sizes in European metric presets (`a4`, `a4_land`, `a5`, `a5_land`, `b4`, `folio`, or `WxH` in mm).
    musescore_page_sizes: Vec<String>,
    /// Disable `-j` job-file batching (one MuseScore launch per group).
    no_batch: bool,
    /// Fail loudly instead of falling back to the stub renderer.
    /// Training corpora must never contain placeholder images.
    require_backend: bool,
    /// PNG export resolution in DPI (musescore backend `-r`; default 300 ≈
    /// final training size, ~16x fewer pixels than the 1200dpi default).
    musescore_dpi: u32,
    width: u32,
    height: u32,
}

impl Config {
    fn defaults() -> Self {
        Self {
            input: PathBuf::from("input/scores"),
            out: PathBuf::from("output/renders"),
            renderer: "auto".to_string(),
            musescore_bin: None,
            symbox_bin: None,
            musescore_fonts: Vec::new(),
            musescore_text_fonts: Vec::new(),
            musescore_page_sizes: Vec::new(),
            require_backend: false,
        no_batch: false,
            musescore_dpi: 300,
            width: 1000,
            height: 700,
        }
    }
}

fn print_usage(program: &str) {
    eprintln!(
        "Usage: {program} [--input DIR] [--out DIR] [--renderer auto|stub|verovio|lilypond|musescore] [--musescore-bin PATH] [--musescore-fonts A,B,C] [--musescore-text-fonts X,Y] [--no-batch] [--require-backend] [--musescore-dpi N] [--width PX] [--height PX] [--symbox-bin PATH]\n\
         \n\
         Reads MusicXML (.musicxml/.xml), MEI (.mei), ABC (.abc) and MuseScore\n\
         (.mscx/.mscz) files from --input (default: input/scores) and writes,\n\
         per score:\n\
         \n  <stem>.png        rendered page image (one file per page for musescore)\n  <stem>.musicxml  engraved ground truth (musescore backend only)\n  <stem>.json       ground-truth record\n\
         \ninto --out (default: output/renders).\n\
         \nRenderer backends `verovio`, `lilypond` and `musescore` shell out to\n\
         those binaries when present and fall back to the built-in stub\n\
         otherwise. `--musescore-bin` pins the MuseScore binary."
    );
}

fn parse_args() -> Result<Config, String> {
    let mut cfg = Config::defaults();
    let raw: Vec<String> = env::args().collect();
    let program = raw.first().cloned().unwrap_or_else(|| "score-render".into());
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "-h" | "--help" => {
                print_usage(&program);
                std::process::exit(0);
            }
            "-i" | "--input" => {
                i += 1;
                cfg.input = PathBuf::from(raw.get(i).ok_or("--input needs a value")?);
            }
            "-o" | "--out" | "--output" => {
                i += 1;
                cfg.out = PathBuf::from(raw.get(i).ok_or("--out needs a value")?);
            }
            "-r" | "--renderer" => {
                i += 1;
                cfg.renderer = raw.get(i).ok_or("--renderer needs a value")?.to_ascii_lowercase();
            }
            "--musescore-bin" => {
                i += 1;
                cfg.musescore_bin = Some(PathBuf::from(raw.get(i).ok_or("--musescore-bin needs a value")?));
            }
            "--symbox-bin" => {
                i += 1;
                cfg.symbox_bin = Some(PathBuf::from(raw.get(i).ok_or("--symbox-bin needs a value")?));
            }
            "--musescore-fonts" => {
                i += 1;
                cfg.musescore_fonts = split_fonts(raw.get(i).ok_or("--musescore-fonts needs a value")?);
            }
            "--musescore-text-fonts" => {
                i += 1;
                cfg.musescore_text_fonts = split_fonts(raw.get(i).ok_or("--musescore-text-fonts needs a value")?);
            }
            "--musescore-page-sizes" => {
                i += 1;
                cfg.musescore_page_sizes = split_fonts(raw.get(i).ok_or("--musescore-page-sizes needs a value")?);
            }
            "--no-batch" => {
                cfg.no_batch = true;
            }
            "--require-backend" => {
                cfg.require_backend = true;
            }
            "--musescore-dpi" => {
                i += 1;
                cfg.musescore_dpi = raw
                    .get(i)
                    .ok_or("--musescore-dpi needs a value")?
                    .parse()
                    .map_err(|_| "--musescore-dpi needs an integer DPI value")?;
            }
            "--width" => {
                i += 1;
                cfg.width = raw
                    .get(i)
                    .ok_or("--width needs a value")?
                    .parse()
                    .map_err(|_| "--width must be an integer")?;
            }
            "--height" => {
                i += 1;
                cfg.height = raw
                    .get(i)
                    .ok_or("--height needs a value")?
                    .parse()
                    .map_err(|_| "--height must be an integer")?;
            }
            other => return Err(format!("unknown argument: {other} (see --help)")),
        }
        i += 1;
    }
    if cfg.width == 0 || cfg.height == 0 {
        return Err("--width/--height must be > 0".into());
    }
    if cfg.width > 4000 || cfg.height > 4000 {
        return Err("--width/--height capped at 4000px for the stub encoder".into());
    }
    match cfg.renderer.as_str() {
        "auto" | "stub" | "verovio" | "lilypond" | "musescore" => {}
        other => return Err(format!("unknown renderer: {other} (expected auto|stub|verovio|lilypond|musescore)")),
    }
    Ok(cfg)
}

/// One input score to render.
#[derive(Clone)]
struct RenderJob {
    source: PathBuf,
    format: Format,
    stem: String,
    /// Optional MuseScore style file (font switch); musescore backend only.
    style: Option<PathBuf>,
}

/// One extracted glyph box on one page (symbox-extract output, PNG pixels).
struct SymbolRec {
    page: usize,
    cls: String,
    bbox: [f64; 4],
}

/// Outcome reported by a renderer; serialized into the `.json` sidecar.
struct RenderOutcome {
    backend_used: String,
    note_count: u32,
    /// Rendered page images, in order (single entry for single-page backends).
    pages: Vec<String>,
    /// Exported symbolic ground truth, if the backend produces one.
    musicxml: Option<String>,
    /// Per-page glyph boxes (musescore + symbox-extract only, else empty).
    symbols: Vec<SymbolRec>,
    warnings: Vec<String>,
}

/// Rendering backend. Real engravers implement this by shelling out;
/// [`StubRenderer`] always succeeds offline.
trait ScoreRenderer {
    fn name(&self) -> &'static str;
    fn render(&self, job: &RenderJob, png: &Path, width: u32, height: u32) -> Result<RenderOutcome, String>;
}

/// Built-in backend: writes a valid white-page PNG with staff lines.
struct StubRenderer;

impl ScoreRenderer for StubRenderer {
    fn name(&self) -> &'static str {
        "stub"
    }

    fn render(&self, job: &RenderJob, png: &Path, width: u32, height: u32) -> Result<RenderOutcome, String> {
        let content = fs::read_to_string(&job.source).map_err(|e| format!("read {}: {e}", job.source.display()))?;
        let note_count = count_notes(&content, job.format);
        write_stub_png(png, width, height)
            .map_err(|e| format!("write {}: {e}", png.display()))?;
        Ok(RenderOutcome {
            backend_used: "stub".into(),
            note_count,
            pages: vec![],
            musicxml: None,
            symbols: vec![],
            warnings: vec!["stub renderer: placeholder image, not engraved notation".into()],
        })
    }
}

/// Verovio hook: `verovio -o <png> <src>` when the binary exists, else stub.
struct VerovioRenderer {
    fallback: StubRenderer,
    require_backend: bool,
}

impl ScoreRenderer for VerovioRenderer {
    fn name(&self) -> &'static str {
        "verovio"
    }

    fn render(&self, job: &RenderJob, png: &Path, width: u32, height: u32) -> Result<RenderOutcome, String> {
        let probe = Command::new("verovio").arg("--help").output();
        match probe {
            Ok(out) if out.status.success() => {
                let status = Command::new("verovio")
                    .arg("-o")
                    .arg(png)
                    .arg(&job.source)
                    .status()
                    .map_err(|e| format!("spawn verovio: {e}"))?;
                if status.success() && png.exists() {
                    let content = fs::read_to_string(&job.source).unwrap_or_default();
                    return Ok(RenderOutcome {
                        backend_used: "verovio".into(),
                        note_count: count_notes(&content, job.format),
                        pages: vec![],
                        musicxml: None,
                        symbols: vec![],
                        warnings: vec![],
                    });
                }
                if self.require_backend {
                    return Err("refusing stub fallback: verovio failed and --require-backend is set".into());
                }
                let mut o = self.fallback.render(job, png, width, height)?;
                o.backend_used = "stub (verovio failed)".into();
                o.warnings.push("verovio exited non-zero; used stub fallback".into());
                Ok(o)
            }
            _ => {
                if self.require_backend {
                    return Err("refusing stub fallback: verovio not found and --require-backend is set".into());
                }
                let mut o = self.fallback.render(job, png, width, height)?;
                o.backend_used = "stub (verovio not found)".into();
                o.warnings.push("verovio binary not on PATH; used stub fallback".into());
                Ok(o)
            }
        }
    }
}

/// One `-j` job-file entry per unit: absolute input score + outputs
/// (PNG pages, `.musicxml` ground truth, per-page SVG for symbox boxes).
fn job_file_json(entries: &[(String, String, String, String)]) -> String {
    let mut s = String::from("[\n");
    for (i, (inp, png, mxl, svg)) in entries.iter().enumerate() {
        s.push_str(&format!(
            "  {{\"in\": \"{}\", \"out\": [\"{}\", \"{}\", \"{}\"]}}{}\n",
            json_escape(inp),
            json_escape(png),
            json_escape(mxl),
            json_escape(svg),
            if i + 1 < entries.len() { "," } else { "" }
        ));
    }
    s.push(']');
    s
}

/// Files in `after` that were absent from `before` and belong to `stem` as
/// `<stem>.<ext>` or `<stem>-*.<ext>`, sorted. `before` is a prebuilt set so
/// the per-unit batch scan stays linear in the directory size (a Vec +
/// `contains` per unit turned the Emmentaler group quadratic: units x
/// after-dir x before-dir).
fn new_unit_files(before: &HashSet<PathBuf>, after: &[PathBuf], stem: &str, ext: &str) -> Vec<String> {
    let mut v: Vec<String> = after
        .iter()
        .filter(|p| !before.contains(*p))
        .filter_map(|p| {
            let n = p.file_name()?.to_str()?;
            let tagged = format!("{stem}-");
            let wanted = n == format!("{stem}.{ext}")
                || (n.starts_with(&tagged) && n.ends_with(&format!(".{ext}")));
            if wanted { Some(n.to_string()) } else { None }
        })
        .collect();
    v.sort();
    v
}

/// Remove previous outputs for one stem (`<stem>.png`, `<stem>-*.png`,
/// `<stem>.svg`, `<stem>-*.svg`, `<stem>.musicxml`, `<stem>.json`) so a
/// re-render into a dirty directory is detected by the path-diff below.
/// Without this, overwritten pages keep their path, look pre-existing, and
/// the unit wrongly falls back to stub.
fn clear_stem_outputs(out_dir: &Path, stem: &str) {
    clear_stems_outputs(out_dir, &[stem]);
}

/// Remove previous outputs for a whole batch of stems with a SINGLE
/// directory scan. The old per-stem loop re-read a 64k-file output
/// directory once per stem (quadratic); this keeps the exact per-stem
/// match rules (`{stem}-` dash prefix, so `ds0001` never matches
/// `ds00010-1.png`) while scanning once.
fn clear_stems_outputs(out_dir: &Path, stems: &[&str]) {
    if stems.is_empty() {
        return;
    }
    let dashes: Vec<String> = stems.iter().map(|s| format!("{s}-")).collect();
    // Index of the stem owning a file name, if any.
    let owner = |n: &str| -> Option<usize> {
        for (i, stem) in stems.iter().enumerate() {
            if n == &format!("{stem}.png")
                || (n.starts_with(&dashes[i]) && n.ends_with(".png"))
                || n == &format!("{stem}.svg")
                || (n.starts_with(&dashes[i]) && n.ends_with(".svg"))
                || n == &format!("{stem}.musicxml")
                || n == &format!("{stem}.json")
            {
                return Some(i);
            }
        }
        None
    };
    let rd = match fs::read_dir(out_dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    for e in rd.filter_map(|e| e.ok()) {
        let p = e.path();
        if !p.is_file() {
            continue;
        }
        let Some(n) = p.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if owner(n).is_some() {
            let _ = fs::remove_file(&p);
        }
    }
}

/// Render a same-style group with ONE MuseScore invocation (`-j` job file).
/// Returns outcomes only for units whose pages + musicxml actually landed;
/// the caller falls back to per-file renders for the rest.
fn render_batch_musescore(
    bin: &Path,
    style: Option<&Path>,
    out_dir: &Path,
    units: &[(RenderJob, PathBuf)],
    dpi: u32,
    symbox_bin: Option<&Path>,
) -> Vec<(usize, RenderOutcome)> {
    let abs_out = match fs::canonicalize(out_dir) {
        Ok(p) => p,
        Err(_) => return Vec::new(),
    };
    let mut entries: Vec<(String, String, String, String)> = Vec::new();
    for (job, png) in units {
        let Ok(src) = fs::canonicalize(&job.source) else { return Vec::new() };
        let Some(name) = png.file_name().and_then(|s| s.to_str()) else { return Vec::new() };
        let Some(stem) = Path::new(name).file_stem().and_then(|s| s.to_str()) else { return Vec::new() };
        entries.push((
            src.to_string_lossy().into_owned(),
            abs_out.join(name).to_string_lossy().into_owned(),
            abs_out.join(format!("{stem}.musicxml")).to_string_lossy().into_owned(),
            abs_out.join(format!("{stem}.svg")).to_string_lossy().into_owned(),
        ));
    }
    let tag = style.and_then(|s| s.file_stem()).and_then(|s| s.to_str()).unwrap_or("default");
    let job_path = std::env::temp_dir().join(format!("score-render-{tag}-{}.json", std::process::id()));
    if fs::write(&job_path, job_file_json(&entries)).is_err() {
        return Vec::new();
    }
    let snapshot = |d: &Path| -> Vec<PathBuf> {
        fs::read_dir(d).map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect()).unwrap_or_default()
    };
    {
        let stems: Vec<&str> = units.iter().map(|(job, _)| job.stem.as_str()).collect();
        clear_stems_outputs(&abs_out, &stems);
    }
    let before: HashSet<PathBuf> = snapshot(&abs_out).into_iter().collect();
    // Isolated TEMP/TMP defeats the single-instance socket: parallel
    // score-render processes each get their own MuseScore server, and a
    // stale lock can no longer swallow this launch silently.
    let shard = std::env::temp_dir().join(format!("score-render-ms-{}", std::process::id()));
    let _ = fs::create_dir_all(&shard);
    let mut cmd = Command::new(bin);
    cmd.env_remove("QT_QPA_PLATFORM");
    cmd.env("TEMP", &shard).env("TMP", &shard);
    if let Some(st) = style {
        cmd.arg("-S").arg(st);
    }
    let dpi_arg = dpi.to_string();
    let ok = cmd
        .arg("-F")
        .arg("-f")
        .arg("-r")
        .arg(&dpi_arg)
        .arg("-j")
        .arg(&job_path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    let _ = fs::remove_file(&job_path);
    if !ok {
        return Vec::new();
    }
    let after = snapshot(&abs_out);
    let mut out = Vec::new();
    for (k, (job, _)) in units.iter().enumerate() {
        let stem = &job.stem;
        let pages = new_unit_files(&before, &after, stem, "png");
        let mxl = abs_out.join(format!("{stem}.musicxml"));
        if pages.is_empty() || !mxl.exists() {
            continue;
        }
        let svg_files = new_unit_files(&before, &after, stem, "svg");
        let (symbols, warnings) = match symbox_bin {
            Some(bin) => {
                let pairs = pair_page_files(&pages, &svg_files);
                if pairs.is_empty() && !svg_files.is_empty() {
                    (Vec::new(), vec!["svg pages do not pair with png pages; symbols empty".to_string()])
                } else {
                    extract_page_symbols(bin, &abs_out, stem, &pairs)
                }
            }
            None => (
                Vec::new(),
                if svg_files.is_empty() {
                    Vec::new()
                } else {
                    vec!["symbox-extract not found; symbols empty".to_string()]
                },
            ),
        };
        let content = fs::read_to_string(&mxl).unwrap_or_default();
        out.push((
            k,
            RenderOutcome {
                backend_used: "musescore-batch".into(),
                note_count: count_notes(&content, Format::MusicXml),
                pages,
                musicxml: Some(format!("{stem}.musicxml")),
                symbols,
                warnings,
            },
        ));
    }
    out
}

/// Minimal MuseScore style selecting an engraving font plus its text font
/// (SMuFL families ship as "<Name>" + "<Name> Text").
fn font_style_mss(symbol_font: &str, text_font: &str) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<museScore version=\"4.00\">\n  <Style>\n    <musicalSymbolFont>{symbol_font}</musicalSymbolFont>\n    <musicalTextFont>{text_font}</musicalTextFont>\n  </Style>\n</museScore>\n"
    )
}

/// Write `<out>/.styles/<font>.mss` selecting `font` (+ "<font> Text").
fn write_font_style(out_dir: &Path, font: &str) -> Result<PathBuf, String> {
    write_combo_style(out_dir, Some(font), None)
}

/// All score-text `*FontFace` style keys: setting them re-skins every lyric,
/// dynamic, title, fingering and mark in one go.
const TEXT_FONT_KEYS: &[&str] = &[
    "bendFontFace", "chordSymbolAFontFace", "chordSymbolBFontFace", "composerFontFace",
    "defaultFontFace", "dynamicsFontFace", "expressionFontFace", "figuredBassFontFace",
    "fingeringFontFace", "footerFontFace", "frameFontFace", "glissandoFontFace",
    "hairpinFontFace", "headerFontFace", "instrumentChangeFontFace", "letRingFontFace",
    "lhGuitarFingeringFontFace", "longInstrumentFontFace", "lyricistFontFace",
    "lyricsEvenFontFace", "lyricsOddFontFace", "measureNumberFontFace", "metronomeFontFace",
    "mmRestRangeFontFace", "nashvilleNumberFontFace", "ottavaFontFace", "palmMuteFontFace",
    "partInstrumentFontFace", "pedalFontFace", "rehearsalMarkFontFace", "repeatLeftFontFace",
    "repeatRightFontFace", "rhGuitarFingeringFontFace", "romanNumeralFontFace",
    "shortInstrumentFontFace", "staffFontFace", "stickingFontFace", "stringNumberFontFace",
    "subTitleFontFace", "systemFontFace", "systemTextLineFontFace", "tempoFontFace",
    "textLineFontFace", "titleFontFace", "translatorFontFace", "tupletFontFace",
    "user10FontFace", "user11FontFace", "user12FontFace", "user1FontFace", "user2FontFace",
    "user3FontFace", "user4FontFace", "user5FontFace", "user6FontFace", "user7FontFace",
    "user8FontFace", "user9FontFace", "voltaFontFace",
];

/// Style selecting a music-font/text-font combo. `None` leaves that axis at
/// the score default. Text fonts apply to musicalTextFont plus every text
/// style key; a missing music font keeps "<music> Text" pairing off and the
/// text font carries dynamics/lyrics/titles on its own.
/// European metric paper-size preset (dimensions in mm; converted to inches in `.mss`).
#[derive(Debug, Clone, PartialEq)]
struct PageSizeSpec {
    name: String,
    width_mm: f64,
    height_mm: f64,
    spatium_mm: Option<f64>,
}

fn parse_page_size(spec: &str) -> Result<PageSizeSpec, String> {
    let key = spec.trim().to_ascii_lowercase();
    match key.as_str() {
        "a4" => Ok(PageSizeSpec { name: "a4".into(), width_mm: 210.0, height_mm: 297.0, spatium_mm: None }),
        "a4_land" | "a4-land" => Ok(PageSizeSpec { name: "a4_land".into(), width_mm: 297.0, height_mm: 210.0, spatium_mm: None }),
        "a5" | "pocket" => Ok(PageSizeSpec { name: "a5".into(), width_mm: 148.0, height_mm: 210.0, spatium_mm: Some(1.45) }),
        "a5_land" | "a5-land" | "excerpt" => Ok(PageSizeSpec { name: "a5_land".into(), width_mm: 210.0, height_mm: 148.0, spatium_mm: None }),
        "b4" => Ok(PageSizeSpec { name: "b4".into(), width_mm: 250.0, height_mm: 353.0, spatium_mm: Some(1.55) }),
        "folio" => Ok(PageSizeSpec { name: "folio".into(), width_mm: 240.0, height_mm: 355.0, spatium_mm: Some(1.55) }),
        other => {
            let parts: Vec<&str> = other.split('x').collect();
            if parts.len() == 2 {
                let w: f64 = parts[0].parse().map_err(|_| format!("invalid page size width in {spec:?}"))?;
                let h: f64 = parts[1].parse().map_err(|_| format!("invalid page size height in {spec:?}"))?;
                if (80.0..=600.0).contains(&w) && (80.0..=600.0).contains(&h) {
                    return Ok(PageSizeSpec { name: format!("{}x{}mm", w.round() as u32, h.round() as u32), width_mm: w, height_mm: h, spatium_mm: None });
                }
            }
            Err(format!("unknown --musescore-page-sizes spec {spec:?} (use a4, a4_land, a5, a5_land, b4, folio, or WxH in mm)"))
        }
    }
}

fn combo_style_mss_with_page(music: Option<&str>, text: Option<&str>, page: Option<&PageSizeSpec>) -> String {
    let mut s = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<museScore version=\"4.00\">\n  <Style>\n");
    if let Some(p) = page {
        s.push_str(&format!("    <pageWidth>{:.4}</pageWidth>\n", p.width_mm / 25.4));
        s.push_str(&format!("    <pageHeight>{:.4}</pageHeight>\n", p.height_mm / 25.4));
        if let Some(sp) = p.spatium_mm {
            s.push_str(&format!("    <Spatium>{:.3}</Spatium>\n", sp));
        }
    }
    if let Some(m) = music {
        s.push_str(&format!("    <musicalSymbolFont>{m}</musicalSymbolFont>\n"));
        s.push_str(&format!("    <musicalTextFont>{}</musicalTextFont>\n", text.unwrap_or(&format!("{m} Text"))));
    } else if let Some(t) = text {
        s.push_str(&format!("    <musicalTextFont>{t}</musicalTextFont>\n"));
    }
    if let Some(t) = text {
        for key in TEXT_FONT_KEYS {
            s.push_str(&format!("    <{key}>{t}</{key}>\n"));
        }
    }
    s.push_str("  </Style>\n</museScore>\n");
    s
}

fn combo_style_mss(music: Option<&str>, text: Option<&str>) -> String {
    combo_style_mss_with_page(music, text, None)
}

/// Filename-safe slug: alphanumerics kept, everything else dropped.
fn slug(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).collect()
}

/// Write `<out>/.styles/<combo>.mss` for a music/text font + optional metric page-size combo.
fn write_combo_style_with_page(out_dir: &Path, music: Option<&str>, text: Option<&str>, page: Option<&PageSizeSpec>) -> Result<PathBuf, String> {
    let dir = out_dir.join(".styles");
    fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let mut name = match (music, text) {
        (Some(m), Some(t)) => format!("{}_{}", slug(m), slug(t)),
        (Some(m), None) => slug(m),
        (None, Some(t)) => format!("text_{}", slug(t)),
        (None, None) => "default".to_string(),
    };
    if let Some(p) = page {
        name.push('_');
        name.push_str(&slug(&p.name));
    }
    let path = dir.join(format!("{name}.mss"));
    fs::write(&path, combo_style_mss_with_page(music, text, page)).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

/// Write `<out>/.styles/<combo>.mss` for a music/text font combo.
fn write_combo_style(out_dir: &Path, music: Option<&str>, text: Option<&str>) -> Result<PathBuf, String> {
    write_combo_style_with_page(out_dir, music, text, None)
}

/// Locate a MuseScore binary: explicit flag, then PATH names, then the
/// well-known install locations (Windows / macOS / Linux).
fn find_musescore(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        if p.exists() {
            return Some(p.to_path_buf());
        }
        return None;
    }
    for name in ["MuseScore4", "musescore", "mscore"] {
        if Command::new(name).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
            return Some(PathBuf::from(name));
        }
    }
    for probe in [
        "C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe",
        "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
        "/usr/bin/musescore",
        "/usr/local/bin/musescore",
    ] {
        let p = PathBuf::from(probe);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// MuseScore backend: real engraved rendering via headless export.
/// Exports every page as PNG plus a `.musicxml` ground-truth file, so each
/// rendered page pairs 1:1 with symbolic notation for OMR training.
/// Falls back to the stub when no binary is available.
struct MuseScoreRenderer {
    fallback: StubRenderer,
    bin: Option<PathBuf>,
    dpi: u32,
    shard_tmp: PathBuf,
    symbox_bin: Option<PathBuf>,
    require_backend: bool,
}

impl ScoreRenderer for MuseScoreRenderer {
    fn name(&self) -> &'static str {
        "musescore"
    }

    fn render(&self, job: &RenderJob, png: &Path, _width: u32, _height: u32) -> Result<RenderOutcome, String> {
        let Some(bin) = self.bin.clone() else {
            if self.require_backend {
                return Err("refusing stub fallback: MuseScore binary not found and --require-backend is set".into());
            }
            let mut o = self.fallback.render(job, png, _width, _height)?;
            o.backend_used = "stub (musescore not found)".into();
            o.warnings.push("no MuseScore binary found (see --musescore-bin); used stub fallback".into());
            return Ok(o);
        };
        let out_dir = png.parent().ok_or_else(|| format!("no parent dir for {}", png.display()))?;
        let stem = png.file_stem().and_then(|s| s.to_str()).unwrap_or("score");
        clear_stem_outputs(out_dir, &job.stem);
        let before: HashSet<PathBuf> = fs::read_dir(out_dir)
            .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        // Page images. MuseScore appends page numbers (`<stem>-1.png`,
        // `<stem>-2.png`, ...). Diff the directory so any naming convention
        // is picked up. NOTE: do NOT force QT_QPA_PLATFORM=offscreen — the
        // stock Windows install ships only qwindows.dll, and a missing
        // platform plugin hangs the process on an invisible error dialog.
        // The native platform exports headlessly without any open window.
        let run_export = |out: &Path| -> Result<bool, String> {
            let mut cmd = Command::new(&bin);
            cmd.env_remove("QT_QPA_PLATFORM");
            cmd.env("TEMP", &self.shard_tmp).env("TMP", &self.shard_tmp);
            if let Some(st) = &job.style {
                cmd.arg("-S").arg(st);
            }
            let dpi_arg = self.dpi.to_string();
            let st = cmd
                .arg("-F")
                .arg("-f")
                .arg("-r")
                .arg(&dpi_arg)
                .arg("--export-to")
                .arg(out)
                .arg(&job.source)
                .status()
                .map_err(|e| format!("spawn {}: {e}", bin.display()))?;
            Ok(st.success())
        };
        if !run_export(png)? {
            if self.require_backend {
                return Err("refusing stub fallback: musescore export failed and --require-backend is set".into());
            }
            let mut o = self.fallback.render(job, png, _width, _height)?;
            o.backend_used = "stub (musescore export failed)".into();
            o.warnings.push("musescore --export-to exited non-zero; used stub fallback".into());
            return Ok(o);
        }
        let mut pages: Vec<String> = fs::read_dir(out_dir)
            .map_err(|e| format!("list {}: {e}", out_dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                !before.contains(p)
                    && p.extension().and_then(|x| x.to_str()).map(|x| x.eq_ignore_ascii_case("png")).unwrap_or(false)
                    && p.file_stem().and_then(|s| s.to_str()).map(|s| s.starts_with(stem)).unwrap_or(false)
            })
            .filter_map(|p| p.file_name().and_then(|s| s.to_str()).map(|s| s.to_string()))
            .collect();
        pages.sort();
        if pages.is_empty() {
            let mut o = self.fallback.render(job, png, _width, _height)?;
            o.backend_used = "stub (musescore produced no png)".into();
            o.warnings.push("musescore export produced no PNG; used stub fallback".into());
            return Ok(o);
        }
        // Symbolic ground truth: export MusicXML alongside the pages.
        let mxl_name = format!("{stem}.musicxml");
        let mxl_path = out_dir.join(&mxl_name);
        let mut warnings = Vec::new();
        let note_count = match run_export(&mxl_path) {
            Ok(true) if mxl_path.exists() => {
                let content = fs::read_to_string(&mxl_path).unwrap_or_default();
                count_notes(&content, Format::MusicXml)
            }
            _ => {
                warnings.push("musescore musicxml export failed; note_count unknown".into());
                0
            }
        };
        // Vector ground truth: one SVG export, then symbox-extract per
        // paired page. SVG export failing (or no symbox binary) only
        // empties `symbols`; PNG + MusicXML output is unaffected.
        let mut symbols = Vec::new();
        if run_export(&out_dir.join(format!("{stem}.svg"))).unwrap_or(false) {
            let svg_files: Vec<String> = fs::read_dir(out_dir)
                .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect::<Vec<PathBuf>>())
                .unwrap_or_default()
                .into_iter()
                .filter(|p| {
                    !before.contains(p)
                        && p.extension().and_then(|x| x.to_str()).map(|x| x.eq_ignore_ascii_case("svg")).unwrap_or(false)
                        && p.file_stem().and_then(|s| s.to_str()).map(|s| s.starts_with(stem)).unwrap_or(false)
                })
                .filter_map(|p| p.file_name().and_then(|s| s.to_str()).map(|s| s.to_string()))
                .collect();
            match &self.symbox_bin {
                Some(bin) => {
                    let pairs = pair_page_files(&pages, &svg_files);
                    if pairs.is_empty() && !svg_files.is_empty() {
                        warnings.push("svg pages do not pair with png pages; symbols empty".to_string());
                    } else {
                        let (s, w) = extract_page_symbols(bin, out_dir, stem, &pairs);
                        symbols = s;
                        warnings.extend(w);
                    }
                }
                None if !svg_files.is_empty() => {
                    warnings.push("symbox-extract not found; symbols empty".to_string());
                }
                _ => {}
            }
        } else {
            warnings.push("musescore svg export failed; symbols empty".to_string());
        }
        Ok(RenderOutcome {
            backend_used: "musescore".into(),
            note_count,
            pages,
            musicxml: if mxl_path.exists() { Some(mxl_name) } else { None },
            symbols,
            warnings,
        })
    }
}

/// LilyPond hook: only `.ly`-adjacent flows render natively; score formats
/// fall back to the stub with a warning unless a converter is configured.
struct LilyPondRenderer {
    fallback: StubRenderer,
    require_backend: bool,
}

impl ScoreRenderer for LilyPondRenderer {
    fn name(&self) -> &'static str {
        "lilypond"
    }

    fn render(&self, job: &RenderJob, png: &Path, width: u32, height: u32) -> Result<RenderOutcome, String> {
        let probe = Command::new("lilypond").arg("--version").output();
        let available = matches!(probe, Ok(out) if out.status.success());
        if self.require_backend {
            return Err("refusing stub fallback: no lilypond converter wired and --require-backend is set".into());
        }
        let mut o = self.fallback.render(job, png, width, height)?;
        if available {
            o.backend_used = "stub (lilypond hook: no direct MusicXML/MEI/ABC path configured)".into();
            o.warnings.push(
                "lilypond present but no converter wired for this format; used stub fallback (hook: musicxml2ly / custom converter here)".into(),
            );
        } else {
            o.backend_used = "stub (lilypond not found)".into();
            o.warnings.push("lilypond binary not on PATH; used stub fallback".into());
        }
        Ok(o)
    }
}

/// Heuristic ground-truth note count per format (exact pitches come from a
/// real engraver/backend parse; the stub records counts only).
fn count_notes(content: &str, format: Format) -> u32 {
    match format {
        // MuseScore .mscx nests <Note><pitch> inside <Chord>; <mscz> is a zip
        // whose note count is taken from the exported MusicXML instead.
        Format::Mscx => {
            let mut n = 0u32;
            let mut rest = content;
            while let Some(pos) = rest.find("<Note") {
                let after = &rest[pos + 5..];
                match after.chars().next() {
                    Some(c) if c == '>' || c == '/' || c.is_whitespace() => n += 1,
                    _ => {}
                }
                rest = after;
            }
            n
        }
        Format::Mscz => 0,
        Format::MusicXml | Format::Mei => {
            // Count opening <note ...> elements, skipping <notations>/<notehead>.
            let mut n = 0u32;
            let mut rest = content;
            while let Some(pos) = rest.find("<note") {
                let after = &rest[pos + 5..];
                match after.chars().next() {
                    // Must be whitespace, '>', or '/' to be a <note> element.
                    Some(c) if c == '>' || c == '/' || c.is_whitespace() => {
                        // Exclude <notehead>, <notations>, <notetype>.
                        if after.starts_with("head") || after.starts_with("ations") || after.starts_with("type") {
                            // not a note element
                        } else {
                            n += 1;
                        }
                    }
                    _ => {}
                }
                rest = after;
            }
            n
        }
        Format::Abc => {
            // Count note letters in the body (after the header block).
            let mut in_body = false;
            let mut n = 0u32;
            for line in content.lines() {
                let t = line.trim();
                if !in_body {
                    if t.starts_with("K:") {
                        in_body = true;
                    }
                    continue;
                }
                let trimmed = t.strip_prefix("w:").unwrap_or(t);
                for c in trimmed.chars() {
                    if matches!(c, 'A'..='G' | 'a'..='g') {
                        n += 1;
                    }
                }
            }
            n
        }
    }
}

fn collect_scores(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let entries = fs::read_dir(&d).map_err(|e| format!("read {}: {e}", d.display()))?;
        for e in entries {
            let e = e.map_err(|e| format!("list {}: {e}", d.display()))?;
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.is_file() {
                if let Some(ext) = p.extension().and_then(|x| x.to_str()) {
                    match ext.to_ascii_lowercase().as_str() {
                        "musicxml" | "xml" | "mei" | "abc" | "mscx" | "mscz" => out.push(p),
                        _ => {}
                    }
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Page number from a page-file name: trailing `-N` before the
/// extension, else 1 (bare `<stem>.png` / `<stem>.svg`).
fn page_num(name: &str) -> usize {
    let base = name.rsplit('.').nth(1).unwrap_or(name);
    if let Some((_, tail)) = base.rsplit_once('-') {
        if let Ok(n) = tail.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    1
}

/// Pair page PNGs with their SVG twins by full stem (`<stem>-N` or bare
/// `<stem>`); orphans on either side are dropped. Sorted by page.
fn pair_page_files(pngs: &[String], svgs: &[String]) -> Vec<(String, String, usize)> {
    let mut out = Vec::new();
    for png in pngs {
        let key = png.rsplit('.').nth(1).unwrap_or(png);
        for svg in svgs {
            if svg.rsplit('.').nth(1).unwrap_or(svg) == key {
                out.push((png.clone(), svg.clone(), page_num(png)));
                break;
            }
        }
    }
    out.sort_by_key(|(_, _, p)| *p);
    out
}

/// Minimal parse of a symbox-extract BoxesDoc: `(cls, bbox)` per glyph.
/// Only `"cls"` string and `"bbox"` number-quadruple are read; every other
/// key is skipped. Anything misshapen is an error, never a silent drop.
fn parse_symbox_glyphs(json: &str) -> Result<Vec<(String, [f64; 4])>, String> {
    let mut p = Parser { s: json.as_bytes(), i: 0 };
    p.expect_key("glyphs")?;
    p.ws();
    p.expect_char(b'[')?;
    let mut out = Vec::new();
    loop {
        p.ws();
        if p.eat(b']') {
            break;
        }
        if !out.is_empty() {
            p.expect_char(b',')?;
            p.ws();
        }
        p.expect_char(b'{')?;
        let mut cls: Option<String> = None;
        let mut bbox: Option<[f64; 4]> = None;
        let mut first = true;
        loop {
            p.ws();
            if p.eat(b'}') {
                break;
            }
            if !first {
                p.expect_char(b',')?;
                p.ws();
            }
            first = false;
            let key = p.string()?;
            p.ws();
            p.expect_char(b':')?;
            p.ws();
            if key == "cls" {
                cls = Some(p.string()?);
            } else if key == "bbox" {
                bbox = Some(p.num_array4()?);
            } else {
                p.skip_value()?;
            }
            p.ws();
        }
        match (cls, bbox) {
            (Some(c), Some(b)) => out.push((c, b)),
            _ => return Err("symbox glyph without cls+bbox".to_string()),
        }
    }
    Ok(out)
}

/// Tiny recursive-descent reader over the known BoxesDoc shape.
struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }
    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect_char(&mut self, c: u8) -> Result<(), String> {
        if self.eat(c) {
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }
    fn expect_key(&mut self, want: &str) -> Result<(), String> {
        let pos = self.s.windows(want.len() + 2).position(|w| {
            w[0] == b'"' && &w[1..1 + want.len()] == want.as_bytes() && w[1 + want.len()] == b'"'
        });
        match pos {
            Some(_) => {
                // Advance to just past the key's closing quote.
                let key_start = self.s[self.i..]
                    .windows(want.len() + 2)
                    .position(|w| {
                        w[0] == b'"' && &w[1..1 + want.len()] == want.as_bytes() && w[1 + want.len()] == b'"'
                    })
                    .ok_or_else(|| format!("missing key \"{want}\""))?;
                self.i += key_start + want.len() + 2;
                self.ws();
                self.expect_char(b':')?;
                Ok(())
            }
            None => Err(format!("missing key \"{want}\"")),
        }
    }
    fn string(&mut self) -> Result<String, String> {
        self.expect_char(b'"')?;
        let mut o = String::new();
        loop {
            match self.peek() {
                None => return Err("unterminated string".to_string()),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(o);
                }
                Some(b'\\') => {
                    self.i += 1;
                    match self.peek() {
                        None => return Err("unterminated escape".to_string()),
                        Some(c) => {
                            o.push(c as char);
                            self.i += 1;
                        }
                    }
                }
                Some(c) => {
                    o.push(c as char);
                    self.i += 1;
                }
            }
        }
    }
    fn number(&mut self) -> Result<f64, String> {
        let start = self.i;
        if self.peek() == Some(b'-') || self.peek() == Some(b'+') {
            self.i += 1;
        }
        let mut any = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == b'.' || c == b'e' || c == b'E' || c == b'-' || c == b'+' {
                self.i += 1;
                any = true;
            } else {
                break;
            }
        }
        if !any {
            return Err(format!("expected number at byte {}", start));
        }
        std::str::from_utf8(&self.s[start..self.i])
            .map_err(|_| "bad utf8 in number".to_string())?
            .parse()
            .map_err(|_| format!("bad number at byte {start}"))
    }
    fn num_array4(&mut self) -> Result<[f64; 4], String> {
        self.expect_char(b'[')?;
        let mut v = [0.0; 4];
        for (k, slot) in v.iter_mut().enumerate() {
            self.ws();
            *slot = self.number()?;
            self.ws();
            if k < 3 {
                self.expect_char(b',')?;
            }
        }
        self.expect_char(b']')?;
        Ok(v)
    }
    fn skip_value(&mut self) -> Result<(), String> {
        self.ws();
        match self.peek() {
            Some(b'"') => {
                self.string()?;
            }
            Some(b'{') => {
                self.i += 1;
                let mut depth = 1;
                let mut in_str = false;
                while depth > 0 {
                    match self.s.get(self.i) {
                        None => return Err("unterminated object".to_string()),
                        Some(b'"') if !in_str => in_str = true,
                        Some(b'"') => in_str = false,
                        Some(b'\\') if in_str => self.i += 1,
                        Some(b'{') if !in_str => depth += 1,
                        Some(b'}') if !in_str => depth -= 1,
                        _ => {}
                    }
                    self.i += 1;
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut depth = 1;
                let mut in_str = false;
                while depth > 0 {
                    match self.s.get(self.i) {
                        None => return Err("unterminated array".to_string()),
                        Some(b'"') if !in_str => in_str = true,
                        Some(b'"') => in_str = false,
                        Some(b'\\') if in_str => self.i += 1,
                        Some(b'[') if !in_str => depth += 1,
                        Some(b']') if !in_str => depth -= 1,
                        _ => {}
                    }
                    self.i += 1;
                }
            }
            Some(c) if c == b'-' || c == b'+' || c.is_ascii_digit() => {
                self.number()?;
            }
            Some(_) => {
                // true / false / null
                while let Some(c) = self.peek() {
                    if c.is_ascii_alphabetic() {
                        self.i += 1;
                    } else {
                        break;
                    }
                }
            }
            None => return Err("unexpected end in value".to_string()),
        }
        Ok(())
    }
}

/// Locate a symbox-extract binary: explicit flag, then PATH probe.
fn find_symbox_extract(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        if p.exists() {
            return Some(p.to_path_buf());
        }
        return None;
    }
    match Command::new("symbox-extract").arg("--version").output() {
        Ok(out) if out.status.success() => Some(PathBuf::from("symbox-extract")),
        _ => None,
    }
}

/// Run the symbox-extract binary per paired page and collect glyph boxes.
/// Failures warn per page and yield partial symbols, never a fatal error.
fn extract_page_symbols(
    bin: &Path,
    out_dir: &Path,
    stem: &str,
    pairs: &[(String, String, usize)],
) -> (Vec<SymbolRec>, Vec<String>) {
    let mut symbols = Vec::new();
    let mut warnings = Vec::new();
    let tmp = std::env::temp_dir().join(format!("score-render-symbox-{}-{}", std::process::id(), stem));
    if fs::create_dir_all(&tmp).is_err() {
        warnings.push("symbox tmp dir unavailable; symbols empty".to_string());
        return (symbols, warnings);
    }
    for (png_name, svg_name, page) in pairs {
        let tmp_json = tmp.join(format!("p{page}.json"));
        let st = Command::new(bin)
            .arg("--svg")
            .arg(out_dir.join(svg_name))
            .arg("--png")
            .arg(out_dir.join(png_name))
            .arg("--out")
            .arg(&tmp_json)
            .arg("--page")
            .arg(page.to_string())
            .status();
        match st {
            Ok(s) if s.success() => match fs::read_to_string(&tmp_json) {
                Ok(content) => match parse_symbox_glyphs(&content) {
                    Ok(glyphs) => {
                        for (cls, bbox) in glyphs {
                            symbols.push(SymbolRec { page: *page, cls, bbox });
                        }
                    }
                    Err(e) => warnings.push(format!("{svg_name}: symbox parse: {e}")),
                },
                Err(e) => warnings.push(format!("{svg_name}: read symbox json: {e}")),
            },
            _ => warnings.push(format!("{svg_name}: symbox-extract failed")),
        }
    }
    let _ = fs::remove_dir_all(&tmp);
    symbols.sort_by_key(|s| (s.page, s.cls.clone()));
    (symbols, warnings)
}

/// `"symbols": [...]` sidecar fragment from extracted records.
fn symbols_json(symbols: &[SymbolRec]) -> String {
    if symbols.is_empty() {
        return "[]".to_string();
    }
    let parts: Vec<String> = symbols
        .iter()
        .map(|s| {
            format!(
                "{{\"page\": {}, \"cls\": \"{}\", \"bbox\": [{}, {}, {}, {}]}}",
                s.page,
                json_escape(&s.cls),
                s.bbox[0],
                s.bbox[1],
                s.bbox[2],
                s.bbox[3]
            )
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

fn write_ground_truth(
    path: &Path,
    job: &RenderJob,
    png_name: &str,
    width: u32,
    height: u32,
    outcome: &RenderOutcome,
    score_path_display: &str,
    font: Option<&str>,
    text_font: Option<&str>,
) -> std::io::Result<()> {
    let warnings: Vec<String> = outcome.warnings.iter().map(|w| format!("\"{}\"", json_escape(w))).collect();
    let pages: Vec<String> = if outcome.pages.is_empty() {
        vec![png_name.to_string()]
    } else {
        outcome.pages.clone()
    };
    let pages_json = pages.iter().map(|p| format!("\"{}\"", json_escape(p))).collect::<Vec<_>>().join(", ");
    let musicxml_json = match &outcome.musicxml {
        Some(m) => format!("\"{}\"", json_escape(m)),
        None => "null".to_string(),
    };
    let font_json = match font {
        Some(f) => format!("\"{}\"", json_escape(f)),
        None => "null".to_string(),
    };
    let symbols_json = symbols_json(&outcome.symbols);
    let text_font_json = match text_font {
        Some(f) => format!("\"{}\"", json_escape(f)),
        None => "null".to_string(),
    };
    let body = format!(
        "{{\n  \"source_file\": \"{}\",\n  \"format\": \"{}\",\n  \"renderer\": \"{}\",\n  \"font\": {},\n  \"text_font\": {},\n  \"image\": \"{}\",\n  \"pages\": [{}],\n  \"musicxml\": {},\n  \"image_width\": {},\n  \"image_height\": {},\n  \"note_count\": {},\n  \"symbols\": {},\n  \"warnings\": [{}]\n}}\n",
        json_escape(score_path_display),
        job.format.as_str(),
        json_escape(&outcome.backend_used),
        font_json,
        text_font_json,
        json_escape(png_name),
        pages_json,
        musicxml_json,
        width,
        height,
        outcome.note_count,
        symbols_json,
        warnings.join(", ")
    );
    fs::write(path, body)
}

// ---- Minimal PNG encoder (truecolor, stored deflate, no deps) ----

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = if crc & 1 == 1 { 0xEDB8_8320 } else { 0 };
            crc = (crc >> 1) ^ mask;
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    let mut c = Vec::with_capacity(4 + data.len());
    c.extend_from_slice(tag);
    c.extend_from_slice(data);
    out.extend_from_slice(&crc32(&c).to_be_bytes());
}

/// White page with two 5-line staves; enough for a "clean" placeholder.
fn write_stub_png(path: &Path, width: u32, height: u32) -> std::io::Result<()> {
    let w = width as usize;
    let h = height as usize;
    // RGB buffer, white.
    let mut px = vec![255u8; w * h * 3];
    let mut staff = |y: usize| {
        for x in 0..w {
            for dy in 0..2 {
                let yy = y + dy;
                if yy < h {
                    let i = (yy * w + x) * 3;
                    px[i] = 0;
                    px[i + 1] = 0;
                    px[i + 2] = 0;
                }
            }
        }
    };
    // Two systems positioned proportionally.
    let systems = [h / 4, h / 2];
    for &top in &systems {
        for line in 0..5 {
            staff(top + line * (h / 60).max(8));
        }
    }
    // Title bar placeholder.
    for y in (h / 12)..(h / 12 + 3.min(h)) {
        for x in (w / 3)..(2 * w / 3) {
            let i = (y * w + x) * 3;
            px[i] = 30;
            px[i + 1] = 30;
            px[i + 2] = 30;
        }
    }

    // Raw scanlines with filter byte 0.
    let row_len = 1 + w * 3;
    let mut raw = Vec::with_capacity(h * row_len);
    for y in 0..h {
        raw.push(0);
        raw.extend_from_slice(&px[y * w * 3..(y + 1) * w * 3]);
    }

    // zlib wrapper around stored (uncompressed) deflate blocks.
    let mut zit = vec![0x78u8, 0x01];
    for (bi, blk) in raw.chunks(65535).enumerate() {
        let last = bi == raw.len().div_ceil(65535) - 1;
        zit.push(if last { 0x01 } else { 0x00 });
        zit.extend_from_slice(&(blk.len() as u16).to_le_bytes());
        zit.extend_from_slice(&(!(blk.len() as u16)).to_le_bytes());
        zit.extend_from_slice(blk);
    }
    zit.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut png = Vec::new();
    png.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit truecolor
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &zit);
    chunk(&mut png, b"IEND", &[]);
    fs::write(path, png)
}

fn main() {
    let cfg = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("score-render: {e}");
            std::process::exit(2);
        }
    };

    if !cfg.input.exists() {
        eprintln!("score-render: input dir {} not found", cfg.input.display());
        std::process::exit(2);
    }

    let scores = match collect_scores(&cfg.input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("score-render: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = fs::create_dir_all(&cfg.out) {
        eprintln!("score-render: create {}: {e}", cfg.out.display());
        std::process::exit(1);
    }

    // Pick renderer.
    let stub = StubRenderer;
    let verovio = VerovioRenderer { fallback: StubRenderer, require_backend: cfg.require_backend };
    let lilypond = LilyPondRenderer { fallback: StubRenderer, require_backend: cfg.require_backend };
    let musescore_bin = find_musescore(cfg.musescore_bin.as_deref());
    let shard_tmp = std::env::temp_dir().join(format!("score-render-ms-{}", std::process::id()));
    let _ = fs::create_dir_all(&shard_tmp);
    let musescore = MuseScoreRenderer { fallback: StubRenderer, bin: musescore_bin.clone(), dpi: cfg.musescore_dpi, shard_tmp, symbox_bin: find_symbox_extract(cfg.symbox_bin.as_deref()), require_backend: cfg.require_backend };
    let renderer: &dyn ScoreRenderer = match cfg.renderer.as_str() {
        "stub" => &stub,
        "verovio" => &verovio,
        "lilypond" => &lilypond,
        "musescore" => &musescore,
        _ => {
            // auto: prefer real engravers (musescore, verovio, lilypond), else stub.
            if musescore_bin.is_some() {
                &musescore
            } else if Command::new("verovio").arg("--help").output().map(|o| o.status.success()).unwrap_or(false) {
                &verovio
            } else if Command::new("lilypond").arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
                &lilypond
            } else {
                &stub
            }
        }
    };
    eprintln!("score-render: backend={} input={} out={}", renderer.name(), cfg.input.display(), cfg.out.display());
    if cfg.require_backend && renderer.name() == "stub" {
        eprintln!("score-render: refusing stub renderer: no real backend available and --require-backend is set");
        std::process::exit(2);
    }

    let mut ok = 0u32;
    let mut skipped = 0u32;
    let mut failed = 0u32;
    /// A musescore unit deferred for `-j` batch rendering.
    struct PendingUnit {
        job: RenderJob,
        format: Format,
        png_name: String,
        png_path: PathBuf,
        json_path: PathBuf,
        display: String,
        music: Option<String>,
        text: Option<String>,
    }
    let mut pending: Vec<PendingUnit> = Vec::new();
    let batch_mode = renderer.name() == "musescore" && !cfg.no_batch;
    for src in scores {
        // .mscz is a zip: MuseScore reads it natively, no text sniffing.
        let is_zip = src
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("mscz"))
            .unwrap_or(false);
        let (content, format) = if is_zip {
            (String::new(), Format::Mscz)
        } else {
            let content = match fs::read_to_string(&src) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("  SKIP {}: {e}", src.display());
                    skipped += 1;
                    continue;
                }
            };
            let Some(format) = Format::detect(&src, &content) else {
                eprintln!("  SKIP {}: unknown format", src.display());
                skipped += 1;
                continue;
            };
            (content, format)
        };
        let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("score").to_string();
        // One render per requested music font (musescore backend only);
        // everything else renders once, as-is.
        // One render per music-font x text-font x page-size combo (musescore only).
        let mut units: Vec<(Option<String>, Option<String>, Option<String>, Option<PathBuf>)> = Vec::new();
        if renderer.name() == "musescore"
            && (!cfg.musescore_fonts.is_empty() || !cfg.musescore_text_fonts.is_empty() || !cfg.musescore_page_sizes.is_empty())
        {
            let musics: Vec<Option<String>> = if cfg.musescore_fonts.is_empty() {
                vec![None]
            } else {
                cfg.musescore_fonts.iter().map(|f| Some(f.clone())).collect()
            };
            let texts: Vec<Option<String>> = if cfg.musescore_text_fonts.is_empty() {
                vec![None]
            } else {
                cfg.musescore_text_fonts.iter().map(|f| Some(f.clone())).collect()
            };
            let pages: Vec<Option<PageSizeSpec>> = if cfg.musescore_page_sizes.is_empty() {
                vec![None]
            } else {
                let mut v = Vec::new();
                for ps in &cfg.musescore_page_sizes {
                    match parse_page_size(ps) {
                        Ok(spec) => v.push(Some(spec)),
                        Err(e) => {
                            eprintln!("  FAIL {stem}: {e}");
                            failed += 1;
                        }
                    }
                }
                v
            };
            for m in &musics {
                for t in &texts {
                    for p in &pages {
                        match write_combo_style_with_page(&cfg.out, m.as_deref(), t.as_deref(), p.as_ref()) {
                            Ok(path) => units.push((m.clone(), t.clone(), p.as_ref().map(|x| x.name.clone()), Some(path))),
                            Err(e) => {
                                eprintln!("  FAIL {stem}: style {m:?}/{t:?}/{p:?}: {e}");
                                failed += 1;
                            }
                        }
                    }
                }
            }
        } else {
            units.push((None, None, None, None));
        }
        for (music, text, page_name, style) in &units {
            let mut stem = stem.clone();
            if let Some(m) = music {
                stem.push('_');
                stem.push_str(&slug(m));
            }
            if let Some(t) = text {
                stem.push('_');
                stem.push_str(&slug(t));
            }
            if let Some(pn) = page_name {
                stem.push('_');
                stem.push_str(&slug(pn));
            }
            let job = RenderJob { source: src.clone(), format, stem, style: style.clone() };
            let png_name = format!("{}.png", job.stem);
            let json_name = format!("{}.json", job.stem);
            let png_path = cfg.out.join(&png_name);
            let json_path = cfg.out.join(&json_name);
            let display = src.strip_prefix(&cfg.input).unwrap_or(&src).to_string_lossy().replace('\\', "/");
            if batch_mode {
                pending.push(PendingUnit {
                    job,
                    format,
                    png_name,
                    png_path,
                    json_path,
                    display,
                    music: music.clone(),
                    text: text.clone(),
                });
                continue;
            }
            match renderer.render(&job, &png_path, cfg.width, cfg.height) {
                Ok(outcome) => {
                    if let Err(e) = write_ground_truth(&json_path, &job, &png_name, cfg.width, cfg.height, &outcome, &display, music.as_deref(), text.as_deref()) {
                        eprintln!("  FAIL {}: json write: {e}", job.stem);
                        failed += 1;
                        continue;
                    }
                    println!("  OK {} [{} via {}] notes={}", job.stem, format.as_str(), outcome.backend_used, outcome.note_count);
                    ok += 1;
                }
                Err(e) => {
                    eprintln!("  FAIL {}: {e}", job.stem);
                    failed += 1;
                }
            }
        }
    }
    if batch_mode && !pending.is_empty() {
        // Group by style so one MuseScore launch covers each group.
        let mut groups: Vec<(Option<PathBuf>, Vec<usize>)> = Vec::new();
        for (idx, u) in pending.iter().enumerate() {
            match groups.iter_mut().find(|(s, _)| *s == u.job.style) {
                Some((_, v)) => v.push(idx),
                None => groups.push((u.job.style.clone(), vec![idx])),
            }
        }
        let mut results: Vec<Option<RenderOutcome>> = (0..pending.len()).map(|_| None).collect();
        if let Some(bin) = find_musescore(cfg.musescore_bin.as_deref()) {
            let symbox_bin = find_symbox_extract(cfg.symbox_bin.as_deref());
            for (style, idxs) in &groups {
                let units: Vec<(RenderJob, PathBuf)> = idxs
                    .iter()
                    .map(|&i| (pending[i].job.clone(), pending[i].png_path.clone()))
                    .collect();
                for (k, outcome) in render_batch_musescore(&bin, style.as_deref(), &cfg.out, &units, cfg.musescore_dpi, symbox_bin.as_deref()) {
                    results[idxs[k]] = Some(outcome);
                }
            }
        }
        for (idx, u) in pending.iter().enumerate() {
            let outcome = match results[idx].take() {
                Some(o) => o,
                None => match renderer.render(&u.job, &u.png_path, cfg.width, cfg.height) {
                    Ok(o) => o,
                    Err(e) => {
                        eprintln!("  FAIL {}: {e}", u.job.stem);
                        failed += 1;
                        continue;
                    }
                },
            };
            if let Err(e) = write_ground_truth(
                &u.json_path, &u.job, &u.png_name, cfg.width, cfg.height,
                &outcome, &u.display, u.music.as_deref(), u.text.as_deref(),
            ) {
                eprintln!("  FAIL {}: json write: {e}", u.job.stem);
                failed += 1;
                continue;
            }
            println!("  OK {} [{} via {}] notes={}", u.job.stem, u.format.as_str(), outcome.backend_used, outcome.note_count);
            ok += 1;
        }
    }
    println!("done: {ok} rendered, {skipped} skipped, {failed} failed -> {}", cfg.out.display());
    if failed > 0 {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn style_selects_symbol_and_text_fonts() {
        let s = font_style_mss("Bravura", "Bravura Text");
        assert!(s.contains("<musicalSymbolFont>Bravura</musicalSymbolFont>"));
        assert!(s.contains("<musicalTextFont>Bravura Text</musicalTextFont>"));
    }

    #[test]
    fn font_list_splits_trims_and_drops_empties() {
        assert_eq!(split_fonts("Bravura, Emmentaler ,,Gonville"), vec!["Bravura", "Emmentaler", "Gonville"]);
        assert!(split_fonts("").is_empty());
        assert!(split_fonts("   ").is_empty());
    }

    #[test]
    fn combo_style_covers_symbols_text_and_all_text_keys() {
        let s = combo_style_mss(Some("Bravura"), Some("Edwin"));
        assert!(s.contains("<musicalSymbolFont>Bravura</musicalSymbolFont>"));
        assert!(s.contains("<musicalTextFont>Edwin</musicalTextFont>"));
        assert!(s.contains("<lyricsOddFontFace>Edwin</lyricsOddFontFace>"));
        assert!(s.contains("<titleFontFace>Edwin</titleFontFace>"));
        // No music font: text still restyles everything, symbols untouched.
        let t = combo_style_mss(None, Some("Edwin"));
        assert!(!t.contains("musicalSymbolFont"));
        assert!(t.contains("<dynamicsFontFace>Edwin</dynamicsFontFace>"));
    }

    #[test]
    fn metric_page_sizes_emit_correct_mss_dimensions() {
        let a4 = parse_page_size("a4").unwrap();
        let s = combo_style_mss_with_page(Some("Bravura"), None, Some(&a4));
        assert!(s.contains("<pageWidth>8.2677</pageWidth>"));
        assert!(s.contains("<pageHeight>11.6929</pageHeight>"));
        assert!(!s.contains("<Spatium>"));

        let b4 = parse_page_size("b4").unwrap();
        let sb = combo_style_mss_with_page(None, None, Some(&b4));
        assert!(sb.contains("<pageWidth>9.8425</pageWidth>"));
        assert!(sb.contains("<pageHeight>13.8976</pageHeight>"));
        assert!(sb.contains("<Spatium>1.550</Spatium>"));

        let custom = parse_page_size("210x148").unwrap();
        assert_eq!(custom.name, "210x148mm");
    }

    #[test]
    fn slug_strips_spaces_and_punctuation() {
        assert_eq!(slug("Bravura Text"), "BravuraText");
        assert_eq!(slug("MuseJazz"), "MuseJazz");
    }

    #[test]
    fn batch_diff_excludes_preexisting_files_with_nonempty_before() {
        // Regression pin for the Emmentaler stall: with a populated out dir
        // the per-unit `before.contains` scan turned the batch path-diff
        // quadratic (units x after-dir x before-dir, ~420B compares for the
        // Emmentaler group). The helper takes a prebuilt set so the scan is
        // linear per unit; stale same-stem files stay excluded.
        let dir = std::env::temp_dir().join(format!("score-render-test-diff-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        for n in ["ds0001_Leland.png", "ds0001_Leland-1.png", "other.png"] {
            fs::write(dir.join(n), b"x").unwrap();
        }
        let before: std::collections::HashSet<PathBuf> =
            fs::read_dir(&dir).map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect()).unwrap_or_default();
        let after: Vec<PathBuf> = before.iter().cloned()
            .chain([dir.join("ds0001_Leland-2.png"), dir.join("ds0001_Emmentaler.png")])
            .collect();
        // New page picked up; pre-existing files and other-stem files excluded.
        assert_eq!(new_unit_files(&before, &after, "ds0001_Leland", "png"), vec!["ds0001_Leland-2.png".to_string()]);
        assert_eq!(new_unit_files(&before, &after, "ds0001_Emmentaler", "png"), vec!["ds0001_Emmentaler.png".to_string()]);
        assert!(new_unit_files(&before, &after, "ds0009_Leland", "png").is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn clear_stems_outputs_batch_removes_only_listed_stems() {
        // Batch path must clear every listed stem with a SINGLE directory
        // scan (the old loop called clear_stem_outputs once per stem, i.e.
        // one full 64k-file read_dir per stem). Functional pin: batch result
        // equals repeated single-stem clears, including the ds00010 guard.
        let dir = std::env::temp_dir().join(format!("score-render-test-batch-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        for n in ["ds0001.png", "ds0001-1.png", "ds0001.json", "ds0002.png", "ds0002-3.svg", "ds0002.musicxml", "ds00010-1.png", "other-1.png"] {
            fs::write(dir.join(n), b"x").unwrap();
        }
        clear_stems_outputs(&dir, &["ds0001", "ds0002"]);
        assert!(!dir.join("ds0001.png").exists());
        assert!(!dir.join("ds0001-1.png").exists());
        assert!(!dir.join("ds0001.json").exists());
        assert!(!dir.join("ds0002.png").exists());
        assert!(!dir.join("ds0002-3.svg").exists());
        assert!(!dir.join("ds0002.musicxml").exists());
        assert!(dir.join("ds00010-1.png").exists());
        assert!(dir.join("other-1.png").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn clear_stem_outputs_removes_only_that_stem() {
        let dir = std::env::temp_dir().join(format!("score-render-test-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        for n in ["ds0001.png", "ds0001-1.png", "ds0001.svg", "ds0001-2.svg", "ds0001.musicxml", "ds0001.json", "ds0001.txt", "ds00010-1.png", "ds00010.svg", "other-1.png"] {
            fs::write(dir.join(n), b"x").unwrap();
        }
        clear_stem_outputs(&dir, "ds0001");
        assert!(!dir.join("ds0001.png").exists());
        assert!(!dir.join("ds0001-1.png").exists());
        assert!(!dir.join("ds0001.svg").exists());
        assert!(!dir.join("ds0001-2.svg").exists());
        assert!(!dir.join("ds0001.musicxml").exists());
        assert!(!dir.join("ds0001.json").exists());
        assert!(dir.join("ds0001.txt").exists());
        assert!(dir.join("ds00010-1.png").exists());
        assert!(dir.join("ds00010.svg").exists());
        assert!(dir.join("other-1.png").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    fn require_test_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("score-render-req-{tag}-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    fn require_test_job(dir: &std::path::Path) -> (RenderJob, std::path::PathBuf) {
        // The stub fallback reads the source for a note count, so the
        // dummy input must exist (content irrelevant).
        let _ = std::fs::write(dir.join("x.musicxml"), "<score/>");
        let job = RenderJob {
            source: dir.join("x.musicxml"),
            format: Format::MusicXml,
            stem: "x".to_string(),
            style: None,
        };
        (job, dir.join("x.png"))
    }

    #[test]
    fn require_backend_rejects_stub_fallback_when_binary_missing() {
        // MuseScoreRenderer with bin: None never spawns; with the flag it
        // must Err instead of writing a placeholder.
        let dir = require_test_dir("ms");
        let r = MuseScoreRenderer {
            fallback: StubRenderer,
            bin: None,
            dpi: 300,
            shard_tmp: dir.clone(),
            symbox_bin: None,
            require_backend: true,
        };
        let (job, png) = require_test_job(&dir);
        let err = match r.render(&job, &png, 100, 70) {
            Ok(_) => panic!("require_backend must fail, not stub"),
            Err(e) => e,
        };
        assert!(err.contains("require-backend"), "unexpected error: {err}");
        assert!(!png.exists(), "no placeholder may be written on refusal");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn require_backend_off_keeps_stub_fallback() {
        // Control: same setup without the flag keeps today's behavior.
        let dir = require_test_dir("ms-off");
        let r = MuseScoreRenderer {
            fallback: StubRenderer,
            bin: None,
            dpi: 300,
            shard_tmp: dir.clone(),
            symbox_bin: None,
            require_backend: false,
        };
        let (job, png) = require_test_job(&dir);
        let o = r.render(&job, &png, 100, 70).expect("stub fallback");
        assert!(o.backend_used.starts_with("stub"), "got {}", o.backend_used);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn require_backend_rejects_stub_fallback_for_lilypond() {
        // The lilypond backend always stubs (no converter wired), so the
        // flag must fail it on every machine, binary or not.
        let dir = require_test_dir("ly");
        let r = LilyPondRenderer { fallback: StubRenderer, require_backend: true };
        let (job, png) = require_test_job(&dir);
        let err = match r.render(&job, &png, 100, 70) {
            Ok(_) => panic!("require_backend must fail, not stub"),
            Err(e) => e,
        };
        assert!(err.contains("require-backend"), "unexpected error: {err}");
        assert!(!png.exists(), "no placeholder may be written on refusal");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn page_files_pair_by_trailing_page_number() {
        let pngs = vec!["s-2.png".to_string(), "s-1.png".to_string(), "s.png".to_string()];
        let svgs = vec!["s-1.svg".to_string(), "s-2.svg".to_string(), "orphan.svg".to_string()];
        let pairs = pair_page_files(&pngs, &svgs);
        assert_eq!(
            pairs,
            vec![
                ("s-1.png".to_string(), "s-1.svg".to_string(), 1),
                ("s-2.png".to_string(), "s-2.svg".to_string(), 2),
            ]
        );
    }

    #[test]
    fn bare_stem_pages_pair_at_page_one() {
        let pairs = pair_page_files(&["s.png".to_string()], &["s.svg".to_string()]);
        assert_eq!(pairs, vec![("s.png".to_string(), "s.svg".to_string(), 1)]);
    }

    const SYMBOX_SAMPLE: &str = r#"{
  "page": 1,
  "svg_view": [100.0, 200.0],
  "png_size": [25, 50],
  "scale": [0.25, 0.25],
  "staves": [],
  "glyphs": [
    {
      "cls": "NoteBlack",
      "bbox": [
        1.5,
        2.5,
        3.5,
        4.5
      ]
    },
    {
      "cls": "Stem",
      "bbox": [5.0, 6.0, 7.0, 8.0]
    }
  ]
}"#;

    #[test]
    fn symbox_glyphs_parse_cls_and_bbox() {
        let g = parse_symbox_glyphs(SYMBOX_SAMPLE).unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].0, "NoteBlack");
        assert_eq!(g[0].1, [1.5, 2.5, 3.5, 4.5]);
        assert_eq!(g[1].0, "Stem");
    }

    #[test]
    fn symbox_glyphs_reject_malformed() {
        assert!(parse_symbox_glyphs(r#"{"glyphs": [{"cls": "X"}]}"#).is_err());
        assert!(parse_symbox_glyphs("not json at all").is_err());
        assert!(parse_symbox_glyphs(r#"{"page": 1}"#).is_err());
    }

    #[test]
    fn symbols_json_round_trips_records() {
        assert_eq!(symbols_json(&[]), "[]");
        let recs = vec![
            SymbolRec { page: 1, cls: "NoteBlack".to_string(), bbox: [1.5, 2.5, 3.5, 4.5] },
            SymbolRec { page: 2, cls: "Stem".to_string(), bbox: [5.0, 6.0, 7.0, 8.0] },
        ];
        let j = symbols_json(&recs);
        assert!(j.starts_with('[') && j.ends_with(']'));
        assert!(j.contains("\"page\": 1") && j.contains("\"cls\": \"NoteBlack\""));
        assert!(j.contains("\"page\": 2") && j.contains("\"cls\": \"Stem\""));
        // The fragment must re-parse as glyph records.
        let back = parse_symbox_glyphs(&format!("{{\"glyphs\": {j}}}")).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].0, "NoteBlack");
    }

    #[test]
    fn job_file_lists_each_input_once_with_png_musicxml_and_svg_outputs() {
        let entries = vec![
            ("C:/x/a.mscx".to_string(), "C:/o/a.png".to_string(), "C:/o/a.musicxml".to_string(), "C:/o/a.svg".to_string()),
            ("C:/x/b.mscx".to_string(), "C:/o/b.png".to_string(), "C:/o/b.musicxml".to_string(), "C:/o/b.svg".to_string()),
        ];
        let j = job_file_json(&entries);
        assert!(j.starts_with("[\n"));
        assert!(j.ends_with("]"));
        assert!(j.contains("\"in\": \"C:/x/a.mscx\""));
        assert!(j.contains("\"out\": [\"C:/o/b.png\", \"C:/o/b.musicxml\", \"C:/o/b.svg\"]"));
        assert_eq!(j.matches("\"in\":").count(), 2);
    }
}
