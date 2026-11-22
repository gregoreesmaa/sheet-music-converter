//! symbox-extract CLI: MuseScore SVG page -> glyph/staff boxes JSON.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use symbox_extract::{
    classify_note, group_staves, parse_svg_elems, path_bbox_affine, polyline_bbox_affine,
    relabel_flags, staff_space, svg_viewbox, GlyphBox, StaffGeom, IDENTITY,
};

#[derive(Parser, Debug)]
#[command(
    name = "symbox-extract",
    version,
    about = "Extract glyph/staff boxes from a MuseScore SVG page, scaled to the sister PNG"
)]
struct Cli {
    /// MuseScore SVG page input.
    #[arg(long)]
    svg: PathBuf,
    /// Sister PNG (only its pixel dimensions are read).
    #[arg(long)]
    png: PathBuf,
    /// Output boxes JSON path.
    #[arg(long)]
    out: PathBuf,
    /// 1-based page number recorded in the output.
    #[arg(long, default_value_t = 1)]
    page: u32,
}

type Err = Box<dyn std::error::Error>;

fn run(cli: &Cli) -> Result<(), Err> {
    let svg =
        fs::read_to_string(&cli.svg).map_err(|e| format!("read {}: {e}", cli.svg.display()))?;
    // Header-only: full 139 MPix pages trip the decoder memory limit.
    let (pw, ph) = image::image_dimensions(&cli.png)
        .map_err(|e| format!("read {}: {e}", cli.png.display()))?;
    let [vw, vh] = svg_viewbox(&svg).map_err(|e| format!("{}: {e}", cli.svg.display()))?;
    let (sx, sy) = (pw as f64 / vw, ph as f64 / vh);
    let mut rows: Vec<(f64, f64, f64)> = Vec::new();
    let mut glyphs: Vec<GlyphBox> = Vec::new();
    for el in parse_svg_elems(&svg).map_err(|e| format!("{}: {e}", cli.svg.display()))? {
        if el.is_polyline {
            if el.cls == "Stem" || el.cls == "BarLine" {
                // Note stems and barlines are strokes, not fills: the box
                // is the endpoint span expanded by half the stroke width.
                // (StaffLines stay centerlines for staff grouping below.)
                let m = el.transform.unwrap_or(IDENTITY);
                let stroke = el.stroke.ok_or(format!(
                    "{}: {} without stroke-width",
                    cli.svg.display(),
                    el.cls
                ))?;
                let b = polyline_bbox_affine(&el.data, stroke, m)
                    .map_err(|e| format!("{}: {e}", cli.svg.display()))?;
                glyphs.push(GlyphBox {
                    cls: el.cls.clone(),
                    bbox: [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy],
                });
            }
            if el.cls == "StaffLines" {
                let pts: Vec<f64> = el
                    .data
                    .split([',', ' ', '\t', '\n', '\r'])
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        s.parse::<f64>()
                            .map_err(|_| format!("bad StaffLines point {s:?}"))
                    })
                    .collect::<Result<_, _>>()?;
                if !pts.len().is_multiple_of(2) {
                    return Err("odd StaffLines point count".to_string().into());
                }
                for w in pts.as_chunks::<4>().0 {
                    // Horizontal staff line from (x0, y) to (x1, y).
                    rows.push((w[0] * sx, w[2] * sx, w[1] * sy));
                }
            }
            continue;
        }
        let m = el.transform.unwrap_or(IDENTITY);
        let b = path_bbox_affine(&el.data, m).map_err(|e| format!("{}: {e}", cli.svg.display()))?;
        // Open noteheads carry an inner counter subpath: split them off so
        // the detector learns the duration-critical black/void bit.
        let cls = if el.cls == "Note" {
            classify_note(&el.data)
                .map_err(|e| format!("{}: {e}", cli.svg.display()))?
                .to_string()
        } else {
            el.cls
        };
        glyphs.push(GlyphBox {
            cls,
            bbox: [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy],
        });
    }
    let staves: Vec<StaffGeom> = group_staves(&rows)
        .into_iter()
        .map(|s| StaffGeom {
            lines: s.lines,
            bbox: s.bbox,
        })
        .collect();
    // Beam-distant, stem-near Hooks are flags, not beamlets.
    if let Some(sp) = staff_space(&staves) {
        relabel_flags(&mut glyphs, sp);
    }
    let doc = symbox_extract::BoxesDoc {
        page: cli.page,
        svg_view: [vw, vh],
        png_size: [pw, ph],
        scale: [sx, sy],
        staves,
        glyphs,
    };
    if let Some(dir) = cli.out.parent() {
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        }
    }
    fs::write(&cli.out, serde_json::to_string_pretty(&doc)?)
        .map_err(|e| format!("write {}: {e}", cli.out.display()))?;
    println!(
        "symbox-extract: page {} staves={} glyphs={} scale=({:.4},{:.4}) -> {}",
        doc.page,
        doc.staves.len(),
        doc.glyphs.len(),
        sx,
        sy,
        cli.out.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("symbox-extract: error: {e}");
            ExitCode::FAILURE
        }
    }
}
