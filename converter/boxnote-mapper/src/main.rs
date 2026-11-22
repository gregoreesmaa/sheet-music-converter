//! boxnote-mapper CLI: glyph boxes/labels + staff layout -> note-list JSON.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use boxnote_mapper::{map_naive, map_pianoform, map_voiced, GlyphDoc, LayoutDoc};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "boxnote-mapper",
    version,
    about = "Map glyph boxes/labels plus staff layout to a note-list JSON file"
)]
struct Cli {
    /// symbol-detect glyphs JSON.
    #[arg(long)]
    glyphs: PathBuf,
    /// Optional second glyphs JSON (aux detector: stems/beams/dots for the
    /// `voiced` backend, joined on exact bbox). Defaults to empty.
    #[arg(long)]
    glyphs_aux: Option<PathBuf>,
    /// layout-staff layout JSON (`*_layout.json`).
    #[arg(long)]
    layout: PathBuf,
    /// Output note-list JSON path.
    #[arg(long)]
    out: PathBuf,
    /// Staff spacing fallback when a staff carries no line positions.
    #[arg(long, default_value_t = 10.0)]
    staff_spacing: f64,
    /// Part name for the single output part.
    #[arg(long, default_value = "Transcription")]
    part_name: String,
    /// Backend selector (`naive`, `pianoform`, `voiced`; learned mappers
    /// plug in here).
    #[arg(long, default_value = "naive")]
    backend: String,
}

type Err = Box<dyn std::error::Error>;

fn read_json<T>(path: &Path) -> Result<T, Err>
where
    T: serde::de::DeserializeOwned,
{
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?)
}

fn run(cli: &Cli) -> Result<(), Err> {
    let glyphs: GlyphDoc = read_json(&cli.glyphs)?;
    let layout: LayoutDoc = read_json(&cli.layout)?;
    if cli.backend == "voiced" {
        let aux: GlyphDoc = match &cli.glyphs_aux {
            Some(p) => read_json(p)?,
            None => serde_json::from_value(serde_json::json!({"crops": []}))?,
        };
        let parts = map_voiced(&glyphs, &aux, &layout, cli.staff_spacing);
        let json_parts: Vec<serde_json::Value> = parts
            .iter()
            .map(|part| {
                let notes: Vec<serde_json::Value> = part
                    .notes
                    .iter()
                    .map(|n| {
                        let mut v = if let Some(p) = &n.pitch {
                            serde_json::json!({"pitch": p})
                        } else {
                            serde_json::json!({"rest": true})
                        };
                        v["onset"] = serde_json::json!(n.onset);
                        v["dur"] = serde_json::json!(n.dur);
                        v["voice"] = serde_json::json!(n.voice);
                        v
                    })
                    .collect();
                serde_json::json!({"name": part.name, "notes": notes})
            })
            .collect();
        let total: usize = parts.iter().map(|p| p.notes.len()).sum();
        let doc = serde_json::json!({
            "title": "Transcription",
            "divisions": 480,
            "parts": json_parts,
        });
        write_doc(cli, &doc, total)?;
        return Ok(());
    }
    let mapped = match cli.backend.as_str() {
        "naive" => map_naive(&glyphs, &layout, cli.staff_spacing),
        "pianoform" => map_pianoform(&glyphs, &layout, cli.staff_spacing),
        other => return Err(format!("unknown backend {other:?}").into()),
    };
    let notes: Vec<serde_json::Value> = mapped
        .iter()
        .map(|n| {
            if let Some(p) = &n.pitch {
                serde_json::json!({"pitch": p, "onset": n.onset, "dur": n.dur})
            } else {
                serde_json::json!({"rest": true, "onset": n.onset, "dur": n.dur})
            }
        })
        .collect();
    let doc = serde_json::json!({
        "title": "Transcription",
        "divisions": 480,
        "parts": [{"name": cli.part_name, "notes": notes}],
    });
    write_doc(cli, &doc, notes.len())?;
    Ok(())
}

fn write_doc(cli: &Cli, doc: &serde_json::Value, total: usize) -> Result<(), Err> {
    if let Some(dir) = cli.out.parent() {
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        }
    }
    fs::write(&cli.out, serde_json::to_string_pretty(&doc)?)
        .map_err(|e| format!("write {}: {e}", cli.out.display()))?;
    println!(
        "boxnote-mapper: backend={} notes={} -> {}",
        cli.backend,
        total,
        cli.out.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("boxnote-mapper: error: {e}");
            ExitCode::FAILURE
        }
    }
}
