//! eval-harness: in-tree OMR evaluation harness (Phase 0).
//!
//! `symbox-adapter` turns renderer symbol boxes (`*-boxes.json` from
//! `data-gen/symbox-extract`) into oracle [`GlyphDocOut`] primary/aux docs
//! plus an oracle [`LayoutDocOut`] for `boxnote-mapper --backend voiced`.
//! Deterministic: input order preserved, no RNG.

use std::fs;
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod fixes;
mod notation;
mod staffmap;
mod strict;
mod strict_xml;
mod summary;

#[derive(Parser, Debug)]
#[command(
    name = "eval-harness",
    version,
    about = "In-tree OMR evaluation harness (oracle docs + strict-F1 scorer)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// symbox `*-boxes.json` -> oracle primary/aux/layout docs.
    SymboxAdapter(SymboxAdapterArgs),
    /// Strict note-F1 (pitch + onset-in-bar + duration) of hyp vs ref.
    StrictF1(StrictF1Args),
    /// layout-staff output -> single-entry symbol-detect staff map.
    Staffmap(StaffmapArgs),
    /// Reference-exact strict pair score of two MusicXML files.
    StrictXml(StrictXmlArgs),
    /// Notation-as-written score of two MusicXML files: spelling-exact
    /// (B# is NOT C), staff/voice/rests/tuplets/attachments priced. The
    /// replacement ruler; strict-xml stays frozen as the reference port.
    NotationF1(StrictXmlArgs),
    /// Aggregate per-piece reports into a suite summary (mean + median +
    /// usable-output rate, strict-F1 and OMR-NED side by side).
    SuiteSummary(SuiteSummaryArgs),
    /// Measure-atomic correction gestures for a ground-truth piece
    /// against its top-2 review lattice (0 = top-1 exact, 1 = accept
    /// candidate #2, else note-level token edits).
    #[command(name = "fixes-estimate")]
    FixesEstimate(FixesEstimateArgs),
}

#[derive(Args, Debug)]
struct SymboxAdapterArgs {
    /// symbox `*-boxes.json` input.
    #[arg(long)]
    boxes: PathBuf,
    /// Primary GlyphDoc output (heads + rests).
    #[arg(long)]
    primary: PathBuf,
    /// Aux GlyphDoc output (beam/flag/dot/accidental/clef evidence).
    #[arg(long)]
    aux: PathBuf,
    /// LayoutDoc output (oracle staves grouped into systems).
    #[arg(long)]
    layout: PathBuf,
}

#[derive(Args, Debug)]
struct StrictF1Args {
    /// Reference (ground-truth) note-list JSON.
    #[arg(long)]
    reference: PathBuf,
    /// Hypothesis note-list JSON to score.
    #[arg(long)]
    hypothesis: PathBuf,
    /// Where to write the JSON F1 report.
    #[arg(long)]
    report: Option<PathBuf>,
    /// Reference OMR-NED for the same pair (pinned musicdiff value) to
    /// record alongside strict-F1. Never scored here, only attached.
    #[arg(long)]
    omr_ned: Option<f64>,
}

#[derive(Args, Debug)]
struct StrictXmlArgs {
    /// Ground-truth MusicXML.
    #[arg(long)]
    reference: PathBuf,
    /// Hypothesis MusicXML.
    #[arg(long)]
    hypothesis: PathBuf,
    /// Keep only ground-truth measures up to this ordinal (page window).
    #[arg(long)]
    max_measures: Option<u32>,
    /// Where to write the JSON pair report.
    #[arg(long)]
    report: Option<PathBuf>,
    /// Reference OMR-NED for the same pair (pinned musicdiff value) to
    /// record alongside strict-F1. Never scored here, only attached.
    #[arg(long)]
    omr_ned: Option<f64>,
}

#[derive(Args, Debug)]
struct SuiteSummaryArgs {
    /// Suite manifest JSON (expected pieces + report paths).
    #[arg(long)]
    manifest: PathBuf,
    /// Where to write the JSON suite summary.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct FixesEstimateArgs {
    /// Producer lattice file (`<stem>_lattice.json`).
    #[arg(long)]
    lattice: PathBuf,
    /// Ground-truth bekern token file (whitespace-separated, one page;
    /// measures split on `bar` / `bar-*` tokens).
    #[arg(long)]
    ground_truth: PathBuf,
    /// Where to write the JSON estimate report (stdout if omitted).
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct StaffmapArgs {
    /// layout-staff `*_layout.json` input.
    #[arg(long)]
    layout: PathBuf,
    /// Staff-map JSON output (single global entry, union of all rows).
    #[arg(long)]
    out: PathBuf,
}

// ---------------------------------------------------------------------------
// Input model (symbox-extract `*-boxes.json`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct SymStaff {
    #[serde(default)]
    lines: Vec<f64>,
    #[serde(default)]
    bbox: [f64; 4],
}

#[derive(Debug, Clone, Deserialize)]
struct SymGlyph {
    #[serde(default)]
    cls: String,
    #[serde(default)]
    bbox: [f64; 4],
}

#[derive(Debug, Clone, Deserialize)]
struct BoxesDoc {
    #[serde(default)]
    staves: Vec<SymStaff>,
    #[serde(default)]
    glyphs: Vec<SymGlyph>,
}

// ---------------------------------------------------------------------------
// Output models (boxnote-mapper GlyphDoc / LayoutDoc compatible)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BBox {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GlyphOut {
    bbox: BBox,
    label: String,
    confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CropOut {
    crop_id: String,
    glyphs: Vec<GlyphOut>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GlyphDocOut {
    backend: String,
    crops: Vec<CropOut>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StaffOut {
    bbox: [f64; 4],
    lines: Vec<f64>,
    slope: f64,
    pivot: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SystemOut {
    staves: Vec<StaffOut>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LayoutDocOut {
    systems: Vec<SystemOut>,
}

// ---------------------------------------------------------------------------
// Mapping (stubs: real implementation lands after RED)
// ---------------------------------------------------------------------------

/// Primary-doc label for a symbox class, if it carries note events.
/// Mirrors the old 3-class oracle primary (heads + rests, no beams).
fn primary_label(cls: &str) -> Option<&'static str> {
    match cls {
        "Note" => Some("notehead-black"),
        "Rest" => Some("rest"),
        _ => None,
    }
}

/// Aux-doc (evidence) label for a symbox class, if the mapper consumes it.
fn aux_label(cls: &str) -> Option<&'static str> {
    match cls {
        "Beam" => Some("beam"),
        "Hook" => Some("flag"),
        "NoteDot" => Some("dot"),
        "Accidental" => Some("accidental"),
        "Clef" => Some("clef"),
        _ => None,
    }
}

/// Median staff-line spacing across all staves (fallback 16.0).
fn median_line_spacing(staves: &[SymStaff]) -> f64 {
    let mut diffs: Vec<f64> = Vec::new();
    for st in staves {
        for w in st.lines.windows(2) {
            let d = w[1] - w[0];
            if d > 0.0 {
                diffs.push(d);
            }
        }
    }
    if diffs.is_empty() {
        return 16.0;
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = diffs.len() / 2;
    if diffs.len() % 2 == 1 {
        diffs[m]
    } else {
        (diffs[m - 1] + diffs[m]) / 2.0
    }
}

/// Group stave indices into systems: y-sorted, split when the gap exceeds
/// 10x the median line spacing.
fn group_systems(staves: &[SymStaff]) -> Vec<Vec<usize>> {
    let mut idx: Vec<usize> = (0..staves.len()).collect();
    idx.sort_by(|&a, &b| staves[a].bbox[1].partial_cmp(&staves[b].bbox[1]).unwrap());
    let med = median_line_spacing(staves);
    let mut systems: Vec<Vec<usize>> = Vec::new();
    for i in idx {
        let split = match systems.last() {
            None => true,
            Some(sys) => {
                let prev = sys.last().unwrap();
                staves[i].bbox[1] - staves[*prev].bbox[3] > 10.0 * med
            }
        };
        if split {
            systems.push(vec![i]);
        } else {
            systems.last_mut().unwrap().push(i);
        }
    }
    systems
}

fn glyph_out(cls_box: &[f64; 4], label: &'static str) -> GlyphOut {
    GlyphOut {
        bbox: BBox {
            x: cls_box[0],
            y: cls_box[1],
            w: cls_box[2] - cls_box[0],
            h: cls_box[3] - cls_box[1],
        },
        label: label.to_string(),
        confidence: 1.0,
    }
}

/// Convert one boxes document into (primary, aux, layout) oracle docs.
fn convert(doc: &BoxesDoc) -> (GlyphDocOut, GlyphDocOut, LayoutDocOut) {
    let mut primary_glyphs: Vec<GlyphOut> = Vec::new();
    let mut aux_glyphs: Vec<GlyphOut> = Vec::new();
    for g in &doc.glyphs {
        if let Some(label) = primary_label(&g.cls) {
            primary_glyphs.push(glyph_out(&g.bbox, label));
        } else if let Some(label) = aux_label(&g.cls) {
            aux_glyphs.push(glyph_out(&g.bbox, label));
        }
    }
    let primary = GlyphDocOut {
        backend: "symbox-oracle".to_string(),
        crops: vec![CropOut {
            crop_id: "page".to_string(),
            glyphs: primary_glyphs,
        }],
    };
    let aux = GlyphDocOut {
        backend: "symbox-oracle-aux".to_string(),
        crops: vec![CropOut {
            crop_id: "page".to_string(),
            glyphs: aux_glyphs,
        }],
    };
    let systems = group_systems(&doc.staves)
        .into_iter()
        .map(|sys| SystemOut {
            staves: sys
                .into_iter()
                .map(|i| StaffOut {
                    bbox: doc.staves[i].bbox,
                    lines: doc.staves[i].lines.clone(),
                    slope: 0.0,
                    pivot: 0.0,
                })
                .collect(),
        })
        .collect();
    (primary, aux, LayoutDocOut { systems })
}

type Err = Box<dyn std::error::Error>;

fn write_file(path: &PathBuf, bytes: &[u8]) -> Result<(), Err> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        }
    }
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(())
}

fn run(cli: &Cli) -> Result<(), Err> {
    match &cli.cmd {
        Cmd::SymboxAdapter(a) => run_adapter(a),
        Cmd::StrictF1(a) => run_strict_f1(a),
        Cmd::Staffmap(a) => run_staffmap(a),
        Cmd::StrictXml(a) => run_strict_xml(a),
        Cmd::NotationF1(a) => run_notation_f1(a),
        Cmd::SuiteSummary(a) => run_suite_summary(a),
        Cmd::FixesEstimate(a) => run_fixes_estimate(a),
    }
}

/// Read a staged input file, returning its bytes. Callers hash the bytes
/// (SHA-256 staging log) and parse from the same buffer.
fn read_staged(path: &PathBuf) -> Result<Vec<u8>, Err> {
    Ok(fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?)
}

/// SHA-256 hex of staged input bytes.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Dual-metric trailer shared by both scoring runs: the OMR-NED value when
/// the driver supplied one, plus full SHA-256 hashes of the staged inputs.
/// Appended after the historic one-line format, which is left byte-identical
/// so existing single-metric parsers keep matching.
fn dual_trailer(omr_ned: Option<f64>, ref_sha: &str, hyp_sha: &str) -> String {
    let ned = omr_ned
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "n/a".to_string());
    format!("omr-ned {ned} sha256 ref={ref_sha} hyp={hyp_sha}")
}

fn parse_score_input(path: &Path, bytes: &[u8]) -> Result<assembly_serialize::ScoreInput, Err> {
    Ok(serde_json::from_slice(bytes).map_err(|e| format!("parse {}: {e}", path.display()))?)
}

/// Empty-reference report: the exact-match column. Written instead of an
/// [`strict::F1Report`] when the reference holds no pitched notes, so no
/// bare F1 float can leak into suite medians. The historic `F1 ...` line
/// is deliberately NOT printed here — single-metric parsers keyed on it
/// must not match an empty page.
#[derive(Debug, Clone, Serialize)]
struct EmptyRefReport {
    empty_ref: bool,
    exact_match: f64,
    hyp_empty: bool,
    ref_len: usize,
    hyp_len: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    f1_zero_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    omr_ned: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hypothesis_sha256: Option<String>,
}

fn write_empty_report(
    args_report: &Option<PathBuf>,
    exact_match: f64,
    hyp_empty: bool,
    hyp_len: usize,
    reason: Option<String>,
    omr_ned: Option<f64>,
    ref_sha: &str,
    hyp_sha: &str,
) -> Result<(), Err> {
    println!(
        "EMPTY exact_match {:.0} (empty reference, {} hyp notes{}) {}",
        exact_match,
        hyp_len,
        reason
            .as_ref()
            .map(|r| format!("; {r}"))
            .unwrap_or_default(),
        dual_trailer(omr_ned, ref_sha, hyp_sha),
    );
    if let Some(p) = args_report {
        let rep = EmptyRefReport {
            empty_ref: true,
            exact_match,
            hyp_empty,
            ref_len: 0,
            hyp_len,
            f1_zero_reason: reason,
            omr_ned,
            reference_sha256: Some(ref_sha.to_string()),
            hypothesis_sha256: Some(hyp_sha.to_string()),
        };
        write_file(p, serde_json::to_string_pretty(&rep)?.as_bytes())?;
    }
    Ok(())
}

fn run_strict_f1(args: &StrictF1Args) -> Result<(), Err> {
    let ref_bytes = read_staged(&args.reference)?;
    let hyp_bytes = read_staged(&args.hypothesis)?;
    let reference = parse_score_input(&args.reference, &ref_bytes)?;
    let hypothesis = parse_score_input(&args.hypothesis, &hyp_bytes)?;
    let ref_sha = sha256_hex(&ref_bytes);
    let hyp_sha = sha256_hex(&hyp_bytes);
    // EMPTY gate: input-level emptiness or a parsed score with zero pitched
    // notes (rest-only references are notation-free too). Routes to the
    // exact-match column and exits 0 — never a CLI failure, never an F1.
    let ref_parsed = if assembly_serialize::is_empty_score_input(&reference) {
        None
    } else {
        Some(
            assembly_serialize::parse_score(&reference)
                .map_err(|e| format!("reference: {e}"))?,
        )
    };
    let ref_empty = ref_parsed
        .as_ref()
        .map_or(true, |(s, _)| {
            assembly_serialize::score_pitched_count(s) == 0
        });
    if ref_empty {
        let (hyp_len, hyp_empty, reason): (usize, bool, Option<String>) =
            if assembly_serialize::is_empty_score_input(&hypothesis) {
                (0, true, None)
            } else {
                match assembly_serialize::parse_score(&hypothesis) {
                    Ok((h, _)) => {
                        let n = assembly_serialize::score_pitched_count(&h);
                        (n, n == 0, None)
                    }
                    // Undecodable hypothesis content is not an empty
                    // prediction: exact 0 with the reason attached.
                    Err(e) => (0, false, Some(format!("hypothesis: {e}"))),
                }
            };
        return write_empty_report(
            &args.report,
            if hyp_empty { 1.0 } else { 0.0 },
            hyp_empty,
            hyp_len,
            reason,
            args.omr_ned,
            &ref_sha,
            &hyp_sha,
        );
    }
    let (ref_score, _) = ref_parsed.unwrap();
    // A non-empty reference against an empty hypothesis stays included as
    // F1 = 0.0 (an empty shell scores zero pitched notes, never an error);
    // any other hypothesis failure is still a hard error.
    let hyp_score = if assembly_serialize::is_empty_score_input(&hypothesis) {
        assembly_serialize::Score {
            title: ref_score.title.clone(),
            composer: ref_score.composer.clone(),
            divisions: ref_score.divisions,
            beats: ref_score.beats,
            beat_type: ref_score.beat_type,
            key_fifths: ref_score.key_fifths,
            parts: Vec::new(),
            measure_attrs: Vec::new(),
        }
    } else {
        assembly_serialize::parse_score(&hypothesis)
            .map_err(|e| format!("hypothesis: {e}"))?
            .0
    };
    let mut report = match strict::strict_f1(&ref_score, &hyp_score) {
        strict::StrictOutcome::Scored(r) => r,
        strict::StrictOutcome::EmptyReference {
            hyp_len,
            hyp_empty,
            exact_match,
        } => {
            return Err(format!(
                "internal: empty reference past the emptiness gate \
                 (hyp_len={hyp_len} hyp_empty={hyp_empty} exact={exact_match})"
            )
            .into());
        }
    };
    report.omr_ned = args.omr_ned;
    report.reference_sha256 = Some(ref_sha.clone());
    report.hypothesis_sha256 = Some(hyp_sha.clone());
    println!(
        "F1 {:.4} ({} matches / {} ref notes, {} hyp notes; P {:.4} R {:.4}) {}",
        report.f1,
        report.matches,
        report.ref_len,
        report.hyp_len,
        report.precision,
        report.recall,
        dual_trailer(args.omr_ned, &ref_sha, &hyp_sha),
    );
    if let Some(p) = &args.report {
        write_file(p, serde_json::to_string_pretty(&report)?.as_bytes())?;
    }
    Ok(())
}

fn run_staffmap(args: &StaffmapArgs) -> Result<(), Err> {
    let text = fs::read_to_string(&args.layout)
        .map_err(|e| format!("read {}: {e}", args.layout.display()))?;
    let out = staffmap::convert_str(&text)?;
    write_file(&args.out, out.as_bytes())?;
    println!("wrote {}", args.out.display());
    Ok(())
}

/// Split a bekern token stream into per-measure token strings.
/// Measures are separated by `bar` / `bar-*` tokens; separators never
/// appear inside a measure, and trailing tokens after the last bar
/// form the final (possibly incomplete pickup) measure.
fn split_bekern_measures(tokens: &str) -> Vec<String> {
    let mut measures = vec![String::new()];
    for tok in tokens.split_whitespace() {
        if tok == "bar" || tok.starts_with("bar-") {
            measures.push(String::new());
        } else if let Some(cur) = measures.last_mut() {
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(tok);
        }
    }
    measures.into_iter().filter(|m| !m.trim().is_empty()).collect()
}

fn run_fixes_estimate(args: &FixesEstimateArgs) -> Result<(), Err> {
    let lat_text = fs::read_to_string(&args.lattice)
        .map_err(|e| format!("read {}: {e}", args.lattice.display()))?;
    let lattice: assembly_serialize::lattice::LatticeDoc =
        serde_json::from_str(&lat_text)
            .map_err(|e| format!("parse {}: {e}", args.lattice.display()))?;
    let gt_text = fs::read_to_string(&args.ground_truth).map_err(|e| {
        format!("read {}: {e}", args.ground_truth.display())
    })?;
    let gt_measures = split_bekern_measures(&gt_text);
    let est = fixes::fixes_estimate(&gt_measures, &lattice)
        .map_err(|e| format!("fixes-estimate: {e}"))?;
    let report = serde_json::to_string_pretty(&est)?;
    if let Some(p) = &args.report {
        write_file(p, report.as_bytes())?;
    }
    println!("fixes-estimate total={} per_measure={:?}", est.total, est.per_measure);
    Ok(())
}

fn run_strict_xml(args: &StrictXmlArgs) -> Result<(), Err> {
    let ref_bytes = read_staged(&args.reference)?;
    let hyp_bytes = read_staged(&args.hypothesis)?;
    let gt_xml = String::from_utf8(ref_bytes.clone())
        .map_err(|e| format!("read {}: {e}", args.reference.display()))?;
    let hyp_xml = String::from_utf8(hyp_bytes.clone())
        .map_err(|e| format!("read {}: {e}", args.hypothesis.display()))?;
    let gt = assembly_serialize::musicxml_import::score_events(&gt_xml)
        .map_err(|e| format!("reference events: {e}"))?;
    let gt: Vec<_> = match args.max_measures {
        Some(n) => gt.into_iter().filter(|e| e.measure <= n).collect(),
        None => gt,
    };
    let ref_sha = sha256_hex(&ref_bytes);
    let hyp_sha = sha256_hex(&hyp_bytes);
    // EMPTY gate, mirroring run_strict_f1: empty ground truth routes to
    // the exact-match column (exit 0). An unparseable hypothesis against
    // an empty ground truth is not an empty prediction (exact 0 + reason).
    if gt.is_empty() {
        let hyp_events = assembly_serialize::musicxml_import::score_events(&hyp_xml);
        let (hyp_len, hyp_empty, reason): (usize, bool, Option<String>) =
            match hyp_events {
                Ok(h) => (h.len(), h.is_empty(), None),
                Err(e) => (0, false, Some(format!("hypothesis events: {e}"))),
            };
        return write_empty_report(
            &args.report,
            if hyp_empty { 1.0 } else { 0.0 },
            hyp_empty,
            hyp_len,
            reason,
            args.omr_ned,
            &ref_sha,
            &hyp_sha,
        );
    }
    let hyp = assembly_serialize::musicxml_import::score_events(&hyp_xml)
        .map_err(|e| format!("hypothesis events: {e}"))?;
    let mut report = match strict_xml::score_pair(&gt, &hyp) {
        strict_xml::PairOutcome::Scored(r) => r,
        strict_xml::PairOutcome::EmptyReference {
            omr_len,
            hyp_empty,
            exact_match,
        } => {
            return Err(format!(
                "internal: empty ground truth past the emptiness gate \
                 (omr_len={omr_len} hyp_empty={hyp_empty} exact={exact_match})"
            )
            .into());
        }
    };
    report.omr_ned = args.omr_ned;
    report.reference_sha256 = Some(ref_sha.clone());
    report.hypothesis_sha256 = Some(hyp_sha.clone());
    println!(
        "strict-F1 {:.3} ({} / {} ref, {} hyp; note-F1 {:.3} contour {:.3} measure-exact {:?}) {}",
        report.strict_f1,
        report.strict_matched,
        report.gt_notes,
        report.omr_notes,
        report.note_f1,
        report.contour_f1,
        report.measure_exact,
        dual_trailer(args.omr_ned, &ref_sha, &hyp_sha),
    );
    if let Some(p) = &args.report {
        write_file(p, serde_json::to_string_pretty(&report)?.as_bytes())?;
    }
    Ok(())
}

fn run_notation_f1(args: &StrictXmlArgs) -> Result<(), Err> {
    let ref_bytes = read_staged(&args.reference)?;
    let hyp_bytes = read_staged(&args.hypothesis)?;
    let gt_xml = String::from_utf8(ref_bytes.clone())
        .map_err(|e| format!("read {}: {e}", args.reference.display()))?;
    let hyp_xml = String::from_utf8(hyp_bytes.clone())
        .map_err(|e| format!("read {}: {e}", args.hypothesis.display()))?;
    let gt = assembly_serialize::musicxml_import::notation_events(&gt_xml)
        .map_err(|e| format!("reference events: {e}"))?;
    let gt: Vec<_> = match args.max_measures {
        Some(n) => gt.into_iter().filter(|e| e.measure <= n).collect(),
        None => gt,
    };
    let ref_sha = sha256_hex(&ref_bytes);
    let hyp_sha = sha256_hex(&hyp_bytes);
    if gt.is_empty() {
        let hyp_events = assembly_serialize::musicxml_import::notation_events(&hyp_xml);
        let (hyp_len, hyp_empty, reason): (usize, bool, Option<String>) =
            match hyp_events {
                Ok(h) => (h.len(), h.is_empty(), None),
                Err(e) => (0, false, Some(format!("hypothesis events: {e}"))),
            };
        return write_empty_report(
            &args.report,
            if hyp_empty { 1.0 } else { 0.0 },
            hyp_empty,
            hyp_len,
            reason,
            args.omr_ned,
            &ref_sha,
            &hyp_sha,
        );
    }
    let hyp = assembly_serialize::musicxml_import::notation_events(&hyp_xml)
        .map_err(|e| format!("hypothesis events: {e}"))?;
    let mut report = match notation::score_notation_pair(&gt, &hyp) {
        notation::NotationOutcome::Scored(r) => r,
        notation::NotationOutcome::EmptyReference {
            omr_len,
            hyp_empty,
            exact_match,
        } => {
            return Err(format!(
                "internal: empty ground truth past the emptiness gate \
                 (omr_len={omr_len} hyp_empty={hyp_empty} exact={exact_match})"
            )
            .into());
        }
    };
    let gt_headers = assembly_serialize::musicxml_import::measure_staff_headers(&gt_xml)
        .map_err(|e| format!("reference headers: {e}"))?;
    let gt_headers: Vec<_> = match args.max_measures {
        Some(n) => gt_headers.into_iter().filter(|h| h.measure <= n).collect(),
        None => gt_headers,
    };
    let hyp_headers = assembly_serialize::musicxml_import::measure_staff_headers(&hyp_xml)
        .map_err(|e| format!("hypothesis headers: {e}"))?;
    report.header_match = notation::measure_header_match_rate(&gt_headers, &hyp_headers);
    report.omr_ned = args.omr_ned;
    report.reference_sha256 = Some(ref_sha.clone());
    report.hypothesis_sha256 = Some(hyp_sha.clone());
    println!(
        "notation-F1 {:.3} drift {:.3} ({} / {} ref, {} hyp; spell {:.3} staff {:.3} voice {:.3} rest {:.3} grace {:.3} attach-P/R {:.3}/{:.3} dyn-P/R {:.3}/{:.3} hair-P/R {:.3}/{:.3} measure-exact {:?} headers {:?}) {}",
        report.notation_f1,
        report.drift_f1,
        report.notation_matched,
        report.gt_events,
        report.omr_events,
        report.spell_f1,
        report.staff_f1,
        report.voice_f1,
        report.rest_f1,
        report.grace_f1,
        report.attach_precision,
        report.attach_recall,
        report.dynamics_precision,
        report.dynamics_recall,
        report.hairpin_precision,
        report.hairpin_recall,
        report.measure_exact,
        report.header_match,
        dual_trailer(args.omr_ned, &ref_sha, &hyp_sha),
    );
    if let Some(p) = &args.report {
        write_file(p, serde_json::to_string_pretty(&report)?.as_bytes())?;
    }
    Ok(())
}

fn run_suite_summary(args: &SuiteSummaryArgs) -> Result<(), Err> {
    let text = fs::read_to_string(&args.manifest)
        .map_err(|e| format!("read {}: {e}", args.manifest.display()))?;
    let manifest: summary::Manifest = serde_json::from_str(&text)
        .map_err(|e| format!("parse {}: {e}", args.manifest.display()))?;
    let s = summary::summarize(&manifest).map_err(|e| format!("summarize: {e}"))?;
    println!(
        "suite={} engine={} pieces={} usable={} (rate {:.3})",
        s.suite, s.engine, s.n_total, s.n_usable, s.usable_rate,
    );
    println!(
        "strict-F1 mean {:.4} median(all) {:.4} median(transcribed) {:.4}",
        s.strict_f1.mean, s.strict_f1.median_all, s.strict_f1.median_transcribed,
    );
    match &s.exact_match {
        Some(e) => println!(
            "empty-ref {} exact {} (rate {:.3}; excluded from F1/NED)",
            e.n, e.n_exact, e.rate,
        ),
        None => println!("empty-ref 0 (excluded from F1/NED)"),
    }
    println!(
        "OMR-NED mean {:.4} median(all) {:.4} median(transcribed) {:.4} (n={} no-output={} missing={})",
        s.omr_ned.mean,
        s.omr_ned.median_all,
        s.omr_ned.median_transcribed,
        s.omr_ned.n,
        s.omr_ned.n_no_output,
        s.omr_ned.n_missing,
    );
    if let Some(p) = &args.out {
        write_file(p, serde_json::to_string_pretty(&s)?.as_bytes())?;
    }
    Ok(())
}

fn run_adapter(args: &SymboxAdapterArgs) -> Result<(), Err> {
    let text = fs::read_to_string(&args.boxes)
        .map_err(|e| format!("read {}: {e}", args.boxes.display()))?;
    let doc: BoxesDoc =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", args.boxes.display()))?;
    let (primary, aux, layout) = convert(&doc);
    for (doc, path) in [
        (serde_json::to_string_pretty(&primary)?, &args.primary),
        (serde_json::to_string_pretty(&aux)?, &args.aux),
        (serde_json::to_string_pretty(&layout)?, &args.layout),
    ] {
        write_file(path, doc.as_bytes())?;
    }
    println!(
        "wrote primary ({} glyphs) + aux ({} glyphs) + layout ({} systems)",
        primary.crops.iter().map(|c| c.glyphs.len()).sum::<usize>(),
        aux.crops.iter().map(|c| c.glyphs.len()).sum::<usize>(),
        layout.systems.len(),
    );
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(&cli) {
        eprintln!("eval-harness: error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxes_doc() -> BoxesDoc {
        serde_json::from_value(serde_json::json!({
            "staves": [
                {"lines": [100.0, 116.0, 132.0, 148.0, 164.0], "bbox": [50.0, 100.0, 500.0, 164.0]},
                {"lines": [200.0, 216.0, 232.0, 248.0, 264.0], "bbox": [50.0, 200.0, 500.0, 264.0]},
                {"lines": [900.0, 916.0, 932.0, 948.0, 964.0], "bbox": [50.0, 900.0, 500.0, 964.0]}
            ],
            "glyphs": [
                {"cls": "Note", "bbox": [60.0, 120.0, 80.0, 140.0]},
                {"cls": "Rest", "bbox": [90.0, 210.0, 110.0, 240.0]},
                {"cls": "Beam", "bbox": [60.0, 100.0, 120.0, 108.0]},
                {"cls": "Hook", "bbox": [130.0, 100.0, 140.0, 130.0]},
                {"cls": "NoteDot", "bbox": [85.0, 130.0, 90.0, 135.0]},
                {"cls": "Text", "bbox": [60.0, 10.0, 200.0, 40.0]},
                {"cls": "MeasureNumber", "bbox": [10.0, 300.0, 30.0, 320.0]}
            ]
        }))
        .unwrap()
    }

    #[test]
    fn primary_labels_cover_heads_and_rests_only() {
        assert_eq!(primary_label("Note"), Some("notehead-black"));
        assert_eq!(primary_label("Rest"), Some("rest"));
        assert_eq!(primary_label("Beam"), None);
        assert_eq!(primary_label("Hook"), None);
        assert_eq!(primary_label("Text"), None);
    }

    #[test]
    fn aux_labels_cover_evidence_classes_only() {
        assert_eq!(aux_label("Beam"), Some("beam"));
        assert_eq!(aux_label("Hook"), Some("flag"));
        assert_eq!(aux_label("NoteDot"), Some("dot"));
        assert_eq!(aux_label("Accidental"), Some("accidental"));
        assert_eq!(aux_label("Clef"), Some("clef"));
        assert_eq!(aux_label("Note"), None);
        assert_eq!(aux_label("Rest"), None);
        assert_eq!(aux_label("Text"), None);
        assert_eq!(aux_label("MeasureNumber"), None);
    }

    #[test]
    fn boxes_convert_to_xywh_with_full_confidence() {
        let (primary, aux, _) = convert(&boxes_doc());
        let head = &primary.crops[0].glyphs[0];
        assert_eq!(head.label, "notehead-black");
        assert_eq!(
            (head.bbox.x, head.bbox.y, head.bbox.w, head.bbox.h),
            (60.0, 120.0, 20.0, 20.0)
        );
        assert_eq!(head.confidence, 1.0);
        let beam = &aux.crops[0].glyphs[0];
        assert_eq!(beam.label, "beam");
        assert_eq!(
            (beam.bbox.x, beam.bbox.y, beam.bbox.w, beam.bbox.h),
            (60.0, 100.0, 60.0, 8.0)
        );
    }

    #[test]
    fn non_musical_classes_reach_neither_doc() {
        let (primary, aux, _) = convert(&boxes_doc());
        let labels: Vec<&str> = primary.crops[0]
            .glyphs
            .iter()
            .chain(aux.crops[0].glyphs.iter())
            .map(|g| g.label.as_str())
            .collect();
        assert_eq!(labels.len(), 5);
        assert!(!labels.contains(&"Text"));
        assert!(!labels.contains(&"MeasureNumber"));
    }

    #[test]
    fn staves_group_into_systems_at_large_gaps() {
        let systems = group_systems(&boxes_doc().staves);
        assert_eq!(systems, vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn sha256_hex_matches_nist_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(sha256_hex(b"").len(), 64);
    }

    #[test]
    fn layout_carries_oracle_lines_with_zero_tilt() {
        let (_, _, layout) = convert(&boxes_doc());
        assert_eq!(layout.systems.len(), 2);
        let staff = &layout.systems[0].staves[0];
        assert_eq!(staff.lines, vec![100.0, 116.0, 132.0, 148.0, 164.0]);
        assert_eq!(staff.slope, 0.0);
        assert_eq!(staff.pivot, 0.0);
    }
}
