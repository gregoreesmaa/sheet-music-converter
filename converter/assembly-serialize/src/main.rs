//! assembly-serialize CLI: note list -> MusicXML 3.1 + loss table +
//! validators -> .mscz. See README.md for the input schema and outputs.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use assembly_serialize::{
    bekern::to_bekern,
    decode::{decode_bekern, decode_bekern_unmerged},
    lattice::{rank_measures, read_lattice},
    example_input, has_errors,
    musicxml_import::{
        bekern_events, page_breaks, score_input_from_musicxml, score_inputs_by_page,
        split_measures, staff_headers, staff_headers_at_measure, system_breaks,
    },
    is_empty_score_input, pack_mscz, parse_score, score_pitched_count, score_stem,
    score_symbols, ser_for_symbols, to_empty_mscx, to_empty_musicxml, to_mscx, to_musicxml,
    validate_mscz, validate_musicxml, validate_notes, LossTable, Score, ScoreInput,
};
use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "assembly-serialize",
    version,
    about = "Note list -> MusicXML 3.1 + loss table + validators -> .mscz"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Convert a note-list JSON file to .musicxml + loss.json.
    Serialize(SerializeArgs),
    /// Run validators over a note-list JSON file (and optionally its outputs).
    Validate(ValidateArgs),
    /// Convert a note-list JSON file straight to a .mscz file.
    Mscz(MsczArgs),
    /// Full pipeline: .musicxml + .mscx + .mscz + loss.json + validation.json.
    Pack(PackArgs),
    /// List engraved system breaks (page, system, measure span) as JSON.
    #[command(name = "system-breaks")]
    SystemBreaks(SystemBreaksArgs),
    /// Write a small example note-list JSON file.
    Example(ExampleArgs),
    /// Symbol Error Rate of a hypothesis note-list against a reference.
    Ser(SerArgs),
    /// Convert a MusicXML file to note-list JSON (reference adapter).
    Import(ImportArgs),
    /// Emit bekern target tokens for one score page (MVD training).
    Bekern(BekernArgs),
    /// Split a MusicXML score into consecutive measure-window chunks
    /// (E5 single-system strip training: `split` then `bekern --page 1`
    /// per chunk).
    Split(SplitArgs),
    /// Decode a bekern token file to note-list JSON (MVD scoring).
    BekernDecode(BekernDecodeArgs),
    /// Decode a bekern token file straight to pack outputs
    /// (.musicxml + .mscx + .mscz + loss.json + validation.json).
    #[command(name = "bekern2musicxml")]
    Bekern2Musicxml(Bekern2MusicxmlArgs),
}

#[derive(Args, Debug)]
struct SerializeArgs {
    /// Note-list JSON input.
    #[arg(long)]
    input: PathBuf,
    /// Output directory for <stem>.musicxml + loss.json.
    #[arg(long)]
    out: PathBuf,
    /// Output file stem (default: derived from the score title).
    #[arg(long)]
    stem: Option<String>,
}

#[derive(Args, Debug)]
struct ValidateArgs {
    /// Note-list JSON input.
    #[arg(long)]
    input: PathBuf,
    /// Also validate this MusicXML file.
    #[arg(long)]
    musicxml: Option<PathBuf>,
    /// Also validate this .mscz file.
    #[arg(long)]
    mscz: Option<PathBuf>,
    /// Where to write the JSON issue report.
    #[arg(long, default_value = "validation.json")]
    report: PathBuf,
}

#[derive(Args, Debug)]
struct MsczArgs {
    /// Note-list JSON input.
    #[arg(long)]
    input: PathBuf,
    /// Output .mscz path.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args, Debug)]
struct PackArgs {
    /// Note-list JSON input.
    #[arg(long)]
    input: PathBuf,
    /// Output directory.
    #[arg(long)]
    out: PathBuf,
    /// Output file stem (default: derived from the score title).
    #[arg(long)]
    stem: Option<String>,
    /// Optional producer lattice file; validated and copied to
    /// `<out>/<stem>_lattice.json`, ranked for review.
    #[arg(long)]
    lattice_in: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct ImportArgs {
    /// MusicXML Partwise input (e.g. MuseScore export).
    #[arg(long)]
    input: PathBuf,
    /// Output directory for note-list JSON.
    #[arg(long)]
    out: PathBuf,
    /// Also write one file per page (`<stem>-p<page>.json`).
    #[arg(long, default_value_t = false)]
    pages: bool,
}

#[derive(Args, Debug)]
struct BekernArgs {
    /// MusicXML Partwise input (e.g. MuseScore export).
    #[arg(long)]
    input: PathBuf,
    /// Output directory for `<stem>-bekern.txt` (matches the page PNG stem).
    #[arg(long)]
    out: PathBuf,
    /// 1-based page number to emit (see page breaks in the score).
    /// Exactly one of `--page` or `--measures` must be given.
    #[arg(long)]
    page: Option<u32>,
    /// 1-based inclusive measure span to emit, e.g. `6..9` (one staff
    /// system; see `system-breaks`). Exactly one of `--page` or
    /// `--measures` must be given.
    #[arg(long)]
    measures: Option<String>,
    /// Output file stem, e.g. `ds0001-1` for page image `ds0001-1.png`.
    #[arg(long)]
    stem: String,
}

#[derive(Args, Debug)]
struct SystemBreaksArgs {
    /// MusicXML Partwise input (e.g. MuseScore export).
    #[arg(long)]
    input: PathBuf,
    /// Where to write the JSON array of system spans (prints to stdout
    /// when omitted).
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct SplitArgs {
    /// MusicXML Partwise input (e.g. MuseScore export).
    #[arg(long)]
    input: PathBuf,
    /// Output directory for `<stem>-m<first>.musicxml` chunk files
    /// (`<first>` = 1-based first measure of the chunk).
    #[arg(long)]
    out: PathBuf,
    /// Output file stem, e.g. `ds0001`.
    #[arg(long)]
    stem: String,
    /// Consecutive measures per chunk.
    #[arg(long, default_value_t = 4)]
    size: u32,
}

#[derive(Args, Debug)]
struct BekernDecodeArgs {
    /// bekern token file (whitespace-separated, one page).
    #[arg(long)]
    input: PathBuf,
    /// Where to write the decoded note-list JSON.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args, Debug)]
struct Bekern2MusicxmlArgs {
    /// bekern token file (whitespace-separated, one page).
    #[arg(long)]
    input: PathBuf,
    /// Output directory for pack outputs
    /// (`<stem>.musicxml` + `<stem>.mscx` + `<stem>.mscz` + loss.json + validation.json).
    #[arg(long)]
    out: PathBuf,
    /// Output file stem (default: input file stem, minus a trailing `-bekern`).
    #[arg(long)]
    stem: Option<String>,
    /// Optional producer lattice file; validated and copied to
    /// `<out>/<stem>_lattice.json`, ranked for review.
    #[arg(long)]
    lattice_in: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct SerArgs {
    /// Reference (ground-truth) note-list JSON.
    #[arg(long)]
    reference: PathBuf,
    /// Hypothesis note-list JSON to score.
    #[arg(long)]
    hypothesis: PathBuf,
    /// Where to write the JSON SER report.
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct ExampleArgs {
    /// Where to write the example note-list JSON.
    #[arg(long, default_value = "example-notes.json")]
    out: PathBuf,
}

type Err = Box<dyn std::error::Error>;

fn read_input(path: &Path) -> Result<ScoreInput, Err> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?)
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), Err> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        }
    }
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(())
}

fn summarize_losses(losses: &LossTable) -> String {
    format!(
        "{} loss, {} info",
        losses.loss_count(),
        losses.entries.len() - losses.loss_count()
    )
}

fn cmd_serialize(a: &SerializeArgs) -> Result<(), Err> {
    let input = read_input(&a.input)?;
    let (score, mut losses) =
        parse_score(&input).map_err(|e| format!("{}: {e}", a.input.display()))?;
    let xml = to_musicxml(&score, &mut losses);
    let stem = a.stem.clone().unwrap_or_else(|| score_stem(&score.title));
    fs::create_dir_all(&a.out).map_err(|e| format!("mkdir {}: {e}", a.out.display()))?;
    write_file(&a.out.join(format!("{stem}.musicxml")), xml.as_bytes())?;
    write_file(
        &a.out.join("loss.json"),
        serde_json::to_string_pretty(&losses)?.as_bytes(),
    )?;
    println!(
        "wrote {stem}.musicxml + loss.json ({})",
        summarize_losses(&losses)
    );
    let issues = validate_musicxml(&xml);
    if has_errors(&issues) {
        eprintln!("warning: generated MusicXML fails validation:");
        for i in &issues {
            eprintln!("  [{}] {}: {}", i.severity, i.code, i.message);
        }
    }
    Ok(())
}

fn cmd_validate(a: &ValidateArgs) -> Result<bool, Err> {
    let input = read_input(&a.input)?;
    let mut issues = validate_notes(&input);
    if let Some(p) = &a.musicxml {
        let xml = fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))?;
        issues.extend(validate_musicxml(&xml));
    }
    if let Some(p) = &a.mscz {
        let data = fs::read(p).map_err(|e| format!("read {}: {e}", p.display()))?;
        issues.extend(validate_mscz(&data));
    }
    write_file(&a.report, serde_json::to_string_pretty(&issues)?.as_bytes())?;
    let (errs, warns) = issues
        .iter()
        .partition::<Vec<_>, _>(|i| i.severity == "error");
    println!(
        "{} error(s), {} warning(s) -> {}",
        errs.len(),
        warns.len(),
        a.report.display()
    );
    for i in &issues {
        println!("[{}] {}: {}", i.severity, i.code, i.message);
    }
    Ok(errs.is_empty())
}

fn cmd_mscz(a: &MsczArgs) -> Result<(), Err> {
    let input = read_input(&a.input)?;
    let (score, mut losses) =
        parse_score(&input).map_err(|e| format!("{}: {e}", a.input.display()))?;
    let mscx = to_mscx(&score, &mut losses);
    let data = pack_mscz(&score, &mscx)?;
    write_file(&a.out, &data)?;
    println!(
        "wrote {} ({} bytes; {})",
        a.out.display(),
        data.len(),
        summarize_losses(&losses)
    );
    let issues = validate_mscz(&data);
    if has_errors(&issues) {
        return Err(format!("generated .mscz fails validation: {issues:?}").into());
    }
    Ok(())
}

/// Shared pack tail: score -> .musicxml + .mscx + .mscz + loss.json +
/// validation.json. Used by both `pack` (note-list input) and
/// `bekern2musicxml` (bekern input) so loss-table and validator behavior
/// are identical by construction.
fn write_pack_outputs(
    score: &Score,
    losses: &mut LossTable,
    input: &ScoreInput,
    stem: &str,
    out: &Path,
) -> Result<(), Err> {
    fs::create_dir_all(out).map_err(|e| format!("mkdir {}: {e}", out.display()))?;

    let xml = to_musicxml(score, losses);
    let mscx = to_mscx(score, losses);
    let mscz = pack_mscz(score, &mscx)?;

    write_file(&out.join(format!("{stem}.musicxml")), xml.as_bytes())?;
    write_file(&out.join(format!("{stem}.mscx")), mscx.as_bytes())?;
    write_file(&out.join(format!("{stem}.mscz")), &mscz)?;
    write_file(
        &out.join("loss.json"),
        serde_json::to_string_pretty(losses)?.as_bytes(),
    )?;

    let mut issues = validate_notes(input);
    issues.extend(validate_musicxml(&xml));
    issues.extend(validate_mscz(&mscz));
    write_file(
        &out.join("validation.json"),
        serde_json::to_string_pretty(&issues)?.as_bytes(),
    )?;

    println!(
        "wrote {stem}.musicxml, {stem}.mscx, {stem}.mscz, loss.json, validation.json ({})",
        summarize_losses(losses)
    );
    if has_errors(&issues) {
        eprintln!("warning: validation errors present, see validation.json");
        for i in issues.iter().filter(|i| i.severity == "error") {
            eprintln!("  [{}] {}", i.code, i.message);
        }
    }
    Ok(())
}

fn cmd_pack(a: &PackArgs) -> Result<(), Err> {
    let input = read_input(&a.input)?;
    let (score, mut losses) =
        parse_score(&input).map_err(|e| format!("{}: {e}", a.input.display()))?;
    let stem = a.stem.clone().unwrap_or_else(|| score_stem(&score.title));
    write_pack_outputs(&score, &mut losses, &input, &stem, &a.out)?;
    write_lattice_sidecar(a.lattice_in.as_deref(), &stem, &a.out)
}

/// Validate an optional producer lattice file and copy it, ranked for
/// review, to `<out>/<stem>_lattice.json`. `None` is a no-op so
/// lattice-free runs are byte-identical to before.
fn write_lattice_sidecar(
    lattice_in: Option<&Path>,
    stem: &str,
    out: &Path,
) -> Result<(), Err> {
    let Some(path) = lattice_in else {
        return Ok(());
    };
    let doc = read_lattice(path).map_err(|e| format!("lattice: {e}"))?;
    let ranked = rank_measures(&doc);
    let text = serde_json::to_string_pretty(&ranked)
        .map_err(|e| format!("lattice serialize: {e}"))?;
    write_file(&out.join(format!("{stem}_lattice.json")), text.as_bytes())?;
    println!(
        "wrote {stem}_lattice.json ({} measures, top rank score {:.3})",
        ranked.len(),
        ranked.first().map_or(0.0, |r| r.rank_score)
    );
    Ok(())
}

/// Default stem for `bekern2musicxml`: the input file stem, minus a
/// trailing `-bekern` (the `bekern` subcommand writes `<stem>-bekern.txt`).
fn bekern_default_stem(input: &Path) -> String {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "score".to_string());
    stem.strip_suffix("-bekern").unwrap_or(&stem).to_string()
}

fn cmd_bekern2musicxml(a: &Bekern2MusicxmlArgs) -> Result<(), Err> {
    let text =
        fs::read_to_string(&a.input).map_err(|e| format!("read {}: {e}", a.input.display()))?;
    // Link-faithful decode: the strict-xml oracle compares tie
    // segments, so every tie link must render as its own note.
    let input =
        decode_bekern_unmerged(&text).map_err(|e| format!("{}: {e}", a.input.display()))?;
    let stem = a
        .stem
        .clone()
        .unwrap_or_else(|| bekern_default_stem(&a.input));
    if is_empty_score_input(&input) {
        fs::create_dir_all(&a.out).map_err(|e| format!("mkdir {}: {e}", a.out.display()))?;
        let title = input.title.as_deref().unwrap_or(&stem);
        let xml = to_empty_musicxml(title);
        let mscx = to_empty_mscx(title);
        let empty_score = Score {
            title: title.to_string(),
            composer: input.composer.clone(),
            divisions: input.divisions,
            beats: input.time.map(|t| t.beats).unwrap_or(4),
            beat_type: input.time.map(|t| t.beat_type).unwrap_or(4),
            key_fifths: input.key_fifths,
            parts: Vec::new(),
            measure_attrs: Vec::new(),
        };
        let mscz = pack_mscz(&empty_score, &mscx)?;
        let losses = LossTable::default();
        write_file(&a.out.join(format!("{stem}.musicxml")), xml.as_bytes())?;
        write_file(&a.out.join(format!("{stem}.mscx")), mscx.as_bytes())?;
        write_file(&a.out.join(format!("{stem}.mscz")), &mscz)?;
        write_file(
            &a.out.join("loss.json"),
            serde_json::to_string_pretty(&losses)?.as_bytes(),
        )?;
        write_file(&a.out.join("validation.json"), b"[]")?;
        println!("wrote empty {stem}.musicxml (0 measures; abstained zero-note page)");
        return write_lattice_sidecar(a.lattice_in.as_deref(), &stem, &a.out);
    }
    let (score, mut losses) =
        parse_score(&input).map_err(|e| format!("{}: {e}", a.input.display()))?;
    write_pack_outputs(&score, &mut losses, &input, &stem, &a.out)?;
    write_lattice_sidecar(a.lattice_in.as_deref(), &stem, &a.out)
}

fn cmd_ser(a: &SerArgs) -> Result<(), Err> {
    let reference: ScoreInput = {
        let text = fs::read_to_string(&a.reference)
            .map_err(|e| format!("read {}: {e}", a.reference.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", a.reference.display()))?
    };
    let hypothesis: ScoreInput = {
        let text = fs::read_to_string(&a.hypothesis)
            .map_err(|e| format!("read {}: {e}", a.hypothesis.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", a.hypothesis.display()))?
    };
    // Metric hygiene: an empty reference never pools into the SER median
    // (SER 0.0 on both-empty reads as a perfect score). Route to the
    // exact-match column and exit 0; a hypothesis that will not parse
    // counts as non-empty (exact 0) with its reason attached.
    let ref_parsed = if is_empty_score_input(&reference) {
        None
    } else {
        Some(parse_score(&reference).map_err(|e| format!("ser: reference: {e}"))?)
    };
    let ref_empty = ref_parsed
        .as_ref()
        .map_or(true, |(s, _)| score_pitched_count(s) == 0);
    if ref_empty {
        let (hyp_len, hyp_empty, reason) = if is_empty_score_input(&hypothesis) {
            (0usize, true, None)
        } else {
            match parse_score(&hypothesis) {
                Ok((h, _)) => {
                    let pitched = score_pitched_count(&h);
                    (score_symbols(&h).len(), pitched == 0, None)
                }
                Err(e) => (0, false, Some(format!("hypothesis: {e}"))),
            }
        };
        let exact_match = if hyp_empty { 1.0 } else { 0.0 };
        println!(
            "SER-EMPTY exact_match {:.0} (empty reference, {} hyp symbols{})",
            exact_match,
            hyp_len,
            reason.as_ref().map(|r| format!("; {r}")).unwrap_or_default(),
        );
        if let Some(p) = &a.report {
            let mut v = serde_json::json!({
                "empty_ref": true,
                "exact_match": exact_match,
                "ref_len": 0,
                "hyp_len": hyp_len,
                "hyp_empty": hyp_empty,
                "note": "empty reference: excluded from SER aggregates; see exact_match",
            });
            if let Some(r) = reason {
                v["f1_zero_reason"] = serde_json::json!(r);
            }
            write_file(p, serde_json::to_string_pretty(&v)?.as_bytes())?;
        }
        return Ok(());
    }
    // Non-empty reference: an empty hypothesis stays included (all
    // reference symbols deleted, SER 1.0); anything else unparseable is
    // still a hard error.
    let (ref_score, _) = ref_parsed.unwrap();
    let ref_syms = score_symbols(&ref_score);
    let hyp_syms = match parse_score(&hypothesis) {
        Ok((h, _)) => score_symbols(&h),
        Err(_) if is_empty_score_input(&hypothesis) => Vec::new(),
        Err(e) => return Err(format!("ser: hypothesis: {e}").into()),
    };
    let report = ser_for_symbols(&ref_syms, &hyp_syms);
    println!(
        "SER {:.4} ({} edits / {} ref symbols, {} hyp symbols)",
        report.ser, report.distance, report.ref_len, report.hyp_len
    );
    if let Some(p) = &a.report {
        write_file(p, serde_json::to_string_pretty(&report)?.as_bytes())?;
    }
    Ok(())
}

fn cmd_import(a: &ImportArgs) -> Result<(), Err> {
    let xml =
        fs::read_to_string(&a.input).map_err(|e| format!("read {}: {e}", a.input.display()))?;
    let stem = a
        .input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "score".to_string());
    fs::create_dir_all(&a.out).map_err(|e| format!("mkdir {}: {e}", a.out.display()))?;
    if a.pages {
        let pages =
            score_inputs_by_page(&xml).map_err(|e| format!("{}: {e}", a.input.display()))?;
        let mut total = 0;
        for p in &pages {
            let n: usize = p.input.parts.iter().map(|pt| pt.notes.len()).sum();
            total += n;
            write_file(
                &a.out.join(format!("{stem}-p{}.json", p.page)),
                serde_json::to_string_pretty(&p.input)?.as_bytes(),
            )?;
        }
        println!(
            "wrote {} page files ({total} notes) -> {}",
            pages.len(),
            a.out.display()
        );
    } else {
        let input =
            score_input_from_musicxml(&xml).map_err(|e| format!("{}: {e}", a.input.display()))?;
        let n: usize = input.parts.iter().map(|p| p.notes.len()).sum();
        write_file(
            &a.out.join(format!("{stem}.json")),
            serde_json::to_string_pretty(&input)?.as_bytes(),
        )?;
        println!("wrote {stem}.json ({n} notes) -> {}", a.out.display());
    }
    Ok(())
}

/// Parse a `--measures START..END` span (1-based inclusive).
fn parse_measures_span(s: &str) -> Result<(u32, u32), String> {
    let (lo, hi) = s
        .split_once("..")
        .ok_or_else(|| format!("bad --measures {s:?}: want START..END, e.g. 6..9"))?;
    let start: u32 = lo
        .trim()
        .parse()
        .map_err(|_| format!("bad --measures {s:?}: want START..END, e.g. 6..9"))?;
    let end: u32 = hi
        .trim()
        .parse()
        .map_err(|_| format!("bad --measures {s:?}: want START..END, e.g. 6..9"))?;
    if start < 1 || end < start {
        return Err(format!(
            "bad --measures {s:?}: need 1 <= START <= END"
        ));
    }
    Ok((start, end))
}

fn cmd_bekern(a: &BekernArgs) -> Result<(), Err> {
    let xml =
        fs::read_to_string(&a.input).map_err(|e| format!("read {}: {e}", a.input.display()))?;
    let (start, end, scope) = match (&a.page, &a.measures) {
        (Some(_), Some(_)) => {
            return Err("give exactly one of --page or --measures".into());
        }
        (None, None) => {
            return Err("give exactly one of --page or --measures".into());
        }
        (Some(page), None) => {
            let breaks = page_breaks(&xml);
            let idx = breaks
                .iter()
                .position(|(p, _)| *p == *page)
                .ok_or(format!("no page {page} in {}", a.input.display()))?;
            let lo = breaks[idx].1;
            let hi = breaks.get(idx + 1).map(|(_, m)| m - 1).unwrap_or(u32::MAX);
            (lo, hi, format!("page {page}"))
        }
        (None, Some(span)) => {
            let (lo, hi) = parse_measures_span(span)
                .map_err(|e| format!("{}: {e}", a.input.display()))?;
            (lo, hi, format!("measures {lo}..{hi}"))
        }
    };
    let events: Vec<_> = bekern_events(&xml)
        .map_err(|e| format!("{}: {e}", a.input.display()))?
        .into_iter()
        .filter(|e| e.measure >= start && e.measure <= end)
        .collect();
    // Measure spans start mid-score: carry the clef/key/time signature
    // in effect at the span start, not the score-initial header.
    let headers = if a.measures.is_some() {
        staff_headers_at_measure(&xml, start)
            .map_err(|e| format!("{}: {e}", a.input.display()))?
    } else {
        staff_headers(&xml).map_err(|e| format!("{}: {e}", a.input.display()))?
    };
    let (toks, graces) = to_bekern(&events, &headers)
        .map_err(|e| format!("{} {scope}: {e}", a.input.display()))?;
    fs::create_dir_all(&a.out).map_err(|e| format!("mkdir {}: {e}", a.out.display()))?;
    write_file(
        &a.out.join(format!("{}-bekern.txt", a.stem)),
        toks.join(" ").as_bytes(),
    )?;
    println!(
        "wrote {}-bekern.txt ({} tokens, {} graces skipped) -> {}",
        a.stem,
        toks.len(),
        graces,
        a.out.display()
    );
    Ok(())
}

fn cmd_system_breaks(a: &SystemBreaksArgs) -> Result<(), Err> {
    let xml =
        fs::read_to_string(&a.input).map_err(|e| format!("read {}: {e}", a.input.display()))?;
    let spans = system_breaks(&xml);
    let json = serde_json::to_string_pretty(&spans)?;
    match &a.out {
        Some(p) => {
            write_file(p, json.as_bytes())?;
            println!("wrote {} system spans -> {}", spans.len(), p.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}

fn cmd_split(a: &SplitArgs) -> Result<(), Err> {
    let xml =
        fs::read_to_string(&a.input).map_err(|e| format!("read {}: {e}", a.input.display()))?;
    let chunks =
        split_measures(&xml, a.size).map_err(|e| format!("{}: {e}", a.input.display()))?;
    fs::create_dir_all(&a.out).map_err(|e| format!("mkdir {}: {e}", a.out.display()))?;
    for (first, chunk_xml) in &chunks {
        write_file(
            &a.out.join(format!("{}-m{first}.musicxml", a.stem)),
            chunk_xml.as_bytes(),
        )?;
    }
    println!(
        "wrote {} chunks ({} bars each) -> {}",
        chunks.len(),
        a.size,
        a.out.display()
    );
    Ok(())
}

fn cmd_bekern_decode(a: &BekernDecodeArgs) -> Result<(), Err> {
    let text =
        fs::read_to_string(&a.input).map_err(|e| format!("read {}: {e}", a.input.display()))?;
    let score = decode_bekern(&text).map_err(|e| format!("{}: {e}", a.input.display()))?;
    let n: usize = score.parts.iter().map(|p| p.notes.len()).sum();
    write_file(&a.out, serde_json::to_string_pretty(&score)?.as_bytes())?;
    println!(
        "wrote {} ({n} notes) -> {}",
        a.out.display(),
        a.input.display()
    );
    Ok(())
}

fn cmd_example(a: &ExampleArgs) -> Result<(), Err> {
    let text = serde_json::to_string_pretty(&serde_json::to_value(example_input())?)?;
    write_file(&a.out, text.as_bytes())?;
    println!("wrote {}", a.out.display());
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let res: Result<bool, Err> = match &cli.cmd {
        Cmd::Serialize(a) => cmd_serialize(a).map(|_| true),
        Cmd::Validate(a) => cmd_validate(a),
        Cmd::Mscz(a) => cmd_mscz(a).map(|_| true),
        Cmd::Pack(a) => cmd_pack(a).map(|_| true),
        Cmd::Example(a) => cmd_example(a).map(|_| true),
        Cmd::Ser(a) => cmd_ser(a).map(|_| true),
        Cmd::Import(a) => cmd_import(a).map(|_| true),
        Cmd::Bekern(a) => cmd_bekern(a).map(|_| true),
        Cmd::Split(a) => cmd_split(a).map(|_| true),
        Cmd::BekernDecode(a) => cmd_bekern_decode(a).map(|_| true),
        Cmd::Bekern2Musicxml(a) => cmd_bekern2musicxml(a).map(|_| true),
        Cmd::SystemBreaks(a) => cmd_system_breaks(a).map(|_| true),
    };
    match res {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("assembly-serialize: error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    /// Two 4/4 bars: a chord tone, a cross-bar tie, and a half rest —
    /// exercises headers, voices, chords, ties, rests, and bars.
    const FIXTURE: &str = "clef-G-2 key+0 time-4-4 \
        voice-1 quarter C 4 chord quarter E 4 half G 4 tie-start bar \
        voice-1 half G 4 tie-stop rest half bar";

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "assembly-serialize-bekern2musicxml-{tag}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn run_bekern2musicxml(tag: &str, stem: Option<String>) -> (PathBuf, PathBuf) {
        let dir = scratch_dir(tag);
        let input = dir.join("page-bekern.txt");
        fs::write(&input, FIXTURE).expect("fixture");
        let out = dir.join("out");
        cmd_bekern2musicxml(&Bekern2MusicxmlArgs {
            input: input.clone(),
            out: out.clone(),
            stem,
            lattice_in: None,
        })
        .expect("bekern2musicxml succeeds");
        (dir, out)
    }

    const SPLIT_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1">
<measure number="1">
<attributes><divisions>2</divisions><clef><sign>G</sign><line>2</line></clef><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
</measure>
<measure number="2">
<note><pitch><step>D</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
</measure>
</part>
</score-partwise>"#;

    #[test]
    fn split_writes_measure_window_chunks() {
        let dir = scratch_dir("split");
        let input = dir.join("in.musicxml");
        fs::write(&input, SPLIT_XML).expect("fixture");
        let out = dir.join("chunks");
        cmd_split(&SplitArgs {
            input: input.clone(),
            out: out.clone(),
            stem: "dsX".to_string(),
            size: 1,
        })
        .expect("split succeeds");
        let c1 = fs::read_to_string(out.join("dsX-m1.musicxml")).expect("chunk 1");
        let c2 = fs::read_to_string(out.join("dsX-m2.musicxml")).expect("chunk 2");
        let pitches = |xml: &str| {
            bekern_events(xml)
                .unwrap()
                .iter()
                .filter_map(|e| e.pitch)
                .collect::<Vec<_>>()
        };
        assert_eq!(pitches(&c1), vec![('C', 4, 0)]);
        assert_eq!(pitches(&c2), vec![('D', 4, 0)]);
        // Chunk 2 carries the running headers despite having no
        // <attributes> of its own in the source score.
        assert!(c2.contains("<clef>") && c2.contains("<time>"));
    }

    #[test]
    fn parse_measures_span_accepts_start_dot_dot_end() {
        assert_eq!(parse_measures_span("6..9"), Ok((6, 9)));
        assert_eq!(parse_measures_span("1..5"), Ok((1, 5)));
        assert!(parse_measures_span("6-9").is_err());
        assert!(parse_measures_span("9..6").is_err());
        assert!(parse_measures_span("0..4").is_err());
        assert!(parse_measures_span("abc").is_err());
    }

    #[test]
    fn bekern_default_stem_strips_bekern_suffix() {
        assert_eq!(
            bekern_default_stem(Path::new("some/dir/ds0091-3-bekern.txt")),
            "ds0091-3"
        );
        assert_eq!(bekern_default_stem(Path::new("plain.txt")), "plain");
    }

    #[test]
    fn bekern2musicxml_writes_pack_outputs() {
        let (_dir, out) = run_bekern2musicxml("pack", None);
        for name in [
            "page.musicxml",
            "page.mscx",
            "page.mscz",
            "loss.json",
            "validation.json",
        ] {
            assert!(out.join(name).is_file(), "{name} written");
        }
        let xml = fs::read_to_string(out.join("page.musicxml")).expect("read musicxml");
        assert!(validate_musicxml(&xml)
            .iter()
            .all(|i| i.severity != "error"));
        let losses: LossTable =
            serde_json::from_str(&fs::read_to_string(out.join("loss.json")).expect("read loss"))
                .expect("loss.json parses");
        let _ = losses.loss_count();
        let issues: Vec<assembly_serialize::Issue> = serde_json::from_str(
            &fs::read_to_string(out.join("validation.json")).expect("read validation"),
        )
        .expect("validation.json parses");
        assert!(!has_errors(&issues));
    }

    #[test]
    fn bekern2musicxml_matches_pack_on_same_notes() {
        // The new command must behave identically to pack: decode the
        // fixture to notes, pack those notes, and require byte-identical
        // .musicxml / loss.json / validation.json. Both sides decode
        // link-faithfully (unmerged): tie links render as their own
        // notes with tie marks, never merged.
        let dir = scratch_dir("equiv");
        let bekern = dir.join("page-bekern.txt");
        fs::write(&bekern, FIXTURE).expect("fixture");
        let hyp_out = dir.join("hyp");
        cmd_bekern2musicxml(&Bekern2MusicxmlArgs {
            input: bekern.clone(),
            out: hyp_out.clone(),
            stem: Some("page".to_string()),
            lattice_in: None,
        })
        .expect("bekern2musicxml succeeds");

        let decoded = decode_bekern_unmerged(FIXTURE).expect("fixture decodes");
        let notes_path = dir.join("notes.json");
        fs::write(
            &notes_path,
            serde_json::to_string_pretty(&decoded).expect("notes json"),
        )
        .expect("write notes");
        let ref_out = dir.join("ref");
        cmd_pack(&PackArgs {
            input: notes_path,
            out: ref_out.clone(),
            stem: Some("page".to_string()),
            lattice_in: None,
        })
        .expect("pack succeeds");

        for name in ["page.musicxml", "loss.json", "validation.json"] {
            let hyp = fs::read(hyp_out.join(name)).expect("hyp file");
            let refr = fs::read(ref_out.join(name)).expect("ref file");
            assert_eq!(hyp, refr, "{name} identical to pack");
        }
    }

    #[test]
    fn bekern2musicxml_bad_input_errors() {
        let dir = scratch_dir("bad");
        let input = dir.join("bad-bekern.txt");
        fs::write(&input, "quarter C").expect("bad fixture");
        let err = cmd_bekern2musicxml(&Bekern2MusicxmlArgs {
            input: input.clone(),
            out: dir.join("out"),
            stem: None,
            lattice_in: None,
        })
        .expect_err("truncated stream must fail");
        assert!(
            err.to_string().contains(&input.display().to_string()),
            "error names the input file: {err}"
        );
    }

    #[test]
    fn bekern2musicxml_lattice_sidecar_ranked() {
        let dir = scratch_dir("lattice");
        let input = dir.join("page-bekern.txt");
        fs::write(&input, FIXTURE).expect("fixture");
        let lat = dir.join("in-lattice.json");
        // Deliberately worst-first: rank must reorder uncertain first.
        fs::write(
            &lat,
            r#"{"stem": "page",
                "measures": [
                  {"measure": 1, "confidence": 0.97, "cluster_size": 12,
                   "candidates": ["voice-1 half E 4 bar", "voice-1 half E 4 bar"]},
                  {"measure": 0, "confidence": 0.42, "cluster_size": 14,
                   "bbox": [0.0, 0.0, 100.0, 50.0],
                   "candidates": ["voice-1 quarter C 4 bar", "voice-1 quarter D 4 bar"]}]}"#,
        )
        .expect("lattice fixture");
        let out = dir.join("out");
        cmd_bekern2musicxml(&Bekern2MusicxmlArgs {
            input: input.clone(),
            out: out.clone(),
            stem: Some("page".to_string()),
            lattice_in: Some(lat),
        })
        .expect("bekern2musicxml with lattice succeeds");
        let text =
            fs::read_to_string(out.join("page_lattice.json")).expect("sidecar written");
        let rows: Vec<serde_json::Value> = serde_json::from_str(&text).expect("sidecar parses");
        assert_eq!(rows.len(), 2);
        // 14 * (1 - 0.42) = 8.12 outranks 12 * (1 - 0.97) = 0.36.
        assert_eq!(rows[0]["measure"], 0);
        assert_eq!(rows[0]["rank"], 1);
        assert!((rows[0]["rank_score"].as_f64().unwrap() - 8.12).abs() < 1e-9);
        assert_eq!(rows[1]["measure"], 1);
        assert_eq!(rows[1]["rank"], 2);
    }

    #[test]
    fn bekern2musicxml_bad_lattice_errors() {
        let dir = scratch_dir("lattice-bad");
        let input = dir.join("page-bekern.txt");
        fs::write(&input, FIXTURE).expect("fixture");
        let lat = dir.join("bad-lattice.json");
        fs::write(
            &lat,
            r#"{"stem": "page",
                "measures": [
                  {"measure": 0, "confidence": 2.0, "cluster_size": 3,
                   "candidates": ["a", "b"]}]}"#,
        )
        .expect("bad lattice fixture");
        let err = cmd_bekern2musicxml(&Bekern2MusicxmlArgs {
            input: input.clone(),
            out: dir.join("out"),
            stem: Some("page".to_string()),
            lattice_in: Some(lat),
        })
        .expect_err("bad lattice must fail");
        assert!(err.to_string().contains("confidence"), "names the violation: {err}");
    }
}
