//! dataset-pack: validate sheet-music pairs, split train/val/test, pack manifest + card.
//!
//! Pair convention: `<id>.<image-ext>` + `<id>.<notation-ext>` share a stem.
//! See README.md for the extension lists and `pack` output layout.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"];
const NOTATION_EXTS: &[&str] = &[
    "musicxml", "xml", "mxl", "mei", "krn", "kern", "abc", "mid", "midi",
];

#[derive(Parser, Debug)]
#[command(
    name = "dataset-pack",
    version,
    about = "Validate pairs, split, and pack a DVC-friendly dataset"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Scan --input for image<->notation pairs and write a JSON report.
    Validate(ValidateArgs),
    /// Deterministic train/val/test split of valid pair ids.
    Split(SplitArgs),
    /// Validate + split + manifest.jsonl + dataset card + dvc.yaml.
    Pack(PackArgs),
}

#[derive(Args, Debug)]
struct ValidateArgs {
    /// Directory holding the raw pairs (scanned recursively).
    #[arg(long)]
    input: PathBuf,
    /// Where to write the JSON validation report.
    #[arg(long, default_value = "validation.json")]
    report: PathBuf,
}

#[derive(Args, Debug)]
struct SplitArgs {
    #[arg(long)]
    input: PathBuf,
    /// Output directory for splits/*.txt.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 0.8)]
    train: f64,
    #[arg(long, default_value_t = 0.1)]
    val: f64,
    #[arg(long, default_value_t = 0.1)]
    test: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Args, Debug)]
struct PackArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "sheet-music-dataset")]
    name: String,
    #[arg(long, default_value_t = 0.8)]
    train: f64,
    #[arg(long, default_value_t = 0.1)]
    val: f64,
    #[arg(long, default_value_t = 0.1)]
    test: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Pack only valid pairs instead of failing on orphans/duplicates.
    #[arg(long, default_value_t = false)]
    allow_partial: bool,
    /// Enable strict training-data checks: box schema + bounds, class
    /// histogram, piece-level split integrity, degradation-param linkage.
    /// Any firing check fails the pack. Default off; off = legacy behavior.
    #[arg(long, default_value_t = false)]
    strict: bool,
    /// Comma-separated class names expected in strict box labels. A kept
    /// class with zero boxes fails a strict pack.
    #[arg(long, default_value = STRICT_DEFAULT_CLASSES)]
    strict_classes: String,
}

/// Kept detector classes required by `--strict` unless overridden with
/// `--strict-classes` (mirrors the `mkdet_data.py` proposer ontology).
const STRICT_DEFAULT_CLASSES: &str =
    "NoteBlack,Rest,Beam,Hook,Accidental,NoteDot,Clef,NoteVoid,Stem,Flag";

// ---------- model ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SideFile {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PairRecord {
    id: String,
    split: String,
    image: String,
    notation: String,
    image_sha256: String,
    notation_sha256: String,
    image_bytes: u64,
    notation_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PairProblem {
    id: String,
    kind: String,
    detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ValidationReport {
    input: String,
    samples_total: usize,
    valid: usize,
    invalid: usize,
    problems: Vec<PairProblem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetMeta {
    name: String,
    seed: u64,
    train_ratio: f64,
    val_ratio: f64,
    test_ratio: f64,
    counts: BTreeMap<String, usize>,
    total: usize,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("dataset-pack: error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Validate(a) => {
            let rep = validate_dir(&a.input)?;
            write_json(&a.report, &rep)?;
            println!(
                "validate: {} samples, {} valid, {} invalid -> {}",
                rep.samples_total,
                rep.valid,
                rep.invalid,
                a.report.display()
            );
            if rep.invalid > 0 {
                return Err(format!(
                    "{} invalid pair(s); see {}",
                    rep.invalid,
                    a.report.display()
                ));
            }
            Ok(())
        }
        Cmd::Split(a) => {
            let ratios = norm_ratios(a.train, a.val, a.test)?;
            let rep = validate_dir(&a.input)?;
            if rep.invalid > 0 {
                return Err(format!(
                    "{} invalid pair(s); run `validate` for details or drop orphans first",
                    rep.invalid
                ));
            }
            let valid_ids: Vec<String> = valid_pairs(&a.input)?
                .into_iter()
                .map(|(id, _, _)| id)
                .collect();
            let assignment = assign_splits(&valid_ids, a.seed, ratios);
            write_splits(&a.out, &assignment)?;
            let counts = count_splits(&assignment);
            println!("split: {counts:?} -> {}", a.out.display());
            Ok(())
        }
        Cmd::Pack(a) => pack(&a),
    }
}

// ---------- discovery ----------

/// Group files under `root` by sample id (relative parent + stem).
/// Label/params sidecars the strict gates consume (`<stem>-boxes.json`,
/// `<stem>_params.json`, e.g. photo-degrade `<stem>{suffix}_params.json`).
/// They are pack inputs, not samples: pairing skips them so they never
/// form groups or trip `unknown-extension`.
fn is_strict_sidecar(p: &Path) -> bool {
    match p.file_name().and_then(|s| s.to_str()) {
        Some(n) => n.ends_with("-boxes.json") || n.ends_with("_params.json"),
        None => false,
    }
}

fn discover(root: &Path) -> Result<BTreeMap<String, Vec<PathBuf>>, String> {
    if !root.is_dir() {
        return Err(format!("input dir not found: {}", root.display()));
    }
    let mut groups: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && !is_strict_sidecar(e.path()))
    {
        let p = entry.path().to_path_buf();
        let id = sample_id(root, &p);
        groups.entry(id).or_default().push(p);
    }
    // Deterministic order inside each group.
    for files in groups.values_mut() {
        files.sort();
    }
    Ok(groups)
}

/// `batch1/0001.png` under root -> `batch1/0001`.
fn sample_id(root: &Path, file: &Path) -> String {
    let rel = file.strip_prefix(root).unwrap_or(file);
    let parent = rel.parent().map(|p| p.to_string_lossy().replace('\\', "/"));
    let stem = file
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    match parent {
        Some(p) if !p.is_empty() => format!("{p}/{stem}"),
        _ => stem,
    }
}

fn ext_of(p: &Path) -> String {
    p.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

fn rel_posix(root: &Path, file: &Path) -> String {
    file.strip_prefix(root)
        .map(|r| r.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| file.to_string_lossy().replace('\\', "/"))
}

fn sha256_of(p: &Path) -> Result<(u64, String), String> {
    let f = fs::File::open(p).map_err(|e| format!("open {}: {e}", p.display()))?;
    let mut reader = BufReader::new(f);
    let mut hasher = Sha256::new();
    let mut bytes: u64 = 0;
    loop {
        let buf = reader
            .fill_buf()
            .map_err(|e| format!("read {}: {e}", p.display()))?;
        if buf.is_empty() {
            break;
        }
        bytes += buf.len() as u64;
        hasher.update(buf);
        let n = buf.len();
        reader.consume(n);
    }
    Ok((bytes, hex_of(&hasher.finalize())))
}

fn hex_of(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ---------- validation ----------

/// Full validation report over every discovered id.
fn validate_dir(root: &Path) -> Result<ValidationReport, String> {
    let groups = discover(root)?;
    let mut problems = Vec::new();
    let mut valid = 0usize;
    for (id, files) in &groups {
        let images: Vec<&PathBuf> = files
            .iter()
            .filter(|p| IMAGE_EXTS.contains(&ext_of(p).as_str()))
            .collect();
        let notations: Vec<&PathBuf> = files
            .iter()
            .filter(|p| NOTATION_EXTS.contains(&ext_of(p).as_str()))
            .collect();
        let unknown: Vec<String> = files
            .iter()
            .filter(|p| {
                !IMAGE_EXTS.contains(&ext_of(p).as_str())
                    && !NOTATION_EXTS.contains(&ext_of(p).as_str())
            })
            .map(|p| rel_posix(root, p))
            .collect();

        let mut bad = false;
        if images.len() != 1 {
            bad = true;
            problems.push(PairProblem {
                id: id.clone(),
                kind: if images.is_empty() {
                    "missing-image".into()
                } else {
                    "duplicate-image".into()
                },
                detail: format!("expected 1 image side, found {}", images.len()),
            });
        }
        if notations.len() != 1 {
            bad = true;
            problems.push(PairProblem {
                id: id.clone(),
                kind: if notations.is_empty() {
                    "missing-notation".into()
                } else {
                    "duplicate-notation".into()
                },
                detail: format!("expected 1 notation side, found {}", notations.len()),
            });
        }
        if !unknown.is_empty() {
            bad = true;
            problems.push(PairProblem {
                id: id.clone(),
                kind: "unknown-extension".into(),
                detail: format!("unpaired files: {}", unknown.join(", ")),
            });
        }
        // Empty-file check on the paired sides.
        for p in images.iter().chain(notations.iter()) {
            let meta = fs::metadata(p).map_err(|e| format!("stat {}: {e}", p.display()))?;
            if meta.len() == 0 {
                bad = true;
                problems.push(PairProblem {
                    id: id.clone(),
                    kind: "empty-file".into(),
                    detail: rel_posix(root, p),
                });
            }
        }
        if !bad {
            valid += 1;
        }
    }
    let samples_total = groups.len();
    Ok(ValidationReport {
        input: root.to_string_lossy().into_owned(),
        samples_total,
        valid,
        invalid: samples_total - valid,
        problems,
    })
}

/// Valid pairs only: (id, image_side, notation_side).
fn valid_pairs(root: &Path) -> Result<Vec<(String, SideFile, SideFile)>, String> {
    let groups = discover(root)?;
    let mut out = Vec::new();
    for (id, files) in &groups {
        let images: Vec<&PathBuf> = files
            .iter()
            .filter(|p| IMAGE_EXTS.contains(&ext_of(p).as_str()))
            .collect();
        let notations: Vec<&PathBuf> = files
            .iter()
            .filter(|p| NOTATION_EXTS.contains(&ext_of(p).as_str()))
            .collect();
        let unknown = files.iter().any(|p| {
            !IMAGE_EXTS.contains(&ext_of(p).as_str())
                && !NOTATION_EXTS.contains(&ext_of(p).as_str())
        });
        if images.len() == 1 && notations.len() == 1 && !unknown {
            let (ibytes, isha) = sha256_of(images[0])?;
            let (nbytes, nsha) = sha256_of(notations[0])?;
            if ibytes == 0 || nbytes == 0 {
                continue;
            }
            out.push((
                id.clone(),
                SideFile {
                    path: rel_posix(root, images[0]),
                    bytes: ibytes,
                    sha256: isha,
                },
                SideFile {
                    path: rel_posix(root, notations[0]),
                    bytes: nbytes,
                    sha256: nsha,
                },
            ));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

// ---------- splitting ----------

fn norm_ratios(train: f64, val: f64, test: f64) -> Result<(f64, f64, f64), String> {
    if train < 0.0 || val < 0.0 || test < 0.0 {
        return Err("ratios must be >= 0".into());
    }
    let sum = train + val + test;
    if sum <= 0.0 {
        return Err("ratios must sum to > 0".into());
    }
    Ok((train / sum, val / sum, test / sum))
}

/// Deterministic bucket from sha256("{seed}/{id}"): r in [0,1).
fn bucket(id: &str, seed: u64) -> f64 {
    let mut h = Sha256::new();
    h.update(format!("{seed}/{id}").as_bytes());
    let d = h.finalize();
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&d[..8]);
    (u64::from_be_bytes(raw) % 10_000) as f64 / 10_000.0
}

fn assign_splits(ids: &[String], seed: u64, ratios: (f64, f64, f64)) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for id in ids {
        let r = bucket(id, seed);
        let split = if r < ratios.0 {
            "train"
        } else if r < ratios.0 + ratios.1 {
            "val"
        } else {
            "test"
        };
        map.insert(id.clone(), split.to_string());
    }
    map
}

fn count_splits(assignment: &BTreeMap<String, String>) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> =
        BTreeMap::from([("train".into(), 0), ("val".into(), 0), ("test".into(), 0)]);
    for s in assignment.values() {
        *counts.entry(s.clone()).or_insert(0) += 1;
    }
    counts
}

fn write_splits(out: &Path, assignment: &BTreeMap<String, String>) -> Result<(), String> {
    let dir = out.join("splits");
    fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let mut per: BTreeMap<&str, Vec<&String>> = BTreeMap::new();
    for (id, split) in assignment {
        per.entry(split.as_str()).or_default().push(id);
    }
    for name in ["train", "val", "test"] {
        let mut ids = per.remove(name).unwrap_or_default();
        ids.sort();
        let mut f = fs::File::create(dir.join(format!("{name}.txt")))
            .map_err(|e| format!("write {name}.txt: {e}"))?;
        for id in ids {
            writeln!(f, "{id}").map_err(|e| format!("write {name}.txt: {e}"))?;
        }
    }
    Ok(())
}

// ---------- pack ----------

fn pack(a: &PackArgs) -> Result<(), String> {
    let ratios = norm_ratios(a.train, a.val, a.test)?;
    let rep = validate_dir(&a.input)?;
    if rep.invalid > 0 && !a.allow_partial {
        return Err(format!(
            "{} invalid pair(s); fix them or re-run with --allow-partial (details in validation report)",
            rep.invalid
        ));
    }
    let pairs = valid_pairs(&a.input)?;
    if pairs.is_empty() {
        return Err("no valid pairs found".into());
    }
    let ids: Vec<String> = pairs.iter().map(|(id, _, _)| id.clone()).collect();
    let assignment = assign_splits(&ids, a.seed, ratios);
    if a.strict {
        let expected = parse_classes(&a.strict_classes);
        let problems = strict_problems(&a.input, &pairs, &assignment, &expected);
        if !problems.is_empty() {
            return Err(strict_report(&problems));
        }
    }

    fs::create_dir_all(&a.out).map_err(|e| format!("mkdir {}: {e}", a.out.display()))?;
    write_json(&a.out.join("validation.json"), &rep)?;
    write_splits(&a.out, &assignment)?;

    // manifest.jsonl (sorted by id for stable diffs).
    let mut records: Vec<PairRecord> = pairs
        .into_iter()
        .map(|(id, img, note)| {
            let split = assignment.get(&id).cloned().unwrap_or_default();
            PairRecord {
                id,
                split,
                image: img.path,
                notation: note.path,
                image_sha256: img.sha256,
                notation_sha256: note.sha256,
                image_bytes: img.bytes,
                notation_bytes: note.bytes,
            }
        })
        .collect();
    records.sort_by(|x, y| x.id.cmp(&y.id));
    let mut mf = fs::File::create(a.out.join("manifest.jsonl"))
        .map_err(|e| format!("write manifest.jsonl: {e}"))?;
    for r in &records {
        let line = serde_json::to_string(r).map_err(|e| e.to_string())?;
        writeln!(mf, "{line}").map_err(|e| format!("write manifest.jsonl: {e}"))?;
    }

    let counts = count_splits(&assignment);
    let meta = DatasetMeta {
        name: a.name.clone(),
        seed: a.seed,
        train_ratio: ratios.0,
        val_ratio: ratios.1,
        test_ratio: ratios.2,
        counts: counts.clone(),
        total: records.len(),
    };
    write_json(&a.out.join("dataset-meta.json"), &meta)?;
    fs::write(a.out.join("DATASET_CARD.md"), dataset_card(&meta, &rep, a))
        .map_err(|e| format!("write DATASET_CARD.md: {e}"))?;
    fs::write(a.out.join("dvc.yaml"), dvc_yaml(a)).map_err(|e| format!("write dvc.yaml: {e}"))?;
    ensure_gitignore(&a.out)?;

    println!(
        "pack: {} valid (+{} invalid skipped) manifest.jsonl splits={counts:?} -> {}",
        rep.valid,
        rep.invalid,
        a.out.display()
    );
    Ok(())
}

fn dataset_card(meta: &DatasetMeta, rep: &ValidationReport, a: &PackArgs) -> String {
    format!(
        concat!(
            "# Dataset card: {name}\n",
            "\n",
            "Sheet-music image<->notation pairs packed with `dataset-pack`.\n",
            "\n",
            "## Splits (seed {seed})\n",
            "\n",
            "| split | count |\n",
            "|-------|-------|\n",
            "| train | {train} |\n",
            "| val   | {val} |\n",
            "| test  | {test} |\n",
            "| total | {total} |\n",
            "\n",
            "Ratios train/val/test = {rt:.3}/{rv:.3}/{rtest:.3}.\n",
            "Assignment is deterministic: `sha256(\"{{seed}}/{{id}}\")`.\n",
            "\n",
            "## Source\n",
            "\n",
            "- Raw pairs: `{input}` ({valid} valid, {invalid} invalid at pack time)\n",
            "- Manifest: `manifest.jsonl` (one JSON object per sample: id, split, ",
            "relative image/notation paths, sha256, byte sizes)\n",
            "- Meta: `dataset-meta.json`; id lists: `splits/{{train,val,test}}.txt`\n",
            "\n",
            "## TODO (fill in before release)\n",
            "\n",
            "- [ ] Provenance: where did the scores/images come from? License?\n",
            "- [ ] Preprocessing: binarization, deskew, cropping, normalization?\n",
            "- [ ] Intended use / known limitations (engraving styles, eras, noise)\n",
            "- [ ] DVC remote: `dvc remote add -d storage <url>` then `dvc push`\n",
            "\n",
            "## Reproduce\n",
            "\n",
            "```bash\n",
            "dataset-pack pack --input {input} --out . --name {name} --seed {seed}\n",
            "dvc repro\n",
            "```\n",
        ),
        name = meta.name,
        seed = meta.seed,
        train = meta.counts.get("train").copied().unwrap_or(0),
        val = meta.counts.get("val").copied().unwrap_or(0),
        test = meta.counts.get("test").copied().unwrap_or(0),
        total = meta.total,
        rt = meta.train_ratio,
        rv = meta.val_ratio,
        rtest = meta.test_ratio,
        input = a.input.to_string_lossy(),
        valid = rep.valid,
        invalid = rep.invalid,
    )
}

fn dvc_yaml(a: &PackArgs) -> String {
    format!(
        concat!(
            "stages:\n",
            "  validate:\n",
            "    cmd: dataset-pack validate --input {input} --report validation.json\n",
            "    deps:\n",
            "    - {input}\n",
            "    outs:\n",
            "    - validation.json\n",
            "  pack:\n",
            "    cmd: dataset-pack pack --input {input} --out . --name {name} ",
            "--seed {seed} --train {train} --val {val} --test {test} --allow-partial\n",
            "    deps:\n",
            "    - {input}\n",
            "    params:\n",
            "    - seed: {seed}\n",
            "    outs:\n",
            "    - manifest.jsonl\n",
            "    - dataset-meta.json\n",
            "    - splits/train.txt\n",
            "    - splits/val.txt\n",
            "    - splits/test.txt\n",
            "    - DATASET_CARD.md\n",
        ),
        input = a.input.to_string_lossy(),
        name = a.name,
        seed = a.seed,
        train = a.train,
        val = a.val,
        test = a.test,
    )
}

/// Create `<out>/.gitignore` when missing so raw dumps stay out of git
/// while the manifest/card stay versioned.
fn ensure_gitignore(out: &Path) -> Result<(), String> {
    let gi = out.join(".gitignore");
    if gi.exists() {
        return Ok(());
    }
    fs::write(
        &gi,
        concat!(
            "# dataset-pack: raw data lives in DVC, descriptors live in git.\n",
            "/raw/\n",
            "/data/\n",
            "*.png\n",
            "*.jpg\n",
            "*.jpeg\n",
            "*.tif\n",
            "*.tiff\n",
            "!manifest.jsonl\n",
            "!dataset-meta.json\n",
            "!splits/\n",
            "!DATASET_CARD.md\n",
            "!dvc.yaml\n",
            "!validation.json\n",
        ),
    )
    .map_err(|e| format!("write .gitignore: {e}"))?;
    Ok(())
}

// ---------- strict checks (`pack --strict`) ----------
//
// Four training-data gates that the legacy pairing/emptiness/hash checks do
// not cover. All helpers are pure over (root, pairs, assignment) except for
// the sidecar reads they document; non-strict runs never call them.

/// Run every strict gate and return the firing problems (empty = pass).
fn strict_problems(
    root: &Path,
    pairs: &[(String, SideFile, SideFile)],
    assignment: &BTreeMap<String, String>,
    expected_classes: &[String],
) -> Vec<PairProblem> {
    let mut out = Vec::new();
    out.extend(find_split_leaks(assignment));
    let (histogram, mut box_problems) = strict_box_problems(root, pairs);
    out.append(&mut box_problems);
    out.extend(strict_histogram_problems(&histogram, expected_classes));
    out.extend(strict_params_problems(root, pairs));
    out.sort_by(|a, b| (&a.kind, &a.id).cmp(&(&b.kind, &b.id)));
    out
}

fn strict_report(problems: &[PairProblem]) -> String {
    let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
    for p in problems {
        *kinds.entry(p.kind.as_str()).or_insert(0) += 1;
    }
    let summary = kinds
        .iter()
        .map(|(k, n)| format!("{k}x{n}"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut msg = format!(
        "strict validation failed: {} problem(s) [{summary}]",
        problems.len()
    );
    for p in problems.iter().take(10) {
        msg.push_str(&format!("\n  {} {}: {}", p.kind, p.id, p.detail));
    }
    if problems.len() > 10 {
        msg.push_str(&format!("\n  ... and {} more", problems.len() - 10));
    }
    msg
}

fn parse_classes(s: &str) -> Vec<String> {
    s.split(',')
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect()
}

// ----- piece-level split integrity -----

/// Map a sample id to its piece key: all aug variants and all pages of one
/// score share a key, so they must land in the same split.
/// `ds0001/ds0001-1` -> `ds0001`; `ds0001-1.aug0` -> `ds0001`;
/// `ds0001-1_d0` / `sample01-1_degraded` -> `ds0001` / `sample01`.
fn piece_of(id: &str) -> String {
    let stem = id.rsplit('/').next().unwrap_or(id);
    let mut s = stem;
    if let Some(b) = s.strip_suffix("_degraded") {
        s = b;
    }
    s = strip_degrade_variant(s);
    s = strip_aug_variant(s);
    if let Some(pos) = s.rfind('-') {
        let tail = &s[pos + 1..];
        if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) && pos > 0 {
            s = &s[..pos];
        }
    }
    s.to_string()
}

/// Strip a trailing `_d<N>` photo-degrade variant suffix (`ds0001-1_d0`).
fn strip_degrade_variant(s: &str) -> &str {
    if let Some(pos) = s.rfind("_d") {
        let tail = &s[pos + 2..];
        if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) && pos > 0 {
            return &s[..pos];
        }
    }
    s
}

/// Strip a trailing `.aug<N>` / `_aug<N>` notation-vary suffix.
fn strip_aug_variant(s: &str) -> &str {
    for marker in [".aug", "_aug"] {
        if let Some(pos) = s.rfind(marker) {
            let tail = &s[pos + marker.len()..];
            if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) && pos > 0 {
                return &s[..pos];
            }
        }
    }
    s
}

/// Label sidecars (`*-boxes.json`) are per page: aug/degraded variants of a
/// page share its labels, but different pages do not.
fn label_key(stem: &str) -> String {
    let mut s = stem;
    if let Some(b) = s.strip_suffix("_degraded") {
        s = b;
    }
    s = strip_degrade_variant(s);
    s = strip_aug_variant(s);
    s.to_string()
}

/// Every piece whose ids straddle more than one split is a leakage hole:
// today `assign_splits` hashes each page/aug id independently, so pages of
// one score routinely land in different splits.
fn find_split_leaks(assignment: &BTreeMap<String, String>) -> Vec<PairProblem> {
    let mut by_piece: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();
    for (id, split) in assignment {
        by_piece
            .entry(piece_of(id))
            .or_default()
            .entry(split.clone())
            .or_default()
            .push(id.clone());
    }
    let mut out = Vec::new();
    for (piece, per_split) in &by_piece {
        if per_split.len() > 1 {
            let mut detail: Vec<String> = per_split
                .iter()
                .map(|(split, ids)| {
                    let mut ids = ids.clone();
                    ids.sort();
                    format!("{split}: {}", ids.join(", "))
                })
                .collect();
            detail.sort();
            out.push(PairProblem {
                id: piece.clone(),
                kind: "split-leakage".into(),
                detail: format!("piece straddles splits ({})", detail.join(" | ")),
            });
        }
    }
    out
}

// ----- box schema + bounds -----

/// Box sidecars: `<stem>-boxes.json` anywhere under `root`, keyed by the
/// stem minus the `-boxes` suffix (`ds0001-1-boxes.json` -> `ds0001-1`).
fn discover_box_files(root: &Path) -> BTreeMap<String, PathBuf> {
    let mut map = BTreeMap::new();
    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let p = entry.path();
        let is_json = p.extension().and_then(|e| e.to_str()) == Some("json");
        if !is_json {
            continue;
        }
        if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
            if let Some(base) = stem.strip_suffix("-boxes") {
                map.insert(base.to_string(), p.to_path_buf());
            }
        }
    }
    map
}

/// Validate every `*-boxes.json` sidecar; returns (class histogram, problems).
/// Glyphs need `cls` (non-empty string) + `bbox` ([x0,y0,x1,y1], finite,
/// non-degenerate, inside the page bounds from `png_size`). When any sidecar
/// exists, every valid pair (or its label-sharing variants) needs one.
fn strict_box_problems(
    root: &Path,
    pairs: &[(String, SideFile, SideFile)],
) -> (BTreeMap<String, usize>, Vec<PairProblem>) {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut problems = Vec::new();
    let files = discover_box_files(root);
    if !files.is_empty() {
        for (id, _, _) in pairs {
            let stem = id.rsplit('/').next().unwrap_or(id);
            if !files.contains_key(stem) && !files.contains_key(label_key(stem).as_str()) {
                problems.push(PairProblem {
                    id: id.clone(),
                    kind: "box-missing".into(),
                    detail: "valid pair has no <stem>-boxes.json sidecar".into(),
                });
            }
        }
    }
    for (key, path) in &files {
        let rel = rel_posix(root, path);
        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                problems.push(PairProblem {
                    id: key.clone(),
                    kind: "box-unreadable".into(),
                    detail: format!("{rel}: {e}"),
                });
                continue;
            }
        };
        let v: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                problems.push(PairProblem {
                    id: key.clone(),
                    kind: "box-unreadable".into(),
                    detail: format!("{rel}: invalid JSON: {e}"),
                });
                continue;
            }
        };
        let glyphs = v.get("glyphs").unwrap_or(&v);
        let arr = match glyphs.as_array() {
            Some(a) => a,
            None => {
                problems.push(PairProblem {
                    id: key.clone(),
                    kind: "box-schema".into(),
                    detail: format!("{rel}: expected `glyphs` array of {{cls, bbox}}"),
                });
                continue;
            }
        };
        let mut bounds: Option<(f64, f64)> = v
            .get("png_size")
            .and_then(|s| s.as_array())
            .filter(|a| a.len() == 2)
            .and_then(|a| {
                let w = a[0].as_f64()?;
                let h = a[1].as_f64()?;
                (w.is_finite() && h.is_finite() && w > 0.0 && h > 0.0).then_some((w, h))
            });
        if bounds.is_none() {
            bounds = files_png_fallback(path, key);
        }
        for (i, g) in arr.iter().enumerate() {
            let cls = g
                .get("cls")
                .and_then(|c| c.as_str())
                .filter(|c| !c.is_empty());
            let bbox = g.get("bbox").and_then(|b| b.as_array());
            let (Some(cls), Some(bbox)) = (cls, bbox) else {
                problems.push(PairProblem {
                    id: key.clone(),
                    kind: "box-schema".into(),
                    detail: format!("{rel}: glyph #{i} needs `cls` string + `bbox` array"),
                });
                continue;
            };
            if bbox.len() != 4 {
                problems.push(PairProblem {
                    id: key.clone(),
                    kind: "box-schema".into(),
                    detail: format!(
                        "{rel}: glyph #{i} bbox needs 4 numbers, found {}",
                        bbox.len()
                    ),
                });
                continue;
            }
            let nums: Vec<f64> = bbox.iter().filter_map(|n| n.as_f64()).collect();
            if nums.len() != 4 || nums.iter().any(|n| !n.is_finite()) {
                problems.push(PairProblem {
                    id: key.clone(),
                    kind: "box-schema".into(),
                    detail: format!("{rel}: glyph #{i} bbox must be 4 finite numbers"),
                });
                continue;
            }
            let [x0, y0, x1, y1] = [nums[0], nums[1], nums[2], nums[3]];
            if !(x1 > x0 && y1 > y0) {
                problems.push(PairProblem {
                    id: key.clone(),
                    kind: "box-degenerate".into(),
                    detail: format!("{rel}: glyph #{i} (`{cls}`) has non-positive area"),
                });
                continue;
            }
            if let Some((w, h)) = bounds {
                if x0 < 0.0 || y0 < 0.0 || x1 > w || y1 > h {
                    problems.push(PairProblem {
                        id: key.clone(),
                        kind: "box-out-of-bounds".into(),
                        detail: format!(
                            "{rel}: glyph #{i} (`{cls}`) [{x0:.1},{y0:.1},{x1:.1},{y1:.1}] outside {w:.0}x{h:.0}"
                        ),
                    });
                    continue;
                }
            }
            *counts.entry(cls.to_string()).or_insert(0) += 1;
        }
    }
    (counts, problems)
}

/// Bounds fallback: same-dir `<key>.png` IHDR dims when the sidecar has no
/// `png_size` (symbox files always carry `png_size`; this covers hand-made
/// sidecars without adding an image dependency).
fn files_png_fallback(boxes_path: &Path, key: &str) -> Option<(f64, f64)> {
    let dir = boxes_path.parent()?;
    for ext in ["png", "PNG"] {
        let cand = dir.join(format!("{key}.{ext}"));
        if let Some((w, h)) = png_dims(&cand) {
            return Some((w as f64, h as f64));
        }
    }
    None
}

/// PNG IHDR width/height without an image dependency.
fn png_dims(path: &Path) -> Option<(u32, u32)> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() < 24 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let h = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    (w > 0 && h > 0).then_some((w, h))
}

// ----- class histogram -----

/// Kept-class coverage + single-class dominance over the strict box corpus.
/// Mirrors the old imbalance warning, escalated: in strict mode a kept class
/// with zero boxes, or one class holding >95% of boxes, fails the pack.
fn strict_histogram_problems(
    counts: &BTreeMap<String, usize>,
    expected: &[String],
) -> Vec<PairProblem> {
    let mut problems = Vec::new();
    for c in expected {
        if counts.get(c).copied().unwrap_or(0) == 0 {
            problems.push(PairProblem {
                id: "corpus".into(),
                kind: "class-absent".into(),
                detail: format!("kept class `{c}` has zero boxes"),
            });
        }
    }
    let total: usize = counts.values().sum();
    if total > 0 {
        for (c, n) in counts {
            let frac = *n as f64 / total as f64;
            if frac > 0.95 {
                problems.push(PairProblem {
                    id: "corpus".into(),
                    kind: "class-dominance".into(),
                    detail: format!("class `{c}` holds {n}/{total} boxes ({:.1}%)", frac * 100.0),
                });
            }
        }
    }
    problems
}

// ----- degradation-param linkage -----

/// If `stem` looks like a degraded output, return the params filenames to
/// try (in the image's dir) plus the base stem the params must reference.
/// photo-degrade writes `<stem>{suffix}_params.json`, so `_d<N>` variants
/// link `<full-stem>_params.json` while legacy `_degraded` outputs also
/// accept `<base>_params.json`.
fn degraded_link(stem: &str) -> Option<(Vec<String>, String)> {
    if let Some(base) = stem.strip_suffix("_degraded") {
        return Some((
            vec![format!("{stem}_params.json"), format!("{base}_params.json")],
            base.to_string(),
        ));
    }
    if let Some(pos) = stem.rfind("_d") {
        let tail = &stem[pos + 2..];
        if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) && pos > 0 {
            return Some((vec![format!("{stem}_params.json")], stem[..pos].to_string()));
        }
    }
    None
}

/// Every degraded image must link an existing params JSON recording its
/// seed (`file_seed`/`seed`) and stem (`input`/`output` contains the base).
fn strict_params_problems(root: &Path, pairs: &[(String, SideFile, SideFile)]) -> Vec<PairProblem> {
    let mut problems = Vec::new();
    for (id, img, _) in pairs {
        let rel = Path::new(&img.path);
        let Some(stem) = rel.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some((candidates, base)) = degraded_link(stem) else {
            continue;
        };
        let dir = rel
            .parent()
            .map(|p| root.join(p))
            .unwrap_or_else(|| root.to_path_buf());
        let found = candidates.iter().map(|c| dir.join(c)).find(|p| p.is_file());
        let Some(path) = found else {
            problems.push(PairProblem {
                id: id.clone(),
                kind: "params-missing".into(),
                detail: format!(
                    "degraded image `{}` has no {}",
                    img.path,
                    candidates.join(" or ")
                ),
            });
            continue;
        };
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                problems.push(PairProblem {
                    id: id.clone(),
                    kind: "params-unreadable".into(),
                    detail: format!("{}: {e}", rel_posix(root, &path)),
                });
                continue;
            }
        };
        let v: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                problems.push(PairProblem {
                    id: id.clone(),
                    kind: "params-unreadable".into(),
                    detail: format!("{}: invalid JSON: {e}", rel_posix(root, &path)),
                });
                continue;
            }
        };
        let has_seed = v
            .get("file_seed")
            .or_else(|| v.get("seed"))
            .is_some_and(|n| n.is_number());
        if !has_seed {
            problems.push(PairProblem {
                id: id.clone(),
                kind: "params-seed-missing".into(),
                detail: format!("{}: no `file_seed`/`seed` field", rel_posix(root, &path)),
            });
        }
        let linked = ["input", "output"]
            .iter()
            .filter_map(|k| v.get(k).and_then(|s| s.as_str()))
            .any(|r| r.contains(&base));
        if !linked {
            problems.push(PairProblem {
                id: id.clone(),
                kind: "params-stem-mismatch".into(),
                detail: format!(
                    "{}: neither `input` nor `output` references stem `{base}`",
                    rel_posix(root, &path)
                ),
            });
        }
    }
    problems
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
    }
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    fs::write(path, text + "\n").map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEST_SEQ: AtomicU64 = AtomicU64::new(0);

    /// Unique scratch dir per test (parallel-safe).
    fn scratch(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let seq = TEST_SEQ.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "dataset-pack-strict-{name}-{}-{seq}-{}",
            std::process::id(),
            nanos
        ));
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn write_file(dir: &Path, rel: &str, content: &[u8]) -> PathBuf {
        let p = dir.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).expect("mkdirs");
        }
        fs::write(&p, content).expect("write");
        p
    }

    fn pair(id: &str, image: &str) -> (String, SideFile, SideFile) {
        (
            id.to_string(),
            SideFile {
                path: image.to_string(),
                bytes: 8,
                sha256: "aa".into(),
            },
            SideFile {
                path: format!("{id}.musicxml"),
                bytes: 8,
                sha256: "bb".into(),
            },
        )
    }

    fn assignment(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(id, split)| ((*id).to_string(), (*split).to_string()))
            .collect()
    }

    fn kinds(problems: &[PairProblem]) -> Vec<String> {
        let mut k: Vec<String> = problems.iter().map(|p| p.kind.clone()).collect();
        k.sort();
        k.dedup();
        k
    }

    // ----- pairing -----

    #[test]
    fn strict_sidecars_do_not_break_pairing() {
        // Label + params sidecars are pack inputs the strict gates consume;
        // pairing must not flag them as unknown extensions.
        let dir = scratch("sidecar-pairing");
        write_file(&dir, "a-1.png", b"png");
        write_file(&dir, "a-1.musicxml", b"mxl");
        write_file(&dir, "a-1-boxes.json", boxes_doc("").as_bytes());
        write_file(&dir, "a-1_d0_params.json", b"{}");
        let rep = validate_dir(&dir).expect("validate");
        assert_eq!(rep.invalid, 0, "problems: {:?}", rep.problems);
        assert_eq!(rep.valid, 1);
    }

    #[test]
    fn junk_extensions_still_break_pairing() {
        let dir = scratch("junk-pairing");
        write_file(&dir, "a-1.png", b"png");
        write_file(&dir, "a-1.musicxml", b"mxl");
        write_file(&dir, "notes.txt", b"junk");
        let rep = validate_dir(&dir).expect("validate");
        assert!(rep.problems.iter().any(|p| p.kind == "unknown-extension"));
    }

    // ----- piece keys -----

    #[test]
    fn piece_groups_pages_and_aug_variants() {
        assert_eq!(piece_of("ds0001/ds0001-1"), "ds0001");
        assert_eq!(piece_of("ds0001-1"), "ds0001");
        assert_eq!(piece_of("ds0001-1.aug0"), "ds0001");
        assert_eq!(piece_of("ds0001-2_aug3"), "ds0001");
        assert_eq!(piece_of("ds0001-1_d0"), "ds0001");
        assert_eq!(piece_of("sample01-1_degraded"), "sample01");
        assert_eq!(piece_of("plain"), "plain");
        // Non-numeric tails are not page suffixes.
        assert_eq!(piece_of("r0000v0d"), "r0000v0d");
    }

    // ----- split integrity -----

    #[test]
    fn leakage_detector_flags_straddling_aug_variants() {
        let a = assignment(&[
            ("ds0001-1", "train"),
            ("ds0001-2", "val"),
            ("ds0001-1.aug0", "val"),
            ("ds0002-1", "train"),
        ]);
        let leaks = find_split_leaks(&a);
        assert_eq!(leaks.len(), 1);
        assert_eq!(leaks[0].kind, "split-leakage");
        assert_eq!(leaks[0].id, "ds0001");
        assert!(leaks[0].detail.contains("train"));
        assert!(leaks[0].detail.contains("val"));
    }

    #[test]
    fn no_leak_when_piece_shares_one_split() {
        let a = assignment(&[
            ("ds0001-1", "train"),
            ("ds0001-2", "train"),
            ("ds0001-1.aug0", "train"),
            ("ds0002-1", "val"),
        ]);
        assert!(find_split_leaks(&a).is_empty());
    }

    // ----- box schema -----

    fn boxes_doc(glyphs: &str) -> String {
        format!("{{\"png_size\": [100, 200], \"glyphs\": [{glyphs}]}}")
    }

    #[test]
    fn box_schema_rejects_missing_keys() {
        let dir = scratch("box-schema");
        write_file(
            &dir,
            "p1-boxes.json",
            boxes_doc(
                "{\"cls\": \"NoteBlack\", \"bbox\": [1, 2, 3, 4]}, \
                 {\"cls\": \"Rest\"}, \
                 {\"bbox\": [1, 2, 3, 4]}, \
                 {\"cls\": \"\", \"bbox\": [1, 2, 3, 4]}, \
                 {\"cls\": \"Clef\", \"bbox\": [1, 2, 3]}, \
                 {\"cls\": \"Clef\", \"bbox\": [1, 2, \"x\", 4]}",
            )
            .as_bytes(),
        );
        let (_, problems) = strict_box_problems(&dir, &[]);
        let schema: Vec<_> = problems.iter().filter(|p| p.kind == "box-schema").collect();
        // Missing bbox, missing cls, empty cls, 3-elem bbox, string coord.
        assert_eq!(schema.len(), 5, "problems: {problems:?}");
    }

    #[test]
    fn box_out_of_bounds_and_degenerate_flagged() {
        let dir = scratch("box-bounds");
        write_file(
            &dir,
            "p1-boxes.json",
            boxes_doc(
                "{\"cls\": \"NoteBlack\", \"bbox\": [10, 10, 20, 20]}, \
                 {\"cls\": \"Rest\", \"bbox\": [90, 10, 120, 20]}, \
                 {\"cls\": \"Clef\", \"bbox\": [5, 5, 5, 9]}, \
                 {\"cls\": \"Beam\", \"bbox\": [-1, 0, 5, 5]}",
            )
            .as_bytes(),
        );
        let (counts, problems) = strict_box_problems(&dir, &[]);
        // Only the in-bounds box counts toward the histogram.
        assert_eq!(counts.get("NoteBlack"), Some(&1));
        assert_eq!(counts.get("Rest"), None);
        let k = kinds(&problems);
        assert!(k.contains(&"box-out-of-bounds".to_string()), "{k:?}");
        assert!(k.contains(&"box-degenerate".to_string()), "{k:?}");
        assert_eq!(
            problems
                .iter()
                .filter(|p| p.kind == "box-out-of-bounds")
                .count(),
            2
        );
    }

    #[test]
    fn box_missing_fires_only_when_sidecars_exist() {
        // No sidecars anywhere: no box-missing (labels simply absent).
        let dir = scratch("box-none");
        let (_, problems) = strict_box_problems(&dir, &[pair("a-1", "a-1.png")]);
        assert!(problems.is_empty());
        // One sidecar: pairs without (a share of) one are reported.
        write_file(&dir, "a-1-boxes.json", boxes_doc("").as_bytes());
        let (_, problems) =
            strict_box_problems(&dir, &[pair("a-1", "a-1.png"), pair("b-2", "b-2.png")]);
        assert_eq!(kinds(&problems), vec!["box-missing".to_string()]);
        assert_eq!(problems[0].id, "b-2");
        // Aug variants share their page's labels.
        let (_, problems) = strict_box_problems(
            &dir,
            &[pair("a-1", "a-1.png"), pair("a-1.aug0", "a-1.aug0.png")],
        );
        assert!(
            problems.iter().all(|p| p.kind != "box-missing"),
            "{problems:?}"
        );
    }

    // ----- histogram -----

    #[test]
    fn histogram_flags_absent_and_dominant_classes() {
        let counts: BTreeMap<String, usize> =
            [("NoteBlack".to_string(), 96), ("Rest".to_string(), 4)]
                .into_iter()
                .collect();
        let expected = vec!["NoteBlack".into(), "Rest".into(), "Flag".into()];
        let problems = strict_histogram_problems(&counts, &expected);
        let k = kinds(&problems);
        assert_eq!(
            k,
            vec!["class-absent".to_string(), "class-dominance".to_string()]
        );
        // Balanced coverage passes.
        let counts: BTreeMap<String, usize> =
            [("NoteBlack".to_string(), 5), ("Rest".to_string(), 5)]
                .into_iter()
                .collect();
        let expected = vec!["NoteBlack".into(), "Rest".into()];
        assert!(strict_histogram_problems(&counts, &expected).is_empty());
    }

    // ----- params linkage -----

    #[test]
    fn params_missing_for_degraded_image_without_sidecar() {
        let dir = scratch("params-missing");
        write_file(&dir, "ds0001-1_d0.jpg", b"fakejpg");
        let pairs = vec![pair("ds0001-1_d0", "ds0001-1_d0.jpg")];
        let problems = strict_params_problems(&dir, &pairs);
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].kind, "params-missing");
        // Plain renders are exempt.
        write_file(&dir, "clean.png", b"fakepng");
        let pairs = vec![pair("clean", "clean.png")];
        assert!(strict_params_problems(&dir, &pairs).is_empty());
    }

    #[test]
    fn params_link_ok_with_matching_seed_and_stem() {
        let dir = scratch("params-ok");
        write_file(&dir, "ds0001-1_d0.jpg", b"fakejpg");
        write_file(
            &dir,
            "ds0001-1_d0_params.json",
            br#"{"input": "ds0001-1.png", "output": "ds0001-1_d0.jpg",
                 "file_seed": 7, "backend": "cpu"}"#,
        );
        let pairs = vec![pair("ds0001-1_d0", "ds0001-1_d0.jpg")];
        assert!(strict_params_problems(&dir, &pairs).is_empty());
    }

    #[test]
    fn params_seed_and_stem_mismatch_flagged() {
        let dir = scratch("params-bad");
        write_file(&dir, "a-1_d0.jpg", b"fakejpg");
        write_file(
            &dir,
            "a-1_d0_params.json",
            br#"{"input": "other.png", "output": "other_d0.jpg"}"#,
        );
        write_file(&dir, "b-2_degraded.jpg", b"fakejpg");
        write_file(
            &dir,
            "b-2_params.json",
            br#"{"seed": 3, "input": "zzz.png"}"#,
        );
        let pairs = vec![
            pair("a-1_d0", "a-1_d0.jpg"),
            pair("b-2_degraded", "b-2_degraded.jpg"),
        ];
        let problems = strict_params_problems(&dir, &pairs);
        let k = kinds(&problems);
        assert!(k.contains(&"params-seed-missing".to_string()), "{k:?}");
        assert!(k.contains(&"params-stem-mismatch".to_string()), "{k:?}");
    }
}
