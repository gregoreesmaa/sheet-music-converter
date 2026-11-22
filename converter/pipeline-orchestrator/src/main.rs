//! pipeline-orchestrator: deterministic, resumable
//! `ingest -> layout -> symbols -> serialize` conversion driver.
//!
//! Reads a TOML config, derives every layout box and symbol
//! deterministically from `(seed, stage, page)` so reruns over the same
//! inputs are bit-identical, and records each finished work unit in a
//! state file so an interrupted run can continue with `--resume`.
//!
//! Zero third-party dependencies (std only) so workers build offline.
//! The layout/symbol stages are deterministic placeholders behind a
//! stable `(seed, page)` contract: swap them for real detectors without
//! changing the CLI, config, state, or on-disk layout.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const STAGES: [&str; 4] = ["ingest", "layout", "symbols", "serialize"];
const STATE_VERSION: &str = "v1";
const KINDS: [&str; 6] = ["note", "rest", "clef", "sharp", "flat", "barline"];
const STEPS: [&str; 7] = ["C", "D", "E", "F", "G", "A", "B"];

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct Args {
    config: Option<String>,
    resume: bool,
    from: Option<String>,
    only: Option<String>,
    dry_run: bool,
    seed: Option<u64>,
    fail_after: Option<usize>,
}

fn print_help() {
    println!(
        "pipeline-orchestrator {}\n\
         Deterministic, resumable ingest -> layout -> symbols -> serialize driver.\n\
         \n\
         USAGE:\n  \
         pipeline-orchestrator [OPTIONS]\n\
         \n\
         OPTIONS:\n  \
         -c, --config <PATH>   Config file (default: config.toml next to CWD)\n      \
         --resume            Skip work units already recorded in the state file\n      \
         --from <STAGE>      Start at STAGE, skipping earlier stages\n      \
         --only <STAGE>      Run only STAGE (its inputs must exist from a prior run)\n      \
         --dry-run           Print the plan without writing anything\n      \
         --seed <U64>        Override [global].seed for this run\n      \
         --fail-after <N>    Exit(2) after N units (test aid for --resume)\n  \
         -h, --help            Print this help",
        env!("CARGO_PKG_VERSION")
    );
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "-c" | "--config" => {
                i += 1;
                a.config = Some(arg_value(argv, i, "--config")?);
            }
            "--resume" => a.resume = true,
            "--from" => {
                i += 1;
                a.from = Some(arg_value(argv, i, "--from")?);
            }
            "--only" => {
                i += 1;
                a.only = Some(arg_value(argv, i, "--only")?);
            }
            "--dry-run" => a.dry_run = true,
            "--seed" => {
                i += 1;
                let s = arg_value(argv, i, "--seed")?;
                a.seed = Some(
                    s.parse::<u64>()
                        .map_err(|_| "--seed needs an unsigned integer")?,
                );
            }
            "--fail-after" => {
                i += 1;
                let s = arg_value(argv, i, "--fail-after")?;
                a.fail_after = Some(
                    s.parse::<usize>()
                        .map_err(|_| "--fail-after needs a number")?,
                );
            }
            other => return Err(format!("unknown argument: {other} (see --help)")),
        }
        i += 1;
    }
    if let Some(s) = &a.from {
        check_stage("--from", s)?;
    }
    if let Some(s) = &a.only {
        check_stage("--only", s)?;
    }
    if a.from.is_some() && a.only.is_some() {
        return Err("--from and --only are mutually exclusive".to_string());
    }
    Ok(a)
}

fn arg_value(argv: &[String], i: usize, flag: &str) -> Result<String, String> {
    argv.get(i)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn check_stage(flag: &str, s: &str) -> Result<(), String> {
    if STAGES.contains(&s) {
        Ok(())
    } else {
        Err(format!("{flag} must be one of: {}", STAGES.join(", ")))
    }
}

// ---------------------------------------------------------------------------
// Config (minimal TOML for our known schema; std only)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Debug, Clone)]
struct Config {
    seed: u64,
    out_dir: PathBuf,
    state_file: PathBuf,
    ingest: IngestCfg,
    layout: LayoutCfg,
    symbols: SymbolsCfg,
    serialize: SerializeCfg,
}

#[derive(Debug, Clone)]
struct IngestCfg {
    enabled: bool,
    input_dir: PathBuf,
    extensions: Vec<String>,
}

#[derive(Debug, Clone)]
struct LayoutCfg {
    enabled: bool,
    systems_per_page: usize,
    page_width: u32,
    page_height: u32,
}

#[derive(Debug, Clone)]
struct SymbolsCfg {
    enabled: bool,
    symbols_per_system: usize,
    min_confidence: f64,
}

#[derive(Debug, Clone)]
struct SerializeCfg {
    enabled: bool,
    format: String,
}

/// Strip a `#` comment that is not inside double quotes.
fn strip_comment(line: &str) -> &str {
    let mut in_str = false;
    let mut prev = '\0';
    for (idx, ch) in line.char_indices() {
        if ch == '"' && prev != '\\' {
            in_str = !in_str;
        } else if ch == '#' && !in_str {
            return line[..idx].trim_end();
        }
        prev = ch;
    }
    line
}

fn parse_toml(text: &str) -> Result<HashMap<(String, String), Value>, String> {
    let mut map = HashMap::new();
    let mut section = String::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].trim().to_string();
            if section.is_empty() {
                return Err(format!("line {}: empty section", lineno + 1));
            }
            continue;
        }
        let (k, v) = line
            .split_once('=')
            .ok_or_else(|| format!("line {}: expected `key = value`", lineno + 1))?;
        let key = k.trim().to_string();
        let raw_v = v.trim();
        if section.is_empty() || key.is_empty() {
            return Err(format!(
                "line {}: keys must live under a [section]",
                lineno + 1
            ));
        }
        let value = if raw_v.starts_with('"') {
            let inner = raw_v
                .strip_suffix('"')
                .ok_or_else(|| format!("line {}: unterminated string", lineno + 1))?;
            Value::Str(inner[1..].to_string())
        } else if raw_v == "true" {
            Value::Bool(true)
        } else if raw_v == "false" {
            Value::Bool(false)
        } else if let Ok(n) = raw_v.parse::<i64>() {
            Value::Int(n)
        } else if let Ok(f) = raw_v.parse::<f64>() {
            Value::Float(f)
        } else {
            return Err(format!("line {}: cannot parse value `{raw_v}`", lineno + 1));
        };
        map.insert((section.clone(), key), value);
    }
    Ok(map)
}

fn get_str(
    m: &HashMap<(String, String), Value>,
    s: &str,
    k: &str,
) -> Result<Option<String>, String> {
    match m.get(&(s.to_string(), k.to_string())) {
        None => Ok(None),
        Some(Value::Str(v)) => Ok(Some(v.clone())),
        Some(other) => Err(format!("[{s}].{k} must be a string, got `{other:?}`")),
    }
}

fn get_i64(m: &HashMap<(String, String), Value>, s: &str, k: &str) -> Result<Option<i64>, String> {
    match m.get(&(s.to_string(), k.to_string())) {
        None => Ok(None),
        Some(Value::Int(v)) => Ok(Some(*v)),
        Some(other) => Err(format!("[{s}].{k} must be an integer, got `{other:?}`")),
    }
}

fn get_num(m: &HashMap<(String, String), Value>, s: &str, k: &str) -> Result<Option<f64>, String> {
    match m.get(&(s.to_string(), k.to_string())) {
        None => Ok(None),
        Some(Value::Int(v)) => Ok(Some(*v as f64)),
        Some(Value::Float(v)) => Ok(Some(*v)),
        Some(other) => Err(format!("[{s}].{k} must be a number, got `{other:?}`")),
    }
}

fn get_bool(
    m: &HashMap<(String, String), Value>,
    s: &str,
    k: &str,
) -> Result<Option<bool>, String> {
    match m.get(&(s.to_string(), k.to_string())) {
        None => Ok(None),
        Some(Value::Bool(v)) => Ok(Some(*v)),
        Some(other) => Err(format!("[{s}].{k} must be true/false, got `{other:?}`")),
    }
}

fn load_config(path: &Path, seed_override: Option<u64>) -> Result<Config, String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let m = parse_toml(&text)?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));

    let seed = match seed_override {
        Some(s) => s,
        None => {
            let n = get_i64(&m, "global", "seed")?.unwrap_or(1);
            if n < 0 {
                return Err("[global].seed must be >= 0".to_string());
            }
            n as u64
        }
    };
    let resolve = |p: &str| {
        let pb = PathBuf::from(p);
        if pb.is_absolute() {
            pb
        } else {
            base.join(pb)
        }
    };
    let out_dir = resolve(&get_str(&m, "global", "out_dir")?.unwrap_or_else(|| "out".to_string()));
    let state_file = match get_str(&m, "global", "state_file")? {
        Some(p) => resolve(&p),
        None => out_dir.join(".pipeline-state"),
    };

    let en = |s: &str| -> Result<bool, String> { Ok(get_bool(&m, s, "enabled")?.unwrap_or(true)) };

    let systems = get_i64(&m, "layout", "systems_per_page")?.unwrap_or(4);
    if !(1..=32).contains(&systems) {
        return Err("[layout].systems_per_page must be in 1..=32".to_string());
    }
    let page_width = get_i64(&m, "layout", "page_width")?.unwrap_or(2480);
    if page_width <= 0 {
        return Err("[layout].page_width must be > 0".to_string());
    }
    let page_height = get_i64(&m, "layout", "page_height")?.unwrap_or(3508);
    if page_height <= 0 {
        return Err("[layout].page_height must be > 0".to_string());
    }

    let per_system = get_i64(&m, "symbols", "symbols_per_system")?.unwrap_or(24);
    if !(1..=512).contains(&per_system) {
        return Err("[symbols].symbols_per_system must be in 1..=512".to_string());
    }
    let min_conf = get_num(&m, "symbols", "min_confidence")?.unwrap_or(0.5);
    if !(0.0..=1.0).contains(&min_conf) {
        return Err("[symbols].min_confidence must be in 0.0..=1.0".to_string());
    }

    let format = get_str(&m, "serialize", "format")?.unwrap_or_else(|| "json".to_string());
    if format != "json" && format != "musicxml" {
        return Err("[serialize].format must be \"json\" or \"musicxml\"".to_string());
    }

    let extensions = get_str(&m, "ingest", "extensions")?
        .unwrap_or_else(|| "png,jpg,jpeg,tif,tiff,bmp".to_string())
        .split(',')
        .map(|e| e.trim().trim_start_matches('.').to_lowercase())
        .filter(|e| !e.is_empty())
        .collect::<Vec<_>>();
    if extensions.is_empty() {
        return Err("[ingest].extensions must list at least one extension".to_string());
    }

    Ok(Config {
        seed,
        out_dir,
        state_file,
        ingest: IngestCfg {
            enabled: en("ingest")?,
            input_dir: resolve(
                &get_str(&m, "ingest", "input_dir")?.unwrap_or_else(|| "samples".to_string()),
            ),
            extensions,
        },
        layout: LayoutCfg {
            enabled: en("layout")?,
            systems_per_page: systems as usize,
            page_width: page_width as u32,
            page_height: page_height as u32,
        },
        symbols: SymbolsCfg {
            enabled: en("symbols")?,
            symbols_per_system: per_system as usize,
            min_confidence: min_conf,
        },
        serialize: SerializeCfg {
            enabled: en("serialize")?,
            format,
        },
    })
}

fn stage_enabled(cfg: &Config, stage: &str) -> bool {
    match stage {
        "ingest" => cfg.ingest.enabled,
        "layout" => cfg.layout.enabled,
        "symbols" => cfg.symbols.enabled,
        "serialize" => cfg.serialize.enabled,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Deterministic hashing + RNG (SplitMix64 over an FNV-1a stream key)
// ---------------------------------------------------------------------------

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Key stream for one (seed, stage, page): stable across runs and resume
/// slices, so the same config + inputs always yield identical outputs.
fn stream_key(seed: u64, stage: &str, page: &str) -> u64 {
    let mut buf = Vec::with_capacity(32);
    buf.extend_from_slice(&seed.to_le_bytes());
    buf.extend_from_slice(stage.as_bytes());
    buf.push(0);
    buf.extend_from_slice(page.as_bytes());
    fnv1a(&buf)
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        self.next() % n
    }

    fn unit(&mut self) -> f64 {
        // f64 in [0, 1): top 53 bits over 2^53.
        ((self.next() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        debug_assert!(hi >= lo);
        lo + (self.below((hi - lo + 1) as u64) as i64)
    }
}

// ---------------------------------------------------------------------------
// Small JSON helpers (std only)
// ---------------------------------------------------------------------------

fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

// ---------------------------------------------------------------------------
// Pages + state file
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Page {
    /// Unit key: file stem, e.g. `page-001`.
    name: String,
    source: PathBuf,
}

fn discover_pages(cfg: &Config) -> Result<Vec<Page>, String> {
    let dir = &cfg.ingest.input_dir;
    let entries = fs::read_dir(dir)
        .map_err(|e| format!("cannot list [ingest].input_dir {}: {e}", dir.display()))?;
    let mut names: Vec<String> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot read entry in {}: {e}", dir.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let lower = name.to_lowercase();
        if cfg
            .ingest
            .extensions
            .iter()
            .any(|ext| lower.ends_with(&format!(".{ext}")))
        {
            names.push(name);
        }
    }
    names.sort();
    if names.is_empty() {
        return Err(format!(
            "no page images (extensions: {}) in {}",
            cfg.ingest.extensions.join(","),
            dir.display()
        ));
    }
    Ok(names
        .into_iter()
        .map(|n| {
            let stem = n.rsplit_once('.').map(|(s, _)| s).unwrap_or(&n).to_string();
            Page {
                name: stem,
                source: dir.join(&n),
            }
        })
        .collect())
}

/// Recover the page list from a previous ingest run's outputs, for
/// `--from` / `--only` slices that skip ingest.
fn recover_pages(cfg: &Config) -> Result<Vec<Page>, String> {
    let dir = cfg.out_dir.join("ingest");
    let mut names: Vec<String> = Vec::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".json") {
                names.push(name[..name.len() - 5].to_string());
            }
        }
    }
    names.sort();
    if names.is_empty() {
        return Err(format!(
            "ingest is skipped but no prior ingest outputs exist in {}; \
             run the full pipeline or start --from ingest",
            dir.display()
        ));
    }
    Ok(names
        .into_iter()
        .map(|stem| Page {
            source: cfg.ingest.input_dir.join(format!("{stem}.*")),
            name: stem,
        })
        .collect())
}

fn load_state(path: &Path) -> HashSet<(String, String)> {
    let mut set = HashSet::new();
    let Ok(text) = fs::read_to_string(path) else {
        return set;
    };
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        match (parts.next(), parts.next(), parts.next()) {
            (Some(v), Some(stage), Some(unit)) if v == STATE_VERSION => {
                set.insert((stage.to_string(), unit.to_string()));
            }
            _ => {}
        }
    }
    set
}

fn append_state(path: &Path, stage: &str, unit: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create state dir {}: {e}", parent.display()))?;
    }
    use std::io::Write as _;
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("cannot open state file {}: {e}", path.display()))?;
    writeln!(f, "{STATE_VERSION} {stage} {unit}")
        .map_err(|e| format!("cannot write state file {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Stage data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SystemBox {
    id: usize,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

#[derive(Debug, Clone)]
struct Symbol {
    id: String,
    system: usize,
    kind: String,
    step: String,
    octave: u8,
    x: u32,
    confidence: f64,
}

/// Canonical `(seed, page)` -> staff-system boxes. Every stage that needs
/// boxes calls this, so layout outputs, symbols, and scores can never drift.
fn derive_systems(cfg: &Config, page: &Page) -> Vec<SystemBox> {
    let mut rng = SplitMix64(stream_key(cfg.seed, "layout", &page.name));
    let n = cfg.layout.systems_per_page;
    let w = cfg.layout.page_width;
    let h = cfg.layout.page_height;
    let margin = (w / 20).max(16);
    let band = h / (n as u32 + 1);
    let mut systems = Vec::with_capacity(n);
    for id in 0..n {
        let jitter_y = rng.range(-(band as i64) / 16, band as i64 / 16);
        let jitter_h = rng.range(-(band as i64) / 12, band as i64 / 12);
        let y = ((band * (id as u32 + 1)) as i64 + jitter_y).max(0) as u32;
        let sys_h = ((band * 2 / 3) as i64 + jitter_h).max(32) as u32;
        systems.push(SystemBox {
            id,
            x: margin + rng.below((margin / 2 + 1) as u64) as u32,
            y,
            w: w - 2 * margin,
            h: sys_h.min(h.saturating_sub(y)),
        });
    }
    systems
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create dir {}: {e}", parent.display()))?;
    }
    fs::write(path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

// --- ingest: validate the page image, record its identity ------------------

fn run_ingest(cfg: &Config, page: &Page) -> Result<String, String> {
    let bytes = fs::read(&page.source)
        .map_err(|e| format!("ingest: cannot read {}: {e}", page.source.display()))?;
    if bytes.is_empty() {
        return Err(format!("ingest: {} is empty", page.source.display()));
    }
    // Short content id: FNV-1a of the bytes (full bytes would bloat outputs).
    let digest = hex_bytes(&fnv1a(&bytes).to_le_bytes());
    let mut ext = String::from("bin");
    if let Some(e) = page.source.extension().and_then(|s| s.to_str()) {
        ext = e.to_lowercase();
    }
    let doc = format!(
        "{{\n  \"page\": \"{}\",\n  \"source\": \"{}\",\n  \"bytes\": {},\n  \"sha\": \"fnv1a:{}\",\n  \
         \"format\": \"{}\",\n  \"seed\": {}\n}}\n",
        esc(&page.name),
        esc(&page.source.to_string_lossy()),
        bytes.len(),
        digest,
        esc(&ext),
        cfg.seed,
    );
    write_file(&cfg.out_dir.join("ingest").join(format!("{}.json", page.name)), doc.as_bytes())?;
    Ok(digest)
}

// --- layout: deterministic staff-system boxes -------------------------------
// Placeholder for a real staff detector. Boxes tile the reference page
// with a seeded jitter; the contract (seed, page) -> boxes is what a
// future detector must preserve for reproducibility.

fn run_layout(cfg: &Config, page: &Page) -> Result<Vec<SystemBox>, String> {
    let ingest_path = cfg.out_dir.join("ingest").join(format!("{}.json", page.name));
    if !ingest_path.exists() {
        return Err(format!(
            "layout: missing ingest output for `{}` (run ingest first)",
            page.name
        ));
    }
    let systems = derive_systems(cfg, page);
    let w = cfg.layout.page_width;
    let h = cfg.layout.page_height;
    let mut doc = format!(
        "{{\n  \"page\": \"{}\",\n  \"seed\": {},\n  \"page_size\": [{}, {}],\n  \"systems\": [\n",
        esc(&page.name),
        cfg.seed,
        w,
        h
    );
    for (i, s) in systems.iter().enumerate() {
        let _ = write!(
            doc,
            "    {{\"id\": {}, \"x\": {}, \"y\": {}, \"w\": {}, \"h\": {}}}{}\n",
            s.id,
            s.x,
            s.y,
            s.w,
            s.h,
            if i + 1 < systems.len() { "," } else { "" },
        );
    }
    doc.push_str("  ]\n}\n");
    write_file(&cfg.out_dir.join("layout").join(format!("{}.json", page.name)), doc.as_bytes())?;
    Ok(systems)
}

// --- symbols: deterministic per-system symbol stream ------------------------

fn run_symbols(cfg: &Config, page: &Page, systems: &[SystemBox]) -> Result<Vec<Symbol>, String> {
    let mut rng = SplitMix64(stream_key(cfg.seed, "symbols", &page.name));
    let mut out = Vec::new();
    for sys in systems {
        for k in 0..cfg.symbols.symbols_per_system {
            let kind = KINDS[rng.below(KINDS.len() as u64) as usize].to_string();
            let step = STEPS[rng.below(STEPS.len() as u64) as usize].to_string();
            let octave = (4 + rng.below(2)) as u8;
            let span = sys.w.max(1) as u64;
            let x = sys.x + (rng.below(span + 1)) as u32;
            // Confidence in [0.5, 1.0); the floor keeps min_confidence
            // cutoffs meaningful while staying deterministic.
            let confidence = 0.5 + rng.unit() / 2.0;
            if confidence < cfg.symbols.min_confidence {
                continue;
            }
            out.push(Symbol {
                id: format!("{}-s{:02}-{:03}", page.name, sys.id, k),
                system: sys.id,
                kind,
                step,
                octave,
                x,
                confidence,
            });
        }
    }
    let mut doc = format!(
        "{{\n  \"page\": \"{}\",\n  \"seed\": {},\n  \"symbols\": [\n",
        esc(&page.name),
        cfg.seed
    );
    for (i, s) in out.iter().enumerate() {
        let _ = write!(
            doc,
            "    {{\"id\": \"{}\", \"system\": {}, \"kind\": \"{}\", \
             \"pitch\": \"{}{}\", \"x\": {}, \"confidence\": {:.4}}}{}\n",
            esc(&s.id),
            s.system,
            esc(&s.kind),
            s.step,
            s.octave,
            s.x,
            s.confidence,
            if i + 1 < out.len() { "," } else { "" },
        );
    }
    doc.push_str("  ]\n}\n");
    write_file(
        &cfg.out_dir.join("symbols").join(format!("{}.json", page.name)),
        doc.as_bytes(),
    )?;
    Ok(out)
}

fn read_systems_for_serialize(cfg: &Config, page: &Page) -> Result<Vec<SystemBox>, String> {
    // Re-derive deterministically rather than re-parsing JSON: same key,
    // same stream, same boxes the layout stage wrote.
    let systems = derive_systems(cfg, page);
    // Sanity: the layout file must exist (guards against --only misuse).
    let layout_path = cfg.out_dir.join("layout").join(format!("{}.json", page.name));
    if !layout_path.exists() {
        return Err(format!(
            "serialize: missing layout output for `{}` (run layout first)",
            page.name
        ));
    }
    let symbols_path = cfg.out_dir.join("symbols").join(format!("{}.json", page.name));
    if !symbols_path.exists() {
        return Err(format!(
            "serialize: missing symbols output for `{}` (run symbols first)",
            page.name
        ));
    }
    Ok(systems)
}

// --- serialize: final score per page + manifest ------------------------------

fn serialize_json(page: &Page, seed: u64, systems: &[SystemBox], symbols: &[Symbol]) -> String {
    let mut doc = format!(
        "{{\n  \"page\": \"{}\",\n  \"seed\": {},\n  \"part\": \"Piano\",\n  \"systems\": [\n",
        esc(&page.name),
        seed
    );
    for (si, sys) in systems.iter().enumerate() {
        let notes: Vec<&Symbol> = symbols
            .iter()
            .filter(|s| s.system == sys.id && s.kind == "note")
            .collect();
        doc.push_str(&format!("    {{\"id\": {}, \"notes\": [", sys.id));
        for (i, s) in notes.iter().enumerate() {
            let _ = write!(
                doc,
                "{{\"pitch\": \"{}{}\", \"x\": {}}}",
                s.step,
                s.octave,
                s.x
            );
            if i + 1 < notes.len() {
                doc.push_str(", ");
            }
        }
        doc.push_str(&format!(
            "]}}{}\n",
            if si + 1 < systems.len() { "," } else { "" }
        ));
    }
    let _ = write!(doc, "  ],\n  \"symbol_count\": {}\n}}\n", symbols.len());
    doc
}

fn serialize_musicxml(page: &Page, systems: &[SystemBox], symbols: &[Symbol]) -> String {
    let mut doc = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <score-partwise version=\"3.1\">\n  <movement-title>{}</movement-title>\n  \
         <part-list><score-part id=\"P1\"><part-name>Piano</part-name></score-part></part-list>\n  \
         <part id=\"P1\">\n",
        esc(&page.name)
    );
    let mut num = 1u32;
    for sys in systems {
        let notes: Vec<&Symbol> = symbols
            .iter()
            .filter(|s| s.system == sys.id && s.kind == "note")
            .collect();
        if notes.is_empty() {
            continue;
        }
        doc.push_str(&format!("    <measure number=\"{num}\">\n"));
        num += 1;
        for s in notes {
            let _ = write!(
                doc,
                "      <note><pitch><step>{}</step><octave>{}</octave></pitch>\
                 <duration>1</duration><type>quarter</type></note>\n",
                s.step, s.octave
            );
        }
        doc.push_str("    </measure>\n");
    }
    doc.push_str("  </part>\n</score-partwise>\n");
    doc
}

fn run_serialize(cfg: &Config, page: &Page) -> Result<PathBuf, String> {
    let systems = read_systems_for_serialize(cfg, page)?;
    // Re-derive the symbol stream (same key as the symbols stage wrote).
    let mut rng = SplitMix64(stream_key(cfg.seed, "symbols", &page.name));
    let mut symbols = Vec::new();
    for sys in &systems {
        for k in 0..cfg.symbols.symbols_per_system {
            let kind = KINDS[rng.below(KINDS.len() as u64) as usize].to_string();
            let step = STEPS[rng.below(STEPS.len() as u64) as usize].to_string();
            let octave = (4 + rng.below(2)) as u8;
            let span = sys.w.max(1) as u64;
            let x = sys.x + (rng.below(span + 1)) as u32;
            let confidence = 0.5 + rng.unit() / 2.0;
            if confidence < cfg.symbols.min_confidence {
                continue;
            }
            symbols.push(Symbol {
                id: format!("{}-s{:02}-{:03}", page.name, sys.id, k),
                system: sys.id,
                kind,
                step,
                octave,
                x,
                confidence,
            });
        }
    }
    let (ext, doc) = match cfg.serialize.format.as_str() {
        "musicxml" => (
            "musicxml",
            serialize_musicxml(page, &systems, &symbols),
        ),
        _ => ("json", serialize_json(page, cfg.seed, &systems, &symbols)),
    };
    let path = cfg
        .out_dir
        .join("score")
        .join(format!("{}.{}", page.name, ext));
    write_file(&path, doc.as_bytes())?;
    Ok(path)
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn active_stages(cfg: &Config, args: &Args) -> Vec<&'static str> {
    let mut stages: Vec<&'static str> = STAGES.to_vec();
    if let Some(only) = &args.only {
        stages.retain(|s| s == only);
    } else if let Some(from) = &args.from {
        let pos = STAGES.iter().position(|s| s == from).unwrap_or(0);
        stages.drain(..pos);
    }
    stages.retain(|s| stage_enabled(cfg, s));
    stages
}

fn write_manifest(cfg: &Config, pages: &[Page], stages: &[&str]) -> Result<(), String> {
    let mut doc = format!(
        "{{\n  \"seed\": {},\n  \"stages\": [",
        cfg.seed
    );
    for (i, s) in stages.iter().enumerate() {
        let _ = write!(
            doc,
            "{}\"{}\"",
            if i > 0 { ", " } else { "" },
            esc(s)
        );
    }
    doc.push_str("],\n  \"pages\": [\n");
    for (i, p) in pages.iter().enumerate() {
        let _ = write!(
            doc,
            "    {{\"page\": \"{}\"}}{}\n",
            esc(&p.name),
            if i + 1 < pages.len() { "," } else { "" },
        );
    }
    doc.push_str("  ]\n}\n");
    write_file(&cfg.out_dir.join("manifest.json"), doc.as_bytes())
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().collect();
    let args = parse_args(&argv)?;
    let config_path = args
        .config
        .clone()
        .unwrap_or_else(|| "config.toml".to_string());
    let cfg = load_config(Path::new(&config_path), args.seed)?;
    let stages = active_stages(&cfg, &args);

    if stages.is_empty() {
        return Err("no stages selected (all disabled or sliced out?)".to_string());
    }
    // Stage slices that skip ingest recover the page list from a prior run.
    let with_ingest = stages.contains(&"ingest");
    let pages = if with_ingest {
        discover_pages(&cfg)?
    } else {
        recover_pages(&cfg)?
    };

    if args.dry_run {
        println!(
            "plan: {} page(s) x {} stage(s) (seed {}, out {})",
            pages.len(),
            stages.len(),
            cfg.seed,
            cfg.out_dir.display()
        );
        for p in &pages {
            for s in &stages {
                println!("  {s} {}", p.name);
            }
        }
        return Ok(());
    }

    let state = load_state(&cfg.state_file);
    let mut done: usize = 0;
    let mut skipped: usize = 0;

    for stage in &stages {
        for page in &pages {
            let key = (stage.to_string(), page.name.clone());
            if args.resume && state.contains(&key) {
                skipped += 1;
                continue;
            }
            match *stage {
                "ingest" => {
                    run_ingest(&cfg, page)?;
                }
                "layout" => {
                    run_layout(&cfg, page)?;
                }
                "symbols" => {
                    // Re-derive the layout boxes (same stream the layout
                    // stage wrote) so symbols never parse JSON.
                    let layout_path =
                        cfg.out_dir.join("layout").join(format!("{}.json", page.name));
                    if !layout_path.exists() {
                        return Err(format!(
                            "symbols: missing layout output for `{}` (run layout first)",
                            page.name
                        ));
                    }
                    run_symbols(&cfg, page, &derive_systems(&cfg, page))?;
                }
                "serialize" => {
                    run_serialize(&cfg, page)?;
                }
                _ => unreachable!(),
            }
            append_state(&cfg.state_file, stage, &page.name)?;
            done += 1;
            if let Some(limit) = args.fail_after {
                if done >= limit {
                    eprintln!("--fail-after {limit} reached ({done} unit(s) done)");
                    std::process::exit(2);
                }
            }
        }
    }

    write_manifest(&cfg, &pages, &stages)?;
    println!(
        "done: {done} unit(s), {skipped} skipped, {} page(s) -> {}",
        pages.len(),
        cfg.out_dir.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
