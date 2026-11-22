//! pipeline-orchestrator: deterministic, resumable `render -> vary -> degrade -> pack` driver.
//!
//! Reads a TOML config, derives every output byte from `(seed, stage, unit key)`
//! so reruns are bit-identical, and records each finished work unit in a
//! state file so an interrupted run can continue with `--resume`.
//!
//! Zero third-party dependencies (std only) so farm workers build offline.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use image::codecs::jpeg::JpegEncoder;
use image::{GrayImage, Luma};

const STAGES: [&str; 4] = ["render", "vary", "degrade", "pack"];
const STATE_VERSION: &str = "v1";

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
         Deterministic, resumable render -> vary -> degrade -> pack driver.\n\
         \n\
         USAGE:\n  \
         pipeline-orchestrator [OPTIONS]\n\
         \n\
         OPTIONS:\n  \
         -c, --config <PATH>   Config file (default: config.toml next to CWD)\n      \
         --resume            Skip work units already recorded in the state file\n      \
         --from <STAGE>      Start at STAGE, skipping earlier stages\n      \
         --only <STAGE>      Run only STAGE (inputs must exist from a prior run)\n      \
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
    count: usize,
    out_dir: PathBuf,
    state_file: PathBuf,
    render: RenderCfg,
    vary: VaryCfg,
    degrade: DegradeCfg,
    pack: PackCfg,
}

#[derive(Debug, Clone)]
struct RenderCfg {
    enabled: bool,
    backend: String,
    width: u32,
    height: u32,
}
#[derive(Debug, Clone)]
struct VaryCfg {
    enabled: bool,
    variants_per_sample: usize,
}
#[derive(Debug, Clone)]
struct DegradeCfg {
    enabled: bool,
    level: f64,
    degrades_per_variant: usize,
}
#[derive(Debug, Clone)]
struct PackCfg {
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
    let count = get_i64(&m, "global", "count")?.unwrap_or(16);
    if !(1..=100_000).contains(&count) {
        return Err("[global].count must be in 1..=100000".to_string());
    }
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

    let be = |s: &str| -> Result<bool, String> { Ok(get_bool(&m, s, "enabled")?.unwrap_or(true)) };
    let variants = get_i64(&m, "vary", "variants_per_sample")?.unwrap_or(3);
    if !(1..=64).contains(&variants) {
        return Err("[vary].variants_per_sample must be in 1..=64".to_string());
    }
    let level = get_num(&m, "degrade", "level")?.unwrap_or(0.4);
    if !(0.0..=1.0).contains(&level) {
        return Err("[degrade].level must be in 0.0..=1.0".to_string());
    }
    let degrades = get_i64(&m, "degrade", "degrades_per_variant")?.unwrap_or(1);
    if !(1..=64).contains(&degrades) {
        return Err("[degrade].degrades_per_variant must be in 1..=64".to_string());
    }

    Ok(Config {
        seed,
        count: count as usize,
        out_dir,
        state_file,
        render: RenderCfg {
            enabled: be("render")?,
            backend: get_str(&m, "render", "backend")?.unwrap_or_else(|| "placeholder".to_string()),
            width: get_i64(&m, "render", "width")?.unwrap_or(2480).max(1) as u32,
            height: get_i64(&m, "render", "height")?.unwrap_or(3508).max(1) as u32,
        },
        vary: VaryCfg {
            enabled: be("vary")?,
            variants_per_sample: variants as usize,
        },
        degrade: DegradeCfg {
            enabled: be("degrade")?,
            level,
            degrades_per_variant: degrades as usize,
        },
        pack: PackCfg {
            enabled: be("pack")?,
            format: get_str(&m, "pack", "format")?.unwrap_or_else(|| "manifest".to_string()),
        },
    })
}

// ---------------------------------------------------------------------------
// Deterministic RNG + hashing (std only)
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit over bytes.
fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn fnv1a_hex(data: &[u8]) -> String {
    format!("{:016x}", fnv1a(data))
}

/// SplitMix64: cheap, high-quality, and fully deterministic per stream.
struct StreamRng(u64);

impl StreamRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next_u64() % (hi - lo + 1)
    }
    fn next_unit(&mut self) -> f64 {
        // 53-bit precision double in [0, 1).
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// One independent deterministic stream per work unit: reseeding from
/// (master seed, stage, key) means resume/parallelism never change output.
fn stream_for(master: u64, stage: &str, key: &str) -> StreamRng {
    let mut tag = Vec::with_capacity(64);
    tag.extend_from_slice(stage.as_bytes());
    tag.push(0);
    tag.extend_from_slice(key.as_bytes());
    StreamRng::new(master ^ fnv1a(&tag).wrapping_mul(0x9E3779B97F4A7C15))
}

// ---------------------------------------------------------------------------
// Resume state
// ---------------------------------------------------------------------------

struct State {
    path: PathBuf,
    done: HashSet<String>, // "stage\tkey"
}

impl State {
    fn load(path: &Path) -> Self {
        let mut done = HashSet::new();
        if let Ok(text) = fs::read_to_string(path) {
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if line == STATE_VERSION || line.starts_with("seed=") {
                    continue;
                }
                done.insert(line.to_string());
            }
        }
        Self {
            path: path.to_path_buf(),
            done,
        }
    }

    fn fresh(path: &Path, seed: u64) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        let header = format!("{STATE_VERSION}\nseed={seed}\n");
        fs::write(path, header).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            done: HashSet::new(),
        })
    }

    fn contains(&self, stage: &str, key: &str) -> bool {
        self.done.contains(&format!("{stage}\t{key}"))
    }

    /// Append one completion record (append-only => crash-safe, resumable).
    fn mark_done(&mut self, stage: &str, key: &str) -> Result<(), String> {
        use std::io::Write as _;
        let record = format!("{stage}\t{key}\n");
        let mut f = fs::OpenOptions::new()
            .append(true)
            .open(&self.path)
            .map_err(|e| format!("cannot append to {}: {e}", self.path.display()))?;
        f.write_all(record.as_bytes())
            .map_err(|e| format!("cannot append to {}: {e}", self.path.display()))?;
        self.done.insert(format!("{stage}\t{key}"));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Work plan
// ---------------------------------------------------------------------------

struct Unit {
    stage: &'static str,
    key: String,
}

/// Degrade unit keys for one vary output.
///
/// With the default `degrades_per_variant = 1` the key keeps the legacy
/// `r..v..d` shape so existing configs reproduce bit-identical outputs.
/// With a fan-out of N > 1 each vary output fans out to N seeded variants
/// `r..v..d0 .. r..v..d{N-1}`; every key gets its own `(seed, stage, key)`
/// stream (see `stream_for`), so variants are mutually independent and
/// order-independent.
fn degrade_keys_for(vary_key: &str, degrades_per_variant: usize) -> Vec<String> {
    if degrades_per_variant <= 1 {
        vec![format!("{vary_key}d")]
    } else {
        (0..degrades_per_variant)
            .map(|k| format!("{vary_key}d{k}"))
            .collect()
    }
}

/// Streaming work plan: sample-major order (render sample 0, its varies,
/// its degrades, then sample 1, ...) instead of whole-stage serial order.
/// The first degrade unit therefore runs long before the last render, so
/// render and degrade overlap in wall-clock pipeline terms; `--from` /
/// `--only` still filter by stage exactly as before.
fn build_plan(cfg: &Config, args: &Args) -> Vec<Unit> {
    let from_idx = args
        .from
        .as_deref()
        .map(|s| STAGES.iter().position(|x| *x == s).unwrap_or(0));
    let mut units: Vec<Unit> = Vec::new();
    let mut push = |stage: &'static str, cond: bool, key: String| {
        if !cond {
            return;
        }
        if let Some(only) = &args.only {
            if only != stage {
                return;
            }
        }
        if let Some(fi) = from_idx {
            let si = STAGES.iter().position(|x| *x == stage).unwrap_or(0);
            if si < fi {
                return;
            }
        }
        units.push(Unit { stage, key });
    };
    for i in 0..cfg.count {
        let r = format!("r{i:04}");
        push("render", cfg.render.enabled, r.clone());
        for j in 0..cfg.vary.variants_per_sample {
            let v = format!("{r}v{j}");
            push("vary", cfg.vary.enabled, v.clone());
            if cfg.degrade.enabled {
                for d in degrade_keys_for(&v, cfg.degrade.degrades_per_variant) {
                    push("degrade", true, d);
                }
            }
        }
    }
    push("pack", cfg.pack.enabled, "pack".to_string());
    units
}

// ---------------------------------------------------------------------------
// Stage runners. Each writes deterministic placeholder artifacts; swap the
// body of a runner for a real backend (LilyPond CLI, augmentation script,
// ImageMagick, tar, ...) while keeping the seed->key contract and the
// state-file discipline so determinism and resume keep working.
// ---------------------------------------------------------------------------

fn stage_dir(cfg: &Config, stage: &str) -> PathBuf {
    cfg.out_dir.join(stage)
}

fn write_artifact(path: &Path, body: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    // Write-then-rename so a killed run never leaves a half-written file
    // that a later --resume would mistake for complete.
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, body).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|e| format!("cannot rename to {}: {e}", path.display()))?;
    Ok(())
}

#[derive(Debug, Clone)]
struct Note {
    pitch: i64,
    dur: u64,
    x: u32,
    y: u32,
}

/// Draw staff systems + note blobs into a grayscale page. Deterministic in
/// (notes, systems, w, h); all randomness must enter via `notes`.
fn draw_score(notes: &[Note], systems: u64, w: u32, h: u32) -> GrayImage {
    let mut img = GrayImage::from_pixel(w, h, Luma([255u8]));
    let put = |img: &mut GrayImage, x: i64, y: i64, v: u8| {
        if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
            img.put_pixel(x as u32, y as u32, Luma([v]));
        }
    };
    let margin = (w / 20).max(8) as i64;
    let top = (h / 20).max(8) as i64;
    let systems = systems.max(1) as i64;
    let sys_h = ((h as i64 - 2 * top) / systems).max(20);
    let gap = (sys_h / 12).max(3);
    let thick = ((h / 2000).max(1)) as i64;
    // Staff lines.
    for s in 0..systems {
        let base = top + s * sys_h + sys_h / 3;
        for l in 0..5 {
            let y = base + l * gap;
            for t in 0..thick {
                let yy = y + t;
                if yy >= h as i64 {
                    break;
                }
                for x in margin..(w as i64 - margin) {
                    img.put_pixel(x as u32, yy as u32, Luma([0u8]));
                }
            }
        }
    }
    // Notes: filled head + stem.
    let r = ((w / 300).max(2)) as i64;
    let stem_h = (sys_h / 4).max(8);
    let stem_w = ((w / 1200).max(1)) as i64;
    for n in notes {
        let cx = (n.x as i64).clamp(margin, w as i64 - margin - 1);
        let cy = (n.y as i64).clamp(top, h as i64 - top - 1);
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    put(&mut img, cx + dx, cy + dy, 0);
                }
            }
        }
        for i in 0..stem_h {
            for t in 0..stem_w {
                put(&mut img, cx + r + t, cy - i, 0);
            }
        }
    }
    img
}

fn encode_jpeg(img: &GrayImage, quality: u8) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut buf, quality);
    enc.encode_image(img)
        .map_err(|e| format!("jpeg encode failed: {e}"))?;
    Ok(buf)
}

fn render_notes_to_jpeg(notes: &[Note], systems: u64, w: u32, h: u32, quality: u8) -> Result<Vec<u8>, String> {
    let img = draw_score(notes, systems, w, h);
    encode_jpeg(&img, quality)
}

fn write_bytes_artifact(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|e| format!("cannot rename to {}: {e}", path.display()))?;
    Ok(())
}

/// Parse the `# ... systems=N` header our own writers emit.
fn parse_systems(header: &str) -> Result<u64, String> {
    for tok in header.split_whitespace() {
        if let Some(n) = tok.strip_prefix("systems=") {
            return n.parse::<u64>().map_err(|_| format!("bad systems count in `{header}`"));
        }
    }
    Err(format!("no systems= in header `{header}`"))
}

fn parse_notes(body: &str) -> Vec<Note> {
    let mut out = Vec::new();
    for line in body.lines().skip(1) {
        let mut parts = line.split_whitespace();
        if parts.next() != Some("n") {
            continue;
        }
        let pitch = parts.next().unwrap_or("60").parse().unwrap_or(60);
        let dur = parts.next().unwrap_or("4").parse().unwrap_or(4);
        let x = parts.next().unwrap_or("0").parse().unwrap_or(0);
        let y = parts.next().unwrap_or("0").parse().unwrap_or(0);
        out.push(Note { pitch, dur, x, y });
    }
    out
}

fn run_render(cfg: &Config, key: &str) -> Result<PathBuf, String> {
    let out_jpg = stage_dir(cfg, "render").join(format!("{key}.jpg"));
    let out_page = stage_dir(cfg, "render").join(format!("{key}.page"));
    let mut rng = stream_for(cfg.seed, "render", key);
    let n = rng.next_range(40, 400);
    let systems = rng.next_range(2, 8);
    let w = cfg.render.width;
    let h = cfg.render.height;
    let mut notes = Vec::with_capacity(n as usize);
    let mut body = String::new();
    let _ = writeln!(
        body,
        "# render {key} seed={} backend={} {}x{} notes={n} systems={systems}",
        cfg.seed, cfg.render.backend, w, h
    );
    // Deterministic pseudo-engraving payload (also the JPEG ground truth).
    for _ in 0..n {
        let pitch = rng.next_range(21, 108) as i64;
        let dur = [1, 2, 4, 8, 16][rng.next_range(0, 4) as usize];
        let x = rng.next_range(0, w as u64) as u32;
        let y = rng.next_range(0, h as u64) as u32;
        let _ = writeln!(body, "n {pitch} {dur} {x} {y}");
        notes.push(Note { pitch, dur, x, y });
    }
    let jpeg = render_notes_to_jpeg(&notes, systems, w, h, 85)?;
    write_bytes_artifact(&out_jpg, &jpeg)?;
    write_artifact(&out_page, &body)?;
    Ok(out_jpg)
}

fn parent_sample(key: &str) -> &str {
    // "r0003v1" -> "r0003"; "r0003v1d" handled by caller via trim.
    match key.find('v') {
        Some(i) => &key[..i],
        None => key,
    }
}

fn run_vary(cfg: &Config, key: &str) -> Result<PathBuf, String> {
    let parent = stage_dir(cfg, "render").join(format!("{}.page", parent_sample(key)));
    let parent_body = fs::read_to_string(&parent).map_err(|e| {
        format!(
            "vary {key}: missing render input {}: {e} (run render first)",
            parent.display()
        )
    })?;
    let out_jpg = stage_dir(cfg, "vary").join(format!("{key}.jpg"));
    let out_page = stage_dir(cfg, "vary").join(format!("{key}.page"));
    let mut rng = stream_for(cfg.seed, "vary", key);
    let transpose = rng.next_range(0, 2) as i64 - 1; // -1, 0, +1
    let spacing = 0.9 + rng.next_unit() * 0.3;
    let w = cfg.render.width;
    let h = cfg.render.height;
    let systems = parse_systems(parent_body.lines().next().unwrap_or(""))?;
    let cx = w as f64 / 2.0;
    let mut body = String::new();
    let _ = writeln!(
        body,
        "# vary {key} from={} transpose={transpose} spacing={spacing:.3} systems={systems}",
        parent_sample(key)
    );
    let mut notes = Vec::new();
    for n in parse_notes(&parent_body) {
        let pitch = (n.pitch + transpose).clamp(21, 108);
        let x = (cx + (n.x as f64 - cx) * spacing).round().clamp(0.0, w as f64 - 1.0) as u32;
        let _ = writeln!(body, "n {pitch} {} {x} {}", n.dur, n.y);
        notes.push(Note { pitch, dur: n.dur, x, y: n.y });
    }
    let jpeg = render_notes_to_jpeg(&notes, systems, w, h, 85)?;
    write_bytes_artifact(&out_jpg, &jpeg)?;
    write_artifact(&out_page, &body)?;
    Ok(out_jpg)
}

/// Pixel-level phone-photo degradation: sensor noise, ink blots, a vertical
/// brightness gradient plus one glare band, then a lossy JPEG re-encode.
/// Deterministic in (input bytes, master seed, key, level).
fn degrade_jpeg_bytes(input: &[u8], master_seed: u64, key: &str, level: f64) -> Result<Vec<u8>, String> {
    let img = image::load_from_memory(input)
        .map_err(|e| format!("degrade {key}: cannot decode vary JPEG: {e}"))?
        .to_luma8();
    let (w, h) = (img.width(), img.height());
    let mut out = img;
    let mut rng = stream_for(master_seed, "degrade", key);
    // Sensor noise.
    let noise_p = level * 0.15;
    let noise_amp = (level * 60.0) as i16;
    for px in out.pixels_mut() {
        if rng.next_unit() < noise_p {
            let j = (rng.next_unit() * 2.0 - 1.0) * noise_amp as f64;
            let v = (px.0[0] as f64 + j).round().clamp(0.0, 255.0) as u8;
            px.0[0] = v;
        }
    }
    // Ink blots + a few dropped-symbol smudges (dark rects).
    let blots = (level * 30.0) as u64;
    for _ in 0..blots {
        let bw = rng.next_range(2, (w / 20).max(4) as u64) as u32;
        let bh = rng.next_range(2, (h / 40).max(4) as u64) as u32;
        let bx = rng.next_range(0, w.max(1) as u64) as u32;
        let by = rng.next_range(0, h.max(1) as u64) as u32;
        for dy in 0..bh {
            for dx in 0..bw {
                let (x, y) = (bx + dx, by + dy);
                if x < w && y < h {
                    out.put_pixel(x, y, Luma([20u8]));
                }
            }
        }
    }
    // Lighting: vertical falloff plus one horizontal glare band.
    let glare_y = rng.next_range(0, h.max(1) as u64) as f64;
    let glare_h = (h as f64 * 0.08).max(4.0);
    let glare_amp = level * 50.0;
    for y in 0..h {
        let shade = 0.85 + 0.3 * (y as f64 / h.max(1) as f64);
        let glare = glare_amp * (1.0 - ((y as f64 - glare_y).abs() / glare_h).min(1.0));
        for x in 0..w {
            let v = out.get_pixel(x, y).0[0] as f64 * shade + glare;
            out.put_pixel(x, y, Luma([v.round().clamp(0.0, 255.0) as u8]));
        }
    }
    encode_jpeg(&out, 72)
}

/// Vary parent of a degrade key: `r0003v1d` -> `r0003v1` (legacy single
/// variant) and `r0003v1d7` -> `r0003v1` (fan-out). The `r..v..` prefix
/// never contains `d`, so stripping from the last `d` is exact.
fn vary_parent_of(degrade_key: &str) -> &str {
    match degrade_key.rfind('d') {
        Some(i) => &degrade_key[..i],
        None => degrade_key,
    }
}

fn run_degrade(cfg: &Config, key: &str) -> Result<PathBuf, String> {
    let vkey = vary_parent_of(key);
    let parent_jpg = stage_dir(cfg, "vary").join(format!("{vkey}.jpg"));
    let parent_page = stage_dir(cfg, "vary").join(format!("{vkey}.page"));
    let parent_body = fs::read_to_string(&parent_page).map_err(|e| {
        format!(
            "degrade {key}: missing vary input {}: {e} (run vary first)",
            parent_page.display()
        )
    })?;
    let out_jpg = stage_dir(cfg, "degrade").join(format!("{key}.jpg"));
    let out_page = stage_dir(cfg, "degrade").join(format!("{key}.page"));
    let density = cfg.degrade.level;
    let input = fs::read(&parent_jpg).map_err(|e| {
        format!(
            "degrade {key}: missing vary JPEG {}: {e} (run vary first)",
            parent_jpg.display()
        )
    })?;
    let jpeg = degrade_jpeg_bytes(&input, cfg.seed, key, density)?;
    let mut body = String::new();
    let _ = writeln!(body, "# degrade {key} from={vkey} level={density}");
    for line in parent_body.lines().skip(1) {
        let _ = writeln!(body, "{line}");
    }
    write_bytes_artifact(&out_jpg, &jpeg)?;
    write_artifact(&out_page, &body)?;
    Ok(out_jpg)
}

fn collect_artifacts(dir: &Path, ext: &str, acc: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("cannot list {}: {e}", dir.display()))?;
    let mut names: Vec<PathBuf> = Vec::new();
    for e in entries {
        let e = e.map_err(|e| format!("cannot list {}: {e}", dir.display()))?;
        names.push(e.path());
    }
    names.sort();
    for p in names {
        if p.is_dir() {
            collect_artifacts(&p, ext, acc)?;
        } else if p.extension().and_then(|e| e.to_str()) == Some(ext) {
            acc.push(p);
        }
    }
    Ok(())
}

fn run_pack(cfg: &Config) -> Result<PathBuf, String> {
    let mut files: Vec<PathBuf> = Vec::new();
    for stage in ["render", "vary", "degrade"] {
        let d = stage_dir(cfg, stage);
        if d.exists() {
            collect_artifacts(&d, "jpg", &mut files)?;
        }
    }
    files.sort();
    let mut manifest = String::new();
    let _ = writeln!(
        manifest,
        "# dataset manifest seed={} format={}",
        cfg.seed, cfg.pack.format
    );
    for f in &files {
        let bytes = fs::read(f).map_err(|e| format!("pack: cannot read {}: {e}", f.display()))?;
        let rel = f.strip_prefix(&cfg.out_dir).unwrap_or(f);
        let _ = writeln!(manifest, "{}  {}", fnv1a_hex(&bytes), rel.display());
    }
    let pack_dir = stage_dir(cfg, "pack");
    write_artifact(&pack_dir.join("manifest.txt"), &manifest)?;
    let mut summary = String::new();
    let _ = writeln!(summary, "seed: {}", cfg.seed);
    let _ = writeln!(summary, "count: {}", cfg.count);
    let _ = writeln!(summary, "artifacts: {}", files.len());
    let _ = writeln!(summary, "manifest_sha: {}", fnv1a_hex(manifest.as_bytes()));
    let summary_path = pack_dir.join("summary.txt");
    write_artifact(&summary_path, &summary)?;
    Ok(summary_path)
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn run() -> Result<i32, String> {
    let argv: Vec<String> = std::env::args().collect();
    let args = parse_args(&argv)?;
    let cfg_path = PathBuf::from(
        args.config
            .clone()
            .unwrap_or_else(|| "config.toml".to_string()),
    );
    let cfg = load_config(&cfg_path, args.seed)?;

    let plan = build_plan(&cfg, &args);
    if plan.is_empty() {
        println!("nothing to do: all selected stages are disabled or filtered out");
        return Ok(0);
    }

    if args.dry_run {
        let mut per_stage: HashMap<&str, usize> = HashMap::new();
        for u in &plan {
            *per_stage.entry(u.stage).or_default() += 1;
        }
        println!("plan for {} (seed {}):", cfg_path.display(), cfg.seed);
        for s in STAGES {
            if let Some(n) = per_stage.get(s) {
                println!("  {s}: {n} unit(s)");
            }
        }
        println!("total: {} unit(s) -> {}", plan.len(), cfg.out_dir.display());
        return Ok(0);
    }

    fs::create_dir_all(&cfg.out_dir)
        .map_err(|e| format!("cannot create {}: {e}", cfg.out_dir.display()))?;

    let mut state = if args.resume {
        let st = State::load(&cfg.state_file);
        // Guard against resuming a different dataset.
        if let Ok(text) = fs::read_to_string(&cfg.state_file) {
            for line in text.lines() {
                if let Some(seed) = line.strip_prefix("seed=") {
                    if seed.trim() != cfg.seed.to_string() {
                        return Err(format!(
                            "state file {} belongs to seed={seed}, but this run uses seed={} \
                             (drop --resume or point [global].state_file elsewhere)",
                            cfg.state_file.display(),
                            cfg.seed
                        ));
                    }
                }
            }
        }
        st
    } else {
        State::fresh(&cfg.state_file, cfg.seed)?
    };

    let stats = execute_plan(&cfg, &plan, &mut state, args.fail_after)?;
    let degrade_pages = plan.iter().filter(|u| u.stage == "degrade").count();
    println!(
        "done: {} unit(s) ran, {} skipped (already in state), seed={}, out={}",
        stats.done,
        stats.skipped,
        cfg.seed,
        cfg.out_dir.display()
    );
    println!(
        "timing ms: render={} vary={} degrade={} pack={} ({} degrade pages)",
        stats.render_ms.as_millis(),
        stats.vary_ms.as_millis(),
        stats.degrade_ms.as_millis(),
        stats.pack_ms.as_millis(),
        degrade_pages,
    );
    Ok(stats.code)
}

/// Per-stage wall-clock totals for one `execute_plan` run. Measurement only;
/// never mixed into artifact bytes, so determinism is unaffected.
#[derive(Debug, Default)]
struct PlanStats {
    done: usize,
    skipped: usize,
    code: i32,
    render_ms: std::time::Duration,
    vary_ms: std::time::Duration,
    degrade_ms: std::time::Duration,
    pack_ms: std::time::Duration,
}

fn execute_plan(
    cfg: &Config,
    plan: &[Unit],
    state: &mut State,
    fail_after: Option<usize>,
) -> Result<PlanStats, String> {
    let mut stats = PlanStats::default();
    let mut finished_this_run = 0usize;
    for u in plan {
        if state.contains(u.stage, &u.key) {
            stats.skipped += 1;
            continue;
        }
        let t0 = std::time::Instant::now();
        let res = match u.stage {
            "render" => run_render(cfg, &u.key).map(|_| ()),
            "vary" => run_vary(cfg, &u.key).map(|_| ()),
            "degrade" => run_degrade(cfg, &u.key).map(|_| ()),
            "pack" => run_pack(cfg).map(|_| ()),
            s => return Err(format!("unknown stage: {s}")),
        };
        let dt = t0.elapsed();
        if let Err(e) = res {
            eprintln!("FAILED {} {}: {e}", u.stage, u.key);
            eprintln!(
                "state saved in {}; rerun with --resume to continue",
                cfg.state_file.display()
            );
            return Err(e);
        }
        match u.stage {
            "render" => stats.render_ms += dt,
            "vary" => stats.vary_ms += dt,
            "degrade" => stats.degrade_ms += dt,
            "pack" => stats.pack_ms += dt,
            _ => {}
        }
        state.mark_done(u.stage, &u.key)?;
        stats.done += 1;
        finished_this_run += 1;
        if stats.done.is_multiple_of(50) {
            println!("  ... {} units done", stats.done);
        }
        if let Some(limit) = fail_after {
            if finished_this_run >= limit {
                eprintln!(
                    "--fail-after {limit} reached after {} {}; rerun with --resume",
                    u.stage, u.key
                );
                stats.code = 2;
                return Ok(stats);
            }
        }
    }
    Ok(stats)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_notes() -> Vec<Note> {
        vec![
            Note { pitch: 60, dur: 4, x: 80, y: 100 },
            Note { pitch: 64, dur: 8, x: 120, y: 110 },
            Note { pitch: 67, dur: 2, x: 200, y: 90 },
        ]
    }

    #[test]
    fn render_emits_decodable_jpeg_at_configured_dims() {
        let bytes = render_notes_to_jpeg(&sample_notes(), 2, 320, 400, 85).unwrap();
        assert_eq!(&bytes[0..3], &[0xFF, 0xD8, 0xFF], "missing JPEG magic");
        let img = image::load_from_memory(&bytes).unwrap().to_luma8();
        assert_eq!((img.width(), img.height()), (320, 400));
        // Staff lines + note blobs must leave dark pixels on the white page.
        assert!(img.pixels().any(|p| p.0[0] < 128), "page is blank");
        assert!(img.pixels().any(|p| p.0[0] > 200), "page has no background");
    }

    #[test]
    fn render_is_deterministic() {
        let a = render_notes_to_jpeg(&sample_notes(), 2, 320, 400, 85).unwrap();
        let b = render_notes_to_jpeg(&sample_notes(), 2, 320, 400, 85).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn degrade_keeps_dims_and_changes_pixels() {
        let base = render_notes_to_jpeg(&sample_notes(), 2, 320, 400, 85).unwrap();
        let deg = degrade_jpeg_bytes(&base, 1234, "r0000v0d", 0.5).unwrap();
        assert_eq!(&deg[0..3], &[0xFF, 0xD8, 0xFF], "missing JPEG magic");
        let (bi, di) = (
            image::load_from_memory(&base).unwrap().to_luma8(),
            image::load_from_memory(&deg).unwrap().to_luma8(),
        );
        assert_eq!((di.width(), di.height()), (bi.width(), bi.height()));
        let diff = bi
            .pixels()
            .zip(di.pixels())
            .filter(|(a, b)| a.0[0] != b.0[0])
            .count();
        assert!(diff > 0, "degrade left pixels untouched");
    }

    fn test_config(out_dir: &std::path::Path, degrades_per_variant: usize) -> Config {
        Config {
            seed: 20261022,
            count: 2,
            out_dir: out_dir.to_path_buf(),
            state_file: out_dir.join(".pipeline-state"),
            render: RenderCfg {
                enabled: true,
                backend: "placeholder".to_string(),
                width: 160,
                height: 200,
            },
            vary: VaryCfg {
                enabled: true,
                variants_per_sample: 2,
            },
            degrade: DegradeCfg {
                enabled: true,
                level: 0.4,
                degrades_per_variant,
            },
            pack: PackCfg {
                enabled: true,
                format: "manifest".to_string(),
            },
        }
    }

    fn test_args() -> Args {
        Args::default()
    }

    fn plan_keys(cfg: &Config, stage: &str) -> Vec<String> {
        build_plan(cfg, &test_args())
            .into_iter()
            .filter(|u| u.stage == stage)
            .map(|u| u.key)
            .collect()
    }

    #[test]
    fn fanout_default_keeps_legacy_degrade_keys() {
        let dir = std::env::temp_dir().join("orch-legacy-keys");
        let cfg = test_config(&dir, 1);
        assert_eq!(
            plan_keys(&cfg, "degrade"),
            vec!["r0000v0d", "r0000v1d", "r0001v0d", "r0001v1d"]
        );
    }

    #[test]
    fn fanout_keys_are_distinct_with_independent_streams() {
        let dir = std::env::temp_dir().join("orch-fanout-keys");
        let cfg = test_config(&dir, 3);
        let keys = plan_keys(&cfg, "degrade");
        assert_eq!(keys.len(), 2 * 2 * 3);
        assert_eq!(&keys[0..3], &["r0000v0d0", "r0000v0d1", "r0000v0d2"]);
        let mut seen = std::collections::HashSet::new();
        for k in &keys {
            assert!(seen.insert(k.clone()), "duplicate degrade key {k}");
        }
        // Every unit's stream derives from (seed, stage, key) alone: distinct
        // keys must yield distinct streams (order-independent by construction).
        let mut heads = std::collections::HashSet::new();
        for k in &keys {
            let h = stream_for(cfg.seed, "degrade", k).next_u64();
            assert!(heads.insert(h), "stream collision for key {k}");
        }
        assert_eq!(vary_parent_of("r0003v1d"), "r0003v1");
        assert_eq!(vary_parent_of("r0003v1d7"), "r0003v1");
    }

    #[test]
    fn plan_streams_sample_major_render_overlaps_degrade() {
        let dir = std::env::temp_dir().join("orch-stream-order");
        let cfg = test_config(&dir, 2);
        let plan = build_plan(&cfg, &test_args());
        let pos = |stage: &str, key: &str| {
            plan.iter()
                .position(|u| u.stage == stage && u.key == key)
                .unwrap()
        };
        // Sample 0 fully pipelines before sample 1's render starts ...
        assert!(pos("degrade", "r0000v1d1") < pos("render", "r0001"));
        // ... so the first degrade no longer waits for the last render.
        let first_degrade = plan.iter().position(|u| u.stage == "degrade").unwrap();
        let last_render = plan.iter().rposition(|u| u.stage == "render").unwrap();
        assert!(first_degrade < last_render);
        // Pack still runs last.
        assert_eq!(plan.last().unwrap().stage, "pack");
        // Unit total: count * (1 + V + V*D) + 1.
        assert_eq!(plan.len(), 2 * (1 + 2 + 2 * 2) + 1);
    }

    #[test]
    fn end_to_end_fanout_is_deterministic_and_resumable() {
        let tag = format!("orch-e2e-{}", std::process::id());
        let out_a = std::env::temp_dir().join(format!("{tag}-a"));
        let out_b = std::env::temp_dir().join(format!("{tag}-b"));
        let _ = std::fs::remove_dir_all(&out_a);
        let _ = std::fs::remove_dir_all(&out_b);

        let run_once = |out: &std::path::Path| {
            let cfg = test_config(out, 3);
            let plan = build_plan(&cfg, &test_args());
            let mut state = State::fresh(&cfg.state_file, cfg.seed).unwrap();
            let stats = execute_plan(&cfg, &plan, &mut state, None).unwrap();
            assert_eq!(stats.code, 0);
            assert_eq!(stats.done, plan.len());
            assert_eq!(stats.skipped, 0);
            // Resume with the recorded state: everything skipped, nothing rerun.
            let plan2 = build_plan(&cfg, &test_args());
            let mut state2 = State::load(&cfg.state_file);
            let stats2 = execute_plan(&cfg, &plan2, &mut state2, None).unwrap();
            assert_eq!(stats2.done, 0);
            assert_eq!(stats2.skipped, plan2.len());
        };
        run_once(&out_a);
        run_once(&out_b);

        // Same seed + inputs => bit-identical dataset in both out dirs.
        let (ma, mb) = (
            std::fs::read(out_a.join("pack/manifest.txt")).unwrap(),
            std::fs::read(out_b.join("pack/manifest.txt")).unwrap(),
        );
        assert_eq!(ma, mb);
        let (ja, jb) = (
            std::fs::read(out_a.join("degrade/r0000v0d0.jpg")).unwrap(),
            std::fs::read(out_b.join("degrade/r0000v0d0.jpg")).unwrap(),
        );
        assert_eq!(ja, jb);

        // Fan-out variants of one vary parent are all present and differ.
        let mut variants = std::collections::HashSet::new();
        for k in 0..3 {
            let b = std::fs::read(out_a.join(format!("degrade/r0001v1d{k}.jpg"))).unwrap();
            variants.insert(fnv1a_hex(&b));
        }
        assert_eq!(variants.len(), 3, "fan-out variants must differ");

        let _ = std::fs::remove_dir_all(&out_a);
        let _ = std::fs::remove_dir_all(&out_b);
    }

    #[test]
    fn fanout_config_validation_rejects_zero() {
        let text = "[global]\nseed = 1\ncount = 2\n[render]\n[vary]\n[degrade]\ndegrades_per_variant = 0\n[pack]\n";
        let dir = std::env::temp_dir().join("orch-bad-cfg");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bad.toml");
        std::fs::write(&path, text).unwrap();
        assert!(load_config(&path, None).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
