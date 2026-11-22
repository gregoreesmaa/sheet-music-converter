//! Suite aggregation: mean + median + usable-output rate over per-piece
//! reports, with strict-F1 and OMR-NED side by side.
//!
//! Pure aggregation — no scoring logic, so scorer semantics are untouched.
//!
//! Input manifest (written by the run scripts in `suites/`):
//! ```json
//! {
//!   "suite": "clean31", "engine": "ours",
//!   "pieces": [
//!     {"slug": "repeats-melody", "report": "reports/repeats-melody.json",
//!      "omr_ned": 0.9818,
//!      "reference_sha256": "..", "hypothesis_sha256": ".."},
//!     {"slug": "palestrina-agnus", "report": null, "no_output": true}
//!   ]
//! }
//! ```
//!
//! Rules (mirroring the benchmark bundle conventions):
//! * the manifest piece list is the denominator — every expected piece is
//!   counted, output or not;
//! * no-output (null report path, missing file, or `"no_output": true`)
//!   scores strict-F1 0.0 and OMR-NED 1.0 (`missing=1.0`, as in the
//!   bundle's `omr_ned.py`), and counts as unusable;
//! * a present-but-unparseable report file is a hard error, not a silent
//!   zero — the file exists, so something is wrong with the run;
//! * a usable piece with no OMR-NED value is excluded from the NED
//!   aggregates (counted separately as `n_missing`);
//! * medians follow the reference `score.py med()`: the sorted middle
//!   (average of the two middles for even counts), 0.0 when empty.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// One expected piece in a suite manifest.
#[derive(Debug, Clone, Deserialize)]
pub struct ManifestPiece {
    pub slug: String,
    /// Per-piece report JSON (`strict-xml --report` / `strict-f1 --report`).
    /// Null (or a missing file) means no output: scored zero, counted.
    #[serde(default)]
    pub report: Option<PathBuf>,
    #[serde(default)]
    pub no_output: bool,
    /// OMR-NED for this piece (reference musicdiff value). Overrides the
    /// report's own `omr_ned` when both are present; useful when NED is
    /// scored after F1 (musicdiff is slow).
    #[serde(default)]
    pub omr_ned: Option<f64>,
    #[serde(default)]
    pub reference_sha256: Option<String>,
    #[serde(default)]
    pub hypothesis_sha256: Option<String>,
}

/// Suite manifest: the full expected piece list plus provenance.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub suite: String,
    pub engine: String,
    #[serde(default)]
    pub pieces: Vec<ManifestPiece>,
}

/// Mean + both medians for one metric.
#[derive(Debug, Clone, Serialize)]
pub struct MetricSummary {
    pub mean: f64,
    pub median_all: f64,
    pub median_transcribed: f64,
}

/// NED aggregates: same triple, over the pieces that have an NED value
/// (usable pieces with a value, plus no-output pieces at the missing
/// convention). Usable pieces without a value are excluded and counted
/// as `n_missing` instead of poisoning the mean.
#[derive(Debug, Clone, Serialize)]
pub struct NedSummary {
    /// Pieces contributing to the NED aggregates below.
    pub n: usize,
    pub mean: f64,
    pub median_all: f64,
    pub median_transcribed: f64,
    /// No-output pieces score this (bundle `missing=1.0` convention).
    pub missing_convention: f64,
    pub n_no_output: usize,
    pub n_missing: usize,
}

/// Exact-match column over empty-reference pieces (notation-free pages).
#[derive(Debug, Clone, Serialize)]
pub struct EmptySummary {
    /// Pieces whose reference holds no pitched notes.
    pub n: usize,
    /// ... of which the hypothesis was also empty.
    pub n_exact: usize,
    pub rate: f64,
}

/// One aggregated per-piece row.
///
/// `strict_f1` is `None` for empty-reference pieces: they carry
/// `exact_match` instead and are excluded from every F1 aggregate, so no
/// median can silently pool them (in either direction).
#[derive(Debug, Clone, Serialize)]
pub struct PieceRow {
    pub slug: String,
    pub usable: bool,
    pub no_output: bool,
    pub strict_f1: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_match: Option<f64>,
    #[serde(default)]
    pub empty_ref: bool,
    pub omr_ned: Option<f64>,
    pub reference_sha256: Option<String>,
    pub hypothesis_sha256: Option<String>,
    /// Measure-atomic correction gestures from the `fixes-estimate`
    /// subcommand (`None` until a lattice is scored for this piece).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixes_estimate: Option<usize>,
}

/// Full suite summary.
///
/// `strict_f1` aggregates run over non-empty pieces only; empty-reference
/// pieces are counted in `n_empty_ref` / `exact_match` and leave the
/// F1 (and NED) medians untouched.
#[derive(Debug, Clone, Serialize)]
pub struct SuiteSummary {
    pub suite: String,
    pub engine: String,
    pub scorer: String,
    pub n_total: usize,
    pub n_usable: usize,
    pub usable_rate: f64,
    pub strict_f1: MetricSummary,
    pub omr_ned: NedSummary,
    pub n_empty_ref: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exact_match: Option<EmptySummary>,
    pub pieces: Vec<PieceRow>,
}

/// OMR-NED scored for a piece with no usable output.
pub const NED_MISSING: f64 = 1.0;

/// Median following the reference `score.py med()`: sorted middle, average
/// of the two middles for even counts, 0.0 when empty.
fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = values.len() / 2;
    if values.len() % 2 == 1 {
        values[m]
    } else {
        (values[m - 1] + values[m]) / 2.0
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Strict-F1 plus passthrough metadata out of a per-piece report file.
///
/// Three shapes: `strict-xml` (`strict_f1`), `strict-f1` (`f1`), and the
/// empty-reference report (`empty_ref: true` + `exact_match`, no F1
/// field). The empty shape aggregates into the exact-match column only.
#[derive(Debug, Clone)]
struct ReportValues {
    strict_f1: Option<f64>,
    exact_match: Option<f64>,
    empty_ref: bool,
    omr_ned: Option<f64>,
    reference_sha256: Option<String>,
    hypothesis_sha256: Option<String>,
}

fn read_strict_f1(path: &PathBuf) -> Result<ReportValues, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    let ned = v.get("omr_ned").and_then(|x| x.as_f64());
    let ref_sha = v
        .get("reference_sha256")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());
    let hyp_sha = v
        .get("hypothesis_sha256")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());
    if v.get("empty_ref").and_then(|x| x.as_bool()) == Some(true) {
        let exact = v
            .get("exact_match")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| format!("parse {}: empty_ref without exact_match", path.display()))?;
        return Ok(ReportValues {
            strict_f1: None,
            exact_match: Some(exact),
            empty_ref: true,
            omr_ned: ned,
            reference_sha256: ref_sha,
            hypothesis_sha256: hyp_sha,
        });
    }
    let f1 = v
        .get("strict_f1")
        .or_else(|| v.get("f1"))
        .and_then(|x| x.as_f64())
        .ok_or_else(|| format!("parse {}: no strict_f1/f1 field", path.display()))?;
    Ok(ReportValues {
        strict_f1: Some(f1),
        exact_match: None,
        empty_ref: false,
        omr_ned: ned,
        reference_sha256: ref_sha,
        hypothesis_sha256: hyp_sha,
    })
}

/// Aggregate one manifest into a suite summary.
pub fn summarize(manifest: &Manifest) -> Result<SuiteSummary, String> {
    let mut rows: Vec<PieceRow> = Vec::new();
    for p in &manifest.pieces {
        let missing = p.no_output || p.report.is_none();
        if missing {
            rows.push(PieceRow {
                slug: p.slug.clone(),
                usable: false,
                no_output: true,
                strict_f1: Some(0.0),
                exact_match: None,
                empty_ref: false,
                omr_ned: Some(p.omr_ned.unwrap_or(NED_MISSING)),
                reference_sha256: p.reference_sha256.clone(),
                hypothesis_sha256: p.hypothesis_sha256.clone(),
                fixes_estimate: None,
            });
            continue;
        }
        let path = p.report.as_ref().unwrap();
        if !path.exists() {
            rows.push(PieceRow {
                slug: p.slug.clone(),
                usable: false,
                no_output: true,
                strict_f1: Some(0.0),
                exact_match: None,
                empty_ref: false,
                omr_ned: Some(p.omr_ned.unwrap_or(NED_MISSING)),
                reference_sha256: p.reference_sha256.clone(),
                hypothesis_sha256: p.hypothesis_sha256.clone(),
                fixes_estimate: None,
            });
            continue;
        }
        let v = read_strict_f1(path).map_err(|e| format!("{}: {e}", p.slug))?;
        if v.empty_ref {
            // Notation-free reference: output exists (usable), but the
            // piece carries no F1 observation — it aggregates into the
            // exact-match column and leaves every F1/NED median alone.
            // NED is undefined without a reference, so the row holds no
            // NED value (counted as n_missing like any usable piece
            // without one).
            rows.push(PieceRow {
                slug: p.slug.clone(),
                usable: true,
                no_output: false,
                strict_f1: None,
                exact_match: v.exact_match,
                empty_ref: true,
                omr_ned: None,
                reference_sha256: p.reference_sha256.clone().or(v.reference_sha256),
                hypothesis_sha256: p.hypothesis_sha256.clone().or(v.hypothesis_sha256),
                fixes_estimate: None,
            });
            continue;
        }
        rows.push(PieceRow {
            slug: p.slug.clone(),
            usable: true,
            no_output: false,
            strict_f1: v.strict_f1,
            exact_match: None,
            empty_ref: false,
            omr_ned: p.omr_ned.or(v.omr_ned),
            reference_sha256: p.reference_sha256.clone().or(v.reference_sha256),
            hypothesis_sha256: p.hypothesis_sha256.clone().or(v.hypothesis_sha256),
            fixes_estimate: None,
        });
    }

    let n_total = rows.len();
    let n_usable = rows.iter().filter(|r| r.usable).count();
    let usable_rate = if n_total == 0 {
        0.0
    } else {
        n_usable as f64 / n_total as f64
    };

    // F1 aggregates run over non-empty pieces only (no-output pieces
    // still score 0.0; empty-reference pieces hold no F1 observation).
    let all_f1: Vec<f64> = rows.iter().filter_map(|r| r.strict_f1).collect();
    let tx_f1: Vec<f64> = rows
        .iter()
        .filter(|r| r.usable)
        .filter_map(|r| r.strict_f1)
        .collect();
    let strict_f1 = MetricSummary {
        mean: mean(&all_f1),
        median_all: median(all_f1),
        median_transcribed: median(tx_f1),
    };

    let n_empty_ref = rows.iter().filter(|r| r.empty_ref).count();
    let exact_match = if n_empty_ref == 0 {
        None
    } else {
        let n_exact = rows
            .iter()
            .filter(|r| r.empty_ref && r.exact_match.is_some_and(|e| (e - 1.0).abs() < 1e-9))
            .count();
        Some(EmptySummary {
            n: n_empty_ref,
            n_exact,
            rate: n_exact as f64 / n_empty_ref as f64,
        })
    };

    let ned_all: Vec<f64> = rows.iter().filter_map(|r| r.omr_ned).collect();
    let ned_tx: Vec<f64> = rows
        .iter()
        .filter(|r| r.usable)
        .filter_map(|r| r.omr_ned)
        .collect();
    let n_no_output = rows.iter().filter(|r| r.no_output).count();
    let n_missing = rows
        .iter()
        .filter(|r| r.usable && r.omr_ned.is_none())
        .count();
    let omr_ned = NedSummary {
        n: ned_all.len(),
        mean: mean(&ned_all),
        median_all: median(ned_all),
        median_transcribed: median(ned_tx),
        missing_convention: NED_MISSING,
        n_no_output,
        n_missing,
    };

    Ok(SuiteSummary {
        suite: manifest.suite.clone(),
        engine: manifest.engine.clone(),
        scorer: format!("eval-harness {}", env!("CARGO_PKG_VERSION")),
        n_total,
        n_usable,
        usable_rate,
        strict_f1,
        omr_ned,
        n_empty_ref,
        exact_match,
        pieces: rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_of(entries: Vec<ManifestPiece>) -> Manifest {
        Manifest {
            suite: "s".to_string(),
            engine: "e".to_string(),
            pieces: entries,
        }
    }

    fn usable(slug: &str, report: &str) -> ManifestPiece {
        ManifestPiece {
            slug: slug.to_string(),
            report: Some(PathBuf::from(report)),
            no_output: false,
            omr_ned: None,
            reference_sha256: None,
            hypothesis_sha256: None,
        }
    }

    fn write_report(dir: &std::path::Path, name: &str, body: serde_json::Value) -> String {
        let p = dir.join(name);
        std::fs::write(&p, serde_json::to_string(&body).unwrap()).unwrap();
        p.to_str().unwrap().to_string()
    }

    #[test]
    fn median_matches_reference_definition() {
        assert_eq!(median(vec![]), 0.0);
        assert_eq!(median(vec![0.5]), 0.5);
        assert_eq!(median(vec![0.3, 0.1, 0.2]), 0.2);
        // Even count: average of the two middles (reference score.py med).
        assert_eq!(median(vec![0.4, 0.1, 0.3, 0.2]), 0.25);
    }

    #[test]
    fn empty_manifest_scores_zero() {
        let s = summarize(&manifest_of(vec![])).unwrap();
        assert_eq!((s.n_total, s.n_usable, s.usable_rate), (0, 0, 0.0));
        assert_eq!((s.strict_f1.mean, s.strict_f1.median_all), (0.0, 0.0));
        assert_eq!(s.omr_ned.n, 0);
    }

    #[test]
    fn no_output_scores_zero_but_counts() {
        let m = manifest_of(vec![
            ManifestPiece {
                slug: "gone".to_string(),
                report: None,
                no_output: true,
                omr_ned: None,
                reference_sha256: None,
                hypothesis_sha256: None,
            },
            ManifestPiece {
                slug: "missing-file".to_string(),
                report: Some(PathBuf::from("does-not-exist.json")),
                no_output: false,
                omr_ned: None,
                reference_sha256: None,
                hypothesis_sha256: None,
            },
        ]);
        let s = summarize(&m).unwrap();
        assert_eq!((s.n_total, s.n_usable), (2, 0));
        assert_eq!(s.usable_rate, 0.0);
        assert_eq!(s.strict_f1.mean, 0.0);
        assert_eq!(s.strict_f1.median_all, 0.0);
        // Bundle missing=1.0 convention for NED.
        assert_eq!(s.omr_ned.mean, 1.0);
        assert_eq!(s.omr_ned.n_no_output, 2);
        assert!(s.pieces.iter().all(|r| !r.usable && r.no_output));
    }

    #[test]
    fn mixed_suite_reports_mean_median_usable_rate() {
        let dir = std::env::temp_dir().join("eval-harness-suite-test");
        std::fs::create_dir_all(&dir).unwrap();
        let a = write_report(
            &dir,
            "a.json",
            serde_json::json!({"strict_f1": 1.0, "omr_ned": 0.1}),
        );
        let b = write_report(&dir, "b.json", serde_json::json!({"f1": 0.5}));
        let m = manifest_of(vec![
            usable("a", &a),
            usable("b", &b),
            ManifestPiece {
                slug: "c".to_string(),
                report: None,
                no_output: true,
                omr_ned: None,
                reference_sha256: None,
                hypothesis_sha256: None,
            },
        ]);
        let s = summarize(&m).unwrap();
        assert_eq!((s.n_total, s.n_usable), (3, 2));
        assert!((s.usable_rate - 2.0 / 3.0).abs() < 1e-12);
        // No-output zero stays in the mean and median_all denominators.
        assert!((s.strict_f1.mean - 0.5).abs() < 1e-12);
        assert_eq!(s.strict_f1.median_all, 0.5);
        assert_eq!(s.strict_f1.median_transcribed, 0.75);
        // NED: usable-with-value (0.1) + no-output at 1.0; usable "b"
        // without a value is excluded, counted as n_missing.
        assert_eq!(s.omr_ned.n, 2);
        assert_eq!(s.omr_ned.n_missing, 1);
        assert!((s.omr_ned.mean - 0.55).abs() < 1e-12);
        assert_eq!(s.omr_ned.median_transcribed, 0.1);
    }

    #[test]
    fn empty_reference_rows_leave_f1_median_and_count_exact() {
        let dir = std::env::temp_dir().join("eval-harness-suite-test");
        std::fs::create_dir_all(&dir).unwrap();
        let e1 = write_report(
            &dir,
            "e1.json",
            serde_json::json!({"empty_ref": true, "exact_match": 1.0}),
        );
        let e0 = write_report(
            &dir,
            "e0.json",
            serde_json::json!({"empty_ref": true, "exact_match": 0.0}),
        );
        let a = write_report(&dir, "a2.json", serde_json::json!({"strict_f1": 0.5}));
        let m = manifest_of(vec![usable("e1", &e1), usable("e0", &e0), usable("a", &a)]);
        let s = summarize(&m).unwrap();
        assert_eq!((s.n_total, s.n_usable), (3, 3));
        assert_eq!(s.n_empty_ref, 2);
        let em = s.exact_match.unwrap();
        assert_eq!((em.n, em.n_exact), (2, 1));
        assert!((em.rate - 0.5).abs() < 1e-12);
        // F1 aggregates see only the scored piece — never a pooled 1.0.
        assert!((s.strict_f1.mean - 0.5).abs() < 1e-12);
        assert_eq!(s.strict_f1.median_all, 0.5);
        assert_eq!(s.strict_f1.median_transcribed, 0.5);
        // Empty rows hold no F1 and no NED (counted missing, not pooled).
        assert!(s
            .pieces
            .iter()
            .filter(|r| r.empty_ref)
            .all(|r| r.strict_f1.is_none() && r.omr_ned.is_none()));
        assert_eq!(s.omr_ned.n, 0);
        assert_eq!(s.omr_ned.n_missing, 3);
    }

    #[test]
    fn empty_ref_without_exact_match_is_an_error() {
        let dir = std::env::temp_dir().join("eval-harness-suite-test");
        std::fs::create_dir_all(&dir).unwrap();
        let bad = write_report(&dir, "noe.json", serde_json::json!({"empty_ref": true}));
        let m = manifest_of(vec![usable("noe", &bad)]);
        assert!(summarize(&m).is_err());
    }

    #[test]
    fn manifest_ned_overrides_report_ned() {
        let dir = std::env::temp_dir().join("eval-harness-suite-test");
        std::fs::create_dir_all(&dir).unwrap();
        let a = write_report(
            &dir,
            "c.json",
            serde_json::json!({"strict_f1": 0.8, "omr_ned": 0.5}),
        );
        let m = manifest_of(vec![ManifestPiece {
            slug: "c".to_string(),
            report: Some(PathBuf::from(&a)),
            no_output: false,
            omr_ned: Some(0.2),
            reference_sha256: None,
            hypothesis_sha256: None,
        }]);
        let s = summarize(&m).unwrap();
        assert_eq!(s.pieces[0].omr_ned, Some(0.2));
    }

    #[test]
    fn corrupt_present_report_is_an_error_not_a_zero() {
        let dir = std::env::temp_dir().join("eval-harness-suite-test");
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("bad.json");
        std::fs::write(&bad, b"not json").unwrap();
        let m = manifest_of(vec![usable("bad", bad.to_str().unwrap())]);
        assert!(summarize(&m).is_err());
    }
}
