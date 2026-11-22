//! Measure-atomic top-2 lattice sidecar (Phase 4, Task T4.2).
//!
//! A `<stem>_lattice.json` file carries, per measure, a confidence
//! score, an optional pixel bounding box, and the top-2 candidate
//! transcriptions for 1-click user review. `bekern2musicxml` (and
//! `pack`) accept an external `--lattice-in <path.json>` producer
//! file (e.g. from the S17 measure verifier), validate it, rank
//! measures by `cluster_size * (1.0 - confidence)` descending, and
//! copy it to `<out>/<stem>_lattice.json`.
//!
//! Format:
//! ```json
//! {"stem": "ds0091-3",
//!  "measures": [
//!    {"measure": 26, "confidence": 0.42, "cluster_size": 14,
//!     "bbox": [104.0, 208.0, 2276.0, 655.0],
//!     "candidates": ["voice-1 quarter C 4 bar", "voice-1 quarter D 4 bar"]},
//!    {"measure": 27, "confidence": 0.97, "cluster_size": 12,
//!     "candidates": ["voice-1 half E 4 bar", "voice-1 half E 4 bar"]}
//!  ]}
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;

/// One measure entry: confidence in `[0, 1]`, note count for ranking,
/// optional `[x1, y1, x2, y2]` pixel box, and exactly the top-2
/// candidate transcriptions (bekern+ token strings).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LatticeMeasure {
    pub measure: usize,
    pub confidence: f64,
    pub cluster_size: usize,
    #[serde(default)]
    pub bbox: Option<[f64; 4]>,
    pub candidates: Vec<String>,
}

/// Producer lattice document, keyed by score stem.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LatticeDoc {
    pub stem: String,
    pub measures: Vec<LatticeMeasure>,
}

/// Ranked (descending) output row: review the most uncertain,
/// highest-note-count measures first.
#[derive(Debug, Clone, Serialize)]
pub struct RankedMeasure {
    pub rank: usize,
    pub rank_score: f64,
    #[serde(flatten)]
    pub measure: LatticeMeasure,
}

/// Rank key: big uncertain clusters first.
pub fn rank_score(m: &LatticeMeasure) -> f64 {
    m.cluster_size as f64 * (1.0 - m.confidence)
}

/// Read + validate a producer lattice file. Fails loud on any shape
/// violation: confidence outside `[0, 1]`, non-finite bbox entries,
/// anything but exactly 2 candidates, or duplicate measure ordinals.
pub fn read_lattice(path: &Path) -> Result<LatticeDoc, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let doc: LatticeDoc = serde_json::from_str(&text)
        .map_err(|e| format!("parse {}: {e}", path.display()))?;
    if doc.stem.trim().is_empty() {
        return Err(format!("{}: empty stem", path.display()));
    }
    let mut seen = std::collections::BTreeSet::new();
    for m in &doc.measures {
        if !(0.0..=1.0).contains(&m.confidence) || !m.confidence.is_finite() {
            return Err(format!(
                "{}: measure {} confidence {} not in [0, 1]",
                path.display(),
                m.measure,
                m.confidence
            ));
        }
        if m.candidates.len() != 2
            || m.candidates.iter().any(|c| c.trim().is_empty())
        {
            return Err(format!(
                "{}: measure {} needs exactly 2 non-empty candidates, got {}",
                path.display(),
                m.measure,
                m.candidates.len()
            ));
        }
        if let Some(b) = m.bbox {
            if b.iter().any(|v| !v.is_finite()) {
                return Err(format!(
                    "{}: measure {} bbox not finite",
                    path.display(),
                    m.measure
                ));
            }
        }
        if !seen.insert(m.measure) {
            return Err(format!(
                "{}: duplicate measure {}",
                path.display(),
                m.measure
            ));
        }
    }
    Ok(doc)
}

/// Rank measures descending by [`rank_score`] and number ranks from 1.
pub fn rank_measures(doc: &LatticeDoc) -> Vec<RankedMeasure> {
    let mut rows: Vec<RankedMeasure> = doc
        .measures
        .iter()
        .cloned()
        .map(|measure| {
            let rank_score = rank_score(&measure);
            RankedMeasure { rank: 0, rank_score, measure }
        })
        .collect();
    rows.sort_by(|a, b| {
        b.rank_score
            .partial_cmp(&a.rank_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, row) in rows.iter_mut().enumerate() {
        row.rank = i + 1;
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc() -> LatticeDoc {
        serde_json::from_str(
            r#"{"stem": "dsX",
                "measures": [
                  {"measure": 26, "confidence": 0.42, "cluster_size": 14,
                   "bbox": [104.0, 208.0, 2276.0, 655.0],
                   "candidates": ["voice-1 quarter C 4 bar",
                                  "voice-1 quarter D 4 bar"]},
                  {"measure": 27, "confidence": 0.97, "cluster_size": 12,
                   "candidates": ["voice-1 half E 4 bar",
                                  "voice-1 half E 4 bar"]}]}"#,
        )
        .unwrap()
    }

    #[test]
    fn rank_orders_uncertain_first() {
        let rows = rank_measures(&doc());
        assert_eq!(rows.len(), 2);
        // 14 * 0.58 = 8.12 > 12 * 0.03 = 0.36.
        assert_eq!(rows[0].measure.measure, 26);
        assert_eq!(rows[0].rank, 1);
        assert!((rows[0].rank_score - 8.12).abs() < 1e-9);
        assert_eq!(rows[1].measure.measure, 27);
        assert_eq!(rows[1].rank, 2);
    }

    #[test]
    fn rejects_bad_confidence() {
        let mut d = doc();
        d.measures[0].confidence = 1.5;
        let dir = std::env::temp_dir();
        let p = dir.join("lattice-bad-conf.json");
        std::fs::write(&p, serde_json::to_string(&d).unwrap()).unwrap();
        assert!(read_lattice(&p).is_err());
    }

    #[test]
    fn rejects_wrong_candidate_count() {
        let mut d = doc();
        d.measures[1].candidates.pop();
        let dir = std::env::temp_dir();
        let p = dir.join("lattice-bad-cands.json");
        std::fs::write(&p, serde_json::to_string(&d).unwrap()).unwrap();
        let err = read_lattice(&p).unwrap_err();
        assert!(err.contains("exactly 2"), "{err}");
    }

    #[test]
    fn rejects_duplicate_measures() {
        let mut d = doc();
        d.measures.push(d.measures[0].clone());
        let dir = std::env::temp_dir();
        let p = dir.join("lattice-dup.json");
        std::fs::write(&p, serde_json::to_string(&d).unwrap()).unwrap();
        assert!(read_lattice(&p).is_err());
    }

    #[test]
    fn accepts_missing_bbox() {
        let rows = rank_measures(&doc());
        assert!(rows[1].measure.bbox.is_none());
        assert_eq!(rows[0].measure.bbox, Some([104.0, 208.0, 2276.0, 655.0]));
    }
}
