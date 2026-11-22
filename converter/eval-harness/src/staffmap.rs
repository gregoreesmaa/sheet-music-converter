//! layout-staff output -> symbol-detect staff map.
//!
//! Flattens every staff line row on the page into one global entry (full
//! page used as a single crop). Spacing is the median inter-line gap over
//! 1px (sub-pixel gaps are peak-fitter duplicates, not spacings).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
struct InStaff {
    #[serde(default)]
    lines: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct InSystem {
    #[serde(default)]
    staves: Vec<InStaff>,
}

#[derive(Debug, Clone, Deserialize)]
struct InLayout {
    #[serde(default)]
    systems: Vec<InSystem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MapEntry {
    staff_lines: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    staff_spacing: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MapDoc {
    systems: Vec<MapEntry>,
}

/// Sorted union of every staff line row on the page.
fn union_lines(layout: &InLayout) -> Vec<f64> {
    let mut lines: Vec<f64> = layout
        .systems
        .iter()
        .flat_map(|s| s.staves.iter())
        .flat_map(|st| st.lines.iter().copied())
        .collect();
    lines.sort_by(|a, b| a.partial_cmp(b).unwrap());
    lines
}

/// Median inter-line gap over 1px. `None` when fewer than two rows qualify.
fn median_gap(lines: &[f64]) -> Option<f64> {
    let mut gaps: Vec<f64> = lines
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&g| g > 1.0)
        .collect();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = gaps.len() / 2;
    Some(if gaps.len() % 2 == 1 {
        gaps[m]
    } else {
        (gaps[m - 1] + gaps[m]) / 2.0
    })
}

/// Convert a layout-staff document into a single-entry staff map.
fn convert(layout: &InLayout) -> MapDoc {
    let lines = union_lines(layout);
    let staff_spacing = median_gap(&lines);
    MapDoc {
        systems: vec![MapEntry {
            staff_lines: lines,
            staff_spacing,
        }],
    }
}

pub fn convert_str(text: &str) -> Result<String, String> {
    let layout: InLayout = serde_json::from_str(text).map_err(|e| format!("parse layout: {e}"))?;
    let doc = convert(&layout);
    serde_json::to_string_pretty(&doc).map_err(|e| format!("encode map: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> InLayout {
        serde_json::from_value(serde_json::json!({
            "systems": [
                {"staves": [
                    {"lines": [900.0, 916.0, 932.0, 948.0, 964.0]},
                    {"lines": [100.0, 116.0, 132.0, 148.0, 164.0]}
                ]},
                {"staves": [
                    {"lines": [200.0, 216.0, 232.0, 248.0, 264.0]}
                ]}
            ]
        }))
        .unwrap()
    }

    #[test]
    fn union_collects_all_rows_sorted() {
        let lines = union_lines(&layout());
        assert_eq!(lines.len(), 15);
        assert_eq!(lines[0], 100.0);
        assert_eq!(lines[14], 964.0);
        assert!(lines.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn spacing_is_median_gap_over_one_px() {
        let lines = union_lines(&layout());
        assert_eq!(median_gap(&lines), Some(16.0));
    }

    #[test]
    fn subpixel_gaps_do_not_set_spacing() {
        assert_eq!(median_gap(&[100.0, 100.4, 116.0, 132.0, 148.0]), Some(16.0));
        assert_eq!(median_gap(&[100.0, 100.4]), None);
        assert_eq!(median_gap(&[]), None);
    }

    #[test]
    fn entry_carries_union_and_spacing() {
        let doc = convert(&layout());
        assert_eq!(doc.systems.len(), 1);
        assert_eq!(doc.systems[0].staff_lines.len(), 15);
        assert_eq!(doc.systems[0].staff_spacing, Some(16.0));
    }

    #[test]
    fn empty_layout_yields_empty_entry() {
        let layout: InLayout = serde_json::from_value(serde_json::json!({
            "systems": []
        }))
        .unwrap();
        let doc = convert(&layout);
        assert_eq!(doc.systems.len(), 1);
        assert!(doc.systems[0].staff_lines.is_empty());
        assert_eq!(doc.systems[0].staff_spacing, None);
    }
}
