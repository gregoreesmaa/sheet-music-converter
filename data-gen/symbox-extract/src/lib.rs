//! MuseScore SVG page -> glyph/staff boxes.
//!
//! MuseScore SVG exports carry semantic classes (`Note`, `Stem`, `Clef`,
//! `Rest`, `StaffLines`, ...) but no element ids. This module parses the
//! path data (absolute `M`/`L`/`C` only — anything else is an error, so a
//! new exporter shape fails loudly instead of emitting bad boxes),
//! computes exact curve bboxes, and groups staff lines into staves.

use serde::{Deserialize, Serialize};

/// Identity affine map.
pub const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// One vector element of interest from the SVG.
#[derive(Debug, Clone)]
pub struct SvgElem {
    /// Class attribute (`Note`, `StaffLines`, ...). Polylines use it too.
    pub cls: String,
    /// Path data for `<path>`, point list for `<polyline points>`.
    pub data: String,
    /// True for `<polyline>`, false for `<path>`.
    pub is_polyline: bool,
    /// Affine `matrix(a,b,c,d,e,f)` from `transform`, if present.
    pub transform: Option<[f64; 6]>,
    /// `stroke-width` for polylines (ink half-width source); paths are
    /// filled and leave this as `None`.
    pub stroke: Option<f64>,
}

/// 2D affine map: x' = a*x + c*y + e, y' = b*x + d*y + f.
pub fn parse_affine(transform: Option<&str>) -> Result<[f64; 6], String> {
    let Some(t) = transform else {
        return Ok([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    };
    let t = t.trim();
    if let Some(inner) = t.strip_prefix("matrix(").and_then(|s| s.strip_suffix(')')) {
        let nums: Vec<f64> = inner
            .split([',', ' ', '\t'])
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<f64>()
                    .map_err(|_| format!("bad matrix number {s:?}"))
            })
            .collect::<Result<_, _>>()?;
        if nums.len() != 6 {
            return Err("matrix needs 6 numbers".to_string());
        }
        return Ok([nums[0], nums[1], nums[2], nums[3], nums[4], nums[5]]);
    }
    if let Some(inner) = t
        .strip_prefix("translate(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let nums: Vec<f64> = inner
            .split([',', ' ', '\t'])
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<f64>()
                    .map_err(|_| format!("bad translate number {s:?}"))
            })
            .collect::<Result<_, _>>()?;
        return match nums[..] {
            [x, y] => Ok([1.0, 0.0, 0.0, 1.0, x, y]),
            [x] => Ok([1.0, 0.0, 0.0, 1.0, x, 0.0]),
            _ => Err("translate needs 1-2 numbers".to_string()),
        };
    }
    Err(format!("unsupported transform {t:?}"))
}

/// Staff geometry in SVG pixels: ascending line rows plus page bbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaffGeom {
    pub lines: Vec<f64>,
    pub bbox: [f64; 4],
}

/// Glyph record in output pixels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlyphBox {
    pub cls: String,
    pub bbox: [f64; 4],
}

/// Page-level box document, already scaled to the sister PNG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxesDoc {
    pub page: u32,
    pub svg_view: [f64; 2],
    pub png_size: [u32; 2],
    pub scale: [f64; 2],
    pub staves: Vec<StaffGeom>,
    pub glyphs: Vec<GlyphBox>,
}

/// All `<path class>` / `<polyline class>` elements in document order.
/// Elements without a (non-empty) class are skipped (page background).
pub fn parse_svg_elems(svg: &str) -> Result<Vec<SvgElem>, String> {
    let mut out = Vec::new();
    let mut rest = svg;
    loop {
        let pp = rest.find("<path");
        let lp = rest.find("<polyline");
        let (is_poly, start) = match (pp, lp) {
            (None, None) => break,
            (Some(p), None) => (false, p),
            (None, Some(p)) => (true, p),
            (Some(p), Some(q)) => {
                if p < q {
                    (false, p)
                } else {
                    (true, q)
                }
            }
        };
        let tag_end = rest[start..].find('>').ok_or("unterminated svg element")? + start;
        let tag = &rest[start..=tag_end];
        let cls = attr(tag, "class").unwrap_or_default();
        rest = &rest[tag_end + 1..];
        if cls.is_empty() {
            continue;
        }
        let transform = match attr(tag, "transform") {
            Some(t) => Some(parse_affine(Some(&t))?),
            None => None,
        };
        if is_poly {
            let data = attr(tag, "points").ok_or("polyline without points")?;
            let stroke = match attr(tag, "stroke-width") {
                Some(s) => Some(
                    s.parse::<f64>()
                        .map_err(|_| format!("bad stroke-width {s:?}"))?,
                ),
                None => None,
            };
            out.push(SvgElem {
                cls,
                data,
                is_polyline: true,
                transform,
                stroke,
            });
        } else if tag.ends_with("/>") {
            let d = attr(tag, "d").ok_or("path without d")?;
            out.push(SvgElem {
                cls,
                data: d,
                is_polyline: false,
                transform,
                stroke: None,
            });
        } else {
            let close = "</path>";
            let end = rest.find(close).ok_or("path without closing tag")?;
            let body = &rest[..end];
            let d = attr(body, "d")
                .or_else(|| attr(tag, "d"))
                .ok_or("path without d")?;
            out.push(SvgElem {
                cls,
                data: d,
                is_polyline: false,
                transform,
                stroke: None,
            });
            rest = &rest[end + close.len()..];
        }
    }
    Ok(out)
}

fn attr(tag: &str, name: &str) -> Option<String> {
    let pat = format!("{name}=\"");
    let p = tag.find(&pat)? + pat.len();
    let e = tag[p..].find('"')? + p;
    Some(tag[p..e].to_string())
}

/// Exact bbox `[x0, y0, x1, y1]` of absolute `M`/`L`/`C` path data.
/// Anything else (relative commands, arcs, quadratics) is an error.
pub fn path_bbox(d: &str) -> Result<[f64; 4], String> {
    path_bbox_affine(d, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
}

/// Exact bbox with a 2D affine map applied. Affines send cubics to
/// cubics, so control points transform first and extrema come after —
/// exact under rotation and shear, not just scale+translate.
pub fn path_bbox_affine(d: &str, m: [f64; 6]) -> Result<[f64; 4], String> {
    let segs = segments(d)?;
    segs_bbox(segs.iter().flatten(), m)
}

fn segs_bbox<'a, I>(it: I, m: [f64; 6]) -> Result<[f64; 4], String>
where
    I: Iterator<Item = &'a Seg>,
{
    let mut b: Option<[f64; 4]> = None;
    let mut eat = |x: f64, y: f64| {
        if let Some(bb) = b.as_mut() {
            bb[0] = bb[0].min(x);
            bb[1] = bb[1].min(y);
            bb[2] = bb[2].max(x);
            bb[3] = bb[3].max(y);
        } else {
            b = Some([x, y, x, y]);
        }
    };
    for s in it {
        match *s {
            Seg::Dot(p) => {
                let (x, y) = map_pt(p, m);
                eat(x, y);
            }
            Seg::Curve(p0, p1, p2, p3) => {
                let (p0, p1, p2, p3) = (map_pt(p0, m), map_pt(p1, m), map_pt(p2, m), map_pt(p3, m));
                cubic_extrema(p0, p1, p2, p3, &mut eat);
            }
        }
    }
    b.ok_or("path has no points".to_string())
}

/// Per-subpath bboxes (split at each `M`), affine-mapped.
pub fn subpath_bboxes_affine(d: &str, m: [f64; 6]) -> Result<Vec<[f64; 4]>, String> {
    segments(d)?
        .iter()
        .map(|sp| segs_bbox(sp.iter(), m))
        .collect()
}

/// Per-subpath bboxes with the identity map.
pub fn subpath_bboxes(d: &str) -> Result<Vec<[f64; 4]>, String> {
    subpath_bboxes_affine(d, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
}

/// `"NoteVoid"` when one subpath bbox strictly contains another (open
/// notehead: outer ring + inner counter), else `"NoteBlack"`.
/// Affines preserve containment, so raw path data suffices.
pub fn classify_note(d: &str) -> Result<&'static str, String> {
    let boxes = subpath_bboxes(d)?;
    for (i, a) in boxes.iter().enumerate() {
        for b in boxes.iter().skip(i + 1) {
            if contains(a, b) || contains(b, a) {
                return Ok("NoteVoid");
            }
        }
    }
    Ok("NoteBlack")
}

fn contains(outer: &[f64; 4], inner: &[f64; 4]) -> bool {
    const EPS: f64 = 1e-9;
    outer[0] <= inner[0] + EPS
        && outer[1] <= inner[1] + EPS
        && outer[2] >= inner[2] - EPS
        && outer[3] >= inner[3] + EPS
        && (outer[0] < inner[0] - EPS
            || outer[1] < inner[1] - EPS
            || outer[2] > inner[2] + EPS
            || outer[3] > inner[3] + EPS)
}

/// Ink-tight bbox of a `<polyline points>` stroke: point bbox expanded
/// by half the stroke width (scaled for non-uniform affines).
pub fn polyline_bbox_affine(points: &str, stroke: f64, m: [f64; 6]) -> Result<[f64; 4], String> {
    let nums: Vec<f64> = points
        .split([',', ' ', '\t', '\n', '\r'])
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<f64>()
                .map_err(|_| format!("bad polyline point {s:?}"))
        })
        .collect::<Result<_, _>>()?;
    if nums.len() < 4 || !nums.len().is_multiple_of(2) {
        return Err("polyline needs 2+ points".to_string());
    }
    let mut b: Option<[f64; 4]> = None;
    for w in nums.as_chunks::<2>().0 {
        let (x, y) = map_pt((w[0], w[1]), m);
        if let Some(bb) = b.as_mut() {
            bb[0] = bb[0].min(x);
            bb[1] = bb[1].min(y);
            bb[2] = bb[2].max(x);
            bb[3] = bb[3].max(y);
        } else {
            b = Some([x, y, x, y]);
        }
    }
    let mut b = b.ok_or("polyline needs 2+ points".to_string())?;
    // Uniform scale of the affine; exact for translate/scale/rotate.
    let s = (((m[0] * m[0] + m[1] * m[1]) + (m[2] * m[2] + m[3] * m[3])) / 2.0).sqrt();
    let h = stroke / 2.0 * s;
    b[0] -= h;
    b[1] -= h;
    b[2] += h;
    b[3] += h;
    Ok(b)
}

/// Median staff-line gap over all staves (the staff space), if any.
pub fn staff_space(staves: &[StaffGeom]) -> Option<f64> {
    let mut gaps: Vec<f64> = staves
        .iter()
        .flat_map(|s| s.lines.windows(2).map(|w| w[1] - w[0]))
        .filter(|g| *g > 0.0)
        .collect();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(gaps[gaps.len() / 2])
}

/// Edge-to-edge gap between boxes (0 when they touch or overlap).
/// Center distance is meaningless against long spans like beam groups:
/// a beamlet under a 300px beam can sit 150px from the beam's center.
fn gap(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let dx = (a[0] - b[2]).max(b[0] - a[2]).max(0.0);
    let dy = (a[1] - b[3]).max(b[1] - a[3]).max(0.0);
    (dx * dx + dy * dy).sqrt()
}

/// Relabel `Hook` boxes as `Flag` when they are beam-distant (no `Beam
/// center within 3 staff spaces) but stem-near (a `Stem` center within
/// 4 staff spaces). Hooks near beams are beamlets, not flags.
pub fn relabel_flags(glyphs: &mut [GlyphBox], sp: f64) {
    let beams: Vec<[f64; 4]> = glyphs
        .iter()
        .filter(|g| g.cls == "Beam")
        .map(|g| g.bbox)
        .collect();
    let stems: Vec<[f64; 4]> = glyphs
        .iter()
        .filter(|g| g.cls == "Stem")
        .map(|g| g.bbox)
        .collect();
    for g in glyphs.iter_mut() {
        if g.cls != "Hook" {
            continue;
        }
        let beam_near = beams.iter().any(|b| gap(&g.bbox, b) <= 3.0 * sp);
        let stem_near = stems.iter().any(|s| gap(&g.bbox, s) <= 4.0 * sp);
        if !beam_near && stem_near {
            g.cls = "Flag".to_string();
        }
    }
}

fn map_pt(p: (f64, f64), m: [f64; 6]) -> (f64, f64) {
    (
        m[0] * p.0 + m[2] * p.1 + m[4],
        m[1] * p.0 + m[3] * p.1 + m[5],
    )
}

#[derive(Debug, Clone, Copy)]
enum Seg {
    /// A point the bbox must include (move/line endpoint).
    Dot((f64, f64)),
    /// Cubic from p0 with controls p1, p2 to p3.
    Curve((f64, f64), (f64, f64), (f64, f64), (f64, f64)),
}

/// Parse path data into per-subpath segments. Each `M` starts a new
/// subpath; `Z` closes the subpath (no new point: the start point is
/// already recorded).
fn segments(d: &str) -> Result<Vec<Vec<Seg>>, String> {
    let tokens = tokenize(d)?;
    if tokens.is_empty() {
        return Err("empty path data".to_string());
    }
    let mut out: Vec<Vec<Seg>> = Vec::new();
    let mut curp: Vec<Seg> = Vec::new();
    let mut i = 0;
    let mut cur = (0.0, 0.0);
    let mut have_point = false;
    while i < tokens.len() {
        let t = &tokens[i];
        i += 1;
        match t {
            Tok::Cmd('M') => {
                if !curp.is_empty() {
                    out.push(std::mem::take(&mut curp));
                }
                let (x, y) = pair(&tokens, &mut i)?;
                cur = (x, y);
                curp.push(Seg::Dot(cur));
                have_point = true;
                while peek_num(&tokens, i) {
                    let (x, y) = pair(&tokens, &mut i)?;
                    curp.push(Seg::Dot((x, y)));
                    cur = (x, y);
                }
            }
            Tok::Cmd('L') => {
                if !have_point {
                    return Err("L before any point".to_string());
                }
                if !peek_num(&tokens, i) {
                    return Err("L without parameters".to_string());
                }
                while peek_num(&tokens, i) {
                    let (x, y) = pair(&tokens, &mut i)?;
                    curp.push(Seg::Dot((x, y)));
                    cur = (x, y);
                }
            }
            Tok::Cmd('C') => {
                if !have_point {
                    return Err("C before any point".to_string());
                }
                if !peek_num(&tokens, i) {
                    return Err("C without parameters".to_string());
                }
                while peek_num(&tokens, i) {
                    let p = sextuple(&tokens, &mut i)?;
                    curp.push(Seg::Curve(cur, (p[0], p[1]), (p[2], p[3]), (p[4], p[5])));
                    cur = (p[4], p[5]);
                }
            }
            Tok::Cmd('Z') => {
                let _ = cur;
            }
            Tok::Cmd(c) => return Err(format!("unsupported path command {c}")),
            Tok::Num(_) => return Err("number without command".to_string()),
        }
    }
    if !curp.is_empty() {
        out.push(curp);
    }
    Ok(out)
}

#[derive(Debug, Clone)]
enum Tok {
    Cmd(char),
    Num(f64),
}

fn tokenize(d: &str) -> Result<Vec<Tok>, String> {
    let mut out = Vec::new();
    let bytes = d.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == ',' || c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_alphabetic() {
            out.push(Tok::Cmd(c));
            i += 1;
        } else if c == '-' || c == '+' || c == '.' || c.is_ascii_digit() {
            let start = i;
            if bytes[i] as char == '-' || bytes[i] as char == '+' {
                i += 1;
            }
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] as char == '.') {
                i += 1;
            }
            if i < bytes.len() && (bytes[i] as char == 'e' || bytes[i] as char == 'E') {
                i += 1;
                if i < bytes.len() && (bytes[i] as char == '-' || bytes[i] as char == '+') {
                    i += 1;
                }
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let num: f64 = d[start..i]
                .parse()
                .map_err(|_| format!("bad number {:?}", &d[start..i]))?;
            out.push(Tok::Num(num));
        } else {
            return Err(format!("bad path character {c:?}"));
        }
    }
    Ok(out)
}

fn peek_num(tokens: &[Tok], i: usize) -> bool {
    matches!(tokens.get(i), Some(Tok::Num(_)))
}

fn take_num(tokens: &[Tok], i: &mut usize) -> Result<f64, String> {
    match tokens.get(*i) {
        Some(Tok::Num(v)) => {
            *i += 1;
            Ok(*v)
        }
        _ => Err("expected number".to_string()),
    }
}

fn pair(tokens: &[Tok], i: &mut usize) -> Result<(f64, f64), String> {
    Ok((take_num(tokens, i)?, take_num(tokens, i)?))
}

fn sextuple(tokens: &[Tok], i: &mut usize) -> Result<[f64; 6], String> {
    Ok([
        take_num(tokens, i)?,
        take_num(tokens, i)?,
        take_num(tokens, i)?,
        take_num(tokens, i)?,
        take_num(tokens, i)?,
        take_num(tokens, i)?,
    ])
}

/// Feed segment endpoints plus interior x/y extrema of a cubic to `eat`.
fn cubic_extrema(
    p0: (f64, f64),
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
    eat: &mut impl FnMut(f64, f64),
) {
    eat(p0.0, p0.1);
    eat(p3.0, p3.1);
    for dim in [0, 1] {
        let (a0, a1, a2, a3) = if dim == 0 {
            (p0.0, p1.0, p2.0, p3.0)
        } else {
            (p0.1, p1.1, p2.1, p3.1)
        };
        // B'(t)/3 = A t^2 + B t + C.
        let a = a3 - 3.0 * a2 + 3.0 * a1 - a0;
        let b = 2.0 * a2 - 4.0 * a1 + 2.0 * a0;
        let c = a1 - a0;
        for t in solve_quad(a, b, c) {
            if t > 0.0 && t < 1.0 {
                let mt = 1.0 - t;
                let v = mt * mt * mt * a0
                    + 3.0 * mt * mt * t * a1
                    + 3.0 * mt * t * t * a2
                    + t * t * t * a3;
                if dim == 0 {
                    eat(v, p0.1);
                } else {
                    eat(p0.0, v);
                }
            }
        }
    }
}

fn solve_quad(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a.abs() < 1e-12 {
        if b.abs() < 1e-12 {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return Vec::new();
    }
    let s = disc.sqrt();
    vec![(-b - s) / (2.0 * a), (-b + s) / (2.0 * a)]
}

/// Group ascending staff-line rows `(x0, x1, y)` into staves of 3+ lines.
/// A gap over 2.5x the median line gap starts a new staff.
pub fn group_staves(rows: &[(f64, f64, f64)]) -> Vec<StaffGeom> {
    if rows.len() < 3 {
        return Vec::new();
    }
    let mut sorted = rows.to_vec();
    sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    let mut gaps: Vec<f64> = sorted.windows(2).map(|w| w[1].2 - w[0].2).collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = gaps[gaps.len() / 2].max(1e-9);
    let mut staves = Vec::new();
    let mut cur: Vec<(f64, f64, f64)> = vec![sorted[0]];
    for w in sorted.windows(2) {
        if w[1].2 - w[0].2 > 2.5 * median {
            if cur.len() >= 3 {
                staves.push(finish_staff(&cur));
            }
            cur = Vec::new();
        }
        cur.push(w[1]);
    }
    if cur.len() >= 3 {
        staves.push(finish_staff(&cur));
    }
    staves
}

fn finish_staff(rows: &[(f64, f64, f64)]) -> StaffGeom {
    let lines: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let x0 = rows.iter().map(|r| r.0).fold(f64::INFINITY, f64::min);
    let x1 = rows.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max);
    let (top, bottom) = (lines[0], lines[lines.len() - 1]);
    StaffGeom {
        lines,
        bbox: [x0, top, x1, bottom],
    }
}

/// `[width, height]` of the SVG `viewBox`.
pub fn svg_viewbox(svg: &str) -> Result<[f64; 2], String> {
    let p = svg.find("viewBox=\"").ok_or("no viewBox")? + "viewBox=\"".len();
    let e = svg[p..].find('"').ok_or("unterminated viewBox")? + p;
    let nums: Vec<f64> = svg[p..e]
        .split([',', ' ', '\t', '\n', '\r'])
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<f64>()
                .map_err(|_| "bad viewBox number".to_string())
        })
        .collect::<Result<_, _>>()?;
    if nums.len() != 4 {
        return Err("viewBox needs 4 numbers".to_string());
    }
    Ok([nums[2], nums[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_bbox_uses_curve_extrema_not_control_hull() {
        // S-curve: control hull reaches y=10, the curve peaks at y=7.5.
        let b = path_bbox("M0,0 C0,10 10,10 10,0").unwrap();
        assert!((b[0] - 0.0).abs() < 1e-9, "{b:?}");
        assert!((b[1] - 0.0).abs() < 1e-9, "{b:?}");
        assert!((b[2] - 10.0).abs() < 1e-9, "{b:?}");
        assert!((b[3] - 7.5).abs() < 1e-9, "{b:?}");
    }

    #[test]
    fn line_and_move_bbox() {
        let b = path_bbox("M10,20 L30,40").unwrap();
        assert_eq!(b, [10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn unknown_command_errors() {
        assert!(path_bbox("M0,0 Q5,5 10,10").is_err());
        assert!(path_bbox("").is_err());
    }

    #[test]
    fn staff_grouping_finds_five_line_staves() {
        let rows: Vec<(f64, f64, f64)> = [100, 110, 120, 130, 140, 300, 310, 320, 330, 340]
            .iter()
            .map(|y| (0.0, 500.0, *y as f64))
            .collect();
        let staves = group_staves(&rows);
        assert_eq!(staves.len(), 2);
        assert_eq!(staves[0].lines, vec![100.0, 110.0, 120.0, 130.0, 140.0]);
        assert_eq!(staves[1].lines[4], 340.0);
    }

    const SNIPPET: &str = r#"<svg viewBox="0 0 100 200">
<path class="Note" d="M10,20 L30,40"/>
<polyline class="StaffLines" points="0,100 90,100"/>
<path class="Clef" d="M0,0 C0,10 10,10 10,0"/>
</svg>"#;

    #[test]
    fn svg_elems_pick_classes_in_order() {
        let elems = parse_svg_elems(SNIPPET).unwrap();
        assert_eq!(elems.len(), 3);
        assert_eq!(elems[0].cls, "Note");
        assert!(!elems[0].is_polyline);
        assert_eq!(elems[1].cls, "StaffLines");
        assert!(elems[1].is_polyline);
        assert_eq!(elems[2].cls, "Clef");
    }

    #[test]
    fn affine_bbox_scales_and_translates_extrema() {
        // Same S-curve under matrix(2,0,0,2,10,20): extrema transform too.
        let b =
            path_bbox_affine("M0,0 C0,10 10,10 10,0", [2.0, 0.0, 0.0, 2.0, 10.0, 20.0]).unwrap();
        assert!((b[0] - 10.0).abs() < 1e-9, "{b:?}");
        assert!((b[1] - 20.0).abs() < 1e-9, "{b:?}");
        assert!((b[2] - 30.0).abs() < 1e-9, "{b:?}");
        assert!((b[3] - 35.0).abs() < 1e-9, "{b:?}");
    }

    #[test]
    fn self_closing_path_with_matrix_parses() {
        let svg = r#"<svg viewBox="0 0 100 200"><path class="Note" transform="matrix(1,0,0,1,5,6)" d="M0,0 L10,10"/></svg>"#;
        let elems = parse_svg_elems(svg).unwrap();
        assert_eq!(elems.len(), 1);
        assert_eq!(elems[0].transform, Some([1.0, 0.0, 0.0, 1.0, 5.0, 6.0]));
        let b = path_bbox_affine(&elems[0].data, elems[0].transform.unwrap()).unwrap();
        assert_eq!(b, [5.0, 6.0, 15.0, 16.0]);
    }

    #[test]
    fn empty_class_elements_are_skipped() {
        let svg = r#"<svg viewBox="0 0 100 200"><path class="" d="M0,0 L10,10"/><path class="Note" d="M0,0 L1,1"/></svg>"#;
        let elems = parse_svg_elems(svg).unwrap();
        assert_eq!(elems.len(), 1);
        assert_eq!(elems[0].cls, "Note");
    }

    #[test]
    fn viewbox_parses() {
        assert_eq!(svg_viewbox(SNIPPET).unwrap(), [100.0, 200.0]);
    }

    #[test]
    fn subpaths_split_at_each_moveto() {
        let b = subpath_bboxes("M0,0 L10,0 L10,10 L0,10 Z M3,3 L7,3 L7,7 L3,7 Z").unwrap();
        assert_eq!(b.len(), 2);
        assert_eq!(b[0], [0.0, 0.0, 10.0, 10.0]);
        assert_eq!(b[1], [3.0, 3.0, 7.0, 7.0]);
    }

    #[test]
    fn nested_subpaths_classify_void() {
        // Outer ring + inner counter: a void (open) notehead.
        assert_eq!(
            classify_note("M0,0 L10,0 L10,10 L0,10 Z M3,3 L7,3 L7,7 L3,7 Z").unwrap(),
            "NoteVoid"
        );
    }

    #[test]
    fn single_subpath_classifies_black() {
        assert_eq!(
            classify_note("M0,0 L10,0 L10,10 L0,10 Z").unwrap(),
            "NoteBlack"
        );
    }

    #[test]
    fn stem_polyline_expands_by_half_stroke() {
        // Real Stem element shape: two endpoints + stroke-width.
        let b = polyline_bbox_affine("6378.38,6125.38 6378.38,5891.8", 8.31, IDENTITY).unwrap();
        assert_eq!(b, [6374.225, 5887.645, 6382.535, 6129.535]);
    }

    #[test]
    fn barline_polyline_bbox_is_ink_tight() {
        let b = polyline_bbox_affine("9444.85,5944.34 9444.85,6255.01", 13.61, IDENTITY).unwrap();
        assert!((b[0] - (9444.85 - 6.805)).abs() < 1e-9, "{b:?}");
        assert!((b[2] - (9444.85 + 6.805)).abs() < 1e-9, "{b:?}");
    }

    #[test]
    fn staff_space_is_median_line_gap() {
        let staves = vec![StaffGeom {
            lines: vec![0.0, 10.0, 20.0, 30.0, 40.0],
            bbox: [0.0, 0.0, 100.0, 40.0],
        }];
        assert_eq!(staff_space(&staves), Some(10.0));
    }

    fn hook_box(x0: f64, y0: f64) -> GlyphBox {
        GlyphBox {
            cls: "Hook".to_string(),
            bbox: [x0, y0, x0 + 5.0, y0 + 5.0],
        }
    }

    #[test]
    fn beam_distant_stem_near_hook_becomes_flag() {
        // sp = 10: Beam center > 30 away, Stem center within 40.
        let mut glyphs = vec![
            hook_box(0.0, 0.0),
            GlyphBox {
                cls: "Beam".to_string(),
                bbox: [100.0, 0.0, 120.0, 6.0],
            },
            GlyphBox {
                cls: "Stem".to_string(),
                bbox: [20.0, -20.0, 22.0, 10.0],
            },
        ];
        relabel_flags(&mut glyphs, 10.0);
        assert_eq!(glyphs[0].cls, "Flag");
        assert_eq!(glyphs[1].cls, "Beam");
        assert_eq!(glyphs[2].cls, "Stem");
    }

    #[test]
    fn beam_near_hook_stays_hook() {
        let mut glyphs = vec![
            hook_box(0.0, 0.0),
            GlyphBox {
                cls: "Beam".to_string(),
                bbox: [10.0, 0.0, 30.0, 6.0],
            },
            GlyphBox {
                cls: "Stem".to_string(),
                bbox: [20.0, -20.0, 22.0, 10.0],
            },
        ];
        relabel_flags(&mut glyphs, 10.0);
        assert_eq!(glyphs[0].cls, "Hook");
    }

    #[test]
    fn beamlet_under_long_beam_stays_hook() {
        // Center distance to the beam group center is ~149px (> 3sp),
        // but the beam's x-range overlaps the hook: it is a beamlet.
        let mut glyphs = vec![
            hook_box(0.0, 0.0),
            GlyphBox {
                cls: "Beam".to_string(),
                bbox: [0.0, 20.0, 300.0, 26.0],
            },
            GlyphBox {
                cls: "Stem".to_string(),
                bbox: [20.0, -20.0, 22.0, 10.0],
            },
        ];
        relabel_flags(&mut glyphs, 10.0);
        assert_eq!(glyphs[0].cls, "Hook");
    }

    #[test]
    fn stemless_hook_stays_hook() {
        let mut glyphs = vec![
            hook_box(0.0, 0.0),
            GlyphBox {
                cls: "Beam".to_string(),
                bbox: [100.0, 0.0, 120.0, 6.0],
            },
        ];
        relabel_flags(&mut glyphs, 10.0);
        assert_eq!(glyphs[0].cls, "Hook");
    }

    #[test]
    fn disjoint_subpaths_classify_black() {
        // Two separate blobs are not a ring band.
        assert_eq!(
            classify_note("M0,0 L2,0 L2,2 L0,2 Z M5,5 L7,5 L7,7 L5,7 Z").unwrap(),
            "NoteBlack"
        );
    }
}
