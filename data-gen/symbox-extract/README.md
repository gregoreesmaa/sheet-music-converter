# symbox-extract

Rust lib + CLI that turns a MuseScore SVG page into glyph/staff boxes
JSON aligned to the sister PNG (the pilot label source for
`DETECTOR_DESIGN.md` §5).

MuseScore SVG exports carry semantic classes (`Note`, `Stem`, `Clef`,
`Rest`, `StaffLines`, ...) plus per-glyph `transform` matrices, but no
element ids. Staff geometry comes from `StaffLines` polylines; glyph
boxes come from exact path bboxes (cubic extrema, not control hulls)
with the affine applied to control points first, so rotation/shear
stay exact.

```bash
symbox-extract --svg page.svg --png page.png --out boxes.json --page 1
```

Output records `svg_view`, `png_size`, and the applied `scale`, plus
`staves` (line rows + bbox) and `glyphs` (class + bbox), all in PNG
pixels. For MuseScore-engraved pairs the scale is 1.0 (shared layout).

Verified on `renders/sample01` (all 7 pages, scale exactly 1.0):

| Page | SVG Notes | Import pitched | Tie stops | Expected | Match |
|---|---|---|---|---|---|
| 1 | 228 | 218 | 9 | 227 | +1 |
| 2 | 48 | 46 | 4 | 50 | −2 |
| 3 | 184 | 176 | 10 | 186 | −2 |
| 4–7 | exact on all four pages |||| OK |

Rule: SVG Notes = imported pitched notes + `<tie type="stop"/>` segments
rendered on that page (each stop tag is one extra drawn segment; the
import merges chains). 1048 vs 1051 total (99.7%): the ±3 residuals fit
known phenomena the fusion step handles — shared noteheads across
voices (one drawn head, two logical notes) and overlap-dropped notes
still drawn. Staff geometry is exact independently of counts: 10
staves / 50 rows on page 1, all 50 pixel-verified as ink rows
(row-band means 45–53 on a 0–255 scale).

## Checks

```sh
cargo test
cargo fmt --check
cargo clippy --all-targets
```
