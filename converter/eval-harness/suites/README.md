# eval-harness suite drivers

Hardened evaluation drivers for the 31-piece (clean/photo) and OLiMPiC-60
suites. Every run carries the same triple:

1. **Dual metrics** — each scored pair reports strict-F1 (in-tree
   `strict-xml`, reference-exact port of the bundle `score.py`) **and**
   OMR-NED (reference `musicdiff`, pinned in
   [`requirements.txt`](../requirements.txt)) side by side.
2. **Suite summary** — `eval-harness suite-summary` aggregates per-piece
   reports into **mean + median + usable-output rate** per metric.
   No-output stays scored zero (strict-F1 0.0, OMR-NED 1.0 per the bundle
   `missing=1.0` convention) but is **counted** in every denominator.
   Medians follow the reference `score.py med()` (sorted middle, average
   of the two middles for even counts; 0.0 when empty), reported both
   over all pieces and over transcribed pieces only.
3. **Staged-input hashes** — every run logs SHA-256 of the inputs it
   staged: `strict-xml`/`strict-f1 --report` JSONs carry
   `reference_sha256` + `hypothesis_sha256`, and the drivers write
   `staged_inputs.tsv` (page PNGs, GT cuts, system hyps).

Scorer semantics are byte-identical to before hardening: the new fields
are attached by the CLI layer only (`score_pair`/`strict_f1` still return
them as `None`; pinned by unit test). The historic stdout one-liners are
unchanged — the dual-metric trailer (`omr-ned … sha256 …`) is appended
after the closing parenthesis so existing single-metric parsers keep
matching.

## Scripts (run under Git Bash on Windows)

| Script | What it does |
|---|---|
| `reproduce-published.sh` | Scorer gate: scores every published (GT, hyp) pair in the Sheetly benchmark bundle with our `strict-xml` vs the bundle's own `score.py`, joins published OMR-NED side by side, logs staged-input hashes. Exit 0 only on full agreement + full slot coverage. |
| `run-31.sh` | Our chain (release ingest → layout → staffmap → classical detect → voiced mapper → serialize) over staged page PNGs, both domains, then dual scoring + per-domain manifests + summaries. Chain failures are no-output (zero, counted), never fatal. |
| `run-olympic60.sh` | Scores `--hyp <engine>=<dir>` system outputs against fetched OLiMPiC GT (`--gt-dir`), then per-engine manifests + summaries. Missing hyp or GT is no-output (zero, counted). |
| `lib.sh` | Shared helpers (sourced, not run directly). |

Pinned inputs/versions per run: bundle files by SHA-256 (`hashes.tsv` /
`staged_inputs.tsv`), competitor versions in the bundle's
`COMPETITOR-RUNS.txt` (Audiveris 5.11.0, oemer 0.1.x, Sheetly 2026-10-30),
musicdiff in `requirements.txt`, harness version in each summary's
`scorer` field.

## Examples

```bash
# Gate: our scorer vs the reference on all published pairs (fast, stdlib).
bash suites/reproduce-published.sh \
  --bundle "$TEMP/sheetly/bundle/bundle" --out out/repro \
  --python D:/projects/sheet-music-converter/.venv/Scripts/python.exe

# Our chain on staged pages (needs release binaries + GT cuts + musicdiff).
bash suites/run-31.sh \
  --pages-clean "$TEMP/ours" --pages-photo "$TEMP/oursph" \
  --gt-dir "$TEMP/omrleg" --out out/ours-31 \
  --manifest "$TEMP/sheetly/bundle/bundle/manifest.json" \
  --python D:/projects/sheet-music-converter/.venv/Scripts/python.exe

# OLiMPiC-60 (GT fetched separately; without it the run honestly
# reports usable-rate 0/60).
bash suites/run-olympic60.sh \
  --samples-file "$BUNDLE/olimpic60/samples.txt" --out out/olimpic60 \
  --hyp sheetly="$BUNDLE/olimpic60/sheetly" \
  --hyp audiveris="$BUNDLE/olimpic60/audiveris" \
  --gt-dir "$OLIMPIC/samples" \
  --python D:/projects/sheet-music-converter/.venv/Scripts/python.exe
```

## Manifest schema (`suite-summary --manifest`)

```json
{
  "suite": "31-clean", "engine": "ours",
  "pieces": [
    {"slug": "repeats-melody", "report": "reports/clean-repeats-melody.json",
     "omr_ned": 0.9818, "reference_sha256": "..", "hypothesis_sha256": ".."},
    {"slug": "palestrina-agnus", "report": null, "no_output": true}
  ]
}
```

`report: null`, a missing file, or `"no_output": true` means no output
(zero, counted, unusable). A present-but-unparseable report is a hard
error, not a silent zero. Manifest `omr_ned` overrides the report's own
value (NED is slow; score it after F1). Usable pieces with no NED value
are excluded from the NED aggregates and counted as `n_missing`.

## Current reproduction status

`reproduce-published.sh` on the October 2026 bundle: clean **85/85**
pairs exact; photo **76/76** exact. The last photo miss
(`photo31/audiveris/luca-gloria`, true strict-F1 exactly 0.0625) was a
round-half divergence — Python `round()` (half-even) gives 0.062, Rust
`round()` (half-away) gave 0.063 — fixed by making the port's `round3`
round-half-to-even like the reference. Slot coverage: 186/186 (all 25
no-output slots match null published cells).
