#!/usr/bin/env bash
# run-olympic60.sh — score OLiMPiC-60 system outputs with the hardened
# evaluation triple: dual metrics, mean+median+usable-rate, hashes.
#
# The bundle ships engine outputs only (sheetly: 60 systems, audiveris: 50);
# ground truth is CC BY-SA and NOT redistributed — fetch OLiMPiC 1.0-Scanned
# (LINDAT hdl.handle.net/11234/1-5419) and point --gt-dir at it. Sample ids
# come from the bundle's samples.txt (see select.py for the rule).
#
# Per (engine, sample) the script stages hyp + GT (SHA-256 logged), scores
# strict-F1 via `eval-harness strict-xml` (whole system, no measure window)
# plus OMR-NED via pinned musicdiff, then aggregates per engine with
# `eval-harness suite-summary`. A sample with no hyp, or no GT to score
# against, is no-output: scored zero per benchmark rules, never dropped —
# so a GT-less run honestly reports usable-rate 0/60 instead of failing.
#
# Usage:
#   run-olympic60.sh --samples-file samples.txt --out DIR \
#     --hyp <engine>=<dir> [--hyp <engine>=<dir> ...] \
#     [--gt-dir DIR] [--harness-bin EXE] [--python PY] [--skip-omrned]
#
# File layout: GT is <gt-dir>/<sample-id>.musicxml (samples.txt lines
# verbatim, e.g. "score/system" nesting preserved, matching the dataset's
# samples/ tree); bundle hyp files flatten the id to "score-system".
set -euo pipefail

SAMPLES=""; OUT=""; GT_DIR=""; HARNESS_BIN=""; PYTHON="python3"; SKIP_OMRNED=0
HYPS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --samples-file) SAMPLES="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --gt-dir) GT_DIR="$2"; shift 2;;
    --harness-bin) HARNESS_BIN="$2"; shift 2;;
    --python) PYTHON="$2"; shift 2;;
    --hyp) HYPS="$HYPS $2"; shift 2;;
    --skip-omrned) SKIP_OMRNED=1; shift 1;;
    -h|--help) sed -n '1,35p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$SAMPLES" ] && [ -n "$OUT" ] && [ -n "$HYPS" ] || { echo "--samples-file/--out/--hyp required" >&2; exit 2; }

# shellcheck disable=SC1091
source "$(dirname "$0")/lib.sh"
LIB_PYTHON="$PYTHON"; export LIB_PYTHON
[ -z "$HARNESS_BIN" ] && HARNESS_BIN="$(lib_default_harness)"
[ -x "$HARNESS_BIN" ] || lib_die "harness binary not found: $HARNESS_BIN"
"$PYTHON" -c "import musicdiff" 2>/dev/null || { [ "$SKIP_OMRNED" = 1 ] || lib_die "musicdiff not importable with $PYTHON (see requirements.txt) or pass --skip-omrned"; }

mkdir -p "$OUT/reports" "$OUT/manifests" "$OUT/summaries"
: > "$OUT/staged_inputs.tsv"
: > "$OUT/pieces.tsv"

for spec in $HYPS; do
  engine="${spec%%=*}"; hypdir="${spec#*=}"
  [ -d "$hypdir" ] || lib_die "--hyp dir missing: $hypdir"
  while IFS= read -r sample; do
    case "$sample" in ''|\#*) continue;; esac
    # Bundle hyp files flatten the id ("score/system" -> "score-system").
    hyp="$hypdir/${sample//\//-}.musicxml"
    gt="${GT_DIR:-__absent__}/$sample.musicxml"
    rep="$OUT/reports/$engine-${sample//\//_}.json"
    if [ ! -f "$hyp" ]; then
      lib_log "$engine $sample: hyp missing -> no-output (zero, counted)"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$engine" "$sample" "NULL" "" "" "" >> "$OUT/pieces.tsv"
      continue
    fi
    if [ ! -f "$gt" ]; then
      hyp_sha="$(lib_sha256_file "$hyp")"
      printf '%s\t%s\t%s\n' "$hyp_sha" "hyp/$engine/$sample" "$hyp" >> "$OUT/staged_inputs.tsv"
      lib_log "$engine $sample: GT missing -> no-output (zero, counted)"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$engine" "$sample" "NULL" "" "" "$hyp_sha" >> "$OUT/pieces.tsv"
      continue
    fi
    hyp_sha="$(lib_sha256_file "$hyp")"; gt_sha="$(lib_sha256_file "$gt")"
    printf '%s\t%s\t%s\n' "$hyp_sha" "hyp/$engine/$sample" "$hyp" >> "$OUT/staged_inputs.tsv"
    printf '%s\t%s\t%s\n' "$gt_sha" "gt/$sample" "$gt" >> "$OUT/staged_inputs.tsv"
    ned=""
    if [ "$SKIP_OMRNED" = 0 ]; then
      ned="$(lib_musicdiff_ned "$gt" "$hyp")"
      [ -n "$ned" ] || lib_log "$engine $sample: musicdiff failed -> NED missing (F1 still counts)"
    fi
    if [ -n "$ned" ]; then
      "$HARNESS_BIN" strict-xml --reference "$gt" --hypothesis "$hyp" \
        --omr-ned "$ned" --report "$rep" >/dev/null
    else
      "$HARNESS_BIN" strict-xml --reference "$gt" --hypothesis "$hyp" \
        --report "$rep" >/dev/null
    fi
    f1="$("$PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1]))['strict_f1'])" "$rep")"
    lib_log "$engine $sample: strict-F1=$f1 omr-ned=${ned:-n/a}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$engine" "$sample" "$rep" "$ned" "$gt_sha" "$hyp_sha" >> "$OUT/pieces.tsv"
  done < "$SAMPLES"
done

"$PYTHON" - "$OUT/pieces.tsv" "$OUT" <<'PYEOF'
import csv, json, os, sys
rows = list(csv.DictReader(open(sys.argv[1]), fieldnames=[
    "engine", "slug", "report", "omr_ned", "ref_sha", "hyp_sha"], delimiter="\t"))
out = sys.argv[2]
by_engine = {}
for r in rows:
    by_engine.setdefault(r["engine"], []).append(r)
for engine, items in by_engine.items():
    pieces = []
    for r in items:
        if r["report"] == "NULL":
            pieces.append({"slug": r["slug"], "report": None, "no_output": True,
                           "reference_sha256": r["ref_sha"] or None,
                           "hypothesis_sha256": r["hyp_sha"] or None})
        else:
            pieces.append({"slug": r["slug"], "report": r["report"], "no_output": False,
                           "omr_ned": float(r["omr_ned"]) if r["omr_ned"] else None,
                           "reference_sha256": r["ref_sha"] or None,
                           "hypothesis_sha256": r["hyp_sha"] or None})
    mp = os.path.join(out, "manifests", engine + ".json")
    json.dump({"suite": "olympic60", "engine": engine, "pieces": pieces},
              open(mp, "w"), indent=1)
    print("wrote", mp)
PYEOF

for mf in "$OUT"/manifests/*.json; do
  [ -f "$mf" ] || continue
  eng="$(basename "$mf" .json)"
  "$HARNESS_BIN" suite-summary --manifest "$mf" --out "$OUT/summaries/$eng.json"
done
lib_log "done: summaries in $OUT/summaries"
