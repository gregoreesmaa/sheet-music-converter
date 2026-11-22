#!/usr/bin/env bash
# run-31.sh — our full chain over the 31-piece suite, both domains, with the
# hardened evaluation triple: dual metrics, mean+median+usable-rate, hashes.
#
# Per (domain, piece) the script stages the page PNG (SHA-256 logged), runs
# the release chain ingest -> layout -> staffmap -> classical detect ->
# voiced mapper -> serialize (a chain failure is no-output: scored zero per
# benchmark rules, never dropped), scores strict-F1 via `eval-harness
# strict-xml` against the page-one GT cut, scores OMR-NED via the pinned
# musicdiff (requirements.txt), then aggregates both domains with
# `eval-harness suite-summary`.
#
# Usage:
#   run-31.sh --pages-clean DIR --pages-photo DIR --gt-dir DIR --out DIR \
#     --manifest BUNDLE_MANIFEST.json \
#     [--bin-dir DIR] [--python PY] [--pieces "a b c"] \
#     [--domains "clean photo"] [--page-pattern 'cr-{slug}/page.png'] \
#     [--skip-omrned]
#
# Inputs (nothing is redistributed; stage them first):
#   pages-clean/photo: <root>/cr-<slug>/page.png per piece (override name
#     shape with --page-pattern, {slug} substituted).
#   gt-dir: gt-p1-<slug>.musicxml page-one GT cuts (bundle omr_ned.py
#     cut_ground_truth, or the %TEMP%/omrleg set).
#   manifest: benchmark bundle manifest.json (page-one measure counts).
#
# Outputs under --out:
#   work/<domain>/<slug>/...        chain work dirs (clean.png, hyp/...).
#   reports/<domain>-<slug>.json    per-piece strict-xml reports (+hashes).
#   staged_inputs.tsv               SHA-256 of every staged page PNG + GT.
#   manifests/<domain>.json         suite manifests (piece list + NED join).
#   summaries/<domain>.json         mean/median/usable-rate, dual metrics.
set -euo pipefail

PAGES_CLEAN=""; PAGES_PHOTO=""; GT_DIR=""; OUT=""; MANIFEST=""
BIN_DIR=""; PYTHON="python3"; PIECES=""; DOMAINS="clean photo"
PAGE_PATTERN="cr-{slug}/page.png"; SKIP_OMRNED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --pages-clean) PAGES_CLEAN="$2"; shift 2;;
    --pages-photo) PAGES_PHOTO="$2"; shift 2;;
    --gt-dir) GT_DIR="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --manifest) MANIFEST="$2"; shift 2;;
    --bin-dir) BIN_DIR="$2"; shift 2;;
    --python) PYTHON="$2"; shift 2;;
    --pieces) PIECES="$2"; shift 2;;
    --domains) DOMAINS="$2"; shift 2;;
    --page-pattern) PAGE_PATTERN="$2"; shift 2;;
    --skip-omrned) SKIP_OMRNED=1; shift 1;;
    -h|--help) sed -n '1,40p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$PAGES_CLEAN" ] && [ -n "$PAGES_PHOTO" ] && [ -n "$GT_DIR" ] \
  && [ -n "$OUT" ] && [ -n "$MANIFEST" ] || { echo "--pages-clean/--pages-photo/--gt-dir/--out/--manifest required" >&2; exit 2; }

# shellcheck disable=SC1091
source "$(dirname "$0")/lib.sh"
LIB_PYTHON="$PYTHON"; export LIB_PYTHON
[ -z "$BIN_DIR" ] && BIN_DIR="$(cd "$(dirname "$0")/../../" && pwd)"
INGEST="$(lib_resolve_bin "$BIN_DIR/ingest-cleanup/target/release" ingest-cleanup)"
LAYOUT="$(lib_resolve_bin "$BIN_DIR/layout-staff/target/release" layout-staff)"
HARNESS="$(lib_resolve_bin "$BIN_DIR/eval-harness/target/release" eval-harness)"
DETECT="$(lib_resolve_bin "$BIN_DIR/symbol-detect/target/release" symbol-detect)"
MAPPER="$(lib_resolve_bin "$BIN_DIR/boxnote-mapper/target/release" boxnote-mapper)"
SERIALIZE="$(lib_resolve_bin "$BIN_DIR/assembly-serialize/target/release" assembly-serialize)"
for b in "$INGEST" "$LAYOUT" "$HARNESS" "$DETECT" "$MAPPER" "$SERIALIZE"; do
  [ -x "$b" ] || lib_die "binary not found: $b (build release first)"
done
export HARNESS
"$PYTHON" -c "import musicdiff" 2>/dev/null || { [ "$SKIP_OMRNED" = 1 ] || lib_die "musicdiff not importable with $PYTHON (see requirements.txt) or pass --skip-omrned"; }

mkdir -p "$OUT/work" "$OUT/reports" "$OUT/manifests" "$OUT/summaries"

# slug <tab> p1-measures for the requested pieces (manifest = truth).
MEASURES_TSV="$OUT/measures.tsv"
"$PYTHON" - "$MANIFEST" "$PIECES" "$MEASURES_TSV" <<'PYEOF'
import json, sys
manifest = json.load(open(sys.argv[1]))
wanted = set(sys.argv[2].split()) if len(sys.argv) > 2 and sys.argv[2] else None
# newline="": this TSV is consumed by bash, which chokes on \r.
with open(sys.argv[3], "w", encoding="utf-8", newline="") as f:
    # The 31 slugs are shared by both suites; page-one measures come
    # from the clean31 entries.
    for piece in manifest["suites"]["clean31"]["pieces"]:
        if wanted is not None and piece["slug"] not in wanted:
            continue
        f.write(f"{piece['slug']}\t{piece['pages'][0]['measures']}\n")
PYEOF

: > "$OUT/staged_inputs.tsv"
: > "$OUT/pieces.tsv"

run_chain() { # <page-png> <workdir> — full release chain; nonzero on failure.
  local page="$1" work="$2" lay
  mkdir -p "$work" "$work/crops"
  "$INGEST" --input "$page" --output "$work/clean.png" >/dev/null || return 1
  "$LAYOUT" --input "$work/clean.png" --out-dir "$work" >/dev/null || return 1
  lay="$(echo "$work"/*_layout.json)" ; [ -f "$lay" ] || return 1
  "$HARNESS" staffmap --layout "$lay" --out "$work/staffmap.json" >/dev/null || return 1
  cp -f "$work/clean.png" "$work/crops/page.png" || return 1
  "$DETECT" --crops-dir "$work/crops" --staff-map "$work/staffmap.json" \
    --out "$work/glyphs.json" --backend classical >/dev/null || return 1
  "$MAPPER" --glyphs "$work/glyphs.json" --layout "$lay" \
    --out "$work/notes.json" --backend voiced >/dev/null || return 1
  "$SERIALIZE" serialize --input "$work/notes.json" --out "$work/hyp" --stem hyp >/dev/null || return 1
  [ -f "$work/hyp/hyp.musicxml" ] || return 1
  return 0
}

while IFS=$'\t' read -r slug meas; do
  [ -n "$slug" ] || continue
  for domain in $DOMAINS; do
    if [ "$domain" = clean ]; then root="$PAGES_CLEAN"; else root="$PAGES_PHOTO"; fi
    page="$root/${PAGE_PATTERN//\{slug\}/$slug}"
    work="$OUT/work/$domain/$slug"
    gt="$GT_DIR/gt-p1-$slug.musicxml"
    rep="$OUT/reports/$domain-$slug.json"
    if [ ! -f "$page" ]; then
      lib_log "$domain $slug: page missing -> no-output (zero, counted)"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$domain" "$slug" "NULL" "" "" "" >> "$OUT/pieces.tsv"
      continue
    fi
    if [ ! -f "$gt" ]; then
      lib_log "$domain $slug: GT cut missing -> no-output (zero, counted)"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$domain" "$slug" "NULL" "" "" "" >> "$OUT/pieces.tsv"
      continue
    fi
    page_sha="$(lib_sha256_file "$page")"; gt_sha="$(lib_sha256_file "$gt")"
    printf '%s\t%s\t%s\n' "$page_sha" "page/$domain/$slug" "$page" >> "$OUT/staged_inputs.tsv"
    printf '%s\t%s\t%s\n' "$gt_sha" "gt/gt-p1-$slug.musicxml" "$gt" >> "$OUT/staged_inputs.tsv"
    if run_chain "$page" "$work"; then
      hyp="$work/hyp/hyp.musicxml"
      ned=""
      if [ "$SKIP_OMRNED" = 0 ]; then
        ned="$(lib_musicdiff_ned "$gt" "$hyp")"
        [ -n "$ned" ] || lib_log "$domain $slug: musicdiff failed -> NED missing (F1 still counts)"
      fi
      if [ -n "$ned" ]; then
        "$HARNESS" strict-xml --reference "$gt" --hypothesis "$hyp" \
          --max-measures "$meas" --omr-ned "$ned" --report "$rep" >/dev/null
      else
        "$HARNESS" strict-xml --reference "$gt" --hypothesis "$hyp" \
          --max-measures "$meas" --report "$rep" >/dev/null
      fi
      f1="$("$PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1]))['strict_f1'])" "$rep")"
      lib_log "$domain $slug: strict-F1=$f1 omr-ned=${ned:-n/a}"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$domain" "$slug" "$rep" "$ned" "$gt_sha" \
        "$(lib_sha256_file "$hyp")" >> "$OUT/pieces.tsv"
    else
      lib_log "$domain $slug: CHAIN-FAIL -> no-output (zero, counted)"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$domain" "$slug" "NULL" "" "$gt_sha" "" >> "$OUT/pieces.tsv"
    fi
  done
done < "$MEASURES_TSV"

# Manifests + suite summaries per domain (mean/median/usable-rate, dual).
"$PYTHON" - "$OUT/pieces.tsv" "$OUT" <<'PYEOF'
import csv, json, os, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), fieldnames=[
    "domain", "slug", "report", "omr_ned", "ref_sha", "hyp_sha"], delimiter="\t"))
out = sys.argv[2]
by_domain = {}
for r in rows:
    by_domain.setdefault(r["domain"], []).append(r)
for domain, items in by_domain.items():
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
    man = {"suite": f"31-{domain}", "engine": "ours", "pieces": pieces}
    mp = os.path.join(out, "manifests", domain + ".json")
    json.dump(man, open(mp, "w"), indent=1)
    print("wrote", mp)
PYEOF

for domain in $DOMAINS; do
  [ -f "$OUT/manifests/$domain.json" ] || continue
  "$HARNESS" suite-summary --manifest "$OUT/manifests/$domain.json" \
    --out "$OUT/summaries/$domain.json"
done
lib_log "done: summaries in $OUT/summaries"
