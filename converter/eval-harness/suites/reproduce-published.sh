#!/usr/bin/env bash
# reproduce-published.sh — validate `eval-harness strict-xml` against every
# published (ground-truth, hypothesis) pair in the Sheetly OMR benchmark
# bundle, with the bundle's own reference `score.py` as oracle.
#
# What it checks, per suite (clean31 / photo31) and per engine:
#   * pair agreement: our strict-F1 == reference strict-F1 on every pair
#     file present in the bundle (clean: 85, photo: 76 p1 files);
#   * slot coverage: every one of the 93 (31 pieces x 3 engines) slots is
#     accounted for — a missing hypothesis is no-output, scored zero per
#     benchmark rules, and must match a null published cell;
#   * dual metrics: each row carries strict-F1 (live, ours) side by side
#     with OMR-NED (bundle `omrned-results.json`, reference musicdiff run);
#   * staged-input hashes: SHA-256 of every GT/hyp file staged, in
#     `hashes.tsv`.
#
# Usage (Git Bash on Windows, or any bash + python3):
#   reproduce-published.sh --bundle <bundle-root> --out <dir> \
#     [--harness-bin <eval-harness.exe>] [--python <python.exe>]
#
# Exit 0 when every present pair agrees and every slot is accounted for,
# 1 otherwise. Scorer semantics live in strict_xml.rs; this script only
# compares outputs, so it runs unchanged before and after harness edits.
set -euo pipefail

BUNDLE=""; OUT=""; HARNESS_BIN=""; PYTHON="python3"
while [ $# -gt 0 ]; do
  case "$1" in
    --bundle) BUNDLE="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --harness-bin) HARNESS_BIN="$2"; shift 2;;
    --python) PYTHON="$2"; shift 2;;
    -h|--help) sed -n '1,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$BUNDLE" ] && [ -n "$OUT" ] || { echo "--bundle and --out are required" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -z "$HARNESS_BIN" ]; then
  if [ -x "$SCRIPT_DIR/../target/release/eval-harness.exe" ]; then
    HARNESS_BIN="$SCRIPT_DIR/../target/release/eval-harness.exe"
  else
    HARNESS_BIN="$SCRIPT_DIR/../target/release/eval-harness"
  fi
fi
[ -x "$HARNESS_BIN" ] || { echo "harness binary not found: $HARNESS_BIN" >&2; exit 2; }
mkdir -p "$OUT"

export REPRO_BUNDLE="$BUNDLE" REPRO_OUT="$OUT" REPRO_HARNESS="$HARNESS_BIN"
"$PYTHON" - "$BUNDLE" <<'PYEOF'
import csv, hashlib, json, os, subprocess, sys

bundle = sys.argv[1]
out = os.environ["REPRO_OUT"]
harness = os.environ["REPRO_HARNESS"]
sys.path.insert(0, bundle)
import score as ref

manifest = json.load(open(os.path.join(bundle, "manifest.json")))
published = json.load(open(os.path.join(bundle, "benchmark-results.json")))
try:
    omrned = json.load(open(os.path.join(bundle, "omrned-results.json")))
except FileNotFoundError:
    omrned = {}

ENGINES = ("sheetly", "audiveris", "oemer")
SUITES = ("clean31", "photo31")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

os.makedirs(out, exist_ok=True)
hash_rows = []
pair_rows = []
slot_total = 0
slot_ok = 0
pair_total = {"clean31": 0, "photo31": 0}
pair_agree = {"clean31": 0, "photo31": 0}
mismatches = []

pub_by_suite = {s: {r["slug"]: r for r in published["suites"][s]} for s in SUITES}

for suite in SUITES:
    for piece in manifest["suites"][suite]["pieces"]:
        slug = piece["slug"]
        p1meas = piece["pages"][0]["measures"]
        gt_path = os.path.join(bundle, suite, "ground-truth", slug + ".musicxml")
        gt_sha = sha256(gt_path)
        hash_rows.append((sha256(gt_path), suite + "/ground-truth/" + slug + ".musicxml"))
        gt_xml = open(gt_path, encoding="utf-8", errors="ignore").read()
        gt_events = ref.note_events(gt_xml, max_measures=p1meas)
        for eng in ENGINES:
            slot_total += 1
            hyp_path = os.path.join(bundle, suite, eng, slug, "p1.musicxml")
            pub_val = pub_by_suite[suite][slug].get(eng)
            if not os.path.exists(hyp_path):
                # No-output slot: scores zero per benchmark rules; the
                # published cell must be null (nothing to reproduce).
                if pub_val is None:
                    slot_ok += 1
                    pair_rows.append({
                        "suite": suite, "engine": eng, "slug": slug,
                        "present": 0, "ref_f1": "", "our_f1": 0.0,
                        "agree": 1, "published": "", "omr_ned": 1.0,
                        "gt_sha256": gt_sha, "hyp_sha256": "",
                    })
                else:
                    mismatches.append(f"{suite}/{eng}/{slug}: hyp missing but published={pub_val}")
                continue
            hyp_sha = sha256(hyp_path)
            hash_rows.append((hyp_sha, suite + "/" + eng + "/" + slug + "/p1.musicxml"))
            hyp_xml = open(hyp_path, encoding="utf-8", errors="ignore").read()
            expected = ref.score_pair(gt_events, ref.note_events(hyp_xml))["strict_f1"]
            rep_path = os.path.join(out, f"{suite}-{eng}-{slug}.json")
            subprocess.run([harness, "strict-xml", "--reference", gt_path,
                            "--hypothesis", hyp_path, "--max-measures", str(p1meas),
                            "--report", rep_path],
                           check=True, capture_output=True)
            ours = json.load(open(rep_path))["strict_f1"]
            try:
                ned = omrned[suite][eng][slug].get("OMR-NED")
            except KeyError:
                ned = ""
            agree = 1 if ours == expected else 0
            pair_total[suite] += 1
            pair_agree[suite] += agree
            if agree:
                slot_ok += 1
            else:
                mismatches.append(
                    f"{suite}/{eng}/{slug}: ref={expected} ours={ours} published={pub_val}")
            if pub_val is not None and abs(pub_val - expected) > 1e-9:
                mismatches.append(
                    f"{suite}/{eng}/{slug}: reference rerun {expected} != published {pub_val}")
            pair_rows.append({
                "suite": suite, "engine": eng, "slug": slug,
                "present": 1, "ref_f1": expected, "our_f1": ours,
                "agree": agree, "published": pub_val if pub_val is not None else "",
                "omr_ned": ned if ned != "" else "",
                "gt_sha256": gt_sha, "hyp_sha256": hyp_sha,
            })

with open(os.path.join(out, "pairs.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["suite", "engine", "slug", "present",
                                      "ref_f1", "our_f1", "agree", "published",
                                      "omr_ned", "gt_sha256", "hyp_sha256"])
    w.writeheader()
    w.writerows(pair_rows)

with open(os.path.join(out, "hashes.tsv"), "w", encoding="utf-8") as f:
    for digest, name in hash_rows:
        f.write(f"{digest}  {name}\n")

print(f"clean31 pairs: {pair_agree['clean31']}/{pair_total['clean31']} agree")
print(f"photo31 pairs: {pair_agree['photo31']}/{pair_total['photo31']} agree")
print(f"slots accounted: {slot_ok}/{slot_total} (missing hyp == null published cell)")
if mismatches:
    print("MISMATCHES:")
    for m in mismatches:
        print("  " + m)
    sys.exit(1)
print("REPRODUCTION GREEN")
PYEOF
