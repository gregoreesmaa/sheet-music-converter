"""Stage a permanent SHA-256-verified in-tree copy of the 31-piece Sheetly benchmark.

Task T2.0 (todo/phase-2-hybrid-dewarp.md): the benchmark inputs and page-1
ground-truth cuts live under %TEMP% (ours, oursph, omrleg, sheetly bundle),
where a Windows temp cleanup would erase them. This script copies them into
data-gen/score-corpus/suite31/ and records SHA-256 hashes of all 93 files
(31 clean + 31 photo + 31 gt) in suite31_manifest.json.

Read-only on %TEMP% sources: this script only ever opens source files with
'rb'. It never writes, moves, renames, or deletes anything outside the
in-tree destination directory.

Source layout (see converter/eval-harness/suites/run-31.sh --page-pattern
'cr-{slug}/page.png' and converter/EVAL-ORACLE-BASELINE.md):
  %TEMP%/ours/cr-<slug>/page.png            -> suite31/clean/<slug>.png
  %TEMP%/oursph/cr-<slug>/page.png|.jpg     -> suite31/photo/<slug>.<same ext>
  %TEMP%/omrleg/gt-p1-<slug>.musicxml       -> suite31/gt/gt-p1-<slug>.musicxml
  %TEMP%/sheetly/bundle/bundle/manifest.json (+ benchmark-results.json)
                                            -> competitor reference scores
                                               embedded per piece in the manifest.

Usage:
  .\\.venv\\Scripts\\python.exe tmp-train/mvd/stage_suite31.py [--check-only]

Exit 0: all 31 pieces staged, every file present, every hash matches on
re-read. Exit 1: usage/config error. Exit 2: any verification failure
(missing source, missing staged file, hash mismatch) -- fails loud with
the exact slugs/files listed.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile

EXPECTED_PIECES = 31
PHOTO_CANDIDATES = ("page.png", "page.jpg", "page.jpeg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEST_ROOT = os.path.join(REPO_ROOT, "data-gen", "score-corpus", "suite31")
MANIFEST_PATH = os.path.join(DEST_ROOT, "suite31_manifest.json")

TMP = tempfile.gettempdir()
SRC_CLEAN = os.path.join(TMP, "ours")
SRC_PHOTO = os.path.join(TMP, "oursph")
SRC_GT = os.path.join(TMP, "omrleg")
SRC_BUNDLE = os.path.join(TMP, "sheetly", "bundle", "bundle")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(errors):
    for e in errors:
        print("stage_suite31 ERROR: " + e, file=sys.stderr)
    return 2


def load_bundle_truth(errors):
    """Slug list + p1 measures come from the bundle manifest (clean31)."""
    manifest_path = os.path.join(SRC_BUNDLE, "manifest.json")
    if not os.path.isfile(manifest_path):
        errors.append("bundle manifest missing: %s" % manifest_path)
        return None, None
    try:
        manifest = json.load(open(manifest_path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        errors.append("bundle manifest unreadable %s: %s" % (manifest_path, e))
        return None, None
    try:
        clean_pieces = manifest["suites"]["clean31"]["pieces"]
        photo_pieces = manifest["suites"]["photo31"]["pieces"]
    except (KeyError, TypeError) as e:
        errors.append("bundle manifest has no suites.clean31/photo31.pieces: %s" % e)
        return None, None
    clean_slugs = [p["slug"] for p in clean_pieces]
    photo_slugs = [p["slug"] for p in photo_pieces]
    if set(clean_slugs) != set(photo_slugs):
        errors.append(
            "clean31/photo31 slug sets differ: clean-only=%s photo-only=%s"
            % (sorted(set(clean_slugs) - set(photo_slugs)),
               sorted(set(photo_slugs) - set(clean_slugs)))
        )
        return None, None
    if len(clean_slugs) != EXPECTED_PIECES:
        errors.append(
            "bundle clean31 has %d pieces, expected %d" % (len(clean_slugs), EXPECTED_PIECES)
        )
        return None, None
    measures = {p["slug"]: p["pages"][0]["measures"] for p in clean_pieces}
    return clean_slugs, measures


def load_reference_scores(errors):
    """Per-piece competitor scores from benchmark-results.json (may be null = no-output)."""
    bench_path = os.path.join(SRC_BUNDLE, "benchmark-results.json")
    if not os.path.isfile(bench_path):
        errors.append("benchmark-results.json missing: %s" % bench_path)
        return None
    try:
        bench = json.load(open(bench_path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        errors.append("benchmark-results.json unreadable %s: %s" % (bench_path, e))
        return None
    ref = {}
    for suite in ("clean31", "photo31"):
        try:
            entries = bench["suites"][suite]
        except (KeyError, TypeError) as e:
            errors.append("benchmark-results.json has no suites.%s: %s" % (suite, e))
            return None
        for entry in entries:
            ref.setdefault(entry["slug"], {})[suite.replace("31", "")] = entry
    return ref


def find_source(kind, slug, errors):
    if kind == "clean":
        cand = [os.path.join(SRC_CLEAN, "cr-" + slug, "page.png")]
    elif kind == "photo":
        cand = [os.path.join(SRC_PHOTO, "cr-" + slug, name) for name in PHOTO_CANDIDATES]
    else:
        cand = [os.path.join(SRC_GT, "gt-p1-%s.musicxml" % slug)]
    for path in cand:
        if os.path.isfile(path):
            return path
    errors.append("%s source missing for slug '%s' (tried: %s)" % (kind, slug, cand))
    return None


def atomic_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    with open(src, "rb") as fsrc, open(tmp, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, 1 << 20)
    os.replace(tmp, dst)


def main(argv):
    check_only = "--check-only" in argv[1:]
    errors = []

    slugs, measures = load_bundle_truth(errors)
    ref = load_reference_scores(errors)
    if errors:
        return fail(errors)

    pieces = []
    for slug in slugs:
        src_clean = find_source("clean", slug, errors)
        src_photo = find_source("photo", slug, errors)
        src_gt = find_source("gt", slug, errors)
        if src_clean is None or src_photo is None or src_gt is None:
            continue
        for domain in ("clean", "photo"):
            if domain not in ref.get(slug, {}):
                errors.append("reference scores missing for slug '%s' domain '%s'" % (slug, domain))
        photo_ext = os.path.splitext(src_photo)[1].lower() or ".png"
        dest_clean = os.path.join(DEST_ROOT, "clean", slug + ".png")
        dest_photo = os.path.join(DEST_ROOT, "photo", slug + photo_ext)
        dest_gt = os.path.join(DEST_ROOT, "gt", "gt-p1-%s.musicxml" % slug)

        if not check_only:
            for src, dst in ((src_clean, dest_clean), (src_photo, dest_photo), (src_gt, dest_gt)):
                src_hash = sha256_file(src)
                if os.path.isfile(dst) and sha256_file(dst) == src_hash:
                    continue  # already staged, byte-identical
                atomic_copy(src, dst)

        # Verify on re-read: destination must exist and match the source hash.
        staged = {}
        for label, src, dst in (("clean", src_clean, dest_clean),
                                ("photo", src_photo, dest_photo),
                                ("gt", src_gt, dest_gt)):
            if not os.path.isfile(dst):
                errors.append("staged file missing: %s (slug '%s')" % (dst, slug))
                continue
            want = sha256_file(src)
            got = sha256_file(dst)
            if want != got:
                errors.append("hash mismatch for slug '%s' %s: src %s != staged %s"
                              % (slug, label, want, got))
                continue
            staged[label] = {
                "file": os.path.relpath(dst, DEST_ROOT).replace(os.sep, "/"),
                "sha256": got,
                "bytes": os.path.getsize(dst),
            }
        if len(staged) != 3:
            continue
        pieces.append({
            "slug": slug,
            "p1_measures": measures[slug],
            "clean": staged["clean"],
            "photo": staged["photo"],
            "gt": staged["gt"],
            "reference": ref[slug],
        })

    if len(pieces) != EXPECTED_PIECES:
        staged_slugs = sorted(p["slug"] for p in pieces)
        errors.append("staged %d/%d pieces; missing: %s"
                      % (len(pieces), EXPECTED_PIECES,
                         sorted(set(slugs) - set(staged_slugs)) if slugs else "?"))

    manifest = {
        "suite": "suite31",
        "generated_by": "tmp-train/mvd/stage_suite31.py",
        "pieces_expected": EXPECTED_PIECES,
        "sources": {
            "clean_root": SRC_CLEAN,
            "photo_root": SRC_PHOTO,
            "gt_root": SRC_GT,
            "bundle_root": SRC_BUNDLE,
        },
        "bundle_files": {
            name: {"file": name, "sha256": sha256_file(os.path.join(SRC_BUNDLE, name))}
            for name in ("manifest.json", "benchmark-results.json")
            if os.path.isfile(os.path.join(SRC_BUNDLE, name))
        },
        "pieces": pieces,
    }

    if errors:
        return fail(errors)

    # Manifest must carry exactly 31 pieces (verification asserts len == 31).
    if len(manifest["pieces"]) != EXPECTED_PIECES:
        return fail(["manifest holds %d pieces, expected %d"
                     % (len(manifest["pieces"]), EXPECTED_PIECES)])

    if not check_only:
        os.makedirs(DEST_ROOT, exist_ok=True)
        tmp = MANIFEST_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, sort_keys=False)
            f.write("\n")
        os.replace(tmp, MANIFEST_PATH)
        # Re-read the written manifest and re-verify every staged hash from disk.
        reread = json.load(open(MANIFEST_PATH, encoding="utf-8"))
        if len(reread["pieces"]) != EXPECTED_PIECES:
            return fail(["re-read manifest holds %d pieces" % len(reread["pieces"])])
        for p in reread["pieces"]:
            for key in ("clean", "photo", "gt"):
                fp = os.path.join(DEST_ROOT, p[key]["file"].replace("/", os.sep))
                if sha256_file(fp) != p[key]["sha256"]:
                    return fail(["post-write hash mismatch: %s" % fp])

    print("stage_suite31: %d pieces staged (31 clean + 31 photo + 31 gt = 93 files), "
          "all SHA-256 verified on re-read." % len(pieces))
    print("manifest: %s" % MANIFEST_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
