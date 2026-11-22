#!/usr/bin/env bash
# Render the sample corpus across every available music-engraving x text
# font combo (MuseScore backend, release build).
#
# Usage (Git Bash on Windows — bare `bash` resolves to the WSL stub):
#   "C:\Program Files\Git\bin\bash.exe" scripts/run-font-matrix.sh [OUT_DIR]
#
# One command produces clean PNG + ground-truth JSON for each combo:
#   <stem>_<Music>_<Text>-<p>.png + <stem>_<Music>_<Text>.musicxml
#   + <stem>_<Music>_<Text>.json  (records "font" + "text_font")
#
# Verified font coverage on MuseScore 4.7.5 (see header comments below):
#   music (6): Bravura Emmentaler Gonville Leland MuseJazz Petaluma
#   text (11): Edwin, 4 shipped companions (Bravura/Leland/MuseJazz/
#     Petaluma Text), 6 system serifs/sans/mono/casual (Times New Roman,
#     Georgia, Palatino Linotype, Arial, Courier New, Comic Sans MS)
# Not covered (verified by byte-compare probes, same score, same build):
#   "Emmentaler Text" / "Gonville Text" do not ship as text families —
#     both render byte-identical to a bogus control font (fallback).
#   "FreeSerif" resolves in text-only styles but, paired with a music
#     font change, deterministically renders the paired default
#     (byte-identical to no-text runs) — excluded to avoid mislabeled
#     duplicates. Unknown engraving fonts (Edwin/November/Cadence/MScore
#     probed) collapse byte-identical to Bravura.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- sample pipeline config (edit here) ----
SAMPLES="$ROOT/data-gen/score-corpus/sample01.mscx $ROOT/data-gen/score-corpus/sample02.mscx $ROOT/data-gen/score-corpus/sample03.mscx $ROOT/data-gen/score-corpus/sample04.mscx"
OUT="${1:-$ROOT/data-gen/score-corpus/renders-fontmatrix}"
MUSIC_FONTS="Bravura,Emmentaler,Gonville,Leland,MuseJazz,Petaluma"
TEXT_FONTS="Edwin,Bravura Text,Leland Text,MuseJazz Text,Petaluma Text,Times New Roman,Georgia,Palatino Linotype,Arial,Courier New,Comic Sans MS"
DPI=300
# ---- end config ----

if tasklist.exe //FI "IMAGENAME eq MuseScore4.exe" 2>/dev/null | grep -q MuseScore4; then
  echo "refusing: a MuseScore4 process lingers (stale single-instance socket swallows launches); close it first" >&2
  exit 1
fi

cargo build --release --manifest-path "$ROOT/data-gen/score-render/Cargo.toml" -p score-render
BIN="$ROOT/data-gen/score-render/target/release/score-render"
[ -x "$BIN" ] || BIN="$BIN.exe"

mkdir -p "$OUT/_input"
for s in $SAMPLES; do cp -u "$s" "$OUT/_input/"; done

slug() { echo "$1" | tr -cd '[:alnum:]'; }

ok=0; skip=0; fail=0
OLD_IFS="$IFS"; IFS=','
# shellcheck disable=SC2162
for M in $MUSIC_FONTS; do for T in $TEXT_FONTS; do
  IFS="$OLD_IFS"
  M="$(echo "$M" | xargs)"; T="$(echo "$T" | xargs)"
  tag="$(slug "$M")_$(slug "$T")"
  done_one=0
  for src in "$OUT"/_input/*.mscx; do
    stem="$(basename "$src" .mscx)_$tag"
    js="$OUT/$stem.json"
    if [ -f "$js" ] && grep -q '"renderer": "musescore-batch"' "$js"; then
      pages_ok=1
      for p in $(grep -o '"pages": \[[^]]*\]' "$js" | grep -o '"[^"]*\.png"'); do
        p="$(echo "$p" | tr -d '"')"
        [ -f "$OUT/$p" ] || pages_ok=0
      done
      [ -f "$OUT/$stem.musicxml" ] || pages_ok=0
      if [ "$pages_ok" = 1 ]; then done_one=1; else done_one=0; break; fi
    else
      done_one=0; break
    fi
  done
  if [ "$done_one" = 1 ]; then echo "SKIP $tag (complete)"; skip=$((skip+1)); IFS=','; continue; fi
  echo "--- combo music=[$M] text=[$T]"
  if "$BIN" --renderer musescore --input "$OUT/_input" --out "$OUT" \
      --musescore-fonts "$M" --musescore-text-fonts "$T" --musescore-dpi "$DPI"; then
    ok=$((ok+1))
  else
    fail=$((fail+1))
  fi
  IFS=','
done done
IFS="$OLD_IFS"

echo "matrix done: $ok rendered, $skip skipped, $fail failed -> $OUT"
[ "$fail" = 0 ]
