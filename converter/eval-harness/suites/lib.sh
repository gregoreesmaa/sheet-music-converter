#!/usr/bin/env bash
# lib.sh — shared helpers for the eval-harness suite drivers.
# Source from run-*.sh:  source "$(dirname "$0")/lib.sh"
# Requires: bash 4+, python3 (stdlib only) for JSON plumbing.

lib_log() { echo "[$(date +%H:%M:%S)] $*"; }
lib_die() { echo "error: $*" >&2; exit 1; }
lib_need() { command -v "$1" >/dev/null 2>&1 || lib_die "missing required tool: $1"; }

# Default harness binary next to these scripts (sibling crate layout).
lib_default_harness() {
  local d
  d="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if [ -x "$d/target/release/eval-harness.exe" ]; then
    echo "$d/target/release/eval-harness.exe"
  else
    echo "$d/target/release/eval-harness"
  fi
}

# Resolve "<dir>/<base>" preferring the .exe variant on Windows.
lib_resolve_bin() { # <dir> <base>
  if [ -x "$1/$2.exe" ]; then echo "$1/$2.exe"; else echo "$1/$2"; fi
}

# SHA-256 hex of a file (coreutils, BSD, or python fallback).
# Strips GNU sha256sum's leading escape marker: when the path contains a
# backslash (Windows), the digest field comes out as "\HASH".
lib_sha256_file() { # <path>
  local digest
  if command -v sha256sum >/dev/null 2>&1; then
    digest="$(sha256sum "$1")"; digest="${digest%% *}"; digest="${digest#\\}"
  elif command -v shasum >/dev/null 2>&1; then
    digest="$(shasum -a 256 "$1")"; digest="${digest%% *}"
  else
    digest="$("$LIB_PYTHON" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1")"
  fi
  printf '%s' "$digest"
}

# OMR-NED via the pinned reference implementation (requirements.txt).
# Prints the value, or nothing when musicdiff fails. Never fails the caller.
lib_musicdiff_ned() { # <gt> <hyp>
  local raw ned
  raw="$("$LIB_PYTHON" -m musicdiff "$1" "$2" -o omrned 2>/dev/null)" || return 0
  ned="$(printf '%s' "$raw" | "$LIB_PYTHON" -c "
import json,sys
try:
    blob = sys.stdin.read()
    print(json.loads(blob[blob.find('{'):blob.rfind('}')+1])['OMR-NED'])
except Exception:
    pass
")"
  printf '%s' "$ned"
}

LIB_PYTHON="${LIB_PYTHON:-python3}"
