#!/usr/bin/env bash
# Run the sheet-music converter pipeline (ingest -> layout -> symbols -> serialize).
# Usage: run-converter.sh [CONFIG] [EXTRA_ARGS...]
#   CONFIG defaults to converter/pipeline-orchestrator/config.example.toml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -gt 0 ]; then
  CONFIG="$1"
  shift
else
  CONFIG="$ROOT/converter/pipeline-orchestrator/config.example.toml"
fi

cargo run --release --manifest-path "$ROOT/converter/pipeline-orchestrator/Cargo.toml" \
  -p pipeline-orchestrator -- --config "$CONFIG" "$@"
