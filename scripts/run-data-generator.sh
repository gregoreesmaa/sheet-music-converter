#!/usr/bin/env bash
# Run the data-generation pipeline (render -> vary -> degrade -> pack).
# Usage: run-data-generator.sh [CONFIG] [EXTRA_ARGS...]
#   CONFIG defaults to data-gen/pipeline-orchestrator/config.example.toml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -gt 0 ]; then
  CONFIG="$1"
  shift
else
  CONFIG="$ROOT/data-gen/pipeline-orchestrator/config.example.toml"
fi

cargo run --release --manifest-path "$ROOT/data-gen/pipeline-orchestrator/Cargo.toml" \
  -p pipeline-orchestrator -- --config "$CONFIG" "$@"
