#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export LIMIT="${LIMIT:-5}"
export DATASET_SIZE="${DATASET_SIZE:-5}"
export EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX:-smoke_limit5}"
export FORCE="${FORCE:-1}"
export RESUME="${RESUME:-0}"

exec bash scripts/run_cares_all_models.sh
