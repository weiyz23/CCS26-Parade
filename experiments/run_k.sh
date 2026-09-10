#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARADE_REPO="$(cd "$EXPERIMENT_DIR/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
if [[ -d "$PARADE_REPO/.conda/bin" ]]; then
    export PATH="$PARADE_REPO/.conda/bin:$PATH"
fi
exec python "$EXPERIMENT_DIR/scripts/run_experiment.py" k "$@"
