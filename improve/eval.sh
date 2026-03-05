#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/prepare_data.py"
python "$SCRIPT_DIR/optimize_prompt.py"
python "$SCRIPT_DIR/infer.py"

python - <<'PY'
import json
from pathlib import Path

results_path = Path("improve/results.json")
with results_path.open("r", encoding="utf-8") as f:
    results = json.load(f)

accuracy = results.get("accuracy", 0.0)
print(f"Final accuracy: {accuracy:.2f}")
PY
