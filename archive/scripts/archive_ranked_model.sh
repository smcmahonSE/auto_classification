#!/usr/bin/env bash
set -euo pipefail

# Archive a chosen ranked model from registry metrics.
# Usage:
#   bash scripts/archive_ranked_model.sh
#   bash scripts/archive_ranked_model.sh --sort-by scores.macro_f1 --rank 1
#   bash scripts/archive_ranked_model.sh --sort-by scores.accuracy --rank 2 --ascending

SORT_BY="scores.macro_f1"
RANK="1"
ASCENDING="false"
REGISTRY_PATH="artifacts/model/model_registry.jsonl"
DEST_DIR="${HOME}/model_backups/auto_classification"
INCLUDE_CACHE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sort-by)
      SORT_BY="${2:-}"
      shift 2
      ;;
    --rank)
      RANK="${2:-}"
      shift 2
      ;;
    --ascending)
      ASCENDING="true"
      shift
      ;;
    --registry-path)
      REGISTRY_PATH="${2:-}"
      shift 2
      ;;
    --dest-dir)
      DEST_DIR="${2:-}"
      shift 2
      ;;
    --include-cache)
      INCLUDE_CACHE="true"
      shift
      ;;
    -h|--help)
      sed -n '1,15p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${REGISTRY_PATH}" ]]; then
  echo "Registry not found: ${REGISTRY_PATH}" >&2
  exit 1
fi

RUN_DIR="$(
python - "${REGISTRY_PATH}" "${SORT_BY}" "${RANK}" "${ASCENDING}" <<'PY'
import json
import sys
from pathlib import Path

registry_path = Path(sys.argv[1])
sort_by = sys.argv[2]
rank = int(sys.argv[3])
ascending = sys.argv[4].lower() == "true"

if rank < 1:
    raise SystemExit("rank must be >= 1")

rows = []
for line in registry_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    rows.append(json.loads(line))

if not rows:
    raise SystemExit("registry has no records")

def nested_get(d, key_path):
    cur = d
    for part in key_path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur

sorted_rows = sorted(rows, key=lambda r: nested_get(r, sort_by), reverse=not ascending)
idx = rank - 1
if idx >= len(sorted_rows):
    raise SystemExit(f"rank {rank} out of range (records={len(sorted_rows)})")

best = sorted_rows[idx]
model_path = best.get("model_artifact_path")
if not model_path:
    raise SystemExit("selected record missing model_artifact_path")

print(str(Path(model_path).parent))
PY
)"

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Selected run directory missing: ${RUN_DIR}" >&2
  exit 1
fi

ARCHIVE_ARGS=(bash scripts/archive_run.sh --run-dir "${RUN_DIR}" --dest-dir "${DEST_DIR}")
if [[ "${INCLUDE_CACHE}" == "true" ]]; then
  ARCHIVE_ARGS+=(--include-cache)
fi

"${ARCHIVE_ARGS[@]}"
