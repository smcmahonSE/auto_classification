#!/usr/bin/env bash
set -euo pipefail

# Archive a specific run directory (or latest run) to a zip file.
# Usage:
#   bash scripts/archive_run.sh
#   bash scripts/archive_run.sh --run-id 20260226T235959Z
#   bash scripts/archive_run.sh --run-dir artifacts/model/runs/20260226T235959Z
#   bash scripts/archive_run.sh --include-cache
#   bash scripts/archive_run.sh --dest-dir "$HOME/model_backups/auto_classification"

RUN_ID=""
RUN_DIR=""
DEST_DIR="${HOME}/model_backups/auto_classification"
INCLUDE_CACHE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="${2:-}"
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
      sed -n '1,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_DIR}" ]]; then
  if [[ -n "${RUN_ID}" ]]; then
    RUN_DIR="artifacts/model/runs/${RUN_ID}"
  else
    if [[ ! -d "artifacts/model/runs" ]]; then
      echo "No runs directory found at artifacts/model/runs" >&2
      exit 1
    fi
    RUN_DIR="$(ls -dt artifacts/model/runs/* 2>/dev/null | head -n 1 || true)"
  fi
fi

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Run directory not found: ${RUN_DIR}" >&2
  exit 1
fi

RUN_BASENAME="$(basename "${RUN_DIR}")"
mkdir -p "${DEST_DIR}"
ZIP_PATH="${DEST_DIR}/${RUN_BASENAME}_share.zip"

ZIP_INPUTS=(
  "${RUN_DIR}"
  "artifacts/model/model_registry.jsonl"
  "artifacts/model/metrics_history.jsonl"
)

if [[ "${INCLUDE_CACHE}" == "true" ]]; then
  ZIP_INPUTS+=("artifacts/cache/embedding_cache.pkl")
fi

EXISTING_INPUTS=()
for path in "${ZIP_INPUTS[@]}"; do
  if [[ -e "${path}" ]]; then
    EXISTING_INPUTS+=("${path}")
  fi
done

if [[ ${#EXISTING_INPUTS[@]} -eq 0 ]]; then
  echo "Nothing to archive." >&2
  exit 1
fi

zip -r "${ZIP_PATH}" "${EXISTING_INPUTS[@]}"
echo "Created: ${ZIP_PATH}"
