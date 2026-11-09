#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-${ROOT_DIR}/external/PhysCoop-base}"
BASE_URL="https://github.com/sodaball/H3-DDG.git"
BASE_COMMIT="6c7a9093909215a571f359e2c915031418b631aa"

if [[ ! -d "${TARGET_DIR}" ]]; then
  git clone "${BASE_URL}" "${TARGET_DIR}"
fi

if [[ "$(git -C "${TARGET_DIR}" rev-parse HEAD)" != "${BASE_COMMIT}" ]]; then
  printf 'Expected base commit %s, found %s in %s\n' \
    "${BASE_COMMIT}" "$(git -C "${TARGET_DIR}" rev-parse HEAD)" "${TARGET_DIR}" >&2
  printf 'Use a clean checkout of the pinned base commit before installing.\n' >&2
  exit 1
fi

cp -R "${ROOT_DIR}/physcoop" "${TARGET_DIR}/"
cp "${ROOT_DIR}/ddg_predictor.py" "${TARGET_DIR}/"
cp "${ROOT_DIR}/dataset.py" "${TARGET_DIR}/"
cp "${ROOT_DIR}/train.py" "${TARGET_DIR}/"
mkdir -p "${TARGET_DIR}/config/physcoop" "${TARGET_DIR}/scripts/physcoop"
mkdir -p "${TARGET_DIR}/tests/physcoop"
cp "${ROOT_DIR}"/configs/*.json "${TARGET_DIR}/config/physcoop/"
cp "${ROOT_DIR}"/scripts/*.py "${TARGET_DIR}/scripts/physcoop/"
cp "${ROOT_DIR}"/tests/*.py "${TARGET_DIR}/tests/physcoop/"

printf 'Installed PhysCoop-DDG into %s\n' "${TARGET_DIR}"
