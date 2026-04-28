#!/bin/bash
set -euo pipefail

fail() {
  echo "stage_simlingo_to_scratch.sh: $*" >&2
  return 1 2>/dev/null || exit 1
}

if [ -z "${REPO_DIR:-}" ]; then
  fail "source thesis/env.sh before sourcing this script."
fi

manifest_path="${SIMLINGO_ARCHIVE_ROOT}/dataset_archives.txt"
if [ ! -s "${manifest_path}" ]; then
  fail "missing archive manifest ${manifest_path}"
fi

mapfile -t dataset_archives < "${manifest_path}"
if [ "${#dataset_archives[@]}" -eq 0 ]; then
  fail "archive manifest ${manifest_path} is empty"
fi

dataset_archive_paths=()
for archive_name in "${dataset_archives[@]}"; do
  if [ ! -f "${SIMLINGO_ARCHIVE_ROOT}/${archive_name}" ]; then
    fail "missing archive ${SIMLINGO_ARCHIVE_ROOT}/${archive_name}"
  fi
  dataset_archive_paths+=("${SIMLINGO_ARCHIVE_ROOT}/${archive_name}")
done

if [ ! -f "${SIMLINGO_ARCHIVE_ROOT}/bucketsv2_simlingo.tar" ]; then
  fail "missing archive ${SIMLINGO_ARCHIVE_ROOT}/bucketsv2_simlingo.tar"
fi

parent_dirs_path="${SIMLINGO_ARCHIVE_ROOT}/stage_parent_dirs.txt"
if [ ! -s "${parent_dirs_path}" ]; then
  fail "missing parent directory manifest ${parent_dirs_path}; rerun thesis/scripts/pack_simlingo_dataset.sh"
fi

metadata_path="${SIMLINGO_ARCHIVE_ROOT}/dataset_metadata.env"
if [ -f "${metadata_path}" ]; then
  source "${metadata_path}"
fi

if [ -n "${SIMLINGO_ROUTE_COUNT:-}" ]; then
  if ! [[ "${SIMLINGO_ROUTE_COUNT}" =~ ^[0-9]+$ ]] || [ "${SIMLINGO_ROUTE_COUNT}" -lt 1 ]; then
    fail "invalid SIMLINGO_ROUTE_COUNT in ${metadata_path}: ${SIMLINGO_ROUTE_COUNT}"
  fi
fi

job_token="${SLURM_JOB_ID:-manual}"
scratch_user="${USERNAME:-${USER}}"

if [ -z "${SIMLINGO_SCRATCH_BASE:-}" ]; then
  export SIMLINGO_SCRATCH_BASE="/scratch/local/${scratch_user}"
fi

export SIMLINGO_STAGE_ROOT="${SIMLINGO_STAGE_ROOT:-${SIMLINGO_SCRATCH_BASE}/simlingo/${job_token}}"
export STAGED_DATA_ROOT="${SIMLINGO_STAGE_ROOT}/database/simlingo"
export STAGED_BUCKET_ROOT="${SIMLINGO_STAGE_ROOT}/database/bucketsv2_simlingo"
export SIMLINGO_STAGE_JOBS="${SIMLINGO_STAGE_JOBS:-8}"

if ! [[ "${SIMLINGO_STAGE_JOBS}" =~ ^[0-9]+$ ]] || [ "${SIMLINGO_STAGE_JOBS}" -lt 1 ]; then
  fail "SIMLINGO_STAGE_JOBS must be a positive integer, got: ${SIMLINGO_STAGE_JOBS}"
fi

stage_marker="${SIMLINGO_STAGE_ROOT}/.stage_complete"
signature_marker="${SIMLINGO_STAGE_ROOT}/.archive_signature"
archive_signature=$(
  {
    cat "${manifest_path}"
    [ ! -f "${metadata_path}" ] || cat "${metadata_path}"
    cat "${parent_dirs_path}"
    for archive_name in "${dataset_archives[@]}"; do
      stat -c '%n %s %Y' "${SIMLINGO_ARCHIVE_ROOT}/${archive_name}"
    done
    stat -c '%n %s %Y' "${SIMLINGO_ARCHIVE_ROOT}/bucketsv2_simlingo.tar"
  } | sha256sum | awk '{print $1}'
)

echo "Scratch stage root: ${SIMLINGO_STAGE_ROOT}"
echo "Dataset archive count: ${#dataset_archives[@]}"
echo "Parallel unpack jobs: ${SIMLINGO_STAGE_JOBS}"
if [ -n "${SIMLINGO_ROUTE_COUNT:-}" ]; then
  echo "Expected route count: ${SIMLINGO_ROUTE_COUNT}"
fi

if [ ! -f "${stage_marker}" ] || [ ! -f "${signature_marker}" ] || [ "$(cat "${signature_marker}")" != "${archive_signature}" ]; then
  rm -rf "${STAGED_DATA_ROOT}" "${STAGED_BUCKET_ROOT}"
  rm -f "${stage_marker}" "${signature_marker}"
  mkdir -p "${STAGED_DATA_ROOT}" "${SIMLINGO_STAGE_ROOT}/database"

  echo "Creating shared parent directories from ${parent_dirs_path}"
  while IFS= read -r parent_dir; do
    [ -n "${parent_dir}" ] || continue
    mkdir -p "${STAGED_DATA_ROOT}/${parent_dir}"
  done < "${parent_dirs_path}"

  for archive_name in "${dataset_archives[@]}"; do
    echo "Queueing ${archive_name} for extraction"
  done
  printf '%s\0' "${dataset_archive_paths[@]}" | xargs -0 -P "${SIMLINGO_STAGE_JOBS}" -I '{}' tar -xf '{}' -C "${STAGED_DATA_ROOT}"

  echo "Extracting bucketsv2_simlingo.tar to ${SIMLINGO_STAGE_ROOT}/database"
  tar -xf "${SIMLINGO_ARCHIVE_ROOT}/bucketsv2_simlingo.tar" -C "${SIMLINGO_STAGE_ROOT}/database"

  test -d "${STAGED_DATA_ROOT}/data/simlingo"
  test -f "${STAGED_BUCKET_ROOT}/buckets_paths.pkl"

  if [ -n "${SIMLINGO_ROUTE_COUNT:-}" ]; then
    staged_route_count=$(find "${STAGED_DATA_ROOT}/data/simlingo" -mindepth 4 -maxdepth 4 -type d -name 'Town*' | wc -l | tr -d ' ')
    if [ "${staged_route_count}" -ne "${SIMLINGO_ROUTE_COUNT}" ]; then
      fail "staged route count mismatch: expected ${SIMLINGO_ROUTE_COUNT}, got ${staged_route_count}"
    fi
  fi

  printf '%s\n' "${archive_signature}" > "${signature_marker}"
  touch "${stage_marker}"
else
  echo "Scratch stage already matches archive manifest"
fi

echo "Using staged data root: ${STAGED_DATA_ROOT}"
echo "Using staged bucket root: ${STAGED_BUCKET_ROOT}"
