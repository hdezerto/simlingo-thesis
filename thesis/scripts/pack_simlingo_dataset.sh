#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR_DEFAULT=$(cd "${SCRIPT_DIR}/../.." && pwd)

if [ -z "${REPO_DIR:-}" ]; then
  export REPO_DIR="${REPO_DIR_DEFAULT}"
fi

source "${REPO_DIR}/thesis/env.sh"

ARCHIVE_ROOT="${SIMLINGO_ARCHIVE_ROOT%/}"
DATA_ROOT="${SIMLINGO_DATA_ROOT}"
BUCKET_ROOT="${SIMLINGO_BUCKET_ROOT}"
ARCHIVE_PARTS="${SIMLINGO_ARCHIVE_PARTS}"
ARCHIVE_PARENT="$(dirname "${ARCHIVE_ROOT}")"
ARCHIVE_NAME="$(basename "${ARCHIVE_ROOT}")"
LOCK_DIR="${ARCHIVE_PARENT}/.${ARCHIVE_NAME}.pack.lock"
mkdir -p "${ARCHIVE_PARENT}"

TMP_ARCHIVE_ROOT="$(mktemp -d "${ARCHIVE_PARENT}/.${ARCHIVE_NAME}.pack.XXXXXX")"
TMP_DIR="$(mktemp -d "${ARCHIVE_PARENT}/.${ARCHIVE_NAME}.work.XXXXXX")"
MANIFEST_DIR="${TMP_ARCHIVE_ROOT}/manifests"
DATASET_MANIFEST="${TMP_ARCHIVE_ROOT}/dataset_archives.txt"
METADATA_FILE="${TMP_ARCHIVE_ROOT}/dataset_metadata.env"
PARENT_DIRS_FILE="${TMP_ARCHIVE_ROOT}/stage_parent_dirs.txt"
PARENT_DIRS_TMP="${TMP_DIR}/stage_parent_dirs.raw"
mkdir -p "${MANIFEST_DIR}"
backup_root=""
lock_acquired=0
cleanup() {
  [ -z "${TMP_ARCHIVE_ROOT:-}" ] || rm -rf "${TMP_ARCHIVE_ROOT}"
  [ -z "${TMP_DIR:-}" ] || rm -rf "${TMP_DIR}"
  if [ "${lock_acquired:-0}" -eq 1 ]; then
    rmdir "${LOCK_DIR}" 2>/dev/null || true
  fi
}
on_exit() {
  status=$?
  if [ "${status}" -ne 0 ] && [ -n "${backup_root:-}" ] && [ -d "${backup_root}" ] && [ ! -e "${ARCHIVE_ROOT}" ]; then
    mv "${backup_root}" "${ARCHIVE_ROOT}"
  fi
  cleanup
}
trap on_exit EXIT

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another SimLingo pack job appears to be running: ${LOCK_DIR}" >&2
  exit 1
fi
lock_acquired=1

if ! [[ "${ARCHIVE_PARTS}" =~ ^[0-9]+$ ]] || [ "${ARCHIVE_PARTS}" -lt 1 ]; then
  echo "SIMLINGO_ARCHIVE_PARTS must be a positive integer, got: ${ARCHIVE_PARTS}" >&2
  exit 1
fi

if [ ! -d "${DATA_ROOT}/data/simlingo" ]; then
  echo "Missing dataset root: ${DATA_ROOT}/data/simlingo" >&2
  exit 1
fi

if [ ! -f "${BUCKET_ROOT}/buckets_paths.pkl" ]; then
  echo "Missing bucket file: ${BUCKET_ROOT}/buckets_paths.pkl" >&2
  exit 1
fi

echo "Packing SimLingo dataset for scratch staging"
echo "Source data root: ${DATA_ROOT}"
echo "Source bucket root: ${BUCKET_ROOT}"
echo "Archive root: ${ARCHIVE_ROOT}"
echo "Temporary archive root: ${TMP_ARCHIVE_ROOT}"
echo "Archive parts: ${ARCHIVE_PARTS}"

route_sizes_tsv="${TMP_DIR}/route_sizes.tsv"

find "${DATA_ROOT}/data/simlingo" -mindepth 4 -maxdepth 4 -type d -name 'Town*' \
  | LC_ALL=C sort \
  | while IFS= read -r route_dir; do
      relative_data_route="${route_dir#${DATA_ROOT}/}"
      relative_suffix="${relative_data_route#data/}"
      total_size_kib=0
      total_file_count=0

      for branch in data commentary drivelm dreamer; do
        branch_route="${branch}/${relative_suffix}"
        if [ -d "${DATA_ROOT}/${branch_route}" ]; then
          branch_size_kib=$(du -s "${DATA_ROOT}/${branch_route}" | awk '{print $1}')
          branch_file_count=$(find "${DATA_ROOT}/${branch_route}" -type f | wc -l | tr -d ' ')
          total_size_kib=$((total_size_kib + branch_size_kib))
          total_file_count=$((total_file_count + branch_file_count))
        fi
      done

      printf '%s\t%s\t%s\n' "${total_size_kib}" "${total_file_count}" "${relative_data_route}"
    done > "${route_sizes_tsv}"

if [ ! -s "${route_sizes_tsv}" ]; then
  echo "No route directories found under ${DATA_ROOT}/data/simlingo" >&2
  exit 1
fi

expected_route_count=$(wc -l < "${route_sizes_tsv}" | tr -d ' ')
expected_file_count=$(awk -F'\t' '{sum += $2} END {print sum + 0}' "${route_sizes_tsv}")
echo "Route count: ${expected_route_count}"
echo "File count: ${expected_file_count}"

for part_idx in $(seq 0 $((ARCHIVE_PARTS - 1))); do
  : > "${MANIFEST_DIR}/simlingo_part_$(printf '%03d' "${part_idx}").lst"
done

total_size_kib=$(awk -F'\t' '{sum += $1} END {print sum + 0}' "${route_sizes_tsv}")

sort -nr -k1,1 -k2,2 "${route_sizes_tsv}" | awk -F'\t' \
  -v parts="${ARCHIVE_PARTS}" \
  -v total_size="${total_size_kib}" \
  -v total_files="${expected_file_count}" \
  -v outdir="${MANIFEST_DIR}" \
  -v summary="${MANIFEST_DIR}/shard_summary.tsv" '
  BEGIN {
    for (i = 1; i <= parts; i++) {
      shard_sum[i] = 0
      shard_files[i] = 0
      shard_count[i] = 0
    }
  }
  {
    target = 1
    for (i = 2; i <= parts; i++) {
      score_i = (total_size > 0 ? shard_sum[i] / total_size : 0) + (total_files > 0 ? shard_files[i] / total_files : 0)
      score_target = (total_size > 0 ? shard_sum[target] / total_size : 0) + (total_files > 0 ? shard_files[target] / total_files : 0)
      if (score_i < score_target) {
        target = i
      }
    }
    print $3 >> sprintf("%s/simlingo_part_%03d.lst", outdir, target - 1)
    shard_sum[target] += $1
    shard_files[target] += $2
    shard_count[target] += 1
  }
  END {
    for (i = 1; i <= parts; i++) {
      printf("simlingo_part_%03d\t%d\t%d\t%d\n", i - 1, shard_count[i], shard_sum[i], shard_files[i]) >> summary
    }
  }
'

echo "Shard summary (archive, routes, total_size_kib, files):"
cat "${MANIFEST_DIR}/shard_summary.tsv"

: > "${DATASET_MANIFEST}"
: > "${PARENT_DIRS_TMP}"
for manifest in "${MANIFEST_DIR}"/simlingo_part_*.lst; do
  if [ ! -s "${manifest}" ]; then
    rm -f "${manifest}"
    continue
  fi

  archive_base="$(basename "${manifest}" .lst)"
  archive_tmp="${TMP_ARCHIVE_ROOT}/${archive_base}.tar.tmp"
  archive_final="${TMP_ARCHIVE_ROOT}/${archive_base}.tar"
  archive_paths="${TMP_DIR}/${archive_base}.paths"
  : > "${archive_paths}"

  while IFS= read -r relative_data_route; do
    [ -n "${relative_data_route}" ] || continue
    printf '%s\n' "${relative_data_route}" >> "${archive_paths}"
    printf '%s\n' "${relative_data_route%/*}" >> "${PARENT_DIRS_TMP}"
    relative_suffix="${relative_data_route#data/}"

    for branch in commentary drivelm dreamer; do
      branch_route="${branch}/${relative_suffix}"
      if [ -d "${DATA_ROOT}/${branch_route}" ]; then
        printf '%s\n' "${branch_route}" >> "${archive_paths}"
        printf '%s\n' "${branch_route%/*}" >> "${PARENT_DIRS_TMP}"
      fi
    done
  done < "${manifest}"

  LC_ALL=C sort -u "${archive_paths}" -o "${archive_paths}"

  echo "Creating ${archive_final}"
  tar -cf "${archive_tmp}" -C "${DATA_ROOT}" -T "${archive_paths}"
  mv "${archive_tmp}" "${archive_final}"
  printf '%s\n' "$(basename "${archive_final}")" >> "${DATASET_MANIFEST}"
done

LC_ALL=C sort -u "${PARENT_DIRS_TMP}" > "${PARENT_DIRS_FILE}"

bucket_tmp="${TMP_ARCHIVE_ROOT}/bucketsv2_simlingo.tar.tmp"
bucket_final="${TMP_ARCHIVE_ROOT}/bucketsv2_simlingo.tar"
echo "Creating ${bucket_final}"
tar -cf "${bucket_tmp}" -C "$(dirname "${BUCKET_ROOT}")" "$(basename "${BUCKET_ROOT}")"
mv "${bucket_tmp}" "${bucket_final}"

manifest_route_count=0
for manifest in "${MANIFEST_DIR}"/simlingo_part_*.lst; do
  [ -e "${manifest}" ] || continue
  count=$(wc -l < "${manifest}" | tr -d ' ')
  manifest_route_count=$((manifest_route_count + count))
done

if [ "${manifest_route_count}" -ne "${expected_route_count}" ]; then
  echo "Packed route manifest count mismatch: expected ${expected_route_count}, got ${manifest_route_count}" >&2
  exit 1
fi

archive_count=0
while IFS= read -r archive_name; do
  [ -n "${archive_name}" ] || continue
  if [ ! -s "${TMP_ARCHIVE_ROOT}/${archive_name}" ]; then
    echo "Archive was not created or is empty: ${TMP_ARCHIVE_ROOT}/${archive_name}" >&2
    exit 1
  fi
  archive_count=$((archive_count + 1))
done < "${DATASET_MANIFEST}"

if [ "${archive_count}" -eq 0 ]; then
  echo "No dataset archives were created" >&2
  exit 1
fi

if [ ! -s "${bucket_final}" ]; then
  echo "Bucket archive was not created or is empty: ${bucket_final}" >&2
  exit 1
fi

{
  printf 'SIMLINGO_ROUTE_COUNT=%s\n' "${expected_route_count}"
  printf 'SIMLINGO_FILE_COUNT=%s\n' "${expected_file_count}"
  printf 'SIMLINGO_ARCHIVE_COUNT=%s\n' "${archive_count}"
  printf 'SIMLINGO_ARCHIVE_PARTS=%s\n' "${ARCHIVE_PARTS}"
  printf 'SIMLINGO_PACKED_AT_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${METADATA_FILE}"

if [ -e "${ARCHIVE_ROOT}" ]; then
  backup_root="${ARCHIVE_ROOT}.previous.$(date -u +%Y%m%d%H%M%S)"
  mv "${ARCHIVE_ROOT}" "${backup_root}"
fi
mv "${TMP_ARCHIVE_ROOT}" "${ARCHIVE_ROOT}"
TMP_ARCHIVE_ROOT=""

if [ -n "${backup_root}" ]; then
  rm -rf "${backup_root}"
  backup_root=""
fi

echo "Dataset archive manifest: ${ARCHIVE_ROOT}/dataset_archives.txt"
echo "Finished. Archives written to ${ARCHIVE_ROOT}"
