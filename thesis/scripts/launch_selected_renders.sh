#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: launch_selected_renders.sh [--dry-run] <target>

Targets:
  main            baseline + Q-former + Delta diagnostic/result-selected renders
  baseline        corrected baseline diagnostic/result-selected renders only
  qformer-fresh   Q-former v8 fresh diagnostic/result-selected renders only
  diagnostics     loaded-LoRA + no-temporal diagnostic renders
  loaded          Q-former v8 loaded-LoRA diagnostic renders only
  no-temporal     Q-former v8 no-temporal diagnostic renders only
  delta           Delta v5 fresh diagnostic/result-selected renders only
  repeat-current  Q-former v8 fresh repeated-current diagnostic renders only
  all             main + diagnostics + repeat-current
USAGE
}

DRY_RUN=""
TARGET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN="--dry-run"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -n "${TARGET}" ]]; then
        usage
        exit 2
      fi
      TARGET="$1"
      shift
      ;;
  esac
done

if [[ -z "${TARGET}" ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_DIR}"
source thesis/env.sh

submit_manifest() {
  local manifest="$1"
  python thesis/rendering/submit_render_jobs.py \
    --manifest "${manifest}" \
    ${DRY_RUN}
}

submit_main() {
  submit_manifest thesis/rendering/manifests/render_manifest_baseline_evalfix_diagnostic_and_result_selected.json
  submit_manifest thesis/rendering/manifests/render_manifest_temporal_qformer_v8_fresh_lora_diagnostic_and_result_selected.json
  submit_manifest thesis/rendering/manifests/render_manifest_temporal_delta_feature_v5_fresh_lora_diagnostic_and_result_selected.json
}

submit_diagnostics() {
  submit_manifest thesis/rendering/manifests/render_manifest_temporal_qformer_v8_loaded_lora_diagnostic_selected.json
  submit_manifest thesis/rendering/manifests/render_manifest_temporal_qformer_v8_fresh_lora_no_temporal_diagnostic_selected.json
}


case "${TARGET}" in
  main)
    submit_main
    ;;
  baseline)
    submit_manifest thesis/rendering/manifests/render_manifest_baseline_evalfix_diagnostic_and_result_selected.json
    ;;
  qformer-fresh)
    submit_manifest thesis/rendering/manifests/render_manifest_temporal_qformer_v8_fresh_lora_diagnostic_and_result_selected.json
    ;;
  diagnostics)
    submit_diagnostics
    ;;
  loaded)
    submit_manifest thesis/rendering/manifests/render_manifest_temporal_qformer_v8_loaded_lora_diagnostic_selected.json
    ;;
  no-temporal)
    submit_manifest thesis/rendering/manifests/render_manifest_temporal_qformer_v8_fresh_lora_no_temporal_diagnostic_selected.json
    ;;
  delta)
    submit_manifest thesis/rendering/manifests/render_manifest_temporal_delta_feature_v5_fresh_lora_diagnostic_and_result_selected.json
    ;;
  repeat-current)
    submit_manifest thesis/rendering/manifests/render_manifest_temporal_qformer_v8_fresh_lora_repeat_current_diagnostic_selected.json
    ;;
  all)
    submit_main
    submit_diagnostics
    submit_manifest thesis/rendering/manifests/render_manifest_temporal_qformer_v8_fresh_lora_repeat_current_diagnostic_selected.json
    ;;
  *)
    usage
    exit 2
    ;;
esac
