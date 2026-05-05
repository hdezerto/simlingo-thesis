import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


REQUIRED_DEFAULT_KEYS = {
  "agent",
  "benchmark",
  "out_root",
  "repo_root",
  "carla_root",
  "agent_file",
  "checkpoint",
  "routes_file",
  "partition",
  "cpus_per_task",
  "mem_gb",
  "time",
  "fps",
}


def _truthy(value):
  if isinstance(value, bool):
    return value
  if value is None:
    return False
  return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _has_completed_result(result_file: Path):
  if not result_file.exists():
    return False
  try:
    with result_file.open("r", encoding="utf-8") as f:
      result = json.load(f)
  except (json.JSONDecodeError, OSError):
    return False

  progress = result.get("_checkpoint", {}).get("progress", [])
  return (
    isinstance(progress, list)
    and len(progress) >= 2
    and progress[1] > 0
    and progress[0] >= progress[1]
  )


def _has_reusable_frames(frames_dir: Path, result_file: Path):
  if not _has_completed_result(result_file):
    return False

  rgb_dir = frames_dir / "rgb_front"
  left_dir = frames_dir / "rgb_left"
  right_dir = frames_dir / "rgb_right"
  rear_dir = frames_dir / "rgb_rear"
  meta_dir = frames_dir / "meta"
  if not rgb_dir.exists() or not left_dir.exists() or not right_dir.exists() or not rear_dir.exists() or not meta_dir.exists():
    return False

  frame_count = len([p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png"}])
  left_count = len([p for p in left_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png"}])
  right_count = len([p for p in right_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png"}])
  rear_count = len([p for p in rear_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png"}])
  meta_count = len([p for p in meta_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
  return frame_count > 0 and left_count > 0 and right_count > 0 and rear_count > 0 and meta_count > 0


def build_paths(cfg, route_id, seed):
    base_seed_dir = Path(cfg["out_root"]) / cfg["agent"] / cfg["benchmark"] / str(seed)
    run_dir = base_seed_dir / "run"
    res_dir = base_seed_dir / "res"
    recordings_dir = base_seed_dir / "recordings"
    frames_dir = base_seed_dir / "frames" / f"route{route_id}_seed{seed}"
    viz_dir = base_seed_dir / "viz" / str(route_id)
    out_dir = base_seed_dir / "out"
    err_dir = base_seed_dir / "err"
    out_log = out_dir / f"render_route{route_id}_seed{seed}.out"
    err_log = err_dir / f"render_route{route_id}_seed{seed}.err"

    run_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    recordings_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    err_dir.mkdir(parents=True, exist_ok=True)

    slurm_script = run_dir / f"render_route{route_id}_seed{seed}.slurm"
    result_file = res_dir / f"rendering_route{route_id}_seed{seed}_res.json"
    mp4_file = recordings_dir / f"RouteScenario_{route_id}_seed{seed}_rep0.mp4"

    return {
        "base_seed_dir": base_seed_dir,
        "run_dir": run_dir,
        "res_dir": res_dir,
        "recordings_dir": recordings_dir,
        "frames_dir": frames_dir,
        "viz_dir": viz_dir,
        "out_log": out_log,
        "err_log": err_log,
        "slurm_script": slurm_script,
        "result_file": result_file,
        "mp4_file": mp4_file,
    }


def _validate_manifest(manifest, manifest_path: Path):
  if not isinstance(manifest, dict):
    raise ValueError(f"Manifest must be a JSON object: {manifest_path}")

  if "defaults" not in manifest or "cases" not in manifest:
    raise ValueError(f"Manifest must contain 'defaults' and 'cases': {manifest_path}")

  defaults = manifest["defaults"]
  cases = manifest["cases"]

  if not isinstance(defaults, dict):
    raise ValueError("Manifest 'defaults' must be an object")
  if not isinstance(cases, list):
    raise ValueError("Manifest 'cases' must be a list")

  missing_keys = sorted(REQUIRED_DEFAULT_KEYS - set(defaults.keys()))
  if missing_keys:
    raise ValueError(f"Manifest defaults missing keys: {missing_keys}")

  for idx, case in enumerate(cases):
    if not isinstance(case, dict):
      raise ValueError(f"Case at index {idx} must be an object")
    if "route_id" not in case or "seed" not in case:
      raise ValueError(f"Case at index {idx} must include 'route_id' and 'seed'")


def _build_case_cfg(defaults, case):
  cfg = dict(defaults)
  cfg.update({k: v for k, v in case.items() if k in defaults})
  return cfg


def _expand_env_in_obj(obj):
  if isinstance(obj, dict):
    return {k: _expand_env_in_obj(v) for k, v in obj.items()}
  if isinstance(obj, list):
    return [_expand_env_in_obj(v) for v in obj]
  if isinstance(obj, str):
    return os.path.expandvars(obj)
  return obj


def _write_case_slurm_script(cfg, route_id, seed, p, render_only):
  script_text = slurm_text(cfg, route_id, seed, p, render_only=render_only)
  p["slurm_script"].write_text(script_text, encoding="utf-8")
  print(f"Wrote SLURM script: {p['slurm_script']}")


def _submit_case_slurm(slurm_script: Path):
  return subprocess.check_output(["sbatch", str(slurm_script)], universal_newlines=True).strip()


def slurm_text(cfg, route_id, seed, p, render_only=False):
  mail_directives = ""
  mail_user = cfg.get("mail_user", "")
  if mail_user:
    mail_type = cfg.get("mail_type", "END,FAIL")
    mail_directives = f"#SBATCH --mail-user={mail_user}\n#SBATCH --mail-type={mail_type}\n"

  partition = cfg.get("cpu_partition", "berzelius-cpu") if render_only else cfg["partition"]
  gres_line = "" if render_only else "#SBATCH --gres=gpu:1"
  max_attempts = int(cfg.get("eval_max_attempts", 2))
  case_index = int(cfg.get("_case_index", 0))
  port_stride = int(cfg.get("port_stride", 100))
  world_port_start = min(12000 + case_index * port_stride, 19999)
  world_port_end = min(world_port_start + port_stride - 1, 19999)
  tm_port_start = min(32000 + case_index * port_stride, 39999)
  tm_port_end = min(tm_port_start + port_stride - 1, 39999)
  temporal_history_mode = str(
    cfg.get("temporal_history_mode", os.environ.get("TEMPORAL_HISTORY_MODE", "real"))
  ).strip().lower().replace("-", "_")
  if temporal_history_mode not in {"real", "repeat_current", "zero"}:
    raise ValueError(
      "temporal_history_mode must be one of: real, repeat_current, zero. "
      f"Got: {temporal_history_mode}"
    )

  return f'''#!/bin/bash
#SBATCH --job-name=render_{route_id}_s{seed}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cfg["cpus_per_task"]}
#SBATCH --mem={cfg["mem_gb"]}gb
#SBATCH --time={cfg["time"]}
{gres_line}
#SBATCH -o {p["out_log"]}
#SBATCH -e {p["err_log"]}
{mail_directives}

set -euo pipefail

echo "JOB ID $SLURM_JOB_ID"

log() {{
  echo "[$(date '+%F %T')] $*"
}}

cd {cfg["repo_root"]}
log "Changed directory to {cfg["repo_root"]}"

module load Miniforge3/24.7.1-2-hpc1-bdist
export CONDA_DEFAULT_ENV=${{CONDA_DEFAULT_ENV:-}}
set +u
conda activate simlingo
set -u
export PYTHONPATH=${{PYTHONPATH:-}}
source thesis/env.sh
log "Environment initialized"

export CARLA_ROOT={cfg["carla_root"]}
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH={cfg["repo_root"]}:{cfg["repo_root"]}/Bench2Drive/leaderboard:{cfg["repo_root"]}/Bench2Drive/scenario_runner:$PYTHONPATH
export SCENARIO_RUNNER_ROOT={cfg["repo_root"]}/Bench2Drive/scenario_runner
export SAVE_PATH={p["viz_dir"]}/
log "Paths configured"

mkdir -p /tmp/$SLURM_JOB_ID
echo '#!/bin/bash' > /tmp/$SLURM_JOB_ID/xdg-user-dir
echo 'echo /scratch/local' >> /tmp/$SLURM_JOB_ID/xdg-user-dir
chmod +x /tmp/$SLURM_JOB_ID/xdg-user-dir
export PATH=/tmp/$SLURM_JOB_ID:$PATH

ROUTES_FILE={cfg["routes_file"]}
RESULT_FILE={p["result_file"]}
FRAME_DIR={p["frames_dir"]}
MP4_OUT={p["mp4_file"]}
DEBUG_VIZ={1 if _truthy(cfg.get("debug_viz", False)) else 0}
DEBUG_STRIDE={int(cfg.get("debug_stride", 5))}
DEBUG_SAVE_LANGUAGE={1 if _truthy(cfg.get("debug_save_language", True)) else 0}
SIMLINGO_CONDITION_ACTION_ON_LANGUAGE={1 if _truthy(cfg.get("condition_action_on_language", True)) else 0}
FORCE_RENDER_ONLY={1 if render_only else 0}
REUSE_EXISTING_FRAMES={1 if render_only else 0}
log "Case route={route_id} seed={seed}"
log "Frame render dir: $FRAME_DIR"
log "MP4 output: $MP4_OUT"
log "Debug viz: $DEBUG_VIZ, stride: $DEBUG_STRIDE, save language: $DEBUG_SAVE_LANGUAGE"
log "Condition action on language: $SIMLINGO_CONDITION_ACTION_ON_LANGUAGE"
log "Render-only mode: $FORCE_RENDER_ONLY"
log "Evaluator max attempts: {max_attempts}"

find_free_port() {{
  local start_port="$1"
  local end_port="$2"
  local port
  for port in $(seq "$start_port" "$end_port"); do
    if ! ss -ltn | awk '{{print $4}}' | grep -Eq "[:.]${{port}}$"; then
      echo "$port"
      return 0
    fi
  done
  return 1
}}

export DEBUG_VIZ="$DEBUG_VIZ"
export DEBUG_STRIDE="$DEBUG_STRIDE"
export DEBUG_SAVE_LANGUAGE="$DEBUG_SAVE_LANGUAGE"
export SIMLINGO_CONDITION_ACTION_ON_LANGUAGE="$SIMLINGO_CONDITION_ACTION_ON_LANGUAGE"
export TEMPORAL_HISTORY_MODE={shlex.quote(temporal_history_mode)}
log "Temporal history mode: $TEMPORAL_HISTORY_MODE"

if [[ "$DEBUG_VIZ" -eq 1 ]]; then
  mkdir -p "$FRAME_DIR/rgb_front" "$FRAME_DIR/rgb_left" "$FRAME_DIR/rgb_right" "$FRAME_DIR/rgb_rear" "$FRAME_DIR/meta"
  export FRAME_RENDER_DIR="$FRAME_DIR"
  FRAME_COUNT_EXISTING=$(find "$FRAME_DIR/rgb_front" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
  LEFT_COUNT_EXISTING=$(find "$FRAME_DIR/rgb_left" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
  RIGHT_COUNT_EXISTING=$(find "$FRAME_DIR/rgb_right" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
  REAR_COUNT_EXISTING=$(find "$FRAME_DIR/rgb_rear" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
  META_COUNT_EXISTING=$(find "$FRAME_DIR/meta" -maxdepth 1 -type f -name '*.json' | wc -l)
  if [[ "$FORCE_RENDER_ONLY" -eq 1 ]]; then
    if [[ "$FRAME_COUNT_EXISTING" -gt 0 && "$LEFT_COUNT_EXISTING" -gt 0 && "$RIGHT_COUNT_EXISTING" -gt 0 && "$REAR_COUNT_EXISTING" -gt 0 && "$META_COUNT_EXISTING" -gt 0 ]]; then
      EVAL_OK=1
      SKIP_EVAL=1
      log "Render-only mode: reusing existing multiview frames/meta (front=$FRAME_COUNT_EXISTING left=$LEFT_COUNT_EXISTING right=$RIGHT_COUNT_EXISTING rear=$REAR_COUNT_EXISTING meta=$META_COUNT_EXISTING), evaluator disabled"
    else
      echo "Render-only mode requested but reusable multiview frames/meta were not found"
      exit 1
    fi
  elif [[ "$REUSE_EXISTING_FRAMES" -eq 1 && "$FRAME_COUNT_EXISTING" -gt 0 && "$LEFT_COUNT_EXISTING" -gt 0 && "$RIGHT_COUNT_EXISTING" -gt 0 && "$REAR_COUNT_EXISTING" -gt 0 && "$META_COUNT_EXISTING" -gt 0 ]]; then
    EVAL_OK=1
    SKIP_EVAL=1
    log "Debug render enabled: reusing existing multiview frames/meta (front=$FRAME_COUNT_EXISTING left=$LEFT_COUNT_EXISTING right=$RIGHT_COUNT_EXISTING rear=$REAR_COUNT_EXISTING meta=$META_COUNT_EXISTING), skipping evaluator"
  else
    if [[ "$FRAME_COUNT_EXISTING" -gt 0 || "$LEFT_COUNT_EXISTING" -gt 0 || "$RIGHT_COUNT_EXISTING" -gt 0 || "$REAR_COUNT_EXISTING" -gt 0 || "$META_COUNT_EXISTING" -gt 0 ]]; then
      log "Ignoring incomplete previous frames/meta, cleaning $FRAME_DIR before evaluator rerun"
      rm -rf "$FRAME_DIR"
      mkdir -p "$FRAME_DIR/rgb_front" "$FRAME_DIR/rgb_left" "$FRAME_DIR/rgb_right" "$FRAME_DIR/rgb_rear" "$FRAME_DIR/meta"
      export FRAME_RENDER_DIR="$FRAME_DIR"
    fi
    EVAL_OK=0
    SKIP_EVAL=0
    log "Debug render enabled: no reusable multiview frames found, evaluator will run"
  fi
else
  unset FRAME_RENDER_DIR
  EVAL_OK=0
  SKIP_EVAL=0
  log "Debug render disabled: metrics-only run (no frame generation, no video stitching)"
fi

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  log "Starting evaluator run to generate frame sequence"
  for ATTEMPT in $(seq 1 {max_attempts}); do
    EVAL_WORLD_PORT=$(find_free_port {world_port_start} {world_port_end}) || {{
      echo "Could not find a free evaluation world port"
      exit 1
    }}
    EVAL_TM_PORT=$(find_free_port {tm_port_start} {tm_port_end}) || {{
      echo "Could not find a free traffic manager port"
      exit 1
    }}
    log "Evaluator attempt $ATTEMPT using world:$EVAL_WORLD_PORT tm:$EVAL_TM_PORT"

    set +e
    python -u {cfg["repo_root"]}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py \
      --routes="$ROUTES_FILE" \
      --routes-subset={route_id} \
      --repetitions=1 \
      --track=SENSORS \
      --checkpoint="$RESULT_FILE" \
      --timeout=600 \
      --agent={cfg["agent_file"]} \
      --agent-config={cfg["checkpoint"]} \
      --traffic-manager-seed={seed} \
      --port=$EVAL_WORLD_PORT \
      --traffic-manager-port=$EVAL_TM_PORT
    EVAL_RC=$?
    set -e

    if [[ "$DEBUG_VIZ" -eq 1 ]]; then
      FRAME_COUNT=$(find "$FRAME_DIR/rgb_front" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
      if [[ "$FRAME_COUNT" -gt 0 ]]; then
        EVAL_OK=1
        log "Evaluator attempt $ATTEMPT produced $FRAME_COUNT frames"
        break
      fi
    else
      if [[ "$EVAL_RC" -eq 0 ]]; then
        EVAL_OK=1
        log "Evaluator attempt $ATTEMPT completed successfully"
        break
      fi
    fi

    log "Evaluator attempt $ATTEMPT failed (rc=$EVAL_RC), retrying with fresh ports"
    sleep 5
  done
fi

if [[ "$EVAL_OK" -ne 1 ]]; then
  echo "Frame sequence was not generated after evaluator retries"
  exit 1
fi

if [[ "$DEBUG_VIZ" -eq 1 ]]; then
  log "Starting frame-stitch rendering"
  python -u {cfg["repo_root"]}/thesis/rendering/generate_video_from_frames.py \
    --images-folder "$FRAME_DIR" \
    --output-video "$MP4_OUT" \
    --sim-fps {float(cfg["fps"])}

  log "Render completed"
  echo "Done: $MP4_OUT"
  ls -lh "$MP4_OUT"
else
  log "Skipping frame stitching (debug_viz is disabled)"
fi
'''


def main():
    parser = argparse.ArgumentParser(description="Submit multi-case route render jobs from a manifest.")
    parser.add_argument(
        "--manifest",
        default="thesis/rendering/manifests/render_manifest_template.json",
        help="Path to a render manifest JSON file.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as f:
      manifest = _expand_env_in_obj(json.load(f))

    _validate_manifest(manifest, manifest_path)

    defaults = manifest["defaults"]
    cases = manifest["cases"]

    print(f"Loaded manifest: {manifest_path}")
    print(f"Total cases: {len(cases)}")

    for idx, case in enumerate(cases):
        route_id = str(case["route_id"])
        seed = int(case["seed"])

        print(f"Preparing case route={route_id} seed={seed}")

        cfg = _build_case_cfg(defaults, case)
        cfg["_case_index"] = idx

        p = build_paths(cfg, route_id, seed)
        debug_viz_enabled = _truthy(cfg.get("debug_viz", False))
        render_only = debug_viz_enabled and _has_reusable_frames(p["frames_dir"], p["result_file"])
        _write_case_slurm_script(cfg, route_id, seed, p, render_only)

        if render_only:
            print(f"{route_id} seed={seed}: reusable frames found -> render-only mode (evaluator skipped)")

        if args.dry_run:
            print(f"[DRY] would submit: {p['slurm_script']}")
            continue

        output = _submit_case_slurm(p["slurm_script"])
        print(f"{route_id} seed={seed} -> {output}")

    print("Done.")


if __name__ == "__main__":
    main()
