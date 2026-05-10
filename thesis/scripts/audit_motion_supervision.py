#!/usr/bin/env python3
"""Audit motion-related supervision in the staged SimLingo dataset.

The goal is not to reproduce training. It cheaply checks whether the labels
that supervise temporal behaviour agree with each other:

- generated commentary text
- current nearby dynamic actors from boxes
- future ego motion from waypoint/measurement labels
"""

import argparse
import gzip
import json
import math
import random
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SAVE_PERIOD_SECONDS = 0.25
MOVING_SPEED_MS = 0.5
STOPPED_SPEED_MS = 0.2
SAME_LANE_HALF_WIDTH_M = 1.8
RELEVANT_LEAD_DISTANCE_M = 25.0


MOTION_WORD_RE = re.compile(
    r"\b(vehicle|vehicles|car|cars|walker|pedestrian|moving|move|stopped|"
    r"stationary|cross|crossing|yield|junction|intersection|brake|slow|"
    r"accelerate|clear|blocked|approaching|oncoming)\b",
    re.IGNORECASE,
)


TEXT_FLAGS = {
    "says_stopped": re.compile(r"\b(stopped|stationary|not moving)\b", re.IGNORECASE),
    "says_moving": re.compile(r"\b(moving|move|approaching|oncoming|crossing|cross)\b", re.IGNORECASE),
    "says_clear": re.compile(r"\b(clear|moving away)\b", re.IGNORECASE),
    "says_accelerate": re.compile(r"\b(accelerate|speed up|continue)\b", re.IGNORECASE),
    "says_brake": re.compile(r"\b(brake|slow|stop|yield|wait)\b", re.IGNORECASE),
    "mentions_actor": re.compile(
        r"\b(car|cars|vehicle|vehicles|walker|pedestrian|object|traffic|lead|front|behind|follow|stay behind)\b",
        re.IGNORECASE,
    ),
    "says_following": re.compile(r"\b(follow|stay behind|behind|reduced speed)\b", re.IGNORECASE),
    "says_target_speed": re.compile(r"\btarget speed\b", re.IGNORECASE),
}


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_optional_json_gz(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return read_json_gz(path)
    except (OSError, json.JSONDecodeError):
        return None


def route_dirs(data_root: Path) -> List[Path]:
    return sorted((data_root / "data" / "simlingo").glob("*/*/*/Town*"))


def filtered_measurement_files(route_dir: Path, args: argparse.Namespace) -> List[Path]:
    measurement_files = sorted((route_dir / "measurements").glob("*.json.gz"))
    filtered = []
    for measurement_file in measurement_files[:: args.frame_step]:
        frame_idx = int(measurement_file.stem.split(".")[0])
        if args.frame_start is not None and frame_idx < args.frame_start:
            continue
        if args.frame_end is not None and frame_idx > args.frame_end:
            continue
        filtered.append(measurement_file)
    return filtered


def commentary_path_for_measurement(measurement_path: Path, data_root: Path) -> Path:
    relative = measurement_path.relative_to(data_root)
    parts = list(relative.parts)
    parts[0] = "commentary"
    parts[-2] = "commentary"
    return data_root.joinpath(*parts)


def box_path_for_measurement(measurement_path: Path) -> Path:
    return measurement_path.parent.parent / "boxes" / measurement_path.name


def rgb_path_for_measurement(measurement_path: Path) -> Path:
    return measurement_path.parent.parent / "rgb" / measurement_path.name.replace(".json.gz", ".jpg")


def measurement_path(route_dir: Path, frame_idx: int) -> Path:
    return route_dir / "measurements" / f"{frame_idx:04}.json.gz"


def get_commentary_text(commentary_data: Optional[Any]) -> str:
    if commentary_data is None:
        return ""
    if isinstance(commentary_data, str):
        return commentary_data
    if isinstance(commentary_data, dict):
        for key in ("commentary", "answer", "text", "caption"):
            value = commentary_data.get(key)
            if isinstance(value, str):
                return value
        return " ".join(str(value) for value in commentary_data.values() if isinstance(value, str))
    if isinstance(commentary_data, list):
        return " ".join(str(value) for value in commentary_data if isinstance(value, str))
    return str(commentary_data)


def ego_xy(measurement: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    matrix = measurement.get("ego_matrix")
    if not matrix:
        return None
    try:
        return float(matrix[0][3]), float(matrix[1][3])
    except (IndexError, TypeError, ValueError):
        return None


def future_motion(route_dir: Path, current_idx: int, pred_len: int) -> Dict[str, Any]:
    measurements = []
    for idx in range(current_idx, current_idx + pred_len + 1):
        path = measurement_path(route_dir, idx)
        if not path.is_file():
            break
        data = load_optional_json_gz(path)
        if isinstance(data, dict):
            measurements.append(data)

    positions = [ego_xy(measurement) for measurement in measurements]
    positions = [position for position in positions if position is not None]

    speeds = []
    for before, after in zip(positions, positions[1:]):
        distance = math.hypot(after[0] - before[0], after[1] - before[1])
        speeds.append(distance / SAVE_PERIOD_SECONDS)

    current_speed = None
    if measurements:
        speed_value = measurements[0].get("speed")
        if isinstance(speed_value, (int, float)):
            current_speed = float(speed_value)

    if not speeds:
        return {
            "current_speed": current_speed,
            "future_speeds": [],
            "future_mean_speed": None,
            "future_min_speed": None,
            "future_first_speed": None,
            "future_action_hint": "unknown",
        }

    mean_speed = sum(speeds) / len(speeds)
    min_speed = min(speeds)
    first_speed = speeds[0]
    initial_mean_speed = sum(speeds[: min(3, len(speeds))]) / min(3, len(speeds))
    late_mean_speed = sum(speeds[-min(3, len(speeds)):]) / min(3, len(speeds))
    action_hint = "maintain"
    if current_speed is not None:
        if current_speed < 0.5:
            if late_mean_speed > 1.0 or mean_speed > 1.0:
                action_hint = "accelerate"
            else:
                action_hint = "slow_or_stop"
        elif late_mean_speed < current_speed - 1.0 and late_mean_speed < initial_mean_speed - 0.75:
            action_hint = "slow_or_stop"
        elif late_mean_speed > current_speed + 1.0 and late_mean_speed > initial_mean_speed + 0.75:
            action_hint = "accelerate"

    return {
        "current_speed": current_speed,
        "future_speeds": [round(speed, 3) for speed in speeds],
        "future_mean_speed": round(mean_speed, 3),
        "future_min_speed": round(min_speed, 3),
        "future_first_speed": round(first_speed, 3),
        "future_initial_mean_speed": round(initial_mean_speed, 3),
        "future_late_mean_speed": round(late_mean_speed, 3),
        "future_action_hint": action_hint,
    }


def normalize_boxes(box_data: Any) -> List[Dict[str, Any]]:
    if isinstance(box_data, list):
        return [item for item in box_data if isinstance(item, dict)]
    if isinstance(box_data, dict):
        for key in ("boxes", "bounding_boxes", "data"):
            value = box_data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def actor_summary(box_data: Optional[Any]) -> Dict[str, Any]:
    boxes = normalize_boxes(box_data)
    nearby = []
    moving = []
    stopped = []
    same_lane = []
    same_lane_moving = []
    same_lane_stopped = []

    for box in boxes:
        actor_class = box.get("class")
        if actor_class not in {"car", "walker"}:
            continue
        position = box.get("position")
        if not isinstance(position, list) or len(position) < 2:
            continue
        try:
            x_pos = float(position[0])
            y_pos = float(position[1])
            speed = float(box.get("speed", 0.0))
        except (TypeError, ValueError):
            continue

        if not (0.0 <= x_pos <= 35.0 and abs(y_pos) <= 12.0):
            continue

        actor = {
            "class": actor_class,
            "x": round(x_pos, 2),
            "y": round(y_pos, 2),
            "distance": round(math.hypot(x_pos, y_pos), 2),
            "speed": round(speed, 3),
        }
        nearby.append(actor)
        if abs(y_pos) <= SAME_LANE_HALF_WIDTH_M:
            same_lane.append(actor)
            if speed > MOVING_SPEED_MS:
                same_lane_moving.append(actor)
            elif speed <= STOPPED_SPEED_MS:
                same_lane_stopped.append(actor)

        if speed > MOVING_SPEED_MS:
            moving.append(actor)
        elif speed <= STOPPED_SPEED_MS:
            stopped.append(actor)

    nearby.sort(key=lambda actor: actor["distance"])
    moving.sort(key=lambda actor: actor["distance"])
    stopped.sort(key=lambda actor: actor["distance"])
    same_lane.sort(key=lambda actor: actor["distance"])
    same_lane_moving.sort(key=lambda actor: actor["distance"])
    same_lane_stopped.sort(key=lambda actor: actor["distance"])

    return {
        "nearby_dynamic_count": len(nearby),
        "nearby_moving_count": len(moving),
        "nearby_stopped_count": len(stopped),
        "same_lane_dynamic_count": len(same_lane),
        "same_lane_moving_count": len(same_lane_moving),
        "same_lane_stopped_count": len(same_lane_stopped),
        "closest_dynamic": nearby[0] if nearby else None,
        "closest_moving": moving[0] if moving else None,
        "closest_stopped": stopped[0] if stopped else None,
        "closest_same_lane_dynamic": same_lane[0] if same_lane else None,
        "closest_same_lane_moving": same_lane_moving[0] if same_lane_moving else None,
        "closest_same_lane_stopped": same_lane_stopped[0] if same_lane_stopped else None,
    }


def text_flags(text: str) -> Dict[str, bool]:
    flags = {name: bool(pattern.search(text)) for name, pattern in TEXT_FLAGS.items()}
    flags["has_commentary"] = bool(text.strip())
    return flags


def is_close_lead_actor(actor: Optional[Dict[str, Any]]) -> bool:
    return actor is not None and actor["distance"] <= RELEVANT_LEAD_DISTANCE_M


def inconsistency_reasons(flags: Dict[str, bool], actors: Dict[str, Any], motion: Dict[str, Any]) -> List[str]:
    reasons = []
    if not flags["has_commentary"]:
        return reasons

    lead_actor = actors["closest_same_lane_dynamic"]
    moving_lead = actors["closest_same_lane_moving"]
    has_close_lead = is_close_lead_actor(lead_actor)
    has_close_moving_lead = is_close_lead_actor(moving_lead)
    action_hint = motion["future_action_hint"]

    if flags["says_stopped"] and has_close_lead and lead_actor["speed"] > MOVING_SPEED_MS:
        reasons.append("commentary_says_stopped_but_same_lane_lead_is_moving")
    if flags["says_clear"] and has_close_lead:
        reasons.append("commentary_says_clear_but_same_lane_lead_exists")
    if flags["says_accelerate"] and action_hint == "slow_or_stop":
        reasons.append("commentary_says_accelerate_but_future_waypoints_slow")
    if flags["says_brake"] and action_hint == "accelerate":
        reasons.append("commentary_says_brake_but_future_waypoints_accelerate")
    if has_close_moving_lead and not flags["mentions_actor"]:
        reasons.append("same_lane_moving_lead_not_referenced")

    return reasons


def should_sample(text: str, actors: Dict[str, Any], only_motion_text: bool) -> bool:
    if only_motion_text:
        return bool(MOTION_WORD_RE.search(text))
    return bool(MOTION_WORD_RE.search(text)) or actors["nearby_dynamic_count"] > 0


def safe_stem(route_dir: Path, data_root: Path, frame_idx: int) -> str:
    route_name = str(route_dir.relative_to(data_root)).replace("/", "__")
    route_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", route_name)
    return f"{route_name}__frame_{frame_idx:04d}"


def maybe_copy_rgb(
    record: Dict[str, Any],
    rgb_path: Path,
    route_dir: Path,
    data_root: Path,
    frame_idx: int,
    output_dir: Path,
    copy_rgb: bool,
    copy_all_rgb: bool,
) -> None:
    should_copy = copy_all_rgb or (copy_rgb and bool(record["inconsistency_reasons"]))
    if not should_copy:
        return

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(route_dir, data_root, frame_idx)
    copied_rgb = frames_dir / f"{stem}.jpg"
    sidecar_json = frames_dir / f"{stem}.json"

    if rgb_path.is_file():
        shutil.copy2(str(rgb_path), str(copied_rgb))
        record["copied_rgb"] = str(copied_rgb)
    else:
        record["copied_rgb"] = None
        record["missing_rgb"] = str(rgb_path)

    with sidecar_json.open("w", encoding="utf-8") as sidecar_file:
        json.dump(record, sidecar_file, indent=2, sort_keys=True)
        sidecar_file.write("\n")
    record["sidecar_json"] = str(sidecar_json)


def audit(args: argparse.Namespace) -> Tuple[Path, Path, Counter]:
    data_root = Path(args.data_root).expanduser().resolve()
    if not (data_root / "data" / "simlingo").is_dir():
        raise SystemExit(
            f"Missing {data_root / 'data' / 'simlingo'}.\n"
            "Use a staged dataset root, for example $STAGED_DATA_ROOT after sourcing "
            "thesis/scripts/stage_simlingo_to_scratch.sh inside a job."
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"motion_supervision_audit_{stamp}.jsonl"
    summary_path = output_dir / f"motion_supervision_audit_{stamp}_summary.txt"

    counts: Counter[str] = Counter()
    sampled = 0
    sampled_routes = set()
    rng = random.Random(args.seed)

    routes = route_dirs(data_root)
    if not args.include_validation:
        routes = [route for route in routes if "validation_" not in str(route)]
    if args.route_contains:
        routes = [route for route in routes if args.route_contains in str(route)]
    rng.shuffle(routes)
    if args.max_routes is not None:
        routes = routes[: args.max_routes]

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        route_files = []
        for route_dir in routes:
            measurement_files = filtered_measurement_files(route_dir, args)
            if args.shuffle_frames:
                rng.shuffle(measurement_files)
            if measurement_files:
                route_files.append((route_dir, measurement_files))

        active = True
        file_offset = 0
        while active and sampled < args.max_samples:
            active = False
            for route_dir, measurement_files in route_files:
                if sampled >= args.max_samples:
                    break
                if file_offset >= len(measurement_files):
                    continue
                active = True

                measurement_file = measurement_files[file_offset]
                frame_idx = int(measurement_file.stem.split(".")[0])
                commentary_data = load_optional_json_gz(commentary_path_for_measurement(measurement_file, data_root))
                commentary = get_commentary_text(commentary_data)
                box_path = box_path_for_measurement(measurement_file)
                rgb_path = rgb_path_for_measurement(measurement_file)
                actors = actor_summary(load_optional_json_gz(box_path))

                if not should_sample(commentary, actors, args.only_motion_text):
                    continue

                motion = future_motion(route_dir, frame_idx, args.pred_len)
                flags = text_flags(commentary)
                reasons = inconsistency_reasons(flags, actors, motion)

                record = {
                    "route": str(route_dir.relative_to(data_root)),
                    "frame": frame_idx,
                    "measurement": str(measurement_file.relative_to(data_root)),
                    "rgb": str(rgb_path.relative_to(data_root)),
                    "boxes": str(box_path.relative_to(data_root)),
                    "commentary": commentary,
                    "text_flags": flags,
                    "actors": actors,
                    "future_motion": motion,
                    "inconsistency_reasons": reasons,
                }
                maybe_copy_rgb(
                    record,
                    rgb_path,
                    route_dir,
                    data_root,
                    frame_idx,
                    output_dir,
                    args.copy_rgb,
                    args.copy_all_rgb,
                )
                jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")

                sampled += 1
                sampled_routes.add(str(route_dir.relative_to(data_root)))
                counts["samples_written"] += 1
                counts["samples_with_reasons"] += int(bool(reasons))
                counts["samples_without_commentary"] += int(not bool(commentary.strip()))
                for reason in reasons:
                    counts[reason] += 1
            file_offset += 1

    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write("Motion supervision audit\n")
        summary_file.write(f"data_root: {data_root}\n")
        summary_file.write(f"routes_seen: {len(routes)}\n")
        summary_file.write(f"include_validation: {args.include_validation}\n")
        summary_file.write(f"route_contains: {args.route_contains}\n")
        summary_file.write(f"frame_start: {args.frame_start}\n")
        summary_file.write(f"frame_end: {args.frame_end}\n")
        summary_file.write(f"frame_step: {args.frame_step}\n")
        summary_file.write(f"max_samples: {args.max_samples}\n")
        summary_file.write(f"seed: {args.seed}\n")
        summary_file.write(f"shuffle_frames: {args.shuffle_frames}\n")
        summary_file.write(f"routes_sampled: {len(sampled_routes)}\n")
        summary_file.write(f"jsonl: {jsonl_path}\n\n")
        if args.copy_rgb or args.copy_all_rgb:
            summary_file.write(f"copied frames: {output_dir / 'frames'}\n\n")
        for key, value in counts.most_common():
            summary_file.write(f"{key}: {value}\n")

    return jsonl_path, summary_path, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Staged dataset root, usually $STAGED_DATA_ROOT.")
    parser.add_argument("--output-dir", default="thesis/results/supervision_audit")
    parser.add_argument("--max-routes", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--frame-step", type=int, default=5, help="Read every Nth saved frame.")
    parser.add_argument("--seed", type=int, default=9876, help="Seed for route/frame sampling.")
    parser.add_argument(
        "--shuffle-frames",
        action="store_true",
        help="Shuffle candidate frames within each route before round-robin sampling.",
    )
    parser.add_argument("--route-contains", default="", help="Only audit routes whose path contains this text.")
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Include validation/evaluation routes. By default the audit checks training routes only.",
    )
    parser.add_argument("--frame-start", type=int, default=None, help="First saved frame index to inspect.")
    parser.add_argument("--frame-end", type=int, default=None, help="Last saved frame index to inspect.")
    parser.add_argument("--pred-len", type=int, default=10, help="Number of future saved frames to inspect.")
    parser.add_argument(
        "--copy-rgb",
        action="store_true",
        help="Copy front RGB frames for suspicious samples to output-dir/frames.",
    )
    parser.add_argument(
        "--copy-all-rgb",
        action="store_true",
        help="Copy front RGB frames for every written sample to output-dir/frames.",
    )
    parser.add_argument(
        "--only-motion-text",
        action="store_true",
        help="Only sample frames where commentary contains motion-related words.",
    )
    return parser.parse_args()


def main() -> None:
    jsonl_path, summary_path, counts = audit(parse_args())
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {summary_path}")
    if counts:
        print("Top counts:")
        for key, value in counts.most_common(8):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
