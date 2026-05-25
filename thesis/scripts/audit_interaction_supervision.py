#!/usr/bin/env python3
"""Audit interaction-aware temporal supervision labels and cleaned commentary.

This checks the next-version training signal added for temporal methods:

- box + future-waypoint interaction labels
- removal of stale "junction is clear / vehicles are stopped" commentary
- appended interaction-aware commentary

It does not run the simulator or load the model.
"""

import argparse
import gzip
import json
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import ujson

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simlingo_training.dataloader.dataset_driving import Data_Driving


LABEL_NAMES = (
    "moving_front_or_path_actor",
    "moving_lateral_cross_actor",
    "stopped_blocking_actor",
    "dynamic_yield_interaction",
)


def load_json_gz(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return ujson.load(handle)
    except (OSError, ujson.JSONDecodeError):
        return None


def route_dirs(data_root: Path) -> List[Path]:
    return sorted((data_root / "data" / "simlingo").glob("*/*/*/Town*"))


def commentary_path_for_measurement(measurement_path: Path, data_root: Path) -> Path:
    relative = measurement_path.relative_to(data_root)
    parts = list(relative.parts)
    parts[0] = "commentary"
    parts[-2] = "commentary"
    return data_root.joinpath(*parts)


def get_commentary_text(commentary_data: Optional[Any]) -> str:
    if commentary_data is None:
        return ""
    if isinstance(commentary_data, str):
        return commentary_data
    if isinstance(commentary_data, dict):
        value = commentary_data.get("commentary")
        if isinstance(value, str):
            return value
    return ""


def measurement_path(route_dir: Path, frame_idx: int) -> Path:
    return route_dir / "measurements" / f"{frame_idx:04}.json.gz"


def box_path(route_dir: Path, frame_idx: int) -> Path:
    return route_dir / "boxes" / f"{frame_idx:04}.json.gz"


def rgb_path(route_dir: Path, frame_idx: int) -> Path:
    return route_dir / "rgb" / f"{frame_idx:04}.jpg"


def waypoints_from_measurements(measurements: List[Dict[str, Any]]) -> np.ndarray:
    """Return future ego positions in the current ego frame."""
    origin_matrix = np.asarray(measurements[0]["ego_matrix"], dtype=np.float32)[:3]
    origin_translation = origin_matrix[:, 3:4]
    origin_rotation = origin_matrix[:, :3]

    waypoints = []
    for measurement in measurements:
        waypoint = np.asarray(measurement["ego_matrix"], dtype=np.float32)[:3, 3:4]
        waypoint_ego_frame = origin_rotation.T @ (waypoint - origin_translation)
        waypoints.append(waypoint_ego_frame[:2, 0])
    return np.asarray(waypoints[1:-1], dtype=np.float32)


def future_measurements(route_dir: Path, frame_idx: int, pred_len: int) -> List[Dict[str, Any]]:
    measurements = []
    for offset in range(pred_len + 2):
        path = measurement_path(route_dir, frame_idx + offset)
        data = load_json_gz(path)
        if isinstance(data, dict):
            measurements.append(data)
        elif measurements:
            measurements.append(measurements[-1])
    return measurements


def future_motion_stats(waypoints: np.ndarray) -> Dict[str, Optional[float]]:
    if waypoints.shape[0] < 4:
        return {
            "early_disp": None,
            "late_disp": None,
            "min_future_disp": None,
            "expert_slows_or_waits": False,
        }

    displacements = np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=-1)
    early_disp = float(np.mean(displacements[: min(3, len(displacements))]))
    late_disp = float(np.mean(displacements[-min(3, len(displacements)):]))
    min_future_disp = float(displacements[min(1, len(displacements) - 1):].min())
    expert_slows_or_waits = bool(Data_Driving._expert_slows_or_waits_from_waypoints(waypoints))
    return {
        "early_disp": round(early_disp, 4),
        "late_disp": round(late_disp, 4),
        "min_future_disp": round(min_future_disp, 4),
        "expert_slows_or_waits": expert_slows_or_waits,
    }


def actor_details(box_data: Optional[Any], waypoints: np.ndarray, current_measurement: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    boxes = box_data if isinstance(box_data, list) else []
    details = []
    future_path_turns = Data_Driving._future_path_turns(waypoints)
    ego_box = next((box for box in boxes if box.get("class") == "ego_info"), None)
    vehicle_hazard_id = current_measurement.get("vehicle_affecting_id") if current_measurement else None
    walker_close_id = current_measurement.get("walker_close_id") if current_measurement else None
    speed_reduced_id = current_measurement.get("speed_reduced_by_obj_id") if current_measurement else None
    vehicle_hazard = bool(current_measurement and Data_Driving._truthy(current_measurement.get("vehicle_hazard")))
    walker_hazard = bool(current_measurement and Data_Driving._truthy(current_measurement.get("walker_hazard")))
    speed_reduced_type = str(current_measurement.get("speed_reduced_by_obj_type", "") if current_measurement else "").lower()

    for box in boxes:
        if box.get("class") not in ("car", "walker"):
            continue
        pos = box.get("position", [0, 0, 0])
        if len(pos) < 2:
            continue
        pos_xy = np.asarray(pos[:2], dtype=np.float32)
        dist = float(np.linalg.norm(pos_xy))
        if dist > 35.0:
            continue
        path_distance = float(np.linalg.norm(waypoints[: min(len(waypoints), 12)] - pos_xy, axis=-1).min()) if len(waypoints) else None
        speed = float(box.get("speed", 0.0))
        actor_id = Data_Driving._box_id(box)
        vehicle_hazard_actor = bool(
            box.get("class") == "car"
            and vehicle_hazard
            and (
                Data_Driving._ids_match(actor_id, vehicle_hazard_id)
                or ("vehicle" in speed_reduced_type and Data_Driving._ids_match(actor_id, speed_reduced_id))
            )
        )
        walker_hazard_actor = bool(
            box.get("class") == "walker"
            and walker_hazard
            and (
                Data_Driving._ids_match(actor_id, walker_close_id)
                or Data_Driving._ids_match(actor_id, speed_reduced_id)
                or dist < 15.0
            )
        )
        details.append(
            {
                "id": actor_id,
                "class": box.get("class"),
                "x": round(float(pos[0]), 3),
                "y": round(float(pos[1]), 3),
                "distance": round(dist, 3),
                "path_distance": round(path_distance, 3) if path_distance is not None else None,
                "speed": round(speed, 3),
                "num_points": box.get("num_points"),
                "visible_for_supervision": bool(Data_Driving._actor_visible_for_supervision(box)),
                "vehicle_hazard_actor": vehicle_hazard_actor,
                "walker_hazard_actor": walker_hazard_actor,
                "points_towards_junction": bool(Data_Driving._actor_points_towards_junction(ego_box, box)),
                "near_path": bool(
                    path_distance is not None
                    and (path_distance < 4.5 or (pos[0] > 0.0 and abs(pos[1]) < 4.0))
                ),
                "side": bool(abs(pos[1]) >= 3.0),
                "same_road_as_ego": box.get("same_road_as_ego"),
                "same_direction_as_ego": box.get("same_direction_as_ego"),
                "lane_relative_to_ego": box.get("lane_relative_to_ego"),
                "vehicle_cuts_in": box.get("vehicle_cuts_in"),
                "turn_path_conflict_candidate": bool(
                    future_path_turns
                    and path_distance is not None
                    and path_distance < 2.5
                    and abs(pos[1]) > 1.0
                    and speed > 0.5
                ),
            }
        )
    details.sort(key=lambda item: (item["path_distance"] if item["path_distance"] is not None else 999.0, item["distance"]))
    return details[:8]


def yield_decision(scene_facts: Dict[str, Any], labels: np.ndarray) -> Dict[str, Any]:
    rule_or_static_stop = bool(
        scene_facts.get("verified_traffic_stop")
        or scene_facts.get("verified_stop_sign")
        or scene_facts.get("verified_construction")
    )
    pre_filter_candidate = bool(
        scene_facts.get("strict_dynamic_conflict_actor")
        and scene_facts.get("expert_slows_or_waits")
    )
    accepted = bool(labels[3] > 0.0)
    rejected_by_rule_or_static_stop = bool(pre_filter_candidate and rule_or_static_stop and not accepted)
    rejected_by_lead_following = bool(
        pre_filter_candidate
        and not rule_or_static_stop
        and scene_facts.get("lead_confounds_yield")
        and not accepted
    )
    rejected_other = bool(
        pre_filter_candidate
        and not accepted
        and not rejected_by_rule_or_static_stop
        and not rejected_by_lead_following
    )
    if accepted:
        status = "accepted_yield"
    elif rejected_by_rule_or_static_stop:
        status = "rejected_rule_or_static_stop"
    elif rejected_by_lead_following:
        status = "rejected_lead_following"
    elif rejected_other:
        status = "rejected_other"
    else:
        status = "not_yield_candidate"
    return {
        "pre_filter_yield_candidate": pre_filter_candidate,
        "accepted_yield": accepted,
        "rule_or_static_stop": rule_or_static_stop,
        "rejected_by_rule_or_static_stop": rejected_by_rule_or_static_stop,
        "rejected_by_lead_following": rejected_by_lead_following,
        "rejected_other": rejected_other,
        "status": status,
    }


def should_keep(labels: np.ndarray, args: argparse.Namespace, scene_facts: Optional[Dict[str, Any]] = None) -> bool:
    if args.sample_mode == "all":
        return True
    if args.sample_mode == "interaction":
        return bool(labels.any())
    if args.sample_mode == "yield":
        return bool(labels[3] > 0.0)
    if args.sample_mode == "yield_review":
        if scene_facts is None:
            return bool(labels[3] > 0.0)
        decision = yield_decision(scene_facts, labels)
        return bool(decision["pre_filter_yield_candidate"] or decision["accepted_yield"])
    raise ValueError(f"Unknown sample mode: {args.sample_mode}")


def audit(args: argparse.Namespace):
    random.seed(args.seed)
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "frames"
    if args.copy_rgb:
        frame_dir.mkdir(parents=True, exist_ok=True)

    routes = route_dirs(data_root)
    if not args.include_validation:
        routes = [route for route in routes if "routes_training" in str(route)]
    if args.route_contains:
        routes = [route for route in routes if args.route_contains in str(route)]
    random.shuffle(routes)
    routes = routes[: args.max_routes]

    helper = object.__new__(Data_Driving)
    counts = Counter()
    records = []

    for route_dir in routes:
        measurement_files = sorted((route_dir / "measurements").glob("*.json.gz"))
        if args.shuffle_frames:
            random.shuffle(measurement_files)

        for measurement_file in measurement_files[:: args.frame_step]:
            if len(records) >= args.max_samples:
                break

            frame_idx = int(measurement_file.name.split(".")[0])
            if args.frame_start is not None and frame_idx < args.frame_start:
                continue
            if args.frame_end is not None and frame_idx > args.frame_end:
                continue

            measurements = future_measurements(route_dir, frame_idx, args.pred_len)
            if len(measurements) < 4:
                counts["skipped_missing_future"] += 1
                continue

            waypoints = waypoints_from_measurements(measurements)
            boxes = load_json_gz(box_path(route_dir, frame_idx))
            commentary_path = commentary_path_for_measurement(measurement_file, data_root)
            original_commentary = get_commentary_text(load_json_gz(commentary_path))
            has_commentary = bool(original_commentary.strip())
            if args.require_commentary and not has_commentary:
                counts["skipped_missing_commentary"] += 1
                continue

            if not isinstance(boxes, list):
                counts["skipped_missing_boxes"] += 1
                continue

            scene_facts = helper._build_interaction_scene_facts(
                boxes,
                waypoints,
                measurements[0],
                future_measurements=measurements,
            )
            labels, interaction_text = helper._get_actor_motion_labels(
                box_path(route_dir, frame_idx),
                waypoints,
                current_measurement=measurements[0],
                scene_facts=scene_facts,
            )
            if labels is None:
                counts["skipped_missing_boxes"] += 1
                continue

            decision = yield_decision(scene_facts, labels)
            counts["frames_with_boxes"] += 1
            counts["scanned_pre_filter_yield_candidate"] += int(decision["pre_filter_yield_candidate"])
            counts["scanned_accepted_yield"] += int(decision["accepted_yield"])
            counts["scanned_rejected_rule_or_static_stop"] += int(decision["rejected_by_rule_or_static_stop"])
            counts["scanned_rejected_lead_following"] += int(decision["rejected_by_lead_following"])
            counts["scanned_rejected_other"] += int(decision["rejected_other"])

            cleaned_commentary = helper._remove_conflicting_junction_claims(
                original_commentary,
                labels,
                scene_facts,
            )
            if not should_keep(labels, args, scene_facts):
                counts["skipped_sample_mode"] += 1
                continue

            final_commentary = cleaned_commentary
            if interaction_text and has_commentary:
                final_commentary = final_commentary.rstrip(".") + ". " + interaction_text

            label_dict = {name: bool(value > 0.0) for name, value in zip(LABEL_NAMES, labels)}
            removed_text = original_commentary != cleaned_commentary
            record = {
                "route": str(route_dir.relative_to(data_root)),
                "frame": frame_idx,
                "measurement": str(measurement_file.relative_to(data_root)),
                "rgb": str(rgb_path(route_dir, frame_idx).relative_to(data_root)),
                "boxes": str(box_path(route_dir, frame_idx).relative_to(data_root)),
                "labels": label_dict,
                "label_vector": [float(value) for value in labels.tolist()],
                "has_commentary": has_commentary,
                "interaction_sample_weight_mask": float(labels[3] > 0.0),
                "yield_decision": decision,
                "interaction_text": interaction_text,
                "original_commentary": original_commentary,
                "cleaned_commentary": cleaned_commentary,
                "final_commentary": final_commentary,
                "removed_conflicting_claim": removed_text,
                "future_motion": future_motion_stats(waypoints),
                "future_path_turns": Data_Driving._future_path_turns(waypoints),
                "scene_facts": {
                    "dynamic_conflict_actor": bool(scene_facts.get("dynamic_conflict_actor")),
                    "strict_dynamic_conflict_actor": bool(scene_facts.get("strict_dynamic_conflict_actor")),
                    "metadata_dynamic_conflict_actor": bool(scene_facts.get("metadata_dynamic_conflict_actor")),
                    "lead_confounds_yield": bool(scene_facts.get("lead_confounds_yield")),
                    "ego_near_junction": bool(scene_facts.get("ego_near_junction")),
                    "expert_slows_or_waits": bool(scene_facts.get("expert_slows_or_waits")),
                    "expert_slows_or_waits_from_waypoints": bool(scene_facts.get("expert_slows_or_waits_from_waypoints")),
                    "expert_slows_or_waits_from_metadata": bool(scene_facts.get("expert_slows_or_waits_from_metadata")),
                    "moving_actor_near_path": bool(scene_facts.get("moving_actor_near_path")),
                    "moving_junction_actor": bool(scene_facts.get("moving_junction_actor")),
                    "verified_green_light": bool(scene_facts.get("verified_green_light")),
                    "verified_traffic_stop": bool(scene_facts.get("verified_traffic_stop")),
                    "verified_stop_sign": bool(scene_facts.get("verified_stop_sign")),
                    "verified_construction": bool(scene_facts.get("verified_construction")),
                    "verified_lead_following": bool(scene_facts.get("verified_lead_following")),
                    "verified_black_lead_following": bool(scene_facts.get("verified_black_lead_following")),
                    "vehicle_hazard_actor": bool(scene_facts.get("vehicle_hazard_actor")),
                    "walker_hazard_actor": bool(scene_facts.get("walker_hazard_actor")),
                    "visible_actor_count": int(scene_facts.get("visible_actor_count", 0)),
                },
                "measurement_facts": {
                    "speed": measurements[0].get("speed"),
                    "target_speed": measurements[0].get("target_speed"),
                    "control_brake": measurements[0].get("control_brake"),
                    "light_hazard": measurements[0].get("light_hazard"),
                    "stop_sign_hazard": measurements[0].get("stop_sign_hazard"),
                    "vehicle_hazard": measurements[0].get("vehicle_hazard"),
                    "vehicle_affecting_id": measurements[0].get("vehicle_affecting_id"),
                    "walker_hazard": measurements[0].get("walker_hazard"),
                    "walker_close_id": measurements[0].get("walker_close_id"),
                    "speed_reduced_by_obj_type": measurements[0].get("speed_reduced_by_obj_type"),
                    "speed_reduced_by_obj_id": measurements[0].get("speed_reduced_by_obj_id"),
                    "speed_reduced_by_obj_distance": measurements[0].get("speed_reduced_by_obj_distance"),
                },
                "actors": actor_details(boxes, waypoints, measurements[0]),
            }

            if args.copy_rgb:
                source_rgb = rgb_path(route_dir, frame_idx)
                if source_rgb.is_file():
                    frame_name = f"{str(route_dir.relative_to(data_root)).replace('/', '__')}__frame_{frame_idx:04}.jpg"
                    copied = frame_dir / frame_name
                    shutil.copy2(source_rgb, copied)
                    record["copied_rgb"] = str(copied)

            records.append(record)
            counts["samples_written"] += 1
            counts["samples_with_commentary"] += int(has_commentary)
            counts["removed_conflicting_claim"] += int(removed_text)
            for name, value in label_dict.items():
                counts[name] += int(value)
            for name, value in record["scene_facts"].items():
                counts[f"scene_{name}"] += int(value)
            counts[f"yield_status_{decision['status']}"] += 1

            final_lower = final_commentary.lower()
            tracked_stale_claims = (
                "other vehicles are stopped at the junction",
                "vehicles are stopped at the junction",
                "traffic is stopped at the junction",
                "junction is clear",
                "clear junction",
            )
            counts["final_stale_junction_phrase"] += int(
                any(claim in final_lower for claim in tracked_stale_claims)
            )
            counts["final_black_car_following_phrase"] += int(
                "black car" in final_lower
                and any(phrase in final_lower for phrase in ("front", "behind", "stay behind", "follow the"))
            )
            counts["final_green_light_phrase"] += int("traffic light is green" in final_lower)
            counts["final_red_light_phrase"] += int(
                "red traffic light" in final_lower or "traffic light is red" in final_lower
            )
            counts["final_speed_up_phrase"] += int(
                Data_Driving._sentence_says_to_enter_or_speed_up(final_lower)
            )
            counts["final_yield_text"] += int("ego should yield until the path is clear" in final_lower)

        if len(records) >= args.max_samples:
            break

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"interaction_supervision_audit_{stamp}.jsonl"
    summary_path = output_dir / f"interaction_supervision_audit_{stamp}_summary.txt"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Interaction supervision audit\n")
        handle.write(f"data_root: {data_root}\n")
        handle.write(f"routes_seen: {len(routes)}\n")
        handle.write(f"sample_mode: {args.sample_mode}\n")
        handle.write(f"max_samples: {args.max_samples}\n")
        handle.write(f"frame_step: {args.frame_step}\n")
        handle.write(f"seed: {args.seed}\n")
        handle.write(f"jsonl: {jsonl_path}\n")
        if args.copy_rgb:
            handle.write(f"copied frames: {frame_dir}\n")
        handle.write("\n")
        for key, value in counts.most_common():
            handle.write(f"{key}: {value}\n")

    return jsonl_path, summary_path, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Staged dataset root, usually $STAGED_DATA_ROOT.")
    parser.add_argument("--output-dir", default="thesis/results/interaction_supervision_audit")
    parser.add_argument("--max-routes", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9876)
    parser.add_argument("--pred-len", type=int, default=11)
    parser.add_argument("--sample-mode", choices=("all", "interaction", "yield", "yield_review"), default="yield")
    parser.add_argument("--require-commentary", action="store_true")
    parser.add_argument("--route-contains", default="")
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument("--shuffle-frames", action="store_true")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--copy-rgb", action="store_true")
    return parser.parse_args()


def main() -> None:
    jsonl_path, summary_path, counts = audit(parse_args())
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {summary_path}")
    for key, value in counts.most_common(8):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
