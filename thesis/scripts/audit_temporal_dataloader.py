#!/usr/bin/env python3
"""Audit temporal sample alignment in the SimLingo training dataloader.

This checks the data plumbing that matters for temporal models:

- history frame order and stride
- current frame alignment
- future measurement availability for waypoint supervision
- optional loaded RGB tensor temporal shape

It does not run the model and does not use CARLA.
"""

import argparse
import gzip
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))

import numpy as np
import ujson
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import simlingo_training.config  # Registers structured Hydra configs.
import simlingo_training.dataloader.dataset_base as dataset_base_module
from simlingo_training.dataloader.dataset_dreamer import Data_Dreamer
from simlingo_training.dataloader.dataset_driving import Data_Driving


def decode_path(path_value: Any) -> str:
    if isinstance(path_value, bytes):
        return path_value.decode("utf-8")
    if isinstance(path_value, np.bytes_):
        return path_value.decode("utf-8")
    return str(path_value)


def frame_index(path_value: Any) -> int:
    name = Path(decode_path(path_value)).name
    return int(name.split(".", 1)[0])


def read_json_gz(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = ujson.load(handle)
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, OSError, ValueError, ujson.JSONDecodeError):
        return None


def ego_xy(measurement: Dict[str, Any]) -> Optional[List[float]]:
    matrix = measurement.get("ego_matrix")
    if not matrix:
        return None
    try:
        return [float(matrix[0][3]), float(matrix[1][3])]
    except (IndexError, TypeError, ValueError):
        return None


def future_speed_summary(route_dir: Path, current_idx: int, pred_len: int) -> Dict[str, Any]:
    positions = []
    speeds = []
    for idx in range(current_idx, current_idx + pred_len + 1):
        measurement = read_json_gz(route_dir / "measurements" / f"{idx:04}.json.gz")
        if not measurement:
            break
        position = ego_xy(measurement)
        if position is not None:
            positions.append(position)
        speed = measurement.get("speed")
        if isinstance(speed, (int, float)):
            speeds.append(float(speed))

    displacement_speeds = []
    for before, after in zip(positions, positions[1:]):
        distance = float(np.hypot(after[0] - before[0], after[1] - before[1]))
        displacement_speeds.append(distance / 0.25)

    return {
        "measurement_speeds": [round(speed, 3) for speed in speeds[: pred_len + 1]],
        "displacement_speeds": [round(speed, 3) for speed in displacement_speeds],
        "mean_displacement_speed": (
            round(float(np.mean(displacement_speeds)), 3) if displacement_speeds else None
        ),
    }


def dataset_kwargs(cfg: Any, data_root: Optional[str], bucket_root: Optional[str], max_routes: Optional[int]) -> Dict[str, Any]:
    data_module_cfg = OmegaConf.to_container(cfg.data_module, resolve=True)
    base_cfg = OmegaConf.to_container(cfg.data_module.base_dataset, resolve=True)
    data_module_cfg.pop("base_dataset", None)
    kwargs = {**data_module_cfg, **base_cfg}
    kwargs.pop("_target_", None)
    if data_root:
        kwargs["data_path"] = data_root
    if bucket_root:
        kwargs["bucket_path"] = bucket_root
    if max_routes is not None:
        kwargs["max_routes"] = max_routes
    return kwargs


def make_dataset(name: str, cfg: Any, args: argparse.Namespace):
    kwargs = dataset_kwargs(cfg, args.data_root, args.bucket_root, args.max_routes)
    if name == "driving":
        return Data_Driving(split=args.split, bucket_name=args.bucket, **kwargs)
    if name == "dreamer":
        return Data_Dreamer(split=args.split, bucket_name=args.bucket, **kwargs)
    raise ValueError(f"Unknown dataset: {name}")


def choose_indices(length: int, max_samples: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    if length <= max_samples:
        return list(range(length))
    return sorted(rng.sample(range(length), max_samples))


def temporal_image_diffs(item: Any) -> List[float]:
    rgb = np.asarray(item.image_ff).astype(np.float32)
    if rgb.ndim != 4 or rgb.shape[0] < 2:
        return []
    return [
        round(float(np.mean(np.abs(rgb[idx + 1] - rgb[idx]))), 3)
        for idx in range(rgb.shape[0] - 1)
    ]


def inspect_sample(dataset: Any, dataset_name: str, index: int, load_item: bool) -> Dict[str, Any]:
    sample_start = int(dataset.sample_start[index])
    hist_len = int(dataset.hist_len)
    history_stride = int(dataset.history_stride)
    history_span = int(dataset.history_span)
    pred_len = int(dataset.pred_len)
    current_idx = sample_start + history_span

    image_paths = [decode_path(path) for path in dataset.images[index]]
    box_paths = [decode_path(path) for path in dataset.boxes[index]]
    measurements_dir = Path(decode_path(dataset.measurements[index][0]))
    route_dir = measurements_dir.parent

    expected_history_indices = [sample_start + idx * history_stride for idx in range(hist_len)]
    image_indices = [frame_index(path) for path in image_paths]
    box_indices = [frame_index(path) for path in box_paths]
    future_indices = list(range(current_idx + 1, current_idx + pred_len + 1))
    current_measurement = measurements_dir / f"{current_idx:04}.json.gz"
    future_measurements = [measurements_dir / f"{idx:04}.json.gz" for idx in future_indices]

    checks = {
        "history_indices_match_images": image_indices == expected_history_indices,
        "history_indices_match_boxes": box_indices == expected_history_indices,
        "last_history_is_current": image_indices[-1] == current_idx if image_indices else False,
        "history_stride_ok": all(
            (after - before) == history_stride
            for before, after in zip(image_indices, image_indices[1:])
        ),
        "history_images_exist": all(Path(path).is_file() for path in image_paths),
        "history_boxes_exist": all(Path(path).is_file() for path in box_paths),
        "current_measurement_exists": current_measurement.is_file(),
        "future_measurements_exist": all(path.is_file() for path in future_measurements),
        "history_same_route": all(Path(path).parent.parent == route_dir for path in image_paths + box_paths),
    }

    record = {
        "dataset": dataset_name,
        "index": index,
        "route_dir": str(route_dir),
        "hist_len": hist_len,
        "history_stride": history_stride,
        "history_span": history_span,
        "pred_len": pred_len,
        "sample_start": sample_start,
        "current_idx": current_idx,
        "expected_history_indices": expected_history_indices,
        "image_indices": image_indices,
        "box_indices": box_indices,
        "future_indices": future_indices,
        "current_measurement": str(current_measurement),
        "history_tick_offsets_from_current": [idx - current_idx for idx in image_indices],
        "checks": checks,
        "future_motion": future_speed_summary(route_dir, current_idx, pred_len),
    }

    if load_item:
        item = dataset[index]
        record["loaded_item"] = {
            "image_ff_shape": list(np.asarray(item.image_ff).shape),
            "waypoints_shape": list(np.asarray(item.waypoints).shape),
            "route_shape": list(np.asarray(item.path).shape),
            "target_points": np.asarray(item.target_points).round(3).tolist(),
            "speed": round(float(item.speed), 3),
            "measurement_path": item.measurement_path,
            "temporal_image_mean_abs_diff": temporal_image_diffs(item),
        }
        record["checks"]["loaded_temporal_shape_ok"] = (
            list(np.asarray(item.image_ff).shape)[0] == hist_len
        )
        record["checks"]["loaded_waypoints_shape_ok"] = (
            list(np.asarray(item.waypoints).shape) == [pred_len - 1, 2]
        )

    record["passed"] = all(record["checks"].values())
    return record


def write_summary(records: Iterable[Dict[str, Any]], summary_path: Path, args: argparse.Namespace, jsonl_path: Path):
    records = list(records)
    failures = [record for record in records if not record["passed"]]
    check_counts = Counter()
    for record in records:
        for check_name, passed in record["checks"].items():
            if not passed:
                check_counts[check_name] += 1

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Temporal dataloader alignment audit\n")
        handle.write(f"experiment: {args.experiment}\n")
        handle.write(f"split: {args.split}\n")
        handle.write(f"bucket: {args.bucket}\n")
        handle.write(f"dataset: {args.dataset}\n")
        handle.write(f"data_root: {args.data_root}\n")
        handle.write(f"bucket_root: {args.bucket_root}\n")
        handle.write(f"max_routes: {args.max_routes}\n")
        handle.write(f"load_item: {args.load_item}\n")
        handle.write(f"jsonl: {jsonl_path}\n\n")
        handle.write(f"samples_written: {len(records)}\n")
        handle.write(f"samples_failed: {len(failures)}\n")
        for check_name, count in sorted(check_counts.items()):
            handle.write(f"{check_name}: {count}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="temporal_qformer_v4_nogate")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--bucket-root", default=None)
    parser.add_argument("--dataset", choices=("driving", "dreamer", "all"), default="all")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--bucket", default="all")
    parser.add_argument("--max-routes", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=9876)
    parser.add_argument("--load-item", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="thesis/results/temporal_dataloader_audit",
        help="Directory where JSONL and summary files are written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_dir = REPO_DIR
    config_dir = repo_dir / "simlingo_training" / "config"
    dataset_base_module.get_original_cwd = lambda: str(repo_dir)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name="config", overrides=[f"experiment={args.experiment}"])

    datasets = ["driving", "dreamer"] if args.dataset == "all" else [args.dataset]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"temporal_dataloader_audit_{timestamp}.jsonl"
    summary_path = output_dir / f"temporal_dataloader_audit_{timestamp}_summary.txt"

    records = []
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for dataset_name in datasets:
            dataset = make_dataset(dataset_name, cfg, args)
            if len(dataset) == 0:
                print(f"WARNING: {dataset_name} dataset has 0 samples for this audit configuration.")
            indices = choose_indices(len(dataset), args.max_samples, args.seed)
            for index in indices:
                record = inspect_sample(dataset, dataset_name, index, args.load_item)
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    write_summary(records, summary_path, args, jsonl_path)
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {summary_path}")
    failed = sum(1 for record in records if not record["passed"])
    if failed:
        raise SystemExit(f"{failed} sampled temporal dataloader records failed checks")


if __name__ == "__main__":
    main()
