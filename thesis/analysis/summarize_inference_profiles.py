#!/usr/bin/env python3
"""Summarize SimLingo agent-side inference profiling JSON files."""


import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PREFERRED_ORDER = [
    "Baseline",
    "Q-former v8 fresh-LoRA",
    "Delta v5 fresh-LoRA",
]


def load_profiles(root: Path):
    grouped = defaultdict(lambda: {"latencies": [], "peak_memory": 0.0, "files": []})
    for profile_path in sorted(root.rglob("*_profile.json")):
        with profile_path.open("r", encoding="utf-8") as f:
            profile = json.load(f)
        variant = profile.get("variant") or "unknown"
        records = profile.get("records") or []
        latencies = [
            float(record["latency_ms"])
            for record in records
            if not record.get("warmup", False) and record.get("latency_ms") is not None
        ]
        if not latencies and profile.get("mean_latency_ms") is not None:
            latencies = [float(profile["mean_latency_ms"])]

        grouped[variant]["latencies"].extend(latencies)
        grouped[variant]["peak_memory"] = max(
            grouped[variant]["peak_memory"],
            float(profile.get("peak_gpu_memory_allocated_gb") or 0.0),
        )
        grouped[variant]["files"].append(str(profile_path))
    return grouped


def ordered_variants(grouped):
    known = [variant for variant in PREFERRED_ORDER if variant in grouped]
    extra = sorted(variant for variant in grouped if variant not in PREFERRED_ORDER)
    return known + extra


def format_summary(grouped):
    lines = []
    lines.append("Inference profiling summary")
    lines.append("===========================")
    lines.append("")
    lines.append("Variant | Files | Calls | Mean latency (ms) | p95 latency (ms) | Peak GPU memory (GB)")
    lines.append("--- | ---: | ---: | ---: | ---: | ---:")

    for variant in ordered_variants(grouped):
        latencies = grouped[variant]["latencies"]
        if latencies:
            mean_latency = float(np.mean(latencies))
            p95_latency = float(np.percentile(latencies, 95))
            calls = len(latencies)
            mean_text = f"{mean_latency:.1f}"
            p95_text = f"{p95_latency:.1f}"
        else:
            calls = 0
            mean_text = "TODO"
            p95_text = "TODO"
        peak_memory = grouped[variant]["peak_memory"]
        peak_text = f"{peak_memory:.2f}" if peak_memory else "TODO"
        file_count = len(grouped[variant]["files"])
        lines.append(f"{variant} | {file_count} | {calls} | {mean_text} | {p95_text} | {peak_text}")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing *_profile.json files.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output text file.")
    args = parser.parse_args()

    grouped = load_profiles(args.root)
    if not grouped:
        raise SystemExit(f"No profile JSON files found under {args.root}")

    summary = format_summary(grouped)
    print(summary)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
