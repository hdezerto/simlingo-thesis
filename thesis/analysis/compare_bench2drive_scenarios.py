import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from contextlib import redirect_stdout


SEEDS = ("1", "2", "3")
IGNORED_INFRACTIONS = {"min_speed_infractions"}



def route_index_from_filename(filename):
    match = re.search(r"_(\d+)\.xml$", filename)
    if not match:
        return None
    return int(match.group(1))


def parse_split(split_dir):
    routes = {}
    for filename in sorted(os.listdir(split_dir)):
        if not filename.endswith(".xml"):
            continue
        index = route_index_from_filename(filename)
        if index is None:
            continue
        path = os.path.join(split_dir, filename)
        tree = ET.parse(path)
        route = tree.getroot().find("route")
        if route is None:
            continue
        scenarios = route.findall(".//scenario")
        if len(scenarios) != 1:
            raise ValueError(f"Expected one scenario in {path}, found {len(scenarios)}")
        scenario = scenarios[0]
        scenario_type = scenario.attrib.get("type", "Unknown")
        scenario_name = scenario.attrib.get("name", scenario_type)
        routes[index] = {
            "index": index,
            "res_id": f"{index:03d}",
            "xml": filename,
            "route_id": route.attrib.get("id", f"UNKNOWN_{index:03d}"),
            "town": route.attrib.get("town", "Unknown"),
            "scenario_type": scenario_type,
            "scenario_name": scenario_name,
        }
    return routes


def success_and_cause(record):
    status = record.get("status", "Unknown")
    infractions = record.get("infractions", {})

    if status in {"Completed", "Perfect"}:
        for key, values in infractions.items():
            if key in IGNORED_INFRACTIONS:
                continue
            if values:
                return False, key
        return True, "success"

    return False, status


def read_attempt(base_folder, seed, res_id):
    result_path = os.path.join(base_folder, str(seed), "res", f"{res_id}_res.json")
    if not os.path.exists(result_path):
        return {
            "attempted": False,
            "success": False,
            "cause": "missing_result",
            "status": "missing_result",
            "path": result_path,
        }

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return {
            "attempted": False,
            "success": False,
            "cause": f"invalid_json:{type(exc).__name__}",
            "status": "invalid_json",
            "path": result_path,
        }

    records = data.get("_checkpoint", {}).get("records", [])
    if not records:
        return {
            "attempted": False,
            "success": False,
            "cause": "missing_record",
            "status": "missing_record",
            "path": result_path,
        }

    record = records[0]
    success, cause = success_and_cause(record)
    return {
        "attempted": True,
        "success": success,
        "cause": cause,
        "status": record.get("status", "Unknown"),
        "score": record.get("scores", {}),
        "path": result_path,
    }


def collect_run(base_folder, routes):
    route_stats = {}
    scenario_stats = defaultdict(lambda: {
        "expected": 0,
        "attempted": 0,
        "failed": 0,
        "causes": Counter(),
        "missing": 0,
    })
    for index, route in routes.items():
        attempts = {}
        failures = 0
        attempted = 0
        causes = Counter()
        missing = 0

        for seed in SEEDS:
            attempt = read_attempt(base_folder, seed, route["res_id"])
            attempts[seed] = attempt
            if attempt["attempted"]:
                attempted += 1
                if not attempt["success"]:
                    failures += 1
                    causes[attempt["cause"]] += 1
            else:
                missing += 1

            scenario = scenario_stats[route["scenario_name"]]
            scenario["expected"] += 1
            if attempt["attempted"]:
                scenario["attempted"] += 1
                if not attempt["success"]:
                    scenario["failed"] += 1
                    scenario["causes"][attempt["cause"]] += 1
            else:
                scenario["missing"] += 1

        route_stats[index] = {
            **route,
            "attempts": attempts,
            "expected": len(SEEDS),
            "attempted": attempted,
            "failed": failures,
            "missing": missing,
            "causes": causes,
        }

    return {
        "routes": route_stats,
        "scenarios": scenario_stats,
    }


def pct(numerator, denominator):
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def fmt_count(stats):
    suffix = "" if stats["missing"] == 0 else f" ({stats['missing']} missing)"
    return f"{stats['failed']}/{stats['expected']}{suffix}"


def main_cause(stats):
    if not stats["causes"]:
        return "-"
    cause, count = stats["causes"].most_common(1)[0]
    return f"{cause} ({count})"


def transition_label(base_failed, cand_failed):
    if base_failed > 0 and cand_failed == 0:
        return "fixed"
    if base_failed == 0 and cand_failed > 0:
        return "regressed"
    if base_failed > 0 and cand_failed > 0:
        if cand_failed < base_failed:
            return "improved_but_not_fixed"
        if cand_failed > base_failed:
            return "worse_but_both_fail"
        return "failed_both_same_count"
    return "passed_both"


def route_transition_label(base_route, cand_route):
    if base_route["missing"] > 0 or cand_route["missing"] > 0:
        return "has_missing_attempt"
    return transition_label(base_route["failed"], cand_route["failed"])


def render_candidate_sort_key(label, pair):
    base_route, cand_route = pair
    base_failed = base_route["failed"]
    cand_failed = cand_route["failed"]

    if label == "fixed":
        severity = base_failed - cand_failed
    elif label in {"regressed", "worse_but_both_fail"}:
        severity = cand_failed - base_failed
    elif label == "failed_both_same_count":
        severity = base_failed
    else:
        severity = abs(cand_failed - base_failed)

    return (
        -severity,
        base_route["scenario_name"],
        base_route["res_id"],
    )


def print_table(rows, headers, widths=None):
    rows = [tuple(str(value) for value in row) for row in rows]
    headers = tuple(str(header) for header in headers)
    if widths is None:
        widths = [len(header) for header in headers]
    else:
        widths = list(widths)
    for i, header in enumerate(headers):
        widths[i] = max(widths[i], len(header))
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    print(" | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)).rstrip())
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(f"{value:<{w}}" for value, w in zip(row, widths)).rstrip())


def hardest_scenario_rows(run, limit=12):
    rows = []
    for scenario, stats in run["scenarios"].items():
        if stats["expected"] == 0:
            continue
        rows.append((
            scenario,
            fmt_count(stats),
            f"{pct(stats['failed'], stats['expected']):5.1f}%",
            main_cause(stats),
            stats["failed"],
            stats["expected"],
        ))
    rows.sort(key=lambda row: (row[4] / row[5] if row[5] else 0.0, row[4]), reverse=True)
    return rows[:limit]


def print_hardest_scenarios(title, run):
    print(title)
    rows = [(r[0], r[1], r[2], r[3]) for r in hardest_scenario_rows(run)]
    print_table(rows, ("scenario", "failures", "fail %", "main cause"), (44, 20, 7, 32))
    print()


def systematic_failure_rows(run):
    rows = []
    for route in run["routes"].values():
        if route["missing"] == 0 and route["failed"] == len(SEEDS):
            rows.append((
                route["route_id"],
                route["res_id"],
                route["scenario_name"],
                main_cause(route),
            ))
    rows.sort(key=lambda row: (row[2], row[0]))
    return rows


def failure_cause_counts(run):
    counts = Counter()
    for route in run["routes"].values():
        for attempt in route["attempts"].values():
            if attempt["attempted"]:
                if not attempt["success"]:
                    counts[attempt["cause"]] += 1
            else:
                counts[attempt["cause"]] += 1
    return counts


def print_failure_cause_distribution(title, baseline, candidate):
    print(title)
    print("Counts are unsuccessful route-seed attempts, ordered by total count across baseline and compared model.")
    base_counts = failure_cause_counts(baseline)
    cand_counts = failure_cause_counts(candidate)
    causes = sorted(
        set(base_counts) | set(cand_counts),
        key=lambda cause: (-(base_counts[cause] + cand_counts[cause]), cause),
    )
    rows = [
        (cause, base_counts[cause], cand_counts[cause], cand_counts[cause] - base_counts[cause])
        for cause in causes
    ]
    print_table(rows, ("cause", "baseline", "model", "change"), (44, 10, 10, 8))
    print()


def print_systematic_failures(title, run):
    print(title)
    rows = systematic_failure_rows(run)
    if rows:
        print_table(rows, ("route", "res", "scenario", "main cause"), (8, 5, 44, 32))
    else:
        print("(none)")
    print()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def compare(args):
    routes = parse_split(args.split_dir)
    baseline = collect_run(args.baseline, routes)
    candidate = collect_run(args.candidate, routes)
    baseline_label = args.baseline_name
    comparison_label = args.candidate_name

    print("BENCH2DRIVE SCENARIO COMPARISON")
    print(f"Baseline model:  {baseline_label}")
    print(f"Compared model: {comparison_label}")
    print(f"Split:           {args.split_dir}")
    print(f"Routes:          {len(routes)}")
    print(f"Seeds:           {', '.join(SEEDS)}")
    print("Change convention: compared-model failures minus baseline failures; negative means the compared model improved.")
    print()

    base_failed_routes = sum(1 for r in baseline["routes"].values() if r["failed"] > 0)
    cand_failed_routes = sum(1 for r in candidate["routes"].values() if r["failed"] > 0)
    base_systematic = sum(1 for r in baseline["routes"].values() if r["failed"] == len(SEEDS))
    cand_systematic = sum(1 for r in candidate["routes"].values() if r["failed"] == len(SEEDS))
    base_attempt_failures = sum(r["failed"] for r in baseline["routes"].values())
    cand_attempt_failures = sum(r["failed"] for r in candidate["routes"].values())
    base_missing = sum(r["missing"] for r in baseline["routes"].values())
    cand_missing = sum(r["missing"] for r in candidate["routes"].values())
    expected_attempts = len(routes) * len(SEEDS)

    print("1. OVERALL FAILURE SUMMARY")
    rows = [
        ("failed route-seed attempts", base_attempt_failures, cand_attempt_failures, cand_attempt_failures - base_attempt_failures),
        ("routes failed in >=1 seed", base_failed_routes, cand_failed_routes, cand_failed_routes - base_failed_routes),
        ("routes failed in all 3 seeds", base_systematic, cand_systematic, cand_systematic - base_systematic),
        ("missing/incomplete attempts", base_missing, cand_missing, cand_missing - base_missing),
        ("expected attempts", expected_attempts, expected_attempts, 0),
    ]
    print_table(rows, ("metric", "baseline", "model", "change"), (32, 10, 10, 8))
    print()

    transition_counts = Counter()
    transition_rows = defaultdict(list)
    for index in sorted(routes):
        b = baseline["routes"][index]
        c = candidate["routes"][index]
        label = route_transition_label(b, c)
        transition_counts[label] += 1
        transition_rows[label].append((b, c))

    print_failure_cause_distribution("2. PRIMARY FAILURE CAUSE DISTRIBUTION", baseline, candidate)

    print("3. ROUTE-LEVEL TRANSITIONS")
    order = [
        "fixed",
        "improved_but_not_fixed",
        "regressed",
        "worse_but_both_fail",
        "failed_both_same_count",
        "has_missing_attempt",
        "passed_both",
    ]
    rows = [(label, transition_counts[label]) for label in order]
    print_table(rows, ("transition", "routes"), (28, 8))
    print()

    scenario_rows = []
    for scenario in sorted(set(baseline["scenarios"]) | set(candidate["scenarios"])):
        b = baseline["scenarios"][scenario]
        c = candidate["scenarios"][scenario]
        delta = c["failed"] - b["failed"]
        scenario_rows.append((
            scenario,
            fmt_count(b),
            f"{pct(b['failed'], b['expected']):5.1f}%",
            fmt_count(c),
            f"{pct(c['failed'], c['expected']):5.1f}%",
            delta,
            main_cause(b),
            main_cause(c),
        ))

    print("4. BIGGEST SCENARIO IMPROVEMENTS")
    improved = [r for r in scenario_rows if r[5] < 0]
    improved.sort(key=lambda row: row[5])
    print_table(
        [(r[0], r[1], r[2], r[3], r[4], f"{r[5]:+d}", r[6], r[7]) for r in improved[:12]],
        ("scenario", "baseline", "baseline %", "model", "model %", "change", "baseline cause", "model cause"),
        (44, 18, 10, 18, 11, 7, 30, 30),
    )
    print()

    print("5. BIGGEST SCENARIO REGRESSIONS")
    regressed = [r for r in scenario_rows if r[5] > 0]
    regressed.sort(key=lambda row: row[5], reverse=True)
    print_table(
        [(r[0], r[1], r[2], r[3], r[4], f"{r[5]:+d}", r[6], r[7]) for r in regressed[:12]],
        ("scenario", "baseline", "baseline %", "model", "model %", "change", "baseline cause", "model cause"),
        (44, 18, 10, 18, 11, 7, 30, 30),
    )
    print()

    print("6. UNCHANGED HARD SCENARIOS")
    unchanged = [r for r in scenario_rows if r[5] == 0 and ("/15" in r[1] or "/15" in r[3])]
    unchanged.sort(key=lambda row: (int(row[1].split("/")[0]) + int(row[3].split("/")[0])), reverse=True)
    unchanged = [r for r in unchanged if int(r[1].split("/")[0]) >= args.hard_threshold or int(r[3].split("/")[0]) >= args.hard_threshold]
    print_table(
        [(r[0], r[1], r[2], r[3], r[4], r[6], r[7]) for r in unchanged[:12]],
        ("scenario", "baseline", "baseline %", "model", "model %", "baseline cause", "model cause"),
        (44, 18, 10, 18, 11, 30, 30),
    )
    print()

    print("7. RENDER CANDIDATES")
    print("Rows are sorted by route-level change severity, then by scenario and result id.")
    print()
    candidate_specs = [
        ("fixed", "baseline fails, compared model succeeds"),
        ("regressed", "baseline succeeds, compared model fails"),
        ("worse_but_both_fail", "both fail, compared model fails in more seeds"),
        ("failed_both_same_count", "persistent hard cases"),
    ]
    for label, description in candidate_specs:
        print(f"{label}: {description}")
        rows = []
        selected = sorted(
            transition_rows[label],
            key=lambda pair: render_candidate_sort_key(label, pair),
        )
        for b, c in selected[: args.render_candidates_per_group]:
            rows.append((
                b["route_id"],
                b["res_id"],
                b["scenario_name"],
                f"{b['failed']}/3",
                f"{c['failed']}/3",
                main_cause(b),
                main_cause(c),
            ))
        if rows:
            print_table(
                rows,
                ("route", "res", "scenario", "baseline", "model", "baseline cause", "model cause"),
                (8, 5, 44, 8, 9, 30, 30),
            )
        else:
            print("(none)")
        print()

    print("8. HARDEST SCENARIOS PER MODEL")
    print_hardest_scenarios(f"HARDEST SCENARIOS: {baseline_label}", baseline)
    print_hardest_scenarios(f"HARDEST SCENARIOS: {comparison_label}", candidate)

    print("9. ALL 44 SCENARIO TYPES")
    scenario_rows.sort(key=lambda row: (row[0]))
    print_table(
        [(r[0], r[1], r[2], r[3], r[4], f"{r[5]:+d}", r[6], r[7]) for r in scenario_rows],
        ("scenario", "baseline", "baseline %", "model", "model %", "change", "baseline cause", "model cause"),
        (44, 18, 10, 18, 11, 7, 30, 30),
    )
    print()

    print("10. SYSTEMATIC ROUTE FAILURES PER MODEL")
    print_systematic_failures(f"SYSTEMATIC ROUTE FAILURES: {baseline_label}", baseline)
    print_systematic_failures(f"SYSTEMATIC ROUTE FAILURES: {comparison_label}", candidate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline bench2drive folder containing seed directories 1/2/3.")
    parser.add_argument("--candidate", required=True, help="Candidate bench2drive folder containing seed directories 1/2/3.")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument(
        "--split-dir",
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "leaderboard", "data", "bench2drive_split")
        ),
        help="Bench2Drive split XML directory.",
    )
    parser.add_argument("-o", "--output-file", default=None)
    parser.add_argument("--hard-threshold", type=int, default=7)
    parser.add_argument("--render-candidates-per-group", type=int, default=12)
    args = parser.parse_args()

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            with redirect_stdout(Tee(sys.stdout, f)):
                compare(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
