import json
import glob
import os
import argparse
import re
import sys
from contextlib import redirect_stdout
from collections import defaultdict, Counter


def _extract_case_index(filename):
    stem = os.path.splitext(filename)[0]
    if not stem.endswith('_res'):
        return None
    prefix = stem[:-4]
    if not prefix.isdigit():
        return None
    return int(prefix)


def _route_id_without_rep(route_id):
    route_str = re.sub(r'_rep\d+$', '', str(route_id))
    match = re.search(r'RouteScenario_(\d+)$', route_str)
    if match:
        return match.group(1)
    return route_str


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

def analyze_multiseed(base_folder):
    print(f"🕵️  STARTING MULTI-SEED SCENARIO FAILURE ANALYSIS: {base_folder}\n")
    
    seeds = ['1', '2', '3']
    total_expected = 220
    
    # Storage
    # route_info[route_id] = {'scenario': "Name", 'failures': ["cause1", "cause2"], 'res_indices': set([...])}
    route_info = defaultdict(lambda: {'scenario': "Unknown", 'failures': [], 'res_indices': set()})
    
    # Statistics
    scenario_counts = defaultdict(lambda: {'total': 0, 'failed': 0})

    for seed in seeds:
        seed_path = os.path.join(base_folder, seed, 'res')
        if not os.path.isdir(seed_path):
            print(f"⚠️  Skipping Seed {seed} (Not found)")
            continue
            
        json_files = glob.glob(os.path.join(seed_path, "*.json"))
        
        for file_path in json_files:
            if "merged.json" in file_path: continue
            filename = os.path.basename(file_path)

            file_index = _extract_case_index(filename)
            if file_index is None or file_index < 0 or file_index >= total_expected:
                continue
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if "_checkpoint" not in data or not data["_checkpoint"]["records"]:
                    continue

                record = data["_checkpoint"]["records"][0]
                route_id = record.get("route_id", f"UNKNOWN_{file_index}")
                scenario = record["scenario_name"] 
                status = record["status"]
                
                # Update Scenario Name and source res index for this route id
                route_info[route_id]['scenario'] = scenario
                route_info[route_id]['res_indices'].add(file_index)

                # Global Stats
                scenario_counts[scenario]['total'] += 1

                # --- STRICT SUCCESS CHECK ---
                is_success = False
                if status in ['Completed', 'Perfect']:
                    success_flag = True
                    for k, v in record['infractions'].items():
                        if len(v) > 0 and k != 'min_speed_infractions':
                            success_flag = False
                            break
                    if success_flag:
                        is_success = True
                
                # --- FAILURE LOGGING ---
                if not is_success:
                    scenario_counts[scenario]['failed'] += 1
                    
                    # Determine Primary Cause
                    cause = "Unknown"
                    if status not in ['Completed', 'Perfect']:
                        cause = status
                    else:
                        for k, v in record['infractions'].items():
                            if len(v) > 0 and k != 'min_speed_infractions':
                                cause = k
                                break
                    
                    # Store failure
                    route_info[route_id]['failures'].append(cause)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    # --- REPORT GENERATION ---

    # 1. SYSTEMATIC FAILURES
    print(f"🚨 SYSTEMATIC FAILURES (Routes that failed in ALL 3 seeds)")
    print(f"{'Route ID':<12} | {'Res ID':<8} | {'Scenario Name':<50} | {'Common Cause':<30}")
    print("-" * (12 + 3 + 8 + 3 + 50 + 3 + 30))
    
    systematic_count = 0
    
    # Sort by Scenario Name so similar failures are grouped together
    sorted_routes = sorted(route_info.items(), key=lambda x: x[1]['scenario'])
    
    for r_id, info in sorted_routes:
        failures = info['failures']
        if len(failures) == 3: # Failed in Seed 1, 2, AND 3
            counter = Counter(failures)
            primary_cause, count = counter.most_common(1)[0]
            route_display = _route_id_without_rep(r_id)
            res_ids = sorted(info['res_indices'])
            res_display = ','.join(f"{idx:03d}" for idx in res_ids) if res_ids else "-"
            print(f"{route_display:<12} | {res_display:<8} | {info['scenario']:<50} | {primary_cause:<30} ({count}/3)")
            systematic_count += 1

    print("\n" + "="*85 + "\n")

    # 2. HARDEST SCENARIOS
    print(f"📊 HARDEST SCENARIOS (Aggregated across 3 Seeds)")
    print(f"{'Scenario Name':<40} | {'Failures/Attempts':<17} | {'Fail Rate':<8}")
    print("-" * (40 + 3 + 17 + 3 + 8))
    
    scen_list = []
    for name, stats in scenario_counts.items():
        if stats['total'] > 0: # Obs: Irrelevant check since we only count scenarios that were attempted
            rate = (stats['failed'] / stats['total']) * 100
            scen_list.append((name, rate, stats['total']))
            
    scen_list.sort(key=lambda x: x[1], reverse=True)
    
    for name, rate, count in scen_list[:12]: # Show Top 12
        if count > 5:
            failed = scenario_counts[name]['failed']
            print(f"{name:<40} | {failed}/{count:<15} | {rate:5.1f}%   ")

    print("\n" + "="*85 + "\n")
    print(f"Total routes analyzed: {len(route_info)}")
    print(f"Routes that failed in at least one seed: {len([r for r, i in route_info.items() if len(i['failures']) > 0])}")
    print(f"Routes that failed in all 3 seeds: {systematic_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_folder', required=True, help="Path to 'bench2drive' folder containing 1, 2, 3")
    parser.add_argument('-o', '--output_file', default=None, help="Path to save the exact same printed report")
    args = parser.parse_args()
    if os.path.isdir(args.base_folder):
        output_file = args.output_file or os.path.join(os.path.dirname(__file__), 'scenario_failure_report.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            with redirect_stdout(_Tee(sys.stdout, f)):
                analyze_multiseed(args.base_folder)
