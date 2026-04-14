import json
import glob
import argparse
import os
import math


def _extract_case_index(filename):
    stem = os.path.splitext(filename)[0]
    if not stem.endswith('_res'):
        return None
    prefix = stem[:-4]
    if not prefix.isdigit():
        return None
    return int(prefix)

def process_single_seed(folder_path):
    """Processes one seed folder and returns (seed_ds, seed_sr)"""
    file_paths = glob.glob(f'{folder_path}/*.json')
    merged_records = []
    driving_scores = []
    success_num = 0
    processed_indices = set() 

    for file_path in file_paths:
        if 'merged.json' in file_path: continue
        filename = os.path.basename(file_path)
        try:
            file_index = _extract_case_index(filename)
            if file_index is None or file_index < 0 or file_index >= 220:
                continue
            with open(file_path) as file:
                data = json.load(file)
                records = data['_checkpoint']['records']
                
                if not records:
                    continue 

                for rd in records:
                    if rd['status'] == 'Failed - Agent crashed':
                        if 'scores' not in rd: rd['scores'] = {'score_composed': 0.0}
                    
                    rd.pop('index', None)
                    merged_records.append(rd)
                    driving_scores.append(rd['scores']['score_composed'])
                    processed_indices.add(file_index)

                    if rd['status'] in ['Completed', 'Perfect']:
                        success_flag = True
                        for k, v in rd['infractions'].items():
                            if len(v) > 0 and k != 'min_speed_infractions':
                                success_flag = False
                                break
                        if success_flag:
                            success_num += 1
        except Exception:
            continue

    # Inject Missing/Empty (The Route 111 Fix)
    total_expected = 220
    missing_indices = set(range(total_expected)) - processed_indices 
    for idx in missing_indices:
        driving_scores.append(0.0)
        merged_records.append({"route_id": f"MISSING_{idx}", "scores": {"score_composed": 0.0}, "status": "Failed"})

    seed_ds = sum(driving_scores) / len(driving_scores) if driving_scores else 0
    seed_sr = success_num / len(driving_scores) if driving_scores else 0
    
    # Save individual merged.json for this seed
    merged_data = {
        "_checkpoint": {"records": merged_records},
        "driving score": seed_ds,
        "success rate": seed_sr,
        "eval num": len(driving_scores),
    }
    with open(os.path.join(folder_path, 'merged.json'), 'w') as file:
        json.dump(merged_data, file, indent=4)

    return seed_ds, seed_sr

def calculate_stats(data_list):
    """Manual calculation of mean and std dev without numpy"""
    if not data_list: return 0.0, 0.0
    mean = sum(data_list) / len(data_list)
    variance = sum((x - mean) ** 2 for x in data_list) / len(data_list)
    std_dev = math.sqrt(variance)
    return mean, std_dev

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_folder', required=True, help="Path to 'bench2drive' folder containing 1, 2, 3")
    args = parser.parse_args()

    seeds = ['1', '2', '3']
    final_ds_list = []
    final_sr_list = []

    print(f"{'Seed':<10} | {'Driving Score':<15} | {'Success Rate'}")
    print("-" * 45)

    for s in seeds:
        seed_path = os.path.join(args.base_folder, s, 'res')
        if os.path.isdir(seed_path):
            ds, sr = process_single_seed(seed_path)
            final_ds_list.append(ds)
            final_sr_list.append(sr * 100)
            print(f"Seed {s:<5} | {ds:<15.2f} | {sr*100:.2f}%")
        else:
            print(f"⚠️  Folder for seed {s} not found at {seed_path}")

    if final_ds_list:
        mean_ds, std_ds = calculate_stats(final_ds_list)
        mean_sr, std_sr = calculate_stats(final_sr_list)
        
        print("-" * 45)
        print(f"📊 AGGREGATED METRICS (N=3 Seeds)")
        print(f"Driving Score: {mean_ds:.2f} ± {std_ds:.2f}")
        print(f"Success Rate:  {mean_sr:.2f}% ± {std_sr:.2f}%")
        print("-" * 45)

if __name__ == '__main__':
    main()