# %%
import os
import subprocess
import time
import ujson
import shutil
from tqdm.autonotebook import tqdm

# %%
def get_num_jobs(job_name, username):
    len_usrn = len(username)
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            shell=True,
        ).decode('utf-8').replace('\n', ''))
    try:
        with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
            max_num_parallel_jobs = int(f.read())
    except:
        max_num_parallel_jobs = 1

    return num_running_jobs, max_num_parallel_jobs

# %%

def bash_file_bench2drive(job, port, tm_port, partition_name):
    cfg = job["cfg"]
    route = job["route"]
    route_id = job["route_id"]
    seed = job["seed"]
    viz_path = job["viz_path"]
    result_file = job["result_file"]
    log_file = job["log_file"]
    err_file = job["err_file"]
    job_file = job["job_file"]
    account_line = f"#SBATCH --account={cfg['slurm_account']}\n" if cfg.get("slurm_account") else ""

    with open(job_file, 'w', encoding='utf-8') as rsh:
            rsh.write(f'''#!/bin/bash
#SBATCH --job-name={cfg["agent"]}_{seed}_{cfg["benchmark"]}_{route_id}
#SBATCH --partition={partition_name}
{account_line}#SBATCH -o {log_file}
#SBATCH -e {err_file}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --time=12:00:00       # EDITED for Berzelius
#SBATCH --gres=gpu:1

echo JOB ID $SLURM_JOB_ID

module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo
source env.sh

cd {cfg["repo_root"]}

export CARLA_ROOT={cfg["carla_root"]}
export PYTHONPATH=$PYTHONPATH:{cfg["carla_root"]}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:{cfg["carla_root"]}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg

# --- EDITED for Berzelius: Repo Root first, then Bench2Drive ---
export PYTHONPATH={cfg["repo_root"]}:{cfg["repo_root"]}/Bench2Drive/leaderboard:{cfg["repo_root"]}/Bench2Drive/scenario_runner:$PYTHONPATH
export SCENARIO_RUNNER_ROOT={cfg["repo_root"]}/Bench2Drive/scenario_runner
export SAVE_PATH={viz_path}

# --- EDITED for Berzelius: Thread-safe xdg-user-dir (Unique /tmp folder) ---
mkdir -p /tmp/$SLURM_JOB_ID
echo '#!/bin/bash' > /tmp/$SLURM_JOB_ID/xdg-user-dir
echo 'echo /scratch/local' >> /tmp/$SLURM_JOB_ID/xdg-user-dir
chmod +x /tmp/$SLURM_JOB_ID/xdg-user-dir
export PATH=/tmp/$SLURM_JOB_ID:$PATH
# -----------------------------------------------------------

python -u {cfg["repo_root"]}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py --routes={route} \
--repetitions=1 \
--track=SENSORS \
--checkpoint={result_file} \
--timeout=600 \
--agent={cfg["agent_file"]} \
--agent-config={cfg["checkpoint"]} \
--traffic-manager-seed={seed} \
--port={port} \
--traffic-manager-port={tm_port}
''')


# def bash_file_bench2drive(job, port, tm_port, partition_name):
#     cfg = job["cfg"]
#     route = job["route"]
#     route_id = job["route_id"]
#     seed = job["seed"]
#     viz_path = job["viz_path"]
#     result_file = job["result_file"]
#     log_file = job["log_file"]
#     err_file = job["err_file"]
#     job_file = job["job_file"]

#     with open(job_file, 'w', encoding='utf-8') as rsh:
#             rsh.write(f'''#!/bin/bash
# #SBATCH --job-name={cfg["agent"]}_{seed}_{cfg["benchmark"]}_{route_id}
# #SBATCH --partition={partition_name}
# #SBATCH -o {log_file}
# #SBATCH -e {err_file}
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=40gb
# #SBATCH --time=00:30:00       # EDITED for Berzelius
# #SBATCH --gres=gpu:1

# echo JOB ID $SLURM_JOB_ID

# # --- EDITED for Berzelius ---
# module load Miniforge3/24.7.1-2-hpc1-bdist
# conda activate simlingo
# source env.sh
# # ------------------------------

# cd {cfg["repo_root"]}


# export CARLA_ROOT={cfg["carla_root"]}
# export PYTHONPATH=$PYTHONPATH:{cfg["carla_root"]}/PythonAPI/carla
# export PYTHONPATH=$PYTHONPATH:{cfg["carla_root"]}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg

# # export PYTHONPATH=$PYTHONPATH:{cfg["repo_root"]}/Bench2Drive/leaderboard
# # export PYTHONPATH=$PYTHONPATH:{cfg["repo_root"]}/Bench2Drive/scenario_runner
# # --- EDITED for Berzelius: ADDED NEW FIX (Prepend paths) ---
# export PYTHONPATH={cfg["repo_root"]}:{cfg["repo_root"]}/Bench2Drive/leaderboard:{cfg["repo_root"]}/Bench2Drive/scenario_runner:$PYTHONPATH
# # --------------------------------------------------------

# export SCENARIO_RUNNER_ROOT={cfg["repo_root"]}/Bench2Drive/scenario_runner

# export SAVE_PATH={viz_path}

# # --- EDITED for Berzelius: Create a fake xdg-user-dir to prevent crash ---
# echo '#!/bin/bash' > xdg-user-dir
# echo 'echo /scratch/local' >> xdg-user-dir
# chmod +x xdg-user-dir
# export PATH=$PWD:$PATH
# # --------------------------------------------------------------------------


# python -u {cfg["repo_root"]}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py --routes={route} \
# --repetitions=1 \
# --track=SENSORS \
# --checkpoint={result_file} \
# --timeout=600 \
# --agent={cfg["agent_file"]} \
# --agent-config={cfg["checkpoint"]} \
# --traffic-manager-seed={seed} \
# --port={port} \
# --traffic-manager-port={tm_port}
# ''')


# %%
def get_running_jobs():
    running_jobs = subprocess.check_output(f'squeue --me',shell=True).decode('utf-8').splitlines()
    running_jobs = set(x.strip().split(" ")[0] for x in running_jobs[1:])
    return running_jobs

# %%
def filter_completed(jobs):
    filtered_jobs = []

    running_jobs = get_running_jobs()
    for job in jobs:

        # If job is running we keep it in list (other function does killing)
        if "job_id" in job:
           if job["job_id"] in running_jobs:
              filtered_jobs.append(job)
              continue

        # Keep failed jobs to resubmit
        result_file = job["result_file"]
        if os.path.exists(result_file):
            try:
                with open(result_file, "r") as f:
                    evaluation_data = ujson.load(f)
            except:
                if job["tries"] > 0:
                    filtered_jobs.append(job)
                    continue


            progress = evaluation_data['_checkpoint']['progress']

            need_to_resubmit = False
            if len(progress) < 2 or progress[0] < progress[1]:
                need_to_resubmit = True
            else:
                for record in evaluation_data['_checkpoint']['records']:
                    if record['status'] == 'Failed - Agent couldn\'t be set up':
                        need_to_resubmit = True
                    elif record['status'] == 'Failed':
                        need_to_resubmit = True
                    elif record['status'] == 'Failed - Simulation crashed':
                        need_to_resubmit = True
                    elif record['status'] == 'Failed - Agent crashed':
                        need_to_resubmit = True

            if need_to_resubmit and job["tries"] > 0:
                filtered_jobs.append(job)
        # Results file doesnt exist
        elif job["tries"] > 0:
            filtered_jobs.append(job)
    return filtered_jobs

# %%
def kill_dead_jobs(jobs):

    running_jobs = get_running_jobs()

    for job in jobs:

        if "job_id" in job:
            job_id = job["job_id"]

        elif os.path.exists(job["log_file"]):
            with open(job["log_file"], "r") as f:
                job_id = f.readline().strip().replace("JOB ID ", "")
        
        else:
            continue

        if job_id not in running_jobs:
            continue

        log = job["log_file"]
        if not os.path.exists(job["log_file"]):
            continue

        with open(log) as f:
            lines = f.readlines()
        if len(lines)==0:
            continue

        # TODO this needs to be improved, maybe also consider the err file?
        if any(["Watchdog exception" in line for line in lines]) or \
            "Engine crash handling finished; re-raising signal 11 for the default handler. Good bye.\n" in lines or \
            "[91mStopping the route, the agent has crashed:\n" in lines:

            subprocess.Popen(f"scancel {job_id}", shell=True)

# ------ EDITED for Berzelius ------- #
# For Baseline evaluation on Bench2Drive
_env = os.environ
_repo_root = _env.get("REPO_DIR", os.path.abspath(os.path.dirname(__file__)))
_base_dir = _env.get("BASE_DIR", os.path.dirname(_repo_root))
_carla_root = _env.get("CARLA_ROOT", os.path.join(_base_dir, "carla", "CARLA_0.9.15"))
_checkpoint = _env.get(
    "MODEL_CKPT",
    os.path.join(
        _base_dir,
        "checkpoints",
        "simlingo_pretrained",
        "simlingo",
        "checkpoints",
        "epoch=013.ckpt",
        "pytorch_model.pt",
    ),
)

configs = [
    {
        "agent": "simlingo",
        "checkpoint": _checkpoint,
        "benchmark": "bench2drive",
        "route_path": os.path.join(_repo_root, "leaderboard", "data", "bench2drive_split"),
        "seeds": [1, 2, 3],
        "tries": 2,
        "out_root": os.path.join(_base_dir, "eval_results", "Bench2Drive"),
        "carla_root": _carla_root,
        "repo_root": _repo_root,
        "agent_file": os.path.join(_repo_root, "team_code", "agent_simlingo.py"),
        "team_code": "team_code",
        "agent_config": "not_used",
        "username": _env.get("USERNAME", _env.get("USER", "x_hugaf")),
        "slurm_partition": _env.get("SLURM_PARTITION", "berzelius"),
        "slurm_account": _env.get("SLURM_ACCOUNT", "")
    }
]
# -------------------------------------- #

# configs = [
#     {
#     "agent": "simlingo",
#     "checkpoint": "/PATH/TO/REPO/outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt",
#     "benchmark": "bench2drive",
#     "route_path": "/PATH/TO/REPO/leaderboard/data/bench2drive_split",
#     "seeds": [1,2,3], # TODO: change depending on how many eval seeds you wanna run (paper uses one eval seed on three train seeds)
#     "tries": 2,
#     "out_root": "/PATH/TO/REPO/eval_results/Bench2Drive",
#     "carla_root": "~/software/carla0915",
#     "repo_root": "/PATH/TO/REPO",
#     "agent_file": "/PATH/TO/REPO/team_code/agent_simlingo.py",
#     "team_code": "team_code",
#     "agent_config": "not_used",
#     "username": "YOUR_USERNAME"
#     }
#     ] # TODO: change to your paths and model, you can add multiple configs here, whch get evaluated after each other


# %%
job_queue = []
for cfg_idx, cfg in enumerate(configs):
    route_path = cfg["route_path"]
    routes = [x for x in os.listdir(route_path) if x[-4:]==".xml"] #########################################

    if cfg["benchmark"] == "bench2drive":
        fill_zeros = 3
    else: 
        fill_zeros = 2

    for seed in cfg["seeds"]:
        seed = str(seed)

        base_dir = os.path.join(cfg["out_root"], cfg["agent"], cfg["benchmark"], seed)
        os.makedirs(os.path.join(base_dir, "run"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "res"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "out"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "err"), exist_ok=True)

        for route in routes:
            route_id = route.split("_")[-1][:-4].zfill(fill_zeros)
            route = os.path.join(route_path, route)

            viz_path = os.path.join(base_dir, "viz", route_id)
            os.makedirs(viz_path, exist_ok=True)

            result_file = os.path.join(base_dir, "res", f"{route_id}_res.json")
            log_file = os.path.join(base_dir, "out", f"{route_id}_out.log")
            err_file = os.path.join(base_dir, "err", f"{route_id}_err.log")

            job_file = os.path.join(base_dir, "run", f'eval_{route_id}.sh')
            
            job = {
                "cfg": cfg,
                "route": route,
                "route_id": route_id,
                "seed": seed,
                "viz_path": viz_path,
                "result_file": result_file,
                "log_file": log_file,
                "err_file": err_file,
                "job_file": job_file,
                "tries": cfg["tries"]
            }

            job_queue.append(job)

# %%
carla_world_ports = set(range(10000, 20000, 50))
carla_streaming_ports = set(range(20000, 30000, 50))
carla_tm_ports = set(range(30000, 40000, 50))

# %%
jobs = len(job_queue)
progress = tqdm(total = jobs)
while job_queue:
    kill_dead_jobs(job_queue)
    job_queue = filter_completed(job_queue)

    progress.update(jobs - len(job_queue) - progress.n)

    running_jobs = get_running_jobs()

    used_ports = set()
    for job in job_queue:
        if "job_id" in job and job["job_id"] in running_jobs:
            used_ports.update(job["ports"])

    with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
        max_num_parallel_jobs = int(f.read())

    if len(running_jobs) >= max_num_parallel_jobs:
        time.sleep(5)
        continue

    for job in job_queue:
        if job["tries"] <= 0:
            continue

        if "job_id" in job and job["job_id"] in running_jobs: # Das funktioniert nicht wenn man neu startet...
            continue

        if os.path.exists(job["log_file"]):
            with open(job["log_file"], "r") as f:
                job_id = f.readline().strip().replace("JOB ID ", "")
                if job_id in running_jobs:
                    print(f"{job['log_file']} already started.")
                    continue

        # Need to submit this job
        # Make bash file:
        carla_world_port_start = next(iter(carla_world_ports.difference(used_ports)))
        carla_streaming_port_start = next(iter(carla_streaming_ports.difference(used_ports)))
        carla_tm_port_start = next(iter(carla_tm_ports.difference(used_ports)))

        # EDITED for Berzelius: Corrected port order
        if job["cfg"]["benchmark"].lower() == "bench2drive":
            # CORRECTED: World Port first, then TM Port
            partition_name = job["cfg"].get("slurm_partition", "berzelius")
            bash_file_bench2drive(job, carla_world_port_start, carla_tm_port_start, partition_name)
            job["ports"] = {carla_world_port_start, carla_tm_port_start}

        # if job["cfg"]["benchmark"].lower() == "bench2drive":
        #     bash_file_bench2drive(job, carla_tm_port_start, carla_world_port_start, partition_name)
        #     job["ports"] = {carla_world_port_start, carla_tm_port_start}
        else:
            raise NotImplementedError(f"Benchmark {job['cfg']['benchmark']} not implemented.")

        # submit
        shutil.rmtree(job["viz_path"])
        os.mkdir(job["viz_path"])
        job_id = subprocess.check_output(f'sbatch {job["job_file"]}', shell=True).decode('utf-8').strip().rsplit(' ', maxsplit=1)[-1]
        
        job["job_id"] = job_id
        job["tries"] -= 1

        print(f'submit {job["job_file"]}')
        print(len(job_queue))
        break

    time.sleep(10)