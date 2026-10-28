#!/bin/bash
#SBATCH --job-name=lmac_2o
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=1-5
#SBATCH --output=/users/k25000843/lmac_project/logs/2o_20b_vs_2r/lmac/lmac_2o_job%A_seed%a.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate epymarl-llm
cd ~/lmac_project

export SC2PATH="$HOME/lmac_project/3rdparty/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Confirmed fix for the cuBLAS crash on this map at full hyperparameter size
# (verified via test_lmac_2o_tf32off.sbatch: survived t_max=2000 with
# batch_size=32, hidden_dim=64 — no crash, no hyperparameter changes needed).
export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SEED=$SLURM_ARRAY_TASK_ID

IMP_STATE=$(python -c "import re; match=re.search(r'\[.*?\]', open('src/llm_source_archive/2o_20b_vs_2r/imp_state_select.py').read().replace('\n','')); print(match.group(0)) if match else print('[]')")

echo "Running LMAC on 2o_20b_vs_2r with seed $SEED using state $IMP_STATE"

python src/main_llm_final.py --config=lmac --env-config=sc2 with \
env_args.map_name=2o_20b_vs_2r \
t_max=5000000 seed=$SEED \
timestep_wise=True \
"comm_code_paths=['src/llm_source_archive/2o_20b_vs_2r/comm_init.py','src/llm_source_archive/2o_20b_vs_2r/comm_update.py','src/llm_source_archive/2o_20b_vs_2r/comm_update_timestep_wise2.py']" \
"important_state=$IMP_STATE"
