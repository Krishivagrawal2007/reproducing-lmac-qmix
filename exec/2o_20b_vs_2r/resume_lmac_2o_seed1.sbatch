#!/bin/bash
#SBATCH --job-name=lmac_2o_seed1_resume
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=erc-hpc-comp040
#SBATCH --output=/users/k25000843/lmac_project/logs/2o_20b_vs_2r/lmac/lmac_2o_seed1_resume_job%A.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate epymarl-llm
cd ~/lmac_project

export SC2PATH="$HOME/lmac_project/3rdparty/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CKPT_PATH="results/models/LMAC_seed1_2o_20b_vs_2r_2026-08-31 23:25:04.504882"

if [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: checkpoint dir not found: $CKPT_PATH" >&2
    exit 1
fi

IMP_STATE=$(python -c "import re; match=re.search(r'\[.*?\]', open('src/llm_source_archive/2o_20b_vs_2r/imp_state_select.py').read().replace('\n','')); print(match.group(0)) if match else print('[]')")

echo "Resuming LMAC on 2o_20b_vs_2r seed 1 from: $CKPT_PATH"

python src/main_llm_final.py --config=lmac --env-config=sc2 with \
env_args.map_name=2o_20b_vs_2r \
t_max=5000000 seed=1 \
timestep_wise=True \
checkpoint_path="$CKPT_PATH" \
load_step=0 \
"comm_code_paths=['src/llm_source_archive/2o_20b_vs_2r/comm_init.py','src/llm_source_archive/2o_20b_vs_2r/comm_update.py','src/llm_source_archive/2o_20b_vs_2r/comm_update_timestep_wise2.py']" \
"important_state=$IMP_STATE"
