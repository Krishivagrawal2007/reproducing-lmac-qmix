#!/bin/bash
#SBATCH --job-name=lmac_1o
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-5
#SBATCH --output=lmac_1o_job%A_seed%a.log

eval "$(conda shell.bash hook)"
conda activate epymarl-llm
cd ~/lmac_project

export SC2PATH="$HOME/lmac_project/3rdparty/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

SEED=$SLURM_ARRAY_TASK_ID

echo "Running LMAC on 1o_10b_vs_1r with seed $SEED"
python src/main_llm_final.py --config=lmac --env-config=sc2 with \
env_args.map_name=1o_10b_vs_1r \
t_max=2000000 seed=$SEED \
timestep_wise=True \
"comm_code_paths=['src/llm_source_archive/1o_10b_vs_1r/comm_init.py','src/llm_source_archive/1o_10b_vs_1r/comm_update.py','src/llm_source_archive/1o_10b_vs_1r/comm_update_timestep_wise2.py']" \
"important_state=[62, 63, 67, 68]"