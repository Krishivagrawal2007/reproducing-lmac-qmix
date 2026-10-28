#!/bin/bash
#SBATCH --job-name=qmix_1o
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-5
#SBATCH --output=qmix_1o_job%A_seed%a.log

eval "$(conda shell.bash hook)"
conda activate epymarl-llm
cd ~/lmac_project

export SC2PATH="$HOME/lmac_project/3rdparty/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Use the Slurm Array ID (1 through 5) as the seed
SEED=$SLURM_ARRAY_TASK_ID

echo "Running QMIX on 1o_10b_vs_1r with seed $SEED"
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=1o_10b_vs_1r t_max=2000000 seed=$SEED
