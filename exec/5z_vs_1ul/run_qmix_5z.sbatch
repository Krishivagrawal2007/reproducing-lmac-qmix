#!/bin/bash
#SBATCH --job-name=qmix_5z
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-5
#SBATCH --output=qmix_5z_job%A_seed%a.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate epymarl-llm
cd ~/lmac_project

export SC2PATH="$HOME/lmac_project/3rdparty/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

SEED=$SLURM_ARRAY_TASK_ID
echo "Running QMIX on 5z_vs_1ul with seed $SEED"

python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=5z_vs_1ul t_max=5000000 seed=$SEED
