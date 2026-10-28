#!/bin/bash
#SBATCH --job-name=qmix_forage11x11_extraseeds
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=16-20
#SBATCH --exclude=erc-hpc-comp040
#SBATCH --output=/users/k25000843/lmac_project/logs/Foraging-11x11-6p-4f-s1True/qmix/qmix_forage11x11_extraseeds_job%A_seed%a.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate epymarl-llm
cd ~/lmac_project

SEED=$SLURM_ARRAY_TASK_ID
echo "Running QMIX (extra exploration-variance seed) on Foraging-11x11-6p-4f-coop-v3 with seed $SEED"

python src/main.py --config=qmix --env-config=lbf with \
env_args.key=Foraging-11x11-6p-4f-s1True \
env_args.players=6 \
env_args.max_player_level=2 \
env_args.field_size=11 \
env_args.max_food=4 \
env_args.sight=2 \
env_args.force_coop=True \
env_args.normalize_reward=True \
env_args.grid_observation=True \
t_max=2000000 seed=$SEED
