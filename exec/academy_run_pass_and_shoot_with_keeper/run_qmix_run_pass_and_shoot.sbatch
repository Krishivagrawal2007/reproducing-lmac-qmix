#!/bin/bash
#SBATCH --job-name=qmix_runpass
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=1-5
#SBATCH --exclude=erc-hpc-comp040
#SBATCH --output=/users/k25000843/lmac_project/logs/academy_run_pass_and_shoot_with_keeper/qmix/qmix_runpass_job%A_seed%a.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate epymarl-llm
cd ~/lmac_project

SEED=$SLURM_ARRAY_TASK_ID
echo "Running QMIX on academy_run_pass_and_shoot_with_keeper with seed $SEED"

python src/main.py --config=qmix --env-config=grf with \
env_args.map_name=academy_run_pass_and_shoot_with_keeper \
t_max=5000000 seed=$SEED
