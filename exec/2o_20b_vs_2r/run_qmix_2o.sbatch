#!/bin/bash
#SBATCH --job-name=qmix_2o
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=1-5
#SBATCH --output=/users/k25000843/lmac_project/logs/2o_20b_vs_2r/qmix/qmix_2o_job%A_seed%a.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate epymarl-llm
cd ~/lmac_project

export SC2PATH="$HOME/lmac_project/3rdparty/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Same TF32-off fix as LMAC — QMIX crashed with the identical cuBLAS error
# on this map (job36661727), so applying it here too rather than assuming
# it was LMAC-specific.
export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SEED=$SLURM_ARRAY_TASK_ID
echo "Running QMIX on 2o_20b_vs_2r with seed $SEED"

python src/main.py --config=qmix --env-config=sc2 with \
env_args.map_name=2o_20b_vs_2r t_max=5000000 seed=$SEED
