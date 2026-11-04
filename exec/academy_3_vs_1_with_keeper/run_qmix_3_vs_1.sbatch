#!/bin/bash
#SBATCH --job-name=qmix_3v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --array=1-5
#SBATCH --exclude=erc-hpc-comp040
#SBATCH --output=logs/academy_3_vs_1_with_keeper/qmix/qmix_3v1_job%A_seed%a.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate epymarl_gfootball     # the env where gfootball was compiled
cd ~/lmac_project

SEED=$SLURM_ARRAY_TASK_ID
echo "Node: $(hostname)  Env: $CONDA_PREFIX  Seed: $SEED"

# ---- preflight (seconds): fail now instead of training on CPU / crashing later ----
python - << 'PY' || { echo "ERROR: preflight failed on $(hostname); not starting training." >&2; exit 1; }
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("CUDA is not available on this node (training would silently fall back to CPU)")
import gfootball.env as football_env
env = football_env.create_environment(
    env_name="academy_3_vs_1_with_keeper",
    representation="simple115", rewards="scoring", render=False,
    number_of_left_players_agent_controls=3)
env.reset(); env.step([0, 0, 0]); env.close()
print("preflight OK:", torch.__version__, torch.cuda.get_device_name(0))
PY

echo "Running QMIX on academy_3_vs_1_with_keeper with seed $SEED"

python src/main.py --config=qmix --env-config=grf with \
env_args.map_name=academy_3_vs_1_with_keeper \
t_max=3000000 seed=$SEED
