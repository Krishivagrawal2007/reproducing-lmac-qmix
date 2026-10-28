#!/bin/bash
#SBATCH --job-name=lmac_forage8x8_cont
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-5
#SBATCH --exclude=erc-hpc-comp040
#SBATCH --output=/users/k25000843/lmac_project/logs/Foraging-8x8-2p-2f-s1True/lmac/lmac_forage8x8_cont_job%A_seed%a.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate epymarl-llm
cd ~/lmac_project

SEED=$SLURM_ARRAY_TASK_ID

# Find this seed's checkpoint directory automatically (folder name includes a
# timestamp we don't want to hardcode per-seed). If a seed has multiple runs
# for any reason, this picks the most recently created one.
CKPT_PATH=$(ls -dt results/models/LMAC_seed${SEED}_Foraging-8x8-2p-2f-s1-coop_* 2>/dev/null | head -1)

if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: no checkpoint directory found for seed $SEED matching "
    echo "  results/models/LMAC_seed${SEED}_Foraging-8x8-2p-2f-s1-coop_*" >&2
    exit 1
fi

if [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: resolved path is not a directory: $CKPT_PATH" >&2
    exit 1
fi

# Call select_important_state() directly instead of regex-scraping the source
# file — this was the actual bug last time (the regex grabbed a range-style
# string like '[111-118]' instead of a real list).
IMP_STATE=$(python -c "
import sys
sys.path.insert(0, 'src/llm_source_archive/Foraging-8x8-2p-2f-s1True')
from imp_state_select import select_important_state
print(select_important_state())
")

echo "Continuing LMAC on Foraging-8x8-2p-2f-coop-v3 seed $SEED from: $CKPT_PATH"
echo "Using important_state: $IMP_STATE"

python src/main_llm_final.py --config=lmac --env-config=lbf with \
env_args.key=Foraging-8x8-2p-2f-s1True \
env_args.players=2 \
env_args.max_player_level=2 \
env_args.field_size=8 \
env_args.max_food=2 \
env_args.sight=1 \
env_args.force_coop=True \
env_args.normalize_reward=True \
env_args.grid_observation=True \
t_max=2000000 seed=$SEED \
timestep_wise=True \
checkpoint_path="$CKPT_PATH" \
load_step=0 \
"comm_code_paths=['src/llm_source_archive/Foraging-8x8-2p-2f-s1True/comm_init.py','src/llm_source_archive/Foraging-8x8-2p-2f-s1True/comm_update.py','src/llm_source_archive/Foraging-8x8-2p-2f-s1True/comm_update_timestep_wise2.py']" \
"important_state=$IMP_STATE"
