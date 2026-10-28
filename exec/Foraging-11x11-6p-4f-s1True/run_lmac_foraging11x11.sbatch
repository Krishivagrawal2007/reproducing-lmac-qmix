#!/bin/bash
#SBATCH --job-name=lmac_forage11x11
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=1-5
#SBATCH --exclude=erc-hpc-comp040
#SBATCH --output=/users/k25000843/lmac_project/logs/Foraging-11x11-6p-4f-s1True/lmac/lmac_forage11x11_job%A_seed%a.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate epymarl-llm
cd ~/lmac_project

SEED=$SLURM_ARRAY_TASK_ID

ARCHIVE_DIR="src/llm_source_archive/Foraging-11x11-6p-4f-s1True"

if [ ! -f "$ARCHIVE_DIR/imp_state_select.py" ]; then
    echo "ERROR: $ARCHIVE_DIR/imp_state_select.py does not exist." >&2
    echo "This map's LLM-source archive is missing the important-state selection file." >&2
    echo "Refusing to launch training, since it would fail downstream with a confusing" >&2
    echo "tensor-shape error instead of this clear one. Generate the file first." >&2
    exit 1
fi

IMP_STATE=$(python -c "
import sys
sys.path.insert(0, '$ARCHIVE_DIR')
from imp_state_select import select_important_state
print(select_important_state())
")

if [ -z "$IMP_STATE" ]; then
    echo "ERROR: important_state extraction returned empty output for seed $SEED." >&2
    echo "Refusing to launch training with an empty important_state." >&2
    exit 1
fi

echo "Running LMAC on Foraging-11x11-6p-4f-coop-v3 with seed $SEED using state $IMP_STATE"

python src/main_llm_final.py --config=lmac --env-config=lbf with \
env_args.key=Foraging-11x11-6p-4f-s1True \
env_args.players=6 \
env_args.max_player_level=2 \
env_args.field_size=11 \
env_args.max_food=4 \
env_args.sight=1 \
env_args.force_coop=True \
env_args.normalize_reward=True \
env_args.grid_observation=True \
t_max=2000000 seed=$SEED \
timestep_wise=True \
"comm_code_paths=['src/llm_source_archive/Foraging-11x11-6p-4f-s1True/comm_init.py','src/llm_source_archive/Foraging-11x11-6p-4f-s1True/comm_update.py','src/llm_source_archive/Foraging-11x11-6p-4f-s1True/comm_update_timestep_wise2.py']" \
"important_state=$IMP_STATE"
