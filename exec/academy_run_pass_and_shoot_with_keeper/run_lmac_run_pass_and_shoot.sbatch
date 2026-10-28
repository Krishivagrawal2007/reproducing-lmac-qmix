#!/bin/bash
#SBATCH --job-name=lmac_runpass
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=1-5
#SBATCH --exclude=erc-hpc-comp040
#SBATCH --output=/users/k25000843/lmac_project/logs/academy_run_pass_and_shoot_with_keeper/lmac/lmac_runpass_job%A_seed%a.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate epymarl-llm
cd ~/lmac_project

SEED=$SLURM_ARRAY_TASK_ID

ARCHIVE_DIR="src/llm_source_archive/academy_run_pass_and_shoot_with_keeper"

if [ ! -f "$ARCHIVE_DIR/imp_state_select.py" ]; then
    echo "ERROR: $ARCHIVE_DIR/imp_state_select.py does not exist." >&2
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
    exit 1
fi

echo "Running LMAC on academy_run_pass_and_shoot_with_keeper with seed $SEED using state $IMP_STATE"

python src/main_llm_final.py --config=lmac --env-config=grf with \
env_args.map_name=academy_run_pass_and_shoot_with_keeper \
t_max=5000000 seed=$SEED \
timestep_wise=True \
"comm_code_paths=['$ARCHIVE_DIR/comm_init.py','$ARCHIVE_DIR/comm_update.py','$ARCHIVE_DIR/comm_update_timestep_wise2.py']" \
"important_state=$IMP_STATE"
