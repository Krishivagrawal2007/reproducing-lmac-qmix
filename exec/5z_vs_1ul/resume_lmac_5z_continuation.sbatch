#!/bin/bash
#SBATCH --job-name=lmac_5z_continuation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=1,3,4,5%1
#SBATCH --output=/users/k25000843/lmac_project/logs/5z_vs_1ul/lmac/lmac_5z_continuation_job%A_seed%a.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate epymarl-llm
cd ~/lmac_project

export SC2PATH="$HOME/lmac_project/3rdparty/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

SEED=$SLURM_ARRAY_TASK_ID

# Map each seed to the exact checkpoint dir it timed out in.
# NOTE: these paths contain a space (the datetime.now() stamp) — kept quoted throughout.
case $SEED in
  1) CKPT_PATH="results/models/LMAC_seed1_5z_vs_1ul_2026-08-19 19:13:06.100593" ;;
  3) CKPT_PATH="results/models/LMAC_seed3_5z_vs_1ul_2026-08-19 19:48:41.805844" ;;
  4) CKPT_PATH="results/models/LMAC_seed4_5z_vs_1ul_2026-08-19 19:59:27.087839" ;;
  5) CKPT_PATH="results/models/LMAC_seed5_5z_vs_1ul_2026-08-19 20:30:04.838389" ;;
  *) echo "No checkpoint mapping for seed $SEED" >&2; exit 1 ;;
esac

if [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: checkpoint dir not found: $CKPT_PATH" >&2
    exit 1
fi

# Dynamically extract the important_state array for this map (same as the original script)
IMP_STATE=$(python -c "import re; match=re.search(r'\[.*?\]', open('src/llm_source_archive/5z_vs_1ul/imp_state_select.py').read().replace('\n','')); print(match.group(0)) if match else print('[]')")

echo "Resuming (continuation) LMAC on 5z_vs_1ul seed $SEED from: $CKPT_PATH"

python src/main_llm_final.py --config=lmac --env-config=sc2 with \
env_args.map_name=5z_vs_1ul \
t_max=5000000 seed=$SEED \
timestep_wise=True \
checkpoint_path="$CKPT_PATH" \
load_step=0 \
"comm_code_paths=['src/llm_source_archive/5z_vs_1ul/comm_init.py','src/llm_source_archive/5z_vs_1ul/comm_update.py','src/llm_source_archive/5z_vs_1ul/comm_update_timestep_wise2.py']" \
"important_state=$IMP_STATE"