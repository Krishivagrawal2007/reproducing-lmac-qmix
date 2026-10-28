#!/bin/bash
#SBATCH --job-name=lmac_5z
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-5
#SBATCH --output=lmac_5z_job%A_seed%a.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate epymarl-llm
cd ~/lmac_project

export SC2PATH="$HOME/lmac_project/3rdparty/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Dynamically extract the important_state array for this specific map
IMP_STATE=$(python -c "import re; match=re.search(r'\[.*?\]', open('src/llm_source_archive/5z_vs_1ul/imp_state_select.py').read().replace('\n','')); print(match.group(0)) if match else print('[]')")

SEED=$SLURM_ARRAY_TASK_ID
echo "Running LMAC on 5z_vs_1ul with seed $SEED using state $IMP_STATE"

python src/main_llm_final.py --config=lmac --env-config=sc2 with \
env_args.map_name=5z_vs_1ul \
t_max=5000000 seed=$SEED \
timestep_wise=True \
"comm_code_paths=['src/llm_source_archive/5z_vs_1ul/comm_init.py','src/llm_source_archive/5z_vs_1ul/comm_update.py','src/llm_source_archive/5z_vs_1ul/comm_update_timestep_wise2.py']" \
"important_state=$IMP_STATE"
