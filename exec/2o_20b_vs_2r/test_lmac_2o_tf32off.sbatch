#!/bin/bash
#SBATCH --job-name=lmac_2o_tf32test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/2o_20b_vs_2r/lmac/lmac_2o_tf32test_job%A.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate epymarl-llm
cd ~/lmac_project

export SC2PATH="$HOME/lmac_project/3rdparty/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# The actual test: force off TF32 tensor-core kernels at the driver level,
# and pin cuBLAS to a deterministic workspace. Neither changes the algorithm's
# hyperparameters, only which GPU kernel cuBLAS picks.
export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

IMP_STATE=$(python -c "import re; match=re.search(r'\[.*?\]', open('src/llm_source_archive/2o_20b_vs_2r/imp_state_select.py').read().replace('\n','')); print(match.group(0)) if match else print('[]')")

echo "=== TF32-off test: 2o_20b_vs_2r, FULL hyperparams (batch_size=32, hidden_dim=64), t_max=2000 ==="
echo "If this survives past t_env~50, TF32 kernel selection was the trigger and we can keep exact paper hyperparameters."

python src/main_llm_final.py --config=lmac --env-config=sc2 with \
env_args.map_name=2o_20b_vs_2r \
t_max=2000 seed=1 \
timestep_wise=True \
save_model_interval=1000 \
test_interval=100000 \
"comm_code_paths=['src/llm_source_archive/2o_20b_vs_2r/comm_init.py','src/llm_source_archive/2o_20b_vs_2r/comm_update.py','src/llm_source_archive/2o_20b_vs_2r/comm_update_timestep_wise2.py']" \
"important_state=$IMP_STATE"

echo "=== Test run finished without crashing (if you see this line, TF32-off + full shapes survived) ==="
