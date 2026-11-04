# Archived One-Off Resume, Retry, and Extra-Seed Slurm Scripts

This directory contains one-off Slurm submission scripts that were used during the KURF reproduction runs on KCL CREATE HPC to recover from Slurm wall-time interruptions, faulty GPU nodes, or high-variance exploration seeds. They are preserved here as a record of how the Sacred runs in `results/sacred/` were produced.

For standard training runs from scratch, use the canonical `run_lmac_*.sbatch` and `run_qmix_*.sbatch` scripts in the parent `exec/<environment>/` directories.

## Contents

### `2o_20b_vs_2r/`
- `run_lmac_2o_retry.sbatch`: Resubmission of LMAC seeds 1–5 with `#SBATCH --exclude=erc-hpc-comp040` after the initial array job landed on a node whose GPU reported `CUDA-capable device(s) is/are busy or unavailable`.
- `run_qmix_2o_retry.sbatch`: Resubmission of QMIX seeds 3–5 with `#SBATCH --exclude=erc-hpc-comp040` for the same GPU-availability failure on `erc-hpc-comp040` (seeds 1–2 had completed on healthy nodes).
- `resume_lmac_2o_seed1.sbatch`: One-off checkpoint continuation for LMAC seed 1 (`results/sacred/LMAC/2o_20b_vs_2r/7`), resuming from `results/models/LMAC_seed1_2o_20b_vs_2r_2026-08-31 23:25:04.504882` at $t_{\text{env}} \approx 3.5\text{M}$ after StarCraft II exited prematurely at $t_{\text{env}} \approx 3.85\text{M}$ in run `1`. `plot_results.py` stitches runs `1` and `7` into a single 5M-step curve for seed 1.

### `5z_vs_1ul/`
- `resume_lmac_5z.sbatch`, `resume_lmac_5z_continuation.sbatch`, `resume_lmac_5z_continuation2.sbatch`: Checkpoint continuation scripts for LMAC seeds 1–5 on `5z_vs_1ul`, which exceeded the initial 12h wall-time limit at $t_{\text{env}} \approx 1.3\text{M}\text{--}3.1\text{M}$ and were resumed from their last saved checkpoints in `results/models/LMAC_seed*_5z_vs_1ul_*` with 24h–40h wall-time limits (`results/sacred/LMAC/5z_vs_1ul/1`–`5`). Note that these scripts hard-code the timestamped checkpoint directory names from those cluster runs.

### `Foraging-8x8-2p-2f-s1True/`
- `continue_lmac_foraging8x8.sbatch`, `continue_lmac_foraging8x8_again.sbatch`, `continue_lmac_foraging8x8_again2.sbatch`: Automatic checkpoint continuation scripts (resolving the most recent `results/models/LMAC_seed${SEED}_Foraging-8x8-2p-2f-s1-coop_*` folder) used to carry LMAC seeds 1–5 from $t_{\text{env}} \approx 1.1\text{M}$ (`6`–`10`) through $1.85\text{M}$ (`11`–`15`) to completion at $2.0\text{M}$ (`20`–`24`). `plot_results.py` stitches all segments per seed chronologically.

### `Foraging-11x11-6p-4f-s1True/`
- `run_qmix_foraging11x11_extraseeds.sbatch`: Additional QMIX seeds (`--array=6-15`, saved as `results/sacred/qmix/Foraging-11x11-6p-4f-s1True/6`–`15`) run with identical hyperparameters (`sight=1`, `force_coop=True`) to quantify bimodal exploration variance on the sparse-reward 6-player cooperative foraging task. Use `python plot_results.py --seeds 1 2 3 4 5` to restrict plotting to the standard 5 seeds, or omit `--seeds` to include all 15 QMIX seeds.
