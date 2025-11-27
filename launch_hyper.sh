#!/bin/bash
#
#SBATCH --partition=gpu_min80gb   # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min80gb          # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=fm_nlst     # Job name
#SBATCH --output=slurm_%x.%j.out  # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err   # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
# (...)
conda activate ai4lungs
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_SWEEP_ID=ra6ghk0l

wandb agent $WANDB_SWEEP_ID