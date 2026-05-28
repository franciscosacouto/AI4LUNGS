#!/bin/bash
#
#SBATCH --partition=gpu_min80gb   # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min80gb          # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=radiofrozen     # Job name
#SBATCH --output=output_file/slurm_%x.%j.out  # File containing STDOUT output
#SBATCH --error=erors/slurm_%x.%j.err   # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
# (...)

# 1. Deterministic flags (keep these to ensure GPU behavior is consistent)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 2. Run the specific setup/test script
# We don't need the loop or the config files yet because we are 
# just verifying the model loading and the forward passes.
echo "Running Encoder Setup Test..."

python /nas-ctm01/homes/fmferreira/AI4LUNGS/src/modules/script/full_deep_learning/encoder_survivalhead.py


echo "Test complete. Check the .out file for Success/Failure logs."