#!/bin/bash
#
#SBATCH --partition=gpu_min80gb   # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min80gb          # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=radiofrozen     # Job name
#SBATCH --output=output_file/slurm_%x.%j.out  # File containing STDOUT output
#SBATCH --error=erors/slurm_%x.%j.err   # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
# (...)
export CUBLAS_WORKSPACE_CONFIG=:4096:8




# Define your configs in an array
CONFIGS=("config_ws_2.yaml" )

for CFG in "${CONFIGS[@]}"; do
    echo "Starting job with config: $CFG"
    # Note: we use --config-name to tell Hydra which file to load from the search path
    python encoder_survivalhead.py --config-name "$CFG"
done