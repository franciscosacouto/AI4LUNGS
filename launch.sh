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

if [ -z "$1" ]; then
    echo "Error: No config file name provided. Usage: sbatch launch.sh config_name.yaml"
    exit 1
fi

CONFIG_FILE_NAME=$1

echo "Starting job with config file: $CONFIG_FILE_NAME"

# Run the Python script using the Hydra command-line override
# The 'train.py' script uses @hydra.main to handle the configuration.
# We are overriding the default config with the provided file name.
python encoder_decoder_approach.py --config-name $CONFIG_FILE_NAME