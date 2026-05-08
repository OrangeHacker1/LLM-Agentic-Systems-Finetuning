#!/bin/bash
set -e
set -o pipefail

echo "Starting Phase 2 (Stage 1 Training)"

#
#   This needs to be changed to match the propper abc123.
#   You need to modify this to ensure the code works.
#
export CONDA_ENVS_PATH="/work/vxt660/.conda_envs"
export HF_HOME="/work/vxt660/.HF_cache"

ENV_PATH="/work/vxt660/.conda_envs/llm"

# source activate $ENV_PATH

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate env
# THis is my specific location.
conda activate $ENV_PATH

# Ensure you aree in the right directory.
#cd /work/vxt660/assignment3

mkdir -p outputs
mkdir -p logs

echo "Phase 2 Starting Code"

python train_stage1.py | tee logs/phase2.log

echo "Phase 2 Complete"