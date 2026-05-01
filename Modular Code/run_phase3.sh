#!/bin/bash
set -e
set -o pipefail

echo "Starting Phase 3 (Stage 2 Training)"

export CONDA_ENVS_PATH="/work/vxt660/.conda_envs"
export HF_HOME="/work/vxt660/.HF_cache"

ENV_PATH="/work/vxt660/.conda_envs/llm"

# source activate $ENV_PATH

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate env
conda activate /work/vxt660/.conda_envs/llm

cd /work/vxt660/assignment3

mkdir -p outputs
mkdir -p logs

echo "Phase 3 Starting Code"

python run_stage2.py | tee logs/phase3.log

echo "Phase 3 Complete"