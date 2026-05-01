#!/bin/bash
set -e
set -o pipefail

echo "Starting Phase 4 (Evaluation)"

export CONDA_ENVS_PATH="/work/vxt660/.conda_envs"
export HF_HOME="/work/vxt660/.HF_cache"

ENV_PATH="/work/vxt660/.conda_envs/llm"

# source activate $ENV_PATH

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate env
conda activate /work/vxt660/.conda_envs/llm

cd /work/vxt660/assignment3

mkdir -p results/phase4
mkdir -p logs

echo "Phase 4 Starting Code"

echo "Phase 4: Generating Outputs"
python run_evaluation.py | tee logs/evaluation.log

echo "Phase 4: Running Forgetting Analysis"
python run_forgetting.py | tee logs/forgetting.log


echo "Phase 4 Complete"