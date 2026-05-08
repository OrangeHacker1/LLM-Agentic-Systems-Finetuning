#!/bin/bash
set -e
set -o pipefail

echo "Starting Phase 4 (Evaluation)"

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
conda activate $ENV_PATH

#cd /work/vxt660/assignment3

mkdir -p results/phase4_NEW
mkdir -p results/phase4
mkdir -p logs

echo "Phase 4 Starting Code"

echo "Phase 4: Generating Outputs for Model Base"
python run_phase4_base_model.py | tee logs/phase4_base_model.log

echo "Phase 4: Generating Outputs for Model 1"
python run_phase4_model1.py | tee logs/phase4_model1.log

echo "Phase 4: Generating Outputs for Model 2"
python run_phase4_model2_advanced.py | tee logs/phase4_model2.log

# The following are for forgetting and basic debugging.

echo "Phase 4: Generating Outputs for Model 2"
python run_evaluation_clean.py | tee logs/evaluation_new.log

echo "Phase 4: Running Forgetting Analysis Model 2"
python run_forgetting_clean.py | tee logs/forgetting.log


echo "Phase 4 Complete"