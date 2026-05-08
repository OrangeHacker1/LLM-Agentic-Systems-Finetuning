#!/bin/bash
set -e

echo "Starting Ablation Study"

#
#   This needs to be changed to match the propper abc123.
#   You need to modify this to ensure the code works.
#
export CONDA_ENVS_PATH="/work/vxt660/.conda_envs"
export HF_HOME="/work/vxt660/.HF_cache"

ENV_PATH="/work/vxt660/.conda_envs/llm"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_PATH

#cd /work/vxt660/assignment3

echo "Starting Ablation Code"

python train_stage2.py | tee logs/ablation_train.log
python run_ablation.py | tee logs/ablation_eval.log

echo "Ablation Study Complete"