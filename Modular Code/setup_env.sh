#!/bin/bash
set -e


#
#   This needs to be changed to match the propper abc123.
#   You need to modify this to ensure the code works.
#
export CONDA_ENVS_PATH="/work/vxt660/.conda_envs"
export HF_HOME="/work/vxt660/.HF_cache"

ENV_PATH="/work/vxt660/.conda_envs/llm"

echo "Loading Anaconda..."

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Checking if environment exists..."

if [ ! -d "$ENV_PATH" ]; then
    echo "Creating conda environment..."
    conda create -p $ENV_PATH python=3.10 -y
fi

echo "Activating environment..."
conda activate $ENV_PATH

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "Installing PyTorch (CUDA 12.1 build)..."
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing requirements.txt..."
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# If there are errors for the pip installs.
# pip install --upgrade --force-reinstall -r requirements.txt

echo "Environment setup complete."
python --version
pip --version