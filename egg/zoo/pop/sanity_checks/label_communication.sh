#!/bin/bash

# This script runs label communication experiments for various datasets and models.

# Example usage:
# ./label_communication.sh <dataset_name> <model_name>

# Parameters
#SBATCH --mem=100G
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH --exclude=node044,node043
#SBATCH --error=/home/mmahaut/projects/exps/tmlr/label_com_%j_0_log.err
#SBATCH --output=/home/mmahaut/projects/exps/tmlr/label_com_%j_0_log.out
#SBATCH --job-name=label_com

source ~/.bashrc

# Load necessary modules
cd /home/mmahaut/projects/EGG
echo $SLURMD_NODENAME
conda activate omelette2
# export PATH=$PATH:/soft/easybuild/x86_64/software/Miniconda3/4.9.2/bin/
which python


# Run the Python script
python /home/mmahaut/projects/EGG/egg/zoo/pop/label_communication.py