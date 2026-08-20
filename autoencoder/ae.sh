#!/bin/bash

#SBATCH --job-name=eeg_ae
#SBATCH --partition=gpu

#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:1


DATASET=$1

#SBATCH --output=/home1/amadapur/projects/eeg_trait_state_geometry/autoencoder/logs/slurm_%x_%j.out
#SBATCH --error=/home1/amadapur/projects/eeg_trait_state_geometry/autoencoder/logs/slurm_%x_%j.err

python /home1/amadapur/projects/eeg_trait_state_geometry/autoencoder/main_simple.py --d $DATASET 