#!/bin/bash

#SBATCH --job-name=create_embeddings
#SBATCH --output=logs/create_embeddings_%j.out
#SBATCH --error=logs/create_embeddings_%j.err

#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

#SBATCH --partition=main


python -u /home1/amadapur/projects/eeg_trait_state_geometry/create_embeddings.py \
    --config /home1/amadapur/projects/eeg_trait_state_geometry/config.yaml \
    --n_jobs "${SLURM_CPUS_PER_TASK}"