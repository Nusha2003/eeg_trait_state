#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16 
#SBATCH --mem=32GB         
#SBATCH --partition=epyc-64
#SBATCH --time=02:00:00
#SBATCH --output=/home1/amadapur/projects/eeg_trait_state_geometry/logs/%x_%j.out
#SBATCH --error=/home1/amadapur/projects/eeg_trait_state_geometry/logs/%x_%j.err

python /home1/amadapur/projects/eeg_trait_state_geometry/new_decoding.py --n_jobs=16
