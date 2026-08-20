#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB         
#SBATCH --partition=main
#SBATCH --time=3:00:00
#SBATCH --output=/home1/amadapur/projects/eeg_trait_state_geometry/logs/%x_%j.out
#SBATCH --error=/home1/amadapur/projects/eeg_trait_state_geometry/logs/%x_%j.err

python /home1/amadapur/projects/eeg_trait_state_geometry/run_decoding.py 
