#!/bin/bash  
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/hierarchy_%j.out
#SBATCH --error=logs/hierarchy_%j.err
#SBATCH --partition=main

python /home1/amadapur/projects/eeg_trait_state_geometry/calculate_hierarchy.py 
