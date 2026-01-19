#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  
#SBATCH --mem=64GB         
#SBATCH --partition=main
#SBATCH --time=12:00:00

python /home1/amadapur/projects/eeg_trait_state_geometry/permutation.py