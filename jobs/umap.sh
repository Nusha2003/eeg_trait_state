#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  
#SBATCH --mem=16GB         
#SBATCH --partition=main

python /home1/amadapur/projects/eeg_trait_state_geometry/plot_umap.py