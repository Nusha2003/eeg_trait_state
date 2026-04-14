#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16 
#SBATCH --mem=32GB         
#SBATCH --partition=epyc-64

python /home1/amadapur/projects/eeg_trait_state_geometry/new_decoding.py --subjects="14" --num_classes="4"
