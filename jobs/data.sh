#!/bin/bash
#SBATCH --mem=32GB     
#SBATCH --time=1:00:00

python /home1/amadapur/projects/eeg_trait_state_geometry/gamma_data.py --feature='entropy' --dataset='gamma'
