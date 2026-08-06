#!/bin/bash
#SBATCH --mem=32GB     
#SBATCH --time=3:00:00
#SBATCH --output=autoencoder/logs/physionet_ae_%j.out
#SBATCH --error=autoencoder/logs/physionet_ae_%j.err

python /home1/amadapur/projects/eeg_trait_state_geometry/autoencoder/calculate_ae_hierarchy.py --embedding_dir /home1/amadapur/projects/eeg_trait_state_geometry/autoencoder/outputs --output_dir=/home1/amadapur/projects/eeg_trait_state_geometry/results_dir/autoencoder/hierarchy --condition_source=conditions --n_perm=1000 --space_name=Autoencoder --feature=autoencoder
