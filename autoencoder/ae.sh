#!/bin/bash

#SBATCH --job-name=eeg_ae
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1

#SBATCH --output=/home1/amadapur/projects/eeg_trait_state_geometry/autoencoder/run_logs/slurm_%x_%j.out
#SBATCH --error=/home1/amadapur/projects/eeg_trait_state_geometry/autoencoder/run_logs/slurm_%x_%j.err


DATASET=$1

echo "Dataset: $DATASET"
echo "N subjects: $N_SUBJECTS"
echo "Rep: $REP"

export MNE_DATA=/scratch1/amadapur/mne_data
mkdir -p "$MNE_DATA"

echo "MNE_DATA=$MNE_DATA"


python -c "
import os
import mne
print('ENV   =', os.environ.get('MNE_DATA'))
print('CONFIG=', mne.get_config('MNE_DATA'))
print('EXISTS=', os.path.isdir(os.environ['MNE_DATA']))
"

python /home1/amadapur/projects/eeg_trait_state_geometry/autoencoder/main_simple.py \
    --dataset "$DATASET" \
    --num_subjects "$N_SUBJECTS" \
    --rep "$REP"