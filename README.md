This repository contains the analysis pipeline for studying the organization of trait-state-related variability in EEG representations. 

## Datasets 
The analysis currently includes four EEG datasets spanning motor and non-motor paradigms.
| Dataset | Paradigm |
|---|---|
| PhysioNet EEG Motor Movement/Imagery | Executed and imagined motor tasks |
| BNCI 2014-002 | Motor imagery |
| High Gamma Dataset | Executed movement |
| ERPCore ERN | Error-related potentials |

## EEG Features
To create features from raw datasets, run:
''' bash
python datasets/make_[DATASET_NAME].py --feature=[FEATURE_NAME]
'''

## Generate Data Splits 
Generate dataset-specific train/test splits:
''' bash 
python dataset/make_[DATASET]_splits.py --labels=[LABELS_PATH] --out_dir=[PATH TO OUT DIR] 

## Config File
Make sure to modify the config file settings for each dataset. Finally, when running experiments, all you have to do is change the "dataset" and "feature" fields at the top, (and "num_classes" for motor) 

## Create Representation Embeddings

Generate Trait-LDA, State-LDA, and Joint-LDA representations using the experiment configuration:

''bash
python create_embeddings.py \
       --config config.yaml
       --n_jobs 8
'''

## Evaluate Trait-State Hierarchy
Computes the trait-state hierarchy on held-out representations

'''bash
python calculate_hierarchy.py 

### Run Decoding
Run trait, within-subject state, and between-subject state decoding:

'''bash 
python decoding.py \
      --config config.yaml
'''

## Train Autoencoder Representations

Autoencoder experiments are submitted through autoencoder/array.sh by specifying the runs, subject counts, and dataset.
You can modify the specific parameters for each job in ae.sh

'''bash
sbatch autoencoder/array.sh
'''
