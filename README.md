# EEG Trait-State Geometry

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

To extract features from the raw EEG datasets, run:

```bash
python datasets/make_[DATASET_NAME].py --feature [FEATURE_NAME]
```

For example:

```bash
python datasets/make_motor.py --feature psd
```

## Generate Data Splits

Generate the dataset-specific train/test splits using:

```bash
python datasets/make_[DATASET_NAME]_splits.py \
    --labels [LABELS_PATH] \
    --out_dir [OUTPUT_DIR]
```

## Configuration

Experiment settings are specified in `config.yaml`. Before running an experiment, set the appropriate `dataset` and `feature` fields. For the PhysioNet Motor Movement/Imagery dataset, also specify `num_classes`.

## Create Representation Embeddings

Generate the Trait-LDA, State-LDA, and Joint-LDA representations using:

```bash
python create_embeddings.py \
    --config config.yaml \
    --n_jobs 8
```

## Evaluate Trait-State Hierarchy

Compute the trait-state hierarchy on the held-out representations:

```bash
python calculate_hierarchy.py
```

## Run Decoding

Run trait, within-subject state, and between-subject state decoding:

```bash
python decoding.py \
    --config config.yaml
```

## Train Autoencoder Representations

Autoencoder experiments are submitted through `autoencoder/array.sh`. The dataset, subject counts, and repetitions to run can be specified in the array submission script.

Job-specific training parameters can be modified in `autoencoder/ae.sh`.

Submit the experiments using:

```bash
sbatch autoencoder/array.sh
```
