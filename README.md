# TESS TSFresh + MLP Classification

This repository contains the implementation of a machine learning pipeline for classifying exoplanet candidates using preprocessed TESS light curves. Statistical time-series features were extracted using the [TSFresh](https://tsfresh.readthedocs.io/) library and used as input to a multilayer perceptron (MLP) and gradient boosted tree (GBT) classifier.

This experiment is part of the Bachelor's thesis **"Machine Learning for Exoplanet Detection: Investigating Feature Engineering Approaches and Stochastic Resonance Effects"** by Alicia Martinelli (2025).

## Folder Structure

```
kepler_array_tsfresh_mlp/
├── raw/                       # TESS light curves as TFRecords
├── convert_tfrecords.py       # Convert the TFRecords into .npy files and split into train, val and test folders
├── feature_extraction.py      # Feature extraction with TSFresh library to generate the train, val and test datasets
├── gbt.py                     # GBT training and evaluation with Optuna
├── mlp.py                     # MLP training and evaluation with Optuna
├── tess_gbt.pkl               # Best GBT model
├── tess_mlp.h3                # Best MLP model
└── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## Preprocessed TESS dataset
The preprocessed TESS dataset used in this project is based on the public release from Yu et al. (2019) and is available via the Astronet-Vetting GitHub repository [https://github.com/yuliang419/Astronet-Vetting/tree/master/astronet/tfrecords](https://github.com/yuliang419/Astronet-Vetting/tree/master/astronet/tfrecords)

These TFRecord files are already downloaded and placed in the `raw` folder.


## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/AliciaMartinelli/tess_array_tsfresh_mlp.git
    cd tess_array_tsfresh_mlp
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    You may need to install `optuna`, `scikit-learn`, `tsfresh`, `matplotlib`, `numpy`, and `tensorflow` (and more).

## Usage

1. Convert TFRecords into .npy files:
```bash
python convert_tfrecords.py
```
This will generate structured .npy arrays and split them into training, validation, and test sets.

2. Extract features from the TESS `.npy` dataset:
```bash
python feature_extraction.py
```
This step will generate cleaned and scaled TSFresh features for both global and local views.

3. Train and tune the MLP model using Optuna (with evaluation of test set):
```bash
python train_mlp.py
```
The model will be optimized with cross-validation and evaluated on the held-out test set.

4. Train and tune the GBT model using Optuna (with evaluation of test set):
```bash
python train_gbt.py
```
Similar to the MLP step, the GBT model is tuned and evaluated.


## Thesis Context

This repository corresponds to the experiment described in:
- **Section 3.1**: Time-series feature extraction with TSFresh and MLP classification


**Author**: Alicia Martinelli  
**Email**: alicia.martinelli@stud.unibas.ch  
**Year**: 2025
