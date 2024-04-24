# PenSLR: Persian End-to-End Sign Language Recognition Using Ensembling

Welcome to the README for PenSLR, a project focused on Persian end-to-end Sign Language Recognition utilizing ensembling techniques. This README will guide you through the structure of the codebase and provide instructions for setting up and using the project.

## Table of Contents
1. [Introduction](#introduction)
2. [Structure](#structure)
3. [Usage](#usage)
4. [Dataset](#dataset)
5. [Dependencies](#dependencies)
6. [Acknowledgements](#acknowledgements)

## Introduction
PenSLR is a research project aimed at advancing sign language recognition, particularly in the context of Persian sign language. Our approach leverages ensembling methods to improve recognition accuracy and robustness.

## Structure
The codebase follows a modular structure for better organization and maintainability:

- **`dataset/dataset.py`**: Contains the implementation of the DataLoader for handling the dataset.
  
- **`models/ablation.py`**: Implementation of models used for the ablation study.

- **`models/cnnlstm.py`**: Contains the proposed model for Persian Sign Language Recognition.

- **`trainer.py`**: Implementation of the trainer module, which includes functions to facilitate effective model training.

- **`options/option.py`**: Configuration file containing hyperparameters, paths, and other settings.

- **`main.py`**: Entry point for training and testing the models.

## Usage
To use this project:

1. Clone the repository:
   ```bash
   git clone https://github.com/Persian-Sign-Language/PenSLR.git
   ```

2. Download the dataset from [this repository](https://github.com/Persian-Sign-Language/PenSLR-dataset).

3. Move the `dataset1-3` and `dataset4-8` folders to a folder named `data` in the main directory of the PenSLR repository.

4. Customize hyperparameters and other settings in `options/option.py` if necessary.

5. Run `main.py` to start training and/or testing the models.

## Dataset
The dataset used in this project can be found at [PenSLR-dataset](https://github.com/Persian-Sign-Language/PenSLR-dataset). It contains the necessary data for training and evaluating the models.

## Dependencies
Ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- Other dependencies as specified in `requirements.txt`

## Acknowledgements
We extend our deepest gratitude to the reviewers whose insightful feedback significantly enhanced
the quality of this paper. Their expertise and constructive critiques were of great value to the
development of our research.

We would also like to express our profound appreciation to the deaf community association for
their unwavering support and collaboration.

Special thanks go to the five volunteers who generously dedicated their time and effort to help us
record and collect the necessary data. Their commitment was crucial to the success of our project.