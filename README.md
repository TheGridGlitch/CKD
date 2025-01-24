# Chronic Kidney Disorder Classifier Using MLflow

This project aims to develop a machine learning model to classify Chronic Kidney Disorder (CKD) using MLflow for experiment tracking and model management.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Chronic Kidney Disease is a significant health concern worldwide. Early detection and classification can aid in better management and treatment. This project utilizes machine learning techniques to classify CKD stages based on various medical parameters.

## Features

- Data preprocessing and feature engineering
- Model training and evaluation
- Experiment tracking with MLflow
- User interface for model inference

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/TheGridGlitch/CKD.git
   cd CKD
Create a virtual environment:

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Data Preprocessing:

The dataset.py script handles data loading and preprocessing. Ensure your dataset is in the correct format and update the script as necessary.

Model Training:

Run the main.py script to train the model. This script will:

Load and preprocess the data
Train a logistic regression model
Log the model and parameters using MLflow
bash
Copy
Edit
python main.py
Model Inference:

Use the ckd_ui.py script to launch a simple user interface for model inference.

bash
Copy
Edit
python ckd_ui.py
Dataset
The project includes a synthetic dataset named synthetic_kidney_disease_data.csv. If you have a real dataset, replace this file and ensure it matches the expected format.

Model
The current model is a logistic regression classifier. The trained model is saved as logistic_regression_model.pkl, and the training feature names are stored in training_feature_names.pkl.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.
