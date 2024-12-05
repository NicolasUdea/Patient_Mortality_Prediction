# Mortality Prediction Model

This repository contains a machine learning model to predict patient mortality based on various medical features. The model is built using a Multi-Layer Perceptron (MLP) neural network implemented with TensorFlow and Keras.

## Project Structure

- `modelo.py`: Script to preprocess data, train the MLP model, and save the trained model.
- `test_modelo.py`: Script to evaluate the trained model on test data and print the results.
- `graficos.py`: Script to generate evaluation plots, including the confusion matrix and ROC curve.
- `torax modelacion - copia.xlsx`: Dataset used for training and testing the model.

## Data Preprocessing

The dataset is preprocessed by:
- Converting categorical variables to binary.
- Mapping categorical variables to numerical values.
- Scaling numerical features.

## Model Training

The MLP model is trained using the following architecture:
- Input layer
- Four hidden layers with ReLU activation
- Output layer with sigmoid activation

Early stopping is used to prevent overfitting.

## Model Evaluation

The model is evaluated using:
- Test loss and accuracy
- Confusion matrix
- ROC curve

## Usage

1. **Train the model**:
   ```sh
   python modelo.py