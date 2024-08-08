# Disease Prediction using Decision Tree Classifier

### DISCLAIMER
This is the first time using ChatGPT to create documentation of my project. I think it did a pretty good job!! Thanks Chat

This project implements a machine learning model to predict diseases based on symptoms using a Decision Tree Classifier. The model is trained on a dataset containing symptom severity weights and their associated diseases. The primary objective is to predict the correct disease based on input symptoms and analyze the performance of the model with different hyperparameters.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [Model Training](#model-training)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
- `dataset/`: Contains the dataset files required for training and testing.
  - `Symptom-severity.csv`: Symptom severity weights.
  - `reformated_dataset.csv`: Reformatted dataset for training.
  - `symptom_precaution.csv`: Precautions associated with each disease.
  - `symptom_Description.csv`: Descriptions of each disease.
- `decision_tree.png`: Visualization of the trained decision tree.
- `README.md`: Documentation for the project.

## Installation
To run this project, ensure you have the following dependencies installed:
- Python 3.x
- Pandas
- Seaborn
- Scikit-learn
- Matplotlib
- CSV


## Usage
1. Clone the repository.
2. Place your datasets in the `dataset/` directory.
3. Run the `train_model()` function to train the model on the dataset.
4. Use the `predict_disease()` function to predict a disease based on symptoms.

Example:

```python
from your_module import train_model, predict_disease

clf, symptom_columns = train_model()
test_symptoms = ["vomiting", "dehydration", "diarrhoea"]
predicted_disease = predict_disease(clf, symptom_columns, test_symptoms, severity_dic, disease_dict)

## Functions

### `load_symptom_weights(file_path)`
Loads symptom weights from a CSV file into a dictionary.

### `train_multiple(X_train, y_train, min_samples_split)`
Trains a Decision Tree model with a specified `min_samples_split` parameter.

### `train_model()`
Trains the model with different hyperparameters and returns the model with the best accuracy.

### `plot_classification_report(y_true, y_pred)`
Plots the classification report metrics using Seaborn.

### `predict_disease(clf, symptom_columns, symptoms, severity_dic, disease_dict)`
Predicts the disease based on input symptoms.

### `combined_sum(sum_dict)`
Computes the combined sum of severity scores for each disease.

### `plot_data(range_set, names)`
Plots the range of symptom severity scores on a number line.

### `sym_range_compute()`
Computes the range of severity scores for each disease.

### `get_disease_info(disease_name)`
Retrieves information and precautions for a given disease.

## Model Training
The `train_model()` function reads in the dataset, trains the model with various hyperparameters, and selects the best-performing model based on accuracy. The decision tree model's performance is visualized, and the best hyperparameter settings are printed.

## Visualization
The project provides visualizations for the decision tree and classification report metrics. These can be used to understand the model's decision-making process and evaluate its performance.

Example of a Decision Tree visualization:

