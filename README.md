# Introduction
This repository contains the source code and files for a machine learning model designed to predict Pokemon fight winner. This README file is intended to provide a brief overview of the repository's contents and the structure of the files included.

# Files
## `data/`
This folder contains the raw as well as preprocessed data used to train and test the model. The data is stored in CSV format and includes features and labels used in the model and is provided by the course instructor.

### `data/combats.csv`
This file contains the training data for the model.

### `data/pokemon_data.csv`
This file contains the data for each pokemon, including its id, name, stats, generation and whether it is legendary or not.

### `data/preprocessed.csv`
This file contains a preprocessed and feature engineered data from pokemon_data.csv.

### `data/test.csv`
This file contains test data on which the model will run and predict the winner.

## `notebooks/`
This folder contains Jupyter notebooks that were used for data exploration, data preprocessing, feature engineering, model selection, and model evaluation.

### `notebooks/data_visualisation.ipynb`
This file contains the notebook used for plotting the various graphs which are used in this project.

### `notebooks/preprocessing_and_feature_sel.ipynb`
This file contains the notebook used for preprocessing and feature selection of the data.

 ### `notebooks/training.ipynb`
 This file contains the various models used for training data. 

## `pictures/`
This folder contains the various pictures used in the project

### `pictures/graphs`
This folder contains all of the graphs and plots constructed during the course of these project.

 ## `src/`
 This folder contains the source code for the machine learning model. The source code is written in Python and includes modules for data preprocessing, feature engineering and model selection.

 ### `src/main.py`
 This is the main file, running it will run the training model and return the results in a csv file '`data\result.csv`'. 

### `src/preprocessing_and_feature_sel.py`
This file contains the source code used for preprocessing the data which includes *one hot encoding* the features 'Type 1' and 'Type 2', removing the unncessary features - Pokemon Name, Type 1 and Type 2 after one hot encoding, generation and legendary, and returns it in the form of a csv file '`data\preprocessed.csv`'.

### `src/training.py`
This file contains the source code used for training the data using various models. The models used are -
* Decision Tree Classifier
* Logistic Regression
* Random Forest Classifier
* k-Nearest Neighbours algorithm
  
  Out of these, Random Forest classifier is used for testing data after some parameter tuning due to having the maximum accuracy and f-1 score among these.

##  `requirements.txt`
This file contains the list of Python packages required to run the model.

## `README.md`
This file provides an overview of the contents of this repository.'

# Getting Started
To get started, clone this repository to your local machine and navigate to the root directory. Ensure that you have installed all the necessary packages listed in the `requirements.txt` file.

**Note:**  *The model is developed on a linux-based environment so running it on a Windows environment may result in errors, ensure that the path to the csv files are changed according to Windows path conventions.*

# Usage
Once you have installed the necessary packages, you can run the model by executing the `main.py` script in the `src` folder. 

This will import the files '`data/combats.csv`' and '`data/pokemon_data.csv`', create a file '`data/preprocessed.csv`' which contains the dataset after preprocessing and finally train the model, provide results on the console and predict the data provided in '`data/test.csv`' in a file '`data/result.csv`'.

For prediction of your own dataset, create a csv in the format of the file '`data/test.csv`' and replace it with the same file.
