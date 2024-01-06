# 🚀 Project Name

Stock Price prediction

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)

## 📄 Introduction

It is difficult to predict the price of a stock in the future manually as it is difficult to go through all of the indicators and check which one works better for which stock and even then stock data is susceptible to randomness. Therefore, here I an developing a single system to predict the future price of any stock by dynamically selecting the best indicators and the best model for the stock.

In this project I have used CNN to predict future values of a stock using multiple stock data.

## 🌟 Features

- [Exploratory Data Analysis and Experiments](notebooks/Stock-Price-Prediction.ipynb)
- [Data Collection](src/get_data.py)
- [Feature Engineering](src/feature_engineering.py)
- [Feature Selection](src/feature_selection.py)
- [Training](src/training.py)
- [Best Model Selection](src/log_production_model.py)

## 🛠️ Requirements

- pandas
- yfinance
- seaborn
- statsmodels
- xgboost
- scikit-learn
- torch
- torchvision
- torchaudio
- tdqm
- dvc
- mlflow
- psycopg2

## 🚚 Installation

```bash
# Clone the repository
git clone https://github.com/aniketpoojari/Stock-Price-Prediction.git

# Change directory
cd Stock-Price-Prediction

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```python
# Run mlflow server in the background before running the pipeline
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0

# Change values in the params.yaml file

# training
dvc repro
```

## 📊 Data

- Here we use Yahoo API to get stock data
- Check [notebook](notebooks/Stock-Price-Prediction.ipynb) to look at all the _Exploratory Data Anlaysis_ and _Experimentations_ done.

## 🤖 Model Training

```bash
# Train the model
dvc repro
```

## 📈 Evaluation

- R2 score is used to evaluate the model

## 🎉 Results

- Go to localhost:5000/ to look at results on MLflow server.
- saved_models folder will contain the final model after the pipeline is executed using MLFlow
