## 📝 Description

- In this project we have used CNN to predict future values of a stock using multiple stock data.

## ⏳ Dataset

- Here we use Yahoo API to get stock data

## Experimentation

- Check notebooks folder to see all the experientation done before creating the final pipeline

## Pipeline

- DVC is used to create pipeline
- Change the values in the params.yaml to get different models based on the requirement

## Final Model

- saved_models folder will contain the final model after the pipeline is executed using MLFlow

## Requirements

- Use `pip install -r requirements.txt` to install the requirements

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
git clone https://github.com/aniketpoojari/NYC-Taxi-Trip-Duration-Prediction.git

# Change directory
cd NYC-Taxi-Trip-Duration-Prediction

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```python
# Add your data in the data\raw folder and track it by DVC using the command:
dvc add data\raw\<filename>

# Run mlflow server in the background before running the pipeline
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0

# Change values in the params.yaml file

# training
dvc repro
```

## 📊 Data

- The data is from the [kaggle competetion](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)
- Data fields

  - id - a unique identifier for each trip
  - vendor_id - a code indicating the provider associated with the trip record
  - pickup_datetime - date and time when the meter was engaged
  - dropoff_datetime - date and time when the meter was disengaged
  - passenger_count - the number of passengers in the vehicle (driver entered value)
  - pickup_longitude - the longitude where the meter was engaged
  - pickup_latitude - the latitude where the meter was engaged
  - dropoff_longitude - the longitude where the meter was disengaged
  - dropoff_latitude - the latitude where the meter was disengaged
  - store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
  - trip_duration - duration of the trip in seconds

- Check [notebook](notebooks/NYC-Taxi-Trip-Duration-Prediction.ipynb) to look at all the _Exploratory Data Anlaysis_ and _Experimentations_ done.

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
