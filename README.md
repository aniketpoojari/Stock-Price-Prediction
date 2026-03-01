# Automated Stock Price Prediction System

An end-to-end machine learning pipeline that predicts future stock prices by dynamically selecting the most relevant technical indicators and training a Convolutional Neural Network (CNN). 

Stock market data is inherently noisy. Instead of manually guessing which technical indicators (RSI, MACD, Moving Averages, etc.) work best for a specific stock, this system uses an **XGBoost-assisted feature selection** step to dynamically identify the strongest signals before feeding them into a deep learning model.

## 🌟 Key Features

- **Dynamic Feature Selection**: Uses XGBoost feature importance to automatically select the best technical indicators for any given stock.
- **CNN Forecasting**: Formulates time-series forecasting as a 1D Convolutional Neural Network problem using PyTorch.
- **Automated Data Pipeline**: Pulls live historical data from the Yahoo Finance API (`yfinance`).
- **MLOps Integration**: Fully orchestrated using **DVC** for data/pipeline versioning and **MLflow** for experiment tracking and model registry.

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch (1D CNN)
- **Feature Engineering**: Pandas, XGBoost
- **MLOps**: DVC, MLflow
- **Data Source**: Yahoo Finance API (`yfinance`)

## 🚚 Installation

```bash
# Clone the repository
git clone https://github.com/aniketpoojari/Stock-Price-Prediction.git
cd Stock-Price-Prediction

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

The entire training process is managed via DVC. Before running the pipeline, start the MLflow tracking server:

```bash
# Run mlflow server in the background
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0
```

Configure your parameters (stock ticker, dates, hyperparameters) in `params.yaml`, then run the pipeline:

```bash
# Execute the end-to-end pipeline (Data Download -> Engineering -> Selection -> Train -> Log)
dvc repro
```

## 📈 Evaluation & Results

- The model uses **Mean Squared Error (MSE)** as its primary evaluation metric.
- Navigate to `http://localhost:5000/` to view the MLflow dashboard, compare experiments, and analyze training loss curves.
- The best-performing model is automatically exported and saved in the `saved_models` directory by the `log_production_model.py` script.
