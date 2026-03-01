# System Design: Automated Stock Price Prediction

## 1. Overview
The **Automated Stock Price Prediction System** is an end-to-end Machine Learning pipeline designed to forecast future stock prices. Instead of relying on a static set of features, the system dynamically selects the most relevant technical indicators for a given stock using tree-based methods (XGBoost) before training a deep learning model (1D CNN).

## 2. Core Architecture

The system is fully orchestrated using **DVC (Data Version Control)** to manage the execution graph and ensure reproducibility.

### Pipeline Stages
1. **Data Collection (`get_data.py`)**: Fetches historical Open, High, Low, Close, Volume (OHLCV) data using the Yahoo Finance API (`yfinance`).
2. **Feature Engineering (`feature_engineering.py`)**: Computes a wide variety of technical indicators (e.g., MACD, RSI, Bollinger Bands, Moving Averages).
3. **Feature Selection (`feature_selection.py`)**: Trains an XGBoost regressor on the engineered features and extracts feature importances. Only the top `N` features (configurable in `params.yaml`) are kept to reduce noise and dimensionality.
4. **Model Training (`training.py`)**: 
   - Prepares sequences (sliding windows) from the selected features.
   - Trains a 1D Convolutional Neural Network (PyTorch) to predict the $t+1$ closing price.
   - Logs hyperparameters, loss curves, and model weights to **MLflow**.
5. **Model Evaluation & Registry (`log_production_model.py`)**: Compares the newly trained model against the current best model in the MLflow Model Registry. If it performs better (lower MSE), it is promoted and saved to the `saved_models/` directory for deployment.

### MLflow & Tracking
- **Backend Store**: SQLite (`mlflow.db`)
- **Artifacts**: Stored locally in `./artifacts`
- **Metrics Tracked**: Train Loss, Validation Loss, MSE.

## 3. Design Choices & Trade-offs

* **CNN vs. LSTM/RNN**: While LSTMs are traditional for time-series, 1D CNNs offer significantly faster training times and are often highly effective at capturing local temporal patterns (e.g., sudden spikes or drops over a rolling window).
* **Dynamic Feature Selection**: Stock behavior varies wildly between sectors and individual tickers. A static set of indicators might work for Apple but fail for a volatile biotech stock. Using XGBoost for dynamic feature selection ensures the CNN only learns from the most predictive signals for that specific asset.
* **DVC Orchestration**: Rather than writing custom bash scripts, `dvc.yaml` provides a formalized DAG where stages are only re-run if their inputs or parameters change, saving compute time during experimentation.

## 4. Future Enhancements
* **Transformer Architectures**: Swapping the CNN for a Time-Series Transformer (e.g., Informer).
* **Multi-variate Targets**: Predicting not just price, but volatility or volume simultaneously.
* **Live Deployment**: Wrapping the output in a FastAPI endpoint and setting up a cron job for daily inference.
