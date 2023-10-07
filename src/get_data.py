import argparse
import datetime
import yfinance as yf
from common import read_params

def get_data(config_path):
    config = read_params(config_path)
    raw_data_csv = config["get_data"]["raw_data_csv"]
    symbol = config["get_data"]["symbol"]
    start = config["get_data"]["start"]
    end = (datetime.datetime.today() - datetime.timedelta(days = 1)).strftime('%Y-%m-%d')
    data = yf.download(tickers = symbol, start = start, end = end)
    data.to_csv(raw_data_csv)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    get_data(config_path=parsed_args.config)