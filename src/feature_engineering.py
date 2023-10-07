import argparse
from common import read_params
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def fill_missing(df):
    # remove date as index
    df = df.reset_index()
    # change datatype of Date column
    df.Date = df.Date.astype("str")
    # getting list of complete dates from the range
    dates = pd.date_range(
        start=df.iloc[0]["Date"], end=df.iloc[-1]["Date"], freq="D"
    ).astype("str")
    # making new full dataframe
    new_df = pd.DataFrame(columns=df.columns)
    for date in dates:
        if len(df[df["Date"] == date]):
            new_df.loc[len(new_df.index)] = df[df["Date"] == date].values[0]
        else:
            # filling missing date with previous date values
            row = new_df.iloc[len(new_df.index) - 1].values
            row[1] = date
            new_df.loc[len(new_df.index)] = row
    new_df = new_df.drop(["index", "Date"], axis=1)
    return new_df


def tech_indi(df):
    # 44 days moving average
    df["ma44"] = df["Close"].rolling(window=44).mean()
    # create 7 and 21 day moving average
    df["ma7"] = df["Close"].rolling(window=7).mean()
    df["ma21"] = df["Close"].rolling(window=21).mean()
    # create MACD
    df["26ema"] = df["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
    df["12ema"] = df["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
    df["MACD"] = df["12ema"] - df["26ema"]
    # create bollinger bands
    df["20sd"] = df["Close"].rolling(window=20).std()
    df["upper_band"] = df["Close"].rolling(window=20).mean() + (df["20sd"] * 2)
    df["lower_band"] = df["Close"].rolling(window=20).mean() - (df["20sd"] * 2)
    # create exponential moving average
    df["ema"] = df["Close"].ewm(com=0.5).mean()
    # create momentum
    df["momentum"] = (df["Close"] / 100) - 1
    return df


def fourier(df):
    close_fft = np.fft.fft(np.asarray(df["Close"].tolist()))
    fft_df = pd.DataFrame({"fft": close_fft})
    fft_list = np.asarray(fft_df["fft"].tolist())
    fft_list_m10 = np.copy(fft_list)
    fft_list_m10[100:-100] = 0
    df["Fourier"] = pd.DataFrame(fft_list_m10).apply(lambda x: np.abs(x))
    return df


def price_movement(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["movement"] = scaler.fit_transform(df[["Close"]])
    return df


def slope(df, c):
    df["slope"] = 0
    df["row"] = df.index
    df["slope"] = df["row"].apply(
        lambda x: (df["Close"][x] - df["Close"][max(0, x - c)]) / c
    )
    df = df.drop(["row"], axis=1)
    return df


def dtype_correction(df):
    dtypes = df.dtypes.reset_index()
    for i in range(dtypes.shape[0]):
        if dtypes.iloc[i][0] == "object":
            df[dtypes.iloc[i]["index"]] = df[dtypes.iloc[i]["index"]].astype("float64")
    return df


def feat_eng(config_path):
    config = read_params(config_path)
    raw_data_csv = config["get_data"]["raw_data_csv"]
    feature_engineering_data_csv = config["feature_engineering"][
        "feature_engineering_data_csv"
    ]

    data = pd.read_csv(raw_data_csv)

    data = fill_missing(data)
    data = tech_indi(data)
    data = fourier(data)
    data = price_movement(data)
    data = slope(data, 10)
    data = dtype_correction(data)

    data.to_csv(feature_engineering_data_csv, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    feat_eng(config_path=parsed_args.config)
