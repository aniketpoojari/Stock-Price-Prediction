import pickle
import argparse
from common import read_params
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch


def add_target_column(data, target_column, peek):
    # name of new column
    new_target_column = "Pred_" + target_column

    # save pred column in variable
    ref_col = data[target_column]

    # saving future values in a variable
    pred_data = data[-peek:]
    pred_data[new_target_column] = [[0] * peek] * peek
    # print(pred_data)

    # removing the future values as they are to be predicted
    data = data[:-peek]

    # adding index
    data["row"] = data.index

    # making the prediction data
    data[new_target_column] = data["row"].apply(
        lambda x: ref_col[x + 1 : x + peek + 1].values
    )
    data = data.drop(["row"], axis=1)
    data = data.dropna()
    data = pd.concat([data, pred_data], axis=0)

    return data, new_target_column


def split_data(data, new_target_column, split):
    div = int(data.shape[0] * split)

    X = data.drop(new_target_column, axis=1)
    Y = data[new_target_column]

    train_x = X[:div]
    train_y = Y[:div]

    val_x = X[div:]
    val_y = Y[div:]

    return train_x, train_y, val_x, val_y


def scaling_data(train_x, val_x):
    scaler = MinMaxScaler().fit(train_x)

    train_x = scaler.transform(train_x)
    val_x = scaler.transform(val_x)

    return train_x, val_x


def making_batches(train_x, train_y, val_x, val_y, lookback):
    batch_training_x = []
    batch_training_y = []
    batch_validation_x = []
    batch_validation_y = []

    for index in range(train_x.shape[0] - lookback):
        batch_training_x.append(train_x[index : index + lookback])
        batch_training_y.append(train_y.iloc[index + lookback])

    for index in range(val_x.shape[0] - lookback):
        batch_validation_x.append(val_x[index : index + lookback])
        batch_validation_y.append(val_y.iloc[index + lookback])

    batch_training_x = torch.from_numpy(np.array(batch_training_x)).float()
    batch_training_y = torch.from_numpy(np.array(batch_training_y)).float()
    batch_validation_x = torch.from_numpy(np.array(batch_validation_x)).float()
    batch_validation_y = torch.from_numpy(np.array(batch_validation_y)).float()

    return batch_training_x, batch_training_y, batch_validation_x, batch_validation_y


def batch_data_maker(config_path):
    config = read_params(config_path)

    feature_selection_data_csv = config["feature_selection"][
        "feature_selection_data_csv"
    ]
    target_column = config["base"]["target_column"]
    split = config["batch_data_maker"]["split"]
    peek = config["batch_data_maker"]["peek"]
    lookback = config["batch_data_maker"]["lookback"]
    batch_training_x_data_pt = config["batch_data_maker"]["batch_training_x_data_pt"]
    batch_training_y_data_pt = config["batch_data_maker"]["batch_training_y_data_pt"]
    batch_validation_x_data_pt = config["batch_data_maker"][
        "batch_validation_x_data_pt"
    ]
    batch_validation_y_data_pt = config["batch_data_maker"][
        "batch_validation_y_data_pt"
    ]
    pred_data_pt = config["batch_data_maker"]["pred_data_pt"]

    data = pd.read_csv(feature_selection_data_csv)

    data, new_target_column = add_target_column(data, target_column, peek)
    train_x, train_y, val_x, val_y = split_data(data, new_target_column, split)
    train_x, val_x = scaling_data(train_x, val_x)

    val_y = val_y[:-peek]
    pred_data = val_x[-lookback:]
    print(pred_data.shape)
    val_x = val_x[:-peek]

    (
        batch_training_x,
        batch_training_y,
        batch_validation_x,
        batch_validation_y,
    ) = making_batches(train_x, train_y, val_x, val_y, lookback)

    torch.save(batch_training_x, batch_training_x_data_pt)
    torch.save(batch_training_y, batch_training_y_data_pt)
    torch.save(batch_validation_x, batch_validation_x_data_pt)
    torch.save(batch_validation_y, batch_validation_y_data_pt)

    pred_data = torch.from_numpy(np.array(pred_data)).float()
    torch.save(pred_data, pred_data_pt)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    batch_data_maker(config_path=parsed_args.config)
