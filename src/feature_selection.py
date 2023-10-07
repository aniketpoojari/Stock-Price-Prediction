import argparse
from email.mime import base
from common import read_params
import pandas as pd
import xgboost as xgb

def add_target_col(data, tar_col, peek):
    ncol = "{}+{}".format(tar_col, str(peek))
    data[ncol] = data[tar_col].shift(-peek)
    data = data.dropna()
    return data, ncol

def split_data(data, split, ncol):
    div = int(data.shape[0] * split)
    train = data.iloc[:div] 
    val = data.iloc[div:]
    train_x = train.drop(ncol, axis = 1)
    train_y = train[ncol]
    val_x = val.drop(ncol, axis = 1) 
    val_y = val[ncol]
    return train_x, train_y, val_x, val_y

def define_model(gamma, n_estimators, base_score, learning_rate):

    model = xgb.XGBRegressor(
        gamma = gamma, 
        n_estimators = n_estimators, 
        base_score = base_score, 
        learning_rate = learning_rate
    )

    return model

def imp_cols(model, train_x, train_y, val_x, val_y, threshold):
    
    xgbModel = model.fit(
        train_x,
        train_y,
        eval_set = [(train_x, train_y), (val_x, val_y)], 
        verbose = False
        )
    
    # extracting important valiable
    imp_col = [i[0] for i in [i for i in list(zip(train_x, xgbModel.feature_importances_)) if i[1] > threshold]]
    
    return imp_col

def feature_selection(config_path):

    # getting configs
    config = read_params(config_path)
    feature_engineering_data_csv = config["feature_engineering"]["feature_engineering_data_csv"]
    target_column = config["base"]["target_column"]
    peek = config["feature_selection"]["peek"]
    split = config["feature_selection"]["split"]
    gamma = config["feature_selection"]["gamma"]
    n_estimators = config["feature_selection"]["n_estimators"]
    base_score = config["feature_selection"]["base_score"]
    learning_rate = config["feature_selection"]["learning_rate"]
    threshold = config["feature_selection"]["threshold"]
    feature_selection_data_csv = config["feature_selection"]["feature_selection_data_csv"]
    
    # reading data
    data = pd.read_csv(feature_engineering_data_csv)
    
    # adding target column
    data, new_target_col = add_target_col(data, target_column, peek)
    
    # splitting data
    train_x, train_y, val_x, val_y = split_data(data, split, new_target_col)
    
    # define model
    model = define_model(gamma, n_estimators, base_score, learning_rate)
    
    # get important columns
    cols = imp_cols(model, train_x, train_y, val_x, val_y, threshold)
    
    # save data
    data = data[cols]
    data.to_csv(feature_selection_data_csv, index=False)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    feature_selection(config_path=parsed_args.config)