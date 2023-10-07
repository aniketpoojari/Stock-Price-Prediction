import argparse
from common import read_params
import torch
from tqdm import tqdm
import torch.nn as nn
import mlflow
import itertools
from urllib.parse import urlparse
import warnings

warnings.filterwarnings("ignore")


class BATCH_CONV(nn.Module):
    def __init__(self, num_feat, channel_size, lookback, kernel_size, peek):
        super(BATCH_CONV, self).__init__()

        self.conv0 = nn.Conv2d(1, channel_size, kernel_size=(1, num_feat), stride=1)
        self.conv_layers = nn.ModuleList()

        # making layes with decreasing size
        while True:
            new_channel_size = channel_size // 2
            new_lookback = lookback - kernel_size + 1
            if new_channel_size > 0 and new_lookback > 0:
                con = nn.Conv2d(
                    channel_size,
                    new_channel_size,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                )
                self.conv_layers.append(con)
                channel_size = new_channel_size
                lookback = new_lookback
            else:
                break

        self.fc = nn.Linear(lookback * channel_size, peek)

    def forward(self, x):
        x = self.conv0(x)
        for c in self.conv_layers:
            x = c(x)
        x = torch.flatten(x)
        out = self.fc(x)
        return out


def tuning(
    data,
    num_col,
    lookback,
    peek,
    first_conv_channels,
    epoches,
    kernel_size,
    learning_rate,
):
    # make model
    model = BATCH_CONV(num_col, first_conv_channels, lookback, kernel_size, peek)
    criterion = torch.nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training
    tr_loss = None
    for t in tqdm(range(epoches)):
        epoch_losses = []
        for i in range(0, len(data[0])):
            y_train_pred = model(data[0][i].unsqueeze(0))
            loss = criterion(y_train_pred.cpu(), data[1][i])
            epoch_losses.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        tr_loss = sum(epoch_losses) / len(epoch_losses)

    # validation
    val_losses = []
    for i in range(0, len(data[2])):
        y_val_pred = model(data[2][i].unsqueeze(0))
        loss = criterion(y_val_pred.cpu(), data[3][i])
        val_losses.append(loss.item())
    val_loss = sum(val_losses) / len(val_losses)
    return tr_loss, val_loss, model


def training(config_path):
    config = read_params(config_path)

    batch_training_x_data_pt = config["batch_data_maker"]["batch_training_x_data_pt"]
    batch_training_y_data_pt = config["batch_data_maker"]["batch_training_y_data_pt"]
    batch_validation_x_data_pt = config["batch_data_maker"][
        "batch_validation_x_data_pt"
    ]
    batch_validation_y_data_pt = config["batch_data_maker"][
        "batch_validation_y_data_pt"
    ]
    lookback = config["batch_data_maker"]["lookback"]
    peek = config["batch_data_maker"]["peek"]
    first_layer_size = config["training"]["first_layer_size"]
    epochs = config["training"]["epochs"]
    kernel_size = config["training"]["kernel_size"]
    learning_rate = config["training"]["learning_rate"]

    batch_training_x = torch.load(batch_training_x_data_pt)
    batch_training_y = torch.load(batch_training_y_data_pt)
    batch_validation_x = torch.load(batch_validation_x_data_pt)
    batch_validation_y = torch.load(batch_validation_y_data_pt)

    mlflow_config = config["mlflow_config"]

    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(mlflow_config["experiment_name"])

    dic = {
        "first_layer_size": first_layer_size,
        "epoches": epochs,
        "kernel_size": kernel_size,
        "learning_rate": learning_rate,
    }

    combinations = itertools.product(*list(dic.values()))

    for i in combinations:
        with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
            t, v, model = tuning(
                data=[
                    batch_training_x,
                    batch_training_y,
                    batch_validation_x,
                    batch_validation_y,
                ],
                num_col=batch_training_x.shape[2],
                lookback=lookback,
                peek=peek,
                first_conv_channels=i[0],
                epoches=i[1],
                kernel_size=i[2],
                learning_rate=i[3],
            )
            mlflow.log_param("feat", batch_training_x.shape[2])
            mlflow.log_param("lookback", lookback)
            mlflow.log_param("peek", peek)
            mlflow.log_param("first_conv_channels", i[0])
            mlflow.log_param("epoches", i[1])
            mlflow.log_param("kernel_size", i[2])
            mlflow.log_param("learnig_rate", i[3])
            mlflow.log_metric("train_loss", t)
            mlflow.log_metric("val_loss", v)

            tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(
                    model,
                    "model",
                    registered_model_name=mlflow_config["registered_model_name"],
                )
            else:
                mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
