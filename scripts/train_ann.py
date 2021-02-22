# PyTorch Playground
# Copyright (C) 2021 Justin Gerard Ricardo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Script for training a 1-hidden layer ANN model"""
import time
import argparse

import torch

from ptpg.utils import (
    load_dataset_from_file,
    set_global_seed,
    torch_data_loader,
    torch_acc_score,
    send_message
)
from ptpg.models import ANN


def arg_parse() -> None:
    """
    Parses arguments passed from a terminal execution.
    """
    parser = argparse.ArgumentParser()
    args = {
        "--dataset": {
            "type": str,
            "help": "The dataset you want to train the WISHNET on.",
            "default": "linear"
        },
        "--hidden_nodes": {
            "type": int,
            "help": "The number of nodes in the hidden layer.",
            "default": 10
        },
        "--learning_rate": {
            "type": float,
            "help": "The learning rate (alpha).",
            "default": 0.01
        },
        "--epochs": {
            "type": int,
            "help": "The number of epochs to train the WISHNET.",
            "default": 1
        },
        "--script_dir": {
            "type": str,
            "help": "The script's directory.",
            "default": ""
        }
    }

    for key in args.keys():
        parser.add_argument(
            key,
            type=args[key]["type"],
            help=args[key]["help"],
            default=args[key]["default"]
        )

    parsed_args = parser.parse_args()

    return parsed_args


def main(args):
    set_global_seed(1024)

    abs_prefix = args.script_dir.replace('\\', '/')+'/scripts/'
    dataset_paths = {
        "linear": abs_prefix+"data/synthetic/linear_dataset-500.csv",
        "square_root": abs_prefix+"data/synthetic/square_root_dataset-500.csv",
        "hyperbola": abs_prefix+"data/synthetic/hyperbola_dataset-500.csv",
        "circle": abs_prefix+"data/synthetic/circle_dataset-500.csv",
        "checkered_2x2": abs_prefix+"data/synthetic/checkered_2x2_dataset-500.csv",
        "checkered_4x4": abs_prefix+"data/synthetic/checkered_4x4_dataset-500.csv"
    }

    X_train, y_train, X_test, y_test = load_dataset_from_file(
        dataset_path=dataset_paths[args.dataset],
        random_seed=1024,
        feature_range=None, lbl_encoding=(-1, 1)
    )

    train_data_loader = torch_data_loader(
        features=X_train, labels=y_train,
        batch_size=1, shuffle=False, num_workers=0
    )

    model = ANN(
        input_size=X_train.shape[1],
        hidden_size=args.hidden_nodes,
        output_size=y_train.shape[1],
        learning_rate=args.learning_rate,
    )

    init_data = {
        "epochs": args.epochs,
        "epoch_steps": X_train.shape[0],
        "start_time": time.time(),
        "model": str(model),
        "W1": model.model[0].weight.detach().numpy().tolist(),
        "W2": model.model[2].weight.detach().numpy().tolist()
    }
    send_message("init_data", init_data)

    model.fit(
        train_data_loader=train_data_loader,
        X_test=torch.from_numpy(X_test),
        epochs=args.epochs,
        verbose=False,
        X_train=torch.from_numpy(X_train),
        y_train=torch.from_numpy(y_train),
        y_test=torch.from_numpy(y_test)
    )

    pred_train = model.predict(features=X_train)
    pred_test = model.predict(features=X_test)

    acc_train = torch_acc_score(y_pred=pred_train, y_true=y_train)
    acc_test = torch_acc_score(y_pred=pred_test, y_true=y_test)

    print("Train Accuracy: {:.6f}".format(acc_train))
    print("Test Accuracy: {:.6f}".format(acc_test))


if __name__ == "__main__":
    args = arg_parse()
    main(args)
