# PyTorch Playground
# Copyright (C) 2021  Justin Gerard Ricardo
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
"""Implementation of the 1-layer NN and MLR"""
import os
import time

import torch

from ptpg.utils import send_message

__author__ = "Justin Gerard E. Ricardo"


class ANN(torch.nn.Module):
    """
    Implementation of a 1-hidden layer neural network
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 1e-2
    ):
        """
        Constructs the 1-hidden layer neural network
        model class and its internal layers.

        Parameters
        ---------
        input_size: int
            The number of nodse in the input layer.
        hidden_size: int
            The number of nodes in the hidden layer.
        output_size: int
            The number of nodes in the output layer.
        epochs: int
            The number of times to cycle the training dataset.

        learning_rate: float, default: 1e-2
            The learning rate
        """
        super(ANN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.model = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.input_size, out_features=self.hidden_size
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=self.hidden_size, out_features=self.output_size
            ),
            torch.nn.Sigmoid()
        )

        self.model_device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.optimizer = torch.optim.SGD(
            params=self.parameters(), lr=self.learning_rate
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.model_device)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset all parameters.
        """
        self.model = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.input_size, out_features=self.hidden_size
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=self.hidden_size, out_features=self.output_size
            ),
            torch.nn.Sigmoid()
        )
        for index, layer in enumerate(self.model):
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()
                torch.nn.init.normal_(layer.weight)
        self.optimizer = torch.optim.SGD(
            params=self.parameters(), lr=self.learning_rate
        )

    def forward(self, X) -> torch.Tensor:
        """
        Performs a forward-pass on an input feature set.

        Parameters
        ----------
        X: torch.Tensor
            The input features.

        Returns
        -------
        pred: torch.Tensor
            The model output.
        """
        if not isinstance(X, torch.Tensor):
            X = X.astype("float32")
            X = torch.from_numpy(X)

        pred = self.model(X)

        return pred

    def predict(self, features) -> torch.Tensor:
        """
        Performs a forward-pass on an input feature set.

        Parameters
        ----------
        X: torch.Tensor
            The input features.

        Returns
        -------
        torch.Tensor
            The model output.
        """
        return self.forward(features)

    def fit(
        self,
        train_data_loader: torch.utils.data.DataLoader,
        X_test: torch.Tensor,
        epochs: int,
        verbose: bool = False,
        X_train: torch.Tensor = None,
        y_train: torch.Tensor = None,
        y_test: torch.Tensor = None
    ) -> None:
        """
        Trains the 1-hidden layer neural network model.

        Parameters
        ----------
        data_loader: torch.utils.dataloader.DataLoader
            The data loader object that contains the training
            data pipeline (grouped in batches of size 1)
        X_test: torch.Tensor
            The subset for evaluating model per epoch.
        epochs: int
            The number of times the model will cycle over the entire dataset
        verbose: bool, optional, default: False
            Flag indicating whether to display epoch loss or not.
        """
        self.epochs = epochs
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        self.to(self.model_device)
        for epoch in range(epochs):
            train_epoch_loss = self.epoch_train(train_data_loader)
            self.train_loss.append(train_epoch_loss)

            if verbose:
                delta = self.train_loss[-2]-self.train_loss[-1] \
                    if len(self.train_loss) > 1 else self.train_loss[0]
                print("Epoch ({}/{}): MSE Loss = {:.6f} Delta({:.6f})".format(
                    epoch+1, epochs, self.train_loss[-1], delta
                    )
                )

            # epoch test loss & accuracy
            if y_test is not None:
                if X_test.dtype != torch.float32:
                    X_test = X_test.float()
                test_pred = self.forward(X_test)
                test_epoch_loss = self.criterion(
                    test_pred, torch.argmax(y_test, 1)
                ).item()
                self.test_loss.append(test_epoch_loss)
                test_epoch_acc = self.acc_score(test_pred, y_test)
                self.test_acc.append(test_epoch_acc)

            # epoch train accuracy
            if X_train is not None and y_train is not None:
                if X_train.dtype != torch.float32:
                    X_train = X_train.float()
                train_pred = self.forward(X_train)
                train_epoch_acc = self.acc_score(train_pred, y_train)
                self.train_acc.append(train_epoch_acc)

            time.sleep(2)

            # prep message data
            epoch_data = {
                "W1": self.model[0].weight.detach().numpy().tolist(),
                "W2": self.model[2].weight.detach().numpy().tolist(),
                "epoch_loss": {
                    "train": self.train_loss[-1],
                    "test": self.test_loss[-1]
                },
                "epoch_acc": {
                    "train": self.train_acc[-1]*100.,
                    "test": self.test_acc[-1]*100.
                },
                "epoch": epoch
            }
            send_message("epoch_data", epoch_data)

    def epoch_train(
        self, data_loader: object
    ) -> float:
        """
        An epoch in the model training process.

        Parameters
        ----------
        model: torch.nn.Module
            The model that will be trained.
        data_loader: torch.utils.dataloader.DataLoader
            The data loader object that contains the
            training data pipeline (grouped in batches of size 1)

        Returns
        -------
        epoch_loss: float
            Epoch loss.
        """
        epoch_loss = 0
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(self.model_device)
            batch_labels = batch_labels.to(self.model_device)

            self.optimizer.zero_grad()
            pred = self.model(batch_features)
            train_loss = self.criterion(
                pred,
                torch.argmax(batch_labels, 1)
            )
            train_loss.backward()
            self.optimizer.step()

            epoch_loss += train_loss.item()

        epoch_loss /= len(data_loader)
        return epoch_loss

    def acc_score(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor
    ) -> float:
        """
        Evaluate performance between predictions and ground truth labels.

        Parameters
        ----------
        y_pred: torch.Tensor
            Model predictions.
        y_true: numpy.ndarray
            Ground truth labels.
        Returns
        -------
        float: Accuracy score of model's predictions.
        """
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.from_numpy(y_pred)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.from_numpy(y_true)
        return (
            y_pred.argmax(1) == y_true.argmax(1)
        ).sum().item() / len(y_true)

    def save_model(self, savepath: str) -> None:
        """
        Export model.

        Parameter
        ---------
        savepath: str
            File path of the model to export.
        """
        if not os.path.exists(os.path.dirname(savepath)):
            os.mkdir(os.path.dirname(savepath))
        torch.save(self.state_dict(), savepath)

    def load_model(self, filename: str) -> None:
        """
        Loads an external ANN model.

        Parameter
        ---------
        filename: str
            File path of where the model is located.
        """
        if os.path.isfile(filename):
            self.load_state_dict(torch.load(filename))
        else:
            raise IOError(f"Filepath not found: {filename}")

    def __repr__(self):
        """
        Official string representation of this model object.

        Returns
        -------
        str: Official string representation
        """
        return "ANN"

    def __str__(self):
        """
        Informal string representation of this model object.

        Returns
        -------
        str: Informal string representation
        """
        return "ANN"
