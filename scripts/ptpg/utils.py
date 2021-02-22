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
"""Library of utility and helper modules"""
from typing import Tuple
import json
import random

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelBinarizer,
    MinMaxScaler
)

__author__ = "Justin Gerard E. Ricardo"


def load_dataset_from_file(
    dataset_path: str,
    feature_range: Tuple[int] = None,
    lbl_encoding: Tuple[int] = None,
    random_seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the dataset from a file.

    Parameters
    ----------
    dataset_path: str
        File path of the dataset.
    feature_range: list, optional, default: None
        Tuple containing the min and max value for each feature's range.
    lbl_encoding: list, optional, default: None
        Tuple containing the encoding for positive and negative labels.
    random_seed: int, optional, default: None
        Seed used for dataset split reproducibility.

    Returns
    -------
    X_train: numpy.ndarray
        Training subset for features
    y_train: numpy.ndarray
        Training subset for target labels
    X_test: numpy.ndarray
        Test subset for features
    y_test: numpy.ndarray
        Test subset for target labels
    """
    dataset = np.genfromtxt(dataset_path, delimiter=',')
    # slice all rows and up until 2nd last column
    data_X = dataset[:, :dataset.shape[1]-1]
    # slice all rows and only last column
    data_y = dataset[:, dataset.shape[1]-1].astype(int)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data_X,
        data_y,
        test_size=0.2,
        stratify=data_y,
        random_state=random_seed
    )

    # rescale features to specified range
    if isinstance(feature_range, tuple):
        minmaxsc = MinMaxScaler(feature_range=feature_range)
        X_train = minmaxsc.fit_transform(X_train)
        X_test = minmaxsc.transform(X_test)

    # label transformations to match NN output layer
    onehotenc = OneHotEncoder(sparse=False)
    onehotenc.fit(data_y.reshape(-1, 1))

    y_train = onehotenc.transform(y_train.reshape(-1, 1))
    y_test = onehotenc.transform(y_test.reshape(-1, 1))

    # encode one-hot labels to specified positive and negative values
    if isinstance(lbl_encoding, tuple):
        lbl_binarizer = LabelBinarizer(
            neg_label=lbl_encoding[0],
            pos_label=lbl_encoding[1]
        )
        lbl_binarizer.fit(y_train)

        y_train = lbl_binarizer.transform(y_train)
        y_test = lbl_binarizer.transform(y_test)

    return (X_train, y_train, X_test, y_test)


def set_global_seed(seed: int) -> None:
    """
    Set all pseudorandom functions to the same seed

    Parameter
    ---------
    seed: int
        global seed to use for all pseudorandom generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_data_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = None,
    num_workers: int = 0
) -> Tuple[torch.utils.data.DataLoader]:
    """
    Creates the data loader for the train and test dataset.

    Parameters
    ---------
    features: numpy.ndarray
        Training input features.
    labels: numpy.ndarray
        Training labels.
    batch_size: int
        Batch size
    shuffle: bool, optional, default: None
        Flag indicating whether to shuffle dataset or not.
    num_workers: int, optional, default: 0
        Number of workers to use during sampling iteration.
    Returns
    -------
    torch.utils.data.DataLoader: data loader
    """
    features = torch.Tensor(features)
    labels = torch.Tensor(labels)

    dataset = torch.utils.data.TensorDataset(features, labels)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    return data_loader


def encode_labels(labels: np.ndarray, encoding: tuple) -> np.ndarray:
    """
    Encode labels in another encoding.

    Parameter
    ---------
    labels: numpy.ndarray
        One-hot encoded labels.
    encoding: tuple
        Coding scheme for the labels.
    Returns
    -------
    numpy.ndarray: Transformed labels.
    """
    lbl_bin = LabelBinarizer(encoding[0], encoding[1])
    return lbl_bin.fit_transform(labels)


def acc_score(y_pred: np.ndarray, y_labels: np.ndarray) -> float:
    """
    Evaluate performance between predictions and ground truth labels.

    Parameters
    ----------
    y_pred: numpy.ndarray
        Numpy list of one-hot encoded predictions or logits.
    y_labels: numpy.ndarray
        Numpy list of ground truth labels.

    Returns
    -------
    float:
        Accuracy score of model's predictions.
    """
    correct = 0
    total = y_pred.shape[0]
    for i in range(total):
        predicted = np.argmax(y_pred[i])
        test = np.argmax(y_labels[i])
        correct = correct + (1 if predicted == test else 0)

    return (float(correct)/float(total))*100.


def torch_acc_score(
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
    return (y_pred.argmax(1) == y_true.argmax(1)).sum().item() / len(y_true)


def export_to_json(
    results_data: dict, filename: str,
    sort_keys: bool = True, indent: int = 4
) -> None:
    """
    Export dictionary to a JSON file.

    Parameters
    ----------
    results_data: dict
        Dictionary containing results data.
    filename: str
        Output file name.
    sort_keys: bool, optional, default: True
        Flag to sort keys in the dictionary, including inner dictionaries.
    indent: int, optional, default: 4
        Number of tabs to insert per level.
    """
    with open(filename, 'w') as out_file:
        json.dump(results_data, out_file, sort_keys=sort_keys, indent=indent)


def load_results_from_json(filename: str) -> dict:
    """
    Parses a JSON file and returns results as a dictionary.

    Parameters
    ----------
    filename: str
        Input file name to load.
    sort_keys: bool, optional, default: True
        Flag to sort keys in the dictionary, including inner dictionaries.
    """
    with open(filename, 'r') as in_file:
        data = json.load(in_file)

    return data


def make_json_readable(
    data: dict, sort_keys: bool = True, indent: int = 4
) -> str:
    """
    Converts a dictionary into a human readable string.

    Parameters
    ----------
    data: dict
        Input file name to load.
    sort_keys: bool, optional, default: True
        Flag to sort keys in the dictionary, including inner dictionaries.
    indent: int, optional, default: 4
        Number of tabs to insert per level.
    """
    return json.dumps(data, sort_keys=sort_keys, indent=indent)


def load_datasets(
    dataset_paths: dict,
    feature_range: Tuple[int, int] = None,
    label_encoding: Tuple[int, int] = None,
    random_seed: int = None
) -> dict:
    """
    Helper function to load multiple datasets with consistent
    feature ranges and label encodings.

    Parameters
    ---------
    dataset_paths: dict
        A dictionary of dataset paths with custom keys.
    feature_range: Tuple[int, int], optional, default=None
        Feature range to use for all the dataset features.
    label_encoding: Tuple[int, int], optional, default=None
        Label encoding to use for all the dataset labels.
    random_seed: int, optional, default=None
        Fixed random seed to use for each dataset.

    Returns
    -------
    datasets: dict
        A dictionary of datasets using the same custom keys
        as the passed dictionary of paths (for easier lookup).
    """
    datasets = {}
    for key in dataset_paths.keys():
        set_global_seed(random_seed)
        dataset = load_dataset_from_file(
            dataset_path=dataset_paths[key],
            feature_range=feature_range,
            lbl_encoding=label_encoding,
            random_seed=random_seed
        )
        datasets[key] = dataset

    return datasets


def send_message(key: str, value: any) -> None:
    """
    Wrapper for print. Serializes data in JSON format.

    Parameters
    ----------
    key: str
        The dictionary key.
    value: any
        The list or single item for the dictionary value.
    """
    print(json.dumps({key: value}, indent=2, sort_keys=True))
