# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
from loguru import logger
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split(
    data: np.ndarray, target: np.ndarray, train_size: float
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """takes data and target in, outputs tuples for train, valid and test, with
    every tuple consisting of an X and y

    Args: data (npt.ArrayLike): input training data (X) target (npt.ArrayLike):
        input target data (y) train_size (float): float to split data into train
        and test set

    Returns: Tuple[Tuple[npt.ArrayLike, npt.ArrayLike], Tuple[npt.ArrayLike,
        npt.ArrayLike]]: train, valid and test tuple, where every set contains a
        tuple (X,y)
    """

    logger.info(f"Splitting data into {train_size} trainsize.")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        data, target, train_size=train_size
    )
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def scale(X: List[np.ndarray]) -> List[np.ndarray]:
    """takes as input a list of arrays, with a minimum of 2 items. Expects the
    first item to be the trainset, fits the scaler on the first item, and
    transforms the rest. Returns a Tuple with all sets.

    Args: X (Iterable[npt.ArrayLike]): [description]

    Returns: Tuple[npt.ArrayLike]: [description]
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[0])
    sets = [scaler.transform(x) for x in X[1:]]

    return [X_train, *sets]


def save_raw(raw: Union[str, Path]) -> Any:
    raw = Path(raw)
    datafile = raw / "data.npy"
    targetfile = raw / "target.npy"
    housing = fetch_california_housing()
    np.save(datafile, housing.data)
    np.save(targetfile, housing.target)

    logger.info(f"Saving data to {raw}")
    return housing


def process(input_filepath: Path, output_filepath: Path) -> None:
    logger.info("making final data set from raw data")
    data = save_raw(input_filepath)
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split(
        data.data, data.target, train_size=0.8
    )

    logger.info("Scaling data")
    X_train, X_valid, X_test = scale([X_train, X_valid, X_test])
    data = np.array(
        [(X_train, y_train), (X_valid, y_valid), (X_test, y_test)], dtype=object
    )
    datafile = Path(output_filepath) / "data.npy"
    logger.info(f"Saving scaled data to {datafile}")
    np.save(datafile, data)


def load(
    datapath: Path,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    train, valid, test = np.load(datapath, allow_pickle=True)
    return (train, valid, test)
