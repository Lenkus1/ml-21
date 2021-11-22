from pathlib import Path
from loguru import logger
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
import sys

sys.path.insert(0, "..")


def scale(X: np.ndarray) -> np.ndarray:
    logger.info("Run standardscaler on data.")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


def set_number(
    y_train: np.ndarray, y_test: np.ndarray, nbr: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    return (y_train == nbr, y_test == nbr)


def main(
    split: float = 0.2, seed: int = 123, validation: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.array, np.ndarray, np.ndarray]:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if validation:
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_test, y_test, test_size=split / 2, random_state=seed
        )
        return (X_train, y_train, X_valid, y_valid, X_test, y_test)
    else:
        return (X_train, y_train, X_test, y_test)


def generate_linear_data(n: int, k: int, noise_range: float = 1):
    """
    This generates data for n cases and  k features
    Returns a tuple of X, y and weights.
    """
    bias = np.random.randint(low=1, high=5, size=(1, k))
    data = np.random.rand(n, k) + bias

    # we specify weights for every feature
    W = np.random.randint(low=5, high=25, size=(k, 1))
    # add some uniform noise
    noise = np.random.uniform(low=-noise_range, high=noise_range, size=(n, 1))
    # and multiply every feature with a weight
    y = np.dot(data, W) + noise

    return (data, y, W)


def get_cancer_data(
    split: float, seed: int = 123, validation: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.array]:
    # we load the data
    df = pd.read_csv("../src/data/cancer_data.csv", index_col="id")
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, random_state=seed
    )

    if validation:
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_test, y_test, test_size=split / 2, random_state=seed
        )
        return (X_train, y_train, X_valid, y_valid, X_test, y_test)
    else:
        return (X_train, y_train, X_test, y_test)
