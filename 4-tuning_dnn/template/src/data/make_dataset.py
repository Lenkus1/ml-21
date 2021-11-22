from pathlib import Path
from loguru import logger
import numpy as np
from typing import Tuple, Optional
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split


def get_mnist(
    split: float = 0.2, seed: int = 123, validation: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.array, np.ndarray, np.ndarray]:
    (X_train, y_train), (X_valid, y_valid) = mnist.load_data()
    if validation:
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_valid, y_valid, test_size=split, random_state=seed
        )
        return (X_train, y_train, X_valid, y_valid, X_test, y_test)
    else:
        return (X_train, y_train, X_valid, y_valid)


def scale(
    scaler,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_trains = scaler.fit_transform(X_train.reshape(len(X_train), -1))
    X_valids = scaler.transform(X_valid.reshape(len(X_valid), -1))
    if X_test is not None:
        X_tests = scaler.transform(X_test.reshape(len(X_test), -1))
        return X_trains, X_valids, X_tests
    else:
        return X_trains, X_valids
