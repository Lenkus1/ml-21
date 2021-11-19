from pathlib import Path
from loguru import logger
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_iris() -> pd.DataFrame:
    logger.info("Obtained iris dataset")
    df = sns.load_dataset("iris")
    return df


def scale(X: np.ndarray) -> np.ndarray:
    logger.info("Run standardscaler on data.")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


def main(
    target: str, split: float = 0.2, seed: int = 123, validation: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.array, np.ndarray, np.ndarray]:
    df = get_iris()
    y = df[target].values
    data = df.drop(target, axis=1)
    labels = list(data.columns)
    X = scale(data.values)
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
