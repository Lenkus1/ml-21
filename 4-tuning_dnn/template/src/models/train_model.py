import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Input,
    Dropout,
    BatchNormalization,
    Reshape,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow import keras
from tensorflow_addons.layers import GELU
from tensorflow.keras.layers import LeakyReLU

from typing import Tuple, Optional, Union


def get_baseline(
    batchnorm: bool = False, dropout: Union[float, Tuple[float, float, float]] = 0.0
):

    if type(dropout) == float:
        dropout = [dropout] * 3

    if not batchnorm:
        model = Sequential(
            [
                Flatten(),
                Rescaling(1.0 / 255),
                Dense(256, activation="relu"),
                Dropout(dropout[0]),
                Dense(256, activation="relu"),
                Dropout(dropout[1]),
                Dense(256, activation="relu"),
                Dropout(dropout[2]),
                Dense(10, activation="softmax"),
            ]
        )
    else:
        model = Sequential(
            [
                Flatten(),
                Rescaling(1.0 / 255),
                Dense(256, activation="relu"),
                BatchNormalization(),
                Dropout(dropout[0]),
                Dense(256, activation="relu"),
                BatchNormalization(),
                Dropout(dropout[1]),
                Dense(256, activation="relu"),
                BatchNormalization(),
                Dropout(dropout[2]),
                Dense(10, activation="softmax"),
            ]
        )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model


def get_selu_model():
    model = Sequential(
        [
            Dense(256, activation="selu", kernel_initializer="lecun_normal"),
            Dense(256, activation="selu", kernel_initializer="lecun_normal"),
            Dense(256, activation="selu", kernel_initializer="lecun_normal"),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model


def get_gelu_model(batchnorm: bool = False, dropout: Optional[float] = None):
    if not batchnorm:
        model = Sequential(
            [
                Flatten(),
                Rescaling(1.0 / 255),
                Dense(256),
                GELU(),
                Dense(256),
                GELU(),
                Dense(256),
                GELU(),
                Dense(10, activation="softmax"),
            ]
        )
    else:
        model = Sequential(
            [
                Flatten(),
                Rescaling(1.0 / 255),
                Dense(256),
                GELU(),
                BatchNormalization(),
                Dense(256),
                GELU(),
                BatchNormalization(),
                Dense(256),
                GELU(),
                BatchNormalization(),
                Dense(10, activation="softmax"),
            ]
        )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model


def get_elu_model():
    model = Sequential(
        [
            Flatten(),
            Rescaling(1.0 / 255),
            Dense(256, activation="elu"),
            Dense(256, activation="elu"),
            Dense(256, activation="elu"),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model


def get_leakyrelu_model(batchnorm: bool = False, dropout: Optional[float] = None):
    if not batchnorm:
        model = Sequential(
            [
                Flatten(),
                Rescaling(1.0 / 255),
                Dense(256),
                LeakyReLU(alpha=0.01),
                Dense(256),
                LeakyReLU(alpha=0.01),
                Dense(256),
                LeakyReLU(alpha=0.01),
                Dense(10, activation="softmax"),
            ]
        )
    else:
        model = Sequential(
            [
                Flatten(),
                Rescaling(1.0 / 255),
                Dense(256),
                LeakyReLU(alpha=0.01),
                BatchNormalization(),
                Dense(256),
                LeakyReLU(alpha=0.01),
                BatchNormalization(),
                Dense(256),
                LeakyReLU(alpha=0.01),
                BatchNormalization(),
                Dense(10, activation="softmax"),
            ]
        )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model
