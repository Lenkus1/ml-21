import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Input,
    Dropout,
    BatchNormalization,
    Reshape,
    Conv2D,
    MaxPool2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow import keras
from tensorflow_addons.layers import GELU
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import Model

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


def get_baseline_conv(dropout: float = 0.2):
    model = Sequential(
        [
            Reshape((28, 28, 1)),
            Rescaling(1.0 / 255),
            Conv2D(16, (3, 3), activation="relu"),
            MaxPool2D((2, 2)),
            Flatten(),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(dropout),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(dropout),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(dropout),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model


def get_cnn_model(hp):
    # input are 2D images
    input = Input(shape=[28, 28])
    x = Rescaling(1.0 / 255)(input)
    # but we need to add a channel for color if we want to use Conv2D layers
    x = Reshape((28, 28, 1))(x)

    filters = hp.Int("filters", 16, 64, 4)
    x = Conv2D(filters, (3, 3), activation="relu")(x)
    x = MaxPool2D((2, 2))(x)

    for i in range(hp.Int("conv_layers", 0, 2)):
        x = Conv2D(filters, (3, 3), activation="relu")(x)
        x = MaxPool2D((2, 2))(x)
        name = "convlayer_{0}".format(i)

    flat = Flatten()(x)

    units = hp.Int("units", 128, 320, 64)
    drops = hp.Float("drops", 0.1, 0.4)
    leak = hp.Float("leak", 0, 0.2)

    x = Dense(units)(flat)
    x = LeakyReLU(alpha=leak)(x)
    x = BatchNormalization()(x)
    x = Dropout(drops)(x)

    for i in range(hp.Int("dense_layers", 1, 5)):
        name = "layer_{0}".format(i)
        x = Dense(units=units)(x)
        x = LeakyReLU(alpha=leak)(x)
        x = BatchNormalization()(x)
        x = Dropout(drops)(x)

    output = Dense(10, activation="softmax")(x)
    model = Model(inputs=[input], outputs=[output])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model
