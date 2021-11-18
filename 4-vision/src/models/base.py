from typing import Tuple

import tensorflow as tf


def base_imagemodel(shape: Tuple, classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=shape),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(classes),  # no activation here
        ]
    )

    return model
