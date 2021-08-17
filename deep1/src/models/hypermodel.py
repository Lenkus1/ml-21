from typing import Dict

import tensorflow as tf


class hypermodel(tf.keras.Model):
    def __init__(self: tf.keras.Model, config: Dict) -> None:
        super(hypermodel, self).__init__()

        self.x1 = tf.keras.layers.Dense(
            config["units"], activation=config["activation"]
        )
        self.hidden = [
            tf.keras.layers.Dense(
                units=config["units"], activation=config["activation"]
            )
            for _ in range(config["dense_layers"])
        ]
        self.out = tf.keras.layers.Dense(1)

    def call(self: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:
        # print("self:", type(self))
        # print("input", type(x))
        x = self.x1(x)
        for layer in self.hidden:
            x = layer(x)
        x = self.out(x)
        return x
