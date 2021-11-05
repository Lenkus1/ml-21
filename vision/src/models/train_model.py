from pathlib import Path
from typing import Any

import tensorflow as tf


def train(
    log_dir: Path,
    checkpoint_dir: Path,
    model: tf.keras.Model,
    traingen: tf.keras.preprocessing.image.ImageDataGenerator,
    validgen: tf.keras.preprocessing.image.ImageDataGenerator,
    totalepochs: int = 10,
    initial_epochs: int = 1,
    verbose: int = 0,
    save: bool = False,
) -> Any:
    tb = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)

    if save:
        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir, verbose=0, save_best_only=True
        )
        callbacks = [tb, checkpointer]
    else:
        callbacks = [tb]

    hist = model.fit(
        traingen,
        validation_data=validgen,
        callbacks=callbacks,
        epochs=totalepochs,
        initial_epoch=initial_epochs,
        verbose=verbose,
    )
    return hist.history
