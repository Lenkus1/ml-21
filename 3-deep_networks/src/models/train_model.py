from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from ray import tune
from ray.tune import JupyterNotebookReporter
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer

from src.data import make_dataset
from src.models import hypermodel


def train(
    model: Model,
    train_data: Tuple[np.ndarray, np.ndarray],
    valid_data: Tuple[np.ndarray, np.ndarray],
    epochs: int = 20,
    batch_size: int = 32,
    loss: str = "mse",
    optimizer: Union[str, Optimizer] = "Adam",
    metrics: List[str] = ["mae", "mape"],
    callbacks: List = [],
    verbose: int = 1,
) -> Any:
    """train a give model with train and validation data

    Args: model (Model): keras model train_data (Tuple[npt.ArrayLike]): tuple of
        X and y valid_data (Tuple[npt.ArrayLike]): tuple of X and y epochs (int,
        optional): amount of epochs to train. Defaults to 20. batch_size (int,
        optional): batch size. Defaults to 32. loss (str, optional): string for
        loss function. Defaults to "mse". optimizer (str, optional): string for
        optimizer. Defaults to "Adam". metrics (List[str], optional): List of
        strings with metrics. Defaults to ["mae", "mape"].

    Returns: Dict: returns the history.history dictionary with loss and metrics
        for train and valid
    """
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    X_train, y_train = train_data
    X_valid, y_valid = valid_data
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_valid, y_valid),
        callbacks=callbacks,
        verbose=verbose,
    )
    return history.history


def train_hypermodel(config: Dict, verbose: int = 0, tuning: bool = True) -> None:
    """Use a config file to train the hypermodel

    Args: config (Dict): the config dict verbose (int, optional): Output
        information to screen (1) or not (0). Defaults to 0. tuning (bool,
        optional): Switch between TuneReportCallback (if true) and Earlystopping
        plus tensorboard (if false). Defaults to True.
    """

    train, valid, test = make_dataset.load(config["datafile"])
    X_train, y_train = train
    X_valid, y_valid = valid

    model = hypermodel.hypermodel(config)

    if tuning:
        callbacks = [
            TuneReportCallback({"mean_accuracy": "mape", "val_loss": "val_loss"})
        ]
    else:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir=config["log_dir"], histogram_freq=1),
        ]

    model.compile(loss="mse", optimizer=config["optimizer"], metrics=["mape"])

    model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=config["epochs"],
        verbose=verbose,
        validation_data=(X_valid, y_valid),
        callbacks=callbacks,
    )


def hypertune(
    iterations: int, config: Dict
) -> tune.analysis.experiment_analysis.ExperimentAnalysis:

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=200, grace_period=5
    )

    reporter = JupyterNotebookReporter(overwrite=True)

    analysis = tune.run(
        train_hypermodel,
        name="hypertune",
        scheduler=sched,
        metric="val_loss",
        mode="min",
        progress_reporter=reporter,
        local_dir=config["local_dir"],
        stop={"training_iteration": iterations},
        num_samples=config["samples"],
        resources_per_trial={"cpu": 4, "gpu": 0},
        config=config,
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    return analysis
