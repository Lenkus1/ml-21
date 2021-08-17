from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def movingaverage(interval: List, window_size: int) -> np.ndarray:
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


def plot_results(
    result: Dict[str, Dict],
    ymin: int = 0,
    ymax: int = None,
    yscale: str = "linear",
    moving: int = 5,
    alpha: float = 0.5,
) -> None:
    """workaround for tensorboard

    Args:
        result (Dict[str, Dict]): dictionary of history.history dicts
        ymin (int, optional): minimum y axis
        ymax (int, optional): maximum y axis. Defaults to None.
        yscale (str, optional): Linear of log scaling of y-axis. Defaults to "linear".
        moving (int, optional): Overlay a moving average. Defaults to 5.
        alpha (float, optional): transparancy. Defaults to 0.5.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    move = type(moving) == int

    for key in result.keys():
        loss = result[key]["loss"]
        if move:
            z = movingaverage(loss, moving)
            z = np.concatenate([[np.nan] * moving, z[moving:-moving]])
            color = next(ax1._get_lines.prop_cycler)["color"]
            ax1.plot(z, label=key, color=color)
            ax1.plot(loss, label=key, alpha=alpha, color=color)
        else:
            ax1.plot(loss, label=key)

        ax1.set_yscale(yscale)
        ax1.set_ylim(ymin, ymax)
        ax1.set_title("train")

        valloss = result[key]["val_loss"]

        if move:
            z = movingaverage(valloss, moving)
            z = np.concatenate([[np.nan] * moving, z[moving:-moving]])
            color = next(ax2._get_lines.prop_cycler)["color"]
            ax2.plot(z, label=key, color=color)
            ax2.plot(valloss, label=key, alpha=alpha, color=color)
        else:
            ax2.plot(valloss, label=key)

        ax2.set_yscale(yscale)
        ax2.set_ylim(ymin, ymax)
        ax2.set_title("test")

    plt.legend()
