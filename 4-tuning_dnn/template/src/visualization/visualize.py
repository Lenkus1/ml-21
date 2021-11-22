import matplotlib.pyplot as plt
import re
import numpy as np
from typing import Tuple, Optional


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


def plot_results(
    result: dict,
    ymin: float = 0,
    ymax: float = None,
    yscale: str = "linear",
    moving: Optional[int] = None,
    alpha: float = 0.5,
    patience: int = 1,
    subset: str = ".",
    grid: bool = False,
    figsize: Tuple[int, int] = (15, 10),
):

    if not grid:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        move = type(moving) == int

        for key in result.keys():
            if bool(re.search(subset, key)):
                loss = result[key].history["loss"]
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

                valloss = result[key].history["val_loss"]

                if move:
                    z = movingaverage(valloss, moving)
                    z = np.concatenate([[np.nan] * moving, z[moving:-moving]])[
                        :-patience
                    ]
                    color = next(ax2._get_lines.prop_cycler)["color"]
                    ax2.plot(z, label=key, color=color)
                    ax2.plot(valloss, label=key, alpha=alpha, color=color)
                else:
                    ax2.plot(valloss[:-patience], label=key)

                ax2.set_yscale(yscale)
                ax2.set_ylim(ymin, ymax)
                ax2.set_title("valid")

        plt.legend()
    if grid:

        keyset = list(filter(lambda x: re.search(subset, x), [*result.keys()]))
        gridsize = int(np.ceil(np.sqrt(len(keyset))))

        plt.figure(figsize=(15, 15))
        for i, key in enumerate(keyset):
            ax = plt.subplot(gridsize, gridsize, i + 1)
            loss = result[key].history["loss"]
            valloss = result[key].history["val_loss"]
            plt.plot(loss, label="train")
            plt.ylim(0, ymax)
            plt.plot(valloss, label="valid")
            plt.title(key)
            plt.legend()


def plot_scores(
    score: dict, ymin: float = 0, ymax: float = 1, figsize: Tuple[int, int] = (8, 8)
) -> None:
    plt.figure(figsize=figsize)
    sorted_dict = {k: v for k, v in sorted(score.items(), key=lambda item: item[1][1])}
    allkeys = sorted_dict.keys()
    for key in allkeys:
        plt.bar(key, sorted_dict[key])
    plt.ylim(ymin, ymax)
    x = [*range(len(allkeys))]
    plt.xticks(x, allkeys, rotation=-90)
