import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def melted_boxplot(
    X: np.ndarray,
    y: np.ndarray,
    labels: List[str],
    labelname: str = "labels",
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    p = pd.DataFrame(X, columns=labels)
    p[labelname] = y
    p = p.melt(id_vars=labelname)
    plt.figure(figsize=figsize)
    sns.boxplot(data=p, x="variable", y="value", hue=labelname)
    plt.xticks(rotation=90)


def gridsearch_heatmap(
    gridresults: GridSearchCV,
    param_grid: Dict,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 10),
):
    """Takes as input the result of a sklearn GridSearchCV and the parametergrid
    used to do the search. Outputs a heatmap

    Args:
        gridresults (GridSearchCV)
        param_grid (Dict): A named dict used for tuning
        vmin (Optional[float], optional): Minimum threshold for colors. Defaults to None.
        vmax (Optional[float], optional): Maximum threshold for colers. Defaults to None.
        figsize (Tuple[int, int], optional): Changes size of plot. Defaults to (10, 10).
    """
    idx, col = ["param_" + [*param_grid.keys()][i] for i in range(2)]
    pivoted = pd.pivot_table(
        pd.DataFrame(gridresults.cv_results_),
        values="mean_test_score",
        index=idx,
        columns=col,
    )
    pivoted.index = ["{:.4f}".format(x) for x in pivoted.index]
    pivoted.columns = ["{:.4f}".format(x) for x in pivoted.columns]
    plt.figure(figsize=figsize)
    sns.heatmap(pivoted, vmin=vmin, vmax=vmax, annot=True)
