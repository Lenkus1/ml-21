import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Tuple
import matplotlib.pyplot as plt


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
