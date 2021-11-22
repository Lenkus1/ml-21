from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from typing import Dict, List, Callable, Optional, Tuple
import numpy as np
from src.visualization import visualize


def search_and_fit(
    model,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List[float]],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 10),
    cv: int = 3,
):
    grid_search = GridSearchCV(model, param_grid, cv=cv)
    grid_search.fit(X, y)

    visualize.gridsearch_heatmap(
        grid_search, param_grid, vmin=vmin, vmax=vmax, figsize=figsize
    )
    return grid_search
