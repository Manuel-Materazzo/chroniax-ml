from typing import Tuple

import numpy as np


class BinnedMedianFitter:
    """Handles binned median fitting operations."""

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Bin x values and compute median y values; drop empty bins."""
        indices = np.digitize(x, bins) - 1
        x_medians, y_medians = [], []

        for bin_idx in range(len(bins) - 1):
            mask = indices == bin_idx
            if np.any(mask):
                x_medians.append(np.median(x[mask]))
                y_medians.append(np.median(y[mask]))

        return np.array(x_medians), np.array(y_medians)
