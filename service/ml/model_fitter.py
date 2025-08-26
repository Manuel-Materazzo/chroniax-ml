from typing import Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

from dto.enums.model_kind import ModelKind
from dto.model_meta import ModelMeta
from service.ml.binned_median_fitter import BinnedMedianFitter


class ModelFitter:
    """Factory class for creating different types of heart rate models."""

    DEFAULT_CLIP_RANGE = (35, 220)
    DEFAULT_NUM_BINS = 15
    MIN_POINTS_FOR_PCHIP = 3

    @classmethod
    def fit_isotonic(cls, x: np.ndarray, y: np.ndarray,
                     context: str = 'global',
                     clip_range: Tuple[float, float] = DEFAULT_CLIP_RANGE) -> ModelMeta:
        """Fit an isotonic regression model."""
        # Sort by x values
        sort_order = np.argsort(x)
        x_sorted, y_sorted = x[sort_order], y[sort_order]

        isotonic_reg = IsotonicRegression(increasing=True, out_of_bounds='clip')
        y_fitted = isotonic_reg.fit_transform(x_sorted, y_sorted)

        # Extract unique x values and corresponding fitted y values as knots
        x_knots, unique_indices = np.unique(x_sorted, return_index=True)
        y_knots = y_fitted[unique_indices]

        return ModelMeta(
            kind=ModelKind.ISOTONIC.value,
            context=context,
            x_knots=x_knots.tolist(),
            y_knots=y_knots.tolist(),
            clip_lo=float(clip_range[0]),
            clip_hi=float(clip_range[1])
        )

    @classmethod
    def fit_pchip_from_binned(cls, x: np.ndarray, y: np.ndarray,
                              context: str = 'global',
                              clip_range: Tuple[float, float] = DEFAULT_CLIP_RANGE,
                              num_bins: int = DEFAULT_NUM_BINS) -> ModelMeta:
        """Fit a PCHIP model using binned medians for monotone-friendly knots."""
        x_min, x_max = np.nanmin(x), np.nanmax(x)
        bins = np.linspace(x_min, x_max, num_bins + 1)

        binned_x, binned_y = BinnedMedianFitter.fit(x, y, bins)

        # Ensure strictly increasing x values for PCHIP
        unique_indices = np.unique(binned_x, return_index=True)[1]
        binned_x, binned_y = binned_x[unique_indices], binned_y[unique_indices]

        # Fallback to isotonic if insufficient points
        if len(binned_x) < cls.MIN_POINTS_FOR_PCHIP:
            return cls.fit_isotonic(x, y, context=context, clip_range=clip_range)

        return ModelMeta(
            kind=ModelKind.PCHIP.value,
            context=context,
            x_knots=binned_x.tolist(),
            y_knots=binned_y.tolist(),
            clip_lo=float(clip_range[0]),
            clip_hi=float(clip_range[1])
        )
