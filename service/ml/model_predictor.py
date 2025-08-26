import numpy as np
from scipy.interpolate import PchipInterpolator

from dto.enums.model_kind import ModelKind
from dto.model_meta import ModelMeta


class ModelPredictor:
    """Handles model application and prediction."""

    @staticmethod
    def apply_model(meta: ModelMeta, x: np.ndarray) -> np.ndarray:
        """Apply a trained model to input data."""
        x_array = np.asarray(x, dtype=float)
        x_clipped = np.clip(x_array, meta.x_knots[0], meta.x_knots[-1])

        x_knots_array = np.array(meta.x_knots)
        y_knots_array = np.array(meta.y_knots)

        if meta.kind == ModelKind.ISOTONIC.value:
            return np.interp(x_clipped, x_knots_array, y_knots_array)
        elif meta.kind == ModelKind.PCHIP.value:
            interpolator = PchipInterpolator(x_knots_array, y_knots_array, extrapolate=True)
            y_predicted = interpolator(x_clipped)
            return np.clip(y_predicted, meta.clip_lo, meta.clip_hi)
        else:
            raise ValueError(f"Unknown model kind: {meta.kind}")
