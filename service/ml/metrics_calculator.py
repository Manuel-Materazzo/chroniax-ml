import pandas as pd

from service.ml.heart_rate_classifier import HeartRateZoneClassifier


class MetricsCalculator:
    """Calculates evaluation metrics for model performance."""

    @staticmethod
    def calculate_metrics(df: pd.DataFrame, pred_col: str = 'y_hat') -> pd.DataFrame:
        """Calculate MAE, bias, and other metrics grouped by heart rate zones."""
        df_copy = df.copy()
        df_copy['abs_err'] = (df_copy['t10_bpm'] - df_copy[pred_col]).abs()
        df_copy['bias'] = df_copy['t10_bpm'] - df_copy[pred_col]
        df_copy['zone'] = df_copy['scan_bpm'].apply(HeartRateZoneClassifier.classify)

        metrics = df_copy.groupby('zone').agg(
            n=('abs_err', 'size'),
            MAE=('abs_err', 'mean'),
            bias=('bias', 'mean')
        ).reset_index()

        return metrics