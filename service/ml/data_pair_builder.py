from typing import Optional

import pandas as pd

from util.file_utils import get_scanwatch_data, get_t10_data
from util.scanwatch_utils import resample_scanwatch_by_overlap, annotate_context


class DataPairBuilder:
    """Builds paired datasets from ScanWatch and T10 data."""

    HR_VALID_RANGE = (30, 230)
    MIN_T10_POINTS = 1

    @classmethod
    def build_pairs(cls, scan_csv: str, sqlite_path: str, user_id: Optional[int],
                    bin_size: str = "1min", min_scan_coverage_s: int = None,
                    local_tz: str = None) -> pd.DataFrame:
        """Build paired minute-level data from ScanWatch and T10 sources."""
        # Load ScanWatch data
        sw_raw = get_scanwatch_data(scan_csv)
        sw_resampled = resample_scanwatch_by_overlap(sw_raw, min_scan_coverage_s, freq=bin_size)

        # Load T10 data with context
        t10_resampled, df_sleep, df_sport = get_t10_data(
            sqlite_path, local_tz, user_id=user_id, bin_size=bin_size
        )

        # Join datasets
        paired_df = pd.merge(
            t10_resampled, sw_resampled,
            left_on='window_utc', right_on='window_utc',
            how='inner'
        )

        # Apply filters
        paired_df = cls._apply_filters(paired_df)

        # Add context annotations
        paired_df = annotate_context(paired_df, df_sleep, df_sport, freq=bin_size)
        paired_df = cls._add_temporal_features(paired_df)

        return paired_df.sort_values('window_utc').reset_index(drop=True)

    @classmethod
    def _apply_filters(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality filters."""
        # Filter heart rate ranges
        hr_filter = (
                df['t10_bpm'].between(*cls.HR_VALID_RANGE) &
                df['scan_bpm'].between(*cls.HR_VALID_RANGE)
        )

        # Filter minimum T10 points
        points_filter = df['t10_points'] >= cls.MIN_T10_POINTS

        return df[hr_filter & points_filter].copy()

    @staticmethod
    def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features like hour and day of week."""
        df['hour'] = df['window_utc'].dt.hour
        df['dow'] = df['window_utc'].dt.dayofweek
        return df
