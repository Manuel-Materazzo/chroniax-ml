from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from pandas import Timestamp, DataFrame, Timedelta


def resample_scanwatch_by_overlap(df: DataFrame, min_coverage_s: int, freq: str = "1min") -> DataFrame:
    """
    Resample scanwatch data using time-weighted aggregation into frequency windows.

    Args:
        df: DataFrame with columns 'start_utc', 'durations', 'values'
        freq: Resampling frequency (e.g., '1min', '5min')

    Returns:
        DataFrame with columns 'window_utc', 'scan_bpm', 'scan_coverage_s'
    """
    if df.empty:
        return _create_empty_result()

    step = pd.to_timedelta(freq)
    window_data = defaultdict(lambda: {'weighted_sum': 0.0, 'coverage': 0.0})

    for row in df.itertuples():
        _process_row(row, step, freq, window_data)

    if not window_data:
        return _create_empty_result()

    return _build_result_dataframe(window_data, freq, min_coverage_s)


def annotate_context(min_df: pd.DataFrame, df_sleep: pd.DataFrame, df_sport: pd.DataFrame,
                     freq: str = "1min") -> pd.DataFrame:
    """
    Annotate minute-level data with sleep status and sport activity flags.

    Args:
        min_df: DataFrame with 'window_utc' column
        df_sleep: DataFrame with 'start_utc', 'end_utc', 'status' columns
        df_sport: DataFrame with 'start_utc', 'end_utc' columns
        freq: Time frequency for window duration

    Returns:
        DataFrame with added 'sleep_status' and 'is_sport' columns
    """
    # Initialize columns for empty input
    if min_df.empty:
        return _add_empty_columns(min_df)

    # Calculate time windows
    step = pd.to_timedelta(freq)
    window_starts = min_df['window_utc'].to_numpy()
    window_ends = (min_df['window_utc'] + step).to_numpy()

    # Annotate sleep status and sport activity
    sleep_status = _calculate_sleep_status(window_starts, window_ends, df_sleep)
    is_sport = _calculate_sport_flags(window_starts, window_ends, df_sport)

    # Add columns to dataframe
    min_df = min_df.copy()
    min_df['sleep_status'] = pd.Series(sleep_status, index=min_df.index, dtype='Int64')
    min_df['is_sport'] = is_sport

    return min_df


def _create_empty_result() -> DataFrame:
    """Create empty result DataFrame with correct columns."""
    return DataFrame(columns=['window_utc', 'scan_bpm', 'scan_coverage_s'])


def _process_row(row, step: Timedelta, freq: str, window_data: dict) -> None:
    """Process a single row of data and update window_data."""
    durations = row.durations
    values = row.values

    if not _is_valid_row_data(durations, values):
        return

    current_time = row.start_utc.to_pydatetime()

    for duration, value in zip(durations, values):
        if not _is_valid_segment(duration):
            continue

        segment_start = Timestamp(current_time)
        segment_end = segment_start + Timedelta(seconds=int(duration))
        current_time = segment_end.to_pydatetime()

        _process_segment(segment_start, segment_end, value, step, freq, window_data)


def _is_valid_row_data(durations: List[int], values: List[float]) -> bool:
    """Check if row data is valid for processing."""
    return durations is not None and values is not None and len(durations) == len(values)


def _is_valid_segment(duration) -> bool:
    """Check if segment duration is valid."""
    return duration is not None and duration > 0


def _process_segment(
        segment_start: Timestamp,
        segment_end: Timestamp,
        value: float,
        step: Timedelta,
        freq: str,
        window_data: dict
) -> None:
    """Process a single segment and update overlapping windows."""
    window_start = segment_start.floor(freq)
    last_window = (segment_end - Timedelta(seconds=1)).floor(freq)

    current_window = window_start
    while current_window <= last_window:
        window_end = current_window + step
        overlap_seconds = _calculate_overlap_seconds(segment_start, segment_end, current_window, window_end)

        if overlap_seconds > 0:
            _update_window_data(current_window, value, overlap_seconds, window_data)

        current_window += step


def _calculate_overlap_seconds(
        seg_start: Timestamp,
        seg_end: Timestamp,
        win_start: Timestamp,
        win_end: Timestamp
) -> float:
    """Calculate overlap in seconds between segment and window."""
    overlap_start = max(seg_start, win_start)
    overlap_end = min(seg_end, win_end)
    overlap_timedelta = overlap_end - overlap_start
    return max(0.0, overlap_timedelta.total_seconds())


def _update_window_data(window_start: Timestamp, value: float, overlap_seconds: float, window_data: dict) -> None:
    """Update window data with weighted value and coverage."""
    window_data[window_start]['weighted_sum'] += float(value) * overlap_seconds
    window_data[window_start]['coverage'] += overlap_seconds


def _build_result_dataframe(window_data: dict, freq: str, min_coverage_s: int) -> DataFrame:
    """Build the final result DataFrame from accumulated window data."""
    rows = []
    for window_start, data in window_data.items():
        weighted_sum = data['weighted_sum']
        coverage = data['coverage']

        avg_value = weighted_sum / coverage if coverage > 0 else np.nan
        rows.append((Timestamp(window_start), avg_value, coverage))

    result_df = DataFrame(rows, columns=['window_utc', 'scan_bpm', 'scan_coverage_s'])
    result_df = result_df.sort_values('window_utc').reset_index(drop=True)

    return _apply_coverage_filter(result_df, freq, min_coverage_s)


def _apply_coverage_filter(df: DataFrame, freq: str, min_coverage_s: int) -> DataFrame:
    """Apply coverage filtering to the result DataFrame."""
    min_coverage = min(min_coverage_s, int(pd.to_timedelta(freq).total_seconds()))
    return df[df['scan_coverage_s'] >= min_coverage]


def _add_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add empty sleep_status and is_sport columns to DataFrame."""
    df = df.copy()
    df['sleep_status'] = pd.Series(dtype='Int64')
    df['is_sport'] = pd.Series(dtype=bool)
    return df


def _calculate_sleep_status(window_starts: np.ndarray, window_ends: np.ndarray,
                            df_sleep: pd.DataFrame) -> np.ndarray:
    """Calculate sleep status for each time window based on maximum overlap."""
    n_windows = len(window_starts)
    sleep_status = np.zeros(n_windows, dtype='int64')

    if df_sleep.empty:
        return sleep_status

    sleep_starts = df_sleep['start_utc'].to_numpy()
    sleep_ends = df_sleep['end_utc'].to_numpy()
    sleep_statuses = df_sleep['status'].to_numpy()

    for i in range(n_windows):
        best_status = 0
        max_overlap = pd.Timedelta(0)

        for sleep_start, sleep_end, status in zip(sleep_starts, sleep_ends, sleep_statuses):
            overlap = _calculate_time_overlap(window_starts[i], window_ends[i], sleep_start, sleep_end)

            if overlap > max_overlap:
                max_overlap = overlap
                best_status = int(status)

        sleep_status[i] = best_status

    return sleep_status


def _calculate_sport_flags(window_starts: np.ndarray, window_ends: np.ndarray,
                           df_sport: pd.DataFrame) -> np.ndarray:
    """Calculate sport activity flags for each time window."""
    n_windows = len(window_starts)
    is_sport = np.zeros(n_windows, dtype=bool)

    if df_sport.empty:
        return is_sport

    sport_starts = df_sport['start_utc'].to_numpy()
    sport_ends = df_sport['end_utc'].to_numpy()

    for i in range(n_windows):
        for sport_start, sport_end in zip(sport_starts, sport_ends):
            if _has_overlap(window_starts[i], window_ends[i], sport_start, sport_end):
                is_sport[i] = True
                break

    return is_sport


def _calculate_time_overlap(start1: np.datetime64, end1: np.datetime64,
                            start2: np.datetime64, end2: np.datetime64) -> pd.Timedelta:
    """Calculate overlap duration between two time intervals."""
    if start2 >= end1 or end2 <= start1:
        return pd.Timedelta(0)

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return overlap_end - overlap_start


def _has_overlap(start1: np.datetime64, end1: np.datetime64,
                 start2: np.datetime64, end2: np.datetime64) -> bool:
    """Check if two time intervals overlap."""
    return start2 < end1 and end2 > start1
