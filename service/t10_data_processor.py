import sqlite3
import pandas as pd

from dto.query_columns import QueryColumns
from util.pandas_utils import pandas_query_table
from util.time_utlis import to_utc_from_local_string, floor_to_bin
from typing import Optional
from contextlib import contextmanager


class T10DataProcessor:
    """Handles T10 data extraction and processing from SQLite database"""

    def __init__(self, sqlite_path: str, timezone: str, user_id: Optional[int] = None):
        self.sqlite_path = sqlite_path
        self.timezone = timezone
        self.user_id = user_id

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connection"""
        conn = sqlite3.connect(self.sqlite_path)
        try:
            yield conn
        finally:
            conn.close()

    def process_heart_rate_data(self, conn: sqlite3.Connection, hr_table: str, bin_size: str) -> pd.DataFrame:
        """Process heart rate data and return aggregated T10 data"""
        df_hr = pandas_query_table(conn, hr_table, QueryColumns.HR_COLUMNS, self.user_id)

        if df_hr.empty:
            return pd.DataFrame(columns=['window_utc', 't10_bpm', 't10_points'])

        df_hr['timestamp_utc'] = to_utc_from_local_string(df_hr['time'], self.timezone)
        df_hr['window_utc'] = floor_to_bin(df_hr['timestamp_utc'], bin_size)

        t10_min = df_hr.groupby('window_utc').agg(
            t10_bpm=('heartRate', 'mean'),
            t10_points=('heartRate', 'size')
        ).reset_index().sort_values('window_utc')

        return t10_min

    def process_sleep_data(self, conn: sqlite3.Connection, sleep_table: str) -> pd.DataFrame:
        """Process sleep data and return formatted DataFrame"""
        df_sleep = pandas_query_table(conn, sleep_table, QueryColumns.SLEEP_COLUMNS, self.user_id)

        if df_sleep.empty:
            return pd.DataFrame(columns=['start_utc', 'end_utc', 'status'])

        df_sleep['start_utc'] = to_utc_from_local_string(df_sleep['startTime'], self.timezone)
        df_sleep['end_utc'] = to_utc_from_local_string(df_sleep['endTime'], self.timezone)

        return df_sleep[['start_utc', 'end_utc', 'status']].sort_values('start_utc')

    def process_sport_data(self, conn: sqlite3.Connection, sport_table: str) -> pd.DataFrame:
        """Process sport data and return formatted DataFrame"""
        df_sport = pandas_query_table(conn, sport_table, QueryColumns.SPORT_COLUMNS, self.user_id)

        if df_sport.empty:
            return pd.DataFrame(columns=['start_utc', 'end_utc', 'sportType', 'sportId'])

        df_sport['start_utc'] = to_utc_from_local_string(df_sport['time'], self.timezone)
        df_sport['end_utc'] = df_sport['start_utc'] + pd.to_timedelta(df_sport['duration'], unit='s')

        return df_sport[['start_utc', 'end_utc', 'sportType', 'sportId']].sort_values('start_utc')
