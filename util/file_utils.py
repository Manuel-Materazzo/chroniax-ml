import ast
import sqlite3
from typing import Optional, Tuple

import pandas as pd
from pandas import DataFrame

from service.t10_data_processor import T10DataProcessor
from util.time_utlis import parse_iso_to_utc, to_utc_from_local_string, floor_to_bin


def get_scanwatch_data(path: str) -> DataFrame:
    """Load a ScanWatch CSV file into a DataFrame, parsing list-like strings into Python lists.

    Args:
        path: Path to the CSV file to load.

    Returns:
        A DataFrame containing the parsed data with columns:
        - start_utc: UTC datetime objects parsed from ISO format strings
        - durations: Lists of durations parsed from string representations
        - values: Lists of values parsed from string representations
    """
    df = pd.read_csv(path)

    def parse_list_cell(cell):
        s = str(cell).strip()
        if s == "" or s == "[]":
            return []
        return ast.literal_eval(s)

    df['durations'] = df['duration'].apply(parse_list_cell)
    df['values'] = df['value'].apply(parse_list_cell)
    df['start_utc'] = df['start'].apply(parse_iso_to_utc)
    return df[['start_utc', 'durations', 'values']]


def get_t10_data(
        sqlite_path: str,
        timezone: str,
        hr_table: str = 'HeartRateItemEntity',
        sleep_table: str = 'SleepItemEntity',
        sport_table: str = 'SportRecordEntity',
        user_id: Optional[int] = None,
        bin_size: str = "1min"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract and process T10 data from SQLite database.

    Args:
        sqlite_path: Path to SQLite database
        timezone: Timezone string for time conversion
        hr_table: Heart rate table name
        sleep_table: Sleep data table name
        sport_table: Sport data table name
        user_id: Optional user ID filter
        bin_size: Frequency for heart rate aggregation

    Returns:
        Tuple of (heart_rate_data, sleep_data, sport_data) DataFrames

    Raises:
        sqlite3.Error: If database connection or query fails
        Exception: If data processing fails
    """
    processor = T10DataProcessor(sqlite_path, timezone, user_id)

    try:
        with processor.get_db_connection() as conn:
            t10_min = processor.process_heart_rate_data(conn, hr_table, bin_size)
            df_sleep = processor.process_sleep_data(conn, sleep_table)
            df_sport = processor.process_sport_data(conn, sport_table)

        return t10_min, df_sleep, df_sport

    except sqlite3.Error as e:
        raise sqlite3.Error(f"Database error while processing T10 data: {e}")
    except Exception as e:
        raise Exception(f"Error processing T10 data: {e}")
