from typing import Optional

import pandas as pd
from pandas import DataFrame


def pandas_query_table(conn, table: str, columns: list, user_id: Optional[int] = None) -> DataFrame:
    """Query a database table and return the results as a pandas DataFrame.

    Args:
        conn: Database connection object
        table: Name of the table to query
        columns: List of column names to select
        user_id: Optional user ID to filter results by

    Returns:
        DataFrame containing the query results
    """
    q = f"SELECT {', '.join(columns)} FROM {table}"
    params = []
    if user_id is not None:
        q += " WHERE userId = ?"
        params.append(user_id)
    return pd.read_sql_query(q, conn, params=params)
