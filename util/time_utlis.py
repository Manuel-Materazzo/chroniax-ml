import pandas as pd
from datetime import timezone
from dateutil import parser as dtparser
from pandas import Timestamp, Series


def parse_iso_to_utc(ts: str) -> Timestamp:
    """Parse an ISO 8601 formatted datetime string to a UTC timezone-aware Timestamp.

    Args:
        ts: An ISO 8601 formatted datetime string with timezone information.

    Returns:
        A pandas Timestamp object localized to UTC timezone.

    Raises:
        ValueError: If the input string is not a valid ISO 8601 format or lacks timezone information.
    """
    dt = dtparser.isoparse(ts)
    if dt.tzinfo is None:
        raise ValueError("Timestamp must include timezone offset")
    return pd.Timestamp(dt.astimezone(timezone.utc))


def to_utc_from_local_string(s: Series, start_timezone: str) -> Series:
    """Convert a Series of timezone-naive datetime strings to UTC timestamps.

    Args:
        s: A pandas Series containing datetime strings without timezone information.
        start_timezone: The timezone to localize the naive datetimes before conversion to UTC.

    Returns:
        A pandas Series of UTC timezone-aware Timestamps.

    Raises:
        TypeError: If the input Series contains non-string values.
        ValueError: If the timezone string is invalid or the datetime strings are malformed.
    """
    ts = pd.to_datetime(s)
    return ts.dt.tz_localize(start_timezone).dt.tz_convert('UTC')


def floor_to_bin(ts: Series, freq: str) -> Series:
    """Floor a Series of Timestamps to the specified frequency.

    Args:
        ts: A pandas Series of Timestamps to be floored.
        freq: A string representing the frequency to floor to (e.g., 'D', 'H', 'T').

    Returns:
        A pandas Series of Timestamps floored to the specified frequency.

    Raises:
        ValueError: If the frequency string is invalid or unsupported.
    """
    return ts.dt.floor(freq)
