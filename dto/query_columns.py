from dataclasses import dataclass


@dataclass
class QueryColumns:
    """Column configurations for different data types"""
    HR_COLUMNS = ["time", "heartRate"]
    SLEEP_COLUMNS = ["startTime", "endTime", "status"]
    SPORT_COLUMNS = ["sportId", "time", "duration", "sportType"]
