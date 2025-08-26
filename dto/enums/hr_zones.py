from enum import Enum


class HeartRateZone(Enum):
    VERY_LOW = '<60'
    LOW = '60-90'
    MODERATE = '90-120'
    HIGH = '>120'
