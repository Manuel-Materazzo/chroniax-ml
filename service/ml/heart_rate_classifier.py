from dto.enums.hr_zones import HeartRateZone

class HeartRateZoneClassifier:
    """Classifies heart rate values into zones."""

    @staticmethod
    def classify(heart_rate: float) -> str:
        """Classify heart rate into predefined zones."""
        if heart_rate < 60:
            return HeartRateZone.VERY_LOW.value
        elif heart_rate < 90:
            return HeartRateZone.LOW.value
        elif heart_rate < 120:
            return HeartRateZone.MODERATE.value
        else:
            return HeartRateZone.HIGH.value