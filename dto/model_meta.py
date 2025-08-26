from dataclasses import dataclass
from typing import List


@dataclass
class ModelMeta:
    kind: str  # 'isotonic' or 'pchip' or 'piecewise'
    context: str  # 'rest' or 'active' or 'global'
    x_knots: List[float]
    y_knots: List[float]
    clip_lo: float
    clip_hi: float
