from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class PredictionOutput:
    predictions: np.ndarray
    prediction_probabilities: np.ndarray
    probabilities_per_label: Dict[int, np.ndarray]
    