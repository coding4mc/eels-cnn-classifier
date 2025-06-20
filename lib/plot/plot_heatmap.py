from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class PlotHeatmap:
    data: np.ndarray
    title: Optional[str] = None
    colourbar_title: Optional[str] = None
    colourbar_min_override: Optional[float] = None
    colourbar_max_override: Optional[float] = None

    def __post_init__(self):
        assert len(self.data.shape) == 2
        assert self.colorbar_min < self.colorbar_max, f"Max: {self.colorbar_max}, Min: {self.colorbar_min}"

    @property
    def height(self) -> int:
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        return self.data.shape[1]
    
    @property
    def max(self) -> float:
        return np.max(self.data)
    
    @property
    def min(self) -> float:
        return np.min(self.data)
    
    @property
    def colorbar_min(self) -> float:
        if self.colourbar_min_override:
            return self.colourbar_min_override
        if self.max == self.min:
            return self.min * 0.9

        return self.min - (self.max - self.min) * 0.1
    
    @property
    def colorbar_max(self) -> float:
        if self.colourbar_max_override:
            return self.colourbar_max_override
        if self.max == self.min:
            return self.max * 1.1

        return self.max + (self.max - self.min) * 0.1
