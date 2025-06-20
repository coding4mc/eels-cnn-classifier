from typing import Optional
from dataclasses import dataclass
import numpy as np


from .rgb_colour import RgbColour


@dataclass(frozen=True)
class Curve1D:
    x: np.ndarray
    y: np.ndarray
    label: Optional[str] = None
    std: Optional[np.ndarray] = None
    color: Optional[RgbColour] = None
    
    def __post_init__(self):
        assert len(self.x.shape) == 1
        assert len(self.y.shape) == 1
        assert self.x.size == self.y.size

        if self.std is not None:
            assert len(self.std.shape) == 1, self.std.shape
            assert self.std.size == self.y.size