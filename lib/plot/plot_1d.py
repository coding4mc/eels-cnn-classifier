from dataclasses import dataclass
from typing import List

from .curve_1d import Curve1D


@dataclass(frozen=True)
class Plot1D:
    curves: List[Curve1D]
    title: str

    def __post_init__(self):
        assert len(self.curves) > 0