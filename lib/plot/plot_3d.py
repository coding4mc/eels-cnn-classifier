from dataclasses import dataclass
from typing import List
import numpy as np

from .curve_1d import Curve1D
from .plot_1d import Plot1D


@dataclass(frozen=True)
class Plot3D:
    plots: List[List[Plot1D]]

    @classmethod
    def from_3d_matrix(
        cls,
        matrix: np.ndarray,
        x_axis: np.ndarray,
        title: str
    ) -> "Plot3D":
        assert len(matrix.shape) == 3
        assert len(x_axis.shape) == 1

        return Plot3D(
            plots=[
                [
                    Plot1D(
                        curves=[Curve1D(x=x_axis, y=data)],
                        title=title
                    )
                    for data in row
                ]
                for row in matrix
            ]
        )


    @property
    def height(self) -> int:
        return len(self.plots)
    
    @property
    def width(self) -> int:
        return len(self.plots[0])

    def __post_init__(self):
        assert len(self.plots) > 0
        
        # Verify all rows have same number of columns
        column_count = len(self.plots[0])
        for row in self.plots:
            assert len(row) == column_count

