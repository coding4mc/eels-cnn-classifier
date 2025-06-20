from dataclasses import dataclass
from typing import List

from .plot_heatmap import PlotHeatmap
from .plot_3d import Plot3D


@dataclass(frozen=True)
class PlotViewerPageData:
    plot_heatmaps: List[PlotHeatmap]
    plot_3ds: List[Plot3D]
    title: str = ""

    def __post_init__(self):
        for heatmap in self.plot_heatmaps:
            assert heatmap.height == self.height
            assert heatmap.width == self.width
            
        for plot_3d in self.plot_3ds:
            assert plot_3d.height == self.height
            assert plot_3d.width == self.width

    @property
    def width(self) -> int:
        return self.plot_heatmaps[0].width
    
    @property
    def height(self) -> int:
        return self.plot_heatmaps[0].height