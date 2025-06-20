"""
EELS (Electron Energy Loss Spectroscopy) data analysis library.

This package contains functions for processing, analyzing, and visualizing EELS data.
"""

# Core functions
from .core import (
    get_signal_x_axis
)

# Gaussian fitting functions
from .gaussian_fitting import (
    gauss,
    double_gauss,
    triple_gauss,
    fit_double_gaussian,
    fit_triple_gaussian,
    do_fitting_double,
    do_fitting_triple,
    do_fitting_double_updated,
    do_fitting_triple_updated,
)

from .analysis import (
    fit_double_gaussian,
    calculate_l_peak_area_ratio_row,
    calculate_l_peak_area_ratios,
    calculate_double_gauss_areas
)

