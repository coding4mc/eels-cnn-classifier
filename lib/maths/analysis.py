"""
Analysis and metrics functions for EELS data analysis.
"""

import numpy as np
from scipy.integrate import simpson as simps
from .gaussian_fitting import double_gauss, fit_double_gaussian


def calculate_double_gauss_areas(g12_coeffs, energy_range, num_points=10000):
    """
    Calculate the area under fitted Gaussian peaks for L3 or L2 edges
    
    Parameters:
    ----------
    g12_coeffs : list
        List of double Gaussian coefficients for each spectrum
    energy_range : list
        Range of energies to use for integration [min, max]
    num_points : int
        Number of points to use for integration
        
    Returns:
    -------
    areas : numpy array
        Array of integrated areas for each spectrum
    centers : numpy array
        Array of peak centers for each spectrum
    """
    # Define energy range for integration
    x_coeff = np.linspace(energy_range[0], energy_range[1], num_points)
    
    # Calculate areas under the curve using Simpson's rule
    areas = []
    for coeff in g12_coeffs:
        y_values = double_gauss(x_coeff, *coeff)
        area = simps(y_values, x_coeff)
        areas.append(area)
    
    # Convert to numpy array
    areas = np.array(areas)
    
    return areas


def calculate_l_peak_area_ratio_row(
        spectrum_row,
        l2_signal_range,
        l2_peak_range,
        l3_signal_range,
        l3_peak_range,
        num_points=10000
):
    """
    Calculate the ratio of peak areas for a row of spectra
    
    Parameters:
    ----------
    spectrum_row : HyperSpy signal
        Row of spectra
    signal_range : list
        Range of energies to fit [min, max]
    peak_range : float
        Range around peak center to use for fitting
    num_points : int
        Number of points to use for integration
        
    Returns:
    -------
    numpy.ndarray
        Array of peak area ratios
    """
    
    g12_coeffs_l2 = fit_double_gaussian(spectrum_row, signal_range=l2_signal_range, peak_range=l2_peak_range)
    g12_coeffs_l3 = fit_double_gaussian(spectrum_row, signal_range=l3_signal_range, peak_range=l3_peak_range)

    l2_area = calculate_double_gauss_areas(g12_coeffs_l2, l2_signal_range, num_points)
    l3_area = calculate_double_gauss_areas(g12_coeffs_l3, l3_signal_range, num_points)

    return l3_area / l2_area


def calculate_l_peak_area_ratios(
        spectrum,
        l2_signal_range,
        l2_peak_range,
        l3_signal_range,
        l3_peak_range,
        num_points=10000
):
    # Get the shape of the first two dimensions (spatial dimensions)
    height, width = spectrum.data.shape[0], spectrum.data.shape[1]
    print(f"Processing data with dimensions: {height} x {width}")

    # Create an empty array to store the ratios
    ratio_matrix = np.zeros((height, width))

    # Process each pixel
    for i in range(height):
        # Extract the spectrum at this row
        row_spectrum = spectrum.inav[:, i:i+1]
        
        # Calculate the ratio
        ratio_row = calculate_l_peak_area_ratio_row(
            row_spectrum,
            l2_signal_range,
            l2_peak_range,
            l3_signal_range,
            l3_peak_range,
            num_points
        )
        
        # Store in the ratio array
        ratio_matrix[i] = ratio_row

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{height} rows.")

    return ratio_matrix

