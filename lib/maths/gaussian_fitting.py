"""
Gaussian fitting functions for EELS data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .core import get_signal_x_axis


def gauss(x, h, sigma, center):
    """Gaussian function"""
    return h * np.exp(-((x-center)**2) / (2*sigma**2))


def double_gauss(x, a0, sigma0, center0, a1, sigma1, center1):
    """Sum of two Gaussians"""
    return gauss(x, a0, sigma0, center0) + gauss(x, a1, sigma1, center1)


def triple_gauss(x, a0, sigma0, center0, a1, sigma1, center1, a2, sigma2, center2):
    """Sum of three Gaussians"""
    return gauss(x, a0, sigma0, center0) + gauss(x, a1, sigma1, center1) + gauss(x, a2, sigma2, center2)


def fit_double_gaussian(signal, signal_range, peak_range=1.5, manual_fine_tune=False):
    """
    Fit double Gaussian to a signal
    
    Parameters:
    ----------
    signal : HyperSpy signal
        Signal to fit
    signal_range : list
        Range of energies to fit [min, max]
    peak_range : float
        Range around peak center to use for fitting
    manual_fine_tune : bool
        Whether to manually fine-tune the fit
        
    Returns:
    -------
    coeffs : list
        List of double Gaussian coefficients for each spectrum
    """
    all_x = get_signal_x_axis(signal)
    
    # Build x array within signal range
    signal_range_min, signal_range_max = signal_range
    signal_range_min_index = np.argmin(np.abs(all_x - signal_range_min))
    signal_range_max_index = np.argmin(np.abs(all_x - signal_range_max))
    x_signal_range = all_x[signal_range_min_index : signal_range_max_index]

    coeffs = []
    for index, data_record in enumerate(signal.data[0]):
        y_signal_range = data_record[signal_range_min_index : signal_range_max_index]
        
        x_peak_index = np.argmax(y_signal_range)
        x_peak = x_signal_range[x_peak_index]
        x_min = x_peak - peak_range
        x_max = x_peak + peak_range
        
        # Find an array of X values start from x_min to x_max
        x_min_index = np.argmin(np.abs(x_signal_range - x_min))
        x_max_index = np.argmin(np.abs(x_signal_range - x_max))
        x = x_signal_range[x_min_index : x_max_index]
        y = y_signal_range[x_min_index : x_max_index]

        # Fit parameters
        if not manual_fine_tune:
            min_height_0 = 0
            max_height_0 = max(y)
            min_height_1 = 0
            max_height_1 = max(y)
            min_center_0, max_center_0 = x[0], x[-1]
            min_center_1, max_center_1 = x[0], x[-1]
        else:
            if index == 0:
                # Left gaussian starts big, right starts small
                min_height_0 = max(y) / 2
                max_height_0 = max(y)
                min_height_1 = 0
                max_height_1 = max(y) / 2
                min_center_0, max_center_0 = x[0], x[-1]
                min_center_1, max_center_1 = x[0], x[-1]
            else:
                prev_coeffs = coeffs[index - 1]
                prev_height_0 = prev_coeffs[0]
                prev_height_1 = prev_coeffs[3]
                prev_center_0 = prev_coeffs[2]
                prev_center_1 = prev_coeffs[5]
    
                min_height = 0
                max_height = 1
                max_height_change = 0.1
                max_height_change_opposite = 0.01
                min_height_0 = max(prev_height_0 - max_height_change, min_height)
                max_height_0 = min(prev_height_0 + max_height_change_opposite, max_height)
                min_height_1 = max(prev_height_1 - max_height_change_opposite, min_height)
                max_height_1 = min(prev_height_1 + max_height_change, max_height)
    
                max_center_change = 0.5
                min_center_0 = max(prev_center_0 - max_center_change, x[0])
                max_center_0 = min(prev_center_0 + max_center_change, x[-1])
                min_center_1 = max(prev_center_1 - max_center_change, x[0])
                max_center_1 = min(prev_center_1 + max_center_change, x[-1])
           
        min_bound = [min_height_0, 0, min_center_0, min_height_1, 0, min_center_1] 
        max_bound = [max_height_0, 10, max_center_0, max_height_1, 10, max_center_1]

        initial_height_0 = np.mean([min_height_0, max_height_0])
        initial_height_1 = np.mean([min_height_1, max_height_1])
        initial_center_0 = np.mean([min_center_0, max_center_0])
        initial_center_1 = np.mean([min_center_1, max_center_1])
        initial_conditions = [initial_height_0, 1., initial_center_0, initial_height_1, 1, initial_center_1]
        max_iterations = 100000
        
        try:
            coeff, _ = curve_fit(double_gauss, x, y, p0=initial_conditions, maxfev=max_iterations, bounds=(min_bound, max_bound))
            
            # Sort by gaussian centers
            c0, c1 = coeff[2], coeff[5]
            if c0 > c1:
                coeff = [*coeff[3:], *coeff[:3]]
        except Exception as e:
            print(f"Fitting failed for index {index}: {str(e)}")
            # Use previous coefficients if available, otherwise use initial guess
            if index > 0 and coeffs:
                coeff = coeffs[index - 1]
            else:
                coeff = initial_conditions
                
        coeffs.append(coeff)
    
    return coeffs


def fit_triple_gaussian(signal, signal_range, peak_range=3.0):
    """
    Fit triple Gaussian to a signal, specifically designed for oxygen main peak
    
    Parameters:
    ----------
    signal : HyperSpy signal
        Signal to fit
    signal_range : list
        Range of energies to fit [min, max]
    peak_range : float
        Range around peak center to use for fitting
        
    Returns:
    -------
    coeffs : list
        List of triple Gaussian coefficients for each spectrum
    """
    all_x = get_signal_x_axis(signal)
    
    # Build x array within signal range
    signal_range_min, signal_range_max = signal_range
    signal_range_min_index = np.argmin(np.abs(all_x - signal_range_min))
    signal_range_max_index = np.argmin(np.abs(all_x - signal_range_max))
    x_signal_range = all_x[signal_range_min_index : signal_range_max_index]

    coeffs = []
    for index, data_record in enumerate(signal.data[0]):
        y_signal_range = data_record[signal_range_min_index : signal_range_max_index]
        
        x_peak_index = np.argmax(y_signal_range)
        x_peak = x_signal_range[x_peak_index]
        x_min = x_peak - peak_range
        x_max = x_peak + peak_range
        
        # Find an array of X values start from x_min to x_max
        x_min_index = np.argmin(np.abs(x_signal_range - x_min))
        x_max_index = np.argmin(np.abs(x_signal_range - x_max))
        x = x_signal_range[x_min_index : x_max_index]
        y = y_signal_range[x_min_index : x_max_index]

        # Set parameters for fitting
        max_height = max(y) * 1.2
        min_center, max_center = x[0], x[-1]
        min_sigma = 0.1
        max_sigma = 3.0

        # Parameters for triple Gaussian
        min_bound = [0, min_sigma, min_center, 0, min_sigma, min_center, 0, min_sigma, min_center] 
        max_bound = [max_height, max_sigma, max_center, max_height, max_sigma, max_center, max_height, max_sigma, max_center]

        # Initial guesses - distribute centers across the peak range
        peak_width = max_center - min_center
        initial_height = max_height * 0.4
        initial_center_0 = min_center + peak_width * 0.2
        initial_center_1 = min_center + peak_width * 0.5  # Middle
        initial_center_2 = min_center + peak_width * 0.8
        initial_sigma = peak_width / 6  # Reasonable starting width
        
        initial_conditions = [
            initial_height, initial_sigma, initial_center_0,
            initial_height, initial_sigma, initial_center_1, 
            initial_height, initial_sigma, initial_center_2
        ]
        max_iterations = 100000
        
        try:
            coeff, _ = curve_fit(triple_gauss, x, y, p0=initial_conditions, maxfev=max_iterations, 
                                bounds=(min_bound, max_bound))
            
            # Sort by gaussian centers
            centers = [coeff[2], coeff[5], coeff[8]]
            idx_sorted = np.argsort(centers)
            
            # Rearrange coefficients based on sorted centers
            sorted_coeff = np.zeros_like(coeff)
            for i, idx in enumerate(idx_sorted):
                sorted_coeff[i*3:(i+1)*3] = coeff[idx*3:(idx+1)*3]
            
            coeff = sorted_coeff
                
        except Exception as e:
            print(f"Triple Gaussian fitting failed for index {index}: {str(e)}")
            # Use previous coefficients if available, otherwise use initial guess
            if index > 0 and coeffs:
                coeff = coeffs[index - 1]
            else:
                coeff = initial_conditions
                
        coeffs.append(coeff)
    
    return coeffs


def do_fitting_double(element, index, signal_range, peak_range=1.5):
    """
    Function fits the data at a specific index, plots the raw data and the fitted Gaussians,
    and returns the fitted parameters and y-values for further analysis.
    
    Parameters:
    ----------
    element : HyperSpy signal
        Element signal to fit
    index : int
        Index of spectrum to process and plot
    signal_range : list
        Range of energies to fit [min, max]
    peak_range : float
        Range around peak center to use for fitting
        
    Returns:
    -------
    g1_coeffs, g2_coeffs, g12_coeffs, g1_y, g2_y, g12_y
    """
    g12_coeffs = fit_double_gaussian(element, signal_range=signal_range, peak_range=peak_range)
    g1_coeffs = [coeff[:3] for coeff in g12_coeffs]
    g2_coeffs = [coeff[3:] for coeff in g12_coeffs]
    
    g12_coeff = g12_coeffs[index]
    g1_coeff = g12_coeff[:3]
    g2_coeff = g12_coeff[3:]

    x = get_signal_x_axis(element)
    g1_y = gauss(x, *g1_coeff)
    g2_y = gauss(x, *g2_coeff)
    g12_y = double_gauss(x, *g12_coeff)

    plt.plot(x, element.data[0][index], 'rx', label="Raw data")
    plt.plot(x, g1_y, label="K1a")
    plt.plot(x, g2_y, label="K1b")
    plt.plot(x, g12_y, label="K1a + K1b")
    plt.legend()
    plt.xlabel("eV")
    plt.ylabel("Intensity")
    
    return g1_coeffs, g2_coeffs, g12_coeffs, g1_y, g2_y, g12_y


def do_fitting_triple(element, index, signal_range):
    """
    Fit triple Gaussian for oxygen main peak
    
    Parameters:
    ----------
    element : HyperSpy signal
        Element signal to fit
    index : int
        Index of spectrum to process and plot
    signal_range : list
        Range of energies to fit [min, max]
        
    Returns:
    -------
    g123_coeffs, g1_y, g2_y, g3_y, g123_y
    """
    g123_coeffs = fit_triple_gaussian(element, signal_range=signal_range)
    g123_coeff = g123_coeffs[index]
    g1_coeff = g123_coeff[:3]
    g2_coeff = g123_coeff[3:6]
    g3_coeff = g123_coeff[6:]

    x = get_signal_x_axis(element)
    g1_y = gauss(x, *g1_coeff)
    g2_y = gauss(x, *g2_coeff)
    g3_y = gauss(x, *g3_coeff)
    g123_y = triple_gauss(x, *g123_coeff)
    
    plt.plot(x, element.data[0][index], 'rx', label="Raw data")
    plt.plot(x, g1_y, label="K1a")
    plt.plot(x, g2_y, label="K1b")
    plt.plot(x, g3_y, label="K1c")
    plt.plot(x, g123_y, label="K1a + K1b + K1c")
    plt.legend()
    plt.xlabel("eV")
    plt.ylabel("Intensity")
    
    return g123_coeffs, g1_y, g2_y, g3_y, g123_y


def do_fitting_double_updated(element, index, signal_range, peak_range=1.5, manual_fine_tune=False, show_plots=True):
    """
    Fit double Gaussian with updated labels (g1 and g2 instead of K1a and K1b)
    
    Parameters:
    ----------
    element : HyperSpy signal
        Element signal to fit
    index : int
        Index of spectrum to process and plot
    signal_range : list
        Range of energies to fit [min, max]
    peak_range : float
        Range around peak center to use for fitting
    manual_fine_tune : bool
        Whether to manually fine-tune the fit
    show_plots : bool
        Whether to show diagnostic plots
        
    Returns:
    -------
    g1_coeffs, g2_coeffs, g12_coeffs, g1_y, g2_y, g12_y
    """
    g12_coeffs = fit_double_gaussian(element, signal_range=signal_range, peak_range=peak_range, 
                                    manual_fine_tune=manual_fine_tune)
    g1_coeffs = [coeff[:3] for coeff in g12_coeffs]
    g2_coeffs = [coeff[3:] for coeff in g12_coeffs]
    
    g12_coeff = g12_coeffs[index]
    g1_coeff = g12_coeff[:3]
    g2_coeff = g12_coeff[3:]

    x = get_signal_x_axis(element)
    g1_y = gauss(x, *g1_coeff)
    g2_y = gauss(x, *g2_coeff)
    g12_y = double_gauss(x, *g12_coeff)

    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.plot(x, element.data[0][index], 'rx', label="Raw data")
        plt.plot(x, g1_y, label="g1")
        plt.plot(x, g2_y, label="g2")
        plt.plot(x, g12_y, label="g1 + g2")
        plt.legend()
        plt.xlabel("eV")
        plt.ylabel("Intensity")
        plt.tight_layout()
        plt.show()
    
    return g1_coeffs, g2_coeffs, g12_coeffs, g1_y, g2_y, g12_y


def do_fitting_triple_updated(element, index, signal_range, peak_range=3.0, manual_fine_tune=False, show_plots=True):
    """
    Fit triple Gaussian for oxygen main peak
    
    Parameters:
    ----------
    element : HyperSpy signal
        Element signal to fit
    index : int
        Index of spectrum to process and plot
    signal_range : list
        Range of energies to fit [min, max]
    peak_range : float
        Range around peak center to use for fitting
    manual_fine_tune : bool
        Whether to manually fine-tune the fit
    show_plots : bool
        Whether to show diagnostic plots
        
    Returns:
    -------
    g1_coeffs, g2_coeffs, g3_coeffs, g123_coeffs, g123_y
    """
    g123_coeffs = fit_triple_gaussian(element, signal_range=signal_range, peak_range=peak_range)
    
    # Extract individual Gaussian coefficients
    g1_coeffs = [coeff[0:3] for coeff in g123_coeffs]
    g2_coeffs = [coeff[3:6] for coeff in g123_coeffs]
    g3_coeffs = [coeff[6:9] for coeff in g123_coeffs]
    
    # Get coefficients for the requested index
    g123_coeff = g123_coeffs[index]
    g1_coeff = g123_coeff[0:3]
    g2_coeff = g123_coeff[3:6]
    g3_coeff = g123_coeff[6:9]

    # Get x-axis and calculate fitted curves
    x = get_signal_x_axis(element)
    g1_y = gauss(x, *g1_coeff)
    g2_y = gauss(x, *g2_coeff)
    g3_y = gauss(x, *g3_coeff)
    g123_y = triple_gauss(x, *g123_coeff)

    # Show diagnostic plot if requested
    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.plot(x, element.data[0][index], 'rx', label="Raw data")
        plt.plot(x, g1_y, label="g1", color='#1f77b4')  # Blue
        plt.plot(x, g2_y, label="g2", color='#ff7f0e')  # Orange
        plt.plot(x, g3_y, label="g3", color='#2ca02c')  # Green
        plt.plot(x, g123_y, label="g1 + g2 + g3", color='red')
        plt.legend()
        plt.xlabel("eV")
        plt.ylabel("Intensity")
        plt.title(f"Triple Gaussian Fit - O Main Peak")
        plt.tight_layout()
        plt.show()
    
    return g1_coeffs, g2_coeffs, g3_coeffs, g123_coeffs, g123_y
