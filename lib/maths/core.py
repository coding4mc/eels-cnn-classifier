"""
Core utility functions for EELS data analysis.
"""

from exspy.signals import EELSSpectrum


def get_signal_x_axis(signal: EELSSpectrum):
    """
    Get the x-axis array from a signal.
    
    Parameters:
    ----------
    signal : HyperSpy signal
        Signal to get x-axis from
        
    Returns:
    -------
    numpy.ndarray
        X-axis array
    """
    return signal.axes_manager.signal_axes[0].axis

