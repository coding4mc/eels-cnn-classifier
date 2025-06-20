from typing import Tuple
from dataclasses import dataclass
from exspy.signals.eels import EELSSpectrum
from hyperspy.axes import AxesManager
import numpy as np
import pickle


from lib.model import TrainingData, TestData


def load_training_data(pickle_file_path: str) -> TrainingData:
    """
    Loads training data
    """
    with open(pickle_file_path, "rb") as file:
        data = pickle.load(file)
    
    return TrainingData(
        spectra=data['spectra'],
        labels=data['labels'],
        x_axis=data['x_axis']
    )


def save_training_data(pickle_file_path: str, training_data: TrainingData):
    with open(pickle_file_path, "wb") as file:
        pickle.dump({
            "spectra": training_data.spectra,
            "labels": training_data.labels,
            "x_axis": training_data.x_axis
        }, file)


def load_test_data(pickle_file_path: str) -> TestData:
    with open(pickle_file_path, "rb") as file:
        pickle_data = pickle.load(file)

    return TestData(
        spectra=pickle_data['spectra'],
        height=pickle_data['height'],
        width=pickle_data['width']
    )


def save_test_data(pickle_file_path: str, spectra: np.ndarray, height: int, width: int):
    with open(pickle_file_path, "wb") as file:
        pickle.dump({
            "spectra": spectra,
            "height": height,
            "width": width
        }, file)


def convert_to_hspy(
        spectra_3d: np.ndarray,
        scale: float,
        offset: float
): 
    # Create Signal2D object and set it as EELS
    s = EELSSpectrum(spectra_3d)
    
    # Set energy axis
    s.axes_manager[-1].name = 'Energy loss'
    s.axes_manager[-1].units = 'eV'
    s.axes_manager[-1].scale = scale
    s.axes_manager[-1].offset = offset
    
    # Set spatial axes
    s.axes_manager[0].name = 'y'
    s.axes_manager[1].name = 'x'
    s.axes_manager[0].units = 'nm'
    s.axes_manager[1].units = 'nm'
    return s


def save_as_hspy(
        hspy_file_path: str,
        spectra_3d: np.ndarray,
        scale: float,
        offset: float
):
    s = convert_to_hspy(
        spectra_3d=spectra_3d,
        scale=scale,
        offset=offset
    )
    s.save(hspy_file_path, overwrite=True)