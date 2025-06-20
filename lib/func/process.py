from exspy.signals import EELSSpectrum
from typing import Tuple, List
import numpy as np

from lib.model import TrainingData

from .load import convert_to_hspy



def change_resolution(spectra: EELSSpectrum, target_scale: float) -> EELSSpectrum:
    signal_axes = spectra.axes_manager.signal_axes[0]
    offset = signal_axes.offset
    scale = signal_axes.scale
    size = signal_axes.size
    max_x = offset + size * scale

    print(f"Initial spectra: {offset=}, {scale=}, {size=}, {max_x}")

    initial_x_axis = np.linspace(offset, max_x, size)
    target_x_axis = np.arange(offset, max_x + 1, target_scale)

    print(f"Changing from scale {scale} to scale {target_scale}")

    height, width, _ = spectra.data.shape
    new_data = np.zeros((height, width, target_x_axis.size))

    for i in range(height):
        for j in range(width):
            new_data[i, j] = np.interp(target_x_axis, initial_x_axis, spectra.data[i, j])

    return convert_to_hspy(
        spectra_3d=new_data,
        scale=target_scale,
        offset=offset
    )


def normalise_spectra(spectra_data: np.ndarray) -> np.ndarray:
    min_values = np.expand_dims(np.min(spectra_data, axis=-1), axis=-1) * np.ones(spectra_data.shape)
    data = spectra_data - min_values

    max_values = np.expand_dims(np.max(data, axis=-1), axis=-1) * np.ones(data.shape)
    return data / max_values

def flatten_data(data) -> np.ndarray:
    data_point_count = data.shape[-1]
    flattened_data = data.reshape((-1, data_point_count))
    print(f"Flattened shape: {flattened_data.shape}")
    return flattened_data


def process_test_data(
        spectra: EELSSpectrum,
        energy_index_range: Tuple[float, float],
        target_scale: float
) -> EELSSpectrum:
    # spectra = change_resolution(spectra, target_scale=target_scale)
    # print(f"Rescaled to {spectra.data.shape}")

    spectra = spectra.isig[energy_index_range[0]:energy_index_range[1]]
    print(f"Cropped the data to shape: {spectra.data.shape}")

    spectra.data[:,:,:] = normalise_spectra(spectra.data)
    print(f"Normalised the data")

    return spectra


def shift_spectra(spectra: np.ndarray, shift_by: int) -> np.ndarray:
    """
    Give spectra of (NxM), shift each spectrum by "shift_by" amount.
    Args:
        spectra: The NxM spectra, where N is the number of samples, and M is the spectrum.
        shift_by: The number of indexes to shift the spectra by.
    Return:
        New shifted spectra
    """

    new_spectra = np.zeros((spectra.shape))
    if shift_by > 0:
        spectra_end_vals = np.mean(spectra[:, -shift_by:], axis=1).reshape((-1, 1))
        new_spectra[:, shift_by:] = spectra[:, :-shift_by]
        new_spectra[:, :shift_by] = np.repeat(spectra_end_vals, abs(shift_by), axis=1)
    elif shift_by < 0:
        spectra_start_vals = np.mean(spectra[:, :shift_by], axis=1).reshape((-1, 1))
        new_spectra[:, :-abs(shift_by)] = spectra[:, abs(shift_by):]
        new_spectra[:, :abs(shift_by)] = np.repeat(spectra_start_vals, abs(shift_by), axis=1)
    else:
        return spectra.copy()

    return new_spectra

def augment_by_shifting(training_data: TrainingData, shift_by_list: List[int]) -> TrainingData:
    shifted_spectra_sets = [training_data.spectra]
    shifted_label_sets = [training_data.labels]
    for shift_by in shift_by_list:
        print(f"Shift by: {shift_by}")
        if shift_by == 0:
            continue

        shifted_spectra = shift_spectra(spectra=training_data.spectra, shift_by=shift_by)
        shifted_spectra_sets.append(shifted_spectra)
        shifted_label_sets.append(training_data.labels)

    print(f"Combining all shifted datasets...")
    new_spectra = np.concatenate(shifted_spectra_sets, axis=0)
    new_labels = np.concatenate(shifted_label_sets, axis=0)

    print(f"Augmentation by shifting is complete.")
    return TrainingData(
        spectra=new_spectra,
        labels=new_labels,
        x_axis=training_data.x_axis
    )


def augment_with_noise(training_data: TrainingData, noise_sigma: float) -> TrainingData:
    spectra_with_noise = training_data.spectra + np.random.normal(0, noise_sigma, training_data.spectra.shape)
    new_spectra = np.concatenate((training_data.spectra, spectra_with_noise), axis=0)
    new_labels = np.concatenate((training_data.labels, training_data.labels), axis=0)

    return TrainingData(
        spectra=new_spectra,
        labels=new_labels,
        x_axis=training_data.x_axis
    )
