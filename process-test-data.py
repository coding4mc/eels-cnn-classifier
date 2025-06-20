import hyperspy.api as hs
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from exspy.signals.eels import EELSSpectrum

from lib.func import (
    process_test_data, change_resolution,
    load_training_data, save_test_data, save_as_hspy,
    flatten_data
)
from lib.plot import PlotViewer1D
from PyQt5.QtWidgets import QApplication
import os

training_data = load_training_data("Training Data/250409_ni_2_filtered.pkl")

test_file_name = "250410_Pristine_A_ni.hspy"
test_base_name, _ = os.path.splitext(test_file_name)

test_data_hspy: EELSSpectrum = hs.load(f"Test Data/{test_file_name}")
x, y = test_data_hspy.axes_manager.navigation_axes
height, width = y.size, x.size

print(f"Initial shape:", test_data_hspy.data.shape)
processed_spectra = process_test_data(test_data_hspy, energy_index_range=(0,170), target_scale=training_data.scale)
processed_spectra.save(f"Processed Test Data/{test_base_name}-scale-0.3.hspy", overwrite=True)

processed_data = flatten_data(processed_spectra.data)

save_test_data(
    pickle_file_path=f"Processed Test Data/{test_base_name}-scale-0.3.pkl",
    spectra=processed_data,
    height=height,
    width=width
)

# fib_1a_data_processed_reshaped = fib_1a_data_processed.reshape((height, width, -1))
# save_as_hspy(
#     hspy_file_path="Processed Test Data/fib1a-2025-04-13.hspy",
#     spectra_3d=fib_1a_data_processed_reshaped,
#     scale=training_data.scale,
#     offset=fib_1a.axes_manager[-1].offset,
# )

# save_test_data(
#     pickle_file_path="Processed Test Data/fib1a-2025-04-13.pkl",
#     spectra=fib_1a_data_processed,
#     height=height,
#     width=width
# )

# plt.plot(training_data.x_axis, training_data.get_spectra(label=2), 'b-')
# plt.plot(training_data.x_axis, fib_1a_data_processed[5], 'r-')
# plt.title("Compare training 2+ to test 2+")
# plt.show()

