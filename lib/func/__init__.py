from .load import (
    load_training_data, load_test_data,
    save_test_data, save_training_data,
    save_as_hspy, convert_to_hspy
)
from .process import (
    process_test_data, shift_spectra,
    change_resolution, flatten_data,
    normalise_spectra, augment_by_shifting,
    augment_with_noise
)
