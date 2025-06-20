from lib.func import (
    load_training_data,
    augment_by_shifting,
    augment_with_noise,
    save_training_data,
    normalise_spectra,
)
from lib.model import TrainingData
import matplotlib.pyplot as plt
import numpy as np


training_data = load_training_data("Training Data/250413_Ni_training-data.pkl")
print(f"Initial: Spectra shape={training_data.spectra.shape}, Labels shape={training_data.labels.shape}")

count_per_label = 1000
training_data = TrainingData(
    spectra=normalise_spectra(training_data.spectra),
    labels=training_data.labels,
    x_axis=training_data.x_axis
).reduce_sample_size(count_per_label=count_per_label).change_spectra_resolution(target_scale=0.3)


print(f"Reduced: Spectra shape={training_data.spectra.shape}, Labels shape={training_data.labels.shape}")


# augmented_data = augment_by_shifting(
#     training_data=training_data,
#     shift_by_list=list(range(-20, 20, 2))
# )
augmented_data = augment_with_noise(
    training_data=training_data,
    noise_sigma=0.05
)

print(f"After: Spectra shape={augmented_data.spectra.shape}, Labels shape={augmented_data.labels.shape}")

save_training_data(
    pickle_file_path=f"Training Data/250413_Ni_training-data-reduced-{count_per_label}-rescaled-augmented-noise.pkl",
    training_data=augmented_data
)
