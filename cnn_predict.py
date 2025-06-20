from lib.ml import CNN, CNN2, CNN3, CNN4, CNN5, CNN6, ModelExecutor
from lib.func import load_training_data, load_test_data
from typing import Type
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np

from lib.plot import (
    PlotHeatmap, Plot3D, PlotViewer3D, Plot1D, Curve1D, RgbColour,
    PlotViewerPageData
)


device = torch.device('mps')

training_data_folder_path = "Training Data"
test_data_folder_path = "Processed Test Data"
l3_l2_ratio_folder_path = "L3-L2 Ratios"
model_folder_path = "Trained CNN Model"

# training_data_file_name = "250409_ni_2_filtered.pkl"
# training_data_file_name = "250413_Ni_training-data.pkl"
# training_data_file_name = "250413_Ni_training-data-augmented.pkl"
# training_data_file_name = "250415_reference_data_23_augmented_24k_cropped_eV_px03.pkl"
# training_data_file_name = "250413_Ni_training-data-reduced-1000-rescaled-augmented-20x-noise.pkl"
training_data_file_name = "250413_Ni_training-data-reduced-1000-rescaled-augmented-noise.pkl"

# test_data_file_name = "fib1a-2025-04-13.pkl"
# test_data_file_name = "fib1b-2025-04-14.pkl"
# test_data_file_name = "250410_VCP_E_ni.pkl"
# test_data_file_name = "fib1b-l3.pkl"
# test_data_file_name = "fib_1a_l3_only.pkl"
# test_data_file_name = "fib_1b_l3_only.pkl"
# test_data_file_name = "vcp_b_l3_only.pkl"
# test_data_file_name = "vcp_e_l3_only.pkl"
# test_data_file_name = "vcp_g_l3_only.pkl"
# test_data_file_name = "pristine_a_l3_only.pkl"
# test_data_file_name = "pristine_c_l3_only.pkl"
# test_data_file_name = "fib1a-2025-04-13-scale-0.3.pkl"
# test_data_file_name = "fib1b-2025-04-14-scale-0.3.pkl"
# test_data_file_name = "vcp-g-cropped-deconvolve-scale-0.3.pkl"

# test_data_file_names = [
#     "fib_1a_l3_only.pkl",
#     "fib_1b_l3_only.pkl",
#     "vcp_b_l3_only.pkl",
#     "vcp_e_l3_only.pkl",
#     "vcp_g_l3_only.pkl",
#     "pristine_a_l3_only.pkl",
#     "pristine_c_l3_only.pkl",
# ]

test_data_file_names = [
    "fib1a-2025-04-13-scale-0.3.pkl",
    "fib1b-2025-04-14-scale-0.3.pkl",
    "vcp-g-cropped-deconvolve-scale-0.3.pkl",
    "250410_Pristine_A_ni-scale-0.3.pkl"
]


# l3_l2_ratio_file_name = "fib1a-2025-04-13.npy"
# l3_l2_ratio_file_name = "fib1b-2025-04-14.npy"
# l3_l2_ratio_file_name = "250410_VCP_E_ni.npy"

# model_file_name = "250413_Ni_training-data-0.5dropout-3fc.torch"
# model_file_name = "250413_Ni_training-data-augmented-cnn2-0.5dropout.torch"
# model_file_name = "250413_Ni_training-data-augmented-cnn3-0.5dropout-3epoch.torch"
# model_file_name = "250413_Ni_training-data-augmented-10x-noise-cnn3-0.5dropout-0.5stndo-5epoch.torch"
# model_file_name = "250413_Ni_training-data-augmented-10x-noise-cnn3-0.5dropout-0.5stndo.torch"
# model_file_name = "250413_Ni_training-data-augmented-10x-noise-cnn3-0.5dropout-5epoch.torch"
# model_file_name = "250413_Ni_training-data-augmented-x50-cnn3-0.5dropout-0.5stndo-1epoch.torch"
# model_file_name = "250413_Ni_training-data-augmented-x50-cnn3-0.5dropout-0.5stndo.torch"
# model_file_name = "250413_Ni_training-data-augmented-x50-cnn3-0.5dropout-0stndo.torch"
# model_file_name = "250413_Ni_training-data-augmented-x50-cnn4.torch"
# model_file_name = "250413_Ni_training-data-augmented-x50-cnn5.torch"
# model_file_name = f"250413_Ni_training-data-reduced-1000-augmented-50x-noise-{Model.__name__.lower()}.torch"
# model_file_name_prefix = "250415_reference_data_23_augmented_24k_cropped_eV_px03"
model_file_name_prefix = "250413_Ni_training-data-reduced-1000-rescaled-augmented-20x-noise"
min_epoch_count = 1
max_epoch_count = 10

training_data = load_training_data(f"{training_data_folder_path}/{training_data_file_name}")
test_data_by_file_name = {
    test_data_file_name: load_test_data(f"{test_data_folder_path}/{test_data_file_name}")
    for test_data_file_name in test_data_file_names
}

# l3_l2_ratio_matrix = np.load(f"{l3_l2_ratio_folder_path}/{l3_l2_ratio_file_name}")

def build_model_file_name(Model: Type[nn.Module], epoch: int) -> str:
    return f"{model_folder_path}/{model_file_name_prefix}-{Model.__name__.lower()}-{epoch}epoch.torch"

# ALL_MODELS = [CNN, CNN2, CNN3, CNN4, CNN5, CNN6]
ALL_MODELS = [CNN5]
for Model in ALL_MODELS:
    # Assert all model files exist first
    for epoch_count in range(min_epoch_count, max_epoch_count + 1):
        model_path = build_model_file_name(Model, epoch_count)
        assert os.path.exists(model_path), f"File does not exist: {model_path}"

prediction_infos = []
for test_data_file_name in test_data_file_names:
    test_data = test_data_by_file_name[test_data_file_name]
    for Model in ALL_MODELS:
        for epoch_count in range(min_epoch_count, max_epoch_count + 1):
            model_path = build_model_file_name(Model, epoch_count)

            print(f"Predicting {test_data_file_name} - {Model.__name__} - epoch count: {epoch_count} - {model_path}")
            model = Model(
                spectra_data_point_count=training_data.spectra_data_point_count,
                unique_label_count=training_data.unique_label_count,
                device=device
            )
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()

            executor = ModelExecutor(model=model, device=device, training_data=training_data)
            prediction = executor.predict(test_data)
            prediction_infos.append((prediction, test_data_file_name, Model, epoch_count))

curve_ni_2 = Curve1D(
    x=training_data.x_axis,
    y=training_data.get_mean_data(2),
    std=training_data.get_std_data(2),
    label="Training data Ni 2+",
    color=RgbColour(r=0, g=32, b=255) # Blue
)
curve_ni_3 = Curve1D(
    x=training_data.x_axis,
    y=training_data.get_mean_data(3),
    std=training_data.get_std_data(3),
    label="Training data Ni 3+",
    color=RgbColour(r=250, g=156, b=28) # Orange
)
plot_3ds_by_test_data_file_name = {
    test_data_file_name: [
        Plot3D(
            plots=[
                [
                    Plot1D(
                        curves=[
                            Curve1D(
                                x=training_data.x_axis,
                                y=test_y,
                                label="Test data",
                                color=RgbColour(r=255, g=32, b=0) # Red
                            ),
                            curve_ni_2,
                            curve_ni_3
                        ],
                        title=test_data_file_name
                    )
                    for test_y in row
                ]
                for row in test_data_by_file_name[test_data_file_name].spectra_3d
            ]
        )
    ] for test_data_file_name in test_data_file_names
}

PlotViewer3D(
    title="CNN Predictions",
    page_data_list=[
        PlotViewerPageData(
            title=f"Training data: {training_data_file_name} - {test_data_file_name} - {Model.__name__} - Epoch {epoch}",
            plot_heatmaps=[
                    PlotHeatmap(
                        data=prediction.predictions,
                        title="Predictions",
                        colourbar_title="Label"
                    ),
                    PlotHeatmap(
                        data=prediction.probabilities_per_label[2],
                        title="Probabilities for 2+",
                        colourbar_title="Probability",
                        colourbar_min_override=-0.1,
                        colourbar_max_override=1.1
                    ),
                    PlotHeatmap(
                        data=prediction.probabilities_per_label[3],
                        # title="Probabilities for 3+",
                        colourbar_title="Probability",
                        colourbar_min_override=-0.1,
                        colourbar_max_override=1.1
                    ),
                    # PlotHeatmap(
                    #     data=l3_l2_ratio_matrix,
                    #     title="L3/L2 Ratio",
                    #     colourbar_title="Ratio"
                    # )
                ],
                plot_3ds=plot_3ds_by_test_data_file_name[test_data_file_name]
        ) for prediction, test_data_file_name, Model, epoch in prediction_infos
    ]
).run()


