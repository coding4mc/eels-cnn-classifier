import torch.nn as nn
import torch.nn.functional as nnf
import torch
import numpy as np
from lib.model import TrainingData, TestData, PredictionOutput


class ModelExecutor:

    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            training_data: TrainingData
    ):
        self._model = model
        self._device = device
        self._training_data = training_data


    def predict(self, data: TestData) -> PredictionOutput:
        input_tensor = data.as_tensor(device=self._device)
        with torch.no_grad():
            result = self._model(input_tensor)
            probabiltiies = nnf.softmax(result, dim=1)
            prediction_probabilities, predictions = probabiltiies.topk(k=1, dim=1)

        # Convert classifications to labels
        predictions_numpy = predictions.cpu().numpy()
        predicted_labels = np.zeros(predictions_numpy.shape)
        classification_to_label_map = self._training_data.classification_to_label_map
        for i in range(predictions_numpy.size):
            classification = int(predictions_numpy[i])
            predicted_labels[i] = classification_to_label_map[classification]

        # Get probabilities per label
        probabiltiies_numpy = probabiltiies.cpu().numpy()
        probabiltiies_per_label = {}
        for label in self._training_data.unique_labels:
            classification = self._training_data.label_to_classification_map[label]
            probabiltiies_per_label[label] = probabiltiies_numpy[:, classification].reshape((data.height, data.width))

        predicted_labels = predicted_labels.reshape((data.height, data.width))
        probabiltiies_probs_reshape = prediction_probabilities.cpu().numpy().reshape((data.height, data.width))
        return PredictionOutput(
            predictions=predicted_labels,
            prediction_probabilities=probabiltiies_probs_reshape,
            probabilities_per_label=probabiltiies_per_label
        )