from dataclasses import dataclass
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np


@dataclass(frozen=True)
class TrainingData:
    spectra: np.ndarray
    labels: np.ndarray
    x_axis: np.ndarray

    @property
    def sample_count(self) -> int:
        return self.labels.size
    
    @property
    def spectra_data_point_count(self) -> int:
        return self.spectra[0].size
    
    @property
    def offset(self) -> float:
        return float(self.x_axis[0])
    
    @property
    def scale(self) -> float:
        return float(self.x_axis[1] - self.x_axis[0])
    
    @property
    def unique_label_count(self) -> int:
        return len(set(self.labels))
    
    @property
    def unique_labels(self) -> set:
        return set(self.labels.tolist())
    
    @property
    def classification_to_label_map(self) -> Dict[int, int]:
        return {
            index: label
            for index, label in enumerate(sorted(self.unique_labels))
        }
    
    @property
    def label_to_classification_map(self) -> int:
        return {
            label: index
            for index, label in enumerate(sorted(self.unique_labels))
        }
    
    @property
    def classifications(self) -> np.ndarray:
        mapping = self.label_to_classification_map
        classifications = []
        for label in self.labels:
            classifications.append(mapping[label])

        return np.array(classifications)
    
    def shuffle_samples(self) -> "TrainingData":
        random_indexes = np.random.permutation(self.sample_count)
        return TrainingData(
            spectra=self.spectra[random_indexes, :],
            labels=self.labels[random_indexes],
            x_axis=self.x_axis
        )
    
    def reduce_sample_size(self, count_per_label: int) -> "TrainingData":
        shuffled_data = self.shuffle_samples()
        return TrainingData(
            spectra=shuffled_data.spectra[:count_per_label, :],
            labels=shuffled_data.labels[:count_per_label],
            x_axis=self.x_axis
        )
    
    def change_spectra_resolution(self, target_scale: float) -> "TrainingData":
        max_x = self.x_axis[-1]
        target_x_axis = np.arange(self.offset, max_x + self.scale, target_scale)

        new_spectra = np.zeros((self.sample_count, target_x_axis.size))
        for i in range(self.sample_count):
            new_spectra[i] = np.interp(target_x_axis, self.x_axis, self.spectra[i])

        return TrainingData(
            spectra=new_spectra,
            labels=self.labels,
            x_axis=target_x_axis
        )

    def get_spectra(self, label: int) -> np.ndarray:
        """
        Returns an NxM matrix of spectra where each spectrum corresponds to the given label.
        """
        indexes = np.argwhere(self.labels == label).reshape(-1)
        if indexes.size == 0:
            raise ValueError(f"No spectra with label {label} found.")
        return self.spectra[indexes, :]
    
    def get_spectrum(self, label: int) -> np.ndarray:
        """
        Returns an NxM matrix of spectra where each spectrum corresponds to the given label.
        """
        return self.get_spectra(label=label)[0]
    
    def get_mean_data(self, label: int = None) -> np.ndarray:
        """
        Produces a 1D array representing the mean across all spectra.
        """
        if label is not None:
            indexes = np.argwhere(self.labels == label).reshape(-1)
            if indexes.size == 0:
                raise ValueError(f"No spectra with label {label} found.")
            spectra = self.spectra[indexes]
        else:
            spectra = self.spectra

        reshaped = spectra.reshape(-1, self.spectra_data_point_count)
        mean_spectra = np.mean(reshaped, axis=0)
        return mean_spectra.reshape((self.spectra_data_point_count))
    
    def get_std_data(self, label: int = None) -> np.ndarray:
        """
        Produces a 1D array representing the mean across all spectra.
        """
        if label is not None:
            indexes = np.argwhere(self.labels == label).reshape(-1)
            if indexes.size == 0:
                raise ValueError(f"No spectra with label {label} found.")
            spectra = self.spectra[indexes]
        else:
            spectra = self.spectra

        reshaped = spectra.reshape(-1, self.spectra_data_point_count)
        std_spectra = np.std(reshaped, axis=0)
        return std_spectra.reshape((self.spectra_data_point_count))
    
    def split_train_test(self) -> Tuple["TrainingData", "TrainingData"]:
        X_train, X_test, y_train, y_test = train_test_split(
            self.spectra,
            self.labels,
            test_size=0.2,
            random_state=42
        )
        train_data = TrainingData(
            spectra=X_train,
            labels=y_train,
            x_axis=self.x_axis
        )
        test_data = TrainingData(
            spectra=X_test,
            labels=y_test,
            x_axis=self.x_axis
        )

        return train_data, test_data

    def get_data_loader(self, batch_size: int, device: torch.device) -> DataLoader:
        return DataLoader(
            dataset=TensorDataset(
                torch.tensor(
                    self.spectra.reshape((self.sample_count, 1, self.spectra_data_point_count)),
                    dtype=torch.float32,
                    device=device
                ),
                torch.tensor(self.classifications, dtype=torch.float32, device=device),
            ),
            batch_size=batch_size,
            shuffle=True,
        )