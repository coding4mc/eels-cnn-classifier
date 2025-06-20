from dataclasses import dataclass
import torch
import numpy as np


@dataclass(frozen=True)
class TestData:
    spectra: np.ndarray
    height: int
    width: int

    @property
    def sample_count(self) -> int:
        return self.spectra.shape[0]
    
    @property
    def spectra_data_point_count(self) -> int:
        return self.spectra[0].size
    
    @property
    def spectra_3d(self) -> np.ndarray:
        return self.spectra.reshape((self.height, self.width, self.spectra_data_point_count))
    
    def as_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            self.spectra.reshape((self.sample_count, 1, self.spectra_data_point_count)),
            dtype=torch.float32,
            device=device
        )
    