import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for processing 1D spectra data.

    Attributes:
        conv1 (nn.Conv1d): The first convolutional layer.
        conv2 (nn.Conv1d): The second convolutional layer.
        conv3 (nn.Conv1d): The third convolutional layer.
        pool (nn.MaxPool1d): The max pooling layer.
        fc1 (nn.Linear): The first fully connected layer.
        dropout (nn.Dropout): The dropout layer.
        fc2 (nn.Linear): The second fully connected layer.
    """

    def __init__(self, spectra_data_point_count: int, unique_label_count: int, device: torch.device):
        """
        Initialize the CNN with a ModelDataset.

        Args:
            dataset (ModelDataset): The dataset used to determine the architecture of the CNN.
        """
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128 * (spectra_data_point_count // 8), 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, unique_label_count)
        self.dropout = nn.Dropout(0.5)

        self.to(device=device)

    def forward(self, x):
        """
        Define the forward pass of the CNN.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the CNN.
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x