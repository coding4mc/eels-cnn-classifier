import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN4(nn.Module):
    def __init__(self, spectra_data_point_count: int, unique_label_count: int, device: torch.device):
        super().__init__()
        
        # 1D convolutional layers with increasing filter sizes to capture features at different scales
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Dilated convolutions to increase receptive field without losing resolution
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn4 = nn.BatchNorm1d(128)
        
        # Max pooling to reduce dimensionality
        self.pool = nn.MaxPool1d(2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size after convolutions and pooling
        # After 2 pooling layers with stride 2, the size is reduced by factor of 4
        feature_size = spectra_data_point_count // 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * feature_size, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.fc_bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, unique_label_count)
        
        # Data augmentation parameters
        self.shift_range = 5  # Maximum number of points to shift during training

        self.to(device=device)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x