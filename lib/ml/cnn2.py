import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN2(nn.Module):
    def __init__(self, spectra_data_point_count: int, unique_label_count: int, device: torch.device):
        super(CNN2, self).__init__()
        
        self.shift_range = 10
        
        # Larger kernels with dilation
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3, dilation=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.avg_pool = nn.AvgPool1d(kernel_size=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, unique_label_count)
        self.dropout = nn.Dropout(0.5)
        
        self.to(device=device)

    def forward(self, x):            
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.avg_pool(x)
        x = F.relu(self.conv3(x))
        x = self.global_avg_pool(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x