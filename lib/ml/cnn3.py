import torch
import torch.nn as nn
import torch.nn.functional as F


class STN1D(nn.Module):
    def __init__(self, data_point_count: int):
        super(STN1D, self).__init__()

        self._data_point_count = data_point_count
        self.loc_conv1 = nn.Conv1d(1, 32, kernel_size=7)
        self.loc_conv2 = nn.Conv1d(32, 32, kernel_size=5)

        # Calculate the size after convolutions and pooling
        size_after_conv1 = data_point_count - 6  # kernel_size=7
        size_after_pool1 = size_after_conv1 // 2  # max_pool size=2
        size_after_conv2 = size_after_pool1 - 4  # kernel_size=5
        self.final_size = size_after_conv2 // 2  # max_pool size=2

        self.loc_fc1 = nn.Linear(32 * self.final_size, 32)
        self.loc_fc2 = nn.Linear(32, 1)  # Changed from 2 to 1 to output a single shift value
        
        # Initialize the weights/bias with identity transformation
        self.loc_fc2.weight.data.zero_()
        self.loc_fc2.bias.data.copy_(torch.tensor([0], dtype=torch.float))  # Initialize with no shift

    def forward(self, x):
        batch_size = x.size(0)
        
        # Localization network
        xs = F.relu(self.loc_conv1(x))
        xs = F.max_pool1d(xs, 2)
        xs = F.relu(self.loc_conv2(xs))
        xs = F.max_pool1d(xs, 2)
        xs = xs.view(-1, 32 * self.final_size)
        xs = F.relu(self.loc_fc1(xs))
        translation = self.loc_fc2(xs)  # [batch_size, 1]
        
        # Convert translation to integer shifts
        shifts = torch.round(translation * self._data_point_count / 4).long()  # Scale factor for reasonable shifts
        
        # Apply shifts using roll
        shifted_x = torch.zeros_like(x)
        for i in range(batch_size):
            shift = shifts[i].item()
            shifted_x[i] = torch.roll(x[i], shift, dims=-1)
            
        return shifted_x


class CNN3(nn.Module):
    def __init__(self, spectra_data_point_count: int, unique_label_count: int, device: torch.device):
        super().__init__()
        
        self.stn = STN1D(spectra_data_point_count)
        
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
        # Apply spatial transformer
        x = self.stn(x)
            
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
