import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN6(nn.Module):
        
    def __init__(self, spectra_data_point_count: int, unique_label_count: int, device: torch.device):
        super().__init__()
        
        # 1. Average Pooling for initial noise reduction
        self.initial_pooling = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        
        # 2. Larger kernel sizes in first layers to capture broader patterns
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),  # Larger kernel
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(16),
            nn.AvgPool1d(2)  # Average pooling instead of max pooling
        )
        
        # 3. Residual block for better feature preservation
        self.res_block1 = ResidualBlock(16, 16)
        
        # 4. Second convolutional block with moderate kernel size
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(2)
        )
        
        self.res_block2 = ResidualBlock(32, 32)
        
        # 5. Final convolutional block focusing on fine details
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.AvgPool1d(2)
        )
        
        # Calculate the size of flattened features
        self.flatten_size = 64 * (spectra_data_point_count // 8)
        
        # 6. Attention mechanism
        self.attention = SEBlock(64)
        
        # 7. Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, unique_label_count)
        )

        self.to(device=device)
        
    def forward(self, x):
        # Initial noise reduction
        x = self.initial_pooling(x)
        
        # Convolutional blocks with residual connections
        x = self.conv1(x)
        x = self.res_block1(x)
        
        x = self.conv2(x)
        x = self.res_block2(x)
        
        x = self.conv3(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Residual Block for better feature preservation
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.leaky_relu(x)
        return x

# Squeeze-and-Excitation Block for channel attention
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)