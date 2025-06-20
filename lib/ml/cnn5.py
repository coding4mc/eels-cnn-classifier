import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN5(nn.Module):

    def __init__(self, spectra_data_point_count: int, unique_label_count: int, device: torch.device):
        super(CNN5, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fourth convolutional block with dilated convolutions for capturing wider context
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size of flattened features
        self._to_linear = self._get_conv_output_size(spectra_data_point_count)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, unique_label_count)
        
        # Global average pooling branch for additional feature extraction
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc_gap = nn.Linear(256, 64)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.to(device=device)

        
    def _get_conv_output_size(self, shape):
        # Helper function to calculate the flattened size
        batch_size = 1
        input = torch.zeros(batch_size, 1, shape)
        
        x = F.relu(self.bn1(self.conv1(input)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        return x.numel() // batch_size
        
    def forward(self, x):
        # x shape: [batch_size, 1, sequence_length]
        
        # Apply convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Apply attention mechanism
        attention_weights = self.attention(x)
        x_attention = x * attention_weights
        
        # Global average pooling branch
        gap_features = self.gap(x).view(x.size(0), -1)
        gap_features = F.relu(self.fc_gap(gap_features))
        
        # Main branch
        x = self.pool4(x_attention)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Combine features from both branches
        x = torch.cat([x, gap_features], dim=1)
        
        x = self.fc3(x[:, :128])  # Use only the main branch features for final classification
        
        return x