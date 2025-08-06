import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor3D(nn.Module):
    def __init__(self, in_channels=1, feature_dim=64):
        super(FeatureExtractor3D, self).__init__()
        # Convolutional layers with increased capacity
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(256, feature_dim, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool3d(2, 2)

        # Batch normalization
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm3d(feature_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout3d(0.2)

    def forward(self, x):
        # Ensure input is on the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)

        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Conv block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Conv block 5
        x = F.relu(self.bn5(self.conv5(x)))

        # Global average pooling
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        return x


if __name__ == "__main__":
    from config import *
    from utils.gpu_utils import get_device

    # Test feature extractor
    print("Testing 3D Feature Extractor...")
    # Get device
    device = get_device()
    # Create a dummy input (batch_size=1, channels=1, depth=128, height=128, width=128)
    x = torch.randn(1, 1, 128, 128, 128).to(device)
    # Create feature extractor
    feature_extractor = FeatureExtractor3D(in_channels=1, feature_dim=64).to(device)
    # Test forward pass
    features = feature_extractor(x)
    print(f"Input shape: {x.shape}")
    print(f"Output features shape: {features.shape}")
    print("3D Feature Extractor test completed successfully!")