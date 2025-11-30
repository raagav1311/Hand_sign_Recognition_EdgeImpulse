"""
CNN Model for Sign Language Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SignLanguageCNN(nn.Module):
    """
    Convolutional Neural Network for Sign Language Detection
    """
    
    def __init__(self, num_classes=29):
        super(SignLanguageCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Adaptive pooling to make model input-size agnostic
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Fully connected layers (input flattened after adaptive pool)
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Conv block 5
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        # Adaptive pooling -> fixed spatial size (7x7)
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 512*7*7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

class LightSignLanguageCNN(nn.Module):
    """
    Lighter version of CNN for faster training and inference
    """
    
    def __init__(self, num_classes=29):
        super(LightSignLanguageCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)

        # Adaptive pooling to make model input-size agnostic
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Fully connected layers (flatten after adaptive pool)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Adaptive pooling -> fixed spatial size (7x7)
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 128*7*7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x

def get_model(model_type='standard', num_classes=29):
    """
    Get model by type
    
    Args:
        model_type: 'standard' or 'light'
        num_classes: Number of output classes
    
    Returns:
        model
    """
    if model_type == 'light':
        return LightSignLanguageCNN(num_classes)
    else:
        return SignLanguageCNN(num_classes)

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test models
    print("Testing Standard Model:")
    model = SignLanguageCNN(num_classes=29)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\nTesting Light Model:")
    light_model = LightSignLanguageCNN(num_classes=29)
    print(f"Parameters: {count_parameters(light_model):,}")
    
    output = light_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
