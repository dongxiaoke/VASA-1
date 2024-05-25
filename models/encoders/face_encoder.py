import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

class FaceEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(FaceEncoder, self).__init__()
        self.embedding_dim = embedding_dim

        # Using a pretrained ResNet model for feature extraction
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        layers = list(resnet.children())[:-2]  # Remove the last fully connected layer and avgpool
        self.feature_extractor = nn.Sequential(*layers)
        
        # Additional convolutional layers to reduce the feature map size
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, embedding_dim, kernel_size=3, stride=1, padding=1)
        
        # Identity and appearance encoding layers
        self.fc_identity = nn.Linear(embedding_dim * 8 * 8, embedding_dim)
        self.fc_appearance = nn.Linear(embedding_dim * 8 * 8, embedding_dim)

    def forward(self, x):
        # Extract features using the pretrained ResNet
        x = self.feature_extractor(x)  # Shape: (batch_size, 2048, 8, 8)
        
        # Further reduce the feature map size using additional conv layers
        x = F.relu(self.conv1(x))  # Shape: (batch_size, 1024, 8, 8)
        x = F.relu(self.conv2(x))  # Shape: (batch_size, 512, 8, 8)
        x = F.relu(self.conv3(x))  # Shape: (batch_size, embedding_dim, 8, 8)
        
        # Flatten the feature maps
        x_flat = x.view(x.size(0), -1)  # Shape: (batch_size, embedding_dim * 8 * 8)
        
        # Separate identity and appearance features
        identity_features = self.fc_identity(x_flat)  # Shape: (batch_size, embedding_dim)
        appearance_features = self.fc_appearance(x_flat)  # Shape: (batch_size, embedding_dim)
        
        return identity_features, appearance_features

# Testing the encoder
if __name__ == "__main__":
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Instantiate the model
    model = FaceEncoder()
    
    # Perform a forward pass
    identity_features, appearance_features = model(dummy_input)
    
    print(f"Identity Features: {identity_features.shape}")
    print(f"Appearance Features: {appearance_features.shape}")
