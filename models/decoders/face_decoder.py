import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceDecoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(FaceDecoder, self).__init__()
        self.embedding_dim = embedding_dim

        # 1x1 Conv layer to reduce channels after concatenation
        self.reduce_channels = nn.Conv2d(embedding_dim * 3, embedding_dim, kernel_size=1, stride=1, padding=0)
        
        # Decoder layers: Transpose Convolutional layers for upsampling
        self.conv1 = nn.ConvTranspose2d(embedding_dim, 512, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        
        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
    
    def forward(self, appearance_features, identity_features, motion_latent_codes):
        # Concatenate appearance, identity features, and motion latent codes along the channel dimension
        x = torch.cat([appearance_features, identity_features, motion_latent_codes], dim=1)  # Shape: (batch_size, 3*embedding_dim, H, W)
        
        # Reduce the number of channels
        x = self.reduce_channels(x)  # Shape: (batch_size, embedding_dim, H, W)
        
        # Decode to reconstruct the image
        x = F.relu(self.bn1(self.conv1(x)))  # Shape: (batch_size, 512, 16, 16)
        x = F.relu(self.bn2(self.conv2(x)))  # Shape: (batch_size, 256, 32, 32)
        x = F.relu(self.bn3(self.conv3(x)))  # Shape: (batch_size, 128, 64, 64)
        x = F.relu(self.bn4(self.conv4(x)))  # Shape: (batch_size, 64, 128, 128)
        x = torch.tanh(self.conv5(x))  # Shape: (batch_size, 3, 256, 256)
        
        return x

# Testing the decoder
if __name__ == "__main__":
    # Create dummy input tensors
    batch_size = 1
    H, W = 8, 8  # Height and Width of the latent feature maps
    appearance_features = torch.randn(batch_size, 512, H, W)
    identity_features = torch.randn(batch_size, 512, H, W)
    motion_latent_codes = torch.randn(batch_size, 512, H, W)
    
    # Instantiate the model
    model = FaceDecoder(embedding_dim=512)
    
    # Perform a forward pass
    reconstructed_image = model(appearance_features, identity_features, motion_latent_codes)
    
    print(f"Reconstructed Image: {reconstructed_image.shape}")
