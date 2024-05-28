import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceDecoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(FaceDecoder, self).__init__()
        self.embedding_dim = embedding_dim

        # Decoder layers: Transpose Convolutional layers for upsampling
        self.conv1 = nn.ConvTranspose2d(embedding_dim * 3, 512, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.conv7 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.conv8 = nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1)
        self.conv9 = nn.ConvTranspose2d(4, 3, kernel_size=4, stride=2, padding=1)
        
        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(16)
        self.bn7 = nn.BatchNorm2d(8)
        self.bn8 = nn.BatchNorm2d(4)
    
    def forward(self, appearance_features, identity_features, motion_latent_codes):
        # Concatenate appearance, identity features, and motion latent codes along the channel dimension
        x = torch.cat([appearance_features, identity_features, motion_latent_codes], dim=1)  # Shape: (batch_size, 3*embedding_dim)
        
        # Reshape to 4D tensor for Conv2d: (batch_size, 3*embedding_dim, 1, 1)
        x = x.view(x.size(0), -1, 1, 1)
        
        # Decode to reconstruct the image
        x = F.relu(self.bn1(self.conv1(x)))  # Shape: (batch_size, 512, 2, 2)
        print(f"Shape after conv1: {x.shape}")
        x = F.relu(self.bn2(self.conv2(x)))  # Shape: (batch_size, 256, 4, 4)
        print(f"Shape after conv2: {x.shape}")
        x = F.relu(self.bn3(self.conv3(x)))  # Shape: (batch_size, 128, 8, 8)
        print(f"Shape after conv3: {x.shape}")
        x = F.relu(self.bn4(self.conv4(x)))  # Shape: (batch_size, 64, 16, 16)
        print(f"Shape after conv4: {x.shape}")
        x = F.relu(self.bn5(self.conv5(x)))  # Shape: (batch_size, 32, 32, 32)
        print(f"Shape after conv5: {x.shape}")
        x = F.relu(self.bn6(self.conv6(x)))  # Shape: (batch_size, 16, 64, 64)
        print(f"Shape after conv6: {x.shape}")
        x = F.relu(self.bn7(self.conv7(x)))  # Shape: (batch_size, 8, 128, 128)
        print(f"Shape after conv7: {x.shape}")
        x = F.relu(self.bn8(self.conv8(x)))  # Shape: (batch_size, 4, 256, 256)
        print(f"Shape after conv8: {x.shape}")
        x = torch.tanh(self.conv9(x))  # Shape: (batch_size, 3, 512, 512)
        print(f"Shape after conv9: {x.shape}")
        
        return x

# Testing the decoder
if __name__ == "__main__":
    # Create dummy input tensors
    batch_size = 1
    embedding_dim = 512
    appearance_features = torch.randn(batch_size, embedding_dim)
    identity_features = torch.randn(batch_size, embedding_dim)
    motion_latent_codes = torch.randn(batch_size, embedding_dim)
    
    # Instantiate the model
    model = FaceDecoder(embedding_dim=embedding_dim)
    
    # Perform a forward pass
    reconstructed_image = model(appearance_features, identity_features, motion_latent_codes)
    
    print(f"Reconstructed Image: {reconstructed_image.shape}")
