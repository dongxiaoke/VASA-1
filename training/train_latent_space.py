import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.encoders.face_encoder import FaceEncoder
from models.decoders.face_decoder import FaceDecoder
from utils.model_utils import save_model, load_model
from utils.data_utils import PreprocessedVideoDataset, pad_collate  # Assuming this is implemented to handle video frames
import yaml

# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Loss function
class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        return self.mse_loss(input, target)

def train_latent_space(video_dir, model_save_path, embedding_dim, learning_rate, batch_size, num_epochs, save_interval):
    print("Loading dataset...")
    # Load preprocessed data
    dataset = PreprocessedVideoDataset(video_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    print("Dataset loaded successfully.")

    # Initialize models
    print("Initializing models...")
    face_encoder = FaceEncoder(embedding_dim).to(device)
    face_decoder = FaceDecoder(embedding_dim).to(device)
    print("Models initialized.")

    # Optimizers
    optimizer = optim.Adam(list(face_encoder.parameters()) + list(face_decoder.parameters()), lr=learning_rate)

    # Loss function
    criterion = ReconstructionLoss()

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        face_encoder.train()
        face_decoder.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            print(f"Processing batch {batch_idx+1}/{len(dataloader)}")
            batch = batch.to(device)

            # Forward pass
            identity_features, appearance_features = face_encoder(batch)
            reconstructed_images = face_decoder(appearance_features, identity_features, identity_features)

            # Compute loss
            loss = criterion(reconstructed_images, batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Save model checkpoint
        if (epoch + 1) % save_interval == 0:
            save_model(face_encoder, os.path.join(model_save_path, f'face_encoder_epoch_{epoch+1}.pth'))
            save_model(face_decoder, os.path.join(model_save_path, f'face_decoder_epoch_{epoch+1}.pth'))

    print("Training completed.")
    # Save final models
    save_model(face_encoder, os.path.join(model_save_path, 'face_encoder_final.pth'))
    save_model(face_decoder, os.path.join(model_save_path, 'face_decoder_final.pth'))

# Example usage
if __name__ == "__main__":
    # Load configuration
    with open('training/configs/config_latent.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Extract configuration parameters
    embedding_dim = config['model']['embedding_dim']
    learning_rate = float(config['training']['learning_rate'])  # Convert learning rate to float
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    save_interval = config['training']['save_interval']
    video_dir = config['data']['video_dir']
    model_save_path = config['output']['model_save_path']

    # Start training
    train_latent_space(video_dir, model_save_path, embedding_dim, learning_rate, batch_size, num_epochs, save_interval)
