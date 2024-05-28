import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.encoders.face_encoder import FaceEncoder
from models.decoders.face_decoder import FaceDecoder
from utils.model_utils import save_model, load_model
from utils.data_utils import PreprocessedVideoDataset, load_frame, frame_collate
import torch.nn.functional as F
import yaml

# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load configuration
config_path = 'training/configs/config_latent.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Extract configuration parameters
embedding_dim = int(config['model']['embedding_dim'])
learning_rate = float(config['training']['learning_rate'])
batch_size = int(config['training']['batch_size'])
num_epochs = int(config['training']['num_epochs'])
save_interval = int(config['training']['save_interval'])
video_dir = config['data']['video_dir']
model_save_path = config['output']['model_save_path']
log_dir = config['output']['log_dir']

# Create directories if they do not exist
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Load preprocessed data
dataset = PreprocessedVideoDataset(video_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=frame_collate)
print("Dataset loaded successfully.")

# Initialize models
face_encoder = FaceEncoder(embedding_dim).to(device)
face_decoder = FaceDecoder(embedding_dim).to(device)
print("Models initialized.")

# Optimizer
optimizer = optim.Adam(list(face_encoder.parameters()) + list(face_decoder.parameters()), lr=learning_rate)

# Loss function
criterion = nn.MSELoss()

# Create directories if they do not exist
os.makedirs("output/models/latent", exist_ok=True)

# Training loop
print("Starting training...")
total_loss = 0.0
start_time = time.time()

for epoch in range(num_epochs):
    face_encoder.train()
    face_decoder.train()

    epoch_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        frames = batch[0]

        batch_loss = 0.0
        for frame_path in frames:
            frame = load_frame(frame_path).unsqueeze(0).to(device)

            # Forward pass
            identity_features, appearance_features = face_encoder(frame)
            reconstructed_images = face_decoder(
                appearance_features.unsqueeze(2).unsqueeze(3), 
                identity_features.unsqueeze(2).unsqueeze(3), 
                identity_features.unsqueeze(2).unsqueeze(3)
            )

            # Resize the target frame to match the reconstructed image dimensions
            resized_frame = F.interpolate(frame, size=(512, 512))

            # Compute loss
            loss = criterion(reconstructed_images, resized_frame)
            batch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += batch_loss
        epoch_loss += batch_loss
        print(f"Batch {batch_idx+1}/{len(dataloader)} processed with loss {batch_loss}")

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

    # Save model checkpoint
    if (epoch + 1) % save_interval == 0:
        save_model(face_encoder, os.path.join(model_save_path, f'face_encoder_epoch_{epoch+1}.pth'))
        save_model(face_decoder, os.path.join(model_save_path, f'face_decoder_epoch_{epoch+1}.pth'))

    # Save loss to log file
    with open(os.path.join(log_dir, 'training_log.txt'), 'a') as log_file:
        log_file.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}\n')

total_time = time.time() - start_time
print(f"Training completed in {total_time:.2f} seconds.")
print(f'Final Average Loss: {total_loss / (num_epochs * len(dataloader)):.4f}')

# Save final models
save_model(face_encoder, os.path.join(model_save_path, 'face_encoder_final.pth'))
save_model(face_decoder, os.path.join(model_save_path, 'face_decoder_final.pth'))

print("Final models saved.")
